use crate::utils::{
    block_scale::{self, BlockScaledFormat},
    parity::parity_from_idx,
    view::{View1D, View2D, ViewPosWithinTensor2D},
};
use nvvm_builder::{
    AddrSpace, DynVector, Global, IntoVal, Mbarrier, Ptr, Scalar, Shared, SharedCarve, Val,
    ValueType, Var, Vector, bf16, cluster_block_rank,
    cp_async::{self, CopyOpts},
    fence_sync_restrict_acquire, fence_sync_restrict_release, if_, if_else, lane_id, mbarrier,
    shfl_sync_bfly, tma, vote_any, with_leader_of_warp,
};

#[derive(Debug, Copy, Clone)]
pub struct ChunkedThreadLayout1D {
    chunk_size: usize,
    chunks_per_thread: usize,
    threads: usize,
}

impl ChunkedThreadLayout1D {
    pub const MAX_CHUNK_SIZE: usize = 8;

    pub fn new(elements: usize, threads: usize) -> Self {
        assert!(elements.is_multiple_of(threads));
        let elements_per_thread = elements / threads;

        let chunk_size = (0..=elements_per_thread.ilog2().min(Self::MAX_CHUNK_SIZE as u32))
            .rev()
            .find(|x| elements_per_thread.is_multiple_of(*x as usize))
            .unwrap() as usize;

        Self {
            chunk_size,
            chunks_per_thread: elements_per_thread / chunk_size,
            threads,
        }
    }

    /// Loads all chunks owned by the lanes in active control flow,
    /// from the given source pointer (SMEM or GMEM). Does not introduce
    /// bounds checks.
    pub fn load_aligned_unchecked<T: ValueType + Scalar, S: AddrSpace>(
        &self,
        base_ptr: Val<Ptr<T, S>>,
        thread_idx: Val<u32>,
    ) -> Val<Vector<T>> {
        let mut concat = None;
        for chunk in 0..self.chunks_per_thread {
            let offset_ptr = base_ptr.offset(
                thread_idx * self.chunk_size as u32
                    + (chunk * self.threads * self.chunk_size) as u32,
            );
            let chunk_data = offset_ptr.load_vector(self.chunk_size, self.chunk_size * T::SIZE);
            concat = match concat {
                None => Some(chunk_data),
                Some(c) => Some(c.concat(chunk_data)),
            };
        }
        concat.unwrap()
    }

    /// Performs an asynchronous copy operation of this row of data from GMEM to SMEM,
    /// using TMA bulk (non-tensor) copies. If the tail is not aligned to 16 bytes,
    /// a LDGSTS (cp.async) is issued to handle it. This should be called by one thread only.
    /// Returns # of bytes copied; an arrive_expect_tx operation on the barrier with appropriate byte count
    /// should follow this call.
    ///
    /// The length of `dst` (SMEM buffer) must be the layout's element count; this is currently
    /// not checked.
    pub fn async_bulk_copy_g2s_aligned<T: ValueType + Scalar>(
        &self,
        src: View1D<T, Global>,
        dst: View1D<T, Shared>,
        bar: Mbarrier,
    ) -> Val<u32> {
        let aligned_bytes_to_copy = src.len * T::SIZE as u32 / 16 * 16;
        tma::bulk_load(
            dst.base,
            src.base,
            aligned_bytes_to_copy / T::SIZE as u32,
            bar,
        );

        let tail_bytes = src.len * T::SIZE as u32 - aligned_bytes_to_copy;
        if_(tail_bytes.gt(0), || {
            cp_async::copy_with(
                dst.base.offset(aligned_bytes_to_copy / T::SIZE as u32),
                src.base.offset(aligned_bytes_to_copy / T::SIZE as u32),
                16,
                CopyOpts::new().src_bytes(tail_bytes),
            );
            cp_async::mbarrier_arrive(bar);
        });

        // tail does not contribute to expect-tx count; only count bulk-copy bytes
        aligned_bytes_to_copy
    }

    pub fn num_elements(&self) -> usize {
        self.chunk_size * self.chunks_per_thread * self.threads
    }

    pub fn store_aligned_unchecked<T: ValueType + Scalar, S: AddrSpace>(
        &self,
        base_ptr: Val<Ptr<T, S>>,
        vals: Val<Vector<T>>,
        thread_idx: Val<u32>,
    ) {
        for chunk in 0..self.chunks_per_thread {
            let vals = vals.slice(chunk * self.chunk_size..(chunk + 1) * self.chunk_size);
            let offset_ptr = base_ptr.offset(
                thread_idx * self.chunk_size as u32
                    + (chunk * self.threads * self.chunk_size) as u32,
            );
            offset_ptr.store_vector(vals, self.chunk_size * T::SIZE);
        }
    }

    /// Stores values with bounds checks. Stores fall back to elementwise
    /// stores if any lane in the warp is partially out of bounds.
    /// View must still be aligned (this is not checked at runtime).
    pub fn store_aligned<T: ValueType + Scalar, S: AddrSpace>(
        &self,
        view: View1D<T, S>,
        vals: Val<Vector<T>>,
        thread_idx: Val<u32>,
    ) {
        for chunk in 0..self.chunks_per_thread {
            let vals = vals.slice(chunk * self.chunk_size..(chunk + 1) * self.chunk_size);

            let offset = thread_idx * self.chunk_size as u32
                + (chunk * self.threads * self.chunk_size) as u32;

            let entire_chunk_in_bounds = (offset + self.chunk_size as u32).le(view.len);
            let base_element_in_bounds = offset.le(view.len);
            let is_partial = !entire_chunk_in_bounds & base_element_in_bounds;

            let needs_elementwise_store = vote_any(u32::MAX, is_partial);

            if_else(
                needs_elementwise_store,
                || {
                    for i in 0..self.chunk_size {
                        let offset = offset + i as u32;
                        let in_bounds = offset.le(view.len);
                        if_(in_bounds, || {
                            let offset_ptr = view.base.offset(offset);
                            offset_ptr.store(vals.get(i));
                        });
                    }
                },
                || {
                    if_(entire_chunk_in_bounds, || {
                        let offset_ptr = view.base.offset(offset);
                        offset_ptr.store_vector(vals, self.chunk_size * T::SIZE);
                    });
                },
            );
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct ChunkedThreadLayout2D {
    inner: ChunkedThreadLayout1D,
    num_rows: usize,
}

impl ChunkedThreadLayout2D {
    pub fn new(elements_per_row: usize, rows: usize, threads: usize) -> Self {
        Self {
            inner: ChunkedThreadLayout1D::new(elements_per_row, threads),
            num_rows: rows,
        }
    }

    pub fn num_elements_per_row(&self) -> usize {
        self.inner.num_elements()
    }

    pub fn num_elements_total(&self) -> usize {
        self.num_elements_per_row() * self.num_rows
    }

    pub fn load_aligned_unchecked<T: ValueType + Scalar, S: AddrSpace>(
        &self,
        view: View2D<T, S>,
        thread_idx: Val<u32>,
    ) -> Vec<Val<Vector<T>>> {
        let mut rows = Vec::new();
        for row_idx in 0..self.num_rows {
            let offset_ptr = view.base.offset(row_idx as u32 * view.outer_stride);
            rows.push(self.inner.load_aligned_unchecked(offset_ptr, thread_idx));
        }
        rows
    }

    /// Performs an asynchronous copy operation of this tile of data from GMEM to SMEM,
    /// using TMA bulk (non-tensor) copies. If the tail of each row is not aligned to 16 bytes,
    /// LDGSTS (cp.async) operations are issued to handle it. This function should be called by one thread only.
    /// Returns # of bytes copied; an arrive_expect_tx operation on the barrier with appropriate byte count
    /// should follow this call.
    ///
    /// The sizes of `dst` (SMEM buffer) must match the layout's sizes; this is currently
    /// not checked.
    pub fn async_bulk_copy_g2s_aligned<T: ValueType + Scalar>(
        &self,
        src: View2D<T, Global>,
        dst: View2D<T, Shared>,
        bar: Mbarrier,
    ) -> Val<u32> {
        let bytes_expect_tx = Var::<u32>::new(0);
        for row_idx in 0..self.num_rows {
            if_(Val::<u32>::new(row_idx as u32).lt(src.outer_len), || {
                let row_bytes = self.inner.async_bulk_copy_g2s_aligned(
                    View1D {
                        base: src.base.offset(row_idx as u32 * src.outer_stride),
                        len: src.inner_len,
                    },
                    View1D {
                        base: dst.base.offset(row_idx as u32 * dst.outer_stride),
                        len: dst.inner_len,
                    },
                    bar,
                );
                bytes_expect_tx.set(bytes_expect_tx.get() + row_bytes);
            });
        }
        bytes_expect_tx.into_val()
    }

    pub fn store_aligned_unchecked<T: ValueType + Scalar, S: AddrSpace>(
        &self,
        view: View2D<T, S>,
        vals: &[Val<Vector<T>>],
        thread_idx: Val<u32>,
    ) {
        for (row_idx, vals) in vals.iter().enumerate() {
            let offset_ptr = view.base.offset(row_idx as u32 * view.outer_stride);
            self.inner
                .store_aligned_unchecked(offset_ptr, *vals, thread_idx);
        }
    }

    pub fn store_aligned<T: ValueType + Scalar, S: AddrSpace>(
        &self,
        view: View2D<T, S>,
        vals: &[Val<Vector<T>>],
        thread_idx: Val<u32>,
    ) {
        for (row_idx, vals) in vals.iter().enumerate() {
            let is_in_bounds = view.outer_len.gt(row_idx as u32);
            if_(is_in_bounds, || {
                let offset_ptr = view.base.offset(row_idx as u32 * view.outer_stride);
                self.inner.store_aligned(
                    View1D {
                        base: offset_ptr,
                        len: view.inner_len,
                    },
                    *vals,
                    thread_idx,
                );
            });
        }
    }

    /// Performs rowwise block-scaled quantization using warp collectives,
    /// then stores quantized values and scale factors to GMEM, with bounds checking.
    /// Requires num_elements >= block_size and block_size a power of two, at most 32.
    pub fn store_aligned_rowwise_block_scaled<Q: ValueType + Scalar, SF: ValueType + Scalar>(
        &self,
        format: BlockScaledFormat,
        quantized_vals_view: View2D<Q, Global>,
        quantized_vals_pos: ViewPosWithinTensor2D,
        scale_factors_base: Val<Ptr<SF, Global>>,
        vals: &[Val<Vector<bf16>>],
        thread_idx: Val<u32>,
    ) {
        let block_size = format.block_size();
        assert!(block_size.is_power_of_two());
        assert!(block_size <= 32);
        assert!(
            self.inner.threads * self.inner.chunk_size * self.inner.chunks_per_thread >= block_size
        );

        for (row_idx, vals) in vals.iter().enumerate() {
            let is_in_bounds = quantized_vals_view.outer_len.gt(row_idx as u32);
            if_(is_in_bounds, || {
                let mut amaxs = Val::<Vector<bf16>>::splat(self.inner.chunks_per_thread, 0.0f32);

                for chunk_idx in 0..self.inner.chunks_per_thread {
                    let chunk_vals = vals.slice(
                        chunk_idx * self.inner.chunk_size..(chunk_idx + 1) * self.inner.chunk_size,
                    );
                    let chunk_amax = chunk_vals.abs().reduce_max();
                    amaxs = amaxs.set(chunk_idx, chunk_amax);
                }

                let amax_reduction_lanes = block_size / self.inner.chunk_size;
                assert!(amax_reduction_lanes.is_power_of_two());
                assert!(self.inner.chunk_size <= format.block_size());
                for bit in 0..amax_reduction_lanes.ilog2() {
                    amaxs = amaxs.max(shfl_sync_bfly(u32::MAX, amaxs, 1 << bit));
                }

                let scale_factors = format.compute_scale_factors_from_amaxs(amaxs);

                let mut quantized_vals = None;

                for chunk_idx in 0..self.inner.chunks_per_thread {
                    let chunk_sf = scale_factors.get(chunk_idx);
                    let chunk_vals = vals.slice(
                        chunk_idx * self.inner.chunk_size..(chunk_idx + 1) * self.inner.chunk_size,
                    );
                    let chunk_vals_quantized = format.quantize(
                        chunk_vals,
                        DynVector::splat(self.inner.chunk_size, &chunk_sf),
                    );
                    quantized_vals = match quantized_vals {
                        None => Some(chunk_vals_quantized),
                        Some(v) => Some(v.concat(&chunk_vals_quantized)),
                    };

                    let col = thread_idx * self.inner.chunk_size as u32
                        + chunk_idx as u32 * (self.inner.chunk_size * self.inner.threads) as u32;
                    let sf_col = col / format.block_size() as u32;

                    let n_sf_cols = quantized_vals_pos
                        .total_tensor_cols
                        .div_ceil(format.block_size() as u32);
                    let sf_in_bounds = sf_col.lt(n_sf_cols);
                    let should_store_sf =
                        (lane_id() & !((1 << amax_reduction_lanes as u32) - 1)).eq(0);
                    if_(sf_in_bounds & should_store_sf, || {
                        let index = block_scale::compute_scale_factor_index(
                            format,
                            row_idx as u32 + quantized_vals_pos.row_offset,
                            (col + quantized_vals_pos.col_offset) / format.block_size() as u32,
                            n_sf_cols,
                        );
                        scale_factors_base
                            .offset(index)
                            .store(chunk_sf.downcast().unwrap());
                    });
                }

                self.inner.store_aligned(
                    View1D {
                        base: quantized_vals_view
                            .base
                            .offset(quantized_vals_view.outer_stride * row_idx as u32),
                        len: quantized_vals_view.outer_len,
                    },
                    quantized_vals.unwrap().downcast::<Q>().unwrap(),
                    thread_idx,
                );
            });
        }
    }

    /// Column-major block-scaled store. Thread layout's num_rows must be a multiple of, and at least,
    /// the block size. Note that `quantized_vals_view` is interpreted as column-major.
    pub fn store_aligned_colwise_block_scaled<Q: ValueType + Scalar, SF: ValueType + Scalar>(
        &self,
        format: BlockScaledFormat,
        quantized_vals_view: View2D<Q, Global>,
        quantized_vals_pos: ViewPosWithinTensor2D,
        scale_factors_base: Val<Ptr<SF, Global>>,
        vals: &[Val<Vector<bf16>>],
        thread_idx: Val<u32>,
    ) {
        assert!(
            self.num_rows >= format.block_size()
                && self.num_rows.is_multiple_of(format.block_size())
        );

        for block in 0..self.num_rows / format.block_size() {
            for chunk_idx in 0..self.inner.chunks_per_thread {
                let mut chunk_amaxs = None;

                for row in &vals[block * format.block_size()..][..format.block_size()] {
                    chunk_amaxs = match chunk_amaxs {
                        None => Some(row.abs()),
                        Some(x) => Some(x.max(row.abs())),
                    };
                }

                let chunk_amaxs = chunk_amaxs.unwrap();

                let chunk_sfs = format.compute_scale_factors_from_amaxs(chunk_amaxs);

                // Quantize and store elements
                for col_within_chunk in 0..chunk_idx {
                    let col = thread_idx * self.inner.chunk_size as u32
                        + chunk_idx as u32
                            * self.inner.chunk_size as u32
                            * self.inner.chunks_per_thread as u32
                        + col_within_chunk as u32;

                    let col_in_bounds = col.lt(quantized_vals_view.outer_len);

                    if_(col_in_bounds, || {
                        let mut col_vals: Option<Val<Vector<bf16>>> = None;
                        for row in &vals[block * format.block_size()..][..format.block_size()] {
                            col_vals = Some(match col_vals {
                                None => Val::<Vector<bf16>>::splat(1, row.get(col_within_chunk)),
                                Some(v) => v.concat(Val::<Vector<bf16>>::splat(
                                    1,
                                    row.get(col_within_chunk),
                                )),
                            });
                        }

                        let col_vals = col_vals.unwrap();
                        let col_vals_quantized = format.quantize(
                            col_vals,
                            DynVector::splat(format.block_size(), &chunk_sfs.get(col_within_chunk)),
                        );

                        let last_row = block * format.block_size() + format.block_size() - 1;
                        let all_rows_in_bounds = quantized_vals_view.inner_len.gt(last_row as u32);

                        let store_sfs = || {
                            scale_factors_base
                                .offset(block_scale::compute_scale_factor_index(
                                    format,
                                    col + quantized_vals_pos.col_offset,
                                    block as u32 * format.block_size() as u32
                                        + quantized_vals_pos.row_offset,
                                    quantized_vals_pos.total_tensor_cols
                                        / format.block_size() as u32,
                                ))
                                .store(chunk_sfs.get(col_within_chunk));
                        };

                        if_else(
                            all_rows_in_bounds,
                            || {
                                // Vectorized store
                                quantized_vals_view
                                    .base
                                    .offset(
                                        block as u32 * format.block_size() as u32
                                            + col * quantized_vals_view.outer_stride,
                                    )
                                    .store(&col_vals_quantized);
                                store_sfs();
                            },
                            || {
                                let any_row_in_bounds = quantized_vals_view
                                    .inner_len
                                    .gt((block * format.block_size()) as u32);
                                if_else(
                                    any_row_in_bounds,
                                    || {
                                        // Elementwise store
                                        for row_in_block in 0..format.block_size() {
                                            let row = (block * format.block_size()) as u32
                                                + row_in_block as u32;
                                            let row_in_bounds =
                                                quantized_vals_view.inner_len.gt(row);
                                            if_(row_in_bounds, || {
                                                quantized_vals_view
                                                    .base
                                                    .offset(
                                                        block as u32 * format.block_size() as u32
                                                            + row_in_block as u32
                                                            + col
                                                                * quantized_vals_view.outer_stride,
                                                    )
                                                    .store(col_vals_quantized.get(row_in_block));
                                            });
                                        }

                                        store_sfs();
                                    },
                                    || {
                                        // No store - all rows out of bounds
                                    },
                                );
                            },
                        );
                    });
                }
            }
        }
    }
}

/// Holds data (SMEM and mbarriers) needed for resharding.
///
/// A resharding reassigns elements within a CGA-wide 2D tile, such that:
/// - Before the resharding, each CTA holds different _rows_ but the same _columns_;
/// - After the resharding, each CTA holds different _columns_ but the same _rows_.
///
/// In other words, we change the dimension a tensor is sharded across within a CGA.
pub struct Resharder<T: Scalar + ValueType> {
    buffer: Val<Ptr<T, Shared>>,
    /// Arrived on by async stores from peer CTAs.
    produced: Mbarrier,
    /// Arrived on by each peer upon consuming their own SMEM buffers,
    /// signaling that we can do another reshard without a data race.
    consumed: Mbarrier,
    /// Number of reshards performed.
    i: Var<u32>,
    src_layout: ChunkedThreadLayout2D,
    dst_layout: ChunkedThreadLayout2D,
    cluster_size: u32,
    leader_warp: u32,
}

impl<T: Scalar + ValueType> Resharder<T> {
    /// Must be called by every warp in every CTA. A cluster sync
    /// must be completed before the first reshard operations.
    pub fn init(
        cluster_size: u32,
        src_layout: &ChunkedThreadLayout2D,
        smem: &mut SharedCarve,
        leader_warp: u32,
    ) -> Self {
        assert!(
            src_layout
                .num_elements_per_row()
                .is_multiple_of(cluster_size as usize)
        );

        let dst_layout = ChunkedThreadLayout2D::new(
            src_layout.num_elements_per_row() / cluster_size as usize,
            src_layout.num_rows * cluster_size as usize,
            src_layout.inner.threads,
        );

        let needed_elements = src_layout.num_elements_total();
        let buffer = smem.array::<T>(needed_elements, 16);

        let produced = Mbarrier::shared("reshard_produced");
        let consumed = Mbarrier::shared("reshard_consumed");

        with_leader_of_warp(0, || {
            produced.init(1);

            // Initial arming of producer barrier
            produced
                .cluster()
                .relaxed()
                .arrive_expect_tx((needed_elements * T::SIZE) as u32);

            consumed.init(src_layout.inner.threads as u32 * cluster_size);

            mbarrier::fence_init();
        });

        Self {
            buffer,
            produced,
            consumed,
            i: Var::new(0),
            cluster_size,
            src_layout: *src_layout,
            dst_layout,
            leader_warp,
        }
    }

    /// This function initiates resharding by asynchronously sending our data to peer CTAs.
    /// Complete the resharding by calling `finish_resharding` on the returned object.
    /// In between the calls, other work can overlap the cluster transfers, hiding latency.
    #[must_use]
    pub fn initiate_resharding<'a>(
        &'a self,
        vals: &[Val<Vector<T>>],
        pos_in_tensor: ViewPosWithinTensor2D,
        thread_idx: Val<u32>,
    ) -> ReshardingInProgress<'a, T> {
        let needs_wait = self.i.ge(1);
        if_(needs_wait, || {
            self.consumed
                .cluster()
                .relaxed()
                .wait_parity(parity_from_idx(self.i, 1));
        });

        let dst_layout_elements_per_row =
            self.src_layout.num_elements_per_row() as u32 / self.cluster_size;

        for (row_idx, row_vals) in vals.iter().copied().enumerate() {
            for chunk_idx in 0..self.src_layout.inner.chunks_per_thread {
                let col = thread_idx * self.src_layout.inner.chunk_size as u32
                    + chunk_idx as u32
                        * (self.src_layout.inner.chunks_per_thread
                            * self.src_layout.inner.chunk_size) as u32;
                let dst_rank = col / dst_layout_elements_per_row;

                let dst_col = col % dst_layout_elements_per_row;
                let dst_row =
                    cluster_block_rank() * self.dst_layout.num_rows as u32 + row_idx as u32;

                let ptr_local = self
                    .buffer
                    .offset(dst_col + dst_row * self.dst_layout.num_elements_per_row() as u32);
                let ptr_remote = ptr_local.map_rank(dst_rank);
                let bar_remote = self.produced.remote(dst_rank);

                ptr_remote
                    .cast_shaped::<Vector<T>>(self.src_layout.inner.chunk_size)
                    .store_async(
                        row_vals.slice(
                            chunk_idx * self.src_layout.inner.chunk_size
                                ..(chunk_idx + 1) * self.src_layout.inner.chunk_size,
                        ),
                        bar_remote,
                    );
            }
        }

        ReshardingInProgress {
            resharder: self,
            thread_idx,
        }
    }
}

pub struct ReshardingInProgress<'a, T: Scalar + ValueType> {
    resharder: &'a Resharder<T>,
    thread_idx: Val<u32>,
}

impl<'a, T: Scalar + ValueType> ReshardingInProgress<'a, T> {
    /// Waits and receives data from peer CTAs. Returns the thread layout of the resulting
    /// resharded data (which remains row-major, but with a smaller row size and more rows),
    /// as well as the data itself.
    pub fn finish_resharding(&self) -> (ChunkedThreadLayout2D, Vec<Val<Vector<T>>>) {
        self.resharder
            .produced
            .cluster()
            .relaxed()
            .wait_parity(parity_from_idx(self.resharder.i, 1));
        fence_sync_restrict_acquire();

        let vals = self.resharder.dst_layout.load_aligned_unchecked(
            View2D {
                base: self.resharder.buffer,
                inner_len: Val::new(self.resharder.dst_layout.num_elements_per_row() as u32),
                outer_len: Val::new(self.resharder.dst_layout.num_rows as u32),
                outer_stride: Val::new(self.resharder.dst_layout.num_elements_per_row() as u32),
            },
            self.thread_idx,
        );

        // Re-arm our producer barrier to receive the next reshard.
        with_leader_of_warp(self.resharder.leader_warp, || {
            self.resharder
                .produced
                .cluster()
                .relaxed()
                .arrive_expect_tx(
                    (self.resharder.dst_layout.num_elements_total() * T::SIZE) as u32,
                );
        });

        fence_sync_restrict_release();
        for peer in 0..self.resharder.cluster_size {
            self.resharder.consumed.remote(peer).relaxed().arrive();
        }

        self.resharder.i.set(self.resharder.i + 1);

        (self.resharder.dst_layout, vals)
    }
}
