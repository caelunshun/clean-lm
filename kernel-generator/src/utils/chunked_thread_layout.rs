use crate::utils::{
    block_scale::{self, BlockScaledFormat},
    view::{View1D, View2D, ViewPosWithinTensor2D},
};
use nvvm_builder::{
    AddrSpace, DynVector, Global, Ptr, Scalar, Val, ValueType, Vector, bf16, if_, if_else, lane_id,
    shfl_sync_bfly, vote_any,
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
}
