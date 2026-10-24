use crate::utils::view::{View1D, View2D};
use nvvm_builder::{AddrSpace, Ptr, Scalar, Val, ValueType, Vector, if_, if_else, vote_any};

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
                self.inner
                    .store_aligned_unchecked(offset_ptr, *vals, thread_idx);
            });
        }
    }
}
