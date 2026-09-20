use core::{
    hint::assert_unchecked,
    iter::TrustedLen,
    marker::PhantomData,
    mem::transmute,
    ops::{Index, IndexMut, Range, RangeBounds},
    slice,
};

use bytemuck::Pod;
use cuda_device::{Barrier, async_copy, tma};

use crate::lane_id;

/// 128-bit-aligned view into a buffer.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct AlignedView<'a, T> {
    view: &'a [T],
}

impl<'a, T> AlignedView<'a, T>
where
    T: Pod,
{
    #[inline(always)]
    pub fn new(view: &'a [u128]) -> Self {
        Self {
            view: bytemuck::must_cast_slice(view),
        }
    }

    /// # Safety
    /// Pointer must be aligned to 128 bits and be valid for num_elements.
    /// num_elements must be aligned to 128 bits.
    #[inline(always)]
    pub unsafe fn from_ptr(ptr: *const T, num_elements: usize) -> Self {
        unsafe {
            assert_unchecked(ptr.is_aligned_to(align_of::<u128>()));
            Self::new(slice::from_raw_parts(
                ptr.cast(),
                num_elements / elements_per_u128::<T>(),
            ))
        }
    }

    /// Slices in units of _elements_, not 128-bit values.
    ///
    /// Behavior is unspecified (but not UB) if the range is not 128-bit aligned.
    #[inline(always)]
    pub fn slice(&self, range: impl RangeBounds<usize>) -> Self {
        let start = match range.start_bound() {
            core::ops::Bound::Included(x) => *x,
            core::ops::Bound::Excluded(x) => *x + 1,
            core::ops::Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            core::ops::Bound::Included(x) => *x + 1,
            core::ops::Bound::Excluded(x) => *x,
            core::ops::Bound::Unbounded => self.num_elements(),
        };

        assert!(start.is_multiple_of(elements_per_u128::<T>()));
        assert!(end.is_multiple_of(elements_per_u128::<T>()));

        Self {
            view: &self.view[start..end],
        }
    }

    /// Length in _elements_.
    #[inline(always)]
    pub fn num_elements(&self) -> usize {
        self.view.len()
    }

    #[inline(always)]
    pub fn as_u128_slice(&self) -> &[u128] {
        unsafe {
            slice::from_raw_parts(
                self.view.as_ptr().cast(),
                self.view.len() / elements_per_u128::<T>(),
            )
        }
    }

    #[inline(always)]
    pub fn partition_iter<P: PartitionViewIter<T>>(
        &self,
        partition: P,
    ) -> impl Iterator<Item = AlignedView<'a, T>> + ExactSizeIterator + TrustedLen {
        partition
            .partition(self.num_elements())
            .map(move |range| self.slice(range))
    }

    #[inline(always)]
    pub fn num_bytes(&self) -> usize {
        self.num_elements() * size_of::<T>()
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        self.view
    }
}

impl<'a, T> Index<usize> for AlignedView<'a, T> {
    type Output = T;

    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        &self.view[index]
    }
}

impl<'a, T> Index<u32> for AlignedView<'a, T> {
    type Output = T;

    #[inline(always)]
    fn index(&self, index: u32) -> &Self::Output {
        &self.view[index as usize]
    }
}

impl<'a, T> IntoIterator for AlignedView<'a, T>
where
    T: Pod + Copy,
{
    type Item = T;
    type IntoIter = impl Iterator<Item = T> + ExactSizeIterator + TrustedLen;

    fn into_iter(self) -> Self::IntoIter {
        self.view.iter().copied()
    }
}

/// 128-bit-aligned view into a mutable buffer.
#[repr(C)]
pub struct AlignedViewMut<'a, T> {
    ptr: *mut T,
    num_elements: usize,
    _lifetime: PhantomData<&'a mut T>,
}

impl<'a, T> AlignedViewMut<'a, T>
where
    T: Pod,
{
    #[inline(always)]
    pub fn new(view: &'a mut [u128]) -> Self {
        Self {
            ptr: view.as_mut_ptr().cast(),
            num_elements: view.len() * elements_per_u128::<T>(),
            _lifetime: PhantomData,
        }
    }

    /// # Safety
    /// Pointer must be aligned to 128 bits and be valid for num_elements.
    /// num_elements must be aligned to 128 bits.
    #[inline(always)]
    pub unsafe fn from_ptr(ptr: *mut T, num_elements: usize) -> Self {
        unsafe {
            assert_unchecked(ptr.is_aligned_to(align_of::<u128>()));
            Self::new(slice::from_raw_parts_mut(
                ptr.cast(),
                num_elements / elements_per_u128::<T>(),
            ))
        }
    }

    /// Slices in units of _elements_, not 128-bit values.
    ///
    /// Behavior is unspecified (but not UB) if the range is not 128-bit aligned.
    #[inline(always)]
    pub fn slice(&self, range: impl RangeBounds<usize>) -> AlignedView<'_, T> {
        self.as_view().slice(range)
    }

    /// Slices in units of _elements_, not 128-bit values.
    ///
    /// Behavior is unspecified (but not UB) if the range is not 128-bit aligned.
    #[inline(always)]
    pub fn slice_mut(&mut self, range: impl RangeBounds<usize>) -> AlignedViewMut<'_, T> {
        let start = match range.start_bound() {
            core::ops::Bound::Included(x) => *x,
            core::ops::Bound::Excluded(x) => *x + 1,
            core::ops::Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            core::ops::Bound::Included(x) => *x + 1,
            core::ops::Bound::Excluded(x) => *x,
            core::ops::Bound::Unbounded => self.num_elements(),
        };

        assert!(start.is_multiple_of(elements_per_u128::<T>()));
        assert!(end.is_multiple_of(elements_per_u128::<T>()));

        assert!(start <= self.num_elements);
        assert!(end <= self.num_elements);
        assert!(start <= end);

        AlignedViewMut {
            ptr: unsafe { self.ptr.add(start) },
            num_elements: end - start,
            _lifetime: PhantomData,
        }
    }

    #[inline(always)]
    pub fn as_view(&self) -> AlignedView<'_, T> {
        AlignedView {
            view: unsafe { slice::from_raw_parts(self.ptr, self.num_elements) },
        }
    }

    /// Length in _elements_.
    #[inline(always)]
    pub fn num_elements(&self) -> usize {
        self.num_elements
    }

    #[inline(always)]
    pub fn partition_iter<'i, P: PartitionViewIter<T>>(
        &'i mut self,
        partitioner: P,
    ) -> impl Iterator<Item = AlignedViewMut<'i, T>> + ExactSizeIterator + TrustedLen {
        partitioner
            .partition(self.num_elements())
            .map(move |range| {
                let surrogate = self.slice_mut(range);
                // lifetime laundering
                // https://doc.rust-lang.org/nightly/src/core/slice/iter.rs.html#2011
                unsafe { transmute::<AlignedViewMut<'_, T>, AlignedViewMut<'i, T>>(surrogate) }
            })
    }

    #[inline(always)]
    pub fn as_u128_slice_mut(&mut self) -> &mut [u128] {
        unsafe {
            slice::from_raw_parts_mut(
                self.ptr.cast(),
                self.num_elements / elements_per_u128::<T>(),
            )
        }
    }

    #[inline(always)]
    pub fn copy_from<'b>(&mut self, view: impl Into<AlignedView<'b, T>>) {
        let view = view.into();

        for (dst, src) in self
            .as_u128_slice_mut()
            .iter_mut()
            .zip(view.as_u128_slice())
        {
            *dst = *src;
        }
    }

    /// # Safety
    /// * `self` must point to SMEM.
    /// * `src` must point to GMEM.
    /// * The memory pointed to by `self` cannot be accessed again
    ///   until the copy completes.
    ///
    /// Note: this method does not commit the copy group.
    #[inline(always)]
    pub unsafe fn async_copy_from_cg<'b>(&mut self, gmem: impl Into<AlignedView<'b, T>>) {
        let gmem = gmem.into();

        self.as_u128_slice_mut()
            .iter_mut()
            .zip(gmem.as_u128_slice().iter())
            .for_each(|(dst, src)| unsafe {
                async_copy::cp_async_cg_16(
                    dst as *mut u128 as *mut u32,
                    src as *const u128 as *const u32,
                );
            });
    }

    /// # Safety
    /// * `self` must point to SMEM.
    /// * `src` must point to GMEM.
    /// * The memory pointed to by `self` cannot be accessed again
    ///   until the copy completes.
    ///
    /// Note: this method does not commit the copy group.
    #[inline(always)]
    pub unsafe fn async_copy_from_ca<'b>(&mut self, gmem: impl Into<AlignedView<'b, T>>) {
        let gmem = gmem.into();

        self.as_u128_slice_mut()
            .iter_mut()
            .zip(gmem.as_u128_slice().iter())
            .for_each(|(dst, src)| unsafe {
                async_copy::cp_async_ca_16(
                    dst as *mut u128 as *mut u32,
                    src as *const u128 as *const u32,
                );
            });
    }

    /// # Safety
    /// * `self` must point to SMEM.
    /// * `src` must point to GMEM.
    /// * The memory pointed to by `self` cannot be accessed again
    ///   until the copy completes.
    ///
    /// Requires >= SM90. Should be called by a single thread.
    #[inline(always)]
    pub unsafe fn async_copy_bulk_g2s<'b>(
        &mut self,
        gmem: impl Into<AlignedView<'b, T>>,
        barrier: &mut Barrier,
    ) {
        let gmem = gmem.into();

        unsafe {
            tma::cp_async_bulk_g2s_cta(
                self.ptr.cast(),
                gmem.view.as_ptr().cast(),
                self.num_bytes() as u32,
                barrier,
            );
        }
    }

    #[inline(always)]
    pub fn num_bytes(&self) -> usize {
        size_of::<T>() * self.num_elements
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.ptr, self.num_elements) }
    }

    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.ptr, self.num_elements) }
    }
}

impl<'a, 'b, T> IntoIterator for &'b AlignedViewMut<'a, T>
where
    T: Pod + Copy,
{
    type Item = T;
    type IntoIter = impl Iterator<Item = T> + ExactSizeIterator + TrustedLen;

    fn into_iter(self) -> Self::IntoIter {
        self.as_slice().iter().copied()
    }
}

impl<'a, 'b, T> IntoIterator for &'b mut AlignedViewMut<'a, T>
where
    T: Pod,
{
    type Item = &'b mut T;
    type IntoIter = impl Iterator<Item = &'b mut T> + ExactSizeIterator + TrustedLen;

    fn into_iter(self) -> Self::IntoIter {
        self.as_mut_slice().iter_mut()
    }
}

#[macro_export]
macro_rules! make_register_view {
    ($name:ident, $t:ty, $len:expr) => {
        let mut $name: [$t; $len] = ::core::array::from_fn(|_| <$t>::ZERO);
        #[allow(unused_unsafe)]
        let mut $name = unsafe { $crate::view::AlignedViewMut::from_ptr($name.as_mut_ptr(), $len) };
    };
}

impl<'a, 'b, T> From<&'a AlignedViewMut<'b, T>> for AlignedView<'a, T>
where
    T: Pod,
{
    #[inline(always)]
    fn from(value: &'a AlignedViewMut<'a, T>) -> Self {
        value.as_view()
    }
}

impl<'a, T> From<&'a AlignedView<'a, T>> for AlignedView<'a, T>
where
    T: Pod,
{
    #[inline(always)]
    fn from(value: &'a AlignedView<'a, T>) -> Self {
        *value
    }
}

impl<'a, T> Index<usize> for AlignedViewMut<'a, T> {
    type Output = T;

    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        assert!(index < self.num_elements);
        unsafe { &*self.ptr.add(index) }
    }
}

impl<'a, T> IndexMut<usize> for AlignedViewMut<'a, T> {
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        assert!(index < self.num_elements);
        unsafe { &mut *self.ptr.add(index) }
    }
}

impl<'a, T> Index<u32> for AlignedViewMut<'a, T> {
    type Output = T;

    #[inline(always)]
    fn index(&self, index: u32) -> &Self::Output {
        &self[index as usize]
    }
}

impl<'a, T> IndexMut<u32> for AlignedViewMut<'a, T> {
    #[inline(always)]
    fn index_mut(&mut self, index: u32) -> &mut Self::Output {
        &mut self[index as usize]
    }
}

/// A way to partition a view into slices.
///
/// # Safety
/// SlicesIter must yield slices with no overlap.
pub unsafe trait PartitionViewIter<T> {
    /// Iterates the slices assigned to this thread.
    /// Units of the range are elements and should be 128-bit-aligned
    /// (see AlignedView::slice)
    type SlicesIter: Iterator<Item = Range<usize>> + ExactSizeIterator + TrustedLen;

    fn partition(self, view_len: usize) -> Self::SlicesIter;
}

/// Partition the view across lanes within a warp, interpreting
/// the view as a slice shared within the warp. The partition is such
/// that thread 0 gets 128-bit slices [0, 32, 64, ..], thread 1 gets [1, 33, 65, ..], etc.
/// This allows loads and stores operating on the partitioned slices to be fully coalesced.
#[derive(Copy, Clone, PartialEq, Eq)]
pub struct PartitionWithinWarpCoalesced;

unsafe impl<T> PartitionViewIter<T> for PartitionWithinWarpCoalesced {
    type SlicesIter = impl Iterator<Item = Range<usize>> + ExactSizeIterator + TrustedLen;

    #[inline(always)]
    fn partition(self, view_len: usize) -> Self::SlicesIter {
        let stride = 32 * elements_per_u128::<T>();
        assert!(view_len.is_multiple_of(stride));
        let n = view_len / stride;
        let lane_id = lane_id();
        (0..n).map(move |i| {
            let start = lane_id * elements_per_u128::<T>() + i * stride;
            let end = start + elements_per_u128::<T>();
            start..end
        })
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub struct PartitionU128Chunks;

unsafe impl<T> PartitionViewIter<T> for PartitionU128Chunks {
    type SlicesIter = impl Iterator<Item = Range<usize>> + ExactSizeIterator + TrustedLen;

    #[inline(always)]
    fn partition(self, view_len: usize) -> Self::SlicesIter {
        let stride = elements_per_u128::<T>();
        let n = view_len / stride;
        (0..n).map(move |i| {
            // If we don't recompute this inside the loop, rustc_codegen_cuda crashes
            // in MIR constant lowering. ¯\_(ツ)_/¯
            let stride = elements_per_u128::<T>();
            i * stride..(i + 1) * stride
        })
    }
}

#[inline(always)]
pub fn elements_per_u128<T>() -> usize {
    const {
        assert!(size_of::<u128>().is_multiple_of(size_of::<T>()));
    }
    size_of::<u128>() / size_of::<T>()
}
