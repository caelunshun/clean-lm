use core::{
    hint::assert_unchecked,
    iter::TrustedLen,
    marker::PhantomData,
    ops::{Index, IndexMut, Range, RangeBounds},
    slice,
};

use bytemuck::Pod;
use cuda_device::{Barrier, async_copy, tma};
use cuda_macros::device;

use crate::{WARP_SIZE, lane_id};

/// 128-bit-aligned view into a buffer.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct AlignedView<'a, T> {
    view: &'a [u128],
    _marker: PhantomData<&'a T>,
}

impl<'a, T> AlignedView<'a, T>
where
    T: Pod,
{
    #[inline(always)]
    pub fn new(view: &'a [u128]) -> Self {
        Self {
            view,
            _marker: PhantomData,
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

        Self {
            view: &self.view[start / elements_per_u128::<T>()..end / elements_per_u128::<T>()],
            _marker: PhantomData,
        }
    }

    /// Length in _elements_.
    #[inline(always)]
    pub fn num_elements(&self) -> usize {
        self.view.len() * elements_per_u128::<T>()
    }

    #[inline(always)]
    pub fn as_u128_slice(&self) -> &[u128] {
        self.view
    }

    #[inline(always)]
    pub(crate) fn partition_iter<'b, P: PartitionViewIter<T>>(
        &'b self,
        partition: P,
    ) -> impl IterWithLen<Item = AlignedView<'b, T>>
    where
        'a: 'b,
    {
        let ranges = partition.partition(self.as_u128_slice().len());

        struct Iter<'a, T, RI> {
            view: AlignedView<'a, T>,
            ranges: RI,
        }

        impl<'a, T, RI> IterWithLen for Iter<'a, T, RI>
        where
            T: Pod,
            RI: RangeIter,
        {
            type Item = AlignedView<'a, T>;

            fn len(&self) -> usize {
                self.ranges.len()
            }

            unsafe fn get_unchecked(&mut self, i: usize) -> AlignedView<'a, T> {
                unsafe {
                    let range = self.ranges.get_unchecked(i);
                    AlignedView {
                        view: &self.view.view[range.start..range.end],
                        _marker: PhantomData,
                    }
                }
            }
        }

        Iter::<T, _> {
            view: *self,
            ranges,
        }
    }

    #[inline(always)]
    pub fn num_bytes(&self) -> usize {
        self.num_elements() * size_of::<T>()
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        bytemuck::must_cast_slice(self.view)
    }
}

impl<'a, T> Index<usize> for AlignedView<'a, T>
where
    T: Pod,
{
    type Output = T;

    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        &self.as_slice()[index]
    }
}

impl<'a, T> Index<u32> for AlignedView<'a, T>
where
    T: Pod,
{
    type Output = T;

    #[inline(always)]
    fn index(&self, index: u32) -> &Self::Output {
        &self.as_slice()[index as usize]
    }
}

impl<'a, T> IntoIterator for AlignedView<'a, T>
where
    T: Pod + Copy,
{
    type Item = T;
    type IntoIter = impl Iterator<Item = T> + ExactSizeIterator + TrustedLen;

    fn into_iter(self) -> Self::IntoIter {
        bytemuck::must_cast_slice::<u128, T>(self.view)
            .iter()
            .copied()
    }
}

/// 128-bit-aligned view into a mutable buffer.
#[repr(C)]
pub struct AlignedViewMut<'a, T> {
    ptr: *mut u128,
    len: usize,
    _lifetime: PhantomData<&'a mut T>,
}

impl<'a, T> AlignedViewMut<'a, T>
where
    T: Pod,
{
    #[inline(always)]
    pub fn new(view: &'a mut [u128]) -> Self {
        Self {
            ptr: view.as_mut_ptr(),
            len: view.len(),
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

        assert!(start <= self.num_elements());
        assert!(end <= self.num_elements());
        assert!(start <= end);

        AlignedViewMut {
            ptr: unsafe { self.ptr.add(start / elements_per_u128::<T>()) },
            len: end / elements_per_u128::<T>() - start / elements_per_u128::<T>(),
            _lifetime: PhantomData,
        }
    }

    #[inline(always)]
    pub fn as_view(&self) -> AlignedView<'_, T> {
        AlignedView {
            view: unsafe { slice::from_raw_parts(self.ptr, self.len) },
            _marker: PhantomData,
        }
    }

    /// Length in _elements_.
    #[inline(always)]
    pub fn num_elements(&self) -> usize {
        self.len * elements_per_u128::<T>()
    }

    #[inline(always)]
    pub(crate) fn partition_iter<'b, P: PartitionViewIter<T>>(
        &'b mut self,
        partitioner: P,
    ) -> impl IterWithLen<Item = AlignedViewMut<'b, T>>
    where
        'a: 'b,
    {
        let ranges = partitioner.partition(self.len);

        struct Iter<'a, 'b, T, RI> {
            view: &'b mut AlignedViewMut<'a, T>,
            ranges: RI,
        }

        impl<'a, 'b, T, RI> IterWithLen for Iter<'a, 'b, T, RI>
        where
            T: Pod,
            RI: RangeIter,
        {
            type Item = AlignedViewMut<'b, T>;

            fn len(&self) -> usize {
                self.ranges.len()
            }

            unsafe fn get_unchecked(&mut self, i: usize) -> AlignedViewMut<'b, T> {
                unsafe {
                    let range = self.ranges.get_unchecked(i);
                    AlignedViewMut {
                        ptr: self.view.ptr.add(range.start),
                        len: range.end - range.start,
                        _lifetime: PhantomData,
                    }
                }
            }
        }

        Iter::<T, _> { view: self, ranges }
    }

    #[inline(always)]
    pub fn as_u128_slice_mut(&mut self) -> &mut [u128] {
        unsafe { slice::from_raw_parts_mut(self.ptr, self.len) }
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
        size_of::<T>() * self.num_elements()
    }

    #[inline(always)]
    pub fn as_u128_slice(&self) -> &[u128] {
        unsafe { slice::from_raw_parts(self.ptr, self.len) }
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        bytemuck::must_cast_slice(self.as_u128_slice())
    }

    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        bytemuck::must_cast_slice_mut(self.as_u128_slice_mut())
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

impl<'a, T> Index<usize> for AlignedViewMut<'a, T>
where
    T: Pod,
{
    type Output = T;

    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        &self.as_slice()[index]
    }
}

impl<'a, T> IndexMut<usize> for AlignedViewMut<'a, T>
where
    T: Pod,
{
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.as_mut_slice()[index]
    }
}

impl<'a, T> Index<u32> for AlignedViewMut<'a, T>
where
    T: Pod,
{
    type Output = T;

    #[inline(always)]
    fn index(&self, index: u32) -> &Self::Output {
        &self[index as usize]
    }
}

impl<'a, T> IndexMut<u32> for AlignedViewMut<'a, T>
where
    T: Pod,
{
    #[inline(always)]
    fn index_mut(&mut self, index: u32) -> &mut Self::Output {
        &mut self[index as usize]
    }
}

/// A way to partition a view into slices.
///
/// # Safety
/// SlicesIter must yield slices with no overlap.
pub(crate) unsafe trait PartitionViewIter<T> {
    /// Iterates the slices assigned to this thread.
    /// Units of the range are _128-bit values_, not elements.
    type SlicesIter: RangeIter;

    /// `view_len` has units of _128-bit values_, not elements.
    fn partition(self, view_len: usize) -> Self::SlicesIter;
}

/// Partition the view across lanes within a warp, interpreting
/// the view as a slice shared within the warp. The partition is such
/// that thread 0 gets 128-bit slices [0, 32, 64, ..], thread 1 gets [1, 33, 65, ..], etc.
/// This allows loads and stores operating on the partitioned slices to be fully coalesced.
#[derive(Copy, Clone, PartialEq, Eq)]
pub(crate) struct PartitionWithinWarpCoalesced;

unsafe impl<T> PartitionViewIter<T> for PartitionWithinWarpCoalesced
where
    T: 'static,
{
    type SlicesIter = impl RangeIter;

    #[inline(always)]
    fn partition(self, view_len: usize) -> Self::SlicesIter {
        assert!(view_len.is_multiple_of(WARP_SIZE));
        let n = view_len / WARP_SIZE;
        let lane_id = lane_id();

        struct Iter<U> {
            n: usize,
            lane_id: usize,
            _marker: PhantomData<U>,
        }
        impl<U> RangeIter for Iter<U>
        where
            U: 'static,
        {
            fn len(&self) -> usize {
                self.n
            }

            unsafe fn get_unchecked(&self, i: usize) -> Range<usize> {
                let start = self.lane_id + i * WARP_SIZE;
                start..start + 1
            }
        }

        Iter::<T> {
            n,
            lane_id,
            _marker: PhantomData,
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub(crate) struct PartitionU128Chunks;

unsafe impl<T> PartitionViewIter<T> for PartitionU128Chunks {
    type SlicesIter = impl RangeIter;

    #[inline(always)]
    fn partition(self, view_len: usize) -> Self::SlicesIter {
        struct Iter {
            n: usize,
        }

        impl RangeIter for Iter {
            fn len(&self) -> usize {
                self.n
            }

            unsafe fn get_unchecked(&self, i: usize) -> Range<usize> {
                i..i + 1
            }
        }

        Iter { n: view_len }
    }
}

#[inline(always)]
pub fn elements_per_u128<T>() -> usize {
    const {
        assert!(size_of::<u128>().is_multiple_of(size_of::<T>()));
    }
    size_of::<u128>() / size_of::<T>()
}

#[allow(clippy::len_without_is_empty)]
pub(crate) trait RangeIter: 'static {
    fn len(&self) -> usize;
    unsafe fn get_unchecked(&self, i: usize) -> Range<usize>;
}

pub(crate) trait IterWithLen {
    type Item;

    fn len(&self) -> usize;
    unsafe fn get_unchecked(&mut self, i: usize) -> Self::Item;
}

#[allow(dead_code)]
#[device]
pub(crate) fn for_each<T, I: IterWithLen<Item = T>, F: FnMut(T)>(mut iter: I, mut f: F) {
    #[llvm_unroll]
    for i in 0..iter.len() {
        f(unsafe { iter.get_unchecked(i) });
    }
}

#[device]
pub(crate) fn for_each_zip2<
    A,
    B,
    IA: IterWithLen<Item = A>,
    IB: IterWithLen<Item = B>,
    F: FnMut(A, B),
>(
    mut iter_a: IA,
    mut iter_b: IB,
    mut f: F,
) {
    assert_eq!(iter_a.len(), iter_b.len());
    #[llvm_unroll]
    for i in 0..iter_a.len() {
        f(unsafe { iter_a.get_unchecked(i) }, unsafe {
            iter_b.get_unchecked(i)
        });
    }
}
