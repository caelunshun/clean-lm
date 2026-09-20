use core::marker::PhantomData;

use bytemuck::Pod;
use cuda_macros::device;

use crate::{
    WARP_SIZE, lane_id,
    view::{AlignedChunk, AlignedChunkMut, AlignedSlice, AlignedSliceMut},
};

#[derive(Copy, Clone)]
pub(crate) struct PartitionedView<T, F> {
    base_ptr: *const u128,
    offset: u32,
    len: u32,
    get_nth_u128_offset: F,
    _marker: PhantomData<*const T>,
}

impl<T, F> AnyPartitionedView for PartitionedView<T, F>
where
    T: Pod,
    F: Fn(u32) -> u32,
{
    type Item = AlignedChunk<T>;

    #[inline(always)]
    fn len(&self) -> u32 {
        self.len
    }

    #[inline(always)]
    unsafe fn get_nth_chunk(&self, n: u32) -> AlignedChunk<T> {
        unsafe {
            AlignedChunk::from_u128_ptr_with_offset(
                self.base_ptr,
                self.offset.unchecked_add((self.get_nth_u128_offset)(n)),
            )
        }
    }
}

#[derive(Copy, Clone)]
pub(crate) struct PartitionedViewMut<T, F> {
    base_ptr: *mut u128,
    offset: u32,
    len: u32,
    get_nth_u128_offset: F,
    _marker: PhantomData<*mut T>,
}

impl<T, F> AnyPartitionedView for PartitionedViewMut<T, F>
where
    T: Pod,
    F: Fn(u32) -> u32,
{
    type Item = AlignedChunkMut<T>;

    #[inline(always)]
    fn len(&self) -> u32 {
        self.len
    }

    #[inline(always)]
    unsafe fn get_nth_chunk(&self, n: u32) -> AlignedChunkMut<T> {
        unsafe {
            AlignedChunkMut::from_u128_ptr_with_offset(
                self.base_ptr,
                self.offset.unchecked_add((self.get_nth_u128_offset)(n)),
            )
        }
    }
}

pub(crate) trait AnyPartitionedView {
    type Item;

    fn len(&self) -> u32;
    unsafe fn get_nth_chunk(&self, n: u32) -> Self::Item;
}

#[inline(always)]
pub(crate) unsafe fn partition<T: Pod, P: Partitioner>(
    slice: AlignedSlice<T>,
    partitioner: P,
) -> PartitionedView<T, P::GetNthU128Offset> {
    let (len, get_nth_u128_offset) = partitioner.partition(slice.len());
    PartitionedView {
        base_ptr: slice.ptr_and_offset().0,
        offset: slice.ptr_and_offset().1,
        len,
        get_nth_u128_offset,
        _marker: PhantomData,
    }
}

#[inline(always)]
pub(crate) unsafe fn partition_mut<T: Pod, P: Partitioner>(
    slice: AlignedSliceMut<T>,
    partitioner: P,
) -> PartitionedViewMut<T, P::GetNthU128Offset> {
    let (len, get_nth_u128_offset) = partitioner.partition(slice.len());
    PartitionedViewMut {
        base_ptr: slice.ptr_and_offset().0,
        offset: slice.ptr_and_offset().1,
        len,
        get_nth_u128_offset,
        _marker: PhantomData,
    }
}

#[device]
pub(crate) unsafe fn for_each_chunk<V, F>(view: V, mut callback: F)
where
    V: AnyPartitionedView,
    F: FnMut(V::Item),
{
    let len = view.len();
    let mut i = 0;
    #[llvm_unroll]
    while i < len {
        unsafe {
            callback(view.get_nth_chunk(i));
        }
        i += 1;
    }
}

#[device]
pub(crate) unsafe fn for_each_chunk_zip2<V1, V2, F>(view1: V1, view2: V2, mut callback: F)
where
    V1: AnyPartitionedView,
    V2: AnyPartitionedView,
    F: FnMut(V1::Item, V2::Item),
{
    let len = view1.len();
    debug_assert_eq!(view1.len(), view2.len());

    let mut i = 0;
    #[llvm_unroll]
    while i < len {
        unsafe {
            callback(view1.get_nth_chunk(i), view2.get_nth_chunk(i));
        }
        i += 1;
    }
}

pub(crate) trait Partitioner {
    type GetNthU128Offset: Fn(u32) -> u32 + 'static;

    fn partition(&self, num_u128_in_view: u32) -> (u32, Self::GetNthU128Offset);
}

pub struct AllChunksPartitioner;

impl Partitioner for AllChunksPartitioner {
    type GetNthU128Offset = impl Fn(u32) -> u32 + 'static;

    fn partition(&self, num_u128_in_view: u32) -> (u32, Self::GetNthU128Offset) {
        (num_u128_in_view, |i| i)
    }
}

pub struct DistributeCoalescedWithinWarpPartitioner;

impl Partitioner for DistributeCoalescedWithinWarpPartitioner {
    type GetNthU128Offset = impl Fn(u32) -> u32 + 'static;

    fn partition(&self, num_u128_in_view: u32) -> (u32, Self::GetNthU128Offset) {
        debug_assert!(num_u128_in_view.is_multiple_of(WARP_SIZE));
        (num_u128_in_view / WARP_SIZE, |i| lane_id() + i * WARP_SIZE)
    }
}
