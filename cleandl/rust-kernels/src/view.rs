use core::{marker::PhantomData, ops::RangeBounds, slice};

use bytemuck::Pod;

#[derive(Copy, Clone, Debug)]
pub(crate) struct AlignedSlice<T> {
    ptr: *const u128,
    len: u32,
    offset: u32,
    _marker: PhantomData<*const T>,
}

impl<T> AlignedSlice<T>
where
    T: Pod,
{
    #[inline(always)]
    pub unsafe fn from_ptr(ptr: *const T, len_in_elements: u32) -> Self {
        debug_assert!(len_in_elements.is_multiple_of(elements_per_u128::<T>()));
        Self {
            ptr: ptr.cast(),
            len: len_in_elements / elements_per_u128::<T>(),
            offset: 0,
            _marker: PhantomData,
        }
    }

    #[inline(always)]
    pub unsafe fn from_u128_ptr_with_offset(ptr: *const u128, offset: u32, len: u32) -> Self {
        Self {
            ptr,
            len,
            offset,
            _marker: PhantomData,
        }
    }

    #[inline(always)]
    pub fn len(&self) -> u32 {
        self.len
    }

    #[inline(always)]
    pub fn ptr_and_offset(&self) -> (*const u128, u32) {
        (self.ptr, self.offset)
    }

    #[inline(always)]
    pub unsafe fn slice_elements(&self, range: impl RangeBounds<u32>) -> Self {
        let start = match range.start_bound() {
            core::ops::Bound::Included(x) => *x,
            core::ops::Bound::Excluded(x) => *x + 1,
            core::ops::Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            core::ops::Bound::Included(x) => *x + 1,
            core::ops::Bound::Excluded(x) => *x,
            core::ops::Bound::Unbounded => self.len,
        };

        debug_assert!(start.is_multiple_of(elements_per_u128::<T>()));
        debug_assert!(end.is_multiple_of(elements_per_u128::<T>()));

        unsafe { self.slice_u128(start / elements_per_u128::<T>()..end / elements_per_u128::<T>()) }
    }

    #[inline(always)]
    pub unsafe fn slice_u128(&self, range: impl RangeBounds<u32>) -> Self {
        let start = match range.start_bound() {
            core::ops::Bound::Included(x) => *x,
            core::ops::Bound::Excluded(x) => *x + 1,
            core::ops::Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            core::ops::Bound::Included(x) => *x + 1,
            core::ops::Bound::Excluded(x) => *x,
            core::ops::Bound::Unbounded => self.len,
        };

        debug_assert!(start <= self.len);
        debug_assert!(end <= self.len);
        debug_assert!(start <= end);

        unsafe {
            Self {
                ptr: self.ptr,
                offset: self.offset.unchecked_add(start),
                len: end.unchecked_sub(start),
                _marker: PhantomData,
            }
        }
    }

    #[inline(always)]
    pub unsafe fn iter(&self) -> impl Iterator<Item = T> {
        struct Iter<T> {
            ptr: *const u128,
            offset_elements: u32,
            len_elements: u32,
            _marker: PhantomData<T>,
        }

        impl<T> Iterator for Iter<T>
        where
            T: Pod,
        {
            type Item = T;

            #[inline(always)]
            fn next(&mut self) -> Option<Self::Item> {
                if self.len_elements == 0 {
                    None
                } else {
                    unsafe {
                        let val = *self.ptr.cast::<T>().add(self.offset_elements as usize);
                        self.len_elements = self.len_elements.unchecked_sub(1);
                        self.offset_elements = self.offset_elements.unchecked_add(1);
                        Some(val)
                    }
                }
            }
        }

        unsafe {
            Iter::<T> {
                ptr: self.ptr,
                offset_elements: self.offset.unchecked_mul(elements_per_u128::<T>()),
                len_elements: self.len.unchecked_mul(elements_per_u128::<T>()),
                _marker: PhantomData,
            }
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub(crate) struct AlignedSliceMut<T> {
    ptr: *mut u128,
    len: u32,
    offset: u32,
    _marker: PhantomData<*mut T>,
}

impl<T> AlignedSliceMut<T>
where
    T: Pod,
{
    #[inline(always)]
    pub unsafe fn from_ptr(ptr: *mut T, len_in_elements: u32) -> Self {
        debug_assert!(len_in_elements.is_multiple_of(elements_per_u128::<T>()));
        Self {
            ptr: ptr.cast(),
            len: len_in_elements / elements_per_u128::<T>(),
            offset: 0,
            _marker: PhantomData,
        }
    }

    #[inline(always)]
    pub unsafe fn from_u128_ptr_with_offset(ptr: *mut u128, offset: u32, len: u32) -> Self {
        Self {
            ptr,
            len,
            offset,
            _marker: PhantomData,
        }
    }

    #[inline(always)]
    pub fn len(&self) -> u32 {
        self.len
    }

    #[inline(always)]
    pub fn ptr_and_offset(&self) -> (*mut u128, u32) {
        (self.ptr, self.offset)
    }

    pub fn as_ref(&self) -> AlignedSlice<T> {
        AlignedSlice {
            ptr: self.ptr,
            len: self.len,
            offset: self.offset,
            _marker: PhantomData,
        }
    }

    #[inline(always)]
    pub unsafe fn slice_elements(&self, range: impl RangeBounds<u32>) -> AlignedSlice<T> {
        unsafe { self.as_ref().slice_elements(range) }
    }

    #[inline(always)]
    pub unsafe fn slice_elements_mut(&self, range: impl RangeBounds<u32>) -> Self {
        let start = match range.start_bound() {
            core::ops::Bound::Included(x) => *x,
            core::ops::Bound::Excluded(x) => *x + 1,
            core::ops::Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            core::ops::Bound::Included(x) => *x + 1,
            core::ops::Bound::Excluded(x) => *x,
            core::ops::Bound::Unbounded => self.len,
        };

        debug_assert!(start.is_multiple_of(elements_per_u128::<T>()));
        debug_assert!(end.is_multiple_of(elements_per_u128::<T>()));

        unsafe {
            self.slice_u128_mut(start / elements_per_u128::<T>()..end / elements_per_u128::<T>())
        }
    }

    #[inline(always)]
    pub unsafe fn slice_u128(&self, range: impl RangeBounds<u32>) -> AlignedSlice<T> {
        unsafe { self.as_ref().slice_u128(range) }
    }

    #[inline(always)]
    pub unsafe fn slice_u128_mut(&self, range: impl RangeBounds<u32>) -> Self {
        let start = match range.start_bound() {
            core::ops::Bound::Included(x) => *x,
            core::ops::Bound::Excluded(x) => *x + 1,
            core::ops::Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            core::ops::Bound::Included(x) => *x + 1,
            core::ops::Bound::Excluded(x) => *x,
            core::ops::Bound::Unbounded => self.len,
        };

        debug_assert!(start <= self.len);
        debug_assert!(end <= self.len);
        debug_assert!(start <= end);

        unsafe {
            Self {
                ptr: self.ptr,
                offset: self.offset.unchecked_add(start),
                len: end.unchecked_sub(start),
                _marker: PhantomData,
            }
        }
    }

    #[inline(always)]
    pub unsafe fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        struct Iter<'a, T> {
            ptr: *mut u128,
            offset_elements: u32,
            len_elements: u32,
            _marker: PhantomData<&'a mut T>,
        }

        impl<'a, T> Iterator for Iter<'a, T>
        where
            T: Pod,
        {
            type Item = &'a mut T;

            fn next(&mut self) -> Option<Self::Item> {
                if self.len_elements == 0 {
                    None
                } else {
                    unsafe {
                        let val = &mut *self.ptr.cast::<T>().add(self.offset_elements as usize);
                        self.len_elements = self.len_elements.unchecked_sub(1);
                        self.offset_elements = self.offset_elements.unchecked_add(1);
                        Some(val)
                    }
                }
            }
        }

        unsafe {
            Iter::<T> {
                ptr: self.ptr,
                offset_elements: self.offset.unchecked_mul(elements_per_u128::<T>()),
                len_elements: self.len.unchecked_mul(elements_per_u128::<T>()),
                _marker: PhantomData,
            }
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub(crate) struct AlignedChunk<T> {
    ptr: *const u128,
    offset: u32,
    _marker: PhantomData<*const T>,
}

impl<T> AlignedChunk<T>
where
    T: Pod,
{
    pub unsafe fn from_u128_ptr_with_offset(ptr: *const u128, offset: u32) -> Self {
        Self {
            ptr,
            offset,
            _marker: PhantomData,
        }
    }

    pub unsafe fn as_slice(&self) -> &[T] {
        unsafe {
            slice::from_raw_parts(
                self.ptr.add(self.offset as usize).cast(),
                elements_per_u128::<T>() as usize,
            )
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub(crate) struct AlignedChunkMut<T> {
    ptr: *mut u128,
    offset: u32,
    _marker: PhantomData<*mut T>,
}

impl<T> AlignedChunkMut<T>
where
    T: Pod,
{
    pub unsafe fn from_u128_ptr_with_offset(ptr: *mut u128, offset: u32) -> Self {
        Self {
            ptr,
            offset,
            _marker: PhantomData,
        }
    }

    pub fn as_chunk(&self) -> AlignedChunk<T> {
        AlignedChunk {
            ptr: self.ptr,
            offset: self.offset,
            _marker: PhantomData,
        }
    }

    pub unsafe fn as_slice(&self) -> &[T] {
        unsafe {
            slice::from_raw_parts(
                self.ptr.add(self.offset as usize).cast(),
                elements_per_u128::<T>() as usize,
            )
        }
    }

    pub unsafe fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe {
            slice::from_raw_parts_mut(
                self.ptr.add(self.offset as usize).cast(),
                elements_per_u128::<T>() as usize,
            )
        }
    }

    pub unsafe fn copy_from(&mut self, src: AlignedChunk<T>) {
        unsafe {
            self.ptr
                .add(self.offset as usize)
                .write(src.ptr.add(src.offset as usize).read());
        }
    }
}

pub fn elements_per_u128<T>() -> u32 {
    const {
        assert!(size_of::<u128>().is_multiple_of(size_of::<T>()));
    }
    (size_of::<u128>() / size_of::<T>()) as u32
}

#[macro_export]
macro_rules! make_register_slice {
    ($ident:ident, $t:ty, $len:expr) => {
        let mut $ident = [<$t>::ZERO; $len as usize];
        let mut $ident = $crate::view::AlignedSliceMut::from_ptr($ident.as_mut_ptr(), $len as u32);
    };
}
