use crate::PassableAs;
use nvvm_builder::Ptr;
use skycuda::device_memory::{DeviceVisibleMemory, DeviceVisibleMemoryMut};

unsafe impl<'a, T, M> PassableAs<'a, Ptr<T>> for &'a M where M: DeviceVisibleMemory {}
unsafe impl<'a, T, M> PassableAs<'a, Ptr<T>> for &'a mut M where M: DeviceVisibleMemoryMut {}

macro_rules! scalar_passable {
    ($ty:ty as $as:ty) => {
        unsafe impl<'a> PassableAs<'a, $as> for $ty {}
    };
}

scalar_passable!(u8 as u8);
scalar_passable!(u16 as u16);
scalar_passable!(u32 as u32);
scalar_passable!(u64 as u64);

scalar_passable!(i8 as i8);
scalar_passable!(i16 as i16);
scalar_passable!(i32 as i32);
scalar_passable!(i64 as i64);

scalar_passable!(f32 as f32);
scalar_passable!(f64 as f64);
