use bytemuck::Pod;

use crate::view::{AlignedView, AlignedViewMut, elements_per_u128};

pub struct StagingBufferRing<'a, T> {
    view: AlignedViewMut<'a, T>,
    elements_per_slot: usize,
    num_slots: usize,
}

impl<'a, T> StagingBufferRing<'a, T>
where
    T: Pod,
{
    #[inline]
    pub fn new(smem: &'a mut [u128], elements_per_slot: usize, num_slots: usize) -> Self {
        assert_eq!(
            smem.len(),
            elements_per_slot * num_slots / elements_per_u128::<T>()
        );
        Self {
            view: AlignedViewMut::new(smem),
            elements_per_slot,
            num_slots,
        }
    }

    #[inline]
    pub fn num_slots(&self) -> usize {
        self.num_slots
    }

    #[inline]
    pub fn slot(&self, i: usize) -> AlignedView<'_, T> {
        self.view
            .slice(i * self.elements_per_slot..(i + 1) * self.elements_per_slot)
    }

    #[inline]
    pub fn slot_mut(&mut self, i: usize) -> AlignedViewMut<'_, T> {
        self.view
            .slice_mut(i * self.elements_per_slot..(i + 1) * self.elements_per_slot)
    }
}
