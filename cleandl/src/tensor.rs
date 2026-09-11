use std::ops::{Bound, RangeBounds};

use skycuda::{
    device_memory::DeviceMemory,
    memory_slice::{MemorySlice, MemorySliceMut},
    stream::Stream,
};
use skytens_core::{
    data_type::ElementType,
    shape::{Shape, Strides},
};

use crate::context::Context;

/// Tensor stored on some device, owning its backing allocation.
#[derive(Debug)]
pub struct Tensor {
    virt: VirtTensorView,
    cx: Context,
    storage: DeviceMemory,
}

impl Tensor {
    /// Returns a zeroed, row-major tensor allocation.
    pub fn new_zeroed(
        shape: impl Into<Shape>,
        element_type: impl Into<ElementType>,
        stream: &Stream,
        cx: &Context,
    ) -> anyhow::Result<Self> {
        let shape = shape.into();
        let element_type = element_type.into();
        let virt = VirtTensorView::new(shape, element_type);
        let allocation_bytes = virt.required_bytes();

        let storage = cx
            .device_memory_pool()
            .alloc_zeroed_on(allocation_bytes, stream)?;

        Ok(Self {
            virt,
            cx: cx.clone(),
            storage,
        })
    }

    pub fn cx(&self) -> &Context {
        &self.cx
    }

    pub fn storage(&self) -> &DeviceMemory {
        &self.storage
    }

    pub fn storage_mut(&mut self) -> &mut DeviceMemory {
        &mut self.storage
    }

    pub fn element_type(&self) -> ElementType {
        self.virt.element_type()
    }

    pub fn shape(&self) -> Shape {
        self.virt.shape()
    }

    pub fn strides(&self) -> Strides {
        self.virt.strides()
    }

    pub fn virt(&self) -> &VirtTensorView {
        &self.virt
    }

    pub fn is_contiguous(&self) -> bool {
        self.virt.is_contiguous()
    }

    pub fn is_packed_row_major(&self) -> bool {
        self.virt.is_packed_row_major()
    }

    /// Returns a view over the entirety of this tensor.
    #[must_use]
    pub fn view(&self) -> TensorView<'_> {
        TensorView {
            virt: self.virt.clone(),
            cx: self.cx.clone(),
            storage: &self.storage,
            offset: 0,
            len: self.storage.len(),
        }
    }

    /// Returns a mutable view over the entirety of this tensor.
    #[must_use]
    pub fn view_mut(&mut self) -> TensorViewMut<'_> {
        let len = self.storage.len();
        TensorViewMut {
            virt: self.virt.clone(),
            cx: self.cx.clone(),
            storage: &mut self.storage,
            offset: 0,
            len,
        }
    }

    /// Attempts to reshape the logical contents without changing their order.
    #[must_use]
    pub fn try_reshaped(&self, target_shape: impl Into<Shape>) -> Option<TensorView<'_>> {
        self.view().try_reshaped(target_shape)
    }

    #[must_use]
    pub fn try_reshaped_mut(
        &mut self,
        target_shape: impl Into<Shape>,
    ) -> Option<TensorViewMut<'_>> {
        let virt = self.virt.try_reshaped(target_shape.into())?;
        let mut view = self.view_mut();
        view.virt = virt;
        Some(view)
    }

    #[must_use]
    pub fn reshaped(&self, target_shape: impl Into<Shape>) -> TensorView<'_> {
        self.view().reshaped(target_shape)
    }

    #[must_use]
    pub fn reshaped_mut(&mut self, target_shape: impl Into<Shape>) -> TensorViewMut<'_> {
        let virt = self.virt.reshaped(target_shape.into());
        let mut view = self.view_mut();
        view.virt = virt;
        view
    }

    #[must_use]
    pub fn transposed(&self, dim_a: usize, dim_b: usize) -> TensorView<'_> {
        self.view().transposed(dim_a, dim_b)
    }

    #[must_use]
    pub fn transposed_mut(&mut self, dim_a: usize, dim_b: usize) -> TensorViewMut<'_> {
        let virt = self.virt.transposed(dim_a, dim_b);
        let mut view = self.view_mut();
        view.virt = virt;
        view
    }

    #[must_use]
    pub fn unsqueezed(&self) -> TensorView<'_> {
        self.view().unsqueezed()
    }

    #[must_use]
    pub fn unsqueezed_mut(&mut self) -> TensorViewMut<'_> {
        let shape = self.shape().prepend(1);
        self.reshaped_mut(shape)
    }

    /// Returns a contiguous slice view along a dimension.
    #[must_use]
    pub fn sliced(&self, dim: usize, bounds: impl RangeBounds<usize>) -> TensorView<'_> {
        self.view().sliced(dim, bounds)
    }

    #[must_use]
    pub fn sliced_mut(&mut self, dim: usize, bounds: impl RangeBounds<usize>) -> TensorViewMut<'_> {
        let mut view = self.view_mut();
        view.slice_in_place(dim, bounds);
        view
    }
}

/// Immutable view into a (sub-slice of a) tensor.
#[derive(Debug)]
pub struct TensorView<'a> {
    virt: VirtTensorView,
    cx: Context,
    storage: &'a DeviceMemory,
    /// Byte offset of this view within `storage`.
    offset: usize,
    /// Length of this view, in bytes.
    len: usize,
}

impl<'a> Clone for TensorView<'a> {
    fn clone(&self) -> Self {
        Self {
            virt: self.virt.clone(),
            cx: self.cx.clone(),
            storage: self.storage,
            offset: self.offset,
            len: self.len,
        }
    }
}

impl<'a> TensorView<'a> {
    pub fn cx(&self) -> &Context {
        &self.cx
    }

    pub fn storage(&self) -> MemorySlice<&'a DeviceMemory> {
        self.storage.slice(self.offset..self.offset + self.len)
    }

    pub fn element_type(&self) -> ElementType {
        self.virt.element_type()
    }

    pub fn shape(&self) -> Shape {
        self.virt.shape()
    }

    pub fn strides(&self) -> Strides {
        self.virt.strides()
    }

    pub fn virt(&self) -> &VirtTensorView {
        &self.virt
    }

    pub fn is_contiguous(&self) -> bool {
        self.virt.is_contiguous()
    }

    pub fn is_packed_row_major(&self) -> bool {
        self.virt.is_packed_row_major()
    }

    /// Attempts to reshape the logical contents without changing their order.
    #[must_use]
    pub fn try_reshaped(&self, target_shape: impl Into<Shape>) -> Option<Self> {
        Some(Self {
            virt: self.virt.try_reshaped(target_shape.into())?,
            ..self.clone()
        })
    }

    #[must_use]
    pub fn reshaped(&self, target_shape: impl Into<Shape>) -> Self {
        Self {
            virt: self.virt.reshaped(target_shape.into()),
            ..self.clone()
        }
    }

    #[must_use]
    pub fn transposed(&self, dim_a: usize, dim_b: usize) -> Self {
        Self {
            virt: self.virt.transposed(dim_a, dim_b),
            ..self.clone()
        }
    }

    #[must_use]
    pub fn unsqueezed(&self) -> Self {
        self.reshaped(self.shape().prepend(1))
    }

    /// Returns a contiguous slice view along a dimension.
    #[must_use]
    pub fn sliced(&self, dim: usize, bounds: impl RangeBounds<usize>) -> Self {
        let (virt, offset_bytes) = self.virt.sliced(dim, bounds);
        Self {
            virt,
            offset: self.offset + offset_bytes,
            len: self.len - offset_bytes,
            ..self.clone()
        }
    }
}

/// Mutable view into a (subslice of a) tensor.
#[derive(Debug)]
pub struct TensorViewMut<'a> {
    virt: VirtTensorView,
    cx: Context,
    storage: &'a mut DeviceMemory,
    /// Byte offset of this view within `storage`.
    offset: usize,
    /// Length of this view, in bytes.
    len: usize,
}

impl<'a> TensorViewMut<'a> {
    pub fn cx(&self) -> &Context {
        &self.cx
    }

    pub fn storage(&self) -> MemorySlice<&DeviceMemory> {
        self.storage.slice(self.offset..self.offset + self.len)
    }

    pub fn storage_mut(&mut self) -> MemorySliceMut<&mut DeviceMemory> {
        self.storage.slice_mut(self.offset..self.offset + self.len)
    }

    pub fn element_type(&self) -> ElementType {
        self.virt.element_type()
    }

    pub fn shape(&self) -> Shape {
        self.virt.shape()
    }

    pub fn strides(&self) -> Strides {
        self.virt.strides()
    }

    pub fn virt(&self) -> &VirtTensorView {
        &self.virt
    }

    pub fn is_contiguous(&self) -> bool {
        self.virt.is_contiguous()
    }

    pub fn is_packed_row_major(&self) -> bool {
        self.virt.is_packed_row_major()
    }

    /// Returns an immutable view over the same memory.
    #[must_use]
    pub fn view(&self) -> TensorView<'_> {
        TensorView {
            virt: self.virt.clone(),
            cx: self.cx.clone(),
            storage: &*self.storage,
            offset: self.offset,
            len: self.len,
        }
    }

    /// Reborrows this view mutably.
    #[must_use]
    pub fn view_mut(&mut self) -> TensorViewMut<'_> {
        TensorViewMut {
            virt: self.virt.clone(),
            cx: self.cx.clone(),
            storage: &mut *self.storage,
            offset: self.offset,
            len: self.len,
        }
    }

    /// Attempts to reshape the logical contents without changing their order.
    #[must_use]
    pub fn try_reshaped(&self, target_shape: impl Into<Shape>) -> Option<TensorView<'_>> {
        self.view().try_reshaped(target_shape)
    }

    #[must_use]
    pub fn try_reshaped_mut(
        &mut self,
        target_shape: impl Into<Shape>,
    ) -> Option<TensorViewMut<'_>> {
        let virt = self.virt.try_reshaped(target_shape.into())?;
        let mut view = self.view_mut();
        view.virt = virt;
        Some(view)
    }

    #[must_use]
    pub fn reshaped(&self, target_shape: impl Into<Shape>) -> TensorView<'_> {
        self.view().reshaped(target_shape)
    }

    #[must_use]
    pub fn reshaped_mut(&mut self, target_shape: impl Into<Shape>) -> TensorViewMut<'_> {
        let virt = self.virt.reshaped(target_shape.into());
        let mut view = self.view_mut();
        view.virt = virt;
        view
    }

    #[must_use]
    pub fn transposed(&self, dim_a: usize, dim_b: usize) -> TensorView<'_> {
        self.view().transposed(dim_a, dim_b)
    }

    #[must_use]
    pub fn transposed_mut(&mut self, dim_a: usize, dim_b: usize) -> TensorViewMut<'_> {
        let virt = self.virt.transposed(dim_a, dim_b);
        let mut view = self.view_mut();
        view.virt = virt;
        view
    }

    #[must_use]
    pub fn unsqueezed(&self) -> TensorView<'_> {
        self.view().unsqueezed()
    }

    #[must_use]
    pub fn unsqueezed_mut(&mut self) -> TensorViewMut<'_> {
        let shape = self.shape().prepend(1);
        self.reshaped_mut(shape)
    }

    /// Returns a contiguous slice view along a dimension.
    #[must_use]
    pub fn sliced(&self, dim: usize, bounds: impl RangeBounds<usize>) -> TensorView<'_> {
        self.view().sliced(dim, bounds)
    }

    #[must_use]
    pub fn sliced_mut(&mut self, dim: usize, bounds: impl RangeBounds<usize>) -> TensorViewMut<'_> {
        let mut view = self.view_mut();
        view.slice_in_place(dim, bounds);
        view
    }

    fn slice_in_place(&mut self, dim: usize, bounds: impl RangeBounds<usize>) {
        let (virt, offset_bytes) = self.virt.sliced(dim, bounds);
        self.virt = virt;
        self.offset += offset_bytes;
        self.len -= offset_bytes;
    }
}

fn row_major_strides(shape: Shape, align: usize) -> Strides {
    let mut strides = Strides::new([]);
    let mut stride = 1;
    for dim in shape.dims().rev() {
        strides = strides.prepend(stride);
        stride *= dim;
        stride = stride.next_multiple_of(align);
    }
    strides
}

/// Tensor view shape and stride tracking without backing physical storage.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VirtTensorView {
    element_type: ElementType,
    shape: Shape,
    strides: Strides,
}

impl VirtTensorView {
    pub fn new(shape: impl Into<Shape>, element_type: impl Into<ElementType>) -> Self {
        let element_type = element_type.into();
        let shape = shape.into();
        let stride_alignment = 8 / element_type.bits_per_element().min(8);
        let strides = row_major_strides(shape, stride_alignment);
        Self::with_strides(shape, strides, element_type)
    }

    pub fn with_strides(
        shape: impl Into<Shape>,
        strides: impl Into<Strides>,
        element_type: impl Into<ElementType>,
    ) -> Self {
        let element_type = element_type.into();
        let shape = shape.into();
        let strides = strides.into();

        assert_eq!(shape.num_dims(), strides.num_dims());

        assert!(element_type.bits_per_element().is_power_of_two());

        Self {
            element_type,
            shape,
            strides,
        }
    }

    pub fn same_layout_as(&self, other: &Self) -> bool {
        self.shape == other.shape
            && self.strides == other.strides
            && self.element_type == other.element_type
    }

    pub fn element_type(&self) -> ElementType {
        self.element_type
    }

    pub fn shape(&self) -> Shape {
        self.shape
    }

    pub fn strides(&self) -> Strides {
        self.strides
    }

    /// Attempts to reshape the logical contents without changing their order.
    #[must_use]
    pub fn try_reshaped(&self, target_shape: Shape) -> Option<Self> {
        let element_count = self.shape.element_count();
        if element_count != target_shape.element_count() {
            return None;
        }
        if self.shape == target_shape {
            return Some(self.clone());
        }

        let mut target_strides = vec![0; target_shape.num_dims()];
        if element_count == 0 || self.shape.num_dims() == 0 {
            let mut stride = 1usize;
            for dim in (0..target_shape.num_dims()).rev() {
                target_strides[dim] = stride;
                stride = stride.saturating_mul(target_shape[dim]);
            }
            return Some(Self {
                shape: target_shape,
                strides: Strides::new(target_strides),
                ..self.clone()
            });
        }

        let mut target_dim = target_shape.num_dims();
        let mut chunk_base_stride = self.strides[self.shape.num_dims() - 1];
        let mut source_chunk_elements = 1usize;
        let mut target_chunk_elements = 1usize;

        for source_dim in (0..self.shape.num_dims()).rev() {
            source_chunk_elements *= self.shape[source_dim];
            let at_chunk_boundary = source_dim == 0
                || (self.shape[source_dim - 1] != 1
                    && source_chunk_elements.checked_mul(chunk_base_stride)
                        != Some(self.strides[source_dim - 1]));

            if at_chunk_boundary {
                while target_dim > 0
                    && (target_chunk_elements < source_chunk_elements
                        || target_shape[target_dim - 1] == 1)
                {
                    target_dim -= 1;
                    target_strides[target_dim] =
                        match target_chunk_elements.checked_mul(chunk_base_stride) {
                            Some(stride) => stride,
                            None if target_shape[target_dim] == 1 => chunk_base_stride,
                            None => return None,
                        };
                    target_chunk_elements *= target_shape[target_dim];
                }
                if target_chunk_elements != source_chunk_elements {
                    return None;
                }
                if source_dim > 0 {
                    chunk_base_stride = self.strides[source_dim - 1];
                    source_chunk_elements = 1;
                    target_chunk_elements = 1;
                }
            }
        }

        if target_dim != 0 {
            return None;
        }
        Some(Self {
            shape: target_shape,
            strides: Strides::new(target_strides),
            ..self.clone()
        })
    }

    #[must_use]
    pub fn reshaped(&self, target_shape: Shape) -> Self {
        self.try_reshaped(target_shape).unwrap_or_else(|| {
            panic!(
                "cannot reshape tensor of shape {:?}, strides {:?} into shape {:?}",
                self.shape, self.strides, target_shape
            )
        })
    }

    #[must_use]
    pub fn transposed(&self, dim_a: usize, dim_b: usize) -> Self {
        assert!(dim_a < self.shape.num_dims());
        assert!(dim_b < self.shape.num_dims());

        let shape = self.shape.swap(dim_a, dim_b);
        let strides = self.strides.swap(dim_a, dim_b);

        Self {
            shape,
            strides,
            ..self.clone()
        }
    }

    /// Returns a contiguous slice view along a dimension, plus the byte offset
    /// from the start of this view's base pointer needed to reach it.
    #[must_use]
    pub fn sliced(&self, dim: usize, bounds: impl RangeBounds<usize>) -> (Self, usize) {
        assert!(dim < self.shape.num_dims());
        let start = match bounds.start_bound() {
            Bound::Included(x) => *x,
            Bound::Excluded(x) => x.checked_add(1).expect("invalid slice bounds"),
            Bound::Unbounded => 0,
        };
        let end = match bounds.end_bound() {
            Bound::Included(x) => x.checked_add(1).expect("invalid slice bounds"),
            Bound::Excluded(x) => *x,
            Bound::Unbounded => self.shape[dim],
        };
        assert!(end <= self.shape[dim], "slice bounds out of range");

        let offset_elements = start * self.strides[dim];
        assert!(
            (offset_elements * self.element_type.bits_per_element()).is_multiple_of(8),
            "slice starts at a partial byte"
        );
        let offset_bytes = (offset_elements * self.element_type.bits_per_element()) / 8;

        let mut shape = self.shape;
        shape[dim] = end.checked_sub(start).expect("invalid slice bounds");
        (
            Self {
                shape,
                ..self.clone()
            },
            offset_bytes,
        )
    }

    /// Number of bytes of storage this view addresses, starting at its base pointer.
    pub fn required_bytes(&self) -> usize {
        let elements = self.shape.last_coord().flatten_with_strides(self.strides) + 1;
        (elements * self.element_type.bits_per_element()).div_ceil(8)
    }

    pub fn is_contiguous(&self) -> bool {
        let mut packed_dims = Vec::new();
        for dim in 0..self.shape.num_dims() {
            packed_dims.push((dim, self.shape[dim], self.strides[dim]));
        }

        packed_dims.sort_unstable_by_key(|(_, _, stride)| *stride);

        let mut expected_stride = 1;
        for (_, size, stride) in packed_dims {
            if stride != expected_stride {
                return false;
            }
            expected_stride *= size;
        }

        true
    }

    pub fn is_packed_row_major(&self) -> bool {
        self.strides
            == row_major_strides(self.shape, 8 / self.element_type.bits_per_element().min(8))
    }
}
