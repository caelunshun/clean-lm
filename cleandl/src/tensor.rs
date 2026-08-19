use std::{
    ops::{Bound, RangeBounds},
    sync::Arc,
};

use anyhow::ensure;
use singe_cuda::{memory::DeviceMemory, types::DevicePtr, view::DeviceView};
use skytens_core::{
    data_type::ElementType,
    shape::{Shape, Strides},
};

use crate::context::{Context, CudaStream};

/// View into a tensor stored on some device.
///
/// Backing storage may be aliased by multiple views, including with mutability.
#[derive(Debug, Clone)]
pub struct TensorView {
    virt: VirtTensorView,
    cx: Context,
    storage: Arc<DeviceMemory<u8>>,
    allocation_size: usize,
}

impl TensorView {
    /// Returns a zeroed, row-major tensor allocation.
    pub fn new_zeroed(
        shape: Shape,
        element_type: ElementType,
        cx: &Context,
    ) -> anyhow::Result<Self> {
        let virt = VirtTensorView::new(shape, element_type);
        let allocation_elements = shape.last_coord().flatten_with_strides(virt.strides()) + 1;
        let allocation_bytes = (allocation_elements * element_type.bits_per_element()).div_ceil(8);

        cx.cuda().bind()?;
        let storage = Arc::new(DeviceMemory::create(allocation_bytes)?);

        Ok(Self {
            virt,
            cx: cx.clone(),
            storage,
            allocation_size: allocation_bytes,
        })
    }

    pub fn cx(&self) -> &Context {
        &self.cx
    }

    pub fn allocation_size(&self) -> usize {
        self.allocation_size
    }

    pub fn total_elements_including_padding(&self) -> usize {
        self.shape()
            .last_coord()
            .flatten_with_strides(self.strides())
            + 1
    }

    pub fn data_ptr(&self) -> DevicePtr {
        let ptr = self.storage.as_device_ptr().as_raw();
        // TODO: This is unsound because the pointer has no provenance on the host.
        // However, singe-cuda does this too throughout its codebase.
        let ptr = unsafe {
            ptr.cast::<u8>()
                .add(self.virt.offset() * self.virt.element_type().bits_per_element() / 8)
        };
        unsafe { DevicePtr::from_raw(ptr.cast()) }
    }

    pub fn data_view(&self) -> DeviceView<'_, u8> {
        let ptr = self.data_ptr();
        unsafe { DeviceView::from_raw_parts(ptr.as_const_ptr().cast(), self.allocation_size) }
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
        let mut packed_dims = Vec::new();
        for dim in 0..self.shape().num_dims() {
            packed_dims.push((dim, self.shape()[dim], self.strides()[dim]));
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

    pub fn memset_u8(&self, x: u8, stream: &CudaStream) -> anyhow::Result<()> {
        ensure!(
            self.is_contiguous(),
            "memset unimplemented for non-contiguous tensors"
        );
        unsafe {
            DeviceMemory::set_async(
                self.data_ptr().as_raw().cast::<u8>(),
                x,
                self.allocation_size,
                stream.get(),
            )?;
        }

        Ok(())
    }

    /// Attempts to reshape the logical contents without changing their order.
    #[must_use]
    pub fn try_reshaped(&self, target_shape: Shape) -> Option<Self> {
        Some(Self {
            virt: self.virt.try_reshaped(target_shape)?,
            ..self.clone()
        })
    }

    #[must_use]
    pub fn reshaped(&self, target_shape: Shape) -> Self {
        Self {
            virt: self.virt.reshaped(target_shape),
            ..self.clone()
        }
    }

    /// Returns a contiguous slice view along a dimension.
    #[must_use]
    pub fn sliced(&self, dim: usize, bounds: impl RangeBounds<usize>) -> Self {
        let virt = self.virt.sliced(dim, bounds);
        let allocation_size = virt.offset - self.virt.offset;
        Self {
            virt,
            allocation_size,
            ..self.clone()
        }
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
#[derive(Debug, Clone)]
pub struct VirtTensorView {
    element_type: ElementType,
    /// Offset from the start of storage, in elements.
    offset: usize,
    shape: Shape,
    strides: Strides,
}

impl VirtTensorView {
    /// Creates a row-major virtual tensor view. Strides are aligned so that
    /// every row of a sub-byte element type starts at a byte boundary.
    pub fn new(shape: Shape, element_type: ElementType) -> Self {
        assert!(element_type.bits_per_element().is_power_of_two());
        let stride_alignment = 8 / element_type.bits_per_element().min(8);
        Self {
            element_type,
            offset: 0,
            shape,
            strides: row_major_strides(shape, stride_alignment),
        }
    }

    pub fn element_type(&self) -> ElementType {
        self.element_type
    }

    pub fn offset(&self) -> usize {
        self.offset
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

    /// Returns a contiguous slice view along a dimension.
    #[must_use]
    pub fn sliced(&self, dim: usize, bounds: impl RangeBounds<usize>) -> Self {
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

        let offset = self.offset + start * self.strides[dim];
        assert!(
            offset.is_multiple_of(8 / self.element_type.bits_per_element().min(8)),
            "slice starts at a partial byte"
        );
        let mut shape = self.shape;
        shape[dim] = end.checked_sub(start).expect("invalid slice bounds");
        Self {
            offset,
            shape,
            ..self.clone()
        }
    }
}
