use std::{
    ops::{Bound, RangeBounds},
    sync::Arc,
};

use singe_cuda::{context::Context, memory::DeviceMemory, types::DevicePtr};
use skytens_core::{
    data_type::ElementType,
    shape::{Shape, Strides},
};

/// View into a tensor stored on some device.
///
/// Backing storage may be aliased by multiple views,
/// including with mutability.
#[derive(Debug, Clone)]
pub struct TensorView {
    cx: Arc<Context>,
    storage: Arc<DeviceMemory<u8>>,
    element_type: ElementType,
    /// Offset from start of storage of this view, in units of elements.
    offset: usize,
    shape: Shape,
    strides: Strides,
    allocation_size: usize,
}

impl TensorView {
    /// Returns a zeroed, row-major tensor allocation.
    pub fn new_zeroed(
        shape: Shape,
        element_type: ElementType,
        cx: &Arc<Context>,
    ) -> anyhow::Result<Self> {
        assert!(element_type.bits_per_element().is_power_of_two());
        let stride_alignment = 8 / element_type.bits_per_element().min(8);
        let strides = row_major_strides(shape, stride_alignment);

        let allocation_elements = shape.last_coord().flatten_with_strides(strides) + 1;
        let allocation_bytes = (allocation_elements * element_type.bits_per_element()).div_ceil(8);

        cx.bind()?;
        let storage = Arc::new(DeviceMemory::create(allocation_bytes)?);

        Ok(Self {
            cx: cx.clone(),
            storage,
            element_type,
            offset: 0,
            shape,
            strides,
            allocation_size: allocation_bytes
        })
    }

    pub fn allocation_size(&self) -> usize {
        self.allocation_size
    }

    pub fn data_ptr(&self) -> DevicePtr {
        let ptr = self.storage.as_device_ptr().as_raw();
        // TODO: This is unsound because the pointer has no provenance on the host.
        // However, singe-cuda does this too throughout its codebase,
        // so it's unavoidable for now.
        let ptr = unsafe {
            ptr.cast::<u8>()
                .add(self.offset * self.element_type.bits_per_element() / 8)
        };
        unsafe { DevicePtr::from_raw(ptr.cast()) }
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

    /// Attempts to create a view with a new shape by
    /// interpreting this view's LOGICAL contents in flattened
    /// row-major order (i.e. last dimension varies fastest),
    /// then interpreting those flattened elements as row-major
    /// representation of the target shape. The strides/physical layout of `Self`
    /// do not affect the logical view that is returned, but certain
    /// layouts will prevent this operation from working.
    /// In particular, this operation fails if the desired shape view cannot be encoded
    /// using one stride per dimension without physically
    /// changing the tensor element ordering.
    ///
    /// In practice, this means that supported reshape patterns
    /// consist of either (1) folding physically-adjacent
    /// dimensions into one dimension, or (2) unfolding a dimension
    /// into multiple dimensions.
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
                storage: Arc::clone(&self.storage),
                element_type: self.element_type,
                offset: self.offset,
                shape: target_shape,
                strides: Strides::new(target_strides),
                cx: self.cx.clone(),
                allocation_size: self.allocation_size
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

    /// Returns a contiguous slice view of the tensor view
    /// along a dimension.
    #[must_use]
    pub fn sliced(&self, dim: usize, bounds: impl RangeBounds<usize>) -> Self {
        assert!(dim < self.shape.num_dims());

        let start = match bounds.start_bound() {
            Bound::Included(x) => *x,
            Bound::Excluded(x) => *x + 1,
            Bound::Unbounded => 0,
        };
        let end = match bounds.end_bound() {
            Bound::Included(x) => *x + 1,
            Bound::Excluded(x) => *x,
            Bound::Unbounded => self.shape[dim],
        };

        let offset = self.offset + start * self.strides[dim];
        assert!(offset.is_multiple_of(8 / self.element_type.bits_per_element().min(8)));
        let mut shape = self.shape;
        shape[dim] = end.checked_sub(start).expect("invalid slice bounds");

        Self {
            offset,
            shape,
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
