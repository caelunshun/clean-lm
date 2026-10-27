use crate::{context::Context, tensor::Tensor};
use anyhow::ensure;
use skycuda::{device_memory_pool::DeviceMemoryPool, stream::Stream};
use skytens_core::{
    data_type::{ElementType, FloatType},
    shape::{Shape, Strides},
};

/// A quantized tensor and its block scales. Currently, it must be two-dimensional.
///
/// The blocks run along either the rows or the columns; the stride of
/// the block dimension must be 1. The stride of the other dimension must equal
/// the inner size rounded to the next multiple of the block size.
///
/// Scale factors are stored in the 128x4 tiled format with partition-swizzling, as read
/// by CUTLASS, cuBLASlt, etc.
#[derive(Debug)]
pub struct MxTensor {
    values: Tensor,
    scale_factors: Tensor,
    format: MxFormat,
    shape: Shape,
    strides: Strides,
    dim: MxDim,
}

impl MxTensor {
    /// # Safety
    /// Tensor data must be written to before it is read, else the behavior
    /// is undefined.
    #[track_caller]
    pub unsafe fn new_uninit(
        shape: impl Into<Shape>,
        format: MxFormat,
        dim: MxDim,
        stream: &Stream,
        cx: &Context,
    ) -> anyhow::Result<Self> {
        unsafe {
            Self::new_uninit_with_pool(shape, format, dim, stream, cx, cx.device_memory_pool())
        }
    }

    /// # Safety
    /// Tensor data must be written to before it is read, else the behavior
    /// is undefined.
    #[track_caller]
    pub unsafe fn new_uninit_with_pool(
        shape: impl Into<Shape>,
        format: MxFormat,
        dim: MxDim,
        stream: &Stream,
        cx: &Context,
        pool: &DeviceMemoryPool,
    ) -> anyhow::Result<Self> {
        let shape = shape.into();

        ensure!(shape.num_dims() == 2);

        let mut strides = Strides::new([1, 1]);
        strides[dim.non_block_dimension_2d()] =
            shape[dim.block_dimension_2d()].next_multiple_of(format.block_size());

        let mut values = unsafe {
            Tensor::new_uninit_with_pool(
                if dim == MxDim::Colwise {
                    shape.swap(0, 1)
                } else {
                    shape
                },
                format.values_type(),
                stream,
                cx,
                pool,
            )?
        };
        if dim == MxDim::Colwise {
            values = values.into_transposed(0, 1);
        }

        let sf_rows = shape[dim.non_block_dimension_2d()].next_multiple_of(128);
        let sf_cols = shape[dim.block_dimension_2d()]
            .div_ceil(format.block_size())
            .next_multiple_of(4);
        let scale_factors = unsafe {
            Tensor::new_uninit_with_pool(
                [sf_rows, sf_cols],
                format.scale_factors_type(),
                stream,
                cx,
                pool,
            )?
        };

        Ok(Self {
            values,
            scale_factors,
            format,
            shape,
            strides,
            dim,
        })
    }

    #[track_caller]
    pub fn new_zeroed(
        shape: impl Into<Shape>,
        format: MxFormat,
        dim: MxDim,
        stream: &Stream,
        cx: &Context,
    ) -> anyhow::Result<Self> {
        Self::new_zeroed_with_pool(shape, format, dim, stream, cx, cx.device_memory_pool())
    }

    #[track_caller]
    pub fn new_zeroed_with_pool(
        shape: impl Into<Shape>,
        format: MxFormat,
        dim: MxDim,
        stream: &Stream,
        cx: &Context,
        pool: &DeviceMemoryPool,
    ) -> anyhow::Result<Self> {
        let mut mx_tensor =
            unsafe { Self::new_uninit_with_pool(shape, format, dim, stream, cx, pool) }?;

        stream.fill(mx_tensor.values.storage_mut(), 0u8)?;
        stream.fill(mx_tensor.scale_factors.storage_mut(), 0u8)?;

        Ok(mx_tensor)
    }

    pub fn values(&self) -> &Tensor {
        &self.values
    }

    pub fn scale_factors(&self) -> &Tensor {
        &self.scale_factors
    }

    pub fn format(&self) -> MxFormat {
        self.format
    }

    pub fn mx_dim(&self) -> MxDim {
        self.dim
    }

    pub fn shape(&self) -> Shape {
        self.shape
    }

    pub fn strides(&self) -> Strides {
        self.strides
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum MxFormat {
    Mxfp8,
    Nvfp4,
}

impl MxFormat {
    pub fn block_size(self) -> usize {
        match self {
            MxFormat::Mxfp8 => 32,
            MxFormat::Nvfp4 => 16,
        }
    }

    pub fn values_type(self) -> ElementType {
        match self {
            MxFormat::Mxfp8 => ElementType::Float(FloatType::F8E4M3),
            MxFormat::Nvfp4 => ElementType::Float(FloatType::F4E2M1),
        }
    }

    pub fn scale_factors_type(self) -> ElementType {
        match self {
            MxFormat::Mxfp8 => ElementType::Float(FloatType::UF8E8M0),
            MxFormat::Nvfp4 => ElementType::Float(FloatType::F8E4M3),
        }
    }
}

/// Whether a tensor is block-scaled along its rows or along its columns.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum MxDim {
    Rowwise,
    Colwise,
}

impl MxDim {
    pub fn block_dimension_2d(self) -> usize {
        match self {
            MxDim::Rowwise => 1,
            MxDim::Colwise => 0,
        }
    }

    pub fn non_block_dimension_2d(self) -> usize {
        1 - self.block_dimension_2d()
    }
}
