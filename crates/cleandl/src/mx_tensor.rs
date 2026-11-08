use crate::{
    context::Context,
    tensor::{Tensor, TensorView, TensorViewMut, VirtTensorView},
};
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
        let layout = MxLayout::new(shape, format, dim)?;

        let values_storage =
            unsafe { pool.alloc_uninit_on(layout.values.required_bytes(), stream)? };
        let scale_factors_storage =
            unsafe { pool.alloc_uninit_on(layout.scale_factors.required_bytes(), stream)? };

        Ok(Self {
            values: Tensor::from_parts(layout.values, values_storage, cx),
            scale_factors: Tensor::from_parts(layout.scale_factors, scale_factors_storage, cx),
            format,
            shape,
            strides: layout.strides,
            dim,
        })
    }

    /// Wraps existing values and scale factors, which must have the layouts given
    /// by [`MxLayout::new`].
    pub fn from_parts(
        values: Tensor,
        scale_factors: Tensor,
        format: MxFormat,
        dim: MxDim,
        shape: impl Into<Shape>,
    ) -> anyhow::Result<Self> {
        let shape = shape.into();
        let layout = MxLayout::new(shape, format, dim)?;
        layout.validate(values.virt(), scale_factors.virt())?;
        Ok(Self {
            values,
            scale_factors,
            format,
            shape,
            strides: layout.strides,
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

    pub fn scale_factors_mut(&mut self) -> &mut Tensor {
        &mut self.scale_factors
    }

    pub fn values_and_scale_factors_mut(&mut self) -> (&mut Tensor, &mut Tensor) {
        (&mut self.values, &mut self.scale_factors)
    }

    /// Whether the scale factor tensor contains padding beyond the logical blocks,
    /// due to the 128x4 tile layout.
    pub fn has_scale_factor_padding(&self) -> bool {
        has_scale_factor_padding(self.shape, self.format, self.dim)
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

    pub fn view(&self) -> MxTensorView<'_> {
        MxTensorView {
            values: self.values.view(),
            scale_factors: self.scale_factors.view(),
            format: self.format,
            shape: self.shape,
            strides: self.strides,
            dim: self.dim,
        }
    }

    pub fn view_mut(&mut self) -> MxTensorViewMut<'_> {
        MxTensorViewMut {
            values: self.values.view_mut(),
            scale_factors: self.scale_factors.view_mut(),
            format: self.format,
            shape: self.shape,
            strides: self.strides,
            dim: self.dim,
        }
    }
}

/// Borrowed counterpart of [`MxTensor`].
#[derive(Debug, Clone)]
pub struct MxTensorView<'a> {
    values: TensorView<'a>,
    scale_factors: TensorView<'a>,
    format: MxFormat,
    shape: Shape,
    strides: Strides,
    dim: MxDim,
}

impl<'a> MxTensorView<'a> {
    /// `values` and `scale_factors` must have the layouts given by [`MxLayout::new`].
    pub fn from_parts(
        values: TensorView<'a>,
        scale_factors: TensorView<'a>,
        format: MxFormat,
        dim: MxDim,
        shape: impl Into<Shape>,
    ) -> anyhow::Result<Self> {
        let shape = shape.into();
        let layout = MxLayout::new(shape, format, dim)?;
        layout.validate(values.virt(), scale_factors.virt())?;
        Ok(Self {
            values,
            scale_factors,
            format,
            shape,
            strides: layout.strides,
            dim,
        })
    }

    pub fn values(&self) -> &TensorView<'a> {
        &self.values
    }

    pub fn scale_factors(&self) -> &TensorView<'a> {
        &self.scale_factors
    }

    pub fn has_scale_factor_padding(&self) -> bool {
        has_scale_factor_padding(self.shape, self.format, self.dim)
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

/// Mutably borrowed counterpart of [`MxTensor`].
#[derive(Debug)]
pub struct MxTensorViewMut<'a> {
    values: TensorViewMut<'a>,
    scale_factors: TensorViewMut<'a>,
    format: MxFormat,
    shape: Shape,
    strides: Strides,
    dim: MxDim,
}

impl<'a> MxTensorViewMut<'a> {
    /// `values` and `scale_factors` must have the layouts given by [`MxLayout::new`].
    pub fn from_parts(
        values: TensorViewMut<'a>,
        scale_factors: TensorViewMut<'a>,
        format: MxFormat,
        dim: MxDim,
        shape: impl Into<Shape>,
    ) -> anyhow::Result<Self> {
        let shape = shape.into();
        let layout = MxLayout::new(shape, format, dim)?;
        layout.validate(values.virt(), scale_factors.virt())?;
        Ok(Self {
            values,
            scale_factors,
            format,
            shape,
            strides: layout.strides,
            dim,
        })
    }

    pub fn view(&self) -> MxTensorView<'_> {
        MxTensorView {
            values: self.values.view(),
            scale_factors: self.scale_factors.view(),
            format: self.format,
            shape: self.shape,
            strides: self.strides,
            dim: self.dim,
        }
    }

    pub fn values(&self) -> &TensorViewMut<'a> {
        &self.values
    }

    pub fn scale_factors(&self) -> &TensorViewMut<'a> {
        &self.scale_factors
    }

    pub fn scale_factors_mut(&mut self) -> &mut TensorViewMut<'a> {
        &mut self.scale_factors
    }

    pub fn values_and_scale_factors_mut(
        &mut self,
    ) -> (&mut TensorViewMut<'a>, &mut TensorViewMut<'a>) {
        (&mut self.values, &mut self.scale_factors)
    }

    pub fn has_scale_factor_padding(&self) -> bool {
        has_scale_factor_padding(self.shape, self.format, self.dim)
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

/// Storage layouts of the values and scale factors of a block-scaled tensor.
#[derive(Debug, Clone)]
pub struct MxLayout {
    /// Values, in the logical `[rows, cols]` shape. Colwise values are stored transposed.
    pub values: VirtTensorView,
    /// Scale factors, of shape `[next_multiple_of(rows, 128), next_multiple_of(blocks, 4)]`
    /// in the block-scaled orientation.
    pub scale_factors: VirtTensorView,
    /// Logical strides recorded by [`MxTensor::strides`].
    pub strides: Strides,
}

impl MxLayout {
    pub fn new(shape: impl Into<Shape>, format: MxFormat, dim: MxDim) -> anyhow::Result<Self> {
        let shape = shape.into();
        ensure!(
            shape.num_dims() == 2,
            "block-scaled tensors must be 2D, got shape {shape:?}"
        );

        let mut strides = Strides::new([1, 1]);
        strides[dim.non_block_dimension_2d()] =
            shape[dim.block_dimension_2d()].next_multiple_of(format.block_size());

        let values = match dim {
            MxDim::Rowwise => VirtTensorView::new(shape, format.values_type()),
            MxDim::Colwise => {
                VirtTensorView::new(shape.swap(0, 1), format.values_type()).transposed(0, 1)
            }
        };

        let sf_rows = shape[dim.non_block_dimension_2d()].next_multiple_of(128);
        let sf_cols = shape[dim.block_dimension_2d()]
            .div_ceil(format.block_size())
            .next_multiple_of(4);
        let scale_factors = VirtTensorView::new([sf_rows, sf_cols], format.scale_factors_type());

        Ok(Self {
            values,
            scale_factors,
            strides,
        })
    }

    fn validate(
        &self,
        values: &VirtTensorView,
        scale_factors: &VirtTensorView,
    ) -> anyhow::Result<()> {
        ensure!(
            values.same_layout_as(&self.values),
            "block-scaled values have layout {values:?}, expected {:?}",
            self.values
        );
        ensure!(
            scale_factors.same_layout_as(&self.scale_factors),
            "block scale factors have layout {scale_factors:?}, expected {:?}",
            self.scale_factors
        );
        Ok(())
    }
}

fn has_scale_factor_padding(shape: Shape, format: MxFormat, dim: MxDim) -> bool {
    let rows = shape[dim.non_block_dimension_2d()];
    let blocks = shape[dim.block_dimension_2d()].div_ceil(format.block_size());
    !rows.is_multiple_of(128) || !blocks.is_multiple_of(4)
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

    pub fn has_global_scale_factor(self) -> bool {
        matches!(self, MxFormat::Nvfp4)
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
