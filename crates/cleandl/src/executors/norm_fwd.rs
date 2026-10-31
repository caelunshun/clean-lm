use crate::{
    context::{Context, ResourceKey},
    disk_cache::DiskCacheKey,
    executors::{
        Executor, ensure_base_aligned, ensure_divisible, ensure_element_type, ensure_min_dims,
        ensure_nonempty, ensure_option_presence, ensure_packed_row_major, ensure_row_bytes_aligned,
        ensure_shape, to_u32,
    },
    mx_tensor::{MxDim, MxFormat, MxTensor},
    tensor::{Tensor, TensorView},
};
use anyhow::{bail, ensure};
use kernel_build_util::PtrArg;
use kernel_generator::{
    Arch,
    kernels::{
        KernelGenerator,
        norm::{
            self, OutputTypes,
            forward::{ExecArgs, NormForwardKernelGenerator, OutputBuffers},
        },
    },
};
use serde::{Deserialize, Serialize};
use skycuda::{device_memory_pool::DeviceMemoryPool, stream::Stream};
use skytens_core::data_type::{ElementType, FloatType};
use std::sync::Arc;

/// Required alignment of row starts for TMA bulk copies.
const ROW_ALIGN_BYTES: usize = 16;

const BF16: ElementType = ElementType::Float(FloatType::Bf16);
const F32: ElementType = ElementType::Float(FloatType::F32);

pub struct NormFwdExecutor {
    kernel: norm::forward::BuiltKernel,
}

impl NormFwdExecutor {
    pub fn get(config: norm::Config, cx: &Context) -> anyhow::Result<Arc<Self>> {
        cx.get_or_create_resource(ConfigWrapper { inner: config })
    }

    fn validate(&self, args: &Args) -> anyhow::Result<(usize, usize)> {
        let config = self.kernel.config();
        let x = &args.x;

        ensure_element_type("x", x, BF16)?;
        ensure_min_dims("x", x, 2)?;
        ensure_nonempty("x", x)?;
        ensure_packed_row_major("x", x)?;
        ensure_base_aligned("x", x, ROW_ALIGN_BYTES)?;
        ensure_row_bytes_aligned("x", x, ROW_ALIGN_BYTES)?;

        let shape = x.shape();
        let row_size = shape[shape.num_dims() - 1];
        let num_rows = shape.element_count() / row_size;
        ensure!(
            row_size <= config.max_row_size,
            "row size {row_size} exceeds the kernel's max row size {}",
            config.max_row_size
        );

        for (name, param, expected_present) in [
            ("scale", &args.scale, config.has_scale),
            ("bias", &args.bias, config.has_bias),
        ] {
            ensure_option_presence(name, param, expected_present)?;
            if let Some(param) = param {
                ensure_element_type(name, param, BF16)?;
                ensure_shape(name, param, [row_size])?;
                ensure_packed_row_major(name, param)?;
                ensure_base_aligned(name, param, ROW_ALIGN_BYTES)?;
            }
        }

        match config.output_types {
            OutputTypes::Bf16 => {}
            OutputTypes::Mxfp8Rowwise => {
                ensure_divisible("row size", row_size, MxFormat::Mxfp8.block_size())?;
            }
            OutputTypes::Mxfp8RowwiseAndColwise => {
                ensure_divisible("row size", row_size, MxFormat::Mxfp8.block_size())?;
                ensure_divisible("row count", num_rows, MxFormat::Mxfp8.block_size())?;
            }
            OutputTypes::Nvfp4Rowwise => bail!("NVFP4 norm output is not supported"),
        }

        Ok((num_rows, row_size))
    }
}

fn alloc_mx_output(
    shape: [usize; 2],
    dim: MxDim,
    stream: &Stream,
    pool: &DeviceMemoryPool,
    cx: &Context,
) -> anyhow::Result<MxTensor> {
    // Soundness: the kernel writes all values and all non-padding scale factors;
    // padding scale factors are zeroed.
    let mut tensor =
        unsafe { MxTensor::new_uninit_with_pool(shape, MxFormat::Mxfp8, dim, stream, cx, pool)? };
    if tensor.has_scale_factor_padding() {
        stream.fill(tensor.scale_factors_mut().storage_mut(), 0u8)?;
    }
    Ok(tensor)
}

impl Executor for NormFwdExecutor {
    type Args<'a> = Args<'a>;
    type Output = Output;

    fn execute_with_pool<'a>(
        &self,
        args: Self::Args<'a>,
        stream: &Stream,
        pool: &DeviceMemoryPool,
        cx: &Context,
    ) -> anyhow::Result<Self::Output> {
        let (num_rows, row_size) = self.validate(&args)?;
        let x_shape = args.x.shape();

        // Soundness (all uninit allocations): the kernel writes every element.
        let mut output = Output {
            y_bf16: None,
            y_mxfp8_rowwise: None,
            y_mxfp8_colwise: None,
            r_inv: None,
        };
        match self.kernel.config().output_types {
            OutputTypes::Bf16 => {
                output.y_bf16 =
                    Some(unsafe { Tensor::new_uninit_with_pool(x_shape, BF16, stream, cx, pool)? });
            }
            OutputTypes::Mxfp8Rowwise => {
                output.y_mxfp8_rowwise = Some(alloc_mx_output(
                    [num_rows, row_size],
                    MxDim::Rowwise,
                    stream,
                    pool,
                    cx,
                )?);
            }
            OutputTypes::Mxfp8RowwiseAndColwise => {
                output.y_mxfp8_rowwise = Some(alloc_mx_output(
                    [num_rows, row_size],
                    MxDim::Rowwise,
                    stream,
                    pool,
                    cx,
                )?);
                output.y_mxfp8_colwise = Some(alloc_mx_output(
                    [num_rows, row_size],
                    MxDim::Colwise,
                    stream,
                    pool,
                    cx,
                )?);
            }
            OutputTypes::Nvfp4Rowwise => unreachable!("rejected during validation"),
        }
        if args.write_r_inv {
            let r_inv_shape = x_shape.remove(x_shape.num_dims() - 1);
            output.r_inv =
                Some(unsafe { Tensor::new_uninit_with_pool(r_inv_shape, F32, stream, cx, pool)? });
        }

        let Output {
            y_bf16,
            y_mxfp8_rowwise,
            y_mxfp8_colwise,
            r_inv,
        } = &mut output;

        let y = match (y_bf16, y_mxfp8_rowwise, y_mxfp8_colwise) {
            (Some(y), None, None) => OutputBuffers::Bf16 {
                y: PtrArg::new_mut(y.storage_mut()),
            },
            (None, Some(y_row), None) => {
                let (values, scale_factors) = y_row.values_and_scale_factors_mut();
                OutputBuffers::Mxfp8Rowwise {
                    y_row: PtrArg::new_mut(values.storage_mut()),
                    y_row_sf: PtrArg::new_mut(scale_factors.storage_mut()),
                }
            }
            (None, Some(y_row), Some(y_col)) => {
                let (row_values, row_scale_factors) = y_row.values_and_scale_factors_mut();
                let (col_values, col_scale_factors) = y_col.values_and_scale_factors_mut();
                OutputBuffers::Mxfp8RowwiseAndColwise {
                    y_row: PtrArg::new_mut(row_values.storage_mut()),
                    y_row_sf: PtrArg::new_mut(row_scale_factors.storage_mut()),
                    y_col: PtrArg::new_mut(col_values.storage_mut()),
                    y_col_sf: PtrArg::new_mut(col_scale_factors.storage_mut()),
                }
            }
            _ => unreachable!(),
        };

        // Rows are packed, so the stride equals the row size.
        let row_size_u32 = to_u32("row size", row_size)?;
        let exec_args = ExecArgs {
            x: PtrArg::new(args.x.storage()),
            num_rows: to_u32("row count", num_rows)?,
            row_size: row_size_u32,
            row_stride: row_size_u32,
            r_inv: r_inv
                .as_mut()
                .map(|r_inv| PtrArg::new_mut(r_inv.storage_mut())),
            scale: args
                .scale
                .as_ref()
                .map(|scale| PtrArg::new(scale.storage())),
            bias: args.bias.as_ref().map(|bias| PtrArg::new(bias.storage())),
            y,
        };

        // Soundness: buffer shapes, layouts, and dtypes were validated above
        // and match the kernel's expectations.
        unsafe {
            NormForwardKernelGenerator.execute_kernel(&self.kernel, exec_args, stream)?;
        }

        Ok(output)
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct ConfigWrapper {
    pub inner: norm::Config,
}

impl ResourceKey for ConfigWrapper {
    type Resource = NormFwdExecutor;

    fn create(&self, cx: &Context) -> anyhow::Result<Self::Resource> {
        #[derive(Debug, Serialize, Deserialize)]
        struct DiskKey {
            config: norm::Config,
            arch: Arch,
        }

        #[typetag::serde]
        impl DiskCacheKey for DiskKey {}

        let kernel: norm::forward::BuiltKernel = cx.disk_cache().get_or_create(
            &DiskKey {
                config: self.inner,
                arch: cx.nvvm_arch(),
            },
            || {
                NormForwardKernelGenerator
                    .generate_kernel(self.inner, cx.nvvm_arch())
                    .map_err(From::from)
            },
        )?;

        Ok(NormFwdExecutor { kernel })
    }
}

/// Arguments for [`NormFwdExecutor`]. The norm is taken over the last
/// dimension of `x`, which must be packed row-major.
pub struct Args<'a> {
    pub x: TensorView<'a>,
    /// Shape `[row_size]`. Must be present iff `has_scale` is set in the config.
    pub scale: Option<TensorView<'a>>,
    /// Shape `[row_size]`. Must be present iff `has_bias` is set in the config.
    pub bias: Option<TensorView<'a>>,
    /// Whether to output the per-row reciprocal RMS (or standard deviation).
    pub write_r_inv: bool,
}

/// Outputs of [`NormFwdExecutor`]. Which `y_*` fields are present depends on
/// the config's `output_types`. `y_bf16` has the same shape as `x`; MXFP8 outputs have
/// shape `[num_rows, row_size]`; `r_inv` has the shape of `x` without its last dimension.
pub struct Output {
    pub y_bf16: Option<Tensor>,
    pub y_mxfp8_rowwise: Option<MxTensor>,
    pub y_mxfp8_colwise: Option<MxTensor>,
    pub r_inv: Option<Tensor>,
}
