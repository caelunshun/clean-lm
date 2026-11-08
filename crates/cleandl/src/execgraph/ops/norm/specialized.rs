//! Executor for the norm kernels of `cleandl-specialized-kernels`.

use super::{BF16, NormFwd, NormOutput, NormType};
use crate::{
    context::{Context, ResourceKey},
    execgraph::{Executor, GraphInfo, OpIo, TensorRefMut, TensorTy, sym::Monomial},
};
use anyhow::{anyhow, bail, ensure};
use kernel_generator::{
    Arch, PtrArg,
    kernels::{
        KernelGenerator,
        norm::{
            self, Config, MAX_SUPPORTED_ROW_SIZE, OutputTypes,
            forward::{BuiltKernel, ExecArgs, NormForwardKernelGenerator, OutputBuffers},
        },
    },
};
use serde::Serialize;
use skycuda::stream::Stream;
use std::sync::Arc;

/// Whether the specialized kernel supports the op on the context's device.
pub(super) fn config(op: &NormFwd, cx: &Context, graph: &GraphInfo) -> Option<Config> {
    let dims = graph.dims();
    let arch = cx.nvvm_arch();
    if cx.cuda().compute_capability_major() < 10
        || matches!(arch, Arch::Other(_))
        || op.norm_type != NormType::RmsNorm
    {
        return None;
    }
    let output_types = match op.output {
        NormOutput::Plain(BF16) => OutputTypes::Bf16,
        NormOutput::Plain(_) => return None,
        NormOutput::Mxfp8Rowwise => OutputTypes::Mxfp8Rowwise,
        NormOutput::Mxfp8RowwiseAndColwise => OutputTypes::Mxfp8RowwiseAndColwise,
        NormOutput::Nvfp4Rowwise => OutputTypes::Nvfp4Rowwise,
    };
    if graph.ty(op.x) != TensorTy::Plain(BF16) {
        return None;
    }

    let x = graph.layout(op.x)?;
    let cols = x.shape[1].as_fixed()?;
    // Rows (and the output, which shares the input's row stride) must be 16-byte aligned.
    if cols == 0 || cols > MAX_SUPPORTED_ROW_SIZE || !cols.is_multiple_of(8) {
        return None;
    }
    let packed = (x.strides[1].is_one() || cols == 1)
        && (x.strides[0] == Monomial::fixed(cols) || x.shape[0].is_one());
    if !packed || !x.offset_is_aligned(16, BF16, dims) {
        return None;
    }
    for id in [op.scale, op.bias].into_iter().flatten() {
        let layout = graph.layout(id)?;
        if graph.ty(id) != TensorTy::Plain(BF16)
            || !layout.strides[0].is_one()
            || !layout.offset_is_aligned(16, BF16, dims)
        {
            return None;
        }
    }

    let config = Config {
        norm_type: norm::NormType::RmsNorm,
        has_scale: op.scale.is_some(),
        has_bias: op.bias.is_some(),
        output_types,
        max_row_size: cols,
    };
    NormForwardKernelGenerator::is_supported(config, &arch).then_some(config)
}

#[derive(Serialize)]
struct KernelKey {
    config: Config,
    arch: String,
}

impl ResourceKey for KernelKey {
    type Resource = BuiltKernel;

    fn create(&self, cx: &Context) -> anyhow::Result<BuiltKernel> {
        tracing::debug!(config = ?self.config, "compiling specialized norm_fwd kernel");
        Ok(NormForwardKernelGenerator.generate_kernel(self.config, cx.nvvm_arch())?)
    }
}

pub(super) struct SpecializedExecutor {
    kernel: Arc<BuiltKernel>,
    cols: usize,
    has_scale: bool,
    has_bias: bool,
    save_stats: bool,
    eps: f32,
}

impl SpecializedExecutor {
    pub fn new(op: &NormFwd, config: Config, cx: &Context) -> anyhow::Result<Self> {
        let kernel = cx.get_or_create_resource(KernelKey {
            config,
            arch: format!("{:?}", cx.nvvm_arch()),
        })?;
        Ok(Self {
            kernel,
            cols: config.max_row_size,
            has_scale: op.scale.is_some(),
            has_bias: op.bias.is_some(),
            save_stats: op.save_stats,
            eps: op.eps,
        })
    }
}

impl Executor for SpecializedExecutor {
    fn execute(&self, io: &mut OpIo<'_>, stream: &Stream, _cx: &Context) -> anyhow::Result<()> {
        let cols = self.cols;
        let (inputs, outputs, _) = io.split();
        let x = inputs[0].as_ref();
        let x = x.plain();
        ensure!(
            x.shape().num_dims() == 2 && x.shape()[1] == cols,
            "norm_fwd: expected input rows of {cols} elements, got shape {:?}",
            x.shape()
        );
        let rows = x.shape()[0];
        ensure!(
            rows <= 1 || x.strides()[0] == cols,
            "norm_fwd: input must be packed row-major, got strides {:?}",
            x.strides()
        );
        if rows == 0 {
            return Ok(());
        }
        // The kernel computes element offsets in 32 bits.
        checked_u32(rows * cols, "element count")?;
        let scale = self.has_scale.then(|| inputs[1].as_ref());
        let bias = self
            .has_bias
            .then(|| inputs[1 + self.has_scale as usize].as_ref());

        let (ys, stats) = outputs.split_at_mut(outputs.len() - self.save_stats as usize);
        let r_inv = match stats {
            [TensorRefMut::Plain(t)] => Some(PtrArg::new_mut(t.storage_mut())),
            [] => None,
            _ => bail!("norm_fwd: unexpected stats outputs"),
        };
        let y = match ys {
            [TensorRefMut::Plain(y)] => OutputBuffers::Bf16 {
                y: PtrArg::new_mut(y.storage_mut()),
            },
            [TensorRefMut::Mx(y)] => {
                let (values, scales) = y.values_and_scale_factors_mut();
                match self.kernel.config().output_types {
                    OutputTypes::Mxfp8Rowwise => OutputBuffers::Mxfp8Rowwise {
                        y_row: PtrArg::new_mut(values.storage_mut()),
                        y_row_sf: PtrArg::new_mut(scales.storage_mut()),
                    },
                    OutputTypes::Nvfp4Rowwise => OutputBuffers::Nvfp4Rowwise {
                        y_row: PtrArg::new_mut(values.storage_mut()),
                        y_row_sf: PtrArg::new_mut(scales.storage_mut()),
                    },
                    _ => bail!("norm_fwd: unexpected block-scaled output"),
                }
            }
            [TensorRefMut::Mx(row), TensorRefMut::Mx(col)] => {
                let (row_values, row_scales) = row.values_and_scale_factors_mut();
                let (col_values, col_scales) = col.values_and_scale_factors_mut();
                OutputBuffers::Mxfp8RowwiseAndColwise {
                    y_row: PtrArg::new_mut(row_values.storage_mut()),
                    y_row_sf: PtrArg::new_mut(row_scales.storage_mut()),
                    y_col: PtrArg::new_mut(col_values.storage_mut()),
                    y_col_sf: PtrArg::new_mut(col_scales.storage_mut()),
                }
            }
            _ => bail!("norm_fwd: unexpected outputs"),
        };

        // Soundness: the bound views have the packed layouts the kernel expects, as
        // checked when the op was built and above, and 16-byte aligned storage.
        unsafe {
            NormForwardKernelGenerator.execute_kernel(
                &self.kernel,
                ExecArgs {
                    x: PtrArg::new(x.storage()),
                    num_rows: checked_u32(rows, "row count")?,
                    row_size: cols as u32,
                    row_stride: cols as u32,
                    eps: self.eps,
                    r_inv,
                    scale: scale.as_ref().map(|s| PtrArg::new(s.plain().storage())),
                    bias: bias.as_ref().map(|b| PtrArg::new(b.plain().storage())),
                    y,
                },
                stream,
            )?;
        }
        Ok(())
    }
}

fn checked_u32(v: usize, what: &str) -> anyhow::Result<u32> {
    u32::try_from(v).map_err(|_| anyhow!("norm_fwd: {what} {v} exceeds u32::MAX"))
}
