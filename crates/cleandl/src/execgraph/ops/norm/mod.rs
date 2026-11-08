//! Row normalization (RMSNorm and LayerNorm) with optional affine transform and
//! block-scaled outputs.
//!
//! Uses the specialized NVVM kernels from `cleandl-specialized-kernels` when the device,
//! types and layouts allow it, else a JIT-compiled Tile IR kernel.

mod fallback;
mod specialized;

use crate::{
    execgraph::{
        BuildCx, Executor, GraphInfo, Init, Operand, Operator, OutputSpec, SymTensor, SymTensorId,
        TensorTy, sym::SymShape,
    },
    mx_tensor::{MxDim, MxFormat},
};
use anyhow::ensure;
use serde::{Deserialize, Serialize};
use skytens_core::data_type::{ElementType, FloatType};

const F32: ElementType = ElementType::Float(FloatType::F32);
const BF16: ElementType = ElementType::Float(FloatType::Bf16);

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NormType {
    /// `y = x / sqrt(mean(x^2) + eps)`
    RmsNorm,
    /// `y = (x - mean(x)) / sqrt(var(x) + eps)`
    LayerNorm,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NormOutput {
    /// `bf16` or `f32`.
    Plain(ElementType),
    Mxfp8Rowwise,
    /// Two outputs: rowwise, then colwise MXFP8.
    Mxfp8RowwiseAndColwise,
    /// Without a global scale factor.
    Nvfp4Rowwise,
}

impl NormOutput {
    fn specs(self, shape: &SymShape) -> Vec<OutputSpec> {
        let mx = |format, dim| {
            OutputSpec::new(
                TensorTy::BlockScaled { format, dim },
                shape.clone(),
                Init::Uninit,
            )
        };
        match self {
            NormOutput::Plain(ty) => vec![OutputSpec::new(ty, shape.clone(), Init::Uninit)],
            NormOutput::Mxfp8Rowwise => vec![mx(MxFormat::Mxfp8, MxDim::Rowwise)],
            NormOutput::Mxfp8RowwiseAndColwise => vec![
                mx(MxFormat::Mxfp8, MxDim::Rowwise),
                mx(MxFormat::Mxfp8, MxDim::Colwise),
            ],
            NormOutput::Nvfp4Rowwise => vec![mx(MxFormat::Nvfp4, MxDim::Rowwise)],
        }
    }

    fn num_outputs(self) -> usize {
        match self {
            NormOutput::Mxfp8RowwiseAndColwise => 2,
            _ => 1,
        }
    }
}

#[derive(Clone, Debug)]
pub struct NormFwd {
    pub x: SymTensorId,
    /// `[cols]`, `bf16` or `f32`.
    pub scale: Option<SymTensorId>,
    /// `[cols]`, `bf16` or `f32`.
    pub bias: Option<SymTensorId>,
    pub norm_type: NormType,
    pub eps: f32,
    pub output: NormOutput,
    pub save_stats: bool,
    /// Always use the Tile IR kernel, for testing.
    #[doc(hidden)]
    pub force_fallback: bool,
}

#[derive(Copy, Clone, Debug)]
pub struct NormFwdOpts<'g> {
    pub norm_type: NormType,
    pub scale: Option<SymTensor<'g>>,
    pub bias: Option<SymTensor<'g>>,
    pub eps: f32,
    pub output: NormOutput,
    pub save_stats: bool,
    #[doc(hidden)]
    pub force_fallback: bool,
}

impl Default for NormFwdOpts<'_> {
    fn default() -> Self {
        Self {
            norm_type: NormType::RmsNorm,
            scale: None,
            bias: None,
            eps: 1e-6,
            output: NormOutput::Plain(BF16),
            save_stats: false,
            force_fallback: false,
        }
    }
}

/// Outputs of [`SymTensor::norm_fwd`].
#[derive(Copy, Clone, Debug)]
pub struct NormFwdOutputs<'g> {
    /// Plain or rowwise block-scaled output.
    pub y: SymTensor<'g>,
    /// Colwise output of [`NormOutput::Mxfp8RowwiseAndColwise`].
    pub y_colwise: Option<SymTensor<'g>>,
    /// Per-row mean, for LayerNorm with `save_stats`.
    pub mean: Option<SymTensor<'g>>,
    /// Per-row reciprocal standard deviation (or RMS), with `save_stats`.
    pub r_inv: Option<SymTensor<'g>>,
}

impl<'g> SymTensor<'g> {
    /// Normalizes the rows of this 2D tensor; see [`NormFwd`].
    pub fn norm_fwd(self, opts: NormFwdOpts<'g>) -> anyhow::Result<NormFwdOutputs<'g>> {
        let gb = self.builder();
        ensure!(
            [opts.scale, opts.bias]
                .into_iter()
                .flatten()
                .all(|t| std::ptr::eq(t.builder(), gb)),
            "norm_fwd operands belong to different graphs"
        );
        let op = NormFwd {
            x: self.id(),
            scale: opts.scale.map(SymTensor::id),
            bias: opts.bias.map(SymTensor::id),
            norm_type: opts.norm_type,
            eps: opts.eps,
            output: opts.output,
            save_stats: opts.save_stats,
            force_fallback: opts.force_fallback,
        };
        let mut outputs = gb.add_op(op)?.into_iter();
        let y = outputs.next().unwrap();
        let y_colwise = (opts.output.num_outputs() == 2).then(|| outputs.next().unwrap());
        let mean = (opts.save_stats && opts.norm_type == NormType::LayerNorm)
            .then(|| outputs.next().unwrap());
        let r_inv = opts.save_stats.then(|| outputs.next().unwrap());
        Ok(NormFwdOutputs {
            y,
            y_colwise,
            mean,
            r_inv,
        })
    }
}

impl NormFwd {
    fn validate(&self, graph: &GraphInfo) -> anyhow::Result<()> {
        let dims = graph.dims();
        let x_ty = graph.ty(self.x);
        ensure!(
            x_ty == TensorTy::Plain(BF16) || x_ty == TensorTy::Plain(F32),
            "input must be bf16 or f32, got {x_ty:?}"
        );
        let x_shape = graph.shape(self.x);
        ensure!(
            x_shape.rank() == 2,
            "input must be 2D, got shape {}",
            x_shape.display(dims)
        );
        for (what, id) in [("scale", self.scale), ("bias", self.bias)] {
            let Some(id) = id else { continue };
            let ty = graph.ty(id);
            ensure!(
                ty == TensorTy::Plain(BF16) || ty == TensorTy::Plain(F32),
                "{what} must be bf16 or f32, got {ty:?}"
            );
            let shape = graph.shape(id);
            ensure!(
                shape.rank() == 1 && shape[0] == x_shape[1],
                "{what} must have shape [{}], got {}",
                x_shape[1].display(dims),
                shape.display(dims)
            );
        }
        if let NormOutput::Plain(ty) = self.output {
            ensure!(
                ty == BF16 || ty == F32,
                "plain output must be bf16 or f32, got {ty:?}"
            );
        }
        ensure!(
            self.eps.is_finite() && self.eps >= 0.0,
            "eps must be finite and non-negative, got {}",
            self.eps
        );
        Ok(())
    }

    fn x_shape<'a>(&self, graph: &'a GraphInfo) -> &'a SymShape {
        graph.shape(self.x)
    }

    /// The specialized kernel configuration, if it applies on the context's device.
    #[doc(hidden)]
    pub fn specialized_config(
        &self,
        cx: &crate::context::Context,
        graph: &GraphInfo,
    ) -> Option<kernel_generator::kernels::norm::Config> {
        if self.force_fallback {
            return None;
        }
        specialized::config(self, cx, graph)
    }
}

impl Operator for NormFwd {
    fn name(&self) -> &str {
        "norm_fwd"
    }

    fn operands(&self) -> Vec<Operand> {
        [Some(self.x), self.scale, self.bias]
            .into_iter()
            .flatten()
            .map(Operand::read)
            .collect()
    }

    fn infer_outputs(&self, graph: &GraphInfo) -> anyhow::Result<Vec<OutputSpec>> {
        self.validate(graph)?;
        let shape = self.x_shape(graph);
        let mut specs = self.output.specs(shape);
        if self.save_stats {
            let rows: SymShape = [shape[0].clone()].into_iter().collect();
            let count = match self.norm_type {
                NormType::RmsNorm => 1,
                NormType::LayerNorm => 2,
            };
            for _ in 0..count {
                specs.push(OutputSpec::new(F32, rows.clone(), Init::Uninit));
            }
        }
        Ok(specs)
    }

    fn build(&self, bcx: &BuildCx<'_>) -> anyhow::Result<Box<dyn Executor>> {
        if let Some(config) = self.specialized_config(bcx.cx, bcx.graph) {
            tracing::debug!(?config, "norm_fwd: using the specialized kernel");
            return Ok(Box::new(specialized::SpecializedExecutor::new(
                self, config, bcx.cx,
            )?));
        }
        fallback::build(self, bcx)
    }
}
