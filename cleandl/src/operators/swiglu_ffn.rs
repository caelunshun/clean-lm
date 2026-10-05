use crate::{
    context::{Context, ResourceKey},
    operators::{
        matmul::{MatmulConfig, MatmulExecutor, MatrixFormat, ResidualConfig},
        sky_matmul::{
            SkyMatmulConfig, SkyMatmulExecutor, SwiGluDownTrainDgradArgs,
            SwiGluDownTrainDgradEpilogue, SwiGluDownTrainDgradOutputs, SwiGluGateTrainFwdEpilogue,
            SwiGluGateTrainFwdOutputs,
        },
    },
    tensor::{Tensor, TensorView, TensorViewMut},
};
use serde::Serialize;
use skycuda::stream::Stream;
use skytens_core::data_type::{ElementType, FloatType};
use std::sync::Arc;

/// SwiGLU FFN forward and backward pass executor.
/// Forward: regular matmul for up-projection and down-projection; fused epilogue swish+hadamard for
/// gate projection. Backward: regular matmul for up-projection dgrad,
/// fused total-gradient add epilogue for gate-projection dgrad,
/// fused swish+hadamard backward epilogue for down-projection dgrad.
#[derive(Debug)]
pub struct SwiGluFfnExecutor {
    config: SwiGluConfig,
    fwd_up_proj: Arc<MatmulExecutor>,
    fwd_down_proj: Arc<MatmulExecutor>,

    fwd_fused_gate: Arc<SkyMatmulExecutor<SwiGluGateTrainFwdEpilogue>>,

    bwd_down_proj_wgrad: Arc<MatmulExecutor>,
    bwd_up_gate_proj_wgrad: Arc<MatmulExecutor>,
    bwd_up_proj_dgrad: Arc<MatmulExecutor>,
    bwd_gate_proj_dgrad: Arc<MatmulExecutor>,
    bwd_down_proj_dgrad: Arc<SkyMatmulExecutor<SwiGluDownTrainDgradEpilogue>>,
}

/// Outputs of [`SwiGluFfnExecutor::forward_gate_proj`].
#[derive(Debug)]
pub struct GateProjForwardOutput {
    /// Gate projection, before the activation.
    pub gate: Tensor,
    /// `swish(gate) * up`, the input to the down projection.
    pub activation: Tensor,
}

/// Outputs of [`SwiGluFfnExecutor::backward_down_proj_dgrad`].
#[derive(Debug)]
pub struct DownProjDgradOutput {
    pub dgate: Tensor,
    pub dup: Tensor,
}

impl SwiGluFfnExecutor {
    pub fn get(config: SwiGluConfig, cx: &Context) -> anyhow::Result<Arc<Self>> {
        cx.get_or_create_resource(config)
    }

    pub fn config(&self) -> SwiGluConfig {
        self.config
    }

    /// Returns the up projection `[batch_size, intermediate_features]`.
    pub fn forward_up_proj(
        &self,
        data: &TensorView<'_>,
        weight: &TensorView<'_>,
        stream: &Stream,
    ) -> anyhow::Result<Tensor> {
        self.fwd_up_proj.matmul(data, weight, None, stream)
    }

    /// Gate projection fused with swish + hadamard product.
    pub fn forward_gate_proj(
        &self,
        data: &TensorView<'_>,
        up: &TensorView<'_>,
        weight: &TensorView<'_>,
        stream: &Stream,
    ) -> anyhow::Result<GateProjForwardOutput> {
        let SwiGluGateTrainFwdOutputs { gate, activation } =
            self.fwd_fused_gate.execute(data, weight, up, stream)?;
        Ok(GateProjForwardOutput { gate, activation })
    }

    /// Returns the FFN output `[batch_size, out_features]`, with the residual added.
    pub fn forward_down_proj(
        &self,
        activation: &TensorView<'_>,
        weight: &TensorView<'_>,
        residual: &TensorView<'_>,
        stream: &Stream,
    ) -> anyhow::Result<Tensor> {
        self.fwd_down_proj
            .matmul(activation, weight, Some(residual), stream)
    }

    pub fn backward_down_proj_dgrad(
        &self,
        weight: &TensorView<'_>,
        dy: &TensorView<'_>,
        gate: &TensorView<'_>,
        up: &TensorView<'_>,
        stream: &Stream,
    ) -> anyhow::Result<DownProjDgradOutput> {
        let SwiGluDownTrainDgradOutputs { dgate, dx } = self.bwd_down_proj_dgrad.execute(
            dy,
            &weight.transposed(0, 1),
            SwiGluDownTrainDgradArgs { gate, x: up },
            stream,
        )?;
        Ok(DownProjDgradOutput { dgate, dup: dx })
    }

    pub fn backward_down_proj_wgrad_accumulate(
        &self,
        activation: &TensorView<'_>,
        dy: &TensorView<'_>,
        gradient: &mut TensorViewMut<'_>,
        stream: &Stream,
    ) -> anyhow::Result<()> {
        self.bwd_down_proj_wgrad.matmul_accumulate(
            &activation.transposed(0, 1),
            dy,
            gradient,
            stream,
        )
    }

    pub fn backward_up_gate_proj_wgrad_accumulate(
        &self,
        data: &TensorView<'_>,
        dout: &TensorView<'_>,
        gradient: &mut TensorViewMut<'_>,
        stream: &Stream,
    ) -> anyhow::Result<()> {
        self.bwd_up_gate_proj_wgrad.matmul_accumulate(
            &data.transposed(0, 1),
            dout,
            gradient,
            stream,
        )
    }

    /// Returns the up-projection data gradient `[batch_size, in_features]`.
    pub fn backward_up_proj_dgrad(
        &self,
        dout: &TensorView<'_>,
        weight: &TensorView<'_>,
        stream: &Stream,
    ) -> anyhow::Result<Tensor> {
        self.bwd_up_proj_dgrad
            .matmul(dout, &weight.transposed(0, 1), None, stream)
    }

    /// Returns the total FFN data gradient `[batch_size, in_features]`, summing the
    /// gate-projection dgrad with `ddata_up`.
    pub fn backward_gate_proj_dgrad(
        &self,
        dout: &TensorView<'_>,
        weight: &TensorView<'_>,
        ddata_up: &TensorView<'_>,
        stream: &Stream,
    ) -> anyhow::Result<Tensor> {
        self.bwd_gate_proj_dgrad
            .matmul(dout, &weight.transposed(0, 1), Some(ddata_up), stream)
    }
}

#[derive(Debug, Copy, Clone, Serialize)]
pub struct SwiGluConfig {
    pub estimated_batch_size: usize,
    pub in_features: usize,
    pub intermediate_features: usize,
    pub out_features: usize,
}

impl ResourceKey for SwiGluConfig {
    type Resource = SwiGluFfnExecutor;

    fn create(&self, cx: &Context) -> anyhow::Result<Self::Resource> {
        let batch_size = u32::try_from(self.estimated_batch_size)?;
        let in_features = u32::try_from(self.in_features)?;
        let intermediate_features = u32::try_from(self.intermediate_features)?;
        let out_features = u32::try_from(self.out_features)?;

        let fwd_up_proj = MatmulExecutor::get(
            MatmulConfig {
                m: self.estimated_batch_size,
                n: self.intermediate_features,
                k: self.in_features,
                residual: None,
                a_format: MatrixFormat::RowMajor,
                b_format: MatrixFormat::RowMajor,
                out_format: MatrixFormat::RowMajor,
                allow_shape_override: true,
                ab_element_type: ElementType::Float(FloatType::Bf16),
                out_element_type: ElementType::Float(FloatType::Bf16),
            },
            cx,
        )?;
        let fwd_down_proj = MatmulExecutor::get(
            MatmulConfig {
                m: self.estimated_batch_size,
                n: self.out_features,
                k: self.intermediate_features,
                residual: Some(ResidualConfig {
                    format: MatrixFormat::RowMajor,
                    element_type: ElementType::Float(FloatType::Bf16),
                }),
                a_format: MatrixFormat::RowMajor,
                b_format: MatrixFormat::RowMajor,
                out_format: MatrixFormat::RowMajor,
                allow_shape_override: true,
                ab_element_type: ElementType::Float(FloatType::Bf16),
                out_element_type: ElementType::Float(FloatType::Bf16),
            },
            cx,
        )?;

        let fwd_fused_gate = SkyMatmulExecutor::get(
            SkyMatmulConfig {
                tune_for_m: batch_size,
                tune_for_n: intermediate_features,
                tune_for_k: in_features,
                format_a: MatrixFormat::RowMajor,
                format_b: MatrixFormat::RowMajor,
                epilogue: SwiGluGateTrainFwdEpilogue,
            },
            cx,
        )?;

        let bwd_down_proj_wgrad = MatmulExecutor::get(
            MatmulConfig {
                m: self.intermediate_features,
                n: self.out_features,
                k: self.estimated_batch_size,
                residual: None,
                a_format: MatrixFormat::ColumnMajor, // transposed
                b_format: MatrixFormat::RowMajor,
                out_format: MatrixFormat::RowMajor,
                allow_shape_override: true,
                ab_element_type: ElementType::Float(FloatType::Bf16),
                out_element_type: ElementType::Float(FloatType::F32),
            },
            cx,
        )?;

        let bwd_up_gate_proj_wgrad = MatmulExecutor::get(
            MatmulConfig {
                m: self.in_features,
                n: self.intermediate_features,
                k: self.estimated_batch_size,
                residual: None,
                a_format: MatrixFormat::ColumnMajor, // transposed
                b_format: MatrixFormat::RowMajor,
                out_format: MatrixFormat::RowMajor,
                allow_shape_override: true,
                ab_element_type: ElementType::Float(FloatType::Bf16),
                out_element_type: ElementType::Float(FloatType::F32),
            },
            cx,
        )?;
        let bwd_up_proj_dgrad = MatmulExecutor::get(
            MatmulConfig {
                m: self.estimated_batch_size,
                n: self.in_features,
                k: self.intermediate_features,
                residual: None,
                a_format: MatrixFormat::RowMajor,
                b_format: MatrixFormat::ColumnMajor, // transposed
                out_format: MatrixFormat::RowMajor,
                allow_shape_override: true,
                ab_element_type: ElementType::Float(FloatType::Bf16),
                out_element_type: ElementType::Float(FloatType::Bf16),
            },
            cx,
        )?;

        let bwd_gate_proj_dgrad = MatmulExecutor::get(
            MatmulConfig {
                m: self.estimated_batch_size,
                n: self.in_features,
                k: self.intermediate_features,
                residual: Some(ResidualConfig {
                    format: MatrixFormat::RowMajor,
                    element_type: ElementType::Float(FloatType::Bf16),
                }),
                a_format: MatrixFormat::RowMajor,
                b_format: MatrixFormat::ColumnMajor, // transposed
                out_format: MatrixFormat::RowMajor,
                allow_shape_override: true,
                ab_element_type: ElementType::Float(FloatType::Bf16),
                out_element_type: ElementType::Float(FloatType::Bf16),
            },
            cx,
        )?;

        let bwd_down_proj_dgrad = SkyMatmulExecutor::get(
            SkyMatmulConfig {
                tune_for_m: batch_size,
                tune_for_n: intermediate_features,
                tune_for_k: out_features,
                format_a: MatrixFormat::RowMajor,
                format_b: MatrixFormat::ColumnMajor, // transposed
                epilogue: SwiGluDownTrainDgradEpilogue,
            },
            cx,
        )?;

        Ok(SwiGluFfnExecutor {
            config: *self,
            fwd_up_proj,
            fwd_down_proj,
            fwd_fused_gate,
            bwd_down_proj_wgrad,
            bwd_up_gate_proj_wgrad,
            bwd_up_proj_dgrad,
            bwd_gate_proj_dgrad,
            bwd_down_proj_dgrad,
        })
    }
}
