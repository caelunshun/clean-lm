use std::sync::Arc;

use serde::Serialize;
use skycuda::stream::Stream;
use skycuda_cudnn::{DataType, MatmulAttributes, PointwiseAttributes, PointwiseMode, TensorId};
use skytens_core::data_type::{ElementType, FloatType};

use crate::{
    context::{Context, ResourceKey},
    cudnn::{CudnnGraphBuilder, ExecutableCudnnGraph},
    operators::matmul::{MatmulConfig, MatmulExecutor, MatrixFormat, ResidualConfig},
    tensor::{Tensor, TensorView, VirtTensorView},
};

/// SwiGLU FFN forward and backward pass executor.
/// Forward: regular matmul for up-projection and down-projection; fused epilogue swish+hadamard for
/// gate projection. Backward: regular matmul for up-projection dgrad,
/// fused total-gradient add epilogue for gate-projection dgrad,
/// fused swish+hadamard backward epilogue for down-projection dgrad.
#[derive(Debug)]
pub struct SwiGluFfnExecutor {
    config: SwiGluConfig,
    cx: Context,
    fwd_up_proj: Arc<MatmulExecutor>,
    fwd_down_proj: Arc<MatmulExecutor>,

    fwd_fused_gate_graph: ExecutableCudnnGraph,
    fwd_fused_gate_data: TensorId,
    fwd_fused_gate_up: TensorId,
    fwd_fused_gate_weight: TensorId,
    fwd_fused_gate_out_gate: TensorId,
    fwd_fused_gate_out_activation: TensorId,

    bwd_down_proj_wgrad: Arc<MatmulExecutor>,
    bwd_up_gate_proj_wgrad: Arc<MatmulExecutor>,
    bwd_up_proj_dgrad: Arc<MatmulExecutor>,

    bwd_gate_proj_dgrad_graph: ExecutableCudnnGraph,
    bwd_gate_proj_dgrad_weight: TensorId,
    bwd_gate_proj_dgrad_dgate: TensorId,
    bwd_gate_proj_dgrad_dup_in: TensorId,
    bwd_gate_proj_dgrad_out_dffn: TensorId,

    bwd_down_proj_dgrad_graph: ExecutableCudnnGraph,
    bwd_down_proj_dgrad_weight: TensorId,
    bwd_down_proj_dgrad_dy: TensorId,
    bwd_down_proj_dgrad_gate: TensorId,
    bwd_down_proj_dgrad_up: TensorId,
    bwd_down_proj_dgrad_out_dgate: TensorId,
    bwd_down_proj_dgrad_out_dup: TensorId,
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
        let mut out_gate = Tensor::new_zeroed(up.shape(), up.element_type(), &self.cx)?;
        let mut out_activation = Tensor::new_zeroed(up.shape(), up.element_type(), &self.cx)?;

        self.fwd_fused_gate_graph.execute(
            &[
                (self.fwd_fused_gate_data, &data.unsqueezed()),
                (self.fwd_fused_gate_up, &up.unsqueezed()),
                (self.fwd_fused_gate_weight, &weight.unsqueezed()),
            ],
            &mut [
                (self.fwd_fused_gate_out_gate, out_gate.unsqueezed_mut()),
                (
                    self.fwd_fused_gate_out_activation,
                    out_activation.unsqueezed_mut(),
                ),
            ],
            stream,
        )?;

        Ok(GateProjForwardOutput {
            gate: out_gate,
            activation: out_activation,
        })
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
        let mut out_dgate = Tensor::new_zeroed(gate.shape(), gate.element_type(), &self.cx)?;
        let mut out_dup = Tensor::new_zeroed(up.shape(), up.element_type(), &self.cx)?;

        self.bwd_down_proj_dgrad_graph.execute(
            &[
                (
                    self.bwd_down_proj_dgrad_weight,
                    &weight.transposed(0, 1).unsqueezed(),
                ),
                (self.bwd_down_proj_dgrad_dy, &dy.unsqueezed()),
                (self.bwd_down_proj_dgrad_gate, &gate.unsqueezed()),
                (self.bwd_down_proj_dgrad_up, &up.unsqueezed()),
            ],
            &mut [
                (
                    self.bwd_down_proj_dgrad_out_dgate,
                    out_dgate.unsqueezed_mut(),
                ),
                (self.bwd_down_proj_dgrad_out_dup, out_dup.unsqueezed_mut()),
            ],
            stream,
        )?;

        Ok(DownProjDgradOutput {
            dgate: out_dgate,
            dup: out_dup,
        })
    }

    /// Returns the down-projection weight gradient
    /// `[intermediate_features, out_features]`.
    pub fn backward_down_proj_wgrad(
        &self,
        activation: &TensorView<'_>,
        dy: &TensorView<'_>,
        stream: &Stream,
    ) -> anyhow::Result<Tensor> {
        self.bwd_down_proj_wgrad
            .matmul(&activation.transposed(0, 1), dy, None, stream)
    }

    /// Returns an up- or gate-projection weight gradient
    /// `[in_features, intermediate_features]`.
    pub fn backward_up_gate_proj_wgrad(
        &self,
        data: &TensorView<'_>,
        dout: &TensorView<'_>,
        stream: &Stream,
    ) -> anyhow::Result<Tensor> {
        self.bwd_up_gate_proj_wgrad
            .matmul(&data.transposed(0, 1), dout, None, stream)
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
        let mut out_dffn = Tensor::new_zeroed(ddata_up.shape(), ddata_up.element_type(), &self.cx)?;

        self.bwd_gate_proj_dgrad_graph.execute(
            &[
                (self.bwd_gate_proj_dgrad_dgate, &dout.unsqueezed()),
                (self.bwd_gate_proj_dgrad_dup_in, &ddata_up.unsqueezed()),
                (
                    self.bwd_gate_proj_dgrad_weight,
                    &weight.transposed(0, 1).unsqueezed(),
                ),
            ],
            &mut [(self.bwd_gate_proj_dgrad_out_dffn, out_dffn.unsqueezed_mut())],
            stream,
        )?;
        Ok(out_dffn)
    }
}

#[derive(Debug, Copy, Clone, Serialize)]
pub struct SwiGluConfig {
    pub estimated_batch_size: usize,
    pub in_features: usize,
    pub intermediate_features: usize,
    pub out_features: usize,
}

fn matmul_attrs() -> MatmulAttributes {
    MatmulAttributes::new().compute_data_type(DataType::Float)
}

fn pointwise_attrs(mode: PointwiseMode) -> PointwiseAttributes {
    PointwiseAttributes::new(mode).compute_data_type(DataType::Float)
}

impl ResourceKey for SwiGluConfig {
    type Resource = SwiGluFfnExecutor;

    fn create(&self, cx: &Context) -> anyhow::Result<Self::Resource> {
        let bf16 = ElementType::Float(FloatType::Bf16);
        let data_layout =
            VirtTensorView::new([1, self.estimated_batch_size, self.in_features], bf16);
        let intermediate_layout = VirtTensorView::new(
            [1, self.estimated_batch_size, self.intermediate_features],
            bf16,
        );
        let out_layout =
            VirtTensorView::new([1, self.estimated_batch_size, self.out_features], bf16);
        let up_gate_weight_layout =
            VirtTensorView::new([1, self.in_features, self.intermediate_features], bf16);
        let down_weight_layout =
            VirtTensorView::new([1, self.intermediate_features, self.out_features], bf16);

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

        let fwd_fused_gate_graph = CudnnGraphBuilder::new(cx)?;
        fwd_fused_gate_graph
            .graph()
            .set_override_shape_enabled(true);
        let fwd_fused_gate_data = fwd_fused_gate_graph.bind_tensor(&data_layout);
        let fwd_fused_gate_up = fwd_fused_gate_graph.bind_tensor(&intermediate_layout);
        let fwd_fused_gate_weight = fwd_fused_gate_graph.bind_tensor(&up_gate_weight_layout);

        let gate = fwd_fused_gate_graph.graph().matmul(
            fwd_fused_gate_data,
            fwd_fused_gate_weight,
            &matmul_attrs(),
        )?;
        let fwd_fused_gate_out_gate = fwd_fused_gate_graph.mark_output(gate, &intermediate_layout);
        let tmp = fwd_fused_gate_graph
            .graph()
            .pointwise_unary(gate, &pointwise_attrs(PointwiseMode::SwishFwd))?;
        let activation = fwd_fused_gate_graph.graph().pointwise_binary(
            tmp,
            fwd_fused_gate_up,
            &pointwise_attrs(PointwiseMode::Mul),
        )?;
        let fwd_fused_gate_out_activation =
            fwd_fused_gate_graph.mark_output(activation, &intermediate_layout);
        let (fwd_fused_gate_data, fwd_fused_gate_up, fwd_fused_gate_weight) = (
            fwd_fused_gate_data.uid(),
            fwd_fused_gate_up.uid(),
            fwd_fused_gate_weight.uid(),
        );
        let fwd_fused_gate_graph = fwd_fused_gate_graph.build()?;

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
                out_element_type: ElementType::Float(FloatType::Bf16),
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
                out_element_type: ElementType::Float(FloatType::Bf16),
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

        let bwd_gate_proj_dgrad_graph = CudnnGraphBuilder::new(cx)?;
        bwd_gate_proj_dgrad_graph
            .graph()
            .set_override_shape_enabled(true);
        let bwd_gate_proj_dgrad_dgate = bwd_gate_proj_dgrad_graph.bind_tensor(&intermediate_layout);
        let bwd_gate_proj_dgrad_weight =
            bwd_gate_proj_dgrad_graph.bind_tensor(&up_gate_weight_layout.transposed(1, 2));
        let bwd_gate_proj_dgrad_dup_in = bwd_gate_proj_dgrad_graph.bind_tensor(&data_layout);

        let tmp_dgate_in = bwd_gate_proj_dgrad_graph.graph().matmul(
            bwd_gate_proj_dgrad_dgate,
            bwd_gate_proj_dgrad_weight,
            &matmul_attrs(),
        )?;
        let dffn = bwd_gate_proj_dgrad_graph.graph().pointwise_binary(
            tmp_dgate_in,
            bwd_gate_proj_dgrad_dup_in,
            &pointwise_attrs(PointwiseMode::Add),
        )?;
        let bwd_gate_proj_dgrad_out_dffn =
            bwd_gate_proj_dgrad_graph.mark_output(dffn, &data_layout);
        let (bwd_gate_proj_dgrad_dgate, bwd_gate_proj_dgrad_weight, bwd_gate_proj_dgrad_dup_in) = (
            bwd_gate_proj_dgrad_dgate.uid(),
            bwd_gate_proj_dgrad_weight.uid(),
            bwd_gate_proj_dgrad_dup_in.uid(),
        );
        let bwd_gate_proj_dgrad_graph = bwd_gate_proj_dgrad_graph.build()?;

        let bwd_down_proj_dgrad_graph = CudnnGraphBuilder::new(cx)?;
        bwd_down_proj_dgrad_graph
            .graph()
            .set_override_shape_enabled(true);

        let bwd_down_proj_dgrad_weight =
            bwd_down_proj_dgrad_graph.bind_tensor(&down_weight_layout.transposed(1, 2));
        let bwd_down_proj_dgrad_dy = bwd_down_proj_dgrad_graph.bind_tensor(&out_layout);
        let bwd_down_proj_dgrad_gate = bwd_down_proj_dgrad_graph.bind_tensor(&intermediate_layout);
        let bwd_down_proj_dgrad_up = bwd_down_proj_dgrad_graph.bind_tensor(&intermediate_layout);

        let tmp_grad = bwd_down_proj_dgrad_graph.graph().matmul(
            bwd_down_proj_dgrad_dy,
            bwd_down_proj_dgrad_weight,
            &matmul_attrs(),
        )?;

        // Recompute swish forward
        let tmp_gate_swish = bwd_down_proj_dgrad_graph.graph().pointwise_unary(
            bwd_down_proj_dgrad_gate,
            &pointwise_attrs(PointwiseMode::SwishFwd),
        )?;

        // Apply product rule
        // (1) dup
        let dup = bwd_down_proj_dgrad_graph.graph().pointwise_binary(
            tmp_gate_swish,
            tmp_grad,
            &pointwise_attrs(PointwiseMode::Mul),
        )?;
        let bwd_down_proj_dgrad_out_dup =
            bwd_down_proj_dgrad_graph.mark_output(dup, &intermediate_layout);
        // (2) dgate, requiring dswish operation
        let tmp_dswish = bwd_down_proj_dgrad_graph.graph().pointwise_binary(
            tmp_grad,
            bwd_down_proj_dgrad_up,
            &pointwise_attrs(PointwiseMode::Mul),
        )?;
        let dgate = bwd_down_proj_dgrad_graph.graph().pointwise_binary(
            tmp_dswish,
            bwd_down_proj_dgrad_gate,
            &pointwise_attrs(PointwiseMode::SwishBwd),
        )?;
        let bwd_down_proj_dgrad_out_dgate =
            bwd_down_proj_dgrad_graph.mark_output(dgate, &intermediate_layout);
        let (
            bwd_down_proj_dgrad_weight,
            bwd_down_proj_dgrad_dy,
            bwd_down_proj_dgrad_gate,
            bwd_down_proj_dgrad_up,
        ) = (
            bwd_down_proj_dgrad_weight.uid(),
            bwd_down_proj_dgrad_dy.uid(),
            bwd_down_proj_dgrad_gate.uid(),
            bwd_down_proj_dgrad_up.uid(),
        );
        let bwd_down_proj_dgrad_graph = bwd_down_proj_dgrad_graph.build()?;

        Ok(SwiGluFfnExecutor {
            config: *self,
            cx: cx.clone(),
            fwd_up_proj,
            fwd_down_proj,
            fwd_fused_gate_graph,
            fwd_fused_gate_data,
            fwd_fused_gate_up,
            fwd_fused_gate_weight,
            fwd_fused_gate_out_gate,
            fwd_fused_gate_out_activation,
            bwd_down_proj_wgrad,
            bwd_up_gate_proj_wgrad,

            bwd_down_proj_dgrad_graph,
            bwd_down_proj_dgrad_weight,
            bwd_down_proj_dgrad_dy,
            bwd_down_proj_dgrad_gate,
            bwd_down_proj_dgrad_up,
            bwd_down_proj_dgrad_out_dgate,
            bwd_down_proj_dgrad_out_dup,
            bwd_up_proj_dgrad,
            bwd_gate_proj_dgrad_graph,
            bwd_gate_proj_dgrad_weight,
            bwd_gate_proj_dgrad_dgate,
            bwd_gate_proj_dgrad_dup_in,
            bwd_gate_proj_dgrad_out_dffn,
        })
    }
}
