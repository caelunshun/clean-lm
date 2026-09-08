use std::sync::Arc;

use serde::Serialize;
use singe_cudnn::{
    backend::{pointwise::PointwiseMode, tensor::TensorId},
    data_type::DataType,
    frontend::operation::PointwiseOperation,
    math::NanPropagation,
};
use skytens_core::data_type::{ElementType, FloatType};

use crate::{
    context::{Context, ResourceKey},
    cudnn::{CudnnGraphBuilder, ExecutableCudnnGraph},
    operators::matmul::{MatmulConfig, MatmulExecutor, MatrixFormat},
    tensor::VirtTensorView,
};

/// SwiGLU FFN forward and backward pass executor.
/// Forward: regular matmul for up-projection and down-projection; fused epilogue swish+hadamard for
/// gate projection. Backward: regular matmul for up-projection and gate-projection dgrads,
/// fused swish+hadamard backward epilogue for down-projection dgrad.
#[derive(Debug)]
pub struct SwiGluFfnExecutor {
    config: SwiGluConfig,
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
    bwd_up_gate_proj_dgrad: Arc<MatmulExecutor>,

    bwd_down_proj_dgrad_graph: ExecutableCudnnGraph,
    bwd_down_proj_dgrad_weight: TensorId,
    bwd_down_proj_dgrad_dy: TensorId,
    bwd_down_proj_dgrad_gate: TensorId,
    bwd_down_proj_dgrad_up: TensorId,
    bwd_down_proj_dgrad_out_dgate: TensorId,
    bwd_down_proj_dgrad_out_dup: TensorId,
}

impl SwiGluFfnExecutor {
    pub fn get(config: SwiGluConfig, cx: &Context) -> anyhow::Result<Arc<Self>> {
        cx.get_or_create_resource(config)
    }
}

#[derive(Debug, Copy, Clone, Serialize)]
pub struct SwiGluConfig {
    pub batch_size: usize,
    pub in_features: usize,
    pub intermediate_features: usize,
    pub out_features: usize,
}

impl ResourceKey for SwiGluConfig {
    type Resource = SwiGluFfnExecutor;

    fn create(&self, cx: &Context) -> anyhow::Result<Self::Resource> {
        let fwd_up_proj = MatmulExecutor::get(
            MatmulConfig {
                m: self.batch_size,
                n: self.intermediate_features,
                k: self.in_features,
                a_format: MatrixFormat::RowMajor,
                b_format: MatrixFormat::RowMajor,
                out_format: MatrixFormat::RowMajor,
            },
            cx,
        )?;
        let fwd_down_proj = MatmulExecutor::get(
            MatmulConfig {
                m: self.batch_size,
                n: self.out_features,
                k: self.intermediate_features,
                a_format: MatrixFormat::RowMajor,
                b_format: MatrixFormat::RowMajor,
                out_format: MatrixFormat::RowMajor,
            },
            cx,
        )?;

        let mut fwd_fused_gate_graph = CudnnGraphBuilder::new(cx);
        let fwd_fused_gate_data = fwd_fused_gate_graph.bind_tensor(&VirtTensorView::new(
            [self.batch_size, self.in_features],
            ElementType::Float(FloatType::Bf16),
        ));
        let fwd_fused_gate_up = fwd_fused_gate_graph.bind_tensor(&VirtTensorView::new(
            [self.batch_size, self.intermediate_features],
            ElementType::Float(FloatType::Bf16),
        ));
        let fwd_fused_gate_weight = fwd_fused_gate_graph.bind_tensor(&VirtTensorView::new(
            [self.in_features, self.intermediate_features],
            ElementType::Float(FloatType::Bf16),
        ));

        let fwd_fused_gate_out_gate = fwd_fused_gate_graph.bind_tensor(&VirtTensorView::new(
            [self.batch_size, self.intermediate_features],
            ElementType::Float(FloatType::Bf16),
        ));

        let tmp = fwd_fused_gate_graph.bind_tensor(&VirtTensorView::new(
            [self.batch_size, self.intermediate_features],
            ElementType::Float(FloatType::Bf16),
        ));
        fwd_fused_gate_graph.graph_mut().mark_as_virtual(tmp)?;

        let fwd_fused_gate_out_activation = fwd_fused_gate_graph.bind_tensor(&VirtTensorView::new(
            [self.batch_size, self.intermediate_features],
            ElementType::Float(FloatType::Bf16),
        ));

        fwd_fused_gate_graph.graph_mut().matmul(
            fwd_fused_gate_data,
            fwd_fused_gate_weight,
            fwd_fused_gate_out_gate,
            DataType::F32,
        )?;
        fwd_fused_gate_graph
            .graph_mut()
            .pointwise(PointwiseOperation::Unary {
                mode: PointwiseMode::SwishFwd,
                input: fwd_fused_gate_out_gate,
                output: tmp,
                compute_type: DataType::F32,
                nan_propagation: NanPropagation::Propagate,
                alpha1: 1.0,
                axis: None,
            });
        fwd_fused_gate_graph
            .graph_mut()
            .pointwise(PointwiseOperation::Binary {
                mode: PointwiseMode::Mul,
                lhs: tmp,
                rhs: fwd_fused_gate_up,
                output: fwd_fused_gate_out_activation,
                compute_type: DataType::F32,
                nan_propagation: NanPropagation::Propagate,
                alpha1: 1.0,
                alpha2: 1.0,
            });
        let fwd_fused_gate_graph = fwd_fused_gate_graph.build()?;

        let bwd_down_proj_wgrad = MatmulExecutor::get(
            MatmulConfig {
                m: self.intermediate_features,
                n: self.out_features,
                k: self.batch_size,
                a_format: MatrixFormat::ColumnMajor, // transposed
                b_format: MatrixFormat::RowMajor,
                out_format: MatrixFormat::RowMajor,
            },
            cx,
        )?;

        let bwd_up_gate_proj_wgrad = MatmulExecutor::get(
            MatmulConfig {
                m: self.in_features,
                n: self.intermediate_features,
                k: self.batch_size,
                a_format: MatrixFormat::ColumnMajor, // transposed
                b_format: MatrixFormat::RowMajor,
                out_format: MatrixFormat::RowMajor,
            },
            cx,
        )?;
        let bwd_up_gate_proj_dgrad = MatmulExecutor::get(
            MatmulConfig {
                m: self.batch_size,
                n: self.in_features,
                k: self.intermediate_features,
                a_format: MatrixFormat::RowMajor,
                b_format: MatrixFormat::ColumnMajor, // transposed
                out_format: MatrixFormat::RowMajor,
            },
            cx,
        )?;

        let mut bwd_down_proj_dgrad_graph = CudnnGraphBuilder::new(cx);

        let bwd_down_proj_dgrad_weight = bwd_down_proj_dgrad_graph.bind_tensor(
            &VirtTensorView::new(
                [self.intermediate_features, self.out_features],
                ElementType::Float(FloatType::Bf16),
            )
            .transposed(0, 1),
        );
        let bwd_down_proj_dgrad_dy = bwd_down_proj_dgrad_graph.bind_tensor(&VirtTensorView::new(
            [self.batch_size, self.out_features],
            ElementType::Float(FloatType::Bf16),
        ));
        let bwd_down_proj_dgrad_gate = bwd_down_proj_dgrad_graph.bind_tensor(&VirtTensorView::new(
            [self.batch_size, self.intermediate_features],
            ElementType::Float(FloatType::Bf16),
        ));
        let bwd_down_proj_dgrad_up = bwd_down_proj_dgrad_graph.bind_tensor(&VirtTensorView::new(
            [self.batch_size, self.intermediate_features],
            ElementType::Float(FloatType::Bf16),
        ));
        let bwd_down_proj_dgrad_out_dgate =
            bwd_down_proj_dgrad_graph.bind_tensor(&VirtTensorView::new(
                [self.batch_size, self.intermediate_features],
                ElementType::Float(FloatType::Bf16),
            ));
        let bwd_down_proj_dgrad_out_dup =
            bwd_down_proj_dgrad_graph.bind_tensor(&VirtTensorView::new(
                [self.batch_size, self.intermediate_features],
                ElementType::Float(FloatType::Bf16),
            ));

        let tmp_grad = bwd_down_proj_dgrad_graph
            .graph_mut()
            .matmul_infer_with_config(
                bwd_down_proj_dgrad_dy,
                bwd_down_proj_dgrad_weight,
                singe_cudnn::frontend::operation::MatmulConfig::new(DataType::F32),
            )?;
        bwd_down_proj_dgrad_graph
            .graph_mut()
            .mark_as_virtual(tmp_grad)?;

        // Recompute swish forward
        let tmp_gate_swish = bwd_down_proj_dgrad_graph.bind_tensor(&VirtTensorView::new(
            [self.batch_size, self.intermediate_features],
            ElementType::Float(FloatType::Bf16),
        ));
        bwd_down_proj_dgrad_graph
            .graph_mut()
            .pointwise(PointwiseOperation::Unary {
                mode: PointwiseMode::SwishFwd,
                input: bwd_down_proj_dgrad_gate,
                output: tmp_gate_swish,
                compute_type: DataType::F32,
                nan_propagation: NanPropagation::Propagate,
                alpha1: 1.0,
                axis: None,
            });

        // Apply product rule
        // (1) dup
        bwd_down_proj_dgrad_graph
            .graph_mut()
            .pointwise(PointwiseOperation::Binary {
                mode: PointwiseMode::Mul,
                lhs: tmp_gate_swish,
                rhs: tmp_grad,
                output: bwd_down_proj_dgrad_out_dup,
                compute_type: DataType::F32,
                nan_propagation: NanPropagation::Propagate,
                alpha1: 1.0,
                alpha2: 1.0,
            });
        // (2) dgate, requiring extra dswish operation
        let tmp_dswish = bwd_down_proj_dgrad_graph
            .graph_mut()
            .pointwise_binary_infer(
                tmp_grad,
                bwd_down_proj_dgrad_up,
                PointwiseMode::Mul,
                DataType::F32,
            )?;
        bwd_down_proj_dgrad_graph
            .graph_mut()
            .pointwise(PointwiseOperation::Unary {
                mode: PointwiseMode::SwishBwd,
                input: tmp_dswish,
                output: bwd_down_proj_dgrad_out_dgate,
                compute_type: DataType::F32,
                nan_propagation: NanPropagation::Propagate,
                alpha1: 1.0,
                axis: None,
            });
        let bwd_down_proj_dgrad_graph = bwd_down_proj_dgrad_graph.build()?;

        Ok(SwiGluFfnExecutor {
            config: self.clone(),
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
            bwd_up_gate_proj_dgrad,
            bwd_down_proj_dgrad_graph,
            bwd_down_proj_dgrad_weight,
            bwd_down_proj_dgrad_dy,
            bwd_down_proj_dgrad_gate,
            bwd_down_proj_dgrad_up,
            bwd_down_proj_dgrad_out_dgate,
            bwd_down_proj_dgrad_out_dup,
        })
    }
}
