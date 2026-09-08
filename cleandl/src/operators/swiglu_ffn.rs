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
    context::{Context, CudaStream, ResourceKey},
    cudnn::{CudnnGraphBuilder, ExecutableCudnnGraph},
    operators::matmul::{MatmulConfig, MatmulExecutor, MatrixFormat},
    tensor::{TensorView, VirtTensorView},
};

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

impl SwiGluFfnExecutor {
    pub fn get(config: SwiGluConfig, cx: &Context) -> anyhow::Result<Arc<Self>> {
        cx.get_or_create_resource(config)
    }

    pub fn config(&self) -> SwiGluConfig {
        self.config
    }

    pub fn forward_up_proj(
        &self,
        data: &TensorView,
        weight: &TensorView,
        out_up: &TensorView,
        stream: &CudaStream,
    ) -> anyhow::Result<()> {
        self.fwd_up_proj.matmul(data, weight, out_up, stream)
    }

    /// Gate projection fused with swish + hadamard product.
    pub fn forward_gate_proj(
        &self,
        data: &TensorView,
        up: &TensorView,
        weight: &TensorView,
        out_gate: &TensorView,
        out_activation: &TensorView,
        stream: &CudaStream,
    ) -> anyhow::Result<()> {
        self.fwd_fused_gate_graph.execute(
            [
                (self.fwd_fused_gate_data, data),
                (self.fwd_fused_gate_up, up),
                (self.fwd_fused_gate_weight, weight),
                (self.fwd_fused_gate_out_gate, out_gate),
                (self.fwd_fused_gate_out_activation, out_activation),
            ],
            stream,
        )
    }

    pub fn forward_down_proj(
        &self,
        activation: &TensorView,
        weight: &TensorView,
        out: &TensorView,
        stream: &CudaStream,
    ) -> anyhow::Result<()> {
        self.fwd_down_proj.matmul(activation, weight, out, stream)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn backward_down_proj_dgrad(
        &self,
        weight: &TensorView,
        dy: &TensorView,
        gate: &TensorView,
        up: &TensorView,
        out_dgate: &TensorView,
        out_dup: &TensorView,
        stream: &CudaStream,
    ) -> anyhow::Result<()> {
        self.bwd_down_proj_dgrad_graph.execute(
            [
                (self.bwd_down_proj_dgrad_weight, &weight.transposed(0, 1)),
                (self.bwd_down_proj_dgrad_dy, dy),
                (self.bwd_down_proj_dgrad_gate, gate),
                (self.bwd_down_proj_dgrad_up, up),
                (self.bwd_down_proj_dgrad_out_dgate, out_dgate),
                (self.bwd_down_proj_dgrad_out_dup, out_dup),
            ],
            stream,
        )
    }

    pub fn backward_down_proj_wgrad(
        &self,
        activation: &TensorView,
        dy: &TensorView,
        out_dweight: &TensorView,
        stream: &CudaStream,
    ) -> anyhow::Result<()> {
        self.bwd_down_proj_wgrad
            .matmul(&activation.transposed(0, 1), dy, out_dweight, stream)
    }

    pub fn backward_up_gate_proj_wgrad(
        &self,
        data: &TensorView,
        dout: &TensorView,
        out_dweight: &TensorView,
        stream: &CudaStream,
    ) -> anyhow::Result<()> {
        self.bwd_up_gate_proj_wgrad
            .matmul(&data.transposed(0, 1), dout, out_dweight, stream)
    }

    pub fn backward_up_proj_dgrad(
        &self,
        dout: &TensorView,
        weight: &TensorView,
        out_ddata: &TensorView,
        stream: &CudaStream,
    ) -> anyhow::Result<()> {
        self.bwd_up_proj_dgrad
            .matmul(dout, &weight.transposed(0, 1), out_ddata, stream)
    }

    pub fn backward_gate_proj_dgrad(
        &self,
        dout: &TensorView,
        weight: &TensorView,
        ddata_up: &TensorView,
        out_dffn: &TensorView,
        stream: &CudaStream,
    ) -> anyhow::Result<()> {
        self.bwd_gate_proj_dgrad_graph.execute(
            [
                (self.bwd_gate_proj_dgrad_dgate, dout),
                (self.bwd_gate_proj_dgrad_dup_in, ddata_up),
                (self.bwd_gate_proj_dgrad_weight, &weight.transposed(0, 1)),
                (self.bwd_gate_proj_dgrad_out_dffn, out_dffn),
            ],
            stream,
        )?;
        Ok(())
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
        let bwd_up_proj_dgrad = MatmulExecutor::get(
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

        let mut bwd_gate_proj_dgrad_graph = CudnnGraphBuilder::new(cx);
        let bwd_gate_proj_dgrad_dgate =
            bwd_gate_proj_dgrad_graph.bind_tensor(&VirtTensorView::new(
                [self.batch_size, self.intermediate_features],
                ElementType::Float(FloatType::Bf16),
            ));
        let bwd_gate_proj_dgrad_weight = bwd_gate_proj_dgrad_graph.bind_tensor(
            &VirtTensorView::new(
                [self.in_features, self.intermediate_features],
                ElementType::Float(FloatType::Bf16),
            )
            .transposed(0, 1),
        );
        let bwd_gate_proj_dgrad_dup_in =
            bwd_gate_proj_dgrad_graph.bind_tensor(&VirtTensorView::new(
                [self.batch_size, self.in_features],
                ElementType::Float(FloatType::Bf16),
            ));
        let bwd_gate_proj_dgrad_out_dffn =
            bwd_gate_proj_dgrad_graph.bind_tensor(&VirtTensorView::new(
                [self.batch_size, self.in_features],
                ElementType::Float(FloatType::Bf16),
            ));

        let tmp_dgate_in = bwd_gate_proj_dgrad_graph
            .graph_mut()
            .matmul_infer_with_config(
                bwd_gate_proj_dgrad_dgate,
                bwd_gate_proj_dgrad_weight,
                singe_cudnn::frontend::operation::MatmulConfig::new(DataType::F32),
            )?;
        bwd_gate_proj_dgrad_graph
            .graph_mut()
            .pointwise(PointwiseOperation::Binary {
                mode: PointwiseMode::Add,
                lhs: tmp_dgate_in,
                rhs: bwd_gate_proj_dgrad_dup_in,
                output: bwd_gate_proj_dgrad_out_dffn,
                compute_type: DataType::F32,
                nan_propagation: NanPropagation::Propagate,
                alpha1: 1.0,
                alpha2: 1.0,
            });
        let bwd_gate_proj_dgrad_graph = bwd_gate_proj_dgrad_graph.build()?;

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
