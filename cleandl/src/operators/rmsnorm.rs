use std::sync::Arc;

use singe_cudnn::{
    backend::tensor::{TensorId, TensorSpec},
    data_type::DataType,
    frontend::{
        graph::{RmsNormalizationBackwardOutputs, RmsNormalizationOutputs},
        operation::{RmsNormalizationBackwardConfig, RmsNormalizationConfig},
    },
};
use skytens_core::{
    data_type::{ElementType, FloatType},
    shape::Shape,
};

use crate::{
    context::{Context, CudaStream, ResourceKey},
    cudnn::{CudnnGraphBuilder, ExecutableCudnnGraph},
    tensor::{TensorView, VirtTensorView},
};

#[derive(Debug)]
pub struct RmsNormExecutor {
    config: RmsNormConfig,
    forward_graph: ExecutableCudnnGraph,
    forward_x: TensorId,
    forward_scale: TensorId,
    forward_y: TensorId,
    forward_inv_variance: TensorId,
    backward_graph: ExecutableCudnnGraph,
    backward_x: TensorId,
    backward_inv_variance: TensorId,
    backward_dy: TensorId,
    backward_scale: TensorId,
    backward_dscale: TensorId,
    backward_dx: TensorId,
}

impl RmsNormExecutor {
    pub fn get(config: RmsNormConfig, cx: &Context) -> anyhow::Result<Arc<Self>> {
        cx.get_or_create_resource(config)
    }

    pub fn forward(
        &self,
        x: &TensorView,
        scale: &TensorView,
        out_y: &TensorView,
        out_inv_variance: &TensorView,
        stream: &CudaStream,
    ) -> anyhow::Result<()> {
        self.forward_graph.execute(
            [
                (self.forward_x, x),
                (self.forward_scale, scale),
                (self.forward_y, out_y),
                (self.forward_inv_variance, out_inv_variance),
            ],
            stream,
        )?;

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn backward(
        &self,
        x: &TensorView,
        dy: &TensorView,
        inv_variance: &TensorView,
        scale: &TensorView,
        out_dx: &TensorView,
        out_dscale: &TensorView,
        stream: &CudaStream,
    ) -> anyhow::Result<()> {
        self.backward_graph.execute(
            [
                (self.backward_x, x),
                (self.backward_dy, dy),
                (self.backward_inv_variance, inv_variance),
                (self.backward_scale, scale),
                (self.backward_dx, out_dx),
                (self.backward_dscale, out_dscale),
            ],
            stream,
        )?;

        Ok(())
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct RmsNormConfig {
    pub batch_size: usize,
    pub token_dim: usize,
}

impl ResourceKey for RmsNormConfig {
    type Resource = RmsNormExecutor;

    fn create(&self, cx: &Context) -> anyhow::Result<Self::Resource> {
        let mut forward_graph = CudnnGraphBuilder::new(cx);

        let forward_x = forward_graph.bind_tensor(&VirtTensorView::new(
            Shape::new([self.batch_size, self.token_dim]),
            ElementType::Float(FloatType::Bf16),
        ));
        let epsilon = forward_graph.scalar(1e-6);
        let RmsNormalizationOutputs {
            output: forward_y,
            inv_variance: forward_inv_variance,
            scale: forward_scale,
        } = forward_graph
            .graph_mut()
            .rms_normalization_infer(forward_x, RmsNormalizationConfig::training(epsilon))?;
        let forward_graph = forward_graph.build()?;

        let mut backward_graph = CudnnGraphBuilder::new(cx);

        let backward_x = backward_graph.bind_tensor(&VirtTensorView::new(
            Shape::new([self.batch_size, self.token_dim]),
            ElementType::Float(FloatType::Bf16),
        ));
        let backward_dy = backward_graph.bind_tensor(&VirtTensorView::new(
            Shape::new([self.batch_size, self.token_dim]),
            ElementType::Float(FloatType::Bf16),
        ));
        let backward_inv_variance = backward_graph.bind_tensor(&VirtTensorView::new(
            Shape::new([self.batch_size, 1]),
            ElementType::Float(FloatType::F32),
        ));
        let backward_scale = backward_graph.bind_tensor(&VirtTensorView::new(
            Shape::new([1, self.token_dim]),
            ElementType::Float(FloatType::Bf16),
        ));

        let RmsNormalizationBackwardOutputs {
            dx: backward_dx,
            dscale: backward_dscale,
            bias_gradient: _,
        } = backward_graph
            .graph_mut()
            .rms_normalization_backward_infer(
                backward_x,
                backward_inv_variance,
                backward_dy,
                backward_scale,
                RmsNormalizationBackwardConfig::new(),
            )?;
        let backward_graph = backward_graph.build()?;

        Ok(RmsNormExecutor {
            config: self.clone(),
            forward_graph,
            forward_x,
            forward_scale,
            forward_y,
            forward_inv_variance,
            backward_graph,
            backward_x,
            backward_inv_variance,
            backward_dy,
            backward_scale,
            backward_dscale,
            backward_dx,
        })
    }
}
