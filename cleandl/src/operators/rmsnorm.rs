use std::sync::Arc;

use skycuda::stream::Stream;
use skycuda_cudnn::{NormAttributes, NormBackwardAttributes, NormFwdPhase, TensorId};
use skytens_core::{
    data_type::{ElementType, FloatType},
    shape::Shape,
};

use crate::{
    context::{Context, ResourceKey},
    cudnn::{CudnnGraphBuilder, ExecutableCudnnGraph},
    tensor::{TensorView, TensorViewMut, VirtTensorView},
};

#[derive(Debug)]
pub struct RmsNormExecutor {
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
        x: &TensorView<'_>,
        scale: &TensorView<'_>,
        out_y: &mut TensorViewMut<'_>,
        out_inv_variance: &mut TensorViewMut<'_>,
        stream: &Stream,
    ) -> anyhow::Result<()> {
        self.forward_graph.execute(
            &[(self.forward_x, x), (self.forward_scale, scale)],
            &mut [
                (self.forward_y, out_y.view_mut()),
                (self.forward_inv_variance, out_inv_variance.view_mut()),
            ],
            stream,
        )?;

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn backward(
        &self,
        x: &TensorView<'_>,
        dy: &TensorView<'_>,
        inv_variance: &TensorView<'_>,
        scale: &TensorView<'_>,
        out_dx: &mut TensorViewMut<'_>,
        out_dscale: &mut TensorViewMut<'_>,
        stream: &Stream,
    ) -> anyhow::Result<()> {
        self.backward_graph.execute(
            &[
                (self.backward_x, x),
                (self.backward_dy, dy),
                (self.backward_inv_variance, inv_variance),
                (self.backward_scale, scale),
            ],
            &mut [
                (self.backward_dx, out_dx.view_mut()),
                (self.backward_dscale, out_dscale.view_mut()),
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
        let x_layout = VirtTensorView::new(
            Shape::new([self.batch_size, self.token_dim]),
            ElementType::Float(FloatType::Bf16),
        );
        let scale_layout = VirtTensorView::new(
            Shape::new([1, self.token_dim]),
            ElementType::Float(FloatType::Bf16),
        );
        let inv_variance_layout = VirtTensorView::new(
            Shape::new([self.batch_size, 1]),
            ElementType::Float(FloatType::F32),
        );

        let forward_graph = CudnnGraphBuilder::new(cx)?;

        let forward_x = forward_graph.bind_tensor(&x_layout);
        let forward_scale = forward_graph.bind_tensor(&scale_layout);
        let epsilon = forward_graph.scalar(1e-6);
        let [forward_y, forward_inv_variance, _] = forward_graph.graph().rmsnorm(
            forward_x,
            forward_scale,
            None,
            Some(epsilon),
            &NormAttributes::new().forward_phase(NormFwdPhase::Training),
        )?;
        let forward_y = forward_graph.mark_output(forward_y.unwrap(), &x_layout);
        let forward_inv_variance =
            forward_graph.mark_output(forward_inv_variance.unwrap(), &inv_variance_layout);
        let (forward_x, forward_scale) = (forward_x.uid(), forward_scale.uid());
        let forward_graph = forward_graph.build()?;

        let backward_graph = CudnnGraphBuilder::new(cx)?;

        let backward_x = backward_graph.bind_tensor(&x_layout);
        let backward_dy = backward_graph.bind_tensor(&x_layout);
        let backward_inv_variance = backward_graph.bind_tensor(&inv_variance_layout);
        let backward_scale = backward_graph.bind_tensor(&scale_layout);

        let [backward_dx, backward_dscale, _] = backward_graph.graph().rmsnorm_backward(
            backward_dy,
            backward_x,
            backward_scale,
            Some(backward_inv_variance),
            &NormBackwardAttributes::new().has_dbias(false),
        )?;
        let backward_dx = backward_graph.mark_output(backward_dx.unwrap(), &x_layout);
        let backward_dscale = backward_graph.mark_output(backward_dscale.unwrap(), &scale_layout);
        let (backward_x, backward_dy, backward_inv_variance, backward_scale) = (
            backward_x.uid(),
            backward_dy.uid(),
            backward_inv_variance.uid(),
            backward_scale.uid(),
        );
        let backward_graph = backward_graph.build()?;

        Ok(RmsNormExecutor {
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
