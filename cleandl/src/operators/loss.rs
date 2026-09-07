use std::sync::Arc;

use singe_cuda::{
    dim::Dim3,
    module::{KernelParameters, LaunchConfig, Module},
};
use skytens_core::{
    data_type::{ElementType, FloatType, SIntType},
    shape::Shape,
};

use crate::{
    context::{Context, CudaStream, ResourceKey},
    include_cubins,
    tensor::TensorView,
};

#[derive(Debug)]
pub struct LossExecutor {
    module: Arc<Module>,
}

impl LossExecutor {
    pub fn get(cx: &Context) -> anyhow::Result<Arc<Self>> {
        cx.get_or_create_resource(LossExecutorKey)
    }

    /// `x` has shape `[batch_size, hidden_size]`; `out_stats` has shape `[batch_size]`.
    pub fn online_softmax_stats(
        &self,
        x: &TensorView,
        out_stats: &TensorView,
        stream: &CudaStream,
    ) -> anyhow::Result<()> {
        assert_eq!(x.element_type(), ElementType::Float(FloatType::Bf16));
        assert_eq!(out_stats.element_type(), ElementType::Float(FloatType::F32));

        assert!(x.is_packed_row_major());
        assert!(out_stats.is_packed_row_major());

        let &[batch_size, hidden_size] = x.shape().dims_slice() else {
            panic!("wrong shape for x");
        };
        assert!(hidden_size.is_multiple_of(16));
        assert_eq!(out_stats.shape(), Shape::new([batch_size]));

        let kernel = self.module.function("online_softmax_stats_128")?;

        kernel.launch_on(
            &LaunchConfig::new(
                Dim3::new(batch_size.div_ceil(128) as u32, 1, 1),
                Dim3::new(1, 1, 1), // overriden by TileIR
                0,                  // overriden by TileIR
            )?,
            KernelParameters::new()
                .push(x.data_view())
                .push(out_stats.data_view())
                .push(i32::try_from(batch_size)?)
                .push(i32::try_from(hidden_size)?),
            stream.get(),
        )?;
        Ok(())
    }

    /// `logits` and `out_logits_grad` have shape `[batch_size, num_classes]`,
    /// `stats` and `labels` shape `[batch_size]`, and `out_loss` is a scalar.
    pub fn cross_entropy_loss_forward(
        &self,
        logits: &TensorView,
        stats: &TensorView,
        labels: &TensorView,
        out_loss: &TensorView,
        out_logits_grad: &TensorView,
        stream: &CudaStream,
    ) -> anyhow::Result<()> {
        assert_eq!(logits.element_type(), ElementType::Float(FloatType::Bf16));
        assert_eq!(stats.element_type(), ElementType::Float(FloatType::F32));
        assert_eq!(labels.element_type(), ElementType::SInt(SIntType::S32));
        assert_eq!(out_loss.element_type(), ElementType::Float(FloatType::F32));
        assert_eq!(
            out_logits_grad.element_type(),
            ElementType::Float(FloatType::Bf16)
        );

        assert!(logits.is_packed_row_major());
        assert!(stats.is_packed_row_major());
        assert!(labels.is_packed_row_major());
        assert!(out_logits_grad.is_packed_row_major());

        let &[batch_size, num_classes] = logits.shape().dims_slice() else {
            panic!("wrong shape for logits");
        };
        assert!(num_classes.is_multiple_of(16));
        assert_eq!(stats.shape(), Shape::new([batch_size]));
        assert_eq!(labels.shape(), Shape::new([batch_size]));
        assert_eq!(out_loss.shape(), Shape::new([]));
        assert_eq!(out_logits_grad.shape(), logits.shape());

        assert!(batch_size > 0);

        // Atomic variable needs pre-zeroed.
        out_loss.memset_u8(0, stream)?;

        let kernel = self.module.function("cross_entropy_loss_fwd_128")?;

        kernel.launch_on(
            &LaunchConfig::new(
                Dim3::new(batch_size.div_ceil(128) as u32, 1, 1),
                Dim3::new(1, 1, 1), // overriden by TileIR
                0,                  // overriden by TileIR
            )?,
            KernelParameters::new()
                .push(logits.data_view())
                .push(stats.data_view())
                .push(labels.data_view())
                .push(out_loss.data_view())
                .push(out_logits_grad.data_view())
                .push(i32::try_from(batch_size)?)
                .push((batch_size as f32).recip())
                .push(i32::try_from(num_classes)?),
            stream.get(),
        )?;
        Ok(())
    }
}

#[derive(Debug, serde::Serialize)]
struct LossExecutorKey;

impl ResourceKey for LossExecutorKey {
    type Resource = LossExecutor;

    fn create(&self, cx: &Context) -> anyhow::Result<Self::Resource> {
        Ok(LossExecutor {
            module: cx.get_or_create_resource(include_cubins!("../../cutile-kernels/loss"))?,
        })
    }
}
