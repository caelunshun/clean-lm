use std::sync::Arc;

use skycuda::{module::Module, stream::Stream};
use skytens_core::{
    data_type::{ElementType, FloatType, SIntType},
    shape::Shape,
};

use crate::{
    context::{Context, ResourceKey},
    include_cubins,
    tensor::{TensorView, TensorViewMut},
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
        x: &TensorView<'_>,
        out_stats: &mut TensorViewMut<'_>,
        stream: &Stream,
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

        let kernel = self.module.get_function("online_softmax_stats_128")?;

        let mut launch = stream.launch_builder(&kernel);
        launch
            .arg(x.storage())?
            .arg(out_stats.storage_mut())?
            .arg(i32::try_from(batch_size)?)?
            .arg(i32::try_from(hidden_size)?)?;
        // Block dim and smem are overriden by TileIR.
        unsafe {
            launch.launch(batch_size.div_ceil(128) as u32, 1, 1)?;
        }
        Ok(())
    }

    /// `logits` and `out_logits_grad` have shape `[batch_size, num_classes]`,
    /// `stats` and `labels` shape `[batch_size]`, and `out_loss` is a scalar.
    pub fn cross_entropy_loss_forward(
        &self,
        logits: &TensorView<'_>,
        stats: &TensorView<'_>,
        labels: &TensorView<'_>,
        out_loss: &mut TensorViewMut<'_>,
        out_logits_grad: &mut TensorViewMut<'_>,
        stream: &Stream,
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
        stream.fill(&mut out_loss.storage_mut(), 0u8)?;

        let kernel = self.module.get_function("cross_entropy_loss_fwd_128")?;

        let logits_memory = logits.storage();
        let stats_memory = stats.storage();
        let labels_memory = labels.storage();
        let mut out_loss_memory = out_loss.storage_mut();
        let mut out_logits_grad_memory = out_logits_grad.storage_mut();
        let mut launch = stream.launch_builder(&kernel);
        launch
            .arg(logits_memory)?
            .arg(stats_memory)?
            .arg(labels_memory)?
            .arg(&mut out_loss_memory)?
            .arg(&mut out_logits_grad_memory)?
            .arg(i32::try_from(batch_size)?)?
            .arg((batch_size as f32).recip())?
            .arg(i32::try_from(num_classes)?)?;
        // Block dim and smem are overriden by TileIR.
        unsafe {
            launch.launch(batch_size.div_ceil(128) as u32, 1, 1)?;
        }
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
