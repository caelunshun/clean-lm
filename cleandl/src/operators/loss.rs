use crate::{
    context::{Context, ResourceKey},
    include_cubins,
    tensor::{Tensor, TensorView},
};
use skycuda::{module::Module, stream::Stream};
use skytens_core::{
    data_type::{ElementType, FloatType, SIntType},
    shape::Shape,
};
use std::sync::Arc;

#[derive(Debug)]
pub struct LossExecutor {
    cx: Context,
    module: Arc<Module>,
}

/// Outputs of [`LossExecutor::cross_entropy_loss_fwd_bwd`].
#[derive(Debug)]
pub struct CrossEntropyLossOutput {
    /// Mean loss over the batch, as a scalar.
    pub loss: Tensor,
    /// Gradient with respect to the logits, same shape as `logits`.
    pub logits_grad: Tensor,
}

impl LossExecutor {
    pub fn get(cx: &Context) -> anyhow::Result<Arc<Self>> {
        cx.get_or_create_resource(LossExecutorKey)
    }

    /// `x` has shape `[batch_size, hidden_size]`; the returned stats have shape `[batch_size]`.
    pub fn online_softmax_stats(
        &self,
        x: &TensorView<'_>,
        stream: &Stream,
    ) -> anyhow::Result<Tensor> {
        assert_eq!(x.element_type(), ElementType::Float(FloatType::Bf16));
        assert!(x.is_packed_row_major());

        let &[batch_size, hidden_size] = x.shape().dims_slice() else {
            panic!("wrong shape for x");
        };
        assert!(hidden_size.is_multiple_of(16));

        let mut out_stats = Tensor::new_zeroed(
            Shape::new([batch_size]),
            ElementType::Float(FloatType::F32),
            &self.cx,
        )?;

        let kernel = self.module.get_function("online_softmax_stats_128")?;

        {
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
        }

        Ok(out_stats)
    }

    /// `logits` has shape `[batch_size, num_classes]` and `stats` and `labels`
    /// shape `[batch_size]`.
    pub fn cross_entropy_loss_fwd_bwd(
        &self,
        logits: &TensorView<'_>,
        stats: &TensorView<'_>,
        labels: &TensorView<'_>,
        stream: &Stream,
    ) -> anyhow::Result<CrossEntropyLossOutput> {
        assert_eq!(logits.element_type(), ElementType::Float(FloatType::Bf16));
        assert_eq!(stats.element_type(), ElementType::Float(FloatType::F32));
        assert_eq!(labels.element_type(), ElementType::SInt(SIntType::S32));

        assert!(logits.is_packed_row_major());
        assert!(stats.is_packed_row_major());
        assert!(labels.is_packed_row_major());

        let &[batch_size, num_classes] = logits.shape().dims_slice() else {
            panic!("wrong shape for logits");
        };
        assert!(num_classes.is_multiple_of(16));
        assert_eq!(stats.shape(), Shape::new([batch_size]));
        assert_eq!(labels.shape(), Shape::new([batch_size]));

        assert!(batch_size > 0);

        // The loss is an atomic accumulator, so it relies on the allocation being zeroed.
        let mut out_loss =
            Tensor::new_zeroed(Shape::new([]), ElementType::Float(FloatType::F32), &self.cx)?;
        let mut out_logits_grad = Tensor::new_zeroed(
            logits.shape(),
            ElementType::Float(FloatType::Bf16),
            &self.cx,
        )?;

        let kernel = self.module.get_function("cross_entropy_loss_fwd_bwd_128")?;

        {
            let logits_memory = logits.storage();
            let stats_memory = stats.storage();
            let labels_memory = labels.storage();
            let out_loss_memory = out_loss.storage_mut();
            let out_logits_grad_memory = out_logits_grad.storage_mut();
            let mut launch = stream.launch_builder(&kernel);
            launch
                .arg(logits_memory)?
                .arg(stats_memory)?
                .arg(labels_memory)?
                .arg(out_loss_memory)?
                .arg(out_logits_grad_memory)?
                .arg(i32::try_from(batch_size)?)?
                .arg((batch_size as f32).recip())?
                .arg(i32::try_from(num_classes)?)?;
            // Block dim and smem are overriden by TileIR.
            unsafe {
                launch.launch(batch_size.div_ceil(128) as u32, 1, 1)?;
            }
        }

        Ok(CrossEntropyLossOutput {
            loss: out_loss,
            logits_grad: out_logits_grad,
        })
    }
}

#[derive(Debug, serde::Serialize)]
struct LossExecutorKey;

impl ResourceKey for LossExecutorKey {
    type Resource = LossExecutor;

    fn create(&self, cx: &Context) -> anyhow::Result<Self::Resource> {
        Ok(LossExecutor {
            cx: cx.clone(),
            module: cx.get_or_create_resource(include_cubins!("../../cutile-kernels/loss"))?,
        })
    }
}
