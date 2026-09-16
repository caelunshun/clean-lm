use std::sync::Arc;

use skycuda::{module::Module, stream::Stream};
use skytens_core::{
    data_type::{ElementType, FloatType},
    shape::Shape,
};

use crate::{
    context::{Context, ResourceKey},
    include_cubins,
    tensor::{TensorView, TensorViewMut},
};

#[derive(Debug)]
pub struct AdamWExecutor {
    module: Arc<Module>,
}

impl AdamWExecutor {
    pub fn get(cx: &Context) -> anyhow::Result<Arc<Self>> {
        cx.get_or_create_resource(AdamwExecutorKey)
    }

    /// Note: does not pre-zero the global sum.
    pub fn gradient_l2_norm_reduction(
        &self,
        gradient: &TensorView<'_>,
        global_sum: &mut TensorViewMut<'_>,
        stream: &Stream,
    ) -> anyhow::Result<()> {
        assert_eq!(gradient.element_type(), ElementType::Float(FloatType::F32));
        assert_eq!(
            global_sum.element_type(),
            ElementType::Float(FloatType::F32)
        );
        assert!(gradient.is_contiguous());
        let n = gradient.shape().element_count();
        assert!(n.is_multiple_of(16));

        assert_eq!(global_sum.shape(), Shape::new([]));

        let kernel = self
            .module
            .get_function("gradient_l2_norm_reduction_bf16_16384")?;
        let gradient_memory = gradient.storage();
        let mut global_sum_memory = global_sum.storage_mut();
        let mut launch = stream.launch_builder(&kernel);
        launch
            .arg(gradient_memory)?
            .arg(&mut global_sum_memory)?
            .arg(i32::try_from(n)?)?;
        // Block dim and smem are overriden by TileIR.
        unsafe {
            launch.launch(n.div_ceil(16384) as u32, 1, 1)?;
        }

        Ok(())
    }

    /// Applies AdamW weight update with global gradient clipping.
    ///
    /// `gradient_l2_global_sum` must be a scalar holding the sum of squared gradients computed
    /// by [`Self::gradient_l2_norm_reduction`].
    #[allow(clippy::too_many_arguments)]
    pub fn adamw_weight_update(
        &self,
        gradient: &TensorView<'_>,
        gradient_l2_global_sum: &TensorView<'_>,
        weight: &mut TensorViewMut<'_>,
        weight_bf16_replica: &mut TensorViewMut<'_>,
        adam_m: &mut TensorViewMut<'_>,
        adam_v: &mut TensorViewMut<'_>,
        beta1: f32,
        beta2: f32,
        weight_decay: f32,
        max_gradient_norm: f32,
        lr: f32,
        train_step: u32,
        stream: &Stream,
    ) -> anyhow::Result<()> {
        assert!(train_step > 0);
        assert_eq!(gradient.element_type(), ElementType::Float(FloatType::F32));
        assert_eq!(
            gradient_l2_global_sum.element_type(),
            ElementType::Float(FloatType::F32)
        );
        assert_eq!(weight.element_type(), ElementType::Float(FloatType::F32));
        assert_eq!(
            weight_bf16_replica.element_type(),
            ElementType::Float(FloatType::Bf16)
        );
        assert_eq!(adam_m.element_type(), ElementType::Float(FloatType::F32));
        assert_eq!(adam_v.element_type(), ElementType::Float(FloatType::F32));

        assert_eq!(gradient.shape(), weight.shape());
        assert_eq!(gradient.strides(), weight.strides());
        assert_eq!(weight_bf16_replica.shape(), weight.shape());
        assert_eq!(weight_bf16_replica.strides(), weight.strides());
        assert_eq!(adam_m.shape(), weight.shape());
        assert_eq!(adam_m.strides(), weight.strides());
        assert_eq!(adam_v.shape(), weight.shape());
        assert_eq!(adam_v.strides(), weight.strides());

        assert_eq!(gradient_l2_global_sum.shape(), Shape::new([]));

        assert!(weight.is_contiguous());
        let n = weight.shape().element_count();
        assert!(n.is_multiple_of(16));

        // Persistent kernel, grid size is min(occupancy, num tiles).
        let kernel = self.module.get_function(
            "adamw_weight_update_clipped_f32grad_f32master_bf16replica_4096_persistent",
        )?;
        let num_tiles = n.div_ceil(4096);
        let occupancy = kernel.max_active_blocks_per_sm(1, 0)? * stream.device().sm_count();

        let grid_size = occupancy.min(num_tiles) as u32;

        let gradient_memory = gradient.storage();
        let gradient_l2_global_sum_memory = gradient_l2_global_sum.storage();
        let mut weight_memory = weight.storage_mut();
        let mut weight_bf16_replica_memory = weight_bf16_replica.storage_mut();
        let mut adam_m_memory = adam_m.storage_mut();
        let mut adam_v_memory = adam_v.storage_mut();
        let mut launch = stream.launch_builder(&kernel);
        launch
            .arg(gradient_memory)?
            .arg(gradient_l2_global_sum_memory)?
            .arg(&mut weight_memory)?
            .arg(&mut weight_bf16_replica_memory)?
            .arg(&mut adam_m_memory)?
            .arg(&mut adam_v_memory)?
            .arg(beta1)?
            .arg(beta2)?
            .arg(weight_decay)?
            .arg(max_gradient_norm)?
            .arg(lr)?
            .arg(train_step as f32)?
            .arg(i32::try_from(n)?)?;
        // Block dim and smem are overriden by TileIR.
        unsafe {
            launch.launch(grid_size, 1, 1)?;
        }

        Ok(())
    }
}

#[derive(Debug, serde::Serialize)]
struct AdamwExecutorKey;

impl ResourceKey for AdamwExecutorKey {
    type Resource = AdamWExecutor;

    fn create(&self, cx: &Context) -> anyhow::Result<Self::Resource> {
        let module = cx.get_or_create_resource(include_cubins!("../../cutile-kernels/adamw"))?;

        Ok(AdamWExecutor { module })
    }
}
