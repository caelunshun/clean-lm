use std::sync::Arc;

use singe_cuda::{
    dim::Dim3,
    module::{KernelParameters, LaunchConfig, Module},
};
use skytens_core::{
    data_type::{ElementType, FloatType},
    shape::Shape,
};

use crate::{
    context::{Context, CudaStream, ResourceKey},
    include_cubins,
    tensor::TensorView,
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
        gradient: &TensorView,
        global_sum: &TensorView,
        stream: &CudaStream,
    ) -> anyhow::Result<()> {
        assert_eq!(gradient.element_type(), ElementType::Float(FloatType::Bf16));
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
            .function("gradient_l2_norm_reduction_bf16_16384")?;
        kernel.launch_on(
            &LaunchConfig::new(
                Dim3::new(n.div_ceil(16384) as u32, 1, 1),
                Dim3::new(1, 1, 1), // block dim overriden by TileIR
                0,                  // smem overriden by TileIR
            )?,
            KernelParameters::new()
                .push(gradient.data_view())
                .push(global_sum.data_view())
                .push(i32::try_from(n)?),
            stream.get(),
        )?;

        Ok(())
    }

    /// Applies AdamW weight update with global gradient clipping.
    ///
    /// `gradient_l2_global_sum` must be a scalar holding the sum of squared gradients computed
    /// by [`Self::gradient_l2_norm_reduction`].
    #[allow(clippy::too_many_arguments)]
    pub fn adamw_weight_update(
        &self,
        gradient: &TensorView,
        gradient_l2_global_sum: &TensorView,
        weight: &TensorView,
        weight_bf16_replica: &TensorView,
        adam_m: &TensorView,
        adam_v: &TensorView,
        beta1: f32,
        beta2: f32,
        weight_decay: f32,
        max_gradient_norm: f32,
        lr: f32,
        train_step: u32,
        stream: &CudaStream,
    ) -> anyhow::Result<()> {
        assert!(train_step > 0);
        assert_eq!(gradient.element_type(), ElementType::Float(FloatType::Bf16));
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
        let kernel = self.module.function(
            "adamw_weight_update_clipped_bf16grad_f32master_bf16replica_4096_persistent",
        )?;
        let num_tiles = n.div_ceil(4096);
        let occupancy = kernel.occupancy_max_active_blocks_per_multiprocessor(1, 0)?;
        let occupancy = occupancy * stream.cuda().device().properties()?.multi_processor_count;

        let grid_size = occupancy.min(num_tiles as i32) as u32;

        kernel.launch_on(
            &LaunchConfig::new(
                Dim3::new(grid_size, 1, 1),
                Dim3::new(1, 1, 1), // overriden by TileIR
                0,                  // overriden by TileIR
            )?,
            KernelParameters::new()
                .push(gradient.data_view())
                .push(gradient_l2_global_sum.data_view())
                .push(weight.data_view())
                .push(weight_bf16_replica.data_view())
                .push(adam_m.data_view())
                .push(adam_v.data_view())
                .push(beta1)
                .push(beta2)
                .push(weight_decay)
                .push(max_gradient_norm)
                .push(lr)
                .push(train_step as f32)
                .push(i32::try_from(n)?),
            stream.get(),
        )?;

        Ok(())
    }
}

#[derive(Debug, serde::Serialize)]
struct AdamwExecutorKey;

impl ResourceKey for AdamwExecutorKey {
    type Resource = AdamWExecutor;

    fn create(&self, cx: &Context) -> anyhow::Result<Self::Resource> {
        cx.cuda().bind()?;

        let module = cx.get_or_create_resource(include_cubins!("../../cutile-kernels/adamw"))?;

        Ok(AdamWExecutor { module })
    }
}
