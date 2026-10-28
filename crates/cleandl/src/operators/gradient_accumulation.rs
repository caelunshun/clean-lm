use crate::{
    context::{Context, ResourceKey},
    include_cubins,
    tensor::{TensorView, TensorViewMut},
};
use serde::Serialize;
use skycuda::{function::Function, stream::Stream};
use skytens_core::data_type::{ElementType, FloatType};
use std::sync::Arc;

#[derive(Debug)]
pub struct GradientAccumulationExecutor {
    kernel: Function,
}

impl GradientAccumulationExecutor {
    pub fn get(cx: &Context) -> anyhow::Result<Arc<Self>> {
        cx.get_or_create_resource(Key)
    }

    pub fn add_contribution_to_gradient(
        &self,
        contribution: &TensorView,
        gradient: &mut TensorViewMut,
        stream: &Stream,
    ) -> anyhow::Result<()> {
        assert_eq!(contribution.shape(), gradient.shape());
        assert_eq!(
            contribution.element_type(),
            ElementType::Float(FloatType::Bf16)
        );
        assert_eq!(gradient.element_type(), ElementType::Float(FloatType::F32));
        assert!(contribution.is_packed_row_major());
        assert!(gradient.is_packed_row_major());
        let num_elements = i32::try_from(contribution.shape().element_count())?;

        let occupancy = stream.device().sm_count() * self.kernel.max_active_blocks_per_sm(1, 0)?;

        unsafe {
            stream
                .launch_builder(&self.kernel)
                .arg(contribution.storage())?
                .arg(gradient.storage_mut())?
                .arg(num_elements)?
                .launch(occupancy as u32, 1, 1)?;
        }

        Ok(())
    }
}

#[derive(Debug, Serialize)]
struct Key;

impl ResourceKey for Key {
    type Resource = GradientAccumulationExecutor;

    fn create(&self, cx: &Context) -> anyhow::Result<Self::Resource> {
        let module = cx.get_or_create_resource(include_cubins!(
            "../../cutile-kernels/gradient_accumulation"
        ))?;
        let kernel = module.get_function("accumulate_gradient_f32grad_bf16contrib_persistent")?;
        Ok(GradientAccumulationExecutor { kernel })
    }
}
