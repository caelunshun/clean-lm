use crate::{
    context::{Context, ResourceKey},
    include_cubins,
    tensor::{Tensor, TensorView},
};
use skycuda::{module::Module, stream::Stream};
use skytens_core::data_type::{ElementType, FloatType};
use std::sync::Arc;

pub struct ConversionExecutor {
    cx: Context,
    module: Arc<Module>,
}

impl ConversionExecutor {
    pub fn get(cx: &Context) -> anyhow::Result<Arc<Self>> {
        cx.get_or_create_resource(ConversionExecutorKey)
    }

    /// Converts a bf16 tensor to a newly allocated f32 tensor of the same shape.
    pub fn bf16_to_f32(&self, bf16: &TensorView<'_>, stream: &Stream) -> anyhow::Result<Tensor> {
        assert_eq!(bf16.element_type(), ElementType::Float(FloatType::Bf16));
        assert!(bf16.is_packed_row_major());
        let n = bf16.shape().element_count();
        assert!(n.is_multiple_of(16));

        let mut f32 =
            Tensor::new_zeroed(bf16.shape(), ElementType::Float(FloatType::F32), &self.cx)?;

        let kernel = self.module.get_function("bf16_to_f32_16384")?;
        {
            let bf16_memory = bf16.storage();
            let mut launch = stream.launch_builder(&kernel);
            launch
                .arg(bf16_memory)?
                .arg(f32.storage_mut())?
                .arg(i32::try_from(n)?)?;
            // Block dim and smem are overriden by TileIR.
            unsafe {
                launch.launch(n.div_ceil(16384) as u32, 1, 1)?;
            }
        }

        Ok(f32)
    }

    /// Converts an f32 tensor to a newly allocated bf16 tensor of the same shape.
    pub fn f32_to_bf16(&self, f32: &TensorView<'_>, stream: &Stream) -> anyhow::Result<Tensor> {
        assert_eq!(f32.element_type(), ElementType::Float(FloatType::F32));
        assert!(f32.is_packed_row_major());
        let n = f32.shape().element_count();
        assert!(n.is_multiple_of(16));

        let mut bf16 =
            Tensor::new_zeroed(f32.shape(), ElementType::Float(FloatType::Bf16), &self.cx)?;

        let kernel = self.module.get_function("f32_to_bf16_16384")?;
        {
            let f32_memory = f32.storage();
            let mut launch = stream.launch_builder(&kernel);
            launch
                .arg(f32_memory)?
                .arg(bf16.storage_mut())?
                .arg(i32::try_from(n)?)?;
            // Block dim and smem are overriden by TileIR.
            unsafe {
                launch.launch(n.div_ceil(16384) as u32, 1, 1)?;
            }
        }

        Ok(bf16)
    }
}

#[derive(Debug, serde::Serialize)]
struct ConversionExecutorKey;

impl ResourceKey for ConversionExecutorKey {
    type Resource = ConversionExecutor;

    fn create(&self, cx: &crate::context::Context) -> anyhow::Result<Self::Resource> {
        Ok(ConversionExecutor {
            cx: cx.clone(),
            module: cx
                .get_or_create_resource(include_cubins!("../../cutile-kernels/conversion"))?,
        })
    }
}
