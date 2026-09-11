use std::sync::Arc;

use skycuda::{module::Module, stream::Stream};
use skytens_core::data_type::{ElementType, FloatType};

use crate::{
    context::{Context, ResourceKey},
    include_cubins,
    tensor::{TensorView, TensorViewMut},
};

pub struct ConversionExecutor {
    module: Arc<Module>,
}

impl ConversionExecutor {
    pub fn get(cx: &Context) -> anyhow::Result<Arc<Self>> {
        cx.get_or_create_resource(ConversionExecutorKey)
    }

    /// Converts bf16 to f32 tensor. Shapes and strides
    /// must match.
    pub fn bf16_to_f32(
        &self,
        bf16: &TensorView<'_>,
        f32: &mut TensorViewMut<'_>,
        stream: &Stream,
    ) -> anyhow::Result<()> {
        assert_eq!(bf16.shape(), f32.shape());
        assert_eq!(bf16.strides(), f32.strides());
        assert_eq!(bf16.element_type(), ElementType::Float(FloatType::Bf16));
        assert_eq!(f32.element_type(), ElementType::Float(FloatType::F32));
        assert!(bf16.is_contiguous());
        let n = bf16.shape().element_count();
        assert!(n.is_multiple_of(16));

        let kernel = self.module.get_function("bf16_to_f32_16384")?;
        let bf16_memory = bf16.storage();
        let mut f32_memory = f32.storage_mut();
        let mut launch = stream.launch_builder(&kernel);
        launch
            .arg(bf16_memory)?
            .arg(&mut f32_memory)?
            .arg(i32::try_from(n)?)?;
        // Block dim and smem are overriden by TileIR.
        unsafe {
            launch.launch(n.div_ceil(16384) as u32, 1, 1)?;
        }

        Ok(())
    }

    /// Converts f32 to bf16 tensor. Shapes and strides
    /// must match.
    pub fn f32_to_bf16(
        &self,
        f32: &TensorView<'_>,
        bf16: &mut TensorViewMut<'_>,
        stream: &Stream,
    ) -> anyhow::Result<()> {
        assert_eq!(bf16.shape(), f32.shape());
        assert_eq!(bf16.strides(), f32.strides());
        assert_eq!(bf16.element_type(), ElementType::Float(FloatType::Bf16));
        assert_eq!(f32.element_type(), ElementType::Float(FloatType::F32));
        assert!(bf16.is_contiguous());
        let n = bf16.shape().element_count();
        assert!(n.is_multiple_of(16));

        let kernel = self.module.get_function("f32_to_bf16_16384")?;
        let f32_memory = f32.storage();
        let mut bf16_memory = bf16.storage_mut();
        let mut launch = stream.launch_builder(&kernel);
        launch
            .arg(f32_memory)?
            .arg(&mut bf16_memory)?
            .arg(i32::try_from(n)?)?;
        // Block dim and smem are overriden by TileIR.
        unsafe {
            launch.launch(n.div_ceil(16384) as u32, 1, 1)?;
        }

        Ok(())
    }
}

#[derive(Debug, serde::Serialize)]
struct ConversionExecutorKey;

impl ResourceKey for ConversionExecutorKey {
    type Resource = ConversionExecutor;

    fn create(&self, cx: &crate::context::Context) -> anyhow::Result<Self::Resource> {
        Ok(ConversionExecutor {
            module: cx
                .get_or_create_resource(include_cubins!("../../cutile-kernels/conversion"))?,
        })
    }
}
