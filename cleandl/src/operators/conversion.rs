use std::sync::Arc;

use singe_cuda::{
    dim::Dim3,
    module::{KernelParameters, LaunchConfig, Module},
};
use skytens_core::data_type::{ElementType, FloatType};

use crate::{
    context::{CudaStream, ResourceKey},
    cutile::CutileKernelKey,
    tensor::TensorView,
};

pub struct ConversionExecutor {
    module: Arc<Module>,
}

impl ConversionExecutor {
    /// Converts bf16 to f32 tensor. Shapes and strides
    /// must match.
    pub fn bf16_to_f32(
        &self,
        bf16: &TensorView,
        f32: &TensorView,
        stream: &CudaStream,
    ) -> anyhow::Result<()> {
        assert_eq!(bf16.shape(), f32.shape());
        assert_eq!(bf16.strides(), f32.strides());
        assert_eq!(bf16.element_type(), ElementType::Float(FloatType::Bf16));
        assert_eq!(f32.element_type(), ElementType::Float(FloatType::F32));
        assert!(bf16.total_elements_including_padding().is_multiple_of(16));

        let kernel = self.module.function("bf16_to_f32_16384")?;
        kernel.launch_on(
            &LaunchConfig::new(
                Dim3::new(
                    bf16.total_elements_including_padding().div_ceil(16384) as u32,
                    1,
                    1,
                ),
                Dim3::new(1, 1, 1), // block dim overriden by TileIR
                0,                  // smem overriden by TileIR
            )?,
            KernelParameters::new()
                .push(bf16.data_view())
                .push(f32.data_view())
                .push(i32::try_from(bf16.total_elements_including_padding())?),
            stream.get(),
        )?;

        Ok(())
    }

    /// Converts f32 to bf16 tensor. Shapes and strides
    /// must match.
    pub fn f32_to_bf16(
        &self,
        f32: &TensorView,
        bf16: &TensorView,

        stream: &CudaStream,
    ) -> anyhow::Result<()> {
        assert_eq!(bf16.shape(), f32.shape());
        assert_eq!(bf16.strides(), f32.strides());
        assert_eq!(bf16.element_type(), ElementType::Float(FloatType::Bf16));
        assert_eq!(f32.element_type(), ElementType::Float(FloatType::F32));
        assert!(bf16.total_elements_including_padding().is_multiple_of(16));

        let kernel = self.module.function("f32_to_bf16_16384")?;
        kernel.launch_on(
            &LaunchConfig::new(
                Dim3::new(
                    bf16.total_elements_including_padding().div_ceil(16384) as u32,
                    1,
                    1,
                ),
                Dim3::new(1, 1, 1), // block dim overriden by TileIR
                0,                  // smem overriden by TileIR
            )?,
            KernelParameters::new()
                .push(f32.data_view())
                .push(bf16.data_view())
                .push(i32::try_from(bf16.total_elements_including_padding())?),
            stream.get(),
        )?;

        Ok(())
    }
}

#[derive(Debug, serde::Serialize)]
pub struct ConversionExecutorKey;

impl ResourceKey for ConversionExecutorKey {
    type Resource = ConversionExecutor;

    fn create(&self, cx: &crate::context::Context) -> anyhow::Result<Self::Resource> {
        Ok(ConversionExecutor {
            module: cx.get_or_create_resource(CutileKernelKey {
                src: include_str!("../../cutile-kernels/conversion.cu"),
            })?,
        })
    }
}
