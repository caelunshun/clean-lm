use crate::{
    context::{Context, ResourceKey},
    operators::matmul::{MatmulConfig, MatmulExecutor, MatrixFormat, ResidualConfig},
    tensor::{Tensor, TensorView},
};
use serde::Serialize;
use skycuda::stream::Stream;
use skytens_core::data_type::{ElementType, FloatType};
use std::sync::Arc;

#[derive(Debug)]
pub struct LinearExecutor {
    config: LinearConfig,
    fwd: Arc<MatmulExecutor>,
    bwd_dgrad: Arc<MatmulExecutor>,
    bwd_wgrad: Arc<MatmulExecutor>,
}

impl LinearExecutor {
    pub fn get(config: LinearConfig, cx: &Context) -> anyhow::Result<Arc<Self>> {
        cx.get_or_create_resource(config)
    }

    pub fn config(&self) -> LinearConfig {
        self.config
    }

    /// Returns the output activations `[batch_size, out_features]`.
    pub fn forward(
        &self,
        data: &TensorView<'_>,
        weight: &TensorView<'_>,
        stream: &Stream,
    ) -> anyhow::Result<Tensor> {
        self.fwd.matmul(data, weight, None, stream)
    }

    /// Returns the output activations `[batch_size, out_features]`.
    pub fn forward_with_residual(
        &self,
        data: &TensorView,
        weight: &TensorView,
        residual: &TensorView,
        stream: &Stream,
    ) -> anyhow::Result<Tensor> {
        self.fwd.matmul(data, weight, Some(residual), stream)
    }

    /// Returns the data gradient `[batch_size, in_features]`.
    pub fn backward_dgrad(
        &self,
        dy: &TensorView<'_>,
        weight: &TensorView<'_>,
        stream: &Stream,
    ) -> anyhow::Result<Tensor> {
        self.bwd_dgrad
            .matmul(dy, &weight.transposed(0, 1), None, stream)
    }

    /// Returns the weight gradient `[in_features, out_features]`.
    pub fn backward_wgrad(
        &self,
        data: &TensorView<'_>,
        dy: &TensorView<'_>,
        stream: &Stream,
    ) -> anyhow::Result<Tensor> {
        self.bwd_wgrad
            .matmul(&data.transposed(0, 1), dy, None, stream)
    }
}

#[derive(Debug, Copy, Clone, Serialize)]
pub struct LinearConfig {
    pub estimated_batch_size: usize,
    pub in_features: usize,
    pub out_features: usize,
    pub forward_fused_residual_add: bool,
}

impl ResourceKey for LinearConfig {
    type Resource = LinearExecutor;

    fn create(&self, cx: &Context) -> anyhow::Result<Self::Resource> {
        let fwd = MatmulExecutor::get(
            MatmulConfig {
                m: self.estimated_batch_size,
                n: self.out_features,
                k: self.in_features,
                residual: self.forward_fused_residual_add.then_some(ResidualConfig {
                    format: MatrixFormat::RowMajor,
                    element_type: ElementType::Float(FloatType::Bf16),
                }),
                a_format: MatrixFormat::RowMajor,
                b_format: MatrixFormat::RowMajor,
                out_format: MatrixFormat::RowMajor,
                allow_shape_override: true,
                ab_element_type: ElementType::Float(FloatType::Bf16),
                out_element_type: ElementType::Float(FloatType::Bf16),
            },
            cx,
        )?;
        let bwd_dgrad = MatmulExecutor::get(
            MatmulConfig {
                m: self.estimated_batch_size,
                n: self.in_features,
                k: self.out_features,
                residual: None,
                a_format: MatrixFormat::RowMajor,
                b_format: MatrixFormat::ColumnMajor, // transposed
                out_format: MatrixFormat::RowMajor,
                allow_shape_override: true,
                ab_element_type: ElementType::Float(FloatType::Bf16),
                out_element_type: ElementType::Float(FloatType::Bf16),
            },
            cx,
        )?;
        let bwd_wgrad = MatmulExecutor::get(
            MatmulConfig {
                m: self.in_features,
                n: self.out_features,
                k: self.estimated_batch_size,
                residual: None,
                a_format: MatrixFormat::ColumnMajor, // transposed
                b_format: MatrixFormat::RowMajor,
                out_format: MatrixFormat::RowMajor,
                ab_element_type: ElementType::Float(FloatType::Bf16),
                out_element_type: ElementType::Float(FloatType::Bf16),
                allow_shape_override: true,
            },
            cx,
        )?;

        Ok(LinearExecutor {
            config: *self,
            fwd,
            bwd_dgrad,
            bwd_wgrad,
        })
    }
}
