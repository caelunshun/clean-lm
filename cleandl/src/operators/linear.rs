use std::sync::Arc;

use serde::Serialize;
use skycuda::stream::Stream;

use crate::{
    context::{Context, ResourceKey},
    operators::matmul::{MatmulConfig, MatmulExecutor, MatrixFormat},
    tensor::{TensorView, TensorViewMut},
};

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

    pub fn forward(
        &self,
        data: &TensorView<'_>,
        weight: &TensorView<'_>,
        out: &mut TensorViewMut<'_>,
        stream: &Stream,
    ) -> anyhow::Result<()> {
        self.fwd.matmul(data, weight, out, stream)
    }

    pub fn backward_dgrad(
        &self,
        dy: &TensorView<'_>,
        weight: &TensorView<'_>,
        out_ddata: &mut TensorViewMut<'_>,
        stream: &Stream,
    ) -> anyhow::Result<()> {
        self.bwd_dgrad
            .matmul(dy, &weight.transposed(0, 1), out_ddata, stream)
    }

    pub fn backward_wgrad(
        &self,
        data: &TensorView<'_>,
        dy: &TensorView<'_>,
        out_dweight: &mut TensorViewMut<'_>,
        stream: &Stream,
    ) -> anyhow::Result<()> {
        self.bwd_wgrad
            .matmul(&data.transposed(0, 1), dy, out_dweight, stream)
    }
}

#[derive(Debug, Copy, Clone, Serialize)]
pub struct LinearConfig {
    pub batch_size: usize,
    pub in_features: usize,
    pub out_features: usize,
}

impl ResourceKey for LinearConfig {
    type Resource = LinearExecutor;

    fn create(&self, cx: &Context) -> anyhow::Result<Self::Resource> {
        let fwd = MatmulExecutor::get(
            MatmulConfig {
                m: self.batch_size,
                n: self.out_features,
                k: self.in_features,
                a_format: MatrixFormat::RowMajor,
                b_format: MatrixFormat::RowMajor,
                out_format: MatrixFormat::RowMajor,
            },
            cx,
        )?;
        let bwd_dgrad = MatmulExecutor::get(
            MatmulConfig {
                m: self.batch_size,
                n: self.in_features,
                k: self.out_features,
                a_format: MatrixFormat::RowMajor,
                b_format: MatrixFormat::ColumnMajor, // transposed
                out_format: MatrixFormat::RowMajor,
            },
            cx,
        )?;
        let bwd_wgrad = MatmulExecutor::get(
            MatmulConfig {
                m: self.in_features,
                n: self.out_features,
                k: self.batch_size,
                a_format: MatrixFormat::ColumnMajor, // transposed
                b_format: MatrixFormat::RowMajor,
                out_format: MatrixFormat::RowMajor,
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
