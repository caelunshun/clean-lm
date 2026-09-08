use std::sync::Arc;

use serde::Serialize;
use singe_cudnn::{backend::tensor::TensorId, data_type::DataType};
use skytens_core::{
    data_type::{ElementType, FloatType},
    shape::{Shape, Strides},
};

use crate::{
    context::{Context, CudaStream, ResourceKey},
    cudnn::{CudnnGraphBuilder, ExecutableCudnnGraph},
    tensor::{TensorView, VirtTensorView},
};

/// Unfused matmul executor.
#[derive(Debug)]
pub struct MatmulExecutor {
    graph: ExecutableCudnnGraph,
    a: TensorId,
    b: TensorId,
    out: TensorId,
}

impl MatmulExecutor {
    pub fn get(config: MatmulConfig, cx: &Context) -> anyhow::Result<Arc<Self>> {
        cx.get_or_create_resource(config)
    }

    pub fn matmul(
        &self,
        a: &TensorView,
        b: &TensorView,
        out: &TensorView,
        stream: &CudaStream,
    ) -> anyhow::Result<()> {
        self.graph
            .execute([(self.a, a), (self.b, b), (self.out, out)], stream)?;
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct MatmulConfig {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub a_format: MatrixFormat,
    pub b_format: MatrixFormat,
    pub out_format: MatrixFormat,
}

#[derive(Debug, Copy, Clone, Serialize)]
pub enum MatrixFormat {
    RowMajor,
    ColumnMajor,
}

impl ResourceKey for MatmulConfig {
    type Resource = MatmulExecutor;

    fn create(&self, cx: &crate::context::Context) -> anyhow::Result<Self::Resource> {
        let mut graph = CudnnGraphBuilder::new(cx);

        let make_view = |dim0: usize, dim1: usize, format: MatrixFormat| match format {
            MatrixFormat::RowMajor => VirtTensorView::new(
                Shape::new([1, dim0, dim1]),
                ElementType::Float(FloatType::Bf16),
            ),
            MatrixFormat::ColumnMajor => VirtTensorView::with_strides(
                Shape::new([1, dim0, dim1]),
                Strides::new([dim0 * dim1, 1, dim0]),
                ElementType::Float(FloatType::Bf16),
            ),
        };

        let a = graph.bind_tensor(&make_view(self.m, self.k, self.a_format));
        let b = graph.bind_tensor(&make_view(self.k, self.n, self.b_format));
        let out = graph.bind_tensor(&make_view(self.m, self.n, self.out_format));

        graph.graph_mut().matmul(a, b, out, DataType::F32)?;

        Ok(MatmulExecutor {
            graph: graph.build()?,
            a,
            b,
            out,
        })
    }
}
