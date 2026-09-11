use std::sync::Arc;

use serde::Serialize;
use skycuda::stream::Stream;
use skycuda_cudnn::{DataType, MatmulAttributes, TensorId};
use skytens_core::{
    data_type::{ElementType, FloatType},
    shape::{Shape, Strides},
};

use crate::{
    context::{Context, ResourceKey},
    cudnn::{CudnnGraphBuilder, ExecutableCudnnGraph},
    tensor::{TensorView, TensorViewMut, VirtTensorView},
};

/// Unfused matmul executor.
#[derive(Debug)]
pub struct MatmulExecutor {
    config: MatmulConfig,
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
        a: &TensorView<'_>,
        b: &TensorView<'_>,
        out: &mut TensorViewMut<'_>,
        stream: &Stream,
    ) -> anyhow::Result<()> {
        assert_eq!(a.shape().dims_slice(), &[self.config.m, self.config.k]);
        assert_eq!(b.shape().dims_slice(), &[self.config.k, self.config.n]);
        assert_eq!(out.shape().dims_slice(), &[self.config.m, self.config.n]);
        self.graph.execute(
            &[
                (self.a, &a.reshaped([1, self.config.m, self.config.k])),
                (self.b, &b.reshaped([1, self.config.k, self.config.n])),
            ],
            &mut [(
                self.out,
                out.reshaped_mut([1, self.config.m, self.config.n]),
            )],
            stream,
        )?;
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
        let graph = CudnnGraphBuilder::new(cx)?;

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
        let out = graph.graph().matmul(
            a,
            b,
            &MatmulAttributes::new().compute_data_type(DataType::Float),
        )?;
        let out = graph.mark_output(out, &make_view(self.m, self.n, self.out_format));
        let (a, b) = (a.uid(), b.uid());

        Ok(MatmulExecutor {
            config: self.clone(),
            graph: graph.build()?,
            a,
            b,
            out,
        })
    }
}
