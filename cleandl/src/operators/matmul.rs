use std::sync::Arc;

use serde::Serialize;
use skycuda::stream::Stream;
use skycuda_cudnn::{DataType, MatmulAttributes, PointwiseAttributes, PointwiseMode, TensorId};
use skytens_core::{
    data_type::{ElementType, FloatType},
    shape::{Shape, Strides},
};

use crate::{
    context::{Context, ResourceKey},
    cudnn::{CudnnGraphBuilder, ExecutableCudnnGraph},
    tensor::{TensorView, TensorViewMut, VirtTensorView},
};

/// Matmul executor with optional residual add.
#[derive(Debug)]
pub struct MatmulExecutor {
    #[allow(unused)]
    config: MatmulConfig,
    graph: ExecutableCudnnGraph,
    a: TensorId,
    b: TensorId,
    residual: Option<TensorId>,
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
        residual: Option<&TensorView>,
        out: &mut TensorViewMut<'_>,
        stream: &Stream,
    ) -> anyhow::Result<()> {
        let &[m, k] = a.shape().dims_slice() else {
            panic!()
        };
        let &[_, n] = b.shape().dims_slice() else {
            panic!()
        };

        let mut inputs = vec![
            (self.a, a.reshaped([1, m, k])),
            (self.b, b.reshaped([1, k, n])),
        ];
        if let Some(residual) = residual {
            inputs.push((self.residual.unwrap(), residual.reshaped([1, m, n])));
        }
        self.graph.execute(
            &inputs
                .iter()
                .map(|(id, view)| (*id, view))
                .collect::<Vec<_>>(),
            &mut [(self.out, out.reshaped_mut([1, m, n]))],
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
    pub residual_format: Option<MatrixFormat>,
    pub out_format: MatrixFormat,
    pub allow_shape_override: bool,
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
        let residual = self
            .residual_format
            .map(|residual_format| graph.bind_tensor(&make_view(self.m, self.n, residual_format)));
        let mut out = graph.graph().matmul(
            a,
            b,
            &MatmulAttributes::new().compute_data_type(DataType::Float),
        )?;
        if let Some(residual) = residual {
            out = graph.graph().pointwise_binary(
                residual,
                out,
                &PointwiseAttributes::new(PointwiseMode::Add).compute_data_type(DataType::Float),
            )?;
        }
        let out = graph.mark_output(out, &make_view(self.m, self.n, self.out_format));
        let (a, b) = (a.uid(), b.uid());

        if self.allow_shape_override {
            graph.graph().set_override_shape_enabled(true);
        }

        Ok(MatmulExecutor {
            config: self.clone(),
            residual: residual.map(|r| r.uid()),
            graph: graph.build()?,
            a,
            b,
            out,
        })
    }
}
