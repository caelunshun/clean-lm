use std::sync::Arc;

use rand::{RngExt, SeedableRng};
use rand_distr::Uniform;
use rand_xoshiro::Xoroshiro128StarStar;
use serde::{Deserialize, Serialize};
use skycuda::stream::Stream;
use skycuda_cublaslt::{
    Algorithm, ComputeType, DataType, MatmulDesc, MatmulPlan, MatmulPreference, MatrixLayout, Order,
};
use skytens_core::data_type::{ElementType, FloatType};

use crate::{
    context::{Context, ResourceKey},
    disk_cache::DiskCacheKey,
    par_random,
    tensor::{Tensor, TensorView},
};

/// Matmul executor with optional residual add.
#[derive(Debug)]
pub struct MatmulExecutor {
    config: MatmulConfig,
    cx: Context,
    plan: MatmulPlan,
    desc: MatmulDesc,
    algo: Algorithm,
}

impl MatmulExecutor {
    pub fn get(config: MatmulConfig, cx: &Context) -> anyhow::Result<Arc<Self>> {
        cx.get_or_create_resource(config)
    }

    /// Returns the product as a freshly allocated row-major bf16 tensor `[m, n]`.
    pub fn matmul(
        &self,
        a: &TensorView<'_>,
        b: &TensorView<'_>,
        residual: Option<&TensorView>,
        stream: &Stream,
    ) -> anyhow::Result<Tensor> {
        let validate_matrix = |tensor: &TensorView,
                               rows: usize,
                               cols: usize,
                               format: MatrixFormat,
                               ty: ElementType| {
            assert_eq!(tensor.element_type(), ty);
            assert_eq!(tensor.shape().dims_slice(), &[rows, cols]);
            match format {
                MatrixFormat::RowMajor => assert_eq!(tensor.strides().dims_slice(), &[cols, 1]),
                MatrixFormat::ColumnMajor => assert_eq!(tensor.strides().dims_slice(), &[1, rows]),
            }
        };

        let (m, n, k) = if self.config.allow_shape_override {
            (a.shape()[0], b.shape()[1], a.shape()[1])
        } else {
            (self.config.m, self.config.n, self.config.k)
        };
        validate_matrix(a, m, k, self.config.a_format, self.config.ab_element_type);
        validate_matrix(b, k, n, self.config.b_format, self.config.ab_element_type);

        match &self.config.residual {
            Some(residual_config) => {
                let residual = residual.expect("missing residual for matmul execute");
                validate_matrix(
                    residual,
                    m,
                    n,
                    residual_config.format,
                    residual_config.element_type,
                );
            }
            None => assert!(residual.is_none()),
        }

        let mut out = Tensor::new_zeroed([m, n], self.config.out_element_type, &self.cx)?;

        let reshaped_plan;
        let plan = if m != self.config.m || n != self.config.n || k != self.config.k {
            assert!(self.config.allow_shape_override);
            reshaped_plan = self.plan.reshape(&MatmulDesc {
                a: make_matrix_layout(m, k, self.config.a_format, self.config.ab_element_type),
                b: make_matrix_layout(k, n, self.config.b_format, self.config.ab_element_type),
                c: if let Some(residual) = &self.config.residual {
                    Some(make_matrix_layout(
                        m,
                        n,
                        residual.format,
                        residual.element_type,
                    ))
                } else {
                    None
                },
                d: make_matrix_layout(m, n, self.config.out_format, self.config.out_element_type),
                ..self.desc.clone()
            })?;
            &reshaped_plan
        } else {
            &self.plan
        };

        let mut workspace = self.cx.device_memory_pool().alloc_zeroed_on(
            plan.check(&self.algo)?.workspace_size,
            self.cx.memset_stream(),
        )?;

        let mut operands = plan.operands();
        operands.a(a.storage()).b(b.storage()).d(out.storage_mut());
        if let Some(residual) = residual {
            operands.c(residual.storage());
        }

        unsafe {
            plan.execute_with(&self.algo, stream, &operands, &mut workspace)?;
        }

        drop(operands);

        Ok(out)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct MatmulConfig {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub a_format: MatrixFormat,
    pub b_format: MatrixFormat,
    pub residual: Option<ResidualConfig>,
    pub out_format: MatrixFormat,
    pub ab_element_type: ElementType,
    pub out_element_type: ElementType,
    pub allow_shape_override: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResidualConfig {
    /// Currently, must match out_format
    pub format: MatrixFormat,
    pub element_type: ElementType,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize)]
pub enum MatrixFormat {
    RowMajor,
    ColumnMajor,
}

impl ResourceKey for MatmulConfig {
    type Resource = MatmulExecutor;

    fn create(&self, cx: &Context) -> anyhow::Result<Self::Resource> {
        let a_layout = make_matrix_layout(self.m, self.k, self.a_format, self.ab_element_type);
        let b_layout = make_matrix_layout(self.k, self.n, self.b_format, self.ab_element_type);
        let d_layout = make_matrix_layout(self.m, self.n, self.out_format, self.out_element_type);

        let mut desc = MatmulDesc::new(ComputeType::F32, a_layout, b_layout, d_layout);
        if let Some(residual) = &self.residual {
            desc = desc.c(make_matrix_layout(
                self.m,
                self.n,
                residual.format,
                self.out_element_type,
            ));
        }

        let plan = cx
            .cublaslt()
            .build_plan(&desc, &MatmulPreference::default().requested_algo_count(32))?;

        let algo = if cx.is_autotuning_enabled() {
            let index = autotune(&plan, self, cx)?;
            plan.algorithms()
                .get(index)
                .copied()
                .unwrap_or_else(|| plan.algorithms()[0])
        } else {
            plan.algorithms()[0]
        };

        Ok(MatmulExecutor {
            config: self.clone(),
            desc,
            cx: cx.clone(),
            plan,
            algo,
        })
    }
}

fn autotune(plan: &MatmulPlan, config: &MatmulConfig, cx: &Context) -> anyhow::Result<usize> {
    #[derive(Debug, Serialize, Deserialize)]
    struct MatmulAutotuningCacheKey {
        config: serde_json::Value,
        num_algorithms: usize,
    }

    #[typetag::serde]
    impl DiskCacheKey for MatmulAutotuningCacheKey {}

    cx.disk_cache().get_or_create(
        &MatmulAutotuningCacheKey {
            config: serde_json::to_value(config)?,
            num_algorithms: plan.algorithms().len(),
        },
        || {
            cx.cuda().set_as_current()?;

            let make_dummy = |rows: usize, cols: usize, element_type: ElementType| {
                let mut tensor = Tensor::new_zeroed([rows, cols], element_type, cx)?;
                let mut data = vec![0u8; tensor.storage().len()];
                par_random::par_fill_from_distr(
                    &Uniform::<u8>::new_inclusive(0, u8::MAX).unwrap(),
                    &mut data,
                    rng,
                );
                cx.upload_to_device(&data, tensor.storage_mut(), cx.h2d_copy_stream())?;
                anyhow::Ok(tensor)
            };

            let a = make_dummy(config.m, config.k, config.ab_element_type)?;
            let b = make_dummy(config.k, config.n, config.ab_element_type)?;
            let residual = config
                .residual
                .as_ref()
                .map(|residual| make_dummy(config.m, config.n, residual.element_type))
                .transpose()?;
            let mut out = make_dummy(config.m, config.n, config.out_element_type)?;
            cx.h2d_copy_stream().block_on()?;

            let mut workspace = cx
                .device_memory_pool()
                .alloc_zeroed_on(plan.max_workspace_size(), cx.memset_stream())?;

            let mut operands = plan.operands();
            operands.a(a.storage()).b(b.storage()).d(out.storage_mut());
            if let Some(residual) = &residual {
                operands.c(residual.storage());
            }

            let mut best_algo_idx = 0;
            let mut best_algo_time = None;
            for (i, algo) in plan.algorithms().iter().enumerate() {
                let name = format!("{:?}", algo.config()?);
                let time = crate::autotuning::measure_kernel(
                    |stream| unsafe {
                        plan.execute_with(algo, stream, &operands, &mut workspace)?;
                        Ok(())
                    },
                    &name,
                    cx,
                );
                let time = match time {
                    Ok(time) => time,
                    Err(e) => {
                        tracing::debug!(
                            "skipping matmul algorithm {name} due to execution failure: {e}"
                        );
                        continue;
                    }
                };
                if best_algo_time.is_none() || time < best_algo_time.unwrap() {
                    best_algo_idx = i;
                    best_algo_time = Some(time);
                }
            }
            tracing::info!(
                "winning matmul algorithm for shape {}x{}x{}: {:?} in {:.1?}",
                config.m,
                config.n,
                config.k,
                plan.algorithms()[best_algo_idx].config()?,
                best_algo_time.unwrap_or_default(),
            );

            drop(operands);

            Ok(best_algo_idx)
        },
    )
}

fn make_matrix_layout(
    rows: usize,
    cols: usize,
    format: MatrixFormat,
    dtype: ElementType,
) -> MatrixLayout {
    let dtype = match dtype {
        ElementType::Float(FloatType::Bf16) => DataType::BF16,
        ElementType::Float(FloatType::F16) => DataType::F16,
        ElementType::Float(FloatType::F32) => DataType::F32,
        _ => panic!("unsupported data type in matmul: {dtype:?}"),
    };

    MatrixLayout {
        data_type: dtype,
        rows: rows as u64,
        cols: cols as u64,
        ld: match format {
            MatrixFormat::RowMajor => cols as i64,
            MatrixFormat::ColumnMajor => rows as i64,
        },
        order: match format {
            MatrixFormat::RowMajor => Order::Row,
            MatrixFormat::ColumnMajor => Order::Col,
        },
        batch_count: 1,
        batch_stride: 0,
    }
}
