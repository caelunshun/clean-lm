use std::sync::Arc;

use serde::Serialize;
use skycuda::{
    function::Function,
    launch::LaunchBuilder,
    stream::Stream,
    tensor_map::{
        TensorMap, TensorMapDataType, TensorMapDesc, TensorMapL2Promotion, TensorMapMut,
        TensorMapOobFill, TensorMapSwizzle,
    },
};
use skytens_core::data_type::{ElementType, FloatType};

use crate::{
    context::{Context, ResourceKey},
    include_cubins,
    kernel::RustKernels,
    operators::matmul::{MatmulConfig, MatmulExecutor, MatrixFormat},
    tensor::{Tensor, TensorView},
};

const BF16: ElementType = ElementType::Float(FloatType::Bf16);

// Must match the constants in `rust-kernels/src/sky_matmul/tcgen05_bf16.rs`.
const NATIVE_TM_CGA: usize = 256;
const NATIVE_TN_CGA: usize = 256;
const NATIVE_CGA_SIZE: u32 = 2;
const NATIVE_CTA_THREADS: u32 = 384;

/// Executes custom matrix multiplications with high-performance fused epilogues,
/// with kernels implemented in `rust-kernels` crate. Currently only supports SM10x
/// for the fast implementation; other architectures will fall back to standard
/// MatmulExecutor + cuTile unfused epilogue implementation.
///
/// If plain GEMMs are needed, `MatmulExecutor` should be preferred
/// over this implementation, as cublasLt exposes many
/// more tuning options to adapt to various problem sizes.
#[derive(Debug)]
pub struct SkyMatmulExecutor<E: Epilogue> {
    config: SkyMatmulConfig<E>,
    backend: Backend,
    cx: Context,
}

impl<E: Epilogue> SkyMatmulExecutor<E> {
    pub fn get(config: SkyMatmulConfig<E>, cx: &Context) -> anyhow::Result<Arc<Self>> {
        cx.get_or_create_resource(config)
    }

    pub fn is_native(&self) -> bool {
        matches!(self.backend, Backend::Native(_))
    }

    /// Computes `epilogue(a @ b)`, where `a` is `[m, k]` and `b` is `[k, n]`.
    pub fn execute(
        &self,
        a: &TensorView,
        b: &TensorView,
        epilogue_args: E::Args<'_>,
        stream: &Stream,
    ) -> anyhow::Result<E::Outputs> {
        let &[m, k] = a.shape().dims_slice() else {
            panic!("A must be a matrix");
        };
        let &[b_k, n] = b.shape().dims_slice() else {
            panic!("B must be a matrix");
        };
        assert_eq!(k, b_k, "inner dimension mismatch");

        match &self.backend {
            Backend::Native(native) => {
                validate_matrix(a, m, k, self.config.format_a);
                validate_matrix(b, k, n, self.config.format_b);
                self.execute_native(native, a, b, m, n, k, epilogue_args, stream)
            }
            Backend::Fallback { matmul, epilogue } => {
                let product = matmul.matmul(a, b, None, stream)?;
                let mut outputs = self
                    .config
                    .epilogue
                    .alloc_fallback_outputs(product, &self.cx)?;
                if let Some(epilogue_kernel) = epilogue {
                    let mut launch = stream.launch_builder(epilogue_kernel);
                    let (grid_x, grid_y, grid_z) =
                        self.config.epilogue.pass_args_to_fallback_epilogue_kernel(
                            epilogue_args,
                            &mut outputs,
                            &mut launch,
                        )?;
                    unsafe {
                        launch.launch(grid_x, grid_y, grid_z)?;
                    }
                }
                Ok(outputs)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_native(
        &self,
        native: &NativeBackend,
        a: &TensorView,
        b: &TensorView,
        m: usize,
        n: usize,
        k: usize,
        epilogue_args: E::Args<'_>,
        stream: &Stream,
    ) -> anyhow::Result<E::Outputs> {
        assert!(m > 0 && n > 0 && k > 0, "empty matmuls are unsupported");
        assert!(k.is_multiple_of(8), "K must be a multiple of 8");
        assert!(n.is_multiple_of(8), "N must be a multiple of 8");
        assert!(m.is_multiple_of(8));

        let (m_u32, n_u32, k_u32) = (u32::try_from(m)?, u32::try_from(n)?, u32::try_from(k)?);
        let (raster_along_m, cga_panel_length) =
            choose_raster_params(m_u32, n_u32, k_u32, &self.cx);

        // Every CGA must receive at least one work tile.
        let num_tiles = m.div_ceil(NATIVE_TM_CGA) * n.div_ceil(NATIVE_TN_CGA);
        let max_cgas = (self.cx.cuda().sm_count() / NATIVE_CGA_SIZE as usize).max(1);
        let num_cgas = u32::try_from(num_tiles.min(max_cgas))?;

        let mut outputs = self.config.epilogue.alloc_outputs(m, n, &self.cx)?;
        {
            let a_tma = TensorMap::new(
                a.storage(),
                bf16_tma_desc(m, k, self.config.format_a == MatrixFormat::RowMajor)
                    .l2_promotion(TensorMapL2Promotion::B256),
            );
            let b_tma = TensorMap::new(
                b.storage(),
                bf16_tma_desc(n, k, self.config.format_b == MatrixFormat::ColumnMajor)
                    .l2_promotion(TensorMapL2Promotion::B256),
            );

            let mut launch = stream.launch_builder(&native.kernel);
            launch
                .arg(m_u32)?
                .arg(n_u32)?
                .arg(k_u32)?
                .arg(u8::from(raster_along_m))?
                .arg(cga_panel_length)?
                .arg(a_tma)?
                .arg(b_tma)?;
            self.config.epilogue.pass_args_to_native_kernel(
                epilogue_args,
                &mut outputs,
                &mut launch,
            )?;
            launch
                .block_size(NATIVE_CTA_THREADS, 1, 1)?
                .dynamic_smem(native.dynamic_smem)?;
            unsafe {
                launch.launch(num_cgas * NATIVE_CGA_SIZE, 1, 1)?;
            }
        }

        Ok(outputs)
    }
}

fn validate_matrix(tensor: &TensorView, rows: usize, cols: usize, format: MatrixFormat) {
    assert_eq!(tensor.element_type(), BF16);
    assert_eq!(tensor.shape().dims_slice(), &[rows, cols]);
    match format {
        MatrixFormat::RowMajor => assert_eq!(tensor.strides().dims_slice(), &[cols, 1]),
        MatrixFormat::ColumnMajor => assert_eq!(tensor.strides().dims_slice(), &[1, rows]),
    }
}

fn bf16_tma_desc(rows: usize, cols: usize, is_k_major: bool) -> TensorMapDesc {
    if is_k_major {
        TensorMapDesc::new(
            TensorMapDataType::BF16,
            &[cols as u64, rows as u64],
            &[cols as u64 * 2],
            &[64, 128],
        )
    } else {
        TensorMapDesc::new(
            TensorMapDataType::BF16,
            &[rows as u64, cols as u64],
            &[rows as u64 * 2],
            &[64, 64],
        )
    }
    .swizzle(TensorMapSwizzle::B128)
    .oob_fill(TensorMapOobFill::Zero)
}

#[derive(Debug)]
enum Backend {
    Native(NativeBackend),
    Fallback {
        matmul: Arc<MatmulExecutor>,
        epilogue: Option<Function>,
    },
}

#[derive(Debug)]
struct NativeBackend {
    kernel: Function,
    dynamic_smem: usize,
}

pub trait Epilogue: Serialize + Clone + Send + Sync + 'static {
    type Args<'a>;
    type Outputs;

    #[doc(hidden)]
    fn native_kernel_name(&self) -> &'static str;
    #[doc(hidden)]
    fn alloc_outputs(&self, m: usize, n: usize, cx: &Context) -> anyhow::Result<Self::Outputs>;
    #[doc(hidden)]
    fn fallback_epilogue_kernel(&self, cx: &Context) -> anyhow::Result<Option<Function>>;
    /// Allocates the fallback outputs, possibly reusing the row-major `[m, n]` bf16
    /// matmul product. The outputs are returned as-is if there is no fallback epilogue kernel.
    #[doc(hidden)]
    fn alloc_fallback_outputs(
        &self,
        product: Tensor,
        cx: &Context,
    ) -> anyhow::Result<Self::Outputs>;
    /// Called after the common arguments (m, n, k, raster params, A/B descriptors)
    /// have been passed.
    #[doc(hidden)]
    fn pass_args_to_native_kernel<'a: 'b, 'b>(
        &self,
        args: Self::Args<'a>,
        outputs: &'b mut Self::Outputs,
        builder: &mut LaunchBuilder<'b>,
    ) -> anyhow::Result<()>;
    /// Must pass all kernel arguments (the matmul product is held in `outputs`),
    /// set the block size if needed, and return the grid size.
    #[doc(hidden)]
    fn pass_args_to_fallback_epilogue_kernel<'a: 'b, 'b>(
        &self,
        args: Self::Args<'a>,
        outputs: &'b mut Self::Outputs,
        builder: &mut LaunchBuilder<'b>,
    ) -> anyhow::Result<(u32, u32, u32)>;
}

#[derive(Debug, Clone, Serialize)]
pub struct SkyMatmulConfig<E: Epilogue> {
    pub tune_for_m: u32,
    pub tune_for_n: u32,
    pub tune_for_k: u32,
    pub format_a: MatrixFormat,
    pub format_b: MatrixFormat,
    pub epilogue: E,
}

impl<E: Epilogue> ResourceKey for SkyMatmulConfig<E> {
    type Resource = SkyMatmulExecutor<E>;

    fn create(&self, cx: &Context) -> anyhow::Result<Self::Resource> {
        let native_supported = cx.cuda().compute_capability_major() == 10;

        let backend = if native_supported {
            let module = cx.get_or_create_resource(RustKernels)?;

            let a_format_suffix = match self.format_a {
                MatrixFormat::RowMajor => "arow",
                MatrixFormat::ColumnMajor => "acol",
            };
            let b_format_suffix = match self.format_b {
                MatrixFormat::RowMajor => "brow",
                MatrixFormat::ColumnMajor => "bcol",
            };
            let kernel_name = format!(
                "{}_{a_format_suffix}_{b_format_suffix}",
                self.epilogue.native_kernel_name()
            );
            let kernel = module.get_function(&kernel_name)?;
            let dynamic_smem = cx.cuda().max_dynamic_smem_per_cta();
            kernel.set_max_dynamic_smem(dynamic_smem)?;
            Backend::Native(NativeBackend {
                kernel,
                dynamic_smem,
            })
        } else {
            let matmul = MatmulExecutor::get(
                MatmulConfig {
                    m: self.tune_for_m as usize,
                    n: self.tune_for_n as usize,
                    k: self.tune_for_k as usize,
                    a_format: self.format_a,
                    b_format: self.format_b,
                    residual: None,
                    out_format: MatrixFormat::RowMajor,
                    ab_element_type: BF16,
                    out_element_type: BF16,
                    allow_shape_override: true,
                },
                cx,
            )?;
            Backend::Fallback {
                matmul,
                epilogue: self.epilogue.fallback_epilogue_kernel(cx)?,
            }
        };

        Ok(SkyMatmulExecutor {
            config: self.clone(),
            backend,
            cx: cx.clone(),
        })
    }
}

/// Only for testing; prefer to use MatmulExecutor if no epilogue.
#[derive(Debug, Clone, Serialize)]
pub struct IdentityEpilogue;

impl Epilogue for IdentityEpilogue {
    type Args<'a> = ();
    type Outputs = Tensor;

    fn native_kernel_name(&self) -> &'static str {
        "sky_matmul_sm10x_128x256_2cta_64x6"
    }

    fn alloc_outputs(&self, m: usize, n: usize, cx: &Context) -> anyhow::Result<Self::Outputs> {
        Tensor::new_zeroed([m, n], BF16, cx)
    }

    fn fallback_epilogue_kernel(&self, _cx: &Context) -> anyhow::Result<Option<Function>> {
        Ok(None)
    }

    fn alloc_fallback_outputs(
        &self,
        product: Tensor,
        _cx: &Context,
    ) -> anyhow::Result<Self::Outputs> {
        Ok(product)
    }

    fn pass_args_to_native_kernel<'a: 'b, 'b>(
        &self,
        _args: Self::Args<'a>,
        outputs: &'b mut Self::Outputs,
        builder: &mut LaunchBuilder<'b>,
    ) -> anyhow::Result<()> {
        let &[m, n] = outputs.shape().dims_slice() else {
            unreachable!()
        };
        builder.arg(TensorMapMut::new(
            outputs.storage_mut(),
            bf16_tma_desc(m, n, true),
        ))?;
        Ok(())
    }

    fn pass_args_to_fallback_epilogue_kernel<'a: 'b, 'b>(
        &self,
        _args: Self::Args<'a>,
        _outputs: &'b mut Self::Outputs,
        _builder: &mut LaunchBuilder<'b>,
    ) -> anyhow::Result<(u32, u32, u32)> {
        unreachable!()
    }
}

/// SwiGLU gate projection for the training forward pass. Outputs the pre-activation
/// gate `G = A @ B` and the activation `Y = swish(G) * X`, where `X` is the result of up-projection.
#[derive(Debug, Clone, Serialize)]
pub struct SwiGluGateTrainFwdEpilogue;

#[derive(Debug)]
pub struct SwiGluGateTrainFwdOutputs {
    /// Gate projection, before the activation.
    pub gate: Tensor,
    /// `swish(gate) * x`.
    pub activation: Tensor,
}

impl Epilogue for SwiGluGateTrainFwdEpilogue {
    /// The up projection `x`, a packed row-major bf16 `[m, n]` tensor.
    type Args<'a> = &'a TensorView<'a>;
    type Outputs = SwiGluGateTrainFwdOutputs;

    fn native_kernel_name(&self) -> &'static str {
        "sky_matmul_swiglu_gate_train_fwd_sm10x_128x256_2cta_64x6_epi1"
    }

    fn alloc_outputs(&self, m: usize, n: usize, cx: &Context) -> anyhow::Result<Self::Outputs> {
        Ok(SwiGluGateTrainFwdOutputs {
            gate: Tensor::new_zeroed([m, n], BF16, cx)?,
            activation: Tensor::new_zeroed([m, n], BF16, cx)?,
        })
    }

    fn fallback_epilogue_kernel(&self, cx: &Context) -> anyhow::Result<Option<Function>> {
        let module =
            cx.get_or_create_resource(include_cubins!("../../cutile-kernels/sky_matmul_fallback"))?;
        Ok(Some(
            module.get_function("swiglu_gate_train_fwd_bf16_4096")?,
        ))
    }

    fn alloc_fallback_outputs(
        &self,
        product: Tensor,
        cx: &Context,
    ) -> anyhow::Result<Self::Outputs> {
        let activation = Tensor::new_zeroed(product.shape(), BF16, cx)?;
        Ok(SwiGluGateTrainFwdOutputs {
            gate: product,
            activation,
        })
    }

    fn pass_args_to_native_kernel<'a: 'b, 'b>(
        &self,
        x: Self::Args<'a>,
        outputs: &'b mut Self::Outputs,
        builder: &mut LaunchBuilder<'b>,
    ) -> anyhow::Result<()> {
        let &[m, n] = outputs.gate.shape().dims_slice() else {
            unreachable!()
        };
        validate_matrix(x, m, n, MatrixFormat::RowMajor);
        let SwiGluGateTrainFwdOutputs { gate, activation } = outputs;
        builder
            .arg(TensorMap::new(
                x.storage(),
                bf16_tma_desc(m, n, true).l2_promotion(TensorMapL2Promotion::B256),
            ))?
            .arg(TensorMapMut::new(
                gate.storage_mut(),
                bf16_tma_desc(m, n, true),
            ))?
            .arg(TensorMapMut::new(
                activation.storage_mut(),
                bf16_tma_desc(m, n, true),
            ))?;
        Ok(())
    }

    fn pass_args_to_fallback_epilogue_kernel<'a: 'b, 'b>(
        &self,
        x: Self::Args<'a>,
        outputs: &'b mut Self::Outputs,
        builder: &mut LaunchBuilder<'b>,
    ) -> anyhow::Result<(u32, u32, u32)> {
        let &[m, n] = outputs.gate.shape().dims_slice() else {
            unreachable!()
        };
        validate_matrix(x, m, n, MatrixFormat::RowMajor);
        let count = m * n;
        let SwiGluGateTrainFwdOutputs { gate, activation } = outputs;
        builder
            .arg(x.storage())?
            .arg(gate.storage())?
            .arg(activation.storage_mut())?
            .arg(u32::try_from(count)?)?;
        Ok((u32::try_from(count.div_ceil(4096))?, 1, 1))
    }
}

/// Given
/// `A @ B = dActivation` (the gradient of `swish(gate) * x`), outputs `dx = swish(gate) * dActivation`
/// and `dgate = swish'(gate) * x * dActivation`.
#[derive(Debug, Clone, Serialize)]
pub struct SwiGluDownTrainDgradEpilogue;

#[derive(Debug, Clone, Copy)]
pub struct SwiGluDownTrainDgradArgs<'a> {
    pub gate: &'a TensorView<'a>,
    pub x: &'a TensorView<'a>,
}

#[derive(Debug)]
pub struct SwiGluDownTrainDgradOutputs {
    /// Gradient of the gate projection, before the activation.
    pub dgate: Tensor,
    /// Gradient of the up projection.
    pub dx: Tensor,
}

impl Epilogue for SwiGluDownTrainDgradEpilogue {
    type Args<'a> = SwiGluDownTrainDgradArgs<'a>;
    type Outputs = SwiGluDownTrainDgradOutputs;

    fn native_kernel_name(&self) -> &'static str {
        "sky_matmul_swiglu_down_train_dgrad_sm10x_128x256_2cta_64x6_epi1"
    }

    fn alloc_outputs(&self, m: usize, n: usize, cx: &Context) -> anyhow::Result<Self::Outputs> {
        Ok(SwiGluDownTrainDgradOutputs {
            dgate: Tensor::new_zeroed([m, n], BF16, cx)?,
            dx: Tensor::new_zeroed([m, n], BF16, cx)?,
        })
    }

    fn fallback_epilogue_kernel(&self, cx: &Context) -> anyhow::Result<Option<Function>> {
        let module =
            cx.get_or_create_resource(include_cubins!("../../cutile-kernels/sky_matmul_fallback"))?;
        Ok(Some(
            module.get_function("swiglu_down_train_dgrad_bf16_4096")?,
        ))
    }

    fn alloc_fallback_outputs(
        &self,
        product: Tensor,
        cx: &Context,
    ) -> anyhow::Result<Self::Outputs> {
        let dgate = Tensor::new_zeroed(product.shape(), BF16, cx)?;
        Ok(SwiGluDownTrainDgradOutputs { dgate, dx: product })
    }

    fn pass_args_to_native_kernel<'a: 'b, 'b>(
        &self,
        SwiGluDownTrainDgradArgs { gate, x }: Self::Args<'a>,
        outputs: &'b mut Self::Outputs,
        builder: &mut LaunchBuilder<'b>,
    ) -> anyhow::Result<()> {
        let &[m, n] = outputs.dx.shape().dims_slice() else {
            unreachable!()
        };
        validate_matrix(gate, m, n, MatrixFormat::RowMajor);
        validate_matrix(x, m, n, MatrixFormat::RowMajor);
        let SwiGluDownTrainDgradOutputs { dgate, dx } = outputs;
        builder
            .arg(TensorMap::new(
                gate.storage(),
                bf16_tma_desc(m, n, true).l2_promotion(TensorMapL2Promotion::B256),
            ))?
            .arg(TensorMap::new(
                x.storage(),
                bf16_tma_desc(m, n, true).l2_promotion(TensorMapL2Promotion::B256),
            ))?
            .arg(TensorMapMut::new(
                dx.storage_mut(),
                bf16_tma_desc(m, n, true),
            ))?
            .arg(TensorMapMut::new(
                dgate.storage_mut(),
                bf16_tma_desc(m, n, true),
            ))?;
        Ok(())
    }

    fn pass_args_to_fallback_epilogue_kernel<'a: 'b, 'b>(
        &self,
        SwiGluDownTrainDgradArgs { gate, x }: Self::Args<'a>,
        outputs: &'b mut Self::Outputs,
        builder: &mut LaunchBuilder<'b>,
    ) -> anyhow::Result<(u32, u32, u32)> {
        let &[m, n] = outputs.dx.shape().dims_slice() else {
            unreachable!()
        };
        validate_matrix(gate, m, n, MatrixFormat::RowMajor);
        validate_matrix(x, m, n, MatrixFormat::RowMajor);
        let count = m * n;
        let SwiGluDownTrainDgradOutputs { dgate, dx } = outputs;
        builder
            .arg(gate.storage())?
            .arg(x.storage())?
            .arg(dx.storage_mut())?
            .arg(dgate.storage_mut())?
            .arg(u32::try_from(count)?)?;
        Ok((u32::try_from(count.div_ceil(4096))?, 1, 1))
    }
}

/// Chooses (raster_along_m, cga_panel_length) for a problem size.
/// currently panel length is hardcoded, may use a smarter choice in the future.
fn choose_raster_params(m: u32, n: u32, _k: u32, _cx: &Context) -> (bool, u32) {
    if m > n { (false, 4) } else { (true, 4) }
}
