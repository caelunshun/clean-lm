use crate::{
    context::{Context, ResourceKey},
    kernel::{RustKernels, rust_kernels_have_227kb_smem, rust_kernels_have_tma},
    tensor::{Tensor, TensorView, TensorViewMut},
};
use anyhow::bail;
use half::bf16;
use skycuda::{device_memory_pool::DeviceMemoryPool, module::Module, stream::Stream};
use skytens_core::{
    data_type::{ElementType, FloatType},
    shape::Shape,
};
use std::sync::Arc;

#[derive(Debug)]
pub struct NormExecutor {
    cx: Context,
    module: Arc<Module>,
}

/// Outputs of [`NormExecutor::rmsnorm_forward`].
#[derive(Debug)]
pub struct RmsNormForwardOutput {
    /// Normalized activations, same shape as `x`.
    pub y: Tensor,
    /// Per-row inverse RMS, saved for the backward pass.
    pub r_inv: Tensor,
}

impl NormExecutor {
    pub fn get(cx: &Context) -> anyhow::Result<Arc<Self>> {
        cx.get_or_create_resource(NormExecutorKey)
    }

    /// Computes `RMSNorm(x) * gamma`
    pub fn rmsnorm_forward_with_pool(
        &self,
        x: &TensorView<'_>,
        gamma: &TensorView<'_>,
        stream: &Stream,
        pool: &DeviceMemoryPool,
    ) -> anyhow::Result<RmsNormForwardOutput> {
        let &[batch_size, inner_dim] = x.shape().dims_slice() else {
            panic!("wrong shape for prev_layer_output");
        };

        check_bf16_matrix(x, batch_size, inner_dim);

        assert_eq!(gamma.shape().dims_slice(), &[inner_dim]);
        assert_eq!(gamma.element_type(), ElementType::Float(FloatType::Bf16));

        // Soundness: the kernel overwrites the data before we can read it.
        let mut out_y = unsafe {
            Tensor::new_uninit_with_pool(
                Shape::new([batch_size, inner_dim]),
                ElementType::Float(FloatType::Bf16),
                stream,
                &self.cx,
                pool,
            )?
        };
        // Soundness: the kernel overwrites the data before we can read it.
        let mut out_r_inv = unsafe {
            Tensor::new_uninit_with_pool(
                Shape::new([batch_size]),
                ElementType::Float(FloatType::F32),
                stream,
                &self.cx,
                pool,
            )?
        };

        let KernelSpec {
            name: kernel_name,
            cta_threads,
            dynamic_smem,
        } = fwd_kernel_spec(inner_dim, &self.cx)?;
        let kernel = self.module.get_function(kernel_name)?;
        kernel.set_max_dynamic_smem(dynamic_smem)?;

        let occupancy = stream.device().sm_count()
            * kernel.max_active_blocks_per_sm(cta_threads, dynamic_smem)?;

        {
            let x_memory = x.storage();
            let gamma_memory = gamma.storage();
            let out_y_memory = out_y.storage_mut();
            let out_r_inv_memory = out_r_inv.storage_mut();

            let mut launch = stream.launch_builder(&kernel);
            launch
                .arg(x_memory)?
                .arg(gamma_memory)?
                .arg(out_y_memory)?
                .arg(out_r_inv_memory)?
                .arg(i32::try_from(batch_size)?)?
                .block_size(cta_threads, 1, 1)?
                .dynamic_smem(dynamic_smem)?;

            unsafe {
                launch.launch(occupancy as u32, 1, 1)?;
            }
        }

        Ok(RmsNormForwardOutput {
            y: out_y,
            r_inv: out_r_inv,
        })
    }

    pub fn rmsnorm_forward(
        &self,
        x: &TensorView<'_>,
        gamma: &TensorView<'_>,
        stream: &Stream,
    ) -> anyhow::Result<RmsNormForwardOutput> {
        self.rmsnorm_forward_with_pool(x, gamma, stream, self.cx.device_memory_pool())
    }

    /// Backward pass for [`Self::rmsnorm_forward`], returning `dx`.
    ///
    /// `out_dgamma` is accumulated into by the kernel, so it must be pre-zeroed.
    #[allow(clippy::too_many_arguments)]
    pub fn rmsnorm_backward_with_pool(
        &self,
        x: &TensorView<'_>,
        dy: &TensorView<'_>,
        residual_gradient_stream: Option<&TensorView<'_>>,
        gamma: &TensorView<'_>,
        r_inv: &TensorView<'_>,
        out_dgamma: &mut TensorViewMut<'_>,
        stream: &Stream,
        pool: &DeviceMemoryPool,
    ) -> anyhow::Result<Tensor> {
        let &[batch_size, inner_dim] = x.shape().dims_slice() else {
            panic!("wrong shape for prev_layer_output");
        };

        check_bf16_matrix(x, batch_size, inner_dim);
        check_bf16_matrix(dy, batch_size, inner_dim);
        if let Some(residual_gradient_stream) = residual_gradient_stream {
            check_bf16_matrix(residual_gradient_stream, batch_size, inner_dim);
        }

        assert_eq!(gamma.shape().dims_slice(), &[inner_dim]);
        assert_eq!(gamma.element_type(), ElementType::Float(FloatType::Bf16));

        assert_eq!(r_inv.element_type(), ElementType::Float(FloatType::F32));
        assert_eq!(r_inv.shape(), Shape::new([batch_size]));
        assert!(r_inv.is_packed_row_major());

        assert_eq!(
            out_dgamma.element_type(),
            ElementType::Float(FloatType::F32)
        );
        assert_eq!(out_dgamma.shape(), Shape::new([inner_dim]));
        assert!(out_dgamma.is_packed_row_major());

        // Soundness: the kernel overwrites the data before we can read it.
        let mut out_dx = unsafe {
            Tensor::new_uninit_with_pool(
                Shape::new([batch_size, inner_dim]),
                ElementType::Float(FloatType::Bf16),
                stream,
                &self.cx,
                pool,
            )?
        };

        let KernelSpec {
            name: kernel_name,
            cta_threads,
            dynamic_smem,
        } = bwd_kernel_spec(inner_dim, &self.cx)?;
        let kernel = self.module.get_function(kernel_name)?;
        kernel.set_max_dynamic_smem(dynamic_smem)?;

        let occupancy = stream.device().sm_count()
            * kernel.max_active_blocks_per_sm(cta_threads, dynamic_smem)?;

        {
            let x_memory = x.storage();
            let dy_memory = dy.storage();
            let residual_gradient_stream_memory = residual_gradient_stream.map(|t| t.storage());
            let gamma_memory = gamma.storage();
            let r_inv_memory = r_inv.storage();
            let out_dx_memory = out_dx.storage_mut();
            let out_dgamma_memory = out_dgamma.storage_mut();

            let mut launch = stream.launch_builder(&kernel);
            launch.arg(x_memory)?;
            launch.arg(dy_memory)?;
            match residual_gradient_stream_memory {
                Some(memory) => launch.arg(memory)?,
                None => launch.arg(NULL_POINTER)?,
            };
            launch
                .arg(gamma_memory)?
                .arg(r_inv_memory)?
                .arg(out_dx_memory)?
                .arg(out_dgamma_memory)?
                .arg(i32::try_from(batch_size)?)?
                .dynamic_smem(dynamic_smem)?
                .block_size(cta_threads, 1, 1)?;
            // Block dim and smem are overriden by TileIR.
            unsafe {
                launch.launch(occupancy as u32, 1, 1)?;
            }
        }

        Ok(out_dx)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rmsnorm_backward(
        &self,
        x: &TensorView<'_>,
        dy: &TensorView<'_>,
        residual_gradient_stream: Option<&TensorView<'_>>,
        gamma: &TensorView<'_>,
        r_inv: &TensorView<'_>,
        out_dgamma: &mut TensorViewMut<'_>,
        stream: &Stream,
    ) -> anyhow::Result<Tensor> {
        self.rmsnorm_backward_with_pool(
            x,
            dy,
            residual_gradient_stream,
            gamma,
            r_inv,
            out_dgamma,
            stream,
            self.cx.device_memory_pool(),
        )
    }
}

const NULL_POINTER: u64 = 0;

struct KernelSpec {
    name: &'static str,
    cta_threads: u32,
    dynamic_smem: usize,
}

/// Prefetch stage count used by the norm kernels in `rust-kernels/src/norm.rs`.
fn prefetch_stages(cx: &Context) -> usize {
    if rust_kernels_have_227kb_smem(cx) {
        3
    } else {
        1
    }
}

fn fwd_kernel_spec(inner_dim: usize, cx: &Context) -> anyhow::Result<KernelSpec> {
    match inner_dim {
        2048 => Ok(KernelSpec {
            name: "rmsnorm_bf16_warp_persistent_inner2048",
            cta_threads: 384,
            dynamic_smem: fwd_required_smem(2048, prefetch_stages(cx), 12, cx),
        }),
        1536 => Ok(KernelSpec {
            name: "rmsnorm_bf16_warp_persistent_inner1536",
            cta_threads: 384,
            dynamic_smem: fwd_required_smem(1536, prefetch_stages(cx), 12, cx),
        }),
        dim => bail!("unimplemented inner dimension {dim} for rms prenorm forward"),
    }
}

fn bwd_kernel_spec(inner_dim: usize, cx: &Context) -> anyhow::Result<KernelSpec> {
    match inner_dim {
        2048 => Ok(KernelSpec {
            name: "rmsnorm_bwd_bf16_cta_persistent_inner2048",
            cta_threads: 256,
            dynamic_smem: bwd_required_smem(2048, 4, prefetch_stages(cx), 8),
        }),
        1536 => Ok(KernelSpec {
            name: "rmsnorm_bwd_bf16_cta_persistent_inner1536",
            cta_threads: 64,
            dynamic_smem: bwd_required_smem(1536, 4, prefetch_stages(cx), 2),
        }),
        dim => bail!("unimplemented inner dimension {dim} for rms prenorm backward"),
    }
}

const MBARRIER_SIZE: usize = 8;

/// Must match the SMEM calculation in `rmsnorm_bf16_warp_persistent`.
fn fwd_required_smem(
    vector_size: usize,
    prefetch_stages: usize,
    num_warps: usize,
    cx: &Context,
) -> usize {
    let num_mbarriers = if rust_kernels_have_tma(cx) {
        prefetch_stages * num_warps
    } else {
        0
    };
    let store_staging_space = if rust_kernels_have_227kb_smem(cx) {
        size_of::<bf16>() * vector_size * num_warps
    } else {
        0
    };
    size_of::<bf16>() * vector_size
        + size_of::<bf16>() * (vector_size * prefetch_stages * num_warps)
        + num_mbarriers * MBARRIER_SIZE
        + store_staging_space
}

/// Must match the SMEM calculation in `rmsnorm_bwd_bf16_cta_persistent`.
fn bwd_required_smem(
    vector_size: usize,
    rows_per_iter: usize,
    prefetch_stages: usize,
    num_warps: usize,
) -> usize {
    (4 * size_of::<bf16>() * vector_size * rows_per_iter * prefetch_stages)
        + (size_of::<f32>() * rows_per_iter * prefetch_stages)
        + (size_of::<f32>() * rows_per_iter * num_warps)
}

fn check_bf16_matrix(tensor: &TensorView<'_>, rows: usize, cols: usize) {
    assert_eq!(tensor.element_type(), ElementType::Float(FloatType::Bf16));
    assert_eq!(tensor.shape(), Shape::new([rows, cols]));
    assert!(tensor.is_packed_row_major());
}

#[derive(Debug, serde::Serialize)]
struct NormExecutorKey;

impl ResourceKey for NormExecutorKey {
    type Resource = NormExecutor;

    fn create(&self, cx: &Context) -> anyhow::Result<Self::Resource> {
        Ok(NormExecutor {
            cx: cx.clone(),
            module: cx.get_or_create_resource(RustKernels)?,
        })
    }
}
