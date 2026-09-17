use std::sync::Arc;

use anyhow::bail;
use skycuda::{module::Module, stream::Stream};
use skytens_core::{
    data_type::{ElementType, FloatType},
    shape::Shape,
};

use crate::{
    context::{Context, ResourceKey},
    include_cubins,
    tensor::{Tensor, TensorView, TensorViewMut},
};

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
    pub fn rmsnorm_forward(
        &self,
        x: &TensorView<'_>,
        gamma: &TensorView<'_>,
        stream: &Stream,
    ) -> anyhow::Result<RmsNormForwardOutput> {
        let &[batch_size, inner_dim] = x.shape().dims_slice() else {
            panic!("wrong shape for prev_layer_output");
        };

        check_bf16_matrix(x, batch_size, inner_dim);

        assert_eq!(gamma.shape().dims_slice(), &[inner_dim]);
        assert_eq!(gamma.element_type(), ElementType::Float(FloatType::Bf16));

        let mut out_y = Tensor::new_zeroed(
            Shape::new([batch_size, inner_dim]),
            ElementType::Float(FloatType::Bf16),
            &self.cx,
        )?;
        let mut out_r_inv = Tensor::new_zeroed(
            Shape::new([batch_size]),
            ElementType::Float(FloatType::F32),
            &self.cx,
        )?;

        let kernel_name = fwd_kernel_name(inner_dim)?;
        let kernel = self.module.get_function(kernel_name)?;

        let occupancy = kernel.max_active_blocks_per_sm(1, 0)? * stream.device().sm_count();

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
                .arg(i32::try_from(batch_size)?)?;

            unsafe {
                launch.launch(occupancy as u32, 1, 1)?;
            }
        }

        Ok(RmsNormForwardOutput {
            y: out_y,
            r_inv: out_r_inv,
        })
    }

    /// Backward pass for [`Self::rmsnorm_forward`], returning `dx`.
    ///
    /// `out_dgamma` is accumulated into by the kernel, so it must be pre-zeroed.
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

        let mut out_dx = Tensor::new_zeroed(
            Shape::new([batch_size, inner_dim]),
            ElementType::Float(FloatType::Bf16),
            &self.cx,
        )?;

        let kernel_name = bwd_kernel_name(inner_dim)?;
        let kernel = self.module.get_function(kernel_name)?;

        let occupancy = kernel.max_active_blocks_per_sm(1, 0)? * stream.device().sm_count();

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
                .arg(i32::try_from(batch_size)?)?;
            // Block dim and smem are overriden by TileIR.
            unsafe {
                launch.launch(occupancy as u32, 1, 1)?;
            }
        }

        Ok(out_dx)
    }
}

const NULL_POINTER: u64 = 0;

fn fwd_kernel_name(inner_dim: usize) -> anyhow::Result<&'static str> {
    match inner_dim {
        2048 => Ok("rmsnorm_fwd_inner2048_persistent"),
        dim => bail!("unimplemented inner dimension {dim} for rms prenorm forward"),
    }
}

fn bwd_kernel_name(inner_dim: usize) -> anyhow::Result<&'static str> {
    match inner_dim {
        2048 => Ok("rmsnorm_bwd_inner2048_persistent"),
        dim => bail!("unimplemented inner dimension {dim} for rms prenorm backward"),
    }
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
            module: cx.get_or_create_resource(include_cubins!("../../cutile-kernels/norm"))?,
        })
    }
}
