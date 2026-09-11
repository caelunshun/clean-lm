use cleandl::{
    context::Context,
    operators::rmsnorm::{RmsNormConfig, RmsNormExecutor},
    tensor::Tensor,
};
use skycuda::{
    device_memory_pool::DeviceMemoryPoolOptions,
    host_pinned_memory_pool::HostPinnedMemoryPoolOptions,
};
use skytens_core::data_type::{ElementType, FloatType};

#[test]
fn rmsnorm() -> anyhow::Result<()> {
    let cx = Context::new(
        0,
        DeviceMemoryPoolOptions {
            max_reserved_free_bytes: 1 << 30,
        },
        HostPinnedMemoryPoolOptions {
            max_reserved_free_bytes: 1 << 30,
        },
    )?;
    let stream = cx.cuda().create_stream()?;

    let rmsnorm = RmsNormExecutor::get(
        RmsNormConfig {
            batch_size: 1024 * 1024,
            token_dim: 1536,
        },
        &cx,
    )?;

    let x = Tensor::new_zeroed(
        [1024 * 1024, 1536],
        ElementType::Float(FloatType::Bf16),
        &stream,
        &cx,
    )?;
    let scale = Tensor::new_zeroed([1, 1536], ElementType::Float(FloatType::Bf16), &stream, &cx)?;
    let mut out_inv_variance = Tensor::new_zeroed(
        [1024 * 1024, 1],
        ElementType::Float(FloatType::F32),
        &stream,
        &cx,
    )?;
    let mut out_y = Tensor::new_zeroed(
        [1024 * 1024, 1536],
        ElementType::Float(FloatType::Bf16),
        &stream,
        &cx,
    )?;

    rmsnorm.forward(
        &x.view(),
        &scale.view(),
        &mut out_y.view_mut(),
        &mut out_inv_variance.view_mut(),
        &stream,
    )?;

    stream.block_on()?;

    Ok(())
}
