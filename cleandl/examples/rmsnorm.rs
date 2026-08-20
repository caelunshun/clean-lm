use cleandl::{
    context::{Context, CudaStream},
    operators::rmsnorm::{RmsNormConfig, RmsNormExecutor},
    tensor::TensorView,
};
use skytens_core::data_type::{ElementType, FloatType};

fn main() -> anyhow::Result<()> {
    let cx = Context::new(0)?;

    let rmsnorm = RmsNormExecutor::get(
        RmsNormConfig {
            batch_size: 1024 * 1024,
            token_dim: 1536,
        },
        &cx,
    )?;

    let x = TensorView::new_zeroed(
        [1024 * 1024, 1536],
        ElementType::Float(FloatType::Bf16),
        &cx,
    )?;
    let scale = TensorView::new_zeroed([1, 1536], ElementType::Float(FloatType::Bf16), &cx)?;
    let out_inv_variance =
        TensorView::new_zeroed([1024 * 1024, 1], ElementType::Float(FloatType::F32), &cx)?;
    let out_y = TensorView::new_zeroed(
        [1024 * 1024, 1536],
        ElementType::Float(FloatType::Bf16),
        &cx,
    )?;

    let stream = CudaStream::new(&cx)?;
    rmsnorm.forward(&x, &scale, &out_y, &out_inv_variance, &stream)?;

    cx.cuda().synchronize()?;

    Ok(())
}
