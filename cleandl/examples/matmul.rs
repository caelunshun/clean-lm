
use cleandl::{cudnn::CudnnGraphBuilder, tensor::TensorView};
use singe_cuda::context::Context;
use singe_cudnn::data_type::DataType;
use skytens_core::{data_type::{ElementType, FloatType}, shape::Shape};

fn main() -> anyhow::Result<()> {
    let cx = Context::create()?;

    let a = TensorView::new_zeroed(Shape::new([1, 16384, 4096]), ElementType::Float(FloatType::Bf16), &cx)?;
    let b = TensorView::new_zeroed(Shape::new([1, 4096, 8192]), ElementType::Float(FloatType::Bf16), &cx)?;
    let c  = TensorView::new_zeroed(Shape::new([1, 16384, 8192]), ElementType::Float(FloatType::Bf16), &cx)?;

    let mut graph = CudnnGraphBuilder::new(&cx);
    let a_id = graph.bind_tensor(a.virt());
    let b_id = graph.bind_tensor(b.virt());
    let c_id = graph.bind_tensor(c.virt());
    graph.graph_mut().matmul(a_id, b_id, c_id, DataType::BF16)?;
    let mut graph = graph.build()?;

    graph.execute([(a_id, &a), (b_id, &b), (c_id, &c)], & cx.create_stream()?)?;

    cx.synchronize()?;

    Ok(())
}