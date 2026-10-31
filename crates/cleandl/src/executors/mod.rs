use crate::{context::Context, tensor::TensorView};
use anyhow::{bail, ensure};
use skycuda::{device_memory_pool::DeviceMemoryPool, stream::Stream};
use skytens_core::{data_type::ElementType, shape::Shape};

pub mod norm_fwd;

pub trait Executor: Send + Sync + 'static {
    type Args<'a>;
    type Output;

    fn execute_with_pool<'a>(
        &self,
        args: Self::Args<'a>,
        stream: &Stream,
        pool: &DeviceMemoryPool,
        cx: &Context,
    ) -> anyhow::Result<Self::Output>;

    fn execute<'a>(
        &self,
        args: Self::Args<'a>,
        stream: &Stream,
        cx: &Context,
    ) -> anyhow::Result<Self::Output> {
        self.execute_with_pool(args, stream, cx.device_memory_pool(), cx)
    }
}

pub fn ensure_element_type(
    name: &str,
    tensor: &TensorView,
    expected: impl Into<ElementType>,
) -> anyhow::Result<()> {
    let expected = expected.into();
    ensure!(
        tensor.element_type() == expected,
        "{name}: expected element type {expected:?}, got {:?}",
        tensor.element_type()
    );
    Ok(())
}

pub fn ensure_shape(
    name: &str,
    tensor: &TensorView,
    expected: impl Into<Shape>,
) -> anyhow::Result<()> {
    let expected = expected.into();
    ensure!(
        tensor.shape() == expected,
        "{name}: expected shape {expected:?}, got {:?}",
        tensor.shape()
    );
    Ok(())
}

pub fn ensure_min_dims(name: &str, tensor: &TensorView, min_dims: usize) -> anyhow::Result<()> {
    ensure!(
        tensor.shape().num_dims() >= min_dims,
        "{name}: expected at least {min_dims} dimensions, got shape {:?}",
        tensor.shape()
    );
    Ok(())
}

pub fn ensure_nonempty(name: &str, tensor: &TensorView) -> anyhow::Result<()> {
    ensure!(
        tensor.shape().element_count() > 0,
        "{name}: tensor must be nonempty, got shape {:?}",
        tensor.shape()
    );
    Ok(())
}

pub fn ensure_packed_row_major(name: &str, tensor: &TensorView) -> anyhow::Result<()> {
    ensure!(
        tensor.is_packed_row_major(),
        "{name}: expected packed row-major layout, got shape {:?} with strides {:?}",
        tensor.shape(),
        tensor.strides()
    );
    Ok(())
}

/// Ensures the tensor's base pointer is aligned to `align_bytes`, assuming
/// the underlying allocation is at least that aligned.
pub fn ensure_base_aligned(
    name: &str,
    tensor: &TensorView,
    align_bytes: usize,
) -> anyhow::Result<()> {
    ensure!(
        tensor.storage().offset().is_multiple_of(align_bytes),
        "{name}: base pointer must be aligned to {align_bytes} bytes (offset is {})",
        tensor.storage().offset()
    );
    Ok(())
}

/// Ensures each row (innermost dimension) of the tensor spans a multiple of `align_bytes`.
pub fn ensure_row_bytes_aligned(
    name: &str,
    tensor: &TensorView,
    align_bytes: usize,
) -> anyhow::Result<()> {
    let Some(row_size) = tensor.shape().dims().last() else {
        bail!("{name}: tensor must have at least one dimension");
    };
    let row_bits = row_size * tensor.element_type().bits_per_element();
    ensure!(
        row_bits.is_multiple_of(align_bytes * 8),
        "{name}: row size {row_size} of {:?} must be a multiple of {align_bytes} bytes",
        tensor.element_type()
    );
    Ok(())
}

pub fn ensure_divisible(name: &str, value: usize, divisor: usize) -> anyhow::Result<()> {
    ensure!(
        value.is_multiple_of(divisor),
        "{name} ({value}) must be a multiple of {divisor}"
    );
    Ok(())
}

pub fn ensure_option_presence<T>(
    name: &str,
    value: &Option<T>,
    expected_present: bool,
) -> anyhow::Result<()> {
    match (value.is_some(), expected_present) {
        (true, false) => bail!("{name} was provided, but the executor does not accept it"),
        (false, true) => bail!("{name} is required by the executor, but was not provided"),
        _ => Ok(()),
    }
}

pub fn to_u32(name: &str, value: usize) -> anyhow::Result<u32> {
    u32::try_from(value).map_err(|_| anyhow::anyhow!("{name} ({value}) does not fit in u32"))
}
