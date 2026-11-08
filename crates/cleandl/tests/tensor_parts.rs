mod common;

use cleandl::{
    mx_tensor::{MxDim, MxFormat, MxLayout, MxTensor, MxTensorView},
    tensor::{Tensor, TensorView, VirtTensorView},
};
use common::test_context;
use skytens_core::{
    data_type::{ElementType, FloatType},
    shape::Strides,
};

const F32: ElementType = ElementType::Float(FloatType::F32);

#[test]
fn tensor_from_parts() -> anyhow::Result<()> {
    let cx = test_context();
    let stream = cx.default_compute_stream();
    let storage = cx.device_memory_pool().alloc_zeroed_on(4096, stream)?;

    let virt = VirtTensorView::with_strides([8, 4], Strides::new([1, 16]), F32);
    let view = TensorView::from_parts(virt.clone(), storage.slice(256..), &cx);
    assert_eq!(view.storage().offset(), 256);
    assert_eq!(view.strides(), virt.strides());

    let tensor = Tensor::from_parts(VirtTensorView::new([32, 32], F32), storage, &cx);
    assert!(tensor.is_packed_row_major());
    Ok(())
}

#[test]
#[should_panic]
fn tensor_view_from_parts_too_small() {
    let cx = test_context();
    let storage = cx
        .device_memory_pool()
        .alloc_zeroed_on(4096, cx.default_compute_stream())
        .unwrap();
    let _ = TensorView::from_parts(VirtTensorView::new([32, 32], F32), storage.slice(4..), &cx);
}

#[test]
fn mx_from_parts_matches_allocation() -> anyhow::Result<()> {
    let cx = test_context();
    let stream = cx.default_compute_stream();
    for format in [MxFormat::Mxfp8, MxFormat::Nvfp4] {
        for dim in [MxDim::Rowwise, MxDim::Colwise] {
            let shape = [200, 96];
            let mut tensor = MxTensor::new_zeroed(shape, format, dim, stream, &cx)?;
            let layout = MxLayout::new(shape, format, dim)?;
            assert!(tensor.values().virt().same_layout_as(&layout.values));
            assert!(
                tensor
                    .scale_factors()
                    .virt()
                    .same_layout_as(&layout.scale_factors)
            );
            assert_eq!(layout.values.shape(), shape.into());

            let view = MxTensorView::from_parts(
                tensor.values().view(),
                tensor.scale_factors().view(),
                format,
                dim,
                shape,
            )?;
            assert_eq!(view.strides(), tensor.strides());
            assert_eq!(
                view.has_scale_factor_padding(),
                tensor.has_scale_factor_padding()
            );

            // Mismatched shape is rejected.
            assert!(
                MxTensorView::from_parts(
                    tensor.values().view(),
                    tensor.scale_factors().view(),
                    format,
                    dim,
                    [200, 64],
                )
                .is_err()
            );
            let _ = tensor.view_mut();

            let values = Tensor::from_parts(
                layout.values.clone(),
                tensor_storage(&cx, &layout.values)?,
                &cx,
            );
            let sfs = Tensor::from_parts(
                layout.scale_factors.clone(),
                tensor_storage(&cx, &layout.scale_factors)?,
                &cx,
            );
            MxTensor::from_parts(values, sfs, format, dim, shape)?;
        }
    }
    Ok(())
}

fn tensor_storage(
    cx: &cleandl::context::Context,
    virt: &VirtTensorView,
) -> anyhow::Result<skycuda::device_memory::DeviceMemory> {
    Ok(cx
        .device_memory_pool()
        .alloc_zeroed_on(virt.required_bytes(), cx.default_compute_stream())?)
}
