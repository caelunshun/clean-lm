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
    let shape = [192, 96];
    // (format, dim, values strides, scale factor shape)
    let cases = [
        (MxFormat::Mxfp8, MxDim::Rowwise, [96, 1], [256, 4]),
        (MxFormat::Mxfp8, MxDim::Colwise, [1, 192], [128, 8]),
        (MxFormat::Nvfp4, MxDim::Rowwise, [96, 1], [256, 8]),
        (MxFormat::Nvfp4, MxDim::Colwise, [1, 192], [128, 12]),
    ];
    for (format, dim, values_strides, sf_shape) in cases {
        let tensor = MxTensor::new_zeroed(shape, format, dim, stream, &cx)?;
        let expected_values =
            VirtTensorView::with_strides(shape, Strides::new(values_strides), format.values_type());
        let expected_sfs = VirtTensorView::new(sf_shape, format.scale_factors_type());
        assert!(
            tensor.values().virt().same_layout_as(&expected_values),
            "{format:?} {dim:?}"
        );
        assert!(
            tensor.scale_factors().virt().same_layout_as(&expected_sfs),
            "{format:?} {dim:?}"
        );
        assert_eq!(tensor.strides(), Strides::new(values_strides));

        let layout = MxLayout::new(shape, format, dim)?;
        assert!(layout.values.same_layout_as(&expected_values));
        assert!(layout.scale_factors.same_layout_as(&expected_sfs));

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
                [192, 64],
            )
            .is_err()
        );

        let values = Tensor::from_parts(expected_values, tensor_storage(&cx, &layout.values)?, &cx);
        let sfs = Tensor::from_parts(
            expected_sfs,
            tensor_storage(&cx, &layout.scale_factors)?,
            &cx,
        );
        MxTensor::from_parts(values, sfs, format, dim, shape)?;
    }
    Ok(())
}

#[test]
fn mx_rejects_bad_layouts() -> anyhow::Result<()> {
    let cx = test_context();
    let stream = cx.default_compute_stream();
    let format = MxFormat::Mxfp8;

    // Block dimension not a multiple of the block size.
    assert!(MxLayout::new([128, 48], format, MxDim::Rowwise).is_err());
    assert!(MxLayout::new([48, 128], format, MxDim::Colwise).is_err());
    assert!(MxTensor::new_zeroed([128, 48], format, MxDim::Rowwise, stream, &cx).is_err());

    // Padded row stride.
    let shape = [128, 64];
    let layout = MxLayout::new(shape, format, MxDim::Rowwise)?;
    let padded = VirtTensorView::with_strides(shape, Strides::new([96, 1]), format.values_type());
    let values = Tensor::from_parts(padded.clone(), tensor_storage(&cx, &padded)?, &cx);
    let sfs = Tensor::from_parts(
        layout.scale_factors.clone(),
        tensor_storage(&cx, &layout.scale_factors)?,
        &cx,
    );
    assert!(MxTensor::from_parts(values, sfs, format, MxDim::Rowwise, shape).is_err());
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
