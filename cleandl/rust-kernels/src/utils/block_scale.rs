// use core::simd::{f32x2, f32x4};
// use ptx_std::{
//     F32Ext, Overflow, Relu, Rounding, bf16x2, bf16x4, bf16x8, f8e4m3, f8e4m3x2, f8e4m3x4, f8e4m3x8,
//     f8e8m0, f8e8m0x2, f8e8m0x4, f8e8m0x8,
// };

// pub trait BlockScaledFormat {
//     const BLOCK_SIZE: usize;
//     type ScaleFactor;
//     type ScaleFactorX2;
//     type ScaleFactorX4;
//     type ScaleFactorX8;
//     type QuantizedX2;
//     type QuantizedX4;
//     type QuantizedX8;

//     fn compute_scale_factor(amax: f32) -> Self::ScaleFactor;
//     fn compute_scale_factor_x2(amax: f32x2) -> Self::ScaleFactorX2;
//     fn compute_scale_factor_x4(amax: f32x4) -> Self::ScaleFactorX4;

//     fn quantize_bf16x2(x: bf16x2, scales: Self::ScaleFactorX2) -> Self::QuantizedX2;
//     fn quantize_bf16x4(x: bf16x4, scales: Self::ScaleFactorX4) -> Self::QuantizedX4;
//     fn quantize_bf16x8(x: bf16x8, scales: Self::ScaleFactorX8) -> Self::QuantizedX8;

//     fn quantize_f32x2(x: f32x2, scales: Self::ScaleFactorX2) -> Self::QuantizedX2;
//     fn quantize_f32x4(x: f32x4, scales: Self::ScaleFactorX4) -> Self::QuantizedX4;

//     fn dequantize_to_bf16x2(q: Self::QuantizedX2, scales: Self::ScaleFactorX2) -> bf16x2;
//     fn dequantize_to_bf16x4(q: Self::QuantizedX4, scales: Self::ScaleFactorX4) -> bf16x4;

//     fn dequantize_to_f32x2(q: Self::QuantizedX2, scales: Self::ScaleFactorX2) -> f32x2;
//     fn dequantize_to_f32x4(q: Self::QuantizedX4, scales: Self::ScaleFactorX4) -> f32x4;
// }

// pub struct Mxfp8;

// impl BlockScaledFormat for Mxfp8 {
//     const BLOCK_SIZE: usize = 32;
//     type ScaleFactor = f8e8m0;
//     type ScaleFactorX2 = f8e8m0x2;
//     type ScaleFactorX4 = f8e8m0x4;
//     type ScaleFactorX8 = f8e8m0x8;
//     type QuantizedX2 = f8e4m3x2;
//     type QuantizedX4 = f8e4m3x4;
//     type QuantizedX8 = f8e4m3x8;

//     #[inline(always)]
//     fn compute_scale_factor(amax: f32) -> Self::ScaleFactor {
//         let scale_unrounded = amax.rcp_approx_ftz() * f8e4m3::MAX.to_f32::<{ Relu::Disabled }>();
//         f8e8m0::from_f32::<{ Rounding::Up }, { Overflow::SaturateFinite }>(scale_unrounded)
//     }

//     #[inline(always)]
//     fn compute_scale_factor_x2(amax: f32x2) -> Self::ScaleFactorX2 {
//         let scale_unrounded =
//             amax.rcp_approx_ftz() * f32x2::splat(f8e4m3::MAX.to_f32::<{ Relu::Disabled }>());
//         let scale_rounded =
//             f8e8m0x2::from_f32::<{ Rounding::Up }, { Overflow::SaturateFinite }>(scale_unrounded);
//         scale_rounded
//     }

//     #[inline(always)]
//     fn compute_scale_factor_x4(amax: f32x4) -> Self::ScaleFactorX4 {
//         f8e8m0x4::from_x2s([
//             Self::compute_scale_factor_x2(amax.extract::<0, 2>()),
//             Self::compute_scale_factor_x2(amax.extract::<2, 2>()),
//         ])
//     }

//     #[inline(always)]
//     fn quantize_bf16x2(x: bf16x2, scales: Self::ScaleFactorX2) -> Self::QuantizedX2 {
//         let descales = rcp_f8e8m0x2(scales).to_bf16();
//         f8e4m3x2::from_bf16::<{ Relu::Disabled }>(descales * x)
//     }

//     #[inline(always)]
//     fn quantize_bf16x4(x: bf16x4, scales: Self::ScaleFactorX4) -> Self::QuantizedX4 {
//         f8e4m3x4::from_x2s([
//             Self::quantize_bf16x2(x.to_x2s()[0], scales.to_x2s()[0]),
//             Self::quantize_bf16x2(x.to_x2s()[1], scales.to_x2s()[1]),
//         ])
//     }

//     #[inline(always)]
//     fn quantize_bf16x8(x: bf16x8, scales: Self::ScaleFactorX8) -> Self::QuantizedX8 {
//         f8e4m3x8::from_x2s([
//             Self::quantize_bf16x2(x.to_x2s()[0], scales.to_x2s()[0]),
//             Self::quantize_bf16x2(x.to_x2s()[1], scales.to_x2s()[1]),
//             Self::quantize_bf16x2(x.to_x2s()[2], scales.to_x2s()[2]),
//             Self::quantize_bf16x2(x.to_x2s()[3], scales.to_x2s()[3]),
//         ])
//     }

//     #[inline(always)]
//     fn quantize_f32x2(x: f32x2, scales: Self::ScaleFactorX2) -> Self::QuantizedX2 {
//         let descales = rcp_f8e8m0x2(scales).to_f32();
//         f8e4m3x2::from_f32::<{ Relu::Disabled }>(x * descales)
//     }

//     #[inline(always)]
//     fn quantize_f32x4(x: f32x4, scales: Self::ScaleFactorX4) -> Self::QuantizedX4 {
//         f8e4m3x4::from_x2s([
//             Self::quantize_f32x2(x.extract::<0, 2>(), scales.to_x2s()[0]),
//             Self::quantize_f32x2(x.extract::<2, 2>(), scales.to_x2s()[1]),
//         ])
//     }

//     #[inline(always)]
//     fn dequantize_to_bf16x2(q: Self::QuantizedX2, scales: Self::ScaleFactorX2) -> bf16x2 {
//         q.to_bf16_scaled::<{ Relu::Disabled }, { Overflow::SaturateFinite }>(scales)
//     }

//     #[inline(always)]
//     fn dequantize_to_bf16x4(q: Self::QuantizedX4, scales: Self::ScaleFactorX4) -> bf16x4 {
//         q.to_bf16_scaled::<{ Relu::Disabled }, { Overflow::SaturateFinite }>(scales)
//     }

//     #[inline(always)]
//     fn dequantize_to_f32x2(q: Self::QuantizedX2, scales: Self::ScaleFactorX2) -> f32x2 {
//         Self::dequantize_to_bf16x2(q, scales).to_f32()
//     }

//     #[inline(always)]
//     fn dequantize_to_f32x4(q: Self::QuantizedX4, scales: Self::ScaleFactorX4) -> f32x4 {
//         Self::dequantize_to_bf16x4(q, scales).to_f32()
//     }
// }

// #[inline(always)]
// fn rcp_f8e8m0(x: f8e8m0) -> f8e8m0 {
//     // Exponent is [-127, 127] encoded as an unsigned 8-bit value with a bias of 127. Encoding 0xFF is NaN.
//     // We want to invert the encoded exponent: E' = -(E - 127) + 127 = -E + 127 + 127 = 254 - E
//     // Special cases:
//     // - Exponent is 127: inverting produces -127 which leads to a correct denormal in bf16 after the conversion.
//     // - Exponent is 128 (NaN): inverting undergoes unsigned overflow, wrapping to encoded value of 255, which is still NaN.
//     f8e8m0::from_bits(254u8.wrapping_sub(x.to_bits()))
// }

// #[inline(always)]
// fn rcp_f8e8m0x2(x: f8e8m0x2) -> f8e8m0x2 {
//     f8e8m0x2::from_array([rcp_f8e8m0(x.to_array()[0]), rcp_f8e8m0(x.to_array()[1])])
// }

// pub const SF_TILE_NROWS: u32 = 128;
// pub const SF_TILE_NCOLS: u32 = 4;

// /// Computes global index of a scale factor in the scale factor tensor (in units of elements)
// /// given its coordinates as (row, block), where `block` is along the blockscaling dimension
// /// and `row` is the other dimension.
// ///
// /// This uses the 128x4 swizzled tiled layout for scale factors, as consumed e.g. by cuBLASlt.
// ///
// /// Note that `ncols` is the width of the scale factors tensor, not the original tensor.
// /// It must be padded to a multiple of SF_TILE_NCOLS.
// #[inline(always)]
// pub fn scale_factor_pos(row: u32, col: u32, block_size: u32, ncols: u32) -> usize {
//     let n_col_blocks = ncols / 4;
//     let offset = ((row / 128) * n_col_blocks + (col / 4)) * 128 * 4
//         + (row % block_size) * 16
//         + ((row % 128) / block_size) * 4
//         + (col % 4);
//     offset as usize
// }
