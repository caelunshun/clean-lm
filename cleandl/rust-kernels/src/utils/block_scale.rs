use ptx_std::{
    Overflow, Relu, Rounding, SupportedVectorLen, Vector, VectorElement, VectorLen, WARP_SIZE,
    bf16xN, f8e4m3, f8e4m3xN, f8e8m0, f8e8m0xN, f32xN,
};

pub trait BlockScaledFormat {
    const BLOCK_SIZE: usize;
    type QuantizedVal<const N: usize>: VectorElement<N>;
    type ScaleFactor<const N: usize>: VectorElement<N>;

    fn compute_scale_factor<const N: usize>(amax: f32xN<N>) -> Vector<Self::ScaleFactor<N>, N>;

    fn quantize_bf16<const N: usize>(
        vals: bf16xN<N>,
        sfs: Vector<Self::ScaleFactor<N>, N>,
    ) -> Vector<Self::QuantizedVal<N>, N>
    where
        VectorLen<N>: SupportedVectorLen;
    fn quantize_f32<const N: usize>(
        vals: f32xN<N>,
        sfs: Vector<Self::ScaleFactor<N>, N>,
    ) -> Vector<Self::QuantizedVal<N>, N>;

    // fn dequantize_to_bf16<const N: usize>(
    //     vals: Vector<Self::QuantizedVal, N>,
    //     sfs: Vector<Self::ScaleFactor, N>,
    // ) -> bf16xN<N>
    // where
    //     VectorLen<N>: SupportedVectorLen;
}

pub struct Mxfp8;

impl BlockScaledFormat for Mxfp8 {
    const BLOCK_SIZE: usize = 32;
    type ScaleFactor<const N: usize> = f8e8m0;
    type QuantizedVal<const N: usize> = f8e4m3;

    fn compute_scale_factor<const N: usize>(amax: f32xN<N>) -> Vector<Self::ScaleFactor<N>, N> {
        let sf_unrounded = amax * f8e4m3::MAX.to_f32::<{ Relu::Disabled }>();
        Vector::<f8e8m0, N>::from_f32::<{ Rounding::Up }, { Overflow::SaturateFinite }>(
            sf_unrounded,
        )
    }

    fn quantize_bf16<const N: usize>(
        vals: bf16xN<N>,
        sfs: Vector<Self::ScaleFactor<N>, N>,
    ) -> Vector<Self::QuantizedVal<N>, N> {
        f8e4m3xN::from_bf16::<{ Relu::Disabled }>(vals * rcp_f8e8m0(sfs).to_bf16())
    }

    fn quantize_f32<const N: usize>(
        vals: f32xN<N>,
        sfs: Vector<Self::ScaleFactor<N>, N>,
    ) -> Vector<Self::QuantizedVal<N>, N> {
        f8e4m3xN::from_f32::<{ Relu::Disabled }>(vals * rcp_f8e8m0(sfs).to_f32())
    }
}

#[inline(always)]
fn rcp_f8e8m0<const N: usize>(x: f8e8m0xN<N>) -> f8e8m0xN<N> {
    // Exponent is [-127, 127] encoded as an unsigned 8-bit value with a bias of 127. Encoding 0xFF is NaN.
    // We want to invert the encoded exponent: E' = -(E - 127) + 127 = -E + 127 + 127 = 254 - E
    // Special cases:
    // - Exponent is 127: inverting produces -127 which leads to a correct denormal in bf16 after the conversion.
    // - Exponent is 128 (NaN): inverting undergoes unsigned overflow, wrapping to encoded value of 255, which is still NaN.
    f8e8m0xN::from_array(
        x.to_array()
            .map(|x| f8e8m0::from_bits(254u8.wrapping_sub(x.to_bits()))),
    )
}

pub const SF_TILE_NROWS: u32 = 128;
pub const SF_TILE_NCOLS: u32 = 4;

/// Computes global index of a scale factor in the scale factor tensor (in units of elements)
/// given its coordinates as (row, block), where `col` increases along the blockscaling dimension
/// and `row` increases along the other dimension.
///
/// This uses the 128x4 swizzled tiled layout for scale factors, as consumed e.g. by cuBLASlt.
///
/// Note that `ncols` is the width of the scale factors tensor, not the original tensor.
/// It must be padded to a multiple of SF_TILE_NCOLS.
#[inline(always)]
pub fn scale_factor_pos(row: u32, col: u32, ncols: u32) -> usize {
    let n_col_blocks = ncols / 4;
    let offset = ((row / 128) * n_col_blocks + (col / 4)) * 128 * 4
        + (row % WARP_SIZE) * 16
        + ((row % 128) / WARP_SIZE) * 4
        + (col % 4);
    offset as usize
}
