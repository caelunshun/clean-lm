use ptx_std::{F32Ext, Overflow, Relu, Rounding, bf16x2, f8e4m3x2, f8e8m0x2};

const MAX_F8E4M3: f32 = 448.0;

pub trait BlockScaledFormat {
    const BLOCK_SIZE: usize;
    const QUANTIZED_BYTES_PER_BLOCK: usize;
    type BlockBf16;
    type BlockQuantized;
    type ScaleFactorX2;

    fn quantize_within_lane_bf16_2blocks(
        block0: &Self::BlockBf16,
        block1: &Self::BlockBf16,
        quant0: &mut Self::BlockQuantized,
        quant1: &mut Self::BlockQuantized,
    ) -> Self::ScaleFactorX2;
}

pub struct Mxfp8;

impl BlockScaledFormat for Mxfp8 {
    const BLOCK_SIZE: usize = 32;
    const QUANTIZED_BYTES_PER_BLOCK: usize = 32;
    type BlockBf16 = [bf16x2; 16];
    type BlockQuantized = [f8e4m3x2; 16];
    type ScaleFactorX2 = f8e8m0x2;

    #[inline(always)]
    fn quantize_within_lane_bf16_2blocks(
        block0: &Self::BlockBf16,
        block1: &Self::BlockBf16,
        quant0: &mut Self::BlockQuantized,
        quant1: &mut Self::BlockQuantized,
    ) -> Self::ScaleFactorX2 {
        let mut amax0 = bf16x2::ZERO;
        let mut amax1 = bf16x2::ZERO;
        #[rustc_unroll(full)]
        for x0 in block0.iter().copied() {
            amax0 = amax0.max(x0.abs());
        }
        #[rustc_unroll(full)]
        for x1 in block1.iter().copied() {
            amax1 = amax1.max(x1.abs());
        }

        let amax0 = max_bf16x2_f32(amax0);
        let amax1 = max_bf16x2_f32(amax1);

        let scale0 = amax0 * MAX_F8E4M3.recip();
        let scale1 = amax1 * MAX_F8E4M3.recip();

        let scales =
            f8e8m0x2::from_f32s::<{ Rounding::Up }, { Overflow::SaturateFinite }>([scale0, scale1]);
        let descales = scales.to_bf16x2().to_f32x2().map(|x| x.rcp_approx_ftz());
        let [descale0, descale1] = bf16x2::from_f32x2(descales[0], descales[1]).to_array();

        #[rustc_unroll(full)]
        for i in 0..Self::BLOCK_SIZE / 2 {
            quant0[i] =
                f8e4m3x2::from_bf16x2::<{ Relu::Disabled }>(block0[i] * bf16x2::splat(descale0));
            quant1[i] =
                f8e4m3x2::from_bf16x2::<{ Relu::Disabled }>(block1[i] * bf16x2::splat(descale1));
        }

        scales
    }
}

fn max_bf16x2_f32(x: bf16x2) -> f32 {
    let [a, b] = x.to_f32x2();
    a.max(b)
}
