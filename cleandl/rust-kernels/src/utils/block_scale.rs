const MAX_F8E4M3: f32 = 448.0;

pub trait BlockScaledFormat {
    type const BLOCK_SIZE: usize;
    type const QUANTIZED_BYTES_PER_BLOCK: usize;
    type ScaleFactor;

    fn quantize(
        block: &[f32; Self::BLOCK_SIZE],
        output: &mut [u8; Self::QUANTIZED_BYTES_PER_BLOCK],
    ) -> Self::ScaleFactor;
}

pub struct Mxfp8;

impl BlockScaledFormat for Mxfp8 {
    type const BLOCK_SIZE: usize = 32;
    type const QUANTIZED_BYTES_PER_BLOCK: usize = 32;
    type ScaleFactor = u8;

    #[cuda_annotate]
    fn quantize(
        block: &[f32; Self::BLOCK_SIZE],
        output: &mut [u8; Self::QUANTIZED_BYTES_PER_BLOCK],
    ) -> Self::ScaleFactor {
        todo!()
    }
}
