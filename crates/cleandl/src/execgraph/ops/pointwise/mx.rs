//! Block-scaled quantization on tiles
use tileir_builder::{F8E4M3, RoundingMode, Tile, select};

/// Largest finite magnitudes of the value formats.
pub const E4M3_MAX: f32 = 448.0;
pub const E5M2_MAX: f32 = 57344.0;
pub const E2M1_MAX: f32 = 6.0;

pub fn clamp(v: Tile<f32>, max: f32) -> Tile<f32> {
    v.maximum(-max).minimum(max)
}

pub fn mxfp8_scale(amax: Tile<f32>) -> (Tile<u8>, Tile<f32>) {
    let bits = (amax / E4M3_MAX).bitcast::<i32>();
    let exp = (bits >> 23) & 0xFF;
    let mantissa = bits & 0x7F_FFFF;

    let normal = exp + select(mantissa.ne(0), 1i32, 0i32);
    let subnormal = select(mantissa.gt(0x40_0000), 1i32, 0i32);
    let encoded = select(exp.eq(0), subnormal, normal).minimum(254);
    let encoded = select(exp.eq(255) & mantissa.ne(0), 255i32, encoded);

    let descale_exp = Tile::<i32>::scalar(254) - encoded;
    let descale_bits = select(
        descale_exp.ge(1),
        descale_exp << 23,
        select(descale_exp.eq(0), 0x0040_0000i32, 0x7FC0_0000i32),
    );
    let scale = encoded.astype::<u8>(RoundingMode::Zero);
    (scale, descale_bits.bitcast::<f32>())
}

pub fn nvfp4_scale(amax: Tile<f32>) -> (Tile<F8E4M3>, Tile<f32>) {
    let scale = (amax / E2M1_MAX)
        .minimum(E4M3_MAX)
        .astype::<F8E4M3>(RoundingMode::NearestEven);
    let rounded = scale.astype::<f32>(RoundingMode::NearestEven);
    let descale = select(
        rounded.eq(0.0f32),
        0.0f32,
        Tile::<f32>::scalar(1.0) / rounded,
    );
    (scale, descale)
}

pub fn scale_factor_index(row: Tile<i64>, col: Tile<i64>, num_blocks: Tile<i64>) -> Tile<i64> {
    ((row / 128i64) * ((num_blocks + 3i64) / 4i64) + col / 4i64) * 512i64
        + (row % 32i64) * 16i64
        + ((row % 128i64) / 32i64) * 4i64
        + col % 4i64
}
