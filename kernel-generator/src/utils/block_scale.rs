use nvvm_builder::{
    ConvertOpts, DynVal, DynVector, F4E2M1, F8E4M3, Rounding, UE4M3, UE8M0, Val, Vector, bf16,
};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BlockScaledFormat {
    Mxfp8,
    Nvfp4,
}

impl BlockScaledFormat {
    pub fn block_size(&self) -> usize {
        match self {
            BlockScaledFormat::Mxfp8 => 32,
            BlockScaledFormat::Nvfp4 => 16,
        }
    }

    /// Computes scale factors for one or more blocks, from the provided per-block
    /// amax values. Returned type is a vector of the same length as `amaxs`, whose
    /// type is the scale factor type of this format.
    ///
    /// For formats with global scale factors (NVFP4), the amaxs should have been
    /// computed after already applying the global descale.
    pub fn compute_scale_factors_from_amaxs(&self, amaxs: Val<Vector<bf16>>) -> DynVector {
        match self {
            BlockScaledFormat::Mxfp8 => (amaxs / 448.0f32)
                .convert::<UE8M0>(ConvertOpts::new().rounding(Rounding::Rp).satfinite())
                .into(),
            BlockScaledFormat::Nvfp4 => (amaxs / 6.0f32)
                .convert::<F8E4M3>(ConvertOpts::new().rounding(Rounding::Rn).satfinite())
                .into(),
        }
    }

    /// Quantizes some values using their scale factors.
    pub fn quantize(&self, vals: Val<Vector<bf16>>, scales: DynVector) -> DynVector {
        match self {
            BlockScaledFormat::Mxfp8 => {
                let scales = scales.downcast::<UE8M0>().unwrap();

                // Exponent is [-127, 127] encoded as an unsigned 8-bit value with a bias of 127. Encoding 0xFF is NaN.
                // We want to invert the encoded exponent: E' = -(E - 127) + 127 = -E + 127 + 127 = 254 - E
                // Special cases:
                // - Exponent is 127: inverting produces -127 which leads to a correct denormal in bf16 after the conversion.
                // - Exponent is 128 (NaN): inverting undergoes unsigned overflow, wrapping to encoded value of 255, which is still NaN.
                let scale_bits = scales.bitcast_lanes::<u8>();
                let descale_bits = 254u8 - scale_bits;
                let descales = descale_bits
                    .bitcast_lanes::<UE8M0>()
                    .convert::<bf16>(ConvertOpts::default());

                (vals * descales)
                    .convert::<F8E4M3>(ConvertOpts::new().rounding(Rounding::Rn).satfinite())
                    .into()
            }
            BlockScaledFormat::Nvfp4 => {
                let scales = scales.downcast::<F8E4M3>().unwrap();

                let descales = scales.convert::<bf16>(ConvertOpts::default()).rcp_approx();

                (vals * descales)
                    .convert::<F4E2M1>(ConvertOpts::default().rounding(Rounding::Rn).satfinite())
                    .into()
            }
        }
    }
}
