use nvvm_builder::{IntoVal, Val};

pub fn parity_from_idx(idx: impl IntoVal<u32>, num_stages: u32) -> Val<u32> {
    (idx.into_val() / num_stages) & 0b1
}
