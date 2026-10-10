#[cfg(has_tcgen05)]
pub mod tcgen05_bf16;

// Only used by the tcgen05 kernels.
#[cfg(not(has_tcgen05))]
use pastey as _;
