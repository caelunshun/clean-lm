// --- LLM-generated --- //

use cfg_aliases::cfg_aliases;

fn main() {
    cfg_aliases! {
        has_227kb_smem: {
            any(
                cuda_arch_specific = "90",
                cuda_arch_specific = "100",
                cuda_arch_specific = "103",
                cuda_arch_specific = "107",
                cuda_arch_specific = "110"
            )
        },
        has_tma: {
            cuda_arch_min = "90"
        }
    }
}
