// --- LLM-generated --- //

use cfg_aliases::cfg_aliases;

fn main() {
    cfg_aliases! {
        has_227kb_smem: {
            any(
                target_feature = "sm_90a",
                target_feature = "sm_100f",
                target_feature = "sm_110f"
            )
        },
        has_tcgen05: {
            any(target_feature = "sm_100f", target_feature = "sm_110f")
        },
        has_mx_formats: {
            target_feature = "sm_100"
        }
    }
}
