use cleandl::{
    context::Context,
    operators::attention::{AttentionConfig, AttentionExecutor},
};

fn main() -> anyhow::Result<()> {
    let cx = Context::new(0)?;

    AttentionExecutor::get(
        AttentionConfig {
            max_q_seqlen: 1024 * 1024,
            max_kv_seqlen: 1024 * 1024,
            max_num_sequences: 1024,
            q_heads: 64,
            kv_heads: 8,
            head_dim: 128,
        },
        &cx,
    )?;

    Ok(())
}
