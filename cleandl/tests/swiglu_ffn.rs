use cleandl::{
    context::Context,
    operators::swiglu_ffn::{SwiGluConfig, SwiGluFfnExecutor},
};

#[test]
fn swiglu_ffn() -> anyhow::Result<()> {
    let cx = &Context::new(0)?;

    SwiGluFfnExecutor::get(
        SwiGluConfig {
            batch_size: 16384,
            in_features: 4096,
            intermediate_features: 8192,
            out_features: 2048,
        },
        cx,
    )?;

    Ok(())
}
