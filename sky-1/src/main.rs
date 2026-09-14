use cleandl::{context::Context, tensor::Tensor};
use rand_distr::Uniform;
use sky_1::{
    model::{BlocksConfig, Sky1Config},
    training::model::{MetaTensors, Sky1Executor, Sky1Weights},
};
use skycuda::{
    device_memory_pool::DeviceMemoryPoolOptions,
    host_pinned_memory_pool::HostPinnedMemoryPoolOptions,
};

// Temporary testbed code
fn main() -> anyhow::Result<()> {
    let cx = &Context::new(
        0,
        DeviceMemoryPoolOptions {
            max_reserved_free_bytes: 256 << 30,
        },
        HostPinnedMemoryPoolOptions {
            max_reserved_free_bytes: 8 << 30,
        },
    )?;
    let stream = cx.default_stream();

    let config = Sky1Config {
        blocks: BlocksConfig {
            hidden_size: 2048,
            ffn_intermediate_size: 3 * 2048,
            num_decoder_q_heads: 32,
            num_decoder_kv_heads: 4,
            head_dim: 64,
            max_batch_decoder_tokens: 8 * 1024,
            max_num_sequences: 32,
            max_decoder_seqlen: 2048,
            rope_base_freq: 10_000.0,
        },
        vocab_size: 65536,
        num_encoder_blocks: 0,
        num_decoder_blocks: 1,
    };

    let weights = Sky1Weights::new_random(&config, &mut rand::rng(), stream, cx)?;

    // --- Dummy batch initialization --- //
    let token_indices = Tensor::from_distr(
        Uniform::<i32>::new(0, config.vocab_size as i32)?,
        &mut rand::rng(),
        [config.blocks.max_batch_decoder_tokens],
        stream,
        cx,
    )?;
    let labels = Tensor::from_distr(
        Uniform::<i32>::new(0, config.vocab_size as i32)?,
        &mut rand::rng(),
        [config.blocks.max_batch_decoder_tokens],
        stream,
        cx,
    )?;

    assert!(
        config
            .blocks
            .max_batch_decoder_tokens
            .is_multiple_of(config.blocks.max_num_sequences)
    );

    let seqlen = vec![
        (config.blocks.max_batch_decoder_tokens / config.blocks.max_num_sequences)
            as i32;
        config.blocks.max_num_sequences
    ];
    let mut cu_seqlen = vec![0i64];
    for &len in &seqlen {
        let last = cu_seqlen.last().copied().unwrap();
        cu_seqlen.push(last + len as i64);
    }

    let meta_tensors = MetaTensors {
        decoder_seqlen: Tensor::from_slice(&seqlen, [config.blocks.max_num_sequences], stream, cx)?,
        decoder_cu_seqlen: Tensor::from_slice(
            &cu_seqlen,
            [config.blocks.max_num_sequences + 1],
            stream,
            cx,
        )?,
        decoder_cu_seqlen_times_hd_times_qh: Tensor::from_slice(
            &cu_seqlen
                .iter()
                .map(|&x| x * (config.blocks.head_dim * config.blocks.num_decoder_q_heads) as i64)
                .collect::<Vec<_>>(),
            [config.blocks.max_num_sequences + 1],
            stream,
            cx,
        )?,
        decoder_cu_seqlen_times_hd_times_kvh: Tensor::from_slice(
            &cu_seqlen
                .iter()
                .map(|&x| x * (config.blocks.head_dim * config.blocks.num_decoder_kv_heads) as i64)
                .collect::<Vec<_>>(),
            [config.blocks.max_num_sequences + 1],
            stream,
            cx,
        )?,
    };

    let executor = Sky1Executor::get(config.clone(), cx)?;
    let res = executor.train_forward(
        &token_indices.view(),
        &labels.view(),
        &meta_tensors,
        &weights,
        stream,
    )?;

    let loss = bytemuck::cast_slice::<u8, f32>(
        cx.download_from_device(res.loss().storage(), stream)?
            .view()?,
    )[0];

    dbg!(loss);

    Ok(())
}
