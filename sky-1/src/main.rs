use std::env;

use clap::Parser;
use cleandl::{context::Context, operators::attention::RaggedSeqlenTensors, tensor::Tensor};
use mimalloc::MiMalloc;
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::Uniform;
use sky_1::{
    model::{BlocksConfig, Sky1Config},
    training::{
        config::TrainingConfig,
        model::{MetaTensors, Sky1Executor, Sky1Weights},
    },
};
use skycuda::{
    device_memory_pool::DeviceMemoryPoolOptions,
    host_pinned_memory_pool::HostPinnedMemoryPoolOptions,
};

#[global_allocator]
static ALLOCATOR: MiMalloc = MiMalloc;

#[derive(Debug, Parser)]
struct Cli {
    #[arg(long, default_value = "4096")]
    batch_size: usize,
    #[arg(long, default_value = "128")]
    max_decoder_seqlen: usize,
}

// Temporary testbed code
fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    let cx = &Context::new(
        0,
        DeviceMemoryPoolOptions {
            max_reserved_free_bytes: 1024 << 30,
        },
        HostPinnedMemoryPoolOptions {
            max_reserved_free_bytes: 256 << 30,
        },
        env::var("CLEANDL_AUTOTUNE").is_ok(),
    )?;
    let stream = cx.default_compute_stream();

    let config = Sky1Config {
        blocks: BlocksConfig {
            hidden_size: 2048,
            ffn_intermediate_size: 3 * 2048,
            num_decoder_q_heads: 32,
            num_decoder_kv_heads: 4,
            head_dim: 64,
            max_batch_decoder_tokens: cli.batch_size,
            max_num_sequences: 32usize.max(cli.batch_size / cli.max_decoder_seqlen),
            max_decoder_seqlen: cli.max_decoder_seqlen,
            rope_base_freq: 10_000.0,
        },
        vocab_size: 65536,
        num_encoder_blocks: 0,
        num_decoder_blocks: 1,
    };

    let mut rng = StdRng::seed_from_u64(66);

    let mut weights = Sky1Weights::new_random(&config, &mut rng, stream, cx)?;

    // --- Dummy batch initialization --- //
    let token_indices = Tensor::from_distr(
        Uniform::<i32>::new(0, config.vocab_size as i32)?,
        &mut rng,
        [config.blocks.max_batch_decoder_tokens],
        stream,
        cx,
    )?;
    let labels = Tensor::from_distr(
        Uniform::<i32>::new(0, config.vocab_size as i32)?,
        &mut rng,
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
            as u32;
        config.blocks.max_num_sequences
    ];
    let meta_tensors = MetaTensors {
        decoder_ragged_seqlen: RaggedSeqlenTensors::build(
            &seqlen,
            cx,
            &config.decoder_attention_config(),
        )?,
    };

    let training_config = TrainingConfig {
        adam_beta1: 0.9,
        adam_beta2: 0.95,
        weight_decay: 0.1,
        max_gradient_norm: 10.0,
        lr: 3e-4,
    };

    let executor = Sky1Executor::get(config.clone(), cx)?;

    for train_step in 1..=10000 {
        let res = executor.train_forward(
            &token_indices.view(),
            &labels.view(),
            &meta_tensors,
            &weights,
            stream,
        )?;

        executor.train_backward(
            &token_indices.view(),
            &meta_tensors,
            &mut weights,
            &res,
            stream,
        )?;

        let loss = bytemuck::cast_slice::<u8, f32>(
            cx.download_from_device(res.loss().storage(), stream)?
                .view()?,
        )[0];

        dbg!(loss);

        executor.weight_update(&mut weights, &training_config, train_step, stream)?;
    }

    Ok(())
}
