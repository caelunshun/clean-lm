use crate::{
    skyred::{
        config::{Config, LrSchedule},
        dataset::{
            document_file::DocumentReader,
            tokenization::{SkyredTokenizer, SpecialToken},
        },
    },
    training::{
        checkpoint::{CheckpointBuilder, CheckpointLoader},
        model::{MetaTensors, Sky1Executor, Sky1Weights},
        weight_update_params::WeightUpdateParams,
    },
};
use clap::Parser;
use cleandl::{context::Context, operators::attention::RaggedSeqlenTensors, tensor::Tensor};
use rand::{SeedableRng, rngs::StdRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use skycuda::{
    device_memory_pool::DeviceMemoryPoolOptions,
    host_pinned_memory_pool::HostPinnedMemoryPoolOptions,
};
use skytokens::vocab::{TokenId, Vocab};
use std::{f32::consts::PI, mem, path::PathBuf, process::exit, sync::Arc, thread};

#[derive(Debug, Parser)]
pub struct Cli {
    #[arg(long)]
    pub resume: Option<PathBuf>,
}

pub fn run(cli: &Cli, config: &Config) -> anyhow::Result<()> {
    let vocab: Vocab<SpecialToken> =
        serde_json::from_slice(&fs_err::read(config.dataset.work_dir.join("vocab.json"))?)?;
    let tokenizer = Arc::new(SkyredTokenizer::new(
        vocab.clone(),
        config.dataset.use_pretokenization,
    )?);

    let mut num_microbatches = 0usize;
    let mut max_num_sequences = 0;
    for batch in stream_batches(tokenizer.clone(), config)? {
        num_microbatches += 1;
        max_num_sequences = max_num_sequences.max(batch.seqlens.len());
    }

    let num_steps = num_microbatches.div_ceil(config.training.gradient_accumulation_steps as usize);

    tracing::info!(
        num_microbatches,
        num_steps,
        num_million_tokens =
            num_microbatches * config.training.microbatch_size as usize / 1_000_000,
        max_num_sequences,
        "scanned dataset"
    );

    let cx = Context::new(
        0,
        DeviceMemoryPoolOptions {
            max_reserved_free_bytes: 1 << 40,
        },
        HostPinnedMemoryPoolOptions {
            max_reserved_free_bytes: 1 << 40,
        },
        config.training.enable_autotuning,
    )?;

    let model_config = crate::model::Sky1Config {
        blocks: crate::model::BlocksConfig {
            hidden_size: config.model.hidden_size,
            ffn_intermediate_size: config.model.ffn_intermediate_size,
            num_decoder_q_heads: config.model.num_q_heads,
            num_decoder_kv_heads: config.model.num_kv_heads,
            head_dim: config.model.head_dim,
            max_batch_decoder_tokens: config.training.microbatch_size as usize,
            max_num_sequences,
            max_decoder_seqlen: config.model.max_seqlen,
            rope_base_freq: config.model.rope_base_freq,
        },
        vocab_size: vocab.num_tokens() as usize,
        num_encoder_blocks: 0,
        num_decoder_blocks: config.model.num_blocks,
    };

    #[derive(Debug, Serialize, Deserialize)]
    struct LossRecord {
        batch: usize,
        loss: f32,
    }

    let (mut weights, mut loss_history, mut train_step) = match &cli.resume {
        Some(checkpoint_dir) => {
            let mut loader = CheckpointLoader::load_from_dir(checkpoint_dir)?;
            let weights = Sky1Weights::load_from_checkpoint(
                &model_config,
                &mut loader,
                &cx,
                cx.h2d_copy_stream(),
            )?;

            let loss_history = csv::Reader::from_path(checkpoint_dir.join("loss.csv"))?
                .into_deserialize::<LossRecord>()
                .collect::<Result<Vec<_>, csv::Error>>()?;

            (weights, loss_history, loader.train_step())
        }
        None => {
            let weights = Sky1Weights::new_random(
                &model_config,
                &mut StdRng::seed_from_u64(config.training.seed),
                cx.h2d_copy_stream(),
                &cx,
            )?;
            (weights, Vec::new(), 0)
        }
    };

    let executor = Sky1Executor::get(model_config.clone(), &cx)?;

    let checkpoint = |weights: &Sky1Weights,
                      train_step: u32,
                      loss_history: &[LossRecord]|
     -> anyhow::Result<()> {
        let mut builder = CheckpointBuilder::default();
        weights.save_checkpoint(&mut builder, &cx, cx.d2h_copy_stream())?;
        let dir = config
            .training
            .checkpoints_dir
            .join(format!("checkpoint{train_step:06}"));
        fs_err::create_dir_all(&dir)?;

        builder.save_to_dir(train_step, &dir)?;

        let mut loss_csv = csv::Writer::from_path(dir.join("loss.csv"))?;
        for loss_record in loss_history {
            loss_csv.serialize(loss_record)?;
        }
        loss_csv.flush()?;

        tracing::info!("saved checkpoint to {}", dir.display());

        let mut all_checkpoints = fs_err::read_dir(&config.training.checkpoints_dir)?
            .map(|entry| entry.map(|e| e.file_name().to_string_lossy().into_owned()))
            .collect::<Result<Vec<_>, _>>()?;
        all_checkpoints.retain(|c| c.starts_with("checkpoint"));
        all_checkpoints.sort_unstable();
        if all_checkpoints.len() > config.training.max_kept_checkpoints {
            fs_err::remove_dir_all(config.training.checkpoints_dir.join(&all_checkpoints[0]))?;
        }

        Ok(())
    };

    let (shutdown_tx, shutdown) = flume::bounded(1);
    ctrlc::set_handler(move || {
        shutdown_tx.send(()).ok();
    })?;

    for (i, batch) in stream_batches(tokenizer, config)?
        .into_iter()
        .enumerate()
        .skip(train_step as usize * config.training.gradient_accumulation_steps as usize)
    {
        let token_indices = Tensor::from_slice(
            bytemuck::cast_slice::<TokenId, u32>(&batch.tokens),
            [batch.tokens.len()],
            cx.h2d_copy_stream(),
            &cx,
        )?;
        let labels = Tensor::from_slice(
            bytemuck::cast_slice::<TokenId, u32>(&batch.labels),
            [batch.tokens.len()],
            cx.h2d_copy_stream(),
            &cx,
        )?;
        let meta_tensors = MetaTensors {
            decoder_ragged_seqlen: RaggedSeqlenTensors::build(
                &batch.seqlens,
                &cx,
                &model_config.decoder_attention_config(),
            )?,
            seqlen: batch.seqlens,
        };

        let saved_tensors = executor.train_forward(
            &token_indices.view(),
            &labels.view(),
            &batch.labels_mask,
            &meta_tensors,
            &weights,
            cx.default_compute_stream(),
        )?;
        let loss = cx.download_from_device(saved_tensors.loss().storage(), cx.d2h_copy_stream())?;
        executor.train_backward(
            &token_indices.view(),
            &meta_tensors,
            &mut weights,
            &saved_tensors,
            cx.default_compute_stream(),
        )?;

        let loss = *bytemuck::from_bytes::<f32>(loss.view()?);
        tracing::info!(batch = i, loss, "processed batch");
        loss_history.push(LossRecord { batch: i, loss });

        if (i + 1).is_multiple_of(config.training.gradient_accumulation_steps as usize)
            || i == num_microbatches - 1
        {
            let lr = if let Some(lr_warmup_steps) = config.training.lr_warmup_steps
                && train_step < lr_warmup_steps
            {
                let first_lr = match config.training.lr_schedule {
                    LrSchedule::Constant { lr } => lr,
                    LrSchedule::Cosine { lr_max, .. } => lr_max,
                };
                (train_step + 1) as f32 / lr_warmup_steps as f32 * first_lr
            } else {
                match config.training.lr_schedule {
                    LrSchedule::Constant { lr } => lr,
                    LrSchedule::Cosine { lr_min, lr_max } => {
                        let warmup_steps = config.training.lr_warmup_steps.unwrap_or(0);
                        lr_min
                            + 0.5
                                * (lr_max - lr_min)
                                * (1.0
                                    + (PI * (train_step - warmup_steps) as f32
                                        / (num_steps - warmup_steps as usize) as f32)
                                        .cos())
                    }
                }
            };

            train_step += 1;
            executor.weight_update(
                &mut weights,
                &WeightUpdateParams {
                    adam_beta1: config.training.adam_beta1,
                    adam_beta2: config.training.adam_beta2,
                    weight_decay: config.training.weight_decay,
                    max_gradient_norm: config.training.max_gradient_norm,
                    lr,
                },
                train_step,
                cx.default_compute_stream(),
            )?;
            tracing::info!(lr, "weight update");

            if train_step.is_multiple_of(config.training.steps_per_checkpoint as u32) {
                checkpoint(&weights, train_step, &loss_history)?;
            }
        }

        if shutdown.try_recv().is_ok() {
            tracing::info!("received shutdown signal");
            break;
        }
    }

    Ok(())
}

#[derive(Debug, Clone, Default)]
struct Batch {
    tokens: Vec<TokenId>,
    labels: Vec<TokenId>,
    labels_mask: Vec<bool>,
    seqlens: Vec<u32>,
}

/// Streams batches of exactly the microbatch size in token count.
/// Drops any tail that does not fill a full batch.
fn stream_batches(
    tokenizer: Arc<SkyredTokenizer>,
    config: &Config,
) -> anyhow::Result<flume::Receiver<Batch>> {
    let mut dataset = DocumentReader::open(&config.dataset.work_dir.join("posts.zst"))?;

    let (batches_tx, batches_rx) = flume::bounded(4);

    const DOCUMENT_GROUP_SIZE: usize = 16384;

    let config = config.clone();

    thread::spawn(move || {
        let res = try {
            let mut current_batch = Batch::default();

            'outer: loop {
                let documents = (0..DOCUMENT_GROUP_SIZE)
                    .filter_map(|_| dataset.next())
                    .collect::<anyhow::Result<Vec<_>>>()?;

                let tokenized_documents = documents
                    .into_par_iter()
                    .map(|doc| tokenizer.tokenize_document(&doc))
                    .collect::<anyhow::Result<Vec<_>>>()?;

                if tokenized_documents.is_empty() {
                    break;
                }

                for doc in tokenized_documents {
                    let mut consumed = 0;
                    while consumed < doc.len() {
                        let needed_batch_tokens =
                            config.training.microbatch_size as usize - current_batch.tokens.len();
                        let tokens_to_take = needed_batch_tokens
                            .min(doc.len() - consumed)
                            .min(config.model.max_seqlen);

                        current_batch
                            .tokens
                            .extend_from_slice(&doc[consumed..][..tokens_to_take]);
                        current_batch.seqlens.push(tokens_to_take as u32);

                        let (labels, labels_mask): (Vec<_>, Vec<_>) = (consumed + 1
                            ..consumed + 1 + tokens_to_take)
                            .map(|i| match doc.get(i) {
                                Some(&tok) => (tok, true),
                                None => (TokenId::PLACEHOLDER, false),
                            })
                            .collect();
                        current_batch.labels.extend_from_slice(&labels);
                        current_batch.labels_mask.extend_from_slice(&labels_mask);

                        if current_batch.tokens.len() == config.training.microbatch_size as usize {
                            let res = batches_tx.send(mem::take(&mut current_batch));
                            if res.is_err() {
                                break 'outer;
                            }
                        }

                        consumed += tokens_to_take;
                    }
                }
            }
        };
        if let Err(e) = res {
            tracing::error!("error: {e:?}");
            exit(1);
        }
    });

    Ok(batches_rx)
}
