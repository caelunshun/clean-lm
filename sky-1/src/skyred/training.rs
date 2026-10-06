use crate::{
    skyred::{
        config::{Config, LrSchedule},
        dataset::{
            tokenization::{SpecialToken, VOCAB_FILE_NAME},
            tokenized_file::{SChunkReader, TOKEN_COUNTS_FILE_NAME, TOKENIZED_FILE_NAME},
        },
    },
    training::{
        checkpoint::{CheckpointBuilder, CheckpointLoader},
        model::{MetaTensors, Sky1Executor, Sky1Weights},
        weight_update_params::WeightUpdateParams,
    },
};
use anyhow::anyhow;
use clap::Parser;
use cleandl::{context::Context, operators::attention::RaggedSeqlenTensors, tensor::Tensor};
use rand::{SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};
use skycuda::{
    device_memory_pool::DeviceMemoryPoolOptions,
    host_pinned_memory_pool::HostPinnedMemoryPoolOptions,
};
use skytokens::vocab::{TokenId, Vocab};
use std::{
    f32::consts::PI,
    mem,
    ops::{ControlFlow, Range},
    path::PathBuf,
    process::exit,
    thread::{self, JoinHandle},
};

#[derive(Debug, Parser)]
pub struct Cli {
    #[arg(long)]
    pub resume: Option<PathBuf>,
    /// Stop training after reaching this global step. The LR schedule
    /// is unaffected and still spans the full dataset.
    #[arg(long)]
    pub max_steps: Option<u32>,
}

pub fn run(cli: &Cli, config: &Config) -> anyhow::Result<()> {
    let vocab: Vocab<SpecialToken> = serde_json::from_slice(&fs_err::read(
        config.dataset.work_dir.join(VOCAB_FILE_NAME),
    )?)?;

    let token_counts =
        SChunkReader::<u32>::open(&config.dataset.work_dir.join(TOKEN_COUNTS_FILE_NAME))?
            .read_all()?;

    let mut num_microbatches = 0usize;
    let mut max_num_sequences = 0;
    let mut num_sequences = 0;
    let mut packer = Packer::new(config);
    for &count in &token_counts {
        let _ = packer.pack_document(count as usize, |_, completes_batch| {
            num_sequences += 1;
            if completes_batch {
                num_microbatches += 1;
                max_num_sequences = max_num_sequences.max(num_sequences);
                num_sequences = 0;
            }
            ControlFlow::Continue(())
        });
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
            enable_profiling: true,
            max_total_bytes: 0,
            max_reserved_free_bytes: 2 << 30,
        },
        HostPinnedMemoryPoolOptions {
            max_reserved_free_bytes: 1 << 40,
        },
        config.training.enable_autotuning,
    )?;
    // let activations_pool = cx
    //     .cuda()
    //     .create_device_memory_pool(DeviceMemoryPoolOptions {
    //         enable_profiling: true,
    //         max_reserved_free_bytes: 2 << 30,
    //         max_total_bytes: 0,
    //     })?;
    let activations_pool = cx.device_memory_pool().clone();

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

    #[derive(Debug, Clone, Serialize, Deserialize)]
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

    let mut checkpoint_writer = CheckpointWriter::default();
    let mut checkpoint = |weights: &Sky1Weights,
                          train_step: u32,
                          loss_history: &[LossRecord]|
     -> anyhow::Result<()> {
        // Ensure at most one checkpoint in flight at a time.
        checkpoint_writer.wait()?;

        tracing::info!("dispatching checkpoint save");

        let mut builder = CheckpointBuilder::default();
        weights.save_checkpoint(&mut builder, &cx, cx.d2h_copy_stream())?;

        let checkpoints_dir = config.training.checkpoints_dir.clone();
        let max_kept_checkpoints = config.training.max_kept_checkpoints;
        let loss_history = loss_history.to_vec();
        checkpoint_writer.spawn(move || {
            let dir = checkpoints_dir.join(format!("checkpoint{train_step:06}"));
            fs_err::create_dir_all(&dir)?;

            builder.save_to_dir(train_step, &dir)?;

            let mut loss_csv = csv::Writer::from_path(dir.join("loss.csv"))?;
            for loss_record in &loss_history {
                loss_csv.serialize(loss_record)?;
            }
            loss_csv.flush()?;

            tracing::info!("saved checkpoint to {}", dir.display());

            let mut all_checkpoints = fs_err::read_dir(&checkpoints_dir)?
                .map(|entry| entry.map(|e| e.file_name().to_string_lossy().into_owned()))
                .collect::<Result<Vec<_>, _>>()?;
            all_checkpoints.retain(|c| c.starts_with("checkpoint"));
            all_checkpoints.sort_unstable();
            if all_checkpoints.len() > max_kept_checkpoints {
                fs_err::remove_dir_all(checkpoints_dir.join(&all_checkpoints[0]))?;
            }

            Ok(())
        });

        Ok(())
    };

    let (shutdown_tx, shutdown) = flume::bounded(1);
    ctrlc::set_handler(move || {
        shutdown_tx.try_send(()).ok();
    })?;

    let sidecar_stream = cx.cuda().create_stream()?;

    for (i, batch) in stream_batches(token_counts, config)
        .into_iter()
        .enumerate()
        .skip(train_step as usize * config.training.gradient_accumulation_steps as usize)
    {
        if cli
            .max_steps
            .is_some_and(|max_steps| train_step >= max_steps)
        {
            tracing::info!(train_step, "reached max steps");
            break;
        }

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
                &model_config.decoder_attention_config(batch.seqlens.len()),
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
            &activations_pool,
        )?;
        let loss = cx.download_from_device(saved_tensors.loss().storage(), cx.d2h_copy_stream())?;
        executor.train_backward(
            &token_indices.view(),
            &meta_tensors,
            &mut weights,
            saved_tensors,
            cx.default_compute_stream(),
            &sidecar_stream,
            &activations_pool,
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

            // Gradients are summed (not averaged) across microbatches, so need to correct
            // gradient clipping accordingly.
            let accumulated_microbatches =
                i + 1 - train_step as usize * config.training.gradient_accumulation_steps as usize;

            train_step += 1;
            executor.weight_update(
                &mut weights,
                &WeightUpdateParams {
                    adam_beta1: config.training.adam_beta1,
                    adam_beta2: config.training.adam_beta2,
                    weight_decay: config.training.weight_decay,
                    max_gradient_norm: config.training.max_gradient_norm
                        * accumulated_microbatches as f32,
                    lr,
                },
                train_step,
                cx.default_compute_stream(),
            )?;
            tracing::info!(lr, "weight update");

            if train_step.is_multiple_of(config.training.steps_per_checkpoint as u32)
                || i == num_microbatches - 1
                || cli.max_steps == Some(train_step)
            {
                checkpoint(&weights, train_step, &loss_history)?;
            }
        }

        if shutdown.try_recv().is_ok() {
            tracing::info!("received shutdown signal");
            break;
        }
    }

    checkpoint_writer.wait()?;

    Ok(())
}

/// Runs checkpoint save on a background thread.
/// waits for completion on drop.
#[derive(Default)]
struct CheckpointWriter {
    in_flight: Option<JoinHandle<anyhow::Result<()>>>,
}

impl CheckpointWriter {
    fn wait(&mut self) -> anyhow::Result<()> {
        if let Some(handle) = self.in_flight.take() {
            handle
                .join()
                .map_err(|_| anyhow!("checkpoint thread panicked"))??;
        }
        Ok(())
    }

    fn spawn(&mut self, save: impl FnOnce() -> anyhow::Result<()> + Send + 'static) {
        assert!(self.in_flight.is_none());
        self.in_flight = Some(thread::spawn(save));
    }
}

impl Drop for CheckpointWriter {
    fn drop(&mut self) {
        if let Err(e) = self.wait() {
            tracing::error!("failed to save checkpoint: {e:?}");
        }
    }
}

#[derive(Debug, Clone, Default)]
struct Batch {
    tokens: Vec<TokenId>,
    labels: Vec<TokenId>,
    labels_mask: Vec<bool>,
    seqlens: Vec<u32>,
}

/// Packs documents into microbatches of exactly the microbatch size in token count.
/// Documents can be sharded across microbatches.
struct Packer {
    microbatch_size: usize,
    max_seqlen: usize,
    batch_fill: usize,
}

impl Packer {
    fn new(config: &Config) -> Self {
        Self {
            microbatch_size: config.training.microbatch_size as usize,
            max_seqlen: config.model.max_seqlen,
            batch_fill: 0,
        }
    }

    /// Calls `f` with the token range of each sequence cut from the document.
    fn pack_document(
        &mut self,
        doc_len: usize,
        mut f: impl FnMut(Range<usize>, bool) -> ControlFlow<()>,
    ) -> ControlFlow<()> {
        let mut consumed = 0;
        while consumed < doc_len {
            let tokens_to_take = (self.microbatch_size - self.batch_fill)
                .min(doc_len - consumed)
                .min(self.max_seqlen);
            self.batch_fill += tokens_to_take;
            let completes_batch = self.batch_fill == self.microbatch_size;
            if completes_batch {
                self.batch_fill = 0;
            }
            f(consumed..consumed + tokens_to_take, completes_batch)?;
            consumed += tokens_to_take;
        }
        ControlFlow::Continue(())
    }
}

/// Streams batches of exactly the microbatch size in token count.
/// Drops any tail that does not fill a full batch.
fn stream_batches(token_counts: Vec<u32>, config: &Config) -> flume::Receiver<Batch> {
    let (batches_tx, batches_rx) = flume::bounded(4);

    let config = config.clone();

    thread::spawn(move || {
        let res: anyhow::Result<()> = try {
            let mut tokens =
                SChunkReader::<TokenId>::open(&config.dataset.work_dir.join(TOKENIZED_FILE_NAME))?;
            let mut packer = Packer::new(&config);
            let mut current_batch = Batch::default();

            for &count in &token_counts {
                let doc = tokens
                    .read(count as usize)?
                    .ok_or_else(|| anyhow!("tokenized file has fewer tokens than counted"))?;

                let flow = packer.pack_document(doc.len(), |range, completes_batch| {
                    current_batch.tokens.extend_from_slice(&doc[range.clone()]);
                    current_batch.seqlens.push(range.len() as u32);

                    for i in range.start + 1..range.end + 1 {
                        let (label, mask) = match doc.get(i) {
                            Some(&tok) => (tok, true),
                            None => (TokenId::PLACEHOLDER, false),
                        };
                        current_batch.labels.push(label);
                        current_batch.labels_mask.push(mask);
                    }

                    if completes_batch && batches_tx.send(mem::take(&mut current_batch)).is_err() {
                        return ControlFlow::Break(());
                    }
                    ControlFlow::Continue(())
                });
                if flow.is_break() {
                    break;
                }
            }
        };
        if let Err(e) = res {
            tracing::error!("error: {e:?}");
            exit(1);
        }
    });

    batches_rx
}
