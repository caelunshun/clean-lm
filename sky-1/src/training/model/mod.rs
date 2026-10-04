use crate::{
    model::Sky1Config,
    training::{
        checkpoint::{CheckpointBuilder, CheckpointLoader},
        parameter::Parameter,
        weight_update_params::WeightUpdateParams,
    },
};
use cleandl::{
    context::{Context, ResourceKey},
    operators::{
        adamw::AdamWExecutor,
        attention::RaggedSeqlenTensors,
        embedding::EmbeddingExecutor,
        gradient_accumulation::GradientAccumulationExecutor,
        linear::{LinearConfig, LinearExecutor},
        loss::{CrossEntropyLossOutput, LossExecutor},
        norm::{NormExecutor, RmsNormForwardOutput},
    },
    tensor::{Tensor, TensorView},
};
use decoder::{DecoderExecutor, DecoderSavedTensors, DecoderWeights};
use rand::Rng;
use serde::Serialize;
use skycuda::stream::Stream;
use skytens_core::data_type::{ElementType, FloatType};
use std::sync::Arc;

pub mod decoder;

#[derive(Debug)]
pub struct MetaTensors {
    pub decoder_ragged_seqlen: RaggedSeqlenTensors,
    pub seqlen: Vec<u32>,
}

#[derive(Debug)]
pub struct Sky1Weights {
    embeddings: Parameter,
    decoder_blocks: Vec<DecoderWeights>,
    final_norm_gamma: Parameter,
    lm_head: Parameter,
}

impl Sky1Weights {
    pub fn new_random(
        config: &Sky1Config,
        rng: &mut impl Rng,
        stream: &Stream,
        cx: &Context,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            embeddings: Parameter::from_normal_distr(
                [config.vocab_size, config.blocks.hidden_size],
                0.0,
                1.0 / (config.blocks.hidden_size as f32).sqrt(),
                rng,
                stream,
                cx,
            )?,
            decoder_blocks: (0..config.num_decoder_blocks)
                .map(|_| DecoderWeights::new_random(config, rng, stream, cx))
                .collect::<Result<Vec<_>, anyhow::Error>>()?,
            final_norm_gamma: Parameter::constant([config.blocks.hidden_size], 1.0, stream, cx)?,
            lm_head: Parameter::from_normal_distr(
                [config.blocks.hidden_size, config.vocab_size],
                0.0,
                1.0 / (config.blocks.hidden_size as f32).sqrt(),
                rng,
                stream,
                cx,
            )?,
        })
    }

    pub fn save_checkpoint(
        &self,
        builder: &mut CheckpointBuilder,
        cx: &Context,
        stream: &Stream,
    ) -> anyhow::Result<()> {
        builder.build_child("embeddings", |builder| {
            builder.add_param("weight", &self.embeddings, cx, stream)?;
            Ok(())
        })?;

        for (i, decoder) in self.decoder_blocks.iter().enumerate() {
            builder.build_child(&format!("decoder_blocks[{i}]"), |builder| {
                decoder.save_checkpoint(builder, cx, stream)
            })?;
        }

        builder.build_child("final_norm", |builder| {
            builder.add_param("gamma", &self.final_norm_gamma, cx, stream)?;
            Ok(())
        })?;

        builder.build_child("lm_head", |builder| {
            builder.add_param("weight", &self.lm_head, cx, stream)?;
            Ok(())
        })?;

        Ok(())
    }

    pub fn load_from_checkpoint(
        config: &Sky1Config,
        loader: &mut CheckpointLoader,
        cx: &Context,
        stream: &Stream,
    ) -> anyhow::Result<Self> {
        let embeddings = loader.load_child("embeddings", |loader| {
            loader.load_param("weight", cx, stream)
        })?;

        let decoder_blocks = (0..config.num_decoder_blocks)
            .map(|i| {
                loader.load_child(&format!("decoder_blocks[{i}]"), |loader| {
                    DecoderWeights::load_from_checkpoint(loader, cx, stream)
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        let final_norm_gamma = loader.load_child("final_norm", |loader| {
            loader.load_param("gamma", cx, stream)
        })?;

        let lm_head =
            loader.load_child("lm_head", |loader| loader.load_param("weight", cx, stream))?;

        Ok(Self {
            embeddings,
            decoder_blocks,
            final_norm_gamma,
            lm_head,
        })
    }
}

#[derive(Debug)]
pub struct Sky1Executor {
    #[expect(unused)]
    config: Sky1Config,
    cx: Context,
    embedding: Arc<EmbeddingExecutor>,
    decoder: Arc<DecoderExecutor>,
    final_norm: Arc<NormExecutor>,
    lm_head: Arc<LinearExecutor>,
    loss: Arc<LossExecutor>,
    gradient_accumulation: Arc<GradientAccumulationExecutor>,
    optimizer: Arc<AdamWExecutor>,
}

impl Sky1Executor {
    pub fn get(config: Sky1Config, cx: &Context) -> anyhow::Result<Arc<Self>> {
        cx.get_or_create_resource(Sky1ExecutorKey { config })
    }
}

#[derive(Debug, Serialize)]
struct Sky1ExecutorKey {
    config: Sky1Config,
}

impl ResourceKey for Sky1ExecutorKey {
    type Resource = Sky1Executor;

    fn create(&self, cx: &Context) -> anyhow::Result<Self::Resource> {
        let embedding = EmbeddingExecutor::get(cx)?;
        let decoder = DecoderExecutor::get(self.config.blocks.clone(), cx)?;
        let final_norm = NormExecutor::get(cx)?;
        let lm_head = LinearExecutor::get(
            LinearConfig {
                estimated_batch_size: self.config.blocks.max_batch_decoder_tokens,
                in_features: self.config.blocks.hidden_size,
                out_features: self.config.vocab_size,
                forward_fused_residual_add: false,
            },
            cx,
        )?;
        let loss = LossExecutor::get(cx)?;
        let optimizer = AdamWExecutor::get(cx)?;

        Ok(Sky1Executor {
            config: self.config.clone(),
            cx: cx.clone(),
            embedding,
            decoder,
            final_norm,
            lm_head,
            loss,
            gradient_accumulation: GradientAccumulationExecutor::get(cx)?,
            optimizer,
        })
    }
}

#[derive(Debug)]
pub struct Sky1SavedTensors {
    embedding_output: Tensor,
    decoder: Vec<DecoderSavedTensors>,
    final_norm_r_inv: Tensor,
    final_norm_result: Tensor,
    logits_grad: Tensor,
    loss: Tensor,
}

impl Sky1SavedTensors {
    pub fn loss(&self) -> &Tensor {
        &self.loss
    }
}

impl Sky1Executor {
    pub fn train_forward(
        &self,
        token_indices: &TensorView,
        labels: &TensorView,
        labels_mask: &[bool],
        meta_tensors: &MetaTensors,
        weights: &Sky1Weights,
        stream: &Stream,
    ) -> anyhow::Result<Sky1SavedTensors> {
        let embedding_output = self.embedding.embedding_lookup(
            &weights.embeddings.master_weights.view(),
            token_indices,
            stream,
        )?;

        let mut decoder_saves: Vec<DecoderSavedTensors> = Vec::new();
        for decoder_block_weights in &weights.decoder_blocks {
            let block_input = match decoder_saves.last() {
                Some(last) => last.output().view(),
                None => embedding_output.view(),
            };

            let saves = self.decoder.train_forward(
                decoder_block_weights,
                &block_input,
                meta_tensors,
                stream,
            )?;
            decoder_saves.push(saves);
        }

        let output = decoder_saves.last().unwrap().output().view();

        let RmsNormForwardOutput {
            y: final_norm_result,
            r_inv: final_norm_r_inv,
        } = self.final_norm.rmsnorm_forward(
            &output,
            &weights.final_norm_gamma.bf16_weights().view(),
            stream,
        )?;

        let logits = self.lm_head.forward(
            &final_norm_result.view(),
            &weights.lm_head.bf16_weights().view(),
            stream,
        )?;

        let loss_stats = self.loss.online_softmax_stats(&logits.view(), stream)?;
        let CrossEntropyLossOutput { loss, logits_grad } = self.loss.cross_entropy_loss_fwd_bwd(
            &logits.view(),
            &loss_stats.view(),
            labels,
            labels_mask,
            stream,
        )?;

        Ok(Sky1SavedTensors {
            embedding_output,
            decoder: decoder_saves,
            final_norm_r_inv,
            final_norm_result,
            logits_grad,
            loss,
        })
    }

    pub fn train_backward(
        &self,
        token_indices: &TensorView,
        meta_tensors: &MetaTensors,
        weights: &mut Sky1Weights,
        saves: &Sky1SavedTensors,
        stream: &Stream,
    ) -> anyhow::Result<()> {
        let lm_head_dw = self.lm_head.backward_wgrad(
            &saves.final_norm_result.view(),
            &saves.logits_grad.view(),
            stream,
        )?;
        self.gradient_accumulation.add_contribution_to_gradient(
            &lm_head_dw.view(),
            &mut weights.lm_head.gradients.view_mut(),
            stream,
        )?;

        let lm_head_ddata = self.lm_head.backward_dgrad(
            &saves.logits_grad.view(),
            &weights.lm_head.bf16_weights().view(),
            stream,
        )?;

        let final_norm_ddata = self.final_norm.rmsnorm_backward(
            &saves.decoder.last().unwrap().output().view(),
            &lm_head_ddata.view(),
            None,
            &weights.final_norm_gamma.bf16_weights_replica.view(),
            &saves.final_norm_r_inv.view(),
            &mut weights.final_norm_gamma.gradients.view_mut(),
            stream,
        )?;

        let mut dy = final_norm_ddata;
        for i in (0..weights.decoder_blocks.len()).rev() {
            let x = if i == 0 {
                saves.embedding_output.view()
            } else {
                saves.decoder[i - 1].output().view()
            };
            dy = self.decoder.train_backward(
                &dy.view(),
                &x,
                &mut weights.decoder_blocks[i],
                &saves.decoder[i],
                meta_tensors,
                stream,
            )?;
        }

        self.embedding.embedding_scatter_add(
            token_indices,
            &dy.view(),
            &mut weights.embeddings.gradients.view_mut(),
            stream,
        )?;

        Ok(())
    }

    pub fn weight_update(
        &self,
        weights: &mut Sky1Weights,
        config: &WeightUpdateParams,
        train_step: u32,
        stream: &Stream,
    ) -> anyhow::Result<()> {
        let mut grad_sum2 = Tensor::new_zeroed([], ElementType::Float(FloatType::F32), &self.cx)?;
        for param in [
            &weights.embeddings,
            &weights.final_norm_gamma,
            &weights.lm_head,
        ] {
            self.optimizer.gradient_l2_norm_reduction(
                &param.gradients.view(),
                &mut grad_sum2.view_mut(),
                stream,
            )?;
        }
        for block in &weights.decoder_blocks {
            self.decoder.visit_gradients(block, |gradient| {
                self.optimizer.gradient_l2_norm_reduction(
                    gradient,
                    &mut grad_sum2.view_mut(),
                    stream,
                )
            })?;
        }

        for param in [
            &mut weights.embeddings,
            &mut weights.final_norm_gamma,
            &mut weights.lm_head,
        ] {
            self.optimizer.adamw_weight_update(
                &param.gradients.view(),
                &grad_sum2.view(),
                &mut param.master_weights.view_mut(),
                &mut param.bf16_weights_replica.view_mut(),
                &mut param.adam_m.view_mut(),
                &mut param.adam_v.view_mut(),
                config.adam_beta1,
                config.adam_beta2,
                config.weight_decay,
                config.max_gradient_norm,
                config.lr,
                train_step,
                stream,
            )?;
            stream.fill(&mut param.gradients.storage_mut(), 0u8)?;
        }

        for decoder in &mut weights.decoder_blocks {
            self.decoder
                .weight_update(decoder, &grad_sum2.view(), config, train_step, stream)?;
        }

        Ok(())
    }
}
