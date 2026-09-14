use std::sync::Arc;

use cleandl::{
    context::{Context, ResourceKey},
    operators::{
        embedding::EmbeddingExecutor,
        linear::{LinearConfig, LinearExecutor},
        loss::LossExecutor,
        norm::NormExecutor,
    },
    tensor::{Tensor, TensorView},
};
use rand::Rng;
use serde::Serialize;
use skycuda::stream::Stream;
use skytens_core::data_type::{ElementType, FloatType};

use crate::{
    model::{BlocksConfig, Sky1Config},
    training::parameter::Parameter,
};
use decoder::{DecoderExecutor, DecoderSavedTensors, DecoderWeights};

pub mod decoder;

#[derive(Debug)]
pub struct MetaTensors {
    pub decoder_seqlen: Tensor,
    pub decoder_cu_seqlen: Tensor,
    pub decoder_cu_seqlen_times_hd_times_qh: Tensor,
    pub decoder_cu_seqlen_times_hd_times_kvh: Tensor,
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
}

#[derive(Debug)]
pub struct Sky1Executor {
    config: Sky1Config,
    cx: Context,
    embedding: Arc<EmbeddingExecutor>,
    decoder: Arc<DecoderExecutor>,
    final_norm: Arc<NormExecutor>,
    lm_head: Arc<LinearExecutor>,
    loss: Arc<LossExecutor>,
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

        Ok(Sky1Executor {
            config: self.config.clone(),
            cx: cx.clone(),
            embedding,
            decoder,
            final_norm,
            lm_head,
            loss,
        })
    }
}

#[derive(Debug)]
pub struct Sky1SavedTensors {
    decoder: Vec<DecoderSavedTensors>,
    logits: Tensor,
    logits_grad: Tensor,
    loss: Tensor,
}

impl Sky1SavedTensors {
    pub fn logits(&self) -> &Tensor {
        &self.logits
    }

    pub fn loss(&self) -> &Tensor {
        &self.loss
    }
}

impl Sky1Executor {
    pub fn train_forward(
        &self,
        token_indices: &TensorView,
        labels: &TensorView,
        meta_tensors: &MetaTensors,
        weights: &Sky1Weights,
        stream: &Stream,
    ) -> anyhow::Result<Sky1SavedTensors> {
        let &[batch_size] = token_indices.shape().dims_slice() else {
            panic!()
        };

        let mut x = Tensor::new_zeroed(
            [batch_size, self.config.blocks.hidden_size],
            ElementType::Float(FloatType::Bf16),
            stream,
            &self.cx,
        )?;
        self.embedding.embedding_lookup(
            &weights.embeddings.master_weights.view(),
            token_indices,
            &mut x.view_mut(),
            stream,
        )?;

        let mut decoder_saves: Vec<DecoderSavedTensors> = Vec::new();
        for decoder_block_weights in &weights.decoder_blocks {
            let block_input = match decoder_saves.last() {
                Some(last) => last.output().view(),
                None => x.view(),
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

        let mut final_norm_r_inv = Tensor::new_zeroed(
            [batch_size],
            ElementType::Float(FloatType::F32),
            stream,
            &self.cx,
        )?;
        let mut final_norm_result =
            Tensor::new_zeroed(output.shape(), output.element_type(), stream, &self.cx)?;
        self.final_norm.rmsnorm_forward(
            &output,
            &weights.final_norm_gamma.bf16_weights().view(),
            &mut final_norm_result.view_mut(),
            &mut final_norm_r_inv.view_mut(),
            stream,
        )?;

        let mut logits = Tensor::new_zeroed(
            [batch_size, self.config.vocab_size],
            ElementType::Float(FloatType::Bf16),
            stream,
            &self.cx,
        )?;
        self.lm_head.forward(
            &final_norm_result.view(),
            &weights.lm_head.bf16_weights().view(),
            &mut logits.view_mut(),
            stream,
        )?;

        let mut loss_stats = Tensor::new_zeroed(
            [batch_size],
            ElementType::Float(FloatType::F32),
            stream,
            &self.cx,
        )?;
        let mut loss =
            Tensor::new_zeroed([], ElementType::Float(FloatType::F32), stream, &self.cx)?;
        let mut logits_grad =
            Tensor::new_zeroed(logits.shape(), logits.element_type(), stream, &self.cx)?;

        self.loss
            .online_softmax_stats(&logits.view(), &mut loss_stats.view_mut(), stream)?;
        self.loss.cross_entropy_loss_fwd_bwd(
            &logits.view(),
            &loss_stats.view(),
            labels,
            &mut loss.view_mut(),
            &mut logits_grad.view_mut(),
            stream,
        )?;

        Ok(Sky1SavedTensors {
            decoder: decoder_saves,
            logits: final_norm_result,
            logits_grad,
            loss,
        })
    }
}
