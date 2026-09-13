use std::sync::Arc;

use cleandl::{
    context::Context,
    operators::{
        embedding::EmbeddingExecutor, linear::LinearExecutor, loss::LossExecutor,
        norm::NormExecutor,
    },
    parameter::Parameter,
    tensor::{Tensor, TensorView},
};
use serde::{Deserialize, Serialize};
use skycuda::stream::Stream;
use skytens_core::data_type::{ElementType, FloatType};

use crate::model::decoder::{DecoderExecutor, DecoderSavedTensors, DecoderWeights};

pub mod decoder;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sky1Config {
    pub blocks_config: BlocksConfig,
    pub num_encoder_blocks: usize,
    pub num_decoder_blocks: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlocksConfig {
    pub hidden_size: usize,
    pub ffn_intermediate_size: usize,
    pub num_decoder_q_heads: usize,
    pub num_decoder_kv_heads: usize,
    pub head_dim: usize,
    pub max_batch_decoder_tokens: usize,
    pub max_num_sequences: usize,
    pub max_decoder_seqlen: usize,
    pub rope_base_freq: f32,
}

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
            [batch_size, self.config.blocks_config.hidden_size],
            ElementType::Float(FloatType::Bf16),
            stream,
            &self.cx,
        )?;
        self.embedding.embedding_lookup(
            &weights.embeddings.unwrap_training().master_weights.view(),
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

        let mut loss_stats = Tensor::new_zeroed(
            [batch_size],
            ElementType::Float(FloatType::F32),
            stream,
            &self.cx,
        )?;
        let mut loss =
            Tensor::new_zeroed([1], ElementType::Float(FloatType::F32), stream, &self.cx)?;
        let mut logits_grad = Tensor::new_zeroed(
            final_norm_result.shape(),
            final_norm_result.element_type(),
            stream,
            &self.cx,
        )?;

        self.loss.online_softmax_stats(
            &final_norm_result.view(),
            &mut loss_stats.view_mut(),
            stream,
        )?;
        self.loss.cross_entropy_loss_fwd_bwd(
            &final_norm_result.view(),
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
