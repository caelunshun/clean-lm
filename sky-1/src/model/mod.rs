use std::sync::Arc;

use cleandl::{
    context::Context,
    operators::{embedding::EmbeddingExecutor, linear::LinearExecutor, norm::NormExecutor},
    parameter::Parameter,
    tensor::Tensor,
};
use serde::{Deserialize, Serialize};

use crate::model::decoder::{DecoderExecutor, DecoderWeights};

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
    decoder: Vec<DecoderWeights>,
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
}
