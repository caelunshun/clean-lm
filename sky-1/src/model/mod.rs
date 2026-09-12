use cleandl::tensor::Tensor;
use serde::{Deserialize, Serialize};

pub mod decoder;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
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
