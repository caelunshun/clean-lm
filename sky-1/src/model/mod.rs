use serde::{Deserialize, Serialize};

pub mod encoder;

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
    pub num_heads: usize,
    pub num_decoder_kv_heads: usize,
    pub head_dim: usize,
    pub max_batch_encoder_tokens: usize,
    pub max_batch_decoder_tokens: usize,
    pub max_num_sequences: usize,
    pub max_encoder_seqlen: usize,
    pub max_decoder_seqlen: usize,
    pub rope_base_freq: f32,
}
