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
    pub total_batch_tokens: usize,
    pub max_num_sequences: usize,
}
