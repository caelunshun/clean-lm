use cleandl::operators::attention::AttentionConfig;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sky1Config {
    pub blocks: BlocksConfig,
    pub vocab_size: usize,
    pub num_encoder_blocks: usize,
    pub num_decoder_blocks: usize,
}

impl Sky1Config {
    pub fn decoder_attention_config(&self) -> AttentionConfig {
        self.blocks.decoder_attention_config()
    }
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

impl BlocksConfig {
    /// Output width of the merged QKV projection.
    pub fn qkv_features(&self) -> usize {
        self.head_dim * (self.num_decoder_q_heads + 2 * self.num_decoder_kv_heads)
    }

    pub fn decoder_attention_config(&self) -> AttentionConfig {
        AttentionConfig {
            is_causal: true,
            max_q_seqlen: self.max_decoder_seqlen,
            max_kv_seqlen: self.max_decoder_seqlen,
            max_num_sequences: self.max_num_sequences,
            q_heads: self.num_decoder_q_heads,
            kv_heads: self.num_decoder_kv_heads,
            head_dim: self.head_dim,
        }
    }
}
