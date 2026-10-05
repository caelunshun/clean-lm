use compact_str::CompactString;
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub dataset: DatasetConfig,
    pub training: TrainingConfig,
    pub model: ModelConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DatasetConfig {
    pub seed: u64,
    pub work_dir: PathBuf,
    pub reddit_ingest_bucket: String,
    pub reddit_ingest_path: String,
    pub post_sample_size: u32,
    /// Posts created before this year are excluded from the sample.
    #[serde(default)]
    pub min_year: Option<i16>,
    /// In `[0, 1]`: quotas are a mixture of `(1 - mix) * proportional + mix * uniform`
    /// across subreddits within a year.
    pub subreddit_uniform_mix: f64,
    /// As `subreddit_uniform_mix`, but across years.
    pub year_uniform_mix: f64,
    /// Matches exact text after whitespace trimming. ASCII case-sensitive.
    pub prune_texts: Vec<CompactString>,
    /// Matches substrings of text. ASCII case-insensitive.
    pub prune_phrases: Vec<CompactString>,
    /// Matches exact author username. Case-sensitive.
    pub prune_authors: Vec<CompactString>,
    /// Posts whose absolute score is below this are excluded.
    #[serde(default)]
    pub min_post_score_magnitude: Option<u64>,
    #[serde(default)]
    pub min_comment_score_magnitude: Option<u64>,
    /// Total vocabulary size including special tokens.
    pub total_vocab_size: u32,
    pub tokenizer_train_sample_limit: Option<u64>,
    pub bpe_training_fused_merges: usize,
    pub use_pretokenization: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TrainingConfig {
    pub seed: u64,
    pub checkpoints_dir: PathBuf,
    pub microbatch_size: u32,
    pub gradient_accumulation_steps: u32,
    pub adam_beta1: f32,
    pub adam_beta2: f32,
    pub weight_decay: f32,
    pub max_gradient_norm: f32,
    pub lr_schedule: LrSchedule,
    pub lr_warmup_steps: Option<u32>,
    pub enable_autotuning: bool,
    pub steps_per_checkpoint: usize,
    pub max_kept_checkpoints: usize,
    pub max_gpu_memory_gib: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub num_blocks: usize,
    pub hidden_size: usize,
    pub ffn_intermediate_size: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub max_seqlen: usize,
    pub rope_base_freq: f32,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum LrSchedule {
    Constant { lr: f32 },
    Cosine { lr_min: f32, lr_max: f32 },
}
