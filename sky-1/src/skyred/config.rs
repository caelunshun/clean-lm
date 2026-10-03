use std::path::PathBuf;

use compact_str::CompactString;
use serde::Deserialize;
use skytens_core::FastIndexSet;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub dataset: DatasetConfig,
    pub training: TrainingConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DatasetConfig {
    pub seed: u64,
    pub work_dir: PathBuf,
    pub reddit_ingest_bucket: String,
    pub reddit_ingest_path: String,
    pub post_sample_size: u32,
    /// In `[0, 1]`: quotas are a mixture of `(1 - mix) * proportional + mix * uniform`
    /// across subreddits within a year.
    pub subreddit_uniform_mix: f64,
    /// As `subreddit_uniform_mix`, but across years.
    pub year_uniform_mix: f64,
    pub prune_texts: FastIndexSet<CompactString>,
    pub prune_authors: FastIndexSet<CompactString>,
    /// Total vocabulary size including special tokens.
    pub total_vocab_size: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TrainingConfig {
    pub seed: u64,
    pub microbatch_size: u32,
    pub gradient_accumulation_steps: u32,
    pub adam_beta1: f32,
    pub adam_beta2: f32,
    pub weight_decay: f32,
    pub max_gradient_norm: f32,
    pub lr_schedule: LrSchedule,
    pub lr_warmup_steps: Option<u32>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum LrSchedule {
    Constant { lr: f32 },
}
