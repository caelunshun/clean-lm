use std::path::PathBuf;

use compact_str::CompactString;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub dataset: DatasetConfig,
    pub training: TrainingConfig,
}

#[derive(Debug, Deserialize)]
pub struct DatasetConfig {
    pub seed: u64,
    pub work_dir: PathBuf,
    pub reddit_ingest_bucket: String,
    pub reddit_ingest_path: String,
    pub post_sample_size: u32,
    pub subreddit_sampling_bias: f64,
    pub year_sampling_bias: f64,
    pub prune_texts: Vec<CompactString>,
    pub prune_authors: Vec<CompactString>,
    /// Total vocabulary size including special tokens.
    pub total_vocab_size: u32,
}

#[derive(Debug, Deserialize)]
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

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum LrSchedule {
    Constant(f32),
}
