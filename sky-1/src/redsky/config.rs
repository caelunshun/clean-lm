use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub seed: u64,
    pub dataset: DatasetConfig,
    pub training: TrainingConfig,
}

#[derive(Debug, Deserialize)]
pub struct DatasetConfig {}

#[derive(Debug, Deserialize)]
pub struct TrainingConfig {
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
