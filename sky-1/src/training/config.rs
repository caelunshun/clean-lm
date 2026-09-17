#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub adam_beta1: f32,
    pub adam_beta2: f32,
    pub weight_decay: f32,
    pub max_gradient_norm: f32,
    pub lr: f32,
}
