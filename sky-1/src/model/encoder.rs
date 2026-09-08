use std::sync::Arc;

use cleandl::{
    cudnn::ExecutableCudnnGraph,
    operators::{attention::AttentionExecutor, rmsnorm::RmsNormExecutor},
};
use singe_cudnn::backend::tensor::TensorId;

use crate::model::BlocksConfig;

#[derive(Debug)]
pub struct EncoderExecutor {
    config: BlocksConfig,
    rmsnorm: Arc<RmsNormExecutor>,
    attention: Arc<AttentionExecutor>,
}
