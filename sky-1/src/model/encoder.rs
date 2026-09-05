use cleandl::cudnn::ExecutableCudnnGraph;
use singe_cudnn::backend::tensor::TensorId;

use crate::model::BlocksConfig;

#[derive(Debug)]
pub struct EncoderExecutor {
    config: BlocksConfig,
}
