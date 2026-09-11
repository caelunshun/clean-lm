use std::sync::Arc;

use cleandl::{
    cudnn::ExecutableCudnnGraph,
    operators::{
        attention::AttentionExecutor, matmul::MatmulExecutor, rmsnorm::RmsNormExecutor,
        rope::RopeExecutor, swiglu_ffn::SwiGluFfnExecutor,
    },
};
use singe_cudnn::backend::tensor::TensorId;

use crate::model::BlocksConfig;

#[derive(Debug)]
pub struct EncoderExecutor {
    config: BlocksConfig,
    input_norm: Arc<RmsNormExecutor>,
    rope: Arc<RopeExecutor>,
    qk_norm: Arc<RmsNormExecutor>,
    attention: Arc<AttentionExecutor>,
    o_proj_fwd: Arc<MatmulExecutor>,
    o_proj_bwd: Arc<MatmulExecutor>,
    o_norm: Arc<RmsNormExecutor>,
    ffn: Arc<SwiGluFfnExecutor>,
}
