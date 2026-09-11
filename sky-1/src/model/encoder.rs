use std::sync::Arc;

use cleandl::{
    cudnn::ExecutableCudnnGraph,
    operators::{
        attention::AttentionExecutor, matmul::MatmulExecutor, norm::NormExecutor,
        rope::RopeExecutor, swiglu_ffn::SwiGluFfnExecutor,
    },
};
use singe_cudnn::backend::tensor::TensorId;

use crate::model::BlocksConfig;

#[derive(Debug)]
pub struct EncoderExecutor {
    config: BlocksConfig,
    input_norm: Arc<NormExecutor>,
    rope: Arc<RopeExecutor>,
    qk_norm: Arc<NormExecutor>,
    attention: Arc<AttentionExecutor>,
    o_proj_fwd: Arc<MatmulExecutor>,
    o_proj_bwd: Arc<MatmulExecutor>,
    o_norm: Arc<NormExecutor>,
    ffn: Arc<SwiGluFfnExecutor>,
}
