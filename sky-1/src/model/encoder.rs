use std::sync::Arc;

use cleandl::{
    context::{Context, ResourceKey},
    operators::{
        attention::{AttentionConfig, AttentionExecutor},
        linear::{LinearConfig, LinearExecutor},
        norm::NormExecutor,
        rope::{RopeConfig, RopeExecutor},
        swiglu_ffn::{SwiGluConfig, SwiGluFfnExecutor},
    },
    parameter::Parameter,
    tensor::Tensor,
};
use serde::Serialize;

use crate::model::BlocksConfig;

#[derive(Debug)]
pub struct EncoderExecutor {
    config: BlocksConfig,
    pre_qkv_norm: Arc<NormExecutor>,
    qkv_proj: Arc<LinearExecutor>,
    rope: Arc<RopeExecutor>,
    qk_norm: Arc<NormExecutor>,
    attention: Arc<AttentionExecutor>,
    o_proj: Arc<LinearExecutor>,
    pre_ffn_norm: Arc<NormExecutor>,
    ffn: Arc<SwiGluFfnExecutor>,
}

#[derive(Debug, Serialize)]
pub struct EncoderExecutorKey {
    pub config: BlocksConfig,
}

impl ResourceKey for EncoderExecutorKey {
    type Resource = EncoderExecutor;

    fn create(&self, cx: &Context) -> anyhow::Result<Self::Resource> {
        let pre_qkv_norm = NormExecutor::get(cx)?;
        let qkv_proj = LinearExecutor::get(
            LinearConfig {
                estimated_batch_size: self.config.max_batch_encoder_tokens,
                in_features: self.config.hidden_size,
                out_features: 3 * self.config.head_dim * self.config.num_heads,
                forward_fused_residual_add: false,
            },
            cx,
        )?;
        let rope = RopeExecutor::get(
            RopeConfig {
                base_freq: self.config.rope_base_freq,
                max_seqlen: self.config.max_encoder_seqlen,
                head_dim: self.config.head_dim,
            },
            cx,
        )?;
        let qk_norm = NormExecutor::get(cx)?;
        let attention = AttentionExecutor::get(
            AttentionConfig {
                is_causal: false,
                max_q_seqlen: self.config.max_encoder_seqlen,
                max_kv_seqlen: self.config.max_encoder_seqlen,
                max_num_sequences: self.config.max_num_sequences,
                q_heads: self.config.num_heads,
                kv_heads: self.config.num_heads,
                head_dim: self.config.head_dim,
            },
            cx,
        )?;
        let o_proj = LinearExecutor::get(
            LinearConfig {
                estimated_batch_size: self.config.max_batch_encoder_tokens,
                in_features: self.config.num_heads * self.config.head_dim,
                out_features: self.config.hidden_size,
                forward_fused_residual_add: true,
            },
            cx,
        )?;
        let pre_ffn_norm = NormExecutor::get(cx)?;
        let ffn = SwiGluFfnExecutor::get(
            SwiGluConfig {
                estimated_batch_size: self.config.max_batch_encoder_tokens,
                in_features: self.config.hidden_size,
                intermediate_features: self.config.ffn_intermediate_size,
                out_features: self.config.hidden_size,
            },
            cx,
        )?;

        Ok(EncoderExecutor {
            config: self.config.clone(),
            pre_qkv_norm,
            qkv_proj,
            rope,
            qk_norm,
            attention,
            o_proj,
            pre_ffn_norm,
            ffn,
        })
    }
}

#[derive(Debug)]
pub struct EncoderWeights {
    pub pre_qkv_norm_gamma: Parameter,
    pub qkv_proj: Parameter,
    pub qk_norm_gamma: Parameter,
    pub o_proj: Parameter,
    pub pre_ffn_norm_gamma: Parameter,
    pub ffn_up: Parameter,
    pub ffn_gate: Parameter,
    pub ffn_down: Parameter,
}

/// Saved tensors for the backward pass.
#[derive(Debug)]
pub struct EncoderSavedTensors {
    pre_qkv_norm_r_inv: Tensor,
    pre_qkv_norm_result: Tensor,
    qkv_proj_result: Tensor,
    qk_norm_result: Tensor,
    qk_norm_r_inv: Tensor,
    attention_result: Tensor,
    attention_stats: Tensor,
    o_proj_result: Tensor,
    pre_ffn_norm_result: Tensor,
    pre_ffn_norm_r_inv: Tensor,
    ffn_up_result: Tensor,
    ffn_gate_result: Tensor,
    ffn_down_result: Tensor,
}

impl EncoderSavedTensors {
    pub fn output(&self) -> &Tensor {
        &self.ffn_down_result
    }
}

impl EncoderExecutor {
    /// Executes forward pass given output
    /// of the previous layer.
    pub fn train_forward(&self, weights: &EncoderWeights) -> anyhow::Result<EncoderSavedTensors> {
        todo!()
    }
}
