use std::sync::Arc;

use cleandl::{
    context::{Context, ResourceKey},
    operators::{
        attention::{AttentionConfig, AttentionExecutor, CuSeqlenTensors},
        linear::{LinearConfig, LinearExecutor},
        norm::NormExecutor,
        rope::{RopeConfig, RopeExecutor},
        swiglu_ffn::{SwiGluConfig, SwiGluFfnExecutor},
    },
    parameter::Parameter,
    tensor::{Tensor, TensorView},
};
use serde::Serialize;
use skycuda::stream::Stream;
use skytens_core::data_type::{ElementType, FloatType};

use crate::model::{BlocksConfig, MetaTensors};

#[derive(Debug)]
pub struct DecoderExecutor {
    config: BlocksConfig,
    cx: Context,
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
pub struct DecoderExecutorKey {
    pub config: BlocksConfig,
}

impl ResourceKey for DecoderExecutorKey {
    type Resource = DecoderExecutor;

    fn create(&self, cx: &Context) -> anyhow::Result<Self::Resource> {
        let pre_qkv_norm = NormExecutor::get(cx)?;
        let qkv_proj = LinearExecutor::get(
            LinearConfig {
                estimated_batch_size: self.config.max_batch_decoder_tokens,
                in_features: self.config.hidden_size,
                out_features: self.config.head_dim
                    * (self.config.num_decoder_q_heads + 2 * self.config.num_decoder_kv_heads),
                forward_fused_residual_add: false,
            },
            cx,
        )?;
        let rope = RopeExecutor::get(
            RopeConfig {
                base_freq: self.config.rope_base_freq,
                max_seqlen: self.config.max_decoder_seqlen,
                head_dim: self.config.head_dim,
            },
            cx,
        )?;
        let qk_norm = NormExecutor::get(cx)?;
        let attention = AttentionExecutor::get(
            AttentionConfig {
                is_causal: true,
                max_q_seqlen: self.config.max_decoder_seqlen,
                max_kv_seqlen: self.config.max_decoder_seqlen,
                max_num_sequences: self.config.max_num_sequences,
                q_heads: self.config.num_decoder_q_heads,
                kv_heads: self.config.num_decoder_kv_heads,
                head_dim: self.config.head_dim,
            },
            cx,
        )?;
        let o_proj = LinearExecutor::get(
            LinearConfig {
                estimated_batch_size: self.config.max_batch_decoder_tokens,
                in_features: self.config.num_decoder_q_heads * self.config.head_dim,
                out_features: self.config.hidden_size,
                forward_fused_residual_add: true,
            },
            cx,
        )?;
        let pre_ffn_norm = NormExecutor::get(cx)?;
        let ffn = SwiGluFfnExecutor::get(
            SwiGluConfig {
                estimated_batch_size: self.config.max_batch_decoder_tokens,
                in_features: self.config.hidden_size,
                intermediate_features: self.config.ffn_intermediate_size,
                out_features: self.config.hidden_size,
            },
            cx,
        )?;

        Ok(DecoderExecutor {
            config: self.config.clone(),
            cx: cx.clone(),
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
pub struct DecoderWeights {
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
pub struct DecoderSavedTensors {
    pre_qkv_norm_r_inv: Tensor,
    pre_qkv_norm_result: Tensor,
    qkv_proj_result: Tensor,
    qk_norm_result: Tensor,
    qk_norm_r_inv: Tensor,
    attn_result: Tensor,
    attn_stats: Tensor,
    o_proj_result: Tensor,
    pre_ffn_norm_result: Tensor,
    pre_ffn_norm_r_inv: Tensor,
    ffn_up_result: Tensor,
    ffn_gate_result: Tensor,
    ffn_gate_activation_result: Tensor,
    ffn_down_result: Tensor,
}

impl DecoderSavedTensors {
    pub fn output(&self) -> &Tensor {
        &self.ffn_down_result
    }
}

impl DecoderExecutor {
    pub fn train_forward(
        &self,
        weights: &DecoderWeights,
        x: &TensorView,
        meta_tensors: &MetaTensors,
        stream: &Stream,
    ) -> anyhow::Result<DecoderSavedTensors> {
        let &[batch_size, hidden_size] = x.shape().dims_slice() else {
            panic!()
        };
        assert_eq!(hidden_size, self.config.hidden_size);

        // --- Attention pre-norm --- //
        let mut pre_qkv_norm_r_inv = Tensor::new_zeroed(
            [batch_size],
            ElementType::Float(FloatType::F32),
            stream,
            &self.cx,
        )?;
        let mut pre_qkv_norm_result = Tensor::new_zeroed(
            [batch_size],
            ElementType::Float(FloatType::Bf16),
            stream,
            &self.cx,
        )?;
        self.pre_qkv_norm.rmsnorm_forward(
            x,
            &weights.pre_qkv_norm_gamma.bf16_weights().view(),
            &mut pre_qkv_norm_result.view_mut(),
            &mut pre_qkv_norm_r_inv.view_mut(),
            stream,
        )?;

        // --- QKV projection --- //
        let qkv_dim = self.config.head_dim
            * (self.config.num_decoder_q_heads + 2 * self.config.num_decoder_kv_heads);
        let mut qkv_proj_result = Tensor::new_zeroed(
            [batch_size, qkv_dim],
            ElementType::Float(FloatType::Bf16),
            stream,
            &self.cx,
        )?;
        self.qkv_proj.forward(
            &pre_qkv_norm_result.view(),
            &weights.qkv_proj.bf16_weights().view(),
            &mut qkv_proj_result.view_mut(),
            stream,
        )?;

        // --- QK norm --- //
        let qk = qkv_proj_result.sliced(
            1,
            0..self.config.head_dim
                * (self.config.num_decoder_q_heads + self.config.num_decoder_kv_heads),
        );
        let mut qk_norm_result =
            Tensor::new_zeroed(qk.shape(), qk.element_type(), stream, &self.cx)?;
        let mut qk_norm_r_inv = Tensor::new_zeroed(
            [batch_size],
            ElementType::Float(FloatType::F32),
            stream,
            &self.cx,
        )?;
        self.qk_norm.rmsnorm_forward(
            &qk,
            &weights.qk_norm_gamma.bf16_weights().view(),
            &mut qk_norm_result.view_mut(),
            &mut qk_norm_r_inv.view_mut(),
            stream,
        )?;

        // --- Attention --- //
        let q = qk_norm_result.sliced(1, 0..self.config.head_dim * self.config.num_decoder_q_heads);
        let k = qk_norm_result.sliced(
            1,
            self.config.head_dim * self.config.num_decoder_q_heads
                ..self.config.head_dim
                    * (self.config.num_decoder_q_heads + self.config.num_decoder_kv_heads),
        );
        let v = qkv_proj_result.sliced(
            1,
            self.config.head_dim
                * (self.config.num_decoder_q_heads + self.config.num_decoder_kv_heads)..,
        );
        let mut attn_result = Tensor::new_zeroed(q.shape(), q.element_type(), stream, &self.cx)?;
        let mut attn_stats = Tensor::new_zeroed(
            [
                self.config.max_num_sequences,
                self.config.max_decoder_seqlen,
                self.config.num_decoder_q_heads,
            ],
            ElementType::Float(FloatType::F32),
            stream,
            &self.cx,
        )?;
        self.attention.forward_attn(
            &q,
            &k,
            &v,
            &CuSeqlenTensors {
                cu_seqlen_q_times_hd_times_qh: meta_tensors
                    .decoder_cu_seqlen_times_hd_times_qh
                    .view(),
                cu_seqlen_kv_times_hd_times_kvh: meta_tensors
                    .decoder_cu_seqlen_times_hd_times_kvh
                    .view(),
            },
            &meta_tensors.decoder_seqlen.view(),
            &meta_tensors.decoder_seqlen.view(),
            &mut attn_result.view_mut(),
            &mut attn_stats.view_mut(),
            stream,
        )?;

        // --- O-proj --- //
        let mut o_proj_result = Tensor::new_zeroed(x.shape(), x.element_type(), stream, &self.cx)?;
        self.o_proj.forward_with_residual(
            &attn_result.view(),
            &weights.o_proj.bf16_weights().view(),
            x,
            &mut o_proj_result.view_mut(),
            stream,
        )?;

        // --- FFN pre-norm --- //
        let mut pre_ffn_norm_r_inv = Tensor::new_zeroed(
            [batch_size],
            ElementType::Float(FloatType::F32),
            stream,
            &self.cx,
        )?;
        let mut pre_ffn_norm_result = Tensor::new_zeroed(
            [batch_size],
            ElementType::Float(FloatType::Bf16),
            stream,
            &self.cx,
        )?;
        self.pre_ffn_norm.rmsnorm_forward(
            x,
            &weights.pre_ffn_norm_gamma.bf16_weights().view(),
            &mut pre_ffn_norm_result.view_mut(),
            &mut pre_ffn_norm_r_inv.view_mut(),
            stream,
        )?;

        // --- FFN --- //
        let mut ffn_up_result = Tensor::new_zeroed(
            [batch_size, self.config.ffn_intermediate_size],
            x.element_type(),
            stream,
            &self.cx,
        )?;
        self.ffn.forward_up_proj(
            &pre_ffn_norm_result.view(),
            &weights.ffn_up.bf16_weights().view(),
            &mut ffn_up_result.view_mut(),
            stream,
        )?;

        let mut ffn_gate_result = Tensor::new_zeroed(
            [batch_size, self.config.ffn_intermediate_size],
            x.element_type(),
            stream,
            &self.cx,
        )?;
        let mut ffn_gate_activation_result = Tensor::new_zeroed(
            [batch_size, self.config.ffn_intermediate_size],
            x.element_type(),
            stream,
            &self.cx,
        )?;
        self.ffn.forward_gate_proj(
            &pre_ffn_norm_result.view(),
            &ffn_up_result.view(),
            &weights.ffn_gate.bf16_weights().view(),
            &mut ffn_gate_result.view_mut(),
            &mut ffn_gate_activation_result.view_mut(),
            stream,
        )?;

        let mut ffn_down_result =
            Tensor::new_zeroed(x.shape(), x.element_type(), stream, &self.cx)?;
        self.ffn.forward_down_proj(
            &ffn_gate_activation_result.view(),
            &weights.ffn_down.bf16_weights().view(),
            &o_proj_result.view(),
            &mut ffn_down_result.view_mut(),
            stream,
        )?;

        Ok(DecoderSavedTensors {
            pre_qkv_norm_r_inv,
            pre_qkv_norm_result,
            qkv_proj_result,
            qk_norm_result,
            qk_norm_r_inv,
            attn_result,
            attn_stats,
            o_proj_result,
            pre_ffn_norm_result,
            pre_ffn_norm_r_inv,
            ffn_up_result,
            ffn_gate_result,
            ffn_gate_activation_result,
            ffn_down_result,
        })
    }
}
