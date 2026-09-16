use std::sync::Arc;

use cleandl::{
    context::{Context, ResourceKey},
    operators::{
        attention::{AttentionConfig, AttentionExecutor, CuSeqlenTensors},
        gradient_accumulation::GradientAccumulationExecutor,
        linear::{LinearConfig, LinearExecutor},
        matmul::{MatmulConfig, MatmulExecutor, MatrixFormat},
        norm::NormExecutor,
        rope::{RopeConfig, RopeExecutor},
        swiglu_ffn::{SwiGluConfig, SwiGluFfnExecutor},
    },
    tensor::{Tensor, TensorView},
};
use rand::Rng;
use serde::Serialize;
use skycuda::stream::Stream;
use skytens_core::data_type::{ElementType, FloatType};

use crate::{
    model::{BlocksConfig, Sky1Config},
    training::{model::MetaTensors, parameter::Parameter},
};

#[derive(Debug)]
pub struct DecoderExecutor {
    config: BlocksConfig,
    cx: Context,
    pre_qkv_norm: Arc<NormExecutor>,
    q_proj: Arc<LinearExecutor>,
    k_proj: Arc<LinearExecutor>,
    v_proj: Arc<LinearExecutor>,
    kv_proj_dgrad_with_res: Arc<MatmulExecutor>,
    #[expect(unused)] // RoPE disabled for now
    rope: Arc<RopeExecutor>,
    // qk_norm: Arc<NormExecutor>,
    attention: Arc<AttentionExecutor>,
    o_proj: Arc<LinearExecutor>,
    pre_ffn_norm: Arc<NormExecutor>,
    ffn: Arc<SwiGluFfnExecutor>,
    gradient_accumulation: Arc<GradientAccumulationExecutor>,
}

impl DecoderExecutor {
    pub fn get(config: BlocksConfig, cx: &Context) -> anyhow::Result<Arc<Self>> {
        cx.get_or_create_resource(DecoderExecutorKey { config })
    }
}

#[derive(Debug, Serialize)]
struct DecoderExecutorKey {
    config: BlocksConfig,
}

impl ResourceKey for DecoderExecutorKey {
    type Resource = DecoderExecutor;

    fn create(&self, cx: &Context) -> anyhow::Result<Self::Resource> {
        let pre_qkv_norm = NormExecutor::get(cx)?;
        let q_proj = LinearExecutor::get(
            LinearConfig {
                estimated_batch_size: self.config.max_batch_decoder_tokens,
                in_features: self.config.hidden_size,
                out_features: self.config.head_dim * self.config.num_decoder_q_heads,
                forward_fused_residual_add: false,
            },
            cx,
        )?;
        let k_proj = LinearExecutor::get(
            LinearConfig {
                estimated_batch_size: self.config.max_batch_decoder_tokens,
                in_features: self.config.hidden_size,
                out_features: self.config.head_dim * self.config.num_decoder_kv_heads,
                forward_fused_residual_add: false,
            },
            cx,
        )?;
        let kv_proj_dgrad_with_res = MatmulExecutor::get(
            MatmulConfig {
                m: self.config.max_batch_decoder_tokens,
                n: self.config.hidden_size,
                k: self.config.head_dim * self.config.num_decoder_kv_heads,
                a_format: MatrixFormat::RowMajor,
                b_format: MatrixFormat::ColumnMajor, // transpose
                residual_format: Some(MatrixFormat::RowMajor),
                out_format: MatrixFormat::RowMajor,
                allow_shape_override: true,
            },
            cx,
        )?;
        let v_proj = LinearExecutor::get(
            LinearConfig {
                estimated_batch_size: self.config.max_batch_decoder_tokens,
                in_features: self.config.hidden_size,
                out_features: self.config.head_dim * self.config.num_decoder_kv_heads,
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
        // let qk_norm = NormExecutor::get(cx)?;
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

        let gradient_accumulation = GradientAccumulationExecutor::get(cx)?;

        Ok(DecoderExecutor {
            config: self.config.clone(),
            cx: cx.clone(),
            pre_qkv_norm,
            q_proj,
            k_proj,
            kv_proj_dgrad_with_res,
            v_proj,
            rope,
            // qk_norm,
            attention,
            o_proj,
            pre_ffn_norm,
            ffn,
            gradient_accumulation,
        })
    }
}

#[derive(Debug)]
pub struct DecoderWeights {
    pub pre_qkv_norm_gamma: Parameter,
    pub q_proj: Parameter,
    pub k_proj: Parameter,
    pub v_proj: Parameter,
    // pub q_norm_gamma: Parameter,
    // pub k_norm_gamma: Parameter,
    pub o_proj: Parameter,
    pub pre_ffn_norm_gamma: Parameter,
    pub ffn_up: Parameter,
    pub ffn_gate: Parameter,
    pub ffn_down: Parameter,
}

impl DecoderWeights {
    pub fn new_random(
        config: &Sky1Config,
        rng: &mut impl Rng,
        stream: &Stream,
        cx: &Context,
    ) -> anyhow::Result<Self> {
        let ffn_up_gate_stdev = 1.0 / (config.blocks.hidden_size as f32).sqrt();
        let o_proj_stdev = 1.0
            / (config.blocks.hidden_size as f32 * 2.0 * config.num_decoder_blocks as f32).sqrt();
        let ffn_down_stdev = 1.0
            / (config.blocks.ffn_intermediate_size as f32 * 2.0 * config.num_decoder_blocks as f32)
                .sqrt();

        Ok(Self {
            pre_qkv_norm_gamma: Parameter::constant([config.blocks.hidden_size], 1.0, stream, cx)?,
            q_proj: Parameter::from_normal_distr(
                [
                    config.blocks.hidden_size,
                    config.blocks.head_dim * config.blocks.num_decoder_q_heads,
                ],
                0.0,
                ffn_up_gate_stdev,
                rng,
                stream,
                cx,
            )?,
            k_proj: Parameter::from_normal_distr(
                [
                    config.blocks.hidden_size,
                    config.blocks.head_dim * config.blocks.num_decoder_kv_heads,
                ],
                0.0,
                ffn_up_gate_stdev,
                rng,
                stream,
                cx,
            )?,
            v_proj: Parameter::from_normal_distr(
                [
                    config.blocks.hidden_size,
                    config.blocks.head_dim * config.blocks.num_decoder_kv_heads,
                ],
                0.0,
                ffn_up_gate_stdev,
                rng,
                stream,
                cx,
            )?,
            // q_norm_gamma: Parameter::constant([config.blocks.head_dim], 1.0, stream, cx)?,
            // k_norm_gamma: Parameter::constant([config.blocks.head_dim], 1.0, stream, cx)?,
            o_proj: Parameter::from_normal_distr(
                [
                    config.blocks.head_dim * config.blocks.num_decoder_q_heads,
                    config.blocks.hidden_size,
                ],
                0.0,
                o_proj_stdev,
                rng,
                stream,
                cx,
            )?,
            pre_ffn_norm_gamma: Parameter::constant([config.blocks.hidden_size], 1.0, stream, cx)?,
            ffn_up: Parameter::from_normal_distr(
                [
                    config.blocks.hidden_size,
                    config.blocks.ffn_intermediate_size,
                ],
                0.0,
                ffn_up_gate_stdev,
                rng,
                stream,
                cx,
            )?,
            ffn_gate: Parameter::from_normal_distr(
                [
                    config.blocks.hidden_size,
                    config.blocks.ffn_intermediate_size,
                ],
                0.0,
                ffn_up_gate_stdev,
                rng,
                stream,
                cx,
            )?,
            ffn_down: Parameter::from_normal_distr(
                [
                    config.blocks.ffn_intermediate_size,
                    config.blocks.hidden_size,
                ],
                0.0,
                ffn_down_stdev,
                rng,
                stream,
                cx,
            )?,
        })
    }
}

/// Saved tensors for the backward pass.
#[derive(Debug)]
pub struct DecoderSavedTensors {
    pre_qkv_norm_r_inv: Tensor,
    pre_qkv_norm_result: Tensor,
    q_proj_result: Tensor,
    k_proj_result: Tensor,
    v_proj_result: Tensor,
    // q_norm_result: Tensor,
    // q_norm_r_inv: Tensor,
    // k_norm_result: Tensor,
    // k_norm_r_inv: Tensor,
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
            [batch_size, self.config.hidden_size],
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
        let mut q_proj_result = Tensor::new_zeroed(
            [
                batch_size,
                self.config.head_dim * self.config.num_decoder_q_heads,
            ],
            ElementType::Float(FloatType::Bf16),
            stream,
            &self.cx,
        )?;
        self.q_proj.forward(
            &pre_qkv_norm_result.view(),
            &weights.q_proj.bf16_weights().view(),
            &mut q_proj_result.view_mut(),
            stream,
        )?;

        let mut k_proj_result = Tensor::new_zeroed(
            [
                batch_size,
                self.config.head_dim * self.config.num_decoder_kv_heads,
            ],
            ElementType::Float(FloatType::Bf16),
            stream,
            &self.cx,
        )?;
        self.k_proj.forward(
            &pre_qkv_norm_result.view(),
            &weights.k_proj.bf16_weights().view(),
            &mut k_proj_result.view_mut(),
            stream,
        )?;

        let mut v_proj_result = Tensor::new_zeroed(
            [
                batch_size,
                self.config.head_dim * self.config.num_decoder_kv_heads,
            ],
            ElementType::Float(FloatType::Bf16),
            stream,
            &self.cx,
        )?;
        self.v_proj.forward(
            &pre_qkv_norm_result.view(),
            &weights.v_proj.bf16_weights().view(),
            &mut v_proj_result.view_mut(),
            stream,
        )?;

        // --- Q norm --- //
        // let q =
        //     qkv_proj_result.sliced(1, 0..self.config.head_dim * self.config.num_decoder_q_heads);
        // let mut q_norm_result = Tensor::new_zeroed(q.shape(), q.element_type(), stream, &self.cx)?;
        // let mut q_norm_r_inv = Tensor::new_zeroed(
        //     [batch_size * self.config.num_decoder_q_heads],
        //     ElementType::Float(FloatType::F32),
        //     stream,
        //     &self.cx,
        // )?;
        // self.qk_norm.rmsnorm_forward(
        //     &q.reshaped([
        //         batch_size * self.config.num_decoder_q_heads,
        //         self.config.head_dim,
        //     ]),
        //     &weights.q_norm_gamma.bf16_weights().view(),
        //     &mut q_norm_result.reshaped_mut([
        //         batch_size * self.config.num_decoder_q_heads,
        //         self.config.head_dim,
        //     ]),
        //     &mut q_norm_r_inv.view_mut(),
        //     stream,
        // )?;

        // // --- K norm --- //
        // let k = qkv_proj_result.sliced(
        //     1,
        //     self.config.head_dim * self.config.num_decoder_q_heads
        //         ..self.config.head_dim
        //             * (self.config.num_decoder_q_heads + self.config.num_decoder_kv_heads),
        // );
        // let mut k_norm_result = Tensor::new_zeroed(k.shape(), k.element_type(), stream, &self.cx)?;
        // let mut k_norm_r_inv = Tensor::new_zeroed(
        //     [batch_size * self.config.num_decoder_kv_heads],
        //     ElementType::Float(FloatType::F32),
        //     stream,
        //     &self.cx,
        // )?;
        // self.qk_norm.rmsnorm_forward(
        //     &k.reshaped([
        //         batch_size * self.config.num_decoder_kv_heads,
        //         self.config.head_dim,
        //     ]),
        //     &weights.k_norm_gamma.bf16_weights().view(),
        //     &mut k_norm_result.reshaped_mut([
        //         batch_size * self.config.num_decoder_kv_heads,
        //         self.config.head_dim,
        //     ]),
        //     &mut k_norm_r_inv.view_mut(),
        //     stream,
        // )?;

        // --- Attention --- //
        // let q = q_norm_result.view();
        // let k = k_norm_result.view();
        // let v = qkv_proj_result.sliced(
        //     1,
        //     self.config.head_dim
        //         * (self.config.num_decoder_q_heads + self.config.num_decoder_kv_heads)..,
        // );
        let mut attn_result = Tensor::new_zeroed(
            q_proj_result.shape(),
            q_proj_result.element_type(),
            stream,
            &self.cx,
        )?;
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
            &q_proj_result.reshaped([
                batch_size,
                self.config.num_decoder_q_heads,
                self.config.head_dim,
            ]),
            &k_proj_result.reshaped([
                batch_size,
                self.config.num_decoder_kv_heads,
                self.config.head_dim,
            ]),
            &v_proj_result.reshaped([
                batch_size,
                self.config.num_decoder_kv_heads,
                self.config.head_dim,
            ]),
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
            &mut attn_result.reshaped_mut([
                batch_size,
                self.config.num_decoder_q_heads,
                self.config.head_dim,
            ]),
            &mut attn_stats.view_mut(),
            stream,
        )?;

        // --- O-proj --- //
        let mut o_proj_result = Tensor::new_zeroed(x.shape(), x.element_type(), stream, &self.cx)?;
        self.o_proj.forward_with_residual(
            &attn_result.reshaped([
                batch_size,
                self.config.head_dim * self.config.num_decoder_q_heads,
            ]),
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
            [batch_size, self.config.hidden_size],
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
            q_proj_result,
            k_proj_result,
            v_proj_result,
            // q_norm_result,
            // q_norm_r_inv,
            // k_norm_result,
            // k_norm_r_inv,
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

    pub fn train_backward(
        &self,
        dy: &TensorView,
        x: &TensorView,
        weights: &mut DecoderWeights,
        saves: &DecoderSavedTensors,
        meta_tensors: &MetaTensors,
        stream: &Stream,
    ) -> anyhow::Result<Tensor> {
        let &[batch_size, _] = dy.shape().dims_slice() else {
            panic!()
        };

        // --- FFN down --- //

        let mut ffn_down_grad = Tensor::new_zeroed(
            weights.ffn_down.shape(),
            ElementType::Float(FloatType::Bf16),
            stream,
            &self.cx,
        )?;
        self.ffn.backward_down_proj_wgrad(
            &saves.ffn_gate_activation_result.view(),
            dy,
            &mut ffn_down_grad.view_mut(),
            stream,
        )?;
        self.gradient_accumulation.add_contribution_to_gradient(
            &ffn_down_grad.view(),
            &mut weights.ffn_down.gradients.view_mut(),
            stream,
        )?;

        let mut dgate = Tensor::new_zeroed(
            [batch_size, self.config.ffn_intermediate_size],
            ElementType::Float(FloatType::Bf16),
            stream,
            &self.cx,
        )?;
        let mut dup = Tensor::new_zeroed(
            [batch_size, self.config.ffn_intermediate_size],
            ElementType::Float(FloatType::Bf16),
            stream,
            &self.cx,
        )?;
        self.ffn.backward_down_proj_dgrad(
            &weights.ffn_down.bf16_weights().view(),
            dy,
            &saves.ffn_gate_result.view(),
            &saves.ffn_up_result.view(),
            &mut dgate.view_mut(),
            &mut dup.view_mut(),
            stream,
        )?;

        // --- FFN up ---
        let mut ffn_up_gradient = Tensor::new_zeroed(
            weights.ffn_up.shape(),
            ElementType::Float(FloatType::Bf16),
            stream,
            &self.cx,
        )?;
        self.ffn.backward_up_gate_proj_wgrad(
            &saves.o_proj_result.view(),
            &dup.view(),
            &mut ffn_up_gradient.view_mut(),
            stream,
        )?;
        self.gradient_accumulation.add_contribution_to_gradient(
            &ffn_up_gradient.view(),
            &mut weights.ffn_up.gradients.view_mut(),
            stream,
        )?;

        let mut dup_in = Tensor::new_zeroed(
            [batch_size, self.config.hidden_size],
            ElementType::Float(FloatType::Bf16),
            stream,
            &self.cx,
        )?;
        self.ffn.backward_up_proj_dgrad(
            &dup.view(),
            &weights.ffn_up.bf16_weights().view(),
            &mut dup_in.view_mut(),
            stream,
        )?;

        // --- FFN gate ---
        let mut ffn_gate_gradient = Tensor::new_zeroed(
            weights.ffn_gate.shape(),
            ElementType::Float(FloatType::Bf16),
            stream,
            &self.cx,
        )?;
        self.ffn.backward_up_gate_proj_wgrad(
            &saves.o_proj_result.view(),
            &dgate.view(),
            &mut ffn_gate_gradient.view_mut(),
            stream,
        )?;
        self.gradient_accumulation.add_contribution_to_gradient(
            &ffn_gate_gradient.view(),
            &mut weights.ffn_gate.gradients.view_mut(),
            stream,
        )?;

        let mut ffn_in_gradient = Tensor::new_zeroed(
            saves.pre_ffn_norm_result.shape(),
            ElementType::Float(FloatType::Bf16),
            stream,
            &self.cx,
        )?;
        self.ffn.backward_gate_proj_dgrad(
            &dgate.view(),
            &weights.ffn_gate.bf16_weights().view(),
            &dup_in.view(),
            &mut ffn_in_gradient.view_mut(),
            stream,
        )?;

        // --- FFN pre-norm --- //
        let mut d_ffn_norm = Tensor::new_zeroed(
            ffn_in_gradient.shape(),
            ffn_in_gradient.element_type(),
            stream,
            &self.cx,
        )?;
        self.pre_ffn_norm.rmsnorm_backward(
            &saves.o_proj_result.view(),
            &ffn_in_gradient.view(),
            Some(dy),
            &weights.pre_ffn_norm_gamma.bf16_weights_replica.view(),
            &saves.pre_ffn_norm_r_inv.view(),
            &mut d_ffn_norm.view_mut(),
            &mut weights.pre_ffn_norm_gamma.gradients.view_mut(),
            stream,
        )?;

        // --- O-proj --- //
        let mut o_proj_wgrad = Tensor::new_zeroed(
            weights.o_proj.bf16_weights().shape(),
            ElementType::Float(FloatType::Bf16),
            stream,
            &self.cx,
        )?;
        self.o_proj.backward_wgrad(
            &saves.attn_result.view(),
            &d_ffn_norm.view(),
            &mut o_proj_wgrad.view_mut(),
            stream,
        )?;
        self.gradient_accumulation.add_contribution_to_gradient(
            &o_proj_wgrad.view(),
            &mut weights.o_proj.gradients.view_mut(),
            stream,
        )?;

        let mut d_o_proj = Tensor::new_zeroed(
            saves.attn_result.shape(),
            saves.attn_result.element_type(),
            stream,
            &self.cx,
        )?;
        self.o_proj.backward_dgrad(
            &d_ffn_norm.view(),
            &weights.o_proj.bf16_weights().view(),
            &mut d_o_proj.view_mut(),
            stream,
        )?;

        // --- Attention --- //
        let mut dq = Tensor::new_zeroed(
            saves.q_proj_result.shape(),
            ElementType::Float(FloatType::Bf16),
            stream,
            &self.cx,
        )?;
        let mut dk = Tensor::new_zeroed(
            saves.k_proj_result.shape(),
            ElementType::Float(FloatType::Bf16),
            stream,
            &self.cx,
        )?;
        let mut dv = Tensor::new_zeroed(
            saves.v_proj_result.shape(),
            ElementType::Float(FloatType::Bf16),
            stream,
            &self.cx,
        )?;
        self.attention.backward_attn(
            &saves.q_proj_result.reshaped([
                batch_size,
                self.config.num_decoder_q_heads,
                self.config.head_dim,
            ]),
            &saves.k_proj_result.reshaped([
                batch_size,
                self.config.num_decoder_kv_heads,
                self.config.head_dim,
            ]),
            &saves.v_proj_result.reshaped([
                batch_size,
                self.config.num_decoder_kv_heads,
                self.config.head_dim,
            ]),
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
            &saves.attn_result.reshaped([
                batch_size,
                self.config.num_decoder_q_heads,
                self.config.head_dim,
            ]),
            &d_o_proj.reshaped([
                batch_size,
                self.config.num_decoder_q_heads,
                self.config.head_dim,
            ]),
            &saves.attn_stats.view(),
            &mut dq.reshaped_mut([
                batch_size,
                self.config.num_decoder_q_heads,
                self.config.head_dim,
            ]),
            &mut dk.reshaped_mut([
                batch_size,
                self.config.num_decoder_kv_heads,
                self.config.head_dim,
            ]),
            &mut dv.reshaped_mut([
                batch_size,
                self.config.num_decoder_kv_heads,
                self.config.head_dim,
            ]),
            stream,
        )?;

        // --- QKV-proj --- //
        let mut dwq = Tensor::new_zeroed(
            weights.q_proj.shape(),
            ElementType::Float(FloatType::Bf16),
            stream,
            &self.cx,
        )?;
        self.q_proj.backward_wgrad(
            &saves.pre_qkv_norm_result.view(),
            &dq.view(),
            &mut dwq.view_mut(),
            stream,
        )?;
        self.gradient_accumulation.add_contribution_to_gradient(
            &dwq.view(),
            &mut weights.q_proj.gradients.view_mut(),
            stream,
        )?;
        let mut q_ddata = Tensor::new_zeroed(
            saves.pre_qkv_norm_result.shape(),
            ElementType::Float(FloatType::Bf16),
            stream,
            &self.cx,
        )?;
        self.q_proj.backward_dgrad(
            &dq.view(),
            &weights.q_proj.bf16_weights().view(),
            &mut q_ddata.view_mut(),
            stream,
        )?;

        let mut dwk = Tensor::new_zeroed(
            weights.k_proj.shape(),
            ElementType::Float(FloatType::Bf16),
            stream,
            &self.cx,
        )?;
        self.k_proj.backward_wgrad(
            &saves.pre_qkv_norm_result.view(),
            &dk.view(),
            &mut dwk.view_mut(),
            stream,
        )?;
        self.gradient_accumulation.add_contribution_to_gradient(
            &dwk.view(),
            &mut weights.k_proj.gradients.view_mut(),
            stream,
        )?;
        let mut k_ddata = Tensor::new_zeroed(
            saves.pre_qkv_norm_result.shape(),
            ElementType::Float(FloatType::Bf16),
            stream,
            &self.cx,
        )?;
        self.kv_proj_dgrad_with_res.matmul(
            &dq.view(),
            &weights.q_proj.bf16_weights().view().transposed(0, 1),
            Some(&q_ddata.view()),
            &mut k_ddata.view_mut(),
            stream,
        )?;

        let mut dwv = Tensor::new_zeroed(
            weights.v_proj.shape(),
            ElementType::Float(FloatType::Bf16),
            stream,
            &self.cx,
        )?;
        self.v_proj.backward_wgrad(
            &saves.pre_qkv_norm_result.view(),
            &dv.view(),
            &mut dwv.view_mut(),
            stream,
        )?;
        self.gradient_accumulation.add_contribution_to_gradient(
            &dwv.view(),
            &mut weights.v_proj.gradients.view_mut(),
            stream,
        )?;
        let mut qkv_ddata = Tensor::new_zeroed(
            saves.pre_qkv_norm_result.shape(),
            ElementType::Float(FloatType::Bf16),
            stream,
            &self.cx,
        )?;
        self.kv_proj_dgrad_with_res.matmul(
            &dv.view(),
            &weights.v_proj.bf16_weights().view().transposed(0, 1),
            Some(&k_ddata.view()),
            &mut qkv_ddata.view_mut(),
            stream,
        )?;

        // --- Pre-QKV norm --- //

        let mut pre_qkv_ddata = Tensor::new_zeroed(
            x.shape(),
            ElementType::Float(FloatType::Bf16),
            stream,
            &self.cx,
        )?;
        self.pre_qkv_norm.rmsnorm_backward(
            x,
            &qkv_ddata.view(),
            Some(&d_ffn_norm.view()),
            &weights.pre_qkv_norm_gamma.bf16_weights_replica.view(),
            &saves.pre_qkv_norm_r_inv.view(),
            &mut pre_qkv_ddata.view_mut(),
            &mut weights.pre_qkv_norm_gamma.gradients.view_mut(),
            stream,
        )?;

        Ok(pre_qkv_ddata)
    }
}
