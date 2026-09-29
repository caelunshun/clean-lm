use std::sync::Arc;

use cleandl::{
    context::{Context, ResourceKey},
    operators::{
        adamw::AdamWExecutor,
        attention::{AttentionBackwardOutput, AttentionExecutor, AttentionForwardOutput},
        gradient_accumulation::GradientAccumulationExecutor,
        linear::{LinearConfig, LinearExecutor},
        norm::{NormExecutor, RmsNormForwardOutput},
        rope::{RopeConfig, RopeExecutor, RopeMode},
        swiglu_ffn::{DownProjDgradOutput, GateProjForwardOutput, SwiGluConfig, SwiGluFfnExecutor},
    },
    tensor::{Tensor, TensorView},
};
use rand::Rng;
use serde::Serialize;
use skycuda::stream::Stream;

use crate::{
    model::{BlocksConfig, Sky1Config},
    training::{weight_update_params::WeightUpdateParams, model::MetaTensors, parameter::Parameter},
};

#[derive(Debug)]
pub struct DecoderExecutor {
    config: BlocksConfig,
    pre_qkv_norm: Arc<NormExecutor>,
    qkv_proj: Arc<LinearExecutor>,
    rope: Arc<RopeExecutor>,
    // qk_norm: Arc<NormExecutor>,
    attention: Arc<AttentionExecutor>,
    o_proj: Arc<LinearExecutor>,
    pre_ffn_norm: Arc<NormExecutor>,
    ffn: Arc<SwiGluFfnExecutor>,
    gradient_accumulation: Arc<GradientAccumulationExecutor>,
    optimizer: Arc<AdamWExecutor>,
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
        let qkv_proj = LinearExecutor::get(
            LinearConfig {
                estimated_batch_size: self.config.max_batch_decoder_tokens,
                in_features: self.config.hidden_size,
                out_features: self.config.qkv_features(),
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

        let attention = AttentionExecutor::get(self.config.decoder_attention_config(), cx)?;
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
        let optimizer = AdamWExecutor::get(cx)?;

        Ok(DecoderExecutor {
            config: self.config.clone(),
            pre_qkv_norm,
            qkv_proj,
            rope,
            // qk_norm,
            attention,
            o_proj,
            pre_ffn_norm,
            ffn,
            gradient_accumulation,
            optimizer,
        })
    }
}

#[derive(Debug)]
pub struct DecoderWeights {
    pub pre_qkv_norm_gamma: Parameter,
    pub qkv_proj: Parameter,
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
            qkv_proj: Parameter::from_normal_distr(
                [config.blocks.hidden_size, config.blocks.qkv_features()],
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
    qkv_proj_result: Tensor,
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
        let RmsNormForwardOutput {
            y: pre_qkv_norm_result,
            r_inv: pre_qkv_norm_r_inv,
        } = self.pre_qkv_norm.rmsnorm_forward(
            x,
            &weights.pre_qkv_norm_gamma.bf16_weights().view(),
            stream,
        )?;

        // --- QKV projection --- //
        let mut qkv_proj_result = self.qkv_proj.forward(
            &pre_qkv_norm_result.view(),
            &weights.qkv_proj.bf16_weights().view(),
            stream,
        )?;

        let mut qkv_reshaped = qkv_proj_result.reshaped_mut([
            batch_size,
            self.config.num_decoder_q_heads + 2 * self.config.num_decoder_kv_heads,
            self.config.head_dim,
        ]);
        let mut qk = qkv_reshaped.sliced_mut(
            1,
            0..(self.config.num_decoder_q_heads + self.config.num_decoder_kv_heads),
        );
        self.rope.execute(
            &mut qk,
            &meta_tensors.decoder_ragged_seqlen.cu_seqlen(),
            &meta_tensors.seqlen,
            RopeMode::Forward,
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
        let AttentionForwardOutput {
            out: attn_result,
            stats: attn_stats,
        } = self.attention.forward_attn(
            &qkv_proj_result.reshaped([
                batch_size,
                self.config.num_decoder_q_heads + 2 * self.config.num_decoder_kv_heads,
                self.config.head_dim,
            ]),
            &meta_tensors.decoder_ragged_seqlen,
            stream,
        )?;

        // --- O-proj --- //
        let o_proj_result = self.o_proj.forward_with_residual(
            &attn_result.reshaped([
                batch_size,
                self.config.head_dim * self.config.num_decoder_q_heads,
            ]),
            &weights.o_proj.bf16_weights().view(),
            x,
            stream,
        )?;

        // --- FFN pre-norm --- //
        let RmsNormForwardOutput {
            y: pre_ffn_norm_result,
            r_inv: pre_ffn_norm_r_inv,
        } = self.pre_ffn_norm.rmsnorm_forward(
            &o_proj_result.view(),
            &weights.pre_ffn_norm_gamma.bf16_weights().view(),
            stream,
        )?;

        // --- FFN --- //
        let ffn_up_result = self.ffn.forward_up_proj(
            &pre_ffn_norm_result.view(),
            &weights.ffn_up.bf16_weights().view(),
            stream,
        )?;

        let GateProjForwardOutput {
            gate: ffn_gate_result,
            activation: ffn_gate_activation_result,
        } = self.ffn.forward_gate_proj(
            &pre_ffn_norm_result.view(),
            &ffn_up_result.view(),
            &weights.ffn_gate.bf16_weights().view(),
            stream,
        )?;

        let ffn_down_result = self.ffn.forward_down_proj(
            &ffn_gate_activation_result.view(),
            &weights.ffn_down.bf16_weights().view(),
            &o_proj_result.view(),
            stream,
        )?;

        Ok(DecoderSavedTensors {
            pre_qkv_norm_r_inv,
            pre_qkv_norm_result,
            qkv_proj_result,
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
        let q_features = self.config.head_dim * self.config.num_decoder_q_heads;

        // --- FFN down --- //

        let ffn_down_dw = self.ffn.backward_down_proj_wgrad(
            &saves.ffn_gate_activation_result.view(),
            dy,
            stream,
        )?;
        self.gradient_accumulation.add_contribution_to_gradient(
            &ffn_down_dw.view(),
            &mut weights.ffn_down.gradients.view_mut(),
            stream,
        )?;

        let DownProjDgradOutput {
            dgate: ffn_gate_dy,
            dup: ffn_up_dy,
        } = self.ffn.backward_down_proj_dgrad(
            &weights.ffn_down.bf16_weights().view(),
            dy,
            &saves.ffn_gate_result.view(),
            &saves.ffn_up_result.view(),
            stream,
        )?;

        // --- FFN up ---
        let ffn_up_dw = self.ffn.backward_up_gate_proj_wgrad(
            &saves.pre_ffn_norm_result.view(),
            &ffn_up_dy.view(),
            stream,
        )?;
        self.gradient_accumulation.add_contribution_to_gradient(
            &ffn_up_dw.view(),
            &mut weights.ffn_up.gradients.view_mut(),
            stream,
        )?;

        let ffn_up_dx = self.ffn.backward_up_proj_dgrad(
            &ffn_up_dy.view(),
            &weights.ffn_up.bf16_weights().view(),
            stream,
        )?;

        // --- FFN gate ---
        let ffn_gate_dw = self.ffn.backward_up_gate_proj_wgrad(
            &saves.pre_ffn_norm_result.view(),
            &ffn_gate_dy.view(),
            stream,
        )?;
        self.gradient_accumulation.add_contribution_to_gradient(
            &ffn_gate_dw.view(),
            &mut weights.ffn_gate.gradients.view_mut(),
            stream,
        )?;

        let ffn_dx = self.ffn.backward_gate_proj_dgrad(
            &ffn_gate_dy.view(),
            &weights.ffn_gate.bf16_weights().view(),
            &ffn_up_dx.view(),
            stream,
        )?;

        // --- FFN pre-norm --- //
        let pre_ffn_norm_dx = self.pre_ffn_norm.rmsnorm_backward(
            &saves.o_proj_result.view(),
            &ffn_dx.view(),
            Some(dy),
            &weights.pre_ffn_norm_gamma.bf16_weights_replica.view(),
            &saves.pre_ffn_norm_r_inv.view(),
            &mut weights.pre_ffn_norm_gamma.gradients.view_mut(),
            stream,
        )?;

        // --- O-proj --- //
        let attn_result = saves.attn_result.reshaped([batch_size, q_features]);
        let o_proj_dw =
            self.o_proj
                .backward_wgrad(&attn_result, &pre_ffn_norm_dx.view(), stream)?;
        self.gradient_accumulation.add_contribution_to_gradient(
            &o_proj_dw.view(),
            &mut weights.o_proj.gradients.view_mut(),
            stream,
        )?;

        let o_proj_dx = self.o_proj.backward_dgrad(
            &pre_ffn_norm_dx.view(),
            &weights.o_proj.bf16_weights().view(),
            stream,
        )?;

        // --- Attention --- //
        let AttentionBackwardOutput { mut dqkv } = self.attention.backward_attn(
            &saves.qkv_proj_result.reshaped([
                batch_size,
                self.config.num_decoder_q_heads + 2 * self.config.num_decoder_kv_heads,
                self.config.head_dim,
            ]),
            &meta_tensors.decoder_ragged_seqlen,
            &saves.attn_result.view(),
            &o_proj_dx.reshaped([
                batch_size,
                self.config.num_decoder_q_heads,
                self.config.head_dim,
            ]),
            &saves.attn_stats.view(),
            stream,
        )?;

        let mut dqk = dqkv.sliced_mut(
            1,
            0..(self.config.num_decoder_q_heads + self.config.num_decoder_kv_heads),
        );
        self.rope.execute(
            &mut dqk,
            &meta_tensors.decoder_ragged_seqlen.cu_seqlen(),
            &meta_tensors.seqlen,
            RopeMode::Backward,
            stream,
        )?;

        let dqkv = dqkv.reshaped([batch_size, self.config.qkv_features()]);

        // --- QKV-proj --- //
        let dw_qkv =
            self.qkv_proj
                .backward_wgrad(&saves.pre_qkv_norm_result.view(), &dqkv, stream)?;
        self.gradient_accumulation.add_contribution_to_gradient(
            &dw_qkv.view(),
            &mut weights.qkv_proj.gradients.view_mut(),
            stream,
        )?;
        let qkv_ddata =
            self.qkv_proj
                .backward_dgrad(&dqkv, &weights.qkv_proj.bf16_weights().view(), stream)?;

        // --- Pre-QKV norm --- //

        let pre_qkv_ddata = self.pre_qkv_norm.rmsnorm_backward(
            x,
            &qkv_ddata.view(),
            Some(&pre_ffn_norm_dx.view()),
            &weights.pre_qkv_norm_gamma.bf16_weights_replica.view(),
            &saves.pre_qkv_norm_r_inv.view(),
            &mut weights.pre_qkv_norm_gamma.gradients.view_mut(),
            stream,
        )?;

        Ok(pre_qkv_ddata)
    }

    pub fn visit_gradients(
        &self,
        weights: &DecoderWeights,
        mut callback: impl FnMut(&TensorView) -> anyhow::Result<()>,
    ) -> anyhow::Result<()> {
        for param in [
            &weights.pre_qkv_norm_gamma,
            &weights.qkv_proj,
            &weights.o_proj,
            &weights.pre_ffn_norm_gamma,
            &weights.ffn_up,
            &weights.ffn_gate,
            &weights.ffn_down,
        ] {
            callback(&param.gradients.view())?;
        }
        Ok(())
    }

    pub fn weight_update(
        &self,
        weights: &mut DecoderWeights,
        gradients_sum2: &TensorView,
        config: &WeightUpdateParams,
        train_step: u32,
        stream: &Stream,
    ) -> anyhow::Result<()> {
        for param in [
            &mut weights.pre_qkv_norm_gamma,
            &mut weights.qkv_proj,
            &mut weights.o_proj,
            &mut weights.pre_ffn_norm_gamma,
            &mut weights.ffn_up,
            &mut weights.ffn_gate,
            &mut weights.ffn_down,
        ] {
            self.optimizer.adamw_weight_update(
                &param.gradients.view(),
                gradients_sum2,
                &mut param.master_weights.view_mut(),
                &mut param.bf16_weights_replica.view_mut(),
                &mut param.adam_m.view_mut(),
                &mut param.adam_v.view_mut(),
                config.adam_beta1,
                config.adam_beta2,
                config.weight_decay,
                config.max_gradient_norm,
                config.lr,
                train_step,
                stream,
            )?;
            stream.fill(&mut param.gradients.storage_mut(), 0u8)?;
        }

        Ok(())
    }
}
