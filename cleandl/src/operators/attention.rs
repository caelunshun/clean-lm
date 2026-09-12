use std::sync::Arc;

use skycuda::stream::Stream;
use skycuda_cudnn::{SdpaAttributes, SdpaBackwardAttributes, TensorId};
use skytens_core::data_type::{ElementType, FloatType, SIntType};

use crate::{
    context::{Context, ResourceKey},
    cudnn::{CudnnGraphBuilder, ExecutableCudnnGraph},
    tensor::{TensorView, TensorViewMut, VirtTensorView},
};

/// Various cumulative sequence length tensors
/// with different scales needed for cuDNN attention.
/// All tensors are expected as sint64 `[B + 1]`.
#[derive(Debug)]
pub struct CuSeqlenTensors<'a> {
    /// Output cu_seqlen scaled by head_dim * num_q_heads.
    pub cu_seqlen_q_times_hd_times_qh: TensorView<'a>,
    /// Input cu_seqlen scaled by head_dim * num_kv_heads.
    pub cu_seqlen_kv_times_hd_times_kvh: TensorView<'a>,
}

/// Attention with varlen and ragged offsets.
///
/// Due to apparent cuDNN API inconsistencies, the stats tensor
/// is not ragged like the others. It must be padded to shape [max_num_sequences, max_output_seqlen, num_q_heads].
#[derive(Debug)]
pub struct AttentionExecutor {
    config: AttentionConfig,
    forward_graph: ExecutableCudnnGraph,
    forward_input_cu_seqlen_q_times_hd_times_qh: TensorId,
    forward_input_cu_seqlen_kv_times_hd_times_kvh: TensorId,
    forward_input_seqlen_q: TensorId,
    forward_input_seqlen_kv: TensorId,
    forward_input_q: TensorId,
    forward_input_k: TensorId,
    forward_input_v: TensorId,
    forward_output_o: TensorId,
    forward_output_stats: TensorId,
    backward_graph: ExecutableCudnnGraph,
    backward_input_q: TensorId,
    backward_input_k: TensorId,
    backward_input_v: TensorId,
    backward_input_cu_seqlen_q_times_hd_times_qh: TensorId,
    backward_input_cu_seqlen_kv_times_hd_times_kvh: TensorId,
    backward_input_seqlen_q: TensorId,
    backward_input_seqlen_kv: TensorId,
    backward_input_o: TensorId,
    backward_input_do: TensorId,
    backward_input_stats: TensorId,
    backward_output_dq: TensorId,
    backward_output_dk: TensorId,
    backward_output_dv: TensorId,
}

impl AttentionExecutor {
    pub fn get(config: AttentionConfig, cx: &Context) -> anyhow::Result<Arc<Self>> {
        cx.get_or_create_resource(config)
    }

    /// Q, K, V, O expected as row-major bf16 [T, H, D].
    /// Stats expected as row-major f32 [T, H].
    /// `seqlen_*` expected as s32 [B].
    #[allow(clippy::too_many_arguments)]
    pub fn forward_attn(
        &self,
        q: &TensorView<'_>,
        k: &TensorView<'_>,
        v: &TensorView<'_>,
        cu_seqlens: &CuSeqlenTensors<'_>,
        seqlen_q: &TensorView<'_>,
        seqlen_kv: &TensorView<'_>,
        out: &mut TensorViewMut<'_>,
        out_stats: &mut TensorViewMut<'_>,
        stream: &Stream,
    ) -> anyhow::Result<()> {
        let b = self.config.max_num_sequences;
        self.forward_graph.execute(
            &[
                (self.forward_input_q, q),
                (self.forward_input_k, k),
                (self.forward_input_v, v),
                (
                    self.forward_input_cu_seqlen_q_times_hd_times_qh,
                    &cu_seqlens
                        .cu_seqlen_q_times_hd_times_qh
                        .reshaped([b + 1, 1, 1, 1]),
                ),
                (
                    self.forward_input_cu_seqlen_kv_times_hd_times_kvh,
                    &cu_seqlens
                        .cu_seqlen_kv_times_hd_times_kvh
                        .reshaped([b + 1, 1, 1, 1]),
                ),
                (
                    self.forward_input_seqlen_q,
                    &seqlen_q.reshaped([b, 1, 1, 1]),
                ),
                (
                    self.forward_input_seqlen_kv,
                    &seqlen_kv.reshaped([b, 1, 1, 1]),
                ),
            ],
            &mut [
                (self.forward_output_o, out.view_mut()),
                (
                    self.forward_output_stats,
                    out_stats
                        .reshaped_mut([
                            self.config.max_num_sequences,
                            self.config.max_q_seqlen,
                            self.config.q_heads,
                            1,
                        ])
                        .transposed_mut(1, 2),
                ),
            ],
            stream,
        )?;
        Ok(())
    }

    /// Q, K, V, O, dO, dQ, dK, dV expected as row-major bf16 [T, H, D].
    /// Stats expected as row-major f32 [T, H].
    /// `seqlen_*` expected as s32 [B].
    #[allow(clippy::too_many_arguments)]
    pub fn backward_attn(
        &self,
        q: &TensorView<'_>,
        k: &TensorView<'_>,
        v: &TensorView<'_>,
        cu_seqlens: &CuSeqlenTensors<'_>,
        seqlen_q: &TensorView<'_>,
        seqlen_kv: &TensorView<'_>,
        out: &TensorView<'_>,
        out_grad: &TensorView<'_>,
        stats: &TensorView<'_>,
        grad_q: &mut TensorViewMut<'_>,
        grad_k: &mut TensorViewMut<'_>,
        grad_v: &mut TensorViewMut<'_>,
        stream: &Stream,
    ) -> anyhow::Result<()> {
        let b = self.config.max_num_sequences;
        self.backward_graph.execute(
            &[
                (self.backward_input_q, q),
                (self.backward_input_k, k),
                (self.backward_input_v, v),
                (
                    self.backward_input_cu_seqlen_q_times_hd_times_qh,
                    &cu_seqlens
                        .cu_seqlen_q_times_hd_times_qh
                        .reshaped([b + 1, 1, 1, 1]),
                ),
                (
                    self.backward_input_cu_seqlen_kv_times_hd_times_kvh,
                    &cu_seqlens
                        .cu_seqlen_kv_times_hd_times_kvh
                        .reshaped([b + 1, 1, 1, 1]),
                ),
                (
                    self.backward_input_seqlen_q,
                    &seqlen_q.reshaped([b, 1, 1, 1]),
                ),
                (
                    self.backward_input_seqlen_kv,
                    &seqlen_kv.reshaped([b, 1, 1, 1]),
                ),
                (self.backward_input_o, out),
                (self.backward_input_do, out_grad),
                (
                    self.backward_input_stats,
                    &stats
                        .reshaped([
                            self.config.max_num_sequences,
                            self.config.max_q_seqlen,
                            self.config.q_heads,
                            1,
                        ])
                        .transposed(1, 2),
                ),
            ],
            &mut [
                (self.backward_output_dq, grad_q.view_mut()),
                (self.backward_output_dk, grad_k.view_mut()),
                (self.backward_output_dv, grad_v.view_mut()),
            ],
            stream,
        )?;
        Ok(())
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct AttentionConfig {
    pub is_causal: bool,
    pub max_q_seqlen: usize,
    pub max_kv_seqlen: usize,
    pub max_num_sequences: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
}

impl AttentionConfig {
    fn cu_seqlen_layout(&self) -> VirtTensorView {
        VirtTensorView::new(
            [self.max_num_sequences + 1, 1, 1, 1],
            ElementType::SInt(SIntType::S64),
        )
    }

    fn seqlen_layout(&self) -> VirtTensorView {
        VirtTensorView::new(
            [self.max_num_sequences, 1, 1, 1],
            ElementType::SInt(SIntType::S32),
        )
    }

    fn q_layout(&self) -> VirtTensorView {
        VirtTensorView::new(
            [
                self.max_num_sequences,
                self.max_q_seqlen,
                self.q_heads,
                self.head_dim,
            ],
            ElementType::Float(FloatType::Bf16),
        )
        .transposed(1, 2)
    }

    fn kv_layout(&self) -> VirtTensorView {
        VirtTensorView::new(
            [
                self.max_num_sequences,
                self.max_kv_seqlen,
                self.kv_heads,
                self.head_dim,
            ],
            ElementType::Float(FloatType::Bf16),
        )
        .transposed(1, 2)
    }

    fn stats_layout(&self) -> VirtTensorView {
        VirtTensorView::new(
            [self.max_num_sequences, self.max_q_seqlen, self.q_heads, 1],
            ElementType::Float(FloatType::F32),
        )
        .transposed(1, 2)
    }
}

impl ResourceKey for AttentionConfig {
    type Resource = AttentionExecutor;

    fn create(&self, cx: &Context) -> anyhow::Result<Self::Resource> {
        let attn_scale = (1.0 / (self.head_dim as f64).sqrt()) as f32;

        let forward_graph = CudnnGraphBuilder::new(cx)?;

        let forward_cu_seqlen_q_times_hd_times_qh =
            forward_graph.bind_tensor(&self.cu_seqlen_layout());
        let forward_cu_seqlen_kv_times_hd_times_kvh =
            forward_graph.bind_tensor(&self.cu_seqlen_layout());
        let forward_seqlen_q = forward_graph.bind_tensor(&self.seqlen_layout());
        let forward_seqlen_kv = forward_graph.bind_tensor(&self.seqlen_layout());

        let forward_input_q = forward_graph.bind_tensor(&self.q_layout());
        let forward_input_k = forward_graph.bind_tensor(&self.kv_layout());
        let forward_input_v = forward_graph.bind_tensor(&self.kv_layout());

        forward_input_q.set_ragged_offset(forward_cu_seqlen_q_times_hd_times_qh, None);
        forward_input_k.set_ragged_offset(forward_cu_seqlen_kv_times_hd_times_kvh, None);
        forward_input_v.set_ragged_offset(forward_cu_seqlen_kv_times_hd_times_kvh, None);

        let [forward_output_o, forward_output_stats] = forward_graph.graph().sdpa(
            forward_input_q,
            forward_input_k,
            forward_input_v,
            &SdpaAttributes::new()
                .attn_scale(attn_scale)
                .generate_stats(true)
                .causal_mask(self.is_causal)
                .padding_mask(true)
                .seq_len(forward_seqlen_q, forward_seqlen_kv),
        )?;
        let forward_output_o = forward_output_o.unwrap();
        let forward_output_stats = forward_output_stats.unwrap();
        let forward_output_o_id = forward_graph.mark_output(forward_output_o, &self.q_layout());
        forward_output_o.set_ragged_offset(forward_cu_seqlen_q_times_hd_times_qh, None);
        let forward_output_stats_id =
            forward_graph.mark_output(forward_output_stats, &self.stats_layout());

        let executor_forward = (
            forward_cu_seqlen_q_times_hd_times_qh.uid(),
            forward_cu_seqlen_kv_times_hd_times_kvh.uid(),
            forward_seqlen_q.uid(),
            forward_seqlen_kv.uid(),
            forward_input_q.uid(),
            forward_input_k.uid(),
            forward_input_v.uid(),
        );
        let forward_graph = forward_graph.build()?;

        let backward_graph = CudnnGraphBuilder::new(cx)?;

        let backward_cu_seqlen_q_times_hd_times_qh =
            backward_graph.bind_tensor(&self.cu_seqlen_layout());
        let backward_cu_seqlen_kv_times_hd_times_kvh =
            backward_graph.bind_tensor(&self.cu_seqlen_layout());
        let backward_seqlen_q = backward_graph.bind_tensor(&self.seqlen_layout());
        let backward_seqlen_kv = backward_graph.bind_tensor(&self.seqlen_layout());

        let backward_input_q = backward_graph.bind_tensor(&self.q_layout());
        let backward_input_k = backward_graph.bind_tensor(&self.kv_layout());
        let backward_input_v = backward_graph.bind_tensor(&self.kv_layout());
        let backward_input_o = backward_graph.bind_tensor(&self.q_layout());
        let backward_input_do = backward_graph.bind_tensor(&self.q_layout());
        let backward_input_stats = backward_graph.bind_tensor(&self.stats_layout());

        for tensor in [backward_input_q, backward_input_o, backward_input_do] {
            tensor.set_ragged_offset(backward_cu_seqlen_q_times_hd_times_qh, None);
        }
        for tensor in [backward_input_k, backward_input_v] {
            tensor.set_ragged_offset(backward_cu_seqlen_kv_times_hd_times_kvh, None);
        }

        let [backward_output_dq, backward_output_dk, backward_output_dv] =
            backward_graph.graph().sdpa_backward(
                backward_input_q,
                backward_input_k,
                backward_input_v,
                backward_input_o,
                backward_input_do,
                backward_input_stats,
                &SdpaBackwardAttributes {
                    use_padding_mask: true,
                    seq_len_q: Some(backward_seqlen_q),
                    seq_len_kv: Some(backward_seqlen_kv),
                    ..SdpaBackwardAttributes::new()
                        .attn_scale(attn_scale)
                        .causal_mask(self.is_causal)
                },
            )?;
        let backward_output_dq = backward_output_dq.unwrap();
        let backward_output_dk = backward_output_dk.unwrap();
        let backward_output_dv = backward_output_dv.unwrap();
        let backward_output_dq_id =
            backward_graph.mark_output(backward_output_dq, &self.q_layout());
        backward_output_dq.set_ragged_offset(backward_cu_seqlen_q_times_hd_times_qh, None);
        let backward_output_dk_id =
            backward_graph.mark_output(backward_output_dk, &self.kv_layout());
        backward_output_dk.set_ragged_offset(backward_cu_seqlen_kv_times_hd_times_kvh, None);
        let backward_output_dv_id =
            backward_graph.mark_output(backward_output_dv, &self.kv_layout());
        backward_output_dv.set_ragged_offset(backward_cu_seqlen_kv_times_hd_times_kvh, None);

        let executor_backward = (
            backward_cu_seqlen_q_times_hd_times_qh.uid(),
            backward_cu_seqlen_kv_times_hd_times_kvh.uid(),
            backward_seqlen_q.uid(),
            backward_seqlen_kv.uid(),
            backward_input_q.uid(),
            backward_input_k.uid(),
            backward_input_v.uid(),
            backward_input_o.uid(),
            backward_input_do.uid(),
            backward_input_stats.uid(),
        );
        let backward_graph = backward_graph.build()?;

        let (
            forward_input_cu_seqlen_q_times_hd_times_qh,
            forward_input_cu_seqlen_kv_times_hd_times_kvh,
            forward_input_seqlen_q,
            forward_input_seqlen_kv,
            forward_input_q,
            forward_input_k,
            forward_input_v,
        ) = executor_forward;
        let (
            backward_input_cu_seqlen_q_times_hd_times_qh,
            backward_input_cu_seqlen_kv_times_hd_times_kvh,
            backward_input_seqlen_q,
            backward_input_seqlen_kv,
            backward_input_q,
            backward_input_k,
            backward_input_v,
            backward_input_o,
            backward_input_do,
            backward_input_stats,
        ) = executor_backward;

        Ok(AttentionExecutor {
            config: self.clone(),
            forward_graph,
            forward_input_cu_seqlen_q_times_hd_times_qh,
            forward_input_cu_seqlen_kv_times_hd_times_kvh,
            forward_input_seqlen_q,
            forward_input_seqlen_kv,
            forward_input_q,
            forward_input_k,
            forward_input_v,
            forward_output_o: forward_output_o_id,
            forward_output_stats: forward_output_stats_id,
            backward_graph,
            backward_input_q,
            backward_input_k,
            backward_input_v,
            backward_input_cu_seqlen_q_times_hd_times_qh,
            backward_input_cu_seqlen_kv_times_hd_times_kvh,
            backward_input_seqlen_q,
            backward_input_seqlen_kv,
            backward_input_o,
            backward_input_do,
            backward_input_stats,
            backward_output_dq: backward_output_dq_id,
            backward_output_dk: backward_output_dk_id,
            backward_output_dv: backward_output_dv_id,
        })
    }
}
