use crate::{
    context::{Context, ResourceKey},
    cudnn::{CudnnGraphBuilder, ExecutableCudnnGraph},
    tensor::{Tensor, TensorView, VirtTensorView},
};
use skycuda::{device_memory_pool::DeviceMemoryPool, stream::Stream};
use skycuda_cudnn::{SdpaAttributes, SdpaBackwardAttributes, TensorId};
use skytens_core::{
    data_type::{ElementType, FloatType, SIntType},
    shape::{Shape, Strides},
};
use std::sync::Arc;

/// Sequence length and ragged offset tensors needed for attention
/// on a packed, ragged QKV tensor.
///
/// Sequences beyond those given (up to `max_num_sequences`) are padded as zero-length.
#[derive(Debug)]
pub struct RaggedSeqlenTensors {
    /// s32 `[max_num_sequences]`.
    seqlen: Tensor,
    /// s64 `[max_num_sequences + 1]`.
    cu_seqlen: Tensor,
    /// Offset of each sequence's Q-head span within the packed QKV tensor, in units of elements.
    offset_q: Tensor,
    /// Offset of each sequence's K-head span within the packed QKV tensor, in units of elements.
    offset_k: Tensor,
    /// Offset of each sequence's V-head span within the packed QKV tensor, in units of elements.
    offset_v: Tensor,
    /// Offset of each sequence within the `[T, q_heads, head_dim]` output tensor, in units of elements.
    offset_o: Tensor,
    /// Offset of each sequence within the ragged `[T, q_heads]` stats tensor, in units of elements.
    offset_stats: Tensor,
}

impl RaggedSeqlenTensors {
    pub fn build(seqlens: &[u32], cx: &Context, config: &AttentionConfig) -> anyhow::Result<Self> {
        let b = config.max_num_sequences;
        anyhow::ensure!(
            seqlens.len() <= b,
            "got {} sequences, but max_num_sequences is {b}",
            seqlens.len()
        );

        let packed_row_stride = config.packed_row_stride();
        let q_span = config.head_dim * config.q_heads;
        let kv_span = config.head_dim * config.kv_heads;

        let mut seqlen_data = vec![0i32; b];
        for (dst, &seqlen) in seqlen_data.iter_mut().zip(seqlens) {
            *dst = i32::try_from(seqlen)?;
        }

        let mut cu_seqlens = Vec::with_capacity(b + 1);
        let mut cu_seqlen = 0usize;
        for &seqlen in &seqlen_data {
            cu_seqlens.push(cu_seqlen as i64);
            cu_seqlen += seqlen as usize;
        }
        cu_seqlens.push(cu_seqlen as i64);

        let offsets = |row_stride: usize, base: usize| -> Vec<i64> {
            cu_seqlens
                .iter()
                .map(|&cu_seqlen| cu_seqlen * row_stride as i64 + base as i64)
                .collect()
        };
        let upload = |data: &[i64]| Tensor::from_slice(data, [b + 1], cx.h2d_copy_stream(), cx);

        Ok(Self {
            seqlen: Tensor::from_slice(&seqlen_data, [b], cx.h2d_copy_stream(), cx)?,
            offset_q: upload(&offsets(packed_row_stride, 0))?,
            offset_k: upload(&offsets(packed_row_stride, q_span))?,
            offset_v: upload(&offsets(packed_row_stride, q_span + kv_span))?,
            offset_o: upload(&offsets(q_span, 0))?,
            offset_stats: upload(&offsets(config.q_heads, 0))?,
            cu_seqlen: upload(&cu_seqlens)?,
        })
    }

    pub fn cu_seqlen(&self) -> TensorView<'_> {
        self.cu_seqlen.view()
    }
}

/// Attention with varlen and ragged offsets, operating on a packed QKV tensor
/// laid out as row-major bf16 `[T, q_heads + 2 * kv_heads, head_dim]`
/// (Q heads first, then K heads, then V heads).
///
/// The stats tensor is ragged only on SM90 and SM10x. On other architectures, it is padded to shape
/// `[max_num_sequences, max_q_seqlen, q_heads]`.
#[derive(Debug)]
pub struct AttentionExecutor {
    config: AttentionConfig,
    cx: Context,
    ragged_stats: bool,
    forward_graph: ExecutableCudnnGraph,
    /// Q, K, V tensor IDs, all bound to the packed QKV buffer.
    forward_input_qkv: [TensorId; 3],
    forward_input_offset_q: TensorId,
    forward_input_offset_k: TensorId,
    forward_input_offset_v: TensorId,
    forward_input_offset_o: TensorId,
    forward_input_offset_stats: Option<TensorId>,
    forward_input_seqlen_q: TensorId,
    forward_input_seqlen_kv: TensorId,
    forward_output_o: TensorId,
    forward_output_stats: TensorId,
    backward_graph: ExecutableCudnnGraph,
    backward_input_qkv: [TensorId; 3],
    backward_input_offset_q: TensorId,
    backward_input_offset_k: TensorId,
    backward_input_offset_v: TensorId,
    backward_input_offset_o: TensorId,
    backward_input_offset_stats: Option<TensorId>,
    backward_input_seqlen_q: TensorId,
    backward_input_seqlen_kv: TensorId,
    backward_input_o: TensorId,
    backward_input_do: TensorId,
    backward_input_stats: TensorId,
    /// dQ, dK, dV tensor IDs, all aliased onto the packed dQKV buffer.
    backward_output_dqkv: [TensorId; 3],
}

/// Outputs of [`AttentionExecutor::forward_attn`].
#[derive(Debug)]
pub struct AttentionForwardOutput {
    /// Attention output, row-major bf16 `[T, q_heads, head_dim]`.
    pub out: Tensor,
    /// Softmax stats, row-major f32 `[T, q_heads]` if ragged stats are supported on the device,
    /// else `[max_num_sequences, max_q_seqlen, q_heads]`.
    pub stats: Tensor,
}

/// Outputs of [`AttentionExecutor::backward_attn`].
#[derive(Debug)]
pub struct AttentionBackwardOutput {
    /// Gradient of the packed QKV tensor, with the same layout as the input.
    pub dqkv: Tensor,
}

impl AttentionExecutor {
    pub fn get(config: AttentionConfig, cx: &Context) -> anyhow::Result<Arc<Self>> {
        cx.get_or_create_resource(config)
    }

    /// Validates the packed QKV layout and returns the number of tokens `T`.
    fn check_qkv(&self, qkv: &TensorView<'_>) -> usize {
        let shape = qkv.shape();
        assert_eq!(
            shape.num_dims(),
            3,
            "packed QKV must be [T, heads, head_dim]"
        );
        assert_eq!(shape[1], self.config.q_heads + 2 * self.config.kv_heads);
        assert_eq!(shape[2], self.config.head_dim);
        assert_eq!(qkv.element_type(), ElementType::Float(FloatType::Bf16));
        assert!(qkv.is_packed_row_major());
        shape[0]
    }

    fn new_stats(
        &self,
        num_tokens: usize,
        stream: &Stream,
        pool: &DeviceMemoryPool,
    ) -> anyhow::Result<Tensor> {
        let shape = if self.ragged_stats {
            Shape::new([num_tokens, self.config.q_heads])
        } else {
            Shape::new([
                self.config.max_num_sequences,
                self.config.max_q_seqlen,
                self.config.q_heads,
            ])
        };
        Tensor::new_zeroed_with_pool(
            shape,
            ElementType::Float(FloatType::F32),
            stream,
            &self.cx,
            pool,
        )
    }

    /// Shape of stats as returned by [`Self::new_stats`], in the layout bound in the graphs.
    /// For padded stats, dims 1 and 2 must additionally be transposed.
    fn stats_graph_shape(&self, num_tokens: usize) -> Shape {
        if self.ragged_stats {
            Shape::new([num_tokens, self.config.q_heads, 1])
        } else {
            Shape::new([
                self.config.max_num_sequences,
                self.config.max_q_seqlen,
                self.config.q_heads,
                1,
            ])
        }
    }

    /// `qkv` expected as packed row-major bf16 `[T, q_heads + 2 * kv_heads, head_dim]`.
    /// Outputs are allocated from `pool`; the cuDNN workspace from `workspace_pool`.
    pub fn forward_attn_with_pool(
        &self,
        qkv: &TensorView<'_>,
        seqlens: &RaggedSeqlenTensors,
        stream: &Stream,
        workspace_pool: &DeviceMemoryPool,
        pool: &DeviceMemoryPool,
    ) -> anyhow::Result<AttentionForwardOutput> {
        let b = self.config.max_num_sequences;
        let num_tokens = self.check_qkv(qkv);
        assert!(num_tokens <= self.config.max_total_tokens);

        let mut out = Tensor::new_zeroed_with_pool(
            [num_tokens, self.config.q_heads, self.config.head_dim],
            qkv.element_type(),
            stream,
            &self.cx,
            pool,
        )?;
        let mut out_stats = self.new_stats(num_tokens, stream, pool)?;
        let stats_shape = self.stats_graph_shape(num_tokens);

        let [offset_q, offset_k, offset_v, offset_o, offset_stats] = [
            &seqlens.offset_q,
            &seqlens.offset_k,
            &seqlens.offset_v,
            &seqlens.offset_o,
            &seqlens.offset_stats,
        ]
        .map(|offset| offset.reshaped([b + 1, 1, 1, 1]));
        let seqlen = seqlens.seqlen.reshaped([b, 1, 1, 1]);
        let mut inputs = vec![
            (self.forward_input_qkv[0], qkv),
            (self.forward_input_qkv[1], qkv),
            (self.forward_input_qkv[2], qkv),
            (self.forward_input_offset_q, &offset_q),
            (self.forward_input_offset_k, &offset_k),
            (self.forward_input_offset_v, &offset_v),
            (self.forward_input_offset_o, &offset_o),
            (self.forward_input_seqlen_q, &seqlen),
            (self.forward_input_seqlen_kv, &seqlen),
        ];
        if let Some(id) = self.forward_input_offset_stats {
            inputs.push((id, &offset_stats));
        }
        let mut padded_stats;
        let stats_view = if self.ragged_stats {
            out_stats.reshaped_mut(stats_shape)
        } else {
            padded_stats = out_stats.reshaped_mut(stats_shape);
            padded_stats.transposed_mut(1, 2)
        };
        self.forward_graph.execute(
            &inputs,
            &mut [
                (self.forward_output_o, out.view_mut()),
                (self.forward_output_stats, stats_view),
            ],
            stream,
            workspace_pool,
        )?;
        Ok(AttentionForwardOutput {
            out,
            stats: out_stats,
        })
    }

    pub fn forward_attn(
        &self,
        qkv: &TensorView<'_>,
        seqlens: &RaggedSeqlenTensors,
        stream: &Stream,
        workspace_pool: &DeviceMemoryPool,
    ) -> anyhow::Result<AttentionForwardOutput> {
        self.forward_attn_with_pool(
            qkv,
            seqlens,
            stream,
            workspace_pool,
            self.cx.device_memory_pool(),
        )
    }

    /// `qkv` expected as packed row-major bf16 `[T, q_heads + 2 * kv_heads, head_dim]`.
    /// `out`, `out_grad` expected as row-major bf16 `[T, q_heads, head_dim]`.
    /// Stats expected as returned by [`Self::forward_attn`].
    /// Outputs are allocated from `pool`; the cuDNN workspace from `workspace_pool`.
    #[allow(clippy::too_many_arguments)]
    pub fn backward_attn_with_pool(
        &self,
        qkv: &TensorView<'_>,
        seqlens: &RaggedSeqlenTensors,
        out: &TensorView<'_>,
        out_grad: &TensorView<'_>,
        stats: &TensorView<'_>,
        stream: &Stream,
        workspace_pool: &DeviceMemoryPool,
        pool: &DeviceMemoryPool,
    ) -> anyhow::Result<AttentionBackwardOutput> {
        let b = self.config.max_num_sequences;
        let num_tokens = self.check_qkv(qkv);
        assert!(num_tokens <= self.config.max_total_tokens);

        let mut grad_qkv =
            Tensor::new_zeroed_with_pool(qkv.shape(), qkv.element_type(), stream, &self.cx, pool)?;

        let stats_shape = self.stats_graph_shape(num_tokens);
        let stats = if self.ragged_stats {
            stats.reshaped(stats_shape)
        } else {
            stats.reshaped(stats_shape).transposed(1, 2)
        };
        let [offset_q, offset_k, offset_v, offset_o, offset_stats] = [
            &seqlens.offset_q,
            &seqlens.offset_k,
            &seqlens.offset_v,
            &seqlens.offset_o,
            &seqlens.offset_stats,
        ]
        .map(|offset| offset.reshaped([b + 1, 1, 1, 1]));
        let seqlen = seqlens.seqlen.reshaped([b, 1, 1, 1]);
        let mut inputs = vec![
            (self.backward_input_qkv[0], qkv),
            (self.backward_input_qkv[1], qkv),
            (self.backward_input_qkv[2], qkv),
            (self.backward_input_offset_q, &offset_q),
            (self.backward_input_offset_k, &offset_k),
            (self.backward_input_offset_v, &offset_v),
            (self.backward_input_offset_o, &offset_o),
            (self.backward_input_seqlen_q, &seqlen),
            (self.backward_input_seqlen_kv, &seqlen),
            (self.backward_input_o, out),
            (self.backward_input_do, out_grad),
            (self.backward_input_stats, &stats),
        ];
        if let Some(id) = self.backward_input_offset_stats {
            inputs.push((id, &offset_stats));
        }
        self.backward_graph.execute_aliased(
            &inputs,
            &mut [(&self.backward_output_dqkv[..], grad_qkv.view_mut())],
            stream,
            workspace_pool,
        )?;
        Ok(AttentionBackwardOutput { dqkv: grad_qkv })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn backward_attn(
        &self,
        qkv: &TensorView<'_>,
        seqlens: &RaggedSeqlenTensors,
        out: &TensorView<'_>,
        out_grad: &TensorView<'_>,
        stats: &TensorView<'_>,
        stream: &Stream,
        workspace_pool: &DeviceMemoryPool,
    ) -> anyhow::Result<AttentionBackwardOutput> {
        self.backward_attn_with_pool(
            qkv,
            seqlens,
            out,
            out_grad,
            stats,
            stream,
            workspace_pool,
            self.cx.device_memory_pool(),
        )
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct AttentionConfig {
    pub is_causal: bool,
    pub max_q_seqlen: usize,
    pub max_kv_seqlen: usize,
    pub max_num_sequences: usize,
    /// Upper bound on the total number of packed tokens `T`.
    pub max_total_tokens: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
}

impl AttentionConfig {
    /// Elements per token row of the packed QKV tensor.
    fn packed_row_stride(&self) -> usize {
        self.head_dim * (self.q_heads + 2 * self.kv_heads)
    }

    fn ragged_offset_layout(&self) -> VirtTensorView {
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

    /// `[B, H, S, D]` layout of one of Q, K, V within the packed QKV tensor.
    /// Per-sequence base pointers come from the ragged offset tensors,
    /// so the batch stride is nominal.
    fn packed_layout(&self, heads: usize, max_seqlen: usize) -> VirtTensorView {
        let row = self.packed_row_stride();
        VirtTensorView::with_strides(
            Shape::new([self.max_num_sequences, heads, max_seqlen, self.head_dim]),
            Strides::new([max_seqlen * row, self.head_dim, row, 1]),
            ElementType::Float(FloatType::Bf16),
        )
    }

    fn q_layout(&self) -> VirtTensorView {
        self.packed_layout(self.q_heads, self.max_q_seqlen)
    }

    fn kv_layout(&self) -> VirtTensorView {
        self.packed_layout(self.kv_heads, self.max_kv_seqlen)
    }

    /// `[B, H, S, D]` layout of the `[T, q_heads, head_dim]` output tensor.
    fn o_layout(&self) -> VirtTensorView {
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
        let ragged_stats = !matches!(cx.cuda().compute_capability_major(), 8 | 12);

        let forward_graph = CudnnGraphBuilder::new(cx)?;

        let forward_offset_q = forward_graph.bind_tensor(&self.ragged_offset_layout());
        let forward_offset_k = forward_graph.bind_tensor(&self.ragged_offset_layout());
        let forward_offset_v = forward_graph.bind_tensor(&self.ragged_offset_layout());
        let forward_offset_o = forward_graph.bind_tensor(&self.ragged_offset_layout());
        let forward_seqlen_q = forward_graph.bind_tensor(&self.seqlen_layout());
        let forward_seqlen_kv = forward_graph.bind_tensor(&self.seqlen_layout());

        let forward_input_q = forward_graph.bind_tensor(&self.q_layout());
        let forward_input_k = forward_graph.bind_tensor(&self.kv_layout());
        let forward_input_v = forward_graph.bind_tensor(&self.kv_layout());

        forward_input_q.set_ragged_offset(forward_offset_q, None);
        forward_input_k.set_ragged_offset(forward_offset_k, None);
        forward_input_v.set_ragged_offset(forward_offset_v, None);

        let [forward_output_o, forward_output_stats] = forward_graph.graph().sdpa(
            forward_input_q,
            forward_input_k,
            forward_input_v,
            &SdpaAttributes::new()
                .attn_scale(attn_scale)
                .generate_stats(true)
                .causal_mask(self.is_causal)
                .padding_mask(true)
                .seq_len(forward_seqlen_q, forward_seqlen_kv)
                .max_total_seq_len(self.max_total_tokens, self.max_total_tokens),
        )?;
        let forward_output_o = forward_output_o.unwrap();
        let forward_output_stats = forward_output_stats.unwrap();
        let forward_output_o_id = forward_graph.mark_output(forward_output_o, &self.o_layout());
        forward_output_o.set_ragged_offset(forward_offset_o, None);
        let forward_output_stats_id =
            forward_graph.mark_output(forward_output_stats, &self.stats_layout());
        let forward_offset_stats = ragged_stats.then(|| {
            let offset = forward_graph.bind_tensor(&self.ragged_offset_layout());
            forward_output_stats.set_ragged_offset(offset, None);
            offset
        });

        let forward_input_qkv = [
            forward_input_q.uid(),
            forward_input_k.uid(),
            forward_input_v.uid(),
        ];
        let forward_input_offset_q = forward_offset_q.uid();
        let forward_input_offset_k = forward_offset_k.uid();
        let forward_input_offset_v = forward_offset_v.uid();
        let forward_input_offset_o = forward_offset_o.uid();
        let forward_input_offset_stats = forward_offset_stats.map(|t| t.uid());
        let forward_input_seqlen_q = forward_seqlen_q.uid();
        let forward_input_seqlen_kv = forward_seqlen_kv.uid();
        let forward_graph = forward_graph.build()?;

        let backward_graph = CudnnGraphBuilder::new(cx)?;

        let backward_offset_q = backward_graph.bind_tensor(&self.ragged_offset_layout());
        let backward_offset_k = backward_graph.bind_tensor(&self.ragged_offset_layout());
        let backward_offset_v = backward_graph.bind_tensor(&self.ragged_offset_layout());
        let backward_offset_o = backward_graph.bind_tensor(&self.ragged_offset_layout());
        let backward_seqlen_q = backward_graph.bind_tensor(&self.seqlen_layout());
        let backward_seqlen_kv = backward_graph.bind_tensor(&self.seqlen_layout());

        let backward_input_q = backward_graph.bind_tensor(&self.q_layout());
        let backward_input_k = backward_graph.bind_tensor(&self.kv_layout());
        let backward_input_v = backward_graph.bind_tensor(&self.kv_layout());
        let backward_input_o = backward_graph.bind_tensor(&self.o_layout());
        let backward_input_do = backward_graph.bind_tensor(&self.o_layout());
        let backward_input_stats = backward_graph.bind_tensor(&self.stats_layout());

        backward_input_q.set_ragged_offset(backward_offset_q, None);
        backward_input_k.set_ragged_offset(backward_offset_k, None);
        backward_input_v.set_ragged_offset(backward_offset_v, None);
        backward_input_o.set_ragged_offset(backward_offset_o, None);
        backward_input_do.set_ragged_offset(backward_offset_o, None);
        let backward_offset_stats = ragged_stats.then(|| {
            let offset = backward_graph.bind_tensor(&self.ragged_offset_layout());
            backward_input_stats.set_ragged_offset(offset, None);
            offset
        });

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
                        .max_total_seq_len(self.max_total_tokens, self.max_total_tokens)
                },
            )?;
        let backward_output_dq = backward_output_dq.unwrap();
        let backward_output_dk = backward_output_dk.unwrap();
        let backward_output_dv = backward_output_dv.unwrap();
        let backward_output_dq_id =
            backward_graph.mark_output(backward_output_dq, &self.q_layout());
        backward_output_dq.set_ragged_offset(backward_offset_q, None);
        let backward_output_dk_id =
            backward_graph.mark_output(backward_output_dk, &self.kv_layout());
        backward_output_dk.set_ragged_offset(backward_offset_k, None);
        let backward_output_dv_id =
            backward_graph.mark_output(backward_output_dv, &self.kv_layout());
        backward_output_dv.set_ragged_offset(backward_offset_v, None);

        let backward_input_qkv = [
            backward_input_q.uid(),
            backward_input_k.uid(),
            backward_input_v.uid(),
        ];
        let backward_input_offset_q = backward_offset_q.uid();
        let backward_input_offset_k = backward_offset_k.uid();
        let backward_input_offset_v = backward_offset_v.uid();
        let backward_input_offset_o = backward_offset_o.uid();
        let backward_input_offset_stats = backward_offset_stats.map(|t| t.uid());
        let backward_input_seqlen_q = backward_seqlen_q.uid();
        let backward_input_seqlen_kv = backward_seqlen_kv.uid();
        let backward_input_o = backward_input_o.uid();
        let backward_input_do = backward_input_do.uid();
        let backward_input_stats = backward_input_stats.uid();
        let backward_graph = backward_graph.build()?;

        Ok(AttentionExecutor {
            config: self.clone(),
            cx: cx.clone(),
            ragged_stats,
            forward_graph,
            forward_input_qkv,
            forward_input_offset_q,
            forward_input_offset_k,
            forward_input_offset_v,
            forward_input_offset_o,
            forward_input_offset_stats,
            forward_input_seqlen_q,
            forward_input_seqlen_kv,
            forward_output_o: forward_output_o_id,
            forward_output_stats: forward_output_stats_id,
            backward_graph,
            backward_input_qkv,
            backward_input_offset_q,
            backward_input_offset_k,
            backward_input_offset_v,
            backward_input_offset_o,
            backward_input_offset_stats,
            backward_input_seqlen_q,
            backward_input_seqlen_kv,
            backward_input_o,
            backward_input_do,
            backward_input_stats,
            backward_output_dqkv: [
                backward_output_dq_id,
                backward_output_dk_id,
                backward_output_dv_id,
            ],
        })
    }
}
