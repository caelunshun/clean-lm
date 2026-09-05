use std::sync::Arc;

use singe_cudnn::{
    backend::tensor::TensorId,
    frontend::graph::{
        SdpaForwardConfig, SdpaForwardRaggedConfig, SdpaInputs, SdpaMask, SdpaRaggedOffset,
        SdpaSequenceLengths,
    },
};
use skytens_core::data_type::{ElementType, FloatType, SIntType};

use crate::{
    context::{Context, CudaStream, ResourceKey},
    cudnn::{CudnnGraphBuilder, ExecutableCudnnGraph},
    tensor::{TensorView, VirtTensorView},
};

#[derive(Debug)]
pub struct AttentionExecutor {
    forward_graph: ExecutableCudnnGraph,
    forward_input_cu_seqlen_q: TensorId,
    forward_input_cu_seqlen_kv: TensorId,
    forward_input_seqlen_q: TensorId,
    forward_input_seqlen_kv: TensorId,
    forward_input_q: TensorId,
    forward_input_k: TensorId,
    forward_input_v: TensorId,
    forward_output_o: TensorId,
    forward_output_stats: TensorId,
    // backward_graph: ExecutableCudnnGraph,
    // backward_input_q: TensorId,
    // backward_input_k: TensorId,
    // backward_input_v: TensorId,
    // backward_input_cu_seqlen_q: TensorId,
    // backward_input_cu_seqlen_kv: TensorId,
    // backward_input_o: TensorId,
    // backward_input_do: TensorId,
    // backward_input_stats: TensorId,
    // backward_output_dq: TensorId,
    // backward_output_dk: TensorId,
    // backward_output_dv: TensorId,
}

impl AttentionExecutor {
    pub fn get(config: AttentionConfig, cx: &Context) -> anyhow::Result<Arc<Self>> {
        cx.get_or_create_resource(config)
    }

    /// Q, K, V, O expected as row-major bf16 [T, H, D].
    /// Stats expected as row-major f32 [T, H].
    #[allow(clippy::too_many_arguments)]
    pub fn forward_attn(
        &self,
        q: &TensorView,
        k: &TensorView,
        v: &TensorView,
        cu_seqlen_q: &TensorView,
        cu_seqlen_kv: &TensorView,
        seqlen_q: &TensorView,
        seqlen_kv: &TensorView,
        out: &TensorView,
        out_stats: &TensorView,
        stream: &CudaStream,
    ) -> anyhow::Result<()> {
        self.forward_graph.execute(
            [
                (self.forward_input_q, q),
                (self.forward_input_k, k),
                (self.forward_input_v, v),
                (self.forward_input_cu_seqlen_q, cu_seqlen_q),
                (self.forward_input_cu_seqlen_kv, cu_seqlen_kv),
                (self.forward_input_seqlen_q, seqlen_q),
                (self.forward_input_seqlen_kv, seqlen_kv),
                (self.forward_output_o, out),
                (self.forward_output_stats, out_stats),
            ],
            stream,
        )?;
        Ok(())
    }
}

#[derive(Debug, serde::Serialize)]
pub struct AttentionConfig {
    pub max_q_seqlen: usize,
    pub max_kv_seqlen: usize,
    pub max_num_sequences: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
}

impl ResourceKey for AttentionConfig {
    type Resource = AttentionExecutor;

    fn create(&self, cx: &Context) -> anyhow::Result<Self::Resource> {
        let attn_scale = (1.0 / (self.head_dim as f64).sqrt()) as f32;

        let mut forward_graph = CudnnGraphBuilder::new(cx);

        let forward_cu_seqlen_q = forward_graph.bind_tensor(&VirtTensorView::new(
            [self.max_num_sequences + 1],
            ElementType::SInt(SIntType::S64),
        ));
        let forward_seqlen_q = forward_graph.bind_tensor(&VirtTensorView::new(
            [self.max_num_sequences],
            ElementType::SInt(SIntType::S32),
        ));
        let forward_cu_seqlen_kv = forward_graph.bind_tensor(&VirtTensorView::new(
            [self.max_num_sequences + 1],
            ElementType::SInt(SIntType::S64),
        ));
        let forward_seqlen_kv = forward_graph.bind_tensor(&VirtTensorView::new(
            [self.max_num_sequences],
            ElementType::SInt(SIntType::S32),
        ));

        let forward_input_q = forward_graph.bind_tensor(
            &VirtTensorView::new(
                [
                    self.max_num_sequences,
                    self.max_q_seqlen,
                    self.q_heads,
                    self.head_dim,
                ],
                ElementType::Float(FloatType::Bf16),
            )
            .transposed(1, 2),
        );
        let forward_input_k = forward_graph.bind_tensor(
            &VirtTensorView::new(
                [
                    self.max_num_sequences,
                    self.max_kv_seqlen,
                    self.kv_heads,
                    self.head_dim,
                ],
                ElementType::Float(FloatType::Bf16),
            )
            .transposed(1, 2),
        );
        let forward_input_v = forward_graph.bind_tensor(
            &VirtTensorView::new(
                [
                    self.max_num_sequences,
                    self.max_kv_seqlen,
                    self.kv_heads,
                    self.head_dim,
                ],
                ElementType::Float(FloatType::Bf16),
            )
            .transposed(1, 2),
        );

        forward_graph.graph_mut().set_ragged_layout(
            forward_input_q,
            forward_cu_seqlen_q,
            (self.head_dim * self.q_heads) as i64,
        )?;
        forward_graph.graph_mut().set_ragged_layout(
            forward_input_k,
            forward_cu_seqlen_kv,
            (self.head_dim * self.kv_heads) as i64,
        )?;
        forward_graph.graph_mut().set_ragged_layout(
            forward_input_v,
            forward_cu_seqlen_kv,
            (self.head_dim * self.kv_heads) as i64,
        )?;

        let forward_outputs = forward_graph.graph_mut().sdpa_infer(
            SdpaInputs {
                query: forward_input_q,
                key: forward_input_k,
                value: forward_input_v,
            },
            SdpaForwardConfig::new(attn_scale)
                .with_mask(SdpaMask::Causal)
                .with_ragged(SdpaForwardRaggedConfig {
                    sequence_lengths: SdpaSequenceLengths {
                        query: forward_seqlen_q,
                        key_value: forward_seqlen_kv,
                    },
                    output_ragged_offset: SdpaRaggedOffset::new(forward_cu_seqlen_q)
                        .with_multiplier((self.head_dim * self.q_heads) as i64),
                    stats_ragged_offset: SdpaRaggedOffset::new(forward_cu_seqlen_q)
                        .with_multiplier(self.q_heads as i64),
                }),
        )?;
        let forward_graph = forward_graph.build()?;

        Ok(AttentionExecutor {
            forward_graph,
            forward_input_cu_seqlen_q: forward_cu_seqlen_q,
            forward_input_cu_seqlen_kv: forward_cu_seqlen_kv,

            forward_input_q,
            forward_input_k,
            forward_input_v,
            forward_output_o: forward_outputs.output(),
            forward_output_stats: forward_outputs.stats(),
            forward_input_seqlen_q: forward_seqlen_q,
            forward_input_seqlen_kv: forward_seqlen_kv,
        })
    }
}
