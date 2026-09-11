use std::{array, cell::RefCell, sync::Arc};

use anyhow::bail;
use half::{f16, slice::HalfFloatSliceExt};
use serde::Serialize;
use singe_cuda::module::Module;
use skytens_core::data_type::{ElementType, FloatType, SIntType};
use thread_local::ThreadLocal;
use wide::f32x16;

use crate::{
    bulk_copy::BulkCopyEngine,
    context::{Context, CudaStream, PerStreamResource, ResourceKey},
    include_cubins,
    tensor::TensorView,
};

#[derive(Debug)]
pub struct RopeExecutor {
    config: RopeConfig,
    module: Arc<Module>,
    fwd_cos_table: TensorView,
    fwd_sin_table: TensorView,
    bwd_cos_table: TensorView,
    bwd_sin_table: TensorView,
    bulk_copy_engine: ThreadLocal<RefCell<BulkCopyEngine>>,
}

impl RopeExecutor {
    pub fn get(config: RopeConfig, cx: &Context) -> anyhow::Result<Arc<Self>> {
        cx.get_or_create_resource(config)
    }

    pub fn execute(
        &self,
        x: &TensorView,
        cu_seqlen: &TensorView,
        seqlens: &[u32],
        out: &TensorView,
        mode: RopeMode,
        stream: &CudaStream,
    ) -> anyhow::Result<()> {
        assert_eq!(cu_seqlen.element_type(), ElementType::SInt(SIntType::S64));
        assert_eq!(x.element_type(), ElementType::Float(FloatType::Bf16));
        assert_eq!(x.shape().num_dims(), 3);
        assert_eq!(x.shape()[2], self.config.head_dim);
        assert_eq!(out.shape(), x.shape());
        assert_eq!(out.element_type(), x.element_type());
        assert_eq!(cu_seqlen.shape().dims_slice(), &[seqlens.len() + 1]);

        let (block_size, kernel_name) = match self.config.head_dim {
            64 => (64, "rope_hd64_64x1x64"),
            hd => bail!("unimplemented head dimension {hd} for rope"),
        };

        let mut cta_work_items = Vec::<u64>::new();
        for (sequence_idx, &seqlen) in seqlens.iter().enumerate() {
            for block_idx in 0..seqlen.div_ceil(block_size) {
                cta_work_items.push(sequence_idx as u64 | ((block_idx as u64) << 32));
            }
        }

        // Sort CTAs by block index (position in sequence)
        // to optimize for sin/cos table cache locality
        cta_work_items.sort_unstable_by_key(|encoded| *encoded >> 32);

        todo!()
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum RopeMode {
    Forward,
    Backward,
}

#[derive(Debug, Clone, Serialize)]
pub struct RopeConfig {
    pub base_freq: f32,
    pub max_seqlen: usize,
    pub head_dim: usize,
}

impl ResourceKey for RopeConfig {
    type Resource = RopeExecutor;

    fn create(&self, cx: &Context) -> anyhow::Result<Self::Resource> {
        let (fwd_cos_table, fwd_sin_table) =
            build_tables(self.base_freq, self.max_seqlen, self.head_dim, 1.0, cx)?;
        let (bwd_cos_table, bwd_sin_table) =
            build_tables(self.base_freq, self.max_seqlen, self.head_dim, -1.0, cx)?;

        let module = cx.get_or_create_resource(include_cubins!("../../cutile-kernels/rope"))?;
        Ok(RopeExecutor {
            config: self.clone(),
            module,
            fwd_cos_table,
            fwd_sin_table,
            bwd_cos_table,
            bwd_sin_table,
            bulk_copy_engine: ThreadLocal::new(),
        })
    }
}

fn build_tables(
    base_freq: f32,
    max_seqlen: usize,
    head_dim: usize,
    dir: f32,
    cx: &Context,
) -> anyhow::Result<(TensorView, TensorView)> {
    let (cos_data, sin_data) = build_table_data(base_freq, max_seqlen, head_dim, dir);
    let cos_tensor = TensorView::new_zeroed(
        [max_seqlen, head_dim / 2],
        ElementType::Float(FloatType::F16),
        cx,
    )?;
    let sin_tensor = TensorView::new_zeroed(
        [max_seqlen, head_dim / 2],
        ElementType::Float(FloatType::F16),
        cx,
    )?;

    let mut copier = BulkCopyEngine::new(cx)?;
    let mut batch = copier.start_batch()?;
    batch.copy_to_device(&cos_tensor, |dst| {
        dst.copy_from_slice(bytemuck::cast_slice(&cos_data))
    })?;
    batch.copy_to_device(&sin_tensor, |dst| {
        dst.copy_from_slice(bytemuck::cast_slice(&sin_data))
    })?;
    batch.commit()?;
    copier.block_on_last_batch()?;

    Ok((cos_tensor, sin_tensor))
}

fn build_table_data(
    base_freq: f32,
    max_seqlen: usize,
    head_dim: usize,
    dir: f32,
) -> (Vec<f16>, Vec<f16>) {
    let mut cos_table = vec![f16::ZERO; max_seqlen * head_dim / 2];
    let mut sin_table = vec![f16::ZERO; max_seqlen * head_dim / 2];

    assert!(head_dim.is_multiple_of(2 * 16));

    for (pos, (cos_table, sin_table)) in cos_table
        .chunks_exact_mut(head_dim / 2)
        .zip(sin_table.chunks_exact_mut(head_dim / 2))
        .enumerate()
    {
        for (i, (cos_chunk_f16, sin_chunk_f16)) in cos_table
            .chunks_exact_mut(16)
            .zip(sin_table.chunks_exact_mut(16))
            .enumerate()
        {
            let index = f32x16::new(array::from_fn(|offset| (i + offset) as u32 as f32));
            let freq = f32x16::splat(base_freq).powf_simd(index * (-2.0 / head_dim as f32));
            let theta = freq * (pos as f32 * dir);

            let (sin_chunk, cos_chunk) = theta.sin_cos();
            cos_chunk_f16.convert_from_f32_slice(cos_chunk.as_array());
            sin_chunk_f16.convert_from_f32_slice(sin_chunk.as_array());
        }
    }

    (cos_table, sin_table)
}
