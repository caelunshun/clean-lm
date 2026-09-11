use std::{array, sync::Arc};

use anyhow::bail;
use half::{f16, slice::HalfFloatSliceExt};
use serde::Serialize;
use skycuda::{module::Module, stream::Stream};
use skytens_core::data_type::{ElementType, FloatType, SIntType};
use wide::f32x16;

use crate::{
    context::{Context, ResourceKey},
    include_cubins,
    tensor::{Tensor, TensorView, TensorViewMut},
};

#[derive(Debug)]
#[allow(dead_code)]
pub struct RopeExecutor {
    cx: Context,
    config: RopeConfig,
    module: Arc<Module>,
    fwd_cos_table: Tensor,
    fwd_sin_table: Tensor,
    bwd_cos_table: Tensor,
    bwd_sin_table: Tensor,
}

impl RopeExecutor {
    pub fn get(config: RopeConfig, cx: &Context) -> anyhow::Result<Arc<Self>> {
        cx.get_or_create_resource(config)
    }

    #[allow(unused_variables)]
    pub fn execute(
        &self,
        x: &TensorView<'_>,
        cu_seqlen: &TensorView<'_>,
        seqlens: &[u32],
        out: &mut TensorViewMut<'_>,
        mode: RopeMode,
        stream: &Stream,
    ) -> anyhow::Result<()> {
        assert_eq!(cu_seqlen.element_type(), ElementType::SInt(SIntType::S64));
        assert_eq!(x.element_type(), ElementType::Float(FloatType::Bf16));
        assert_eq!(x.shape().num_dims(), 3);
        assert_eq!(x.shape()[2], self.config.head_dim);
        assert_eq!(out.shape(), x.shape());
        assert_eq!(out.element_type(), x.element_type());
        assert_eq!(cu_seqlen.shape().dims_slice(), &[seqlens.len() + 1]);

        assert!(x.is_packed_row_major());
        assert!(cu_seqlen.is_packed_row_major());
        assert!(out.is_packed_row_major());

        let (block_size, kernel_name) = match self.config.head_dim {
            64 => (64, "rope_hd64_64x1x64"),
            hd => bail!("unimplemented head dimension {hd} for rope"),
        };

        let mut cta_work_items = Vec::<u64>::new();
        for (sequence_idx, &seqlen) in seqlens.iter().enumerate() {
            assert!(seqlen as usize <= self.config.max_seqlen);
            for block_idx in 0..seqlen.div_ceil(block_size) {
                cta_work_items.push(sequence_idx as u64 | ((block_idx as u64) << 32));
            }
        }

        // Sort CTAs by block index (position in sequence)
        // to optimize for sin/cos table cache locality
        cta_work_items.sort_unstable_by_key(|encoded| *encoded >> 32);

        let mut cta_work_items_mem = self
            .cx
            .host_pinned_memory_pool()
            .alloc_zeroed(size_of_val(cta_work_items.as_slice()))?;
        cta_work_items_mem
            .view_mut()?
            .copy_from_slice(bytemuck::cast_slice(&cta_work_items));

        let kernel = self.module.get_function("rope_hd64_64x1x64")?;

        let (cos_table, sin_table) = match mode {
            RopeMode::Forward => (&self.fwd_cos_table, &self.fwd_sin_table),
            RopeMode::Backward => (&self.bwd_cos_table, &self.bwd_sin_table),
        };

        unsafe {
            stream
                .launch_builder(&kernel)
                .arg(x.storage())?
                .arg(cu_seqlen.storage())?
                .arg(cos_table.storage())?
                .arg(sin_table.storage())?
                .arg(&cta_work_items_mem)?
                .arg(out.storage_mut())?
                .arg(i32::try_from(x.shape()[0])?)?
                .arg(i32::try_from(x.shape()[1])?)?
                .arg(i32::try_from(self.config.max_seqlen)?)?
                .launch(cta_work_items.len().try_into()?, 1, 1)?;
        }

        Ok(())
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
        assert!(self.max_seqlen.is_multiple_of(64));
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
            cx: cx.clone(),
        })
    }
}

fn build_tables(
    base_freq: f32,
    max_seqlen: usize,
    head_dim: usize,
    dir: f32,
    cx: &Context,
) -> anyhow::Result<(Tensor, Tensor)> {
    let (cos_data, sin_data) = build_table_data(base_freq, max_seqlen, head_dim, dir);
    let cos_tensor = upload_table(&cos_data, [max_seqlen, head_dim / 2], cx)?;
    let sin_tensor = upload_table(&sin_data, [max_seqlen, head_dim / 2], cx)?;
    Ok((cos_tensor, sin_tensor))
}

fn upload_table(data: &[f16], shape: [usize; 2], cx: &Context) -> anyhow::Result<Tensor> {
    let stream = cx.default_stream();
    let mut tensor = Tensor::new_zeroed(shape, ElementType::Float(FloatType::F16), stream, cx)?;
    let bytes: &[u8] = bytemuck::cast_slice(data);
    let mut staging = cx.host_pinned_memory_pool().alloc_zeroed(bytes.len())?;
    staging.view_mut()?.copy_from_slice(bytes);
    stream.copy(&staging, tensor.storage_mut())?;
    Ok(tensor)
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
            .as_chunks_mut::<16>()
            .0
            .iter_mut()
            .zip(sin_table.as_chunks_mut::<16>().0)
            .enumerate()
        {
            let index = f32x16::new(array::from_fn(|offset| (i * 16 + offset) as u32 as f32));
            let freq = f32x16::splat(base_freq).powf_simd(index * (-2.0 / head_dim as f32));
            let theta = freq * (pos as f32 * dir);

            let (sin_chunk, cos_chunk) = theta.sin_cos();
            cos_chunk_f16.convert_from_f32_slice(cos_chunk.as_array());
            sin_chunk_f16.convert_from_f32_slice(sin_chunk.as_array());
        }
    }

    (cos_table, sin_table)
}
