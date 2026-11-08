//! Buffer liveness, operator building, and execution.

use super::{
    ExecGraph, GraphOutput, Input,
    graph::{BufferId, BufferPart, BufferSource, Graph, SymTensorData, SymTensorId},
    op::{
        Access, BuildCx, Executor, Init, Mem, MemMap, TensorRef, TensorRefMut, bind_io,
        op_buffer_access,
    },
    sym::{DimValues, Unifier},
    tune::{Autotune, Tuner},
};
use crate::{
    context::Context,
    mx_tensor::{MxLayout, MxTensor},
    tensor::Tensor,
};
use anyhow::{Context as _, anyhow, bail, ensure};
use rayon::prelude::*;
use skycuda::{
    device::Device, device_memory::DeviceMemory, device_memory_pool::DeviceMemoryPool,
    memory_slice::MemorySlice, stream::Stream,
};
use skytens_core::data_type::ElementType;
use std::collections::HashSet;
use tensor_ref_storage::{mut_storage, ro_storage};

#[derive(Debug)]
struct OpPlan {
    operands: Vec<SymTensorId>,
    access: Vec<Access>,
    /// Distinct buffers used by the op, and whether each is written.
    buffers: Vec<(BufferId, bool)>,
    /// Buffers allocated right before the op runs.
    alloc: Vec<BufferId>,
    /// Buffers freed right after the op runs.
    free: Vec<BufferId>,
}

#[derive(Debug)]
pub(crate) struct Plan {
    ops: Vec<OpPlan>,
}

impl Plan {
    pub fn new(graph: &Graph) -> Self {
        let info = &graph.info;
        let mut ops: Vec<OpPlan> = graph
            .ops
            .iter()
            .map(|op| {
                let operands: Vec<SymTensorId> = op.operands.iter().map(|o| o.tensor).collect();
                let access: Vec<Access> = op.operands.iter().map(|o| o.access).collect();
                let buffers = op_buffer_access(info, &operands, &access, &op.outputs);
                OpPlan {
                    operands,
                    access,
                    buffers,
                    alloc: Vec::new(),
                    free: Vec::new(),
                }
            })
            .collect();

        let kept: HashSet<BufferId> = graph
            .outputs
            .iter()
            .flat_map(|&t| info.tensor(t).buffers())
            .collect();
        let mut last_use = vec![None; info.buffers.len()];
        for (i, op) in ops.iter().enumerate() {
            for &(buffer, _) in &op.buffers {
                last_use[buffer.0] = Some(i);
            }
        }
        for (b, buffer) in info.buffers.iter().enumerate() {
            let BufferSource::Op { op, .. } = buffer.source else {
                continue;
            };
            let id = BufferId(b);
            ops[op].alloc.push(id);
            if !kept.contains(&id) {
                ops[last_use[b].unwrap_or(op)].free.push(id);
            }
        }
        Self { ops }
    }
}

pub(crate) fn build_executors(
    graph: &Graph,
    cx: &Context,
    autotune: Autotune,
    pool: &DeviceMemoryPool,
) -> anyhow::Result<Vec<Box<dyn Executor>>> {
    let tuner = Tuner::new(cx, autotune);
    let results: Vec<anyhow::Result<Box<dyn Executor>>> = graph
        .ops
        .par_iter()
        .enumerate()
        .map(|(i, record)| {
            let operands: Vec<SymTensorId> = record.operands.iter().map(|o| o.tensor).collect();
            let access: Vec<Access> = record.operands.iter().map(|o| o.access).collect();
            let bcx = BuildCx {
                cx,
                graph: &graph.info,
                operands: &operands,
                outputs: &record.outputs,
                tuner: &tuner,
                autotune,
                op_name: record.op.name(),
                operand_access: &access,
                pool,
            };
            record
                .op
                .build(&bcx)
                .with_context(|| format!("failed to build op {i} ({})", record.op.name()))
        })
        .collect();

    let mut executors = Vec::with_capacity(results.len());
    let mut errors = Vec::new();
    for result in results {
        match result {
            Ok(executor) => executors.push(executor),
            Err(e) => errors.push(format!("{e:#}")),
        }
    }
    match errors.len() {
        0 => Ok(executors),
        1 => Err(anyhow!("{}", errors[0])),
        n => Err(anyhow!(
            "{n} ops failed to build:\n  {}",
            errors.join("\n  ")
        )),
    }
}

/// Borrows the storage behind caller-provided tensor references.
mod tensor_ref_storage {
    use super::*;

    pub fn ro_storage<'a>(t: &TensorRef<'a>, part: BufferPart) -> MemorySlice<&'a DeviceMemory> {
        match (t, part) {
            (TensorRef::Plain(t), _) => *t.storage(),
            (TensorRef::Mx(t), BufferPart::Scales) => *t.scale_factors().storage(),
            (TensorRef::Mx(t), _) => *t.values().storage(),
        }
    }

    /// Shared or exclusive access to the parts of a caller-provided mutable tensor.
    pub fn mut_storage<'x>(
        t: &'x mut TensorRefMut<'_>,
        parts: &[(BufferId, BufferPart, bool)],
        mems: &mut MemMap<'x>,
    ) {
        let (values, scales) = match t {
            TensorRefMut::Plain(t) => (t.storage_mut(), None),
            TensorRefMut::Mx(t) => {
                let (values, scales) = t.values_and_scale_factors_mut();
                (values.storage_mut(), Some(scales.storage_mut()))
            }
        };
        let mut scales = scales;
        let mut values = Some(values);
        for &(buffer, part, write) in parts {
            let slot = match part {
                BufferPart::Scales => scales.take(),
                _ => values.take(),
            };
            let slice = slot.expect("tensor part bound twice");
            let (offset, len) = (slice.offset(), slice.len());
            let mem = if write {
                Mem::Rw(slice.inner_mut().slice_mut(offset..offset + len))
            } else {
                let inner: &'x DeviceMemory = slice.inner_mut();
                Mem::Ro(inner.slice(offset..offset + len))
            };
            mems.insert(buffer, mem);
        }
    }
}

fn check_device(
    mem: &MemorySlice<&DeviceMemory>,
    device: &Device,
    name: &str,
) -> anyhow::Result<()> {
    ensure!(
        mem.inner().device() == device,
        "input {name} is on device {}, but the graph runs on device {}",
        mem.inner().device().ordinal(),
        device.ordinal()
    );
    Ok(())
}

/// Checks the inputs against their declarations and infers the dimension values.
fn bind_inputs(graph: &Graph, inputs: &[Input<'_>], cx: &Context) -> anyhow::Result<DimValues> {
    ensure!(
        inputs.len() == graph.inputs.len(),
        "graph has {} inputs, but {} were given",
        graph.inputs.len(),
        inputs.len()
    );
    let mut unifier = Unifier::new();
    for (info, input) in graph.inputs.iter().zip(inputs) {
        let name = &info.name;
        let tensor = match input {
            Input::Ro(t) => {
                ensure!(
                    !info.mutable,
                    "input {name} is mutable and must be given as Input::Rw"
                );
                t.clone()
            }
            Input::Rw(t) => t.as_ref(),
        };
        ensure!(
            tensor.ty() == info.ty,
            "input {name}: expected {:?}, got {:?}",
            info.ty,
            tensor.ty()
        );
        match (graph.info.tensor(info.tensor), &tensor) {
            (SymTensorData::Plain { layout, .. }, TensorRef::Plain(t)) => {
                check_device(t.storage(), cx.cuda(), name)?;
                unifier.add_layout(name, layout, &t.shape(), &t.strides(), 0)?;
            }
            (SymTensorData::BlockScaled { shape, .. }, TensorRef::Mx(t)) => {
                check_device(t.values().storage(), cx.cuda(), name)?;
                check_device(t.scale_factors().storage(), cx.cuda(), name)?;
                for (i, dim) in shape.dims().iter().enumerate() {
                    unifier.add_equation(format!("{name} shape[{i}]"), dim.clone(), t.shape()[i]);
                }
            }
            _ => unreachable!("input kinds were checked above"),
        }
    }
    let mut dims = DimValues::new();
    unifier.solve(&mut dims, &graph.info.dims)?;
    Ok(dims)
}

fn alloc_buffer(
    bytes: usize,
    ty: ElementType,
    init: Init,
    stream: &Stream,
    pool: &DeviceMemoryPool,
) -> anyhow::Result<DeviceMemory> {
    Ok(match init {
        Init::Zero => pool.alloc_zeroed_on(bytes, stream)?,
        // Soundness: operators with uninitialized outputs write them fully before reading.
        Init::Uninit => unsafe { pool.alloc_uninit_on(bytes, stream)? },
        Init::Fill32(bits) => {
            ensure!(ty.bits_per_element() == 32, "Fill32 of a {ty:?} buffer");
            let mut mem = unsafe { pool.alloc_uninit_on(bytes, stream)? };
            stream.fill(&mut mem, bits)?;
            mem
        }
    })
}

pub(crate) fn execute(
    eg: &ExecGraph,
    inputs: &mut [Input<'_>],
    stream: &Stream,
    pool: &DeviceMemoryPool,
) -> anyhow::Result<Vec<GraphOutput>> {
    let graph = &eg.graph;
    let info = &graph.info;
    let cx = &eg.cx;
    ensure!(
        pool.device() == cx.cuda(),
        "memory pool is on device {}, but the graph runs on device {}",
        pool.device().ordinal(),
        cx.cuda().ordinal()
    );
    let dims = bind_inputs(graph, inputs, cx)?;

    let scratch_bytes = eg
        .executors
        .iter()
        .map(|e| e.scratch_bytes(&dims))
        .max()
        .unwrap_or(0);
    let mut scratch = if scratch_bytes > 0 {
        // Soundness: scratch memory has no defined contents for operators.
        Some(unsafe { pool.alloc_uninit_on(scratch_bytes, stream)? })
    } else {
        None
    };

    let mut buffers: Vec<Option<DeviceMemory>> = (0..info.buffers.len()).map(|_| None).collect();
    for (i, (op, executor)) in eg.plan.ops.iter().zip(&eg.executors).enumerate() {
        let record = &graph.ops[i];
        let op_context = || format!("op {i} ({})", record.op.name());

        for &buffer in &op.alloc {
            let buffer_info = info.buffer(buffer);
            let BufferSource::Op { init, .. } = buffer_info.source else {
                unreachable!()
            };
            let bytes = buffer_info.bytes.eval(&dims).with_context(op_context)?;
            buffers[buffer.0] = Some(
                alloc_buffer(bytes, buffer_info.ty, init, stream, pool).with_context(op_context)?,
            );
        }

        let mut mems = MemMap::new();
        // Graph-owned buffers, borrowed in one pass so that writes are exclusive.
        let mut wanted: Vec<(BufferId, bool)> = op
            .buffers
            .iter()
            .copied()
            .filter(|(b, _)| matches!(info.buffer(*b).source, BufferSource::Op { .. }))
            .collect();
        wanted.sort_unstable();
        let mut wanted_iter = wanted.iter().peekable();
        for (b, mem) in buffers.iter_mut().enumerate() {
            let Some(&&(buffer, write)) = wanted_iter.peek() else {
                break;
            };
            if buffer.0 != b {
                continue;
            }
            wanted_iter.next();
            let mem = mem.as_mut().expect("buffer used before allocation");
            mems.insert(
                buffer,
                if write {
                    Mem::Rw(mem.slice_mut(..))
                } else {
                    let mem: &DeviceMemory = mem;
                    Mem::Ro(mem.slice(..))
                },
            );
        }
        // Graph inputs.
        for (input_index, input) in inputs.iter_mut().enumerate() {
            let parts: Vec<(BufferId, BufferPart, bool)> = op
                .buffers
                .iter()
                .filter_map(|&(buffer, write)| match info.buffer(buffer).source {
                    BufferSource::Input { input, part } if input == input_index => {
                        Some((buffer, part, write))
                    }
                    _ => None,
                })
                .collect();
            if parts.is_empty() {
                continue;
            }
            match input {
                Input::Ro(t) => {
                    for &(buffer, part, write) in &parts {
                        ensure!(!write, "input {input_index} is written but read-only");
                        mems.insert(buffer, Mem::Ro(ro_storage(t, part)));
                    }
                }
                Input::Rw(t) => mut_storage(t, &parts, &mut mems),
            }
        }

        let op_scratch_bytes = executor.scratch_bytes(&dims);
        let op_scratch = match &mut scratch {
            Some(scratch) if op_scratch_bytes > 0 => Some(scratch.slice_mut(..op_scratch_bytes)),
            _ => None,
        };
        let mut io = bind_io(
            info,
            &op.operands,
            &op.access,
            &record.outputs,
            mems,
            op_scratch,
            &dims,
            cx,
        )
        .with_context(op_context)?;
        executor
            .execute(&mut io, stream, cx)
            .with_context(op_context)?;
        drop(io);

        for &buffer in &op.free {
            buffers[buffer.0] = None;
        }
    }

    graph
        .outputs
        .iter()
        .map(|&tensor| {
            let mut take = |buffer: BufferId| {
                buffers[buffer.0]
                    .take()
                    .ok_or_else(|| anyhow!("output buffer {buffer:?} was not allocated"))
            };
            Ok(match info.tensor(tensor) {
                SymTensorData::Plain { buffer, layout, ty } => {
                    let (shape, strides, offset) = layout.eval(&dims)?;
                    if offset != 0 {
                        bail!("output has nonzero offset {offset}");
                    }
                    let virt = crate::tensor::VirtTensorView::with_strides(shape, strides, *ty);
                    GraphOutput::Plain(Tensor::from_parts(virt, take(*buffer)?, cx))
                }
                SymTensorData::BlockScaled {
                    values,
                    scales,
                    format,
                    dim,
                    shape,
                } => {
                    let shape = shape.eval(&dims)?;
                    let layout = MxLayout::new(shape, *format, *dim)?;
                    GraphOutput::Mx(MxTensor::from_parts(
                        Tensor::from_parts(layout.values, take(*values)?, cx),
                        Tensor::from_parts(layout.scale_factors, take(*scales)?, cx),
                        *format,
                        *dim,
                        shape,
                    )?)
                }
            })
        })
        .collect()
}
