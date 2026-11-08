//! Buffer liveness, operator building, and execution.

use super::{
    ExecGraph, GraphOutput, Input,
    bind::{Mem, MemMap, OpPlacement, Placement, bind_io, disjoint_mut, place, place_op},
    graph::{BufferId, BufferPart, BufferSource, Graph, SymTensorId},
    op::{Access, BuildCx, Executor, Init, TensorRef, TensorRefMut, op_buffer_access},
    sym::{DimValues, SymDimId, Unifier},
    tune::{Autotune, Tuner},
};
use crate::{context::Context, mx_tensor::MxTensor, tensor::Tensor};
use anyhow::{Context as _, anyhow, ensure};
use rayon::prelude::*;
use skycuda::{
    device::Device, device_memory::DeviceMemory, device_memory_pool::DeviceMemoryPool,
    memory_slice::MemorySlice, stream::Stream,
};
use skytens_core::data_type::ElementType;
use std::collections::HashSet;

/// Every input's storage must start at a multiple of this many bytes.
pub(crate) const INPUT_ALIGNMENT: usize = 16;

/// Parts of a graph input used by one op.
#[derive(Debug)]
struct InputUse {
    input: usize,
    /// `(buffer, part, written)`
    parts: Vec<(BufferId, BufferPart, bool)>,
}

#[derive(Debug)]
struct OpPlan {
    operands: Vec<SymTensorId>,
    access: Vec<Access>,
    /// Graph-owned buffers used by the op, sorted, and whether each is written.
    owned: Vec<(BufferId, bool)>,
    /// Graph inputs used by the op, sorted by input index.
    inputs: Vec<InputUse>,
    /// Buffers allocated right before the op runs.
    alloc: Vec<BufferId>,
    /// Buffers freed right after the op runs.
    free: Vec<BufferId>,
}

#[derive(Debug)]
pub(crate) struct Plan {
    ops: Vec<OpPlan>,
    /// Dimensions appearing in some tensor shape. Others only affect strides.
    shape_dims: HashSet<SymDimId>,
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
                let mut owned = Vec::new();
                let mut inputs: Vec<InputUse> = Vec::new();
                for (buffer, write) in op_buffer_access(info, &operands, &access, &op.outputs) {
                    match info.buffer(buffer).source {
                        BufferSource::Op { .. } => owned.push((buffer, write)),
                        BufferSource::Input { input, part } => {
                            match inputs.iter_mut().find(|u| u.input == input) {
                                Some(u) => u.parts.push((buffer, part, write)),
                                None => inputs.push(InputUse {
                                    input,
                                    parts: vec![(buffer, part, write)],
                                }),
                            }
                        }
                    }
                }
                owned.sort_unstable();
                inputs.sort_unstable_by_key(|u| u.input);
                OpPlan {
                    operands,
                    access,
                    owned,
                    inputs,
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
            for &(buffer, _) in &op.owned {
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

        let shape_dims = info
            .tensors
            .iter()
            .flat_map(|t| t.shape().dims().iter().flat_map(|d| d.atoms().to_vec()))
            .collect();
        Self { ops, shape_dims }
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
            Err(e) => errors.push(e),
        }
    }
    if errors.len() > 1 {
        let messages: Vec<String> = errors.iter().map(|e| format!("{e:#}")).collect();
        return Err(anyhow!(
            "{} ops failed to build:\n  {}",
            errors.len(),
            messages.join("\n  ")
        ));
    }
    match errors.pop() {
        Some(e) => Err(e),
        None => Ok(executors),
    }
}

fn ro_storage<'a>(t: &TensorRef<'a>, part: BufferPart) -> MemorySlice<&'a DeviceMemory> {
    match (t, part) {
        (TensorRef::Plain(t), _) => *t.storage(),
        (TensorRef::Mx(t), BufferPart::Scales) => *t.scale_factors().storage(),
        (TensorRef::Mx(t), _) => *t.values().storage(),
    }
}

/// Shared or exclusive access to the parts of a caller-provided mutable tensor.
fn mut_storage<'x>(
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
    let mut values = Some(values);
    let mut scales = scales;
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

fn input_ref<'x>(input: &'x Input<'_>) -> TensorRef<'x> {
    match input {
        Input::Ro(t) => t.clone(),
        Input::Rw(t) => t.as_ref(),
    }
}

fn check_storage(
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
    ensure!(
        mem.offset().is_multiple_of(INPUT_ALIGNMENT),
        "input {name}: storage starts at byte offset {} of its allocation, which is not {INPUT_ALIGNMENT}-byte aligned",
        mem.offset()
    );
    Ok(())
}

/// Checks the inputs against their declarations and infers the dimension values.
fn bind_inputs(
    graph: &Graph,
    plan: &Plan,
    inputs: &[Input<'_>],
    cx: &Context,
) -> anyhow::Result<DimValues> {
    ensure!(
        inputs.len() == graph.inputs.len(),
        "graph has {} inputs, but {} were given",
        graph.inputs.len(),
        inputs.len()
    );
    let mut unifier = Unifier::new();
    for (info, input) in graph.inputs.iter().zip(inputs) {
        let name = &info.name;
        ensure!(
            !info.mutable || matches!(input, Input::Rw(_)),
            "input {name} is mutable and must be given as Input::Rw"
        );
        let tensor = input_ref(input);
        ensure!(
            tensor.ty() == info.ty,
            "input {name}: expected {:?}, got {:?}",
            info.ty,
            tensor.ty()
        );
        match (graph.info.tensor(info.tensor).layout(), &tensor) {
            (Some(layout), TensorRef::Plain(t)) => {
                check_storage(t.storage(), cx.cuda(), name)?;
                unifier.add_layout(name, layout, &t.shape(), &t.strides(), 0)?;
            }
            (None, TensorRef::Mx(t)) => {
                check_storage(t.values().storage(), cx.cuda(), name)?;
                check_storage(t.scale_factors().storage(), cx.cuda(), name)?;
                let shape = graph.info.tensor(info.tensor).shape();
                for (i, dim) in shape.dims().iter().enumerate() {
                    unifier.add_equation(format!("{name} shape[{i}]"), dim.clone(), t.shape()[i]);
                }
            }
            _ => unreachable!("input kinds were checked above"),
        }
    }
    let mut dims = DimValues::new();
    unifier.solve(&mut dims, &graph.info.dims)?;

    // Dimensions whose only equations were skipped (strides of size-1 or empty
    // dimensions) don't affect addressing; bind them to their hints.
    for (id, dim) in graph.info.dims.iter() {
        if !dims.contains(id) {
            ensure!(
                !plan.shape_dims.contains(&id),
                "dimension {} cannot be inferred from the given inputs",
                dim.name
            );
            dims.bind(id, dim.hint);
        }
    }
    Ok(dims)
}

fn input_storage_len(input: &Input<'_>, part: BufferPart) -> usize {
    ro_storage(&input_ref(input), part).len()
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
    ensure!(
        stream.device() == cx.cuda(),
        "stream is on device {}, but the graph runs on device {}",
        stream.device().ordinal(),
        cx.cuda().ordinal()
    );

    // Validate everything before enqueuing any work, so errors leave inputs untouched.
    let dims = bind_inputs(graph, &eg.plan, inputs, cx)?;
    let buffer_lens = info
        .buffers
        .iter()
        .enumerate()
        .map(|(b, buffer)| match buffer.source {
            BufferSource::Op { .. } => buffer
                .bytes
                .eval(&dims)
                .with_context(|| format!("size of buffer {b}")),
            BufferSource::Input { input, part } => Ok(input_storage_len(&inputs[input], part)),
        })
        .collect::<anyhow::Result<Vec<usize>>>()?;
    let buffer_len = |b: BufferId| buffer_lens[b.0];
    let placements = eg
        .plan
        .ops
        .iter()
        .zip(&graph.ops)
        .enumerate()
        .map(|(i, (op, record))| {
            place_op(info, &op.operands, &record.outputs, &dims, &buffer_len)
                .with_context(|| format!("op {i} ({})", record.op.name()))
        })
        .collect::<anyhow::Result<Vec<OpPlacement>>>()?;
    let output_placements = graph
        .outputs
        .iter()
        .enumerate()
        .map(|(i, &t)| {
            place(info.tensor(t), &dims, &buffer_len).with_context(|| format!("output {i}"))
        })
        .collect::<anyhow::Result<Vec<Placement>>>()?;
    let op_scratch: Vec<usize> = eg
        .executors
        .iter()
        .map(|e| e.scratch_bytes(&dims))
        .collect();
    let scratch_bytes = op_scratch.iter().copied().max().unwrap_or(0);

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
            buffers[buffer.0] = Some(
                alloc_buffer(buffer_lens[buffer.0], buffer_info.ty, init, stream, pool)
                    .with_context(op_context)?,
            );
        }

        let mut mems = MemMap::with_capacity(op.owned.len() + op.inputs.len());
        let owned = disjoint_mut(&mut buffers, op.owned.iter().map(|(b, _)| b.0));
        for (&(buffer, write), mem) in op.owned.iter().zip(owned) {
            let mem = mem.as_mut().expect("buffer used before allocation");
            let mem = if write {
                Mem::Rw(mem.slice_mut(..))
            } else {
                let mem: &DeviceMemory = mem;
                Mem::Ro(mem.slice(..))
            };
            mems.insert(buffer, mem);
        }
        let used_inputs = disjoint_mut(inputs, op.inputs.iter().map(|u| u.input));
        for (input_use, input) in op.inputs.iter().zip(used_inputs) {
            match input {
                Input::Ro(t) => {
                    for &(buffer, part, write) in &input_use.parts {
                        debug_assert!(!write, "read-only input written");
                        mems.insert(buffer, Mem::Ro(ro_storage(t, part)));
                    }
                }
                Input::Rw(t) => mut_storage(t, &input_use.parts, &mut mems),
            }
        }

        let op_scratch = match &mut scratch {
            Some(scratch) if op_scratch[i] > 0 => Some(scratch.slice_mut(..op_scratch[i])),
            _ => None,
        };
        let mut io = bind_io(&placements[i], &op.access, mems, op_scratch, &dims, cx)
            .with_context(op_context)?;
        executor
            .execute(&mut io, stream, cx)
            .with_context(op_context)?;
        drop(io);

        for &buffer in &op.free {
            buffers[buffer.0] = None;
        }
    }

    output_placements
        .into_iter()
        .map(|placement| {
            let mut take = |buffer: BufferId| {
                buffers[buffer.0]
                    .take()
                    .ok_or_else(|| anyhow!("output buffer {buffer:?} was not allocated"))
            };
            Ok(match placement {
                Placement::Plain { buffer, virt, .. } => {
                    GraphOutput::Plain(Tensor::from_parts(virt, take(buffer)?, cx))
                }
                Placement::Mx {
                    values,
                    scales,
                    format,
                    dim,
                    shape,
                    layout,
                } => GraphOutput::Mx(MxTensor::from_parts(
                    Tensor::from_parts(layout.values, take(values)?, cx),
                    Tensor::from_parts(layout.scale_factors, take(scales)?, cx),
                    format,
                    dim,
                    shape,
                )?),
            })
        })
        .collect()
}
