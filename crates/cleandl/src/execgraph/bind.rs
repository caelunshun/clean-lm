//! Binding of symbolic tensors to device memory.

use super::{
    graph::{BufferId, SymTensorData, SymTensorId},
    op::{Access, Bound, BuildCx, GraphInfo, OpIo, TensorRef, TensorRefMut, op_buffer_access},
    sym::{DimValues, SymLayout},
};
use crate::{
    context::Context,
    mx_tensor::{MxDim, MxFormat, MxLayout, MxTensorView, MxTensorViewMut},
    tensor::{TensorView, TensorViewMut, VirtTensorView},
};
use anyhow::{Context as _, anyhow, bail, ensure};
use skycuda::{
    device_memory::DeviceMemory,
    memory_slice::{MemorySlice, MemorySliceMut},
};
use skytens_core::{data_type::ElementType, shape::Shape};
use std::collections::HashMap;

/// Device memory of one graph buffer, borrowed for one operator execution.
pub(crate) enum Mem<'a> {
    Ro(MemorySlice<&'a DeviceMemory>),
    Rw(MemorySliceMut<&'a mut DeviceMemory>),
}

pub(crate) type MemMap<'a> = HashMap<BufferId, Mem<'a>>;

/// A tensor's concrete layout within its buffers, validated to fit.
#[derive(Debug, Clone)]
pub(crate) enum Placement {
    Plain {
        buffer: BufferId,
        virt: VirtTensorView,
        offset: usize,
    },
    Mx {
        values: BufferId,
        scales: BufferId,
        format: MxFormat,
        dim: MxDim,
        shape: Shape,
        layout: MxLayout,
    },
}

/// Concrete layouts of an op's operands and outputs.
#[derive(Debug, Clone)]
pub(crate) struct OpPlacement {
    pub operands: Vec<Placement>,
    pub outputs: Vec<Placement>,
}

fn check_fits(
    virt: &VirtTensorView,
    offset: usize,
    buffer: BufferId,
    buffer_len: &impl Fn(BufferId) -> usize,
) -> anyhow::Result<()> {
    let len = buffer_len(buffer);
    let end = offset
        .checked_add(virt.required_bytes())
        .ok_or_else(|| anyhow!("tensor extent overflows"))?;
    ensure!(
        end <= len,
        "tensor with shape {:?} and strides {:?} at byte offset {offset} needs {end} bytes, but its buffer has {len}",
        virt.shape(),
        virt.strides()
    );
    Ok(())
}

/// Evaluates a plain tensor's layout, returning its view and byte offset into the buffer.
fn eval_plain(
    layout: &SymLayout,
    ty: ElementType,
    dims: &DimValues,
) -> anyhow::Result<(VirtTensorView, usize)> {
    let (shape, strides, offset) = layout.eval(dims)?;
    let virt = VirtTensorView::with_strides(shape, strides, ty);
    if shape.element_count() == 0 {
        return Ok((virt, 0));
    }
    let offset_bits = offset
        .checked_mul(ty.bits_per_element())
        .ok_or_else(|| anyhow!("tensor offset overflows"))?;
    ensure!(
        offset_bits.is_multiple_of(8),
        "tensor offset of {offset} {ty:?} elements is not byte-aligned"
    );
    Ok((virt, offset_bits / 8))
}

pub(crate) fn place(
    tensor: &SymTensorData,
    dims: &DimValues,
    buffer_len: &impl Fn(BufferId) -> usize,
) -> anyhow::Result<Placement> {
    Ok(match tensor {
        SymTensorData::Plain { buffer, layout, ty } => {
            let (virt, offset) = eval_plain(layout, *ty, dims)?;
            check_fits(&virt, offset, *buffer, buffer_len)?;
            Placement::Plain {
                buffer: *buffer,
                virt,
                offset,
            }
        }
        SymTensorData::BlockScaled {
            values,
            scales,
            format,
            dim,
            shape,
        } => {
            let shape = shape.eval(dims)?;
            ensure!(
                shape.element_count() != 0,
                "block-scaled tensors must not be empty, got shape {shape:?}"
            );
            let layout = MxLayout::new(shape, *format, *dim)?;
            check_fits(&layout.values, 0, *values, buffer_len)?;
            check_fits(&layout.scale_factors, 0, *scales, buffer_len)?;
            Placement::Mx {
                values: *values,
                scales: *scales,
                format: *format,
                dim: *dim,
                shape,
                layout,
            }
        }
    })
}

pub(crate) fn place_op(
    graph: &GraphInfo,
    operands: &[SymTensorId],
    outputs: &[SymTensorId],
    dims: &DimValues,
    buffer_len: &impl Fn(BufferId) -> usize,
) -> anyhow::Result<OpPlacement> {
    let place_all = |tensors: &[SymTensorId], what: &str| {
        tensors
            .iter()
            .enumerate()
            .map(|(i, &t)| {
                place(graph.tensor(t), dims, buffer_len).with_context(|| format!("{what} {i}"))
            })
            .collect::<anyhow::Result<Vec<_>>>()
    };
    Ok(OpPlacement {
        operands: place_all(operands, "operand")?,
        outputs: place_all(outputs, "output")?,
    })
}

fn take_ro<'a>(mems: &MemMap<'a>, buffer: BufferId) -> MemorySlice<&'a DeviceMemory> {
    match &mems[&buffer] {
        Mem::Ro(mem) => *mem,
        Mem::Rw(_) => unreachable!("buffer {buffer:?} is both read and written by one op"),
    }
}

fn take_rw<'a>(mems: &mut MemMap<'a>, buffer: BufferId) -> MemorySliceMut<&'a mut DeviceMemory> {
    match mems.remove(&buffer) {
        Some(Mem::Rw(mem)) => mem,
        _ => unreachable!("buffer {buffer:?} is written through multiple operands"),
    }
}

fn view_ro<'a>(
    mem: MemorySlice<&'a DeviceMemory>,
    virt: &VirtTensorView,
    offset: usize,
    cx: &Context,
) -> TensorView<'a> {
    let (start, end) = (mem.offset() + offset, mem.offset() + mem.len());
    TensorView::from_parts(virt.clone(), mem.into_inner().slice(start..end), cx)
}

fn view_rw<'a>(
    mem: MemorySliceMut<&'a mut DeviceMemory>,
    virt: &VirtTensorView,
    offset: usize,
    cx: &Context,
) -> TensorViewMut<'a> {
    let (start, end) = (mem.offset() + offset, mem.offset() + mem.len());
    TensorViewMut::from_parts(virt.clone(), mem.into_inner().slice_mut(start..end), cx)
}

fn bind_ro<'a>(
    placement: &Placement,
    mems: &MemMap<'a>,
    cx: &Context,
) -> anyhow::Result<TensorRef<'a>> {
    Ok(match placement {
        Placement::Plain {
            buffer,
            virt,
            offset,
        } => TensorRef::Plain(view_ro(take_ro(mems, *buffer), virt, *offset, cx)),
        Placement::Mx {
            values,
            scales,
            format,
            dim,
            shape,
            layout,
        } => TensorRef::Mx(MxTensorView::from_parts(
            view_ro(take_ro(mems, *values), &layout.values, 0, cx),
            view_ro(take_ro(mems, *scales), &layout.scale_factors, 0, cx),
            *format,
            *dim,
            *shape,
        )?),
    })
}

fn bind_rw<'a>(
    placement: &Placement,
    mems: &mut MemMap<'a>,
    cx: &Context,
) -> anyhow::Result<TensorRefMut<'a>> {
    Ok(match placement {
        Placement::Plain {
            buffer,
            virt,
            offset,
        } => TensorRefMut::Plain(view_rw(take_rw(mems, *buffer), virt, *offset, cx)),
        Placement::Mx {
            values,
            scales,
            format,
            dim,
            shape,
            layout,
        } => TensorRefMut::Mx(MxTensorViewMut::from_parts(
            view_rw(take_rw(mems, *values), &layout.values, 0, cx),
            view_rw(take_rw(mems, *scales), &layout.scale_factors, 0, cx),
            *format,
            *dim,
            *shape,
        )?),
    })
}

/// Binds an op's operands and outputs to the buffers in `mems`.
pub(crate) fn bind_io<'a>(
    placement: &OpPlacement,
    access: &[Access],
    mut mems: MemMap<'a>,
    scratch: Option<MemorySliceMut<&'a mut DeviceMemory>>,
    dims: &'a DimValues,
    cx: &Context,
) -> anyhow::Result<OpIo<'a>> {
    let inputs = placement
        .operands
        .iter()
        .zip(access)
        .map(|(p, access)| match access {
            Access::ReadOnly => bind_ro(p, &mems, cx).map(Bound::Ro),
            Access::ReadWrite => bind_rw(p, &mut mems, cx).map(Bound::Rw),
        })
        .collect::<anyhow::Result<_>>()?;
    let outputs = placement
        .outputs
        .iter()
        .map(|p| bind_rw(p, &mut mems, cx))
        .collect::<anyhow::Result<_>>()?;
    Ok(OpIo {
        inputs,
        outputs,
        scratch,
        dims,
    })
}

/// Mutable references to the elements at strictly increasing `indices`.
pub(crate) fn disjoint_mut<T>(
    mut slice: &mut [T],
    indices: impl IntoIterator<Item = usize>,
) -> Vec<&mut T> {
    let mut base = 0;
    let mut out = Vec::new();
    for i in indices {
        let rest = std::mem::take(&mut slice);
        let (item, tail) = rest[i - base..]
            .split_first_mut()
            .expect("index out of range");
        out.push(item);
        slice = tail;
        base = i + 1;
    }
    out
}

/// Zeroed, hint-sized buffers for one op, for autotuning.
pub(crate) struct HintBuffers {
    pub dims: DimValues,
    buffers: Vec<(BufferId, bool, DeviceMemory)>,
    placement: OpPlacement,
    scratch: Option<DeviceMemory>,
}

impl HintBuffers {
    /// Must be called with the GPU lock held: zeroing runs on the default compute stream.
    pub fn new(bcx: &BuildCx<'_>) -> anyhow::Result<Self> {
        debug_assert!(bcx.cx.holds_gpu_lock());
        let dims = bcx.graph.hint_values();
        let stream = bcx.cx.default_compute_stream();
        let mut sizes = HashMap::new();
        let buffers = op_buffer_access(bcx.graph, bcx.operands, bcx.operand_access, bcx.outputs)
            .into_iter()
            .map(|(buffer, write)| {
                let bytes = bcx.graph.buffer(buffer).bytes.eval(&dims)?;
                sizes.insert(buffer, bytes);
                Ok((buffer, write, bcx.pool.alloc_zeroed_on(bytes, stream)?))
            })
            .collect::<anyhow::Result<_>>()?;
        let placement = place_op(bcx.graph, bcx.operands, bcx.outputs, &dims, &|b| sizes[&b])
            .context("at hint dimensions")?;
        Ok(Self {
            dims,
            buffers,
            placement,
            scratch: None,
        })
    }

    /// Must be called with the GPU lock held.
    pub fn ensure_scratch(&mut self, bytes: usize, bcx: &BuildCx<'_>) -> anyhow::Result<()> {
        if bytes > 0 && self.scratch.as_ref().is_none_or(|s| s.len() < bytes) {
            self.scratch = Some(
                bcx.pool
                    .alloc_zeroed_on(bytes, bcx.cx.default_compute_stream())?,
            );
        }
        Ok(())
    }

    pub fn io<'a>(
        &'a mut self,
        scratch_bytes: usize,
        bcx: &BuildCx<'_>,
    ) -> anyhow::Result<OpIo<'a>> {
        let mems = self
            .buffers
            .iter_mut()
            .map(|(buffer, write, mem)| {
                let mem = if *write {
                    Mem::Rw(mem.slice_mut(..))
                } else {
                    Mem::Ro(mem.slice(..))
                };
                (*buffer, mem)
            })
            .collect();
        let scratch = match (&mut self.scratch, scratch_bytes) {
            (_, 0) => None,
            (Some(scratch), n) if scratch.len() >= n => Some(scratch.slice_mut(..n)),
            _ => bail!("scratch memory was not allocated"),
        };
        bind_io(
            &self.placement,
            bcx.operand_access,
            mems,
            scratch,
            &self.dims,
            bcx.cx,
        )
    }
}
