use super::{Analysis, AttentionSpec, KvSpec};
use crate::{
    context::{Context, ResourceKey},
    execgraph::{
        BuildCx, Executor, OpIo,
        sym::{DimValues, Monomial, SymLayout, SymShape},
    },
};
use anyhow::{Context as _, anyhow, bail, ensure};
use flash_attention_rs::{
    AttnProblem, BackwardArgs, CompileOpts, CompiledAttention, DType, DeviceInfo, DeviceTensor,
    ForwardArgs, InTensor, KernelParams, OutTensor, SplitPolicy, TensorMeta, TensorStorage,
    select_kernel_params,
};
use serde::Serialize;
use skycuda::stream::Stream;
use skytens_core::data_type::{ElementType, FloatType, SIntType};
use std::sync::Arc;

#[derive(Copy, Clone, Debug)]
pub(super) enum Direction {
    Forward {
        lse: bool,
        fp8: bool,
        split_policy: SplitPolicy,
    },
    Backward,
}

#[derive(Serialize)]
struct ModuleKey {
    module: String,
    #[serde(skip)]
    params: KernelParams,
}

impl ResourceKey for ModuleKey {
    type Resource = CompiledAttention;

    fn create(&self, cx: &Context) -> anyhow::Result<CompiledAttention> {
        tracing::debug!(params = ?self.params, "compiling flash attention kernels");
        flash_attention_rs::compile(&self.params, cx.cuda(), &CompileOpts::default())
            .with_context(|| format!("compiling flash attention kernels for {:?}", self.params))
    }
}

/// Where the optional operands are among the op's operands.
#[derive(Clone, Debug, Default)]
struct Slots {
    cu_seqlens_q: Option<usize>,
    cu_seqlens_k: Option<usize>,
    seqused_q: Option<usize>,
    seqused_k: Option<usize>,
    page_table: Option<usize>,
    descale: Option<[usize; 3]>,
}

pub(super) struct AttentionExecutor {
    compiled: Arc<CompiledAttention>,
    direction: Direction,
    slots: Slots,
    operands: Vec<(SymLayout, DType)>,
    outputs: Vec<(SymShape, DType)>,
    packed_q: bool,
    packed_k: bool,
    max_seqlen_q: Option<Monomial>,
    max_seqlen_k: Option<Monomial>,
    softmax_scale: f32,
    causal: bool,
    window_left: Option<i32>,
    window_right: Option<i32>,
    softcap: f32,
}

fn dtype(ty: ElementType) -> anyhow::Result<DType> {
    Ok(match ty {
        ElementType::Float(FloatType::Bf16) => DType::Bf16,
        ElementType::Float(FloatType::F16) => DType::F16,
        ElementType::Float(FloatType::F32) => DType::F32,
        ElementType::Float(FloatType::F8E4M3) => DType::Fp8E4M3,
        ElementType::SInt(SIntType::S32) => DType::I32,
        ty => bail!("flash attention does not take {ty:?} tensors"),
    })
}

fn to_u32(v: usize, what: &str) -> anyhow::Result<u32> {
    u32::try_from(v).map_err(|_| anyhow!("{what} = {v} exceeds u32::MAX"))
}

/// The call's tensors, by role.
struct Tensors<'a> {
    inputs: Vec<Option<InTensor<'a>>>,
    outputs: Vec<Option<OutTensor<'a>>>,
}

impl<'a> Tensors<'a> {
    fn input(&mut self, i: usize) -> InTensor<'a> {
        self.inputs[i].take().expect("each operand is used once")
    }

    fn opt_input(&mut self, i: Option<usize>) -> Option<InTensor<'a>> {
        i.map(|i| self.input(i))
    }

    fn output(&mut self, i: usize) -> OutTensor<'a> {
        self.outputs[i].take().expect("each output is used once")
    }
}

enum Args<'a> {
    Forward(ForwardArgs<'a>),
    Backward(BackwardArgs<'a>),
}

impl AttentionExecutor {
    pub(super) fn new(
        bcx: &BuildCx<'_>,
        problem: &AttnProblem,
        info: &DeviceInfo,
        spec: &AttentionSpec,
        an: &Analysis,
        direction: Direction,
    ) -> anyhow::Result<Self> {
        let params = select_kernel_params(problem, info)
            .with_context(|| format!("selecting flash attention kernels for {problem:?}"))?;
        tracing::debug!(?problem, ?params, "{}: selected kernels", bcx.op_name);
        let compiled = bcx.cx.get_or_create_resource(ModuleKey {
            module: format!("{:?}", params.module_key()),
            params,
        })?;

        let mut next = match direction {
            Direction::Forward { .. } => 3,
            Direction::Backward => 6,
        };
        let mut slot = |present: Option<_>| {
            present.map(|_| {
                next += 1;
                next - 1
            })
        };
        let [cu_q, cu_k, used_q, used_k, page_table] = spec.aux_operands();
        let mut slots = Slots {
            cu_seqlens_q: slot(cu_q),
            cu_seqlens_k: slot(cu_k),
            seqused_q: slot(used_q),
            seqused_k: slot(used_k),
            page_table: slot(page_table),
            descale: None,
        };
        if let Direction::Forward { fp8: true, .. } = direction {
            slots.descale = Some([next, next + 1, next + 2]);
            next += 3;
        }

        let graph = bcx.graph;
        let operands = bcx
            .operands
            .iter()
            .map(|&id| {
                let ty = graph.ty(id).element_type().context("plain operand")?;
                Ok((
                    graph.layout(id).context("plain operand")?.clone(),
                    dtype(ty)?,
                ))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        ensure!(
            operands.len() == next,
            "{}: unexpected operand count",
            bcx.op_name
        );
        let outputs = bcx
            .outputs
            .iter()
            .map(|&id| {
                let ty = graph.ty(id).element_type().context("plain output")?;
                Ok((graph.shape(id).clone(), dtype(ty)?))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        let (causal, window_left, window_right) = spec.mask.resolve()?;
        let (packed_k, max_seqlen_k) = match &spec.kv {
            KvSpec::Dense => (false, None),
            KvSpec::Packed { max_seqlen, .. } => (true, max_seqlen.clone()),
            KvSpec::Paged { max_seqlen, .. } => (false, max_seqlen.clone()),
        };
        Ok(Self {
            compiled,
            direction,
            slots,
            operands,
            outputs,
            packed_q: spec.cu_seqlens_q.is_some(),
            packed_k,
            max_seqlen_q: spec.max_seqlen_q.clone(),
            max_seqlen_k,
            softmax_scale: spec
                .softmax_scale
                .unwrap_or(1.0 / (an.head_dim as f32).sqrt()),
            causal,
            window_left,
            window_right,
            softcap: spec.softcap,
        })
    }

    /// `max_seqlen_q` and `max_seqlen_k` of a call: the given bounds, or the packed totals.
    fn max_seqlens(
        &self,
        q_rows: usize,
        k_rows: usize,
        dims: &DimValues,
    ) -> anyhow::Result<(Option<u32>, Option<u32>)> {
        let eval = |m: &Option<Monomial>, default: Option<usize>, what: &str| {
            m.as_ref()
                .map(|m| m.eval(dims))
                .transpose()?
                .or(default)
                .map(|v| to_u32(v, what))
                .transpose()
        };
        Ok((
            eval(
                &self.max_seqlen_q,
                self.packed_q.then_some(q_rows),
                "max_seqlen_q",
            )?,
            eval(
                &self.max_seqlen_k,
                self.packed_k.then_some(k_rows),
                "max_seqlen_k",
            )?,
        ))
    }

    fn args<'a>(&self, mut t: Tensors<'a>, dims: &DimValues) -> anyhow::Result<Args<'a>> {
        let s = &self.slots;
        let (q, k, v) = (t.input(0), t.input(1), t.input(2));
        let (max_seqlen_q, max_seqlen_k) =
            self.max_seqlens(q.meta().shape[0], k.meta().shape[0], dims)?;
        let (cu_seqlens_q, cu_seqlens_k) =
            (t.opt_input(s.cu_seqlens_q), t.opt_input(s.cu_seqlens_k));
        let (seqused_q, seqused_k) = (t.opt_input(s.seqused_q), t.opt_input(s.seqused_k));
        Ok(match self.direction {
            Direction::Forward {
                lse, split_policy, ..
            } => {
                let mut args = ForwardArgs::new(q, k, v, t.output(0), self.softmax_scale);
                args.lse = lse.then(|| t.output(1));
                args.cu_seqlens_q = cu_seqlens_q;
                args.cu_seqlens_k = cu_seqlens_k;
                args.seqused_q = seqused_q;
                args.seqused_k = seqused_k;
                args.page_table = t.opt_input(s.page_table);
                if let Some([dq, dk, dv]) = s.descale {
                    args.q_descale = Some(t.input(dq));
                    args.k_descale = Some(t.input(dk));
                    args.v_descale = Some(t.input(dv));
                }
                args.max_seqlen_q = max_seqlen_q;
                args.max_seqlen_k = max_seqlen_k;
                args.causal = self.causal;
                args.window_left = self.window_left;
                args.window_right = self.window_right;
                args.softcap = self.softcap;
                args.split_policy = split_policy;
                Args::Forward(args)
            }
            Direction::Backward => Args::Backward(BackwardArgs {
                q,
                k,
                v,
                o: t.input(3),
                do_: t.input(4),
                lse: t.input(5),
                dq: t.output(0),
                dk: t.output(1),
                dv: t.output(2),
                cu_seqlens_q,
                cu_seqlens_k,
                seqused_q,
                seqused_k,
                max_seqlen_q,
                max_seqlen_k,
                softmax_scale: self.softmax_scale,
                causal: self.causal,
                window_left: self.window_left,
                window_right: self.window_right,
                softcap: self.softcap,
            }),
        })
    }

    fn scratch_bytes_of(&self, args: &Args<'_>) -> anyhow::Result<usize> {
        Ok(match args {
            Args::Forward(args) => self.compiled.forward_scratch_bytes(args)?,
            Args::Backward(args) => self.compiled.backward_scratch_bytes(args)?,
        })
    }

    fn try_scratch_bytes(&self, dims: &DimValues) -> anyhow::Result<usize> {
        let meta = |shape: Vec<usize>, strides: Vec<usize>, dtype| TensorMeta {
            dtype,
            shape,
            strides,
            // Alignment does not change the scratch size.
            base_offset: 0,
        };
        let inputs = self
            .operands
            .iter()
            .map(|(layout, dtype)| {
                let (shape, strides, _) = layout.eval(dims)?;
                let m = meta(
                    shape.dims_slice().to_vec(),
                    strides.dims_slice().to_vec(),
                    *dtype,
                );
                Ok(Some(InTensor::meta_only(m)))
            })
            .collect::<anyhow::Result<_>>()?;
        let outputs = self
            .outputs
            .iter()
            .map(|(shape, dtype)| {
                let shape = shape.eval(dims)?.dims_slice().to_vec();
                let mut strides = vec![1; shape.len()];
                for i in (0..shape.len().saturating_sub(1)).rev() {
                    strides[i] = strides[i + 1] * shape[i + 1];
                }
                Ok(Some(OutTensor::meta_only(meta(shape, strides, *dtype))))
            })
            .collect::<anyhow::Result<_>>()?;
        let args = self.args(Tensors { inputs, outputs }, dims)?;
        self.scratch_bytes_of(&args)
    }
}

fn device_tensor<M: TensorStorage>(
    memory: M,
    ty: ElementType,
    shape: &[usize],
    strides: &[usize],
) -> anyhow::Result<DeviceTensor<M>> {
    Ok(DeviceTensor::new(memory, dtype(ty)?, shape, strides)?)
}

impl Executor for AttentionExecutor {
    fn scratch_bytes(&self, dims: &DimValues) -> usize {
        self.try_scratch_bytes(dims).unwrap_or_else(|e| {
            // The same error is reported by `execute`.
            tracing::debug!("attention: sizing scratch failed: {e:#}");
            0
        })
    }

    fn execute(&self, io: &mut OpIo<'_>, stream: &Stream, cx: &Context) -> anyhow::Result<()> {
        let dims = io.dims;
        let (inputs, outputs, scratch) = io.split();
        let inputs = inputs
            .iter()
            .map(|bound| {
                let t = bound.ro().plain();
                device_tensor(
                    *t.storage(),
                    t.element_type(),
                    t.shape().dims_slice(),
                    t.strides().dims_slice(),
                )
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let mut outputs = outputs
            .iter_mut()
            .map(|out| {
                let t = out.plain();
                let (ty, shape, strides) = (t.element_type(), t.shape(), t.strides());
                device_tensor(
                    t.storage_mut(),
                    ty,
                    shape.dims_slice(),
                    strides.dims_slice(),
                )
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let tensors = Tensors {
            inputs: inputs.iter().map(|t| Some(InTensor::from(t))).collect(),
            outputs: outputs
                .iter_mut()
                .map(|t| Some(OutTensor::from(t)))
                .collect(),
        };
        let mut args = self.args(tensors, dims)?;
        let pool = cx.device_memory_pool();
        match (&mut args, scratch) {
            (Args::Forward(args), Some(scratch)) => self
                .compiled
                .run_forward_with_scratch(args, stream, scratch)?,
            (Args::Backward(args), Some(scratch)) => self
                .compiled
                .run_backward_with_scratch(args, stream, scratch)?,
            (args, None) => {
                let bytes = self.scratch_bytes_of(args)?;
                ensure!(
                    bytes == 0,
                    "attention: the call needs {bytes} bytes of scratch, but none was given"
                );
                // Without scratch buffers, nothing is allocated from the pool.
                match args {
                    Args::Forward(args) => self.compiled.run_forward(args, stream, pool)?,
                    Args::Backward(args) => self.compiled.run_backward(args, stream, pool)?,
                }
            }
        }
        Ok(())
    }
}
