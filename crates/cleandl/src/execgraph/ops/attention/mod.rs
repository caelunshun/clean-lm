//! Forward and backward flash attention using `flash-attention-rs`.

mod exec;

use crate::execgraph::{
    BuildCx, Executor, GraphInfo, Init, Operand, Operator, OutputSpec, SymTensor, SymTensorId,
    TensorTy,
    sym::{Monomial, SymLayout, SymShape},
};
use anyhow::{Context as _, bail, ensure};
use flash_attention_rs::{AttnDtype, AttnProblem, DeviceInfo, OutDtype, SplitPolicy};
use serde::{Deserialize, Serialize};
use skytens_core::data_type::{ElementType, FloatType, SIntType};

const F32: ElementType = ElementType::Float(FloatType::F32);
const BF16: ElementType = ElementType::Float(FloatType::Bf16);
const F16: ElementType = ElementType::Float(FloatType::F16);
const F8E4M3: ElementType = ElementType::Float(FloatType::F8E4M3);
const I32: ElementType = ElementType::SInt(SIntType::S32);

const SUPPORTED_HEAD_DIMS: [usize; 4] = [64, 96, 128, 256];
const TMA_ALIGN: usize = 16;

/// NB Masks are aligned to the bottom-right. With `sq` queries
/// and `sk` keys, query `i` is at key position `i + sk - sq`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum AttentionMask {
    /// Dense attention
    #[default]
    None,
    /// Keys at or before the query's position.
    Causal,
    /// Keys from `left` before to `right` after the query's position; `None` is unbounded.
    SlidingWindow {
        left: Option<u32>,
        right: Option<u32>,
    },
}

impl AttentionMask {
    fn resolve(self) -> anyhow::Result<(bool, Option<i32>, Option<i32>)> {
        Ok(match self {
            AttentionMask::None => (false, None, None),
            AttentionMask::Causal => (true, None, None),
            AttentionMask::SlidingWindow { left, right } => {
                let to_i32 = |w: Option<u32>| {
                    w.map(|w| i32::try_from(w).context("window size exceeds i32::MAX"))
                        .transpose()
                };
                (false, to_i32(left)?, to_i32(right)?)
            }
        })
    }
}

#[derive(Clone, Debug, Default)]
pub enum QueryLayout<'g> {
    /// `[batch, seqlen_q, heads, head_dim]`.
    #[default]
    Dense,
    /// `[total_q, heads, head_dim]`; sequence `b` is rows `cu_seqlens[b]..cu_seqlens[b + 1]`.
    Packed {
        /// i32 `[batch + 1]`.
        cu_seqlens: SymTensor<'g>,
        /// Bound on every sequence's length; `None` is `total_q`. Tighter bounds launch fewer
        /// blocks. Its dimensions must appear in some tensor shape of the graph.
        max_seqlen: Option<Monomial>,
    },
}

/// Layout of K and V (and of their gradients).
#[derive(Clone, Debug, Default)]
pub enum KvLayout<'g> {
    /// `[batch, seqlen_k, kv_heads, head_dim]`.
    #[default]
    Dense,
    /// `[total_k, kv_heads, head_dim]`, as [`QueryLayout::Packed`].
    Packed {
        cu_seqlens: SymTensor<'g>,
        /// `None` is `total_k`.
        max_seqlen: Option<Monomial>,
    },
    /// `[pages, page_size, kv_heads, head_dim]`. Forward only. Sequence `b` reads its keys from
    /// pages `page_table[b, ..]`.
    Paged {
        /// i32 `[batch, max_pages]`.
        page_table: SymTensor<'g>,
        /// Keys of every sequence without `seqused_k`, a bound on `seqused_k` with it; `None` is
        /// `max_pages * page_size`.
        max_seqlen: Option<Monomial>,
    },
}

/// Options shared by [`SymTensor::attention_fwd`] and [`SymTensor::attention_bwd`].
#[derive(Clone, Debug, Default)]
pub struct AttentionOpts<'g> {
    /// Scale of `q · k`; `None` is `1 / sqrt(head_dim)`.
    pub softmax_scale: Option<f32>,
    pub mask: AttentionMask,
    /// Scores become `softcap * tanh(scores / softcap)`; 0 disables it.
    pub softcap: f32,
    pub q_layout: QueryLayout<'g>,
    pub kv_layout: KvLayout<'g>,
    /// i32 `[batch]`: sequence `b` has only its first `seqused_q[b]` queries. Outputs and
    /// gradients of the other rows are zero (and their LSE `-inf`).
    pub seqused_q: Option<SymTensor<'g>>,
    /// i32 `[batch]`: sequence `b` has only its first `seqused_k[b]` keys.
    pub seqused_k: Option<SymTensor<'g>>,
}

/// fp8 descale factors, f32 `[batch, kv_heads]`: the inputs' values are multiplied by these.
#[derive(Copy, Clone, Debug)]
pub struct Fp8Descale<'g> {
    pub q: SymTensor<'g>,
    pub k: SymTensor<'g>,
    pub v: SymTensor<'g>,
}

/// Forward-only options.
#[derive(Copy, Clone, Debug, Default)]
pub struct AttentionFwdOpts<'g> {
    /// Required for fp8 e4m3 inputs.
    pub fp8_descale: Option<Fp8Descale<'g>>,
    /// Only supports bf16 or f16. Required for fp8 inputs. Uses
    /// the input type for bf16 and f16 inputs.
    pub out_type: Option<ElementType>,
    /// Also output the log-sum-exp of the scaled scores (natural log) for bwd
    pub lse: bool,
    /// KV splits, None for heuristic
    pub num_splits: Option<u32>,
}

#[derive(Copy, Clone, Debug)]
pub struct AttentionFwdOutputs<'g> {
    pub out: SymTensor<'g>,
    /// f32 `[batch, heads, seqlen_q]`, or `[heads, total_q]` for packed queries.
    pub lse: Option<SymTensor<'g>>,
}

#[derive(Copy, Clone, Debug)]
pub struct AttentionGrads<'g> {
    pub dq: SymTensor<'g>,
    pub dk: SymTensor<'g>,
    pub dv: SymTensor<'g>,
}

#[derive(Clone, Debug)]
pub struct AttentionSpec {
    pub softmax_scale: Option<f32>,
    pub mask: AttentionMask,
    pub softcap: f32,
    pub cu_seqlens_q: Option<SymTensorId>,
    pub max_seqlen_q: Option<Monomial>,
    pub kv: KvSpec,
    pub seqused_q: Option<SymTensorId>,
    pub seqused_k: Option<SymTensorId>,
}

#[derive(Clone, Debug)]
pub enum KvSpec {
    Dense,
    Packed {
        cu_seqlens: SymTensorId,
        max_seqlen: Option<Monomial>,
    },
    Paged {
        page_table: SymTensorId,
        max_seqlen: Option<Monomial>,
    },
}

impl AttentionSpec {
    fn new(opts: &AttentionOpts<'_>) -> Self {
        let (cu_seqlens_q, max_seqlen_q) = match &opts.q_layout {
            QueryLayout::Dense => (None, None),
            QueryLayout::Packed {
                cu_seqlens,
                max_seqlen,
            } => (Some(cu_seqlens.id()), max_seqlen.clone()),
        };
        let kv = match &opts.kv_layout {
            KvLayout::Dense => KvSpec::Dense,
            KvLayout::Packed {
                cu_seqlens,
                max_seqlen,
            } => KvSpec::Packed {
                cu_seqlens: cu_seqlens.id(),
                max_seqlen: max_seqlen.clone(),
            },
            KvLayout::Paged {
                page_table,
                max_seqlen,
            } => KvSpec::Paged {
                page_table: page_table.id(),
                max_seqlen: max_seqlen.clone(),
            },
        };
        Self {
            softmax_scale: opts.softmax_scale,
            mask: opts.mask,
            softcap: opts.softcap,
            cu_seqlens_q,
            max_seqlen_q,
            kv,
            seqused_q: opts.seqused_q.map(SymTensor::id),
            seqused_k: opts.seqused_k.map(SymTensor::id),
        }
    }

    fn cu_seqlens_k(&self) -> Option<SymTensorId> {
        match self.kv {
            KvSpec::Packed { cu_seqlens, .. } => Some(cu_seqlens),
            _ => None,
        }
    }

    fn page_table(&self) -> Option<SymTensorId> {
        match self.kv {
            KvSpec::Paged { page_table, .. } => Some(page_table),
            _ => None,
        }
    }

    fn max_seqlen_k(&self) -> Option<&Monomial> {
        match &self.kv {
            KvSpec::Dense => None,
            KvSpec::Packed { max_seqlen, .. } | KvSpec::Paged { max_seqlen, .. } => {
                max_seqlen.as_ref()
            }
        }
    }

    fn is_varlen(&self) -> bool {
        self.cu_seqlens_q.is_some()
            || self.cu_seqlens_k().is_some()
            || self.seqused_q.is_some()
            || self.seqused_k.is_some()
    }

    /// Auxiliary operands in operand order: cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k,
    /// page_table.
    fn aux_operands(&self) -> [Option<SymTensorId>; 5] {
        [
            self.cu_seqlens_q,
            self.cu_seqlens_k(),
            self.seqused_q,
            self.seqused_k,
            self.page_table(),
        ]
    }
}

fn same_graph<'g>(a: SymTensor<'g>, others: impl IntoIterator<Item = SymTensor<'g>>) -> bool {
    others
        .into_iter()
        .all(|t| std::ptr::eq(a.builder(), t.builder()))
}

fn opts_tensors<'g>(opts: &AttentionOpts<'g>) -> Vec<SymTensor<'g>> {
    let mut tensors: Vec<SymTensor<'g>> = [opts.seqused_q, opts.seqused_k]
        .into_iter()
        .flatten()
        .collect();
    if let QueryLayout::Packed { cu_seqlens, .. } = &opts.q_layout {
        tensors.push(*cu_seqlens);
    }
    match &opts.kv_layout {
        KvLayout::Dense => {}
        KvLayout::Packed { cu_seqlens: t, .. } | KvLayout::Paged { page_table: t, .. } => {
            tensors.push(*t)
        }
    }
    tensors
}

impl<'g> SymTensor<'g> {
    /// Attention of this Q over `k` and `v`; see the [module docs](self).
    pub fn attention_fwd(
        self,
        k: SymTensor<'g>,
        v: SymTensor<'g>,
        opts: &AttentionOpts<'g>,
        fwd: AttentionFwdOpts<'g>,
    ) -> anyhow::Result<AttentionFwdOutputs<'g>> {
        let descale = fwd.fp8_descale.map(|d| [d.q, d.k, d.v]);
        ensure!(
            same_graph(
                self,
                [k, v]
                    .into_iter()
                    .chain(opts_tensors(opts))
                    .chain(descale.into_iter().flatten())
            ),
            "attention_fwd operands belong to different graphs"
        );
        let op = AttentionFwd {
            q: self.id(),
            k: k.id(),
            v: v.id(),
            spec: AttentionSpec::new(opts),
            descale: descale.map(|d| d.map(SymTensor::id)),
            out_type: fwd.out_type,
            lse: fwd.lse,
            num_splits: fwd.num_splits,
        };
        let mut outputs = self.builder().add_op(op)?.into_iter();
        Ok(AttentionFwdOutputs {
            out: outputs.next().unwrap(),
            lse: outputs.next(),
        })
    }

    /// Gradients of [`Self::attention_fwd`] of this Q, given its output `out` and `lse`, and
    /// the gradient `d_out` of the output. `opts` must be those of the forward. Gradients have
    /// the types of the inputs, which must be 16-bit.
    pub fn attention_bwd(
        self,
        k: SymTensor<'g>,
        v: SymTensor<'g>,
        out: SymTensor<'g>,
        lse: SymTensor<'g>,
        d_out: SymTensor<'g>,
        opts: &AttentionOpts<'g>,
    ) -> anyhow::Result<AttentionGrads<'g>> {
        ensure!(
            same_graph(
                self,
                [k, v, out, lse, d_out]
                    .into_iter()
                    .chain(opts_tensors(opts))
            ),
            "attention_bwd operands belong to different graphs"
        );
        let op = AttentionBwd {
            q: self.id(),
            k: k.id(),
            v: v.id(),
            out: out.id(),
            lse: lse.id(),
            d_out: d_out.id(),
            spec: AttentionSpec::new(opts),
        };
        let mut outputs = self.builder().add_op(op)?.into_iter();
        Ok(AttentionGrads {
            dq: outputs.next().unwrap(),
            dk: outputs.next().unwrap(),
            dv: outputs.next().unwrap(),
        })
    }
}

#[derive(Clone, Debug)]
pub struct AttentionFwd {
    pub q: SymTensorId,
    pub k: SymTensorId,
    pub v: SymTensorId,
    pub spec: AttentionSpec,
    /// Q, K and V descales.
    pub descale: Option<[SymTensorId; 3]>,
    pub out_type: Option<ElementType>,
    pub lse: bool,
    pub num_splits: Option<u32>,
}

#[derive(Clone, Debug)]
pub struct AttentionBwd {
    pub q: SymTensorId,
    pub k: SymTensorId,
    pub v: SymTensorId,
    pub out: SymTensorId,
    pub lse: SymTensorId,
    pub d_out: SymTensorId,
    pub spec: AttentionSpec,
}

/// Everything derived from the graph that both directions need.
#[derive(Clone, Debug)]
struct Analysis {
    dtype: AttnDtype,
    in_type: ElementType,
    out_type: ElementType,
    head_dim: usize,
    heads_q: usize,
    heads_kv: usize,
    q_shape: SymShape,
    k_shape: SymShape,
    lse_shape: SymShape,
    /// Selection hints.
    batch_hint: usize,
    max_seqlen_q_hint: usize,
    max_seqlen_k_hint: usize,
    total_q_hint: usize,
    total_k_hint: usize,
    page_size: Option<usize>,
}

fn attn_dtype(ty: TensorTy) -> anyhow::Result<AttnDtype> {
    Ok(match ty {
        TensorTy::Plain(BF16) => AttnDtype::Bf16,
        TensorTy::Plain(F16) => AttnDtype::F16,
        TensorTy::Plain(F8E4M3) => AttnDtype::Fp8E4M3,
        ty => bail!("attention inputs must be bf16, f16 or f8e4m3, got {ty:?}"),
    })
}

fn out_dtype(ty: ElementType) -> anyhow::Result<OutDtype> {
    Ok(match ty {
        BF16 => OutDtype::Bf16,
        F16 => OutDtype::F16,
        ty => bail!("attention outputs must be bf16 or f16, got {ty:?}"),
    })
}

fn fixed(m: &Monomial, what: &str, graph: &GraphInfo) -> anyhow::Result<usize> {
    m.as_fixed().with_context(|| {
        format!(
            "{what} must be a fixed size, got {}",
            m.display(graph.dims())
        )
    })
}

/// Checks a Q/K/V-like tensor: plain, the given rank, contiguous along the head dimension and
/// with 16-byte aligned outer strides and offset where provable.
fn check_head_tensor(
    graph: &GraphInfo,
    id: SymTensorId,
    what: &str,
    rank: usize,
) -> anyhow::Result<()> {
    let dims = graph.dims();
    let ty = graph.ty(id);
    let TensorTy::Plain(elem) = ty else {
        bail!("{what} must be a plain tensor, got {ty:?}");
    };
    let layout = graph.layout(id).expect("plain tensor");
    ensure!(
        layout.rank() == rank,
        "{what} must have rank {rank}, got shape {}",
        layout.shape.display(dims)
    );
    ensure!(
        layout.strides[rank - 1].is_one(),
        "{what} must be contiguous along the head dimension, got strides [{}]",
        display_strides(layout, graph)
    );
    ensure!(
        layout.offset_is_aligned(TMA_ALIGN, elem, dims),
        "{what} must start at a {TMA_ALIGN}-byte aligned offset"
    );
    for dim in 0..rank - 1 {
        ensure!(
            layout.shape[dim].is_one() || layout.stride_is_aligned(dim, TMA_ALIGN, elem, dims),
            "{what} stride {} of dimension {dim} is not provably a multiple of {TMA_ALIGN} bytes",
            layout.strides[dim].display(dims)
        );
    }
    Ok(())
}

fn display_strides(layout: &SymLayout, graph: &GraphInfo) -> String {
    layout
        .strides
        .iter()
        .map(|s| s.display(graph.dims()).to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

/// Checks an element-wise read tensor (index tensors, descales, LSE): its type, rank and a
/// contiguous last dimension. Returns its shape.
fn check_aux<'a>(
    graph: &'a GraphInfo,
    id: SymTensorId,
    what: &str,
    ty: ElementType,
    rank: usize,
) -> anyhow::Result<&'a SymShape> {
    let dims = graph.dims();
    ensure!(
        graph.ty(id) == TensorTy::Plain(ty),
        "{what} must be {ty:?}, got {:?}",
        graph.ty(id)
    );
    let layout = graph.layout(id).expect("plain tensor");
    ensure!(
        layout.rank() == rank,
        "{what} must have rank {rank}, got shape {}",
        layout.shape.display(dims)
    );
    ensure!(
        layout.strides[rank - 1].is_one() || layout.shape[rank - 1].is_one(),
        "{what} must be contiguous in its last dimension, got strides [{}]",
        display_strides(layout, graph)
    );
    Ok(&layout.shape)
}

fn same_dim(
    a: &Monomial,
    b: &Monomial,
    what: impl FnOnce() -> String,
    graph: &GraphInfo,
) -> anyhow::Result<()> {
    ensure!(
        a == b,
        "{}: {} differs from {}",
        what(),
        a.display(graph.dims()),
        b.display(graph.dims())
    );
    Ok(())
}

/// The dimensions of `m` must be bound at execution, i.e. appear in some tensor shape.
fn check_bound(m: &Monomial, what: &str, graph: &GraphInfo) -> anyhow::Result<()> {
    for &atom in m.atoms() {
        ensure!(
            graph.tensors.iter().any(|t| t
                .shape()
                .dims()
                .iter()
                .any(|d| d.atoms().contains(&atom))),
            "{what} uses dimension {}, which appears in no tensor shape and so cannot be inferred at execution",
            graph.dims().name(atom)
        );
    }
    Ok(())
}

impl AttentionSpec {
    fn analyze(
        &self,
        graph: &GraphInfo,
        [q, k, v]: [SymTensorId; 3],
        out_type: Option<ElementType>,
    ) -> anyhow::Result<Analysis> {
        let dims = graph.dims();
        let in_ty = graph.ty(q);
        let dtype = attn_dtype(in_ty)?;
        for (what, id) in [("K", k), ("V", v)] {
            ensure!(
                graph.ty(id) == in_ty,
                "{what} has type {:?}, but Q has {in_ty:?}",
                graph.ty(id)
            );
        }
        let in_type = in_ty.element_type().unwrap();
        let out_type = match (dtype.is_fp8(), out_type) {
            (true, None) => bail!("fp8 attention needs an output type"),
            (true, Some(ty)) => ty,
            (false, None) => in_type,
            (false, Some(ty)) => {
                ensure!(
                    ty == in_type,
                    "the output type of {in_type:?} attention must be {in_type:?}, got {ty:?}"
                );
                ty
            }
        };
        out_dtype(out_type)?;

        let packed_q = self.cu_seqlens_q.is_some();
        let paged = self.page_table().is_some();
        let packed_k = self.cu_seqlens_k().is_some();
        check_head_tensor(graph, q, "Q", if packed_q { 3 } else { 4 })?;
        for (what, id) in [("K", k), ("V", v)] {
            check_head_tensor(graph, id, what, if packed_k { 3 } else { 4 })?;
        }
        let (q_shape, k_shape) = (graph.shape(q).clone(), graph.shape(k).clone());
        ensure!(
            graph.shape(v) == &k_shape,
            "V has shape {}, but K has {}",
            graph.shape(v).display(dims),
            k_shape.display(dims)
        );
        let qr = q_shape.rank();
        let kr = k_shape.rank();
        let head_dim = fixed(&q_shape[qr - 1], "the head dimension", graph)?;
        ensure!(
            SUPPORTED_HEAD_DIMS.contains(&head_dim),
            "the head dimension must be one of {SUPPORTED_HEAD_DIMS:?}, got {head_dim}"
        );
        same_dim(
            &k_shape[kr - 1],
            &q_shape[qr - 1],
            || "head dimension of K".into(),
            graph,
        )?;
        let heads_q = fixed(&q_shape[qr - 2], "the number of query heads", graph)?;
        let heads_kv = fixed(&k_shape[kr - 2], "the number of KV heads", graph)?;
        ensure!(
            heads_kv > 0 && heads_q % heads_kv == 0,
            "the {heads_q} query heads must be a multiple of the {heads_kv} KV heads"
        );

        // Batch-sized dimensions that must agree, where they can be compared symbolically.
        let mut batch: Option<(Monomial, String)> = None;
        let mut batch_hint: Option<usize> = None;
        let mut set_batch = |m: &Monomial, what: &str| -> anyhow::Result<()> {
            match &batch {
                Some((b, from)) => same_dim(m, b, || format!("batch of {what} and {from}"), graph),
                None => {
                    batch_hint = Some(m.eval_hint(dims));
                    batch = Some((m.clone(), what.to_owned()));
                    Ok(())
                }
            }
        };
        if !packed_q {
            set_batch(&q_shape[0], "Q")?;
        }
        if !packed_k && !paged {
            set_batch(&k_shape[0], "K")?;
        }
        for (what, id) in [("seqused_q", self.seqused_q), ("seqused_k", self.seqused_k)] {
            if let Some(id) = id {
                set_batch(&check_aux(graph, id, what, I32, 1)?[0], what)?;
            }
        }
        let mut page_size = None;
        let mut max_pages_hint = 0;
        if let Some(id) = self.page_table() {
            let shape = check_aux(graph, id, "page_table", I32, 2)?;
            set_batch(&shape[0], "page_table")?;
            max_pages_hint = shape[1].eval_hint(dims);
            page_size = Some(fixed(&k_shape[1], "the page size", graph)?);
        }
        let mut cu_batch_hint = None;
        for (what, id) in [
            ("cu_seqlens_q", self.cu_seqlens_q),
            ("cu_seqlens_k", self.cu_seqlens_k()),
        ] {
            if let Some(id) = id {
                let entries = check_aux(graph, id, what, I32, 1)?[0].eval_hint(dims);
                ensure!(entries > 0, "{what} must have batch + 1 entries");
                cu_batch_hint = Some(entries - 1);
            }
        }
        let batch_hint = batch_hint.or(cu_batch_hint).unwrap_or(1);

        for (what, m) in [
            ("max_seqlen_q", self.max_seqlen_q.as_ref()),
            ("max_seqlen_k", self.max_seqlen_k()),
        ] {
            if let Some(m) = m {
                check_bound(m, what, graph)?;
            }
        }
        let (total_q_hint, max_seqlen_q_hint) = if packed_q {
            let total = q_shape[0].eval_hint(dims);
            let max = self
                .max_seqlen_q
                .as_ref()
                .map_or(total, |m| m.eval_hint(dims));
            (total, max)
        } else {
            let s = q_shape[1].eval_hint(dims);
            (batch_hint * s, s)
        };
        let (total_k_hint, max_seqlen_k_hint) = if packed_k {
            let total = k_shape[0].eval_hint(dims);
            let max = self.max_seqlen_k().map_or(total, |m| m.eval_hint(dims));
            (total, max)
        } else if let Some(ps) = page_size {
            let capacity = max_pages_hint * ps;
            let max = self.max_seqlen_k().map_or(capacity, |m| m.eval_hint(dims));
            (k_shape[0].eval_hint(dims) * ps, max)
        } else {
            let s = k_shape[1].eval_hint(dims);
            (batch_hint * s, s)
        };

        let lse_shape: SymShape = if packed_q {
            [Monomial::fixed(heads_q), q_shape[0].clone()]
                .into_iter()
                .collect()
        } else {
            [
                q_shape[0].clone(),
                Monomial::fixed(heads_q),
                q_shape[1].clone(),
            ]
            .into_iter()
            .collect()
        };
        self.mask.resolve()?;
        ensure!(
            self.softcap.is_finite() && self.softcap >= 0.0,
            "softcap must be finite and non-negative, got {}",
            self.softcap
        );
        if let Some(scale) = self.softmax_scale {
            ensure!(
                scale.is_finite() && scale > 0.0,
                "softmax_scale must be finite and positive, got {scale}"
            );
        }
        Ok(Analysis {
            dtype,
            in_type,
            out_type,
            head_dim,
            heads_q,
            heads_kv,
            q_shape,
            k_shape,
            lse_shape,
            batch_hint,
            max_seqlen_q_hint,
            max_seqlen_k_hint,
            total_q_hint,
            total_k_hint,
            page_size,
        })
    }

    /// The flash-attention problem at the graph's hints.
    fn problem(&self, an: &Analysis, info: &DeviceInfo) -> anyhow::Result<AttnProblem> {
        let arch = info.arch_family().with_context(|| {
            format!(
                "flash attention does not support compute capability {}.{}",
                info.cc_major, info.cc_minor
            )
        })?;
        let hint = |v: usize, what: &str| u32::try_from(v).with_context(|| format!("{what} hint"));
        let mut problem = AttnProblem::forward(
            arch,
            an.dtype,
            an.head_dim as u32,
            an.heads_q as u32,
            an.heads_kv as u32,
            hint(an.batch_hint, "batch")?.max(1),
            hint(an.max_seqlen_q_hint, "max_seqlen_q")?.max(1),
            hint(an.max_seqlen_k_hint, "max_seqlen_k")?.max(1),
        );
        if an.dtype.is_fp8() {
            problem.out_dtype = Some(out_dtype(an.out_type)?);
        }
        let (causal, left, right) = self.mask.resolve()?;
        problem.causal = causal;
        problem.window_left = left;
        problem.window_right = right;
        problem.softcap = self.softcap != 0.0;
        problem.cu_seqlens_q = self.cu_seqlens_q.is_some();
        problem.cu_seqlens_k = self.cu_seqlens_k().is_some();
        problem.seqused_q = self.seqused_q.is_some();
        problem.seqused_k = self.seqused_k.is_some();
        problem.page_size = an.page_size.map(|p| p as u32);
        // Packed totals, consistent with the other hints.
        if problem.cu_seqlens_q {
            let total = an.total_q_hint.min(an.batch_hint * an.max_seqlen_q_hint);
            problem.total_q = Some(hint(total, "total_q")?.max(1));
        }
        if problem.cu_seqlens_k {
            let total = an.total_k_hint.min(an.batch_hint * an.max_seqlen_k_hint);
            problem.total_k = Some(hint(total, "total_k")?.max(1));
        }
        Ok(problem)
    }
}

fn split_policy(num_splits: Option<u32>) -> anyhow::Result<SplitPolicy> {
    Ok(match num_splits {
        None => SplitPolicy::Heuristic,
        Some(0) => bail!("num_splits must be positive"),
        Some(1) => SplitPolicy::Disabled,
        Some(n) => SplitPolicy::Fixed(n),
    })
}

impl AttentionFwd {
    fn analyze(&self, graph: &GraphInfo) -> anyhow::Result<Analysis> {
        let an = self
            .spec
            .analyze(graph, [self.q, self.k, self.v], self.out_type)?;
        match (an.dtype.is_fp8(), self.descale) {
            (true, None) => bail!("fp8 attention needs descale factors"),
            (false, Some(_)) => bail!("descale factors are only for fp8 inputs"),
            (true, Some(descale)) => {
                for (what, id) in ["q_descale", "k_descale", "v_descale"].iter().zip(descale) {
                    let shape = check_aux(graph, id, what, F32, 2)?;
                    same_dim(
                        &shape[1],
                        &Monomial::fixed(an.heads_kv),
                        || format!("KV heads of {what}"),
                        graph,
                    )?;
                }
            }
            (false, None) => {}
        }
        split_policy(self.num_splits)?;
        Ok(an)
    }

    fn output_init(&self) -> (Init, Init) {
        // Rows past the sequences are not written.
        if self.spec.cu_seqlens_q.is_some() || self.spec.seqused_q.is_some() {
            (Init::Zero, Init::Fill32(f32::NEG_INFINITY.to_bits()))
        } else {
            (Init::Uninit, Init::Uninit)
        }
    }
}

impl Operator for AttentionFwd {
    fn name(&self) -> &str {
        "attention_fwd"
    }

    fn operands(&self) -> Vec<Operand> {
        [self.q, self.k, self.v]
            .into_iter()
            .chain(self.spec.aux_operands().into_iter().flatten())
            .chain(self.descale.into_iter().flatten())
            .map(Operand::read)
            .collect()
    }

    fn infer_outputs(&self, graph: &GraphInfo) -> anyhow::Result<Vec<OutputSpec>> {
        let an = self.analyze(graph)?;
        let (out_init, lse_init) = self.output_init();
        let mut outputs = vec![OutputSpec::new(an.out_type, an.q_shape, out_init)];
        if self.lse {
            outputs.push(OutputSpec::new(F32, an.lse_shape, lse_init));
        }
        Ok(outputs)
    }

    fn build(&self, bcx: &BuildCx<'_>) -> anyhow::Result<Box<dyn Executor>> {
        let an = self.analyze(bcx.graph)?;
        let info = DeviceInfo::from_device(bcx.cx.cuda());
        let mut problem = self.spec.problem(&an, &info)?;
        problem.lse = self.lse;
        problem.split_policy = split_policy(self.num_splits)?;
        Ok(Box::new(exec::AttentionExecutor::new(
            bcx,
            &problem,
            &info,
            &self.spec,
            &an,
            exec::Direction::Forward {
                lse: self.lse,
                fp8: self.descale.is_some(),
                split_policy: problem.split_policy,
            },
        )?))
    }
}

impl AttentionBwd {
    fn analyze(&self, graph: &GraphInfo) -> anyhow::Result<Analysis> {
        let dims = graph.dims();
        ensure!(
            self.spec.page_table().is_none(),
            "the attention backward does not support paged K and V"
        );
        let an = self.spec.analyze(graph, [self.q, self.k, self.v], None)?;
        ensure!(
            !an.dtype.is_fp8(),
            "the attention backward does not support fp8 inputs"
        );
        let rank = an.q_shape.rank();
        for (what, id) in [
            ("the output", self.out),
            ("the output gradient", self.d_out),
        ] {
            ensure!(
                graph.ty(id) == TensorTy::Plain(an.in_type),
                "{what} must be {:?} like Q, got {:?}",
                an.in_type,
                graph.ty(id)
            );
            check_head_tensor(graph, id, what, rank)?;
            ensure!(
                graph.shape(id) == &an.q_shape,
                "{what} has shape {}, but Q has {}",
                graph.shape(id).display(dims),
                an.q_shape.display(dims)
            );
        }
        let lse = check_aux(graph, self.lse, "LSE", F32, an.lse_shape.rank())?;
        ensure!(
            lse == &an.lse_shape,
            "LSE has shape {}, expected {}",
            lse.display(dims),
            an.lse_shape.display(dims)
        );
        Ok(an)
    }
}

impl Operator for AttentionBwd {
    fn name(&self) -> &str {
        "attention_bwd"
    }

    fn operands(&self) -> Vec<Operand> {
        [self.q, self.k, self.v, self.out, self.d_out, self.lse]
            .into_iter()
            .chain(self.spec.aux_operands().into_iter().flatten())
            .map(Operand::read)
            .collect()
    }

    fn infer_outputs(&self, graph: &GraphInfo) -> anyhow::Result<Vec<OutputSpec>> {
        let an = self.analyze(graph)?;
        // Rows past the sequences are not written.
        let init = if self.spec.is_varlen() {
            Init::Zero
        } else {
            Init::Uninit
        };
        Ok(vec![
            OutputSpec::new(an.in_type, an.q_shape.clone(), init),
            OutputSpec::new(an.in_type, an.k_shape.clone(), init),
            OutputSpec::new(an.in_type, an.k_shape, init),
        ])
    }

    fn build(&self, bcx: &BuildCx<'_>) -> anyhow::Result<Box<dyn Executor>> {
        let an = self.analyze(bcx.graph)?;
        let info = DeviceInfo::from_device(bcx.cx.cuda());
        let problem = self.spec.problem(&an, &info)?.backward();
        Ok(Box::new(exec::AttentionExecutor::new(
            bcx,
            &problem,
            &info,
            &self.spec,
            &an,
            exec::Direction::Backward,
        )?))
    }
}
