//! Matrix multiplication through cuBLASLt.
//!
//! Our tensors are row-major, cuBLASLt's are column-major. By default the
//! transposed product is computed: cuBLASLt's `A' = B`, `B' = A` and `D' = D`,
//! so that the column-major `D'[N, M]` is exactly the row-major `D[M, N]`.
//! cuBLASLt only computes block scales of D along its contiguous (column-major
//! row) dimension, so a colwise block-scaled D is produced with swapped roles
//! (`A' = A`, `B' = B`), which yields the column-major `D[M, N]`.

use crate::{
    context::Context,
    execgraph::{
        Bound, BuildCx, Executor, GraphInfo, Init, OpIo, Operand, Operator, OutputSpec, SymTensor,
        SymTensorData, SymTensorId, TensorRef, TensorRefMut, TensorTy,
        sym::{DimValues, Monomial, SymDims, SymLayout, SymShape},
    },
    mx_tensor::{MxDim, MxFormat},
};
use anyhow::{Context as _, anyhow, bail, ensure};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use skycuda::{
    cudarc::driver::sys::CUdeviceptr_v2,
    device_memory::{
        DeviceMemory, DevicePtrGuard, DevicePtrGuardMut, DeviceVisibleMemory,
        DeviceVisibleMemoryMut,
    },
    memory_slice::{MemorySlice, MemorySliceMut},
    stream::Stream,
};
use skycuda_cublaslt::{
    Algorithm, ClusterShapeId, ComputeType, DataType, InnerShapeId, MatmulDesc, MatmulPlan,
    MatmulPreference, MatmulTypes, MatrixLayout, Operation, ReductionScheme, Scalar, ScaleMode,
    Scales, StagesId, Status, TileId,
};
use skytens_core::{
    data_type::{ElementType, FloatType},
    shape::{Shape, Strides},
};
use std::{collections::HashMap, sync::Arc};

/// Operand alignment in bytes admitting the fastest kernels; required for
/// block-scaled operands.
const FAST_ALIGNMENT: u32 = 16;
/// cuBLASLt requires the workspace pointer to be aligned to this many bytes.
const WORKSPACE_ALIGNMENT: usize = 256;
/// Plans cached per executor for concrete shapes other than the hint.
const MAX_CACHED_PLANS: usize = 64;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize)]
pub enum MatmulOutput {
    /// `Bf16` or `F32`.
    Plain(ElementType),
    BlockScaled {
        format: MxFormat,
        dim: MxDim,
    },
}

impl MatmulOutput {
    fn ty(self) -> TensorTy {
        match self {
            MatmulOutput::Plain(ty) => TensorTy::Plain(ty),
            MatmulOutput::BlockScaled { format, dim } => TensorTy::BlockScaled { format, dim },
        }
    }
}

/// `D = alpha * A·B + beta * C` with `A: [.., M, K]`, `B: [.., K, N]` and
/// `C, D: [.., M, N]`, with at most one leading batch dimension.
#[derive(Clone, Debug)]
pub struct Matmul {
    pub a: SymTensorId,
    pub b: SymTensorId,
    pub c: Option<SymTensorId>,
    /// `None` picks f32 for f32 inputs and bf16 otherwise.
    pub out: Option<MatmulOutput>,
    pub alpha: f32,
    pub beta: f32,
}

#[derive(Copy, Clone, Debug)]
pub struct MatmulOpts<'g> {
    pub out: Option<MatmulOutput>,
    /// Requires a nonzero `beta` to have an effect.
    pub c: Option<SymTensor<'g>>,
    pub alpha: f32,
    pub beta: f32,
}

impl Default for MatmulOpts<'_> {
    fn default() -> Self {
        Self {
            out: None,
            c: None,
            alpha: 1.0,
            beta: 0.0,
        }
    }
}

impl<'g> SymTensor<'g> {
    pub fn matmul(self, b: SymTensor<'g>, opts: MatmulOpts<'g>) -> anyhow::Result<SymTensor<'g>> {
        ensure!(
            std::ptr::eq(self.builder(), b.builder())
                && opts
                    .c
                    .is_none_or(|c| std::ptr::eq(self.builder(), c.builder())),
            "matmul operands belong to different graphs"
        );
        self.builder().add_op1(Matmul {
            a: self.id(),
            b: b.id(),
            c: opts.c.map(SymTensor::id),
            out: opts.out,
            alpha: opts.alpha,
            beta: opts.beta,
        })
    }
}

/// How one of our (row-major, possibly strided) matrices maps onto a cuBLASLt
/// operand.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct MatMap {
    /// Whether cuBLASLt's `op(X')` is our matrix transposed.
    transposed: bool,
    /// Whether the rows of `op(X')` are contiguous, i.e. `X'` is used as is
    /// rather than transposed.
    unit_rows: bool,
}

impl MatMap {
    fn op(self) -> Operation {
        if self.unit_rows {
            Operation::N
        } else {
            Operation::T
        }
    }

    /// Picks the contiguous dimension of a matrix `[r, c]` with strides
    /// `[s_r, s_c]`, with `op(X')` being the matrix (or its transpose).
    fn new(
        transposed: bool,
        shape: [&Monomial; 2],
        strides: [&Monomial; 2],
        what: &str,
        dims: &SymDims,
    ) -> anyhow::Result<Self> {
        let (p, q) = if transposed { (1, 0) } else { (0, 1) };
        let unit_rows = if strides[p].is_one() {
            true
        } else if strides[q].is_one() {
            false
        } else if shape[p].is_one() {
            true
        } else if shape[q].is_one() {
            false
        } else {
            bail!(
                "{what} has strides [{}, {}]; one of its two innermost dimensions must be contiguous",
                strides[0].display(dims),
                strides[1].display(dims)
            );
        };
        Ok(Self {
            transposed,
            unit_rows,
        })
    }

    /// Index (within the two innermost dimensions) of the dimension whose
    /// stride is the leading dimension.
    fn ld_dim(self) -> usize {
        // The leading dimension is the stride of op(X')'s columns if X' is
        // not transposed, else of its rows.
        let (p, q) = if self.transposed { (1, 0) } else { (0, 1) };
        if self.unit_rows { q } else { p }
    }

    /// The cuBLASLt layout of a matrix with concrete `[batch?, r, c]` shape.
    fn layout(
        self,
        data_type: DataType,
        shape: Shape,
        strides: Strides,
        what: &str,
    ) -> anyhow::Result<MatrixLayout> {
        let rank = shape.num_dims();
        let (r, c) = (shape[rank - 2], shape[rank - 1]);
        let (sr, sc) = (strides[rank - 2], strides[rank - 1]);
        let (p, q, sp, sq) = if self.transposed {
            (c, r, sc, sr)
        } else {
            (r, c, sr, sc)
        };
        let (rows, cols, ld) = if self.unit_rows {
            (p, q, sq)
        } else {
            (q, p, sp)
        };
        // The leading dimension only matters when it steps over more than one column.
        let ld = if cols <= 1 || rows == 0 {
            ld.max(rows).max(1)
        } else {
            ensure!(
                ld >= rows,
                "{what} has a leading-dimension stride of {ld} elements, smaller than the {rows} contiguous elements it steps over"
            );
            ld
        };
        let layout = MatrixLayout::new(data_type, rows as u64, cols as u64, ld as i64);
        Ok(if rank == 3 {
            layout.batched(shape[0] as i32, strides[0] as i64)
        } else {
            layout
        })
    }
}

/// Symbolic description of one operand, enough to evaluate its concrete layout.
#[derive(Clone, Debug)]
enum SymOperand {
    Plain(SymLayout),
    Mx { shape: SymShape, dim: MxDim },
}

impl SymOperand {
    fn of(data: &SymTensorData) -> Self {
        match data {
            SymTensorData::Plain { layout, .. } => SymOperand::Plain(layout.clone()),
            SymTensorData::BlockScaled { shape, dim, .. } => SymOperand::Mx {
                shape: shape.clone(),
                dim: *dim,
            },
        }
    }

    fn shape(&self) -> &SymShape {
        match self {
            SymOperand::Plain(layout) => &layout.shape,
            SymOperand::Mx { shape, .. } => shape,
        }
    }

    /// Element strides of the operand (of the values, for block-scaled tensors).
    fn strides(&self) -> Vec<Monomial> {
        match self {
            SymOperand::Plain(layout) => layout.strides.clone(),
            SymOperand::Mx { shape, dim } => match dim {
                MxDim::Rowwise => vec![shape[1].clone(), Monomial::one()],
                MxDim::Colwise => vec![Monomial::one(), shape[0].clone()],
            },
        }
    }

    fn eval(&self, dims: &DimValues) -> anyhow::Result<(Shape, Strides)> {
        match self {
            SymOperand::Plain(layout) => {
                let (shape, strides, _) = layout.eval(dims)?;
                Ok((shape, strides))
            }
            SymOperand::Mx { shape, dim } => {
                let shape = shape.eval(dims)?;
                let strides = match dim {
                    MxDim::Rowwise => Strides::new([shape[1], 1]),
                    MxDim::Colwise => Strides::new([1, shape[0]]),
                };
                Ok((shape, strides))
            }
        }
    }

    fn hint_offset(&self, dims: &SymDims) -> usize {
        match self {
            SymOperand::Plain(layout) => layout.offset.eval_hint(dims),
            SymOperand::Mx { .. } => 0,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum InputKind {
    Plain { a: ElementType, b: ElementType },
    Mx(MxFormat),
}

/// Everything derived from the graph needed to describe the cuBLASLt problem.
#[derive(Clone, Debug)]
struct Analysis {
    inputs: InputKind,
    out: MatmulOutput,
    /// Whether cuBLASLt's A'/B' are our A/B (instead of B/A).
    swapped: bool,
    a: SymOperand,
    b: SymOperand,
    c: Option<SymOperand>,
    d: SymOperand,
    a_map: MatMap,
    b_map: MatMap,
    c_map: Option<MatMap>,
    d_map: MatMap,
    /// Provable byte alignment of each operand's pointer, leading dimension
    /// and batch stride, capped at [`FAST_ALIGNMENT`].
    a_align: u32,
    b_align: u32,
    c_align: Option<u32>,
    d_align: u32,
    out_shape: SymShape,
}

fn data_type(ty: ElementType) -> anyhow::Result<DataType> {
    Ok(match ty {
        ElementType::Float(FloatType::F32) => DataType::F32,
        ElementType::Float(FloatType::Bf16) => DataType::BF16,
        ElementType::Float(FloatType::F8E4M3) => DataType::F8E4M3,
        ElementType::Float(FloatType::F4E2M1) => DataType::F4E2M1,
        _ => bail!("element type {ty:?} is not supported by cuBLASLt matmul"),
    })
}

fn scale_mode(format: MxFormat) -> ScaleMode {
    match format {
        MxFormat::Mxfp8 => ScaleMode::Vec32UE8M0,
        MxFormat::Nvfp4 => ScaleMode::Vec16UE4M3,
    }
}

/// Rejects zero strides in the matrix dimensions of a plain operand.
fn check_strides(layout: &SymLayout, what: &str, dims: &SymDims) -> anyhow::Result<()> {
    let rank = layout.rank();
    for dim in rank - 2..rank {
        ensure!(
            layout.shape[dim].is_one() || !layout.strides[dim].is_zero(),
            "{what} has a zero stride in dimension {dim} of size {}",
            layout.shape[dim].display(dims)
        );
    }
    Ok(())
}

/// Largest power-of-two byte alignment, at most [`FAST_ALIGNMENT`], provable for
/// the base pointer, leading dimension and batch stride of a plain operand.
fn provable_alignment(layout: &SymLayout, ty: ElementType, map: MatMap, dims: &SymDims) -> u32 {
    let element_log2 = (ty.bits_per_element() / 8).ilog2();
    let bytes_log2 = |elements_log2: u32| elements_log2.saturating_add(element_log2);
    let rank = layout.rank();
    let mut log2 = bytes_log2(layout.offset.pow2_divisibility(dims));
    let ld_dim = rank - 2 + map.ld_dim();
    if !layout.shape[ld_dim].is_one() {
        log2 = log2.min(bytes_log2(layout.strides[ld_dim].pow2_divisibility(dims)));
    }
    if rank == 3 && !layout.shape[0].is_one() {
        log2 = log2.min(bytes_log2(layout.strides[0].pow2_divisibility(dims)));
    }
    1 << log2.min(FAST_ALIGNMENT.ilog2())
}

/// Log2 of the number of `format` values in [`FAST_ALIGNMENT`] bytes.
fn fast_alignment_values_log2(format: MxFormat) -> u32 {
    (FAST_ALIGNMENT as usize * 8 / format.values_type().bits_per_element()).ilog2()
}

impl Matmul {
    fn analyze(&self, g: &GraphInfo) -> anyhow::Result<Analysis> {
        let dims = g.dims();
        let (a_data, b_data) = (g.tensor(self.a), g.tensor(self.b));
        let (a_shape, b_shape) = (a_data.shape(), b_data.shape());

        let inputs = match (a_data.ty(), b_data.ty()) {
            (TensorTy::Plain(a), TensorTy::Plain(b)) => {
                for (name, ty) in [("A", a), ("B", b)] {
                    ensure!(
                        matches!(ty, ElementType::Float(FloatType::F32 | FloatType::Bf16)),
                        "{name} has element type {ty:?}; plain matmul inputs must be f32 or bf16"
                    );
                }
                InputKind::Plain { a, b }
            }
            (
                TensorTy::BlockScaled {
                    format: fa,
                    dim: da,
                },
                TensorTy::BlockScaled {
                    format: fb,
                    dim: db,
                },
            ) => {
                ensure!(
                    fa == fb,
                    "block-scaled inputs must have the same format, got {fa:?} and {fb:?}"
                );
                ensure!(
                    da == MxDim::Rowwise,
                    "block-scaled A must be {:?} (blocks along K), got {da:?}",
                    MxDim::Rowwise
                );
                ensure!(
                    db == MxDim::Colwise,
                    "block-scaled B must be {:?} (blocks along K), got {db:?}",
                    MxDim::Colwise
                );
                InputKind::Mx(fa)
            }
            (a, b) => {
                bail!("cannot mix plain and block-scaled matmul inputs, got A {a:?} and B {b:?}")
            }
        };

        let rank = a_shape.rank();
        ensure!(
            rank == 2 || rank == 3,
            "A must have rank 2 or 3, got shape {}",
            a_shape.display(dims)
        );
        ensure!(
            b_shape.rank() == rank,
            "A and B must have the same rank, got shapes {} and {}",
            a_shape.display(dims),
            b_shape.display(dims)
        );
        if rank == 3 {
            ensure!(
                a_shape[0] == b_shape[0],
                "batch dimensions of A {} and B {} differ",
                a_shape.display(dims),
                b_shape.display(dims)
            );
        }
        let k = &a_shape[rank - 1];
        let n = &b_shape[rank - 1];
        ensure!(
            &b_shape[rank - 2] == k,
            "contraction dimensions differ: A has shape {} and B has shape {}",
            a_shape.display(dims),
            b_shape.display(dims)
        );
        ensure!(
            k.as_fixed() != Some(0),
            "the contraction dimension K must not be zero"
        );
        let mut out_shape = a_shape.clone();
        out_shape[rank - 1] = n.clone();

        let out = match self.out {
            Some(out) => out,
            None => match inputs {
                InputKind::Plain {
                    a: ElementType::Float(FloatType::F32),
                    b: ElementType::Float(FloatType::F32),
                } => MatmulOutput::Plain(ElementType::Float(FloatType::F32)),
                _ => MatmulOutput::Plain(ElementType::Float(FloatType::Bf16)),
            },
        };
        let swapped = match out {
            MatmulOutput::Plain(ty) => {
                ensure!(
                    matches!(ty, ElementType::Float(FloatType::F32 | FloatType::Bf16)),
                    "plain matmul output must be f32 or bf16, got {ty:?}"
                );
                false
            }
            MatmulOutput::BlockScaled { format, dim } => {
                ensure!(
                    rank == 2,
                    "block-scaled matmul outputs must be 2D, got shape {}",
                    out_shape.display(dims)
                );
                // The leading dimension of D' runs along its block dimension.
                let block_dim = &out_shape[dim.block_dimension_2d()];
                let log2 = fast_alignment_values_log2(format);
                ensure!(
                    block_dim.is_divisible_by_pow2(log2, dims),
                    "{format:?} {dim:?} output of shape {} needs its block dimension to be a provable multiple of {} for {FAST_ALIGNMENT}-byte alignment",
                    out_shape.display(dims),
                    1usize << log2
                );
                dim == MxDim::Colwise
            }
        };
        ensure!(
            self.c.is_some() || self.beta == 0.0,
            "beta is {} but no C operand was given",
            self.beta
        );

        let a = SymOperand::of(a_data);
        let b = SymOperand::of(b_data);
        // op(A') is B^T and op(B') is A^T, or with swapped roles A and B.
        let map = |op: &SymOperand, what: &str| {
            let shape = op.shape();
            let strides = op.strides();
            let r = shape.rank();
            MatMap::new(
                !swapped,
                [&shape[r - 2], &shape[r - 1]],
                [&strides[r - 2], &strides[r - 1]],
                what,
                dims,
            )
        };
        // Checks a plain operand and returns its alignment.
        let plain_alignment = |op: &SymOperand, ty: ElementType, map: MatMap, what: &str| {
            let SymOperand::Plain(layout) = op else {
                unreachable!("{what} is plain")
            };
            check_strides(layout, what, dims)?;
            anyhow::Ok(provable_alignment(layout, ty, map, dims))
        };

        let a_map = map(&a, "A")?;
        let b_map = map(&b, "B")?;
        let (a_align, b_align) = match inputs {
            InputKind::Plain { a: a_ty, b: b_ty } => (
                plain_alignment(&a, a_ty, a_map, "A")?,
                plain_alignment(&b, b_ty, b_map, "B")?,
            ),
            InputKind::Mx(format) => {
                // cuBLASLt requires the TN form for block-scaled inputs: A' transposed, B' not.
                let (a_role, b_role) = if swapped {
                    (a_map, b_map)
                } else {
                    (b_map, a_map)
                };
                ensure!(
                    a_role.op() == Operation::T && b_role.op() == Operation::N,
                    "block-scaled inputs do not map to cuBLASLt's TN form"
                );
                let log2 = fast_alignment_values_log2(format);
                ensure!(
                    k.is_divisible_by_pow2(log2, dims),
                    "{format:?} inputs need K = {} to be a provable multiple of {} for {FAST_ALIGNMENT}-byte alignment",
                    k.display(dims),
                    1usize << log2
                );
                (FAST_ALIGNMENT, FAST_ALIGNMENT)
            }
        };

        let d = match out {
            MatmulOutput::Plain(ty) => SymOperand::Plain(
                SymLayout::contiguous_row_major(out_shape.clone(), ty, dims)
                    .context("output layout")?,
            ),
            MatmulOutput::BlockScaled { dim, .. } => SymOperand::Mx {
                shape: out_shape.clone(),
                dim,
            },
        };
        let d_map = map(&d, "D")?;
        ensure!(
            d_map.op() == Operation::N,
            "output does not map to cuBLASLt's D"
        );
        let d_align = match out {
            MatmulOutput::Plain(ty) => plain_alignment(&d, ty, d_map, "D")?,
            MatmulOutput::BlockScaled { .. } => FAST_ALIGNMENT,
        };

        let (c, c_map, c_align) = match self.c {
            None => (None, None, None),
            Some(c_id) => {
                let MatmulOutput::Plain(out_ty) = out else {
                    bail!("C is only supported for plain outputs");
                };
                let c_data = g.tensor(c_id);
                ensure!(
                    c_data.ty() == TensorTy::Plain(out_ty),
                    "C must have the output element type {out_ty:?}, got {:?}",
                    c_data.ty()
                );
                ensure!(
                    c_data.shape() == &out_shape,
                    "C has shape {}, expected {}",
                    c_data.shape().display(dims),
                    out_shape.display(dims)
                );
                let c = SymOperand::of(c_data);
                let c_map = map(&c, "C")?;
                ensure!(
                    c_map.op() == Operation::N,
                    "C must be contiguous along N, i.e. not a transposed view"
                );
                let c_align = plain_alignment(&c, out_ty, c_map, "C")?;
                (Some(c), Some(c_map), Some(c_align))
            }
        };

        Ok(Analysis {
            inputs,
            out,
            swapped,
            a,
            b,
            c,
            d,
            a_map,
            b_map,
            c_map,
            d_map,
            a_align,
            b_align,
            c_align,
            d_align,
            out_shape,
        })
    }

    fn describe_types(&self, g: &GraphInfo, out: MatmulOutput) -> String {
        let ty = |ty: TensorTy| match ty {
            TensorTy::Plain(ty) => format!("{ty:?}"),
            TensorTy::BlockScaled { format, dim } => format!("{format:?} {dim:?}"),
        };
        format!(
            "{} x {} -> {}",
            ty(g.ty(self.a)),
            ty(g.ty(self.b)),
            ty(out.ty())
        )
    }
}

impl Operator for Matmul {
    fn name(&self) -> &str {
        "matmul"
    }

    fn operands(&self) -> Vec<Operand> {
        [self.a, self.b]
            .into_iter()
            .chain(self.c)
            .map(Operand::read)
            .collect()
    }

    fn infer_outputs(&self, graph: &GraphInfo) -> anyhow::Result<Vec<OutputSpec>> {
        let analysis = self.analyze(graph)?;
        let out = OutputSpec::new(analysis.out.ty(), analysis.out_shape.clone(), Init::Uninit);
        // Catches strides that are only invalid for concrete sizes, e.g. a
        // leading dimension smaller than the row extent.
        Problem::new(self, analysis)?.layouts(&graph.hint_values())?;
        Ok(vec![out])
    }

    fn build(&self, bcx: &BuildCx<'_>) -> anyhow::Result<Box<dyn Executor>> {
        let analysis = self.analyze(bcx.graph)?;
        let combo = self.describe_types(bcx.graph, analysis.out);
        let problem = Arc::new(Problem::new(self, analysis)?);

        let hint_dims = bcx.graph.hint_values();
        let hint_layouts = problem.layouts(&hint_dims)?;
        let plan = bcx
            .cx
            .cublaslt()
            .build_plan(&problem.desc(&hint_layouts), &problem.preference())
            .map_err(|e| {
                if matches!(
                    e.status(),
                    Some(Status::NotSupported | Status::InvalidValue)
                ) {
                    anyhow!(e).context(format!("cuBLASLt does not support matmul {combo}"))
                } else {
                    anyhow!(e).context(format!("building cuBLASLt plan for matmul {combo}"))
                }
            })?;
        let plan = Arc::new(plan);

        let candidates = plan
            .algorithms()
            .iter()
            .map(AlgoConfig::of)
            .collect::<anyhow::Result<Vec<_>>>()?;
        let payload = problem.tune_payload(&hint_layouts, bcx.graph.dims());
        let key = bcx.tune_key(&payload)?;
        let (config, executor) = bcx.tune_executor(&key, candidates, |config| {
            let algo = config.algorithm(&plan)?;
            Ok(MatmulExecutor::new(
                problem.clone(),
                plan.clone(),
                hint_layouts,
                config.clone(),
                algo,
            ))
        })?;
        tracing::debug!(
            target: "cleandl::matmul",
            "matmul {combo} at hint D' {:?}: chose {config:?}",
            hint_layouts.d
        );
        Ok(Box::new(executor))
    }
}

/// Serializable form of an [`Algorithm`] configuration, for tuning.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct AlgoConfig {
    id: i32,
    tile: u32,
    splitk: i32,
    reduction_scheme: u32,
    cta_swizzling: u32,
    custom_option: u32,
    stages: u32,
    inner_shape: u16,
    cluster_shape: u16,
}

impl AlgoConfig {
    fn of(algo: &Algorithm) -> anyhow::Result<Self> {
        let config = algo.config()?;
        Ok(Self {
            id: config.id,
            tile: config.tile.0,
            splitk: config.splitk,
            reduction_scheme: config.reduction_scheme.raw(),
            cta_swizzling: config.cta_swizzling,
            custom_option: config.custom_option,
            stages: config.stages.0,
            inner_shape: config.inner_shape.0,
            cluster_shape: config.cluster_shape.0,
        })
    }

    /// The plan's algorithm with this configuration, if the plan has it.
    fn find(&self, plan: &MatmulPlan) -> Option<Algorithm> {
        plan.algorithms()
            .iter()
            .find(|algo| Self::of(algo).is_ok_and(|config| config == *self))
            .copied()
    }

    /// The plan's algorithm with this configuration, or one rebuilt from it.
    fn algorithm(&self, plan: &MatmulPlan) -> anyhow::Result<Algorithm> {
        match self.find(plan) {
            Some(algo) => Ok(algo),
            None => self.rebuild(plan),
        }
    }

    fn rebuild(&self, plan: &MatmulPlan) -> anyhow::Result<Algorithm> {
        let mut algo =
            Algorithm::init(plan.handle(), &MatmulTypes::from_desc(plan.desc()), self.id)?;
        algo.set_tile(TileId(self.tile))?;
        algo.set_splitk(self.splitk)?;
        algo.set_reduction_scheme(
            ReductionScheme::from_raw(self.reduction_scheme)
                .ok_or_else(|| anyhow!("unknown reduction scheme {}", self.reduction_scheme))?,
        )?;
        algo.set_cta_swizzling(self.cta_swizzling)?;
        algo.set_custom_option(self.custom_option)?;
        algo.set_stages(StagesId(self.stages))?;
        algo.set_inner_shape(InnerShapeId(self.inner_shape))?;
        algo.set_cluster_shape(ClusterShapeId(self.cluster_shape))?;
        algo.workspace_size = plan.check(&algo)?.workspace_size;
        Ok(algo)
    }
}

#[derive(Serialize)]
struct LayoutPayload {
    data_type: u32,
    rows: u64,
    cols: u64,
    ld: i64,
    batch_count: i32,
    batch_stride: i64,
}

impl From<&MatrixLayout> for LayoutPayload {
    fn from(l: &MatrixLayout) -> Self {
        Self {
            data_type: l.data_type.raw(),
            rows: l.rows,
            cols: l.cols,
            ld: l.ld,
            batch_count: l.batch_count,
            batch_stride: l.batch_stride,
        }
    }
}

#[derive(Serialize)]
struct TunePayload {
    compute: u32,
    trans_a: u32,
    trans_b: u32,
    scales: [Option<u32>; 3],
    a: LayoutPayload,
    b: LayoutPayload,
    c: Option<LayoutPayload>,
    d: LayoutPayload,
    out: MatmulOutput,
    has_c: bool,
    beta_nonzero: bool,
    /// Hint element offsets of A, B and C.
    offsets: Vec<usize>,
    /// Promised alignment of A', B', C' and D'.
    alignments: [u32; 4],
}

/// Concrete cuBLASLt layouts of A', B', C' (if separate from D') and D'.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
struct Layouts {
    a: MatrixLayout,
    b: MatrixLayout,
    c: Option<MatrixLayout>,
    d: MatrixLayout,
    k: usize,
}

impl Layouts {
    fn is_empty(&self) -> bool {
        self.d.rows == 0 || self.d.cols == 0 || self.d.batch_count == 0
    }
}

/// The problem, independent of the chosen algorithm.
#[derive(Debug)]
struct Problem {
    analysis: Analysis,
    compute: ComputeType,
    scales: Scales,
    /// Value types of our A and B.
    our_a_type: DataType,
    our_b_type: DataType,
    d_type: DataType,
    /// Type of the C layout when C is not D: the user's C, or an unread bf16
    /// C for block-scaled outputs, for which cuBLASLt requires a 16-bit C type.
    c_type: Option<DataType>,
    alpha: f32,
    beta: f32,
}

impl Problem {
    fn new(op: &Matmul, analysis: Analysis) -> anyhow::Result<Self> {
        let (our_a_type, our_b_type, compute) = match analysis.inputs {
            InputKind::Plain { a, b } => {
                let compute = if a == ElementType::Float(FloatType::F32)
                    && b == ElementType::Float(FloatType::F32)
                {
                    ComputeType::F32FastTF32
                } else {
                    ComputeType::F32
                };
                (data_type(a)?, data_type(b)?, compute)
            }
            InputKind::Mx(format) => {
                let ty = data_type(format.values_type())?;
                (ty, ty, ComputeType::F32)
            }
        };
        let mut scales = Scales::default();
        if let InputKind::Mx(format) = analysis.inputs {
            scales.a = Some(scale_mode(format));
            scales.b = Some(scale_mode(format));
        }
        let (d_type, c_type) = match analysis.out {
            MatmulOutput::Plain(ty) => (data_type(ty)?, analysis.c.as_ref().map(|_| ty)),
            MatmulOutput::BlockScaled { format, .. } => {
                scales.d_out = Some(scale_mode(format));
                (
                    data_type(format.values_type())?,
                    Some(ElementType::Float(FloatType::Bf16)),
                )
            }
        };
        Ok(Self {
            compute,
            scales,
            our_a_type,
            our_b_type,
            d_type,
            c_type: c_type.map(data_type).transpose()?,
            alpha: op.alpha,
            beta: op.beta,
            analysis,
        })
    }

    fn layouts(&self, dims: &DimValues) -> anyhow::Result<Layouts> {
        let an = &self.analysis;
        let (a_shape, a_strides) = an.a.eval(dims)?;
        let k = a_shape[a_shape.num_dims() - 1];
        let our_a = an.a_map.layout(self.our_a_type, a_shape, a_strides, "A")?;
        let our_b = {
            let (shape, strides) = an.b.eval(dims)?;
            an.b_map.layout(self.our_b_type, shape, strides, "B")?
        };
        let (a, b) = if an.swapped {
            (our_a, our_b)
        } else {
            (our_b, our_a)
        };
        let (d_shape, d_strides) = an.d.eval(dims)?;
        let d = an.d_map.layout(self.d_type, d_shape, d_strides, "D")?;
        let c = match (&an.c, an.c_map, self.c_type) {
            (Some(c), Some(map), Some(ty)) => {
                let (shape, strides) = c.eval(dims)?;
                Some(map.layout(ty, shape, strides, "C")?)
            }
            // The unread C of a block-scaled output is bound to A''s buffer,
            // which is too small for a full C, so it is a broadcast (ld 0).
            (None, _, Some(ty)) => Some(MatrixLayout::new(ty, d.rows, d.cols, 0)),
            _ => None,
        };
        Ok(Layouts { a, b, c, d, k })
    }

    fn desc(&self, layouts: &Layouts) -> MatmulDesc {
        let an = &self.analysis;
        let (a_map, b_map) = if an.swapped {
            (an.a_map, an.b_map)
        } else {
            (an.b_map, an.a_map)
        };
        let desc = MatmulDesc::new(self.compute, layouts.a, layouts.b, layouts.d)
            .scale_type(DataType::F32)
            .trans_a(a_map.op())
            .trans_b(b_map.op())
            .scales(self.scales);
        match layouts.c {
            Some(c) => desc.c(c),
            None => desc,
        }
    }

    /// Promised pointer alignments of A', B', C' and D'.
    fn alignments(&self) -> [u32; 4] {
        let an = &self.analysis;
        let (a, b) = if an.swapped {
            (an.a_align, an.b_align)
        } else {
            (an.b_align, an.a_align)
        };
        let c = match (an.c_align, self.c_type) {
            (Some(c), _) => c,
            // The unread C is bound to A''s buffer.
            (None, Some(_)) => a,
            (None, None) => an.d_align,
        };
        [a, b, c, an.d_align]
    }

    fn preference(&self) -> MatmulPreference {
        let mut pref = MatmulPreference::default()
            .aux_alignment_bytes(FAST_ALIGNMENT)
            .requested_algo_count(16);
        pref.min_alignment_bytes = self.alignments();
        pref
    }

    fn tune_payload(&self, layouts: &Layouts, dims: &SymDims) -> TunePayload {
        let an = &self.analysis;
        let desc = self.desc(layouts);
        TunePayload {
            compute: self.compute.raw(),
            trans_a: desc.trans_a.raw(),
            trans_b: desc.trans_b.raw(),
            scales: [self.scales.a, self.scales.b, self.scales.d_out].map(|s| s.map(|s| s.raw())),
            a: (&layouts.a).into(),
            b: (&layouts.b).into(),
            c: layouts.c.as_ref().map(Into::into),
            d: (&layouts.d).into(),
            out: an.out,
            has_c: an.c.is_some(),
            beta_nonzero: self.beta != 0.0,
            offsets: [Some(&an.a), Some(&an.b), an.c.as_ref()]
                .into_iter()
                .flatten()
                .map(|op| op.hint_offset(dims))
                .collect(),
            alignments: self.alignments(),
        }
    }
}

/// A plan for one concrete problem shape, with the algorithm to run.
struct Resolved {
    plan: Arc<MatmulPlan>,
    algo: Algorithm,
}

/// Plans for concrete shapes. The hint plan is never evicted; the others are
/// evicted least recently used first.
struct PlanCache {
    hint: (Layouts, Arc<Resolved>),
    others: HashMap<Layouts, (Arc<Resolved>, u64)>,
    clock: u64,
}

impl PlanCache {
    fn get(&mut self, layouts: &Layouts) -> Option<Arc<Resolved>> {
        if *layouts == self.hint.0 {
            return Some(self.hint.1.clone());
        }
        self.clock += 1;
        let (resolved, last_used) = self.others.get_mut(layouts)?;
        *last_used = self.clock;
        Some(resolved.clone())
    }

    fn insert(&mut self, layouts: Layouts, resolved: Arc<Resolved>) {
        if self.others.len() >= MAX_CACHED_PLANS
            && let Some(oldest) = self
                .others
                .iter()
                .min_by_key(|(_, (_, last_used))| *last_used)
                .map(|(layouts, _)| *layouts)
        {
            self.others.remove(&oldest);
        }
        self.clock += 1;
        self.others.insert(layouts, (resolved, self.clock));
    }
}

struct MatmulExecutor {
    problem: Arc<Problem>,
    hint_plan: Arc<MatmulPlan>,
    config: AlgoConfig,
    plans: Mutex<PlanCache>,
}

impl MatmulExecutor {
    fn new(
        problem: Arc<Problem>,
        hint_plan: Arc<MatmulPlan>,
        hint_layouts: Layouts,
        config: AlgoConfig,
        algo: Algorithm,
    ) -> Self {
        let resolved = Arc::new(Resolved {
            plan: hint_plan.clone(),
            algo,
        });
        Self {
            problem,
            hint_plan,
            config,
            plans: Mutex::new(PlanCache {
                hint: (hint_layouts, resolved),
                others: HashMap::new(),
                clock: 0,
            }),
        }
    }

    /// The plan for a concrete shape, reshaping the hint plan on first use.
    fn resolve(&self, layouts: &Layouts) -> anyhow::Result<Arc<Resolved>> {
        if let Some(resolved) = self.plans.lock().get(layouts) {
            return Ok(resolved);
        }
        let desc = self.problem.desc(layouts);
        let plan = match self.hint_plan.reshape(&desc) {
            Ok(plan) => plan,
            Err(e) if e.status() == Some(Status::NotSupported) => {
                tracing::debug!(
                    "matmul: hint algorithms cannot run {layouts:?}; rerunning heuristic"
                );
                self.hint_plan
                    .handle()
                    .build_plan(&desc, self.hint_plan.preference())
                    .with_context(|| format!("building cuBLASLt plan for {layouts:?}"))?
            }
            Err(e) => return Err(e.into()),
        };
        // Keep the tuned algorithm only if it survived the plan's checks,
        // including its workspace budget.
        let algo = self.config.find(&plan).unwrap_or_else(|| *plan.candidate());
        let resolved = Arc::new(Resolved {
            plan: Arc::new(plan),
            algo,
        });
        self.plans.lock().insert(*layouts, resolved.clone());
        Ok(resolved)
    }
}

/// The plain storage or block-scaled values of a tensor.
fn values<'a>(t: &TensorRef<'a>) -> MemorySlice<&'a DeviceMemory> {
    match t {
        TensorRef::Plain(t) => *t.storage(),
        TensorRef::Mx(t) => *t.values().storage(),
    }
}

/// Fails unless the memory's address meets the alignment promised to cuBLASLt.
fn check_address(
    memory: &impl DeviceVisibleMemory,
    align: u32,
    what: &str,
    stream: &Stream,
) -> anyhow::Result<()> {
    // Only the address is read; the operand's binding takes its own access.
    let ptr = unsafe { memory.access_on(stream)?.ptr() };
    ensure!(
        ptr.is_multiple_of(u64::from(align)),
        "{what} is at address {ptr:#x}, which is not {align}-byte aligned as its layout implies"
    );
    Ok(())
}

impl Executor for MatmulExecutor {
    fn scratch_bytes(&self, dims: &DimValues) -> usize {
        let Ok(layouts) = self.problem.layouts(dims) else {
            return 0;
        };
        if layouts.is_empty() || layouts.k == 0 {
            return 0;
        }
        match self.resolve(&layouts) {
            Ok(resolved) if resolved.algo.workspace_size > 0 => {
                // Room to align the workspace.
                resolved.algo.workspace_size + WORKSPACE_ALIGNMENT
            }
            _ => 0,
        }
    }

    fn execute(&self, io: &mut OpIo<'_>, stream: &Stream, _cx: &Context) -> anyhow::Result<()> {
        let problem = &*self.problem;
        let layouts = problem.layouts(io.dims)?;
        if layouts.is_empty() {
            return Ok(());
        }
        ensure!(
            layouts.k > 0,
            "matmul contraction dimension K is zero at execution time"
        );
        let resolved = self.resolve(&layouts)?;
        let (plan, algo) = (&*resolved.plan, &resolved.algo);
        let [a_align, b_align, c_align, _] = problem.alignments();

        let OpIo {
            inputs,
            outputs,
            scratch,
            ..
        } = io;
        let input = |i: usize| match &inputs[i] {
            Bound::Ro(t) => t.clone(),
            Bound::Rw(t) => t.as_ref(),
        };
        let (our_a, our_b) = (input(0), input(1));
        let c = problem.analysis.c.as_ref().map(|_| input(2));
        let (a, b) = if problem.analysis.swapped {
            (&our_a, &our_b)
        } else {
            (&our_b, &our_a)
        };
        let (a_values, b_values) = (values(a), values(b));
        check_address(&a_values, a_align, "A'", stream)?;
        check_address(&b_values, b_align, "B'", stream)?;

        let mut ops = plan.operands();
        ops.alpha(Scalar::F32(problem.alpha))
            .beta(Scalar::F32(problem.beta))
            .a(&a_values)
            .b(&b_values);
        if let TensorRef::Mx(t) = a {
            ops.a_scale(t.scale_factors().storage());
        }
        if let TensorRef::Mx(t) = b {
            ops.b_scale(t.scale_factors().storage());
        }
        if let Some(c) = &c {
            let c = c.plain().storage();
            check_address(c, c_align, "C", stream)?;
            ops.c(c);
        }
        match &mut outputs[0] {
            TensorRefMut::Plain(d) => {
                ops.d(d.storage_mut());
            }
            TensorRefMut::Mx(d) => {
                // cuBLASLt requires a bf16 C distinct from D (and not null)
                // for block-scaled outputs. beta is zero, so it is never read;
                // bind A''s buffer as a broadcast C. Unverified until run on sm_100+.
                ops.c(&a_values);
                let (values, scales) = d.values_and_scale_factors_mut();
                // cuBLASLt writes the output scale factors, which the operand
                // builder only binds for reading: record the write here.
                scales.storage_mut().access_mut_on(stream)?;
                ops.d(values.storage_mut()).d_out_scale(scales.storage());
            }
        }

        if algo.workspace_size == 0 {
            unsafe { plan.execute_with(algo, stream, &ops, &mut EmptyWorkspace)? };
            return Ok(());
        }
        let Some(scratch) = scratch else {
            bail!(
                "matmul needs {} bytes of scratch memory, but none was given",
                algo.workspace_size
            );
        };
        let address = unsafe { scratch.access_mut_on(stream)?.ptr() };
        let padding = (address as usize).next_multiple_of(WORKSPACE_ALIGNMENT) - address as usize;
        ensure!(
            scratch.len() >= padding + algo.workspace_size,
            "matmul needs {} bytes of aligned scratch memory, but {} bytes were given",
            algo.workspace_size,
            scratch.len()
        );
        let start = scratch.offset() + padding;
        let end = scratch.offset() + scratch.len();
        let mut workspace = MemorySliceMut::new(&mut **scratch.inner_mut(), start..end);
        unsafe { plan.execute_with(algo, stream, &ops, &mut workspace)? };
        Ok(())
    }
}

/// A zero-byte workspace, passed as a null pointer.
struct EmptyWorkspace;

struct NullPtr;

impl DevicePtrGuard<'_> for NullPtr {
    unsafe fn ptr(&self) -> CUdeviceptr_v2 {
        0
    }
}

impl DevicePtrGuardMut<'_> for NullPtr {
    unsafe fn ptr(&mut self) -> CUdeviceptr_v2 {
        0
    }
}

impl DeviceVisibleMemory for EmptyWorkspace {
    type Guard<'a> = NullPtr;

    fn len(&self) -> usize {
        0
    }

    fn access_on<'a>(&'a self, _stream: &'a Stream) -> skycuda::error::Result<NullPtr> {
        Ok(NullPtr)
    }
}

impl DeviceVisibleMemoryMut for EmptyWorkspace {
    type GuardMut<'a> = NullPtr;

    fn access_mut_on<'a>(&'a mut self, _stream: &'a Stream) -> skycuda::error::Result<NullPtr> {
        Ok(NullPtr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execgraph::GraphBuilder;

    const BF16: ElementType = ElementType::Float(FloatType::Bf16);

    fn setup(
        gb: GraphBuilder,
        f: impl FnOnce(&GraphBuilder) -> anyhow::Result<Matmul>,
    ) -> anyhow::Result<(Problem, Layouts)> {
        let op = f(&gb)?;
        let graph = gb.finish()?;
        let analysis = op.analyze(graph.info())?;
        let problem = Problem::new(&op, analysis)?;
        let layouts = problem.layouts(&graph.info().hint_values())?;
        Ok((problem, layouts))
    }

    fn plain(a: &SymTensor<'_>, b: &SymTensor<'_>) -> Matmul {
        Matmul {
            a: a.id(),
            b: b.id(),
            c: None,
            out: None,
            alpha: 1.0,
            beta: 0.0,
        }
    }

    #[test]
    fn row_major_maps_to_transposed_product() -> anyhow::Result<()> {
        let (m, k, n) = (24, 40, 56);
        let (problem, l) = setup(GraphBuilder::new(), |gb| {
            Ok(plain(
                &gb.input("a", BF16, [m, k])?,
                &gb.input("b", BF16, [k, n])?,
            ))
        })?;
        let desc = problem.desc(&l);
        assert_eq!((desc.trans_a, desc.trans_b), (Operation::N, Operation::N));
        assert_eq!((l.a.rows, l.a.cols, l.a.ld), (n as u64, k as u64, n as i64));
        assert_eq!((l.b.rows, l.b.cols, l.b.ld), (k as u64, m as u64, k as i64));
        assert_eq!((l.d.rows, l.d.cols, l.d.ld), (n as u64, m as u64, n as i64));
        assert!(l.c.is_none());
        Ok(())
    }

    #[test]
    fn transposed_views_map_to_transposes() -> anyhow::Result<()> {
        let (m, k, n) = (24, 40, 56);
        let (problem, l) = setup(GraphBuilder::new(), |gb| {
            Ok(plain(
                &gb.input("a", BF16, [k, m])?.transposed(0, 1)?,
                &gb.input("b", BF16, [n, k])?.transposed(0, 1)?,
            ))
        })?;
        let desc = problem.desc(&l);
        assert_eq!((desc.trans_a, desc.trans_b), (Operation::T, Operation::T));
        // B is K-contiguous: col-major K x N with ld K.
        assert_eq!((l.a.rows, l.a.cols, l.a.ld), (k as u64, n as u64, k as i64));
        // A is M-contiguous: col-major M x K with ld M.
        assert_eq!((l.b.rows, l.b.cols, l.b.ld), (m as u64, k as u64, m as i64));
        Ok(())
    }

    #[test]
    fn batched_layouts() -> anyhow::Result<()> {
        let (batch, m, k, n) = (3, 24, 40, 56);
        let (_, l) = setup(GraphBuilder::new(), |gb| {
            Ok(plain(
                &gb.input("a", BF16, [batch, m, k])?,
                &gb.input("b", BF16, [batch, k, n])?,
            ))
        })?;
        for (layout, stride) in [(l.a, k * n), (l.b, m * k), (l.d, m * n)] {
            assert_eq!(layout.batch_count, batch as i32);
            assert_eq!(layout.batch_stride, stride as i64);
        }
        Ok(())
    }

    #[test]
    fn block_scaled_maps_to_tn() -> anyhow::Result<()> {
        let (m, k, n) = (128, 256, 64);
        for out_dim in [None, Some(MxDim::Rowwise), Some(MxDim::Colwise)] {
            let (problem, l) = setup(GraphBuilder::new(), |gb| {
                let a = gb.input_mx("a", MxFormat::Mxfp8, MxDim::Rowwise, [m, k])?;
                let b = gb.input_mx("b", MxFormat::Mxfp8, MxDim::Colwise, [k, n])?;
                Ok(Matmul {
                    out: out_dim.map(|dim| MatmulOutput::BlockScaled {
                        format: MxFormat::Mxfp8,
                        dim,
                    }),
                    ..plain(&a, &b)
                })
            })?;
            let desc = problem.desc(&l);
            assert_eq!((desc.trans_a, desc.trans_b), (Operation::T, Operation::N));
            assert_eq!(
                (l.a.rows, l.a.ld, l.b.rows, l.b.ld),
                (k as u64, k as i64, k as u64, k as i64)
            );
            if out_dim == Some(MxDim::Colwise) {
                // D' is our D in column-major form, blocks along M.
                assert_eq!((l.d.rows, l.d.cols), (m as u64, n as u64));
            } else {
                assert_eq!((l.d.rows, l.d.cols), (n as u64, m as u64));
            }
            assert_eq!(l.c.is_some(), out_dim.is_some());
        }
        Ok(())
    }

    fn test_context() -> anyhow::Result<Context> {
        Context::new(
            0,
            Default::default(),
            skycuda::host_pinned_memory_pool::HostPinnedMemoryPoolOptions {
                max_reserved_free_bytes: 1 << 20,
            },
        )
    }

    #[test]
    fn alignment_is_per_operand() -> anyhow::Result<()> {
        let (problem, _) = setup(GraphBuilder::new(), |gb| {
            let a = gb.input("a", BF16, [16, 40])?.sliced(1, 3, 32)?;
            let b = gb.input("b", BF16, [32, 12])?;
            Ok(plain(&a, &b))
        })?;
        // [A' = B (ld 12 bf16 = 24 bytes), B' = A (offset 3 bf16 = 6 bytes), C = D, D (ld 24 bytes)].
        assert_eq!(problem.alignments(), [8, 2, 8, 8]);
        Ok(())
    }

    #[test]
    fn plans_build() -> anyhow::Result<()> {
        let cx = test_context()?;
        let (problem, l) = setup(GraphBuilder::new(), |gb| {
            Ok(plain(
                &gb.input("a", BF16, [64, 64])?,
                &gb.input("b", BF16, [64, 64])?,
            ))
        })?;
        let plan = cx
            .cublaslt()
            .build_plan(&problem.desc(&l), &problem.preference())?;
        let config = AlgoConfig::of(plan.candidate())?;
        assert_eq!(AlgoConfig::of(&config.algorithm(&plan)?)?, config);

        // Block-scaled descriptors are well-formed; they only build on sm_100+.
        let (problem, l) = setup(GraphBuilder::new(), |gb| {
            let a = gb.input_mx("a", MxFormat::Mxfp8, MxDim::Rowwise, [128, 128])?;
            let b = gb.input_mx("b", MxFormat::Mxfp8, MxDim::Colwise, [128, 128])?;
            Ok(Matmul {
                out: Some(MatmulOutput::BlockScaled {
                    format: MxFormat::Mxfp8,
                    dim: MxDim::Rowwise,
                }),
                ..plain(&a, &b)
            })
        })?;
        match cx
            .cublaslt()
            .build_plan(&problem.desc(&l), &problem.preference())
        {
            Ok(_) => {}
            Err(e) if cx.cuda().compute_capability_major() < 10 => {
                assert_eq!(e.status(), Some(Status::NotSupported), "{e}");
            }
            Err(e) => return Err(e.into()),
        }
        Ok(())
    }

    #[test]
    fn rebuilt_algorithms_round_trip() -> anyhow::Result<()> {
        let cx = test_context()?;
        let (problem, l) = setup(GraphBuilder::new(), |gb| {
            Ok(plain(
                &gb.input("a", BF16, [128, 256])?,
                &gb.input("b", BF16, [256, 96])?,
            ))
        })?;
        let plan = cx
            .cublaslt()
            .build_plan(&problem.desc(&l), &problem.preference())?;
        for algo in plan.algorithms() {
            let config = AlgoConfig::of(algo)?;
            let rebuilt = config.rebuild(&plan)?;
            assert_eq!(AlgoConfig::of(&rebuilt)?, config);
            assert_eq!(rebuilt.workspace_size, plan.check(algo)?.workspace_size);
        }
        Ok(())
    }

    /// An executor for `[m, 256] x [256, 128]` with a dynamic `m` (hint 64).
    fn dynamic_executor(
        cx: &Context,
        config: impl FnOnce(&MatmulPlan) -> AlgoConfig,
    ) -> anyhow::Result<(MatmulExecutor, crate::execgraph::sym::SymDimId)> {
        let gb = GraphBuilder::new();
        let m = gb.dynamic_dim("m", 64, 0);
        let op = plain(
            &gb.input("a", BF16, [Monomial::atom(m), Monomial::fixed(256)])?,
            &gb.input("b", BF16, [256, 128])?,
        );
        let graph = gb.finish()?;
        let problem = Arc::new(Problem::new(&op, op.analyze(graph.info())?)?);
        let hint_layouts = problem.layouts(&graph.info().hint_values())?;
        let plan = Arc::new(
            cx.cublaslt()
                .build_plan(&problem.desc(&hint_layouts), &problem.preference())?,
        );
        let config = config(&plan);
        let algo = config.algorithm(&plan)?;
        let executor = MatmulExecutor::new(problem, plan, hint_layouts, config, algo);
        Ok((executor, m))
    }

    fn dims_with(m: crate::execgraph::sym::SymDimId, value: usize) -> DimValues {
        let mut dims = DimValues::new();
        dims.bind(m, value);
        dims
    }

    #[test]
    fn resolve_reshapes_and_caches() -> anyhow::Result<()> {
        let cx = test_context()?;
        let (executor, m) =
            dynamic_executor(&cx, |plan| AlgoConfig::of(plan.candidate()).unwrap())?;
        let hint = executor.problem.layouts(&dims_with(m, 64))?;
        let hint_plan = executor.resolve(&hint)?.plan.clone();
        assert!(Arc::ptr_eq(&hint_plan, &executor.hint_plan));
        assert!(executor.plans.lock().others.is_empty());

        let other = executor.problem.layouts(&dims_with(m, 1000))?;
        let resolved = executor.resolve(&other)?;
        assert!(!Arc::ptr_eq(&resolved.plan, &executor.hint_plan));
        assert_eq!(resolved.plan.desc().b.cols, 1000);
        assert_eq!(executor.plans.lock().others.len(), 1);
        // Cached on the second lookup.
        assert!(Arc::ptr_eq(&executor.resolve(&other)?, &resolved));
        assert_eq!(executor.plans.lock().others.len(), 1);
        Ok(())
    }

    #[test]
    fn resolve_keeps_tuned_algorithm_only_within_budget() -> anyhow::Result<()> {
        let cx = test_context()?;
        let (executor, m) = dynamic_executor(&cx, |plan| {
            AlgoConfig::of(plan.algorithms().last().unwrap()).unwrap()
        })?;
        let in_plan = |resolved: &Resolved| -> anyhow::Result<AlgoConfig> {
            let chosen = AlgoConfig::of(&resolved.algo)?;
            // Algorithms of a reshaped plan all fit the workspace budget.
            assert!(
                resolved
                    .plan
                    .algorithms()
                    .iter()
                    .any(|a| AlgoConfig::of(a).unwrap() == chosen)
            );
            Ok(chosen)
        };
        let resolved = executor.resolve(&executor.problem.layouts(&dims_with(m, 512))?)?;
        let chosen = in_plan(&resolved)?;
        if executor.config.find(&resolved.plan).is_some() {
            assert_eq!(chosen, executor.config);
        }

        // A tuned config the reshaped plan does not contain falls back to its candidate.
        let (mut executor, m) =
            dynamic_executor(&cx, |plan| AlgoConfig::of(plan.candidate()).unwrap())?;
        executor.config.splitk = 12345;
        let resolved = executor.resolve(&executor.problem.layouts(&dims_with(m, 300))?)?;
        assert_eq!(
            in_plan(&resolved)?,
            AlgoConfig::of(resolved.plan.candidate())?
        );
        Ok(())
    }

    #[test]
    fn plan_cache_is_bounded() -> anyhow::Result<()> {
        let cx = test_context()?;
        let (executor, m) =
            dynamic_executor(&cx, |plan| AlgoConfig::of(plan.candidate()).unwrap())?;
        for value in 1..=MAX_CACHED_PLANS + 10 {
            let layouts = executor.problem.layouts(&dims_with(m, value * 8))?;
            executor.resolve(&layouts)?;
        }
        let cache = executor.plans.lock();
        assert_eq!(cache.others.len(), MAX_CACHED_PLANS);
        // The oldest entries were evicted, the newest kept.
        let newest = executor
            .problem
            .layouts(&dims_with(m, (MAX_CACHED_PLANS + 10) * 8))?;
        let oldest = executor.problem.layouts(&dims_with(m, 8))?;
        assert!(cache.others.contains_key(&newest));
        assert!(!cache.others.contains_key(&oldest));
        Ok(())
    }

    #[test]
    fn leading_dimension_validation() -> anyhow::Result<()> {
        let map = MatMap {
            transposed: true,
            unit_rows: true,
        };
        // A [8, 16] read as col-major 16 x 8 with ld 4 < 16.
        let err = map
            .layout(
                DataType::BF16,
                Shape::new([8, 16]),
                Strides::new([4, 1]),
                "X",
            )
            .unwrap_err();
        assert!(err.to_string().contains("smaller than"), "{err}");
        // A single column may have any leading dimension.
        let layout = map.layout(
            DataType::BF16,
            Shape::new([1, 16]),
            Strides::new([0, 1]),
            "X",
        )?;
        assert_eq!(layout.ld, 16);
        Ok(())
    }
}
