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
        DevicePtrGuard, DevicePtrGuardMut, DeviceVisibleMemory, DeviceVisibleMemoryMut,
    },
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

/// Alignment in bytes required of every operand's base pointer, leading
/// dimension and batch stride.
const ALIGNMENT: usize = 16;

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
    fn layout(self, data_type: DataType, shape: Shape, strides: Strides) -> MatrixLayout {
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
        let layout = MatrixLayout::new(
            data_type,
            rows as u64,
            cols as u64,
            ld.max(rows).max(1) as i64,
        );
        if rank == 3 {
            layout.batched(shape[0] as i32, strides[0] as i64)
        } else {
            layout
        }
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

fn check_aligned_layout(
    layout: &SymLayout,
    ty: ElementType,
    map: MatMap,
    what: &str,
    dims: &SymDims,
) -> anyhow::Result<()> {
    let rank = layout.rank();
    ensure!(
        layout.offset_is_aligned(ALIGNMENT, ty, dims),
        "{what} starts at element offset {}, which is not provably {ALIGNMENT}-byte aligned",
        layout.offset.display(dims)
    );
    // The leading dimension is irrelevant if it only steps over one row/column.
    let ld_dim = rank - 2 + map.ld_dim();
    ensure!(
        layout.shape[ld_dim].is_one() || layout.stride_is_aligned(ld_dim, ALIGNMENT, ty, dims),
        "{what} has leading-dimension stride {} elements, which is not provably {ALIGNMENT}-byte aligned",
        layout.strides[ld_dim].display(dims)
    );
    if rank == 3 && !layout.shape[0].is_one() {
        ensure!(
            layout.stride_is_aligned(0, ALIGNMENT, ty, dims),
            "{what} has batch stride {} elements, which is not provably {ALIGNMENT}-byte aligned",
            layout.strides[0].display(dims)
        );
    }
    Ok(())
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
                let log2 = (ALIGNMENT * 8 / format.values_type().bits_per_element()).ilog2();
                ensure!(
                    block_dim.is_divisible_by_pow2(log2, dims),
                    "{format:?} {dim:?} output of shape {} needs its block dimension to be a provable multiple of {} for {ALIGNMENT}-byte alignment",
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
        let map = |op: &SymOperand, transposed: bool, what: &str| {
            let shape = op.shape();
            let strides = op.strides();
            let r = shape.rank();
            MatMap::new(
                transposed,
                [&shape[r - 2], &shape[r - 1]],
                [&strides[r - 2], &strides[r - 1]],
                what,
                dims,
            )
        };
        let a_map = map(&a, !swapped, "A")?;
        let b_map = map(&b, !swapped, "B")?;
        if let InputKind::Mx(format) = inputs {
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
            let log2 = (ALIGNMENT * 8 / format.values_type().bits_per_element()).ilog2();
            ensure!(
                k.is_divisible_by_pow2(log2, dims),
                "{format:?} inputs need K = {} to be a provable multiple of {} for {ALIGNMENT}-byte alignment",
                k.display(dims),
                1usize << log2
            );
        }
        for (op, data, map, name) in [(&a, a_data, a_map, "A"), (&b, b_data, b_map, "B")] {
            if let (SymOperand::Plain(layout), SymTensorData::Plain { ty, .. }) = (op, data) {
                check_aligned_layout(layout, *ty, map, name, dims)?;
            }
        }

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
        let d_map = map(&d, !swapped, "D")?;
        ensure!(
            d_map.op() == Operation::N,
            "output does not map to cuBLASLt's D"
        );
        if let SymOperand::Plain(layout) = &d {
            let MatmulOutput::Plain(ty) = out else {
                unreachable!()
            };
            check_aligned_layout(layout, ty, d_map, "D", dims)?;
        }

        let (c, c_map) = match self.c {
            None => (None, None),
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
                let c_map = map(&c, !swapped, "C")?;
                ensure!(
                    c_map.op() == Operation::N,
                    "C must be contiguous along N, i.e. not a transposed view"
                );
                let SymOperand::Plain(layout) = &c else {
                    unreachable!()
                };
                check_aligned_layout(layout, out_ty, c_map, "C", dims)?;
                (Some(c), Some(c_map))
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
        Ok(vec![OutputSpec::new(
            analysis.out.ty(),
            analysis.out_shape,
            Init::Uninit,
        )])
    }

    fn build(&self, bcx: &BuildCx<'_>) -> anyhow::Result<Box<dyn Executor>> {
        let analysis = self.analyze(bcx.graph)?;
        let combo = self.describe_types(bcx.graph, analysis.out);
        let problem = Arc::new(Problem::new(self, analysis)?);

        let hint_dims = bcx.graph.hint_values();
        let hint_layouts = problem.layouts(&hint_dims)?;
        let pref = MatmulPreference::default()
            .min_alignment_bytes(ALIGNMENT as u32)
            .aux_alignment_bytes(ALIGNMENT as u32)
            .requested_algo_count(16);
        let plan = bcx
            .cx
            .cublaslt()
            .build_plan(&problem.desc(&hint_layouts), &pref)
            .map_err(|e| {
                if e.status() == Some(Status::NotSupported) {
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
        let (_, executor) = bcx.tune_executor(&key, candidates, |config| {
            let algo = config.algorithm(&plan)?;
            Ok(MatmulExecutor::new(
                problem.clone(),
                plan.clone(),
                hint_layouts,
                algo,
            ))
        })?;
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

    /// The plan's algorithm with this configuration, or one rebuilt from it.
    fn algorithm(&self, plan: &MatmulPlan) -> anyhow::Result<Algorithm> {
        for algo in plan.algorithms() {
            if Self::of(algo).is_ok_and(|config| config == *self) {
                return Ok(*algo);
            }
        }
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
    alignment: usize,
}

/// Concrete cuBLASLt layouts of A', B', C' (if separate from D') and D'.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
struct Layouts {
    a: MatrixLayout,
    b: MatrixLayout,
    c: Option<MatrixLayout>,
    d: MatrixLayout,
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
        let our_a = {
            let (shape, strides) = an.a.eval(dims)?;
            an.a_map.layout(self.our_a_type, shape, strides)
        };
        let our_b = {
            let (shape, strides) = an.b.eval(dims)?;
            an.b_map.layout(self.our_b_type, shape, strides)
        };
        let (a, b) = if an.swapped {
            (our_a, our_b)
        } else {
            (our_b, our_a)
        };
        let (d_shape, d_strides) = an.d.eval(dims)?;
        let d = an.d_map.layout(self.d_type, d_shape, d_strides);
        let c = match (&an.c, an.c_map, self.c_type) {
            (Some(c), Some(map), Some(ty)) => {
                let (shape, strides) = c.eval(dims)?;
                Some(map.layout(ty, shape, strides))
            }
            // An unread C with D's shape, packed.
            (None, _, Some(ty)) => Some(MatrixLayout {
                data_type: ty,
                ld: d.rows as i64,
                batch_stride: (d.rows * d.cols) as i64,
                ..d
            }),
            _ => None,
        };
        Ok(Layouts { a, b, c, d })
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
            alignment: ALIGNMENT,
        }
    }
}

/// A plan for one concrete problem shape, with the algorithm to run.
struct Resolved {
    plan: Arc<MatmulPlan>,
    algo: Algorithm,
}

struct MatmulExecutor {
    problem: Arc<Problem>,
    hint_plan: Arc<MatmulPlan>,
    algo: Algorithm,
    plans: Mutex<HashMap<Layouts, Arc<Resolved>>>,
}

impl MatmulExecutor {
    fn new(
        problem: Arc<Problem>,
        hint_plan: Arc<MatmulPlan>,
        hint_layouts: Layouts,
        algo: Algorithm,
    ) -> Self {
        let resolved = Arc::new(Resolved {
            plan: hint_plan.clone(),
            algo,
        });
        Self {
            problem,
            hint_plan,
            algo,
            plans: Mutex::new(HashMap::from([(hint_layouts, resolved)])),
        }
    }

    /// The plan for a concrete shape, reshaping the hint plan on first use.
    fn resolve(&self, layouts: &Layouts) -> anyhow::Result<Arc<Resolved>> {
        if let Some(resolved) = self.plans.lock().get(layouts) {
            return Ok(resolved.clone());
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
        let algo = match plan.check(&self.algo) {
            Ok(check) => {
                let mut algo = self.algo;
                algo.workspace_size = check.workspace_size;
                algo
            }
            Err(_) => *plan.candidate(),
        };
        let resolved = Arc::new(Resolved {
            plan: Arc::new(plan),
            algo,
        });
        self.plans.lock().insert(*layouts, resolved.clone());
        Ok(resolved)
    }
}

impl Executor for MatmulExecutor {
    fn scratch_bytes(&self, dims: &DimValues) -> usize {
        let Ok(layouts) = self.problem.layouts(dims) else {
            return 0;
        };
        if layouts.is_empty() {
            return 0;
        }
        self.resolve(&layouts)
            .map_or(0, |resolved| resolved.algo.workspace_size)
    }

    fn execute(&self, io: &mut OpIo<'_>, stream: &Stream, _cx: &Context) -> anyhow::Result<()> {
        let problem = &*self.problem;
        let layouts = problem.layouts(io.dims)?;
        if layouts.is_empty() {
            return Ok(());
        }
        let resolved = self.resolve(&layouts)?;
        let (plan, algo) = (&*resolved.plan, &resolved.algo);

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

        let unread_c;
        let mut ops = plan.operands();
        ops.alpha(Scalar::F32(problem.alpha))
            .beta(Scalar::F32(problem.beta));
        match a {
            TensorRef::Plain(t) => ops.a(t.storage()),
            TensorRef::Mx(t) => ops
                .a(t.values().storage())
                .a_scale(t.scale_factors().storage()),
        };
        match b {
            TensorRef::Plain(t) => ops.b(t.storage()),
            TensorRef::Mx(t) => ops
                .b(t.values().storage())
                .b_scale(t.scale_factors().storage()),
        };
        if let Some(c) = &c {
            ops.c(c.plain().storage());
        }
        match &mut outputs[0] {
            TensorRefMut::Plain(d) => {
                ops.d(d.storage_mut());
            }
            TensorRefMut::Mx(d) => {
                let (values, scales) = d.values_and_scale_factors_mut();
                // beta is zero, so cuBLASLt never reads C; any aligned pointer will do.
                let ptr = unsafe { values.storage_mut().access_mut_on(stream)?.ptr() };
                let c = layouts.c.expect("block-scaled outputs have a C layout");
                unread_c = RawMemory {
                    ptr,
                    len: c.min_len_bytes().unwrap_or(0),
                };
                ops.c(&unread_c)
                    .d(values.storage_mut())
                    .d_out_scale(scales.storage());
            }
        }

        match scratch {
            Some(scratch) => unsafe { plan.execute_with(algo, stream, &ops, scratch)? },
            None => {
                ensure!(
                    algo.workspace_size == 0,
                    "matmul needs {} bytes of scratch memory, but none was given",
                    algo.workspace_size
                );
                let mut empty = RawMemory { ptr: 0, len: 0 };
                unsafe { plan.execute_with(algo, stream, &ops, &mut empty)? }
            }
        }
        Ok(())
    }
}

/// Device memory given by a raw pointer, without stream-ordered access tracking.
/// Only for memory cuBLASLt never touches.
struct RawMemory {
    ptr: CUdeviceptr_v2,
    len: usize,
}

struct RawGuard(CUdeviceptr_v2);

impl DevicePtrGuard<'_> for RawGuard {
    unsafe fn ptr(&self) -> CUdeviceptr_v2 {
        self.0
    }
}

impl DevicePtrGuardMut<'_> for RawGuard {
    unsafe fn ptr(&mut self) -> CUdeviceptr_v2 {
        self.0
    }
}

impl DeviceVisibleMemory for RawMemory {
    type Guard<'a> = RawGuard;

    fn len(&self) -> usize {
        self.len
    }

    fn access_on<'a>(&'a self, _stream: &'a Stream) -> skycuda::error::Result<RawGuard> {
        Ok(RawGuard(self.ptr))
    }
}

impl DeviceVisibleMemoryMut for RawMemory {
    type GuardMut<'a> = RawGuard;

    fn access_mut_on<'a>(&'a mut self, _stream: &'a Stream) -> skycuda::error::Result<RawGuard> {
        Ok(RawGuard(self.ptr))
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

    #[test]
    fn plans_build() -> anyhow::Result<()> {
        let cx = Context::new(
            0,
            Default::default(),
            skycuda::host_pinned_memory_pool::HostPinnedMemoryPoolOptions {
                max_reserved_free_bytes: 1 << 20,
            },
        )?;
        let pref = MatmulPreference::default()
            .min_alignment_bytes(ALIGNMENT as u32)
            .aux_alignment_bytes(ALIGNMENT as u32);
        let (problem, l) = setup(GraphBuilder::new(), |gb| {
            Ok(plain(
                &gb.input("a", BF16, [64, 64])?,
                &gb.input("b", BF16, [64, 64])?,
            ))
        })?;
        let plan = cx.cublaslt().build_plan(&problem.desc(&l), &pref)?;
        let algo = plan.candidate();
        let config = AlgoConfig::of(algo)?;
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
        match cx.cublaslt().build_plan(&problem.desc(&l), &pref) {
            Ok(_) => {}
            Err(e) if cx.cuda().compute_capability_major() < 10 => {
                assert_eq!(e.status(), Some(Status::NotSupported), "{e}");
            }
            Err(e) => return Err(e.into()),
        }
        Ok(())
    }
}
