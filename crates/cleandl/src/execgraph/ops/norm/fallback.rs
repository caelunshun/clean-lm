//! TileIR fallback norm kernel.

use super::{F32, NormFwd, NormOutput, NormType};
use crate::{
    context::Context,
    execgraph::{
        BuildCx, Executor, GraphInfo, OpIo, SymTensorId, TensorRefMut,
        ops::pointwise::{
            codegen::{ArrayParam, dtype_of},
            mx,
        },
        sym::SymShape,
    },
    mx_tensor::{MxDim, MxFormat},
};
use anyhow::{bail, ensure};
use kernel_build_util::{
    ArgBundle, LoadedKernel, Param, PtrArg,
    tileir::{Args, KernelBuilder, KernelSpec},
};
use serde::{Deserialize, Serialize};
use skycuda::stream::Stream;
use skytens_core::data_type::ElementType;
use std::sync::Arc;
use tileir_builder::{
    ArrayOpts, Bool, DType, DynArray, F4E2M1, F8E4M3, KernelOpts, ModuleBuilder, Ptr, RoundingMode,
    Tile, bid, for_range, for_range_fold, iota, num_blocks, where_,
};

const KERNEL_NAME: &str = "norm_fwd";

const MAX_TILE_ELEMENTS: u32 = 1 << 14;
const TARGET_TILE_ELEMENTS: u32 = 1 << 12;
const MAX_GRID_DIM: u64 = (1 << 24) - 1;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(super) struct TileShape {
    rows: u32,
    cols: u32,
    /// Whether every row fits in one tile.
    single_chunk: bool,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize)]
struct ArrayFacts {
    ty: ElementType,
    unit_stride: bool,
}

/// Identifies a compiled kernel, apart from its tile shape.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize)]
struct Facts {
    norm_type: NormType,
    x: ArrayFacts,
    scale: Option<ArrayFacts>,
    bias: Option<ArrayFacts>,
    output: NormOutput,
    save_stats: bool,
}

#[derive(Serialize)]
struct KernelKey<'a> {
    facts: &'a Facts,
    tile: TileShape,
}

#[derive(Serialize)]
struct TunePayload<'a> {
    facts: &'a Facts,
    rows_hint: usize,
    cols_hint: usize,
    cols_fixed: bool,
}

impl Facts {
    fn new(op: &NormFwd, graph: &GraphInfo) -> Self {
        let array = |id: SymTensorId, axis: usize| {
            let layout = graph.layout(id).unwrap();
            ArrayFacts {
                ty: graph.ty(id).element_type().unwrap(),
                unit_stride: layout.strides[axis].is_one(),
            }
        };
        Self {
            norm_type: op.norm_type,
            x: array(op.x, 1),
            scale: op.scale.map(|id| array(id, 0)),
            bias: op.bias.map(|id| array(id, 0)),
            output: op.output,
            save_stats: op.save_stats,
        }
    }

    /// Smallest tile `(rows, cols)` the outputs allow.
    fn min_tile(&self) -> (u32, u32) {
        match self.output {
            NormOutput::Plain(_) => (1, 16),
            NormOutput::Mxfp8Rowwise => (1, MxFormat::Mxfp8.block_size() as u32),
            NormOutput::Mxfp8RowwiseAndColwise => (
                MxFormat::Mxfp8.block_size() as u32,
                MxFormat::Mxfp8.block_size() as u32,
            ),
            NormOutput::Nvfp4Rowwise => (1, MxFormat::Nvfp4.block_size() as u32),
        }
    }
}

/// Tile shapes to choose from, the heuristic default first.
fn candidates(facts: &Facts, cols_hint: usize, cols_fixed: bool) -> Vec<TileShape> {
    let (min_rows, min_cols) = facts.min_tile();
    let full = u32::try_from(cols_hint.next_power_of_two())
        .unwrap_or(u32::MAX)
        .max(min_cols);
    let rows_for = |cols: u32| (TARGET_TILE_ELEMENTS / cols).clamp(min_rows, 64.max(min_rows));
    let mut out = Vec::new();
    if cols_fixed && full.saturating_mul(min_rows) <= MAX_TILE_ELEMENTS {
        let default = rows_for(full);
        out.push(TileShape {
            rows: default,
            cols: full,
            single_chunk: true,
        });
        let mut rows = min_rows;
        while rows * full <= MAX_TILE_ELEMENTS && rows <= 64 {
            if rows != default {
                out.push(TileShape {
                    rows,
                    cols: full,
                    single_chunk: true,
                });
            }
            rows *= 2;
        }
    }
    for cols in [1024, 512, 2048, 256] {
        let cols = u32::max(cols, min_cols);
        if cols_fixed && cols >= full && !out.is_empty() {
            continue;
        }
        let rows = rows_for(cols);
        for rows in [rows, rows * 2] {
            let tile = TileShape {
                rows,
                cols,
                single_chunk: false,
            };
            if rows * cols <= MAX_TILE_ELEMENTS && !out.contains(&tile) {
                out.push(tile);
            }
        }
    }
    out
}

pub(super) fn build(op: &NormFwd, bcx: &BuildCx<'_>) -> anyhow::Result<Box<dyn Executor>> {
    let graph = bcx.graph;
    let facts = Facts::new(op, graph);
    let shape = graph.shape(op.x);
    let cols_hint = shape[1].eval_hint(graph.dims());
    let cols_fixed = shape[1].as_fixed().is_some();
    let key = bcx.tune_key(&TunePayload {
        facts: &facts,
        rows_hint: shape[0].eval_hint(graph.dims()),
        cols_hint,
        cols_fixed,
    })?;
    let (_, executor) =
        bcx.tune_executor(&key, candidates(&facts, cols_hint, cols_fixed), |&tile| {
            FallbackExecutor::new(op, bcx.cx, &facts, tile, shape.clone())
        })?;
    Ok(Box::new(executor))
}

#[derive(Clone, Copy, Debug)]
enum ScaleParam {
    Ue8m0(Param<Ptr<u8>>),
    E4m3(Param<Ptr<F8E4M3>>),
}

#[derive(Clone, Copy, Debug)]
enum Stat {
    Mean,
    RInv,
}

#[derive(Clone, Copy, Debug)]
enum OutputParam {
    Plain(ElementType, ArrayParam),
    Mx {
        format: MxFormat,
        dim: MxDim,
        values: ArrayParam,
        scales: ScaleParam,
    },
    Stat(Stat, ArrayParam),
}

#[derive(Clone, Debug)]
struct Params {
    rows: Param<i64>,
    cols: Param<i64>,
    eps: Param<f32>,
    x: ArrayParam,
    scale: Option<ArrayParam>,
    bias: Option<ArrayParam>,
    outputs: Vec<OutputParam>,
}

fn array_opts(rank: usize, unit_stride_axis: Option<usize>) -> ArrayOpts {
    let opts = ArrayOpts::rank(rank).index_i64();
    match unit_stride_axis {
        Some(axis) => opts.static_stride(axis, 1),
        None => opts,
    }
}

impl Params {
    fn declare(kb: &mut KernelBuilder, facts: &Facts) -> Self {
        let rows = kb.param("rows");
        let cols = kb.param("cols");
        let eps = kb.param("eps");
        let x = ArrayParam::declare(
            kb,
            "x".into(),
            facts.x.ty,
            array_opts(2, facts.x.unit_stride.then_some(1)),
        );
        let mut vector = |name: &str, f: Option<ArrayFacts>| {
            f.map(|f| {
                ArrayParam::declare(
                    kb,
                    name.into(),
                    f.ty,
                    array_opts(1, f.unit_stride.then_some(0)),
                )
            })
        };
        let scale = vector("scale", facts.scale);
        let bias = vector("bias", facts.bias);

        let mx_output = |kb: &mut KernelBuilder, format: MxFormat, dim: MxDim| {
            let name = match dim {
                MxDim::Rowwise => "y_row",
                MxDim::Colwise => "y_col",
            };
            // Colwise values are stored transposed.
            let unit_axis = match dim {
                MxDim::Rowwise => 1,
                MxDim::Colwise => 0,
            };
            OutputParam::Mx {
                format,
                dim,
                values: ArrayParam::declare(
                    kb,
                    format!("{name}_values"),
                    format.values_type(),
                    array_opts(2, Some(unit_axis)),
                ),
                scales: match format {
                    MxFormat::Mxfp8 => ScaleParam::Ue8m0(kb.param(format!("{name}_scales"))),
                    MxFormat::Nvfp4 => ScaleParam::E4m3(kb.param(format!("{name}_scales"))),
                },
            }
        };
        let mut outputs = match facts.output {
            NormOutput::Plain(ty) => vec![OutputParam::Plain(
                ty,
                ArrayParam::declare(kb, "y".into(), ty, array_opts(2, Some(1))),
            )],
            NormOutput::Mxfp8Rowwise => vec![mx_output(kb, MxFormat::Mxfp8, MxDim::Rowwise)],
            NormOutput::Mxfp8RowwiseAndColwise => vec![
                mx_output(kb, MxFormat::Mxfp8, MxDim::Rowwise),
                mx_output(kb, MxFormat::Mxfp8, MxDim::Colwise),
            ],
            NormOutput::Nvfp4Rowwise => vec![mx_output(kb, MxFormat::Nvfp4, MxDim::Rowwise)],
        };
        if facts.save_stats {
            let mut stat = |stat, name: &str| {
                OutputParam::Stat(
                    stat,
                    ArrayParam::declare(kb, name.into(), F32, array_opts(1, Some(0))),
                )
            };
            if facts.norm_type == NormType::LayerNorm {
                outputs.push(stat(Stat::Mean, "mean"));
            }
            outputs.push(stat(Stat::RInv, "r_inv"));
        }
        Self {
            rows,
            cols,
            eps,
            x,
            scale,
            bias,
            outputs,
        }
    }
}

fn define(module: &mut ModuleBuilder, facts: &Facts, tile: TileShape) -> KernelSpec {
    let mut kb = KernelBuilder::new();
    let params = Params::declare(&mut kb, facts);
    kb.define(module, KERNEL_NAME, KernelOpts::new(), |args| {
        body(args, &params, facts, tile)
    })
}

const NE: RoundingMode = RoundingMode::NearestEven;

fn load_f32(array: &DynArray, index: Vec<Tile<i64>>, shape: Vec<u32>) -> Tile<f32> {
    array.load(index, shape).astype(&DType::F32, NE).downcast()
}

fn body(args: &Args, params: &Params, facts: &Facts, tile: TileShape) {
    let (tr, tc) = (tile.rows, tile.cols);
    let full = vec![tr, tc];
    let rows = args.get(params.rows);
    let cols = args.get(params.cols);

    let row_tile =
        bid(0).astype::<i64>(NE) + bid(1).astype::<i64>(NE) * num_blocks(0).astype::<i64>(NE);
    let row_coords = iota::<i64>(tr) + row_tile * i64::from(tr);
    let row_ok = row_coords.lt(rows).reshape(vec![tr, 1]);
    let col_coords = |j: Tile<i64>| iota::<i64>(tc) + j * i64::from(tc);
    let col_ok = |j: Tile<i64>| col_coords(j).lt(cols).reshape(vec![1, tc]);
    let in_bounds =
        |j: Tile<i64>| row_ok.broadcast_to(full.clone()) & col_ok(j).broadcast_to(full.clone());

    let x = params.x.get(args);
    let load_x = |j: Tile<i64>| load_f32(&x, vec![row_tile, j], full.clone());
    let cached = tile.single_chunk.then(|| load_x(Tile::scalar(0)));
    let x_chunk = |j: Tile<i64>| cached.unwrap_or_else(|| load_x(j));
    let num_chunks = (cols + i64::from(tc - 1)) / i64::from(tc);

    // Out-of-bounds elements load as zero, so `f` must map zero to zero there.
    let row_sum = |f: &dyn Fn(Tile<f32>, Tile<i64>) -> Tile<f32>| -> Tile<f32> {
        if tile.single_chunk {
            let j = Tile::scalar(0);
            f(x_chunk(j), j).sum(1)
        } else {
            let zero = Tile::<f32>::scalar(0.0).broadcast_to(vec![tr]);
            for_range_fold(0i64, num_chunks, zero, |j, acc| {
                acc + f(x_chunk(j), j).sum(1)
            })
        }
    };
    let recip_cols = Tile::<f32>::scalar(1.0) / cols.astype::<f32>(NE);
    let eps = args.get(params.eps);
    let (mean, r_inv) = match facts.norm_type {
        NormType::RmsNorm => {
            let mean_square = row_sum(&|x, _| x * x) * recip_cols;
            (None, (mean_square + eps).rsqrt())
        }
        NormType::LayerNorm => {
            let mean = row_sum(&|x, _| x) * recip_cols;
            let centered_mean = mean.reshape(vec![tr, 1]);
            let var = row_sum(&|x, j| {
                let d = where_(in_bounds(j), x - centered_mean, 0.0f32);
                d * d
            }) * recip_cols;
            (Some(mean), (var + eps).rsqrt())
        }
    };

    for output in &params.outputs {
        if let OutputParam::Stat(stat, array) = output {
            let value = match stat {
                Stat::Mean => mean.unwrap(),
                Stat::RInv => r_inv,
            };
            array.get(args).store(vec![row_tile], value.erase());
        }
    }

    let load_vector = |param: Option<ArrayParam>, j: Tile<i64>| {
        param.map(|p| load_f32(&p.get(args), vec![j], vec![tc]).reshape(vec![1, tc]))
    };
    let emit = |j: Tile<i64>| {
        let x = x_chunk(j);
        let centered = match mean {
            Some(mean) => x - mean.reshape(vec![tr, 1]),
            None => x,
        };
        let mut y = centered * r_inv.reshape(vec![tr, 1]);
        if let Some(scale) = load_vector(params.scale, j) {
            y = y * scale;
        }
        if let Some(bias) = load_vector(params.bias, j) {
            y = y + bias;
        }
        for output in &params.outputs {
            match *output {
                OutputParam::Plain(ty, array) => {
                    array
                        .get(args)
                        .store(vec![row_tile, j], y.erase().astype(&dtype_of(ty), NE));
                }
                OutputParam::Mx {
                    format,
                    dim,
                    values,
                    scales,
                } => {
                    let magnitude = where_(in_bounds(j), y.abs(), 0.0f32);
                    let quantized = MxStore {
                        format,
                        dim,
                        tile,
                        rows,
                        cols,
                        row_tile,
                        row_coords,
                        col_coords: col_coords(j),
                    };
                    quantized.store(args, y, magnitude, values, scales, j);
                }
                OutputParam::Stat(..) => {}
            }
        }
    };
    if tile.single_chunk {
        emit(Tile::scalar(0));
    } else {
        for_range(0i64, num_chunks, emit);
    }
}

/// Quantization and store of one `[tile rows, tile cols]` chunk of a block-scaled output.
struct MxStore {
    format: MxFormat,
    dim: MxDim,
    tile: TileShape,
    rows: Tile<i64>,
    cols: Tile<i64>,
    row_tile: Tile<i64>,
    /// `[tile rows]`
    row_coords: Tile<i64>,
    /// `[tile cols]`
    col_coords: Tile<i64>,
}

impl MxStore {
    fn store(
        &self,
        args: &Args,
        y: Tile<f32>,
        magnitude: Tile<f32>,
        values: ArrayParam,
        scales: ScaleParam,
        j: Tile<i64>,
    ) {
        let (tr, tc) = (self.tile.rows, self.tile.cols);
        let bs = self.format.block_size() as u32;
        let bs64 = i64::from(bs);
        let full = vec![tr, tc];
        // Blocks as `[outer, block, inner]`, reduced over the middle axis.
        let (blocked, scale_shape) = match self.dim {
            MxDim::Rowwise => (vec![tr, tc / bs, bs], vec![tr, tc / bs]),
            MxDim::Colwise => (vec![tr / bs, bs, tc], vec![tr / bs, tc]),
        };
        let reduce_axis = match self.dim {
            MxDim::Rowwise => 2,
            MxDim::Colwise => 1,
        };
        let amax = magnitude.reshape(blocked.clone()).max(reduce_axis);
        let expand = |descale: Tile<f32>| {
            descale
                .expand_dims(reduce_axis)
                .broadcast_to(blocked.clone())
                .reshape(full.clone())
        };

        // Scale factor coordinates `(non-block index, block index)` of each block.
        let (sf_row, sf_col, num_blocks, mask): (Tile<i64>, Tile<i64>, Tile<i64>, Bool) =
            match self.dim {
                MxDim::Rowwise => {
                    let block = iota::<i64>(tc / bs) + j * i64::from(tc / bs);
                    (
                        self.row_coords.reshape(vec![tr, 1]),
                        block.reshape(vec![1, tc / bs]),
                        self.cols / bs64,
                        self.row_coords
                            .lt(self.rows)
                            .reshape(vec![tr, 1])
                            .broadcast_to(scale_shape.clone())
                            & (block * bs64)
                                .lt(self.cols)
                                .reshape(vec![1, tc / bs])
                                .broadcast_to(scale_shape.clone()),
                    )
                }
                MxDim::Colwise => {
                    let block = iota::<i64>(tr / bs) + self.row_tile * i64::from(tr / bs);
                    (
                        self.col_coords.reshape(vec![1, tc]),
                        block.reshape(vec![tr / bs, 1]),
                        self.rows / bs64,
                        (block * bs64)
                            .lt(self.rows)
                            .reshape(vec![tr / bs, 1])
                            .broadcast_to(scale_shape.clone())
                            & self
                                .col_coords
                                .lt(self.cols)
                                .reshape(vec![1, tc])
                                .broadcast_to(scale_shape.clone()),
                    )
                }
            };
        let index = mx::scale_factor_index(sf_row, sf_col, num_blocks).broadcast_to(scale_shape);

        let values = values.get(args);
        let tile_index = vec![self.row_tile, j];
        match (self.format, scales) {
            (MxFormat::Mxfp8, ScaleParam::Ue8m0(p)) => {
                let (scale, descale) = mx::mxfp8_scale(amax);
                let q = mx::clamp(y * expand(descale), mx::E4M3_MAX).astype::<F8E4M3>(NE);
                values.store(tile_index, q.erase());
                args.get(p).offset(index).store(scale, Some(mask));
            }
            (MxFormat::Nvfp4, ScaleParam::E4m3(p)) => {
                let (scale, descale) = mx::nvfp4_scale(amax);
                let q = mx::clamp(y * expand(descale), mx::E2M1_MAX).astype::<F4E2M1>(NE);
                values.store(tile_index, q.erase());
                args.get(p).offset(index).store(scale, Some(mask));
            }
            _ => unreachable!("scale factor type does not match the format"),
        }
    }
}

pub(super) struct FallbackExecutor {
    kernel: Arc<LoadedKernel>,
    params: Params,
    tile: TileShape,
    x_shape: SymShape,
    eps: f32,
}

impl FallbackExecutor {
    fn new(
        op: &NormFwd,
        cx: &Context,
        facts: &Facts,
        tile: TileShape,
        x_shape: SymShape,
    ) -> anyhow::Result<Self> {
        let key = KernelKey { facts, tile };
        let kernel = crate::execgraph::jit::get_or_compile_kernel(cx, &key, KERNEL_NAME, |m| {
            define(m, facts, tile)
        })?;
        Ok(Self {
            kernel,
            params: Params::declare(&mut KernelBuilder::new(), facts),
            tile,
            x_shape,
            eps: op.eps,
        })
    }

    fn grid(&self, rows: usize) -> anyhow::Result<(u32, u32, u32)> {
        let tiles = (rows as u64).div_ceil(u64::from(self.tile.rows));
        let y = tiles.div_ceil(MAX_GRID_DIM);
        let x = tiles.div_ceil(y);
        ensure!(
            y <= MAX_GRID_DIM,
            "norm_fwd: {rows} rows are too many for the grid"
        );
        Ok((x as u32, y as u32, 1))
    }
}

/// Strides of a bound view, as `i64`, with size-1 dimensions given unit strides.
fn view_strides(shape: &[usize], strides: &[usize]) -> anyhow::Result<Vec<i64>> {
    shape
        .iter()
        .zip(strides)
        .map(|(&n, &s)| Ok(if n == 1 { 1 } else { i64::try_from(s)? }))
        .collect()
}

impl Executor for FallbackExecutor {
    fn execute(&self, io: &mut OpIo<'_>, stream: &Stream, _cx: &Context) -> anyhow::Result<()> {
        let expected = self.x_shape.eval(io.dims)?;
        let (rows, cols) = (expected[0], expected[1]);
        if rows == 0 {
            return Ok(());
        }
        ensure!(cols > 0, "norm_fwd: cannot normalize empty rows");
        ensure!(
            !self.tile.single_chunk || cols <= self.tile.cols as usize,
            "norm_fwd: {cols} columns exceed the tile width {}",
            self.tile.cols
        );
        let grid = self.grid(rows)?;
        let (rows_i, cols_i) = (i64::try_from(rows)?, i64::try_from(cols)?);

        let (inputs, outputs, _) = io.split();
        let x = inputs[0].as_ref();
        let x = x.plain();
        ensure!(
            x.shape() == expected,
            "norm_fwd: expected input of shape {expected:?}, got {:?}",
            x.shape()
        );
        let mut vectors = Vec::new();
        for (k, param) in [self.params.scale, self.params.bias]
            .into_iter()
            .enumerate()
        {
            let Some(param) = param else { continue };
            let index = 1 + (k == 1 && self.params.scale.is_some()) as usize;
            vectors.push((param, inputs[index].as_ref()));
        }

        let mut args = ArgBundle::new();
        args.set(self.params.rows, rows_i);
        args.set(self.params.cols, cols_i);
        args.set(self.params.eps, self.eps);
        let x_strides = view_strides(&[rows, cols], &[x.strides()[0], x.strides()[1]])?;
        self.params
            .x
            .bind(&mut args, x.storage(), &[rows_i, cols_i], &x_strides);
        for (param, vector) in &vectors {
            let view = vector.plain();
            ensure!(
                view.shape().num_dims() == 1 && view.shape()[0] == cols,
                "norm_fwd: expected a vector of {cols} elements, got shape {:?}",
                view.shape()
            );
            let strides = view_strides(&[cols], &[view.strides()[0]])?;
            param.bind(&mut args, view.storage(), &[cols_i], &strides);
        }

        ensure!(
            outputs.len() == self.params.outputs.len(),
            "norm_fwd: output count mismatch"
        );
        for (i, (output, param)) in outputs.iter_mut().zip(&self.params.outputs).enumerate() {
            match (param, output) {
                (OutputParam::Plain(_, array), TensorRefMut::Plain(view)) => {
                    ensure!(
                        view.shape() == expected && view.is_packed_row_major(),
                        "norm_fwd: output {i} must be packed with shape {expected:?}"
                    );
                    array.bind_mut(
                        &mut args,
                        view.storage_mut(),
                        &[rows_i, cols_i],
                        &[cols_i, 1],
                    );
                }
                (OutputParam::Stat(_, array), TensorRefMut::Plain(view)) => {
                    ensure!(
                        view.shape().num_dims() == 1 && view.shape()[0] == rows,
                        "norm_fwd: output {i} must have shape [{rows}]"
                    );
                    array.bind_mut(&mut args, view.storage_mut(), &[rows_i], &[1]);
                }
                (
                    OutputParam::Mx {
                        format,
                        dim,
                        values,
                        scales,
                    },
                    TensorRefMut::Mx(mx),
                ) => {
                    ensure!(
                        mx.shape() == expected && mx.format() == *format && mx.mx_dim() == *dim,
                        "norm_fwd: output {i} must be {format:?} {dim:?} with shape {expected:?}"
                    );
                    let strides = match dim {
                        MxDim::Rowwise => [cols_i, 1],
                        MxDim::Colwise => [1, rows_i],
                    };
                    let (values_view, scales_view) = mx.values_and_scale_factors_mut();
                    values.bind_mut(
                        &mut args,
                        values_view.storage_mut(),
                        &[rows_i, cols_i],
                        &strides,
                    );
                    match *scales {
                        ScaleParam::Ue8m0(p) => {
                            args.set(p, PtrArg::new_mut(scales_view.storage_mut()))
                        }
                        ScaleParam::E4m3(p) => {
                            args.set(p, PtrArg::new_mut(scales_view.storage_mut()))
                        }
                    }
                }
                _ => bail!("norm_fwd: output {i} has an unexpected tensor kind"),
            }
        }

        // Soundness: arguments match the declared parameters, and every access is in
        // bounds of the bound views.
        unsafe { self.kernel.launch(args, grid, stream)? };
        Ok(())
    }
}
