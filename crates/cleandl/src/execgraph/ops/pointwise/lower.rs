//! Iteration space inference and lowering to a collapsed, tiled iteration space.

use super::ir::{PwIr, PwOutput, Reduce, ValueTy};
use crate::{
    execgraph::{
        Init, OutputSpec, TensorTy,
        sym::{Monomial, SymDims, SymLayout, SymShape},
    },
    mx_tensor::{MxDim, MxFormat},
};
use anyhow::{anyhow, bail, ensure};
use serde::{Deserialize, Serialize};
use skytens_core::data_type::ElementType;

#[derive(Clone, Copy, Debug)]
pub struct OperandDesc<'a> {
    pub layout: &'a SymLayout,
    pub ty: ElementType,
}

/// Static facts about the lowered iteration space; part of the kernel cache key.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Facts {
    /// Rank of the lowered iteration space (at least 1).
    pub rank: usize,
    pub inputs: Vec<InputFacts>,
    pub outputs: Vec<OutputFacts>,
    /// `(iteration axis, lowered axis)` for each axis read by `Index`; `None`
    /// for axes of size one.
    pub index_axes: Vec<(usize, Option<usize>)>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct InputFacts {
    pub ty: ElementType,
    /// Whether the kernel reads the input at all.
    pub live: bool,
    /// Lowered axes along which the input varies; it is broadcast along the others.
    pub axes: Vec<usize>,
    /// Per entry of `axes`, whether the stride is provably one.
    pub unit_stride: Vec<bool>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OutputFacts {
    Elementwise {
        ty: ElementType,
        unit_stride: Vec<bool>,
    },
    Reduce {
        op: Reduce,
        /// Lowered axes that are not reduced.
        kept: Vec<usize>,
    },
    Mx {
        format: MxFormat,
        dim: MxDim,
        /// Of the values array, per lowered axis.
        unit_stride: Vec<bool>,
        /// Lowered axis along which blocks run.
        block_axis: usize,
    },
}

/// Symbolic sizes and strides of the lowered iteration space, evaluated at execution.
#[derive(Clone, Debug)]
pub struct Plan {
    pub sizes: Vec<Monomial>,
    /// Per input, the stride along each entry of [`InputFacts::axes`].
    pub inputs: Vec<Vec<Monomial>>,
    pub outputs: Vec<OutputPlan>,
    /// Layouts the bound operands must have; the kernel addresses them from the
    /// start of their storage.
    pub input_layouts: Vec<SymLayout>,
    /// Shapes of the (packed) outputs.
    pub output_shapes: Vec<SymShape>,
}

#[derive(Clone, Debug)]
pub enum OutputPlan {
    Elementwise {
        strides: Vec<Monomial>,
    },
    /// Strides of the kept axes in the output.
    Reduce {
        strides: Vec<Monomial>,
    },
    Mx {
        values_strides: Vec<Monomial>,
        /// Contribution of each lowered axis but the last to the row of the 2D view.
        row_strides: Vec<Monomial>,
        /// Size of the block dimension of the 2D view.
        block_dim: Monomial,
    },
}

#[derive(Clone, Debug)]
pub struct Lowering {
    pub output_specs: Vec<OutputSpec>,
    pub facts: Facts,
    pub plan: Plan,
}

/// Broadcasts operand shapes, right-aligned. Each axis must have provably equal
/// sizes in all operands, or size one.
pub fn iteration_shape(shapes: &[&SymShape], dims: &SymDims) -> anyhow::Result<SymShape> {
    let rank = shapes.iter().map(|s| s.rank()).max().unwrap_or(0);
    let mut iter = Vec::with_capacity(rank);
    for axis in 0..rank {
        let mut size: Option<(usize, &Monomial)> = None;
        for (k, shape) in shapes.iter().enumerate() {
            let Some(d) = (axis + shape.rank()).checked_sub(rank).map(|i| &shape[i]) else {
                continue;
            };
            if d.is_one() {
                continue;
            }
            match size {
                None => size = Some((k, d)),
                Some((other, s)) => ensure!(
                    s == d,
                    "cannot broadcast operand {k} with shape {} against operand {other} with shape {}: \
                     sizes {} and {} on iteration axis {axis} are not provably equal",
                    shape.display(dims),
                    shapes[other].display(dims),
                    d.display(dims),
                    s.display(dims),
                ),
            }
        }
        iter.push(size.map_or_else(Monomial::one, |(_, d)| d.clone()));
    }
    Ok(SymShape::new(iter))
}

fn overflow() -> anyhow::Error {
    anyhow!("pointwise iteration space overflows")
}

fn row_major(sizes: &[&Monomial]) -> anyhow::Result<Vec<Monomial>> {
    let mut strides = vec![Monomial::one(); sizes.len()];
    for i in (0..sizes.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1]
            .checked_mul(sizes[i + 1])
            .ok_or_else(overflow)?;
    }
    Ok(strides)
}

fn product<'a>(sizes: impl IntoIterator<Item = &'a Monomial>) -> anyhow::Result<Monomial> {
    sizes.into_iter().try_fold(Monomial::one(), |acc, s| {
        acc.checked_mul(s).ok_or_else(overflow)
    })
}

/// A tensor addressed by the iteration space: a stride per iteration axis, or
/// `None` where it is broadcast.
type Participant = Vec<Option<Monomial>>;

pub fn lower(ir: &PwIr, operands: &[OperandDesc<'_>], dims: &SymDims) -> anyhow::Result<Lowering> {
    let input_tys = operands
        .iter()
        .enumerate()
        .map(|(k, op)| ValueTy::of_element(op.ty).map_err(|e| anyhow!("operand {k}: {e}")))
        .collect::<anyhow::Result<Vec<_>>>()?;
    ir.check(&input_tys)?;
    ensure!(
        !operands.is_empty(),
        "pointwise op needs at least one input"
    );

    let shapes: Vec<&SymShape> = operands.iter().map(|op| &op.layout.shape).collect();
    let logical = iteration_shape(&shapes, dims)?;
    let logical_rank = logical.rank();
    // A rank-zero iteration space runs as a single point of a rank-one space.
    let iter: Vec<Monomial> = if logical_rank == 0 {
        vec![Monomial::one()]
    } else {
        logical.dims().to_vec()
    };
    let rank = iter.len();
    let iter_refs: Vec<&Monomial> = iter.iter().collect();

    let indexed = ir.indexed_axes();
    if let Some(&axis) = indexed.iter().find(|&&a| a >= logical_rank) {
        bail!("index axis {axis} out of range for iteration rank {logical_rank}");
    }

    let mut specs = Vec::new();
    let mut participants: Vec<Participant> = Vec::new();
    let live_inputs = ir.live_inputs(operands.len());
    let mut input_participant = Vec::new();
    for (k, op) in operands.iter().enumerate() {
        if !live_inputs[k] {
            input_participant.push(None);
            continue;
        }
        let r = op.layout.rank();
        let p: Participant = (0..rank)
            .map(|a| {
                let i = (a + r).checked_sub(rank)?;
                (!op.layout.shape[i].is_one()).then(|| op.layout.strides[i].clone())
            })
            .collect();
        input_participant.push(Some(participants.len()));
        participants.push(p);
    }

    let has_mx = ir.outputs.iter().any(|o| matches!(o, PwOutput::Mx { .. }));
    let has_elementwise = ir
        .outputs
        .iter()
        .any(|o| matches!(o, PwOutput::Elementwise { .. }));

    // Index of the participant(s) of each output.
    let mut output_participants = Vec::new();
    for (i, output) in ir.outputs.iter().enumerate() {
        match output {
            PwOutput::Elementwise { ty, .. } => {
                specs.push(OutputSpec::new(*ty, logical.clone(), Init::Uninit));
                output_participants.push(vec![participants.len()]);
                participants.push(row_major(&iter_refs)?.into_iter().map(Some).collect());
            }
            PwOutput::Reduce { op, axes, .. } => {
                ensure!(!axes.is_empty(), "output {i}: reduction over no axes");
                for (j, &axis) in axes.iter().enumerate() {
                    ensure!(
                        axis < logical_rank,
                        "output {i}: reduction axis {axis} out of range for iteration rank {logical_rank}"
                    );
                    ensure!(
                        !axes[..j].contains(&axis),
                        "output {i}: duplicate reduction axis {axis}"
                    );
                }
                let kept: Vec<usize> = (0..rank)
                    .filter(|a| logical_rank > 0 && !axes.contains(a))
                    .collect();
                let kept_sizes: Vec<&Monomial> = kept.iter().map(|&a| &iter[a]).collect();
                let shape = if kept.is_empty() {
                    SymShape::new([Monomial::one()])
                } else {
                    kept_sizes.iter().map(|&s| s.clone()).collect()
                };
                let init = match op {
                    Reduce::Sum => Init::Zero,
                    Reduce::Max | Reduce::Min => Init::Fill32(op.identity().to_bits()),
                };
                specs.push(OutputSpec::new(
                    ElementType::Float(skytens_core::data_type::FloatType::F32),
                    shape,
                    init,
                ));
                let kept_strides = row_major(&kept_sizes)?;
                let mut p: Participant = vec![None; rank];
                for (&a, s) in kept.iter().zip(kept_strides) {
                    p[a] = Some(s);
                }
                output_participants.push(vec![participants.len()]);
                participants.push(p);
            }
            PwOutput::Mx { format, dim, .. } => {
                ensure!(
                    logical_rank >= 1,
                    "output {i}: block-scaled outputs need an iteration space of rank at least 1"
                );
                let leading = &iter_refs[..rank - 1];
                let rows = product(leading.iter().copied())?;
                let cols = iter[rank - 1].clone();
                let shape = SymShape::new([rows.clone(), cols.clone()]);
                let block_size = format.block_size();
                let block_dim = &shape[dim.block_dimension_2d()];
                ensure!(
                    block_dim.is_divisible_by_pow2(block_size.ilog2(), dims),
                    "output {i}: block dimension {} of the {format:?} {dim:?} output with shape {} \
                     is not provably a multiple of the block size {block_size}",
                    block_dim.display(dims),
                    shape.display(dims),
                );
                specs.push(OutputSpec::new(
                    TensorTy::BlockScaled {
                        format: *format,
                        dim: *dim,
                    },
                    shape,
                    Init::Uninit,
                ));

                let row_strides = row_major(leading)?;
                let mut values: Participant = Vec::with_capacity(rank);
                let mut rows_p: Participant = Vec::with_capacity(rank);
                for s in &row_strides {
                    values.push(Some(match dim {
                        MxDim::Rowwise => s.checked_mul(&cols).ok_or_else(overflow)?,
                        MxDim::Colwise => s.clone(),
                    }));
                    rows_p.push(Some(s.clone()));
                }
                values.push(Some(match dim {
                    MxDim::Rowwise => Monomial::one(),
                    MxDim::Colwise => rows.clone(),
                }));
                rows_p.push(None);
                output_participants.push(vec![participants.len(), participants.len() + 1]);
                participants.push(values);
                participants.push(rows_p);
            }
        }
    }

    // Axes of size one are dropped, except the last axis of a block-scaled output.
    let pinned = has_mx.then_some(rank - 1);
    let mut order: Vec<usize> = (0..rank)
        .filter(|&a| !iter[a].is_one() || Some(a) == pinned)
        .collect();
    if order.is_empty() {
        order.push(rank - 1);
    }
    if !has_elementwise && !has_mx {
        // Only reductions: iterate in the memory order of the largest input.
        let largest = operands
            .iter()
            .enumerate()
            .filter(|(k, _)| live_inputs[*k])
            .max_by_key(|(_, op)| op.layout.shape.numel().eval_hint(dims));
        if let Some((k, _)) = largest {
            let p = &participants[input_participant[k].unwrap()];
            order.sort_by_key(|&a| {
                std::cmp::Reverse(p[a].as_ref().map_or(usize::MAX, |s| s.eval_hint(dims)))
            });
        }
    }

    let mergeable = |a: usize, b: usize| -> bool {
        if indexed.contains(&a) || indexed.contains(&b) || pinned == Some(a) || pinned == Some(b) {
            return false;
        }
        participants.iter().all(|p| match (&p[a], &p[b]) {
            (None, None) => true,
            (Some(sa), Some(sb)) => sb.checked_mul(&iter[b]).as_ref() == Some(sa),
            _ => false,
        })
    };
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for &a in &order {
        match groups.last_mut() {
            Some(group) if mergeable(*group.last().unwrap(), a) => group.push(a),
            _ => groups.push(vec![a]),
        }
    }
    let lowered_rank = groups.len();
    let sizes = groups
        .iter()
        .map(|g| product(g.iter().map(|&a| &iter[a])))
        .collect::<anyhow::Result<Vec<_>>>()?;
    // Strides of a participant along the lowered axes: those of each group's innermost axis.
    let lowered = |p: &Participant| -> Vec<Option<Monomial>> {
        groups
            .iter()
            .map(|g| p[*g.last().unwrap()].clone())
            .collect()
    };
    let unit = |strides: &[Monomial]| strides.iter().map(Monomial::is_one).collect::<Vec<_>>();

    let mut input_facts = Vec::new();
    let mut input_plans = Vec::new();
    for (k, op) in operands.iter().enumerate() {
        let (axes, strides): (Vec<usize>, Vec<Monomial>) = match input_participant[k] {
            Some(p) => lowered(&participants[p])
                .into_iter()
                .enumerate()
                .filter_map(|(a, s)| Some((a, s?)))
                .unzip(),
            None => Default::default(),
        };
        input_facts.push(InputFacts {
            ty: op.ty,
            live: live_inputs[k],
            unit_stride: unit(&strides),
            axes,
        });
        input_plans.push(strides);
    }

    let mut output_facts = Vec::new();
    let mut output_plans = Vec::new();
    for (output, ps) in ir.outputs.iter().zip(&output_participants) {
        let lp = lowered(&participants[ps[0]]);
        match output {
            PwOutput::Elementwise { ty, .. } => {
                let strides: Vec<Monomial> = lp.into_iter().map(Option::unwrap).collect();
                output_facts.push(OutputFacts::Elementwise {
                    ty: *ty,
                    unit_stride: unit(&strides),
                });
                output_plans.push(OutputPlan::Elementwise { strides });
            }
            PwOutput::Reduce { op, .. } => {
                let (kept, strides): (Vec<usize>, Vec<Monomial>) = lp
                    .into_iter()
                    .enumerate()
                    .filter_map(|(a, s)| Some((a, s?)))
                    .unzip();
                output_facts.push(OutputFacts::Reduce { op: *op, kept });
                output_plans.push(OutputPlan::Reduce { strides });
            }
            PwOutput::Mx { format, dim, .. } => {
                let values_strides: Vec<Monomial> = lp.into_iter().map(Option::unwrap).collect();
                let row_strides: Vec<Monomial> = lowered(&participants[ps[1]])[..lowered_rank - 1]
                    .iter()
                    .map(|s| s.clone().unwrap())
                    .collect();
                let block_size = format.block_size();
                let block_axis = match dim {
                    MxDim::Rowwise => lowered_rank - 1,
                    MxDim::Colwise => {
                        // Blocks run down the innermost leading axis, which must hold whole blocks.
                        ensure!(
                            lowered_rank >= 2
                                && sizes[lowered_rank - 2]
                                    .is_divisible_by_pow2(block_size.ilog2(), dims),
                            "colwise block-scaled output: the innermost axis of size other than one \
                             before the last axis must be provably a multiple of the block size {block_size}"
                        );
                        lowered_rank - 2
                    }
                };
                let block_dim_size = match dim {
                    MxDim::Rowwise => sizes[lowered_rank - 1].clone(),
                    MxDim::Colwise => product(&sizes[..lowered_rank - 1])?,
                };
                output_facts.push(OutputFacts::Mx {
                    format: *format,
                    dim: *dim,
                    unit_stride: unit(&values_strides),
                    block_axis,
                });
                output_plans.push(OutputPlan::Mx {
                    values_strides,
                    row_strides,
                    block_dim: block_dim_size,
                });
            }
        }
    }

    let index_axes = indexed
        .iter()
        .map(|&a| (a, groups.iter().position(|g| g.contains(&a))))
        .collect();

    let output_shapes: Vec<SymShape> = specs.iter().map(|spec| spec.shape.clone()).collect();
    Ok(Lowering {
        output_specs: specs,
        facts: Facts {
            rank: lowered_rank,
            inputs: input_facts,
            outputs: output_facts,
            index_axes,
        },
        plan: Plan {
            sizes,
            inputs: input_plans,
            outputs: output_plans,
            input_layouts: operands.iter().map(|op| op.layout.clone()).collect(),
            output_shapes,
        },
    })
}

impl Facts {
    /// Minimum tile size along each lowered axis.
    pub fn min_tile(&self) -> Vec<u32> {
        let mut min = vec![1u32; self.rank];
        for output in &self.outputs {
            if let OutputFacts::Mx {
                format, block_axis, ..
            } = output
            {
                min[*block_axis] = min[*block_axis].max(format.block_size() as u32);
            }
        }
        min
    }
}

fn pow2_at_least(n: usize) -> u32 {
    n.max(1).next_power_of_two().min(1 << 30) as u32
}

/// Tile shapes to try, the heuristic default first. `hints` are the lowered axis sizes.
pub fn tile_candidates(hints: &[usize], min_tile: &[u32]) -> Vec<Vec<u32>> {
    let rank = hints.len();
    // The minimum tiles are reserved first; the remaining budget scales them up,
    // innermost axis first.
    let make = |total: u32, inner_div: u32| -> Vec<u32> {
        let mut tile = min_tile.to_vec();
        let reserved: u32 = min_tile.iter().product();
        let mut budget = (total / reserved).max(1);
        let max_inner = if rank == 1 { total } else { 1024 };
        for a in (0..rank).rev() {
            let wanted = if a == rank - 1 {
                (pow2_at_least(hints[a]).min(max_inner) / inner_div).max(1)
            } else {
                pow2_at_least(hints[a])
            };
            let factor = (wanted / min_tile[a]).clamp(1, budget);
            tile[a] = min_tile[a] * factor;
            budget /= factor;
        }
        tile
    };
    let mut candidates: Vec<Vec<u32>> = Vec::new();
    // Larger tiles measured slower for memory-bound kernels on sm_89.
    for (total, inner_div) in [
        (2048, 1),
        (1024, 1),
        (4096, 1),
        (8192, 1),
        (2048, 4),
        (1024, 4),
        (4096, 4),
        (8192, 8),
    ] {
        if rank == 1 && inner_div > 1 {
            continue;
        }
        let tile = make(total, inner_div);
        if !candidates.contains(&tile) {
            candidates.push(tile);
        }
    }
    candidates
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execgraph::{
        ops::pointwise::ir::{BinaryOp, Node, NodeId},
        sym::{Polynomial, SymDim},
    };
    use skytens_core::data_type::FloatType;

    const F32: ElementType = ElementType::Float(FloatType::F32);

    fn contiguous(shape: &[Monomial], dims: &SymDims) -> SymLayout {
        SymLayout::contiguous_row_major(shape.iter().cloned().collect(), F32, dims).unwrap()
    }

    fn add_ir(inputs: usize, outputs: Vec<PwOutput>) -> PwIr {
        let mut nodes: Vec<Node> = (0..inputs).map(Node::Input).collect();
        for i in 1..inputs {
            let prev = if i == 1 { 0 } else { nodes.len() - 1 };
            nodes.push(Node::Binary(
                BinaryOp::Add,
                NodeId(prev as u32),
                NodeId(i as u32),
            ));
        }
        let mut ir = PwIr {
            nodes,
            outputs: vec![],
        };
        let last = NodeId(ir.nodes.len() as u32 - 1);
        ir.outputs = outputs
            .into_iter()
            .map(|mut o| {
                match &mut o {
                    PwOutput::Elementwise { value, .. }
                    | PwOutput::Reduce { value, .. }
                    | PwOutput::Mx { value, .. } => *value = last,
                }
                o
            })
            .collect();
        ir
    }

    fn elementwise() -> PwOutput {
        PwOutput::Elementwise {
            value: NodeId(0),
            ty: F32,
        }
    }

    fn dims_mn() -> (SymDims, Monomial, Monomial) {
        let mut dims = SymDims::new();
        let m = dims.insert(SymDim {
            name: "m".into(),
            hint: 100,
            alignment_log2: 0,
        });
        let n = dims.insert(SymDim {
            name: "n".into(),
            hint: 256,
            alignment_log2: 5,
        });
        (dims, Monomial::atom(m), Monomial::atom(n))
    }

    #[test]
    fn broadcasting_and_collapse() {
        let (dims, m, n) = dims_mn();
        let a = contiguous(&[m.clone(), n.clone()], &dims);
        let bias = contiguous(std::slice::from_ref(&n), &dims);
        let col = contiguous(&[m.clone(), Monomial::one()], &dims);

        // Same-shape operands collapse to one axis.
        let ir = add_ir(2, vec![elementwise()]);
        let ops = [OperandDesc {
            layout: &a,
            ty: F32,
        }; 2];
        let l = lower(&ir, &ops, &dims).unwrap();
        assert_eq!(l.facts.rank, 1);
        assert_eq!(l.plan.sizes, [m.clone() * n.clone()]);
        assert_eq!(
            l.output_specs[0].shape,
            SymShape::new([m.clone(), n.clone()])
        );

        // A bias vector prevents collapsing, and is declared at reduced rank.
        let ops = [
            OperandDesc {
                layout: &a,
                ty: F32,
            },
            OperandDesc {
                layout: &bias,
                ty: F32,
            },
            OperandDesc {
                layout: &col,
                ty: F32,
            },
        ];
        let ir = add_ir(3, vec![elementwise()]);
        let l = lower(&ir, &ops, &dims).unwrap();
        assert_eq!(l.facts.rank, 2);
        assert_eq!(l.facts.inputs[1].axes, [1]);
        assert_eq!(l.facts.inputs[2].axes, [0]);
        assert_eq!(l.plan.inputs[2], [Monomial::one()]);
        assert_eq!(l.facts.inputs[0].unit_stride, [false, true]);
    }

    #[test]
    fn broadcast_mismatch() {
        let (dims, m, n) = dims_mn();
        let a = contiguous(&[m.clone(), n.clone()], &dims);
        let b = contiguous(std::slice::from_ref(&m), &dims);
        let ir = add_ir(2, vec![elementwise()]);
        let ops = [
            OperandDesc {
                layout: &a,
                ty: F32,
            },
            OperandDesc {
                layout: &b,
                ty: F32,
            },
        ];
        assert!(lower(&ir, &ops, &dims).is_err());
    }

    #[test]
    fn transposed_input_blocks_collapse() {
        let (dims, m, n) = dims_mn();
        let a = contiguous(&[m.clone(), n.clone()], &dims);
        let t = contiguous(&[n.clone(), m.clone()], &dims).transposed(0, 1);
        let ir = add_ir(2, vec![elementwise()]);
        let ops = [
            OperandDesc {
                layout: &a,
                ty: F32,
            },
            OperandDesc {
                layout: &t,
                ty: F32,
            },
        ];
        let l = lower(&ir, &ops, &dims).unwrap();
        assert_eq!(l.facts.rank, 2);
        assert_eq!(l.plan.inputs[1], [Monomial::one(), m.clone()]);
    }

    #[test]
    fn reductions() {
        let (dims, m, n) = dims_mn();
        let k = Monomial::fixed(8);
        let a = contiguous(&[k.clone(), m.clone(), n.clone()], &dims);
        let ops = [OperandDesc {
            layout: &a,
            ty: F32,
        }];
        let reduce = |axes: Vec<usize>| PwOutput::Reduce {
            value: NodeId(0),
            op: Reduce::Sum,
            axes,
        };

        // Leading axes kept and collapsed, last reduced.
        let l = lower(&add_ir(1, vec![reduce(vec![2])]), &ops, &dims).unwrap();
        assert_eq!(l.facts.rank, 2);
        assert_eq!(
            l.output_specs[0].shape,
            SymShape::new([k.clone(), m.clone()])
        );
        assert_eq!(l.output_specs[0].init, Init::Zero);
        let OutputFacts::Reduce { kept, .. } = &l.facts.outputs[0] else {
            panic!()
        };
        assert_eq!(kept, &[0]);

        // Everything reduced collapses to a single axis with a [1] output.
        let l = lower(&add_ir(1, vec![reduce(vec![0, 1, 2])]), &ops, &dims).unwrap();
        assert_eq!(l.facts.rank, 1);
        assert_eq!(l.output_specs[0].shape, SymShape::new([Monomial::one()]));

        // Reduced and kept axes don't merge.
        let l = lower(&add_ir(1, vec![reduce(vec![0])]), &ops, &dims).unwrap();
        assert_eq!(l.facts.rank, 2);

        assert!(lower(&add_ir(1, vec![reduce(vec![3])]), &ops, &dims).is_err());
        assert!(lower(&add_ir(1, vec![reduce(vec![1, 1])]), &ops, &dims).is_err());
    }

    #[test]
    fn reduction_only_follows_input_memory_order() {
        let (dims, m, n) = dims_mn();
        // A transposed [m, n] input summed over m: iterate with m innermost.
        let t = contiguous(&[n.clone(), m.clone()], &dims).transposed(0, 1);
        let ops = [OperandDesc {
            layout: &t,
            ty: F32,
        }];
        let ir = add_ir(
            1,
            vec![PwOutput::Reduce {
                value: NodeId(0),
                op: Reduce::Max,
                axes: vec![0],
            }],
        );
        let l = lower(&ir, &ops, &dims).unwrap();
        assert_eq!(l.plan.sizes, [n.clone(), m.clone()]);
        assert_eq!(l.plan.inputs[0], [m.clone(), Monomial::one()]);
        let OutputFacts::Reduce { kept, .. } = &l.facts.outputs[0] else {
            panic!()
        };
        assert_eq!(kept, &[0]);
        assert_eq!(
            l.output_specs[0].init,
            Init::Fill32(f32::NEG_INFINITY.to_bits())
        );
    }

    #[test]
    fn mx_outputs() {
        let (dims, m, n) = dims_mn();
        let b = Monomial::fixed(4);
        let a = contiguous(&[b.clone(), m.clone(), n.clone()], &dims);
        let ops = [OperandDesc {
            layout: &a,
            ty: F32,
        }];
        let mx = |dim| PwOutput::Mx {
            value: NodeId(0),
            format: MxFormat::Mxfp8,
            dim,
        };
        let l = lower(&add_ir(1, vec![mx(MxDim::Rowwise)]), &ops, &dims).unwrap();
        // Leading axes collapse, the last never merges.
        assert_eq!(l.facts.rank, 2);
        assert_eq!(
            l.output_specs[0].shape,
            SymShape::new([b.clone() * m.clone(), n.clone()])
        );
        let OutputPlan::Mx {
            block_dim,
            row_strides,
            ..
        } = &l.plan.outputs[0]
        else {
            panic!()
        };
        assert_eq!(row_strides, &[Monomial::one()]);
        assert_eq!(block_dim, &n);

        // m is not provably a multiple of 32.
        assert!(lower(&add_ir(1, vec![mx(MxDim::Colwise)]), &ops, &dims).is_err());

        let a = contiguous(&[n.clone(), m.clone()], &dims);
        let ops = [OperandDesc {
            layout: &a,
            ty: F32,
        }];
        let l = lower(&add_ir(1, vec![mx(MxDim::Colwise)]), &ops, &dims).unwrap();
        let OutputFacts::Mx {
            block_axis,
            unit_stride,
            ..
        } = &l.facts.outputs[0]
        else {
            panic!()
        };
        assert_eq!(*block_axis, 0);
        assert_eq!(unit_stride, &[true, false]);
        // n is a multiple of 32 but m isn't.
        assert!(lower(&add_ir(1, vec![mx(MxDim::Rowwise)]), &ops, &dims).is_err());
    }

    #[test]
    fn index_axes_and_scalars() {
        let (dims, m, n) = dims_mn();
        let a = contiguous(&[m.clone(), n.clone()], &dims);
        let ops = [OperandDesc {
            layout: &a,
            ty: F32,
        }];
        let ir = PwIr {
            nodes: vec![
                Node::Input(0),
                Node::Index(1),
                Node::Cast(ValueTy::F32, NodeId(1)),
                Node::Binary(BinaryOp::Add, NodeId(0), NodeId(2)),
            ],
            outputs: vec![PwOutput::Elementwise {
                value: NodeId(3),
                ty: F32,
            }],
        };
        let l = lower(&ir, &ops, &dims).unwrap();
        assert_eq!(l.facts.rank, 2);
        assert_eq!(l.facts.index_axes, [(1, Some(1))]);

        let mut bad = ir.clone();
        bad.nodes[1] = Node::Index(2);
        assert!(lower(&bad, &ops, &dims).is_err());

        let scalar = SymLayout::new(SymShape::new([]), vec![], Polynomial::zero());
        let ops = [OperandDesc {
            layout: &scalar,
            ty: F32,
        }];
        let l = lower(&add_ir(1, vec![elementwise()]), &ops, &dims).unwrap();
        assert_eq!(l.facts.rank, 1);
        assert!(l.facts.inputs[0].axes.is_empty());
        assert_eq!(l.output_specs[0].shape.rank(), 0);
    }

    #[test]
    fn candidates() {
        let c = tile_candidates(&[1000, 4096], &[1, 1]);
        assert_eq!(c[0], [2, 1024]);
        assert!(c.len() <= 8);
        let c = tile_candidates(&[3], &[1]);
        assert_eq!(c, [vec![4]]);
        let c = tile_candidates(&[64, 8], &[32, 1]);
        assert!(c.iter().all(|t| t[0] >= 32));
        // A colwise MX output over a wide input keeps the default tile small.
        let c = tile_candidates(&[4096, 8192], &[32, 1]);
        assert_eq!(c[0], [32, 64]);
        assert!(
            c.iter()
                .all(|t| t[0] >= 32 && t.iter().product::<u32>() <= 8192)
        );
    }
}
