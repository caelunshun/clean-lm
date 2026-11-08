//! Serializable dataflow IR of a pointwise kernel.

use crate::mx_tensor::{MxDim, MxFormat};
use anyhow::{bail, ensure};
use serde::{Deserialize, Serialize};
use skytens_core::data_type::{ElementType, FloatType};

/// Type of a value computed by a pointwise kernel.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ValueTy {
    F32,
    I32,
    Bool,
}

impl ValueTy {
    /// The value type a stored element is loaded as.
    pub fn of_element(ty: ElementType) -> anyhow::Result<Self> {
        match ty {
            ElementType::Float(FloatType::UF8E8M0 | FloatType::F4E2M1) => {
                bail!("pointwise inputs of type {ty:?} are not supported")
            }
            ElementType::Float(_) => Ok(Self::F32),
            ElementType::SInt(_) | ElementType::UInt(_) => Ok(Self::I32),
            ElementType::Bool => bail!("bit-packed bool inputs are not supported"),
        }
    }

    pub fn is_numeric(self) -> bool {
        matches!(self, Self::F32 | Self::I32)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct NodeId(pub u32);

impl NodeId {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UnaryOp {
    /// `f32` or `i32`.
    Neg,
    /// `f32` or `i32`.
    Abs,
    Exp,
    Log,
    Sqrt,
    Rsqrt,
    Tanh,
    Sigmoid,
    Floor,
    /// `bool` only.
    Not,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    /// `i32` only.
    Rem,
    Min,
    Max,
    /// `f32` only.
    Pow,
    /// `bool` only.
    And,
    /// `bool` only.
    Or,
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

impl BinaryOp {
    pub fn is_comparison(self) -> bool {
        matches!(
            self,
            Self::Lt | Self::Le | Self::Gt | Self::Ge | Self::Eq | Self::Ne
        )
    }
}

/// A node of the dataflow graph. Operands always refer to earlier nodes, and must
/// have the types required by the op (the DSL inserts casts).
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Node {
    /// Element of the operand with this index, broadcast over the iteration space.
    Input(usize),
    /// `i32` coordinate along an iteration axis.
    Index(usize),
    /// Bit pattern of an `f32`.
    ConstF32(u32),
    ConstI32(i32),
    ConstBool(bool),
    Unary(UnaryOp, NodeId),
    Binary(BinaryOp, NodeId, NodeId),
    /// `a * b + c` on `f32`.
    Fma(NodeId, NodeId, NodeId),
    Select(NodeId, NodeId, NodeId),
    Cast(ValueTy, NodeId),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Reduce {
    Sum,
    Max,
    Min,
}

impl Reduce {
    /// Initial value of the `f32` output buffer.
    pub fn identity(self) -> f32 {
        match self {
            Reduce::Sum => 0.0,
            Reduce::Max => f32::NEG_INFINITY,
            Reduce::Min => f32::INFINITY,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PwOutput {
    /// Stored at every point of the iteration space.
    Elementwise { value: NodeId, ty: ElementType },
    /// `f32` reduction over iteration axes.
    Reduce {
        value: NodeId,
        op: Reduce,
        axes: Vec<usize>,
    },
    /// Block-scaled quantization of the iteration space viewed as 2D
    /// `[product of all but the last axis, last axis]`.
    Mx {
        value: NodeId,
        format: MxFormat,
        dim: MxDim,
    },
}

impl PwOutput {
    pub fn value(&self) -> NodeId {
        match self {
            PwOutput::Elementwise { value, .. }
            | PwOutput::Reduce { value, .. }
            | PwOutput::Mx { value, .. } => *value,
        }
    }
}

/// Element types elementwise outputs may be stored as.
pub fn check_output_type(ty: ElementType) -> anyhow::Result<()> {
    match ty {
        ElementType::Float(FloatType::UF8E8M0 | FloatType::F4E2M1) | ElementType::Bool => {
            bail!("pointwise outputs of type {ty:?} are not supported")
        }
        _ => Ok(()),
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PwIr {
    pub nodes: Vec<Node>,
    pub outputs: Vec<PwOutput>,
}

impl PwIr {
    /// Checks the graph structure and operand types, returning the type of every node.
    pub fn check(&self, input_tys: &[ValueTy]) -> anyhow::Result<Vec<ValueTy>> {
        let mut tys: Vec<ValueTy> = Vec::with_capacity(self.nodes.len());
        for (i, node) in self.nodes.iter().enumerate() {
            let ty_of = |id: NodeId| -> anyhow::Result<ValueTy> {
                ensure!(
                    id.index() < i,
                    "node {i} refers to node {} which is not defined before it",
                    id.0
                );
                Ok(tys[id.index()])
            };
            let ty = match *node {
                Node::Input(k) => {
                    ensure!(k < input_tys.len(), "node {i} refers to unknown input {k}");
                    input_tys[k]
                }
                Node::Index(_) => ValueTy::I32,
                Node::ConstF32(_) => ValueTy::F32,
                Node::ConstI32(_) => ValueTy::I32,
                Node::ConstBool(_) => ValueTy::Bool,
                Node::Unary(op, a) => {
                    let a = ty_of(a)?;
                    let ok = match op {
                        UnaryOp::Neg | UnaryOp::Abs => a.is_numeric(),
                        UnaryOp::Not => a == ValueTy::Bool,
                        _ => a == ValueTy::F32,
                    };
                    ensure!(ok, "node {i}: {op:?} is not defined for {a:?}");
                    a
                }
                Node::Binary(op, a, b) => {
                    let (a, b) = (ty_of(a)?, ty_of(b)?);
                    ensure!(
                        a == b,
                        "node {i}: {op:?} operands have different types {a:?} and {b:?}"
                    );
                    let ok = match op {
                        BinaryOp::Add
                        | BinaryOp::Sub
                        | BinaryOp::Mul
                        | BinaryOp::Div
                        | BinaryOp::Min
                        | BinaryOp::Max
                        | BinaryOp::Lt
                        | BinaryOp::Le
                        | BinaryOp::Gt
                        | BinaryOp::Ge => a.is_numeric(),
                        BinaryOp::Rem => a == ValueTy::I32,
                        BinaryOp::Pow => a == ValueTy::F32,
                        BinaryOp::And | BinaryOp::Or => a == ValueTy::Bool,
                        BinaryOp::Eq | BinaryOp::Ne => true,
                    };
                    ensure!(ok, "node {i}: {op:?} is not defined for {a:?}");
                    if op.is_comparison() { ValueTy::Bool } else { a }
                }
                Node::Fma(a, b, c) => {
                    for x in [a, b, c] {
                        ensure!(ty_of(x)? == ValueTy::F32, "node {i}: fma requires f32");
                    }
                    ValueTy::F32
                }
                Node::Select(c, a, b) => {
                    ensure!(
                        ty_of(c)? == ValueTy::Bool,
                        "node {i}: select condition must be bool"
                    );
                    let (a, b) = (ty_of(a)?, ty_of(b)?);
                    ensure!(
                        a == b,
                        "node {i}: select operands have different types {a:?} and {b:?}"
                    );
                    a
                }
                Node::Cast(ty, a) => {
                    ty_of(a)?;
                    ty
                }
            };
            tys.push(ty);
        }

        ensure!(!self.outputs.is_empty(), "pointwise op has no outputs");
        for (i, output) in self.outputs.iter().enumerate() {
            let value = output.value();
            ensure!(
                value.index() < self.nodes.len(),
                "output {i} refers to unknown node {}",
                value.0
            );
            let ty = tys[value.index()];
            match output {
                PwOutput::Elementwise { ty: elem, .. } => check_output_type(*elem)?,
                PwOutput::Reduce { .. } | PwOutput::Mx { .. } => ensure!(
                    ty == ValueTy::F32,
                    "output {i}: reductions and block-scaled outputs require f32 values, got {ty:?}"
                ),
            }
        }
        Ok(tys)
    }

    /// Whether each node contributes to an output.
    pub fn live_nodes(&self) -> Vec<bool> {
        let mut live = vec![false; self.nodes.len()];
        for output in &self.outputs {
            live[output.value().index()] = true;
        }
        for i in (0..self.nodes.len()).rev() {
            if !live[i] {
                continue;
            }
            let mut mark = |id: NodeId| live[id.index()] = true;
            match self.nodes[i] {
                Node::Unary(_, a) | Node::Cast(_, a) => mark(a),
                Node::Binary(_, a, b) => {
                    mark(a);
                    mark(b);
                }
                Node::Fma(a, b, c) | Node::Select(a, b, c) => {
                    mark(a);
                    mark(b);
                    mark(c);
                }
                Node::Input(_)
                | Node::Index(_)
                | Node::ConstF32(_)
                | Node::ConstI32(_)
                | Node::ConstBool(_) => {}
            }
        }
        live
    }

    /// Iteration axes read through [`Node::Index`] by live nodes.
    pub fn indexed_axes(&self) -> Vec<usize> {
        let live = self.live_nodes();
        let mut axes: Vec<usize> = self
            .nodes
            .iter()
            .zip(&live)
            .filter_map(|(node, &live)| match node {
                Node::Index(axis) if live => Some(*axis),
                _ => None,
            })
            .collect();
        axes.sort_unstable();
        axes.dedup();
        axes
    }

    /// Whether each input is read by a live node.
    pub fn live_inputs(&self, num_inputs: usize) -> Vec<bool> {
        let live = self.live_nodes();
        let mut inputs = vec![false; num_inputs];
        for (node, &live) in self.nodes.iter().zip(&live) {
            if let (Node::Input(k), true) = (node, live) {
                inputs[*k] = true;
            }
        }
        inputs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checks_types() {
        let ir = PwIr {
            nodes: vec![
                Node::Input(0),
                Node::ConstF32(2.0f32.to_bits()),
                Node::Binary(BinaryOp::Mul, NodeId(0), NodeId(1)),
                Node::Binary(BinaryOp::Gt, NodeId(2), NodeId(1)),
                Node::Select(NodeId(3), NodeId(2), NodeId(1)),
            ],
            outputs: vec![PwOutput::Elementwise {
                value: NodeId(4),
                ty: ElementType::Float(FloatType::Bf16),
            }],
        };
        let tys = ir.check(&[ValueTy::F32]).unwrap();
        assert_eq!(tys[3], ValueTy::Bool);
        assert_eq!(tys[4], ValueTy::F32);
        assert!(ir.check(&[ValueTy::I32]).is_err());
    }

    #[test]
    fn rejects_forward_references_and_bad_types() {
        let forward = PwIr {
            nodes: vec![Node::Unary(UnaryOp::Exp, NodeId(0))],
            outputs: vec![],
        };
        assert!(forward.check(&[]).is_err());

        let bad_reduce = PwIr {
            nodes: vec![Node::Index(0)],
            outputs: vec![PwOutput::Reduce {
                value: NodeId(0),
                op: Reduce::Sum,
                axes: vec![0],
            }],
        };
        assert!(bad_reduce.check(&[]).is_err());
    }

    #[test]
    fn liveness() {
        let ir = PwIr {
            nodes: vec![
                Node::Input(0),
                Node::Input(1),
                Node::Index(1),
                Node::Unary(UnaryOp::Exp, NodeId(1)),
            ],
            outputs: vec![PwOutput::Elementwise {
                value: NodeId(3),
                ty: ElementType::Float(FloatType::F32),
            }],
        };
        assert_eq!(ir.live_nodes(), [false, true, false, true]);
        assert_eq!(ir.live_inputs(2), [false, true]);
        assert!(ir.indexed_axes().is_empty());
    }
}
