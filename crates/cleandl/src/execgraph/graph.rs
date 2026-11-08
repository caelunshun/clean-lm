use super::sym::{SymDim, SymDimId, SymShape};
use crate::mx_tensor::MxFormat;
use skytens_core::data_type::ElementType;
use slotmap::{SlotMap, new_key_type};
use std::range::Range;

new_key_type! {
    pub struct OpId;
}

new_key_type! {
    pub struct SymTensorId;
}

pub struct ExecGraphBuilder {
    sym_tensors: SlotMap<SymTensorId, SymTensor>,
    sym_dims: SlotMap<SymDimId, SymDim>,
    ops: SlotMap<OpId, Box<dyn ErasedOp>>,
}

pub trait Op: Send + Sync + 'static {
    type Executor;

    fn operands(&self) -> &[Operand];
}

trait ErasedOp: Send + Sync + 'static {
    // TODO
}

#[derive(Debug, Clone)]
pub struct Operand {
    pub tensor_slice: SymTensorSlice,
    pub access: OperandAccess,
}

#[derive(Debug, Clone)]
pub struct SymTensorSlice {
    pub tensor: SymTensorId,
    /// Set of dimensions that are sliced (i.e. not always fully included
    /// in the tensor view). Slice sizes currently must be fixed.
    pub slice_dims: Vec<(usize, Range<usize>)>, // (dim, slice range)
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum OperandAccess {
    ReadOnly,
    ReadWrite,
}

#[derive(Debug, Clone)]
pub struct SymTensor {
    pub shape: SymShape,
    pub ty: TensorTy,
}

#[derive(Debug, Clone)]
pub enum TensorTy {
    Plain(ElementType),
    BlockScaled(MxFormat),
}
