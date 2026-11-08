//! Programmable fused GEMM epilogue.

use serde::{Deserialize, Serialize};

/// List of operations executed in the epilogue. This can be seen
/// as a sort of simple bytecode.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Epilogue {
    ops: Vec<EpilogueOp>,
}

impl Epilogue {
    pub fn new(ops: impl IntoIterator<Item = EpilogueOp>) -> Self {
        Self {
            ops: ops.into_iter().collect(),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]

pub enum EpilogueOp {
    Load {
        dst_slot: SlotId,
        id: LoadTensorId,
        ty: LoadTensorTy,
    },
    Unary {
        dst_slot: SlotId,
        src_slot: SlotId,
        op: UnaryOp,
    },
    Binary {
        dst_slot: SlotId,
        lhs_slot: SlotId,
        rhs_slot: SlotId,
        op: BinaryOp,
    },
    Store {
        src_slot: SlotId,
        ty: StoreTensorTy,
    },
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SlotId(pub u32);

impl SlotId {
    pub const MMA_OUTPUT: SlotId = SlotId(u32::MAX);
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScalarId(pub u32);

#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoadTensorId(pub u32);

#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoreTensorId(pub u32);

#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LoadTensorTy {
    Bf16RowMajor,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StoreTensorTy {
    Bf16RowMajor,
    Mxfp8Rowwise,
    Mxfp8Colwise,
    Nvfp8Rowwise,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnaryOp {
    Neg,
    Exp2Approx,
    RcpApprox,
    AddWithScalar(ScalarId),
    MulWithScalar(ScalarId),
    MinWithScalar(ScalarId),
    MaxWithScalar(ScalarId),
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    DivApprox,
    Min,
    Max,
}
