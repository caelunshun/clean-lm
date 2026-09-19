//! # GPU-side UB
//! GPU-side UB is not considered Rust UB,
//! and is instead seen as a logic bug from the Rust
//! language perspective. Thus many operations which might
//! require `unsafe` if executing on the CPU are not `unsafe`.

pub mod cudnn;
pub mod tensor;
// pub mod module;
pub mod autotuning;
pub mod context;
pub mod disk_cache;
pub mod kernel;
pub mod operators;
