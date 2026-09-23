//! WIP
//! 
//! tcgen05 (SM10x / SM11x) family matmul kernel with programmable pointwise+auxiliary input/output epilogue.
//! It is intended for heavy epilogue-fusion cases (e.g. fused SwiGLU gate) where cuDNN currently unperforms
//! and which cublas/cublasLt/nvjet do not support.
//!
//! Currently, it is built for a hardcoded tile size of 128x256 per CTA /
//! 256x256 per CTA-pair. It does not support a cluster size beyond the CTA-pair (2x1x1)
//! and thus does not use TMA multicast.
//! It supports bf16 inputs and outputs only.
//!
//! Each input can be either row-major or column-major. Outputs must be row-major.
//!
//! Epilogues can consume auxiliary bf16 row-major inputs and produce multiple
//! row-major bf16 outputs. Epilogue is overlapped with MMA via accumulator double-buffering.
//!
//! Epilogue uses subtiling, currently with a fixed subtile size of 128x64,
//! distributed across 4 epilogue warps each processing 32x64 elements.
//! The epilogue uses one or more 128x64 bf16 SMEM slots for staging auxiliary inputs
//! and one 128x64 bf16 SMEM slot for staging outputs.
//!
//! For the MMA mainloop, we use two or more SMEM slots each storing 128x64 bf16 elements for A
//! and 64x128 bf16 elements for B (noting that B's `N`-extent of 256 is split across the two CTAs).
//!
//! We use the following warp specialization:
//! * Warp 0: A/B TMA loads
//! * Warp 1: MMA issue
//! * Warp 2: Epilogue TMA loads
//! * Warp 3: Unused / register donor for epilogue warps
//! * Warps 4..8: Epilogue warps; TMA stores issued by warp 4

use cuda_device::{Barrier, warp::warp_id};
use cuda_device_ext::bf16;

const TM_CTA: u32 = 128;
const TN_CTA: u32 = 256;
const TK: u32 = 64;

const EPI_TM: u32 = 128;
const EPI_TN: u32 = 64;

#[derive(Copy, Clone, PartialEq, Eq)]
enum WarpType {
    AbTmaLoad,
    MmaIssue,
    EpilogueTmaLoad,
    Unused,
    Epilogue,
}

#[inline(always)]
fn warp_type() -> WarpType {
    match warp_id() {
        0 => WarpType::AbTmaLoad,
        1 => WarpType::MmaIssue,
        2 => WarpType::EpilogueTmaLoad,
        3 => WarpType::Unused,
        _ => WarpType::Epilogue,
    }
}

struct Barriers<const AB_PIPE_DEPTH: usize, const EPILOGUE_PIPE_DEPTH: usize> {
    ab_slot_produced: [Barrier; AB_PIPE_DEPTH],
    ab_slot_consumed: [Barrier; AB_PIPE_DEPTH],
    epi_load_slot_produced: [Barrier; EPILOGUE_PIPE_DEPTH],
    epi_load_slot_consumed: [Barrier; EPILOGUE_PIPE_DEPTH],
    accumulator_slot_produced: [Barrier; 2],
    accumulator_slot_consumed: [Barrier; 2],
}

#[derive(Copy, Clone)]
struct Smem<const AB_PIPE_DEPTH: usize, const EPILOGUE_PIPE_DEPTH: usize> {
    base: *mut u8,
}

impl<const AB_PIPE_DEPTH: usize, const EPILOGUE_PIPE_DEPTH: usize>
    Smem<AB_PIPE_DEPTH, EPILOGUE_PIPE_DEPTH>
{
    #[inline(always)]
    fn a_staging_slot_offset(&self, slot_idx: u32) -> u32 {
        slot_idx * size_of::<bf16>() as u32 * (TM_CTA * TK + TN_CTA * TK)
    }

    #[inline(always)]
    pub fn a_staging_slot(&self, slot_idx: u32) -> *mut u8 {
        let offset = self.a_staging_slot_offset(slot_idx);
        unsafe { self.base.add(offset as usize) }
    }

    #[inline(always)]
    pub fn b_staging_slot(&self, slot_idx: u32) -> *mut u8 {
        let offset = self.a_staging_slot_offset(slot_idx) + size_of::<bf16>() as u32 * TM_CTA * TK;
        unsafe { self.base.add(offset as usize) }
    }

    #[inline(always)]
    fn epi_load_staging_slot_offset(&self, slot_idx: u32) -> u32 {
        self.a_staging_slot_offset(AB_PIPE_DEPTH as u32)
            + size_of::<bf16>() as u32 * EPI_TM * EPI_TN * slot_idx
    }

    #[inline(always)]
    pub fn epi_load_staging_slot(&self, slot_idx: u32) -> *mut u8 {
        let offset = self.epi_load_staging_slot_offset(slot_idx);
        unsafe { self.base.add(offset as usize) }
    }

    #[inline(always)]
    pub fn epi_store_staging_slot(&self) -> *mut u8 {
        let offset = self.epi_load_staging_slot_offset(EPILOGUE_PIPE_DEPTH as u32);
        unsafe { self.base.add(offset as usize) }
    }

    #[inline(always)]
    pub fn barriers_raw(&mut self) -> *mut Barriers<AB_PIPE_DEPTH, EPILOGUE_PIPE_DEPTH> {
        let offset = self.epi_load_staging_slot_offset(EPILOGUE_PIPE_DEPTH as u32 + 1);
        unsafe { self.base.add(offset as usize).cast() }
    }

    #[inline(always)]
    pub fn barriers(&mut self) -> &mut Barriers<AB_PIPE_DEPTH, EPILOGUE_PIPE_DEPTH> {
        &mut *self.barriers()
    }
}

/// Should be called by one thread.
fn init_barriers(smem: &mut Smem) {
    
}
