//! tcgen05 (SM10x / SM11x) family matmul kernel with programmable pointwise+auxiliary input/output epilogue.
//! It is intended for heavy epilogue-fusion cases (e.g. fused SwiGLU gate) where cuDNN currently unperforms
//! and which cublas/cublasLt/nvjet do not support.
//!
//! Currently, it is built for a hardcoded tile size of 128x256 per CTA /
//! 256x256 per CTA-pair. It does not support a cluster size beyond the CTA-pair (2x1x1)
//! and thus does not use TMA multicast.
//! It supports bf16 inputs and outputs only.
//!
//! A and B can be either row- or column-major. Outputs and auxiliary inputs must be row-major.
//!
//! Epilogues can consume auxiliary bf16 row-major inputs and produce multiple
//! row-major bf16 outputs. Epilogue is overlapped with MMA via accumulator double-buffering.
//!
//! Epilogue uses subtiling, currently with a fixed subtile size of 128x64,
//! distributed across 8 epilogue warps each processing 32x32 elements.
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
//! * Warp 3: Epilogue TMA stores
//! * Warps 4..12: Epilogue compute warps

use crate::{thread_idx, warp_id};
use core::f32::consts::LOG2_E;
use ptx_std::{
    Denormals, F32Ext, MBarrier, MemoryOrdering, PhaseParity, Scope,
    barrier::BarrierAlignment,
    bf16, bf16x2, block_id, block_id_in_cluster, cluster, dynamic_smem_size, dynamic_smem_start,
    fence::{ProxyKind, fence_proxy},
    grid_dim,
    launch::{decrease_max_registers, increase_max_registers},
    memory::{CachePolicy, L2EvictionPriority, to_shared_address},
    tcgen05::{
        self, CtaGroup, CtaMask, F16ElementType, FloatAccumulator, InstrDescriptor, LdStShape,
        Major, MmaKind, MmaModifiers, Packing, SmemDescriptor, Swizzle, TmemAddr,
    },
    tma::{self, BulkWait, TmaDescriptor},
    warp_collectives::{LaneMask, elect},
};

const TM_CTA: u32 = 128;
const TM_CGA: u32 = 256;
const TN_CTA: u32 = 256;
const TN_CGA: u32 = 256;
/// Only half of the N dimension for the B tile is held per CTA due to the use of 2CTA MMA.
const TN_B_CTA: u32 = TN_CGA / CGA_SIZE;
const TK_CTA: u32 = 64;
const TK_MMA: u32 = 16;
const CGA_SIZE: u32 = 2;

const TM_EPI: u32 = 128;
const TN_EPI: u32 = 64;
const EPI_ELEMENTS_PER_THREAD: usize = 32;

#[derive(Copy, Clone, PartialEq, Eq)]
enum WarpRole {
    AbTmaLoad,
    MmaIssue,
    EpilogueTmaLoad,
    EpilogueTmaStore,
    Epilogue,
}

#[inline(always)]
fn warp_role() -> WarpRole {
    match warp_id() {
        0 => WarpRole::AbTmaLoad,
        1 => WarpRole::MmaIssue,
        2 => WarpRole::EpilogueTmaLoad,
        3 => WarpRole::EpilogueTmaStore,
        _ => WarpRole::Epilogue,
    }
}

/// See CUTLASS `Sm100MmaPeerBitMask` for why this is needed....
#[inline(always)]
fn leader_cta_barrier(bar: &MBarrier) -> &MBarrier {
    unsafe { &*(((bar as *const MBarrier as usize) & !(1 << 24)) as *const MBarrier) }
}

struct Barriers<const AB_PIPE_DEPTH: usize, const EPILOGUE_PIPE_DEPTH: usize> {
    ab_slot_produced: [MBarrier; AB_PIPE_DEPTH],
    ab_slot_consumed: [MBarrier; AB_PIPE_DEPTH],
    epi_load_slot_produced: [MBarrier; EPILOGUE_PIPE_DEPTH],
    epi_load_slot_consumed: [MBarrier; EPILOGUE_PIPE_DEPTH],
    accumulator_slot_produced: [MBarrier; 2],
    accumulator_slot_consumed: [MBarrier; 2],
    epi_store_produced: MBarrier,
    epi_store_consumed: MBarrier,
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
        slot_idx * size_of::<bf16>() as u32 * (TM_CTA * TK_CTA + TN_B_CTA * TK_CTA)
    }

    #[inline(always)]
    pub fn a_staging_slot(&self, slot_idx: u32) -> *mut u8 {
        let offset = self.a_staging_slot_offset(slot_idx);
        unsafe { self.base.add(offset as usize) }
    }

    #[inline(always)]
    pub fn b_staging_slot(&self, slot_idx: u32) -> *mut u8 {
        let offset =
            self.a_staging_slot_offset(slot_idx) + size_of::<bf16>() as u32 * TM_CTA * TK_CTA;
        unsafe { self.base.add(offset as usize) }
    }

    #[inline(always)]
    fn epi_load_staging_slot_offset(&self, slot_idx: u32) -> u32 {
        self.a_staging_slot_offset(AB_PIPE_DEPTH as u32)
            + size_of::<bf16>() as u32 * TM_EPI * TN_EPI * slot_idx
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
    pub fn barriers_raw(&self) -> *mut Barriers<AB_PIPE_DEPTH, EPILOGUE_PIPE_DEPTH> {
        let offset = self.epi_load_staging_slot_offset(EPILOGUE_PIPE_DEPTH as u32 + 1);
        unsafe { self.base.add(offset as usize).cast() }
    }

    #[inline(always)]
    pub fn barriers(&self) -> &Barriers<AB_PIPE_DEPTH, EPILOGUE_PIPE_DEPTH> {
        unsafe { &*self.barriers_raw() }
    }

    #[inline(always)]
    pub fn tmem_addr_dst(&mut self) -> *mut TmemAddr {
        unsafe { self.barriers_raw().add(1).cast() }
    }

    #[inline(always)]
    pub fn required_space(&mut self) -> u32 {
        unsafe { self.tmem_addr_dst().add(1).byte_offset_from(self.base) as u32 }
    }
}

/// Should be called by one thread.
fn init_barriers<const AB_PIPE_DEPTH: usize, const EPILOGUE_PIPE_DEPTH: usize>(
    smem: &mut Smem<AB_PIPE_DEPTH, EPILOGUE_PIPE_DEPTH>,
) {
    unsafe {
        let barriers = smem.barriers();
        #[rustc_unroll(full)]
        for bar in &barriers.ab_slot_produced {
            bar.init(1);
        }
        #[rustc_unroll(full)]
        for bar in &barriers.ab_slot_consumed {
            bar.init(1);
        }
        #[rustc_unroll(full)]
        for bar in &barriers.accumulator_slot_produced {
            bar.init(1);
        }
        #[rustc_unroll(full)]
        for bar in &barriers.accumulator_slot_consumed {
            bar.init(512); // barrier is shared across the CTA-pair (only leader CTA's barrier is used)
        }
        #[rustc_unroll(full)]
        for bar in &barriers.epi_load_slot_produced {
            bar.init(1);
        }
        #[rustc_unroll(full)]
        for bar in &barriers.epi_load_slot_consumed {
            bar.init(256);
        }
        barriers.epi_store_produced.init(256);
        barriers.epi_store_consumed.init(1);
    }
}

struct State<
    'a,
    const AB_PIPE_DEPTH: usize,
    const EPILOGUE_PIPE_DEPTH: usize,
    const A_K_MAJOR: bool,
    const B_K_MAJOR: bool,
> {
    m: u32,
    n: u32,
    k: u32,
    smem: Smem<AB_PIPE_DEPTH, EPILOGUE_PIPE_DEPTH>,
    raster_along_m: bool,
    cga_panel_length: u32,
    cta_rank: u32,
    current_work_id: u32,
    current_cta_tm: u32,
    current_cta_tn: u32,
    a_tma: &'a TmaDescriptor,
    b_tma: &'a TmaDescriptor,
}

impl<
    'a,
    const AB_PIPE_DEPTH: usize,
    const EPILOGUE_PIPE_DEPTH: usize,
    const A_K_MAJOR: bool,
    const B_K_MAJOR: bool,
> State<'a, AB_PIPE_DEPTH, EPILOGUE_PIPE_DEPTH, A_K_MAJOR, B_K_MAJOR>
{
    fn advance_to_next_tile(&mut self) -> bool {
        self.current_work_id += grid_dim().x / CGA_SIZE;

        let num_m_cgas = self.m.div_ceil(TM_CGA);
        let num_n_cgas = self.n.div_ceil(TN_CGA);
        if self.current_work_id >= num_m_cgas * num_n_cgas {
            return false;
        }

        self.set_tile_from_work_id();

        true
    }

    fn set_tile_from_work_id(&mut self) {
        let num_m_cgas = self.m.div_ceil(TM_CGA);
        let num_n_cgas = self.n.div_ceil(TN_CGA);
        let (cga_tm, cga_tn) = if self.raster_along_m {
            // raster along M (column-major before swizzling)
            let tiles_per_panel = self.cga_panel_length * num_m_cgas;
            let panel = self.current_work_id / tiles_per_panel;
            let panel_start = panel * self.cga_panel_length;
            let this_panel_length = self.cga_panel_length.min(num_n_cgas - panel_start);
            let idx_in_panel = self.current_work_id - panel * tiles_per_panel;

            (
                idx_in_panel / this_panel_length,
                panel_start + idx_in_panel % this_panel_length,
            )
        } else {
            // raster along N (row-major before swizzling)
            let tiles_per_panel = self.cga_panel_length * num_n_cgas;
            let panel = self.current_work_id / tiles_per_panel;
            let panel_start = panel * self.cga_panel_length;
            let this_panel_length = self.cga_panel_length.min(num_m_cgas - panel_start);
            let idx_in_panel = self.current_work_id - panel * tiles_per_panel;

            (
                panel_start + idx_in_panel % this_panel_length,
                idx_in_panel / this_panel_length,
            )
        };

        self.current_cta_tm = cga_tm * (TM_CGA / TM_CTA) + self.cta_rank;
        self.current_cta_tn = cga_tn * (TN_CGA / TN_CTA);
    }

    #[inline(always)]
    pub fn run_ab_load_warp(&mut self) {
        unsafe {
            let is_elected = elect(LaneMask::FULL).is_leader;

            if is_elected {
                let mut slot_idx = 0;
                let mut parity = 0;
                let mut is_first_iter = true;
                let num_k_tiles = self.k.div_ceil(TK_CTA);
                let is_leader = self.cta_rank == 0;
                let cta_mask = Some(CtaMask::single(self.cta_rank));
                loop {
                    for k_tile_idx in 0..num_k_tiles {
                        if !is_first_iter {
                            // Wait for previous data in slot to be consumed
                            self.smem.barriers().ab_slot_consumed[slot_idx as usize]
                                .wait_scoped::<{ Scope::Cluster }, { MemoryOrdering::Acquire }>(
                                    PhaseParity::of_phase(parity),
                                );
                        }

                        if is_leader {
                            let bytes =
                                size_of::<bf16>() as u32 * (TM_CTA + TN_B_CTA) * TK_CTA * CGA_SIZE;
                            self.smem.barriers().ab_slot_produced[slot_idx as usize]
                                .arrive_expect_tx(bytes);
                        }
                        let produced_bar = leader_cta_barrier(
                            &self.smem.barriers().ab_slot_produced[slot_idx as usize],
                        );

                        if A_K_MAJOR {
                            tma::tma_load_tile_to_cluster::<2, { CtaGroup::Two }>(
                                self.smem.a_staging_slot(slot_idx),
                                produced_bar,
                                self.a_tma,
                                [
                                    (k_tile_idx * TK_CTA) as i32,
                                    (self.current_cta_tm * TM_CTA) as i32,
                                ],
                                cta_mask,
                                None,
                            );
                        } else {
                            // A is M-major. Split our 128x64 tile into two panels, each 64x64,
                            // packed next to each other.
                            tma::tma_load_tile_to_cluster::<2, { CtaGroup::Two }>(
                                self.smem.a_staging_slot(slot_idx),
                                produced_bar,
                                self.a_tma,
                                [
                                    (self.current_cta_tm * TM_CTA) as i32,
                                    (k_tile_idx * TK_CTA) as i32,
                                ],
                                cta_mask,
                                None,
                            );
                            tma::tma_load_tile_to_cluster::<2, { CtaGroup::Two }>(
                                self.smem
                                    .a_staging_slot(slot_idx)
                                    .add(64 * 64 * size_of::<bf16>()),
                                produced_bar,
                                self.a_tma,
                                [
                                    (self.current_cta_tm * TM_CTA + 64) as i32,
                                    (k_tile_idx * TK_CTA) as i32,
                                ],
                                cta_mask,
                                None,
                            );
                        }

                        if B_K_MAJOR {
                            tma::tma_load_tile_to_cluster::<2, { CtaGroup::Two }>(
                                self.smem.b_staging_slot(slot_idx),
                                produced_bar,
                                self.b_tma,
                                [
                                    (k_tile_idx * TK_CTA) as i32,
                                    (self.current_cta_tn * TN_CTA + self.cta_rank * TN_B_CTA)
                                        as i32,
                                ],
                                cta_mask,
                                None,
                            );
                        } else {
                            // B is N-major. Split our 64x128 tile into two panels, each 64x64,
                            // packed next to each other.
                            tma::tma_load_tile_to_cluster::<2, { CtaGroup::Two }>(
                                self.smem.b_staging_slot(slot_idx),
                                produced_bar,
                                self.b_tma,
                                [
                                    (self.current_cta_tn * TN_CTA + self.cta_rank * TN_B_CTA)
                                        as i32,
                                    (k_tile_idx * TK_CTA) as i32,
                                ],
                                cta_mask,
                                None,
                            );
                            tma::tma_load_tile_to_cluster::<2, { CtaGroup::Two }>(
                                self.smem
                                    .b_staging_slot(slot_idx)
                                    .add(64 * 64 * size_of::<bf16>()),
                                produced_bar,
                                self.b_tma,
                                [
                                    (self.current_cta_tn * TN_CTA + self.cta_rank * TN_B_CTA + 64)
                                        as i32,
                                    (k_tile_idx * TK_CTA) as i32,
                                ],
                                cta_mask,
                                None,
                            );
                        }

                        slot_idx = (slot_idx + 1) % AB_PIPE_DEPTH as u32;
                        if slot_idx == 0 {
                            if !is_first_iter {
                                parity ^= 1;
                            }
                            is_first_iter = false;
                        }
                    }

                    if !self.advance_to_next_tile() {
                        break;
                    }
                }
            }
        }
    }

    #[inline(always)]
    pub fn run_mma_warp(&mut self) {
        unsafe {
            let mut accumulator_tmem_addr = *self.smem.tmem_addr_dst();
            let mut accumulator_slot = 0;
            let mut accumulator_parity = 0;
            let mut is_first_tile = true;

            let is_elected = elect(LaneMask::FULL).is_leader;

            if is_elected && self.cta_rank == 0 {
                let mut ab_slot_idx = 0;
                let mut ab_parity = 0;
                let num_k_tiles = self.k.div_ceil(TK_CTA);

                loop {
                    for k_tile_idx in 0..num_k_tiles {
                        self.smem.barriers().ab_slot_produced[ab_slot_idx as usize]
                            .wait(PhaseParity::of_phase(ab_parity));

                        let a_smem_addr = to_shared_address(self.smem.a_staging_slot(ab_slot_idx));
                        let b_smem_addr = to_shared_address(self.smem.b_staging_slot(ab_slot_idx));

                        #[rustc_unroll(full)]
                        for k_subtile_idx in 0..(TK_CTA / TK_MMA) {
                            let k_offset = k_subtile_idx * TK_MMA * size_of::<bf16>() as u32;
                            let a_smem_desc = if A_K_MAJOR {
                                SmemDescriptor::from_shared_address(
                                    a_smem_addr + k_offset,
                                    0,
                                    8 * 128,
                                    Swizzle::Bytes128,
                                )
                            } else {
                                SmemDescriptor::from_shared_address(
                                    a_smem_addr + k_offset * 64,
                                    (64 * 64 * size_of::<bf16>()) as u32,
                                    128 * 8,
                                    Swizzle::Bytes128,
                                )
                            };
                            let b_smem_desc = if B_K_MAJOR {
                                SmemDescriptor::from_shared_address(
                                    b_smem_addr + k_offset,
                                    0,
                                    8 * 128,
                                    Swizzle::Bytes128,
                                )
                            } else {
                                SmemDescriptor::from_shared_address(
                                    b_smem_addr + k_offset * 64,
                                    (64 * 64 * size_of::<bf16>()) as u32,
                                    128 * 8,
                                    Swizzle::Bytes128,
                                )
                            };
                            let instr_desc = InstrDescriptor::<{ MmaKind::F16 }>::new(
                                F16ElementType::Bf16,
                                F16ElementType::Bf16,
                                FloatAccumulator::F32,
                                TM_CGA,
                                TN_CGA,
                            )
                            .with_major_a(if A_K_MAJOR { Major::K } else { Major::Mn })
                            .with_major_b(if B_K_MAJOR { Major::K } else { Major::Mn });
                            tcgen05::mma::<
                                { CtaGroup::Two },
                                { MmaModifiers::DEFAULT },
                                { MmaKind::F16 },
                            >(
                                accumulator_tmem_addr,
                                a_smem_desc,
                                b_smem_desc,
                                instr_desc,
                                None,
                                k_tile_idx > 0 || k_subtile_idx > 0,
                            );
                        }

                        tcgen05::commit_multicast::<{ CtaGroup::Two }>(
                            &self.smem.barriers().ab_slot_consumed[ab_slot_idx as usize],
                            CtaMask::first(CGA_SIZE),
                        );

                        ab_slot_idx = (ab_slot_idx + 1) % AB_PIPE_DEPTH as u32;
                        if ab_slot_idx == 0 {
                            ab_parity ^= 1;
                        }
                    }

                    tcgen05::commit_multicast::<{ CtaGroup::Two }>(
                        &self.smem.barriers().accumulator_slot_produced[accumulator_slot as usize],
                        CtaMask::first(CGA_SIZE),
                    );

                    if !self.advance_to_next_tile() {
                        break;
                    }

                    // Swap accumulator between columns 0-255 and 256-511
                    accumulator_tmem_addr =
                        TmemAddr::from_bits(accumulator_tmem_addr.to_bits() ^ 256);
                    accumulator_slot ^= 1;

                    if accumulator_slot == 0 {
                        if !is_first_tile {
                            accumulator_parity ^= 1;
                        }
                        is_first_tile = false;
                    }

                    if !is_first_tile {
                        self.smem.barriers().accumulator_slot_consumed[accumulator_slot as usize]
                            .wait_scoped::<{ Scope::Cluster }, { MemoryOrdering::Acquire }>(
                                PhaseParity::of_phase(accumulator_parity),
                            );
                        tcgen05::fence_after_thread_sync();
                    }
                }
            }
        }
    }

    #[inline(always)]
    pub fn run_epi_load_warp<E: Epilogue<'a>>(&mut self, aux_input_descs: E::AuxInputDescs) {
        if E::NUM_AUX_INPUTS == 0 {
            return;
        }

        unsafe {
            let is_elected = elect(LaneMask::FULL).is_leader;
            if is_elected {
                let mut epi_slot_idx = 0u32;
                let mut epi_parity = 0;
                let mut is_first_iter = true;
                loop {
                    #[rustc_unroll(full)]
                    for subtile_n in 0..TN_CTA / TN_EPI {
                        #[rustc_unroll(full)]
                        for &aux_input in aux_input_descs.as_ref() {
                            if !is_first_iter {
                                self.smem.barriers().epi_load_slot_consumed[epi_slot_idx as usize]
                                    .wait(PhaseParity::of_phase(epi_parity));
                            }

                            let bytes = size_of::<bf16>() as u32 * TM_EPI * TN_EPI;
                            self.smem.barriers().epi_load_slot_produced[epi_slot_idx as usize]
                                .arrive_expect_tx(bytes);

                            // We expect that auxiliary inputs likely won't be re-accessed soon.
                            let cache_hint = CachePolicy::new::<{ L2EvictionPriority::First }>();
                            tma::tma_load_tile::<2>(
                                self.smem.epi_load_staging_slot(epi_slot_idx),
                                &self.smem.barriers().epi_load_slot_produced[epi_slot_idx as usize],
                                aux_input,
                                [
                                    (self.current_cta_tn * TN_CTA + subtile_n * TN_EPI) as i32,
                                    (self.current_cta_tm * TM_CTA) as i32,
                                ],
                                Some(cache_hint),
                            );

                            epi_slot_idx = (epi_slot_idx + 1) % EPILOGUE_PIPE_DEPTH as u32;
                            if epi_slot_idx == 0 {
                                if !is_first_iter {
                                    epi_parity ^= 1;
                                }
                                is_first_iter = false;
                            }
                        }
                    }

                    if !self.advance_to_next_tile() {
                        break;
                    }
                }
            }
        }
    }

    #[inline(always)]
    pub fn run_epi_store_warp<E: Epilogue<'a>>(&mut self, epilogue: E) {
        unsafe {
            let is_elected = elect(LaneMask::FULL).is_leader;
            if is_elected {
                let mut store_parity = 0;
                loop {
                    #[rustc_unroll(full)]
                    for subtile_n in 0..(TN_CTA / TN_EPI) {
                        #[rustc_unroll(full)]
                        for &(tma_desc, cache_hint) in epilogue.output_descs().as_ref() {
                            // Wait for epilogue warps to stage to SMEM
                            self.smem
                                .barriers()
                                .epi_store_produced
                                .wait(PhaseParity::of_phase(store_parity));

                            tma::tma_store_tile::<2>(
                                self.smem.epi_store_staging_slot(),
                                tma_desc,
                                [
                                    (self.current_cta_tn * TN_CTA + subtile_n * TN_EPI) as i32,
                                    (self.current_cta_tm * TM_CTA) as i32,
                                ],
                                Some(cache_hint),
                            );
                            tma::bulk_commit_group();

                            tma::bulk_wait_group::<0, { BulkWait::SourceRead }>();
                            self.smem.barriers().epi_store_consumed.arrive();

                            store_parity ^= 1;
                        }
                    }

                    if !self.advance_to_next_tile() {
                        break;
                    }
                }
                tma::bulk_wait_group::<0, { BulkWait::Completion }>();
            }
        }
    }

    #[inline(always)]
    pub fn run_epi_warps<E: Epilogue<'a>>(&mut self, mut epilogue: E) {
        unsafe {
            let thread_m = (thread_idx() - 128) % 128;
            let warp_tmem_lane_base = warp_id() % 4 * 32;
            // Offset of this warp's elements within a subtile.
            let warp_n = (thread_idx() - 128) / 128 * 32;

            let mut accum_slot_idx = 0;
            let mut accum_parity = 0;
            let mut epi_slot_idx = 0;
            let mut epi_parity = 0;
            let mut has_stored = false;
            let mut store_parity = 0;

            loop {
                // Wait for accumulator to be ready in TMEM
                self.smem.barriers().accumulator_slot_produced[accum_slot_idx as usize]
                    .wait_scoped::<{ Scope::Cluster }, { MemoryOrdering::Acquire }>(
                        PhaseParity::of_phase(accum_parity),
                    );
                tcgen05::fence_after_thread_sync();

                let base_tmem_addr = (*self.smem.tmem_addr_dst()).to_bits() + (accum_slot_idx << 8);

                let num_subtiles_n = TN_CTA / TN_EPI;
                #[rustc_unroll(full)]
                for subtile_n in 0..num_subtiles_n {
                    let tmem_addr = base_tmem_addr
                        + (warp_tmem_lane_base << 16)
                        + (subtile_n * TN_EPI)
                        + warp_n;
                    let mma_result = tcgen05::load::<
                        { LdStShape::S32x32b },
                        { Packing::None },
                        f32,
                        EPI_ELEMENTS_PER_THREAD,
                    >(TmemAddr::from_bits(tmem_addr));
                    tcgen05::wait_load();

                    if subtile_n == num_subtiles_n - 1 {
                        // We're done with the TMEM accumulator once we've loaded the last subtile.
                        tcgen05::fence_before_thread_sync();
                        MBarrier::arrive_remote::<{ Scope::Cluster }, { MemoryOrdering::Release }>(
                            self.smem.barriers().accumulator_slot_consumed[accum_slot_idx as usize]
                                .map_to_cta_rank(0),
                            1,
                        );
                    }

                    let mut write_output = #[inline(always)]
                    |thr_row: &[f32; EPI_ELEMENTS_PER_THREAD]| {
                        // Convert to bf16.
                        let mut thr_row_bf16 = [bf16x2::ZERO; EPI_ELEMENTS_PER_THREAD / 2];
                        #[rustc_unroll(full)]
                        for i in 0..EPI_ELEMENTS_PER_THREAD / 2 {
                            thr_row_bf16[i] =
                                bf16x2::from_f32x2(thr_row[i * 2], thr_row[i * 2 + 1]);
                        }

                        let smem = self.smem.epi_store_staging_slot().cast::<bf16>();
                        let row = thread_m;
                        let smem_row = smem.add((row * TN_EPI) as usize).cast::<u128>();

                        if has_stored {
                            // Wait for the previous TMA store to finish reading smem.
                            self.smem
                                .barriers()
                                .epi_store_consumed
                                .wait(PhaseParity::of_phase(store_parity));
                            store_parity ^= 1;
                        }

                        // Store to SMEM panel with vectorized stores, in TMA 128B swizzled layout
                        // (noting that the swizzle pattern width of 128 B = 64 elements = TN_EPI)
                        #[rustc_unroll(full)]
                        for chunk in 0..EPI_ELEMENTS_PER_THREAD / 8 {
                            let src = thr_row_bf16.as_ptr().cast::<u128>().add(chunk);
                            let col = chunk as u32 + warp_n / 8;
                            let col_swizzle = col ^ (row & 0b111);
                            smem_row.add(col_swizzle as usize).write(src.read());
                        }

                        fence_proxy::<{ ProxyKind::AsyncSharedCta }>();
                        self.smem.barriers().epi_store_produced.arrive();

                        has_stored = true;
                    };

                    epilogue.visit_mma_result(mma_result, &mut write_output);

                    #[rustc_unroll(full)]
                    for aux_idx in 0..E::NUM_AUX_INPUTS {
                        self.smem.barriers().epi_load_slot_produced[epi_slot_idx as usize]
                            .wait(PhaseParity::of_phase(epi_parity));

                        // Load from SMEM, convert to f32, feed to epilogue.
                        let mut thr_row = [0.0f32; EPI_ELEMENTS_PER_THREAD];

                        let smem = self.smem.epi_load_staging_slot(epi_slot_idx).cast::<bf16>();
                        let row = thread_m;
                        let smem_row = smem.add((row * TN_EPI) as usize).cast::<u128>();

                        // Load pattern mirrors the store pattern above.
                        #[rustc_unroll(full)]
                        for chunk in 0..EPI_ELEMENTS_PER_THREAD / 8 {
                            let col = chunk as u32 + warp_n / 8;
                            let col_swizzle = col ^ (row & 0b111);
                            let src = smem_row.add(col_swizzle as usize).read();
                            let src = bytemuck::cast::<u128, [bf16x2; 4]>(src);

                            #[rustc_unroll(full)]
                            for i in 0..src.len() {
                                let [a, b] = src[i].to_f32x2();
                                thr_row[chunk * 8 + i * 2] = a;
                                thr_row[chunk * 8 + i * 2 + 1] = b;
                            }
                        }

                        self.smem.barriers().epi_load_slot_consumed[epi_slot_idx as usize].arrive();

                        epilogue.visit_aux_input(aux_idx as u32, thr_row, &mut write_output);

                        epi_slot_idx = (epi_slot_idx + 1) % EPILOGUE_PIPE_DEPTH as u32;
                        if epi_slot_idx == 0 {
                            epi_parity ^= 1;
                        }
                    }
                }

                if !self.advance_to_next_tile() {
                    break;
                }

                accum_slot_idx ^= 1;
                if accum_slot_idx == 0 {
                    accum_parity ^= 1;
                }
            }
        }
    }
}

/// Stores state and processes data for the epilogue.
/// Each thread handles a row of 32 elements per epilogue subtile.
/// After the last call to `visit_aux_input` (or `visit_mma_result` if no aux inputs), the epilogue should have
/// written to all its output tensors via write_output.
trait Epilogue<'a> {
    const NUM_AUX_INPUTS: usize;
    /// `[&'a TmaDescriptor; Self::NUM_AUX_INPUTS]`
    type AuxInputDescs: AsRef<[&'a TmaDescriptor]>;
    /// `[(&'a TmaDescriptor, CacheHint); NUM_OUTPUTS]`
    type OutputDescs: AsRef<[(&'a TmaDescriptor, CacheHint)]>;

    fn aux_input_descs(&self) -> Self::AuxInputDescs;

    fn output_descs(&self) -> Self::OutputDescs;

    fn visit_mma_result(
        &mut self,
        result: [f32; EPI_ELEMENTS_PER_THREAD],
        write_output: impl FnMut(&[f32; EPI_ELEMENTS_PER_THREAD]),
    );

    fn visit_aux_input(
        &mut self,
        idx: u32,
        aux_data: [f32; EPI_ELEMENTS_PER_THREAD],
        write_output: impl FnMut(&[f32; EPI_ELEMENTS_PER_THREAD]),
    );
}

type CacheHint = CachePolicy;

struct IdentityEpilogue<'a> {
    out_tma: &'a TmaDescriptor,
}

impl<'a> Epilogue<'a> for IdentityEpilogue<'a> {
    const NUM_AUX_INPUTS: usize = 0;
    type AuxInputDescs = [&'a TmaDescriptor; 0];
    type OutputDescs = [(&'a TmaDescriptor, CacheHint); 1];

    #[inline(always)]
    fn aux_input_descs(&self) -> Self::AuxInputDescs {
        []
    }

    #[inline(always)]
    fn output_descs(&self) -> Self::OutputDescs {
        [(
            self.out_tma,
            CachePolicy::new::<{ L2EvictionPriority::Unchanged }>(),
        )]
    }

    #[inline(always)]
    fn visit_mma_result(
        &mut self,
        result: [f32; EPI_ELEMENTS_PER_THREAD],
        mut write_output: impl FnMut(&[f32; EPI_ELEMENTS_PER_THREAD]),
    ) {
        write_output(&result);
    }

    #[inline(always)]
    fn visit_aux_input(
        &mut self,
        _idx: u32,
        _aux_data: [f32; EPI_ELEMENTS_PER_THREAD],
        _write_output: impl FnMut(&[f32; EPI_ELEMENTS_PER_THREAD]),
    ) {
        unreachable!()
    }
}

#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    (1.0 + (-x * LOG2_E).exp2_approx::<{ Denormals::FlushToZero }>()).rcp_approx_ftz()
}

#[inline(always)]
fn silu(x: f32) -> f32 {
    silu_using_sigmoid(x, sigmoid(x))
}

/// SiLU with precomputed sigmoid(x).
#[inline(always)]
fn silu_using_sigmoid(x: f32, sigmoid_x: f32) -> f32 {
    x * sigmoid_x
}

#[inline(always)]
#[allow(unused)]
fn dsilu(dy: f32, x: f32) -> f32 {
    dsilu_using_sigmoid(dy, x, sigmoid(x))
}

/// dSiLU with precomputed sigmoid(x).
#[inline(always)]
fn dsilu_using_sigmoid(dy: f32, x: f32, sigmoid_x: f32) -> f32 {
    dy * (sigmoid_x + x * sigmoid_x * (1.0 - sigmoid_x))
}

/// Epilogue with two outputs:
/// 1. G = mma_output
/// 2. Y = SiLU(mma_output) * X, where X is an auxiliary input coming from up-projection
struct SwiGluGateTrainFwdEpilogue<'a> {
    x_desc: &'a TmaDescriptor,
    g_desc: &'a TmaDescriptor,
    y_desc: &'a TmaDescriptor,
    vals: [f32; EPI_ELEMENTS_PER_THREAD],
}

impl<'a> Epilogue<'a> for SwiGluGateTrainFwdEpilogue<'a> {
    const NUM_AUX_INPUTS: usize = 1;
    type AuxInputDescs = [&'a TmaDescriptor; 1];
    type OutputDescs = [(&'a TmaDescriptor, CacheHint); 2];

    #[inline(always)]
    fn aux_input_descs(&self) -> Self::AuxInputDescs {
        [self.x_desc]
    }

    #[inline(always)]
    fn output_descs(&self) -> Self::OutputDescs {
        [
            (
                self.g_desc,
                CachePolicy::new::<{ L2EvictionPriority::First }>(),
            ),
            (
                self.y_desc,
                CachePolicy::new::<{ L2EvictionPriority::Unchanged }>(),
            ),
        ]
    }

    #[inline(always)]
    fn visit_mma_result(
        &mut self,
        result: [f32; EPI_ELEMENTS_PER_THREAD],
        mut write_output: impl FnMut(&[f32; EPI_ELEMENTS_PER_THREAD]),
    ) {
        write_output(&result);

        // Apply SiLU
        #[rustc_unroll(full)]
        for i in 0..EPI_ELEMENTS_PER_THREAD {
            self.vals[i] = silu(result[i]);
        }
    }

    #[inline(always)]
    fn visit_aux_input(
        &mut self,
        _idx: u32,
        aux_data: [f32; EPI_ELEMENTS_PER_THREAD],
        mut write_output: impl FnMut(&[f32; EPI_ELEMENTS_PER_THREAD]),
    ) {
        // Apply Hadamard product
        #[rustc_unroll(full)]
        for i in 0..EPI_ELEMENTS_PER_THREAD {
            self.vals[i] *= aux_data[i];
        }
        write_output(&self.vals);
    }
}

/// Epilogue for backward-pass fused SwiGLU down-projection dgrad.
/// Auxiliary inputs:
/// - G: output of gate-projection from forward pass
/// - X: output of up-projection from forward pass
///
/// Computes:
/// 1. dX = SiLU(G) * mma_result
/// 2. dG = dSiLU(X * mma_result, G)
struct SwiGluDownTrainDgradEpilogue<'a> {
    g_desc: &'a TmaDescriptor,
    x_desc: &'a TmaDescriptor,
    dg_desc: &'a TmaDescriptor,
    dx_desc: &'a TmaDescriptor,

    mma_result: [f32; EPI_ELEMENTS_PER_THREAD],
    g: [f32; EPI_ELEMENTS_PER_THREAD],
    sigmoid_g: [f32; EPI_ELEMENTS_PER_THREAD],
}

impl<'a> Epilogue<'a> for SwiGluDownTrainDgradEpilogue<'a> {
    const NUM_AUX_INPUTS: usize = 2;
    type AuxInputDescs = [&'a TmaDescriptor; 2];
    type OutputDescs = [(&'a TmaDescriptor, CacheHint); 2];

    #[inline(always)]
    fn aux_input_descs(&self) -> Self::AuxInputDescs {
        [self.g_desc, self.x_desc]
    }

    #[inline(always)]
    fn output_descs(&self) -> Self::OutputDescs {
        [
            (
                self.dx_desc,
                CachePolicy::new::<{ L2EvictionPriority::Unchanged }>(),
            ),
            (
                self.dg_desc,
                CachePolicy::new::<{ L2EvictionPriority::Unchanged }>(),
            ),
        ]
    }

    #[inline(always)]
    fn visit_mma_result(
        &mut self,
        result: [f32; EPI_ELEMENTS_PER_THREAD],
        _write_output: impl FnMut(&[f32; EPI_ELEMENTS_PER_THREAD]),
    ) {
        #[rustc_unroll(full)]
        for i in 0..EPI_ELEMENTS_PER_THREAD {
            self.mma_result[i] = result[i];
        }
    }

    #[inline(always)]
    fn visit_aux_input(
        &mut self,
        idx: u32,
        aux_data: [f32; EPI_ELEMENTS_PER_THREAD],
        mut write_output: impl FnMut(&[f32; EPI_ELEMENTS_PER_THREAD]),
    ) {
        match idx {
            0 => {
                // aux_data contains G; save it and compute dX.
                let mut dx = [0.0f32; EPI_ELEMENTS_PER_THREAD];
                #[rustc_unroll(full)]
                for i in 0..EPI_ELEMENTS_PER_THREAD {
                    self.g[i] = aux_data[i];
                    self.sigmoid_g[i] = sigmoid(self.g[i]);
                    dx[i] = silu_using_sigmoid(self.g[i], self.sigmoid_g[i]) * self.mma_result[i];
                }
                write_output(&dx);
            }
            1 => {
                // aux_data contains X; compute dG.
                let mut dg = [0.0f32; EPI_ELEMENTS_PER_THREAD];
                #[rustc_unroll(full)]
                for i in 0..EPI_ELEMENTS_PER_THREAD {
                    let dy_i = self.mma_result[i] * aux_data[i];
                    dg[i] = dsilu_using_sigmoid(dy_i, self.g[i], self.sigmoid_g[i]);
                }
                write_output(&dg);
            }
            _ => unreachable!(),
        }
    }
}

/// Note: we rely on every CGA having at least one work tile.
/// Host must clamp CGA count to this constraint.
fn kernel_template<
    'a,
    E: Epilogue<'a>,
    const AB_PIPE_DEPTH: usize,
    const EPILOGUE_PIPE_DEPTH: usize,
    const A_K_MAJOR: bool,
    const B_K_MAJOR: bool,
>(
    m: u32,
    n: u32,
    k: u32,
    raster_along_m: bool,
    cga_panel_length: u32,
    a_tma: &'a TmaDescriptor,
    b_tma: &'a TmaDescriptor,
    epilogue: E,
) {
    unsafe {
        let mut smem = Smem::<AB_PIPE_DEPTH, EPILOGUE_PIPE_DEPTH> {
            base: dynamic_smem_start(),
        };
        assert!(dynamic_smem_size() == smem.required_space());

        if thread_idx() == 0 {
            init_barriers(&mut smem);
            MBarrier::fence_init();
        }

        if warp_role() == WarpRole::MmaIssue {
            tcgen05::alloc::<{ CtaGroup::Two }>(smem.tmem_addr_dst(), 512);
            tcgen05::relinquish_alloc_permit::<{ CtaGroup::Two }>();
            tcgen05::fence_before_thread_sync();
        }

        cluster::barrier_sync::<{ BarrierAlignment::Aligned }>();
        tcgen05::fence_after_thread_sync();

        let mut state = State::<AB_PIPE_DEPTH, EPILOGUE_PIPE_DEPTH, A_K_MAJOR, B_K_MAJOR> {
            m,
            n,
            k,
            smem,
            raster_along_m,
            cga_panel_length,
            cta_rank: block_id_in_cluster().x,
            current_work_id: block_id().x / CGA_SIZE,
            current_cta_tm: 0,
            current_cta_tn: 0,
            a_tma,
            b_tma,
        };
        state.set_tile_from_work_id();

        let role = warp_role();
        match role {
            WarpRole::Epilogue => {
                increase_max_registers::<224>();
                state.run_epi_warps(epilogue);
            }
            _ => {
                decrease_max_registers::<48>();
                match role {
                    WarpRole::AbTmaLoad => {
                        state.run_ab_load_warp();
                    }
                    WarpRole::MmaIssue => state.run_mma_warp(),
                    WarpRole::EpilogueTmaLoad => {
                        state.run_epi_load_warp::<E>(epilogue.aux_input_descs());
                    }
                    WarpRole::EpilogueTmaStore => state.run_epi_store_warp(epilogue),
                    WarpRole::Epilogue => unreachable!(),
                }
            }
        }

        tcgen05::fence_before_thread_sync();
        cluster::barrier_sync::<{ BarrierAlignment::Aligned }>();
        if warp_role() == WarpRole::MmaIssue {
            tcgen05::fence_after_thread_sync();
            tcgen05::dealloc::<{ CtaGroup::Two }>(*smem.tmem_addr_dst(), 512);
        }
    }
}

macro_rules! kernel_variant {
    (
        $name_base:ident $name_suffix:ident($($(#[$attr:meta])? $arg_i:ident: $arg_t:ty),* $(,)?),
        epilogue = $epilogue:expr,
        AB_PIPE_DEPTH = $ab_pipe_depth:literal,
        EPILOGUE_PIPE_DEPTH = $epilogue_pipe_depth:literal,
        A_K_MAJOR = $a_k_major:literal,
        B_K_MAJOR = $b_k_major:literal
        $(,)?
    ) => {
        pastey::paste! {
            #[unsafe(no_mangle)]
            #[nvptx(max_threads_per_cta(384, 1, 1), min_ctas_per_sm(1), exact_cluster_dim(2, 1, 1))]
            pub unsafe extern "ptx-kernel" fn [< $name_base _ $name_suffix >](m: u32, n: u32, k: u32, raster_along_m: bool, cga_panel_length: u32, #[nvptx(grid_constant)] a_tma: &TmaDescriptor,
                    #[nvptx(grid_constant)] b_tma: &TmaDescriptor, $($(#[$attr])? $arg_i: &$arg_t),*) {
                kernel_template::<_, $ab_pipe_depth, $epilogue_pipe_depth, $a_k_major, $b_k_major>(
                    m,
                    n,
                    k,
                    raster_along_m,
                    cga_panel_length,
                    a_tma,
                    b_tma,
                    $epilogue
                );
            }
        }
    };
}

macro_rules! kernel_with_variants {
    (
        $name_base:ident($($(#[$attr:meta])? $arg_i:ident: $arg_t:ty),* $(,)?),
        epilogue = $epilogue:expr,
        AB_PIPE_DEPTH = $ab_pipe_depth:literal,
        EPILOGUE_PIPE_DEPTH = $epilogue_pipe_depth:literal
        $(,)?
    ) => {
        kernel_variant! {
            $name_base arow_brow($($(#[$attr])? $arg_i: $arg_t),*),
            epilogue = $epilogue,
            AB_PIPE_DEPTH = $ab_pipe_depth,
            EPILOGUE_PIPE_DEPTH = $epilogue_pipe_depth,
            A_K_MAJOR = true,
            B_K_MAJOR = false
        }
        kernel_variant! {
            $name_base arow_bcol($($(#[$attr])? $arg_i: $arg_t),*),
            epilogue = $epilogue,
            AB_PIPE_DEPTH = $ab_pipe_depth,
            EPILOGUE_PIPE_DEPTH = $epilogue_pipe_depth,
            A_K_MAJOR = true,
            B_K_MAJOR = true
        }
        kernel_variant! {
            $name_base acol_brow($($(#[$attr])? $arg_i: $arg_t),*),
            epilogue = $epilogue,
            AB_PIPE_DEPTH = $ab_pipe_depth,
            EPILOGUE_PIPE_DEPTH = $epilogue_pipe_depth,
            A_K_MAJOR = false,
            B_K_MAJOR = false
        }
        kernel_variant! {
            $name_base acol_bcol($($(#[$attr])? $arg_i: $arg_t),*),
            epilogue = $epilogue,
            AB_PIPE_DEPTH = $ab_pipe_depth,
            EPILOGUE_PIPE_DEPTH = $epilogue_pipe_depth,
            A_K_MAJOR = false,
            B_K_MAJOR = true
        }
    };
}

kernel_with_variants! {
    sky_matmul_sm10x_128x256_2cta_64x6(
        #[nvptx(grid_constant)] out_tma: TmaDescriptor
    ),
    epilogue = IdentityEpilogue { out_tma },
    AB_PIPE_DEPTH = 6,
    EPILOGUE_PIPE_DEPTH = 1
}

kernel_with_variants! {
    sky_matmul_swiglu_gate_train_fwd_sm10x_128x256_2cta_64x6_epi1(
        #[nvptx(grid_constant)] x_tma: TmaDescriptor,
        #[nvptx(grid_constant)] g_tma: TmaDescriptor,
        #[nvptx(grid_constant)] y_tma: TmaDescriptor,
    ),
    epilogue = SwiGluGateTrainFwdEpilogue { x_desc: x_tma, g_desc: g_tma, y_desc: y_tma, vals: [0.0; EPI_ELEMENTS_PER_THREAD] },
    AB_PIPE_DEPTH = 6,
    EPILOGUE_PIPE_DEPTH = 1
}

kernel_with_variants! {
    sky_matmul_swiglu_down_train_dgrad_sm10x_128x256_2cta_64x6_epi1(
        #[nvptx(grid_constant)] g_desc: TmaDescriptor,
        #[nvptx(grid_constant)] x_desc: TmaDescriptor,
        #[nvptx(grid_constant)] dx_desc: TmaDescriptor,
        #[nvptx(grid_constant)] dg_desc: TmaDescriptor,
    ),
    epilogue = SwiGluDownTrainDgradEpilogue { g_desc, x_desc, dg_desc, dx_desc,
        mma_result: [0.0; EPI_ELEMENTS_PER_THREAD], g: [0.0; EPI_ELEMENTS_PER_THREAD], sigmoid_g: [0.0; EPI_ELEMENTS_PER_THREAD] },
    AB_PIPE_DEPTH = 6,
    EPILOGUE_PIPE_DEPTH = 1
}
