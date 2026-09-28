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

use core::{f32::consts::LOG2_E, mem};

use cuda_device::{
    Barrier, DynamicSharedArray, TmaDescriptor,
    barrier::{
        fence_mbarrier_init_release_cluster, fence_proxy_async_shared_cta, mbarrier_arrive,
        mbarrier_arrive_cluster, mbarrier_arrive_expect_tx, mbarrier_init,
        mbarrier_try_wait_parity, mbarrier_try_wait_parity_cluster,
    },
    blockIdx_x, cache_policy,
    cluster::{cluster_ctaidX, map_shared_rank},
    cooperative_groups::{ThreadGroup, this_cluster},
    float::{ex2_approx_ftz_f32, rcp_approx_ftz_f32},
    setmaxnreg_dec, setmaxnreg_inc,
    shared::{cvta_generic_to_shared_offset, dynamic_smem_size},
    tcgen05::{
        Tcgen05AccumulatorType, Tcgen05ElementType, Tcgen05InstructionDescriptorBuilder,
        Tcgen05SmemDescriptorBuilder, Tcgen05SwizzleMode, tcgen05_alloc_cg2,
        tcgen05_commit_multicast_cg2, tcgen05_dealloc_cg2, tcgen05_fence_after_thread_sync,
        tcgen05_fence_before_thread_sync, tcgen05_ld_32x32b_x32_raw, tcgen05_load_wait,
        tcgen05_mma_f16_cg2, tcgen05_relinquish_alloc_permit_cg2,
    },
    threadIdx_x, tma,
    uniform::grid_dim_x,
    warp::{is_elected_sync, warp_id},
};
use cuda_device_ext::{bf16, bf16x2};
use cuda_macros::{cluster_launch, cuda_annotate, device, kernel, launch_bounds};

use crate::lane_id;

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
fn leader_cta_barrier(bar: *mut Barrier) -> *mut Barrier {
    ((bar as usize) & !(1 << 24)) as *mut Barrier
}

struct Barriers<const AB_PIPE_DEPTH: usize, const EPILOGUE_PIPE_DEPTH: usize> {
    ab_slot_produced: [Barrier; AB_PIPE_DEPTH],
    ab_slot_consumed: [Barrier; AB_PIPE_DEPTH],
    epi_load_slot_produced: [Barrier; EPILOGUE_PIPE_DEPTH],
    epi_load_slot_consumed: [Barrier; EPILOGUE_PIPE_DEPTH],
    accumulator_slot_produced: [Barrier; 2],
    accumulator_slot_consumed: [Barrier; 2],
    epi_store_produced: Barrier,
    epi_store_consumed: Barrier,
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
    pub fn barriers_mut(&mut self) -> &mut Barriers<AB_PIPE_DEPTH, EPILOGUE_PIPE_DEPTH> {
        unsafe { &mut *self.barriers_raw() }
    }

    #[inline(always)]
    pub fn barriers(&self) -> &Barriers<AB_PIPE_DEPTH, EPILOGUE_PIPE_DEPTH> {
        unsafe { &*self.barriers_raw() }
    }

    #[inline(always)]
    pub fn tmem_addr_dst(&mut self) -> *mut u32 {
        unsafe { self.barriers_raw().add(1).cast() }
    }

    #[inline(always)]
    pub fn required_space(&mut self) -> u32 {
        unsafe { self.tmem_addr_dst().add(1).byte_offset_from(self.base) as u32 }
    }
}

/// Should be called by one thread.
#[device]
fn init_barriers<const AB_PIPE_DEPTH: usize, const EPILOGUE_PIPE_DEPTH: usize>(
    smem: &mut Smem<AB_PIPE_DEPTH, EPILOGUE_PIPE_DEPTH>,
) {
    unsafe {
        let barriers = smem.barriers_mut();
        #[llvm_unroll]
        for bar in &mut barriers.ab_slot_produced {
            mbarrier_init(bar, 1);
        }
        #[llvm_unroll]
        for bar in &mut barriers.ab_slot_consumed {
            mbarrier_init(bar, 1);
        }
        #[llvm_unroll]
        for bar in &mut barriers.accumulator_slot_produced {
            mbarrier_init(bar, 1);
        }
        #[llvm_unroll]
        for bar in &mut barriers.accumulator_slot_consumed {
            mbarrier_init(bar, 512); // barrier is shared across the CTA-pair (only leader CTA's barrier is used)
        }
        #[llvm_unroll]
        for bar in &mut barriers.epi_load_slot_produced {
            mbarrier_init(bar, 1);
        }
        #[llvm_unroll]
        for bar in &mut barriers.epi_load_slot_consumed {
            mbarrier_init(bar, 256);
        }
        mbarrier_init(&mut barriers.epi_store_produced, 256);
        mbarrier_init(&mut barriers.epi_store_consumed, 1);
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
        self.current_work_id += grid_dim_x().get() / CGA_SIZE;

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
    #[cuda_annotate]
    pub fn run_ab_load_warp(&mut self) {
        unsafe {
            let is_elected = is_elected_sync(u32::MAX);

            if is_elected {
                let mut slot_idx = 0;
                let mut parity = 0;
                let mut is_first_iter = true;
                let num_k_tiles = self.k.div_ceil(TK_CTA);
                let is_leader = self.cta_rank == 0;
                let cta_mask = 1u16 << self.cta_rank;
                loop {
                    for k_tile_idx in 0..num_k_tiles {
                        if !is_first_iter {
                            // Wait for previous data in slot to be consumed
                            while !mbarrier_try_wait_parity_cluster(
                                &self.smem.barriers_mut().ab_slot_consumed[slot_idx as usize],
                                parity,
                            ) {}
                        }

                        if is_leader {
                            let bytes =
                                size_of::<bf16>() as u32 * (TM_CTA + TN_B_CTA) * TK_CTA * CGA_SIZE;
                            mbarrier_arrive_expect_tx(
                                &self.smem.barriers_mut().ab_slot_produced[slot_idx as usize],
                                1, // ignored, unsure why this param is in cuda-device
                                bytes,
                            );
                        }
                        let produced_bar = leader_cta_barrier(
                            &mut self.smem.barriers_mut().ab_slot_produced[slot_idx as usize],
                        );

                        if A_K_MAJOR {
                            tma::cp_async_bulk_tensor_2d_g2s_multicast_cg2(
                                self.smem.a_staging_slot(slot_idx),
                                self.a_tma,
                                (k_tile_idx * TK_CTA) as i32,
                                (self.current_cta_tm * TM_CTA) as i32,
                                produced_bar,
                                cta_mask,
                            );
                        } else {
                            // A is M-major. Split our 128x64 tile into two panels, each 64x64,
                            // packed next to each other.
                            tma::cp_async_bulk_tensor_2d_g2s_multicast_cg2(
                                self.smem.a_staging_slot(slot_idx),
                                self.a_tma,
                                (self.current_cta_tm * TM_CTA) as i32,
                                (k_tile_idx * TK_CTA) as i32,
                                produced_bar,
                                cta_mask,
                            );
                            tma::cp_async_bulk_tensor_2d_g2s_multicast_cg2(
                                self.smem
                                    .a_staging_slot(slot_idx)
                                    .add(64 * 64 * size_of::<bf16>()),
                                self.a_tma,
                                (self.current_cta_tm * TM_CTA + 64) as i32,
                                (k_tile_idx * TK_CTA) as i32,
                                produced_bar,
                                cta_mask,
                            );
                        }

                        if B_K_MAJOR {
                            tma::cp_async_bulk_tensor_2d_g2s_multicast_cg2(
                                self.smem.b_staging_slot(slot_idx),
                                self.b_tma,
                                (k_tile_idx * TK_CTA) as i32,
                                (self.current_cta_tn * TN_CTA + self.cta_rank * TN_B_CTA) as i32,
                                produced_bar,
                                cta_mask,
                            );
                        } else {
                            // B is N-major. Split our 64x128 tile into two panels, each 64x64,
                            // packed next to each other.
                            tma::cp_async_bulk_tensor_2d_g2s_multicast_cg2(
                                self.smem.b_staging_slot(slot_idx),
                                self.b_tma,
                                (self.current_cta_tn * TN_CTA + self.cta_rank * TN_B_CTA) as i32,
                                (k_tile_idx * TK_CTA) as i32,
                                produced_bar,
                                cta_mask,
                            );
                            tma::cp_async_bulk_tensor_2d_g2s_multicast_cg2(
                                self.smem
                                    .b_staging_slot(slot_idx)
                                    .add(64 * 64 * size_of::<bf16>()),
                                self.b_tma,
                                (self.current_cta_tn * TN_CTA + self.cta_rank * TN_B_CTA + 64)
                                    as i32,
                                (k_tile_idx * TK_CTA) as i32,
                                produced_bar,
                                cta_mask,
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
    #[cuda_annotate]
    pub fn run_mma_warp(&mut self) {
        unsafe {
            let mut accumulator_tmem_addr = *self.smem.tmem_addr_dst();
            let mut accumulator_slot = 0;
            let mut accumulator_parity = 0;
            let mut is_first_tile = true;

            let is_elected = is_elected_sync(u32::MAX);

            if is_elected && self.cta_rank == 0 {
                let mut ab_slot_idx = 0;
                let mut ab_parity = 0;
                let num_k_tiles = self.k.div_ceil(TK_CTA);

                loop {
                    for k_tile_idx in 0..num_k_tiles {
                        while !mbarrier_try_wait_parity(
                            &self.smem.barriers_mut().ab_slot_produced[ab_slot_idx as usize],
                            ab_parity,
                        ) {}

                        let a_smem_addr =
                            cvta_generic_to_shared_offset(self.smem.a_staging_slot(ab_slot_idx));
                        let b_smem_addr =
                            cvta_generic_to_shared_offset(self.smem.b_staging_slot(ab_slot_idx));

                        #[llvm_unroll]
                        for k_subtile_idx in 0..(TK_CTA / TK_MMA) {
                            let k_offset =
                                (k_subtile_idx * TK_MMA * size_of::<bf16>() as u32) as u64;
                            let a_smem_desc = if A_K_MAJOR {
                                Tcgen05SmemDescriptorBuilder::new()
                                    .address(a_smem_addr + k_offset)
                                    .base_offset(0)
                                    .swizzle(Tcgen05SwizzleMode::Swizzle128B)
                                    .stride_bytes(8 * 128)
                                    .build()
                            } else {
                                Tcgen05SmemDescriptorBuilder::new()
                                    .address(a_smem_addr + k_offset * 64)
                                    .base_offset(0)
                                    .swizzle(Tcgen05SwizzleMode::Swizzle128B)
                                    .stride_bytes(128 * 8)
                                    .leading_dim_bytes((64 * 64 * size_of::<bf16>()) as u32)
                                    .build()
                            };
                            let b_smem_desc = if B_K_MAJOR {
                                Tcgen05SmemDescriptorBuilder::new()
                                    .address(b_smem_addr + k_offset)
                                    .base_offset(0)
                                    .swizzle(Tcgen05SwizzleMode::Swizzle128B)
                                    .stride_bytes(8 * 128)
                                    .build()
                            } else {
                                Tcgen05SmemDescriptorBuilder::new()
                                    .address(b_smem_addr + k_offset * 64)
                                    .base_offset(0)
                                    .swizzle(Tcgen05SwizzleMode::Swizzle128B)
                                    .stride_bytes(128 * 8)
                                    .leading_dim_bytes((64 * 64 * size_of::<bf16>()) as u32)
                                    .build()
                            };
                            let instr_desc = Tcgen05InstructionDescriptorBuilder::new()
                                .a_type(Tcgen05ElementType::BF16)
                                .b_type(Tcgen05ElementType::BF16)
                                .transpose_a(!A_K_MAJOR)
                                .transpose_b(!B_K_MAJOR)
                                .accumulator_type(Tcgen05AccumulatorType::F32)
                                .m(TM_CGA)
                                .n(TN_CGA)
                                .build();
                            tcgen05_mma_f16_cg2(
                                accumulator_tmem_addr,
                                a_smem_desc.raw(),
                                b_smem_desc.raw(),
                                instr_desc.raw(),
                                k_tile_idx > 0 || k_subtile_idx > 0,
                            );
                        }

                        tcgen05_commit_multicast_cg2(
                            (&mut self.smem.barriers_mut().ab_slot_consumed[ab_slot_idx as usize]
                                as *mut Barrier)
                                .cast(),
                            0b11,
                        );

                        ab_slot_idx = (ab_slot_idx + 1) % AB_PIPE_DEPTH as u32;
                        if ab_slot_idx == 0 {
                            ab_parity ^= 1;
                        }
                    }

                    tcgen05_commit_multicast_cg2(
                        (&mut self.smem.barriers_mut().accumulator_slot_produced
                            [accumulator_slot as usize] as *mut Barrier)
                            .cast(),
                        0b11,
                    );

                    if !self.advance_to_next_tile() {
                        break;
                    }

                    // Swap accumulator between columns 0-255 and 256-511
                    accumulator_tmem_addr ^= 256;
                    accumulator_slot ^= 1;

                    if accumulator_slot == 0 {
                        if !is_first_tile {
                            accumulator_parity ^= 1;
                        }
                        is_first_tile = false;
                    }

                    if !is_first_tile {
                        while !mbarrier_try_wait_parity_cluster(
                            &self.smem.barriers_mut().accumulator_slot_consumed
                                [accumulator_slot as usize],
                            accumulator_parity,
                        ) {}
                        tcgen05_fence_after_thread_sync();
                    }
                }
            }
        }
    }

    #[inline(always)]
    #[cuda_annotate]
    pub fn run_epi_load_warp<const NUM_AUX_INPUTS: usize>(
        &mut self,
        aux_input_descs: [&'a TmaDescriptor; NUM_AUX_INPUTS],
    ) {
        if NUM_AUX_INPUTS == 0 {
            return;
        }

        unsafe {
            let is_elected = is_elected_sync(u32::MAX);
            if is_elected {
                let mut epi_slot_idx = 0u32;
                let mut epi_parity = 0;
                let mut is_first_iter = true;
                loop {
                    #[llvm_unroll]
                    for subtile_n in 0..TN_CTA / TN_EPI {
                        #[llvm_unroll]
                        for &aux_input in &aux_input_descs {
                            if !is_first_iter {
                                while !mbarrier_try_wait_parity(
                                    &self.smem.barriers_mut().epi_load_slot_consumed
                                        [epi_slot_idx as usize],
                                    epi_parity,
                                ) {}
                            }

                            let bytes = size_of::<bf16>() as u32 * TM_EPI * TN_EPI;
                            mbarrier_arrive_expect_tx(
                                &self.smem.barriers_mut().epi_load_slot_produced
                                    [epi_slot_idx as usize],
                                1,
                                bytes,
                            );

                            // We expect that auxiliary inputs likely won't be re-accessed soon.
                            let cache_hint = cache_policy::createpolicy_fractional_evict_first(1.0);
                            tma::cp_async_bulk_tensor_2d_g2s_cache_hint(
                                self.smem.epi_load_staging_slot(epi_slot_idx),
                                aux_input,
                                (self.current_cta_tn * TN_CTA + subtile_n * TN_EPI) as i32,
                                (self.current_cta_tm * TM_CTA) as i32,
                                &mut self.smem.barriers_mut().epi_load_slot_produced
                                    [epi_slot_idx as usize],
                                cache_hint,
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
    #[cuda_annotate]
    pub fn run_epi_store_warp<E: Epilogue<'a>>(&mut self, epilogue: E)
    where
        [(); E::NUM_OUTPUTS]:,
    {
        unsafe {
            let is_elected = is_elected_sync(u32::MAX);
            if is_elected {
                let mut store_parity = 0;
                loop {
                    #[llvm_unroll]
                    for subtile_n in 0..(TN_CTA / TN_EPI) {
                        #[llvm_unroll]
                        for &(tma_desc, cache_hint) in &epilogue.output_descs() {
                            // Wait for epilogue warps to stage to SMEM
                            while !mbarrier_try_wait_parity(
                                &self.smem.barriers().epi_store_produced,
                                store_parity,
                            ) {}

                            tma::cp_async_bulk_tensor_2d_s2g_cache_hint(
                                self.smem.epi_store_staging_slot(),
                                tma_desc,
                                (self.current_cta_tn * TN_CTA + subtile_n * TN_EPI) as i32,
                                (self.current_cta_tm * TM_CTA) as i32,
                                cache_hint,
                            );
                            tma::cp_async_bulk_commit_group();

                            tma::cp_async_bulk_wait_group_read(0);
                            mbarrier_arrive(&self.smem.barriers().epi_store_consumed);

                            store_parity ^= 1;
                        }
                    }

                    if !self.advance_to_next_tile() {
                        break;
                    }
                }
                tma::cp_async_bulk_wait_group(0);
            }
        }
    }

    #[inline(always)]
    #[cuda_annotate]
    pub fn run_epi_warps<E: Epilogue<'a>>(&mut self, mut epilogue: E) {
        unsafe {
            let thread_m = (threadIdx_x() - 128) % 128;
            let warp_tmem_lane_base = warp_id() % 4 * 32;
            // Offset of this warp's elements within a subtile.
            let warp_n = (threadIdx_x() - 128) / 128 * 32;

            let mut accum_slot_idx = 0;
            let mut accum_parity = 0;
            let mut epi_slot_idx = 0;
            let mut epi_parity = 0;
            let mut has_stored = false;
            let mut store_parity = 0;

            loop {
                // Wait for accumulator to be ready in TMEM
                while !mbarrier_try_wait_parity_cluster(
                    &self.smem.barriers_mut().accumulator_slot_produced[accum_slot_idx as usize],
                    accum_parity,
                ) {}
                tcgen05_fence_after_thread_sync();

                let base_tmem_addr = *self.smem.tmem_addr_dst() + (accum_slot_idx << 8);

                let num_subtiles_n = TN_CTA / TN_EPI;
                #[llvm_unroll]
                for subtile_n in 0..num_subtiles_n {
                    let tmem_addr = base_tmem_addr
                        + (warp_tmem_lane_base << 16)
                        + (subtile_n * TN_EPI)
                        + warp_n;
                    let mma_result = tcgen05_ld_32x32b_x32_raw(tmem_addr);
                    tcgen05_load_wait();

                    if subtile_n == num_subtiles_n - 1 {
                        // We're done with the TMEM accumulator once we've loaded the last subtile.
                        tcgen05_fence_before_thread_sync();
                        mbarrier_arrive_cluster(cvta_generic_to_shared_offset(
                            map_shared_rank(
                                &self.smem.barriers_mut().accumulator_slot_consumed
                                    [accum_slot_idx as usize],
                                0,
                            )
                            .cast::<u8>(),
                        ));
                    }

                    let mma_result = mem::transmute::<
                        [u32; EPI_ELEMENTS_PER_THREAD],
                        [f32; EPI_ELEMENTS_PER_THREAD],
                    >(mma_result.to_array());

                    let mut write_output = #[inline(always)]
                    |thr_row: &[f32; EPI_ELEMENTS_PER_THREAD]| {
                        // Convert to bf16.
                        let mut thr_row_bf16 = [bf16x2::ZERO; EPI_ELEMENTS_PER_THREAD / 2];
                        #[llvm_unroll]
                        for i in 0..EPI_ELEMENTS_PER_THREAD / 2 {
                            thr_row_bf16[i] =
                                bf16x2::from_f32_pair(thr_row[i * 2], thr_row[i * 2 + 1]);
                        }

                        let smem = self.smem.epi_store_staging_slot().cast::<bf16>();
                        let row = thread_m;
                        let smem_row = smem.add((row * TN_EPI) as usize).cast::<u128>();

                        if has_stored {
                            // Wait for the previous TMA store to finish reading smem.
                            while !mbarrier_try_wait_parity(
                                &self.smem.barriers().epi_store_consumed,
                                store_parity,
                            ) {}
                            store_parity ^= 1;
                        }

                        // Store to SMEM panel with vectorized stores, in TMA 128B swizzled layout
                        // (noting that the swizzle pattern width of 128 B = 64 elements = TN_EPI)
                        #[llvm_unroll]
                        for chunk in 0..EPI_ELEMENTS_PER_THREAD / 8 {
                            let src = thr_row_bf16.as_ptr().cast::<u128>().add(chunk);
                            let col = chunk as u32 + warp_n / 8;
                            let col_swizzle = col ^ (row & 0b111);
                            smem_row.add(col_swizzle as usize).write(src.read());
                        }

                        fence_proxy_async_shared_cta();
                        mbarrier_arrive(&self.smem.barriers().epi_store_produced);

                        has_stored = true;
                    };

                    epilogue.visit_mma_result(mma_result, &mut write_output);

                    #[llvm_unroll]
                    for aux_idx in 0..E::NUM_AUX_INPUTS {
                        while !mbarrier_try_wait_parity(
                            &self.smem.barriers().epi_load_slot_produced[epi_slot_idx as usize],
                            epi_parity,
                        ) {}

                        // Load from SMEM, convert to f32, feed to epilogue.
                        let mut thr_row = [0.0f32; EPI_ELEMENTS_PER_THREAD];

                        let smem = self.smem.epi_load_staging_slot(epi_slot_idx).cast::<bf16>();
                        let row = thread_m;
                        let smem_row = smem.add((row * TN_EPI) as usize).cast::<u128>();

                        // Load pattern mirrors the store pattern above.
                        #[llvm_unroll]
                        for chunk in 0..EPI_ELEMENTS_PER_THREAD / 8 {
                            let col = chunk as u32 + warp_n / 8;
                            let col_swizzle = col ^ (row & 0b111);
                            let src = smem_row.add(col_swizzle as usize).read();
                            let src = bytemuck::cast::<u128, [bf16x2; 4]>(src);

                            #[llvm_unroll]
                            for i in 0..src.len() {
                                let (a, b) = src[i].to_f32_pair();
                                thr_row[chunk * 8 + i * 2] = a;
                                thr_row[chunk * 8 + i * 2 + 1] = b;
                            }
                        }

                        mbarrier_arrive(
                            &self.smem.barriers().epi_load_slot_consumed[epi_slot_idx as usize],
                        );

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
    const NUM_OUTPUTS: usize;

    fn aux_input_descs(&self) -> [&'a TmaDescriptor; Self::NUM_AUX_INPUTS];

    fn output_descs(&self) -> [(&'a TmaDescriptor, CacheHint); Self::NUM_OUTPUTS];

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

type CacheHint = u64;

struct IdentityEpilogue<'a> {
    out_tma: &'a TmaDescriptor,
}

impl<'a> Epilogue<'a> for IdentityEpilogue<'a> {
    const NUM_AUX_INPUTS: usize = 0;
    const NUM_OUTPUTS: usize = 1;

    #[inline(always)]
    fn aux_input_descs(&self) -> [&'a TmaDescriptor; Self::NUM_AUX_INPUTS] {
        []
    }

    #[inline(always)]
    fn output_descs(&self) -> [(&'a TmaDescriptor, CacheHint); Self::NUM_OUTPUTS] {
        [(
            self.out_tma,
            cache_policy::createpolicy_fractional_evict_unchanged(1.0),
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
    rcp_approx_ftz_f32(1.0 + ex2_approx_ftz_f32(-x * LOG2_E))
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
    const NUM_OUTPUTS: usize = 2;

    #[inline(always)]
    fn aux_input_descs(&self) -> [&'a TmaDescriptor; Self::NUM_AUX_INPUTS] {
        [self.x_desc]
    }

    #[inline(always)]
    fn output_descs(&self) -> [(&'a TmaDescriptor, CacheHint); Self::NUM_OUTPUTS] {
        [
            (
                self.g_desc,
                cache_policy::createpolicy_fractional_evict_first(1.0),
            ),
            (
                self.y_desc,
                cache_policy::createpolicy_fractional_evict_unchanged(1.0),
            ),
        ]
    }

    #[inline(always)]
    #[cuda_annotate]
    fn visit_mma_result(
        &mut self,
        result: [f32; EPI_ELEMENTS_PER_THREAD],
        mut write_output: impl FnMut(&[f32; EPI_ELEMENTS_PER_THREAD]),
    ) {
        write_output(&result);

        // Apply SiLU
        #[llvm_unroll]
        for i in 0..EPI_ELEMENTS_PER_THREAD {
            self.vals[i] = silu(result[i]);
        }
    }

    #[inline(always)]
    #[cuda_annotate]
    fn visit_aux_input(
        &mut self,
        _idx: u32,
        aux_data: [f32; EPI_ELEMENTS_PER_THREAD],
        mut write_output: impl FnMut(&[f32; EPI_ELEMENTS_PER_THREAD]),
    ) {
        // Apply Hadamard product
        #[llvm_unroll]
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
    const NUM_OUTPUTS: usize = 2;

    #[inline(always)]
    fn aux_input_descs(&self) -> [&'a TmaDescriptor; Self::NUM_AUX_INPUTS] {
        [self.g_desc, self.x_desc]
    }

    #[inline(always)]
    fn output_descs(&self) -> [(&'a TmaDescriptor, CacheHint); Self::NUM_OUTPUTS] {
        [
            (
                self.dx_desc,
                cache_policy::createpolicy_fractional_evict_unchanged(1.0),
            ),
            (
                self.dg_desc,
                cache_policy::createpolicy_fractional_evict_unchanged(1.0),
            ),
        ]
    }

    #[inline(always)]
    #[cuda_annotate]
    fn visit_mma_result(
        &mut self,
        result: [f32; EPI_ELEMENTS_PER_THREAD],
        _write_output: impl FnMut(&[f32; EPI_ELEMENTS_PER_THREAD]),
    ) {
        #[llvm_unroll]
        for i in 0..EPI_ELEMENTS_PER_THREAD {
            self.mma_result[i] = result[i];
        }
    }

    #[inline(always)]
    #[cuda_annotate]
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
                #[llvm_unroll]
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
                #[llvm_unroll]
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
#[cuda_annotate]
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
) where
    [(); E::NUM_AUX_INPUTS]:,
    [(); E::NUM_OUTPUTS]:,
{
    unsafe {
        let mut smem = Smem::<AB_PIPE_DEPTH, EPILOGUE_PIPE_DEPTH> {
            base: DynamicSharedArray::<u8, 1024>::get_raw(),
        };
        assert!(dynamic_smem_size() >= smem.required_space());

        if threadIdx_x() == 0 {
            init_barriers(&mut smem);
            fence_mbarrier_init_release_cluster();
        }

        if warp_role() == WarpRole::MmaIssue {
            tcgen05_alloc_cg2(smem.tmem_addr_dst(), 512);
            tcgen05_relinquish_alloc_permit_cg2();
            tcgen05_fence_before_thread_sync();
        }

        this_cluster().sync();
        tcgen05_fence_after_thread_sync();

        let mut state = State::<AB_PIPE_DEPTH, EPILOGUE_PIPE_DEPTH, A_K_MAJOR, B_K_MAJOR> {
            m,
            n,
            k,
            smem,
            raster_along_m,
            cga_panel_length,
            cta_rank: cluster_ctaidX(),
            current_work_id: blockIdx_x() / CGA_SIZE,
            current_cta_tm: 0,
            current_cta_tn: 0,
            a_tma,
            b_tma,
        };
        state.set_tile_from_work_id();

        let role = warp_role();
        match role {
            WarpRole::Epilogue => {
                setmaxnreg_inc::<224>();
                state.run_epi_warps(epilogue);
            }
            _ => {
                setmaxnreg_dec::<48>();
                match role {
                    WarpRole::AbTmaLoad => {
                        state.run_ab_load_warp();
                    }
                    WarpRole::MmaIssue => state.run_mma_warp(),
                    WarpRole::EpilogueTmaLoad => {
                        state.run_epi_load_warp(epilogue.aux_input_descs());
                    }
                    WarpRole::EpilogueTmaStore => state.run_epi_store_warp(epilogue),
                    WarpRole::Epilogue => unreachable!(),
                }
            }
        }

        tcgen05_fence_before_thread_sync();
        this_cluster().sync();
        if warp_role() == WarpRole::MmaIssue {
            tcgen05_fence_after_thread_sync();
            tcgen05_dealloc_cg2(*smem.tmem_addr_dst(), 512);
        }
    }
}

macro_rules! kernel_variant {
    (
        $name_base:ident $name_suffix:ident($($(#[$attr:tt])? $arg_i:ident: $arg_t:ty),* $(,)?),
        epilogue = $epilogue:expr,
        AB_PIPE_DEPTH = $ab_pipe_depth:literal,
        EPILOGUE_PIPE_DEPTH = $epilogue_pipe_depth:literal,
        A_K_MAJOR = $a_k_major:literal,
        B_K_MAJOR = $b_k_major:literal
        $(,)?
    ) => {
        pastey::paste! {
            #[kernel]
            #[launch_bounds(384, 1)]
            #[cluster_launch(2, 1, 1)]
            pub fn [< $name_base _ $name_suffix >](m: u32, n: u32, k: u32, raster_along_m: bool, cga_panel_length: u32, #[grid_constant] a_tma: &TmaDescriptor,
                    #[grid_constant] b_tma: &TmaDescriptor, $($(#[$attr])? $arg_i: &$arg_t),*) {
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
        $name_base:ident($($(#[$attr:tt])? $arg_i:ident: $arg_t:ty),* $(,)?),
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
        #[grid_constant] out_tma: TmaDescriptor
    ),
    epilogue = IdentityEpilogue { out_tma },
    AB_PIPE_DEPTH = 6,
    EPILOGUE_PIPE_DEPTH = 1
}

kernel_with_variants! {
    sky_matmul_swiglu_gate_train_fwd_sm10x_128x256_2cta_64x5_epi3(
        #[grid_constant] x_tma: TmaDescriptor,
        #[grid_constant] g_tma: TmaDescriptor,
        #[grid_constant] y_tma: TmaDescriptor,
    ),
    epilogue = SwiGluGateTrainFwdEpilogue { x_desc: x_tma, g_desc: g_tma, y_desc: y_tma, vals: [0.0; EPI_ELEMENTS_PER_THREAD] },
    AB_PIPE_DEPTH = 5,
    EPILOGUE_PIPE_DEPTH = 3
}

kernel_with_variants! {
    sky_matmul_swiglu_down_train_dgrad_sm10x_128x256_2cta_64x5_epi3(
        #[grid_constant] g_desc: TmaDescriptor,
        #[grid_constant] x_desc: TmaDescriptor,
        #[grid_constant] dx_desc: TmaDescriptor,
        #[grid_constant] dg_desc: TmaDescriptor,
    ),
    epilogue = SwiGluDownTrainDgradEpilogue { g_desc, x_desc, dg_desc, dx_desc,
        mma_result: [0.0; EPI_ELEMENTS_PER_THREAD], g: [0.0; EPI_ELEMENTS_PER_THREAD], sigmoid_g: [0.0; EPI_ELEMENTS_PER_THREAD] },
    AB_PIPE_DEPTH = 5,
    EPILOGUE_PIPE_DEPTH = 3
}
