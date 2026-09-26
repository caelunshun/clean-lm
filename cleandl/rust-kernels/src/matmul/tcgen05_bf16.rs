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
//! A must be row-major and B must be column-major (i.e. both K-major). Outputs must be row-major.
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
//! * Warp 3: Unused / register donor for epilogue warps
//! * Warps 4..12: Epilogue warps; TMA stores issued by warp 4

use core::mem;

use cuda_device::{
    Barrier, DynamicSharedArray, TmaDescriptor,
    barrier::{
        fence_proxy_async_shared_cta, mbarrier_arrive, mbarrier_arrive_cluster,
        mbarrier_arrive_expect_tx, mbarrier_init, mbarrier_try_wait, mbarrier_try_wait_parity,
        mbarrier_try_wait_parity_cluster,
    },
    blockIdx_x,
    cluster::{cluster_ctaidX, map_shared_rank},
    cooperative_groups::{ThreadGroup, this_cluster},
    setmaxnreg_dec, setmaxnreg_inc,
    shared::dynamic_smem_size,
    tcgen05::{
        Tcgen05AccumulatorType, Tcgen05ElementType, Tcgen05InstructionDescriptorBuilder,
        Tcgen05SmemDescriptorBuilder, Tcgen05SwizzleMode, tcgen05_alloc_cg2,
        tcgen05_commit_multicast_cg2, tcgen05_dealloc_cg2, tcgen05_fence_after_thread_sync,
        tcgen05_fence_before_thread_sync, tcgen05_ld_32x32b_x32_raw, tcgen05_load_wait,
        tcgen05_mma_f16_cg2,
    },
    threadIdx_x,
    tma::{self, cp_async_bulk_wait_group_read},
    uniform::grid_dim_x,
    warp::{is_elected_sync, warp_id},
};
use cuda_device_ext::{bf16, bf16x2};
use cuda_macros::{cuda_annotate, device, kernel};

use crate::lane_id;

const TM_CTA: u32 = 128;
const TM_CGA: u32 = 256;
const TN_CTA: u32 = 256;
const TN_CGA: u32 = 256;
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
    Unused,
    Epilogue,
}

#[inline(always)]
fn warp_role() -> WarpRole {
    match warp_id() {
        0 => WarpRole::AbTmaLoad,
        1 => WarpRole::MmaIssue,
        2 => WarpRole::EpilogueTmaLoad,
        3 => WarpRole::Unused,
        _ => WarpRole::Epilogue,
    }
}

struct Barriers<const AB_PIPE_DEPTH: usize, const EPILOGUE_PIPE_DEPTH: usize> {
    ab_slot_produced: [Barrier; AB_PIPE_DEPTH],
    ab_slot_consumed: [Barrier; AB_PIPE_DEPTH],
    epi_load_slot_produced: [Barrier; EPILOGUE_PIPE_DEPTH],
    epi_load_slot_consumed: [Barrier; EPILOGUE_PIPE_DEPTH],
    accumulator_slot_produced: [Barrier; 2],
    accumulator_slot_consumed: [Barrier; 2],
    epi_helper_barrier: Barrier,
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
        slot_idx * size_of::<bf16>() as u32 * (TM_CTA * TK_CTA + TN_CTA * TK_CTA)
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
    pub fn barriers_raw(&mut self) -> *mut Barriers<AB_PIPE_DEPTH, EPILOGUE_PIPE_DEPTH> {
        let offset = self.epi_load_staging_slot_offset(EPILOGUE_PIPE_DEPTH as u32 + 1);
        unsafe { self.base.add(offset as usize).cast() }
    }

    #[inline(always)]
    pub fn barriers(&mut self) -> &mut Barriers<AB_PIPE_DEPTH, EPILOGUE_PIPE_DEPTH> {
        unsafe { &mut *self.barriers_raw() }
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
        let barriers = smem.barriers();
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
        mbarrier_init(&mut barriers.epi_helper_barrier, 256);
    }
}

struct State<'a, const AB_PIPE_DEPTH: usize, const EPILOGUE_PIPE_DEPTH: usize> {
    m: u32,
    n: u32,
    k: u32,
    smem: Smem<AB_PIPE_DEPTH, EPILOGUE_PIPE_DEPTH>,
    raster_along_m: bool,
    cga_panel_length: u32,
    current_work_id: u32,
    current_cta_tm: u32,
    current_cta_tn: u32,
    a_tma: &'a TmaDescriptor,
    b_tma: &'a TmaDescriptor,
}

impl<'a, const AB_PIPE_DEPTH: usize, const EPILOGUE_PIPE_DEPTH: usize>
    State<'a, AB_PIPE_DEPTH, EPILOGUE_PIPE_DEPTH>
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
        if self.raster_along_m {
            // raster along M (column-major before swizzling)
            let panel = self.current_work_id / (self.cga_panel_length * num_m_cgas);
            let this_panel_length = if ((panel + 1) * self.cga_panel_length) > num_n_cgas {
                self.cga_panel_length - (((panel + 1) * self.cga_panel_length) - num_n_cgas)
            } else {
                self.cga_panel_length
            };
            let idx_in_panel = self.current_work_id % (this_panel_length * num_m_cgas);

            let cga_tm = idx_in_panel / this_panel_length;
            let cga_tn = idx_in_panel % this_panel_length;

            self.current_cta_tm = cga_tm * (TM_CGA / TM_CTA);
            self.current_cta_tn = cga_tn * (TN_CGA / TN_CTA);
        } else {
            // raster along N (row-major before swizzling)
            let panel = self.current_work_id / (self.cga_panel_length * num_n_cgas);
            let this_panel_length = if ((panel + 1) * self.cga_panel_length) > num_m_cgas {
                self.cga_panel_length - (((panel + 1) * self.cga_panel_length) - num_m_cgas)
            } else {
                self.cga_panel_length
            };
            let idx_in_panel = self.current_work_id % (this_panel_length * num_n_cgas);

            let cga_tm = idx_in_panel % this_panel_length;
            let cga_tn = idx_in_panel / this_panel_length;

            self.current_cta_tm = cga_tm * (TM_CGA / TM_CTA);
            self.current_cta_tn = cga_tn * (TN_CGA / TN_CTA);
        }
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
                loop {
                    for k_tile_idx in 0..num_k_tiles {
                        if !is_first_iter {
                            // Wait for previous data in slot to be consumed
                            while !mbarrier_try_wait_parity_cluster(
                                &self.smem.barriers().ab_slot_consumed[slot_idx as usize],
                                parity,
                            ) {}
                        }

                        let bytes = size_of::<bf16>() as u32 * (TM_CTA + TN_CTA) * TK_CTA;
                        mbarrier_arrive_expect_tx(
                            &self.smem.barriers().ab_slot_produced[slot_idx as usize],
                            1, // ignored, unsure why this param is in cuda-device
                            bytes,
                        );
                        tma::cp_async_bulk_tensor_2d_g2s(
                            self.smem.a_staging_slot(slot_idx),
                            self.a_tma,
                            (k_tile_idx * TK_CTA) as i32,
                            (self.current_cta_tm * TM_CTA) as i32,
                            &mut self.smem.barriers().ab_slot_produced[slot_idx as usize],
                        );
                        tma::cp_async_bulk_tensor_2d_g2s(
                            self.smem.b_staging_slot(slot_idx),
                            self.b_tma,
                            (k_tile_idx * TK_CTA) as i32,
                            (self.current_cta_tn * TN_CTA) as i32,
                            &mut self.smem.barriers().ab_slot_produced[slot_idx as usize],
                        );
                    }

                    slot_idx = (slot_idx + 1) % AB_PIPE_DEPTH as u32;
                    if slot_idx == 0 {
                        if !is_first_iter {
                            parity ^= 1;
                        }
                        is_first_iter = false;
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

            if is_elected && cluster_ctaidX().is_multiple_of(2) {
                let mut ab_slot_idx = 0;
                let mut ab_parity = 0;
                let mut last_ab_slot_idx = 0;
                let mut last_ab_parity = 0;
                let num_k_tiles = self.k.div_ceil(TK_CTA);

                loop {
                    for k_tile_idx in 0..num_k_tiles {
                        while !mbarrier_try_wait_parity(
                            &self.smem.barriers().ab_slot_produced[ab_slot_idx as usize],
                            ab_parity,
                        ) {}

                        #[llvm_unroll]
                        for k_subtile_idx in 0..(TK_CTA / TK_MMA) {
                            let a_smem_desc = Tcgen05SmemDescriptorBuilder::new()
                                .address(self.smem.a_staging_slot(ab_slot_idx).add(TK_MMA as usize)
                                    as usize as u64)
                                .base_offset(0)
                                .swizzle(Tcgen05SwizzleMode::Swizzle128B)
                                .stride_bytes(8 * 128)
                                .build();
                            let b_smem_desc = Tcgen05SmemDescriptorBuilder::new()
                                .address(self.smem.b_staging_slot(ab_slot_idx).add(TK_MMA as usize)
                                    as usize as u64)
                                .base_offset(0)
                                .swizzle(Tcgen05SwizzleMode::Swizzle128B)
                                .stride_bytes(8 * 128)
                                .build();
                            let instr_desc = Tcgen05InstructionDescriptorBuilder::new()
                                .a_type(Tcgen05ElementType::BF16)
                                .b_type(Tcgen05ElementType::BF16)
                                .accumulator_type(Tcgen05AccumulatorType::F32)
                                .m(128)
                                .n(256)
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
                            (&mut self.smem.barriers().ab_slot_consumed[ab_slot_idx as usize]
                                as *mut Barrier)
                                .cast(),
                            0b11,
                        );

                        last_ab_parity = ab_parity;
                        last_ab_slot_idx = ab_slot_idx;

                        ab_slot_idx = (ab_slot_idx + 1) % AB_PIPE_DEPTH as u32;
                        if ab_slot_idx == 0 {
                            ab_parity ^= 1;
                        }
                    }

                    while !mbarrier_try_wait_parity(
                        &self.smem.barriers().ab_slot_consumed[last_ab_slot_idx as usize],
                        last_ab_parity,
                    ) {}
                    tcgen05_fence_before_thread_sync();

                    mbarrier_arrive(
                        &self.smem.barriers().accumulator_slot_produced[accumulator_slot as usize],
                    );
                    mbarrier_arrive_cluster(map_shared_rank(
                        &self.smem.barriers().accumulator_slot_produced[accumulator_slot as usize],
                        1,
                    ) as usize as u64);

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
                        while !mbarrier_try_wait_parity(
                            &self.smem.barriers().accumulator_slot_consumed
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
                        for aux_input in aux_input_descs {
                            if !is_first_iter {
                                while !mbarrier_try_wait_parity(
                                    &self.smem.barriers().epi_load_slot_consumed
                                        [epi_slot_idx as usize],
                                    epi_parity,
                                ) {}
                            }

                            let bytes = size_of::<bf16>() as u32 * TM_EPI * TN_EPI;
                            mbarrier_arrive_expect_tx(
                                &self.smem.barriers().epi_load_slot_produced[epi_slot_idx as usize],
                                1,
                                bytes,
                            );
                            tma::cp_async_bulk_tensor_2d_g2s(
                                self.smem.epi_load_staging_slot(epi_slot_idx),
                                aux_input,
                                (self.current_cta_tn * TN_CTA + subtile_n * TN_EPI) as i32,
                                (self.current_cta_tm * TM_CTA) as i32,
                                &mut self.smem.barriers().epi_load_slot_produced
                                    [epi_slot_idx as usize],
                            );

                            epi_slot_idx = (epi_slot_idx + 1) % EPILOGUE_PIPE_DEPTH as u32;
                            if epi_slot_idx == 0 {
                                if !is_first_iter {
                                    epi_parity ^= 1;
                                }
                                is_first_iter = false;
                            }

                            if !self.advance_to_next_tile() {
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    #[inline(always)]
    #[cuda_annotate]
    pub fn run_epi_warps<E: Epilogue<'a>>(&mut self, mut epilogue: E) {
        unsafe {
            let thread_m = (threadIdx_x() - 128) % 128;
            let warp_tmem_lane_base = warp_id() % 4 * 32;

            let mut accum_slot_idx = 0;
            let mut accum_parity = 0;
            let mut epi_slot_idx = 0;
            let mut epi_parity = 0;
            let mut has_stored = false;

            loop {
                // Wait for accumulator to be ready in TMEM
                while !mbarrier_try_wait_parity_cluster(
                    &self.smem.barriers().accumulator_slot_produced[accum_slot_idx as usize],
                    accum_parity,
                ) {}
                tcgen05_fence_after_thread_sync();

                let base_tmem_addr = *self.smem.tmem_addr_dst() & (accum_slot_idx << 8);

                for subtile_n in 0..(TN_CTA / TN_EPI) {
                    let warp_n = (threadIdx_x() - 128) / 128 * 32 + subtile_n * TN_EPI;
                    let tmem_addr = base_tmem_addr
                        + (warp_tmem_lane_base << 16)
                        + (subtile_n * TN_EPI)
                        + warp_n;
                    let mma_result = tcgen05_ld_32x32b_x32_raw(tmem_addr);
                    tcgen05_load_wait();
                    let mma_result = mem::transmute::<
                        [u32; EPI_ELEMENTS_PER_THREAD],
                        [f32; EPI_ELEMENTS_PER_THREAD],
                    >(mma_result.to_array());

                    let mut write_output =
                        |thr_row: &[f32; EPI_ELEMENTS_PER_THREAD], tma_desc: &TmaDescriptor| {
                            // Convert to bf16.
                            let mut thr_row_bf16 = [bf16x2::ZERO; EPI_ELEMENTS_PER_THREAD / 2];
                            #[llvm_unroll]
                            for i in 0..EPI_ELEMENTS_PER_THREAD / 2 {
                                thr_row_bf16[i] =
                                    bf16x2::from_f32_pair(thr_row[i * 2], thr_row[i * 2 + 1]);
                            }

                            let smem = self.smem.epi_store_staging_slot().cast::<bf16>();
                            let row = thread_m;
                            let smem_base = smem.add((row * TN_EPI + warp_n) as usize);

                            if has_stored {
                                // Wait for smem to be done.
                                if warp_id() == 4 && lane_id() == 0 {
                                    cp_async_bulk_wait_group_read(0);
                                }
                                let tok = mbarrier_arrive(&self.smem.barriers().epi_helper_barrier);
                                while !mbarrier_try_wait(
                                    &self.smem.barriers().epi_helper_barrier,
                                    tok,
                                ) {}
                            }

                            // Store to SMEM panel with vectorized stores, in TMA 128B swizzled layout
                            // (noting that the swizzle pattern width of 128 B = 64 elements = TN_EPI)
                            #[llvm_unroll]
                            for chunk in 0..EPI_ELEMENTS_PER_THREAD / 8 {
                                let src = thr_row_bf16.as_ptr().cast::<u128>().add(chunk);
                                let col = chunk as u32 + warp_n / 8;
                                let col_swizzle = col ^ (row & 0b111);
                                smem_base
                                    .cast::<u128>()
                                    .add(col_swizzle as usize)
                                    .write(src.read());
                            }

                            fence_proxy_async_shared_cta();
                            let tok = mbarrier_arrive(&self.smem.barriers().epi_helper_barrier);

                            if warp_id() == 4 && lane_id() == 0 {
                                while !mbarrier_try_wait(
                                    &self.smem.barriers().epi_helper_barrier,
                                    tok,
                                ) {}
                                tma::cp_async_bulk_tensor_2d_s2g(
                                    smem.cast(),
                                    tma_desc,
                                    (self.current_cta_tn * TN_CTA + subtile_n * TN_EPI) as i32,
                                    (self.current_cta_tm * TM_CTA) as i32,
                                );
                            }

                            has_stored = true;
                        };

                    epilogue.visit_mma_result(mma_result, &mut write_output);

                    // TODO aux inputs
                }

                tcgen05_fence_before_thread_sync();
                mbarrier_arrive_cluster(map_shared_rank(
                    &self.smem.barriers().accumulator_slot_consumed[accum_slot_idx as usize],
                    0,
                ) as usize as u64);

                if !self.advance_to_next_tile() {
                    break;
                }

                accum_slot_idx ^= 1;
                if accum_slot_idx == 0 {
                    accum_parity ^= 1;
                }
            }

            if has_stored && warp_id() == 0 && lane_id() == 0 {
                tma::cp_async_bulk_wait_group(0);
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

    fn aux_input_descs(&self) -> [&'a TmaDescriptor; Self::NUM_AUX_INPUTS];

    fn visit_mma_result(
        &mut self,
        result: [f32; EPI_ELEMENTS_PER_THREAD],
        write_output: impl FnMut(&[f32; EPI_ELEMENTS_PER_THREAD], &TmaDescriptor),
    );

    fn visit_aux_input(
        &mut self,
        aux_data: [f32; EPI_ELEMENTS_PER_THREAD],
        write_output: impl FnMut(&[f32; EPI_ELEMENTS_PER_THREAD], &TmaDescriptor),
    );
}

struct IdentityEpilogue<'a> {
    out_tma: &'a TmaDescriptor,
}

impl<'a> Epilogue<'a> for IdentityEpilogue<'a> {
    const NUM_AUX_INPUTS: usize = 0;

    #[inline(always)]
    fn aux_input_descs(&self) -> [&'a TmaDescriptor; Self::NUM_AUX_INPUTS] {
        []
    }

    #[inline(always)]
    fn visit_mma_result(
        &mut self,
        result: [f32; EPI_ELEMENTS_PER_THREAD],
        mut write_output: impl FnMut(&[f32; EPI_ELEMENTS_PER_THREAD], &TmaDescriptor),
    ) {
        write_output(&result, self.out_tma);
    }

    #[inline(always)]
    fn visit_aux_input(
        &mut self,
        _aux_data: [f32; EPI_ELEMENTS_PER_THREAD],
        _write_output: impl FnMut(&[f32; EPI_ELEMENTS_PER_THREAD], &TmaDescriptor),
    ) {
        unreachable!()
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
{
    unsafe {
        let mut smem = Smem::<AB_PIPE_DEPTH, EPILOGUE_PIPE_DEPTH> {
            base: DynamicSharedArray::<u8, 1024>::get_raw(),
        };
        assert!(dynamic_smem_size() >= smem.required_space());

        if threadIdx_x() == 0 {
            init_barriers(&mut smem);
        }

        if warp_role() == WarpRole::MmaIssue {
            tcgen05_alloc_cg2(smem.tmem_addr_dst(), 512);
        }
        this_cluster().sync();

        let mut state = State {
            m,
            n,
            k,
            smem,
            raster_along_m,
            cga_panel_length,
            current_work_id: blockIdx_x(),
            current_cta_tm: 0,
            current_cta_tn: 0,
            a_tma,
            b_tma,
        };
        state.set_tile_from_work_id();

        let role = warp_role();
        match role {
            WarpRole::Epilogue => {
                setmaxnreg_inc::<232>();
                state.run_epi_warps(epilogue);
            }
            _ => {
                setmaxnreg_dec::<32>();
                match role {
                    WarpRole::AbTmaLoad => {
                        state.run_ab_load_warp();
                    }
                    WarpRole::MmaIssue => state.run_mma_warp(),
                    WarpRole::EpilogueTmaLoad => {
                        state.run_epi_load_warp(epilogue.aux_input_descs());
                    }
                    WarpRole::Unused => {}
                    WarpRole::Epilogue => unreachable!(),
                }
            }
        }

        this_cluster().sync();
        if warp_role() == WarpRole::MmaIssue {
            tcgen05_dealloc_cg2(*smem.tmem_addr_dst(), 512);
        }
    }
}

#[kernel]
pub fn sky_matmul_sm10x_128x256_2cta_64x4_arow_bcol(
    m: u32,
    n: u32,
    k: u32,
    raster_along_m: bool,
    cga_panel_length: u32,
    #[grid_constant] a_tma: &TmaDescriptor,
    #[grid_constant] b_tma: &TmaDescriptor,
    #[grid_constant] out_tma: &TmaDescriptor,
) {
    kernel_template::<_, 4, 1>(
        m,
        n,
        k,
        raster_along_m,
        cga_panel_length,
        a_tma,
        b_tma,
        IdentityEpilogue { out_tma },
    );
}
