//! Forward RMSNorm for row sizes that fit in shared memory.
//! Input is bf16. Output is either bf16 or MXFP8. In MXFP8 mode,
//! a rowwise block scaled output is always written; an optional additional columnwise
//! block scaled output can also be selected. The latter requires a cluster
//! size such that microblock_size * cluster_size == 32. If cluster_size > 1
//! then a global workspace of (num_clusters * cluster_size * row_size) bf16 elements
//! is required.
//!
//! CLC is used to form a persistent kernel.
//!
//! Each block of rows (corresponding to a thread block in the launch)
//! is split into microblocks which are streamed through the SM.
//!
//! The kernel uses 4 worker warps for doing math on the row elements. They keep `gamma`
//! stationary in registers across microblocks (and blocks if using CLC). The warps stream
//! input rows from SMEM, compute statistics, apply normalization, and then execute the configured
//! epilogue. Epilogues stage stores through SMEM to allow TMA.
//!
//! The remaining 4 warps (warps 4-7) are assigned as follows:
//! * warp 4: TMA loads
//! * warp 5: epilogue TMA stores
//! * warp 6: CLC scheduler loop
//! * remainder: no role
//!
//! The kernel is parametrized over a maximum row size. It can handle smaller
//! row sizes with correct zero-padding and mean divisor. The effective row size
//! must be a multiple of 16, else the behavior is undefined.

use crate::{
    utils::{
        clc::{ClcScheduler, ClcSmem},
        copy::copy_aligned16,
    },
    warp_reduce_sum_f32,
};
use core::{cell::UnsafeCell, ptr, simd::f32x8};
use ptx_std::{
    Denormals, F32Ext, MBarrier, PhaseParity, WARP_SIZE,
    barrier::{BarrierAlignment, NamedBarrier, sync_threads},
    bf16, bf16x8, cluster, cluster_size, dynamic_smem_start, f8e4m3, f8e8m0,
    fence::{ProxyKind, fence_proxy},
    launch::{decrease_max_registers, increase_max_registers},
    memory::{CachePolicy, L2EvictionPriority},
    thread_id,
    tma::{self, BulkWait},
    warp_collectives::{LaneMask, elect, sync_warp},
    warp_id,
};

#[derive(Copy, Clone, PartialEq, Eq)]
enum WarpRole {
    Worker,
    TmaLoad,
    TmaStore,
    Clc,
    None,
}

impl WarpRole {
    #[inline(always)]
    pub fn get() -> Self {
        match warp_id() {
            0..4 => WarpRole::Worker,
            4 => WarpRole::TmaLoad,
            5 => WarpRole::TmaStore,
            6 => WarpRole::Clc,
            _ => WarpRole::None,
        }
    }
}

const NUM_WORKER_WARPS: usize = 4;
const NUM_WORKER_THREADS: usize = NUM_WORKER_WARPS * WARP_SIZE as usize;

#[repr(C, align(1024))]
struct SmemData<const LOAD_SLOTS: usize, const ROW_SIZE: usize, const MICROBLOCK_SIZE: usize> {
    load_slots: [[UnsafeCell<bf16x8>; Self::BF16X8_PER_SLOT]; LOAD_SLOTS],
    store_slot: [UnsafeCell<bf16x8>; Self::BF16X8_PER_SLOT],
    reduction_scratchpad: [[UnsafeCell<f32>; MICROBLOCK_SIZE]; NUM_WORKER_WARPS],
    load_slot_produced: [MBarrier; LOAD_SLOTS],
    load_slot_consumed: [MBarrier; LOAD_SLOTS],
    store_slot_produced: MBarrier,
    store_slot_consumed: MBarrier,
    cluster_col_amax_produced: MBarrier,
    cluster_col_amax_consumed: MBarrier,
    gamma_loaded: MBarrier,
    clc: UnsafeCell<ClcSmem>,
}

impl<const LOAD_SLOTS: usize, const ROW_SIZE: usize, const MICROBLOCK_SIZE: usize>
    SmemData<LOAD_SLOTS, ROW_SIZE, MICROBLOCK_SIZE>
{
    const BF16X8_PER_SLOT: usize = MICROBLOCK_SIZE * ROW_SIZE / 8;
}

#[derive(Debug)]
enum Epilogue {
    StoreBf16 {
        y: *mut bf16,
    },
    StoreRowwiseMxfp8 {
        rowwise_y: *mut f8e4m3,
        rowwise_y_sf: *mut f8e8m0,
    },
    StoreRowwiseAndColwiseMxfp8 {
        rowwise_y: *mut f8e4m3,
        rowwise_y_sf: *mut f8e8m0,
        colwise_y: *mut f8e4m3,
        colwise_y_sf: *mut f8e8m0,
    },
}

struct RmsnormFwd<
    'a,
    const ROW_SIZE: usize,
    const MICROBLOCK_SIZE: usize,
    const BLOCK_SIZE: usize,
    const LOAD_SLOTS: usize,
    const WORKER_WARP_REGS: u32,
    const AUX_WARP_REGS: u32,
> {
    smem: &'a SmemData<LOAD_SLOTS, ROW_SIZE, MICROBLOCK_SIZE>,
    cluster_reduction_scratch: *mut bf16,
    gamma: *const bf16,
    r_inv: *mut f32,
    x: *const bf16,
    epilogue: Epilogue,
    actual_row_size: u32,
    actual_row_size_recip: f32,
    num_rows: u32,

    current_block: u32,
    current_microblock: u32,

    store_idx: u32,

    clc: ClcScheduler,
}

impl<
    'a,
    const ROW_SIZE: usize,
    const MICROBLOCK_SIZE: usize,
    const BLOCK_SIZE: usize,
    const LOAD_SLOTS: usize,
    const WORKER_WARP_REGS: u32,
    const AUX_WARP_REGS: u32,
>
    RmsnormFwd<
        'a,
        ROW_SIZE,
        MICROBLOCK_SIZE,
        BLOCK_SIZE,
        LOAD_SLOTS,
        WORKER_WARP_REGS,
        AUX_WARP_REGS,
    >
{
    const ELEMENTS_PER_ROW_PER_THREAD: usize = ROW_SIZE / NUM_WORKER_WARPS / WARP_SIZE as usize;
    const ELEMENTX8_PER_ROW_PER_THREAD: usize = Self::ELEMENTS_PER_ROW_PER_THREAD / 8;

    #[inline(always)]
    pub fn run(
        gamma: *const bf16,
        x: *const bf16,
        epilogue: Epilogue,
        r_inv: *mut f32,
        actual_row_size: u32,
        actual_row_size_recip: f32,
        num_rows: u32,
        cluster_reduction_scratch: *mut bf16,
    ) {
        unsafe {
            let smem = &mut *dynamic_smem_start()
                .cast::<SmemData<LOAD_SLOTS, ROW_SIZE, MICROBLOCK_SIZE>>();
            let mut this = Self {
                clc: ClcScheduler::new(smem.clc.get()),
                smem,
                cluster_reduction_scratch,
                gamma,
                x,
                epilogue,
                actual_row_size,
                actual_row_size_recip,
                current_block: 0,
                // cause first call to next_microblock() to advance to the first block
                current_microblock: (BLOCK_SIZE / MICROBLOCK_SIZE) as u32 - 1,
                store_idx: 0,
                num_rows,
                r_inv,
            };

            if warp_id() == 0 && elect(LaneMask::FULL).is_leader {
                this.init_barriers();

                // zero out the SMEM
                ptx_std::memory::zero_fill_shared(
                    this.smem.load_slots.as_ptr() as *mut u8,
                    size_of_val(&this.smem.load_slots),
                );
                fence_proxy::<{ ProxyKind::AsyncSharedCta }>();

                // load `gamma` into SMEM via TMA
                let bar = &this.smem.gamma_loaded;

                let bytes = size_of::<bf16>() as u32 * this.actual_row_size;
                bar.arrive_expect_tx(bytes);
                tma::bulk_load(
                    this.smem.load_slots[0].as_ptr() as *mut u8,
                    this.gamma.cast(),
                    bytes,
                    bar,
                    None,
                );
            }

            cluster::barrier_sync::<{ BarrierAlignment::Unaligned }>();

            let mut thr_gamma = [bf16x8::ZERO; Self::ELEMENTX8_PER_ROW_PER_THREAD];

            if WarpRole::get() == WarpRole::Worker {
                while !this.smem.gamma_loaded.try_wait(PhaseParity::Even) {}
                copy_aligned16(
                    this.smem.load_slots[0]
                        .as_ptr()
                        .add(thread_id().x as usize)
                        .cast(),
                    thr_gamma.as_mut_ptr(),
                    Self::ELEMENTX8_PER_ROW_PER_THREAD,
                    NUM_WORKER_THREADS,
                    1,
                );
            }

            sync_warp(LaneMask::FULL);
            sync_threads();

            let role = WarpRole::get();
            match role {
                WarpRole::Worker => {
                    increase_max_registers::<WORKER_WARP_REGS>();
                    this.run_worker_warps(&thr_gamma);
                }
                _ => {
                    decrease_max_registers::<AUX_WARP_REGS>();
                    match role {
                        WarpRole::TmaLoad => this.run_tma_load_warp(),
                        WarpRole::TmaStore => this.run_tma_store_warp(),
                        WarpRole::Clc => this.clc.run_scheduler(),
                        WarpRole::None => loop {
                            if !this.next_microblock() {
                                break;
                            }
                        },
                        WarpRole::Worker => unreachable!(),
                    }
                }
            }

            cluster::barrier_sync::<{ BarrierAlignment::Unaligned }>();
        }
    }

    #[inline(always)]
    fn current_block_first_row(&self) -> u32 {
        self.current_block * BLOCK_SIZE as u32
    }

    #[inline(always)]
    fn current_microblock_first_row(&self) -> u32 {
        self.current_block_first_row() + self.current_microblock * MICROBLOCK_SIZE as u32
    }

    #[inline(always)]
    pub fn init_barriers(&self) {
        unsafe {
            #[rustc_unroll(full)]
            for slot_idx in 0..LOAD_SLOTS {
                self.smem.load_slot_produced[slot_idx].init(1);
                self.smem.load_slot_consumed[slot_idx].init(NUM_WORKER_THREADS as u32);
            }
            self.smem
                .store_slot_produced
                .init(NUM_WORKER_THREADS as u32);
            self.smem.store_slot_consumed.init(1);
            self.smem
                .cluster_col_amax_produced
                .init(cluster_size() * NUM_WORKER_THREADS as u32);
            self.smem
                .cluster_col_amax_consumed
                .init(cluster_size() * NUM_WORKER_THREADS as u32);
            self.smem.gamma_loaded.init(1);

            MBarrier::fence_init();
        }
    }

    #[inline(always)]
    fn next_block(&mut self) -> bool {
        // Go to next block via CLC, else done
        unsafe {
            match self.clc.get_next_block() {
                Some(block) => {
                    self.current_block = block.x;
                    self.current_microblock = 0;
                    true
                }
                None => false,
            }
        }
    }

    #[inline(always)]
    fn next_microblock(&mut self) -> bool {
        const {
            assert!(BLOCK_SIZE.is_multiple_of(MICROBLOCK_SIZE));
        }
        let num_microblocks = BLOCK_SIZE as u32 / MICROBLOCK_SIZE as u32;
        if self.current_microblock + 1 == num_microblocks {
            self.next_block()
        } else {
            self.current_microblock += 1;
            true
        }
    }

    #[inline(always)]
    pub fn run_worker_warps(&mut self, thr_gamma: &[bf16x8; Self::ELEMENTX8_PER_ROW_PER_THREAD]) {
        unsafe {
            let mut microblock_idx = 0u32;

            loop {
                if !self.next_microblock() {
                    break;
                }

                let slot_idx = microblock_idx as usize % LOAD_SLOTS;
                let ready = &self.smem.load_slot_produced[slot_idx];
                while !ready.try_wait(PhaseParity::of_phase(microblock_idx / LOAD_SLOTS as u32)) {}

                // Load `x`
                let mut thr_x =
                    [[bf16x8::ZERO; Self::ELEMENTX8_PER_ROW_PER_THREAD]; MICROBLOCK_SIZE];
                #[rustc_unroll(full)]
                for row in 0..MICROBLOCK_SIZE {
                    let smem_x = self.smem.load_slots[slot_idx]
                        .as_ptr()
                        .add(row * ROW_SIZE / 8 + thread_id().x as usize);
                    copy_aligned16(
                        smem_x.cast(),
                        thr_x[row].as_mut_ptr(),
                        Self::ELEMENTX8_PER_ROW_PER_THREAD,
                        NUM_WORKER_THREADS,
                        1,
                    );
                }

                // Load has finished
                self.smem.load_slot_consumed[slot_idx].arrive();

                // Compute sum of squares per row per warp
                let mut sum2 = [0.0f32; MICROBLOCK_SIZE];
                #[rustc_unroll(full)]
                for row in 0..MICROBLOCK_SIZE {
                    #[rustc_unroll(full)]
                    for i in 0..Self::ELEMENTX8_PER_ROW_PER_THREAD {
                        let xs = thr_x[row][i].to_f32();
                        sum2[row] += (xs * xs).reduce_sum();
                    }
                    sum2[row] = warp_reduce_sum_f32(sum2[row]);
                }

                NamedBarrier::new(0x02)
                    .sync_among::<{ BarrierAlignment::Unaligned }>(NUM_WORKER_THREADS as u32);

                // Publish our warp's sums to SMEM
                if elect(LaneMask::FULL).is_leader {
                    #[rustc_unroll(full)]
                    for row in 0..MICROBLOCK_SIZE {
                        self.smem.reduction_scratchpad[warp_id() as usize]
                            .as_ptr()
                            .add(row)
                            .cast_mut()
                            .write(UnsafeCell::new(sum2[row]));
                    }
                }

                NamedBarrier::new(0x03)
                    .sync_among::<{ BarrierAlignment::Unaligned }>(NUM_WORKER_THREADS as u32);

                // Each warp aggregates the sums across lanes
                #[rustc_unroll(full)]
                for row in 0..MICROBLOCK_SIZE {
                    sum2[row] = 0.0;
                    #[rustc_unroll(full)]
                    for w in 0..NUM_WORKER_WARPS {
                        sum2[row] += self.smem.reduction_scratchpad[w]
                            .as_ptr()
                            .add(row)
                            .read()
                            .into_inner();
                    }
                }

                let mut r_inv = [0.0f32; MICROBLOCK_SIZE];

                // Apply normalization
                #[rustc_unroll(full)]
                for row in 0..MICROBLOCK_SIZE {
                    let ms = self.actual_row_size_recip * sum2[row] + 1e-6;
                    let rms = ms.rsqrt_approx::<{ Denormals::FlushToZero }>();
                    #[rustc_unroll(full)]
                    for i in 0..Self::ELEMENTX8_PER_ROW_PER_THREAD {
                        let xs = thr_x[row][i].to_f32();
                        let gammas = thr_gamma[i].to_f32();
                        thr_x[row][i] = bf16x8::from_f32(xs * gammas * f32x8::splat(rms));
                    }
                    r_inv[row] = rms;
                }

                // Store r_inv
                if warp_id() == 0 && elect(LaneMask::FULL).is_leader {
                    #[rustc_unroll(full)]
                    for i in 0..MICROBLOCK_SIZE {
                        let row = i as u32 + self.current_microblock_first_row();
                        if row < self.num_rows {
                            self.r_inv.add(row as usize).write(r_inv[i]);
                        }
                    }
                }

                self.do_epilogue(&thr_x);

                microblock_idx += 1;
            }
        }
    }

    #[inline(always)]
    fn do_epilogue(
        &mut self,
        thr_y: &[[bf16x8; Self::ELEMENTX8_PER_ROW_PER_THREAD]; MICROBLOCK_SIZE],
    ) {
        match self.epilogue {
            Epilogue::StoreBf16 { .. } => {
                self.do_epilogue_store_bf16(thr_y);
            }
            Epilogue::StoreRowwiseMxfp8 { .. } => {
                self.do_epilogue_store_rowwise_mxfp8(thr_y);
            }
            Epilogue::StoreRowwiseAndColwiseMxfp8 { .. } => {
                self.do_epilogue_store_rowwise_mxfp8(thr_y);
                self.do_epilogue_store_colwise_mxfp8(thr_y);
            }
        }
    }

    #[inline(always)]
    fn do_epilogue_store_bf16(
        &mut self,
        thr_y: &[[bf16x8; Self::ELEMENTX8_PER_ROW_PER_THREAD]; MICROBLOCK_SIZE],
    ) {
        unsafe {
            self.wait_for_store_slot();

            #[rustc_unroll(full)]
            for row in 0..MICROBLOCK_SIZE {
                let smem_y = self
                    .smem
                    .store_slot
                    .as_ptr()
                    .add(row * ROW_SIZE / 8 + thread_id().x as usize);
                copy_aligned16(
                    thr_y[row].as_ptr(),
                    smem_y as *mut bf16x8,
                    Self::ELEMENTX8_PER_ROW_PER_THREAD,
                    1,
                    NUM_WORKER_THREADS,
                );
            }

            fence_proxy::<{ ProxyKind::AsyncSharedCta }>();
            self.arrive_on_store_slot();
        }
    }

    #[inline(always)]
    fn do_epilogue_store_rowwise_mxfp8(
        &mut self,
        thr_y: &[[bf16x8; Self::ELEMENTX8_PER_ROW_PER_THREAD]; MICROBLOCK_SIZE],
    ) {
        todo!()
    }

    #[inline(always)]
    fn do_epilogue_store_colwise_mxfp8(
        &mut self,
        thr_y: &[[bf16x8; Self::ELEMENTX8_PER_ROW_PER_THREAD]; MICROBLOCK_SIZE],
    ) {
        todo!()
    }

    #[inline(always)]
    fn wait_for_store_slot(&mut self) {
        if self.store_idx > 0 {
            unsafe {
                let bar = &self.smem.store_slot_consumed;
                while !bar.try_wait(PhaseParity::of_phase(self.store_idx).flip()) {}
            }
        }
        self.store_idx += 1;
    }

    #[inline(always)]
    fn arrive_on_store_slot(&mut self) {
        unsafe {
            self.smem.store_slot_produced.arrive();
        }
    }

    #[inline(always)]
    fn run_tma_load_warp(&mut self) {
        let mut microblock_idx = 0;
        unsafe {
            loop {
                if !self.next_microblock() {
                    break;
                }

                let slot_idx = microblock_idx as usize % LOAD_SLOTS;

                if microblock_idx >= LOAD_SLOTS as u32 {
                    let bar = &self.smem.load_slot_consumed[slot_idx];
                    while !bar
                        .try_wait(PhaseParity::of_phase(microblock_idx / LOAD_SLOTS as u32).flip())
                    {
                    }
                }

                let bar = &self.smem.load_slot_produced[slot_idx];
                if elect(LaneMask::FULL).is_leader {
                    let actual_rows = if self.current_microblock_first_row()
                        + MICROBLOCK_SIZE as u32
                        > self.num_rows
                    {
                        self.num_rows
                            .saturating_sub(self.current_microblock_first_row())
                    } else {
                        MICROBLOCK_SIZE as u32
                    };
                    let bytes = actual_rows * self.actual_row_size * size_of::<bf16>() as u32;

                    bar.arrive_expect_tx(bytes);

                    #[rustc_unroll(full)]
                    for i in 0..MICROBLOCK_SIZE {
                        let row = i as u32 + self.current_microblock_first_row();
                        if row < self.num_rows {
                            tma::bulk_load(
                                self.smem.load_slots[slot_idx]
                                    .as_ptr()
                                    .add(ROW_SIZE / 8 * i)
                                    as *mut u8,
                                self.x
                                    .add(row as usize * self.actual_row_size as usize)
                                    .cast(),
                                size_of::<bf16>() as u32 * self.actual_row_size,
                                bar,
                                Some(CachePolicy::new::<{ L2EvictionPriority::First }>()),
                            );
                        }
                    }
                }

                microblock_idx += 1;
            }
        }
    }

    #[inline(always)]
    fn run_tma_store_warp(&mut self) {
        unsafe {
            loop {
                if !self.next_microblock() {
                    break;
                }

                match self.epilogue {
                    Epilogue::StoreBf16 { y } => {
                        self.wait_for_store_data_ready();
                        if elect(LaneMask::FULL).is_leader {
                            self.tma_store_bf16(y);
                            self.wait_for_and_notify_store_read_smem();
                        }
                    }
                    Epilogue::StoreRowwiseMxfp8 {
                        rowwise_y,
                        rowwise_y_sf,
                    } => {
                        self.wait_for_store_data_ready();
                        if elect(LaneMask::FULL).is_leader {
                            self.tma_store_mxfp8(rowwise_y, rowwise_y_sf);
                            self.wait_for_and_notify_store_read_smem();
                        }
                    }
                    Epilogue::StoreRowwiseAndColwiseMxfp8 {
                        rowwise_y,
                        rowwise_y_sf,
                        colwise_y,
                        colwise_y_sf,
                    } => {
                        self.wait_for_store_data_ready();
                        if elect(LaneMask::FULL).is_leader {
                            self.tma_store_mxfp8(rowwise_y, rowwise_y_sf);
                            self.wait_for_and_notify_store_read_smem();
                        }
                        self.wait_for_store_data_ready();
                        if elect(LaneMask::FULL).is_leader {
                            self.tma_store_mxfp8(colwise_y, colwise_y_sf);
                            self.wait_for_and_notify_store_read_smem();
                        }
                    }
                }
            }
        }
    }

    #[inline(always)]
    fn wait_for_store_data_ready(&mut self) {
        unsafe {
            while !self
                .smem
                .store_slot_produced
                .try_wait(PhaseParity::of_phase(self.store_idx))
            {}
            self.store_idx += 1;
        }
    }

    #[inline(always)]
    fn wait_for_and_notify_store_read_smem(&mut self) {
        unsafe {
            tma::bulk_commit_group();
            tma::bulk_wait_group::<0, { BulkWait::SourceRead }>();
            self.smem.store_slot_consumed.arrive();
        }
    }

    #[inline(always)]
    fn tma_store_bf16(&self, y: *mut bf16) {
        unsafe {
            for i in 0..MICROBLOCK_SIZE {
                let row = i as u32 + self.current_microblock_first_row();
                if row < self.num_rows {
                    tma::bulk_store(
                        y.add(self.actual_row_size as usize * row as usize).cast(),
                        self.smem.store_slot.as_ptr().add(ROW_SIZE / 8 * i).cast(),
                        size_of::<bf16>() as u32 * self.actual_row_size,
                        None,
                    );
                }
            }
        }
    }

    #[inline(always)]
    fn tma_store_mxfp8(&self, y: *mut f8e4m3, y_sf: *mut f8e8m0) -> u32 {
        todo!()
    }
}

#[unsafe(no_mangle)]
#[nvptx(exact_threads_per_cta(256, 1, 1), max_registers(128))]
pub unsafe extern "ptx-kernel" fn rmsnorm_fwd_bf16_in_bf16_out_row_size_2048(
    gamma: *const bf16,
    x: *const bf16,
    y: *mut bf16,
    r_inv: *mut f32,
    actual_row_size: u32,
    actual_row_size_recip: f32,
    num_rows: u32,
) {
    RmsnormFwd::<2048, 8, 64, 2, 224, 32>::run(
        gamma,
        x,
        Epilogue::StoreBf16 { y },
        r_inv,
        actual_row_size,
        actual_row_size_recip,
        num_rows,
        ptr::null_mut(),
    );
}
