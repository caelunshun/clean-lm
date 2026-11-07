use crate::{
    kernels::matmul::{
        config::Config,
        epilogue::{LoadTensorId, StoreTensorId},
    },
    utils::clc::ClcScheduler,
};
use kernel_build_util::nvvm::Args;
use nvvm_builder::{
    F4E2M1, F8E4M3, Global, Mbarrier, Ptr, Shared, TensorMap, UE8M0, Val, bf16,
    mbarrier::{MbarrierArray, Pipeline},
};
use skytens_core::FastHashMap;

const CTA_TM: usize = 128;
const CTA_TN: usize = 256;

/// In 2-cta mode, only half of the N dimension resides in each CTA's
/// SMEM.
const CTA_TN_FOR_SMEM_INPUTS: usize = 128;

const CTA_PAIR_TM: usize = 256;

struct BuilderState {
    config: Config,

    epilogue_layout: EpilogueLayout,

    a_tma: OperandTmaDesc,
    b_tma: OperandTmaDesc,
    epi_outputs_tma: FastHashMap<StoreTensorId, OperandTmaDesc>,
    epi_inputs_tma: FastHashMap<LoadTensorId, OperandTmaDesc>,

    m: Val<u32>,
    n: Val<u32>,
    k: Val<u32>,

    clc: ClcScheduler,

    tmem_base_addr: Val<u32>,

    log2_cluster_size_m: Val<u32>,
    log2_cluster_size_n: Val<u32>,

    mma_load_a_slots: Val<Ptr<u32, Shared>>,
    mma_load_b_slots: Val<Ptr<u32, Shared>>,
    mma_load_sf_a_slots: Option<Val<Ptr<u8, Shared>>>,
    mma_load_sf_b_slots: Option<Val<Ptr<u8, Shared>>>,

    epi_load_slots: Val<Ptr<bf16, Shared>>,

    epi_store_slot: Val<Ptr<u8, Shared>>,
    epi_sf_store_slot: Option<Val<Ptr<u8, Shared>>>,

    /// Only used for the leader CTA of each CTA-pair.
    mma_load_produced: MbarrierArray,
    /// Every CTA has an instance of this barrier. All CTAs
    /// that participate in a multicast load must receive
    /// arrivals once that load is consumed.
    mma_load_consumed: MbarrierArray,

    epi_load_pipeline: Pipeline,
    /// Pipeline depth is always 1.
    epi_store_pipeline: Pipeline,
}

enum OperandTmaDesc {
    Bf16(TensorMap<bf16>),
    Mxfp8 {
        elements: TensorMap<F8E4M3>,
        sf: Val<Ptr<UE8M0, Global>>,
    },
    F4E2M1 {
        elements: TensorMap<F4E2M1>,
        sf: Val<Ptr<F8E4M3, Global>>,
    },
}

/// Selected thread->value layout for the epilogue subtiles.
/// Epilogue subtile size is always 128x64 per CTA, with the first
/// 128x32 half handled by the first epilogue warpgroup and the second
/// 128x32 half handled by the second epilogue warpgroup.
/// This layout determines how, within a warp's 32x32 tile,
/// the elements are mapped to lanes.
enum EpilogueLayout {
    /// Each thread holds a contiguous row of elements. Follows from 32x32b TMEM load pattern.
    /// This layout does not support columnwise block-scaled stores.
    Ldtm32x32b,
    /// Follows from 16x256b TMEM load pattern, which also happens to match
    /// the 8-bit stmatrix.trans layout. Threads hold blocks of 2 elements
    /// within the same row; these blocks are assigned to threads row-major to fill 8x8 tiles,
    /// and these tiles then repeat in row-major order to fill the 32x32 warp tile.
    ///
    /// This layout must be selected if there is any columnwise block-scaled
    /// store. However, it will have reduced performance for rowwise block-scaled
    /// stores, due to use of narrow SMEM store instructions.
    Ldtm16x256b,
}

impl BuilderState {
    fn ab_load_stages(&self) -> usize {
        self.mma_load_produced.len()
    }

    fn cluster_size_m(&self) -> Val<u32> {
        1u32 << self.log2_cluster_size_m
    }

    fn cluster_size_n(&self) -> Val<u32> {
        1u32 << self.log2_cluster_size_n
    }

    fn cga_tm(&self) -> Val<u32> {
        CTA_TM as u32 * self.cluster_size_m()
    }

    fn cga_tn(&self) -> Val<u32> {
        CTA_TN as u32 * self.cluster_size_n()
    }

    fn tk(&self) -> Val<u32> {

    }

    fn mma_tk (&self) -> Val<u32> {
        
    }

    fn num_cgas_m(&self) -> Val<u32> {
        self.m.div_ceil(self.cga_tm())
    }

    fn num_cgas_n(&self) -> Val<u32> {
        self.n.div_ceil(self.cga_tn())
    }
}
