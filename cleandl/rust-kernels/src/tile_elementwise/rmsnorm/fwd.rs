use crate::tile_elementwise::template::{
    self, KernelVisitor, LoadDescriptor, LoadTy, Mode1xN, ModeMxN, Params, StoreDescriptor,
    StoreTy, TileElementwiseKernel,
};
use ptx_std::{
    Denormals, Dim3, F32Ext, f32xN,
    memory::{CachePolicy, L2EvictionPriority},
    tma::TmaDescriptor,
};

#[unsafe(no_mangle)]
#[nvptx(max_registers(128), exact_threads_per_cta(256, 1, 1))]
pub unsafe extern "ptx-kernel" fn rmsnorm_fwd_Xbf16_Ybf16_RS2048_BS128(
    #[nvptx(grid_constant)] x: &TmaDescriptor,
    #[nvptx(grid_constant)] gamma: &TmaDescriptor,
    #[nvptx(grid_constant)] y: &TmaDescriptor,
    actual_row_size: u32,
    rcp_actual_row_size: f32,
) {
    assert!(actual_row_size <= 2048);

    struct P;

    impl Params for P {
        const TM: usize = 8;
        const TN: usize = 2048;

        const CLUSTER_SIZE: usize = 1;
        const MACROBLOCK_SIZE: usize = 16;
        const NUM_LOAD_SLOTS: usize = 2;
        const LARGEST_LOAD_BITS_PER_ELEMENT: usize = 16;
        const LARGEST_STORE_BITS_PER_ELEMENT: usize = 16;
        const DOES_STORE_ROWWISE_BLOCK_SCALES: bool = false;
        const DOES_STORE_COLWISE_BLOCK_SCALES: bool = false;
    }

    struct K<'a> {
        x: &'a TmaDescriptor,
        gamma: &'a TmaDescriptor,
        y: &'a TmaDescriptor,
        actual_row_size: u32,
        rcp_actual_row_size: f32,
    }

    unsafe impl<'a> TileElementwiseKernel for K<'a> {
        type Params = P;
        type WorkerState = Worker;

        #[inline(always)]
        fn map_coord(&self, macroblock: u32, idx_in_macroblock: u32) -> Dim3 {
            let row = macroblock * Self::Params::MACROBLOCK_SIZE as u32 * Self::Params::TM as u32
                + idx_in_macroblock * Self::Params::TM as u32;
            Dim3 { x: row, y: 0, z: 0 }
        }

        #[inline(always)]
        fn init_worker_state(&self) -> Self::WorkerState {
            Worker {
                gamma: f32xN::ZERO,
                x: [f32xN::ZERO; _],
            }
        }

        #[inline(always)]
        fn prologue(&mut self, mut v: impl KernelVisitor<Self, Self::Params>) {
            v.load(
                LoadDescriptor {
                    mode: Mode1xN,
                    ty: LoadTy::Bf16,
                    tma_desc: self.gamma,
                    coord_mn: [0, 0],
                    cache_policy: None,
                },
                #[inline(always)]
                |state, gamma| {
                    state.gamma = bytemuck::must_cast(gamma);
                },
            );
        }

        #[inline(always)]
        fn process_tile(&mut self, coord: Dim3, mut v: impl KernelVisitor<Self, Self::Params>) {
            v.load(
                LoadDescriptor {
                    mode: ModeMxN,
                    ty: LoadTy::Bf16,
                    tma_desc: self.x,
                    coord_mn: [coord.x, 0],
                    cache_policy: Some(CachePolicy::new::<{ L2EvictionPriority::First }>()),
                },
                #[inline(always)]
                |state, x| {
                    state.x = bytemuck::must_cast(x);
                },
            );

            v.rowwise_reduce(
                #[inline(always)]
                |a, b| a + b,
                #[inline(always)]
                |state| {
                    let mut sum_of_squares = [0.0; <P as Params>::THREAD_M];
                    #[rustc_unroll(full)]
                    for i in 0..<P as Params>::THREAD_M {
                        sum_of_squares[i] = (state.x[i] * state.x[i]).reduce_sum();
                    }
                    sum_of_squares
                },
                #[inline(always)]
                |state, sum_of_squares| {
                    let sum_of_squares: f32xN<{ <P as Params>::THREAD_M }> =
                        bytemuck::must_cast(sum_of_squares);
                    let r_inv = (sum_of_squares * self.rcp_actual_row_size + 1e-6)
                        .rsqrt_approx::<{ Denormals::FlushToZero }>()
                        .to_array();
                    #[rustc_unroll(full)]
                    for i in 0..<P as Params>::THREAD_M {
                        state.x[i] = state.x[i] * state.gamma * r_inv[i];
                    }
                },
            );

            v.store(
                StoreDescriptor {
                    mode: ModeMxN,
                    ty: StoreTy::Bf16,
                    tma_desc: self.y,
                    coord_mn: [coord.x, 0],
                    reduction: None,
                    cache_policy: None,
                },
                #[inline(always)]
                |state| bytemuck::must_cast(state.x),
            );
        }

        #[inline(always)]
        fn epilogue(&mut self, _v: impl KernelVisitor<Self, Self::Params>) {}
    }

    struct Worker {
        gamma: f32xN<{ <P as Params>::THREAD_N }>,
        x: [f32xN<{ <P as Params>::THREAD_N }>; <P as Params>::THREAD_M],
    }

    unsafe {
        template::run(K {
            x,
            gamma,
            y,
            actual_row_size,
            rcp_actual_row_size,
        })
    };
}
