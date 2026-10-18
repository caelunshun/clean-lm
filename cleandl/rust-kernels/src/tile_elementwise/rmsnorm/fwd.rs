use crate::{
    tile_elementwise::template::{
        self, BlockScaleDim, BlockScaledStoreDescriptor, KernelVisitor, LoadDescriptor, LoadTy,
        Mode1xN, ModeMx1, ModeMxN, Params, StoreDescriptor, StoreTy, TileElementwiseKernel,
    },
    utils::{block_scale::Mxfp8, tma::CompoundTmaDescriptor},
};
use core::marker::PhantomData;
use ptx_std::{
    Denormals, Dim3, F32Ext, f32xN,
    memory::{CachePolicy, L2EvictionPriority},
};

struct Impl<'a, P> {
    x: &'a CompoundTmaDescriptor,
    gamma: &'a CompoundTmaDescriptor,
    r_inv: &'a CompoundTmaDescriptor,
    rcp_actual_row_size: f32,
    store_mode: StoreMode<'a>,
    _marker: PhantomData<P>,
}

enum StoreMode<'a> {
    Bf16(&'a CompoundTmaDescriptor),
    Mxfp8Row(&'a CompoundTmaDescriptor, *mut u8),
    Mxfp8RowAndCol(
        &'a CompoundTmaDescriptor,
        *mut u8,
        &'a CompoundTmaDescriptor,
        *mut u8,
    ),
}

unsafe impl<'a, P> TileElementwiseKernel for Impl<'a, P>
where
    P: Params + 'static,
{
    type Params = P;
    type WorkerState = Worker<P>;

    #[inline(always)]
    fn map_coord(&self, macroblock: u32, idx_in_macroblock: u32) -> Dim3 {
        let row = macroblock * Self::Params::MACROBLOCK_SIZE as u32 * Self::Params::CGA_TM as u32
            + idx_in_macroblock * Self::Params::CGA_TM as u32;
        Dim3 { x: row, y: 0, z: 0 }
    }

    #[inline(always)]
    fn init_worker_state(&self) -> Self::WorkerState {
        Worker {
            gamma: f32xN::ZERO,
            x: [f32xN::ZERO; _],
            r_inv: [0.0; _],
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
                state.r_inv = r_inv;
            },
        );

        match self.store_mode {
            StoreMode::Bf16(tma_desc) => {
                v.store(
                    StoreDescriptor {
                        mode: ModeMxN,
                        ty: StoreTy::Bf16,
                        tma_desc,
                        coord_mn: [coord.x, 0],
                        reduction: None,
                        cache_policy: None,
                    },
                    #[inline(always)]
                    |state| bytemuck::must_cast(state.x),
                );
            }
            StoreMode::Mxfp8Row(tma_desc, sf_ptr) => {
                v.store_block_scaled::<Mxfp8>(
                    BlockScaledStoreDescriptor {
                        sf_dim: BlockScaleDim::Rowwise,
                        tma_desc,
                        sf_ptr,
                        coord_mn: [coord.x, 0],
                        cache_policy: None,
                    },
                    #[inline(always)]
                    |state| bytemuck::must_cast(state.x),
                );
            }
            StoreMode::Mxfp8RowAndCol(row_tma_desc, row_sf_ptr, col_tma_desc, col_sf_ptr) => {
                v.store_block_scaled::<Mxfp8>(
                    BlockScaledStoreDescriptor {
                        sf_dim: BlockScaleDim::Rowwise,
                        tma_desc: row_tma_desc,
                        sf_ptr: row_sf_ptr,
                        coord_mn: [coord.x, 0],
                        cache_policy: None,
                    },
                    #[inline(always)]
                    |state| bytemuck::must_cast(state.x),
                );
                v.store_block_scaled::<Mxfp8>(
                    BlockScaledStoreDescriptor {
                        sf_dim: BlockScaleDim::Colwise,
                        tma_desc: col_tma_desc,
                        sf_ptr: col_sf_ptr,
                        coord_mn: [coord.x, 0],
                        cache_policy: None,
                    },
                    #[inline(always)]
                    |state| bytemuck::must_cast(state.x),
                );
            }
        }

        v.store(
            StoreDescriptor {
                mode: ModeMx1,
                ty: StoreTy::F32,
                tma_desc: self.r_inv,
                coord_mn: [coord.x, 0],
                reduction: None,
                cache_policy: None,
            },
            #[inline(always)]
            |state| state.r_inv,
        );
    }

    #[inline(always)]
    fn epilogue(&mut self, _v: impl KernelVisitor<Self, Self::Params>) {}
}

struct Worker<P: Params> {
    gamma: f32xN<{ <P as Params>::THREAD_N }>,
    x: [f32xN<{ <P as Params>::THREAD_N }>; <P as Params>::THREAD_M],
    r_inv: [f32; <P as Params>::THREAD_M],
}

#[unsafe(no_mangle)]
#[nvptx(max_registers(128), exact_threads_per_cta(256, 1, 1))]
pub unsafe extern "ptx-kernel" fn rmsnorm_fwd_Xbf16_Ybf16_RS2048_BS128(
    #[nvptx(grid_constant)] x: &CompoundTmaDescriptor,
    #[nvptx(grid_constant)] gamma: &CompoundTmaDescriptor,
    #[nvptx(grid_constant)] y: &CompoundTmaDescriptor,
    #[nvptx(grid_constant)] r_inv: &CompoundTmaDescriptor,
    actual_row_size: u32,
    rcp_actual_row_size: f32,
) {
    struct P;

    impl Params for P {
        const CGA_TM: usize = 8;
        const CGA_TN: usize = 2048;
        const CLUSTER_SIZE_M: usize = 1;
        const CLUSTER_SIZE_N: usize = 1;

        const MACROBLOCK_SIZE: usize = 16;
        const NUM_LOAD_SLOTS: usize = 2;

        const LARGEST_MXN_LOAD_BITS_PER_ELEMENT: usize = 16;
        const LARGEST_1XN_LOAD_BITS_PER_ELEMENT: usize = 16;
        const LARGEST_MX1_LOAD_BITS_PER_ELEMENT: usize = 0;

        const DOES_STORE_COLWISE_BLOCK_SCALED_DATA: bool = false;
    }

    assert!(actual_row_size <= 2048);

    unsafe {
        template::run(Impl::<P> {
            x,
            gamma,
            r_inv,
            rcp_actual_row_size,
            store_mode: StoreMode::Bf16(y),
            _marker: PhantomData,
        })
    };
}

#[unsafe(no_mangle)]
#[nvptx(max_registers(128), exact_threads_per_cta(256, 1, 1))]
pub unsafe extern "ptx-kernel" fn rmsnorm_fwd_Xbf16_Ymxfp8r_RS2048_BS128(
    #[nvptx(grid_constant)] x: &CompoundTmaDescriptor,
    #[nvptx(grid_constant)] gamma: &CompoundTmaDescriptor,
    #[nvptx(grid_constant)] y: &CompoundTmaDescriptor,
    y_sf_ptr: *mut u8,
    #[nvptx(grid_constant)] r_inv: &CompoundTmaDescriptor,
    actual_row_size: u32,
    rcp_actual_row_size: f32,
) {
    struct P;

    impl Params for P {
        const CGA_TM: usize = 8;
        const CGA_TN: usize = 2048;
        const CLUSTER_SIZE_M: usize = 1;
        const CLUSTER_SIZE_N: usize = 1;

        const MACROBLOCK_SIZE: usize = 16;
        const NUM_LOAD_SLOTS: usize = 2;

        const LARGEST_MXN_LOAD_BITS_PER_ELEMENT: usize = 16;
        const LARGEST_1XN_LOAD_BITS_PER_ELEMENT: usize = 16;
        const LARGEST_MX1_LOAD_BITS_PER_ELEMENT: usize = 0;

        const DOES_STORE_COLWISE_BLOCK_SCALED_DATA: bool = false;
    }

    assert!(actual_row_size <= 2048);

    unsafe {
        template::run(Impl::<P> {
            x,
            gamma,
            r_inv,
            rcp_actual_row_size,
            store_mode: StoreMode::Mxfp8Row(y, y_sf_ptr),
            _marker: PhantomData,
        })
    };
}

#[unsafe(no_mangle)]
#[nvptx(
    max_registers(128),
    exact_threads_per_cta(256, 1, 1),
    exact_cluster_dim(4, 1, 1)
)]
pub unsafe extern "ptx-kernel" fn rmsnorm_fwd_Xbf16_Ymxfp8rc_RS2048_BS128(
    #[nvptx(grid_constant)] x: &CompoundTmaDescriptor,
    #[nvptx(grid_constant)] gamma: &CompoundTmaDescriptor,
    #[nvptx(grid_constant)] y_rowwise: &CompoundTmaDescriptor,
    y_rowwise_sf_ptr: *mut u8,
    #[nvptx(grid_constant)] y_colwise: &CompoundTmaDescriptor,
    y_colwise_sf_ptr: *mut u8,
    #[nvptx(grid_constant)] r_inv: &CompoundTmaDescriptor,
    actual_row_size: u32,
    rcp_actual_row_size: f32,
) {
    struct P;

    impl Params for P {
        const CGA_TM: usize = 32;
        const CGA_TN: usize = 2048;
        const CLUSTER_SIZE_M: usize = 4;
        const CLUSTER_SIZE_N: usize = 1;

        const MACROBLOCK_SIZE: usize = 16;

        // Needs reduced to 1 to make room for the resharding buffers (64 KiB) while keeping
        // an occupancy of 2 CTA/SM on B200.
        const NUM_LOAD_SLOTS: usize = 1;

        const LARGEST_MXN_LOAD_BITS_PER_ELEMENT: usize = 16;
        const LARGEST_1XN_LOAD_BITS_PER_ELEMENT: usize = 16;
        const LARGEST_MX1_LOAD_BITS_PER_ELEMENT: usize = 0;

        const DOES_STORE_COLWISE_BLOCK_SCALED_DATA: bool = true;
    }

    assert!(actual_row_size <= 2048);

    unsafe {
        template::run(Impl::<P> {
            x,
            gamma,
            r_inv,
            rcp_actual_row_size,
            store_mode: StoreMode::Mxfp8RowAndCol(
                y_rowwise,
                y_rowwise_sf_ptr,
                y_colwise,
                y_colwise_sf_ptr,
            ),
            _marker: PhantomData,
        })
    };
}
