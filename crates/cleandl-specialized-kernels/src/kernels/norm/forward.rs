use super::NormType;
use crate::{
    error::{Error, Result},
    kernels::{
        HasCubin, KernelGenerator,
        norm::{MAX_SUPPORTED_ROW_SIZE, OutputTypes, compile_kernel_to_cubin},
    },
    utils::{
        arch::{max_cluster_size, max_smem, reg_bytes_per_sm},
        block_scale::BlockScaledFormat,
        chunked_thread_layout::{ChunkedThreadLayout2D, Resharder},
        clc::ClcScheduler,
        view::{View1D, View2D, ViewPosWithinTensor2D},
    },
};
use kernel_build_util::{
    ArgBundle, Kernel, LoadedKernelCache, Param, PtrArg,
    nvvm::{Args, KernelBuilder},
};
use nvvm_builder::{
    Scalar, ValueType, mbarrier::Pipeline, prelude::*, setmaxnreg_dec, setmaxnreg_inc,
};
use serde::{Deserialize, Serialize};
use skycuda::stream::Stream;
use std::cmp::Ordering;

pub struct NormForwardKernelGenerator;

impl NormForwardKernelGenerator {
    pub fn is_supported(config: super::Config, arch: &Arch) -> bool {
        Self::validate(config, arch).is_ok()
    }

    fn validate(mut config: super::Config, arch: &Arch) -> Result<(super::Config, TuningConfig)> {
        // NVFP4 values are packed in pairs, so each thread must hold at least two elements.
        let min_row_size = match config.output_types {
            OutputTypes::Nvfp4Rowwise => 256,
            _ => 128,
        };
        config.max_row_size = config.max_row_size.next_power_of_two().max(min_row_size);

        if config.max_row_size > MAX_SUPPORTED_ROW_SIZE {
            return Err(Error::Unsupported("row size > 8192"));
        }
        if config.norm_type == NormType::LayerNorm {
            return Err(Error::Unsupported("layer norm is not implemented"));
        }
        let tuning_config = TuningConfig::decide(&config, arch)?;
        Ok((config, tuning_config))
    }
}

impl KernelGenerator for NormForwardKernelGenerator {
    type Config = super::Config;
    type ExecArgs<'a> = ExecArgs<'a>;
    type BuiltKernel = BuiltKernel;

    fn generate_kernel(&self, config: Self::Config, arch: Arch) -> Result<BuiltKernel> {
        let (config, tuning_config) = Self::validate(config, &arch)?;

        let mut module = ModuleBuilder::new("norm_fwd");

        let mut kb = KernelBuilder::new();
        let params = Params {
            x: kb.param("x"),
            actual_row_size: kb.param("actual_row_size"),
            recip_actual_row_size: kb.param("recip_actual_row_size"),
            num_rows: kb.param("num_rows"),
            row_stride: kb.param("row_stride"),
            eps: kb.param("eps"),
            r_inv: kb.param("r_inv"),
            bias: config.has_bias.then(|| kb.param("bias")),
            scale: config.has_scale.then(|| kb.param("scale")),
            y: match config.output_types {
                OutputTypes::Bf16 => OutputParams::Bf16 {
                    y: kb.param("y_bf16"),
                },
                OutputTypes::Mxfp8Rowwise => OutputParams::Mxfp8Rowwise {
                    y_row: kb.param("y_mxfp8_row"),
                    y_row_sf: kb.param("y_mxfp8_row_sf"),
                },
                OutputTypes::Mxfp8RowwiseAndColwise => OutputParams::Mxfp8RowwiseAndColwise {
                    y_row: kb.param("y_mxfp8_row"),
                    y_row_sf: kb.param("y_mxfp8_row_sf"),
                    y_col: kb.param("y_mxfp8_col"),
                    y_col_sf: kb.param("y_mxfp8_col_sf"),
                },
                OutputTypes::Nvfp4Rowwise => OutputParams::Nvfp4Rowwise {
                    y_row: kb.param("y_nvfp4_row"),
                    y_row_sf: kb.param("y_nvfp4_row_sf"),
                },
            },
        };

        tracing::debug!(?config, ?tuning_config, "selected tuning config");

        let spec = kb.define(
            &mut module,
            "norm_fwd",
            KernelOpts::new()
                .cluster_dims([tuning_config.cluster_size, 1, 1])
                .req_threads([tuning_config.threads_per_block(), 1, 1])
                .restrict_pointers(true)
                .min_blocks_per_sm(tuning_config.occupancy)
                .max_regs(tuning_config.max_regs_per_thread(&arch)),
            |args| {
                let mut smem = SharedCarve::dynamic();

                const CLC_STAGES: usize = 3;

                let layout = ChunkedThreadLayout2D::new(
                    config.max_row_size,
                    tuning_config.rows_per_tile as usize,
                    tuning_config.num_worker_threads as usize,
                );

                let builder = Builder {
                    config,
                    tuning_config,
                    worker_thread_regs: tuning_config.worker_regs_per_thread(&arch),
                    load_pipeline: Pipeline::shared(
                        "load_pipeline",
                        tuning_config.num_smem_staging_slots,
                        mbarrier::Release::Threads(tuning_config.num_worker_threads),
                    ),
                    load_slots_base: smem.array::<bf16>(
                        (tuning_config.num_smem_staging_slots * tuning_config.rows_per_tile)
                            as usize
                            * config.max_row_size,
                        16,
                    ),
                    clc: ClcScheduler::init(CLC_STAGES, 0, &mut smem),
                    resharder: (tuning_config.cluster_size > 1).then(|| {
                        Resharder::init(tuning_config.cluster_size, &layout, &mut smem, 4)
                    }),
                    layout,
                    params,
                    args,
                };

                args.set_dynamic_smem_bytes(smem.size());

                with_leader_of_warp(0, || {
                    builder.load_pipeline.init();
                    builder.load_slots_base.zero_bulk(
                        tuning_config.num_smem_staging_slots
                            * tuning_config.rows_per_tile
                            * config.max_row_size as u32,
                    )
                });

                if tuning_config.cluster_size > 1 {
                    cluster_sync();
                } else {
                    sync_threads();
                }

                builder.build();

                if tuning_config.cluster_size > 1 {
                    cluster_sync();
                }
            },
        );

        let kernel = spec.finish(compile_kernel_to_cubin(module, arch)?)?;

        Ok(BuiltKernel {
            config,
            tuning_config,
            params,
            kernel,
            loaded: LoadedKernelCache::default(),
        })
    }

    unsafe fn execute_kernel<'a>(
        &self,
        kernel: &'a BuiltKernel,
        args: ExecArgs<'a>,
        stream: &'a Stream,
    ) -> Result<()> {
        let BuiltKernel {
            config,
            tuning_config,
            params,
            ..
        } = kernel;

        assert!(
            args.row_size as usize <= config.max_row_size,
            "row size {} exceeds the kernel's max row size {}",
            args.row_size,
            config.max_row_size
        );
        assert!(args.row_size > 0, "row size must be nonzero");
        if args.num_rows == 0 {
            return Ok(());
        }
        match config.output_types {
            OutputTypes::Bf16 => {}
            OutputTypes::Mxfp8Rowwise => assert!(
                args.row_size
                    .is_multiple_of(BlockScaledFormat::Mxfp8.block_size() as u32),
                "MXFP8 output requires the row size to be a multiple of the block size"
            ),
            OutputTypes::Mxfp8RowwiseAndColwise => assert!(
                args.row_size
                    .is_multiple_of(BlockScaledFormat::Mxfp8.block_size() as u32)
                    && args
                        .num_rows
                        .is_multiple_of(BlockScaledFormat::Mxfp8.block_size() as u32),
                "MXFP8 rowwise and colwise output requires the row size and row count to be multiples of the block size"
            ),
            OutputTypes::Nvfp4Rowwise => assert!(
                args.row_size
                    .is_multiple_of(BlockScaledFormat::Nvfp4.block_size() as u32),
                "NVFP4 output requires the row size to be a multiple of the block size"
            ),
        }

        let mut bundle = ArgBundle::new();
        bundle.set(params.x, args.x);
        bundle.set(params.actual_row_size, args.row_size);
        bundle.set(params.recip_actual_row_size, 1.0 / args.row_size as f32);
        bundle.set(params.num_rows, args.num_rows);
        bundle.set(params.row_stride, args.row_stride);
        bundle.set(params.eps, args.eps);
        bundle.set(params.r_inv, args.r_inv.unwrap_or_else(PtrArg::null));

        match (params.scale, args.scale) {
            (Some(param), Some(scale)) => bundle.set(param, scale),
            (None, None) => {}
            _ => panic!("scale must be passed iff config.has_scale is set"),
        }
        match (params.bias, args.bias) {
            (Some(param), Some(bias)) => bundle.set(param, bias),
            (None, None) => {}
            _ => panic!("bias must be passed iff config.has_bias is set"),
        }

        match (params.y, args.y) {
            (OutputParams::Bf16 { y }, OutputBuffers::Bf16 { y: y_buf }) => bundle.set(y, y_buf),
            (
                OutputParams::Mxfp8Rowwise { y_row, y_row_sf },
                OutputBuffers::Mxfp8Rowwise {
                    y_row: y_row_buf,
                    y_row_sf: y_row_sf_buf,
                },
            ) => {
                bundle.set(y_row, y_row_buf);
                bundle.set(y_row_sf, y_row_sf_buf);
            }
            (
                OutputParams::Mxfp8RowwiseAndColwise {
                    y_row,
                    y_row_sf,
                    y_col,
                    y_col_sf,
                },
                OutputBuffers::Mxfp8RowwiseAndColwise {
                    y_row: y_row_buf,
                    y_row_sf: y_row_sf_buf,
                    y_col: y_col_buf,
                    y_col_sf: y_col_sf_buf,
                },
            ) => {
                bundle.set(y_row, y_row_buf);
                bundle.set(y_row_sf, y_row_sf_buf);
                bundle.set(y_col, y_col_buf);
                bundle.set(y_col_sf, y_col_sf_buf);
            }
            (
                OutputParams::Nvfp4Rowwise { y_row, y_row_sf },
                OutputBuffers::Nvfp4Rowwise {
                    y_row: y_row_buf,
                    y_row_sf: y_row_sf_buf,
                },
            ) => {
                bundle.set(y_row, y_row_buf);
                bundle.set(y_row_sf, y_row_sf_buf);
            }
            _ => panic!("output buffers do not match config.output_types"),
        }

        let num_cluster_tiles = args
            .num_rows
            .div_ceil(tuning_config.rows_per_cluster_tile());
        let num_cluster_schedule_units =
            num_cluster_tiles.div_ceil(tuning_config.tiles_per_schedule_unit);
        let grid_x = num_cluster_schedule_units * tuning_config.cluster_size;

        unsafe {
            let loaded = kernel.loaded.get_or_load(&kernel.kernel, stream.device())?;
            loaded.launch(bundle, (grid_x, 1, 1), stream)?;
        }

        Ok(())
    }
}

/// Arguments for [`NormForwardKernelGenerator`]. Tensors are row-major
/// with `row_stride` elements between rows, except for colwise outputs, which
/// are column-major with `num_rows` elements between columns.
///
/// Block-scaled outputs require the row size to be a multiple of the block size;
/// colwise outputs additionally require this of the row count. NVFP4 outputs
/// are quantized without a global (per-tensor) scale.
pub struct ExecArgs<'a> {
    pub x: PtrArg<'a, bf16>,
    pub num_rows: u32,
    /// Must not exceed the kernel's `max_row_size`.
    pub row_size: u32,
    pub row_stride: u32,
    /// Added to the mean square (or variance) before taking the reciprocal square root.
    pub eps: f32,
    /// Per-row reciprocal of the RMS (or standard deviation), written if present.
    pub r_inv: Option<PtrArg<'a, f32>>,
    /// Must be present iff `has_scale` is set in the config.
    pub scale: Option<PtrArg<'a, bf16>>,
    /// Must be present iff `has_bias` is set in the config.
    pub bias: Option<PtrArg<'a, bf16>>,
    /// Must match `output_types` in the config.
    pub y: OutputBuffers<'a>,
}

pub enum OutputBuffers<'a> {
    Bf16 {
        y: PtrArg<'a, bf16>,
    },
    Mxfp8Rowwise {
        y_row: PtrArg<'a, F8E4M3>,
        y_row_sf: PtrArg<'a, UE8M0>,
    },
    Mxfp8RowwiseAndColwise {
        y_row: PtrArg<'a, F8E4M3>,
        y_row_sf: PtrArg<'a, UE8M0>,
        y_col: PtrArg<'a, F8E4M3>,
        y_col_sf: PtrArg<'a, UE8M0>,
    },
    Nvfp4Rowwise {
        y_row: PtrArg<'a, F4E2M1>,
        y_row_sf: PtrArg<'a, F8E4M3>,
    },
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
struct Params {
    x: Param<Ptr<bf16, Global>>,
    actual_row_size: Param<u32>,
    recip_actual_row_size: Param<f32>,
    num_rows: Param<u32>,
    row_stride: Param<u32>,
    eps: Param<f32>,
    r_inv: Param<Ptr<f32, Global>>,
    bias: Option<Param<Ptr<bf16, Global>>>,
    scale: Option<Param<Ptr<bf16, Global>>>,
    y: OutputParams,
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
enum OutputParams {
    Bf16 {
        y: Param<Ptr<bf16, Global>>,
    },
    Mxfp8Rowwise {
        y_row: Param<Ptr<F8E4M3, Global>>,
        y_row_sf: Param<Ptr<UE8M0, Global>>,
    },
    Mxfp8RowwiseAndColwise {
        y_row: Param<Ptr<F8E4M3, Global>>,
        y_row_sf: Param<Ptr<UE8M0, Global>>,
        y_col: Param<Ptr<F8E4M3, Global>>,
        y_col_sf: Param<Ptr<UE8M0, Global>>,
    },
    Nvfp4Rowwise {
        y_row: Param<Ptr<F4E2M1, Global>>,
        y_row_sf: Param<Ptr<F8E4M3, Global>>,
    },
}

struct Builder<'a> {
    config: super::Config,
    tuning_config: TuningConfig,
    worker_thread_regs: u32,
    load_pipeline: Pipeline,
    load_slots_base: Val<Ptr<bf16, Shared>>,
    clc: ClcScheduler,
    resharder: Option<Resharder<bf16>>,
    layout: ChunkedThreadLayout2D,
    params: Params,
    args: &'a Args,
}

impl<'a> Builder<'a> {
    pub fn build(&self) {
        with_warps(0..4, || {
            setmaxnreg_dec(AUX_THREAD_REGS);

            with_warp(0, || {
                self.clc.emit_scheduler_loop_on_leader_warp();
            });

            with_warp(1, || {
                self.build_tma_loader_warp();
            });

            with_warps(2..4, || {
                self.build_idle_warp();
            });
        });

        with_warps(4..4 + self.tuning_config.num_worker_warps, || {
            setmaxnreg_inc(self.worker_thread_regs);
            self.build_worker_warps();
        });
    }

    fn cluster_tile_idx(
        &self,
        schedule_unit: Val<u32>,
        tile_in_schedule_unit: Val<u32>,
    ) -> Val<u32> {
        (schedule_unit / self.tuning_config.cluster_size)
            * self.tuning_config.tiles_per_schedule_unit
            + tile_in_schedule_unit
    }

    /// Index of the CTA-local tile processed in the given iteration.
    fn tile_idx(&self, schedule_unit: Val<u32>, tile_in_schedule_unit: Val<u32>) -> Val<u32> {
        self.cluster_tile_idx(schedule_unit, tile_in_schedule_unit)
            * self.tuning_config.cluster_size
            + schedule_unit % self.tuning_config.cluster_size
    }

    fn build_idle_warp(&self) {
        loop_(|l| {
            let (has_next, _, _, _) = self.clc.get_next_block_idx();
            if_(!has_next, || {
                l.break_();
            });
        });
    }

    fn build_tma_loader_warp(&self) {
        let pipeline_state = self.load_pipeline.producer_state();

        let current_schedule_unit = Var::new(block_idx_x());

        loop_(|l| {
            for_range(
                0,
                self.tuning_config.tiles_per_schedule_unit,
                |tile_in_schedule_unit| {
                    let first_row_in_tile = self
                        .tile_idx(current_schedule_unit.into_val(), tile_in_schedule_unit)
                        * self.tuning_config.rows_per_tile;

                    let stage = self.load_pipeline.producer_acquire(pipeline_state);

                    let dst = View2D {
                        base: self.load_slots_base.offset(
                            stage.index()
                                * self.tuning_config.rows_per_tile
                                * self.config.max_row_size as u32,
                        ),
                        inner_len: Val::new(self.config.max_row_size as u32),
                        outer_len: Val::new(self.tuning_config.rows_per_tile),
                        outer_stride: Val::new(self.config.max_row_size as u32),
                    };
                    let src = View2D {
                        base: self
                            .args
                            .get(self.params.x)
                            .offset(first_row_in_tile * self.args.get(self.params.row_stride)),
                        inner_len: self.args.get(self.params.actual_row_size),
                        outer_len: (self
                            .args
                            .get(self.params.num_rows)
                            .saturating_sub(first_row_in_tile))
                        .min(self.tuning_config.rows_per_tile),
                        outer_stride: self.args.get(self.params.row_stride),
                    };

                    with_leader(|| {
                        let bar = stage.arrive_expect_tx(
                            self.layout.async_bulk_copy_g2s_aligned_num_bytes(src, dst),
                        );
                        self.layout.async_bulk_copy_g2s_aligned(src, dst, bar);
                    });

                    pipeline_state.advance();
                },
            );

            let (has_next, next_schedule_unit, _, _) = self.clc.get_next_block_idx();
            if_else(
                has_next,
                || {
                    current_schedule_unit.set(next_schedule_unit);
                },
                || {
                    l.break_();
                },
            );
        });
    }

    fn build_worker_warps(&self) {
        let pipeline_state = self.load_pipeline.consumer_state();

        let current_schedule_unit = Var::new(block_idx_x());

        let thread_idx = thread_idx_x() - 128;

        // Load scale and bias, if needed.
        let scale = self.params.scale.map(|scale| {
            self.layout.as_1d().load_aligned_or_zero(
                View1D::<bf16, Global> {
                    base: self.args.get(scale),
                    len: self.args.get(self.params.actual_row_size),
                },
                thread_idx,
            )
        });
        let bias = self.params.bias.map(|bias| {
            self.layout.as_1d().load_aligned_or_zero(
                View1D::<bf16, Global> {
                    base: self.args.get(bias),
                    len: self.args.get(self.params.actual_row_size),
                },
                thread_idx,
            )
        });

        let i = Var::<u32>::new(0);

        loop_(|l| {
            for_range(
                0,
                self.tuning_config.tiles_per_schedule_unit,
                |tile_in_schedule_unit| {
                    let cluster_tile = self
                        .cluster_tile_idx(current_schedule_unit.into_val(), tile_in_schedule_unit);
                    let first_row_in_cluster_tile =
                        cluster_tile * self.tuning_config.rows_per_cluster_tile();
                    let first_row_in_tile = self
                        .tile_idx(current_schedule_unit.into_val(), tile_in_schedule_unit)
                        * self.tuning_config.rows_per_tile;
                    let stage = self.load_pipeline.consumer_wait(pipeline_state);
                    let reduction_slot_idx = i % 2;

                    // Staging slots are only filled up to the actual row size, so
                    // elements beyond it must be zeroed to not pollute the reduction.
                    let slot_base = self.load_slots_base.offset(
                        stage.index()
                            * self.tuning_config.rows_per_tile
                            * self.config.max_row_size as u32,
                    );
                    let x = (0..self.tuning_config.rows_per_tile)
                        .map(|row_idx| {
                            self.layout.as_1d().load_aligned_unchecked(
                                slot_base.offset(row_idx * self.config.max_row_size as u32),
                                thread_idx,
                            )
                        })
                        .collect::<Vec<_>>();

                    let y = match self.config.norm_type {
                        NormType::RmsNorm => {
                            let smem_row_sums_of_squares = shared_shaped::<Vector<f32>>(
                                "row_sums_of_squares",
                                2 * self.tuning_config.num_worker_warps as usize,
                                16,
                                self.tuning_config.rows_per_tile as usize,
                            );

                            let local_sums_of_squares = x
                                .iter()
                                .map(|row_chunks| {
                                    let row_chunks = row_chunks.convert::<f32>(Default::default());
                                    (row_chunks * row_chunks).reduce_add()
                                })
                                .collect::<Vec<_>>();

                            let local_sums_of_squares =
                                Var::new(Val::from_lanes(local_sums_of_squares));
                            for bit in 0..5 {
                                local_sums_of_squares.set(
                                    local_sums_of_squares
                                        + shfl_sync_bfly(u32::MAX, local_sums_of_squares, 1 << bit),
                                );
                            }

                            with_leader(|| {
                                smem_row_sums_of_squares
                                    .offset(
                                        self.tuning_config.num_worker_warps * reduction_slot_idx
                                            + (warp_id() - 4),
                                    )
                                    .store(local_sums_of_squares);
                            });
                            named_barrier(0x2, self.tuning_config.num_worker_threads);

                            let global_sums_of_squares = Var::new(Val::<Vector<f32>>::zero_shaped(
                                local_sums_of_squares.lanes(),
                            ));
                            for w in 0..self.tuning_config.num_worker_warps {
                                global_sums_of_squares.set(
                                    global_sums_of_squares
                                        + smem_row_sums_of_squares
                                            .offset(
                                                self.tuning_config.num_worker_warps
                                                    * reduction_slot_idx
                                                    + w,
                                            )
                                            .load(),
                                );
                            }

                            let r_inv = (global_sums_of_squares
                                * self.args.get(self.params.recip_actual_row_size)
                                + self.args.get(self.params.eps))
                            .rsqrt();

                            // Store r_inv
                            if_(thread_idx.eq(0), || {
                                for row_idx in 0..self.tuning_config.rows_per_tile {
                                    let row_r_inv = r_inv.get(row_idx as usize);
                                    let coord = first_row_in_tile + row_idx;
                                    if_(coord.lt(self.args.get(self.params.num_rows)), || {
                                        let dst = self.args.get(self.params.r_inv);
                                        if_(!dst.is_null(), || {
                                            dst.offset(coord).store(row_r_inv);
                                        });
                                    });
                                }
                            });

                            let mut y_rows = Vec::new();
                            for row_idx in 0..self.tuning_config.rows_per_tile {
                                let row_r_inv = r_inv.get(row_idx as usize);

                                let mut y_row = x[row_idx as usize]
                                    .convert::<f32>(Default::default())
                                    * row_r_inv;
                                if let Some(scale) = scale {
                                    y_row = y_row * scale.convert::<f32>(Default::default());
                                }
                                if let Some(bias) = bias {
                                    y_row = y_row + bias.convert::<f32>(Default::default());
                                }
                                y_rows.push(y_row.convert::<bf16>(Default::default()));
                            }

                            y_rows
                        }
                        NormType::LayerNorm => todo!(),
                    };

                    // Store results. If we need to reshard for MXFP8 colwise store, initiate this first
                    // to hide latency.
                    let resharding_in_progress = self
                        .resharder
                        .as_ref()
                        .map(|resharder| resharder.initiate_resharding(&y, thread_idx));

                    let valid_rows_in_tile = (self
                        .args
                        .get(self.params.num_rows)
                        .saturating_sub(first_row_in_tile))
                    .min(self.tuning_config.rows_per_tile);

                    match self.params.y {
                        OutputParams::Bf16 { y: y_param } => {
                            self.layout.store_aligned(
                                View2D::<_, Global> {
                                    base: self.args.get(y_param).offset(
                                        first_row_in_tile * self.args.get(self.params.row_stride),
                                    ),
                                    inner_len: self.args.get(self.params.actual_row_size),
                                    outer_len: valid_rows_in_tile,
                                    outer_stride: self.args.get(self.params.row_stride),
                                },
                                &y,
                                thread_idx,
                            );
                        }
                        OutputParams::Mxfp8Rowwise { y_row, y_row_sf } => {
                            self.store_rowwise_block_scaled(
                                BlockScaledFormat::Mxfp8,
                                y_row,
                                y_row_sf,
                                first_row_in_tile,
                                valid_rows_in_tile,
                                &y,
                                thread_idx,
                            );
                        }
                        OutputParams::Mxfp8RowwiseAndColwise {
                            y_row,
                            y_row_sf,
                            y_col,
                            y_col_sf,
                        } => {
                            self.store_rowwise_block_scaled(
                                BlockScaledFormat::Mxfp8,
                                y_row,
                                y_row_sf,
                                first_row_in_tile,
                                valid_rows_in_tile,
                                &y,
                                thread_idx,
                            );

                            let (resharded_layout, resharded_vals) = match resharding_in_progress {
                                Some(reshard) => reshard.finish_resharding(),
                                None => (self.layout, y),
                            };

                            // After resharding, each CTA holds all rows of the cluster tile
                            // for a contiguous range of columns.
                            let cols_per_cta = resharded_layout.num_elements_per_row() as u32;
                            let first_col = if self.tuning_config.cluster_size > 1 {
                                cluster_block_rank() * cols_per_cta
                            } else {
                                Val::new(0)
                            };
                            let num_rows = self.args.get(self.params.num_rows);
                            let row_size = self.args.get(self.params.actual_row_size);
                            resharded_layout.store_aligned_colwise_block_scaled::<F8E4M3, UE8M0>(
                                BlockScaledFormat::Mxfp8,
                                View2D::<_, Global> {
                                    base: self
                                        .args
                                        .get(y_col)
                                        .offset(first_col * num_rows + first_row_in_cluster_tile),
                                    inner_len: num_rows
                                        .saturating_sub(first_row_in_cluster_tile)
                                        .min(self.tuning_config.rows_per_cluster_tile()),
                                    outer_len: row_size.saturating_sub(first_col).min(cols_per_cta),
                                    outer_stride: num_rows,
                                },
                                ViewPosWithinTensor2D {
                                    row_offset: first_row_in_cluster_tile,
                                    col_offset: first_col,
                                    total_tensor_rows: num_rows,
                                    total_tensor_cols: row_size,
                                },
                                self.args.get(y_col_sf),
                                &resharded_vals,
                                thread_idx,
                            );
                        }
                        OutputParams::Nvfp4Rowwise { y_row, y_row_sf } => {
                            self.store_rowwise_block_scaled(
                                BlockScaledFormat::Nvfp4,
                                y_row,
                                y_row_sf,
                                first_row_in_tile,
                                valid_rows_in_tile,
                                &y,
                                thread_idx,
                            );
                        }
                    }

                    stage.release();
                    pipeline_state.advance();
                    i.set(i + 1);
                },
            );

            let (has_next, next_schedule_unit, _, _) = self.clc.get_next_block_idx();
            if_else(
                has_next,
                || {
                    current_schedule_unit.set(next_schedule_unit);
                },
                || {
                    l.break_();
                },
            );
        });
    }
}

impl Builder<'_> {
    #[allow(clippy::too_many_arguments)]
    fn store_rowwise_block_scaled<Q: ValueType + Scalar, SF: ValueType + Scalar>(
        &self,
        format: BlockScaledFormat,
        y_row: Param<Ptr<Q, Global>>,
        y_row_sf: Param<Ptr<SF, Global>>,
        first_row_in_tile: Val<u32>,
        valid_rows_in_tile: Val<u32>,
        y: &[Val<Vector<bf16>>],
        thread_idx: Val<u32>,
    ) {
        let row_stride = self.args.get(self.params.row_stride);
        let row_size = self.args.get(self.params.actual_row_size);
        self.layout.store_aligned_rowwise_block_scaled::<Q, SF>(
            format,
            View2D::<_, Global> {
                base: self.args.get(y_row).offset(first_row_in_tile * row_stride),
                inner_len: row_size,
                outer_len: valid_rows_in_tile,
                outer_stride: row_stride,
            },
            ViewPosWithinTensor2D {
                row_offset: first_row_in_tile,
                col_offset: Val::new(0),
                total_tensor_rows: self.args.get(self.params.num_rows),
                total_tensor_cols: row_size,
            },
            self.args.get(y_row_sf),
            y,
            thread_idx,
        );
    }
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
struct TuningConfig {
    occupancy: u32,
    num_worker_warps: u32,
    num_worker_threads: u32,
    num_smem_staging_slots: u32,
    cluster_size: u32,
    rows_per_tile: u32,
    tiles_per_schedule_unit: u32,
}

/// Register count for the 4 non-worker warps after `setmaxnreg.dec`.
const AUX_THREAD_REGS: u32 = 32;
const NUM_AUX_THREADS: u32 = 128;

impl TuningConfig {
    fn threads_per_block(&self) -> u32 {
        NUM_AUX_THREADS + self.num_worker_threads
    }

    fn rows_per_cluster_tile(&self) -> u32 {
        self.rows_per_tile * self.cluster_size
    }

    /// Launch-time per-thread register limit; register allocation granularity is 8 per thread.
    fn max_regs_per_thread(&self, arch: &Arch) -> u32 {
        let regs = reg_bytes_per_sm(arch) as u32
            / self.occupancy
            / size_of::<u32>() as u32
            / self.threads_per_block();
        regs.min(256) / 8 * 8
    }

    /// Largest `setmaxnreg.inc` target for worker warps that the CTA's register
    /// pool can satisfy after the aux warps release down to `AUX_THREAD_REGS`.
    /// Requesting more than this makes `setmaxnreg.inc` block forever.
    fn worker_regs_per_thread(&self, arch: &Arch) -> u32 {
        let pool = self.max_regs_per_thread(arch) * self.threads_per_block();
        let available = pool - AUX_THREAD_REGS * NUM_AUX_THREADS;
        (available / self.num_worker_threads).min(256) / 8 * 8
    }

    pub fn decide(config: &super::Config, arch: &Arch) -> Result<Self> {
        let reg_capacity_per_sm = reg_bytes_per_sm(arch);
        let smem_capacity_per_sm = max_smem(arch);

        let reg_bytes_per_row = config.max_row_size * size_of::<f32>();
        let reg_aux_bytes = (config.has_scale as usize + config.has_bias as usize)
            * config.max_row_size
            * size_of::<bf16>();

        let staging_slot_bytes_per_row = reg_bytes_per_row;

        fn cmp(first: &TuningConfig, second: &TuningConfig) -> Ordering {
            (
                first.occupancy,
                first.rows_per_tile,
                first.num_smem_staging_slots,
            )
                .cmp(&(
                    second.occupancy,
                    second.rows_per_tile,
                    second.num_smem_staging_slots,
                ))
        }

        fn apply_reg_overhead(regs: usize) -> usize {
            regs * 3 / 2 // heuristic
        }

        let mut best = None::<TuningConfig>;

        for occupancy in 1..=2 {
            let reg_capacity = reg_capacity_per_sm / occupancy;

            const FIXED_SMEM: usize = 1024; // mbarriers, reduction scratchpad, etc.
            let smem_capacity = smem_capacity_per_sm / occupancy - FIXED_SMEM;

            let num_worker_threads = 8 / occupancy * 32;
            let min_elements_per_thread = match config.output_types {
                OutputTypes::Nvfp4Rowwise => 2,
                _ => 1,
            };
            if config.max_row_size < num_worker_threads * min_elements_per_thread {
                continue;
            }

            for rows_per_tile in [1usize, 2, 4, 8, 12, 16, 20, 24, 28, 32, 48, 64] {
                let cluster_size = match config.output_types {
                    OutputTypes::Mxfp8RowwiseAndColwise => {
                        let block_size = BlockScaledFormat::Mxfp8.block_size();
                        if rows_per_tile >= block_size {
                            if !rows_per_tile.is_multiple_of(block_size) {
                                continue;
                            }
                            1
                        } else {
                            if !block_size.is_multiple_of(rows_per_tile) {
                                continue;
                            }
                            block_size / rows_per_tile
                        }
                    }
                    _ => 1,
                };
                if cluster_size > max_cluster_size(arch) {
                    continue;
                }
                if config.max_row_size / cluster_size < num_worker_threads {
                    continue;
                }

                let regs = apply_reg_overhead(rows_per_tile * reg_bytes_per_row + reg_aux_bytes);
                if regs > reg_capacity {
                    continue;
                }

                let reshard_buffer_bytes = if cluster_size > 1 {
                    rows_per_tile * config.max_row_size * size_of::<bf16>()
                } else {
                    0
                };

                for num_smem_staging_slots in 1..=4 {
                    let smem = staging_slot_bytes_per_row * num_smem_staging_slots * rows_per_tile
                        + reshard_buffer_bytes;
                    if smem > smem_capacity {
                        continue;
                    }

                    let config = TuningConfig {
                        occupancy: occupancy as u32,
                        num_worker_warps: num_worker_threads as u32 / 32,
                        num_worker_threads: num_worker_threads as u32,
                        num_smem_staging_slots: num_smem_staging_slots as u32,
                        cluster_size: cluster_size as u32,
                        rows_per_tile: rows_per_tile as u32,
                        tiles_per_schedule_unit: rows_per_tile as u32 * 8,
                    };

                    match &mut best {
                        Some(best) => {
                            if cmp(&config, best).is_gt() {
                                *best = config;
                            }
                        }
                        None => best = Some(config),
                    }
                }
            }
        }

        best.ok_or(Error::Unsupported("no tuning params valid for config"))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuiltKernel {
    config: super::Config,
    tuning_config: TuningConfig,
    params: Params,
    kernel: Kernel,
    #[serde(skip)]
    loaded: LoadedKernelCache,
}

impl BuiltKernel {
    pub fn config(&self) -> &super::Config {
        &self.config
    }
}

impl HasCubin for BuiltKernel {
    fn cubin(&self) -> &[u8] {
        self.kernel.cubin()
    }
}
