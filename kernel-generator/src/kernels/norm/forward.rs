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
        params::{DynParamsBuilder, dtype},
        view::{View1D, View2D, ViewPosWithinTensor2D},
    },
};
use nvvm_builder::{mbarrier::Pipeline, prelude::*, setmaxnreg_dec, setmaxnreg_inc};
use serde::{Deserialize, Serialize};
use skycuda::stream::Stream;
use std::cmp::Ordering;

pub struct NormForwardKernelGenerator;

impl KernelGenerator for NormForwardKernelGenerator {
    type Config = super::Config;
    type ExecArgs<'a> = ();
    type BuiltKernel = BuiltKernel;

    fn generate_kernel(&self, mut config: Self::Config, arch: Arch) -> Result<BuiltKernel> {
        config.max_row_size = config.max_row_size.next_power_of_two();

        if config.max_row_size > MAX_SUPPORTED_ROW_SIZE {
            return Err(Error::Unsupported("row size > 8192"));
        }

        let mut module = ModuleBuilder::new("norm_fwd");

        let mut params = DynParamsBuilder::new();
        params.add(Param::X, dtype::<Ptr<bf16, Global>>());
        params.add(Param::ActualRowSize, dtype::<u32>());
        params.add(Param::RecipActualRowSize, dtype::<f32>());
        params.add(Param::NumRows, dtype::<u32>());
        params.add(Param::RowStride, dtype::<u32>());
        params.add(Param::RInv, dtype::<Ptr<f32, Global>>());

        if config.has_bias {
            params.add(Param::Bias, dtype::<Ptr<bf16, Global>>());
        }
        if config.has_scale {
            params.add(Param::Scale, dtype::<Ptr<bf16, Global>>());
        }

        match config.output_types {
            OutputTypes::Bf16 => {
                params.add(Param::YBf16, dtype::<Ptr<bf16, Global>>());
            }
            OutputTypes::Mxfp8Rowwise => {
                params.add(Param::YMxfp8Row, dtype::<Ptr<F8E4M3, Global>>());
                params.add(Param::YMxfp8RowSf, dtype::<Ptr<UE8M0, Global>>());
            }
            OutputTypes::Mxfp8RowwiseAndColwise => {
                params.add(Param::YMxfp8Row, dtype::<Ptr<F8E4M3, Global>>());
                params.add(Param::YMxfp8RowSf, dtype::<Ptr<UE8M0, Global>>());
                params.add(Param::YMxfp8Col, dtype::<Ptr<F8E4M3, Global>>());
                params.add(Param::YMxfp8ColSf, dtype::<Ptr<UE8M0, Global>>());
            }
            OutputTypes::Nvfp4Rowwise => {
                params.add(Param::YNvfp4Row, dtype::<Ptr<F4E2M1, Global>>());
                params.add(Param::YNvfp4RowSf, dtype::<Ptr<F8E4M3, Global>>());
            }
        };

        let tuning_config = TuningConfig::decide(&config, &arch)?;

        tracing::debug!(?config, ?tuning_config, "selected tuning config");

        module.kernel_dyn(
            "norm_fwd",
            KernelOpts::new()
                .cluster_dims([tuning_config.cluster_size, 1, 1])
                .req_threads([128 + tuning_config.num_worker_threads, 1, 1])
                .restrict_pointers(true)
                .min_blocks_per_sm(tuning_config.occupancy)
                .max_regs(
                    reg_bytes_per_sm(&arch) as u32
                        / tuning_config.occupancy
                        / size_of::<u32>() as u32
                        / (128 + tuning_config.num_worker_threads),
                ),
            &params.param_dtypes(),
            |args| {
                params.bind(args);

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
                    params: &params,
                };

                with_leader_of_warp(0, || {
                    builder.load_pipeline.init();
                });

                if tuning_config.cluster_size > 1 {
                    cluster_sync();
                } else {
                    sync_threads();
                }

                builder.build();
            },
        );

        let cubin = compile_kernel_to_cubin(module, arch)?;

        Ok(BuiltKernel {
            config,
            tuning_config,
            cubin,
        })
    }

    unsafe fn execute_kernel<'a>(
        &self,
        kernel: &BuiltKernel,
        args: Self::ExecArgs<'a>,
        stream: &Stream,
    ) -> Result<()> {
        todo!()
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
enum Param {
    X,
    Scale,
    Bias,
    YBf16,
    YMxfp8Row,
    YMxfp8RowSf,
    YMxfp8Col,
    YMxfp8ColSf,
    YNvfp4Row,
    YNvfp4RowSf,
    RInv,
    ActualRowSize,
    RecipActualRowSize,
    NumRows,
    RowStride,
}

struct Builder<'a> {
    config: super::Config,
    tuning_config: TuningConfig,
    load_pipeline: Pipeline,
    load_slots_base: Val<Ptr<bf16, Shared>>,
    clc: ClcScheduler,
    resharder: Option<Resharder<bf16>>,
    layout: ChunkedThreadLayout2D,
    params: &'a DynParamsBuilder<Param>,
}

impl<'a> Builder<'a> {
    pub fn build(&self) {
        let (aux_thread_regs, worker_thread_regs) = (32, 232);

        with_warps(0..4, || {
            setmaxnreg_dec(aux_thread_regs);

            with_warp(0, || {
                self.clc.emit_scheduler_loop_on_leader_warp();
            });

            with_warp(1, || {
                self.build_tma_loader_warp();
            });

            with_warps(1..4, || {
                self.build_idle_warp();
            });
        });

        with_warps(4..4 + self.tuning_config.num_worker_warps, || {
            setmaxnreg_inc(worker_thread_regs);
            self.build_worker_warps();
        });
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
                current_schedule_unit * self.tuning_config.tiles_per_schedule_unit,
                (current_schedule_unit + 1) * self.tuning_config.tiles_per_schedule_unit,
                |tile| {
                    let first_row_in_tile = tile * self.tuning_config.rows_per_tile;

                    let stage = self.load_pipeline.producer_acquire(pipeline_state);

                    let dst = View2D {
                        base: self.load_slots_base.offset(
                            stage.index()
                                * size_of::<bf16>() as u32
                                * self.config.max_row_size as u32,
                        ),
                        inner_len: Val::new(self.config.max_row_size as u32),
                        outer_len: Val::new(self.tuning_config.rows_per_tile),
                        outer_stride: Val::new(self.config.max_row_size as u32),
                    };
                    let src = View2D {
                        base: self
                            .params
                            .get(Param::X)
                            .offset(first_row_in_tile * self.params.get::<u32>(Param::RowStride)),
                        inner_len: self.params.get(Param::ActualRowSize),
                        outer_len: (self
                            .params
                            .get::<u32>(Param::NumRows)
                            .saturating_sub(first_row_in_tile))
                        .max(self.tuning_config.rows_per_tile),
                        outer_stride: self.params.get(Param::RowStride),
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
        let scale = self.config.has_scale.then(|| {
            self.layout.as_1d().load_aligned_or_zero(
                View1D::<bf16, Global> {
                    base: self.params.get(Param::Scale),
                    len: self.params.get(Param::ActualRowSize),
                },
                thread_idx,
            )
        });
        let bias = self.config.has_scale.then(|| {
            self.layout.as_1d().load_aligned_or_zero(
                View1D::<bf16, Global> {
                    base: self.params.get(Param::Bias),
                    len: self.params.get(Param::ActualRowSize),
                },
                thread_idx,
            )
        });

        let i = Var::<u32>::new(0);

        loop_(|l| {
            for_range(
                current_schedule_unit * self.tuning_config.tiles_per_schedule_unit,
                (current_schedule_unit + 1) * self.tuning_config.tiles_per_schedule_unit,
                |tile| {
                    let first_row_in_tile = tile * self.tuning_config.rows_per_tile;
                    let stage = self.load_pipeline.consumer_wait(pipeline_state);
                    let reduction_slot_idx = i % 2;

                    let x = self.layout.load_aligned_unchecked(
                        View2D {
                            base: self.load_slots_base.offset(
                                size_of::<bf16>() as u32
                                    * stage.index()
                                    * self.config.max_row_size as u32,
                            ),
                            inner_len: Val::new(self.config.max_row_size as u32),
                            outer_len: Val::new(self.tuning_config.rows_per_tile),
                            outer_stride: Val::new(self.config.max_row_size as u32),
                        },
                        thread_idx,
                    );

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
                                            + warp_id(),
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

                            let r_inv = (global_sums_of_squares + 1e-6f32).rsqrt();

                            // Store r_inv
                            if_(thread_idx.eq(0), || {
                                for row_idx in 0..self.tuning_config.rows_per_tile {
                                    let row_r_inv = r_inv.get(row_idx as usize);
                                    let coord = first_row_in_tile + row_idx;
                                    if_(coord.lt(self.params.get(Param::NumRows)), || {
                                        let dst = self.params.get::<Ptr<f32, Global>>(Param::RInv);
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
                        .params
                        .get::<u32>(Param::NumRows)
                        .saturating_sub(first_row_in_tile))
                    .max(self.tuning_config.rows_per_tile);

                    match self.config.output_types {
                        OutputTypes::Bf16 => {
                            self.layout.store_aligned(
                                View2D::<_, Global> {
                                    base: self.params.get(Param::YBf16).offset(
                                        first_row_in_tile
                                            * self.params.get::<u32>(Param::RowStride),
                                    ),
                                    inner_len: self.params.get(Param::ActualRowSize),
                                    outer_len: valid_rows_in_tile,
                                    outer_stride: self.params.get(Param::RowStride),
                                },
                                &y,
                                thread_idx,
                            );
                        }
                        OutputTypes::Mxfp8Rowwise | OutputTypes::Mxfp8RowwiseAndColwise => {
                            self.layout
                                .store_aligned_rowwise_block_scaled::<F8E4M3, UE8M0>(
                                    BlockScaledFormat::Mxfp8,
                                    View2D::<_, Global> {
                                        base: self.params.get(Param::YMxfp8Row).offset(
                                            first_row_in_tile
                                                * self.params.get::<u32>(Param::RowStride),
                                        ),
                                        inner_len: self.params.get(Param::ActualRowSize),
                                        outer_len: valid_rows_in_tile,
                                        outer_stride: self.params.get(Param::RowStride),
                                    },
                                    ViewPosWithinTensor2D {
                                        row_offset: first_row_in_tile,
                                        col_offset: Val::new(0),
                                        total_tensor_rows: self.params.get(Param::NumRows),
                                        total_tensor_cols: self.params.get(Param::ActualRowSize),
                                    },
                                    self.params.get(Param::YMxfp8RowSf),
                                    &y,
                                    thread_idx,
                                );

                            if self.config.output_types == OutputTypes::Mxfp8RowwiseAndColwise {
                                let (resharded_layout, resharded_vals) =
                                    match resharding_in_progress {
                                        Some(reshard) => reshard.finish_resharding(),
                                        None => (self.layout, y),
                                    };

                                todo!()
                            }
                        }
                        OutputTypes::Nvfp4Rowwise => todo!(),
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

impl TuningConfig {
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

            for rows_per_tile in [1usize, 2, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32, 48, 64] {
                let cluster_size = match config.output_types {
                    OutputTypes::Mxfp8RowwiseAndColwise => {
                        if !rows_per_tile.is_power_of_two() {
                            continue;
                        }
                        BlockScaledFormat::Mxfp8.block_size() / rows_per_tile
                    }
                    _ => 1,
                };
                if cluster_size > max_cluster_size(arch) {
                    continue;
                }

                let regs = apply_reg_overhead(rows_per_tile * reg_bytes_per_row + reg_aux_bytes);
                if regs > reg_capacity {
                    continue;
                }

                let reshard_buffer_bytes = if cluster_size > 1 {
                    staging_slot_bytes_per_row
                } else {
                    0
                };

                for num_smem_staging_slots in 1..=4 {
                    let smem =
                        staging_slot_bytes_per_row * num_smem_staging_slots + reshard_buffer_bytes;
                    if smem > smem_capacity {
                        continue;
                    }

                    let config = TuningConfig {
                        occupancy: occupancy as u32,
                        num_worker_warps: 8 / occupancy as u32,
                        num_worker_threads: 8 / occupancy as u32 * 32,
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
    cubin: Vec<u8>,
}

impl HasCubin for BuiltKernel {
    fn cubin(&self) -> &[u8] {
        &self.cubin
    }
}
