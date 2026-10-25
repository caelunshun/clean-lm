use crate::{
    error::{Error, Result},
    kernels::{
        BuiltKernel, KernelGenerator,
        norm::{MAX_SUPPORTED_ROW_SIZE, OutputTypes},
    },
    utils::{
        arch::{max_cluster_size, max_smem, reg_bytes_per_sm},
        block_scale::BlockScaledFormat,
        params::{DynParamsBuilder, dtype},
    },
};
use nvvm_builder::{mbarrier::Pipeline, prelude::*};
use skycuda::stream::Stream;
use std::cmp::Ordering;

pub struct NormForwardKernelGenerator;

impl KernelGenerator for NormForwardKernelGenerator {
    type Config = super::Config;
    type ExecArgs<'a> = ();

    fn generate_kernel(&self, mut config: Self::Config, arch: Arch) -> Result<BuiltKernel<Self>> {
        config.max_row_size = config.max_row_size.next_power_of_two();

        if config.max_row_size > MAX_SUPPORTED_ROW_SIZE {
            return Err(Error::Unsupported("row size > 8192"));
        }

        let mut module = ModuleBuilder::new("norm_fwd");

        let mut params = DynParamsBuilder::new();
        params.add(Param::X, dtype::<Ptr<bf16, Global>>());
        params.add(Param::ActualRowSize, dtype::<u32>());
        params.add(Param::RecipActualRowSize, dtype::<f32>());
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

        let tuning_params = TuningParams::decide(&config, &arch)?;

        module.kernel_dyn(
            "norm_fwd",
            KernelOpts::new()
                .cluster_dims([tuning_params.cluster_size, 1, 1])
                .req_threads([128 + tuning_params.num_worker_threads, 1, 1])
                .restrict_pointers(true)
                .min_blocks_per_sm(tuning_params.occupancy)
                .max_regs(
                    reg_bytes_per_sm(&arch) as u32
                        / tuning_params.occupancy
                        / size_of::<u32>() as u32
                        / (128 + tuning_params.num_worker_threads),
                ),
            &params.param_dtypes(),
            |args| {
                params.bind(args);
            },
        );

        todo!()
    }

    unsafe fn execute_kernel<'a>(
        &self,
        kernel: &BuiltKernel<Self>,
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
}

struct Builder {
    config: super::Config,
    tuning_params: TuningParams,
    load_pipeline: Pipeline,
    load_slots: Vec<Val<Ptr<bf16, Shared>>>,
}

impl Builder {}

struct TuningParams {
    occupancy: u32,
    num_worker_warps: u32,
    num_worker_threads: u32,
    num_smem_staging_slots: u32,
    cluster_size: u32,
    rows_per_tile: u32,
    tiles_per_schedule_unit: u32,
}

impl TuningParams {
    pub fn decide(config: &super::Config, arch: &Arch) -> Result<Self> {
        let reg_capacity_per_sm = reg_bytes_per_sm(arch);
        let smem_capacity_per_sm = max_smem(arch);

        let reg_bytes_per_row = config.max_row_size * size_of::<bf16>();
        let reg_aux_bytes = (config.has_scale as usize + config.has_bias as usize)
            * config.max_row_size
            * size_of::<bf16>();

        let staging_slot_bytes_per_row = reg_bytes_per_row;

        fn cmp(first: &TuningParams, second: &TuningParams) -> Ordering {
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

        let mut best = None::<TuningParams>;

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

                    let config = TuningParams {
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
