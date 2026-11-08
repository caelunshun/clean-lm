//! CLI to emit SASS for a generated kernel, for introspection purposes.

use clap::{Parser, Subcommand};
use kernel_generator::{
    Arch,
    kernels::{
        HasCubin, KernelGenerator,
        norm::{self, forward::NormForwardKernelGenerator},
    },
};
use nvvm_builder::tools::sass;
use std::path::PathBuf;

#[derive(Debug, Parser)]
pub struct Args {
    #[clap(subcommand)]
    pub kernel: Kernel,
    #[clap(long)]
    pub arch: CliArch,
    #[clap(long, short)]
    pub output: Option<PathBuf>,
}

#[derive(Debug, Clone, clap::ValueEnum)]
pub enum CliArch {
    Sm100a,
    Sm100f,
    Sm103a,
    Sm103f,
    Sm110a,
    Sm110f,
    Sm120a,
    Sm120f,
}

impl From<CliArch> for Arch {
    fn from(value: CliArch) -> Self {
        match value {
            CliArch::Sm100a => Arch::Compute100a,
            CliArch::Sm100f => Arch::Compute100f,
            CliArch::Sm103a => Arch::Compute103a,
            CliArch::Sm103f => Arch::Compute103f,
            CliArch::Sm110a => Arch::Compute110a,
            CliArch::Sm110f => Arch::Compute110f,
            CliArch::Sm120a => Arch::Compute120a,
            CliArch::Sm120f => Arch::Compute120f,
        }
    }
}

#[derive(Debug, Subcommand)]
pub enum Kernel {
    NormFwd(norm::Config),
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    let cubin = match args.kernel {
        Kernel::NormFwd(config) => NormForwardKernelGenerator
            .generate_kernel(config, args.arch.into())?
            .cubin()
            .to_owned(),
    };

    let sass = sass(&cubin)?;

    match args.output {
        Some(path) => fs_err::write(path, sass.as_bytes())?,
        None => println!("{sass}"),
    }

    Ok(())
}
