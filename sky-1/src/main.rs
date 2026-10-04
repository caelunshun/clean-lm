use std::env;

use clap::{Parser, Subcommand};
use cleandl::context::Context;
use mimalloc::MiMalloc;
use sky_1::skyred;
use skycuda::{
    device_memory_pool::DeviceMemoryPoolOptions,
    host_pinned_memory_pool::HostPinnedMemoryPoolOptions,
};

#[global_allocator]
static ALLOCATOR: MiMalloc = MiMalloc;

#[derive(Debug, Parser)]
struct Cli {
    #[clap(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Skyred(skyred::cli::Cli),
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    match Cli::parse().command {
        Command::Skyred(cli) => {
            skyred::cli::run(cli)?;
        }
    }

    Ok(())
}

#[allow(unused)]
fn init_cx() -> anyhow::Result<Context> {
    Context::new(
        0,
        DeviceMemoryPoolOptions {
            max_reserved_free_bytes: 1024 << 30,
        },
        HostPinnedMemoryPoolOptions {
            max_reserved_free_bytes: 256 << 30,
        },
        env::var("CLEANDL_AUTOTUNE").is_ok(),
    )
}
