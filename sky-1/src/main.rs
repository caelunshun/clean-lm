use std::env;

use cleandl::context::Context;
use mimalloc::MiMalloc;
use skycuda::{
    device_memory_pool::DeviceMemoryPoolOptions,
    host_pinned_memory_pool::HostPinnedMemoryPoolOptions,
};

#[global_allocator]
static ALLOCATOR: MiMalloc = MiMalloc;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let cx = &Context::new(
        0,
        DeviceMemoryPoolOptions {
            max_reserved_free_bytes: 1024 << 30,
        },
        HostPinnedMemoryPoolOptions {
            max_reserved_free_bytes: 256 << 30,
        },
        env::var("CLEANDL_AUTOTUNE").is_ok(),
    )?;
    let stream = cx.default_compute_stream();

    Ok(())
}
