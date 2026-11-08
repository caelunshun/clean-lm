mod common;

use cleandl::{context::Context, execgraph::jit::get_or_compile_kernel};
use common::{test_context, test_context_in};
use kernel_build_util::{
    ArgBundle, LoadedKernel, Param, PtrArg,
    tileir::{ArrayArg, KernelBuilder},
};
use rayon::prelude::*;
use serde::Serialize;
use std::sync::Arc;
use tileir_builder::{Array, ArrayOpts, KernelOpts, bid};

struct Params {
    a: Param<Array<f32>>,
    b: Param<Array<f32>>,
    c: Param<Array<f32>>,
}

/// Declares the kernel parameters; called both when defining and when binding arguments.
fn declare(kb: &mut KernelBuilder) -> Params {
    Params {
        a: kb.array::<f32>("a", ArrayOpts::rank(1)),
        b: kb.array::<f32>("b", ArrayOpts::rank(1)),
        c: kb.array::<f32>("c", ArrayOpts::rank(1)),
    }
}

/// Computes `c = a * scale + b`.
#[derive(Serialize, Clone, Copy, Debug)]
struct AxpyKey {
    tile: u32,
    scale: f32,
}

fn axpy_kernel(cx: &Context, key: AxpyKey) -> anyhow::Result<Arc<LoadedKernel>> {
    get_or_compile_kernel(cx, &key, "test_axpy", |module| {
        let mut kb = KernelBuilder::new();
        let p = declare(&mut kb);
        kb.define(module, "test_axpy", KernelOpts::new(), |args| {
            let pid = bid(0);
            let a = args
                .get_array(p.a)
                .load([pid], [key.tile])
                .downcast::<f32>();
            let b = args
                .get_array(p.b)
                .load([pid], [key.tile])
                .downcast::<f32>();
            args.get_array(p.c).store([pid], a * key.scale + b);
        })
    })
}

fn check_axpy(cx: &Context, kernel: &LoadedKernel, key: AxpyKey) -> anyhow::Result<()> {
    let stream = cx.default_compute_stream();
    let n = key.tile as usize * 37;

    let a_host: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b_host: Vec<f32> = (0..n).map(|i| 2.0 * i as f32 + 0.5).collect();
    let mut a = cx.device_memory_pool().alloc_zeroed_on(n * 4, stream)?;
    let mut b = cx.device_memory_pool().alloc_zeroed_on(n * 4, stream)?;
    let mut c = cx.device_memory_pool().alloc_zeroed_on(n * 4, stream)?;
    cx.upload_to_device(&a_host, &mut a, stream)?;
    cx.upload_to_device(&b_host, &mut b, stream)?;

    let p = declare(&mut KernelBuilder::new());
    let shape = [n as i64];
    let strides = [1i64];
    let mut args = ArgBundle::new();
    args.set(p.a, ArrayArg::new(PtrArg::new(&a), &shape, &strides));
    args.set(p.b, ArrayArg::new(PtrArg::new(&b), &shape, &strides));
    args.set(
        p.c,
        ArrayArg::new(PtrArg::new_mut(&mut c), &shape, &strides),
    );
    unsafe { kernel.launch(args, ((n / key.tile as usize) as u32, 1, 1), stream)? };

    let c_host = cx.download_from_device(&c, stream)?;
    stream.block_on()?;
    let c_host: &[f32] = bytemuck::cast_slice(c_host.view()?);
    for i in 0..n {
        assert_eq!(
            c_host[i],
            a_host[i] * key.scale + b_host[i],
            "{key:?}: mismatch at {i}"
        );
    }
    Ok(())
}

#[test]
fn compiles_launches_and_caches() -> anyhow::Result<()> {
    let cx = test_context();
    let key = AxpyKey {
        tile: 128,
        scale: 1.0,
    };
    let kernel = axpy_kernel(&cx, key)?;
    check_axpy(&cx, &kernel, key)?;
    assert!(Arc::ptr_eq(&kernel, &axpy_kernel(&cx, key)?));
    Ok(())
}

#[test]
fn fresh_context_loads_from_disk() -> anyhow::Result<()> {
    let dir = tempfile::tempdir_in(env!("CARGO_TARGET_TMPDIR"))?;
    let cache_file = dir.path().join("cache.json");
    let key = AxpyKey {
        tile: 64,
        scale: 3.0,
    };

    let cx = test_context_in(dir.path());
    axpy_kernel(&cx, key)?;
    let cached = std::fs::read(&cache_file)?;
    assert!(!cached.is_empty());

    // A cache hit must not write to the file.
    let cx2 = test_context_in(dir.path());
    let kernel = axpy_kernel(&cx2, key)?;
    assert_eq!(std::fs::read(&cache_file)?, cached);
    check_axpy(&cx2, &kernel, key)?;
    Ok(())
}

#[test]
fn concurrent_compiles() -> anyhow::Result<()> {
    let cx = test_context();
    let keys: Vec<AxpyKey> = [64, 128]
        .into_iter()
        .flat_map(|tile| [0.5, 2.0, -1.0].map(|scale| AxpyKey { tile, scale }))
        .collect();
    // Each key requested several times concurrently.
    let requests: Vec<usize> = (0..4).flat_map(|_| 0..keys.len()).collect();
    let kernels: Vec<(usize, Arc<LoadedKernel>)> = requests
        .par_iter()
        .map(|&i| Ok((i, axpy_kernel(&cx, keys[i])?)))
        .collect::<anyhow::Result<_>>()?;

    for (i, key) in keys.iter().enumerate() {
        let canonical = axpy_kernel(&cx, *key)?;
        for (j, kernel) in &kernels {
            assert_eq!(*j == i, Arc::ptr_eq(kernel, &canonical), "{key:?}");
        }
        check_axpy(&cx, &canonical, *key)?;
    }
    Ok(())
}
