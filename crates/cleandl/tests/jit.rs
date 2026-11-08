mod common;

use common::test_context;
use kernel_build_util::{
    ArgBundle, Param, PtrArg,
    tileir::{ArrayArg, KernelBuilder},
};
use serde::Serialize;
use std::sync::Arc;
use tileir_builder::{Array, ArrayOpts, KernelOpts, bid};

const TILE: u32 = 128;

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

#[derive(Serialize)]
struct VecAddKey {
    tile: u32,
}

#[test]
fn vecadd_compiles_launches_and_caches() -> anyhow::Result<()> {
    let cx = test_context();
    let stream = cx.default_compute_stream();
    let n = TILE as usize * 37;

    let compile = || {
        cleandl::execgraph::jit::get_or_compile_kernel(
            &cx,
            &VecAddKey { tile: TILE },
            "test_vecadd",
            |module| {
                let mut kb = KernelBuilder::new();
                let p = declare(&mut kb);
                kb.define(module, "test_vecadd", KernelOpts::new(), |args| {
                    let pid = bid(0);
                    let a = args.get_array(p.a).load([pid], [TILE]).downcast::<f32>();
                    let b = args.get_array(p.b).load([pid], [TILE]).downcast::<f32>();
                    args.get_array(p.c).store([pid], a + b);
                })
            },
        )
    };
    let kernel = compile()?;
    let p = declare(&mut KernelBuilder::new());

    let a_host: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b_host: Vec<f32> = (0..n).map(|i| 2.0 * i as f32 + 0.5).collect();
    let mut a = cx.device_memory_pool().alloc_zeroed_on(n * 4, stream)?;
    let mut b = cx.device_memory_pool().alloc_zeroed_on(n * 4, stream)?;
    let mut c = cx.device_memory_pool().alloc_zeroed_on(n * 4, stream)?;
    cx.upload_to_device(&a_host, &mut a, stream)?;
    cx.upload_to_device(&b_host, &mut b, stream)?;

    let shape = [n as i64];
    let strides = [1i64];
    let mut args = ArgBundle::new();
    args.set(p.a, ArrayArg::new(PtrArg::new(&a), &shape, &strides));
    args.set(p.b, ArrayArg::new(PtrArg::new(&b), &shape, &strides));
    args.set(
        p.c,
        ArrayArg::new(PtrArg::new_mut(&mut c), &shape, &strides),
    );
    unsafe { kernel.launch(args, ((n / TILE as usize) as u32, 1, 1), stream)? };

    let c_host = cx.download_from_device(&c, stream)?;
    stream.block_on()?;
    let c_host: &[f32] = bytemuck::cast_slice(c_host.view()?);
    for i in 0..n {
        assert_eq!(c_host[i], a_host[i] + b_host[i], "mismatch at {i}");
    }

    let again = compile()?;
    assert!(Arc::ptr_eq(&kernel, &again));
    Ok(())
}
