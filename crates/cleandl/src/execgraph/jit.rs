use crate::{context::Context, disk_cache::DiskCacheKey};
use kernel_build_util::{Kernel, LoadedKernel, tileir::KernelSpec};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use skycuda::device::Device;
use std::{
    collections::HashMap,
    process::Command,
    sync::{Arc, OnceLock},
};
use tileir_builder::{ModuleBuilder, TileirasOptions};

/// Bump to invalidate all cached JIT kernels.
const FORMAT_VERSION: u32 = 1;

/// Compiles (or fetches from the disk cache) a TileIR kernel, loading it once per context.
///
/// `name` and `key` (together with the key's type) must uniquely identify the kernel
/// body within a context: kernels are loaded once per such identity, and `define` is
/// not called again for it.
/// `define` must define exactly one kernel in the module, via `KernelBuilder::define`.
/// It is traced on the first request per context; the disk cache is keyed on the
/// resulting bytecode, so changes to the kernel body never hit stale cubins.
pub fn get_or_compile_kernel<K: Serialize>(
    cx: &Context,
    key: &K,
    name: &str,
    define: impl FnOnce(&mut ModuleBuilder) -> KernelSpec,
) -> anyhow::Result<Arc<LoadedKernel>> {
    get_or_compile_kernel_with_options(cx, key, name, TileirasOptions::new(), define)
}

/// Like [`get_or_compile_kernel`], with extra `tileiras` options. The target arch is
/// always set from the context's device.
pub fn get_or_compile_kernel_with_options<K: Serialize>(
    cx: &Context,
    key: &K,
    name: &str,
    options: TileirasOptions,
    define: impl FnOnce(&mut ModuleBuilder) -> KernelSpec,
) -> anyhow::Result<Arc<LoadedKernel>> {
    let options = options.arch(tileiras_arch(cx.cuda()));
    let config = serde_json::to_value(key)?;
    let tileiras_args = options.to_args();

    let local_key =
        serde_json::to_string(&(name, std::any::type_name::<K>(), &config, &tileiras_args))?;
    if let Some(kernel) = cx.jit_kernels().get(&local_key) {
        return Ok(kernel);
    }

    let mut module = ModuleBuilder::new(name);
    let spec = define(&mut module);
    let built = module.finish()?;
    let bytecode = built.bytecode()?;

    let disk_key = JitKernelKey {
        format_version: FORMAT_VERSION,
        name: name.to_owned(),
        config,
        tileiras_args,
        tileiras_version: tileiras_version().to_owned(),
        bytecode_hash: blake3::hash(&bytecode).to_hex().to_string(),
    };
    let kernel: Kernel = cx.disk_cache().get_or_create(&disk_key, || {
        tracing::debug!("compiling TileIR kernel {name}");
        let cubin = built.compile(&options)?;
        if !cubin.log.is_empty() {
            tracing::debug!("tileiras output for {name}:\n{}", cubin.log);
        }
        Ok(spec.finish(&built, cubin)?)
    })?;

    // Soundness: the cubin was produced by tileiras, possibly via our own disk cache.
    let loaded = Arc::new(unsafe { kernel.load(cx.cuda())? });
    Ok(cx.jit_kernels().insert(local_key, loaded))
}

/// The `tileiras` target name for a device, e.g. `sm_89` or `sm_100`.
pub fn tileiras_arch(device: &Device) -> String {
    format!(
        "sm_{}{}",
        device.compute_capability_major(),
        device.compute_capability_minor()
    )
}

/// The `tileiras` release string, e.g. `V13.5.37`, or `unknown` if it can't be determined.
pub fn tileiras_version() -> &'static str {
    static VERSION: OnceLock<String> = OnceLock::new();
    VERSION.get_or_init(|| {
        let output = tileir::tools::Tileiras::locate()
            .ok()
            .and_then(|tileiras| Command::new(tileiras.path()).arg("--version").output().ok());
        output
            .and_then(|output| {
                String::from_utf8_lossy(&output.stdout)
                    .lines()
                    .find(|line| line.contains("compilation tools"))
                    .and_then(|line| line.rsplit(", ").next())
                    .map(|version| version.trim().to_owned())
            })
            .unwrap_or_else(|| "unknown".to_owned())
    })
}

#[derive(Debug, Serialize, Deserialize)]
struct JitKernelKey {
    format_version: u32,
    name: String,
    config: serde_json::Value,
    tileiras_args: Vec<String>,
    tileiras_version: String,
    bytecode_hash: String,
}

#[typetag::serde]
impl DiskCacheKey for JitKernelKey {}

/// Kernels loaded on a context, by name, config and compile options.
#[derive(Debug, Default)]
pub(crate) struct JitKernelCache {
    kernels: Mutex<HashMap<String, Arc<LoadedKernel>>>,
}

impl JitKernelCache {
    fn get(&self, key: &str) -> Option<Arc<LoadedKernel>> {
        self.kernels.lock().get(key).cloned()
    }

    /// Inserts `kernel` unless another thread got there first, returning the cached kernel.
    fn insert(&self, key: String, kernel: Arc<LoadedKernel>) -> Arc<LoadedKernel> {
        self.kernels.lock().entry(key).or_insert(kernel).clone()
    }
}
