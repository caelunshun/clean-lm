use std::{
    any::{Any, TypeId},
    cell::RefCell,
    collections::{HashMap, hash_map::Entry},
    sync::{Arc, Mutex},
};

use bytemuck::Pod;
use parking_lot::ReentrantMutex;
use serde::Serialize;
use skycuda::{
    device::Device,
    device_memory::{DeviceVisibleMemory, DeviceVisibleMemoryMut},
    device_memory_pool::{DeviceMemoryPool, DeviceMemoryPoolOptions},
    host_pinned_memory::HostPinnedMemory,
    host_pinned_memory_pool::{HostPinnedMemoryPool, HostPinnedMemoryPoolOptions},
    stream::Stream,
};
use skycuda_cudnn::CudnnHandle;
use thread_local::ThreadLocal;

#[derive(Debug, Clone)]
pub struct Context(Arc<Inner>);

impl Context {
    pub fn new(
        device_index: u32,
        device_memory_options: DeviceMemoryPoolOptions,
        host_pinned_memory_options: HostPinnedMemoryPoolOptions,
    ) -> anyhow::Result<Self> {
        let cuda = Device::from_index(device_index)?;

        let device_memory_pool = cuda.create_device_memory_pool(device_memory_options)?;
        let host_pinned_memory_pool =
            cuda.create_host_pinned_memory_pool(host_pinned_memory_options)?;

        Ok(Self(Arc::new(Inner {
            default_compute_stream: cuda.create_stream()?,
            h2d_copy_stream: cuda.create_stream()?,
            d2h_copy_stream: cuda.create_stream()?,
            memset_stream: cuda.create_stream()?,
            cuda,
            resource_cache: ReentrantMutex::new(RefCell::new(HashMap::new())),
            default_cudnn_contexts: ThreadLocal::new(),
            stream_cudnn_contexts: PerStreamResource::new(),
            device_memory_pool,
            host_pinned_memory_pool,
        })))
    }

    pub fn upload_to_device<T: Pod>(
        &self,
        src: &[T],
        dst: &mut impl DeviceVisibleMemoryMut,
        stream: &Stream,
    ) -> anyhow::Result<()> {
        let mut staging = self
            .host_pinned_memory_pool()
            .alloc_zeroed(std::mem::size_of_val(src))?;
        staging
            .view_mut()?
            .copy_from_slice(bytemuck::cast_slice(src));
        stream.copy(&staging, dst)?;
        Ok(())
    }

    pub fn download_from_device(
        &self,
        src: &impl DeviceVisibleMemory,
        stream: &Stream,
    ) -> anyhow::Result<HostPinnedMemory> {
        let mut staging = self.host_pinned_memory_pool().alloc_zeroed(src.len())?;
        stream.copy(src, &mut staging)?;
        Ok(staging)
    }

    pub fn cuda(&self) -> &Device {
        &self.0.cuda
    }

    pub fn default_compute_stream(&self) -> &Stream {
        &self.0.default_compute_stream
    }

    pub fn h2d_copy_stream(&self) -> &Stream {
        &self.0.h2d_copy_stream
    }

    pub fn d2h_copy_stream(&self) -> &Stream {
        &self.0.d2h_copy_stream
    }

    pub fn memset_stream(&self) -> &Stream {
        &self.0.memset_stream
    }

    pub fn device_memory_pool(&self) -> &DeviceMemoryPool {
        &self.0.device_memory_pool
    }

    pub fn host_pinned_memory_pool(&self) -> &HostPinnedMemoryPool {
        &self.0.host_pinned_memory_pool
    }

    pub fn get_or_create_resource<K: ResourceKey>(
        &self,
        key: K,
    ) -> anyhow::Result<Arc<K::Resource>> {
        let cache = self.0.resource_cache.lock();

        let encoded_key = (TypeId::of::<K>(), postcard::to_allocvec(&key)?);

        let cache_guard = cache.borrow();
        if let Some(resource) = cache_guard.get(&encoded_key) {
            return Ok(Arc::downcast(resource.clone()).ok().unwrap());
        }

        // Allow reentrant get_or_create_resource calls in creation callback
        drop(cache_guard);
        let resource = key.create(self)?;
        let resource = Arc::new(resource);

        cache.borrow_mut().insert(encoded_key, resource.clone());
        Ok(resource)
    }

    /// cuDNN handle bound to the default compute stream, for graph construction.
    pub fn cudnn(&self) -> anyhow::Result<&CudnnHandle> {
        self.0
            .default_cudnn_contexts
            .get_or_try(|| CudnnHandle::new(&self.0.default_compute_stream))
            .map_err(From::from)
    }

    /// cuDNN handle bound to the given stream, for graph execution.
    pub fn cudnn_on(&self, stream: &Stream) -> anyhow::Result<Arc<CudnnHandle>> {
        self.0
            .stream_cudnn_contexts
            .get_or_create(stream, || CudnnHandle::new(stream).map_err(From::from))
    }
}

type Resource = Arc<dyn Any + Send + Sync>;
type ResourceKeyEncoded = (TypeId, Vec<u8>);

#[derive(Debug)]
struct Inner {
    cuda: Device,
    device_memory_pool: DeviceMemoryPool,
    host_pinned_memory_pool: HostPinnedMemoryPool,
    default_compute_stream: Stream,
    h2d_copy_stream: Stream,
    d2h_copy_stream: Stream,
    memset_stream: Stream,
    default_cudnn_contexts: ThreadLocal<CudnnHandle>,
    stream_cudnn_contexts: PerStreamResource<CudnnHandle>,
    resource_cache: ReentrantMutex<RefCell<HashMap<ResourceKeyEncoded, Resource>>>,
}

pub trait ResourceKey: Serialize + 'static {
    type Resource: Send + Sync + 'static;

    fn create(&self, cx: &Context) -> anyhow::Result<Self::Resource>;
}

#[derive(Debug)]
pub struct PerStreamResource<T> {
    by_stream: Mutex<HashMap<u64, Arc<T>>>,
}

impl<T> PerStreamResource<T> {
    pub fn new() -> Self {
        Self {
            by_stream: Default::default(),
        }
    }

    pub fn get_or_create(
        &self,
        stream: &Stream,
        create: impl FnOnce() -> anyhow::Result<T>,
    ) -> anyhow::Result<Arc<T>> {
        let mut map = self.by_stream.lock().unwrap();
        match map.entry(stream.id()) {
            Entry::Occupied(e) => Ok(e.get().clone()),
            Entry::Vacant(e) => {
                let resource = Arc::new(create()?);
                e.insert(resource.clone());
                Ok(resource)
            }
        }
    }
}

impl<T> Default for PerStreamResource<T> {
    fn default() -> Self {
        Self::new()
    }
}
