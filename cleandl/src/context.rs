use std::{
    any::{Any, TypeId},
    cell::RefCell,
    collections::{HashMap, hash_map::Entry},
    sync::{Arc, Mutex},
};

use parking_lot::ReentrantMutex;
use serde::Serialize;
use skycuda::{
    device::Device,
    device_memory_pool::{DeviceMemoryPool, DeviceMemoryPoolOptions},
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
        let auxiliary_stream = cuda.create_stream()?;

        let device_memory_pool = cuda.create_device_memory_pool(device_memory_options)?;
        let host_pinned_memory_pool =
            cuda.create_host_pinned_memory_pool(host_pinned_memory_options)?;

        Ok(Self(Arc::new(Inner {
            cuda,
            default_stream: auxiliary_stream,
            resource_cache: ReentrantMutex::new(RefCell::new(HashMap::new())),
            default_cudnn_contexts: ThreadLocal::new(),
            stream_cudnn_contexts: PerStreamResource::new(),
            device_memory_pool,
            host_pinned_memory_pool,
        })))
    }

    pub fn cuda(&self) -> &Device {
        &self.0.cuda
    }

    pub fn default_stream(&self) -> &Stream {
        &self.0.default_stream
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

    /// cuDNN handle bound to the default stream, for graph construction.
    pub fn cudnn(&self) -> anyhow::Result<&CudnnHandle> {
        self.0
            .default_cudnn_contexts
            .get_or_try(|| CudnnHandle::new(&self.0.default_stream))
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
    default_stream: Stream,
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
