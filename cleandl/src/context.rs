use std::{
    any::{Any, TypeId},
    cell::RefCell,
    collections::{HashMap, hash_map::Entry},
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

use parking_lot::ReentrantMutex;
use serde::Serialize;
use singe_cuda::{
    context::ContextFlags,
    device::Device,
    stream::{Stream, StreamFlags},
};
use thread_local::ThreadLocal;

pub type CudaContext = Arc<singe_cuda::context::Context>;
pub type CudnnContext = singe_cudnn::context::Context;

/// Context for a CUDA device.
#[derive(Debug, Clone)]
pub struct Context(Arc<Inner>);

impl Context {
    pub fn new(device_index: i32) -> anyhow::Result<Self> {
        let device = Device::new(device_index);
        let cuda = singe_cuda::context::Context::create_for_device_with_flags(
            device,
            ContextFlags::empty(),
        )?;

        Ok(Self(Arc::new(Inner {
            cuda,
            resource_cache: ReentrantMutex::new(RefCell::new(HashMap::new())),
            default_cudnn_contexts: ThreadLocal::new(),
        })))
    }

    pub fn cuda(&self) -> &CudaContext {
        &self.0.cuda
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

    pub fn cudnn(&self) -> anyhow::Result<&CudnnContext> {
        self.0
            .default_cudnn_contexts
            .get_or_try(|| CudnnContext::create(self.cuda()))
            .map_err(From::from)
    }
}

type Resource = Arc<dyn Any + Send + Sync>;
type ResourceKeyEncoded = (TypeId, Vec<u8>);

#[derive(Debug)]
struct Inner {
    cuda: CudaContext,
    default_cudnn_contexts: ThreadLocal<CudnnContext>,
    resource_cache: ReentrantMutex<RefCell<HashMap<ResourceKeyEncoded, Resource>>>,
}

pub trait ResourceKey: Serialize + 'static {
    type Resource: Send + Sync + 'static;

    fn create(&self, cx: &Context) -> anyhow::Result<Self::Resource>;
}

#[derive(Debug, Clone)]
pub struct CudaStream {
    id: u64,
    stream: Stream,
    cuda: CudaContext,
    cudnn_cx: Arc<ThreadLocal<CudnnContext>>,
}

impl CudaStream {
    pub fn new(cx: &Context) -> anyhow::Result<Self> {
        static NEXT_ID: AtomicU64 = AtomicU64::new(0);
        Ok(Self {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            stream: cx
                .cuda()
                .create_stream_with_flags(StreamFlags::NON_BLOCKING)?,
            cuda: cx.cuda().clone(),
            cudnn_cx: Arc::new(ThreadLocal::new()),
        })
    }

    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn cuda(&self) -> &CudaContext {
        &self.cuda
    }

    pub fn get(&self) -> &Stream {
        &self.stream
    }

    pub fn cudnn_cx(&self) -> anyhow::Result<&CudnnContext> {
        self.cudnn_cx
            .get_or_try(|| {
                CudnnContext::create(&self.cuda).and_then(|cx| {
                    cx.set_stream(Some(&self.stream))?;
                    Ok(cx)
                })
            })
            .map_err(From::from)
    }
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
        stream: &CudaStream,
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
