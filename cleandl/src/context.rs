use std::{
    any::{Any, TypeId},
    collections::{HashMap, hash_map::Entry},
    sync::{Arc, Mutex},
};

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
            resource_cache: Mutex::new(HashMap::new()),
            default_cudnn_contexts: ThreadLocal::new(),
        })))
    }

    pub fn cuda(&self) -> &CudaContext {
        &self.0.cuda
    }

    pub fn get_or_create_resource<K: ResourceKey>(
        &self,
        key: K,
        create: impl FnOnce() -> anyhow::Result<K::Resource>,
    ) -> anyhow::Result<Arc<K::Resource>> {
        let mut cache = self.0.resource_cache.lock().unwrap();
        let key = (TypeId::of::<K>(), postcard::to_allocvec(&key)?);
        match cache.entry(key) {
            Entry::Occupied(e) => Ok(Arc::downcast(e.get().clone()).unwrap()),
            Entry::Vacant(e) => {
                let resource = Arc::new(create()?);
                e.insert(resource.clone());
                Ok(resource)
            }
        }
    }

    pub fn cudnn(&self) -> anyhow::Result<&CudnnContext> {
        self.0
            .default_cudnn_contexts
            .get_or_try(|| CudnnContext::create(self.cuda()))
            .map_err(From::from)
    }
}

#[derive(Debug)]
struct Inner {
    cuda: CudaContext,
    default_cudnn_contexts: ThreadLocal<CudnnContext>,
    resource_cache: Mutex<HashMap<(TypeId, Vec<u8>), Arc<dyn Any + Send + Sync>>>,
}

pub trait ResourceKey: Serialize + 'static {
    type Resource: Send + Sync + 'static;
}

#[derive(Debug, Clone)]
pub struct CudaStream {
    stream: Stream,
    cuda: CudaContext,
    cudnn_cx: Arc<ThreadLocal<CudnnContext>>,
}

impl CudaStream {
    pub fn new(cx: &Context) -> anyhow::Result<Self> {
        Ok(Self {
            stream: cx
                .cuda()
                .create_stream_with_flags(StreamFlags::NON_BLOCKING)?,
            cuda: cx.cuda().clone(),
            cudnn_cx: Arc::new(ThreadLocal::new()),
        })
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
