use std::{
    cell::RefCell,
    collections::{HashMap, hash_map::Entry},
    ptr,
    sync::Arc,
};

use singe_cuda::{context::Context, memory::DeviceMemory, stream::Stream};
use singe_cudnn::{
    backend::tensor::{Shape, TensorId, TensorSpec},
    data_type::DataType,
    frontend::{
        graph::{Graph, GraphConfig},
        operation::HeuristicMode,
        plan::CompiledGraph,
    },
};
use skytens_core::data_type::{ElementType, FloatType, SIntType, UIntType};

use crate::tensor::{TensorView, VirtTensorView};

/// Helper for building cuDNN graph.
#[derive(Debug)]
pub struct CudnnGraphBuilder {
    cx: Arc<Context>,
    graph: Graph,
}

impl CudnnGraphBuilder {
    pub fn new(cx: &Arc<Context>) -> Self {
        Self {
            cx: cx.clone(),
            graph: Graph::with_config(GraphConfig {
                override_shape_enabled: true,
                ..Default::default()
            }),
        }
    }

    pub fn bind_tensor(&mut self, tensor_view: &VirtTensorView) -> TensorId {
        self.graph.tensor(TensorSpec {
            data_type: match tensor_view.element_type() {
                ElementType::Float(float_type) => match float_type {
                    FloatType::F64 => DataType::F64,
                    FloatType::F32 => DataType::F32,
                    FloatType::Bf16 => DataType::BF16,
                    FloatType::F16 => DataType::F16,
                    FloatType::F8E4M3 => DataType::F8E4M3,
                    FloatType::F8E5M2 => DataType::F8E5M2,
                    FloatType::UF8E8M0 => DataType::F8E8M0,
                    FloatType::F4E2M1 => DataType::F4E2M1,
                },
                ElementType::SInt(sint_type) => match sint_type {
                    SIntType::S64 => DataType::I64,
                    SIntType::S32 => DataType::I32,
                    SIntType::S16 => unimplemented!("s16"),
                    SIntType::S8 => DataType::I8,
                },
                ElementType::UInt(uint_type) => match uint_type {
                    UIntType::U64 => unimplemented!("u64"),
                    UIntType::U32 => DataType::U32,
                    UIntType::U16 => unimplemented!("u16"),
                    UIntType::U8 => DataType::U8,
                },
                ElementType::Bool => DataType::Boolean,
            },
            shape: Shape::contiguous(tensor_view.shape().dims().map(|x| x as i64))
                .unwrap()
                .with_strides(tensor_view.strides().dims().map(|x| x as i64))
                .unwrap(),
            id: None,
            name: None,
            byte_alignment: 16,
            is_virtual: false,
            is_by_value: false,
            vector_count: None,
            vectorized_dimension: None,
            reordering: None,
            ragged_offset: None,
            scalar_value: None,
        })
    }

    pub fn graph_mut(&mut self) -> &mut Graph {
        &mut self.graph
    }

    pub fn build(self) -> anyhow::Result<ExecutableCudnnGraph> {
        thread_local! {
            static CUDNN_CONTEXT_MAP: RefCell<HashMap<*const Context, Arc<singe_cudnn::context::Context>>> = RefCell::new(HashMap::new());
        }

        let cudnn_context = CUDNN_CONTEXT_MAP.with(|map| {
            let mut map = map.borrow_mut();
            Ok::<_, anyhow::Error>(match map.entry(Arc::as_ptr(&self.cx)) {
                Entry::Occupied(e) => e.get().clone(),
                Entry::Vacant(e) => e
                    .insert(Arc::new(singe_cudnn::context::Context::create(&self.cx)?))
                    .clone(),
            })
        })?;

        let compiled = self.graph.compile(&cudnn_context, &[HeuristicMode::B])?;

        let plan_index = compiled.default_plan_index();
        let workspace_bytes = compiled.workspace_size_at(plan_index)?;
        self.cx.bind()?;
        let workspace = DeviceMemory::<u8>::zeroes(workspace_bytes)?;

        Ok(ExecutableCudnnGraph {
            cx: self.cx.clone(),
            cudnn_cx: cudnn_context.clone(),
            graph: compiled,
            plan_index,
            workspace,
        })
    }
}

#[derive(Debug)]
pub struct ExecutableCudnnGraph {
    cx: Arc<Context>,
    cudnn_cx: Arc<singe_cudnn::context::Context>,
    graph: CompiledGraph,
    plan_index: usize,
    workspace: DeviceMemory<u8>,
}

impl ExecutableCudnnGraph {
    /// Executes the graph with the given tensor inputs, on the given stream.
    ///
    /// Note that if the same ExecutableCudnnGraph executes in parallel on a device,
    /// the GPU-side behavior becomes undefined.
    pub fn execute<'a>(
        &mut self,
        operands: impl IntoIterator<Item = (TensorId, &'a TensorView)>,
        stream: &Stream,
    ) -> anyhow::Result<()> {
        assert!(ptr::eq(&*self.cx, stream.context()));

        let mut bindings = self.graph.bindings();
        let mut runtime_overrides = self.graph.runtime_overrides();
        for (tensor_id, tensor_view) in operands {
            unsafe {
                bindings.set_ptr(tensor_id, tensor_view.data_ptr())?;
            }
            runtime_overrides.set(
                tensor_id,
                tensor_view
                    .shape()
                    .dims()
                    .map(|x| x as i64)
                    .collect::<Vec<_>>(),
                tensor_view
                    .strides()
                    .dims()
                    .map(|x| x as i64)
                    .collect::<Vec<_>>(),
            )?;
        }
        let runtime_overrides = runtime_overrides.build();

        self.cudnn_cx.set_stream(Some(stream))?;
        self.graph.execute_at_with_runtime_overrides(
            &self.cudnn_cx,
            &bindings,
            Some(&mut self.workspace),
            self.plan_index,
            &runtime_overrides,
        )?;

        Ok(())
    }
}
