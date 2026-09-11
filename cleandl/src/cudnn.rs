use std::{collections::HashMap, mem, sync::Arc};

use skycuda_cudnn::{CudnnHandle, GraphBuilder, Tensor, TensorId};
use skytens_core::data_type::{ElementType, FloatType, SIntType, UIntType};

use crate::{
    context::Context,
    tensor::{TensorView, VirtTensorView},
};

/// Helper for building cuDNN graph.
pub struct CudnnGraphBuilder<'a> {
    cx: Context,
    cudnn: &'a CudnnHandle,
    graph: GraphBuilder<'a>,
}

impl<'a> CudnnGraphBuilder<'a> {
    pub fn new(cx: &'a Context) -> anyhow::Result<Self> {
        let cudnn = cx.cudnn()?;
        Ok(Self {
            cx: cx.clone(),
            cudnn,
            graph: GraphBuilder::new(cudnn)?,
        })
    }

    pub fn bind_tensor(&mut self, tensor_view: &VirtTensorView) -> Tensor<'_, 'a> {
        self.graph
            .tensor(TensorSpec {
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
                ragged_tensor_multiplier: None,
                scalar_value: None,
            })
            .expect("failed to bind tensor")
    }

    pub fn graph_mut(&mut self) -> &mut Graph {
        &mut self.graph
    }

    pub fn scalar(&mut self, value: f32) -> TensorId {
        self.graph.tensor(TensorSpec {
            data_type: DataType::F32,
            shape: Shape::contiguous([1]).unwrap(),
            id: None,
            scalar_value: Some(ScalarValue::F32(value)),
            name: None,
            byte_alignment: 16,
            is_virtual: false,
            is_by_value: true,
            vector_count: None,
            vectorized_dimension: None,
            reordering: None,
            ragged_offset: None,
            ragged_tensor_multiplier: None,
        })
    }

    pub fn build(self) -> anyhow::Result<ExecutableCudnnGraph> {
        let mut tensor_virt_descs = HashMap::new();

        let cudnn_context = self.cx.cudnn()?;
        let compiled = self.graph.compile(cudnn_context, &[HeuristicMode::B])?;

        for (tensor_id, tensor) in self.graph.tensors() {
            let element_type = match tensor.data_type {
                DataType::F32 => ElementType::Float(FloatType::F32),
                DataType::BF16 => ElementType::Float(FloatType::Bf16),
                DataType::U32 => ElementType::UInt(UIntType::U32),
                DataType::I64 => ElementType::SInt(SIntType::S64),
                DataType::I32 => ElementType::SInt(SIntType::S32),
                _ => todo!("{:?}", tensor.data_type),
            };
            match tensor.ragged_offset {
                Some(_) => {
                    tensor_virt_descs.insert(
                        tensor_id,
                        ExpectedVirtDesc::Ragged {
                            num_dims: tensor.shape.dimensions().len().checked_sub(1).unwrap(),
                            element_type,
                        },
                    );
                }
                None => {
                    tensor_virt_descs.insert(
                        tensor_id,
                        ExpectedVirtDesc::Exact(VirtTensorView::with_strides(
                            skytens_core::shape::Shape::new(
                                tensor
                                    .shape
                                    .dimensions()
                                    .iter()
                                    .copied()
                                    .map(|x| x as usize),
                            ),
                            skytens_core::shape::Strides::new(
                                tensor.shape.strides().iter().copied().map(|x| x as usize),
                            ),
                            element_type,
                        )),
                    );
                }
            }
        }

        let plan_index = compiled.default_plan_index();
        let workspace_bytes = compiled.workspace_size_at(plan_index)?;
        self.cx.cuda().bind()?;

        Ok(ExecutableCudnnGraph {
            cx: self.cx.clone(),
            graph: compiled,
            plan_index,
            workspace_bytes,
            tensor_virt_descs,
        })
    }
}

#[derive(Debug)]
pub struct ExecutableCudnnGraph {
    cx: Context,
    graph: CompiledGraph,
    tensor_virt_descs: HashMap<TensorId, ExpectedVirtDesc>,
    plan_index: usize,
    workspace_bytes: usize,
}

#[derive(Debug)]
enum ExpectedVirtDesc {
    Exact(VirtTensorView),
    Ragged {
        num_dims: usize,
        element_type: ElementType,
    },
}

impl ExecutableCudnnGraph {
    /// Executes the graph with the given tensor inputs, on the given stream.
    pub fn execute<'a>(
        &self,
        operands: impl IntoIterator<Item = (TensorId, &'a TensorView)>,
        stream: &CudaStream,
    ) -> anyhow::Result<()> {
        assert!(Arc::ptr_eq(stream.cuda(), self.cx.cuda()));
        let operands = operands.into_iter().collect::<Vec<_>>();

        let workspace_ptr = if self.workspace_bytes > 0 {
            Some(unsafe { DeviceMemory::<u8>::alloc_async(self.workspace_bytes, stream.get())? })
        } else {
            None
        };
        let mut workspace = workspace_ptr.map(|workspace_ptr| unsafe {
            DeviceMemory::<u8>::from_raw_parts(workspace_ptr, self.workspace_bytes)
        });

        let mut bindings = self.graph.bindings();
        for (tensor_id, tensor_view) in operands {
            match &self.tensor_virt_descs[&tensor_id] {
                ExpectedVirtDesc::Exact(virt_tensor_view) => {
                    assert!(
                        tensor_view.virt().same_layout_as(virt_tensor_view),
                        "expected layout of {:?}, found {:?}",
                        virt_tensor_view,
                        tensor_view.virt()
                    );
                }
                ExpectedVirtDesc::Ragged {
                    num_dims,
                    element_type,
                } => {
                    assert_eq!(tensor_view.shape().num_dims(), *num_dims);
                    assert!(tensor_view.is_packed_row_major());
                    assert_eq!(tensor_view.element_type(), *element_type);
                }
            }

            unsafe {
                bindings.set_ptr(tensor_id, tensor_view.data_ptr())?;
            }
        }

        self.graph.execute_at(
            stream.cudnn_cx()?,
            &bindings,
            workspace.as_mut(),
            self.plan_index,
        )?;

        mem::forget(workspace);
        if let Some(workspace_ptr) = workspace_ptr {
            unsafe {
                DeviceMemory::<u8>::free_async(workspace_ptr, stream.get())?;
            }
        }

        Ok(())
    }
}
