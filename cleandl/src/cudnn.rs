use std::{collections::HashMap, fmt};

use skycuda::{
    device_memory::DeviceMemory,
    host_pinned_memory::HostPinnedMemory,
    memory_slice::{MemorySlice, MemorySliceMut},
    stream::Stream,
};
use skycuda_cudnn::{
    BuiltGraph, DataType, GraphBuilder, HeurMode, Tensor, TensorAttributes, TensorId,
};
use skytens_core::{
    data_type::{ElementType, FloatType, SIntType, UIntType},
    shape::{Shape, Strides},
};

use crate::{
    context::Context,
    tensor::{TensorView, TensorViewMut, VirtTensorView},
};

pub fn to_cudnn_data_type(element_type: ElementType) -> DataType {
    match element_type {
        ElementType::Float(float_type) => match float_type {
            FloatType::F64 => DataType::Double,
            FloatType::F32 => DataType::Float,
            FloatType::Bf16 => DataType::Bfloat16,
            FloatType::F16 => DataType::Half,
            FloatType::F8E4M3 => DataType::Fp8E4m3,
            FloatType::F8E5M2 => DataType::Fp8E5m2,
            FloatType::UF8E8M0 => DataType::Fp8E8m0,
            FloatType::F4E2M1 => DataType::Fp4E2m1,
        },
        ElementType::SInt(sint_type) => match sint_type {
            SIntType::S64 => DataType::Int64,
            SIntType::S32 => DataType::Int32,
            SIntType::S16 => unimplemented!("s16"),
            SIntType::S8 => DataType::Int8,
        },
        ElementType::UInt(uint_type) => match uint_type {
            UIntType::U8 => DataType::Uint8,
            other => unimplemented!("{other:?}"),
        },
        ElementType::Bool => DataType::Boolean,
    }
}

pub fn from_cudnn_data_type(data_type: DataType) -> ElementType {
    match data_type {
        DataType::Double => ElementType::Float(FloatType::F64),
        DataType::Float => ElementType::Float(FloatType::F32),
        DataType::Bfloat16 => ElementType::Float(FloatType::Bf16),
        DataType::Half => ElementType::Float(FloatType::F16),
        DataType::Fp8E4m3 => ElementType::Float(FloatType::F8E4M3),
        DataType::Fp8E5m2 => ElementType::Float(FloatType::F8E5M2),
        DataType::Fp8E8m0 => ElementType::Float(FloatType::UF8E8M0),
        DataType::Fp4E2m1 => ElementType::Float(FloatType::F4E2M1),
        DataType::Int64 => ElementType::SInt(SIntType::S64),
        DataType::Int32 => ElementType::SInt(SIntType::S32),
        DataType::Int8 => ElementType::SInt(SIntType::S8),
        DataType::Uint8 => ElementType::UInt(UIntType::U8),
        DataType::Boolean => ElementType::Bool,
        other => todo!("{other:?}"),
    }
}

/// Helper for building cuDNN graphs.
pub struct CudnnGraphBuilder<'a> {
    cx: Context,
    graph: GraphBuilder<'a>,
    next_id: std::cell::Cell<i64>,
    scalars: std::cell::RefCell<Vec<(TensorId, [u8; 4])>>,
}

impl<'a> CudnnGraphBuilder<'a> {
    pub fn new(cx: &'a Context) -> anyhow::Result<Self> {
        let graph = GraphBuilder::new(cx.cudnn()?)?;
        graph
            .set_compute_data_type(DataType::Float)
            .set_intermediate_data_type(DataType::Float);
        Ok(Self {
            cx: cx.clone(),
            graph,
            next_id: std::cell::Cell::new(1),
            scalars: Default::default(),
        })
    }

    pub fn graph(&self) -> &GraphBuilder<'a> {
        &self.graph
    }

    fn alloc_id(&self) -> TensorId {
        let id = self.next_id.get();
        self.next_id.set(id + 1);
        TensorId(id)
    }

    /// Adds a non-virtual input tensor with the given layout.
    pub fn bind_tensor(&self, tensor_view: &VirtTensorView) -> Tensor<'_, 'a> {
        let dims = tensor_view.shape().dims().collect::<Vec<_>>();
        let strides = tensor_view.strides().dims().collect::<Vec<_>>();
        self.graph
            .tensor(
                &TensorAttributes::new(&dims, &strides)
                    .data_type(to_cudnn_data_type(tensor_view.element_type()))
                    .uid(self.alloc_id())
                    .alignment(16),
            )
            .expect("failed to bind tensor")
    }

    /// Marks an operation output as a non-virtual graph output with the given layout,
    /// returning its ID.
    pub fn mark_output(&self, tensor: Tensor<'_, 'a>, tensor_view: &VirtTensorView) -> TensorId {
        let id = self.alloc_id();
        let dims = tensor_view.shape().dims().collect::<Vec<_>>();
        let strides = tensor_view.strides().dims().collect::<Vec<_>>();
        tensor.set_uid(id);
        tensor.set_output(true);
        tensor.set_data_type(to_cudnn_data_type(tensor_view.element_type()));
        tensor.set_dim(&dims);
        tensor.set_stride(&strides);
        id
    }

    pub fn scalar(&self, value: f32) -> Tensor<'_, 'a> {
        let id = self.alloc_id();
        self.scalars.borrow_mut().push((id, value.to_ne_bytes()));
        self.graph
            .tensor(
                &TensorAttributes::new(&[1, 1], &[1, 1])
                    .data_type(DataType::Float)
                    .uid(id)
                    .pass_by_value(true),
            )
            .expect("failed to create scalar tensor")
    }

    pub fn build(self) -> anyhow::Result<ExecutableCudnnGraph> {
        self.cx.cuda().set_as_current()?;
        let graph = self.graph.build(&[HeurMode::A, HeurMode::B])?;

        let mut tensor_virt_descs = HashMap::new();
        for req in graph.all_tensor_requirements() {
            if req.is_pass_by_value() {
                continue;
            }
            let element_type = from_cudnn_data_type(req.data_type());
            let desc = if req.is_ragged() {
                ExpectedVirtDesc::Ragged {
                    num_dims: req.dim().len().checked_sub(1).unwrap(),
                    element_type,
                }
            } else {
                ExpectedVirtDesc::Exact(VirtTensorView::with_strides(
                    skytens_core::shape::Shape::new(req.dim().iter().copied()),
                    skytens_core::shape::Strides::new(req.stride().iter().copied()),
                    element_type,
                ))
            };
            tensor_virt_descs.insert(req.uid(), desc);
        }

        let mut scalars = Vec::new();
        for (id, bytes) in self.scalars.into_inner() {
            let mut memory = self
                .cx
                .host_pinned_memory_pool()
                .alloc_zeroed(bytes.len())?;
            memory.view_mut()?.copy_from_slice(&bytes);
            scalars.push((id, memory));
        }

        Ok(ExecutableCudnnGraph {
            cx: self.cx,
            graph,
            tensor_virt_descs,
            scalars,
        })
    }
}

pub struct ExecutableCudnnGraph {
    cx: Context,
    graph: BuiltGraph,
    tensor_virt_descs: HashMap<TensorId, ExpectedVirtDesc>,
    scalars: Vec<(TensorId, HostPinnedMemory)>,
}

impl fmt::Debug for ExecutableCudnnGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExecutableCudnnGraph")
            .field("tensor_virt_descs", &self.tensor_virt_descs)
            .finish_non_exhaustive()
    }
}

#[derive(Debug)]
enum ExpectedVirtDesc {
    Exact(VirtTensorView),
    Ragged {
        num_dims: usize,
        element_type: ElementType,
    },
}

impl ExpectedVirtDesc {
    fn check(&self, actual: &VirtTensorView, override_shape_enabled: bool) -> bool {
        match self {
            ExpectedVirtDesc::Exact(expected) => {
                if override_shape_enabled {
                    assert_eq!(expected.shape().num_dims(), actual.shape().num_dims());
                    assert_eq!(expected.element_type(), actual.element_type());
                    expected.shape() != actual.shape() || expected.strides() != actual.strides()
                } else {
                    assert!(
                        actual.same_layout_as(expected),
                        "expected layout of {expected:?}, found {actual:?}"
                    );
                    false
                }
            }
            ExpectedVirtDesc::Ragged {
                num_dims,
                element_type,
            } => {
                assert_eq!(actual.shape().num_dims(), *num_dims);
                assert!(actual.is_packed_row_major());
                assert_eq!(actual.element_type(), *element_type);
                false
            }
        }
    }
}

impl ExecutableCudnnGraph {
    /// Executes the graph with the given tensor operands, on the given stream.
    pub fn execute(
        &self,
        inputs: &[(TensorId, &TensorView<'_>)],
        outputs: &mut [(TensorId, TensorViewMut<'_>)],
        stream: &Stream,
    ) -> anyhow::Result<()> {
        assert_eq!(stream.device(), self.cx.cuda());

        let cudnn = self.cx.cudnn_on(stream)?;
        let mut workspace = self
            .cx
            .device_memory_pool()
            .alloc_zeroed_on(self.graph.workspace_size(), stream)?;

        enum Input<'a, 'b> {
            StaticShape(&'a MemorySlice<&'b DeviceMemory>),
            OverrideShape(&'a MemorySlice<&'b DeviceMemory>, Shape, Strides),
        }
        enum InputMut<'a, 'b> {
            StaticShape(&'a mut MemorySliceMut<&'b mut DeviceMemory>),
            OverrideShape(&'a mut MemorySliceMut<&'b mut DeviceMemory>, Shape, Strides),
        }

        let input_memories: Vec<(TensorId, _)> = inputs
            .iter()
            .map(|(id, view)| {
                if self.tensor_virt_descs[id]
                    .check(view.virt(), self.graph.is_override_shape_enabled())
                {
                    (
                        *id,
                        Input::OverrideShape(view.storage(), view.shape(), view.strides()),
                    )
                } else {
                    (*id, Input::StaticShape(view.storage()))
                }
            })
            .collect();
        let mut output_memories: Vec<(TensorId, _)> = outputs
            .iter_mut()
            .map(|(id, view)| {
                if self.tensor_virt_descs[id]
                    .check(view.virt(), self.graph.is_override_shape_enabled())
                {
                    let shape = view.shape();
                    let strides = view.strides();
                    (
                        *id,
                        InputMut::OverrideShape(view.storage_mut(), shape, strides),
                    )
                } else {
                    (*id, InputMut::StaticShape(view.storage_mut()))
                }
            })
            .collect();

        let mut variant_pack = self.graph.new_variant_pack(&cudnn);
        for (id, input) in &input_memories {
            match input {
                Input::StaticShape(memory) => {
                    variant_pack.add(*id, *memory)?;
                }
                Input::OverrideShape(memory, shape, strides) => {
                    variant_pack.add_with_shape(
                        *id,
                        *memory,
                        shape.dims_slice(),
                        strides.dims_slice(),
                    )?;
                }
            }
        }
        for (id, input) in &mut output_memories {
            match input {
                InputMut::StaticShape(memory) => {
                    variant_pack.add_mut(*id, *memory)?;
                }
                InputMut::OverrideShape(memory, shape, strides) => {
                    variant_pack.add_mut_with_shape(
                        *id,
                        *memory,
                        shape.dims_slice(),
                        strides.dims_slice(),
                    )?;
                }
            }
        }
        for (id, memory) in &self.scalars {
            variant_pack.add(*id, memory)?;
        }

        unsafe {
            self.graph.execute(&variant_pack, &mut workspace)?;
        }

        Ok(())
    }
}
