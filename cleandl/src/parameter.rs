use crate::{context::Context, tensor::TensorView};
use skytens_core::{
    data_type::{ElementType, FloatType},
    shape::Shape,
};

#[derive(Debug, Clone)]
pub enum Parameter {
    InferenceOnly(TensorView),
    InferenceAndTraining(TrainParameter),
}

impl Parameter {
    pub fn bf16_weights(&self) -> &TensorView {
        match self {
            Parameter::InferenceOnly(tensor_view) => tensor_view,
            Parameter::InferenceAndTraining(train_parameter) => {
                &train_parameter.bf16_weights_replica
            }
        }
    }

    pub fn unwrap_training(&self) -> &TrainParameter {
        match self {
            Parameter::InferenceAndTraining(train_parameter) => train_parameter,
            _ => panic!("expected Parameter in training mode"),
        }
    }

    pub fn into_inference_only(self) -> TensorView {
        match self {
            Parameter::InferenceOnly(weights) => weights,
            Parameter::InferenceAndTraining(param) => param.bf16_weights_replica,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TrainParameter {
    pub master_weights: TensorView,
    pub bf16_weights_replica: TensorView,
    pub adam_m: TensorView,
    pub adam_v: TensorView,
}

impl TrainParameter {
    pub fn new(shape: Shape, cx: &Context) -> anyhow::Result<Self> {
        Ok(Self {
            master_weights: TensorView::new_zeroed(shape, ElementType::Float(FloatType::F32), cx)?,
            bf16_weights_replica: TensorView::new_zeroed(
                shape,
                ElementType::Float(FloatType::Bf16),
                cx,
            )?,
            adam_m: TensorView::new_zeroed(shape, ElementType::Float(FloatType::F32), cx)?,
            adam_v: TensorView::new_zeroed(shape, ElementType::Float(FloatType::F32), cx)?,
        })
    }
}
