use crate::{
    context::Context,
    tensor::{Tensor, TensorView},
};
use skytens_core::{
    data_type::{ElementType, FloatType},
    shape::Shape,
};

#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
pub enum Parameter {
    InferenceOnly(Tensor),
    InferenceAndTraining(TrainParameter),
}

impl Parameter {
    pub fn bf16_weights(&self) -> &Tensor {
        match self {
            Parameter::InferenceOnly(tensor) => tensor,
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

    pub fn into_inference_only(self) -> Tensor {
        match self {
            Parameter::InferenceOnly(weights) => weights,
            Parameter::InferenceAndTraining(param) => param.bf16_weights_replica,
        }
    }
}

#[derive(Debug)]
pub struct TrainParameter {
    pub master_weights: Tensor,
    pub bf16_weights_replica: Tensor,
    pub optimizer_states: OptimizerStates,
}

impl TrainParameter {
    pub fn new_adamw(shape: Shape, cx: &Context) -> anyhow::Result<Self> {
        Ok(Self {
            master_weights: Tensor::new_zeroed(
                shape,
                ElementType::Float(FloatType::F32),
                cx.default_stream(),
                cx,
            )?,
            bf16_weights_replica: Tensor::new_zeroed(
                shape,
                ElementType::Float(FloatType::Bf16),
                cx.default_stream(),
                cx,
            )?,
            optimizer_states: OptimizerStates::AdamW(AdamWStates {
                adam_m: Tensor::new_zeroed(
                    shape,
                    ElementType::Float(FloatType::F32),
                    cx.default_stream(),
                    cx,
                )?,
                adam_v: Tensor::new_zeroed(
                    shape,
                    ElementType::Float(FloatType::F32),
                    cx.default_stream(),
                    cx,
                )?,
            }),
        })
    }
}

#[derive(Debug)]
pub enum OptimizerStates {
    AdamW(AdamWStates),
}

#[derive(Debug)]
pub struct AdamWStates {
    pub adam_m: Tensor,
    pub adam_v: Tensor,
}
