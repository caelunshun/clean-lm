use cleandl::{context::Context, operators::conversion::ConversionExecutor, tensor::Tensor};
use rand::Rng;
use rand_distr::Normal;
use skycuda::stream::Stream;
use skytens_core::{
    data_type::{ElementType, FloatType},
    shape::Shape,
};

#[derive(Debug)]
pub struct Parameter {
    pub master_weights: Tensor,
    pub bf16_weights_replica: Tensor,
    pub adam_m: Tensor,
    pub adam_v: Tensor,
    pub gradients: Tensor,
}

impl Parameter {
    pub fn from_normal_distr(
        shape: impl Into<Shape>,
        mean: f32,
        stdev: f32,
        rng: &mut impl Rng,
        stream: &Stream,
        cx: &Context,
    ) -> anyhow::Result<Self> {
        let distr = Normal::<f32>::new(mean, stdev)?;
        let initial_master_weights = Tensor::from_distr(distr, rng, shape, stream, cx)?;
        Self::with_initial_weights(initial_master_weights, stream, cx)
    }

    pub fn constant(
        shape: impl Into<Shape>,
        x: f32,
        stream: &Stream,
        cx: &Context,
    ) -> anyhow::Result<Self> {
        let mut initial_master_weights =
            Tensor::new_zeroed(shape, ElementType::Float(FloatType::F32), stream, cx)?;
        stream.fill(initial_master_weights.storage_mut(), x)?;
        Self::with_initial_weights(initial_master_weights, stream, cx)
    }

    pub fn with_initial_weights(
        initial_master_weights: Tensor,
        stream: &Stream,
        cx: &Context,
    ) -> anyhow::Result<Self> {
        let mut bf16_weights_replica = Tensor::new_zeroed(
            initial_master_weights.shape(),
            ElementType::Float(FloatType::Bf16),
            stream,
            cx,
        )?;
        ConversionExecutor::get(cx)?.f32_to_bf16(
            &initial_master_weights.view(),
            &mut bf16_weights_replica.view_mut(),
            stream,
        )?;

        let adam_m = Tensor::new_zeroed(
            initial_master_weights.shape(),
            ElementType::Float(FloatType::F32),
            stream,
            cx,
        )?;
        let adam_v = Tensor::new_zeroed(
            initial_master_weights.shape(),
            ElementType::Float(FloatType::F32),
            stream,
            cx,
        )?;
        let gradients = Tensor::new_zeroed(
            initial_master_weights.shape(),
            ElementType::Float(FloatType::F32),
            cx.default_stream(),
            cx,
        )?;

        Ok(Self {
            master_weights: initial_master_weights,
            bf16_weights_replica,
            adam_m,
            adam_v,
            gradients,
        })
    }

    pub fn bf16_weights(&self) -> &Tensor {
        &self.bf16_weights_replica
    }
}
