use cleandl::{context::Context, operators::loss::LossExecutor, tensor::Tensor};
use float_cmp::{ApproxEq, F32Margin};
use half::bf16;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use rand_distr::StandardNormal;
use skycuda::{
    device_memory_pool::DeviceMemoryPoolOptions,
    host_pinned_memory_pool::HostPinnedMemoryPoolOptions,
};
use skytens_core::{
    data_type::{ElementType, FloatType, SIntType},
    shape::Shape,
};

const BATCH_SIZE: usize = 300;
const NUM_CLASSES: usize = 592;

const STATS_REL_TOLERANCE: f32 = 1e-5;
const LOSS_REL_TOLERANCE: f32 = 1e-4;
const GRAD_REL_TOLERANCE: f32 = 1e-2;
const ABS_TOLERANCE: f32 = 1e-9;

#[test]
fn cross_entropy_loss() -> anyhow::Result<()> {
    let cx = Context::new(
        0,
        DeviceMemoryPoolOptions {
            max_reserved_free_bytes: 1 << 30,
        },
        HostPinnedMemoryPoolOptions {
            max_reserved_free_bytes: 1 << 30,
        },
    )?;
    let stream = cx.cuda().create_stream()?;
    let loss_op = LossExecutor::get(&cx)?;

    let mut logits = Tensor::new_zeroed(
        Shape::new([BATCH_SIZE, NUM_CLASSES]),
        ElementType::Float(FloatType::Bf16),
        &stream,
        &cx,
    )?;
    let mut stats = Tensor::new_zeroed(
        Shape::new([BATCH_SIZE]),
        ElementType::Float(FloatType::F32),
        &stream,
        &cx,
    )?;
    let mut labels = Tensor::new_zeroed(
        Shape::new([BATCH_SIZE]),
        ElementType::SInt(SIntType::S32),
        &stream,
        &cx,
    )?;
    let mut loss = Tensor::new_zeroed(
        Shape::new([]),
        ElementType::Float(FloatType::F32),
        &stream,
        &cx,
    )?;
    let mut logits_grad = Tensor::new_zeroed(
        Shape::new([BATCH_SIZE, NUM_CLASSES]),
        ElementType::Float(FloatType::Bf16),
        &stream,
        &cx,
    )?;

    let mut rng = StdRng::seed_from_u64(0x105);
    let logits_host: Vec<bf16> = (0..BATCH_SIZE * NUM_CLASSES)
        .map(|_| bf16::from_f32(rng.sample::<f32, _>(StandardNormal) * 3.0))
        .collect();
    let labels_host: Vec<i32> = (0..BATCH_SIZE)
        .map(|_| rng.random_range(0..NUM_CLASSES as i32))
        .collect();

    cx.upload_to_device(&logits_host, logits.storage_mut(), &stream)?;
    cx.upload_to_device(&labels_host, labels.storage_mut(), &stream)?;

    loss_op.online_softmax_stats(&logits.view(), &mut stats.view_mut(), &stream)?;
    loss_op.cross_entropy_loss_forward(
        &logits.view(),
        &stats.view(),
        &labels.view(),
        &mut loss.view_mut(),
        &mut logits_grad.view_mut(),
        &stream,
    )?;

    let stats_staging = cx.download_from_device(stats.storage(), &stream)?;
    let loss_staging = cx.download_from_device(loss.storage(), &stream)?;
    let grad_staging = cx.download_from_device(logits_grad.storage(), &stream)?;
    stream.block_on()?;

    let stats_out: &[f32] = bytemuck::cast_slice(stats_staging.view()?);
    let loss_out: f32 = bytemuck::cast_slice::<_, f32>(loss_staging.view()?)[0];
    let grad_out: &[bf16] = bytemuck::cast_slice(grad_staging.view()?);

    let expected = reference_loss(&logits_host, &labels_host);

    let check =
        |name: &str, i: usize, actual: f32, expected: f32, rel: f32| -> anyhow::Result<()> {
            let margin = F32Margin {
                epsilon: ABS_TOLERANCE.max(rel * expected.abs()),
                ulps: 0,
            };
            anyhow::ensure!(
                actual.approx_eq(expected, margin),
                "{name} mismatch at index {i}: expected {expected}, got {actual}"
            );
            Ok(())
        };

    for i in 0..BATCH_SIZE {
        check(
            "logsumexp",
            i,
            stats_out[i],
            expected.stats[i],
            STATS_REL_TOLERANCE,
        )?;
    }

    let mut max_grad_error = 0.0f32;
    for i in 0..BATCH_SIZE * NUM_CLASSES {
        let actual = grad_out[i].to_f32();
        assert!(actual.is_finite());
        max_grad_error = max_grad_error.max((actual - expected.grad[i]).abs());
        check(
            "logits_grad",
            i,
            actual,
            expected.grad[i],
            GRAD_REL_TOLERANCE,
        )?;
    }

    check("loss", 0, loss_out, expected.loss, LOSS_REL_TOLERANCE)?;

    println!(
        "results match the reference (loss: {loss_out}, max absolute gradient error: {max_grad_error})"
    );

    Ok(())
}

struct Reference {
    stats: Vec<f32>,
    loss: f32,
    grad: Vec<f32>,
}

fn reference_loss(logits: &[bf16], labels: &[i32]) -> Reference {
    let mut out = Reference {
        stats: Vec::with_capacity(BATCH_SIZE),
        loss: 0.0,
        grad: vec![0.0; BATCH_SIZE * NUM_CLASSES],
    };

    let mut loss_sum = 0.0f64;
    for row in 0..BATCH_SIZE {
        let row_logits: Vec<f32> = logits[row * NUM_CLASSES..(row + 1) * NUM_CLASSES]
            .iter()
            .map(|x| x.to_f32())
            .collect();

        let max = row_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum_exp: f64 = row_logits.iter().map(|&x| ((x - max) as f64).exp()).sum();
        let logsumexp = max + (sum_exp as f32).ln();
        out.stats.push(logsumexp);

        let label = labels[row] as usize;
        loss_sum += -(row_logits[label] - logsumexp) as f64;

        for class in 0..NUM_CLASSES {
            let prob = (row_logits[class] - logsumexp).exp();
            let target = f32::from(class == label);
            out.grad[row * NUM_CLASSES + class] = (prob - target) / BATCH_SIZE as f32;
        }
    }

    out.loss = (loss_sum / BATCH_SIZE as f64) as f32;
    out
}
