use cleandl::{
    bulk_copy::BulkCopyEngine,
    context::{Context, CudaStream},
    operators::loss::LossExecutor,
    tensor::TensorView,
};
use float_cmp::{ApproxEq, F32Margin};
use half::bf16;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use rand_distr::StandardNormal;
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
    let cx = Context::new(0)?;
    let loss_op = LossExecutor::get(&cx)?;

    let logits = TensorView::new_zeroed(
        Shape::new([BATCH_SIZE, NUM_CLASSES]),
        ElementType::Float(FloatType::Bf16),
        &cx,
    )?;
    let stats = TensorView::new_zeroed(
        Shape::new([BATCH_SIZE]),
        ElementType::Float(FloatType::F32),
        &cx,
    )?;
    let labels = TensorView::new_zeroed(
        Shape::new([BATCH_SIZE]),
        ElementType::SInt(SIntType::S32),
        &cx,
    )?;
    let loss = TensorView::new_zeroed(Shape::new([]), ElementType::Float(FloatType::F32), &cx)?;
    let logits_grad = TensorView::new_zeroed(
        Shape::new([BATCH_SIZE, NUM_CLASSES]),
        ElementType::Float(FloatType::Bf16),
        &cx,
    )?;

    let mut rng = StdRng::seed_from_u64(0x105);
    let logits_host: Vec<bf16> = (0..BATCH_SIZE * NUM_CLASSES)
        .map(|_| bf16::from_f32(rng.sample::<f32, _>(StandardNormal) * 3.0))
        .collect();
    let labels_host: Vec<i32> = (0..BATCH_SIZE)
        .map(|_| rng.random_range(0..NUM_CLASSES as i32))
        .collect();

    let mut copy_engine = BulkCopyEngine::new(&cx)?;
    let mut batch = copy_engine.start_batch()?;
    batch.copy_to_device(&logits, |dest| {
        dest.copy_from_slice(bytemuck::cast_slice(&logits_host));
    })?;
    batch.copy_to_device(&labels, |dest| {
        dest.copy_from_slice(bytemuck::cast_slice(&labels_host));
    })?;
    batch.commit()?;

    let stream = CudaStream::new(&cx)?;
    loss_op.online_softmax_stats(&logits, &stats, &stream)?;
    loss_op.cross_entropy_loss_forward(&logits, &stats, &labels, &loss, &logits_grad, &stream)?;
    cx.cuda().synchronize()?;

    let mut batch = copy_engine.start_batch()?;
    let stats_token = batch.copy_to_host(&stats)?;
    let loss_token = batch.copy_to_host(&loss)?;
    let grad_token = batch.copy_to_host(&logits_grad)?;
    batch.commit()?;

    let stats_out: &[f32] = bytemuck::cast_slice(copy_engine.retrieve(stats_token)?);
    let loss_out: f32 = bytemuck::cast_slice::<_, f32>(copy_engine.retrieve(loss_token)?)[0];
    let grad_out: &[bf16] = bytemuck::cast_slice(copy_engine.retrieve(grad_token)?);

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
