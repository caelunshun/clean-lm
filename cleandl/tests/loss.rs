use cleandl::{
    context::Context,
    operators::loss::{CrossEntropyLossOutput, LossExecutor},
    par_random::par_vec_from_distr,
    tensor::Tensor,
};
use float_cmp::{ApproxEq, F32Margin};
use half::bf16;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;
use skycuda::{
    device_memory_pool::DeviceMemoryPoolOptions,
    host_pinned_memory_pool::HostPinnedMemoryPoolOptions,
};
use skytens_core::{
    data_type::{ElementType, FloatType, UIntType},
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
            max_total_bytes: None,
        },
        HostPinnedMemoryPoolOptions {
            max_reserved_free_bytes: 1 << 30,
        },
        false,
    )?;
    let stream = cx.cuda().create_stream()?;
    let loss_op = LossExecutor::get(&cx)?;

    let mut logits = Tensor::new_zeroed(
        Shape::new([BATCH_SIZE, NUM_CLASSES]),
        ElementType::Float(FloatType::Bf16),
        &cx,
    )?;
    let mut labels = Tensor::new_zeroed(
        Shape::new([BATCH_SIZE]),
        ElementType::UInt(UIntType::U32),
        &cx,
    )?;

    let mut rng = StdRng::seed_from_u64(0x105);
    let logits_host = par_vec_from_distr(
        StandardNormal.map(|x: f32| bf16::from_f32(x * 3.0)),
        BATCH_SIZE * NUM_CLASSES,
        &mut rng,
    );
    let labels_host: Vec<u32> = (0..BATCH_SIZE)
        .map(|_| rng.random_range(0..NUM_CLASSES as u32))
        .collect();
    let labels_mask: Vec<bool> = (0..BATCH_SIZE).map(|_| rng.random_bool(0.8)).collect();

    cx.upload_to_device(&logits_host, logits.storage_mut(), &stream)?;
    cx.upload_to_device(&labels_host, labels.storage_mut(), &stream)?;

    let stats = loss_op.online_softmax_stats(&logits.view(), &stream)?;
    let CrossEntropyLossOutput { loss, logits_grad } = loss_op.cross_entropy_loss_fwd_bwd(
        &logits.view(),
        &stats.view(),
        &labels.view(),
        &labels_mask,
        &stream,
    )?;

    let stats_staging = cx.download_from_device(stats.storage(), &stream)?;
    let loss_staging = cx.download_from_device(loss.storage(), &stream)?;
    let grad_staging = cx.download_from_device(logits_grad.storage(), &stream)?;
    stream.block_on()?;

    let stats_out: &[f32] = bytemuck::cast_slice(stats_staging.view()?);
    let loss_out: f32 = bytemuck::cast_slice::<_, f32>(loss_staging.view()?)[0];
    let grad_out: &[bf16] = bytemuck::cast_slice(grad_staging.view()?);

    let expected = reference_loss(&logits_host, &labels_host, &labels_mask);

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

    (0..BATCH_SIZE).into_par_iter().try_for_each(|i| {
        check(
            "logsumexp",
            i,
            stats_out[i],
            expected.stats[i],
            STATS_REL_TOLERANCE,
        )
    })?;

    let max_grad_error = (0..BATCH_SIZE * NUM_CLASSES)
        .into_par_iter()
        .try_fold(
            || 0.0f32,
            |max_grad_error, i| -> anyhow::Result<f32> {
                let actual = grad_out[i].to_f32();
                assert!(actual.is_finite());
                check(
                    "logits_grad",
                    i,
                    actual,
                    expected.grad[i],
                    GRAD_REL_TOLERANCE,
                )?;
                Ok(max_grad_error.max((actual - expected.grad[i]).abs()))
            },
        )
        .try_reduce(|| 0.0, |a, b| Ok(a.max(b)))?;

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

fn reference_loss(logits: &[bf16], labels: &[u32], labels_mask: &[bool]) -> Reference {
    let num_unmasked = labels_mask.iter().filter(|&&m| m).count();
    let mut grad = vec![0.0; BATCH_SIZE * NUM_CLASSES];

    let (stats, row_losses): (Vec<f32>, Vec<f64>) = grad
        .par_chunks_mut(NUM_CLASSES)
        .enumerate()
        .map(|(row, row_grad)| {
            let row_logits: Vec<f32> = logits[row * NUM_CLASSES..(row + 1) * NUM_CLASSES]
                .iter()
                .map(|x| x.to_f32())
                .collect();

            let max = row_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum_exp: f64 = row_logits.iter().map(|&x| ((x - max) as f64).exp()).sum();
            let logsumexp = max + (sum_exp as f32).ln();

            if !labels_mask[row] {
                row_grad.fill(0.0);
                return (logsumexp, 0.0);
            }

            let label = labels[row] as usize;
            let row_loss = -(row_logits[label] - logsumexp) as f64;

            for (class, grad) in row_grad.iter_mut().enumerate() {
                let prob = (row_logits[class] - logsumexp).exp();
                let target = f32::from(class == label);
                *grad = (prob - target) / num_unmasked as f32;
            }

            (logsumexp, row_loss)
        })
        .unzip();

    Reference {
        stats,
        loss: (row_losses.iter().sum::<f64>() / num_unmasked as f64) as f32,
        grad,
    }
}
