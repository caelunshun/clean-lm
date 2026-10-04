use crate::context::Context;
use skycuda::stream::Stream;
use std::time::Duration;

/// Utility to reliably benchmark a kernel, provided
/// our access to the GPU is exclusive.
pub fn measure_kernel(
    mut submit_measured_work: impl FnMut(&Stream) -> anyhow::Result<()>,
    name: &str,
    cx: &Context,
) -> anyhow::Result<Duration> {
    const MIN_WARMUP_DURATION: Duration = Duration::from_millis(250);
    const MIN_MEASURE_DURATION: Duration = Duration::from_millis(150);
    const ESTIMATE_MEASUREMENTS: u32 = 5;

    let stream = cx.default_compute_stream();

    // One warmup iteration e.g. for JIT caches.
    submit_measured_work(stream)?;
    stream.block_on()?;

    // Obtain initial estimate of kernel runtime from a few measurements.
    let start = stream.record_timed_event()?;
    for _ in 0..ESTIMATE_MEASUREMENTS {
        submit_measured_work(stream)?;
    }
    let elapsed = stream.record_timed_event()?.elapsed_since(&start)?;
    let approx_kernel_duration = elapsed / ESTIMATE_MEASUREMENTS;

    let warmup_invocations = MIN_WARMUP_DURATION
        .as_nanos()
        .div_ceil(approx_kernel_duration.as_nanos());
    for _ in 0..warmup_invocations {
        submit_measured_work(stream)?;
    }

    let measured_invocations = MIN_MEASURE_DURATION
        .as_nanos()
        .div_ceil(approx_kernel_duration.as_nanos()) as u32;
    let start = stream.record_timed_event()?;
    for _ in 0..measured_invocations {
        submit_measured_work(stream)?;
    }
    let elapsed = stream.record_timed_event()?.elapsed_since(&start)?;

    tracing::debug!("{name}: {:.1?}", elapsed / measured_invocations);

    Ok(elapsed / measured_invocations)
}
