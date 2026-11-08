use crate::{context::Context, disk_cache::DiskCacheKey, execgraph::jit};
use anyhow::ensure;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use skycuda::stream::Stream;
use std::time::Duration;

/// Bump to invalidate all cached tuning results.
const TUNER_VERSION: u32 = 1;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum Autotune {
    /// Always use the heuristic default.
    Off,
    /// Use cached tuning results if present, otherwise the heuristic default.
    #[default]
    CachedOnly,
    /// Use cached tuning results if present, otherwise measure all candidates.
    Full,
}

/// Disk cache key for a tuning result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TuneKey {
    pub op: String,
    /// Op configuration, including hint shapes and data types.
    pub payload: serde_json::Value,
    pub arch: String,
    pub sm_count: usize,
    /// `(component, version)` pairs, e.g. library and compiler versions.
    pub versions: Vec<(String, String)>,
}

#[typetag::serde]
impl DiskCacheKey for TuneKey {}

impl TuneKey {
    /// Creates a key for the context's device, including the tuner, cuBLASLt
    /// and `tileiras` versions.
    pub fn new(
        op: impl Into<String>,
        payload: &impl Serialize,
        cx: &Context,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            op: op.into(),
            payload: serde_json::to_value(payload)?,
            arch: jit::tileiras_arch(cx.cuda()),
            sm_count: cx.cuda().sm_count(),
            versions: vec![
                ("tuner".into(), TUNER_VERSION.to_string()),
                ("cublaslt".into(), skycuda_cublaslt::version().to_string()),
                ("tileiras".into(), jit::tileiras_version().into()),
            ],
        })
    }

    #[must_use]
    pub fn with_version(
        mut self,
        component: impl Into<String>,
        version: impl Into<String>,
    ) -> Self {
        self.versions.push((component.into(), version.into()));
        self
    }
}

pub struct Tuner<'a> {
    cx: &'a Context,
    mode: Autotune,
}

impl<'a> Tuner<'a> {
    pub fn new(cx: &'a Context, mode: Autotune) -> Self {
        Self { cx, mode }
    }

    pub fn mode(&self) -> Autotune {
        self.mode
    }

    /// `candidates[0]` must be the heuristic default. Returns the chosen candidate.
    ///
    /// `run` submits one execution of a candidate on the stream. Any compilation
    /// should happen before calling this, as measurement holds the context's GPU lock.
    pub fn tune<C: Serialize + DeserializeOwned + Clone>(
        &self,
        key: &TuneKey,
        candidates: Vec<C>,
        mut run: impl FnMut(&C, &Stream) -> anyhow::Result<()>,
    ) -> anyhow::Result<C> {
        ensure!(!candidates.is_empty(), "{}: no tuning candidates", key.op);
        if self.mode == Autotune::Off || candidates.len() == 1 {
            return Ok(candidates.into_iter().next().unwrap());
        }

        if let Some(cached) = self.cached(key, &candidates)? {
            return Ok(cached);
        }
        if self.mode == Autotune::CachedOnly {
            tracing::warn!(
                "{}: no cached tuning result for {}; using the default candidate",
                key.op,
                key.payload
            );
            return Ok(candidates.into_iter().next().unwrap());
        }

        let best = {
            let _gpu_guard = self.cx.gpu_lock().lock();
            let mut best: Option<(usize, Duration)> = None;
            for (i, candidate) in candidates.iter().enumerate() {
                let label = serde_json::to_string(candidate)?;
                match measure(|stream| run(candidate, stream), self.cx) {
                    Ok(time) => {
                        tracing::debug!("{} candidate {label}: {time:.1?}", key.op);
                        if best.is_none_or(|(_, best_time)| time < best_time) {
                            best = Some((i, time));
                        }
                    }
                    Err(e) => tracing::warn!("{} candidate {label} failed: {e:#}", key.op),
                }
            }
            best
        };
        let Some((best, time)) = best else {
            anyhow::bail!("{}: all tuning candidates failed", key.op);
        };

        let winner = candidates[best].clone();
        tracing::debug!(
            "{}: chose {} ({time:.1?}) for {}",
            key.op,
            serde_json::to_string(&winner)?,
            key.payload
        );
        self.cx.disk_cache().insert(key, &winner)?;
        Ok(winner)
    }

    /// Returns the cached winner, if it is still among the candidates.
    fn cached<C: Serialize + DeserializeOwned + Clone>(
        &self,
        key: &TuneKey,
        candidates: &[C],
    ) -> anyhow::Result<Option<C>> {
        let Some(cached) = self.cx.disk_cache().get::<_, serde_json::Value>(key)? else {
            return Ok(None);
        };
        for candidate in candidates {
            if serde_json::to_value(candidate)? == cached {
                return Ok(Some(candidate.clone()));
            }
        }
        tracing::warn!(
            "{}: cached tuning result {cached} is no longer a candidate",
            key.op
        );
        Ok(None)
    }
}

/// Measures the average runtime of the submitted work, assuming exclusive access to the GPU.
fn measure(
    mut submit: impl FnMut(&Stream) -> anyhow::Result<()>,
    cx: &Context,
) -> anyhow::Result<Duration> {
    const MIN_WARMUP_DURATION: Duration = Duration::from_millis(250);
    const MIN_MEASURE_DURATION: Duration = Duration::from_millis(500);
    const ESTIMATE_MEASUREMENTS: u32 = 5;

    let stream = cx.default_compute_stream();

    // One warmup iteration e.g. for JIT caches.
    submit(stream)?;
    stream.block_on()?;

    // Obtain initial estimate of runtime from a few measurements.
    let start = stream.record_timed_event()?;
    for _ in 0..ESTIMATE_MEASUREMENTS {
        submit(stream)?;
    }
    let elapsed = stream.record_timed_event()?.elapsed_since(&start)?;
    let approx_duration_ns = (elapsed / ESTIMATE_MEASUREMENTS).as_nanos().max(1);

    let warmup_invocations = MIN_WARMUP_DURATION.as_nanos().div_ceil(approx_duration_ns);
    for _ in 0..warmup_invocations {
        submit(stream)?;
    }

    let measured_invocations = MIN_MEASURE_DURATION.as_nanos().div_ceil(approx_duration_ns) as u32;
    let start = stream.record_timed_event()?;
    for _ in 0..measured_invocations {
        submit(stream)?;
    }
    let elapsed = stream.record_timed_event()?.elapsed_since(&start)?;
    stream.block_on()?;

    Ok(elapsed / measured_invocations)
}
