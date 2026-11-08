use crate::{context::Context, disk_cache::DiskCacheKey, execgraph::jit};
use anyhow::ensure;
use rayon::prelude::*;
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

    /// `candidates[0]` must be the heuristic default. Returns the chosen candidate
    /// and its prepared state.
    ///
    /// `prepare` compiles or otherwise sets up a candidate. It runs outside the GPU
    /// lock, only for candidates that will be used: just the chosen one if tuning
    /// isn't needed, else all of them in parallel. `run` submits one execution of a
    /// prepared candidate on the stream.
    pub fn tune<C, P>(
        &self,
        key: &TuneKey,
        candidates: Vec<C>,
        prepare: impl Fn(&C) -> anyhow::Result<P> + Sync,
        mut run: impl FnMut(&P, &Stream) -> anyhow::Result<()>,
    ) -> anyhow::Result<(C, P)>
    where
        C: Serialize + DeserializeOwned + Clone + Send + Sync,
        P: Send,
    {
        ensure!(!candidates.is_empty(), "{}: no tuning candidates", key.op);

        let chosen = if self.mode == Autotune::Off || candidates.len() == 1 {
            Some(0)
        } else if let Some(cached) = self.cached(key, &candidates)? {
            Some(cached)
        } else if self.mode == Autotune::CachedOnly {
            tracing::warn!(
                "{}: no cached tuning result for {}; using the default candidate",
                key.op,
                key.payload
            );
            Some(0)
        } else {
            None
        };
        if let Some(chosen) = chosen {
            let candidate = candidates.into_iter().nth(chosen).unwrap();
            let prepared = prepare(&candidate)?;
            return Ok((candidate, prepared));
        }

        let mut prepared: Vec<(usize, P)> = candidates
            .par_iter()
            .enumerate()
            .filter_map(|(i, candidate)| match prepare(candidate) {
                Ok(prepared) => Some((i, prepared)),
                Err(e) => {
                    tracing::warn!(
                        "{} candidate {} failed to prepare: {e:#}",
                        key.op,
                        candidate_label(candidate)
                    );
                    None
                }
            })
            .collect();
        ensure!(
            !prepared.is_empty(),
            "{}: all tuning candidates failed to prepare",
            key.op
        );

        let gpu_guard = self.cx.gpu_lock().lock();

        // Another thread may have tuned the same key while we were preparing.
        if let Some(cached) = self.cached(key, &candidates)?
            && let Some(pos) = prepared.iter().position(|(i, _)| *i == cached)
        {
            drop(gpu_guard);
            let (_, prepared) = prepared.swap_remove(pos);
            return Ok((candidates[cached].clone(), prepared));
        }

        let mut best: Option<(usize, Duration)> = None;
        for (pos, (i, prepared)) in prepared.iter().enumerate() {
            let label = candidate_label(&candidates[*i]);
            match measure(|stream| run(prepared, stream), self.cx) {
                Ok(time) => {
                    tracing::debug!("{} candidate {label}: {time:.1?}", key.op);
                    if best.is_none_or(|(_, best_time)| time < best_time) {
                        best = Some((pos, time));
                    }
                }
                Err(e) => tracing::warn!("{} candidate {label} failed: {e:#}", key.op),
            }
        }
        drop(gpu_guard);

        let Some((best_pos, time)) = best else {
            anyhow::bail!("{}: all tuning candidates failed", key.op);
        };
        let (best, prepared) = prepared.swap_remove(best_pos);
        let winner = candidates[best].clone();
        tracing::debug!(
            "{}: chose {} ({time:.1?}) for {}",
            key.op,
            candidate_label(&winner),
            key.payload
        );
        if let Err(e) = self.cx.disk_cache().insert(key, &winner) {
            tracing::warn!("{}: failed to cache tuning result: {e:#}", key.op);
        }
        Ok((winner, prepared))
    }

    /// Returns the index of the cached winner, if it is still among the candidates.
    fn cached<C: Serialize>(
        &self,
        key: &TuneKey,
        candidates: &[C],
    ) -> anyhow::Result<Option<usize>> {
        let Some(cached) = self.cx.disk_cache().get::<_, serde_json::Value>(key)? else {
            return Ok(None);
        };
        for (i, candidate) in candidates.iter().enumerate() {
            if serde_json::to_value(candidate)? == cached {
                return Ok(Some(i));
            }
        }
        tracing::warn!(
            "{}: cached tuning result {cached} is no longer a candidate",
            key.op
        );
        Ok(None)
    }
}

fn candidate_label(candidate: &impl Serialize) -> String {
    serde_json::to_string(candidate).unwrap_or_else(|e| format!("<unserializable: {e}>"))
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
