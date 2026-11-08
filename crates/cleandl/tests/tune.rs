mod common;

use cleandl::execgraph::tune::{Autotune, TuneKey, Tuner};
use common::test_context;
use parking_lot::Mutex;
use serde::Serialize;
use skycuda::stream::Stream;

#[derive(Serialize)]
struct Payload {
    nonce: u64,
}

const FAILS_TO_PREPARE: usize = 0xbeef;
const FAILS_TO_RUN: usize = 0xdead;

#[test]
fn tune_picks_fastest_and_caches() -> anyhow::Result<()> {
    let cx = test_context();
    let key = TuneKey::new(
        "test_fill",
        &Payload {
            nonce: rand::random(),
        },
        &cx,
    )?;
    let mut buffer = cx
        .device_memory_pool()
        .alloc_zeroed_on(16 << 20, cx.default_compute_stream())?;
    let candidates = vec![32usize, 1, 8, FAILS_TO_PREPARE, FAILS_TO_RUN];

    let prepared = Mutex::new(Vec::new());
    let prepare = |&fills: &usize| {
        prepared.lock().push(fills);
        anyhow::ensure!(fills != FAILS_TO_PREPARE, "failing preparation");
        Ok(fills)
    };
    let take_prepared = || {
        let mut prepared = std::mem::take(&mut *prepared.lock());
        prepared.sort();
        prepared
    };
    let mut run = |&fills: &usize, stream: &Stream| {
        anyhow::ensure!(fills != FAILS_TO_RUN, "failing candidate");
        for _ in 0..fills {
            stream.fill(&mut buffer, 0u8)?;
        }
        Ok(())
    };

    let tuner = |mode| Tuner::new(&cx, mode);
    assert_eq!(
        tuner(Autotune::Off).tune(&key, candidates.clone(), prepare, &mut run)?,
        (32, 32)
    );
    assert_eq!(take_prepared(), [32]);
    assert_eq!(
        tuner(Autotune::CachedOnly).tune(&key, candidates.clone(), prepare, &mut run)?,
        (32, 32)
    );
    assert_eq!(take_prepared(), [32]);
    assert_eq!(
        tuner(Autotune::Full).tune(&key, candidates.clone(), prepare, &mut run)?,
        (1, 1)
    );
    let mut all = candidates.clone();
    all.sort();
    assert_eq!(take_prepared(), all);

    // Cached results are used without measuring, preparing only the winner.
    let never_run = |_: &usize, _: &Stream| -> anyhow::Result<()> {
        panic!("cached result should not be measured")
    };
    for mode in [Autotune::CachedOnly, Autotune::Full] {
        assert_eq!(
            tuner(mode).tune(&key, candidates.clone(), prepare, never_run)?,
            (1, 1)
        );
        assert_eq!(take_prepared(), [1]);
    }
    // A fresh context reads the result back from disk.
    let cx2 = test_context();
    assert_eq!(
        Tuner::new(&cx2, Autotune::CachedOnly).tune(
            &key,
            candidates.clone(),
            prepare,
            never_run
        )?,
        (1, 1)
    );

    // Stale cached winners fall back to the default.
    assert_eq!(
        tuner(Autotune::CachedOnly).tune(&key, vec![32usize, 8], prepare, never_run)?,
        (32, 32)
    );
    Ok(())
}

#[test]
fn concurrent_tunes_agree() -> anyhow::Result<()> {
    let cx = test_context();
    let key = TuneKey::new(
        "test_fill_concurrent",
        &Payload {
            nonce: rand::random(),
        },
        &cx,
    )?;
    let results: Vec<usize> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..2)
            .map(|_| {
                s.spawn(|| {
                    let mut buffer = cx
                        .device_memory_pool()
                        .alloc_zeroed_on(16 << 20, cx.default_compute_stream())?;
                    let (chosen, _) = Tuner::new(&cx, Autotune::Full).tune(
                        &key,
                        vec![16usize, 1, 4],
                        |&fills| Ok(fills),
                        |&fills, stream| {
                            for _ in 0..fills {
                                stream.fill(&mut buffer, 0u8)?;
                            }
                            Ok(())
                        },
                    )?;
                    anyhow::Ok(chosen)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect::<anyhow::Result<_>>()
    })?;
    assert_eq!(results, [1, 1]);
    Ok(())
}

#[test]
fn tune_fails_if_all_candidates_fail() -> anyhow::Result<()> {
    let cx = test_context();
    let key = TuneKey::new(
        "test_fail",
        &Payload {
            nonce: rand::random(),
        },
        &cx,
    )?;
    let tuner = Tuner::new(&cx, Autotune::Full);
    assert!(
        tuner
            .tune(
                &key,
                vec![1, 2],
                |&c: &i32| Ok(c),
                |_, _| anyhow::bail!("nope")
            )
            .is_err()
    );
    assert!(
        tuner
            .tune(
                &key,
                vec![1, 2],
                |_: &i32| -> anyhow::Result<()> { anyhow::bail!("nope") },
                |_, _| Ok(())
            )
            .is_err()
    );
    Ok(())
}
