mod common;

use cleandl::execgraph::tune::{Autotune, TuneKey, Tuner};
use common::test_context;
use serde::Serialize;

#[derive(Serialize)]
struct Payload {
    nonce: u64,
}

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
    let candidates = vec![32usize, 1, 8, 0xdead];

    let mut run = |&fills: &usize, stream: &skycuda::stream::Stream| {
        anyhow::ensure!(fills != 0xdead, "failing candidate");
        for _ in 0..fills {
            stream.fill(&mut buffer, 0u8)?;
        }
        Ok(())
    };

    assert_eq!(
        Tuner::new(&cx, Autotune::Off).tune(&key, candidates.clone(), &mut run)?,
        32
    );
    assert_eq!(
        Tuner::new(&cx, Autotune::CachedOnly).tune(&key, candidates.clone(), &mut run)?,
        32
    );
    assert_eq!(
        Tuner::new(&cx, Autotune::Full).tune(&key, candidates.clone(), &mut run)?,
        1
    );

    // Cached results are used without measuring.
    let mut never_run = |_: &usize, _: &skycuda::stream::Stream| -> anyhow::Result<()> {
        panic!("cached result should not be measured")
    };
    for mode in [Autotune::CachedOnly, Autotune::Full] {
        assert_eq!(
            Tuner::new(&cx, mode).tune(&key, candidates.clone(), &mut never_run)?,
            1
        );
    }
    // A fresh context reads the result back from disk.
    let cx2 = test_context();
    assert_eq!(
        Tuner::new(&cx2, Autotune::CachedOnly).tune(&key, candidates.clone(), &mut never_run)?,
        1
    );

    // Stale cached winners fall back to the default.
    assert_eq!(
        Tuner::new(&cx, Autotune::CachedOnly).tune(&key, vec![32usize, 8], &mut never_run)?,
        32
    );
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
    let res =
        Tuner::new(&cx, Autotune::Full).tune(&key, vec![1, 2], |_: &i32, _| anyhow::bail!("nope"));
    assert!(res.is_err());
    Ok(())
}
