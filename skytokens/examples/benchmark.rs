// --- LLM-generated --- //

use std::{
    env,
    hash::{DefaultHasher, Hash, Hasher},
    time::Instant,
};

use anyhow::bail;

use skytokens::{
    training::{self, Compression, CorpusBuilder},
    vocab::Vocab,
};

const TARGET_BYTES: usize = 2 << 30;
const VOCAB_SIZE: u32 = 512;

fn main() -> anyhow::Result<()> {
    let compression = match env::args().nth(1).as_deref() {
        None | Some("lz4") => Compression::Lz4,
        Some("none") => Compression::None,
        Some("blosclz") => Compression::BloscLz,
        Some(other) => bail!("unknown compression: {other}"),
    };

    let text = include_str!("shakespeare.txt");
    let repeats = TARGET_BYTES.div_ceil(text.len());

    let start = Instant::now();
    let mut corpus = CorpusBuilder::with_compression(compression);
    for _ in 0..repeats {
        for line in text.lines() {
            corpus.add_document(line)?;
        }
    }
    let corpus = corpus.build();
    eprintln!(
        "built {compression:?} corpus of {} MiB in {:.2?} ({} MiB stored)",
        repeats * text.len() / (1 << 20),
        start.elapsed(),
        corpus.stored_bytes() / (1 << 20),
    );
    if env::args().nth(2).as_deref() == Some("build-only") {
        return Ok(());
    }

    let start = Instant::now();
    let vocab: Vocab<()> = training::train_vocab(corpus, VOCAB_SIZE, 4)?;
    eprintln!(
        "trained {} token vocab in {:.2?}",
        vocab.num_tokens(),
        start.elapsed()
    );

    let mut hasher = DefaultHasher::new();
    vocab.merges().collect::<Vec<_>>().hash(&mut hasher);
    eprintln!("merges hash: {:016x}", hasher.finish());

    Ok(())
}
