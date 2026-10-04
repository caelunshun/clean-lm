// --- LLM-generated --- //
//! File of [`Document`]s, each compressed separately with a shared zstd dictionary
//! so that decoding can be streamed and parallelized. Layout (integers little-endian):
//!
//! - `dict_len: u32`, followed by the zstd dictionary (empty if training failed)
//! - `num_documents: u64`
//! - per document: `compressed_len: u32`, `encoded_len: u32`, then a zstd frame
//!   holding the postcard encoding of the document

use std::{
    io::{BufReader, BufWriter, Read, Write},
    path::Path,
    sync::mpsc::{Receiver, SyncSender, sync_channel},
    thread::JoinHandle,
};

use anyhow::{Context, anyhow, ensure};
use rayon::prelude::*;
use zstd::{
    bulk::{Compressor, Decompressor},
    dict::{DecoderDictionary, EncoderDictionary},
};

use crate::skyred::dataset::schema::Document;

const ZSTD_LEVEL: i32 = 4;
const DICT_MAX_SIZE: usize = 256 * 1024;
/// Upper bound on the encoded document bytes used to train the dictionary.
const DICT_TRAINING_BYTES: usize = 64 * 1024 * 1024;
const WRITE_CHUNK_DOCUMENTS: usize = 16 * 1024;
const READ_BATCH_DOCUMENTS: usize = 4096;
const READ_BATCHES_IN_FLIGHT: usize = 4;

/// The dictionary is trained on a prefix of `documents`, so they should be in random order.
pub fn write_documents(path: &Path, documents: &[Document]) -> anyhow::Result<()> {
    let dict = train_dictionary(documents)?;
    let cdict = EncoderDictionary::copy(&dict, ZSTD_LEVEL);

    let mut writer = BufWriter::with_capacity(256 * 1024, fs_err::File::create(path)?);
    writer.write_all(&u32::try_from(dict.len())?.to_le_bytes())?;
    writer.write_all(&dict)?;
    writer.write_all(&(documents.len() as u64).to_le_bytes())?;

    let mut compressed_bytes = 0;
    for chunk in documents.chunks(WRITE_CHUNK_DOCUMENTS) {
        let records = chunk
            .par_iter()
            .map_init(
                || {
                    Compressor::with_prepared_dictionary(&cdict)
                        .expect("failed to create zstd compressor")
                },
                |compressor, document| -> anyhow::Result<_> {
                    let encoded = postcard::to_allocvec(document)?;
                    Ok((encoded.len(), compressor.compress(&encoded)?))
                },
            )
            .collect::<anyhow::Result<Vec<_>>>()?;
        for (encoded_len, compressed) in records {
            writer.write_all(&u32::try_from(compressed.len())?.to_le_bytes())?;
            writer.write_all(&u32::try_from(encoded_len)?.to_le_bytes())?;
            writer.write_all(&compressed)?;
            compressed_bytes += compressed.len();
        }
    }
    writer.flush()?;

    tracing::info!(dict_bytes = dict.len(), compressed_bytes, "wrote documents");
    Ok(())
}

fn train_dictionary(documents: &[Document]) -> anyhow::Result<Vec<u8>> {
    let mut samples = Vec::new();
    let mut sample_sizes = Vec::new();
    for document in documents {
        if samples.len() >= DICT_TRAINING_BYTES {
            break;
        }
        let start = samples.len();
        samples = postcard::to_extend(document, samples)?;
        sample_sizes.push(samples.len() - start);
    }

    match zstd::dict::from_continuous(&samples, &sample_sizes, DICT_MAX_SIZE) {
        Ok(dict) => Ok(dict),
        Err(e) => {
            tracing::warn!(
                num_samples = sample_sizes.len(),
                "zstd dictionary training failed, compressing without dictionary: {e}"
            );
            Ok(Vec::new())
        }
    }
}

/// Streams documents in file order. Decoding runs on a background thread
/// (fanned out to the rayon pool) a few batches ahead of the consumer.
pub struct DocumentReader {
    num_documents: u64,
    batches: Receiver<anyhow::Result<Vec<Document>>>,
    current: std::vec::IntoIter<Document>,
    thread: Option<JoinHandle<()>>,
}

impl DocumentReader {
    pub fn open(path: &Path) -> anyhow::Result<Self> {
        let mut reader = BufReader::new(fs_err::File::open(path)?);
        let mut dict = vec![0u8; read_u32(&mut reader)? as usize];
        reader.read_exact(&mut dict)?;
        let num_documents = read_u64(&mut reader)?;

        let (sender, batches) = sync_channel(READ_BATCHES_IN_FLIGHT);
        let thread = std::thread::spawn(move || {
            decode_batches(reader, &dict, num_documents, sender);
        });

        Ok(Self {
            num_documents,
            batches,
            current: Vec::new().into_iter(),
            thread: Some(thread),
        })
    }

    pub fn num_documents(&self) -> u64 {
        self.num_documents
    }
}

impl Iterator for DocumentReader {
    type Item = anyhow::Result<Document>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(document) = self.current.next() {
                return Some(Ok(document));
            }
            match self.batches.recv() {
                Ok(Ok(batch)) => self.current = batch.into_iter(),
                Ok(Err(e)) => return Some(Err(e)),
                Err(_) => {
                    if let Some(thread) = self.thread.take()
                        && thread.join().is_err()
                    {
                        return Some(Err(anyhow!("document decoding thread panicked")));
                    }
                    return None;
                }
            }
        }
    }
}

fn decode_batches(
    mut reader: impl Read,
    dict: &[u8],
    num_documents: u64,
    sender: SyncSender<anyhow::Result<Vec<Document>>>,
) {
    let ddict = DecoderDictionary::copy(dict);
    let mut remaining = num_documents;
    while remaining > 0 {
        let batch_size = remaining.min(READ_BATCH_DOCUMENTS as u64);
        remaining -= batch_size;
        let batch = decode_batch(&mut reader, &ddict, batch_size as usize).and_then(|batch| {
            if remaining == 0 {
                expect_eof(&mut reader)?;
            }
            Ok(batch)
        });
        let failed = batch.is_err();
        if sender.send(batch).is_err() || failed {
            return;
        }
    }
}

fn decode_batch(
    reader: &mut impl Read,
    ddict: &DecoderDictionary,
    batch_size: usize,
) -> anyhow::Result<Vec<Document>> {
    let mut records = Vec::with_capacity(batch_size);
    for _ in 0..batch_size {
        let compressed_len = read_u32(reader)? as usize;
        let encoded_len = read_u32(reader)? as usize;
        let mut compressed = vec![0u8; compressed_len];
        reader
            .read_exact(&mut compressed)
            .context("truncated document file")?;
        records.push((encoded_len, compressed));
    }

    records
        .into_par_iter()
        .map_init(
            || {
                Decompressor::with_prepared_dictionary(ddict)
                    .expect("failed to create zstd decompressor")
            },
            |decompressor, (encoded_len, compressed)| {
                let encoded = decompressor.decompress(&compressed, encoded_len)?;
                ensure!(encoded.len() == encoded_len, "document length mismatch");
                Ok(postcard::from_bytes(&encoded)?)
            },
        )
        .collect()
}

fn expect_eof(reader: &mut impl Read) -> anyhow::Result<()> {
    ensure!(
        reader.read(&mut [0u8])? == 0,
        "trailing data after last document"
    );
    Ok(())
}

fn read_u32(reader: &mut impl Read) -> anyhow::Result<u32> {
    let mut bytes = [0u8; 4];
    reader
        .read_exact(&mut bytes)
        .context("truncated document file")?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64(reader: &mut impl Read) -> anyhow::Result<u64> {
    let mut bytes = [0u8; 8];
    reader
        .read_exact(&mut bytes)
        .context("truncated document file")?;
    Ok(u64::from_le_bytes(bytes))
}

#[cfg(test)]
mod tests {
    use jiff::civil::date;
    use rand::{RngExt, SeedableRng, rngs::StdRng, seq::IndexedRandom};

    use super::*;
    use crate::skyred::dataset::schema::Comment;

    const WORDS: &[&str] = &[
        "the", "reddit", "post", "comment", "thread", "upvote", "deleted", "edit:", "lol", "this",
        "is", "why", "I", "think", "you", "are", "wrong", "source?", "thanks", "https://",
    ];

    fn random_text(rng: &mut StdRng, max_words: usize) -> String {
        let num_words = rng.random_range(0..=max_words);
        (0..num_words)
            .map(|_| *WORDS.choose(rng).unwrap())
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn random_documents(n: usize, seed: u64) -> Vec<Document> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|i| Document {
                subreddit: format!("sub{}", i % 17).into(),
                date: date(
                    2010 + (i % 10) as i16,
                    1 + (i % 12) as i8,
                    1 + (i % 28) as i8,
                ),
                title: random_text(&mut rng, 20),
                author: format!("user{}", rng.random_range(0..1000)).into(),
                selftext: random_text(&mut rng, 200).into(),
                comment_chain: (0..rng.random_range(0..5))
                    .map(|_| Comment {
                        author: format!("user{}", rng.random_range(0..1000)).into(),
                        text: random_text(&mut rng, 100).into(),
                    })
                    .collect(),
            })
            .collect()
    }

    fn assert_round_trip(documents: &[Document]) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("documents");
        write_documents(&path, documents).unwrap();

        let reader = DocumentReader::open(&path).unwrap();
        assert_eq!(reader.num_documents(), documents.len() as u64);
        let read = reader.collect::<anyhow::Result<Vec<_>>>().unwrap();
        assert_eq!(read.len(), documents.len());
        for (a, b) in read.iter().zip(documents) {
            assert_eq!(
                postcard::to_allocvec(a).unwrap(),
                postcard::to_allocvec(b).unwrap()
            );
        }
    }

    #[test]
    fn round_trip_with_dictionary() {
        let documents = random_documents(3 * READ_BATCH_DOCUMENTS + 17, 0);
        assert!(!train_dictionary(&documents).unwrap().is_empty());
        assert_round_trip(&documents);
    }

    #[test]
    fn round_trip_without_dictionary() {
        let documents = random_documents(3, 1);
        assert!(train_dictionary(&documents).unwrap().is_empty());
        assert_round_trip(&documents);
    }

    #[test]
    fn round_trip_empty() {
        assert_round_trip(&[]);
    }

    #[test]
    fn rejects_truncated_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("documents");
        write_documents(&path, &random_documents(100, 2)).unwrap();
        let len = fs_err::metadata(&path).unwrap().len();
        fs_err::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(len - 1)
            .unwrap();

        let result = DocumentReader::open(&path)
            .unwrap()
            .collect::<anyhow::Result<Vec<_>>>();
        assert!(result.is_err());
    }
}
