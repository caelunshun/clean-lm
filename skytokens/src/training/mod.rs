//! BPE tokenizer training for an in-memory corpus.
//!
//! The corpus can be compressed in memory with LZ4 or zstd
//! to reduce required capacity (and possibly reduce DRAM bandwidth
//! pressure).

use std::{
    collections::{BTreeSet, BinaryHeap},
    sync::{Arc, nonpoison::Mutex},
};

use compact_bytes::CompactBytes;

use crate::training::corpus::Corpus;

mod config;
mod corpus;

struct Trainer {
    corpus: Corpus,
    tokens: Vec<CompactBytes>,

    pairs_heap: BinaryHeap<PairHeapEntry>,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct PairHeapEntry {
    occurrence_count: u64,
    first_token: u32,
    second_token: u32,
}

fn token_id_for_byte(byte: u8) -> u32 {
    byte as u32
}
