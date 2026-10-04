//! Corpus is represented as chunks of tokens, where documents
//! cannot be spread across chunks, and each chunk is blosc-compressed.
//! Documents within a chunk are separated by DOC_TERMINATOR_TOKEN.

use std::{
    cell::RefCell,
    mem,
    sync::{Arc, OnceLock},
    thread::available_parallelism,
};

use blosc2::{
    CParams, CompressAlgo, DParams,
    chunk::{Decoder, Encoder},
};
use dashmap::DashMap;
use hashbrown::HashMap;
use priority_queue::PriorityQueue;
use rayon::prelude::*;
use simple_semaphore::Semaphore;

use crate::{
    training::{DOC_TERMINATOR_TOKEN, token_id_for_byte},
    vocab::{Merge, TokenId},
};

const TOKENS_PER_CHUNK: usize = 16 * 1024;

thread_local! {
    static ENCODER: RefCell<Encoder> = {
        let mut params = CParams::default();
        params.nthreads(1).typesize(size_of::<TokenId>()).unwrap().compressor(CompressAlgo::Lz4);
        RefCell::new(Encoder::new(params).expect("failed to init blosc"))
    };
}

thread_local! {
    static DECODER: RefCell<Decoder> = {
        let mut params = DParams::default();
        params.nthreads(1);

        RefCell::new(Decoder::new(params).expect("failed to init blosc"))
    };
}

#[derive(Debug)]
pub struct CorpusBuilder {
    compressed_chunks: Vec<Arc<OnceLock<Arc<[u8]>>>>,
    concurrency_limiter: Arc<Semaphore>,
    current_chunk: Vec<TokenId>,
    token_pair_occurrence_counts: HashMap<(TokenId, TokenId), usize>,
    pool: rayon::ThreadPool,
}

impl CorpusBuilder {
    pub fn new() -> Self {
        let threads = available_parallelism().unwrap().get();
        Self {
            compressed_chunks: Default::default(),
            concurrency_limiter: Semaphore::new(4 * threads),
            current_chunk: Default::default(),
            token_pair_occurrence_counts: Default::default(),
            pool: rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap(),
        }
    }

    pub fn add_document(&mut self, document: impl AsRef<[u8]>) {
        for &byte in document.as_ref() {
            self.add_token(token_id_for_byte(byte));
        }
        for &[first, second] in document.as_ref().array_windows::<2>() {
            *self
                .token_pair_occurrence_counts
                .entry((token_id_for_byte(first), token_id_for_byte(second)))
                .or_default() += 1;
        }
        self.add_token(DOC_TERMINATOR_TOKEN);
        if self.current_chunk.len() >= TOKENS_PER_CHUNK {
            self.flush_chunk();
        }
    }

    fn add_token(&mut self, token: TokenId) {
        self.current_chunk.push(token);
    }

    fn flush_chunk(&mut self) {
        if self.current_chunk.is_empty() {
            return;
        }

        let chunk_cell = Arc::new(OnceLock::new());
        self.compressed_chunks.push(chunk_cell.clone());

        let chunk_uncompressed = mem::take(&mut self.current_chunk);
        let permit = self.concurrency_limiter.acquire();
        self.pool.spawn(move || {
            ENCODER.with(|cell| {
                let mut encoder = cell.borrow_mut();
                let compressed_chunk = encoder
                    .compress(bytemuck::cast_slice(chunk_uncompressed.as_slice()))
                    .expect("blosc failed to compress");
                let compressed_chunk = Arc::<[u8]>::from(compressed_chunk.as_bytes());
                chunk_cell.set(compressed_chunk).ok().unwrap();
            });
            drop(permit);
        });

        self.current_chunk.reserve(TOKENS_PER_CHUNK);
    }

    pub fn build(mut self) -> Corpus {
        self.flush_chunk();
        Corpus {
            compressed_chunks: self
                .compressed_chunks
                .into_iter()
                .map(|cell| Arc::clone(cell.wait()))
                .collect(),
            token_pair_occurrence_counts: self
                .token_pair_occurrence_counts
                .into_iter()
                .map(|((first, second), count)| ((first, second), (count, first, second)))
                .collect(),
        }
    }
}

impl Default for CorpusBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct Corpus {
    compressed_chunks: Vec<Arc<[u8]>>,
    token_pair_occurrence_counts:
        PriorityQueue<(TokenId, TokenId), (usize, TokenId, TokenId), hashbrown::DefaultHashBuilder>,
}

impl Corpus {
    pub(super) fn most_frequent_token_pair(&self) -> Option<(TokenId, TokenId)> {
        let (&(first, second), &(count, ..)) = self.token_pair_occurrence_counts.peek()?;
        if count == 0 {
            None
        } else {
            Some((first, second))
        }
    }

    /// Updates the corpus in-place to apply the given token merge.
    /// Parallelized.
    pub(super) fn apply_merge(&mut self, merge: Merge) {
        assert_ne!(merge.first, DOC_TERMINATOR_TOKEN);
        assert_ne!(merge.second, DOC_TERMINATOR_TOKEN);
        assert_ne!(merge.merged, DOC_TERMINATOR_TOKEN);

        let token_pair_occurrence_count_deltas =
            DashMap::<(TokenId, TokenId), isize, hashbrown::DefaultHashBuilder>::default();

        self.compressed_chunks
            .par_iter_mut()
            .for_each(|compressed_chunk| {
                let mut v = None;
                let original_tokens = DECODER.with(|cell| {
                    let mut decoder = cell.borrow_mut();
                    let vec = decoder
                        .decompress(compressed_chunk)
                        .expect("blosc failed to decompress");
                    v = Some(vec);
                    bytemuck::cast_slice::<u8, TokenId>(v.as_ref().unwrap().as_slice())
                });
                let mut new_tokens = Vec::with_capacity(original_tokens.len());
                let mut did_change = false;
                let mut local_token_pair_occurrence_count_deltas =
                    HashMap::<(TokenId, TokenId), isize>::with_capacity(TOKENS_PER_CHUNK / 8);

                let mut i = 0;
                while i < original_tokens.len() {
                    let tok1 = original_tokens[i];
                    if let Some(tok2) = original_tokens.get(i + 1).copied()
                        && tok1 == merge.first
                        && tok2 == merge.second
                    {
                        // Match; apply merge.
                        did_change = true;
                        new_tokens.push(merge.merged);

                        // Update counts:
                        // -1 occurrence for the merged pair
                        *local_token_pair_occurrence_count_deltas
                            .entry((tok1, tok2))
                            .or_default() -= 1;

                        if let Some(tok0) = new_tokens.len().checked_sub(2).map(|i0| new_tokens[i0])
                            && tok0 != DOC_TERMINATOR_TOKEN
                        {
                            // -1 occurrence for (tok0, tok1)
                            *local_token_pair_occurrence_count_deltas
                                .entry((tok0, tok1))
                                .or_default() -= 1;
                            // +1 occurrence for (tok0, merged_token)
                            *local_token_pair_occurrence_count_deltas
                                .entry((tok0, merge.merged))
                                .or_default() += 1;
                        }

                        if let Some(tok3) = original_tokens.get(i + 2).copied()
                            && tok3 != DOC_TERMINATOR_TOKEN
                        {
                            // -1 occurrence for (tok2, tok)
                            *local_token_pair_occurrence_count_deltas
                                .entry((tok2, tok3))
                                .or_default() -= 1;

                            // +1 occurrence for (merged_token, tok3)
                            *local_token_pair_occurrence_count_deltas
                                .entry((merge.merged, tok3))
                                .or_default() += 1;
                        }

                        // Skip past tok2 in the original token stream.
                        i += 1;
                    } else {
                        new_tokens.push(tok1);
                    }

                    i += 1;
                }

                for ((first, second), delta) in local_token_pair_occurrence_count_deltas {
                    *token_pair_occurrence_count_deltas
                        .entry((first, second))
                        .or_default() += delta;
                }

                if did_change {
                    ENCODER.with(|cell| {
                        let mut encoder = cell.borrow_mut();
                        let data = encoder
                            .compress(bytemuck::cast_slice::<TokenId, u8>(&new_tokens))
                            .expect("blosc failed to compress");
                        *compressed_chunk = data.as_bytes().into();
                    });
                }
            });

        for ((first, second), delta) in token_pair_occurrence_count_deltas {
            let count = self
                .token_pair_occurrence_counts
                .get_priority(&(first, second))
                .copied()
                .map(|tup| tup.0)
                .unwrap_or(0);

            let new_count = count
                .checked_add_signed(delta)
                .expect("token count should not be negative");

            if new_count != count {
                self.token_pair_occurrence_counts
                    .push((first, second), (new_count, first, second));
            }
        }
    }
}
