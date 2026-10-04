//! Corpus is represented as chunks of tokens, where documents
//! cannot be spread across chunks, and each chunk is optionally blosc-compressed.
//! Documents within a chunk are separated by DOC_TERMINATOR_TOKEN.

use std::{cell::RefCell, iter, mem, sync::Arc, thread::available_parallelism};

use anyhow::ensure;
use blosc2::{
    CParams, CompressAlgo, DParams, SplitMode,
    chunk::{Decoder, Encoder},
};
use hashbrown::HashMap;
use priority_queue::PriorityQueue;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use simple_semaphore::Semaphore;
use thread_local::ThreadLocal;

use crate::{
    training::{DOC_TERMINATOR, token_id_for_byte},
    vocab::{Merge, TokenId},
};

const TOKENS_PER_CHUNK: usize = 64 * 1024;

/// Compression applied to corpus chunks held in memory during training.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum Compression {
    None,
    #[default]
    Lz4,
    BloscLz,
}

fn new_encoder(algo: CompressAlgo) -> Encoder {
    let mut params = CParams::default();
    params
        .nthreads(1)
        .typesize(size_of::<TokenId>())
        .unwrap()
        .splitmode(SplitMode::Never)
        .compressor(algo);
    Encoder::new(params).expect("failed to init blosc")
}

thread_local! {
    static LZ4_ENCODER: RefCell<Encoder> = RefCell::new(new_encoder(CompressAlgo::Lz4));
    static BLOSCLZ_ENCODER: RefCell<Encoder> = RefCell::new(new_encoder(CompressAlgo::Blosclz));
}

thread_local! {
    static DECODER: RefCell<Decoder> = {
        let mut params = DParams::default();
        params.nthreads(1);

        RefCell::new(Decoder::new(params).expect("failed to init blosc"))
    };
}

/// Holds blosc-compressed data for a chunk if compression is enabled,
/// otherwise raw packed TokenIds.
type ChunkData = Vec<u8>;

fn compress_chunk_into(compression: Compression, tokens: &[TokenId], dst: &mut Vec<u8>) {
    let encoder = match compression {
        Compression::None => return dst.extend_from_slice(bytemuck::cast_slice(tokens)),
        Compression::Lz4 => &LZ4_ENCODER,
        Compression::BloscLz => &BLOSCLZ_ENCODER,
    };

    // Blosc's maximum overhead, see docs on Encoder::compress
    let max_compressed_size = size_of_val::<[TokenId]>(tokens) + 32;

    if dst.capacity() == 0 {
        // Compression will always fail with a zero-capacity buffer,
        // so pre-allocate the required space.
        dst.reserve_exact(max_compressed_size);
    }

    encoder.with(|cell| {
        let mut encoder = cell.borrow_mut();

        let compressed_size =
            match encoder.compress_into(bytemuck::cast_slice(tokens), dst.spare_capacity_mut()) {
                Ok(sz) => sz,
                Err(blosc2::Error::MaxBufsizeExceeded | blosc2::Error::WriteBuffer) => {
                    dst.reserve_exact(max_compressed_size);
                    encoder
                        .compress_into(bytemuck::cast_slice(tokens), dst.spare_capacity_mut())
                        .expect("blosc failed to compress")
                }
                Err(e) => panic!("blosc failed to compress: {e}"),
            };

        // Soundness: we trust blosc's returned compressed_size.
        unsafe {
            dst.set_len(dst.len() + compressed_size);
        }

        dst.shrink_to_fit();
    })
}

#[derive(Debug)]
struct BuiltChunk {
    data: ChunkData,
    token_pair_occurrence_counts: HashMap<(TokenId, TokenId), usize>,
}

#[derive(Debug)]
pub struct CorpusBuilder {
    compression: Compression,
    chunks: Vec<flume::Receiver<BuiltChunk>>,
    concurrency_limiter: Arc<Semaphore>,
    current_chunk: Vec<TokenId>,
    pool: rayon::ThreadPool,
    num_tokens: usize,
}

impl CorpusBuilder {
    pub fn new() -> Self {
        Self::with_compression(Compression::default())
    }

    pub fn with_compression(compression: Compression) -> Self {
        let threads = available_parallelism().unwrap().get();
        Self {
            compression,
            chunks: Default::default(),
            concurrency_limiter: Semaphore::new(4 * threads),
            current_chunk: Default::default(),
            pool: rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap(),
            num_tokens: 0,
        }
    }

    pub fn add_document(&mut self, document: impl AsRef<[u8]>) -> anyhow::Result<()> {
        ensure!((document.as_ref().len()) < u32::MAX as usize);
        self.current_chunk.extend(
            document
                .as_ref()
                .iter()
                .copied()
                .map(token_id_for_byte)
                .chain(iter::once(DOC_TERMINATOR)),
        );
        self.num_tokens += document.as_ref().len();

        if self.current_chunk.len() >= TOKENS_PER_CHUNK {
            self.flush_chunk();
        }
        Ok(())
    }

    fn flush_chunk(&mut self) {
        if self.current_chunk.is_empty() {
            return;
        }

        let (chunk_tx, chunk) = flume::bounded(1);
        self.chunks.push(chunk);

        let chunk_uncompressed = mem::take(&mut self.current_chunk);
        let compression = self.compression;
        let permit = self.concurrency_limiter.acquire();
        self.pool.spawn(move || {
            let mut buf = Vec::new();
            compress_chunk_into(compression, &chunk_uncompressed, &mut buf);

            let mut token_pair_occurrence_counts = HashMap::<(TokenId, TokenId), usize>::default();
            for &[first, second] in chunk_uncompressed.array_windows::<2>() {
                if first != DOC_TERMINATOR && second != DOC_TERMINATOR {
                    *token_pair_occurrence_counts
                        .entry((first, second))
                        .or_default() += 1;
                }
            }

            chunk_tx
                .send(BuiltChunk {
                    data: buf,
                    token_pair_occurrence_counts,
                })
                .ok()
                .unwrap();
            drop(permit);
        });

        self.current_chunk.reserve(TOKENS_PER_CHUNK);
    }

    pub fn build(mut self) -> Corpus {
        self.flush_chunk();
        let (chunks, token_pair_occurrence_counts): (Vec<_>, Vec<_>) = self
            .chunks
            .into_iter()
            .map(|cell| cell.recv().unwrap())
            .map(
                |BuiltChunk {
                     data,
                     token_pair_occurrence_counts,
                 }| (data, token_pair_occurrence_counts),
            )
            .collect();

        let token_pair_occurrence_counts = token_pair_occurrence_counts.into_par_iter().reduce(
            HashMap::default,
            |mut first, second| {
                for (pair, count) in second {
                    *first.entry(pair).or_default() += count;
                }
                first
            },
        );

        tracing::info!(
            num_tokens = self.num_tokens,
            num_chunks = chunks.len(),
            "built corpus"
        );

        Corpus {
            compression: self.compression,
            chunks,
            token_pair_occurrence_counts: token_pair_occurrence_counts
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
    compression: Compression,
    chunks: Vec<ChunkData>,
    token_pair_occurrence_counts:
        PriorityQueue<(TokenId, TokenId), (usize, TokenId, TokenId), hashbrown::DefaultHashBuilder>,
}

impl Corpus {
    pub fn stored_bytes(&self) -> usize {
        self.chunks.iter().map(|chunk| chunk.len()).sum()
    }

    /// Returns up to `n` of the most frequent token pairs,
    /// ordered from most to least frequent.
    pub(super) fn n_most_frequent_token_pairs(&mut self, n: usize) -> Vec<(TokenId, TokenId)> {
        let mut pairs = Vec::new();
        let mut counts = Vec::new();

        while pairs.len() < n
            && let Some(((first, second), (count, ..))) = self.token_pair_occurrence_counts.pop()
        {
            pairs.push((first, second));
            counts.push(count);
        }

        // Add back to heap
        for (&pair, &count) in pairs.iter().zip(&counts) {
            self.token_pair_occurrence_counts
                .push(pair, (count, pair.0, pair.1));
        }

        pairs
    }

    /// Updates the corpus in-place to apply the given sequence of token merges.
    /// Parallelized, but each merge runs in sequence.
    /// The cost of compressing and decompressing each chunk, as well as loading
    /// from DRAM into cache (provided some cache level is big enough to hold
    /// uncompressed tokens of a chunk + bookkeeping and overhead),
    /// is amortized across the provided merges.
    pub(super) fn apply_merges(&mut self, merges: &[Merge]) {
        for merge in merges {
            assert_ne!(merge.first, DOC_TERMINATOR);
            assert_ne!(merge.second, DOC_TERMINATOR);
            assert_ne!(merge.merged, DOC_TERMINATOR);
        }

        let compression = self.compression;

        let thr_match_buffer = ThreadLocal::<RefCell<Vec<u32>>>::new();

        // In parallel, map each chunk to its contribution to the per-token-pair count deltas
        // by applying the merge. Then reduce the delta maps across chunks,
        // producing a final summed delta map.
        let token_pair_occurrence_count_deltas: HashMap<(TokenId, TokenId), isize> = self
            .chunks
            .par_iter_mut()
            .map(|chunk| {
                let chunk = RefCell::new(chunk);
                let mut chunk_guard = None;

                let mut decompressed;
                let mut tokens: &mut [TokenId] = if compression == Compression::None {
                    let guard = chunk_guard.insert(chunk.borrow_mut());
                    bytemuck::cast_slice_mut(&mut *guard)
                } else {
                    decompressed = DECODER.with(|cell| {
                        cell.borrow_mut()
                            .decompress(&chunk.borrow())
                            .expect("blosc failed to decompress")
                    });
                    bytemuck::cast_slice_mut(&mut decompressed)
                };

                let mut did_change = false;
                let mut token_pair_occurrence_count_deltas =
                    HashMap::<(TokenId, TokenId), isize>::with_capacity(TOKENS_PER_CHUNK / 8);

                let mut match_buffer = thr_match_buffer.get_or_default().borrow_mut();

                for merge in merges {
                    let max_matches = tokens.len() - 1;
                    match_buffer.resize(max_matches, u32::MAX);

                    let num_matches =
                        scan_merge_matches(tokens, &mut match_buffer, merge.first, merge.second);
                    did_change |= num_matches > 0;

                    // Apply merges. Applied matches are compacted into the front of
                    // `match_buffer` (overwriting already-visited entries) for use by the
                    // compaction pass below.
                    let mut num_applied = 0usize;
                    for m in 0..num_matches {
                        let i = match_buffer[m] as usize;
                        let prev_i = num_applied.checked_sub(1).map(|a| match_buffer[a] as usize);

                        // In the case of overlapping matches, the leftmost takes priority, so skip
                        // the one following it.
                        if let Some(i_minus_1) = i.checked_sub(1)
                            && prev_i == Some(i_minus_1)
                        {
                            continue;
                        }

                        let tok1 = tokens[i];
                        let tok2 = tokens[i + 1];

                        // Update counts:
                        // -1 occurrence for the merged pair
                        *token_pair_occurrence_count_deltas
                            .entry((tok1, tok2))
                            .or_default() -= 1;

                        if let Some(i_minus_1) = i.checked_sub(1) {
                            // Adjust token before the merged pair.
                            let tok0 = if let Some(i_minus_2) = i.checked_sub(2)
                                && prev_i == Some(i_minus_2)
                            {
                                // - Need to handle the case where the immediately preceding
                                // token was just merged.
                                merge.merged
                            } else {
                                tokens[i_minus_1]
                            };

                            if tok0 != DOC_TERMINATOR {
                                // -1 occurrence for (tok0, tok1)
                                *token_pair_occurrence_count_deltas
                                    .entry((tok0, tok1))
                                    .or_default() -= 1;
                                // +1 occurrence for (tok0, merged_token)
                                *token_pair_occurrence_count_deltas
                                    .entry((tok0, merge.merged))
                                    .or_default() += 1;
                            }
                        }

                        if let Some(tok3) = tokens.get(i + 2).copied()
                            && tok3 != DOC_TERMINATOR
                        {
                            // -1 occurrence for (tok2, tok)
                            *token_pair_occurrence_count_deltas
                                .entry((tok2, tok3))
                                .or_default() -= 1;

                            // +1 occurrence for (merged_token, tok3)
                            *token_pair_occurrence_count_deltas
                                .entry((merge.merged, tok3))
                                .or_default() += 1;
                        }

                        tokens[i] = merge.merged;
                        // tokens[i + 1] will be eliminated by compaction below.
                        match_buffer[num_applied] = i as u32;
                        num_applied += 1;
                    }

                    // Now that we've applied the matches, we need to compact
                    // the token buffer by removing those tokens which were matches.
                    // `removed` stores the number of tokens logically removed from the tokens buffer so far,
                    // which affects how `i` (an index into the _original_ token buffer
                    // before applying this merge) maps to destination indexes in the overwritten
                    // buffer. See Vec::retain for a similar pattern.
                    let mut removed = 0;
                    let mut applied = match_buffer[..num_applied]
                        .iter()
                        .map(|&j| j as usize)
                        .peekable();
                    let mut i = 0;
                    while i < tokens.len() {
                        tokens[i - removed] = tokens[i];

                        if applied.next_if_eq(&i).is_some() {
                            // Next token is removed, so skip.
                            removed += 1;
                            i += 1;
                        }

                        i += 1;
                    }

                    // Shrink our tokens view to exclude the leftover space.
                    let l = tokens.len();
                    tokens = &mut tokens[..l - removed];
                }

                if did_change {
                    match compression {
                        Compression::None => {
                            // We modified the tokens buffer in-place, so we can just truncate it.
                            let remaining_size = size_of_val::<[TokenId]>(tokens);
                            drop(chunk_guard);
                            chunk.borrow_mut().truncate(remaining_size);
                        }
                        _ => {
                            // Recompress, reusing the chunk's allocation.
                            // Borrow can't fail because on the compressed path,
                            // we don't initialize chunk_guard.
                            chunk.borrow_mut().clear();
                            compress_chunk_into(compression, tokens, &mut chunk.borrow_mut());
                        }
                    }
                }

                token_pair_occurrence_count_deltas
            })
            .reduce(HashMap::default, |mut first, second| {
                for (pair, delta) in second {
                    *first.entry(pair).or_default() += delta;
                }
                first
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
                if new_count > 0 {
                    self.token_pair_occurrence_counts
                        .push((first, second), (new_count, first, second));
                } else {
                    self.token_pair_occurrence_counts.remove(&(first, second));
                }
            }
        }
    }
}

/// Builds a list of all positions `i` in the given token buffer
/// where `tokens[i] == first && tokens[i+1] == second`. The provided
/// positions list must have sufficient capacity to hold the maximum
/// number of possible indexes. Return match count.
fn scan_merge_matches(
    tokens: &[TokenId],
    matches: &mut [u32],
    first: TokenId,
    second: TokenId,
) -> usize {
    let max_matches = tokens.len() - 1;
    assert!(matches.len() >= max_matches);
    assert!(tokens.len() < u32::MAX as usize);

    let tokens: &[u32] = bytemuck::cast_slice(tokens);

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx512f") {
        return unsafe { scan_merge_matches_avx512(tokens, matches, first.idx(), second.idx()) };
    }

    scan_merge_matches_fallback(tokens, matches, first.idx(), second.idx())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn scan_merge_matches_avx512(
    tokens: &[u32],
    matches: &mut [u32],
    first: u32,
    second: u32,
) -> usize {
    use core::arch::x86_64::*;

    let mut match_count = 0;

    let (head, middle, tail) = unsafe { tokens.align_to::<__m512i>() };
    if middle.is_empty() {
        return scan_merge_matches_fallback(tokens, matches, first, second);
    }

    if !head.is_empty() {
        for (i, [a, b]) in head
            .array_windows::<2>()
            .copied()
            .chain(iter::once([
                head.last().copied().unwrap(),
                tokens[head.len()],
            ]))
            .enumerate()
        {
            if a == first && b == second {
                matches[match_count] = i as u32;
                match_count += 1;
            }
        }
    }

    // SIMD loop
    unsafe {
        let v_first = _mm512_set1_epi32(first as i32);
        let v_second = _mm512_set1_epi32(second as i32);

        for (i, &v_toks) in middle.iter().enumerate() {
            let tail_match = if i < middle.len() - 1 || !tail.is_empty() {
                tokens[head.len() + (i + 1) * 16] == second
            } else {
                false
            };

            let m1 = _mm512_cmpeq_epi32_mask(v_toks, v_first);
            let m2 = _mm512_cmpeq_epi32_mask(v_toks, v_second) >> 1 | ((tail_match as u16) << 15);

            let m_match = m1 & m2;

            let base = (head.len() + i * 16) as u32 as i32;
            let indexes = _mm512_setr_epi32(
                base,
                base + 1,
                base + 2,
                base + 3,
                base + 4,
                base + 5,
                base + 6,
                base + 7,
                base + 8,
                base + 9,
                base + 10,
                base + 11,
                base + 12,
                base + 13,
                base + 14,
                base + 15,
            );

            let v_match_count = m_match.count_ones();
            let store_mask = ((1 << v_match_count) - 1) as u16;

            let compressed_indexes = _mm512_maskz_compress_epi32(m_match, indexes);

            _mm512_mask_storeu_epi32(
                matches.as_mut_ptr().add(match_count).cast(),
                store_mask,
                compressed_indexes,
            );
            match_count += v_match_count as usize;
        }
    }

    if !tail.is_empty() {
        for (i, [a, b]) in tail.array_windows::<2>().copied().enumerate() {
            let i = i + head.len() + 16 * middle.len();

            if a == first && b == second {
                matches[match_count] = i as u32;
                match_count += 1;
            }
        }
    }

    match_count
}

fn scan_merge_matches_fallback(
    tokens: &[u32],
    matches: &mut [u32],
    first: u32,
    second: u32,
) -> usize {
    let mut match_count = 0;
    for (i, &[a, b]) in tokens.array_windows::<2>().enumerate() {
        if a == first && b == second {
            matches[match_count] = i as u32;
            match_count += 1;
        }
    }
    match_count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scan_merge_matches() {
        let mut buf = Vec::new();
        for _ in 0..1000 {
            buf.extend([
                TokenId::from_idx(0).unwrap(),
                TokenId::from_idx(1).unwrap(),
                TokenId::from_idx(2).unwrap(),
                TokenId::from_idx(3).unwrap(),
            ]);
        }

        let mut match_buffer = vec![0u32; buf.len() - 1];
        let count = scan_merge_matches(
            &buf,
            &mut match_buffer,
            TokenId::from_idx(3).unwrap(),
            TokenId::from_idx(0).unwrap(),
        );
        assert_eq!(count, 999);

        for (i, idx) in match_buffer[..count].iter().copied().enumerate() {
            assert_eq!(idx, (i * 4 + 3) as u32);
        }
    }
}
