//! BPE tokenizer training for an in-memory corpus.
//!
//! The corpus can be compressed in memory with LZ4 or zstd
//! to reduce required capacity (and possibly reduce DRAM bandwidth
//! pressure).

use anyhow::ensure;
use dashmap::DashMap;
use rayon::prelude::*;
use std::borrow::Cow;

use compact_bytes::CompactBytes;
use priority_queue::PriorityQueue;

use crate::{
    text_tokenizer::TextTokenizer,
    training::corpus::decompress,
    vocab::{TextToken, Token, Vocab},
};

mod config;
mod corpus;

pub use config::InMemoryCompression;
pub use corpus::{Corpus, CorpusBuilder};

pub fn train_vocab(corpus: &Corpus, target_vocab_size: u32) -> anyhow::Result<Vocab> {
    ensure!(
        target_vocab_size >= 256,
        "target vocab must be large enough for at least a token for each byte (256 entries)"
    );
    ensure!(target_vocab_size < u32::MAX);

    Ok(Trainer::new(corpus, target_vocab_size).run())
}

struct Trainer<'a> {
    target_vocab_size: u32,
    corpus: &'a Corpus,
    tokens: Vec<CompactBytes>,
    tokenizer: TextTokenizer,

    pairs_heap: PriorityQueue<(u32, u32), usize, hashbrown::DefaultHashBuilder>,
}

fn token_id_for_byte(byte: u8) -> u32 {
    byte as u32
}

impl<'a> Trainer<'a> {
    fn new(corpus: &'a Corpus, target_vocab_size: u32) -> Self {
        let mut tokens = Vec::new();
        let mut tokenizer = TextTokenizer::empty();
        for tok in 0..=u8::MAX {
            tokens.push(CompactBytes::new(&[tok]));
            tokenizer.add_token(&[tok], token_id_for_byte(tok));
        }

        let mut pairs_heap =
            PriorityQueue::<(u32, u32), usize, hashbrown::DefaultHashBuilder>::default();
        for (&(tok_a, tok_b), &count) in &corpus.byte_pair_occurrences {
            if count > 0 {
                pairs_heap.push((token_id_for_byte(tok_a), token_id_for_byte(tok_b)), count);
            }
        }

        Self {
            target_vocab_size,
            corpus,
            tokens,
            tokenizer,
            pairs_heap,
        }
    }

    fn run(mut self) -> Vocab {
        while self.tokens.len() < self.target_vocab_size as usize {
            if !self.do_merge_step() {
                // No adjacent token pairs left
                break;
            }
        }

        Vocab::new(
            self.tokens
                .into_iter()
                .map(|text| Token::Text(TextToken(text)))
                .collect::<Vec<_>>(),
        )
    }

    /// Returns `false` if there are no token pairs left to merge
    fn do_merge_step(&mut self) -> bool {
        let Some((&(tok_a, tok_b), _)) = self.pairs_heap.peek() else {
            return false;
        };

        let new_token_text = CompactBytes::new(
            &self.tokens[tok_a as usize]
                .iter()
                .copied()
                .chain(self.tokens[tok_b as usize].iter().copied())
                .collect::<Vec<u8>>(),
        );
        let new_token_id = self.tokens.len() as u32;
        assert_ne!(new_token_id, u32::MAX);

        // Update token pair occurrences accordingly.
        let deltas = DashMap::<(u32, u32), isize, hashbrown::DefaultHashBuilder>::default();
        self.corpus.chunks.par_iter().for_each(|chunk| {
            let mut local_deltas = hashbrown::HashMap::<(u32, u32), isize>::default();

            let data = match self.corpus.compression {
                Some(compression) => Cow::Owned(decompress(&chunk.data, compression)),
                None => Cow::Borrowed(&*chunk.data),
            };

            let mut old_tokens = Vec::new();
            let mut new_segment = Vec::new();
            let mut prev_doc_end_index = 0;
            for &doc_end_index in &*chunk.doc_boundaries {
                let doc = &data[prev_doc_end_index as usize..doc_end_index as usize];
                old_tokens.clear();
                self.tokenizer.tokenize_into(doc, &mut old_tokens);
                self.accumulate_deltas(
                    doc,
                    &old_tokens,
                    (new_token_id, &new_token_text),
                    &mut new_segment,
                    &mut local_deltas,
                );

                prev_doc_end_index = doc_end_index;
            }

            for (tok_pair, delta) in local_deltas {
                *deltas.entry(tok_pair).or_default() += delta;
            }
        });

        self.tokenizer.add_token(&new_token_text, new_token_id);
        self.tokens.push(new_token_text);

        for (tok_pair, delta) in deltas {
            let old_count = self
                .pairs_heap
                .get_priority(&tok_pair)
                .copied()
                .unwrap_or(0);
            let new_count = old_count
                .checked_add_signed(delta)
                .expect("pair count went negative");
            if new_count == 0 {
                self.pairs_heap.remove(&tok_pair);
            } else {
                self.pairs_heap.push(tok_pair, new_count);
            }
        }
        assert!(self.pairs_heap.get_priority(&(tok_a, tok_b)).is_none());

        true
    }

    /// Accumulates the pair count changes within one document caused by
    /// adding a new token, given the document's tokenization `old_tokens`
    /// from before the addition.
    ///
    /// The longest match changes only at token boundaries where the text starts
    /// with the new token and the old token is shorter. From each such boundary,
    /// the document is re-segmented until it realigns with an old boundary
    /// where the longest match is unchanged.
    fn accumulate_deltas(
        &self,
        doc: &[u8],
        old_tokens: &[u32],
        (new_token_id, new_token_text): (u32, &[u8]),
        new_segment: &mut Vec<u32>,
        deltas: &mut hashbrown::HashMap<(u32, u32), isize>,
    ) {
        let token_len = |token: u32| self.tokens[token as usize].len();
        let longest_match_changes = |pos: usize, old_len: usize| {
            old_len < new_token_text.len() && doc[pos..].starts_with(new_token_text)
        };

        let mut i = 0;
        let mut pos = 0;
        while i < old_tokens.len() {
            if !longest_match_changes(pos, token_len(old_tokens[i])) {
                pos += token_len(old_tokens[i]);
                i += 1;
                continue;
            }

            let segment_start = i;
            new_segment.clear();
            let mut new_pos = pos;
            loop {
                let (mut token, mut len) = self.tokenizer.longest_prefix(&doc[new_pos..]);
                if longest_match_changes(new_pos, len) {
                    (token, len) = (new_token_id, new_token_text.len());
                }
                new_segment.push(token);
                new_pos += len;

                while pos < new_pos {
                    pos += token_len(old_tokens[i]);
                    i += 1;
                }
                if pos == new_pos
                    && (i == old_tokens.len()
                        || !longest_match_changes(pos, token_len(old_tokens[i])))
                {
                    break;
                }
            }

            let old_segment = &old_tokens[segment_start..i];
            if segment_start > 0 {
                let left = old_tokens[segment_start - 1];
                *deltas.entry((left, old_segment[0])).or_default() -= 1;
                *deltas.entry((left, new_segment[0])).or_default() += 1;
            }
            if i < old_tokens.len() {
                let right = old_tokens[i];
                *deltas
                    .entry((*old_segment.last().unwrap(), right))
                    .or_default() -= 1;
                *deltas
                    .entry((*new_segment.last().unwrap(), right))
                    .or_default() += 1;
            }
            for pair in old_segment.windows(2) {
                *deltas.entry((pair[0], pair[1])).or_default() -= 1;
            }
            for pair in new_segment.windows(2) {
                *deltas.entry((pair[0], pair[1])).or_default() += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple() {
        let mut corpus = Corpus::builder(None);
        corpus.add_document(b"The cat sat on the mat and was really happy when the mat came to arrive at the matrix multiplier. The cat was really happy about this.").unwrap();
        let corpus = corpus.build();

        let vocab = train_vocab(&corpus, 300).unwrap();
        println!("{}", serde_json::to_string_pretty(&vocab).unwrap());
    }

    fn random_docs(seed: u64, num_docs: usize, alphabet: &[u8]) -> Vec<Vec<u8>> {
        let mut rng = seed;
        let mut next = move |n: usize| {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng as usize % n
        };
        (0..num_docs)
            .map(|_| {
                (0..next(200))
                    .map(|_| alphabet[next(alphabet.len())])
                    .collect()
            })
            .collect()
    }

    fn build_corpus(docs: &[Vec<u8>], compression: Option<InMemoryCompression>) -> Corpus {
        let mut corpus = Corpus::builder(compression);
        for doc in docs {
            corpus.add_document(doc).unwrap();
        }
        corpus.build()
    }

    fn recount_pairs(trainer: &Trainer) -> hashbrown::HashMap<(u32, u32), usize> {
        let mut counts = hashbrown::HashMap::new();
        for chunk in &trainer.corpus.chunks {
            let data = match trainer.corpus.compression {
                Some(compression) => decompress(&chunk.data, compression),
                None => chunk.data.to_vec(),
            };
            let mut prev_doc_end_index = 0;
            for &doc_end_index in &*chunk.doc_boundaries {
                let tokens = trainer
                    .tokenizer
                    .tokenize_to_vec(&data[prev_doc_end_index as usize..doc_end_index as usize]);
                for pair in tokens.windows(2) {
                    *counts.entry((pair[0], pair[1])).or_default() += 1;
                }
                prev_doc_end_index = doc_end_index;
            }
        }
        counts
    }

    fn assert_pair_counts_exact(docs: &[Vec<u8>], compression: Option<InMemoryCompression>) {
        let corpus = build_corpus(docs, compression);
        let mut trainer = Trainer::new(&corpus, u32::MAX - 1);
        loop {
            let heap = trainer
                .pairs_heap
                .iter()
                .map(|(&pair, &count)| (pair, count))
                .collect::<hashbrown::HashMap<_, _>>();
            assert_eq!(
                heap,
                recount_pairs(&trainer),
                "after {} tokens",
                trainer.tokens.len()
            );
            if !trainer.do_merge_step() {
                break;
            }
        }
    }

    #[test]
    fn pair_counts_exact_for_repeats() {
        assert_pair_counts_exact(
            &[
                b"xababy".to_vec(),
                b"xaaay".to_vec(),
                b"aaaaaaa".to_vec(),
                vec![0xFF, 0xFF, 0xFF],
            ],
            None,
        );
    }

    #[test]
    fn pair_counts_exact_when_greedy_resegments() {
        // "he" is merged before "th", after which "the." is re-segmented
        // from [t, he, .] into [th, e, .].
        let mut doc = Vec::new();
        doc.extend(b"he.x".repeat(30));
        doc.extend(b"the.".repeat(5));
        doc.extend(b"thy.".repeat(20));
        assert_pair_counts_exact(&[doc], None);
    }

    #[test]
    fn pair_counts_exact_for_random_corpus() {
        assert_pair_counts_exact(&random_docs(1, 200, b"aab c"), None);
        assert_pair_counts_exact(&random_docs(2, 100, b"abcd \n"), None);
    }

    #[test]
    fn pair_counts_exact_across_compressed_chunks() {
        let docs = random_docs(3, 2000, b"aaabbc d");
        assert!(docs.iter().map(Vec::len).sum::<usize>() > 2 * corpus::CHUNK_SIZE);
        let corpus = build_corpus(&docs, Some(InMemoryCompression::Lz4));
        let mut trainer = Trainer::new(&corpus, u32::MAX - 1);
        for _ in 0..50 {
            assert!(trainer.do_merge_step());
        }
        let heap = trainer
            .pairs_heap
            .iter()
            .map(|(&pair, &count)| (pair, count))
            .collect::<hashbrown::HashMap<_, _>>();
        assert_eq!(heap, recount_pairs(&trainer));
    }

    #[test]
    fn stops_when_no_pairs_left() {
        let corpus = build_corpus(&[b"ab".to_vec()], None);
        let vocab = train_vocab(&corpus, 1000).unwrap();
        assert_eq!(vocab.num_tokens(), 257);
    }

    #[test]
    fn trained_vocab_roundtrips_without_duplicates() {
        let docs = random_docs(4, 300, b"aab c\n\xFF");
        let corpus = build_corpus(&docs, None);
        let vocab = train_vocab(&corpus, 1000).unwrap();

        let token_texts = vocab
            .tokens()
            .map(|(_, token)| match token {
                Token::Text(text) => text.0.as_slice().to_vec(),
                Token::Special(_) => unreachable!(),
            })
            .collect::<Vec<_>>();
        let unique = token_texts.iter().collect::<hashbrown::HashSet<_>>();
        assert_eq!(unique.len(), token_texts.len());

        let tokenizer = TextTokenizer::new(&vocab);
        for doc in &docs {
            let decoded = tokenizer
                .tokenize_to_vec(doc)
                .into_iter()
                .flat_map(|id| token_texts[id as usize].iter().copied())
                .collect::<Vec<_>>();
            assert_eq!(&decoded, doc);
        }
    }

    #[test]
    fn byte_pairs_counted_within_documents() {
        let corpus = build_corpus(&[b"a".to_vec(), b"b".to_vec(), vec![0xFF, 0xFF]], None);
        assert_eq!(corpus.byte_pair_occurrences[&(b'a', b'b')], 0);
        assert_eq!(corpus.byte_pair_occurrences[&(0xFF, 0xFF)], 1);
    }
}
