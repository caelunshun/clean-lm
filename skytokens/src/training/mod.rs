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
        target_vocab_size >= u8::MAX as u32,
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
            pairs_heap.push((token_id_for_byte(tok_a), token_id_for_byte(tok_b)), count);
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
            self.do_merge_step();
        }

        Vocab::new(
            self.tokens
                .into_iter()
                .map(|text| Token::Text(TextToken(text)))
                .collect::<Vec<_>>(),
        )
    }

    fn do_merge_step(&mut self) {
        let Some(((tok_a, tok_b), _)) = self.pairs_heap.pop() else {
            return;
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

            let mut prev_doc_end_index = 0;
            for &doc_end_index in &*chunk.doc_boundaries {
                let mut prev_token = None;
                let mut prev_prev_token = None;
                self.tokenizer.tokenize(
                    &data[prev_doc_end_index as usize..doc_end_index as usize],
                    |token| {
                        if let Some(prev_token) = prev_token
                            && let Some(prev_prev_token) = prev_prev_token
                        {
                            if prev_token == tok_a && token == tok_b {
                                *local_deltas
                                    .entry((prev_prev_token, prev_token))
                                    .or_default() -= 1;
                                *local_deltas
                                    .entry((prev_prev_token, new_token_id))
                                    .or_default() += 1;
                            } else if prev_prev_token == tok_a && prev_token == tok_b {
                                *local_deltas.entry((prev_token, token)).or_default() -= 1;
                                *local_deltas.entry((new_token_id, token)).or_default() += 1;
                            }
                        }
                        prev_prev_token = prev_token;
                        prev_token = Some(token);
                    },
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
            self.pairs_heap
                .push(tok_pair, old_count.checked_add_signed(delta).unwrap_or(0));
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
}
