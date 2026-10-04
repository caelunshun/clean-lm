//! BPE tokenizer training for an in-memory corpus.
//!
//! Does not use pre-tokenization, since those techniques have language/script assumptions.

use anyhow::ensure;

use crate::vocab::{Merge, SpecialTokenData, TextToken, Token, TokenId, Vocab};
use compact_bytes::CompactBytes;

mod corpus;

pub use corpus::{Compression, Corpus, CorpusBuilder};

/// Trains a vocabulary using a BPE algorithm.
///
/// `fused_merges` specifies how many merges can be applied
/// in one pass, without updating the token pair occurrence
/// table in between. Values >1 produce an approximation of the
/// BPE algorithm but may help amortize compression and memory bandwidth
/// overhead.
pub fn train_vocab<SP: SpecialTokenData>(
    mut corpus: Corpus,
    target_vocab_size: u32,
    fused_merges: usize,
) -> anyhow::Result<Vocab<SP>> {
    ensure!(
        target_vocab_size >= 256,
        "target vocab must be large enough for at least a token for each byte (256 entries)"
    );
    ensure!(target_vocab_size < u32::MAX);
    ensure!(fused_merges > 0);

    let mut vocab = Vocab::empty();

    // Initialize a token per byte
    for byte in 0..=u8::MAX {
        assert_eq!(
            vocab.add_token(Token::Text(TextToken(CompactBytes::new(&[byte]))))?,
            token_id_for_byte(byte)
        );
    }

    while vocab.num_tokens() < target_vocab_size {
        let num_merges = fused_merges.min((target_vocab_size - vocab.num_tokens()) as usize);

        let pairs = corpus.n_most_frequent_token_pairs(num_merges);

        if pairs.is_empty() {
            // No token pairs in corpus
            break;
        }

        let mut merges = Vec::with_capacity(pairs.len());

        for (first, second) in pairs {
            let mut merged_token_bytes = CompactBytes::new(vocab.token(first).text().unwrap());
            merged_token_bytes.extend_from_slice(vocab.token(second).text().unwrap());

            let merged = vocab.add_token(Token::Text(TextToken(merged_token_bytes)))?;
            let merge = Merge {
                first,
                second,
                merged,
            };
            vocab.add_merge(merge)?;

            merges.push(merge);

            tracing::debug!(
                vocab_size = vocab.num_tokens(),
                first = %vocab.token(first).text_token().unwrap(),
                second = %vocab.token(second).text_token().unwrap(),
                "applying merge"
            );
        }

        corpus.apply_merges(&merges);
        if fused_merges > 1 {
            tracing::debug!("end fused merge group");
        }
    }

    Ok(vocab)
}

const DOC_TERMINATOR: TokenId = TokenId::PLACEHOLDER;

fn token_id_for_byte(byte: u8) -> TokenId {
    TokenId::from_idx(byte as u32).unwrap()
}
