//! BPE tokenizer training for an in-memory corpus.
//!
//! Does not use pre-tokenization, since those techniques have language/script assumptions.

use anyhow::ensure;

use crate::vocab::{Merge, SpecialTokenData, TextToken, Token, TokenId, Vocab};
use compact_bytes::CompactBytes;

mod corpus;

pub use corpus::{Corpus, CorpusBuilder};

pub fn train_vocab<SP: SpecialTokenData>(
    corpus: &Corpus,
    target_vocab_size: u32,
) -> anyhow::Result<Vocab<SP>> {
    ensure!(
        target_vocab_size >= 256,
        "target vocab must be large enough for at least a token for each byte (256 entries)"
    );
    ensure!(target_vocab_size < u32::MAX);

    let mut corpus = corpus.clone();
    let mut vocab = Vocab::empty();

    // Initialize a token per byte
    for byte in 0..=u8::MAX {
        assert_eq!(
            vocab.add_token(Token::Text(TextToken(CompactBytes::new(&[byte]))))?,
            token_id_for_byte(byte)
        );
    }

    while vocab.num_tokens() < target_vocab_size {
        let Some((first, second)) = corpus.most_frequent_token_pair() else {
            // No token pairs in corpus
            break;
        };

        let mut merged_token_bytes = CompactBytes::new(vocab.token(first).text().unwrap());
        merged_token_bytes.extend_from_slice(vocab.token(second).text().unwrap());

        let merged = vocab.add_token(Token::Text(TextToken(merged_token_bytes)))?;
        let merge = Merge {
            first,
            second,
            merged,
        };
        vocab.add_merge(merge)?;

        corpus.apply_merge(merge);
    }

    Ok(vocab)
}

const DOC_TERMINATOR_TOKEN: TokenId = TokenId::PLACEHOLDER;

fn token_id_for_byte(byte: u8) -> TokenId {
    TokenId::from_idx(byte as u32).unwrap()
}
