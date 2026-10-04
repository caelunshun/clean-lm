use crate::vocab::{SpecialTokenData, TokenId, Vocab};
use bit_vec::BitVec;
use hashbrown::HashMap;
use priority_queue::PriorityQueue;
use std::cmp;

/// Merge-style tokenizer implementation for a Vocab.
#[derive(Debug, Clone)]
pub struct TextTokenizer {
    /// u8 index => corresponding token, TokenId::PLACEHOLDER
    /// if a byte has no token
    byte_tokens: Vec<TokenId>,
    /// Merged token pair => index in the order of merges for the merge.
    /// Values are indexes into merge_result_tokens
    merge_order: HashMap<(TokenId, TokenId), u32>,
    merge_result_tokens: Vec<TokenId>,
}

impl TextTokenizer {
    pub fn new<SP: SpecialTokenData>(vocab: &Vocab<SP>) -> Self {
        let mut byte_tokens = Vec::new();
        for byte in 0..=u8::MAX {
            byte_tokens.push(vocab.text_to_id([byte]).unwrap_or(TokenId::PLACEHOLDER));
        }

        let mut merge_order = HashMap::default();
        let mut merge_result_tokens = Vec::new();

        for merge in vocab.merges() {
            merge_order.insert(
                (merge.first, merge.second),
                merge_result_tokens.len() as u32,
            );
            merge_result_tokens.push(merge.merged);
        }

        Self {
            byte_tokens,
            merge_order,
            merge_result_tokens,
        }
    }

    pub fn tokenize(&self, text: impl AsRef<[u8]>) -> Result<Vec<TokenId>, TokenizationFailure> {
        let text = text.as_ref();

        if text.len() > u32::MAX as usize - 1 {
            return Err(TokenizationFailure::TextTooLong);
        }

        let mut tokens: Vec<TokenId> = text
            .iter()
            .copied()
            .map(|byte| {
                let tok = self.byte_tokens[byte as usize];
                if tok == TokenId::PLACEHOLDER {
                    Err(TokenizationFailure::MissingTokenForByte(byte))
                } else {
                    Ok(tok)
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut valid_tokens_set = BitVec::<usize>::from_elem_general(tokens.len(), true);

        let mut merge_queue = PriorityQueue::<(TokenId, TokenId), cmp::Reverse<u32>>::new();
        for &[first, second] in tokens.array_windows::<2>() {
            if let Some(&merge_idx) = self.merge_order.get(&(first, second)) {
                merge_queue.push((first, second), cmp::Reverse(merge_idx));
            }
        }

        while let Some(((first, second), cmp::Reverse(merge_idx))) = merge_queue.pop() {
            let merged_token = self.merge_result_tokens[merge_idx as usize];

            let mut pos = 0;
            while let Some(i1) = tokens[pos..tokens.len() - 1]
                .iter()
                .position(|t| *t == first)
            {
                let i1 = pos + i1;

                let Some(i2) = (i1 + 1..tokens.len()).find(|&i2| valid_tokens_set[i2]) else {
                    break;
                };

                if tokens[i2] != second {
                    pos = i2;
                    continue;
                }

                tokens[i1] = merged_token;
                tokens[i2] = TokenId::PLACEHOLDER;
                valid_tokens_set.set(i2, false);

                if let Some(i0) = (0..i1).rfind(|&i0| valid_tokens_set[i0]) {
                    debug_assert_ne!(tokens[i0], TokenId::PLACEHOLDER);

                    if let Some(&next_merge_idx) = self.merge_order.get(&(tokens[i0], merged_token))
                        && next_merge_idx > merge_idx
                    {
                        merge_queue.push((tokens[i0], merged_token), cmp::Reverse(next_merge_idx));
                    }
                }

                if let Some(i3) = (i2 + 1..tokens.len()).find(|&i3| valid_tokens_set[i3]) {
                    debug_assert_ne!(tokens[i3], TokenId::PLACEHOLDER);

                    if let Some(&next_merge_idx) = self.merge_order.get(&(merged_token, tokens[i3]))
                        && next_merge_idx > merge_idx
                    {
                        merge_queue.push((merged_token, tokens[i3]), cmp::Reverse(next_merge_idx));
                    }

                    pos = i3;
                } else {
                    break;
                }
            }
        }

        tokens.retain(|t| *t != TokenId::PLACEHOLDER);
        Ok(tokens)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum TokenizationFailure {
    #[error("missing token for byte encountered in text: {0:x}")]
    MissingTokenForByte(u8),
    #[error("maximum supported text length is u32::MAX - 1")]
    TextTooLong,
}
