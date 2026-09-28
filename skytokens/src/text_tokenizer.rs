use hashbrown::HashMap;

use crate::vocab::{Token, TokenId, Vocab};

/// Fast text tokenizer for a vocabulary,
/// backed by a hash-array mapped trie with instruction-parallel
/// lookahead to reduce pointer chasing.
#[derive(Debug, Clone)]
pub struct TextTokenizer {
    nodes: Vec<Node>,
    root: NodeIdx,
}

type NodeIdx = u32;

#[derive(Debug, Clone)]
struct Node {
    /// Might be a sentinel value (u32::MAX)
    /// if no token defined at this node; in such
    /// case `anchor_distance > 0`.
    token: TokenId,
    /// Pointer to last ancestor node that has a valid token.
    anchor: NodeIdx,
    /// Path length to node `anchor`.
    anchor_distance: u32,
    next1: HashMap<u8, NodeIdx>,
    next2: HashMap<[u8; 2], NodeIdx>,
    next3: HashMap<[u8; 3], NodeIdx>,
    next4: HashMap<[u8; 4], NodeIdx>,
}

impl TextTokenizer {
    /// Note that special tokens are ignored.
    pub fn new(vocab: &Vocab) -> Self {
        let mut tokenizer = Self::empty();
        for (token_id, token) in vocab.tokens() {
            if let Token::Text(text_token) = token {
                tokenizer.add_token(&text_token.0, token_id);
            }
        }
        tokenizer
    }

    pub(crate) fn empty() -> Self {
        Self {
            nodes: [Node {
                token: u32::MAX,
                anchor: 0,
                anchor_distance: 0,
                next1: Default::default(),
                next2: Default::default(),
                next3: Default::default(),
                next4: Default::default(),
            }]
            .to_vec(),
            root: 0,
        }
    }

    pub(crate) fn add_token(&mut self, token_text: &[u8], token_id: TokenId) {
        assert_ne!(token_id, u32::MAX);
        let mut remaining_token_text = token_text;

        // (1) traverse shared token prefix with existing tokens
        let mut node = self.root;
        let mut path = vec![node];
        while !remaining_token_text.is_empty() {
            let b0 = remaining_token_text[0];
            let next = self.nodes[node as usize].next1.get(&b0).copied();
            match next {
                Some(next) => {
                    remaining_token_text = &remaining_token_text[1..];
                    node = next;
                    path.push(next);
                }
                None => {
                    break;
                }
            }
        }

        let suffix_start = path.len();

        // (2) create nodes for token suffix not shared with existing tokens
        for _ in 0..remaining_token_text.len() {
            let node = Node {
                token: u32::MAX,

                // Filled in later.
                anchor: 0,
                anchor_distance: 0,
                next1: HashMap::new(),
                next2: HashMap::new(),
                next3: HashMap::new(),
                next4: HashMap::new(),
            };
            let node_idx = self.nodes.len() as u32;
            self.nodes.push(node);
            path.push(node_idx);
        }

        assert_eq!(path.len(), token_text.len() + 1);

        // Connect the nodes.
        let mut anchor = self.root;
        let mut anchor_distance = 0;
        for (i, node_idx) in path.iter().copied().enumerate() {
            if path.len() >= 4 && i < path.len() - 4 {
                self.nodes[node_idx as usize].next4.insert(
                    [
                        token_text[i],
                        token_text[i + 1],
                        token_text[i + 2],
                        token_text[i + 3],
                    ],
                    path[i + 4],
                );
            }
            if path.len() >= 3 && i < path.len() - 3 {
                self.nodes[node_idx as usize].next3.insert(
                    [token_text[i], token_text[i + 1], token_text[i + 2]],
                    path[i + 3],
                );
            }
            if path.len() >= 2 && i < path.len() - 2 {
                self.nodes[node_idx as usize]
                    .next2
                    .insert([token_text[i], token_text[i + 1]], path[i + 2]);
            }
            if i < path.len() - 1 {
                self.nodes[node_idx as usize]
                    .next1
                    .insert(token_text[i], path[i + 1]);
            }

            if i < suffix_start {
                if self.nodes[node_idx as usize].anchor_distance == 0 {
                    anchor = node_idx;
                    anchor_distance = 0;
                } else {
                    anchor_distance += 1;
                }
            } else {
                self.nodes[node_idx as usize].anchor = anchor;
                anchor_distance += 1;
                self.nodes[node_idx as usize].anchor_distance = anchor_distance;
            }
        }

        let leaf = &mut self.nodes[*path.last().unwrap() as usize];
        leaf.anchor = *path.last().unwrap();
        leaf.anchor_distance = 0;
        leaf.token = token_id;
    }

    /// Tokenizes the given byte string, invoking the callback
    /// for each produced token.
    pub fn tokenize(&self, text: impl AsRef<[u8]>, mut emit_token: impl FnMut(TokenId)) {
        let mut text_at_prev_token = text.as_ref();
        let mut remaining_text = text_at_prev_token;

        let mut node_idx = self.root;

        while remaining_text.len() >= 4 {
            let node = &self.nodes[node_idx as usize];

            let [b0, b1, b2, b3] = [
                remaining_text[0],
                remaining_text[1],
                remaining_text[2],
                remaining_text[3],
            ];
            let skip1 = node.next1.get(&b0).copied();
            let skip2 = node.next2.get(&[b0, b1]).copied();
            let skip3 = node.next3.get(&[b0, b1, b2]).copied();
            let skip4 = node.next4.get(&[b0, b1, b2, b3]).copied();

            if let Some(skip4_node) = skip4 {
                node_idx = skip4_node;
                remaining_text = &remaining_text[4..];
            } else if let Some(skip3_node) = skip3 {
                node_idx = skip3_node;
                remaining_text = &remaining_text[3..];
            } else if let Some(skip2_node) = skip2 {
                node_idx = skip2_node;
                remaining_text = &remaining_text[2..];
            } else if let Some(skip1_node) = skip1 {
                node_idx = skip1_node;
                remaining_text = &remaining_text[1..];
            } else {
                // End of token.
                let distance = (text_at_prev_token.len() - remaining_text.len())
                    - self.nodes[node.anchor as usize].anchor_distance as usize;
                text_at_prev_token = &text_at_prev_token[distance..];
                remaining_text = text_at_prev_token;

                let token = self.nodes[node.anchor as usize].token;
                assert_ne!(token, u32::MAX);
                emit_token(token);

                node_idx = self.root;
            }
        }

        while !remaining_text.is_empty() {
            let node = &self.nodes[node_idx as usize];

            let b0 = remaining_text[0];
            let skip1 = node.next1.get(&b0).copied();

            if let Some(skip1_node) = skip1 {
                node_idx = skip1_node;
                remaining_text = &remaining_text[1..];
            } else {
                // End of token.
                let distance = (text_at_prev_token.len() - remaining_text.len())
                    - self.nodes[node.anchor as usize].anchor_distance as usize;
                assert!(distance > 0);
                text_at_prev_token = &text_at_prev_token[distance..];
                remaining_text = text_at_prev_token;

                let token = self.nodes[node.anchor as usize].token;
                emit_token(token);

                node_idx = self.root;
            }
        }

        // Emit final token if present.
        if text_at_prev_token.len() != remaining_text.len() {
            let node = &self.nodes[node_idx as usize];
            let token = self.nodes[node.anchor as usize].token;
            assert_ne!(token, u32::MAX);
            emit_token(token);
        }
    }

    pub fn tokenize_to_vec(&self, text: impl AsRef<[u8]>) -> Vec<TokenId> {
        let mut tokens = Vec::with_capacity(text.as_ref().len() / 2);
        self.tokenize(text, |tok| tokens.push(tok));
        tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple() {
        let mut tokenizer = TextTokenizer::empty();
        tokenizer.add_token(b" ", 0);
        tokenizer.add_token(b"the", 1);
        tokenizer.add_token(b"them", 2);
        tokenizer.add_token(b"themselves", 3);

        assert_eq!(
            tokenizer.tokenize_to_vec(b"the them themselves them  "),
            vec![1, 0, 2, 0, 3, 0, 2, 0, 0]
        );
    }
}
