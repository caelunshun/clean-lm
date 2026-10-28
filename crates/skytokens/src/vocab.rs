use anyhow::{bail, ensure};
use bytemuck::{Pod, Zeroable};
use compact_bytes::CompactBytes;
use serde::{
    Deserialize, Serialize,
    de::{self, DeserializeOwned},
};
use std::{fmt::Display, hash::Hash};

#[derive(Debug, Clone)]
pub struct Vocab<SP> {
    tokens: Vec<Token<SP>>,
    merges: Vec<Merge>,
    text_token_index: hashbrown::HashMap<CompactBytes, TokenId>,
    special_token_index: hashbrown::HashMap<SP, TokenId>,
}

#[derive(Debug, Serialize, Deserialize)]
struct SerdeRepr<SP> {
    tokens: Vec<Token<SP>>,
    merges: Vec<Merge>,
}

impl<SP> Serialize for Vocab<SP>
where
    SP: SpecialTokenData,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        SerdeRepr {
            tokens: self.tokens.clone(),
            merges: self.merges.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de, SP> Deserialize<'de> for Vocab<SP>
where
    SP: SpecialTokenData,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let SerdeRepr { tokens, merges } = SerdeRepr::deserialize(deserializer)?;
        Self::new(tokens, merges).map_err(de::Error::custom)
    }
}

impl<SP> Vocab<SP>
where
    SP: SpecialTokenData,
{
    pub fn empty() -> Self {
        Self {
            tokens: Vec::new(),
            merges: Vec::new(),
            special_token_index: Default::default(),
            text_token_index: Default::default(),
        }
    }

    pub fn new(
        tokens: impl IntoIterator<Item = Token<SP>>,
        merges: impl IntoIterator<Item = Merge>,
    ) -> anyhow::Result<Self> {
        let mut vocab = Self::empty();
        for tok in tokens {
            vocab.add_token(tok)?;
        }
        for merge in merges {
            vocab.add_merge(merge)?;
        }
        Ok(vocab)
    }

    pub fn add_token(&mut self, token: Token<SP>) -> anyhow::Result<TokenId> {
        if self.tokens.len() == u32::MAX as usize - 1 {
            bail!("too many tokens");
        }

        if let Token::Special(special_data) = &token {
            if self.special_token_index.contains_key(&special_data.data) {
                bail!("duplicate special tokens");
            }
            self.special_token_index
                .insert(special_data.data.clone(), TokenId(self.tokens.len() as u32));
        }

        if let Token::Text(text) = &token {
            if self.text_token_index.contains_key(&text.0) {
                bail!("duplicate text tokens");
            }
            self.text_token_index
                .insert(text.0.clone(), TokenId(self.tokens.len() as u32));
        }

        self.tokens.push(token);
        Ok(TokenId(self.tokens.len() as u32 - 1))
    }

    pub fn add_merge(&mut self, merge: Merge) -> anyhow::Result<()> {
        ensure!((merge.first.0 as usize) < self.tokens.len());
        ensure!((merge.second.0 as usize) < self.tokens.len());
        ensure!((merge.merged.0 as usize) < self.tokens.len());
        self.merges.push(merge);
        Ok(())
    }

    pub fn merges(&self) -> impl Iterator<Item = Merge> {
        self.merges.iter().copied()
    }

    pub fn tokens(&self) -> impl Iterator<Item = (TokenId, &Token<SP>)> {
        self.tokens
            .iter()
            .enumerate()
            .map(|(i, tok)| (TokenId(i as u32), tok))
    }

    pub fn num_tokens(&self) -> u32 {
        self.tokens.len() as u32
    }

    pub fn token(&self, id: TokenId) -> &Token<SP> {
        &self.tokens[id.0 as usize]
    }

    pub fn special_token_to_id(&self, sp: &SP) -> Option<TokenId> {
        self.special_token_index.get(sp).copied()
    }

    pub fn text_to_id(&self, text: impl AsRef<[u8]>) -> Option<TokenId> {
        self.text_token_index
            .get(&CompactBytes::new(text.as_ref()))
            .copied()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Token<SP> {
    Text(TextToken),
    Special(SpecialToken<SP>),
}

impl<SP> Token<SP> {
    pub fn text(&self) -> Option<&[u8]> {
        match self {
            Token::Text(text_token) => Some(&text_token.0),
            Token::Special(_) => None,
        }
    }

    pub fn text_token(&self) -> Option<&TextToken> {
        match self {
            Token::Text(text_token) => Some(text_token),
            Token::Special(_) => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TextToken(pub CompactBytes);

impl Serialize for TextToken {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        String::from_utf8(escape_bytes::escape(self.0.as_slice().iter().copied()))
            .unwrap()
            .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for TextToken {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        let d = escape_bytes::unescape(s.as_bytes().iter().copied())
            .map_err(|_| de::Error::custom("incorrect escape"))?;

        Ok(Self(CompactBytes::new(d.as_slice())))
    }
}

impl Display for TextToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        String::from_utf8(escape_bytes::escape(self.0.iter().copied()))
            .unwrap()
            .fmt(f)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecialToken<S> {
    #[serde(rename = "special")]
    pub data: S,
}

pub trait SpecialTokenData: Clone + Eq + Hash + Serialize + DeserializeOwned {}

impl SpecialTokenData for () {}

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, Pod, Zeroable,
)]
#[serde(transparent)]
#[repr(transparent)]
pub struct TokenId(u32);

impl TokenId {
    pub const PLACEHOLDER: TokenId = TokenId(u32::MAX);

    pub fn from_idx(idx: u32) -> anyhow::Result<Self> {
        ensure!(idx < u32::MAX);
        Ok(Self(idx))
    }

    pub fn idx(self) -> u32 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Merge {
    pub first: TokenId,
    pub second: TokenId,
    pub merged: TokenId,
}
