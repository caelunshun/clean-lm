use compact_bytes::CompactBytes;
use serde::{Deserialize, Serialize, de};

#[derive(Debug, Clone)]
pub struct Vocab {
    tokens: Vec<Token>,
    special_token_index: hashbrown::HashMap<u64, TokenId>,
}

impl Serialize for Vocab {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.tokens.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Vocab {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let tokens = Vec::<Token>::deserialize(deserializer)?;
        if tokens.len() >= u32::MAX as usize {
            return Err(de::Error::custom("too many tokens"));
        }
        Ok(Self::new(tokens))
    }
}

impl Vocab {
    pub fn new(tokens: impl Into<Vec<Token>>) -> Self {
        let tokens = tokens.into();
        assert!(tokens.len() < u32::MAX as usize);
        let mut special_token_index = hashbrown::HashMap::default();
        for (id, token) in tokens.iter().enumerate() {
            if let Token::Special(SpecialToken { tag }) = token {
                special_token_index.insert(*tag, id as u32);
            }
        }
        Self {
            tokens,
            special_token_index,
        }
    }

    pub fn tokens(&self) -> impl Iterator<Item = (TokenId, &Token)> {
        self.tokens
            .iter()
            .enumerate()
            .map(|(i, tok)| (i as u32, tok))
    }

    pub fn num_tokens(&self) -> u32 {
        self.tokens.len() as u32
    }

    pub fn token(&self, id: TokenId) -> &Token {
        &self.tokens[id as usize]
    }

    pub fn special_token_tag_to_id(&self, tag: u64) -> Option<TokenId> {
        self.special_token_index.get(&tag).copied()
    }
}

pub type TokenId = u32;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Token {
    Text(TextToken),
    Special(SpecialToken),
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecialToken {
    #[serde(rename = "special")]
    pub tag: u64,
}
