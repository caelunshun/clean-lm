use bstr::{ByteSlice, ByteVec};
use compact_bytes::CompactBytes;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vocab {
    pub tokens: Vec<Token>,
}

pub type TokenId = u32;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Token {
    Text(TextToken),
    Special(SpecialToken),
}

#[derive(Debug, Clone)]
pub struct TextToken(CompactBytes);

impl Serialize for TextToken {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.0
            .as_slice()
            .escape_bytes()
            .collect::<String>()
            .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for TextToken {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Ok(Self(CompactBytes::new(
            Vec::<u8>::unescape_bytes(s).as_slice(),
        )))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecialToken {
    #[serde(rename = "special")]
    pub data: u64,
}
