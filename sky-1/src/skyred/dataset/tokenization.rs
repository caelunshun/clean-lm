use crate::skyred::{
    config::DatasetConfig,
    dataset::{
        document_file::DocumentReader,
        sampling::SAMPLE_FILE_NAME,
        schema::Document,
        tokenized_file::{SChunkWriter, TOKEN_COUNTS_FILE_NAME, TOKENIZED_FILE_NAME},
    },
};
use anyhow::Context;
use jiff::civil::Date;
use memchr::memchr2;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use skytokens::{
    text_tokenizer::TextTokenizer,
    training::CorpusBuilder,
    vocab::{SpecialTokenData, Token, TokenId, Vocab},
};
use std::io::BufWriter;
use strum::{EnumIter, IntoEnumIterator};

pub const VOCAB_FILE_NAME: &str = "vocab.json";

pub struct SkyredTokenizer {
    vocab: Vocab<SpecialToken>,
    text_tokenizer: TextTokenizer,
    use_pretokenization: bool,
}

impl SkyredTokenizer {
    pub fn new(vocab: Vocab<SpecialToken>, use_pretokenization: bool) -> anyhow::Result<Self> {
        Ok(Self {
            text_tokenizer: TextTokenizer::new(&vocab),
            vocab,
            use_pretokenization,
        })
    }

    pub fn tokenize_document(&self, document: &Document) -> anyhow::Result<Vec<TokenId>> {
        let mut tokens = Vec::new();

        self.push_text(&mut tokens, &document.subreddit)?;

        self.push_special(&mut tokens, SpecialToken::BeginDate)?;
        self.push_text(&mut tokens, format_date(document.date))?;

        self.push_special(&mut tokens, SpecialToken::BeginPostTitle)?;
        self.push_text(&mut tokens, &document.title)?;

        self.push_special(&mut tokens, SpecialToken::BeginAuthor)?;
        self.push_text(&mut tokens, &document.author)?;

        self.push_special(&mut tokens, SpecialToken::BeginText)?;
        self.push_text(&mut tokens, &document.selftext)?;

        for comment in &document.comment_chain {
            self.push_special(&mut tokens, SpecialToken::BeginAuthor)?;
            self.push_text(&mut tokens, &comment.author)?;

            self.push_special(&mut tokens, SpecialToken::BeginText)?;
            self.push_text(&mut tokens, &comment.text)?;
        }

        self.push_special(&mut tokens, SpecialToken::EndChain)?;

        Ok(tokens)
    }

    fn push_text(&self, tokens: &mut Vec<TokenId>, text: impl AsRef<[u8]>) -> anyhow::Result<()> {
        if self.use_pretokenization {
            for pretoken in pretokenize(text.as_ref()) {
                tokens.extend(self.text_tokenizer.tokenize(pretoken)?);
            }
        } else {
            tokens.extend(self.text_tokenizer.tokenize(text)?);
        }
        Ok(())
    }

    fn push_special(&self, tokens: &mut Vec<TokenId>, special: SpecialToken) -> anyhow::Result<()> {
        tokens.push(
            self.vocab
                .special_token_to_id(&special)
                .context("special token missing from vocab")?,
        );
        Ok(())
    }

    pub fn vocab(&self) -> &Vocab<SpecialToken> {
        &self.vocab
    }
}

fn pretokenize(text: &[u8]) -> Vec<&[u8]> {
    let mut remaining = text;
    let mut pretokens = Vec::new();
    while !remaining.is_empty() {
        let pretoken_end = memchr2(b' ', b'\n', remaining);
        match pretoken_end {
            Some(i) => {
                pretokens.push(&remaining[..=i]);
                remaining = &remaining[i + 1..];
            }
            None => {
                pretokens.push(remaining);
                break;
            }
        }
    }
    pretokens
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, EnumIter, Serialize, Deserialize)]
pub enum SpecialToken {
    BeginDate,
    BeginPostTitle,
    BeginAuthor,
    BeginText,
    EndChain,
}

impl SpecialTokenData for SpecialToken {}

pub fn train_tokenizer(config: &DatasetConfig) -> anyhow::Result<()> {
    let mut corpus = CorpusBuilder::new();

    let documents = DocumentReader::open(&config.work_dir.join(SAMPLE_FILE_NAME))?;
    let count = documents
        .num_documents()
        .min(config.tokenizer_train_sample_limit.unwrap_or(u64::MAX));
    tracing::info!(num_documents = count, "building corpus");

    for document in documents.into_iter().take(count.try_into()?) {
        let document = document?;
        for text in [
            document.subreddit.as_str(),
            format_date(document.date).as_str(),
            document.author.as_str(),
            document.title.as_str(),
            document.selftext.as_str(),
        ] {
            if config.use_pretokenization {
                for pretoken in pretokenize(text.as_bytes()) {
                    corpus.add_document(pretoken)?;
                }
            } else {
                corpus.add_document(text)?;
            }
        }

        for comment in document.comment_chain {
            for text in [comment.author.as_str(), comment.text.as_str()] {
                if config.use_pretokenization {
                    for pretoken in pretokenize(text.as_bytes()) {
                        corpus.add_document(pretoken)?;
                    }
                } else {
                    corpus.add_document(text)?;
                }
            }
        }
    }

    let corpus = corpus.build();
    tracing::info!("built corpus");

    let mut vocab = skytokens::training::train_vocab::<SpecialToken>(
        corpus,
        config
            .total_vocab_size
            .checked_sub(SpecialToken::iter().count() as u32)
            .context("too few tokens")?,
        config.bpe_training_fused_merges,
    )?;
    for special in SpecialToken::iter() {
        vocab.add_token(Token::Special(skytokens::vocab::SpecialToken {
            data: special,
        }))?;
    }

    serde_json::to_writer_pretty(
        BufWriter::new(fs_err::File::create(config.work_dir.join(VOCAB_FILE_NAME))?),
        &vocab,
    )?;

    Ok(())
}

pub fn tokenize(config: &DatasetConfig) -> anyhow::Result<()> {
    const DOCUMENT_GROUP_SIZE: usize = 16384;
    const TOKEN_CHUNK_ITEMS: usize = 4 << 20;
    const COUNT_CHUNK_ITEMS: usize = 1 << 20;

    let vocab: Vocab<SpecialToken> =
        serde_json::from_slice(&fs_err::read(config.work_dir.join(VOCAB_FILE_NAME))?)?;
    let tokenizer = SkyredTokenizer::new(vocab, config.use_pretokenization)?;

    let mut documents = DocumentReader::open(&config.work_dir.join(SAMPLE_FILE_NAME))?;
    let num_documents = documents.num_documents();
    tracing::info!(num_documents, "tokenizing documents");

    let mut tokens_writer = SChunkWriter::<TokenId>::create(
        &config.work_dir.join(TOKENIZED_FILE_NAME),
        TOKEN_CHUNK_ITEMS,
    )?;
    let mut counts_writer = SChunkWriter::<u32>::create(
        &config.work_dir.join(TOKEN_COUNTS_FILE_NAME),
        COUNT_CHUNK_ITEMS,
    )?;

    let mut num_processed = 0u64;
    let mut num_tokens = 0u64;
    loop {
        let group = (&mut documents)
            .take(DOCUMENT_GROUP_SIZE)
            .collect::<anyhow::Result<Vec<_>>>()?;
        if group.is_empty() {
            break;
        }

        let tokenized = group
            .par_iter()
            .map(|doc| tokenizer.tokenize_document(doc))
            .collect::<anyhow::Result<Vec<_>>>()?;

        for doc in &tokenized {
            counts_writer.extend(&[u32::try_from(doc.len())?])?;
            tokens_writer.extend(doc)?;
            num_tokens += doc.len() as u64;
        }

        num_processed += group.len() as u64;
        tracing::info!(num_processed, num_documents, num_tokens, "tokenized group");
    }

    tokens_writer.finish()?;
    counts_writer.finish()?;
    tracing::info!(num_documents, num_tokens, "wrote tokenized documents");

    Ok(())
}

fn format_date(date: Date) -> String {
    date.to_string()
}
