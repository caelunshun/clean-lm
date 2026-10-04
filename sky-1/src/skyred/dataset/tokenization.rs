use crate::skyred::{
    config::DatasetConfig,
    dataset::{document_file::DocumentReader, sampling::SAMPLE_FILE_NAME, schema::Document},
};
use anyhow::Context;
use jiff::civil::Date;
use serde::{Deserialize, Serialize};
use skytokens::{
    text_tokenizer::TextTokenizer,
    training::CorpusBuilder,
    vocab::{SpecialTokenData, Token, TokenId, Vocab},
};
use std::io::BufWriter;
use strum::{EnumIter, IntoEnumIterator};

pub struct SkyredTokenizer {
    vocab: Vocab<SpecialToken>,
    text_tokenizer: TextTokenizer,
}

impl SkyredTokenizer {
    pub fn new(vocab: Vocab<SpecialToken>) -> anyhow::Result<Self> {
        Ok(Self {
            text_tokenizer: TextTokenizer::new(&vocab),
            vocab,
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
        tokens.extend(self.text_tokenizer.tokenize(text)?);
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
    tracing::info!(num_documents = documents.num_documents(), "building corpus");

    for document in documents {
        let document = document?;
        corpus.add_document(&document.subreddit)?;
        corpus.add_document(format_date(document.date))?;
        corpus.add_document(&document.author)?;
        corpus.add_document(&document.title)?;
        corpus.add_document(&document.selftext)?;

        for comment in document.comment_chain {
            corpus.add_document(&comment.author)?;
            corpus.add_document(&comment.text)?;
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
        BufWriter::new(fs_err::File::create(config.work_dir.join("vocab.json"))?),
        &vocab,
    )?;

    Ok(())
}

fn format_date(date: Date) -> String {
    date.to_string()
}
