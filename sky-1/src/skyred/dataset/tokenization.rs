use std::io::BufWriter;

use anyhow::Context;
use jiff::civil::Date;
use skytens_core::FastHashMap;
use skytokens::{
    text_tokenizer::TextTokenizer,
    training::Corpus,
    vocab::{TokenId, Vocab},
};
use strum::{EnumIter, FromRepr, IntoEnumIterator};

use crate::skyred::{config::DatasetConfig, dataset::schema::Document};

pub struct SkyredTokenizer {
    vocab: Vocab,
    special_token_map: FastHashMap<SpecialToken, TokenId>,
    text_tokenizer: TextTokenizer,
}

impl SkyredTokenizer {
    pub fn new(vocab: Vocab) -> anyhow::Result<Self> {
        let mut special_token_map = FastHashMap::default();
        for tok in SpecialToken::iter() {
            special_token_map.insert(
                tok,
                vocab
                    .special_token_tag_to_id(tok as u64)
                    .with_context(|| format!("vocab is missing a token for {tok:?}"))?,
            );
        }
        Ok(Self {
            text_tokenizer: TextTokenizer::new(&vocab),
            vocab,
            special_token_map,
        })
    }

    pub fn tokenize_document(&self, document: &Document) -> Vec<TokenId> {
        let mut tokens = Vec::new();

        self.text_tokenizer
            .tokenize_into(&document.subreddit, &mut tokens);

        tokens.push(self.special_token_map[&SpecialToken::BeginDate]);
        self.text_tokenizer
            .tokenize_into(format_date(document.date), &mut tokens);

        tokens.push(self.special_token_map[&SpecialToken::BeginPostTitle]);
        self.text_tokenizer
            .tokenize_into(&document.title, &mut tokens);

        tokens.push(self.special_token_map[&SpecialToken::BeginAuthor]);
        self.text_tokenizer
            .tokenize_into(&document.author, &mut tokens);

        tokens.push(self.special_token_map[&SpecialToken::BeginText]);
        self.text_tokenizer
            .tokenize_into(&document.selftext, &mut tokens);

        for comment in &document.comment_chain {
            tokens.push(self.special_token_map[&SpecialToken::BeginAuthor]);
            self.text_tokenizer
                .tokenize_into(&comment.author, &mut tokens);

            tokens.push(self.special_token_map[&SpecialToken::BeginText]);
            self.text_tokenizer
                .tokenize_into(&comment.text, &mut tokens);
        }

        tokens.push(self.special_token_map[&SpecialToken::EndChain]);

        tokens
    }

    pub fn vocab(&self) -> &Vocab {
        &self.vocab
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, EnumIter, FromRepr)]
#[repr(u64)]
enum SpecialToken {
    BeginDate,
    BeginPostTitle,
    BeginAuthor,
    BeginText,
    EndChain,
}

pub fn train_tokenizer(config: &DatasetConfig) -> anyhow::Result<()> {
    let mut corpus = Corpus::builder(None);

    let documents: Vec<Document> = postcard::from_io((
        zstd::Decoder::new(fs_err::File::open(config.work_dir.join("posts.zst"))?)?,
        &mut vec![0u8; 16384],
    ))?
    .0;

    for document in documents {
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

    let vocab = skytokens::training::train_vocab(
        &corpus,
        config
            .total_vocab_size
            .checked_sub(SpecialToken::iter().count() as u32)
            .context("too few tokens")?,
    )?;

    serde_json::to_writer_pretty(
        BufWriter::new(fs_err::File::create(config.work_dir.join("vocab.json"))?),
        &vocab,
    )?;

    Ok(())
}

fn format_date(date: Date) -> String {
    date.to_string()
}
