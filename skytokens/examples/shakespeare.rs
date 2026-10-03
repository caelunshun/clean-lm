use skytokens::{
    text_tokenizer::TextTokenizer,
    training::{self, CorpusBuilder},
    vocab::Vocab,
};

fn main() -> anyhow::Result<()> {
    let mut corpus = CorpusBuilder::new();
    for line in include_str!("shakespeare.txt").lines() {
        corpus.add_document(line);
    }
    let corpus = corpus.build();

    const VOCAB_SIZE: u32 = 4096;
    let vocab: Vocab<()> = training::train_vocab(&corpus, VOCAB_SIZE)?;

    println!("{}", serde_json::to_string_pretty(&vocab)?);

    let tokenizer = TextTokenizer::new(&vocab);

    let tokens = tokenizer
        .tokenize(
            r#"Neither a borrower nor a lender be;
For loan oft loses both itself and friend,
And borrowing dulls the edge of husbandry.
This above all: to thine ownself be true,
And it must follow, as the night the day,
Thou canst not then be false to any man.
Farewell: my blessing season this in thee!"#,
        )
        .unwrap();

    let tokens = tokens
        .into_iter()
        .map(|t| str::from_utf8(vocab.token(t).text().unwrap()).unwrap())
        .collect::<Vec<_>>();

    println!("Tokens: {tokens:?}");

    Ok(())
}
