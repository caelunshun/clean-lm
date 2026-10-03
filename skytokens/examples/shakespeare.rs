use skytokens::{
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

    Ok(())
}
