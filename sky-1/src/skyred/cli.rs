use crate::skyred::{
    config::{Config, SensitivePhrase},
    dataset, training,
};
use clap::{Parser, Subcommand};
use std::{
    io::{Write, stdin, stdout},
    path::PathBuf,
};

#[derive(Debug, Parser)]
pub struct Cli {
    #[clap(long)]
    pub config: PathBuf,
    #[clap(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    BuildSample,
    TrainTokenizer,
    Tokenize,
    Train(training::Cli),
    EncodeSensitivePhrase,
}

pub fn run(cli: Cli) -> anyhow::Result<()> {
    let config: Config = json5::from_str(&fs_err::read_to_string(&cli.config)?)?;

    match cli.command {
        Command::BuildSample => dataset::sampling::build_sample(&config.dataset)?,
        Command::TrainTokenizer => dataset::tokenization::train_tokenizer(&config.dataset)?,
        Command::Tokenize => dataset::tokenization::tokenize(&config.dataset)?,
        Command::Train(cli) => training::run(&cli, &config)?,
        Command::EncodeSensitivePhrase => {
            let mut phrase = String::new();
            print!("Enter phrase: ");
            stdout().flush()?;
            stdin().read_line(&mut phrase)?;
            let phrase = phrase.trim();
            println!("{}", SensitivePhrase(phrase.into()).encode());
        }
    }

    Ok(())
}
