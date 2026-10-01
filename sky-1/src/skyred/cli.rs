use std::path::PathBuf;

use clap::{Parser, Subcommand};

use crate::skyred::{config::Config, dataset};

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
}

pub fn run(cli: Cli) -> anyhow::Result<()> {
    let config: Config = json5::from_str(&fs_err::read_to_string(&cli.config)?)?;

    match cli.command {
        Command::BuildSample => dataset::sampling::build_sample(&config.dataset)?,
        Command::TrainTokenizer => dataset::tokenization::train_tokenizer(&config.dataset)?,
    }

    Ok(())
}
