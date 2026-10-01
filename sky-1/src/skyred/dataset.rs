use std::io::Cursor;

use compact_str::CompactString;
use object_store::{ObjectStoreExt, aws::AmazonS3Builder};
use pollster::FutureExt;
use rand::{SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};

use crate::skyred::config::DatasetConfig;

/// One post and a list of nested comments replying to it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub subreddit: CompactString,
    pub title: String,
    pub author: String,
    pub selftext: CompactString,
    pub comment_chain: Vec<Comment>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Comment {
    pub author: CompactString,
    pub text: CompactString,
}

pub fn build_sample(config: &DatasetConfig) -> anyhow::Result<()> {
    let object_store = AmazonS3Builder::from_env()
        .with_bucket_name(&config.reddit_ingest_bucket)
        .build()?;

    let index: reddit_ingest::assemble::index_json::IndexJson =
        serde_json::from_reader(zstd::Decoder::new(Cursor::new(
            object_store
                .get(&format!("{}/index.json.zst", config.reddit_ingest_path).into())
                .block_on()?
                .bytes()
                .block_on()?,
        ))?)?;

    let mut rng = StdRng::seed_from_u64(config.seed);

    // TODO:
    // - Split post sample count across years, then across subreddits,
    //   proportional to (# of posts in category + sample_bias * total_sampled_posts)
    // - Iterate over each parquet file in the index, read using parquet
    //   async reader in parallel (selecting only necessary columns),
    //   for each post select the deepest comment chain (ties broken by RNG)
    // - Write documents into work_dir/posts.zst using postcard + zstd

    Ok(())
}
