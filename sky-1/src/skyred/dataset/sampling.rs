#![allow(deprecated)]

use std::{
    cmp::Reverse,
    collections::BinaryHeap,
    io::{BufWriter, Cursor, Write},
    ops::Range,
    sync::Arc,
};

use anyhow::Context;
use arrow_array::{
    Array, ArrayRef, LargeStringArray, RecordBatch, StringArray, StructArray,
    cast::AsArray,
    types::{Date32Type, Int64Type, UInt64Type},
};
use compact_str::{CompactString, ToCompactString};
use object_store::{ObjectStore, ObjectStoreExt, aws::AmazonS3Builder, path::Path as ObjectPath};
use parquet::arrow::{
    ProjectionMask,
    async_reader::{ParquetObjectReader, ParquetRecordBatchStreamBuilder},
};
use rand::{RngExt, SeedableRng, rngs::StdRng, seq::IndexedRandom};
use reddit_ingest::{
    assemble::{
        index_json::{INDEX_FILE_NAME, IndexJson},
        subreddit_daily::SUBREDDIT_DAILY_FILE_NAME,
    },
    stats::day_to_date,
};
use skytens_core::FastHashMap;
use tokio::task::JoinSet;

use crate::skyred::{
    config::DatasetConfig,
    dataset::schema::{Comment, Document},
};

const SECONDS_PER_DAY: i64 = 86_400;
/// Used to impute author name when null.
const DELETED_AUTHOR: &str = "[deleted]";
const MAX_COMMENT_CHAIN_DEPTH: usize = 16;

type Year = i16;

/// Values keyed by (year, subreddit).
type Strata<T> = FastHashMap<Year, FastHashMap<CompactString, T>>;

pub fn build_sample(config: &DatasetConfig) -> anyhow::Result<()> {
    tokio::runtime::Runtime::new()?.block_on(build_sample_async(config))
}

async fn build_sample_async(config: &DatasetConfig) -> anyhow::Result<()> {
    let object_store: Arc<dyn ObjectStore> = Arc::new(
        AmazonS3Builder::from_env()
            .with_bucket_name(&config.reddit_ingest_bucket)
            .build()?,
    );
    let ingest_path =
        |name: &str| ObjectPath::from(format!("{}/{name}", config.reddit_ingest_path));

    let index: IndexJson = serde_json::from_reader(zstd::Decoder::new(Cursor::new(
        object_store
            .get(&ingest_path(INDEX_FILE_NAME))
            .await?
            .bytes()
            .await?,
    ))?)?;

    let mut rng = StdRng::seed_from_u64(config.seed);

    let post_counts =
        read_post_counts(object_store.clone(), ingest_path(SUBREDDIT_DAILY_FILE_NAME)).await?;
    let quotas = Arc::new(allocate_quotas(&post_counts, config));
    tracing::info!(
        strata = quotas.values().map(FastHashMap::len).sum::<usize>(),
        "allocated sample quotas"
    );

    let max_in_flight = std::thread::available_parallelism()?.get();
    let mut files = index
        .files
        .iter()
        .map(|entry| (ingest_path(&entry.file), rng.random::<u64>()))
        .collect::<Vec<_>>()
        .into_iter();
    let mut tasks = JoinSet::new();
    let mut sample = Sample::default();
    let mut files_done = 0;
    loop {
        while tasks.len() < max_in_flight
            && let Some((path, seed)) = files.next()
        {
            let (object_store, quotas) = (object_store.clone(), quotas.clone());
            let config = config.clone();
            tasks.spawn(async move {
                sample_file(object_store, &path, &quotas, seed, &config)
                    .await
                    .with_context(|| format!("failed to sample from {path}"))
            });
        }
        let Some(file_sample) = tasks.join_next().await else {
            break;
        };
        sample.merge(file_sample??, &quotas);
        files_done += 1;
        tracing::info!("sampled {files_done}/{} files", index.files.len());
    }

    let documents = sample.into_documents();
    tracing::info!(num_documents = documents.len(), "writing sampled documents");

    fs_err::create_dir_all(&config.work_dir)?;
    let mut encoder = zstd::Encoder::new(
        BufWriter::new(fs_err::File::create(config.work_dir.join("posts.zst"))?),
        3,
    )?;
    postcard::to_io(&documents, &mut encoder)?;
    encoder.finish()?.flush()?;

    Ok(())
}

/// Reads the number of posts per (year, subreddit), excluding posts without a subreddit.
async fn read_post_counts(
    object_store: Arc<dyn ObjectStore>,
    path: ObjectPath,
) -> anyhow::Result<Strata<u64>> {
    let mut stream =
        ParquetRecordBatchStreamBuilder::new(ParquetObjectReader::new(object_store, path))
            .await?
            .build()?;
    let mut counts = Strata::<u64>::default();
    while let Some(reader) = stream.next_row_group().await? {
        for batch in reader {
            let batch = StructArray::from(batch?);
            let dates = column(&batch, "date")?.as_primitive::<Date32Type>();
            let entries = column(&batch, "subreddits")?.as_list::<i32>();
            let entry_fields = entries.values().as_struct();
            let subreddits = column(entry_fields, "subreddit")?.as_string::<i32>();
            let posts = column(entry_fields, "posts")?.as_primitive::<UInt64Type>();
            let offsets = entries.value_offsets();
            for row in 0..batch.len() {
                let Some(date) = day_to_date(dates.value(row).into()) else {
                    continue;
                };
                let by_subreddit = counts.entry(date.year()).or_default();
                for i in offsets[row] as usize..offsets[row + 1] as usize {
                    if subreddits.is_null(i) || posts.value(i) == 0 {
                        continue;
                    }
                    *by_subreddit.entry(subreddits.value(i).into()).or_default() += posts.value(i);
                }
            }
        }
    }
    Ok(counts)
}

fn allocate_quotas(post_counts: &Strata<u64>, config: &DatasetConfig) -> Strata<u64> {
    let mut years: Vec<_> = post_counts.iter().collect();
    years.sort_unstable_by_key(|(year, _)| **year);
    let year_counts: Vec<u64> = years.iter().map(|(_, s)| s.values().sum()).collect();
    let year_quotas = apportion(
        config.post_sample_size.into(),
        &year_counts,
        config.year_sampling_bias,
    );

    years
        .into_iter()
        .zip(year_quotas)
        .filter(|(_, quota)| *quota > 0)
        .map(|((&year, by_subreddit), year_quota)| {
            let mut subreddits: Vec<_> = by_subreddit.iter().collect();
            subreddits.sort_unstable_by_key(|(subreddit, _)| *subreddit);
            let counts: Vec<u64> = subreddits.iter().map(|(_, count)| **count).collect();
            let quotas = apportion(year_quota, &counts, config.subreddit_sampling_bias);
            let by_subreddit = subreddits
                .into_iter()
                .zip(quotas)
                .filter(|(_, quota)| *quota > 0)
                .map(|((subreddit, _), quota)| (subreddit.clone(), quota))
                .collect();
            (year, by_subreddit)
        })
        .collect()
}

/// Splits `total` across categories proportional to `count(category) + bias * total`,
/// capping each category at its count and redistributing the excess.
fn apportion(total: u64, counts: &[u64], bias: f64) -> Vec<u64> {
    let mut quotas = vec![0u64; counts.len()];
    let mut remaining = total.min(counts.iter().sum());
    let mut active: Vec<usize> = (0..counts.len()).filter(|&i| counts[i] > 0).collect();
    let weight = |i: usize| counts[i] as f64 + bias * total as f64;
    loop {
        let total_weight: f64 = active.iter().map(|&i| weight(i)).sum();
        let ideal = |i: usize| remaining as f64 * weight(i) / total_weight;
        let (capped, uncapped): (Vec<usize>, Vec<usize>) =
            active.iter().partition(|&&i| ideal(i) >= counts[i] as f64);

        if capped.is_empty() {
            let mut remainders = Vec::with_capacity(uncapped.len());
            for &i in &uncapped {
                let x = ideal(i);
                quotas[i] = x.floor() as u64;
                remainders.push((x - x.floor(), i));
            }
            let leftover =
                remaining.saturating_sub(uncapped.iter().map(|&i| quotas[i]).sum::<u64>());
            remainders.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
            for &(_, i) in remainders.iter().take(leftover as usize) {
                quotas[i] = (quotas[i] + 1).min(counts[i]);
            }
            return quotas;
        }

        for &i in &capped {
            quotas[i] = counts[i];
            remaining -= counts[i];
        }
        active = uncapped;
    }
}

async fn sample_file(
    object_store: Arc<dyn ObjectStore>,
    path: &ObjectPath,
    quotas: &Strata<u64>,
    seed: u64,
    config: &DatasetConfig,
) -> anyhow::Result<Sample> {
    let builder =
        ParquetRecordBatchStreamBuilder::new(ParquetObjectReader::new(object_store, path.clone()))
            .await?;
    let schema = builder.parquet_schema();
    let leaves = schema
        .columns()
        .iter()
        .enumerate()
        .filter(|(_, col)| {
            let parts = col.path().parts();
            let (top, leaf) = (parts[0].as_str(), parts[parts.len() - 1].as_str());
            if top == "comments" {
                matches!(leaf, "id" | "parent_id" | "author_name" | "body")
            } else {
                parts.len() == 1
                    && matches!(
                        top,
                        "created_utc" | "subreddit" | "title" | "selftext" | "author_name"
                    )
            }
        })
        .map(|(i, _)| i);
    let projection = ProjectionMask::leaves(schema, leaves);
    let mut stream = builder.with_projection(projection).build()?;

    let mut rng = StdRng::seed_from_u64(seed);
    let mut sample = Sample::default();
    while let Some(reader) = stream.next_row_group().await? {
        for batch in reader {
            sample_batch(&batch?, quotas, &mut rng, &mut sample, config)?;
        }
    }
    Ok(sample)
}

fn sample_batch(
    batch: &RecordBatch,
    quotas: &Strata<u64>,
    rng: &mut StdRng,
    sample: &mut Sample,
    config: &DatasetConfig,
) -> anyhow::Result<()> {
    let batch = StructArray::from(batch.clone());
    let created_utc = column(&batch, "created_utc")?.as_primitive::<Int64Type>();
    let subreddit = column(&batch, "subreddit")?.as_string::<i32>();
    let title = column(&batch, "title")?.as_string::<i32>();
    let selftext = column(&batch, "selftext")?.as_string::<i64>();
    let author = column(&batch, "author_name")?.as_string::<i32>();
    let comments = column(&batch, "comments")?.as_list::<i32>();
    let comment_fields = comments.values().as_struct();
    let comments = CommentColumns {
        offsets: comments.value_offsets(),
        id: column(comment_fields, "id")?.as_string(),
        parent_id: column(comment_fields, "parent_id")?.as_string(),
        author: column(comment_fields, "author_name")?.as_string(),
        body: column(comment_fields, "body")?.as_string(),
    };

    for row in 0..batch.len() {
        if subreddit.is_null(row) {
            continue;
        }
        let Some(date) = day_to_date(created_utc.value(row).div_euclid(SECONDS_PER_DAY)) else {
            continue;
        };

        let title = title.value(row).to_owned();
        let selftext = selftext.value(row).to_compact_string();

        let author = if author.is_null(row) {
            DELETED_AUTHOR.to_compact_string()
        } else {
            author.value(row).to_compact_string()
        };

        if config.prune_texts.contains(title.trim())
            || config.prune_texts.contains(selftext.trim())
            || config.prune_authors.contains(author.trim())
        {
            continue;
        }

        let Some(reservoir) = sample.reservoir(quotas, date.year(), subreddit.value(row)) else {
            continue;
        };
        let key = rng.random::<u64>();
        if !reservoir.accepts(key) {
            continue;
        }

        let mut doc = Document {
            subreddit: subreddit.value(row).into(),
            title,
            author,
            selftext,
            comment_chain: comments.deepest_chain(comments.range(row), rng, config),
        };
        doc.comment_chain.truncate(MAX_COMMENT_CHAIN_DEPTH);

        reservoir.offer(Keyed { key, doc });
    }
    Ok(())
}

fn column<'a>(array: &'a StructArray, name: &str) -> anyhow::Result<&'a ArrayRef> {
    array
        .column_by_name(name)
        .with_context(|| format!("missing column `{name}`"))
}

struct CommentColumns<'a> {
    offsets: &'a [i32],
    id: &'a StringArray,
    parent_id: &'a StringArray,
    author: &'a StringArray,
    body: &'a LargeStringArray,
}

impl CommentColumns<'_> {
    fn range(&self, post_row: usize) -> Range<usize> {
        self.offsets[post_row] as usize..self.offsets[post_row + 1] as usize
    }

    fn passes_filter(&self, i: usize, config: &DatasetConfig) -> bool {
        if config.prune_authors.contains(self.author.value(i).trim()) {
            return false;
        }

        if config.prune_texts.contains(self.body.value(i).trim()) {
            return false;
        }

        true
    }

    /// Selects a root-to-leaf comment chain of maximal depth,
    /// breaking ties uniformly at random. Comments are filtered
    /// according to the author and text filters.
    fn deepest_chain(
        &self,
        range: Range<usize>,
        rng: &mut StdRng,
        config: &DatasetConfig,
    ) -> Vec<Comment> {
        let mut valid_indices = range
            .into_iter()
            .filter(|i| self.passes_filter(*i, config))
            .collect::<Vec<_>>();
        if valid_indices.is_empty() {
            return Vec::new();
        }

        let id_to_index: FastHashMap<&str, usize> = valid_indices
            .iter()
            .copied()
            .map(|i| (self.id.value(i), i))
            .collect();

        // Remove comments whose parent is dangling.
        valid_indices.retain(|&i| match self.parent_id.value(i).strip_prefix("t1_") {
            Some(s) => id_to_index.contains_key(s),
            None => {
                // Parent ID starts with t3 so is the post, which is never dangling.
                true
            }
        });

        let parents: FastHashMap<usize, Option<usize>> = valid_indices
            .iter()
            .copied()
            .map(|i| {
                (
                    i,
                    self.parent_id
                        .value(i)
                        .strip_prefix("t1_")
                        .map(|parent| id_to_index[parent]),
                )
            })
            .collect();

        let mut depths: FastHashMap<usize, usize> = Default::default();
        let mut stack = Vec::new();
        for &i in &valid_indices {
            let mut cur = Some(i);
            // Length bound guards against cycles in malformed data.
            while let Some(c) = cur
                && depths[&c] == 0
                && stack.len() <= valid_indices.len()
            {
                stack.push(c);
                cur = parents[&c];
            }
            let mut depth = cur.map_or(0, |c| depths.get(&c).copied().unwrap_or(0));
            while let Some(c) = stack.pop() {
                depth += 1;
                depths.insert(c, depth);
            }
        }

        let max_depth = *depths.values().max().unwrap();
        let deepest: Vec<usize> = valid_indices
            .iter()
            .copied()
            .filter(|i| depths[i] == max_depth)
            .collect();
        let leaf = *deepest.choose(rng).unwrap();

        let mut chain = Vec::with_capacity(max_depth as usize);
        let mut cur = Some(leaf);
        while let Some(i) = cur
            && chain.len() < max_depth as usize
        {
            chain.push(Comment {
                author: self.author.value(i).into(),
                text: self.body.value(i).into(),
            });
            cur = parents[&i];
        }
        chain.reverse();
        chain
    }
}

struct Keyed {
    key: u64,
    doc: Document,
}

impl PartialEq for Keyed {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}

impl Eq for Keyed {}

impl PartialOrd for Keyed {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Keyed {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.key.cmp(&other.key)
    }
}

struct Reservoir {
    capacity: usize,
    /// Min-heap on key.
    heap: BinaryHeap<Reverse<Keyed>>,
}

impl Reservoir {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            heap: BinaryHeap::new(),
        }
    }

    fn accepts(&self, key: u64) -> bool {
        self.heap.len() < self.capacity || self.heap.peek().is_some_and(|min| key > min.0.key)
    }

    fn offer(&mut self, item: Keyed) {
        if !self.accepts(item.key) {
            return;
        }
        if self.heap.len() == self.capacity {
            self.heap.pop();
        }
        self.heap.push(Reverse(item));
    }
}

#[derive(Default)]
struct Sample {
    strata: Strata<Reservoir>,
}

impl Sample {
    /// Returns `None` if the stratum has no quota.
    fn reservoir(
        &mut self,
        quotas: &Strata<u64>,
        year: Year,
        subreddit: &str,
    ) -> Option<&mut Reservoir> {
        let &capacity = quotas.get(&year)?.get(subreddit)?;
        let by_subreddit = self.strata.entry(year).or_default();
        if !by_subreddit.contains_key(subreddit) {
            by_subreddit.insert(subreddit.into(), Reservoir::new(capacity as usize));
        }
        by_subreddit.get_mut(subreddit)
    }

    fn merge(&mut self, other: Sample, quotas: &Strata<u64>) {
        for (year, by_subreddit) in other.strata {
            for (subreddit, reservoir) in by_subreddit {
                let ours = self
                    .reservoir(quotas, year, &subreddit)
                    .expect("stratum came from the same quotas");
                for Reverse(item) in reservoir.heap {
                    ours.offer(item);
                }
            }
        }
    }

    /// Documents ordered by key, i.e. deterministically shuffled.
    fn into_documents(self) -> Vec<Document> {
        let mut items: Vec<Keyed> = self
            .strata
            .into_values()
            .flat_map(FastHashMap::into_values)
            .flat_map(|reservoir| reservoir.heap.into_iter().map(|Reverse(item)| item))
            .collect();
        items.sort_unstable();
        items.into_iter().map(|item| item.doc).collect()
    }
}
