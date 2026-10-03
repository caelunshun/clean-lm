#![allow(deprecated)]

use std::{
    cmp::Reverse, collections::BinaryHeap, future::Future, io::Cursor, ops::Range, sync::Arc,
};

use anyhow::{Context, ensure};
use arrow_array::{
    Array, ArrayRef, LargeStringArray, PrimitiveArray, RecordBatch, StringArray, StructArray,
    cast::AsArray, types::Int64Type,
};
use compact_str::CompactString;
use jiff::civil::Date;
use object_store::{ObjectStore, ObjectStoreExt, aws::AmazonS3Builder, path::Path as ObjectPath};
use parquet::arrow::{
    ProjectionMask,
    async_reader::{
        ParquetObjectReader, ParquetRecordBatchStream, ParquetRecordBatchStreamBuilder,
    },
};
use rand::{
    RngExt, SeedableRng,
    rngs::StdRng,
    seq::{IndexedRandom, SliceRandom},
};
use reddit_ingest::{
    assemble::index_json::{INDEX_FILE_NAME, IndexJson},
    stats::day_to_date,
};
use skytens_core::FastHashMap;
use tokio::task::JoinSet;

use crate::skyred::{
    config::DatasetConfig,
    dataset::{
        document_file,
        schema::{Comment, Document},
    },
};

const SECONDS_PER_DAY: i64 = 86_400;
/// Used to impute author name when null.
const DELETED_AUTHOR: &str = "[deleted]";
const MAX_COMMENT_CHAIN_DEPTH: usize = 16;

type Year = i16;

/// Values keyed by (year, subreddit).
type Strata<T> = FastHashMap<Year, FastHashMap<CompactString, T>>;

pub const SAMPLE_FILE_NAME: &str = "posts.zst";

pub fn build_sample(config: &DatasetConfig) -> anyhow::Result<()> {
    for mix in [config.year_uniform_mix, config.subreddit_uniform_mix] {
        ensure!((0.0..=1.0).contains(&mix), "uniform mix must be in [0, 1]");
    }

    let documents = tokio::runtime::Runtime::new()?.block_on(build_sample_async(config))?;
    tracing::info!(num_documents = documents.len(), "writing sampled documents");

    fs_err::create_dir_all(&config.work_dir)?;
    document_file::write_documents(&config.work_dir.join(SAMPLE_FILE_NAME), &documents)
}

async fn build_sample_async(config: &DatasetConfig) -> anyhow::Result<Vec<Document>> {
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
    let paths: Vec<ObjectPath> = index
        .files
        .iter()
        .map(|entry| ingest_path(&entry.file))
        .collect();

    let mut post_counts = Strata::<u64>::default();
    let mut files_done = 0;
    for_each_file(
        paths.iter().cloned(),
        |path| {
            let object_store = object_store.clone();
            let config = config.clone();
            async move {
                count_file(object_store, &path, &config)
                    .await
                    .with_context(|| format!("failed to count posts in {path}"))
            }
        },
        |file_counts| {
            merge_counts(&mut post_counts, file_counts);
            files_done += 1;
            tracing::info!("counted posts in {files_done}/{} files", paths.len());
        },
    )
    .await?;

    let quotas = Arc::new(allocate_quotas(&post_counts, config));
    tracing::info!(
        strata = quotas.values().map(FastHashMap::len).sum::<usize>(),
        "allocated sample quotas"
    );

    let files: Vec<_> = paths
        .iter()
        .map(|path| (path.clone(), rng.random::<u64>()))
        .collect();
    let mut sample = Sample::default();
    let mut files_done = 0;
    for_each_file(
        files,
        |(path, seed)| {
            let (object_store, quotas) = (object_store.clone(), quotas.clone());
            let config = config.clone();
            async move {
                sample_file(object_store, &path, &quotas, seed, &config)
                    .await
                    .with_context(|| format!("failed to sample from {path}"))
            }
        },
        |file_sample| {
            sample.merge(file_sample, &quotas);
            files_done += 1;
            tracing::info!("sampled {files_done}/{} files", paths.len());
        },
    )
    .await?;

    Ok(sample.into_documents(&mut rng))
}

/// Runs a task per file with bounded concurrency, merging results in completion order.
async fn for_each_file<I, T, Fut>(
    items: impl IntoIterator<Item = I>,
    mut spawn: impl FnMut(I) -> Fut,
    mut merge: impl FnMut(T),
) -> anyhow::Result<()>
where
    Fut: Future<Output = anyhow::Result<T>> + Send + 'static,
    T: Send + 'static,
{
    let max_in_flight = 2 * std::thread::available_parallelism()?.get();
    let mut items = items.into_iter();
    let mut tasks = JoinSet::new();
    loop {
        while tasks.len() < max_in_flight
            && let Some(item) = items.next()
        {
            tasks.spawn(spawn(item));
        }
        let Some(result) = tasks.join_next().await else {
            break;
        };
        merge(result??);
    }
    Ok(())
}

fn merge_counts(counts: &mut Strata<u64>, other: Strata<u64>) {
    for (year, other_by_subreddit) in other {
        let by_subreddit = counts.entry(year).or_default();
        for (subreddit, count) in other_by_subreddit {
            *by_subreddit.entry(subreddit).or_default() += count;
        }
    }
}

/// Counts posts per (year, subreddit) that pass the filters applied during sampling.
async fn count_file(
    object_store: Arc<dyn ObjectStore>,
    path: &ObjectPath,
    config: &DatasetConfig,
) -> anyhow::Result<Strata<u64>> {
    let mut stream = open_posts(object_store, path, false).await?;
    let mut counts = Strata::<u64>::default();
    while let Some(reader) = stream.next_row_group().await? {
        for batch in reader {
            count_batch(&batch?, config, &mut counts)?;
        }
    }
    Ok(counts)
}

fn count_batch(
    batch: &RecordBatch,
    config: &DatasetConfig,
    counts: &mut Strata<u64>,
) -> anyhow::Result<()> {
    let batch = StructArray::from(batch.clone());
    let posts = PostColumns::new(&batch)?;
    for row in 0..batch.len() {
        if let Some((date, subreddit)) = posts.stratum(row, config) {
            *counts
                .entry(date.year())
                .or_default()
                .entry(subreddit.into())
                .or_default() += 1;
        }
    }
    Ok(())
}

fn allocate_quotas(post_counts: &Strata<u64>, config: &DatasetConfig) -> Strata<u64> {
    let mut years: Vec<_> = post_counts.iter().collect();
    years.sort_unstable_by_key(|(year, _)| **year);
    let year_counts: Vec<u64> = years.iter().map(|(_, s)| s.values().sum()).collect();
    let year_quotas = apportion(
        config.post_sample_size.into(),
        &year_counts,
        config.year_uniform_mix,
    );

    years
        .into_iter()
        .zip(year_quotas)
        .filter(|(_, quota)| *quota > 0)
        .map(|((&year, by_subreddit), year_quota)| {
            let mut subreddits: Vec<_> = by_subreddit.iter().collect();
            subreddits.sort_unstable_by_key(|(subreddit, _)| *subreddit);
            let counts: Vec<u64> = subreddits.iter().map(|(_, count)| **count).collect();
            let quotas = apportion(year_quota, &counts, config.subreddit_uniform_mix);
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

/// Splits `total` across categories by shares mixing `(1 - uniform_mix)` proportional to count
/// with `uniform_mix` uniform over nonempty categories, capping each category at its count
/// and redistributing the excess in proportion to the remaining shares.
fn apportion(total: u64, counts: &[u64], uniform_mix: f64) -> Vec<u64> {
    let mut quotas = vec![0u64; counts.len()];
    let count_sum: u64 = counts.iter().sum();
    let mut remaining = total.min(count_sum);
    let mut active: Vec<usize> = (0..counts.len()).filter(|&i| counts[i] > 0).collect();
    let num_nonempty = active.len() as f64;
    let weight = |i: usize| {
        (1.0 - uniform_mix) * counts[i] as f64 / count_sum as f64 + uniform_mix / num_nonempty
    };
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

/// Opens a stream over a posts file, projecting the post-level columns used for
/// stratification and filtering and, if `with_comments`, the comment columns.
async fn open_posts(
    object_store: Arc<dyn ObjectStore>,
    path: &ObjectPath,
    with_comments: bool,
) -> anyhow::Result<ParquetRecordBatchStream<ParquetObjectReader>> {
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
                with_comments && matches!(leaf, "id" | "parent_id" | "author_name" | "body")
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
    Ok(builder.with_projection(projection).build()?)
}

async fn sample_file(
    object_store: Arc<dyn ObjectStore>,
    path: &ObjectPath,
    quotas: &Strata<u64>,
    seed: u64,
    config: &DatasetConfig,
) -> anyhow::Result<Sample> {
    let mut stream = open_posts(object_store, path, true).await?;

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
    let posts = PostColumns::new(&batch)?;
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
        let Some((date, subreddit)) = posts.stratum(row, config) else {
            continue;
        };
        let Some(reservoir) = sample.reservoir(quotas, date.year(), subreddit) else {
            continue;
        };
        let key = rng.random::<u64>();
        if !reservoir.accepts(key) {
            continue;
        }

        let mut doc = Document {
            date,
            subreddit: subreddit.into(),
            title: posts.title.value(row).to_owned(),
            author: posts.author(row).into(),
            selftext: posts.selftext(row).into(),
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

struct PostColumns<'a> {
    created_utc: &'a PrimitiveArray<Int64Type>,
    subreddit: &'a StringArray,
    title: &'a StringArray,
    selftext: &'a LargeStringArray,
    author: &'a StringArray,
}

impl<'a> PostColumns<'a> {
    fn new(batch: &'a StructArray) -> anyhow::Result<Self> {
        Ok(Self {
            created_utc: column(batch, "created_utc")?.as_primitive(),
            subreddit: column(batch, "subreddit")?.as_string(),
            title: column(batch, "title")?.as_string(),
            selftext: column(batch, "selftext")?.as_string(),
            author: column(batch, "author_name")?.as_string(),
        })
    }

    fn selftext(&self, row: usize) -> &'a str {
        if self.selftext.is_null(row) {
            ""
        } else {
            self.selftext.value(row)
        }
    }

    fn author(&self, row: usize) -> &'a str {
        if self.author.is_null(row) {
            DELETED_AUTHOR
        } else {
            self.author.value(row)
        }
    }

    /// Returns the post's date and subreddit, or `None` if the post is excluded
    /// by missing data or the author and text filters.
    fn stratum(&self, row: usize, config: &DatasetConfig) -> Option<(Date, &'a str)> {
        if self.subreddit.is_null(row) {
            return None;
        }
        let date = day_to_date(self.created_utc.value(row).div_euclid(SECONDS_PER_DAY))?;
        if config.prune_texts.contains(self.title.value(row).trim())
            || config.prune_texts.contains(self.selftext(row).trim())
            || config.prune_authors.contains(self.author(row).trim())
        {
            return None;
        }
        Some((date, self.subreddit.value(row)))
    }
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

        let mut id_to_index: FastHashMap<&str, usize> = valid_indices
            .iter()
            .copied()
            .map(|i| (self.id.value(i), i))
            .collect();

        // Remove comments whose parent is dangling, to fixpoint.
        let mut prev_len = valid_indices.len();
        loop {
            valid_indices.retain(|&i| match self.parent_id.value(i).strip_prefix("t1_") {
                Some(s) => {
                    let keep = id_to_index.contains_key(s);
                    if !keep {
                        id_to_index.remove(self.id.value(i));
                    }
                    keep
                }
                None => {
                    // Parent ID starts with t3 so is the post, which is never dangling.
                    true
                }
            });
            if valid_indices.len() == prev_len {
                break;
            }
            prev_len = valid_indices.len();
        }

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
        for &i in &valid_indices {
            depths.insert(i, 0);
        }

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

        let Some(max_depth) = depths.values().max().copied() else {
            return Vec::new();
        };
        let deepest: Vec<usize> = valid_indices
            .iter()
            .copied()
            .filter(|i| depths[i] == max_depth)
            .collect();
        let leaf = *deepest.choose(rng).unwrap();

        let mut chain = Vec::with_capacity(max_depth);
        let mut cur = Some(leaf);
        while let Some(i) = cur
            && chain.len() < max_depth
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

    /// Documents in a deterministic random order. Ordering by key alone is not a uniform
    /// shuffle, since the keys kept by heavily subsampled strata cluster near the maximum.
    fn into_documents(self, rng: &mut StdRng) -> Vec<Document> {
        let mut items: Vec<Keyed> = self
            .strata
            .into_values()
            .flat_map(FastHashMap::into_values)
            .flat_map(|reservoir| reservoir.heap.into_iter().map(|Reverse(item)| item))
            .collect();
        items.sort_unstable();
        items.shuffle(rng);
        items.into_iter().map(|item| item.doc).collect()
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::{Int64Array, LargeStringArray, StringArray};

    use super::*;

    /// (created_utc, subreddit, title, selftext, author_name)
    type PostRow = (
        i64,
        Option<&'static str>,
        &'static str,
        Option<&'static str>,
        Option<&'static str>,
    );

    fn test_config() -> DatasetConfig {
        DatasetConfig {
            seed: 0,
            work_dir: Default::default(),
            reddit_ingest_bucket: String::new(),
            reddit_ingest_path: String::new(),
            post_sample_size: 0,
            subreddit_uniform_mix: 0.0,
            year_uniform_mix: 0.0,
            prune_texts: ["[removed]".into()].into_iter().collect(),
            prune_authors: ["AutoModerator".into()].into_iter().collect(),
            total_vocab_size: 0,
        }
    }

    #[test]
    fn counts_exclude_filtered_posts() {
        // 2024-12-31T00:00:00Z and 2025-01-01T00:00:00Z.
        let (y2024, y2025) = (1_735_603_200, 1_735_689_600);
        let rows: [PostRow; 8] = [
            (y2024, Some("a"), "t", Some("s"), Some("u")),
            (y2024, Some("a"), "t", None, None),
            (y2024, Some("b"), "t", Some("s"), Some("u")),
            (y2025, Some("a"), "t", Some("s"), Some("u")),
            (y2025, None, "t", Some("s"), Some("u")),
            (y2025, Some("a"), " [removed] ", Some("s"), Some("u")),
            (y2025, Some("a"), "t", Some("[removed]"), Some("u")),
            (y2025, Some("b"), "t", Some("s"), Some("AutoModerator")),
        ];
        let batch = RecordBatch::try_from_iter([
            (
                "created_utc",
                Arc::new(Int64Array::from_iter_values(rows.iter().map(|r| r.0))) as ArrayRef,
            ),
            (
                "subreddit",
                Arc::new(StringArray::from_iter(rows.iter().map(|r| r.1))),
            ),
            (
                "title",
                Arc::new(StringArray::from_iter_values(rows.iter().map(|r| r.2))),
            ),
            (
                "selftext",
                Arc::new(LargeStringArray::from_iter(rows.iter().map(|r| r.3))),
            ),
            (
                "author_name",
                Arc::new(StringArray::from_iter(rows.iter().map(|r| r.4))),
            ),
        ])
        .unwrap();

        let mut counts = Strata::<u64>::default();
        count_batch(&batch, &test_config(), &mut counts).unwrap();
        let expected: Strata<u64> = [
            (
                2024,
                [("a".into(), 2), ("b".into(), 1)].into_iter().collect(),
            ),
            (2025, [("a".into(), 1)].into_iter().collect()),
        ]
        .into_iter()
        .collect();
        assert_eq!(counts, expected);
    }

    #[test]
    fn apportion_proportional() {
        assert_eq!(apportion(50, &[10, 30, 60], 0.0), [5, 15, 30]);
    }

    #[test]
    fn apportion_uniform() {
        assert_eq!(apportion(30, &[100, 0, 100, 100], 1.0), [10, 0, 10, 10]);
    }

    #[test]
    fn apportion_mixture() {
        // Shares: 0.5 * [0.1, 0.9] + 0.5 * [0.5, 0.5] = [0.3, 0.7].
        assert_eq!(apportion(100, &[100, 900], 0.5), [30, 70]);
    }

    #[test]
    fn apportion_redistributes_capped_excess() {
        // Uniform would give ~10 each, but the first category only has 1.
        assert_eq!(apportion(31, &[1, 1000, 1000], 1.0), [1, 15, 15]);
        // Shares are ~[0.18, 0.26, 0.56]; the remaining 90 is split ~[0.32, 0.68].
        assert_eq!(apportion(100, &[10, 100, 400], 0.5), [10, 29, 61]);
    }

    #[test]
    fn apportion_total_exceeds_counts() {
        assert_eq!(apportion(1000, &[3, 0, 7], 0.3), [3, 0, 7]);
    }

    #[test]
    fn apportion_sums_to_total() {
        let counts: Vec<u64> = (0..1000).map(|i| (i * i) % 997 + i / 10).collect();
        for mix in [0.0, 0.1, 0.5, 0.9, 1.0] {
            for total in [0, 1, 999, 12_345, 100_000] {
                let quotas = apportion(total, &counts, mix);
                assert_eq!(
                    quotas.iter().sum::<u64>(),
                    total.min(counts.iter().sum()),
                    "mix={mix} total={total}"
                );
                assert!(quotas.iter().zip(&counts).all(|(q, c)| q <= c));
            }
        }
    }
}
