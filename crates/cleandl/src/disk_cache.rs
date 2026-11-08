use anyhow::Context;
use directories::ProjectDirs;
use parking_lot::{Mutex, RwLock};
use serde::{Serialize, de::DeserializeOwned};
use skytens_core::FastHashMap;
use std::{
    fmt::Debug,
    fs::{self, File},
    io::{BufWriter, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    time::SystemTime,
};

/// Overrides the directory containing the cache file.
pub const CACHE_DIR_ENV: &str = "CLEANDL_CACHE_DIR";

#[derive(Debug)]
pub struct DiskCache {
    cache_file: Mutex<CacheFile>,
    in_memory_cache: RwLock<FastHashMap<String, String>>,
}

#[derive(Debug)]
struct CacheFile {
    file: File,
    /// `(len, mtime)` of the file when we last read or wrote it.
    last_seen: Option<(u64, SystemTime)>,
}

impl DiskCache {
    /// Opens the cache in `$CLEANDL_CACHE_DIR` if set, else the user's cache directory.
    pub fn new() -> anyhow::Result<Self> {
        let dir = match std::env::var_os(CACHE_DIR_ENV) {
            Some(dir) => PathBuf::from(dir),
            None => ProjectDirs::from("me.caelunshun", "caelunshun", "cleandl")
                .context("can't obtain dir for cache")?
                .cache_dir()
                .to_owned(),
        };
        Self::new_in(dir)
    }

    pub fn new_in(dir: impl AsRef<Path>) -> anyhow::Result<Self> {
        let dir = dir.as_ref();
        fs::create_dir_all(dir)?;

        let file = fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .read(true)
            .open(dir.join("cache.json"))?;

        Ok(Self {
            cache_file: Mutex::new(CacheFile {
                file,
                last_seen: None,
            }),
            in_memory_cache: RwLock::new(Default::default()),
        })
    }

    /// Returns the cached value for `key`, or runs `create` and caches its result.
    ///
    /// `create` runs without holding any locks, so concurrent callers may create
    /// the same key; the last write wins.
    pub fn get_or_create<K: DiskCacheKey, V: Serialize + DeserializeOwned>(
        &self,
        key: &K,
        create: impl FnOnce() -> anyhow::Result<V>,
    ) -> anyhow::Result<V> {
        if let Some(value) = self.get(key)? {
            return Ok(value);
        }
        let value = create()?;
        self.insert(key, &value)?;
        Ok(value)
    }

    pub fn get<K: DiskCacheKey, V: DeserializeOwned>(&self, key: &K) -> anyhow::Result<Option<V>> {
        let key_json = encode_key(key)?;

        if let Some(val) = self.in_memory_cache.read().get(&key_json) {
            return Ok(Some(serde_json::from_str(val)?));
        }

        // Pick up entries written by other processes or instances.
        let cached = self.with_locked_file(|file| {
            self.refresh(file)?;
            Ok(self.in_memory_cache.read().get(&key_json).cloned())
        })?;
        cached
            .map(|val| serde_json::from_str(&val))
            .transpose()
            .map_err(From::from)
    }

    pub fn insert<K: DiskCacheKey, V: Serialize>(&self, key: &K, value: &V) -> anyhow::Result<()> {
        let key_json = encode_key(key)?;
        let value_json = serde_json::to_string(value)?;

        self.with_locked_file(|file| {
            self.refresh(file)?;
            // After refreshing, the in-memory map is a superset of the file.
            let mut in_memory_cache = self.in_memory_cache.write();
            in_memory_cache.insert(key_json, value_json);

            file.file.seek(SeekFrom::Start(0))?;
            file.file.set_len(0)?;
            let mut buf_writer = BufWriter::new(&mut file.file);
            serde_json::to_writer_pretty(&mut buf_writer, &*in_memory_cache)?;
            buf_writer.flush()?;
            drop(buf_writer);
            file.last_seen = Some(file_stamp(&file.file)?);
            Ok(())
        })
    }

    /// Merges the file's contents into the in-memory map if it changed since we last saw it.
    /// Must be called with the file locked.
    fn refresh(&self, file: &mut CacheFile) -> anyhow::Result<()> {
        let stamp = file_stamp(&file.file)?;
        if file.last_seen == Some(stamp) {
            return Ok(());
        }

        file.file.seek(SeekFrom::Start(0))?;
        let mut file_contents = String::new();
        file.file.read_to_string(&mut file_contents)?;
        let on_disk_cache: FastHashMap<String, String> =
            serde_json::from_str(&file_contents).unwrap_or_default();

        self.in_memory_cache.write().extend(on_disk_cache);
        file.last_seen = Some(stamp);
        Ok(())
    }

    fn with_locked_file<R>(
        &self,
        f: impl FnOnce(&mut CacheFile) -> anyhow::Result<R>,
    ) -> anyhow::Result<R> {
        let mut file_guard = self.cache_file.lock();
        file_guard.file.lock()?;
        let res = f(&mut file_guard);
        file_guard.file.unlock()?;
        res
    }
}

fn encode_key<K: DiskCacheKey>(key: &K) -> anyhow::Result<String> {
    Ok(serde_json::to_string(key as &dyn DiskCacheKey)?)
}

fn file_stamp(file: &File) -> anyhow::Result<(u64, SystemTime)> {
    let metadata = file.metadata()?;
    Ok((metadata.len(), metadata.modified()?))
}

impl Drop for DiskCache {
    fn drop(&mut self) {
        self.cache_file.lock().file.unlock().ok();
    }
}

#[typetag::serde(tag = "type")]
pub trait DiskCacheKey {}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Serialize, Deserialize)]
    struct TestKey(u32);

    #[typetag::serde]
    impl DiskCacheKey for TestKey {}

    #[test]
    fn create_runs_unlocked() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let cache = DiskCache::new_in(dir.path())?;
        let value: u32 = cache.get_or_create(&TestKey(1), || {
            let inner: u32 = cache.get_or_create(&TestKey(2), || Ok(2))?;
            Ok(inner + 1)
        })?;
        assert_eq!(value, 3);
        assert_eq!(cache.get::<_, u32>(&TestKey(2))?, Some(2));
        Ok(())
    }

    #[test]
    fn instances_share_file() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let a = DiskCache::new_in(dir.path())?;
        let b = DiskCache::new_in(dir.path())?;
        assert_eq!(a.get::<_, u32>(&TestKey(1))?, None);
        b.insert(&TestKey(1), &10u32)?;
        a.insert(&TestKey(2), &20u32)?;
        assert_eq!(a.get::<_, u32>(&TestKey(1))?, Some(10));
        assert_eq!(b.get::<_, u32>(&TestKey(2))?, Some(20));

        let c = DiskCache::new_in(dir.path())?;
        assert_eq!(c.get::<_, u32>(&TestKey(1))?, Some(10));
        assert_eq!(c.get::<_, u32>(&TestKey(2))?, Some(20));
        Ok(())
    }
}
