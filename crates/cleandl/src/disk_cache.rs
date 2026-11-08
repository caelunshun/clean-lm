use anyhow::Context;
use directories::ProjectDirs;
use parking_lot::{Mutex, RwLock};
use serde::{Serialize, de::DeserializeOwned};
use skytens_core::FastHashMap;
use std::{
    fmt::Debug,
    fs::{self, File},
    io::{BufWriter, Read, Seek, SeekFrom, Write},
};

#[derive(Debug)]
pub struct DiskCache {
    cache_file: Mutex<File>,
    in_memory_cache: RwLock<FastHashMap<String, String>>,
}

impl DiskCache {
    pub fn new() -> anyhow::Result<Self> {
        let dirs = ProjectDirs::from("me.caelunshun", "caelunshun", "cleandl")
            .context("can't obtain dir for cache")?;
        fs::create_dir_all(dirs.cache_dir())?;

        let cache_file = fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .read(true)
            .open(dirs.cache_dir().join("cache.json"))?;

        Ok(Self {
            cache_file: Mutex::new(cache_file),
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

        // Re-read the file to pick up entries written by other processes.
        let on_disk_cache = self.with_locked_file(read_cache_file)?;
        let cached = on_disk_cache.get(&key_json).cloned();
        *self.in_memory_cache.write() = on_disk_cache;
        cached
            .map(|val| serde_json::from_str(&val))
            .transpose()
            .map_err(From::from)
    }

    pub fn insert<K: DiskCacheKey, V: Serialize>(&self, key: &K, value: &V) -> anyhow::Result<()> {
        let key_json = encode_key(key)?;
        let value_json = serde_json::to_string(value)?;

        let on_disk_cache = self.with_locked_file(|file| {
            let mut on_disk_cache = read_cache_file(file)?;
            on_disk_cache.insert(key_json, value_json);

            file.seek(SeekFrom::Start(0))?;
            file.set_len(0)?;
            let mut buf_writer = BufWriter::new(&mut *file);
            serde_json::to_writer_pretty(&mut buf_writer, &on_disk_cache)?;
            buf_writer.flush()?;
            Ok(on_disk_cache)
        })?;
        *self.in_memory_cache.write() = on_disk_cache;
        Ok(())
    }

    fn with_locked_file<R>(
        &self,
        f: impl FnOnce(&mut File) -> anyhow::Result<R>,
    ) -> anyhow::Result<R> {
        let mut file_guard = self.cache_file.lock();
        file_guard.lock()?;
        let res = f(&mut file_guard);
        file_guard.unlock()?;
        res
    }
}

fn encode_key<K: DiskCacheKey>(key: &K) -> anyhow::Result<String> {
    Ok(serde_json::to_string(key as &dyn DiskCacheKey)?)
}

fn read_cache_file(file: &mut File) -> anyhow::Result<FastHashMap<String, String>> {
    file.seek(SeekFrom::Start(0))?;
    let mut file_contents = String::new();
    file.read_to_string(&mut file_contents)?;
    Ok(serde_json::from_str(&file_contents).unwrap_or_default())
}

impl Drop for DiskCache {
    fn drop(&mut self) {
        self.cache_file.lock().unlock().ok();
    }
}

#[typetag::serde(tag = "type")]
pub trait DiskCacheKey {}
