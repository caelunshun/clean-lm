use std::{
    fmt::Debug,
    fs::{self, File},
    io::{BufWriter, Read, Seek, SeekFrom, Write},
};

use anyhow::Context;
use directories::ProjectDirs;
use parking_lot::{Mutex, RwLock};
use serde::{Serialize, de::DeserializeOwned};
use skytens_core::FastHashMap;

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

    pub fn get_or_create<K: DiskCacheKey, V: Serialize + DeserializeOwned>(
        &self,
        key: &K,
        create: impl FnOnce() -> anyhow::Result<V>,
    ) -> anyhow::Result<V> {
        let key_json = serde_json::to_string(key as &dyn DiskCacheKey)?;

        // Try to get from the in-memory cache.
        {
            let in_memory_cache = self.in_memory_cache.read();
            if let Some(val) = in_memory_cache.get(&key_json) {
                return serde_json::from_str(val).map_err(From::from);
            }
        }

        // Otherwise re-read the file and check for updates on disk.
        // (This is racy but not a correctness issue.)
        let mut file_guard = self.cache_file.lock();

        file_guard.lock()?;
        let res = (|| {
            file_guard.seek(SeekFrom::Start(0))?;
            let mut file_contents = String::new();
            file_guard.read_to_string(&mut file_contents)?;

            let mut on_disk_cache: FastHashMap<String, String> =
                serde_json::from_str(&file_contents).unwrap_or_default();
            if let Some(val) = on_disk_cache.get(&key_json) {
                return serde_json::from_str(val).map_err(From::from);
            }

            // Not in disk cache either. Create.
            let value = create()?;
            on_disk_cache.insert(key_json, serde_json::to_string(&value)?);

            file_guard.seek(SeekFrom::Start(0))?;
            file_guard.set_len(0)?;
            let mut buf_writer = BufWriter::new(&mut *file_guard);
            serde_json::to_writer_pretty(&mut buf_writer, &on_disk_cache)?;
            buf_writer.flush()?;
            drop(buf_writer);

            file_guard.unlock()?;

            *self.in_memory_cache.write() = on_disk_cache;

            Ok(value)
        })();
        file_guard.unlock().ok();
        res
    }
}

impl Drop for DiskCache {
    fn drop(&mut self) {
        self.cache_file.lock().unlock().ok();
    }
}

#[typetag::serde(tag = "type")]
pub trait DiskCacheKey {}
