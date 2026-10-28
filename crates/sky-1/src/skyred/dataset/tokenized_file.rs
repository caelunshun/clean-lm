// --- LLM-generated --- //
//! Contiguous on-disk Blosc super-chunks of fixed-size items, written and read sequentially.

use anyhow::Context;
use blosc2::{
    CParams, CompressAlgo, DParams,
    chunk::{SChunk, SChunkStorageParams},
};
use bytemuck::Pod;
use std::{path::Path, thread::available_parallelism};

pub const TOKENIZED_FILE_NAME: &str = "posts_tokenized.sc";
pub const TOKEN_COUNTS_FILE_NAME: &str = "posts_token_counts.sc";

fn num_threads() -> usize {
    available_parallelism().map_or(1, |n| n.get())
}

fn dparams() -> DParams {
    let mut params = DParams::default();
    params.nthreads(num_threads());
    params
}

/// Buffers items and appends them to the super-chunk in chunks of exactly
/// `chunk_items` items (blosc requires all chunks but the last to be equal in size).
pub struct SChunkWriter<T> {
    schunk: SChunk,
    chunk_items: usize,
    pending: Vec<T>,
}

impl<T: Pod> SChunkWriter<T> {
    pub fn create(path: &Path, chunk_items: usize) -> anyhow::Result<Self> {
        if path.exists() {
            fs_err::remove_file(path)?;
        }

        let mut cparams = CParams::default();
        cparams
            .compressor(CompressAlgo::Zstd)
            .typesize(size_of::<T>())?
            .nthreads(num_threads());
        let schunk = SChunk::new_at(
            SChunkStorageParams {
                contiguous: true,
                urlpath: Some(path),
            },
            cparams,
            dparams(),
        )
        .with_context(|| format!("failed to create {}", path.display()))?;

        Ok(Self {
            schunk,
            chunk_items,
            pending: Vec::with_capacity(chunk_items),
        })
    }

    pub fn extend(&mut self, items: &[T]) -> anyhow::Result<()> {
        self.pending.extend_from_slice(items);
        if self.pending.len() >= self.chunk_items {
            for chunk in self.pending.chunks_exact(self.chunk_items) {
                self.schunk.append(bytemuck::cast_slice(chunk))?;
            }
            let num_flushed = self.pending.len() - self.pending.len() % self.chunk_items;
            self.pending.drain(..num_flushed);
        }
        Ok(())
    }

    pub fn finish(mut self) -> anyhow::Result<()> {
        if !self.pending.is_empty() {
            self.schunk.append(bytemuck::cast_slice(&self.pending))?;
        }
        Ok(())
    }
}

/// Decompresses one chunk at a time and hands out consecutive runs of items.
pub struct SChunkReader<T> {
    schunk: SChunk,
    next_chunk: usize,
    buffer: Vec<T>,
    pos: usize,
}

impl<T: Pod> SChunkReader<T> {
    pub fn open(path: &Path) -> anyhow::Result<Self> {
        let schunk =
            SChunk::open(path).with_context(|| format!("failed to open {}", path.display()))?;
        anyhow::ensure!(
            schunk.typesize() == size_of::<T>(),
            "{} has typesize {}, expected {}",
            path.display(),
            schunk.typesize(),
            size_of::<T>()
        );
        Ok(Self {
            schunk,
            next_chunk: 0,
            buffer: Vec::new(),
            pos: 0,
        })
    }

    /// Returns the next `n` items, or `None` if the file is exhausted.
    /// Errors if the file ends partway through.
    pub fn read(&mut self, n: usize) -> anyhow::Result<Option<&[T]>> {
        while self.buffer.len() - self.pos < n {
            if self.next_chunk == self.schunk.num_chunks() {
                anyhow::ensure!(
                    self.buffer.len() == self.pos,
                    "unexpected end of file: requested {n} items, {} remain",
                    self.buffer.len() - self.pos
                );
                return Ok(None);
            }
            let bytes = self.decompress_chunk(self.next_chunk)?;
            self.next_chunk += 1;

            self.buffer.drain(..self.pos);
            self.pos = 0;
            extend_from_bytes(&mut self.buffer, &bytes);
        }

        let items = &self.buffer[self.pos..][..n];
        self.pos += n;
        Ok(Some(items))
    }

    pub fn read_all(mut self) -> anyhow::Result<Vec<T>> {
        let mut items = self.buffer.split_off(self.pos);
        for i in self.next_chunk..self.schunk.num_chunks() {
            extend_from_bytes(&mut items, &self.decompress_chunk(i)?);
        }
        Ok(items)
    }

    fn decompress_chunk(&mut self, index: usize) -> anyhow::Result<Vec<u8>> {
        let chunk = self.schunk.get_chunk(index)?;
        chunk.set_dparams(dparams())?;
        Ok(chunk.decompress()?)
    }
}

/// `bytes` may not be aligned for `T`.
fn extend_from_bytes<T: Pod>(items: &mut Vec<T>, bytes: &[u8]) {
    items.extend(
        bytes
            .chunks_exact(size_of::<T>())
            .map(bytemuck::pod_read_unaligned::<T>),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.sc");
        let items: Vec<u32> = (0..10_000u32).map(|i| i.wrapping_mul(2654435761)).collect();

        // Existing files are replaced.
        fs_err::write(&path, b"garbage").unwrap();

        let mut writer = SChunkWriter::<u32>::create(&path, 1000).unwrap();
        for group in items.chunks(337) {
            writer.extend(group).unwrap();
        }
        writer.extend(&[1, 2, 3]).unwrap();
        writer.finish().unwrap();

        let mut expected = items.clone();
        expected.extend([1, 2, 3]);

        assert_eq!(
            SChunkReader::<u32>::open(&path)
                .unwrap()
                .read_all()
                .unwrap(),
            expected
        );

        let mut reader = SChunkReader::<u32>::open(&path).unwrap();
        let mut pos = 0;
        for n in [1, 999, 1500, 0, 2501, 5002] {
            assert_eq!(reader.read(n).unwrap().unwrap(), &expected[pos..][..n]);
            pos += n;
        }
        assert_eq!(pos, expected.len());
        assert!(reader.read(1).unwrap().is_none());

        let mut reader = SChunkReader::<u32>::open(&path).unwrap();
        reader.read(10_000).unwrap();
        assert!(reader.read(4).is_err());
    }
}
