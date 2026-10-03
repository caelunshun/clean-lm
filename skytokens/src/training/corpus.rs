use std::{
    io::Cursor,
    iter, mem,
    sync::{
        Arc, OnceLock,
        atomic::{AtomicUsize, Ordering},
    },
    thread::available_parallelism,
};

use simple_semaphore::Semaphore;

use crate::training::config::InMemoryCompression;

/// Approx. size of uncompressed text data in each chunk.
pub(super) const CHUNK_SIZE: usize = 64 * 1024;

#[derive(Debug, Clone)]
pub struct CorpusBuilder {
    compression: Option<InMemoryCompression>,
    chunks: Vec<Arc<OnceLock<Chunk>>>,
    concurrency_limiter: Arc<Semaphore>,

    current_chunk_data: Vec<u8>,
    current_chunk_doc_boundaries: Vec<u32>,

    /// index i stores number of occurrences of byte pair (`i & 0xFF`, `i >> 8`)
    byte_pair_occurrences: Arc<[AtomicUsize]>,
    total_uncompressed_bytes: usize,
}

impl CorpusBuilder {
    pub fn new(compression: Option<InMemoryCompression>) -> Self {
        Self {
            compression,
            chunks: Vec::new(),
            concurrency_limiter: Semaphore::new(available_parallelism().unwrap().get() * 2),
            current_chunk_data: Vec::with_capacity(2 * CHUNK_SIZE),
            current_chunk_doc_boundaries: Vec::new(),
            byte_pair_occurrences: iter::repeat_with(|| AtomicUsize::new(0))
                .take(1 << 16)
                .collect(),
            total_uncompressed_bytes: 0,
        }
    }

    fn flush_chunk(&mut self) {
        let data = mem::take(&mut self.current_chunk_data);
        self.current_chunk_data.reserve(2 * CHUNK_SIZE);
        let doc_boundaries = mem::take(&mut self.current_chunk_doc_boundaries);

        let chunk = Arc::new(OnceLock::new());
        self.chunks.push(chunk.clone());

        let compression = self.compression;
        let permit = self.concurrency_limiter.acquire();

        let byte_pair_occurrences = self.byte_pair_occurrences.clone();

        rayon::spawn(move || {
            let mut local_byte_pair_occurrences = hashbrown::HashMap::new();
            let mut prev_doc_end_index = 0;
            for &doc_end_index in &doc_boundaries {
                let doc = &data[prev_doc_end_index as usize..doc_end_index as usize];
                for &[a, b] in doc.array_windows::<2>() {
                    let i = (a as u16) | ((b as u16) << 8);
                    *local_byte_pair_occurrences.entry(i).or_default() += 1;
                }
                prev_doc_end_index = doc_end_index;
            }

            for (i, count) in local_byte_pair_occurrences {
                byte_pair_occurrences[i as usize].fetch_add(count, Ordering::Relaxed);
            }

            let data = match compression {
                Some(compression) => compress(&data, compression),
                None => data,
            };
            chunk
                .set(Chunk {
                    data: data.into(),
                    doc_boundaries: doc_boundaries.into(),
                })
                .ok()
                .unwrap();

            drop(permit);
        });
    }

    pub fn add_document(&mut self, document: impl AsRef<[u8]>) -> anyhow::Result<()> {
        self.total_uncompressed_bytes += document.as_ref().len();
        self.current_chunk_data.extend_from_slice(document.as_ref());
        self.current_chunk_doc_boundaries
            .push(self.current_chunk_data.len().try_into()?);

        if self.current_chunk_data.len() >= CHUNK_SIZE {
            self.flush_chunk();
        }

        Ok(())
    }

    pub fn build(mut self) -> Corpus {
        self.flush_chunk();
        let chunks = self
            .chunks
            .iter()
            .map(|c| c.wait().clone())
            .collect::<Vec<_>>();

        let mut byte_pair_occurrences = hashbrown::HashMap::new();
        for (i, count) in self.byte_pair_occurrences.iter().enumerate() {
            let a = (i & 0xFF) as u8;
            let b = ((i >> 8) & 0xFF) as u8;
            byte_pair_occurrences.insert((a, b), count.load(Ordering::Relaxed));
        }

        Corpus {
            chunks,
            total_uncompressed_bytes: self.total_uncompressed_bytes,
            byte_pair_occurrences,
            compression: self.compression,
        }
    }
}

fn compress(data: &[u8], compression: InMemoryCompression) -> Vec<u8> {
    match compression {
        InMemoryCompression::Lz4 => lz4_flex::compress_prepend_size(data),
        InMemoryCompression::Zstd { level } => {
            zstd::bulk::compress(data, level).expect("zstd failed?")
        }
    }
}

pub(super) fn decompress(compressed_data: &[u8], compression: InMemoryCompression) -> Vec<u8> {
    match compression {
        InMemoryCompression::Lz4 => lz4_flex::decompress_size_prepended(compressed_data)
            .expect("lz4 compression failed roundtrip"),
        InMemoryCompression::Zstd { .. } => zstd::decode_all(Cursor::new(compressed_data))
            .expect("zstd compression failed roundtrip"),
    }
}

#[derive(Debug, Clone)]
pub struct Corpus {
    pub(super) compression: Option<InMemoryCompression>,
    pub(super) chunks: Vec<Chunk>,
    #[allow(unused)]
    pub(super) total_uncompressed_bytes: usize,
    pub(super) byte_pair_occurrences: hashbrown::HashMap<(u8, u8), usize>,
}

impl Corpus {
    pub fn builder(compression: Option<InMemoryCompression>) -> CorpusBuilder {
        CorpusBuilder::new(compression)
    }
}

#[derive(Debug, Clone)]
pub(super) struct Chunk {
    /// Compressed if in-memory compression is enabled.
    pub(super) data: Arc<[u8]>,
    /// Sorted boundary positions between different documents
    /// within the chunk.
    pub(super) doc_boundaries: Arc<[u32]>,
}
