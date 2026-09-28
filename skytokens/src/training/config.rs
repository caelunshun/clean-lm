#[derive(Debug, Clone)]
pub struct Config {
    /// Must be at least 256 to allow the initial per-byte token set.
    pub target_vocab_size: u32,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum InMemoryCompression {
    Lz4,
    Zstd { level: i32 },
}
