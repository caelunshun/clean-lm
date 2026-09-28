
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum InMemoryCompression {
    Lz4,
    Zstd { level: i32 },
}
