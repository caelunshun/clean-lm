#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Cuda(#[from] skycuda::error::Error),
    #[error("unsupported operation regime: {0}")]
    Unsupported(&'static str),
}

pub type Result<T> = std::result::Result<T, Error>;
