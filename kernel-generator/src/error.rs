#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Cuda(#[from] skycuda::error::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
