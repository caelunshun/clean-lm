#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Cuda(#[from] skycuda::error::Error),
    #[error(transparent)]
    BuildError(#[from] nvvm_builder::BuildError),
    #[error(transparent)]
    CompileError(#[from] nvvm_builder::CompileError),
    #[error(transparent)]
    ToolError(#[from] nvvm_builder::tools::ToolError),
    #[error("unsupported operation regime: {0}")]
    Unsupported(&'static str),
}

pub type Result<T> = std::result::Result<T, Error>;
