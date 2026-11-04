use thiserror::Error;

pub type Result<T> = std::result::Result<T, MemoraError>;

#[derive(Debug, Error)]
pub enum MemoraError {
    #[error("store: {0}")]
    Store(String),
    #[error("not found: {0}")]
    NotFound(String),
    #[error("invalid state: {0}")]
    InvalidState(String),
    /// Write rejected by credential / content-safety filter.
    #[error("rejected: {0}")]
    Rejected(String),
    #[error("embedding: {0}")]
    Embedding(String),
    #[error("retrieval: {0}")]
    Retrieval(String),
    #[error("import: {0}")]
    Import(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("task join: {0}")]
    Join(#[from] tokio::task::JoinError),
    #[error("sqlite: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("pool: {0}")]
    Pool(#[from] r2d2::Error),
    #[error("migration: {0}")]
    Migration(String),
    #[error("zip: {0}")]
    Zip(String),
    #[error("{0}")]
    Other(String),
}

impl From<rusqlite_migration::Error> for MemoraError {
    fn from(value: rusqlite_migration::Error) -> Self {
        Self::Migration(value.to_string())
    }
}

impl From<zip::result::ZipError> for MemoraError {
    fn from(value: zip::result::ZipError) -> Self {
        Self::Zip(value.to_string())
    }
}
