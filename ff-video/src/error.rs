use crate::parse::ParseError;
pub use thiserror::Error;

/// Results from processing a video.
pub type VideoResult<T> = std::result::Result<T, VideoProcError>;
/// Results from parsing video info while processing it.
pub type InfoResult<T> = std::result::Result<T, ParseError>;

#[derive(Error, Debug)]
pub enum VideoProcError {
    #[error("video IO failed with {msg}")]
    IO {
        msg: String,
        #[source]
        source: std::io::Error,
    },
    #[error("couldn't parse stream info within deadline ({0})")]
    Start(String),
    #[error("couldn't obtain {0}")]
    MissingValue(String),
    #[error("video process exit code: {0}")]
    ExitCode(i32),
    #[error("other error: {0}")]
    Other(String),
}

impl VideoProcError {
    pub(crate) fn is_missing(msg: impl ToString) -> Self {
        Self::MissingValue(msg.to_string())
    }
    pub(crate) fn explain_io(msg: impl ToString, e: std::io::Error) -> Self {
        Self::IO { msg: msg.to_string(), source: e }
    }
}

#[derive(Error, Debug)]
pub enum FFVideoError {
    #[error("video processing error: {0}")]
    Processing(#[from] VideoProcError),
    #[error("parsing error: {0}")]
    Parsing(#[from] ParseError),
}

#[cfg(test)]
mod tests {
    // test lifted from image crate
    use super::*;
    use std::mem;

    #[allow(dead_code)]
    // This will fail to compile if the size of this type is large.
    const ASSERT_SMALLISH: usize = [0][(mem::size_of::<FFVideoError>() >= 200) as usize];

    #[test]
    fn test_send_sync_stability() {
        fn assert_send_sync<T: Send + Sync>() {}

        assert_send_sync::<FFVideoError>();
    }
}
