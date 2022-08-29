mod decoder;
mod error;
mod parse;

pub use crate::error::{FFVideoError, VideoResult};
pub use decoder::{FFMpegDecoder, FFMpegDecoderBuilder};
