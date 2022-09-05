mod decoder;
mod error;
mod parse;

pub use crate::error::{FFVideoError, VideoProcError, VideoResult};
pub use decoder::{FFMpegDecoder, FFMpegDecoderBuilder};
