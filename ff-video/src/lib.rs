mod decoder;
mod error;
mod parse;

pub use crate::error::{Error, VideoResult};
pub use decoder::{FFMpegDecoder, FFMpegDecoderBuilder};
