mod decoder;
mod error;
mod parse;

pub use crate::error::Error;
pub use decoder::{FFMpegDecoder, FFMpegDecoderBuilder};
