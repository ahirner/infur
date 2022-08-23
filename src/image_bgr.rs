use std::ops::{Deref, DerefMut};

use image::RgbImage;

/// Type to distinguish BGR from RGB encoding at compile time.
///
/// The native BGR PixelFormat was removed in 0.24:
/// https://github.com/image-rs/image/issues/1686
pub(crate) struct BgrImage(RgbImage);

impl BgrImage {
    pub(crate) fn new(image: RgbImage) -> Self {
        Self(image)
    }
}

impl Deref for BgrImage {
    type Target = RgbImage;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for BgrImage {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
