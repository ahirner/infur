use image::{ImageBuffer, Rgb};

/// For Bgr ImageBuffer.
///
/// The native BGR PixelFormat was removed in 0.24:
/// <https://github.com/image-rs/image/issues/1686>
#[repr(C)]
#[derive(Copy, Clone)]
pub struct Bgr(pub [u8; 3]);

pub type BgrImage = ImageBuffer<Bgr, Vec<u8>>;

impl image::Pixel for Bgr {
    type Subpixel = u8;

    const CHANNEL_COUNT: u8 = 3;
    const COLOR_MODEL: &'static str = "BGR";

    fn channels(&self) -> &[Self::Subpixel] {
        &self.0
    }

    fn channels_mut(&mut self) -> &mut [Self::Subpixel] {
        &mut self.0
    }

    fn channels4(&self) -> (Self::Subpixel, Self::Subpixel, Self::Subpixel, Self::Subpixel) {
        let mut channels = [Self::Subpixel::MAX; 4];
        channels[0..3].copy_from_slice(&self.0);
        (channels[0], channels[1], channels[2], channels[3])
    }

    fn from_channels(
        a: Self::Subpixel,
        b: Self::Subpixel,
        c: Self::Subpixel,
        _d: Self::Subpixel,
    ) -> Self {
        Self([a, b, c])
    }

    fn from_slice(slice: &[Self::Subpixel]) -> &Self {
        let slice3 = slice.get(..3).unwrap();
        // overheard that RangeTo is const on nightly...
        unsafe { &*(slice3.as_ptr() as *const Bgr) }
    }

    fn from_slice_mut(slice: &mut [Self::Subpixel]) -> &mut Self {
        let slice3 = slice.get_mut(..3).unwrap();
        // overheard that RangeTo is const on nightly...
        unsafe { &mut *(slice3.as_mut_ptr() as *mut Bgr) }
    }

    fn to_rgb(&self) -> Rgb<Self::Subpixel> {
        #[allow(deprecated)]
        Rgb::from_channels(self.0[2], self.0[1], self.0[0], 0)
    }

    fn to_rgba(&self) -> image::Rgba<Self::Subpixel> {
        #[allow(deprecated)]
        image::Rgba::from_channels(self.0[2], self.0[1], self.0[0], Self::Subpixel::MAX)
    }

    fn to_luma(&self) -> image::Luma<Self::Subpixel> {
        todo!()
    }

    fn to_luma_alpha(&self) -> image::LumaA<Self::Subpixel> {
        todo!()
    }

    fn map<F>(&self, f: F) -> Self
    where
        F: FnMut(Self::Subpixel) -> Self::Subpixel,
    {
        Self(self.0.map(f))
    }

    fn apply<F>(&mut self, mut f: F)
    where
        F: FnMut(Self::Subpixel) -> Self::Subpixel,
    {
        for v in &mut self.0 {
            *v = f(*v)
        }
    }

    fn map_with_alpha<F, G>(&self, f: F, _g: G) -> Self
    where
        F: FnMut(Self::Subpixel) -> Self::Subpixel,
        G: FnMut(Self::Subpixel) -> Self::Subpixel,
    {
        self.map(f)
    }

    fn apply_with_alpha<F, G>(&mut self, f: F, _g: G)
    where
        F: FnMut(Self::Subpixel) -> Self::Subpixel,
        G: FnMut(Self::Subpixel) -> Self::Subpixel,
    {
        self.apply(f)
    }

    fn map2<F>(&self, other: &Self, mut f: F) -> Self
    where
        F: FnMut(Self::Subpixel, Self::Subpixel) -> Self::Subpixel,
    {
        let pixels = [f(self.0[0], other.0[0]), f(self.0[1], other.0[1]), f(self.0[1], other.0[1])];
        Self(pixels)
    }

    fn apply2<F>(&mut self, other: &Self, mut f: F)
    where
        F: FnMut(Self::Subpixel, Self::Subpixel) -> Self::Subpixel,
    {
        self.0[0] = f(self.0[0], other.0[0]);
        self.0[1] = f(self.0[1], other.0[1]);
        self.0[2] = f(self.0[1], other.0[2]);
    }

    fn invert(&mut self) {
        self.0[0] = Self::Subpixel::MAX - self.0[0];
        self.0[1] = Self::Subpixel::MAX - self.0[1];
        self.0[2] = Self::Subpixel::MAX - self.0[2];
    }

    #[allow(unused_variables)]
    fn blend(&mut self, other: &Self) {
        todo!()
    }
}
