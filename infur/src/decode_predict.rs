use crate::app::Processor;
use eframe::epaint::{Color32, ColorImage};
use onnxruntime::ndarray::Array3;

/// 20 RGB high-contrast BGR/RGB triplets
///
/// adapted from: http://www.color-hex.com/color-palette/23381
/// and: http://www.color-hex.com/color-palette/52402
const COLORS_PALATTE: [(u8, u8, u8); 20] = [
    (75, 180, 60),
    (75, 25, 230),
    (25, 225, 255),
    (200, 130, 0),
    (48, 130, 245),
    (240, 240, 70),
    (230, 50, 240),
    (60, 245, 210),
    (180, 30, 145),
    (190, 190, 250),
    (128, 128, 0),
    (255, 190, 230),
    (40, 110, 170),
    (200, 250, 255),
    (0, 0, 128),
    (195, 255, 170),
    (0, 128, 128),
    (180, 215, 255),
    (128, 0, 0),
    (128, 128, 128),
];

fn color_code(klass: usize, alpha: f32) -> Color32 {
    let (r, g, b) = COLORS_PALATTE[klass % COLORS_PALATTE.len()];
    Color32::from_rgba_unmultiplied(r, g, b, (alpha * 255.0f32) as u8)
}

#[derive(Default)]
pub(crate) struct ColorCode;

impl Processor for ColorCode {
    type Command = ();
    type ControlError = ();
    /// KxHxW confidences
    type Input = Array3<f32>;
    type Output = Option<ColorImage>;
    type ProcessResult = ();

    fn control(&mut self, _cmd: Self::Command) -> Result<&mut Self, Self::ControlError> {
        Ok(self)
    }

    fn advance(&mut self, inp: &Self::Input, out: &mut Self::Output) -> Self::ProcessResult {
        let shape = inp.shape();
        let (k, h, w) = (shape[0], shape[1], shape[2]);

        // get or re-create output image
        let img = if let Some(ref mut img) = out {
            if img.width() != w || img.height() != h {
                *img = ColorImage::new([w, h], Color32::BLACK);
            }
            img
        } else {
            out.get_or_insert_with(|| ColorImage::new([w, h], Color32::BLACK))
        };

        let inp_flat = inp.exact_chunks([k, 1, 1]);
        img.pixels.iter_mut().zip(inp_flat).for_each(|(col, klasses)| {
            let mut k_max = 0;
            let mut c_max = 0f32;
            klasses.iter().enumerate().for_each(|(i, confidence)| {
                //println!("{}, {}", i, confidence);
                if confidence > &c_max {
                    k_max = i;
                    c_max = *confidence;
                }
            });
            *col = color_code(k_max, c_max);
        });
    }

    fn is_dirty(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod test {

    use onnxruntime::ndarray::Array1;

    use super::*;

    #[test]
    fn color_2() {
        let c = COLORS_PALATTE[2];
        assert_eq!(color_code(2, 0.5), Color32::from_rgba_unmultiplied(c.0, c.1, c.2, 127));
    }

    #[test]
    fn decode_0to1() {
        let hm = <Array1<f32>>::linspace(0., 1., 22 * 24 * 32).into_shape([22, 24, 32]).unwrap();
        let mut img = None;
        let mut decoder = ColorCode;
        decoder.advance(&hm, &mut img);

        let img = img.unwrap();
        assert_eq!(img.width(), 32);
        assert_eq!(img.height(), 24);
        let mut conf = 0;
        for p in img.pixels {
            assert_eq!(p, color_code(21, p.a() as f32 / 255f32));
            assert!(conf <= p.a(), "expected monotically rising confidence/alpha");
            conf = p.a();
        }
        assert_eq!(conf, 255);
    }
}
