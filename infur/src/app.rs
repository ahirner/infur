/// Example Application
use eframe::epaint::ColorImage;
use ff_video::{FFVideoError, VideoProcError};
use image_ext::Pixel;
use onnxruntime::ndarray::Array3;
use thiserror::Error;

use crate::{
    decode_predict::ColorCode,
    predict_onnx::{Model, ModelCmd, ModelCmdError, ModelInfo, ModelProcError},
    processing::{Frame, Scale, ScaleProcError, ValidScaleError, VideoCmd, VideoPlayer},
};

pub(crate) use crate::processing::Processor;

/// Application processing error
#[derive(Error, Debug)]
pub(crate) enum AppProcError {
    #[error(transparent)]
    Video(#[from] VideoProcError),
    #[error(transparent)]
    Scale(#[from] ScaleProcError),
    #[error(transparent)]
    Model(#[from] ModelProcError),
}

/// Application command processing error
#[derive(Error, Debug)]
pub(crate) enum AppCmdError {
    #[error(transparent)]
    Scale(#[from] ValidScaleError),
    #[error(transparent)]
    Video(#[from] FFVideoError),
    #[error(transparent)]
    Model(#[from] ModelCmdError),
}

/// Control entire application
#[derive(Clone, Debug)]
pub(crate) enum AppCmd {
    /// Control video input
    Video(VideoCmd),
    /// Control scale factor
    Scale(f32),
    /// Control loaded model, empty disables it
    Model(ModelCmd),
    /// Exit App
    Exit,
}

/// Example app
#[derive(Default)]
pub(crate) struct ProcessingApp<'m> {
    vid: VideoPlayer,
    scale: Scale,
    frame: Option<Frame>,
    scaled_frame: Option<Frame>,
    model: Model<'m>,
    decoder: ColorCode,
    decoded_img: Option<ColorImage>,
    pub(crate) to_exit: bool,
}

/// Frame transmitted to GUI
pub(crate) struct GUIFrame {
    pub(crate) id: u64,
    pub(crate) buffer: ColorImage,
    pub(crate) decoded_buffer: Option<ColorImage>,
}

#[derive(Clone, Debug)]
/// Information on current state of app
pub(crate) struct AppInfo {
    pub(crate) model_info: Option<ModelInfo>,
}

impl ProcessingApp<'_> {
    pub(crate) fn info(&self) -> AppInfo {
        let model_info = self.model.get_info().cloned();
        AppInfo { model_info }
    }
}

impl Processor for ProcessingApp<'_> {
    type Command = AppCmd;
    type ControlError = AppCmdError;
    type Input = ();
    type Output = ();
    type ProcessResult = Result<Option<GUIFrame>, AppProcError>;

    fn control(&mut self, cmd: Self::Command) -> Result<&mut Self, Self::ControlError> {
        match cmd {
            AppCmd::Video(cmd) => {
                self.vid.control(cmd)?;
            }
            AppCmd::Scale(cmd) => {
                self.scale.control(cmd)?;
            }
            AppCmd::Exit => self.to_exit = true,
            AppCmd::Model(cmd) => {
                self.model.control(cmd)?;
            }
        };
        Ok(self)
    }

    fn advance(&mut self, input: &(), _out: &mut ()) -> Self::ProcessResult {
        self.vid.advance(input, &mut self.frame)?;
        if self.is_dirty() {
            self.scale.advance(&self.frame, &mut self.scaled_frame)?;
        };
        if let Some(scaled_frame) = &self.scaled_frame {
            let mut out = vec![];
            self.model.advance(&scaled_frame.img, &mut out)?;
            if !out.is_empty() {
                let out = &out[0];
                let shape = out.shape();
                let shape = [shape[0], shape[1], shape[2]];
                // todo: find way to not clone
                let hm: Array3<f32> =
                    Array3::from_shape_vec(shape, out.clone().into_raw_vec()).unwrap();

                self.decoder.advance(&hm, &mut self.decoded_img);

                // todo: send input image and color coded img
                //return Ok(Some(GUIFrame { id: scaled_frame.id, buffer: img.unwrap() }));
            } else {
                self.decoded_img = None;
            }

            // todo: trait and/or processor
            let rgba_pixels = scaled_frame
                .img
                .pixels()
                .map(|p| {
                    let cs = p.channels();
                    eframe::epaint::Color32::from_rgb(cs[2], cs[1], cs[0])
                })
                .collect::<Vec<_>>();

            let col_img = ColorImage {
                size: [scaled_frame.img.width() as usize, scaled_frame.img.height() as usize],
                pixels: rgba_pixels,
            };
            Ok(Some(GUIFrame {
                id: scaled_frame.id,
                buffer: col_img,
                decoded_buffer: self.decoded_img.clone(),
            }))
        } else {
            Ok(None)
        }
    }

    fn is_dirty(&self) -> bool {
        self.vid.is_dirty() || self.scale.is_dirty()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use infur_test_gen::{long_small_video, short_large_video};

    /// 640x480
    fn short_large_input() -> Vec<String> {
        vec![short_large_video().to_string_lossy().to_string()]
    }
    /// 1280x720
    fn long_small_input() -> Vec<String> {
        vec![long_small_video().to_string_lossy().to_string()]
    }

    #[test]
    fn void() {
        let mut app = ProcessingApp::default();
        assert!(app.generate().unwrap().is_none());
        assert!(app.generate().unwrap().is_none());
    }

    #[test]
    fn scale() {
        let mut app = ProcessingApp::default();
        app.control(AppCmd::Video(VideoCmd::Play(short_large_input()))).unwrap();
        app.control(AppCmd::Scale(0.5)).unwrap();
        let f2 = app.generate().unwrap().expect("video should already play");
        assert_eq!(f2.buffer.size, [1280 / 2, 720 / 2]);
    }

    #[test]
    fn switch_scale() {
        let mut app = ProcessingApp::default();
        app.control(AppCmd::Video(VideoCmd::Play(long_small_input()))).unwrap();
        let f1 = app.generate().unwrap().expect("video should already play");
        assert_eq!(f1.buffer.size, [640, 480]);

        app.control(AppCmd::Scale(0.5)).unwrap();
        let f2 = app.generate().unwrap().expect("video should keep playing");
        assert_eq!(f2.buffer.size, [640 / 2, 480 / 2]);
    }

    #[test]
    fn switch_video_then_scale() {
        let mut app = ProcessingApp::default();
        // in 640x480 out same
        app.control(AppCmd::Video(VideoCmd::Play(long_small_input()))).unwrap();
        let f1 = app.generate().unwrap().unwrap();
        assert_eq!(f1.buffer.size, [640, 480]);
        // in 1280x720 out same
        app.control(AppCmd::Video(VideoCmd::Play(short_large_input()))).unwrap();
        let f2 = app.generate().unwrap().unwrap();
        assert_eq!(f2.buffer.size, [1280, 720]);
        // in 1280x720 out twice
        app.control(AppCmd::Scale(2.0)).unwrap();
        let f3 = app.generate().unwrap().unwrap();
        assert_eq!(f3.buffer.size, [1280 * 2, 720 * 2]);
    }

    #[test]
    fn scaled_frame_after_stopped_video() {
        let mut app = ProcessingApp::default();
        app.control(AppCmd::Video(VideoCmd::Play(short_large_input()))).unwrap();
        let f1 = app.generate().unwrap().unwrap();
        assert_eq!(f1.buffer.size, [1280, 720]);
        app.control(AppCmd::Video(VideoCmd::Stop)).unwrap();
        let f2 = app.generate().unwrap().unwrap();
        assert_eq!(f1.id, f2.id);
        assert!(!app.is_dirty());

        app.control(AppCmd::Scale(0.5)).unwrap();
        assert!(app.is_dirty());
        let f3 = app.generate().unwrap().unwrap();
        assert_eq!(f2.id, f3.id);
        assert_eq!(f3.buffer.size, [1280 / 2, 720 / 2]);
    }

    #[test]
    fn pause_video() {
        let mut app = ProcessingApp::default();
        app.control(AppCmd::Video(VideoCmd::Play(long_small_input()))).unwrap();
        let f1 = app.generate().unwrap().unwrap();
        app.control(AppCmd::Video(VideoCmd::Pause(true))).unwrap();
        assert!(!app.is_dirty());
        let f2 = app.generate().unwrap().unwrap();
        assert_eq!(f1.id, f2.id);
        assert!(!app.is_dirty());

        app.control(AppCmd::Video(VideoCmd::Pause(false))).unwrap();
        assert!(app.is_dirty());
        let f3 = app.generate().unwrap().unwrap();
        assert_ne!(f2.id, f3.id);
    }
}
