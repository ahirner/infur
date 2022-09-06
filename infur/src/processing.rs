use std::{error::Error as StdError, fmt::Display, num::NonZeroU32, ops::Deref};

use eframe::epaint::ColorImage;
use fast_image_resize as fr;
use ff_video::{FFMpegDecoder, FFMpegDecoderBuilder, FFVideoError, VideoProcError, VideoResult};
use image_ext::{BgrImage, Pixel};
use thiserror::Error;

/// Frame transmitted to GUI
pub(crate) struct GUIFrame {
    pub(crate) id: u64,
    pub(crate) buffer: ColorImage,
}

/// Frame produced and processed
struct Frame {
    id: u64,
    img: BgrImage,
}

impl PartialEq for Frame {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

/// Transform expensive to clone data
///
/// Other parameters affecting output and results are controlled by a single message type.
pub(crate) trait Processor {
    /// Type of commands to control the processor
    type Command;

    /// Error to expect when controlling the processor
    type ControlError;

    /// Input to advance with
    type Input;

    /// Item mutated on advance
    ///
    /// For why we can't impl Iterator see:
    /// <http://lukaskalbertodt.github.io/2018/08/03/solving-the-generalized-streaming-iterator-problem-without-gats.html>
    type Output;

    /// Return value of processing input
    type ProcessResult;

    /// Affect processor parameters
    fn control(&mut self, cmd: Self::Command) -> Result<&mut Self, Self::ControlError>;

    /// Process and store a new result
    fn advance(&mut self, inp: &Self::Input, out: &mut Self::Output) -> Self::ProcessResult;

    /// True if passing the same input into advance writes new results
    fn is_dirty(&self) -> bool;

    /// Generate results for default input/output (usually final processing nodes)
    fn generate(&mut self) -> <Self as Processor>::ProcessResult
    where
        Self::Input: Default,
        Self::Output: Default,
    {
        self.advance(&Self::Input::default(), &mut Self::Output::default())
    }
}

/// Commands that control VideoPlayer
#[derive(Debug, Clone)]
pub(crate) enum VideoCmd {
    /// Start or restart playing video from this ffmpeg input
    Play(Vec<String>),
    /// Stop whenever
    Stop,
}

/// Writes video frames at command
#[derive(Default)]
struct VideoPlayer {
    vid: Option<FFMpegDecoder>,
    input: Vec<String>,
}

impl VideoPlayer {
    fn close_video(&mut self) -> VideoResult<()> {
        self.vid.take().map_or(Ok(()), |vid| vid.close())
    }
}

impl Processor for VideoPlayer {
    type Command = VideoCmd;
    type ControlError = FFVideoError;
    type Input = ();
    type Output = Option<Frame>;
    type ProcessResult = VideoResult<()>;

    fn control(&mut self, cmd: Self::Command) -> Result<&mut Self, Self::ControlError> {
        match cmd {
            Self::Command::Play(input) => {
                self.close_video()?;
                self.input = input;
                let builder = FFMpegDecoderBuilder::default().input(self.input.clone());
                self.vid = Some(FFMpegDecoder::try_new(builder)?);
            }
            Self::Command::Stop => {
                self.close_video()?;
            }
        }
        Ok(self)
    }

    fn is_dirty(&self) -> bool {
        self.vid.is_some()
    }

    fn advance(&mut self, _inp: &(), out: &mut Self::Output) -> Self::ProcessResult {
        if let Some(vid) = self.vid.as_mut() {
            // either reuse, re-create (on size change) or create a frame with suitable buffer
            let frame = if let Some(ref mut frame) = out {
                if frame.img.width() != vid.video_output.width
                    || frame.img.height() != vid.video_output.height
                {
                    frame.img = vid.empty_image();
                }
                frame
            } else {
                out.get_or_insert_with(|| Frame { id: 0, img: vid.empty_image() })
            };
            let id = vid.read_frame(&mut frame.img);
            if let Err(VideoProcError::FinishedNormally { .. }) = id {
                self.close_video()?;
            }
            frame.id = id?;
        };
        Ok(())
    }
}

#[derive(PartialEq, Debug)]
struct ValidScale(f32);

#[derive(Debug, Clone)]
pub(crate) struct ValidScaleError {
    msg: &'static str,
}

impl Display for ValidScaleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.msg)
    }
}

impl StdError for ValidScaleError {}

impl TryFrom<f32> for ValidScale {
    type Error = ValidScaleError;

    fn try_from(value: f32) -> Result<Self, Self::Error> {
        if value <= 0.0f32 {
            Err(ValidScaleError { msg: "Cannot scale by negative number" })
        } else {
            Ok(Self(value))
        }
    }
}

impl Deref for ValidScale {
    type Target = f32;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Scale frames by a constant factor
struct Scale {
    factor: ValidScale,
    resizer: fr::Resizer,
}

impl Default for Scale {
    fn default() -> Self {
        Self { factor: ValidScale(1.0f32), resizer: fr::Resizer::new(fr::ResizeAlg::Nearest) }
    }
}

impl Scale {
    fn is_unit_scale(&self) -> bool {
        self.factor.0 == 1.0f32
    }
}
/// Error processing scale
#[derive(Error, Debug)]
pub(crate) enum ScaleProcError {
    #[error("scaling to 0-sized output")]
    ZeroSize,
    #[error(transparent)]
    PixelType(#[from] fr::DifferentTypesOfPixelsError),
    #[error(transparent)]
    BufferError(#[from] fr::ImageBufferError),
}

impl Processor for Scale {
    type Command = f32;
    type ControlError = <ValidScale as TryFrom<f32>>::Error;
    type Input = Frame;
    type Output = Option<Frame>;
    type ProcessResult = Result<(), ScaleProcError>;

    fn control(&mut self, cmd: Self::Command) -> Result<&mut Self, Self::ControlError> {
        let factor = cmd.try_into()?;
        self.factor = factor;
        // todo: change resizer to bilinear for some factors?
        Ok(self)
    }

    fn is_dirty(&self) -> bool {
        // todo: but what if scale changed?
        false
    }

    fn advance(&mut self, input: &Self::Input, out: &mut Self::Output) -> Self::ProcessResult {
        if self.is_unit_scale() {
            // todo: can we at all avoid a clone?
            *out = Some(Frame { id: input.id, img: input.img.clone() });
            return Ok(());
        }

        let nwidth = (input.img.width() as f32 * self.factor.0) as _;
        let nheight = (input.img.height() as f32 * self.factor.0) as _;

        // todo some conversion trait
        // get input view
        let img_view = fr::ImageView::from_buffer(
            NonZeroU32::new(input.img.width()).unwrap(),
            NonZeroU32::new(input.img.height()).unwrap(),
            input.img.as_raw(),
            fr::PixelType::U8x3,
        )?;

        // todo some conversion trait
        // get or create new frame
        let frame = if let Some(ref mut frame) = out {
            if frame.img.width() != nwidth || frame.img.height() != nheight {
                frame.img = BgrImage::new(nwidth, nheight);
            }
            frame.id = input.id;
            frame
        } else {
            out.get_or_insert_with(|| Frame { id: input.id, img: BgrImage::new(nwidth, nheight) })
        };

        // get output view
        let mut img_view_mut = fr::ImageViewMut::from_buffer(
            NonZeroU32::new(nwidth).ok_or(ScaleProcError::ZeroSize)?,
            NonZeroU32::new(nheight).ok_or(ScaleProcError::ZeroSize)?,
            frame.img.as_mut(),
            fr::PixelType::U8x3,
        )?;

        self.resizer.resize(&img_view, &mut img_view_mut)?;
        Ok(())
    }
}

/// Application processing error
#[derive(Error, Debug)]
pub(crate) enum AppProcError {
    #[error(transparent)]
    Video(#[from] VideoProcError),
    #[error(transparent)]
    Scale(#[from] ScaleProcError),
}

/// Application command processing error
#[derive(Error, Debug)]
pub(crate) enum AppCmdError {
    #[error(transparent)]
    Scale(#[from] ValidScaleError),
    #[error(transparent)]
    Video(#[from] FFVideoError),
}

/// Control entire application
#[derive(Clone, Debug)]
pub(crate) enum AppCmd {
    /// Control video input
    Video(VideoCmd),
    /// Control scale factor
    Scale(f32),
    /// Exit App
    Exit,
}

/// Example app
#[derive(Default)]
pub(crate) struct ProcessingApp {
    vid: VideoPlayer,
    scale: Scale,
    frame: Option<Frame>,
    scaled_frame: Option<Frame>,

    pub(crate) to_exit: bool,
}

impl Processor for ProcessingApp {
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
        };
        Ok(self)
    }

    fn advance(&mut self, input: &(), _out: &mut ()) -> Self::ProcessResult {
        self.vid.advance(input, &mut self.frame)?;
        if let Some(frame) = &self.frame {
            self.scale.advance(frame, &mut self.scaled_frame)?;
        }
        // todo: trait and/or processor
        if let Some(scaled_frame) = &self.scaled_frame {
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
            Ok(Some(GUIFrame { id: scaled_frame.id, buffer: col_img }))
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
}
