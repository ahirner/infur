use std::{
    error::Error as StdError,
    fmt::Display,
    ops::Deref,
    sync::mpsc::{RecvError, SendError, TryRecvError},
};

use eframe::epaint::ColorImage;
use ff_video::{FFMpegDecoder, FFMpegDecoderBuilder, FFVideoError, VideoProcError, VideoResult};
use image_ext::{imageops::FilterType, BgrImage, Pixel};
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

/// Streaming Iterator whose item processing can be controlled
pub(crate) trait Processor<ProcessResult = ()> {
    /// Type of commands to control the processor
    type Command;

    /// Error to expect when controlling the processor
    type ControlError;

    /// The expected input to advance
    type Input;

    /// Item mutated on advance
    ///
    /// For why we can't impl Iterator see:
    /// <http://lukaskalbertodt.github.io/2018/08/03/solving-the-generalized-streaming-iterator-problem-without-gats.html>
    type Item;

    /// Affect processor parameters
    fn control(&mut self, cmd: Self::Command) -> Result<&mut Self, Self::ControlError>;

    /// Process and store a new result
    fn advance(&mut self, inp: &Self::Input, out: &mut Self::Item) -> ProcessResult;

    /// True if passing the same input into advance gives a new results
    fn is_dirty(&self) -> bool;
}

/// Commands that control VideoPlayer
#[derive(Debug, Clone)]
pub(crate) enum VideoCmd {
    /// Start or restart playing video from this ffmpeg input
    Play(Vec<String>),
    /// Stop whenever
    Stop,
}

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

impl Processor<VideoResult<()>> for VideoPlayer {
    type Command = VideoCmd;
    type ControlError = FFVideoError;
    type Input = ();
    type Item = Option<Frame>;

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

    fn advance(&mut self, _inp: &(), out: &mut Self::Item) -> VideoResult<()> {
        if let Some(vid) = self.vid.as_mut() {
            //todo: what if size changed?
            let frame = out.get_or_insert_with(|| Frame { id: 0, img: vid.empty_image() });
            let id = vid.read_frame(&mut frame.img)?;
            frame.id = id;
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

struct Scale {
    factor: ValidScale,
}

impl Default for Scale {
    fn default() -> Self {
        Self { factor: ValidScale(1.0f32) }
    }
}

impl Scale {
    fn is_unit_scale(&self) -> bool {
        self.factor.0 == 1.0f32
    }
}

impl Processor for Scale {
    type Command = f32;
    type ControlError = <ValidScale as TryFrom<f32>>::Error;
    type Input = Frame;
    type Item = Option<Frame>;

    fn control(&mut self, cmd: Self::Command) -> Result<&mut Self, Self::ControlError> {
        let factor = cmd.try_into()?;
        self.factor = factor;
        Ok(self)
    }

    fn is_dirty(&self) -> bool {
        // todo: but what if scale changed?
        false
    }

    fn advance(&mut self, frame: &Self::Input, out: &mut Self::Item) {
        if self.is_unit_scale() {
            let _ = out.as_ref().insert(frame);
            return;
        }
        let nwidth = (frame.img.width() as f32 * self.factor.0) as _;
        let nheight = (frame.img.height() as f32 * self.factor.0) as _;
        let img = image_ext::imageops::resize(&frame.img, nwidth, nheight, FilterType::Nearest);
        *out = Some(Frame { id: frame.id, img });
    }
}

/// Processing Error
#[derive(Error, Debug)]
pub(crate) enum ProcError {
    #[error(transparent)]
    Video(#[from] FFVideoError),
    #[error("channel communication failed")]
    Communication(#[from] CommunicationError),
}

#[derive(Error, Debug)]
pub(crate) enum CommunicationError {
    #[error(transparent)]
    Recv(#[from] RecvError),
    #[error(transparent)]
    TryRecv(#[from] TryRecvError),
    #[error(transparent)]
    Send(#[from] SendError<VideoResult<GUIFrame>>),
}

/// Commands transmitted to processing backend
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

#[derive(Error, Debug)]
pub(crate) enum AppCmdError {
    #[error(transparent)]
    Scale(#[from] ValidScaleError),
    #[error(transparent)]
    Video(#[from] FFVideoError),
}

impl Processor<VideoResult<Option<GUIFrame>>> for ProcessingApp {
    type Command = AppCmd;
    type ControlError = AppCmdError;
    type Input = ();
    type Item = (); //Result<GUIFrame, Box<dyn StdError>>;

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

    fn advance(&mut self, input: &(), _out: &mut ()) -> VideoResult<Option<GUIFrame>> {
        self.vid.advance(input, &mut self.frame)?;
        if let Some(frame) = &self.frame {
            self.scale.advance(frame, &mut self.scaled_frame);
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
                size: [scaled_frame.img.width() as usize, scaled_frame.img.width() as usize],
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
#[derive(Error, Debug)]
pub(crate) enum ProcessingError {
    #[error("error processing command")]
    Command(#[from] AppCmdError),
    #[error("error processing video feed")]
    Video(#[from] VideoProcError),
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn void() {
        let mut app = ProcessingApp::default();
        assert!(app.advance(&(), &mut ()).unwrap().is_none());
        assert!(app.advance(&(), &mut ()).unwrap().is_none());
    }

    #[test]
    fn scale() {
        let mut app = ProcessingApp::default();
        // todo: own fixtures
        app.control(AppCmd::Video(VideoCmd::Play(vec!["http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/WeAreGoingOnBullrun.mp4".to_string()]))).unwrap();
        app.control(AppCmd::Scale(0.5)).unwrap();
        let f2 = app.advance(&(), &mut ()).unwrap().expect("video should already play");
        assert_eq!(f2.buffer.size, [1280 / 2, 720 / 2]);
    }

    #[test]
    fn switch_scale() {
        let mut app = ProcessingApp::default();
        // todo: own fixtures
        app.control(AppCmd::Video(VideoCmd::Play(vec!["http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/WeAreGoingOnBullrun.mp4".to_string()]))).unwrap();

        let f1 = app.advance(&(), &mut ()).unwrap().expect("video should already play");
        assert_eq!(f1.buffer.size, [1280, 720]);

        app.control(AppCmd::Scale(0.5)).unwrap();
        let f2 = app.advance(&(), &mut ()).unwrap().expect("video should keep playing");
        assert_eq!(f2.buffer.size, [1280 / 2, 720 / 2]);
    }
}
