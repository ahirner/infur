use std::{error::Error as StdError, fmt::Display, num::NonZeroU32, ops::Deref};

use fast_image_resize as fr;
use ff_video::{FFMpegDecoder, FFMpegDecoderBuilder, FFVideoError, VideoProcError, VideoResult};
use image_ext::BgrImage;
use thiserror::Error;

/// Frame produced and processed
pub(crate) struct Frame {
    pub(crate) id: u64,
    pub(crate) img: BgrImage,
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
    /// Pause generating new frames
    Pause(bool),
    /// Stop whenever
    Stop,
}

/// Writes video frames at command
#[derive(Default)]
pub(crate) struct VideoPlayer {
    vid: Option<FFMpegDecoder>,
    input: Vec<String>,
    paused: bool,
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
            Self::Command::Pause(paused) => {
                self.paused = paused;
            }
            Self::Command::Stop => {
                self.close_video()?;
            }
        }
        Ok(self)
    }

    fn is_dirty(&self) -> bool {
        !self.paused && self.vid.is_some()
    }

    fn advance(&mut self, _inp: &(), out: &mut Self::Output) -> Self::ProcessResult {
        if self.paused {
            return Ok(());
        }
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
pub(crate) struct ValidScale(f32);

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
pub(crate) struct Scale {
    factor: ValidScale,
    resizer: fr::Resizer,
    dirty: bool,
}

impl Default for Scale {
    fn default() -> Self {
        Self {
            factor: ValidScale(1.0f32),
            resizer: fr::Resizer::new(fr::ResizeAlg::Nearest),
            dirty: true,
        }
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
    #[error("scaling from 0-sized input")]
    ZeroSizeIn,
    #[error("scaling to 0-sized output")]
    ZeroSizeOut,
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
        self.dirty = factor != self.factor;
        self.factor = factor;
        // todo: change resizer to bilinear for some factors?
        Ok(self)
    }

    fn is_dirty(&self) -> bool {
        self.dirty
    }

    fn advance(&mut self, input: &Self::Input, out: &mut Self::Output) -> Self::ProcessResult {
        if self.is_unit_scale() {
            // todo: can we at all avoid a clone?
            *out = Some(Frame { id: input.id, img: input.img.clone() });
            self.dirty = false;
            return Ok(());
        }

        // todo: some conversion trait
        // get input view
        let img_view = fr::ImageView::from_buffer(
            NonZeroU32::new(input.img.width()).ok_or(ScaleProcError::ZeroSizeIn)?,
            NonZeroU32::new(input.img.height()).ok_or(ScaleProcError::ZeroSizeIn)?,
            input.img.as_raw(),
            fr::PixelType::U8x3,
        )?;

        let nwidth = (input.img.width() as f32 * self.factor.0) as _;
        let nheight = (input.img.height() as f32 * self.factor.0) as _;
        let nwidth0 = NonZeroU32::new(nwidth).ok_or(ScaleProcError::ZeroSizeOut)?;
        let nheight0 = NonZeroU32::new(nheight).ok_or(ScaleProcError::ZeroSizeOut)?;

        // todo: some conversion trait
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
            nwidth0,
            nheight0,
            frame.img.as_mut(),
            fr::PixelType::U8x3,
        )?;

        self.resizer.resize(&img_view, &mut img_view_mut)?;
        self.dirty = false;
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn scale_from_size0() {
        let zero = Frame { id: 0, img: BgrImage::new(0, 10) };
        let mut out = None;
        let mut scale = Scale::default();
        scale.control(0.99).unwrap();
        assert!(matches!(scale.advance(&zero, &mut out), Err(ScaleProcError::ZeroSizeIn)));
    }
    #[test]
    fn scale_to_size0() {
        let img = Frame { id: 0, img: BgrImage::new(10, 10) };
        let mut out = None;
        let mut scale = Scale::default();
        scale.control(0.00000001).unwrap();
        assert!(matches!(scale.advance(&img, &mut out), Err(ScaleProcError::ZeroSizeOut)));
    }
}
