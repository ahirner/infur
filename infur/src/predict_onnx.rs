use std::marker::PhantomData;

use image_ext::BgrImage;
use once_cell::sync::Lazy;
use onnxruntime::{
    environment::Environment,
    ndarray,
    ndarray::{arr1, Array4, ArrayD, ArrayView4, Axis, IxDyn},
    session::{Input, Session},
    tensor::OrtOwnedTensor,
    GraphOptimizationLevel, LoggingLevel, OrtError, TensorElementDataType,
    TypeToTensorElementDataType,
};
use thiserror::Error;

use crate::app::Processor;

// ONNX global environment to provide 'static to any session
static ENVIRONMENT: Lazy<Environment> = Lazy::new(|| {
    #[cfg(debug_assertions)]
    const LOGGING_LEVEL: LoggingLevel = LoggingLevel::Verbose;
    #[cfg(not(debug_assertions))]
    const LOGGING_LEVEL: LoggingLevel = LoggingLevel::Warning;

    Environment::builder()
        .with_name(env!("CARGO_PKG_NAME"))
        .with_log_level(LOGGING_LEVEL)
        .build()
        .unwrap()
});

/// Error processing model
#[derive(Error, Debug)]
pub(crate) enum ModelProcError {
    #[error("couldn't transform image")]
    ShapeError(#[from] ndarray::ShapeError),
    #[error("scaling to 0-sized output")]
    RuntimeError(#[from] OrtError),
}

/// Error loading model
#[derive(Error, Debug)]
pub(crate) enum ModelCmdError {
    #[error(transparent)]
    OrtError(#[from] OrtError),
    #[error(transparent)]
    RuntimeError(#[from] ModelInputFormatError),
}

#[derive(Error, Debug)]
pub(crate) enum ModelInputFormatError {
    #[error("couldn't infer image input")]
    Infer(String),
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct ModelInfo {
    input_names: Vec<String>,
    input0_dtype: String,
    output_names: Vec<String>,
}

#[derive(Debug)]
struct ImageSession<'s> {
    session: Session<'s>,
    img_proc: ImgPreProc,
    model_info: ModelInfo,
}

/// ONNX session with pre-processing u8 images.
impl<'s> ImageSession<'s> {
    /// Constract an `ImageSession` by inferring some required image input meta data.
    ///
    /// The basic assumption is that images are passed as batches at position 0.
    ///
    /// #Arguments
    ///
    /// * `session` - ONNX session with desired runtime behavior
    /// * `color_seq` - Order of color channels in the color dimension
    /// * `norm_float` - Whether `f32` input should not be scaled from 0-1 but around another mean and std deviation
    fn try_from_session(
        session: Session<'s>,
        color_seq: ColorSeq,
        norm_float: Option<ColorNorm<f32>>,
    ) -> Result<Self, ModelInputFormatError> {
        let (dim_seq, color_range) = infer_img_pre_proc(&session.inputs[0], norm_float)?;
        let img_proc = ImgPreProc { dim_seq, color_seq, color_range };
        let input_names = session.inputs.iter().map(|i| i.name.clone()).collect();
        let input0_dtype = format!("{:?}", session.inputs[0].input_type);
        let output_names = session.outputs.iter().map(|o| o.name.clone()).collect();
        let model_info = ModelInfo { input_names, input0_dtype, output_names };
        Ok(Self { session, img_proc, model_info })
    }

    /// Forward pass an NHWC(BGR) image batch
    fn forward<T: Clone + std::fmt::Debug + onnxruntime::TypeToTensorElementDataType>(
        &mut self,
        mut img_tensor: ArrayView4<'_, u8>,
    ) -> Result<Vec<OrtOwnedTensor<T, IxDyn>>, ModelProcError> {
        let pre = &self.img_proc;

        match pre.color_seq {
            ColorSeq::BGR => {}
            ColorSeq::RGB => {
                img_tensor.invert_axis(Axis(3));
            }
        };
        let (col_axis, img_tensor) = match pre.dim_seq {
            DimSeq::NHWC => (Axis(3), img_tensor),
            DimSeq::NCHW => (Axis(1), img_tensor.permuted_axes([0, 3, 1, 2])),
        };

        // todo: can at least keep an input vec around
        let model_tensors: Vec<OrtOwnedTensor<T, _>> = match &pre.color_range {
            ColorRange::Uint8 => {
                // todo: why .run() doesn't accept a view?
                // if it actually requires contiguity, to_owned may not provide that
                // and we may get BGR flipped if RGB is required (negative stride ignored) or a segfault..
                let owned_img = img_tensor.to_owned();
                self.session.run(vec![owned_img])?
            }
            ColorRange::Float32(norm) => {
                // instead of mapv, we have to recollect to ensure c contiguity
                // given potential prior permutations, otherwise onnxruntime segfaults
                let mut img_tensor_float = Array4::from_shape_vec(
                    img_tensor.raw_dim(),
                    img_tensor.iter().cloned().map(|v| f32::from(v) * 1f32 / 255f32).collect(),
                )?;
                if let Some(norm) = norm {
                    let mean = arr1(&norm.mean);
                    let std1 = 1.0f32 / arr1(&norm.std);
                    for mut lane in img_tensor_float.lanes_mut(col_axis) {
                        lane -= &mean;
                        lane *= &std1;
                    }
                };
                self.session.run(vec![img_tensor_float])?
            }
        };
        Ok(model_tensors)
    }
}

/// ONNX model session
pub(crate) struct Model<'s, T = f32> {
    img_session: Option<ImageSession<'s>>,
    _marker: PhantomData<T>,
}

impl<T> Default for Model<'_, T> {
    fn default() -> Self {
        Self { img_session: None, _marker: PhantomData }
    }
}

/// Order of color channels of a model's image input
#[derive(Debug, Clone)]
#[allow(clippy::upper_case_acronyms)]
pub(crate) enum ColorSeq {
    RGB,
    BGR,
}

/// Relative to a 0-1 range, subtract each target channel by mean and divide by std
#[derive(Debug, Clone)]
pub(crate) struct ColorNorm<T: Copy> {
    mean: [T; 3],
    std: [T; 3],
}

// todo: numtraits
impl<T: From<f32> + Copy> ColorNorm<T> {
    /// Default of torchvision's imagenet and many other pre-trained models
    fn new_torchvision_rgb() -> Self {
        Self {
            mean: [0.485.into(), 0.456.into(), 0.406.into()],
            std: [0.229.into(), 0.224.into(), 0.225.into()],
        }
    }
    /// Return a version with color channels flipped (last becomes first)
    fn flip(&self) -> Self {
        Self {
            mean: [self.mean[2], self.mean[1], self.mean[0]],
            std: [self.std[2], self.std[1], self.std[0]],
        }
    }
}

/// DType and nominal color range of a model's image input
#[derive(Debug, Clone)]
pub(crate) enum ColorRange {
    // byte normalized to u8::MIN - u8::MAX
    Uint8,
    // f32 normalized to 0-1 if None or by ColorNorm
    Float32(Option<ColorNorm<f32>>),
}

/// Order of semantic dimensions of a model's image input
#[derive(Debug, Clone)]
#[allow(clippy::upper_case_acronyms)]
pub(crate) enum DimSeq {
    /// Batch + TorchVision convention e.g.
    NHWC,
    /// Batch + OpenCV convention e.g.
    NCHW,
}

/// Specification of a model's image input
#[derive(Debug, Clone)]
pub(crate) struct ImgPreProc {
    dim_seq: DimSeq,
    color_seq: ColorSeq,
    color_range: ColorRange,
}

/// Determine partially model's image input requirements heuristically
///
/// # Arguments
///
/// * `input` - The session model's description of an image input tensor
/// * `norm_float` - Normalization to apply if and only if input is Float32 (ignored otherwise)
fn infer_img_pre_proc(
    input: &Input,
    norm_float: Option<ColorNorm<f32>>,
) -> Result<(DimSeq, ColorRange), ModelInputFormatError> {
    // find first dim with length 3
    let col_dim =
        input.dimensions.iter().position(|d| d.as_ref() == Some(&3)).ok_or_else(|| {
            ModelInputFormatError::Infer(
                "couldn't locate model's color input by dimension length 3".to_string(),
            )
        })?;

    if input.dimensions.len() != 4 {
        return Err(ModelInputFormatError::Infer(format!(
            "only 4 dimensions supported got {}",
            input.dimensions.len()
        )));
    };

    let dim_seq = match col_dim {
        1 => DimSeq::NCHW,
        3 => DimSeq::NHWC,
        p => {
            return Err(ModelInputFormatError::Infer(format!(
                "color dimension only at NCHW or NHWC but not in position {} supported",
                p
            )));
        }
    };

    let color_range = match &input.input_type {
        TensorElementDataType::Float => ColorRange::Float32(norm_float),
        TensorElementDataType::Uint8 => ColorRange::Uint8,
        dtype => {
            return Err(ModelInputFormatError::Infer(format!(
                "only Float (f32) and Uint8 (u8) input supported, got {:?}",
                dtype
            )));
        }
    };

    Ok((dim_seq, color_range))
}

#[derive(Clone, Debug)]
pub(crate) enum ModelCmd {
    Load(String),
}

impl<'s, 'session, T: TypeToTensorElementDataType + std::fmt::Debug + Clone> Processor
    for Model<'session, T>
where
    's: 'session,
{
    type Command = ModelCmd;
    type ControlError = ModelCmdError;
    type Input = BgrImage;
    type Output = Vec<ArrayD<T>>;
    type ProcessResult = Result<(), ModelProcError>;

    fn control(&mut self, cmd: Self::Command) -> Result<&mut Self, Self::ControlError> {
        match cmd {
            // todo: could use a more advanced fork to control intra vs. inter threads
            // e.g.: https://github.com/VOICEVOX/onnruntime-rs
            // discussion to migrate to official org:  https://github.com/nbigaouette/onnxruntime-rs/issues/112
            ModelCmd::Load(string_path) if !string_path.is_empty() => {
                let session = ENVIRONMENT
                    .new_session_builder()?
                    .with_optimization_level(GraphOptimizationLevel::Extended)?
                    .with_number_threads(1)?
                    .with_model_from_file(string_path)?;

                // todo: control col_seq properly instead of hardcoding our conventions
                let col_seq =
                    if matches!(session.inputs[0].input_type, TensorElementDataType::Float) {
                        ColorSeq::RGB
                    } else {
                        ColorSeq::BGR
                    };
                // todo: control color_norm properly instead of hardcoding our conventions
                let norm_float = match col_seq {
                    ColorSeq::RGB => ColorNorm::new_torchvision_rgb(),
                    ColorSeq::BGR => ColorNorm::new_torchvision_rgb().flip(),
                };
                self.img_session =
                    Some(ImageSession::try_from_session(session, col_seq, Some(norm_float))?);
            }
            ModelCmd::Load(_) => {
                self.img_session = None;
            }
        }
        Ok(self)
    }

    fn advance(&mut self, img: &Self::Input, out: &mut Self::Output) -> Self::ProcessResult {
        if let Some(ref mut session) = self.img_session {
            let img_shape = [1, img.height() as _, img.width() as _, 3];
            let img_tensor = ArrayView4::from_shape(img_shape, img)?;

            // todo: to return a Deref ArrayViewD with &session from &self, we'd need
            // maybe some Rc<Session> or GATs: https://github.com/rust-lang/rust/pull/96709
            // set cloned output without batch dim
            let model_tensors = session.forward(img_tensor)?;
            out.clear();
            // strip batch dim and clone
            for t in model_tensors {
                out.push(t.index_axis(ndarray::Axis(0), 0).into_owned());
            }
        }

        Ok(())
    }

    fn is_dirty(&self) -> bool {
        false
    }
}

impl<T> Model<'_, T> {
    pub(crate) fn get_info(&self) -> Option<&ModelInfo> {
        self.img_session.as_ref().map(|s| &s.model_info)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use infur_test_gen::fcn_resnet50_12_int8_onnx;

    fn fcn_seg_int8() -> String {
        fcn_resnet50_12_int8_onnx().to_string_lossy().to_string()
    }

    #[test]
    fn load_seg_model() {
        let mut m = Model::<f32>::default();
        m.control(ModelCmd::Load(fcn_seg_int8())).unwrap();
        let session = m.img_session.unwrap();
        eprintln!("model {} session {:?}", fcn_seg_int8(), session);
        for (i, inp) in session.session.inputs.iter().enumerate() {
            eprintln!("input {}: {:?} {} {:?}", i, inp.dimensions, inp.name, inp.input_type);
        }
        for (i, out) in session.session.outputs.iter().enumerate() {
            eprintln!("output {}: {:?} {} {:?}", i, out.dimensions, out.name, out.output_type);
        }
    }

    #[test]
    fn infer_seg_model() {
        let mut m = Model::<f32>::default();
        m.control(ModelCmd::Load(fcn_seg_int8())).unwrap();
        let img = BgrImage::new(320, 240);
        let mut tensors = vec![];
        m.advance(&img, &mut tensors).unwrap();

        assert_eq!(tensors.len(), 2, "this sementation model should return two tensors");
        assert_eq!(tensors[0].shape(), [21, 240, 320], "out should be 21 classes upscaled");
        assert_eq!(tensors[1].shape(), [21, 240, 320], "aux should be 21 classes upscaled");
    }
}
