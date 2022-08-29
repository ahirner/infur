use std::str::FromStr;

use anyhow::Result;
use ff_video::{FFMpegDecoder, FFMpegDecoderBuilder};
use image_ext::{imageops::FilterType, BgrImage};

#[derive(Debug)]
enum Engine {
    Tract,
    ONNXRt,
}

impl FromStr for Engine {
    type Err = pico_args::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "tract" => Ok(Self::Tract),
            "onnx" => Ok(Self::ONNXRt),
            _ => Err(pico_args::Error::Utf8ArgumentParsingFailed {
                value: s.to_string(),
                cause: "can be either 'onnx' (default) or 'tract'".to_string(),
            }),
        }
    }
}

#[derive(Debug)]
struct Options {
    engine: Engine,
    /// number of frames to run of the video
    frames_max: u16,
    /// only for onnx, best on intel macbook pro: 2 (debug), 4 (release) but 2 almost as fast and more efficient
    threads: u8,
}

fn infer_tract(
    img_shape: [usize; 4],
    frames_max: u16,
    vid: &mut FFMpegDecoder,
    mut img: BgrImage,
) -> Result<std::time::Instant, anyhow::Error> {
    use tract_onnx::prelude::*;
    let (nwidth, nheight) = (img_shape[2] as _, img_shape[1] as _);
    let img_shape_fact = ShapeFact::from_dims(img_shape);
    let model = tract_onnx::onnx()
        .model_for_path("../models/mobilenet.onnx")?
        // aka image in NHWC(BGR<u8>)
        .with_input_fact(0, InferenceFact::dt_shape(u8::datum_type(), img_shape_fact.to_tvec()))?
        .into_optimized()?
        .into_runnable()?;
    let t0 = std::time::Instant::now();
    for _ in 0..frames_max {
        let _id = vid.read_frame(&mut img)?;
        let img_scaled = image_ext::imageops::resize(&img, nwidth, nheight, FilterType::Nearest);
        let ten_scaled =
            tract_ndarray::Array4::from_shape_vec(img_shape, img_scaled.to_vec())?.into();
        let result = model.run(tvec![ten_scaled])?;
        println!("result: {:?}", result);
    }
    Ok(t0)
}

fn infer_onnx(
    img_shape: [usize; 4],
    opts: &Options,
    vid: &mut FFMpegDecoder,
    mut img: BgrImage,
) -> Result<std::time::Instant, anyhow::Error> {
    use onnxruntime::{
        environment::Environment, ndarray, ndarray::Array4, tensor::OrtOwnedTensor,
        GraphOptimizationLevel, LoggingLevel,
    };

    let environment = Environment::builder()
        .with_name("test")
        // The ONNX Runtime's log level can be different than the one of the wrapper crate or the application.
        .with_log_level(LoggingLevel::Verbose)
        .build()?;

    let mut session = environment
        .new_session_builder()?
        .with_optimization_level(GraphOptimizationLevel::Extended)?
        .with_number_threads(opts.threads.into())?
        .with_model_from_file("../models/mobilenet.onnx")
        .unwrap();

    eprintln!("model session {:?}", session);
    for (i, input) in session.inputs.iter().enumerate() {
        eprintln!("input {}: {:?} {}", i, input.dimensions, input.name);
    }
    let output_names = session.outputs.iter().map(|o| o.name.clone()).collect::<Vec<_>>();

    let (nwidth, nheight) = (img_shape[2] as _, img_shape[1] as _);
    let t0 = std::time::Instant::now();
    for _ in 0..opts.frames_max {
        let _id = vid.read_frame(&mut img)?;
        let img_scaled = image_ext::imageops::resize(&img, nwidth, nheight, FilterType::Nearest);
        let ten_scaled = Array4::from_shape_vec(img_shape, img_scaled.to_vec())?;
        let result: Vec<OrtOwnedTensor<f32, _>> = session.run(vec![ten_scaled])?;
        // remove all batch dims
        let result = result.iter().map(|t| t.index_axis(ndarray::Axis(0), 0)).collect::<Vec<_>>();

        println!(
            "result: {:?}",
            result.iter().zip(output_names.iter()).map(|(t, o)| (o, t.shape())).collect::<Vec<_>>()
        );
        // perform max across classes
        let hm_max = result[0].index_axis(ndarray::Axis(0), 0).fold_axis(
            ndarray::Axis(0),
            f32::MIN,
            |&a, &b| a.max(b),
        );
        println!("hm_max: {:?} max: {}", hm_max.shape(), hm_max.fold(f32::MIN, |a, &b| a.max(b)));
    }
    Ok(t0)
}

fn init_logs() {
    let format =
        tracing_subscriber::fmt::format().with_thread_names(true).with_target(false).compact();
    tracing_subscriber::fmt().event_format(format).init();
}

fn main() -> Result<()> {
    init_logs();
    let mut pargs = pico_args::Arguments::from_env();

    let options = Options {
        engine: pargs.opt_value_from_str("--engine")?.unwrap_or(Engine::ONNXRt),
        frames_max: pargs.opt_value_from_str("--frames-max")?.unwrap_or(10),
        threads: pargs.opt_value_from_str("--threads")?.unwrap_or(1),
    };

    eprintln!("options: {:?}", &options);
    if matches!(options.engine, Engine::Tract) && options.threads > 1 {
        anyhow::bail!("tract doesn't support internal multi-threading (https://github.com/sonos/tract/discussions/690)");
    }

    let args = pargs.finish();

    let builder = FFMpegDecoderBuilder::default().input(args);
    let mut vid = FFMpegDecoder::try_new(builder)?;
    let img = vid.empty_image();

    let img_shape = [1, (img.height() / 2) as _, (img.width() / 2) as _, 3];
    let (nwidth, nheight) = (img_shape[2], img_shape[1]);

    let t0 = match options.engine {
        Engine::Tract => infer_tract(img_shape, options.frames_max, &mut vid, img)?,
        Engine::ONNXRt => infer_onnx(img_shape, &options, &mut vid, img)?,
    };

    let inf_time = (std::time::Instant::now() - t0).as_secs_f64() / (options.frames_max) as f64;
    vid.close()?;

    println!("video read+scale+model latency {nwidth}x{nheight}: {inf_time}");
    Ok(())
}
