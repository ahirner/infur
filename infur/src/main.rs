use anyhow::Result;
use ff_video::{FFMpegDecoder, FFMpegDecoderBuilder};
use image_ext::imageops::FilterType;

fn init_logs() {
    let format =
        tracing_subscriber::fmt::format().with_thread_names(true).with_target(false).compact();
    tracing_subscriber::fmt().event_format(format).init();
}

fn main() -> Result<()> {
    init_logs();
    let args = std::env::args().skip(1);
    let builder = FFMpegDecoderBuilder::default().input(args);
    let mut vid = FFMpegDecoder::try_new(builder)?;
    let mut img = vid.empty_image();

    use tract_onnx::prelude::*;
    // NHWC
    let img_shape = [1, (img.height() / 2) as _, (img.width() / 2) as _, 3];

    let img_shape_fact = ShapeFact::from_dims(img_shape);
    let model = tract_onnx::onnx()
        .model_for_path("models/mobilenet.onnx")?
        // aka image in NHWC(BGR<u8>)
        .with_input_fact(0, InferenceFact::dt_shape(u8::datum_type(), img_shape_fact.to_tvec()))?
        .into_optimized()?
        .into_runnable()?;

    let t0 = std::time::Instant::now();
    let (nwidth, nheight) = (img_shape[2] as _, img_shape[1] as _);
    let frames_max = 10;
    for _ in 0..frames_max {
        let _id = vid.read_frame(&mut img)?;
        let img_scaled = image_ext::imageops::resize(&img, nwidth, nheight, FilterType::Nearest);
        let ten_scaled =
            tract_ndarray::Array4::from_shape_vec(img_shape, img_scaled.to_vec())?.into();
        let result = model.run(tvec![ten_scaled])?;
        println!("result: {:?}", result);
    }

    let inf_time = (std::time::Instant::now() - t0).as_secs_f64() / (frames_max) as f64;
    vid.close()?;

    println!("video read+scale+model latency {nwidth}x{nheight}: {inf_time}");
    Ok(())
}
