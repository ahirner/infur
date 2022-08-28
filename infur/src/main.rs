use std::sync::{Arc, Mutex};

use eframe::{
    egui::{SidePanel, Slider},
    NativeOptions,
};
use stable_eyre::eyre::Report;

use ff_video::{FFMpegDecoder, FFMpegDecoderBuilder};
use image_ext::{imageops::FilterType, BgrImage};

type Result<T> = std::result::Result<T, Report>;

#[allow(dead_code)]
fn infer_onnx(
    img_shape: [usize; 4],
    vid: &mut FFMpegDecoder,
    mut img: BgrImage,
) -> Result<std::time::Instant> {
    use onnxruntime::{
        environment::Environment, ndarray, ndarray::Array4, tensor::OrtOwnedTensor,
        GraphOptimizationLevel, LoggingLevel,
    };

    let environment = Environment::builder()
        .with_name("test")
        // The ONNX Runtime's log level can be different than the one of the wrapper crate or the application.
        .with_log_level(LoggingLevel::Warning)
        .build()?;

    let mut session = environment
        .new_session_builder()?
        .with_optimization_level(GraphOptimizationLevel::Extended)?
        .with_number_threads(1)?
        .with_model_from_file("models/mobilenet.onnx")
        .unwrap();

    eprintln!("model session {:?}", session);
    for (i, input) in session.inputs.iter().enumerate() {
        eprintln!("input {}: {:?} {}", i, input.dimensions, input.name);
    }
    let output_names = session.outputs.iter().map(|o| o.name.clone()).collect::<Vec<_>>();

    let (nwidth, nheight) = (img_shape[2] as _, img_shape[1] as _);
    let t0 = std::time::Instant::now();
    for _ in 0..10 {
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

fn init_logs() -> Result<()> {
    stable_eyre::install()?;
    let format =
        tracing_subscriber::fmt::format().with_thread_names(true).with_target(false).compact();
    tracing_subscriber::fmt().event_format(format).init();
    Ok(())
}

struct InFur {}

impl eframe::App for InFur {
    fn update(&mut self, ctx: &eframe::egui::Context, _frame: &mut eframe::Frame) {
        SidePanel::left("Options").show(ctx, |sidebar| {
            sidebar.spacing_mut().item_spacing.y = 10.0;
            let mut value = 0.5;
            let slider = Slider::new(&mut value, 0f32..=1.0).step_by(0.01f64).text("min_conf");
            sidebar.add(slider);
            sidebar.label("Test");
        });
        ctx.request_repaint();
    }
}

fn main() -> Result<()> {
    init_logs()?;
    let args = std::env::args_os().skip(1);
    let builder = FFMpegDecoderBuilder::default().input(args);
    let vid = Arc::new(Mutex::new(Some(FFMpegDecoder::try_new(builder)?)));

    let thread_vid = vid.clone();
    let infur_thread = std::thread::spawn(move || {
        let mut img = thread_vid.lock().unwrap().as_mut().unwrap().empty_image();
        //let img_shape = [1, (img.height() / 2) as _, (img.width() / 2) as _, 3];
        //let (nwidth, nheight) = (img_shape[2], img_shape[1]);
        loop {
            let mut guard = thread_vid.lock().unwrap();
            if let Some(ref mut vid) = guard.as_mut().take() {
                let id = vid.read_frame(&mut img);
                match id {
                    Ok(_) => {}
                    Err(_) => break,
                }
            } else {
                return;
            };
        }
    });

    let app = InFur {};
    let window_opts = NativeOptions::default();
    eframe::run_native("InFur", window_opts, Box::new(|_| Box::new(app)));

    vid.lock().unwrap().take().unwrap().close()?;
    infur_thread.join().unwrap();
    Ok(())
}
