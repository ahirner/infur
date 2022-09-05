mod processing;

use std::{
    collections::VecDeque,
    sync::mpsc::{Receiver, Sender, SyncSender, TryRecvError},
};

use eframe::{
    egui::{self, CentralPanel, RichText, SidePanel, Slider, TextureFilter, TextureHandle},
    epaint::FontId,
    NativeOptions,
};
use ff_video::FFMpegDecoder;
use image_ext::{imageops::FilterType, BgrImage};
use processing::{AppCmd, AppCmdError, AppProcError, GUIFrame, ProcessingApp, Processor};
use stable_eyre::eyre::{eyre, Report};
use thiserror::Error;
use tracing::{debug, error, warn};
use tracing_subscriber::{fmt, EnvFilter};

use crate::processing::VideoCmd;

/// Error processing commands or inputs
#[derive(Error, Debug)]
pub(crate) enum AppError {
    #[error("error processing command")]
    Command(#[from] AppCmdError),
    #[error("error processing video feed")]
    Processing(#[from] AppProcError),
}

type Result<T> = std::result::Result<T, Report>;
type ProcessingResult<T> = std::result::Result<T, AppError>;

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
    let format = fmt::format().with_thread_names(true).with_target(false).compact();
    let filter = EnvFilter::try_from_default_env().or_else(|_| EnvFilter::try_new("info")).unwrap();
    tracing_subscriber::fmt::fmt().with_env_filter(filter).event_format(format).init();
    Ok(())
}

struct TextureFrame {
    id: u64,
    handle: TextureHandle,
}

struct ProcConfig {
    min_conf: f32,
    scale: f32,
}

impl Default for ProcConfig {
    fn default() -> Self {
        Self { min_conf: 0.5, scale: 0.5 }
    }
}

struct InFur {
    ctrl_tx: Sender<AppCmd>,
    frame_rx: Receiver<ProcessingResult<GUIFrame>>,
    main_texture: Option<ProcessingResult<TextureFrame>>,
    config: ProcConfig,
    video_input: Vec<String>,
    error_history: VecDeque<String>,
}

impl InFur {
    fn new(ctrl_tx: Sender<AppCmd>, frame_rx: Receiver<ProcessingResult<GUIFrame>>) -> Self {
        let config = ProcConfig::default();
        Self {
            ctrl_tx,
            frame_rx,
            main_texture: None,
            error_history: VecDeque::with_capacity(3),
            config,
            video_input: vec![],
        }
    }

    fn send(&mut self, cmd: AppCmd) {
        self.error_history.truncate(2);
        _ = self.ctrl_tx.send(cmd).map_err(|e| self.error_history.push_front(e.to_string()));
    }
}

impl eframe::App for InFur {
    fn update(&mut self, ctx: &eframe::egui::Context, _frame: &mut eframe::Frame) {
        // update texture from new frame
        if let Ok(frame) = self.frame_rx.try_recv() {
            let result = frame.map(|frame| TextureFrame {
                id: frame.id,
                handle: ctx.load_texture("frame", frame.buffer, TextureFilter::Linear),
            });
            self.main_texture = Some(result);
        }

        // show last_texture
        // todo: maintain aspect ratio
        if let Some(Ok(tex_frame)) = &self.main_texture {
            CentralPanel::default().show(ctx, |ui| {
                ui.image(&tex_frame.handle, ui.available_size());
            });
        };

        // stringify last frame's status
        let frame_status = match &self.main_texture {
            Some(Ok(tex)) => tex.id.to_string(),
            Some(Err(e)) => e.to_string(),
            None => "..waiting".to_string(),
        };

        SidePanel::left("Options").show(ctx, |ui| {
            ui.spacing_mut().item_spacing.y = 10.0;
            // video input
            ui.label(RichText::new("Video").font(FontId::proportional(30.0)));
            ui.label(frame_status);

            let mut vid_input_changed = false;
            for inp in self.video_input.iter_mut() {
                let textbox = ui.text_edit_singleline(inp);
                vid_input_changed = vid_input_changed || textbox.lost_focus();
            }
            if vid_input_changed {
                dbg!(&self.video_input);
                self.send(AppCmd::Video(VideoCmd::Play(self.video_input.clone())));
            }

            ui.label(RichText::new("Detection").font(FontId::proportional(30.0)));
            let scale = Slider::new(&mut self.config.scale, 0.1f32..=1.0)
                .step_by(0.01f64)
                .text("scale")
                .clamp_to_range(true);
            let scale_response = ui.add(scale);
            if scale_response.changed {
                self.send(AppCmd::Scale(self.config.scale));
            };
            // todo: actual model
            let min_conf = Slider::new(&mut self.config.min_conf, 0f32..=1.0)
                .step_by(0.01f64)
                .text("min_conf")
                .clamp_to_range(true);
            ui.add(min_conf);

            // quite fatal errors
            ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
                for (i, err) in self.error_history.iter().cloned().enumerate() {
                    let col = egui::Color32::RED.linear_multiply(1.0 - (i as f32 / 4.0));
                    ui.colored_label(col, err);
                }
            });
        });

        ctx.request_repaint();
    }
}

fn proc_loop(
    ctrl_rx: Receiver<AppCmd>,
    frame_tx: SyncSender<ProcessingResult<GUIFrame>>,
) -> Result<()> {
    let mut app = ProcessingApp::default();
    loop {
        // todo: exit on closed channel?
        loop {
            let cmd = if !app.is_dirty() {
                // video is not playing, block
                match ctrl_rx.recv() {
                    Ok(c) => Some(c),
                    // unfixable (hung-up)
                    Err(e) => return Err(eyre!(e)),
                }
            } else {
                // video is playing, don't block
                match ctrl_rx.try_recv() {
                    Ok(c) => Some(c),
                    Err(TryRecvError::Empty) => break,
                    // unfixable (hung-up)
                    Err(e) => return Err(eyre!(e)),
                }
            };
            if let Some(cmd) = cmd {
                debug!("relaying command: {:?}", cmd);
                if let Err(e) = app.control(cmd) {
                    // Control Error
                    let _ = frame_tx.try_send(Err(e.into()));
                }
            };
            if app.to_exit {
                return Ok(());
            };
        }

        match app.advance(&(), &mut ()) {
            Ok(Some(frame)) => {
                // block for now, but need to think of dropping behavior
                let _ = frame_tx.send(Ok(frame));
            }
            // todo: handle better
            Ok(None) => {
                warn!("Didn't expect None result because we should have waited for dirty video")
            }
            Err(e) => {
                let _ = frame_tx.try_send(Err(e.into()));
            }
        };
    }
}

fn main() -> Result<()> {
    init_logs()?;
    let args = std::env::args().skip(1).collect::<Vec<_>>();

    let (frame_tx, frame_rx) = std::sync::mpsc::sync_channel(2);
    let (ctrl_tx, ctrl_rx) = std::sync::mpsc::channel();
    let mut app = InFur::new(ctrl_tx.clone(), frame_rx);

    debug!("spawning Proc thread");
    let infur_thread = std::thread::Builder::new()
        .name("Proc".to_string())
        .spawn(move || proc_loop(ctrl_rx, frame_tx))?;
    // send defaults and config from args
    {
        let config = ProcConfig::default();
        ctrl_tx.send(AppCmd::Scale(config.scale))?;
        // set video from args
        ctrl_tx.send(AppCmd::Video(VideoCmd::Play(args.clone())))?;
        app.video_input = args;
    }

    let window_opts = NativeOptions { vsync: false, ..Default::default() };
    debug!("starting InFur frontend");
    eframe::run_native("InFur", window_opts, Box::new(|_| Box::new(app)));

    // exit verbose on error
    ctrl_tx.send(AppCmd::Video(VideoCmd::Stop)).map_err(|e| error!("{:?}", e)).unwrap();
    ctrl_tx.send(AppCmd::Exit).map_err(|e| error!("{:?}", e)).unwrap();
    infur_thread.join().unwrap().unwrap();

    Ok(())
}
