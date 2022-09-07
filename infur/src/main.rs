mod app;
mod processing;

use std::{
    collections::VecDeque,
    sync::mpsc::{Receiver, Sender, SyncSender, TryRecvError},
    time::{Duration, Instant},
};

use app::{AppCmd, AppCmdError, AppProcError, GUIFrame, ProcessingApp, Processor};
use eframe::{
    egui::{self, CentralPanel, RichText, SidePanel, Slider, TextureFilter, TextureHandle},
    epaint::FontId,
    NativeOptions,
};
use ff_video::FFMpegDecoder;
use image_ext::{imageops::FilterType, BgrImage};
use stable_eyre::eyre::{eyre, Report};
use thiserror::Error;
use tracing::{debug, error, warn};
use tracing_subscriber::{fmt, EnvFilter};

use crate::processing::VideoCmd;

/// Error processing commands or inputs
#[derive(Error, Debug)]
pub(crate) enum AppError {
    #[error("processing command: {0}")]
    Command(#[from] AppCmdError),
    #[error("processing video feed: {0}")]
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

struct FrameCounter {
    recvd_id: Option<u64>,
    shown_id: u64,
    since: Instant,
    elapsed_since: Duration,
    shown_since: u64,
    recvd_since: Option<u64>,
}

/// Counts how many frames were received and shown over one time strip
impl FrameCounter {
    // set start of new time strip (call per measurement)
    fn set_on(&mut self, now: Instant, shown_id: u64, recvd_id: Option<u64>) {
        // deltas
        self.shown_since = shown_id - self.shown_id;
        self.recvd_since = match (self.recvd_id, recvd_id) {
            (None, _) => None,
            (_, None) => None,
            (Some(r0), Some(r1)) => Some(r1 - r0),
        };
        self.elapsed_since = self.elapsed(now);
        // new 0
        self.recvd_id = recvd_id;
        self.shown_id = shown_id;
        self.since = now;
    }

    // time elapsed since last setting
    fn elapsed(&self, now: Instant) -> Duration {
        now - self.since
    }

    // fps of shown frames with respect to last time
    fn shown_fps(&self) -> f64 {
        self.shown_since as f64 / self.elapsed_since.as_secs_f64()
    }

    // fps of received frames with respect to last time
    fn recvd_fps(&self) -> f64 {
        match self.recvd_since {
            Some(r) => r as f64 / self.elapsed_since.as_secs_f64(),
            None => f64::NAN,
        }
    }

    // frames dropped (not shown) or skipped (also not shown)
    fn dropped_since(&self) -> i64 {
        self.recvd_since.unwrap_or_default() as i64 - self.shown_since as i64
    }
}

impl Default for FrameCounter {
    fn default() -> Self {
        Self {
            recvd_id: None,
            shown_id: 0,
            since: Instant::now(),
            elapsed_since: Duration::ZERO,
            shown_since: 0,
            recvd_since: None,
        }
    }
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
    proc_result: ProcessingResult<()>,
    main_texture: Option<TextureFrame>,
    config: ProcConfig,
    video_input: Vec<String>,
    error_history: VecDeque<String>,
    counter: FrameCounter,
    show_count: u64,
}

impl InFur {
    fn new(ctrl_tx: Sender<AppCmd>, frame_rx: Receiver<ProcessingResult<GUIFrame>>) -> Self {
        let config = ProcConfig::default();
        Self {
            ctrl_tx,
            frame_rx,
            proc_result: Ok(()),
            main_texture: None,
            error_history: VecDeque::with_capacity(3),
            config,
            video_input: vec![],
            counter: FrameCounter::default(),
            show_count: 0,
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
        // this limits UI updates if no frames are sent to ca. 30fps
        if let Ok(frame) = self.frame_rx.recv_timeout(Duration::from_millis(30)) {
            let result = frame.map(|frame| TextureFrame {
                id: frame.id,
                handle: ctx.load_texture("frame", frame.buffer, TextureFilter::Linear),
            });
            match result {
                Ok(tex) => {
                    self.main_texture = Some(tex);
                    self.proc_result = Ok(());
                }
                Err(e) => {
                    self.proc_result = Err(e);
                }
            }
        }

        // advance and reset counters
        self.show_count += 1;
        let now = std::time::Instant::now();
        if self.counter.elapsed(now) > Duration::from_secs(1) {
            self.counter.set_on(now, self.show_count, self.main_texture.as_ref().map(|t| t.id));
        }

        // stringify last frame's status
        let frame_status = match (&self.main_texture, &self.proc_result) {
            (_, Err(e)) => e.to_string(),
            (Some(tex), Ok(_)) => tex.id.to_string(),
            _ => "..waiting".to_string(),
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

            // frame stats
            let frame_stats = format!(
                "fps UI: {:>3.1}  processed: {:>3.1}  drops/skips: {}",
                self.counter.shown_fps(),
                self.counter.recvd_fps(),
                self.counter.dropped_since()
            );
            ui.label(frame_stats);

            // quite fatal errors
            ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
                for (i, err) in self.error_history.iter().cloned().enumerate() {
                    let col = egui::Color32::RED.linear_multiply(1.0 - (i as f32 / 4.0));
                    ui.colored_label(col, err);
                }
            });
        });

        // show last_texture
        // todo: maintain aspect ratio
        if let Some(tex_frame) = &self.main_texture {
            CentralPanel::default().show(ctx, |ui| {
                ui.image(&tex_frame.handle, ui.available_size());
            });
        };

        ctx.request_repaint();
    }
}

fn proc_loop(
    ctrl_rx: Receiver<AppCmd>,
    frame_tx: SyncSender<ProcessingResult<GUIFrame>>,
    mut app: ProcessingApp,
) -> Result<()> {
    loop {
        // todo: exit on closed channel?
        loop {
            let cmd = if !app.is_dirty() {
                // video is not playing, block
                debug!("blocking on new command");
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
                    let _ = frame_tx.send(Err(e.into()));
                }
            };
            if app.to_exit {
                return Ok(());
            };
        }

        match app.generate() {
            Ok(Some(frame)) => {
                // block for now, but need to think of dropping behavior
                let _ = frame_tx.send(Ok(frame));
            }
            // todo: handle better
            Ok(None) => {
                warn!("Didn't expect None result because we should have waited for dirty video")
            }

            Err(e) => {
                let _ = frame_tx.send(Err(e.into()));
            }
        };
    }
}

fn main() -> Result<()> {
    init_logs()?;
    let args = std::env::args().skip(1).collect::<Vec<_>>();

    let (frame_tx, frame_rx) = std::sync::mpsc::sync_channel(2);
    let (ctrl_tx, ctrl_rx) = std::sync::mpsc::channel();

    let mut app_gui = InFur::new(ctrl_tx.clone(), frame_rx);
    let app_proc = ProcessingApp::default();

    debug!("spawning Proc thread");
    let infur_thread = std::thread::Builder::new()
        .name("Proc".to_string())
        .spawn(move || proc_loop(ctrl_rx, frame_tx, app_proc))?;
    // send defaults and config from args
    {
        let config = ProcConfig::default();
        ctrl_tx.send(AppCmd::Scale(config.scale))?;
        // set video from args
        ctrl_tx.send(AppCmd::Video(VideoCmd::Play(args.clone())))?;
        app_gui.video_input = args;
    }

    let window_opts = NativeOptions { vsync: true, ..Default::default() };
    debug!("starting InFur GUI");
    eframe::run_native("InFur", window_opts, Box::new(|_| Box::new(app_gui)));

    // exit verbose on error
    ctrl_tx.send(AppCmd::Video(VideoCmd::Stop)).map_err(|e| error!("{:?}", e)).unwrap();
    ctrl_tx.send(AppCmd::Exit).map_err(|e| error!("{:?}", e)).unwrap();
    infur_thread.join().unwrap().unwrap();

    Ok(())
}
