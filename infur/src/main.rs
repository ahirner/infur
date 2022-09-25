mod app;
mod predict_onnx;
mod processing;

use std::{
    collections::VecDeque,
    result::Result as StdResult,
    sync::mpsc::{Receiver, RecvTimeoutError, Sender, SyncSender, TryRecvError},
    time::{Duration, Instant},
};

use app::{AppCmd, AppCmdError, AppInfo, AppProcError, GUIFrame, ProcessingApp, Processor};
use eframe::{
    egui::{self, CentralPanel, RichText, SidePanel, Slider, TextureFilter, TextureHandle},
    epaint::FontId,
    NativeOptions,
};
use predict_onnx::ModelCmd;
use stable_eyre::eyre::{eyre, Report};
use tracing::{debug, warn};
use tracing_subscriber::{fmt, EnvFilter};

use crate::processing::VideoCmd;

/// Result with user facing error
type Result<T> = std::result::Result<T, Report>;

/// Result from processing a frame
type FrameResult = StdResult<GUIFrame, AppProcError>;

/// Result from processing commands
type CtrlResult = StdResult<AppInfo, AppCmdError>;

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

#[derive(serde::Deserialize, serde::Serialize)]
struct ProcConfig {
    video_input: Vec<String>,
    scale: f32,
    paused: bool,
    model_input: String,
    min_conf: f32,
}

impl Default for ProcConfig {
    fn default() -> Self {
        Self {
            video_input: vec![],
            min_conf: 0.5,
            scale: 0.5,
            paused: false,
            model_input: String::default(),
        }
    }
}

#[derive(Default, Clone)]
struct ProcStatus {
    video: String,
    scale: String,
    model: String,
}

struct InFur {
    ctrl_tx: Sender<AppCmd>,
    frame_rx: Receiver<FrameResult>,
    proc_result: Option<AppProcError>,
    ctrl_rx: Receiver<CtrlResult>,
    main_texture: Option<TextureFrame>,
    config: ProcConfig,
    closing: bool,
    allow_closing: bool,
    error_history: VecDeque<String>,
    counter: FrameCounter,
    show_count: u64,
    proc_status: ProcStatus,
}

impl InFur {
    fn new(
        config: ProcConfig,
        ctrl_tx: Sender<AppCmd>,
        frame_rx: Receiver<FrameResult>,
        ctrl_rx: Receiver<CtrlResult>,
    ) -> Self {
        let mut app = Self {
            ctrl_tx,
            frame_rx,
            ctrl_rx,
            proc_result: None,
            main_texture: None,
            config,
            closing: false,
            allow_closing: false,
            error_history: VecDeque::with_capacity(3),
            counter: FrameCounter::default(),
            show_count: 0,
            proc_status: ProcStatus::default(),
        };
        // send initial config
        app.send(AppCmd::Scale(app.config.scale));
        app.send(AppCmd::Video(VideoCmd::Play(
            app.config.video_input.iter().cloned().filter(|s| !s.is_empty()).collect(),
        )));
        app.send(AppCmd::Video(VideoCmd::Pause(app.config.paused)));
        app.send(AppCmd::Model(ModelCmd::Load(app.config.model_input.clone())));
        app
    }

    fn send(&mut self, cmd: AppCmd) {
        self.error_history.truncate(2);
        _ = self.ctrl_tx.send(cmd).map_err(|e| self.error_history.push_front(e.to_string()));
    }
}

impl eframe::App for InFur {
    fn update(&mut self, ctx: &eframe::egui::Context, frame_: &mut eframe::Frame) {
        // update texture from new frame or close if disconnected
        // this limits UI updates if no frames are sent to ca. 30fps
        match self.frame_rx.recv_timeout(Duration::from_millis(30)) {
            Ok(frame) => {
                let result = frame.map(|frame| TextureFrame {
                    id: frame.id,
                    handle: ctx.load_texture("frame", frame.buffer, TextureFilter::Linear),
                });
                match result {
                    Ok(tex) => {
                        self.main_texture = Some(tex);
                        self.proc_result = None;
                    }
                    Err(e) => {
                        self.proc_result = Some(e);
                    }
                }
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => {
                self.allow_closing = true;
                frame_.close();
            }
        }
        if self.closing && !self.allow_closing {
            self.send(AppCmd::Video(VideoCmd::Stop));
            self.send(AppCmd::Exit);
        }

        // advance and reset counters every second
        self.show_count += 1;
        let now = std::time::Instant::now();
        if self.counter.elapsed(now) > Duration::from_secs(1) {
            self.counter.set_on(now, self.show_count, self.main_texture.as_ref().map(|t| t.id));
        }

        // stringify last frame's statuses
        self.proc_status.video = match (&self.main_texture, &self.proc_result) {
            (_, Some(AppProcError::Video(e))) => e.to_string(),
            (Some(tex), _) => tex.id.to_string(),
            _ => "..waiting".to_string(),
        };
        match &self.proc_result {
            Some(AppProcError::Scale(e)) => self.proc_status.scale = e.to_string(),
            None => self.proc_status.scale = String::default(),
            _ => {}
        }
        match &self.proc_result {
            Some(AppProcError::Model(e)) => self.proc_status.model = e.to_string(),
            None if self.config.model_input.is_empty() => {
                self.proc_status.model = String::default()
            }
            _ => {}
        }

        // stringify control errors or app infos, may override frame status
        match self.ctrl_rx.try_recv() {
            Ok(info) => match info {
                Ok(info) => {
                    if let Some(model_info) = info.model_info {
                        self.proc_status.model = format!(
                            "Model loaded: {} -> {}",
                            model_info.input_names.join(","),
                            model_info.output_names.join(",")
                        );
                    }
                }
                Err(AppCmdError::Video(e)) => {
                    self.proc_status.video = e.to_string();
                }
                Err(AppCmdError::Scale(e)) => {
                    self.proc_status.scale = e.to_string();
                }
                Err(AppCmdError::Model(e)) => {
                    self.proc_status.model = e.to_string();
                }
            },
            Err(TryRecvError::Disconnected) => {
                self.error_history.push_front("lost processing control".to_string());
                frame_.close();
            }
            Err(_) => {}
        }

        SidePanel::left("Options").show(ctx, |ui| {
            ui.spacing_mut().item_spacing.y = 10.0;
            // video input
            ui.label(RichText::new("Video").font(FontId::proportional(30.0)));
            // (un-)pause video
            if ui.checkbox(&mut self.config.paused, "Pause").changed {
                self.send(AppCmd::Video(VideoCmd::Pause(self.config.paused)))
            };
            // (re-)play video
            if self.config.video_input.is_empty() {
                self.config.video_input.push(String::default());
            }
            let mut vid_input_changed = false;
            for inp in self.config.video_input.iter_mut() {
                let textbox = ui.text_edit_singleline(inp);
                vid_input_changed = vid_input_changed || textbox.lost_focus();
            }
            if vid_input_changed {
                self.send(AppCmd::Video(VideoCmd::Play(
                    self.config.video_input.iter().cloned().filter(|s| !s.is_empty()).collect(),
                )));
            }
            ui.label(&self.proc_status.video);

            ui.label(RichText::new("Detection").font(FontId::proportional(30.0)));
            let scale = Slider::new(&mut self.config.scale, 0.1f32..=1.0)
                .step_by(0.01f64)
                .text("scale")
                .clamp_to_range(true);
            let scale_response = ui.add(scale);
            if scale_response.changed {
                self.send(AppCmd::Scale(self.config.scale));
            };
            if !self.proc_status.scale.is_empty() {
                ui.label(&self.proc_status.model);
            }

            // (re-)load model
            let model_input = ui.text_edit_singleline(&mut self.config.model_input);
            if model_input.lost_focus() {
                self.send(AppCmd::Model(ModelCmd::Load(self.config.model_input.clone())));
            }
            ui.label(&self.proc_status.model);

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

            // rather fatal errors or final messages
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

    fn on_close_event(&mut self) -> bool {
        self.error_history.push_front("exiting...".to_string());
        // send exit once
        if !self.closing {
            // we could not close the video to exit faster, but
            // would end with an ffmpeg error
            self.send(AppCmd::Video(VideoCmd::Stop));
            self.send(AppCmd::Exit);
        }
        self.closing = true;
        self.allow_closing
    }

    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, &self.config);
    }
}

fn proc_loop(
    ctrl_rx: Receiver<AppCmd>,
    frame_tx: SyncSender<FrameResult>,
    app_tx: SyncSender<CtrlResult>,
) -> Result<()> {
    fn send_app_info(app: &ProcessingApp, app_tx: &SyncSender<CtrlResult>) {
        let app_info = app.info();
        debug!("sending updated app info {:?}", &app_info);
        let _ = app_tx.send(Ok(app_info));
    }

    let mut app = ProcessingApp::default();

    loop {
        // todo: exit on closed channel?
        let mut state_change = false;
        loop {
            let cmd = if !app.is_dirty() {
                // video is not playing, block
                debug!("blocking on new command");
                if state_change {
                    send_app_info(&app, &app_tx);
                    state_change = false;
                };
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
                    let _ = app_tx.send(Err(e));
                } else {
                    state_change = true;
                }
            };
            if app.to_exit {
                return Ok(());
            };
        }

        if state_change {
            send_app_info(&app, &app_tx);
        }

        match app.generate() {
            Ok(Some(frame)) => {
                // block for now, but need to think of dropping behavior
                let _ = frame_tx.send(Ok(frame));
            }
            Ok(None) => {
                warn!("Nothing to process yet")
            }

            Err(e) => {
                let _ = frame_tx.send(Err(e));
            }
        };
    }
}

fn main() -> Result<()> {
    init_logs()?;
    let args = std::env::args().skip(1).collect::<Vec<_>>();

    let (frame_tx, frame_rx) = std::sync::mpsc::sync_channel(2);
    let (ctrl_tx, ctrl_rx) = std::sync::mpsc::channel();
    let (ctrl_result_tx, ctrl_result_rx) = std::sync::mpsc::sync_channel(2);

    debug!("spawning Proc thread");
    let infur_thread = std::thread::Builder::new()
        .name("Proc".to_string())
        .spawn(move || proc_loop(ctrl_rx, frame_tx, ctrl_result_tx))?;

    debug!("starting InFur GUI");
    let window_opts = NativeOptions { vsync: true, ..Default::default() };
    let ctrl_tx_gui = ctrl_tx;
    eframe::run_native(
        "InFur",
        window_opts,
        Box::new(|cc| {
            let config = match cc.storage {
                Some(storage) => eframe::get_value(storage, eframe::APP_KEY).unwrap_or_default(),
                None => todo!(),
            };
            let mut app_gui = InFur::new(config, ctrl_tx_gui, frame_rx, ctrl_result_rx);
            // still override video from args
            if !args.is_empty() {
                app_gui.config.video_input = args;
            }
            Box::new(app_gui)
        }),
    );

    // ensure exit code
    infur_thread.join().unwrap().unwrap();
    Ok(())
}
