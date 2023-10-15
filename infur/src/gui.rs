use std::collections::VecDeque;
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, TryRecvError};
use std::time::{Duration, Instant};

use crate::app::{AppCmd, AppCmdError, AppInfo, AppProcError, GUIFrame};
use crate::predict_onnx::ModelCmd;
use crate::processing::VideoCmd;
use eframe::{
    egui::{self, CentralPanel, RichText, SidePanel, Slider, TextureHandle, TextureOptions},
    epaint::{FontId, Vec2},
};

/// Result from processing a frame
pub(crate) type FrameResult = std::result::Result<GUIFrame, AppProcError>;

/// Result from processing commands
pub(crate) type CtrlResult = std::result::Result<AppInfo, AppCmdError>;

/// GUI textures of model in-/output
pub(crate) struct TextureFrame {
    pub(crate) id: u64,
    pub(crate) handle: TextureHandle,
    pub(crate) decoded_handle: Option<TextureHandle>,
}

/// Count frames and time between set points
pub(crate) struct FrameCounter {
    pub(crate) recvd_id: Option<u64>,
    pub(crate) shown_id: u64,
    pub(crate) since: Instant,
    pub(crate) elapsed_since: Duration,
    pub(crate) shown_since: u64,
    pub(crate) recvd_since: Option<u64>,
}

impl FrameCounter {
    // set start of new time strip (call per measurement)
    pub(crate) fn set_on(&mut self, now: Instant, shown_id: u64, recvd_id: Option<u64>) {
        // deltas
        self.shown_since = shown_id - self.shown_id;
        self.recvd_since = match (self.recvd_id, recvd_id) {
            (None, _) => None,
            (_, None) => None,
            (Some(r0), Some(r1)) if r0 > r1 => None,
            (Some(r0), Some(r1)) => Some(r1 - r0),
        };
        self.elapsed_since = self.elapsed(now);
        // new 0
        self.recvd_id = recvd_id;
        self.shown_id = shown_id;
        self.since = now;
    }

    // time elapsed since last setting
    pub(crate) fn elapsed(&self, now: Instant) -> Duration {
        now - self.since
    }

    // fps of shown frames with respect to last time
    pub(crate) fn shown_fps(&self) -> f64 {
        self.shown_since as f64 / self.elapsed_since.as_secs_f64()
    }

    // fps of received frames with respect to last time
    pub(crate) fn recvd_fps(&self) -> f64 {
        match self.recvd_since {
            Some(r) => r as f64 / self.elapsed_since.as_secs_f64(),
            None => f64::NAN,
        }
    }

    // frames dropped (not shown) or skipped (also not shown)
    pub(crate) fn dropped_since(&self) -> i64 {
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
pub(crate) struct ProcConfig {
    pub(crate) video_input: Vec<String>,
    pub(crate) scale: f32,
    pub(crate) paused: bool,
    pub(crate) model_input: String,
}

impl Default for ProcConfig {
    fn default() -> Self {
        Self { video_input: vec![], scale: 0.5, paused: false, model_input: String::default() }
    }
}

#[derive(Default, Clone)]
pub(crate) struct ProcStatus {
    pub(crate) video: String,
    pub(crate) scale: String,
    pub(crate) model: String,
}

pub(crate) struct InFur {
    pub(crate) ctrl_tx: Sender<AppCmd>,
    pub(crate) frame_rx: Receiver<FrameResult>,
    pub(crate) proc_result: Option<AppProcError>,
    pub(crate) ctrl_rx: Receiver<CtrlResult>,
    pub(crate) main_texture: Option<TextureFrame>,
    pub(crate) config: ProcConfig,
    pub(crate) closing: bool,
    pub(crate) allow_closing: bool,
    pub(crate) error_history: VecDeque<String>,
    pub(crate) counter: FrameCounter,
    pub(crate) show_count: u64,
    pub(crate) proc_status: ProcStatus,
}

impl InFur {
    pub(crate) fn new(
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

    pub(crate) fn send(&mut self, cmd: AppCmd) {
        self.error_history.truncate(2);
        _ = self.ctrl_tx.send(cmd).map_err(|e| self.error_history.push_front(e.to_string()));
    }
}

impl eframe::App for InFur {
    fn update(&mut self, ctx: &eframe::egui::Context, frame_: &mut eframe::Frame) {
        // update texture from new frame or close if disconnected
        // this limits UI updates if no frames are sent to ca. 30fps
        let mut new_frame = false;
        match self.frame_rx.recv_timeout(Duration::from_millis(30)) {
            Ok(Ok(frame)) => {
                let decoded_handle = frame.decoded_buffer.map(|decoded_img| {
                    ctx.load_texture("decoded_texture", decoded_img, TextureOptions::NEAREST)
                });

                let tex = TextureFrame {
                    id: frame.id,
                    handle: ctx.load_texture(
                        "main_texture",
                        frame.buffer,
                        TextureOptions {
                            magnification: egui::TextureFilter::Nearest,
                            minification: egui::TextureFilter::Linear,
                        },
                    ),
                    decoded_handle,
                };
                new_frame = true;
                self.main_texture = Some(tex);
                self.proc_result = None;
            }
            Ok(Err(e)) => {
                self.proc_result = Some(e);
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
        match (new_frame, &self.main_texture, &self.proc_result) {
            (true, _, Some(AppProcError::Video(e))) => self.proc_status.video = e.to_string(),
            (true, Some(tex), _) => self.proc_status.video = tex.id.to_string(),
            _ => {}
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

            ui.label(RichText::new("Inference").font(FontId::proportional(30.0)));
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

            // frame stats
            ui.label(RichText::new("Stats").font(FontId::proportional(30.0)));
            let frame_stats = format!(
                "fps UI: {:>3.1}\nprocessed: {:>3.1}\ndrops/skips: {}",
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
        if let Some(tex_frame) = &self.main_texture {
            CentralPanel::default().show(ctx, |ui| {
                // occupy max width with constant aspect ratio
                let max_width = ui.available_width();
                let [w, h] = tex_frame.handle.size();
                let w_scale = max_width / w as f32;
                let (w, h) = (w as f32 * w_scale, h as f32 * w_scale);
                ui.add(egui::Image::new(&tex_frame.handle).fit_to_exact_size(Vec2::new(w, h)));
                // prop decoded image underneath
                // todo: blend somehow?
                if let Some(ref handle) = tex_frame.decoded_handle {
                    ui.add(egui::Image::new(handle).fit_to_exact_size(Vec2::new(w, h)));
                };
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

    #[cfg(feature = "persistence")]
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, &self.config);
    }
}
