use std::sync::mpsc::{Receiver, SyncSender, TryRecvError};

use eframe::{
    egui::{CentralPanel, SidePanel, Slider, TextureFilter, TextureHandle},
    epaint::ColorImage,
    NativeOptions,
};
use stable_eyre::eyre::{eyre, Report};

use ff_video::{FFMpegDecoder, FFMpegDecoderBuilder, FFVideoError, VideoResult};
use image_ext::{imageops::FilterType, BgrImage, Pixel};

type Result<T> = std::result::Result<T, Report>;
type VideoProcResult<T> = std::result::Result<T, FFVideoError>;

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

struct InFur {
    frame_rx: Receiver<VideoResult<Frame>>,
    last_texture: Option<VideoProcResult<TextureFrame>>,
}

impl InFur {
    fn new(frame_rx: Receiver<ff_video::VideoResult<Frame>>) -> Self {
        Self { frame_rx, last_texture: None }
    }
}

struct TextureFrame {
    id: u64,
    handle: TextureHandle,
}

impl eframe::App for InFur {
    fn update(&mut self, ctx: &eframe::egui::Context, _frame: &mut eframe::Frame) {
        // update text from new frame
        if let Ok(frame) = self.frame_rx.try_recv() {
            let result = match frame {
                Ok(frame) => Ok(TextureFrame {
                    id: frame.id,
                    handle: ctx.load_texture("frame", frame.buffer, TextureFilter::Linear),
                }),
                Err(e) => Err(FFVideoError::Processing(e)),
            };
            self.last_texture = Some(result);
        }

        // show last_texture
        if let Some(Ok(tex_frame)) = &self.last_texture {
            CentralPanel::default().show(ctx, |image_area| {
                image_area.image(&tex_frame.handle, image_area.available_size());
            });
        };

        // stringify last frame's status
        let frame_status = match &self.last_texture {
            Some(Ok(tex)) => tex.id.to_string(),
            Some(Err(e)) => e.to_string(),
            None => "..waiting".to_string(),
        };

        // some bogus placeholder for now
        SidePanel::left("Options").show(ctx, |sidebar| {
            sidebar.spacing_mut().item_spacing.y = 10.0;
            let mut value = 0.5;
            let slider = Slider::new(&mut value, 0f32..=1.0).step_by(0.01f64).text("min_conf");
            sidebar.add(slider);
            sidebar.label(frame_status);
        });

        ctx.request_repaint();
    }
}

/// Frame transmitted to GUI
struct Frame {
    id: u64,
    buffer: ColorImage,
}

/// Commands transmitted to processing backend
#[derive(Clone, Debug)]
enum ProcCtrl {
    /// Start or restart playing video from this ffmpeg input
    Play(Vec<String>),
    /// Stop whatever and return
    Exit,
    /// Set's the scale factor for pre-processing
    SetScale(f32),
}

/// Manages prcoessing resources and reacts to control messages
struct ProcessingApp {
    scale: f32,
    vid: Option<FFMpegDecoder>,
    img: Option<BgrImage>,
    id: u64,
    input: Vec<String>,
    to_exit: bool,
}

impl Default for ProcessingApp {
    fn default() -> Self {
        Self { scale: 1.0, vid: None, img: None, id: 0, input: vec![], to_exit: false }
    }
}

impl ProcessingApp {
    fn control(&mut self, cmd: ProcCtrl) -> VideoResult<()> {
        match cmd {
            ProcCtrl::Play(input) => {
                self.close_video()?;
                self.input = input;
                let builder = FFMpegDecoderBuilder::default().input(self.input.clone());
                self.vid = Some(FFMpegDecoder::try_new(builder)?);
            }
            ProcCtrl::Exit => {
                self.close_video()?;
                self.to_exit = true;
            }
            ProcCtrl::SetScale(s) => {
                self.scale = s;
            }
        };
        Ok(())
    }

    /// If available, read a new frame and return true
    fn next_frame(&mut self) -> VideoResult<bool> {
        self.vid
            .as_mut()
            .map(|vid| {
                let img = self.img.get_or_insert_with(|| vid.empty_image());
                self.id = vid.read_frame(img)?;
                Ok(true)
            })
            .unwrap_or(Ok(false))
    }

    fn is_video(&self) -> bool {
        self.vid.is_some()
    }
    fn close_video(&mut self) -> VideoResult<()> {
        self.vid.take().map_or(Ok(()), |vid| vid.close())
    }
}

fn proc_loop(
    cmds: Receiver<ProcCtrl>,
    frame_tx: SyncSender<ff_video::VideoResult<Frame>>,
) -> Result<()> {
    let mut app = ProcessingApp::default();

    loop {
        let cmd = if !app.is_video() {
            // video is not playing, block
            match cmds.recv() {
                Ok(c) => Some(c),
                Err(e) => break Err(eyre!(e)),
            }
        } else {
            // video is playing, don't block
            match cmds.try_recv() {
                Ok(c) => Some(c),
                Err(TryRecvError::Empty) => None,
                Err(e) => break Err(eyre!(e)),
            }
        };
        if let Some(cmd) = cmd {
            app.control(cmd)?
        };
        if app.to_exit {
            break Ok(());
        }
        let new_frame = match app.next_frame() {
            Ok(new) => new,
            Err(e) => {
                frame_tx.send(Err(e))?;
                false
            }
        };
        // current logic is to send only new frames,
        // later we could process also if model is available and params changed
        if new_frame {
            if let Some(ref img) = app.img {
                let id = app.id;
                let nwidth = (img.width() as f32 * app.scale) as _;
                let nheight = (img.height() as f32 * app.scale) as _;

                let img_temp;
                let img_scaled = if nwidth == img.width() && nheight == img.height() {
                    img
                } else {
                    img_temp =
                        image_ext::imageops::resize(img, nwidth, nheight, FilterType::Nearest);
                    &img_temp
                };
                // todo: own conversion trait or Color32 ImageBuffer
                let rgba_pixels = img_scaled
                    .pixels()
                    .map(|p| {
                        let cs = p.channels();
                        eframe::epaint::Color32::from_rgb(cs[2], cs[1], cs[0])
                    })
                    .collect::<Vec<_>>();
                let img_col =
                    ColorImage { size: [nwidth as usize, nheight as usize], pixels: rgba_pixels };
                // todo: broadcast or really drop here?
                let frame = Frame { id, buffer: img_col };
                let _ = frame_tx.try_send(Ok(frame));
            }
        }
    }
}

fn main() -> Result<()> {
    init_logs()?;
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let (frame_tx, frame_rx) = std::sync::mpsc::sync_channel(2);
    let (ctrl_tx, ctrl_rx) = std::sync::mpsc::channel();

    let infur_thread = std::thread::spawn(move || proc_loop(ctrl_rx, frame_tx));
    ctrl_tx.send(ProcCtrl::SetScale(0.5))?;
    ctrl_tx.send(ProcCtrl::Play(args))?;

    let app = InFur::new(frame_rx);
    let window_opts = NativeOptions::default();
    eframe::run_native("InFur", window_opts, Box::new(|_| Box::new(app)));

    ctrl_tx.send(ProcCtrl::Exit)?;
    // todo: something i didn't get from eyre..??
    infur_thread.join().map_err(|_| eyre!("video processing thread errored"))??;

    Ok(())
}
