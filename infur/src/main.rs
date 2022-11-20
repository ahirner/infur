mod app;
mod decode_predict;
mod gui;
mod predict_onnx;
mod processing;

use std::sync::mpsc::{Receiver, SyncSender, TryRecvError};

use app::{AppCmd, ProcessingApp, Processor};
use gui::{CtrlResult, FrameResult};
use stable_eyre::eyre::{eyre, Report};
use tracing::{debug, warn};
use tracing_subscriber::{fmt, EnvFilter};

/// Result with user facing error
type Result<T> = std::result::Result<T, Report>;

fn init_logs() -> Result<()> {
    stable_eyre::install()?;
    let format = fmt::format().with_thread_names(true).with_target(false).compact();
    let filter = EnvFilter::try_from_default_env().or_else(|_| EnvFilter::try_new("info")).unwrap();
    tracing_subscriber::fmt::fmt().with_env_filter(filter).event_format(format).init();
    Ok(())
}

/// Channel events from and processing results to GUI
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

    // instantiate app in processing thread,
    // since ort session can't be moved/sent
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
                // block on GUI backpressure
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
    let window_opts = eframe::NativeOptions { vsync: true, ..Default::default() };
    let ctrl_tx_gui = ctrl_tx;
    eframe::run_native(
        "InFur",
        window_opts,
        Box::new(|cc| {
            let config = match cc.storage {
                Some(storage) => eframe::get_value(storage, eframe::APP_KEY).unwrap_or_default(),
                None => todo!(),
            };
            let mut app_gui = gui::InFur::new(config, ctrl_tx_gui, frame_rx, ctrl_result_rx);
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
