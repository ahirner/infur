use std::{
    io::{ErrorKind, Read, Write},
    process::Command,
    sync::mpsc::{Receiver, RecvTimeoutError},
    thread::{self, JoinHandle},
    time::Duration,
};

use image_ext::{BgrImage, ImageBuffer};
use tracing::{debug, error, info, warn};

use crate::{error::VideoResult, parse::FFMpegLineIter};
use crate::{
    error::{InfoResult, VideoProcError},
    parse::{InfoParser, Stream, StreamInfo, VideoInfo},
};

pub struct FFMpegDecoderBuilder {
    cmd: Command,
}

pub struct FFMpegDecoder {
    child: std::process::Child,
    stdout: std::process::ChildStdout,
    info_thread: JoinHandle<()>,
    pub frame_counter: u64,
    pub video_output: Stream,
}

impl Default for FFMpegDecoderBuilder {
    fn default() -> Self {
        let mut cmd = Command::new("ffmpeg");
        // options
        cmd.arg("-hide_banner");
        // escape input
        cmd.arg("-i");
        Self { cmd }
    }
}

impl FFMpegDecoderBuilder {
    pub fn input<I, S>(mut self, input: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<std::ffi::OsStr>,
    {
        self.cmd.args(input);
        self
    }

    fn cmd(mut self) -> Command {
        // output
        self.cmd.args([
            "-an",
            "-f",
            "image2pipe",
            "-fflags",
            "nobuffer",
            "-pix_fmt",
            "bgr24",
            "-c:v",
            "rawvideo",
            "pipe:1",
        ]);
        // piping
        use std::process::Stdio;
        self.cmd.stderr(Stdio::piped()).stdout(Stdio::piped()).stdin(Stdio::piped());
        // todo: rm
        eprintln!(
            "ffmpeg {}",
            self.cmd.get_args().map(|s| s.to_string_lossy()).collect::<Vec<_>>().join(" ")
        );
        self.cmd
    }
}

impl FFMpegDecoder {
    pub fn try_new(builder: FFMpegDecoderBuilder) -> VideoResult<Self> {
        let mut cmd = builder.cmd();
        let mut child = cmd
            .spawn()
            .map_err(|e| VideoProcError::explain_io("couldn't spawn video process", e))?;
        let stderr =
            child.stderr.take().ok_or_else(|| VideoProcError::is_missing("stderr pipe"))?;
        let (stream_info_rx, info_thread) = spawn_info_thread(stderr)?;

        // determine output
        let video_output = loop {
            let msg = stream_info_rx.recv_timeout(Duration::from_secs(10)).map_err(|e| {
                let why = match e {
                    RecvTimeoutError::Timeout => "timeout",
                    RecvTimeoutError::Disconnected => "disconnected",
                };
                VideoProcError::Start(why.to_string())
            })?;
            if let Ok(StreamInfo::Output { stream, .. }) = msg {
                break stream;
            };
        };

        let stdout =
            child.stdout.take().ok_or_else(|| VideoProcError::is_missing("stdout pipe"))?;
        Ok(Self { child, stdout, info_thread, video_output, frame_counter: 0 })
    }

    /// stop process gracefully and await exit code
    pub fn close(mut self) -> VideoResult<()> {
        let mut stdin =
            self.child.stdin.take().ok_or_else(|| VideoProcError::is_missing("stdin pipe"))?;
        // to close, we first send quit message (stdin is available since we encode nothing from it)
        // any stdin buffer is flushed in the drop() of wait()
        // thus it should break output pipe since ffmpeg has already hung up
        match stdin.write_all(b"q") {
            Ok(_) => {}
            Err(e) if e.kind() == ErrorKind::BrokenPipe => {} // process probably already exited
            Err(e) => {
                return Err(VideoProcError::explain_io("couldn't send q to process", e));
            }
        };
        //  ... unless we don't drain stdout as well, which we do here
        self.stdout.bytes().for_each(|_| {});

        let exit_code = self
            .child
            .wait()
            .map_err(|e| VideoProcError::explain_io("waiting on video process", e))?;
        self.info_thread
            .join()
            .map_err(|_| VideoProcError::Other("error joining meta data thread".to_string()))?;
        match exit_code.code() {
            Some(c) if c > 0 => Err(VideoProcError::ExitCode(c)),
            None => Err(VideoProcError::Other("video child process killed by signal".to_string())),
            _ => Ok(()),
        }
    }

    pub fn empty_image(&self) -> BgrImage {
        let (width, height) = (self.video_output.width, self.video_output.height);
        ImageBuffer::new(width, height)
    }

    /// Write new image and return its frame id.
    pub fn read_frame(&mut self, image: &mut BgrImage) -> VideoResult<u64> {
        self.stdout.read_exact(image.as_mut()).map_err(|e| {
            VideoProcError::explain_io("error reading full frame from video process", e)
        })?;
        self.frame_counter += 1;
        Ok(self.frame_counter)
    }
}

/// Deliver infos about an ffmpeg video process trhough its stderr file
///
/// The receiver can be read until satisfying info was obtained and dropped anytime.
/// By default, frame updates and other infos are logged as tracing event.
/// todo: offer a custom callback
fn spawn_info_thread<R>(
    stderr: R,
) -> VideoResult<(Receiver<InfoResult<StreamInfo>>, JoinHandle<()>)>
where
    R: Read + Send + 'static,
{
    let (stream_info_tx, stream_info_rx) =
        std::sync::mpsc::sync_channel::<InfoResult<StreamInfo>>(2);

    let info_thread = thread::Builder::new()
        .name("Video".to_string())
        .spawn(move || {
            let reader = std::io::BufReader::new(stderr);
            let mut ffmpeg_lines = reader.bytes().ffmpeg_lines();
            let lines = ffmpeg_lines.by_ref().filter_map(|r| match r {
                Err(e) => {
                    error!("couldn't read stderr {:?}", e);
                    None
                }
                Ok(r) => Some(r),
            });
            //.inspect(|l| println!("!!{}", l));

            // Delivery semantics depend on the message type:
            // - Stream and Error: must be delivered until the recv hangs up (thus isn't interested anymore)
            // - Any info: logged
            for msg in InfoParser::default().iter_on(lines) {
                match msg {
                    Ok(VideoInfo::Stream(msg)) => {
                        _ = stream_info_tx
                            .send(Ok(msg.clone()))
                            .map_err(|e| warn!("could not send stream info: {:?}", e));
                        log_info_handler(Ok(VideoInfo::Stream(msg)));
                    }
                    Ok(msg) => {
                        log_info_handler(Ok(msg));
                    }
                    Err(e) => {
                        _ = stream_info_tx.send(Err(e.clone()));
                        log_info_handler(Err(e));
                    }
                };
            }
            let last_line = String::from_utf8_lossy(ffmpeg_lines.state());
            info!("finished reading stderr: {}", last_line);
        })
        .map_err(|e| VideoProcError::explain_io("couldn't spawn info parsing thread", e))?;
    Ok((stream_info_rx, info_thread))
}

fn log_info_handler(msg: crate::parse::Result) {
    match msg {
        Ok(msg) => match msg {
            VideoInfo::Stream(msg) => {
                info!("new stream info: {:?}", msg);
            }
            VideoInfo::Frame(msg) => {
                debug!("frame: {:?}", msg);
            }
            VideoInfo::Codec(msg) => {
                info!("codec: {}", msg);
            }
        },
        Err(e) => {
            error!("parsing video update: {:?}", e);
        }
    }
}
