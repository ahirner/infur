use std::io::{ErrorKind, Read, Write};
use std::process::Command;
use std::thread::JoinHandle;
use std::time::Duration;
use std::{ffi::OsStr, thread};

mod image_bgr;
mod parse;

use anyhow::{anyhow, Context, Result};
use bus::Bus;
use image::ImageBuffer;
use image_bgr::BgrImage;
use parse::{FFMpegLineIter, FrameUpdate, InfoParser, Stream, StreamInfo, VideoInfo};
use tracing::{event, Level};

struct FFMpegDecoder {
    child: std::process::Child,
    stdout: std::process::ChildStdout,
    info_thread: JoinHandle<()>,
    pub frame_counter: u64,
    pub video_output: Stream,
    //frame_updates: Bus<FrameUpdate>,
}

struct FFMpegDecoderBuilder {
    cmd: Command,
}

impl FFMpegDecoderBuilder {
    fn new() -> Self {
        let mut cmd = Command::new("ffmpeg");
        // options
        cmd.arg("-hide_banner");
        // escape input
        cmd.arg("-i");
        Self { cmd }
    }

    fn input<I, S>(mut self, input: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
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
    fn try_new(builder: FFMpegDecoderBuilder) -> Result<Self> {
        let mut cmd = builder.cmd();
        let mut child = cmd.spawn()?;
        let stderr = child.stderr.take().with_context(|| "couldn't take stderr pipe")?;
        // The semenatics depending on the message type:
        // - stream info: must be delivered until the recv hangs up (and isn't interested anymore)
        // - parse errors: log but otherwise treat like stream info
        // - read errors: log and but fuilter
        // - frame updates: can be delivered and drop if not recvd
        // - codec info: log as info
        let (stream_info_tx, stream_info_rx) =
            std::sync::mpsc::sync_channel::<Result<StreamInfo>>(2);
        let mut frame_updates = Bus::<FrameUpdate>::new(5);
        //let frame_rx = frame_updates.add_rx();

        let info_thread = thread::Builder::new().name("Video".to_string()).spawn(move || {
            let reader = std::io::BufReader::new(stderr);
            let lines = reader.bytes().ffmpeg_lines().filter_map(|r| match r {
                Err(e) => {
                    event!(Level::ERROR, "couldn't read stderr {:?}", e);
                    None
                }
                Ok(r) => Some(r),
            });
            //.inspect(|x| println!("!! {}", x));
            for msg in InfoParser::default().iter_on(lines) {
                match msg {
                    Ok(msg) => match msg {
                        VideoInfo::Stream(msg) => {
                            _ = stream_info_tx.send(Ok(msg)).map_err(|e| {
                                event!(Level::WARN, "could not send stream info: {:?}", e)
                            });
                        }
                        VideoInfo::Frame(msg) => {
                            event!(Level::INFO, "{:?}", msg);
                            _ = frame_updates.try_broadcast(msg);
                        }
                        VideoInfo::Codec(msg) => {
                            event!(Level::INFO, "codec: {}", msg);
                        }
                    },
                    Err(e) => {
                        event!(Level::ERROR, "parsing video update: {:?}", e);
                        _ = stream_info_tx.send(Err(e).with_context(|| "video parsing")).map_err(
                            |e| event!(Level::WARN, "could not send parsing error: {:?}", e),
                        );
                    }
                }
            }
            event!(Level::INFO, "finished reading stderr");
        })?;

        // determine output
        let (video_output, to) = loop {
            let msg = stream_info_rx
                .recv_timeout(Duration::from_secs(10))
                .with_context(|| "couldn't parse and transmit stream info within deadline")?;
            if let Ok(StreamInfo::Output { to, stream, .. }) = msg {
                break (stream, to);
            };
        };

        event!(Level::INFO, "determined video output: {:?} to {}", video_output, to);

        let stdout = child.stdout.take().with_context(|| "couldn't take stdout pipe")?;
        Ok(Self { child, stdout, info_thread, video_output, frame_counter: 0 })
    }

    /// stop process gracefully and await exit code
    fn close(mut self) -> Result<()> {
        let mut stdin = self.child.stdin.take().with_context(|| "couldn't take stdin pipe")?;
        // to close, we first send quit message (stdin is available since we encode nothing from it)
        // any stdin buffer is flushed in the drop() of wait()
        // thus it should break output pipe since ffmpeg has already hung up
        match stdin.write_all(b"q") {
            Ok(_) => {}
            Err(e) if e.kind() == ErrorKind::BrokenPipe => {} // process probably already exited
            Err(e) => {
                Err(e).with_context(|| "Couldn't send q to process")?;
            }
        };
        //  ... unless we don't drain stdout as well, which we do here
        self.stdout.bytes().for_each(|_| {});

        let exit_code = self.child.wait()?;
        self.info_thread.join().map_err(|_| anyhow!("error joining meta data thread"))?;
        match exit_code.code() {
            Some(c) if c > 0 => Err(anyhow!("video child process exited with {}", exit_code)),
            None => Err(anyhow!("video child process killed by signal.")),
            _ => Ok(()),
        }
    }

    fn empty_image(&self) -> BgrImage {
        let (width, height) = (self.video_output.width, self.video_output.height);
        let img_rgb: image::RgbImage = ImageBuffer::new(width, height);
        BgrImage::new(img_rgb)
    }

    /// Write new image and return its frame id.
    fn read_frame(&mut self, image: &mut BgrImage) -> Result<u64> {
        self.stdout
            .read_exact(image.as_mut())
            .with_context(|| "error reading full frame from video process")?;
        self.frame_counter += 1;
        Ok(self.frame_counter)
    }
}

fn init_logs() {
    let format =
        tracing_subscriber::fmt::format().with_thread_names(true).with_target(false).compact();
    tracing_subscriber::fmt().event_format(format).init();
}

fn main() -> Result<()> {
    init_logs();
    let args = std::env::args().skip(1);
    let builder = FFMpegDecoderBuilder::new().input(args);
    let mut vid = FFMpegDecoder::try_new(builder)?;
    let mut img = vid.empty_image();
    for i in 0..1000 {
        match vid.read_frame(&mut img) {
            Ok(id) if (i % 100 == 0) => {
                event!(Level::INFO, "{}, {:?}", id, img.dimensions());
            }
            Ok(_) => {}
            Err(e) => {
                event!(Level::WARN, "{:?}", e);
                break;
            }
        };
    }
    vid.close()?;

    Ok(())
}
