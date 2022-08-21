use anyhow::{anyhow, Context, Result};
use std::io::{BufRead, Write};
use std::process::Command;
use std::thread::JoinHandle;
use std::time::Duration;
use std::{ffi::OsStr, thread};

mod parse;
use bus::Bus;
use parse::{FrameUpdate, InfoParser, StreamInfo, VideoInfo};
use tracing::{event, Level};

struct FFMpegDecoder {
    child: std::process::Child,
    info_thread: JoinHandle<()>,
    //video_output: Stream,
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
        self.cmd
    }
}

impl FFMpegDecoder {
    fn try_new(builder: FFMpegDecoderBuilder) -> Result<Self> {
        let mut cmd = builder.cmd();
        let mut child = cmd.spawn()?;
        let stderr = child.stderr.take().with_context(|| "couldn't take stderr pipe")?;
        let _stdout = child.stdout.take().with_context(|| "couldn't take stdout pipe")?;

        // we have three semenatics depending on the message type:
        // - stream info: must be delivered until the recv hangs up (and isn't interested anymore)
        // - frame updates: can be delivered and drop if not recvd
        // - codec info: log as info
        // - errors: log but otherwise treat like stream info
        let (stream_info_tx, stream_info_rx) =
            std::sync::mpsc::sync_channel::<Result<StreamInfo>>(2);
        let mut frame_updates = Bus::<FrameUpdate>::new(5);
        //let frame_rx = frame_updates.add_rx();

        let info_thread =
            thread::Builder::new().name("VideoInfo".to_string()).spawn(move || {
                let reader = std::io::BufReader::new(stderr);
                let lines = reader
                    .lines()
                    .filter_map(|line| {
                        if let Ok(line) = line {
                            Some(line)
                        } else {
                            event!(Level::ERROR, "couldn't read stderr {:?}", line);
                            None
                        }
                    })
                    .inspect(|m| println!("!! {}", m));
                for msg in InfoParser::default().iter_on(lines) {
                    match msg {
                        Ok(msg) => match msg {
                            VideoInfo::Stream(msg) => {
                                _ = stream_info_tx.send(Ok(msg)).map_err(|e| {
                                    event!(Level::WARN, "could not send stream info: {:?}", e)
                                });
                            }
                            VideoInfo::Frame(msg) => {
                                _ = frame_updates.try_broadcast(msg);
                            }
                            VideoInfo::Codec(msg) => {
                                event!(Level::INFO, "codec: {}", msg);
                            }
                        },
                        Err(e) => {
                            event!(Level::ERROR, "parsing video update: {:?}", e);
                            _ = stream_info_tx
                                .send(Err(e).with_context(|| "video parsing"))
                                .map_err(|e| {
                                    event!(Level::WARN, "could not send parsing error: {:?}", e)
                                });
                        }
                    }
                }
            })?;

        // determine output
        let (video_output, to) = loop {
            let msg = stream_info_rx
                .recv_timeout(Duration::from_millis(2000))
                .with_context(|| "couldn't parse and transimt stream info within deadline")?;
            if let Ok(StreamInfo::Output { to, stream, .. }) = msg {
                break (stream, to);
            };
        };

        event!(Level::INFO, "determined video output: {:?} to {}", video_output, to);

        Ok(Self { child, info_thread })
    }

    /// stop process gracefully and await exit code
    fn close(mut self) -> Result<()> {
        let stdin = self.child.stdin.take().with_context(|| "couldn't take stdin pipe")?;

        // we may send graceful quit message and when dropped, buffer is flushed
        // afterwards, closing stdin on wait() won't break output pipe since ffmpeg
        // has already hung up? // todo: ... sometimes
        std::io::BufWriter::new(stdin).write_all(b"q")?;
        let exit_code = self.child.wait()?;

        match exit_code.code() {
            Some(c) if c > 0 => Err(anyhow!("video child process exited with {}", exit_code)),
            None => Err(anyhow!("video child process killed by signal.")),
            _ => Ok(()),
        }?;

        self.info_thread.join().map_err(|_| anyhow!("error joining meta data thread"))
    }
}

fn init_logs() {
    let format = tracing_subscriber::fmt::format().with_thread_names(true).compact();
    tracing_subscriber::fmt().event_format(format).init();
}

fn main() -> Result<()> {
    init_logs();
    let args = std::env::args().skip(1);
    let builder = FFMpegDecoderBuilder::new().input(args);
    let vid = FFMpegDecoder::try_new(builder)?;
    vid.close()?;

    Ok(())
}
