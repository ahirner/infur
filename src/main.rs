use anyhow::{anyhow, Context, Result};
use std::io::{ErrorKind, Read, Write};
use std::process::Command;
use std::thread::JoinHandle;
use std::time::Duration;
use std::{ffi::OsStr, thread};

mod parse;
use bus::Bus;
use parse::{FrameUpdate, InfoParser, Stream, StreamInfo, VideoInfo};
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

#[derive(Clone)]
struct Frame {
    id: u64,
    width: u32,
    height: u32,
    buffer: Box<[u8]>,
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
                    // emit lines on either \n or \r (CR), since
                    // ffmpeg terminates frame= lines only by \r without the -progress flag.
                    // alternatives considered:
                    // 1) -progress makes fields appear across lines, thus makes parsing them stateful
                    // 2) somehow switch terminator from std-lib lines() to CR after stream header,
                    //    but seems finicky
                    // 3) byte-based splits could probably be done more efficiently on BufReader
                    .bytes()
                    .scan(vec![0u8; 0], |state, b| match b {
                        Ok(b) if b == b'\n' || b == b'\r' => {
                            let line = String::from_utf8_lossy(state).into_owned();
                            state.clear();
                            Some(line)
                        }
                        Ok(b) => {
                            state.push(b);
                            Some(String::new())
                        }
                        Err(ref e) if e.kind() == ErrorKind::Interrupted => Some(String::new()),
                        Err(e) => {
                            event!(Level::ERROR, "couldn't read stderr {:?}", e);
                            None
                        }
                    })
                    .filter(|l| !l.is_empty());
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
                            _ = stream_info_tx
                                .send(Err(e).with_context(|| "video parsing"))
                                .map_err(|e| {
                                    event!(Level::WARN, "could not send parsing error: {:?}", e)
                                });
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
        // we may send graceful quit message and when dropped, buffer is flushed
        // afterwards, closing stdin on wait() won't break output pipe since ffmpeg
        // has already hung up
        stdin.write_all(b"q")?;
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

    fn read_frame(&mut self) -> Result<Frame> {
        let (width, height) = (self.video_output.width, self.video_output.height);
        let buflen = (width as usize) * (height as usize) * 3;
        let mut buffer = vec![0u8; buflen].into_boxed_slice();
        self.stdout.read_exact(buffer.as_mut())?;
        self.frame_counter += 1;
        Ok(Frame {
            id: self.frame_counter,
            width: self.video_output.width,
            height: self.video_output.height,
            buffer,
        })
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
    let mut vid = FFMpegDecoder::try_new(builder)?;
    for i in 0..1000 {
        let frame = vid.read_frame();
        if i % 100 == 0 {
            println!("{:?}", frame.map(|f| (f.id, f.width, f.height, f.buffer[500])));
        }
    }
    vid.close()?;

    Ok(())
}
