use anyhow::{anyhow, Context, Result};
use std::ffi::OsStr;
use std::io::BufRead;

mod parse;

struct FFMpegDecode {
    child: std::process::Child,
}

impl FFMpegDecode {
    fn try_new<I, S>(input: I) -> Result<FFMpegDecode>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
    {
        let mut cmd = std::process::Command::new("ffmpeg");
        // options
        cmd.arg("-hide_banner");
        // input
        cmd.args(input);
        // output
        cmd.args([
            "-an",
            "-f",
            "image2pipe",
            "-fflags",
            "nobuffer",
            "-pix_fmt",
            "bgr24",
            "-c:v",
            "rawvideo",
            "-",
        ]);
        // piping
        cmd.stderr(std::process::Stdio::piped()).stdout(std::process::Stdio::piped());
        Ok(Self { child: cmd.spawn()? })
    }
    fn run(&mut self) -> Result<()> {
        let stderr = self.child.stderr.take().with_context(|| "Couldn't take stderr pipe")?;
        let _stdout = self.child.stdout.take().with_context(|| "Couldn't take stdout pipe")?;

        let info_thread = std::thread::spawn(move || {
            let reader = std::io::BufReader::new(stderr);
            reader.lines().for_each(|l| {
                eprintln!("{}", l.unwrap());
            });
        });

        info_thread.join().map_err(|_| anyhow!("Error joining meta data thread"))?;

        let exit_code = self.child.wait()?;
        match exit_code.code() {
            Some(c) if c > 0 => Err(anyhow!("Video child process exited with {}", exit_code)),
            None => Err(anyhow!("Video child process killed by signal.")),
            _ => Ok(()),
        }
    }
}

fn main() -> Result<()> {
    let args = std::env::args().skip(1);

    let mut vid = FFMpegDecode::try_new(args)?;
    vid.run()?;

    Ok(())
}
