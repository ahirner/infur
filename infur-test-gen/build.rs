use std::{
    env,
    ffi::OsStr,
    path::{Path, PathBuf},
    process::Command,
};

fn run_ffmpeg_synth(
    out_file: impl AsRef<OsStr>,
    width: usize,
    height: usize,
    rate: usize,
    duration: usize,
) {
    let mut cmd = Command::new("ffmpeg");
    cmd.args(["-f", "lavfi", "-i"])
        .arg(format!("testsrc=duration={duration}:size={width}x{height}:rate={rate}"))
        .args(["-pix_fmt", "yuv420p", "-y"])
        .arg(out_file);

    let status = cmd
        .spawn()
        .expect("synthesizing video couldn't start, do you have ffmpeg in PATH?")
        .wait()
        .expect("synthesizing video didn't finish");
    assert!(status.success(), "synthesizing videos didn't finish succesfully");
}

pub fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    //let out_dir = env::var_os("OUT_DIR").unwrap();
    let gen_root = PathBuf::from(&env::var("CARGO_MANIFEST_DIR").unwrap());
    let gen_root =
        Path::new(&gen_root).parent().expect("wanted parent of manifest for generating test files");

    // video files
    for (width, height, rate, dur) in [(1280, 720, 30, 5), (640, 480, 10, 40)] {
        let file = format!("synth_{width}x{height}_{dur}secs_{rate}fps.mp4");
        let dest_path = gen_root.join("media").join(file);

        run_ffmpeg_synth(&dest_path, width, height, rate, dur);
        eprintln!("{:?}", &dest_path);
        println!("cargo:rerun-if-changed={}", &dest_path.to_string_lossy());
    }
}
