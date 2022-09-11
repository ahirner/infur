use std::{
    env,
    ffi::OsStr,
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
    process::Command,
};

use filetime::{set_file_mtime, FileTime};

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

fn download(source_url: &str, target_file: impl AsRef<Path>) {
    // borrowed from onnxruntime
    let resp = ureq::get(source_url)
        .timeout(std::time::Duration::from_secs(300))
        .call()
        .unwrap_or_else(|err| panic!("ERROR: Failed to download {}: {:?}", source_url, err));

    let len = resp.header("Content-Length").and_then(|s| s.parse::<usize>().ok()).unwrap();
    let mut reader = resp.into_reader();
    // FIXME: Save directly to the file
    let mut buffer = vec![];
    let read_len = reader.read_to_end(&mut buffer).unwrap();
    assert_eq!(buffer.len(), len);
    assert_eq!(buffer.len(), read_len);

    let f = fs::File::create(&target_file).unwrap();
    let mut writer = io::BufWriter::new(f);
    writer.write_all(&buffer).unwrap();
}

/// Makes files look like they were there 60 seconds earlier.
///
/// We need that for generated files since cargo tests that
/// they are not stale iff. mtime of build artifacts > rerun-if-changed.
fn make_younger(file: impl AsRef<Path>) {
    let file_meta = fs::metadata(&file).unwrap();
    let mtime = FileTime::from_last_modification_time(&file_meta);
    let mtime_before = mtime.unix_seconds() - 60;
    set_file_mtime(&file, filetime::FileTime::from_unix_time(mtime_before, 0)).unwrap();
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
        make_younger(&dest_path);
        println!("cargo:rerun-if-changed={}", &dest_path.to_string_lossy());
    }

    // models
    // segementation model, see: https://github.com/onnx/models/tree/main/vision/object_detection_segmentation/fcn
    let fcn_resnet50_12_int8 = gen_root.join("models").join("fcn-resnet50-12-int8.onnx");
    download("https://github.com/onnx/models/raw/main/vision/object_detection_segmentation/fcn/model/fcn-resnet50-12-int8.onnx",
        &fcn_resnet50_12_int8);
    make_younger(&fcn_resnet50_12_int8);
    println!("cargo:rerun-if-changed={}", &fcn_resnet50_12_int8.to_string_lossy());
}
