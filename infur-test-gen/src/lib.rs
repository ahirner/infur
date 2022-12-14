use std::{env, path::PathBuf};

fn media_root() -> PathBuf {
    let gen_root = PathBuf::from(&env::var("CARGO_MANIFEST_DIR").unwrap());
    gen_root.parent().unwrap().join("media")
}

pub fn long_small_video() -> PathBuf {
    media_root().join("synth_640x480_40secs_10fps.mp4")
}

pub fn short_large_video() -> PathBuf {
    media_root().join("synth_1280x720_5secs_30fps.mp4")
}

pub fn fcn_resnet50_12_int8_onnx() -> PathBuf {
    let gen_root = PathBuf::from(&env::var("CARGO_MANIFEST_DIR").unwrap());
    gen_root.parent().unwrap().join("models").join("fcn-resnet50-12-int8.onnx")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn long_small_video_exists() {
        assert!(long_small_video().is_file())
    }

    #[test]
    fn short_large_exists() {
        assert!(short_large_video().is_file())
    }

    #[test]
    fn fcn_resnet50_12_int8_onnx_exists() {
        assert!(fcn_resnet50_12_int8_onnx().is_file())
    }
}
