#![allow(dead_code)]

/// Describes one output's video stream
#[derive(Debug, Clone, PartialEq)]
struct OutputStream {
    num: u32,
    to: String,
    width: u32,
    height: u32,
    fps: Option<f32>,
}

/// Describes one stream's update
#[derive(Debug, Clone, PartialEq)]
struct FrameUpdate {
    frame: u64,
}

//#[derive(Debug, Clone, PartialEq)]
//struct DecoderUpdate {}

#[derive(Debug, Clone, PartialEq)]
enum VideoInfoMessage {
    OutputMeta(OutputStream),
    Frame(FrameUpdate),
    //    Decoder(DecoderUpdate),
}

#[derive(Debug, Clone, PartialEq)]
enum ParseContext {
    Stateless,
    Output(u32, String),
}

#[derive(Debug, Clone)]
struct InfoParser {
    mode: ParseContext,
}

#[derive(Debug, Clone, PartialEq)]
struct ParseError {
    context: ParseContext,
    line: String,
    reason: String,
}

type InfoResult = Result<Option<VideoInfoMessage>, ParseError>;
type FilteredInfo = Result<VideoInfoMessage, ParseError>;

impl InfoParser {
    fn default() -> Self {
        InfoParser { mode: ParseContext::Stateless }
    }

    fn error_on(&self, reason: impl Into<String>, line: &str) -> ParseError {
        ParseError { context: self.mode.clone(), line: line.to_string(), reason: reason.into() }
    }

    fn push(&mut self, line: &str) -> InfoResult {
        dbg!(line);
        let error_on = |reason| self.error_on(reason, line);
        let error_on_ = |reason| self.error_on(reason, line); // no generic closures

        // Begin Output Stream
        let output = line.trim_start_matches("Output #");
        if output.len() < line.len() {
            if !matches!(self.mode, ParseContext::Stateless) {
                return Err(error_on("already parsing an Output"));
            }
            let mut parts = output.split(',');
            let out_num = parts
                .next()
                .ok_or_else(|| error_on("no delimiter after output number"))?
                .trim()
                .parse::<u32>()
                .map_err(|e| error_on_(format!("Output # not a number {:?}", e)))?;
            let _codec = parts.next().ok_or("nothing after output number");
            let to = parts.next().ok_or_else(|| error_on("nothing after codec"))?.trim();

            self.mode = ParseContext::Output(out_num, to.to_string());
            return Ok(None);
        }

        // Parse Output Video Stream infos
        let line_trimmed = line.trim();
        let stream = line_trimmed.trim_start_matches("Stream #");
        if stream.len() < line_trimmed.len() {
            let context = match self.mode {
                ParseContext::Output(out_num, ref to) => (out_num, to),
                _ => return Err(error_on("found Stream while not parsing an Output")),
            };
            let mut parts = stream.split(':');
            let out_num_stream = parts
                .next()
                .ok_or_else(|| error_on("no delimiter after stream number"))?
                .parse::<u32>()
                .map_err(|e| error_on_(format!("Stream # not a number {:?}", e)))?;

            if context.0 != out_num_stream {
                return Err(error_on_(format!("Stream {} didn't match Output", out_num_stream)));
            };

            let mut is_video = false;
            let mut width_height = None;
            let mut fps = None;
            for p in parts {
                if !is_video && p.trim() == "Video" {
                    is_video = true;
                }
                if is_video {
                    for key_vals in p.split(',') {
                        let key_vals = key_vals.trim();
                        let fps_vals = key_vals.trim_end_matches(" fps");
                        if fps_vals.len() < key_vals.len() {
                            fps = fps_vals
                                .parse::<f32>()
                                .map_err(|_| error_on("fps not a number"))?
                                .into();
                        } else {
                            let mut dim_vals = key_vals.splitn(2, 'x');
                            if let (Some(width_str), Some(height_str)) =
                                (dim_vals.next(), dim_vals.next())
                            {
                                let height_str =
                                    height_str.split_once(' ').map_or_else(|| "", |v| v.0);
                                if let (Ok(w), Ok(h)) =
                                    (width_str.parse::<u32>(), height_str.parse::<u32>())
                                {
                                    width_height = Some((w, h))
                                };
                            }
                        }
                    }
                }
            }
            return if let Some((width, height)) = width_height {
                Ok(Some(VideoInfoMessage::OutputMeta(OutputStream {
                    num: context.0,
                    to: context.1.clone(),
                    width,
                    height,
                    fps,
                })))
            } else {
                Err(error_on("didn't find <width>x<height> in first video Stream Output"))
            };
        }

        let frame_str = line_trimmed.trim_start_matches("frame=");
        if frame_str.len() < line_trimmed.len() {
            if let Some((frame_num_str, _others)) = frame_str.split_once(' ') {
                let frame = frame_num_str
                    .trim()
                    .parse::<u64>()
                    .map_err(|_| error_on("frame is no number"))?;

                let frame_upd = FrameUpdate { frame };
                Ok(Some(VideoInfoMessage::Frame(frame_upd)))
            } else {
                Ok(None)
            }
        } else {
            Ok(None)
        }
    }

    fn iter_on<'a, I>(&'a mut self, lines: I) -> impl Iterator<Item = FilteredInfo> + 'a
    where
        I: IntoIterator<Item = &'a str> + 'a,
    {
        fn un_opt(info: InfoResult) -> Option<FilteredInfo> {
            dbg!(&info);
            match info {
                Err(e) => Some(Err(e)),
                Ok(Some(m)) => Some(Ok(m)),
                Ok(None) => None,
            }
        }

        lines.into_iter().map(|l| self.push(l)).filter_map(un_opt)
    }
}

#[cfg(test)]
mod test {
    use super::{InfoParser, OutputStream, VideoInfoMessage};

    static TEST_INFO: &str = r#"Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'media/huhu_test.mp4':
  Metadata:
    major_brand     : isom
    minor_version   : 512
    compatible_brands: isomiso2avc1mp41
    title           : Session streamed with GStreamer
    encoder         : Lavf58.45.100
    comment         : rtsp-server
  Duration: 00:29:58.68, start: 0.000000, bitrate: 650 kb/s
  Stream #0:0(und): Video: h264 (Main) (avc1 / 0x31637661), yuvj420p(pc, bt709), 1280x720 [SAR 1:1 DAR 16:9], 647 kb/s, 29.59 fps, 30 tbr, 90k tbn, 180k tbc (default)
Metadata:
  handler_name    : VideoHandler
  vendor_id       : [0][0][0][0]
Stream mapping:
  Stream #0:0 -> #0:0 (h264 (native) -> rawvideo (native))
Press [q] to stop, [?] for help
[swscaler @ 0x7fb0ac4dc000] deprecated pixel format used, make sure you did set range correctly
Output #0, image2pipe, to 'pipe:':
  Metadata:
    major_brand     : isom
    minor_version   : 512
    compatible_brands: isomiso2avc1mp41
    title           : Session streamed with GStreamer
    comment         : rtsp-server
        encoder         : Lavf58.76.100
  Stream #0:0(und): Video: rawvideo (BGR[24] / 0x18524742), bgr24(pc, gbr/bt709/bt709, progressive), 1280x720 [SAR 1:1 DAR 16:9], q=2-31, 663552 kb/s, 30 fps, 30 tbn (default)
    Metadata:
      handler_name    : VideoHandler
      vendor_id       : [0][0][0][0]
      encoder         : Lavc58.134.100 rawvideo

frame= 3926 fps=978 q=-0.0 size=10600200kB time=00:02:10.86 bitrate=663552.0kbits/s speed=32.6x
frame= 4026 fps=1002 q=-0.0 size=10870200kB time=00:02:14.20 bitrate=663552.0kbits/s speed=33.4x
frame=27045 fps=1019 q=-0.0 size=73021500kB time=00:15:01.50 bitrate=663552.0kbits/s dup=0 drop=5 speed=  34x"#;

    #[test]
    fn test_parse_info() {
        let mut parser = InfoParser::default();
        let mut infos = parser.iter_on(TEST_INFO.lines());

        assert_eq!(
            infos.next().unwrap(),
            Ok(VideoInfoMessage::OutputMeta(OutputStream {
                num: 0,
                to: "pipe:".to_string(),
                width: 720,
                height: 1280,
                fps: Some(30f32),
            }))
        );
    }
}
