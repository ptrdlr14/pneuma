//! Cosmos3 Edge 的文本提示。
//!
//! 官方把任务写成 JSON（取景、时长、分辨率等），再用固定的聊天模板包起来，末尾追加
//! `<|im_end|><|vision_start|>`。模板来源于 checkpoint 的 `chat_template.jinja`，对
//! 「单条文本用户消息」这一固定形状可以展开成常量前后缀。

// 提示在策略接线前不参与运行时构建；届时移除此豁免。
#![allow(dead_code)]

/// 用户消息之前的部分：空系统回合加用户回合开头。
pub(super) const PREFIX: &str = "\n<|im_start|>system\n<|im_end|>\n<|im_start|>user\n";
/// 用户消息之后的部分：结束回合、助手回合开头、思考标记、结束回合与生成流起始符。
pub(super) const SUFFIX: &str =
    "<|im_end|>\n<|im_start|>assistant\n<think>\n<|im_end|><|vision_start|>";
/// 多视角画布的固定取景说明。
pub(super) const CONCAT_VIEW: &str =
    "This video contains concatenated views from multiple camera perspectives.";

/// 画布尺寸。
#[derive(Clone, Copy, Debug)]
pub(crate) struct Canvas {
    pub(crate) height: usize,
    pub(crate) width: usize,
}

/// 生成 JSON 提示。`task` 为空时返回空串（对应官方的空提示）。
pub(crate) fn caption(
    task: &str,
    canvas: Canvas,
    fps: f64,
    idle_frames: usize,
    chunk_size: usize,
    use_state: bool,
    view: &str,
) -> String {
    let task = task.trim();
    if task.is_empty() {
        return String::new();
    }
    let task = if task.ends_with(['.', '!', '?']) {
        task.to_string()
    } else {
        format!("{task}.")
    };
    let duration = (chunk_size + 1) as f64 / fps;
    // 官方用 Python 的 `round`（银行家舍入），这里对 .5 做同样处理。
    let rounded = round_half_even(duration) as i64;
    let framing = [CONCAT_VIEW, view.trim()]
        .iter()
        .filter(|part| !part.is_empty())
        .copied()
        .collect::<Vec<_>>()
        .join(" ");
    let duration_seconds = duration as i64;
    let time = format!("0:00-{}:{:02}", rounded / 60, rounded % 60);
    let action_count = chunk_size + usize::from(use_state);
    format!(
        "{{\"cinematography\": {{\"framing\": \"{}\"}}, \"actions\": [{{\"time\": \"{}\", \"description\": \"{}\", \"idle_frame\": \"{} out of {}.\"}}], \"duration\": \"{}s\", \"fps\": {}, \"resolution\": {{\"H\": {}, \"W\": {}}}, \"aspect_ratio\": \"{}\"}}",
        escape(&framing),
        time,
        escape(&task),
        idle_frames,
        action_count,
        duration_seconds,
        format_float(fps.trunc()),
        canvas.height,
        canvas.width,
        aspect_ratio(canvas),
    )
}

/// 包上固定模板，得到送进分词器的完整文本。
pub(crate) fn text(prompt: &str) -> String {
    format!("{PREFIX}{prompt}{SUFFIX}")
}

/// Python `json.dumps` 默认会转义的字符；非 ASCII 保留原样（官方会转成 `\uXXXX`）。
fn escape(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for character in value.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            control if (control as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", control as u32));
            }
            other => out.push(other),
        }
    }
    out
}

/// JSON 里的浮点：`20.0` 而不是 `20`。
fn format_float(value: f64) -> String {
    if value.fract() == 0.0 {
        format!("{value:.1}")
    } else {
        format!("{value}")
    }
}

fn round_half_even(value: f64) -> f64 {
    let floor = value.floor();
    let fraction = value - floor;
    if (fraction - 0.5).abs() < f64::EPSILON {
        if (floor as i64) % 2 == 0 {
            floor
        } else {
            floor + 1.0
        }
    } else {
        value.round()
    }
}

/// 画布宽高比：命中官方尺寸表时用名字，否则用约分后的 `宽,高`。
fn aspect_ratio(canvas: Canvas) -> String {
    const TIERS: [&[(&str, usize, usize)]; 3] = [
        &[
            ("1,1", 256, 256),
            ("4,3", 320, 256),
            ("3,4", 256, 320),
            ("16,9", 320, 192),
            ("9,16", 192, 320),
        ],
        &[
            ("1,1", 640, 640),
            ("4,3", 736, 544),
            ("3,4", 544, 736),
            ("16,9", 832, 480),
            ("9,16", 480, 832),
            ("2,3", 512, 768),
        ],
        &[
            ("1,1", 960, 960),
            ("4,3", 1104, 832),
            ("3,4", 832, 1104),
            ("16,9", 1280, 720),
            ("9,16", 720, 1280),
        ],
    ];
    for tier in TIERS {
        for (name, width, height) in tier {
            if (*width, *height) == (canvas.width, canvas.height) {
                return (*name).to_string();
            }
        }
    }
    let divisor = gcd(canvas.width, canvas.height);
    format!("{},{}", canvas.width / divisor, canvas.height / divisor)
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let remainder = a % b;
        a = b;
        b = remainder;
    }
    a.max(1)
}

#[cfg(test)]
mod reference {
    use super::*;

    #[test]
    #[ignore = "需要 cosmos3 提示参照"]
    fn caption_matches_reference() {
        let parity = std::path::PathBuf::from(
            std::env::var("PNEUMA_COSMOS3_PROMPT").expect("PNEUMA_COSMOS3_PROMPT"),
        );
        let expected = std::fs::read_to_string(parity.join("caption.txt"))
            .expect("caption")
            .trim_end()
            .to_string();
        let view = "The left half shows the third-person view; the right half shows the wrist-mounted camera.";
        let got = caption(
            "pick up the \"cube\".",
            Canvas {
                height: 256,
                width: 256,
            },
            20.0,
            0,
            32,
            true,
            view,
        );
        assert_eq!(got, expected);
        let constants: Vec<String> = std::fs::read_to_string(parity.join("constants.txt"))
            .expect("constants")
            .lines()
            .map(str::to_string)
            .collect();
        assert_eq!(CONCAT_VIEW, constants[0]);
    }
}
