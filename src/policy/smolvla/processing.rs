//! 本地 tokenizer、checkpoint processor 契约及与上游一致的数值预处理。

use std::{
    collections::HashMap,
    path::{Component, Path},
};

use anyhow::{Context, Result, bail, ensure};
use burn::prelude::{Device, Tensor, TensorData};
use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;
use serde_json::Value;
use tokenizers::{
    PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer, TruncationDirection,
    TruncationParams,
};

use super::config::{PolicyFeature, SmolVlaConfig};
use crate::protocol::FeatureSpec;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Pipeline {
    #[serde(rename = "name")]
    _name: String,
    steps: Vec<Step>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Step {
    registry_name: String,
    config: Value,
    state_file: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct NormalizerConfig {
    #[serde(default = "default_eps")]
    eps: f32,
    features: HashMap<String, PolicyFeature>,
    norm_map: HashMap<String, String>,
}

fn default_eps() -> f32 {
    1e-8
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TokenizerConfig {
    max_length: usize,
    task_key: String,
    padding_side: String,
    padding: String,
    truncation: bool,
    // 路径由调用方显式提供，不从 checkpoint 下载资源。
    #[serde(rename = "tokenizer_name")]
    _tokenizer_name: Option<String>,
}

#[derive(Clone, Debug)]
enum Normalization {
    Identity,
    MeanStd {
        mean: Vec<f32>,
        std: Vec<f32>,
        eps: f32,
    },
    Range {
        low: Vec<f32>,
        span: Vec<f32>,
    },
}

impl Normalization {
    fn normalize(&self, value: f32, index: usize) -> f32 {
        match self {
            Self::Identity => value,
            Self::MeanStd { mean, std, eps } => (value - mean[index]) / (std[index] + eps),
            Self::Range { low, span } => 2.0 * (value - low[index]) / span[index] - 1.0,
        }
    }
    fn unnormalize(&self, value: f32, index: usize) -> f32 {
        match self {
            Self::Identity => value,
            // 官方反归一化不加 eps，因此不是 normalize 的严格数学逆。
            Self::MeanStd { mean, std, .. } => value * std[index] + mean[index],
            Self::Range { low, span } => (value + 1.0) / 2.0 * span[index] + low[index],
        }
    }
}

/// 加载时检查所有可支持语义，运行时只处理已验证的向量和 RGB 数据。
#[derive(Debug)]
pub(super) struct Processor {
    tokenizer: Tokenizer,
    state_norm: Normalization,
    action_norm: Normalization,
    state_dim: usize,
    max_state_dim: usize,
    action_dim: usize,
    resize: Option<[usize; 2]>,
}

impl Processor {
    /// 从 checkpoint 的两条 pipeline 和本地 tokenizer 目录加载，错误均在启动阶段报告。
    pub fn load(dir: &Path, cfg: &SmolVlaConfig, tokenizer_dir: &Path) -> Result<Self> {
        let pre = load_pipeline(
            &dir.join("policy_preprocessor.json"),
            &[
                "rename_observations_processor",
                "to_batch_processor",
                "smolvla_new_line_processor",
                "tokenizer_processor",
                "device_processor",
                "normalizer_processor",
            ],
        )?;
        let post = load_pipeline(
            &dir.join("policy_postprocessor.json"),
            &["unnormalizer_processor", "device_processor"],
        )?;
        // 先验证统计形状，防止错误 checkpoint 被后续 pad 或切片掩盖。
        let pre_norms = load_normalizers(dir, &pre.steps[5], cfg, false)?;
        let post_norms = load_normalizers(dir, &post.steps[0], cfg, true)?;
        validate_steps(&pre, &post)?;
        let token_config: TokenizerConfig = serde_json::from_value(pre.steps[3].config.clone())
            .context("parsing tokenizer_processor config")?;
        ensure!(
            token_config.task_key == "task",
            "unsupported tokenizer task_key={}",
            token_config.task_key
        );
        ensure!(
            token_config.padding_side == "right",
            "tokenizer requires right padding"
        );
        ensure!(
            token_config.truncation,
            "tokenizer requires truncation=true"
        );
        ensure!(
            token_config.max_length == cfg.tokenizer_max_length
                && token_config.padding == cfg.pad_language_to,
            "tokenizer processor disagrees with tokenizer_max_length/pad_language_to in config.json"
        );
        let tokenizer = load_tokenizer(tokenizer_dir, &token_config)?;
        Ok(Self {
            tokenizer,
            state_norm: pre_norms
                .get("observation.state")
                .context("missing state normalizer")?
                .clone(),
            action_norm: post_norms
                .get("action")
                .context("missing action unnormalizer")?
                .clone(),
            state_dim: cfg.state_dim(),
            max_state_dim: cfg.max_state_dim,
            action_dim: cfg.action_dim(),
            resize: cfg.resize_imgs_with_padding,
        })
    }

    /// 任务只在缺少末尾换行时补一次；返回单样本的 token 与有效位置 mask。
    pub fn tokenize(&self, task: &str) -> Result<(Vec<i64>, Vec<bool>)> {
        let task = newline_task(task);
        let encoded = self
            .tokenizer
            .encode(task.as_ref(), true)
            .map_err(|e| anyhow::anyhow!("tokenizing task: {e}"))?;
        Ok((
            encoded.get_ids().iter().map(|&id| i64::from(id)).collect(),
            encoded
                .get_attention_mask()
                .iter()
                .map(|&mask| mask != 0)
                .collect(),
        ))
    }

    /// 按真实 state 维度归一化，再在右侧补零到模型输入维度。
    ///
    /// # Panics
    /// 调用方必须先按合约验证输入长度。
    pub fn state(&self, data: &[f32], device: &Device) -> Tensor<2> {
        assert_eq!(
            data.len(),
            self.state_dim,
            "state input length must match validated dimension {}",
            self.state_dim
        );
        let mut padded = vec![0.0; self.max_state_dim];
        for (i, &value) in data.iter().enumerate() {
            padded[i] = self.state_norm.normalize(value, i);
        }
        Tensor::from_data(TensorData::new(padded, [1, self.max_state_dim]), device)
    }

    /// 输入为 uint8 HWC；输出为 [1,3,H,W]，适用于 SigLIP 的 [-1,1] 视觉输入。
    ///
    /// # Panics
    /// 调用方必须验证图像长度、非零尺寸和尺寸乘积。
    pub fn image(&self, data: &[u8], h: usize, w: usize, device: &Device) -> Tensor<4> {
        let (pixels, height, width) = resize_with_pad(data, h, w, self.resize);
        Tensor::from_data(TensorData::new(pixels, [1, 3, height, width]), device)
    }

    /// 输入按 action_dim 连续排列，父模块须先从模型 max_action_dim 中取出真实动作维度。
    ///
    /// # Panics
    /// 输入长度必须为 action_dim 的整数倍。
    pub fn unnormalize(&self, data: &[f32]) -> Vec<f32> {
        assert!(
            data.len().is_multiple_of(self.action_dim),
            "action input length {} must be divisible by {}",
            data.len(),
            self.action_dim
        );
        data.iter()
            .enumerate()
            .map(|(i, &value)| self.action_norm.unnormalize(value, i % self.action_dim))
            .collect()
    }
}

fn newline_task(task: &str) -> std::borrow::Cow<'_, str> {
    if task.ends_with('\n') {
        task.into()
    } else {
        format!("{task}\n").into()
    }
}

fn load_pipeline(path: &Path, expected: &[&str]) -> Result<Pipeline> {
    let bytes =
        std::fs::read(path).with_context(|| format!("reading processor {}", path.display()))?;
    let pipeline: Pipeline = serde_json::from_slice(&bytes)
        .with_context(|| format!("parsing processor {}", path.display()))?;
    let names: Vec<_> = pipeline
        .steps
        .iter()
        .map(|step| step.registry_name.as_str())
        .collect();
    ensure!(
        names == expected,
        "unsupported processor steps in {}: {names:?}; expected {expected:?}",
        path.display()
    );
    for step in &pipeline.steps {
        ensure!(
            step.config.is_object(),
            "{} config must be an object",
            step.registry_name
        );
        ensure!(
            step.state_file.is_none()
                || matches!(
                    step.registry_name.as_str(),
                    "normalizer_processor" | "unnormalizer_processor"
                ),
            "unexpected state_file for {}",
            step.registry_name
        );
    }
    Ok(pipeline)
}

fn validate_steps(pre: &Pipeline, post: &Pipeline) -> Result<()> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct RenameConfig {
        rename_map: HashMap<String, String>,
    }
    let rename: RenameConfig =
        serde_json::from_value(pre.steps[0].config.clone()).context("parsing rename processor")?;
    ensure!(
        rename.rename_map.is_empty(),
        "non-empty rename_map is unsupported; deployment features must use checkpoint names directly"
    );
    for index in [1, 2] {
        ensure!(
            pre.steps[index]
                .config
                .as_object()
                .is_some_and(|map| map.is_empty()),
            "unsupported {} options",
            pre.steps[index].registry_name
        );
    }
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct DeviceConfig {
        device: Option<String>,
        float_dtype: Option<String>,
    }
    for step in [&pre.steps[4], &post.steps[1]] {
        let config: DeviceConfig =
            serde_json::from_value(step.config.clone()).context("parsing device_processor")?;
        // 设备选择由 Burn 的运行时 Device 接管；这里只允许等价的 float32。
        ensure!(
            config
                .float_dtype
                .as_deref()
                .is_none_or(|s| matches!(s, "float32" | "torch.float32")),
            "unsupported processor float_dtype={:?}",
            config.float_dtype
        );
        ensure!(
            config
                .device
                .as_deref()
                .is_none_or(|s| s == "cpu" || s == "cuda" || s == "mps" || s.starts_with("cuda:")),
            "unsupported processor device={:?}",
            config.device
        );
    }
    Ok(())
}

fn load_tokenizer(dir: &Path, config: &TokenizerConfig) -> Result<Tokenizer> {
    let config_path = dir.join("tokenizer_config.json");
    let metadata: Value = serde_json::from_slice(
        &std::fs::read(&config_path)
            .with_context(|| format!("reading {}", config_path.display()))?,
    )
    .context("parsing tokenizer_config.json")?;
    let mut tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("loading local tokenizer {}: {e}", dir.display()))?;
    let pad_token = metadata
        .get("pad_token")
        .and_then(|v| {
            v.as_str()
                .or_else(|| v.get("content").and_then(Value::as_str))
        })
        .context("tokenizer_config.json must define pad_token")?;
    let pad_id = tokenizer
        .token_to_id(pad_token)
        .with_context(|| format!("pad_token {pad_token:?} missing from tokenizer vocabulary"))?;
    if let Some(explicit_id) = metadata.get("pad_token_id") {
        ensure!(
            explicit_id.as_u64() == Some(u64::from(pad_id)),
            "tokenizer pad_token_id disagrees with pad_token vocabulary ID"
        );
    }
    let direction = match metadata
        .get("truncation_side")
        .and_then(Value::as_str)
        .unwrap_or("right")
    {
        "right" => TruncationDirection::Right,
        "left" => TruncationDirection::Left,
        other => bail!("unsupported tokenizer truncation_side={other}"),
    };
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length: config.max_length,
            direction,
            ..Default::default()
        }))
        .map_err(|e| anyhow::anyhow!("configuring tokenizer truncation: {e}"))?;
    let strategy = match config.padding.as_str() {
        "max_length" => PaddingStrategy::Fixed(config.max_length),
        "longest" => PaddingStrategy::BatchLongest,
        other => bail!("unsupported tokenizer padding={other}"),
    };
    tokenizer.with_padding(Some(PaddingParams {
        strategy,
        direction: PaddingDirection::Right,
        pad_id,
        pad_token: pad_token.into(),
        ..Default::default()
    }));
    Ok(tokenizer)
}

fn load_normalizers(
    dir: &Path,
    step: &Step,
    cfg: &SmolVlaConfig,
    inverse: bool,
) -> Result<HashMap<String, Normalization>> {
    let config: NormalizerConfig = serde_json::from_value(step.config.clone())
        .with_context(|| format!("parsing {} config", step.registry_name))?;
    ensure!(
        config.eps.is_finite() && config.eps > 0.0,
        "normalizer eps must be finite and positive"
    );
    let expected: HashMap<_, _> = if inverse {
        cfg.output_features
            .iter()
            .map(|(n, f)| (n.clone(), f.clone()))
            .collect()
    } else {
        cfg.input_features
            .iter()
            .chain(cfg.output_features.iter())
            .map(|(n, f)| (n.clone(), f.clone()))
            .collect()
    };
    ensure!(
        config.features == expected,
        "{} features disagree with config.json",
        step.registry_name
    );
    for kind in ["VISUAL", "STATE", "ACTION"] {
        let actual = config.norm_map.get(kind).map_or("IDENTITY", String::as_str);
        let wanted = cfg
            .normalization_mapping
            .get(kind)
            .map_or("IDENTITY", String::as_str);
        ensure!(
            actual == wanted,
            "{} normalization mapping for {kind}: {actual} != config.json {wanted}",
            step.registry_name
        );
    }
    ensure!(
        config
            .norm_map
            .keys()
            .all(|k| matches!(k.as_str(), "VISUAL" | "STATE" | "ACTION")),
        "unsupported normalization feature type"
    );
    let stats = if let Some(file) = &step.state_file {
        let mut parts = Path::new(file).components();
        ensure!(
            matches!(parts.next(), Some(Component::Normal(_))) && parts.next().is_none(),
            "state_file must be a local filename: {file}"
        );
        Some(
            std::fs::read(dir.join(file))
                .with_context(|| format!("reading normalizer stats {file}"))?,
        )
    } else {
        None
    };
    let tensors = stats
        .as_deref()
        .map(SafeTensors::deserialize)
        .transpose()
        .context("parsing normalizer safetensors")?;
    let mut normalizers = HashMap::new();
    for (name, feature) in &config.features {
        let mode = config
            .norm_map
            .get(&feature.feature_type)
            .map_or("IDENTITY", String::as_str);
        if feature.feature_type != "VISUAL" {
            // 即使某种模式不使用 mean/std，也不容许相同向量携带另一维度的统计。
            if let Some(tensors) = &tensors {
                for (key, tensor) in tensors.tensors() {
                    if key
                        .strip_prefix(name)
                        .is_some_and(|suffix| suffix.starts_with('.') && suffix != ".count")
                    {
                        ensure!(
                            tensor.shape() == feature.shape,
                            "normalization stats {key}: shape {:?} disagrees with feature shape {:?}; refusing to truncate",
                            tensor.shape(),
                            feature.shape
                        );
                    }
                }
            }
        }
        let read = |stat: &str| -> Result<Vec<f32>> {
            let key = format!("{name}.{stat}");
            let tensor = tensors
                .as_ref()
                .context("normalization requires state_file")?
                .tensor(&key)
                .with_context(|| format!("missing normalization statistic {key}"))?;
            ensure!(
                tensor.dtype() == Dtype::F32,
                "normalization statistic {key} must be F32, got {:?}",
                tensor.dtype()
            );
            ensure!(
                tensor.shape() == feature.shape,
                "normalization statistic {key}: shape {:?} != {:?}",
                tensor.shape(),
                feature.shape
            );
            let values: Vec<f32> = tensor
                .data()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|bytes| f32::from_le_bytes(*bytes))
                .collect();
            ensure!(
                values.iter().all(|v| v.is_finite()),
                "non-finite normalization statistic {key}"
            );
            Ok(values)
        };
        let norm = match mode {
            "IDENTITY" => Normalization::Identity,
            "MEAN_STD" => {
                let mean = read("mean")?;
                let std = read("std")?;
                ensure!(
                    std.iter().all(|&v| v >= 0.0),
                    "negative normalization std for {name}"
                );
                Normalization::MeanStd {
                    mean,
                    std,
                    eps: config.eps,
                }
            }
            "MIN_MAX" | "QUANTILES" | "QUANTILE10" => {
                let (low_key, high_key) = match mode {
                    "MIN_MAX" => ("min", "max"),
                    "QUANTILES" => ("q01", "q99"),
                    _ => ("q10", "q90"),
                };
                let low = read(low_key)?;
                let high = read(high_key)?;
                let mut span: Vec<f32> = high.iter().zip(&low).map(|(&hi, &lo)| hi - lo).collect();
                ensure!(
                    span.iter().all(|&v| v.is_finite() && v >= 0.0),
                    "invalid normalization range for {name}"
                );
                for value in &mut span {
                    if *value == 0.0 {
                        *value = config.eps;
                    }
                }
                Normalization::Range { low, span }
            }
            other => bail!("unsupported normalization mode {other} for {name}"),
        };
        normalizers.insert(name.clone(), norm);
    }
    Ok(normalizers)
}

/// 对齐上游 F.interpolate(..., bilinear, align_corners=false)，补零只在左侧和顶部。
fn resize_with_pad(
    data: &[u8],
    h: usize,
    w: usize,
    resize: Option<[usize; 2]>,
) -> (Vec<f32>, usize, usize) {
    assert!(
        h > 0 && w > 0,
        "RGB dimensions must be positive, got {h}x{w}"
    );
    assert_eq!(
        Some(data.len()),
        h.checked_mul(w).and_then(|v| v.checked_mul(3)),
        "RGB byte length does not match {h}x{w}x3"
    );
    let [width, height] = resize.unwrap_or([w, h]);
    assert!(
        width > 0 && height > 0,
        "resize dimensions must be positive"
    );
    let output_len = height
        .checked_mul(width)
        .and_then(|v| v.checked_mul(3))
        .expect("validated RGB output dimensions must not overflow");
    let ratio = (w as f64 / width as f64).max(h as f64 / height as f64);
    let rh = (h as f64 / ratio) as usize;
    let rw = (w as f64 / ratio) as usize;
    assert!(
        rh > 0 && rw > 0,
        "aspect ratio produces an empty resized image: {h}x{w} -> {height}x{width}"
    );
    let pad_h = height - rh;
    let pad_w = width - rw;
    let coordinates = |size: usize, resized: usize| -> Vec<(usize, usize, f32)> {
        let scale = (size as f64 / resized as f64) as f32;
        (0..resized)
            .map(|i| {
                let position = (scale * (i as f32 + 0.5) - 0.5).max(0.0);
                let first = position as usize;
                (first, (first + 1).min(size - 1), position - first as f32)
            })
            .collect()
    };
    let ys = coordinates(h, rh);
    let xs = coordinates(w, rw);
    let mut out = vec![-1.0; output_len];
    for channel in 0..3 {
        for (y, &(y0, y1, wy)) in ys.iter().enumerate() {
            for (x, &(x0, x1, wx)) in xs.iter().enumerate() {
                let pixel = |yy, xx| f32::from(data[(yy * w + xx) * 3 + channel]) * (1.0 / 255.0);
                let top = (1.0 - wx) * pixel(y0, x0) + wx * pixel(y0, x1);
                let bottom = (1.0 - wx) * pixel(y1, x0) + wx * pixel(y1, x1);
                let value = (1.0 - wy) * top + wy * bottom;
                out[(channel * height + y + pad_h) * width + x + pad_w] = value * 2.0 - 1.0;
            }
        }
    }
    (out, height, width)
}

/// 部署使用 checkpoint 的特征名；图像在线协议为 HWC，checkpoint 为 CHW。
pub(super) fn validate_contract(
    cfg: &SmolVlaConfig,
    features: &[FeatureSpec],
    action: &FeatureSpec,
) -> Result<()> {
    ensure!(
        features.len() == cfg.input_features.len(),
        "SmolVLA feature count disagrees with checkpoint"
    );
    for (name, expected) in &cfg.input_features {
        let mut matches = features.iter().filter(|feature| &feature.name == name);
        let feature = matches
            .next()
            .with_context(|| format!("missing SmolVLA feature {name}"))?;
        ensure!(matches.next().is_none(), "duplicate SmolVLA feature {name}");
        match expected.feature_type.as_str() {
            "STATE" => ensure!(
                feature.kind == "tensor"
                    && feature.dtype == "float32"
                    && feature.shape == [expected.shape[0] as u64],
                "SmolVLA requires {name} float32 {:?}",
                expected.shape
            ),
            "VISUAL" => {
                let [_, h, w] = expected.shape.as_slice() else {
                    bail!("invalid checkpoint image shape for {name}");
                };
                ensure!(
                    feature.kind == "rgb"
                        && feature.dtype == "uint8"
                        && feature.shape == [*h as u64, *w as u64, 3],
                    "SmolVLA requires {name} uint8 HWC [{h},{w},3]"
                );
            }
            other => bail!("unsupported feature type {other}"),
        }
    }
    ensure!(
        action.name == "action"
            && action.kind == "tensor"
            && action.dtype == "float32"
            && action.shape == [cfg.action_dim() as u64],
        "SmolVLA action contract disagrees with checkpoint dimension {}",
        cfg.action_dim()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn newline_is_idempotent_and_preserves_existing_newlines() {
        assert_eq!(newline_task("pick"), "pick\n");
        assert_eq!(newline_task("pick\n"), "pick\n");
        assert_eq!(newline_task("pick\n\n"), "pick\n\n");
    }

    #[test]
    fn bilinear_uses_half_pixels_and_top_left_padding() {
        let data = [0, 0, 0, 255, 255, 255];
        let (pixels, h, w) = resize_with_pad(&data, 1, 2, Some([4, 4]));
        assert_eq!((h, w), (4, 4));
        for channel in pixels.as_chunks::<16>().0 {
            assert_eq!(&channel[..8], &[-1.0; 8]);
            assert_eq!(&channel[8..12], &[-1.0, -0.5, 0.5, 1.0]);
            assert_eq!(&channel[12..], &[-1.0, -0.5, 0.5, 1.0]);
        }
        let (pixels, _, _) = resize_with_pad(&[255; 6], 2, 1, Some([4, 2]));
        assert_eq!(
            &pixels[..8],
            &[-1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0]
        );
    }

    #[test]
    fn non_integer_resize_matches_torch_211() {
        let data: Vec<u8> = (0..45).map(|i| ((i * 37 + 11) % 256) as u8).collect();
        let (pixels, _, _) = resize_with_pad(&data, 3, 5, Some([4, 6]));
        // torch 2.11 interpolate(size=(2,4), align_corners=False)，顶部补四行。
        let expected = [
            -0.7205882,
            0.17941177,
            0.012745142,
            0.53627443,
            -0.21470588,
            0.1205883,
            -0.046078384,
            -0.27549016,
        ];
        for (&actual, wanted) in pixels[16..24].iter().zip(expected) {
            assert!(
                (actual - wanted).abs() < 1e-6,
                "actual={actual}, torch={wanted}"
            );
        }
    }

    struct Fixture(std::path::PathBuf);
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    impl Fixture {
        fn new(state_stats_dim: usize) -> (Self, SmolVlaConfig) {
            use safetensors::tensor::TensorView;
            use tokenizers::{
                models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace,
            };
            let dir = std::env::temp_dir().join(format!("pneuma-smolvla-{}", uuid::Uuid::new_v4()));
            std::fs::create_dir(&dir).unwrap();
            let fixture = Self(dir);
            let input = serde_json::json!({"observation.state":{"type":"STATE","shape":[6]},
                "observation.images.camera":{"type":"VISUAL","shape":[3,3,5]}});
            let output = serde_json::json!({"action":{"type":"ACTION","shape":[2]}});
            let config = serde_json::json!({"input_features":input,"output_features":output,
                "tokenizer_max_length":4,"pad_language_to":"max_length"});
            fixture.write("config.json", &config);
            let cfg = super::super::config::parse(&fixture.0.join("config.json")).unwrap();
            let mut features = input.as_object().unwrap().clone();
            features.extend(output.as_object().unwrap().clone());
            let mapping =
                serde_json::json!({"VISUAL":"IDENTITY","STATE":"MEAN_STD","ACTION":"MEAN_STD"});
            fixture.write("policy_preprocessor.json", &serde_json::json!({"name":"policy_preprocessor","steps":[
                {"registry_name":"rename_observations_processor","config":{"rename_map":{}}},
                {"registry_name":"to_batch_processor","config":{}},
                {"registry_name":"smolvla_new_line_processor","config":{}},
                {"registry_name":"tokenizer_processor","config":{"max_length":4,"task_key":"task",
                    "padding_side":"right","padding":"max_length","truncation":true,"tokenizer_name":"unused"}},
                {"registry_name":"device_processor","config":{"device":"cuda","float_dtype":null}},
                {"registry_name":"normalizer_processor","config":{"features":features,"norm_map":mapping},"state_file":"stats.safetensors"}
            ]}));
            fixture.write("policy_postprocessor.json", &serde_json::json!({"name":"policy_postprocessor","steps":[
                {"registry_name":"unnormalizer_processor","config":{"features":output,"norm_map":mapping},"state_file":"stats.safetensors"},
                {"registry_name":"device_processor","config":{"device":"cpu","float_dtype":null}}
            ]}));
            let state_mean = vec![1.0f32; state_stats_dim]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>();
            let state_std = vec![2.0f32; state_stats_dim]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>();
            let action_mean = [10.0f32, 20.0]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>();
            let action_std = [2.0f32, 4.0]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>();
            let tensors = [
                (
                    "observation.state.mean",
                    TensorView::new(Dtype::F32, vec![state_stats_dim], &state_mean).unwrap(),
                ),
                (
                    "observation.state.std",
                    TensorView::new(Dtype::F32, vec![state_stats_dim], &state_std).unwrap(),
                ),
                (
                    "action.mean",
                    TensorView::new(Dtype::F32, vec![2], &action_mean).unwrap(),
                ),
                (
                    "action.std",
                    TensorView::new(Dtype::F32, vec![2], &action_std).unwrap(),
                ),
            ];
            std::fs::write(
                fixture.0.join("stats.safetensors"),
                safetensors::tensor::serialize(tensors, None).unwrap(),
            )
            .unwrap();
            let model = WordLevel::builder()
                .vocab(
                    [
                        ("<unk>".into(), 0),
                        ("<pad>".into(), 7),
                        ("a".into(), 1),
                        ("b".into(), 2),
                        ("c".into(), 3),
                        ("d".into(), 4),
                        ("e".into(), 5),
                    ]
                    .into_iter()
                    .collect(),
                )
                .build()
                .unwrap();
            let mut tokenizer = Tokenizer::new(model);
            tokenizer.with_pre_tokenizer(Some(Whitespace));
            tokenizer
                .save(fixture.0.join("tokenizer.json"), false)
                .unwrap();
            fixture.write(
                "tokenizer_config.json",
                &serde_json::json!({"pad_token":"<pad>","truncation_side":"left"}),
            );
            (fixture, cfg)
        }
        fn write(&self, name: &str, value: &Value) {
            std::fs::write(self.0.join(name), serde_json::to_vec(value).unwrap()).unwrap();
        }
    }

    #[test]
    fn pipeline_loads_local_tokenizer_normalizes_then_pads() {
        let (fixture, cfg) = Fixture::new(6);
        let processor = Processor::load(&fixture.0, &cfg, &fixture.0).unwrap();
        assert_eq!(
            processor.tokenize("a b").unwrap(),
            (vec![1, 2, 7, 7], vec![true, true, false, false])
        );
        assert_eq!(processor.tokenize("a b c d e").unwrap().0, vec![2, 3, 4, 5]);
        let state = processor
            .state(&[3.0; 6], &Device::flex())
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        assert_eq!(&state[..6], &[1.0; 6]);
        assert_eq!(&state[6..], &[0.0; 26]);
        assert_eq!(
            processor.unnormalize(&[1.0, 2.0, -1.0, -2.0]),
            vec![12.0, 28.0, 8.0, 12.0]
        );
    }

    #[test]
    fn state_six_stats_eight_is_rejected_during_load() {
        let (fixture, cfg) = Fixture::new(8);
        let error = Processor::load(&fixture.0, &cfg, &fixture.0).unwrap_err();
        let message = format!("{error:#}");
        assert!(
            message.contains("observation.state.")
                && message.contains("[8]")
                && message.contains("[6]"),
            "{message}"
        );
    }

    #[test]
    fn unsupported_step_and_rename_are_explicit_errors() {
        let (fixture, cfg) = Fixture::new(6);
        let path = fixture.0.join("policy_preprocessor.json");
        let mut value: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        value["steps"][0]["config"]["rename_map"] = serde_json::json!({"old":"new"});
        fixture.write("policy_preprocessor.json", &value);
        assert!(
            Processor::load(&fixture.0, &cfg, &fixture.0)
                .unwrap_err()
                .to_string()
                .contains("rename_map")
        );
        value["steps"][2]["registry_name"] = "custom_processor".into();
        fixture.write("policy_preprocessor.json", &value);
        assert!(
            Processor::load(&fixture.0, &cfg, &fixture.0)
                .unwrap_err()
                .to_string()
                .contains("custom_processor")
        );
    }

    #[test]
    fn normalization_matches_official_epsilon_and_inverse() {
        let norm = Normalization::MeanStd {
            mean: vec![2.0],
            std: vec![0.0],
            eps: 0.25,
        };
        assert_eq!(norm.normalize(3.0, 0), 4.0);
        assert_eq!(norm.unnormalize(4.0, 0), 2.0);
        let range = Normalization::Range {
            low: vec![1.0],
            span: vec![4.0],
        };
        assert_eq!(range.normalize(3.0, 0), 0.0);
        assert_eq!(range.unnormalize(0.0, 0), 3.0);
    }
}
