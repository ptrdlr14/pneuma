//! 将严格校验的 LeRobot checkpoint processor 导入通用 pipeline。

use std::{
    collections::{BTreeMap, HashMap},
    path::{Component, Path},
};

use anyhow::{Context, Result, bail, ensure};
use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;
use serde_json::Value;

use super::config::{PolicyFeature, SmolVlaConfig};
use crate::contract::FeatureSpec;
use crate::processor::{DType, Field, Layout, Pipeline, Schema, Step};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ImportedPipeline {
    #[serde(rename = "name")]
    _name: String,
    steps: Vec<ImportedStep>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ImportedStep {
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
    /// 导入时展开为通用算术步骤，保留官方公式的运算顺序。
    fn steps(&self, key: &str, axis: usize, inverse: bool) -> Vec<Step> {
        let affine = |offset, scale| Step::Affine {
            key: key.into(),
            offset,
            scale,
            axis,
        };
        let linear = |offset, scale| Step::Linear {
            key: key.into(),
            offset,
            scale,
            axis,
        };
        match self {
            Self::Identity => Vec::new(),
            Self::MeanStd { mean, std, eps } if !inverse => vec![affine(
                mean.clone(),
                std.iter().map(|value| value + eps).collect(),
            )],
            // 官方 inverse 不加 eps，且 std=0 合法。
            Self::MeanStd { mean, std, .. } => vec![linear(mean.clone(), std.clone())],
            Self::Range { low, span } if !inverse => vec![
                affine(low.clone(), span.clone()),
                linear(vec![-1.0], vec![2.0]),
            ],
            Self::Range { low, span } => vec![
                linear(vec![1.0], vec![1.0]),
                affine(vec![0.0], vec![2.0]),
                linear(low.clone(), span.clone()),
            ],
        }
    }
}

/// 只在导入阶段解释 LeRobot 配置，运行时复用通用步骤。
pub(super) fn import_pipelines(
    dir: &Path,
    cfg: &SmolVlaConfig,
    tokenizer_dir: &Path,
) -> Result<(Pipeline, Pipeline)> {
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
    // 先检查统计形状，避免 pad/crop 掩盖 state6/stats8 等错误。
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

    let mut schema = Schema::new();
    schema.insert(
        "observation.state".into(),
        Field {
            shape: vec![Some(cfg.state_dim())],
            dtype: DType::F32,
            layout: Layout::Flat,
            optional: false,
        },
    );
    let mut steps = vec![Step::Batch];
    for (name, feature) in cfg.image_features() {
        let (h, w) = (feature.shape[1], feature.shape[2]);
        schema.insert(
            name.into(),
            Field {
                shape: vec![Some(h), Some(w), Some(3)],
                dtype: DType::U8,
                layout: Layout::Hwc,
                optional: false,
            },
        );
        steps.push(Step::Rgb { key: name.into() });
        if let Some([width, height]) = cfg.resize_imgs_with_padding {
            steps.push(Step::ResizePad {
                key: name.into(),
                width,
                height,
            });
        }
        steps.push(Step::Affine {
            key: name.into(),
            offset: vec![0.5],
            scale: vec![0.5],
            axis: 1,
        });
    }
    schema.insert(
        "action".into(),
        Field {
            shape: vec![Some(cfg.chunk_size), Some(cfg.action_dim())],
            dtype: DType::F32,
            layout: Layout::Flat,
            optional: true,
        },
    );
    steps.extend([
        Step::Newline,
        Step::Tokenize {
            tokenizer: tokenizer_dir.to_path_buf(),
            max_length: token_config.max_length,
            padding: token_config.padding,
            padding_side: token_config.padding_side,
            special_tokens: true,
            negative: None,
        },
    ]);
    for (key, axis, width) in [
        ("observation.state", 1, cfg.max_state_dim),
        ("action", 2, cfg.max_action_dim),
    ] {
        steps.extend(
            pre_norms
                .get(key)
                .with_context(|| format!("missing {key} normalizer"))?
                .steps(key, axis, false),
        );
        steps.push(Step::Pad {
            key: key.into(),
            width,
        });
    }
    steps.push(Step::Device);
    let input = Pipeline::new(schema, steps)?;
    let output_schema = BTreeMap::from([(
        "action".into(),
        Field {
            shape: vec![None, Some(cfg.chunk_size), Some(cfg.max_action_dim)],
            dtype: DType::F32,
            layout: Layout::Flat,
            optional: false,
        },
    )]);
    let mut output_steps = vec![Step::Crop {
        key: "action".into(),
        width: cfg.action_dim(),
    }];
    output_steps.extend(
        post_norms
            .get("action")
            .context("missing action unnormalizer")?
            .steps("action", 2, true),
    );
    let output = Pipeline::new(output_schema, output_steps)?;
    Ok((input, output))
}

fn load_pipeline(path: &Path, expected: &[&str]) -> Result<ImportedPipeline> {
    let bytes =
        std::fs::read(path).with_context(|| format!("reading processor {}", path.display()))?;
    let pipeline: ImportedPipeline = serde_json::from_slice(&bytes)
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

fn validate_steps(pre: &ImportedPipeline, post: &ImportedPipeline) -> Result<()> {
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

fn load_normalizers(
    dir: &Path,
    step: &ImportedStep,
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

    use crate::processor::{Batch, Context, Data, Mode};
    use burn::tensor::{Device, Tensor, TensorData};

    fn context() -> Context {
        Context {
            device: Device::flex(),
            mode: Mode::Eval,
        }
    }

    fn prepare_image(data: Vec<u8>, h: usize, w: usize, width: usize, height: usize) -> Vec<f32> {
        let schema = BTreeMap::from([(
            "image".into(),
            Field {
                shape: vec![Some(h), Some(w), Some(3)],
                dtype: DType::U8,
                layout: Layout::Hwc,
                optional: false,
            },
        )]);
        let pipeline = Pipeline::new(
            schema,
            vec![
                Step::Batch,
                Step::Rgb {
                    key: "image".into(),
                },
                Step::ResizePad {
                    key: "image".into(),
                    width,
                    height,
                },
                Step::Affine {
                    key: "image".into(),
                    offset: vec![0.5],
                    scale: vec![0.5],
                    axis: 1,
                },
                Step::Device,
            ],
        )
        .unwrap();
        let mut batch = Batch::default();
        batch
            .values
            .insert("image".into(), Data::u8(data, vec![h, w, 3]));
        pipeline
            .run(batch, &context())
            .unwrap()
            .take("image")
            .unwrap()
            .into_float4()
            .unwrap()
            .into_data()
            .try_to_vec::<f32>()
            .unwrap()
    }

    #[test]
    fn newline_is_idempotent_and_preserves_existing_newlines() {
        let pipeline = Pipeline::new(Schema::new(), vec![Step::Newline]).unwrap();
        for (task, expected) in [
            ("pick", "pick\n"),
            ("pick\n", "pick\n"),
            ("pick\n\n", "pick\n\n"),
        ] {
            let batch = Batch {
                tasks: vec![task.into()],
                ..Batch::default()
            };
            assert_eq!(pipeline.run(batch, &context()).unwrap().tasks, [expected]);
        }
    }

    #[test]
    fn bilinear_uses_half_pixels_and_top_left_padding() {
        let pixels = prepare_image(vec![0, 0, 0, 255, 255, 255], 1, 2, 4, 4);
        for channel in pixels.as_chunks::<16>().0 {
            assert_eq!(&channel[..8], &[-1.0; 8]);
            assert_eq!(&channel[8..12], &[-1.0, -0.5, 0.5, 1.0]);
            assert_eq!(&channel[12..], &[-1.0, -0.5, 0.5, 1.0]);
        }
        let pixels = prepare_image(vec![255; 6], 2, 1, 4, 2);
        assert_eq!(
            &pixels[..8],
            &[-1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0]
        );
    }

    #[test]
    fn non_integer_resize_matches_torch_211() {
        let data = (0..45).map(|i| ((i * 37 + 11) % 256) as u8).collect();
        let pixels = prepare_image(data, 3, 5, 4, 6);
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
            Self::with_std(state_stats_dim, 2.0, [2.0, 4.0])
        }
        fn with_std(
            state_stats_dim: usize,
            state_std_value: f32,
            action_std_values: [f32; 2],
        ) -> (Self, SmolVlaConfig) {
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
            let state_std = vec![state_std_value; state_stats_dim]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>();
            let action_mean = [10.0f32, 20.0]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>();
            let action_std = action_std_values
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
            let mut tokenizer = tokenizers::Tokenizer::new(model);
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
        fn batch(&self, task: &str, targets: Option<Vec<f32>>) -> Batch {
            let mut batch = Batch {
                values: BTreeMap::from([
                    ("observation.state".into(), Data::f32(vec![3.0; 6], vec![6])),
                    (
                        "observation.images.camera".into(),
                        Data::u8(vec![0; 45], vec![3, 5, 3]),
                    ),
                ]),
                tasks: vec![task.into()],
            };
            if let Some(values) = targets {
                batch.values.insert(
                    "action".into(),
                    Data::f32(values.clone(), vec![values.len() / 2, 2]),
                );
            }
            batch
        }
        fn write(&self, name: &str, value: &Value) {
            std::fs::write(self.0.join(name), serde_json::to_vec(value).unwrap()).unwrap();
        }
    }

    #[test]
    fn pipeline_loads_local_tokenizer_normalizes_then_pads() {
        let (fixture, cfg) = Fixture::new(6);
        let (pre, post) = import_pipelines(&fixture.0, &cfg, &fixture.0).unwrap();
        let mut batch = pre.run(fixture.batch("a b", None), &context()).unwrap();
        let tokens = match batch.take("observation.language.tokens").unwrap() {
            Data::Int2(tokens) => tokens.into_data().iter::<i64>().collect::<Vec<_>>(),
            other => panic!("expected integer tokens, got {other:?}"),
        };
        let mask = match batch.take("observation.language.attention_mask").unwrap() {
            Data::Bool2(mask) => mask.into_data().try_to_vec::<bool>().unwrap(),
            other => panic!("expected boolean mask, got {other:?}"),
        };
        assert_eq!(tokens, [1, 2, 7, 7]);
        assert_eq!(mask, [true, true, false, false]);
        let mut truncated = pre
            .run(fixture.batch("a b c d e", None), &context())
            .unwrap();
        let Data::Int2(tokens) = truncated.take("observation.language.tokens").unwrap() else {
            panic!("expected integer tokens");
        };
        assert_eq!(
            tokens.into_data().iter::<i64>().collect::<Vec<_>>(),
            [2, 3, 4, 5]
        );
        let state = batch
            .take("observation.state")
            .unwrap()
            .into_float2()
            .unwrap()
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        assert_eq!(&state[..6], &[1.0; 6]);
        assert_eq!(&state[6..], &[0.0; 26]);
        assert!(
            !batch.values.contains_key("action"),
            "inference must not require training targets"
        );

        let targets = [12.0, 28.0].repeat(cfg.chunk_size);
        let mut train_context = context();
        train_context.mode = Mode::Train;
        let mut training = pre
            .run(fixture.batch("a", Some(targets)), &train_context)
            .unwrap();
        let action = training.take("action").unwrap().into_float3().unwrap();
        assert_eq!(action.dims(), [1, cfg.chunk_size, cfg.max_action_dim]);
        for row in action
            .into_data()
            .try_to_vec::<f32>()
            .unwrap()
            .chunks_exact(cfg.max_action_dim)
        {
            assert_eq!(&row[..2], &[1.0, 2.0]);
            assert!(row[2..].iter().all(|&value| value == 0.0));
        }
        let mut padded = vec![999.0; cfg.chunk_size * cfg.max_action_dim];
        for (index, row) in padded.chunks_exact_mut(cfg.max_action_dim).enumerate() {
            let sign = if index % 2 == 0 { 1.0 } else { -1.0 };
            row[..2].copy_from_slice(&[sign, 2.0 * sign]);
        }
        let output = Batch {
            values: BTreeMap::from([(
                "action".into(),
                Data::Float3(Tensor::from_data(
                    TensorData::new(padded, [1, cfg.chunk_size, cfg.max_action_dim]),
                    &context().device,
                )),
            )]),
            ..Batch::default()
        };
        let action = post
            .run(output, &context())
            .unwrap()
            .take("action")
            .unwrap()
            .into_float3()
            .unwrap();
        assert_eq!(action.dims(), [1, cfg.chunk_size, 2]);
        assert_eq!(
            &action.into_data().try_to_vec::<f32>().unwrap()[..4],
            &[12.0, 28.0, 8.0, 12.0]
        );
    }

    #[test]
    fn state_six_stats_eight_is_rejected_during_load() {
        let (fixture, cfg) = Fixture::new(8);
        let error = import_pipelines(&fixture.0, &cfg, &fixture.0).unwrap_err();
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
            import_pipelines(&fixture.0, &cfg, &fixture.0)
                .unwrap_err()
                .to_string()
                .contains("rename_map")
        );
        value["steps"][2]["registry_name"] = "custom_processor".into();
        fixture.write("policy_preprocessor.json", &value);
        assert!(
            import_pipelines(&fixture.0, &cfg, &fixture.0)
                .unwrap_err()
                .to_string()
                .contains("custom_processor")
        );
    }

    #[test]
    fn range_normalization_and_inverse_keep_canonical_midpoints() {
        let (fixture, mut cfg) = Fixture::with_std(6, 5.0, [14.0, 28.0]);
        for kind in ["STATE", "ACTION"] {
            cfg.normalization_mapping
                .insert(kind.into(), "MIN_MAX".into());
        }
        for (filename, index) in [
            ("policy_preprocessor.json", 5),
            ("policy_postprocessor.json", 0),
        ] {
            let mut value: Value =
                serde_json::from_slice(&std::fs::read(fixture.0.join(filename)).unwrap()).unwrap();
            for kind in ["STATE", "ACTION"] {
                value["steps"][index]["config"]["norm_map"][kind] = "MIN_MAX".into();
            }
            fixture.write(filename, &value);
        }
        let stats = std::fs::read(fixture.0.join("stats.safetensors")).unwrap();
        let tensors = SafeTensors::deserialize(&stats).unwrap();
        let ranges = tensors.tensors().into_iter().map(|(name, tensor)| {
            (
                name.replace(".mean", ".min").replace(".std", ".max"),
                tensor,
            )
        });
        std::fs::write(
            fixture.0.join("stats.safetensors"),
            safetensors::tensor::serialize(ranges, None).unwrap(),
        )
        .unwrap();
        let (pre, post) = import_pipelines(&fixture.0, &cfg, &fixture.0).unwrap();
        let mut prepared = pre
            .run(
                fixture.batch("a", Some([12.0, 24.0].repeat(cfg.chunk_size))),
                &context(),
            )
            .unwrap();
        let state = prepared
            .take("observation.state")
            .unwrap()
            .into_float2()
            .unwrap()
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        assert_eq!(state, vec![0.0; cfg.max_state_dim]);
        let action = prepared.take("action").unwrap().into_float3().unwrap();
        assert!(
            action
                .clone()
                .into_data()
                .try_to_vec::<f32>()
                .unwrap()
                .iter()
                .all(|&value| value == 0.0)
        );
        let output = Batch {
            values: BTreeMap::from([("action".into(), Data::Float3(action))]),
            ..Batch::default()
        };
        let canonical = post
            .run(output, &context())
            .unwrap()
            .take("action")
            .unwrap()
            .into_float3()
            .unwrap()
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        assert_eq!(canonical, [12.0, 24.0].repeat(cfg.chunk_size));
    }

    #[test]
    fn normalization_matches_official_epsilon_and_inverse() {
        let (fixture, cfg) = Fixture::with_std(6, 0.0, [0.0, 0.0]);
        let path = fixture.0.join("policy_preprocessor.json");
        let mut value: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        value["steps"][5]["config"]["eps"] = 0.25.into();
        fixture.write("policy_preprocessor.json", &value);
        let (pre, post) = import_pipelines(&fixture.0, &cfg, &fixture.0).unwrap();
        let mut batch = pre
            .run(
                fixture.batch("a", Some([11.0, 21.0].repeat(cfg.chunk_size))),
                &context(),
            )
            .unwrap();
        let state = batch
            .take("observation.state")
            .unwrap()
            .into_float2()
            .unwrap()
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        assert_eq!(&state[..6], &[8.0; 6]);
        let action = batch.take("action").unwrap().into_float3().unwrap();
        assert_eq!(
            &action.clone().into_data().try_to_vec::<f32>().unwrap()[..2],
            &[4.0, 4.0]
        );
        let output = Batch {
            values: BTreeMap::from([("action".into(), Data::Float3(action))]),
            ..Batch::default()
        };
        let canonical = post
            .run(output, &context())
            .unwrap()
            .take("action")
            .unwrap()
            .into_float3()
            .unwrap()
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        assert_eq!(canonical, [10.0, 20.0].repeat(cfg.chunk_size));
    }
}
