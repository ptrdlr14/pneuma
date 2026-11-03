//! 将 π₀.₅ checkpoint 的 processor 严格导入通用 pipeline。

use std::{
    collections::{BTreeMap, HashMap},
    path::{Component, Path},
};

use anyhow::{Context, Result, bail, ensure};
use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;
use serde_json::Value;

use super::config::{Pi05Config, PolicyFeature};
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
    #[serde(rename = "tokenizer_name")]
    _tokenizer_name: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RelativeConfig {
    enabled: bool,
    #[serde(default)]
    exclude_joints: Vec<String>,
    #[serde(default)]
    action_names: Option<Vec<String>>,
}

#[derive(Clone, Debug)]
enum Normalization {
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
            Self::MeanStd { mean, std, eps } if !inverse => vec![affine(
                mean.clone(),
                std.iter().map(|value| value + eps).collect(),
            )],
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
    cfg: &Pi05Config,
    tokenizer_dir: &Path,
) -> Result<(Pipeline, Pipeline)> {
    let pre = load_pipeline(
        &dir.join("policy_preprocessor.json"),
        &[
            "rename_observations_processor",
            "to_batch_processor",
            "normalizer_processor",
            "pi05_prepare_state_tokenizer_processor_step",
            "tokenizer_processor",
            "device_processor",
        ],
        &["relative_actions_processor"],
    )?;
    let post = load_pipeline(
        &dir.join("policy_postprocessor.json"),
        &["unnormalizer_processor", "device_processor"],
        &["absolute_actions_processor"],
    )?;
    validate_steps(&pre, &post)?;
    let pre_norms = load_normalizers(dir, &pre.steps[2], cfg, false)?;
    let post_norms = load_normalizers(dir, &post.steps[0], cfg, true)?;
    let token_config: TokenizerConfig = serde_json::from_value(pre.steps[4].config.clone())
        .context("parsing tokenizer_processor config")?;
    ensure!(
        token_config.task_key == "task",
        "unsupported tokenizer task_key"
    );
    ensure!(
        token_config.padding_side == "right",
        "tokenizer requires right padding"
    );
    ensure!(
        token_config.padding == "max_length",
        "tokenizer requires max_length padding"
    );
    ensure!(
        token_config.truncation,
        "tokenizer requires truncation=true"
    );
    ensure!(
        token_config.max_length == cfg.tokenizer_max_length,
        "tokenizer processor disagrees with tokenizer_max_length in config.json"
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
        steps.push(Step::ResizePad {
            key: name.into(),
            width: cfg.image_resolution[1],
            height: cfg.image_resolution[0],
        });
        // PaliGemma 期望 [-1, 1]。
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
    steps.extend(
        pre_norms
            .get("observation.state")
            .with_context(|| missing_stats(dir))?
            .steps("observation.state", 1, false),
    );
    steps.extend(
        pre_norms
            .get("action")
            .map(|norm| norm.steps("action", 2, false))
            .unwrap_or_default(),
    );
    steps.extend([
        Step::Prompt {
            state_key: "observation.state".into(),
            bins: 256,
        },
        Step::Tokenize {
            tokenizer: tokenizer_dir.to_path_buf(),
            max_length: token_config.max_length,
            padding: token_config.padding,
            padding_side: token_config.padding_side,
            special_tokens: true,
            negative: None,
        },
        Step::Device,
    ]);
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
            .with_context(|| missing_stats(dir))?
            .steps("action", 2, true),
    );
    let output = Pipeline::new(output_schema, output_steps)?;
    Ok((input, output))
}

fn missing_stats(dir: &Path) -> String {
    format!(
        "{} 缺少归一化统计；该 checkpoint 的 processor JSON 没有 state_file，\
         请改用带统计的 checkpoint，或提供 --stats",
        dir.display()
    )
}

fn load_pipeline(path: &Path, expected: &[&str], optional: &[&str]) -> Result<ImportedPipeline> {
    let bytes =
        std::fs::read(path).with_context(|| format!("reading processor {}", path.display()))?;
    let pipeline: ImportedPipeline = serde_json::from_slice(&bytes)
        .with_context(|| format!("parsing processor {}", path.display()))?;
    let names: Vec<_> = pipeline
        .steps
        .iter()
        .map(|step| step.registry_name.as_str())
        .filter(|name| !optional.contains(name))
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
    ensure!(
        pre.steps[1]
            .config
            .as_object()
            .is_some_and(|map| map.is_empty()),
        "unsupported to_batch options"
    );
    for step in pre.steps.iter().chain(&post.steps) {
        if step.registry_name.ends_with("actions_processor") {
            let relative: RelativeConfig =
                serde_json::from_value(step.config.clone()).context("parsing actions processor")?;
            ensure!(
                !relative.enabled
                    && relative.exclude_joints.is_empty()
                    && relative.action_names.is_none(),
                "relative/absolute action conversion is not supported"
            );
        }
    }
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct DeviceConfig {
        device: Option<String>,
        float_dtype: Option<String>,
    }
    for step in pre.steps.iter().chain(&post.steps) {
        if step.registry_name != "device_processor" {
            continue;
        }
        let config: DeviceConfig =
            serde_json::from_value(step.config.clone()).context("parsing device_processor")?;
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
    cfg: &Pi05Config,
    inverse: bool,
) -> Result<HashMap<String, Normalization>> {
    let config: NormalizerConfig = serde_json::from_value(step.config.clone())
        .with_context(|| format!("parsing {} config", step.registry_name))?;
    ensure!(
        config.eps.is_finite() && config.eps > 0.0,
        "normalizer eps must be finite and positive"
    );
    let expected: HashMap<String, PolicyFeature> = if inverse {
        cfg.output_features.iter().cloned().collect()
    } else {
        cfg.input_features
            .iter()
            .chain(cfg.output_features.iter())
            .cloned()
            .collect()
    };
    ensure!(
        config.features == expected,
        "{} features disagree with config.json: checkpoint declares {:?}",
        step.registry_name,
        config.features.keys().collect::<Vec<_>>()
    );
    let stats = if let Some(file) = &step.state_file {
        let relative = Path::new(file);
        ensure!(
            matches!(relative.components().next(), Some(Component::Normal(_)))
                && relative.components().count() == 1,
            "state_file must be a local filename: {file}"
        );
        Some(
            std::fs::read(dir.join(relative))
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
        if mode == "IDENTITY" {
            continue;
        }
        ensure!(
            feature.feature_type == "STATE" || feature.feature_type == "ACTION",
            "normalization of {name} ({}) is not supported",
            feature.feature_type
        );
        let read = |stat: &str| -> Result<Vec<f32>> {
            let key = format!("{name}.{stat}");
            let tensor = tensors
                .as_ref()
                .context("normalization requires state_file")?
                .tensor(&key)
                .with_context(|| format!("missing normalization statistic {key}"))?;
            ensure!(
                tensor.dtype() == Dtype::F32,
                "normalization statistic {key} must be F32"
            );
            ensure!(
                tensor.shape() == feature.shape,
                "normalization statistic {key}: shape {:?} != {:?}; refusing to truncate",
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
                "non-finite statistic {key}"
            );
            Ok(values)
        };
        let norm = match mode {
            "MEAN_STD" => {
                let mean = read("mean")?;
                let std = read("std")?;
                ensure!(std.iter().all(|&v| v >= 0.0), "negative std for {name}");
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
                    "invalid range for {name}"
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
    cfg: &Pi05Config,
    features: &[FeatureSpec],
    action: &FeatureSpec,
) -> Result<()> {
    let declared: Vec<_> = cfg
        .input_features
        .iter()
        .filter(|(name, _)| !super::config::is_empty_camera(name))
        .collect();
    ensure!(
        features.len() == declared.len(),
        "pi05 feature count disagrees with checkpoint"
    );
    for (name, expected) in declared {
        let mut matches = features.iter().filter(|feature| &feature.name == name);
        let feature = matches
            .next()
            .with_context(|| format!("missing pi05 feature {name}"))?;
        ensure!(matches.next().is_none(), "duplicate pi05 feature {name}");
        match expected.feature_type.as_str() {
            "STATE" => ensure!(
                feature.kind == "tensor"
                    && feature.dtype == "float32"
                    && feature.shape == [expected.shape[0] as u64],
                "pi05 requires {name} float32 {:?}",
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
                    "pi05 requires {name} uint8 HWC [{h},{w},3]"
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
        "pi05 action contract disagrees with checkpoint dimension {}",
        cfg.action_dim()
    );
    Ok(())
}
