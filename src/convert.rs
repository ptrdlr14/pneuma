//! LeRobot 与原生 checkpoint 的转换；未执行的源参数作为独立 bpk 保留。

use std::{
    borrow::Cow,
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context as _, Result, bail, ensure};
use burn_pack::{Bytes, DType, Reader, Tensor, Writer};
use safetensors::{
    Dtype, SafeTensors,
    tensor::{View, serialize_to_file},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    checkpoint::{self, Checkpoint, CheckpointParts},
    config::Device,
    contract::FeatureSpec,
    policy::{
        PolicyKind, act::ActPolicy, cosmos3::Cosmos3Policy, pi05::Pi05Policy,
        smolvla::SmolVlaPolicy,
    },
    processor::Pipeline,
};

/// 转换边界使用整套 checkpoint 格式，而非单个文件扩展名。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Lerobot,
    Pneuma,
}

/// 输入输出均为目录；输出已存在时拒绝覆盖。
#[derive(Debug)]
pub struct ConvertOptions {
    pub input: PathBuf,
    pub output: PathBuf,
    pub from: Format,
    pub to: Format,
    pub vlm_path: Option<PathBuf>,
    /// checkpoint 没有特征描述时使用的外部合约。
    pub features: Option<PathBuf>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TensorSpec {
    shape: Vec<usize>,
    dtype: String,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Interop {
    version: u32,
    tensors: BTreeMap<String, TensorSpec>,
    active: BTreeSet<String>,
    active_dtype: String,
    dtype_changes: BTreeMap<String, String>,
    pipelines_sha256: String,
    unused_parameters_trainable: bool,
}

/// 转换完整模型与配套资源；不转换优化器或训练恢复状态。
///
/// # Errors
/// 不支持的方向、分片输入、损坏模型、参数覆盖不完整或输出已存在时失败。
pub fn run(options: &ConvertOptions) -> Result<()> {
    ensure!(
        options.from != options.to,
        "conversion requires different formats"
    );
    checkpoint::ensure_absent(&options.output)?;
    match (options.from, options.to) {
        (Format::Lerobot, Format::Pneuma) => import(options),
        (Format::Pneuma, Format::Lerobot) => export(options),
        _ => bail!("unsupported conversion direction"),
    }
}

fn import(options: &ConvertOptions) -> Result<()> {
    ensure!(
        !options.input.join("model.safetensors.index.json").exists(),
        "sharded safetensors conversion is not implemented; provide a single model.safetensors"
    );
    let config_path = options.input.join("config.json");
    let config: serde_json::Value = serde_json::from_slice(&fs::read(&config_path)?)?;
    let kind = config["type"]
        .as_str()
        .context("checkpoint config requires type")?
        .parse::<PolicyKind>()?;
    let vlm = if matches!(kind, PolicyKind::Smolvla | PolicyKind::Pi05) {
        Some(
            options
                .vlm_path
                .clone()
                .unwrap_or_else(|| options.input.join("vlm")),
        )
    } else {
        ensure!(
            options.vlm_path.is_none(),
            "vlm-path is only valid for SmolVLA and pi05"
        );
        None
    };
    match kind {
        PolicyKind::Act => {
            let (model, pre, post, _) = ActPolicy::import_configured(&options.input, Device::Flex)?;
            save_import(
                options,
                kind,
                vlm.as_deref(),
                &pre,
                &post,
                |path| model.save_native(path),
                |path| model.export_lerobot(path),
            )
        }
        PolicyKind::Smolvla => {
            let (features, action) = feature_contract(&config, options.features.as_deref())?;
            let (model, pre, post, _) = SmolVlaPolicy::import_configured(
                &options.input,
                vlm.as_deref(),
                Device::Flex,
                &features,
                &action,
            )?;
            save_import(
                options,
                kind,
                vlm.as_deref(),
                &pre,
                &post,
                |path| model.save_native(path),
                |path| model.export_lerobot(path),
            )
        }
        PolicyKind::Pi05 => {
            // 补零相机由 checkpoint 的 empty_cameras 表达，客户端不发送，也不进入部署合约。
            let (features, action) = feature_contract(&config, options.features.as_deref())?;
            let features: Vec<FeatureSpec> = features
                .into_iter()
                .filter(|feature| !feature.name.starts_with("observation.images.empty_camera_"))
                .collect();
            let (model, pre, post, _) = Pi05Policy::import_configured(
                &options.input,
                vlm.as_deref(),
                Device::Flex,
                &features,
                &action,
            )?;
            save_import(
                options,
                kind,
                vlm.as_deref(),
                &pre,
                &post,
                |path| model.save_native(path),
                |path| model.export_lerobot(path),
            )
        }
        PolicyKind::Cosmos3Edge => {
            let (features, action) = feature_contract(&config, options.features.as_deref())?;
            let tokenizer = options.input.join("tokenizer");
            let (model, pre, post, _) =
                Cosmos3Policy::import_configured(&options.input, Device::Flex, &features, &action)?;
            save_import(
                options,
                kind,
                Some(&tokenizer),
                &pre,
                &post,
                |path| model.save_native(path),
                |path| model.export_lerobot(path),
            )
        }
        PolicyKind::Zero => bail!("unsupported conversion policy {kind}"),
    }
}

fn save_import(
    options: &ConvertOptions,
    kind: PolicyKind,
    vlm: Option<&Path>,
    pre: &Pipeline,
    post: &Pipeline,
    save: impl FnOnce(&Path) -> Result<()>,
    export_active: impl FnOnce(&Path) -> Result<()>,
) -> Result<()> {
    let parts = CheckpointParts {
        policy_kind: kind,
        config: &options.input.join("config.json"),
        vlm,
        preprocessor: pre,
        postprocessor: post,
    };
    Checkpoint::save(&options.output, &parts, |root, weights| {
        save(weights)?;
        let active_path = root.join("active.safetensors");
        export_active(&active_path)?;
        let original = fs::read(options.input.join("model.safetensors"))?;
        let source = SafeTensors::deserialize(&original)?;
        let active_bytes = fs::read(&active_path)?;
        let active = SafeTensors::deserialize(&active_bytes)?;
        let inventory = inventory(&source);
        let active_names: BTreeSet<String> =
            active.names().into_iter().map(str::to_owned).collect();
        ensure!(
            !active_names.is_empty(),
            "model export contains no active parameters"
        );
        let mut dtype_changes = BTreeMap::new();
        for name in &active_names {
            let input = source.tensor(name).with_context(|| {
                format!("active export introduced unknown source parameter {name}")
            })?;
            let output = active.tensor(name)?;
            if input.dtype() != output.dtype() {
                dtype_changes.insert(
                    name.clone(),
                    format!("{:?} -> {:?}", input.dtype(), output.dtype()),
                );
            }
            ensure!(
                input.shape() == output.shape(),
                "export changed source parameter shape {name}"
            );
        }
        let directory = root.join("interop");
        fs::create_dir(&directory)?;
        write_extras(&source, &active_names, &directory.join("extra.bpk"))?;
        fs::remove_file(active_path)?;
        copy_interop(&options.input, &directory)?;
        let provenance = Interop {
            version: 1,
            tensors: inventory,
            active: active_names,
            active_dtype: "F32".into(),
            dtype_changes,
            pipelines_sha256: String::new(),
            unused_parameters_trainable: false,
        };
        fs::write(
            directory.join("conversion.json"),
            serde_json::to_vec_pretty(&provenance)?,
        )?;
        // Pipeline::save 尚未执行；在临时独立目录导出同一份定义，记录可逆导出的边界。
        let pipeline_root = root.join("pipeline-source");
        fs::create_dir(&pipeline_root)?;
        pre.save(&pipeline_root.join("preprocessor"))?;
        post.save(&pipeline_root.join("postprocessor"))?;
        let mut provenance = provenance;
        provenance.pipelines_sha256 = pipeline_digest(&pipeline_root)?;
        fs::write(
            directory.join("conversion.json"),
            serde_json::to_vec_pretty(&provenance)?,
        )?;
        fs::remove_dir_all(pipeline_root)?;
        Ok(())
    })
}

fn export(options: &ConvertOptions) -> Result<()> {
    ensure!(
        options.vlm_path.is_none(),
        "native checkpoint owns its VLM resources"
    );
    let native = Checkpoint::load(&options.input)?;
    let interop = native
        .interop_dir
        .as_deref()
        .context("checkpoint has no LeRobot conversion metadata")?;
    let provenance: Interop = serde_json::from_slice(&fs::read(interop.join("conversion.json"))?)?;
    ensure!(
        provenance.version == 1 && !provenance.unused_parameters_trainable,
        "unsupported conversion metadata"
    );
    ensure!(
        pipeline_digest(&options.input)? == provenance.pipelines_sha256,
        "pipeline changed after import; exporting modified pipelines to LeRobot is not implemented"
    );
    checkpoint::staging(&options.output, |root| {
        let active = root.join("active.safetensors");
        match native.policy_kind {
            PolicyKind::Act => {
                ActPolicy::load_model(&native.model_dir, Device::Flex)?.export_lerobot(&active)?
            }
            PolicyKind::Smolvla => SmolVlaPolicy::load_model(
                &native.model_dir,
                native.vlm_path.as_deref(),
                Device::Flex,
            )?
            .export_lerobot(&active)?,
            PolicyKind::Pi05 => {
                Pi05Policy::load_model(&native.model_dir, Device::Flex)?.export_lerobot(&active)?
            }
            _ => bail!("unsupported native policy"),
        }
        merge_weights(
            &active,
            &interop.join("extra.bpk"),
            &root.join("model.safetensors"),
            &provenance,
        )?;
        fs::remove_file(active)?;
        for entry in fs::read_dir(interop)? {
            let entry = entry?;
            let name = entry.file_name();
            if matches!(name.to_str(), Some("extra.bpk" | "conversion.json")) {
                continue;
            }
            ensure!(entry.file_type()?.is_file(), "unexpected interop directory");
            fs::copy(entry.path(), root.join(name))?;
        }
        if let Some(vlm) = &native.vlm_path {
            fs::create_dir(root.join("vlm"))?;
            for entry in fs::read_dir(vlm)? {
                let entry = entry?;
                ensure!(
                    entry.file_type()?.is_file(),
                    "unexpected VLM resource directory"
                );
                fs::copy(entry.path(), root.join("vlm").join(entry.file_name()))?;
            }
            // tokenizer/VLM 绑定到最终输出目录；不保留导入机器的绝对路径。
            let final_vlm = absolute_output(&options.output)?.join("vlm");
            bind_vlm(root, &final_vlm)?;
        }
        Ok(())
    })
}

fn inventory(tensors: &SafeTensors<'_>) -> BTreeMap<String, TensorSpec> {
    tensors
        .tensors()
        .into_iter()
        .map(|(name, t)| {
            (
                name,
                TensorSpec {
                    shape: t.shape().to_vec(),
                    dtype: format!("{:?}", t.dtype()),
                },
            )
        })
        .collect()
}

fn write_extras(source: &SafeTensors<'_>, active: &BTreeSet<String>, path: &Path) -> Result<()> {
    let mut extras = Vec::new();
    for (name, view) in source.tensors() {
        if !active.contains(&name) {
            extras.push(Tensor::new(
                name,
                to_burn_dtype(view.dtype())?,
                view.shape().to_vec(),
                None,
                Bytes::from_bytes_vec(view.data().to_vec()),
            ));
        }
    }
    Writer::new(extras).write_to_file(path)?;
    Ok(())
}

struct OwnedTensor {
    dtype: Dtype,
    shape: Vec<usize>,
    bytes: Bytes,
}
impl View for OwnedTensor {
    fn dtype(&self) -> Dtype {
        self.dtype
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn data(&self) -> Cow<'_, [u8]> {
        Cow::Borrowed(&self.bytes)
    }
    fn data_len(&self) -> usize {
        self.bytes.len()
    }
}

fn merge_weights(
    active_path: &Path,
    extras_path: &Path,
    output: &Path,
    provenance: &Interop,
) -> Result<()> {
    let data = fs::read(active_path)?;
    let active = SafeTensors::deserialize(&data)?;
    let mut tensors = BTreeMap::new();
    let actual_active: BTreeSet<_> = active.names().into_iter().map(str::to_owned).collect();
    ensure!(
        actual_active == provenance.active,
        "native model active parameters changed after import"
    );
    for (name, tensor) in active.tensors() {
        let expected = provenance
            .tensors
            .get(&name)
            .context("unexpected active key")?;
        ensure!(
            tensor.shape() == expected.shape,
            "active parameter shape changed: {name}"
        );
        tensors.insert(
            name,
            OwnedTensor {
                dtype: tensor.dtype(),
                shape: tensor.shape().to_vec(),
                bytes: Bytes::from_bytes_vec(tensor.data().to_vec()),
            },
        );
    }
    for tensor in Reader::from_file_exact(extras_path)?.into_tensors()? {
        let (name, dtype, shape, _, bytes) = tensor.into_parts()?;
        ensure!(
            !tensors.contains_key(&name),
            "duplicate active/extra parameter {name}"
        );
        let expected = provenance
            .tensors
            .get(&name)
            .context("unexpected extra key")?;
        let dtype = to_safetensors_dtype(dtype)?;
        let shape = shape.to_vec();
        ensure!(
            shape == expected.shape && format!("{dtype:?}") == expected.dtype,
            "extra parameter representation changed: {name}"
        );
        tensors.insert(
            name,
            OwnedTensor {
                dtype,
                shape,
                bytes,
            },
        );
    }
    ensure!(
        tensors.keys().eq(provenance.tensors.keys()),
        "converted checkpoint parameter coverage is incomplete"
    );
    serialize_to_file(tensors, None, output)?;
    Ok(())
}

fn copy_interop(source: &Path, destination: &Path) -> Result<()> {
    fs::copy(source.join("config.json"), destination.join("config.json"))?;
    for name in ["policy_preprocessor.json", "policy_postprocessor.json"] {
        if source.join(name).is_file() {
            let bytes = fs::read(source.join(name))?;
            let value: serde_json::Value = serde_json::from_slice(&bytes)?;
            for step in value["steps"]
                .as_array()
                .context("processor requires steps")?
            {
                if let Some(file) = step.get("state_file").and_then(serde_json::Value::as_str) {
                    let relative = Path::new(file);
                    ensure!(
                        matches!(
                            relative.components().next(),
                            Some(std::path::Component::Normal(_))
                        ) && relative.components().count() == 1,
                        "processor state_file must be a local filename"
                    );
                    // 导入阶段允许显式源 checkpoint 的统计链接；复制后产物自包含。
                    fs::copy(source.join(relative), destination.join(relative))?;
                }
            }
            fs::write(destination.join(name), bytes)?;
        }
    }
    Ok(())
}

fn feature_contract(
    config: &serde_json::Value,
    external: Option<&Path>,
) -> Result<(Vec<FeatureSpec>, FeatureSpec)> {
    let owned;
    let config = match external {
        Some(path) => {
            owned = serde_json::from_slice::<serde_json::Value>(&fs::read(path)?)
                .with_context(|| format!("reading features from {}", path.display()))?;
            &owned
        }
        None => config,
    };
    let input = config["input_features"]
        .as_object()
        .context("input_features must be an object")?;
    let mut features = Vec::new();
    for (name, feature) in input {
        let visual = feature["type"].as_str() == Some("VISUAL");
        let shape: Vec<u64> = serde_json::from_value(feature["shape"].clone())?;
        let shape = if visual {
            ensure!(shape.len() == 3, "visual shape must be CHW");
            vec![shape[1], shape[2], shape[0]]
        } else {
            shape
        };
        features.push(FeatureSpec {
            name: name.clone(),
            shape,
            dtype: if visual { "uint8" } else { "float32" }.into(),
            kind: if visual { "rgb" } else { "tensor" }.into(),
            ..FeatureSpec::default()
        });
    }
    let shape = serde_json::from_value(config["output_features"]["action"]["shape"].clone())?;
    Ok((
        features,
        FeatureSpec {
            name: "action".into(),
            shape,
            dtype: "float32".into(),
            ..FeatureSpec::default()
        },
    ))
}

fn pipeline_digest(root: &Path) -> Result<String> {
    let mut files = BTreeMap::new();
    for role in ["preprocessor", "postprocessor"] {
        let mut pending = vec![root.join(role)];
        while let Some(directory) = pending.pop() {
            for entry in fs::read_dir(directory)? {
                let entry = entry?;
                if entry.file_type()?.is_dir() {
                    pending.push(entry.path());
                } else {
                    files.insert(
                        entry.path().strip_prefix(root)?.to_path_buf(),
                        fs::read(entry.path())?,
                    );
                }
            }
        }
    }
    let mut digest = Sha256::new();
    for (name, bytes) in files {
        let name = name.to_string_lossy();
        digest.update((name.len() as u64).to_le_bytes());
        digest.update(name.as_bytes());
        digest.update((bytes.len() as u64).to_le_bytes());
        digest.update(bytes);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn absolute_output(path: &Path) -> Result<PathBuf> {
    Ok(if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()?.join(path)
    })
}

fn bind_vlm(root: &Path, vlm: &Path) -> Result<()> {
    let config_path = root.join("config.json");
    let mut config: serde_json::Value = serde_json::from_slice(&fs::read(&config_path)?)?;
    config["vlm_model_name"] = serde_json::to_value(vlm)?;
    config["load_vlm_weights"] = false.into();
    fs::write(config_path, serde_json::to_vec_pretty(&config)?)?;
    let pre_path = root.join("policy_preprocessor.json");
    if pre_path.is_file() {
        let mut pre: serde_json::Value = serde_json::from_slice(&fs::read(&pre_path)?)?;
        for step in pre["steps"]
            .as_array_mut()
            .context("processor requires steps")?
        {
            if step["registry_name"].as_str() == Some("tokenizer_processor") {
                step["config"]["tokenizer_name"] = serde_json::to_value(vlm)?;
            }
        }
        fs::write(pre_path, serde_json::to_vec_pretty(&pre)?)?;
    }
    Ok(())
}

fn to_burn_dtype(dtype: Dtype) -> Result<DType> {
    Ok(match dtype {
        Dtype::F64 => DType::F64,
        Dtype::F32 => DType::F32,
        Dtype::F16 => DType::F16,
        Dtype::BF16 => DType::BF16,
        Dtype::I64 => DType::I64,
        Dtype::I32 => DType::I32,
        Dtype::I16 => DType::I16,
        Dtype::I8 => DType::I8,
        Dtype::U64 => DType::U64,
        Dtype::U32 => DType::U32,
        Dtype::U16 => DType::U16,
        Dtype::U8 => DType::U8,
        Dtype::BOOL => DType::Bool(burn::tensor::BoolStore::Native),
        other => bail!("unsupported extra parameter dtype {other:?}"),
    })
}

fn to_safetensors_dtype(dtype: DType) -> Result<Dtype> {
    Ok(match dtype {
        DType::F64 => Dtype::F64,
        DType::F32 => Dtype::F32,
        DType::F16 => Dtype::F16,
        DType::BF16 => Dtype::BF16,
        DType::I64 => Dtype::I64,
        DType::I32 => Dtype::I32,
        DType::I16 => Dtype::I16,
        DType::I8 => Dtype::I8,
        DType::U64 => Dtype::U64,
        DType::U32 => Dtype::U32,
        DType::U16 => Dtype::U16,
        DType::U8 => Dtype::U8,
        DType::Bool(burn::tensor::BoolStore::Native) => Dtype::BOOL,
        other => bail!("unsupported extra parameter dtype {other:?}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Directory(PathBuf);
    impl Directory {
        fn new() -> Self {
            let path =
                std::env::temp_dir().join(format!("pneuma-convert-{}", uuid::Uuid::new_v4()));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for Directory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn extras_roundtrip_preserves_unused_dtype_and_active_current_values() {
        use safetensors::tensor::TensorView;
        let directory = Directory::new();
        let weight = [1f32, 2.]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let extra = 7i64.to_le_bytes();
        let source = safetensors::serialize(
            [
                (
                    "weight",
                    TensorView::new(Dtype::F32, vec![2], &weight).unwrap(),
                ),
                ("vae", TensorView::new(Dtype::I64, vec![], &extra).unwrap()),
            ],
            None,
        )
        .unwrap();
        let source = SafeTensors::deserialize(&source).unwrap();
        let active_names = BTreeSet::from(["weight".into()]);
        let extra_path = directory.0.join("extra.bpk");
        write_extras(&source, &active_names, &extra_path).unwrap();
        let current = [3f32, 4.]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let active = directory.0.join("active.safetensors");
        serialize_to_file(
            [(
                "weight",
                TensorView::new(Dtype::F32, vec![2], &current).unwrap(),
            )],
            None,
            &active,
        )
        .unwrap();
        let provenance = Interop {
            version: 1,
            tensors: inventory(&source),
            active: active_names,
            active_dtype: "F32".into(),
            dtype_changes: BTreeMap::new(),
            pipelines_sha256: String::new(),
            unused_parameters_trainable: false,
        };
        let output = directory.0.join("model.safetensors");
        merge_weights(&active, &extra_path, &output, &provenance).unwrap();
        let bytes = fs::read(output).unwrap();
        let merged = SafeTensors::deserialize(&bytes).unwrap();
        assert_eq!(merged.tensor("weight").unwrap().data(), current);
        assert_eq!(merged.tensor("vae").unwrap().data(), extra);
        assert_eq!(merged.tensor("vae").unwrap().dtype(), Dtype::I64);
        let bad = directory.0.join("bad.bpk");
        Writer::new(vec![Tensor::new(
            "weight".into(),
            DType::F32,
            vec![2],
            None,
            Bytes::from_bytes_vec(weight),
        )])
        .write_to_file(&bad)
        .unwrap();
        assert!(
            merge_weights(
                &active,
                &bad,
                &directory.0.join("bad.safetensors"),
                &provenance
            )
            .is_err()
        );
    }

    #[cfg(unix)]
    #[test]
    fn imported_statistics_symlink_becomes_an_independent_file() {
        let directory = Directory::new();
        let source = directory.0.join("source");
        let destination = directory.0.join("interop");
        fs::create_dir(&source).unwrap();
        fs::create_dir(&destination).unwrap();
        fs::write(source.join("config.json"), br#"{"type":"smolvla"}"#).unwrap();
        let external = directory.0.join("original-statistics.safetensors");
        let payload = [0u8, 1, 2, 3];
        fs::write(&external, payload).unwrap();
        std::os::unix::fs::symlink(&external, source.join("statistics.safetensors")).unwrap();
        fs::write(
            source.join("policy_preprocessor.json"),
            br#"{"steps":[{"registry_name":"normalizer_processor","state_file":"statistics.safetensors","config":{}}]}"#,
        ).unwrap();
        copy_interop(&source, &destination).unwrap();
        let saved = destination.join("statistics.safetensors");
        assert!(
            !fs::symlink_metadata(&saved)
                .unwrap()
                .file_type()
                .is_symlink()
        );
        fs::remove_file(external).unwrap();
        fs::remove_dir_all(source).unwrap();
        assert_eq!(fs::read(saved).unwrap(), payload);
        let processor: serde_json::Value = serde_json::from_slice(
            &fs::read(destination.join("policy_preprocessor.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(
            processor["steps"][0]["state_file"],
            "statistics.safetensors"
        );
    }

    #[test]
    fn conversion_rejects_same_format_and_sharded_input() {
        let directory = Directory::new();
        let mut options = ConvertOptions {
            input: directory.0.clone(),
            output: directory.0.join("output"),
            from: Format::Lerobot,
            to: Format::Lerobot,
            vlm_path: None,
            features: None,
        };
        assert!(run(&options).is_err());
        options.to = Format::Pneuma;
        fs::write(directory.0.join("model.safetensors.index.json"), b"{}").unwrap();
        assert!(run(&options).unwrap_err().to_string().contains("sharded"));
        assert!(!options.output.exists());
    }
}
