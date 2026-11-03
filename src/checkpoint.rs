//! 原生 checkpoint 绑定模型 record、两条 pipeline 与可离线解析的资源。

use std::{
    collections::BTreeMap,
    fs,
    io::Read,
    path::{Component, Path, PathBuf},
};

use anyhow::{Context as _, Result, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::policy::PolicyKind;
use crate::processor::Pipeline;

const VERSION: u32 = 2;
const MANIFEST: &str = "manifest.json";

/// 一套经过摘要校验的原生模型与数据转换。
#[derive(Debug)]
pub struct Checkpoint {
    pub policy_kind: PolicyKind,
    pub model_dir: PathBuf,
    pub vlm_path: Option<PathBuf>,
    pub preprocessor: Pipeline,
    pub postprocessor: Pipeline,
    pub interop_dir: Option<PathBuf>,
}

/// 保存时的只读配置与 pipeline；权重由调用方从活模型写入。
pub struct CheckpointParts<'a> {
    pub policy_kind: PolicyKind,
    pub config: &'a Path,
    pub vlm: Option<&'a Path>,
    pub preprocessor: &'a Pipeline,
    pub postprocessor: &'a Pipeline,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    version: u32,
    policy_kind: PolicyKind,
    model_config: PathBuf,
    model_weights: PathBuf,
    preprocessor: PathBuf,
    postprocessor: PathBuf,
    vlm_path: Option<PathBuf>,
    interop_dir: Option<PathBuf>,
    resources: BTreeMap<PathBuf, String>,
}

impl Checkpoint {
    /// 从活模型写入 bpk，配置与 pipeline 一起在临时目录验证后发布。
    ///
    /// `write` 接收临时根目录与模型权重目标，可额外写入 `interop/` 转换资源。
    /// 调用方须保证源配置复制期间不变、目标单写者。已有目标拒绝覆盖。
    ///
    /// # Errors
    /// 不支持的模型、资源缺失、写入失败或校验失败时返回错误。
    pub fn save(
        destination: &Path,
        parts: &CheckpointParts<'_>,
        write: impl FnOnce(&Path, &Path) -> Result<()>,
    ) -> Result<()> {
        staging(destination, |root| {
            fs::create_dir(root.join("model"))?;
            fs::copy(parts.config, root.join("model/config.json"))?;
            write(root, &root.join("model/model.bpk"))?;
            // 文件名不能冒充格式；实际解析 bpk 容器。
            burn_pack::Reader::from_file_exact(root.join("model/model.bpk"))?;
            parts.preprocessor.save(&root.join("preprocessor"))?;
            parts.postprocessor.save(&root.join("postprocessor"))?;
            let vlm_path = if let Some(vlm) = parts.vlm {
                fs::create_dir(root.join("vlm"))?;
                ensure!(
                    vlm.join("tokenizer.json").is_file(),
                    "tokenizer resource {} is missing",
                    vlm.join("tokenizer.json").display()
                );
                for file in ["config.json", "tokenizer.json", "tokenizer_config.json"] {
                    if vlm.join(file).is_file() {
                        fs::copy(vlm.join(file), root.join("vlm").join(file))?;
                    }
                }
                for file in [
                    "preprocessor_config.json",
                    "processor_config.json",
                    "special_tokens_map.json",
                    "added_tokens.json",
                    "chat_template.json",
                    "generation_config.json",
                    "merges.txt",
                    "vocab.json",
                ] {
                    if vlm.join(file).is_file() {
                        fs::copy(vlm.join(file), root.join("vlm").join(file))?;
                    }
                }
                Some(PathBuf::from("vlm"))
            } else {
                None
            };
            let manifest = Manifest {
                version: VERSION,
                policy_kind: parts.policy_kind,
                model_config: "model/config.json".into(),
                model_weights: "model/model.bpk".into(),
                preprocessor: "preprocessor".into(),
                postprocessor: "postprocessor".into(),
                vlm_path,
                interop_dir: root.join("interop").is_dir().then(|| "interop".into()),
                resources: resource_hashes(root)?,
            };
            fs::write(root.join(MANIFEST), serde_json::to_vec_pretty(&manifest)?)?;
            Self::load(root).context("validating staged checkpoint")?;
            Ok(())
        })
    }

    /// 加载原生 bpk checkpoint。
    ///
    /// # Errors
    /// 未知版本、资源摘要不符、越界引用或非法 bpk 均拒绝加载。
    pub fn load(directory: &Path) -> Result<Self> {
        let root = directory.canonicalize().with_context(|| {
            format!(
                "opening native checkpoint {}; run pneuma convert first",
                directory.display()
            )
        })?;
        let manifest: Manifest =
            serde_json::from_slice(&fs::read(local_path(&root, Path::new(MANIFEST))?)?)?;
        ensure!(
            manifest.version == VERSION,
            "unsupported checkpoint version {}; convert the LeRobot source again",
            manifest.version
        );
        let actual = resource_hashes(&root)?;
        ensure!(
            actual == manifest.resources,
            "checkpoint resource digest mismatch or unlisted resource"
        );
        let config = listed_file(&root, &actual, &manifest.model_config)?;
        let weights = listed_file(&root, &actual, &manifest.model_weights)?;
        ensure!(
            config.file_name().is_some_and(|n| n == "config.json")
                && weights.file_name().is_some_and(|n| n == "model.bpk")
                && config.parent() == weights.parent(),
            "native model references must name config.json and model.bpk in one directory"
        );
        burn_pack::Reader::from_file_exact(&weights)?;
        let model_dir = config
            .parent()
            .context("missing model directory")?
            .to_path_buf();
        let pipeline = |relative: &Path| -> Result<Pipeline> {
            listed_file(&root, &actual, &relative.join("pipeline.json"))?;
            Pipeline::load(&local_path(&root, relative)?)
        };
        let vlm_path = manifest
            .vlm_path
            .as_ref()
            .map(|relative| -> Result<PathBuf> {
                // `config.json` 只有部分分词器目录带；`tokenizer.json` 必须存在。
                listed_file(&root, &actual, &relative.join("tokenizer.json"))?;
                let config = relative.join("config.json");
                if actual.contains_key(&config) {
                    listed_file(&root, &actual, &config)?;
                }
                let tokenizer_config = relative.join("tokenizer_config.json");
                if actual.contains_key(&tokenizer_config) {
                    listed_file(&root, &actual, &tokenizer_config)?;
                }
                local_path(&root, relative)
            })
            .transpose()?;
        let interop_dir = manifest
            .interop_dir
            .as_ref()
            .map(|relative| -> Result<PathBuf> {
                let path = local_path(&root, relative)?;
                ensure!(path.is_dir(), "interop resource must be a directory");
                Ok(path)
            })
            .transpose()?;
        Ok(Self {
            policy_kind: manifest.policy_kind,
            model_dir,
            vlm_path,
            preprocessor: pipeline(&manifest.preprocessor)?,
            postprocessor: pipeline(&manifest.postprocessor)?,
            interop_dir,
        })
    }
}

fn listed_file(
    root: &Path,
    resources: &BTreeMap<PathBuf, String>,
    relative: &Path,
) -> Result<PathBuf> {
    ensure!(
        resources.contains_key(relative),
        "unlisted checkpoint resource {}",
        relative.display()
    );
    let path = local_path(root, relative)?;
    ensure!(path.is_file(), "checkpoint resource is not a file");
    Ok(path)
}

pub(crate) fn local_path(root: &Path, relative: &Path) -> Result<PathBuf> {
    ensure!(
        !relative.as_os_str().is_empty()
            && relative
                .components()
                .all(|part| matches!(part, Component::Normal(_))),
        "resource must be checkpoint-relative: {}",
        relative.display()
    );
    let path = root.join(relative).canonicalize()?;
    ensure!(path.starts_with(root), "resource escapes checkpoint");
    Ok(path)
}

fn file_digest(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let length = file.read(&mut buffer)?;
        if length == 0 {
            break;
        }
        digest.update(&buffer[..length]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn resource_hashes(root: &Path) -> Result<BTreeMap<PathBuf, String>> {
    let mut resources = BTreeMap::new();
    let mut directories = vec![root.to_path_buf()];
    while let Some(directory) = directories.pop() {
        for entry in fs::read_dir(directory)? {
            let entry = entry?;
            let path = entry.path();
            let relative = path.strip_prefix(root)?;
            let kind = entry.file_type()?;
            ensure!(
                !kind.is_symlink(),
                "symlink checkpoint resource is unsupported"
            );
            if kind.is_dir() {
                directories.push(path);
            } else {
                ensure!(kind.is_file(), "checkpoint resource is not a regular file");
                if relative != Path::new(MANIFEST) {
                    resources.insert(relative.into(), file_digest(&path)?);
                }
            }
        }
    }
    Ok(resources)
}

pub(crate) fn ensure_absent(destination: &Path) -> Result<()> {
    match fs::symlink_metadata(destination) {
        Ok(_) => anyhow::bail!(
            "checkpoint destination already exists: {}",
            destination.display()
        ),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

/// 目标单写者；失败清理临时目录，已有目录不覆盖。
pub(crate) fn staging(destination: &Path, write: impl FnOnce(&Path) -> Result<()>) -> Result<()> {
    ensure_absent(destination)?;
    let parent = destination
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    fs::create_dir_all(parent)?;
    let temporary = parent.join(format!(".checkpoint-{}", uuid::Uuid::new_v4()));
    fs::create_dir(&temporary)?;
    let result = write(&temporary).and_then(|()| {
        ensure_absent(destination)?;
        fs::rename(&temporary, destination)?;
        Ok(())
    });
    if result.is_err() {
        let _ = fs::remove_dir_all(temporary);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::processor::{Batch, Context, Data, Field, Layout, Mode, Schema, Step};
    use burn_pack::{Bytes, DType, Tensor, Writer};

    struct Directory(PathBuf);
    impl Directory {
        fn new() -> Self {
            let path =
                std::env::temp_dir().join(format!("pneuma-checkpoint-{}", uuid::Uuid::new_v4()));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for Directory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn save(root: &Path, output: &Path) -> Result<()> {
        let config = root.join("config.json");
        fs::write(&config, br#"{"type":"act"}"#)?;
        let schema = Schema::from([(
            "state".into(),
            Field {
                shape: vec![Some(1)],
                dtype: crate::processor::DType::F32,
                layout: Layout::Flat,
                optional: false,
            },
        )]);
        let pre = Pipeline::new(
            schema,
            vec![Step::Affine {
                key: "state".into(),
                offset: vec![1.],
                scale: vec![2.],
                axis: 0,
            }],
        )?;
        let post = Pipeline::new(Schema::new(), vec![])?;
        Checkpoint::save(
            output,
            &CheckpointParts {
                policy_kind: PolicyKind::Act,
                config: &config,
                vlm: None,
                preprocessor: &pre,
                postprocessor: &post,
            },
            |_, path| {
                Writer::new(vec![Tensor::new(
                    "weight".into(),
                    DType::F32,
                    vec![1],
                    Some(42),
                    Bytes::from_bytes_vec(3f32.to_le_bytes().to_vec()),
                )])
                .write_to_file(path)?;
                Ok(())
            },
        )
    }

    #[test]
    fn native_checkpoint_roundtrip_and_integrity() {
        let dir = Directory::new();
        let output = dir.0.join("native");
        save(&dir.0, &output).unwrap();
        let loaded = Checkpoint::load(&output).unwrap();
        assert_eq!(loaded.policy_kind, PolicyKind::Act);
        assert!(!loaded.model_dir.join("model.safetensors").exists());
        let tensors = burn_pack::Reader::from_file_exact(loaded.model_dir.join("model.bpk"))
            .unwrap()
            .into_tensors()
            .unwrap();
        assert_eq!(tensors[0].param_id, Some(42));
        let batch = Batch {
            values: [("state".into(), Data::f32(vec![5.], vec![1]))].into(),
            ..Batch::default()
        };
        let output_batch = loaded
            .preprocessor
            .run(
                batch,
                &Context {
                    device: burn::prelude::Device::flex(),
                    mode: Mode::Eval,
                },
            )
            .unwrap();
        let values = output_batch.get("state").unwrap().host_f32().unwrap();
        assert_eq!(values, &[2.]);
        assert!(save(&dir.0, &output).is_err());
        fs::write(output.join("model/model.bpk"), b"bad").unwrap();
        assert!(Checkpoint::load(&output).is_err());
    }

    #[test]
    fn old_versions_and_unlisted_resources_are_rejected() {
        for case in 0..3 {
            let dir = Directory::new();
            let output = dir.0.join("native");
            save(&dir.0, &output).unwrap();
            if case == 0 {
                fs::write(output.join("extra"), b"unlisted").unwrap();
            } else {
                let path = output.join(MANIFEST);
                let mut manifest: Manifest =
                    serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
                if case == 1 {
                    manifest.version = 1;
                } else {
                    manifest.model_weights = "../config.json".into();
                }
                fs::write(path, serde_json::to_vec(&manifest).unwrap()).unwrap();
            }
            assert!(Checkpoint::load(&output).is_err());
        }
    }
}
