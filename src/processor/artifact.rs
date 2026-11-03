use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Component, Path, PathBuf},
};

use anyhow::{Context as _, Result, ensure};
use safetensors::{
    Dtype, SafeTensors,
    tensor::{TensorView, serialize_to_file},
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use super::{Pipeline, Schema, Step};

const VERSION: u32 = 1;
const MANIFEST: &str = "pipeline.json";
const STATISTICS: &str = "statistics.safetensors";

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    version: u32,
    input_schema: Schema,
    output_schema: Schema,
    steps: Vec<SavedStep>,
    resources: BTreeMap<PathBuf, String>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SavedStep {
    version: u32,
    config: Value,
}

impl Pipeline {
    /// 写入相邻临时目录，验证完整后原子发布；已有产物不可覆盖。
    pub fn save(&self, path: &Path) -> Result<()> {
        ensure!(
            !path.exists(),
            "pipeline destination already exists: {}",
            path.display()
        );
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        fs::create_dir_all(parent)?;
        let staging = parent.join(format!(".pipeline-{}", uuid::Uuid::new_v4()));
        fs::create_dir(&staging)?;
        let result = self.write_artifact(&staging).and_then(|()| {
            Self::load(&staging).context("validating staged pipeline")?;
            ensure!(!path.exists(), "pipeline destination appeared during save");
            fs::rename(&staging, path).context("publishing pipeline")
        });
        if result.is_err() {
            let _ = fs::remove_dir_all(&staging);
        }
        result
    }

    fn write_artifact(&self, root: &Path) -> Result<()> {
        let mut resources = BTreeMap::new();
        let mut tensors: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        let mut steps = Vec::with_capacity(self.steps.len());
        for (index, step) in self.steps.iter().enumerate() {
            let mut config = serde_json::to_value(step)?;
            if let Step::Affine { offset, scale, .. } | Step::Linear { offset, scale, .. } = step {
                for (name, values) in [("offset", offset), ("scale", scale)] {
                    let key = format!("step_{index}.{name}");
                    tensors.insert(
                        key.clone(),
                        values.iter().flat_map(|v| v.to_le_bytes()).collect(),
                    );
                    config[name] = serde_json::json!({ "resource": STATISTICS, "tensor": key });
                }
            }
            if let Step::Tokenize { .. } = step {
                let directory = PathBuf::from(format!("assets/tokenizer_{index}"));
                fs::create_dir_all(root.join(&directory))?;
                let prepared = self.tokenizers[index]
                    .as_ref()
                    .context("missing tokenizer snapshot")?;
                for (file, bytes) in &prepared.files {
                    let relative = directory.join(file);
                    fs::write(root.join(&relative), bytes)?;
                    resources.insert(relative, digest(bytes));
                }
                config["tokenizer"] = serde_json::to_value(directory)?;
            }
            steps.push(SavedStep {
                version: VERSION,
                config,
            });
        }
        if !tensors.is_empty() {
            let views = tensors
                .iter()
                .map(|(key, data)| {
                    Ok((
                        key.as_str(),
                        TensorView::new(Dtype::F32, vec![data.len() / 4], data)?,
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
            serialize_to_file(views, None, &root.join(STATISTICS))?;
            resources.insert(
                PathBuf::from(STATISTICS),
                digest(&fs::read(root.join(STATISTICS))?),
            );
        }
        let manifest = Manifest {
            version: VERSION,
            input_schema: self.input.clone(),
            output_schema: self.output_schema().clone(),
            steps,
            resources,
        };
        fs::write(root.join(MANIFEST), serde_json::to_vec_pretty(&manifest)?)?;
        Ok(())
    }

    /// 只解析产物内相对资源；未知版本、损坏资源和未知配置立即失败。
    pub fn load(path: &Path) -> Result<Self> {
        let root = path
            .canonicalize()
            .with_context(|| format!("opening pipeline {}", path.display()))?;
        let manifest: Manifest = serde_json::from_slice(&fs::read(root.join(MANIFEST))?)?;
        ensure!(
            manifest.version == VERSION,
            "unsupported pipeline version {}",
            manifest.version
        );
        let mut resources = BTreeMap::new();
        for (relative, expected) in &manifest.resources {
            let absolute = local_path(&root, relative)?;
            let bytes = fs::read(absolute)?;
            ensure!(
                digest(&bytes) == *expected,
                "resource digest mismatch: {}",
                relative.display()
            );
            resources.insert(relative.clone(), bytes);
        }
        let statistics = resources
            .get(Path::new(STATISTICS))
            .map(|bytes| SafeTensors::deserialize(bytes))
            .transpose()?;
        let mut used_resources = BTreeSet::new();
        let mut used_tensors = BTreeSet::new();
        let mut steps = Vec::with_capacity(manifest.steps.len());
        let mut prepared = Vec::with_capacity(manifest.steps.len());
        for saved in manifest.steps {
            ensure!(
                saved.version == VERSION,
                "unsupported processor step version {}",
                saved.version
            );
            let mut config = saved.config;
            let kind = config
                .get("kind")
                .and_then(Value::as_str)
                .context("missing step kind")?;
            if matches!(kind, "affine" | "linear") {
                for name in ["offset", "scale"] {
                    let reference: StatisticRef = serde_json::from_value(
                        config
                            .get(name)
                            .cloned()
                            .context("missing statistic reference")?,
                    )?;
                    ensure!(
                        reference.resource == Path::new(STATISTICS),
                        "unexpected statistic resource"
                    );
                    let tensor = statistics
                        .as_ref()
                        .context("missing statistics resource")?
                        .tensor(&reference.tensor)?;
                    ensure!(
                        tensor.dtype() == Dtype::F32 && tensor.shape().len() == 1,
                        "statistics must be rank-1 f32"
                    );
                    let values: Vec<f32> = tensor
                        .data()
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|v| f32::from_le_bytes(*v))
                        .collect();
                    config[name] = serde_json::to_value(values)?;
                    used_resources.insert(reference.resource);
                    used_tensors.insert(reference.tensor);
                }
            }
            let mut step: Step = serde_json::from_value(config)?;
            let cached = if let Step::Tokenize {
                tokenizer,
                max_length,
                padding,
                padding_side,
                ..
            } = &mut step
            {
                let mut files = BTreeMap::new();
                for file in ["tokenizer.json", "tokenizer_config.json"] {
                    let relative = tokenizer.join(file);
                    let bytes = resources.get(&relative).with_context(|| {
                        format!("unlisted tokenizer resource {}", relative.display())
                    })?;
                    files.insert(file.to_string(), bytes.clone());
                    used_resources.insert(relative);
                }
                *tokenizer = local_path(&root, tokenizer)?;
                Some(super::tokenize::from_files(
                    files,
                    *max_length,
                    padding,
                    padding_side,
                )?)
            } else {
                None
            };
            prepared.push(cached);
            steps.push(step);
        }
        ensure!(
            used_resources.len() == resources.len(),
            "unreferenced pipeline resources"
        );
        if let Some(statistics) = statistics {
            ensure!(
                statistics.names().iter().all(|n| used_tensors.contains(*n)),
                "unreferenced statistics tensors"
            );
        }
        let pipeline = Self::from_prepared(manifest.input_schema, steps, Some(prepared))?;
        ensure!(
            pipeline.output_schema() == &manifest.output_schema,
            "saved output schema disagrees with inferred schema"
        );
        Ok(pipeline)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StatisticRef {
    resource: PathBuf,
    tensor: String,
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn local_path(root: &Path, relative: &Path) -> Result<PathBuf> {
    ensure!(
        !relative.as_os_str().is_empty()
            && relative
                .components()
                .all(|c| matches!(c, Component::Normal(_))),
        "resource must be an artifact-relative path: {}",
        relative.display()
    );
    let absolute = root.join(relative).canonicalize()?;
    ensure!(
        absolute.starts_with(root),
        "resource escapes artifact: {}",
        relative.display()
    );
    Ok(absolute)
}
