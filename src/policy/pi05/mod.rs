//! Burn 原生 π₀.₅：PaliGemma 前缀、Gemma 动作专家与 flow-matching Euler 推理。

mod config;
mod gemma;
mod model;
mod processing;
mod vision;

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use burn::module::Module;
use burn::tensor::{DType, Device, Distribution, Tensor};
use burn_store::{
    FloatCastAdapter, ModuleAdapter, ModuleSnapshot, ModuleStore, PyTorchToBurnAdapter,
    SafetensorsStore,
};

use crate::contract::FeatureSpec;
use crate::policy::Policy;
use crate::processor::{Batch, Data};

use config::Pi05Config;
use model::Pi05Net;

pub struct Pi05Policy {
    model: Pi05Net,
    cfg: Pi05Config,
    device: Device,
    /// 官方 PaliGemma 只存 `lm_head`，`embed_tokens` 与之绑定。
    tied_embeddings: bool,
    /// 不同族系的 checkpoint 带 `model.` 前缀；导出时还原同一布局。
    source_prefix: String,
}

const EMBEDDINGS: &str = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight";
const LM_HEAD: &str = "paligemma_with_expert.paligemma.lm_head.weight";

/// π₀.₅ 策略的构造配置。
///
/// 用 bon 生成具名 builder；`init` 只构建模型结构，
/// 权重由 `Pi05Policy::load_record` 单独恢复。
#[derive(Debug, bon::Builder)]
#[builder(start_fn = new)]
pub struct Pi05PolicyConfig {
    #[builder(into)]
    path: PathBuf,
    vlm_path: Option<PathBuf>,
    device: crate::config::Device,
}

impl Pi05PolicyConfig {
    /// 解析 `config.json` 并构建未加载权重的策略。
    pub fn init(&self) -> Result<Pi05Policy> {
        self.init_with_prefix(source_prefix_from_interop(&self.path))
    }

    fn init_with_prefix(&self, source_prefix: String) -> Result<Pi05Policy> {
        let cfg = config::parse(&self.path.join("config.json"))?;
        let device: Device = self.device.into();
        let model = Pi05Net::init(&cfg, &device)?;
        Ok(Pi05Policy {
            model,
            cfg,
            device,
            tied_embeddings: true,
            source_prefix,
        })
    }

    /// 严格加载原生模型 record，外部 safetensors 必须先转换。
    pub fn load_model(&self) -> Result<Pi05Policy> {
        let policy = self.init()?;
        let record = burn::store::ModuleRecord::load(self.path.join("model.bpk"))
            .context("reading native pi05 model.bpk; convert the LeRobot checkpoint first")?;
        policy.load_record(record)
    }

    pub fn import_configured(
        &self,
        features: &[FeatureSpec],
        action: &FeatureSpec,
    ) -> Result<(
        Pi05Policy,
        crate::processor::Pipeline,
        crate::processor::Pipeline,
        Vec<String>,
    )> {
        let cfg = config::parse(&self.path.join("config.json"))?;
        processing::validate_contract(&cfg, features, action)?;
        let tokenizer_dir = self
            .vlm_path
            .clone()
            .unwrap_or_else(|| self.path.join("vlm"));
        let (preprocessor, postprocessor) =
            processing::import_pipelines(&self.path, &cfg, &tokenizer_dir)?;
        let (policy, unused) = self.import_lerobot()?;
        Ok((policy, preprocessor, postprocessor, unused))
    }

    /// 转换边界导入 LeRobot 参数；未使用参数（专家 lm_head）留给 checkpoint 转换器保存。
    pub fn import_lerobot(&self) -> Result<(Pi05Policy, Vec<String>)> {
        let weights = self.path.join("model.safetensors");
        let names = tensor_names(&weights)?;
        let prefixed = names.iter().any(|name| name.starts_with("model."));
        let source_prefix = if prefixed { "model." } else { "" }.to_string();
        let mut policy = self.init_with_prefix(source_prefix)?;
        policy.tied_embeddings =
            !names.contains(EMBEDDINGS) && !names.contains(&format!("model.{EMBEDDINGS}"));
        let mut store = SafetensorsStore::from_file(&weights)
            .with_from_adapter(PyTorchToBurnAdapter.chain(FloatCastAdapter::to(DType::F32)))
            .allow_partial(true);
        if prefixed {
            store = store.with_key_remapping(r"^model\.", "");
        }
        if policy.tied_embeddings {
            // 前缀可能已在上一条规则中移除，所以两种写法都接受。
            store = store.with_key_remapping(
                r"^(model\.)?paligemma_with_expert\.paligemma\.lm_head\.weight$",
                EMBEDDINGS,
            );
        }
        let keys = store.keys().context("listing pi05 safetensors keys")?;
        tracing::debug!(
            count = keys.len(),
            sample = ?keys.iter().take(3).collect::<Vec<_>>(),
            "pi05 safetensors keys"
        );
        let applied = policy
            .model
            .load_from(&mut store)
            .context("loading pi05 weights")?;
        ensure!(
            applied.missing.is_empty() && applied.errors.is_empty(),
            "incomplete pi05 weights: missing={:?} errors={:?}",
            applied.missing,
            applied.errors
        );
        tracing::info!(
            applied = applied.applied.len(),
            unused = applied.unused.len(),
            device = ?policy.device,
            "pi05 weights imported"
        );
        Ok((policy, applied.unused))
    }
}

impl Pi05Policy {
    /// 应用已加载的原生 record；结构由 `Pi05PolicyConfig::init` 构建。
    pub fn load_record(self, record: burn::store::ModuleRecord) -> Result<Self> {
        let model = self
            .model
            .try_load_record(record)
            .context("restoring pi05 model record")?;
        Ok(Self {
            model,
            cfg: self.cfg,
            device: self.device,
            tied_embeddings: self.tied_embeddings,
            source_prefix: self.source_prefix,
        })
    }

    pub fn save_native(&self, path: &Path) -> Result<()> {
        self.model.clone().into_record().save(path)?;
        Ok(())
    }

    /// 导出参数名与 PyTorch 布局；绑定嵌入写回官方 `lm_head` 键。
    pub fn export_lerobot(&self, path: &Path) -> Result<()> {
        let mut store =
            SafetensorsStore::from_file(path).with_to_adapter(burn_store::BurnToPyTorchAdapter);
        if self.tied_embeddings {
            store = store.with_key_remapping(
                r"^paligemma_with_expert\.paligemma\.model\.language_model\.embed_tokens\.weight$",
                LM_HEAD,
            );
        }
        if !self.source_prefix.is_empty() {
            store = store.with_key_remapping(r"^", self.source_prefix.as_str());
        }
        self.model.save_into(&mut store)?;
        Ok(())
    }

    pub fn horizon(&self) -> (u64, u64) {
        (self.cfg.chunk_size as u64, self.cfg.n_action_steps as u64)
    }

    pub fn device(&self) -> Device {
        self.device.clone()
    }
}

impl Policy for Pi05Policy {
    fn predict(&self, batch: Batch) -> Result<Batch> {
        self.predict_with_noise(batch, None)
    }
}

impl Pi05Policy {
    /// 推理主体；`noise` 为 `None` 时按标准正态采样，测试可用固定噪声。
    pub(super) fn predict_with_noise(
        &self,
        mut batch: Batch,
        noise: Option<Tensor<3>>,
    ) -> Result<Batch> {
        let tokens = match batch.take("observation.language.tokens")? {
            Data::Int2(tokens) => {
                ensure!(tokens.dims()[0] == 1, "pi05 requires one task sample");
                tokens.into_data().iter::<i64>().collect::<Vec<_>>()
            }
            _ => anyhow::bail!("pi05 requires prepared integer language tokens"),
        };
        let valid = match batch.take("observation.language.attention_mask")? {
            Data::Bool2(mask) => {
                ensure!(
                    mask.dims() == [1, tokens.len()],
                    "pi05 language mask shape differs from tokens"
                );
                mask.into_data().iter::<bool>().collect::<Vec<_>>()
            }
            _ => anyhow::bail!("pi05 requires prepared boolean language mask"),
        };
        let mut images = Vec::new();
        for (name, _) in self.cfg.image_features() {
            let image = batch.take(name)?.into_float4()?;
            ensure!(
                image.dims()[0] == 1,
                "pi05 requires one prepared image sample"
            );
            ensure!(
                image.dims()[1] == 3,
                "pi05 requires CHW camera input for {name}"
            );
            images.push(image);
        }
        let noise = noise.unwrap_or_else(|| {
            Tensor::random(
                [1, self.cfg.chunk_size, self.cfg.max_action_dim],
                Distribution::Normal(0., 1.),
                &self.device,
            )
        });
        let action = self
            .model
            .sample(images, &tokens, &valid, noise, &self.cfg, &self.device)?;
        Ok(Batch {
            values: [("action".into(), Data::Float3(action))].into(),
            ..Batch::default()
        })
    }
}

/// 从转换记录恢复源 checkpoint 的键前缀，保证反向导出使用原布局。
fn source_prefix_from_interop(model_dir: &Path) -> String {
    let Some(root) = model_dir.parent() else {
        return String::new();
    };
    let Ok(bytes) = std::fs::read(root.join("interop/conversion.json")) else {
        return String::new();
    };
    let Ok(value) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
        return String::new();
    };
    let prefixed = value
        .get("tensors")
        .and_then(|tensors| tensors.as_object())
        .is_some_and(|tensors| tensors.keys().any(|name| name.starts_with("model.")));
    if prefixed {
        "model.".into()
    } else {
        String::new()
    }
}

/// 只读 safetensors 头部，判断绑定权重是否存在，不加载张量数据。
fn tensor_names(path: &Path) -> Result<std::collections::BTreeSet<String>> {
    use std::io::Read;
    let mut file =
        std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut length = [0u8; 8];
    file.read_exact(&mut length)
        .context("reading safetensors header length")?;
    let length = usize::try_from(u64::from_le_bytes(length))
        .context("safetensors header length overflow")?;
    ensure!(
        length > 0 && length <= 64 * 1024 * 1024,
        "invalid safetensors header length"
    );
    let mut header = vec![0u8; length];
    file.read_exact(&mut header)
        .context("reading safetensors header")?;
    let value: serde_json::Value =
        serde_json::from_slice(&header).context("parsing safetensors header")?;
    Ok(value
        .as_object()
        .context("safetensors header must be an object")?
        .keys()
        .filter(|key| key.as_str() != "__metadata__")
        .cloned()
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Int, TensorData};

    /// 加载原生 checkpoint 并跑一次前向；需要已转换的 pi05 产物。
    /// 运行：`PNEUMA_PI05_CKPT=/path/to/native cargo test --release -- --ignored pi05_native --nocapture`
    #[test]
    #[ignore = "需要真实 pi05 权重"]
    fn pi05_native_checkpoint_predicts_finite_actions() {
        let path = std::env::var("PNEUMA_PI05_CKPT").expect("set PNEUMA_PI05_CKPT");
        let device = std::env::var("PNEUMA_DEVICE").unwrap_or_else(|_| "flex".into());
        let backend: crate::config::Device = device.parse().expect("PNEUMA_DEVICE");
        let artifact =
            crate::checkpoint::Checkpoint::load(Path::new(&path)).expect("loading pi05 checkpoint");
        let policy = Pi05PolicyConfig::new()
            .path(&artifact.model_dir)
            .maybe_vlm_path(artifact.vlm_path.clone())
            .device(backend)
            .build()
            .load_model()
            .expect("loading pi05");
        let resolved = policy.device();
        let mut batch = Batch::default();
        for (name, _) in policy.cfg.image_features() {
            batch.values.insert(
                name.clone(),
                Data::Float4(burn::tensor::Tensor::zeros([1, 3, 224, 224], &resolved)),
            );
        }
        let tokens: Vec<i64> = (0..policy.cfg.tokenizer_max_length)
            .map(|index| i64::from(index as i32 % 100))
            .collect();
        let count = tokens.len();
        batch.values.insert(
            "observation.language.tokens".into(),
            Data::Int2(Tensor::<2, Int>::from_data(
                TensorData::new(tokens, [1, count]),
                &resolved,
            )),
        );
        batch.values.insert(
            "observation.language.attention_mask".into(),
            Data::Bool2(Tensor::<2, burn::tensor::Bool>::ones([1, count], &resolved)),
        );
        let output = policy.predict(batch).expect("pi05 predict");
        let action = output.values.get("action").expect("action");
        let tensor = match action {
            Data::Float3(tensor) => tensor.clone(),
            other => panic!("unexpected action data: {other:?}"),
        };
        assert_eq!(
            tensor.dims(),
            [1, policy.cfg.chunk_size, policy.cfg.max_action_dim]
        );
        // 全零或常数输出说明权重没有真正参与计算。
        let values = tensor.into_data().try_to_vec::<f32>().expect("f32 actions");
        assert!(values.iter().all(|x| x.is_finite()));
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32;
        let (min, max) = values
            .iter()
            .fold((f32::MAX, f32::MIN), |(lo, hi), x| (lo.min(*x), hi.max(*x)));
        println!(
            "pi05 action mean={mean:.4} std={:.4} min={min:.4} max={max:.4}",
            variance.sqrt()
        );
        assert!(variance.sqrt() > 1e-3, "degenerate action output");
    }

    use std::path::{Path, PathBuf};

    fn read_f32(path: &Path) -> Vec<f32> {
        let bytes = std::fs::read(path)
            .unwrap_or_else(|error| panic!("reading {}: {error}", path.display()));
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|value| f32::from_le_bytes(*value))
            .collect()
    }

    fn compare(name: &str, got: &[f32], want: &[f32], tolerance: f32) {
        assert_eq!(got.len(), want.len(), "{name}: length");
        let deltas: Vec<f32> = got.iter().zip(want).map(|(a, b)| (a - b).abs()).collect();
        let max = deltas.iter().copied().fold(0.0f32, f32::max);
        let mean = deltas.iter().sum::<f32>() / deltas.len() as f32;
        let count = deltas.iter().filter(|value| **value > 1e-3).count();
        println!(
            "{name}: max {max:.3e} mean {mean:.3e} over {} values, {count} above 1e-3; first got {:?} want {:?}",
            got.len(),
            &got[..got.len().min(4)],
            &want[..want.len().min(4)]
        );
        assert!(
            max <= tolerance,
            "{name} differs by {max:.3e} > {tolerance:.1e}"
        );
    }

    /// 与 `scripts/dump_pi05_reference.py` 的参照逐值比较；覆盖预处理、分词、噪声与动作。
    #[test]
    #[ignore = "需要官方 π₀.₅ 参照张量"]
    fn pi05_matches_official_reference() {
        let parity =
            PathBuf::from(std::env::var("PNEUMA_PI05_PARITY").expect("PNEUMA_PI05_PARITY"));
        let native = std::env::var("PNEUMA_PI05_CKPT").expect("PNEUMA_PI05_CKPT");
        let device = std::env::var("PNEUMA_DEVICE").unwrap_or_else(|_| "flex".into());
        let backend: crate::config::Device = device.parse().expect("PNEUMA_DEVICE");
        let artifact = crate::checkpoint::Checkpoint::load(Path::new(&native)).expect("checkpoint");
        let policy = Pi05PolicyConfig::new()
            .path(&artifact.model_dir)
            .maybe_vlm_path(artifact.vlm_path.clone())
            .device(backend)
            .build()
            .load_model()
            .expect("pi05");
        let device = policy.device();
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(parity.join("manifest.json")).expect("manifest"))
                .expect("manifest json");
        let task = manifest["task"].as_str().expect("manifest task").to_owned();

        let mut batch = Batch {
            tasks: vec![task],
            ..Batch::default()
        };
        batch.values.insert(
            "observation.state".into(),
            Data::f32(read_f32(&parity.join("raw_state.bin")), vec![8]),
        );
        for (index, (name, _)) in policy.cfg.image_features().enumerate() {
            batch.values.insert(
                name.clone(),
                Data::u8(
                    // 参照把原始图像按 f32 存储；这里转回协议要求的 uint8。
                    read_f32(&parity.join(format!("raw_image_{index}.bin")))
                        .into_iter()
                        .map(|value| value as u8)
                        .collect(),
                    vec![256, 256, 3],
                ),
            );
        }
        let context = crate::processor::Context {
            device: device.clone(),
            mode: crate::processor::Mode::Eval,
        };
        let prepared = artifact
            .preprocessor
            .run(batch, &context)
            .expect("preprocessing");

        let state = prepared
            .get("observation.state")
            .expect("state")
            .clone()
            .into_float2()
            .expect("state tensor")
            .into_data()
            .try_to_vec::<f32>()
            .expect("state values");
        compare(
            "state",
            &state,
            &read_f32(&parity.join("state_normalized.bin")),
            1e-5,
        );

        for (index, (name, _)) in policy.cfg.image_features().enumerate() {
            let image = prepared
                .get(name)
                .expect("image")
                .clone()
                .into_float4()
                .expect("image tensor")
                .into_data()
                .try_to_vec::<f32>()
                .expect("image values");
            compare(
                &format!("image{index}"),
                &image,
                &read_f32(&parity.join(format!("prepared_image_{index}.bin"))),
                1e-5,
            );
        }

        let token_ids: Vec<i64> = match prepared.get("observation.language.tokens").expect("tokens")
        {
            Data::Int2(tensor) => tensor.clone().into_data().iter::<i64>().collect(),
            other => panic!("unexpected tokens {other:?}"),
        };
        let tokens: Vec<f32> = token_ids.iter().map(|value| *value as f32).collect();
        compare(
            "tokens",
            &tokens,
            &read_f32(&parity.join("tokens.bin")),
            0.0,
        );
        let mask: Vec<f32> = match prepared
            .get("observation.language.attention_mask")
            .expect("mask")
        {
            Data::Bool2(tensor) => tensor
                .clone()
                .into_data()
                .iter::<bool>()
                .map(|value| f32::from(u8::from(value)))
                .collect(),
            other => panic!("unexpected mask {other:?}"),
        };
        compare(
            "mask",
            &mask,
            &read_f32(&parity.join("token_mask.bin")),
            0.0,
        );

        let valid: Vec<bool> = mask.iter().map(|value| *value > 0.5).collect();
        let mut images = Vec::new();
        for (name, _) in policy.cfg.image_features() {
            images.push(
                prepared
                    .get(name)
                    .expect("image")
                    .clone()
                    .into_float4()
                    .expect("image tensor"),
            );
        }
        let (prefix, _) = policy
            .model
            .embed_prefix(
                images.clone(),
                &token_ids,
                &valid,
                &device,
                &policy.cfg.dtype,
            )
            .expect("prefix");
        let dims = prefix.dims();
        let (length, width) = (dims[1], dims[2]);
        assert_eq!(length, 512 + token_ids.len());
        let prefix_values = prefix
            .into_data()
            .try_to_vec::<f32>()
            .expect("prefix values");
        let reference_prefix = read_f32(&parity.join("prefix_embs.bin"));
        let images_end = 512 * width;
        let camera = 256 * width;
        for index in 0..2 {
            let range = index * camera..(index + 1) * camera;
            compare(
                &format!("prefix_camera{index}"),
                &prefix_values[range.clone()],
                &reference_prefix[range],
                // 官方视觉塔权重经 bf16 往返取整；本端口保留 F32 原值。
                1e-1,
            );
        }
        // 参照在语言 token 前额外有 256 个被 mask 的补零相机 token。
        let language = &prefix_values[images_end..images_end + token_ids.len() * width];
        let reference_language =
            &reference_prefix[768 * width..768 * width + token_ids.len() * width];
        compare("prefix_language", language, reference_language, 1e-6);

        let noise = Tensor::<3>::from_data(
            TensorData::new(
                read_f32(&parity.join("noise.bin")),
                [1, policy.cfg.chunk_size, policy.cfg.max_action_dim],
            ),
            &device,
        );

        let mut one_step = policy.cfg.clone();
        one_step.num_inference_steps = 1;
        let step1 = policy
            .model
            .sample(
                images,
                &token_ids,
                &valid,
                noise.clone(),
                &one_step,
                &device,
            )
            .expect("one Euler step");
        compare(
            "actions_step1",
            &step1.into_data().try_to_vec::<f32>().expect("step1"),
            &read_f32(&parity.join("actions_step1.bin")),
            2e-2,
        );

        let output = policy
            .predict_with_noise(prepared, Some(noise))
            .expect("pi05 predict");
        let actions = match output.values.get("action").expect("action") {
            Data::Float3(tensor) => tensor
                .clone()
                .into_data()
                .try_to_vec::<f32>()
                .expect("actions"),
            other => panic!("unexpected action {other:?}"),
        };
        compare(
            "actions",
            &actions,
            &read_f32(&parity.join("actions_model.bin")),
            // 参照的视觉权重被 bf16 取整；F32 端口与其最大差约 1e-2，均值约 3e-4。
            2e-2,
        );
    }
}
