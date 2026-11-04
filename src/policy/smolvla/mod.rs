//! Burn 原生 SmolVLA：视觉和语言前缀缓存、动作专家及 flow-matching Euler 推理。

mod config;
mod processing;
#[cfg(test)]
mod tests;
mod transformer;
mod vision;

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::silu;
use burn::tensor::{DType, Device, Distribution, Int, Tensor, TensorData};
use burn_store::{
    FloatCastAdapter, ModuleAdapter, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore,
};
use serde::Deserialize;

use crate::contract::FeatureSpec;
use crate::policy::Policy;
use crate::processor::{Batch, Data, Pipeline};

use config::SmolVlaConfig;
use transformer::{
    SmolVlmExpertCache, SmolVlmExpertConfig, SmolVlmExpertModel, SmolVlmKvCache, SmolVlmTextConfig,
    SmolVlmTextModel,
};
use vision::{SmolVlmConnector, SmolVlmConnectorConfig, SmolVlmVisionConfig, SmolVlmVisionModel};

#[derive(Deserialize)]
struct SmolVlmConfig {
    text_config: SmolVlmTextConfig,
    vision_config: SmolVlmVisionConfig,
    scale_factor: usize,
}

#[derive(Module, Debug)]
struct SmolVlmModel {
    vision_model: SmolVlmVisionModel,
    connector: SmolVlmConnector,
    text_model: SmolVlmTextModel,
}

#[derive(Module, Debug)]
struct SmolVlm {
    model: SmolVlmModel,
}

#[derive(Module, Debug)]
struct SmolVlmWithExpert {
    vlm: SmolVlm,
    lm_expert: SmolVlmExpertModel,
}

#[derive(Module, Debug)]
struct SmolVlaFlowMatching {
    vlm_with_expert: SmolVlmWithExpert,
    state_proj: Linear,
    action_in_proj: Linear,
    action_out_proj: Linear,
    action_time_mlp_in: Linear,
    action_time_mlp_out: Linear,
    #[module(skip)]
    expert_dim: usize,
}

/// 动作模型的构造配置，组合策略超参与 VLM 结构。
struct SmolVlaFlowMatchingConfig {
    policy: SmolVlaConfig,
    vlm: SmolVlmConfig,
}

impl SmolVlaFlowMatchingConfig {
    fn new(policy: SmolVlaConfig, vlm: SmolVlmConfig) -> Self {
        Self { policy, vlm }
    }

    fn init(&self, device: &Device) -> SmolVlaFlowMatching {
        let cfg = &self.policy;
        let vlm = &self.vlm;
        let text = &vlm.text_config;
        let max_sequence_length = max_sequence_length(cfg, vlm);
        let expert_dim = (text.hidden_size as f64 * cfg.expert_width_multiplier) as usize;
        let mut expert = text.clone();
        expert.hidden_size = expert_dim;
        // 与官方 get_intermediate_size 的逐步整数取整顺序一致。
        let intermediate = (4.0 * (2 * expert_dim / 3) as f64) as usize;
        expert.intermediate_size = intermediate.div_ceil(256) * 256;
        let cross = (0..text.num_hidden_layers)
            .map(|index| {
                cfg.attention_mode == "cross_attn"
                    && !(cfg.self_attn_every_n_layers > 0
                        && index % cfg.self_attn_every_n_layers as usize == 0)
            })
            .collect();
        SmolVlaFlowMatching {
            vlm_with_expert: SmolVlmWithExpert {
                vlm: SmolVlm {
                    model: SmolVlmModel {
                        vision_model: vlm.vision_config.init(device),
                        connector: SmolVlmConnectorConfig::new(
                            &vlm.vision_config,
                            text.hidden_size,
                        )
                        .init(device),
                        text_model: text.init(max_sequence_length, device),
                    },
                },
                lm_expert: SmolVlmExpertConfig::new(expert, cross, max_sequence_length)
                    .init(device),
            },
            state_proj: LinearConfig::new(cfg.max_state_dim, text.hidden_size).init(device),
            action_in_proj: LinearConfig::new(cfg.max_action_dim, expert_dim).init(device),
            action_out_proj: LinearConfig::new(expert_dim, cfg.max_action_dim).init(device),
            action_time_mlp_in: LinearConfig::new(expert_dim * 2, expert_dim).init(device),
            action_time_mlp_out: LinearConfig::new(expert_dim, expert_dim).init(device),
            expert_dim,
        }
    }
}

/// 旋转表需要覆盖前缀与动作块的最大位置。
fn max_sequence_length(cfg: &SmolVlaConfig, vlm: &SmolVlmConfig) -> usize {
    let board = vlm.vision_config.image_size / vlm.vision_config.patch_size;
    let tokens_per_image = (board / vlm.vision_config.scale_factor).pow(2);
    let actual = cfg.image_features().count() * tokens_per_image + cfg.tokenizer_max_length + 1;
    (cfg.prefix_length.max(0) as usize).max(actual) + cfg.chunk_size
}

impl SmolVlaFlowMatching {
    fn prefix(
        &self,
        images: Vec<Tensor<4>>,
        tokens: &[i64],
        language_valid: &[bool],
        state: Tensor<2>,
        cfg: &SmolVlaConfig,
    ) -> (Vec<SmolVlmKvCache>, Vec<bool>) {
        let vlm = &self.vlm_with_expert.vlm.model;
        let dim = state.dims()[1];
        let device = state.device();
        let mut embeddings = Vec::new();
        let mut valid = Vec::new();
        let mut groups = Vec::new();
        for image in images {
            let image = vlm.connector.forward(vlm.vision_model.forward(image));
            let count = image.dims()[1];
            let hidden = image.dims()[2];
            embeddings.push(image * (hidden as f64).sqrt());
            valid.extend(std::iter::repeat_n(true, count));
            groups.extend(std::iter::repeat_n(0, count));
        }
        let token_tensor = Tensor::<2, Int>::from_data(
            TensorData::new(tokens.to_vec(), [1, tokens.len()]),
            &device,
        );
        let language = vlm.text_model.embed_tokens.forward(token_tensor);
        let hidden = language.dims()[2];
        embeddings.push(language * (hidden as f64).sqrt());
        valid.extend_from_slice(language_valid);
        groups.extend(std::iter::repeat_n(0, tokens.len()));
        let state = self.state_proj.forward(state.reshape([1, dim]));
        let hidden = state.dims()[1];
        embeddings.push(state.reshape([1, 1, hidden]));
        valid.push(true);
        groups.push(1);
        let prefix_length = cfg.prefix_length.max(0) as usize;
        if valid.len() < prefix_length {
            let pad = prefix_length - valid.len();
            embeddings.push(Tensor::zeros([1, pad, hidden], &device));
            valid.extend(std::iter::repeat_n(false, pad));
            groups.extend(std::iter::repeat_n(1, pad));
        }
        (
            vlm.text_model
                .prefill(Tensor::cat(embeddings, 1), &valid, &groups),
            valid,
        )
    }

    /// 时间步嵌入只依赖固定步表，在去噪循环前一次算完。
    fn time_embedding(&self, timestep: f32, cfg: &SmolVlaConfig, device: &Device) -> Tensor<3> {
        let half = self.expert_dim / 2;
        let angles: Vec<f64> = (0..half)
            .map(|index| {
                let fraction = index as f64 / (half - 1) as f64;
                let period = cfg.min_period * (cfg.max_period / cfg.min_period).powf(fraction);
                f64::from(timestep) * std::f64::consts::TAU / period
            })
            .collect();
        let time: Vec<f32> = angles
            .iter()
            .map(|x| x.sin() as f32)
            .chain(angles.iter().map(|x| x.cos() as f32))
            .collect();
        Tensor::<3>::from_data(TensorData::new(time, [1, 1, self.expert_dim]), device).expand([
            1,
            cfg.chunk_size,
            self.expert_dim,
        ])
    }

    fn velocity(&self, x: Tensor<3>, time: Tensor<3>, cache: &mut SmolVlmExpertCache) -> Tensor<3> {
        let action = self.action_in_proj.forward(x);
        let suffix = self.action_time_mlp_out.forward(silu(
            self.action_time_mlp_in
                .forward(Tensor::cat(vec![action, time], 2)),
        ));
        self.action_out_proj
            .forward(self.vlm_with_expert.lm_expert.forward(suffix, cache))
    }

    fn sample(
        &self,
        images: Vec<Tensor<4>>,
        tokens: &[i64],
        valid: &[bool],
        state: Tensor<2>,
        noise: Tensor<3>,
        cfg: &SmolVlaConfig,
    ) -> Tensor<3> {
        let (prefix, valid) = self.prefix(images, tokens, valid, state, cfg);
        let device = noise.device();
        let mut cache = self
            .vlm_with_expert
            .lm_expert
            .cache(&prefix, &valid, cfg.chunk_size);
        let dt = -1.0 / cfg.num_steps as f64;
        let times: Vec<Tensor<3>> = (0..cfg.num_steps)
            .map(|step| self.time_embedding((1.0 + step as f64 * dt) as f32, cfg, &device))
            .collect();
        let mut x = noise;
        for time in times {
            let velocity = self.velocity(x.clone(), time, &mut cache);
            x = x + velocity * dt;
        }
        x
    }
}

pub struct SmolVlaPolicy {
    model: SmolVlaFlowMatching,
    cfg: SmolVlaConfig,
    device: Device,
}

/// SmolVLA 策略的构造配置。
///
/// 用 bon 生成具名 builder；`init` 只构建模型结构，
/// 权重由 `SmolVlaPolicy::load_record` 单独恢复。
#[derive(Debug, bon::Builder)]
#[builder(start_fn = new)]
pub struct SmolVlaPolicyConfig {
    #[builder(into)]
    path: PathBuf,
    vlm_path: Option<PathBuf>,
}

impl SmolVlaPolicyConfig {
    /// 解析 checkpoint 配置与 VLM 结构，构建未加载权重的策略。
    pub fn init(&self, device: &Device) -> Result<SmolVlaPolicy> {
        let cfg = config::parse(&self.path.join("config.json"))?;
        let vlm_path = self
            .vlm_path
            .clone()
            .unwrap_or_else(|| self.path.join("vlm"));
        let mut vlm: SmolVlmConfig =
            serde_json::from_slice(&std::fs::read(vlm_path.join("config.json")).context(
                "reading local SmolVLM config; provide convert --vlm-path when importing",
            )?)?;
        vlm.text_config.validate()?;
        let layers = if cfg.num_vlm_layers > 0 {
            cfg.num_vlm_layers as usize
        } else {
            vlm.text_config.num_hidden_layers
        };
        ensure!(
            layers <= vlm.text_config.num_hidden_layers,
            "num_vlm_layers exceeds VLM layer count"
        );
        vlm.text_config.num_hidden_layers = layers;
        vlm.vision_config.scale_factor = vlm.scale_factor;
        vlm.vision_config.validate()?;
        ensure!(
            vlm.text_config.hidden_size == 960
                && vlm.text_config.intermediate_size == 2560
                && vlm.text_config.num_attention_heads == 15
                && vlm.text_config.num_key_value_heads == 5
                && vlm.text_config.head_dim == 64
                && vlm.text_config.vocab_size == 49280
                && vlm.vision_config.hidden_size == 768
                && vlm.vision_config.intermediate_size == 3072
                && vlm.vision_config.num_hidden_layers == 12
                && vlm.vision_config.num_attention_heads == 12
                && vlm.vision_config.image_size == 512
                && vlm.vision_config.patch_size == 16
                && vlm.scale_factor == 4,
            "only the SmolVLM2-500M architecture is supported"
        );
        let vision_size = vlm.vision_config.image_size;
        for (_, feature) in cfg.image_features() {
            let output = cfg
                .resize_imgs_with_padding
                .unwrap_or([feature.shape[2], feature.shape[1]]);
            ensure!(
                output == [vision_size, vision_size],
                "SmolVLA prepared images must be {vision_size}x{vision_size}"
            );
            let ratio = (feature.shape[2] as f64 / output[0] as f64)
                .max(feature.shape[1] as f64 / output[1] as f64);
            ensure!(
                (feature.shape[1] as f64 / ratio) >= 1. && (feature.shape[2] as f64 / ratio) >= 1.,
                "image aspect ratio produces an empty resized image"
            );
        }
        let expert_dim =
            (vlm.text_config.hidden_size as f64 * cfg.expert_width_multiplier) as usize;
        ensure!(
            expert_dim >= 4 && expert_dim.is_multiple_of(2),
            "expert hidden size must be even and at least 4"
        );
        ensure!(
            !cfg.add_image_special_tokens,
            "SmolVLA add_image_special_tokens=true is not supported"
        );
        let model = SmolVlaFlowMatchingConfig::new(cfg.clone(), vlm).init(device);
        Ok(SmolVlaPolicy {
            model,
            cfg,
            device: device.clone(),
        })
    }

    /// 严格加载原生模型 record，外部 safetensors 必须先转换。
    pub fn load_model(&self, device: &Device) -> Result<SmolVlaPolicy> {
        let policy = self.init(device)?;
        let record = burn::store::ModuleRecord::load(self.path.join("model.bpk"))
            .context("reading native SmolVLA model.bpk; convert the LeRobot checkpoint first")?;
        policy.load_record(record)
    }

    pub fn import_configured(
        &self,
        features: &[FeatureSpec],
        action: &FeatureSpec,
        device: &Device,
    ) -> Result<(SmolVlaPolicy, Pipeline, Pipeline, Vec<String>)> {
        let cfg = config::parse(&self.path.join("config.json"))?;
        processing::validate_contract(&cfg, features, action)?;
        let tokenizer_dir = self
            .vlm_path
            .clone()
            .unwrap_or_else(|| self.path.join("vlm"));
        let (preprocessor, postprocessor) =
            processing::import_pipelines(&self.path, &cfg, &tokenizer_dir)?;
        let (policy, unused) = self.import_lerobot(device)?;
        Ok((policy, preprocessor, postprocessor, unused))
    }

    /// 转换边界导入 LeRobot 参数；未使用参数留给 checkpoint 转换器保存。
    pub fn import_lerobot(&self, device: &Device) -> Result<(SmolVlaPolicy, Vec<String>)> {
        let mut policy = self.init(device)?;
        let mut store = SafetensorsStore::from_file(self.path.join("model.safetensors"))
            .with_from_adapter(PyTorchToBurnAdapter.chain(FloatCastAdapter::to(DType::F32)))
            .with_key_remapping(r"^model\.", "")
            .allow_partial(true);
        let applied = policy
            .model
            .load_from(&mut store)
            .context("loading SmolVLA weights")?;
        ensure!(
            applied.missing.is_empty() && applied.errors.is_empty(),
            "incomplete SmolVLA weights: missing={:?} errors={:?}",
            applied.missing,
            applied.errors
        );
        tracing::info!(
            applied = applied.applied.len(),
            unused = applied.unused.len(),
            device = ?policy.device,
            "SmolVLA weights imported"
        );
        Ok((policy, applied.unused))
    }
}

impl SmolVlaPolicy {
    /// 应用已加载的原生 record；结构由 `SmolVlaPolicyConfig::init` 构建。
    pub fn load_record(self, record: burn::store::ModuleRecord) -> Result<Self> {
        let model = self
            .model
            .try_load_record(record)
            .context("restoring SmolVLA model record")?;
        Ok(Self {
            model,
            cfg: self.cfg,
            device: self.device,
        })
    }

    /// 保存当前模型参数与参数身份，pipeline 由 checkpoint 独立保存。
    pub fn save_native(&self, path: &Path) -> Result<()> {
        self.model.clone().into_record().save(path)?;
        Ok(())
    }

    /// 导出当前动作预测模块的 LeRobot 参数名及 PyTorch 布局。
    pub fn export_lerobot(&self, path: &Path) -> Result<()> {
        let mut store = SafetensorsStore::from_file(path)
            .with_to_adapter(burn_store::BurnToPyTorchAdapter)
            .with_key_remapping(r"^", "model.");
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

impl Policy for SmolVlaPolicy {
    fn predict(&self, mut batch: Batch) -> Result<Batch> {
        let state = batch.take("observation.state")?.into_float2()?;
        ensure!(
            state.dims() == [1, self.cfg.max_state_dim],
            "SmolVLA requires one prepared state sample"
        );
        let tokens = match batch.take("observation.language.tokens")? {
            Data::Int2(tokens) => {
                ensure!(tokens.dims()[0] == 1, "SmolVLA requires one task sample");
                tokens.into_data().iter::<i64>().collect::<Vec<_>>()
            }
            _ => anyhow::bail!("SmolVLA requires prepared integer language tokens"),
        };
        let valid = match batch.take("observation.language.attention_mask")? {
            Data::Bool2(mask) => {
                ensure!(
                    mask.dims() == [1, tokens.len()],
                    "SmolVLA language mask shape differs from tokens"
                );
                mask.into_data().iter::<bool>().collect::<Vec<_>>()
            }
            _ => anyhow::bail!("SmolVLA requires prepared boolean language mask"),
        };
        let images = self
            .cfg
            .image_features()
            .map(|(name, _)| {
                let image = batch.take(name)?.into_float4()?;
                ensure!(
                    image.dims()[0] == 1,
                    "SmolVLA requires one prepared image sample"
                );
                Ok(image)
            })
            .collect::<Result<Vec<_>>>()?;
        let noise = Tensor::random(
            [1, self.cfg.chunk_size, self.cfg.max_action_dim],
            Distribution::Normal(0., 1.),
            &self.device,
        );
        let action = self
            .model
            .sample(images, &tokens, &valid, state, noise, &self.cfg);
        Ok(Batch {
            values: [("action".into(), Data::Float3(action))].into(),
            ..Batch::default()
        })
    }
}
