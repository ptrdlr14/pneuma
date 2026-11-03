//! Burn 原生 SmolVLA：视觉和语言前缀缓存、动作专家及 flow-matching Euler 推理。

mod config;
mod processing;
#[cfg(test)]
mod tests;
mod transformer;
mod vision;

use std::path::Path;

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
use transformer::{ExpertModel, KvCache, TextConfig, TextModel};
use vision::{Connector, VisionConfig, VisionModel};

#[derive(Deserialize)]
struct VlmConfig {
    text_config: TextConfig,
    vision_config: VisionConfig,
    scale_factor: usize,
}

#[derive(Module, Debug)]
struct VlmModel {
    vision_model: VisionModel,
    connector: Connector,
    text_model: TextModel,
}

#[derive(Module, Debug)]
struct Vlm {
    model: VlmModel,
}

#[derive(Module, Debug)]
struct VlmWithExpert {
    vlm: Vlm,
    lm_expert: ExpertModel,
}

#[derive(Module, Debug)]
struct VLAFlowMatching {
    vlm_with_expert: VlmWithExpert,
    state_proj: Linear,
    action_in_proj: Linear,
    action_out_proj: Linear,
    action_time_mlp_in: Linear,
    action_time_mlp_out: Linear,
}

impl VLAFlowMatching {
    fn init(cfg: &SmolVlaConfig, vlm: &VlmConfig, device: &Device) -> Self {
        let text = &vlm.text_config;
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
        Self {
            vlm_with_expert: VlmWithExpert {
                vlm: Vlm {
                    model: VlmModel {
                        vision_model: VisionModel::init(&vlm.vision_config, device),
                        connector: Connector::init(&vlm.vision_config, text.hidden_size, device),
                        text_model: TextModel::init(text, device),
                    },
                },
                lm_expert: ExpertModel::init(&expert, cross, device),
            },
            state_proj: LinearConfig::new(cfg.max_state_dim, text.hidden_size).init(device),
            action_in_proj: LinearConfig::new(cfg.max_action_dim, expert_dim).init(device),
            action_out_proj: LinearConfig::new(expert_dim, cfg.max_action_dim).init(device),
            action_time_mlp_in: LinearConfig::new(expert_dim * 2, expert_dim).init(device),
            action_time_mlp_out: LinearConfig::new(expert_dim, expert_dim).init(device),
        }
    }

    fn prefix(
        &self,
        images: Vec<Tensor<4>>,
        tokens: &[i64],
        language_valid: &[bool],
        state: Tensor<2>,
        cfg: &SmolVlaConfig,
    ) -> (Vec<KvCache>, Vec<bool>) {
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

    fn velocity(
        &self,
        x: Tensor<3>,
        timestep: f32,
        cache: &[KvCache],
        valid: &[bool],
        cfg: &SmolVlaConfig,
    ) -> Tensor<3> {
        let action = self.action_in_proj.forward(x);
        let [batch, steps, dim] = action.dims();
        let half = dim / 2;
        let angles: Vec<_> = (0..half)
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
        let time = Tensor::<3>::from_data(TensorData::new(time, [1, 1, dim]), &action.device())
            .expand([batch, steps, dim]);
        let suffix = self.action_time_mlp_out.forward(silu(
            self.action_time_mlp_in
                .forward(Tensor::cat(vec![action, time], 2)),
        ));
        self.action_out_proj
            .forward(self.vlm_with_expert.lm_expert.forward(suffix, cache, valid))
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
        let (cache, valid) = self.prefix(images, tokens, valid, state, cfg);
        let mut x = noise;
        let dt = -1.0 / cfg.num_steps as f64;
        for step in 0..cfg.num_steps {
            let time = (1.0 + step as f64 * dt) as f32;
            let velocity = self.velocity(x.clone(), time, &cache, &valid, cfg);
            x = x + velocity * dt;
        }
        x
    }
}

pub struct SmolVlaPolicy {
    model: VLAFlowMatching,
    cfg: SmolVlaConfig,
    device: Device,
}

impl SmolVlaPolicy {
    pub fn import_configured(
        path: &Path,
        vlm_path: Option<&Path>,
        backend: crate::config::Device,
        features: &[FeatureSpec],
        action: &FeatureSpec,
    ) -> Result<(Self, Pipeline, Pipeline, Vec<String>)> {
        let cfg = config::parse(&path.join("config.json"))?;
        processing::validate_contract(&cfg, features, action)?;
        let tokenizer_dir = vlm_path
            .map(Path::to_path_buf)
            .unwrap_or_else(|| path.join("vlm"));
        let (preprocessor, postprocessor) =
            processing::import_pipelines(path, &cfg, &tokenizer_dir)?;
        let (policy, unused) = Self::import_lerobot(path, vlm_path, backend)?;
        Ok((policy, preprocessor, postprocessor, unused))
    }

    /// 严格加载原生模型 record，外部 safetensors 必须先转换。
    pub fn load_model(
        path: &Path,
        vlm_path: Option<&Path>,
        backend: crate::config::Device,
    ) -> Result<Self> {
        let mut policy = Self::init_model(path, vlm_path, backend)?;
        let record = burn::store::ModuleRecord::load(path.join("model.bpk"))
            .context("reading native SmolVLA model.bpk; convert the LeRobot checkpoint first")?;
        policy.model = policy
            .model
            .try_load_record(record)
            .context("restoring SmolVLA model record")?;
        Ok(policy)
    }

    fn init_model(
        path: &Path,
        vlm_path: Option<&Path>,
        backend: crate::config::Device,
    ) -> Result<Self> {
        let cfg = config::parse(&path.join("config.json"))?;
        let vlm_path = vlm_path
            .map(Path::to_path_buf)
            .unwrap_or_else(|| path.join("vlm"));
        let mut vlm: VlmConfig =
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
        let device = backend.into();
        let model = VLAFlowMatching::init(&cfg, &vlm, &device);
        Ok(Self { model, cfg, device })
    }

    /// 转换边界导入 LeRobot 参数；未使用参数留给 checkpoint 转换器保存。
    pub fn import_lerobot(
        path: &Path,
        vlm_path: Option<&Path>,
        backend: crate::config::Device,
    ) -> Result<(Self, Vec<String>)> {
        let mut policy = Self::init_model(path, vlm_path, backend)?;
        let mut store = SafetensorsStore::from_file(path.join("model.safetensors"))
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
