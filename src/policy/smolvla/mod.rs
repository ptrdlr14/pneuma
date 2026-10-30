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
use rmpv::Value;
use serde::Deserialize;

use crate::codec;
use crate::policy::{Chunk, PolicyRunner};
use crate::protocol::{Capabilities, ErrorCode, ProtocolError, body_get};

use config::SmolVlaConfig;
use processing::Processor;
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
    processor: Processor,
    device: Device,
}

impl SmolVlaPolicy {
    pub fn load_configured(
        path: &Path,
        vlm_path: Option<&Path>,
        backend: crate::config::Device,
        warmup_calls: usize,
        features: &[crate::protocol::FeatureSpec],
        action: &crate::protocol::FeatureSpec,
    ) -> Result<Self> {
        let cfg = config::parse(&path.join("config.json"))?;
        processing::validate_contract(&cfg, features, action)?;
        let vlm_path = vlm_path
            .map(Path::to_path_buf)
            .unwrap_or_else(|| path.join("vlm"));
        let mut vlm: VlmConfig = serde_json::from_slice(
            &std::fs::read(vlm_path.join("config.json"))
                .context("reading local SmolVLM config; set model.vlm_path or --vlm-path")?,
        )?;
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
        let processor = Processor::load(path, &cfg, &vlm_path)?;
        let device = match backend {
            crate::config::Device::Auto => Device::default(),
            crate::config::Device::Flex => Device::flex(),
            crate::config::Device::Metal => Device::metal(Default::default()),
        };
        let mut model = VLAFlowMatching::init(&cfg, &vlm, &device);
        let mut store = SafetensorsStore::from_file(path.join("model.safetensors"))
            .with_from_adapter(PyTorchToBurnAdapter.chain(FloatCastAdapter::to(DType::F32)))
            .with_key_remapping(r"^model\.", "")
            .allow_partial(true);
        let applied = model
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
            ?device,
            "SmolVLA weights loaded"
        );
        let policy = Self {
            model,
            cfg,
            processor,
            device,
        };
        for _ in 0..warmup_calls {
            let state = vec![0.; policy.cfg.state_dim()];
            let images: Vec<_> = policy
                .cfg
                .image_features()
                .map(|(_, feature)| {
                    let [_, h, w] = feature.shape[..] else {
                        unreachable!("validated image shape")
                    };
                    policy
                        .processor
                        .image(&vec![0; h * w * 3], h, w, &policy.device)
                })
                .collect();
            let (tokens, valid) = policy.processor.tokenize("")?;
            let noise = Tensor::zeros(
                [1, policy.cfg.chunk_size, policy.cfg.max_action_dim],
                &policy.device,
            );
            let values = policy
                .model
                .sample(
                    images,
                    &tokens,
                    &valid,
                    policy.processor.state(&state, &policy.device),
                    noise,
                    &policy.cfg,
                )
                .into_data()
                .try_to_vec::<f32>()?;
            ensure!(
                values.iter().all(|x| x.is_finite()),
                "SmolVLA warmup produced non-finite actions"
            );
        }
        Ok(policy)
    }

    pub fn horizon(&self) -> (u64, u64) {
        (self.cfg.chunk_size as u64, self.cfg.n_action_steps as u64)
    }
}

impl PolicyRunner for SmolVlaPolicy {
    fn validate_against(&self, caps: &Capabilities) -> std::result::Result<(), ProtocolError> {
        processing::validate_contract(&self.cfg, &caps.features, &caps.action_feature)
            .map_err(|error| ProtocolError::new(ErrorCode::Incompatible, error.to_string()))
    }

    fn predict(
        &self,
        caps: &Capabilities,
        features: &Value,
        task: Option<&str>,
    ) -> std::result::Result<Chunk, ProtocolError> {
        let execute = || -> Result<Chunk> {
            let task = task.context("SmolVLA requires observation task text")?;
            ensure!(task.chars().count() <= 4096, "task exceeds max_input_chars");
            let (tokens, valid) = self.processor.tokenize(task)?;
            let state = codec::decode_record(
                body_get(features, "observation.state").context("missing state")?,
            )?;
            let state: Vec<_> = state
                .data
                .as_chunks::<4>()
                .0
                .iter()
                .map(|bytes| f32::from_le_bytes(*bytes))
                .collect();
            let mut images = Vec::new();
            for (name, _) in self.cfg.image_features() {
                let image = codec::decode_record(
                    body_get(features, name).with_context(|| format!("missing image {name}"))?,
                )?;
                images.push(self.processor.image(
                    image.data,
                    image.shape[0] as usize,
                    image.shape[1] as usize,
                    &self.device,
                ));
            }
            let noise = Tensor::random(
                [1, self.cfg.chunk_size, self.cfg.max_action_dim],
                Distribution::Normal(0., 1.),
                &self.device,
            );
            let output = self.model.sample(
                images,
                &tokens,
                &valid,
                self.processor.state(&state, &self.device),
                noise,
                &self.cfg,
            );
            let flat = output.into_data().try_to_vec::<f32>()?;
            let steps = usize::try_from(caps.execution_steps)?;
            ensure!(
                steps > 0 && steps <= self.cfg.chunk_size,
                "invalid execution_steps"
            );
            let trimmed: Vec<_> = flat
                .chunks_exact(self.cfg.max_action_dim)
                .take(steps)
                .flat_map(|row| row[..self.cfg.action_dim()].iter().copied())
                .collect();
            let actions = self.processor.unnormalize(&trimmed);
            ensure!(
                actions.iter().all(|x| x.is_finite()),
                "SmolVLA produced non-finite actions"
            );
            Ok(Chunk {
                canonical_actions: actions.iter().flat_map(|v| v.to_le_bytes()).collect(),
                execution_steps: caps.execution_steps,
            })
        };
        execute().map_err(|error| {
            ProtocolError::new(
                ErrorCode::Execution,
                format!("SmolVLA inference: {error:#}"),
            )
        })
    }
}
