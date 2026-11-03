//! Cosmos3 Edge：世界动作模型。
//!
//! 结构为 Nemotron-3-Dense-VL 与动作 DiT 的 MoT 主干，加 Wan2.2 VAE 和 UniPC 采样。
//! 当前进度：配置解析、SigLIP2 视觉塔、patch merger、MoT 文本主干、Wan2.2 VAE 编码器、
//! 生成流嵌入、uniPC 采样、联合位置索引、整网接线、运行时循环与推理主体已实现。处理器、
//! 合约与协议接入尚未实现，服务不注册该策略。

pub(crate) mod canvas;
pub mod config;
mod gen_embed;
mod net;
mod positions;
mod processing;
mod prompt;
mod runtime;
mod sampler;
mod text;
mod vae;
mod vision;

use std::path::Path;

use anyhow::{Context, Result, ensure};
use burn::module::Module;
use burn::tensor::{Device, Tensor};

use crate::{
    config as app,
    policy::Policy,
    processor::{Batch, Data},
};

use self::config::Cosmos3Config;
pub(crate) use prompt::{Canvas, caption, text as prompt_text};

use positions::{GenGrid, GenPlan, append_generation, understanding_positions};
use runtime::{Cosmos3Runtime, FlowState, Prompt, SequencePlan};

/// Cosmos3 Edge 策略：VAE 编码条件帧、生成流嵌入、uniPC 采样。
pub struct Cosmos3Policy {
    runtime: Cosmos3Runtime,
    cfg: Cosmos3Config,
    device: Device,
    /// 动作维度（真实通道数），由部署合约给出。
    action_dim: usize,
}

impl Cosmos3Policy {
    /// 从原生 bpk 目录加载。
    pub fn load_model(path: &Path, backend: app::Device) -> Result<Self> {
        let cfg = config::parse(&path.join("config.json"))?;
        let device = backend.into();
        let runtime = Cosmos3Runtime::init(&cfg, &device);
        let record = burn::store::ModuleRecord::load(path.join("model.bpk"))
            .context("reading native cosmos3 model.bpk; convert the LeRobot checkpoint first")?;
        let runtime = runtime
            .try_load_record(record)
            .context("restoring cosmos3 model record")?;
        Ok(Self {
            runtime,
            cfg,
            device,
            action_dim: 7,
        })
    }

    /// 从 LeRobot checkpoint 导入权重并构建两条管线。
    pub fn import_configured(
        path: &Path,
        backend: app::Device,
        features: &[crate::contract::FeatureSpec],
        action: &crate::contract::FeatureSpec,
    ) -> Result<(
        Self,
        crate::processor::Pipeline,
        crate::processor::Pipeline,
        Vec<String>,
    )> {
        let cfg = config::parse(&path.join("config.json"))?;
        processing::validate_contract(&cfg, features, action)?;
        let tokenizer_dir = path.join("tokenizer");
        let (pre, post) = processing::import_pipelines(&cfg, &tokenizer_dir, features, action)?;
        let device = backend.into();
        let (runtime, unused) = Cosmos3Runtime::import_lerobot(path, &cfg, &device)?;
        let action_dim = action.shape[0] as usize;
        Ok((
            Self {
                runtime,
                cfg,
                device,
                action_dim,
            },
            pre,
            post,
            unused,
        ))
    }

    /// 以 LeRobot 布局导出权重，供转换时核对张量清单。
    pub fn export_lerobot(&self, path: &Path) -> Result<()> {
        use burn_store::{BurnToPyTorchAdapter, ModuleSnapshot, SafetensorsStore};

        let mut store = SafetensorsStore::from_file(path).with_to_adapter(BurnToPyTorchAdapter);
        store = store
            .with_key_remapping(
                r"^net\.text\.",
                "model.network.mot_backbone.model.language_model.",
            )
            .with_key_remapping(r"^net\.", "model.network.mot_backbone.model.")
            .with_key_remapping(
                r"^vae\.(encoder\.down_blocks\.\d+\.downsampler)\.resample\.",
                "vae.$1.resample.1.",
            )
            .with_key_remapping(r"^vae\.", "model.network.vae.")
            .with_key_remapping(
                r"^embeddings\.action_modality_embed$",
                "model.network.action_modality_embed.weight",
            )
            .with_key_remapping(r"^embeddings\.", "model.network.")
            // Burn 把 `gamma` 写成 PyTorch 的 `weight`；VAE 源文件用的是 `gamma`。
            .with_key_remapping(
                r"^(model\.network\.vae\..*)\.(norm1|norm2|norm_out|norm)\.weight$",
                "$1.$2.gamma",
            );
        self.runtime.save_into(&mut store)?;
        Ok(())
    }

    /// 部署合约给出的动作维度。
    pub fn with_action_dim(mut self, action_dim: usize) -> Self {
        self.action_dim = action_dim;
        self
    }

    /// 预测步数与执行步数。
    pub fn horizon(&self) -> (u64, u64) {
        (self.cfg.chunk_size as u64, self.cfg.n_action_steps as u64)
    }

    pub fn device(&self) -> Device {
        self.device.clone()
    }

    /// 保存为原生 bpk。
    pub fn save_native(&self, path: &Path) -> Result<()> {
        self.runtime.save(path)
    }
}

impl Cosmos3Policy {
    /// 由 token 序列构造理解流位置，再追加生成流位置。
    fn prompt(
        &self,
        tokens: Vec<i64>,
        grid: GenGrid,
        action_offset: i64,
        text_len: usize,
    ) -> Prompt {
        let kinds = vec![0u8; text_len];
        let text = understanding_positions(&kinds, &[], 2);
        let plan = GenPlan {
            batch: 1,
            text_len,
            grid,
            second_per_frame: (self.cfg.vae.scale_factor_temporal as f64
                / self.cfg.fps.unwrap_or(20.0)) as f32,
            num_actions: self.cfg.chunk_size + usize::from(self.cfg.use_state),
            action_offset,
        };
        let positions = append_generation([&text[0], &text[1], &text[2]], &plan, &self.cfg);
        Prompt {
            tokens,
            positions,
            image: None,
            image_span: 0..0,
        }
    }
}

impl Policy for Cosmos3Policy {
    fn predict(&self, batch: Batch) -> Result<Batch> {
        self.predict_with_noise(batch, None)
    }
}

impl Cosmos3Policy {
    /// 推理主体；`noise` 为 `None` 时按标准正态采样，测试可传入固定噪声。
    pub(crate) fn predict_with_noise(
        &self,
        mut batch: Batch,
        noise: Option<FlowState>,
    ) -> Result<Batch> {
        let device = self.device.clone();
        let canvas = match batch.take("observation.images.canvas")? {
            Data::Float4(canvas) => canvas,
            _ => anyhow::bail!("cosmos3 requires a float canvas image"),
        };
        let state = match batch.take("observation.state")? {
            Data::Float2(state) => state,
            _ => anyhow::bail!("cosmos3 requires a float state vector"),
        };
        let tokens: Vec<i64> = match batch.take("observation.language.tokens")? {
            Data::Int2(tokens) => tokens.into_data().iter::<i64>().collect(),
            _ => anyhow::bail!("cosmos3 requires prepared integer tokens"),
        };
        let negative: Option<Vec<i64>> =
            match batch.values.remove("observation.language.negative_tokens") {
                Some(Data::Int2(tokens)) => Some(tokens.into_data().iter::<i64>().collect()),
                Some(other) => {
                    anyhow::bail!("negative tokens must be integer, got {:?}", other.shape())
                }
                None => None,
            };
        ensure!(
            self.cfg.guidance_scale == 1.0 || negative.is_some(),
            "cosmos3 needs observation.language.negative_tokens when guidance_scale != 1"
        );

        let cfg = &self.cfg;
        let plan = SequencePlan::new(cfg);
        let latent_frames = plan.video_condition.len();
        let action_count = plan.action_condition.len();

        // 条件帧编码后放到第 0 个隐变量帧，其余为零。
        let encoded = self.runtime.encode_condition(canvas, cfg, &device)?;
        let shape = encoded.dims();
        let mut video = Tensor::<5>::zeros(
            [1, cfg.vae.z_dim, latent_frames, shape[3], shape[4]],
            &device,
        );
        video = video.slice_assign(
            [0..1, 0..cfg.vae.z_dim, 0..1, 0..shape[3], 0..shape[4]],
            encoded,
        );

        // 动作序列：状态占第一个 token。
        let mut actions = Tensor::<3>::zeros([1, action_count, cfg.max_action_dim], &device);
        let state = pad_state(&state, cfg.max_action_dim, &device);
        actions = actions.slice_assign([0..1, 0..1, 0..cfg.max_action_dim], state);

        let noise = noise.unwrap_or_else(|| FlowState {
            video: Tensor::random(
                [1, cfg.vae.z_dim, latent_frames, shape[3], shape[4]],
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &device,
            ),
            actions: Tensor::random(
                [1, action_count, cfg.max_action_dim],
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &device,
            ),
        });
        let start = self
            .runtime
            .add_noise(&FlowState { video, actions }, &noise, &plan, &device);

        let grid = GenGrid {
            frames: latent_frames,
            height: shape[3],
            width: shape[4],
        };
        let text_len = tokens.len();
        let prompt = self.prompt(tokens, grid, plan.action_offset, text_len);
        let negative = negative.map(|tokens| {
            let length = tokens.len();
            self.prompt(tokens, grid, plan.action_offset, length)
        });
        let rope = Cosmos3Runtime::rope(cfg);
        let velocity = self.runtime.guided_velocity(
            cfg,
            &prompt,
            negative.as_ref(),
            &plan,
            &rope,
            cfg.domain_id.min(cfg.num_embodiment_domains - 1),
            self.action_dim,
            &device,
        );
        let sampled = self
            .runtime
            .sample(cfg, velocity, start, cfg.num_inference_steps, &device);
        let predicted = if cfg.use_state {
            sampled
                .actions
                .slice([0..1, 1..action_count, 0..self.action_dim])
        } else {
            sampled
                .actions
                .slice([0..1, 0..action_count, 0..self.action_dim])
        };
        batch.values.clear();
        batch
            .values
            .insert("action".into(), Data::Float3(predicted));
        Ok(batch)
    }
}

/// 状态补零到 `width`，超出时截断。
fn pad_state(state: &Tensor<2>, width: usize, device: &Device) -> Tensor<3> {
    let [_, dim] = state.dims();
    let mut values = state.clone().reshape([1, 1, dim]);
    if dim < width {
        let zeros = burn::tensor::Tensor::<3>::zeros([1, 1, width - dim], device);
        values = Tensor::cat(vec![values, zeros], 2);
    } else if dim > width {
        values = values.slice([0..1, 0..1, 0..width]);
    }
    values
}

#[cfg(test)]
mod smoke {
    use super::*;
    use burn::tensor::TensorData;
    use std::collections::BTreeMap;

    /// 随机权重下的形状与数值冒烟测试：验证整条推理能跑通且输出有限。
    #[test]
    #[ignore = "随机权重冒烟测试，耗时较长"]
    fn predicts_finite_actions() {
        let device = Device::flex();
        let cfg = Cosmos3Config {
            guidance_scale: 1.0,
            ..Cosmos3Config::default()
        };
        let runtime = Cosmos3Runtime::init(&cfg, &device);
        let policy = Cosmos3Policy {
            runtime,
            cfg,
            device: device.clone(),
            action_dim: 7,
        };
        let mut values = BTreeMap::new();
        values.insert(
            "observation.images.canvas".to_string(),
            Data::Float4(Tensor::zeros([1, 3, 256, 256], &device)),
        );
        values.insert(
            "observation.state".to_string(),
            Data::Float2(Tensor::zeros([1, 7], &device)),
        );
        values.insert(
            "observation.language.tokens".to_string(),
            Data::Int2(Tensor::<2, burn::tensor::Int>::from_data(
                TensorData::new(vec![2i64, 5, 6, 7, 8], [1, 5]),
                &device,
            )),
        );
        let batch = Batch {
            values,
            tasks: Vec::new(),
        };
        let out = policy.predict(batch).expect("predict");
        let action = match &out.values["action"] {
            Data::Float3(action) => action,
            other => panic!("unexpected action type {:?}", other.shape()),
        };
        assert_eq!(action.dims(), [1, 32, 7]);
        let values: Vec<f32> = action.clone().into_data().try_to_vec().expect("action");
        assert!(
            values.iter().all(|value| value.is_finite()),
            "non-finite action"
        );
    }
}

#[cfg(test)]
mod native {
    use super::*;
    use burn::tensor::{Int, TensorData};
    use std::collections::BTreeMap;

    /// 用转换后的原生产物跑一次完整推理，确认权重与管线能落地。
    #[test]
    #[ignore = "需要转换后的产物，耗时数分钟"]
    fn native_artifact_predicts_finite_actions() {
        let directory = std::path::PathBuf::from(
            std::env::var("PNEUMA_COSMOS3_NATIVE").expect("PNEUMA_COSMOS3_NATIVE"),
        );
        let policy = Cosmos3Policy::load_model(&directory.join("model"), app::Device::Flex)
            .expect("loading native artifact")
            .with_action_dim(7);
        let device = policy.device();
        assert_eq!(policy.horizon(), (32, 32));

        let canvas = Data::Float4(Tensor::zeros([1, 3, 256, 256], &device));
        let state = Data::Float2(Tensor::zeros([1, 8], &device));
        let ids =
            Tensor::<2, Int>::from_data(TensorData::new(vec![2i64, 5, 6, 7, 8], [1, 5]), &device);
        let batch = Batch {
            values: BTreeMap::from([
                ("observation.images.canvas".to_string(), canvas),
                ("observation.state".to_string(), state),
                (
                    "observation.language.tokens".to_string(),
                    Data::Int2(ids.clone()),
                ),
                (
                    "observation.language.negative_tokens".to_string(),
                    Data::Int2(ids),
                ),
            ]),
            tasks: Vec::new(),
        };
        let noise = FlowState {
            video: Tensor::zeros([1, 48, 9, 16, 16], &device),
            actions: Tensor::zeros([1, 33, 64], &device),
        };
        let out = policy
            .predict_with_noise(batch, Some(noise))
            .expect("predict");
        let action = match &out.values["action"] {
            Data::Float3(action) => action.clone(),
            other => panic!("unexpected action type {:?}", other.shape()),
        };
        assert_eq!(action.dims(), [1, 32, 7]);
        let values: Vec<f32> = action.into_data().try_to_vec().expect("action");
        assert!(
            values.iter().all(|value| value.is_finite()),
            "non-finite action"
        );
    }
}

#[cfg(test)]
mod parity {
    use super::*;
    use burn::tensor::{Int, TensorData};
    use std::collections::BTreeMap;

    /// 第一步速度对比：定位误差出现在哪一段。
    #[test]
    #[ignore = "需要转换产物与官方参照，耗时数分钟"]
    fn first_velocity_matches_official() {
        let directory = std::path::PathBuf::from(
            std::env::var("PNEUMA_COSMOS3_NATIVE").expect("PNEUMA_COSMOS3_NATIVE"),
        );
        let parity = std::path::PathBuf::from(
            std::env::var("PNEUMA_COSMOS3_GENERATE").expect("PNEUMA_COSMOS3_GENERATE"),
        );
        let policy = Cosmos3Policy::load_model(&directory.join("model"), app::Device::Flex)
            .expect("loading native artifact")
            .with_action_dim(7);
        let device = policy.device();
        let cfg = &policy.cfg;
        let plan = SequencePlan::new(cfg);

        let canvas = Tensor::<4>::zeros([1, 3, 256, 256], &device);
        let encoded = policy
            .runtime
            .encode_condition(canvas, cfg, &device)
            .expect("encode");
        let shape = encoded.dims();
        let latent_frames = plan.video_condition.len();
        let mut video = Tensor::<5>::zeros(
            [1, cfg.vae.z_dim, latent_frames, shape[3], shape[4]],
            &device,
        );
        video = video.slice_assign(
            [0..1, 0..cfg.vae.z_dim, 0..1, 0..shape[3], 0..shape[4]],
            encoded,
        );
        let mut actions = Tensor::<3>::zeros(
            [1, plan.action_condition.len(), cfg.max_action_dim],
            &device,
        );
        actions = actions.slice_assign(
            [0..1, 0..1, 0..cfg.max_action_dim],
            Tensor::<3>::zeros([1, 1, cfg.max_action_dim], &device),
        );
        let noise = FlowState {
            video: Tensor::zeros(
                [1, cfg.vae.z_dim, latent_frames, shape[3], shape[4]],
                &device,
            ),
            actions: Tensor::zeros(
                [1, plan.action_condition.len(), cfg.max_action_dim],
                &device,
            ),
        };
        let clean = FlowState { video, actions };
        let start = policy.runtime.add_noise(&clean, &noise, &plan, &device);

        let tokens: Vec<i64> = vec![2, 5, 6, 7, 8];
        let grid = positions::GenGrid {
            frames: latent_frames,
            height: shape[3],
            width: shape[4],
        };
        let prompt = policy.prompt(tokens, grid, plan.action_offset, 5);
        let rope = Cosmos3Runtime::rope(cfg);
        let (hidden, video_tokens) =
            policy
                .runtime
                .gen_hidden(&start, 999.0, &prompt, &plan, &rope, cfg.domain_id, &device);
        let action_hidden = hidden
            .clone()
            .slice([0..1, video_tokens..609, 0..hidden.dims()[2]]);
        let hid: Vec<f32> = action_hidden.into_data().try_to_vec().expect("hidden");
        let expected_hidden: Vec<f32> = {
            let bytes = std::fs::read(parity.join("action_hidden.bin")).expect("hidden ref");
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|value| f32::from_le_bytes(*value))
                .collect()
        };
        let max_hidden = hid
            .iter()
            .zip(&expected_hidden)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("action_hidden: max {max_hidden:.3e}");

        let reference_actions: Vec<f32> = {
            let bytes = std::fs::read(parity.join("velocity_actions.bin")).expect("actions ref");
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|value| f32::from_le_bytes(*value))
                .collect()
        };
        for domain in 0..8usize {
            let projected = policy
                .runtime
                .actions_from_hidden(
                    hidden
                        .clone()
                        .slice([0..1, video_tokens..609, 0..hidden.dims()[2]]),
                    domain,
                )
                .into_data()
                .try_to_vec::<f32>()
                .expect("projection");
            let max = projected
                .iter()
                .zip(&reference_actions)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            println!("domain {domain}: max {max:.3e}");
        }

        let velocity = policy.runtime.velocities(
            cfg,
            &start,
            999.0,
            &prompt,
            &plan,
            &rope,
            cfg.domain_id,
            &device,
        );

        let read = |name: &str| -> Vec<f32> {
            let bytes = std::fs::read(parity.join(name)).expect("reference");
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|value| f32::from_le_bytes(*value))
                .collect()
        };
        let video_got: Vec<f32> = velocity.video.into_data().try_to_vec().expect("video");
        let actions_got: Vec<f32> = velocity.actions.into_data().try_to_vec().expect("actions");
        for (name, got) in [
            ("velocity_video.bin", video_got),
            ("velocity_actions.bin", actions_got),
        ] {
            let expected = read(name);
            assert_eq!(got.len(), expected.len(), "{name} length");
            let max = got
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let mean = got
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .sum::<f32>()
                / got.len() as f32;
            println!("{name}: max {max:.3e} mean {mean:.3e}");
            if name.starts_with("velocity_actions") {
                let channels = 64usize;
                for token in 0..2 {
                    let base = token * channels;
                    println!(
                        "token {token} got {:?} want {:?}",
                        &got[base..base + 4],
                        &expected[base..base + 4]
                    );
                }
                let mut per_channel = vec![0.0f32; channels];
                for token in 0..33 {
                    for channel in 0..channels {
                        per_channel[channel] +=
                            got[token * channels + channel] - expected[token * channels + channel];
                    }
                }
                for value in &mut per_channel {
                    *value /= 33.0;
                }
                println!("per-channel mean diff {:?}", &per_channel[..6]);
            }
        }
    }

    /// 与官方 `generate` 对比：相同输入与噪声下的动作块。
    #[test]
    #[ignore = "需要转换产物与官方参照，耗时数分钟"]
    fn actions_match_official_generate() {
        let directory = std::path::PathBuf::from(
            std::env::var("PNEUMA_COSMOS3_NATIVE").expect("PNEUMA_COSMOS3_NATIVE"),
        );
        let parity = std::path::PathBuf::from(
            std::env::var("PNEUMA_COSMOS3_GENERATE").expect("PNEUMA_COSMOS3_GENERATE"),
        );
        let policy = Cosmos3Policy::load_model(&directory.join("model"), app::Device::Flex)
            .expect("loading native artifact")
            .with_action_dim(7);
        let device = policy.device();

        let ids =
            Tensor::<2, Int>::from_data(TensorData::new(vec![2i64, 5, 6, 7, 8], [1, 5]), &device);
        let batch = Batch {
            values: BTreeMap::from([
                (
                    "observation.images.canvas".to_string(),
                    Data::Float4(Tensor::zeros([1, 3, 256, 256], &device)),
                ),
                (
                    "observation.state".to_string(),
                    Data::Float2(Tensor::zeros([1, 7], &device)),
                ),
                (
                    "observation.language.tokens".to_string(),
                    Data::Int2(ids.clone()),
                ),
                (
                    "observation.language.negative_tokens".to_string(),
                    Data::Int2(ids),
                ),
            ]),
            tasks: Vec::new(),
        };
        let noise = FlowState {
            video: Tensor::zeros([1, 48, 9, 16, 16], &device),
            actions: Tensor::zeros([1, 33, 64], &device),
        };
        let out = policy
            .predict_with_noise(batch, Some(noise))
            .expect("predict");
        let got: Vec<f32> = match &out.values["action"] {
            Data::Float3(action) => action.clone().into_data().try_to_vec().expect("action"),
            other => panic!("unexpected action type {:?}", other.shape()),
        };
        let bytes = std::fs::read(parity.join("actions.bin")).expect("reference actions");
        let reference: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|value| f32::from_le_bytes(*value))
            .collect();
        // 官方输出补零到 64 通道，这里只比前 7 个通道。
        let mut expected = Vec::with_capacity(32 * 7);
        for step in 0..32 {
            for channel in 0..7 {
                expected.push(reference[step * 64 + channel]);
            }
        }
        assert_eq!(got.len(), expected.len());
        let max = got
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let mean = got
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / got.len() as f32;
        println!("actions: max {max:.3e} mean {mean:.3e}");
        assert!(max < 5e-3, "actions differ by {max:.3e}");
    }
}

#[cfg(test)]
mod steps {
    use super::*;
    use crate::policy::cosmos3::sampler::{Schedule, Trajectory};
    use burn::tensor::{Int, TensorData};

    fn read(path: &std::path::Path) -> Vec<f32> {
        let bytes = std::fs::read(path).expect("reference");
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|value| f32::from_le_bytes(*value))
            .collect()
    }

    /// 逐步比动作状态与速度，定位采样段差异。
    #[test]
    #[ignore = "需要转换产物与官方参照"]
    fn action_trajectory_matches_official() {
        let directory = std::path::PathBuf::from(
            std::env::var("PNEUMA_COSMOS3_NATIVE").expect("PNEUMA_COSMOS3_NATIVE"),
        );
        let parity = std::path::PathBuf::from(
            std::env::var("PNEUMA_COSMOS3_GENERATE").expect("PNEUMA_COSMOS3_GENERATE"),
        );
        let policy = Cosmos3Policy::load_model(&directory.join("model"), app::Device::Flex)
            .expect("loading native artifact")
            .with_action_dim(7);
        let device = policy.device();
        let cfg = &policy.cfg;
        let plan = SequencePlan::new(cfg);

        let ids =
            Tensor::<2, Int>::from_data(TensorData::new(vec![2i64, 5, 6, 7, 8], [1, 5]), &device);
        let encoded = policy
            .runtime
            .encode_condition(Tensor::<4>::zeros([1, 3, 256, 256], &device), cfg, &device)
            .expect("encode");
        let shape = encoded.dims();
        let latent_frames = plan.video_condition.len();
        let mut video = Tensor::<5>::zeros(
            [1, cfg.vae.z_dim, latent_frames, shape[3], shape[4]],
            &device,
        );
        video = video.slice_assign(
            [0..1, 0..cfg.vae.z_dim, 0..1, 0..shape[3], 0..shape[4]],
            encoded,
        );
        let actions = Tensor::<3>::zeros(
            [1, plan.action_condition.len(), cfg.max_action_dim],
            &device,
        );
        let clean = FlowState { video, actions };
        let noise = FlowState {
            video: Tensor::zeros(
                [1, cfg.vae.z_dim, latent_frames, shape[3], shape[4]],
                &device,
            ),
            actions: Tensor::zeros(
                [1, plan.action_condition.len(), cfg.max_action_dim],
                &device,
            ),
        };
        let start = policy.runtime.add_noise(&clean, &noise, &plan, &device);
        let tokens: Vec<i64> = ids.into_data().iter::<i64>().collect();
        let grid = positions::GenGrid {
            frames: latent_frames,
            height: shape[3],
            width: shape[4],
        };
        let prompt = policy.prompt(tokens, grid, plan.action_offset, 5);
        let rope = Cosmos3Runtime::rope(cfg);

        let schedule = Schedule::new(cfg.num_inference_steps, cfg.sampling_shift, 1000);
        let mut video_scheduler = Trajectory::<5>::new(schedule.clone());
        let mut action_scheduler = Trajectory::<3>::new(schedule);
        let mut state = start;
        let mut states = vec![state.actions.clone()];
        let mut velocities = Vec::new();
        for _ in 0..cfg.num_inference_steps {
            let timestep = video_scheduler.timestep() as f32;
            let raw = policy.runtime.velocities(
                cfg,
                &state,
                timestep,
                &prompt,
                &plan,
                &rope,
                cfg.domain_id,
                &device,
            );
            let masked = policy.runtime.mask_velocity(&raw, &plan, 7, &device);
            velocities.push(masked.actions.clone());
            let next_video = video_scheduler.step(masked.video, state.video.clone());
            let next_actions = action_scheduler.step(masked.actions, state.actions.clone());
            state = FlowState {
                video: next_video,
                actions: next_actions,
            };
            states.push(state.actions.clone());
        }

        let expected_states = read(&parity.join("steps_state.bin"));
        let expected_velocities = read(&parity.join("steps_velocity.bin"));
        let stride = 33 * 64;
        let compare = |label: &str, got: &[f32], wanted: &[f32]| {
            let max = got
                .iter()
                .zip(wanted)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            println!("{label}: max {max:.3e}");
        };
        for (index, tensor) in states.iter().enumerate() {
            let got: Vec<f32> = tensor.clone().into_data().try_to_vec().expect("state");
            compare(
                &format!("state {index}"),
                &got,
                &expected_states[index * stride..(index + 1) * stride],
            );
        }
        for (index, tensor) in velocities.iter().enumerate() {
            let got: Vec<f32> = tensor.clone().into_data().try_to_vec().expect("velocity");
            compare(
                &format!("velocity {index}"),
                &got,
                &expected_velocities[index * stride..(index + 1) * stride],
            );
        }
    }
}
