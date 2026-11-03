//! Cosmos3 Edge 的运行时：条件编码、生成流布局、CFG 与采样循环。
//!
//! 一次推理的流程：编码条件帧 → 构造噪声 → 4 步 uniPC，每步跑一次主干并做无分类器引导 →
//! 解包动作块。

// 运行时在策略接线前不参与运行时构建；届时移除此豁免。
#![allow(dead_code)]

use burn::module::Module;
use burn::tensor::{DType, Device, Tensor};

use super::config::Cosmos3Config;
use super::gen_embed::GenEmbeddings;
use super::net::Cosmos3Net;
use super::sampler::{Schedule, Trajectory};
use super::text::MRope;
use super::vae::WanVae;

/// 视频隐变量与动作的状态。
#[derive(Clone, Debug)]
pub(crate) struct FlowState {
    pub(crate) video: Tensor<5>,
    pub(crate) actions: Tensor<3>,
}

/// 文本提示：token、位置与可选的图像。
pub(super) struct Prompt {
    pub(super) tokens: Vec<i64>,
    pub(super) positions: [Vec<f32>; 3],
    pub(super) image: Option<Tensor<4>>,
    pub(super) image_span: std::ops::Range<usize>,
}

/// 序列布局：哪些隐变量帧与动作是条件。
#[derive(Clone, Debug)]
pub(super) struct SequencePlan {
    pub(super) video_condition: Vec<bool>,
    pub(super) action_condition: Vec<bool>,
    pub(super) action_offset: i64,
}

impl SequencePlan {
    /// `mode = "wam"`、`use_state = true` 的布局：条件帧 1 帧，条件动作 1 个（状态）。
    pub(super) fn new(cfg: &Cosmos3Config) -> Self {
        let latent_frames = 1 + cfg.chunk_size / cfg.vae.scale_factor_temporal;
        let actions = cfg.chunk_size + usize::from(cfg.use_state);
        let mut video_condition = vec![false; latent_frames];
        video_condition[0] = true;
        let mut action_condition = vec![false; actions];
        if cfg.use_state {
            action_condition[0] = true;
        }
        Self {
            video_condition,
            action_condition,
            action_offset: if cfg.use_state { 0 } else { 1 },
        }
    }
}

#[derive(Module, Debug)]
pub(super) struct Cosmos3Runtime {
    net: Cosmos3Net,
    embeddings: GenEmbeddings,
    vae: WanVae,
}

impl Cosmos3Runtime {
    pub(super) fn init(cfg: &Cosmos3Config, device: &Device) -> Self {
        Self {
            net: Cosmos3Net::init(cfg, device),
            embeddings: GenEmbeddings::init(cfg, device),
            vae: WanVae::init(&cfg.vae, device),
        }
    }

    /// `x_t = sigma * noise + (1 - sigma) * clean`，条件位置保持干净值。
    pub(super) fn add_noise(
        &self,
        clean: &FlowState,
        noise: &FlowState,
        plan: &SequencePlan,
        device: &Device,
    ) -> FlowState {
        let video = interpolate(&noise.video, &clean.video, &plan.video_condition, device);
        let actions = interpolate_tokens(
            &noise.actions,
            &clean.actions,
            &plan.action_condition,
            device,
        );
        FlowState { video, actions }
    }

    /// 生成流的最终隐藏状态 `[1, 视频+动作, D]`。
    #[allow(clippy::too_many_arguments)]
    pub(super) fn gen_hidden(
        &self,
        state: &FlowState,
        timestep: f32,
        prompt: &Prompt,
        plan: &SequencePlan,
        rope: &MRope,
        domain: usize,
        device: &Device,
    ) -> (Tensor<3>, usize) {
        let latent_frames = state.video.dims()[2];
        let action_count = state.actions.dims()[1];
        let video_timesteps = vec![timestep; latent_frames];
        let action_timesteps = vec![timestep; action_count];
        let video = self.embeddings.embed_video(
            state.video.clone(),
            &video_timesteps,
            &plan.video_condition,
        );
        let actions = self.embeddings.embed_actions(
            state.actions.clone(),
            &action_timesteps,
            &plan.action_condition,
            domain,
        );
        let tokens = video.dims()[1];
        let generation = Tensor::cat(vec![video, actions], 1);
        let (_, hidden) = self.net.forward(
            &prompt.tokens,
            &vec![true; prompt.tokens.len()],
            prompt.image.clone(),
            prompt.image_span.clone(),
            generation,
            [
                &prompt.positions[0],
                &prompt.positions[1],
                &prompt.positions[2],
            ],
            rope,
            device,
        );
        (hidden, tokens)
    }

    /// 一步速度：跑一次主干，返回 `[视频速度, 动作速度]`。
    #[allow(clippy::too_many_arguments)]
    pub(super) fn velocities(
        &self,
        _cfg: &Cosmos3Config,
        x_t: &FlowState,
        timestep: f32,
        prompt: &Prompt,
        plan: &SequencePlan,
        rope: &MRope,
        domain: usize,
        device: &Device,
    ) -> FlowState {
        let (hidden, tokens) = self.gen_hidden(x_t, timestep, prompt, plan, rope, domain, device);
        let [_, total, width] = hidden.dims();
        let video_hidden = hidden.clone().slice([0..1, 0..tokens, 0..width]);
        let action_hidden = hidden.slice([0..1, tokens..total, 0..width]);
        let frame_shape = x_t.video.dims();
        let video_velocity = self.embeddings.video_from_hidden(
            video_hidden,
            frame_shape[2],
            frame_shape[3],
            frame_shape[4],
        );
        let action_velocity = self.embeddings.actions_from_hidden(action_hidden, domain);
        FlowState {
            video: video_velocity,
            actions: action_velocity,
        }
    }

    /// 条件帧、条件动作与填充通道的速度清零。
    pub(super) fn mask_velocity(
        &self,
        velocity: &FlowState,
        plan: &SequencePlan,
        action_dim: usize,
        device: &Device,
    ) -> FlowState {
        let video = zero_frames(&velocity.video, &plan.video_condition, device);
        let mut actions = zero_tokens(&velocity.actions, &plan.action_condition, device);
        let [_, tokens, channels] = actions.dims();
        if action_dim < channels {
            let zeros = Tensor::<3>::zeros([1, tokens, channels - action_dim], device);
            actions = actions.slice_assign([0..1, 0..tokens, action_dim..channels], zeros);
        }
        FlowState { video, actions }
    }

    /// 无分类器引导：`u + scale * (c - u)`。
    pub(super) fn guide(&self, cond: &FlowState, uncond: &FlowState, scale: f64) -> FlowState {
        FlowState {
            video: uncond.video.clone() + (cond.video.clone() - uncond.video.clone()) * scale,
            actions: uncond.actions.clone()
                + (cond.actions.clone() - uncond.actions.clone()) * scale,
        }
    }

    /// 整条采样：4 步 uniPC，每步一次条件、一次无条件（`guidance_scale != 1`）。
    #[allow(clippy::too_many_arguments)]
    pub(super) fn sample<F>(
        &self,
        cfg: &Cosmos3Config,
        mut velocity: F,
        start: FlowState,
        steps: usize,
        device: &Device,
    ) -> FlowState
    where
        F: FnMut(&FlowState, f32) -> FlowState,
    {
        let schedule = Schedule::new(steps, cfg.sampling_shift, 1000);
        let mut video = Trajectory::<5>::new(schedule.clone());
        let mut actions = Trajectory::<3>::new(schedule);
        let mut state = start;
        for _ in 0..steps {
            let timestep = video.timestep() as f32;
            let delta = velocity(&state, timestep);
            let next_video = video.step(delta.video, state.video.clone());
            let next_actions = actions.step(delta.actions, state.actions.clone());
            state = FlowState {
                video: next_video,
                actions: next_actions,
            };
        }
        let _ = device;
        state
    }

    /// 把一步速度包成采样器需要的闭包。
    #[allow(clippy::too_many_arguments)]
    pub(super) fn guided_velocity<'a>(
        &'a self,
        cfg: &'a Cosmos3Config,
        cond: &'a Prompt,
        uncond: Option<&'a Prompt>,
        plan: &'a SequencePlan,
        rope: &'a MRope,
        domain: usize,
        action_dim: usize,
        device: &'a Device,
    ) -> impl FnMut(&FlowState, f32) -> FlowState + 'a {
        move |state, timestep| {
            let conditional =
                self.velocities(cfg, state, timestep, cond, plan, rope, domain, device);
            let velocity = match uncond {
                Some(negative) => {
                    let unconditional =
                        self.velocities(cfg, state, timestep, negative, plan, rope, domain, device);
                    self.guide(&conditional, &unconditional, cfg.guidance_scale)
                }
                None => conditional,
            };
            self.mask_velocity(&velocity, plan, action_dim, device)
        }
    }
}

/// `[1,1,T,1,1]` 的掩码：条件帧为 0，其余为 1。
fn frame_mask(condition: &[bool], device: &Device) -> Tensor<5> {
    let values: Vec<f32> = condition
        .iter()
        .map(|flag| if *flag { 0.0 } else { 1.0 })
        .collect();
    Tensor::from_data(
        burn::tensor::TensorData::new(values, [1, 1, condition.len(), 1, 1]),
        device,
    )
}

/// `[1,T,1]` 的掩码：条件动作为 0，其余为 1。
fn token_mask(condition: &[bool], device: &Device) -> Tensor<3> {
    let values: Vec<f32> = condition
        .iter()
        .map(|flag| if *flag { 0.0 } else { 1.0 })
        .collect();
    Tensor::from_data(
        burn::tensor::TensorData::new(values, [1, condition.len(), 1]),
        device,
    )
}

fn interpolate(
    noise: &Tensor<5>,
    clean: &Tensor<5>,
    condition: &[bool],
    device: &Device,
) -> Tensor<5> {
    let mask = frame_mask(condition, device);
    noise.clone() * mask.clone() + clean.clone() * (mask * -1.0 + 1.0)
}

fn interpolate_tokens(
    noise: &Tensor<3>,
    clean: &Tensor<3>,
    condition: &[bool],
    device: &Device,
) -> Tensor<3> {
    let mask = token_mask(condition, device);
    noise.clone() * mask.clone() + clean.clone() * (mask * -1.0 + 1.0)
}

fn zero_frames(video: &Tensor<5>, condition: &[bool], device: &Device) -> Tensor<5> {
    video.clone() * frame_mask(condition, device)
}

fn zero_tokens(actions: &Tensor<3>, condition: &[bool], device: &Device) -> Tensor<3> {
    actions.clone() * token_mask(condition, device)
}

#[cfg(test)]
mod reference {
    use super::*;
    use std::path::Path;

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

    #[test]
    #[ignore = "需要 cosmos3 布局参照"]
    fn sequence_plan_matches_reference() {
        let parity = std::path::PathBuf::from(
            std::env::var("PNEUMA_COSMOS3_PLAN").expect("PNEUMA_COSMOS3_PLAN"),
        );
        let cfg = super::super::config::Cosmos3Config::default();
        let plan = SequencePlan::new(&cfg);
        let video: Vec<bool> = read_f32(&parity.join("video.bin"))
            .into_iter()
            .map(|value| value > 0.5)
            .collect();
        let actions: Vec<bool> = read_f32(&parity.join("actions.bin"))
            .into_iter()
            .map(|value| value > 0.5)
            .collect();
        let offset = read_f32(&parity.join("offset.bin"))[0] as i64;
        assert_eq!(plan.video_condition, video);
        assert_eq!(plan.action_condition, actions);
        assert_eq!(plan.action_offset, offset);
    }
}

impl Cosmos3Runtime {
    /// 从 LeRobot checkpoint 导入权重，返回未被使用的张量名。
    ///
    /// 前缀映射：主干在 `model.network.mot_backbone.model.` 下，生成流投影在 `model.network.` 下，
    /// VAE 在 `model.network.vae.` 下。理解流主干字段名 `language_model` 对应本端口的 `text`。
    pub(super) fn import_lerobot(
        source: &std::path::Path,
        cfg: &Cosmos3Config,
        device: &Device,
    ) -> anyhow::Result<(Self, Vec<String>)> {
        use anyhow::{Context, ensure};
        use burn_store::{
            FloatCastAdapter, ModuleAdapter, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore,
        };

        let file = source.join("model.safetensors");
        ensure!(file.is_file(), "missing {}", file.display());
        let mut runtime = Self::init(cfg, device);
        let mut store = SafetensorsStore::from_file(&file)
            .with_from_adapter(PyTorchToBurnAdapter.chain(FloatCastAdapter::to(DType::F32)))
            .with_key_remapping(
                r"^model\.network\.mot_backbone\.model\.language_model\.",
                "net.text.",
            )
            .with_key_remapping(r"^model\.network\.mot_backbone\.model\.", "net.")
            .with_key_remapping(r"^model\.network\.vae\.", "vae.")
            .with_key_remapping(
                r"^vae\.(encoder\.down_blocks\.\d+\.downsampler)\.resample\.1\.",
                "vae.$1.resample.",
            )
            .with_key_remapping(r"^model\.network\.", "embeddings.")
            .with_key_remapping(
                r"^embeddings\.action_modality_embed\.weight$",
                "embeddings.action_modality_embed",
            )
            .allow_partial(true);
        let applied = runtime
            .load_from(&mut store)
            .context("loading cosmos3 weights")?;
        ensure!(
            applied.missing.is_empty(),
            "missing cosmos3 weights: {:?}",
            &applied.missing[..applied.missing.len().min(5)]
        );
        Ok((runtime, applied.unused))
    }

    /// 视频 token 投影的权重，导入测试用它确认权重确实读入。
    #[cfg(test)]
    pub(super) fn vae2llm_weight(&self) -> Tensor<2> {
        self.embeddings.vae2llm_weight()
    }

    /// 保存为原生 bpk 文件，供 `serve` 直接加载。
    pub(super) fn save(&self, path: &std::path::Path) -> anyhow::Result<()> {
        self.clone().into_record().save(path)?;
        Ok(())
    }
}

#[cfg(test)]
mod import {
    use super::*;
    use std::path::Path;

    /// 真实 checkpoint 应无缺失张量，且未使用的只剩词表头与 VAE 解码器。
    #[test]
    #[ignore = "需要 cosmos3 权重"]
    fn imports_the_real_checkpoint() {
        let checkpoint = std::path::PathBuf::from(
            std::env::var("PNEUMA_COSMOS3_CKPT").expect("PNEUMA_COSMOS3_CKPT"),
        );
        let cfg = super::super::config::Cosmos3Config::default();
        let (runtime, unused) =
            Cosmos3Runtime::import_lerobot(Path::new(&checkpoint), &cfg, &Device::flex())
                .expect("import");
        println!("unused {} tensors", unused.len());
        assert!(
            unused.iter().any(|name| name.contains("lm_head")),
            "lm_head should be unused"
        );
        assert!(
            unused.iter().any(|name| name.contains("decoder")),
            "VAE decoder should be unused"
        );
        assert!(
            unused.iter().all(|name| name.contains("lm_head")
                || name.contains("decoder")
                || name.contains("post_quant")),
            "unexpected leftovers: {:?}",
            &unused[..unused.len().min(5)]
        );
        let weight = runtime
            .embeddings
            .vae2llm_weight()
            .into_data()
            .try_to_vec::<f32>()
            .expect("vae2llm");
        assert!(weight.iter().any(|value| *value != 0.0));
    }
}

impl Cosmos3Runtime {
    /// 把动作隐藏态投影成速度，供域编号探测使用。
    pub(super) fn actions_from_hidden(&self, hidden: Tensor<3>, domain: usize) -> Tensor<3> {
        self.embeddings.actions_from_hidden(hidden, domain)
    }

    /// 主干的 mRoPE。
    pub(super) fn rope(cfg: &Cosmos3Config) -> MRope {
        Cosmos3Net::rope(cfg)
    }

    /// 把一帧画布编码为归一化隐变量 `[1,z_dim,1,h,w]`。
    pub(super) fn encode_condition(
        &self,
        pixels: Tensor<4>,
        cfg: &Cosmos3Config,
        device: &Device,
    ) -> anyhow::Result<Tensor<5>> {
        let [batch, channels, height, width] = pixels.dims();
        anyhow::ensure!(
            batch == 1 && channels == 3,
            "条件帧必须是单样本三通道图像，收到 {batch}x{channels}"
        );
        anyhow::ensure!(
            height % 16 == 0 && width % 16 == 0,
            "条件帧的宽高必须是 16 的倍数，收到 {height}x{width}"
        );
        let pixels = pixels.reshape([1, 3, 1, height, width]);
        let latents = self.vae.encode(pixels);
        let [_, z_dim, _, latent_h, latent_w] = latents.dims();
        anyhow::ensure!(z_dim == cfg.vae.z_dim, "VAE 输出通道数与配置不一致");
        let mean = Tensor::<5>::from_data(
            burn::tensor::TensorData::new(cfg.vae.latents_mean.clone(), [1, z_dim, 1, 1, 1]),
            device,
        );
        let std = Tensor::<5>::from_data(
            burn::tensor::TensorData::new(cfg.vae.latents_std.clone(), [1, z_dim, 1, 1, 1]),
            device,
        );
        let normalized = (latents - mean) / std;
        debug_assert_eq!(normalized.dims(), [1, z_dim, 1, latent_h, latent_w]);
        Ok(normalized)
    }
}
