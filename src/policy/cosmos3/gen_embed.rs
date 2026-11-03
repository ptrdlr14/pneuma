//! Cosmos3 Edge 的生成流嵌入。
//!
//! 生成流按 `[视频隐变量 | 动作]` 拼接：视频隐变量做 2×2 patch 后线性投影，动作按具身域
//! 选择专属线性层，两者都加上本模态与时间步嵌入。

// 生成流在主干接线前不参与运行时构建；届时移除此豁免。
#![allow(dead_code)]

use burn::module::{Module, Param};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::silu;
use burn::tensor::{Device, Tensor, TensorData};

use super::config::Cosmos3Config;

/// 按具身域索引的权重表，对应 checkpoint 里的 `nn.Embedding`。
#[derive(Module, Debug)]
struct Table {
    weight: Param<Tensor<2>>,
}

#[derive(Module, Debug)]
struct DomainLinear {
    fc: Table,
    bias: Table,
    #[module(skip)]
    input: usize,
    #[module(skip)]
    output: usize,
}

impl DomainLinear {
    fn init(input: usize, output: usize, domains: usize, device: &Device) -> Self {
        Self {
            fc: Table {
                weight: Param::from_tensor(Tensor::zeros([domains, input * output], device)),
            },
            bias: Table {
                weight: Param::from_tensor(Tensor::zeros([domains, output], device)),
            },
            input,
            output,
        }
    }

    /// `[batch, tokens, input]` → `[batch, tokens, output]`。
    fn forward(&self, x: Tensor<3>, domain: usize) -> Tensor<3> {
        let [batch, _, _] = x.dims();
        let weight = self
            .fc
            .weight
            .val()
            .slice([domain..domain + 1, 0..self.input * self.output])
            .reshape([1, self.input, self.output])
            .expand([batch, self.input, self.output]);
        let bias = self
            .bias
            .weight
            .val()
            .slice([domain..domain + 1, 0..self.output])
            .reshape([1, 1, self.output]);
        x.matmul(weight) + bias
    }
}

#[derive(Module, Debug)]
struct TimeEmbedding {
    linear_1: Linear,
    linear_2: Linear,
}

#[derive(Module, Debug)]
pub(super) struct GenEmbeddings {
    vae2llm: Linear,
    llm2vae: Linear,
    action2llm: DomainLinear,
    llm2action: DomainLinear,
    action_modality_embed: Param<Tensor<1>>,
    time_embedder: TimeEmbedding,
    #[module(skip)]
    latent_patch: usize,
    #[module(skip)]
    latent_channels: usize,
    #[module(skip)]
    timestep_dim: usize,
    #[module(skip)]
    timestep_scale: f64,
    #[module(skip)]
    hidden: usize,
}

impl GenEmbeddings {
    pub(super) fn init(cfg: &Cosmos3Config, device: &Device) -> Self {
        let hidden = cfg.mot_backbone.text_config.hidden_size;
        let patches = cfg.latent_patch_size * cfg.latent_patch_size * cfg.vae.z_dim;
        Self {
            vae2llm: LinearConfig::new(patches, hidden).init(device),
            llm2vae: LinearConfig::new(hidden, patches).init(device),
            action2llm: DomainLinear::init(
                cfg.max_action_dim,
                hidden,
                cfg.num_embodiment_domains,
                device,
            ),
            llm2action: DomainLinear::init(
                hidden,
                cfg.max_action_dim,
                cfg.num_embodiment_domains,
                device,
            ),
            action_modality_embed: Param::from_tensor(Tensor::zeros([hidden], device)),
            time_embedder: TimeEmbedding {
                linear_1: LinearConfig::new(cfg.timestep_embedding_dim, hidden).init(device),
                linear_2: LinearConfig::new(hidden, hidden).init(device),
            },
            latent_patch: cfg.latent_patch_size,
            latent_channels: cfg.vae.z_dim,
            timestep_dim: cfg.timestep_embedding_dim,
            timestep_scale: cfg.timestep_scale,
            hidden,
        }
    }

    /// 视频隐变量 `[B,C,T,H,W]` → `[B,T*h*w,D]`。
    ///
    /// `condition` 为真的帧是条件帧，不加时间步嵌入。
    pub(super) fn embed_video(
        &self,
        latents: Tensor<5>,
        timesteps: &[f32],
        condition: &[bool],
    ) -> Tensor<3> {
        let [batch, _, frames, _, _] = latents.dims();
        let latents_device = latents.device();
        let patches = patchify(latents, self.latent_patch);
        let [_, _, tokens, width] = patches.dims();
        let embeds = self
            .vae2llm
            .forward(patches.reshape([batch * frames * tokens, width]))
            .reshape([batch, frames, tokens, self.hidden]);
        let time = self
            .embed_timesteps(timesteps, &latents_device)
            .masked(condition)
            .reshape([batch, frames, 1, self.hidden]);
        (embeds + time).reshape([batch, frames * tokens, self.hidden])
    }

    /// 动作 `[B,T,A]` → `[B,T,D]`，加模态嵌入与时间步嵌入。
    pub(super) fn embed_actions(
        &self,
        actions: Tensor<3>,
        timesteps: &[f32],
        condition: &[bool],
        domain: usize,
    ) -> Tensor<3> {
        let [batch, tokens, _] = actions.dims();
        let actions_device = actions.device();
        let embeds = self.action2llm.forward(actions, domain)
            + self
                .action_modality_embed
                .val()
                .reshape([1, 1, self.hidden]);
        let time = self
            .embed_timesteps(timesteps, &actions_device)
            .masked(condition)
            .reshape([batch, tokens, self.hidden]);
        embeds + time
    }

    /// 生成流末端 `[B,G,D]` 的动作部分 → 速度 `[B,T,A]`。
    pub(super) fn actions_from_hidden(&self, hidden: Tensor<3>, domain: usize) -> Tensor<3> {
        self.llm2action.forward(hidden, domain)
    }

    /// 生成流末端 `[B,T*h*w,D]` 的视频部分 → 速度 `[B,C,T,H,W]`。
    pub(super) fn video_from_hidden(
        &self,
        hidden: Tensor<3>,
        frames: usize,
        height: usize,
        width: usize,
    ) -> Tensor<5> {
        let [batch, _, _] = hidden.dims();
        let patches = self.llm2vae.forward(hidden);
        unpatchify(
            patches,
            batch,
            frames,
            height,
            width,
            self.latent_patch,
            self.latent_channels,
        )
    }

    /// 视频 token 投影权重，供导入测试抽样检查。
    #[cfg(test)]
    pub(super) fn vae2llm_weight(&self) -> Tensor<2> {
        self.vae2llm.weight.val()
    }

    fn embed_timesteps(&self, timesteps: &[f32], device: &Device) -> Tensor<2> {
        let features = sinusoidal(timesteps, self.timestep_dim, self.timestep_scale, device);
        let hidden = self.time_embedder.linear_1.forward(features);
        self.time_embedder.linear_2.forward(silu(hidden))
    }
}

trait Masked {
    /// 把 `condition` 为真的行清零。
    fn masked(self, condition: &[bool]) -> Self;
}

impl Masked for Tensor<2> {
    fn masked(self, condition: &[bool]) -> Tensor<2> {
        let rows = condition.len();
        let keep: Vec<f32> = condition
            .iter()
            .map(|flag| if *flag { 0.0 } else { 1.0 })
            .collect();
        let device = self.device();
        self * Tensor::from_data(TensorData::new(keep, [rows, 1]), &device)
    }
}

fn sinusoidal(timesteps: &[f32], dim: usize, scale: f64, device: &Device) -> Tensor<2> {
    let half = dim / 2;
    let frequency: Vec<f32> = (0..half)
        .map(|index| (-(10000f64).ln() * index as f64 / half as f64).exp() as f32)
        .collect();
    let mut values = Vec::with_capacity(timesteps.len() * dim);
    for timestep in timesteps {
        let scaled = *timestep * scale as f32;
        for frequency in &frequency {
            values.push((scaled * frequency).cos());
        }
        for frequency in &frequency {
            values.push((scaled * frequency).sin());
        }
    }
    Tensor::from_data(TensorData::new(values, [timesteps.len(), dim]), device)
}

/// `[B,C,T,H,W]` → `[B,T,h*w,p*p*C]`；高宽不足整 patch 时在右下补零。
fn patchify(x: Tensor<5>, patch: usize) -> Tensor<4> {
    let [batch, channels, frames, height, width] = x.dims();
    let pad_h = (patch - height % patch) % patch;
    let pad_w = (patch - width % patch) % patch;
    let x = pad5(x, 3, 0, pad_h);
    let x = pad5(x, 4, 0, pad_w);
    let (height, width) = (height + pad_h, width + pad_w);
    let (grid_h, grid_w) = (height / patch, width / patch);
    x.reshape([batch, channels, frames, grid_h, patch, grid_w, patch])
        .permute([0, 2, 3, 5, 4, 6, 1])
        .reshape([batch, frames, grid_h * grid_w, patch * patch * channels])
}

#[allow(clippy::too_many_arguments)]
fn unpatchify(
    patches: Tensor<3>,
    batch: usize,
    frames: usize,
    height: usize,
    width: usize,
    patch: usize,
    channels: usize,
) -> Tensor<5> {
    let grid_h = height.div_ceil(patch);
    let grid_w = width.div_ceil(patch);
    let video = patches
        .reshape([batch, frames, grid_h, grid_w, patch, patch, channels])
        .permute([0, 6, 1, 2, 4, 3, 5])
        .reshape([batch, channels, frames, grid_h * patch, grid_w * patch]);
    video.slice([0..batch, 0..channels, 0..frames, 0..height, 0..width])
}

fn pad5(x: Tensor<5>, dim: usize, before: usize, after: usize) -> Tensor<5> {
    if before == 0 && after == 0 {
        return x;
    }
    let dims = x.dims();
    let device = x.device();
    let zeros = |amount: usize| -> Tensor<5> {
        let mut shape = dims;
        shape[dim] = amount;
        Tensor::zeros(shape, &device)
    };
    let mut parts = Vec::with_capacity(3);
    if before > 0 {
        parts.push(zeros(before));
    }
    parts.push(x);
    if after > 0 {
        parts.push(zeros(after));
    }
    Tensor::cat(parts, dim)
}

#[cfg(test)]
mod reference {
    use super::*;
    use burn::tensor::DType;
    use burn_store::{
        FloatCastAdapter, ModuleAdapter, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore,
    };
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
        let max = got
            .iter()
            .zip(want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let mean = got
            .iter()
            .zip(want)
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / got.len() as f32;
        println!(
            "{name}: max {max:.3e} mean {mean:.3e} over {} values",
            got.len()
        );
        assert!(
            max <= tolerance,
            "{name} differs by {max:.3e} > {tolerance:.1e}"
        );
    }

    #[test]
    #[ignore = "需要 cosmos3 权重与参照张量"]
    fn gen_embeddings_match_reference() {
        let checkpoint =
            PathBuf::from(std::env::var("PNEUMA_COSMOS3_CKPT").expect("PNEUMA_COSMOS3_CKPT"));
        let parity =
            PathBuf::from(std::env::var("PNEUMA_COSMOS3_GEN").expect("PNEUMA_COSMOS3_GEN"));
        let device = Device::flex();
        let cfg = super::super::config::Cosmos3Config::default();
        let mut embeddings = GenEmbeddings::init(&cfg, &device);

        let mut store = SafetensorsStore::from_file(checkpoint.join("model.safetensors"))
            .with_from_adapter(PyTorchToBurnAdapter.chain(FloatCastAdapter::to(DType::F32)))
            .with_key_remapping(r"^model\.network\.", "")
            .with_key_remapping(r"^action_modality_embed\.weight$", "action_modality_embed")
            .allow_partial(true);
        let applied = embeddings
            .load_from(&mut store)
            .expect("loading gen embeddings");
        assert!(
            applied.missing.is_empty(),
            "missing gen weights: {:?}",
            applied.missing
        );

        let latents = Tensor::<5>::from_data(
            TensorData::new(read_f32(&parity.join("latents.bin")), [1, 48, 1, 16, 16]),
            &device,
        );
        let video = embeddings.embed_video(latents, &[0.7], &[false]);
        compare(
            "video_embeds",
            &video
                .clone()
                .into_data()
                .try_to_vec::<f32>()
                .expect("video"),
            &read_f32(&parity.join("video_embeds.bin")),
            2e-3,
        );

        let actions = Tensor::<3>::from_data(
            TensorData::new(read_f32(&parity.join("actions.bin")), [1, 4, 64]),
            &device,
        );
        let action = embeddings.embed_actions(actions, &[0.7; 4], &[false; 4], 0);
        compare(
            "action_embeds",
            &action
                .clone()
                .into_data()
                .try_to_vec::<f32>()
                .expect("action"),
            &read_f32(&parity.join("action_embeds.bin")),
            2e-3,
        );

        let merged = Tensor::cat(vec![video, action], 1);
        compare(
            "merged",
            &merged.into_data().try_to_vec::<f32>().expect("merged"),
            &read_f32(&parity.join("merged.bin")),
            2e-3,
        );

        let video_hidden: Vec<f32> = (0..64 * 2048)
            .map(|index| (index as f32 * 0.001).sin())
            .collect();
        let video_hidden =
            Tensor::<3>::from_data(TensorData::new(video_hidden, [1, 64, 2048]), &device);
        let velocity = embeddings.video_from_hidden(video_hidden, 1, 16, 16);
        compare(
            "video_velocity",
            &velocity.into_data().try_to_vec::<f32>().expect("velocity"),
            &read_f32(&parity.join("video_velocity.bin")),
            2e-3,
        );

        let action_hidden: Vec<f32> = (0..4 * 2048)
            .map(|index| (index as f32 * 0.002).cos())
            .collect();
        let action_hidden =
            Tensor::<3>::from_data(TensorData::new(action_hidden, [1, 4, 2048]), &device);
        let velocity = embeddings.actions_from_hidden(action_hidden, 0);
        compare(
            "action_velocity",
            &velocity
                .into_data()
                .try_to_vec::<f32>()
                .expect("action velocity"),
            &read_f32(&parity.join("action_velocity.bin")),
            2e-3,
        );
    }
}
