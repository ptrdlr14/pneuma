//! `cosmos3_edge` checkpoint 配置与本端口的运行时约束。
//!
//! 官方配置是嵌套 dataclass 树；这里只解释推理需要的字段，未知字段忽略。

use std::path::Path;

use crate::policy::PolicyKind;
use anyhow::{Context, Result, ensure};
use serde::Deserialize;

/// 策略级配置。
#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct Cosmos3Config {
    #[serde(rename = "type")]
    pub policy_type: String,
    pub chunk_size: usize,
    pub n_action_steps: usize,
    pub max_action_dim: usize,
    pub num_inference_steps: usize,
    pub guidance_scale: f64,
    pub sampling_shift: f64,
    pub mode: String,
    pub use_state: bool,
    pub latent_patch_size: usize,
    pub num_embodiment_domains: usize,
    pub domain_id: usize,
    pub timestep_embedding_dim: usize,
    pub timestep_scale: f64,
    pub num_train_timesteps: usize,
    pub temporal_modality_margin: usize,
    pub base_fps: f64,
    pub fps: Option<f64>,
    pub use_peft: bool,
    pub mot_backbone: BackboneConfig,
    pub vae: VaeConfig,
}

impl Default for Cosmos3Config {
    fn default() -> Self {
        Self {
            policy_type: PolicyKind::Cosmos3Edge.as_str().into(),
            chunk_size: 32,
            n_action_steps: 32,
            max_action_dim: 64,
            num_inference_steps: 4,
            guidance_scale: 3.0,
            sampling_shift: 5.0,
            mode: "wam".into(),
            use_state: true,
            latent_patch_size: 2,
            num_embodiment_domains: 32,
            domain_id: 8,
            timestep_embedding_dim: 256,
            timestep_scale: 0.001,
            num_train_timesteps: 1000,
            temporal_modality_margin: 15000,
            base_fps: 24.0,
            fps: None,
            use_peft: false,
            mot_backbone: BackboneConfig::default(),
            vae: VaeConfig::default(),
        }
    }
}

/// MoT 主干：Nemotron-3-Dense-VL 文本塔、SigLIP2 视觉塔与 patch merger。
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct BackboneConfig {
    pub text_config: TextConfig,
    pub vision_config: VisionConfig,
    pub projector_config: ProjectorConfig,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct TextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub hidden_act: String,
    pub rms_norm_eps: f64,
    pub attention_bias: bool,
    pub mlp_bias: bool,
    pub joint_attn_implementation: String,
    pub rope_parameters: Option<RopeConfig>,
}

impl Default for TextConfig {
    fn default() -> Self {
        Self {
            vocab_size: 131_072,
            hidden_size: 2048,
            intermediate_size: 9216,
            num_hidden_layers: 28,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            head_dim: 128,
            hidden_act: "relu2".into(),
            rms_norm_eps: 1e-5,
            attention_bias: false,
            mlp_bias: false,
            joint_attn_implementation: "two_way".into(),
            rope_parameters: Some(RopeConfig::default()),
        }
    }
}

/// 多模态 RoPE：时间、高、宽三个轴各自占一段频率。
#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct RopeConfig {
    pub rope_type: String,
    pub rope_theta: f64,
    pub mrope_section: [usize; 3],
    pub mrope_interleaved: bool,
}

impl Default for RopeConfig {
    fn default() -> Self {
        Self {
            rope_type: "default".into(),
            rope_theta: 100_000_000.0,
            mrope_section: [24, 20, 20],
            mrope_interleaved: true,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct VisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_channels: usize,
    pub patch_size: usize,
    pub layer_norm_eps: f64,
    pub hidden_act: String,
    pub num_patches: usize,
    pub spatial_merge_size: usize,
}

impl Default for VisionConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1152,
            intermediate_size: 4304,
            num_hidden_layers: 27,
            num_attention_heads: 16,
            num_channels: 3,
            patch_size: 16,
            layer_norm_eps: 1e-6,
            hidden_act: "gelu_pytorch_tanh".into(),
            num_patches: 256,
            spatial_merge_size: 2,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct ProjectorConfig {
    pub input_hidden_size: usize,
    pub merger_intermediate_size: usize,
    pub out_hidden_size: usize,
    pub spatial_merge_size: usize,
    pub use_postshuffle_norm: bool,
}

impl Default for ProjectorConfig {
    fn default() -> Self {
        Self {
            input_hidden_size: 1152,
            merger_intermediate_size: 11_520,
            out_hidden_size: 2048,
            spatial_merge_size: 2,
            use_postshuffle_norm: false,
        }
    }
}

/// Wan2.2 VAE 的编码器与归一化统计。
#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct VaeConfig {
    pub base_dim: usize,
    pub decoder_base_dim: usize,
    pub z_dim: usize,
    pub dim_mult: Vec<usize>,
    pub num_res_blocks: usize,
    pub attn_scales: Vec<f64>,
    pub temperal_downsample: Vec<bool>,
    pub latents_mean: Vec<f32>,
    pub latents_std: Vec<f32>,
    pub is_residual: bool,
    pub in_channels: usize,
    pub out_channels: usize,
    pub patch_size: usize,
    pub scale_factor_temporal: usize,
    pub scale_factor_spatial: usize,
}

impl Default for VaeConfig {
    fn default() -> Self {
        Self {
            base_dim: 160,
            decoder_base_dim: 256,
            z_dim: 48,
            dim_mult: vec![1, 2, 4, 4],
            num_res_blocks: 2,
            attn_scales: vec![],
            temperal_downsample: vec![false, true, true],
            latents_mean: vec![0.0; 48],
            latents_std: vec![1.0; 48],
            is_residual: true,
            in_channels: 12,
            out_channels: 12,
            patch_size: 2,
            scale_factor_temporal: 4,
            scale_factor_spatial: 16,
        }
    }
}

impl Cosmos3Config {
    /// 推理需要的尺寸与开关；未知取值在构造模块前拒绝。
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.policy_type == PolicyKind::Cosmos3Edge.as_str(),
            "config type must be cosmos3_edge"
        );
        ensure!(self.mode == "wam", "only mode=wam is supported");
        ensure!(!self.use_peft, "PEFT checkpoints are not supported");
        ensure!(
            self.use_state,
            "cosmos3_edge inference requires use_state=true"
        );
        ensure!(
            self.chunk_size > 0 && self.n_action_steps <= self.chunk_size,
            "invalid chunk size"
        );
        ensure!(self.max_action_dim > 0, "max_action_dim must be positive");
        ensure!(
            self.num_inference_steps > 0 && self.num_inference_steps <= 32,
            "num_inference_steps must be in 1..=32"
        );
        ensure!(
            self.guidance_scale.is_finite() && self.guidance_scale >= 0.0,
            "guidance_scale must be finite and non-negative"
        );
        ensure!(
            self.sampling_shift.is_finite() && self.sampling_shift > 0.0,
            "sampling_shift must be positive"
        );
        ensure!(
            self.latent_patch_size > 0,
            "latent_patch_size must be positive"
        );
        ensure!(
            self.num_embodiment_domains > 0,
            "num_embodiment_domains must be positive"
        );
        ensure!(
            self.timestep_embedding_dim > 0 && self.timestep_embedding_dim.is_multiple_of(2),
            "timestep_embedding_dim must be positive and even"
        );

        let text = &self.mot_backbone.text_config;
        let vision = &self.mot_backbone.vision_config;
        let projector = &self.mot_backbone.projector_config;
        ensure!(
            text.hidden_size == 2048
                && text.num_hidden_layers == 28
                && text.num_attention_heads == 16
                && text.num_key_value_heads == 8
                && text.head_dim == 128
                && text.vocab_size == 131_072,
            "only the Nemotron-3-Dense-VL text tower is supported"
        );
        ensure!(
            text.hidden_act == "relu2",
            "text tower requires relu2 activation"
        );
        ensure!(
            text.joint_attn_implementation == "two_way",
            "only joint_attn_implementation=two_way is supported"
        );
        let rope = text
            .rope_parameters
            .as_ref()
            .context("text rope_parameters required")?;
        ensure!(
            rope.rope_type == "default",
            "only default rope is supported"
        );
        ensure!(
            rope.mrope_section.iter().sum::<usize>() == text.head_dim / 2,
            "mrope_section must cover half of head_dim"
        );
        ensure!(
            vision.hidden_size == 1152
                && vision.num_hidden_layers == 27
                && vision.num_attention_heads == 16
                && vision.patch_size == 16
                && vision.spatial_merge_size == 2,
            "only the SigLIP2 vision tower is supported"
        );
        ensure!(
            projector.input_hidden_size == vision.hidden_size
                && projector.out_hidden_size == text.hidden_size
                && projector.spatial_merge_size == vision.spatial_merge_size
                && projector.merger_intermediate_size > 0,
            "projector geometry disagrees with the vision tower"
        );

        ensure!(
            self.vae.z_dim > 0 && self.vae.patch_size == 2 && self.vae.in_channels == 12,
            "unsupported Wan2.2 VAE geometry"
        );
        ensure!(
            self.vae.latents_mean.len() == self.vae.z_dim
                && self.vae.latents_std.len() == self.vae.z_dim,
            "VAE statistics must cover z_dim channels"
        );
        ensure!(
            self.vae.dim_mult.len() == 4 && self.vae.temperal_downsample.len() == 3,
            "unsupported Wan2.2 VAE depth"
        );
        Ok(())
    }

    /// 图像 token 数：每 `spatial_merge_size²` 个 patch 合并成一个 token。
    pub fn vision_tokens(&self) -> usize {
        let vision = &self.mot_backbone.vision_config;
        let grid = (vision.num_patches as f64).sqrt() as usize;
        (grid / vision.spatial_merge_size).pow(2)
    }

    /// VAE 隐变量单帧的 patch 维度：`latent_patch_size² × z_dim`。
    pub fn latent_patch_dim(&self) -> usize {
        self.latent_patch_size * self.latent_patch_size * self.vae.z_dim
    }
}

pub fn parse(path: &Path) -> Result<Cosmos3Config> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let config: Cosmos3Config = serde_json::from_slice(&bytes)
        .with_context(|| format!("parsing cosmos3 config {}", path.display()))?;
    config.validate()?;
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL: &str = r#"{
        "type": "cosmos3_edge",
        "chunk_size": 32,
        "n_action_steps": 32,
        "max_action_dim": 64,
        "num_inference_steps": 4,
        "guidance_scale": 3.0,
        "sampling_shift": 5.0,
        "mode": "wam",
        "use_state": true,
        "latent_patch_size": 2,
        "num_embodiment_domains": 32,
        "timestep_embedding_dim": 256,
        "mot_backbone": {
            "text_config": {
                "vocab_size": 131072,
                "hidden_size": 2048,
                "intermediate_size": 9216,
                "num_hidden_layers": 28,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "hidden_act": "relu2",
                "rms_norm_eps": 1e-05,
                "joint_attn_implementation": "two_way",
                "rope_parameters": {
                    "rope_type": "default",
                    "rope_theta": 100000000,
                    "mrope_section": [24, 20, 20],
                    "mrope_interleaved": true
                }
            },
            "vision_config": {
                "hidden_size": 1152,
                "intermediate_size": 4304,
                "num_hidden_layers": 27,
                "num_attention_heads": 16,
                "num_channels": 3,
                "patch_size": 16,
                "layer_norm_eps": 1e-06,
                "hidden_act": "gelu_pytorch_tanh",
                "num_patches": 256,
                "spatial_merge_size": 2
            },
            "projector_config": {
                "input_hidden_size": 1152,
                "merger_intermediate_size": 11520,
                "out_hidden_size": 2048,
                "spatial_merge_size": 2,
                "use_postshuffle_norm": false
            }
        },
        "vae": {
            "base_dim": 160,
            "decoder_base_dim": 256,
            "z_dim": 48,
            "dim_mult": [1, 2, 4, 4],
            "num_res_blocks": 2,
            "attn_scales": [],
            "temperal_downsample": [false, true, true],
            "is_residual": true,
            "in_channels": 12,
            "out_channels": 12,
            "patch_size": 2,
            "scale_factor_temporal": 4,
            "scale_factor_spatial": 16
        }
    }"#;

    fn parse_str(text: &str) -> Result<Cosmos3Config> {
        let config: Cosmos3Config = serde_json::from_str(text)?;
        config.validate()?;
        Ok(config)
    }

    #[test]
    fn parses_the_reference_geometry() {
        let config = parse_str(MINIMAL).unwrap();
        assert_eq!(config.chunk_size, 32);
        assert_eq!(config.num_inference_steps, 4);
        assert_eq!(config.vision_tokens(), 64);
        assert_eq!(config.latent_patch_dim(), 4 * 48);
        assert_eq!(config.mot_backbone.text_config.head_dim, 128);
    }

    #[test]
    #[ignore = "需要真实 cosmos3 config.json"]
    fn parses_the_real_checkpoint_config() {
        let path = std::env::var("PNEUMA_COSMOS3_CONFIG").expect("PNEUMA_COSMOS3_CONFIG");
        let config = parse(Path::new(&path)).expect("parse real config");
        println!(
            "chunk {} steps {} z_dim {} mrope {:?}",
            config.chunk_size,
            config.num_inference_steps,
            config.vae.z_dim,
            config
                .mot_backbone
                .text_config
                .rope_parameters
                .as_ref()
                .map(|rope| rope.mrope_section)
        );
    }

    #[test]
    fn rejects_unsupported_switches() {
        assert!(parse_str(&MINIMAL.replace("\"mode\": \"wam\"", "\"mode\": \"vlm\"")).is_err());
        assert!(
            parse_str(&MINIMAL.replace(
                "\"joint_attn_implementation\": \"two_way\"",
                "\"joint_attn_implementation\": \"causal\""
            ))
            .is_err()
        );
        assert!(parse_str(&MINIMAL.replace("[24, 20, 20]", "[24, 20, 21]")).is_err());
        assert!(
            parse_str(&MINIMAL.replace("\"use_state\": true", "\"use_state\": false")).is_err()
        );
    }
}
