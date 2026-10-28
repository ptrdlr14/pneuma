//! checkpoint `config.json` 的 HF 契约：serde 一次性解析，本端口不支持的取值显式报错。

use serde::Deserialize;

/// config.json 键名的忠实映射；缺键/错型都在 serde 层失败。
#[derive(Deserialize)]
struct HfActConfig {
    chunk_size: usize,
    n_action_steps: usize,
    dim_model: usize,
    n_heads: usize,
    dim_feedforward: usize,
    feedforward_activation: String,
    n_encoder_layers: usize,
    n_decoder_layers: usize,
    vision_backbone: String,
    pretrained_backbone_weights: Option<String>,
    replace_final_stride_with_dilation: bool,
    pre_norm: bool,
    use_vae: bool,
    latent_dim: usize,
}

/// 移植后支持的 ACT 超参；字段名用模块自己的语义。
#[derive(Clone, Copy, Debug)]
pub struct ActConfig {
    pub chunk_size: usize,
    pub n_action_steps: usize,
    pub dim_model: usize,
    pub n_heads: usize,
    pub dim_feedforward: usize,
    pub n_encoder_layers: usize,
    pub n_decoder_layers: usize,
    pub latent_dim: usize,
}

const SUPPORTED_BACKBONE: &str = "resnet18";

pub fn parse(path: &std::path::Path) -> anyhow::Result<ActConfig> {
    use anyhow::{Context, anyhow};
    let at = || path.display().to_string();
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("reading ACT config {}", path.display()))?;
    let hf: HfActConfig = serde_json::from_str(&raw)
        .with_context(|| format!("parsing ACT config {}", path.display()))?;
    if hf.vision_backbone != SUPPORTED_BACKBONE {
        return Err(anyhow!(
            "{}: vision_backbone={} 未移植",
            at(),
            hf.vision_backbone
        ));
    }
    if hf.replace_final_stride_with_dilation {
        return Err(anyhow!("{}: 膨胀卷积变体未移植", at()));
    }
    if hf.pre_norm {
        return Err(anyhow!("{}: pre_norm 未移植", at()));
    }
    if hf.feedforward_activation != "relu" {
        return Err(anyhow!(
            "{}: activation={}",
            at(),
            hf.feedforward_activation
        ));
    }
    if hf.use_vae {
        // 推理路径 z=0，VAE 编码器权重不加载；形态本身可支持。
        tracing::info!("use_vae checkpoint：推理按 z=0（eval 语义），跳过 vae_encoder 权重");
    }
    anyhow::ensure!(
        hf.chunk_size > 0 && hf.n_action_steps > 0 && hf.n_action_steps <= hf.chunk_size,
        "checkpoint n_action_steps must be in 1..=chunk_size"
    );
    let _ = hf.pretrained_backbone_weights;
    Ok(ActConfig {
        chunk_size: hf.chunk_size,
        n_action_steps: hf.n_action_steps,
        dim_model: hf.dim_model,
        n_heads: hf.n_heads,
        dim_feedforward: hf.dim_feedforward,
        n_encoder_layers: hf.n_encoder_layers,
        n_decoder_layers: hf.n_decoder_layers,
        latent_dim: hf.latent_dim,
    })
}
