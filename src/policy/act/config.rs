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
    use anyhow::Context;
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("reading ACT config {}", path.display()))?;
    let hf: HfActConfig = serde_json::from_str(&raw)
        .with_context(|| format!("parsing ACT config {}", path.display()))?;
    validate(&hf).with_context(|| format!("validating ACT config {}", path.display()))
}

fn validate(hf: &HfActConfig) -> anyhow::Result<ActConfig> {
    use anyhow::anyhow;
    if hf.vision_backbone != SUPPORTED_BACKBONE {
        return Err(anyhow!(
            "ACT config: vision_backbone={} 未移植",
            hf.vision_backbone
        ));
    }
    if hf.replace_final_stride_with_dilation {
        return Err(anyhow!("ACT config: 膨胀卷积变体未移植"));
    }
    if hf.pre_norm {
        return Err(anyhow!("ACT config: pre_norm 未移植"));
    }
    if hf.feedforward_activation != "relu" {
        return Err(anyhow!(
            "ACT config: activation={}",
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
    for (name, value) in [
        ("dim_model", hf.dim_model),
        ("n_heads", hf.n_heads),
        ("dim_feedforward", hf.dim_feedforward),
        ("n_encoder_layers", hf.n_encoder_layers),
        ("n_decoder_layers", hf.n_decoder_layers),
        ("latent_dim", hf.latent_dim),
    ] {
        anyhow::ensure!(value > 0, "checkpoint {name} must be positive");
    }
    anyhow::ensure!(
        hf.dim_model.is_multiple_of(hf.n_heads),
        "checkpoint dim_model must be divisible by n_heads"
    );
    anyhow::ensure!(
        hf.dim_model.is_multiple_of(4),
        "checkpoint dim_model must be divisible by 4 for paired x/y sinusoidal positions"
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

#[cfg(test)]
mod tests {
    use super::*;

    fn metadata() -> serde_json::Value {
        serde_json::json!({
            "chunk_size": 4, "n_action_steps": 2, "dim_model": 8,
            "n_heads": 2, "dim_feedforward": 16, "feedforward_activation": "relu",
            "n_encoder_layers": 1, "n_decoder_layers": 1, "vision_backbone": "resnet18",
            "pretrained_backbone_weights": null, "replace_final_stride_with_dilation": false,
            "pre_norm": false, "use_vae": true, "latent_dim": 2
        })
    }

    #[test]
    fn invalid_numeric_metadata_is_rejected_before_model_construction() {
        assert!(validate(&serde_json::from_value(metadata()).unwrap()).is_ok());
        for field in [
            "dim_model",
            "n_heads",
            "dim_feedforward",
            "n_encoder_layers",
            "n_decoder_layers",
            "latent_dim",
        ] {
            let mut value = metadata();
            value[field] = 0.into();
            let error = validate(&serde_json::from_value(value).unwrap()).unwrap_err();
            assert!(error.to_string().contains(field), "{field}: {error}");
        }
        for (dim, heads) in [(8, 3), (9, 3)] {
            let mut value = metadata();
            value["dim_model"] = dim.into();
            value["n_heads"] = heads.into();
            assert!(validate(&serde_json::from_value(value).unwrap()).is_err());
        }
    }
}
