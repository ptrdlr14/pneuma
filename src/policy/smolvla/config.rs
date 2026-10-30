//! 官方 SmolVLA checkpoint 配置与本端口的运行时约束。

use std::{collections::HashMap, fmt, path::Path};

use anyhow::{Context, Result, ensure};
use serde::{
    Deserialize, Deserializer,
    de::{MapAccess, Visitor},
};

/// 特征形状沿用 checkpoint 的 CHW / 向量约定。
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub(super) struct PolicyFeature {
    #[serde(rename = "type")]
    pub feature_type: String,
    pub shape: Vec<usize>,
}

// 不经过 serde_json::Value，避免未启用 preserve_order 时相机顺序被排序。
fn ordered_features<'de, D>(
    deserializer: D,
) -> std::result::Result<Vec<(String, PolicyFeature)>, D::Error>
where
    D: Deserializer<'de>,
{
    struct FeaturesVisitor;
    impl<'de> Visitor<'de> for FeaturesVisitor {
        type Value = Vec<(String, PolicyFeature)>;
        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("an ordered feature object")
        }
        fn visit_map<A: MapAccess<'de>>(
            self,
            mut map: A,
        ) -> std::result::Result<Self::Value, A::Error> {
            let mut features = Vec::new();
            while let Some((name, feature)) = map.next_entry::<String, PolicyFeature>()? {
                if features.iter().any(|(key, _)| key == &name) {
                    return Err(serde::de::Error::custom(format!(
                        "duplicate feature {name}"
                    )));
                }
                features.push((name, feature));
            }
            Ok(features)
        }
    }
    deserializer.deserialize_map(FeaturesVisitor)
}

/// 推理有关字段与官方配置同名；训练、Hub 发布选项不影响推理。
#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub(super) struct SmolVlaConfig {
    #[serde(rename = "type")]
    pub policy_type: String,
    #[serde(deserialize_with = "ordered_features")]
    pub input_features: Vec<(String, PolicyFeature)>,
    #[serde(deserialize_with = "ordered_features")]
    pub output_features: Vec<(String, PolicyFeature)>,
    pub n_obs_steps: usize,
    pub chunk_size: usize,
    pub n_action_steps: usize,
    pub normalization_mapping: HashMap<String, String>,
    pub max_state_dim: usize,
    pub max_action_dim: usize,
    /// 官方顺序为 width、height；None 表示不缩放。
    pub resize_imgs_with_padding: Option<[usize; 2]>,
    pub empty_cameras: usize,
    pub adapt_to_pi_aloha: bool,
    pub use_delta_joint_actions_aloha: bool,
    pub tokenizer_max_length: usize,
    pub num_steps: usize,
    pub use_cache: bool,
    pub vlm_model_name: String,
    pub load_vlm_weights: bool,
    pub add_image_special_tokens: bool,
    pub attention_mode: String,
    pub prefix_length: i64,
    pub pad_language_to: String,
    pub num_expert_layers: i64,
    pub num_vlm_layers: i64,
    pub self_attn_every_n_layers: i64,
    pub expert_width_multiplier: f64,
    pub min_period: f64,
    pub max_period: f64,
    pub rtc_config: Option<serde_json::Value>,
}

impl Default for SmolVlaConfig {
    fn default() -> Self {
        Self {
            policy_type: "smolvla".into(),
            input_features: Vec::new(),
            output_features: Vec::new(),
            n_obs_steps: 1,
            chunk_size: 50,
            n_action_steps: 50,
            normalization_mapping: [
                ("VISUAL", "IDENTITY"),
                ("STATE", "MEAN_STD"),
                ("ACTION", "MEAN_STD"),
            ]
            .into_iter()
            .map(|(k, v)| (k.into(), v.into()))
            .collect(),
            max_state_dim: 32,
            max_action_dim: 32,
            resize_imgs_with_padding: Some([512, 512]),
            empty_cameras: 0,
            adapt_to_pi_aloha: false,
            use_delta_joint_actions_aloha: false,
            tokenizer_max_length: 48,
            num_steps: 10,
            use_cache: true,
            vlm_model_name: concat!("HuggingFaceTB/", "SmolVLM2-500M-Video-Instruct").into(),
            load_vlm_weights: false,
            add_image_special_tokens: false,
            attention_mode: "cross_attn".into(),
            prefix_length: -1,
            pad_language_to: "longest".into(),
            num_expert_layers: -1,
            num_vlm_layers: 16,
            self_attn_every_n_layers: 2,
            expert_width_multiplier: 0.75,
            min_period: 0.004,
            max_period: 4.0,
            rtc_config: None,
        }
    }
}

impl SmolVlaConfig {
    pub fn image_features(&self) -> impl Iterator<Item = (&str, &PolicyFeature)> {
        self.input_features
            .iter()
            .filter(|(_, f)| f.feature_type == "VISUAL")
            .map(|(n, f)| (n.as_str(), f))
    }
    pub fn state_dim(&self) -> usize {
        self.input_features
            .iter()
            .find(|(n, _)| n == "observation.state")
            .map_or(0, |(_, f)| f.shape[0])
    }
    pub fn action_dim(&self) -> usize {
        self.output_features
            .iter()
            .find(|(n, _)| n == "action")
            .map_or(0, |(_, f)| f.shape[0])
    }

    fn validate(&mut self) -> Result<()> {
        ensure!(self.policy_type == "smolvla", "expected type=smolvla");
        ensure!(self.n_obs_steps == 1, "SmolVLA supports only n_obs_steps=1");
        ensure!(self.use_cache, "SmolVLA requires use_cache=true");
        ensure!(self.rtc_config.is_none(), "SmolVLA RTC is unsupported");
        ensure!(
            !self.adapt_to_pi_aloha && !self.use_delta_joint_actions_aloha,
            "SmolVLA Aloha/delta adaptation is unsupported"
        );
        ensure!(
            self.empty_cameras == 0,
            "SmolVLA empty_cameras is unsupported"
        );
        ensure!(
            matches!(self.attention_mode.as_str(), "self_attn" | "cross_attn"),
            "unsupported attention_mode={}",
            self.attention_mode
        );
        // 本地路径或 Hub ID 仅记录来源；实际架构由显式 VLM 资源配置校验。
        // 此 backbone 有 32 层；官方 <=0 表示保留全部层。
        ensure!(
            self.num_vlm_layers >= -1 && self.num_vlm_layers <= 32,
            "num_vlm_layers must be -1, 0, or 1..=32"
        );
        if self.num_vlm_layers <= 0 {
            self.num_vlm_layers = 32;
        }
        ensure!(
            self.num_expert_layers >= -1,
            "num_expert_layers must be -1, 0, or positive"
        );
        if self.num_expert_layers <= 0 {
            self.num_expert_layers = self.num_vlm_layers;
        }
        ensure!(
            self.num_expert_layers == self.num_vlm_layers,
            "expert and VLM layer counts must match"
        );
        ensure!(
            self.self_attn_every_n_layers >= -1,
            "self_attn_every_n_layers must be >= -1"
        );
        ensure!(self.prefix_length >= -1, "prefix_length must be >= -1");
        ensure!(
            matches!(self.pad_language_to.as_str(), "longest" | "max_length"),
            "unsupported pad_language_to={}",
            self.pad_language_to
        );
        ensure!(
            self.chunk_size > 0
                && self.n_action_steps > 0
                && self.n_action_steps <= self.chunk_size,
            "n_action_steps must be in 1..=chunk_size"
        );
        ensure!(
            self.chunk_size.checked_mul(self.max_action_dim).is_some(),
            "action horizon size overflows"
        );
        ensure!(
            self.num_steps > 0 && self.tokenizer_max_length > 0,
            "num_steps and tokenizer_max_length must be positive"
        );
        ensure!(
            self.max_state_dim > 0 && self.max_action_dim > 0,
            "max_state_dim and max_action_dim must be positive"
        );
        ensure!(
            self.expert_width_multiplier.is_finite()
                && self.expert_width_multiplier > 0.0
                && (960.0 * self.expert_width_multiplier).floor() >= 1.0,
            "expert_width_multiplier must produce a positive hidden size"
        );
        ensure!(
            self.min_period.is_finite()
                && self.max_period.is_finite()
                && self.min_period > 0.0
                && self.min_period < self.max_period,
            "periods must satisfy 0 < min_period < max_period"
        );
        if let Some([w, h]) = self.resize_imgs_with_padding {
            ensure!(
                w > 0 && h > 0 && w.checked_mul(h).and_then(|v| v.checked_mul(3)).is_some(),
                "invalid resize_imgs_with_padding dimensions"
            );
        }
        let mut states = 0;
        for (name, feature) in &self.input_features {
            ensure!(
                !feature.shape.is_empty() && feature.shape.iter().all(|&d| d > 0),
                "invalid feature shape: {name}"
            );
            match feature.feature_type.as_str() {
                "STATE" => {
                    ensure!(
                        name == "observation.state"
                            && feature.shape.len() == 1
                            && feature.shape[0] <= self.max_state_dim,
                        "invalid state feature {name}"
                    );
                    states += 1;
                }
                "VISUAL" => {
                    ensure!(
                        name.starts_with("observation.images.")
                            && feature.shape.len() == 3
                            && feature.shape[0] == 3,
                        "invalid RGB feature {name}"
                    );
                    let h = feature.shape[1];
                    let w = feature.shape[2];
                    ensure!(
                        h.checked_mul(w).and_then(|v| v.checked_mul(3)).is_some(),
                        "RGB feature size overflows: {name}"
                    );
                    if let Some([target_w, target_h]) = self.resize_imgs_with_padding {
                        let ratio = (w as f64 / target_w as f64).max(h as f64 / target_h as f64);
                        ensure!(
                            (h as f64 / ratio) >= 1.0 && (w as f64 / ratio) >= 1.0,
                            "RGB aspect ratio produces an empty resized image: {name}"
                        );
                    }
                }
                other => anyhow::bail!("unsupported input feature type {other}: {name}"),
            }
        }
        ensure!(
            states == 1 && self.image_features().next().is_some(),
            "SmolVLA requires one state and at least one image"
        );
        ensure!(
            self.output_features.len() == 1,
            "SmolVLA requires only action output"
        );
        let (name, feature) = &self.output_features[0];
        ensure!(
            name == "action"
                && feature.feature_type == "ACTION"
                && feature.shape.len() == 1
                && feature.shape[0] > 0
                && feature.shape[0] <= self.max_action_dim,
            "invalid action feature"
        );
        for (kind, mode) in &self.normalization_mapping {
            ensure!(
                matches!(kind.as_str(), "VISUAL" | "STATE" | "ACTION"),
                "unsupported normalization feature type {kind}"
            );
            ensure!(
                matches!(
                    mode.as_str(),
                    "IDENTITY" | "MEAN_STD" | "MIN_MAX" | "QUANTILES" | "QUANTILE10"
                ),
                "unsupported normalization mode {mode}"
            );
        }
        ensure!(
            self.normalization_mapping
                .get("VISUAL")
                .is_none_or(|v| v == "IDENTITY"),
            "visual normalization must be IDENTITY"
        );
        Ok(())
    }
}

/// 解析一次配置，并在构建张量或加载权重前验证支持范围。
pub(super) fn parse(path: &Path) -> Result<SmolVlaConfig> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("reading SmolVLA config {}", path.display()))?;
    let mut config: SmolVlaConfig = serde_json::from_str(&raw)
        .with_context(|| format!("parsing SmolVLA config {}", path.display()))?;
    config
        .validate()
        .with_context(|| format!("validating SmolVLA config {}", path.display()))?;
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> SmolVlaConfig {
        serde_json::from_str(r#"{"input_features":{"observation.images.z":{"type":"VISUAL","shape":[3,2,4]},"observation.state":{"type":"STATE","shape":[6]},"observation.images.a":{"type":"VISUAL","shape":[3,2,4]}},"output_features":{"action":{"type":"ACTION","shape":[7]}}}"#).unwrap()
    }
    #[test]
    fn preserves_camera_order_and_resolves_default_layers() {
        let mut cfg = fixture();
        cfg.num_vlm_layers = 0;
        cfg.num_expert_layers = -1;
        cfg.validate().unwrap();
        assert_eq!(
            cfg.image_features().map(|(n, _)| n).collect::<Vec<_>>(),
            ["observation.images.z", "observation.images.a"]
        );
        assert_eq!((cfg.num_vlm_layers, cfg.num_expert_layers), (32, 32));
    }
    #[test]
    fn rejects_unsupported_runtime_and_bad_dimensions() {
        for patch in [
            r#"{"n_obs_steps":2}"#,
            r#"{"use_cache":false}"#,
            r#"{"rtc_config":{}}"#,
            r#"{"num_expert_layers":8}"#,
            r#"{"min_period":0}"#,
            r#"{"chunk_size":0}"#,
        ] {
            let mut value = serde_json::to_value(serde_json::json!({})).unwrap();
            value.as_object_mut().unwrap().extend(
                serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(patch).unwrap(),
            );
            let mut cfg = fixture();
            // 保留 feature fixture，仅覆盖待测配置。
            let inputs = cfg.input_features;
            let outputs = cfg.output_features;
            cfg = serde_json::from_value(value).unwrap();
            cfg.input_features = inputs;
            cfg.output_features = outputs;
            assert!(cfg.validate().is_err(), "accepted {patch}");
        }
    }
}
