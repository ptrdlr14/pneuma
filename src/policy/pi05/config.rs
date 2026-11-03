//! 官方 π₀.₅ checkpoint 配置与本端口的运行时约束。

use std::{collections::HashMap, fmt, path::Path};

use crate::policy::PolicyKind;
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

// 不经过 serde_json::Value，避免启用 preserve_order 前相机顺序被排序。
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
                if features
                    .iter()
                    .any(|(key, _): &(String, PolicyFeature)| key == &name)
                {
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

/// 推理有关字段与官方配置同名；训练、优化器与发布选项不影响推理。
#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub(super) struct Pi05Config {
    #[serde(rename = "type")]
    pub policy_type: String,
    #[serde(deserialize_with = "ordered_features")]
    pub input_features: Vec<(String, PolicyFeature)>,
    #[serde(deserialize_with = "ordered_features")]
    pub output_features: Vec<(String, PolicyFeature)>,
    pub n_obs_steps: usize,
    pub chunk_size: usize,
    pub n_action_steps: usize,
    pub max_state_dim: usize,
    pub max_action_dim: usize,
    pub num_inference_steps: usize,
    pub min_period: f64,
    pub max_period: f64,
    pub image_resolution: [usize; 2],
    pub empty_cameras: usize,
    pub tokenizer_max_length: usize,
    pub paligemma_variant: String,
    pub action_expert_variant: String,
    /// checkpoint 的计算精度；官方嵌入缩放按该精度取整。
    pub dtype: String,
    pub normalization_mapping: HashMap<String, String>,
}

impl Default for Pi05Config {
    fn default() -> Self {
        Self {
            policy_type: PolicyKind::Pi05.as_str().into(),
            input_features: Vec::new(),
            output_features: Vec::new(),
            n_obs_steps: 1,
            chunk_size: 50,
            n_action_steps: 50,
            max_state_dim: 32,
            max_action_dim: 32,
            num_inference_steps: 10,
            min_period: 4e-3,
            max_period: 4.0,
            image_resolution: [224, 224],
            empty_cameras: 0,
            tokenizer_max_length: 200,
            paligemma_variant: "gemma_2b".into(),
            action_expert_variant: "gemma_300m".into(),
            dtype: "float32".into(),
            normalization_mapping: HashMap::new(),
        }
    }
}

impl Pi05Config {
    /// 真实相机按 checkpoint 顺序排列；补零相机由 `empty_cameras` 表达，不要求客户端发送。
    pub fn image_features(&self) -> impl Iterator<Item = &(String, PolicyFeature)> {
        self.input_features
            .iter()
            .filter(|(name, feature)| feature.feature_type == "VISUAL" && !is_empty_camera(name))
    }

    pub fn state_feature(&self) -> Result<&PolicyFeature> {
        let mut states = self
            .input_features
            .iter()
            .filter(|(_, feature)| feature.feature_type == "STATE")
            .map(|(_, feature)| feature);
        let state = states
            .next()
            .context("pi05 config requires one STATE feature")?;
        ensure!(states.next().is_none(), "pi05 supports one STATE feature");
        Ok(state)
    }

    pub fn action_feature(&self) -> Result<&PolicyFeature> {
        let mut actions = self
            .output_features
            .iter()
            .filter(|(name, _)| name == "action")
            .map(|(_, feature)| feature);
        let action = actions
            .next()
            .context("pi05 config requires an action feature")?;
        ensure!(actions.next().is_none(), "pi05 supports one action feature");
        ensure!(
            action.feature_type == "ACTION",
            "action feature has wrong type"
        );
        Ok(action)
    }

    pub fn state_dim(&self) -> usize {
        self.state_feature().map_or(0, |f| f.shape[0])
    }

    pub fn action_dim(&self) -> usize {
        self.action_feature().map_or(0, |f| f.shape[0])
    }

    /// 推理期必需的架构常量；未知变体或尺寸在构造模块前拒绝。
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.policy_type == PolicyKind::Pi05.as_str(),
            "config type must be pi05"
        );
        ensure!(
            self.paligemma_variant == "gemma_2b" && self.action_expert_variant == "gemma_300m",
            "only PaliGemma gemma_2b with gemma_300m expert is supported"
        );
        ensure!(
            matches!(self.dtype.as_str(), "float32" | "bfloat16"),
            "unsupported pi05 dtype {}",
            self.dtype
        );
        let state = self.state_feature()?;
        ensure!(
            state.shape.len() == 1 && state.shape[0] > 0,
            "state feature must be a non-empty vector"
        );
        let action = self.action_feature()?;
        ensure!(
            action.shape.len() == 1 && action.shape[0] > 0,
            "action feature must be a non-empty vector"
        );
        ensure!(
            action.shape[0] <= self.max_action_dim && state.shape[0] <= self.max_state_dim,
            "feature width exceeds max_action_dim or max_state_dim"
        );
        ensure!(
            self.n_obs_steps == 1,
            "pi05 supports a single observation step"
        );
        ensure!(
            self.chunk_size > 0 && self.n_action_steps <= self.chunk_size,
            "invalid chunk_size or n_action_steps"
        );
        ensure!(
            self.max_action_dim > 0 && self.max_action_dim.is_multiple_of(2),
            "max_action_dim must be positive and even"
        );
        ensure!(
            self.num_inference_steps > 0 && self.num_inference_steps <= 100,
            "num_inference_steps must be in 1..=100"
        );
        ensure!(
            self.min_period.is_finite()
                && self.max_period.is_finite()
                && self.min_period > 0.0
                && self.max_period > self.min_period,
            "invalid flow-matching time periods"
        );
        ensure!(
            self.image_resolution == [224, 224],
            "pi05 deployment requires 224x224 prepared images"
        );
        ensure!(
            self.tokenizer_max_length > 0 && self.tokenizer_max_length <= 512,
            "tokenizer_max_length must be in 1..=512"
        );
        let images = self.image_features().count();
        ensure!(images > 0, "pi05 requires at least one camera");
        for (name, feature) in self.image_features() {
            ensure!(
                feature.shape.len() == 3
                    && feature.shape[0] == 3
                    && feature.shape[1] > 0
                    && feature.shape[2] > 0,
                "camera {name} must be CHW with three channels"
            );
        }
        for kind in ["VISUAL", "STATE", "ACTION"] {
            let mode = self
                .normalization_mapping
                .get(kind)
                .map_or("IDENTITY", String::as_str);
            ensure!(
                matches!(mode, "IDENTITY" | "MEAN_STD" | "MIN_MAX" | "QUANTILES"),
                "unsupported normalization mode {mode} for {kind}"
            );
        }
        Ok(())
    }
}

pub(super) fn is_empty_camera(name: &str) -> bool {
    name.starts_with("observation.images.empty_camera_")
}

pub(super) fn parse(path: &Path) -> Result<Pi05Config> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let config: Pi05Config = serde_json::from_slice(&bytes)
        .with_context(|| format!("parsing pi05 config {}", path.display()))?;
    config.validate()?;
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL: &str = r#"{
        "type": "pi05",
        "input_features": {
            "observation.images.image": {"type": "VISUAL", "shape": [3, 256, 256]},
            "observation.images.image2": {"type": "VISUAL", "shape": [3, 256, 256]},
            "observation.state": {"type": "STATE", "shape": [8]}
        },
        "output_features": {"action": {"type": "ACTION", "shape": [7]}},
        "empty_cameras": 1,
        "paligemma_variant": "gemma_2b",
        "action_expert_variant": "gemma_300m"
    }"#;

    fn parse_str(text: &str) -> Result<Pi05Config> {
        let config: Pi05Config = serde_json::from_str(text)?;
        config.validate()?;
        Ok(config)
    }

    #[test]
    fn defaults_match_the_official_architecture() {
        let config = parse_str(MINIMAL).unwrap();
        assert_eq!(config.chunk_size, 50);
        assert_eq!(config.max_action_dim, 32);
        assert_eq!(config.num_inference_steps, 10);
        assert_eq!(config.image_resolution, [224, 224]);
        assert_eq!(config.state_dim(), 8);
        assert_eq!(config.action_dim(), 7);
        assert_eq!(config.image_features().count(), 2);
    }

    #[test]
    fn empty_cameras_are_not_required_from_the_client() {
        let config = parse_str(
            &MINIMAL.replace(
                "\"observation.images.image2\": {\"type\": \"VISUAL\", \"shape\": [3, 256, 256]},",
                "\"observation.images.image2\": {\"type\": \"VISUAL\", \"shape\": [3, 256, 256]},\n            \"observation.images.empty_camera_0\": {\"type\": \"VISUAL\", \"shape\": [3, 224, 224]},",
            ),
        )
        .unwrap();
        assert_eq!(config.image_features().count(), 2);
        assert_eq!(config.input_features.len(), 4);
    }

    #[test]
    fn unsupported_variants_and_shapes_are_rejected() {
        assert!(parse_str(&MINIMAL.replace("gemma_2b", "gemma_7b")).is_err());
        assert!(parse_str(&MINIMAL.replace("\"shape\": [7]", "\"shape\": [33]")).is_err());
        assert!(parse_str(&MINIMAL.replace("\"shape\": [8]", "\"shape\": [33]")).is_err());
        assert!(
            parse_str(&MINIMAL.replace("\"empty_cameras\": 1", "\"empty_cameras\": 4")).is_ok()
        );
    }
}
