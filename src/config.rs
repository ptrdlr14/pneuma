//! TOML 部署配置与完整性校验；CLI 在加载后直接覆盖常用启动字段。

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use serde::Deserialize;

use crate::contract::FeatureSpec;
use crate::protocol::{Capabilities, deployment_prefix};

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Policy {
    Echo,
    Act,
    Smolvla,
}

#[derive(Clone, Copy, Debug, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Device {
    #[default]
    Auto,
    Flex,
    Metal,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    pub policy: Policy,
    #[serde(default)]
    pub repo_or_path: PathBuf,
    #[serde(default)]
    pub device: Device,
}

#[derive(Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ExecutionConfig {
    pub supported_modes: Vec<String>,
    pub action_fps: f64,
    pub action_deadline_s: f64,
    pub idle_timeout_s: f64,
    pub warmup_calls: usize,
    /// 模型策略从 checkpoint 读取；echo 必须显式给出预测步数。
    pub prediction_steps: Option<u64>,
    pub n_action_steps: Option<u64>,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            supported_modes: vec!["chunk".into()],
            action_fps: 30.0,
            action_deadline_s: 5.0,
            idle_timeout_s: 10.0,
            warmup_calls: 2,
            prediction_steps: None,
            n_action_steps: None,
        }
    }
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ZenohConfig {
    pub mode: String,
    pub listen_endpoints: Vec<String>,
    pub connect_endpoints: Vec<String>,
    pub config_file: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServerConfig {
    pub deployment: String,
    pub semantics: String,
    pub model: ModelConfig,
    #[serde(default)]
    pub execution: ExecutionConfig,
    pub zenoh: ZenohConfig,
    pub features: Vec<FeatureSpec>,
    pub action_feature: FeatureSpec,
    /// echo 的固定身份；模型策略未指定时由策略名与本地 checkpoint 路径派生。
    #[serde(default)]
    pub artifact_identity: Option<String>,
}

impl ServerConfig {
    /// 读取部署文件；CLI 覆盖完成后由调用方校验最终配置。
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading config {}", path.display()))?;
        Self::parse(&text).with_context(|| format!("loading config {}", path.display()))
    }

    fn parse(text: &str) -> Result<Self> {
        let document: toml::Value = toml::from_str(text).context("parsing TOML")?;
        // FeatureSpec 在线协议上允许默认值；配置入口另行严格检查字段拼写。
        for feature in document
            .get("features")
            .and_then(toml::Value::as_array)
            .into_iter()
            .flatten()
            .chain(document.get("action_feature"))
        {
            let fields = feature.as_table().context("feature must be a table")?;
            for key in fields.keys() {
                ensure!(
                    ["name", "shape", "dtype", "kind", "names", "semantics"]
                        .contains(&key.as_str()),
                    "unknown feature field {key}"
                );
            }
        }
        document.try_into().context("deserializing server config")
    }

    pub fn validate(&self) -> Result<()> {
        deployment_prefix(&self.deployment)?;
        ensure!(!self.semantics.trim().is_empty(), "semantics is required");
        let execution = &self.execution;
        ensure!(
            execution.supported_modes == ["chunk"],
            "execution.supported_modes must be [\"chunk\"]"
        );
        for (name, value) in [
            ("action_fps", execution.action_fps),
            ("action_deadline_s", execution.action_deadline_s),
            ("idle_timeout_s", execution.idle_timeout_s),
        ] {
            ensure!(
                value.is_finite() && value > 0.0,
                "execution.{name} must be finite and positive"
            );
        }
        ensure!(
            execution.warmup_calls > 0,
            "execution.warmup_calls must be positive"
        );
        self.validate_features()?;
        match self.model.policy {
            Policy::Echo => ensure!(
                execution.prediction_steps.is_some(),
                "echo requires execution.prediction_steps"
            ),
            Policy::Act | Policy::Smolvla => {
                ensure!(
                    !self.model.repo_or_path.as_os_str().is_empty(),
                    "model policy requires model.repo_or_path"
                );
                ensure!(
                    execution.prediction_steps.is_none(),
                    "prediction_steps comes from checkpoint; use execution.n_action_steps to limit execution"
                );
            }
        }
        if let Some(steps) = execution.prediction_steps {
            ensure!(steps > 0, "execution.prediction_steps must be positive");
        }
        if let Some(steps) = execution.n_action_steps {
            ensure!(steps > 0, "execution.n_action_steps must be positive");
            if let Some(prediction) = execution.prediction_steps {
                ensure!(
                    steps <= prediction,
                    "execution.n_action_steps exceeds prediction_steps"
                );
            }
        }
        ensure!(
            matches!(self.zenoh.mode.as_str(), "peer" | "client"),
            "zenoh.mode must be peer or client"
        );
        ensure!(
            !self.zenoh.listen_endpoints.is_empty() || !self.zenoh.connect_endpoints.is_empty(),
            "explicit Zenoh endpoints are required"
        );
        ensure!(
            self.zenoh.mode != "client"
                || (self.zenoh.listen_endpoints.is_empty()
                    && !self.zenoh.connect_endpoints.is_empty()),
            "Zenoh client mode requires connect endpoints and cannot listen"
        );
        ensure!(
            self.zenoh
                .listen_endpoints
                .iter()
                .chain(&self.zenoh.connect_endpoints)
                .all(|e| !e.trim().is_empty()),
            "Zenoh endpoints must not be empty"
        );
        Ok(())
    }

    fn validate_features(&self) -> Result<()> {
        ensure!(!self.features.is_empty(), "features must not be empty");
        let mut names = std::collections::HashSet::new();
        for feature in &self.features {
            ensure!(
                names.insert(&feature.name),
                "duplicate feature {}",
                feature.name
            );
        }
        for feature in self.features.iter().chain([&self.action_feature]) {
            ensure!(
                !feature.name.is_empty()
                    && !feature.shape.is_empty()
                    && feature.shape.iter().all(|&d| d > 0),
                "invalid feature name/shape: {}",
                feature.name
            );
            ensure!(
                crate::codec::dtype_size(&feature.dtype).is_some(),
                "unsupported dtype {}",
                feature.dtype
            );
            ensure!(
                !feature.semantics.is_empty(),
                "feature {} requires semantics",
                feature.name
            );
            ensure!(
                feature.names.is_empty()
                    || (feature.shape.len() == 1 && feature.names.len() as u64 == feature.shape[0]),
                "feature {} component names differ from shape",
                feature.name
            );
            ensure!(
                matches!(feature.kind.as_str(), "tensor" | "rgb"),
                "unsupported feature kind {}",
                feature.kind
            );
            if feature.kind == "rgb" {
                ensure!(
                    feature.dtype == "uint8" && feature.shape.len() == 3 && feature.shape[2] == 3,
                    "RGB feature {} must be H×W×3 uint8",
                    feature.name
                );
            }
        }
        ensure!(
            self.action_feature.name == "action"
                && self.action_feature.kind == "tensor"
                && self.action_feature.dtype == "float32"
                && self.action_feature.shape.len() == 1,
            "action_feature must be a one-dimensional float32 action tensor"
        );
        Ok(())
    }

    pub fn capabilities(&self, prediction_steps: u64, execution_steps: u64) -> Capabilities {
        Capabilities {
            modes: self.execution.supported_modes.clone(),
            prediction_steps,
            execution_steps,
            action_interval: 1.0 / self.execution.action_fps,
            features: self.features.clone(),
            action_feature: self.action_feature.clone(),
            language: false,
            training_max_delay: 0,
            rtc_horizon: 0,
            current_observation_only: true,
            retains_session_state: true,
            action_representation: "canonical".into(),
            model_action_dim: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const SO101: &str = include_str!("../examples/so101.toml");
    const ALOHA: &str = include_str!("../examples/aloha_sim.toml");

    #[test]
    fn examples_preserve_contract() {
        let config = ServerConfig::parse(SO101).unwrap();
        config.validate().unwrap();
        let caps = config.capabilities(50, 16);
        assert_eq!(caps.action_feature.shape, [6]);
        assert_eq!(caps.features[1].name, "observation.images.front");
        assert_eq!(caps.action_interval, 1.0 / 30.0);
        let config = ServerConfig::parse(ALOHA).unwrap();
        config.validate().unwrap();
        assert_eq!(config.action_feature.shape, [14]);
        assert_eq!(config.execution.action_fps, 50.0);
    }

    #[test]
    fn rejects_invalid_values_unknown_fields_and_incompatible_steps() {
        for (key, value) in [
            ("execution.action_fps", toml::Value::Integer(0)),
            ("execution.action_fps", toml::Value::Float(f64::NAN)),
            ("execution.action_fps", toml::Value::String("abc".into())),
            ("execution.n_action_steps", toml::Value::Integer(51)),
            ("execution.warmup_calls", toml::Value::Integer(0)),
            ("model.device", toml::Value::String("cuda".into())),
            ("zenoh.mode", toml::Value::String("router".into())),
            ("execution.typo", toml::Value::Integer(1)),
        ] {
            let (section, field) = key.split_once('.').unwrap();
            let mut document: toml::Value = toml::from_str(SO101).unwrap();
            document[section]
                .as_table_mut()
                .unwrap()
                .insert(field.into(), value);
            let error = ServerConfig::parse(&toml::to_string(&document).unwrap())
                .and_then(|config| config.validate())
                .unwrap_err();
            assert!(format!("{error:#}").contains(field), "{key}: {error:#}");
        }
        assert!(ServerConfig::parse(&SO101.replace("shape = [6]", "shpae = [6]")).is_err());
    }
}
