//! 部署配置：TOML 文件与嵌套 CLI 覆盖共用字段，合并后一次性校验。

use std::ffi::OsString;
use std::path::PathBuf;

use anyhow::{Context, Result, ensure};
use clap::{Arg, Command};
use serde::Deserialize;

use crate::protocol::{Capabilities, FeatureSpec, deployment_prefix};

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Policy {
    Echo,
    Act,
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
    /// ACT 默认从 checkpoint 读取；echo 必须显式给出预测步数。
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
    /// echo 的固定身份；ACT 未指定时由本地 checkpoint 路径派生。
    #[serde(default)]
    pub artifact_identity: Option<String>,
}

/// CLI 字段与 TOML 路径同名；值用 TOML 语法，字符串也允许不加引号。
#[derive(Clone, Copy)]
enum OverrideKind {
    String,
    Toml,
}

const OVERRIDES: &[(&str, &str, OverrideKind)] = &[
    ("deployment", "部署名称", OverrideKind::String),
    ("semantics", "状态/动作语义", OverrideKind::String),
    ("artifact_identity", "部署产物身份", OverrideKind::String),
    ("model.policy", "echo 或 act", OverrideKind::String),
    (
        "model.repo_or_path",
        "本地 checkpoint 目录",
        OverrideKind::String,
    ),
    ("model.device", "auto、flex 或 metal", OverrideKind::String),
    (
        "execution.supported_modes",
        "支持模式数组，目前仅 [\"chunk\"]",
        OverrideKind::Toml,
    ),
    ("execution.action_fps", "动作频率 Hz", OverrideKind::Toml),
    (
        "execution.action_deadline_s",
        "推理截止时间（秒）",
        OverrideKind::Toml,
    ),
    (
        "execution.idle_timeout_s",
        "客户端缺席宽限（秒）",
        OverrideKind::Toml,
    ),
    ("execution.warmup_calls", "启动预热次数", OverrideKind::Toml),
    (
        "execution.prediction_steps",
        "echo 预测步数；ACT 从 checkpoint 读取",
        OverrideKind::Toml,
    ),
    (
        "execution.n_action_steps",
        "执行片段步数",
        OverrideKind::Toml,
    ),
    ("zenoh.mode", "peer 或 client", OverrideKind::String),
    (
        "zenoh.listen_endpoints",
        "监听地址数组，例如 [\"tcp/127.0.0.1:7447\"]",
        OverrideKind::Toml,
    ),
    (
        "zenoh.connect_endpoints",
        "连接地址数组",
        OverrideKind::Toml,
    ),
    (
        "zenoh.config_file",
        "Zenoh JSON5 配置底座",
        OverrideKind::String,
    ),
    (
        "features",
        "观测特征的 TOML inline table 数组",
        OverrideKind::Toml,
    ),
    (
        "action_feature",
        "动作特征的 TOML inline table",
        OverrideKind::Toml,
    ),
    ("action_feature.name", "动作特征名", OverrideKind::String),
    ("action_feature.shape", "动作形状数组", OverrideKind::Toml),
    ("action_feature.dtype", "动作 dtype", OverrideKind::String),
    ("action_feature.kind", "动作种类", OverrideKind::String),
    ("action_feature.names", "动作分量名数组", OverrideKind::Toml),
    ("action_feature.semantics", "动作语义", OverrideKind::String),
];

impl ServerConfig {
    pub fn command() -> Command {
        let mut command = Command::new("pneuma")
            .version(env!("CARGO_PKG_VERSION"))
            .about("LeRobot remote-inference v1 服务端（TOML 配置 + CLI 覆盖）")
            .arg(
                Arg::new("config_path")
                    .long("config_path")
                    .required(true)
                    .value_parser(clap::value_parser!(PathBuf))
                    .help("部署 TOML 文件"),
            );
        for &(name, help, _) in OVERRIDES {
            command = command.arg(
                Arg::new(name)
                    .long(name)
                    .help(help)
                    .allow_hyphen_values(true),
            );
        }
        command
    }

    pub fn from_args(args: impl IntoIterator<Item = impl Into<OsString> + Clone>) -> Result<Self> {
        // help/version 使用 clap 的成功退出码；其他参数错误使用其标准诊断。
        let matches = Self::command()
            .try_get_matches_from(args)
            .unwrap_or_else(|e| e.exit());
        let path = matches
            .get_one::<PathBuf>("config_path")
            .context("missing --config_path")?;
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading config {}", path.display()))?;
        let overrides = OVERRIDES.iter().filter_map(|&(name, _, _)| {
            matches
                .get_one::<String>(name)
                .map(|value| (name, value.as_str()))
        });
        Self::parse(&text, overrides).with_context(|| format!("loading config {}", path.display()))
    }

    fn parse<'a>(
        text: &str,
        overrides: impl IntoIterator<Item = (&'a str, &'a str)>,
    ) -> Result<Self> {
        let mut document: toml::Value = toml::from_str(text).context("parsing TOML")?;
        for (path, raw) in overrides {
            let kind = OVERRIDES
                .iter()
                .find(|&&(name, _, _)| name == path)
                .map(|&(_, _, kind)| kind)
                .unwrap_or(OverrideKind::Toml);
            let parsed = format!("value = {raw}").parse::<toml::Table>();
            let value = if matches!(kind, OverrideKind::String) && !raw.starts_with(['\"', '\'']) {
                toml::Value::String(raw.to_string())
            } else {
                let mut table =
                    parsed.with_context(|| format!("invalid TOML value for --{path}"))?;
                ensure!(table.len() == 1, "--{path} must contain one TOML value");
                table.remove("value").context("missing override value")?
            };
            let parts: Vec<_> = path.split('.').collect();
            let mut table = document.as_table_mut().context("config must be a table")?;
            for part in &parts[..parts.len() - 1] {
                table = table
                    .entry((*part).to_string())
                    .or_insert_with(|| toml::Value::Table(toml::Table::new()))
                    .as_table_mut()
                    .with_context(|| format!("{path}: parent must be a table"))?;
            }
            table.insert(parts[parts.len() - 1].to_string(), value);
        }
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
        let config: Self = document.try_into().context("deserializing server config")?;
        config.validate()?;
        Ok(config)
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
        match self.model.policy {
            Policy::Echo => ensure!(
                execution.prediction_steps.is_some(),
                "echo requires execution.prediction_steps"
            ),
            Policy::Act => {
                ensure!(
                    !self.model.repo_or_path.as_os_str().is_empty(),
                    "act requires model.repo_or_path"
                );
                ensure!(
                    execution.prediction_steps.is_none(),
                    "ACT prediction_steps comes from checkpoint; use execution.n_action_steps to limit execution"
                );
                let state = self.features.iter().find(|f| f.name == "observation.state");
                let images: Vec<_> = self.features.iter().filter(|f| f.kind == "rgb").collect();
                ensure!(
                    self.features.len() == 2
                        && state.is_some_and(|f| f.shape == [14]
                            && f.dtype == "float32"
                            && f.kind == "tensor")
                        && images.len() == 1
                        && images[0].name == "observation.images.top"
                        && images[0].shape == [480, 640, 3]
                        && self.action_feature.shape == [14],
                    "ACT requires observation.state [14], observation.images.top [480,640,3], and action [14]"
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
    fn examples_preserve_contract_and_cli_overrides_win() {
        let config = ServerConfig::parse(SO101, []).unwrap();
        let caps = config.capabilities(50, 16);
        assert_eq!(caps.action_feature.shape, [6]);
        assert_eq!(caps.features[1].name, "observation.images.front");
        assert_eq!(caps.action_interval, 1.0 / 30.0);
        let config = ServerConfig::parse(
            ALOHA,
            [
                ("model.repo_or_path", "/tmp/checkpoint"),
                ("execution.action_fps", "25"),
                ("zenoh.listen_endpoints", "[\"tcp/127.0.0.1:7555\"]"),
            ],
        )
        .unwrap();
        assert_eq!(config.action_feature.shape, [14]);
        assert_eq!(config.execution.action_fps, 25.0);
        assert_eq!(config.zenoh.listen_endpoints, ["tcp/127.0.0.1:7555"]);
        assert_eq!(config.model.repo_or_path, PathBuf::from("/tmp/checkpoint"));
        let config = ServerConfig::from_args([
            "pneuma",
            "--config_path",
            "examples/so101.toml",
            "--deployment=123",
            "--execution.action_fps",
            "50",
            "--zenoh.mode=client",
            "--zenoh.listen_endpoints=[]",
            "--zenoh.connect_endpoints=[\"tcp/127.0.0.1:7555\"]",
        ])
        .unwrap();
        assert_eq!(config.deployment, "123", "字符串字段保留数字文本");
        assert_eq!(config.execution.action_fps, 50.0);
        assert!(config.zenoh.listen_endpoints.is_empty());
        assert_eq!(config.zenoh.connect_endpoints, ["tcp/127.0.0.1:7555"]);
        let config = ServerConfig::parse(SO101, [("execution", "{ action_fps = 50 }")]);
        assert!(
            config.is_err(),
            "replacing execution loses required echo prediction_steps"
        );
    }

    #[test]
    fn rejects_invalid_values_unknown_fields_and_incompatible_steps() {
        for (key, value) in [
            ("execution.action_fps", "0"),
            ("execution.action_fps", "nan"),
            ("execution.action_fps", "abc"),
            ("execution.n_action_steps", "51"),
            ("execution.warmup_calls", "0"),
            ("model.device", "cuda"),
            ("zenoh.mode", "router"),
            ("execution.typo", "1"),
        ] {
            let error = ServerConfig::parse(SO101, [(key, value)]).unwrap_err();
            assert!(
                format!("{error:#}").contains(key.split('.').next_back().unwrap()),
                "{key}: {error:#}"
            );
        }
        assert!(
            ServerConfig::parse(SO101, [("features", "[{ name = 'bad', shpae = [6] }]")]).is_err()
        );
        assert!(
            ServerConfig::command()
                .try_get_matches_from(["pneuma", "--config_path=x.toml", "--execution.typo=1"])
                .is_err()
        );
    }
}
