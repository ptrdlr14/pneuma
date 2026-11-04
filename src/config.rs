//! TOML 部署配置与完整性校验；CLI 在加载后直接覆盖常用启动字段。

use std::path::{Path, PathBuf};
use std::str::FromStr;

use anyhow::{Context, Result, ensure};
use serde::Deserialize;

use crate::contract::FeatureSpec;
use crate::policy::PolicyKind;
use crate::protocol::{Capabilities, deployment_prefix};

/// NVIDIA GPU 的硬件序号；`cuda` 无后缀时由后端自选（通常 0 号卡）。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CudaIndex(pub Option<usize>);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Device {
    #[default]
    Auto,
    Flex,
    Metal,
    /// NVIDIA GPU；`CudaIndex(None)` 由后端自选。
    Cuda(CudaIndex),
}

impl Device {
    pub const CUDA_DEFAULT: Device = Device::Cuda(CudaIndex(None));
}

/// 所有策略共用一处映射，避免各模型各自分支。
impl From<Device> for burn::tensor::Device {
    fn from(device: Device) -> Self {
        use burn::tensor::{Device as Burn, DeviceIndex, DeviceKind};
        // `Auto` 由 Burn 按 CUDA → Metal → ROCm → … 顺序挑选已编译的后端，
        // 并接受 BURN_DEVICE 覆盖；这里不重复那份顺序。
        match device {
            Device::Auto => Burn::default(),
            Device::Flex => Burn::flex(),
            Device::Metal => Burn::metal(DeviceKind::default()),
            Device::Cuda(CudaIndex(index)) => Burn::cuda(match index {
                Some(index) => DeviceIndex::Specified(index),
                None => DeviceIndex::Default,
            }),
        }
    }
}

/// 与 PyTorch `cuda`、`cuda:7` 同一套写法，不接受其它分隔符或空白。
impl FromStr for Device {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        let (kind, index) = match value.split_once(':') {
            Some((kind, index)) => {
                // `cuda:` 写了冒号却没给序号，多半是笔误；当成 `cuda` 会跑到非预期的卡上。
                ensure!(!index.is_empty(), "device index is empty in {value:?}");
                (kind, Some(index))
            }
            None => (value, None),
        };
        let index = index
            .map(|index| {
                index
                    .parse::<usize>()
                    .with_context(|| format!("invalid device index in {value:?}"))
            })
            .transpose()?;
        Ok(match kind {
            "auto" => Device::Auto,
            "flex" => Device::Flex,
            "metal" => Device::Metal,
            "cuda" => Device::Cuda(CudaIndex(index)),
            other => anyhow::bail!("unknown device {other:?}"),
        })
    }
}

/// 手写实现以接受 `cuda:7`；`#[serde(rename_all)]` 无法表达带参数的变体。
impl<'de> Deserialize<'de> for Device {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = String::deserialize(deserializer)?;
        value.parse().map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    pub policy: PolicyKind,
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
    /// 模型策略从 checkpoint 读取；zero 必须显式给出预测步数。
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
    /// zero 的固定身份；模型策略未指定时由策略名与本地 checkpoint 路径派生。
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
            PolicyKind::Zero => ensure!(
                execution.prediction_steps.is_some(),
                "zero requires execution.prediction_steps"
            ),
            PolicyKind::Act | PolicyKind::Smolvla | PolicyKind::Pi05 | PolicyKind::Cosmos3Edge => {
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
            ("model.device", toml::Value::String("vulkan".into())),
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

    #[test]
    fn device_parses_pytorch_style_cuda_index() {
        // 与 PyTorch 同一套写法：无后缀由后端自选，`:N` 指定硬件序号。
        for (text, expected) in [
            ("auto", Device::Auto),
            ("flex", Device::Flex),
            ("metal", Device::Metal),
            ("cuda", Device::CUDA_DEFAULT),
            ("cuda:7", Device::Cuda(CudaIndex(Some(7)))),
            ("cuda:0", Device::Cuda(CudaIndex(Some(0)))),
        ] {
            assert_eq!(
                text.parse::<Device>()
                    .unwrap_or_else(|e| panic!("{text}: {e}")),
                expected
            );
        }
        // 无效写法要报错，不能被当成 auto 静默通过。
        for bad in ["vulkan", "cuda:", "cuda:x", "cuda:-1", "cuda:1:2", "CUDA:7"] {
            assert!(bad.parse::<Device>().is_err(), "{bad} must be rejected");
        }
    }

    #[test]
    fn toml_accepts_cuda_index() {
        let config =
            ServerConfig::parse(&SO101.replace("device = \"auto\"", "device = \"cuda:7\""))
                .expect("device=cuda:7 must parse");
        assert_eq!(config.model.device, Device::Cuda(CudaIndex(Some(7))));
    }

    /// 后端真机自检：解析 `PNEUMA_DEVICE`（默认 `cuda`），建设备并跑一次矩阵乘。
    /// 运行：`PNEUMA_DEVICE=cuda:0 cargo test -- --ignored device_backend_runs_a_matmul --nocapture`
    #[test]
    #[ignore = "需要真实设备；用 PNEUMA_DEVICE 指定后端"]
    fn device_backend_runs_a_matmul() {
        use burn::tensor::Tensor;

        let text = std::env::var("PNEUMA_DEVICE").unwrap_or_else(|_| "cuda".into());
        let device = text
            .parse::<Device>()
            .unwrap_or_else(|e| panic!("PNEUMA_DEVICE={text}: {e}"))
            .into();

        // 全 1 的 [2,3] 乘全 2 的 [3,2]，每个元素应为 3×2 = 6。
        let a = Tensor::<2>::ones([2, 3], &device);
        let b = Tensor::<2>::ones([3, 2], &device) * 2.0;
        let got = a.matmul(b).into_data().try_to_vec::<f32>().unwrap();
        assert_eq!(got, vec![6.0; 4], "{text}");
    }
}
