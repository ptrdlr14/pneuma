//! 命令行入口：常见启动参数直接覆盖 TOML 部署配置。

use std::path::PathBuf;

use clap::{Args, Parser, ValueEnum};

use crate::config::{Device, Policy, ServerConfig};

/// 完整部署由 TOML 提供，命令行只调整本次启动需要的选项。
#[derive(Debug, Parser)]
#[command(name = "pneuma", version, about = "LeRobot remote-inference v1 服务端")]
pub struct Cli {
    /// 部署 TOML 文件
    #[arg(long)]
    pub config: PathBuf,
    #[command(flatten)]
    model: ModelArgs,
    #[command(flatten)]
    execution: ExecutionArgs,
    #[command(flatten)]
    network: NetworkArgs,
}

#[derive(Debug, Args)]
#[command(next_help_heading = "模型")]
struct ModelArgs {
    /// 本地 checkpoint 目录
    #[arg(long)]
    checkpoint: Option<PathBuf>,
    /// 推理设备
    #[arg(long, value_enum)]
    device: Option<DeviceArg>,
    /// 策略类型
    #[arg(long, value_enum)]
    policy: Option<PolicyArg>,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum DeviceArg {
    Auto,
    Flex,
    Metal,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum PolicyArg {
    Echo,
    Act,
}

#[derive(Debug, Args)]
#[command(next_help_heading = "执行")]
struct ExecutionArgs {
    /// 动作频率 Hz
    #[arg(long, allow_hyphen_values = true)]
    action_fps: Option<f64>,
    /// 每次预测后执行的动作步数
    #[arg(long)]
    action_steps: Option<u64>,
    /// echo 预测步数；ACT 从 checkpoint 读取
    #[arg(long)]
    prediction_steps: Option<u64>,
    /// 启动预热次数
    #[arg(long)]
    warmup_calls: Option<usize>,
}

#[derive(Debug, Args)]
#[command(next_help_heading = "网络")]
struct NetworkArgs {
    /// Zenoh 运行模式
    #[arg(long, value_enum)]
    zenoh_mode: Option<ZenohMode>,
    /// 监听地址；可重复指定，替换文件中的监听列表
    #[arg(long, conflicts_with = "clear_listen")]
    listen: Vec<String>,
    /// 连接地址；可重复指定，替换文件中的连接列表
    #[arg(long, conflicts_with = "clear_connect")]
    connect: Vec<String>,
    /// 清空文件中的监听地址列表
    #[arg(long)]
    clear_listen: bool,
    /// 清空文件中的连接地址列表
    #[arg(long)]
    clear_connect: bool,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ZenohMode {
    Peer,
    Client,
}

impl Cli {
    /// 只应用显式传入的值，完整配置由服务构造入口统一校验。
    pub fn apply(self, config: &mut ServerConfig) {
        if let Some(path) = self.model.checkpoint {
            config.model.repo_or_path = path;
        }
        if let Some(device) = self.model.device {
            config.model.device = match device {
                DeviceArg::Auto => Device::Auto,
                DeviceArg::Flex => Device::Flex,
                DeviceArg::Metal => Device::Metal,
            };
        }
        if let Some(policy) = self.model.policy {
            config.model.policy = match policy {
                PolicyArg::Echo => Policy::Echo,
                PolicyArg::Act => Policy::Act,
            };
        }
        if let Some(fps) = self.execution.action_fps {
            config.execution.action_fps = fps;
        }
        if let Some(steps) = self.execution.action_steps {
            config.execution.n_action_steps = Some(steps);
        }
        if let Some(steps) = self.execution.prediction_steps {
            config.execution.prediction_steps = Some(steps);
        }
        if let Some(calls) = self.execution.warmup_calls {
            config.execution.warmup_calls = calls;
        }
        if let Some(mode) = self.network.zenoh_mode {
            config.zenoh.mode = match mode {
                ZenohMode::Peer => "peer",
                ZenohMode::Client => "client",
            }
            .into();
        }
        if self.network.clear_listen || !self.network.listen.is_empty() {
            config.zenoh.listen_endpoints = self.network.listen;
        }
        if self.network.clear_connect || !self.network.connect.is_empty() {
            config.zenoh.connect_endpoints = self.network.connect;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load(args: &[&str]) -> ServerConfig {
        let cli = Cli::try_parse_from(
            ["pneuma", "--config=examples/so101.toml"]
                .into_iter()
                .chain(args.iter().copied()),
        )
        .unwrap();
        let mut config = ServerConfig::load(&cli.config).unwrap();
        cli.apply(&mut config);
        config.validate().unwrap();
        config
    }

    #[test]
    fn omitted_options_preserve_file_and_configuration_defaults() {
        let config = load(&[]);
        assert_eq!(config.execution.action_fps, 30.0);
        assert_eq!(config.execution.n_action_steps, Some(16));
        assert_eq!(config.execution.action_deadline_s, 5.0);
        assert_eq!(config.zenoh.listen_endpoints, ["tcp/127.0.0.1:7447"]);
    }

    #[test]
    fn typed_options_override_file_and_network_lists() {
        let config = load(&[
            "--device=flex",
            "--checkpoint=/tmp/model with spaces",
            "--policy=echo",
            "--action-fps=50",
            "--action-steps=20",
            "--prediction-steps=60",
            "--warmup-calls=1",
            "--zenoh-mode=client",
            "--clear-listen",
            "--connect=tcp/127.0.0.1:7555",
            "--connect=tcp/127.0.0.1:7666",
        ]);
        assert!(matches!(config.model.device, Device::Flex));
        assert_eq!(
            config.model.repo_or_path,
            PathBuf::from("/tmp/model with spaces")
        );
        assert_eq!(config.execution.action_fps, 50.0);
        assert_eq!(config.execution.n_action_steps, Some(20));
        assert_eq!(config.execution.prediction_steps, Some(60));
        assert_eq!(config.execution.warmup_calls, 1);
        assert!(config.zenoh.listen_endpoints.is_empty());
        assert_eq!(
            config.zenoh.connect_endpoints,
            ["tcp/127.0.0.1:7555", "tcp/127.0.0.1:7666"]
        );
        let config = load(&["--listen=tcp/127.0.0.1:7999", "--clear-connect"]);
        assert_eq!(config.zenoh.listen_endpoints, ["tcp/127.0.0.1:7999"]);
        assert!(config.zenoh.connect_endpoints.is_empty());
    }

    #[test]
    fn path_arguments_are_preserved_verbatim() {
        for value in [
            "with spaces",
            "'literal quotes'",
            r#""literal\nquotes""#,
            r"C:\models\act",
        ] {
            let arg = format!("--checkpoint={value}");
            let config = load(&[&arg]);
            assert_eq!(config.model.repo_or_path, PathBuf::from(value));
        }
    }

    #[test]
    fn overrides_are_validated_after_loading() {
        let cli = Cli::try_parse_from([
            "pneuma",
            "--config=examples/aloha_sim.toml",
            "--policy=echo",
            "--prediction-steps=100",
        ])
        .unwrap();
        let mut config = ServerConfig::load(&cli.config).unwrap();
        cli.apply(&mut config);
        config.validate().unwrap();
        let cli = Cli::try_parse_from([
            "pneuma",
            "--config=examples/so101.toml",
            "--action-steps=51",
        ])
        .unwrap();
        let mut config = ServerConfig::load(&cli.config).unwrap();
        cli.apply(&mut config);
        assert!(config.validate().is_err(), "执行步数超过预测长度必须被拒绝");
    }

    #[test]
    fn returns_help_version_and_argument_errors_without_exiting() {
        for (args, kind) in [
            (
                vec!["pneuma", "--help"],
                clap::error::ErrorKind::DisplayHelp,
            ),
            (
                vec!["pneuma", "--version"],
                clap::error::ErrorKind::DisplayVersion,
            ),
            (
                vec!["pneuma"],
                clap::error::ErrorKind::MissingRequiredArgument,
            ),
            (
                vec!["pneuma", "--config=x.toml", "--model.device=metal"],
                clap::error::ErrorKind::UnknownArgument,
            ),
            (
                vec!["pneuma", "--config=x.toml", "--device=cuda"],
                clap::error::ErrorKind::InvalidValue,
            ),
            (
                vec!["pneuma", "--config=x.toml", "--action-fps=abc"],
                clap::error::ErrorKind::ValueValidation,
            ),
            (
                vec![
                    "pneuma",
                    "--config=x.toml",
                    "--listen=tcp/127.0.0.1:7447",
                    "--clear-listen",
                ],
                clap::error::ErrorKind::ArgumentConflict,
            ),
            (
                vec![
                    "pneuma",
                    "--config=x.toml",
                    "--connect=tcp/127.0.0.1:7447",
                    "--clear-connect",
                ],
                clap::error::ErrorKind::ArgumentConflict,
            ),
        ] {
            let error = Cli::try_parse_from(args).unwrap_err();
            assert_eq!(error.kind(), kind, "{error}");
        }
    }
}
