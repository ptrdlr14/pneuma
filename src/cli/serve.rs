//! serve 子命令的参数组与配置覆盖。

use std::path::PathBuf;
use std::str::FromStr;

use clap::{Args, ValueEnum};

use crate::config::{Device, ServerConfig};
use crate::policy::PolicyKind;

/// 完整部署由 TOML 提供，命令行只调整本次启动需要的选项。
#[derive(Debug, Args)]
pub struct ServeArgs {
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
    /// 推理设备：auto|flex|metal|cuda|cuda:N（N 为 GPU 序号，同 PyTorch 写法）
    #[arg(long, value_parser = Device::from_str)]
    device: Option<Device>,
    /// 策略类型
    #[arg(long, value_enum)]
    policy: Option<PolicyKind>,
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
    /// zero 预测步数；模型策略从 checkpoint 读取
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

impl ServeArgs {
    /// 只应用显式传入的值，完整配置由服务构造入口统一校验。
    pub fn apply(self, config: &mut ServerConfig) {
        if let Some(path) = self.model.checkpoint {
            config.model.repo_or_path = path;
        }
        if let Some(device) = self.model.device {
            config.model.device = device;
        }
        if let Some(policy) = self.model.policy {
            config.model.policy = policy;
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
