//! 命令行入口：常见启动参数直接覆盖 TOML 部署配置。

mod convert;
mod serve;

use clap::{Parser, Subcommand, ValueEnum};

use crate::policy::PolicyKind;

pub use convert::ConvertArgs;
pub use serve::ServeArgs;

/// 原生策略服务与 LeRobot checkpoint 双向转换。
#[derive(Debug, Parser)]
#[command(name = "pneuma", version, about = "Burn 策略运行时与 checkpoint 转换")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// 启动原生 bpk checkpoint 服务
    Serve(ServeArgs),
    /// 转换 LeRobot 与 Pneuma checkpoint
    Convert(ConvertArgs),
}

impl ValueEnum for PolicyKind {
    fn value_variants<'a>() -> &'a [Self] {
        &PolicyKind::ALL
    }

    fn to_possible_value(&self) -> Option<clap::builder::PossibleValue> {
        Some(clap::builder::PossibleValue::new(self.as_str()))
    }
}
