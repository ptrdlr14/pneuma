//! convert 子命令的参数与执行入口。

use std::path::PathBuf;

use clap::Args;

use crate::convert::{ConvertOptions, Format};

#[derive(Debug, Args)]
pub struct ConvertArgs {
    #[arg(long, value_enum)]
    pub(super) from: Format,
    #[arg(long, value_enum)]
    pub(super) to: Format,
    #[arg(long)]
    pub(super) input: PathBuf,
    #[arg(long)]
    pub(super) output: PathBuf,
    /// 导入 SmolVLA 时的本地 VLM 资源
    #[arg(long)]
    pub(super) vlm_path: Option<PathBuf>,
    /// checkpoint 缺少特征描述时（如 cosmos3）提供合约的 JSON 文件
    #[arg(long)]
    pub(super) features: Option<PathBuf>,
}

impl ConvertArgs {
    pub fn run(self) -> anyhow::Result<()> {
        crate::convert::run(&ConvertOptions {
            from: self.from,
            to: self.to,
            input: self.input,
            output: self.output,
            vlm_path: self.vlm_path,
            features: self.features,
        })
    }
}
