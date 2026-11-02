//! pneuma: lerobot remote-inference v1 协议的 Rust 服务端骨架。
//! 部署由 --config 指定 TOML，常用 CLI 选项调整本次启动配置。

// zenoh 的 IntoHandler 泛型链在默认 128 层内证不完 callback 闭包的 Send。
#![recursion_limit = "256"]

use pneuma::{cli, config::ServerConfig, server::Server};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    use clap::Parser;

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let cli = cli::Cli::parse();
    match cli.command {
        cli::Command::Serve(args) => {
            let mut config = ServerConfig::load(&args.config)?;
            args.apply(&mut config);
            Server::new(config)?.run().await
        }
        cli::Command::Convert(args) => args.run(),
    }
}
