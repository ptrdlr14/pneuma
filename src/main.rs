//! pneuma: lerobot remote-inference v1 协议的 Rust 服务端骨架。
//! 部署由 --config_path 指定 TOML，嵌套 CLI 参数覆盖文件。

// zenoh 的 IntoHandler 泛型链在默认 128 层内证不完 callback 闭包的 Send。
#![recursion_limit = "256"]

mod codec;
mod config;
mod policy;
mod protocol;
mod server;

use config::ServerConfig;
use server::Server;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();
    Server::new(ServerConfig::from_args(std::env::args_os())?)?
        .run()
        .await
}
