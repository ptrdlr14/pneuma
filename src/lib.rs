//! Burn 策略、通用 processor pipeline 与部署服务，可供本地 Rust 调用方复用。

#![recursion_limit = "256"]

pub mod checkpoint;
pub mod cli;
pub mod codec;
pub mod config;
pub mod contract;
pub mod convert;
pub mod deployment;
pub mod inference;
pub mod policy;
pub mod processor;
pub mod protocol;
pub mod server;
