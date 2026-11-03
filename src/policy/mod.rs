//! 策略只处理模型坐标下的 batch；数据转换与执行生命周期由调用方组合。

pub mod act;
pub mod cosmos3;
pub mod pi05;
pub mod smolvla;
pub mod zero;

use std::fmt;
use std::str::FromStr;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::processor::Batch;

/// 策略类型：部署配置的 `policy` 与 checkpoint 清单的 `policy_kind` 共用。
///
/// 字符串形式同时是 LeRobot checkpoint 的 `type` 取值，因此只在一处定义。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PolicyKind {
    Zero,
    Act,
    Smolvla,
    Pi05,
    Cosmos3Edge,
}

impl PolicyKind {
    /// 全部取值，供字符串解析与命令行帮助枚举。
    pub const ALL: [PolicyKind; 5] = [
        PolicyKind::Zero,
        PolicyKind::Act,
        PolicyKind::Smolvla,
        PolicyKind::Pi05,
        PolicyKind::Cosmos3Edge,
    ];

    /// checkpoint 与配置中使用的固定拼写。
    pub const fn as_str(self) -> &'static str {
        match self {
            PolicyKind::Zero => "zero",
            PolicyKind::Act => "act",
            PolicyKind::Smolvla => "smolvla",
            PolicyKind::Pi05 => "pi05",
            PolicyKind::Cosmos3Edge => "cosmos3_edge",
        }
    }
}

impl fmt::Display for PolicyKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for PolicyKind {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        Self::ALL
            .into_iter()
            .find(|kind| kind.as_str() == value)
            .with_context(|| format!("unknown policy type {value:?}"))
    }
}

/// 模型预测完整动作块，不拥有协议、动作播放队列或执行片段配置。
pub trait Policy: Send + Sync {
    fn predict(&self, batch: Batch) -> Result<Batch>;
}

#[cfg(test)]
mod tests {
    use super::PolicyKind;

    /// `as_str` 与 serde 派生必须给出同一套拼写；改名时这里会先失败。
    #[test]
    fn policy_kind_spelling_matches_serde() {
        for kind in PolicyKind::ALL {
            let text = kind.as_str();
            let encoded = serde_json::to_string(&kind).expect("serializing policy kind");
            assert_eq!(
                encoded,
                format!("{text:?}"),
                "serde spelling differs for {text}"
            );
            let decoded: PolicyKind =
                serde_json::from_str(&encoded).expect("deserializing policy kind");
            assert_eq!(decoded, kind);
            assert_eq!(text.parse::<PolicyKind>().expect("parsing"), kind);
        }
    }
}
