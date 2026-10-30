//! 推理后端边界：server 只依赖 `PolicyRunner`，模型接入时替换实现即可。
//! echo 桩返回全零 action chunk，用于打通协议链路和延迟测量；
//! act 与 smolvla 用 Burn 原生执行，直接加载 LeRobot safetensors checkpoint。

pub mod act;
pub mod smolvla;

use rmpv::Value;

use crate::config::ServerConfig;
use crate::protocol::{Capabilities, ProtocolError};

/// 一次推理的产物：canonical 动作块按 `execution_steps × action_dim` 行主序平铺。
pub struct Chunk {
    pub canonical_actions: Vec<u8>,
    pub execution_steps: u64,
}

pub trait PolicyRunner: Send + Sync {
    /// features 是已验证的观测特征 map；task 来自同次观测，用于语言条件策略。
    fn predict(
        &self,
        caps: &Capabilities,
        features: &Value,
        task: Option<&str>,
    ) -> Result<Chunk, ProtocolError>;

    /// 装载期自检：权重固有维度与通告合约不一致属配置错误，必须启动即报错，
    /// 不能拖到首次推理在算子内部 panic。
    fn validate_against(&self, caps: &Capabilities) -> Result<(), ProtocolError>;
}

/// 全零动作桩；不读观测内容，只验证特征键已由 server 校验过。
pub struct ZeroActionPolicy;

impl PolicyRunner for ZeroActionPolicy {
    fn predict(
        &self,
        caps: &Capabilities,
        _features: &Value,
        _task: Option<&str>,
    ) -> Result<Chunk, ProtocolError> {
        let invalid_size = || {
            ProtocolError::new(
                crate::protocol::ErrorCode::Execution,
                "Action size exceeds capacity",
            )
        };
        let bytes = caps
            .execution_steps
            .checked_mul(caps.action_feature.shape[0])
            .and_then(|elements| elements.checked_mul(4))
            .and_then(|bytes| usize::try_from(bytes).ok())
            .ok_or_else(invalid_size)?;
        let mut canonical_actions = Vec::new();
        canonical_actions
            .try_reserve_exact(bytes)
            .map_err(|_| invalid_size())?;
        canonical_actions.resize(bytes, 0);
        Ok(Chunk {
            canonical_actions,
            execution_steps: caps.execution_steps,
        })
    }

    // echo 无权重，维度完全由 caps 派生，没有可自检的固有约束。
    fn validate_against(&self, _caps: &Capabilities) -> Result<(), ProtocolError> {
        Ok(())
    }
}

/// 构造策略并从模型元数据补齐通告步数；配置覆盖只允许缩短执行片段。
pub fn build(
    cfg: &ServerConfig,
) -> anyhow::Result<(std::sync::Arc<dyn PolicyRunner>, Capabilities)> {
    use crate::config::Policy;
    let (policy, prediction_steps, default_steps): (std::sync::Arc<dyn PolicyRunner>, u64, u64) =
        match cfg.model.policy {
            Policy::Echo => {
                let prediction = cfg
                    .execution
                    .prediction_steps
                    .ok_or_else(|| anyhow::anyhow!("echo requires execution.prediction_steps"))?;
                (
                    std::sync::Arc::new(ZeroActionPolicy),
                    prediction,
                    prediction,
                )
            }
            Policy::Act => {
                act::validate_contract(&cfg.features, &cfg.action_feature)?;
                let policy = act::ActPolicy::load_configured(
                    &cfg.model.repo_or_path,
                    cfg.model.device,
                    cfg.execution.warmup_calls,
                )?;
                let (prediction, execution) = policy.horizon();
                (std::sync::Arc::new(policy), prediction, execution)
            }
            Policy::Smolvla => {
                let policy = smolvla::SmolVlaPolicy::load_configured(
                    &cfg.model.repo_or_path,
                    cfg.model.vlm_path.as_deref(),
                    cfg.model.device,
                    cfg.execution.warmup_calls,
                    &cfg.features,
                    &cfg.action_feature,
                )?;
                let (prediction, execution) = policy.horizon();
                (std::sync::Arc::new(policy), prediction, execution)
            }
        };
    let execution_steps = cfg.execution.n_action_steps.unwrap_or(default_steps);
    anyhow::ensure!(
        execution_steps > 0 && execution_steps <= prediction_steps,
        "execution.n_action_steps must be in 1..={prediction_steps}"
    );
    let caps = cfg.capabilities(prediction_steps, execution_steps);
    policy.validate_against(&caps)?;
    Ok((policy, caps))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn echo_rejects_overflowing_action_size() {
        let cfg = ServerConfig::load(std::path::Path::new("examples/so101.toml")).unwrap();
        let (_, mut caps) = build(&cfg).unwrap();
        caps.execution_steps = u64::MAX;
        let error = ZeroActionPolicy
            .predict(&caps, &crate::protocol::map([]), None)
            .err()
            .expect("overflow must return an execution error");
        assert_eq!(error.code, crate::protocol::ErrorCode::Execution);
    }

    #[test]
    fn echo_produces_advertised_zero_actions() {
        let cfg = ServerConfig::load(std::path::Path::new("examples/so101.toml")).unwrap();
        let (policy, caps) = build(&cfg).unwrap();
        let chunk = policy
            .predict(&caps, &crate::protocol::map([]), None)
            .unwrap();
        assert_eq!(chunk.execution_steps, caps.execution_steps);
        assert_eq!(
            chunk.canonical_actions.len() as u64,
            caps.execution_steps * caps.action_feature.shape[0] * 4
        );
        assert!(chunk.canonical_actions.iter().all(|v| *v == 0));
    }
}
