//! 推理后端边界：server 只依赖 `PolicyRunner`，模型接入时替换实现即可。
//! echo 桩返回全零 action chunk，用于打通协议链路和延迟测量；
//! act 是 Burn 原生实现的 ACT 策略（权重直载 lerobot safetensors checkpoint）。

pub mod act;

use rmpv::Value;

use crate::codec;
use crate::config::ServerConfig;
use crate::protocol::{Capabilities, ProtocolError};

/// 一次推理的产物：canonical 动作块按 execution_steps×action_dim 行主序平铺。
pub struct Chunk {
    pub canonical_actions: Vec<u8>,
    pub execution_steps: u64,
}

pub trait PolicyRunner: Send + Sync {
    /// features 是观测 body 里已按名字组织的 msgpack map；实现自行解 tensor/rgb。
    fn predict(&self, caps: &Capabilities, features: &Value) -> Result<Chunk, ProtocolError>;

    /// 装载期自检：权重固有维度与通告合约不一致属配置错误，必须启动即报错，
    /// 不能拖到首次推理在算子内部 panic。
    fn validate_against(&self, caps: &Capabilities) -> Result<(), ProtocolError>;
}

/// 全零动作桩；不读观测内容，只验证特征键已由 server 校验过。
pub struct EchoPolicy;

impl PolicyRunner for EchoPolicy {
    fn predict(&self, caps: &Capabilities, _features: &Value) -> Result<Chunk, ProtocolError> {
        let dim = caps.action_feature.shape[0] as usize;
        let zeros = vec![0f32; caps.execution_steps as usize * dim];
        Ok(Chunk {
            canonical_actions: codec::f32_le(&zeros),
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
                (std::sync::Arc::new(EchoPolicy), prediction, prediction)
            }
            Policy::Act => {
                let policy = act::ActPolicy::load_configured(
                    &cfg.model.repo_or_path,
                    cfg.model.device,
                    cfg.execution.warmup_calls,
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
