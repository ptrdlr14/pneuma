//! 策略只处理模型坐标下的 batch；数据转换与执行生命周期由调用方组合。

pub mod act;
pub mod smolvla;

use anyhow::Result;
use burn::prelude::{Device, Tensor};

use crate::processor::{Batch, Data};

/// 模型预测完整动作块，不拥有协议、动作播放队列或执行片段配置。
pub trait Policy: Send + Sync {
    fn predict(&self, batch: Batch) -> Result<Batch>;
}

/// 无权重的全零策略，用于协议与运行时验证。
#[derive(Debug)]
pub struct ZeroActionPolicy {
    pub steps: usize,
    pub dim: usize,
    pub device: Device,
}

impl Policy for ZeroActionPolicy {
    fn predict(&self, mut batch: Batch) -> Result<Batch> {
        batch.values.clear();
        batch.values.insert(
            "action".into(),
            Data::Float3(Tensor::zeros([1, self.steps, self.dim], &self.device)),
        );
        Ok(batch)
    }
}
