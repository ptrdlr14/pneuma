use anyhow::Result;
use burn::prelude::{Device, Tensor};

use crate::processor::{Batch, Data};

/// 无权重的全零策略，用于协议与运行时验证。
#[derive(Debug)]
pub struct ZeroActionPolicy {
    pub steps: usize,
    pub dim: usize,
    pub device: Device,
}

impl super::Policy for ZeroActionPolicy {
    fn predict(&self, mut batch: Batch) -> Result<Batch> {
        batch.values.clear();
        batch.values.insert(
            "action".into(),
            Data::Float3(Tensor::zeros([1, self.steps, self.dim], &self.device)),
        );
        Ok(batch)
    }
}
