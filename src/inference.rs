//! 同步推理组合：通用输入/输出 pipeline、模型与执行片段，不拥有会话调度。

use anyhow::{Result, ensure};
use burn::prelude::Device;

use crate::{
    policy::Policy,
    processor::{Batch, Context, Mode, Pipeline},
};

/// 主机端 canonical 动作，元素按时间、动作维度连续排列。
#[derive(Debug)]
pub struct ActionChunk {
    pub values: Vec<f32>,
    pub steps: usize,
    pub dim: usize,
}

/// 已在部署阶段绑定模型、processor 与执行长度的推理入口。
pub struct PolicyRunner {
    policy: Box<dyn Policy>,
    pub preprocessor: Pipeline,
    pub postprocessor: Pipeline,
    context: Context,
    prediction_steps: usize,
    execution_steps: usize,
    action_dim: usize,
}

impl PolicyRunner {
    pub fn new(
        policy: Box<dyn Policy>,
        pipelines: (Pipeline, Pipeline),
        device: Device,
        prediction_steps: usize,
        execution_steps: usize,
        action_dim: usize,
    ) -> Result<Self> {
        ensure!(
            execution_steps > 0 && execution_steps <= prediction_steps,
            "execution steps must fit the prediction horizon"
        );
        ensure!(action_dim > 0, "action dimension must be positive");
        prediction_steps
            .checked_mul(action_dim)
            .ok_or_else(|| anyhow::anyhow!("action size exceeds capacity"))?;
        let (preprocessor, postprocessor) = pipelines;
        Ok(Self {
            policy,
            preprocessor,
            postprocessor,
            context: Context {
                device,
                mode: Mode::Eval,
            },
            prediction_steps,
            execution_steps,
            action_dim,
        })
    }

    /// 完整预测验证通过后再截取，避免尾部坏值被执行片段掩盖。
    pub fn predict(&self, batch: Batch) -> Result<ActionChunk> {
        ensure!(
            batch.tasks.iter().all(|task| task.chars().count() <= 4096),
            "task exceeds max_input_chars"
        );
        let prepared = self.preprocessor.run(batch, &self.context)?;
        let predicted = self.policy.predict(prepared)?;
        let mut canonical = self.postprocessor.run(predicted, &self.context)?;
        let action = canonical.take("action")?.into_float3()?;
        ensure!(
            action.dims() == [1, self.prediction_steps, self.action_dim],
            "canonical action shape differs from deployment: {:?}",
            action.dims()
        );
        let mut values = action.into_data().try_to_vec::<f32>()?;
        ensure!(
            values.iter().all(|x| x.is_finite()),
            "model action output contains non-finite values"
        );
        values.truncate(self.execution_steps * self.action_dim);
        Ok(ActionChunk {
            values,
            steps: self.execution_steps,
            dim: self.action_dim,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::processor::{DType, Data, Field, Layout, Schema};

    struct FixedPolicy(Vec<f32>);
    impl Policy for FixedPolicy {
        fn predict(&self, mut batch: Batch) -> Result<Batch> {
            batch.values.insert(
                "action".into(),
                Data::Float3(burn::prelude::Tensor::from_data(
                    burn::prelude::TensorData::new(self.0.clone(), [1, 3, 2]),
                    &Device::flex(),
                )),
            );
            Ok(batch)
        }
    }

    fn runner(values: Vec<f32>) -> PolicyRunner {
        let schema = Schema::from([(
            "action".into(),
            Field {
                shape: vec![Some(1), Some(3), Some(2)],
                dtype: DType::F32,
                layout: Layout::Flat,
                optional: false,
            },
        )]);
        PolicyRunner::new(
            Box::new(FixedPolicy(values)),
            (
                Pipeline::new(Schema::new(), vec![]).unwrap(),
                Pipeline::new(schema, vec![]).unwrap(),
            ),
            Device::flex(),
            3,
            2,
            2,
        )
        .unwrap()
    }

    #[test]
    fn execution_slice_follows_full_canonical_validation() {
        let result = runner(vec![1., 2., 3., 4., 5., 6.])
            .predict(Batch::default())
            .unwrap();
        assert_eq!(result.values, vec![1., 2., 3., 4.]);
        assert_eq!((result.steps, result.dim), (2, 2));
        assert!(
            runner(vec![0., 0., 0., 0., f32::NAN, 0.])
                .predict(Batch::default())
                .is_err()
        );
    }
}
