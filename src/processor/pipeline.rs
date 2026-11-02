use anyhow::{Context as _, Result, ensure};
use burn::tensor::Device;

use super::{Batch, Schema, Step, data};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Train,
    Eval,
}

#[derive(Clone, Debug)]
pub struct Context {
    pub device: Device,
    pub mode: Mode,
}

/// 构造时推导 schema，执行时校验实际动态维度及步骤支持的值域。
#[derive(Debug)]
pub struct Pipeline {
    pub(super) input: Schema,
    pub(super) schemas: Vec<Schema>,
    pub(super) steps: Vec<Step>,
    pub(super) tokenizers: Vec<Option<super::tokenize::Prepared>>,
}

impl Pipeline {
    pub fn new(schema: Schema, steps: Vec<Step>) -> Result<Self> {
        Self::from_prepared(schema, steps, None)
    }

    pub(super) fn from_prepared(
        schema: Schema,
        steps: Vec<Step>,
        prepared: Option<Vec<Option<super::tokenize::Prepared>>>,
    ) -> Result<Self> {
        data::validate_schema(&schema)?;
        let mut prepared = prepared.map(Vec::into_iter);
        let mut current = schema.clone();
        let mut schemas = Vec::with_capacity(steps.len());
        let mut tokenizers = Vec::with_capacity(steps.len());
        let mut device_fields = std::collections::BTreeSet::new();
        for (index, step) in steps.iter().enumerate() {
            step.infer(&mut current, &mut device_fields)
                .with_context(|| format!("step {index}: {step:?}"))?;
            data::validate_schema(&current)?;
            tokenizers.push(match &mut prepared {
                Some(prepared) => prepared.next().context("missing prepared step resources")?,
                None => step.load_tokenizer()?,
            });
            schemas.push(current.clone());
        }
        Ok(Self {
            input: schema,
            schemas,
            steps,
            tokenizers,
        })
    }

    pub fn input_schema(&self) -> &Schema {
        &self.input
    }

    pub fn output_schema(&self) -> &Schema {
        self.schemas.last().unwrap_or(&self.input)
    }

    pub fn run(&self, mut batch: Batch, context: &Context) -> Result<Batch> {
        // Host 在执行前拒绝训练模式，避免任何部分执行导致的梯度截断。
        ensure!(
            context.mode == Mode::Eval || !self.steps.iter().any(|s| matches!(s, Step::Host)),
            "Host step is only supported in eval mode"
        );
        data::validate_batch(&batch, &self.input, true).context("pipeline input")?;
        for (index, step) in self.steps.iter().enumerate() {
            step.run(
                &mut batch,
                context,
                self.tokenizers[index].as_ref().map(|p| &p.tokenizer),
            )
            .with_context(|| format!("executing step {index}: {step:?}"))?;
            data::validate_batch(&batch, &self.schemas[index], false)
                .with_context(|| format!("output of step {index}"))?;
        }
        if !self.steps.is_empty() {
            data::validate_batch(&batch, self.output_schema(), true).context("pipeline output")?;
        }
        Ok(batch)
    }
}
