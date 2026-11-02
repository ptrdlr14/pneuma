use std::collections::BTreeMap;

use anyhow::{Context, Result, bail, ensure};
use burn::tensor::{Bool, Int, Tensor};
use serde::{Deserialize, Serialize};

/// 主机数组与设备 tensor 保持不同值域，转换必须由步骤显式完成。
#[derive(Clone, Debug)]
pub enum Data {
    F32 { values: Vec<f32>, shape: Vec<usize> },
    U8 { values: Vec<u8>, shape: Vec<usize> },
    Float2(Tensor<2>),
    Float3(Tensor<3>),
    Float4(Tensor<4>),
    Int2(Tensor<2, Int>),
    Bool2(Tensor<2, Bool>),
}

impl Data {
    pub fn shape(&self) -> Vec<usize> {
        match self {
            Self::F32 { shape, .. } | Self::U8 { shape, .. } => shape.clone(),
            Self::Float2(t) => t.dims().to_vec(),
            Self::Float3(t) => t.dims().to_vec(),
            Self::Float4(t) => t.dims().to_vec(),
            Self::Int2(t) => t.dims().to_vec(),
            Self::Bool2(t) => t.dims().to_vec(),
        }
    }

    pub fn into_float2(self) -> Result<Tensor<2>> {
        match self {
            Self::Float2(t) => Ok(t),
            _ => bail!("expected rank-2 float tensor"),
        }
    }

    pub fn into_float3(self) -> Result<Tensor<3>> {
        match self {
            Self::Float3(t) => Ok(t),
            _ => bail!("expected rank-3 float tensor"),
        }
    }

    pub fn into_float4(self) -> Result<Tensor<4>> {
        match self {
            Self::Float4(t) => Ok(t),
            _ => bail!("expected rank-4 float tensor"),
        }
    }

    fn dtype(&self) -> DType {
        match self {
            Self::F32 { .. } | Self::Float2(_) | Self::Float3(_) | Self::Float4(_) => DType::F32,
            Self::U8 { .. } => DType::U8,
            Self::Int2(_) => DType::I64,
            Self::Bool2(_) => DType::Bool,
        }
    }

    pub(super) fn validate(&self, field: &Field, numeric: bool) -> Result<()> {
        let shape = self.shape();
        ensure!(self.dtype() == field.dtype, "dtype differs from schema");
        ensure!(shape.len() == field.shape.len(), "rank differs from schema");
        for (actual, expected) in shape.iter().zip(&field.shape) {
            ensure!(*actual > 0, "empty dimensions are unsupported");
            ensure!(
                expected.is_none_or(|n| n == *actual),
                "shape {shape:?} differs from {:?}",
                field.shape
            );
        }
        let count = element_count(&shape)?;
        match self {
            Self::F32 { values, .. } => {
                ensure!(values.len() == count, "array length differs from shape");
                ensure!(
                    !numeric || values.iter().all(|x| x.is_finite()),
                    "non-finite numeric value"
                );
            }
            Self::U8 { values, .. } => {
                ensure!(values.len() == count, "array length differs from shape")
            }
            Self::Float2(t) if numeric => ensure_finite(t)?,
            Self::Float3(t) if numeric => ensure_finite(t)?,
            Self::Float4(t) if numeric => ensure_finite(t)?,
            _ => {}
        }
        Ok(())
    }
}

fn ensure_finite<const D: usize>(tensor: &Tensor<D>) -> Result<()> {
    // 只读回校验标量；原 tensor 的设备与计算图不变。
    ensure!(
        tensor.clone().is_finite().all().into_scalar::<bool>(),
        "non-finite numeric tensor"
    );
    Ok(())
}

pub(super) fn element_count(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1usize, |n, d| {
        ensure!(*d > 0, "empty dimensions are unsupported");
        n.checked_mul(*d).context("shape element count overflow")
    })
}

/// 任务文本的数量由输入适配器与显式 batch 轴对齐。
#[derive(Default, Clone, Debug)]
pub struct Batch {
    pub values: BTreeMap<String, Data>,
    pub tasks: Vec<String>,
}

impl Batch {
    pub fn get(&self, name: &str) -> Result<&Data> {
        self.values
            .get(name)
            .with_context(|| format!("missing field {name}"))
    }

    pub fn take(&mut self, name: &str) -> Result<Data> {
        self.values
            .remove(name)
            .with_context(|| format!("missing field {name}"))
    }
}

pub type Schema = BTreeMap<String, Field>;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Field {
    pub shape: Vec<Option<usize>>,
    pub dtype: DType,
    pub layout: Layout,
    pub optional: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DType {
    U8,
    F32,
    I64,
    Bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Layout {
    Flat,
    Hwc,
    Chw,
}

pub(super) fn validate_schema(schema: &Schema) -> Result<()> {
    for (key, field) in schema {
        ensure!(!key.is_empty(), "field name must not be empty");
        ensure!(
            !field.shape.is_empty(),
            "scalar fields are unsupported: {key}"
        );
        ensure!(
            field.shape.iter().all(|n| n.is_none_or(|n| n > 0)),
            "empty schema dimension: {key}"
        );
        let known: Vec<_> = field.shape.iter().map(|n| n.unwrap_or(1)).collect();
        element_count(&known)?;
        if field.layout != Layout::Flat {
            ensure!(
                matches!(field.shape.len(), 3 | 4),
                "image layout requires rank 3 or 4: {key}"
            );
        }
    }
    Ok(())
}

pub(super) fn validate_batch(batch: &Batch, schema: &Schema, numeric: bool) -> Result<()> {
    for (key, field) in schema {
        match batch.values.get(key) {
            Some(data) => data
                .validate(field, numeric)
                .with_context(|| format!("field {key}"))?,
            None => ensure!(field.optional, "missing required field {key}"),
        }
    }
    for key in batch.values.keys() {
        ensure!(
            schema.contains_key(key),
            "field {key} is not declared in schema"
        );
    }
    Ok(())
}
