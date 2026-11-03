use std::collections::BTreeMap;

use anyhow::{Context, Result, bail, ensure};
use burn::tensor::{Bool, DType as BurnDType, Int, Tensor, TensorData};
use serde::{Deserialize, Serialize};

/// 主机数组与设备 tensor 保持不同值域，转换必须由步骤显式完成。
///
/// 宿主侧统一用 `TensorData`：它自带 dtype 与形状，`f32`/`u8`/`i64`/`bool` 共用一种表示。
/// 构造与取值都是零拷贝。
#[derive(Clone, Debug)]
pub enum Data {
    /// 宿主数据：仍在内存中，未绑定设备。
    Host(TensorData),
    Float2(Tensor<2>),
    Float3(Tensor<3>),
    Float4(Tensor<4>),
    Int2(Tensor<2, Int>),
    Bool2(Tensor<2, Bool>),
}

impl Data {
    pub fn f32(values: Vec<f32>, shape: Vec<usize>) -> Self {
        Self::Host(TensorData::new(values, shape))
    }

    pub fn u8(values: Vec<u8>, shape: Vec<usize>) -> Self {
        Self::Host(TensorData::new(values, shape))
    }

    /// 外部输入使用的构造：长度与形状不符时返回错误，不 panic。
    pub fn try_f32(values: Vec<f32>, shape: Vec<usize>) -> Result<Self> {
        ensure!(
            element_count(&shape)? == values.len(),
            "array length differs from shape"
        );
        Ok(Self::f32(values, shape))
    }

    pub fn try_u8(values: Vec<u8>, shape: Vec<usize>) -> Result<Self> {
        ensure!(
            element_count(&shape)? == values.len(),
            "array length differs from shape"
        );
        Ok(Self::u8(values, shape))
    }

    pub fn host(&self) -> Result<&TensorData> {
        match self {
            Self::Host(data) => Ok(data),
            other => bail!("expected host data, got {:?}", other.shape()),
        }
    }

    pub fn host_mut(&mut self) -> Result<&mut TensorData> {
        match self {
            Self::Host(data) => Ok(data),
            other => bail!("expected host data, got {:?}", other.shape()),
        }
    }

    /// 宿主 f32 视图；元素类型不匹配时报错。
    pub fn host_f32(&self) -> Result<&[f32]> {
        self.host()?
            .as_slice::<f32>()
            .context("host data is not f32")
    }

    pub fn host_f32_mut(&mut self) -> Result<&mut [f32]> {
        self.host_mut()?
            .as_mut_slice::<f32>()
            .context("host data is not f32")
    }

    pub fn host_u8(&self) -> Result<&[u8]> {
        self.host()?.as_slice::<u8>().context("host data is not u8")
    }

    /// 在最前面插入 batch 轴；字节不动，只换形状。
    pub(super) fn insert_batch_dim(&mut self) -> Result<()> {
        let Self::Host(data) = self else {
            bail!("Batch requires host arrays")
        };
        let (bytes, shape, dtype) = data.clone().into_parts();
        let mut shape = shape.to_vec();
        shape.insert(0, 1);
        *data = TensorData::from_bytes(bytes, shape, dtype);
        Ok(())
    }

    pub fn shape(&self) -> Vec<usize> {
        match self {
            Self::Host(data) => data.shape().to_vec(),
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

    /// 宿主 dtype 映射到 schema 的 dtype；未知类型留给 `validate` 报错。
    fn dtype(&self) -> Option<DType> {
        match self {
            Self::Host(data) => match data.dtype() {
                BurnDType::U8 => Some(DType::U8),
                BurnDType::F32 => Some(DType::F32),
                BurnDType::I64 => Some(DType::I64),
                BurnDType::Bool(_) => Some(DType::Bool),
                _ => None,
            },
            Self::Float2(_) | Self::Float3(_) | Self::Float4(_) => Some(DType::F32),
            Self::Int2(_) => Some(DType::I64),
            Self::Bool2(_) => Some(DType::Bool),
        }
    }

    pub(super) fn validate(&self, field: &Field, numeric: bool) -> Result<()> {
        let shape = self.shape();
        let dtype = self
            .dtype()
            .with_context(|| format!("unsupported host dtype for {}", field.shape.len()))?;
        ensure!(dtype == field.dtype, "dtype differs from schema");
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
            Self::Host(data) => {
                ensure!(
                    data.num_elements() == count,
                    "array length differs from shape"
                );
                if numeric && data.dtype() == BurnDType::F32 {
                    ensure!(
                        self.host_f32()?.iter().all(|x| x.is_finite()),
                        "non-finite numeric value"
                    );
                }
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
