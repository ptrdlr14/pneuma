use std::{
    collections::{BTreeMap, BTreeSet},
    path::PathBuf,
};

use anyhow::{Context as _, Result, ensure};
use serde::{Deserialize, Serialize};

use super::{Batch, Context, DType, Field, Layout, Schema};

pub(super) const TOKENS: &str = "observation.language.tokens";
pub(super) const MASK: &str = "observation.language.attention_mask";

/// 只注册已实现的确定性转换，统计由产物层单独保存。
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Step {
    Rename {
        mapping: BTreeMap<String, String>,
    },
    Batch,
    Rgb {
        key: String,
    },
    ResizePad {
        key: String,
        width: usize,
        height: usize,
    },
    Affine {
        key: String,
        offset: Vec<f32>,
        scale: Vec<f32>,
        axis: usize,
    },
    Linear {
        key: String,
        offset: Vec<f32>,
        scale: Vec<f32>,
        axis: usize,
    },
    Pad {
        key: String,
        width: usize,
    },
    Crop {
        key: String,
        width: usize,
    },
    Newline,
    Tokenize {
        tokenizer: PathBuf,
        max_length: usize,
        padding: String,
        padding_side: String,
    },
    Device,
    Host,
}

impl Step {
    pub(super) fn infer(&self, schema: &mut Schema, device: &mut BTreeSet<String>) -> Result<()> {
        match self {
            Self::Rename { mapping } => {
                let mut renamed = Schema::new();
                let mut moved = BTreeSet::new();
                for key in mapping.keys() {
                    ensure!(
                        schema.contains_key(key),
                        "rename source {key} is absent from schema"
                    );
                }
                for (key, field) in std::mem::take(schema) {
                    let target = mapping.get(&key).cloned().unwrap_or_else(|| key.clone());
                    ensure!(!target.is_empty(), "rename destination must not be empty");
                    if device.contains(&key) {
                        moved.insert(target.clone());
                    }
                    ensure!(
                        renamed.insert(target.clone(), field).is_none(),
                        "rename collision at {target}"
                    );
                }
                *schema = renamed;
                *device = moved;
            }
            Self::Batch => {
                ensure!(device.is_empty(), "Batch must precede device conversion");
                for field in schema.values_mut() {
                    ensure!(
                        matches!(field.dtype, DType::U8 | DType::F32),
                        "Batch requires host numeric arrays"
                    );
                    if field.layout != Layout::Flat {
                        ensure!(field.shape.len() == 3, "Batch expects an unbatched image");
                    }
                    field.shape.insert(0, Some(1));
                }
            }
            Self::Rgb { key } => {
                ensure!(!device.contains(key), "Rgb requires host data");
                let field = field_mut(schema, key)?;
                ensure!(
                    field.dtype == DType::U8
                        && field.layout == Layout::Hwc
                        && field.shape.len() == 4,
                    "Rgb requires batched HWC u8"
                );
                ensure!(
                    field.shape[3].is_none_or(|n| n == 3),
                    "Rgb requires 3 channels"
                );
                field.shape = vec![field.shape[0], Some(3), field.shape[1], field.shape[2]];
                field.dtype = DType::F32;
                field.layout = Layout::Chw;
            }
            Self::ResizePad { key, width, height } => {
                ensure!(
                    *width > 0 && *height > 0,
                    "resize dimensions must be positive"
                );
                ensure!(!device.contains(key), "ResizePad requires host data");
                let field = field_mut(schema, key)?;
                ensure!(
                    field.dtype == DType::F32
                        && field.layout == Layout::Chw
                        && field.shape.len() == 4,
                    "ResizePad requires batched CHW f32"
                );
                if let (Some(h), Some(w)) = (field.shape[2], field.shape[3]) {
                    super::transform::resized_shape(h, w, *height, *width)?;
                }
                field.shape[2] = Some(*height);
                field.shape[3] = Some(*width);
            }
            Self::Affine {
                key,
                offset,
                scale,
                axis,
            }
            | Self::Linear {
                key,
                offset,
                scale,
                axis,
            } => {
                let field = field_mut(schema, key)?;
                ensure!(
                    field.dtype == DType::F32 && *axis < field.shape.len(),
                    "affine requires f32 and an existing axis"
                );
                ensure!(
                    !offset.is_empty() && !scale.is_empty(),
                    "affine vectors must not be empty"
                );
                ensure!(
                    offset.iter().chain(scale).all(|v| v.is_finite()),
                    "non-finite affine statistics"
                );
                if matches!(self, Self::Affine { .. }) {
                    ensure!(
                        scale.iter().all(|v| *v != 0.0),
                        "affine scale must be nonzero"
                    );
                }
                if device.contains(key) {
                    ensure!(
                        (2..=4).contains(&field.shape.len()),
                        "tensor affine requires rank 2, 3 or 4"
                    );
                }
                for values in [offset, scale] {
                    ensure!(
                        values.len() == 1 || field.shape[*axis].is_none_or(|n| n == values.len()),
                        "affine vector must be scalar or match axis width"
                    );
                }
                ensure!(
                    offset.len() == 1 || scale.len() == 1 || offset.len() == scale.len(),
                    "affine vectors disagree on dynamic axis width"
                );
            }
            Self::Pad { key, width } | Self::Crop { key, width } => {
                ensure!(*width > 0, "feature width must be positive");
                let field = field_mut(schema, key)?;
                ensure!(
                    field.dtype == DType::F32 && field.layout == Layout::Flat,
                    "pad/crop requires flat f32 features"
                );
                let rank = field.shape.len();
                let last = field
                    .shape
                    .last_mut()
                    .context("missing feature dimension")?;
                if matches!(self, Self::Pad { .. }) {
                    ensure!(!device.contains(key), "Pad requires host data");
                    ensure!(
                        last.is_none_or(|n| n <= *width),
                        "Pad width would truncate data"
                    );
                } else {
                    ensure!(
                        !device.contains(key) || rank == 3,
                        "tensor Crop requires rank 3"
                    );
                    ensure!(
                        last.is_none_or(|n| n >= *width),
                        "Crop width exceeds source"
                    );
                }
                *field
                    .shape
                    .last_mut()
                    .context("missing feature dimension")? = Some(*width);
            }
            Self::Newline => {}
            Self::Tokenize {
                max_length,
                padding,
                padding_side,
                ..
            } => {
                ensure!(
                    matches!(padding_side.as_str(), "left" | "right"),
                    "unsupported tokenizer padding_side"
                );
                ensure!(*max_length > 0, "tokenizer max_length must be positive");
                ensure!(
                    matches!(padding.as_str(), "max_length" | "longest"),
                    "unsupported tokenizer padding"
                );
                let mut batch = None;
                for field in schema.values().filter(|f| !f.optional) {
                    let current = field.shape[0];
                    if let (Some(a), Some(b)) = (batch, current) {
                        ensure!(a == b, "tokenizer input batch dimensions disagree");
                    }
                    batch = batch.or(current);
                }
                let length = if padding == "max_length" {
                    Some(*max_length)
                } else {
                    None
                };
                for (key, dtype) in [(TOKENS, DType::I64), (MASK, DType::Bool)] {
                    ensure!(
                        !schema.contains_key(key),
                        "tokenizer output {key} already exists"
                    );
                    schema.insert(
                        key.into(),
                        Field {
                            shape: vec![batch, length],
                            dtype,
                            layout: Layout::Flat,
                            optional: false,
                        },
                    );
                    device.insert(key.into());
                }
            }
            Self::Device => {
                for (key, field) in schema.iter() {
                    ensure!(
                        field.dtype != DType::U8,
                        "Device requires RGB conversion before upload: {key}"
                    );
                    if field.dtype == DType::F32 {
                        ensure!(
                            (2..=4).contains(&field.shape.len()),
                            "Device requires float rank 2, 3 or 4"
                        );
                    }
                    device.insert(key.clone());
                }
            }
            Self::Host => {
                device.retain(|key| schema[key].dtype != DType::F32);
            }
        }
        Ok(())
    }

    pub(super) fn load_tokenizer(&self) -> Result<Option<super::tokenize::Prepared>> {
        match self {
            Self::Tokenize {
                tokenizer,
                max_length,
                padding,
                padding_side,
            } => super::tokenize::load(tokenizer, *max_length, padding, padding_side).map(Some),
            _ => Ok(None),
        }
    }

    pub(super) fn run(
        &self,
        batch: &mut Batch,
        context: &Context,
        tokenizer: Option<&tokenizers::Tokenizer>,
    ) -> Result<()> {
        super::transform::run(self, batch, context, tokenizer)
    }
}

fn field_mut<'a>(schema: &'a mut Schema, key: &str) -> Result<&'a mut Field> {
    schema
        .get_mut(key)
        .with_context(|| format!("step field {key} is absent from schema"))
}
