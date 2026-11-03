use std::{
    collections::{BTreeMap, BTreeSet},
    path::PathBuf,
};

use anyhow::{Context as _, Result, ensure};
use serde::{Deserialize, Serialize};

use super::{Batch, Context, DType, Field, Layout, Schema};

fn default_true() -> bool {
    true
}

pub(super) const TOKENS: &str = "observation.language.tokens";
/// 负向提示 token（无分类器引导用）。
pub(super) const NEGATIVE: &str = "observation.language.negative_tokens";
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
    /// π₀.₅ 风格提示词：`Task: {task}, State: {bins};\nAction: `。
    Prompt {
        state_key: String,
        bins: usize,
    },
    Tokenize {
        tokenizer: PathBuf,
        max_length: usize,
        padding: String,
        padding_side: String,
        /// 是否自动追加特殊 token；Cosmos3 的提示必须关闭。
        #[serde(default = "default_true")]
        special_tokens: bool,
        /// 可选：额外分词该文本，写到 `observation.language.negative_tokens`。
        #[serde(default)]
        negative: Option<String>,
    },
    /// Cosmos3 的视频画布：把多个相机视角拼成一张画布。
    Cosmos3Canvas {
        cameras: Vec<String>,
        view_width: usize,
        view_height: usize,
        output: String,
    },
    /// Cosmos3 的 JSON 提示：把任务改写成模板包裹的文本。
    Cosmos3Caption {
        height: usize,
        width: usize,
        fps: f64,
        chunk_size: usize,
        use_state: bool,
        view: String,
        /// 可选：用于校验尺寸的画布字段。
        #[serde(default)]
        canvas_key: Option<String>,
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
            Self::Prompt { state_key, bins } => {
                ensure!(*bins >= 2, "prompt bin count must be at least 2");
                ensure!(!device.contains(state_key), "Prompt requires host data");
                let field = field_mut(schema, state_key)?;
                ensure!(
                    field.dtype == DType::F32
                        && field.layout == Layout::Flat
                        && field.shape.len() == 2,
                    "Prompt requires a batched flat f32 state"
                );
            }
            Self::Cosmos3Canvas {
                cameras,
                view_width,
                view_height,
                output,
            } => {
                ensure!(
                    !cameras.is_empty() && *view_width > 0 && *view_height > 0,
                    "cosmos3 canvas needs cameras and a positive view size"
                );
                for camera in cameras {
                    let field = field_mut(schema, camera)?;
                    ensure!(
                        field.dtype == DType::F32
                            && field.shape.len() == 4
                            && field.shape[1] == Some(3),
                        "cosmos3 canvas inputs must be f32 images with three channels"
                    );
                }
                let canvas = crate::policy::cosmos3::canvas::canvas_size(
                    *view_height,
                    *view_width * cameras.len(),
                );
                let (width, height) = canvas.context("unsupported cosmos3 canvas size")?;
                schema.insert(
                    output.clone(),
                    Field {
                        shape: vec![Some(1), Some(3), Some(height), Some(width)],
                        dtype: DType::F32,
                        layout: Layout::Flat,
                        optional: false,
                    },
                );
            }
            Self::Cosmos3Caption {
                height,
                width,
                fps,
                chunk_size,
                canvas_key,
                ..
            } => {
                ensure!(
                    *height > 0 && *width > 0 && *fps > 0.0 && *chunk_size > 0,
                    "cosmos3 caption needs positive canvas, fps and chunk size"
                );
                if let Some(key) = canvas_key {
                    let field = field_mut(schema, key)?;
                    ensure!(
                        field.shape.len() == 4
                            && field.shape[2] == Some(*height)
                            && field.shape[3] == Some(*width),
                        "cosmos3 caption size must match the canvas field"
                    );
                }
            }
            Self::Tokenize {
                max_length,
                padding,
                padding_side,
                negative,
                ..
            } => {
                if negative.is_some() {
                    ensure!(
                        !schema.contains_key(NEGATIVE),
                        "tokenizer output {} already exists",
                        NEGATIVE
                    );
                    schema.insert(
                        NEGATIVE.into(),
                        Field {
                            shape: vec![None, None],
                            dtype: DType::I64,
                            layout: Layout::Flat,
                            optional: false,
                        },
                    );
                    device.insert(NEGATIVE.into());
                }
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
                ..
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
