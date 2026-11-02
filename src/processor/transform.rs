use anyhow::{Context as _, Result, bail, ensure};
use burn::tensor::{Tensor, TensorData};

use super::{Batch, Context, Data, Step, data::element_count};

pub(super) fn run(
    step: &Step,
    batch: &mut Batch,
    context: &Context,
    tokenizer: Option<&tokenizers::Tokenizer>,
) -> Result<()> {
    match step {
        Step::Rename { mapping } => {
            batch.values = std::mem::take(&mut batch.values)
                .into_iter()
                .map(|(key, value)| (mapping.get(&key).cloned().unwrap_or(key), value))
                .collect();
        }
        Step::Batch => {
            for value in batch.values.values_mut() {
                match value {
                    Data::F32 { shape, .. } | Data::U8 { shape, .. } => shape.insert(0, 1),
                    _ => bail!("Batch requires host arrays"),
                }
            }
        }
        Step::Newline => {
            for task in &mut batch.tasks {
                if !task.ends_with('\n') {
                    task.push('\n');
                }
            }
        }
        Step::Tokenize { .. } => super::tokenize::run(
            batch,
            context,
            tokenizer.context("missing prepared tokenizer")?,
        )?,
        Step::Device => {
            for value in batch.values.values_mut() {
                *value = device(value.clone(), context)?;
            }
        }
        Step::Host => {
            for value in batch.values.values_mut() {
                *value = host(value.clone())?;
            }
        }
        Step::Rgb { key }
        | Step::ResizePad { key, .. }
        | Step::Affine { key, .. }
        | Step::Linear { key, .. }
        | Step::Pad { key, .. }
        | Step::Crop { key, .. } => {
            // 缺失字段在每步 schema 校验中只允许 optional；不静默忽略必需字段。
            if let Some(value) = batch.values.remove(key) {
                let value = match step {
                    Step::Rgb { .. } => rgb(value)?,
                    Step::ResizePad { width, height, .. } => resize_pad(value, *height, *width)?,
                    Step::Affine {
                        offset,
                        scale,
                        axis,
                        ..
                    } => affine(value, offset, scale, *axis, false)?,
                    Step::Linear {
                        offset,
                        scale,
                        axis,
                        ..
                    } => affine(value, offset, scale, *axis, true)?,
                    Step::Pad { width, .. } => pad_crop(value, *width, false)?,
                    Step::Crop { width, .. } => pad_crop(value, *width, true)?,
                    _ => unreachable!("keyed steps matched above"),
                };
                batch.values.insert(key.clone(), value);
            }
        }
    }
    Ok(())
}

fn device(value: Data, context: &Context) -> Result<Data> {
    Ok(match value {
        Data::F32 { values, shape } => match shape.len() {
            2 => Data::Float2(Tensor::from_data(
                TensorData::new(values, shape),
                &context.device,
            )),
            3 => Data::Float3(Tensor::from_data(
                TensorData::new(values, shape),
                &context.device,
            )),
            4 => Data::Float4(Tensor::from_data(
                TensorData::new(values, shape),
                &context.device,
            )),
            _ => bail!("Device requires float rank 2, 3 or 4"),
        },
        Data::Float2(t) => Data::Float2(t.to_device(&context.device)),
        Data::Float3(t) => Data::Float3(t.to_device(&context.device)),
        Data::Float4(t) => Data::Float4(t.to_device(&context.device)),
        Data::Int2(t) => Data::Int2(t.to_device(&context.device)),
        Data::Bool2(t) => Data::Bool2(t.to_device(&context.device)),
        Data::U8 { .. } => bail!("Device requires RGB conversion before upload"),
    })
}

fn host(value: Data) -> Result<Data> {
    fn float<const D: usize>(t: Tensor<D>) -> Result<Data> {
        ensure!(
            !t.is_require_grad() && !t.is_autodiff(),
            "Host rejects autodiff tensors; explicitly end autodiff before exporting"
        );
        let shape = t.dims().to_vec();
        let values = t
            .to_data()
            .try_to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("reading tensor: {e}"))?;
        Ok(Data::F32 { values, shape })
    }
    match value {
        Data::Float2(t) => float(t),
        Data::Float3(t) => float(t),
        Data::Float4(t) => float(t),
        other => Ok(other),
    }
}

fn rgb(value: Data) -> Result<Data> {
    let Data::U8 { values, shape } = value else {
        bail!("Rgb requires host u8")
    };
    let &[batch, height, width, channels] = shape.as_slice() else {
        bail!("Rgb requires [B,H,W,3]")
    };
    ensure!(channels == 3, "Rgb requires exactly three channels");
    let mut out = vec![0.0; values.len()];
    for b in 0..batch {
        for y in 0..height {
            for x in 0..width {
                for c in 0..3 {
                    out[((b * 3 + c) * height + y) * width + x] =
                        f32::from(values[((b * height + y) * width + x) * 3 + c]) / 255.0;
                }
            }
        }
    }
    Ok(Data::F32 {
        values: out,
        shape: vec![batch, 3, height, width],
    })
}

pub(super) fn resized_shape(
    h: usize,
    w: usize,
    height: usize,
    width: usize,
) -> Result<(usize, usize)> {
    ensure!(
        h > 0 && w > 0 && height > 0 && width > 0,
        "resize dimensions must be positive"
    );
    let ratio = (w as f64 / width as f64).max(h as f64 / height as f64);
    let rh = (h as f64 / ratio) as usize;
    let rw = (w as f64 / ratio) as usize;
    ensure!(
        rh > 0 && rw > 0 && rh <= height && rw <= width,
        "aspect ratio produces invalid resized dimensions"
    );
    Ok((rh, rw))
}

fn resize_pad(value: Data, height: usize, width: usize) -> Result<Data> {
    let Data::F32 { values, shape } = value else {
        bail!("ResizePad requires host f32")
    };
    let &[batch, channels, h, w] = shape.as_slice() else {
        bail!("ResizePad requires [B,C,H,W]")
    };
    let (rh, rw) = resized_shape(h, w, height, width)?;
    let pad_h = height - rh;
    let pad_w = width - rw;
    // 与 SmolVLA / torch interpolate(align_corners=false) 的 f32 坐标一致。
    let coordinates = |size: usize, resized: usize| -> Vec<(usize, usize, f32)> {
        let scale = (size as f64 / resized as f64) as f32;
        (0..resized)
            .map(|i| {
                let position = (scale * (i as f32 + 0.5) - 0.5).max(0.0);
                let first = position as usize;
                (first, (first + 1).min(size - 1), position - first as f32)
            })
            .collect()
    };
    let ys = coordinates(h, rh);
    let xs = coordinates(w, rw);
    let shape = vec![batch, channels, height, width];
    let mut out = vec![0.0; element_count(&shape)?];
    for b in 0..batch {
        for c in 0..channels {
            for (y, &(y0, y1, wy)) in ys.iter().enumerate() {
                for (x, &(x0, x1, wx)) in xs.iter().enumerate() {
                    let pixel = |yy, xx| values[((b * channels + c) * h + yy) * w + xx];
                    let top = (1.0 - wx) * pixel(y0, x0) + wx * pixel(y0, x1);
                    let bottom = (1.0 - wx) * pixel(y1, x0) + wx * pixel(y1, x1);
                    out[((b * channels + c) * height + y + pad_h) * width + x + pad_w] =
                        (1.0 - wy) * top + wy * bottom;
                }
            }
        }
    }
    Ok(Data::F32 { values: out, shape })
}

fn affine(value: Data, offset: &[f32], scale: &[f32], axis: usize, linear: bool) -> Result<Data> {
    let shape = value.shape();
    let width = *shape.get(axis).context("affine axis out of bounds")?;
    ensure!(
        [offset.len(), scale.len()]
            .iter()
            .all(|n| *n == 1 || *n == width),
        "affine vector differs from actual axis width"
    );
    Ok(match value {
        Data::F32 { mut values, shape } => {
            let stride = element_count(&shape[axis + 1..])?;
            for (i, value) in values.iter_mut().enumerate() {
                let index = (i / stride) % width;
                let offset = offset[if offset.len() == 1 { 0 } else { index }];
                let scale = scale[if scale.len() == 1 { 0 } else { index }];
                *value = if linear {
                    *value * scale + offset
                } else {
                    (*value - offset) / scale
                };
            }
            Data::F32 { values, shape }
        }
        Data::Float2(t) => Data::Float2(tensor_affine(t, offset, scale, axis, linear)),
        Data::Float3(t) => Data::Float3(tensor_affine(t, offset, scale, axis, linear)),
        Data::Float4(t) => Data::Float4(tensor_affine(t, offset, scale, axis, linear)),
        _ => bail!("affine requires host f32 or float tensor"),
    })
}

fn tensor_affine<const D: usize>(
    tensor: Tensor<D>,
    offset: &[f32],
    scale: &[f32],
    axis: usize,
    linear: bool,
) -> Tensor<D> {
    let device = tensor.device();
    let broadcast = |values: &[f32]| {
        let mut shape = [1; D];
        shape[axis] = values.len();
        Tensor::<D>::from_data(TensorData::new(values.to_vec(), shape), &device)
    };
    let offset = broadcast(offset);
    let scale = broadcast(scale);
    if linear {
        tensor * scale + offset
    } else {
        (tensor - offset) / scale
    }
}

fn pad_crop(value: Data, width: usize, crop: bool) -> Result<Data> {
    let mut shape = value.shape();
    let last = shape.len() - 1;
    let old = shape[last];
    ensure!(
        if crop { width <= old } else { width >= old },
        "invalid pad/crop width"
    );
    match value {
        Data::F32 { values, .. } => {
            shape[last] = width;
            let mut out = vec![0.0; element_count(&shape)?];
            for (input, output) in values.chunks_exact(old).zip(out.chunks_exact_mut(width)) {
                let count = old.min(width);
                output[..count].copy_from_slice(&input[..count]);
            }
            Ok(Data::F32 { values: out, shape })
        }
        Data::Float3(t) if crop => Ok(Data::Float3(t.slice_dim(2, 0..width))),
        _ => bail!("Pad requires host f32; Crop accepts host f32 or rank-3 float tensor"),
    }
}
