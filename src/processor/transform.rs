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
                value.insert_batch_dim()?;
            }
        }
        Step::Newline => {
            for task in &mut batch.tasks {
                if !task.ends_with('\n') {
                    task.push('\n');
                }
            }
        }
        Step::Prompt { state_key, bins } => prompt(batch, state_key, *bins)?,
        Step::Tokenize {
            special_tokens,
            negative,
            ..
        } => super::tokenize::run(
            batch,
            context,
            tokenizer.context("missing prepared tokenizer")?,
            *special_tokens,
            negative.as_deref(),
        )?,
        Step::Cosmos3Canvas {
            cameras,
            view_width,
            view_height,
            output,
        } => cosmos3_canvas(batch, cameras, *view_width, *view_height, output, context)?,
        Step::Cosmos3Caption {
            height,
            width,
            fps,
            chunk_size,
            use_state,
            view,
            ..
        } => cosmos3_caption(batch, *height, *width, *fps, *chunk_size, *use_state, view)?,
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

/// 按 openpi `PaligemmaTokenizer` 的 256 桶离散化重写任务文本。
fn prompt(batch: &mut Batch, state_key: &str, bins: usize) -> Result<()> {
    let state = batch
        .values
        .get(state_key)
        .with_context(|| format!("prompt state {state_key} is missing"))?;
    let values = state.host_f32()?;
    let shape = state.shape();
    ensure!(shape.len() == 2, "Prompt requires a batched state");
    let (rows, width) = (shape[0], shape[1]);
    ensure!(
        rows == batch.tasks.len(),
        "prompt state rows disagree with task count"
    );
    // 与 np.linspace(-1, 1, bins + 1)[:-1] 相同的桶边界。
    let edges: Vec<f64> = (0..bins)
        .map(|k| -1.0 + 2.0 * k as f64 / bins as f64)
        .collect();
    for (index, task) in batch.tasks.iter_mut().enumerate() {
        let bins_text: Vec<String> = values[index * width..(index + 1) * width]
            .iter()
            .map(|value| {
                let value = f64::from(*value);
                let bin = edges.iter().filter(|&&edge| edge <= value).count();
                (bin as i64 - 1).to_string()
            })
            .collect();
        let clean = task.trim().replace(['_', '\n'], " ");
        *task = format!("Task: {clean}, State: {};\nAction: ", bins_text.join(" "));
    }
    Ok(())
}

/// 把多个相机视角拼成 Cosmos3 的视频画布，输出 `[-1,1]` 的 `[1,3,H,W]`。
fn cosmos3_canvas(
    batch: &mut Batch,
    cameras: &[String],
    view_width: usize,
    view_height: usize,
    output: &str,
    context: &Context,
) -> Result<()> {
    use crate::policy::cosmos3::canvas;
    let mut views = Vec::with_capacity(cameras.len());
    for key in cameras {
        let data = batch
            .values
            .get(key)
            .with_context(|| format!("cosmos3 canvas camera {key} is missing"))?
            .clone();
        let tensor = match data {
            Data::Float4(tensor) => tensor,
            // `Rgb` 产出宿主 f32；这里搬到设备再继续。
            Data::Host(host) if host.shape().len() == 4 && host.shape()[1] == 3 => {
                burn::tensor::Tensor::<4>::from_data(host, &context.device)
            }
            _ => bail!("cosmos3 canvas camera {key} must be a rank-4 RGB float image"),
        };
        let [batch_size, channels, height, width] = tensor.dims();
        ensure!(
            batch_size == 1 && channels == 3,
            "cosmos3 canvas expects single-sample RGB views"
        );
        let mut view = tensor.clone().reshape([3, height, width]);
        if (height, width) != (view_height, view_width) {
            let plane: Vec<f64> = view
                .clone()
                .into_data()
                .iter::<f32>()
                .map(|value| f64::from(value) * 255.0)
                .collect();
            let mut resized = Vec::with_capacity(3 * view_height * view_width);
            for channel in 0..3 {
                let source = &plane[channel * height * width..(channel + 1) * height * width];
                resized.extend(
                    canvas::resize_bilinear_antialias(
                        source,
                        height,
                        width,
                        view_height,
                        view_width,
                    )
                    .into_iter()
                    .map(|value| value.round().clamp(0.0, 255.0) / 255.0),
                );
            }
            let values: Vec<f32> = resized.into_iter().map(|value| value as f32).collect();
            view = burn::tensor::Tensor::<3>::from_data(
                burn::tensor::TensorData::new(values, [3, view_height, view_width]),
                &context.device,
            );
        }
        views.push(view);
    }
    let references: Vec<&burn::tensor::Tensor<3>> = views.iter().collect();
    let composed = canvas::compose(&references)?;
    let [_, height, width] = composed.dims();
    let quantized: Vec<f32> = composed
        .into_data()
        .iter::<f32>()
        .map(|value| (value * 255.0).clamp(0.0, 255.0) as u8 as f32)
        .collect();
    let quantized = burn::tensor::Tensor::<3>::from_data(
        burn::tensor::TensorData::new(quantized, [3, height, width]),
        &context.device,
    );
    let target = canvas::canvas_size(height, width).context("unsupported cosmos3 canvas size")?;
    let (fitted, _) = canvas::fit(&quantized, target)?;
    let [_, canvas_h, canvas_w] = fitted.dims();
    let pixels: Vec<f32> = fitted
        .into_data()
        .iter::<f32>()
        .map(|value| value / 127.5 - 1.0)
        .collect();
    batch.values.insert(
        output.to_string(),
        Data::Float4(burn::tensor::Tensor::<4>::from_data(
            burn::tensor::TensorData::new(pixels, [1, 3, canvas_h, canvas_w]),
            &context.device,
        )),
    );
    Ok(())
}

/// 把每个任务改写成 Cosmos3 的 JSON 提示，并套上聊天模板的前后缀。
fn cosmos3_caption(
    batch: &mut Batch,
    height: usize,
    width: usize,
    fps: f64,
    chunk_size: usize,
    use_state: bool,
    view: &str,
) -> Result<()> {
    let canvas = crate::policy::cosmos3::Canvas { height, width };
    for task in &mut batch.tasks {
        let caption =
            crate::policy::cosmos3::caption(task, canvas, fps, 0, chunk_size, use_state, view);
        *task = crate::policy::cosmos3::prompt_text(&caption);
    }
    Ok(())
}

fn device(value: Data, context: &Context) -> Result<Data> {
    Ok(match value {
        Data::Host(data) => {
            // `Rgb` 之前的 uint8 图像不能直接上传，必须先转成 f32。
            ensure!(
                data.dtype() == burn::tensor::DType::F32,
                "Device requires RGB conversion before upload"
            );
            match data.shape().len() {
                2 => Data::Float2(Tensor::from_data(data, &context.device)),
                3 => Data::Float3(Tensor::from_data(data, &context.device)),
                4 => Data::Float4(Tensor::from_data(data, &context.device)),
                _ => bail!("Device requires float rank 2, 3 or 4"),
            }
        }
        Data::Float2(t) => Data::Float2(t.to_device(&context.device)),
        Data::Float3(t) => Data::Float3(t.to_device(&context.device)),
        Data::Float4(t) => Data::Float4(t.to_device(&context.device)),
        Data::Int2(t) => Data::Int2(t.to_device(&context.device)),
        Data::Bool2(t) => Data::Bool2(t.to_device(&context.device)),
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
        Ok(Data::f32(values, shape))
    }
    match value {
        Data::Float2(t) => float(t),
        Data::Float3(t) => float(t),
        Data::Float4(t) => float(t),
        other => Ok(other),
    }
}

fn rgb(value: Data) -> Result<Data> {
    let values = value.host_u8()?;
    let shape = value.shape();
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
    Ok(Data::f32(out, vec![batch, 3, height, width]))
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
    let values = value.host_f32()?;
    let shape = value.shape();
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
    Ok(Data::f32(out, shape))
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
        Data::Host(mut data) => {
            let shape = data.shape().to_vec();
            let stride = element_count(&shape[axis + 1..])?;
            for (i, value) in data
                .as_mut_slice::<f32>()
                .context("affine requires host f32")?
                .iter_mut()
                .enumerate()
            {
                let index = (i / stride) % width;
                let offset = offset[if offset.len() == 1 { 0 } else { index }];
                let scale = scale[if scale.len() == 1 { 0 } else { index }];
                *value = if linear {
                    *value * scale + offset
                } else {
                    (*value - offset) / scale
                };
            }
            Data::Host(data)
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
        Data::Host(data) => {
            shape[last] = width;
            let mut out = vec![0.0; element_count(&shape)?];
            let values = data.as_slice::<f32>().context("Pad requires host f32")?;
            for (input, output) in values.chunks_exact(old).zip(out.chunks_exact_mut(width)) {
                let count = old.min(width);
                output[..count].copy_from_slice(&input[..count]);
            }
            Ok(Data::f32(out, shape))
        }
        Data::Float3(t) if crop => Ok(Data::Float3(t.slice_dim(2, 0..width))),
        _ => bail!("Pad requires host f32; Crop accepts host f32 or rank-3 float tensor"),
    }
}
