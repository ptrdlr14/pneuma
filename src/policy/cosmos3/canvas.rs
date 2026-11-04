//! Cosmos3 Edge 的画布：多视角拼接、分辨率档位与双三次抗锯齿缩放。
//!
//! 与官方 `Cosmos3VideoCanvasProcessorStep` 对齐：档位由较短边决定，按宽高比选画布，
//! 只缩小不放大，缩放用双三次（`antialias=True`），右下不足处按反射或复制填充。

// 画布在处理器接线前不参与运行时构建；届时移除此豁免。
#![allow(dead_code)]

use burn::tensor::Tensor;

/// 单个档位：较短边上限与各宽高比画布 `(名字, 宽, 高)`。
type Tier = (usize, &'static [(&'static str, usize, usize)]);

/// 官方分辨率档位。
const TIERS: [Tier; 3] = [
    (
        256,
        &[
            ("1,1", 256, 256),
            ("4,3", 320, 256),
            ("3,4", 256, 320),
            ("16,9", 320, 192),
            ("9,16", 192, 320),
        ],
    ),
    (
        480,
        &[
            ("1,1", 640, 640),
            ("4,3", 736, 544),
            ("3,4", 544, 736),
            ("16,9", 832, 480),
            ("9,16", 480, 832),
            ("2,3", 512, 768),
        ],
    ),
    (
        720,
        &[
            ("1,1", 960, 960),
            ("4,3", 1104, 832),
            ("3,4", 832, 1104),
            ("16,9", 1280, 720),
            ("9,16", 720, 1280),
        ],
    ),
];

/// 画布尺寸 `(宽, 高)`：较短边决定档位，宽高比最接近的画布胜出，平局取表内靠前者。
pub(crate) fn canvas_size(height: usize, width: usize) -> Option<(usize, usize)> {
    let shorter = height.min(width);
    let tier = TIERS
        .iter()
        .find(|(limit, _)| shorter <= *limit)
        .map(|(_, sizes)| *sizes)?;
    let target = height as f64 / width as f64;
    let mut best = tier[0];
    let mut best_error = (target - best.2 as f64 / best.1 as f64).abs();
    for entry in &tier[1..] {
        let error = (target - entry.2 as f64 / entry.1 as f64).abs();
        if error < best_error {
            best = *entry;
            best_error = error;
        }
    }
    Some((best.1, best.2))
}

/// 内容尺寸 `(宽, 高)`：只缩小不放大，四舍五入。
pub(crate) fn content_size(height: usize, width: usize, canvas: (usize, usize)) -> (usize, usize) {
    let scale = (canvas.0 as f64 / width as f64)
        .min(canvas.1 as f64 / height as f64)
        .min(1.0);
    (
        (scale * width as f64 + 0.5) as usize,
        (scale * height as f64 + 0.5) as usize,
    )
}

/// 双三次插值核（`a = -0.5`，与 PyTorch 抗锯齿缩放一致）。
fn cubic(x: f64) -> f64 {
    let a = -0.5;
    let x = x.abs();
    if x <= 1.0 {
        ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * a
    } else {
        0.0
    }
}

/// 单轴采样权重：每个输出位置对应一组 `(输入索引, 权重)`，越界采样点丢弃后重新归一化。
fn axis_weights(input: usize, output: usize) -> Vec<Vec<(usize, f64)>> {
    axis_weights_with(input, output, cubic, 2.0)
}

/// 以给定核函数计算单轴权重。
fn axis_weights_with(
    input: usize,
    output: usize,
    kernel: fn(f64) -> f64,
    support_base: f64,
) -> Vec<Vec<(usize, f64)>> {
    let scale = input as f64 / output as f64;
    let filter_scale = scale.max(1.0);
    let support = support_base * filter_scale;
    (0..output)
        .map(|index| {
            let center = (index as f64 + 0.5) * scale - 0.5;
            let left = (center - support).floor() as i64;
            let right = (center + support).ceil() as i64;
            let mut taps = Vec::new();
            let mut total = 0.0;
            // 与 torch 一致：越界采样点直接丢弃，再按有效权重归一化。
            for position in left..=right {
                if position < 0 || position >= input as i64 {
                    continue;
                }
                let weight = kernel((position as f64 - center) / filter_scale);
                if weight == 0.0 {
                    continue;
                }
                taps.push((position as usize, weight));
                total += weight;
            }
            for tap in &mut taps {
                tap.1 /= total;
            }
            taps
        })
        .collect()
}

/// 双三次抗锯齿缩放单通道平面 `[height, width]`。
pub(crate) fn resize_plane(
    plane: &[f64],
    height: usize,
    width: usize,
    out_height: usize,
    out_width: usize,
) -> Vec<f64> {
    let columns = axis_weights(width, out_width);
    let rows = axis_weights(height, out_height);
    let mut horizontal = vec![0.0; height * out_width];
    for row in 0..height {
        for (index, taps) in columns.iter().enumerate() {
            let mut value = 0.0;
            for (column, weight) in taps {
                value += plane[row * width + column] * weight;
            }
            horizontal[row * out_width + index] = value;
        }
    }
    let mut out = vec![0.0; out_height * out_width];
    for (row, taps) in rows.iter().enumerate() {
        for column in 0..out_width {
            let mut value = 0.0;
            for (source, weight) in taps {
                value += horizontal[source * out_width + column] * weight;
            }
            out[row * out_width + column] = value;
        }
    }
    out
}

#[cfg(test)]
mod reference {
    use super::*;
    use std::path::Path;

    fn read_f32(path: &Path) -> Vec<f32> {
        let bytes = std::fs::read(path)
            .unwrap_or_else(|error| panic!("reading {}: {error}", path.display()));
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|value| f32::from_le_bytes(*value))
            .collect()
    }

    #[test]
    #[ignore = "需要 torch 参照"]
    fn resize_matches_torch() {
        let parity = std::path::PathBuf::from(
            std::env::var("PNEUMA_COSMOS3_CANVAS").expect("PNEUMA_COSMOS3_CANVAS"),
        );
        let plane: Vec<f64> = read_f32(&parity.join("plane.bin"))
            .into_iter()
            .map(f64::from)
            .collect();
        let expected = read_f32(&parity.join("resized.bin"));
        let resized = resize_plane(&plane, 256, 512, 160, 320);
        let max = resized
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - f64::from(*b)).abs())
            .fold(0.0f64, f64::max);
        println!("resize max {max:.3e}");
        assert!(max < 5e-3, "resize differs by {max:.3e}");
    }
}

/// 单平面双线性缩放（`align_corners = false`，无抗锯齿），供三视角布局使用。
pub(crate) fn resize_bilinear(
    plane: &[f64],
    height: usize,
    width: usize,
    out_height: usize,
    out_width: usize,
) -> Vec<f64> {
    let mut out = vec![0.0; out_height * out_width];
    let scale_y = height as f64 / out_height as f64;
    let scale_x = width as f64 / out_width as f64;
    for row in 0..out_height {
        let source_y = ((row as f64 + 0.5) * scale_y - 0.5).max(0.0);
        let y0 = source_y.floor() as usize;
        let y1 = (y0 + 1).min(height - 1);
        let wy = source_y - y0 as f64;
        for column in 0..out_width {
            let source_x = ((column as f64 + 0.5) * scale_x - 0.5).max(0.0);
            let x0 = source_x.floor() as usize;
            let x1 = (x0 + 1).min(width - 1);
            let wx = source_x - x0 as f64;
            let top = plane[y0 * width + x0] * (1.0 - wx) + plane[y0 * width + x1] * wx;
            let bottom = plane[y1 * width + x0] * (1.0 - wx) + plane[y1 * width + x1] * wx;
            out[row * out_width + column] = top * (1.0 - wy) + bottom * wy;
        }
    }
    out
}

/// 三角形核，供双线性抗锯齿缩放使用。
fn triangle(x: f64) -> f64 {
    (1.0 - x.abs()).max(0.0)
}

/// 双线性抗锯齿缩放单平面 `[h, w]`。
pub(crate) fn resize_bilinear_antialias(
    plane: &[f64],
    height: usize,
    width: usize,
    out_height: usize,
    out_width: usize,
) -> Vec<f64> {
    let columns = axis_weights_with(width, out_width, triangle, 1.0);
    let rows = axis_weights_with(height, out_height, triangle, 1.0);
    let mut horizontal = vec![0.0; height * out_width];
    for row in 0..height {
        for (index, taps) in columns.iter().enumerate() {
            let mut value = 0.0;
            for (column, weight) in taps {
                value += plane[row * width + column] * weight;
            }
            horizontal[row * out_width + index] = value;
        }
    }
    let mut out = vec![0.0; out_height * out_width];
    for (row, taps) in rows.iter().enumerate() {
        for column in 0..out_width {
            let mut value = 0.0;
            for (source, weight) in taps {
                value += horizontal[source * out_width + column] * weight;
            }
            out[row * out_width + column] = value;
        }
    }
    out
}

/// 多视角拼接，输入为 `[3,h,w]` 的 `[0,1]` 视图。
pub(crate) fn compose(views: &[&Tensor<3>]) -> anyhow::Result<Tensor<3>> {
    anyhow::ensure!(!views.is_empty(), "至少需要一个相机视角");
    let [channels, height, width] = views[0].dims();
    if views.len() == 1 {
        return Ok(views[0].clone());
    }
    if views.len() == 2 {
        return Ok(Tensor::cat(vec![views[0].clone(), views[1].clone()], 2));
    }
    anyhow::ensure!(views.len() == 3, "最多支持三个相机视角");
    let half_height = height / 2;
    let split = width / 2;
    let halves: Vec<Tensor<3>> = [(split, 0usize), (width - split, split)]
        .iter()
        .zip(views[1..].iter())
        .map(|((target_width, _), view)| {
            let plane = (*view)
                .clone()
                .into_data()
                .iter::<f32>()
                .collect::<Vec<_>>();
            let plane: Vec<f64> = plane.into_iter().map(f64::from).collect();
            let mut out = Vec::with_capacity(channels * half_height * target_width);
            for channel in 0..channels {
                let source = &plane[channel * height * width..(channel + 1) * height * width];
                out.extend(resize_bilinear(
                    source,
                    height,
                    width,
                    half_height,
                    *target_width,
                ));
            }
            let values: Vec<f32> = out.into_iter().map(|value| value as f32).collect();
            Tensor::<3>::from_data(
                burn::tensor::TensorData::new(values, [channels, half_height, *target_width]),
                &view.device(),
            )
        })
        .collect();
    let bottom = Tensor::cat(halves, 2);
    Ok(Tensor::cat(vec![views[0].clone(), bottom], 1))
}

/// 反射索引（不含边缘重复，与 `torch.nn.functional.pad(mode="reflect")` 一致）。
fn reflect_index(index: i64, size: usize) -> usize {
    if size == 1 {
        return 0;
    }
    let period = 2 * (size as i64 - 1);
    let mut value = index % period;
    if value < 0 {
        value += period;
    }
    if value >= size as i64 {
        (period - value) as usize
    } else {
        value as usize
    }
}

/// 补齐到画布：缩放后按反射或复制填右下角。
pub(crate) fn fit(
    frames: &Tensor<3>,
    canvas: (usize, usize),
) -> anyhow::Result<(Tensor<3>, (usize, usize))> {
    let [channels, height, width] = frames.dims();
    let content = content_size(height, width, canvas);
    let (content_w, content_h) = content;
    let plane: Vec<f64> = frames
        .clone()
        .into_data()
        .iter::<f32>()
        .map(f64::from)
        .collect();
    let mut content_plane = Vec::with_capacity(channels * content_h * content_w);
    for channel in 0..channels {
        let source = &plane[channel * height * width..(channel + 1) * height * width];
        if (content_h, content_w) == (height, width) {
            content_plane.extend_from_slice(source);
        } else {
            content_plane.extend(resize_plane(source, height, width, content_h, content_w));
        }
    }
    let pad_right = canvas.0 - content_w;
    let pad_bottom = canvas.1 - content_h;
    let replicate = pad_right >= content_w || pad_bottom >= content_h;
    let mut out = vec![0f64; channels * canvas.1 * canvas.0];
    for channel in 0..channels {
        let source =
            &content_plane[channel * content_h * content_w..(channel + 1) * content_h * content_w];
        for row in 0..canvas.1 {
            for column in 0..canvas.0 {
                let (source_row, source_column) = if row < content_h && column < content_w {
                    (row, column)
                } else if replicate {
                    (row.min(content_h - 1), column.min(content_w - 1))
                } else {
                    (
                        reflect_index(row as i64, content_h),
                        reflect_index(column as i64, content_w),
                    )
                };
                out[channel * canvas.1 * canvas.0 + row * canvas.0 + column] = source
                    [source_row * content_w + source_column]
                    .round()
                    .clamp(0.0, 255.0);
            }
        }
    }
    let values: Vec<f32> = out.into_iter().map(|value| value as f32).collect();
    Ok((
        Tensor::<3>::from_data(
            burn::tensor::TensorData::new(values, [channels, canvas.1, canvas.0]),
            &frames.device(),
        ),
        content,
    ))
}

#[cfg(test)]
mod compose_reference {
    use super::*;
    use burn::tensor::TensorData;
    use std::path::Path;

    fn read_f32(path: &Path) -> Vec<f32> {
        let bytes = std::fs::read(path)
            .unwrap_or_else(|error| panic!("reading {}: {error}", path.display()));
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|value| f32::from_le_bytes(*value))
            .collect()
    }

    #[test]
    #[ignore = "需要官方画布参照"]
    fn canvas_matches_official_step() {
        let parity = std::path::PathBuf::from(
            std::env::var("PNEUMA_COSMOS3_COMPOSE").expect("PNEUMA_COSMOS3_COMPOSE"),
        );
        let device = burn::tensor::Device::flex();
        let first = Tensor::<3>::from_data(
            TensorData::new(read_f32(&parity.join("first.bin")), [3, 256, 256]),
            &device,
        );
        let second = Tensor::<3>::from_data(
            TensorData::new(read_f32(&parity.join("second.bin")), [3, 256, 256]),
            &device,
        );
        let composed = compose(&[&first, &second]).expect("compose");
        assert_eq!(composed.dims(), [3, 256, 512]);
        // 官方先量化到 uint8（截断），再适配画布。
        let quantized: Vec<f32> = composed
            .into_data()
            .iter::<f32>()
            .map(|value| (value * 255.0).clamp(0.0, 255.0) as u8 as f32)
            .collect();
        let quantized = Tensor::<3>::from_data(TensorData::new(quantized, [3, 256, 512]), &device);
        let (canvas, content) = fit(&quantized, (320, 192)).expect("fit");
        assert_eq!(content, (320, 160));
        assert_eq!(canvas.dims(), [3, 192, 320]);
        let pixels: Vec<f32> = canvas
            .into_data()
            .iter::<f32>()
            .map(|value| value / 127.5 - 1.0)
            .collect();
        let expected = read_f32(&parity.join("canvas.bin"));
        assert_eq!(pixels.len(), expected.len());
        // 与官方量化结果最多差一个灰阶（1/127.5），并统计比例。
        let levels: Vec<f32> = pixels
            .iter()
            .zip(&expected)
            .map(|(a, b)| ((a - b).abs() * 127.5).round())
            .collect();
        let max = levels.iter().copied().fold(0.0f32, f32::max);
        let differing = levels.iter().filter(|level| **level > 0.0).count();
        let fraction = differing as f64 / levels.len() as f64;
        println!("canvas levels max {max} differing {differing} ({fraction:.4})");
        assert!(max <= 1.0, "canvas differs by {max} levels");
        assert!(fraction < 0.01, "too many differing pixels: {fraction:.4}");
    }
}
