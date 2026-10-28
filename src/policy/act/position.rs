//! ACTSinusoidalPositionEmbedding2d 的等价纯函数实现（无参数模块）。
//! 逐式对齐 modeling_act.py：位置索引从 1 起、除以 (max+eps) 归一到 [0,2π]、
//! 频率按 10000^(2*(j//2)/dimension)，x/y 各产生 interleave 的 sin/cos，y 在前。

use burn::tensor::{Device, Tensor};

/// 返回 (h*w, 2*dimension) 行主序 (h 外层、w 内层) 的位置编码，
/// 与 torch 端 `rearrange("b c h w -> (h w) b c")` 的展平顺序一致。
pub fn sinusoidal_2d(device: &Device, height: usize, width: usize, dimension: usize) -> Tensor<2> {
    let two_pi = 2.0 * std::f32::consts::PI;
    let eps = 1e-6;
    let temperature = 10000.0f32;
    // inverse_frequency[e] = 10000^(2*(e//2)/dimension)，e in 0..dimension
    let inv_freq: Vec<f32> = (0..dimension)
        .map(|e| temperature.powf(2.0 * ((e / 2) as f32) / dimension as f32))
        .collect();

    let mut values = vec![0f32; height * width * 2 * dimension];
    for h in 0..height {
        for w in 0..width {
            // cumsum 起点为 1（源码注释：匹配原实现，不是 range(0, H)）。
            let y_norm = ((h + 1) as f32) / ((height as f32) + eps) * two_pi;
            let x_norm = ((w + 1) as f32) / ((width as f32) + eps) * two_pi;
            let row = h * width + w;
            let base = row * 2 * dimension;
            // pos_embed_y 占据前 dimension 个通道，pos_embed_x 占据后 dimension 个。
            for (offset, norm) in [(0usize, y_norm), (dimension, x_norm)] {
                for e in 0..dimension {
                    let angle = norm / inv_freq[e];
                    // [...,0::2].sin() 与 [...,1::2].cos() 交错堆叠。
                    let v = if e % 2 == 0 { angle.sin() } else { angle.cos() };
                    values[base + offset + e] = v;
                }
            }
        }
    }
    Tensor::<2, _>::from_data(
        burn::tensor::TensorData::new(values, [height * width, 2 * dimension]),
        device,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 与 Python 参照值逐元素对齐（scripts/dump_act_reference.py 生成）。
    #[test]
    fn matches_python_reference_small() {
        let device = Device::flex();
        let data = sinusoidal_2d(&device, 3, 4, 8).into_data();
        let got: Vec<f32> = data.iter::<f32>().collect();
        // python 值由 dump_act_reference.py 的 pos2d_small.json 提供。
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/scripts/fixtures/pos2d_small.json"
        );
        if !std::path::Path::new(path).exists() {
            eprintln!("跳过：缺少 fixtures（先运行 scripts/dump_act_reference.py）");
            return;
        }
        let want: Vec<f32> = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!((g - w).abs() < 1e-6, "index {i}: {g} vs {w}");
        }
    }
}
