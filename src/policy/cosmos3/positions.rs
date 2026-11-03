//! Cosmos3 Edge 的联合 M-RoPE 位置。
//!
//! 理解流的位置由 Qwen3-VL 规则给出；生成流从「文本最大位置 + 1 + `temporal_modality_margin`」
//! 开始，按帧率在时间轴上推进，空间轴用 patch 网格。

// 位置计算在整条流水线接通前不参与运行时构建；届时移除此豁免。
#![allow(dead_code)]

use super::config::Cosmos3Config;

/// 生成流的隐变量网格。
#[derive(Clone, Copy, Debug)]
pub(super) struct GenGrid {
    pub(super) frames: usize,
    pub(super) height: usize,
    pub(super) width: usize,
}

/// 生成流的布局参数。
#[derive(Clone, Copy, Debug)]
pub(super) struct GenPlan {
    pub(super) batch: usize,
    pub(super) text_len: usize,
    pub(super) grid: GenGrid,
    pub(super) second_per_frame: f32,
    pub(super) num_actions: usize,
    pub(super) action_offset: i64,
}

/// 在理解流位置后追加生成流位置。
///
/// `text` 是 `[3][batch * text_len]` 的扁平位置；返回同布局的
/// `[3][batch * (text_len + 视频 + 动作)]`。
pub(super) fn append_generation(
    text: [&[f32]; 3],
    plan: &GenPlan,
    cfg: &Cosmos3Config,
) -> [Vec<f32>; 3] {
    let GenPlan {
        batch,
        text_len,
        grid,
        second_per_frame,
        num_actions,
        action_offset,
    } = *plan;
    let patch = cfg.latent_patch_size;
    let grid_h = grid.height.div_ceil(patch);
    let grid_w = grid.width.div_ceil(patch);
    let compression = cfg.vae.scale_factor_temporal as f32;
    let tokens_per_second = cfg.base_fps as f32 / compression;

    // 每个样本的生成流起点。
    let mut start = vec![0f32; batch];
    for sample in 0..batch {
        let mut largest = f32::MIN;
        for axis in text {
            for token in 0..text_len {
                largest = largest.max(axis[sample * text_len + token]);
            }
        }
        start[sample] = largest + 1.0 + cfg.temporal_modality_margin as f32;
    }

    let frame_step: Vec<f32> = (0..batch)
        .map(|_| tokens_per_second * second_per_frame)
        .collect();
    let video_tokens = grid.frames * grid_h * grid_w;
    let capacity = batch * (text_len + video_tokens + num_actions);
    let mut temporal = Vec::with_capacity(capacity);
    let mut height = Vec::with_capacity(capacity);
    let mut width = Vec::with_capacity(capacity);

    for sample in 0..batch {
        let range = sample * text_len..(sample + 1) * text_len;
        temporal.extend_from_slice(&text[0][range.clone()]);
        height.extend_from_slice(&text[1][range.clone()]);
        width.extend_from_slice(&text[2][range]);

        for frame in 0..grid.frames {
            let time = start[sample] + frame as f32 * frame_step[sample];
            for patch_row in 0..grid_h {
                for patch_column in 0..grid_w {
                    temporal.push(time);
                    height.push(patch_row as f32);
                    width.push(patch_column as f32);
                }
            }
        }

        for action in 0..num_actions {
            let frame = action as f32 + action_offset as f32;
            temporal.push(start[sample] + frame * (frame_step[sample] / compression));
            height.push(0.0);
            width.push(0.0);
        }
    }
    debug_assert_eq!(temporal.len(), capacity);
    [temporal, height, width]
}

/// 理解流位置：文本 token 依次推进，图像/视频 token 用合并后的 patch 网格。
///
/// `token_types` 是 0（文本）、1（图像）、2（视频）；`grids` 按出现顺序给出每个视觉块的
/// `(t, h, w)` patch 网格，视频按帧拆开。只支持 `t == 1` 的网格。
pub(super) fn understanding_positions(
    token_types: &[u8],
    grids: &[(usize, usize, usize)],
    merge: usize,
) -> [Vec<f32>; 3] {
    let total = token_types.len();
    let mut temporal = Vec::with_capacity(total);
    let mut height = Vec::with_capacity(total);
    let mut width = Vec::with_capacity(total);
    let mut current = 0f32;
    let mut grid_index = 0usize;

    let mut start = 0usize;
    while start < total {
        let kind = token_types[start];
        let mut end = start;
        while end < total && token_types[end] == kind {
            end += 1;
        }
        let length = end - start;
        if kind == 0 {
            for offset in 0..length {
                temporal.push(current + offset as f32);
                height.push(current + offset as f32);
                width.push(current + offset as f32);
            }
            current += length as f32;
        } else {
            let (frames, rows, columns) = grids[grid_index];
            grid_index += 1;
            assert_eq!(frames, 1, "视觉网格必须按帧拆开");
            let grid_h = rows / merge;
            let grid_w = columns / merge;
            debug_assert_eq!(grid_h * grid_w, length, "视觉 token 数与网格不一致");
            for _ in 0..grid_h {
                for column in 0..grid_w {
                    width.push(current + column as f32);
                }
            }
            for row in 0..grid_h {
                for _ in 0..grid_w {
                    height.push(current + row as f32);
                }
            }
            for _ in 0..length {
                temporal.push(current);
            }
            current += (rows.max(columns) / merge) as f32;
        }
        start = end;
    }
    [temporal, height, width]
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
    #[ignore = "需要 cosmos3 位置参照"]
    fn understanding_positions_match_reference() {
        let parity = std::path::PathBuf::from(
            std::env::var("PNEUMA_COSMOS3_POSITIONS").expect("PNEUMA_COSMOS3_POSITIONS"),
        );
        let types: Vec<f32> = read_f32(&parity.join("token_types.bin"));
        let types: Vec<u8> = types.iter().map(|value| *value as u8).collect();
        let expected = read_f32(&parity.join("understanding.bin"));
        let got = understanding_positions(&types, &[(1, 4, 4), (1, 4, 4)], 2);
        let total = types.len();
        for axis in 0..3 {
            let want = &expected[axis * total..(axis + 1) * total];
            assert_eq!(got[axis].len(), total, "axis {axis}");
            let max = got[axis]
                .iter()
                .zip(want)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            println!("understanding axis {axis}: max {max:.3e}");
            assert!(max < 1e-6, "axis {axis} differs by {max:.3e}");
        }
    }

    #[test]
    #[ignore = "需要 cosmos3 位置参照"]
    fn joint_positions_match_reference() {
        let parity = std::path::PathBuf::from(
            std::env::var("PNEUMA_COSMOS3_POSITIONS").expect("PNEUMA_COSMOS3_POSITIONS"),
        );
        let text = read_f32(&parity.join("text_positions.bin"));
        let expected = read_f32(&parity.join("positions.bin"));
        let cfg = super::super::config::Cosmos3Config::default();

        let (text_len, batch) = (5usize, 1usize);
        let got = append_generation(
            [&text[0..5], &text[5..10], &text[10..15]],
            &GenPlan {
                batch,
                text_len,
                grid: GenGrid {
                    frames: 2,
                    height: 16,
                    width: 16,
                },
                second_per_frame: 0.5,
                num_actions: 4,
                action_offset: 1,
            },
            &cfg,
        );
        let total = 137usize;
        assert_eq!(expected.len(), 3 * total);
        for axis in 0..3 {
            assert_eq!(got[axis].len(), total, "axis {axis} length");
            let want = &expected[axis * total..(axis + 1) * total];
            let max = got[axis]
                .iter()
                .zip(want)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            println!("axis {axis}: max {max:.3e}");
            assert!(max < 1e-4, "axis {axis} differs by {max:.3e}");
        }
    }
}
