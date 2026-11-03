//! Cosmos3 Edge 的合约校验与处理器管线。
//!
//! 输入侧：相机视图 → 画布 → JSON 提示 → 分词 → 设备。输出侧只做搬回宿主。

// 处理器在转换接通前不参与运行时构建；届时移除此豁免。
#![allow(dead_code)]

use std::path::Path;

use anyhow::{Context, Result, ensure};

use crate::contract::FeatureSpec;
use crate::processor::{DType, Field, Layout, Pipeline, Schema, Step};

use super::config::Cosmos3Config;

/// 相机视角数量上限，与官方的三视角布局一致。
const MAX_CAMERAS: usize = 3;

/// 校验部署合约与 checkpoint 配置相容。
pub(super) fn validate_contract(
    cfg: &Cosmos3Config,
    features: &[FeatureSpec],
    action: &FeatureSpec,
) -> Result<()> {
    ensure!(
        action.shape.len() == 1,
        "cosmos3 action must be a flat vector, got {:?}",
        action.shape
    );
    let action_dim = action.shape[0] as usize;
    ensure!(
        action_dim > 0 && action_dim <= cfg.max_action_dim,
        "cosmos3 action dim {action_dim} must be within 1..={}",
        cfg.max_action_dim
    );
    let cameras: Vec<&FeatureSpec> = features
        .iter()
        .filter(|feature| feature.kind == "rgb")
        .collect();
    ensure!(
        !cameras.is_empty() && cameras.len() <= MAX_CAMERAS,
        "cosmos3 needs 1..={MAX_CAMERAS} camera views, got {}",
        cameras.len()
    );
    for camera in &cameras {
        ensure!(
            camera.shape.len() == 3 && camera.shape[2] == 3,
            "cosmos3 camera {} must be HWC with three channels",
            camera.name
        );
    }
    if let Some(state) = features
        .iter()
        .find(|feature| feature.kind != "rgb" && feature.name.contains("state"))
    {
        let dim = state.shape.last().copied().unwrap_or(0) as usize;
        ensure!(
            dim > 0 && dim <= cfg.max_action_dim,
            "cosmos3 state dim {dim} must be within 1..={}",
            cfg.max_action_dim
        );
    }
    ensure!(
        cfg.chunk_size > 0 && cfg.n_action_steps > 0,
        "cosmos3 chunk and action steps must be positive"
    );
    Ok(())
}

/// 构建输入与输出管线。
pub(super) fn import_pipelines(
    cfg: &Cosmos3Config,
    tokenizer_dir: &Path,
    features: &[FeatureSpec],
    action: &FeatureSpec,
) -> Result<(Pipeline, Pipeline)> {
    let mut schema = Schema::new();
    let mut steps = vec![Step::Batch];

    let cameras: Vec<&FeatureSpec> = features
        .iter()
        .filter(|feature| feature.kind == "rgb")
        .collect();
    for camera in &cameras {
        let height = camera.shape[0] as usize;
        let width = camera.shape[1] as usize;
        schema.insert(
            camera.name.clone(),
            Field {
                shape: vec![Some(height), Some(width), Some(3)],
                dtype: DType::U8,
                layout: Layout::Hwc,
                optional: false,
            },
        );
        steps.push(Step::Rgb {
            key: camera.name.clone(),
        });
    }

    // 视角尺寸取第一台相机；画布尺寸由档位表推出。
    let view_height = cameras[0].shape[0] as usize;
    let view_width = cameras[0].shape[1] as usize;
    let canvas = super::canvas::canvas_size(view_height, view_width * cameras.len())
        .context("unsupported cosmos3 canvas size")?;
    steps.push(Step::Cosmos3Canvas {
        cameras: cameras.iter().map(|camera| camera.name.clone()).collect(),
        view_width,
        view_height,
        output: "observation.images.canvas".into(),
    });

    if let Some(state) = features
        .iter()
        .find(|feature| feature.kind != "rgb" && feature.name.contains("state"))
    {
        let dim = state.shape.last().copied().unwrap_or(0) as usize;
        schema.insert(
            state.name.clone(),
            Field {
                shape: vec![Some(dim)],
                dtype: DType::F32,
                layout: Layout::Flat,
                optional: false,
            },
        );
    }

    let fps = cfg.fps.unwrap_or(20.0);
    ensure!(fps > 0.0, "cosmos3 fps must be positive");
    steps.push(Step::Cosmos3Caption {
        height: canvas.1,
        width: canvas.0,
        fps,
        chunk_size: cfg.chunk_size,
        use_state: cfg.use_state,
        view: String::new(),
        canvas_key: Some("observation.images.canvas".into()),
    });
    steps.push(Step::Tokenize {
        tokenizer: tokenizer_dir.to_path_buf(),
        max_length: 4096,
        padding: "longest".into(),
        padding_side: "right".into(),
        special_tokens: false,
        negative: Some(super::prompt::text("")),
    });
    steps.push(Step::Device);

    let input = Pipeline::new(schema, steps)?;
    let action_dim = action.shape[0] as usize;
    let output_schema = Schema::from([(
        "action".to_string(),
        Field {
            shape: vec![None, Some(cfg.chunk_size), Some(action_dim)],
            dtype: DType::F32,
            layout: Layout::Flat,
            optional: false,
        },
    )]);
    let output = Pipeline::new(output_schema, Vec::new())?;
    Ok((input, output))
}
