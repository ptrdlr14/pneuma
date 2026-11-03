//! 从部署 profile 装载模型与 processor，并在启动时绑定执行合约。

use std::sync::Arc;

use anyhow::{Result, ensure};
use burn::prelude::Device;

use crate::{
    config::ServerConfig,
    inference::PolicyRunner,
    policy::{self, Policy, PolicyKind, zero::ZeroActionPolicy},
    processor::{Batch, DType, Data, Field, Layout, Pipeline, Schema},
    protocol::Capabilities,
};

/// 模型、输入输出 pipeline、设备及两个 horizon 的装载产物。
type Loaded = (Box<dyn Policy>, Pipeline, Pipeline, Device, u64, u64);

pub fn build(cfg: &ServerConfig) -> Result<(Arc<PolicyRunner>, Capabilities)> {
    let loaded = match cfg.model.policy {
        PolicyKind::Zero => None,
        PolicyKind::Act | PolicyKind::Smolvla | PolicyKind::Pi05 | PolicyKind::Cosmos3Edge => {
            if matches!(cfg.model.policy, PolicyKind::Act) {
                policy::act::validate_contract(&cfg.features, &cfg.action_feature)?;
            }
            ensure!(
                cfg.model.repo_or_path.join("manifest.json").is_file(),
                "serve requires a native bpk checkpoint; run pneuma convert --from lerobot --to pneuma first"
            );
            Some(load_checkpoint(cfg)?)
        }
    };
    let (policy, pre, post, device, prediction, default_steps): Loaded = if let Some(loaded) =
        loaded
    {
        loaded
    } else {
        match cfg.model.policy {
            PolicyKind::Zero => {
                let prediction = cfg
                    .execution
                    .prediction_steps
                    .ok_or_else(|| anyhow::anyhow!("zero requires execution.prediction_steps"))?;
                let steps = usize::try_from(prediction)?;
                let dim = usize::try_from(cfg.action_feature.shape[0])?;
                ensure!(
                    steps > 0
                        && steps
                            .checked_mul(dim)
                            .and_then(|n| n.checked_mul(4))
                            .is_some(),
                    "action size exceeds capacity"
                );
                let device = Device::flex();
                let output = Schema::from([(
                    "action".into(),
                    Field {
                        shape: vec![Some(1), Some(steps), Some(dim)],
                        dtype: DType::F32,
                        layout: Layout::Flat,
                        optional: false,
                    },
                )]);
                (
                    Box::new(ZeroActionPolicy {
                        steps,
                        dim,
                        device: device.clone(),
                    }),
                    Pipeline::new(input_schema(cfg)?, vec![])?,
                    Pipeline::new(output, vec![])?,
                    device,
                    prediction,
                    prediction,
                )
            }
            PolicyKind::Act | PolicyKind::Smolvla | PolicyKind::Pi05 | PolicyKind::Cosmos3Edge => {
                unreachable!("model checkpoint loaded above")
            }
        }
    };
    let execution = cfg.execution.n_action_steps.unwrap_or(default_steps);
    ensure!(
        execution > 0 && execution <= prediction,
        "execution.n_action_steps must be in 1..={prediction}"
    );
    let declared = input_schema(cfg)?;
    for (name, field) in &declared {
        ensure!(
            pre.input_schema().get(name) == Some(field),
            "deployment input differs from pipeline: {name}"
        );
    }
    ensure!(
        pre.input_schema()
            .iter()
            .all(|(name, field)| field.optional || declared.contains_key(name)),
        "pipeline requires an undeclared deployment input"
    );
    let caps = cfg.capabilities(prediction, execution);
    let runner = Arc::new(PolicyRunner::new(
        policy,
        (pre, post),
        device,
        usize::try_from(prediction)?,
        usize::try_from(execution)?,
        usize::try_from(cfg.action_feature.shape[0])?,
    )?);
    let started = std::time::Instant::now();
    for _ in 0..cfg.execution.warmup_calls {
        runner.predict(zero_observation(cfg)?)?;
    }
    tracing::info!(elapsed = ?started.elapsed(), "pipeline warmup complete");
    Ok((runner, caps))
}

fn load_checkpoint(cfg: &ServerConfig) -> Result<Loaded> {
    let artifact = crate::checkpoint::Checkpoint::load(&cfg.model.repo_or_path)?;
    let (model, device, prediction, execution): (Box<dyn Policy>, _, _, _) = match cfg.model.policy
    {
        PolicyKind::Act => {
            ensure!(
                artifact.policy_kind == PolicyKind::Act,
                "artifact policy differs from deployment"
            );
            policy::act::validate_contract(&cfg.features, &cfg.action_feature)?;
            let model = policy::act::ActPolicy::load_model(&artifact.model_dir, cfg.model.device)?;
            let (prediction, execution) = model.horizon();
            let device = model.device();
            (Box::new(model), device, prediction, execution)
        }
        PolicyKind::Smolvla => {
            ensure!(
                artifact.policy_kind == PolicyKind::Smolvla,
                "artifact policy differs from deployment"
            );
            let model = policy::smolvla::SmolVlaPolicy::load_model(
                &artifact.model_dir,
                artifact.vlm_path.as_deref(),
                cfg.model.device,
            )?;
            let (prediction, execution) = model.horizon();
            let device = model.device();
            (Box::new(model), device, prediction, execution)
        }
        PolicyKind::Pi05 => {
            ensure!(
                artifact.policy_kind == PolicyKind::Pi05,
                "artifact policy differs from deployment"
            );
            let model =
                policy::pi05::Pi05Policy::load_model(&artifact.model_dir, cfg.model.device)?;
            let (prediction, execution) = model.horizon();
            let device = model.device();
            (Box::new(model), device, prediction, execution)
        }
        PolicyKind::Cosmos3Edge => {
            ensure!(
                artifact.policy_kind == PolicyKind::Cosmos3Edge,
                "artifact policy differs from deployment"
            );
            let model =
                policy::cosmos3::Cosmos3Policy::load_model(&artifact.model_dir, cfg.model.device)?;
            let action_dim = cfg.action_feature.shape.first().copied().unwrap_or(0) as usize;
            let model = model.with_action_dim(action_dim);
            let (prediction, execution) = model.horizon();
            let device = model.device();
            (Box::new(model), device, prediction, execution)
        }
        PolicyKind::Zero => anyhow::bail!("zero does not load a model artifact"),
    };
    Ok((
        model,
        artifact.preprocessor,
        artifact.postprocessor,
        device,
        prediction,
        execution,
    ))
}

fn input_schema(cfg: &ServerConfig) -> Result<Schema> {
    cfg.features
        .iter()
        .map(|feature| {
            let dtype = match feature.dtype.as_str() {
                "uint8" => DType::U8,
                "float32" => DType::F32,
                other => anyhow::bail!("pipeline does not support input dtype {other}"),
            };
            Ok((
                feature.name.clone(),
                Field {
                    shape: feature
                        .shape
                        .iter()
                        .map(|&n| usize::try_from(n).map(Some))
                        .collect::<Result<_, _>>()?,
                    dtype,
                    layout: if feature.kind == "rgb" {
                        Layout::Hwc
                    } else {
                        Layout::Flat
                    },
                    optional: false,
                },
            ))
        })
        .collect()
}

fn zero_observation(cfg: &ServerConfig) -> Result<Batch> {
    let mut batch = Batch {
        tasks: vec!["warmup".into()],
        ..Batch::default()
    };
    for feature in &cfg.features {
        let shape: Vec<usize> = feature
            .shape
            .iter()
            .map(|&n| usize::try_from(n))
            .collect::<Result<_, _>>()?;
        let count = shape
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d))
            .ok_or_else(|| anyhow::anyhow!("warmup input size exceeds capacity"))?;
        let value = match feature.dtype.as_str() {
            "uint8" => Data::u8(vec![0; count], shape),
            "float32" => Data::f32(vec![0.; count], shape),
            other => anyhow::bail!("pipeline does not support input dtype {other}"),
        };
        batch.values.insert(feature.name.clone(), value);
    }
    Ok(batch)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_produces_advertised_zero_actions() {
        let cfg = ServerConfig::load(std::path::Path::new("examples/so101.toml")).unwrap();
        let (runner, caps) = build(&cfg).unwrap();
        let chunk = runner.predict(zero_observation(&cfg).unwrap()).unwrap();
        assert_eq!(chunk.steps as u64, caps.execution_steps);
        assert_eq!(
            chunk.values.len() as u64,
            caps.execution_steps * caps.action_feature.shape[0]
        );
        assert!(chunk.values.iter().all(|&v| v == 0.));
    }

    #[test]
    fn model_serving_requires_explicit_native_conversion() {
        let cfg = ServerConfig::load(std::path::Path::new("examples/aloha_sim.toml")).unwrap();
        let Err(error) = build(&cfg) else {
            panic!("external checkpoint accepted by serve")
        };
        assert!(error.to_string().contains("run pneuma convert"), "{error}");
    }

    #[test]
    fn zero_rejects_overflowing_action_size_at_startup() {
        let mut cfg = ServerConfig::load(std::path::Path::new("examples/so101.toml")).unwrap();
        cfg.execution.prediction_steps = Some(u64::MAX);
        let Err(error) = build(&cfg) else {
            panic!("overflow accepted")
        };
        assert!(error.to_string().contains("capacity"));
    }
}
