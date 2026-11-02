//! 真实 checkpoint 数值对齐需要先运行 dump_smolvla_reference.py。

use super::*;
use crate::processor::{Context, Mode};

fn read_f32(dir: &Path, manifest: &serde_json::Value, name: &str) -> Vec<f32> {
    let file = manifest["tensors"][name]["file"]
        .as_str()
        .expect("fixture tensor file");
    std::fs::read(dir.join(file))
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

fn compare(name: &str, actual: &[f32], reference: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), reference.len(), "{name} shape differs");
    assert!(
        actual.iter().all(|v| v.is_finite()),
        "{name} contains non-finite values"
    );
    let max = actual
        .iter()
        .zip(reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let mean = actual
        .iter()
        .zip(reference)
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / actual.len() as f32;
    eprintln!("{name}: max_abs={max:.8e} mean_abs={mean:.8e}");
    assert!(
        max <= tolerance,
        "{name} max absolute difference {max} > {tolerance}"
    );
}

#[test]
#[ignore = "requires local checkpoint, VLM tokenizer and Python reference fixtures"]
fn smolvla_parity() {
    let dir = std::env::var_os("PNEUMA_SMOLVLA_PARITY")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| "outputs/smolvla-parity".into());
    let vlm = std::env::var_os("PNEUMA_SMOLVLA_VLM")
        .map(std::path::PathBuf::from)
        .expect("set PNEUMA_SMOLVLA_VLM to local VLM resources");
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).unwrap()).unwrap();
    let cfg = config::parse(&dir.join("checkpoint/config.json")).unwrap();
    let features: Vec<_> = cfg
        .input_features
        .iter()
        .map(|(name, f)| {
            let visual = f.feature_type == "VISUAL";
            let shape = if visual {
                vec![f.shape[1] as u64, f.shape[2] as u64, 3]
            } else {
                vec![f.shape[0] as u64]
            };
            crate::contract::FeatureSpec {
                name: name.clone(),
                shape,
                dtype: if visual { "uint8" } else { "float32" }.into(),
                kind: if visual { "rgb" } else { "tensor" }.into(),
                names: vec![],
                semantics: "parity".into(),
            }
        })
        .collect();
    let action = crate::contract::FeatureSpec {
        name: "action".into(),
        shape: vec![cfg.action_dim() as u64],
        dtype: "float32".into(),
        kind: "tensor".into(),
        names: vec![],
        semantics: "parity".into(),
    };
    let backend = match std::env::var("BURN_DEVICE").as_deref() {
        Ok("metal") => crate::config::Device::Metal,
        _ => crate::config::Device::Flex,
    };
    let (policy, preprocessor, postprocessor) =
        if let Some(native) = std::env::var_os("PNEUMA_SMOLVLA_NATIVE") {
            let checkpoint = crate::checkpoint::Checkpoint::load(Path::new(&native)).unwrap();
            let policy = SmolVlaPolicy::load_model(
                &checkpoint.model_dir,
                checkpoint.vlm_path.as_deref(),
                backend,
            )
            .unwrap();
            (policy, checkpoint.preprocessor, checkpoint.postprocessor)
        } else {
            let (policy, pre, post, _) = SmolVlaPolicy::import_configured(
                &dir.join("checkpoint"),
                Some(&vlm),
                backend,
                &features,
                &action,
            )
            .unwrap();
            (policy, pre, post)
        };
    let task = manifest["task"].as_str().unwrap_or("pick up the red cube");
    let raw_state = read_f32(&dir, &manifest, "raw_state");
    let mut raw = Batch {
        values: [(
            "observation.state".into(),
            Data::F32 {
                values: raw_state,
                shape: vec![cfg.state_dim()],
            },
        )]
        .into(),
        tasks: vec![task.into()],
    };
    for (index, (name, feature)) in cfg.image_features().enumerate() {
        let values = read_f32(&dir, &manifest, &format!("raw_image_{index}"))
            .into_iter()
            .map(|value| value as u8)
            .collect();
        raw.values.insert(
            name.into(),
            Data::U8 {
                values,
                shape: vec![feature.shape[1], feature.shape[2], 3],
            },
        );
    }
    let context = Context {
        device: policy.device(),
        mode: Mode::Eval,
    };
    let mut prepared = preprocessor.run(raw.clone(), &context).unwrap();
    let state = prepared
        .take("observation.state")
        .unwrap()
        .into_float2()
        .unwrap();
    compare(
        "state_padded",
        &state.clone().into_data().try_to_vec::<f32>().unwrap(),
        &read_f32(&dir, &manifest, "state_padded"),
        1e-6,
    );
    let Data::Int2(tokens) = prepared.take("observation.language.tokens").unwrap() else {
        panic!("expected prepared integer tokens");
    };
    let tokens = tokens.into_data().iter::<i64>().collect::<Vec<_>>();
    let Data::Bool2(valid) = prepared
        .take("observation.language.attention_mask")
        .unwrap()
    else {
        panic!("expected prepared boolean token mask");
    };
    let valid = valid.into_data().iter::<bool>().collect::<Vec<_>>();
    assert_eq!(
        tokens.iter().map(|&value| value as f32).collect::<Vec<_>>(),
        read_f32(&dir, &manifest, "tokens")
    );
    assert_eq!(
        valid
            .iter()
            .map(|&value| u8::from(value) as f32)
            .collect::<Vec<_>>(),
        read_f32(&dir, &manifest, "token_mask")
    );
    let mut images = Vec::new();
    for (index, (name, _)) in cfg.image_features().enumerate() {
        let image = prepared.take(name).unwrap().into_float4().unwrap();
        compare(
            &format!("image_prepared_{index}"),
            &image.clone().into_data().try_to_vec::<f32>().unwrap(),
            &read_f32(&dir, &manifest, &format!("image_prepared_{index}")),
            5e-6,
        );
        images.push(image);
    }
    let model = &policy.model.vlm_with_expert.vlm.model;
    let vision = model.vision_model.forward(images[0].clone());
    compare(
        "vision_hidden",
        &vision.clone().into_data().try_to_vec::<f32>().unwrap(),
        &read_f32(&dir, &manifest, "vision_hidden"),
        5e-4,
    );
    let connector = model.connector.forward(vision);
    compare(
        "connector",
        &connector.into_data().try_to_vec::<f32>().unwrap(),
        &read_f32(&dir, &manifest, "connector"),
        1e-3,
    );
    let (cache, prefix_valid) =
        policy
            .model
            .prefix(images.clone(), &tokens, &valid, state.clone(), &cfg);
    for (index, cached) in cache.iter().enumerate() {
        for (kind, value) in [("key", &cached.key), ("value", &cached.value)] {
            let name = format!("prefix_{kind}_{index}");
            compare(
                &name,
                &value.clone().into_data().try_to_vec::<f32>().unwrap(),
                &read_f32(&dir, &manifest, &name),
                3e-3,
            );
        }
    }
    let noise = Tensor::<3>::from_data(
        TensorData::new(
            read_f32(&dir, &manifest, "noise"),
            [1, cfg.chunk_size, cfg.max_action_dim],
        ),
        &policy.device,
    );
    let velocity = policy
        .model
        .velocity(noise.clone(), 1., &cache, &prefix_valid, &cfg);
    compare(
        "velocity_step_0",
        &velocity.into_data().try_to_vec::<f32>().unwrap(),
        &read_f32(&dir, &manifest, "velocity_step_0"),
        2e-3,
    );
    let output = policy
        .model
        .sample(images, &tokens, &valid, state, noise, &cfg)
        .into_data()
        .try_to_vec::<f32>()
        .unwrap();
    compare(
        "actions_padded",
        &output,
        &read_f32(&dir, &manifest, "actions_padded"),
        2e-3,
    );
    let padded = Batch {
        values: [(
            "action".into(),
            Data::Float3(Tensor::from_data(
                TensorData::new(output, [1, cfg.chunk_size, cfg.max_action_dim]),
                &context.device,
            )),
        )]
        .into(),
        ..Batch::default()
    };
    let canonical = postprocessor
        .run(padded, &context)
        .unwrap()
        .take("action")
        .unwrap()
        .into_float3()
        .unwrap()
        .into_data()
        .try_to_vec::<f32>()
        .unwrap();
    compare(
        "actions_canonical",
        &canonical,
        &read_f32(&dir, &manifest, "actions_canonical"),
        2e-3,
    );

    // 通用模型接口返回完整 horizon 和模型动作维度，执行片段由 runner 截取。
    let output = policy
        .predict(preprocessor.run(raw, &context).unwrap())
        .unwrap()
        .take("action")
        .unwrap()
        .into_float3()
        .unwrap();
    assert_eq!(output.dims(), [1, cfg.chunk_size, cfg.max_action_dim]);
    assert!(
        output
            .into_data()
            .try_to_vec::<f32>()
            .unwrap()
            .iter()
            .all(|value| value.is_finite())
    );
}
