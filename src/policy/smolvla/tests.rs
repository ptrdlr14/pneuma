//! 真实 checkpoint 数值对齐需要先运行 dump_smolvla_reference.py。

use super::*;

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
            crate::protocol::FeatureSpec {
                name: name.clone(),
                shape,
                dtype: if visual { "uint8" } else { "float32" }.into(),
                kind: if visual { "rgb" } else { "tensor" }.into(),
                names: vec![],
                semantics: "parity".into(),
            }
        })
        .collect();
    let action = crate::protocol::FeatureSpec {
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
    let policy = SmolVlaPolicy::load_configured(
        &dir.join("checkpoint"),
        Some(&vlm),
        backend,
        0,
        &features,
        &action,
    )
    .unwrap();
    let raw_state = read_f32(&dir, &manifest, "raw_state");
    let state = policy.processor.state(&raw_state, &policy.device);
    compare(
        "state_padded",
        &state.clone().into_data().try_to_vec::<f32>().unwrap(),
        &read_f32(&dir, &manifest, "state_padded"),
        1e-6,
    );
    let task = manifest["task"].as_str().unwrap_or("pick up the red cube");
    let (tokens, valid) = policy.processor.tokenize(task).unwrap();
    assert_eq!(
        tokens.iter().map(|&x| x as f32).collect::<Vec<_>>(),
        read_f32(&dir, &manifest, "tokens")
    );
    assert_eq!(
        valid
            .iter()
            .map(|&x| u8::from(x) as f32)
            .collect::<Vec<_>>(),
        read_f32(&dir, &manifest, "token_mask")
    );
    let mut images = Vec::new();
    let mut feature_records = vec![(
        Value::from("observation.state"),
        codec::tensor_record(
            "float32",
            &[cfg.state_dim() as u64],
            raw_state.iter().flat_map(|v| v.to_le_bytes()).collect(),
        ),
    )];
    for (index, (name, feature)) in cfg.image_features().enumerate() {
        let raw: Vec<_> = read_f32(&dir, &manifest, &format!("raw_image_{index}"))
            .iter()
            .map(|&v| v as u8)
            .collect();
        let (h, w) = (feature.shape[1], feature.shape[2]);
        let image = policy.processor.image(&raw, h, w, &policy.device);
        compare(
            &format!("image_prepared_{index}"),
            &image.clone().into_data().try_to_vec::<f32>().unwrap(),
            &read_f32(&dir, &manifest, &format!("image_prepared_{index}")),
            5e-6,
        );
        let record = crate::protocol::map([
            (codec::MARKER, "rgb".into()),
            ("dtype", "uint8".into()),
            ("shape", Value::Array(vec![h.into(), w.into(), 3.into()])),
            ("channel_order", "RGB".into()),
            ("encoding", "raw".into()),
            ("data", Value::Binary(raw)),
        ]);
        feature_records.push((name.into(), record));
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
    let trimmed: Vec<_> = output
        .chunks_exact(cfg.max_action_dim)
        .flat_map(|row| row[..cfg.action_dim()].iter().copied())
        .collect();
    compare(
        "actions_canonical",
        &policy.processor.unnormalize(&trimmed),
        &read_f32(&dir, &manifest, "actions_canonical"),
        2e-3,
    );
    let caps = Capabilities {
        features,
        action_feature: action,
        execution_steps: 3,
        prediction_steps: cfg.chunk_size as u64,
        ..test_capabilities()
    };
    let chunk = policy
        .predict(&caps, &Value::Map(feature_records), Some(task))
        .unwrap();
    assert_eq!(chunk.execution_steps, 3);
    assert_eq!(chunk.canonical_actions.len(), 3 * cfg.action_dim() * 4);
    assert!(
        chunk
            .canonical_actions
            .as_chunks::<4>()
            .0
            .iter()
            .all(|v| f32::from_le_bytes(*v).is_finite())
    );
}

fn test_capabilities() -> Capabilities {
    crate::config::ServerConfig::load(Path::new("examples/so101.toml"))
        .unwrap()
        .capabilities(50, 3)
}
