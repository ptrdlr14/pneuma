use std::{collections::BTreeMap, fs, path::PathBuf};

use burn::tensor::{Device, Tensor};

use super::*;

struct Directory(PathBuf);
impl Directory {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!("pneuma-processor-{}", uuid::Uuid::new_v4()));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn field(shape: &[usize]) -> Field {
    Field {
        shape: shape.iter().map(|n| Some(*n)).collect(),
        dtype: DType::F32,
        layout: Layout::Flat,
        optional: false,
    }
}

fn schema(shape: &[usize]) -> Schema {
    BTreeMap::from([("state".into(), field(shape))])
}

fn input(shape: &[usize], values: &[f32]) -> Batch {
    Batch {
        values: BTreeMap::from([("state".into(), Data::f32(values.to_vec(), shape.to_vec()))]),
        tasks: vec![],
    }
}

fn context(mode: Mode) -> Context {
    Context {
        device: Device::flex(),
        mode,
    }
}

fn values(batch: &Batch, key: &str) -> Vec<f32> {
    match batch.get(key).unwrap() {
        Data::Host(_) => batch.get(key).unwrap().host_f32().unwrap().to_vec(),
        Data::Float2(t) => t.to_data().try_to_vec().unwrap(),
        Data::Float3(t) => t.to_data().try_to_vec().unwrap(),
        Data::Float4(t) => t.to_data().try_to_vec().unwrap(),
        _ => panic!("expected float output"),
    }
}

#[test]
fn training_affine_and_linear_keep_gradients_on_broadcast_axes() {
    let device = Device::flex().autodiff();
    let x = Tensor::<3>::from_data([[[2.0, 4.0], [8.0, 12.0]]], &device).require_grad();
    let pipeline = Pipeline::new(
        schema(&[1, 2, 2]),
        vec![
            Step::Affine {
                key: "state".into(),
                offset: vec![0.0, 4.0],
                scale: vec![2.0, 4.0],
                axis: 1,
            },
            Step::Linear {
                key: "state".into(),
                offset: vec![1.0],
                scale: vec![3.0, 0.0],
                axis: 2,
            },
        ],
    )
    .unwrap();
    let batch = Batch {
        values: BTreeMap::from([("state".into(), Data::Float3(x.clone()))]),
        tasks: vec![],
    };
    let output = pipeline
        .run(
            batch,
            &Context {
                device,
                mode: Mode::Train,
            },
        )
        .unwrap()
        .take("state")
        .unwrap()
        .into_float3()
        .unwrap();
    assert_eq!(
        output.to_data().try_to_vec::<f32>().unwrap(),
        vec![4.0, 1.0, 4.0, 1.0]
    );
    let gradients = output.sum().backward();
    assert_eq!(
        x.grad(&gradients)
            .unwrap()
            .to_data()
            .try_to_vec::<f32>()
            .unwrap(),
        vec![1.5, 0.0, 0.75, 0.0]
    );
}

#[test]
fn inference_export_is_explicit_and_training_cannot_detach() {
    let pipeline = Pipeline::new(schema(&[1, 2]), vec![Step::Device, Step::Host]).unwrap();
    let batch = input(&[1, 2], &[2.0, 3.0]);
    assert!(pipeline.run(batch.clone(), &context(Mode::Train)).is_err());
    assert_eq!(
        values(&pipeline.run(batch, &context(Mode::Eval)).unwrap(), "state"),
        vec![2.0, 3.0]
    );
    let tracked = Tensor::<2>::ones([1, 2], &Device::flex().autodiff()).require_grad();
    let host = Pipeline::new(schema(&[1, 2]), vec![Step::Host]).unwrap();
    assert!(
        host.run(
            Batch {
                values: BTreeMap::from([("state".into(), Data::Float2(tracked))]),
                tasks: vec![]
            },
            &context(Mode::Eval)
        )
        .is_err()
    );
}

#[test]
fn single_sample_rgb_resize_and_normalized_padding_match_image_contract() {
    let image = Field {
        shape: vec![Some(1), Some(2), Some(3)],
        dtype: DType::U8,
        layout: Layout::Hwc,
        optional: false,
    };
    let pipeline = Pipeline::new(
        BTreeMap::from([("image".into(), image)]),
        vec![
            Step::Batch,
            Step::Rgb {
                key: "image".into(),
            },
            Step::ResizePad {
                key: "image".into(),
                width: 2,
                height: 2,
            },
            Step::Affine {
                key: "image".into(),
                offset: vec![0.5],
                scale: vec![0.5],
                axis: 1,
            },
            Step::Device,
        ],
    )
    .unwrap();
    let batch = Batch {
        values: BTreeMap::from([(
            "image".into(),
            Data::u8(vec![255, 0, 128, 0, 255, 64], vec![1, 2, 3]),
        )]),
        tasks: vec![],
    };
    let output = pipeline.run(batch, &context(Mode::Eval)).unwrap();
    assert_eq!(output.get("image").unwrap().shape(), vec![1, 3, 2, 2]);
    let actual = values(&output, "image");
    assert_eq!(
        &actual[..8],
        &[-1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0]
    );
    assert!((actual[10] - (128.0 / 255.0 * 2.0 - 1.0)).abs() < 1e-7);
    assert!((actual[11] - (64.0 / 255.0 * 2.0 - 1.0)).abs() < 1e-7);
    assert_eq!(pipeline.output_schema()["image"].layout, Layout::Chw);
}

#[test]
fn schema_and_dynamic_values_fail_before_invalid_operations() {
    assert!(
        Pipeline::new(
            schema(&[2]),
            vec![Step::Rgb {
                key: "state".into()
            }]
        )
        .is_err()
    );
    assert!(
        Pipeline::new(
            schema(&[1, 2]),
            vec![Step::Affine {
                key: "state".into(),
                offset: vec![0.0],
                scale: vec![0.0],
                axis: 1
            }]
        )
        .is_err()
    );
    assert!(
        Pipeline::new(
            schema(&[1, 2]),
            vec![Step::Affine {
                key: "state".into(),
                offset: vec![0.0; 3],
                scale: vec![1.0],
                axis: 1
            }]
        )
        .is_err()
    );
    let pipeline = Pipeline::new(schema(&[1, 2]), vec![Step::Device]).unwrap();
    // 长度与形状不符的数据在构造期就被拒绝，不再进入管线。
    assert!(Data::try_f32(vec![1.0], vec![1, 2]).is_err());
    assert!(
        pipeline
            .run(input(&[1, 2], &[1.0, f32::NAN]), &context(Mode::Eval))
            .is_err()
    );
    assert!(
        pipeline
            .run(input(&[2, 1], &[1.0, 2.0]), &context(Mode::Eval))
            .is_err()
    );
    let mut dynamic = field(&[1, 2]);
    dynamic.shape[1] = None;
    let dynamic = Pipeline::new(
        BTreeMap::from([("state".into(), dynamic)]),
        vec![Step::Affine {
            key: "state".into(),
            offset: vec![0.0; 3],
            scale: vec![1.0],
            axis: 1,
        }],
    )
    .unwrap();
    assert!(
        dynamic
            .run(input(&[1, 2], &[1.0, 2.0]), &context(Mode::Train))
            .is_err()
    );
}

#[test]
fn optional_actions_skip_only_when_schema_allows_and_rename_is_simultaneous() {
    let mut action = field(&[2]);
    action.optional = true;
    let schema = BTreeMap::from([("state".into(), field(&[2])), ("action".into(), action)]);
    let pipeline = Pipeline::new(
        schema,
        vec![
            Step::Rename {
                mapping: BTreeMap::from([("state".into(), "observation.state".into())]),
            },
            Step::Batch,
            Step::Affine {
                key: "action".into(),
                offset: vec![1.0],
                scale: vec![2.0],
                axis: 1,
            },
            Step::Pad {
                key: "observation.state".into(),
                width: 4,
            },
            Step::Device,
        ],
    )
    .unwrap();
    let output = pipeline
        .run(input(&[2], &[2.0, 3.0]), &context(Mode::Eval))
        .unwrap();
    assert_eq!(
        values(&output, "observation.state"),
        vec![2.0, 3.0, 0.0, 0.0]
    );
    assert!(pipeline.output_schema()["action"].optional);
    assert!(!output.values.contains_key("action"));
    assert!(
        pipeline
            .run(Batch::default(), &context(Mode::Eval))
            .is_err()
    );
    let swaps = Pipeline::new(
        BTreeMap::from([("a".into(), field(&[1])), ("b".into(), field(&[1]))]),
        vec![Step::Rename {
            mapping: BTreeMap::from([("a".into(), "b".into()), ("b".into(), "a".into())]),
        }],
    )
    .unwrap();
    let batch = Batch {
        values: BTreeMap::from([
            ("a".into(), Data::f32(vec![1.0], vec![1])),
            ("b".into(), Data::f32(vec![2.0], vec![1])),
        ]),
        tasks: vec![],
    };
    assert_eq!(
        values(&swaps.run(batch, &context(Mode::Eval)).unwrap(), "a"),
        vec![2.0]
    );
}

fn tokenizer_fixture(directory: &Directory) {
    fs::write(directory.0.join("tokenizer.json"), serde_json::to_vec(&serde_json::json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": [], "normalizer": null,
        "pre_tokenizer": { "type": "Whitespace" }, "post_processor": null, "decoder": null,
        "model": { "type": "WordLevel", "vocab": { "[UNK]": 0, "[PAD]": 1, "pick": 2, "cube": 3, "now": 4 }, "unk_token": "[UNK]" }
    })).unwrap()).unwrap();
    fs::write(directory.0.join("tokenizer_config.json"), br#"{"pad_token":{"content":"[PAD]"},"pad_token_id":1,"truncation_side":"left","padding_side":"right"}"#).unwrap();
}

#[test]
fn saved_pipeline_is_relocatable_and_validates_versions_and_resource_digests() {
    let source = Directory::new();
    tokenizer_fixture(&source);
    let artifact = Directory::new();
    let pipeline = Pipeline::new(
        schema(&[2, 2]),
        vec![
            Step::Affine {
                key: "state".into(),
                offset: vec![1.0, 2.0],
                scale: vec![2.0, 4.0],
                axis: 1,
            },
            Step::Linear {
                key: "state".into(),
                offset: vec![0.0],
                scale: vec![0.0, 3.0],
                axis: 1,
            },
            Step::Newline,
            Step::Tokenize {
                tokenizer: source.0.clone(),
                max_length: 2,
                padding: "max_length".into(),
                padding_side: "right".into(),
                special_tokens: true,
                negative: None,
            },
            Step::Device,
        ],
    )
    .unwrap();
    let mut batch = input(&[2, 2], &[3.0, 6.0, 5.0, 10.0]);
    batch.tasks = vec!["pick cube now".into(), "cube".into()];
    let expected = pipeline.run(batch.clone(), &context(Mode::Eval)).unwrap();
    // 源资源失效后仍保存构造时的完整快照。
    drop(source);
    let saved = artifact.0.join("saved");
    pipeline.save(&saved).unwrap();
    assert!(pipeline.save(&saved).is_err(), "artifacts are immutable");
    let relocated = artifact.0.join("relocated");
    fs::rename(&saved, &relocated).unwrap();
    drop(pipeline);
    let loaded = Pipeline::load(&relocated).unwrap();
    let actual = loaded.run(batch.clone(), &context(Mode::Train)).unwrap();
    assert_eq!(values(&actual, "state"), values(&expected, "state"));
    let Data::Int2(ids) = actual.get("observation.language.tokens").unwrap() else {
        panic!("missing tokens")
    };
    assert_eq!(
        ids.to_data().iter::<i64>().collect::<Vec<_>>(),
        vec![3, 4, 3, 1]
    );
    let Data::Bool2(mask) = actual.get("observation.language.attention_mask").unwrap() else {
        panic!("missing mask")
    };
    assert_eq!(
        mask.to_data().try_to_vec::<bool>().unwrap(),
        vec![true, true, true, false]
    );
    assert_eq!(actual.tasks, vec!["pick cube now\n", "cube\n"]);
    let manifest_path = relocated.join("pipeline.json");
    let original = fs::read(&manifest_path).unwrap();
    let mut manifest: serde_json::Value = serde_json::from_slice(&original).unwrap();
    manifest["steps"][0]["version"] = 99.into();
    fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    assert!(Pipeline::load(&relocated).is_err());
    fs::write(&manifest_path, original).unwrap();
    fs::write(relocated.join("statistics.safetensors"), b"corrupted").unwrap();
    assert!(Pipeline::load(&relocated).is_err());
}

#[test]
fn tokenizer_configuration_is_checked_at_construction() {
    let source = Directory::new();
    tokenizer_fixture(&source);
    fs::write(
        source.0.join("tokenizer_config.json"),
        br#"{"pad_token":"[PAD]","pad_token_id":9}"#,
    )
    .unwrap();
    assert!(
        Pipeline::new(
            Schema::new(),
            vec![Step::Tokenize {
                tokenizer: source.0.clone(),
                max_length: 2,
                padding: "longest".into(),
                padding_side: "right".into(),
                special_tokens: true,
                negative: None,
            }]
        )
        .is_err()
    );
}

/// Cosmos3 画布步骤与官方预处理结果对齐（最多 1 个灰阶）。
#[test]
#[ignore = "需要官方画布参照"]
fn cosmos3_canvas_matches_official() {
    use burn::tensor::{Tensor, TensorData};

    let parity = std::path::PathBuf::from(
        std::env::var("PNEUMA_COSMOS3_COMPOSE").expect("PNEUMA_COSMOS3_COMPOSE"),
    );
    let read = |name: &str| -> Vec<f32> {
        let bytes = std::fs::read(parity.join(name)).expect("reading reference");
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|value| f32::from_le_bytes(*value))
            .collect()
    };
    let context = context(Mode::Eval);
    let device = context.device.clone();
    let mut schema = Schema::new();
    for key in ["a", "b"] {
        schema.insert(
            key.to_string(),
            Field {
                shape: vec![Some(1), Some(3), Some(256), Some(256)],
                dtype: DType::F32,
                layout: Layout::Flat,
                optional: false,
            },
        );
    }
    let pipeline = Pipeline::new(
        schema,
        vec![Step::Cosmos3Canvas {
            cameras: vec!["a".into(), "b".into()],
            view_width: 256,
            view_height: 256,
            output: "observation.images.canvas".into(),
        }],
    )
    .expect("pipeline");
    let batch = Batch {
        values: BTreeMap::from([
            (
                "a".to_string(),
                Data::Float4(Tensor::<4>::from_data(
                    TensorData::new(read("first.bin"), [1, 3, 256, 256]),
                    &device,
                )),
            ),
            (
                "b".to_string(),
                Data::Float4(Tensor::<4>::from_data(
                    TensorData::new(read("second.bin"), [1, 3, 256, 256]),
                    &device,
                )),
            ),
        ]),
        tasks: vec!["task".into()],
    };
    let batch = pipeline.run(batch, &context).expect("run");
    let canvas = match batch.values.get("observation.images.canvas") {
        Some(Data::Float4(canvas)) => canvas.clone(),
        other => panic!("unexpected output {:?}", other.map(Data::shape)),
    };
    assert_eq!(canvas.dims(), [1, 3, 192, 320]);
    let got: Vec<f32> = canvas.into_data().iter::<f32>().collect();
    let expected = read("canvas.bin");
    let levels: Vec<f32> = got
        .iter()
        .zip(&expected)
        .map(|(a, b)| ((a - b).abs() * 127.5).round())
        .collect();
    let max = levels.iter().copied().fold(0.0f32, f32::max);
    let differing = levels.iter().filter(|level| **level > 0.0).count();
    println!("pipeline canvas levels max {max} differing {differing}");
    assert!(max <= 1.0, "canvas differs by {max} levels");
    assert!(
        differing * 100 < levels.len(),
        "too many differing pixels: {differing}"
    );
}
