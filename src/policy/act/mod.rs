//! Burn 原生 ACT：部署加载 bpk，LeRobot 参数布局只在转换边界适配，
//! 旧 checkpoint 的归一化缓冲在导入时转换为共享输入 / 输出 pipeline。
//! 推理路径 `use_vae` 下 z=0（eval 语义），`vae_encoder` 权重不加载。

pub mod config;
pub mod position;
pub mod resnet;
pub mod transformer;

use burn::module::Module;
use burn::nn::conv::Conv2d;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{Device, DeviceKind, Tensor, TensorData};

use crate::contract::FeatureSpec;
use crate::policy::Policy;
use crate::processor::{Batch, DType, Data, Field, Layout, Pipeline, Schema, Step};

use config::ActConfig;
use position::sinusoidal_2d;
use resnet::{Resnet, ResnetConfig};
use transformer::{ACTDecoder, ACTEncoder, LayerNorm, TransformerLayerConfig};

/// nn.Embedding 的只读等价：weight [n, dim] 直接给 (1, n, dim) 位置编码用。
#[derive(Module, Debug)]
pub struct PosTable {
    pub weight: burn::module::Param<burn::tensor::Tensor<2>>,
}

impl PosTable {
    fn init(n: usize, dim: usize, device: &Device) -> Self {
        Self {
            weight: burn::module::Param::from_data(
                TensorData::new(vec![0.0f32; n * dim], [n, dim]),
                device,
            ),
        }
    }

    fn rows(&self) -> Tensor<3> {
        let [n, dim] = self.weight.dims();
        self.weight.val().reshape([1, n, dim])
    }
}

/// ACT 顶层模块；字段名 = `modeling_act.py` 的 nn.Module 属性名。
#[derive(Module, Debug)]
pub struct Act {
    pub backbone: Resnet,
    pub encoder: ACTEncoder,
    pub decoder: ACTDecoder,
    pub encoder_robot_state_input_proj: Linear,
    pub encoder_latent_input_proj: Linear,
    pub encoder_img_feat_input_proj: Conv2d,
    pub encoder_1d_feature_pos_embed: PosTable,
    pub decoder_pos_embed: PosTable,
    pub action_head: Linear,
}

impl ActConfig {
    pub fn init(&self, device: &Device) -> Act {
        let layer_cfg = TransformerLayerConfig {
            dim_model: self.dim_model,
            n_heads: self.n_heads,
            dim_feedforward: self.dim_feedforward,
        };
        Act {
            backbone: ResnetConfig.init(device),
            encoder: ACTEncoder {
                layers: (0..self.n_encoder_layers)
                    .map(|_| layer_cfg.init_encoder_layer(device))
                    .collect(),
            },
            decoder: ACTDecoder {
                layers: (0..self.n_decoder_layers)
                    .map(|_| layer_cfg.init_decoder_layer(device))
                    .collect(),
                norm: LayerNorm::init(self.dim_model, device),
            },
            encoder_robot_state_input_proj: LinearConfig::new(14, self.dim_model).init(device),
            encoder_latent_input_proj: LinearConfig::new(self.latent_dim, self.dim_model)
                .init(device),
            // resnet18 layer4 输出通道固定 512。
            encoder_img_feat_input_proj: burn::nn::conv::Conv2dConfig::new(
                [512, self.dim_model],
                [1, 1],
            )
            .init(device),
            encoder_1d_feature_pos_embed: PosTable::init(2, self.dim_model, device),
            decoder_pos_embed: PosTable::init(self.chunk_size, self.dim_model, device),
            action_head: LinearConfig::new(self.dim_model, 14).init(device),
        }
    }
}

impl Act {
    /// state (1,14) 与 image (1,3,H,W) 均已归一化；返回 `(1, chunk_size, 14)` 归一化动作。
    pub fn forward(&self, state: Tensor<2>, image: Tensor<4>, latent_dim: usize) -> Tensor<3> {
        let device = state.device();
        let feat = self.backbone.forward(image);
        let (tokens, pos) = self.encoder_inputs(state, feat, latent_dim);
        let dim_model = tokens.dims()[2];
        let enc = self.encoder.forward(tokens, &pos);
        let chunk = self.decoder_pos_embed.rows().dims()[1];
        let dec_in = Tensor::<3, _>::zeros([1, chunk, dim_model], &device);
        let dec_pos = self.decoder_pos_embed.rows();
        let dec = self.decoder.forward(dec_in, &dec_pos, &enc, &pos);
        self.action_head.forward(dec)
    }
    fn encoder_inputs(
        &self,
        state: Tensor<2>,
        feat: Tensor<4>,
        latent_dim: usize,
    ) -> (Tensor<3>, Tensor<3>) {
        let device = state.device();
        let latent = Tensor::<2, _>::zeros([1, latent_dim], &device);
        // 两个投影都映到 dim_model，token 维度从投影输出取。
        let latent_tok = self.encoder_latent_input_proj.forward(latent);
        let state_tok = self.encoder_robot_state_input_proj.forward(state);
        let dim_model = latent_tok.dims()[1];
        let mut tokens = vec![
            latent_tok.reshape([1, 1, dim_model]),
            state_tok.reshape([1, 1, dim_model]),
        ];
        let feat = self.encoder_img_feat_input_proj.forward(feat);
        let [_, dim, h, w] = feat.dims();
        tokens.push(feat.permute([0, 2, 3, 1]).reshape([1, h * w, dim]));
        let tokens = Tensor::cat(tokens, 1);

        let mut pos = vec![self.encoder_1d_feature_pos_embed.rows()];
        pos.push(sinusoidal_2d(&device, h, w, dim / 2).reshape([1, h * w, dim]));
        let pos = Tensor::cat(pos, 1);

        (tokens, pos)
    }
}

/// checkpoint 自带的归一化缓冲（`normalize_inputs.*` / `unnormalize_outputs.*`）。
struct NormStats {
    image_mean: [f32; 3],
    image_std: [f32; 3],
    state_mean: Vec<f32>,
    state_std: Vec<f32>,
    action_mean: Vec<f32>,
    action_std: Vec<f32>,
}

impl NormStats {
    fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        // 归一化缓冲单独从同一文件读取（不属于模块树）。
        use safetensors::SafeTensors;
        let bytes = std::fs::read(path)?;
        let st = SafeTensors::deserialize(&bytes)?;
        let image_mean: [f32; 3] =
            f32_tensor(&st, "normalize_inputs.buffer_observation_images_top.mean")?
                .try_into()
                .map_err(|_| anyhow::anyhow!("image mean must have 3 channels"))?;
        let image_std: [f32; 3] =
            f32_tensor(&st, "normalize_inputs.buffer_observation_images_top.std")?
                .try_into()
                .map_err(|_| anyhow::anyhow!("image std must have 3 channels"))?;
        let stats = Self {
            image_mean,
            image_std,
            state_mean: f32_tensor(&st, "normalize_inputs.buffer_observation_state.mean")?,
            state_std: f32_tensor(&st, "normalize_inputs.buffer_observation_state.std")?,
            action_mean: f32_tensor(&st, "unnormalize_outputs.buffer_action.mean")?,
            action_std: f32_tensor(&st, "unnormalize_outputs.buffer_action.std")?,
        };
        stats.validate()?;
        Ok(stats)
    }

    fn validate(&self) -> anyhow::Result<()> {
        for (name, values, width, standard_deviation) in [
            ("image_mean", self.image_mean.as_slice(), 3, false),
            ("image_std", self.image_std.as_slice(), 3, true),
            ("state_mean", self.state_mean.as_slice(), 14, false),
            ("state_std", self.state_std.as_slice(), 14, true),
            ("action_mean", self.action_mean.as_slice(), 14, false),
            ("action_std", self.action_std.as_slice(), 14, true),
        ] {
            anyhow::ensure!(
                values.len() == width,
                "{name} must contain {width} values, got {}",
                values.len()
            );
            anyhow::ensure!(
                values
                    .iter()
                    .all(|v| v.is_finite() && (!standard_deviation || *v > 0.0)),
                "{name} must contain finite {}values",
                if standard_deviation { "positive " } else { "" }
            );
        }
        Ok(())
    }

    fn into_pipelines(self, chunk_size: usize) -> anyhow::Result<(Pipeline, Pipeline)> {
        self.validate()?;
        let input: Schema = [
            (
                "observation.state".into(),
                Field {
                    shape: vec![Some(14)],
                    dtype: DType::F32,
                    layout: Layout::Flat,
                    optional: false,
                },
            ),
            (
                "observation.images.top".into(),
                Field {
                    shape: vec![Some(480), Some(640), Some(3)],
                    dtype: DType::U8,
                    layout: Layout::Hwc,
                    optional: false,
                },
            ),
            (
                "action".into(),
                Field {
                    shape: vec![None, Some(14)],
                    dtype: DType::F32,
                    layout: Layout::Flat,
                    optional: true,
                },
            ),
        ]
        .into();
        let pre = Pipeline::new(
            input,
            vec![
                Step::Batch,
                Step::Rgb {
                    key: "observation.images.top".into(),
                },
                Step::Affine {
                    key: "observation.images.top".into(),
                    offset: self.image_mean.to_vec(),
                    scale: self.image_std.to_vec(),
                    axis: 1,
                },
                Step::Affine {
                    key: "observation.state".into(),
                    offset: self.state_mean,
                    scale: self.state_std,
                    axis: 1,
                },
                Step::Affine {
                    key: "action".into(),
                    offset: self.action_mean.clone(),
                    scale: self.action_std.clone(),
                    axis: 2,
                },
                Step::Device,
            ],
        )?;
        let output: Schema = [(
            "action".into(),
            Field {
                shape: vec![None, Some(chunk_size), Some(14)],
                dtype: DType::F32,
                layout: Layout::Flat,
                optional: false,
            },
        )]
        .into();
        let post = Pipeline::new(
            output,
            vec![Step::Linear {
                key: "action".into(),
                offset: self.action_mean,
                scale: self.action_std,
                axis: 2,
            }],
        )?;
        Ok((pre, post))
    }
}

pub struct ActPolicy {
    model: Act,
    cfg: ActConfig,
    device: Device,
}

fn f32_tensor(st: &safetensors::SafeTensors, name: &str) -> anyhow::Result<Vec<f32>> {
    let view = st.tensor(name)?;
    if view.dtype() != safetensors::Dtype::F32 {
        return Err(anyhow::anyhow!("{name}: 期望 f32，实际 {:?}", view.dtype()));
    }
    Ok(view
        .data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect())
}

impl ActPolicy {
    /// 导入 LeRobot checkpoint 的模型与共享 pipeline；预热由部署执行器负责。
    pub fn import_configured(
        path: &std::path::Path,
        backend: crate::config::Device,
    ) -> anyhow::Result<(Self, Pipeline, Pipeline, Vec<String>)> {
        let stats = NormStats::load(&path.join("model.safetensors"))?;
        let (policy, unused) = Self::import_lerobot(path, backend)?;
        let (pre, post) = stats.into_pipelines(policy.cfg.chunk_size)?;
        Ok((policy, pre, post, unused))
    }

    /// 从原生 checkpoint 严格恢复模型状态，不接受外部权重格式。
    pub fn load_model(
        path: &std::path::Path,
        backend: crate::config::Device,
    ) -> anyhow::Result<Self> {
        use anyhow::Context;
        let mut policy = Self::init_model(path, backend)?;
        let record = burn::store::ModuleRecord::load(path.join("model.bpk"))
            .context("reading native ACT model.bpk; convert the LeRobot checkpoint first")?;
        policy.model = policy
            .model
            .try_load_record(record)
            .context("restoring ACT model record")?;
        Ok(policy)
    }

    fn init_model(path: &std::path::Path, backend: crate::config::Device) -> anyhow::Result<Self> {
        let cfg = config::parse(&path.join("config.json"))?;
        let device = match backend {
            crate::config::Device::Auto => Device::default(),
            crate::config::Device::Flex => Device::flex(),
            crate::config::Device::Metal => Device::metal(DeviceKind::default()),
        };
        let model = cfg.init(&device);
        Ok(Self { model, cfg, device })
    }

    /// 转换边界导入完整推理模块，额外参数由 checkpoint 转换器保存。
    pub fn import_lerobot(
        path: &std::path::Path,
        backend: crate::config::Device,
    ) -> anyhow::Result<(Self, Vec<String>)> {
        use anyhow::Context;
        use burn_store::ModuleSnapshot;
        let mut policy = Self::init_model(path, backend)?;
        let dir = path.display();

        let mut store = lerobot_import_store(&path.join("model.safetensors"));
        let applied = policy
            .model
            .load_from(&mut store)
            .with_context(|| format!("applying weights from {dir}/model.safetensors"))?;
        // allow_partial 只放行 vae_encoder/归一化缓冲（unused 侧）；
        // 模块树自身的键必须全数命中，否则权重静默留在随机初值。
        if !applied.missing.is_empty() || !applied.errors.is_empty() {
            return Err(anyhow::anyhow!(
                "权重加载不完整 missing={:?} errors={:?}",
                applied.missing,
                applied.errors
            ));
        }
        tracing::debug!(
            "applied={} unused={}",
            applied.applied.len(),
            applied.unused.len()
        );

        Ok((policy, applied.unused))
    }

    /// 保存当前内存参数与参数身份，供原生加载和后续训练恢复使用。
    pub fn save_native(&self, path: &std::path::Path) -> anyhow::Result<()> {
        self.model.clone().into_record().save(path)?;
        Ok(())
    }

    /// 导出当前推理模块的 PyTorch 参数名和布局；附属参数由转换器合并。
    pub fn export_lerobot(&self, path: &std::path::Path) -> anyhow::Result<()> {
        use burn_store::ModuleSnapshot;
        let mut store = lerobot_export_store(path);
        self.model.save_into(&mut store)?;
        Ok(())
    }

    pub fn horizon(&self) -> (u64, u64) {
        (self.cfg.chunk_size as u64, self.cfg.n_action_steps as u64)
    }

    /// 返回模型所在设备，供执行器构造 pipeline 上下文。
    pub fn device(&self) -> Device {
        self.device.clone()
    }
}

fn lerobot_import_store(path: &std::path::Path) -> burn_store::SafetensorsStore {
    use burn_store::ModuleAdapter;
    burn_store::SafetensorsStore::from_file(path)
        .with_from_adapter(
            burn_store::PyTorchToBurnAdapter
                .chain(burn_store::FloatCastAdapter::to(burn::tensor::DType::F32)),
        )
        .with_key_remapping(r"^model\.", "")
        .with_key_remapping(r"(^|\.)in_proj_weight$", "${1}in_proj.weight")
        .with_key_remapping(r"(^|\.)in_proj_bias$", "${1}in_proj.bias")
        .with_key_remapping(r"\.downsample\.0\.", ".downsample.conv.")
        .with_key_remapping(r"\.downsample\.1\.", ".downsample.norm.")
        .allow_partial(true)
}

fn lerobot_export_store(path: &std::path::Path) -> burn_store::SafetensorsStore {
    burn_store::SafetensorsStore::from_file(path)
        .with_to_adapter(burn_store::BurnToPyTorchAdapter)
        .with_key_remapping(r"(^|\.)in_proj\.weight$", "${1}in_proj_weight")
        .with_key_remapping(r"(^|\.)in_proj\.bias$", "${1}in_proj_bias")
        .with_key_remapping(r"\.downsample\.conv\.", ".downsample.0.")
        .with_key_remapping(r"\.downsample\.norm\.", ".downsample.1.")
        .with_key_remapping(r"^", "model.")
}

/// 合约维度 × 权重维度的纯比对：错位是 profile 与 checkpoint 的配置错误，
/// 启动即报错好过一次 matmul panic；提出为纯函数以便用 profile 单测。
pub fn validate_contract(features: &[FeatureSpec], action: &FeatureSpec) -> anyhow::Result<()> {
    let state = features.iter().find(|f| f.name == "observation.state");
    let image = features.iter().find(|f| f.name == "observation.images.top");
    if features.len() != 2
        || !state.is_some_and(|f| f.kind == "tensor" && f.dtype == "float32" && f.shape == [14])
        || !image.is_some_and(|f| f.kind == "rgb" && f.dtype == "uint8" && f.shape == [480, 640, 3])
        || action.name != "action"
        || action.kind != "tensor"
        || action.dtype != "float32"
        || action.shape != [14]
    {
        anyhow::bail!(
            "ACT requires observation.state [14], observation.images.top [480,640,3], and action [14]"
        );
    }
    Ok(())
}

#[test]
fn profile_mismatched_with_checkpoint_is_startup_error() {
    // so101 合约 6 维、官方 ACT checkpoint 14 维：装载期就要拒绝，
    // 否则首次观测在 Burn matmul 里 panic、拖垮整个 server。
    use crate::config::ServerConfig;
    let caps = ServerConfig::load(std::path::Path::new("examples/so101.toml"))
        .unwrap()
        .capabilities(50, 16);
    let error = validate_contract(&caps.features, &caps.action_feature).unwrap_err();
    assert!(error.to_string().contains("ACT requires"), "{error}");
    let mut cfg = ServerConfig::load(std::path::Path::new("examples/aloha_sim.toml")).unwrap();
    assert!(validate_contract(&cfg.features, &cfg.action_feature).is_ok());
    cfg.features[1].shape = vec![240, 320, 3];
    cfg.model.repo_or_path = "/nonexistent/checkpoint".into();
    let Err(error) = crate::deployment::build(&cfg) else {
        panic!("unsupported ACT image contract accepted");
    };
    assert!(error.to_string().contains("ACT requires"), "{error}");
}

impl Policy for ActPolicy {
    fn predict(&self, mut batch: Batch) -> anyhow::Result<Batch> {
        let state = batch.take("observation.state")?.into_float2()?;
        let image = batch.take("observation.images.top")?.into_float4()?;
        // 旧 checkpoint 导入器补单样本 batch 轴；当前 ACT 网络使用固定 B=1。
        anyhow::ensure!(state.dims() == [1, 14], "ACT requires state [1,14]");
        anyhow::ensure!(
            image.dims() == [1, 3, 480, 640],
            "ACT requires image [1,3,480,640]"
        );
        let action = self.model.forward(state, image, self.cfg.latent_dim);
        Ok(Batch {
            values: [("action".into(), Data::Float3(action))].into(),
            tasks: batch.tasks,
        })
    }
}

#[cfg(test)]
fn observation_batch(state: &[f32], image: &[u8]) -> Batch {
    Batch {
        values: [
            (
                "observation.state".into(),
                Data::F32 {
                    values: state.to_vec(),
                    shape: vec![14],
                },
            ),
            (
                "observation.images.top".into(),
                Data::U8 {
                    values: image.to_vec(),
                    shape: vec![480, 640, 3],
                },
            ),
        ]
        .into(),
        ..Batch::default()
    }
}

#[cfg(test)]
mod numerical_tests {
    use super::*;
    use crate::processor::{Context, Mode};

    #[test]
    fn native_weights_export_with_original_attention_keys_and_layout() {
        use burn_store::ModuleSnapshot;
        #[derive(Module, Debug)]
        struct Layer {
            in_proj: Linear,
        }
        let device = Device::flex();
        let model = Layer {
            in_proj: Linear {
                weight: burn::module::Param::from_data(
                    TensorData::new(vec![1f32, 2., 3., 4., 5., 6.], [2, 3]),
                    &device,
                ),
                bias: Some(burn::module::Param::from_data(
                    TensorData::new(vec![7f32, 8., 9.], [3]),
                    &device,
                )),
            },
        };
        let directory = std::env::temp_dir().join(format!("act-convert-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&directory).unwrap();
        model
            .clone()
            .into_record()
            .save(directory.join("model.bpk"))
            .unwrap();
        let record = burn::store::ModuleRecord::load(directory.join("model.bpk")).unwrap();
        let fresh = Layer {
            in_proj: LinearConfig::new(2, 3).init(&device),
        };
        let loaded = fresh.try_load_record(record).unwrap();
        assert_eq!(loaded.in_proj.weight.id, model.in_proj.weight.id);
        let path = directory.join("active.safetensors");
        loaded.save_into(&mut lerobot_export_store(&path)).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let tensors = safetensors::SafeTensors::deserialize(&bytes).unwrap();
        let weight = tensors.tensor("model.in_proj_weight").unwrap();
        assert_eq!(weight.shape(), [3, 2]);
        let values: Vec<_> = weight
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| f32::from_le_bytes(*v))
            .collect();
        assert_eq!(values, [1., 4., 2., 5., 3., 6.]);
        assert!(tensors.tensor("model.in_proj_bias").is_ok());
        let mut restored = Layer {
            in_proj: LinearConfig::new(2, 3).init(&device),
        };
        let applied = restored
            .load_from(&mut lerobot_import_store(&path))
            .unwrap();
        assert!(
            applied.missing.is_empty() && applied.errors.is_empty() && applied.unused.is_empty(),
            "{applied:?}"
        );
        assert_eq!(
            restored
                .in_proj
                .weight
                .val()
                .into_data()
                .try_to_vec::<f32>()
                .unwrap(),
            [1., 2., 3., 4., 5., 6.]
        );
        std::fs::remove_dir_all(directory).unwrap();
    }

    fn stats() -> NormStats {
        NormStats {
            image_mean: [0.1, 0.2, 0.3],
            image_std: [0.5, 0.25, 0.2],
            state_mean: vec![1.0; 14],
            state_std: vec![2.0; 14],
            action_mean: (0..14).map(|i| f32::from(i as u8)).collect(),
            action_std: vec![2.0; 14],
        }
    }

    #[test]
    fn preprocessing_and_actions_follow_checkpoint_statistics() {
        let stats = stats();
        stats.validate().unwrap();
        let mut image = vec![0; 480 * 640 * 3];
        image[..6].copy_from_slice(&[255, 128, 0, 64, 32, 255]);
        let (pre, post) = self::stats().into_pipelines(3).unwrap();
        let context = Context {
            device: Device::flex(),
            mode: Mode::Eval,
        };
        let mut batch = pre
            .run(observation_batch(&[5.0; 14], &image), &context)
            .unwrap();
        assert!(
            batch.get("action").is_err(),
            "optional action must stay absent"
        );
        let state = batch
            .take("observation.state")
            .unwrap()
            .into_float2()
            .unwrap();
        assert_eq!(state.dims(), [1, 14]);
        assert_eq!(
            state.into_data().try_to_vec::<f32>().unwrap(),
            vec![2.0; 14]
        );
        let image = batch
            .take("observation.images.top")
            .unwrap()
            .into_float4()
            .unwrap();
        assert_eq!(image.dims(), [1, 3, 480, 640]);
        let image = image.into_data().try_to_vec::<f32>().unwrap();
        for (channel, pixels) in [[255u8, 64], [128, 32], [0, 255]].iter().enumerate() {
            for (pixel, value) in pixels.iter().enumerate() {
                let expected = (f32::from(*value) / 255.0 - stats.image_mean[channel])
                    / stats.image_std[channel];
                assert!((image[channel * 480 * 640 + pixel] - expected).abs() < 1e-6);
            }
        }
        let mut output = post
            .run(
                Batch {
                    values: [(
                        "action".into(),
                        Data::Float3(Tensor::<3>::ones([1, 3, 14], &context.device)),
                    )]
                    .into(),
                    ..Batch::default()
                },
                &context,
            )
            .unwrap();
        let actions = output.take("action").unwrap().into_float3().unwrap();
        assert_eq!(
            actions.dims(),
            [1, 3, 14],
            "pipeline preserves the full prediction horizon"
        );
        let values = actions.into_data().try_to_vec::<f32>().unwrap();
        assert_eq!(values.len(), 42);
        for (i, value) in values.iter().enumerate() {
            assert_eq!(*value, stats.action_mean[i % 14] + 2.0);
        }
    }

    #[test]
    fn training_targets_use_the_same_statistics_and_roundtrip_to_canonical_actions() {
        let (pre, post) = stats().into_pipelines(3).unwrap();
        let context = Context {
            device: Device::flex(),
            mode: Mode::Train,
        };
        let canonical: Vec<f32> = (0..42).map(|i| (i % 14) as f32 + 6.0).collect();
        let mut batch = observation_batch(&[5.0; 14], &vec![0; 480 * 640 * 3]);
        batch.values.insert(
            "action".into(),
            Data::F32 {
                values: canonical.clone(),
                shape: vec![3, 14],
            },
        );
        let mut normalized = pre.run(batch, &context).unwrap();
        let action = normalized.take("action").unwrap().into_float3().unwrap();
        assert_eq!(action.dims(), [1, 3, 14]);
        assert_eq!(
            action.clone().into_data().try_to_vec::<f32>().unwrap(),
            vec![3.0; 42]
        );
        let mut restored = post
            .run(
                Batch {
                    values: [("action".into(), Data::Float3(action))].into(),
                    ..Batch::default()
                },
                &context,
            )
            .unwrap();
        assert_eq!(
            restored
                .take("action")
                .unwrap()
                .into_float3()
                .unwrap()
                .into_data()
                .try_to_vec::<f32>()
                .unwrap(),
            canonical
        );
    }

    #[test]
    fn unsupported_feature_contracts_fail_before_checkpoint_io() {
        use crate::config::ServerConfig;
        let profile = std::path::Path::new("examples/aloha_sim.toml");
        let valid = ServerConfig::load(profile).unwrap();
        assert!(validate_contract(&valid.features, &valid.action_feature).is_ok());
        for case in 0..15 {
            let mut cfg = ServerConfig::load(profile).unwrap();
            cfg.model.repo_or_path = "/nonexistent/checkpoint".into();
            match case {
                0 => cfg.features[0].name = "observation.other".into(),
                1 => cfg.features[0].kind = "rgb".into(),
                2 => cfg.features[0].dtype = "float64".into(),
                3 => cfg.features[0].shape = vec![6],
                4 => cfg.features[1].name = "observation.images.front".into(),
                5 => cfg.features[1].kind = "tensor".into(),
                6 => cfg.features[1].dtype = "float32".into(),
                7 => cfg.features[1].shape = vec![240, 320, 3],
                8 => cfg.action_feature.name = "other".into(),
                9 => cfg.action_feature.kind = "rgb".into(),
                10 => cfg.action_feature.dtype = "float64".into(),
                11 => cfg.action_feature.shape = vec![6],
                12 => {
                    cfg.features.pop();
                }
                13 => cfg.features.push(cfg.features[1].clone()),
                _ => {
                    let mut extra = cfg.features[0].clone();
                    extra.name = "observation.extra".into();
                    cfg.features.push(extra);
                }
            }
            let error = match crate::deployment::build(&cfg) {
                Ok(_) => panic!("unsupported ACT contract accepted: case {case}"),
                Err(error) => error,
            };
            assert!(
                error.to_string().contains("ACT requires"),
                "case {case}: {error}"
            );
        }
    }

    #[test]
    fn normalization_statistics_reject_invalid_width_and_values() {
        for name in ["state_mean", "state_std", "action_mean", "action_std"] {
            let mut stats = stats();
            match name {
                "state_mean" => stats.state_mean.pop(),
                "state_std" => stats.state_std.pop(),
                "action_mean" => stats.action_mean.pop(),
                _ => stats.action_std.pop(),
            };
            assert!(stats.validate().unwrap_err().to_string().contains(name));
        }
        let mut stats = stats();
        stats.state_std[0] = 0.0;
        assert!(stats.validate().is_err());
        stats.state_std[0] = 1.0;
        stats.action_mean[0] = f32::NAN;
        assert!(stats.validate().is_err());
    }
}

#[cfg(test)]
mod diagnostics {
    use super::*;
    use crate::processor::{Context, Mode};

    fn load_checkpoint(path: &str) -> anyhow::Result<(ActPolicy, Pipeline, Pipeline)> {
        let (policy, pre, post, _) =
            ActPolicy::import_configured(std::path::Path::new(path), crate::config::Device::Auto)?;
        Ok((policy, pre, post))
    }

    fn normalized_input(
        policy: &ActPolicy,
        pre: &Pipeline,
        state: &[f32],
        image: &[u8],
    ) -> anyhow::Result<Batch> {
        pre.run(
            observation_batch(state, image),
            &Context {
                device: policy.device(),
                mode: Mode::Eval,
            },
        )
    }

    fn canonical_actions(
        policy: &ActPolicy,
        pre: &Pipeline,
        post: &Pipeline,
        state: &[f32],
        image: &[u8],
    ) -> anyhow::Result<Vec<f32>> {
        let input = normalized_input(policy, pre, state, image)?;
        let output = policy.predict(input)?;
        let mut output = post.run(
            output,
            &Context {
                device: policy.device(),
                mode: Mode::Eval,
            },
        )?;
        Ok(output
            .take("action")?
            .into_float3()?
            .into_data()
            .try_to_vec::<f32>()?)
    }

    fn max_abs_diff(got: &[f32], reference: &[f32]) -> anyhow::Result<(usize, f32)> {
        anyhow::ensure!(
            got.len() == reference.len(),
            "length differs: {} vs {}",
            got.len(),
            reference.len()
        );
        let mut worst = (0, 0.0);
        for (i, (g, r)) in got.iter().zip(reference).enumerate() {
            anyhow::ensure!(
                g.is_finite() && r.is_finite(),
                "non-finite value at index {i}: got={g}, reference={r}"
            );
            let difference = (g - r).abs();
            if difference > worst.1 {
                worst = (i, difference);
            }
        }
        Ok(worst)
    }

    #[test]
    fn parity_oracle_rejects_non_finite_values_on_either_side() {
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(max_abs_diff(&[invalid], &[0.0]).is_err());
            assert!(max_abs_diff(&[0.0], &[invalid]).is_err());
        }
        assert_eq!(max_abs_diff(&[1.0, 3.0], &[1.0, 2.5]).unwrap(), (1, 0.5));
    }

    /// M2 判据：与 `scripts/dump_act_reference.py` 产物的逐元素对齐。
    /// 运行：`PNEUMA_ACT_CKPT=<ckpt> PNEUMA_PARITY_DIR=<dir> cargo test -- --ignored act_parity`
    #[test]
    #[ignore = "需要官方 checkpoint 与 python 参照产物"]
    fn act_parity_against_python_reference() {
        let ckpt = std::env::var("PNEUMA_ACT_CKPT").expect("PNEUMA_ACT_CKPT");
        let dir = std::env::var("PNEUMA_PARITY_DIR").expect("PNEUMA_PARITY_DIR");
        let dir = std::path::Path::new(&dir);

        let state: Vec<f32> = {
            let b = std::fs::read(dir.join("state.bin")).unwrap();
            b.as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect()
        };
        let image = std::fs::read(dir.join("image_hwc.bin")).unwrap();
        let reference: Vec<f32> = {
            let b = std::fs::read(dir.join("actions_ref.bin")).unwrap();
            b.as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect()
        };

        let (policy, pre, post) = load_checkpoint(&ckpt).unwrap();
        let got = canonical_actions(&policy, &pre, &post, &state, &image).unwrap();
        let worst = max_abs_diff(&got, &reference).expect("模型与参照必须等长且逐元素有限");
        println!("max abs diff = {:.3e} @ {}", worst.1, worst.0);
        assert!(worst.1 < 5e-3, "与 python 参照偏差过大: {worst:?}");
    }

    /// 调试：分段与 python 中间值比对，定位偏差来源（backbone → encoder → head）。
    #[test]
    #[ignore = "需要 PNEUMA_* 环境变量与 dump_act_reference.py 中间产物"]
    fn act_bisect_stages() {
        let ckpt = std::env::var("PNEUMA_ACT_CKPT").unwrap();
        let dirvar = std::env::var("PNEUMA_PARITY_DIR").unwrap();
        let dir = std::path::Path::new(&dirvar);
        let read = |n: &str| -> Vec<f32> {
            std::fs::read(dir.join(n))
                .unwrap()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect()
        };
        let diff = |name: &str, got: &[f32], want: &[f32]| {
            let (_, worst) = max_abs_diff(got, want).expect("中间值必须等长且逐元素有限");
            let n_big = got
                .iter()
                .zip(want)
                .filter(|(g, w)| (*g - *w).abs() > 1e-3)
                .count();
            println!("{name}: worst={worst:.3e}  n(>1e-3)={n_big}/{}", got.len());
        };

        let state: Vec<f32> = read("state.bin");
        let image = std::fs::read(dir.join("image_hwc.bin")).unwrap();
        let (policy, pre, _) = load_checkpoint(&ckpt).unwrap();
        let mut batch = normalized_input(&policy, &pre, &state, &image).unwrap();
        let state_t = batch
            .take("observation.state")
            .unwrap()
            .into_float2()
            .unwrap();
        let image_t = batch
            .take("observation.images.top")
            .unwrap()
            .into_float4()
            .unwrap();

        let feat = policy.model.backbone.forward(image_t.clone());
        let [_, _, fh, fw] = feat.dims();
        let fd: burn::tensor::TensorData = feat.clone().into_data();
        diff(
            "backbone",
            &fd.iter::<f32>().collect::<Vec<_>>(),
            &read("backbone_out.bin"),
        );
        println!("backbone dims = {:?}", feat.dims());

        let (tokens, pos) = policy
            .model
            .encoder_inputs(state_t, feat, policy.cfg.latent_dim);
        println!(
            "tokens={:?} pos={:?} (h,w)=({fh},{fw})",
            tokens.dims(),
            pos.dims()
        );

        let enc = policy.model.encoder.forward(tokens, &pos);
        let ed: TensorData = enc.into_data();
        diff(
            "encoder",
            &ed.iter::<f32>().collect::<Vec<_>>(),
            &read("encoder_out.bin"),
        );
    }

    /// M4 延迟对比：预热后平均单次前向（`BURN_DEVICE` 切后端各跑一遍）。
    /// 运行：`PNEUMA_ACT_CKPT=<ckpt> PNEUMA_PARITY_DIR=<dir> cargo test --release -- --ignored act_forward_latency --nocapture`
    #[test]
    #[ignore = "需要官方 checkpoint 与参照观测"]
    fn act_forward_latency() {
        const N: u32 = 8;

        let ckpt = std::env::var("PNEUMA_ACT_CKPT").unwrap();
        let dirvar = std::env::var("PNEUMA_PARITY_DIR").unwrap();
        let dir = std::path::Path::new(&dirvar);
        let state: Vec<f32> = std::fs::read(dir.join("state.bin"))
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        let image = std::fs::read(dir.join("image_hwc.bin")).unwrap();

        let (policy, pre, post) = load_checkpoint(&ckpt).unwrap();
        let started = std::time::Instant::now();
        let warm = canonical_actions(&policy, &pre, &post, &state, &image).unwrap();
        println!("first forward = {:?}（含后端预热）", started.elapsed());
        let started = std::time::Instant::now();
        for _ in 0..N {
            canonical_actions(&policy, &pre, &post, &state, &image).unwrap();
        }
        println!(
            "steady forward = {:?}（{} 次均值，输出与首次一致={:?}）",
            started.elapsed() / N,
            N,
            canonical_actions(&policy, &pre, &post, &state, &image).unwrap() == warm
        );
    }
}
