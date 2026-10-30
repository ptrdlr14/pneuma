//! Burn 原生 ACT：字段名对齐 torch 属性名，safetensors 直载（`PyTorchToBurnAdapter`），
//! 预处理复刻官方 remote 服务侧 `policy_runner._batch` + normalize/unnormalize 缓冲。
//! 推理路径 `use_vae` 下 z=0（eval 语义），`vae_encoder` 权重不加载。

pub mod config;
pub mod position;
pub mod resnet;
pub mod transformer;

use burn::module::Module;
use burn::nn::conv::Conv2d;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{Device, DeviceKind, Tensor, TensorData};
use rmpv::Value;

use crate::codec;
use crate::policy::{Chunk, PolicyRunner};
use crate::protocol::{Capabilities, ErrorCode, FeatureSpec, ProtocolError, body_get};

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
    fn preprocess(
        &self,
        state: &[f32],
        image_hwc: &[u8],
        device: &Device,
    ) -> (Tensor<2>, Tensor<4>) {
        let s = self;
        let state_n: Vec<f32> = state
            .iter()
            .zip(&s.state_mean)
            .zip(&s.state_std)
            .map(|((x, m), d)| (x - m) / d)
            .collect();
        let state_t = Tensor::<2, _>::from_data(TensorData::new(state_n, [1, state.len()]), device);

        // 480×640 固定合约；行主序 HWC → CHW。
        let (h, w) = (480usize, 640usize);
        let mut chw = vec![0f32; 3 * h * w];
        for (i, &v) in image_hwc.iter().enumerate() {
            let c = i % 3;
            let pos = i / 3;
            chw[c * h * w + pos] = (f32::from(v) / 255.0 - s.image_mean[c]) / s.image_std[c];
        }
        let image_t = Tensor::<4, _>::from_data(TensorData::new(chw, [1, 3, h, w]), device);
        (state_t, image_t)
    }

    fn unnormalize_actions(&self, flat: &[f32]) -> Vec<f32> {
        let dim = self.action_mean.len();
        flat.iter()
            .enumerate()
            .map(|(i, v)| v * self.action_std[i % dim] + self.action_mean[i % dim])
            .collect()
    }
}

pub struct ActPolicy {
    model: Act,
    cfg: ActConfig,
    stats: NormStats,
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
    /// dir 为 checkpoint 目录（config.json + model.safetensors）。
    #[cfg(test)]
    pub fn load(dir: &str) -> anyhow::Result<Self> {
        Self::load_configured(std::path::Path::new(dir), crate::config::Device::Auto, 1)
    }

    pub fn load_configured(
        path: &std::path::Path,
        device: crate::config::Device,
        warmup_calls: usize,
    ) -> anyhow::Result<Self> {
        use anyhow::Context;
        use burn_store::ModuleSnapshot;

        let cfg = config::parse(&path.join("config.json"))?;
        let dir = path.display();
        let device = match device {
            crate::config::Device::Auto => Device::default(),
            crate::config::Device::Flex => Device::flex(),
            crate::config::Device::Metal => Device::metal(DeviceKind::default()),
        };
        tracing::info!("act model on device {device:?}");
        let mut model = cfg.init(&device);

        let mut store = burn_store::SafetensorsStore::from_file(path.join("model.safetensors"))
            .with_from_adapter(burn_store::PyTorchToBurnAdapter)
            // state_dict 以 self.model（ACT）为前缀；act 根模块剥掉它。
            .with_key_remapping(r"^model\.", "")
            // torch 的 in_proj_weight 是单参数名，Rust 侧走子模块路径。
            .with_key_remapping(r"\.in_proj_weight$", ".in_proj.weight")
            .with_key_remapping(r"\.in_proj_bias$", ".in_proj.bias")
            // downsample 的 Sequential 索引段 → 具名字段。
            .with_key_remapping(r"\.downsample\.0\.", ".downsample.conv.")
            .with_key_remapping(r"\.downsample\.1\.", ".downsample.norm.")
            // vae_encoder 与归一化缓冲不属于本模块。
            .allow_partial(true);
        let applied = model
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

        let stats = NormStats::load(&path.join("model.safetensors"))?;
        let policy = Self {
            model,
            cfg,
            stats,
            device,
        };
        // 预热：metal 后端首推理含内核 JIT（秒级），挪进启动期，首个控制周期不受影响。
        // 附带自检：坏权重/后端缺算子在这里快速失败，而不是留到客户端第一次请求。
        let started = std::time::Instant::now();
        for _ in 0..warmup_calls {
            policy.run(
                &vec![0.0; policy.stats.state_mean.len()],
                &vec![0u8; 480 * 640 * 3],
            );
        }
        tracing::info!("warmup forward 完成，用时 {:?}", started.elapsed());
        Ok(policy)
    }

    pub fn horizon(&self) -> (u64, u64) {
        (self.cfg.chunk_size as u64, self.cfg.n_action_steps as u64)
    }

    /// 官方服务侧 _batch 等价：rgb→CHW f32 /255 → (x-mean)/std；state (x-mean)/std。
    fn preprocess(&self, state: &[f32], image_hwc: &[u8]) -> (Tensor<2>, Tensor<4>) {
        self.stats.preprocess(state, image_hwc, &self.device)
    }

    fn run(&self, state: &[f32], image_hwc: &[u8]) -> Vec<f32> {
        let (state_t, image_t) = self.preprocess(state, image_hwc);
        let out = self.model.forward(state_t, image_t, self.cfg.latent_dim);
        let flat: Vec<f32> = out.into_data().iter::<f32>().collect();
        self.stats.unnormalize_actions(&flat)
    }
}

/// 合约维度 × 权重维度的纯比对：错位是 profile 与 checkpoint 的配置错误，
/// 启动即报错好过一次 matmul panic；提出为纯函数以便用 profile 单测。
pub fn validate_contract(
    features: &[FeatureSpec],
    action: &FeatureSpec,
) -> Result<(), ProtocolError> {
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
        return Err(ProtocolError::new(
            ErrorCode::Incompatible,
            "ACT requires observation.state [14], observation.images.top [480,640,3], and action [14]",
        ));
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
    assert_eq!(error.code, ErrorCode::Incompatible);
    let mut cfg = ServerConfig::load(std::path::Path::new("examples/aloha_sim.toml")).unwrap();
    assert!(validate_contract(&cfg.features, &cfg.action_feature).is_ok());
    cfg.features[1].shape = vec![240, 320, 3];
    cfg.model.repo_or_path = "/nonexistent/checkpoint".into();
    let Err(error) = crate::policy::build(&cfg) else {
        panic!("unsupported ACT image contract accepted");
    };
    assert!(error.to_string().contains("ACT requires"), "{error}");
}

impl PolicyRunner for ActPolicy {
    fn validate_against(&self, caps: &Capabilities) -> Result<(), ProtocolError> {
        validate_contract(&caps.features, &caps.action_feature)
    }

    fn predict(
        &self,
        caps: &Capabilities,
        features: &Value,
        _task: Option<&str>,
    ) -> Result<Chunk, ProtocolError> {
        features
            .as_map()
            .ok_or_else(|| ProtocolError::new(ErrorCode::Malformed, "features must be a map"))?;
        // server 已按 caps 校验过形状/类型，这里只做取数。
        let state_rec = codec::decode_record(
            body_get(features, "observation.state")
                .ok_or_else(|| ProtocolError::new(ErrorCode::Incompatible, "missing state"))?,
        )?;
        if state_rec.dtype != "float32" {
            return Err(ProtocolError::new(ErrorCode::Incompatible, "state dtype"));
        }
        let state: Vec<f32> = state_rec
            .data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();

        let image_name = caps
            .features
            .iter()
            .find(|f| f.kind == "rgb")
            .map(|f| f.name.clone())
            .ok_or_else(|| {
                ProtocolError::new(ErrorCode::Execution, "profile has no rgb feature")
            })?;
        let image_rec = codec::decode_record(
            body_get(features, &image_name)
                .ok_or_else(|| ProtocolError::new(ErrorCode::Incompatible, "missing image"))?,
        )?;
        if image_rec.dtype != "uint8" {
            return Err(ProtocolError::new(ErrorCode::Incompatible, "image dtype"));
        }

        let flat = self.run(&state, image_rec.data);
        execution_chunk(&flat, caps.execution_steps, self.cfg.chunk_size)
    }
}

fn execution_chunk(
    flat: &[f32],
    requested_steps: u64,
    chunk_size: usize,
) -> Result<Chunk, ProtocolError> {
    let steps = usize::try_from(requested_steps)
        .map_err(|_| {
            ProtocolError::new(
                ErrorCode::Execution,
                "execution_steps exceeds platform size",
            )
        })?
        .min(chunk_size);
    let length = steps
        .checked_mul(14)
        .ok_or_else(|| ProtocolError::new(ErrorCode::Execution, "action size overflows"))?;
    let actions = flat.get(..length).ok_or_else(|| {
        ProtocolError::new(
            ErrorCode::Execution,
            "model action output is shorter than execution_steps",
        )
    })?;
    if actions.iter().any(|v| !v.is_finite()) {
        return Err(ProtocolError::new(
            ErrorCode::Execution,
            "model action output contains non-finite values",
        ));
    }
    Ok(Chunk {
        canonical_actions: codec::f32_le(actions),
        execution_steps: steps as u64,
    })
}

#[cfg(test)]
mod numerical_tests {
    use super::*;

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
        let (state, image) = stats.preprocess(&[5.0; 14], &image, &Device::flex());
        assert_eq!(
            state.into_data().try_to_vec::<f32>().unwrap(),
            vec![2.0; 14]
        );
        let image = image.into_data().try_to_vec::<f32>().unwrap();
        for (channel, pixels) in [[255u8, 64], [128, 32], [0, 255]].iter().enumerate() {
            for (pixel, value) in pixels.iter().enumerate() {
                let expected = (f32::from(*value) / 255.0 - stats.image_mean[channel])
                    / stats.image_std[channel];
                assert!((image[channel * 480 * 640 + pixel] - expected).abs() < 1e-6);
            }
        }
        let actions = stats.unnormalize_actions(&[1.0; 42]);
        let chunk = execution_chunk(&actions, 2, 3).unwrap();
        assert_eq!(chunk.execution_steps, 2);
        let values: Vec<_> = chunk
            .canonical_actions
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| f32::from_le_bytes(*v))
            .collect();
        assert_eq!(values.len(), 28);
        for (i, value) in values.iter().enumerate() {
            assert_eq!(*value, stats.action_mean[i % 14] + 2.0);
        }
        assert!(execution_chunk(&[f32::NAN; 14], 1, 1).is_err());
        assert!(execution_chunk(&[0.0; 13], 1, 1).is_err());
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
            let error = match crate::policy::build(&cfg) {
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

        let policy = ActPolicy::load(&ckpt).unwrap();
        let got = policy.run(&state, &image);
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
        let policy = ActPolicy::load(&ckpt).unwrap();
        let (state_t, image_t) = policy.preprocess(&state, &image);

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

        let policy = ActPolicy::load(&ckpt).unwrap();
        let started = std::time::Instant::now();
        let warm = policy.run(&state, &image);
        println!("first forward = {:?}（含后端预热）", started.elapsed());
        let started = std::time::Instant::now();
        for _ in 0..N {
            policy.run(&state, &image);
        }
        println!(
            "steady forward = {:?}（{} 次均值，输出与首次一致={:?}）",
            started.elapsed() / N,
            N,
            policy.run(&state, &image) == warm
        );
    }
}
