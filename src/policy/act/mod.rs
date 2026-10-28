//! Burn 原生 ACT：字段名对齐 torch 属性名，safetensors 直载（PyTorchToBurnAdapter），
//! 预处理复刻官方 remote 服务侧 policy_runner._batch + normalize/unnormalize 缓冲。
//! 推理路径 use_vae 下 z=0（eval 语义），vae_encoder 权重不加载。

pub mod config;
pub mod position;
pub mod resnet;
pub mod transformer;

use burn::module::Module;
use burn::nn::conv::Conv2d;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{Device, Tensor, TensorData};
use rmpv::Value;

use crate::codec;
use crate::policy::{Chunk, PolicyRunner};
use crate::protocol::{Capabilities, ErrorCode, ProtocolError, body_get};

use config::ActConfig;
use position::sinusoidal_2d;
use resnet::{Resnet, ResnetConfig};
use transformer::{ACTDecoder, ACTEncoder, EncoderLayerConfig, LayerNorm};

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

/// ACT 顶层模块；字段名 = modeling_act.py 的 nn.Module 属性名。
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
        let layer_cfg = EncoderLayerConfig {
            dim_model: self.dim_model,
            n_heads: self.n_heads,
            dim_feedforward: self.dim_feedforward,
        };
        Act {
            backbone: ResnetConfig.init(device),
            encoder: ACTEncoder {
                layers: (0..self.n_encoder_layers)
                    .map(|_| layer_cfg.init(device))
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
    /// state (1,14) 与 image (1,3,H,W) 均已归一化；返回 (1, chunk_size, 14) 归一化动作。
    pub fn forward(&self, state: Tensor<2>, image: Tensor<4>, latent_dim: usize) -> Tensor<3> {
        let device = state.device();
        let feat = self.backbone.forward(image);
        let (tokens, pos) = self.encoder_inputs(state, feat, latent_dim);
        let dim_model = tokens.dims()[2];
        let enc = self.encoder.forward(tokens, &pos);
        let chunk = self.decoder_pos_embed.rows().dims()[1];
        let dec_in = Tensor::<3, _>::zeros([1, chunk, dim_model], &device);
        let dec_pos = self.decoder_pos_embed.rows();
        let dec = self.decoder.forward(dec_in, &dec_pos, enc, &pos);
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

/// checkpoint 自带的归一化缓冲（normalize_inputs.*/unnormalize_outputs.*）。
struct NormStats {
    image_mean: [f32; 3],
    image_std: [f32; 3],
    state_mean: Vec<f32>,
    state_std: Vec<f32>,
    action_mean: Vec<f32>,
    action_std: Vec<f32>,
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
        let cfg = config::parse(&path.join("config.json"))?;
        let dir = path.display();
        let device = match device {
            crate::config::Device::Auto => Device::default(),
            crate::config::Device::Flex => Device::flex(),
            crate::config::Device::Metal => Device::metal(Default::default()),
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
        use anyhow::Context;
        use burn_store::ModuleSnapshot;
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

        // 归一化缓冲单独从同一文件读取（不属于模块树）。
        use safetensors::SafeTensors;
        let bytes = std::fs::read(path.join("model.safetensors"))?;
        let st = SafeTensors::deserialize(&bytes)?;
        let image_mean: [f32; 3] =
            f32_tensor(&st, "normalize_inputs.buffer_observation_images_top.mean")?
                .try_into()
                .map_err(|_| anyhow::anyhow!("image mean must have 3 channels"))?;
        let image_std: [f32; 3] =
            f32_tensor(&st, "normalize_inputs.buffer_observation_images_top.std")?
                .try_into()
                .map_err(|_| anyhow::anyhow!("image std must have 3 channels"))?;
        let stats = NormStats {
            image_mean,
            image_std,
            state_mean: f32_tensor(&st, "normalize_inputs.buffer_observation_state.mean")?,
            state_std: f32_tensor(&st, "normalize_inputs.buffer_observation_state.std")?,
            action_mean: f32_tensor(&st, "unnormalize_outputs.buffer_action.mean")?,
            action_std: f32_tensor(&st, "unnormalize_outputs.buffer_action.std")?,
        };
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
        let s = &self.stats;
        let state_n: Vec<f32> = state
            .iter()
            .zip(&s.state_mean)
            .zip(&s.state_std)
            .map(|((x, m), d)| (x - m) / d)
            .collect();
        let state_t = Tensor::<2, _>::from_data(
            TensorData::new(state_n.clone(), [1, state_n.len()]),
            &self.device,
        );

        // 480×640 固定合约；行主序 HWC → CHW。
        let (h, w) = (480usize, 640usize);
        let mut chw = vec![0f32; 3 * h * w];
        for (i, &v) in image_hwc.iter().enumerate() {
            let c = i % 3;
            let pos = i / 3;
            chw[c * h * w + pos] = ((v as f32) / 255.0 - s.image_mean[c]) / s.image_std[c];
        }
        let image_t = Tensor::<4, _>::from_data(TensorData::new(chw, [1, 3, h, w]), &self.device);
        (state_t, image_t)
    }

    fn run(&self, state: &[f32], image_hwc: &[u8]) -> Vec<f32> {
        let (state_t, image_t) = self.preprocess(state, image_hwc);
        let out = self.model.forward(state_t, image_t, self.cfg.latent_dim);
        let [_, _steps, dim] = out.dims();
        let data = out.into_data();
        let flat: Vec<f32> = data.iter::<f32>().collect();
        // unnormalize_outputs: a * std + mean（逐分量）。
        flat.iter()
            .enumerate()
            .map(|(i, v)| v * self.stats.action_std[i % dim] + self.stats.action_mean[i % dim])
            .collect()
    }
}

/// 合约维度 × 权重维度的纯比对：错位是 profile 与 checkpoint 的配置错误，
/// 启动即报错好过一次 matmul panic；提出为纯函数以便用 profile 单测。
fn check_dims(
    caps: &Capabilities,
    state_dim: usize,
    action_dim: usize,
) -> Result<(), ProtocolError> {
    let state = caps
        .features
        .iter()
        .find(|f| f.name == "observation.state")
        .ok_or_else(|| {
            ProtocolError::new(
                ErrorCode::Incompatible,
                "contract requires observation.state",
            )
        })?;
    if state.shape != [state_dim as u64]
        || state.dtype != "float32"
        || caps.action_feature.shape != [action_dim as u64]
    {
        return Err(ProtocolError::new(
            ErrorCode::Incompatible,
            "state/action shape or dtype differs from checkpoint",
        ));
    }
    Ok(())
}

#[test]
fn profile_mismatched_with_checkpoint_is_startup_error() {
    // so101 合约 6 维、官方 ACT checkpoint 14 维：装载期就要拒绝，
    // 否则首次观测在 Burn matmul 里 panic、拖垮整个 server。
    use crate::config::ServerConfig;
    let caps = ServerConfig::from_args(["pneuma", "--config_path=examples/so101.toml"])
        .unwrap()
        .capabilities(50, 16);
    let error = check_dims(&caps, 14, 14).unwrap_err();
    assert_eq!(error.code, ErrorCode::Incompatible);
    assert!(check_dims(&caps, 6, 6).is_ok());
}

impl PolicyRunner for ActPolicy {
    fn validate_against(&self, caps: &Capabilities) -> Result<(), ProtocolError> {
        // 归一化统计的分量数即权重侧的合约维度。
        check_dims(
            caps,
            self.stats.state_mean.len(),
            self.stats.action_std.len(),
        )
    }

    fn predict(&self, caps: &Capabilities, features: &Value) -> Result<Chunk, ProtocolError> {
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
        let dim = caps.action_feature.shape[0] as usize;
        let steps = (caps.execution_steps as usize).min(self.cfg.chunk_size);
        Ok(Chunk {
            canonical_actions: codec::f32_le(&flat[..steps * dim]),
            execution_steps: steps as u64,
        })
    }
}

/// M2 判据：与 scripts/dump_act_reference.py 产物的逐元素对齐。
/// 运行：PNEUMA_ACT_CKPT=<ckpt> PNEUMA_PARITY_DIR=<dir> cargo test -- --ignored act_parity
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
    assert_eq!(got.len(), reference.len(), "参照应为同一 chunk×dim 平铺");
    let mut worst = (0usize, 0f32);
    for (i, (g, r)) in got.iter().zip(&reference).enumerate() {
        let d = (g - r).abs();
        if d > worst.1 {
            worst = (i, d);
        }
    }
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
        assert_eq!(got.len(), want.len(), "{name} 长度");
        let mut worst = 0f32;
        let mut n_big = 0;
        for (g, w) in got.iter().zip(want) {
            let d = (g - w).abs();
            worst = worst.max(d);
            if d > 1e-3 {
                n_big += 1;
            }
        }
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

/// M4 延迟对比：预热后平均单次前向（BURN_DEVICE 切后端各跑一遍）。
/// 运行：PNEUMA_ACT_CKPT=<ckpt> PNEUMA_PARITY_DIR=<dir> cargo test --release -- --ignored act_forward_latency --nocapture
#[test]
#[ignore = "需要官方 checkpoint 与参照观测"]
fn act_forward_latency() {
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
    const N: u32 = 8;
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
