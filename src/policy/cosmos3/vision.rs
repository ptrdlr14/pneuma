//! Cosmos3 Edge 的 SigLIP2 视觉塔与 patch merger。
//!
//! 与 π₀.₅ 的视觉塔不同：patch 嵌入是线性层，token 按 2×2 块优先（block-major）打包，
//! 投影层用精确 GELU。

// 视觉塔与投影层在 MoT 主干接通前不参与运行时构建；届时移除此豁免。
#![allow(dead_code)]

use burn::module::{Module, Param};
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::tensor::activation::{gelu, gelu_approximate, softmax};
use burn::tensor::{Device, Tensor};

use super::config::{ProjectorConfig, VisionConfig};

#[derive(Module, Debug)]
pub(super) struct LayerNorm {
    weight: Param<Tensor<1>>,
    bias: Param<Tensor<1>>,
    #[module(skip)]
    epsilon: f64,
}

impl LayerNorm {
    fn init(dim: usize, epsilon: f64, device: &Device) -> Self {
        Self {
            weight: Param::from_tensor(Tensor::ones([dim], device)),
            bias: Param::from_tensor(Tensor::zeros([dim], device)),
            epsilon,
        }
    }

    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let dim = x.dims()[2];
        let centered = x.clone() - x.mean_dim(2);
        let variance = centered.clone().powf_scalar(2.0).mean_dim(2);
        centered / (variance + self.epsilon).sqrt() * self.weight.val().reshape([1, 1, dim])
            + self.bias.val().reshape([1, 1, dim])
    }
}

#[derive(Module, Debug)]
struct VisionEmbeddings {
    patch_embedding: Linear,
    position_embedding: Embedding,
}

impl VisionEmbeddings {
    fn init(cfg: &VisionConfig, device: &Device) -> Self {
        Self {
            patch_embedding: LinearConfig::new(
                cfg.num_channels * cfg.patch_size * cfg.patch_size,
                cfg.hidden_size,
            )
            .init(device),
            position_embedding: EmbeddingConfig::new(cfg.num_patches, cfg.hidden_size).init(device),
        }
    }

    fn forward(&self, patches: Tensor<3>, grid: usize, merge: usize) -> Tensor<3> {
        let weight = self.position_embedding.weight.val();
        let dim = weight.dims()[1];
        let positions = block_major(weight.reshape([1, grid, grid, dim]), merge);
        self.patch_embedding.forward(patches) + positions
    }
}

#[derive(Module, Debug)]
struct VisionAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    #[module(skip)]
    heads: usize,
}

impl VisionAttention {
    fn init(cfg: &VisionConfig, device: &Device) -> Self {
        let linear = || LinearConfig::new(cfg.hidden_size, cfg.hidden_size).init(device);
        Self {
            q_proj: linear(),
            k_proj: linear(),
            v_proj: linear(),
            out_proj: linear(),
            heads: cfg.num_attention_heads,
        }
    }

    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let [batch, seq, dim] = x.dims();
        let width = dim / self.heads;
        let heads = |x: Tensor<3>| {
            x.reshape([batch, seq, self.heads, width])
                .permute([0, 2, 1, 3])
        };
        let q = heads(self.q_proj.forward(x.clone()));
        let k = heads(self.k_proj.forward(x.clone()));
        let v = heads(self.v_proj.forward(x));
        let probabilities = softmax(q.matmul(k.swap_dims(2, 3)) / (width as f64).sqrt(), 3);
        self.out_proj.forward(
            probabilities
                .matmul(v)
                .permute([0, 2, 1, 3])
                .reshape([batch, seq, dim]),
        )
    }
}

#[derive(Module, Debug)]
struct VisionMlp {
    fc1: Linear,
    fc2: Linear,
}

#[derive(Module, Debug)]
struct VisionEncoderLayer {
    self_attn: VisionAttention,
    layer_norm1: LayerNorm,
    mlp: VisionMlp,
    layer_norm2: LayerNorm,
}

impl VisionEncoderLayer {
    fn init(cfg: &VisionConfig, device: &Device) -> Self {
        Self {
            self_attn: VisionAttention::init(cfg, device),
            layer_norm1: LayerNorm::init(cfg.hidden_size, cfg.layer_norm_eps, device),
            mlp: VisionMlp {
                fc1: LinearConfig::new(cfg.hidden_size, cfg.intermediate_size).init(device),
                fc2: LinearConfig::new(cfg.intermediate_size, cfg.hidden_size).init(device),
            },
            layer_norm2: LayerNorm::init(cfg.hidden_size, cfg.layer_norm_eps, device),
        }
    }

    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let x = x.clone() + self.self_attn.forward(self.layer_norm1.forward(x));
        let mlp = self.mlp.fc2.forward(gelu_approximate(
            self.mlp.fc1.forward(self.layer_norm2.forward(x.clone())),
        ));
        x + mlp
    }
}

#[derive(Module, Debug)]
struct VisionEncoder {
    layers: Vec<VisionEncoderLayer>,
}

#[derive(Module, Debug)]
pub(super) struct Visual {
    embeddings: VisionEmbeddings,
    encoder: VisionEncoder,
    post_layernorm: LayerNorm,
    #[module(skip)]
    patch: usize,
    #[module(skip)]
    merge: usize,
}

impl Visual {
    pub(super) fn init(cfg: &VisionConfig, device: &Device) -> Self {
        Self {
            embeddings: VisionEmbeddings::init(cfg, device),
            encoder: VisionEncoder {
                layers: (0..cfg.num_hidden_layers)
                    .map(|_| VisionEncoderLayer::init(cfg, device))
                    .collect(),
            },
            post_layernorm: LayerNorm::init(cfg.hidden_size, cfg.layer_norm_eps, device),
            patch: cfg.patch_size,
            merge: cfg.spatial_merge_size,
        }
    }

    /// 输入 `[1,3,H,W]` 的 `[-1,1]` 图像；输出合并前的 patch 表示。
    pub(super) fn forward(&self, image: Tensor<4>) -> Tensor<3> {
        let grid = image.dims()[2] / self.patch;
        let patches = patchify(image, self.patch, self.merge);
        let mut x = self.embeddings.forward(patches, grid, self.merge);
        for layer in &self.encoder.layers {
            x = layer.forward(x);
        }
        self.post_layernorm.forward(x)
    }
}

#[derive(Module, Debug)]
pub(super) struct Projector {
    norm: LayerNorm,
    linear_fc1: Linear,
    linear_fc2: Linear,
    #[module(skip)]
    merge: usize,
    #[module(skip)]
    out: usize,
}

impl Projector {
    pub(super) fn init(cfg: &ProjectorConfig, epsilon: f64, device: &Device) -> Self {
        let merged = cfg.input_hidden_size * cfg.spatial_merge_size * cfg.spatial_merge_size;
        Self {
            norm: LayerNorm::init(cfg.input_hidden_size, epsilon, device),
            linear_fc1: LinearConfig::new(merged, cfg.merger_intermediate_size).init(device),
            linear_fc2: LinearConfig::new(cfg.merger_intermediate_size, cfg.out_hidden_size)
                .init(device),
            merge: cfg.spatial_merge_size,
            out: cfg.out_hidden_size,
        }
    }

    /// 每 `merge²` 个连续 token 拼成一行；输出 `[1, tokens/merge², out]`。
    pub(super) fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let [batch, tokens, dim] = x.dims();
        let block = self.merge * self.merge;
        let merged = self
            .norm
            .forward(x)
            .reshape([batch * tokens / block, dim * block]);
        let hidden = gelu(self.linear_fc1.forward(merged));
        self.linear_fc2
            .forward(hidden)
            .reshape([batch, tokens / block, self.out])
    }
}

/// 把 `[B,C,H,W]` 切成 patch，并重排为 2×2 块优先。
fn patchify(image: Tensor<4>, patch: usize, merge: usize) -> Tensor<3> {
    let [batch, channels, height, width] = image.dims();
    let (gh, gw) = (height / patch, width / patch);
    let dim = channels * patch * patch;
    let grid = image
        .reshape([batch, channels, gh, patch, gw, patch])
        .permute([0, 2, 4, 1, 3, 5])
        .reshape([batch, gh, gw, dim]);
    block_major(grid, merge)
}

/// `[B,gh,gw,D]` → `[B,gh*gw,D]`，块优先顺序。
///
/// 块内顺序为（行内偏移，列内偏移），与官方 `reshape(h/2,2,w/2,2,-1).transpose(1,2)` 一致。
fn block_major(grid: Tensor<4>, merge: usize) -> Tensor<3> {
    let [batch, gh, gw, dim] = grid.dims();
    grid.reshape([batch, gh / merge, merge, gw / merge, merge, dim])
        .permute([0, 1, 3, 2, 4, 5])
        .reshape([batch, gh * gw, dim])
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData};

    fn small_vision() -> VisionConfig {
        VisionConfig {
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_channels: 3,
            patch_size: 2,
            layer_norm_eps: 1e-6,
            hidden_act: "gelu_pytorch_tanh".into(),
            num_patches: 16,
            spatial_merge_size: 2,
        }
    }

    #[test]
    fn patchify_uses_block_major_order() {
        let device = Device::flex();
        // 8×8 图像，patch=2，网格 4×4；每个像素值 = row*8 + col。
        let values: Vec<f32> = (0..64).map(|value| value as f32).collect();
        let image = Tensor::<4>::from_data(TensorData::new(values, [1, 1, 8, 8]), &device);
        let patches = patchify(image, 2, 2);
        assert_eq!(patches.dims(), [1, 16, 4]);
        let first: Vec<f32> = patches
            .into_data()
            .try_to_vec::<f32>()
            .unwrap()
            .chunks(4)
            .map(|patch| patch[0])
            .collect();
        // 每块 2×2 个 patch，块优先。
        assert_eq!(&first[..8], &[0., 2., 16., 18., 4., 6., 20., 22.]);
    }

    #[test]
    fn visual_and_projector_shapes() {
        let device = Device::flex();
        let cfg = small_vision();
        let visual = Visual::init(&cfg, &device);
        let projector = Projector::init(
            &ProjectorConfig {
                input_hidden_size: 32,
                merger_intermediate_size: 128,
                out_hidden_size: 48,
                spatial_merge_size: 2,
                use_postshuffle_norm: false,
            },
            1e-6,
            &device,
        );
        let image = Tensor::<4>::zeros([1, 3, 8, 8], &device);
        let tokens = visual.forward(image);
        assert_eq!(tokens.dims(), [1, 16, 32]);
        let merged = projector.forward(tokens);
        assert_eq!(merged.dims(), [1, 4, 48]);
    }
}

#[cfg(test)]
mod reference {
    use super::*;
    use burn::tensor::{DType, TensorData};
    use burn_store::{
        FloatCastAdapter, ModuleAdapter, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore,
    };
    use std::path::{Path, PathBuf};

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

    fn compare(name: &str, got: &[f32], want: &[f32], tolerance: f32) {
        assert_eq!(got.len(), want.len(), "{name}: length");
        let max = got
            .iter()
            .zip(want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let mean = got
            .iter()
            .zip(want)
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / got.len() as f32;
        println!(
            "{name}: max {max:.3e} mean {mean:.3e} over {} values",
            got.len()
        );
        assert!(
            max <= tolerance,
            "{name} differs by {max:.3e} > {tolerance:.1e}"
        );
    }

    fn store(path: &PathBuf, prefix: &str) -> SafetensorsStore {
        SafetensorsStore::from_file(path)
            .with_from_adapter(PyTorchToBurnAdapter.chain(FloatCastAdapter::to(DType::F32)))
            .with_key_remapping(prefix, "")
            .allow_partial(true)
    }

    /// 真实权重下与 HF SigLIP2 参照比较视觉塔与投影层。
    #[test]
    #[ignore = "需要 cosmos3 权重与参照张量"]
    fn vision_matches_reference() {
        let checkpoint =
            PathBuf::from(std::env::var("PNEUMA_COSMOS3_CKPT").expect("PNEUMA_COSMOS3_CKPT"));
        let parity =
            PathBuf::from(std::env::var("PNEUMA_COSMOS3_VISION").expect("PNEUMA_COSMOS3_VISION"));
        let device = Device::flex();
        let weights = checkpoint.join("model.safetensors");

        let mut visual = Visual::init(&VisionConfig::default(), &device);
        let mut visual_store = store(&weights, r"^model\.network\.mot_backbone\.model\.visual\.");
        let applied = visual.load_from(&mut visual_store).expect("loading visual");
        assert!(
            applied.missing.is_empty(),
            "missing visual weights: {:?}",
            applied.missing
        );

        let mut projector = Projector::init(&ProjectorConfig::default(), 1e-6, &device);
        let mut projector_store = store(
            &weights,
            r"^model\.network\.mot_backbone\.model\.projector\.",
        );
        let applied = projector
            .load_from(&mut projector_store)
            .expect("loading projector");
        assert!(
            applied.missing.is_empty(),
            "missing projector weights: {:?}",
            applied.missing
        );

        let image = Tensor::<4>::from_data(
            TensorData::new(read_f32(&parity.join("image.bin")), [1, 3, 256, 256]),
            &device,
        );
        let hidden = visual.forward(image);
        assert_eq!(hidden.dims(), [1, 256, 1152]);
        compare(
            "vision_hidden",
            &hidden
                .clone()
                .into_data()
                .try_to_vec::<f32>()
                .expect("hidden"),
            &read_f32(&parity.join("vision_hidden.bin")),
            5e-3,
        );

        let projected = projector.forward(hidden);
        assert_eq!(projected.dims(), [1, 64, 2048]);
        compare(
            "projected",
            &projected
                .into_data()
                .try_to_vec::<f32>()
                .expect("projected"),
            &read_f32(&parity.join("projected.bin")),
            5e-3,
        );
    }
}
