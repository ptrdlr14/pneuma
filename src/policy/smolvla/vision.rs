//! SmolVLM 的 SigLIP 视觉塔及保留 patch 顺序的 pixel-shuffle connector。

use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::tensor::activation::{gelu_approximate, softmax};
use burn::tensor::{Device, Int, Tensor, TensorData};
use serde::Deserialize;

use super::transformer::LayerNorm;

#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub(super) struct VisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub layer_norm_eps: f64,
    pub scale_factor: usize,
}

impl Default for VisionConfig {
    fn default() -> Self {
        Self {
            hidden_size: 768,
            intermediate_size: 3072,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            image_size: 512,
            patch_size: 16,
            layer_norm_eps: 1e-6,
            scale_factor: 4,
        }
    }
}

impl VisionConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.num_attention_heads > 0
                && self.hidden_size > 0
                && self.hidden_size.is_multiple_of(self.num_attention_heads)
                && self.intermediate_size > 0
                && self.num_hidden_layers > 0
                && self.patch_size > 0
                && self.image_size > 0
                && self.image_size.is_multiple_of(self.patch_size)
                && self.scale_factor > 0
                && (self.image_size / self.patch_size).is_multiple_of(self.scale_factor)
                && self.layer_norm_eps.is_finite()
                && self.layer_norm_eps > 0.,
            "invalid SmolVLM vision configuration"
        );
        Ok(())
    }
}

#[derive(Module, Debug)]
struct VisionEmbeddings {
    patch_embedding: Conv2d,
    position_embedding: Embedding,
}

impl VisionEmbeddings {
    fn init(cfg: &VisionConfig, device: &Device) -> Self {
        let grid = cfg.image_size / cfg.patch_size;
        Self {
            patch_embedding: Conv2dConfig::new([3, cfg.hidden_size], [cfg.patch_size; 2])
                .with_stride([cfg.patch_size; 2])
                .init(device),
            position_embedding: EmbeddingConfig::new(grid * grid, cfg.hidden_size).init(device),
        }
    }

    fn forward(&self, image: Tensor<4>) -> Tensor<3> {
        let patches = self.patch_embedding.forward(image);
        let [batch, dim, h, w] = patches.dims();
        let positions: Vec<i64> = (0..h * w).map(|x| x as i64).collect();
        let positions =
            Tensor::<2, Int>::from_data(TensorData::new(positions, [1, h * w]), &patches.device());
        patches.permute([0, 2, 3, 1]).reshape([batch, h * w, dim])
            + self.position_embedding.forward(positions)
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
pub(super) struct VisionModel {
    embeddings: VisionEmbeddings,
    encoder: VisionEncoder,
    post_layernorm: LayerNorm,
}

impl VisionModel {
    pub fn init(cfg: &VisionConfig, device: &Device) -> Self {
        Self {
            embeddings: VisionEmbeddings::init(cfg, device),
            encoder: VisionEncoder {
                layers: (0..cfg.num_hidden_layers)
                    .map(|_| VisionEncoderLayer::init(cfg, device))
                    .collect(),
            },
            post_layernorm: LayerNorm::init(cfg.hidden_size, cfg.layer_norm_eps, device),
        }
    }

    pub fn forward(&self, image: Tensor<4>) -> Tensor<3> {
        let mut x = self.embeddings.forward(image);
        for layer in &self.encoder.layers {
            x = layer.forward(x);
        }
        self.post_layernorm.forward(x)
    }
}

#[derive(Module, Debug)]
struct ModalityProjection {
    proj: Linear,
}

#[derive(Module, Debug)]
pub(super) struct Connector {
    modality_projection: ModalityProjection,
    #[module(skip)]
    scale_factor: usize,
    #[module(skip)]
    grid: usize,
}

impl Connector {
    pub fn init(cfg: &VisionConfig, text_dim: usize, device: &Device) -> Self {
        Self {
            modality_projection: ModalityProjection {
                proj: LinearConfig::new(cfg.hidden_size * cfg.scale_factor.pow(2), text_dim)
                    .with_bias(false)
                    .init(device),
            },
            scale_factor: cfg.scale_factor,
            grid: cfg.image_size / cfg.patch_size,
        }
    }

    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        self.modality_projection
            .proj
            .forward(pixel_shuffle(x, self.grid, self.scale_factor))
    }
}

fn pixel_shuffle(x: Tensor<3>, grid: usize, scale: usize) -> Tensor<3> {
    let [batch, _, dim] = x.dims();
    let small = grid / scale;
    x.reshape([batch, grid, small, dim * scale])
        .permute([0, 2, 1, 3])
        .reshape([batch, small, small, dim * scale * scale])
        .permute([0, 2, 1, 3])
        .reshape([batch, small * small, dim * scale * scale])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pixel_shuffle_preserves_official_patch_order() {
        let x = Tensor::<3>::from_data(
            TensorData::new((0..16).map(|x| x as f32).collect(), [1, 16, 1]),
            &Device::flex(),
        );
        let output = pixel_shuffle(x, 4, 2)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        assert_eq!(
            output,
            [
                0., 1., 4., 5., 2., 3., 6., 7., 8., 9., 12., 13., 10., 11., 14., 15.
            ]
        );
    }

    #[test]
    fn small_vision_tower_and_connector_produce_finite_tokens() {
        let cfg = VisionConfig {
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            image_size: 8,
            patch_size: 2,
            scale_factor: 2,
            ..Default::default()
        };
        cfg.validate().unwrap();
        let device = Device::flex();
        let output = Connector::init(&cfg, 12, &device)
            .forward(VisionModel::init(&cfg, &device).forward(Tensor::ones([1, 3, 8, 8], &device)));
        assert_eq!(output.dims(), [1, 4, 12]);
        assert!(output.into_data().iter::<f32>().all(|x| x.is_finite()));
    }
}
