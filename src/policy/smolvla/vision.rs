//! SmolVLM 的 SigLIP 视觉塔及保留 patch 顺序的 pixel-shuffle connector。

use burn::module::{Module, Param};
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::tensor::activation::{gelu_approximate, softmax};
use burn::tensor::{Device, Int, Tensor, TensorData};
use serde::Deserialize;

/// SigLIP 视觉塔使用的标准 LayerNorm。
#[derive(Module, Debug)]
pub(super) struct SmolVlmVisionLayerNorm {
    weight: Param<Tensor<1>>,
    bias: Param<Tensor<1>>,
    #[module(skip)]
    epsilon: f64,
}

/// SmolVLM 视觉层归一化的构造参数。
#[derive(Clone, Copy, Debug)]
pub(super) struct SmolVlmVisionLayerNormConfig {
    dim: usize,
    epsilon: f64,
}

impl SmolVlmVisionLayerNormConfig {
    pub(super) fn new(dim: usize, epsilon: f64) -> Self {
        Self { dim, epsilon }
    }

    pub(super) fn init(&self, device: &Device) -> SmolVlmVisionLayerNorm {
        SmolVlmVisionLayerNorm {
            weight: Param::from_tensor(Tensor::ones([self.dim], device)),
            bias: Param::from_tensor(Tensor::zeros([self.dim], device)),
            epsilon: self.epsilon,
        }
    }
}

impl SmolVlmVisionLayerNorm {
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let dim = x.dims()[2];
        let centered = x.clone() - x.mean_dim(2);
        let variance = centered.clone().powf_scalar(2.0).mean_dim(2);
        centered / (variance + self.epsilon).sqrt() * self.weight.val().reshape([1, 1, dim])
            + self.bias.val().reshape([1, 1, dim])
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub(super) struct SmolVlmVisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub layer_norm_eps: f64,
    pub scale_factor: usize,
}

impl Default for SmolVlmVisionConfig {
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

impl SmolVlmVisionConfig {
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
struct SmolVlmVisionEmbeddings {
    patch_embedding: Conv2d,
    position_embedding: Embedding,
}

/// 视觉嵌入层的构造参数。
#[derive(Clone, Copy, Debug)]
struct SmolVlmVisionEmbeddingsConfig {
    hidden: usize,
    patch: usize,
    grid: usize,
}

impl SmolVlmVisionEmbeddingsConfig {
    fn new(cfg: &SmolVlmVisionConfig) -> Self {
        Self {
            hidden: cfg.hidden_size,
            patch: cfg.patch_size,
            grid: cfg.image_size / cfg.patch_size,
        }
    }

    fn init(&self, device: &Device) -> SmolVlmVisionEmbeddings {
        SmolVlmVisionEmbeddings {
            patch_embedding: Conv2dConfig::new([3, self.hidden], [self.patch; 2])
                .with_stride([self.patch; 2])
                .init(device),
            position_embedding: EmbeddingConfig::new(self.grid * self.grid, self.hidden)
                .init(device),
        }
    }
}

impl SmolVlmVisionEmbeddings {
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
struct SmolVlmVisionAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    #[module(skip)]
    heads: usize,
}

/// 视觉注意力的构造参数。
#[derive(Clone, Copy, Debug)]
struct SmolVlmVisionAttentionConfig {
    hidden: usize,
    heads: usize,
}

impl SmolVlmVisionAttentionConfig {
    fn init(&self, device: &Device) -> SmolVlmVisionAttention {
        let linear = || LinearConfig::new(self.hidden, self.hidden).init(device);
        SmolVlmVisionAttention {
            q_proj: linear(),
            k_proj: linear(),
            v_proj: linear(),
            out_proj: linear(),
            heads: self.heads,
        }
    }
}

impl SmolVlmVisionAttention {
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
struct SmolVlmVisionMlp {
    fc1: Linear,
    fc2: Linear,
}

/// 视觉 MLP 的构造参数。
#[derive(Clone, Copy, Debug)]
struct SmolVlmVisionMlpConfig {
    hidden: usize,
    intermediate: usize,
}

impl SmolVlmVisionMlpConfig {
    fn init(&self, device: &Device) -> SmolVlmVisionMlp {
        SmolVlmVisionMlp {
            fc1: LinearConfig::new(self.hidden, self.intermediate).init(device),
            fc2: LinearConfig::new(self.intermediate, self.hidden).init(device),
        }
    }
}

#[derive(Module, Debug)]
struct SmolVlmVisionEncoderLayer {
    self_attn: SmolVlmVisionAttention,
    layer_norm1: SmolVlmVisionLayerNorm,
    mlp: SmolVlmVisionMlp,
    layer_norm2: SmolVlmVisionLayerNorm,
}

/// 视觉编码层的构造参数，组合注意力、MLP 与归一化子配置。
#[derive(Clone, Copy, Debug)]
struct SmolVlmVisionEncoderLayerConfig {
    attention: SmolVlmVisionAttentionConfig,
    mlp: SmolVlmVisionMlpConfig,
    norm: SmolVlmVisionLayerNormConfig,
}

impl SmolVlmVisionEncoderLayerConfig {
    fn init(&self, device: &Device) -> SmolVlmVisionEncoderLayer {
        SmolVlmVisionEncoderLayer {
            self_attn: self.attention.init(device),
            layer_norm1: self.norm.init(device),
            mlp: self.mlp.init(device),
            layer_norm2: self.norm.init(device),
        }
    }
}

impl SmolVlmVisionEncoderLayer {
    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let x = x.clone() + self.self_attn.forward(self.layer_norm1.forward(x));
        let mlp = self.mlp.fc2.forward(gelu_approximate(
            self.mlp.fc1.forward(self.layer_norm2.forward(x.clone())),
        ));
        x + mlp
    }
}

#[derive(Module, Debug)]
struct SmolVlmVisionEncoder {
    layers: Vec<SmolVlmVisionEncoderLayer>,
}

#[derive(Module, Debug)]
pub(super) struct SmolVlmVisionModel {
    embeddings: SmolVlmVisionEmbeddings,
    encoder: SmolVlmVisionEncoder,
    post_layernorm: SmolVlmVisionLayerNorm,
}

impl SmolVlmVisionConfig {
    fn layer(&self) -> SmolVlmVisionEncoderLayerConfig {
        SmolVlmVisionEncoderLayerConfig {
            attention: SmolVlmVisionAttentionConfig {
                hidden: self.hidden_size,
                heads: self.num_attention_heads,
            },
            mlp: SmolVlmVisionMlpConfig {
                hidden: self.hidden_size,
                intermediate: self.intermediate_size,
            },
            norm: SmolVlmVisionLayerNormConfig::new(self.hidden_size, self.layer_norm_eps),
        }
    }

    pub fn init(&self, device: &Device) -> SmolVlmVisionModel {
        SmolVlmVisionModel {
            embeddings: SmolVlmVisionEmbeddingsConfig::new(self).init(device),
            encoder: SmolVlmVisionEncoder {
                layers: (0..self.num_hidden_layers)
                    .map(|_| self.layer().init(device))
                    .collect(),
            },
            post_layernorm: SmolVlmVisionLayerNormConfig::new(
                self.hidden_size,
                self.layer_norm_eps,
            )
            .init(device),
        }
    }
}

impl SmolVlmVisionModel {
    pub fn forward(&self, image: Tensor<4>) -> Tensor<3> {
        let mut x = self.embeddings.forward(image);
        for layer in &self.encoder.layers {
            x = layer.forward(x);
        }
        self.post_layernorm.forward(x)
    }
}

#[derive(Module, Debug)]
struct SmolVlmModalityProjection {
    proj: Linear,
}

#[derive(Module, Debug)]
pub(super) struct SmolVlmConnector {
    modality_projection: SmolVlmModalityProjection,
    #[module(skip)]
    scale_factor: usize,
    #[module(skip)]
    grid: usize,
}

/// SmolVlmConnector 的构造参数；`text_dim` 是 VLM 文本宽度。
#[derive(Clone, Copy, Debug)]
pub(super) struct SmolVlmConnectorConfig {
    hidden: usize,
    scale_factor: usize,
    grid: usize,
    text_dim: usize,
}

impl SmolVlmConnectorConfig {
    pub(super) fn new(cfg: &SmolVlmVisionConfig, text_dim: usize) -> Self {
        Self {
            hidden: cfg.hidden_size,
            scale_factor: cfg.scale_factor,
            grid: cfg.image_size / cfg.patch_size,
            text_dim,
        }
    }

    pub(super) fn init(&self, device: &Device) -> SmolVlmConnector {
        SmolVlmConnector {
            modality_projection: SmolVlmModalityProjection {
                proj: LinearConfig::new(self.hidden * self.scale_factor.pow(2), self.text_dim)
                    .with_bias(false)
                    .init(device),
            },
            scale_factor: self.scale_factor,
            grid: self.grid,
        }
    }
}

impl SmolVlmConnector {
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
