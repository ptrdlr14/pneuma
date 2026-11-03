//! PaliGemma 的 SigLIP-So400m 视觉塔。

use burn::module::{Module, Param};
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::tensor::activation::{gelu_approximate, softmax};
use burn::tensor::{Int, Tensor, TensorData};

/// 官方 PaliGemma-3B 的视觉塔常量；checkpoint 不携带这些字段。
#[derive(Clone, Copy, Debug)]
pub(super) struct VisionSpec {
    pub hidden: usize,
    pub mlp: usize,
    pub layers: usize,
    pub heads: usize,
    pub patch: usize,
    pub image: usize,
    pub epsilon: f64,
}

pub(super) const SIGLIP: VisionSpec = VisionSpec {
    hidden: 1152,
    mlp: 4304,
    layers: 27,
    heads: 16,
    patch: 14,
    image: 224,
    epsilon: 1e-6,
};

impl VisionSpec {
    pub(super) fn tokens(&self) -> usize {
        (self.image / self.patch).pow(2)
    }
}

#[derive(Module, Debug)]
pub(super) struct LayerNorm {
    weight: Param<Tensor<1>>,
    bias: Param<Tensor<1>>,
    #[module(skip)]
    epsilon: f64,
}

impl LayerNorm {
    pub(super) fn new(dim: usize, epsilon: f64, device: &burn::tensor::Device) -> Self {
        Self {
            weight: Param::from_tensor(Tensor::ones([dim], device)),
            bias: Param::from_tensor(Tensor::zeros([dim], device)),
            epsilon,
        }
    }

    pub(super) fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let dim = x.dims()[2];
        let centered = x.clone() - x.mean_dim(2);
        let variance = centered.clone().powf_scalar(2.0).mean_dim(2);
        centered / (variance + self.epsilon).sqrt() * self.weight.val().reshape([1, 1, dim])
            + self.bias.val().reshape([1, 1, dim])
    }
}

#[derive(Module, Debug)]
struct VisionEmbeddings {
    patch_embedding: Conv2d,
    position_embedding: Embedding,
}

impl VisionEmbeddings {
    fn init(spec: &VisionSpec, device: &burn::tensor::Device) -> Self {
        Self {
            patch_embedding: Conv2dConfig::new([3, spec.hidden], [spec.patch; 2])
                .with_stride([spec.patch; 2])
                .init(device),
            position_embedding: EmbeddingConfig::new(spec.tokens(), spec.hidden).init(device),
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
    fn init(spec: &VisionSpec, device: &burn::tensor::Device) -> Self {
        let linear = || LinearConfig::new(spec.hidden, spec.hidden).init(device);
        Self {
            q_proj: linear(),
            k_proj: linear(),
            v_proj: linear(),
            out_proj: linear(),
            heads: spec.heads,
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
    fn init(spec: &VisionSpec, device: &burn::tensor::Device) -> Self {
        Self {
            self_attn: VisionAttention::init(spec, device),
            layer_norm1: LayerNorm::new(spec.hidden, spec.epsilon, device),
            mlp: VisionMlp {
                fc1: LinearConfig::new(spec.hidden, spec.mlp).init(device),
                fc2: LinearConfig::new(spec.mlp, spec.hidden).init(device),
            },
            layer_norm2: LayerNorm::new(spec.hidden, spec.epsilon, device),
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
struct VisionModel {
    embeddings: VisionEmbeddings,
    encoder: VisionEncoder,
    post_layernorm: LayerNorm,
}

#[derive(Module, Debug)]
pub(super) struct VisionTower {
    vision_model: VisionModel,
}

impl VisionTower {
    pub(super) fn init(spec: &VisionSpec, device: &burn::tensor::Device) -> Self {
        Self {
            vision_model: VisionModel {
                embeddings: VisionEmbeddings::init(spec, device),
                encoder: VisionEncoder {
                    layers: (0..spec.layers)
                        .map(|_| VisionEncoderLayer::init(spec, device))
                        .collect(),
                },
                post_layernorm: LayerNorm::new(spec.hidden, spec.epsilon, device),
            },
        }
    }

    /// 输入为 `[1,3,224,224]` 的 `[-1,1]` 图像；输出 `[1,256,1152]`。
    pub(super) fn forward(&self, image: Tensor<4>) -> Tensor<3> {
        let mut x = self.vision_model.embeddings.forward(image);
        for layer in &self.vision_model.encoder.layers {
            x = layer.forward(x);
        }
        self.vision_model.post_layernorm.forward(x)
    }
}
