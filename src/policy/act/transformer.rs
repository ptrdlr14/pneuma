//! ACT 的 encoder/decoder：post-norm（pre_norm=false 已在 config 层把关），
//! torch `MultiheadAttention` 的 `in_proj` 融合权重按 q|k|v 行切分等价还原。
//! 布局统一 (B, S, D)；dropout 在推理下为恒等，不建模。

use burn::module::{Module, Param};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::{relu, softmax};
use burn::tensor::module::layer_norm;
use burn::tensor::{Device, Tensor, TensorData, s};

/// torch `LayerNorm`（biased variance，eps=1e-5）。
///
/// 保留 `weight`/`bias` 字段名以对齐 `state_dict` 与原生 bpk；
/// 归一化本身调用 Burn 的原生算子。
#[derive(Module, Debug)]
pub struct LayerNorm {
    pub weight: Param<Tensor<1>>,
    pub bias: Param<Tensor<1>>,
}

impl LayerNorm {
    pub fn init(dim: usize, device: &Device) -> Self {
        Self {
            weight: Param::from_data(TensorData::new(vec![1.0f32; dim], [dim]), device),
            bias: Param::from_data(TensorData::new(vec![0.0f32; dim], [dim]), device),
        }
    }

    pub fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        layer_norm(x, self.weight.val(), Some(self.bias.val()), 1e-5)
    }
}

/// nn.MultiheadAttention：`in_proj_weight` [3D, D] 单 Linear 等价；query/key/value 分源投影。
#[derive(Module, Debug)]
pub struct MultiheadAttention {
    /// `in_proj` 键名带下划线（`in_proj_weight`），字段用点段路径 + 加载时重映射。
    pub in_proj: Linear,
    pub out_proj: Linear,
    #[module(skip)]
    pub n_heads: usize,
}

impl MultiheadAttention {
    pub fn init(dim: usize, n_heads: usize, device: &Device) -> Self {
        Self {
            in_proj: LinearConfig::new(dim, 3 * dim).init(device),
            out_proj: LinearConfig::new(dim, dim).init(device),
            n_heads,
        }
    }

    fn split_heads(&self, t: Tensor<3>) -> Tensor<4> {
        let [batch, seq, dim] = t.dims();
        let head = dim / self.n_heads;
        t.reshape([batch, seq, self.n_heads, head])
            .permute([0, 2, 1, 3])
    }

    /// query/key/value 输入不同，各自做融合投影。
    pub fn forward(&self, query: Tensor<3>, key: Tensor<3>, value: Tensor<3>) -> Tensor<3> {
        let query = self.in_proj.forward(query);
        let key = self.in_proj.forward(key);
        let value = self.in_proj.forward(value);
        self.attend(query, key, value)
    }

    /// query 与 key 输入相同：融合投影只算一次，q/k 从同一结果切分。
    pub fn self_attention(&self, query_key: Tensor<3>, value: Tensor<3>) -> Tensor<3> {
        let qk = self.in_proj.forward(query_key);
        let value = self.in_proj.forward(value);
        self.attend(qk.clone(), qk, value)
    }

    /// 输入是 `in_proj` 的输出 `[B, S, 3D]`，按 q|k|v 切分后计算注意力。
    fn attend(
        &self,
        projected_query: Tensor<3>,
        projected_key: Tensor<3>,
        projected_value: Tensor<3>,
    ) -> Tensor<3> {
        let dim = projected_query.dims()[2] / 3;
        let q = self.split_heads(projected_query.slice(s![.., .., 0..dim]));
        let k = self.split_heads(projected_key.slice(s![.., .., dim..2 * dim]));
        let v = self.split_heads(projected_value.slice(s![.., .., 2 * dim..3 * dim]));
        let head_dim = v.dims()[3];
        let scores = q.matmul(k.swap_dims(2, 3)) * (1.0 / (head_dim as f32).sqrt());
        let merged = softmax(scores, 3).matmul(v).permute([0, 2, 1, 3]);
        let [batch, seq, _, _] = merged.dims();
        self.out_proj.forward(merged.reshape([batch, seq, dim]))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TransformerLayerConfig {
    pub dim_model: usize,
    pub n_heads: usize,
    pub dim_feedforward: usize,
}

#[derive(Module, Debug)]
pub struct ACTEncoderLayer {
    pub self_attn: MultiheadAttention,
    pub linear1: Linear,
    pub linear2: Linear,
    pub norm1: LayerNorm,
    pub norm2: LayerNorm,
}

impl TransformerLayerConfig {
    pub fn init_encoder_layer(&self, device: &Device) -> ACTEncoderLayer {
        ACTEncoderLayer {
            self_attn: MultiheadAttention::init(self.dim_model, self.n_heads, device),
            linear1: LinearConfig::new(self.dim_model, self.dim_feedforward).init(device),
            linear2: LinearConfig::new(self.dim_feedforward, self.dim_model).init(device),
            norm1: LayerNorm::init(self.dim_model, device),
            norm2: LayerNorm::init(self.dim_model, device),
        }
    }
}

impl ACTEncoderLayer {
    /// x/pos 均为 (B, S, D)；对齐 python：q=k=x+pos，post-norm 残差两次。
    pub fn forward(&self, x: Tensor<3>, pos: &Tensor<3>) -> Tensor<3> {
        let skip = x.clone();
        let v = x.clone();
        let qk = x + pos.clone();
        let attn = self.self_attn.self_attention(qk, v);
        let x = self.norm1.forward(skip + attn);
        let ff = self.linear2.forward(relu(self.linear1.forward(x.clone())));
        self.norm2.forward(x + ff)
    }
}

#[derive(Module, Debug)]
pub struct ACTEncoder {
    pub layers: Vec<ACTEncoderLayer>,
}

impl ACTEncoder {
    pub fn forward(&self, x: Tensor<3>, pos: &Tensor<3>) -> Tensor<3> {
        self.layers.iter().fold(x, |acc, l| l.forward(acc, pos))
    }
}

#[derive(Module, Debug)]
pub struct ACTDecoderLayer {
    pub self_attn: MultiheadAttention,
    pub multihead_attn: MultiheadAttention,
    pub linear1: Linear,
    pub linear2: Linear,
    pub norm1: LayerNorm,
    pub norm2: LayerNorm,
    pub norm3: LayerNorm,
}

impl TransformerLayerConfig {
    pub fn init_decoder_layer(&self, device: &Device) -> ACTDecoderLayer {
        ACTDecoderLayer {
            self_attn: MultiheadAttention::init(self.dim_model, self.n_heads, device),
            multihead_attn: MultiheadAttention::init(self.dim_model, self.n_heads, device),
            linear1: LinearConfig::new(self.dim_model, self.dim_feedforward).init(device),
            linear2: LinearConfig::new(self.dim_feedforward, self.dim_model).init(device),
            norm1: LayerNorm::init(self.dim_model, device),
            norm2: LayerNorm::init(self.dim_model, device),
            norm3: LayerNorm::init(self.dim_model, device),
        }
    }
}

impl ACTDecoderLayer {
    /// `x`/`dec_pos` (B,DS,D)；`enc`/`enc_pos` (B,ES,D)。对齐 python decoder 三层残差。
    pub fn forward(
        &self,
        x: Tensor<3>,
        dec_pos: &Tensor<3>,
        enc: Tensor<3>,
        enc_pos: &Tensor<3>,
    ) -> Tensor<3> {
        let skip = x.clone();
        let v = x.clone();
        let qk = x + dec_pos.clone();
        let x = self
            .norm1
            .forward(skip + self.self_attn.self_attention(qk, v));

        let skip = x.clone();
        let query = x + dec_pos.clone();
        let key = enc.clone() + enc_pos.clone();
        let x = self
            .norm2
            .forward(skip + self.multihead_attn.forward(query, key, enc));

        let skip = x.clone();
        let ff = self.linear2.forward(relu(self.linear1.forward(x)));
        self.norm3.forward(skip + ff)
    }
}

#[derive(Module, Debug)]
pub struct ACTDecoder {
    pub layers: Vec<ACTDecoderLayer>,
    pub norm: LayerNorm,
}

impl ACTDecoder {
    pub fn forward(
        &self,
        x: Tensor<3>,
        dec_pos: &Tensor<3>,
        enc: &Tensor<3>,
        enc_pos: &Tensor<3>,
    ) -> Tensor<3> {
        let x = self
            .layers
            .iter()
            .fold(x, |acc, l| l.forward(acc, dec_pos, enc.clone(), enc_pos));
        self.norm.forward(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Tolerance;

    /// self-attention 复用一次融合投影，结果必须与通用路径一致。
    #[test]
    fn fused_self_attention_matches_general_path() {
        let device = Device::flex();
        let attention = MultiheadAttention::init(8, 2, &device);
        let query = Tensor::<3>::from_data(TensorData::new(vec![0.5f32; 24], [1, 3, 8]), &device);
        let value = Tensor::<3>::from_data(TensorData::new(vec![-0.25f32; 24], [1, 3, 8]), &device);
        let expected = attention
            .forward(query.clone(), query.clone(), value.clone())
            .into_data();
        let actual = attention.self_attention(query, value).into_data();
        actual.assert_approx_eq::<f32>(&expected, Tolerance::default());
    }
}
