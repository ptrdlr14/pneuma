//! ACT 的 encoder/decoder：post-norm（pre_norm=false 已在 config 层把关），
//! torch MultiheadAttention 的 in_proj 融合权重按 q|k|v 行切分等价还原。
//! 布局统一 (B, S, D)；dropout 在推理下为恒等，不建模。

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::{relu, softmax};
use burn::tensor::{Device, Tensor, TensorData};

/// torch LayerNorm（biased variance，eps=1e-5），weight/bias 键名直接对齐 state_dict。
#[derive(Module, Debug)]
pub struct LayerNorm {
    pub weight: burn::module::Param<burn::tensor::Tensor<1>>,
    pub bias: burn::module::Param<burn::tensor::Tensor<1>>,
}

impl LayerNorm {
    pub fn init(dim: usize, device: &Device) -> Self {
        Self {
            weight: burn::module::Param::from_data(
                TensorData::new(vec![1.0f32; dim], [dim]),
                device,
            ),
            bias: burn::module::Param::from_data(TensorData::new(vec![0.0f32; dim], [dim]), device),
        }
    }

    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let dim = x.dims()[2];
        let mean = x.clone().mean_dim(2);
        let centered = x - mean;
        let var = centered.clone().powf_scalar(2.0).mean_dim(2);
        let normed = centered / (var + 1e-5).sqrt();
        // rank-1 参数 reshape 后按 dim=0/1 广播。
        normed * self.weight.val().reshape([1, 1, dim]) + self.bias.val().reshape([1, 1, dim])
    }
}

/// nn.MultiheadAttention：in_proj_weight [3D, D] 单 Linear 等价；query/key/value 分源投影。
#[derive(Module, Debug)]
pub struct MultiheadAttention {
    /// in_proj 键名带下划线（in_proj_weight），字段用点段路径 + 加载时重映射。
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

    pub fn forward(&self, query: Tensor<3>, key: Tensor<3>, value: Tensor<3>) -> Tensor<3> {
        let dim = query.dims()[2];
        use burn::tensor::s;
        let q = self.in_proj.forward(query).slice(s![.., .., 0..dim]);
        let k = self
            .in_proj
            .forward(key.clone())
            .slice(s![.., .., dim..2 * dim]);
        let v = self
            .in_proj
            .forward(value)
            .slice(s![.., .., 2 * dim..3 * dim]);

        let q = self.split_heads(q);
        let k = self.split_heads(k);
        let v = self.split_heads(v);
        let head_dim = v.dims()[3];
        let scale = 1.0 / (head_dim as f32).sqrt();
        let scores = q.matmul(k.swap_dims(2, 3)) * scale;
        let attn = softmax(scores, 3);
        let merged = attn.matmul(v).permute([0, 2, 1, 3]);
        let [batch, seq, _, _] = merged.dims();
        self.out_proj.forward(merged.reshape([batch, seq, dim]))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct EncoderLayerConfig {
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

impl EncoderLayerConfig {
    pub fn init(&self, device: &Device) -> ACTEncoderLayer {
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
        let attn = self.self_attn.forward(qk.clone(), qk, v);
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

impl EncoderLayerConfig {
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
    /// x/dec_pos (B,DS,D)；enc/enc_pos (B,ES,D)。对齐 python decoder 三层残差。
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
            .forward(skip + self.self_attn.forward(qk.clone(), qk, v));

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
        enc: Tensor<3>,
        enc_pos: &Tensor<3>,
    ) -> Tensor<3> {
        let x = self
            .layers
            .iter()
            .fold(x, |acc, l| l.forward(acc, dec_pos, enc.clone(), enc_pos));
        self.norm.forward(x)
    }
}
