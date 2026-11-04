//! SmolVLA 的 Llama 层：前缀缓存与动作专家共享 head_dim，交替执行自注意力和交叉注意力。

use burn::module::{Module, Param};
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::tensor::activation::{silu, softmax};
use burn::tensor::{Bool, Device, Int, Tensor, TensorData, s};
use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
pub(super) struct SmolVlmTextConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub vocab_size: usize,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub mlp_bias: bool,
}

impl SmolVlmTextConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.hidden_size > 0
                && self.intermediate_size > 0
                && self.num_hidden_layers > 0
                && self.num_attention_heads > 0
                && self.num_key_value_heads > 0
                && self.head_dim > 0
                && self.head_dim.is_multiple_of(2)
                && self
                    .num_attention_heads
                    .is_multiple_of(self.num_key_value_heads)
                && self.vocab_size > 0
                && self.rms_norm_eps.is_finite()
                && self.rms_norm_eps > 0.0,
            "invalid SmolVLA text configuration"
        );
        Ok(())
    }
}

/// 半维旋转的构造参数；位置表在 init 时一次性生成。
///
/// 与 Burn `RotaryEncoding` 的相邻配对约定不同，这里保持官方 Gemma 的前后半区约定。
#[derive(Clone, Copy, Debug)]
pub(super) struct SmolVlmTextRotaryConfig {
    max_sequence_length: usize,
    head_dim: usize,
    theta: f32,
}

impl SmolVlmTextRotaryConfig {
    pub(super) fn new(max_sequence_length: usize, head_dim: usize) -> Self {
        Self {
            max_sequence_length,
            head_dim,
            theta: 10_000.0,
        }
    }

    pub(super) fn init(&self, device: &Device) -> SmolVlmTextRotary {
        let half = self.head_dim / 2;
        let denominators: Vec<f32> = (0..half)
            .map(|i| self.theta.powf(2.0 * i as f32 / self.head_dim as f32))
            .collect();
        let mut radians = Vec::with_capacity(self.max_sequence_length * half);
        for position in 0..self.max_sequence_length {
            radians.extend(denominators.iter().map(|d| position as f32 / d));
        }
        let angles = Tensor::<2>::from_data(
            TensorData::new(radians, [self.max_sequence_length, half]),
            device,
        );
        SmolVlmTextRotary {
            cos: angles.clone().cos(),
            sin: angles.sin(),
        }
    }
}

/// 预计算的半维旋转表；表是常量，不进入权重记录。
#[derive(Module, Debug)]
pub(super) struct SmolVlmTextRotary {
    cos: Tensor<2>,
    sin: Tensor<2>,
}

impl SmolVlmTextRotary {
    fn apply(&self, x: Tensor<4>, positions: &Tensor<1, Int>) -> Tensor<4> {
        let [_, _, seq, width] = x.dims();
        let half = width / 2;
        let cos = self
            .cos
            .clone()
            .select(0, positions.clone())
            .reshape([1, 1, seq, half]);
        let sin = self
            .sin
            .clone()
            .select(0, positions.clone())
            .reshape([1, 1, seq, half]);
        let first = x.clone().slice(s![.., .., .., ..half]);
        let second = x.slice(s![.., .., .., half..]);
        Tensor::cat(
            vec![
                first.clone() * cos.clone() - second.clone() * sin.clone(),
                second * cos + first * sin,
            ],
            3,
        )
    }
}

#[derive(Module, Debug)]
pub(super) struct SmolVlmTextRmsNorm {
    weight: Param<Tensor<1>>,
    #[module(skip)]
    epsilon: f64,
}

/// SmolVlmTextRmsNorm 的构造参数。
#[derive(Clone, Copy, Debug)]
struct SmolVlmTextRmsNormConfig {
    dim: usize,
    epsilon: f64,
}

impl SmolVlmTextRmsNormConfig {
    fn new(dim: usize, epsilon: f64) -> Self {
        Self { dim, epsilon }
    }

    fn init(&self, device: &Device) -> SmolVlmTextRmsNorm {
        SmolVlmTextRmsNorm {
            weight: Param::from_tensor(Tensor::ones([self.dim], device)),
            epsilon: self.epsilon,
        }
    }
}

impl SmolVlmTextRmsNorm {
    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let dim = x.dims()[2];
        let rms = (x.clone().powf_scalar(2.0).mean_dim(2) + self.epsilon).sqrt();
        x / rms * self.weight.val().reshape([1, 1, dim])
    }
}

#[derive(Module, Debug)]
struct SmolVlmTextMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

/// MLP 的构造参数。
#[derive(Clone, Copy, Debug)]
struct SmolVlmTextMlpConfig {
    hidden: usize,
    intermediate: usize,
    bias: bool,
}

impl SmolVlmTextMlpConfig {
    fn init(&self, device: &Device) -> SmolVlmTextMlp {
        let linear = |input, output| {
            LinearConfig::new(input, output)
                .with_bias(self.bias)
                .init(device)
        };
        SmolVlmTextMlp {
            gate_proj: linear(self.hidden, self.intermediate),
            up_proj: linear(self.hidden, self.intermediate),
            down_proj: linear(self.intermediate, self.hidden),
        }
    }
}

impl SmolVlmTextMlp {
    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        self.down_proj
            .forward(silu(self.gate_proj.forward(x.clone())) * self.up_proj.forward(x))
    }
}

#[derive(Module, Debug)]
struct SmolVlmTextAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    #[module(skip)]
    heads: usize,
    #[module(skip)]
    kv_heads: usize,
    #[module(skip)]
    head_dim: usize,
}

/// 注意力层的构造参数；`cross` 决定 K/V 的输入维度。
#[derive(Clone, Copy, Debug)]
struct SmolVlmTextAttentionConfig {
    hidden: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    bias: bool,
    cross: bool,
}

impl SmolVlmTextAttentionConfig {
    fn init(&self, device: &Device) -> SmolVlmTextAttention {
        let q_dim = self.heads * self.head_dim;
        let kv_dim = self.kv_heads * self.head_dim;
        let kv_input = if self.cross { kv_dim } else { self.hidden };
        let linear = |input, output| {
            LinearConfig::new(input, output)
                .with_bias(self.bias)
                .init(device)
        };
        SmolVlmTextAttention {
            q_proj: linear(self.hidden, q_dim),
            k_proj: linear(kv_input, kv_dim),
            v_proj: linear(kv_input, kv_dim),
            o_proj: linear(q_dim, self.hidden),
            heads: self.heads,
            kv_heads: self.kv_heads,
            head_dim: self.head_dim,
        }
    }
}

impl SmolVlmTextAttention {
    fn heads(&self, x: Tensor<3>, count: usize) -> Tensor<4> {
        let [batch, seq, _] = x.dims();
        x.reshape([batch, seq, count, self.head_dim])
            .permute([0, 2, 1, 3])
    }

    fn query(
        &self,
        x: Tensor<3>,
        rotary: &SmolVlmTextRotary,
        positions: &Tensor<1, Int>,
    ) -> Tensor<4> {
        rotary.apply(self.heads(self.q_proj.forward(x), self.heads), positions)
    }

    fn kv(
        &self,
        x: Tensor<3>,
        rotary: &SmolVlmTextRotary,
        positions: &Tensor<1, Int>,
    ) -> SmolVlmKvCache {
        SmolVlmKvCache {
            key: rotary.apply(
                self.heads(self.k_proj.forward(x.clone()), self.kv_heads),
                positions,
            ),
            value: self.heads(self.v_proj.forward(x), self.kv_heads),
        }
    }

    fn cross_kv(&self, prefix: &SmolVlmKvCache) -> SmolVlmKvCache {
        let flatten = |x: Tensor<4>| {
            let [batch, heads, seq, width] = x.dims();
            x.permute([0, 2, 1, 3]).reshape([batch, seq, heads * width])
        };
        SmolVlmKvCache {
            key: self.heads(
                self.k_proj.forward(flatten(prefix.key.clone())),
                self.kv_heads,
            ),
            value: self.heads(
                self.v_proj.forward(flatten(prefix.value.clone())),
                self.kv_heads,
            ),
        }
    }

    fn attend(&self, query: Tensor<4>, kv: &SmolVlmKvCache, blocked: Tensor<4, Bool>) -> Tensor<3> {
        let [batch, _, seq, _] = query.dims();
        let groups = self.heads / self.kv_heads;
        let repeat = |x: Tensor<4>| {
            let [batch, heads, seq, width] = x.dims();
            x.reshape([batch, heads, 1, seq, width])
                .expand([batch, heads, groups, seq, width])
                .reshape([batch, heads * groups, seq, width])
        };
        let scores =
            query.matmul(repeat(kv.key.clone()).swap_dims(2, 3)) / (self.head_dim as f64).sqrt();
        let probabilities = softmax(scores.mask_fill(blocked, f32::MIN), 3);
        probabilities
            .matmul(repeat(kv.value.clone()))
            .permute([0, 2, 1, 3])
            .reshape([batch, seq, self.heads * self.head_dim])
    }
}

#[derive(Module, Debug)]
struct SmolVlmTextDecoderLayer {
    self_attn: SmolVlmTextAttention,
    mlp: SmolVlmTextMlp,
    input_layernorm: SmolVlmTextRmsNorm,
    post_attention_layernorm: SmolVlmTextRmsNorm,
}

/// 解码层的构造参数，组合注意力和 MLP 的子配置。
#[derive(Clone, Copy, Debug)]
struct SmolVlmTextDecoderLayerConfig {
    attention: SmolVlmTextAttentionConfig,
    mlp: SmolVlmTextMlpConfig,
    norm: SmolVlmTextRmsNormConfig,
}

impl SmolVlmTextDecoderLayerConfig {
    fn init(&self, device: &Device) -> SmolVlmTextDecoderLayer {
        SmolVlmTextDecoderLayer {
            self_attn: self.attention.init(device),
            mlp: self.mlp.init(device),
            input_layernorm: self.norm.init(device),
            post_attention_layernorm: self.norm.init(device),
        }
    }
}

impl SmolVlmTextDecoderLayer {
    fn finish(&self, residual: Tensor<3>, attended: Tensor<3>) -> Tensor<3> {
        let x = residual + self.self_attn.o_proj.forward(attended);
        x.clone() + self.mlp.forward(self.post_attention_layernorm.forward(x))
    }
}

#[derive(Module, Debug)]
pub(super) struct SmolVlmTextModel {
    pub embed_tokens: Embedding,
    layers: Vec<SmolVlmTextDecoderLayer>,
    norm: SmolVlmTextRmsNorm,
    rotary: SmolVlmTextRotary,
}

#[derive(Clone, Debug)]
pub(super) struct SmolVlmKvCache {
    pub key: Tensor<4>,
    pub value: Tensor<4>,
}

impl SmolVlmTextConfig {
    /// 解码层的构造参数；`cross` 为真时 K/V 投影接收前缀宽度。
    fn decoder(&self, cross: bool) -> SmolVlmTextDecoderLayerConfig {
        SmolVlmTextDecoderLayerConfig {
            attention: SmolVlmTextAttentionConfig {
                hidden: self.hidden_size,
                heads: self.num_attention_heads,
                kv_heads: self.num_key_value_heads,
                head_dim: self.head_dim,
                bias: self.attention_bias,
                cross,
            },
            mlp: SmolVlmTextMlpConfig {
                hidden: self.hidden_size,
                intermediate: self.intermediate_size,
                bias: self.mlp_bias,
            },
            norm: SmolVlmTextRmsNormConfig::new(self.hidden_size, self.rms_norm_eps),
        }
    }

    pub fn init(&self, max_sequence_length: usize, device: &Device) -> SmolVlmTextModel {
        SmolVlmTextModel {
            embed_tokens: EmbeddingConfig::new(self.vocab_size, self.hidden_size).init(device),
            layers: (0..self.num_hidden_layers)
                .map(|_| self.decoder(false).init(device))
                .collect(),
            norm: SmolVlmTextRmsNormConfig::new(self.hidden_size, self.rms_norm_eps).init(device),
            rotary: SmolVlmTextRotaryConfig::new(max_sequence_length, self.head_dim).init(device),
        }
    }
}

impl SmolVlmTextModel {
    pub fn prefill(
        &self,
        mut x: Tensor<3>,
        valid: &[bool],
        groups: &[usize],
    ) -> Vec<SmolVlmKvCache> {
        let positions = position_tensor(valid, &x.device());
        let blocked = attention_mask(valid, groups, &x.device());
        let mut cache = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let normalized = layer.input_layernorm.forward(x.clone());
            let query = layer
                .self_attn
                .query(normalized.clone(), &self.rotary, &positions);
            let kv = layer.self_attn.kv(normalized, &self.rotary, &positions);
            let attended = layer.self_attn.attend(query, &kv, blocked.clone());
            x = layer.finish(x, attended);
            cache.push(kv);
        }
        cache
    }
}

#[derive(Module, Debug)]
pub(super) struct SmolVlmExpertModel {
    layers: Vec<SmolVlmTextDecoderLayer>,
    norm: SmolVlmTextRmsNorm,
    rotary: SmolVlmTextRotary,
    #[module(skip)]
    cross_layers: Vec<bool>,
}

/// 动作专家的构造参数；`cross_layers` 指定每层是否使用交叉注意力。
#[derive(Clone, Debug)]
pub(super) struct SmolVlmExpertConfig {
    text: SmolVlmTextConfig,
    cross_layers: Vec<bool>,
    max_sequence_length: usize,
}

impl SmolVlmExpertConfig {
    pub(super) fn new(
        text: SmolVlmTextConfig,
        cross_layers: Vec<bool>,
        max_sequence_length: usize,
    ) -> Self {
        Self {
            text,
            cross_layers,
            max_sequence_length,
        }
    }

    pub(super) fn init(&self, device: &Device) -> SmolVlmExpertModel {
        SmolVlmExpertModel {
            layers: self
                .cross_layers
                .iter()
                .map(|&cross| self.text.decoder(cross).init(device))
                .collect(),
            norm: SmolVlmTextRmsNormConfig::new(self.text.hidden_size, self.text.rms_norm_eps)
                .init(device),
            rotary: SmolVlmTextRotaryConfig::new(self.max_sequence_length, self.text.head_dim)
                .init(device),
            cross_layers: self.cross_layers.clone(),
        }
    }
}

/// 每层专家缓存：交叉层只保留前缀投影，自注意力层保留可写缓冲。
enum SmolVlmExpertLayerCache {
    Full(SmolVlmKvCache),
    Autoregressive { prefix: usize, kv: SmolVlmKvCache },
}

/// 一次去噪循环复用的全部专家状态。
pub(super) struct SmolVlmExpertCache {
    layers: Vec<SmolVlmExpertLayerCache>,
    cross_mask: Tensor<4, Bool>,
    self_mask: Tensor<4, Bool>,
    absolute: Tensor<1, Int>,
    relative: Tensor<1, Int>,
}

impl SmolVlmExpertModel {
    /// 构建一次请求的缓存：前缀投影、掩码与位置只算一次。
    pub(super) fn cache(
        &self,
        prefix: &[SmolVlmKvCache],
        valid: &[bool],
        steps: usize,
    ) -> SmolVlmExpertCache {
        let device = prefix[0].key.device();
        let prefix_len = prefix[0].key.dims()[2];
        let offset = valid.iter().filter(|&&v| v).count();
        let layers = self
            .layers
            .iter()
            .zip(prefix)
            .enumerate()
            .map(|(index, (layer, kv))| {
                if self.cross_layers[index] {
                    SmolVlmExpertLayerCache::Full(layer.self_attn.cross_kv(kv))
                } else {
                    let [batch, heads, _, width] = kv.key.dims();
                    let zeros = Tensor::zeros([batch, heads, steps, width], &device);
                    SmolVlmExpertLayerCache::Autoregressive {
                        prefix: prefix_len,
                        kv: SmolVlmKvCache {
                            key: Tensor::cat(vec![kv.key.clone(), zeros.clone()], 2),
                            value: Tensor::cat(vec![kv.value.clone(), zeros], 2),
                        },
                    }
                }
            })
            .collect();
        let cross_blocked: Vec<_> = (0..steps).flat_map(|_| valid.iter().map(|&v| !v)).collect();
        let cross_mask = Tensor::<4, Bool>::from_data(
            TensorData::new(cross_blocked, [1, 1, steps, valid.len()]),
            &device,
        );
        let self_blocked: Vec<_> = (0..steps)
            .flat_map(|query| {
                valid
                    .iter()
                    .map(|&v| !v)
                    .chain((0..steps).map(move |key| key > query))
            })
            .collect();
        let self_mask = Tensor::<4, Bool>::from_data(
            TensorData::new(self_blocked, [1, 1, steps, valid.len() + steps]),
            &device,
        );
        SmolVlmExpertCache {
            layers,
            cross_mask,
            self_mask,
            absolute: position_range(offset, steps, &device),
            relative: position_range(0, steps, &device),
        }
    }

    pub(super) fn forward(&self, mut x: Tensor<3>, cache: &mut SmolVlmExpertCache) -> Tensor<3> {
        let SmolVlmExpertCache {
            layers,
            cross_mask,
            self_mask,
            absolute,
            relative,
        } = cache;
        for (layer, layer_cache) in self.layers.iter().zip(layers.iter_mut()) {
            let normalized = layer.input_layernorm.forward(x.clone());
            let (query, kv, mask) = match layer_cache {
                SmolVlmExpertLayerCache::Full(kv) => (
                    layer.self_attn.query(normalized, &self.rotary, relative),
                    &*kv,
                    cross_mask.clone(),
                ),
                SmolVlmExpertLayerCache::Autoregressive { prefix, kv } => {
                    let query = layer
                        .self_attn
                        .query(normalized.clone(), &self.rotary, absolute);
                    let action = layer.self_attn.kv(normalized, &self.rotary, absolute);
                    let start = *prefix;
                    let end = start + action.key.dims()[2];
                    kv.key.inplace(|t| {
                        t.slice_assign(s![.., .., start..end, ..], action.key.clone())
                    });
                    kv.value.inplace(|t| {
                        t.slice_assign(s![.., .., start..end, ..], action.value.clone())
                    });
                    (query, &*kv, self_mask.clone())
                }
            };
            let attended = layer.self_attn.attend(query, kv, mask);
            x = layer.finish(x, attended);
        }
        self.norm.forward(x)
    }
}

fn positions(valid: &[bool]) -> Vec<usize> {
    let mut count = 0usize;
    valid
        .iter()
        .map(|&v| {
            count += usize::from(v);
            count.saturating_sub(1)
        })
        .collect()
}

/// 前缀位置由有效 token 计数给出；填充 token 复用最后一个有效位置。
fn position_tensor(valid: &[bool], device: &Device) -> Tensor<1, Int> {
    let values: Vec<i64> = positions(valid).into_iter().map(|p| p as i64).collect();
    Tensor::<1, Int>::from_data(TensorData::new(values, [valid.len()]), device)
}

/// 连续位置区间；专家自注意力使用绝对位置，交叉注意力使用从 0 开始的相对位置。
fn position_range(start: usize, length: usize, device: &Device) -> Tensor<1, Int> {
    let values: Vec<i64> = (start..start + length).map(|p| p as i64).collect();
    Tensor::<1, Int>::from_data(TensorData::new(values, [length]), device)
}

fn attention_mask(valid: &[bool], groups: &[usize], device: &Device) -> Tensor<4, Bool> {
    let seq = valid.len();
    let blocked: Vec<_> = (0..seq)
        .flat_map(|query| {
            (0..seq).map(move |key| !valid[query] || !valid[key] || groups[key] > groups[query])
        })
        .collect();
    Tensor::from_data(TensorData::new(blocked, [1, 1, seq, seq]), device)
}
