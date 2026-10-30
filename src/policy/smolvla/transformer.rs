//! SmolVLA 的 Llama 层：前缀缓存与动作专家共享 head_dim，交替执行自注意力和交叉注意力。

use burn::module::{Module, Param};
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::tensor::activation::{silu, softmax};
use burn::tensor::{Bool, Device, Tensor, TensorData, s};
use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
pub(super) struct TextConfig {
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

impl TextConfig {
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

#[derive(Module, Debug)]
pub(super) struct RmsNorm {
    weight: Param<Tensor<1>>,
    #[module(skip)]
    epsilon: f64,
}

impl RmsNorm {
    fn init(dim: usize, epsilon: f64, device: &Device) -> Self {
        Self {
            weight: Param::from_tensor(Tensor::ones([dim], device)),
            epsilon,
        }
    }

    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let dim = x.dims()[2];
        let rms = (x.clone().powf_scalar(2.0).mean_dim(2) + self.epsilon).sqrt();
        x / rms * self.weight.val().reshape([1, 1, dim])
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
    pub fn init(dim: usize, epsilon: f64, device: &Device) -> Self {
        Self {
            weight: Param::from_tensor(Tensor::ones([dim], device)),
            bias: Param::from_tensor(Tensor::zeros([dim], device)),
            epsilon,
        }
    }

    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let dim = x.dims()[2];
        let centered = x.clone() - x.mean_dim(2);
        let variance = centered.clone().powf_scalar(2.0).mean_dim(2);
        centered / (variance + self.epsilon).sqrt() * self.weight.val().reshape([1, 1, dim])
            + self.bias.val().reshape([1, 1, dim])
    }
}

#[derive(Module, Debug)]
struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    fn init(cfg: &TextConfig, device: &Device) -> Self {
        let linear = |input, output| {
            LinearConfig::new(input, output)
                .with_bias(cfg.mlp_bias)
                .init(device)
        };
        Self {
            gate_proj: linear(cfg.hidden_size, cfg.intermediate_size),
            up_proj: linear(cfg.hidden_size, cfg.intermediate_size),
            down_proj: linear(cfg.intermediate_size, cfg.hidden_size),
        }
    }

    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        self.down_proj
            .forward(silu(self.gate_proj.forward(x.clone())) * self.up_proj.forward(x))
    }
}

#[derive(Module, Debug)]
struct Attention {
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

impl Attention {
    fn init(cfg: &TextConfig, cross: bool, device: &Device) -> Self {
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        let kv_input = if cross { kv_dim } else { cfg.hidden_size };
        let linear = |input, output| {
            LinearConfig::new(input, output)
                .with_bias(cfg.attention_bias)
                .init(device)
        };
        Self {
            q_proj: linear(cfg.hidden_size, q_dim),
            k_proj: linear(kv_input, kv_dim),
            v_proj: linear(kv_input, kv_dim),
            o_proj: linear(q_dim, cfg.hidden_size),
            heads: cfg.num_attention_heads,
            kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
        }
    }

    fn heads(&self, x: Tensor<3>, count: usize) -> Tensor<4> {
        let [batch, seq, _] = x.dims();
        x.reshape([batch, seq, count, self.head_dim])
            .permute([0, 2, 1, 3])
    }

    fn query(&self, x: Tensor<3>, positions: &[usize]) -> Tensor<4> {
        rope(self.heads(self.q_proj.forward(x), self.heads), positions)
    }

    fn kv(&self, x: Tensor<3>, positions: &[usize]) -> KvCache {
        KvCache {
            key: rope(
                self.heads(self.k_proj.forward(x.clone()), self.kv_heads),
                positions,
            ),
            value: self.heads(self.v_proj.forward(x), self.kv_heads),
        }
    }

    fn cross_kv(&self, prefix: &KvCache) -> KvCache {
        let flatten = |x: Tensor<4>| {
            let [batch, heads, seq, width] = x.dims();
            x.permute([0, 2, 1, 3]).reshape([batch, seq, heads * width])
        };
        KvCache {
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

    fn attend(&self, query: Tensor<4>, kv: &KvCache, blocked: Tensor<4, Bool>) -> Tensor<3> {
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
struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn init(cfg: &TextConfig, cross: bool, device: &Device) -> Self {
        Self {
            self_attn: Attention::init(cfg, cross, device),
            mlp: Mlp::init(cfg, device),
            input_layernorm: RmsNorm::init(cfg.hidden_size, cfg.rms_norm_eps, device),
            post_attention_layernorm: RmsNorm::init(cfg.hidden_size, cfg.rms_norm_eps, device),
        }
    }

    fn finish(&self, residual: Tensor<3>, attended: Tensor<3>) -> Tensor<3> {
        let x = residual + self.self_attn.o_proj.forward(attended);
        x.clone() + self.mlp.forward(self.post_attention_layernorm.forward(x))
    }
}

#[derive(Module, Debug)]
pub(super) struct TextModel {
    pub embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
}

#[derive(Clone, Debug)]
pub(super) struct KvCache {
    pub key: Tensor<4>,
    pub value: Tensor<4>,
}

impl TextModel {
    pub fn init(cfg: &TextConfig, device: &Device) -> Self {
        Self {
            embed_tokens: EmbeddingConfig::new(cfg.vocab_size, cfg.hidden_size).init(device),
            layers: (0..cfg.num_hidden_layers)
                .map(|_| DecoderLayer::init(cfg, false, device))
                .collect(),
            norm: RmsNorm::init(cfg.hidden_size, cfg.rms_norm_eps, device),
        }
    }

    pub fn prefill(&self, mut x: Tensor<3>, valid: &[bool], groups: &[usize]) -> Vec<KvCache> {
        let positions = positions(valid);
        let blocked = attention_mask(valid, groups, &x.device());
        let mut cache = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let normalized = layer.input_layernorm.forward(x.clone());
            let query = layer.self_attn.query(normalized.clone(), &positions);
            let kv = layer.self_attn.kv(normalized, &positions);
            let attended = layer.self_attn.attend(query, &kv, blocked.clone());
            x = layer.finish(x, attended);
            cache.push(kv);
        }
        cache
    }
}

#[derive(Module, Debug)]
pub(super) struct ExpertModel {
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    #[module(skip)]
    cross_layers: Vec<bool>,
}

impl ExpertModel {
    pub fn init(cfg: &TextConfig, cross_layers: Vec<bool>, device: &Device) -> Self {
        Self {
            layers: cross_layers
                .iter()
                .map(|&cross| DecoderLayer::init(cfg, cross, device))
                .collect(),
            norm: RmsNorm::init(cfg.hidden_size, cfg.rms_norm_eps, device),
            cross_layers,
        }
    }

    pub fn forward(&self, mut x: Tensor<3>, prefix: &[KvCache], valid: &[bool]) -> Tensor<3> {
        let steps = x.dims()[1];
        let offset = valid.iter().filter(|&&v| v).count();
        let absolute: Vec<_> = (offset..offset + steps).collect();
        let relative: Vec<_> = (0..steps).collect();
        let device = x.device();
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
        for (index, (layer, cache)) in self.layers.iter().zip(prefix).enumerate() {
            let normalized = layer.input_layernorm.forward(x.clone());
            let (query, kv, mask) = if self.cross_layers[index] {
                (
                    layer.self_attn.query(normalized, &relative),
                    layer.self_attn.cross_kv(cache),
                    cross_mask.clone(),
                )
            } else {
                let query = layer.self_attn.query(normalized.clone(), &absolute);
                let action = layer.self_attn.kv(normalized, &absolute);
                let kv = KvCache {
                    key: Tensor::cat(vec![cache.key.clone(), action.key], 2),
                    value: Tensor::cat(vec![cache.value.clone(), action.value], 2),
                };
                (query, kv, self_mask.clone())
            };
            let attended = layer.self_attn.attend(query, &kv, mask);
            x = layer.finish(x, attended);
        }
        self.norm.forward(x)
    }
}

/// 官方 SmolVLA 使用固定 10000 的半维旋转，独立于 VLM config 的 rope_theta。
fn rope(x: Tensor<4>, positions: &[usize]) -> Tensor<4> {
    let [_, _, seq, width] = x.dims();
    let half = width / 2;
    let radians: Vec<f32> = positions
        .iter()
        .flat_map(|&position| {
            (0..half).map(move |i| position as f32 / 10000f32.powf(2.0 * i as f32 / width as f32))
        })
        .collect();
    let angle = Tensor::<4>::from_data(TensorData::new(radians, [1, 1, seq, half]), &x.device());
    let (sin, cos) = (angle.clone().sin(), angle.cos());
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

fn attention_mask(valid: &[bool], groups: &[usize], device: &Device) -> Tensor<4, Bool> {
    let seq = valid.len();
    let blocked: Vec<_> = (0..seq)
        .flat_map(|query| {
            (0..seq).map(move |key| !valid[query] || !valid[key] || groups[key] > groups[query])
        })
        .collect();
    Tensor::from_data(TensorData::new(blocked, [1, 1, seq, seq]), device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_rotates_halves_and_preserves_length() {
        let device = Device::flex();
        let x = Tensor::<4>::from_data(
            TensorData::new(vec![1f32, 2., 3., 4., 1., 2., 3., 4.], [1, 1, 2, 4]),
            &device,
        );
        let values = rope(x, &[0, 1]).into_data().try_to_vec::<f32>().unwrap();
        assert_eq!(&values[..4], &[1., 2., 3., 4.]);
        let norm: f32 = values[4..].iter().map(|x| x * x).sum();
        assert!((norm - 30.).abs() < 1e-5, "rotated norm {norm}");
        assert!((values[4] - (1f32.cos() - 3. * 1f32.sin())).abs() < 1e-6);
    }

    #[test]
    fn state_attends_prefix_but_prefix_cannot_attend_state_or_padding() {
        let mask = attention_mask(&[true, true, false, true], &[0, 0, 0, 1], &Device::flex());
        let mask = mask.into_data().try_to_vec::<bool>().unwrap();
        assert!(!mask[1], "image can attend language");
        assert!(mask[3], "image cannot attend state");
        assert!(mask[2], "padding keys are masked");
        assert!(!mask[12], "state can attend image");
        assert!(!mask[15], "state can attend itself");
    }

    #[test]
    fn cached_expert_output_does_not_depend_on_future_actions() {
        let device = Device::flex();
        let cfg = TextConfig {
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 4,
            rms_norm_eps: 1e-5,
            vocab_size: 12,
            attention_bias: false,
            mlp_bias: false,
        };
        let text = TextModel::init(&cfg, &device);
        let cache = text.prefill(Tensor::ones([1, 2, 8], &device), &[true, true], &[0, 1]);
        let expert = ExpertModel::init(&cfg, vec![false, true], &device);
        let first = Tensor::<3>::ones([1, 3, 8], &device);
        let changed = first
            .clone()
            .slice_assign(s![.., 2..3, ..], Tensor::full([1, 1, 8], 20., &device));
        let original = expert
            .forward(first, &cache, &[true, true])
            .slice(s![.., ..2, ..])
            .into_data();
        let modified = expert
            .forward(changed, &cache, &[true, true])
            .slice(s![.., ..2, ..])
            .into_data();
        original.assert_approx_eq::<f32>(&modified, burn::tensor::Tolerance::default());
    }
}
