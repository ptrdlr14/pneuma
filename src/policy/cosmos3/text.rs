//! Cosmos3 Edge 的 MoT 文本主干：交错 mRoPE、理解/生成双流、联合注意力。
//!
//! 理解流因果注意自身；生成流注意 `[理解 | 生成]` 的全部键，且理解键经过
//! `k_norm_und_for_gen` 的单独归一化。MLP 为无门控的 `relu2`。

// 文本主干在 MoT 主干接线前不参与运行时构建；届时移除此豁免。
#![allow(dead_code)]

use burn::module::{Module, Param};
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::tensor::activation::{relu, softmax};
use burn::tensor::{Bool, Device, Int, Tensor, TensorData, s};

use super::config::TextConfig;

/// Qwen3-VL 的交错 mRoPE。
pub(super) struct MRope {
    inv_freq: Vec<f32>,
    sections: [usize; 3],
    head_dim: usize,
}

impl MRope {
    pub(super) fn new(head_dim: usize, theta: f64, sections: [usize; 3]) -> Self {
        let half = head_dim / 2;
        let inv_freq = (0..half)
            .map(|index| (1.0 / theta.powf(2.0 * index as f64 / head_dim as f64)) as f32)
            .collect();
        Self {
            inv_freq,
            sections,
            head_dim,
        }
    }

    /// 三个轴各 `T` 个位置；返回 `[1,T,head_dim]` 的 `(cos, sin)`。
    pub(super) fn cos_sin(
        &self,
        positions: [&[f32]; 3],
        device: &Device,
    ) -> (Tensor<3>, Tensor<3>) {
        let seq = positions[0].len();
        let half = self.head_dim / 2;
        let mut freqs = vec![0f32; 3 * seq * half];
        for axis in 0..3 {
            for token in 0..seq {
                for index in 0..half {
                    freqs[axis * seq * half + token * half + index] =
                        positions[axis][token] * self.inv_freq[index];
                }
            }
        }
        // T 轴打底，再按偏移覆盖 H 与 W 的频率。
        let mut merged = freqs[..seq * half].to_vec();
        for (axis, offset) in [(1usize, 1usize), (2, 2)] {
            let length = self.sections[axis] * 3;
            let mut index = offset;
            while index < length {
                for token in 0..seq {
                    merged[token * half + index] = freqs[axis * seq * half + token * half + index];
                }
                index += 3;
            }
        }
        let mut cos = Vec::with_capacity(seq * self.head_dim);
        let mut sin = Vec::with_capacity(seq * self.head_dim);
        for token in 0..seq {
            for _pass in 0..2 {
                for index in 0..half {
                    let angle = merged[token * half + index];
                    cos.push(angle.cos());
                    sin.push(angle.sin());
                }
            }
        }
        let shape = [1, seq, self.head_dim];
        (
            Tensor::from_data(TensorData::new(cos, shape), device),
            Tensor::from_data(TensorData::new(sin, shape), device),
        )
    }
}

#[derive(Module, Debug)]
struct RmsNorm {
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

    /// 作用于 head 维度，输入 `[batch, seq, heads, head_dim]`。
    fn forward_head(&self, x: Tensor<4>) -> Tensor<4> {
        let dim = x.dims()[3];
        let rms = (x.clone().powf_scalar(2.0).mean_dim(3) + self.epsilon).sqrt();
        x / rms * self.weight.val().reshape([1, 1, 1, dim])
    }
}

#[derive(Module, Debug)]
struct Mlp {
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
            up_proj: linear(cfg.hidden_size, cfg.intermediate_size),
            down_proj: linear(cfg.intermediate_size, cfg.hidden_size),
        }
    }

    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let hidden = relu(self.up_proj.forward(x));
        self.down_proj.forward(hidden.clone() * hidden)
    }
}

#[derive(Module, Debug)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_proj_gen: Linear,
    k_proj_gen: Linear,
    v_proj_gen: Linear,
    o_proj_gen: Linear,
    q_norm_gen: RmsNorm,
    k_norm_gen: RmsNorm,
    k_norm_und_for_gen: RmsNorm,
    #[module(skip)]
    heads: usize,
    #[module(skip)]
    kv_heads: usize,
    #[module(skip)]
    head_dim: usize,
}

impl Attention {
    fn init(cfg: &TextConfig, device: &Device) -> Self {
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        let linear = |input, output| {
            LinearConfig::new(input, output)
                .with_bias(cfg.attention_bias)
                .init(device)
        };
        Self {
            q_proj: linear(cfg.hidden_size, q_dim),
            k_proj: linear(cfg.hidden_size, kv_dim),
            v_proj: linear(cfg.hidden_size, kv_dim),
            o_proj: linear(q_dim, cfg.hidden_size),
            q_proj_gen: linear(cfg.hidden_size, q_dim),
            k_proj_gen: linear(cfg.hidden_size, kv_dim),
            v_proj_gen: linear(cfg.hidden_size, kv_dim),
            o_proj_gen: linear(q_dim, cfg.hidden_size),
            q_norm_gen: RmsNorm::init(cfg.head_dim, cfg.rms_norm_eps, device),
            k_norm_gen: RmsNorm::init(cfg.head_dim, cfg.rms_norm_eps, device),
            k_norm_und_for_gen: RmsNorm::init(cfg.head_dim, cfg.rms_norm_eps, device),
            heads: cfg.num_attention_heads,
            kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        und: Tensor<3>,
        generation: Tensor<3>,
        cos_und: &Tensor<3>,
        sin_und: &Tensor<3>,
        cos_gen: &Tensor<3>,
        sin_gen: &Tensor<3>,
        und_blocked: &Tensor<4, Bool>,
        gen_blocked: &Tensor<4, Bool>,
    ) -> (Tensor<3>, Tensor<3>) {
        // Nemotron 的理解流没有 QK 范数。
        let q = reshape_heads(self.q_proj.forward(und.clone()), self.heads);
        let q = rope(q.permute([0, 2, 1, 3]), cos_und, sin_und);

        let k = reshape_heads(self.k_proj.forward(und.clone()), self.kv_heads);
        // 生成流看到的理解键：单独的头维范数，再单独施加 RoPE。
        let k_und_for_gen = self.k_norm_und_for_gen.forward_head(k.clone());
        let k_und_for_gen = rope(k_und_for_gen.permute([0, 2, 1, 3]), cos_und, sin_und);
        let k = rope(k.permute([0, 2, 1, 3]), cos_und, sin_und);
        let v = reshape_heads(self.v_proj.forward(und), self.kv_heads).permute([0, 2, 1, 3]);

        let q_gen = reshape_heads(self.q_proj_gen.forward(generation.clone()), self.heads);
        let q_gen = self.q_norm_gen.forward_head(q_gen).permute([0, 2, 1, 3]);
        let q_gen = rope(q_gen, cos_gen, sin_gen);
        let k_gen = reshape_heads(self.k_proj_gen.forward(generation.clone()), self.kv_heads);
        let k_gen = self.k_norm_gen.forward_head(k_gen);
        let k_gen = rope(k_gen.permute([0, 2, 1, 3]), cos_gen, sin_gen);
        let v_gen =
            reshape_heads(self.v_proj_gen.forward(generation), self.kv_heads).permute([0, 2, 1, 3]);

        let und_out = attend(q, k, v.clone(), und_blocked.clone());
        let key_all = Tensor::cat(vec![k_und_for_gen, k_gen], 2);
        let value_all = Tensor::cat(vec![v, v_gen], 2);
        let gen_out = attend(q_gen, key_all, value_all, gen_blocked.clone());
        (
            self.o_proj.forward(und_out),
            self.o_proj_gen.forward(gen_out),
        )
    }
}

/// `[batch, seq, heads*head_dim]` → `[batch, seq, heads, head_dim]`。
fn reshape_heads(x: Tensor<3>, count: usize) -> Tensor<4> {
    let [batch, seq, dim] = x.dims();
    x.reshape([batch, seq, count, dim / count])
}

fn rope(x: Tensor<4>, cos: &Tensor<3>, sin: &Tensor<3>) -> Tensor<4> {
    let [batch, _heads, seq, width] = x.dims();
    let cos = cos
        .clone()
        .reshape([1, 1, seq, width])
        .expand([batch, 1, seq, width]);
    let sin = sin
        .clone()
        .reshape([1, 1, seq, width])
        .expand([batch, 1, seq, width]);
    let half = width / 2;
    let first = x.clone().slice(s![.., .., .., ..half]);
    let second = x.clone().slice(s![.., .., .., half..]);
    let rotated = Tensor::cat(vec![second.clone() * -1.0, first.clone()], 3);
    x * cos + rotated * sin
}

fn repeat_kv(x: Tensor<4>, groups: usize) -> Tensor<4> {
    if groups == 1 {
        return x;
    }
    let [batch, heads_count, seq, width] = x.dims();
    x.reshape([batch, heads_count, 1, seq, width])
        .expand([batch, heads_count, groups, seq, width])
        .reshape([batch, heads_count * groups, seq, width])
}

fn attend(q: Tensor<4>, k: Tensor<4>, v: Tensor<4>, blocked: Tensor<4, Bool>) -> Tensor<3> {
    let [batch, heads_count, seq, width] = q.dims();
    let groups = heads_count / k.dims()[1];
    let scores = q.matmul(repeat_kv(k, groups).swap_dims(2, 3)) / (width as f64).sqrt();
    let probabilities = softmax(scores.mask_fill(blocked, f32::MIN), 3);
    probabilities
        .matmul(repeat_kv(v, groups))
        .permute([0, 2, 1, 3])
        .reshape([batch, seq, heads_count * width])
}

#[derive(Module, Debug)]
struct MotLayer {
    self_attn: Attention,
    mlp: Mlp,
    mlp_gen: Mlp,
    input_layernorm: RmsNorm,
    input_layernorm_gen: RmsNorm,
    post_attention_layernorm: RmsNorm,
    post_attention_layernorm_gen: RmsNorm,
}

impl MotLayer {
    fn init(cfg: &TextConfig, device: &Device) -> Self {
        Self {
            self_attn: Attention::init(cfg, device),
            mlp: Mlp::init(cfg, device),
            mlp_gen: Mlp::init(cfg, device),
            input_layernorm: RmsNorm::init(cfg.hidden_size, cfg.rms_norm_eps, device),
            input_layernorm_gen: RmsNorm::init(cfg.hidden_size, cfg.rms_norm_eps, device),
            post_attention_layernorm: RmsNorm::init(cfg.hidden_size, cfg.rms_norm_eps, device),
            post_attention_layernorm_gen: RmsNorm::init(cfg.hidden_size, cfg.rms_norm_eps, device),
        }
    }
}

#[derive(Module, Debug)]
pub(super) struct TextModel {
    embed_tokens: Embedding,
    layers: Vec<MotLayer>,
    norm: RmsNorm,
    norm_gen: RmsNorm,
}

impl TextModel {
    pub(super) fn init(cfg: &TextConfig, device: &Device) -> Self {
        Self {
            embed_tokens: EmbeddingConfig::new(cfg.vocab_size, cfg.hidden_size).init(device),
            layers: (0..cfg.num_hidden_layers)
                .map(|_| MotLayer::init(cfg, device))
                .collect(),
            norm: RmsNorm::init(cfg.hidden_size, cfg.rms_norm_eps, device),
            norm_gen: RmsNorm::init(cfg.hidden_size, cfg.rms_norm_eps, device),
        }
    }

    /// 文本 token 的嵌入，供外部先融合视觉特征再送入主干。
    pub(super) fn embed(&self, tokens: &[i64], device: &Device) -> Tensor<3> {
        let ids = Tensor::<2, Int>::from_data(
            TensorData::new(tokens.to_vec(), [1, tokens.len()]),
            device,
        );
        self.embed_tokens.forward(ids)
    }

    /// `positions` 的三轴长度必须等于 `文本长度 + 生成长度`。
    pub(super) fn forward(
        &self,
        tokens: &[i64],
        text_valid: &[bool],
        generation: Tensor<3>,
        positions: [&[f32]; 3],
        rope: &MRope,
        device: &Device,
    ) -> (Tensor<3>, Tensor<3>) {
        let und = self.embed(tokens, device);
        self.forward_embeds(und, text_valid, generation, positions, rope, device)
    }

    /// 与 `forward` 相同，但理解流使用外部给定的嵌入（已融合视觉特征）。
    pub(super) fn forward_embeds(
        &self,
        und: Tensor<3>,
        text_valid: &[bool],
        generation: Tensor<3>,
        positions: [&[f32]; 3],
        rope: &MRope,
        device: &Device,
    ) -> (Tensor<3>, Tensor<3>) {
        let text_len = und.dims()[1];
        let gen_len = generation.dims()[1];
        let mut und = und;
        let mut generation = generation;

        let (cos, sin) = rope.cos_sin(positions, device);
        let cos_und = cos.clone().slice(s![.., ..text_len, ..]);
        let sin_und = sin.clone().slice(s![.., ..text_len, ..]);
        let cos_gen = cos.slice(s![.., text_len.., ..]);
        let sin_gen = sin.slice(s![.., text_len.., ..]);

        let und_blocked = causal_blocked(text_valid, device);
        let gen_blocked = generation_blocked(text_valid, gen_len, device);

        for layer in &self.layers {
            let und_norm = layer.input_layernorm.forward(und.clone());
            let gen_norm = layer.input_layernorm_gen.forward(generation.clone());
            let (und_out, gen_out) = layer.self_attn.forward(
                und_norm,
                gen_norm,
                &cos_und,
                &sin_und,
                &cos_gen,
                &sin_gen,
                &und_blocked,
                &gen_blocked,
            );
            und = und + und_out;
            generation = generation + gen_out;

            und = und.clone()
                + layer
                    .mlp
                    .forward(layer.post_attention_layernorm.forward(und.clone()));
            generation = generation.clone()
                + layer.mlp_gen.forward(
                    layer
                        .post_attention_layernorm_gen
                        .forward(generation.clone()),
                );
        }
        (self.norm.forward(und), self.norm_gen.forward(generation))
    }
}

/// 理解流：因果，且屏蔽 padding 列。
fn causal_blocked(valid: &[bool], device: &Device) -> Tensor<4, Bool> {
    let seq = valid.len();
    let blocked: Vec<bool> = (0..seq)
        .flat_map(|query| (0..seq).map(move |key| !valid[query] || !valid[key] || key > query))
        .collect();
    Tensor::from_data(TensorData::new(blocked, [1, 1, seq, seq]), device)
}

/// 生成流：非因果；键为 `[理解 | 生成]`，只屏蔽理解的 padding。
fn generation_blocked(valid: &[bool], gen_len: usize, device: &Device) -> Tensor<4, Bool> {
    let text_len = valid.len();
    let keys = text_len + gen_len;
    let blocked: Vec<bool> = (0..gen_len)
        .flat_map(|_| (0..keys).map(move |key| key < text_len && !valid[key]))
        .collect();
    Tensor::from_data(TensorData::new(blocked, [1, 1, gen_len, keys]), device)
}

#[cfg(test)]
mod reference {
    use super::*;
    use burn::tensor::DType;
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

    /// 真实权重下与官方 MoT 文本主干比较两条流的最终状态。
    #[test]
    #[ignore = "需要 cosmos3 权重与参照张量"]
    fn text_matches_reference() {
        let checkpoint =
            PathBuf::from(std::env::var("PNEUMA_COSMOS3_CKPT").expect("PNEUMA_COSMOS3_CKPT"));
        let parity =
            PathBuf::from(std::env::var("PNEUMA_COSMOS3_TEXT").expect("PNEUMA_COSMOS3_TEXT"));
        let device = Device::flex();
        let cfg = super::super::config::Cosmos3Config::default();
        let text = &cfg.mot_backbone.text_config;

        let mut model = TextModel::init(text, &device);
        let mut store = SafetensorsStore::from_file(checkpoint.join("model.safetensors"))
            .with_from_adapter(PyTorchToBurnAdapter.chain(FloatCastAdapter::to(DType::F32)))
            .with_key_remapping(
                r"^model\.network\.mot_backbone\.model\.language_model\.",
                "",
            )
            .allow_partial(true);
        let applied = model.load_from(&mut store).expect("loading text model");
        assert!(
            applied.missing.is_empty(),
            "missing text weights: {:?}",
            applied.missing
        );

        let tokens: Vec<i64> = read_f32(&parity.join("tokens.bin"))
            .into_iter()
            .map(|value| value as i64)
            .collect();
        let valid: Vec<bool> = read_f32(&parity.join("text_mask.bin"))
            .into_iter()
            .map(|value| value > 0.5)
            .collect();
        let generation = Tensor::<3>::from_data(
            TensorData::new(read_f32(&parity.join("gen_embeds.bin")), [1, 8, 2048]),
            &device,
        );
        let positions = read_f32(&parity.join("positions.bin"));
        let total = tokens.len() + 8;
        let (p0, rest) = positions.split_at(total);
        let (p1, p2) = rest.split_at(total);

        let rope = MRope::new(text.head_dim, 1e8, [24, 20, 20]);
        let (text_hidden, gen_hidden) =
            model.forward(&tokens, &valid, generation, [p0, p1, p2], &rope, &device);

        compare(
            "text_hidden",
            &text_hidden
                .into_data()
                .try_to_vec::<f32>()
                .expect("text hidden"),
            &read_f32(&parity.join("text_hidden.bin")),
            2e-3,
        );
        compare(
            "gen_hidden",
            &gen_hidden
                .into_data()
                .try_to_vec::<f32>()
                .expect("generation hidden"),
            &read_f32(&parity.join("gen_hidden.bin")),
            2e-3,
        );
    }
}
