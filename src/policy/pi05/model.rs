//! π₀.₅ 的 PaliGemma 前缀与 Gemma 动作专家联合前向。

use anyhow::{Result, ensure};
use burn::module::Module;
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::tensor::activation::{silu, softmax};
use burn::tensor::{Bool, Device, Int, Tensor, TensorData, s};

use super::{
    config::Pi05Config,
    gemma::{AdaNorm, ExpertLayer, GemmaSpec, LangLayer, RmsNorm, gemma_spec},
    vision::{SIGLIP, VisionTower},
};

const RMS_EPSILON: f64 = 1e-6;
const ROPE_THETA: f64 = 10_000.0;

#[derive(Clone, Debug)]
pub(super) struct KvCache {
    key: Tensor<4>,
    value: Tensor<4>,
}

#[derive(Module, Debug)]
pub(super) struct Projector {
    linear: Linear,
}

#[derive(Module, Debug)]
pub(super) struct LanguageModel {
    pub(super) embed_tokens: Embedding,
    pub(super) layers: Vec<LangLayer>,
    pub(super) norm: RmsNorm,
}

#[derive(Module, Debug)]
pub(super) struct PaliGemmaModel {
    pub(super) language_model: LanguageModel,
    pub(super) vision_tower: VisionTower,
    pub(super) multi_modal_projector: Projector,
}

#[derive(Module, Debug)]
pub(super) struct Paligemma {
    pub(super) model: PaliGemmaModel,
}

#[derive(Module, Debug)]
pub(super) struct ExpertModel {
    pub(super) layers: Vec<ExpertLayer>,
    pub(super) norm: AdaNorm,
}

#[derive(Module, Debug)]
pub(super) struct GemmaExpert {
    pub(super) model: ExpertModel,
}

#[derive(Module, Debug)]
pub(super) struct PaligemmaWithExpert {
    pub(super) paligemma: Paligemma,
    pub(super) gemma_expert: GemmaExpert,
}

#[derive(Module, Debug)]
pub(super) struct Pi05Net {
    pub(super) paligemma_with_expert: PaligemmaWithExpert,
    pub(super) action_in_proj: Linear,
    pub(super) action_out_proj: Linear,
    pub(super) time_mlp_in: Linear,
    pub(super) time_mlp_out: Linear,
}

impl Pi05Net {
    pub(super) fn init(cfg: &Pi05Config, device: &Device) -> Result<Self> {
        let lang: GemmaSpec = gemma_spec(&cfg.paligemma_variant)?;
        let expert: GemmaSpec = gemma_spec(&cfg.action_expert_variant)?;
        ensure!(
            lang.head_dim == expert.head_dim,
            "language and expert head dims must match for joint attention"
        );
        ensure!(
            SIGLIP.tokens() == (cfg.image_resolution[0] / SIGLIP.patch).pow(2),
            "image resolution disagrees with the SigLIP patch grid"
        );
        Ok(Self {
            paligemma_with_expert: PaligemmaWithExpert {
                paligemma: Paligemma {
                    model: PaliGemmaModel {
                        language_model: LanguageModel {
                            embed_tokens: EmbeddingConfig::new(257_152, lang.width).init(device),
                            layers: (0..lang.depth)
                                .map(|_| LangLayer::init(&lang, RMS_EPSILON, device))
                                .collect(),
                            norm: RmsNorm::init(lang.width, RMS_EPSILON, device),
                        },
                        vision_tower: VisionTower::init(&SIGLIP, device),
                        multi_modal_projector: Projector {
                            linear: LinearConfig::new(SIGLIP.hidden, lang.width).init(device),
                        },
                    },
                },
                gemma_expert: GemmaExpert {
                    model: ExpertModel {
                        layers: (0..expert.depth)
                            .map(|_| ExpertLayer::init(&expert, RMS_EPSILON, device))
                            .collect(),
                        norm: AdaNorm::init(expert.width, RMS_EPSILON, device),
                    },
                },
            },
            action_in_proj: LinearConfig::new(cfg.max_action_dim, expert.width).init(device),
            action_out_proj: LinearConfig::new(expert.width, cfg.max_action_dim).init(device),
            time_mlp_in: LinearConfig::new(expert.width, expert.width).init(device),
            time_mlp_out: LinearConfig::new(expert.width, expert.width).init(device),
        })
    }

    /// 图像与语言 token 拼成前缀；返回嵌入、有效位与相机数。
    pub(super) fn embed_prefix(
        &self,
        images: Vec<Tensor<4>>,
        tokens: &[i64],
        valid: &[bool],
        device: &Device,
        dtype: &str,
    ) -> Result<(Tensor<3>, Vec<bool>)> {
        let inputs = Tensor::<2, Int>::from_data(
            TensorData::new(tokens.to_vec(), [1, tokens.len()]),
            device,
        );
        let language = self
            .paligemma_with_expert
            .paligemma
            .model
            .language_model
            .embed_tokens
            .forward(inputs);
        // 官方在嵌入模块内乘 sqrt(hidden_size)，并按 checkpoint 精度保存该缓冲。
        let scale = embedding_scale(dtype, language.dims()[2])?;
        let language = language * f64::from(scale);
        let mut embeddings = Vec::with_capacity(images.len() + 1);
        let mut pads = Vec::with_capacity(SIGLIP.tokens() * images.len() + valid.len());
        for image in images {
            let visual = self
                .paligemma_with_expert
                .paligemma
                .model
                .vision_tower
                .forward(image);
            let projected = self
                .paligemma_with_expert
                .paligemma
                .model
                .multi_modal_projector
                .linear
                .forward(visual);
            embeddings.push(projected);
            pads.extend(std::iter::repeat_n(true, SIGLIP.tokens()));
        }
        ensure!(
            language.dims()[1] == valid.len(),
            "language tokens and mask disagree"
        );
        embeddings.push(language);
        pads.extend_from_slice(valid);
        Ok((Tensor::cat(embeddings, 1), pads))
    }

    /// 前缀只走语言模型，缓存每层的 K/V 供去噪步骤复用。
    fn prefill(&self, prefix: Tensor<3>, pads: &[bool], device: &Device) -> Vec<KvCache> {
        let positions = prefix_positions(pads);
        let blocked = prefix_mask(pads, device);
        let language = &self.paligemma_with_expert.paligemma.model.language_model;
        let mut x = prefix;
        let mut cache = Vec::with_capacity(language.layers.len());
        for layer in &language.layers {
            let normalized = layer.input_layernorm.forward(x.clone());
            let query = heads(layer.self_attn.q(normalized.clone()), layer.self_attn.heads);
            let query = rope(query, &positions);
            let (key, value) = layer.self_attn.kv(normalized);
            let key = rope(heads(key, layer.self_attn.kv_heads), &positions);
            let value = heads(value, layer.self_attn.kv_heads);
            let attended = attend(query, key.clone(), value.clone(), blocked.clone());
            let residual = x + layer.self_attn.out(attended);
            x = residual.clone()
                + layer
                    .mlp
                    .forward(layer.post_attention_layernorm.forward(residual));
            cache.push(KvCache { key, value });
        }
        cache
    }

    /// 单个去噪步：专家 token 与缓存前缀联合注意力。
    fn denoise(
        &self,
        x: Tensor<3>,
        cond: Tensor<2>,
        cache: &[KvCache],
        pads: &[bool],
        offset: usize,
        device: &Device,
    ) -> Tensor<3> {
        let expert = &self.paligemma_with_expert.gemma_expert.model;
        let steps = x.dims()[1];
        let positions: Vec<usize> = (offset..offset + steps).collect();
        let keys = pads.len() + steps;
        let mut blocked = Vec::with_capacity(steps * keys);
        for _ in 0..steps {
            for &valid in pads {
                blocked.push(!valid);
            }
            blocked.extend(std::iter::repeat_n(false, steps));
        }
        let mask =
            Tensor::<4, Bool>::from_data(TensorData::new(blocked, [1, 1, steps, keys]), device);
        let mut hidden = x;
        for (layer, cache) in expert.layers.iter().zip(cache) {
            let (normalized, gate) = layer.input_layernorm.forward(hidden.clone(), cond.clone());
            let query = heads(layer.self_attn.q(normalized.clone()), layer.self_attn.heads);
            let query = rope(query, &positions);
            let (key, value) = layer.self_attn.kv(normalized);
            let key = rope(heads(key, layer.self_attn.kv_heads), &positions);
            let value = heads(value, layer.self_attn.kv_heads);
            let key = Tensor::cat(vec![cache.key.clone(), key], 2);
            let value = Tensor::cat(vec![cache.value.clone(), value], 2);
            let attended = attend(query, key, value, mask.clone());
            let residual = hidden + layer.self_attn.out(attended) * gate;
            let (normalized, gate) = layer
                .post_attention_layernorm
                .forward(residual.clone(), cond.clone());
            hidden = residual + layer.mlp.forward(normalized) * gate;
        }
        let (hidden, _) = expert.norm.forward(hidden, cond);
        hidden
    }

    /// 完整采样：前缀预填充与 `num_inference_steps` 次 Euler 积分。
    pub(super) fn sample(
        &self,
        images: Vec<Tensor<4>>,
        tokens: &[i64],
        valid: &[bool],
        noise: Tensor<3>,
        cfg: &Pi05Config,
        device: &Device,
    ) -> Result<Tensor<3>> {
        let (prefix, pads) = self.embed_prefix(images, tokens, valid, device, &cfg.dtype)?;
        let cache = self.prefill(prefix, &pads, device);
        let offset = pads.iter().filter(|&&valid| valid).count();
        let mut x = noise;
        let dt = -1.0 / cfg.num_inference_steps as f64;
        for step in 0..cfg.num_inference_steps {
            let time = (1.0 + step as f64 * dt) as f32;
            let cond = self.time_cond(time, cfg, device);
            let suffix = self.action_in_proj.forward(x.clone());
            let velocity = self.denoise(suffix, cond, &cache, &pads, offset, device);
            let velocity = self.action_out_proj.forward(velocity);
            x = x + velocity * dt;
        }
        Ok(x)
    }

    fn time_cond(&self, time: f32, cfg: &Pi05Config, device: &Device) -> Tensor<2> {
        let dim = self.time_mlp_in.weight.val().dims()[0];
        let half = dim / 2;
        let radians: Vec<f32> = (0..half)
            .map(|i| {
                let fraction = i as f64 / (half - 1) as f64;
                let period = cfg.min_period * (cfg.max_period / cfg.min_period).powf(fraction);
                time * (2.0 * std::f64::consts::PI / period) as f32
            })
            .collect();
        let values: Vec<f32> = radians
            .iter()
            .map(|x| x.sin())
            .chain(radians.iter().map(|x| x.cos()))
            .collect();
        let embedding = Tensor::<2>::from_data(TensorData::new(values, [1, dim]), device);
        let hidden = silu(self.time_mlp_in.forward(embedding));
        silu(self.time_mlp_out.forward(hidden))
    }
}

/// 官方 `embed_scale` 是 `sqrt(hidden)` 按其计算精度保存的缓冲。
fn embedding_scale(dtype: &str, width: usize) -> Result<f32> {
    let scale = (width as f32).sqrt();
    match dtype {
        "float32" => Ok(scale),
        "bfloat16" => {
            // 就近舍入到 bfloat16 的 8 位尾数（含隐含位）。
            let bits = scale.to_bits();
            let rounding = 0x0000_7fff + ((bits >> 16) & 1);
            Ok(f32::from_bits(bits.wrapping_add(rounding) & 0xffff_0000))
        }
        other => anyhow::bail!("unsupported pi05 dtype {other}"),
    }
}

fn heads(x: Tensor<3>, count: usize) -> Tensor<4> {
    let [batch, seq, dim] = x.dims();
    x.reshape([batch, seq, count, dim / count])
        .permute([0, 2, 1, 3])
}

fn rope(x: Tensor<4>, positions: &[usize]) -> Tensor<4> {
    let [_, _, seq, width] = x.dims();
    let half = width / 2;
    let values: Vec<f32> = positions
        .iter()
        .flat_map(|&position| {
            (0..half).map(move |i| {
                position as f32 / ROPE_THETA.powf(2.0 * i as f64 / width as f64) as f32
            })
        })
        .collect();
    let angle = Tensor::<4>::from_data(TensorData::new(values, [1, 1, seq, half]), &x.device());
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

fn repeat_kv(x: Tensor<4>, groups: usize) -> Tensor<4> {
    if groups == 1 {
        return x;
    }
    let [batch, heads, seq, width] = x.dims();
    x.reshape([batch, heads, 1, seq, width])
        .expand([batch, heads, groups, seq, width])
        .reshape([batch, heads * groups, seq, width])
}

fn attend(q: Tensor<4>, k: Tensor<4>, v: Tensor<4>, blocked: Tensor<4, Bool>) -> Tensor<3> {
    let [batch, heads, seq, width] = q.dims();
    let groups = heads / k.dims()[1];
    let scores = q.matmul(repeat_kv(k, groups).swap_dims(2, 3)) / (width as f64).sqrt();
    let probabilities = softmax(scores.mask_fill(blocked, f32::MIN), 3);
    probabilities
        .matmul(repeat_kv(v, groups))
        .permute([0, 2, 1, 3])
        .reshape([batch, seq, heads * width])
}

fn prefix_positions(pads: &[bool]) -> Vec<usize> {
    let mut count = 0usize;
    pads.iter()
        .map(|&valid| {
            count += usize::from(valid);
            count.saturating_sub(1)
        })
        .collect()
}

fn prefix_mask(pads: &[bool], device: &Device) -> Tensor<4, Bool> {
    let seq = pads.len();
    let blocked: Vec<bool> = (0..seq)
        .flat_map(|query| (0..seq).map(move |key| !pads[query] || !pads[key]))
        .collect();
    Tensor::from_data(TensorData::new(blocked, [1, 1, seq, seq]), device)
}
