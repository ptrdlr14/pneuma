//! PaliGemma 语言层与 π₀.₅ 动作专家共用的 Gemma 构件。

use anyhow::{Result, ensure};
use burn::module::{Module, Param};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::gelu_approximate;
use burn::tensor::{Device, Tensor, s};

/// 官方 `get_gemma_config` 的常量；变体决定宽度、深度与注意力头数。
#[derive(Clone, Copy, Debug)]
pub(super) struct GemmaSpec {
    pub width: usize,
    pub depth: usize,
    pub mlp: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
}

pub(super) fn gemma_spec(variant: &str) -> Result<GemmaSpec> {
    let spec = match variant {
        "gemma_300m" => GemmaSpec {
            width: 1024,
            depth: 18,
            mlp: 4096,
            heads: 8,
            kv_heads: 1,
            head_dim: 256,
        },
        "gemma_2b" => GemmaSpec {
            width: 2048,
            depth: 18,
            mlp: 16_384,
            heads: 8,
            kv_heads: 1,
            head_dim: 256,
        },
        other => anyhow::bail!("unknown Gemma variant {other}"),
    };
    ensure!(
        spec.heads.is_multiple_of(spec.kv_heads),
        "invalid Gemma attention geometry"
    );
    Ok(spec)
}

/// 标准 Gemma RMSNorm：权重按 `1 + w` 生效，与官方零初始化一致。
#[derive(Module, Debug)]
pub(super) struct RmsNorm {
    weight: Param<Tensor<1>>,
    #[module(skip)]
    epsilon: f64,
}

impl RmsNorm {
    pub(super) fn init(dim: usize, epsilon: f64, device: &Device) -> Self {
        Self {
            weight: Param::from_tensor(Tensor::zeros([dim], device)),
            epsilon,
        }
    }

    pub(super) fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let dim = x.dims()[2];
        normalized(x, self.epsilon) * (self.weight.val().reshape([1, 1, dim]) + 1.0)
    }
}

/// 自适应 RMSNorm：`dense` 输出 scale、shift、gate 三组调制。
#[derive(Module, Debug)]
pub(super) struct AdaNorm {
    dense: Linear,
    #[module(skip)]
    epsilon: f64,
}

impl AdaNorm {
    pub(super) fn init(dim: usize, epsilon: f64, device: &Device) -> Self {
        Self {
            dense: LinearConfig::new(dim, dim * 3).init(device),
            epsilon,
        }
    }

    pub(super) fn forward(&self, x: Tensor<3>, cond: Tensor<2>) -> (Tensor<3>, Tensor<3>) {
        let dim = x.dims()[2];
        let [batch, _] = cond.dims();
        let normed = normalized(x, self.epsilon);
        let modulation = self.dense.forward(cond).reshape([batch, 1, dim * 3]);
        let scale = modulation.clone().slice(s![.., .., 0..dim]);
        let shift = modulation.clone().slice(s![.., .., dim..2 * dim]);
        let gate = modulation.slice(s![.., .., 2 * dim..3 * dim]);
        (normed * (scale + 1.0) + shift, gate)
    }
}

fn normalized(x: Tensor<3>, epsilon: f64) -> Tensor<3> {
    let rms = (x.clone().powf_scalar(2.0).mean_dim(2) + epsilon).sqrt();
    x / rms
}

/// Gemma 的 gated MLP：`down(gelu(gate(x)) * up(x))`。
#[derive(Module, Debug)]
pub(super) struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    pub(super) fn init(hidden: usize, intermediate: usize, device: &Device) -> Self {
        let linear = |input, output| {
            LinearConfig::new(input, output)
                .with_bias(false)
                .init(device)
        };
        Self {
            gate_proj: linear(hidden, intermediate),
            up_proj: linear(hidden, intermediate),
            down_proj: linear(intermediate, hidden),
        }
    }

    pub(super) fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        self.down_proj
            .forward(gelu_approximate(self.gate_proj.forward(x.clone())) * self.up_proj.forward(x))
    }
}

/// 无偏置的 Gemma 投影；`q_dim` 与 `kv_dim` 可不同，供专家层复用。
#[derive(Module, Debug)]
pub(super) struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    #[module(skip)]
    pub(super) heads: usize,
    #[module(skip)]
    pub(super) kv_heads: usize,
    #[module(skip)]
    pub(super) head_dim: usize,
}

impl Attention {
    pub(super) fn init(
        hidden: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        device: &Device,
    ) -> Self {
        let linear = |input, output| {
            LinearConfig::new(input, output)
                .with_bias(false)
                .init(device)
        };
        Self {
            q_proj: linear(hidden, heads * head_dim),
            k_proj: linear(hidden, kv_heads * head_dim),
            v_proj: linear(hidden, kv_heads * head_dim),
            o_proj: linear(heads * head_dim, hidden),
            heads,
            kv_heads,
            head_dim,
        }
    }

    pub(super) fn q(&self, x: Tensor<3>) -> Tensor<3> {
        self.q_proj.forward(x)
    }

    pub(super) fn kv(&self, x: Tensor<3>) -> (Tensor<3>, Tensor<3>) {
        (self.k_proj.forward(x.clone()), self.v_proj.forward(x))
    }

    pub(super) fn out(&self, x: Tensor<3>) -> Tensor<3> {
        self.o_proj.forward(x)
    }
}

#[derive(Module, Debug)]
pub(super) struct LangLayer {
    pub(super) self_attn: Attention,
    pub(super) mlp: Mlp,
    pub(super) input_layernorm: RmsNorm,
    pub(super) post_attention_layernorm: RmsNorm,
}

impl LangLayer {
    pub(super) fn init(spec: &GemmaSpec, epsilon: f64, device: &Device) -> Self {
        Self {
            self_attn: Attention::init(
                spec.width,
                spec.heads,
                spec.kv_heads,
                spec.head_dim,
                device,
            ),
            mlp: Mlp::init(spec.width, spec.mlp, device),
            input_layernorm: RmsNorm::init(spec.width, epsilon, device),
            post_attention_layernorm: RmsNorm::init(spec.width, epsilon, device),
        }
    }
}

#[derive(Module, Debug)]
pub(super) struct ExpertLayer {
    pub(super) self_attn: Attention,
    pub(super) mlp: Mlp,
    pub(super) input_layernorm: AdaNorm,
    pub(super) post_attention_layernorm: AdaNorm,
}

impl ExpertLayer {
    pub(super) fn init(spec: &GemmaSpec, epsilon: f64, device: &Device) -> Self {
        Self {
            self_attn: Attention::init(
                spec.width,
                spec.heads,
                spec.kv_heads,
                spec.head_dim,
                device,
            ),
            mlp: Mlp::init(spec.width, spec.mlp, device),
            input_layernorm: AdaNorm::init(spec.width, epsilon, device),
            post_attention_layernorm: AdaNorm::init(spec.width, epsilon, device),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemma_specs_match_the_official_variants() {
        let small = gemma_spec("gemma_300m").unwrap();
        assert_eq!((small.width, small.mlp, small.heads), (1024, 4096, 8));
        let large = gemma_spec("gemma_2b").unwrap();
        assert_eq!((large.width, large.mlp, large.heads), (2048, 16_384, 8));
        assert!(gemma_spec("gemma_7b").is_err());
    }

    #[test]
    fn rms_norm_applies_one_plus_weight() {
        let device = Device::flex();
        let norm = RmsNorm::init(4, 1e-6, &device);
        let x = Tensor::<3>::from_data(
            burn::tensor::TensorData::new(vec![1f32, 2., 3., 4.], [1, 1, 4]),
            &device,
        );
        let out = norm.forward(x).into_data().try_to_vec::<f32>().unwrap();
        let rms = ((1f32 + 4. + 9. + 16.) / 4. + 1e-6).sqrt();
        for (got, want) in out.iter().zip([1f32, 2., 3., 4.]) {
            assert!((got - want / rms).abs() < 1e-5);
        }
    }

    #[test]
    fn ada_norm_returns_three_way_modulation() {
        let device = Device::flex();
        let norm = AdaNorm::init(8, 1e-6, &device);
        let x = Tensor::<3>::ones([1, 5, 8], &device);
        let cond = Tensor::<2>::zeros([1, 8], &device);
        let (out, gate) = norm.forward(x, cond);
        assert_eq!(out.dims(), [1, 5, 8]);
        // 调制对 token 轴广播，gate 保持 [1,1,8]。
        assert_eq!(gate.dims(), [1, 1, 8]);
    }
}
