//! Cosmos3 Edge 的 Wan2.2 VAE 编码器。
//!
//! 只实现推理需要的编码路径：空间 2×2 打包、因果 3D 卷积、逐帧 2D 下采样、
//! 单头中间注意力；输出取分布的均值。

// VAE 在生成流接通前不参与运行时构建；届时移除此豁免。
#![allow(dead_code)]

use burn::module::{Module, Param};
use burn::nn::PaddingConfig2d;
use burn::nn::PaddingConfig3d;
use burn::nn::conv::{Conv2d, Conv2dConfig, Conv3d, Conv3dConfig};
use burn::tensor::activation::silu;
use burn::tensor::{Device, Tensor};

use super::config::VaeConfig;

/// 因果 3D 卷积：时间轴前补两帧，空间轴对称补边，再由 `Conv3d` 计算。
fn causal(conv: &Conv3d, x: Tensor<5>, pad: [usize; 3]) -> Tensor<5> {
    let x = pad5(x, 2, 2 * pad[0], 0);
    let x = pad5(x, 3, pad[1], pad[1]);
    let x = pad5(x, 4, pad[2], pad[2]);
    conv.forward(x)
}

fn conv3d(
    input: usize,
    output: usize,
    kernel: [usize; 3],
    stride: [usize; 3],
    device: &Device,
) -> Conv3d {
    Conv3dConfig::new([input, output], kernel)
        .with_stride(stride)
        .with_padding(PaddingConfig3d::Explicit(0, 0, 0))
        .init(device)
}

fn pad5(x: Tensor<5>, dim: usize, before: usize, after: usize) -> Tensor<5> {
    let dims = x.dims();
    let device = x.device();
    let unfold = |amount: usize| -> Tensor<5> {
        let mut zeros = dims;
        zeros[dim] = amount;
        Tensor::zeros(zeros, &device)
    };
    let mut parts = Vec::with_capacity(3);
    if before > 0 {
        parts.push(unfold(before));
    }
    parts.push(x);
    if after > 0 {
        parts.push(unfold(after));
    }
    if parts.len() == 1 {
        parts.pop().unwrap()
    } else {
        Tensor::cat(parts, dim)
    }
}

/// Wan 的 RMSNorm：对通道维归一化，再乘 `gamma` 与 `sqrt(C)`。
#[derive(Module, Debug)]
struct RmsNorm {
    gamma: Param<Tensor<4>>,
    #[module(skip)]
    channels: usize,
}

impl RmsNorm {
    fn init(channels: usize, device: &Device) -> Self {
        Self {
            gamma: Param::from_tensor(Tensor::ones([channels, 1, 1, 1], device)),
            channels,
        }
    }

    fn forward(&self, x: Tensor<5>) -> Tensor<5> {
        let norm = x
            .clone()
            .powf_scalar(2.0)
            .sum_dim(1)
            .clamp_min(1e-12)
            .sqrt();
        let scale = (self.channels as f64).sqrt();
        x / norm * scale * self.gamma.val().reshape([1, self.channels, 1, 1, 1])
    }
}

/// 逐帧 RMSNorm：`gamma` 形状为 `[C,1,1]`。
#[derive(Module, Debug)]
struct RmsNorm2d {
    gamma: Param<Tensor<3>>,
    #[module(skip)]
    channels: usize,
}

impl RmsNorm2d {
    fn init(channels: usize, device: &Device) -> Self {
        Self {
            gamma: Param::from_tensor(Tensor::ones([channels, 1, 1], device)),
            channels,
        }
    }

    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let norm = x
            .clone()
            .powf_scalar(2.0)
            .sum_dim(1)
            .clamp_min(1e-12)
            .sqrt();
        let scale = (self.channels as f64).sqrt();
        x / norm * scale * self.gamma.val().reshape([1, self.channels, 1, 1])
    }
}

#[derive(Module, Debug)]
struct Resnet {
    norm1: RmsNorm,
    conv1: Conv3d,
    norm2: RmsNorm,
    conv2: Conv3d,
    conv_shortcut: Option<Conv3d>,
}

impl Resnet {
    fn init(input: usize, output: usize, device: &Device) -> Self {
        Self {
            norm1: RmsNorm::init(input, device),
            conv1: conv3d(input, output, [3; 3], [1; 3], device),
            norm2: RmsNorm::init(output, device),
            conv2: conv3d(output, output, [3; 3], [1; 3], device),
            conv_shortcut: (input != output).then(|| conv3d(input, output, [1; 3], [1; 3], device)),
        }
    }

    fn forward(&self, x: Tensor<5>) -> Tensor<5> {
        let shortcut = match &self.conv_shortcut {
            Some(conv) => causal(conv, x.clone(), [0; 3]),
            None => x.clone(),
        };
        let hidden = causal(&self.conv1, silu(self.norm1.forward(x)), [1; 3]);
        let hidden = causal(&self.conv2, silu(self.norm2.forward(hidden)), [1; 3]);
        shortcut + hidden
    }
}

#[derive(Module, Debug)]
struct Attention {
    norm: RmsNorm2d,
    to_qkv: Conv2d,
    proj: Conv2d,
}

impl Attention {
    fn init(channels: usize, device: &Device) -> Self {
        Self {
            norm: RmsNorm2d::init(channels, device),
            to_qkv: Conv2dConfig::new([channels, channels * 3], [1, 1]).init(device),
            proj: Conv2dConfig::new([channels, channels], [1, 1]).init(device),
        }
    }

    fn forward(&self, x: Tensor<5>) -> Tensor<5> {
        let [batch, channels, time, height, width] = x.dims();
        let frames =
            x.clone()
                .permute([0, 2, 1, 3, 4])
                .reshape([batch * time, channels, height, width]);
        let frames = self.norm.forward(frames);
        let qkv = self.to_qkv.forward(frames);
        let tokens = height * width;
        let qkv = qkv.reshape([batch * time, 3, channels, tokens]);
        let q = qkv
            .clone()
            .slice([0..batch * time, 0..1, 0..channels, 0..tokens]);
        let k = qkv
            .clone()
            .slice([0..batch * time, 1..2, 0..channels, 0..tokens]);
        let v = qkv.slice([0..batch * time, 2..3, 0..channels, 0..tokens]);
        let scale = 1.0 / (channels as f64).sqrt();
        let scores = q
            .clone()
            .reshape([batch * time, channels, tokens])
            .permute([0, 2, 1])
            .matmul(k.reshape([batch * time, channels, tokens]))
            * scale;
        let probabilities = burn::tensor::activation::softmax(scores, 2);
        let attended = probabilities
            .matmul(
                v.reshape([batch * time, channels, tokens])
                    .permute([0, 2, 1]),
            )
            .permute([0, 2, 1])
            .reshape([batch * time, channels, height, width]);
        let out = self.proj.forward(attended);
        let out = out
            .reshape([batch, time, channels, height, width])
            .permute([0, 2, 1, 3, 4]);
        out + x
    }
}

#[derive(Module, Debug)]
struct Downsample {
    #[module(skip)]
    temporal: bool,
    time_conv: Option<Conv3d>,
    resample: Conv2d,
}

impl Downsample {
    fn init(channels: usize, temporal: bool, device: &Device) -> Self {
        Self {
            temporal,
            time_conv: temporal.then(|| conv3d(channels, channels, [3, 1, 1], [2, 1, 1], device)),
            resample: Conv2dConfig::new([channels, channels], [3, 3])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(0, 0, 0, 0))
                .init(device),
        }
    }

    fn forward(&self, x: Tensor<5>) -> Tensor<5> {
        // 推理只编码条件帧（时间长度为 1），官方在无跨块缓存时也不做时间下采样。
        let x = match &self.time_conv {
            Some(conv) if x.dims()[2] > 1 => causal(conv, x, [0; 3]),
            _ => x,
        };
        let [batch, channels, time, height, width] = x.dims();
        let frames =
            x.clone()
                .permute([0, 2, 1, 3, 4])
                .reshape([batch * time, channels, height, width]);
        // ZeroPad2d((0,1,0,1))：右侧与底部各补一列/行。
        let frames = pad4(frames, 2, 0, 1);
        let frames = pad4(frames, 3, 0, 1);
        let frames = self.resample.forward(frames);
        let [_, channels, height, width] = frames.dims();
        frames
            .reshape([batch, time, channels, height, width])
            .permute([0, 2, 1, 3, 4])
    }
}

fn pad4(x: Tensor<4>, dim: usize, before: usize, after: usize) -> Tensor<4> {
    let dims = x.dims();
    let device = x.device();
    let zeros = |amount: usize| -> Tensor<4> {
        let mut shape = dims;
        shape[dim] = amount;
        Tensor::zeros(shape, &device)
    };
    let mut parts = Vec::with_capacity(3);
    if before > 0 {
        parts.push(zeros(before));
    }
    parts.push(x);
    if after > 0 {
        parts.push(zeros(after));
    }
    if parts.len() == 1 {
        parts.pop().unwrap()
    } else {
        Tensor::cat(parts, dim)
    }
}

#[derive(Module, Debug)]
struct DownBlock {
    resnets: Vec<Resnet>,
    downsampler: Option<Downsample>,
    #[module(skip)]
    out_channels: usize,
    #[module(skip)]
    factor_t: usize,
    #[module(skip)]
    factor_s: usize,
    #[module(skip)]
    residual: bool,
}

impl DownBlock {
    fn forward(&self, x: Tensor<5>) -> Tensor<5> {
        let shortcut = if self.residual {
            Some(avg_down(
                x.clone(),
                self.out_channels,
                self.factor_t,
                self.factor_s,
            ))
        } else {
            None
        };
        let mut hidden = x;
        for resnet in &self.resnets {
            hidden = resnet.forward(hidden);
        }
        if let Some(down) = &self.downsampler {
            hidden = down.forward(hidden);
        }
        match shortcut {
            Some(value) => hidden + value,
            None => hidden,
        }
    }
}

/// Wan 2.2 的平均捷径：把 `factor_t × factor_s × factor_s` 的窗口折进通道后按组取均值。
fn avg_down(x: Tensor<5>, out_channels: usize, factor_t: usize, factor_s: usize) -> Tensor<5> {
    if factor_t == 1 && factor_s == 1 {
        return x;
    }
    let time = x.dims()[2];
    let pad_t = (factor_t - time % factor_t) % factor_t;
    let x = pad5(x, 2, pad_t, 0);
    let [batch, channels, time, height, width] = x.dims();
    let (t2, h2, w2) = (time / factor_t, height / factor_s, width / factor_s);
    let grouped = x
        .reshape([batch, channels, t2, factor_t, h2, factor_s, w2, factor_s])
        .permute([0, 1, 3, 5, 7, 2, 4, 6]);
    let group = channels * factor_t * factor_s * factor_s / out_channels;
    grouped
        .reshape([batch, out_channels, group, t2, h2, w2])
        .mean_dim(2)
        .reshape([batch, out_channels, t2, h2, w2])
}

#[derive(Module, Debug)]
struct MidBlock {
    resnets: Vec<Resnet>,
    attentions: Vec<Attention>,
}

#[derive(Module, Debug)]
struct Encoder {
    conv_in: Conv3d,
    down_blocks: Vec<DownBlock>,
    mid_block: MidBlock,
    norm_out: RmsNorm,
    conv_out: Conv3d,
}

#[derive(Module, Debug)]
pub(super) struct WanVae {
    encoder: Encoder,
    quant_conv: Conv3d,
    #[module(skip)]
    z_dim: usize,
}

impl WanVae {
    pub(super) fn init(cfg: &VaeConfig, device: &Device) -> Self {
        let dims: Vec<usize> = std::iter::once(1)
            .chain(cfg.dim_mult.iter().copied())
            .map(|multiple| cfg.base_dim * multiple)
            .collect();
        let mut blocks = Vec::new();
        for (index, pair) in dims.windows(2).enumerate() {
            let (input, output) = (pair[0], pair[1]);
            let mut resnets = Vec::new();
            let mut current = input;
            for _ in 0..cfg.num_res_blocks {
                resnets.push(Resnet::init(current, output, device));
                current = output;
            }
            let last = index == dims.len() - 2;
            let temporal = cfg.temperal_downsample.get(index).copied().unwrap_or(false);
            let factor_t = if temporal { 2 } else { 1 };
            let factor_s = if last { 1 } else { 2 };
            let downsampler = (!last).then(|| Downsample::init(output, temporal, device));
            blocks.push(DownBlock {
                resnets,
                downsampler,
                out_channels: output,
                factor_t,
                factor_s,
                residual: cfg.is_residual,
            });
        }
        let last = *dims.last().unwrap();
        let mid = MidBlock {
            resnets: vec![
                Resnet::init(last, last, device),
                Resnet::init(last, last, device),
            ],
            attentions: vec![Attention::init(last, device)],
        };
        Self {
            encoder: Encoder {
                conv_in: conv3d(cfg.in_channels, dims[0], [3; 3], [1; 3], device),
                down_blocks: blocks,
                mid_block: mid,
                norm_out: RmsNorm::init(last, device),
                conv_out: conv3d(last, cfg.z_dim * 2, [3; 3], [1; 3], device),
            },
            quant_conv: Conv3dConfig::new([cfg.z_dim * 2, cfg.z_dim * 2], [1, 1, 1]).init(device),
            z_dim: cfg.z_dim,
        }
    }

    /// 输入 `[1,3,T,H,W]` 的 `[-1,1]` 像素；输出未归一化的隐变量均值 `[1,z_dim,T',H',W']`。
    pub(super) fn encode(&self, pixels: Tensor<5>) -> Tensor<5> {
        let patched = patchify(pixels, 2);
        let mut x = causal(&self.encoder.conv_in, patched, [1; 3]);
        for block in &self.encoder.down_blocks {
            x = block.forward(x);
        }
        x = self.encoder.mid_block.resnets[0].forward(x);
        for attention in &self.encoder.mid_block.attentions {
            x = attention.forward(x);
        }
        x = self.encoder.mid_block.resnets[1].forward(x);
        x = causal(
            &self.encoder.conv_out,
            silu(self.encoder.norm_out.forward(x)),
            [1; 3],
        );
        let x = self.quant_conv.forward(x);
        let [_, _, time, height, width] = x.dims();
        x.slice([0..1, 0..self.z_dim, 0..time, 0..height, 0..width])
    }
}

/// 空间 2×2 打包：`[B,C,T,H,W]` → `[B,C*4,T,H/2,W/2]`。
///
/// 通道顺序为 `(c, pw, ph)`：官方 `patchify` 先取宽度偏移，与常见卷积实现不同。
fn patchify(x: Tensor<5>, patch: usize) -> Tensor<5> {
    let [batch, channels, time, height, width] = x.dims();
    let (gh, gw) = (height / patch, width / patch);
    x.reshape([batch, channels, time, gh, patch, gw, patch])
        .permute([0, 1, 6, 4, 2, 3, 5])
        .reshape([batch, channels * patch * patch, time, gh, gw])
}

#[cfg(test)]
mod reference {
    use super::*;
    use burn::tensor::{DType, TensorData};
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

    /// 真实权重下与 diffusers 的 Wan2.2 编码器比较。
    #[test]
    #[ignore = "需要 cosmos3 权重与参照张量"]
    fn vae_matches_reference() {
        let checkpoint =
            PathBuf::from(std::env::var("PNEUMA_COSMOS3_CKPT").expect("PNEUMA_COSMOS3_CKPT"));
        let parity =
            PathBuf::from(std::env::var("PNEUMA_COSMOS3_VAE").expect("PNEUMA_COSMOS3_VAE"));
        // Flex 的 conv3d 在这组尺寸上会整型溢出，默认改在 CUDA 上比较。
        let name = std::env::var("PNEUMA_DEVICE").unwrap_or_else(|_| "cuda:0".into());
        let backend: crate::config::Device = name.parse().expect("PNEUMA_DEVICE");
        let device = backend.into();
        let cfg = super::super::config::Cosmos3Config::default();
        let mut vae = WanVae::init(&cfg.vae, &device);

        let mut store = SafetensorsStore::from_file(checkpoint.join("model.safetensors"))
            .with_from_adapter(PyTorchToBurnAdapter.chain(FloatCastAdapter::to(DType::F32)))
            .with_key_remapping(r"^model\.network\.vae\.", "")
            .with_key_remapping(r"\.resample\.1\.", ".resample.")
            .allow_partial(true);
        let applied = vae.load_from(&mut store).expect("loading vae");
        assert!(
            applied.missing.is_empty(),
            "missing vae weights: {:?}",
            applied.missing
        );

        let pixels = Tensor::<5>::from_data(
            TensorData::new(read_f32(&parity.join("pixels.bin")), [1, 3, 1, 256, 256]),
            &device,
        );
        let staged = |name: &str, x: &Tensor<5>| {
            compare(
                name,
                &x.clone().into_data().try_to_vec::<f32>().expect("stage"),
                &read_f32(&parity.join(format!("stage_{name}.bin"))),
                1e-2,
            );
        };
        let mut x = causal(&vae.encoder.conv_in, patchify(pixels.clone(), 2), [1; 3]);
        staged("conv_in", &x);
        for (index, block) in vae.encoder.down_blocks.iter().enumerate() {
            let input = x.clone();
            for (slot, resnet) in block.resnets.iter().enumerate() {
                x = resnet.forward(x);
                staged(&format!("block{index}.resnet{slot}"), &x);
            }
            if let Some(down) = &block.downsampler {
                x = down.forward(x);
                staged(&format!("block{index}.down"), &x);
            }
            if block.residual {
                x = x + avg_down(input, block.out_channels, block.factor_t, block.factor_s);
            }
        }
        x = vae.encoder.mid_block.resnets[0].forward(x);
        for attention in &vae.encoder.mid_block.attentions {
            x = attention.forward(x);
        }
        x = vae.encoder.mid_block.resnets[1].forward(x);
        staged("mid", &x);
        x = causal(
            &vae.encoder.conv_out,
            silu(vae.encoder.norm_out.forward(x)),
            [1; 3],
        );
        staged("conv_out", &x);
        x = vae.quant_conv.forward(x);
        staged("quant", &x);

        let latents = vae.encode(pixels);
        assert_eq!(latents.dims(), [1, 48, 1, 16, 16]);
        compare(
            "latents_raw",
            &latents.into_data().try_to_vec::<f32>().expect("latents"),
            &read_f32(&parity.join("latents_raw.bin")),
            1e-3,
        );
    }
}
