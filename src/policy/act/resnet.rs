//! torchvision resnet18 的移植（FrozenBatchNorm2d 即 eval BN）：conv1/bn1/maxpool/layer1..4。
//! 字段名与 torch state_dict 段一一对应；downsample.0/.1 → conv/norm 两个键在加载处重映射。

use burn::module::Module;
use burn::nn::PaddingConfig2d;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::norm::{BatchNorm, BatchNormConfig};
use burn::nn::pool::{MaxPool2d, MaxPool2dConfig};
use burn::tensor::activation::relu;
use burn::tensor::{Device, Tensor};

#[derive(Clone, Copy, Debug)]
pub struct BasicBlockConfig {
    pub channels: usize,
    pub stride: usize,
}

impl BasicBlockConfig {
    fn conv(&self, from: usize, to: usize, stride: usize, device: &Device) -> Conv2d {
        Conv2dConfig::new([from, to], [3, 3])
            .with_stride([stride, stride])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_bias(false)
            .init(device)
    }

    pub fn init(&self, prev_channels: usize, device: &Device) -> BasicBlock {
        let stride = self.stride;
        let downsample = (stride != 1 || prev_channels != self.channels).then(|| Downsample {
            conv: Conv2dConfig::new([prev_channels, self.channels], [1, 1])
                .with_stride([stride, stride])
                .with_bias(false)
                .init(device),
            norm: BatchNormConfig::new(self.channels).init(device),
        });
        BasicBlock {
            conv1: self.conv(prev_channels, self.channels, stride, device),
            bn1: BatchNormConfig::new(self.channels).init(device),
            conv2: self.conv(self.channels, self.channels, 1, device),
            bn2: BatchNormConfig::new(self.channels).init(device),
            downsample,
        }
    }
}

#[derive(Module, Debug)]
pub struct Downsample {
    pub conv: Conv2d,
    pub norm: BatchNorm,
}

#[derive(Module, Debug)]
pub struct BasicBlock {
    pub conv1: Conv2d,
    pub bn1: BatchNorm,
    pub conv2: Conv2d,
    pub bn2: BatchNorm,
    pub downsample: Option<Downsample>,
}

impl BasicBlock {
    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let identity = x.clone();
        let out = self.conv1.forward(x);
        let out = relu(self.bn1.forward(out));
        let out = self.bn2.forward(self.conv2.forward(out));
        let identity = match &self.downsample {
            Some(d) => d.norm.forward(d.conv.forward(identity)),
            None => identity,
        };
        relu(out + identity)
    }
}

/// IntermediateLayerGetter(resnet18, {"layer4": "feature_map"})——只到 layer4。
#[derive(Module, Debug)]
pub struct Resnet {
    pub conv1: Conv2d,
    pub bn1: BatchNorm,
    #[module(skip)]
    pub maxpool: MaxPool2d,
    pub layer1: Vec<BasicBlock>,
    pub layer2: Vec<BasicBlock>,
    pub layer3: Vec<BasicBlock>,
    pub layer4: Vec<BasicBlock>,
}

pub struct ResnetConfig;

impl ResnetConfig {
    // 首块仅在 stride≠1 或通道变化时才带 downsample（layer1 首块 64→64 s1 是恒等捷径）。
    fn layer(
        &self,
        channels: usize,
        stride: usize,
        prev: usize,
        device: &Device,
    ) -> Vec<BasicBlock> {
        let first = BasicBlockConfig { channels, stride };
        let second = BasicBlockConfig {
            channels,
            stride: 1,
        };
        vec![first.init(prev, device), second.init(channels, device)]
    }

    pub fn init(&self, device: &Device) -> Resnet {
        Resnet {
            conv1: Conv2dConfig::new([3, 64], [7, 7])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(3, 3, 3, 3))
                .with_bias(false)
                .init(device),
            bn1: BatchNormConfig::new(64).init(device),
            maxpool: MaxPool2dConfig::new([3, 3])
                .with_strides([2, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(),
            layer1: self.layer(64, 1, 64, device),
            layer2: self.layer(128, 2, 64, device),
            layer3: self.layer(256, 2, 128, device),
            layer4: self.layer(512, 2, 256, device),
        }
    }
}

impl Resnet {
    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let x = relu(self.bn1.forward(self.conv1.forward(x)));
        let x = self.maxpool.forward(x);
        let x = self.layer1.iter().fold(x, |acc, b| b.forward(acc));
        let x = self.layer2.iter().fold(x, |acc, b| b.forward(acc));
        let x = self.layer3.iter().fold(x, |acc, b| b.forward(acc));
        self.layer4.iter().fold(x, |acc, b| b.forward(acc))
    }
}
