//! Cosmos3 Edge 的主干：SigLIP2 视觉塔、patch merger 与 MoT 文本主干。
//!
//! 一次前向把文本、图像与生成流合在一起，返回两条流的最终隐藏状态。

// 主干在策略接线前不参与运行时构建；届时移除此豁免。
#![allow(dead_code)]

use burn::module::Module;
use burn::tensor::{Device, Tensor};

use super::config::Cosmos3Config;
use super::text::{MRope, TextModel};
use super::vision::{Projector, Visual};

#[derive(Module, Debug)]
pub(super) struct Cosmos3Net {
    visual: Visual,
    projector: Projector,
    text: TextModel,
}

impl Cosmos3Net {
    pub(super) fn init(cfg: &Cosmos3Config, device: &Device) -> Self {
        Self {
            visual: Visual::init(&cfg.mot_backbone.vision_config, device),
            projector: Projector::init(
                &cfg.mot_backbone.projector_config,
                cfg.mot_backbone.vision_config.layer_norm_eps,
                device,
            ),
            text: TextModel::init(&cfg.mot_backbone.text_config, device),
        }
    }

    pub(super) fn rope(cfg: &Cosmos3Config) -> MRope {
        let text = &cfg.mot_backbone.text_config;
        let parameters = text
            .rope_parameters
            .as_ref()
            .expect("文本主干需要 rope_parameters");
        MRope::new(
            text.head_dim,
            parameters.rope_theta,
            parameters.mrope_section,
        )
    }

    /// 一次前向。`image_span` 是图像 token 在文本序列里的连续区间。
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward(
        &self,
        tokens: &[i64],
        text_valid: &[bool],
        image: Option<Tensor<4>>,
        image_span: std::ops::Range<usize>,
        generation: Tensor<3>,
        positions: [&[f32]; 3],
        rope: &MRope,
        device: &Device,
    ) -> (Tensor<3>, Tensor<3>) {
        let mut und = self.text.embed(tokens, device);
        if let Some(image) = image {
            let features = self.projector.forward(self.visual.forward(image));
            let [_, count, width] = features.dims();
            und = und.slice_assign([0..1, image_span.start..image_span.end, 0..width], features);
            debug_assert_eq!(count, image_span.end - image_span.start);
        }
        self.text
            .forward_embeds(und, text_valid, generation, positions, rope, device)
    }
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

    #[test]
    #[ignore = "需要 cosmos3 权重与参照张量"]
    fn network_matches_reference() {
        let checkpoint =
            PathBuf::from(std::env::var("PNEUMA_COSMOS3_CKPT").expect("PNEUMA_COSMOS3_CKPT"));
        let parity =
            PathBuf::from(std::env::var("PNEUMA_COSMOS3_NET").expect("PNEUMA_COSMOS3_NET"));
        let device = Device::flex();
        let cfg = super::super::config::Cosmos3Config::default();
        let mut net = Cosmos3Net::init(&cfg, &device);

        let mut store = SafetensorsStore::from_file(checkpoint.join("model.safetensors"))
            .with_from_adapter(PyTorchToBurnAdapter.chain(FloatCastAdapter::to(DType::F32)))
            .with_key_remapping(r"^model\.network\.mot_backbone\.model\.", "")
            .with_key_remapping(r"^language_model\.", "text.")
            .allow_partial(true);
        let applied = net.load_from(&mut store).expect("loading network");
        assert!(
            applied.missing.is_empty(),
            "missing network weights: {:?}",
            &applied.missing[..applied.missing.len().min(5)]
        );

        let tokens: Vec<i64> = read_f32(&parity.join("tokens.bin"))
            .into_iter()
            .map(|value| value as i64)
            .collect();
        let valid = vec![true; tokens.len()];
        let image = Tensor::<4>::from_data(
            TensorData::new(read_f32(&parity.join("image.bin")), [1, 3, 256, 256]),
            &device,
        );
        let generation = Tensor::<3>::from_data(
            TensorData::new(read_f32(&parity.join("gen_embeds.bin")), [1, 68, 2048]),
            &device,
        );
        let total = tokens.len() + 68;
        let positions: Vec<f32> = (0..total).map(|index| index as f32).collect();
        let (temporal, height, width) = (
            positions.clone(),
            positions
                .iter()
                .map(|value| value * 2.0)
                .collect::<Vec<_>>(),
            positions
                .iter()
                .map(|value| value * 3.0)
                .collect::<Vec<_>>(),
        );
        let rope = Cosmos3Net::rope(&cfg);

        let (text_hidden, gen_hidden) = net.forward(
            &tokens,
            &valid,
            Some(image),
            3..67,
            generation,
            [&temporal, &height, &width],
            &rope,
            &device,
        );
        compare(
            "text_hidden",
            &text_hidden
                .into_data()
                .try_to_vec::<f32>()
                .expect("text hidden"),
            &read_f32(&parity.join("text_hidden.bin")),
            5e-3,
        );
        compare(
            "gen_hidden",
            &gen_hidden
                .into_data()
                .try_to_vec::<f32>()
                .expect("gen hidden"),
            &read_f32(&parity.join("gen_hidden.bin")),
            5e-3,
        );
    }
}
