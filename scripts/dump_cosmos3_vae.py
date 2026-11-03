#!/usr/bin/env python3
"""导出 Cosmos3 Edge 的 Wan2.2 VAE 编码参照。

用 diffusers 的 `AutoencoderKLWan` 加载 `vae.*` 权重，编码固定片段，取分布均值。
"""

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not sys.flags.isolated:
        parser.error("读取下载模型必须使用 python -I")
    checkpoint, output = (path.expanduser().resolve() for path in (args.checkpoint, args.output))

    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    from diffusers import AutoencoderKLWan
    from safetensors import safe_open

    torch.set_num_threads(8)
    torch.set_default_dtype(torch.float32)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest_path.unlink()

    config = json.loads((checkpoint / "config.json").read_text())
    vae_config = config["vae"]
    accepted = set(inspect.signature(AutoencoderKLWan.__init__).parameters)
    kwargs = {key: value for key, value in vae_config.items() if key in accepted}
    model = AutoencoderKLWan(**kwargs).eval()

    prefix = "model.network.vae."
    state = {}
    with safe_open(str(checkpoint / "model.safetensors"), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if key.startswith(prefix):
                state[key[len(prefix) :]] = handle.get_tensor(key).float()
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [name for name in missing if "decoder" not in name and "post_quant" not in name]
    if missing or unexpected:
        raise SystemExit(f"missing={missing} unexpected={unexpected}")

    frames, height, width = 1, 256, 256
    yy, xx = np.indices((height, width), dtype=np.int32)
    clip = np.stack(
        [
            np.stack(
                [
                    (xx + 5 * frame) % 256,
                    (3 * yy + 7) % 256,
                    (xx + 2 * yy + 11 * frame) % 256,
                ],
                axis=0,
            )
            for frame in range(frames)
        ],
        axis=1,
    ).astype(np.float32)
    pixels = torch.from_numpy(clip).unsqueeze(0) / 255.0 * 2 - 1

    stages = {}

    def capture(name):
        def hook(module, inputs, output):
            stages[name] = output.detach().float()

        return hook

    model.encoder.conv_in.register_forward_hook(capture("conv_in"))
    for index, block in enumerate(model.encoder.down_blocks):
        for slot, resnet in enumerate(block.resnets):
            resnet.register_forward_hook(capture(f"block{index}.resnet{slot}"))
        if block.downsampler is not None:
            block.downsampler.register_forward_hook(capture(f"block{index}.down"))
    model.encoder.mid_block.register_forward_hook(capture("mid"))
    model.encoder.conv_out.register_forward_hook(capture("conv_out"))
    model.quant_conv.register_forward_hook(capture("quant"))

    with torch.no_grad():
        latents = model.encode(pixels).latent_dist.mode()
    mean = torch.tensor(vae_config["latents_mean"]).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae_config["latents_std"]).view(1, -1, 1, 1, 1)
    normalized = (latents - mean) / std

    manifest = {
        "format_version": 1,
        "storage": "每个张量独立文件，C 连续 float32 little-endian。",
        "checkpoint": str(checkpoint),
        "frames": frames,
        "tensors": {},
    }

    def dump(name, tensor, description):
        array = tensor.detach().cpu().to(torch.float32).contiguous().numpy().astype("<f4", copy=False)
        if not np.isfinite(array).all():
            raise ValueError(f"张量 {name} 含非有限数值")
        path = output / f"{name}.bin"
        array.tofile(path)
        manifest["tensors"][name] = {
            "shape": list(array.shape),
            "description": description,
            "sha256": sha256(path),
            "first_values": array.reshape(-1)[:8].tolist(),
        }
        print(f"{name}: {list(array.shape)}", flush=True)

    dump("pixels", pixels, "输入条件帧 [1,3,1,256,256]，范围 [-1,1]")
    dump("latents_raw", latents, "VAE 均值输出 [1,48,1,16,16]（单帧）")
    for name, value in stages.items():
        dump(f"stage_{name}", value, f"编码器中间张量 {name}")
    dump("latents_normalized", normalized, "按通道统计归一化后的隐变量")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"完成：{manifest_path}", flush=True)


if __name__ == "__main__":
    main()
