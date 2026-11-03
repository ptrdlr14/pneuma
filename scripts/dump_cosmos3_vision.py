#!/usr/bin/env python3
"""导出 Cosmos3 Edge 视觉塔与 patch merger 的参照张量。

只用 transformers 的 SigLIP2 实现加载视觉权重，不依赖 LeRobot 的 cosmos3 分支。
token 顺序与官方一致：patch 与位置嵌入都按 2×2 块优先重排。

失败时不保留上一次的 manifest。
"""

import argparse
import hashlib
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


def block_major(x, merge):
    """(h*w, d) 行优先 → 块优先，与官方 reshape/transpose 一致。"""
    import torch

    grid = int(x.shape[0] ** 0.5)
    return (
        x.reshape(grid // merge, merge, grid // merge, merge, -1)
        .transpose(1, 2)
        .reshape(grid * grid, -1)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not sys.flags.isolated:
        parser.error("读取下载模型必须使用 python -I")
    checkpoint, output = (path.expanduser().resolve() for path in (args.checkpoint, args.output))

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from safetensors import safe_open
    from transformers.models.siglip2.configuration_siglip2 import Siglip2VisionConfig
    from transformers.models.siglip2.modeling_siglip2 import Siglip2VisionModel

    torch.set_num_threads(4)
    torch.set_default_dtype(torch.float32)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest_path.unlink()

    config = Siglip2VisionConfig(
        hidden_size=1152,
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        num_channels=3,
        image_size=256,
        patch_size=16,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
    )
    model = Siglip2VisionModel(config).eval()

    prefix = "model.network.mot_backbone.model."
    state = {}
    with safe_open(str(checkpoint / "model.safetensors"), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if key.startswith(prefix + "visual."):
                # HF 的 Siglip2VisionModel 把视觉塔包在 vision_model 下。
                state[f"vision_model.{key[len(prefix + 'visual.') :]}"] = handle.get_tensor(
                    key
                ).float()
    missing, unexpected = model.load_state_dict(state, strict=False)
    # checkpoint 不含 attention-pooling head；它不参与 last_hidden_state。
    missing = [name for name in missing if not name.startswith("vision_model.head.")]
    if missing or unexpected:
        raise SystemExit(f"missing={missing} unexpected={unexpected}")

    # 固定图像：每通道不同的确定性图案，范围 [-1,1]。
    yy, xx = np.indices((256, 256), dtype=np.int32)
    image = np.stack(((xx + 7) % 256, (3 * yy + 11) % 256, (xx + 2 * yy + 23) % 256), axis=-1)
    pixels = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0 * 2 - 1
    patch = 16
    channels, height, width = pixels.shape
    patches = (
        pixels.reshape(channels, height // patch, patch, width // patch, patch)
        .permute(1, 3, 0, 2, 4)
        .reshape(-1, channels * patch * patch)
    )
    row_major_patches = patches.clone()
    patches = block_major(patches, 2)

    manifest = {
        "format_version": 1,
        "checkpoint": str(checkpoint),
        "storage": "每个张量独立文件，C 连续 float32 little-endian。",
        "patch_order": "block-major (2x2)",
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

    dump("image", pixels, "输入图像 [3,256,256]，范围 [-1,1]")

    spatial_shapes = torch.tensor([[16, 16]], dtype=torch.long)
    attention_mask = torch.ones([1, 256], dtype=torch.long)
    with torch.no_grad():
        # HF 的位置嵌入是行优先；输入也用行优先，保证 patch 与位置的配对正确。
        hidden = model(
            pixel_values=row_major_patches.unsqueeze(0),
            pixel_attention_mask=attention_mask,
            spatial_shapes=spatial_shapes,
        ).last_hidden_state[0]
        hidden = block_major(hidden, 2)

        # patch merger：每 4 个连续 token 拼成一行。
        with safe_open(str(checkpoint / "model.safetensors"), framework="pt", device="cpu") as handle:
            def tensor(name):
                return handle.get_tensor(prefix + name).float()

            normed = torch.nn.functional.layer_norm(
                hidden, (1152,), tensor("projector.norm.weight"), tensor("projector.norm.bias"), eps=1e-6
            )
            merged = normed.reshape(-1, 4608)
            hidden_fc1 = torch.nn.functional.gelu(
                merged @ tensor("projector.linear_fc1.weight").T + tensor("projector.linear_fc1.bias")
            )
            projected = hidden_fc1 @ tensor("projector.linear_fc2.weight").T + tensor(
                "projector.linear_fc2.bias"
            )

    dump("patches", patches, "block-major patch 序列 [256,768]")
    dump("vision_hidden", hidden, "视觉塔输出，block-major [256,1152]")
    dump("projected", projected, "投影层输出 [64,2048]")

    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"完成：{manifest_path}", flush=True)


if __name__ == "__main__":
    main()
