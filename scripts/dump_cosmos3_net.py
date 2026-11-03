#!/usr/bin/env python3
"""导出 Cosmos3 Edge 主干的参照。

用官方 `Nemotron3DenseVLMoTModel` 加载真实权重，跑一次「文本 + 图像 + 生成流」前向。
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
    parser.add_argument("--lerobot-src", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not sys.flags.isolated:
        parser.error("必须使用 python -I")
    checkpoint, lerobot_src, output = (
        path.expanduser().resolve() for path in (args.checkpoint, args.lerobot_src, args.output)
    )

    os.environ["HF_HUB_OFFLINE"] = "1"
    sys.path.insert(0, str(lerobot_src))
    import torch
    from safetensors import safe_open
    from lerobot.policies.cosmos3.configuration_cosmos3 import Nemotron3DenseVLMoTConfig
    from lerobot.policies.cosmos3.modeling_cosmos3 import Nemotron3DenseVLMoTModel

    torch.set_num_threads(8)
    torch.set_default_dtype(torch.float32)
    output.mkdir(parents=True, exist_ok=True)

    config = Nemotron3DenseVLMoTConfig()
    config.image_token_id = 19
    config.video_token_id = 18
    model = Nemotron3DenseVLMoTModel(config).eval()

    prefix = "model.network.mot_backbone.model."
    state = {}
    with safe_open(str(checkpoint / "model.safetensors"), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if key.startswith(prefix):
                state[key[len(prefix) :]] = handle.get_tensor(key).float()
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(
            f"missing({len(missing)})={missing[:5]} unexpected({len(unexpected)})={unexpected[:5]}"
        )

    yy, xx = np.indices((256, 256), dtype=np.int32)
    image = np.stack(((xx + 7) % 256, (3 * yy + 11) % 256, (xx + 2 * yy + 23) % 256), axis=-1)
    pixels = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0 * 2 - 1
    patch = 16
    channels = pixels.shape[0]
    patches = (
        pixels.reshape(channels, 256 // patch, patch, 256 // patch, patch)
        .permute(1, 3, 0, 2, 4)
        .reshape(-1, channels * patch * patch)
    )
    patches = block_major(patches, 2)

    image_tokens = 64
    tokens = [2, 5, 6] + [19] * image_tokens + [7, 8, 9]
    text_len = len(tokens)
    gen_len = 68
    input_ids = torch.tensor([tokens])
    attention_mask = torch.ones([1, text_len], dtype=torch.long)
    positions = torch.arange(text_len + gen_len, dtype=torch.float32)
    position_ids = torch.stack([positions, 2 * positions, 3 * positions]).unsqueeze(1)
    index = torch.arange(gen_len * 2048, dtype=torch.float32)
    gen_embeds = (torch.sin(index * 0.001) * 0.5).reshape(1, gen_len, 2048)

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds_gen=gen_embeds,
            pixel_values=patches,
            image_grid_thw=torch.tensor([[1, 16, 16]]),
        )

    manifest = {"format_version": 1, "tensors": {}}

    def dump(name, tensor):
        array = tensor.detach().cpu().to(torch.float32).contiguous().numpy().astype("<f4", copy=False)
        if not np.isfinite(array).all():
            raise ValueError(f"张量 {name} 含非有限数值")
        path = output / f"{name}.bin"
        array.tofile(path)
        manifest["tensors"][name] = {"shape": list(array.shape), "sha256": sha256(path)}
        print(f"{name}: {list(array.shape)}", flush=True)

    dump("image", pixels)
    dump("tokens", input_ids.float())
    dump("gen_embeds", gen_embeds)
    dump("text_hidden", out.last_hidden_state)
    dump("gen_hidden", out.last_hidden_state_gen)
    manifest.update({"text_len": text_len, "gen_len": gen_len, "image_tokens": image_tokens})
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("完成", flush=True)


if __name__ == "__main__":
    main()
