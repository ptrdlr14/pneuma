#!/usr/bin/env python3
"""导出 Cosmos3 Edge 生成流位置的参照。

把理解流的位置固定后，调用官方 `get_rope_index` 得到 `[3,B,L]` 的联合位置。
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lerobot-src", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not sys.flags.isolated:
        parser.error("必须使用 python -I")
    lerobot_src, output = (path.expanduser().resolve() for path in (args.lerobot_src, args.output))

    sys.path.insert(0, str(lerobot_src))
    import torch
    from lerobot.policies.cosmos3 import modeling_cosmos3 as M

    torch.set_default_dtype(torch.float32)
    output.mkdir(parents=True, exist_ok=True)

    text_len, batch = 5, 1
    positions = torch.stack(
        [
            torch.arange(text_len, dtype=torch.float32),
            2 * torch.arange(text_len, dtype=torch.float32),
            3 * torch.arange(text_len, dtype=torch.float32),
        ]
    ).unsqueeze(1)  # [3,1,L]

    stub = SimpleNamespace()
    stub.config = SimpleNamespace(
        temporal_modality_margin=15000, base_fps=24.0, latent_patch_size=2
    )
    stub.vae = SimpleNamespace(config=SimpleNamespace(scale_factor_temporal=4))
    stub.mot_backbone = SimpleNamespace(
        model=SimpleNamespace(get_rope_index=lambda *args, **kwargs: (positions, None))
    )
    input_ids = torch.zeros([batch, text_len], dtype=torch.long)
    video_grid = torch.tensor([[2, 16, 16]])
    seconds = torch.tensor([0.5])
    result = M.Cosmos3VFMNetwork.get_rope_index(
        stub,
        input_ids,
        video_grid,
        seconds,
        num_actions=4,
        action_start_frame_offset=torch.tensor([1]),
    )
    print("result", list(result.shape))

    manifest = {"format_version": 1, "tensors": {}}

    def dump(name, tensor):
        array = tensor.detach().cpu().to(torch.float32).contiguous().numpy().astype("<f4", copy=False)
        path = output / f"{name}.bin"
        array.tofile(path)
        manifest["tensors"][name] = {"shape": list(array.shape), "sha256": sha256(path)}
        print(f"{name}: {list(array.shape)}", flush=True)

    # 理解流：3 个文本、4 个图像 token、1 个文本、4 个视频 token、1 个文本。
    token_types = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 0, 2, 2, 2, 2, 0]])
    grids = torch.tensor([[1, 4, 4]])
    understanding_stub = SimpleNamespace(
        config=SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2))
    )
    understanding_stub.get_vision_position_ids = (
        M.MoTModelMixin.get_vision_position_ids.__get__(understanding_stub)
    )
    understanding, _ = M.MoTModelMixin.get_rope_index(
        understanding_stub,
        torch.zeros_like(token_types),
        token_types,
        grids,
        grids,
        None,
    )
    dump("understanding", understanding)
    dump("token_types", token_types)
    dump("text_positions", positions)
    dump("positions", result)
    manifest.update(
        {
            "text_len": text_len,
            "batch": batch,
            "frames": 2,
            "height": 16,
            "width": 16,
            "second_per_frame": 0.5,
            "num_actions": 4,
            "action_offset": 1,
        }
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("完成", flush=True)


if __name__ == "__main__":
    main()
