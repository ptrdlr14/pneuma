#!/usr/bin/env python3
"""导出 Cosmos3 Edge MoT 文本主干的参照张量。

直接使用 LeRobot 的 cosmos3 分支实现（`lerobot.policies.cosmos3`），
固定输入与位置后运行一次双流前向。
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--lerobot-src", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not sys.flags.isolated:
        parser.error("读取下载模型必须使用 python -I")
    checkpoint, lerobot_src, output = (
        path.expanduser().resolve() for path in (args.checkpoint, args.lerobot_src, args.output)
    )

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(lerobot_src))
    import torch
    from safetensors import safe_open
    from lerobot.policies.cosmos3.configuration_cosmos3 import Nemotron3DenseVLMoTTextConfig
    from lerobot.policies.cosmos3.modeling_cosmos3 import Nemotron3DenseVLMoTTextModel

    torch.set_num_threads(4)
    torch.set_default_dtype(torch.float32)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest_path.unlink()

    config = Nemotron3DenseVLMoTTextConfig()
    model = Nemotron3DenseVLMoTTextModel(config).eval()

    prefix = "model.network.mot_backbone.model.language_model."
    state = {}
    with safe_open(str(checkpoint / "model.safetensors"), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if key.startswith(prefix):
                state[key[len(prefix) :]] = handle.get_tensor(key).float()
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise SystemExit(f"unexpected={unexpected}")

    text_len, gen_len = 12, 8
    tokens = torch.tensor([[2, 7071, 2352, 4788, 100, 101, 102, 103, 104, 105, 106, 107]])
    text_mask = torch.ones([1, text_len], dtype=torch.long)
    hidden = config.hidden_size
    index = torch.arange(gen_len, dtype=torch.float32).unsqueeze(1)
    features = torch.arange(hidden, dtype=torch.float32).unsqueeze(0)
    embeds_gen = (torch.sin(index * 0.01 + features * 0.001) * 0.5).unsqueeze(0)

    positions = np.stack(
        [
            np.arange(text_len + gen_len, dtype=np.float32),
            2 * np.arange(text_len + gen_len, dtype=np.float32),
            3 * np.arange(text_len + gen_len, dtype=np.float32),
        ]
    )
    position_ids = torch.from_numpy(positions).unsqueeze(1)

    with torch.no_grad():
        out = model(
            input_ids=tokens,
            attention_mask=text_mask,
            position_ids=position_ids,
            inputs_embeds_gen=embeds_gen,
        )

    manifest = {
        "format_version": 1,
        "storage": "每个张量独立文件，C 连续 float32 little-endian。",
        "checkpoint": str(checkpoint),
        "missing_keys": list(missing),
        "text_len": text_len,
        "gen_len": gen_len,
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

    dump("tokens", tokens.float(), "文本 token id [1,12]")
    dump("text_mask", text_mask.float(), "文本有效位 [1,12]")
    dump("gen_embeds", embeds_gen, "生成流输入 [1,8,2048]")
    dump("positions", torch.from_numpy(positions), "三轴位置 [3,20]")
    dump("text_hidden", out.last_hidden_state, "理解流最终状态 [1,12,2048]")
    dump("gen_hidden", out.last_hidden_state_gen, "生成流最终状态 [1,8,2048]")

    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"完成：{manifest_path}", flush=True)


if __name__ == "__main__":
    main()
