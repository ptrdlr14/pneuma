#!/usr/bin/env python3
"""导出 Cosmos3 Edge 生成流嵌入的参照张量。

直接调用 LeRobot cosmos3 分支里 `Cosmos3VFMNetwork` 的嵌入方法，只加载小权重。
"""

import argparse
import hashlib
import json
import os
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
    sys.path.insert(0, str(lerobot_src))
    import torch
    from diffusers.models.embeddings import TimestepEmbedding
    from safetensors import safe_open
    from lerobot.policies.cosmos3 import modeling_cosmos3 as M

    torch.set_num_threads(4)
    torch.set_default_dtype(torch.float32)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest_path.unlink()

    hidden = 2048
    stub = SimpleNamespace(config=SimpleNamespace(latent_patch_size=2, timestep_scale=0.001))
    stub.time_proj = M.SinusoidalTimesteps(256)
    stub.time_embedder = TimestepEmbedding(256, hidden)
    stub.vae2llm = torch.nn.Linear(192, hidden)
    stub.llm2vae = torch.nn.Linear(hidden, 192)
    stub.action2llm = M.DomainAwareLinear(64, hidden, 32)
    stub.llm2action = M.DomainAwareLinear(hidden, 64, 32)
    stub.action_modality_embed = M.ModalityEmbedding(hidden)
    for name in ("patchify", "embed_timesteps"):
        setattr(stub, name, getattr(M.Cosmos3VFMNetwork, name).__get__(stub))
    modules = {
        "time_embedder": stub.time_embedder,
        "vae2llm": stub.vae2llm,
        "llm2vae": stub.llm2vae,
        "action2llm": stub.action2llm,
        "llm2action": stub.llm2action,
        "action_modality_embed": stub.action_modality_embed,
    }

    prefix = "model.network."
    with safe_open(str(checkpoint / "model.safetensors"), framework="pt", device="cpu") as handle:
        for name, module in modules.items():
            state = {}
            for key in handle.keys():
                if key.startswith(prefix + name + "."):
                    state[key[len(prefix + name + ".") :]] = handle.get_tensor(key).float()
            missing, unexpected = module.load_state_dict(state, strict=False)
            if missing or unexpected:
                raise SystemExit(f"{name}: missing={missing} unexpected={unexpected}")

    frames, height, width, channels = 1, 16, 16, 48
    yy, xx = np.indices((height, width), dtype=np.int32)
    latents = np.stack(
        [
            np.stack(
                [(xx + 3 * channel) % 16 + 0.5 * yy for channel in range(channels)], axis=0
            )
        ],
        axis=1,
    ).astype(np.float32)
    latents = torch.from_numpy(latents).unsqueeze(0)  # [1,C,1,H,W]
    video_timesteps = torch.tensor([[0.7]])
    video_mask = torch.zeros([1, 1], dtype=torch.bool)
    tokens, action_dim = 4, 64
    index = torch.arange(tokens * action_dim, dtype=torch.float32)
    actions = (torch.sin(index * 0.01) * 0.5).reshape(1, tokens, action_dim)
    action_timesteps = torch.full([1, tokens], 0.7)
    action_mask = torch.zeros([1, tokens], dtype=torch.bool)
    domain = torch.zeros([1], dtype=torch.long)

    with torch.no_grad():
        video_embeds = M.Cosmos3VFMNetwork.embed_video(stub, latents, video_timesteps, video_mask)
        action_embeds = M.Cosmos3VFMNetwork.embed_actions(
            stub, actions, action_timesteps, action_mask, domain
        )
        merged = M.Cosmos3VFMNetwork.merge_gen_stream(stub, video_embeds, action_embeds)

        # 反向路径：视频隐状态 → 速度，动作隐状态 → 速度。
        video_hidden = torch.sin(torch.arange(video_embeds.shape[1] * hidden, dtype=torch.float32)
                                 .reshape(1, -1, hidden) * 0.001)
        action_hidden = torch.cos(torch.arange(tokens * hidden, dtype=torch.float32)
                                  .reshape(1, tokens, hidden) * 0.002)
        video_patches = stub.llm2vae(video_hidden)
        video_velocity = M.Cosmos3VFMNetwork.unpatchify(
            stub, video_patches, torch.Size([1, channels, frames, height, width])
        )
        action_velocity = stub.llm2action(action_hidden, domain)

    manifest = {"format_version": 1, "storage": "float32 little-endian，每个张量独立文件。", "tensors": {}}

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
        }
        print(f"{name}: {list(array.shape)}", flush=True)

    dump("latents", latents, "视频隐变量 [1,48,1,16,16]")
    dump("video_timesteps", video_timesteps, "视频时间步 [1,1]")
    dump("actions", actions, "动作 [1,4,64]")
    dump("action_timesteps", action_timesteps, "动作时间步 [1,4]")
    dump("video_embeds", video_embeds, "视频 token 嵌入 [1,256,2048]")
    dump("action_embeds", action_embeds, "动作 token 嵌入 [1,4,2048]")
    dump("merged", merged, "生成流 [1,260,2048]")
    dump("video_velocity", video_velocity, "视频速度 [1,48,1,16,16]")
    dump("action_velocity", action_velocity, "动作速度 [1,4,64]")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"完成：{manifest_path}", flush=True)


if __name__ == "__main__":
    main()
