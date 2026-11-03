#!/usr/bin/env python3
"""导出 π₀.₅ 的固定输入数值参照，供 Rust 侧逐值比较。

需要 Python 3.12、LeRobot 的 pi05 依赖（torch、transformers、safetensors）。
脚本离线运行，不下载任何资源；`--vlm-path` 指向本地 PaliGemma 分词器目录。

输出目录包含 manifest.json 与每个张量的独立 .bin 文件（C 连续 float32）；
失败时不保留上一次的 manifest。
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vlm-path", type=Path, required=True)
    parser.add_argument("--lerobot-src", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not sys.flags.isolated:
        parser.error("读取下载模型必须使用 python -I")
    checkpoint, vlm_path, lerobot_src, output = (
        path.expanduser().resolve()
        for path in (args.checkpoint, args.vlm_path, args.lerobot_src, args.output)
    )
    for path in (checkpoint, vlm_path, lerobot_src):
        if not path.is_dir():
            parser.error(f"目录不存在：{path}")
    if not (lerobot_src / "lerobot/policies/pi05/modeling_pi05.py").is_file():
        parser.error("--lerobot-src 必须指向含 lerobot/ 的官方 src 目录")

    # 显式使用用户授权的官方源码；离线加载防止隐式下载。
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    sys.path.insert(0, str(lerobot_src))
    import numpy as np
    import torch
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05 import PI05Policy

    torch.set_num_threads(4)
    torch.set_default_dtype(torch.float32)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest_path.unlink()

    print("加载官方 PI05Policy CPU/F32 ...", flush=True)
    policy = PI05Policy.from_pretrained(str(checkpoint), local_files_only=True)
    policy.float().eval()
    config = policy.config
    if any(p.is_floating_point() and p.dtype != torch.float32 for p in policy.parameters()):
        raise ValueError("policy 中仍存在非 F32 参数")

    preprocessor, postprocessor = make_pre_post_processors(
        config,
        str(checkpoint),
        preprocessor_overrides={
            "tokenizer_processor": {"tokenizer_name": str(vlm_path)},
            "device_processor": {"device": "cpu"},
        },
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )

    manifest = {
        "format_version": 1,
        "storage": "每个张量独立文件，C 连续 float32 little-endian。",
        "checkpoint": str(checkpoint),
        "vlm_path": str(vlm_path),
        "lerobot_src": str(lerobot_src),
        "chunk_size": config.chunk_size,
        "max_action_dim": config.max_action_dim,
        "num_inference_steps": config.num_inference_steps,
        "image_resolution": list(config.image_resolution),
        "task": "pick up the black bowl and place it on the plate",
        "tensors": {},
    }

    def dump(name, tensor, description):
        tensor = torch.as_tensor(tensor)
        source_dtype = str(tensor.dtype)
        array = (
            tensor.detach().cpu().to(torch.float32).contiguous().numpy().astype("<f4", copy=False)
        )
        if not np.isfinite(array).all():
            raise ValueError(f"张量 {name} 含非有限数值")
        path = output / f"{name}.bin"
        array.tofile(path)
        manifest["tensors"][name] = {
            "shape": list(array.shape),
            "source_dtype": source_dtype,
            "description": description,
            "sha256": sha256(path),
            "first_values": array.reshape(-1)[:8].tolist(),
        }
        print(f"{name}: {list(array.shape)}", flush=True)

    state = torch.tensor(
        [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], dtype=torch.float32
    )
    dump("raw_state", state, "原始 state8")
    # 补零相机由模型内部补齐；参照只提供真实相机。
    camera_keys = [key for key in config.image_features if "empty_camera" not in key]
    if len(camera_keys) != 2:
        raise ValueError(f"本参照需要两路相机，实际 {camera_keys}")
    manifest["camera_keys"] = camera_keys
    yy, xx = np.indices((256, 256), dtype=np.int32)
    batch = {"observation.state": state.clone(), "task": manifest["task"]}
    for index, key in enumerate(camera_keys):
        image = np.stack(
            (
                (xx + 17 * index) % 256,
                (3 * yy + 29 * index) % 256,
                (xx + 2 * yy + 43 * index) % 256,
            ),
            axis=-1,
        ).astype(np.uint8)
        dump(f"raw_image_{index}", torch.from_numpy(image), f"{key} 原始 HWC RGB")
        batch[key] = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

    batch = preprocessor(batch)
    dump("state_normalized", batch["observation.state"], "官方 preprocessor 输出 state")
    dump("tokens", batch["observation.language.tokens"], "token IDs")
    dump("token_mask", batch["observation.language.attention_mask"], "attention mask")
    manifest["processed_task"] = batch["task"]

    images, image_masks = policy._preprocess_images(batch)  # noqa: SLF001
    for index, image in enumerate(images):
        dump(f"prepared_image_{index}", image, f"camera {index} 进入 SigLIP 的 [-1,1] 图像")
        dump(f"camera_mask_{index}", image_masks[index], f"camera {index} 有效位")

    generator = torch.Generator(device="cpu").manual_seed(0)
    noise = torch.randn(
        [1, config.chunk_size, config.max_action_dim], generator=generator, dtype=torch.float32
    )
    dump("noise", noise, "固定初始噪声")

    with torch.no_grad():
        prefix_embs, _, _ = policy.model.embed_prefix(
            images,
            image_masks,
            batch["observation.language.tokens"],
            batch["observation.language.attention_mask"],
        )
    dump("prefix_embs", prefix_embs, "前缀嵌入：相机 token 后接语言 token")

    with torch.no_grad():
        actions = policy.model.sample_actions(
            images,
            image_masks,
            batch["observation.language.tokens"],
            batch["observation.language.attention_mask"],
            noise=noise,
        )
    dump("actions_model", actions, "模型坐标动作 [1, chunk, max_action_dim]")
    with torch.no_grad():
        step1 = policy.model.sample_actions(
            images,
            image_masks,
            batch["observation.language.tokens"],
            batch["observation.language.attention_mask"],
            noise=noise,
            num_steps=1,
        )
    dump("actions_step1", step1, "一次 Euler 步后的动作")
    action_dim = config.output_features["action"].shape[0]
    canonical = postprocessor(actions[:, :, :action_dim])
    dump("actions_canonical", canonical, "反归一化后的 7 维动作")

    manifest["versions"] = {
        name: __import__("importlib.metadata", fromlist=["version"]).version(name)
        for name in ("torch", "numpy", "safetensors", "transformers")
    }
    manifest["source_sha256"] = {
        name: sha256(lerobot_src / "lerobot/policies/pi05" / name)
        for name in (
            "configuration_pi05.py",
            "modeling_pi05.py",
            "processor_pi05.py",
        )
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"完成：{manifest_path}", flush=True)


if __name__ == "__main__":
    main()
