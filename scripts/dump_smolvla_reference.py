#!/usr/bin/env python3
"""生成官方 SmolVLA CPU/F32 参照；必须以 python -I 从可信仓库运行。

四个路径参数均必填。输出 manifest.json 描述逐张量的 f32 小端、C 连续
二进制文件；tokens、mask、原始 RGB 也无损转为 f32，统一 Rust 读取方式。
checkpoint/ 是明确修正过配置的测试 fixture，不能据此声称原 checkpoint
可直接运行。只复制 JSON，权重与统计量使用符号链接，不修改源模型。
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def make_fixture(checkpoint, vlm_path, output, load_file):
    """保留所有原始 JSON 的摘要，逐项记录 fixture 的修改。"""
    original = json.loads((checkpoint / "config.json").read_text())
    pre = json.loads((checkpoint / "policy_preprocessor.json").read_text())
    normalizer = next(s for s in pre["steps"] if s["registry_name"] == "normalizer_processor")
    stats = load_file(str(checkpoint / normalizer["state_file"]), device="cpu")
    shape = original["input_features"]["observation.state"]["shape"]
    stats_shape = list(stats["observation.state.mean"].shape)
    if stats_shape != [8] or list(stats["observation.state.std"].shape) != [8]:
        raise ValueError(f"本参照要求 state8 统计量，实际 mean={stats_shape}")
    conflict = {"config_state_shape": shape, "stats_state_shape": stats_shape}
    if shape != stats_shape:
        print(f"原 checkpoint 配置冲突：state={shape}，归一化统计量={stats_shape}；仅修正输出 fixture。", flush=True)
    if original["num_steps"] != 10:
        raise ValueError("本参照要求保持源配置 num_steps=10，不会暗中覆盖")
    if original["chunk_size"] != 50 or original["max_action_dim"] != 32:
        raise ValueError("本参照要求完整固定 noise 形状 [1,50,32]")
    fixture = output / "checkpoint"
    fixture.mkdir(exist_ok=True)
    changes = []
    hashes = {}
    links = {}

    def patch(value, filename, pointer=""):
        if isinstance(value, dict):
            for key, old in list(value.items()):
                location = f"{pointer}/{key}"
                new = old
                if key == "observation.state" and isinstance(old, dict) and "shape" in old:
                    new = dict(old, shape=[8])
                elif key == "rename_map":
                    new = {}
                elif key == "device":
                    new = "cpu"
                elif key in ("vlm_model_name", "tokenizer_name"):
                    new = str(vlm_path)
                elif key == "load_vlm_weights":
                    new = False
                if new != old:
                    changes.append({"file": filename, "pointer": location, "original": old, "fixture": new})
                    value[key] = new
                patch(value[key], filename, location)
        elif isinstance(value, list):
            for index, entry in enumerate(value):
                patch(entry, filename, f"{pointer}/{index}")

    for source in sorted(checkpoint.glob("*.json")):
        hashes[source.name] = sha256(source)
        value = json.loads(source.read_text())
        patch(value, source.name)
        target = fixture / source.name
        if target.is_symlink():
            raise ValueError(f"拒绝覆盖 fixture 中已有 JSON 符号链接：{target}")
        write_json(target, value)
    for source in sorted(checkpoint.glob("*.safetensors")):
        target = fixture / source.name
        if target.is_symlink():
            if target.resolve() != source.resolve():
                raise ValueError(f"fixture 符号链接指向其他文件：{target}")
        elif target.exists():
            raise ValueError(f"拒绝覆盖已有 fixture 权重：{target}")
        else:
            target.symlink_to(source.resolve())
        links[source.name] = str(source.resolve())
    return fixture, {
        "original_checkpoint_directly_supported": False,
        "reason": "原始 state6 配置与 state8 统计量冲突，且含非空 rename_map 与未解析的 VLM/tokenizer 名称。",
        "state_conflict": conflict,
        "source_json_sha256": hashes,
        "modifications": changes,
        "safetensors_symlinks": links,
    }


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
        p.expanduser().resolve() for p in (args.checkpoint, args.vlm_path, args.lerobot_src, args.output)
    )
    if output == checkpoint or checkpoint in output.parents or output in checkpoint.parents:
        parser.error("输出目录与源 checkpoint 必须互不包含")
    for path in (checkpoint, vlm_path, lerobot_src):
        if not path.is_dir():
            parser.error(f"目录不存在：{path}")
    if not (lerobot_src / "lerobot/policies/smolvla/modeling_smolvla.py").is_file():
        parser.error("--lerobot-src 必须指向含 lerobot/ 的官方 src 目录")

    # 显式使用用户授权的官方源码；离线加载防止隐式下载或远端代码执行。
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    sys.path.insert(0, str(lerobot_src))
    import numpy as np
    import torch
    from safetensors.torch import load_file
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
    import lerobot.policies.smolvla.processor_smolvla  # 注册官方处理器。

    torch.set_num_threads(4)
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(0)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest_path.unlink()  # 失败时不保留上一次的成功标记。
    fixture, provenance = make_fixture(checkpoint, vlm_path, output, load_file)
    config = SmolVLAConfig.from_pretrained(str(fixture), local_files_only=True)
    config.vlm_model_name = str(vlm_path)
    config.load_vlm_weights = False
    config.device = "cpu"
    assert config.num_steps == 10
    print("构建官方 SmolVLAPolicy CPU/F32，严格加载完整 checkpoint ...", flush=True)
    policy = SmolVLAPolicy(config)
    weights = load_file(str(fixture / "model.safetensors"), device="cpu")
    loaded = policy.load_state_dict(weights, strict=True)
    weight_count = len(weights)
    del weights
    policy.float().eval()
    if any(p.is_floating_point() and p.dtype != torch.float32 for p in policy.parameters()):
        raise ValueError("policy 中仍存在非 F32 参数")

    # 从保存的处理器配置和统计量加载，不手写归一化或 tokenizer 参数。
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        fixture, config_filename="policy_preprocessor.json", local_files_only=True,
        overrides={"tokenizer_processor": {"tokenizer_name": str(vlm_path)},
                   "device_processor": {"device": "cpu"}},
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        fixture, config_filename="policy_postprocessor.json", local_files_only=True,
        to_transition=policy_action_to_transition, to_output=transition_to_policy_action,
    )
    manifest = {
        "format_version": 1,
        "storage": "每个张量独立文件，C 连续 float32 little-endian；整数和 bool 无损转换为 f32。",
        "checkpoint": "checkpoint",
        "source_checkpoint": str(checkpoint), "vlm_path": str(vlm_path),
        "lerobot_src": str(lerobot_src),
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "safetensors", "transformers")},
        "upstream_source_sha256": {
            name: sha256(lerobot_src / "lerobot/policies/smolvla" / name)
            for name in ("configuration_smolvla.py", "modeling_smolvla.py", "smolvlm_with_expert.py", "processor_smolvla.py")
        },
        "fixture_provenance": provenance,
        "strict_weight_load": {"keys": weight_count, "missing": list(loaded.missing_keys), "unexpected": list(loaded.unexpected_keys)},
        "task": "pick up the red cube", "device": "cpu", "dtype": "float32",
        "num_steps": config.num_steps, "threads": torch.get_num_threads(),
        "noise_generation": "torch.randn([1,50,32], generator=torch.Generator(device='cpu').manual_seed(0)); 使用落盘 noise.bin 比较",
        "prefix_kv_layout": "[batch, heads, sequence, head_dim]，K 在 RoPE 后，V 未经 RoPE",
        "tensors": {},
    }

    def dump(name, tensor, description):
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor)
        original_dtype = str(tensor.dtype)
        array = tensor.detach().cpu().to(torch.float32).contiguous().numpy().astype("<f4", copy=False)
        if not np.isfinite(array).all():
            raise ValueError(f"张量 {name} 含非有限数值")
        path = output / f"{name}.bin"
        array.tofile(path)
        manifest["tensors"][name] = {
            "file": path.name, "shape": list(array.shape), "dtype": "float32", "byte_order": "little",
            "source_dtype": original_dtype, "description": description,
            "sha256": sha256(path), "min": float(array.min()), "max": float(array.max()),
            "mean": float(array.astype(np.float64).mean()), "first_values": array.reshape(-1)[:8].tolist(),
        }
        print(f"{name}: {list(array.shape)} min={array.min():.8g} max={array.max():.8g}", flush=True)

    raw_state = torch.tensor([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], dtype=torch.float32)
    dump("raw_state", raw_state, "原始 state8")
    batch = {"observation.state": raw_state.clone(), "task": manifest["task"]}
    camera_keys = list(config.image_features)
    if len(camera_keys) != 3:
        raise ValueError(f"本参照需要三相机，实际 {camera_keys}")
    manifest["camera_keys"] = camera_keys
    yy, xx = np.indices((256, 256), dtype=np.int32)
    for index, key in enumerate(camera_keys):
        # 每相机、每通道使用不同的固定 RGB 图案，不依赖随机数库版本。
        image = np.stack(((xx + 17 * index) % 256,
                          (3 * yy + 29 * index) % 256,
                          (xx + 2 * yy + 43 * index) % 256), axis=-1).astype(np.uint8)
        dump(f"raw_image_{index}", torch.from_numpy(image), f"{key} 原始 HWC RGB，0..255")
        # 环境到官方 processor 的输入契约是 CHW float RGB，0..1。
        batch[key] = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    batch = preprocessor(batch)
    dump("state_normalized", batch["observation.state"], "官方 preprocessor 输出 state")
    dump("state_padded", policy.prepare_state(batch), "官方 prepare_state 补零到 32 维")
    dump("tokens", batch["observation.language.tokens"], "官方 tokenizer token IDs")
    dump("token_mask", batch["observation.language.attention_mask"], "官方 tokenizer attention mask")
    manifest["processed_task"] = batch["task"]
    images, image_masks = policy.prepare_images(batch)
    for index, key in enumerate(camera_keys):
        dump(f"image_normalized_{index}", batch[key], f"{key} 官方 preprocessor 输出 BCHW，0..1")
        dump(f"image_prepared_{index}", images[index], f"{key} 官方 prepare_images 输出 BCHW，-1..1")
        dump(f"image_mask_{index}", image_masks[index], f"{key} 有效相机 mask")
    noise = torch.randn((1, 50, 32), generator=torch.Generator(device="cpu").manual_seed(0))
    dump("noise", noise, "完整固定初始 noise，不通过重新播种替代")

    # 只观测官方完整 predict_action_chunk 链路，不重写采样积分。
    model = policy.model
    vlm = model.vlm_with_expert
    observed = set()

    def capture_vision(_module, _inputs, result):
        if "vision_hidden" not in observed:
            dump("vision_hidden", result.last_hidden_state, "首相机 vision_model.last_hidden_state")
            observed.add("vision_hidden")

    def capture_connector(_module, _inputs, result):
        if "connector" not in observed:
            dump("connector", result, "首相机 connector 输出，未乘 sqrt(hidden_size)")
            observed.add("connector")

    original_forward = vlm.forward
    original_denoise = model.denoise_step
    original_sample = model.sample_actions
    step_count = 0
    timesteps = []

    def capture_forward(*forward_args, **kwargs):
        result = original_forward(*forward_args, **kwargs)
        if kwargs.get("past_key_values") is None and "prefix" not in observed:
            cache = result[1]
            if cache is None:
                raise ValueError("参照需要 use_cache=True")
            for index, layer in enumerate(cache.layers):
                dump(f"prefix_key_{index}", layer.keys, f"prefix 第 {index} 层 RoPE 后 K，BHLD")
                dump(f"prefix_value_{index}", layer.values, f"prefix 第 {index} 层 V，BHLD")
            manifest["prefix_layers"] = len(cache.layers)
            observed.add("prefix")
        return result

    def capture_denoise(*denoise_args, **kwargs):
        nonlocal step_count
        result = original_denoise(*denoise_args, **kwargs)
        timesteps.append(float(kwargs["timestep"].reshape(-1)[0]))
        if step_count == 0:
            dump("velocity_step_0", result, "首步 timestep=1 的完整 velocity [1,50,32]")
        step_count += 1
        print(f"官方 Euler 步 {step_count}/{config.num_steps}", flush=True)
        return result

    def capture_sample(*sample_args, **kwargs):
        result = original_sample(*sample_args, **kwargs)
        dump("actions_padded", result, "官方积分最终归一化动作，含 padding [1,50,32]")
        return result

    handles = [vlm.get_vlm_model().vision_model.register_forward_hook(capture_vision),
               vlm.get_vlm_model().connector.register_forward_hook(capture_connector)]
    vlm.forward, model.denoise_step, model.sample_actions = capture_forward, capture_denoise, capture_sample
    try:
        with torch.inference_mode():
            normalized = policy.predict_action_chunk(batch, noise=noise.clone())
            canonical = postprocessor(normalized.clone())
    finally:
        vlm.forward, model.denoise_step, model.sample_actions = original_forward, original_denoise, original_sample
        for handle in handles:
            handle.remove()
    if step_count != 10:
        raise ValueError(f"官方积分步数不是 10：{step_count}")
    dump("actions_normalized", normalized, "官方 policy 去除 padding 后的归一化动作 [1,50,7]")
    dump("actions_canonical", canonical, "官方 postprocessor 反归一化后的 canonical 动作 [1,50,7]")
    manifest["denoise_timesteps"] = timesteps
    for name, expected in provenance["source_json_sha256"].items():
        if sha256(checkpoint / name) != expected:
            raise ValueError(f"源 checkpoint JSON 在生成期间发生变化：{name}")
    manifest["source_json_unchanged"] = True
    write_json(manifest_path, manifest)
    print(f"参照生成成功：{manifest_path}；{len(manifest['tensors'])} 个张量；严格加载 {weight_count} 个权重键。", flush=True)
    print("首个 canonical 动作：", canonical[0, 0].tolist(), flush=True)


if __name__ == "__main__":
    main()
