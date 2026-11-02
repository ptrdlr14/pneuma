#!/usr/bin/env python3
"""核对单文件权重往返的全量键、shape 与值，并严格加载导出的官方策略。"""

import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--converted", type=Path, required=True)
    parser.add_argument("--lerobot-src", type=Path, required=True)
    args = parser.parse_args()
    if not sys.flags.isolated:
        parser.error("请使用 python -I")
    import torch
    from safetensors.torch import load_file

    source = load_file(args.source / "model.safetensors")
    converted = load_file(args.converted / "model.safetensors")
    if source.keys() != converted.keys():
        raise AssertionError("完整参数键集合不一致")
    changes = 0
    for name, before in source.items():
        after = converted[name]
        if before.shape != after.shape or not torch.equal(before.to(after.dtype), after):
            raise AssertionError(f"参数往返不一致：{name}")
        changes += before.dtype != after.dtype
    sys.path.insert(0, str(args.lerobot_src.resolve()))
    kind = json.loads((args.converted / "config.json").read_text())["type"]
    if kind == "act":
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        config = ACTConfig.from_pretrained(args.converted, local_files_only=True)
        config.device = "cpu"
        policy = ACTPolicy(config)
    elif kind == "smolvla":
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        config = SmolVLAConfig.from_pretrained(args.converted, local_files_only=True)
        config.device = "cpu"
        config.load_vlm_weights = False
        policy = SmolVLAPolicy(config)
    else:
        raise AssertionError(f"不支持策略：{kind}")
    if kind == "act":
        # 旧官方 ACT checkpoint 将归一化缓冲放在顶层；当前 LeRobot 已迁到独立 processor。
        # 全部权重/缓冲在上文逐项比较；网络本体用与官方参照脚本一致的严格入口。
        policy.model.load_state_dict({name.removeprefix("model."): value for name, value in converted.items()
            if name.startswith("model.")}, strict=True)
    else:
        policy.load_state_dict(converted, strict=True)
    print(f"{kind}: {len(source)} tensors exact after dtype conversion; dtype_changes={changes}; official network strict load passed")


if __name__ == "__main__":
    main()
