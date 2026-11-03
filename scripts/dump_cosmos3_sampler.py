#!/usr/bin/env python3
"""导出 Cosmos3 Edge uniPC 采样的参照轨迹。

用 diffusers 的 `UniPCMultistepScheduler` 跑一个固定的人造速度场，逐步记录样本。
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

STEPS, SHIFT, TRAIN = 4, 5.0, 1000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not sys.flags.isolated:
        parser.error("必须使用 python -I")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    import torch
    from diffusers import UniPCMultistepScheduler

    torch.set_default_dtype(torch.float32)
    scheduler = UniPCMultistepScheduler(
        num_train_timesteps=TRAIN, prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=1.0
    )
    sigma_max = float(np.float32(1.0 - 1.0 / TRAIN))
    raw = np.linspace(sigma_max, 0.0, STEPS + 1)[:-1]
    scheduler.set_timesteps(sigmas=SHIFT * raw / (1.0 + (SHIFT - 1.0) * raw))

    sample = torch.tensor([[[0.5, -0.25, 0.75]]])
    trajectory = [sample.flatten().tolist()]
    for timestep in scheduler.timesteps:
        velocity = sample * 0.5 + 0.1 * math.sin(float(timestep) / TRAIN)
        sample = scheduler.step(velocity, timestep, sample).prev_sample
        trajectory.append(sample.flatten().tolist())

    manifest = {"format_version": 1, "tensors": {}, "steps": STEPS, "shift": SHIFT, "train": TRAIN}
    values = {
        "sigmas": np.array(scheduler.sigmas.tolist(), dtype="<f4"),
        "timesteps": np.array(scheduler.timesteps.tolist(), dtype="<f4"),
        "trajectory": np.array(trajectory, dtype="<f4"),
    }
    for name, array in values.items():
        path = output / f"{name}.bin"
        array.tofile(path)
        manifest["tensors"][name] = {
            "shape": list(array.shape),
            "sha256": sha256(path),
        }
        print(f"{name}: {list(array.shape)}", flush=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"完成：{output / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
