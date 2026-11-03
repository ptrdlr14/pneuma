#!/usr/bin/env python3
"""导出 Cosmos3 Edge 的序列布局（哪些帧与动作是条件）。"""

import argparse
import json
import numpy as np
from pathlib import Path
import sys
from types import SimpleNamespace


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

    output.mkdir(parents=True, exist_ok=True)
    stub = SimpleNamespace()
    stub.config = SimpleNamespace(mode="wam", n_obs_steps=1)
    stub.network = SimpleNamespace(vae=SimpleNamespace(config=SimpleNamespace(scale_factor_temporal=4)))
    plan = M.OmniMoTModel.build_sequence_plan(
        stub, torch.tensor([[9, 16, 16]]), 33
    )
    video = plan.video_condition_mask[0].tolist()
    actions = plan.action_condition_mask[0].tolist()
    offset = int(plan.action_start_frame_offset[0])
    np.array(video, dtype="<f4").tofile(output / "video.bin")
    np.array(actions, dtype="<f4").tofile(output / "actions.bin")
    np.array([offset], dtype="<f4").tofile(output / "offset.bin")
    print("video", video)
    print("actions", actions)
    print("offset", offset)
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "video_condition": video,
                "action_condition": actions,
                "action_offset": offset,
            },
            indent=2,
        )
    )
    print("完成", flush=True)


if __name__ == "__main__":
    main()
