#!/usr/bin/env python3
"""用官方 Cosmos3 策略跑一次生成，导出动作作为对拍参照。"""

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
        parser.error("必须使用 python -I")
    checkpoint, lerobot_src, output = (
        path.expanduser().resolve() for path in (args.checkpoint, args.lerobot_src, args.output)
    )
    os.environ["HF_HUB_OFFLINE"] = "1"
    sys.path.insert(0, str(lerobot_src))
    import torch
    from safetensors import safe_open
    from lerobot.policies.cosmos3.configuration_cosmos3 import Cosmos3EdgeConfig
    from lerobot.policies.cosmos3.modeling_cosmos3 import Cosmos3Policy, FlowState

    torch.set_num_threads(8)
    torch.set_default_dtype(torch.float32)
    output.mkdir(parents=True, exist_ok=True)

    config = Cosmos3EdgeConfig()
    config.fps = 20.0
    config.guidance_scale = 1.0
    config.mode = "wam"
    config.use_state = True
    config.chunk_size = 32
    config.n_action_steps = 32
    from lerobot.configs.types import FeatureType, PolicyFeature

    config.input_features = {
        "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        "observation.images.image2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
    }
    config.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))
    }
    import json as _json

    _source = _json.loads((checkpoint / "config.json").read_text())
    config.domain_id = int(_source.get("domain_id", 0))
    config.num_inference_steps = int(_source.get("num_inference_steps", 4))
    config.sampling_shift = float(_source.get("sampling_shift", 5.0))
    print("domain_id", config.domain_id)
    policy = Cosmos3Policy(config).eval()

    state = {}
    with safe_open(str(checkpoint / "model.safetensors"), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if key.startswith("model."):
                state[key] = handle.get_tensor(key).float()
    missing, unexpected = policy.load_state_dict(state, strict=False)
    if unexpected:
        raise SystemExit(f"unexpected={unexpected[:5]}")

    device = torch.device("cpu")
    tokens = torch.tensor([[2, 5, 6, 7, 8]])
    canvas = torch.zeros([1, 3, 1, 256, 256])
    state64 = torch.zeros([1, 64])
    noise = FlowState(
        torch.zeros([1, 48, 9, 16, 16]),
        torch.zeros([1, 33, 64]),
    )
    with torch.no_grad():
        actions, latents = policy.model.generate(
            input_ids=tokens,
            pixel_values_videos_gen=canvas,
            video_grid_thw_gen=torch.tensor([[9, 16, 16]]),
            second_per_grid_ts_gen=torch.tensor([0.2]),
            state=state64,
            noise=noise,
        )
    # 第一步的原始速度，用于分段定位。
    model = policy.model
    video_grid = torch.tensor([[9, 16, 16]])
    seconds = torch.tensor([0.2])
    plan = model.build_sequence_plan(video_grid, 33)
    packed = model.pack_actions(torch.zeros([1, 32, 64]), state64)
    clean = model.encode_clean(canvas, video_grid, packed, plan)
    ones = torch.ones([1])
    noisy = model.add_noise(clean, noise, ones, ones, plan)
    position_ids = model.network.get_rope_index(
        tokens, video_grid, seconds, num_actions=33,
        action_start_frame_offset=plan.action_start_frame_offset,
    )
    first = 999.0
    with torch.no_grad():
        velocity = model.predict_velocity(
            noisy,
            torch.full([1], first),
            torch.full([1], first),
            plan,
            tokens,
            position_ids,
        )
    print("velocity", list(velocity.video_latents.shape), list(velocity.actions.shape))
    velocity.video_latents.numpy().astype("<f4").tofile(output / "velocity_video.bin")
    velocity.actions.numpy().astype("<f4").tofile(output / "velocity_actions.bin")
    with torch.no_grad():
        video_embeds = model.network.embed_video(
            noisy.video_latents, torch.full([1], first), plan.video_condition_mask
        )
        action_embeds = model.network.embed_actions(
            noisy.actions,
            torch.full([1], first),
            plan.action_condition_mask,
            torch.zeros([1], dtype=torch.long),
        )
        merged = model.network.merge_gen_stream(video_embeds, action_embeds)
        out = model.network(
            input_ids=tokens,
            position_ids=position_ids,
            video_latents=noisy.video_latents,
            video_timesteps=torch.full([1], first),
            video_condition_mask=plan.video_condition_mask,
            actions=noisy.actions,
            action_timesteps=torch.full([1], first),
            action_condition_mask=plan.action_condition_mask,
        )
    print("merged", list(merged.shape), "hidden", list(out.last_hidden_state_gen.shape))
    merged[0, -33:].numpy().astype("<f4").tofile(output / "action_embeds.bin")
    out.last_hidden_state_gen[0, -33:].numpy().astype("<f4").tofile(output / "action_hidden.bin")
    # 逐步轨迹：每步的动作状态与速度，用于定位采样段差异。
    sched_v = model.make_scheduler()
    sched_a = model.make_scheduler()
    x = noisy
    steps_state, steps_velocity = [x.actions.clone()], []
    with torch.no_grad():
        for timestep in sched_v.timesteps:
            v = model.predict_velocity(
                x, timestep.expand(1), timestep.expand(1), plan, tokens, position_ids
            )
            v = model.mask_velocity(v, plan)
            steps_velocity.append(v.actions.clone())
            x = FlowState(
                sched_v.step(v.video_latents, timestep, x.video_latents).prev_sample,
                sched_a.step(v.actions, timestep, x.actions).prev_sample,
            )
            steps_state.append(x.actions.clone())
    torch.stack(steps_state).numpy().astype("<f4").tofile(output / "steps_state.bin")
    torch.stack(steps_velocity).numpy().astype("<f4").tofile(output / "steps_velocity.bin")
    print("steps", [list(s.shape) for s in (steps_state[0], steps_velocity[0])])
    print("actions", list(actions.shape), "latents", list(latents.shape))
    actions.numpy().astype("<f4").tofile(output / "actions.bin")
    latents.numpy().astype("<f4").tofile(output / "latents.bin")
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "sha256": {"actions": sha256(output / "actions.bin")},
                "tokens": tokens.flatten().tolist(),
                "guidance_scale": config.guidance_scale,
            },
            indent=2,
        )
    )
    print("完成", flush=True)


if __name__ == "__main__":
    main()
