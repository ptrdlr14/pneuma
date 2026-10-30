#!/usr/bin/env python3
"""官方 RemoteClient 的 SmolVLA 服务冒烟；固定噪声数值验收使用独立 parity 测试。"""

import argparse
import sys
import time
import uuid
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lerobot-src", type=Path, required=True)
    parser.add_argument("--endpoint", default="tcp/127.0.0.1:7447")
    parser.add_argument("--deployment", default="libero")
    parser.add_argument("--semantics", default="libero-end-effector-delta-v1")
    args = parser.parse_args()
    if not sys.flags.isolated:
        parser.error("请使用 python -I")
    sys.path.insert(0, str(args.lerobot_src.resolve()))
    import numpy as np
    import torch
    from lerobot.inference import ChunkRequest, ExecutionMode, ObservationSnapshot, RemoteInferenceConfig
    from lerobot.policies.rtc.action_queue import QueueSnapshot
    from lerobot.remote_inference.client import RemoteClient

    config = RemoteInferenceConfig(deployment=args.deployment, semantics=args.semantics,
        chunk_merge="append", mode="chunk", encoding="raw", endpoint=args.endpoint, zenoh_mode="peer")
    client = RemoteClient.connect(config)
    try:
        if not client.artifact_identity.startswith("pneuma-smolvla:"):
            raise AssertionError("需要连接 SmolVLA 服务")
        caps = client.capabilities
        client.admit(features=tuple(caps.features), action_feature=caps.action_feature,
            semantics=args.semantics, action_interval=caps.action_interval, mode="chunk")
        for generation, task in enumerate(["pick up the red cube", "pick up the blue cube"]):
            if generation:
                client.control("reset", generation)
            features = {f.name: np.zeros(tuple(f.shape), dtype=np.float32 if f.dtype == "float32" else np.uint8)
                for f in caps.features}
            observation = ObservationSnapshot(features=features, capture_time=time.time(), task=task,
                task_version=generation, observation_id=uuid.uuid4().hex)
            request = ChunkRequest(observation=observation,
                continuation=QueueSnapshot(generation=generation, cursor=0, index=0,
                    model_actions=None, canonical_actions=None, provenance=()),
                request_id=uuid.uuid4().hex, generation=generation, mode=ExecutionMode.CHUNK,
                delay=0, submitted_at=time.monotonic())
            started = time.monotonic()
            chunk = client.infer(request)
            actions = chunk.canonical_actions
            if tuple(actions.shape) != (caps.execution_steps, *caps.action_feature.shape):
                raise AssertionError(f"错误动作形状 {actions.shape}")
            if actions.dtype != torch.float32 or not torch.isfinite(actions).all().item():
                raise AssertionError("动作必须为有限 float32")
            if chunk.execution_steps != caps.execution_steps:
                raise AssertionError("执行步数与合约不同")
            print(f"infer OK: task={task!r} generation={generation} shape={tuple(actions.shape)} "
                f"roundtrip_ms={(time.monotonic()-started)*1000:.1f}", flush=True)
    finally:
        closed = client.close()
    if not closed:
        raise AssertionError("close 未确认")
    print("SmolVLA remote smoke passed: infer/task update/reset/close", flush=True)


if __name__ == "__main__":
    main()
