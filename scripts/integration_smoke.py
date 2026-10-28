"""无硬件联调冒烟：真实 lerobot RemoteClient ↔ pneuma Rust 服务端。

前置：`cargo run -- --config_path=examples/so101.toml` 启动服务端，
本脚本用 main @ f9e9d739e 的客户端类做 connect → admit → infer → close。

运行： .venv/bin/python scripts/integration_smoke.py
"""

import argparse
import time
import uuid

import numpy as np
import torch

from lerobot.inference import ChunkRequest, ExecutionMode, ObservationSnapshot, RemoteInferenceConfig
from lerobot.policies.rtc.action_queue import QueueSnapshot
from lerobot.remote_inference.client import RemoteClient

parser = argparse.ArgumentParser()
parser.add_argument("--endpoint", default="tcp/127.0.0.1:7447")
parser.add_argument("--semantics", default="so101-degrees-gripper-percent-v1")
args = parser.parse_args()

config = RemoteInferenceConfig(
    deployment="manipulation",
    semantics=args.semantics,
    # 骨架只实现 append；auto 会解析成 aligned 而被服务端以 UNSUPPORTED 拒绝。
    chunk_merge="append",
    mode="chunk",
    encoding="raw",
    endpoint=args.endpoint,
    zenoh_mode="peer",
)

client = RemoteClient.connect(config)
print(f"[1] describe OK: instance={client.instance_id}")

caps = client.capabilities
client.admit(
    features=tuple(caps.features),
    action_feature=caps.action_feature,
    semantics=config.semantics,
    action_interval=caps.action_interval,
    mode="chunk",
)
print(f"[2] admit OK: session={client.session_id}")


def zero_feature(spec):
    dtype = np.float32 if spec.dtype == "float32" else np.uint8
    return np.zeros(tuple(spec.shape), dtype=dtype)


# 观测按服务端通告的 caps 派生；profile 的名字/维度/形状不二次发明。
observation = ObservationSnapshot(
    features={spec.name: zero_feature(spec) for spec in caps.features},
    capture_time=time.time(),
    task="pick up the cup",
    task_version=0,
    observation_id=uuid.uuid4().hex,
)
request = ChunkRequest(
    observation=observation,
    continuation=QueueSnapshot(
        generation=0, cursor=0, index=0,
        model_actions=None, canonical_actions=None, provenance=(),
    ),
    request_id=uuid.uuid4().hex,
    generation=0,
    mode=ExecutionMode.CHUNK,
    delay=0,
    submitted_at=time.monotonic(),
)

started = time.monotonic()
chunk = client.infer(request)
elapsed = time.monotonic() - started
print(
    f"[3] infer OK: canonical_actions shape={tuple(chunk.canonical_actions.shape)} "
    f"dtype={chunk.canonical_actions.dtype} sum={float(chunk.canonical_actions.sum()):.1f} "
    f"roundtrip={elapsed * 1000:.1f} ms"
)

client.control("close", 0)
print("[4] close OK — smoke passed")
