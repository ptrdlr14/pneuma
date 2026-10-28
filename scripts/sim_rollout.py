"""M3 无硬件闭环：gym-aloha transfer-cube ↔ pneuma ACT 服务端（官方 RemoteClient）。

rollout CLI 是 robot-only、eval 不接 remote，本脚本即「无硬件全链路」正式替身：
env obs → ObservationSnapshot → infer → 逐步 env.step，记录 roundtrip 与成功率。

前置：./target/release/pneuma --config_path=examples/aloha_sim.toml --model.repo_or_path=<ckpt>
运行：.venv/bin/python scripts/sim_rollout.py [--episodes N]
"""

import argparse
import statistics
import time
import uuid

import gymnasium as gym
import gym_aloha  # noqa: F401  注册 gym_aloha/* 环境
import numpy as np

from lerobot.inference import ChunkRequest, ExecutionMode, ObservationSnapshot, RemoteInferenceConfig
from lerobot.remote_inference.client import RemoteClient
from lerobot.policies.rtc.action_queue import QueueSnapshot

SEMANTICS = "aloha-sim-14d-v1"
ENV_ID = "gym_aloha/AlohaTransferCube-v0"
TASK = "transfer cube"


def make_client() -> RemoteClient:
    config = RemoteInferenceConfig(
        deployment="manipulation",
        semantics=SEMANTICS,
        # 服务端只实现 append；auto 会解析成 aligned 被 UNSUPPORTED 拒绝。
        chunk_merge="append",
        mode="chunk",
        encoding="raw",
        endpoint="tcp/127.0.0.1:7447",
        zenoh_mode="peer",
    )
    client = RemoteClient.connect(config)
    caps = client.capabilities
    # 语义/特征全等校验由 client.admit 执行（不等则抛错），此处不重复。
    client.admit(
        features=tuple(caps.features),
        action_feature=caps.action_feature,
        semantics=SEMANTICS,
        action_interval=caps.action_interval,
        mode="chunk",
    )
    return client


def snapshot(obs: dict) -> ObservationSnapshot:
    # 合约耦合点：agent_pos→state(float32)、pixels.top→image(uint8 HWC)。
    return ObservationSnapshot(
        features={
            "observation.state": obs["agent_pos"].astype(np.float32),
            "observation.images.top": obs["pixels"]["top"],
        },
        capture_time=time.time(),
        task=TASK,
        task_version=0,
        observation_id=uuid.uuid4().hex,
    )


def run_episode(client: RemoteClient, env, seed: int, latencies: list[float]):
    obs, _ = env.reset(seed=seed)
    queue: list[np.ndarray] = []
    steps = 0
    success = False
    while steps < env.spec.max_episode_steps:
        if not queue:
            request = ChunkRequest(
                observation=snapshot(obs),
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
            latencies.append((time.monotonic() - started) * 1000)
            queue = [a for a in chunk.canonical_actions]
        obs, _, terminated, truncated, info = env.step(queue.pop(0))
        steps += 1
        success = success or bool(info["is_success"])
        if terminated or truncated:
            break
    return steps, success


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    client = make_client()
    env = gym.make(ENV_ID, obs_type="pixels_agent_pos")
    latencies: list[float] = []
    successes = 0
    try:
        for ep in range(args.episodes):
            steps, success = run_episode(client, env, seed=ep, latencies=latencies)
            successes += int(success)
            print(f"episode {ep}: steps={steps} success={success}")
    finally:
        client.control("close", 0)
        env.close()

    print(
        f"success {successes}/{args.episodes} | roundtrip ms "
        f"min={min(latencies):.0f} med={statistics.median(latencies):.0f} max={max(latencies):.0f}"
        f" | inferences={len(latencies)}"
    )


if __name__ == "__main__":
    main()
