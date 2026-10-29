"""ALOHA Transfer Cube 配对评测：官方 Python ACT、Rust Flex、Rust Metal。

.venv/bin/python scripts/sim_rollout.py --backend=all --episodes=50
结果逐局保存；使用同一命令加 --resume 可续跑。--backend=remote 连接已有服务。
"""

import argparse
from contextlib import contextmanager
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import socket
import subprocess
import time
import uuid

import gymnasium as gym
import gym_aloha  # noqa: F401 注册 gym_aloha/* 环境
import numpy as np
import torch

from dump_act_reference import checkpoint_dir, load_reference_policy, predict_reference
from lerobot.inference import ChunkRequest, ExecutionMode, ObservationSnapshot, RemoteInferenceConfig
from lerobot.remote_inference.client import RemoteClient
from lerobot.policies.rtc.action_queue import QueueSnapshot

ROOT = Path(__file__).resolve().parent.parent
SEMANTICS = "aloha-sim-14d-v1"
ENV_ID = "gym_aloha/AlohaTransferCube-v0"
TASK = "transfer cube"


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def observation_hash(obs):
    digest = hashlib.sha256()
    digest.update(obs["agent_pos"].astype("<f4").tobytes())
    digest.update(obs["pixels"]["top"].tobytes())
    return digest.hexdigest()


def make_client(endpoint):
    config = RemoteInferenceConfig(
        deployment="manipulation", semantics=SEMANTICS,
        chunk_merge="append", mode="chunk", encoding="raw",
        endpoint=endpoint, zenoh_mode="peer",
    )
    client = RemoteClient.connect(config)
    try:
        caps = client.capabilities
        client.admit(
            features=tuple(caps.features), action_feature=caps.action_feature,
            semantics=SEMANTICS, action_interval=caps.action_interval, mode="chunk",
        )
    except BaseException:
        client.close()
        raise
    return client


def remote_predict(client, obs):
    observation = ObservationSnapshot(
        features={
            "observation.state": obs["agent_pos"].astype(np.float32),
            "observation.images.top": obs["pixels"]["top"],
        },
        capture_time=time.time(), task=TASK, task_version=0,
        observation_id=uuid.uuid4().hex,
    )
    request = ChunkRequest(
        observation=observation,
        continuation=QueueSnapshot(
            generation=0, cursor=0, index=0, model_actions=None,
            canonical_actions=None, provenance=(),
        ),
        request_id=uuid.uuid4().hex, generation=0,
        mode=ExecutionMode.CHUNK, delay=0, submitted_at=time.monotonic(),
    )
    chunk = client.infer(request)
    return chunk.canonical_actions.numpy(), chunk.server_durations.get("worker")


@contextmanager
def managed_server(args, backend, checkpoint):
    # 只持有和终止本次创建的子进程；端口冲突会作为启动失败报告。
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    endpoint = f"tcp/127.0.0.1:{port}"
    log_path = args.output.parent / f"{args.output.stem}-{backend}.log"
    with log_path.open("w+") as log:
        command = [
            str(args.server.resolve()), "--config=examples/aloha_sim.toml",
            f"--checkpoint={checkpoint}", f"--device={backend}",
            f"--action-steps={args.action_steps}",
            f"--warmup-calls={args.warmup_calls}",
            f"--listen={endpoint}",
        ]
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            env={**os.environ, "RUST_LOG": "info", "NO_COLOR": "1"},
        )
        try:
            deadline = time.monotonic() + 180
            while True:
                log.seek(0)
                output = log.read()
                if process.poll() is not None:
                    raise RuntimeError(f"{backend} server exited: {output[-4000:]}")
                if "policy server ready" in output:
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError(f"{backend} startup timed out; see {log_path}")
                time.sleep(0.1)
            yield endpoint
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


@contextmanager
def predictor(args, backend, checkpoint):
    if backend == "python":
        policy, raw = load_reference_policy(str(checkpoint))

        def predict(obs):
            actions, _ = predict_reference(
                policy, raw, obs["agent_pos"].astype(np.float32), obs["pixels"]["top"],
            )
            return actions[0, :args.action_steps].numpy(), None

        for _ in range(args.warmup_calls):
            predict({"agent_pos": np.zeros(14, np.float32),
                     "pixels": {"top": np.zeros((480, 640, 3), np.uint8)}})
        yield predict
        return

    @contextmanager
    def connected(endpoint):
        client = make_client(endpoint)
        try:
            caps = client.capabilities
            if (caps.execution_steps != args.action_steps
                    or caps.action_feature.shape != (14,)
                    or abs(caps.action_interval - 0.02) > 1e-8):
                raise ValueError("服务合约必须匹配执行步数、14 维动作和 50 Hz")
            yield lambda obs: remote_predict(client, obs)
        finally:
            client.close()

    if backend == "remote":
        with connected(args.endpoint) as predict:
            yield predict
    else:
        with managed_server(args, backend, checkpoint) as endpoint:
            with connected(endpoint) as predict:
                yield predict


def run_episode(predict, env, seed, action_steps):
    obs, _ = env.reset(seed=seed)
    initial_hash = observation_hash(obs)
    # qpos/qvel 包括方块位置，避免只比较机器人关节而漏掉任务初始条件。
    physics = env.unwrapped._env.physics
    initial_state_hash = hashlib.sha256(physics.get_state().tobytes()).hexdigest()
    actions = []
    cursor = 0
    latencies = []
    worker_ms = []
    success = False
    max_reward = 0.0
    started = time.monotonic()
    for step in range(1, env.spec.max_episode_steps + 1):
        if cursor == len(actions):
            before = time.perf_counter()
            actions, worker = predict(obs)
            latencies.append((time.perf_counter() - before) * 1000)
            if actions.shape != (action_steps, 14) or not np.isfinite(actions).all():
                raise ValueError(f"Invalid action chunk: {actions.shape}")
            if worker is not None:
                worker_ms.append(worker * 1000)
            cursor = 0
        obs, reward, terminated, truncated, info = env.step(actions[cursor])
        cursor += 1
        success = success or bool(info["is_success"])
        max_reward = max(max_reward, float(reward))
        if terminated or truncated:
            break
    return {
        "seed": seed, "initial_observation_sha256": initial_hash,
        "initial_state_sha256": initial_state_hash,
        "steps": step, "success": success, "max_reward": max_reward,
        "wall_s": time.monotonic() - started,
        "request_ms": latencies, "worker_ms": worker_ms,
    }


def wilson(successes, count):
    z = 1.959963984540054
    p = successes / count
    denom = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denom
    radius = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denom
    return [max(0.0, center - radius), min(1.0, center + radius)]


def latency_summary(values):
    if not values:
        return None
    return {"count": len(values), "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)), "max": max(values)}


def summarize(results):
    summary = {}
    baseline = {r["seed"]: r for r in results.get("python", [])}
    for backend, rows in results.items():
        if not rows:
            continue
        successes = sum(r["success"] for r in rows)
        entry = {
            "episodes": len(rows), "successes": successes,
            "success_rate": successes / len(rows),
            "success_rate_wilson95": wilson(successes, len(rows)),
            "request_ms": latency_summary([x for r in rows for x in r["request_ms"]]),
            "worker_ms": latency_summary([x for r in rows for x in r["worker_ms"]]),
        }
        if backend != "python" and baseline:
            pairs = [(baseline[r["seed"]], r) for r in rows if r["seed"] in baseline]
            if pairs:
                if any(a["initial_state_sha256"] != b["initial_state_sha256"] for a, b in pairs):
                    raise ValueError("配对初始物理状态不一致，不能比较成功率")
                differences = np.array([int(b["success"]) - int(a["success"]) for a, b in pairs])
                rng = np.random.default_rng(0)
                boot = rng.choice(differences, size=(10000, len(pairs))).mean(axis=1)
                entry["paired_vs_python"] = {
                    "episodes": len(pairs),
                    "initial_image_mismatches": [b["seed"] for a, b in pairs
                                                 if a["initial_observation_sha256"] != b["initial_observation_sha256"]],
                    "success_rate_delta": float(differences.mean()),
                    "delta_bootstrap95": np.quantile(boot, [0.025, 0.975]).tolist(),
                    "wins": int((differences > 0).sum()), "losses": int((differences < 0).sum()),
                    "disagreements": [b["seed"] for a, b in pairs if a["success"] != b["success"]],
                }
        summary[backend] = entry
    return summary


def save_report(path, report):
    report["summary"] = summarize(report["results"])
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def metadata(args, checkpoint):
    packages = ["torch", "numpy", "gymnasium", "gym-aloha", "mujoco", "dm-control", "lerobot"]
    tracked = subprocess.check_output(["git", "ls-files", "src", "Cargo.lock", "Cargo.toml"], cwd=ROOT, text=True).splitlines()
    tracked += ["scripts/sim_rollout.py", "scripts/dump_act_reference.py", "examples/aloha_sim.toml"]
    return {
        "environment": ENV_ID, "seeds": list(range(args.seed_start, args.seed_start + args.episodes)),
        "backends": ["python", "flex", "metal"] if args.backend == "all" else [args.backend],
        "action_steps": args.action_steps, "prediction_steps": args.prediction_steps,
        "warmup_calls": args.warmup_calls, "torch_threads": args.torch_threads,
        "python_device": "cpu", "fps": 50, "image_shape": [480, 640, 3],
        "episode_length": args.episode_length, "torch_deterministic_algorithms": True,
        "chunk_merge": "append", "temporal_ensemble": False,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": {name: file_hash(checkpoint / name) for name in ["config.json", "model.safetensors"]},
        "server_sha256": file_hash(args.server) if args.backend in {"all", "flex", "metal"} else None,
        "remote_endpoint": args.endpoint if args.backend == "remote" else None,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": {name: file_hash(ROOT / name) for name in tracked},
        "platform": platform.platform(), "machine": platform.machine(),
        "python": platform.python_version(), "packages": {name: version(name) for name in packages},
        "notes": ["仿真按步推进，不衡量真实时间控制违约。",
                  "Python request_ms 是本地前后处理和推理；Rust request_ms 另含协议与传输。",
                  "worker_ms 为服务端计时，含模型前后处理和回复张量构造。",
                  "描述性评测；尚未约定非劣性偏差，置信区间不构成验收通过。"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["all", "python", "flex", "metal", "remote"], default="all")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--episode-length", type=int, default=400)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--action-steps", type=int)
    parser.add_argument("--warmup-calls", type=int, default=2)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--server", type=Path, default=ROOT / "target/release/pneuma")
    parser.add_argument("--endpoint", default="tcp/127.0.0.1:7447")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/aloha-paired.json")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.episodes, args.episode_length, args.warmup_calls, args.torch_threads) < 1 or args.seed_start < 0:
        parser.error("episodes、warmup-calls、torch-threads 必须为正数，seed-start 非负")
    checkpoint = (args.checkpoint or Path(checkpoint_dir())).resolve()
    config = json.loads((checkpoint / "config.json").read_text())
    args.prediction_steps = config["chunk_size"]
    if args.action_steps is None:
        args.action_steps = config["n_action_steps"]
    if not 1 <= args.action_steps <= args.prediction_steps:
        parser.error("action-steps 必须在 1..chunk_size 内")
    if config.get("temporal_ensemble_coeff") is not None:
        parser.error("本评测只支持无 temporal ensembling 的 plain chunk checkpoint")
    torch.set_num_threads(args.torch_threads)
    torch.use_deterministic_algorithms(True)
    current_metadata = metadata(args, checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.resume:
        report = json.loads(args.output.read_text())
        if report["metadata"] != current_metadata:
            parser.error("续跑配置、源文件、checkpoint 或依赖版本发生变化；请使用新的输出文件")
    else:
        if args.output.exists():
            parser.error("输出文件已存在；使用 --resume 或指定新的 --output")
        report = {"metadata": current_metadata, "results": {}}
    save_report(args.output, report)
    for backend in current_metadata["backends"]:
        rows = report["results"].setdefault(backend, [])
        done = {row["seed"] for row in rows}
        pending = [seed for seed in current_metadata["seeds"] if seed not in done]
        if not pending:
            continue
        with predictor(args, backend, checkpoint) as predict:
            env = gym.make(
                ENV_ID, obs_type="pixels_agent_pos", observation_height=480,
                observation_width=640, max_episode_steps=args.episode_length,
            )
            try:
                for seed in pending:
                    result = run_episode(predict, env, seed, args.action_steps)
                    rows.append(result)
                    save_report(args.output, report)
                    print(f"{backend} seed={seed} steps={result['steps']} success={result['success']} "
                          f"wall={result['wall_s']:.1f}s completed={len(rows)}/{args.episodes}", flush=True)
            finally:
                env.close()
    print(json.dumps(report["summary"], indent=2), flush=True)
    print(f"Report: {args.output}", flush=True)


if __name__ == "__main__":
    main()
