#!/usr/bin/env python3
"""LIBERO 闭环评测：官方 Python SmolVLA 与 Pneuma，逐局原子保存并支持续跑。"""

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
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parent.parent
SEMANTICS = "libero-end-effector-delta-v1"
CAMERAS = {"image": "observation.images.camera1", "image2": "observation.images.camera2"}


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def prepare_checkpoint(args):
    """8 维官方状态与两路实际相机，修改记录与源摘要随结果保存。"""
    original = json.loads((args.checkpoint / "config.json").read_text())
    if original["type"] != "smolvla" or original["empty_cameras"] != 0:
        raise ValueError("评测入口要求 SmolVLA 且 empty_cameras=0")
    inputs = original["input_features"]
    if not all(name in inputs for name in CAMERAS.values()):
        raise ValueError("checkpoint 缺少 LIBERO 主相机或腕部相机")
    from safetensors import safe_open
    pre = json.loads((args.checkpoint / "policy_preprocessor.json").read_text())
    normalizer = next(s for s in pre["steps"] if s["registry_name"] == "normalizer_processor")
    with safe_open(args.checkpoint / normalizer["state_file"], framework="pt") as stats:
        for key in ["observation.state.mean", "observation.state.std"]:
            if list(stats.get_slice(key).get_shape()) != [8]:
                raise ValueError(f"官方 LIBERO state8 与统计 {key} 不匹配")
    directory = args.output.parent / (args.output.stem + "-checkpoint")
    directory.mkdir(exist_ok=True)
    changes = []
    hashes = {}

    def patch(value, filename, pointer=""):
        if isinstance(value, dict):
            for key, old in list(value.items()):
                location = pointer + "/" + key
                if key.startswith("observation.images.") and key not in CAMERAS.values():
                    changes.append({"file": filename, "pointer": location, "before": old, "after": "removed: absent in official LIBERO observations"})
                    del value[key]
                    continue
                new = old
                if key == "observation.state" and isinstance(old, dict) and "shape" in old:
                    new = dict(old, shape=[8])
                elif key == "rename_map":
                    new = {}
                elif key == "device":
                    new = "cpu"
                elif key == "load_vlm_weights":
                    new = False
                if new != old:
                    changes.append({"file": filename, "pointer": location, "before": old, "after": new})
                    value[key] = new
                patch(value[key], filename, location)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                patch(item, filename, pointer + f"/{index}")

    for name in ["config.json", "policy_preprocessor.json", "policy_postprocessor.json"]:
        source = args.checkpoint / name
        hashes[name] = file_hash(source)
        value = json.loads(source.read_text())
        patch(value, name)
        destination = directory / name
        if destination.is_symlink():
            raise ValueError(f"拒绝覆盖 JSON 符号链接 {destination}")
        save(destination, value)
    for source in args.checkpoint.glob("*.safetensors"):
        destination = directory / source.name
        if destination.is_symlink():
            if destination.resolve() != source.resolve():
                raise ValueError(f"已有链接指向其他资源 {destination}")
        elif destination.exists():
            raise ValueError(f"拒绝覆盖已有模型文件 {destination}")
        else:
            destination.symlink_to(source.resolve())
        hashes[source.name] = file_hash(source)
    return directory, {"source_sha256": hashes, "modifications": changes,
        "state": "eef position(3), quaternion to axis-angle(3), gripper qpos(2)",
        "images": "official LiberoProcessorStep rotates both image axes; camera1=agentview, camera2=wrist; camera3 omitted as official prepare_images(empty_cameras=0)",
        "original_checkpoint_directly_supported": False}


def write_profile(args, checkpoint):
    config = json.loads((checkpoint / "config.json").read_text())
    text = f'''deployment = "libero"
semantics = "{SEMANTICS}"
[model]
policy = "smolvla"
repo_or_path = {json.dumps(str(checkpoint))}
vlm_path = {json.dumps(str(args.vlm_path))}
device = "metal"
[execution]
action_fps = 20.0
action_deadline_s = 120.0
idle_timeout_s = 120.0
warmup_calls = 1
n_action_steps = {args.action_steps}
[zenoh]
mode = "peer"
listen_endpoints = ["tcp/127.0.0.1:7447"]
'''
    for name, feature in config["input_features"].items():
        visual = feature["type"] == "VISUAL"
        shape = [feature["shape"][1], feature["shape"][2], 3] if visual else feature["shape"]
        text += f'''\n[[features]]
name = {json.dumps(name)}
shape = {json.dumps(shape)}
dtype = "{'uint8' if visual else 'float32'}"
kind = "{'rgb' if visual else 'tensor'}"
semantics = {json.dumps('libero-' + name + '-v1')}
'''
    text += f'''\n[action_feature]
name = "action"
shape = [7]
dtype = "float32"
semantics = "{SEMANTICS}"
'''
    path = args.output.parent / (args.output.stem + "-server.toml")
    path.write_text(text)
    return path


@contextmanager
def managed_server(args, backend, profile):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    endpoint = f"tcp/127.0.0.1:{port}"
    ready = threading.Event()
    path = args.output.parent / f"{args.output.stem}-{backend}.log"
    with path.open("a") as log:
        process = subprocess.Popen([str(args.server), f"--config={profile}",
            f"--device={backend}", f"--listen={endpoint}"], cwd=ROOT,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            env={**os.environ, "RUST_LOG": "info", "NO_COLOR": "1"})
        def collect():
            for line in process.stdout:
                log.write(line)
                log.flush()
                if "policy server ready" in line:
                    ready.set()
            ready.set()
        reader = threading.Thread(target=collect, daemon=True)
        reader.start()
        try:
            if not ready.wait(180) or process.poll() is not None:
                raise RuntimeError(f"{backend} 启动失败或超时，见 {path}")
            yield endpoint
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
            reader.join(timeout=5)


@contextmanager
def predictor(args, backend, checkpoint, profile):
    if backend == "python":
        from safetensors.torch import load_file
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.processor import PolicyProcessorPipeline
        from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
        import lerobot.policies.smolvla.processor_smolvla  # 注册保存的 processor。
        cfg = SmolVLAConfig.from_pretrained(checkpoint, local_files_only=True)
        cfg.vlm_model_name = str(args.vlm_path)
        cfg.load_vlm_weights = False
        cfg.device = "cpu"
        policy = SmolVLAPolicy(cfg)
        policy.load_state_dict(load_file(checkpoint / "model.safetensors"), strict=True)
        policy.float().eval()
        pre = PolicyProcessorPipeline.from_pretrained(checkpoint, config_filename="policy_preprocessor.json",
            overrides={"tokenizer_processor": {"tokenizer_name": str(args.vlm_path)}, "device_processor": {"device": "cpu"}})
        post = PolicyProcessorPipeline.from_pretrained(checkpoint, config_filename="policy_postprocessor.json",
            to_transition=policy_action_to_transition, to_output=transition_to_policy_action)
        def predict(features, task, generation):
            batch = {name: torch.from_numpy(value).permute(2, 0, 1).float() / 255. if value.ndim == 3
                else torch.from_numpy(value) for name, value in features.items()}
            batch["task"] = task
            with torch.inference_mode():
                actions = post(policy.predict_action_chunk(pre(batch)))
            return actions[0, :args.action_steps].numpy(), None
        yield predict, lambda generation, seed: (policy.reset(), torch.manual_seed(seed))
    else:
        from lerobot.inference import ChunkRequest, ExecutionMode, ObservationSnapshot, RemoteInferenceConfig
        from lerobot.remote_inference.client import RemoteClient
        from lerobot.policies.rtc.action_queue import QueueSnapshot
        with managed_server(args, backend, profile) as endpoint:
            client = RemoteClient.connect(RemoteInferenceConfig(deployment="libero", semantics=SEMANTICS,
                chunk_merge="append", mode="chunk", encoding="raw", endpoint=endpoint, zenoh_mode="peer"))
            try:
                caps = client.capabilities
                if caps.execution_steps != args.action_steps or caps.action_feature.shape != (7,):
                    raise ValueError("服务端动作合约与评测不符")
                client.admit(features=tuple(caps.features), action_feature=caps.action_feature,
                    semantics=SEMANTICS, action_interval=caps.action_interval, mode="chunk")
                def predict(features, task, generation):
                    observation = ObservationSnapshot(features=features, capture_time=time.time(), task=task,
                        task_version=generation, observation_id=uuid.uuid4().hex)
                    request = ChunkRequest(observation=observation, continuation=QueueSnapshot(generation=generation,
                        cursor=0, index=0, model_actions=None, canonical_actions=None, provenance=()),
                        request_id=uuid.uuid4().hex, generation=generation, mode=ExecutionMode.CHUNK,
                        delay=0, submitted_at=time.monotonic())
                    chunk = client.infer(request)
                    return chunk.canonical_actions.numpy(), chunk.server_durations.get("worker")
                yield predict, lambda generation, seed: client.control("reset", generation)
            finally:
                client.close()


def observation_features(obs, processor):
    state = obs["robot_state"]
    batched = {"observation.robot_state": {
        "eef": {"pos": torch.from_numpy(state["eef"]["pos"]).unsqueeze(0),
                "quat": torch.from_numpy(state["eef"]["quat"]).unsqueeze(0)},
        "gripper": {"qpos": torch.from_numpy(state["gripper"]["qpos"]).unsqueeze(0)}}}
    for key, target in CAMERAS.items():
        batched[target] = torch.from_numpy(obs["pixels"][key].copy()).permute(2, 0, 1).unsqueeze(0)
    processed = processor.observation(batched)
    return {name: value[0].permute(1, 2, 0).contiguous().numpy() if value.ndim == 4
        else value[0].numpy().astype(np.float32) for name, value in processed.items()}


def run_episode(args, env, predict, reset, processor, task_id, episode, generation):
    seed = args.seed + task_id * 1000 + episode
    reset(generation, seed)
    obs, _ = env.reset(seed=seed)
    features = observation_features(obs, processor)
    digest = hashlib.sha256()
    for name, value in features.items():
        digest.update(name.encode())
        digest.update(value.tobytes())
    initial_obs = digest.hexdigest()
    initial_state = fileless_hash(env._env.sim.get_state().flatten())
    actions = []
    cursor = 0
    latencies, workers = [], []
    success = False
    started = time.monotonic()
    for step in range(1, env._max_episode_steps + 1):
        if cursor == len(actions):
            before = time.perf_counter()
            actions, worker = predict(features, env.task_description, generation)
            latencies.append((time.perf_counter() - before) * 1000)
            if worker is not None:
                workers.append(worker * 1000)
            if actions.shape != (args.action_steps, 7) or not np.isfinite(actions).all():
                raise ValueError(f"invalid action chunk {actions.shape}")
            cursor = 0
        obs, _, terminated, truncated, info = env.step(actions[cursor])
        cursor += 1
        success = success or bool(info["is_success"])
        if success or terminated or truncated:
            break
        if cursor == len(actions):
            features = observation_features(obs, processor)
    return {"task_id": task_id, "task": env.task, "language": env.task_description,
        "episode": episode, "init_state_index": episode, "seed": seed,
        "initial_observation_sha256": initial_obs, "initial_state_sha256": initial_state,
        "success": success, "steps": step, "wall_s": time.monotonic() - started,
        "request_ms": latencies, "worker_ms": workers}


def fileless_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def wilson(successes, count):
    if not count:
        return None
    z = 1.959963984540054
    p = successes / count
    denominator = 1 + z*z/count
    center = (p + z*z/(2*count)) / denominator
    radius = z*math.sqrt(p*(1-p)/count + z*z/(4*count*count)) / denominator
    return [center-radius, center+radius]


def summary(episodes):
    count = len(episodes)
    wins = sum(e["success"] for e in episodes)
    def latency(key):
        values = [x for ep in episodes for x in ep[key]]
        return {"count": len(values), "median_ms": float(np.median(values)), "p95_ms": float(np.percentile(values, 95))} if values else None
    return {"episodes": count, "successes": wins, "success_rate": wins/count if count else None,
        "wilson_95": wilson(wins, count), "request_latency": latency("request_ms"), "worker_latency": latency("worker_ms"),
        "per_task": {str(task): {"successes": sum(e["success"] for e in episodes if e["task_id"] == task),
            "episodes": sum(e["task_id"] == task for e in episodes)} for task in sorted({e["task_id"] for e in episodes})}}


def update_summaries(report):
    report["summary"] = {backend: summary(episodes) for backend, episodes in report["episodes"].items()}
    if "python" in report["episodes"]:
        baseline = {(e["task_id"],e["episode"]): e for e in report["episodes"]["python"]}
        report["paired"] = {}
        for backend, episodes in report["episodes"].items():
            if backend == "python":
                continue
            pairs = [(baseline[(e["task_id"],e["episode"])], e) for e in episodes if (e["task_id"],e["episode"]) in baseline]
            differences = [int(rs["success"])-int(py["success"]) for py,rs in pairs]
            report["paired"][backend] = {"pairs": len(pairs),
                "initial_state_mismatches": [[rs["task_id"],rs["episode"]] for py,rs in pairs if py["initial_state_sha256"] != rs["initial_state_sha256"]],
                "initial_observation_mismatches": [[rs["task_id"],rs["episode"]] for py,rs in pairs if py["initial_observation_sha256"] != rs["initial_observation_sha256"]],
                "success_rate_difference": sum(differences)/len(differences) if differences else None,
                "rust_only_successes": differences.count(1), "python_only_successes": differences.count(-1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vlm-path", type=Path, required=True)
    parser.add_argument("--lerobot-src", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/libero-spatial.json")
    parser.add_argument("--server", type=Path, default=ROOT/"target/release/pneuma")
    parser.add_argument("--backend", choices=["all","python","metal","flex"], default="all")
    parser.add_argument("--suite", choices=["libero_spatial","libero_object","libero_goal","libero_10","libero_90"], default="libero_spatial")
    parser.add_argument("--tasks", help="逗号分隔任务编号；默认整个 suite")
    parser.add_argument("--episodes-per-task", type=int, default=5)
    parser.add_argument("--action-steps", type=int, default=50)
    parser.add_argument("--max-steps", type=int, help="默认官方 suite 上限")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not sys.flags.isolated:
        parser.error("请使用 python -I")
    if args.episodes_per_task <= 0 or args.action_steps <= 0 or args.threads <= 0 or (args.max_steps is not None and args.max_steps <= 0):
        parser.error("局数、执行步数、线程数和步数上限必须为正")
    for name in ["checkpoint","vlm_path","lerobot_src","output","server"]:
        setattr(args, name, getattr(args,name).expanduser().resolve())
    if args.output.exists() and not args.resume:
        parser.error("结果已存在；使用 --resume 或指定新的 --output")
    if args.output.parent == args.checkpoint or args.checkpoint in args.output.parents:
        parser.error("输出不能位于原 checkpoint 内")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")
    sys.path.insert(0, str(args.lerobot_src))
    global np, torch
    import numpy as np
    import torch
    from lerobot.envs.libero import _get_suite, LiberoEnv, TASK_SUITE_MAX_STEPS
    from lerobot.processor.env_processor import LiberoProcessorStep
    torch.set_num_threads(args.threads)
    suite = _get_suite(args.suite)
    tasks = sorted(set(map(int,args.tasks.split(",")))) if args.tasks else list(range(suite.n_tasks))
    if not tasks or any(t < 0 or t >= suite.n_tasks for t in tasks):
        parser.error("任务编号超出 suite 范围")
    checkpoint, provenance = prepare_checkpoint(args)
    profile = write_profile(args, checkpoint)
    dimensions = json.loads((checkpoint/"config.json").read_text())["input_features"]["observation.images.camera1"]["shape"]
    if args.action_steps > json.loads((checkpoint/"config.json").read_text())["chunk_size"]:
        parser.error("action-steps 超过 chunk_size")
    from libero.libero import get_libero_path
    import libero.libero.envs.env_wrapper as libero_wrapper
    task_assets = {}
    for task_id in tasks:
        task = suite.get_task(task_id)
        task_assets[str(task_id)] = {
            "bddl_sha256": file_hash(Path(get_libero_path("bddl_files"))/task.problem_folder/task.bddl_file),
            "init_states_sha256": file_hash(Path(get_libero_path("init_states"))/task.problem_folder/task.init_states_file),
        }
    metadata = {"suite": args.suite, "tasks": tasks, "episodes_per_task": args.episodes_per_task,
        "seed": args.seed, "action_steps": args.action_steps, "max_steps": args.max_steps or TASK_SUITE_MAX_STEPS[args.suite],
        "settle_steps": 10, "control_freq": 20, "control_mode": "relative", "image_shape": dimensions,
        "python_device": "cpu", "dtype": "float32", "threads": args.threads,
        "noise": "independent standard normal per chunk; Python seeded per episode, Rust backend RNG is not paired",
        "environment_time": "simulation steps, not real-time scheduling", "provenance": provenance,
        "versions": {name: version(name) for name in ["torch","numpy","transformers","mujoco","robosuite","eclipse-zenoh"]},
        "platform": platform.platform(), "server_sha256": file_hash(args.server),
        "libero_env_wrapper_sha256": file_hash(libero_wrapper.__file__), "task_assets": task_assets,
        "vlm_assets_sha256": {name: file_hash(args.vlm_path/name) for name in ["config.json","tokenizer.json","tokenizer_config.json"]},
        "harness_sha256": file_hash(Path(__file__)),
        "upstream_sha256": {name: file_hash(args.lerobot_src/"lerobot"/name) for name in ["envs/libero.py","processor/env_processor.py","policies/smolvla/modeling_smolvla.py","policies/smolvla/smolvlm_with_expert.py"]}}
    if args.resume:
        report = json.loads(args.output.read_text())
        if report["metadata"] != metadata:
            raise ValueError("续跑元数据不同，拒绝混入不同配置或版本")
    else:
        report = {"metadata": metadata, "episodes": {}, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    backends = ["python","metal"] if args.backend == "all" else [args.backend]
    save(args.output, report)
    for backend in backends:
        episodes = report["episodes"].setdefault(backend, [])
        completed = {(e["task_id"],e["episode"]) for e in episodes}
        if len(completed) == len(tasks)*args.episodes_per_task:
            continue
        with predictor(args, backend, checkpoint, profile) as (predict, reset):
            generation = 0
            for task_id in tasks:
                for episode in range(args.episodes_per_task):
                    if (task_id,episode) in completed:
                        continue
                    env = LiberoEnv(suite,task_id,args.suite,obs_type="pixels_agent_pos",episode_index=episode,
                        observation_height=dimensions[1], observation_width=dimensions[2],
                        episode_length=metadata["max_steps"], control_freq=20, num_steps_wait=10)
                    try:
                        generation += 1
                        result = run_episode(args,env,predict,reset,LiberoProcessorStep(),task_id,episode,generation)
                    finally:
                        env.close()
                    episodes.append(result)
                    update_summaries(report)
                    save(args.output, report)
                    current = report["summary"][backend]
                    print(f"{backend} task={task_id} episode={episode} success={result['success']} steps={result['steps']} "
                        f"wall_s={result['wall_s']:.1f} cumulative={current['successes']}/{current['episodes']}",flush=True)
    report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    update_summaries(report)
    save(args.output,report)
    print(json.dumps(report["summary"],ensure_ascii=False,indent=2),flush=True)


if __name__ == "__main__":
    main()
