"""无硬件 echo 冒烟：真实 lerobot RemoteClient ↔ pneuma Rust 服务端。

前置：`cargo run -- serve --config=examples/so101.toml` 启动 echo 服务端。
客户端对齐官方 main @ f9e9d739e；ACT 数值验收使用独立脚本。

运行：.venv/bin/python scripts/integration_smoke.py
本地 oracle 回归（不连接服务）：追加 --check-oracle
"""

import argparse
import time
import unittest
import uuid

import numpy as np
import torch

from lerobot.inference import ChunkRequest, ExecutionMode, ObservationSnapshot, RemoteInferenceConfig
from lerobot.policies.rtc.action_queue import QueueSnapshot
from lerobot.remote_inference.client import RemoteClient
from lerobot.remote_inference.protocol import ErrorCode, ProtocolError


def assert_echo_actions(actions, execution_steps, expected_shape):
    """echo 的权威结果是通告形状的 float32 全零块，不能只检查总和。"""
    if tuple(actions.shape) != expected_shape:
        raise AssertionError(f"action shape {tuple(actions.shape)} != {expected_shape}")
    if execution_steps != expected_shape[0]:
        raise AssertionError(f"execution_steps {execution_steps} != {expected_shape[0]}")
    if actions.dtype != torch.float32:
        raise AssertionError(f"action dtype {actions.dtype} != torch.float32")
    if not torch.all(actions == 0).item():
        raise AssertionError("echo actions must be zero at every element")


class EchoOracleTests(unittest.TestCase):
    def test_zero_chunk_passes(self):
        assert_echo_actions(torch.zeros((2, 2)), 2, (2, 2))

    def test_finite_nonzero_chunk_fails(self):
        with self.assertRaises(AssertionError):
            assert_echo_actions(torch.ones((2, 2)), 2, (2, 2))

    def test_cancelling_values_fail(self):
        actions = torch.tensor([[1.0, -1.0], [0.0, 0.0]])
        self.assertEqual(actions.sum().item(), 0)
        with self.assertRaises(AssertionError):
            assert_echo_actions(actions, 2, (2, 2))

    def test_contract_mismatch_fails(self):
        for actions, steps in (
            (torch.zeros((1, 2)), 2),
            (torch.zeros((2, 2)), 1),
            (torch.zeros((2, 2), dtype=torch.float64), 2),
        ):
            with self.subTest(shape=tuple(actions.shape), steps=steps, dtype=actions.dtype):
                with self.assertRaises(AssertionError):
                    assert_echo_actions(actions, steps, (2, 2))


def zero_feature(spec):
    dtype = np.float32 if spec.dtype == "float32" else np.uint8
    return np.zeros(tuple(spec.shape), dtype=dtype)


def infer_echo(client, generation):
    caps = client.capabilities
    # 观测与期望动作均取通告合约，不二次发明 profile 的名字和维度。
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
            generation=generation, cursor=0, index=0,
            model_actions=None, canonical_actions=None, provenance=(),
        ),
        request_id=uuid.uuid4().hex,
        generation=generation,
        mode=ExecutionMode.CHUNK,
        delay=0,
        submitted_at=time.monotonic(),
    )
    started = time.monotonic()
    chunk = client.infer(request)
    elapsed = time.monotonic() - started
    assert_echo_actions(
        chunk.canonical_actions,
        chunk.execution_steps,
        (caps.execution_steps, *caps.action_feature.shape),
    )
    print(
        f"infer OK: generation={generation} shape={tuple(chunk.canonical_actions.shape)} "
        f"dtype={chunk.canonical_actions.dtype} all_zero=True "
        f"roundtrip={elapsed * 1000:.1f} ms"
    )


def smoke(config):
    previous_session = None
    for cycle in range(2):
        client = RemoteClient.connect(config)
        try:
            # 默认身份明确限定 echo，避免误对 ACT 应用全零期望。
            if client.artifact_identity != "pneuma-echo-v0":
                raise AssertionError("echo smoke requires artifact pneuma-echo-v0 (examples/so101.toml)")
            caps = client.capabilities
            client.admit(
                features=tuple(caps.features),
                action_feature=caps.action_feature,
                semantics=config.semantics,
                action_interval=caps.action_interval,
                mode="chunk",
            )
            if client.session_id == previous_session:
                raise AssertionError("reopen must allocate a new session_id")
            print(f"admit OK: cycle={cycle} instance={client.instance_id} session={client.session_id}")
            infer_echo(client, 0)
            if cycle == 0:
                client.control("reset", 1)
                try:
                    client.control("status", 0)
                except ProtocolError as error:
                    if error.code is not ErrorCode.STALE:
                        raise
                else:
                    raise AssertionError("status at obsolete generation must return Stale")
                infer_echo(client, 1)
                print("reset/stale status OK")
            previous_session = client.session_id
        finally:
            # close() 同时关闭会话和 transport；下一轮使用新客户端。
            closed = client.close()
        if not closed:
            raise AssertionError("session close was not acknowledged")
        print(f"close OK: cycle={cycle}")
    print("echo lifecycle smoke passed")


def main():
    parser = argparse.ArgumentParser(description="Echo-only protocol and zero-action smoke")
    parser.add_argument("--endpoint", default="tcp/127.0.0.1:7447")
    parser.add_argument("--semantics", default="so101-degrees-gripper-percent-v1")
    parser.add_argument("--check-oracle", action="store_true", help="仅运行本地 oracle 回归，不连接服务")
    args = parser.parse_args()
    if args.check_oracle:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(EchoOracleTests)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if result.wasSuccessful() else 1
    config = RemoteInferenceConfig(
        deployment="manipulation",
        semantics=args.semantics,
        # append 与服务端合约一致；auto 会选择尚未支持的 aligned。
        chunk_merge="append",
        mode="chunk",
        encoding="raw",
        endpoint=args.endpoint,
        zenoh_mode="peer",
    )
    smoke(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
