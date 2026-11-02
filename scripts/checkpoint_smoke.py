#!/usr/bin/env python3
"""加载已导出的真实 SmolVLA 产物，并运行官方客户端冒烟；只清理自建服务。"""

import argparse
from pathlib import Path
import socket
import subprocess
import sys
import threading


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--lerobot-src", type=Path, required=True)
    args = parser.parse_args()
    if not sys.flags.isolated:
        parser.error("请使用 python -I")
    import tomllib

    root = Path(__file__).resolve().parent.parent
    config = tomllib.loads(args.config.read_text())
    config["model"]["repo_or_path"] = str(args.checkpoint.resolve())
    config["model"].pop("vlm_path", None)
    # 保留原 TOML 中的合约；只替换资源绑定。
    source = args.config.read_text().splitlines()
    output = []
    in_model = False
    for line in source:
        if line.startswith("["):
            in_model = line.strip() == "[model]"
        if in_model and line.strip().startswith("vlm_path"):
            continue
        if in_model and line.strip().startswith("repo_or_path"):
            import json
            line = "repo_or_path = " + json.dumps(config["model"]["repo_or_path"])
        output.append(line)
    profile = root / "outputs/pipeline-artifact-smoke.toml"
    profile.write_text("\n".join(output) + "\n")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    endpoint = f"tcp/127.0.0.1:{port}"
    ready = threading.Event()
    log_path = root / "outputs/pipeline-artifact-smoke.log"
    with log_path.open("w") as log:
        process = subprocess.Popen([str(root / "target/release/pneuma"), "serve", f"--config={profile}",
            f"--listen={endpoint}"], cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
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
            if not ready.wait(120) or process.poll() is not None:
                raise RuntimeError(f"服务启动失败或超时：{log_path}")
            subprocess.run([sys.executable, "-I", str(root / "scripts/smolvla_smoke.py"),
                f"--lerobot-src={args.lerobot_src.resolve()}", f"--endpoint={endpoint}"], cwd=root, check=True)
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
            reader.join(timeout=5)


if __name__ == "__main__":
    main()
