#!/usr/bin/env python3
"""导出 Cosmos3 Edge 画布拼接的参照（两个 256 视角）。"""

import argparse
from pathlib import Path
import sys

import numpy as np


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
    from lerobot.policies.cosmos3 import processor_cosmos3 as P

    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(3)
    step = P.Cosmos3VideoCanvasProcessorStep(
        camera_keys=["a", "b"], fps=20.0, chunk_size=32, view_size=(256, 256)
    )
    first = torch.rand([1, 3, 256, 256])
    second = torch.rand([1, 3, 256, 256])
    views = [step.prepare_view(view, step.view_size) for view in (first, second)]
    frames = (step.compose(views) * 255.0).clamp(0, 255).to(torch.uint8)
    canvas, content = step.fit_canvas(frames)
    pixels = canvas.permute(0, 2, 1, 3, 4).float() / 127.5 - 1.0

    first.numpy().astype("<f4").tofile(output / "first.bin")
    second.numpy().astype("<f4").tofile(output / "second.bin")
    pixels.numpy().astype("<f4").tofile(output / "canvas.bin")
    print("canvas", list(pixels.shape), "content", content)
    print("完成", flush=True)


if __name__ == "__main__":
    main()
