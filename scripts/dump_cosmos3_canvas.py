#!/usr/bin/env python3
"""导出 Cosmos3 Edge 画布缩放的参照（torch 双三次抗锯齿）。"""

import argparse
from pathlib import Path
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not sys.flags.isolated:
        parser.error("必须使用 python -I")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    import torch
    import torch.nn.functional as F

    torch.manual_seed(7)
    plane = torch.rand([1, 1, 256, 512], dtype=torch.float32) * 255.0
    out = F.interpolate(plane, (160, 320), mode="bicubic", align_corners=False, antialias=True)
    plane.numpy().astype("<f4").tofile(output / "plane.bin")
    out.numpy().astype("<f4").tofile(output / "resized.bin")
    print("plane", list(plane.shape), "resized", list(out.shape))
    print("完成", flush=True)


if __name__ == "__main__":
    main()
