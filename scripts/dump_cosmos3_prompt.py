#!/usr/bin/env python3
"""导出 Cosmos3 Edge 的 JSON 提示与模板文本。"""

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lerobot-src", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not sys.flags.isolated:
        parser.error("必须使用 python -I")
    lerobot_src, output = (path.expanduser().resolve() for path in (args.lerobot_src, args.output))
    sys.path.insert(0, str(lerobot_src))
    from lerobot.policies.cosmos3 import processor_cosmos3 as P

    output.mkdir(parents=True, exist_ok=True)
    step = P.Cosmos3PromptProcessorStep(
        view_description="The left half shows the third-person view; the right half shows the wrist-mounted camera.",
        fps=20.0,
        chunk_size=32,
        use_state=True,
    )
    caption = step.caption('pick up the "cube".', 256, 256, 0, 20.0)
    text = P.START_OF_GENERATION
    (output / "caption.txt").write_text(caption)
    (output / "constants.txt").write_text(f"{P.CONCAT_VIEW_TEMPLATE}\n{text}")
    print("caption", caption)
    print("完成", flush=True)


if __name__ == "__main__":
    main()
