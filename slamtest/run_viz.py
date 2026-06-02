"""Generate trajectory preview PNGs for all rungs into docs/img/.

Offline, no API. Run:  python -m slamtest.run_viz"""
from __future__ import annotations

import os

from .viz import viz_posegraph, viz_slam, viz_vo


def main() -> None:
    outdir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs", "img")
    os.makedirs(outdir, exist_ok=True)
    for name, fn in [("rung0_posegraph", viz_posegraph),
                     ("rung1_vo", viz_vo),
                     ("rung2_slam", viz_slam)]:
        path = os.path.join(outdir, name + ".png")
        info = fn(path)
        print(f"  wrote docs/img/{name}.png   ({info})")


if __name__ == "__main__":
    main()
