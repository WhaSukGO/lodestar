"""Generate trajectory preview PNGs for all rungs into docs/img/.

Offline, no API. Run:  python -m lodestar.run_viz"""
from __future__ import annotations

import os

from .viz import (
    viz_blender, viz_icl, viz_mesh, viz_posegraph, viz_replica, viz_slam, viz_suite, viz_vo,
)


def main() -> None:
    outdir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs", "img")
    os.makedirs(outdir, exist_ok=True)
    for name, fn in [("rung0_posegraph", viz_posegraph),
                     ("rung1_vo", viz_vo),
                     ("rung2_slam", viz_slam)]:
        path = os.path.join(outdir, name + ".png")
        print(f"  wrote docs/img/{name}.png   ({fn(path)})")
    # Real-world-environment rungs — each needs optional deps (GL stack / blenderproc / datasets),
    # so they're guarded and skip cleanly when those aren't present.
    for name, fn in [("rung4_mesh", viz_mesh),           # pyrender/OSMesa
                     ("rung5_blender", viz_blender),      # BlenderProc/Cycles
                     ("rung6_icl", viz_icl),              # ICL-NUIM dataset
                     ("rung7_replica", viz_replica)]:     # Replica scanned scene
        path = os.path.join(outdir, name + ".png")
        try:
            print(f"  wrote docs/img/{name}.png   ({fn(path)})")
        except Exception as e:
            print(f"  skipped {name}.png ({type(e).__name__}: {e})")
    for rung in ("0", "1", "2"):                         # robustness grids (viz x suite)
        path = os.path.join(outdir, f"rung{rung}_suite.png")
        print(f"  wrote docs/img/rung{rung}_suite.png   ({viz_suite(rung, path)})")


if __name__ == "__main__":
    main()
