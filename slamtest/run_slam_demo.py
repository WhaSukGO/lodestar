"""Demo: full visual SLAM (VO + loop closure + SE(3) pose-graph) graded by Touchstone.

  SOLVER (swappable)                          VERIFIER (ver2 spine, the constant)
  odometry + loop closure + optimize  -->     run it, align to the HIDDEN trajectory,
  (or just odometry = drifts)                 score ATE, gate

Offline, no API spend, no renderer. Run:  python -m slamtest.run_slam_demo"""
from __future__ import annotations

import tempfile

from ._spine import ExperimentRecord, build_implementer_harness
from .worlds.visual_slam import (
    HONEST, VO_ONLY, VisualSlamProvider, _world, slam_task,
)
from .worlds.visual_slam import _graph


def _run(author):
    root = tempfile.mkdtemp(prefix="slam-full-")
    harness = build_implementer_harness(root, slam_task(), author_fn=author,
                                        provider=VisualSlamProvider(), job_mode="local")
    return harness.run_experiment(ExperimentRecord(id="slam", hypothesis="visual slam"))


def main() -> None:
    poses, intr, frames = _world(0)
    _, edges = _graph(intr, frames)
    n_lc = sum(1 for e in edges if e[1] - e[0] > 1)
    print("=== TESTBED: looping RGBD SLAM world (no renderer, hidden ground truth) ===")
    print(f"  {len(frames)} frames (two orbits), {len(edges) - n_lc} odometry edges, "
          f"{n_lc} loop closures")
    print(f"  oracle: SE(3)-aligned ATE <= {slam_task().threshold} m vs the HIDDEN trajectory\n")

    for label, author in [("full SLAM (odometry + loop closure + optimization)", HONEST),
                          ("VO only (odometry, no loop closure)", VO_ONLY)]:
        r = _run(author)
        ate = r.verdict.measured_metrics.get("ate")
        print(f"  SOLVER: {label:50}  ATE={ate:<7}  -> {r.status.value}")

    print("\nBoth solvers RAN. Loop closure is what makes SLAM more than odometry: only the "
          "solver\nthat closes the loop cancels drift and passes — graded on GT it never "
          "saw. It ran != correct.\n")


if __name__ == "__main__":
    main()
