"""Demo: a SLAM algorithm (2D pose-graph optimization) graded by the Touchstone verifier.

  SOLVER (swappable)                       VERIFIER (ver2 spine, the constant)
  optimize the pose graph  ----------->    run it, align to the HIDDEN ground-truth
  (or just dead-reckon)                    trajectory, score ATE, pass/fail the oracle

Offline, no API spend, no renderer. Run:  python -m slamtest.run_posegraph_demo"""
from __future__ import annotations

import tempfile

from ._spine import ExperimentRecord, build_implementer_harness
from .worlds.posegraph2d import (
    HONEST, ODOMETRY, PoseGraphProvider, _world, posegraph_task,
)


def _run(author):
    root = tempfile.mkdtemp(prefix="slam-pg2d-")
    harness = build_implementer_harness(root, posegraph_task(), author_fn=author,
                                        provider=PoseGraphProvider(), job_mode="local")
    return harness.run_experiment(ExperimentRecord(id="slam", hypothesis="pose graph"))


def main() -> None:
    gt, nodes, edges = _world(0)
    n_odo = len(gt) - 1
    print("=== TESTBED: 2D pose-graph world (no renderer, hidden ground truth) ===")
    print(f"  {len(gt)} poses, {n_odo} odometry edges, {len(edges) - n_odo} loop closures")
    print(f"  oracle: SE(2)-aligned ATE <= {posegraph_task().threshold} m on the HIDDEN trajectory\n")

    for label, author in [("honest pose-graph optimizer", HONEST),
                          ("dead-reckoning (odometry only)", ODOMETRY)]:
        r = _run(author)
        ate = r.verdict.measured_metrics.get("ate")
        print(f"  SOLVER: {label:32}  ATE={ate:<7}  -> {r.status.value}")

    print("\nBoth solvers RAN and produced a trajectory. Only the one that actually cancels "
          "drift\nwith the loop closures passes the verifier — graded on GT it never saw. "
          "It ran != correct.\n")


if __name__ == "__main__":
    main()
