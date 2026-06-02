"""Demo: RGBD visual odometry graded by the Touchstone verifier.

  SOLVER (swappable)                        VERIFIER (ver2 spine, the constant)
  recover camera motion from feature  -->   run it, compare each frame-to-frame motion to
  tracks (or assume it never moved)         the HIDDEN ground truth, score RPE, gate

Offline, no API spend, no renderer. Run:  python -m lodestar.run_vo_demo"""
from __future__ import annotations

import tempfile

from ._spine import ExperimentRecord, build_implementer_harness
from .worlds.visual_odometry import (
    HONEST, STATIC, VisualOdometryProvider, _world, vo_task,
)


def _run(author):
    root = tempfile.mkdtemp(prefix="slam-vo-")
    harness = build_implementer_harness(root, vo_task(), author_fn=author,
                                        provider=VisualOdometryProvider(), job_mode="local")
    return harness.run_experiment(ExperimentRecord(id="vo", hypothesis="visual odometry"))


def main() -> None:
    poses, intr, frames = _world(0)
    avg_obs = sum(len(f) for f in frames) / len(frames)
    print("=== TESTBED: RGBD visual-odometry world (no renderer, hidden ground truth) ===")
    print(f"  {len(frames)} frames, ~{avg_obs:.0f} feature obs/frame, intrinsics fx={intr[0]:.0f}")
    print(f"  oracle: translational RPE <= {vo_task().threshold} m vs the HIDDEN trajectory\n")

    for label, author in [("honest RGBD VO (Procrustes per frame pair)", HONEST),
                          ("static (assume the camera never moved)", STATIC)]:
        r = _run(author)
        rpe = r.verdict.measured_metrics.get("rpe")
        print(f"  SOLVER: {label:42}  RPE={rpe:<7}  -> {r.status.value}")

    print("\nBoth solvers RAN and produced a full trajectory. Only the one that actually "
          "estimates\nmotion from the feature tracks passes — graded on GT it never saw. "
          "It ran != correct.\n")


if __name__ == "__main__":
    main()
