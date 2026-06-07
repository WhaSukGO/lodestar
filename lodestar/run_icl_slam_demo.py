"""Demo: visual odometry on a REAL benchmark (ICL-NUIM), graded by the Touchstone verifier.

  SOLVER (swappable, UNCHANGED)                VERIFIER (Touchstone spine, the constant)
  detect+match ORB from real RGB-D frames, --> run it, compare frame-to-frame motion to the
  back-project with depth, estimate motion     dataset's HIDDEN ground truth, score RPE, gate

This is the toy-to-real step: the "world" is the ICL-NUIM living-room sequence — photorealistic
ray-traced frames + metric depth + a perfect ground-truth trajectory that the SLAM community
benchmarks on. Offline, no API; downloads+caches ~700 MB on first run. Same ORB solver as the
rendered rungs. Run: python -m lodestar.run_icl_slam_demo"""
from __future__ import annotations

import tempfile

from ._spine import ExperimentRecord, build_implementer_harness
from .worlds.dataset_slam import HONEST, STATIC, IclNuimProvider, _world, ensure_icl, icl_task, is_available


def _run(author, provider):
    root = tempfile.mkdtemp(prefix="icl-slam-")
    harness = build_implementer_harness(root, icl_task(), author_fn=author,
                                        provider=provider, job_mode="local")
    return harness.run_experiment(ExperimentRecord(id="icl", hypothesis="icl vo"))


def main() -> None:
    if not is_available():
        print("Downloading ICL-NUIM (~700 MB, first run only)...")
        ensure_icl()
    poses, intr, rgb, depth = _world()
    print("=== TESTBED: ICL-NUIM living_room_traj0 — REAL RGB-D SLAM benchmark (hidden GT) ===")
    print(f"  {len(rgb)} frames of {rgb.shape[1]}x{rgb.shape[2]} grayscale + metric depth, "
          f"intrinsics {tuple(round(v,1) for v in intr)}")
    print(f"  oracle: SE(3)-aligned ATE <= {icl_task().threshold} m vs the HIDDEN trajectory\n")

    prov = IclNuimProvider()           # one decode shared across both runs
    for label, author in [("honest image VO (cv2 ORB detect+match)", HONEST),
                          ("static (assume the camera never moved)", STATIC)]:
        r = _run(author, prov)
        ate = r.verdict.measured_metrics.get("ate")
        print(f"  SOLVER: {label:40}  ATE={ate:<7}  -> {r.status.value}")

    print("\nSame ORB solver as the rendered rungs, now graded on a REAL benchmark sequence the "
          "SLAM\ncommunity uses — VERIFIED on a trajectory it never saw; 'never moved' REJECTED. "
          "It ran != correct.\n")


if __name__ == "__main__":
    main()
