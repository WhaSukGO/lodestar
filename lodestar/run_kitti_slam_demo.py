"""Demo: automotive VO on REAL driving data (KITTI odometry), graded by the Touchstone verifier.

  SOLVER (swappable, UNCHANGED)                VERIFIER (Touchstone spine, the constant)
  detect+match ORB on car-camera frames,   --> run it, compare frame-to-frame motion to the
  back-project with stereo depth, estimate     car's HIDDEN GPS/IMU trajectory, score RPE, gate

The world is a real car driving real streets — the KITTI odometry benchmark, the canonical
automotive VO/SLAM dataset. Depth is computed once from the stereo pair (cv2 SGBM), then the same
ORB solver runs. Offline, no API; needs the KITTI odometry data cached under ~/.cache/lodestar/
kitti. Run: python -m lodestar.run_kitti_slam_demo"""
from __future__ import annotations

import tempfile

from ._spine import ExperimentRecord, build_implementer_harness
from .worlds.kitti_slam import HONEST, STATIC, KittiProvider, _world, is_available, kitti_task


def _run(author, provider):
    root = tempfile.mkdtemp(prefix="kitti-slam-")
    harness = build_implementer_harness(root, kitti_task(), author_fn=author,
                                        provider=provider, job_mode="local")
    return harness.run_experiment(ExperimentRecord(id="kitti", hypothesis="kitti vo"))


def main() -> None:
    if not is_available():
        print("KITTI odometry not cached. Download gray+calib+poses to ~/.cache/lodestar/kitti.")
        return
    poses, intr, rgb, depth = _world()
    print("=== TESTBED: KITTI odometry — a REAL car driving real streets (hidden GT) ===")
    print(f"  {len(rgb)} frames of {rgb.shape[1]}x{rgb.shape[2]} grayscale + stereo depth, "
          f"intrinsics {tuple(round(float(v),1) for v in intr)}")
    print(f"  oracle: translational RPE <= {kitti_task().threshold} m vs the HIDDEN trajectory\n")

    prov = KittiProvider()             # one stereo-depth pass shared across both runs
    for label, author in [("honest image VO (cv2 ORB detect+match)", HONEST),
                          ("static (assume the car never moved)", STATIC)]:
        r = _run(author, prov)
        rpe = r.verdict.measured_metrics.get("rpe")
        print(f"  SOLVER: {label:40}  RPE={rpe:<7}  -> {r.status.value}")

    print("\nSame ORB solver as the indoor rungs, now graded on a REAL automotive sequence — "
          "VERIFIED on a\ncar trajectory it never saw; 'the car never moved' is REJECTED. "
          "It ran != correct.\n")


if __name__ == "__main__":
    main()
