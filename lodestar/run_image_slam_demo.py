"""Demo: image-based visual odometry graded by the Touchstone verifier.

  SOLVER (swappable)                          VERIFIER (Touchstone spine, the constant)
  detect+match ORB features from rendered --> run it, compare frame-to-frame motion to the
  pixels, back-project, estimate motion       HIDDEN ground truth, score RPE, gate

Unlike Rungs 1-2 the solver gets actual rendered frames, not feature tracks — it must do its
own perception. Offline, no API, CPU only (needs cv2). Run: python -m lodestar.run_image_slam_demo"""
from __future__ import annotations

import tempfile

from ._spine import ExperimentRecord, build_implementer_harness
from .worlds.image_slam import HONEST, STATIC, ImageSlamProvider, _world, image_slam_task


def _run(author):
    root = tempfile.mkdtemp(prefix="img-slam-")
    harness = build_implementer_harness(root, image_slam_task(), author_fn=author,
                                        provider=ImageSlamProvider(), job_mode="local")
    return harness.run_experiment(ExperimentRecord(id="img", hypothesis="image vo"))


def main() -> None:
    poses, intr, rgb, depth = _world(0)
    print("=== TESTBED: rendered RGBD frames (no feature tracks given, hidden ground truth) ===")
    print(f"  {len(rgb)} frames of {rgb.shape[1]}x{rgb.shape[2]} grayscale + depth")
    print(f"  oracle: translational RPE <= {image_slam_task().threshold} m vs the HIDDEN trajectory\n")

    for label, author in [("honest image VO (cv2 ORB detect+match)", HONEST),
                          ("static (assume the camera never moved)", STATIC)]:
        r = _run(author)
        rpe = r.verdict.measured_metrics.get("rpe")
        print(f"  SOLVER: {label:40}  RPE={rpe:<7}  -> {r.status.value}")

    print("\nThe honest solver detects and matches features from the pixels itself (real data "
          "association,\nno landmark IDs) and recovers the motion — graded on GT it never saw. "
          "It ran != correct.\n")


if __name__ == "__main__":
    main()
