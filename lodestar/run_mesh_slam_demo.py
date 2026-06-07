"""Demo: image VO on a REAL 3D world (offscreen-rendered mesh scene), graded by Touchstone.

  SOLVER (swappable, UNCHANGED from Rung 3)   VERIFIER (Touchstone spine, the constant)
  detect+match ORB from rendered pixels,  -->  run it, compare frame-to-frame motion to the
  back-project with depth, estimate motion     HIDDEN ground truth, score RPE, gate

Rung 4 renders an actual triangle-mesh room with boxes (pyrender/OSMesa, software GL) — real
occlusion and perspective, not Rung 3's camera-facing splats. The frames.npz contract is the
same, so the same ORB front-end is graded on a real 3D world. Offline, no API; needs pyrender
+ trimesh + OSMesa (see README). Run: python -m lodestar.run_mesh_slam_demo"""
from __future__ import annotations

import tempfile

from ._spine import ExperimentRecord, build_implementer_harness
from .worlds.mesh_slam import HONEST, STATIC, MeshSlamProvider, _world, mesh_slam_task


def _run(author):
    root = tempfile.mkdtemp(prefix="mesh-slam-")
    harness = build_implementer_harness(root, mesh_slam_task(), author_fn=author,
                                        provider=MeshSlamProvider(), job_mode="local")
    return harness.run_experiment(ExperimentRecord(id="mesh", hypothesis="mesh vo"))


def main() -> None:
    poses, intr, rgb, depth = _world(0)
    print("=== TESTBED: a real 3D mesh scene, offscreen-rendered (hidden ground truth) ===")
    print(f"  {len(rgb)} frames of {rgb.shape[1]}x{rgb.shape[2]} grayscale + metric depth, "
          f"rendered with pyrender/OSMesa")
    print(f"  textured room + interior boxes: real occlusion + perspective (not billboards)")
    print(f"  oracle: translational RPE <= {mesh_slam_task().threshold} m vs the HIDDEN trajectory\n")

    for label, author in [("honest image VO (cv2 ORB detect+match)", HONEST),
                          ("static (assume the camera never moved)", STATIC)]:
        r = _run(author)
        rpe = r.verdict.measured_metrics.get("rpe")
        print(f"  SOLVER: {label:40}  RPE={rpe:<7}  -> {r.status.value}")

    print("\nSame ORB solver as Rung 3, now graded on a genuine 3D render where near surfaces "
          "occlude\nfar ones and textures foreshorten with view angle. It still recovers the "
          "motion on GT it\nnever saw; 'camera never moved' is still REJECTED. It ran != correct.\n")


if __name__ == "__main__":
    main()
