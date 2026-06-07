"""Demo: image VO on a PHOTOREALISTIC path-traced world (BlenderProc/Cycles), graded by Touchstone.

  SOLVER (swappable, UNCHANGED)                VERIFIER (Touchstone spine, the constant)
  detect+match ORB from path-traced pixels,--> run it, compare frame-to-frame motion to the
  back-project with depth, estimate motion     HIDDEN ground truth, score RPE, gate

Rung 5 renders a textured room with Blender's Cycles path tracer (global illumination, soft
shadows) via BlenderProc — no OpenGL/EGL needed (CPU or CUDA-compute). Same frames.npz contract
and same ORB solver as the rasterized rung. Offline, no API; needs `pip install blenderproc`
(ships its own Blender). Run: python -m lodestar.run_blender_slam_demo"""
from __future__ import annotations

import tempfile

from ._spine import ExperimentRecord, build_implementer_harness
from .worlds.blender_slam import (
    HONEST, STATIC, BlenderSlamProvider, blender_task, blenderproc_available,
)


def _run(author, provider):
    root = tempfile.mkdtemp(prefix="blender-slam-")
    harness = build_implementer_harness(root, blender_task(), author_fn=author,
                                        provider=provider, job_mode="local")
    return harness.run_experiment(ExperimentRecord(id="blender", hypothesis="blender vo"))


def main() -> None:
    if not blenderproc_available():
        print("BlenderProc not installed. Run: pip install blenderproc")
        return
    print("=== TESTBED: a photorealistic path-traced room (BlenderProc/Cycles, hidden GT) ===")
    print("  rendering with Cycles (global illumination + soft shadows; CPU/CUDA, no EGL)...")
    print(f"  oracle: translational RPE <= {blender_task().threshold} m vs the HIDDEN trajectory\n")

    prov = BlenderSlamProvider()        # renders once, shared across both runs
    for label, author in [("honest image VO (cv2 ORB detect+match)", HONEST),
                          ("static (assume the camera never moved)", STATIC)]:
        r = _run(author, prov)
        rpe = r.verdict.measured_metrics.get("rpe")
        print(f"  SOLVER: {label:40}  RPE={rpe:<7}  -> {r.status.value}")

    print("\nSame ORB solver, now graded on a photorealistic path-traced render (real GI + "
          "shadows).\nVERIFIED on a trajectory it never saw; 'camera never moved' is REJECTED. "
          "It ran != correct.\n")


if __name__ == "__main__":
    main()
