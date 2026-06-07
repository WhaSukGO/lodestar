"""Demo: image VO on a REAL SCANNED apartment (Replica), graded by the Touchstone verifier.

  SOLVER (swappable, UNCHANGED)                VERIFIER (Touchstone spine, the constant)
  detect+match ORB from rendered pixels,   --> run it, compare frame-to-frame motion to the
  back-project with depth, estimate motion     HIDDEN ground truth, score RPE, gate

The world is a real laser-scanned apartment from the Replica dataset (FAIR), rendered with
BlenderProc/Cycles (CPU/CUDA, no EGL). Same frames.npz contract and same ORB solver as the other
rendered rungs. Offline, no API; needs blenderproc + the Replica scene cached under
~/.cache/lodestar/replica. Run: python -m lodestar.run_replica_slam_demo"""
from __future__ import annotations

import tempfile

from ._spine import ExperimentRecord, build_implementer_harness
from .worlds.replica_slam import (
    HONEST, STATIC, ReplicaSlamProvider, blenderproc_available, is_available, replica_task,
)


def _run(author, provider):
    root = tempfile.mkdtemp(prefix="replica-slam-")
    harness = build_implementer_harness(root, replica_task(), author_fn=author,
                                        provider=provider, job_mode="local")
    return harness.run_experiment(ExperimentRecord(id="replica", hypothesis="replica vo"))


def main() -> None:
    if not blenderproc_available():
        print("BlenderProc not installed. Run: pip install blenderproc")
        return
    if not is_available():
        print("Replica not cached. Download it to ~/.cache/lodestar/replica (see docs).")
        return
    print("=== TESTBED: a REAL scanned apartment (Replica), path-traced (hidden GT) ===")
    print("  rendering a real laser-scanned interior with Cycles (CPU/CUDA, no EGL)...")
    print(f"  oracle: translational RPE <= {replica_task().threshold} m vs the HIDDEN trajectory\n")

    prov = ReplicaSlamProvider()        # renders once, shared across both runs
    for label, author in [("honest image VO (cv2 ORB detect+match)", HONEST),
                          ("static (assume the camera never moved)", STATIC)]:
        r = _run(author, prov)
        rpe = r.verdict.measured_metrics.get("rpe")
        print(f"  SOLVER: {label:40}  RPE={rpe:<7}  -> {r.status.value}")

    print("\nSame ORB solver, now graded on a REAL scanned 3D environment — VERIFIED on a "
          "trajectory it\nnever saw; 'camera never moved' is REJECTED. It ran != correct.\n")


if __name__ == "__main__":
    main()
