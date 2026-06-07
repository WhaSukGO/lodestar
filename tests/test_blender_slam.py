"""Rung 5 end-to-end through the Touchstone verifier: VO on a PHOTOREALISTIC path-traced world.

The same ORB front-end as the other rungs, graded on a BlenderProc/Cycles render (real global
illumination + soft shadows). Honest VO is VERIFIED (low RPE); "camera never moved" is REJECTED.

Auto-skips if BlenderProc isn't installed. NOTE: a passing run path-traces the world (slow —
tens of seconds), so this is a local/opt-in test, skipped in CI."""
from __future__ import annotations

import numpy as np
import pytest

from lodestar.worlds import blender_slam

pytestmark = pytest.mark.skipif(not blender_slam.blenderproc_available(),
                                reason="blenderproc not installed (pip install blenderproc)")


def _run(root, author, provider):
    from lodestar._spine import ExperimentRecord, build_implementer_harness
    h = build_implementer_harness(str(root), blender_slam.blender_task(), author_fn=author,
                                  provider=provider, job_mode="local")
    return h.run_experiment(ExperimentRecord(id="blender", hypothesis="blender slam"))


def test_blender_vo_verified_static_rejected(tmp_path):
    prov = blender_slam.BlenderSlamProvider()           # one render shared across both runs
    good = _run(tmp_path / "good", blender_slam.HONEST, prov)
    assert good.status.value == "VERIFIED"
    assert good.verdict.measured_metrics["rpe"] <= 0.05      # motion recovered on a path-traced render

    bad = _run(tmp_path / "bad", blender_slam.STATIC, prov)
    assert bad.status.value == "REJECTED"
    assert bad.verdict.measured_metrics["rpe"] > 0.05
