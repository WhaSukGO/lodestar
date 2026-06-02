"""Rung 1 end-to-end through the Touchstone verifier (offline, local mode, numpy-only).

Honest RGBD visual odometry -> VERIFIED on hidden GT (low RPE); a "camera never moved"
solver -> REJECTED. Proves the verifier grades a real motion oracle (RPE), not "did it
emit a trajectory.\""""
from __future__ import annotations

import numpy as np

from lodestar._spine import ExperimentRecord, build_implementer_harness
from lodestar.worlds.visual_odometry import (
    HONEST, STATIC, VisualOdometryProvider, _world, rpe, run_vo, vo_task,
)


def _run(root, author):
    h = build_implementer_harness(str(root), vo_task(), author_fn=author,
                                  provider=VisualOdometryProvider(), job_mode="local")
    return h.run_experiment(ExperimentRecord(id="vo", hypothesis="visual odometry"))


def test_honest_verified_static_rejected(tmp_path):
    good = _run(tmp_path / "good", HONEST)
    assert good.status.value == "VERIFIED"
    assert good.verdict.measured_metrics["rpe"] <= 0.15

    bad = _run(tmp_path / "bad", STATIC)
    assert bad.status.value == "REJECTED"
    assert bad.verdict.measured_metrics["rpe"] > 0.15


def test_vo_beats_static_offline():
    poses, intr, frames = _world(0)
    static = [np.eye(4) for _ in frames]
    assert rpe(run_vo(intr, frames), poses) < 0.25 * rpe(static, poses)
