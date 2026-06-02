"""Rung 3 end-to-end through the Touchstone verifier (offline, local mode, numpy + cv2).

Honest image-based VO — features detected and matched from rendered pixels (no tracks given)
— is VERIFIED on hidden GT (low RPE); a "camera never moved" solver is REJECTED. Proves the
verifier grades a solver that does its own perception, not one handed pre-extracted tracks."""
from __future__ import annotations

import numpy as np

from lodestar._spine import ExperimentRecord, build_implementer_harness
from lodestar.worlds.image_slam import (
    HONEST, STATIC, ImageSlamProvider, _world, image_slam_task, rpe, run_image_vo,
)


def _run(root, author):
    h = build_implementer_harness(str(root), image_slam_task(), author_fn=author,
                                  provider=ImageSlamProvider(), job_mode="local")
    return h.run_experiment(ExperimentRecord(id="img", hypothesis="image slam"))


def test_image_vo_verified_static_rejected(tmp_path):
    good = _run(tmp_path / "good", HONEST)
    assert good.status.value == "VERIFIED"
    assert good.verdict.measured_metrics["rpe"] <= 0.05      # real feature-based motion

    bad = _run(tmp_path / "bad", STATIC)
    assert bad.status.value == "REJECTED"
    assert bad.verdict.measured_metrics["rpe"] > 0.05


def test_image_vo_beats_static_offline():
    poses, intr, rgb, depth = _world(0)
    static = [np.eye(4) for _ in poses]
    assert rpe(run_image_vo(rgb, depth.astype(np.float32), intr), poses) < 0.25 * rpe(static, poses)
