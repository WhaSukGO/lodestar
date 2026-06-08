"""Rung 8 end-to-end through the Touchstone verifier: automotive VO on REAL KITTI driving data.

The same ORB front-end as the indoor rungs, graded on the KITTI odometry benchmark — a real car
with a stereo camera through real streets, with a hidden GPS/IMU ground-truth trajectory. Depth
is computed from the stereo pair (SGBM). Honest VO is VERIFIED; "the car never moved" is REJECTED.

Auto-skips unless the KITTI odometry data is cached locally (it is NOT auto-downloaded — ~22 GB)."""
from __future__ import annotations

import numpy as np
import pytest

from lodestar.worlds import kitti_slam


def _world_or_skip():
    if not kitti_slam.is_available():
        pytest.skip("KITTI odometry not cached (~/.cache/lodestar/kitti) — see run_kitti_slam_demo")
    return kitti_slam._world()


def _run(root, author, provider):
    from lodestar._spine import ExperimentRecord, build_implementer_harness
    h = build_implementer_harness(str(root), kitti_slam.kitti_task(), author_fn=author,
                                  provider=provider, job_mode="local")
    return h.run_experiment(ExperimentRecord(id="kitti", hypothesis="kitti slam"))


def test_kitti_vo_verified_static_rejected(tmp_path):
    _world_or_skip()
    prov = kitti_slam.KittiProvider()
    th = kitti_slam.kitti_task().threshold
    good = _run(tmp_path / "good", kitti_slam.HONEST, prov)
    assert good.status.value == "VERIFIED"
    assert good.verdict.measured_metrics["rpe"] <= th          # real car motion recovered

    bad = _run(tmp_path / "bad", kitti_slam.STATIC, prov)
    assert bad.status.value == "REJECTED"
    assert bad.verdict.measured_metrics["rpe"] > th


def test_kitti_vo_beats_static_offline():
    poses, intr, rgb, depth = _world_or_skip()
    static = [np.eye(4) for _ in poses]
    assert kitti_slam.rpe(kitti_slam.run_image_vo(rgb, depth.astype(np.float32), intr),
                          poses) < 0.3 * kitti_slam.rpe(static, poses)
