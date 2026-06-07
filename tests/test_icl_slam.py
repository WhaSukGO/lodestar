"""Rung 6 end-to-end through the Touchstone verifier: VO on REAL benchmark data (ICL-NUIM).

The same ORB front-end as the rendered rungs, graded on the ICL-NUIM living-room sequence — a
real, widely-benchmarked RGB-D SLAM dataset with a perfect hidden ground-truth trajectory.
Honest VO is VERIFIED (low RPE); "camera never moved" is REJECTED. Proves the verifier and
solver hold up on real data, not just synthetic worlds.

Auto-skips if the dataset isn't cached locally (it is NOT auto-downloaded in CI — ~700 MB)."""
from __future__ import annotations

import numpy as np
import pytest

from lodestar.worlds import dataset_slam


def _world_or_skip():
    if not dataset_slam.is_available():
        pytest.skip("ICL-NUIM not cached (run lodestar.run_icl_slam_demo once to download ~700 MB)")
    return dataset_slam._world()


def _run(root, author, provider):
    from lodestar._spine import ExperimentRecord, build_implementer_harness
    h = build_implementer_harness(str(root), dataset_slam.icl_task(), author_fn=author,
                                  provider=provider, job_mode="local")
    return h.run_experiment(ExperimentRecord(id="icl", hypothesis="icl slam"))


def test_icl_vo_verified_static_rejected(tmp_path):
    _world_or_skip()
    prov = dataset_slam.IclNuimProvider()
    good = _run(tmp_path / "good", dataset_slam.HONEST, prov)
    assert good.status.value == "VERIFIED"
    assert good.verdict.measured_metrics["ate"] <= 0.05        # real motion recovered on real data

    bad = _run(tmp_path / "bad", dataset_slam.STATIC, prov)
    assert bad.status.value == "REJECTED"
    assert bad.verdict.measured_metrics["ate"] > 0.05


def test_icl_vo_beats_static_offline():
    poses, intr, rgb, depth = _world_or_skip()
    static = [np.eye(4) for _ in poses]
    assert dataset_slam.ate(dataset_slam.run_image_vo(rgb, depth.astype(np.float32), intr),
                            poses) < 0.3 * dataset_slam.ate(static, poses)
