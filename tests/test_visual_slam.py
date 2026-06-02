"""Rung 2 end-to-end through the Touchstone verifier (offline, local mode, numpy-only).

Full visual SLAM (VO + loop closure + SE(3) pose-graph) -> VERIFIED on hidden GT (low ATE);
the same VO front-end WITHOUT loop closure -> REJECTED (drift). Proves the verifier measures
global trajectory accuracy, and that loop closure is what makes SLAM more than odometry."""
from __future__ import annotations

from lodestar._spine import ExperimentRecord, build_implementer_harness
from lodestar.worlds.visual_slam import (
    HONEST, VO_ONLY, VisualSlamProvider, _world, ate, run_slam, run_vo_only, slam_task,
)


def _run(root, author):
    h = build_implementer_harness(str(root), slam_task(), author_fn=author,
                                  provider=VisualSlamProvider(), job_mode="local")
    return h.run_experiment(ExperimentRecord(id="slam", hypothesis="visual slam"))


def test_full_slam_verified_vo_only_rejected(tmp_path):
    good = _run(tmp_path / "good", HONEST)
    assert good.status.value == "VERIFIED"
    assert good.verdict.measured_metrics["ate"] <= 0.10        # drift removed by loop closures

    bad = _run(tmp_path / "bad", VO_ONLY)
    assert bad.status.value == "REJECTED"
    assert bad.verdict.measured_metrics["ate"] > 0.10          # uncorrected VO drift


def test_loop_closure_helps_offline():
    # on a looping path, full SLAM must beat VO-only by a clear margin
    poses, intr, frames = _world(0)
    assert ate(run_slam(intr, frames), poses) < 0.5 * ate(run_vo_only(intr, frames), poses)


def test_thin_overlap_degrades_not_crashes():
    # extreme world: consecutive frames may share <3 tracks. The constant-velocity fallback
    # must keep the front-end running (no empty-Procrustes crash); the result is bad, not an error.
    poses, intr, frames = _world(loops=1.0, r=12.0, F=120)
    est = run_slam(intr, frames)
    assert len(est) == len(poses)
