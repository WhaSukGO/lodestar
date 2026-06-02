"""Rung 0 end-to-end through the Touchstone verifier (offline, local mode, numpy-only).

Honest pose-graph optimization -> VERIFIED on the hidden GT trajectory; a dead-reckoning
solver that ignores loop closures -> REJECTED. Proves the verifier grades a real geometric
oracle (SE(2)-aligned ATE), not merely "did it emit a trajectory.\""""
from __future__ import annotations

from slamtest._spine import ExperimentRecord, build_implementer_harness
from slamtest.worlds.posegraph2d import (
    HONEST, ODOMETRY, PoseGraphProvider, ate, posegraph_task, solve_posegraph, _world,
)


def _run(root, author):
    h = build_implementer_harness(str(root), posegraph_task(), author_fn=author,
                                  provider=PoseGraphProvider(), job_mode="local")
    return h.run_experiment(ExperimentRecord(id="slam", hypothesis="posegraph"))


def test_honest_verified_deadreckoning_rejected(tmp_path):
    good = _run(tmp_path / "good", HONEST)
    assert good.status.value == "VERIFIED"
    assert good.verdict.measured_metrics["ate"] <= 0.12        # drift cancelled by loop closures

    bad = _run(tmp_path / "bad", ODOMETRY)
    assert bad.status.value == "REJECTED"
    assert bad.verdict.measured_metrics["ate"] > 0.12          # uncorrected drift


def test_optimizer_beats_odometry_offline():
    # the oracle is real: optimization must materially cut ATE vs the drifted odometry guess
    gt, nodes, edges = _world(0)
    assert ate(solve_posegraph(nodes.tolist(), edges), gt) < 0.5 * ate(nodes, gt)
