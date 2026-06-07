"""Rung 7 end-to-end through the Touchstone verifier: VO on a REAL SCANNED apartment (Replica).

The same ORB front-end as the other rungs, graded on a real laser-scanned Replica interior
rendered with BlenderProc/Cycles. Honest VO is VERIFIED; "camera never moved" is REJECTED.

Auto-skips unless both blenderproc AND a cached Replica scene are present (so CI stays green and
the ~20 GB dataset is never required). A passing run renders the scene (slow) — local/opt-in."""
from __future__ import annotations

import pytest

from lodestar.worlds import replica_slam

pytestmark = pytest.mark.skipif(
    not (replica_slam.blenderproc_available() and replica_slam.is_available()),
    reason="needs blenderproc + a cached Replica scene (~/.cache/lodestar/replica)")


def _run(root, author, provider):
    from lodestar._spine import ExperimentRecord, build_implementer_harness
    h = build_implementer_harness(str(root), replica_slam.replica_task(), author_fn=author,
                                  provider=provider, job_mode="local")
    return h.run_experiment(ExperimentRecord(id="replica", hypothesis="replica slam"))


def test_replica_vo_verified_static_rejected(tmp_path):
    prov = replica_slam.ReplicaSlamProvider()           # one render shared across both runs
    good = _run(tmp_path / "good", replica_slam.HONEST, prov)
    assert good.status.value == "VERIFIED"
    assert good.verdict.measured_metrics["rpe"] <= 0.05     # motion recovered on a real scanned scene

    bad = _run(tmp_path / "bad", replica_slam.STATIC, prov)
    assert bad.status.value == "REJECTED"
    assert bad.verdict.measured_metrics["rpe"] > 0.05
