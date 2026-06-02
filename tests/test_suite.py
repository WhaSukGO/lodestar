"""Selectable environments + suite runner (offline, local mode).

The same honest solver, graded by the same fixed oracle, passes easy/default and is REJECTED
where the environment breaks its assumptions — and the verifier reports that honestly."""
from __future__ import annotations

from slamtest.scenarios import RUNGS
from slamtest.suite import run_suite


def test_posegraph_robustness_table():
    rows = run_suite("0", solver="honest")
    by = {r["scenario"]: r["status"] for r in rows}
    assert by["easy"] == "VERIFIED"
    assert by["default"] == "VERIFIED"
    assert by["no-loops"] == "REJECTED"        # no loop closures -> drift uncorrectable
    assert by["high-noise"] == "REJECTED"      # noise beyond what loops can fix


def test_registry_wires_all_rungs():
    for r in ("0", "1", "2"):
        cfg = RUNGS[r]
        assert "default" in cfg["scenarios"]
        assert callable(cfg["task"]) and cfg["honest"] is not cfg["degenerate"]
