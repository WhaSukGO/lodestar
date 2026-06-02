"""Suite runner — grade one solver across a rung's selectable environments.

Same verifier, same oracle threshold; the world varies by scenario. Produces a robustness
table: which environments the solver survives, which it doesn't. This is how a real benchmark
works (many scenarios), and how overfitting-to-one-world would be caught."""
from __future__ import annotations

import tempfile

from ._spine import ExperimentRecord, build_implementer_harness
from .scenarios import RUNGS


def run_suite(rung: str, solver: str = "honest", scenarios: dict | None = None) -> list[dict]:
    """Grade `solver` ("honest" | "degenerate") across each scenario of `rung`. Returns one
    row per scenario: {scenario, metric, value, threshold, op, status}."""
    cfg = RUNGS[rung]
    author = cfg[solver]
    task = cfg["task"]()
    rows = []
    for name, world_kwargs in (scenarios or cfg["scenarios"]).items():
        provider = cfg["provider"](**world_kwargs)
        root = tempfile.mkdtemp(prefix=f"suite-r{rung}-{name}-")
        harness = build_implementer_harness(root, task, author_fn=author, provider=provider,
                                            job_mode="local")
        res = harness.run_experiment(ExperimentRecord(id=f"{rung}-{name}", hypothesis=name))
        rows.append(dict(scenario=name, metric=task.metric,
                         value=res.verdict.measured_metrics.get(task.metric),
                         threshold=task.threshold, op=task.op, status=res.status.value))
    return rows


def format_table(rung: str, solver: str, rows: list[dict]) -> str:
    cfg = RUNGS[rung]; m = rows[0]["metric"] if rows else "?"
    op = rows[0]["op"] if rows else "<="; th = rows[0]["threshold"] if rows else "?"
    out = [f"=== Rung {rung} — {cfg['label']} | {solver} solver | oracle {m} {op} {th} ===",
           f"  {'scenario':14}{m:>8}   verdict"]
    for r in rows:
        v = r["value"]
        out.append(f"  {r['scenario']:14}{(f'{v:.3f}' if v is not None else 'n/a'):>8}   "
                   f"{r['status']}")
    return "\n".join(out)
