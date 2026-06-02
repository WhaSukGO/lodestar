"""LIVE: a committee of Claude agents AUTHORS a SLAM algorithm; Touchstone grades it.

  SOLVER (live, swappable)                       VERIFIER (ver2 spine, the constant)
  PLANNER -> CODER -> REVIEWER -> CODER     -->   run the authored code, align to the HIDDEN
  (real Claude sessions write the code)          ground-truth trajectory, score, gate

This is the highlight of the two-layer model: the verifier and domain are unchanged from the
offline runs — only the SOLVER is swapped from canned reference code to a real multi-agent
team. The committee authors BLIND (it reasons and writes code; it does not execute it in a
sandbox), so a closed-form rung (RGBD VO) is the best first target.

Spends real API tokens (bounded: 1 planner + up to 2x(coder+reviewer) = <=5 short calls).
Loads the API key from ver2's .env. Run:

  python -m slamtest.run_committee_live            # Rung 1 (RGBD VO), default
  python -m slamtest.run_committee_live --rung 0   # 2D pose-graph
  python -m slamtest.run_committee_live --rung 2   # full visual SLAM
  python -m slamtest.run_committee_live --model claude-opus-4-8
"""
from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

from ._spine import (
    DEFAULT_MODEL, ExperimentRecord, build_implementer_harness, code_committee_author,
    run_agent, ver2_path,
)

_RUNGS = {
    "0": ("slamtest.worlds.posegraph2d", "PoseGraphProvider", "posegraph_task", "2D pose-graph"),
    "1": ("slamtest.worlds.visual_odometry", "VisualOdometryProvider", "vo_task", "RGBD visual odometry"),
    "2": ("slamtest.worlds.visual_slam", "VisualSlamProvider", "slam_task", "full visual SLAM"),
}


def _load(rung: str):
    import importlib
    mod_name, prov, task_fn, label = _RUNGS[rung]
    mod = importlib.import_module(mod_name)
    return getattr(mod, prov)(), getattr(mod, task_fn)(), label


def main(rung: str = "1", model: str = DEFAULT_MODEL) -> None:
    load_dotenv(os.path.join(ver2_path(), ".env"))
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not found (looked in ver2/.env). Aborting.")
        return

    provider, task, label = _load(rung)
    print(f"=== LIVE committee solving Rung {rung}: {label} (model={model}) ===")
    print(f"  oracle: {task.metric} {task.op} {task.threshold} on the HIDDEN trajectory\n")

    spend = {"in": 0, "out": 0, "usd": 0.0, "calls": 0}

    def counting_run_fn(prompt, **kw):
        r = run_agent(prompt, **kw)
        spend["in"] += r.usage.tokens_in; spend["out"] += r.usage.tokens_out
        spend["usd"] += r.cost_usd; spend["calls"] += 1
        return r

    transcript: list = []
    author = code_committee_author(run_fn=counting_run_fn, model=model, rounds=2,
                                   transcript=transcript)

    import tempfile
    root = tempfile.mkdtemp(prefix=f"slam-committee-r{rung}-")
    harness = build_implementer_harness(root, task, author_fn=author, provider=provider,
                                        job_mode="local")
    result = harness.run_experiment(ExperimentRecord(id=f"r{rung}", hypothesis=label))

    print("--- the committee meeting ---")
    for step in transcript:
        d = step["data"]
        if step["role"] == "planner":
            print(f"  PLANNER  {d.get('approach', '')[:100]}")
        elif step["role"] == "coder":
            print(f"  CODER    wrote {len((d.get('code') or '').splitlines())}-line solution")
        else:
            v = "approve" if d.get("approve") else f"REJECT -> {d.get('must_fix')}"
            print(f"  REVIEWER {v}")

    metric = result.verdict.measured_metrics.get(task.metric)
    print(f"\n--- VERIFIER (independent, hidden GT) ---")
    print(f"  {task.metric} = {metric}   ->   {result.status.value}")
    print(f"\n  API: {spend['calls']} calls, {spend['in']}+{spend['out']} tok, "
          f"~${spend['usd']:.3f}")
    verdict = ("A real agent team authored working SLAM, confirmed on data it never saw."
               if result.status.value == "VERIFIED" else
               "The verifier REJECTED the agent team's code on hidden GT — exactly its job: "
               "an agent solver gets no free pass.")
    print(f"  {verdict}\n")

    if result.status.value != "VERIFIED":
        coder_steps = [s for s in transcript if s["role"] == "coder"]
        if coder_steps:
            code = coder_steps[-1]["data"].get("code", "")
            print("--- final authored main.py (for diagnosis) ---")
            print("\n".join(code.splitlines()[:45]))
            print("..." if len(code.splitlines()) > 45 else "")


if __name__ == "__main__":
    args = sys.argv[1:]
    rung = "1"; model = DEFAULT_MODEL
    if "--rung" in args:
        rung = args[args.index("--rung") + 1]
    if "--model" in args:
        model = args[args.index("--model") + 1]
    main(rung=rung, model=model)
