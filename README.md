# Touchstone-SLAM (ver4) — a 3D testbed that *verifies* SLAM

A **ground-truth generator + geometric oracle** for SLAM. SLAM always outputs *a*
trajectory and *a* map — whether it's **correct** can't be known without ground truth you
can't cheaply get in the real world. A synthetic world gives perfect ground truth for free,
so an algorithm's estimate can be scored on a **held-out** trajectory it never saw.

This repo is the **world builder + SLAM oracle**. The **verifier is reused, not forked** —
it's the same Touchstone spine from [`blueberry_ver2`](../blueberry_ver2) (sandboxed run +
held-out grading + anti-tamper grader restoration). The whole thesis: *swap the
solver/domain, keep the verifier.*

```
SOLVER (swappable)                         VERIFIER (ver2 spine, the constant)
a SLAM algorithm: optimize the      -->    run it in a sandbox, align the estimate to the
graph / VO / full SLAM                     HIDDEN ground-truth trajectory, score ATE, gate
```

## Rung 0 (done): "the popcount of SLAM" — 2D pose-graph optimization

No renderer, no ML, pure numpy. A robot drives a self-overlapping 2D path; the solver gets
a pose graph (drifted odometry + loop-closure constraints) and must recover the trajectory.
The verifier grades it against the **hidden** ground-truth trajectory via SE(2)-aligned
**Absolute Trajectory Error (ATE)**.

| Solver | ATE (hidden GT) | Verdict |
|---|---|---|
| Honest pose-graph optimization (Gauss-Newton, uses loop closures) | **0.06 m** | VERIFIED |
| Dead-reckoning (odometry only, ignores closures) | **0.28 m** | REJECTED |

Both *ran* and produced a trajectory; only the one that actually cancels drift passes.
**It ran ≠ it's correct.** Held-out worlds (other seeds the producer never authored
against) are how overfitting to one sequence gets caught.

```bash
python -m slamtest.run_posegraph_demo      # offline, no API spend
python -m pytest tests/ -q
```

## Roadmap (rung by rung — start cheap, add fidelity only when each rung holds)

| Rung | Task | Input | Oracle | Deps |
|---|---|---|---|---|
| **0 ✅** | 2D pose-graph optimization | constraint graph | ATE | numpy |
| 1 | visual odometry | synthetic feature tracks | RPE | numpy |
| 2 | full visual SLAM + loop closure | synthetic 2D observations | ATE + map | numpy |
| 3 | image-based SLAM | rendered frames | ATE | Habitat / Blender headless |

Then layer a **committee solver** (multi-agent, from ver2's `code_committee`) that assembles
the SLAM modules — graded, as always, by the fixed verifier.

## Reuse

`slamtest/_spine.py` locates the ver2 checkout (`$TOUCHSTONE_PATH`, else `../blueberry_ver2`)
and re-exports the seam ver4 plugs into: `build_implementer_harness`, `ImplementationTask`,
`DatasetRef`. The pose-graph world is just a richer `DatasetProvider`; the SLAM algorithm is
the swappable `author_fn`; the ATE grader is the task's fixed `eval_code`.
