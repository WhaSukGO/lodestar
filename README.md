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

**Docs:** [docs/DESIGN.md](docs/DESIGN.md) — architecture, the verifier path, what each rung
is. [docs/ENGINEERING.md](docs/ENGINEERING.md) — harness-engineering principles, the past
failures that shaped them, and the geometry/numerics tradeoffs taken.

## Preview — what the verifier sees

The verdict is decided by a held-out number (ATE/RPE), but the picture shows *why*: drift
accumulates, and loop closure pulls it back. (`python -m slamtest.run_viz`)

| Rung 0 — 2D pose-graph | Rung 1 — RGBD VO | Rung 2 — full SLAM |
|---|---|---|
| ![](docs/img/rung0_posegraph.png) | ![](docs/img/rung1_vo.png) | ![](docs/img/rung2_slam.png) |

Green (honest) lands on the hidden ground truth → VERIFIED; red (degenerate) drifts off →
REJECTED. The plots are an inspection tool only — grading is still the independent oracle, not
eyeballing.

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

## Rung 1 (done): RGBD visual odometry — the SLAM front-end

Still no renderer, no ML, pure numpy. A camera flies through a cloud of 3D landmarks; each
frame yields RGBD feature observations `[landmark_id, u, v, depth]` (a shared `landmark_id`
is a track). The solver back-projects pixels+depth to 3D, matches tracks between consecutive
frames, estimates each frame-to-frame rigid motion (Procrustes/Umeyama SE(3)), and chains
them. RGBD = metric scale, so the verifier scores **Relative Pose Error (RPE)** directly
against the **hidden** ground-truth trajectory.

| Solver | RPE (hidden GT) | Verdict |
|---|---|---|
| Honest RGBD VO (Procrustes per frame pair) | **0.04 m** | VERIFIED |
| Static — "the camera never moved" (identity every frame) | **0.36 m** | REJECTED |

```bash
python -m slamtest.run_vo_demo
```

## Rung 2 (done): full visual SLAM — VO + loop closure + SE(3) pose-graph

The first complete SLAM loop, still pure numpy. It fuses the earlier rungs: Rung 1's RGBD VO
is the **front-end**, Rung 0's pose-graph optimization — now in 3D (SE(3)) — is the
**back-end**. The camera orbits a scene twice, so frames far apart in time re-observe the
same landmarks; those **loop closures** tie the trajectory together and cancel drift. The
oracle returns to global **ATE**.

| Solver | ATE (hidden GT) | Verdict |
|---|---|---|
| Full SLAM (odometry + loop closure + optimization) | **0.04 m** | VERIFIED |
| VO only — Rung 1's odometry with no loop closure | **0.17 m** | REJECTED |

Both *ran*; loop closure is **what makes SLAM more than odometry**, and the verifier
measures exactly that on a held-out trajectory.

```bash
python -m slamtest.run_slam_demo
```

## Roadmap (rung by rung — start cheap, add fidelity only when each rung holds)

| Rung | Task | Input | Oracle | Deps |
|---|---|---|---|---|
| **0 ✅** | 2D pose-graph optimization | constraint graph | ATE | numpy |
| **1 ✅** | RGBD visual odometry | synthetic feature tracks | RPE | numpy |
| **2 ✅** | full visual SLAM + loop closure | feature tracks + revisits | ATE | numpy |
| 3 | image-based SLAM | rendered frames | ATE | Habitat / Blender headless |

## Live: a committee of agents as the solver (done)

The highlight of the two-layer model — the verifier and domain are unchanged; only the
**solver** is swapped from canned reference code to a real multi-agent team (ver2's
`code_committee`: PLANNER → CODER → REVIEWER, each a live Claude session). The committee
authors the algorithm **blind** (it reasons and writes code; it does not execute it), and the
verifier grades the result on the hidden trajectory.

| Solver | Rung | Metric (hidden GT) | Verdict |
|---|---|---|---|
| Live committee (PLANNER→CODER→REVIEWER) | 1 (RGBD VO) | RPE **0.122** | VERIFIED |

A real agent team authored working visual odometry (a 198-line solution, reviewer-approved),
confirmed on data it never saw — ~3 API calls, ≈\$0.78. An agent solver still gets **no free
pass**: it is graded by the same fixed verifier as everything else.

```bash
python -m slamtest.run_committee_live --rung 1     # spends API tokens; key from ver2/.env
```

(The solver runs under a `timeout` guard so blind-authored code that loops is rejected, not
hung.)

## Reuse

`slamtest/_spine.py` locates the ver2 checkout (`$TOUCHSTONE_PATH`, else `../blueberry_ver2`)
and re-exports the seam ver4 plugs into: `build_implementer_harness`, `ImplementationTask`,
`DatasetRef`. The pose-graph world is just a richer `DatasetProvider`; the SLAM algorithm is
the swappable `author_fn`; the ATE grader is the task's fixed `eval_code`.
