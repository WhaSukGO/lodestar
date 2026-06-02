# Design — Touchstone-SLAM (ver4)

> How the project is structured, what is reused vs. new, and how a SLAM algorithm flows
> through the verifier. For the *why* behind the engineering choices (and the past failures
> that shaped them), see [ENGINEERING.md](ENGINEERING.md).

## 1. One sentence

A **ground-truth generator + geometric oracle** for SLAM: you plug in a SLAM algorithm and
its estimate is measured against a **held-out** trajectory it never saw — so "it produced a
trajectory" is never mistaken for "the trajectory is correct."

## 2. Two layers: a swappable SOLVER, a fixed VERIFIER

```
SOLVER (swappable)                         VERIFIER (reused from ver2, the constant)
a SLAM algorithm that writes code:   -->   run it in a sandbox, grade its output on a
  pose-graph / VO / full SLAM /             HELD-OUT split with a harness-owned oracle,
  (later) a committee of agents            restore the grader before judging, gate pass/fail
```

The **verifier is not forked** — it is the same Touchstone spine from
[`blueberry_ver2`](../../blueberry_ver2), bridged in by `slamtest/_spine.py`
(`$TOUCHSTONE_PATH`, else the sibling `../blueberry_ver2`). ver4 only adds the
**domain**: worlds, algorithms, and oracles. The whole thesis is *swap the solver/domain,
keep the verifier* — so every rung below is the same spine with a different provider,
`author_fn`, and `eval_code`.

## 3. What a rung is

Each rung is four things, and nothing else:

| Piece | Type (ver2 seam) | Rung 0 | Rung 1 | Rung 2 |
|---|---|---|---|---|
| **World** | `DatasetProvider` | 2D pose graph | RGBD feature tracks | looping RGBD tracks |
| **Algorithm** | `author_fn` (writes `main.py`) | SE(2) pose-graph GN | Procrustes VO | VO + loop closure + SE(3) GN |
| **Oracle** | task `eval_code` (`eval.py`) | SE(2)-aligned ATE | translational RPE | SE(3)-aligned ATE |
| **Bar** | `Criterion(metric, op, threshold)` | `ate <= 0.12` | `rpe <= 0.15` | `ate <= 0.10` |

All three live in `slamtest/worlds/{posegraph2d, visual_odometry, visual_slam}.py`. Adding a
rung means writing one file; the verifier is untouched.

## 4. Data-flow contract (held-out by construction)

The provider writes **two splits**; the solver only ever sees one:

```
provider.fetch(inputs)   -> graph.json / vo.json     (the problem)      ── solver reads this
provider.fetch(held_out) -> gt_poses.csv             (the answer)       ── ONLY the grader reads this
```

During the **run** step the harness mounts the inputs split as `$LAB_DATA`; the solver writes
its estimate to `$LAB_ARTIFACTS/trajectory.csv`. During **grading** the harness mounts the
*held-out* split as `$LAB_DATA`, and `eval.py` compares the persisted estimate against the
hidden ground truth. The solver never has the ground-truth trajectory in scope. (This is the
exact contract ver2's `vision_blobs` uses — verified in `evaluator.py:_heldout_dir`.)

## 5. The end-to-end path (one experiment)

```
Implementer.propose_contract(rec)
  ├─ writes eval.py  := task.eval_code         # harness owns the grader; solver never writes it
  ├─ author_fn(task, code_dir, rec)            # the SOLVER writes main.py (this is the swap point)
  └─ builds ExperimentContract(criterion, eval_code, code_dir, ...)
Harness.run_experiment
  ├─ run:   python3 main.py   ($LAB_DATA = inputs, $LAB_ARTIFACTS = scratch)
  └─ grade: ScriptEvaluator
        ├─ re-writes eval.py from the contract   # anti-tamper: restore grader before judging
        ├─ run: python3 eval.py  ($LAB_DATA = HELD-OUT)  -> heldout.json {metric: value}
        └─ Criterion.satisfied(value) -> VERIFIED | REJECTED
```

Two properties fall out of this shape:
1. **The producer cannot grade itself.** The grader (`eval.py`) is harness-owned and is
   re-instantiated from the contract immediately before judging, so anything the solver did
   to it during the run is overwritten.
2. **The bar is fixed before the run.** `metric/op/threshold` come from the task, not from
   the solver's report. Measured-vs-claimed cannot diverge in the solver's favor.

## 6. The negative control is shipped, not assumed

Every rung ships **two** solvers and a test that asserts both verdicts:

| Rung | Positive control (must VERIFY) | Negative control (must REJECT) |
|---|---|---|
| 0 | Gauss-Newton pose-graph — ATE 0.06 | dead-reckoning (odometry only) — ATE 0.28 |
| 1 | RGBD VO (Procrustes/frame) — RPE 0.04 | "camera never moved" (identity) — RPE 0.36 |
| 2 | full SLAM (+loop closure) — ATE 0.04 | VO only (no loop closure) — ATE 0.17 |

The negative control is **a real algorithm that runs cleanly and emits a well-formed
trajectory** — it is wrong only on the held-out metric. Each negative control is chosen to be
the *previous* rung's idea applied where it no longer suffices (e.g. Rung 1's honest VO is
Rung 2's negative control on a looping path). If a negative control ever passed, the oracle
would be measuring "did it output a trajectory," not "is it correct" — so the tests guard the
verifier itself, not just the solvers.

## 7. Geometric design choices (summary; rationale in ENGINEERING.md)

- **RGBD / metric scale**, not monocular — removes scale ambiguity, so RPE/ATE are direct
  and the oracle needs no Sim(3) gauge fitting.
- **Oracle owns the alignment** (Umeyama SE(2)/SE(3) on positions) — the solver cannot game
  the global rotation/translation gauge; only the trajectory *shape* is graded.
- **Rung 2 fuses Rung 0 + Rung 1**: VO front-end (motion from tracks) + loop closures +
  SE(3) pose-graph back-end. ATE returns as the metric because loop closure is precisely a
  *global*-consistency claim.

## 8. What is deliberately not here yet

- **Rung 3 (image-based)** — a real renderer (Habitat / Blender headless). Held back on
  purpose: the renderer is a commodity that can swallow months; fidelity is added only after
  the cheap rungs hold. See ENGINEERING.md §"Scoping discipline."
- **Committee solver** — ver2's `code_committee` (multi-agent PLANNER→CODER→REVIEWER) as the
  `author_fn`, so agents *author* the SLAM modules instead of canned reference code. The seam
  is already the swap point; this is a solver swap, not a verifier change.
- **Held-out *worlds*** — currently one fixed seed per rung. Grading on *other* seeds (varied
  scenes/paths the producer never authored against) is how overfitting-to-one-sequence gets
  caught; the provider is already seed-parameterized for this.

## 9. Repository map

```
slamtest/
  _spine.py                 bridge to ver2's verifier (re-exports 6 symbols; no heavy deps)
  worlds/
    posegraph2d.py          Rung 0: world + SE(2) GN optimizer + ATE oracle + 2 solvers
    visual_odometry.py      Rung 1: world + RGBD VO front-end + RPE oracle + 2 solvers
    visual_slam.py          Rung 2: world + VO+loop-closure+SE(3) GN + ATE oracle + 2 solvers
  run_*_demo.py             narrated, offline, no API spend
tests/                      per rung: end-to-end (through the real verifier) + offline algorithm
docs/                       this file + ENGINEERING.md
```
