# Engineering — what was considered, and why

> The reasoning, the failures it came from, and the tradeoffs taken. For the structure
> itself, see [DESIGN.md](DESIGN.md).

## 0. Where this comes from

This project is the SLAM-shaped continuation of a **verification-first harness** (ver2,
"Touchstone"). That harness was itself a reaction to two concrete failures and two Anthropic
write-ups:

- A failed **`claude-code` + ralph-loop** attempt at a long-running autonomous build. It
  failed in instructive ways: turn quotas ran out mid-task, GPU/Docker plumbing was done
  *inside* the agent loop, and — the expensive one — **"it ran" was repeatedly mistaken for
  "it worked."**
- Anthropic's ***building a C compiler*** and ***harness design for long-running agents***
  posts: keep determinism and heavy IO out of the model loop; make success a *gradable
  contract* checked by an *independent* evaluator; don't bite off a task so large it can't be
  verified incrementally.

Every decision below traces back to one of those.

## 1. "It ran ≠ it's correct" — the load-bearing principle

A SLAM system *always* outputs a trajectory. A buggy one outputs a *plausible* trajectory.
So the harness is built so that **producing well-formed output earns nothing** — only the
held-out metric does.

Concretely, every rung ships a **negative control**: a real algorithm that runs cleanly and
emits a valid `trajectory.csv` but is wrong on the hidden ground truth (DESIGN.md §6). The
tests assert it is **REJECTED**. The negative control is the experiment that proves the
oracle measures correctness rather than liveness — if it ever passed, the verifier would be
broken, and we'd know immediately.

This is also why the metrics are **ground-truth-referenced geometric errors** (ATE/RPE on a
hidden trajectory), never self-reported confidences or "the optimizer converged" flags. A
solver cannot make ATE small by asserting it did well.

## 2. Generator ≠ evaluator (independence, enforced mechanically)

The solver writes **only** `main.py`. The grader is `eval.py`, and it is **harness-owned**:

- It is written by `Implementer.propose_contract` from the *task*, not by the solver.
- It is **re-written from the contract immediately before grading** (ver2
  `evaluator.py`: `(code_dir/"eval.py").write_text(contract.eval_code)`), so any tampering
  during the run is overwritten — anti-tamper by restoration, not by detection.
- The **ground-truth split is mounted only at grading time**. During the run, `$LAB_DATA`
  is the inputs split; the answer (`gt_poses.csv`) is never in the solver's scope.

The bar (`metric/op/threshold`) is fixed in the task before the run, so "measured vs
claimed" cannot drift in the solver's favor. In the live committee variant (future), the
authoring agent additionally runs in ver2's sandbox (container-only execution, no host
shell, writes confined to the code dir, `eval.py` off-limits) — the same independence,
enforced at the OS boundary.

## 3. Determinism and "work outside the loop"

- **Worlds are deterministic**: `np.random.default_rng(seed)`; the same seed reproduces the
  exact GT, observations, and noise. This is what makes a held-out metric meaningful and
  tests stable.
- **Heavy work is the harness's job, not the agent's.** World generation, running code, and
  grading happen in the deterministic harness; the solver only authors an algorithm. This is
  the direct lesson from the ralph-loop attempt, where doing IO/plumbing in the loop made
  runs nondeterministic and burned the turn budget.
- **Budget is tokens + experiments, not turns** (inherited from ver2). Offline canned solvers
  don't spend tokens; the live committee path inherits this so a solver can't "run out of
  turns" mid-think.

## 4. Calibration: positive AND negative control, every rung

Autonomy is only trustworthy behind a calibration gate. Here that gate is concrete and
per-rung: a **positive control** (honest algorithm, must VERIFY) and a **negative control**
(degenerate algorithm, must REJECT), both shipped as tests. Thresholds were tuned
empirically across multiple seeds to leave a **margin** between the two, not fitted to a
single run:

| Rung | honest (seeds) | degenerate (seeds) | threshold | margin |
|---|---|---|---|---|
| 0 | 0.036–0.062 | 0.18–0.30 | 0.12 | ~2× |
| 1 | 0.044–0.061 | 0.359 | 0.15 | ~3× |
| 2 | 0.043–0.057 | 0.13–0.25 | 0.10 | ~2× |

If these collapsed together, the task would be uncalibrated and we would *not* trust a
VERIFIED verdict from it.

## 5. Scoping discipline — the anti-"too big to verify" rule

The C-compiler lesson is: a task you can't verify incrementally is a task you can't trust.
So SLAM was deliberately **not** started as "implement ORB-SLAM3 on photorealistic video."
It was decomposed into rungs that each (a) have a crisp numeric oracle and (b) cost seconds:

```
Rung 0  pose-graph back-end      (the "popcount of SLAM": smallest real geometric oracle)
Rung 1  VO front-end             (add the perception step)
Rung 2  front-end + back-end + loop closure   (the first full loop)
Rung 3  swap synthetic geometry for a renderer (fidelity, last)
```

**Hard rule: no renderer until Rung 3.** A renderer is a commodity that can swallow months,
and it is *not* where the value is — the value is the verifier + the oracle. Rungs 0–2 are
pure `numpy`, no renderer, no ML, CPU, deterministic, sub-10-second test suite. Fidelity is
bought only after the cheap thing demonstrably works, and even then by **reusing an existing
sim** (Habitat / Blender headless), never by building one.

The same rule applies to the *solver*: when the committee authors SLAM, it authors **modules**
(VO, loop closure, pose-graph), not a monolith — each module is independently gradable.

## 6. Geometry/numerics tradeoffs (taken deliberately)

- **RGBD over monocular.** Monocular SLAM is scale-ambiguous, which forces a Sim(3) gauge fit
  in the oracle and muddies "is the error real or just scale." RGBD gives metric depth, so
  ATE/RPE are direct. Chosen for a *cleaner oracle*, accepting that it sidesteps the
  scale-estimation sub-problem (a later rung could reintroduce it on purpose).
- **Oracle owns the alignment.** ATE/RPE align the estimate to GT with Umeyama
  (SE(2)/SE(3), no scale) *inside the grader*. The solver therefore cannot win by choosing a
  convenient world frame — only trajectory shape is graded, which is the honest question.
- **Numerical Jacobians in the SE(3) optimizer (Rung 2).** The back-end is Gauss-Newton with
  finite-difference Jacobians and a split translation/SO(3) perturbation, rather than the
  analytic se(3) left-Jacobian. This trades elegance/speed for a **much smaller bug surface**
  — the correctness of the optimizer is easy to see and test. Cost is acceptable at this
  scale (~64 poses, ~178 loop closures, dense 384×384 system, ~12 iterations, ≈3 s).
- **Dense `H` + `np.linalg.solve`.** Fine for tens of poses; a production system would
  exploit the sparse block structure (Cholesky / Schur). This is a deliberate scale limit,
  documented rather than hidden — the rung is about *correctness of the oracle*, not solver
  throughput. A small Levenberg term (`+1e-6·I`) guards conditioning.
- **Constant-velocity fallback for thin overlap (VO).** When two frames share `<3` tracks,
  the front-end reuses the previous motion instead of solving a degenerate Procrustes. This
  mirrors real VO and keeps the pipeline robust without faking a measurement.

## 7. The reuse decision (and its honest cost)

ver4 **reuses** ver2's verifier rather than forking it (`_spine.py` re-exports 6 symbols).

- **Benefit:** one source of truth for the verifier. Improvements to grading/anti-tamper/
  sandboxing in ver2 apply to SLAM for free, and the SLAM rungs are *evidence the verifier
  generalizes across domains* (coding → vision → geometry).
- **Cost:** ver4 is **not standalone** — it needs a ver2 checkout present
  (`$TOUCHSTONE_PATH` or `../blueberry_ver2`); a lone clone fails loudly in `_spine.py`. The
  coupling is thin, though: importing `lab.factory` pulls in **no heavy dependency**
  (`claude_agent_sdk`, `torch`, `docker` are all lazy), so the offline rungs need only
  `Python + numpy + the ver2 spine`. Standalone packaging (pip dependency, submodule, or
  vendoring) is a known, deferred option.

## 8. Testing strategy

Two tests per rung, testing different things:

1. **End-to-end through the real verifier** — honest solver VERIFIED, degenerate REJECTED.
   This tests the *whole spine* (provider → run → anti-tamper → held-out grade → criterion),
   not just the algorithm.
2. **Offline algorithm check** — honest beats degenerate by a margin (e.g. optimized ATE
   `< 0.5 ×` odometry ATE). This pins the *geometry* independent of the harness, so a
   regression is localizable to either the algorithm or the spine.

Both run offline, on CPU, with **no API spend**, in well under ten seconds total — itself a
scoping choice: the verification loop has to be cheap enough to run constantly.

## 9. One-line summary

> Make "it ran" worth nothing; make the held-out number the only currency; keep the grader
> out of the solver's hands; and grow the hard problem one verifiable rung at a time.
