"""Rung 0 — "the popcount of SLAM": 2D pose-graph optimization, hidden-GT graded.

A synthetic 3D-testbed task with NO renderer and NO ML: a robot drives a self-overlapping
2D path; we hand the solver a pose graph (drifted odometry + loop-closure constraints) and
ask it to recover the trajectory. The Touchstone verifier grades the estimate against the
GROUND-TRUTH trajectory it never saw, via SE(2)-aligned Absolute Trajectory Error (ATE).

This is the SLAM analog of vision_blobs: it proves the verifier measures a real geometric
oracle (trajectory accuracy), not "did it output a trajectory." An honest pose-graph
optimizer uses the loop closures to cancel drift (low ATE -> VERIFIED); a degenerate
dead-reckoning solver emits the drifted odometry unchanged — it RUNS and looks like a
trajectory, but drifts (high ATE -> REJECTED). "It ran != it's correct."

Held-out worlds (other seeds the producer never authored against) are how overfitting to a
single sequence is caught downstream. Pure numpy, CPU, deterministic."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .._spine import DatasetRef, FrameworkSpec, ImplementationTask, Usage

_SEED = 0
_N = 60          # poses; path makes TWO full loops so every spot is revisited (rich closures)

# --- the SE(2) Gauss-Newton optimizer: ONE source, embedded in the honest solver's main.py
#     and exec'd into this module so tests + world-gen reuse the exact same code. -----------
_SOLVE_SRC = '''\
import numpy as np

def _v2t(p):
    c, s = np.cos(p[2]), np.sin(p[2])
    return np.array([[c, -s, p[0]], [s, c, p[1]], [0.0, 0.0, 1.0]])

def _t2v(T):
    return np.array([T[0, 2], T[1, 2], np.arctan2(T[1, 0], T[0, 0])])

def _nrm(a):
    return np.arctan2(np.sin(a), np.cos(a))

def solve_posegraph(nodes_init, edges, iters=30):
    """Gauss-Newton on SE(2); node 0 anchored. edges: [i, j, dx, dy, dth, w_xy, w_th]."""
    x = [np.array(p, float) for p in nodes_init]
    n = len(x)
    for _ in range(iters):
        H = np.zeros((3 * n, 3 * n)); b = np.zeros(3 * n)
        for e in edges:
            i, j = int(e[0]), int(e[1])
            Z = _v2t([e[2], e[3], e[4]]); Xi = _v2t(x[i]); Xj = _v2t(x[j])
            err = _t2v(np.linalg.inv(Z) @ (np.linalg.inv(Xi) @ Xj)); err[2] = _nrm(err[2])
            Rij = Z[:2, :2].T; Rit = Xi[:2, :2].T
            s, c = np.sin(x[i][2]), np.cos(x[i][2])
            dRit = np.array([[-s, c], [-c, -s]])
            ti, tj = x[i][:2], x[j][:2]
            A = np.zeros((3, 3)); B = np.zeros((3, 3))
            A[:2, :2] = -Rij @ Rit; A[:2, 2] = Rij @ dRit @ (tj - ti); A[2, 2] = -1.0
            B[:2, :2] = Rij @ Rit; B[2, 2] = 1.0
            Om = np.diag([e[5], e[5], e[6]])
            H[3*i:3*i+3, 3*i:3*i+3] += A.T @ Om @ A
            H[3*i:3*i+3, 3*j:3*j+3] += A.T @ Om @ B
            H[3*j:3*j+3, 3*i:3*i+3] += B.T @ Om @ A
            H[3*j:3*j+3, 3*j:3*j+3] += B.T @ Om @ B
            b[3*i:3*i+3] += A.T @ Om @ err
            b[3*j:3*j+3] += B.T @ Om @ err
        H[:3, :3] += np.eye(3) * 1e6        # anchor node 0 (fix the global gauge)
        d = np.linalg.solve(H, -b)
        for k in range(n):
            x[k] = x[k] + d[3*k:3*k+3]; x[k][2] = _nrm(x[k][2])
    return [list(map(float, p)) for p in x]
'''
exec(_SOLVE_SRC, globals())   # -> _v2t, _t2v, _nrm, solve_posegraph (numpy as np)


def _world(seed: int, n: int = _N):
    """Deterministic world: ground-truth poses, drifted odometry init, constraint edges."""
    rng = np.random.default_rng(seed)
    gt = [np.array([0.0, 0.0, 0.0])]
    for _ in range(1, n):
        step = np.array([0.3, 0.0, 2 * (2 * np.pi) / n])     # two full loops over n steps
        gt.append(_t2v(_v2t(gt[-1]) @ _v2t(step)))
    gt = np.array(gt)

    odo = np.array([0.04, 0.04, 0.025]); lc = np.array([0.02, 0.02, 0.01])
    edges: list = []; nodes = [gt[0].copy()]
    for i in range(n - 1):                                   # sequential odometry (noisy)
        rel = _t2v(np.linalg.inv(_v2t(gt[i])) @ _v2t(gt[i + 1]))
        meas = rel + rng.normal(0, odo)
        edges.append([i, i + 1, *meas.tolist(), 1 / odo[0] ** 2, 1 / odo[2] ** 2])
        nodes.append(_t2v(_v2t(nodes[-1]) @ _v2t(meas)))     # integrate -> drifted guess
    P = gt[:, :2]
    for i in range(n):                                       # loop closures (revisits)
        for j in range(i + 5, n):
            if np.linalg.norm(P[i] - P[j]) < 0.20:
                rel = _t2v(np.linalg.inv(_v2t(gt[i])) @ _v2t(gt[j]))
                meas = rel + rng.normal(0, lc)
                edges.append([i, j, *meas.tolist(), 1 / lc[0] ** 2, 1 / lc[2] ** 2])
    return gt, np.array(nodes), edges


def ate(est_xyt, gt_xyt) -> float:
    """SE(2)-aligned Absolute Trajectory Error (translation RMSE). Harness owns this — the
    solver cannot game the global rotation/translation gauge."""
    P = np.asarray(est_xyt)[:, :2]; Q = np.asarray(gt_xyt)[:, :2]
    n = min(len(P), len(Q)); P, Q = P[:n], Q[:n]
    mp, mq = P.mean(0), Q.mean(0); Pc, Qc = P - mp, Q - mq
    U, _, Vt = np.linalg.svd(Pc.T @ Qc)
    D = np.eye(2); D[1, 1] = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ D @ U.T; t = mq - R @ mp
    aligned = (R @ P.T).T + t
    return float(np.sqrt(((aligned - Q) ** 2).sum(1).mean()))


def _poses_csv(poses) -> str:
    return "\n".join(f"{p[0]},{p[1]},{p[2]}" for p in poses)


class PoseGraphProvider:
    """inputs split: graph.json (nodes_init + edges). held-out: gt_poses.csv (true trajectory)."""

    def fetch(self, ref: DatasetRef, dest: Path) -> None:
        gt, nodes, edges = _world(_SEED)
        if ref.held_out:
            (dest / "gt_poses.csv").write_text(_poses_csv(gt))
        else:
            (dest / "graph.json").write_text(json.dumps(
                {"nodes_init": [list(map(float, p)) for p in nodes], "edges": edges}))


# harness-owned grader: SE(2)-aligned ATE of the estimate vs the hidden GT trajectory
_EVAL = '''\
import csv, json, os
import numpy as np
def load(p):
    return np.array([[float(v) for v in r] for r in csv.reader(open(p)) if r], dtype=float)
est = load(os.path.join(os.environ["LAB_ARTIFACTS"], "trajectory.csv"))
gt  = load(os.path.join(os.environ["LAB_DATA"], "gt_poses.csv"))
n = min(len(est), len(gt))
if n == 0 or est.ndim != 2 or est.shape[1] < 2:
    ate = 9.99
else:
    P, Q = est[:n, :2], gt[:n, :2]
    mp, mq = P.mean(0), Q.mean(0); Pc, Qc = P - mp, Q - mq
    U, _, Vt = np.linalg.svd(Pc.T @ Qc)
    D = np.eye(2); D[1, 1] = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ D @ U.T; t = mq - R @ mp
    al = (R @ P.T).T + t
    ate = float(np.sqrt(((al - Q) ** 2).sum(1).mean()))
json.dump({"ate": round(ate, 4)},
          open(os.path.join(os.environ["LAB_EVAL_OUT"], "heldout.json"), "w"))
'''


def posegraph_task() -> ImplementationTask:
    return ImplementationTask(
        description=(
            "Estimate a robot's 2D trajectory by optimizing a pose graph. Read "
            "$LAB_DATA/graph.json: 'nodes_init' is an initial guess of each pose "
            "[x, y, theta] from drifted odometry; 'edges' are relative-pose constraints "
            "[i, j, dx, dy, dtheta, w_xy, w_theta] — sequential odometry AND loop closures "
            "between revisited places. Recover the poses that best satisfy ALL constraints "
            "(e.g. Gauss-Newton on SE(2), anchoring pose 0) so accumulated drift is "
            "cancelled. Write the optimized trajectory — one 'x,y,theta' per line, same "
            "order as nodes_init — to $LAB_ARTIFACTS/trajectory.csv."),
        framework=FrameworkSpec("numpy", "2.3", "cpu"),
        entry_command="python3 $LAB_CODE/main.py",
        eval_command="python3 $LAB_CODE/eval.py", eval_code=_EVAL,
        metric="ate", op="<=", threshold=0.12,
        datasets=[DatasetRef("pg2d-graph", "synthetic"),
                  DatasetRef("pg2d-gt", "synthetic", held_out=True)],
        entry_filename="main.py")


# --- canned solvers (offline; numpy-only so they run on the host via local mode) ----------
def _writer(body: str):
    def author(task, code_dir: Path, rec) -> Usage:
        (code_dir / "main.py").write_text(body)
        return Usage(0, 0)
    return author


_HONEST_BODY = _SOLVE_SRC + '''
import json, os
g = json.load(open(os.path.join(os.environ["LAB_DATA"], "graph.json")))
x = solve_posegraph(g["nodes_init"], g["edges"])
open(os.path.join(os.environ["LAB_ARTIFACTS"], "trajectory.csv"), "w").write(
    "\\n".join(f"{p[0]},{p[1]},{p[2]}" for p in x))
'''

_ODOMETRY_BODY = '''\
import json, os
g = json.load(open(os.path.join(os.environ["LAB_DATA"], "graph.json")))
# DEGENERATE: dead-reckoning — emit the drifted odometry guess; never optimize or use the
# loop closures. It runs and looks like a trajectory, but the drift is never corrected.
open(os.path.join(os.environ["LAB_ARTIFACTS"], "trajectory.csv"), "w").write(
    "\\n".join(f"{p[0]},{p[1]},{p[2]}" for p in g["nodes_init"]))
'''

HONEST = _writer(_HONEST_BODY)        # real pose-graph optimization -> VERIFIED
ODOMETRY = _writer(_ODOMETRY_BODY)    # dead-reckoning, no loop closures -> REJECTED
