"""Rung 2 — full visual SLAM: VO front-end + loop closure + SE(3) pose-graph, ATE graded.

The first complete SLAM loop, still pure numpy / no renderer / no ML. It fuses the two
earlier rungs: Rung 1's RGBD visual odometry (frame-to-frame motion from feature tracks)
is the FRONT-END, and Rung 0's pose-graph optimization — now in 3D (SE(3)) — is the
BACK-END. The new ingredient is LOOP CLOSURE: the camera orbits a scene twice, so frames
far apart in time re-observe the same landmarks; those revisits become constraints that tie
the trajectory together and cancel accumulated drift. The oracle returns to global
**Absolute Trajectory Error (ATE)**.

This makes the SLAM-vs-VO distinction concrete: the honest solver runs the full pipeline
(odometry + loop closures + optimization) → drift removed → low ATE → VERIFIED. The
degenerate solver is exactly Rung 1's VO with NO loop closure / NO optimization — a perfectly
reasonable algorithm that still RUNS and emits a trajectory, but drifts on a looping path →
high ATE → REJECTED. Loop closure is *what makes SLAM more than odometry*, and the verifier
measures it on a held-out trajectory. Pure numpy, CPU, deterministic."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .._spine import DatasetRef, FrameworkSpec, ImplementationTask, Usage

_SEED = 0
_F = 64          # frames (two orbits)
_L = 500         # landmarks

# --- the full SLAM pipeline: ONE source, embedded in the honest solver's main.py and exec'd
#     into this module so tests reuse the exact same code. ----------------------------------
_SLAM_SRC = '''\
import numpy as np

def _hat(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])

def _so3_exp(w):
    th = np.linalg.norm(w)
    if th < 1e-9:
        return np.eye(3) + _hat(w)
    K = _hat(w / th)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)

def _so3_log(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1); th = np.arccos(c)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return v / 2 if th < 1e-9 else th / (2 * np.sin(th)) * v

def _T(R, t):
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = t; return M

def _bp(obs, intr):
    fx, fy, cx, cy = intr
    return {int(o[0]): np.array([(o[1] - cx) / fx * o[3], (o[2] - cy) / fy * o[3], o[3]])
            for o in obs}

def _proc(A, B):
    a = A.mean(0); b = B.mean(0)
    U, _, Vt = np.linalg.svd((A - a).T @ (B - b))
    D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ D @ U.T
    return R, b - R @ a

def _graph(intr, frames, gap=8, min_shared=12):
    """Front-end: back-project each frame; build odometry edges (consecutive shared tracks)
    and loop-closure edges (far-in-time frames sharing >= min_shared tracks). Also returns
    the drift-prone odometry-chained initial trajectory."""
    bps = [_bp(f, intr) for f in frames]
    T = [np.eye(4)]; edges = []
    for i in range(len(frames) - 1):
        c = sorted(set(bps[i]) & set(bps[i + 1]))
        Xi = np.array([bps[i][k] for k in c]); Xj = np.array([bps[i + 1][k] for k in c])
        R, t = _proc(Xj, Xi); edges.append((i, i + 1, R, t)); T.append(T[-1] @ _T(R, t))
    for i in range(len(frames)):
        for j in range(i + gap, len(frames)):
            c = sorted(set(bps[i]) & set(bps[j]))
            if len(c) >= min_shared:
                Xi = np.array([bps[i][k] for k in c]); Xj = np.array([bps[j][k] for k in c])
                R, t = _proc(Xj, Xi); edges.append((i, j, R, t))
    return T, edges

def _optimize(T0, edges, iters=12, eps=1e-5):
    """SE(3) pose-graph Gauss-Newton with numerical Jacobians and a split translation/SO(3)
    perturbation; node 0 anchored."""
    T = [t.copy() for t in T0]; N = len(T)
    def err(Ti, Tj, R, t):
        M = np.linalg.inv(_T(R, t)) @ (np.linalg.inv(Ti) @ Tj)
        return np.concatenate([M[:3, 3], _so3_log(M[:3, :3])])
    def pert(Tk, d):
        return _T(_so3_exp(d[3:]) @ Tk[:3, :3], Tk[:3, 3] + d[:3])
    for _ in range(iters):
        H = np.zeros((6 * N, 6 * N)); b = np.zeros(6 * N)
        for (i, j, R, t) in edges:
            e0 = err(T[i], T[j], R, t); Ji = np.zeros((6, 6)); Jj = np.zeros((6, 6))
            for d in range(6):
                dd = np.zeros(6); dd[d] = eps
                Ji[:, d] = (err(pert(T[i], dd), T[j], R, t) - e0) / eps
                Jj[:, d] = (err(T[i], pert(T[j], dd), R, t) - e0) / eps
            H[6*i:6*i+6, 6*i:6*i+6] += Ji.T @ Ji; H[6*i:6*i+6, 6*j:6*j+6] += Ji.T @ Jj
            H[6*j:6*j+6, 6*i:6*i+6] += Jj.T @ Ji; H[6*j:6*j+6, 6*j:6*j+6] += Jj.T @ Jj
            b[6*i:6*i+6] += Ji.T @ e0; b[6*j:6*j+6] += Jj.T @ e0
        H[:6, :6] += np.eye(6) * 1e6
        step = np.linalg.solve(H + np.eye(6 * N) * 1e-6, -b)
        for k in range(N):
            T[k] = pert(T[k], step[6*k:6*k+6])
    return T

def run_vo_only(intrinsics, frames):
    """Front-end ONLY: chain consecutive motions; no loop closure, no optimization."""
    T, _ = _graph(intrinsics, frames)
    return T

def run_slam(intrinsics, frames):
    """Full SLAM: odometry + loop closures, then SE(3) pose-graph optimization."""
    T0, edges = _graph(intrinsics, frames)
    return _optimize(T0, edges)

def _pose_line(T):
    return ",".join(repr(float(v)) for v in list(T[:3, :3].flatten()) + list(T[:3, 3]))
'''
exec(_SLAM_SRC, globals())   # -> _T, _bp, _proc, _graph, _optimize, run_slam, run_vo_only, ...


def _orbit_pose(f: int, F: int, r: float = 4.5):
    th = 2 * np.pi * (2 * f / F)                          # two orbits
    pos = np.array([r * np.cos(th), 0.3 * np.sin(f * 0.3), r * np.sin(th)])
    fwd = np.array([np.cos(th), 0.0, np.sin(th)])         # look radially outward
    right = np.cross([0, 1.0, 0], fwd); right /= np.linalg.norm(right)
    up = np.cross(fwd, right)
    return _T(np.column_stack([right, up, fwd]), pos)


def _world(seed: int, F: int = _F, L: int = _L):
    """Deterministic looping world: GT poses (orbit x2), intrinsics, per-frame RGBD obs."""
    rng = np.random.default_rng(seed)
    fx = fy = 320.0; cx = cy = 240.0; W = Hh = 480
    poses = [_orbit_pose(f, F) for f in range(F)]
    ang = rng.uniform(0, 2 * np.pi, L); rad = rng.uniform(7, 12, L)       # ring of landmarks
    M = np.c_[rad * np.cos(ang), rng.uniform(-3, 3, L), rad * np.sin(ang)]
    intr = (fx, fy, cx, cy); frames = []
    for T in poses:
        R, tt = T[:3, :3], T[:3, 3]
        Xc = (R.T @ (M - tt).T).T
        out = []
        for lid, X in enumerate(Xc):
            Z = X[2]
            if Z < 0.5 or Z > 15.0:
                continue
            u = fx * X[0] / Z + cx; v = fy * X[1] / Z + cy
            if 0 <= u < W and 0 <= v < Hh:
                out.append([int(lid), float(u + rng.normal(0, 0.9)),
                            float(v + rng.normal(0, 0.9)), float(Z * (1 + rng.normal(0, 0.012)))])
        frames.append(out)
    return poses, intr, frames


def ate(est_poses, gt_poses) -> float:
    """SE(3)-aligned Absolute Trajectory Error (translation RMSE over camera positions)."""
    n = min(len(est_poses), len(gt_poses))
    P = np.array([est_poses[i][:3, 3] for i in range(n)])
    Q = np.array([gt_poses[i][:3, 3] for i in range(n)])
    R, t = _proc(P, Q)
    aligned = (R @ P.T).T + t
    return float(np.sqrt(((aligned - Q) ** 2).sum(1).mean()))


def _poses_csv(poses) -> str:
    return "\n".join(_pose_line(T) for T in poses)


class VisualSlamProvider:
    """inputs split: vo.json (intrinsics + per-frame RGBD obs over a LOOPING path).
    held-out: gt_poses.csv (true trajectory)."""

    def fetch(self, ref: DatasetRef, dest: Path) -> None:
        poses, intr, frames = _world(_SEED)
        if ref.held_out:
            (dest / "gt_poses.csv").write_text(_poses_csv(poses))
        else:
            (dest / "vo.json").write_text(json.dumps(
                {"intrinsics": list(intr), "frames": frames}))


# harness-owned grader: SE(3)-aligned ATE vs the hidden GT trajectory
_EVAL = '''\
import csv, json, os
import numpy as np
def load(p):
    out = []
    for r in csv.reader(open(p)):
        if len(r) < 12:
            continue
        v = [float(x) for x in r[:12]]
        T = np.eye(4); T[:3, :3] = np.array(v[:9]).reshape(3, 3); T[:3, 3] = v[9:12]
        out.append(T)
    return out
def proc(A, B):
    a = A.mean(0); b = B.mean(0)
    U, _, Vt = np.linalg.svd((A - a).T @ (B - b))
    D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ D @ U.T
    return R, b - R @ a
est = load(os.path.join(os.environ["LAB_ARTIFACTS"], "trajectory.csv"))
gt  = load(os.path.join(os.environ["LAB_DATA"], "gt_poses.csv"))
n = min(len(est), len(gt))
try:
    P = np.array([est[i][:3, 3] for i in range(n)])
    Q = np.array([gt[i][:3, 3] for i in range(n)])
    R, t = proc(P, Q); al = (R @ P.T).T + t
    ate = float(np.sqrt(((al - Q) ** 2).sum(1).mean()))
except Exception:
    ate = 9.99
json.dump({"ate": round(ate, 4)},
          open(os.path.join(os.environ["LAB_EVAL_OUT"], "heldout.json"), "w"))
'''


def slam_task() -> ImplementationTask:
    return ImplementationTask(
        description=(
            "Implement full visual SLAM and output a globally-consistent camera trajectory. "
            "Read $LAB_DATA/vo.json: 'intrinsics' = [fx, fy, cx, cy]; 'frames' is a list "
            "(one per time step, along a path that REVISITS places) of RGBD observations "
            "[landmark_id, u, v, depth] — a shared landmark_id is the same 3D point. Run a "
            "visual-odometry front-end (back-project, match tracks, estimate frame-to-frame "
            "SE(3) motion), DETECT LOOP CLOSURES (frames far apart in time that re-observe "
            "the same landmarks), and optimize a pose graph over both odometry and "
            "loop-closure constraints so accumulated drift is removed. Anchor frame 0 = "
            "identity. Write absolute camera-to-world poses to $LAB_ARTIFACTS/trajectory.csv: "
            "one pose per line as 12 comma-separated numbers — 3x3 rotation row-major (9) "
            "then translation (3), in frame order."),
        framework=FrameworkSpec("numpy", "2.3", "cpu"),
        entry_command="python3 $LAB_CODE/main.py",
        eval_command="python3 $LAB_CODE/eval.py", eval_code=_EVAL,
        metric="ate", op="<=", threshold=0.10,
        datasets=[DatasetRef("slam-frames", "synthetic"),
                  DatasetRef("slam-gt", "synthetic", held_out=True)],
        entry_filename="main.py")


# --- canned solvers (offline; numpy-only so they run on the host via local mode) ----------
def _writer(body: str):
    def author(task, code_dir: Path, rec) -> Usage:
        (code_dir / "main.py").write_text(body)
        return Usage(0, 0)
    return author


_HONEST_BODY = _SLAM_SRC + '''
import json, os
d = json.load(open(os.path.join(os.environ["LAB_DATA"], "vo.json")))
T = run_slam(d["intrinsics"], d["frames"])
open(os.path.join(os.environ["LAB_ARTIFACTS"], "trajectory.csv"), "w").write(
    "\\n".join(_pose_line(t) for t in T))
'''

_VO_ONLY_BODY = _SLAM_SRC + '''
import json, os
d = json.load(open(os.path.join(os.environ["LAB_DATA"], "vo.json")))
# DEGENERATE: Rung-1 visual odometry with NO loop closure and NO optimization. A fine
# algorithm that runs and emits a trajectory, but drift is never corrected on a looping path.
T = run_vo_only(d["intrinsics"], d["frames"])
open(os.path.join(os.environ["LAB_ARTIFACTS"], "trajectory.csv"), "w").write(
    "\\n".join(_pose_line(t) for t in T))
'''

HONEST = _writer(_HONEST_BODY)        # VO + loop closure + SE(3) optimization -> VERIFIED
VO_ONLY = _writer(_VO_ONLY_BODY)      # odometry only, no loop closure -> REJECTED
