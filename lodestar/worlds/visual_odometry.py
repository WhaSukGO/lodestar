"""Rung 1 — RGBD visual odometry: synthetic feature tracks, hidden-GT graded on RPE.

The SLAM front-end, still with NO renderer and NO ML. A camera flies through a cloud of 3D
landmarks; each frame yields RGBD feature observations [landmark_id, u, v, depth] (pixel +
metric depth, with noise). The same landmark_id across frames is a track. The solver must
recover the camera trajectory (absolute SE(3) poses, frame 0 = identity).

Metric scale (RGBD) → no monocular scale ambiguity, so the oracle is a direct **Relative
Pose Error (RPE)**: the standard VO metric, comparing each frame-to-frame motion to ground
truth. RPE measures local consistency and does not penalize the slow global drift inherent
to pure VO (loop closure — which removes drift — is Rung 2).

Honest VO back-projects pixels+depth to 3D and rigid-aligns matched points per frame pair
(Procrustes/Umeyama SE(3)), chaining the motions → low RPE → VERIFIED. A degenerate solver
that assumes the camera never moves (identity every frame) RUNS and emits a trajectory but
has high RPE → REJECTED. "It ran ≠ it's correct." Pure numpy, CPU, deterministic."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .._spine import DatasetRef, FrameworkSpec, ImplementationTask, Usage

_SEED = 0
_F = 30          # frames
_L = 240         # landmarks

# --- the VO front-end: ONE source, embedded in the honest solver's main.py and exec'd into
#     this module so tests reuse the exact same code. ----------------------------------------
_VO_SRC = '''\
import numpy as np

def _bp(obs, intr):
    """back-project RGBD observations [lid, u, v, depth] -> {lid: 3D point in camera frame}."""
    fx, fy, cx, cy = intr
    return {int(o[0]): np.array([(o[1] - cx) / fx * o[3], (o[2] - cy) / fy * o[3], o[3]])
            for o in obs}

def _proc(A, B):
    """rigid SE(3) fit with B ~ R A + t (Umeyama, no scale)."""
    a = A.mean(0); b = B.mean(0)
    U, _, Vt = np.linalg.svd((A - a).T @ (B - b))
    D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ D @ U.T
    return R, b - R @ a

def run_vo(intrinsics, frames):
    """RGBD visual odometry: per consecutive frame, rigid-align matched back-projected 3D
    points; chain frame-to-frame motions into absolute poses. T[0] = identity."""
    T = [np.eye(4)]; prev = np.eye(4)
    for i in range(len(frames) - 1):
        di = _bp(frames[i], intrinsics); dj = _bp(frames[i + 1], intrinsics)
        common = sorted(set(di) & set(dj))
        if len(common) < 3:
            Tij = prev                      # constant-velocity fallback for thin overlap
        else:
            Xi = np.array([di[k] for k in common]); Xj = np.array([dj[k] for k in common])
            R, t = _proc(Xj, Xi)            # Xi ~ R Xj + t  => relative pose i<-j
            Tij = np.eye(4); Tij[:3, :3] = R; Tij[:3, 3] = t
        prev = Tij; T.append(T[-1] @ Tij)
    return T

def _pose_line(T):
    return ",".join(repr(float(v)) for v in list(T[:3, :3].flatten()) + list(T[:3, 3]))
'''
exec(_VO_SRC, globals())   # -> _bp, _proc, run_vo, _pose_line (numpy as np)


def _rot(ax, ay, az):
    cx, sx = np.cos(ax), np.sin(ax); cy, sy = np.cos(ay), np.sin(ay); cz, sz = np.cos(az), np.sin(az)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _world(seed: int = 0, F: int = _F, L: int = _L,
           px_sigma: float = 0.7, depth_sigma: float = 0.01, depth_max: float = 9.0):
    """Deterministic world: GT camera poses, intrinsics, per-frame RGBD observations.

    Knobs (for selectable environments): `L` landmark count (fewer → thinner matches),
    `px_sigma` pixel noise, `depth_sigma` relative depth noise, `depth_max` visibility range."""
    rng = np.random.default_rng(seed)
    fx = fy = 320.0; cx = cy = 240.0; W = Hh = 480
    poses = []
    for f in range(F):                                   # fly forward +z, gentle sway + yaw
        t = np.array([0.7 * np.sin(f * 0.15), 0.25 * np.sin(f * 0.1), 0.35 * f])
        R = _rot(0.0, 0.05 * np.sin(f * 0.2), 0.02 * np.sin(f * 0.1))
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t; poses.append(T)
    M = np.c_[rng.uniform(-7, 7, L), rng.uniform(-6, 6, L), rng.uniform(-2, 13, L)]
    intr = (fx, fy, cx, cy)
    frames = []
    for T in poses:
        R, tt = T[:3, :3], T[:3, 3]
        Xc = (R.T @ (M - tt).T).T                         # landmarks in camera frame
        out = []
        for lid, X in enumerate(Xc):
            Z = X[2]
            if Z < 0.5 or Z > depth_max:                  # in front, within depth range
                continue
            u = fx * X[0] / Z + cx; v = fy * X[1] / Z + cy
            if 0 <= u < W and 0 <= v < Hh:
                out.append([int(lid), float(u + rng.normal(0, px_sigma)),
                            float(v + rng.normal(0, px_sigma)),
                            float(Z * (1 + rng.normal(0, depth_sigma)))])
        frames.append(out)
    return poses, intr, frames


def rpe(est_poses, gt_poses) -> float:
    """Relative Pose Error (translational RMSE over consecutive frame pairs)."""
    n = min(len(est_poses), len(gt_poses))
    errs = []
    for i in range(n - 1):
        rg = np.linalg.inv(gt_poses[i]) @ gt_poses[i + 1]
        re = np.linalg.inv(est_poses[i]) @ est_poses[i + 1]
        errs.append(float(np.linalg.norm((np.linalg.inv(rg) @ re)[:3, 3])))
    return float(np.sqrt(np.mean(np.square(errs)))) if errs else 9.99


def _poses_csv(poses) -> str:
    return "\n".join(_pose_line(T) for T in poses)


class VisualOdometryProvider:
    """inputs split: vo.json (intrinsics + per-frame RGBD obs). held-out: gt_poses.csv.
    Pass `_world` knobs to select an environment, e.g. VisualOdometryProvider(px_sigma=1.5)."""

    def __init__(self, **world_kwargs):
        self.world_kwargs = world_kwargs

    def fetch(self, ref: DatasetRef, dest: Path) -> None:
        poses, intr, frames = _world(**self.world_kwargs)
        if ref.held_out:
            (dest / "gt_poses.csv").write_text(_poses_csv(poses))
        else:
            (dest / "vo.json").write_text(json.dumps(
                {"intrinsics": list(intr), "frames": frames}))


# harness-owned grader: translational RPE of the estimated vs the hidden GT trajectory
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
est = load(os.path.join(os.environ["LAB_ARTIFACTS"], "trajectory.csv"))
gt  = load(os.path.join(os.environ["LAB_DATA"], "gt_poses.csv"))
n = min(len(est), len(gt))
try:
    errs = []
    for i in range(n - 1):
        rg = np.linalg.inv(gt[i]) @ gt[i + 1]
        re = np.linalg.inv(est[i]) @ est[i + 1]
        errs.append(float(np.linalg.norm((np.linalg.inv(rg) @ re)[:3, 3])))
    rpe = float(np.sqrt(np.mean(np.square(errs)))) if errs else 9.99
except Exception:
    rpe = 9.99
json.dump({"rpe": round(rpe, 4)},
          open(os.path.join(os.environ["LAB_EVAL_OUT"], "heldout.json"), "w"))
'''


def vo_task() -> ImplementationTask:
    return ImplementationTask(
        description=(
            "Implement RGBD visual odometry: estimate the camera trajectory from feature "
            "tracks. Read $LAB_DATA/vo.json: 'intrinsics' = [fx, fy, cx, cy]; 'frames' is a "
            "list (one per time step) of RGBD observations [landmark_id, u, v, depth] — the "
            "SAME landmark_id across frames is the same 3D point (a track). Back-project "
            "pixels+depth to 3D, match tracks between consecutive frames, estimate each "
            "frame-to-frame rigid motion (e.g. Procrustes/Umeyama SE(3)), and chain them "
            "into ABSOLUTE camera-to-world poses with frame 0 = identity. Write the "
            "trajectory to $LAB_ARTIFACTS/trajectory.csv: one pose per line as 12 "
            "comma-separated numbers — the 3x3 rotation row-major (9) then translation (3), "
            "in frame order."),
        framework=FrameworkSpec("numpy", "2.3", "cpu"),
        entry_command="timeout 120 python3 $LAB_CODE/main.py",
        eval_command="python3 $LAB_CODE/eval.py", eval_code=_EVAL,
        metric="rpe", op="<=", threshold=0.15,
        datasets=[DatasetRef("vo-frames", "synthetic"),
                  DatasetRef("vo-gt", "synthetic", held_out=True)],
        entry_filename="main.py")


# --- canned solvers (offline; numpy-only so they run on the host via local mode) ----------
def _writer(body: str):
    def author(task, code_dir: Path, rec) -> Usage:
        (code_dir / "main.py").write_text(body)
        return Usage(0, 0)
    return author


_HONEST_BODY = _VO_SRC + '''
import json, os
d = json.load(open(os.path.join(os.environ["LAB_DATA"], "vo.json")))
T = run_vo(d["intrinsics"], d["frames"])
open(os.path.join(os.environ["LAB_ARTIFACTS"], "trajectory.csv"), "w").write(
    "\\n".join(_pose_line(t) for t in T))
'''

_STATIC_BODY = '''\
import json, os
import numpy as np
d = json.load(open(os.path.join(os.environ["LAB_DATA"], "vo.json")))
# DEGENERATE: assume the camera never moves -> identity pose for every frame. It runs and
# emits a full trajectory, but every frame-to-frame motion is wrong.
I = np.eye(4)
line = ",".join(repr(float(v)) for v in list(I[:3, :3].flatten()) + list(I[:3, 3]))
open(os.path.join(os.environ["LAB_ARTIFACTS"], "trajectory.csv"), "w").write(
    "\\n".join(line for _ in d["frames"]))
'''

HONEST = _writer(_HONEST_BODY)        # real RGBD VO (Procrustes per frame pair) -> VERIFIED
STATIC = _writer(_STATIC_BODY)        # "camera never moved" -> REJECTED
