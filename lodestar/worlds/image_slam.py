"""Rung 3 — image-based visual odometry: rendered RGBD frames, features detected from pixels.

The first rung where the solver does NOT receive pre-extracted feature tracks. It gets actual
rendered frames (grayscale + depth) and must do the perception itself: detect keypoints,
describe them, and MATCH across frames by appearance (real data association — no landmark
IDs are given). The roadmap's "renderer" is kept deliberately minimal here — a procedural
patch-splat rasterizer (each 3D landmark is a distinct high-contrast texture patch, scaled by
1/depth, painted back-to-front) rather than Habitat/Blender — but it produces images with
genuine, matchable ORB features, which is the point of this rung.

Honest solver: ORB detect + BFMatcher (Hamming, ratio test) between consecutive frames,
back-project matched keypoints with depth, robustly fit SE(3) (Procrustes + residual trim),
chain. Degenerate: assume the camera never moved. Graded on translational RPE vs the hidden
GT trajectory. Requires OpenCV (cv2); CPU-only, deterministic."""
from __future__ import annotations

import numpy as np

import cv2

from .._spine import DatasetRef, FrameworkSpec, ImplementationTask, Usage

_SEED = 0
_F = 24          # frames
_L = 90          # landmark patches
_IMG = 480
_DMAX = 13.0     # max landmark depth (visibility)
_PSCALE = 900.0  # on-image patch size ~ _PSCALE / depth

# --- the cv2 image-VO front-end: ONE source, embedded in the honest solver's main.py and
#     exec'd into this module so tests reuse the exact same code. ----------------------------
_VO_SRC = '''\
import numpy as np
import cv2

def _proc(A, B):
    a, b = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - a).T @ (B - b))
    D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ D @ U.T
    return R, b - R @ a

def run_image_vo(rgb, depth, intr):
    """Detect+match ORB features across frames, back-project with depth, chain SE(3) motion."""
    fx, fy, cx, cy = [float(v) for v in intr]
    h, w = rgb.shape[1], rgb.shape[2]
    orb = cv2.ORB_create(nfeatures=1500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    feats = [orb.detectAndCompute(im, None) for im in rgb]

    def bp(u, v, z):
        return np.array([(u - cx) / fx * z, (v - cy) / fy * z, z])

    def depth_at(dmap, u, v):
        return float(dmap[min(h - 1, max(0, int(round(v)))), min(w - 1, max(0, int(round(u))))])

    T = [np.eye(4)]; prev = np.eye(4)
    for i in range(len(rgb) - 1):
        (k1, d1), (k2, d2) = feats[i], feats[i + 1]
        if d1 is None or d2 is None or len(k1) < 6 or len(k2) < 6:
            T.append(T[-1] @ prev); continue
        P1, P2 = [], []
        for pair in bf.knnMatch(d1, d2, k=2):
            if len(pair) < 2:
                continue
            a, b = pair
            if a.distance < 0.75 * b.distance:                  # Lowe ratio test
                u1, v1 = k1[a.queryIdx].pt; u2, v2 = k2[a.trainIdx].pt
                z1 = depth_at(depth[i], u1, v1); z2 = depth_at(depth[i + 1], u2, v2)
                if z1 > 0 and z2 > 0:
                    P1.append(bp(u1, v1, z1)); P2.append(bp(u2, v2, z2))
        if len(P1) < 6:
            T.append(T[-1] @ prev); continue
        P1, P2 = np.array(P1), np.array(P2)
        R, t = _proc(P2, P1)
        for _ in range(2):                                      # robust: trim outlier matches
            res = np.linalg.norm((R @ P2.T).T + t - P1, axis=1)
            keep = res < np.median(res) * 2 + 0.05
            if keep.sum() < 6:
                break
            R, t = _proc(P2[keep], P1[keep])
        Tij = np.eye(4); Tij[:3, :3] = R; Tij[:3, 3] = t
        prev = Tij; T.append(T[-1] @ Tij)
    return T

def _pose_line(T):
    return ",".join(repr(float(v)) for v in list(T[:3, :3].flatten()) + list(T[:3, 3]))
'''
exec(_VO_SRC, globals())   # -> _proc, run_image_vo, _pose_line (numpy as np, cv2)


def _rot(ax, ay, az):
    cx, sx = np.cos(ax), np.sin(ax); cy, sy = np.cos(ay), np.sin(ay); cz, sz = np.cos(az), np.sin(az)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _render(M, patches, T, intr):
    fx, fy, cx, cy = intr
    img = np.zeros((_IMG, _IMG), np.uint8); depth = np.zeros((_IMG, _IMG), np.float32)
    R, tt = T[:3, :3], T[:3, 3]
    Xc = (R.T @ (M - tt).T).T
    for lid in np.argsort(-Xc[:, 2]):                    # painter's: far first
        X = Xc[lid]; Z = X[2]
        if Z < 0.5 or Z > _DMAX:
            continue
        u = fx * X[0] / Z + cx; v = fy * X[1] / Z + cy
        s = int(np.clip(_PSCALE / Z, 10, 46))
        p = cv2.resize(patches[lid], (s, s), interpolation=cv2.INTER_NEAREST)
        u0, v0 = int(u - s / 2), int(v - s / 2)
        y1, y2 = max(0, v0), min(_IMG, v0 + s); x1, x2 = max(0, u0), min(_IMG, u0 + s)
        if y1 >= y2 or x1 >= x2:
            continue
        img[y1:y2, x1:x2] = p[y1 - v0:y1 - v0 + (y2 - y1), x1 - u0:x1 - u0 + (x2 - x1)]
        depth[y1:y2, x1:x2] = Z
    return img, depth


def _world(seed: int = 0, F: int = _F, L: int = _L):
    """Deterministic world: GT poses, intrinsics, and rendered (grayscale, depth) frames."""
    rng = np.random.default_rng(seed)
    fx = fy = 320.0; cx = cy = 240.0
    poses = []
    for f in range(F):                                   # forward fly with gentle sway/yaw
        t = np.array([0.6 * np.sin(f * 0.12), 0.2 * np.sin(f * 0.1), 0.3 * f])
        R = _rot(0.0, 0.04 * np.sin(f * 0.2), 0.02 * np.sin(f * 0.1))
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t; poses.append(T)
    M = np.c_[rng.uniform(-6, 6, L), rng.uniform(-5, 5, L), rng.uniform(1, 11, L)]
    patches = [(rng.integers(0, 2, (28, 28)) * 255).astype(np.uint8) for _ in range(L)]
    intr = (fx, fy, cx, cy)
    rendered = [_render(M, patches, T, intr) for T in poses]
    rgb = np.stack([r[0] for r in rendered]); depth = np.stack([r[1] for r in rendered])
    return poses, intr, rgb, depth


def rpe(est_poses, gt_poses) -> float:
    n = min(len(est_poses), len(gt_poses))
    errs = []
    for i in range(n - 1):
        rg = np.linalg.inv(gt_poses[i]) @ gt_poses[i + 1]
        re = np.linalg.inv(est_poses[i]) @ est_poses[i + 1]
        errs.append(float(np.linalg.norm((np.linalg.inv(rg) @ re)[:3, 3])))
    return float(np.sqrt(np.mean(np.square(errs)))) if errs else 9.99


def _poses_csv(poses) -> str:
    return "\n".join(_pose_line(T) for T in poses)


class ImageSlamProvider:
    """inputs split: frames.npz (rgb + depth + intrinsics). held-out: gt_poses.csv."""

    def __init__(self, **world_kwargs):
        self.world_kwargs = world_kwargs

    def fetch(self, ref: DatasetRef, dest) -> None:
        from pathlib import Path
        dest = Path(dest)
        poses, intr, rgb, depth = _world(**self.world_kwargs)
        if ref.held_out:
            (dest / "gt_poses.csv").write_text(_poses_csv(poses))
        else:
            np.savez_compressed(dest / "frames.npz", rgb=rgb,
                                depth=depth.astype(np.float16), intr=np.array(intr))


# harness-owned grader: translational RPE vs the hidden GT trajectory (12-number pose rows)
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


def image_slam_task() -> ImplementationTask:
    return ImplementationTask(
        description=(
            "Implement image-based visual odometry. Read $LAB_DATA/frames.npz: 'rgb' is a "
            "stack of grayscale frames (F, H, W), 'depth' a matching per-pixel depth stack, "
            "'intr' = [fx, fy, cx, cy]. NO feature tracks are given — detect and match "
            "features yourself (e.g. ORB + descriptor matching) between consecutive frames, "
            "read depth at each matched keypoint to back-project it to 3D, estimate the "
            "frame-to-frame rigid motion (e.g. Procrustes/Umeyama SE(3)), and chain into "
            "ABSOLUTE camera-to-world poses with frame 0 = identity. Write the trajectory to "
            "$LAB_ARTIFACTS/trajectory.csv: one pose per line as 12 comma-separated numbers — "
            "3x3 rotation row-major (9) then translation (3), in frame order."),
        framework=FrameworkSpec("opencv", "4", "cpu"),
        entry_command="timeout 180 python3 $LAB_CODE/main.py",
        eval_command="python3 $LAB_CODE/eval.py", eval_code=_EVAL,
        metric="rpe", op="<=", threshold=0.05,
        datasets=[DatasetRef("imgslam-frames", "synthetic"),
                  DatasetRef("imgslam-gt", "synthetic", held_out=True)],
        entry_filename="main.py")


# --- canned solvers (offline; numpy + cv2 so they run on the host via local mode) ----------
def _writer(body: str):
    def author(task, code_dir, rec) -> Usage:
        from pathlib import Path
        (Path(code_dir) / "main.py").write_text(body)
        return Usage(0, 0)
    return author


_HONEST_BODY = _VO_SRC + '''
import os
d = np.load(os.path.join(os.environ["LAB_DATA"], "frames.npz"))
T = run_image_vo(d["rgb"], d["depth"].astype(np.float32), d["intr"])
open(os.path.join(os.environ["LAB_ARTIFACTS"], "trajectory.csv"), "w").write(
    "\\n".join(_pose_line(t) for t in T))
'''

_STATIC_BODY = '''\
import os
import numpy as np
d = np.load(os.path.join(os.environ["LAB_DATA"], "frames.npz"))
# DEGENERATE: assume the camera never moved -> identity pose for every frame.
I = np.eye(4)
line = ",".join(repr(float(v)) for v in list(I[:3, :3].flatten()) + list(I[:3, 3]))
open(os.path.join(os.environ["LAB_ARTIFACTS"], "trajectory.csv"), "w").write(
    "\\n".join(line for _ in range(len(d["rgb"]))))
'''

HONEST = _writer(_HONEST_BODY)        # ORB detect+match -> back-project -> Procrustes -> VERIFIED
STATIC = _writer(_STATIC_BODY)        # "camera never moved" -> REJECTED
