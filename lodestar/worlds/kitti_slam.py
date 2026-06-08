"""Rung 8 — automotive SLAM on REAL driving data: the KITTI odometry benchmark.

The first OUTDOOR, car-mounted rung. KITTI (Geiger et al.) is the canonical visual-odometry /
SLAM benchmark for self-driving: a stereo camera on a car through real streets, with GPS/IMU
ground-truth trajectories. We hold the trajectory out and grade the same ORB VO solver on it.

KITTI ships *stereo*, not dense depth — so this rung adds one step: compute per-pixel depth from
the left/right pair with a stereo block matcher (cv2 StereoSGBM), depth = fx*baseline/disparity.
That depth + the left image + intrinsics are exactly the frames.npz contract every other rung
uses, so the unchanged solver runs here too. KITTI's camera frame is OpenCV optical (x-right,
y-down, z-forward) and its poses are camera-to-world with frame 0 = identity — matching the
solver, so no convention flip is needed.

Honest VO is VERIFIED on the hidden car trajectory; "the car never moved" is REJECTED.

Data: KITTI odometry (gray stereo + calib + poses), cached under ~/.cache/lodestar/kitti
(the gray zip is ~22 GB; a sequence's frames are extracted from it). Free, public."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

import cv2

from .._spine import DatasetRef, FrameworkSpec, ImplementationTask, Usage
from .image_slam import HONEST, STATIC, _EVAL, _pose_line, rpe, run_image_vo

_CACHE = Path(os.environ.get("LODESTAR_DATA", str(Path.home() / ".cache" / "lodestar"))) / "kitti"
_DSET = _CACHE / "dataset"
_DEFAULT_SEQ = "00"
_MAX_DEPTH = 50.0          # stereo depth is unreliable far away -> drop matches beyond this (metres)
_MIN_DISP = 1.0


def seq_dir(seq: str = _DEFAULT_SEQ) -> Path:
    return _DSET / "sequences" / seq


def is_available(seq: str = _DEFAULT_SEQ) -> bool:
    d = seq_dir(seq)
    return (d / "calib.txt").exists() and (_DSET / "poses" / f"{seq}.txt").exists() \
        and (d / "image_0").exists() and (d / "image_1").exists()


def _read_calib(seq: str):
    """Left-camera intrinsics (fx, fy, cx, cy) and the stereo baseline (m) from calib.txt.
    P0 = left gray projection [fx 0 cx 0; 0 fy cy 0; 0 0 1 0]; P1[0,3] = -fx*baseline."""
    vals = {}
    for line in (seq_dir(seq) / "calib.txt").read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            vals[k.strip()] = np.array([float(x) for x in v.split()]).reshape(3, 4)
    P0, P1 = vals["P0"], vals["P1"]
    fx, fy, cx, cy = P0[0, 0], P0[1, 1], P0[0, 2], P0[1, 2]
    baseline = -P1[0, 3] / fx
    return (fx, fy, cx, cy), baseline


def _load_poses(seq: str):
    out = []
    for line in (_DSET / "poses" / f"{seq}.txt").read_text().splitlines():
        v = [float(x) for x in line.split()]
        if len(v) < 12:
            continue
        T = np.eye(4); T[:3, :4] = np.array(v).reshape(3, 4)
        out.append(T)
    return out


def _stereo_depth(left, right, fx, baseline):
    """Dense metric depth from a rectified stereo pair via SGBM. Invalid/too-far -> 0."""
    bs = 5
    sgbm = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=128, blockSize=bs,
        P1=8 * bs * bs, P2=32 * bs * bs, disp12MaxDiff=1, uniquenessRatio=10,
        speckleWindowSize=100, speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    disp = sgbm.compute(left, right).astype(np.float32) / 16.0
    depth = np.zeros_like(disp)
    valid = disp > _MIN_DISP
    depth[valid] = fx * baseline / disp[valid]
    depth[(depth <= 0) | (depth > _MAX_DEPTH)] = 0.0
    return depth


def _world(seq: str = _DEFAULT_SEQ, stride: int = 1, n: int = 30, start: int = 0):
    """Load a strided KITTI sub-sequence: (poses, intr, rgb_gray, depth_metric)."""
    if not is_available(seq):
        raise RuntimeError(f"KITTI sequence {seq!r} not cached under {_DSET} (see ensure note)")
    intr, baseline = _read_calib(seq)
    fx = intr[0]
    poses_all = _load_poses(seq)
    d = seq_dir(seq)
    idxs = [start + k * stride for k in range(n)]
    rgb, depth, poses = [], [], []
    for i in idxs:
        lp = d / "image_0" / f"{i:06d}.png"
        rp = d / "image_1" / f"{i:06d}.png"
        if not lp.exists() or not rp.exists() or i >= len(poses_all):
            raise RuntimeError(f"KITTI frame {i} or its pose missing")
        left = cv2.imread(str(lp), cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(str(rp), cv2.IMREAD_GRAYSCALE)
        rgb.append(left)
        depth.append(_stereo_depth(left, right, fx, baseline))
        poses.append(poses_all[i])
    return poses, intr, np.stack(rgb), np.stack(depth)


def _poses_csv(poses) -> str:
    return "\n".join(_pose_line(T) for T in poses)


class KittiProvider:
    """inputs split: frames.npz (left image + stereo depth + intrinsics). held-out: gt_poses.csv.
    Same on-disk contract as the other rungs — pixels are real KITTI driving frames, depth is
    computed once from the stereo pair."""

    def __init__(self, **world_kwargs):
        self.world_kwargs = world_kwargs
        self._cache = None

    def _load_once(self):
        if self._cache is None:
            self._cache = _world(**self.world_kwargs)
        return self._cache

    def fetch(self, ref: DatasetRef, dest) -> None:
        dest = Path(dest)
        poses, intr, rgb, depth = self._load_once()
        if ref.held_out:
            (dest / "gt_poses.csv").write_text(_poses_csv(poses))
        else:
            np.savez_compressed(dest / "frames.npz", rgb=rgb,
                                depth=depth.astype(np.float16), intr=np.array(intr))


def kitti_task() -> ImplementationTask:
    return ImplementationTask(
        description=(
            "Implement visual odometry on real car-mounted driving frames. Read "
            "$LAB_DATA/frames.npz: 'rgb' is a stack of grayscale frames (F, H, W), 'depth' a "
            "matching per-pixel metric depth stack (metres, from stereo; 0 = invalid), 'intr' = "
            "[fx, fy, cx, cy]. NO feature tracks are given — detect and match features yourself "
            "(e.g. ORB + descriptor matching) between consecutive frames, read depth at each "
            "matched keypoint to back-project it to 3D (skip depth==0), estimate the frame-to-"
            "frame rigid motion (e.g. Procrustes/Umeyama SE(3)), and chain into ABSOLUTE camera-"
            "to-world poses with frame 0 = identity. Write the trajectory to "
            "$LAB_ARTIFACTS/trajectory.csv: one pose per line as 12 comma-separated numbers — "
            "3x3 rotation row-major (9) then translation (3), in frame order."),
        framework=FrameworkSpec("opencv", "4", "cpu"),
        entry_command="timeout 180 python3 $LAB_CODE/main.py",
        eval_command="python3 $LAB_CODE/eval.py", eval_code=_EVAL,
        metric="rpe", op="<=", threshold=0.5,
        datasets=[DatasetRef("kitti-frames", "kitti"),
                  DatasetRef("kitti-gt", "kitti", held_out=True)],
        entry_filename="main.py")
