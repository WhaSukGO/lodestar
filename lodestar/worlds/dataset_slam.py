"""Rung 6 — verify SLAM on REAL benchmark data: the ICL-NUIM living-room sequence.

The previous rungs render their own worlds. This one takes the opposite, most-robust route: a
real, widely-cited RGB-D SLAM benchmark (ICL-NUIM, Handa et al.) is the "world". It ships
photorealistic ray-traced frames with metric depth AND a perfect ground-truth camera
trajectory — exactly what Lodestar needs, with zero render risk. We hold the trajectory out and
grade the same ORB visual-odometry solver on it: honest VO tracks the real sequence (low RPE,
VERIFIED); "the camera never moved" drifts off (REJECTED).

Why this matters: it closes the loop from toy worlds to a dataset the SLAM community actually
benchmarks on. The verifier and solver are unchanged — only the world is now real data.

Data: ICL-NUIM `living_room_traj0` in TUM RGB-D PNG format (depth = uint16 / 5000 = metres,
planar z; poses = TUM `tx ty tz qx qy qz qw`, OpenCV optical-frame camera-to-world; the i-th
PNG pairs with the (i+1)-th pose line; ICL's camera uses fy<0). Auto-downloaded+cached to
~/.cache/lodestar/icl_nuim (~700 MB) on first use; frames are read straight out of the tarball.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

import cv2

from .._spine import DatasetRef, FrameworkSpec, ImplementationTask, Usage
from .image_slam import HONEST, STATIC, _pose_line, run_image_vo
# ICL-NUIM's camera moves very slowly (6.5 m over 1508 frames), so per-frame RPE can't tell
# honest VO from "never moved". ATE can: a static solver collapses to a point while GT spans the
# room. Reuse the SLAM rung's SE(3)-aligned ATE grader + metric.
from .visual_slam import _EVAL, ate

# ICL-NUIM living_room_traj0 (Handa et al., ICL-NUIM). Public, no signup.
_BASE = "https://www.doc.ic.ac.uk/~ahanda"
_ARCHIVE_URL = f"{_BASE}/living_room_traj0_frei_png.tar.gz"
_GT_URL = f"{_BASE}/VaFRIC/livingRoom0.gt.freiburg"
_CACHE = Path(os.environ.get("LODESTAR_DATA", str(Path.home() / ".cache" / "lodestar"))) / "icl_nuim"
_ARCHIVE = _CACHE / "living_room_traj0_frei_png.tar.gz"
_GT = _CACHE / "livingRoom0.gt.freiburg"
_MARKER = _CACHE / ".extracted"      # set once the archive's rgb/ + depth/ PNGs are on disk

# ICL-NUIM camera intrinsics (note the negative fy — ICL's image-y convention).
_INTR = (481.20, -480.00, 319.50, 239.50)
_POSE_OFFSET = 1            # PNG i pairs with the (i+1)-th GT line (GT is 1-indexed)
_DEPTH_SCALE = 5000.0       # TUM: depth_metres = uint16 / 5000   (planar z)


def is_available() -> bool:
    return _MARKER.exists() and _GT.exists()


def ensure_icl(timeout: int = 1800) -> Path:
    """Download + extract + cache the ICL-NUIM sequence on first use. Extracting once to disk
    (vs decompressing the 700 MB gzip on every read) makes frame access fast. Raises on failure."""
    _CACHE.mkdir(parents=True, exist_ok=True)
    import subprocess
    if not _GT.exists():
        subprocess.run(["curl", "-sS", "-L", "-m", str(timeout), "-o", str(_GT), _GT_URL], check=True)
    if not _ARCHIVE.exists():
        tmp = _ARCHIVE.with_suffix(".part")
        subprocess.run(["curl", "-sS", "-L", "-m", str(timeout), "-o", str(tmp), _ARCHIVE_URL], check=True)
        tmp.rename(_ARCHIVE)
    if not _MARKER.exists():
        subprocess.run(["tar", "xzf", str(_ARCHIVE)], cwd=str(_CACHE), check=True, timeout=timeout)
        _MARKER.touch()
    return _CACHE


def _quat_to_T(tx, ty, tz, qx, qy, qz, qw):
    n = np.linalg.norm([qx, qy, qz, qw]) or 1.0
    qx, qy, qz, qw = (v / n for v in (qx, qy, qz, qw))
    R = np.array([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                  [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                  [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = [tx, ty, tz]
    return T


def _load_poses():
    out = {}
    for line in _GT.read_text().splitlines():
        p = line.split()
        if len(p) >= 8:
            out[int(round(float(p[0])))] = _quat_to_T(*map(float, p[1:8]))
    return out


def _world(stride: int = 8, n: int = 24, start: int = 0):
    """Load a strided sub-sequence: (poses, intr, rgb_gray, depth_metric). Reads extracted PNGs
    from disk. `stride` > 1 gives more inter-frame parallax than the 30 fps native rate."""
    if not is_available():
        ensure_icl()
    poses_all = _load_poses()
    idxs = [start + k * stride for k in range(n)]
    rgb, depth, poses = [], [], []
    for i in idxs:
        rp, dp = _CACHE / "rgb" / f"{i}.png", _CACHE / "depth" / f"{i}.png"
        if not rp.exists() or not dp.exists() or (i + _POSE_OFFSET) not in poses_all:
            raise RuntimeError(f"ICL frame {i} or its pose is missing")
        rgb.append(cv2.cvtColor(cv2.imread(str(rp), cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY))
        depth.append(cv2.imread(str(dp), cv2.IMREAD_UNCHANGED).astype(np.float32) / _DEPTH_SCALE)
        poses.append(poses_all[i + _POSE_OFFSET])
    return poses, _INTR, np.stack(rgb), np.stack(depth)


def _poses_csv(poses) -> str:
    return "\n".join(_pose_line(T) for T in poses)


class IclNuimProvider:
    """inputs split: frames.npz (rgb + depth + intrinsics). held-out: gt_poses.csv.

    Same on-disk contract as the rendered rungs — the pixels are now real ICL-NUIM benchmark
    frames, so the same ORB solver and verifier grade SLAM on real data."""

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


def icl_task() -> ImplementationTask:
    return ImplementationTask(
        description=(
            "Implement image-based visual odometry on a real RGB-D sequence. Read "
            "$LAB_DATA/frames.npz: 'rgb' is a stack of grayscale frames (F, H, W), 'depth' a "
            "matching per-pixel metric depth stack (metres), 'intr' = [fx, fy, cx, cy] (note "
            "fy may be negative). NO feature tracks are given — detect and match features "
            "yourself (e.g. ORB + descriptor matching) between consecutive frames, read depth "
            "at each matched keypoint to back-project it to 3D, estimate the frame-to-frame "
            "rigid motion (e.g. Procrustes/Umeyama SE(3)), and chain into ABSOLUTE camera-to-"
            "world poses with frame 0 = identity. Write the trajectory to "
            "$LAB_ARTIFACTS/trajectory.csv: one pose per line as 12 comma-separated numbers — "
            "3x3 rotation row-major (9) then translation (3), in frame order."),
        framework=FrameworkSpec("opencv", "4", "cpu"),
        entry_command="timeout 180 python3 $LAB_CODE/main.py",
        eval_command="python3 $LAB_CODE/eval.py", eval_code=_EVAL,
        metric="ate", op="<=", threshold=0.05,
        datasets=[DatasetRef("icl-frames", "icl-nuim"),
                  DatasetRef("icl-gt", "icl-nuim", held_out=True)],
        entry_filename="main.py")
