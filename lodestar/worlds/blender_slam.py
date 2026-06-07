"""Rung 5 — image VO on a PHOTOREALISTIC world: a path-traced room rendered with BlenderProc.

Rung 4 used a software GL rasterizer (flat shading, hard edges). Rung 5 renders the world with
Blender's Cycles **path tracer** (via BlenderProc): real global illumination, soft shadows, and
inter-reflections on textured PBR surfaces. The pixels carry the lighting cues a real camera
sees — a markedly more realistic image than the rasterized rung. Cycles renders on CPU or via
CUDA-compute, so it needs NO OpenGL/EGL display (the constraint that blocks GPU pyrender here).

Same two-layer payoff: the world emits the identical frames.npz contract, so the UNCHANGED ORB
detect+match+back-project+Procrustes solver is graded on a photorealistic render. Honest VO is
VERIFIED (low RPE); "camera never moved" is REJECTED.

Rendering is host-side and out-of-process: `BlenderSlamProvider.fetch` shells out to
`blenderproc run lodestar/worlds/_blender_render.py`, which builds the scene + camera trajectory
and writes frames.npz + gt_poses.csv. The solver sandbox still needs only numpy + cv2; only
world *generation* needs BlenderProc (`pip install blenderproc`; it ships its own Blender)."""
from __future__ import annotations

import csv
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .._spine import DatasetRef, FrameworkSpec, ImplementationTask, Usage
from .image_slam import HONEST, STATIC, _EVAL, rpe, run_image_vo

_SCRIPT = Path(__file__).with_name("_blender_render.py")
_RENDER_TIMEOUT = 1500


def blenderproc_available() -> bool:
    return shutil.which("blenderproc") is not None


def _load_csv_poses(path):
    out = []
    for r in csv.reader(open(path)):
        if len(r) < 12:
            continue
        v = [float(x) for x in r[:12]]
        T = np.eye(4); T[:3, :3] = np.array(v[:9]).reshape(3, 3); T[:3, 3] = v[9:12]
        out.append(T)
    return out


def render_world(seed: int = 0, frames: int = 24, res: int = 480, samples: int = 48,
                 outdir: str | None = None):
    """Run BlenderProc to render the world; return (poses, intr, rgb_gray, depth, outdir).

    Writes frames.npz + gt_poses.csv (+ preview_color.png) into `outdir` (a temp dir if None)."""
    if not blenderproc_available():
        raise RuntimeError("blenderproc not found on PATH (pip install blenderproc)")
    outdir = outdir or tempfile.mkdtemp(prefix="blender-slam-")
    cmd = ["blenderproc", "run", str(_SCRIPT), "--", "--output", outdir,
           "--frames", str(frames), "--seed", str(seed), "--res", str(res), "--samples", str(samples)]
    subprocess.run(cmd, check=True, timeout=_RENDER_TIMEOUT, capture_output=True, text=True)
    d = np.load(os.path.join(outdir, "frames.npz"))
    poses = _load_csv_poses(os.path.join(outdir, "gt_poses.csv"))
    return poses, tuple(d["intr"]), d["rgb"], d["depth"].astype(np.float32), outdir


def _world(seed: int = 0, frames: int = 24, **kw):
    """Convenience for demo/viz/tests: (poses, intr, rgb, depth)."""
    poses, intr, rgb, depth, _ = render_world(seed=seed, frames=frames, **kw)
    return poses, intr, rgb, depth


class BlenderSlamProvider:
    """inputs split: frames.npz (rgb + depth + intrinsics). held-out: gt_poses.csv.

    Renders ONCE (memoized) with BlenderProc and serves both dataset splits from that render."""

    def __init__(self, **world_kwargs):
        self.world_kwargs = world_kwargs
        self._dir = None

    def _render_once(self):
        if self._dir is None:
            *_, self._dir = render_world(outdir=tempfile.mkdtemp(prefix="blender-slam-"),
                                         **self.world_kwargs)
        return self._dir

    def fetch(self, ref: DatasetRef, dest) -> None:
        src = self._render_once()
        dest = Path(dest)
        if ref.held_out:
            shutil.copy(os.path.join(src, "gt_poses.csv"), dest / "gt_poses.csv")
        else:
            shutil.copy(os.path.join(src, "frames.npz"), dest / "frames.npz")


def blender_task() -> ImplementationTask:
    return ImplementationTask(
        description=(
            "Implement image-based visual odometry on photorealistic rendered frames. Read "
            "$LAB_DATA/frames.npz: 'rgb' is a stack of grayscale frames (F, H, W), 'depth' a "
            "matching per-pixel metric depth stack, 'intr' = [fx, fy, cx, cy]. NO feature "
            "tracks are given — detect and match features yourself (e.g. ORB + descriptor "
            "matching) between consecutive frames, read depth at each matched keypoint to back-"
            "project it to 3D, estimate the frame-to-frame rigid motion (e.g. Procrustes/"
            "Umeyama SE(3)), and chain into ABSOLUTE camera-to-world poses with frame 0 = "
            "identity. Write the trajectory to $LAB_ARTIFACTS/trajectory.csv: one pose per "
            "line as 12 comma-separated numbers — 3x3 rotation row-major (9) then translation "
            "(3), in frame order."),
        framework=FrameworkSpec("opencv", "4", "cpu"),
        entry_command="timeout 180 python3 $LAB_CODE/main.py",
        eval_command="python3 $LAB_CODE/eval.py", eval_code=_EVAL,
        metric="rpe", op="<=", threshold=0.05,
        datasets=[DatasetRef("blender-frames", "synthetic"),
                  DatasetRef("blender-gt", "synthetic", held_out=True)],
        entry_filename="main.py")
