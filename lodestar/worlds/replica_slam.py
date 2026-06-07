"""Rung 7 — image VO on a REAL SCANNED apartment: a Replica scene rendered with BlenderProc.

Rung 5 path-traces a room *I* built; this rung renders a real, laser-scanned indoor environment
from the **Replica** dataset (Straub et al., FAIR) — millions of vertex-colored triangles of an
actual apartment/office — with Blender's Cycles (CPU/CUDA, no EGL). It's the "open world builder"
answer: a professionally captured 3D world, not a hand-authored one, plugged into the exact same
render+grade path as Rung 5.

Same two-layer payoff: identical frames.npz contract, unchanged ORB solver. Honest VO is
VERIFIED on the hidden camera trajectory; "camera never moved" is REJECTED.

Rendering is host-side and out-of-process (`blenderproc run _replica_render.py`). Needs the
Replica dataset cached under ~/.cache/lodestar/replica (see ensure note) and blenderproc."""
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

_SCRIPT = Path(__file__).with_name("_replica_render.py")
_CACHE = Path(os.environ.get("LODESTAR_DATA", str(Path.home() / ".cache" / "lodestar"))) / "replica"
_DEFAULT_SCENE = "room_0"
_RENDER_TIMEOUT = 1500


def blenderproc_available() -> bool:
    return shutil.which("blenderproc") is not None


def scene_path(scene: str = _DEFAULT_SCENE) -> Path:
    return _CACHE / scene / "mesh.ply"


def is_available(scene: str = _DEFAULT_SCENE) -> bool:
    return scene_path(scene).exists()


def _load_csv_poses(path):
    out = []
    for r in csv.reader(open(path)):
        if len(r) < 12:
            continue
        v = [float(x) for x in r[:12]]
        T = np.eye(4); T[:3, :3] = np.array(v[:9]).reshape(3, 3); T[:3, 3] = v[9:12]
        out.append(T)
    return out


def render_world(scene: str = _DEFAULT_SCENE, seed: int = 0, frames: int = 24, res: int = 480,
                 samples: int = 48, outdir: str | None = None):
    """Render a Replica scene; return (poses, intr, rgb_gray, depth, outdir)."""
    if not blenderproc_available():
        raise RuntimeError("blenderproc not found on PATH (pip install blenderproc)")
    if not is_available(scene):
        raise RuntimeError(f"Replica scene {scene!r} not cached at {scene_path(scene)}")
    outdir = outdir or tempfile.mkdtemp(prefix="replica-slam-")
    cmd = ["blenderproc", "run", str(_SCRIPT), "--", "--data_path", str(_CACHE), "--scene", scene,
           "--output", outdir, "--frames", str(frames), "--seed", str(seed),
           "--res", str(res), "--samples", str(samples)]
    subprocess.run(cmd, check=True, timeout=_RENDER_TIMEOUT, capture_output=True, text=True)
    d = np.load(os.path.join(outdir, "frames.npz"))
    poses = _load_csv_poses(os.path.join(outdir, "gt_poses.csv"))
    return poses, tuple(d["intr"]), d["rgb"], d["depth"].astype(np.float32), outdir


def _world(scene: str = _DEFAULT_SCENE, seed: int = 0, frames: int = 24, **kw):
    poses, intr, rgb, depth, _ = render_world(scene=scene, seed=seed, frames=frames, **kw)
    return poses, intr, rgb, depth


class ReplicaSlamProvider:
    """inputs split: frames.npz (rgb + depth + intrinsics). held-out: gt_poses.csv.
    Renders the real scanned scene ONCE (memoized) and serves both dataset splits from it."""

    def __init__(self, **world_kwargs):
        self.world_kwargs = world_kwargs
        self._dir = None

    def _render_once(self):
        if self._dir is None:
            *_, self._dir = render_world(outdir=tempfile.mkdtemp(prefix="replica-slam-"),
                                         **self.world_kwargs)
        return self._dir

    def fetch(self, ref: DatasetRef, dest) -> None:
        src = self._render_once()
        dest = Path(dest)
        if ref.held_out:
            shutil.copy(os.path.join(src, "gt_poses.csv"), dest / "gt_poses.csv")
        else:
            shutil.copy(os.path.join(src, "frames.npz"), dest / "frames.npz")


def replica_task() -> ImplementationTask:
    return ImplementationTask(
        description=(
            "Implement image-based visual odometry on frames rendered from a real scanned 3D "
            "scene. Read $LAB_DATA/frames.npz: 'rgb' is a stack of grayscale frames (F, H, W), "
            "'depth' a matching per-pixel metric depth stack, 'intr' = [fx, fy, cx, cy]. NO "
            "feature tracks are given — detect and match features yourself (e.g. ORB + "
            "descriptor matching) between consecutive frames, read depth at each matched "
            "keypoint to back-project it to 3D, estimate the frame-to-frame rigid motion (e.g. "
            "Procrustes/Umeyama SE(3)), and chain into ABSOLUTE camera-to-world poses with "
            "frame 0 = identity. Write the trajectory to $LAB_ARTIFACTS/trajectory.csv: one "
            "pose per line as 12 comma-separated numbers — 3x3 rotation row-major (9) then "
            "translation (3), in frame order."),
        framework=FrameworkSpec("opencv", "4", "cpu"),
        entry_command="timeout 180 python3 $LAB_CODE/main.py",
        eval_command="python3 $LAB_CODE/eval.py", eval_code=_EVAL,
        metric="rpe", op="<=", threshold=0.05,
        datasets=[DatasetRef("replica-frames", "replica"),
                  DatasetRef("replica-gt", "replica", held_out=True)],
        entry_filename="main.py")
