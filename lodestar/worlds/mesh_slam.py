"""Rung 4 — image VO on an ACTUAL 3D world: a real triangle-mesh scene, rendered offscreen.

Rung 3 faked the renderer (camera-facing patch-splats — flat billboards that never occlude or
perspective-warp). This rung renders a genuine 3D scene with a real GL rasterizer (pyrender on
OSMesa, software, CPU-only): a textured room with boxes standing inside it. Now near surfaces
truly OCCLUDE far ones (a box hides the wall behind it), textures FORESHORTEN with viewing
angle, and features appear/disappear as the camera moves — i.e. real, hard data association.

The payoff for the two-layer thesis: the *solver is unchanged*. Rung 4 emits the exact same
`frames.npz` contract as Rung 3 (grayscale `rgb` + metric `depth` + `intr`), so the very same
ORB-detect+match+back-project+Procrustes front-end (`image_slam.run_image_vo`) is graded here
on a real 3D world — and a "camera never moved" solver is still REJECTED. Same verifier, same
solver, harder world. Graded on translational RPE vs the hidden GT trajectory.

Rendering is host-side (it happens in the dataset Provider, not in the sandbox): only world
*generation* needs pyrender; the solver sandbox still needs nothing but numpy + cv2.

Deps beyond the offline rungs: pyrender + trimesh + an OSMesa software-GL lib (see README).
Deterministic and CPU-only."""
from __future__ import annotations

import os

import numpy as np

# A real GL context with no GPU/display: OSMesa software rasterizer. Must be set before pyrender
# imports PyOpenGL. EGL is unavailable headless on WSL (no enumerable device); OSMesa needs none.
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import trimesh                       # noqa: E402
import pyrender                      # noqa: E402
from pyrender.constants import RenderFlags   # noqa: E402
from PIL import Image               # noqa: E402

from .._spine import DatasetRef, FrameworkSpec, ImplementationTask, Usage
# Reuse Rung 3 verbatim: identical frames.npz contract -> identical solver, grader, and metric.
from .image_slam import HONEST, STATIC, _EVAL, _pose_line, rpe, run_image_vo

_SEED = 0
_F = 24            # frames
_IMG = 480
_F_PIX = 400.0     # focal length in pixels -> intrinsics fx = fy = _F_PIX, cx = cy = _IMG/2

# OpenGL camera (looks down -z, +y up) <- our CV camera (looks down +z, +y down). Fixed basis
# change so a CV camera-to-world pose renders the right view; depth then reads as +z metric.
_CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])


def _rot(ax, ay, az):
    cx, sx = np.cos(ax), np.sin(ax); cy, sy = np.cos(ay), np.sin(ay); cz, sz = np.cos(az), np.sin(az)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _tex(rng, blocks=16, px=16):
    """A crisp high-contrast block texture: lots of stable corners for ORB to detect+match."""
    small = rng.integers(0, 256, (blocks, blocks, 3), dtype=np.uint8)
    return np.repeat(np.repeat(small, px, 0), px, 1)


def _quad(corners, tex):
    """A textured rectangle (two triangles) from 4 world-space corners, UV-mapped to `tex`."""
    V = np.asarray(corners, np.float64)
    F = np.array([[0, 1, 2], [0, 2, 3]])
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float64)
    m = trimesh.Trimesh(vertices=V, faces=F, process=False)
    m.visual = trimesh.visual.TextureVisuals(uv=uv, image=Image.fromarray(tex))
    return m


def _box(cx, cy, cz, sx, sy, sz, rng):
    """An axis-aligned textured box (6 quads) centred at (cx,cy,cz) with half-extents s*."""
    x0, x1 = cx - sx, cx + sx; y0, y1 = cy - sy, cy + sy; z0, z1 = cz - sz, cz + sz
    faces = [
        [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],   # -z
        [(x1, y0, z1), (x0, y0, z1), (x0, y1, z1), (x1, y1, z1)],   # +z
        [(x0, y0, z1), (x0, y0, z0), (x0, y1, z0), (x0, y1, z1)],   # -x
        [(x1, y0, z0), (x1, y0, z1), (x1, y1, z1), (x1, y1, z0)],   # +x
        [(x0, y0, z1), (x1, y0, z1), (x1, y0, z0), (x0, y0, z0)],   # -y
        [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],   # +y
    ]
    return [_quad(f, _tex(rng)) for f in faces]


def _scene(rng):
    """A textured room (floor/ceiling/back/side walls) with boxes standing inside it."""
    X, Yt, Yb, Zf, Zb = 4.0, 3.0, -3.0, 0.0, 16.0
    meshes = [
        _quad([(-X, Yb, Zf), (X, Yb, Zf), (X, Yb, Zb), (-X, Yb, Zb)], _tex(rng, 24)),   # floor
        _quad([(-X, Yt, Zb), (X, Yt, Zb), (X, Yt, Zf), (-X, Yt, Zf)], _tex(rng, 24)),   # ceiling
        _quad([(-X, Yb, Zb), (X, Yb, Zb), (X, Yt, Zb), (-X, Yt, Zb)], _tex(rng, 24)),   # back wall
        _quad([(-X, Yb, Zf), (-X, Yb, Zb), (-X, Yt, Zb), (-X, Yt, Zf)], _tex(rng, 24)),  # left
        _quad([(X, Yb, Zb), (X, Yb, Zf), (X, Yt, Zf), (X, Yt, Zb)], _tex(rng, 24)),     # right
    ]
    for cx, cz, s in [(-1.8, 6.0, 0.9), (2.0, 9.0, 1.1), (-0.5, 12.0, 0.8)]:             # pillars
        meshes += _box(cx, Yb + s, cz, s, s, s, rng)
    return meshes


def _camera_traj(F):
    """CV camera-to-world poses: fly forward into the room with gentle sway, bob and yaw."""
    poses = []
    for f in range(F):
        t = np.array([0.7 * np.sin(f * 0.13), 0.25 * np.sin(f * 0.11), 0.45 * f + 0.5])
        R = _rot(0.0, 0.05 * np.sin(f * 0.18), 0.02 * np.sin(f * 0.12))
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
        poses.append(T)
    return poses


def _world(seed: int = _SEED, F: int = _F):
    """Deterministic world: GT poses, intrinsics, and offscreen-rendered (grayscale, depth)."""
    rng = np.random.default_rng(seed)
    f = _F_PIX; c = _IMG / 2.0
    intr = (f, f, c, c)
    yfov = 2.0 * np.arctan((_IMG / 2.0) / f)

    scene = pyrender.Scene(bg_color=[0, 0, 0], ambient_light=[0.6, 0.6, 0.6])
    for m in _scene(rng):
        scene.add(pyrender.Mesh.from_trimesh(m, smooth=False))
    cam = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=1.0)
    cam_node = scene.add(cam, pose=np.eye(4))
    light_node = scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=4.0), pose=np.eye(4))

    poses = _camera_traj(F)
    renderer = pyrender.OffscreenRenderer(_IMG, _IMG)
    rgb, depth = [], []
    for T in poses:
        gl = T @ _CV_TO_GL
        scene.set_pose(cam_node, gl); scene.set_pose(light_node, gl)
        # SKIP_CULL_FACES: the room is viewed from inside, so its walls' inward faces must draw
        # regardless of winding — otherwise back-facing walls are culled and read as empty space.
        color, d = renderer.render(scene, flags=RenderFlags.SKIP_CULL_FACES)
        rgb.append(cv_gray(color)); depth.append(d.astype(np.float32))
    renderer.delete()
    return poses, intr, np.stack(rgb), np.stack(depth)


def cv_gray(color):
    """RGB uint8 -> single-channel uint8 (ITU-R 601 luma), the format the ORB solver expects."""
    w = np.array([0.299, 0.587, 0.114])
    return (color[..., :3].astype(np.float32) @ w).clip(0, 255).astype(np.uint8)


def _poses_csv(poses) -> str:
    return "\n".join(_pose_line(T) for T in poses)


class MeshSlamProvider:
    """inputs split: frames.npz (rgb + depth + intrinsics). held-out: gt_poses.csv.

    Same on-disk contract as Rung 3 (ImageSlamProvider) — only the pixels are now from a real
    3D mesh render instead of a billboard splat, so the same solver runs unchanged."""

    def __init__(self, **world_kwargs):
        self.world_kwargs = world_kwargs
        self._cache = None

    def _render_once(self):
        # fetch() is called per dataset (frames + held-out GT); render the (slow) OSMesa world
        # ONCE and reuse — both splits must come from the same world anyway.
        if self._cache is None:
            self._cache = _world(**self.world_kwargs)
        return self._cache

    def fetch(self, ref: DatasetRef, dest) -> None:
        from pathlib import Path
        dest = Path(dest)
        poses, intr, rgb, depth = self._render_once()
        if ref.held_out:
            (dest / "gt_poses.csv").write_text(_poses_csv(poses))
        else:
            np.savez_compressed(dest / "frames.npz", rgb=rgb,
                                depth=depth.astype(np.float16), intr=np.array(intr))


def mesh_slam_task() -> ImplementationTask:
    return ImplementationTask(
        description=(
            "Implement image-based visual odometry on rendered 3D-scene frames. Read "
            "$LAB_DATA/frames.npz: 'rgb' is a stack of grayscale frames (F, H, W), 'depth' a "
            "matching per-pixel depth stack, 'intr' = [fx, fy, cx, cy]. NO feature tracks are "
            "given — detect and match features yourself (e.g. ORB + descriptor matching) "
            "between consecutive frames, read depth at each matched keypoint to back-project it "
            "to 3D, estimate the frame-to-frame rigid motion (e.g. Procrustes/Umeyama SE(3)), "
            "and chain into ABSOLUTE camera-to-world poses with frame 0 = identity. Write the "
            "trajectory to $LAB_ARTIFACTS/trajectory.csv: one pose per line as 12 comma-"
            "separated numbers — 3x3 rotation row-major (9) then translation (3), in frame order."),
        framework=FrameworkSpec("opencv", "4", "cpu"),
        entry_command="timeout 180 python3 $LAB_CODE/main.py",
        eval_command="python3 $LAB_CODE/eval.py", eval_code=_EVAL,
        metric="rpe", op="<=", threshold=0.05,
        datasets=[DatasetRef("meshslam-frames", "synthetic"),
                  DatasetRef("meshslam-gt", "synthetic", held_out=True)],
        entry_filename="main.py")
