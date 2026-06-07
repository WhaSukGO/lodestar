"""Rung 4 end-to-end through the Touchstone verifier (offline, local mode, numpy + cv2 + pyrender).

The SAME ORB front-end as Rung 3, now graded on an ACTUAL 3D world: a real triangle-mesh room
with boxes, offscreen-rendered (pyrender/OSMesa) with genuine occlusion and perspective. Honest
image VO is VERIFIED on hidden GT (low RPE); a "camera never moved" solver is REJECTED. Proves
the verifier — and the same solver — hold up when the renderer is real, not a billboard splat.

Skipped automatically if the offscreen GL stack (pyrender + OSMesa) is unavailable."""
from __future__ import annotations

import os

# PyOpenGL locks its platform at first import; select the headless software-GL backend BEFORE
# pyrender pulls it in, or it defaults to EGL (unavailable here) and every render fails.
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import numpy as np
import pytest

pyrender = pytest.importorskip("pyrender")
trimesh = pytest.importorskip("trimesh")


def _world_or_skip():
    try:
        from lodestar.worlds.mesh_slam import _world
        return _world(0)
    except Exception as e:                           # no software GL context available here
        pytest.skip(f"offscreen GL render unavailable: {e}")


def _run(root, author):
    from lodestar._spine import ExperimentRecord, build_implementer_harness
    from lodestar.worlds.mesh_slam import MeshSlamProvider, mesh_slam_task
    h = build_implementer_harness(str(root), mesh_slam_task(), author_fn=author,
                                  provider=MeshSlamProvider(), job_mode="local")
    return h.run_experiment(ExperimentRecord(id="mesh", hypothesis="mesh slam"))


def test_mesh_vo_verified_static_rejected(tmp_path):
    _world_or_skip()                                 # ensure the renderer works before the harness run
    from lodestar.worlds.mesh_slam import HONEST, STATIC

    good = _run(tmp_path / "good", HONEST)
    assert good.status.value == "VERIFIED"
    assert good.verdict.measured_metrics["rpe"] <= 0.05     # real motion recovered on a real 3D render

    bad = _run(tmp_path / "bad", STATIC)
    assert bad.status.value == "REJECTED"
    assert bad.verdict.measured_metrics["rpe"] > 0.05


def test_render_is_truly_3d_with_occlusion():
    """A real render, not a billboard splat: rich features + a genuine depth buffer with
    interior occluders standing in front of the back wall."""
    poses, intr, rgb, depth = _world_or_skip()
    import cv2
    assert len(cv2.ORB_create(nfeatures=1500).detect(rgb[0], None)) > 300   # textured surfaces
    valid = depth[depth > 0]
    assert valid.min() < 6.0 and valid.max() > 10.0        # occluders up close AND far back wall


def test_mesh_vo_beats_static_offline():
    poses, intr, rgb, depth = _world_or_skip()
    from lodestar.worlds.mesh_slam import rpe, run_image_vo
    static = [np.eye(4) for _ in poses]
    assert rpe(run_image_vo(rgb, depth.astype(np.float32), intr), poses) < 0.25 * rpe(static, poses)
