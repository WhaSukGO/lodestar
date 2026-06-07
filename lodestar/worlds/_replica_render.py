import blenderproc as bproc
# Rung 7 renderer — runs INSIDE BlenderProc's Blender. Loads a REAL scanned Replica scene
# (vertex-colored mesh of an actual apartment), gives it an emissive vertex-color material so the
# captured appearance renders without placing lights, flies a camera through the room, and writes
# the Lodestar contract: frames.npz (grayscale + metric depth + intrinsics) + gt_poses.csv.
#
# Camera poses are authored directly in Blender world space (so they sit inside the real room);
# the hidden GT is written in OpenCV convention via the diag(1,-1,-1,1) flip, matching the solver.
# Invoked via: blenderproc run _replica_render.py --data_path DIR --scene apartment_0 --output ...
import argparse
import os
import sys

import bpy
import numpy as np

_argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
_p = argparse.ArgumentParser()
_p.add_argument("--data_path", required=True)
_p.add_argument("--scene", default="apartment_0")
_p.add_argument("--output", required=True)
_p.add_argument("--frames", type=int, default=24)
_p.add_argument("--seed", type=int, default=0)
_p.add_argument("--res", type=int, default=480)
_p.add_argument("--samples", type=int, default=48)
args = _p.parse_args(_argv)

RES = args.res
rng = np.random.default_rng(args.seed)
_CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])


def _emissive_vertex_color_material():
    """Material that emits the mesh's per-vertex scan colors -> the captured appearance renders
    directly (no lights needed, never black) and is rich in ORB features."""
    mat = bpy.data.materials.new("replica_vc"); mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    emit = nt.nodes.new("ShaderNodeEmission")
    emit.inputs["Strength"].default_value = 0.8    # scan colors already carry baked lighting; <1 to
    vc = nt.nodes.new("ShaderNodeVertexColor")     # avoid blowing out bright spots (e.g. windows)
    nt.links.new(emit.inputs["Color"], vc.outputs["Color"])
    nt.links.new(out.inputs["Surface"], emit.outputs["Emission"])
    return mat


def _apply_material(objs, mat):
    for o in objs:
        bo = o.blender_obj
        bo.data.materials.clear()
        bo.data.materials.append(mat)


def _world_bbox(objs):
    lo = np.array([np.inf] * 3); hi = -lo
    for o in objs:
        bo = o.blender_obj
        for corner in bo.bound_box:                # 8 local corners
            w = bo.matrix_world @ __import__("mathutils").Vector(corner)
            p = np.array([w.x, w.y, w.z])
            lo = np.minimum(lo, p); hi = np.maximum(hi, p)
    return lo, hi


def _look_at(eye, target, up):
    """Blender camera-to-world (camera looks along -Z, up +Y)."""
    fwd = target - eye; fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up); right /= np.linalg.norm(right)
    up2 = np.cross(right, fwd)
    R = np.column_stack([right, up2, -fwd])        # camera axes in world
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = eye
    return T


def _camera_traj(lo, hi, F):
    """A gentle path through the middle of the room. Up = smallest-extent axis (rooms are wider
    than tall); move along the largest horizontal axis, looking ahead with slight sway."""
    ext = hi - lo
    up_ax = int(np.argmin(ext))
    horiz = [a for a in range(3) if a != up_ax]
    long_ax = horiz[int(np.argmax(ext[horiz]))]
    side_ax = horiz[1 - horiz.index(long_ax)]
    up = np.zeros(3); up[up_ax] = 1.0
    center = (lo + hi) / 2.0
    floor = lo[up_ax]
    poses = []
    for f in range(F):
        s = (f / max(1, F - 1)) - 0.5              # -0.5 .. 0.5 along the long axis
        eye = center.copy()
        eye[up_ax] = floor + 0.55 * ext[up_ax]     # ~mid height
        eye[long_ax] = center[long_ax] - 0.30 * ext[long_ax] * s    # traverse part of the room
        eye[side_ax] = center[side_ax] + 0.10 * ext[side_ax] * np.sin(f * 0.4)
        target = center.copy()
        target[up_ax] = eye[up_ax]
        target[long_ax] = center[long_ax] + 0.30 * ext[long_ax]     # look down the room
        target[side_ax] = center[side_ax] + 0.15 * ext[side_ax] * np.sin(f * 0.4 + 0.5)
        poses.append(_look_at(eye, target, up))
    return poses


def main():
    bproc.init()
    bpy.context.scene.view_settings.view_transform = "Standard"
    objs = bproc.loader.load_replica(data_path=args.data_path, data_set_name=args.scene)
    _apply_material(objs, _emissive_vertex_color_material())

    lo, hi = _world_bbox(objs)
    f_pix = RES * (400.0 / 480.0); c = RES / 2.0
    K = np.array([[f_pix, 0, c], [0, f_pix, c], [0, 0, 1]])
    bproc.camera.set_intrinsics_from_K_matrix(K, image_width=RES, image_height=RES)

    poses_bl = _camera_traj(lo, hi, args.frames)
    for T in poses_bl:
        bproc.camera.add_camera_pose(T)

    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.renderer.set_max_amount_of_samples(args.samples)
    data = bproc.renderer.render()

    colors = np.stack([np.asarray(c)[..., :3] for c in data["colors"]]).astype(np.float32)
    gray = (colors @ np.array([0.299, 0.587, 0.114])).clip(0, 255).astype(np.uint8)
    depth = np.stack([np.asarray(d) for d in data["depth"]]).astype(np.float32)
    depth[~np.isfinite(depth)] = 0.0
    depth[depth > 1e3] = 0.0

    os.makedirs(args.output, exist_ok=True)
    intr = np.array([f_pix, f_pix, c, c])
    np.savez_compressed(os.path.join(args.output, "frames.npz"),
                        rgb=gray, depth=depth.astype(np.float16), intr=intr)
    rows = []
    for T_bl in poses_bl:
        T_cv = T_bl @ _CV_TO_GL                    # Blender -> OpenCV convention for the solver/GT
        rows.append(",".join(repr(float(v)) for v in list(T_cv[:3, :3].flatten()) + list(T_cv[:3, 3])))
    with open(os.path.join(args.output, "gt_poses.csv"), "w") as fh:
        fh.write("\n".join(rows))
    try:
        from PIL import Image
        Image.fromarray(colors[len(colors) // 2].clip(0, 255).astype(np.uint8)).save(
            os.path.join(args.output, "preview_color.png"))
    except Exception:
        pass
    print("REPLICA_WORLD_OK scene=%s frames=%d res=%d depth=%.2f..%.2f" %
          (args.scene, len(gray), RES, float(depth[depth > 0].min()), float(depth.max())))


main()
