import blenderproc as bproc
# Rung 5 world renderer — runs INSIDE BlenderProc's bundled Blender (Cycles path tracer, CPU or
# CUDA-compute; no OpenGL/EGL). Builds a photorealistic textured room with furniture, flies a
# camera through it, and writes the Lodestar contract: frames.npz (grayscale rgb + metric depth
# + intrinsics) and gt_poses.csv (hidden ground-truth camera-to-world poses, 12 numbers/row).
#
# Real path tracing means global illumination, soft shadows, and inter-reflections — the pixels
# carry lighting cues a real camera sees, on top of textured geometry. Same OpenCV/Blender basis
# flip as the pyrender rung (diag(1,-1,-1,1)). Invoked via:
#   blenderproc run _blender_render.py --output DIR --frames F --seed S --res R --samples N
import argparse
import os
import sys

import bpy
import numpy as np

# blenderproc passes script args after a "--" separator on some versions; tolerate both.
_argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
_p = argparse.ArgumentParser()
_p.add_argument("--output", required=True)
_p.add_argument("--frames", type=int, default=24)
_p.add_argument("--seed", type=int, default=0)
_p.add_argument("--res", type=int, default=480)
_p.add_argument("--samples", type=int, default=48)
args = _p.parse_args(_argv)

RES = args.res
rng = np.random.default_rng(args.seed)
_CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])    # OpenCV cam (+z fwd,+y down) -> Blender (-z fwd,+y up)


def _rot(ax, ay, az):
    cx, sx = np.cos(ax), np.sin(ax); cy, sy = np.cos(ay), np.sin(ay); cz, sz = np.cos(az), np.sin(az)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _block_tex(n=18, px=14):
    """High-contrast block texture -> stable ORB corners survive path-tracer shading."""
    small = rng.integers(20, 235, (n, n, 3), dtype=np.uint8)
    return np.repeat(np.repeat(small, px, 0), px, 1)


def _textured_material(name, tex):
    """A Principled-BSDF material whose base color is an in-memory image texture (raw bpy —
    stable across Blender versions). Slight roughness so path tracing yields real shading."""
    h, w, _ = tex.shape
    img = bpy.data.images.new(name, width=w, height=h)
    rgba = np.ones((h, w, 4), np.float32)
    rgba[..., :3] = (tex.astype(np.float32) / 255.0) ** 2.2          # sRGB tex -> linear (pixels are linear)
    img.colorspace_settings.name = "Non-Color"                       # don't double-convert on read
    img.pixels = rgba[::-1].reshape(-1).tolist()                      # blender is bottom-up
    mat = bpy.data.materials.new(name); mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    bsdf.inputs["Roughness"].default_value = 0.6
    tex_node = nt.nodes.new("ShaderNodeTexImage"); tex_node.image = img
    nt.links.new(bsdf.inputs["Base Color"], tex_node.outputs["Color"])
    return mat


def _attach(obj, mat):
    bo = obj.blender_obj
    bo.data.materials.clear()
    bo.data.materials.append(mat)


def _plane(name, location, rotation, scale):
    p = bproc.object.create_primitive("PLANE")
    p.set_location(location); p.set_rotation_euler(rotation); p.set_scale(scale)
    _attach(p, _textured_material(name, _block_tex(24)))
    return p


def _box(name, location, scale):
    b = bproc.object.create_primitive("CUBE")
    b.set_location(location); b.set_scale(scale)
    _attach(b, _textured_material(name, _block_tex()))
    return b


def build_scene():
    """A real room — floor, ceiling, back wall and two side walls correctly oriented (a Blender
    PLANE lies in XY with +Z normal; rotate it to face the right way), with furniture boxes
    standing on the floor. Camera flies along +z (OpenCV). Room: x[-4,4] y[-3,3] z[-1,16]."""
    HX, HY = 4.0, 3.0                       # half width (x) and half height (y)
    ZN, ZF = -1.0, 16.0                     # near (behind start) and far (back wall) z
    CZ, DZ = (ZN + ZF) / 2, (ZF - ZN) / 2   # room centre z and half-depth
    _plane("floor", [0, -HY, CZ], [np.pi / 2, 0, 0], [HX, DZ, 1])   # horizontal, spans x,z
    _plane("ceil", [0, HY, CZ], [np.pi / 2, 0, 0], [HX, DZ, 1])
    _plane("back", [0, 0, ZF], [0, 0, 0], [HX, HY, 1])             # vertical, spans x,y
    _plane("left", [-HX, 0, CZ], [0, np.pi / 2, 0], [DZ, HY, 1])    # vertical, spans z,y
    _plane("right", [HX, 0, CZ], [0, np.pi / 2, 0], [DZ, HY, 1])
    # Furniture spread through the room so every frame of the fly-through keeps near-field texture
    # and parallax (a forward camera otherwise runs out of trackable content and VO degenerates).
    boxes = [(-2.2, 4.0, 0.9), (2.4, 5.5, 1.1), (-0.8, 7.0, 0.7), (2.6, 8.5, 0.9),
             (-2.6, 10.0, 1.0), (0.9, 11.5, 0.8), (-1.4, 13.0, 1.0), (2.2, 14.0, 0.9)]
    for i, (x, z, s) in enumerate(boxes):
        _box(f"box{i}", [x, -HY + s, z], [s, s, s])                # rests on the floor
    # Ceiling area lights, aimed DOWN (-y), for soft shadows + global illumination. High wattage
    # because a path-traced room this size needs it to be well-exposed.
    for lx, lz in [(-1.5, 4.0), (1.5, 8.0), (-1.0, 12.0)]:
        light = bproc.types.Light(); light.set_type("AREA")
        light.set_location([lx, HY - 0.1, lz]); light.set_rotation_euler([-np.pi / 2, 0, 0])
        light.set_energy(1200); light.set_scale([3, 3, 3])


def camera_traj(F):
    """Gentle forward fly that STAYS in the textured zone (z ~1..7 of a 16-deep room) with sway,
    bob and slow yaw — so every frame sees rich, parallax-rich texture and VO never starves."""
    poses = []
    for f in range(F):
        t = np.array([0.8 * np.sin(f * 0.16), 0.25 * np.sin(f * 0.12), 0.26 * f + 1.0])
        R = _rot(0.0, 0.05 * np.sin(f * 0.15), 0.02 * np.sin(f * 0.12))   # gentle yaw (stay on rich content)
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
        poses.append(T)
    return poses


def main():
    bproc.init()
    # Blender 4.x defaults to the AgX view transform, which crushes brightness; use Standard so the
    # rendered grayscale is well-exposed (ORB needs contrast, not a dim AgX look).
    bpy.context.scene.view_settings.view_transform = "Standard"
    world = bpy.context.scene.world; world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg is not None:                                  # bright, fairly uniform ambient fill so NO
        bg.inputs[0].default_value = (0.55, 0.57, 0.6, 1.0)   # frame is underexposed (ORB needs it)
        bg.inputs[1].default_value = 3.0
    build_scene()

    f_pix = RES * (400.0 / 480.0)                       # match Rung 4 fov (~480px@f400)
    c = RES / 2.0
    K = np.array([[f_pix, 0, c], [0, f_pix, c], [0, 0, 1]])
    bproc.camera.set_intrinsics_from_K_matrix(K, image_width=RES, image_height=RES)

    poses = camera_traj(args.frames)
    for T in poses:
        bproc.camera.add_camera_pose(T @ _CV_TO_GL)

    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.renderer.set_max_amount_of_samples(args.samples)
    data = bproc.renderer.render()

    colors = np.stack([np.asarray(c)[..., :3] for c in data["colors"]]).astype(np.float32)
    gray = (colors @ np.array([0.299, 0.587, 0.114])).clip(0, 255).astype(np.uint8)
    depth = np.stack([np.asarray(d) for d in data["depth"]]).astype(np.float32)
    depth[~np.isfinite(depth)] = 0.0
    depth[depth > 1e3] = 0.0                            # background sky -> invalid

    os.makedirs(args.output, exist_ok=True)
    intr = np.array([f_pix, f_pix, c, c])
    np.savez_compressed(os.path.join(args.output, "frames.npz"),
                        rgb=gray, depth=depth.astype(np.float16), intr=intr)
    rows = []
    for T in poses:
        rows.append(",".join(repr(float(v)) for v in list(T[:3, :3].flatten()) + list(T[:3, 3])))
    with open(os.path.join(args.output, "gt_poses.csv"), "w") as fh:
        fh.write("\n".join(rows))
    # a color preview frame (mid-trajectory) for the README
    try:
        from PIL import Image
        mid = (colors[len(colors) // 2]).clip(0, 255).astype(np.uint8)
        Image.fromarray(mid).save(os.path.join(args.output, "preview_color.png"))
    except Exception:
        pass
    print("BLENDER_WORLD_OK frames=%d res=%d depth=%.2f..%.2f" %
          (len(gray), RES, float(depth[depth > 0].min()), float(depth.max())))


main()
