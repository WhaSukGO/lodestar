"""Visualization previews — make the verifier's verdict legible to a human.

These plots are an INSPECTION tool, not part of grading. The oracle still decides
VERIFIED/REJECTED from the held-out metric (ATE/RPE); these pictures just show *why* — you
can see drift accumulate, and see loop closure pull it back. Pure matplotlib over the same
worlds and solvers the tests use; no renderer, no API.

Each function builds a rung's world, runs the honest and degenerate solvers, overlays their
trajectories on the hidden ground truth, and writes a PNG. Estimates are aligned to GT with
the same Umeyama fit the oracle uses, so the picture shows trajectory *shape* error (not an
arbitrary global frame)."""
from __future__ import annotations

import os

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402


def _umeyama(P, Q, dim):
    """Rigid fit mapping P onto Q (no scale) — same gauge the ATE/RPE oracles use."""
    P = np.asarray(P)[:, :dim]; Q = np.asarray(Q)[:, :dim]
    mp, mq = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - mp).T @ (Q - mq))
    D = np.eye(dim); D[-1, -1] = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ D @ U.T
    return (R @ P.T).T + (mq - R @ mp)


def _panel_data(rung: str, world_kwargs: dict):
    """For one scenario: (gt_xy, est_xy, metric, passed) — honest solver, 2D for plotting.
    Reuses the rung's own solver + metric; estimate aligned to GT exactly as the oracle does
    (Rung 1 needs no alignment — frame 0 is identity for both)."""
    from .scenarios import RUNGS
    th = RUNGS[rung]["task"]().threshold
    if rung == "0":
        from .worlds.posegraph2d import _world, ate, solve_posegraph
        gt, nodes, edges = _world(**world_kwargs)
        est = np.array(solve_posegraph(nodes.tolist(), edges))
        m = ate(est, gt)
        return gt[:, :2], _umeyama(est, gt, 2), m, m <= th
    if rung == "1":
        from .worlds.visual_odometry import _world, rpe, run_vo
        poses, intr, frames = _world(**world_kwargs)
        est = run_vo(intr, frames); m = rpe(est, poses)
        gp = np.array([T[:3, 3] for T in poses])[:, [2, 0]]   # top-down (z, x)
        ep = np.array([T[:3, 3] for T in est])[:, [2, 0]]
        return gp, ep, m, m <= th
    from .worlds.visual_slam import _world, ate, run_slam
    poses, intr, frames = _world(**world_kwargs)
    est = run_slam(intr, frames); m = ate(est, poses)
    gp = np.array([T[:3, 3] for T in poses])
    ep = _umeyama([T[:3, 3] for T in est], gp, 3)
    return gp[:, [2, 0]], ep[:, [2, 0]], m, m <= th


def viz_suite(rung: str, out_path: str) -> str:
    """Robustness grid: the honest solver's trajectory across each scenario of a rung, panels
    colored by the verifier's verdict. Makes the robustness table visual — you see drift grow
    and see exactly where VERIFIED flips to REJECTED."""
    from .scenarios import RUNGS
    cfg = RUNGS[rung]; scen = cfg["scenarios"]; metric = cfg["task"]().metric
    n = len(scen)
    fig, axes = plt.subplots(1, n, figsize=(3.3 * n, 3.7))
    n_pass = 0
    for ax, (name, kw) in zip(np.atleast_1d(axes), scen.items()):
        gt_xy, est_xy, m, passed = _panel_data(rung, kw)
        n_pass += int(passed)
        col = "#16a34a" if passed else "#dc2626"
        ax.plot(gt_xy[:, 0], gt_xy[:, 1], "k-", lw=2.0, label="ground truth")
        ax.plot(est_xy[:, 0], est_xy[:, 1], color=col, lw=1.6, label="honest estimate")
        ax.scatter([gt_xy[0, 0]], [gt_xy[0, 1]], c="k", s=18, zorder=5)
        ax.set_title(f"{name}\n{metric} {m:.3f} → {'VERIFIED' if passed else 'REJECTED'}",
                     color=col, fontsize=9)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.grid(alpha=0.15)
    fig.suptitle(f"Rung {rung} — {cfg['label']}: same honest solver, harder environments "
                 f"(oracle {metric} {cfg['task']().op} {cfg['task']().threshold})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(out_path, dpi=110); plt.close(fig)
    return f"{n_pass}/{n} scenarios VERIFIED"


def viz_posegraph(out_path: str, seed: int = 0) -> str:
    """Rung 0 (2D): GT vs drifted odometry vs optimized, with loop-closure chords."""
    from .worlds.posegraph2d import _world, ate, solve_posegraph
    gt, nodes, edges = _world(seed)
    opt = np.array(solve_posegraph(nodes.tolist(), edges))
    odo_a = _umeyama(nodes, gt, 2)
    opt_a = _umeyama(opt, gt, 2)
    ate_odo, ate_opt = ate(nodes, gt), ate(opt, gt)

    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    for e in edges:
        i, j = int(e[0]), int(e[1])
        if j - i > 1:                                # loop closure: chord between revisits
            ax.plot([gt[i, 0], gt[j, 0]], [gt[i, 1], gt[j, 1]], color="0.85", lw=0.6, zorder=0)
    ax.plot(odo_a[:, 0], odo_a[:, 1], "r--", lw=1.5, label=f"odometry only — ATE {ate_odo:.2f} (drifts)")
    ax.plot(gt[:, 0], gt[:, 1], "k-", lw=2.4, label="ground truth (hidden)")
    ax.plot(opt_a[:, 0], opt_a[:, 1], color="#16a34a", lw=1.8, label=f"optimized — ATE {ate_opt:.2f} → VERIFIED")
    ax.scatter([gt[0, 0]], [gt[0, 1]], c="k", s=40, zorder=5, label="start")
    ax.set_title("Rung 0 — 2D pose-graph: loop closures cancel drift")
    ax.set_aspect("equal"); ax.legend(loc="best", fontsize=8); ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(out_path, dpi=110); plt.close(fig)
    return f"ATE odo={ate_odo:.3f} opt={ate_opt:.3f}, {sum(1 for e in edges if e[1]-e[0]>1)} loop closures"


def viz_vo(out_path: str, seed: int = 0) -> str:
    """Rung 1 (3D, top-down x-z): GT vs honest VO vs 'camera never moved'."""
    from .worlds.visual_odometry import _world, rpe, run_vo
    poses, intr, frames = _world(seed)
    est = run_vo(intr, frames)
    static = [np.eye(4) for _ in frames]
    gp = np.array([T[:3, 3] for T in poses])         # frame0 = identity for both → no align
    ep = np.array([T[:3, 3] for T in est])
    sp = np.array([T[:3, 3] for T in static])
    rpe_vo, rpe_static = rpe(est, poses), rpe(static, poses)

    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    ax.plot(gp[:, 2], gp[:, 0], "k-", lw=2.4, label="ground truth (hidden)")
    ax.plot(ep[:, 2], ep[:, 0], color="#16a34a", lw=1.8, label=f"honest VO — RPE {rpe_vo:.3f} → VERIFIED")
    ax.plot(sp[:, 2], sp[:, 0], "rx", ms=7, label=f"static 'never moved' — RPE {rpe_static:.2f} → REJECTED")
    ax.scatter([gp[0, 2]], [gp[0, 0]], c="k", s=40, zorder=5)
    ax.set_title("Rung 1 — RGBD visual odometry (top-down)")
    ax.set_xlabel("z (forward)"); ax.set_ylabel("x")
    ax.set_aspect("equal"); ax.legend(loc="best", fontsize=8); ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(out_path, dpi=110); plt.close(fig)
    return f"RPE vo={rpe_vo:.3f} static={rpe_static:.3f}"


def viz_icl(out_path: str, seed: int = 0) -> str:
    """Rung 6: a real ICL-NUIM benchmark frame (left) next to GT vs honest VO vs static, top-down
    (right). Real RGB-D data, hidden GT trajectory, SE(3)-aligned ATE. Needs the cached dataset."""
    from .worlds import dataset_slam as ds
    poses, intr, rgb, depth = ds._world()
    est = ds.run_image_vo(rgb, depth.astype(np.float32), intr)
    ate_h = ds.ate(est, poses)
    ate_s = ds.ate([np.eye(4) for _ in poses], poses)
    gp = np.array([T[:3, 3] for T in poses])
    ep = _umeyama([T[:3, 3] for T in est], gp, 3)

    fig, (axi, ax) = plt.subplots(1, 2, figsize=(11.0, 4.6))
    axi.imshow(rgb[len(rgb) // 2], cmap="gray")
    axi.set_title("ICL-NUIM frame — REAL RGB-D benchmark", fontsize=9)
    axi.set_xticks([]); axi.set_yticks([])
    ax.plot(gp[:, 2], gp[:, 0], "k-", lw=2.4, label="ground truth (hidden)")
    ax.plot(ep[:, 2], ep[:, 0], color="#16a34a", lw=1.8, label=f"honest VO — ATE {ate_h:.3f} → VERIFIED")
    ax.scatter([gp[0, 2]], [gp[0, 0]], c="k", s=40, zorder=5, label="start")
    ax.set_title(f"trajectory (top-down) · static 'never moved' ATE {ate_s:.2f} → REJECTED", fontsize=9)
    ax.set_xlabel("z"); ax.set_ylabel("x")
    ax.set_aspect("equal"); ax.legend(loc="best", fontsize=8); ax.grid(alpha=0.2)
    fig.suptitle("Rung 6 — VO graded on a real SLAM benchmark (ICL-NUIM)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(out_path, dpi=110); plt.close(fig)
    return f"ATE honest={ate_h:.3f} static={ate_s:.3f}"


def viz_kitti(out_path: str, seed: int = 0) -> str:
    """Rung 8: a real KITTI driving frame (left) next to the car's GT path vs honest VO, top-down
    (right). Real automotive stereo data; depth from SGBM. Needs the cached KITTI sequence."""
    from .worlds import kitti_slam as K
    poses, intr, rgb, depth = K._world()
    est = K.run_image_vo(rgb, depth.astype(np.float32), intr)
    rpe_h = K.rpe(est, poses)
    rpe_s = K.rpe([np.eye(4) for _ in poses], poses)
    gp = np.array([T[:3, 3] for T in poses])
    ep = np.array([T[:3, 3] for T in est])           # frame0 = identity for both -> no alignment

    fig, (axi, ax) = plt.subplots(1, 2, figsize=(12.0, 4.2),
                                  gridspec_kw={"width_ratios": [1.7, 1]})
    axi.imshow(rgb[len(rgb) // 2], cmap="gray")
    axi.set_title("KITTI frame — REAL car-camera (driving)", fontsize=9)
    axi.set_xticks([]); axi.set_yticks([])
    ax.plot(gp[:, 2], gp[:, 0], "k-", lw=2.4, label="ground truth (hidden)")
    ax.plot(ep[:, 2], ep[:, 0], color="#16a34a", lw=1.8, label=f"honest VO — RPE {rpe_h:.2f} → VERIFIED")
    ax.scatter([gp[0, 2]], [gp[0, 0]], c="k", s=40, zorder=5, label="start")
    ax.set_title(f"car trajectory (top-down) · static RPE {rpe_s:.2f} → REJECTED", fontsize=9)
    ax.set_xlabel("z (forward, m)"); ax.set_ylabel("x (m)")
    ax.set_aspect("equal"); ax.legend(loc="best", fontsize=8); ax.grid(alpha=0.2)
    fig.suptitle("Rung 8 — automotive VO on real KITTI driving data", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(out_path, dpi=110); plt.close(fig)
    return f"RPE honest={rpe_h:.3f} static={rpe_s:.3f}"


def viz_replica(out_path: str, seed: int = 0) -> str:
    """Rung 7: a frame rendered from a REAL scanned Replica apartment (left) next to GT vs honest
    VO, top-down (right). Renders via BlenderProc (slow) — needs blenderproc + a cached scene."""
    from PIL import Image
    from .worlds.replica_slam import render_world, rpe, run_image_vo
    poses, intr, rgb, depth, outdir = render_world(seed=seed)
    est = run_image_vo(rgb, depth.astype(np.float32), intr)
    rpe_h = rpe(est, poses)
    rpe_s = rpe([np.eye(4) for _ in poses], poses)
    gp = np.array([T[:3, 3] for T in poses])
    ep = _umeyama([T[:3, 3] for T in est], gp, 3)

    fig, (axi, ax) = plt.subplots(1, 2, figsize=(11.0, 4.8))
    cpath = os.path.join(outdir, "preview_color.png")
    axi.imshow(np.asarray(Image.open(cpath)) if os.path.exists(cpath) else rgb[len(rgb) // 2])
    axi.set_title("real scanned apartment (Replica) — rendered frame", fontsize=9)
    axi.set_xticks([]); axi.set_yticks([])
    ax.plot(gp[:, 2], gp[:, 0], "k-", lw=2.4, label="ground truth (hidden)")
    ax.plot(ep[:, 2], ep[:, 0], color="#16a34a", lw=1.8, label=f"honest VO — RPE {rpe_h:.3f} → VERIFIED")
    ax.scatter([gp[0, 2]], [gp[0, 0]], c="k", s=40, zorder=5, label="start")
    ax.set_title(f"trajectory (top-down) · static 'never moved' RPE {rpe_s:.2f} → REJECTED", fontsize=9)
    ax.set_xlabel("z"); ax.set_ylabel("x")
    ax.set_aspect("equal"); ax.legend(loc="best", fontsize=8); ax.grid(alpha=0.2)
    fig.suptitle("Rung 7 — image VO on a real scanned apartment (Replica, via BlenderProc)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(out_path, dpi=110); plt.close(fig)
    return f"RPE honest={rpe_h:.3f} static={rpe_s:.3f}"


def viz_blender(out_path: str, seed: int = 0) -> str:
    """Rung 5: a path-traced color frame (left) next to GT vs honest VO, top-down (right).
    Renders the world with BlenderProc (slow) — needs blenderproc installed."""
    from PIL import Image
    from .worlds.blender_slam import render_world, rpe, run_image_vo
    poses, intr, rgb, depth, outdir = render_world(seed=seed)
    est = run_image_vo(rgb, depth.astype(np.float32), intr)
    rpe_h = rpe(est, poses)
    rpe_s = rpe([np.eye(4) for _ in poses], poses)
    gp = np.array([T[:3, 3] for T in poses])
    ep = np.array([T[:3, 3] for T in est])

    fig, (axi, ax) = plt.subplots(1, 2, figsize=(11.0, 5.0))
    cpath = os.path.join(outdir, "preview_color.png")
    axi.imshow(np.asarray(Image.open(cpath)) if os.path.exists(cpath) else rgb[len(rgb) // 2])
    axi.set_title("path-traced frame (Cycles: GI + soft shadows)", fontsize=9)
    axi.set_xticks([]); axi.set_yticks([])
    ax.plot(gp[:, 2], gp[:, 0], "k-", lw=2.4, label="ground truth (hidden)")
    ax.plot(ep[:, 2], ep[:, 0], color="#16a34a", lw=1.8, label=f"honest VO — RPE {rpe_h:.3f} → VERIFIED")
    ax.scatter([gp[0, 2]], [gp[0, 0]], c="k", s=40, zorder=5, label="start")
    ax.set_title(f"trajectory (top-down) · static 'never moved' RPE {rpe_s:.2f} → REJECTED", fontsize=9)
    ax.set_xlabel("z (forward)"); ax.set_ylabel("x")
    ax.set_aspect("equal"); ax.legend(loc="best", fontsize=8); ax.grid(alpha=0.2)
    fig.suptitle("Rung 5 — image VO on a photorealistic path-traced world (BlenderProc/Cycles)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(out_path, dpi=110); plt.close(fig)
    return f"RPE honest={rpe_h:.3f} static={rpe_s:.3f}"


def viz_mesh(out_path: str, seed: int = 0) -> str:
    """Rung 4: a sample rendered frame from the real 3D scene (left) next to GT vs honest VO vs
    static, top-down (right). The frame shows it's a true render — occlusion, perspective — not
    a billboard splat. Needs the offscreen GL stack (pyrender + OSMesa)."""
    from .worlds.mesh_slam import _world, rpe, run_image_vo
    poses, intr, rgb, depth = _world(seed)
    est = run_image_vo(rgb, depth.astype(np.float32), intr)
    static = [np.eye(4) for _ in poses]
    gp = np.array([T[:3, 3] for T in poses])             # frame0 = identity for both → no align
    ep = np.array([T[:3, 3] for T in est])
    rpe_vo, rpe_static = rpe(est, poses), rpe(static, poses)

    fig, (axi, ax) = plt.subplots(1, 2, figsize=(11.0, 5.0))
    axi.imshow(rgb[len(rgb) // 2], cmap="gray")
    axi.set_title("rendered frame (real 3D mesh: occlusion + perspective)", fontsize=9)
    axi.set_xticks([]); axi.set_yticks([])
    ax.plot(gp[:, 2], gp[:, 0], "k-", lw=2.4, label="ground truth (hidden)")
    ax.plot(ep[:, 2], ep[:, 0], color="#16a34a", lw=1.8, label=f"honest VO — RPE {rpe_vo:.3f} → VERIFIED")
    ax.scatter([gp[0, 2]], [gp[0, 0]], c="k", s=40, zorder=5, label="start")
    ax.set_title(f"trajectory (top-down) · static 'never moved' RPE {rpe_static:.2f} → REJECTED", fontsize=9)
    ax.set_xlabel("z (forward)"); ax.set_ylabel("x")
    ax.set_aspect("equal"); ax.legend(loc="best", fontsize=8); ax.grid(alpha=0.2)
    fig.suptitle("Rung 4 — image VO on an actual 3D world (pyrender/OSMesa)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(out_path, dpi=110); plt.close(fig)
    return f"RPE vo={rpe_vo:.3f} static={rpe_static:.3f}"


def viz_slam(out_path: str, seed: int = 0) -> str:
    """Rung 2 (3D, top-down x-z): GT vs full SLAM vs VO-only, over the landmark ring."""
    from .worlds.visual_slam import _world, ate, run_slam, run_vo_only
    poses, intr, frames = _world(seed)
    slam = run_slam(intr, frames)
    vo = run_vo_only(intr, frames)
    gp = np.array([T[:3, 3] for T in poses])
    sp = _umeyama([T[:3, 3] for T in slam], gp, 3)   # GT frame0 != identity → align
    vp = _umeyama([T[:3, 3] for T in vo], gp, 3)
    ate_slam, ate_vo = ate(slam, poses), ate(vo, poses)

    rng = np.random.default_rng(seed)
    ang = rng.uniform(0, 2 * np.pi, 500); rad = rng.uniform(7, 12, 500)
    fig, ax = plt.subplots(figsize=(6.6, 6.0))
    ax.scatter(rad * np.sin(ang), rad * np.cos(ang), s=2, c="0.8", label="landmarks")
    ax.plot(vp[:, 2], vp[:, 0], "r-", lw=1.5, label=f"VO only — ATE {ate_vo:.2f} → REJECTED")
    ax.plot(gp[:, 2], gp[:, 0], "k-", lw=2.4, label="ground truth (hidden)")
    ax.plot(sp[:, 2], sp[:, 0], color="#16a34a", lw=1.8, label=f"full SLAM — ATE {ate_slam:.2f} → VERIFIED")
    ax.set_title("Rung 2 — full visual SLAM: loop closure > odometry")
    ax.set_xlabel("z"); ax.set_ylabel("x")
    ax.set_aspect("equal"); ax.legend(loc="best", fontsize=8); ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(out_path, dpi=110); plt.close(fig)
    return f"ATE slam={ate_slam:.3f} vo_only={ate_vo:.3f}"
