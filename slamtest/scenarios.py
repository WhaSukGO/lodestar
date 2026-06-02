"""Selectable environments — named difficulty presets per rung.

Each preset is a dict of `_world` knobs for that rung's world. The ORACLE (threshold) stays
fixed; only the WORLD changes — so the suite asks "does the same honest algorithm still pass
when the environment gets harder?" That is exactly where the interesting answers live: e.g. a
pose-graph optimizer can beat high odometry noise *because of* loop closures, but with the
loop closures removed (`no-loops`) the drift is uncorrectable and the verifier REJECTS it.

Presets were tuned so the honest solver passes easy/default and fails the hard ones — the
verifier reports that honestly rather than rubber-stamping. `default` == the committed world."""
from __future__ import annotations

from .worlds import posegraph2d as _pg
from .worlds import visual_odometry as _vo
from .worlds import visual_slam as _vs

# Rung 0 — 2D pose-graph (oracle: ate <= 0.12)
POSEGRAPH = {
    "easy":       dict(odo_sigma=(0.02, 0.02, 0.012)),
    "default":    dict(),
    "high-noise": dict(odo_sigma=(0.10, 0.10, 0.06)),    # loops help, but not infinitely
    "no-loops":   dict(lc_radius=0.0),                   # no revisits -> drift uncorrectable
}

# Rung 1 — RGBD visual odometry (oracle: rpe <= 0.15)
VO = {
    "easy":       dict(px_sigma=0.3, depth_sigma=0.005),
    "default":    dict(),
    "high-noise": dict(px_sigma=1.6, depth_sigma=0.035),
    "sparse":     dict(L=70),                            # fewer landmarks -> thinner matches
}

# Rung 2 — full visual SLAM (oracle: ate <= 0.10)
SLAM = {
    "easy":       dict(px_sigma=0.5, depth_sigma=0.006),
    "default":    dict(),
    "high-noise": dict(px_sigma=1.6, depth_sigma=0.03),
    "no-loops":   dict(loops=1.0, r=8.0),                # single pass, no revisits -> ~VO drift
}

# Registry: everything the suite runner needs to grade a solver across a rung's environments.
RUNGS = {
    "0": dict(label="2D pose-graph", provider=_pg.PoseGraphProvider, task=_pg.posegraph_task,
              honest=_pg.HONEST, degenerate=_pg.ODOMETRY, scenarios=POSEGRAPH),
    "1": dict(label="RGBD visual odometry", provider=_vo.VisualOdometryProvider,
              task=_vo.vo_task, honest=_vo.HONEST, degenerate=_vo.STATIC, scenarios=VO),
    "2": dict(label="full visual SLAM", provider=_vs.VisualSlamProvider, task=_vs.slam_task,
              honest=_vs.HONEST, degenerate=_vs.VO_ONLY, scenarios=SLAM),
}
