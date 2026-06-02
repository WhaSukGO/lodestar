"""Smoke test: each preview renders a non-empty PNG (offline, no API)."""
from __future__ import annotations

from slamtest.viz import viz_posegraph, viz_slam, viz_vo


def test_previews_render(tmp_path):
    for name, fn in [("r0", viz_posegraph), ("r1", viz_vo), ("r2", viz_slam)]:
        p = tmp_path / f"{name}.png"
        info = fn(str(p))
        assert p.exists() and p.stat().st_size > 1000      # a real image was written
        assert isinstance(info, str) and info              # returns a metric summary
