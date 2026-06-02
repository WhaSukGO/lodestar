"""Rung 0 — an ALTERNATIVE pose-graph solver: SE(2) Gauss-Newton with scipy.sparse.

A second, independent reference solver for the 2D pose-graph task, kept to show the testbed
accepts *diverse* honest implementations and grades them the same way. Where the in-module
`solve_posegraph` (worlds/posegraph2d.py) is a compact dense-numpy optimizer, this one uses a
sparse system (scipy CG / spsolve), Levenberg-Marquardt damping, and union-find
connected-component anchoring. Both land at the same hidden-GT ATE (≈0.06 → VERIFIED).

Used as a swappable solver via `posegraph2d.SCIPY` (writes this file as the entry `main.py`),
so it runs through the unchanged verifier. Requires scipy. The entry point reads
`$LAB_DATA/graph.json` and writes `$LAB_ARTIFACTS/trajectory.csv`."""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _wrap(a):
    """Wrap angle(s) to (-pi, pi]."""
    return ((np.asarray(a) + np.pi) % (2 * np.pi)) - np.pi


def _objective(x: np.ndarray, edges: list) -> float:
    """Total weighted squared residual."""
    total = 0.0
    for e in edges:
        i, j = int(e[0]), int(e[1])
        dx, dy, dtheta = float(e[2]), float(e[3]), float(e[4])
        w_xy, w_theta = float(e[5]), float(e[6])
        xi, yi, ti = x[3*i], x[3*i+1], x[3*i+2]
        xj, yj = x[3*j], x[3*j+1]
        c, s = np.cos(ti), np.sin(ti)
        dtx, dty = xj - xi, yj - yi
        ex = (c * dtx + s * dty) - dx
        ey = (-s * dtx + c * dty) - dy
        er = _wrap((x[3*j+2] - ti) - dtheta)
        total += w_xy * (ex**2 + ey**2) + w_theta * er**2
    return total


def _find_components(n: int, edges: list) -> list:
    """Union-find: component root (smallest index) per node, for per-component anchoring."""
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for e in edges:
        ra, rb = find(int(e[0])), find(int(e[1]))
        if ra != rb:
            parent[rb] = ra
    return [find(k) for k in range(n)]


def solve_posegraph(nodes_init: list, edges: list, max_iters: int = 50, tol: float = 1e-6):
    """Gauss-Newton on SE(2) with a sparse system. nodes_init: [[x,y,theta],...];
    edges: [[i, j, dx, dy, dtheta, w_xy, w_theta], ...]. Returns (poses Nx3, iters, objective)."""
    n = len(nodes_init)
    x = np.array(nodes_init, dtype=float).flatten()
    anchors = set(_find_components(n, edges))            # one anchor per connected component

    iters_taken = 0
    for it in range(max_iters):
        H = sp.lil_matrix((3 * n, 3 * n), dtype=float)
        b = np.zeros(3 * n, dtype=float)
        for e in edges:
            i, j = int(e[0]), int(e[1])
            dx_m, dy_m, dth_m = float(e[2]), float(e[3]), float(e[4])
            w_xy, w_theta = float(e[5]), float(e[6])
            xi, yi, ti = x[3*i], x[3*i+1], x[3*i+2]
            xj, yj = x[3*j], x[3*j+1]
            c, s = np.cos(ti), np.sin(ti)
            dtx, dty = xj - xi, yj - yi
            e_vec = np.array([(c*dtx + s*dty) - dx_m, (-s*dtx + c*dty) - dy_m,
                              float(_wrap((x[3*j+2] - ti) - dth_m))])
            Om = np.diag([w_xy, w_xy, w_theta])
            Ji = np.array([[-c, -s, (-s)*dtx + c*dty],
                           [s, -c, (-c)*dtx + (-s)*dty],
                           [0, 0, -1.0]])
            Jj = np.array([[c, s, 0.0], [-s, c, 0.0], [0, 0, 1.0]])
            JiT_Om, JjT_Om = Ji.T @ Om, Jj.T @ Om
            si, sj = 3 * i, 3 * j
            H[si:si+3, si:si+3] = H[si:si+3, si:si+3] + JiT_Om @ Ji
            H[si:si+3, sj:sj+3] = H[si:si+3, sj:sj+3] + JiT_Om @ Jj
            H[sj:sj+3, si:si+3] = H[sj:sj+3, si:si+3] + JjT_Om @ Ji
            H[sj:sj+3, sj:sj+3] = H[sj:sj+3, sj:sj+3] + JjT_Om @ Jj
            b[si:si+3] += JiT_Om @ e_vec
            b[sj:sj+3] += JjT_Om @ e_vec

        H_csr = H.tocsr()
        diag = H_csr.diagonal()
        lam = 1e-6 * (np.mean(np.abs(diag[diag != 0])) if np.any(diag != 0) else 1.0)
        H_csr = H_csr + sp.eye(3 * n, format="csr") * lam   # Levenberg-Marquardt damping

        H_lil = H_csr.tolil()                               # anchor: fix gauge per component
        for k in anchors:
            sk = 3 * k
            for r in (sk, sk + 1, sk + 2):
                H_lil[r, :] = 0; H_lil[:, r] = 0; H_lil[r, r] = 1.0; b[r] = 0.0
        H_final = H_lil.tocsr()

        dx, info = spla.cg(H_final, -b, rtol=1e-10)
        if info != 0:
            try:
                dx = spla.spsolve(H_final, -b)
            except Exception:
                print(f"  scipy solver failed at iter {it}, stopping.", file=sys.stderr)
                break
        x = x + dx
        for k in range(n):
            x[3*k+2] = float(_wrap(x[3*k+2]))
        iters_taken = it + 1
        if np.max(np.abs(dx)) < tol:
            break

    return x.reshape(n, 3), iters_taken, _objective(x, edges)


def main() -> None:
    """Harness entry point: read $LAB_DATA/graph.json, write $LAB_ARTIFACTS/trajectory.csv."""
    lab_data = os.environ.get("LAB_DATA"); lab_artifacts = os.environ.get("LAB_ARTIFACTS")
    if not (lab_data and lab_artifacts):
        sys.exit("LAB_DATA / LAB_ARTIFACTS not set — run this through the lodestar harness.")
    graph = json.loads((Path(lab_data) / "graph.json").read_text())
    poses, iters, obj = solve_posegraph(graph["nodes_init"], graph["edges"])
    out = Path(lab_artifacts) / "trajectory.csv"
    out.write_text("\n".join(f"{p[0]},{p[1]},{p[2]}" for p in poses) + "\n")
    rows = [r for r in csv.reader(open(out)) if r]
    assert len(rows) == len(poses) and not np.any(np.isnan(np.array(rows, float)))
    print(f"iterations={iters}  objective={obj:.6f}  rows={len(poses)}  nan_check=PASS")


if __name__ == "__main__":
    main()
