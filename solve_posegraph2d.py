"""
2D Pose-Graph Optimizer using Gauss-Newton on SE(2).

Reads $LAB_DATA/graph.json and writes optimised trajectory to
$LAB_ARTIFACTS/trajectory.csv.

If LAB_DATA/LAB_ARTIFACTS are unset, falls back to a local temp directory
using the synthetic world from slamtest.worlds.posegraph2d (seed 0).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wrap(a: float | np.ndarray) -> float | np.ndarray:
    """Wrap angle(s) to (-pi, pi]."""
    return ((np.asarray(a) + np.pi) % (2 * np.pi)) - np.pi


def _objective(x: np.ndarray, edges: list) -> float:
    """Compute total weighted squared residual."""
    total = 0.0
    for e in edges:
        i, j = int(e[0]), int(e[1])
        dx, dy, dtheta = float(e[2]), float(e[3]), float(e[4])
        w_xy, w_theta = float(e[5]), float(e[6])

        xi, yi, ti = x[3*i], x[3*i+1], x[3*i+2]
        xj, yj = x[3*j], x[3*j+1]
        tj = x[3*j+2]

        c, s = np.cos(ti), np.sin(ti)
        dtx, dty = xj - xi, yj - yi
        pred_tx = c * dtx + s * dty
        pred_ty = -s * dtx + c * dty
        pred_r = tj - ti

        ex = pred_tx - dx
        ey = pred_ty - dy
        er = _wrap(pred_r - dtheta)

        total += w_xy * (ex**2 + ey**2) + w_theta * er**2
    return total


# ---------------------------------------------------------------------------
# Connected component / anchor detection (union-find)
# ---------------------------------------------------------------------------

def _find_components(n: int, edges: list) -> list[int]:
    """Return component label for each node (0-indexed); smaller = earlier root."""
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for e in edges:
        union(int(e[0]), int(e[1]))

    # normalise: each component root = smallest index in that component
    # (find() already does path compression, so just call find on every node)
    return [find(k) for k in range(n)]


# ---------------------------------------------------------------------------
# Main optimiser
# ---------------------------------------------------------------------------

def solve_posegraph(
    nodes_init: list,
    edges: list,
    max_iters: int = 50,
    tol: float = 1e-6,
) -> tuple[np.ndarray, int, float]:
    """
    Gauss-Newton pose-graph optimiser on SE(2).

    Parameters
    ----------
    nodes_init : list of [x, y, theta]
    edges      : list of [i, j, dx, dy, dtheta, w_xy, w_theta]
    max_iters  : maximum GN iterations
    tol        : convergence threshold on max(|dx|)

    Returns
    -------
    x          : (N, 3) optimised poses
    iters      : number of iterations taken
    obj        : final objective value
    """
    n = len(nodes_init)
    x = np.array(nodes_init, dtype=float).flatten()  # shape (3N,)

    # Find one anchor per connected component
    comp = _find_components(n, edges)
    # anchor_nodes: set of node indices that are the root (smallest in component)
    roots = sorted(set(comp))
    anchors = set()
    for r in roots:
        # The root returned by find() IS already the anchor node index
        anchors.add(r)

    iters_taken = 0
    for it in range(max_iters):
        # Build sparse H (3N x 3N) and dense b (3N,)
        H = sp.lil_matrix((3 * n, 3 * n), dtype=float)
        b = np.zeros(3 * n, dtype=float)

        for e in edges:
            i, j = int(e[0]), int(e[1])
            dx_meas = float(e[2])
            dy_meas = float(e[3])
            dtheta_meas = float(e[4])
            w_xy = float(e[5])
            w_theta = float(e[6])

            xi, yi, ti = x[3*i], x[3*i+1], x[3*i+2]
            xj, yj = x[3*j], x[3*j+1]
            tj = x[3*j+2]

            c, s = np.cos(ti), np.sin(ti)
            # R_i^T (world->local rotation at i)
            # R_i^T = [[c, s], [-s, c]]

            dtx = xj - xi
            dty = yj - yi

            pred_tx = c * dtx + s * dty
            pred_ty = -s * dtx + c * dty
            pred_r = tj - ti

            # Residual
            ex = pred_tx - dx_meas
            ey = pred_ty - dy_meas
            er = float(_wrap(pred_r - dtheta_meas))
            e_vec = np.array([ex, ey, er])

            # Information matrix
            Om = np.diag([w_xy, w_xy, w_theta])

            # Jacobian w.r.t. pose i  (3x3)
            # row 0: d(pred_tx)/d(xi,yi,ti)
            #   d/dxi = -c,  d/dyi = -s,  d/dti = -s*dtx + c*dty  (= d(R_i^T)/dti @ dt)[0]
            # row 1: d(pred_ty)/d(xi,yi,ti)
            #   d/dxi =  s,  d/dyi = -c,  d/dti = -c*dtx - s*dty
            # row 2: [0, 0, -1]
            Ji = np.array([
                [-c, -s, (-s) * dtx + c * dty],
                [ s, -c, (-c) * dtx + (-s) * dty],
                [ 0,  0, -1.0],
            ])

            # Jacobian w.r.t. pose j  (3x3)
            # row 0: [c, s, 0]
            # row 1: [-s, c, 0]
            # row 2: [0, 0, 1]
            Jj = np.array([
                [ c,  s, 0.0],
                [-s,  c, 0.0],
                [ 0,  0, 1.0],
            ])

            # Accumulate into H and b
            JiT_Om = Ji.T @ Om
            JjT_Om = Jj.T @ Om

            si, ei = 3 * i, 3 * i + 3
            sj, ej = 3 * j, 3 * j + 3

            H[si:ei, si:ei] = H[si:ei, si:ei] + JiT_Om @ Ji
            H[si:ei, sj:ej] = H[si:ei, sj:ej] + JiT_Om @ Jj
            H[sj:ej, si:ei] = H[sj:ej, si:ei] + JjT_Om @ Ji
            H[sj:ej, sj:ej] = H[sj:ej, sj:ej] + JjT_Om @ Jj

            b[si:ei] += JiT_Om @ e_vec
            b[sj:ej] += JjT_Om @ e_vec

        # Convert to CSR for efficient operations
        H_csr = H.tocsr()

        # Levenberg-Marquardt damping
        diag = H_csr.diagonal()
        mean_diag = np.mean(np.abs(diag[diag != 0])) if np.any(diag != 0) else 1.0
        lam = 1e-6 * mean_diag
        H_csr = H_csr + sp.eye(3 * n, format='csr') * lam

        # Anchor one node per connected component
        H_lil = H_csr.tolil()
        for anchor_k in anchors:
            sk = 3 * anchor_k
            H_lil[sk, :] = 0
            H_lil[sk+1, :] = 0
            H_lil[sk+2, :] = 0
            H_lil[:, sk] = 0
            H_lil[:, sk+1] = 0
            H_lil[:, sk+2] = 0
            H_lil[sk,   sk]   = 1.0
            H_lil[sk+1, sk+1] = 1.0
            H_lil[sk+2, sk+2] = 1.0
            b[sk]   = 0.0
            b[sk+1] = 0.0
            b[sk+2] = 0.0

        H_final = H_lil.tocsr()

        # Solve H dx = -b
        dx, info = spla.cg(H_final, -b, rtol=1e-10)
        if info != 0:
            # Fall back to direct solver for small systems
            try:
                dx = spla.spsolve(H_final, -b)
            except Exception:
                print(f"  Warning: solver failed at iter {it}, stopping.", file=sys.stderr)
                break

        x = x + dx
        # Wrap all theta values
        for k in range(n):
            x[3*k+2] = float(_wrap(x[3*k+2]))

        iters_taken = it + 1
        if np.max(np.abs(dx)) < tol:
            break

    obj = _objective(x, edges)
    return x.reshape(n, 3), iters_taken, obj


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    lab_data = os.environ.get("LAB_DATA", "")
    lab_artifacts = os.environ.get("LAB_ARTIFACTS", "")

    if lab_data and lab_artifacts:
        # Production mode: read from env
        graph_path = Path(lab_data) / "graph.json"
        out_dir = Path(lab_artifacts)
    else:
        # Fallback: generate synthetic world and use temp dirs
        print("LAB_DATA/LAB_ARTIFACTS not set — using synthetic world (seed=0)", file=sys.stderr)
        sys.path.insert(0, "/home/ws/devel/whasuk/blueberry_ver4")
        from slamtest.worlds.posegraph2d import _world

        tmpdir = Path(tempfile.mkdtemp(prefix="posegraph2d_"))
        data_dir = tmpdir / "data"
        out_dir = tmpdir / "artifacts"
        data_dir.mkdir(); out_dir.mkdir()

        gt, nodes, edges_list = _world(0)

        graph = {
            "nodes_init": [list(map(float, p)) for p in nodes],
            "edges": [list(map(float, e)) for e in edges_list],
        }
        graph_path = data_dir / "graph.json"
        graph_path.write_text(json.dumps(graph))

        # Save GT for ATE later
        gt_path = data_dir / "gt_poses.csv"
        gt_path.write_text("\n".join(f"{p[0]},{p[1]},{p[2]}" for p in gt))
        print(f"Synthetic data written to {tmpdir}", file=sys.stderr)

    # Load graph
    graph = json.loads(graph_path.read_text())
    nodes_init = graph["nodes_init"]
    edges = graph["edges"]
    N = len(nodes_init)

    print(f"Graph: {N} nodes, {len(edges)} edges", file=sys.stderr)

    # Run optimiser
    t0 = time.time()
    poses, iters, obj = solve_posegraph(nodes_init, edges, max_iters=50, tol=1e-6)
    elapsed = time.time() - t0

    print(f"Converged in {iters} iterations, {elapsed:.2f}s", file=sys.stderr)
    print(f"Final objective: {obj:.6f}", file=sys.stderr)

    # Write output CSV (no header, x,y,theta per line)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "trajectory.csv"
    lines = [f"{p[0]},{p[1]},{p[2]}" for p in poses]
    out_path.write_text("\n".join(lines) + "\n")

    print(f"Wrote {len(poses)} poses to {out_path}", file=sys.stderr)

    # Verify: check row count and no NaNs
    import csv
    with open(out_path) as f:
        rows = [r for r in csv.reader(f) if r]
    assert len(rows) == N, f"Expected {N} rows, got {len(rows)}"
    arr = np.array([[float(v) for v in r] for r in rows])
    assert not np.any(np.isnan(arr)), "NaNs detected in output!"
    assert arr.shape == (N, 3), f"Unexpected shape {arr.shape}"
    print(f"Verification passed: {N} rows, no NaNs, shape {arr.shape}", file=sys.stderr)

    # Optionally compute ATE if GT is available
    gt_path = Path(str(graph_path).replace("data", "data")).parent / "gt_poses.csv"
    if gt_path.exists():
        import csv as _csv
        with open(gt_path) as f:
            gt_rows = [r for r in _csv.reader(f) if r]
        gt_arr = np.array([[float(v) for v in r] for r in gt_rows])
        # SE(2)-aligned ATE
        P = arr[:, :2]; Q = gt_arr[:, :2]
        nm = min(len(P), len(Q)); P, Q = P[:nm], Q[:nm]
        mp, mq = P.mean(0), Q.mean(0)
        Pc, Qc = P - mp, Q - mq
        U, _, Vt = np.linalg.svd(Pc.T @ Qc)
        D = np.eye(2); D[1, 1] = np.sign(np.linalg.det(Vt.T @ U.T))
        R = Vt.T @ D @ U.T; t = mq - R @ mp
        aligned = (R @ P.T).T + t
        ate = float(np.sqrt(((aligned - Q) ** 2).sum(1).mean()))
        print(f"SE(2)-aligned ATE vs GT: {ate:.6f} m", file=sys.stderr)
    else:
        print("No GT file found — skipping ATE computation", file=sys.stderr)

    # Print summary to stdout
    print(f"iterations={iters}  objective={obj:.6f}  rows={N}  nan_check=PASS")
    return out_path


if __name__ == "__main__":
    main()
