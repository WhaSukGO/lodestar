"""Run the robustness suite: grade a solver across each rung's selectable environments.

Offline, no API. Examples:
  python -m lodestar.run_suite                 # honest solver, all rungs
  python -m lodestar.run_suite --rung 0        # one rung
  python -m lodestar.run_suite --solver degenerate
"""
from __future__ import annotations

import sys

from .scenarios import RUNGS
from .suite import format_table, run_suite


def main(rung: str = "all", solver: str = "honest") -> None:
    rungs = list(RUNGS) if rung == "all" else [rung]
    for r in rungs:
        rows = run_suite(r, solver=solver)
        print(format_table(r, solver, rows))
        print()
    print("Same verifier, same oracle threshold — only the WORLD changes per scenario. "
          "The honest\nsolver survives easy/default and is REJECTED where the environment "
          "breaks its assumptions\n(e.g. no loop closures, or noise beyond what loops can fix).")


if __name__ == "__main__":
    args = sys.argv[1:]
    rung = args[args.index("--rung") + 1] if "--rung" in args else "all"
    solver = args[args.index("--solver") + 1] if "--solver" in args else "honest"
    main(rung=rung, solver=solver)
