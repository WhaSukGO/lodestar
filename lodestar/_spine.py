"""Bridge to the Touchstone verification spine (the `lab` package in WhaSukGO/touchstone).

Lodestar is the *world builder + SLAM oracle*; the verifier is the SAME constant spine from
Touchstone (sandboxed run + held-out grading + anti-tamper grader restoration). We reuse it
rather than fork it — the whole thesis is "swap the solver/domain, keep the verifier."

Resolution order for the Touchstone checkout: $TOUCHSTONE_PATH, else a sibling directory
named `touchstone` (the public repo, github.com/WhaSukGO/touchstone) or `blueberry_ver2`
(the author's local working dir). Importing fails loudly if none is found."""
from __future__ import annotations

import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[1]            # the lodestar repo root
_CANDIDATES = [
    os.environ.get("TOUCHSTONE_PATH"),
    str(_REPO_ROOT.parent / "touchstone"),       # the public repo, cloned as a sibling
    str(_REPO_ROOT.parent / "blueberry_ver2"),   # the author's local working dir
]


def _locate() -> str:
    for c in _CANDIDATES:
        if c and (Path(c) / "lab" / "factory.py").exists():
            return str(Path(c).resolve())
    raise ImportError(
        "Touchstone verifier (lab/) not found. Clone github.com/WhaSukGO/touchstone as a "
        "sibling directory (so it sits at ../touchstone), or set $TOUCHSTONE_PATH to its path.")


_VER2 = _locate()
if _VER2 not in sys.path:
    sys.path.insert(0, _VER2)

# Re-export the verifier seam — present on Touchstone's public master.
from lab.agents.implementer import ImplementationTask  # noqa: E402
from lab.factory import build_implementer_harness        # noqa: E402
from lab.models import (  # noqa: E402
    DatasetRef, ExperimentRecord, FrameworkSpec, Usage,
)

__all__ = ["ImplementationTask", "build_implementer_harness", "DatasetRef",
           "ExperimentRecord", "FrameworkSpec", "Usage", "ver2_path", "live_committee"]


def ver2_path() -> str:
    return _VER2


def live_committee():
    """Lazily import the live multi-agent solver bits (committee + Claude session runner).

    These live in Touchstone's `code_committee` (currently on the coding-solver branch), so
    they are OPTIONAL — only the `run_committee_live` demo needs them. The offline rungs,
    suite, and viz never touch this, so a plain `touchstone` master checkout runs everything
    except the live demo. Returns (code_committee_author, run_agent, DEFAULT_MODEL)."""
    from lab.agents.code_committee import code_committee_author
    from lab.agents.sdk import DEFAULT_MODEL, run_agent
    return code_committee_author, run_agent, DEFAULT_MODEL
