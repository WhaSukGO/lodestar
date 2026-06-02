"""Bridge to the Touchstone verification spine (blueberry_ver2's `lab` package).

ver4 is the *world builder + SLAM oracle*; the verifier is the SAME constant spine from
ver2 (sandboxed run + held-out grading + anti-tamper grader restoration). We reuse it
rather than fork it — the whole thesis is "swap the solver/domain, keep the verifier."

Resolution order for the ver2 checkout: $TOUCHSTONE_PATH, else the sibling
../blueberry_ver2 next to this repo. Importing fails loudly if neither is found."""
from __future__ import annotations

import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[1]            # blueberry_ver4/
_CANDIDATES = [
    os.environ.get("TOUCHSTONE_PATH"),
    str(_REPO_ROOT.parent / "blueberry_ver2"),
]


def _locate() -> str:
    for c in _CANDIDATES:
        if c and (Path(c) / "lab" / "factory.py").exists():
            return str(Path(c).resolve())
    raise ImportError(
        "Touchstone spine (blueberry_ver2/lab) not found. Set $TOUCHSTONE_PATH to the "
        "ver2 checkout, or place it at ../blueberry_ver2.")


_VER2 = _locate()
if _VER2 not in sys.path:
    sys.path.insert(0, _VER2)

# Re-export the exact seam ver4 plugs into (same as vision_blobs used in ver2).
from lab.agents.implementer import ImplementationTask  # noqa: E402
from lab.factory import build_implementer_harness        # noqa: E402
from lab.models import (  # noqa: E402
    DatasetRef, ExperimentRecord, FrameworkSpec, Usage,
)
# Live solver seam: the multi-agent coding committee + the real Claude session runner. These
# pull in NO heavy deps at import time (claude_agent_sdk is lazy inside run_agent).
from lab.agents.code_committee import code_committee_author  # noqa: E402
from lab.agents.sdk import DEFAULT_MODEL, run_agent          # noqa: E402

__all__ = ["ImplementationTask", "build_implementer_harness", "DatasetRef",
           "ExperimentRecord", "FrameworkSpec", "Usage", "ver2_path",
           "code_committee_author", "run_agent", "DEFAULT_MODEL"]


def ver2_path() -> str:
    return _VER2
