"""Fetch the pinned tSS-BO library into external/tSS-BO (git-ignored).

tSS-BO (github.com/PascalInn/tSS-BO, BSD-3) is a raw research repo (no setup.py),
so it is not pip-installable; we clone it and add it to sys.path at use time
(clay/.../tssbo_solver.py). Run once: `python tools/clay_optimize/fetch_tssbo.py`.
Deps (torch / gpytorch / botorch) install via pip separately.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_URL = "https://github.com/PascalInn/tSS-BO.git"
_DEST = Path(__file__).resolve().parents[2] / "external" / "tSS-BO"
_REV = "35d675a3b86a3d8cb539ef17f0916137e516511a"


def main() -> int:
    if (_DEST / "src" / "main.py").is_file():
        try:
            actual = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=_DEST, text=True
            ).strip()
        except (OSError, subprocess.SubprocessError):
            actual = ""
        if actual == _REV:
            print(f"tSS-BO {_REV} already present at {_DEST}")
            return 0
        print(
            f"existing {_DEST} is revision {actual or 'unknown'}, expected {_REV}; "
            "move it aside and rerun",
            file=sys.stderr,
        )
        return 1
    _DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"cloning {_URL} -> {_DEST}")
    rc = subprocess.call(["git", "clone", "--no-checkout", _URL, str(_DEST)])
    if rc == 0:
        rc = subprocess.call(["git", "checkout", "--detach", _REV], cwd=_DEST)
    if rc != 0 or not (_DEST / "src" / "main.py").is_file():
        print("clone failed", file=sys.stderr)
        return 1
    print(f"done: {_REV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
