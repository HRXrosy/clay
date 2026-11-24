"""Private pytest cleanup hook for the suite-contract release verifier."""

from __future__ import annotations

import re
import shutil
from pathlib import Path


_NAME = re.compile(r"\.pytest-suite-contract-[1-9][0-9]*-[0-9a-f]{32}\Z")


def pytest_unconfigure(config) -> None:
    """Delete only the verifier-owned basetemp while the child still owns its ACL."""

    configured = getattr(getattr(config, "option", None), "basetemp", None)
    if configured is None:
        return
    invocation_dir = getattr(getattr(config, "invocation_params", None), "dir", None)
    if invocation_dir is None:
        raise RuntimeError("refusing cleanup without a repository invocation root")
    target = Path(configured).resolve()
    root = Path(invocation_dir).resolve()
    if (
        target.parent != root
        or _NAME.fullmatch(target.name) is None
    ):
        raise RuntimeError("refusing unsafe suite-contract basetemp cleanup")
    if target.exists():
        shutil.rmtree(target)
