"""Entry-point shim so ``python -m tools.clay_optimize`` dispatches to run.main."""
from __future__ import annotations

from .run import main

raise SystemExit(main())
