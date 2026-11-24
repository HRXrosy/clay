"""R139 (BL-139B) — REAL gf180mcu DRC signoff harness.

Runs the actual GlobalFoundries ``gf180mcu.drc`` KLayout rule deck on a GDS
file via the installed KLayout binary in batch mode, then parses the resulting
``.lyrdb`` report database into a structured violation list.

Public entry points (see :mod:`tools.klayout_drc_signoff_v139.run`):

- :func:`run_gf180_drc` — end-to-end: GDS -> klayout batch -> parsed report.
- :func:`parse_lyrdb` — parse a ``.lyrdb`` XML report DB (pure, no binary).
- :func:`build_klayout_command` — build the exact klayout batch command line.
- :class:`Gf180DrcReport`, :class:`DrcViolation` — structured results.

No DRC result is fabricated: violations are read verbatim from the report DB
the foundry deck emits.
"""
from .run import (
    DEFAULT_DECK,
    DEFAULT_KLAYOUT_BIN,
    DrcViolation,
    Gf180DrcError,
    Gf180DrcReport,
    build_klayout_command,
    klayout_binary_available,
    parse_lyrdb,
    run_gf180_drc,
)

__all__ = [
    "DEFAULT_DECK",
    "DEFAULT_KLAYOUT_BIN",
    "DrcViolation",
    "Gf180DrcError",
    "Gf180DrcReport",
    "build_klayout_command",
    "klayout_binary_available",
    "parse_lyrdb",
    "run_gf180_drc",
]
