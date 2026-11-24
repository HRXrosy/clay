"""R145 (BL-145A) — REAL gf180mcu LVS signoff harness package.

Drives the REAL GlobalFoundries ``gf180mcu.lvs`` KLayout rule deck (the
1923-line Ruby LVS runset shipped in the gf180mcu open PDK) against a layout
GDS + a reference SPICE netlist, using the installed KLayout binary in batch
mode, then parses the verdict (netlists match / don't match) plus the
extracted netlist's device/net topology.

See :mod:`klayout_lvs_signoff_v145.run` for the public API.
"""
from __future__ import annotations

from .run import (
    DEFAULT_DECK,
    DEFAULT_KLAYOUT_BIN,
    REPO_ROOT,
    Gf180LvsError,
    Gf180LvsReport,
    LvsMismatch,
    build_klayout_lvs_command,
    klayout_binary_available,
    lvs_deck_available,
    parse_extracted_netlist,
    parse_lvs_stdout,
    run_gf180_lvs,
)

__all__ = [
    "DEFAULT_DECK",
    "DEFAULT_KLAYOUT_BIN",
    "REPO_ROOT",
    "Gf180LvsError",
    "Gf180LvsReport",
    "LvsMismatch",
    "build_klayout_lvs_command",
    "klayout_binary_available",
    "lvs_deck_available",
    "parse_extracted_netlist",
    "parse_lvs_stdout",
    "run_gf180_lvs",
]
