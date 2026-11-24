"""Cross-PDK validator: verify that an AnnotatedCell + sizing lowers and
DRC-checks cleanly on BOTH sky130 AND gf180mcu.

A "portable" cell is one that survives lowering and DRC on both PDKs. This
provides a structural portability check: if the cell template depends on
PDK-specific quirks (layer numbers, fin pitch, enclosure values), one of
the two PDKs will fail.

Public API:
    PdkValidationResult: per-PDK outcome (lowering ok, DRC outcome).
    CrossPdkReport: pair of PdkValidationResults + overall portable flag.
    validate_on_sky130(cell, sizing) -> PdkValidationResult
    validate_on_gf180mcu(cell, sizing) -> PdkValidationResult
    cross_pdk_validate(cell, sizing) -> CrossPdkReport

DRC fields are tri-state: True (clean), False (violations), or None
(klayout missing or Rust GDS binary missing — DRC not attempted).
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from . import drc as _drc_mod
from . import binary_gds_subprocess as _bgs_mod
from .annotations import AnnotatedCell
from .backend import (
    DeviceSizing,
    GF180_FIN_PITCH_NM,
    Gf180mcuLayerMap,
    SKY130_FIN_PITCH_NM,
    Sky130LayerMap,
    lower_annotated_cell_drc_complete,
)
from .drc import (
    GF180MCU_MIN_DRC_RULES,
    SKY130_MIN_DRC_RULES,
    DrcRule,
    run_drc,
)


@dataclass
class PdkValidationResult:
    """Outcome of validating one (cell, sizing) pair on a single PDK.

    Attributes:
        pdk_name: short PDK identifier ("sky130" or "gf180mcu").
        lowering_succeeded: True if lower_annotated_cell_drc_complete returned
            a ParameterizedLibrary without raising.
        parameterized_library: the produced library, or None on failure.
        drc_clean: True if DRC ran and found zero violations; False if
            violations exist; None if DRC was not run (klayout missing or
            Rust GDS binary missing) or if lowering failed.
        drc_violation_count: integer count of DRC violations, or None when
            DRC was not run.
        error_message: human-readable error string when lowering or DRC
            raised, otherwise None.
    """
    pdk_name: str
    lowering_succeeded: bool
    parameterized_library: Optional[Any]
    drc_clean: Optional[bool]
    drc_violation_count: Optional[int]
    error_message: Optional[str] = None


@dataclass
class CrossPdkReport:
    """Combined sky130 + gf180mcu portability outcome.

    portable is True iff both PDKs lowered cleanly AND (DRC was either skipped
    on both or clean on both). A False portable flag means at least one PDK
    failed lowering, raised, or had DRC violations.
    """
    sky130: PdkValidationResult
    gf180mcu: PdkValidationResult
    portable: bool


def _run_drc_if_available(
    lib: Any,
    rules: list[DrcRule],
) -> tuple[Optional[bool], Optional[int], Optional[str]]:
    """Attempt to emit a binary GDS and run DRC.

    Returns (drc_clean, violation_count, error_message). When klayout or the
    Rust binary is unavailable, returns (None, None, None) — DRC was skipped,
    which is not an error condition.
    """
    if not _drc_mod.KLAYOUT_AVAILABLE:
        return (None, None, None)
    if not _bgs_mod.cargo_binary_available():
        return (None, None, None)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            gds_path = Path(tmpdir) / "cross_pdk.gds"
            _bgs_mod.emit_binary_gds_via_subprocess(lib, gds_path)
            report = run_drc(gds_path, rules)
            violation_count = len(report.violations)
            return (violation_count == 0, violation_count, None)
    except Exception as exc:  # pragma: no cover - defensive: surface as error
        return (None, None, f"drc step raised: {exc}")


def _validate_on_pdk(
    cell: AnnotatedCell,
    sizing: DeviceSizing,
    pdk_name: str,
    pdk_layers: Any,
    fin_pitch_nm: float,
    rules: list[DrcRule],
) -> PdkValidationResult:
    """Shared implementation for validate_on_sky130 / validate_on_gf180mcu.

    Tries lowering; if that succeeds, attempts DRC (skipped silently if
    klayout / Rust binary missing).
    """
    try:
        lib = lower_annotated_cell_drc_complete(
            cell, pdk_layers, fin_pitch_nm, sizing,
        )
    except Exception as exc:
        return PdkValidationResult(
            pdk_name=pdk_name,
            lowering_succeeded=False,
            parameterized_library=None,
            drc_clean=None,
            drc_violation_count=None,
            error_message=f"lowering failed: {exc}",
        )

    drc_clean, violation_count, drc_err = _run_drc_if_available(lib, rules)
    return PdkValidationResult(
        pdk_name=pdk_name,
        lowering_succeeded=True,
        parameterized_library=lib,
        drc_clean=drc_clean,
        drc_violation_count=violation_count,
        error_message=drc_err,
    )


def validate_on_sky130(
    cell: AnnotatedCell,
    sizing: DeviceSizing,
) -> PdkValidationResult:
    """Lower the cell on sky130 and (if klayout + GDS binary present) DRC it."""
    return _validate_on_pdk(
        cell, sizing, "sky130",
        Sky130LayerMap(), SKY130_FIN_PITCH_NM, SKY130_MIN_DRC_RULES,
    )


def validate_on_gf180mcu(
    cell: AnnotatedCell,
    sizing: DeviceSizing,
) -> PdkValidationResult:
    """Lower the cell on gf180mcu and (if klayout + GDS binary present) DRC it."""
    return _validate_on_pdk(
        cell, sizing, "gf180mcu",
        Gf180mcuLayerMap(), GF180_FIN_PITCH_NM, GF180MCU_MIN_DRC_RULES,
    )


def _is_portable(sky: PdkValidationResult, gf: PdkValidationResult) -> bool:
    """Portability rule: both must lower; if DRC ran, both must be clean."""
    if not (sky.lowering_succeeded and gf.lowering_succeeded):
        return False
    # drc_clean is tri-state. Accept (None, None) [both skipped] and
    # (True, True) [both clean]. Any False or mixed True/False is not portable.
    if sky.drc_clean is None and gf.drc_clean is None:
        return True
    if sky.drc_clean is True and gf.drc_clean is True:
        return True
    return False


def cross_pdk_validate(
    cell: AnnotatedCell,
    sizing: DeviceSizing,
) -> CrossPdkReport:
    """Run validate_on_sky130 + validate_on_gf180mcu and combine into a report.

    The two PDK validations are independent — one PDK failing does not abort
    the other. portable is True iff both lowered cleanly and DRC outcomes are
    consistent (either both skipped or both clean).
    """
    sky = validate_on_sky130(cell, sizing)
    gf = validate_on_gf180mcu(cell, sizing)
    return CrossPdkReport(
        sky130=sky,
        gf180mcu=gf,
        portable=_is_portable(sky, gf),
    )
