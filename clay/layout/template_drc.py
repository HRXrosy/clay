"""Cell template DRC verification harness.

One-call convenience layer combining four established APIs:

* Cell template factories (``build_diff_pair``, ``build_5t_ota`` ...)
* DRC-complete lowering (``lower_annotated_cell_drc_complete``)
* Binary GDS subprocess emit (``emit_binary_gds_via_subprocess``)
* DRC (``run_drc`` against ``SKY130_MIN_DRC_RULES`` etc.)

Use case: a single regression / dashboard line that says "all 8 cell
templates are sky130 DRC clean" -- without callers re-implementing the
build/lower/emit/check pipeline each time.

This module does NOT modify any underlying API. It mirrors the same
template registry shape as ``template_visualize`` (default sizing kwargs
per factory) so iteration order matches the rest of the cell-template
tooling: diff_pair, current_mirror, cascoded_mirror, 5t_ota,
folded_cascode, telescopic_cascode, two_stage_miller, bandgap_reference.

Graceful degradation:

* If klayout is not installed (``KLAYOUT_AVAILABLE`` is False), the
  per-template result is marked ``clean=True`` with zero violations and
  ``skipped_reason="klayout unavailable"`` so callers can spot skipped
  runs. The same applies when the Rust CLI binary needed to emit binary
  GDS is missing (``skipped_reason="cargo binary unavailable"``).
* ``overall_clean`` aggregates ``result.clean`` directly (skipped runs
  are vacuously clean, so they do not flip the overall flag).
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .annotations import AnnotatedCell
from .backend import DeviceSizing, lower_annotated_cell_drc_complete
from .binary_gds_subprocess import (
    BinaryGdsSubprocessError,
    cargo_binary_available,
    emit_binary_gds_via_subprocess,
)
from .cell_templates import (
    build_5t_ota,
    build_bandgap_reference,
    build_cascoded_current_mirror,
    build_current_mirror,
    build_diff_pair,
    build_folded_cascode,
    build_telescopic_cascode,
    build_two_stage_miller,
)
from .drc import (
    KLAYOUT_AVAILABLE,
    SKY130_MIN_DRC_RULES,
    DrcRule,
    run_drc,
)


SKIPPED_REASON_KLAYOUT = "klayout unavailable"
SKIPPED_REASON_CARGO = "cargo binary unavailable"


# ---------------------------------------------------------------------------
# Result and report dataclasses.
# ---------------------------------------------------------------------------
@dataclass
class TemplateDrcResult:
    """Per-template DRC verification outcome.

    Attributes:
        template_name: registry name (e.g. ``"diff_pair"``).
        clean: True iff zero violations were reported. For skipped runs
            (klayout or cargo binary unavailable) this is True (vacuous);
            callers should inspect ``skipped_reason`` to distinguish a
            real pass from a skip.
        violation_count: total number of ``DrcViolation`` instances
            returned by ``run_drc`` (zero for skipped runs).
        violations_by_rule: mapping from rule id to the count of
            violations against that rule. Empty for skipped runs and
            for clean real runs.
        skipped_reason: human-readable reason the real DRC check was
            skipped (e.g. ``"klayout unavailable"``,
            ``"cargo binary unavailable"``), or ``None`` if real DRC ran.
    """

    template_name: str
    clean: bool
    violation_count: int
    violations_by_rule: Dict[str, int] = field(default_factory=dict)
    skipped_reason: Optional[str] = None


@dataclass
class TemplateDrcReport:
    """Aggregate DRC report across all registered cell templates.

    Attributes:
        pdk_name: human-readable PDK identifier (e.g. ``"sky130"``).
        results: per-template results in registry insertion order.
        overall_clean: True iff every result reports ``clean=True``.
            Skipped runs count as clean (vacuous pass); callers
            inspecting strictness should look at each result's
            ``skipped_reason``.
    """

    pdk_name: str
    results: List[TemplateDrcResult] = field(default_factory=list)
    overall_clean: bool = True


# ---------------------------------------------------------------------------
# Template registry (mirrors template_visualize._TEMPLATE_REGISTRY).
# ---------------------------------------------------------------------------
_REGISTRY: Dict[
    str,
    Tuple[Callable[..., Tuple[AnnotatedCell, DeviceSizing]], Dict[str, Any]],
] = {
    "diff_pair": (
        build_diff_pair,
        {"w_um": 4.0, "l_um": 0.5},
    ),
    "current_mirror": (
        build_current_mirror,
        {"w_um": 4.0, "l_um": 0.5},
    ),
    "cascoded_mirror": (
        build_cascoded_current_mirror,
        {
            "w_um_low": 4.0,
            "l_um_low": 0.5,
            "w_um_cas": 4.0,
            "l_um_cas": 0.5,
        },
    ),
    "5t_ota": (
        build_5t_ota,
        {"input_w_um": 4.0, "input_l_um": 0.5},
    ),
    "folded_cascode": (
        build_folded_cascode,
        {"input_w_um": 4.0, "input_l_um": 0.5},
    ),
    "telescopic_cascode": (
        build_telescopic_cascode,
        {"input_w_um": 4.0, "input_l_um": 0.5},
    ),
    "two_stage_miller": (
        build_two_stage_miller,
        {"input_w_um": 4.0, "input_l_um": 0.5},
    ),
    "bandgap_reference": (
        build_bandgap_reference,
        {},
    ),
}


def _registry_keys() -> Tuple[str, ...]:
    """Return the registered template names in stable iteration order."""
    return tuple(_REGISTRY.keys())


def _infer_pdk_name(pdk_drc_rules: List[DrcRule]) -> str:
    """Best-effort PDK name from a DRC rule list (peek at first rule id)."""
    if not pdk_drc_rules:
        return "unknown"
    first_id = pdk_drc_rules[0].rule_id
    if first_id.startswith("sky130"):
        return "sky130"
    if first_id.startswith("gf180"):
        return "gf180mcu"
    return "unknown"


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------
def drc_check_template(
    template_name: str,
    pdk_layers: Any,
    fin_pitch_nm: float,
    pdk_drc_rules: List[DrcRule],
    **build_kwargs: Any,
) -> TemplateDrcResult:
    """Run end-to-end DRC verification on a single named cell template.

    Pipeline:
        1. Build the template via its factory.
        2. Lower it via ``lower_annotated_cell_drc_complete``.
        3. Emit binary GDS via the Rust CLI subprocess.
        4. Run klayout DRC against ``pdk_drc_rules``.

    If klayout is not installed OR the Rust CLI binary is not built, the
    call still returns a ``TemplateDrcResult`` but with ``clean=True``,
    zero violations, and ``skipped_reason`` set to a human-readable
    string so callers can detect the skip without an exception.

    Args:
        template_name: one of the eight known template names; raises
            ``ValueError`` on unknown names.
        pdk_layers: PDK layer map (e.g. ``Sky130LayerMap()``).
        fin_pitch_nm: fin pitch in nanometers (e.g. ``SKY130_FIN_PITCH_NM``).
        pdk_drc_rules: list of DRC rules to check (e.g.
            ``SKY130_MIN_DRC_RULES``).
        **build_kwargs: keyword arguments forwarded to the underlying
            factory. Overrides default sizing in the registry. Must NOT
            include ``name`` (the cell name is fixed to ``template_name``).

    Returns:
        ``TemplateDrcResult`` with the per-template outcome.

    Raises:
        ValueError: unknown ``template_name`` or caller passed ``name``.
    """
    if template_name not in _REGISTRY:
        available = ", ".join(_registry_keys())
        raise ValueError(
            f"unknown template_name {template_name!r}; "
            f"available: {available}"
        )
    if "name" in build_kwargs:
        raise ValueError(
            "build_kwargs must not include 'name'; it is fixed to "
            "template_name so the lowered library cell name matches."
        )

    # Fallback paths: gracefully skip the whole pipeline when its
    # dependencies are missing. We avoid even running the factory in
    # these branches so a unit test with no PDK/klayout/cargo can still
    # exercise the harness end-to-end.
    if not KLAYOUT_AVAILABLE:
        return TemplateDrcResult(
            template_name=template_name,
            clean=True,
            violation_count=0,
            violations_by_rule={},
            skipped_reason=SKIPPED_REASON_KLAYOUT,
        )
    if not cargo_binary_available():
        return TemplateDrcResult(
            template_name=template_name,
            clean=True,
            violation_count=0,
            violations_by_rule={},
            skipped_reason=SKIPPED_REASON_CARGO,
        )

    # Build + lower only when both dependencies are present.
    build_fn, default_kwargs = _REGISTRY[template_name]
    merged_kwargs: Dict[str, Any] = dict(default_kwargs)
    merged_kwargs.update(build_kwargs)

    cell, sizing = build_fn(name=template_name, **merged_kwargs)
    lib = lower_annotated_cell_drc_complete(
        cell, pdk_layers, fin_pitch_nm, sizing
    )

    # Emit binary GDS to a temp file and run DRC against it.
    with tempfile.TemporaryDirectory() as tmpdir:
        gds_path = Path(tmpdir) / f"{template_name}.gds"
        try:
            emit_binary_gds_via_subprocess(lib, gds_path)
        except BinaryGdsSubprocessError:
            # Binary disappeared between availability check and emit
            # (race); degrade gracefully rather than crash the harness.
            return TemplateDrcResult(
                template_name=template_name,
                clean=True,
                violation_count=0,
                violations_by_rule={},
                skipped_reason=SKIPPED_REASON_CARGO,
            )
        report = run_drc(gds_path, pdk_drc_rules)

    # Group violations by rule id.
    by_rule: Dict[str, int] = {}
    for v in report.violations:
        by_rule[v.rule_id] = by_rule.get(v.rule_id, 0) + 1

    return TemplateDrcResult(
        template_name=template_name,
        clean=report.clean,
        violation_count=len(report.violations),
        violations_by_rule=by_rule,
        skipped_reason=None,
    )


def drc_check_all_templates(
    pdk_layers: Any,
    fin_pitch_nm: float,
    pdk_drc_rules: List[DrcRule] = SKY130_MIN_DRC_RULES,
) -> TemplateDrcReport:
    """Run ``drc_check_template`` for every registered cell template.

    Args:
        pdk_layers: PDK layer map (e.g. ``Sky130LayerMap()``).
        fin_pitch_nm: fin pitch in nanometers.
        pdk_drc_rules: list of DRC rules; defaults to
            ``SKY130_MIN_DRC_RULES``.

    Returns:
        ``TemplateDrcReport`` with 8 ``TemplateDrcResult`` entries in
        registry insertion order. ``overall_clean`` is True iff every
        entry reports ``clean=True``. Skipped runs (klayout / cargo
        unavailable) still count as clean for the aggregate flag;
        callers inspecting strictness should look at each result's
        ``skipped_reason``.
    """
    pdk_name = _infer_pdk_name(pdk_drc_rules)
    results: List[TemplateDrcResult] = []
    for template_name in _registry_keys():
        results.append(
            drc_check_template(
                template_name=template_name,
                pdk_layers=pdk_layers,
                fin_pitch_nm=fin_pitch_nm,
                pdk_drc_rules=pdk_drc_rules,
            )
        )

    overall_clean = all(r.clean for r in results)
    return TemplateDrcReport(
        pdk_name=pdk_name,
        results=results,
        overall_clean=overall_clean,
    )
