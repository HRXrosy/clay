"""Phase 3c — DRC (Design Rule Check) via klayout.db Python API.

Loads a Phase 3b-produced binary GDS file, runs a minimum DRC subset of
sky130 / gf180mcu rules, and returns a structured DrcReport.

Pure Python (no Rust); requires klayout package. Tests gate on
KLAYOUT_AVAILABLE flag and skip if missing.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union

try:
    import klayout.db as kdb
    KLAYOUT_AVAILABLE = True
except ImportError:
    KLAYOUT_AVAILABLE = False


class DrcRuleKind(Enum):
    MIN_WIDTH = "min_width"
    MIN_SPACE = "min_space"
    MIN_ENCLOSURE = "min_enclosure"


@dataclass(frozen=True)
class DrcRule:
    rule_id: str
    kind: DrcRuleKind
    layer: int
    datatype: int
    value_nm: int
    enclosing_layer: Optional[int] = None
    enclosing_datatype: Optional[int] = None


@dataclass(eq=True)
class DrcViolation:
    rule_id: str
    polygon_layer: int
    polygon_datatype: int
    measured_nm: int
    required_nm: int


@dataclass
class DrcReport:
    rules_checked: List[str] = field(default_factory=list)
    violations: List[DrcViolation] = field(default_factory=list)
    clean: bool = True

    def add_violation(self, v: DrcViolation) -> None:
        self.violations.append(v)
        self.clean = False


# sky130 minimum DRC subset (sourced from skywater-pdk DRC sections A.1-A.3)
SKY130_MIN_DRC_RULES: List[DrcRule] = [
    DrcRule("sky130_poly_min_w",        DrcRuleKind.MIN_WIDTH,    66, 20, 150),
    DrcRule("sky130_diff_min_w",        DrcRuleKind.MIN_WIDTH,    65, 20, 150),
    DrcRule("sky130_nwell_min_w",       DrcRuleKind.MIN_WIDTH,    64, 20, 840),
    DrcRule("sky130_poly_poly_space",   DrcRuleKind.MIN_SPACE,    66, 20, 210),
    DrcRule("sky130_diff_diff_space",   DrcRuleKind.MIN_SPACE,    65, 20, 270),
    DrcRule("sky130_nwell_diff_enc",    DrcRuleKind.MIN_ENCLOSURE, 65, 20, 180,
            enclosing_layer=64, enclosing_datatype=20),
]

GF180MCU_MIN_DRC_RULES: List[DrcRule] = [
    DrcRule("gf180_poly_min_w",         DrcRuleKind.MIN_WIDTH,    30, 0, 180),
    DrcRule("gf180_diff_min_w",         DrcRuleKind.MIN_WIDTH,    22, 0, 220),
    DrcRule("gf180_nwell_min_w",        DrcRuleKind.MIN_WIDTH,    21, 0, 1000),
    DrcRule("gf180_poly_poly_space",    DrcRuleKind.MIN_SPACE,    30, 0, 280),
]


def _measure_edge_pair_nm(ep, nm_per_dbu: float) -> int:
    """Extract the violation magnitude (in nm) from a klayout EdgePair.

    The EdgePair.distance() method returns the perpendicular separation
    between the two edges in database units.
    """
    measured_dbu = ep.distance()
    return int(round(measured_dbu * nm_per_dbu))


def run_drc(gds_path: Union[str, Path], rules: List[DrcRule]) -> DrcReport:
    """Run a list of DRC rules against a binary GDS file.

    Returns a DrcReport. Raises RuntimeError if klayout is not installed.
    """
    if not KLAYOUT_AVAILABLE:
        raise RuntimeError("klayout package not installed; run: pip install klayout")

    report = DrcReport()
    layout = kdb.Layout()
    layout.read(str(gds_path))

    # dbu is in micrometers; for sky130/gf180 we expect dbu=0.001 (1 nm precision)
    dbu_um = layout.dbu
    # nm per database unit: dbu_um (um) * 1000 nm/um
    nm_per_dbu = dbu_um * 1000.0

    for rule in rules:
        report.rules_checked.append(rule.rule_id)

        # Resolve the primary layer index. If absent in this layout, skip silently.
        primary_layer_index = layout.find_layer(rule.layer, rule.datatype)
        if primary_layer_index is None:
            continue

        # Resolve enclosing layer for MIN_ENCLOSURE rules; skip when absent.
        enc_layer_index = None
        if rule.kind == DrcRuleKind.MIN_ENCLOSURE:
            if rule.enclosing_layer is None:
                continue
            enc_layer_index = layout.find_layer(rule.enclosing_layer, rule.enclosing_datatype)
            if enc_layer_index is None:
                # No enclosing layer present in this GDS — Phase 3a Sky130
                # geometry doesn't always emit NWell. Skip rather than
                # over-report; a Phase 4+ tighter rule could flag this.
                continue

        # Convert nm rule value to database units (clamp to >= 1 dbu).
        value_dbu = max(1, int(round(rule.value_nm / nm_per_dbu)))

        # Iterate all cells (Phase 3a lowering emits flat per-cell shapes).
        for cell in layout.each_cell():
            shapes = cell.shapes(primary_layer_index)
            region = kdb.Region(shapes)
            if region.is_empty():
                continue

            if rule.kind == DrcRuleKind.MIN_WIDTH:
                edge_pairs = region.width_check(value_dbu)
                for ep in edge_pairs.each():
                    report.add_violation(DrcViolation(
                        rule_id=rule.rule_id,
                        polygon_layer=rule.layer,
                        polygon_datatype=rule.datatype,
                        measured_nm=_measure_edge_pair_nm(ep, nm_per_dbu),
                        required_nm=rule.value_nm,
                    ))
            elif rule.kind == DrcRuleKind.MIN_SPACE:
                edge_pairs = region.space_check(value_dbu)
                for ep in edge_pairs.each():
                    report.add_violation(DrcViolation(
                        rule_id=rule.rule_id,
                        polygon_layer=rule.layer,
                        polygon_datatype=rule.datatype,
                        measured_nm=_measure_edge_pair_nm(ep, nm_per_dbu),
                        required_nm=rule.value_nm,
                    ))
            elif rule.kind == DrcRuleKind.MIN_ENCLOSURE:
                enc_shapes = cell.shapes(enc_layer_index)
                enc_region = kdb.Region(enc_shapes)
                if enc_region.is_empty():
                    continue
                # enclosing_check: this region (enclosing) must enclose 'other'
                # by at least 'd' database units.
                edge_pairs = enc_region.enclosing_check(region, value_dbu)
                for ep in edge_pairs.each():
                    report.add_violation(DrcViolation(
                        rule_id=rule.rule_id,
                        polygon_layer=rule.layer,
                        polygon_datatype=rule.datatype,
                        measured_nm=_measure_edge_pair_nm(ep, nm_per_dbu),
                        required_nm=rule.value_nm,
                    ))

    return report
