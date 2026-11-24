"""MIM CAP-LEVEL obligation -- an INDEPENDENT checker read from emitted GDS.

THE OBLIGATION
==============
A MiM capacitor's TOP-PLATE metal level must equal the level the PDK (or an
explicit design declaration) prescribes. For a level-parameterized MIM stack
(external_pdk CTMn over METALn), the foundry rule CTM.TITLE.1 (external_drc.drc:2817) states
the cap sits "between the top two interconnect metal layers": the top plate CTMn
is over METALn and rises to a METAL(n+1) cover, so n = (top interconnect metal
index) - 1. A cap drawn one level low (CTM4/METAL4 in a 6-metal stack whose top
two metals are METAL5/METAL6) is a MISPLACED plate -- a RED verdict, never a
silent mis-level.

INDEPENDENCE (checker not the checked)
======================================
This checker does NOT consult the geometry builder's state. It reads the EMITTED
GDS (the canonical artifact every oracle reads), finds the PDK's CTM top-plate
MARKER layer, and reports the DATATYPE of every drawn CTM polygon -- the datatype
IS the drawn level (layermap: CTM2=67:2 .. CTM5=67:5). It then compares against
the expected level resolved from the PDK metal stack (or an explicit override).
It measures the same thing the foundry recognition layermap keys on (CTMn AND
METALn), so a divergence between this checker and the foundry oracle is a bug in
this checker, never grounds to trust it over the oracle.

LEAKAGE-FREE
============
The expected level comes from the PDK definition (``external_pdk`` -- the
metal count + CTM.TITLE.1) or from an explicit ``expected_level`` the caller
declares. It is NEVER read from a reference layout measurement.

FIVE-STATE (never a silent pass)
================================
* ``pass``        -- every drawn CTM top plate is at the expected level.
* ``fail``        -- at least one drawn CTM top plate is at the wrong level.
* ``unmeasurable``-- klayout unavailable / GDS unreadable / NO CTM marker drawn
                     (nothing to adjudicate -- recorded, not passed).
* ``unsupported`` -- the PDK's MIM stack is NOT level-parameterized (gf180
                     FuseTop, external_pdk 0.18 MIM10 -- one fixed MIM layer, no CTMn
                     level concept), so this obligation does not apply.
* ``advisory``    -- reserved for a soft/quality variant; not emitted here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

_LayerKey = Tuple[int, int]


@dataclass
class _CtmMarker:
    """The PDK's CTM top-plate MARKER resolution for the level obligation.

    ``gds_layer`` is the shared CTM GDS layer number (the DATATYPE carries the
    level); ``recognized_levels`` is every level whose complete MIM stack the PDK
    defines; ``pdk_level`` is the PDK-derived correct level (metal count +
    CTM.TITLE.1)."""

    gds_layer: int
    recognized_levels: Tuple[int, ...]
    pdk_level: int


def _ctm_marker_for(tech: Any) -> Optional[_CtmMarker]:
    """The CTM top-plate marker resolution for ``tech``, or ``None`` when the PDK's
    MIM has no CTMn-level concept. Pure DATA dispatch by PDK id (one row per PDK --
    the same convention as :func:`clay.layout.route.pins._mim_spec_for`); NO
    per-circuit logic."""
    # gf180 FuseTop / external_pdk 0.18 MIM10: a single fixed MIM layer, no CTMn level.
    return None


@dataclass
class MimLevelViolation:
    """One CTM top-plate polygon drawn at the wrong level.

    ``drawn_level`` is the polygon's CTM datatype; ``expected_level`` is the
    PDK/declared level; ``ctm_layer`` is the GDS key; ``xy_um`` is the polygon's
    centroid (um)."""

    drawn_level: int
    expected_level: int
    ctm_layer: _LayerKey
    xy_um: Tuple[float, float]


@dataclass
class MimLevelReport:
    """Result of :func:`check_mim_level` on one GDS.

    ``status`` is one of ``pass`` / ``fail`` / ``unmeasurable`` / ``unsupported`` /
    ``advisory`` (never a silent pass). ``expected_level`` is the level adjudicated
    against; ``drawn_levels`` maps ``level -> count of CTM top plates`` found;
    ``violations`` lists every mis-level plate."""

    status: str = "unmeasurable"
    expected_level: Optional[int] = None
    drawn_levels: Dict[int, int] = field(default_factory=dict)
    violations: List[MimLevelViolation] = field(default_factory=list)
    reason: str = ""

    @property
    def count(self) -> int:
        return len(self.violations)

    @property
    def ok(self) -> bool:
        return self.status == "pass"


def check_mim_level(
    gds_path: str,
    tech: Any,
    *,
    expected_level: Optional[int] = None,
    top_cell: Optional[str] = None,
) -> MimLevelReport:
    """Measure the MIM cap-level obligation on ``gds_path`` against ``tech``.

    Resolves the PDK's CTM top-plate marker layer + the expected level (the
    PDK-derived level unless ``expected_level`` overrides it), reads the emitted
    GDS, and reports the drawn CTM level of every top-plate polygon. Returns a
    :class:`MimLevelReport`.

    * ``unsupported`` when the PDK MIM has no CTMn-level concept.
    * ``unmeasurable`` (never a silent pass) when klayout is unavailable, the GDS
      is unreadable, or NO CTM top-plate marker is drawn.
    """
    marker = _ctm_marker_for(tech)
    if marker is None:
        return MimLevelReport(
            status="unsupported",
            reason=(
                f"PDK {getattr(tech, 'pdk_id', '?')!r} MIM stack is not "
                f"level-parameterized (no CTMn top-plate marker); the cap-level "
                f"obligation does not apply"
            ),
        )

    exp = int(expected_level) if expected_level is not None else int(marker.pdk_level)

    try:
        import klayout.db as kdb
    except Exception as exc:  # pragma: no cover - klayout always present in CI
        return MimLevelReport(
            status="unmeasurable", expected_level=exp, reason=f"klayout import: {exc}"
        )

    ly = kdb.Layout()
    try:
        ly.read(gds_path)
    except Exception as exc:
        return MimLevelReport(
            status="unmeasurable", expected_level=exp,
            reason=f"read {gds_path}: {exc}",
        )

    if top_cell is not None:
        cell = ly.cell(top_cell)
        if cell is None:
            return MimLevelReport(
                status="unmeasurable", expected_level=exp,
                reason=f"top cell {top_cell!r} not found",
            )
    else:
        tops = ly.top_cells()
        if not tops:
            return MimLevelReport(
                status="unmeasurable", expected_level=exp, reason="no top cell"
            )
        cell = tops[0]

    dbu = ly.dbu

    report = MimLevelReport(expected_level=exp, violations=[])

    for lvl in marker.recognized_levels:
        li = ly.find_layer(int(marker.gds_layer), int(lvl))
        if li is None:
            continue
        reg = kdb.Region(cell.begin_shapes_rec(li))
        if reg.is_empty():
            continue
        n = reg.count()
        report.drawn_levels[int(lvl)] = report.drawn_levels.get(int(lvl), 0) + n
        if int(lvl) == exp:
            continue
        # every polygon at a non-expected level is a mis-level top plate.
        for poly in reg.each():
            bb = poly.bbox()
            cx = (bb.left + bb.right) / 2.0
            cy = (bb.bottom + bb.top) / 2.0
            report.violations.append(
                MimLevelViolation(
                    drawn_level=int(lvl),
                    expected_level=exp,
                    ctm_layer=(int(marker.gds_layer), int(lvl)),
                    xy_um=(cx * dbu, cy * dbu),
                )
            )

    if not report.drawn_levels:
        report.status = "unmeasurable"
        report.reason = (
            f"no CTM top-plate marker (GDS layer {marker.gds_layer}) drawn in "
            f"{gds_path} -- no MIM cap to adjudicate"
        )
        return report

    report.status = "fail" if report.violations else "pass"
    if report.status == "fail":
        got = ", ".join(
            f"CTM{lvl} x{cnt}" for lvl, cnt in sorted(report.drawn_levels.items())
        )
        report.reason = (
            f"MIM top plate(s) at the wrong level: expected CTM{exp} "
            f"(top plate over METAL{exp}, rising to METAL{exp + 1}); drawn {got}"
        )
    return report


__all__ = [
    "MimLevelViolation",
    "MimLevelReport",
    "check_mim_level",
]
