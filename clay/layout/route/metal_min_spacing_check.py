"""METAL MIN-SPACING obligation -- an INDEPENDENT checker read from emitted GDS.

THE OBLIGATION
==============
Every routed metal polygon on layer k must hold edge-to-edge distance >= the PDK
``min_space(metal k)`` to any OTHER same-layer metal it does not merge with -- no
sub-min gap between two distinct conductors (wire segments, via landing pads,
stubs). A foundry deck's ``M{k}.2a`` (e.g. gf180 ``metal1.space(0.23um)``) fires on
any edge pair of the MERGED layer closer than the rule, so this checker names the
same fact as a checkable obligation -- a breakage is a RED verdict, never a silent
sub-min gap.

RELATIONAL (honest scope)
=========================
Unlike width/enclosure (a LOCAL property of one shape), min-spacing is RELATIONAL:
it is the gap BETWEEN two shapes. The foundry deck is net-agnostic -- it MERGES all
of layer k and runs ``metal.space(minw)`` on the merged result, so two shapes that
physically touch (same net, connected) merge into one polygon and never fire, while
two shapes that do NOT touch (whether different nets or a broken same-net run) fire
if closer than the rule. This checker measures EXACTLY that: it merges the layer and
runs the KLayout ``Region.space_check`` at the PDK ``min_space`` -- the identical
geometry the deck measures. A divergence between this checker and the foundry oracle
is a bug in this checker, never grounds to trust it over the oracle.

INDEPENDENCE (checker not the checked)
======================================
This checker does NOT consult the router's internal state. It reads the EMITTED GDS
(the canonical artifact every oracle reads), resolves the routing-metal GDS keys +
per-layer ``min_space`` from the PDK ``tech`` contract, and measures with the
KLayout region ``space_check``.

GENERIC, NO PER-PDK BRANCH
==========================
The metal layer set and the per-layer ``min_space`` come entirely from the PDK
(``tech.metal_key`` / ``tech.min_space``). A PDK with a different stack or a looser
min-space is checked by the same code with its own numbers -- zero ``if pdk == ...``
branch. This is the SAME rule source the deck consumes and the SAME source the
router's emit reads (``tech.min_space`` / ``min_space_nm``).

SCOPE (honest)
==============
This checks SPACING only (the ``M{k}.2a`` minimum-spacing family). It does NOT model
the wide-metal spacing family (``M{k}.2b`` -- a larger space to wide/long metal),
which carries a length+width predicate; a layout clean here can still carry a 2b
violation where a wide conductor is involved. Metal WIDTH (``M{k}.1``) and
via-enclosure are SEPARATE obligations with their own checkers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

_LayerKey = Tuple[int, int]


@dataclass
class MetalSpacingViolation:
    """One location where two same-layer metal shapes sit closer than ``min_space``.

    ``metal_name`` is the rule-container layer name (``"metal1"`` ...);
    ``metal_layer`` is the GDS key; ``required_nm`` is the rule min-space;
    ``xy_um`` is the violating-edge midpoint (um)."""

    metal_name: str
    metal_layer: _LayerKey
    required_nm: int
    xy_um: Tuple[float, float]


@dataclass
class MetalSpacingReport:
    """Result of :func:`check_metal_min_spacing` on one GDS.

    ``status`` is ``"pass"`` (no firing), ``"fail"`` (>=1 firing), or
    ``"unmeasurable"`` (no routing-metal layers present / klayout unavailable --
    never silently pass). ``violations`` lists every sub-min gap location.
    ``per_layer`` is ``{metal_name: count}``."""

    status: str = "unmeasurable"
    violations: List[MetalSpacingViolation] = field(default_factory=list)
    per_layer: Dict[str, int] = field(default_factory=dict)
    n_polys: int = 0
    reason: str = ""

    @property
    def count(self) -> int:
        return len(self.violations)

    @property
    def ok(self) -> bool:
        return self.status == "pass"


def _metal_layers(tech: Any, max_metal: int) -> List[Tuple[int, str, _LayerKey]]:
    """The ``(index, name, gds_key)`` of each routing metal the PDK declares,
    generically (``metal1`` .. ``metal{max_metal}``)."""
    out: List[Tuple[int, str, _LayerKey]] = []
    for n in range(1, max_metal + 1):
        name = f"metal{n}"
        try:
            key = tuple(tech.metal_key(n))
        except Exception:
            continue
        if key is None:
            continue
        out.append((n, name, key))  # type: ignore[arg-type]
    return out


def check_metal_min_spacing(
    gds_path: str,
    tech: Any,
    *,
    top_cell: Optional[str] = None,
    max_metal: int = 6,
) -> MetalSpacingReport:
    """Measure the metal-min-spacing obligation on ``gds_path`` against ``tech``.

    For each routing metal the PDK declares, MERGE all of that layer's polygons
    (the foundry deck runs the spacing rule on the merged layer, so touching
    same-net shapes coalesce and do not fire) and flag every edge pair of distinct
    polygons closer than ``min_space`` -- the KLayout ``Region.space_check`` idiom
    that mirrors the foundry ``metal.space(minw)`` (``M{k}.2a``) rule. Returns a
    :class:`MetalSpacingReport`.

    ``unmeasurable`` (never a silent pass) when klayout is unavailable or no
    routing-metal layers are present in the GDS.
    """
    try:
        import klayout.db as kdb
    except Exception as exc:  # pragma: no cover - klayout always present in CI
        return MetalSpacingReport(status="unmeasurable", reason=f"klayout import: {exc}")

    ly = kdb.Layout()
    try:
        ly.read(gds_path)
    except Exception as exc:
        return MetalSpacingReport(status="unmeasurable", reason=f"read {gds_path}: {exc}")

    if top_cell is not None:
        cell = ly.cell(top_cell)
        if cell is None:
            return MetalSpacingReport(
                status="unmeasurable", reason=f"top cell {top_cell!r} not found"
            )
    else:
        tops = ly.top_cells()
        if not tops:
            return MetalSpacingReport(status="unmeasurable", reason="no top cell")
        cell = tops[0]

    dbu = ly.dbu

    report = MetalSpacingReport(violations=[])
    examined_any = False

    for _n, name, key in _metal_layers(tech, max_metal):
        try:
            ms = tech.min_space(name)
        except Exception:
            ms = None
        if not ms:
            continue  # PDK declares no min-space for this layer -> nothing owed
        li = ly.find_layer(int(key[0]), int(key[1]))
        if li is None:
            continue
        reg = kdb.Region(cell.begin_shapes_rec(li))
        if reg.is_empty():
            continue
        reg.merge()  # the deck runs the space rule on the MERGED layer
        examined_any = True
        report.n_polys += reg.count()
        # FOUNDRY-FAITHFUL: ``metal.space(minw)`` -- the per-edge-pair external
        # spacing check. Each EdgePair marks a place two distinct merged polygons
        # are closer than ``min_space``; group by edge-midpoint so a count is a
        # location (comparable to a DRC instance).
        sc = reg.space_check(int(ms))
        seen: Dict[Tuple[int, int], None] = {}
        for ep in sc.each():
            e = ep.first
            mx = int(round((e.x1 + e.x2) / 2.0))
            my = int(round((e.y1 + e.y2) / 2.0))
            seen[(mx, my)] = None
        for (mx, my) in seen:
            report.violations.append(
                MetalSpacingViolation(
                    metal_name=name,
                    metal_layer=tuple(key),  # type: ignore[arg-type]
                    required_nm=int(ms),
                    xy_um=(mx * dbu, my * dbu),
                )
            )
            report.per_layer[name] = report.per_layer.get(name, 0) + 1

    if not examined_any:
        report.status = "unmeasurable"
        report.reason = report.reason or "no routing-metal layers present in GDS"
        return report
    report.status = "fail" if report.violations else "pass"
    return report


__all__ = [
    "MetalSpacingViolation",
    "MetalSpacingReport",
    "check_metal_min_spacing",
]
