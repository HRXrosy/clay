"""METAL MIN-WIDTH obligation -- an INDEPENDENT checker read from emitted GDS.

THE OBLIGATION
==============
Every routed metal polygon on layer k must have width >= the PDK
``min_width(metal k)`` everywhere -- no sub-min notch, no sliver. This is the
positive contract the router owes for the metal it draws (wire segments, via
landing pads, stubs, bends): a foundry deck's ``M{k}.1`` (e.g. gf180
``metal1.width(0.23um)``) fires on any edge pair closer than the rule, so this
checker names the same fact as a checkable obligation -- a breakage is a RED
verdict, never a silent sub-min sliver.

INDEPENDENCE (checker not the checked)
======================================
This checker does NOT consult the router's internal state. It reads the EMITTED
GDS (the canonical artifact every oracle reads), resolves the routing-metal GDS
keys + per-layer ``min_width`` from the PDK ``tech`` contract, and measures with
the KLayout region ``width_check`` -- the SAME geometry the foundry ``M{k}.1``
width rule measures (``metal.width(minw, euclidian)``). A divergence between this
checker and the foundry oracle is a bug in this checker, never grounds to trust
it over the oracle.

GENERIC, NO PER-PDK BRANCH
==========================
The metal layer set and the per-layer ``min_width`` come entirely from the PDK
(``tech.metal_key`` / ``tech.min_width``). A PDK with a different stack or a
looser min-width is checked by the same code with its own numbers -- zero
``if pdk == ...`` branch. This is the SAME rule source the deck consumes
(``M{k}.1`` is emitted from the same ``metal{k}_width`` the synthesized deck reads
and the foundry deck declares).

SCOPE (honest)
==============
This checks WIDTH only (the ``M{k}.1`` family). Metal SPACING (``M{k}.2a/2b``)
and min-area (``M{k}.3``) are SEPARATE obligations with their own checkers; a
layout that is width-clean here can still carry spacing/area violations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

_LayerKey = Tuple[int, int]


@dataclass
class MetalWidthViolation:
    """One metal location whose width drops below the PDK ``min_width``.

    ``metal_name`` is the rule-container layer name (``"metal1"`` ...);
    ``metal_layer`` is the GDS key; ``required_nm`` is the rule min-width;
    ``xy_um`` is the violating-edge midpoint (um)."""

    metal_name: str
    metal_layer: _LayerKey
    required_nm: int
    xy_um: Tuple[float, float]


@dataclass
class MetalWidthReport:
    """Result of :func:`check_metal_min_width` on one GDS.

    ``status`` is ``"pass"`` (no firing), ``"fail"`` (>=1 firing), or
    ``"unmeasurable"`` (no routing-metal layers present / klayout unavailable --
    never silently pass). ``violations`` lists every sub-min edge location.
    ``per_layer`` is ``{metal_name: count}``."""

    status: str = "unmeasurable"
    violations: List[MetalWidthViolation] = field(default_factory=list)
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


def check_metal_min_width(
    gds_path: str,
    tech: Any,
    *,
    top_cell: Optional[str] = None,
    max_metal: int = 6,
) -> MetalWidthReport:
    """Measure the metal-min-width obligation on ``gds_path`` against ``tech``.

    For each routing metal the PDK declares, merge all of that layer's polygons
    (the foundry deck runs the width rule on the merged layer, not per-net) and
    flag every edge pair whose facing metal is closer than ``min_width`` -- the
    KLayout ``Region.width_check`` idiom that mirrors the foundry
    ``metal.width(minw, euclidian)`` rule. Returns a :class:`MetalWidthReport`.

    ``unmeasurable`` (never a silent pass) when klayout is unavailable or no
    routing-metal layers are present in the GDS.
    """
    try:
        import klayout.db as kdb
    except Exception as exc:  # pragma: no cover - klayout always present in CI
        return MetalWidthReport(status="unmeasurable", reason=f"klayout import: {exc}")

    ly = kdb.Layout()
    try:
        ly.read(gds_path)
    except Exception as exc:
        return MetalWidthReport(status="unmeasurable", reason=f"read {gds_path}: {exc}")

    if top_cell is not None:
        cell = ly.cell(top_cell)
        if cell is None:
            return MetalWidthReport(
                status="unmeasurable", reason=f"top cell {top_cell!r} not found"
            )
    else:
        tops = ly.top_cells()
        if not tops:
            return MetalWidthReport(status="unmeasurable", reason="no top cell")
        cell = tops[0]

    dbu = ly.dbu  # um per dbu

    report = MetalWidthReport(violations=[])
    examined_any = False

    for _n, name, key in _metal_layers(tech, max_metal):
        try:
            mw = tech.min_width(name)
        except Exception:
            mw = None
        if not mw:
            continue  # PDK declares no min-width for this layer -> nothing owed
        li = ly.find_layer(int(key[0]), int(key[1]))
        if li is None:
            continue
        reg = kdb.Region(cell.begin_shapes_rec(li))
        if reg.is_empty():
            continue
        reg.merge()
        examined_any = True
        report.n_polys += reg.count()
        # FOUNDRY-FAITHFUL: ``metal.width(minw, euclidian)`` -- the per-edge-pair
        # internal-width check. Each EdgePair marks a place the polygon is
        # narrower than ``min_width``; group by edge-midpoint so a count is a
        # location (comparable to a DRC instance).
        wc = reg.width_check(int(mw))
        seen: Dict[Tuple[int, int], None] = {}
        for ep in wc.each():
            e = ep.first
            mx = int(round((e.x1 + e.x2) / 2.0))
            my = int(round((e.y1 + e.y2) / 2.0))
            seen[(mx, my)] = None
        for (mx, my) in seen:
            report.violations.append(
                MetalWidthViolation(
                    metal_name=name,
                    metal_layer=tuple(key),  # type: ignore[arg-type]
                    required_nm=int(mw),
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
    "MetalWidthViolation",
    "MetalWidthReport",
    "check_metal_min_width",
]
