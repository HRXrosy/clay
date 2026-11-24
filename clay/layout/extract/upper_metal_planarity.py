"""UPPER-METAL PLANARITY: the per-level over-active fraction of the routed upper
metals (Intent -> Obligation -> Evidence -> Verdict).

WHY THIS IS AN INTENT, NOT A PATCH (the architecture mandate):
  The signed external_pdk reference routes its UPPER metals PLANAR -- M3 sits over active
  diffusion only ~13% of its area, M4 ~0%, M5/M6 ~0% -- because a sensitive
  high-impedance gain/output net carried on metal that floats directly OVER the
  device channels couples capacitively into those channels (degrading gain /
  adding parasitic load) and blocks the upper-metal channel for power routing.
  The reference reserves PLANAR channels (beside / between the device rows) for
  the long upper-metal runs and keeps M2 as the LOCAL interconnect (M2 over
  active ~66% is normal -- local metal must cross the cells it connects).

  Clay's current cp route instead lets the long trunks climb to M3/M4 and run
  straight OVER the device cluster (measured M3 ~40% / M4 ~58% over active), a
  pure over-cell route. That is LEGAL (DRC/LVS clean) but DIVERGES from the
  reference's planar style and is the over-cell coupling defect named here.

THE OBLIGATION (upper-metal-not-over-active):
  For every UPPER metal level (M3 and above -- M1/M2 are local interconnect and
  excluded), the routed metal on that level must sit MOSTLY OFF active diffusion:
  over_active_frac(level) = (routed metal area on the level intersecting device OD
  diffusion) / (total routed metal area on the level) <= a calibrated ceiling.

  The ceiling is RELATIVE (an area fraction) and per-level config-calibrated from
  the signed reference (M3 13% / M4..M6 ~0%) -- the reference CALIBRATES the band,
  never enters the measurement. A level with NO routed metal is unobservable (not
  a silent pass). M2 is INTENTIONALLY excluded (local-over-active is normal).

LEAKAGE BOUNDARY: every value is measured from the artifact-under-test's OWN
canonical bundle (the SAME placement/geometry/routed the LVS/DRC oracles ran on).
The reference NEVER enters a metric's input; the ceiling is a config scalar. The
metric is a RELATIVE ratio (over-area / total-area on a level), so a uniform
translation of the whole layout leaves every value identical -- a coordinate
cannot leak through a ratio of areas.

GDS layer: device OD/diffusion = 3 (production layermap).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from clay.layout.placed_geometry import placed_point_nm

GDS_OD = 3

# The lowest metal level that is UPPER (planar-channel) routing. M1/M2 are local
# interconnect (must cross the cells they connect); M3 and above are the long
# trunks the reference routes in planar channels. A config-overridable scalar,
# never a coordinate.
UPPER_METAL_MIN_LEVEL = 3


def _device_od_region(placement: Dict[str, Any], geometry: Dict[str, Any],
                      cap_names: set):
    """Merged device OD (diffusion, GDS layer 3) Region in the placed frame the
    emitter stamps. Excludes cap footprints (a MIM cap is not active)."""
    import klayout.db as kdb

    od = kdb.Region()
    for name, g in (geometry or {}).items():
        if name in cap_names:
            continue
        pos = placement.get(name)
        if pos is None:
            continue
        for lay, rings in (getattr(g, "polys_by_layer", {}) or {}).items():
            if int(lay[0]) != GDS_OD:
                continue
            for ring in rings:
                pts = [kdb.Point(*placed_point_nm(float(x), float(y), pos))
                       for (x, y) in ring]
                if len(pts) >= 3:
                    od.insert(kdb.Polygon(pts))
    od.merge()
    return od


def upper_metal_over_active(
    intent: Any,
    placement: Dict[str, Any],
    geometry: Dict[str, Any],
    routed: Any,
    *,
    tech: Any,
) -> Dict[int, Dict[str, float]]:
    """V side: per metal LEVEL, the over-active fraction of the routed metal.

    Returns ``{level: {'total_um2', 'over_um2', 'over_active_frac'}}`` for EVERY
    routed metal level (the checker selects the upper levels + their bands). Reads
    ONLY the canonical placement/geometry/routed (Clay's own output) and the PDK
    metal-key map -- never a reference coordinate. A level with no routed metal is
    omitted (unobservable, not a silent zero)."""
    import klayout.db as kdb

    cap_names = {str(p.name) for p in (getattr(intent, "passives", []) or [])
                 if str(getattr(p, "kind", "")).lower().startswith("c")}
    od = _device_od_region(placement, geometry, cap_names)

    # PDK metal (layer, datatype) -> level number (1-based). Never a hardcoded
    # GDS number: the tech map is the single source.
    metal_level: Dict[tuple, int] = {}
    for n in range(1, 12):
        try:
            metal_level[tuple(tech.metal_key(n))] = n
        except Exception:
            break

    per_total: Dict[int, float] = {}
    per_over: Dict[int, float] = {}
    nets = getattr(routed, "nets", {}) or {}
    for _net, nr in nets.items():
        for lp in (getattr(nr, "wires", []) or []):
            key = (int(getattr(lp, "layer")), int(getattr(lp, "datatype")))
            lvl = metal_level.get(key)
            if lvl is None:
                continue
            poly = getattr(lp, "polygon", None)
            pts = list(getattr(poly, "vertices", []) or []) if poly is not None else []
            if len(pts) < 3:
                continue
            reg = kdb.Region(kdb.Polygon(
                [kdb.Point(int(v.x), int(v.y)) for v in pts]))
            area = reg.area() * 1e-6
            over = (reg & od).area() * 1e-6
            per_total[lvl] = per_total.get(lvl, 0.0) + area
            per_over[lvl] = per_over.get(lvl, 0.0) + over

    out: Dict[int, Dict[str, float]] = {}
    for lvl in sorted(per_total):
        tot = per_total[lvl]
        if tot <= 0:
            continue
        ov = per_over.get(lvl, 0.0)
        out[lvl] = {
            "total_um2": round(tot, 2),
            "over_um2": round(ov, 2),
            "over_active_frac": round(ov / tot, 4),
        }
    return out
