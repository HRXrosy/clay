"""The ``@no_route_under`` layout intent: MIM-footprint routing keep-out (E/V).

THE INTENT CHAIN (no engine-baked policy -- the architecture mandate):
  E  -- :func:`extract_no_route_under` measures, on the REFERENCE GDS, which
        routing-metal LEVELS the signed layout keeps clear under its MIM
        regions and emits intent entries (relative/boolean facts + evidence
        AREAS; never a coordinate). The same entries are designer-DECLARABLE on
        ``ExtractedIntent.no_route_under``.
  L  -- ``clay.layout.model.build_layout_model`` lowers the entries into hard
        keep-out :class:`~clay.layout.model.Conductor` rectangles over each
        matching cap's placed footprint; the router consumes the model's
        obstacle projection unchanged.
  V  -- :func:`foreign_metal_under_caps_um2` measures the obligation on the
        routed artwork (foreign metal area per level inside the footprints).

Measured reference evidence (2026-06-11, klayout direct, CTM5 regions of the
signed twoStageAmp): M1 = 0.0 um^2, M2 = 0.0 um^2, M4 = 0.0 um^2, M3 = 76.7
um^2 (~4.9% of the 1568 um^2 CTM area -> M3 is legitimately used). The
unconstrained Clay route ran 299.7 + 204.3 um^2 of M1+M2 straight under the
cap -- the visual "cap overlaps devices" anomaly.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

_Bbox = Tuple[float, float, float, float]

#: A metal level is FORBIDDEN when the reference uses it under the MIM for less
#: than this fraction of the total CTM area (0 measured for M1/M2/M4; M3 sits
#: at ~4.9%). A relative threshold, not an absolute area -- scale-free.
_FORBID_AREA_FRACTION = 0.01

#: CTM datatypes to scan on GDS layer 67 (CTM2..CTM5, external_pdk.layermap 98-101).
_CTM_LAYER = 67
_CTM_LEVELS = (2, 3, 4, 5)


def extract_no_route_under(ref_gds_path: str, *, tech_id: str = "gf180mcu",
                           ) -> List[Dict[str, Any]]:
    """Extract the ``@no_route_under`` intent from a reference GDS (E side).

    Returns intent entries ``{'cap': '*', 'forbidden_metals': [levels],
    'provenance': 'style-extracted', 'evidence': {level: area_um2, ...}}``.
    The payload carries LEVEL numbers and measured AREAS only -- no reference
    coordinate enters the intent (the leakage-safe boundary for style facts).
    Empty list when the reference draws no CTM region (honest degrade)."""
    import klayout.db as db

    from clay.layout.tech import LayoutTech

    tech = LayoutTech.resolve(tech_id)
    ly = db.Layout()
    ly.read(ref_gds_path)
    top = ly.top_cell()
    dbu = ly.dbu

    ctm = db.Region()
    bottom_levels: List[int] = []
    for lv in _CTM_LEVELS:
        li = ly.find_layer(_CTM_LAYER, lv)
        if li is None:
            continue
        r = db.Region(top.begin_shapes_rec(li))
        if r.is_empty():
            continue
        ctm += r
        bottom_levels.append(lv)
    ctm.merge()
    if ctm.is_empty():
        return []
    ctm_area = ctm.area() * dbu * dbu
    min_bottom = min(bottom_levels)

    evidence: Dict[int, float] = {}
    used: List[int] = []
    for level in range(1, min_bottom):
        try:
            mk = tuple(tech.metal_key(level))
        except Exception:
            continue
        li = ly.find_layer(*mk)
        area = 0.0
        if li is not None:
            r = db.Region(top.begin_shapes_rec(li))
            r.merge()
            area = (r & ctm).area() * dbu * dbu
        evidence[level] = round(area, 1)
        if area >= _FORBID_AREA_FRACTION * ctm_area:
            used.append(level)

    # EPISTEMICS: absence of use is NOT evidence of prohibition. The reference
    # ESTABLISHES a crossing level by actually using it (M3 here, 76.7 um^2);
    # only levels strictly BELOW the lowest established crossing level -- which
    # the reference demonstrably keeps at ~zero -- are extracted as FORBIDDEN.
    # Levels between the crossing level and the bottom plate that the reference
    # simply never needed (M4: 0.0 um^2 of demand) stay ALLOWED: forbidding
    # them on no evidence walled off the cap's OWN ascent towers (the V test's
    # 2 unroutable cap nets). If the reference uses NOTHING under the MIM, all
    # levels below the bottom plate are forbidden (clean prohibition evidence).
    cross = min(used) if used else min_bottom
    forbidden = [lv for lv in sorted(evidence) if lv < cross]

    return [{
        "cap": "*",
        "forbidden_metals": forbidden,
        "provenance": "style-extracted",
        "evidence": {f"M{k}_um2": v for k, v in evidence.items()},
    }]


def cap_footprints_um(intent: Any, placement: Dict[str, Tuple[float, float]],
                      geometry: Dict[str, Any], *, tech: Any = None,
                      ) -> Dict[str, _Bbox]:
    """Placed footprint bbox (um) per cap passive.

    Source priority (PIN/ARTWORK single-source discipline): the cap's stamp in
    ``geometry`` when present; else the SAME parametric cell the emit path
    builds (``build_capacitor_for_tech`` with the passive's m/model, extent
    centred on the placement -- exactly how ``lvs_gate`` stamps it). A cap with
    no placement, and no stamp AND no parametric cell, is skipped (honest
    absence)."""
    out: Dict[str, _Bbox] = {}
    for p in getattr(intent, "passives", []) or []:
        kind = str(getattr(p, "kind", "")).lower()
        if not (kind.startswith("cap") or kind == "c"):
            continue
        name = str(getattr(p, "name", ""))
        pos = placement.get(name)
        if pos is None:
            continue
        cg = geometry.get(name) if geometry else None
        if cg is not None:
            xs: List[float] = []
            ys: List[float] = []
            for rings in (getattr(cg, "polys_by_layer", {}) or {}).values():
                for ring in rings:
                    xs += [pos[0] + pt[0] for pt in ring]
                    ys += [pos[1] + pt[1] for pt in ring]
            if xs:
                out[name] = (min(xs), min(ys), max(xs), max(ys))
                continue
        if tech is None:
            continue
        try:
            from clay.layout import passive_device_cell as _pdc

            cell = _pdc.build_capacitor_for_tech(
                tech, float(getattr(p, "value", 0.0) or 0.0),
                nets=getattr(p, "nets", {}) or {},
                m=int(getattr(p, "M", 1) or 1),
                model=str(getattr(p, "model", "") or ""),
            )
        except Exception:
            continue  # no parametric cell on this PDK -> honest absence
        xs2 = [v.x for lp in cell.polygons for v in lp.polygon.vertices]
        ys2 = [v.y for lp in cell.polygons for v in lp.polygon.vertices]
        if not xs2:
            continue
        cx = (min(xs2) + max(xs2)) / 2.0
        cy = (min(ys2) + max(ys2)) / 2.0
        out[name] = (
            pos[0] + (min(xs2) - cx) / 1000.0,
            pos[1] + (min(ys2) - cy) / 1000.0,
            pos[0] + (max(xs2) - cx) / 1000.0,
            pos[1] + (max(ys2) - cy) / 1000.0,
        )
    return out


def matching_entries(intent: Any, cap_name: str) -> List[Dict[str, Any]]:
    """The ``no_route_under`` entries that apply to ``cap_name`` ('*' or exact)."""
    out = []
    for e in getattr(intent, "no_route_under", []) or []:
        tgt = str(e.get("cap", "*"))
        if tgt == "*" or tgt == cap_name:
            out.append(e)
    return out


def foreign_metal_under_caps_um2(
    intent: Any,
    placement: Dict[str, Tuple[float, float]],
    geometry: Dict[str, Any],
    routed: Any,
    *,
    tech_id: str = "gf180mcu",
) -> Dict[int, float]:
    """The V side: routed-wire area (um^2) per metal LEVEL inside the cap
    footprints, counting only nets that are not the cap's own terminals.
    Measured on the ROUTED artwork -- the obligation is 0 for forbidden levels."""
    import klayout.db as db

    from clay.layout.tech import LayoutTech

    tech = LayoutTech.resolve(tech_id)
    feet = cap_footprints_um(intent, placement, geometry, tech=tech)
    if not feet:
        return {}
    cap_nets = {
        str(n) for p in (getattr(intent, "passives", []) or [])
        if str(getattr(p, "kind", "")).lower().startswith("cap")
        for n in (getattr(p, "nets", {}) or {}).values()
    }
    foot_reg = db.Region()
    for (x0, y0, x1, y1) in feet.values():
        foot_reg.insert(db.Box(int(x0 * 1000), int(y0 * 1000),
                               int(x1 * 1000), int(y1 * 1000)))
    foot_reg.merge()

    out: Dict[int, float] = {}
    for level in range(1, 7):
        try:
            mk = tuple(tech.metal_key(level))
        except Exception:
            break
        reg = db.Region()
        for net, nr in (getattr(routed, "nets", {}) or {}).items():
            if net in cap_nets:
                continue
            for lp in getattr(nr, "wires", []) or []:
                if (int(lp.layer), int(lp.datatype)) != mk:
                    continue
                reg.insert(db.Polygon(
                    [db.Point(int(v.x), int(v.y)) for v in lp.polygon.vertices]))
        reg.merge()
        out[level] = round((reg & foot_reg).area() * 1e-6, 2)
    return out
