"""terminal_access_on_conductor obligation checker.

Bug-template contract (the CC0/CR1 LVS open): a device terminal's routed access must
geometrically OVERLAP the device conductor it claims, on the connecting layer. If a MIM
cap plate's own-net routed metal does not cover the plate, the plate floats in extraction
and the cap will not pair in LVS -- exactly the gf180 OTA cap open.

This checker reads the canonical :class:`clay.layout.route.as_built.AsBuiltBundle`: the
net-tagged cap-plate conductors live in ``bundle.layout_model`` (rule 6 single model), the
routed wires in ``bundle.routed``. For every cap-plate conductor it measures the overlap
of its OWN net's routed metal ON THAT PLATE'S LAYER and returns a
:class:`clay.signoff_verdict.CheckerVerdict` (PASS if the access lands, FAIL with the
floating plate + measured overlap named). Independent of the router's own flags.
"""
from __future__ import annotations

from typing import Any, List

from clay.core_contract import VerificationStatus
from clay.signoff_verdict import CheckerVerdict

__all__ = ["terminal_access_verdicts"]

_NM = 1000.0


def terminal_access_verdicts(bundle: Any, *, min_overlap_um2: float = 0.5) -> List[CheckerVerdict]:
    """One verdict per cap-plate conductor: does its own net's routed metal overlap it on
    the plate's layer by >= ``min_overlap_um2``?"""
    import klayout.db as kdb

    model = getattr(bundle, "layout_model", None)
    routed = getattr(bundle, "routed", None)
    if model is None or routed is None:
        return []

    plates = [c for c in model.conductors if c.kind == "cap_plate" and c.net]

    def routed_region(net: str, layer) -> "kdb.Region":
        r = kdb.Region()
        nr = routed.nets.get(net)
        if nr is None:
            return r
        lk = tuple(layer)
        for lp in nr.wires:
            if (int(lp.layer), int(lp.datatype)) == lk:
                r.insert(kdb.Polygon([kdb.Point(int(v.x), int(v.y))
                                      for v in lp.polygon.vertices]))
        r.merge()
        return r

    # group plate conductors by (net, layer) so a multi-rect plate is measured as one.
    out: List[CheckerVerdict] = []
    seen = set()
    for c in plates:
        key = (c.net, tuple(c.layer))
        if key in seen:
            continue
        seen.add(key)
        group = [p for p in plates if (p.net, tuple(p.layer)) == key]
        plate_reg = kdb.Region()
        for p in group:
            plate_reg.insert(kdb.Box(int(p.x0 * _NM), int(p.y0 * _NM),
                                     int(p.x1 * _NM), int(p.y1 * _NM)))
        plate_reg.merge()
        rr = routed_region(c.net, c.layer)
        overlap_um2 = (plate_reg & rr).area() / _NM / _NM
        oid = f"terminal_access.{c.net}.{c.layer[0]}_{c.layer[1]}"
        if overlap_um2 >= min_overlap_um2:
            out.append(CheckerVerdict(
                checker="terminal_access", obligation_id=oid,
                status=VerificationStatus.PASS,
                evidence={"net": c.net, "layer": list(c.layer), "overlap_um2": overlap_um2},
                blocker_reason="", required=True,
            ))
        else:
            out.append(CheckerVerdict(
                checker="terminal_access", obligation_id=oid,
                status=VerificationStatus.FAIL,
                evidence={"net": c.net, "layer": list(c.layer), "overlap_um2": overlap_um2,
                          "min_overlap_um2": min_overlap_um2},
                blocker_reason=(
                    f"cap plate net={c.net} on layer={tuple(c.layer)} FLOATS: own-net routed "
                    f"metal overlap={overlap_um2:.3f}um2 < {min_overlap_um2}um2 -> plate not "
                    "joined to its net (LVS open)"
                ),
                required=True,
            ))
    return out
