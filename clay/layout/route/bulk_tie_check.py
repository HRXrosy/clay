"""BULK-TIE FIDELITY checker: is each s!=b MOS device's declared body net REALLY
tied to its drawn well-tap M1 island in the routed cell?

THE MISSING CONTRACT THIS NAMES (F2, emit/route side)
=====================================================
``build_device`` draws a LOCAL isolated well tap for every source != bulk PMOS
(``well_tap_isolated`` -- the tap M1 is the body net's OWN island, min-spaced from
the source rail; see ``test_pmos_well_tap_net_contract``). Drawing the island is
only HALF the tie: the declared body net must also REACH it (routed metal landing
on the island), or the island is dead artwork and the well it taps is biased by
nobody -- a 4-terminal deck (external_pdk/external_pdk) then extracts the device bulk from a
floating or foreign-tied well and LVS breaks on every s!=b device. gf180's
3-terminal deck cannot SEE bulk at all, so THIS checker is the only oracle there.

WHAT IT CHECKS (per s != b MOS device)
======================================
1. locate the device's tap M1 island: the stamp M1 island containing the bulk
   terminal's M1 access point -- computed with the SAME probe convention
   ``extract_pins`` uses (``_build_mos_cell`` + ``_terminal_m1_stamp`` snapped onto
   the placed stamp's own M1), so the checker and the router share ONE physical
   world-view of where the bulk pin is (no private geometry model);
2. measure the island's net attachment: every net whose ROUTED M1
   (``NetRoute.wires``) overlaps/touches the island (extractor merge physics:
   touching same-layer metal is one conductor), plus every net with a PIN landing
   inside the island (the welded-tap catch: a source-tied tap island contains the
   source pin). The device's OWN bulk pin is NOT counted as attachment evidence --
   a pin is a route TARGET, not a tie; only routed metal proves the tie.
3. verdict: attached == {declared body} -> pass; empty -> "unattached" (an open
   bulk tie IS a violation -- never a silent pass); foreign nets -> "wrong-net"
   (declared body absent) / "multi-net" (declared body present plus foreigners,
   i.e. a short onto the tap island).

FIVE-STATE HONESTY
==================
A device the checker cannot adjudicate is NEVER silently passed: it lands in
``skipped`` with a reason (no well drawn -- an s!=b NMOS rides the GLOBAL
substrate via ``connect_global``, there is no local island to check; no placed
stamp; probe build failure; no M1 access). ``clean`` therefore means "zero
violations among the ADJUDICATED devices"; callers must read ``skipped`` for the
undecidable account.

SCOPE / AUTHORITY
=================
NATIVE + ADVISORY, like ``check_cross_net_shorts`` / ``check_net_connectivity``:
it reads the routed wires + device stamps + pins, NOT the foundry deck, so it does
not model deck well connectivity (a merged well biased through a NEIGHBOUR's tap
is invisible here -- deliberately: the per-device obligation is that EACH declared
tie is real metal). Signoff still gates on the independent KLayout LVS
(``lvs_gate_route_cell``); on gf180 the deck is 3-terminal and blind to bulk, so
this checker is the only bulk-tie oracle -- its verdicts there are the
hidden-defect account made visible, not a gate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from clay.layout.placed_geometry import placed_coord_nm, placed_point_nm

_NM_PER_UM = 1000.0


@dataclass(frozen=True)
class BulkTieViolation:
    """One s!=b device whose declared body net is NOT faithfully tied to its
    drawn tap M1 island. ``attached_nets`` is the MEASURED net set the island is
    actually attached to (empty for "unattached")."""

    device: str
    declared_body_net: str
    kind: str              # "wrong-net" | "unattached" | "multi-net"
    attached_nets: tuple   # measured attached net set (empty when unattached)


@dataclass(frozen=True)
class BulkTieReport:
    """The honest bulk-tie account: ``clean`` covers ONLY the adjudicated devices
    (``checked``); every undecidable device is in ``skipped`` as ``(device,
    reason)`` -- an explicit account, never a silent pass."""

    clean: bool
    violations: tuple      # of BulkTieViolation
    checked: int           # s!=b devices actually adjudicated
    skipped: tuple         # (device, reason) explicit undecidable account


def check_bulk_tie_fidelity(
    routed: Any,
    tech: Any,
    *,
    geometry: Dict[str, Any],
    placement_um: Dict[str, Tuple[float, float]],
    pins_by_net: Dict[str, List[Any]],
    devices: Any,
) -> BulkTieReport:
    """Adjudicate the bulk tie of every s!=b MOS device (see module docstring).

    ``devices`` is any iterable of device views with ``.name`` / ``.kind`` /
    ``.W`` / ``.L`` / ``.M`` and a ROLE-keyed ``.nets`` (``source``/``drain``/
    ``gate``/``body``) -- the ``route_cell`` bridged-device shape. ``tech`` is a
    :class:`~clay.layout.tech.LayoutTech` or a PDK id string.
    """
    import klayout.db as kdb

    from clay.layout.extract.realize_render import resolve_device_fold_target
    from clay.layout.route.pins import (
        _build_mos_cell,
        _stamp_m1_boxes_um,
        _terminal_m1_stamp,
    )
    from clay.layout.tech import LayoutTech

    if not isinstance(tech, LayoutTech):
        tech = LayoutTech.resolve(str(tech))
    m1 = tuple(tech.metal_key(1))
    m1_key = (int(m1[0]), int(m1[1]))
    try:
        wk = tech.build_layer("nwell")
    except Exception:
        wk = None
    well_key = (int(wk[0]), int(wk[1])) if wk is not None else None

    # per-net routed M1 regions (nm) -- built once, reused for every device.
    net_m1: Dict[str, "kdb.Region"] = {}
    for net, nr in (getattr(routed, "nets", {}) or {}).items():
        reg = kdb.Region()
        for lp in getattr(nr, "wires", []) or []:
            if (int(lp.layer), int(lp.datatype)) != m1_key:
                continue
            pts = [kdb.Point(int(v.x), int(v.y)) for v in lp.polygon.vertices]
            if len(pts) >= 3:
                reg.insert(kdb.Polygon(pts))
        if not reg.is_empty():
            reg.merge()
            net_m1[str(net)] = reg

    violations: List[BulkTieViolation] = []
    skipped: List[Tuple[str, str]] = []
    checked = 0

    for d in devices or []:
        kind = str(getattr(d, "kind", "")).lower()
        if kind not in ("nmos", "pmos"):
            continue
        nets = getattr(d, "nets", {}) or {}
        src = str(nets.get("source", "") or "")
        body = str(nets.get("body", "") or nets.get("bulk", "") or "")
        name = str(getattr(d, "name", ""))
        if not name or not body:
            continue  # no declared body tie to adjudicate
        if not src:
            skipped.append((name, "no declared source net (s!=b undecidable)"))
            continue
        if src == body:
            continue  # self-bulk: the source-tied tap contract (parity-tested)

        geom = (geometry or {}).get(name)
        pos = (placement_um or {}).get(name)
        if geom is None or pos is None:
            skipped.append((name, "no placed stamp geometry"))
            continue
        pbl = getattr(geom, "polys_by_layer", {}) or {}
        if well_key is None or not pbl.get(well_key):
            # an s!=b NMOS draws no local tap (band_tap_mode "none"): its bulk
            # rides the GLOBAL substrate (deck connect_global), there is no
            # island to adjudicate natively -- explicit skip, never a pass.
            skipped.append((name, "stamp draws no well (substrate-bulk device; "
                                  "deck-global tie, unmeasurable natively)"))
            continue

        # bulk M1 access point -- the SAME probe convention extract_pins uses.
        try:
            rht = resolve_device_fold_target(
                tech, kind,
                float(getattr(d, "W", 0.0)) * 1e-6,
                float(getattr(d, "L", 0.0)) * 1e-6,
                int(round(float(getattr(d, "M", 1) or 1))),
            )
        except Exception:  # noqa: BLE001 - fold target is an optional refinement
            rht = None
        probe = _build_mos_cell(d, tech, row_height_target_um=rht)
        if probe is None:
            skipped.append((name, "device cell build failed (no bulk probe)"))
            continue
        if "b" not in (getattr(probe, "terminals", {}) or {}):
            skipped.append((name, "device cell exposes no bulk terminal"))
            continue
        snap_boxes = _stamp_m1_boxes_um(geom, m1) or None
        entry = _terminal_m1_stamp(probe, m1, snap_boxes_um=snap_boxes).get("b")
        if entry is None:
            skipped.append((name, "no M1 access at the bulk terminal"))
            continue
        (rx, ry), _dist, _inside = entry
        bx_nm = placed_coord_nm(rx, float(pos[0]))
        by_nm = placed_coord_nm(ry, float(pos[1]))

        # the tap M1 island: the placed stamp M1 island containing the access point.
        stamp_m1 = kdb.Region()
        for ring in pbl.get(m1_key, []) or []:
            pts = [
                kdb.Point(*placed_point_nm(float(x), float(y), pos))
                for (x, y) in ring
            ]
            if len(pts) >= 3:
                stamp_m1.insert(kdb.Polygon(pts))
        stamp_m1.merge()
        probe_box = kdb.Region(
            kdb.Box(bx_nm - 1, by_nm - 1, bx_nm + 1, by_nm + 1)
        )
        island: Optional["kdb.Region"] = None
        for poly in stamp_m1.each():
            r = kdb.Region(poly)
            if not (r & probe_box).is_empty():
                island = r
                break
        if island is None:
            skipped.append((name, "bulk access point lands on no stamp M1 island"))
            continue

        checked += 1
        attached: set = set()
        # 1) routed metal overlapping/touching the island (extractor merge physics:
        #    touching same-layer metal is ONE conductor -> `interacting`).
        for net, reg in net_m1.items():
            if not reg.interacting(island).is_empty():
                attached.add(net)
        # 2) pins landing inside the island (the welded-tap catch). The device's
        #    OWN bulk pin is NOT attachment evidence -- a pin is a route target.
        for net, pins in (pins_by_net or {}).items():
            if net in attached:
                continue
            for p in pins:
                if (
                    str(getattr(p, "device", "")) == name
                    and str(getattr(p, "terminal", "")) == "b"
                ):
                    continue
                pl = getattr(p, "layer", None)
                if pl is not None and (int(pl[0]), int(pl[1])) != m1_key:
                    continue
                px = round(float(p.xy_um[0]) * _NM_PER_UM)
                py = round(float(p.xy_um[1]) * _NM_PER_UM)
                pb = kdb.Region(kdb.Box(px - 1, py - 1, px + 1, py + 1))
                if not (island & pb).is_empty():
                    attached.add(str(net))
                    break

        if not attached:
            violations.append(
                BulkTieViolation(name, body, "unattached", ())
            )
        elif attached == {body}:
            pass  # the declared tie is real routed metal
        elif body in attached:
            violations.append(
                BulkTieViolation(name, body, "multi-net", tuple(sorted(attached)))
            )
        else:
            violations.append(
                BulkTieViolation(name, body, "wrong-net", tuple(sorted(attached)))
            )

    return BulkTieReport(
        clean=not violations,
        violations=tuple(violations),
        checked=checked,
        skipped=tuple(skipped),
    )


__all__ = ["BulkTieViolation", "BulkTieReport", "check_bulk_tie_fidelity"]
