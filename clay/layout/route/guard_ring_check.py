"""GUARD-RING MATERIALIZATION checker: is each DECLARED ``@guard_ring`` cluster
really drawn as a CLOSED tap ring, tied to the declared net, on the declared
well polarity -- in the EMITTED artifact?

THE CONTRACT THIS NAMES
=======================
A ``@guard_ring(group=(...), well=<p|n>, tie=<net>)`` declaration is an
OBLIGATION: the named device cluster MUST end up wrapped in one continuous
tap-diffusion ring, strapped to ``tie``. ``route_cell`` materializes it through
``clay.layout.extract.realize_render.group_guard_rings`` (the PDK-rule builder)
+ a ring-tap Pin so the tie net reaches the ring. Drawing the ring is only HALF:
the ring must (1) be PRESENT, (2) form ONE CLOSED frame enclosing the members,
(3) be TIED to the declared net (routed metal, not dead artwork), and (4) draw
the declared well polarity. This checker adjudicates all four against the
EMITTED geometry + routed metal -- never the builder's own success record.

WHAT IT CHECKS (per DECLARED ring)
==================================
1. locate the drawn ring: among the guard-ring geometry stamps, the one whose M1
   frame outer bbox ENCLOSES the union of the members' placed stamp bboxes and
   whose M1 material does NOT overlap that union (members sit in the ring HOLE);
2. closed frame: the ring's M1 region merges to a donut (>= 1 hole) around the
   members -- else ``ring-open`` (a broken frame is not a guard ring);
3. tap net (union-find, extractor merge physics -- the SAME physics
   ``bulk_tie_check`` uses): every net whose ROUTED M1 overlaps/touches the ring
   frame, plus every FOREIGN pin landing on it. attached == {tie} -> tied; empty
   -> ``untied`` (dead artwork); tie absent -> ``wrong-net``; tie present plus
   foreigners -> ``multi-net`` (a short onto the ring);
4. polarity: the drawn well (nwell present -> ``n``, absent -> ``p``) must equal
   the declared ``well`` -- else ``well-mismatch``.

FIVE-STATE HONESTY (no silent pass)
===================================
A declared ring the checker cannot adjudicate is NEVER silently passed: a ring
whose members have no placed stamp lands in ``skipped`` (reason: no placement);
an ``well='n'`` ring on a PDK that draws no nwell is ``unsupported`` (the builder
cannot draw it -- explicit, never a fabricated pass). ``clean`` therefore means
"zero violations among the ADJUDICATED rings".

SCOPE / AUTHORITY
=================
NATIVE + ADVISORY, like ``check_cross_net_shorts`` / ``check_bulk_tie_fidelity``:
it reads the emitted geometry + routed wires + pins, NOT the foundry deck, so it
does not model deck well connectivity. Signoff still gates on the independent
KLayout LVS / real-deck DRC; this is the native early-warning + the ONLY
bulk-ring oracle on a 3-terminal deck (gf180) that cannot see bulk at all.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from clay.layout.placed_geometry import placed_coord_nm, placed_point_nm

_NM_PER_UM = 1000.0

# the four adjudicated FAILURE states (a ring the checker decided is broken).
# "skipped" / "unsupported" are honest gray, NOT violations.
_VIOLATION_STATES = frozenset(
    {"ring-absent", "ring-open", "untied", "wrong-net", "multi-net", "well-mismatch"}
)


@dataclass(frozen=True)
class GuardRingVerdict:
    """One declared ring's adjudication against the emitted artifact.

    ``status`` is one of ``pass`` | the six violation states above |
    ``skipped`` | ``unsupported``. ``drawn_well`` is the MEASURED polarity
    (``p`` / ``n`` / ``""`` when no ring found); ``attached_nets`` is the MEASURED
    tap-net set (extractor union-find over the ring frame)."""

    members: Tuple[str, ...]
    well_declared: str
    tie_declared: str
    status: str
    drawn_well: str = ""
    attached_nets: Tuple[str, ...] = ()
    reason: str = ""


@dataclass(frozen=True)
class GuardRingReport:
    """The honest guard-ring account: ``clean`` covers ONLY the adjudicated rings;
    every undecidable ring is in ``results`` with a ``skipped`` / ``unsupported``
    status + reason -- an explicit account, never a silent pass."""

    clean: bool
    results: Tuple[GuardRingVerdict, ...] = ()
    checked: int = 0
    skipped: Tuple[Tuple[Tuple[str, ...], str], ...] = ()


def _placed_cluster_index(
    geometry: Dict[str, Any],
    placement_um: Dict[str, Tuple[float, float]],
) -> Any:
    """Build the shared ownership account from emitted-coordinate stamp boxes."""
    from clay.layout.placed_cluster_index import PlacedClusterIndex

    boxes: Dict[str, Tuple[int, int, int, int]] = {}
    composites: Dict[str, Tuple[str, ...]] = {}
    for name, cg in (geometry or {}).items():
        pos = (placement_um or {}).get(name)
        if cg is None or pos is None:
            continue
        pbl = getattr(cg, "polys_by_layer", {}) or {}
        # Match _stamp_layer_region / emitted geometry exactly: quantize each
        # composed placed vertex, then take its bbox.  floor/ceil of float extrema
        # expands the observed artifact and can falsely fail ring enclosure.
        xs = [
            placed_coord_nm(float(x), float(pos[0]))
            for rings in pbl.values()
            for ring in rings
            for (x, _y) in ring
        ]
        ys = [
            placed_coord_nm(float(y), float(pos[1]))
            for rings in pbl.values()
            for ring in rings
            for (_x, y) in ring
        ]
        if not xs or not ys:
            continue
        boxes[str(name)] = (min(xs), min(ys), max(xs), max(ys))
        composite_members = tuple(
            str(member) for member in (getattr(cg, "composite_members", ()) or ())
        )
        if composite_members:
            composites[str(name)] = composite_members

    return PlacedClusterIndex.build(
        boxes=boxes,
        composite_members=composites,
    )


def _member_union_bbox_nm(
    members: Sequence[str],
    geometry: Dict[str, Any],
    placement_um: Dict[str, Tuple[float, float]],
    *,
    cluster_index: Any = None,
) -> Optional[Tuple[int, int, int, int]]:
    """Absolute-nm union bbox for declaration-level ``members``.

    The final geometry may contain one composite/array stamp instead of the
    member-named stamps.  Resolve ownership through ``PlacedClusterIndex`` so
    every checker consumes the same final-stamp provenance account.
    """
    index = cluster_index
    if index is None:
        index = _placed_cluster_index(geometry, placement_um)
    return index.union_for_members(tuple(str(member) for member in members))


def _stamp_layer_region(cg: Any, key: Tuple[int, int], pos: Tuple[float, float]):
    """kdb.Region (absolute nm, merged) of one layer of a placed geometry stamp."""
    import klayout.db as kdb

    reg = kdb.Region()
    for ring in (getattr(cg, "polys_by_layer", {}) or {}).get(key, []) or []:
        pts = [
            kdb.Point(*placed_point_nm(float(x), float(y), pos))
            for (x, y) in ring
        ]
        if len(pts) >= 3:
            reg.insert(kdb.Polygon(pts))
    reg.merge()
    return reg


def check_guard_ring_materialized(
    routed: Any,
    tech: Any,
    *,
    geometry: Dict[str, Any],
    placement_um: Dict[str, Tuple[float, float]],
    pins_by_net: Dict[str, List[Any]],
    guard_ring_intents: Sequence[Any],
) -> GuardRingReport:
    """Adjudicate every DECLARED guard ring against the emitted artifact.

    ``guard_ring_intents`` is any iterable of declared-ring views with ``.members``
    (device labels), ``.well`` (``p`` / ``n``) and ``.tie_net`` (``str``) -- the
    ``clay.layout.intent.GuardRingIntent`` shape. ``tech`` is a
    :class:`~clay.layout.tech.LayoutTech` or a PDK id string.
    """
    import klayout.db as kdb

    from clay.layout.tech import LayoutTech

    if not isinstance(tech, LayoutTech):
        tech = LayoutTech.resolve(str(tech))
    m1 = tuple(tech.metal_key(1))
    m1_key = (int(m1[0]), int(m1[1]))
    try:
        nw = tech.build_layer("nwell")
    except Exception:
        nw = None
    nwell_key = (int(nw[0]), int(nw[1])) if nw is not None else None

    # per-net routed M1 region (nm) -- built once, reused for every ring.
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

    # the drawn ring stamps (kind == "guard_ring") -- the EMITTED ring artwork.
    ring_stamps: List[Tuple[str, Any, Tuple[float, float]]] = []
    for name, cg in (geometry or {}).items():
        if str(getattr(cg, "kind", "")) != "guard_ring":
            continue
        pos = (placement_um or {}).get(name, (0.0, 0.0))
        ring_stamps.append((name, cg, pos))

    # Geometry ownership is independent of the individual declaration.  Build
    # it once for the whole report (the external_pdk OTA currently carries nine rings).
    placed_clusters = _placed_cluster_index(geometry, placement_um)

    results: List[GuardRingVerdict] = []
    skipped: List[Tuple[Tuple[str, ...], str]] = []
    checked = 0

    for gr in guard_ring_intents or []:
        members = tuple(str(m) for m in (getattr(gr, "members", ()) or ()))
        well_decl = str(getattr(gr, "well", "") or "")
        tie_decl = str(getattr(gr, "tie_net", "") or "")

        inner = _member_union_bbox_nm(
            members,
            geometry,
            placement_um,
            cluster_index=placed_clusters,
        )
        if inner is None:
            reason = "no member has a placed stamp (undecidable)"
            skipped.append((members, reason))
            results.append(GuardRingVerdict(
                members, well_decl, tie_decl, "skipped", reason=reason))
            continue

        if well_decl == "n" and nwell_key is None:
            reason = f"{tech.pdk_id}: PDK draws no nwell -> n-ring unsupported"
            skipped.append((members, reason))
            results.append(GuardRingVerdict(
                members, well_decl, tie_decl, "unsupported", reason=reason))
            continue

        inner_reg = kdb.Region(kdb.Box(inner[0], inner[1], inner[2], inner[3]))

        # locate the drawn ring: the guard-ring stamp whose M1 frame ENCLOSES the
        # members' union (outer bbox contains it) and does NOT overlap it (members
        # in the hole). Choose the TIGHTEST such frame (smallest outer bbox).
        best: Optional[Tuple[Any, Tuple[float, float], "kdb.Region",
                             Tuple[int, int, int, int]]] = None
        for _name, cg, pos in ring_stamps:
            ring_m1 = _stamp_layer_region(cg, m1_key, pos)
            if ring_m1.is_empty():
                continue
            bb = ring_m1.bbox()
            outer = (bb.left, bb.bottom, bb.right, bb.top)
            encloses = (
                outer[0] <= inner[0] and outer[1] <= inner[1]
                and outer[2] >= inner[2] and outer[3] >= inner[3]
            )
            if not encloses:
                continue
            if not (ring_m1 & inner_reg).is_empty():
                continue  # members overlap the ring metal -> not the enclosing hole
            area = (outer[2] - outer[0]) * (outer[3] - outer[1])
            if best is None or area < (
                (best[3][2] - best[3][0]) * (best[3][3] - best[3][1])
            ):
                best = (cg, pos, ring_m1, outer)

        if best is None:
            checked += 1
            results.append(GuardRingVerdict(
                members, well_decl, tie_decl, "ring-absent",
                reason="no closed M1 tap ring encloses the cluster members"))
            continue

        checked += 1
        ring_cg, ring_pos, ring_m1, _outer = best

        # (2) CLOSED FRAME: the ring M1 merges to a donut (>= 1 hole).
        holes = sum(p.holes() for p in ring_m1.each())
        if holes < 1:
            results.append(GuardRingVerdict(
                members, well_decl, tie_decl, "ring-open",
                reason=f"M1 ring is not a closed frame (holes={holes})"))
            continue

        # (4) POLARITY: drawn well from the emitted nwell material on this ring.
        drawn_well = "p"
        if nwell_key is not None:
            ring_nw = _stamp_layer_region(ring_cg, nwell_key, ring_pos)
            if not ring_nw.is_empty():
                drawn_well = "n"

        # (3) TAP NET (union-find): routed metal touching the ring frame + FOREIGN
        # pins landing on it (the ring's OWN injected tap pin is NOT tie evidence --
        # a pin is a route target; only routed metal / a foreign device pin proves
        # the tie). Mirrors bulk_tie_check's island-attachment physics.
        attached: set = set()
        for net, reg in net_m1.items():
            if not reg.interacting(ring_m1).is_empty():
                attached.add(net)
        for net, pins in (pins_by_net or {}).items():
            if net in attached:
                continue
            for p in pins:
                dev = str(getattr(p, "device", ""))
                if dev.startswith("GRING") or dev == "__ring_tap__":
                    continue  # the ring's own tap pin is not tie evidence
                pl = getattr(p, "layer", None)
                if pl is not None and (int(pl[0]), int(pl[1])) != m1_key:
                    continue
                px = int(round(float(p.xy_um[0]) * _NM_PER_UM))
                py = int(round(float(p.xy_um[1]) * _NM_PER_UM))
                pb = kdb.Region(kdb.Box(px - 1, py - 1, px + 1, py + 1))
                if not (ring_m1 & pb).is_empty():
                    attached.add(str(net))
                    break

        attached_t = tuple(sorted(attached))

        if drawn_well != well_decl:
            results.append(GuardRingVerdict(
                members, well_decl, tie_decl, "well-mismatch", drawn_well,
                attached_t,
                reason=f"drawn well {drawn_well!r} != declared {well_decl!r}"))
            continue

        # tap-net obligation: only when a tie was declared.
        if tie_decl:
            if not attached:
                status = "untied"
                reason = "ring frame reached by NO routed net (dead artwork)"
            elif attached == {tie_decl}:
                status = "pass"
                reason = ""
            elif tie_decl in attached:
                status = "multi-net"
                reason = f"tie {tie_decl!r} present but ring also shorts {attached_t}"
            else:
                status = "wrong-net"
                reason = f"ring tied to {attached_t}, not declared {tie_decl!r}"
        else:
            # no declared tie: present + closed + polarity is enough for a pass;
            # the drawn tap set is reported as evidence.
            status = "pass"
            reason = "" if attached else "no tie declared; ring frame unrouted"

        results.append(GuardRingVerdict(
            members, well_decl, tie_decl, status, drawn_well, attached_t, reason))

    violations = [r for r in results if r.status in _VIOLATION_STATES]
    return GuardRingReport(
        clean=not violations,
        results=tuple(results),
        checked=checked,
        skipped=tuple(skipped),
    )


__all__ = [
    "GuardRingVerdict",
    "GuardRingReport",
    "check_guard_ring_materialized",
]
