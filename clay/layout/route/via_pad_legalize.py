"""Post-route via-pad legalization: hold the per-layer metal min-space from every
via LANDING PAD to foreign-net metal (the cross-net via-pad spacing obligation).

WHY
===
The negotiated-congestion engine's "DRC-by-pitch" guarantees same-layer spacing for
WIRES (the uniform grid pitch is ``max(min_width + min_space)``), but a via landing
pad is WIDER than the wire (gf180 via1: 260nm cut + 2x60nm enclosure = 380nm vs a
280nm M2 wire). A foreign net's metal can therefore pack to the wire-class clearance
(~230nm) from the wider pad, firing the foundry M2.2a (280nm) -- a cross-net via-pad
spacing violation the wire-pitch model does not cover.

THE PASS (general; net-identity + PDK min_space/enclosure driven, NO circuit / net /
PDK branch)
- Op A NO-CUT OVERHANG TRIM: an over-min-width metal polygon carrying NO via cut that
  sits < min_space from a foreign net's same-layer metal -- trim its foreign-facing
  overhang back to the min-width trunk edge (gratuitous widen-halo metal, no via to
  enclose), then restore min-area by extending along the CLEAR trunk axis.
- Op B VIA-STACK RELOCATION: a real landing pad (a via1/via2 cut under it) that
  violates -- slide the ENTIRE colocated stack (the pad + its via1/via2 cuts + the M1
  pad under via1 + the M3 pad over via2) RIGIDLY along the net's own trunk by the
  smallest on-grid step that makes the pad clear every foreign same-layer metal by
  >= min_space WHILE the pad still overlaps the net's own trunk (connectivity) and
  overlaps NO foreign metal (never a short).

Safety: every relocation is accepted only when it keeps the pad on the net's own
trunk and preserves every exact same-net device-terminal landing carried by the
original stack (no open), while remaining clear of all foreign metal (no short); an
offending pad with no admissible position is left UNCHANGED and reported in
``unfixed`` (an honest residual, never a fabricated short or open). The caller
re-runs the union-find short oracle + net-completeness after this pass, so any
residual is measured, not asserted clean.
The mirror-pinned diff-pair via2 pad whose foreign trunk runs parallel for its full
length within one routing pitch has NO admissible in-trunk position -- that residual
is a placement / track-assignment obligation (the L3 mirror contract), not a
post-route one, and is left in ``unfixed``.
"""
from __future__ import annotations

from typing import Any, List, Sequence, Tuple

import klayout.db as kdb

from clay.layout.backend import LayerPolygon, Polygon, Point

Bbox = Tuple[int, int, int, int]


def _key(lp: Any) -> Tuple[int, int]:
    return (int(lp.layer), int(lp.datatype))


def _bb(lp: Any) -> Bbox:
    xs = [v.x for v in lp.polygon.vertices]
    ys = [v.y for v in lp.polygon.vertices]
    return (min(xs), min(ys), max(xs), max(ys))


def _box(x0: int, y0: int, x1: int, y1: int, k: Tuple[int, int]) -> LayerPolygon:
    return LayerPolygon(
        layer=k[0], datatype=k[1],
        polygon=Polygon(vertices=[Point(x0, y0), Point(x1, y0),
                                  Point(x1, y1), Point(x0, y1)]),
    )


def legalize_via_pads(
    routed: Any,
    tech: Any,
    *,
    device_obstacles: Sequence[Any] = (),
    enforce_device_m1_provenance: bool = False,
) -> Tuple[int, int, List[Tuple[str, Bbox]]]:
    """Legalize cross-net via-pad spacing on ``routed`` (mutates ``routed.nets[net].wires``
    in place). Returns ``(trimmed, relocated, unfixed)``: counts of Op-A trims and Op-B
    relocations, and the list of ``(net, pad_bbox)`` pads with no admissible position
    (honest residual). A no-op where no over-wide pad violates (e.g. the OTA)."""
    M2 = tuple(tech.metal_key(2))
    V1 = tuple(tech.via_key(1))
    M1 = tuple(tech.metal_key(1))
    V2 = tuple(tech.via_key(2))
    M3 = tuple(tech.metal_key(3))
    MS = int(tech.min_space("metal2"))
    MINW = int(tech.min_width("metal2") or 280)
    step = MINW // 2  # on-grid step (140nm on gf180)
    _CUT_KEYS = (V1, V2)

    # The canonical LayoutModel projects obstacle coordinates in um; routed
    # geometry and legalizer candidates use emitted nm integers.  Consume that
    # one projection directly instead of rebuilding a second ownership oracle.
    device_m1: List[Tuple[str | None, str | None, Bbox]] = []
    if enforce_device_m1_provenance:
        for obs in device_obstacles:
            if tuple(getattr(obs, "layer", ())) != M1:
                continue
            if getattr(obs, "kind", None) != "device":
                continue
            device_m1.append((
                getattr(obs, "owner_net", None),
                getattr(obs, "terminal_conductor_ref", None),
                (round(float(obs.x0) * 1000.0), round(float(obs.y0) * 1000.0),
                 round(float(obs.x1) * 1000.0), round(float(obs.y1) * 1000.0)),
            ))

    def terminal_refs_touched(net: str, boxes: Sequence[Bbox]) -> set[str]:
        refs: set[str] = set()
        for x0, y0, x1, y1 in boxes:
            for owner, terminal_ref, (ox0, oy0, ox1, oy1) in device_m1:
                if owner != net or not terminal_ref:
                    continue
                if x1 < ox0 or ox1 < x0 or y1 < oy0 or oy1 < y0:
                    continue
                refs.add(str(terminal_ref))
        return refs

    def device_m1_landing_ok(
        net: str,
        boxes: Sequence[Bbox],
        *,
        required_terminal_refs: frozenset[str] = frozenset(),
        terminal_carriers: Sequence[Bbox] = (),
    ) -> bool:
        """Every moved VIA1 cut/M1 landing must avoid foreign/unowned device M1.

        Edge contact is conservatively treated as overlap: same-layer touching
        artwork can be electrically connected by extraction and must not be
        licensed or severed by a geometric post-pass.  Exact terminal-conductor
        references turn an original same-net landing into a relocation obligation.
        """
        if not enforce_device_m1_provenance:
            return True
        for x0, y0, x1, y1 in boxes:
            for owner, _terminal_ref, (ox0, oy0, ox1, oy1) in device_m1:
                if x1 < ox0 or ox1 < x0 or y1 < oy0 or oy1 < y0:
                    continue
                if owner != net:
                    return False
        return required_terminal_refs <= terminal_refs_touched(
            net, terminal_carriers)

    def foreign_metal(net: str, key: Tuple[int, int]) -> kdb.Region:
        reg = kdb.Region()
        for n, nr in routed.nets.items():
            if n == net:
                continue
            for lp in nr.wires:
                if _key(lp) == key:
                    reg.insert(kdb.Box(*_bb(lp)))
        reg.merge()
        return reg

    def foreign_m2(net: str) -> kdb.Region:
        return foreign_metal(net, M2)

    metal_spacing = {
        M1: int(tech.min_space("metal1") or 0),
        M2: int(tech.min_space("metal2") or 0),
        M3: int(tech.min_space("metal3") or 0),
    }

    def routed_metal_landings_ok(
        net: str, landings: Sequence[Tuple[Tuple[int, int], Bbox]],
    ) -> bool:
        """Moved stack metal must neither contact nor crowd foreign metal.

        The historical pass repaired only the M2 storey while moving the M1
        and M3 storeys blindly.  Safe over-cell mode checks every moved metal
        landing against routed foreign artwork on that same layer.  This is
        flag-gated with the rest of strict provenance so the default pass is
        byte-for-byte unchanged.
        """
        if not enforce_device_m1_provenance:
            return True
        for key, box in landings:
            spacing = metal_spacing.get(key)
            if spacing is None:
                continue
            foreign = foreign_metal(net, key)
            candidate = kdb.Region(kdb.Box(*box))
            if not (candidate & foreign).is_empty():
                return False
            if spacing and candidate.separation_check(foreign, spacing).size():
                return False
        return True

    def own_m2(net: str, exclude: Bbox | None = None) -> kdb.Region:
        reg = kdb.Region()
        for lp in routed.nets[net].wires:
            if _key(lp) == M2:
                b = _bb(lp)
                if exclude is not None and b == exclude:
                    continue
                reg.insert(kdb.Box(*b))
        reg.merge()
        return reg

    def any_cut_under(net: str, box: Bbox) -> bool:
        x0, y0, x1, y1 = box
        for lp in routed.nets[net].wires:
            if _key(lp) in _CUT_KEYS:
                c = _bb(lp)
                cx, cy = (c[0] + c[2]) // 2, (c[1] + c[3]) // 2
                if x0 <= cx <= x1 and y0 <= cy <= y1:
                    return True
        return False

    trimmed = 0
    relocated = 0
    mina = tech.min_area("metal2") or 0

    # ---- Op A: trim no-cut overhangs ----
    for net, nr in routed.nets.items():
        fr = foreign_m2(net)
        new_wires: List[Any] = []
        for lp in nr.wires:
            if _key(lp) != M2:
                new_wires.append(lp)
                continue
            x0, y0, x1, y1 = _bb(lp)
            w, h = x1 - x0, y1 - y0
            if w <= MINW and h <= MINW:
                new_wires.append(lp)
                continue
            if kdb.Region(kdb.Box(x0, y0, x1, y1)).separation_check(fr, MS).size() == 0:
                new_wires.append(lp)
                continue
            if any_cut_under(net, (x0, y0, x1, y1)):
                new_wires.append(lp)  # real via pad -> Op B
                continue

            def faces(side: str) -> bool:
                if side == 'L':
                    p = kdb.Box(x0 - MS, y0, x0, y1)
                elif side == 'R':
                    p = kdb.Box(x1, y0, x1 + MS, y1)
                elif side == 'B':
                    p = kdb.Box(x0, y0 - MS, x1, y0)
                else:
                    p = kdb.Box(x0, y1, x1, y1 + MS)
                return not (kdb.Region(p) & fr).is_empty()

            nx0, ny0, nx1, ny1 = x0, y0, x1, y1
            if w > MINW:
                ex = w - MINW
                if faces('L') and not faces('R'):
                    nx0 = x0 + ex
                elif faces('R') and not faces('L'):
                    nx1 = x1 - ex
                else:
                    cx = (x0 + x1) // 2
                    nx0, nx1 = cx - MINW // 2, cx + MINW // 2
            if h > MINW:
                ex = h - MINW
                if faces('B') and not faces('T'):
                    ny0 = y0 + ex
                elif faces('T') and not faces('B'):
                    ny1 = y1 - ex
                else:
                    cy = (y0 + y1) // 2
                    ny0, ny1 = cy - MINW // 2, cy + MINW // 2

            # min-area restore: extend along a CLEAR trunk side until the component
            # (merged with the net's own M2) meets min-area; never grow toward foreign.
            if mina:
                own_here = own_m2(net, exclude=(x0, y0, x1, y1))

                def _rect_ok(b0: int, b1: int, b2: int, b3: int) -> bool:
                    cand = kdb.Region(kdb.Box(b0, b1, b2, b3))
                    return (cand.separation_check(fr, MS).size() == 0
                            and (cand & fr).is_empty())

                guard = 0
                while guard <= 40:
                    merged = kdb.Region(kdb.Box(nx0, ny0, nx1, ny1)) + own_here
                    merged.merge()
                    cxr, cyr = (nx0 + nx1) // 2, (ny0 + ny1) // 2
                    comp_area = 0
                    for p in merged.each():
                        bp = p.bbox()
                        if bp.left <= cxr <= bp.right and bp.bottom <= cyr <= bp.top:
                            comp_area = p.area()
                            break
                    if comp_area >= mina:
                        break
                    guard += 1
                    grew = False
                    for (dx0, dy0, dx1, dy1) in ((0, 0, 0, step), (0, -step, 0, 0),
                                                 (0, 0, step, 0), (-step, 0, 0, 0)):
                        tx0, ty0, tx1, ty1 = nx0 + dx0, ny0 + dy0, nx1 + dx1, ny1 + dy1
                        if _rect_ok(tx0, ty0, tx1, ty1):
                            nx0, ny0, nx1, ny1 = tx0, ty0, tx1, ty1
                            grew = True
                            break
                    if not grew:
                        break
            new_wires.append(_box(nx0, ny0, nx1, ny1, M2))
            trimmed += 1
        nr.wires = new_wires

    # ---- Op B: relocate the full via stack along the own M2 trunk ----
    unfixed: List[Tuple[str, Bbox]] = []
    for net, nr in routed.nets.items():
        fr = foreign_m2(net)
        offenders: List[Bbox] = []
        for lp in nr.wires:
            if _key(lp) != M2:
                continue
            b = _bb(lp)
            w, h = b[2] - b[0], b[3] - b[1]
            if w <= MINW and h <= MINW:
                continue
            if kdb.Region(kdb.Box(*b)).separation_check(fr, MS).size() == 0:
                continue
            if not any_cut_under(net, b):
                continue
            offenders.append(b)
        for padb in offenders:
            own = own_m2(net, exclude=padb)
            x0, y0, x1, y1 = padb

            def colocated(keys: Tuple[Tuple[int, int], ...]) -> List[Bbox]:
                got = []
                for lp in nr.wires:
                    if _key(lp) in keys:
                        c = _bb(lp)
                        cx, cy = (c[0] + c[2]) // 2, (c[1] + c[3]) // 2
                        if x0 - 50 <= cx <= x1 + 50 and y0 - 50 <= cy <= y1 + 50:
                            got.append(c)
                return got

            stack_items = [
                (_key(lp), _bb(lp)) for lp in nr.wires
                if _key(lp) in (V1, V2, M1, M3)
                and x0 - 50 <= (_bb(lp)[0] + _bb(lp)[2]) // 2 <= x1 + 50
                and y0 - 50 <= (_bb(lp)[1] + _bb(lp)[3]) // 2 <= y1 + 50
            ]
            stack = {tuple(box) for _key_value, box in stack_items}
            stack_via1 = colocated((V1,))
            stack_m1 = [
                _bb(lp) for lp in nr.wires
                if _key(lp) == M1 and tuple(_bb(lp)) in stack
            ]
            terminal_carriers = stack_m1 if stack_m1 else stack_via1
            required_terminal_refs = frozenset(
                terminal_refs_touched(net, terminal_carriers))
            placed = False
            for (dx, dy) in ((-step, 0), (step, 0), (0, -step), (0, step)):
                for k in range(1, 40):
                    sx, sy = dx * k, dy * k
                    npad = (x0 + sx, y0 + sy, x1 + sx, y1 + sy)
                    cand = kdb.Region(kdb.Box(*npad))
                    if (cand & own).is_empty():
                        break  # slid off the net's own trunk -> would open
                    if cand.separation_check(fr, MS).size() == 0 and (cand & fr).is_empty():
                        moved_route_landings = [
                            (key, (b[0] + sx, b[1] + sy,
                                   b[2] + sx, b[3] + sy))
                            for key, b in stack_items if key in (M1, M3)
                        ]
                        if not routed_metal_landings_ok(
                                net, moved_route_landings):
                            continue
                        moved_landings = [
                            (b[0] + sx, b[1] + sy, b[2] + sx, b[3] + sy)
                            for b in (stack_m1 + stack_via1)
                        ]
                        moved_terminal_carriers = [
                            (b[0] + sx, b[1] + sy, b[2] + sx, b[3] + sy)
                            for b in terminal_carriers
                        ]
                        if not device_m1_landing_ok(
                            net,
                            moved_landings,
                            required_terminal_refs=required_terminal_refs,
                            terminal_carriers=moved_terminal_carriers,
                        ):
                            continue
                        nw: List[Any] = []
                        for lp in nr.wires:
                            kk = _key(lp)
                            b = _bb(lp)
                            if kk == M2 and b == padb:
                                nw.append(_box(*npad, M2))
                            elif tuple(b) in stack and kk in (V1, V2, M1, M3):
                                nw.append(_box(b[0] + sx, b[1] + sy, b[2] + sx, b[3] + sy, kk))
                            else:
                                nw.append(lp)
                        nr.wires = nw
                        relocated += 1
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                unfixed.append((net, padb))
    return trimmed, relocated, unfixed
