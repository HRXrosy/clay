"""Stage-2 common-centroid array materializer for a matched device pair.

``build_common_centroid_array`` is the ``@common_centroid(group, rows=)`` sibling
of :func:`clay.layout.device_cell.build_shared_diffusion_array`. Where the
shared-diffusion array lays every member as a contiguous unit-finger block in ONE
row (a current mirror / CS bank), the common-centroid array INTERLEAVES two matched
members (a diff input pair) into a point-reflected ``ABBA`` fold about a shared
centroid so a linear/even-order process gradient averages out across the pair.

It does NOT re-implement the interleave geometry: it DELEGATES to the existing
:func:`clay.layout.common_centroid_cluster.build_interdigitated_cluster` (which
owns the correct-by-construction interleave, split-gate routing and per-net drain
rails) and ADAPTS that ``DeviceGeom`` into a
:class:`~clay.layout.device_cell.SharedDiffusionArray`-shaped object so the same
emit / pin / guard-ring path consumes it identically to the Stage-1 array.

The adaptation primarily re-projects PROVENANCE: the delegate's ``ClusterInfo``
already carries the per-finger owner + centre-x and the per-column net map
row-by-row, so the S/D drain/source columns and per-row finger bands are
reconstructed from that provenance.  The adapter additionally materializes the
continuous-well body-tap obligation and may expand the delegate Nwell to enclose it.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

from .backend import LayerPolygon, Point, Polygon
from .device_cell import (
    ArrayColumn,
    Bbox,
    SharedDiffusionArray,
    _bbox,
    _od2_box_over_comp,
    _od2_key,
    _thick_oxide_is_default,
    _materialize_shared_pmos_well_tap,
    _rect,
    _resolve_layer_keys,
    _rules_from_tech,
    _validate_body_tap_owner_sha256,
)
from .common_centroid_cluster import build_interdigitated_cluster


@dataclass(frozen=True)
class _CCFinger:
    """One poly gate finger of a common-centroid array.

    Duck-types :class:`~clay.layout.device_cell.ArrayFinger` (``member`` /
    ``gate_net`` / ``x0`` / ``x1``) and additionally carries ``cy`` -- the finger's
    row-band channel centre y -- because a common-centroid fold is MULTI-row and the
    band a finger sits in is load-bearing (each member appears in BOTH rows)."""

    member: str
    gate_net: str
    x0: int
    x1: int
    cy: int
    # Optional exact artwork transform for a resolved fixed-PCell occurrence.
    # Generic continuous arrays leave this empty and inherit the rigid macro
    # orientation at the placement bridge.
    orientation: str = ""
    # Exact owner probe through this gate. R0 keeps the historical x-span/cy
    # representation; a quarter-turn carries the rigidly rotated positive-area
    # probe because a horizontal gate cannot be represented by x0/x1/cy alone.
    gate_bbox_nm: Optional[Bbox] = None


class CommonCentroidArray(SharedDiffusionArray):
    """A :class:`SharedDiffusionArray` whose fingers span multiple centro-symmetric
    rows. It overrides :meth:`member_pins` so each member's S/D access pin lands at
    the chosen column's OWN row-band centre (on the drawn M1 pad) rather than the
    union-comp centre (which, for a 2-row fold, sits in the inter-row field gap)."""

    def member_pins(self, name: str) -> Dict[str, Tuple[int, int]]:
        oriented = getattr(self, "oriented_member_pins", None)
        if type(oriented) is tuple:
            for member, pin_items in oriented:
                if member == name:
                    return dict(pin_items)
        drains = [c for c in self.columns if c.role == "d" and c.member == name]
        if not drains:
            owners = sorted({c.member for c in self.columns if c.member})
            raise KeyError(
                f"{name!r} owns no drain column in this common-centroid array "
                f"(members with drains: {owners})"
            )
        blk = [f for f in self.fingers if f.member == name]
        fmid = (
            (min(f.x0 for f in blk) + max(f.x1 for f in blk)) // 2
            if blk else drains[0].x_center
        )
        # representative drain / source columns nearest this member's block centre
        # (deterministic, x_center tiebreak). Because the pair is centroid-coincident
        # the two members' block centres coincide, but their DRAIN columns are
        # disjoint -> the returned drain pins differ per member (what lets the router
        # tell the interleaved drains apart).
        dcol = min(drains, key=lambda c: (abs(c.x_center - fmid), c.x_center))
        sources = [c for c in self.columns if c.role == "s"]
        scol = min(sources, key=lambda c: (abs(c.x_center - fmid), c.x_center))
        # SPLIT-gate pair (VIP != VIN): member A straps UP to a top bus, member B DOWN
        # to a bottom bus, so each member has its OWN gate contact -- return that per
        # member so a diff-pair member's gate landing is distinct, not the single
        # shared contact. A SHARED-gate pair has an empty map -> both fall back to the
        # one ``gate_contact_bbox`` (byte-identical).
        per_member = getattr(self, "gate_contacts_by_member", None) or {}
        gc = per_member.get(name, self.gate_contact_bbox)
        gx = (gc[0] + gc[2]) // 2
        gy = (gc[1] + gc[3]) // 2
        drain_strap = self.drain_straps.get(name)
        drain_xy = (
            (
                (drain_strap[0] + drain_strap[2]) // 2,
                (drain_strap[1] + drain_strap[3]) // 2,
            )
            if drain_strap is not None
            else (dcol.x_center, _box_cy(dcol.contact_bbox))
        )
        pins = {
            "d": (int(drain_xy[0]), int(drain_xy[1])),
            "s": (int(scol.x_center), int(_box_cy(scol.contact_bbox))),
            "g": (int(gx), int(gy)),
        }
        if self.well_tap_isolated and self.body_tap_bbox is not None:
            bx0, by0, bx1, by1 = self.body_tap_bbox
            pins["b"] = ((bx0 + bx1) // 2, (by0 + by1) // 2)
        return pins


def _box_cy(box: Bbox) -> int:
    return (box[1] + box[3]) // 2


def _member_fields(m):
    """Normalize the canonical four-terminal member tuple (legacy 8-tuples mean
    ``body == source``)."""
    values = tuple(m)
    if len(values) == 8:
        name, kind, w_um, l_um, nf, g, d, s = values
        b = s
    elif len(values) == 9:
        name, kind, w_um, l_um, nf, g, d, s, b = values
    else:
        raise ValueError(
            "common-centroid member must have 8 legacy fields or 9 "
            f"four-terminal fields (got {len(values)})"
        )
    return (str(name), str(kind), float(w_um), float(l_um), int(nf),
            str(g), str(d), str(s), str(b))


def build_common_centroid_array(
    tech,
    members,
    rows: int = 2,
    oxide: str = "thin",
    flank_dummies=(),
    orientation: str = "R0",
    body_tap_owner_sha256: str = "",
) -> SharedDiffusionArray:
    """Materialize a ``@common_centroid`` matched pair as a point-reflected fold.

    Parameters
    ----------
    tech : LayoutTech
        The aggregated PDK surface -- every pitch / enclosure / implant constant is
        resolved from it EXACTLY as the single-device builder does. General over any
        PDK; no per-circuit branch.
    members : list of exactly TWO tuples
        ``(name, kind, W_um, L_um, nf, gate_net, drain_net, source_net,
        body_net)``. The two
        members MUST share one polarity (``kind``), one channel width (``W_um``), one
        gate length (``L_um``) and one finger count (``nf``) -- a matched pair -- and
        share their SOURCE net; each drives its OWN drain net (and, for a diff pair,
        its own gate net). This is validated: != 2 members, or a mismatched
        (kind, W, L, nf), raises ``ValueError``.
    rows : int
        The requested band count. ``rows=2`` materialises the 2-band 2-D common-
        centroid fold the reference input pair uses -- for ANY even ``nf``, including
        ``nf`` divisible by 4 (which the delegate would otherwise lay in ONE wide row);
        this is threaded to the delegate as ``force_rows=2``. ``rows=1`` keeps the
        single-row ABBA fold, which is only realisable for ``nf`` divisible by 4. A
        ``rows`` that cannot be realised for the given ``nf`` (e.g. ``rows=1`` on
        ``nf=6``) raises ``ValueError``.
    oxide : {'thin', 'thick'}
        ``thin`` (default) draws no OD2. ``thick`` (3.3V) grows the OD2 thick-oxide
        marker over each row's COMP; ``thick`` on a PDK with no OD2 layer is a hard
        error.
    flank_dummies : sequence of member tuples, optional
        ``@dummy_flank`` common-centroid END-CAP dummies to fold into the tile as
        gap=0 end-cap fingers. Each is a four-terminal 9-tuple ``(name, kind, W_um,
        L_um, M, gate, drain, source, body)``; the legacy 8-tuple form means
        ``body == source``. Two kinds are accepted, both same
        ``kind`` / ``W`` / ``L`` as the pair:
          * DRAIN-EXPOSED (``@dummy_flank(tie=gate_off)``): gate == source, drain in
            the pair's two drains ``{da, db}``, and ``M == rows``: exactly one
            physical finger at ONE matched end of each row. Multiple consecutive
            fingers are rejected because their source-valued inner boundary creates
            an all-source parasitic device.
          * FULLY-OFF RAIL (``@dummy_flank(tie=rail)``): gate == source == drain ==
            the pair's SOURCE rail net -- an ALL-SOURCE end-cap (its S/D columns are
            all the rail net; its gate ties down to the source rail), folded at BOTH
            ends of each row, ``M % (2*rows) == 0`` (``per_end = M // (2*rows)``).
        An honest ``ValueError`` on a mismatch, never a silent wrong-M device. The
        default ``()`` is byte-identical to the un-folded tile.
    orientation : {'R0', 'R90', 'R180', 'R270'}
        Integer rigid rotation of the complete materialized macro, including all
        artwork, feature boxes, provenance centres and terminal accesses. ``R0``
        preserves the historical byte path.

    Returns
    -------
    SharedDiffusionArray
        A :class:`CommonCentroidArray` (a ``SharedDiffusionArray`` subtype) exposing
        the same ``polygons`` / ``extent`` / ``columns`` / ``fingers`` /
        ``member_pins`` / ``comp_bbox`` / ``gate_contact_bbox`` contract, so the emit
        / pin / guard-ring path consumes it identically to the Stage-1 array. The
        delegate geometry is preserved except for the shared body tap and any
        Nwell expansion required to enclose that tap.
    """
    body_tap_owner_sha256 = _validate_body_tap_owner_sha256(
        body_tap_owner_sha256
    )
    _oxide = str(oxide or "thin").strip().lower()
    if _oxide not in ("thin", "thick"):
        raise ValueError("oxide must be 'thin' or 'thick'")
    if type(orientation) is not str:
        raise TypeError("orientation must be an exact string")
    if orientation not in ("R0", "R90", "R180", "R270"):
        raise ValueError("orientation must be one of R0/R90/R180/R270")
    rows = int(rows)
    if rows < 1:
        raise ValueError(f"rows must be >= 1 (got {rows})")

    mem = [_member_fields(m) for m in members]
    if len(mem) != 2:
        raise ValueError(
            "a common-centroid array interleaves EXACTLY two matched members "
            f"(got {len(mem)}); a single device / N>2 group is not a common-centroid "
            "pair"
        )

    (na, ka, wa, la, nfa, ga, da, sa_net, ba_net) = mem[0]
    (nb, kb, wb, lb, nfb, gb, db, sb_net, bb_net) = mem[1]

    if ka != kb:
        raise ValueError(
            f"common-centroid members must share one polarity; got {ka!r} / {kb!r}")
    if ka not in ("nmos", "pmos"):
        raise ValueError("member kind must be 'nmos' or 'pmos'")
    if abs(wa - wb) > 1e-12 or abs(la - lb) > 1e-12:
        raise ValueError(
            "common-centroid members must share one W and one L (a matched pair); "
            f"got W_um={wa}/{wb} L_um={la}/{lb}")
    if nfa != nfb:
        raise ValueError(
            "common-centroid members must share one finger count nf (a matched "
            f"pair); got nf={nfa}/{nfb}")
    if ka == "pmos" and (not ba_net or ba_net != bb_net):
        raise ValueError(
            "PMOS members in one continuous common-centroid N-well must declare "
            f"one body net; got body nets={sorted({ba_net, bb_net})}"
        )
    body_net = ba_net if ba_net == bb_net else ""
    if body_tap_owner_sha256 and (
        ka != "pmos" or body_net != sa_net or body_net != sb_net
    ):
        raise ValueError(
            "external body-tap ownership requires a source-tied PMOS N-well"
        )

    nf = nfa

    # --- validate + resolve the flank END-CAP dummies -----------------------
    # TWO end-cap kinds fold into the tile, both same kind/W/L as the pair:
    #   * DRAIN-EXPOSED (gate == source, drain in the pair's two drains): a matched
    #     gate-off end-cap whose own drain lands on a dead-end outer column that
    #     extends the matching corner's same-net drain rail. It contributes exactly
    #     one finger at ONE matched end of each row -> resolved 3-tuple
    #     ``(name, drain_net, 1)``. More than one would create an all-source
    #     parasitic finger at the source-valued inner boundary.
    #   * FULLY-OFF RAIL (gate == source == drain == the pair's source rail net): an
    #     ALL-SOURCE end-cap (exactly build_shared_diffusion_array's rail end-cap) --
    #     its S/D columns are ALL the rail net and its gate ties DOWN to the source
    #     rail, so it folds at BOTH ends of each row (centro-symmetric, no distinct
    #     drain). The M fingers spread over 2 ends * rows, so per_end = M // (2*rows)
    #     -> resolved 4-tuple ``(name, rail_net, per_end, True)``.
    # HONEST ValueError on a mismatch, never a silent wrong-M / drain-crossing dummy.
    resolved_flanks: List[Tuple] = []
    for fd in (flank_dummies or ()):
        fn, fk, fw, fl, fM, fg, fdr, fs, fb = _member_fields(fd)
        if fk != ka:
            raise ValueError(
                f"flank dummy {fn!r} kind {fk!r} != pair kind {ka!r}")
        if ka == "pmos" and fb != body_net:
            raise ValueError(
                f"flank dummy {fn!r} body net {fb!r} != common-centroid "
                f"N-well body net {body_net!r}"
            )
        if abs(fw - wa) > 1e-9 or abs(fl - la) > 1e-9:
            raise ValueError(
                f"flank dummy {fn!r} W/L ({fw}/{fl}) must match the pair "
                f"({wa}/{la}) -- a shared-diffusion end-cap needs equal W and L")
        is_rail = bool(fg) and fg == fs == fdr and fdr == sa_net
        if is_rail:
            # fully-off ALL-SOURCE rail end-cap at BOTH ends of every row.
            if fM < 1 or fM % (2 * rows) != 0:
                raise ValueError(
                    f"rail flank dummy {fn!r} M={fM} must be a positive multiple of "
                    f"2*rows={2 * rows} (per_end = M // (2*rows), one run per end per "
                    "row); refusing a silent wrong-M dummy")
            resolved_flanks.append((fn, fdr, fM // (2 * rows), True))
            continue
        # DRAIN-EXPOSED gate-off end-cap (unchanged).
        if not (fg and fg == fs):
            raise ValueError(
                f"flank dummy {fn!r} must be gate-off (gate == source; got "
                f"g={fg!r} s={fs!r}) for a common-centroid end-cap")
        if fdr not in (da, db):
            raise ValueError(
                f"flank dummy {fn!r} drain {fdr!r} must be one of the pair drains "
                f"{{{da!r}, {db!r}}} (a drain-net-match end-cap)")
        if fM < 1 or fM % rows != 0:
            raise ValueError(
                f"flank dummy {fn!r} M={fM} must be a positive multiple of rows="
                f"{rows} (per_row = M // rows); refusing a silent wrong-M dummy")
        per_row = fM // rows
        if per_row != 1:
            raise ValueError(
                f"drain-exposed flank dummy {fn!r} requires exactly one finger "
                "per row in a continuous shared-diffusion common-centroid array; "
                "multiple fingers create an all-source parasitic device"
            )
        resolved_flanks.append((fn, fdr, per_row))

    # DELEGATE the interleave geometry (correct-by-construction split-gate routing,
    # per-net drain rails). The row count is the delegate's PARITY default (1-row for
    # nf%4==0 else 2 centro-symmetric rows) UNLESS the caller declares rows=2: then
    # ``force_rows=2`` requests the 2-band 2-D common-centroid fold for the even-nf
    # pair as well (the reference's tall aspect). W/L to METRES (the delegate's unit);
    # each member's {s,d,g} nets are its own. force_rows only overrides for rows=2;
    # any other value leaves the parity default, so rows=1 stays byte-identical.
    geom = build_interdigitated_cluster(
        tech,
        kind=ka,
        W=wa * 1e-6,
        L=la * 1e-6,
        nf=nf,
        device_a=na,
        device_b=nb,
        nets_a={"s": sa_net, "d": da, "g": ga},
        nets_b={"s": sb_net, "d": db, "g": gb},
        force_rows=rows,
        flank_dummies=resolved_flanks,
    )
    cluster = geom.__dict__["cluster"]
    realised_rows = int(cluster.rows)
    if rows != realised_rows:
        raise ValueError(
            f"requested rows={rows} but nf={nf} lays out in {realised_rows} "
            f"centro-symmetric row(s) (a {nf}-finger pair folds into "
            f"{realised_rows} band(s)); declare rows={realised_rows}"
        )

    polygons: List[LayerPolygon] = list(geom.polygons)
    owned_removed_layers, owned_added_polygons = (), []
    body_tap_bbox: Optional[Bbox] = None
    well_tap_isolated = False
    nwell_bbox = geom.nwell_bbox
    if ka == "pmos":
        keys = _resolve_layer_keys(tech, ka)
        if nwell_bbox is None:
            raise ValueError("PMOS common-centroid geometry has no N-well")
        if body_tap_owner_sha256:
            tap_rects = []
        else:
            tap_rects, nwell_bbox, body_tap_bbox, well_tap_isolated = (
                _materialize_shared_pmos_well_tap(
                    keys=keys,
                    r=_rules_from_tech(tech),
                    comp_bbox=geom.comp_bbox,
                    source_rail_bbox=geom.source_rail_bbox,
                    source_net=sa_net,
                    body_net=body_net,
                    base_nwell_bbox=nwell_bbox,
                    forbidden_poly_bboxes=tuple(
                        _bbox(lp)
                        for lp in geom.polygons
                        if (int(lp.layer), int(lp.datatype)) == keys.poly
                    ),
                )
            )
        polygons = [
            lp for lp in polygons
            if (int(lp.layer), int(lp.datatype)) != keys.nwell
        ]
        polygons.extend(_rect(key, box) for key, box in tap_rects)
        polygons.append(_rect(keys.nwell, nwell_bbox))
        owned_removed_layers = (keys.nwell,)
        owned_added_polygons.extend(_rect(key, box) for key, box in tap_rects)
        owned_added_polygons.append(_rect(keys.nwell, nwell_bbox))

    # optional thick-oxide OD2 marker over each row's COMP (the delegate draws none).
    if _oxide == "thick":
        od2_key = _od2_key(tech)
        if od2_key is None:
            if not _thick_oxide_is_default(tech):
                raise ValueError(
                    f"oxide='thick' requested but {getattr(tech, 'pdk_id', '?')!r} "
                    "declares no OD2 (thick-gate-oxide) layer -- the thick-ox intent "
                    "cannot be materialised on this PDK")
        else:
            for comp in _row_comp_bboxes(tech, ka, geom.polygons):
                marker = _rect(od2_key, _od2_box_over_comp(comp))
                polygons.append(marker)
                owned_added_polygons.append(marker)

    columns, fingers = _reproject(tech, ka, geom, cluster, sa_net, da, db, na, nb)

    rails_by_net = dict(getattr(cluster, "drain_rails_by_net", {}) or {})
    drain_straps = {
        member: rails_by_net[net]
        for member, net in ((na, da), (nb, db))
        if net in rails_by_net
    }
    arr = CommonCentroidArray(
        polygons=polygons,
        kind=ka,
        order="common_centroid",
        gate_net="" if ga != gb else ga,
        source_net=sa_net,
        gate_length_nm=int(geom.gate_length_nm),
        channel_width_nm=int(geom.channel_width_nm),
        comp_bbox=geom.comp_bbox,
        columns=columns,
        fingers=fingers,
        source_rail_bbox=geom.source_rail_bbox,
        gate_bus_bbox=geom.gate_bus_bbox,
        gate_contact_bbox=geom.gate_contact_bbox,
        drain_straps=drain_straps,
        nwell_bbox=nwell_bbox,
        body_net=body_net,
        body_tap_bbox=body_tap_bbox,
        well_tap_isolated=well_tap_isolated,
        body_tap_owner_sha256=body_tap_owner_sha256,
    )
    # thread the delegate's per-member SPLIT-gate contacts onto the array so
    # member_pins can return each diff-pair member's OWN gate landing (frozen
    # dataclass -> object.__setattr__, the same escape hatch DeviceGeom uses for its
    # cluster provenance). Empty for a shared-gate pair (member_pins then falls back
    # to the single shared contact -> byte-identical).
    gcm = dict(getattr(cluster, "gate_contacts_by_member", {}) or {})
    if gcm:
        object.__setattr__(arr, "gate_contacts_by_member", gcm)
    from .native_owned_artwork import adapt_native_owned_artwork
    adapt_native_owned_artwork(geom, arr, removed_layers=owned_removed_layers,
                              added_polygons=owned_added_polygons)
    if orientation == "R0":
        object.__setattr__(arr, "materialized_orientation", "R0")
        return arr
    oriented = _orient_common_centroid_array(arr, orientation)
    object.__setattr__(oriented, "materialized_orientation", orientation)
    return oriented


def _oriented_point(
    point: Tuple[int, int], extent: Bbox, orientation: str
) -> Tuple[int, int]:
    """Rotate one exact-nm point and normalize the drawn lower-left to (0, 0)."""

    x, y = point
    x0, y0, x1, y1 = extent
    if orientation == "R90":
        return (y1 - y, x - x0)
    if orientation == "R180":
        return (x1 - x, y1 - y)
    if orientation == "R270":
        return (y - y0, x1 - x)
    raise ValueError("oriented point requires R90/R180/R270")


def _oriented_box(box: Bbox, extent: Bbox, orientation: str) -> Bbox:
    x0, y0, x1, y1 = box
    points = (
        _oriented_point((x0, y0), extent, orientation),
        _oriented_point((x1, y0), extent, orientation),
        _oriented_point((x1, y1), extent, orientation),
        _oriented_point((x0, y1), extent, orientation),
    )
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return (min(xs), min(ys), max(xs), max(ys))


def _orient_common_centroid_array(
    arr: CommonCentroidArray, orientation: str
) -> CommonCentroidArray:
    """Apply one rigid integer transform to the complete, already-legal macro."""

    from .place.row_program import replay_row_terminal_inventory, row_terminal_sources
    from .occurrence_artwork import rotate_occurrence_orientation

    consumed_devices = tuple(sorted({finger.member for finger in arr.fingers}))
    source_terminals = replay_row_terminal_inventory(
        geometry=arr, consumed_devices=consumed_devices,
    )
    extent = arr.extent()
    pin_records = []
    for member in sorted({finger.member for finger in arr.fingers}):
        try:
            pins = arr.member_pins(member)
        except KeyError:
            continue
        pin_records.append((
            member,
            tuple(sorted(
                (role, _oriented_point(point, extent, orientation))
                for role, point in pins.items()
            )),
        ))

    polygons = [
        LayerPolygon(
            layer=lp.layer,
            datatype=lp.datatype,
            polygon=Polygon(vertices=[
                Point(*_oriented_point((vertex.x, vertex.y), extent, orientation))
                for vertex in lp.polygon.vertices
            ]),
        )
        for lp in arr.polygons
    ]
    columns = []
    for column in arr.columns:
        contact = _oriented_box(column.contact_bbox, extent, orientation)
        strip = _oriented_box(column.m1_strip_bbox, extent, orientation)
        columns.append(ArrayColumn(
            role=column.role,
            net=column.net,
            member=column.member,
            x_center=(contact[0] + contact[2]) // 2,
            contact_bbox=contact,
            m1_strip_bbox=strip,
            cy=(contact[1] + contact[3]) // 2,
        ))
    fingers = []
    for finger in arr.fingers:
        original_probe = getattr(finger, "gate_bbox_nm", None)
        gate_probe = _oriented_box(
            (finger.x0, finger.cy - 1, finger.x1, finger.cy + 1)
            if original_probe is None else original_probe,
            extent,
            orientation,
        )
        left = _oriented_point((finger.x0, finger.cy), extent, orientation)
        right = _oriented_point((finger.x1, finger.cy), extent, orientation)
        fingers.append(_CCFinger(
            member=finger.member,
            gate_net=finger.gate_net,
            x0=min(left[0], right[0]),
            x1=max(left[0], right[0]),
            cy=(left[1] + right[1]) // 2,
            orientation=(rotate_occurrence_orientation(finger.orientation, orientation)
                         if getattr(finger, "orientation", "") else ""),
            gate_bbox_nm=gate_probe,
        ))

    rotated = CommonCentroidArray(
        polygons=polygons,
        kind=arr.kind,
        order=arr.order,
        gate_net=arr.gate_net,
        source_net=arr.source_net,
        gate_length_nm=arr.gate_length_nm,
        channel_width_nm=arr.channel_width_nm,
        comp_bbox=_oriented_box(arr.comp_bbox, extent, orientation),
        columns=columns,
        fingers=fingers,
        source_rail_bbox=_oriented_box(arr.source_rail_bbox, extent, orientation),
        gate_bus_bbox=_oriented_box(arr.gate_bus_bbox, extent, orientation),
        gate_contact_bbox=_oriented_box(arr.gate_contact_bbox, extent, orientation),
        drain_straps={
            member: _oriented_box(box, extent, orientation)
            for member, box in arr.drain_straps.items()
        },
        nwell_bbox=(
            _oriented_box(arr.nwell_bbox, extent, orientation)
            if arr.nwell_bbox is not None else None
        ),
        body_net=arr.body_net,
        body_tap_bbox=(
            _oriented_box(arr.body_tap_bbox, extent, orientation)
            if arr.body_tap_bbox is not None else None
        ),
        well_tap_isolated=arr.well_tap_isolated,
        body_tap_owner_sha256=arr.body_tap_owner_sha256,
    )
    object.__setattr__(rotated, "oriented_member_pins", tuple(pin_records))
    object.__setattr__(rotated, "orient", orientation)
    gate_contacts = dict(getattr(arr, "gate_contacts_by_member", {}) or {})
    if gate_contacts:
        object.__setattr__(
            rotated,
            "gate_contacts_by_member",
            {
                member: _oriented_box(box, extent, orientation)
                for member, box in gate_contacts.items()
            },
        )
    if getattr(arr, 'native_owned_artwork', None) is not None:
        from .native_owned_artwork import transform_native_owned_artwork
        transform_native_owned_artwork(arr, rotated, lambda point: _oriented_point(point, extent, orientation))
    if hasattr(arr, "channel_inventory"):
        from .occurrence_channels import replay_row_channel_inventory, transform_row_channel_inventory

        x0, y0, x1, y1 = extent
        transforms = {
            "R90": (((0, -1), (1, 0)), (y1, -x0)),
            "R180": (((-1, 0), (0, -1)), (x1, y1)),
            "R270": (((0, 1), (-1, 0)), (-y0, x1)),
        }
        matrix, translation = transforms[orientation]
        consumed_devices = tuple(sorted({finger.member for finger in arr.fingers}))
        inventory = transform_row_channel_inventory(
            geometry=arr, consumed_devices=consumed_devices, matrix=matrix,
            translation_nm=translation, row_polygons=tuple(rotated.polygons))
        object.__setattr__(rotated, "channel_inventory", inventory)
        replay_row_channel_inventory(geometry=rotated, consumed_devices=consumed_devices)
    if getattr(arr, 'joint_row_binding', None) is not None:
        from .joint_row_transform import transform_joint_row_geometry
        transform_joint_row_geometry(arr, rotated, matrix=matrix, translation_nm=translation)
    elif source_terminals is not None:
        from .physical_terminal_inventory import compile_physical_terminal_inventory
        from .tech import LayoutTech

        # Transport the independently retained replication records as well as
        # the component partition. Recompilation chooses an actual grid seed
        # in the transformed metal, while source identities remain explicit.
        object.__setattr__(rotated, "fixed_occurrence_sources", tuple(
            replace(source,
                origin_nm=_oriented_point(source.origin_nm, extent, orientation),
                orientation=rotate_occurrence_orientation(source.orientation, orientation),
                polygon_records=tuple(sorted(
                    (layer, datatype, tuple(_oriented_point(point, extent, orientation)
                                           for point in vertices))
                    for layer, datatype, vertices in source.polygon_records
                )),
                terminals=tuple(sorted(
                    (role, net, _oriented_box(box, extent, orientation))
                    for role, net, box in source.terminals
                )),
            ) for source in arr.fixed_occurrence_sources
        ))
        if getattr(arr, "fixed_pcell", None) is not None:
            object.__setattr__(rotated, "fixed_pcell", dict(arr.fixed_pcell))
        tech = LayoutTech.resolve(source_terminals.pdk_id)
        expected = row_terminal_sources(
            geometry=rotated, consumed_devices=consumed_devices, tech=tech,
        )
        object.__setattr__(rotated, "physical_terminal_inventory",
            compile_physical_terminal_inventory(geometry=rotated, sources=expected, tech=tech))
        replay_row_terminal_inventory(
            geometry=rotated, consumed_devices=consumed_devices, tech=tech,
        )
    return rotated


def _row_comp_bboxes(tech, kind, polygons) -> List[Bbox]:
    """The per-row COMP (active) rectangles, sorted low-to-high in y."""
    from .device_cell import _bbox
    akey = _resolve_layer_keys(tech, kind).active
    boxes = [_bbox(lp) for lp in polygons if (lp.layer, lp.datatype) == akey]
    return sorted(boxes, key=lambda b: (b[1] + b[3]))


def _reproject(
    tech, kind, geom, cluster, source_net: str, drn_a: str, drn_b: str,
    name_a: str, name_b: str,
) -> Tuple[List[ArrayColumn], List[_CCFinger]]:
    """Reconstruct the per-column / per-finger net map from the delegate's
    provenance (no new geometry).

    ``cluster.finger_owner`` / ``cluster.finger_col_x`` are FLAT row-by-row (row0
    fingers, then row1); ``cluster.sd_column_nets`` is the per-column net map, also
    flat row-by-row with ``n_cols + 1`` columns per row. The INTERIOR S/D columns
    (all drains + the pair-boundary shared sources) sit at the midpoints of adjacent
    finger centres -- recoverable from ``finger_col_x`` alone. The two edge source
    columns are omitted (the interior sources already give ``member_pins`` a source
    to land on)."""
    from .device_cell import _bbox

    finger_owner = list(cluster.finger_owner)
    finger_col_x = [int(x) for x in cluster.finger_col_x]
    col_nets = list(cluster.sd_column_nets)
    n_rows = max(1, int(cluster.rows))
    total = len(finger_owner)
    n_cols = total // n_rows
    sd_per_row = n_cols + 1

    # per-row band channel-centre y, from the COMP rects (row0 lowest).
    band_cys = [(_b[1] + _b[3]) // 2 for _b in _row_comp_bboxes(tech, kind, geom.polygons)]
    if len(band_cys) < n_rows:  # single-row fallback: whole comp centre
        band_cys = [(geom.comp_bbox[1] + geom.comp_bbox[3]) // 2] * n_rows

    l_nm = int(geom.gate_length_nm)
    r = _rules_from_tech(tech)
    csize = int(r.contact_size)
    m1_enc = int(r.metal1_contact_enc)
    m1_cut_enc = max(m1_enc, int(r.metal1_line_end))
    m1_pad = max(csize + 2 * m1_cut_enc, 380)

    # drain net -> owning member name (a diff pair's drains are distinct).
    net_owner: Dict[str, str] = {}
    if drn_a:
        net_owner[drn_a] = name_a
    if drn_b:
        net_owner.setdefault(drn_b, name_b)

    columns: List[ArrayColumn] = []
    fingers: List[_CCFinger] = []
    for ri in range(n_rows):
        fxs = finger_col_x[ri * n_cols:(ri + 1) * n_cols]
        fos = finger_owner[ri * n_cols:(ri + 1) * n_cols]
        nets_row = col_nets[ri * sd_per_row:(ri + 1) * sd_per_row]
        cy = band_cys[ri] if ri < len(band_cys) else band_cys[-1]

        for c, (fx, owner) in enumerate(zip(fxs, fos)):
            fingers.append(_CCFinger(
                member=str(owner), gate_net="",
                x0=int(fx - l_nm // 2), x1=int(fx - l_nm // 2 + l_nm), cy=int(cy)))

        # interior S/D columns: column i (1..n_cols-1) sits between finger i-1 and i.
        for i in range(1, n_cols):
            net = nets_row[i] if i < len(nets_row) else ""
            x_center = (fxs[i - 1] + fxs[i]) // 2
            if net == source_net or net not in net_owner:
                role, member = "s", ""
            else:
                role, member = "d", net_owner[net]
            contact = (x_center - csize // 2, cy - csize // 2,
                       x_center + csize // 2, cy + csize // 2)
            strip = (x_center - m1_pad // 2, cy - m1_pad // 2,
                     x_center + m1_pad // 2, cy + m1_pad // 2)
            columns.append(ArrayColumn(
                role=role, net=str(net), member=member,
                x_center=int(x_center), contact_bbox=contact, m1_strip_bbox=strip))

        # The two edge S/D columns are real drawn contacts too.  Earlier code
        # omitted them on the assumption that every interleave has an interior
        # source column.  A valid two-row ABBA fold can place drains at every
        # interior boundary, leaving member_pins with no source landing.  Project
        # source-valued edges explicitly; dummy drain edges remain owned by the
        # dedicated flank provenance below.
        if len(fxs) >= 2:
            left_pitch = abs(fxs[1] - fxs[0])
            right_pitch = abs(fxs[-1] - fxs[-2])
            edge_columns = (
                (0, fxs[0] - left_pitch // 2),
                (n_cols, fxs[-1] + right_pitch // 2),
            )
            for net_index, x_center in edge_columns:
                net = nets_row[net_index] if net_index < len(nets_row) else ""
                if net != source_net:
                    continue
                contact = (
                    x_center - csize // 2,
                    cy - csize // 2,
                    x_center + csize // 2,
                    cy + csize // 2,
                )
                strip = (
                    x_center - m1_pad // 2,
                    cy - m1_pad // 2,
                    x_center + m1_pad // 2,
                    cy + m1_pad // 2,
                )
                columns.append(
                    ArrayColumn(
                        role="s",
                        net=str(net),
                        member="",
                        x_center=int(x_center),
                        contact_bbox=contact,
                        m1_strip_bbox=strip,
                    )
                )

    # DUMMY-EDGE drain columns: the OUTER edge columns (col 0 / col n_all) the
    # interior loop skips. Each carries the folded dummy's own drain net, attributed
    # to the DUMMY member name (so member_pins / the short oracle see the dummy drain
    # -- risk #6). Empty when no flank dummy was folded (byte-identical-off).
    for (fx, fcy, fnet, fname) in (getattr(cluster, "flank_drain_cols", ()) or ()):
        contact = (int(fx) - csize // 2, int(fcy) - csize // 2,
                   int(fx) + csize // 2, int(fcy) + csize // 2)
        strip = (int(fx) - m1_pad // 2, int(fcy) - m1_pad // 2,
                 int(fx) + m1_pad // 2, int(fcy) + m1_pad // 2)
        columns.append(ArrayColumn(
            role="d", net=str(fnet), member=str(fname),
            x_center=int(fx), contact_bbox=contact, m1_strip_bbox=strip))

    return columns, fingers


__all__ = [
    "build_common_centroid_array",
    "CommonCentroidArray",
]
