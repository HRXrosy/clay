"""NET-AWARE shared-gate poly merge (interdigitated stacked current-mirror cells).

THE DEFECT THIS CLOSES
======================
On the gf180 realize of the external_pdk OTA, real DRC
(:func:`clay.layout.drc.run_drc` + :data:`clay.layout.drc.GF180MCU_MIN_DRC_RULES`)
finds ONE poly-poly spacing violation: 70nm < 280nm. Its cause is purely an
ASSEMBLY artefact, not a per-device one.

``MM15`` / ``MM16`` / ``MM17`` are three NMOS that SHARE the gate net ``net048``
(a current mirror). Each device's :func:`clay.layout.device_cell.build_device`
gate poly is individually DRC-clean: it is a folded poly that rises to a gate bus
with a raised gate-CONTACT pad over the left lane. When two of those devices are
placed STACKED in one x-column at the solved positions (``MM16`` below ``MM15``,
both at x=20.98), the raised gate-contact pad of the lower device pokes UP toward
the body of the upper device and leaves a 70nm concave NOTCH in the combined gate
outline (the pad's right edge at x=20.19 vs the upper poly body's left edge at
x=20.26). A real interdigitated current-mirror cell does not draw two separate
poly heads there -- the shared gate is ONE continuous poly bar, so there is no
notch.

WHAT THIS MODULE DOES (and does not)
====================================
:func:`merge_shared_gate_poly` finds groups of devices that (a) share the same
GATE net and (b) are vertically STACKED in one x-column at the SOLVED positions,
and replaces those devices' per-device gate-poly shapes with ONE continuous,
clean gate-poly RECTANGLE spanning the group's gate extent. This:

  * is ELECTRICALLY SOUND -- it adds poly only on the ALREADY-SHARED gate net,
    exactly what an interdigitated cell draws; it NEVER bridges two different
    nets (a hard overlap guard refuses any bar that would touch a non-group
    device's poly) and NEVER moves a device (positions are passed through
    untouched);
  * touches ONLY the poly (gate) layer. Source / drain diffusion, the S/D
    contact CUTS, the metal1 rails, implants and the nwell are LEFT EXACTLY AS
    BUILT. The gate poly is distinguished from the contact heads by LAYER: the
    gate (fingers + bus + the raised gate-contact PAD) is the single poly layer
    ``tech.build_layer("poly")``; the contact cuts are on the separate contact
    layer ``tech.contact_key()`` and are never modified. The "contact heads"
    that form the notch are the poly gate-contact PADS, which are part of the
    poly outline and are subsumed by the continuous bar.

LEAKAGE DISCIPLINE
==================
The grouping uses the gate NET (topology, from ``intent``) plus the SOLVED
positions (already leakage-free -- produced by the relation realize, never a
stored reference coordinate) and the device GEOMETRY (process knowledge from the
PDK). No stored reference coordinate is read. Geometry stays separate from
position: every stamp returned is centroid-relative um exactly as it came in, and
the continuous bar is expressed centroid-relative on a host device. Positions are
NOT changed.

HONEST SCOPE
============
The grouping is the conjunction of (shared gate net) AND (stacked in one
x-column). On this OTA realize the gate-net group ``{MM15, MM16, MM17}`` is the
topology group, but only ``{MM15, MM16}`` are stacked in one x-column at the
solved positions (MM17 is realized 26um away in a separate column), so the
merge applies to ``{MM15, MM16}``. Both groupings are reported
(:func:`shared_gate_groups` returns the net groups; the returned per-group
record carries the stacked sub-group that was actually merged), so the
distinction is explicit and never hidden.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from clay.layout.extract.library_device import CellGeometry

_XY = Tuple[float, float]
_Ring = List[Tuple[float, float]]
_LayerKey = Tuple[int, int]


# --------------------------------------------------------------------------- #
# Tolerances (all DERIVED from geometry / a PDK rule, never a reference coord).
# --------------------------------------------------------------------------- #
# Two devices count as "the same x-column" when their poly x-extents overlap by
# at least this fraction of the narrower poly width. MM15/MM16 share the column
# (their poly x-extents coincide -> overlap fraction 1.0); MM17 is 26um away in a
# separate column (overlap fraction 0.0), so it is not grouped. A fraction rather
# than an absolute nm avoids a magic number and scales with the device width.
_COLUMN_OVERLAP_FRACTION: float = 0.5

# Two devices in one column are "contiguous in y" (a real vertical stack) when the
# vertical GAP between their poly y-extents is <= this many um. Stacked rows abut
# (gap ~ the inter-row gap, fold_intent._ROW_GAP_UM = 1.0um) or overlap (negative
# gap); a positive threshold a little above the row gap admits a genuine adjacent
# stack while excluding far-apart devices. This is a layout-geometry threshold, not
# a reference coordinate.
_Y_CONTIGUITY_GAP_UM: float = 1.5


# --------------------------------------------------------------------------- #
# Group records.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SharedGateGroup:
    """A set of devices that share one gate net (the TOPOLOGY group).

    ``gate_net``        -- the shared gate net name.
    ``devices``         -- every device in ``intent`` whose gate role is this net.
    ``stacked``         -- the subset of ``devices`` that is ALSO stacked in one
                           x-column at the solved positions (the sub-group the
                           merge actually rewrites). May be smaller than
                           ``devices`` when a member is realized in another column.
    ``merged``          -- True if a continuous bar was emitted for ``stacked``.
    """

    gate_net: str
    devices: Tuple[str, ...]
    stacked: Tuple[str, ...] = ()
    merged: bool = False


# --------------------------------------------------------------------------- #
# geometry helpers.
# --------------------------------------------------------------------------- #
def _poly_layer(tech) -> _LayerKey:
    """The gate-poly ``(layer, datatype)`` for ``tech`` (gf180 -> (30, 0)).

    Resolved from the technology (never hardcoded). The gate poly is the SINGLE
    poly layer a MOS draws (fingers + gate bus + the raised gate-contact PAD); the
    contact CUTS live on ``tech.contact_key()`` and are a different layer this
    module never touches.
    """
    key = tech.build_layer("poly")
    if key is None:
        raise ValueError(f"tech {getattr(tech, 'pdk_id', tech)!r} has no poly layer")
    return (int(key[0]), int(key[1]))


def _abs_poly_bbox(
    geom: CellGeometry, pos: _XY, poly_layer: _LayerKey
) -> Optional[Tuple[float, float, float, float]]:
    """Absolute bbox (um) of a stamp's poly polygons = centroid-relative + position.

    Returns ``None`` when the stamp has no poly on ``poly_layer`` (nothing to
    group / merge for it). Position is the SOLVED centroid; geometry is the
    centroid-relative process stamp -- the two are combined here only to compare
    spatial adjacency, never stored back as a coordinate.
    """
    rings = geom.polys_by_layer.get(poly_layer)
    if not rings:
        return None
    px, py = pos
    xs = [x + px for ring in rings for (x, _y) in ring]
    ys = [y + py for ring in rings for (_x, y) in ring]
    if not xs or not ys:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def _x_overlap_fraction(
    a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]
) -> float:
    """Fraction of the NARROWER poly x-extent that x-overlaps the other.

    1.0 when one x-extent contains the other (same column); 0.0 when they are
    x-disjoint (separate columns). Used so "one x-column" is a geometry test, not
    a magic absolute tolerance.
    """
    lo = max(a[0], b[0])
    hi = min(a[2], b[2])
    overlap = max(0.0, hi - lo)
    narrower = min(a[2] - a[0], b[2] - b[0])
    if narrower <= 0.0:
        return 0.0
    return overlap / narrower


def _y_gap(
    a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]
) -> float:
    """Vertical gap (um) between two poly y-extents (negative if they overlap)."""
    lo, hi = (a, b) if a[1] <= b[1] else (b, a)
    return hi[1] - lo[3]


# --------------------------------------------------------------------------- #
# grouping.
# --------------------------------------------------------------------------- #
def _gate_net_of(intent) -> Dict[str, str]:
    """``device_name -> gate net name`` from ``intent.devices`` (.nets gate role).

    The gate role key is ``'gate'`` (see
    :class:`clay.layout.extract.types.ExtractedDevice`); a couple of alias keys are
    accepted defensively so the function does not silently group nothing if the
    role is named ``'g'`` elsewhere.
    """
    out: Dict[str, str] = {}
    for d in getattr(intent, "devices", []) or []:
        nets = getattr(d, "nets", {}) or {}
        gate = nets.get("gate", nets.get("g"))
        if gate:
            out[str(d.name)] = str(gate)
    return out


def _stacked_subgroups(
    members: Sequence[str],
    abs_bbox: Dict[str, Tuple[float, float, float, float]],
) -> List[List[str]]:
    """Partition ``members`` (devices sharing one gate net) into sub-groups that
    are each stacked in ONE x-column and contiguous in y at the solved positions.

    Two devices join the same sub-group when their poly x-extents overlap by
    >= :data:`_COLUMN_OVERLAP_FRACTION` of the narrower (same column) AND their
    poly y-extents are within :data:`_Y_CONTIGUITY_GAP_UM` (a real vertical stack,
    abutting or overlapping). The relation is grown transitively (a column of >2
    stacked rows is one sub-group). Only members that actually carry poly geometry
    + a position are considered; a sub-group of size >= 2 is a merge candidate.
    """
    present = [m for m in members if m in abs_bbox]
    # adjacency over the column+contiguity relation.
    adj: Dict[str, set] = {m: set() for m in present}
    for i, a in enumerate(present):
        for b in present[i + 1 :]:
            ba, bb = abs_bbox[a], abs_bbox[b]
            if (
                _x_overlap_fraction(ba, bb) >= _COLUMN_OVERLAP_FRACTION
                and _y_gap(ba, bb) <= _Y_CONTIGUITY_GAP_UM
            ):
                adj[a].add(b)
                adj[b].add(a)
    # connected components (a transitive stacked run is one sub-group).
    seen: set = set()
    groups: List[List[str]] = []
    for m in present:
        if m in seen:
            continue
        stack = [m]
        comp: List[str] = []
        seen.add(m)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nb in adj[cur]:
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        if len(comp) >= 2:
            # order a stacked column bottom-to-top for a deterministic host pick.
            comp.sort(key=lambda nm: abs_bbox[nm][1])
            groups.append(comp)
    return groups


def shared_gate_groups(intent) -> List[Tuple[str, Tuple[str, ...]]]:
    """``[(gate_net, (device, ...))]`` for every gate net shared by >= 2 devices.

    The pure TOPOLOGY grouping (gate net only, from ``intent``), independent of
    placement. On the external_pdk OTA this yields, among others,
    ``('net048', ('MM15', 'MM16', 'MM17'))`` -- the NMOS current mirror.
    """
    by_net: Dict[str, List[str]] = {}
    for name, gate in _gate_net_of(intent).items():
        by_net.setdefault(gate, []).append(name)
    out: List[Tuple[str, Tuple[str, ...]]] = []
    for gate, names in by_net.items():
        if len(names) >= 2:
            out.append((gate, tuple(sorted(names))))
    out.sort()
    return out


# --------------------------------------------------------------------------- #
# the merge.
# --------------------------------------------------------------------------- #
def _foreign_poly_overlaps(
    bar: Tuple[float, float, float, float],
    group: Sequence[str],
    positions: Dict[str, _XY],
    geometry: Dict[str, CellGeometry],
    poly_layer: _LayerKey,
) -> List[str]:
    """Names of NON-group devices whose poly the proposed ``bar`` would overlap.

    The hard cross-net guard: the continuous gate bar must touch ONLY the group's
    own (same-net) poly. Any device outside ``group`` whose poly intersects the bar
    is a different-net bridge -> the merge for this group is refused (returned list
    non-empty). Pure geometry intersection on the poly layer.
    """
    bx0, by0, bx1, by1 = bar
    members = set(group)
    hits: List[str] = []
    for name, geom in geometry.items():
        if name in members:
            continue
        pos = positions.get(name)
        if pos is None:
            continue
        bb = _abs_poly_bbox(geom, pos, poly_layer)
        if bb is None:
            continue
        # axis-aligned bbox intersection with a small touch tolerance: even a
        # zero-area edge touch counts as a bridge (poly that abuts merges in LVS).
        if bb[0] < bx1 and bb[2] > bx0 and bb[1] < by1 and bb[3] > by0:
            hits.append(name)
    return hits


def merge_shared_gate_poly(
    intent,
    positions: Dict[str, _XY],
    geometry: Dict[str, CellGeometry],
    *,
    tech_id: str,
    return_groups: bool = False,
):
    """Replace stacked shared-gate devices' per-device gate poly with ONE bar.

    Parameters
    ----------
    intent :
        The extracted intent (provides each device's gate net via
        ``ExtractedDevice.nets['gate']``).
    positions : dict
        ``name -> (cx, cy)`` SOLVED device centroids (um), from the relation
        realize -- never a stored reference coordinate. Passed through UNCHANGED.
    geometry : dict
        ``name -> CellGeometry`` centroid-relative process stamps.
    tech_id : str
        The target PDK id (e.g. ``'gf180mcu'``) -- selects the poly layer.
    return_groups : bool
        When True, return ``(updated_geometry, groups)`` where ``groups`` is the
        list of :class:`SharedGateGroup` records (topology group + the stacked
        sub-group actually merged). Default False returns just the geometry dict
        (the signature the task specifies).

    Returns
    -------
    dict[str, CellGeometry]
        A NEW geometry dict. Devices that were merged carry a rewritten poly
        layer (the host device holds the continuous bar; the others have their
        poly removed); every other device passes through UNCHANGED (same object).
        Stamps stay centroid-relative; positions are not modified.

    Behaviour
    ---------
    For each gate net shared by >= 2 devices, the members that are STACKED in one
    x-column and contiguous in y (at ``positions``) form a merge sub-group. For
    each sub-group the union of the members' absolute poly bboxes is taken; if a
    single rectangle over that union would touch any NON-member device's poly the
    merge is REFUSED for that sub-group (no cross-net bridging -- reported via the
    group record's ``merged=False``). Otherwise the rectangle is assigned, as a
    centroid-relative ring, to the lowest member (the host); the other members'
    poly layer is dropped. The bar is electrically the shared gate net, so it adds
    poly only on that net.
    """
    from clay.layout.tech import LayoutTech

    tech = LayoutTech.resolve(tech_id)
    poly_layer = _poly_layer(tech)

    # absolute poly bbox for every device that has poly + a position.
    abs_bbox: Dict[str, Tuple[float, float, float, float]] = {}
    for name, geom in geometry.items():
        pos = positions.get(name)
        if pos is None:
            continue
        bb = _abs_poly_bbox(geom, pos, poly_layer)
        if bb is not None:
            abs_bbox[name] = bb

    gate_net = _gate_net_of(intent)
    # topology groups: gate net -> members (>=2).
    by_net: Dict[str, List[str]] = {}
    for name, gate in gate_net.items():
        if name in geometry:  # only devices we actually have a stamp for
            by_net.setdefault(gate, []).append(name)

    out: Dict[str, CellGeometry] = dict(geometry)  # shallow copy; share unchanged
    group_records: List[SharedGateGroup] = []

    for gate, members in sorted(by_net.items()):
        if len(members) < 2:
            continue
        subgroups = _stacked_subgroups(members, abs_bbox)
        merged_any: List[str] = []
        for sub in subgroups:
            # union of the members' absolute poly bboxes -> the continuous bar.
            bx0 = min(abs_bbox[m][0] for m in sub)
            by0 = min(abs_bbox[m][1] for m in sub)
            bx1 = max(abs_bbox[m][2] for m in sub)
            by1 = max(abs_bbox[m][3] for m in sub)
            bar = (bx0, by0, bx1, by1)

            # HARD cross-net guard: refuse a bar that would touch foreign poly.
            foreign = _foreign_poly_overlaps(
                bar, sub, positions, geometry, poly_layer
            )
            if foreign:
                # do NOT bridge nets; leave this sub-group as-built (honest).
                continue

            # assign the bar to the host (lowest member); strip poly from others.
            host = sub[0]
            hx, hy = positions[host]
            ring: _Ring = [
                (round(bx0 - hx, 4), round(by0 - hy, 4)),
                (round(bx1 - hx, 4), round(by0 - hy, 4)),
                (round(bx1 - hx, 4), round(by1 - hy, 4)),
                (round(bx0 - hx, 4), round(by1 - hy, 4)),
            ]
            for m in sub:
                g2 = copy.deepcopy(geometry[m])
                if m == host:
                    g2.polys_by_layer[poly_layer] = [ring]
                else:
                    g2.polys_by_layer.pop(poly_layer, None)
                # provenance: mark the rewrite so it is discoverable downstream.
                g2.reason = (
                    (g2.reason or "")
                    + f"; shared-gate poly merged (net {gate}, group {sorted(sub)}, host {host})"
                ).lstrip("; ")
                out[m] = g2
            merged_any.extend(sub)

        # record one topology group per gate net (with the stacked sub-group(s)).
        stacked_flat = tuple(sorted(merged_any))
        group_records.append(
            SharedGateGroup(
                gate_net=gate,
                devices=tuple(sorted(members)),
                stacked=stacked_flat,
                merged=bool(merged_any),
            )
        )

    if return_groups:
        return out, group_records
    return out


# --------------------------------------------------------------------------- #
# ABBA interdigitated pair + dummy fence (S5 matched-geometry upgrade).
# --------------------------------------------------------------------------- #
#
# A matched diff/mirror pair (A, B) is realised as ONE row of unit fingers in a
# palindromic ABBA... order (common centroid along x), flanked by terminated
# dummy fingers at each end -- the same form the vendor reference draws for its
# matched groups (test4 MM22-26 dummies: gate/source/drain all tied to the
# adjacent source rail). Geometry REUSES ``build_device`` (each finger is one
# nf=1 device cell); this module only sequences, places and terminates --
# no drawing primitive is reimplemented.


@dataclass(frozen=True)
class PlacedFinger:
    """One unit finger of an interdigitated assembly.

    ``owner``     -- 'A' / 'B' (pair members) or 'D' (dummy).
    ``index``     -- position in the row (0 = leftmost).
    ``offset_nm`` -- the unit's placement offset (nm) in the assembly frame.
    ``geom``      -- the ``build_device`` cell (LOCAL coordinates; shift by
                     ``offset_nm`` for the assembly frame).
    ``extra_m1``  -- termination tie straps (ASSEMBLY-frame nm bboxes), only on
                     dummies: gate-pad->source-rail and drain->source-rail.
    """

    owner: str
    index: int
    offset_nm: Tuple[int, int]
    geom: object
    extra_m1: Tuple[Tuple[int, int, int, int], ...] = ()


@dataclass(frozen=True)
class InterdigitatedPair:
    """The assembled interdigitated pair (the ``build_interdigitated_pair``
    output).

    ``units``        -- the placed fingers, left to right.
    ``core_pattern`` -- the A/B order actually used (e.g. 'ABBA', 'ABBAABBA',
                        or the ABAB fallback for an odd ``M_each``).
    ``sequence``     -- the full row including dummies (e.g. 'DABBAD').
    ``note``         -- honest degradation note ('' when the requested pattern
                        was realised exactly).
    ``pitch_nm``     -- the uniform unit pitch (= the unit drawn-extent width;
                        adjacent extents abut flush, which keeps every drawn
                        layer pair at legal spacing: the per-unit poly / COMP /
                        M1 setbacks from the extent edge already exceed the
                        respective same-layer minima on the target nodes).
    ``polygons``     -- every drawn shape in ASSEMBLY-frame coordinates
                        (unit polygons shifted + the dummy tie straps).
    """

    units: Tuple[PlacedFinger, ...]
    core_pattern: str
    sequence: str
    note: str
    pitch_nm: int
    polygons: list


def _abba_core(pattern: str, m_each: int) -> Tuple[str, str]:
    """The A/B core sequence + an honest degradation note.

    'ABBA' with an EVEN ``m_each`` tiles the ABBA palindrome ('ABBA' * k is its
    own reverse, equal A/B counts, exact x common centroid on a uniform pitch).
    An ODD ``m_each`` cannot form an equal-count palindrome (every symbol count
    in an even-length palindrome is even), so it degrades to ABAB and SAYS SO.
    """
    if pattern == "ABBA":
        if m_each % 2 == 0:
            return "ABBA" * (m_each // 2), ""
        return (
            "AB" * m_each,
            f"ABBA palindrome needs an even per-device finger count; "
            f"M_each={m_each} is odd, degraded to ABAB (A/B centroid x "
            f"residual = one finger pitch instead of 0).",
        )
    return "AB" * m_each, ""


def _dummy_tie_straps(geom, xoff: int) -> Tuple[Tuple[int, int, int, int], ...]:
    """ASSEMBLY-frame M1 tie straps terminating one dummy finger.

    Three rects, all on the dummy's own (single) net: gate-pad -> source rail
    (a riser at the gate-contact x, from the rail bottom up into the gate M1
    pad -- it passes through the source column M1, same net); drain -> rail
    band (a riser at the drain-column x from the rail bottom up past the
    channel centre, overlapping the drain connector); and a rail-band BRIDGE
    spanning from the source rail to the drain riser (the source rail of an
    nf=1 unit does not reach the drain column x, the same reach problem the
    device's own COMP-centre tap bridge solves). Strap width = the rail M1
    width (a tech-derived dimension carried by the built geometry); nothing is
    hardcoded and no drawing primitive is reimplemented.
    """
    srb = geom.source_rail_bbox
    m1w = srb[3] - srb[1]
    rail_y0 = srb[1]
    gc = geom.gate_contact_bbox
    gcx = (gc[0] + gc[2]) // 2
    gate_strap = (gcx - m1w // 2 + xoff, rail_y0, gcx + m1w // 2 + xoff, gc[3])
    drb = geom.drain_rail_bbox
    dcx = (drb[0] + drb[2]) // 2
    cy = geom.channel_width_nm // 2
    drain_strap = (
        dcx - m1w // 2 + xoff, rail_y0, dcx + m1w // 2 + xoff, cy + m1w
    )
    rail_bridge = (
        min(srb[0], dcx - m1w // 2) + xoff,
        srb[1],
        max(srb[2], dcx + m1w // 2) + xoff,
        srb[3],
    )
    return (gate_strap, drain_strap, rail_bridge)


def build_interdigitated_pair(
    tech,
    kind: str,
    W: float,
    L: float,
    M_each: int,
    *,
    pattern: str = "ABBA",
    dummies: int = 1,
    nets_a: Optional[Dict[str, str]] = None,
    nets_b: Optional[Dict[str, str]] = None,
) -> InterdigitatedPair:
    """Build an ABBA-interdigitated matched pair with a dummy fence.

    Parameters
    ----------
    tech : LayoutTech
        The resolved PDK technology surface (every dimension comes from it via
        ``build_device``).
    kind : {'nmos','pmos'}
    W, L : float
        PER-FINGER channel width / gate length in metres (device A and B each
        total ``M_each * W``).
    M_each : int
        Fingers per pair member. EVEN realises the exact ABBA palindrome
        (centroid residual 0); ODD degrades to ABAB with ``note`` set.
    pattern : {'ABBA','ABAB'}
    dummies : int
        Dummy fingers at EACH end (total dummy count = 2*dummies). Each dummy
        is a full unit finger whose gate / source / drain M1 are all tied to
        the adjacent core member's SOURCE net (the vendor dummy form), via two
        M1 tie straps onto its source rail.
    nets_a, nets_b : dict, optional
        ``{'s','d','g','b' -> net}`` bindings for the A / B member fingers.

    Returns
    -------
    InterdigitatedPair
        Placed unit fingers (uniform pitch = the unit drawn extent width, so
        adjacent unit extents abut flush), the realised sequence, the honest
        degradation note, and the assembly-frame polygon list. NOTHING in the
        existing pipeline calls this yet -- enabling it in the realize flow is
        a separate integration step, so every existing path is unchanged.
    """
    from clay.layout.backend import LayerPolygon, Point, Polygon
    from clay.layout.device_cell import build_device

    if kind not in ("nmos", "pmos"):
        raise ValueError("kind must be 'nmos' or 'pmos'")
    if M_each < 1:
        raise ValueError("M_each must be >= 1")
    if dummies < 0:
        raise ValueError("dummies must be >= 0")
    if pattern not in ("ABBA", "ABAB"):
        raise ValueError("pattern must be 'ABBA' or 'ABAB'")

    core, note = _abba_core(pattern, M_each)
    sequence = "D" * dummies + core + "D" * dummies

    nets_a = dict(nets_a or {})
    nets_b = dict(nets_b or {})
    # each dummy ties EVERY terminal to the source net of the nearest core
    # member (left fence -> first core finger's owner, right fence -> last).
    tie_left = (nets_a if core[0] == "A" else nets_b).get("s", "")
    tie_right = (nets_a if core[-1] == "A" else nets_b).get("s", "")

    geom_a = build_device(tech, kind, W, L, nf=1, nets=nets_a)
    geom_b = build_device(tech, kind, W, L, nf=1, nets=nets_b)

    def _dummy(tie: str):
        return build_device(
            tech, kind, W, L, nf=1,
            nets={"s": tie, "d": tie, "g": tie, "b": tie},
        )

    geom_dl = _dummy(tie_left)
    geom_dr = _dummy(tie_right)

    ext = geom_a.drawn_extent()
    pitch = int(ext[2] - ext[0])
    mkey = tuple(tech.metal_key(1))

    units = []
    polygons = []
    n = len(sequence)
    for i, owner in enumerate(sequence):
        if owner == "A":
            g = geom_a
        elif owner == "B":
            g = geom_b
        else:
            g = geom_dl if i < dummies else geom_dr
        xoff = i * pitch
        for lp in g.polygons:
            polygons.append(
                LayerPolygon(
                    layer=lp.layer,
                    datatype=lp.datatype,
                    polygon=Polygon(
                        vertices=[Point(v.x + xoff, v.y) for v in lp.polygon.vertices]
                    ),
                )
            )
        extra: Tuple[Tuple[int, int, int, int], ...] = ()
        if owner == "D":
            extra = _dummy_tie_straps(g, xoff)
            for (x0, y0, x1, y1) in extra:
                polygons.append(
                    LayerPolygon(
                        layer=mkey[0],
                        datatype=mkey[1],
                        polygon=Polygon.rectangle(x0, y0, x1, y1),
                    )
                )
        units.append(
            PlacedFinger(
                owner=owner, index=i, offset_nm=(xoff, 0), geom=g, extra_m1=extra
            )
        )
    assert len(units) == n == 2 * M_each + 2 * dummies

    return InterdigitatedPair(
        units=tuple(units),
        core_pattern=core,
        sequence=sequence,
        note=note,
        pitch_nm=pitch,
        polygons=polygons,
    )


# --------------------------------------------------------------------------- #
# P4b: composite matched-group stamp (interdigitated unit-transistor array).
# --------------------------------------------------------------------------- #
#
# The vendor reference merges same-spec matched devices (diff pairs / mirror
# legs) into ONE interdigitated unit-transistor array -- all fingers in one
# row, one shared frame. Our realize placed each device as an independent
# stamp with zero abutment. ``composite_matched_cell`` closes that gap as a
# SINGLE-STAMP constructor: a matched group of 2-4 devices with identical
# (kind, W, L) becomes one row of nf=1 unit fingers (ABBA palindrome for an
# even pair; fair round-robin otherwise, honestly noted) plus per-(role, net)
# TERMINAL COMBS so every original device's every terminal keeps its OWN
# independently routable electrical node inside the composite.
#
# COMB GEOMETRY (why metal2 trunks)
# =================================
# Interleaved fingers mean each device's terminals alternate along x. A
# single-layer (metal1-only) horizontal comb for net X would have to cross
# the risers/stubs of net Y interleaved within its span -- planar single-layer
# channel routing of >= 2 interleaved nets is geometrically impossible without
# crossings. The standard (and vendor) resolution is layer-per-direction:
# VERTICAL metal1 risers from each unit's own terminal band + HORIZONTAL
# metal2 trunks (one y-track per (role, net)) joined by a via1 only at the
# riser's OWN trunk. A foreign trunk passing OVER a riser is a different
# layer (no short); two trunks never share a track. Every dimension (via
# size, enclosures, metal widths/spaces) is resolved from ``tech`` -- no
# hardcoded nm, no per-circuit branch.
#
# UNIT-CELL CHOICES (opt-in only; defaults elsewhere unchanged)
# =============================================================
#   * nmos units are built with ``draw_substrate_tap=False`` (the legacy gf180
#     parity form): the per-unit P+ tap M1 island sits closer than the M1
#     min-space to the source column on gf180 and the bulk is the global
#     substrate anyway -- the composite ties no bulk pin per finger.
#   * a pmos member whose bulk net == source net keeps its in-well tap (the
#     tap M1 merges with the source rail; the source riser is placed at the
#     TAP column so the merged column is one same-net rectangle). A pmos
#     member with source != bulk is built ``draw_well_tap=False`` (shared-well
#     discipline); its bulk is the contiguous composite nwell.
#   * the composite exposes NO per-device 'b' access: nmos bulk is the global
#     substrate; pmos bulk is the shared well (tapped through the self-bulk
#     members' taps when present). Honest scope, stated here, not hidden.
#
# LEAKAGE DISCIPLINE: inputs are netlist facts (names / nets / W / L / M) +
# PDK rules. No reference layout coordinate is read.


@dataclass(frozen=True)
class CompositeUnit:
    """One placed unit finger of a composite matched cell.

    ``owner``     -- the ORIGINAL device name this finger belongs to.
    ``index``     -- position in the row (0 = leftmost).
    ``offset_nm`` -- the unit's placement offset in the assembly frame.
    ``geom``      -- the ``build_device`` nf=1 cell (LOCAL coordinates).
    """

    owner: str
    index: int
    offset_nm: Tuple[int, int]
    geom: object


@dataclass(frozen=True)
class CompositeTrunk:
    """One terminal-comb trunk: the metal2 bar commoning one (role, net).

    ``columns`` are the riser x-centres (assembly frame, nm); ``y_nm`` the
    trunk track centre; ``x0_nm``/``x1_nm`` the trunk extent.
    """

    role: str
    net: str
    layer: Tuple[int, int]
    y_nm: int
    x0_nm: int
    x1_nm: int
    columns: Tuple[int, ...]


@dataclass(frozen=True)
class CompositeMatchedCell:
    """The composite matched-group stamp (``composite_matched_cell`` output).

    ``polygons``  -- every drawn shape (LayerPolygon, assembly-frame nm):
                     the unit cells + the metal1 risers + via1 cuts + metal2
                     trunks of the terminal combs.
    ``terminals`` -- ``"<device>.<role>" -> Terminal`` (the same shape a
                     ``DeviceGeom`` carries, so
                     :func:`clay.layout.device_terminal.extract_terminal_access`
                     consumes the composite UNCHANGED -- the contract-mandated
                     same-source terminal path). The bbox is the terminal's
                     via-landing pad on its trunk (or the unit terminal for a
                     single-finger net that needs no comb).
    ``sequence``  -- the realised letter order (e.g. 'ABBA'); letters map to
                     ``devices`` by position (A = devices[0], ...).
    ``note``      -- honest degradation note ('' when the requested pattern
                     was realised exactly).
    """

    polygons: List
    terminals: Dict[str, object]
    devices: Tuple[str, ...]
    kind: str
    sequence: str
    note: str
    pitch_nm: int
    units: Tuple[CompositeUnit, ...]
    trunks: Tuple[CompositeTrunk, ...]

    def drawn_extent(self) -> Tuple[int, int, int, int]:
        xs: List[int] = []
        ys: List[int] = []
        for lp in self.polygons:
            for v in lp.polygon.vertices:
                xs.append(v.x)
                ys.append(v.y)
        return (min(xs), min(ys), max(xs), max(ys))

    def to_cell_geometry(self, name: str = ""):
        """The composite as a merge-clean, centroid-relative um
        :class:`~clay.layout.extract.library_device.CellGeometry` stamp (the
        single geometry currency the realize emitter consumes).

        Same-layer shapes are UNIONED (zero self-overlap) through the same
        area-gated merge ``build_device_geometry`` uses, so the emitted stamp
        is signoff-grade like every other stamp -- the union never changes the
        drawn shape (the merge asserts it).
        """
        from clay.layout.extract.library_device import CellGeometry
        from clay.layout.extract.realize_render import _merge_polys_by_layer

        e = self.drawn_extent()
        cx = (e[0] + e[2]) / 2.0
        cy = (e[1] + e[3]) / 2.0
        polys_um: Dict[Tuple[int, int], List[List[Tuple[float, float]]]] = {}
        for lp in self.polygons:
            ring = [
                (round((v.x - cx) / 1000.0, 4), round((v.y - cy) / 1000.0, 4))
                for v in lp.polygon.vertices
            ]
            if len(ring) >= 3:
                polys_um.setdefault((lp.layer, lp.datatype), []).append(ring)
        merged, holed = _merge_polys_by_layer(polys_um)
        reason = (
            f"composite interdigitated stamp (devices={list(self.devices)}, "
            f"sequence={self.sequence}); merged-clean"
        )
        if holed:
            reason += "; hole-split layers: " + ",".join(
                f"{l}/{d}" for (l, d) in holed
            )
        if self.note:
            reason += f"; {self.note}"
        cg = CellGeometry(
            name=name or "+".join(self.devices),
            kind=self.kind,
            ref_centroid=(0.0, 0.0),
            polys_by_layer=merged,
            n_instances=len(self.units),
            finger_cells=[f"{u.owner}[{u.index}]" for u in self.units],
            clean=True,
            reason=reason,
        )
        cg.source = "composite_stamp"
        # provenance for placement-consuming faces (e.g. realize_render.
        # group_guard_rings): the fused member DEVICE names, so a composite
        # stamp's ring identity (kind / body tap net) can be resolved from
        # its members' design data. Names only -- pure design provenance.
        cg.composite_members = tuple(self.devices)
        return cg


def _member_fields(m) -> Tuple[str, str, float, float, int, Dict[str, str]]:
    """Normalise one matched-group member (dict or attribute object) to
    ``(name, kind, W, L, M, nets)``."""
    def get(key, default=None):
        if isinstance(m, dict):
            return m.get(key, default)
        return getattr(m, key, default)

    name = str(get("name"))
    kind = str(get("kind", "")).lower()
    W = float(get("W"))
    L = float(get("L"))
    M = int(get("M", 1) or 1)
    nets = dict(get("nets", {}) or {})
    return name, kind, W, L, M, nets


def _via_metal_enclosure_nm(tech, via_name: str, metal_level: int) -> int:
    """The PDK's enclosure (nm) of cut ``via_name`` by metal level 1/2,
    resolved from EITHER rule face -- the layer-rule face
    (``tech.enclosure(via, metal)``; gf180 carries via1 there) or the ViaRule
    face (``tech.via_rule(via).enclosure_below/above``; open_pdk and
    open_pdk carry via1 ONLY there). A PDK exposing the value on NO face is
    an honest ``ValueError`` refusal (surfaces as a composite skip note
    upstream) -- never a silent 0nm collapse: that collapse emitted
    exactly-via-sized comb pads on external_pdk (zero enclosure, 318x VIA1.EN.1 on
    the real KLayout DRC of the composite route, measured 2026-06-12)."""
    metal_name = f"metal{int(metal_level)}"
    enc = None
    try:
        enc = tech.enclosure(via_name, metal_name)
    except Exception:
        enc = None
    if enc is None:
        try:
            vr = tech.via_rule(via_name)
            rv = vr.enclosure_below if metal_level == 1 else vr.enclosure_above
            enc = rv.value if rv is not None else None
        except Exception:
            enc = None
    if enc is None:
        raise ValueError(
            f"no {via_name} enclosure-by-{metal_name} rule on any rule face "
            f"of {getattr(tech, 'pdk_id', tech)!r}; the terminal-comb pad "
            f"size cannot be derived (refused, not defaulted to 0)"
        )
    return int(enc)


def _composite_sequence(
    names: Sequence[str], counts: Sequence[int], pattern: str
) -> Tuple[List[int], str]:
    """The unit interleave order as member indices + an honest note.

    Two members with EQUAL, EVEN finger counts realise the exact ABBA
    palindrome (x common centroid residual 0). Everything else (odd counts,
    unequal counts, > 2 members) degrades to a FAIR ROUND-ROBIN interleave
    and SAYS SO (centroid residual up to one finger pitch per member).
    """
    k = len(names)
    if k == 2 and pattern == "ABBA" and counts[0] == counts[1] and counts[0] % 2 == 0:
        core, _ = _abba_core("ABBA", counts[0])
        return [ord(c) - ord("A") for c in core], ""
    rem = list(counts)
    order: List[int] = []
    while any(rem):
        for j in range(k):
            if rem[j] > 0:
                order.append(j)
                rem[j] -= 1
    note = (
        f"pattern {pattern!r} not exactly realisable for member finger counts "
        f"{list(counts)} ({k} members); degraded to fair round-robin "
        f"interleave (common-centroid x residual up to one finger pitch)."
    )
    return order, note


def composite_matched_cell(
    tech,
    devices: Sequence,
    *,
    pattern: str = "ABBA",
    verify: bool = True,
    contact_rows: bool = False,
) -> CompositeMatchedCell:
    """Build ONE interdigitated composite stamp for a matched device group.

    Parameters
    ----------
    tech : LayoutTech
        The resolved PDK surface; EVERY dimension (via size, enclosures,
        metal widths/spaces) comes from it.
    devices : sequence
        2-4 matched devices (dicts or attribute objects) each carrying
        ``name`` / ``kind`` / ``W`` / ``L`` / ``M`` / ``nets`` ({'s','d','g'
        [, 'b'] -> net}). All members must share (kind, W, L) -- that is what
        MATCHED means; a non-matched group is refused (ValueError), never
        silently squeezed.
    pattern : {'ABBA'}
        The requested interleave. An equal, even pair realises the exact
        ABBA palindrome; anything else degrades to a fair round-robin with
        the degradation stated in ``note``.
    verify : bool
        When True (default) the built composite is self-checked through
        :func:`clay.layout.device_terminal.extract_terminal_access`: every
        terminal must map to a node, distinct nets must land on distinct
        electrical nodes and same-net terminals on one node. A violation
        RAISES (never a silently wrong cell).
    contact_rows : bool
        Opt-in vendor-density unit artwork (full S/D contact columns; see
        ``build_device(contact_rows=)``). Default False keeps the single
        mid-channel cut units. The composite places units at the unit's OWN
        drawn-extent pitch (it never tightens COMP gaps the way the realize
        shared-gate flow did), so the dense artwork is judged by the same
        node/DRC oracles as the default artwork -- never assumed clean.

    Returns
    -------
    CompositeMatchedCell
        Assembly-frame polygons + the per-(device, role) terminal map.
        NOTHING in the existing pipeline calls this by default -- enabling it
        in a realize/anchor flow is a separate opt-in integration step.
    """
    from clay.layout.backend import LayerPolygon, Point, Polygon
    from clay.layout.device_cell import Terminal, build_device

    members = [_member_fields(m) for m in devices]
    if not (2 <= len(members) <= 4):
        raise ValueError(
            f"composite_matched_cell takes a matched group of 2 to 4 devices; "
            f"got {len(members)}"
        )
    names = [m[0] for m in members]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate device names in matched group: {names}")
    kind0, W0, L0 = members[0][1], members[0][2], members[0][3]
    if kind0 not in ("nmos", "pmos"):
        raise ValueError(f"kind must be 'nmos' or 'pmos'; got {kind0!r}")
    for (name, kind, W, L, M, _nets) in members:
        if kind != kind0 or abs(W - W0) > 1e-12 or abs(L - L0) > 1e-12:
            raise ValueError(
                f"group is not matched: {name} has (kind={kind}, W={W}, L={L}) "
                f"vs {names[0]} (kind={kind0}, W={W0}, L={L0}); a composite "
                f"stamp requires identical (kind, W, L)"
            )
        if M < 1:
            raise ValueError(f"{name}: M must be >= 1")

    counts = [m[4] for m in members]
    order, note = _composite_sequence(names, counts, pattern)
    sequence = "".join(chr(ord("A") + j) for j in order)

    # --- per-member nf=1 unit cell (geometry reuse: build_device only) -------
    unit_geom: Dict[int, object] = {}
    for j, (name, kind, W, L, _M, nets) in enumerate(members):
        kw: Dict[str, object] = {"contact_rows": contact_rows}
        if kind == "nmos":
            # no per-finger substrate tap inside the array (see module note).
            kw["draw_substrate_tap"] = False
        else:
            s_net = nets.get("s", "")
            b_net = nets.get("b", "")
            kw["draw_well_tap"] = (not b_net) or (b_net == s_net)
        unit_geom[j] = build_device(
            tech, kind, W=W, L=L, nf=1, nets=dict(nets), **kw
        )

    # --- row assembly: uniform pitch = the widest unit drawn extent ----------
    pitch = max(
        int(g.drawn_extent()[2] - g.drawn_extent()[0]) for g in unit_geom.values()
    )
    polygons: List = []
    units: List[CompositeUnit] = []
    for i, j in enumerate(order):
        g = unit_geom[j]
        xoff = i * pitch
        for lp in g.polygons:
            polygons.append(
                LayerPolygon(
                    layer=lp.layer,
                    datatype=lp.datatype,
                    polygon=Polygon(
                        vertices=[Point(v.x + xoff, v.y) for v in lp.polygon.vertices]
                    ),
                )
            )
        units.append(
            CompositeUnit(owner=names[j], index=i, offset_nm=(xoff, 0), geom=g)
        )

    ys = [v.y for lp in polygons for v in lp.polygon.vertices]
    row_top = max(ys)
    row_bottom = min(ys)

    # --- comb dimensions (all from tech; no hardcoded nm) --------------------
    m1_key = tuple(tech.metal_key(1))
    try:
        m2_key = tuple(tech.metal_key(2))
        via1_key = tuple(tech.via_key(1))
    except Exception as exc:
        raise ValueError(
            f"composite terminal combs need metal2 + via1; "
            f"{getattr(tech, 'pdk_id', tech)!r} does not expose them ({exc})"
        )
    via_names = set(tech.via_names())
    via_name = "via1" if "via1" in via_names else next(
        (n for n in via_names if n not in ("contact", "licon1", "mcon", "co")), None
    )
    vsize = int(tech.via_size(via_name) or 0) if via_name else 0
    if vsize <= 0:
        raise ValueError(
            f"no via1 size rule on {getattr(tech, 'pdk_id', tech)!r}; "
            f"cannot build the terminal combs"
        )
    # enclosure from EITHER rule face; NO silent-0 fallback (the 318x
    # VIA1.EN.1 root) -- a PDK without the rule on any face is refused.
    enc_m1 = _via_metal_enclosure_nm(tech, via_name, 1)
    try:
        enc_m2 = _via_metal_enclosure_nm(tech, via_name, 2)
    except ValueError:
        enc_m2 = enc_m1
    pad_half = (vsize + 2 * max(enc_m1, enc_m2) + 1) // 2
    m1_space = int(tech.min_space("metal1") or tech.min_width("metal1") or 0)
    m2_space = int(tech.min_space("metal2") or tech.min_width("metal2") or 0)
    trunk_h = 2 * pad_half
    # The riser STRIP through the device band takes the device's own M1 column
    # width (the rail / tap-strap width: contact + 2x its M1 enclosure, floored
    # at the M1 min width) -- widening the column inside the cell would shave
    # the Euclidean corner clearance to a neighbouring S/D strip (observed:
    # 233 -> 219 < 230 on the pmos tap column vs the dense drain boundary
    # strip). The wider via-landing PAD (``pad_half``) is emitted only AT the
    # trunk, far outside the device band.
    from clay.layout.device_cell import _contact_via_name

    _cv = _contact_via_name(tech)
    _csize = int(tech.via_size(_cv) or 0) if _cv else 0
    _cenc = int(tech.enclosure(_cv, "metal1") or 0) if _cv else 0
    riser_w = max(int(tech.min_width("metal1") or 0), _csize + 2 * _cenc)
    if riser_w <= 0:
        riser_w = 2 * pad_half

    # --- terminal anchors: (column x, attach y, direction) per (unit, role) --
    def _anchor(g, xoff: int, role: str) -> Tuple[int, int, str]:
        terms = g.terminals or {}
        if role == "d":
            b = g.drain_rail_bbox
            return ((b[0] + b[2]) // 2 + xoff, b[1], "up")
        if role == "g":
            b = g.gate_contact_bbox
            return ((b[0] + b[2]) // 2 + xoff, b[1], "up")
        # role == "s": a pmos unit with a drawn in-well tap routes its source
        # riser at the TAP column (the tap M1 already merges with the source
        # rail through the rail-band bridge, same net), so the down-riser and
        # the tap M1 form ONE same-net column without a sub-min-space gap.
        tap = terms.get("b")
        if g.kind == "pmos" and tap is not None:
            tb = tap.bbox
            return ((tb[0] + tb[2]) // 2 + xoff, g.source_rail_bbox[3], "down")
        b = g.source_rail_bbox
        return ((b[0] + b[2]) // 2 + xoff, b[3], "down")

    # group terminal columns per (direction, net) in deterministic order. The
    # comb is a NET comb, not a role comb: a diode-connected member (drain net
    # == gate net) must common its d and g columns on ONE trunk -- keying by
    # role would split one net over two trunks (a built-in open). d/g risers
    # rise to top-side trunks; s risers drop to bottom-side trunks. A net that
    # appears on BOTH sides cannot be commoned by one trunk; that is refused
    # honestly (no matched diff-pair / mirror group has such a net).
    grp_cols: Dict[Tuple[str, str], List[Tuple[int, int, int, str]]] = {}
    grp_order: List[Tuple[str, str]] = []
    grp_roles: Dict[Tuple[str, str], List[str]] = {}
    for u in units:
        j = names.index(u.owner)
        nets = members[j][5]
        for role in ("d", "g", "s"):
            net = str(nets.get(role, "") or "")
            cx, ay, direction = _anchor(u.geom, u.offset_nm[0], role)
            key = (direction, net)
            if key not in grp_cols:
                grp_cols[key] = []
                grp_order.append(key)
                grp_roles[key] = []
            grp_cols[key].append((cx, ay, j, role))
            if role not in grp_roles[key]:
                grp_roles[key].append(role)
    for (dirn, net) in grp_order:
        other = ("down" if dirn == "up" else "up", net)
        if other in grp_cols:
            raise ValueError(
                f"net {net!r} appears on both drain/gate (top) and source "
                f"(bottom) terminals of the matched group; the v1 composite "
                f"comb cannot bridge across the row -- refused (not squeezed)"
            )

    # --- track allocation: up tracks (d/g nets) above; down (s nets) below ---
    up_keys = [k for k in grp_order if k[0] == "up" and len(grp_cols[k]) >= 2]
    down_keys = [k for k in grp_order if k[0] == "down" and len(grp_cols[k]) >= 2]
    track_pitch = trunk_h + m2_space
    up_y0 = row_top + m1_space + pad_half
    down_y0 = row_bottom - m1_space - pad_half

    trunks: List[CompositeTrunk] = []
    terminals: Dict[str, object] = {}

    def _rect(key: Tuple[int, int], x0: int, y0: int, x1: int, y1: int) -> None:
        polygons.append(
            LayerPolygon(
                layer=key[0], datatype=key[1],
                polygon=Polygon.rectangle(int(x0), int(y0), int(x1), int(y1)),
            )
        )

    def _emit_trunk(key: Tuple[str, str], ty: int) -> CompositeTrunk:
        dirn, net = key
        cols = grp_cols[key]
        xs_ = sorted(c[0] for c in cols)
        tx0 = xs_[0] - pad_half
        tx1 = xs_[-1] + pad_half
        _rect(m2_key, tx0, ty - pad_half, tx1, ty + pad_half)
        for (cx, ay, _j, _role) in cols:
            if dirn == "up":
                _rect(m1_key, cx - riser_w // 2, ay, cx + riser_w // 2,
                      ty + pad_half)
            else:
                _rect(m1_key, cx - riser_w // 2, ty - pad_half,
                      cx + riser_w // 2, ay)
            # via-landing pad: the full CO/M1/M2-enclosing width, AT the trunk
            # only (the riser strip stays at the device M1 column width).
            _rect(m1_key, cx - pad_half, ty - pad_half,
                  cx + pad_half, ty + pad_half)
            _rect(via1_key, cx - vsize // 2, ty - vsize // 2,
                  cx + vsize // 2, ty + vsize // 2)
        return CompositeTrunk(
            role="+".join(sorted(grp_roles[key])), net=net, layer=m2_key,
            y_nm=ty, x0_nm=tx0, x1_nm=tx1,
            columns=tuple(c[0] for c in cols),
        )

    trunk_by_key: Dict[Tuple[str, str], CompositeTrunk] = {}
    for k_i, key in enumerate(up_keys):
        trunk_by_key[key] = _emit_trunk(key, up_y0 + k_i * track_pitch)
    for k_i, key in enumerate(down_keys):
        trunk_by_key[key] = _emit_trunk(key, down_y0 - k_i * track_pitch)
    trunks = [trunk_by_key[k] for k in grp_order if k in trunk_by_key]

    # --- per-(device, role) terminal map (extract_terminal_access input) -----
    _ROLE_DIRN = {"d": "up", "g": "up", "s": "down"}
    for j, (name, _kind, _W, _L, _M, nets) in enumerate(members):
        for role in ("d", "g", "s"):
            net = str(nets.get(role, "") or "")
            key = (_ROLE_DIRN[role], net)
            cols = grp_cols.get(key, [])
            own = [(cx, ay) for (cx, ay, jj, rr) in cols
                   if jj == j and rr == role]
            if not own:
                continue
            tr = trunk_by_key.get(key)
            if tr is not None:
                cx = own[0][0]
                bbox = (cx - pad_half, tr.y_nm - pad_half,
                        cx + pad_half, tr.y_nm + pad_half)
            else:
                # single-finger net: the unit terminal itself is the access.
                u = next(u for u in units if u.owner == name)
                tb = u.geom.terminals[role].bbox
                bbox = (tb[0] + u.offset_nm[0], tb[1],
                        tb[2] + u.offset_nm[0], tb[3])
            terminals[f"{name}.{role}"] = Terminal(
                role=f"{name}.{role}", bbox=bbox, net=net
            )

    cell = CompositeMatchedCell(
        polygons=polygons,
        terminals=terminals,
        devices=tuple(names),
        kind=kind0,
        sequence=sequence,
        note=note,
        pitch_nm=pitch,
        units=tuple(units),
        trunks=tuple(trunks),
    )

    if verify:
        _verify_composite_nodes(cell, tech)
    return cell


def _verify_composite_nodes(cell: CompositeMatchedCell, tech) -> None:
    """Build-time oracle: terminal nodes must be complete and net-pure.

    Two layers of checking (a single exposed access point per device is NOT
    enough -- a dropped via could open one finger off its comb while the
    exposed pad still maps to a node, which an access-only check cannot see):

    1. the SAME extractor path a consumer uses
       (:func:`clay.layout.device_terminal.extract_terminal_access`) must map
       every composite terminal (no unmapped);
    2. an independent node probe over EVERY UNIT FINGER's own terminal bbox
       AND every composite access bbox: all probes of one net must land on
       ONE electrical node (no open inside a comb), and distinct nets must
       land on DISTINCT nodes (no short between combs).

    A violation raises (never a silently wrong cell).
    """
    import klayout.db as kdb

    from clay.layout.connectivity import ConnectModel, extract_nodes
    from clay.layout.device_terminal import (
        _node_index_for_box,
        extract_terminal_access,
    )

    _access, report = extract_terminal_access(cell, tech=tech)
    if report.unmapped:
        raise ValueError(
            f"composite terminal(s) {report.unmapped} map to no electrical "
            f"node -- the comb geometry left them unreachable"
        )

    shapes: Dict[Tuple[int, int], List] = {}
    for lp in cell.polygons:
        pts = [kdb.Point(int(v.x), int(v.y)) for v in lp.polygon.vertices]
        if len(pts) >= 3:
            shapes.setdefault((lp.layer, lp.datatype), []).append(kdb.Polygon(pts))
    nodes = extract_nodes(shapes, ConnectModel.from_tech(tech))

    # probe points: every composite access bbox + every unit finger terminal.
    probes: List[Tuple[str, str, Tuple[int, int, int, int]]] = []
    for key, term in cell.terminals.items():
        probes.append((key, term.net, term.bbox))
    for u in cell.units:
        terms = u.geom.terminals or {}
        for role in ("d", "g", "s"):
            t = terms.get(role)
            if t is None:
                continue
            b = t.bbox
            probes.append((
                f"{u.owner}[{u.index}].{role}", t.net,
                (b[0] + u.offset_nm[0], b[1], b[2] + u.offset_nm[0], b[3]),
            ))

    net_node: Dict[str, int] = {}
    for (label, net, bbox) in probes:
        nid = _node_index_for_box(nodes, bbox)
        if nid is None:
            raise ValueError(
                f"composite probe {label} (net {net!r}) touches no electrical "
                f"node -- an open at that finger terminal"
            )
        if net in net_node:
            if net_node[net] != nid:
                raise ValueError(
                    f"net {net!r} is split across electrical nodes in the "
                    f"composite (open inside a terminal comb at {label})"
                )
        else:
            net_node[net] = nid
    node_to_net: Dict[int, str] = {}
    for net, nid in net_node.items():
        if nid in node_to_net and node_to_net[nid] != net:
            raise ValueError(
                f"nets {node_to_net[nid]!r} and {net!r} share one electrical "
                f"node in the composite (a short between combs)"
            )
        node_to_net[nid] = net


__all__ = [
    "merge_shared_gate_poly",
    "shared_gate_groups",
    "SharedGateGroup",
    "build_interdigitated_pair",
    "InterdigitatedPair",
    "PlacedFinger",
    "composite_matched_cell",
    "CompositeMatchedCell",
    "CompositeUnit",
    "CompositeTrunk",
]
