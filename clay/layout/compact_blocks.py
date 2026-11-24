"""Deterministic inter-cluster COMPACTION post-pass (placement geometry only).

The generic placer returns a FEASIBLE floorplan that minimizes a W+H perimeter
proxy; it leaves large inter-cluster WHITESPACE (the OTA place-only cell is
~183x113um vs a 133x80um reference even though each folded cluster is dense).
This module squeezes the already-placed, already-rigid cluster BLOCKS together on
the real emitted geometry -- no routing, no per-cluster re-solve, no reference
coordinate.

A BLOCK is one rigid unit that moves as a whole:
  * each declared ``@guard_ring`` island (its member device/tile stamps + the
    materialized ``GRING`` ring stamp) -- ONE block;
  * each standalone stamp that belongs to no island (e.g. a MIM cap) -- ONE block.

:func:`compact_blocks` is the pure, unit-testable core: it operates on block
bounding boxes only (a list of ``(block_id, bbox_nm)``) and returns a per-block
``(dx_nm, dy_nm)`` translation. The algorithm is a deterministic shadow-based 1-D
compaction (alternating PUSH-DOWN / PUSH-LEFT sweeps): no wall-clock, no
randomness, ties broken by ``block_id``. Each half-sweep provably yields an
overlap-free configuration, so the result never introduces an overlap.

:func:`compacted_pum` wraps the core for the OTA bridge geometry: it derives the
blocks from the bridged stamp bboxes (mapping each stamp to its island through the
device name / ``composite_members`` provenance the rest of the pipeline already
uses), compacts them, shifts every stamp of each block by its block's delta, and
re-materializes the guard rings around the shifted members. It returns the new
``(placement_um, geometry)`` -- the SAME pair every downstream emit/verify consumer
reads, no second geometry path.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

_Bbox = Tuple[int, int, int, int]  # (x0, y0, x1, y1) nm
_NM_PER_UM = 1000.0


# --------------------------------------------------------------------------- #
# pure geometry helpers (nm int bboxes)
# --------------------------------------------------------------------------- #
def _union(bboxes: Sequence[_Bbox]) -> _Bbox:
    """Axis-aligned union of a non-empty sequence of nm bboxes."""
    xs0 = min(b[0] for b in bboxes)
    ys0 = min(b[1] for b in bboxes)
    xs1 = max(b[2] for b in bboxes)
    ys1 = max(b[3] for b in bboxes)
    return (xs0, ys0, xs1, ys1)


def _overlap_area(a: _Bbox, b: _Bbox) -> int:
    """Area (nm^2) of the intersection of two bboxes; 0 if disjoint."""
    ox = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    oy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return ox * oy


def _overlaps(a: _Bbox, b: _Bbox) -> bool:
    """True iff two bboxes overlap in BOTH axes (share positive-area region)."""
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _envelope(bboxes: Dict[str, _Bbox]) -> Tuple[float, float, float]:
    """(W_um, H_um, area_um2) of the union of all stamp bboxes."""
    xs0 = min(b[0] for b in bboxes.values())
    ys0 = min(b[1] for b in bboxes.values())
    xs1 = max(b[2] for b in bboxes.values())
    ys1 = max(b[3] for b in bboxes.values())
    w = (xs1 - xs0) / _NM_PER_UM
    h = (ys1 - ys0) / _NM_PER_UM
    return (w, h, w * h)


# --------------------------------------------------------------------------- #
# the deterministic shadow compaction core (PURE -- unit-testable on synthetic bboxes)
# --------------------------------------------------------------------------- #
def compact_blocks(
    blocks: Sequence[Tuple[str, _Bbox]], gap_nm: int
) -> Dict[str, Tuple[int, int]]:
    """Squeeze rigid blocks together; return ``{block_id -> (dx_nm, dy_nm)}``.

    ``blocks`` is a list of ``(block_id, (x0, y0, x1, y1))`` in nm. ``gap_nm`` is
    the inter-block clearance kept between any two touching blocks.

    Algorithm (deterministic shadow-based 1-D compaction), repeated until the
    positions stop changing or 8 iterations elapse:

      * PUSH-DOWN: process blocks in ``(y0, x0, id)`` order; each block B drops to
        ``floor_y = max`` over ALREADY-processed x-overlapping blocks A of
        ``A.y1 + gap`` (else the global min y). x-overlap is measured on CURRENT
        positions (``A.x0 < B.x1 and B.x0 < A.x1``).
      * PUSH-LEFT: process in ``(x0, y0, id)`` order; each block B slides to
        ``wall_x = max`` over ALREADY-processed y-overlapping A of ``A.x1 + gap``
        (else the global min x).

    Each half-sweep leaves an overlap-free configuration (any pair that overlaps
    on the swept axis is separated on the other), so the returned deltas never
    introduce an overlap. Determinism: no clock, no RNG; every sort tie is broken
    by ``block_id``.
    """
    gap = int(gap_nm)
    if not blocks:
        return {}
    orig: Dict[str, _Bbox] = {bid: tuple(bb) for bid, bb in blocks}  # type: ignore[misc]
    dims: Dict[str, Tuple[int, int]] = {
        bid: (bb[2] - bb[0], bb[3] - bb[1]) for bid, bb in blocks
    }
    pos: Dict[str, List[int]] = {bid: list(bb) for bid, bb in blocks}
    ids = [bid for bid, _ in blocks]

    def x_overlap(a: List[int], b: List[int]) -> bool:
        return a[0] < b[2] and b[0] < a[2]

    def y_overlap(a: List[int], b: List[int]) -> bool:
        return a[1] < b[3] and b[1] < a[3]

    for _ in range(8):
        snapshot = {bid: tuple(pos[bid]) for bid in ids}

        # ---- PUSH-DOWN --------------------------------------------------- #
        gmin_y = min(pos[bid][1] for bid in ids)
        order = sorted(ids, key=lambda bid: (pos[bid][1], pos[bid][0], bid))
        placed: List[str] = []
        for bid in order:
            b = pos[bid]
            floor_y = gmin_y
            for aid in placed:
                a = pos[aid]
                if x_overlap(a, b):
                    floor_y = max(floor_y, a[3] + gap)
            h = dims[bid][1]
            b[1] = floor_y
            b[3] = floor_y + h
            placed.append(bid)

        # ---- PUSH-LEFT --------------------------------------------------- #
        gmin_x = min(pos[bid][0] for bid in ids)
        order = sorted(ids, key=lambda bid: (pos[bid][0], pos[bid][1], bid))
        placed = []
        for bid in order:
            b = pos[bid]
            wall_x = gmin_x
            for aid in placed:
                a = pos[aid]
                if y_overlap(a, b):
                    wall_x = max(wall_x, a[2] + gap)
            w = dims[bid][0]
            b[0] = wall_x
            b[2] = wall_x + w
            placed.append(bid)

        if all(tuple(pos[bid]) == snapshot[bid] for bid in ids):
            break

    return {
        bid: (pos[bid][0] - orig[bid][0], pos[bid][1] - orig[bid][1]) for bid in ids
    }


def assert_no_block_overlap(blocks: Sequence[Tuple[str, _Bbox]]) -> None:
    """Hard-assert that no two block bboxes overlap (share positive area).

    Raises ``AssertionError`` naming the first overlapping pair -- the compaction
    contract: a compacted floorplan the rings/emit will honor must be overlap-free.
    """
    items = list(blocks)
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            (ai, abb), (bj, bbb) = items[i], items[j]
            if _overlaps(abb, bbb):
                ov = _overlap_area(abb, bbb)
                raise AssertionError(
                    f"block overlap: {ai} {abb} <> {bj} {bbb} "
                    f"(intersection {ov} nm^2 = {ov / 1e6:.2f} um^2)"
                )


# --------------------------------------------------------------------------- #
# OTA bridge-geometry wrapper: bridged stamps -> blocks -> compact -> shifted pum
# --------------------------------------------------------------------------- #
def _stamp_islands(
    geom: Dict[str, Any],
    guard_ring_intents: Sequence[Any],
    gring_names: set,
) -> Dict[str, int]:
    """Map each non-GRING stamp -> the index of the declared island it belongs to.

    A stamp identifies its island through the SAME provenance the rest of the
    pipeline uses: a device stamp is named by its device; an array / common-centroid
    tile stamp lists its consumed devices on ``composite_members``. The stamp joins
    the island with which it shares the most member devices (ties -> lowest island
    index). A stamp that matches no island (a standalone cap) is absent from the map.
    """
    out: Dict[str, int] = {}
    for name, cg in geom.items():
        if name in gring_names:
            continue
        keyset = {name} | set(getattr(cg, "composite_members", ()) or ())
        best_i, best_c = None, 0
        for i, gr in enumerate(guard_ring_intents):
            members = set(getattr(gr, "members", ()) or ())
            c = len(keyset & members)
            if c > best_c:
                best_i, best_c = i, c
        if best_i is not None:
            out[name] = best_i
    return out


def extract_blocks(
    geom: Dict[str, Any],
    pum: Dict[str, Tuple[float, float]],
    intent: Any,
) -> Tuple[List[Tuple[str, _Bbox]], Dict[str, set], Dict[str, str], set]:
    """Derive the rigid BLOCKS from the bridged OTA geometry (the SINGLE block model
    shared by the 1-D shadow compactor and the 2-D re-placer -- no second definition).

    ``geom`` / ``pum`` are the bridged geometry + placement (WITH the guard-ring
    ``GRING`` stamps already materialized). Returns
    ``(blocks, block_stamps, block_gring, gring_names)``:

      * ``blocks``       -- ``[(block_id, bbox_nm), ...]``, ONE rigid unit each: every
        declared ``@guard_ring`` island (its member device/tile stamps unioned with
        that island's ``GRING`` ring stamp) and every standalone non-island stamp (a
        MIM cap). Each bbox is the REAL ring-inclusive extent.
      * ``block_stamps`` -- ``{block_id -> {device/tile stamp names}}`` (NOT the GRING).
      * ``block_gring``  -- ``{block_id -> its GRING stamp name}`` for ringed islands.
      * ``gring_names``  -- the set of materialized GRING stamp names in ``geom``.
    """
    from clay.layout.extract.realize_render import _placed_stamp_bboxes_nm

    grs = list(getattr(intent, "guard_rings", ()) or ())
    bboxes = _placed_stamp_bboxes_nm(geom, pum)
    gring_names = {
        n for n, cg in geom.items()
        if str(getattr(cg, "source", "")) == "group_guard_ring" or n.startswith("GRING")
    }
    proven_ring_owner: Dict[str, str] = {}
    proven_ring_members: Dict[str, Tuple[str, ...]] = {}
    provenance_names = {
        name
        for name in gring_names
        if getattr(geom[name], "guard_ring_intent_sha256", "") != ""
    }
    if provenance_names and provenance_names != gring_names:
        raise ValueError("materialized guard-ring provenance is incomplete")
    for name in sorted(provenance_names):
        digest = getattr(geom[name], "guard_ring_intent_sha256", "")
        members = getattr(geom[name], "guard_ring_declared_members", ())
        if (
            type(digest) is not str
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
            or type(members) is not tuple
            or not members
            or members != tuple(sorted(set(members)))
            or any(type(member) is not str or not member for member in members)
        ):
            raise ValueError("materialized guard-ring provenance is malformed")
        if digest in proven_ring_owner:
            raise ValueError("materialized guard-ring declaration owner is not unique")
        proven_ring_owner[digest] = name
        proven_ring_members[name] = members

    # ---- each non-GRING stamp -> its island (device-name / composite provenance)
    stamp_island = _stamp_islands(geom, grs, gring_names)

    # ---- build blocks: one per island (members + its GRING), one per leftover stamp
    blocks: List[Tuple[str, _Bbox]] = []
    block_stamps: Dict[str, set] = {}   # block_id -> device/tile stamp names (NOT GRING)
    block_gring: Dict[str, str] = {}    # block_id -> its GRING stamp name
    used_gring: set = set()
    for i, gr in enumerate(grs):
        mstamps = {s for s, ii in stamp_island.items() if ii == i and s in bboxes}
        if not mstamps:
            continue  # island with no materialized geometry -> honest gap, no block
        member_union = _union([bboxes[s] for s in mstamps])
        # A provenance-bearing materializer binds by declaration SHA.  This is
        # essential when one shared outer ring geometrically overlaps unrelated
        # local islands: largest-bbox overlap is only the legacy fallback for
        # historical captured geometry that carries no owner metadata.
        if provenance_names:
            expected_sha256 = getattr(gr, "sha256", "")
            if type(expected_sha256) is not str or not expected_sha256:
                raise ValueError("source guard-ring declaration lacks exact ownership")
            best_gk = proven_ring_owner.get(expected_sha256)
            if best_gk is not None and proven_ring_members[best_gk] != tuple(
                sorted(getattr(gr, "members", ()) or ())
            ):
                raise ValueError("materialized guard-ring members differ from source")
        else:
            best_gk, best_ov = None, 0
            for gk in gring_names:
                if gk in used_gring or gk not in bboxes:
                    continue
                ov = _overlap_area(member_union, bboxes[gk])
                if ov > best_ov:
                    best_gk, best_ov = gk, ov
        bid = f"ISL{i}"
        bb = member_union
        if best_gk is not None:
            used_gring.add(best_gk)
            block_gring[bid] = best_gk
            bb = _union([member_union, bboxes[best_gk]])
        block_stamps[bid] = set(mstamps)
        blocks.append((bid, bb))

    assigned = set().union(*block_stamps.values()) if block_stamps else set()
    for name in geom:
        if name in gring_names or name in assigned or name not in bboxes:
            continue
        bid = f"BLK_{name}"
        block_stamps[bid] = {name}
        blocks.append((bid, bboxes[name]))

    return blocks, block_stamps, block_gring, gring_names


def apply_block_deltas(
    geom: Dict[str, Any],
    pum: Dict[str, Tuple[float, float]],
    intent: Any,
    tech: Any,
    tech_id: str,
    deltas: Dict[str, Tuple[int, int]],
    block_stamps: Dict[str, set],
    block_gring: Dict[str, str],
    gring_names: set,
    bi: Any = None,
    member_ring_pad_top_nm: Any = None,
    member_ring_pad_bottom_nm: Any = None,
) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, Any]]:
    """Move each block by its ``(dx_nm, dy_nm)`` delta and rebuild the guard rings on
    the shifted members. Returns the new ``(placement_um, geometry)`` -- the SAME pair
    every downstream emit/verify consumer reads (no second geometry path). Shared by
    :func:`compacted_pum` (1-D shadow deltas) and ``place_blocks_2d.reposition_pum_2d``
    (absolute 2-D re-placement deltas): only the delta SOURCE differs, the apply is one.

    * shift every device/tile stamp of each block by its block's ``(dx, dy)``
      (translate ``pum``); then
    * with ``bi`` given, drop the old ``GRING`` stamps and RE-MATERIALIZE the rings
      around the shifted members (``_materialize_guard_rings``) -- or, without ``bi``,
      translate each block's ``GRING`` polys rigidly by the block delta.

    ``member_ring_pad_top_nm`` (opt-in, ``bi``-path only): ``{member name -> extra nm}``
    added to that member's ring cluster INNER top edge, so a band's rings can be
    HEIGHT-ALIGNED to one common extent (only the ring frame grows; the enclosed
    device geometry is untouched). ``None`` -> byte-identical.

    ``member_ring_pad_bottom_nm`` is the downward-symmetric reservation used by an
    internal bottom edge rail.  It grows only a selected same-net ring frame below
    the devices, leaving all enclosed terminals at their original coordinates.
    """
    grs = getattr(intent, "guard_rings", ())
    if type(grs) is not tuple:
        raise TypeError("guard-ring inventory must be an exact tuple")

    # ---- shift every device/tile stamp of each block (translate pum)
    new_pum = dict(pum)
    for bid, stamps in block_stamps.items():
        dx, dy = deltas[bid]
        if dx == 0 and dy == 0:
            continue
        for s in stamps:
            if s in gring_names:
                continue
            px, py = new_pum.get(s, (0.0, 0.0))
            # A stamp centre may legitimately carry a half-nm phase after an
            # integer-nm primitive is re-centred on an odd-sized bbox.  Keep
            # that phase through block motion.  The only quantisation boundary
            # is the final composed local+placed vertex; rounding the centre
            # here independently deforms otherwise rigid device artwork.
            new_pum[s] = (
                float(px) + int(dx) / _NM_PER_UM,
                float(py) + int(dy) / _NM_PER_UM,
            )

    # ---- rebuild the guard rings around the shifted members
    new_geom = dict(geom)
    if bi is not None:
        # re-materialize (cleaner): drop the old ring stamps, rebuild on the shifted pum
        from clay.layout.route.route_cell import _materialize_guard_rings
        for gk in gring_names:
            new_geom.pop(gk, None)
            new_pum.pop(gk, None)
        if grs:
            _materialize_guard_rings(
                grs, bi, new_pum, new_geom, tech, tech_id,
                member_ring_pad_top_nm=member_ring_pad_top_nm,
                member_ring_pad_bottom_nm=member_ring_pad_bottom_nm,
            )
    else:
        # fallback (no bridged intent): translate each block's GRING polys rigidly
        import copy as _copy
        for bid, gk in block_gring.items():
            dx, dy = deltas[bid]
            if dx == 0 and dy == 0:
                continue
            dux, duy = dx / _NM_PER_UM, dy / _NM_PER_UM
            cg = new_geom[gk]
            cg2 = _copy.copy(cg)
            cg2.polys_by_layer = {
                key: [[(x + dux, y + duy) for (x, y) in ring] for ring in rings]
                for key, rings in cg.polys_by_layer.items()
            }
            new_geom[gk] = cg2

    return new_pum, new_geom


def compacted_pum(
    geom: Dict[str, Any],
    pum: Dict[str, Tuple[float, float]],
    intent: Any,
    tech: Any,
    tech_id: str,
    gap_um: float,
    bi: Any = None,
) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, Any]]:
    """Compact the bridged OTA stamps and return the new ``(placement_um, geometry)``.

    Steps: (1-2) derive the rigid blocks with :func:`extract_blocks`; (3) compact the
    block bboxes with the 1-D shadow core :func:`compact_blocks` (``gap_nm =
    gap_um*1000``); (4) HARD-ASSERT the compacted block bboxes do not overlap; (5)
    apply the per-block deltas + re-materialize the rings via :func:`apply_block_deltas`.

    Returns ``(new_pum, new_geom)``.
    """
    gap_nm = int(round(float(gap_um) * _NM_PER_UM))

    # ---- 1-2. derive the rigid blocks (shared model)
    blocks, block_stamps, block_gring, gring_names = extract_blocks(geom, pum, intent)

    # ---- 3. compact + 4. assert no overlap on the compacted block bboxes
    deltas = compact_blocks(blocks, gap_nm)
    compacted = [
        (bid, (bb[0] + deltas[bid][0], bb[1] + deltas[bid][1],
               bb[2] + deltas[bid][0], bb[3] + deltas[bid][1]))
        for bid, bb in blocks
    ]
    assert_no_block_overlap(compacted)

    # ---- 5. shift stamps + rebuild rings (shared apply)
    return apply_block_deltas(
        geom, pum, intent, tech, tech_id,
        deltas, block_stamps, block_gring, gring_names, bi,
    )
