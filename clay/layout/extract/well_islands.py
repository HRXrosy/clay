"""Well-island grouping -- an INTENT-layer concept that drives placement.

Root-cure architecture (2026-06-14): a guard ring groups devices that share a
bulk/well net into a contiguous, isolated region (a "well-island"). WHICH devices
form an island is a DESIGN INTENT (topology-derived: same well-polarity + same
body net), exactly like @symmetric / @match -- it belongs in the intent layer and
must DRIVE the placement, not be reverse-engineered from an already-placed (and
interleaved) layout. The signed external_pdk reference rings at band granularity: one big
ring per well-island band (NMOS p-substrate band, PMOS n-well bands), enclosing
the band's devices, with signals routing freely inside. The earlier post-placement
spatial clustering failed precisely because the placement did not band by island;
every ring-side clustering scheme then overlapped / shorted / choked routing.

This module derives the well-islands from netlist topology and expresses them as a
:class:`~clay.layout.seqpair_model.RelativeOrder` whose ROWS are the islands
(bottom-to-top), so the existing style-row placement path bands by island. Pure
design data (kind + body net), no coordinate -- the same leakage-free contract the
rest of the intent layer holds.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

_MOS = ("nmos", "pmos")


def well_islands(intent: Any) -> List[Tuple[str, str, Tuple[str, ...]]]:
    """The well-islands of ``intent`` as ``(kind, body_net, members)`` tuples.

    A MOS island = the maximal set of same-kind devices sharing a body net (its
    bulk region). NMOS share the p-substrate; a same-body NMOS run is one island.
    PMOS sit in n-wells, one well per body net, so PMOS split by body net. Pure
    topology: groups by (kind, body net) from the netlist, never a coordinate.
    Deterministic: sorted by (kind, body), members name-sorted."""
    buckets: Dict[Tuple[str, str], List[str]] = {}
    for d in (getattr(intent, "devices", None) or []):
        kind = str(getattr(d, "kind", "") or "")
        if kind not in _MOS:
            continue
        body = str((getattr(d, "nets", None) or {}).get("body", "") or "")
        buckets.setdefault((kind, body), []).append(str(d.name))
    out: List[Tuple[str, str, Tuple[str, ...]]] = []
    for (kind, body) in sorted(buckets):
        out.append((kind, body, tuple(sorted(buckets[(kind, body)]))))
    return out


def declared_guard_islands(intent: Any) -> List[Tuple[Tuple[str, ...], str, str]]:
    """The DECLARED ``@guard_ring`` islands as ``(members, well, tie)`` triples.

    Reads whichever declaration face the intent carries, in order:

      * ``ExtractedIntent.guard_islands`` -- dict entries
        ``{'members', 'well', 'tie'}`` carried across the ``LayoutIntent`` ->
        ``ExtractedIntent`` boundary (the compact-placement path); then
      * ``LayoutIntent.guard_rings`` -- ``GuardRingIntent`` records
        (``.members`` / ``.well`` / ``.tie_net``), so an intent that still
        carries the raw declaration works with no conversion step.

    Empty when NEITHER is declared -- the byte-identical (no-@guard_ring) path.
    Pure declared data (member labels + polarity + tie net); no coordinate is
    read, so the leakage-free contract the rest of the intent layer holds is
    preserved."""
    out: List[Tuple[Tuple[str, ...], str, str]] = []
    for e in (getattr(intent, "guard_islands", None) or ()):
        if isinstance(e, dict):
            members = tuple(str(m) for m in (e.get("members") or ()))
            well = str(e.get("well", "") or "")
            tie = str(e.get("tie", e.get("tie_net", "")) or "")
        else:  # a GuardRingIntent-shaped entry stored directly
            members = tuple(str(m) for m in (getattr(e, "members", ()) or ()))
            well = str(getattr(e, "well", "") or "")
            tie = str(getattr(e, "tie", getattr(e, "tie_net", "")) or "")
        if members:
            out.append((members, well, tie))
    if out:
        return out
    for g in (getattr(intent, "guard_rings", None) or ()):
        members = tuple(str(m) for m in (getattr(g, "members", ()) or ()))
        if not members:
            continue
        well = str(getattr(g, "well", "") or "")
        tie = str(getattr(g, "tie_net", getattr(g, "tie", "")) or "")
        out.append((members, well, tie))
    return out


def island_of(intent: Any) -> Dict[str, int]:
    """``member device name -> declared-island index`` (from the DECLARED
    ``@guard_ring`` islands). A member in no declared island is ABSENT, so a
    caller uses ``island_of(intent).get(name)`` -> ``None``.

    Empty when no ``@guard_ring`` is declared -> the compact fold key gains a
    constant ``None`` and the partition is byte-identical (back-compat). The
    FIRST declaring island wins if a member is (erroneously) listed twice."""
    out: Dict[str, int] = {}
    for i, (members, _well, _tie) in enumerate(declared_guard_islands(intent)):
        for m in members:
            out.setdefault(m, i)
    return out


def compact_matched_groups(intent: Any) -> List[Tuple[str, str, Tuple[str, ...]]]:
    """The COMPACT-MATCHED-GROUP partition of the design's MOS as
    ``(kind, body_net, members)`` tuples -- the per-group ring granularity the
    signed reference rings at (~9 tight rings on the OTA), NOT the coarse
    (kind, body) well-island banding (which collapses to ~3 bands).

    A compact matched group is ONE physically-packed matched set:

      * each MIRROR GROUP (connected component of ``mirror_pairs`` -- a
        shared-gate matched set) of >= 2 real MOS is ONE group;
      * each DIFF PAIR not already inside a mirror group is ONE group;
      * every leftover SINGLETON MOS is folded, by its (kind, body net) class,
        into ONE group per leftover (kind, body) class -- same-bulk singletons
        ring together (they place adjacent in their band). This is the
        "physical adjacency" expressed from topology (same well + same bulk
        net => adjacency-eligible), so it stays coordinate-free and
        translation-invariant.

    Pure topology (mirror_pairs / diff_pairs / kind + body); no coordinate is
    read, so a uniform translation of the reference bboxes leaves the partition
    identical (the leakage guard). The ``(kind, body)`` of a multi-member group
    is its members' common kind + common body net (matched sets share both by
    construction). Deterministic: groups sorted by (kind, body, members),
    members name-sorted.

    This is the SAME partition logic :func:`compact_matched_group_count`
    counts -- that count is now ``len(compact_matched_groups(intent))`` (one
    source, no second path).

    REFINE-NEVER-STRADDLE (declared ``@guard_ring`` islands): when the intent
    declares islands (:func:`declared_guard_islands`), every emitted group is a
    SUBSET of exactly one declared island. The topology fold keys on
    ``(kind, body)`` alone, so it would fuse same-kind, same-body singletons that
    the design placed in DIFFERENT declared rings (the straddle proved RED by
    ``clay.layout.route.ring_integrity_check`` -- 12/20 stranded taps). Here the
    step-3 fold key gains the member's declared-island id and a matched
    (mirror/diff) set that spans two islands is SPLIT per island, so a fused
    group can never cross a declared ring boundary. When NO island is declared
    ``island_of`` is empty, the fold key gains a constant ``None`` and the split
    is a no-op -> the partition is BYTE-IDENTICAL (back-compat)."""
    from clay.layout.extract.topo_sketch import mirror_groups

    devs = {str(d.name): d for d in (getattr(intent, "devices", None) or [])}
    mos = {n: d for n, d in devs.items()
           if str(getattr(d, "kind", "") or "") in _MOS}

    def _kb(name: str) -> Tuple[str, str]:
        d = mos[name]
        kind = str(getattr(d, "kind", "") or "")
        body = str((getattr(d, "nets", None) or {}).get("body", "") or "")
        return kind, body

    # DECLARED-island partition: member -> declared ring id (absent when the
    # member is in no declared ring / no @guard_ring declared -> empty map).
    isl_of = island_of(intent)

    def _split_by_island(members: Tuple[str, ...]) -> List[Tuple[str, ...]]:
        """Partition ``members`` by declared-island id (``None`` = no declared
        island), each piece name-sorted. Deterministic; a single-island (or
        island-free) set returns ONE piece == ``members`` (byte-identical)."""
        by: Dict[Any, List[str]] = {}
        for m in members:
            by.setdefault(isl_of.get(m), []).append(m)
        return [tuple(sorted(v)) for _k, v in
                sorted(by.items(), key=lambda kv: (kv[0] is None, kv[0]))]

    out: List[Tuple[str, str, Tuple[str, ...]]] = []
    claimed: set = set()

    # 1. mirror groups (>= 2 real MOS members), each SPLIT to stay within one
    #    declared island. An island-piece with a lone member is left UNCLAIMED
    #    so step 3 folds it with its same-island same-bulk mates (never a
    #    cross-island matched group).
    for g in mirror_groups(intent):
        members = tuple(sorted(m for m in g.members if m in mos))
        if len(members) < 2:
            continue
        for sub in _split_by_island(members):
            if len(sub) < 2:
                continue
            kind, body = _kb(sub[0])
            out.append((kind, body, sub))
            claimed.update(sub)

    # 2. diff pairs not already inside a mirror group (likewise split by island).
    for pair in (getattr(intent, "diff_pairs", None) or []):
        members = tuple(sorted(m for m in (str(pair[0]), str(pair[1]))
                               if m in mos))
        if len(members) < 2 or any(m in claimed for m in members):
            continue
        for sub in _split_by_island(members):
            if len(sub) < 2 or any(m in claimed for m in sub):
                continue
            kind, body = _kb(sub[0])
            out.append((kind, body, sub))
            claimed.update(sub)

    # 3. leftover singletons folded by (kind, body, DECLARED-island) -- one group
    #    per leftover (kind, body) class WITHIN one declared ring (same-bulk
    #    singletons ring together only when they share the declared island; the
    #    island id is a constant None when no @guard_ring is declared, so this is
    #    byte-identical to the old (kind, body) fold in that case).
    leftover: Dict[Tuple[str, str, Any], List[str]] = {}
    for n in mos:
        if n in claimed:
            continue
        kind, body = _kb(n)
        leftover.setdefault((kind, body, isl_of.get(n)), []).append(n)
    for (kind, body, _isl), members in leftover.items():
        out.append((kind, body, tuple(sorted(members))))

    return sorted(out, key=lambda kbm: (kbm[0], kbm[1], kbm[2]))


def well_island_order(intent: Any, *, cell: str = "OTA"):
    """A :class:`RelativeOrder` whose ROWS are the well-islands (band-per-island),
    so the placement bands by island and a band-level ring per island is clean by
    construction.

    Band stack (bottom-to-top): NMOS islands (p-substrate), then a passives band
    (resistors/caps -- ringless, placed between the bulk bands like the reference's
    mid band), then PMOS islands (n-wells). Order within a kind is by body net
    (deterministic). Members within an island keep name order (within-row order is
    free for the solver). Passives that the design declares are appended as their
    own band so they get placed (and route to their nets across bands)."""
    from clay.layout.seqpair_model import RelativeOrder

    isl = well_islands(intent)
    nmos_rows = [members for (k, _b, members) in isl if k == "nmos"]
    pmos_rows = [members for (k, _b, members) in isl if k == "pmos"]
    passives = tuple(sorted(str(p.name)
                            for p in (getattr(intent, "passives", None) or [])))

    rows: List[Tuple[str, ...]] = []
    rows.extend(nmos_rows)
    if passives:
        rows.append(passives)
    rows.extend(pmos_rows)
    rows = [r for r in rows if r]
    devices = tuple(d for r in rows for d in r)
    return RelativeOrder(cell=cell, devices=devices, rows=tuple(rows))


def compact_matched_group_order(intent: Any, *, cell: str = "OTA",
                                groups_per_row: int = 0,
                                sizes: Any = None,
                                aspect_target: Any = None):
    """A :class:`RelativeOrder` whose ROWS pack the COMPACT-MATCHED-GROUPS
    (``compact_matched_groups``) several-per-row, each group kept X-CONTIGUOUS.

    This is the placement banding the per-compact-group ring granularity needs
    (~9 tight rings, the signed reference's granularity, enclosure ~85%). The
    coarse :func:`well_island_order` bands by WHOLE well-island (3 bands -> 3
    coarse band-rings); this finer order keeps every compact group's members
    consecutive within ONE row so :func:`group_guard_rings` can wrap one tight,
    NON-OVERLAPPING frame per group (the F2 contract).

    Contiguity contract (the placement-layer blocker F2 r1 named): each compact
    group is a CONTIGUOUS run within its row (members are listed consecutively),
    and a row never splits a group across rows. Multiple groups share a row to
    keep the cell LANDSCAPE (the reference aspect ~1.67) instead of a tall
    9-row portrait stack -- but a group is never broken to balance a row.

    Band stack (bottom-to-top): NMOS compact groups (p-substrate) packed into
    rows, then a passives band (ringless, the reference mid band), then PMOS
    compact groups (n-wells) packed into rows. Within a kind the groups are in
    ``compact_matched_groups`` deterministic order.

    PACKING (P-B 2D band-grid -- design-derived width target): when ``sizes``
    (``{name -> (w, h)}`` group-member footprints) AND ``aspect_target`` (the
    reference-calibrated cell W/H band CENTER -- a scale-free proportion, a
    CONFIG scalar the reference only calibrates, never a coordinate) are given,
    each kind's groups pack into rows against a TARGET ROW WIDTH solved so the
    emitted cell aspect approaches the band: a deterministic scan of target
    widths (group widths are the only design input; an indivisible wide group
    sets the width floor) picks the packing whose estimated cell aspect is
    log-closest to ``aspect_target``. This drives the cell LANDSCAPE without a
    fixed groups-per-row count (the ceil(sqrt) default lands portrait for this
    design). ``groups_per_row > 0`` forces a fixed count (probe/override).
    ``sizes=None`` keeps the ceil(sqrt) default BYTE-IDENTICAL.

    Pure topology (mirror_pairs / diff_pairs / kind + body via
    ``compact_matched_groups``) + design footprint widths + a config scalar;
    no coordinate is read, so a uniform translation of the reference bboxes
    leaves the order identical (the leakage guard)."""
    import math

    from clay.layout.seqpair_model import RelativeOrder

    cg = compact_matched_groups(intent)
    nmos_groups = [members for (k, _b, members) in cg if k == "nmos"]
    pmos_groups = [members for (k, _b, members) in cg if k == "pmos"]
    passives = tuple(sorted(str(p.name)
                            for p in (getattr(intent, "passives", None) or [])))

    _DEV_GAP = 1.0      # within-row group-to-group gap (sketch dev gap)
    _ROW_CH = 2.0       # inter-row channel (sketch row channel)
    _PASS_H = 5.0       # passive-band height estimate (ringless mid band)

    def _gh(members: Tuple[str, ...]) -> float:
        if not sizes:
            return 1.0
        return max((float(sizes.get(m, (1.0, 1.0))[1]) for m in members),
                   default=1.0)

    def _pack_count(groups: List[Tuple[str, ...]], per: int
                    ) -> List[Tuple[str, ...]]:
        """Pack ``per`` whole groups per row (a group is never split)."""
        rows: List[Tuple[str, ...]] = []
        for i in range(0, len(groups), per):
            chunk = groups[i:i + per]
            rows.append(tuple(m for g in chunk for m in g))
        return rows

    if sizes and aspect_target and (nmos_groups or pmos_groups):
        # P-B band-grid: enumerate the per-kind GROUPS-PER-ROW combinations and
        # pick the one whose ESTIMATED cell aspect is log-closest to the
        # calibrated band center. A single uniform target width cannot reach the
        # band for a design whose two kinds have very different total widths (the
        # band can require nmos in N rows while pmos stays in 1 wide row); a
        # per-kind row-count enumeration can. This is a bounded deterministic
        # SEARCH over design footprints + a config band scalar, NOT a per-circuit
        # constant (no `if cell==`, no hardcoded count).
        def _cell_estimate(n_per: int, p_per: int):
            n_rows = _pack_count(nmos_groups, n_per) if nmos_groups else []
            p_rows = _pack_count(pmos_groups, p_per) if pmos_groups else []
            flat = list(n_rows) + ([passives] if passives else []) + list(p_rows)
            if not flat:
                return None
            W = max(
                _row_w(r, _gw_member=lambda m: float(
                    sizes.get(m, (1.0, 1.0))[0]), gap=_DEV_GAP)
                for r in flat
            )
            hs = []
            for r in flat:
                if passives and set(r) == set(passives):
                    hs.append(_PASS_H)
                else:
                    hs.append(max((_gh((m,)) for m in r), default=1.0))
            H = sum(hs) + _ROW_CH * (len(hs) - 1)
            if W <= 0 or H <= 0:
                return None
            return (W / H, n_rows, p_rows)

        # OBJECTIVE (design-derived, no fitted constant): minimize TOTAL ROWS
        # (the maximally-LANDSCAPE packing -- fewest rows -> least height ->
        # widest aspect) SUBJECT TO the estimated PLACEMENT-region aspect being
        # at least the band CENTER. The metric measures the FULL emitted GDS
        # (placement region PLUS the power-rail corridors + routing halo, which
        # add height but little width), so the full-GDS aspect is FLATTER than
        # the placement-region estimate; requiring the placement estimate to be
        # >= the band center leaves headroom for that flattening so the densest
        # in-headroom packing lands the FULL-GDS aspect in the band. Among the
        # configs clearing that floor, fewest rows wins (then closest-to-center,
        # then key) -- deterministic, leakage-free. If NONE clears the floor
        # (a design too narrow to go landscape), fall back to the config whose
        # estimate is closest to the band center (best achievable, honest).
        n_n = max(len(nmos_groups), 1)
        n_p = max(len(pmos_groups), 1)
        center = float(aspect_target)
        in_floor = None
        fallback = None
        for n_per in range(1, n_n + 1):
            for p_per in range(1, n_p + 1):
                est = _cell_estimate(n_per, p_per)
                if est is None:
                    continue
                aspect_est, n_rows, p_rows = est
                nrows = len(n_rows) + len(p_rows)
                fb_score = (abs(math.log(aspect_est / center)), nrows,
                            n_per, p_per)
                if fallback is None or fb_score < fallback[0]:
                    fallback = (fb_score, n_rows, p_rows)
                if aspect_est >= center:
                    # landscape-enough: minimize rows, then closeness to center.
                    fl_score = (nrows, abs(math.log(aspect_est / center)),
                                n_per, p_per)
                    if in_floor is None or fl_score < in_floor[0]:
                        in_floor = (fl_score, n_rows, p_rows)
        best = in_floor if in_floor is not None else fallback
        if best is not None:
            _score, n_rows, p_rows = best
            rows: List[Tuple[str, ...]] = []
            rows.extend(n_rows)
            if passives:
                rows.append(passives)
            rows.extend(p_rows)
            rows = [r for r in rows if r]
            devices = tuple(d for r in rows for d in r)
            return RelativeOrder(cell=cell, devices=devices, rows=tuple(rows))

    # default / forced-count path (byte-identical when sizes is None).
    def _pack(groups: List[Tuple[str, ...]]) -> List[Tuple[str, ...]]:
        if not groups:
            return []
        n = len(groups)
        per = groups_per_row if groups_per_row > 0 else max(
            1, int(math.ceil(math.sqrt(n)))
        )
        return _pack_count(groups, per)

    rows = []
    rows.extend(_pack(nmos_groups))
    if passives:
        rows.append(passives)
    rows.extend(_pack(pmos_groups))
    rows = [r for r in rows if r]
    devices = tuple(d for r in rows for d in r)
    return RelativeOrder(cell=cell, devices=devices, rows=tuple(rows))


def _row_w(members: Tuple[str, ...], *, _gw_member, gap: float) -> float:
    """Row width = sum of member widths + inter-member gaps (helper for the
    P-B width-target aspect scan; member widths are the only design input)."""
    ws = [_gw_member(m) for m in members]
    return sum(ws) + gap * (len(ws) - 1) if ws else 0.0


def island_rail_intents(intent: Any, *, order: Any = None) -> List[dict]:
    """Design-derived power-rail band intents for the ring-aware (well-island)
    placement -- NO reference read, so shift-invariant by construction.

    Each MOS body/bulk net is a power-rail net; its rail rides the band where its
    island sits in the placement row stack (bottom row -> ``side='bottom'``, top
    row -> ``side='top'``, otherwise ``between_rows=k``). This replaces the
    reference rail extractor on the ring path (whose top/bottom side decision
    compared a GDS trunk island against the device centroids and FLIPPED under a
    reference-bbox shift -- the measured 4.46um leakage). Same dict shape the
    sketch consumes; ``trunk_layer_class=2`` (M2, the analog power level);
    ``source='design-island'`` so it never impersonates reference-style.

    ``order`` (optional): the placement :class:`RelativeOrder` whose row stack
    the rails must ride. Default ``None`` uses :func:`well_island_order` (the
    coarse band stack); the F2 compact-group ring path passes
    :func:`compact_matched_group_order` so the rail side/row decision matches the
    row stack the placement actually banded by (otherwise a body net spanning
    several compact-group rows gets a ``between_rows`` index off the wrong stack
    and the rail corridor lands in the wrong band)."""
    order = order if order is not None else well_island_order(intent)
    rows = order.rows
    n = len(rows)
    row_of = {d: i for i, row in enumerate(rows) for d in row}
    body_rows: Dict[str, set] = {}
    for d in (getattr(intent, "devices", None) or []):
        body = str((getattr(d, "nets", None) or {}).get("body", "") or "")
        nm = str(d.name)
        if not body or nm not in row_of:
            continue
        body_rows.setdefault(body, set()).add(row_of[nm])
    out: List[dict] = []
    for net in sorted(body_rows):
        ridx = sorted(body_rows[net])
        if ridx == [0]:
            side, between = "bottom", None
        elif ridx == [n - 1]:
            side, between = "top", None
        else:
            side, between = None, ridx[0]
        out.append({
            "net": net, "side": side, "between_rows": between,
            "trunk_layer_class": 2, "reference_trunk": False,
            "source": "design-island",
            "reason": f"rail rides its well-island band (rows {ridx})",
        })
    return out


__all__ = ["well_islands", "compact_matched_groups", "well_island_order",
           "compact_matched_group_order", "island_rail_intents",
           "declared_guard_islands", "island_of"]
