"""Deterministic envelope-min placement of matched groups (geometric-synthesis core).

Promoted from the validated scratch probe ``_scratch/fix1_cp_anchor.py`` (FIX1): a
discrete CP-SAT placement of the groups that MINIMIZES the bounding-rectangle ENVELOPE
(W*H) -- with a secondary net-wirelength term and hard symmetry -- collapses the
24.6% init-spread frustration of free continuous relaxation to a single deterministic
basin (envelope spread 0.03%) and reaches 7760um2 on the OTA (OPTIMAL on 2/3 tie-break
orders), tighter than the reference 10704 and far under the band-stack 16555.

Design points (longplan v3):
  * Objective = ENVELOPE (W+H, a monotone CP-linearizable proxy for the bbox), NOT
    fill-to-1. Internal slack is a resource (symmetry / routing), never compacted away.
  * Symmetry is a HARD constraint: all declared pairs share ONE mirror axis and sit on
    equal rows -- exact by construction, never a soft penalty that could drift.
  * Determinism: CP-SAT with ``num_search_workers=1`` + fixed ``random_seed`` -> a
    byte-reproducible result. A pure-python shelf-pack fallback runs when ortools is
    absent (keeps the hermetic gate working); it is deterministic + overlap-free but
    does not minimize the envelope or enforce symmetry (CP does).
  * Backend isolation: the CP solver is an optional dependency reached only here, so a
    transitive-dep conflict cannot break the rest of Clay. See
    reference_ortools_pin_protobuf_vlsir (install ortools==9.8.3296).
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import AbstractSet, Dict, FrozenSet, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Group:
    """A placeable block: a logical key and its footprint (um), gap-exclusive."""
    key: str
    w: float
    h: float


@dataclass(frozen=True)
class ArrayTile:
    """A DECLARED shared-diffusion ``@array`` as ONE atomic placeable footprint.

    The array's member devices are placed as a SINGLE tile (footprint ``w_um`` x
    ``h_um`` = the built ``SharedDiffusionArray`` extent), NOT as N free devices:
    each member's centroid is the tile centre plus its ``rel`` offset (um), the
    SAME offsets the emit path seats the one array stamp from (single source of the
    member layout). ``kind`` (``pmos``/``nmos``) selects the group ring pad the
    tile must reserve, exactly like a compact matched group. ``key`` is the
    declared group label. Built by the compact-placer caller from
    :func:`clay.layout.route.route_cell.resolve_shared_diffusion_arrays`; a cell
    that declares no shared-diffusion array passes an empty list -> byte-identical."""
    key: str
    kind: str
    members: Tuple[str, ...]
    w_um: float
    h_um: float
    rel: Dict[str, Tuple[float, float]]
    # Source-native row extent in nm.  Its exact (possibly half-nm) centre
    # carries the mask's manufacturing-grid phase into the atomic solver.
    native_bbox_nm: Optional[Tuple[int, int, int, int]] = None
    native_grid_nm: Optional[int] = None


@dataclass(frozen=True)
class PlacementResult:
    centers: Dict[str, Tuple[float, float]]  # group key -> (cx, cy) in um
    width: float                             # envelope width (um)
    height: float                            # envelope height (um)
    envelope: float                          # width * height (um^2)
    status: str                              # OPTIMAL | FEASIBLE | FALLBACK
    bound: Optional[float] = None            # CP lower bound on the objective, if any
    # When a CP-SAT attempt falls back, preserve WHY the backend could not return
    # a placement.  ``UNKNOWN`` (budget exhausted before a first solution) is an
    # algorithm/search blocker; ``INFEASIBLE`` or ``MODEL_INVALID`` is a domain/model
    # blocker.  An explicit pure-python fallback has no CP backend status.
    backend_status: Optional[str] = None
    # Canonical materializer pair -> (axis, CP atomic-group owner).  Membership is
    # emitted only when one CP group owns both equal-footprint members and its
    # stored relative offsets prove the perpendicular alignment.  The downstream
    # nm quantizer may therefore replay the coordinate without inferring a new
    # matching relationship from final geometry.
    matching_pair_owners: Dict[
        Tuple[str, str], Tuple[str, str]
    ] = field(default_factory=dict)
    # Present only when source-bound matching spans independent native rows.
    # The captured CP model/solution still needs source and row-seating replay.
    native_cross_row_cp_proof: object = None
    native_row_cp_capture: object = None
    native_cross_row_cp_capture: object = None


class NativeRowAtomicSolveError(ValueError):
    """A native row cannot be published without a completed CP frame proof."""
    def __init__(self, backend_status):
        self.backend_status = str(backend_status or 'UNKNOWN')
        super().__init__('native row atomic placement unresolved: ' + self.backend_status)


def _array_tile_group(tile, pad_by_kind, via_pad, channel_groups, channel_pad, island_members):
    """Compile an array's physical group and its guard-island reservation."""
    key = f'A_{tile.key}'
    pad = float(pad_by_kind.get(tile.kind, 0.0)) + float(via_pad)
    if key in channel_groups:
        pad += float(channel_pad)
    in_island = bool({str(member) for member in tile.members} & island_members)
    footprint_pad = 0.0 if in_island else pad
    return (Group(key=key, w=float(tile.w_um)+2.0*footprint_pad,
                  h=float(tile.h_um)+2.0*footprint_pad), pad if in_island else None)


def _channel_incident_groups(intent, cg, array_owner, nets, channel_pad, passive_only):
    """Compile the same realized-open group projection for solve and replay."""
    relaxed = {str(net).lower() for net in (nets or ())}
    result = set()
    if relaxed and float(channel_pad or 0.0) > 0.0:
        owners = {}
        for kind, _body, members in cg:
            key = f'G_{kind}_{members[0]}'
            for member in members: owners[str(member)] = key
        # Declared array MOS members were removed from compact groups above;
        # their realized-open nets must resolve to the one array owner too.
        for member, owner in array_owner.items():
            owners[str(member)] = str(owner)
        passives = list(getattr(intent, 'passives', None) or [])
        for item in passives:
            name = str(getattr(item, 'name', ''))
            owners[name] = array_owner.get(name, f'P_{name}')
        passive_names = {str(getattr(item, 'name', '')) for item in passives}
        for item in list(getattr(intent, 'devices', None) or []) + passives:
            name = str(getattr(item, 'name', ''))
            if passive_only and name not in passive_names: continue
            if any(str(net).lower() in relaxed for net in (getattr(item, 'nets', None) or {}).values()):
                owner = owners.get(name)
                if owner: result.add(owner)
    return result


# objective weight: envelope (W+H) dominates; wirelength is a tie-break-scale secondary.
_ENVELOPE_WEIGHT = 1000
# axis-centering weight: the symmetry-axis intent (one shared, centred diff axis)
# is a QUALITY obligation that the placer must HONOR, not a tie-break. Aligning
# the per-group reflection axes onto one centred axis is geometrically NOT free
# under a 2D envelope-min pack (it costs vertical stacking), so this weight is set
# so the solver WILL spend bounded envelope to converge the axes -- the honest
# tradeoff the round mandates (centering the diff axis may change the envelope).
# It stays below _ENVELOPE_WEIGHT per unit so a gratuitous area blow-up that does
# NOT improve axis agreement is never taken; it dominates the wirelength tie-break.
_CENTER_WEIGHT = 200
# aspect-band weight: the floorplan aspect intent (rotate the 2D pack toward the
# target W/H band) is a QUALITY obligation. The deviation term carries the rational
# denominator d=100, so this weight times d=100 stays BELOW _ENVELOPE_WEIGHT=1000
# per grid unit -- the solver rotates the pack to the band but never blows up the
# envelope area just to nudge the aspect. Measured: weight 5 sat just under the
# landscape basin (the 2D pack has a portrait + a landscape local optimum); 10
# reliably commits the band-center target to the landscape basin at ~10% area cost
# (env 9047 -> 9922 on the OTA), landing the emitted-GDS aspect in the band.
_ASPECT_WEIGHT = 10
# wirelength-locality global multiplier (the floorplan obligation). The per-net
# clique weight (1/(fanout-1), baked *100 into the span vars) carries the
# relative importance; this scalar sets how much bounded envelope the solver
# spends to shorten the dominant signal nets. Tuned on the cp OTA effect test
# (gf180/external_pdk): pulls the Miller cap to abut its output stage so no low-fanout
# signal net spans most of the cell, at a bounded envelope cost. Env
# CLAY_CP_WL_WEIGHT overrides for sweeps. Engages ONLY on the weighted cp path
# (cp_group_placement_pos); legacy place_envelope_min callers keep wl_weight=1.
_CP_WL_WEIGHT = 2
# wirelength L-infinity (minimax) weight. The GATE measures the MAX single
# low-fanout signal net span (HPWL/(W+H)); minimising the SUM of spans (above)
# lets the solver trade one sprawled net for many short ones and still fail the
# max gate. This term penalises the MAXIMUM signal-net span directly -- the same
# quantity the gate bounds -- so the objective and the obligation are aligned.
# Applies only to low-group-fanout (signal) nets; power rails are excluded (they
# distribute). Sits below the envelope per unit (no unbounded area for locality)
# but high enough to pull the worst net under the band. Env CLAY_CP_MAXSPAN_WEIGHT
# overrides for sweeps. Weighted cp path only. Calibrated on the 3-PDK effect
# test (external_pdk/gf180/external_pdk): at 3000 the minimax dominates the per-PDK landscape
# enough to localize every low-fanout signal net under the gate band on all three
# (worst signal span 0.48/0.47/0.54), where lower weights were non-monotonic
# (the solver traded one sprawled net for many). The placer solve is ~6-20s (the
# router, not the placer, is the route-time cost), so the strong term carries no
# meaningful production-speed penalty.
_CP_MAXSPAN_WEIGHT = 3000
# a net touching more than this many GROUPS is a power/distribution net, excluded
# from the L-infinity signal-locality term (it is meant to span). Group-level
# analogue of FidelityThresholds.signal_fanout_max (which gates on device fanout).
_CP_SIGNAL_GROUP_FANOUT_MAX = 6
# @abut tight-adjacency weight (the declared cross-group @abut intent). A soft L1 pull
# on the two groups' centre distance (AddNoOverlap2D forbids overlap, so minimising the
# centre distance packs them EDGE-ADJACENT). Sits BELOW _ENVELOPE_WEIGHT per grid so an
# adjacency is never bought at a gratuitous area cost, but above the wirelength tie-break
# so a declared @abut pair packs tight. Engaged ONLY when abut_pairs is non-empty ->
# byte-identical for every caller that declares no @abut.
_ABUT_WEIGHT = 100
# @rail_affinity / @anchor EDGE-AFFINITY weight (Wave B4). A soft L1 pull on a group's
# distance to the declared cell EDGE (top / bottom): side='bottom' pulls ys[k] toward 0,
# side='top' pulls the group's top edge toward the envelope top. Sits BELOW
# _ENVELOPE_WEIGHT per grid (an edge pull never buys a gratuitous area blow-up) but above
# the wirelength tie-break so a declared rail-affinity group is drawn to its supply rail.
# Engaged ONLY when rail_affinities is non-empty -> byte-identical for every caller that
# declares no @rail_affinity / @anchor.
_RAIL_AFFINITY_WEIGHT = 150


def place_envelope_min(
    groups: Sequence[Group],
    *,
    nets: Sequence[Sequence[str]] = (),
    net_weights: Sequence[float] = (),
    net_signal: Sequence[bool] = (),
    symmetry_pairs: Sequence[Tuple[str, str]] = (),
    axis_groups: Sequence[Tuple[str, float]] = (),
    axis_mode: str = "shared_hard",
    gap: float = 0.5,
    wl_weight: int = 1,
    aspect_target: Optional[float] = None,
    max_time_s: float = 20.0,
    backend: str = "auto",
    disjoint_clusters: Sequence[Sequence[str]] = (),
    island_pads: Sequence[float] = (),
    band_of_group: Optional[Dict[str, int]] = None,
    order_constraints: Sequence[Tuple[str, str, str]] = (),
    abut_pairs: Sequence[Tuple[str, str]] = (),
    rail_affinities: Sequence[Tuple[str, str]] = (),
    native_cross_row_inventory: object = None,
    native_cross_row_groups: Optional[Dict[str, str]] = None,
    native_cross_row_directions: Optional[Dict[Tuple[str, str], str]] = None,
    native_row_frames: Optional[Dict[str, Tuple[Tuple[int, int, int, int], int]]] = None,
    native_group_phases: Optional[Dict[str, Tuple[int, int, int]]] = None,
) -> PlacementResult:
    """Place ``groups`` to minimize the bounding-rectangle envelope.

    ``nets``: each entry is the list of group keys a net connects (>=2 used).
    ``symmetry_pairs``: matched (a, b) pairs of GROUPS forced mirror-symmetric
        about one shared axis on equal rows (cross-group reflection).
    ``band_of_group``: ``{group_key -> band_ordinal}`` from the declared
        ``@place_group`` floorplan ROWS. A group in a LOWER band is forced to sit
        ENTIRELY BELOW (smaller y) every group in a HIGHER band -- so the pack
        compacts WITHIN each declared tier (free x-packing + within-band y-stacking)
        but NEVER interleaves devices across the declared bands (the tier-preserving
        obligation the area-driven envelope-min pack dropped). Pure topology (the
        declared row ordinals -- no reference coordinate). ``None``/empty adds NO
        constraint -> byte-identical for every caller that declares no floorplan.
    ``disjoint_clusters``: each entry is a set of group keys that must be KEPT
        TOGETHER as one contiguous region whose bounding hull is disjoint from
        every other cluster hull AND from every non-clustered group. This is the
        declared-@guard_ring obligation: a ring frame bounds its member groups, so
        the members must stay contiguous (else the frame sprawls) and two different-
        net ring frames must not overlap (else the rails short). Empty (the default)
        adds NO constraint -> byte-identical for every existing caller.
    ``axis_groups``: ``(group_key, axis_offset_um)`` for every group that hosts a
        declared 2-member reflection pair laid out about its own centre. The
        pair's reflection axis is ``group_centre_x + axis_offset``; this binds
        those axes to a shared centred axis var per ``axis_mode``. Pure
        topology-derived (offset from the within-group abut layout, no
        coordinate). Empty = byte-identical.
    ``axis_mode``: how ``axis_groups`` bind to the shared diff axis.
        ``"shared_hard"`` (the bring-up diagnostic) binds EVERY axis group to ONE
        shared axis var with a HARD +/-1-grid equality and centres it -- exact
        symmetry, but two non-overlapping groups forced onto one x-axis can only
        separate in Y, so the cell stacks into a tall vertical column the
        over-cell router cannot connect (LVS-unmatched, the measured blocker).
        ``"centered_soft"`` (the hybrid co-design) instead pulls each axis group
        toward the shared centred axis with an L1 SOFT term: the solver centres
        the diff axis where the 2D envelope-min pack allows, but never pays
        unbounded vertical stacking to force a hard shared axis -- so net-adjacent
        groups stay in HORIZONTALLY-adjacent rows (the adjacency the channel
        router needs) while the cell's primary axes converge toward centre.
    ``order_constraints``: ``(a, b, rel)`` GROUP triples (``rel`` in {left_of, above})
        from the declared ``@order`` intent -- a HARD relative order (``left_of``: a's
        right edge <= b's left edge; ``above``: b's top edge <= a's bottom edge, higher
        y = above per ``band_of_group``). Empty -> no constraint -> byte-identical.
    ``abut_pairs``: ``(a, b)`` GROUP pairs from the declared cross-group ``@abut`` intent
        -- a tight-adjacency SOFT pull (minimise the two groups' centre distance; the
        NoOverlap keeps them from overlapping, so the pull packs them edge-adjacent).
        Empty -> no term -> byte-identical.
    ``backend``: ``"auto"`` (CP-SAT if available else fallback), ``"cpsat"``, or
        ``"fallback"``.
    """
    if not groups:
        if native_cross_row_inventory is not None or native_row_frames or native_group_phases:
            raise ValueError("native row placement requires actual CP groups")
        return PlacementResult({}, 0.0, 0.0, 0.0, "OPTIMAL", 0.0)

    import os as _os
    _env_backend = (_os.environ.get("CLAY_ENVELOPE_BACKEND") or "").strip().lower()
    effective_backend = _env_backend if (backend == "auto" and _env_backend in ("cpp", "cpsat", "fallback")) else backend

    if effective_backend == "cpp":
        if native_cross_row_inventory is not None or native_row_frames or native_group_phases:
            raise ValueError("native row placement requires the selected CP solver")
        return _place_cpp(
            groups, nets, symmetry_pairs, axis_groups, gap, wl_weight, max_time_s,
            axis_mode=axis_mode, aspect_target=aspect_target,
            net_weights=net_weights, net_signal=net_signal,
            disjoint_clusters=disjoint_clusters,
            island_pads=island_pads,
            band_of_group=band_of_group,
            order_constraints=order_constraints,
            abut_pairs=abut_pairs,
            rail_affinities=rail_affinities,
        )

    # Pure C++20 Main Engine fast path for default `backend="auto"` whenever
    # CP-SAT protobuf capture/replay is not required and `_place_cpsat` has not
    # been monkeypatched by a diagnostic test.
    if (
        effective_backend == "auto"
        and native_cross_row_inventory is None
        and not native_row_frames
        and not native_group_phases
        and _place_cpsat is _ORIGINAL_PLACE_CPSAT
    ):
        cpp_result = _place_cpp(
            groups, nets, symmetry_pairs, axis_groups, gap, wl_weight, max_time_s,
            axis_mode=axis_mode, aspect_target=aspect_target,
            net_weights=net_weights, net_signal=net_signal,
            disjoint_clusters=disjoint_clusters,
            island_pads=island_pads,
            band_of_group=band_of_group,
            order_constraints=order_constraints,
            abut_pairs=abut_pairs,
            rail_affinities=rail_affinities,
        )
        if cpp_result.status in ("OPTIMAL", "FEASIBLE"):
            return cpp_result

    use_cp = effective_backend in ("auto", "cpsat")
    if use_cp:
        try:
            from ortools.sat.python import cp_model  # noqa: F401
        except Exception:
            if effective_backend == "cpsat":
                raise
            use_cp = False

    if use_cp:
        result = _place_cpsat(
            groups, nets, symmetry_pairs, axis_groups, gap, wl_weight, max_time_s,
            axis_mode=axis_mode, aspect_target=aspect_target,
            net_weights=net_weights, net_signal=net_signal,
            disjoint_clusters=disjoint_clusters,
            island_pads=island_pads,
            band_of_group=band_of_group,
            order_constraints=order_constraints,
            abut_pairs=abut_pairs,
            rail_affinities=rail_affinities,
            native_cross_row_inventory=native_cross_row_inventory,
            native_cross_row_groups=native_cross_row_groups,
            native_cross_row_directions=native_cross_row_directions,
            native_row_frames=native_row_frames,
            native_group_phases=native_group_phases,
        )
        if (native_row_frames or native_group_phases) and result.status not in ('OPTIMAL','FEASIBLE'):
            raise NativeRowAtomicSolveError(result.backend_status)
        return result
    if native_cross_row_inventory is not None or native_row_frames or native_group_phases:
        raise ValueError("native row placement requires the selected CP solver")
    return _place_fallback(groups, gap)


def _envelope_from_centers(
    centers: Dict[str, Tuple[float, float]], size: Dict[str, Tuple[float, float]]
) -> Tuple[float, float, float]:
    xs_lo = [centers[k][0] - size[k][0] / 2 for k in centers]
    xs_hi = [centers[k][0] + size[k][0] / 2 for k in centers]
    ys_lo = [centers[k][1] - size[k][1] / 2 for k in centers]
    ys_hi = [centers[k][1] + size[k][1] / 2 for k in centers]
    w = max(xs_hi) - min(xs_lo)
    h = max(ys_hi) - min(ys_lo)
    return w, h, w * h


def _mirror_component_owner_pairs(
    mirror_pairs: Sequence[Tuple[str, str]],
    owner_by_member: Dict[str, str],
) -> Tuple[Tuple[str, str], ...]:
    """Lower only two-owner mirror components into hard reflection pairs.

    Netlist extraction represents a multi-leg current mirror as the clique of all
    same-gate device pairs.  Those edges describe one mirror *component*; they are
    not independent geometric reflection equalities.  Expanding a three-owner
    clique into ``(A,B), (A,C), (B,C)`` forces all three centres equal and makes
    ``NoOverlap2D`` algebraically infeasible. Resolve every edge to its atomic
    owners before grouping: a shared row can join otherwise disjoint source
    components. A projected component has a unique pairwise reflection lowering
    only when it has exactly two owners. Larger owner sets remain available
    to matching/common-centroid quality checks but cannot be fabricated into a
    pairwise placement certificate.
    """
    parent: Dict[str, str] = {}

    def find(member: str) -> str:
        root = member
        while parent[root] != root:
            root = parent[root]
        while parent[member] != member:
            next_member = parent[member]
            parent[member] = root
            member = next_member
        return root

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            parent[right_root] = left_root
        else:
            parent[left_root] = right_root

    for raw_pair in mirror_pairs:
        if type(raw_pair) is not tuple or len(raw_pair) != 2:
            continue
        left, right = str(raw_pair[0]), str(raw_pair[1])
        if left == right or left not in owner_by_member or right not in owner_by_member:
            continue
        left, right = owner_by_member[left], owner_by_member[right]
        if left == right:
            continue
        parent.setdefault(left, left)
        parent.setdefault(right, right)
        union(left, right)

    members_by_root: Dict[str, set[str]] = {}
    for member in parent:
        members_by_root.setdefault(find(member), set()).add(member)
    result = set()
    for members in members_by_root.values():
        owners = tuple(sorted(members))
        if len(owners) == 2:
            result.add(owners)
    return tuple(sorted(result))


def _place_cpsat(
    groups: Sequence[Group],
    nets: Sequence[Sequence[str]],
    symmetry_pairs: Sequence[Tuple[str, str]],
    axis_groups: Sequence[Tuple[str, float]],
    gap: float,
    wl_weight: int,
    max_time_s: float,
    axis_mode: str = "shared_hard",
    aspect_target: Optional[float] = None,
    net_weights: Sequence[float] = (),
    net_signal: Sequence[bool] = (),
    disjoint_clusters: Sequence[Sequence[str]] = (),
    island_pads: Sequence[float] = (),
    band_of_group: Optional[Dict[str, int]] = None,
    order_constraints: Sequence[Tuple[str, str, str]] = (),
    abut_pairs: Sequence[Tuple[str, str]] = (),
    rail_affinities: Sequence[Tuple[str, str]] = (),
    native_cross_row_inventory: object = None,
    native_cross_row_groups: Optional[Dict[str, str]] = None,
    native_cross_row_directions: Optional[Dict[Tuple[str, str], str]] = None,
    native_row_frames: Optional[Dict[str, Tuple[Tuple[int, int, int, int], int]]] = None,
    native_group_phases: Optional[Dict[str, Tuple[int, int, int]]] = None,
) -> PlacementResult:
    from ortools.sat.python import cp_model

    keys = [g.key for g in groups]
    size = {g.key: (g.w, g.h) for g in groups}
    # integer footprint with the gap baked in on each box, so AddNoOverlap2D enforces
    # the >= gap separation directly (1 grid unit = 1 um).
    fw = {g.key: max(1, int(math.ceil(g.w + gap))) for g in groups}
    fh = {g.key: max(1, int(math.ceil(g.h + gap))) for g in groups}
    present_keys = set(keys)
    # Coarse intervals must actually admit at least one source-native phase.
    # When a rounded-up footprint leaves too little slack, the next integer
    # interval is the smallest conservative envelope that can admit that phase.
    if native_row_frames:
        from decimal import Decimal, ROUND_CEILING
        def ceil_nm(value):
            return int((Decimal(str(value)) * 1000).to_integral_value(rounding=ROUND_CEILING))
        gap_nm = max(0, ceil_nm(gap))
        for key, (bbox, grid) in native_row_frames.items():
            if key not in present_keys or type(grid) is not int or grid <= 0:
                raise ValueError('native row frame names a foreign owner or grid')
            if type(bbox) is not tuple or len(bbox) != 4 or any(type(v) is not int for v in bbox):
                raise ValueError('native row frame extent is malformed')
            for axis, widths in enumerate((fw, fh)):
                required = ceil_nm(size[key][axis]) + gap_nm
                phase = (bbox[axis] + bbox[axis + 2]) % math.gcd(2000, 2 * grid)
                modulus = math.gcd(2000, 2 * grid)
                low = required
                first = low + ((phase - low) % modulus)
                while first > 2000 * widths[key] - required:
                    widths[key] += 1
    if native_group_phases:
        from decimal import Decimal, ROUND_CEILING
        if set(native_group_phases) & set(native_row_frames or ()):
            raise ValueError('native standalone group overlaps a stored row frame')
        for key, phase in native_group_phases.items():
            if (key not in present_keys or type(phase) is not tuple or len(phase) != 3
                    or any(type(value) is not int for value in phase) or phase[2] <= 0
                    or any(value < 0 or value >= 2 * phase[2] for value in phase[:2])):
                raise ValueError('native standalone group phase is malformed')
            for axis, widths in enumerate((fw, fh)):
                required = int((Decimal(str(size[key][axis])) * 1000).to_integral_value(rounding=ROUND_CEILING))
                required += int((Decimal(str(max(0.0, gap))) * 1000).to_integral_value(rounding=ROUND_CEILING))
                modulus = math.gcd(2000, 2 * phase[2])
                first = required + ((phase[axis] - required) % modulus)
                while first > 2000 * widths[key] - required:
                    widths[key] += 1
    clusters = [[k for k in cl if k in present_keys] for cl in (disjoint_clusters or ())]
    _pads = list(island_pads or ())
    # Keep clusters and their pads aligned when an empty cluster is dropped.  The
    # island hull constraints below reserve ``ipad`` on BOTH sides of every hull,
    # so those pads must also exist in the coordinate domain.  Omitting them makes
    # even one otherwise-valid singleton island algebraically infeasible
    # (``xs >= ipad`` and ``xs + fw + ipad <= HX`` with ``HX == fw + 1``).
    _cl_pad = [(cl, (_pads[i] if i < len(_pads) else 0.0))
               for i, cl in enumerate(clusters) if cl]
    clusters = [cl for cl, _ in _cl_pad]
    island_domain_budget = 2 * sum(
        max(0, int(math.ceil(float(pad_um)))) for _cl, pad_um in _cl_pad
    )
    HX = sum(fw.values()) + len(groups) + island_domain_budget
    HY = sum(fh.values()) + len(groups) + island_domain_budget

    m = cp_model.CpModel()
    xs: Dict[str, object] = {}
    ys: Dict[str, object] = {}
    xiv: Dict[str, object] = {}
    yiv: Dict[str, object] = {}
    for k in keys:
        xs[k] = m.NewIntVar(0, HX - fw[k], f"x_{k}")
        ys[k] = m.NewIntVar(0, HY - fh[k], f"y_{k}")
        xiv[k] = m.NewIntervalVar(xs[k], fw[k], m.NewIntVar(0, HX, f"xe_{k}"), f"ix_{k}")
        yiv[k] = m.NewIntervalVar(ys[k], fh[k], m.NewIntVar(0, HY, f"ye_{k}"), f"iy_{k}")
    m.AddNoOverlap2D([xiv[k] for k in keys], [yiv[k] for k in keys])

    native_centres2 = {}
    native_frame_binding = None
    if native_row_frames:
        from decimal import Decimal, ROUND_CEILING
        from clay.layout.place.native_row_cp_frames import add_native_row_frames_cp
        if not set(native_row_frames) <= present_keys:
            raise ValueError('native row frame names a foreign atomic owner')
        def extent_nm(value):
            value = Decimal(str(value)) * 1000
            if not value.is_finite() or value <= 0:
                raise ValueError('native atomic group footprint is invalid')
            return int(value.to_integral_value(rounding=ROUND_CEILING))
        coarse = {key: (key, xs[key], ys[key], fw[key], fh[key],
                        extent_nm(size[key][0]), extent_nm(size[key][1]))
                  for key in native_row_frames}
        native_centres2, native_frame_binding = add_native_row_frames_cp(
            m, native_row_frames, coarse, gap_nm=extent_nm(gap) if gap > 0 else 0)

    standalone_centres2 = {}
    if native_group_phases:
        from decimal import Decimal, ROUND_CEILING
        for key, (phase_x, phase_y, grid) in native_group_phases.items():
            coords = []
            for axis, (phase, coarse_pos, coarse_size, domain) in enumerate((
                    (phase_x, xs[key], fw[key], HX), (phase_y, ys[key], fh[key], HY))):
                required = int((Decimal(str(size[key][axis])) * 1000).to_integral_value(rounding=ROUND_CEILING))
                required += int((Decimal(str(max(0.0, gap))) * 1000).to_integral_value(rounding=ROUND_CEILING))
                center = m.NewIntVar(0, 2000 * domain, f'native_group_center2_{axis}_{key}')
                lattice = m.NewIntVar(0, 2000 * domain, f'native_group_phase_{axis}_{key}')
                m.Add(center == phase + 2 * grid * lattice)
                m.Add(center >= 2000 * coarse_pos + required)
                m.Add(center <= 2000 * (coarse_pos + coarse_size) - required)
                coords.append(center)
            standalone_centres2[key] = tuple(coords)
    exact_centres2 = {**native_centres2, **standalone_centres2}

    def actual_centre2_nm(key, axis):
        if key in exact_centres2:
            return exact_centres2[key][axis]
        return 1000 * ((2 * xs[key] + fw[key]) if axis == 0 else (2 * ys[key] + fh[key]))

    def member_origin2_nm(key, axis):
        # Row member offsets use the lower integer local bbox centre.  The
        # physical extent centre can be half a nanometre above that origin.
        if key in native_centres2:
            box = native_row_frames[key][0]
            return actual_centre2_nm(key, axis) - ((box[axis] + box[axis + 2]) % 2)
        return actual_centre2_nm(key, axis)

    def exact_axis_offset2_nm(offset):
        from decimal import Decimal
        value = Decimal(str(offset)) * 2000
        if not value.is_finite() or value != value.to_integral_value():
            raise ValueError('native row matching axis offset is not exact')
        return int(value)

    cross_row_binding = None
    if native_cross_row_inventory is not None:
        from clay.layout.place.cross_row_matching import (add_native_cross_row_cp_constraints,
            add_native_cross_row_cp_constraints_v2)

        if (type(native_cross_row_groups) is not dict
                or any(type(row) is not str or type(owner) is not str or owner not in xs
                       for row, owner in native_cross_row_groups.items())
                or len(set(native_cross_row_groups.values())) != len(native_cross_row_groups)):
            raise ValueError("native cross-row solver owner mapping is malformed")
        if native_frame_binding is not None:
            if not set(native_cross_row_groups.values()) <= set(native_centres2):
                raise ValueError('native cross-row owner lacks its source frame')
            cross_row_binding = add_native_cross_row_cp_constraints_v2(
                m, native_cross_row_inventory,
                {row: (owner, *native_centres2[owner]) for row, owner in native_cross_row_groups.items()},
                native_cross_row_directions, frame_binding=native_frame_binding)
        else:
            cross_row_binding = add_native_cross_row_cp_constraints(
                m, native_cross_row_inventory,
                {row: (owner, xs[owner], ys[owner], fw[owner], fh[owner])
                 for row, owner in native_cross_row_groups.items()}, native_cross_row_directions)

    # KEEP-TOGETHER / RING-FRAME DISJOINTNESS (the declared @guard_ring obligation).
    # A guard-ring island is materialized as ONE frame bounding its member groups. If
    # the pack SCATTERS an island's groups, that frame sprawls cell-wide and two
    # different-tap-net frames (a gnd substrate ring vs a vdd nwell ring) OVERLAP -> a
    # rail-to-rail short that also swallows the bias nodes enclosed between them (the
    # measured ota_dev compact GND,NET048,NET07,NET4,VDD merge). Forbid it BY
    # CONSTRUCTION: build each island's bounding HULL over its member group boxes
    # (fw/fh already carry the per-group ring pad, so a disjoint hull holds the ring-
    # to-ring clearance) and AddNoOverlap2D over { island hulls } U { every group NOT
    # in an island, as its own box } -- so no island frame overlaps another AND no
    # foreign group nestles inside a ring frame (its inter-member gaps included).
    # Engaged ONLY when disjoint_clusters is supplied (a cell that declares guard
    # rings opts in); empty -> zero new vars, byte-identical for every existing caller.
    island_span_terms: List[object] = []  # sum(iw+ih) over islands -> tight-pack term
    if clusters:
        clustered = {k for cl in clusters for k in cl}
        cx_ivs: List[object] = []
        cy_ivs: List[object] = []
        for ci, (cl, pad_um) in enumerate(_cl_pad):
            # RING PAD AT THE ISLAND BOUNDARY (not between members). The member group
            # footprints inside an island are TIGHT (the ring/via pad is dropped by the
            # caller for island members, so they abut like the reference's 0.5um), and
            # the pad is reserved HERE, once, around the whole island hull -- so the ONE
            # ring frame clears its neighbours while the members inside pack tight.
            ipad = max(0, int(math.ceil(float(pad_um))))
            ixlo = m.NewIntVar(0, HX, f"cl{ci}_xlo")
            ixhi = m.NewIntVar(0, HX, f"cl{ci}_xhi")
            iylo = m.NewIntVar(0, HY, f"cl{ci}_ylo")
            iyhi = m.NewIntVar(0, HY, f"cl{ci}_yhi")
            for k in cl:
                m.Add(ixlo + ipad <= xs[k])
                m.Add(ixhi >= xs[k] + fw[k] + ipad)
                m.Add(iylo + ipad <= ys[k])
                m.Add(iyhi >= ys[k] + fh[k] + ipad)
            iw = m.NewIntVar(0, HX, f"cl{ci}_w")
            ih = m.NewIntVar(0, HY, f"cl{ci}_h")
            m.Add(iw == ixhi - ixlo)
            m.Add(ih == iyhi - iylo)
            # INTRA-ISLAND TIGHTNESS: minimise this island's hull spread so its members
            # abut instead of leaving a gratuitous empty band inside the ring frame.
            # (iw/ih include the constant 2*ipad boundary pad -> minimising them
            # minimises the MEMBER span, the argmin is unchanged by the constant.)
            if len(cl) >= 2:
                island_span_terms += [iw, ih]
            cx_ivs.append(m.NewIntervalVar(ixlo, iw, ixhi, f"cl{ci}_ix"))
            cy_ivs.append(m.NewIntervalVar(iylo, ih, iyhi, f"cl{ci}_iy"))
        for k in keys:
            if k not in clustered:
                cx_ivs.append(xiv[k])
                cy_ivs.append(yiv[k])
        if len(cx_ivs) >= 2:
            m.AddNoOverlap2D(cx_ivs, cy_ivs)

    # DECLARED @place_group TIER/BAND ORDERING (the floorplan-fidelity obligation the
    # area-driven envelope-min pack dropped): each banded group carries its declared
    # @place_group ROW ordinal. A group in a LOWER band must sit ENTIRELY BELOW every
    # group in a HIGHER band -- its TOP edge (ys + fh) at or under the higher group's
    # BOTTOM edge (ys). This preserves the declared band Y-ORDER (row=0 tier below the
    # row=2 tier) and forbids CROSS-BAND device interleave, while leaving groups WITHIN
    # one band free to pack in x AND stack in y (compact-within-band). Pairwise over
    # the banded groups only (a group with no declared band is untouched -> it floats).
    # Pure topology (the declared row ordinals -- no reference coordinate). Empty/None
    # -> zero constraint -> byte-identical for every caller that declares no floorplan.
    bg = {k: int(v) for k, v in (band_of_group or {}).items() if k in present_keys}
    if bg:
        banded = [k for k in keys if k in bg]
        for i in range(len(banded)):
            for j in range(i + 1, len(banded)):
                ka, kb = banded[i], banded[j]
                if bg[ka] == bg[kb]:
                    continue  # same tier -> free relative packing
                lo, hi = (ka, kb) if bg[ka] < bg[kb] else (kb, ka)
                m.Add(ys[lo] + fh[lo] <= ys[hi])

        # FLOATING elements form a PASSIVE BANK BESIDE the tier stack (NOT interleaved
        # among the transistors, NOT stacked on top). The banded MOS tiers stack in y
        # (nmos band below pmos band) into a SHORT + WIDE core. A FLOATING tile (a MIM
        # cap / resistor -- excluded from the tier stack) is then free in x AND y, and
        # the envelope-min search either PARKS a big cap ABOVE the top band (portrait,
        # measured aspect 0.9) OR crams it INTO a MOS band, where the cap's plate metal
        # merges with the band's rails -- the measured LVS short (a Miller cap dropped
        # at the nmos-band y merged GND,NET25,VOUT). Both are fixed by banking every
        # floating tile to ONE side: pin its LEFT edge at/after the MOS core's RIGHT
        # edge (a clean passive column to the right of the transistor core) AND its
        # y-CENTRE within the MOS band y-envelope. This is the declared col=3 "the caps
        # bank BESIDE the MOS core beyond the MOS grid" intent + the reference's
        # MOS-core / passive-bank separation -- so the router sees a clean transistor
        # region and a clean passive region, never a cap crammed among the rails.
        # Only engages when the cell BANDS transistors AND floats tiles (the tiered
        # OTA) -> no effect on an all-banded or un-tiered cell.
        floating = [k for k in keys if k not in bg]
        if floating and banded:
            mos_lo = m.NewIntVar(0, HY, "mos_band_lo")
            mos_hi = m.NewIntVar(0, HY, "mos_band_hi")
            mos_right = m.NewIntVar(0, HX, "mos_band_right")
            m.AddMinEquality(mos_lo, [ys[k] for k in banded])
            m.AddMaxEquality(mos_hi, [ys[k] + fh[k] for k in banded])
            m.AddMaxEquality(mos_right, [xs[k] + fw[k] for k in banded])
            for k in floating:
                m.Add(xs[k] >= mos_right)          # clean passive column, right of core
                cy2 = 2 * ys[k] + fh[k]            # 2x the tile y-centre (avoids the /2)
                m.Add(cy2 >= 2 * mos_lo)
                m.Add(cy2 <= 2 * mos_hi)

    # DECLARED @order (HARD relative order) + @abut (tight adjacency) GROUP constraints
    # -- the placement consumption of the .clay @order/@abut intent, mapped to GROUP
    # keys by the caller. @order(a,b,left_of): a ENTIRELY left of b (a's right edge <= b's
    # left edge). @order(a,b,above): a ABOVE b (b's top edge <= a's bottom edge; higher
    # y = above, the same convention band_of_group uses). @abut(a,b): pull the two groups
    # EDGE-ADJACENT via a SOFT L1 term on their centre distance -- AddNoOverlap2D already
    # forbids overlap, so minimising the centre distance packs them touching. Empty (the
    # default) -> no constraint / no aux var / no objective term -> byte-identical for
    # every caller that declares no @order/@abut.
    for (oa, ob, rel_kind) in (order_constraints or ()):
        if oa not in present_keys or ob not in present_keys or oa == ob:
            continue
        if rel_kind == "left_of":
            m.Add(xs[oa] + fw[oa] <= xs[ob])
        elif rel_kind == "above":
            m.Add(ys[ob] + fh[ob] <= ys[oa])
    abut_terms: List[object] = []
    for (aa, ab) in (abut_pairs or ()):
        if aa not in present_keys or ab not in present_keys or aa == ab:
            continue
        # centre-to-centre L1 in 2x-integer form (avoids the /2): 2*cx = 2*xs + fw.
        adx = m.NewIntVar(0, 2 * HX, f"abut_dx_{aa}_{ab}")
        ady = m.NewIntVar(0, 2 * HY, f"abut_dy_{aa}_{ab}")
        cxa, cxb = 2 * xs[aa] + fw[aa], 2 * xs[ab] + fw[ab]
        cya, cyb = 2 * ys[aa] + fh[aa], 2 * ys[ab] + fh[ab]
        m.Add(adx >= cxa - cxb)
        m.Add(adx >= cxb - cxa)
        m.Add(ady >= cya - cyb)
        m.Add(ady >= cyb - cya)
        abut_terms += [adx, ady]

    # bounding-box envelope
    Wv = m.NewIntVar(0, HX, "W")
    Hv = m.NewIntVar(0, HY, "H")
    for k in keys:
        m.Add(Wv >= xs[k] + fw[k])
        m.Add(Hv >= ys[k] + fh[k])

    # hard symmetry: every pair on an equal row, all pairs sharing ONE mirror axis.
    # axis equality in 2x integer form: 2*xa+2*xb+fwa+fwb is constant across pairs
    # (= 2*(cx_a+cx_b)); same-row via ys[a]==ys[b].
    present = set(keys)
    norm_pairs = [(a, b) for (a, b) in symmetry_pairs if a in present and b in present]
    axis_sum_ref = None
    for (a, b) in norm_pairs:
        m.Add(ys[a] == ys[b])
        if a in exact_centres2 or b in exact_centres2:
            m.Add(actual_centre2_nm(a, 1) == actual_centre2_nm(b, 1))
        axis_expr = 2 * xs[a] + 2 * xs[b] + (fw[a] + fw[b])
        if axis_sum_ref is None:
            axis_sum_ref = axis_expr
        else:
            m.Add(axis_expr == axis_sum_ref)
    if exact_centres2 and len(norm_pairs) > 1:
        reference = actual_centre2_nm(norm_pairs[0][0], 0) + actual_centre2_nm(norm_pairs[0][1], 0)
        for a, b in norm_pairs[1:]:
            m.Add(actual_centre2_nm(a, 0) + actual_centre2_nm(b, 0) == reference)

    # SHARED-AXIS INTENT (symmetry-axis obligation): every group that hosts a
    # declared 2-member reflection pair must place that pair about ONE shared
    # vertical axis, and that axis must be CENTRED in the envelope. The pair's
    # axis x = group_centre_x + axis_offset = (xs[k] + fw[k]/2) + off. In 2x
    # integer form (avoids the /2): axis2[k] = 2*xs[k] + fw[k] + 2*round(off).
    # All such axis2 are bound equal (one axis); the shared axis2 is then pulled
    # to the envelope centre 2*W/2 = W via a soft term. Topology-derived: the
    # offset comes from the within-group abut layout, no reference coordinate.
    # axis_groups entries may be (k, off) [legacy] or (k, off, is_primary). Normalise.
    axg = []
    for _e in axis_groups:
        if len(_e) == 3:
            _k, _off, _prim = _e
        else:
            _k, _off = _e
            _prim = False
        if _k in present:
            axg.append((_k, _off, bool(_prim)))
    center_terms: List[object] = []
    if axg:
        # ONE shared axis var; each group's pair-axis is bound to it (HARD -- the
        # "ONE axis" intent is exact, not negotiable). The shared axis is CENTRED
        # in the envelope (HARD 2*axis == W, so the cell is symmetric about the
        # diff axis). The 2D pack then minimises H under these. This collapses the
        # per-pair axis scatter + off-centre float; the envelope is free to grow
        # in H (the honest tradeoff). All terms are 2x-integer to avoid the /2.
        # PARITY SLACK: axis2 (=2x the axis x) has the SAME parity as fw[k] (since
        # 2*xs + 2*off is even), so mixed-width-parity groups (odd vs even fw -- a
        # centre on a half- vs full-grid) can never share the EXACT same integer
        # axis2 (a hard equality across them is INFEASIBLE: the t4 groups have fw
        # 23/18/15/11/28). So bind each group's axis to ONE shared axis var within
        # +/-1 grid unit (the half-um parity quantum -- geometrically negligible,
        # far under the 2% spread band). This realises the "ONE axis" intent EXACTLY
        # up to the discrete-grid floor and converges fast + deterministically
        # (measured: OPTIMAL in ~6s, symmetry_placement PASS). The HONEST cost is
        # vertical stacking (H grows) -- the over-cell router cannot yet route the
        # resulting centred column (LVS unmatched), so this press is OPT-IN
        # (mirror_pairs_rigid) and gated by the checker, NOT the production default.
        axis2 = m.NewIntVar(0, 2 * HX, "axis2_shared")
        _AXIS_TOL = 1  # parity quantum (1 grid = 1 um in 2x => 0.5 um axis x)
        if str(axis_mode) == "centered_soft":
            # HYBRID CO-DESIGN: the PRIMARY axis (a declared DIFF pair -- the
            # differential input VIP/VIN) is HARD-bound to the shared centred axis.
            # One pair on the centred axis costs NO vertical stacking (a single
            # group straddles the axis by construction), so the cell's main symmetry
            # axis is EXACT. The SECONDARY mirror-pair axes get an L1 SOFT pull toward
            # the same axis instead of the hard +/-1 equality: a hard shared axis
            # across many non-overlapping groups forces them onto one x (separable
            # only in Y -> the tall unroutable column). The soft pull lets the 2D
            # envelope-min pack keep net-adjacent groups in HORIZONTALLY-adjacent
            # rows (the channel adjacency the over-cell router needs) while still
            # drawing the secondary axes toward the centred diff axis.
            _have_primary = any(prim for (_k, _off, prim) in axg)
            for (k, off, prim) in axg:
                axk = 2 * xs[k] + fw[k] + 2 * int(round(off))  # 2x the pair axis x
                if prim:
                    m.Add(axk - axis2 <= _AXIS_TOL)
                    m.Add(axis2 - axk <= _AXIS_TOL)
                    if k in exact_centres2:
                        actual = member_origin2_nm(k, 0) + exact_axis_offset2_nm(off)
                        m.Add(actual - 1000 * axis2 <= 1000 * _AXIS_TOL)
                        m.Add(1000 * axis2 - actual <= 1000 * _AXIS_TOL)
                else:
                    adev = m.NewIntVar(0, 2 * HX, f"axis_dev_{k}")
                    m.Add(adev >= axk - axis2)
                    m.Add(adev >= axis2 - axk)
                    center_terms.append(adev)
            if not _have_primary:
                # no declared diff pair -> nothing hard-anchors the shared axis;
                # bind the FIRST axis group hard so axis2 is a real axis (not a
                # free var the soft terms collapse to 0 for free).
                k0, off0, _ = axg[0]
                axk0 = 2 * xs[k0] + fw[k0] + 2 * int(round(off0))
                m.Add(axk0 - axis2 <= _AXIS_TOL)
                m.Add(axis2 - axk0 <= _AXIS_TOL)
                if k0 in exact_centres2:
                    actual = member_origin2_nm(k0, 0) + exact_axis_offset2_nm(off0)
                    m.Add(actual - 1000 * axis2 <= 1000 * _AXIS_TOL)
                    m.Add(1000 * axis2 - actual <= 1000 * _AXIS_TOL)
        else:
            # shared_hard (bring-up diagnostic): bind EVERY group's axis to ONE
            # shared axis var within +/-1 grid (exact symmetry; vertical-stack cost).
            for (k, off, _prim) in axg:
                axk = 2 * xs[k] + fw[k] + 2 * int(round(off))  # 2x the pair axis x
                m.Add(axk - axis2 <= _AXIS_TOL)
                m.Add(axis2 - axk <= _AXIS_TOL)
                if k in exact_centres2:
                    actual = member_origin2_nm(k, 0) + exact_axis_offset2_nm(off)
                    m.Add(actual - 1000 * axis2 <= 1000 * _AXIS_TOL)
                    m.Add(1000 * axis2 - actual <= 1000 * _AXIS_TOL)
        # centre the shared axis: axis2 -> W (axis2 = 2x the axis x; envelope spans
        # [0, W] -> centre x = W/2 -> 2*x = W). SOFT L1 (a HARD 2*axis==W is
        # infeasible against the bbox bounds when a wide group straddles the axis);
        # the _CENTER_WEIGHT pull centres the symmetric cell.
        cdev = m.NewIntVar(0, 2 * HX, "axis_center_dev")
        m.Add(cdev >= axis2 - Wv)
        m.Add(cdev >= Wv - axis2)
        center_terms.append(cdev)

    # WIRELENGTH LOCALITY (the routing/floorplan QUALITY obligation): per
    # multi-group net, the bbox span over member origins (HPWL proxy), weighted
    # PER NET by ``net_weights`` (the clique-model importance the intent layer
    # derives: ~1/(fanout-1), so a low-fanout signal net dominates and a high-
    # fanout power rail is diluted -- it is meant to distribute). With a real
    # ``wl_weight`` this PULLS electrically-coupled groups together (the Miller
    # cap abuts its output stage instead of parking at the cell edge), trading
    # bounded envelope for locality (the approved tradeoff). The weight stays
    # below _ENVELOPE_WEIGHT per unit so a gratuitous area blow-up that does not
    # shorten a net is never taken. net_weights is positional-aligned to ``nets``;
    # default 1.0 each (the legacy unweighted tie-break) when not supplied.
    nw = list(net_weights) if net_weights else []
    nsig = list(net_signal) if net_signal else []
    wl_terms: List[object] = []
    signal_spans: List[object] = []  # (sx+sy) of low-fanout signal nets (L-inf)
    for ni, net in enumerate(nets):
        members = [k for k in net if k in present]
        if len(members) < 2:
            continue
        nxlo = m.NewIntVar(0, HX, f"nxlo{ni}")
        nxhi = m.NewIntVar(0, HX, f"nxhi{ni}")
        nylo = m.NewIntVar(0, HY, f"nylo{ni}")
        nyhi = m.NewIntVar(0, HY, f"nyhi{ni}")
        for k in members:
            m.Add(nxlo <= xs[k])
            m.Add(nxhi >= xs[k])
            m.Add(nylo <= ys[k])
            m.Add(nyhi >= ys[k])
        sx = m.NewIntVar(0, HX, f"sx{ni}")
        sy = m.NewIntVar(0, HY, f"sy{ni}")
        m.Add(sx == nxhi - nxlo)
        m.Add(sy == nyhi - nylo)
        if not nw:
            # LEGACY byte-identical path (no per-net weights supplied): the exact
            # unweighted span sum the determinism/leakage golden tests pin. Adding
            # the weighted aux vars below would perturb the CP-SAT search path even
            # at an algebraically-equal objective, so the legacy path is preserved
            # verbatim and only the weighted caller takes the branch below.
            wl_terms += [sx, sy]
            continue
        w_net = nw[ni] if ni < len(nw) else 1.0
        # integerise the per-net weight (CP-SAT objective is integer-linear): scale
        # by 100 so the clique fractions (1, 1/2, 1/3, ...) survive rounding.
        wi = max(1, int(round(float(w_net) * 100)))
        wsx = m.NewIntVar(0, wi * HX, f"wsx{ni}")
        wsy = m.NewIntVar(0, wi * HY, f"wsy{ni}")
        m.Add(wsx == wi * sx)
        m.Add(wsy == wi * sy)
        wl_terms += [wsx, wsy]
        # L-infinity: a SIGNAL net (low DEVICE fanout, the gate's own currency via
        # net_signal) is the locality-gated kind. Collect its raw (sx+sy) so the
        # minimax var below pins the worst -- the exact quantity the gate bounds.
        # Fall back to the group-fanout proxy only when net_signal is absent.
        is_sig = (nsig[ni] if ni < len(nsig)
                  else len(members) <= _CP_SIGNAL_GROUP_FANOUT_MAX)
        if is_sig:
            ssum = m.NewIntVar(0, HX + HY, f"ssum{ni}")
            m.Add(ssum == sx + sy)
            signal_spans.append(ssum)

    # ASPECT-BAND intent (a GENERAL tile-pack QUALITY obligation, NOT a per-circuit
    # constant): a 2D envelope-min pack of these tiles lands portrait, but the
    # floorplan intent is a target W/H proportion (the reference calibrates the
    # band scalar only -- a scale-free ratio, never a coordinate). Penalize the L1
    # deviation of the pack from the target ratio: target a*H == W, so the deviation
    # is |W - a*H| in integer 2x form (a is rational q/d -> d*W vs q*H). The weight
    # sits BELOW the envelope per unit (a gratuitous area blow-up that does not
    # improve aspect is never taken) but ABOVE the wirelength tie-break, so the
    # solver rotates the pack toward the landscape band when it can do so at bounded
    # area cost (the honest tradeoff). aspect_target=None -> no term (byte-stable).
    aspect_terms: List[object] = []
    if aspect_target is not None and aspect_target > 0:
        # rational target a = q/d (d=100 grid) so the constraint is integer-linear.
        d = 100
        q = max(1, int(round(float(aspect_target) * d)))
        adev = m.NewIntVar(0, (q + d) * (HX + HY), "aspect_dev")
        m.Add(adev >= d * Wv - q * Hv)
        m.Add(adev >= q * Hv - d * Wv)
        aspect_terms.append(adev)

    # CENTERING weight sits BELOW the envelope (never trade area for centering)
    # but ABOVE the wirelength tie-break (centering the diff axis is a stronger
    # intent than a marginal wirelength save). When no envelope solution is
    # affected, the centering term breaks the otherwise-free lateral slide of the
    # axis to the cell centre.
    # L-INFINITY minimax: pin the WORST signal-net span (the gate's quantity) and
    # penalise IT, so the objective minimises the maximum span, not just the sum.
    # Only built on the weighted cp path (signal_spans populated) -> legacy callers
    # are byte-identical. Env CLAY_CP_MAXSPAN_WEIGHT overrides the weight.
    import os as _os
    maxspan_terms: List[object] = []
    if signal_spans:
        mspan = m.NewIntVar(0, HX + HY, "max_signal_span")
        for ss in signal_spans:
            m.Add(mspan >= ss)
        maxspan_terms.append(mspan)
    _mw_env = _os.environ.get("CLAY_CP_MAXSPAN_WEIGHT")
    _maxspan_weight = (int(_mw_env) if (_mw_env and _mw_env.strip())
                       else _CP_MAXSPAN_WEIGHT)

    # EDGE-AFFINITY (@rail_affinity / @anchor, Wave B4): a SOFT L1 pull on a group's
    # distance to a declared cell EDGE. side='bottom' pulls ys[k] toward 0 (the group
    # hugs the bottom supply rail); side='top' pulls the group's TOP edge toward the
    # envelope top (Hv - (ys[k]+fh[k]) -> 0). Generic geometry -- the group / side are
    # DATA (no circuit branch). Empty rail_affinities -> no aux var, no objective term ->
    # byte-identical for every existing caller (mirrors abut_pairs / order_constraints).
    rail_terms: List[object] = []
    for (rk, side) in (rail_affinities or ()):
        if rk not in present:
            continue
        s = str(side).lower()
        if s in ("bottom", "bot", "b", "vss", "gnd", "down"):
            rail_terms.append(ys[rk])                    # minimise y -> hug bottom
        elif s in ("top", "t", "vdd", "up"):
            dev = m.NewIntVar(0, HY, f"rail_top_dev_{rk}")
            m.Add(dev >= Hv - (ys[rk] + fh[rk]))         # top edge -> envelope top
            rail_terms.append(dev)
        # an unknown side is a no-op (never a crash) -- a reported gap, not a guess.

    # Objective. Built EXACTLY as before when no @abut is declared (abut_terms empty ->
    # the term is not appended), so every existing caller's CP model is byte-identical;
    # the @abut adjacency pull is added only when a declared pair is present.
    # A declared @guard_ring island is a LOCAL envelope: its bounding hull (iw x ih) is
    # minimised at the SAME ``_ENVELOPE_WEIGHT`` as the global cell (Wv + Hv). This is
    # not a tuned per-cell knob -- it is the ONE envelope principle applied uniformly at
    # every declared granularity (the whole cell AND each declared cluster), so a ringed
    # cluster's members abut instead of leaving a gratuitous empty band inside the frame.
    # ``island_span_terms`` is empty for a cell that declares no @guard_ring island ->
    # ``sum(...) == 0`` -> byte-identical to the pre-existing objective.
    objective = (
        _ENVELOPE_WEIGHT * (Wv + Hv + sum(island_span_terms))
        + _ASPECT_WEIGHT * sum(aspect_terms)
        + _CENTER_WEIGHT * sum(center_terms)
        + _maxspan_weight * sum(maxspan_terms)
        + wl_weight * sum(wl_terms)
    )
    if abut_terms:
        objective = objective + _ABUT_WEIGHT * sum(abut_terms)
    if rail_terms:
        objective = objective + _RAIL_AFFINITY_WEIGHT * sum(rail_terms)
    m.Minimize(objective)

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    # DETERMINISM CONTRACT (the leakage guard asserts byte-identical placement):
    # a WALL-CLOCK budget (max_time_in_seconds) terminates a FEASIBLE-not-proven
    # solve at a timing-dependent point -> different machines/runs return different
    # solutions (measured: the aspect term makes the model not prove OPTIMAL in
    # budget, and the wall-clock cutoff then yielded run-to-run-different packings,
    # a determinism + leakage violation). Use the DETERMINISTIC time budget so the
    # solve stops at a reproducible search point regardless of wall-clock speed --
    # byte-identical across runs/machines. The scalar is the same budget in
    # deterministic-time units (CP-SAT's internal work metric).
    solver.parameters.max_deterministic_time = float(max_time_s)
    solver.parameters.log_search_progress = False
    st = _observed_atomic_solve(solver, m)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # no solution within budget -> deterministic fallback, honestly labeled
        return replace(
            _place_fallback(groups, gap),
            backend_status=str(solver.StatusName(st)),
        )

    centers = {
        k: (tuple((solver.Value(native_centres2[k][i]) // 2) / 1000.0 for i in (0, 1))
            if k in native_centres2 else
            tuple(solver.Value(standalone_centres2[k][i]) / 2000.0 for i in (0, 1))
            if k in standalone_centres2 else
            (solver.Value(xs[k]) + fw[k] / 2.0, solver.Value(ys[k]) + fh[k] / 2.0))
        for k in keys
    }
    w, h, env = _envelope_from_centers(centers, size)
    status = "OPTIMAL" if st == cp_model.OPTIMAL else "FEASIBLE"
    native_row_capture = None
    if native_frame_binding is not None:
        from clay.layout.place.native_row_cp_frames import replay_native_row_frames_cp
        solution = tuple(int(value) for value in solver.ResponseProto().solution)
        model_bytes = m.Proto().SerializeToString(deterministic=True)
        replayed = replay_native_row_frames_cp(native_frame_binding, m.Proto(), solution,
            native_row_frames, gap_nm=gap_nm,
            group_authority={key:(coarse[key][5],coarse[key][6]) for key in native_row_frames})
        if any(replayed[key] != tuple(solver.Value(v) for v in native_centres2[key])
               for key in native_centres2):
            raise ValueError('native row CP frame solution differs from replay')
        native_row_capture = (native_frame_binding, model_bytes, solution)
    cross_row_proof = None
    cross_row_capture = None
    if cross_row_binding is not None:
        from clay.layout.place.cross_row_matching import (
            NativeCrossRowCpProof, replay_native_cross_row_cp_proof,
        )

        if native_frame_binding is not None:
            cross_row_capture = (cross_row_binding, model_bytes, solution)
        else:
            cross_row_proof = NativeCrossRowCpProof(
                cross_row_binding, m.Proto().SerializeToString(deterministic=True),
                tuple(int(value) for value in solver.ResponseProto().solution),
            )
            replay_native_cross_row_cp_proof(
                native_cross_row_inventory, cross_row_proof, native_cross_row_directions,
            )
    return PlacementResult(centers, w, h, env, status, float(solver.BestObjectiveBound()),
                           native_cross_row_cp_proof=cross_row_proof,
                           native_row_cp_capture=native_row_capture,
                           native_cross_row_cp_capture=cross_row_capture)


_ORIGINAL_PLACE_CPSAT = _place_cpsat


def _place_cpp(
    groups: Sequence[Group],
    nets: Sequence[Sequence[str]],
    symmetry_pairs: Sequence[Tuple[str, str]],
    axis_groups: Sequence[Tuple[str, float]],
    gap: float,
    wl_weight: int,
    max_time_s: float,
    axis_mode: str = "shared_hard",
    aspect_target: Optional[float] = None,
    net_weights: Sequence[float] = (),
    net_signal: Sequence[bool] = (),
    disjoint_clusters: Sequence[Sequence[str]] = (),
    island_pads: Sequence[float] = (),
    band_of_group: Optional[Dict[str, int]] = None,
    order_constraints: Sequence[Tuple[str, str, str]] = (),
    abut_pairs: Sequence[Tuple[str, str]] = (),
    rail_affinities: Sequence[Tuple[str, str]] = (),
) -> PlacementResult:
    """Pure C++20 deterministic 2D envelope-min placer (zero OR-Tools)."""
    import os as _os
    from clay._native import _engine_cpp

    keys = [g.key for g in groups]
    size = {g.key: (g.w, g.h) for g in groups}
    fw = {g.key: max(1, int(math.ceil(g.w + gap))) for g in groups}
    fh = {g.key: max(1, int(math.ceil(g.h + gap))) for g in groups}
    present = set(keys)

    # Respect exact zero-budget contract (e.g., max_time_s <= 0 => UNKNOWN/FALLBACK).
    if float(max_time_s) <= 0.0:
        return replace(_place_fallback(groups, gap), backend_status="UNKNOWN")

    nw = list(net_weights) if net_weights else []
    nsig = list(net_signal) if net_signal else []
    cpp_nets = []
    for ni, net in enumerate(nets):
        members = [k for k in net if k in present]
        if len(members) < 2:
            continue
        if not nw:
            cpp_nets.append((members, 1, False))
        else:
            w_net = nw[ni] if ni < len(nw) else 1.0
            wi = max(1, int(round(float(w_net) * 100)))
            is_sig = bool(nsig[ni]) if ni < len(nsig) else (len(members) <= _CP_SIGNAL_GROUP_FANOUT_MAX)
            cpp_nets.append((members, wi, is_sig))

    cpp_clusters = []
    _pads = list(island_pads or ())
    for ci, cl in enumerate(disjoint_clusters or ()):
        members = [k for k in cl if k in present]
        if not members:
            continue
        pad_um = _pads[ci] if ci < len(_pads) else 0.0
        cpp_clusters.append((members, max(0, int(math.ceil(float(pad_um))))))

    cpp_orders = []
    for (oa, ob, rel_kind) in (order_constraints or ()):
        if oa in present and ob in present and oa != ob:
            if rel_kind == "left_of":
                cpp_orders.append((oa, ob, False))
            elif rel_kind == "above":
                cpp_orders.append((oa, ob, True))

    cpp_axis_groups = []
    for _e in (axis_groups or ()):
        if len(_e) == 3:
            _k, _off, _prim = _e
        else:
            _k, _off = _e
            _prim = False
        if _k in present:
            cpp_axis_groups.append((_k, int(round(float(_off))), bool(_prim)))

    cpp_rails = []
    for (rk, side) in (rail_affinities or ()):
        if rk not in present:
            continue
        s = str(side).lower()
        if s in ("bottom", "bot", "b", "vss", "gnd", "down"):
            cpp_rails.append((rk, False))
        elif s in ("top", "t", "vdd", "up"):
            cpp_rails.append((rk, True))

    _mw_env = _os.environ.get("CLAY_CP_MAXSPAN_WEIGHT")
    _maxspan_weight = (
        int(_mw_env) if (_mw_env and _mw_env.strip()) else _CP_MAXSPAN_WEIGHT
    )

    spec = {
        "groups": [(g.key, float(g.w), float(g.h), int(fw[g.key]), int(fh[g.key])) for g in groups],
        "nets": cpp_nets,
        "has_net_weights": bool(nw),
        "symmetry_pairs": [(a, b) for (a, b) in (symmetry_pairs or ()) if a in present and b in present and a != b],
        "axis_groups": cpp_axis_groups,
        "axis_centered_soft": (str(axis_mode) == "centered_soft"),
        "clusters": cpp_clusters,
        "band_of_group": (
            {k: int(v) for k, v in (band_of_group or {}).items() if k in present}
            if band_of_group
            else None
        ),
        "orders": cpp_orders,
        "abut_pairs": [(a, b) for (a, b) in (abut_pairs or ()) if a in present and b in present and a != b],
        "rail_affinities": cpp_rails,
        "gap": float(gap),
        "wl_weight": int(wl_weight),
        "maxspan_weight": int(_maxspan_weight),
        "aspect_target": float(aspect_target) if (aspect_target is not None and float(aspect_target) > 0.0) else 0.0,
        "max_time_s": float(max_time_s) if math.isfinite(float(max_time_s)) else 20.0,
        "seed": 0,
    }

    sol = _engine_cpp.solve_envelope_min_cpp(spec)
    status = str(sol.get("status", "UNKNOWN"))
    if status not in ("OPTIMAL", "FEASIBLE"):
        return replace(_place_fallback(groups, gap), backend_status=status)

    xs_map = sol["xs"]
    ys_map = sol["ys"]
    centers = {
        k: (float(xs_map[k]) + fw[k] / 2.0, float(ys_map[k]) + fh[k] / 2.0)
        for k in keys
    }
    w, h, env = _envelope_from_centers(centers, size)
    return PlacementResult(centers, w, h, env, status, float(sol.get("lower_bound", 0.0)))


_ATOMIC_OBSERVATION = ContextVar('clay_atomic_cp_observation', default=False)
_ATOMIC_PROTO_MAX_BYTES = 16 * 1024 * 1024


@contextmanager
def atomic_placement_observation():
    """Select the generic row atomic caller, not every CP-SAT invocation."""
    token=_ATOMIC_OBSERVATION.set(True)
    try:
        yield
    finally:
        _ATOMIC_OBSERVATION.reset(token)


def _observed_atomic_solve(solver, model):
    """Record the real CP model and response without changing the solve policy."""
    from clay.flow.execution_session import current_session
    session = current_session()
    if session is None or not _ATOMIC_OBSERVATION.get():
        return solver.Solve(model)
    from collections import Counter
    from hashlib import sha256
    from clay.flow.execution_session import JournalLimitExceeded

    def save(proto, kind, phase):
        size=proto.ByteSize()
        if size > _ATOMIC_PROTO_MAX_BYTES:
            return {kind+'_omitted_bytes':size}
        data=proto.SerializeToString(deterministic=True)
        path=session.event_path.parent/f'atomic-cp-{phase.phase_id}-{kind}.pb'
        try:
            with path.open('xb') as stream: stream.write(data)
        except OSError as exc:
            return {kind+'_write_error':type(exc).__name__,kind+'_bytes':size}
        return {kind+'_sha256':sha256(data).hexdigest(),kind+'_path':str(path),kind+'_bytes':size}

    solved=False
    try:
        with session.phase('placement.atomic_cp_sat') as phase:
            proto=model.Proto()
            kinds=Counter(row.WhichOneof('constraint') or 'unset' for row in proto.constraints)
            details={**save(proto,'model',phase), **save(solver.parameters,'parameters',phase)}
            limit=float(solver.parameters.max_deterministic_time)
            phase.annotate(**details, variables=len(proto.variables), constraints=len(proto.constraints),
                           constraint_types=dict(sorted(kinds.items())),
                           hint_variables=len(proto.solution_hint.vars), model_stats=model.ModelStats(),
                           deterministic_limit=limit if math.isfinite(limit) else None,
                           num_search_workers=solver.parameters.num_search_workers,
                           random_seed=solver.parameters.random_seed)
            status=solver.Solve(model)
            solved=True
            response=solver.ResponseProto()
            phase.annotate(**save(response,'response',phase),solver_status=solver.StatusName(status),
                           solver_response=solver.ResponseStats(),
                           solver_deterministic_seconds=float(response.deterministic_time),
                           solver_wall_seconds=float(response.wall_time),
                           branches=int(response.num_branches),conflicts=int(response.num_conflicts))
            return status
    except JournalLimitExceeded:
        # Instrumentation is observational: an exhausted journal must not
        # replace the backend result or cause a second Solve after completion.
        return status if solved else solver.Solve(model)


def _place_fallback(groups: Sequence[Group], gap: float) -> PlacementResult:
    """Deterministic shelf packer: rows by decreasing height (tie-break by key), each row
    grown to ~a square target width. Overlap-free + byte-reproducible, but does not
    minimize the envelope or enforce symmetry -- the CP backend does both."""
    size = {g.key: (g.w, g.h) for g in groups}
    total_area = sum(g.w * g.h for g in groups)
    target_w = math.sqrt(total_area * 1.3) if total_area > 0 else 1.0
    ordered = sorted(groups, key=lambda g: (-g.h, g.key))

    centers: Dict[str, Tuple[float, float]] = {}
    cur_x = 0.0
    row_y = 0.0
    row_h = 0.0
    row_started = False
    for g in ordered:
        if row_started and cur_x + g.w > target_w:
            row_y += row_h + gap
            cur_x = 0.0
            row_h = 0.0
        centers[g.key] = (cur_x + g.w / 2.0, row_y + g.h / 2.0)
        cur_x += g.w + gap
        row_h = max(row_h, g.h)
        row_started = True

    w, h, env = _envelope_from_centers(centers, size)
    return PlacementResult(centers, w, h, env, "FALLBACK", None)


# --------------------------------------------------------------------------- #
# DEVICE-LEVEL BRIDGE: the CP envelope-min placer places GROUPS; the production
# materialization (anchor_ota.route_amp_leakfree) consumes a DEVICE-level
# ``{name -> (cx, cy)}`` pos dict. This bridge maps the CP group placement to
# that device dict WITHOUT a second materialization path: it is a drop-in for
# ``solve_hier_seqpair`` / ``solve_hierarchical_nonoverlap`` at the SAME seam.
#
# Group footprint: members laid LEFT-TO-RIGHT, tight (raw stamp + gap), same y
# -> a group is one same-kind X-CONTIGUOUS run in one y-row, which is exactly the
# adjacency ``group_guard_rings`` clusters on (transitive y-overlap -> x-run of
# same-kind MOS). N and P groups are INTERLEAVED by the CP 2D pack (NOT segregated
# into a top/bottom band stack) -- the spatial difference the measured root cause
# (ref_compare.py) identified between the 16555 band-stack and the 10704 ref.
#
# Pure topology (compact_matched_groups / mirror_pairs) + design footprints; no
# reference coordinate is read, so the placement is leakage-free by construction.
# --------------------------------------------------------------------------- #
def _order_group_members(
    members: Sequence[str],
    order_constraints: Sequence[Tuple[str, str, str]],
) -> Tuple[str, ...]:
    """Stable topological order for same-group horizontal obligations."""
    ordered = tuple(str(member) for member in members)
    present = set(ordered)
    rank = {name: index for index, name in enumerate(ordered)}
    outgoing = {name: set() for name in ordered}
    indegree = {name: 0 for name in ordered}
    for first, second, relation in order_constraints or ():
        first, second = str(first), str(second)
        if (
            str(relation) != "left_of"
            or first not in present
            or second not in present
            or first == second
            or second in outgoing[first]
        ):
            continue
        outgoing[first].add(second)
        indegree[second] += 1
    ready = sorted(
        (name for name in ordered if indegree[name] == 0),
        key=rank.__getitem__,
    )
    result = []
    while ready:
        name = ready.pop(0)
        result.append(name)
        for follower in sorted(outgoing[name], key=rank.__getitem__):
            indegree[follower] -= 1
            if indegree[follower] == 0:
                ready.append(follower)
                ready.sort(key=rank.__getitem__)
    if len(result) != len(ordered):
        raise ValueError(
            "same-group left_of constraints contain a cycle: "
            f"members={ordered!r}"
        )
    return tuple(result)


def _group_member_layout(
    members: Sequence[str],
    sizes: Dict[str, Tuple[float, float]],
    gap: float,
    *,
    abut_members: Optional[AbstractSet[FrozenSet[str]]] = None,
    od_inset: Optional[Dict[str, float]] = None,
    pair_min_gap_um: Optional[Dict[FrozenSet[str], float]] = None,
) -> Tuple[float, float, Dict[str, Tuple[float, float]]]:
    """Lay ``members`` left-to-right, centred on a common row.

    Returns ``(group_w, group_h, rel_offsets)`` where ``rel_offsets[m]`` is the
    member centre RELATIVE to the group centre (so the group is one X-contiguous
    same-row run -- the ring-clustering adjacency).

    DEFAULT (``abut_members`` empty): each member stamps its OWN footprint separated
    from the next by ``gap`` (``advance = w + gap``) -- byte-identical to the historical
    within-group layout.

    DIFFUSION ABUT (the declared ``@abut`` consumption -- the arraying-density fix):
    when a CONSECUTIVE member pair is in ``abut_members`` (a set of ``frozenset({a, b})``
    holding THIS group's declared ``@abut`` pairs) the inter-member GAP is COLLAPSED to a
    SHARED-OD pitch so the two device COMP (active/diffusion) rectangles ABUT into ONE
    continuous OD island crossed by both members' poly gates -- diffusion abut, not the
    ``w + gap`` stamp abut that fragments the OD into low-gate islands. The shared-OD
    pitch removes the non-diffusion x-margin on BOTH facing sides (``od_inset[m]`` is a
    member's per-side bbox->COMP inset in um) so the COMP EDGES meet:
    ``advance = w_i - od_inset[m_i] - od_inset[m_{i+1}]`` (vs the plain ``w_i + gap``).
    With no ``od_inset`` (or 0 -- the synthetic case) the pitch collapses to the
    footprint edge-to-edge (``advance = w_i``, gap removed, no positive gap), still one
    shared run. Clamped to a small positive floor so a pathological inset never inverts
    two members. Pure geometry of the declared pair + the members' own extents -- no
    reference coordinate, no per-circuit branch."""
    ws = [float(sizes.get(m, (1.0, 1.0))[0]) for m in members]
    hs = [float(sizes.get(m, (1.0, 1.0))[1]) for m in members]
    gh = max(hs) if hs else 0.0
    _abut = abut_members or frozenset()
    _pair_gaps = pair_min_gap_um or {}
    for pair, value in _pair_gaps.items():
        if (type(pair) is not frozenset or len(pair) != 2
                or any(type(name) is not str or not name for name in pair)
                or type(value) not in (int, float)
                or not math.isfinite(float(value)) or float(value) < 0):
            raise ValueError("group member pair spacing must be finite and non-negative")

    if not _abut and not any(
        frozenset((members[i], members[i + 1])) in _pair_gaps
        for i in range(len(members) - 1)
    ):
        # HISTORICAL layout (byte-identical -- the SAME arithmetic + order as before the
        # feature): each member stamps its own footprint separated by ``gap``. Kept
        # verbatim so a cell that declares no @abut re-pack is bit-for-bit unchanged (the
        # group footprint feeds an integer ceil in the packer; a reordered float sum
        # could tip it and perturb the placement).
        gw = sum(ws) + gap * (len(ws) - 1) if ws else 0.0
        rel: Dict[str, Tuple[float, float]] = {}
        cur = -gw / 2.0
        for m, w in zip(members, ws):
            rel[m] = (cur + w / 2.0, 0.0)  # common row centre
            cur += w + gap
        return gw, gh, rel

    # DIFFUSION-ABUT path (a declared @abut pair in this group): collapse the facing gap
    # to a SHARED-OD pitch so the members' COMP rectangles meet into one continuous OD.
    _od = od_inset or {}
    _FLOOR = 0.1  # um: a shared-OD pitch must never invert two members
    lefts: List[float] = []  # running LEFT edges (leftmost == 0 by construction)
    cur = 0.0
    for i, (m, w) in enumerate(zip(members, ws)):
        lefts.append(cur)
        if i + 1 < len(members):
            nxt = members[i + 1]
            if frozenset((m, nxt)) in _abut:
                advance = w - (float(_od.get(m, 0.0)) + float(_od.get(nxt, 0.0)))
                cur += max(advance, _FLOOR)
            else:
                pair = frozenset((m, nxt))
                cur += w + max(gap, float(_pair_gaps.get(pair, 0.0)))
    gw = (lefts[-1] + ws[-1]) if ws else 0.0
    rel = {}
    for m, w, lo in zip(members, ws, lefts):
        rel[m] = (lo + w / 2.0 - gw / 2.0, 0.0)  # centre on the common row
    return gw, gh, rel


def _native_group_exact_offsets(members, sizes, gap, *, abut_members, od_inset, pair_min_gap_um):
    """Re-evaluate the existing group layout without binary-float drift."""
    from decimal import Decimal
    widths = [Decimal(str(sizes.get(member, (1.0, 1.0))[0])) for member in members]
    gap = Decimal(str(gap))
    abut = abut_members or frozenset()
    insets = od_inset or {}
    pair_gaps = pair_min_gap_um or {}
    historical = not abut and not any(
        frozenset((members[i], members[i + 1])) in pair_gaps
        for i in range(len(members) - 1))
    lefts = []
    current = Decimal(0)
    for index, (member, width) in enumerate(zip(members, widths)):
        lefts.append(current)
        if index + 1 == len(members):
            continue
        other = members[index + 1]
        pair = frozenset((member, other))
        if historical:
            current += width + gap
        elif pair in abut:
            current += max(width - Decimal(str(insets.get(member, 0.0)))
                           - Decimal(str(insets.get(other, 0.0))), Decimal('0.1'))
        else:
            current += width + max(gap, Decimal(str(pair_gaps.get(pair, 0.0))))
    total = lefts[-1] + widths[-1] if widths else Decimal(0)
    return {member: (left + width / 2 - total / 2, Decimal(0))
            for member, width, left in zip(members, widths, lefts)}


def _group_ring_pad_um_by_kind(intent, tech_id: str) -> Dict[str, float]:
    """Per-MOS-kind GROUP-ring pad (um, per side): the ring's own outward
    extension for that kind's well polarity + HALF the worst-case ring-to-
    neighbour clearance -- the SAME currency ``group_ring_padded_extents`` adds
    to the band-stack solve sizes (one source, no second derivation). A compact
    matched group shares ONE ring, so the pad belongs at the GROUP's outer
    boundary; padding the group footprint by it makes the CP ``AddNoOverlap2D``
    keep the per-group RINGS disjoint by construction (the missing obligation the
    raw-stamp footprint dropped -> the measured 15-ring-overlap wall).

    Pure PDK rule scale (probe build) -> no reference coordinate. ``{}`` if the
    tech cannot be resolved (the pad collapses to 0; honest, not a silent fake).
    """
    try:
        from clay.layout.extract.realize_render import (
            _group_ring_clearance_nm,
            _ring_extension_nm,
            _RING_WELL_FOR_KIND,
            _norm_kind,
        )
        from clay.layout.tech import LayoutTech
    except Exception:
        return {}
    try:
        tech = LayoutTech.resolve(tech_id)
    except Exception:
        return {}
    half_clear = (_group_ring_clearance_nm(tech) + 1) // 2
    out: Dict[str, float] = {}
    for kind, well in _RING_WELL_FOR_KIND.items():
        out[_norm_kind(kind)] = (
            _ring_extension_nm(tech, well) + half_clear) / 1000.0
    return out


def via_landing_budget_um(tech_id: str) -> float:
    """The per-side ROUTING/VIA budget (um) a placement must reserve around each
    placed element so the router has room to land a rule-enclosed via and hold
    routed-metal min-space WITHOUT overhanging an adjacent element's fixed metal.

    THE MISSING OBLIGATION (R2, the systemic root)
    ==============================================
    The CP envelope-min placer minimises the bounding box: it packs elements as
    tight as ``AddNoOverlap2D`` + the gap allows. But a routed layer-transition
    needs a VIA LANDING PAD (cut/2 + the PDK enclosure) on each metal it touches,
    and that pad's same-layer edge must hold ``min_space`` to any FOREIGN element's
    metal. When the placer leaves only the bare gap between two device footprints,
    the router's emergent via pad overhangs the neighbour's device metal -> a
    device-short -> the via is suppressed -> the net opens (the measured gf180 cp
    net15 open). Reserving this budget at placement time gives the router the room
    the via + spacing needs, so the via PLACES instead of being suppressed.

    THE BUDGET (rule-derived, generic -- no per-PDK / per-circuit constant)
    ======================================================================
    For each routing metal layer k the worst-case foreign-facing reach of a via
    landing on metal k is ``via_pad_half(k) + min_space(metal_k)``:
      * ``via_pad_half(k) = ceil(cut_k / 2) + enclosure_k`` -- the metal pad the
        PDK requires around the cut (the SAME ``via_model.via_landing`` geometry
        the emit draws), per the larger of the below/above enclosure;
      * ``min_space(metal_k)`` -- the same-layer spacing the foundry deck enforces
        to a foreign conductor.
    The budget is the MAX of that reach over all (via, metal) layers the PDK
    declares, halved per side is NOT taken -- the full reach is reserved on the
    element boundary because the neighbour's own metal sits at its boundary, so a
    full ``via_pad_half + min_space`` per side is what keeps the via pad off the
    neighbour. Every term is read from the ``tech`` contract; zero ``if pdk ==``.

    Returns ``0.0`` when the PDK cannot be resolved or declares no via rules
    (honest -- the budget collapses to nothing, never a silent fake)."""
    try:
        from clay.layout.tech import LayoutTech
        from clay.layout.route.via_model import (
            via_cut_size_nm,
            via_enclosure_nm,
        )
    except Exception:
        return 0.0
    try:
        tech = LayoutTech.resolve(tech_id)
    except Exception:
        return 0.0

    def _layer_min_space_nm(layer_name: str) -> Optional[int]:
        try:
            return int(tech.min_space(layer_name))
        except Exception:
            return None

    best_nm = 0
    for vn in (tech.via_names() or []):
        cut = via_cut_size_nm(tech, vn)
        if not cut:
            continue
        eb, ea = via_enclosure_nm(tech, vn)
        pad_half = (int(cut) + 1) // 2 + max(int(eb or 0), int(ea or 0))
        # the via connects two adjacent metals; the foreign-facing reach is on the
        # ROUTING metal layers it lands on. Use the worst (largest) metal min_space
        # among the routing metals so the reserved budget holds on every layer the
        # router might transition on (generic over the PDK's metal stack).
        ms_candidates = []
        for ln in (tech.layer_names() or []):
            if not str(ln).lower().startswith("metal"):
                continue
            ms = _layer_min_space_nm(ln)
            if ms is not None:
                ms_candidates.append(ms)
        ms = max(ms_candidates) if ms_candidates else 0
        best_nm = max(best_nm, pad_half + ms)
    return best_nm / 1000.0


def cp_group_placement_pos(
    intent,
    sizes: Dict[str, Tuple[float, float]],
    *,
    gap: float = 1.0,
    passive_gap: float = 2.0,
    max_time_s: float = 20.0,
    backend: str = "auto",
    ring_pad_by_kind: Optional[Dict[str, float]] = None,
    tech_id: Optional[str] = None,
    via_budget_um: Optional[float] = None,
    mirror_pairs_rigid: bool = False,
    mirror_axis_mode: str = "shared_hard",
    aspect_target: Optional[float] = None,
    wl_weight: Optional[int] = None,
    keep_together_islands: Optional[Sequence[Sequence[str]]] = None,
    channel_relax_nets: Optional[Sequence[str]] = None,
    channel_pad_um: float = 0.0,
    channel_relax_passive_only: bool = True,
    device_bands: Optional[Dict[str, int]] = None,
    order_constraints: Optional[Sequence[Tuple[str, str, str]]] = None,
    abut_pairs: Optional[Sequence[Tuple[str, str]]] = None,
    od_inset_by_member: Optional[Dict[str, float]] = None,
    arrays: Optional[Sequence["ArrayTile"]] = None,
    matching_pair_axes: Optional[
        Dict[Tuple[str, str], Tuple[str, float]]
    ] = None,
    matching_pair_min_gap_um: Optional[Dict[FrozenSet[str], float]] = None,
    explicit_symmetry_pairs: Optional[Sequence[Tuple[str, str]]] = None,
    native_cross_row_inventory: object = None,
    native_cross_row_groups: Optional[Dict[str, str]] = None,
    native_cross_row_directions: Optional[Dict[Tuple[str, str], str]] = None,
    native_device_frames: Optional[Dict[str, Tuple[int, int, int, int]]] = None,
    native_device_grid_nm: Optional[int] = None,
) -> Tuple[Dict[str, Tuple[float, float]], "PlacementResult"]:
    """Device-level pos dict from the UNIFIED CP envelope-min element placement.

    ALL circuit elements are first-class placeable TILES in ONE 2D envelope-min
    pack: the matched MOS GROUPS (``compact_matched_groups``) AND the PASSIVES
    (caps/resistors, one tile each, footprint from ``sizes``). Net edges over the
    full element set pull each passive to its connected nodes; the MOS groups pack
    AROUND it -> a dense MOSAIC (no contiguous dead region from a side/top slab),
    via ONE general mechanism that treats a cap as just another element kind. No
    cap-specific placement mode exists -- the cap is not special.

    ``intent``: the ExtractedIntent (compact_matched_groups + mirror_pairs +
        passives + device nets read).
    ``sizes``: ``{element -> (w, h)}`` RAW stamp/footprint extents (MOS member
        stamps AND passive device-cell footprints) -- the leakage-free design
        footprint.
    ``ring_pad_by_kind`` / ``tech_id``: when guard rings will be materialised at
        compact-matched-group granularity, the CP group footprint MUST reserve the
        per-group ring pad so ``AddNoOverlap2D`` keeps the rings disjoint. Supply
        ``ring_pad_by_kind`` (``{norm_kind -> pad_um_per_side}``) directly, or a
        ``tech_id`` to derive it from :func:`_group_ring_pad_um_by_kind`. ``None``
        (no ring) keeps the bare raw-stamp footprint.
    ``keep_together_islands``: declared ``@guard_ring`` islands as member-DEVICE-name
        lists. Each is mapped to its covering GROUP keys and passed to
        :func:`place_envelope_min` as a ``disjoint_clusters`` entry, so the ring's
        members stay contiguous and the different-net ring frames come out disjoint
        (no rail-to-rail ring short). ``None``/empty -> byte-identical.
    ``channel_relax_nets`` / ``channel_pad_um``: the ROUTING-IN-THE-LOOP feedback.
        A net the router (adjudicated by the REAL LVS) reported OPEN on a prior pass
        needs a wider routing corridor between the groups it spans. Reserve
        ``channel_pad_um`` extra per-side footprint pad on EXACTLY the groups
        incident to a realized-open net, so ``AddNoOverlap2D`` leaves the router that
        corridor by construction. Keyed ONLY on the realized opens the caller passes
        back (never a static congestion estimate, never a net-name special case);
        ``None``/``0.0`` -> byte-identical.
    ``device_bands``: ``{device_name -> band_ordinal}`` from the declared
        ``@place_group`` floorplan ROWS. Resolved to a per-GROUP band (a compact
        matched group / passive tile inherits its members' declared row) and passed
        to :func:`place_envelope_min` as ``band_of_group``, so the pack compacts
        WITHIN each declared tier and preserves the band Y-ORDER (nmos row below the
        pmos row) instead of interleaving the tiers to shave area. Intent-derived (the
        declared rows -- no reference coordinate). ``None``/empty -> byte-identical.
    ``order_constraints`` / ``abut_pairs``: the declared ``@order(a, b, rel)`` /
        ``@abut(a, b)`` DEVICE-level intent (from the .clay, carried on the LayoutIntent).
        Each device is mapped to its covering GROUP key; a CROSS-group pair becomes an
        envelope-min constraint (``@order`` -> a HARD relative order left_of/above,
        ``@abut`` -> a tight-adjacency pull), while a SAME-group ``@abut`` pair drives the
        within-group SHARED-OD (diffusion-abut) member layout so the matched fingers land
        in ONE continuous OD island. Pure topology (device->group); ``None``/empty ->
        no constraint, no shared-OD collapse -> byte-identical.
    ``od_inset_by_member``: ``{device -> per-side bbox->COMP x-inset (um)}`` so a
        same-group ``@abut`` pair packs COMP-flush (the active rectangles meet). Absent
        (or 0) collapses to a footprint edge-to-edge pitch (gap removed) -- honest, never
        a fabricated inset. Only consulted for declared ``@abut`` members.
    ``arrays``: the DECLARED shared-diffusion ``@array`` tiles (:class:`ArrayTile`).
        Each is placed as ONE atomic footprint (the built shared-OD-strip extent), and
        its members are EXCLUDED from the compact-matched-group set so they are NOT
        packed as N free devices; each member's final position is the array tile centre
        plus its ``rel`` offset (the SAME offsets the emit path seats the one array stamp
        from). The tile reserves the group ring pad for its ``kind`` exactly like a
        compact matched group. ``None``/empty -> no array tiles, no member exclusion ->
        BYTE-IDENTICAL.
    ``matching_pair_axes``: canonical materializer-owned matching axes.  CP does
        not trust their old coordinate after compaction; it consumes only the pair
        identity + axis and emits a fresh owner certificate when one atomic CP
        group structurally owns both equal-footprint, perpendicularly aligned
        members.  Missing/invalid/non-owned pairs receive no certificate.
    ``explicit_symmetry_pairs``: canonical source obligations kept as hard owner
        pairs independently of inferred mirror components.
    Returns ``(pos, result)`` where ``pos`` is ``{element -> (cx, cy)}`` (MOS
    members + placed passives) consumed unchanged by the materialization seam, and
    ``result`` is the group-level :class:`PlacementResult` (envelope/status).
    """
    from clay.layout.extract.well_islands import compact_matched_groups

    if ring_pad_by_kind is None and tech_id is not None:
        ring_pad_by_kind = _group_ring_pad_um_by_kind(intent, tech_id)
    pad_by_kind = ring_pad_by_kind or {}

    # R2 VIA-LANDING / SPACING BUDGET (the systemic root): every placed element
    # must reserve room so a routed layer-transition can land its rule-enclosed
    # via pad + hold min-space WITHOUT overhanging a neighbour's fixed metal (else
    # the via is suppressed -> the net opens, the measured gf180 cp net15 open).
    # Add the per-side budget to EVERY element footprint (MOS groups + passives)
    # so the CP ``AddNoOverlap2D`` leaves >= (budget_a + budget_b) between any two
    # element stamps -> a via landing in that channel clears the foreign metal by
    # construction. Rule-derived (via cut + enclosure + min_space); engaged only
    # when a tech_id is supplied (same gating as the ring pad), so the bare path
    # is byte-identical. Explicit ``via_budget_um`` overrides the derivation.
    if via_budget_um is None and tech_id is not None:
        via_budget_um = via_landing_budget_um(tech_id)
    via_pad = float(via_budget_um or 0.0)

    cg = compact_matched_groups(intent)

    # DECLARED shared-diffusion @array members are placed as ONE atomic tile (added
    # below), so EXCLUDE them from the compact-matched-group set here -- else a
    # member would be double-assigned (its free/compact group AND the array tile)
    # and grp_of would be nondeterministic. A member that shares a compact group
    # with a NON-array device drops only itself (the remaining members keep their
    # group). Empty ``arrays`` -> no exclusion -> byte-identical ``cg``.
    array_tiles: List[ArrayTile] = list(arrays or ())
    native_row_frames = {}
    for tile in array_tiles:
        if tile.native_bbox_nm is None and tile.native_grid_nm is None:
            continue
        box, grid = tile.native_bbox_nm, tile.native_grid_nm
        if (type(box) is not tuple or len(box) != 4 or any(type(v) is not int for v in box)
                or box[2] <= box[0] or box[3] <= box[1]
                or type(grid) is not int or grid <= 0):
            raise ValueError('native row atomic source frame is malformed')
        key = f'A_{tile.key}'
        if key in native_row_frames:
            raise ValueError('native row atomic frame is duplicated')
        native_row_frames[key] = (box, grid)
    array_members: set = set()
    array_owner: Dict[str, str] = {}
    for _at in array_tiles:
        for _member in _at.members:
            member_name = str(_member)
            if member_name in array_owner:
                raise ValueError(
                    f"array member belongs to multiple atomic tiles: {member_name}"
                )
            array_members.add(member_name)
            array_owner[member_name] = f"A_{_at.key}"
    if array_members:
        cg = [
            (kind, body, tuple(m for m in members if str(m) not in array_members))
            for (kind, body, members) in cg
        ]
        cg = [g for g in cg if g[2]]
    # ROUTING-IN-THE-LOOP CHANNEL FEEDBACK: pre-map every element -> its group key
    # and mark the groups incident to any REALIZED-open net (the net(s) the router,
    # adjudicated by the real LVS, left open on the prior pass). Those groups get an
    # extra per-side corridor pad below, so the CP ``AddNoOverlap2D`` leaves the
    # router a wider channel around EXACTLY the groups a realized-open net must reach
    # between. Keyed ONLY on the caller's realized opens (no static estimate, no
    # net-name branch); empty / 0.0 pad -> no incident groups -> byte-identical.
    # CASE-INSENSITIVE net match: the real LVS reports schem nets upper-cased
    # (``schem=VOUT``) while the parsed cell_def / intent nets keep the source case
    # (``vout``). Normalise both sides so the corridor engages for the net the deck
    # actually named (else the pad silently no-ops and the loop cannot close the open).
    _chan_pad = float(channel_pad_um or 0.0)
    _channel_groups = _channel_incident_groups(intent, cg, array_owner,
        channel_relax_nets, _chan_pad, channel_relax_passive_only)

    # DECLARED @abut as a set of unordered DEVICE pairs (frozenset). A pair whose BOTH
    # members share ONE compact group drives the within-group SHARED-OD (diffusion-abut)
    # layout below; a cross-group pair becomes a group-adjacency constraint further down.
    # Empty declaration -> no shared-OD collapse -> byte-identical member layout.
    _abut_set: AbstractSet[FrozenSet[str]] = {
        frozenset((str(a), str(b))) for (a, b) in (abut_pairs or ())
        if str(a) != str(b)
    }

    # ISLAND MEMBERS PACK TIGHT: a group whose device(s) belong to a declared
    # @guard_ring island drops the ring/via pad from its footprint -- the ONE ring's
    # pad is reserved ONCE at the island BOUNDARY (place_envelope_min ``island_pads``),
    # not between members -- so island members abut (the reference's 0.5um) instead of
    # sitting 2*pad apart inside the frame. ``island_pad_of_group`` records the per-side
    # pad each island member group WOULD have reserved, so the boundary reservation uses
    # it. No declared island -> empty set -> every group keeps its pad -> byte-identical.
    _island_devs = {str(d) for isl in (keep_together_islands or ()) for d in isl}
    island_pad_of_group: Dict[str, float] = {}

    # build one Group per compact matched group; remember member relative layout.
    groups: List[Group] = []
    rel_by_group: Dict[str, Dict[str, Tuple[float, float]]] = {}
    members_by_group: Dict[str, Tuple[str, ...]] = {}
    kind_by_group: Dict[str, str] = {}
    for (kind, _body, members) in cg:
        key = f"G_{kind}_{members[0]}"
        ordered_members = _order_group_members(members, order_constraints)
        # restrict the declared @abut pairs to THIS group's members -> the consecutive
        # matched fingers laid at shared-OD pitch (one continuous OD island).
        _grp_abut = ({fs for fs in _abut_set if fs <= set(members)}
                     if _abut_set else None)
        gw, gh, rel = _group_member_layout(
            ordered_members, sizes, gap,
            abut_members=_grp_abut, od_inset=od_inset_by_member,
            pair_min_gap_um=matching_pair_min_gap_um)
        if native_device_frames and set(ordered_members) & set(native_device_frames):
            exact_rel = _native_group_exact_offsets(
                ordered_members, sizes, gap, abut_members=_grp_abut,
                od_inset=od_inset_by_member, pair_min_gap_um=matching_pair_min_gap_um)
            rel = {member: tuple(float(value) for value in offsets)
                   for member, offsets in exact_rel.items()}
        # GROUP-RING RESERVATION (the disjointness obligation): the group shares
        # ONE ring, so pad the GROUP envelope (not the inter-member gaps -- those
        # stay raw-abutted via ``rel``) by the ring pad for this kind's well.
        # Two padded group boxes then leave >= (pad_a + pad_b) between stamps ->
        # the two rings fit + hold spacing -> ring-disjoint by construction.
        pad = float(pad_by_kind.get(kind, 0.0)) + via_pad
        if key in _channel_groups:
            pad += _chan_pad  # routing-in-the-loop corridor for a realized-open net
        _in_island = bool(set(members) & _island_devs)
        _fpad = 0.0 if _in_island else pad  # island members abut; pad at the boundary
        groups.append(Group(key=key, w=gw + 2.0 * _fpad, h=gh + 2.0 * _fpad))
        if _in_island:
            island_pad_of_group[key] = pad
        rel_by_group[key] = rel
        members_by_group[key] = ordered_members
        kind_by_group[key] = kind

    # UNIFIED ALL-ELEMENT PLACEMENT: passives (caps/resistors) are FIRST-CLASS
    # placeable tiles in the SAME envelope-min pack, NOT bolted on post-placement.
    # One Group tile per passive, footprint from its device-cell extent in
    # ``sizes`` (the leakage-free design footprint). A passive is a single-member
    # group whose member centre == its group centre (rel offset 0). This is the
    # general fix: the cap stops being special -- it is one element kind among
    # many, and the net wirelength term (built over ALL element tiles below) pulls
    # it to its connected nodes so the MOS groups pack AROUND it (the mosaic).
    # ZERO cap/per-circuit branching: every passive of ANY kind is handled here.
    passive_gap_pad = max(0.0, (passive_gap - gap) / 2.0)
    for p in (getattr(intent, "passives", None) or []):
        pn = str(getattr(p, "name", ""))
        if not pn or pn in array_members:
            continue
        pw, ph = sizes.get(pn, (5.0, 5.0))
        pw = float(pw)
        ph = float(ph)
        key = f"P_{pn}"
        # pad the passive footprint by the same per-side passive clearance the
        # legacy band path held (passive_gap), so the unified pack keeps passives
        # the same distance from neighbours the bolt-on band reserved. PLUS the
        # R2 via-landing budget: a passive (cap plate) is FIXED artwork the router
        # transitions next to, so it too must reserve via-pad + min-space room
        # (the measured W*1.5 M5-pad-vs-CC1-plate sub-min gap).
        _pps = passive_gap_pad + via_pad  # per-side passive pad
        if key in _channel_groups:
            _pps += _chan_pad  # realized-open-net corridor (e.g. a Miller cap)
        if pn in _island_devs:
            island_pad_of_group[key] = _pps  # reserved at the island boundary
            _ppad = 0.0                      # tight footprint -> members abut
        else:
            _ppad = 2.0 * _pps
        groups.append(Group(key=key, w=pw + _ppad, h=ph + _ppad))
        rel_by_group[key] = {pn: (0.0, 0.0)}
        members_by_group[key] = (pn,)
        kind_by_group[key] = "passive"

    # DECLARED shared-diffusion @array: ONE atomic tile per array (footprint = the
    # built shared-OD-strip extent), reserving the SAME group ring + via-landing pad
    # a compact matched group of that kind reserves. The members expand from the tile
    # centre by their ``rel`` offsets (below), so the whole cluster packs, routes, and
    # rings as one unit -- the arraying-density fix at the FLOORPLAN level (the shared
    # OD itself is materialized by the emit path). Keyed ``A_<group>`` to never collide
    # with a ``G_``/``P_`` key. Empty ``arrays`` -> no tiles -> byte-identical.
    for _at in array_tiles:
        akey = f"A_{_at.key}"
        group, island_pad = _array_tile_group(_at, pad_by_kind, via_pad,
            _channel_groups, _chan_pad, _island_devs)
        if island_pad is not None:
            island_pad_of_group[akey] = island_pad
        groups.append(group)
        rel_by_group[akey] = {str(m): (float(dx), float(dy))
                              for m, (dx, dy) in (_at.rel or {}).items()}
        members_by_group[akey] = tuple(str(m) for m in _at.members)
        kind_by_group[akey] = str(_at.kind)

    # Each standalone native member retains its source-bound rigid translation
    # phase, including odd-nanometre extents and half-nanometre centres.  The
    # stored-row frames above have a separate source/certificate authority.
    native_group_phases = {}
    if native_device_frames:
        from decimal import Decimal
        if type(native_device_frames) is not dict or type(native_device_grid_nm) is not int or native_device_grid_nm <= 0:
            raise ValueError('native standalone device frames require one manufacturing grid')
        covered = set()
        modulus = 2 * native_device_grid_nm
        for key, members in members_by_group.items():
            phases = []
            for member in members:
                if member not in native_device_frames:
                    continue
                box = native_device_frames[member]
                if (type(box) is not tuple or len(box) != 4 or any(type(v) is not int for v in box)
                        or box[2] <= box[0] or box[3] <= box[1]):
                    raise ValueError('native standalone device frame is malformed')
                offsets = rel_by_group[key][member]
                offset2 = tuple(Decimal(str(v)) * 2000 for v in offsets)
                if any(not v.is_finite() or v != v.to_integral_value() for v in offset2):
                    raise ValueError('native standalone group offset is not an exact half-nanometre')
                phases.append(tuple((box[axis] + box[axis + 2] - int(offset2[axis])) % modulus
                                    for axis in (0, 1)))
                covered.add(member)
            if phases:
                if len(set(phases)) != 1:
                    raise ValueError('native rigid manufacturing-grid phase is incompatible within group')
                native_group_phases[key] = (*phases[0], native_device_grid_nm)
        if covered != set(native_device_frames):
            raise ValueError('native standalone device frame lacks one solver owner')

    # E2 SYMMETRY-RIGID PRESS (mirror_pairs_rigid): the CP envelope-min solve
    # ALREADY binds every CROSS-group declared mirror pair to ONE shared mirror
    # axis on equal rows (HARD symmetry, exact by construction -- see the
    # symmetry_pairs handling in _place_cpsat). For a SAME-group 2-member mirror
    # pair the within-group raw-abut layout is already centre-symmetric when the
    # members share (W,L) (the matched case -- the common one); rigidly RE-posing
    # unequal-width same-group members about the group centre instead BREAKS their
    # diffusion abutment (measured on the real oracle: LVS unmatched + DRC 29), so
    # the press does NOT touch the within-group rel offsets. The flag is accepted
    # on the cp path (the cross-group hard symmetry is the rigid press the CP
    # placer realises by construction); it records that the declared pairs are
    # placement-symmetric, NOT that the ROUTER mirrors them (the routed-symmetry
    # obligation is judged independently by the mirror_reports oracle).
    _ = mirror_pairs_rigid  # accepted; cross-group symmetry is hard by construction

    # group-level symmetry pairs: a device mirror pair whose two devices live in
    # DIFFERENT compact groups makes those two groups a mirror pair (e.g. the
    # diff-pair halves). Same-group mirrors are handled by the within-group
    # interdigitated layout, not the group axis. Pure topology.
    grp_of: Dict[str, str] = {}
    for key, members in members_by_group.items():
        for m in members:
            grp_of[m] = key

    if native_cross_row_inventory is not None:
        from decimal import Decimal
        from clay.layout.place.cross_row_matching import NativeCrossRowInventory

        if (type(native_cross_row_inventory) is not NativeCrossRowInventory
                or type(native_cross_row_groups) is not dict):
            raise ValueError("native cross-row source owner projection is malformed")
        native_cross_row_inventory.__post_init__()
        source_rows = dict(native_cross_row_inventory.row_source_members)
        if set(native_cross_row_groups) != set(source_rows):
            raise ValueError("native cross-row source owner denominator differs")
        tile_by_key = {f"A_{tile.key}": tile for tile in array_tiles}
        for row, owner in native_cross_row_groups.items():
            tile = tile_by_key.get(owner)
            if (tile is None or set(tile.members) != set(source_rows[row])
                    or set(members_by_group.get(owner, ())) != set(source_rows[row])):
                raise ValueError("native cross-row source partition differs from CP tile")
        for pair in native_cross_row_inventory.pairs:
            for endpoint in pair.endpoints:
                owner = native_cross_row_groups[endpoint.row_owner]
                tile = tile_by_key[owner]
                offset = tile.rel.get(endpoint.member)
                if type(offset) is not tuple or len(offset) != 2:
                    raise ValueError("native cross-row CP member offset is missing")
                for axis in (0, 1):
                    offset2 = Decimal(str(offset[axis])) * Decimal(2000)
                    actual = [channel.center2_nm[axis] for channel in endpoint.channels]
                    # Twice the member offset must equal the midpoint of the
                    # actual source channel centres, without rounding away phase.
                    if offset2 != offset2.to_integral_value() or 2 * offset2 != min(actual) + max(actual):
                        raise ValueError("native cross-row CP member offset differs from source channels")

    # MATCHING-AXIS OWNER CERTIFICATE.  The incoming coordinate may legitimately
    # be stale after an earlier canonical placement transform, so it is never used
    # as a new coordinate.  Its materializer-owned pair identity and orientation
    # are intersected with this CP compilation's own atomic-group membership and
    # member-offset proof.  Only a same-group equal-footprint pair on one common
    # perpendicular coordinate can be replayed after the solver moves its group.
    matching_pair_owners: Dict[Tuple[str, str], Tuple[str, str]] = {}
    for raw_pair, raw_axis in dict(matching_pair_axes or {}).items():
        if (
            type(raw_pair) is not tuple
            or len(raw_pair) != 2
            or raw_pair[0] == raw_pair[1]
            or type(raw_axis) is not tuple
            or len(raw_axis) != 2
            or raw_axis[0] not in {"x", "y"}
            or isinstance(raw_axis[1], bool)
        ):
            continue
        try:
            old_coordinate = float(raw_axis[1])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(old_coordinate):
            continue
        pair = tuple(sorted((str(raw_pair[0]), str(raw_pair[1]))))
        first, second = pair
        owner = grp_of.get(first)
        if owner is None or owner != grp_of.get(second):
            continue
        first_rel = rel_by_group.get(owner, {}).get(first)
        second_rel = rel_by_group.get(owner, {}).get(second)
        first_size = sizes.get(first)
        second_size = sizes.get(second)
        if (
            first_rel is None
            or second_rel is None
            or first_size is None
            or first_size != second_size
        ):
            continue
        axis = str(raw_axis[0])
        perpendicular_index = 1 if axis == "x" else 0
        if first_rel[perpendicular_index] != second_rel[perpendicular_index]:
            continue
        matching_pair_owners[pair] = (axis, owner)

    # DECLARED @order / cross-group @abut -> GROUP-LEVEL envelope-min constraints. The
    # .clay declaration names DEVICES; map each to its covering GROUP key. A SAME-group
    # pair is realised by the within-group shared-OD layout above (a @abut pair) or is
    # vacuous (@order inside one tile), so ONLY cross-group pairs become constraints:
    # @order -> a HARD relative order (left_of/above), @abut -> a tight-adjacency pull.
    # Pure topology (device->group); empty declaration -> empty -> byte-identical.
    group_order: List[Tuple[str, str, str]] = []
    _seen_order: set = set()
    for (a, b, rel_kind) in (order_constraints or ()):
        ga, gb = grp_of.get(str(a)), grp_of.get(str(b))
        if ga and gb and ga != gb and str(rel_kind) in ("left_of", "above"):
            entry = (ga, gb, str(rel_kind))
            if entry not in _seen_order:
                _seen_order.add(entry)
                group_order.append(entry)
    group_abut: List[Tuple[str, str]] = []
    _seen_abut: set = set()
    for (a, b) in (abut_pairs or ()):
        ga, gb = grp_of.get(str(a)), grp_of.get(str(b))
        if ga and gb and ga != gb:
            fs = frozenset((ga, gb))
            if fs not in _seen_abut:
                _seen_abut.add(fs)
                group_abut.append((ga, gb))

    sym = list(_mirror_component_owner_pairs(
        tuple(getattr(intent, "mirror_pairs", None) or ()),
        grp_of,
    ))
    # Source-declared/canonical pairs remain obligations even if inferred
    # mirror connectivity merges their owner graph into a larger component.
    for left, right in explicit_symmetry_pairs or ():
        a, b = grp_of.get(str(left)), grp_of.get(str(right))
        if a and b and a != b:
            pair=tuple(sorted((a,b)))
            if pair not in sym:
                sym.append(pair)
    sym.sort()

    # SHARED-AXIS INTENT (symmetry-axis obligation, topology-derived): every
    # declared 2-member reflection pair (a diff pair / a 2-leg current mirror)
    # whose BOTH members live in ONE compact group reflects about that group's
    # OWN vertical axis. The pair's axis x relative to the group centre is the
    # mean of the two members' within-group rel-x offsets (the abut layout
    # output, pure geometry of the matched stamps -- no reference coordinate).
    # Binding every such group's axis to ONE shared axis var + centring it makes
    # all declared diff/mirror pairs share ONE centered axis (the VIP/VIN off-
    # centre / per-pair-scatter root cause). >=3-member matched sets are an
    # interdigitated row (a distinct common-centroid intent) -- not a single
    # reflection axis, so they are NOT bound here (honest scope, matches the
    # checker's >=3-member ADVISORY treatment).
    # The shared-axis press engages under the opt-in mirror_pairs_rigid flag (the
    # symmetry-rigid intent). Default OFF leaves axis_groups EMPTY -> the CP model
    # adds NO axis term -> the cp cell is BYTE-IDENTICAL (LVS-safe, no regression).
    # The HONEST blocker (measured on the EXACT oracle): pressing all groups onto
    # one centred axis stacks them into a tall column the over-cell PathFinder
    # router cannot connect (LVS unmatched). So the press is a bring-up capability
    # the checker gates, NOT the production default, until the router can route a
    # centred-symmetric floorplan. See the round verdict.
    # axis_groups carries (group_key, axis_offset_um, is_primary). ``is_primary``
    # marks a group hosting a DECLARED DIFF PAIR (the differential input axis --
    # VIP/VIN): in the centered_soft hybrid the primary axis is HARD-centred (one
    # pair, no vertical-stack conflict) so the cell's main symmetry axis is exact,
    # while the secondary mirror-pair axes get the soft centre pull (so they
    # converge without forcing the unroutable column).
    axis_groups: List[Tuple[str, float, bool]] = []
    mirror_decl = [tuple(p) for p in (getattr(intent, "mirror_pairs", None) or [])
                   if len(p) == 2] if mirror_pairs_rigid else []
    diff_decl = [tuple(p) for p in (getattr(intent, "diff_pairs", None) or [])
                 if len(p) == 2] if mirror_pairs_rigid else []
    # diff pairs first so a group hosting one is tagged primary even if it also
    # appears as a mirror pair (the differential axis dominates).
    declared_pairs = [(p, True) for p in diff_decl] + [(p, False) for p in mirror_decl]
    seen_axis_groups: set = set()
    for ((da, db), _is_primary) in declared_pairs:
        a, b = str(da), str(db)
        ga, gb = grp_of.get(a), grp_of.get(b)
        if not ga or ga != gb:
            continue  # cross-group pair handled by the group symmetry axis above
        if len(members_by_group.get(ga, ())) != 2:
            continue  # >=3-member interdigitated set: not a single reflection axis
        if ga in seen_axis_groups:
            continue
        rel = rel_by_group.get(ga, {})
        ra = rel.get(a)
        rb = rel.get(b)
        if ra is None or rb is None:
            continue
        seen_axis_groups.add(ga)
        # axis offset from the group centre = mean of the pair members' rel-x.
        axis_groups.append((ga, (float(ra[0]) + float(rb[0])) / 2.0, bool(_is_primary)))

    # net hyperedges over ALL ELEMENT TILES (the wirelength term that pulls each
    # passive INTO the device mosaic): each net, mapped to the set of element
    # tiles (MOS groups + passive tiles) its terminals touch. Passives are
    # included on EXACTLY the same footing as MOS groups (no cap branch) -- the
    # cap's net edge (e.g. net25/VOUT) is what draws it to the output-stage groups
    # so the groups pack around it instead of leaving a slab for a parked block.
    net_groups: Dict[str, set] = {}
    net_devs: Dict[str, set] = {}   # net -> distinct DEVICE names (gate fanout)
    for d in (getattr(intent, "devices", None) or []):
        g = grp_of.get(str(d.name))
        if g is None:
            continue
        for _term, net in (getattr(d, "nets", None) or {}).items():
            net_groups.setdefault(str(net), set()).add(g)
            net_devs.setdefault(str(net), set()).add(str(d.name))
    for p in (getattr(intent, "passives", None) or []):
        g = grp_of.get(str(getattr(p, "name", "")))
        if g is None:
            continue
        for _term, net in (getattr(p, "nets", None) or {}).items():
            net_groups.setdefault(str(net), set()).add(g)
            net_devs.setdefault(str(net), set()).add(str(getattr(p, "name", "")))
    # WIRELENGTH-LOCALITY INTENT (the floorplan obligation, derived -- never a
    # per-circuit constant): each multi-group net is weighted by the CLIQUE model
    # ``1/(fanout-1)`` over the number of distinct GROUPS it touches. A 2-group
    # signal net (the Miller cap to its output stage) gets full weight 1.0; a
    # high-fanout power rail (VDD/GND touching many groups) is diluted toward 0 --
    # it is MEANT to distribute, so it must not dominate the pack. This is what
    # makes the cap abut the output stage instead of parking at the cell edge.
    # Pure topology (the netlist fanout), no name special-case, no coordinate.
    net_items = [(net, sorted(gs)) for net, gs in net_groups.items()
                 if len(gs) >= 2]
    nets = [gs for _net, gs in net_items]
    net_weights = [1.0 / (len(gs) - 1) for _net, gs in net_items]
    # SIGNAL classification by DEVICE fanout (the SAME currency the wirelength
    # gate uses: FidelityThresholds.signal_fanout_max), so the placer's L-infinity
    # locality term targets EXACTLY the nets the gate bounds -- not a group-fanout
    # proxy that drifts from the gate. A net touching few devices is a signal net
    # (locality-gated); a high-fanout power/distribution net is exempt.
    from clay.config import fidelity_thresholds as _ft
    _sig_max = int(_ft().signal_fanout_max)
    net_signal = [len(net_devs.get(net_name, ())) <= _sig_max
                  for net_name, _gs in net_items]

    # the global wirelength multiplier. The per-net clique weight (above, baked
    # *100 into the span vars) carries the relative importance; this scalar sets
    # how much bounded envelope the solver will spend to shorten the dominant
    # nets. Default _CP_WL_WEIGHT; env CLAY_CP_WL_WEIGHT overrides (the optimizer
    # seam for tuning the locality<->envelope tradeoff). Stays below the envelope
    # dominance so a net is never shortened at unbounded area cost.
    import os as _os
    if wl_weight is None:
        _env = _os.environ.get("CLAY_CP_WL_WEIGHT")
        wl_weight = int(_env) if (_env and _env.strip()) else _CP_WL_WEIGHT
    _ts_env = _os.environ.get("CLAY_CP_MAX_TIME_S")
    if _ts_env and _ts_env.strip():
        max_time_s = float(_ts_env)

    # RING-ISLAND KEEP-TOGETHER (declared @guard_ring obligation): map each declared
    # ring island (member DEVICE names) to the SET of GROUP keys covering it, so the
    # envelope-min solve keeps the island contiguous + the different-net ring frames
    # disjoint (see place_envelope_min ``disjoint_clusters``). A member device that
    # shares a compact group with a non-member drags the whole group into the hull
    # (correct: one ring encloses one contiguous well region). Pure topology (device
    # -> group), no coordinate; None/empty -> byte-identical.
    island_clusters: List[List[str]] = []
    island_pads: List[float] = []  # per-side ring/via pad reserved at each island boundary
    for island in (keep_together_islands or ()):
        gkeys: List[str] = []
        seen_g: set = set()
        pads: List[float] = []
        for d in island:
            g = grp_of.get(str(d))
            if g is not None and g not in seen_g:
                seen_g.add(g)
                gkeys.append(g)
                pads.append(island_pad_of_group.get(g, 0.0))
        if gkeys:
            island_clusters.append(gkeys)
            island_pads.append(max(pads) if pads else 0.0)

    # DECLARED @place_group TIER map (device -> band ordinal) -> per-GROUP band. A
    # compact matched group's members are same-kind and share ONE declared band (the
    # pmos_band / nmos_band); resolve each group's band from its members' declared rows.
    # SCOPE = the transistor tiers (the nmos/pmos band Y-ORDER the tier metric measures):
    # a PASSIVE tile (cap/resistor) is EXCLUDED from the vertical band stack -- its
    # @place_group row is a middle-tier statement paired with a COLUMN (col=3 -> banked
    # BESIDE the MOS core), NOT a full-width horizontal strip between the MOS bands.
    # Forcing the large MIM caps into a middle strip stacks the cell PORTRAIT (measured:
    # aspect 0.92 / 1.71x area, off the landscape band); letting the passives float in
    # the wirelength mosaic (as the un-tiered pack does) keeps the cell landscape while
    # the MOS tiers separate. A group whose members carry NO declared row (or disagree)
    # is left UNBANDED -> it floats (honest -- never a fabricated band). Pure topology
    # (device -> declared row + element kind); None/empty -> byte-identical.
    band_of_group: Dict[str, int] = {}
    if device_bands:
        for key, members in members_by_group.items():
            if kind_by_group.get(key) == "passive":
                continue  # passives float in the mosaic (their row is a Step-B concern)
            rows = {int(device_bands[str(mm)]) for mm in members
                    if str(mm) in device_bands}
            if len(rows) == 1:
                band_of_group[key] = next(iter(rows))

    result = place_envelope_min(
        groups, nets=nets, net_weights=net_weights, net_signal=net_signal,
        symmetry_pairs=sym, axis_groups=axis_groups,
        axis_mode=str(mirror_axis_mode), aspect_target=aspect_target,
        gap=gap, wl_weight=int(wl_weight), max_time_s=max_time_s, backend=backend,
        disjoint_clusters=island_clusters,
        island_pads=island_pads,
        band_of_group=(band_of_group or None),
        order_constraints=group_order,
        abut_pairs=group_abut,
        native_cross_row_inventory=native_cross_row_inventory,
        native_cross_row_groups=native_cross_row_groups,
        native_cross_row_directions=native_cross_row_directions,
        native_row_frames=native_row_frames,
        native_group_phases=native_group_phases,
    )
    result = replace(result, matching_pair_owners=matching_pair_owners)

    # expand ALL tile centres -> element positions (member rel + tile centre).
    # This covers MOS group members AND the passive tiles (single-member, rel 0):
    # every element -- including the caps/resistors -- gets its position from the
    # ONE unified envelope-min solve, so the cap is co-placed in the mosaic, not
    # bolted on afterward. No separate passive-band / cap-target step exists.
    pos: Dict[str, Tuple[float, float]] = {}
    for key, (gcx, gcy) in result.centers.items():
        rel = rel_by_group.get(key, {})
        for m, (rx, ry) in rel.items():
            pos[m] = (gcx + rx, gcy + ry)

    return pos, result
