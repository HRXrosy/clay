"""Owned terminal component indexing and via escape domain mixin for GridlessEngine."""
from __future__ import annotations

import bisect
from collections import OrderedDict
import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from weakref import WeakKeyDictionary

from clay._native import load_native
from clay.layout.route.component_access_session import (
    ComponentAccessSession,
    EscapeProof,
    EscapeWorldKey,
    PartitionContext,
    SiteLegalityKey,
    ViaLegalityProof,
)
from clay.layout.route.gridless_types import (
    GroupRouteBudgetStop,
    NegotiationStopReason,
    Rect,
    _AccessStaticDisconnectionRecorder,
    _CONG_WEIGHT,
    _COST_FACTOR_SCALE,
    _OwnedTerminalAtom,
    _STEP_COST_PER_NM,
    _SpatialIndex,
    _ViaSite,
    _closed_rects_touch,
    _eg_attr,
    _norm,
    _owned_terminal_atom_key,
    _rects_clear as _base_rects_clear,
    _sep as _base_sep,
)


_BATCH_VIA_FORBIDDEN_OVERRIDES = (
    "_via_legal",
    "_via_legal_with_active_scans",
    "_evaluate_via_static_legality",
    "_compute_via_static_legality",
    "_cached_component_access_via_legal",
    "_static_or_direct_obstacle_scan",
    "_clear_obstacles",
    "_reject_fixed_same_net_via_pad_spacing",
    "_scan_committed_metal",
    "_scan_committed_cuts",
    "_record_hard_rejection",
    "_watchdog_check",
    "_owned_landing_has_minimum_throat",
    "_wide_indexes",
    "_reject_intent_spacing",
    "_reject_foreign_committed_metal_contact",
    "_reject_committed_via_pad_spacing",
    "_reject_mim_bottom_routing_clearance",
    "_reject_mim_derived_unrelated_clearance",
    "_reject_mim_carve_bottom_interaction",
    "_reject_group_route_local_via",
    "_reject_group_route_global_via",
    "_local_fixed_cuts",
    "_raw_gap_metallised",
)


_STACK_ACCESS_WORLD_CACHE: "WeakKeyDictionary[Any, Tuple[Any, Dict[str, Any], Dict[Any, Any]]]" = WeakKeyDictionary()
_COMPONENT_ESCAPE_SHARED_CACHE: "WeakKeyDictionary[Any, Tuple[Any, OrderedDict]]" = WeakKeyDictionary()


class _GridlessOwnedEscapeMixin:
    """Owned terminal component graph, via event coordinates, and escape proofs."""

    # ------------------------------------------------------------- via escape
    def _escape_landing(self, net: str, px: int, py: int, gi: int,
                        ) -> Optional[Tuple[int, int, int]]:
        """The M1->M2 escape via that connects the pin UP with NO wire leg.

        In a dense pack the pin often sits within min_space of a foreign finger, so
        NO wire can leave the pin centre (the wire's own half-width pokes into the
        gap). The escape is instead a via whose M1 landing pad is OFFSET just enough
        to clear the foreign metal while STILL COVERING the pin centre (offset <=
        the pad half): the pad overlaps the pin's own metal, so the via connects by
        construction.

        Returns ``(x, y, 0)`` for an authorized engine-default array, or None
        when no such landing is legal. Other planar or layered candidates remain
        available to the normal bounded search."""
        if gi >= self._max_gi:  # top routable metal: nothing to escape up to
            return None
        hit = self._escape_landing_n(net, px, py, gi, self.via_redundancy)
        if hit is not None:
            return (hit[0], hit[1], 0)  # engine default: byte-identical via
        if self._single_via_escape_allowed():
            hit = self._escape_landing_n(net, px, py, gi, 1)
            if hit is not None:
                return (hit[0], hit[1], 1)
        return None

    def _escape_landing_n(self, net: str, px: int, py: int, gi: int, n: int,
                          ) -> Optional[Tuple[int, int]]:
        """Best ``(x, y)`` escape-via site of redundancy ``n`` whose landing pad
        clears the foreign metal while STILL COVERING the pin centre (|offset| <= the
        pad half). Ranked by (congestion, |offset|): congestion includes the
        own-artwork band price (R7a), so the escape stops re-picking the same banded
        offset every rip-up round (the livelock a legality-only smallest-offset rule
        guarantees) and lands touch-or-clear when one is reachable. None if no legal
        covering site exists."""
        bh, _ah = (self._via_pad_half1[gi] if n == 1 else self._via_pad_half[gi])
        nn = 1 if n == 1 else None  # legality/geometry override for the single via
        # offsets keeping the pad over the pin (|offset| <= pad half), fine step.
        step = max(20, bh // 4)
        offs = list(range(-bh, bh + 1, step))
        if 0 not in offs:
            offs.append(0)
        if bh not in offs:
            offs.append(bh)
        x_sites = self._on_manufacturing_grid_lines(
            [px + offset for offset in offs],
            px - bh,
            px + bh,
            preserve_frontier_sides=True,
        )
        y_sites = self._on_manufacturing_grid_lines(
            [py + offset for offset in offs],
            py - bh,
            py + bh,
            preserve_frontier_sides=True,
        )
        x_sites = sorted(x_sites, key=lambda value: (abs(value - px), value))
        y_sites = sorted(y_sites, key=lambda value: (abs(value - py), value))
        best = None
        best_key = None
        for y in y_sites:
            for x in x_sites:
                dx, dy = x - px, y - py
                if not self._via_legal(x, y, gi, net, n=nn):
                    continue
                site = _ViaSite(x, y, gi, 1 if n == 1 else 0)
                if self._via_pad_attempt_conflicts(net, site):
                    continue
                key = (self._via_congestion(
                           x, y, gi, net, n=1 if n == 1 else None),
                       abs(dx) + abs(dy), abs(dy), abs(dx))
                if best_key is None or key < best_key:
                    best = (x, y)
                    best_key = key
        return best

    @staticmethod
    def _point_in_closed_rect(x: int, y: int, rect: Rect) -> bool:
        x0, y0, x1, y1 = _norm(rect)
        return x0 <= x <= x1 and y0 <= y <= y1

    @staticmethod
    def _owned_landing_has_minimum_throat(
        atom_rect: Rect,
        landing_rect: Rect,
        min_width: int,
    ) -> bool:
        """Whether two individually legal rectangles join without a width neck.

        Closed-touch is sufficient for electrical connectivity but not for the
        deck's merged-metal width/space rules: a point or short edge contact makes
        two orthogonal ``2 * min_width`` violation cores.  Containment is already
        clean.  Otherwise use the deck's Euclidean junction-width contract: the
        contact projection diagonal must reach one PDK minimum width.  A full
        shared edge is therefore retained, while a point/short edge is rejected.
        This is pure local geometry; it imports no CandidatePool or exact-plan
        authority.
        """
        ax0, ay0, ax1, ay1 = _norm(atom_rect)
        lx0, ly0, lx1, ly1 = _norm(landing_rect)
        if (
            (ax0 <= lx0 and ay0 <= ly0 and ax1 >= lx1 and ay1 >= ly1)
            or (lx0 <= ax0 and ly0 <= ay0 and lx1 >= ax1 and ly1 >= ay1)
        ):
            return True
        overlap_x = min(ax1, lx1) - max(ax0, lx0)
        overlap_y = min(ay1, ly1) - max(ay0, ly0)
        width = int(min_width)
        return (
            overlap_x >= 0
            and overlap_y >= 0
            and overlap_x * overlap_x + overlap_y * overlap_y >= width * width
        )

    def _endpoint_diagnostic_step(self, net, diagnostic_work, event):
        """Check the native stop first, then debit one opt-in diagnostic event."""
        self._watchdog_check(net, "planar_endpoint_diagnostic")
        diagnostic_work(event)

    def _owned_terminal_component(
        self,
        net: str,
        px: int,
        py: int,
        gi: int,
        *,
        diagnostic_work=None,
    ) -> Tuple[_OwnedTerminalAtom, ...]:
        """Exact drawn same-layer component authorized by the pin's terminal atom.

        The seed must contain the exact pin point, own the routed net, and carry one
        unambiguous non-empty terminal-conductor ref.  Once seeded, connectivity is
        purely physical: same-layer, same-net rectangles join only when their CLOSED
        exact extents touch.  Refs do not partition drawn metal, and neither a bbox
        envelope nor near-by clearance may invent an edge.

        Component membership is intentionally recomputed per access.  The immutable
        atom projection is only a compact view of canonical obstacle provenance; no
        CandidatePool, resource plan, owned-index certificate, or placement fact is
        consulted here.
        """
        # A diagnostic call may stop in any index bucket or subsequent throat
        # traversal.  It can reuse an existing current-source entry, but a new
        # index stays local: neither success nor an interrupted call publishes
        # a partially observed entry into the production cache.
        if diagnostic_work is not None:
            if not callable(diagnostic_work):
                raise TypeError("diagnostic work must be callable")
            def work(event):
                self._endpoint_diagnostic_step(net, diagnostic_work, event)
            work("component.enter")
        else:
            work = None
        source_atoms = getattr(self, "_owned_terminal_atoms", ())
        group_key = (str(net), int(gi))
        group_indexes = getattr(
            self, "_owned_terminal_group_indexes", None
        )
        if not isinstance(group_indexes, dict):
            group_indexes = {}
            if work is None:
                self._owned_terminal_group_indexes = group_indexes
        indexed_group = group_indexes.get(group_key)
        if indexed_group is None or indexed_group[0] is not source_atoms:
            atoms_list = []
            for source_index, atom in enumerate(source_atoms):
                if work is not None:
                    work("component.source_atom")
                if source_index % 1024 == 0:
                    self._watchdog_check(net, "component_membership")
                if atom.gi == int(gi) and atom.owner == str(net):
                    atoms_list.append(atom)
            atoms = tuple(atoms_list)
            spatial_index = _SpatialIndex()
            rect_to_index: Dict[Rect, int] = {}
            for atom_index, atom in enumerate(atoms):
                if work is not None:
                    work("component.index_atom")
                if atom_index % 1024 == 0:
                    self._watchdog_check(net, "component_membership")
                rect = tuple(_norm(atom.rect))
                rect_to_index[rect] = atom_index
                if work is None:
                    spatial_index.add(str(atom_index), rect)
                else:
                    spatial_index.add(str(atom_index), rect, diagnostic_work=work)
            indexed_group = (
                source_atoms,
                atoms,
                spatial_index,
                rect_to_index,
            )
            if work is None:
                group_indexes[group_key] = indexed_group
        else:
            _source_atoms, atoms, spatial_index, rect_to_index = indexed_group

        seed_indices_list = []
        for index, atom in enumerate(atoms):
            if work is not None:
                work("component.seed_atom")
            if index % 1024 == 0:
                self._watchdog_check(net, "component_membership")
            if (
                self._point_in_closed_rect(px, py, atom.rect)
                and type(atom.terminal_conductor_ref) is str
                and bool(atom.terminal_conductor_ref)
            ):
                seed_indices_list.append(index)
        seed_indices = tuple(seed_indices_list)
        if not seed_indices:
            return ()

        pending = list(seed_indices)
        visited = set(seed_indices)
        cursor = 0
        neighbour_work = 0
        while cursor < len(pending):
            if work is not None:
                work("component.visit_atom")
            if cursor % 1024 == 0:
                self._watchdog_check(net, "component_membership")
            atom_index = pending[cursor]
            cursor += 1
            atom = atoms[atom_index]
            neighbours = []
            query = (spatial_index.query(atom.rect) if work is None else
                     spatial_index.query(atom.rect, diagnostic_work=work))
            for _owner, rect in query:
                if work is not None:
                    work("component.neighbour")
                if neighbour_work % 1024 == 0:
                    self._watchdog_check(net, "component_membership")
                neighbour_work += 1
                rect = tuple(_norm(rect))
                neighbour_index = rect_to_index[rect]
                if (
                    neighbour_index not in visited
                    and _closed_rects_touch(atom.rect, rect)
                ):
                    neighbours.append(neighbour_index)
            for neighbour_index in sorted(set(neighbours)):
                if work is not None:
                    work("component.enqueue")
                visited.add(neighbour_index)
                pending.append(neighbour_index)
        return tuple(atoms[index] for index in sorted(visited))

    def _owned_terminal_stack_access(
        self,
        net: str,
        px: int,
        py: int,
        gi: int,
    ) -> Tuple[
        Tuple[_OwnedTerminalAtom, ...],
        Tuple[Tuple[int, int, int], ...],
    ]:
        """Replay literal same-net device cuts across one terminal component.

        Planar membership starts at the ordinary compiler-issued terminal seam.
        A cross-layer edge exists only where one exact ``device_cut`` owns the
        routed net and closed-overlaps real owned metal on both adjacent routing
        layers.  This consumes the stamp's already emitted via ladder; equal ref
        strings, bboxes, and nearby clearance never manufacture a transition.
        """

        planar_seed = self._owned_terminal_component(net, px, py, gi)
        if not planar_seed:
            return (), ()
        net_s = str(net)
        raw_atoms = getattr(self, "_owned_terminal_atoms", ())
        raw_cut_prov = getattr(self, "_obs_cut_provenance", {})
        grid_nm = max(1, int(getattr(self, "_manufacturing_grid_nm", 1)))
        can_cache_stack = "_watchdog_check" not in self.__dict__
        stack_world_key = (
            id(raw_atoms),
            len(raw_atoms),
            id(raw_cut_prov),
            len(raw_cut_prov),
            id(getattr(self, "_obs_idx", None)),
            grid_nm,
            int(self._n_layers),
        )
        seed_cache_key = (net_s, planar_seed)
        world_net_index: Optional[Dict[str, Any]] = None
        seed_result_cache: Optional[Dict[Any, Any]] = None
        if can_cache_stack:
            cached_world = _STACK_ACCESS_WORLD_CACHE.get(self)
            if cached_world is not None and cached_world[0] == stack_world_key:
                world_net_index, seed_result_cache = cached_world[1], cached_world[2]
                cached_res = seed_result_cache.get(seed_cache_key)
                if cached_res is not None:
                    comp_cached, fixed_all_cached = cached_res
                    gi_i = int(gi)
                    return (
                        comp_cached,
                        tuple(st for st in fixed_all_cached if st[2] != gi_i),
                    )
            else:
                world_net_index = {}
                seed_result_cache = {}
                _STACK_ACCESS_WORLD_CACHE[self] = (
                    stack_world_key,
                    world_net_index,
                    seed_result_cache,
                )

        if world_net_index is not None and net_s in world_net_index:
            atoms_boxes_by_gi, cuts_by_lo = world_net_index[net_s]
            if not cuts_by_lo:
                comp_sorted = tuple(sorted(planar_seed, key=_owned_terminal_atom_key))
                if seed_result_cache is not None and len(seed_result_cache) < 4096:
                    seed_result_cache[seed_cache_key] = (comp_sorted, ())
                return comp_sorted, ()
        else:
            atoms_by_gi_lists: Dict[int, List[_OwnedTerminalAtom]] = {}
            for atom in raw_atoms:
                if atom.owner == net_s:
                    atoms_by_gi_lists.setdefault(int(atom.gi), []).append(atom)
            matching_cuts: List[Tuple[int, Rect]] = []
            for raw_key, raw_provenance in raw_cut_prov.items():
                if (
                    type(raw_provenance) is not tuple
                    or len(raw_provenance) != 3
                    or raw_provenance[0] != net_s
                    or raw_provenance[1] != "device_cut"
                    or type(raw_key) is not tuple
                    or len(raw_key) != 2
                    or type(raw_key[0]) is not int
                    or type(raw_key[1]) is not tuple
                    or len(raw_key[1]) != 4
                ):
                    continue
                rk1 = raw_key[1]
                if (
                    type(rk1[0]) is not int
                    or type(rk1[1]) is not int
                    or type(rk1[2]) is not int
                    or type(rk1[3]) is not int
                ):
                    continue
                matching_cuts.append((raw_key[0], rk1))
            cuts_by_lo: Dict[int, List[Rect]] = {}
            if matching_cuts:
                matching_cuts.sort()
                for cut_lo_gi, cut_rect in matching_cuts:
                    cuts_by_lo.setdefault(cut_lo_gi, []).append(tuple(_norm(cut_rect)))
            atoms_boxes_by_gi: Dict[int, List[Tuple[_OwnedTerminalAtom, int, int, int, int]]] = {}
            for a_gi, a_list in atoms_by_gi_lists.items():
                boxes_list = []
                for a_obj in a_list:
                    r0, r1, r2, r3 = a_obj.rect
                    if r0 > r2:
                        r0, r2 = r2, r0
                    if r1 > r3:
                        r1, r3 = r3, r1
                    boxes_list.append((a_obj, r0, r1, r2, r3))
                atoms_boxes_by_gi[a_gi] = boxes_list
            if world_net_index is not None:
                world_net_index[net_s] = (atoms_boxes_by_gi, cuts_by_lo)
            if not cuts_by_lo:
                comp_sorted = tuple(sorted(planar_seed, key=_owned_terminal_atom_key))
                if seed_result_cache is not None and len(seed_result_cache) < 4096:
                    seed_result_cache[seed_cache_key] = (comp_sorted, ())
                return comp_sorted, ()

        def transition_point(
            cut_rect: Rect,
            atom_rect: Rect,
        ) -> Optional[Tuple[int, int]]:
            cx0, cy0, cx1, cy1 = _norm(cut_rect)
            ax0, ay0, ax1, ay1 = _norm(atom_rect)
            ix0, iy0 = max(cx0, ax0), max(cy0, ay0)
            ix1, iy1 = min(cx1, ax1), min(cy1, ay1)
            if ix1 <= ix0 or iy1 <= iy0:
                return None

            def midpoint_on_grid(lo: int, hi: int) -> Optional[int]:
                middle = (lo + hi) // 2
                candidates = self._on_manufacturing_grid_lines(
                    (lo, middle, hi),
                    lo,
                    hi,
                    preserve_frontier_sides=True,
                )
                if not candidates:
                    return None
                return min(
                    candidates,
                    key=lambda value: (abs(2 * value - lo - hi), value),
                )

            x = midpoint_on_grid(ix0, ix1)
            y = midpoint_on_grid(iy0, iy1)
            return None if x is None or y is None else (int(x), int(y))

        visited = set(planar_seed)
        pending = list(sorted(planar_seed, key=_owned_terminal_atom_key))
        fixed_states = set()
        cursor = 0
        work = 0
        while cursor < len(pending):
            atom = pending[cursor]
            cursor += 1
            a_gi = int(atom.gi)
            ax0, ay0, ax1, ay1 = atom.rect
            if ax0 > ax1:
                ax0, ax1 = ax1, ax0
            if ay0 > ay1:
                ay0, ay1 = ay1, ay0
            for neighbour, nx0, ny0, nx1, ny1 in atoms_boxes_by_gi.get(a_gi, ()):
                if work % 1024 == 0:
                    self._watchdog_check(net, "terminal_stack_membership")
                work += 1
                if (
                    neighbour not in visited
                    and ax0 <= nx1 and nx0 <= ax1
                    and ay0 <= ny1 and ny0 <= ay1
                ):
                    visited.add(neighbour)
                    pending.append(neighbour)
            for lo, target_gi in (
                (a_gi, a_gi + 1),
                (a_gi - 1, a_gi - 1),
            ):
                if lo < 0 or target_gi < 0 or target_gi >= self._n_layers:
                    continue
                target_boxes = atoms_boxes_by_gi.get(target_gi)
                if not target_boxes:
                    continue
                for cut_rect in cuts_by_lo.get(lo, ()):
                    cx0, cy0, cx1, cy1 = cut_rect
                    if cx1 < ax0 or ax1 < cx0 or cy1 < ay0 or ay1 < cy0:
                        continue
                    for target, tx0, ty0, tx1, ty1 in target_boxes:
                        if work % 1024 == 0:
                            self._watchdog_check(
                                net, "terminal_stack_membership"
                            )
                        work += 1
                        if tx1 <= cx0 or cx1 <= tx0 or ty1 <= cy0 or cy1 <= ty0:
                            continue
                        point = transition_point(cut_rect, target.rect)
                        if point is None:
                            continue
                        fixed_states.add((point[0], point[1], target_gi))
                        if target not in visited:
                            visited.add(target)
                            pending.append(target)
        comp_out = tuple(sorted(visited, key=_owned_terminal_atom_key))
        fixed_all = tuple(sorted(fixed_states))
        if seed_result_cache is not None and len(seed_result_cache) < 4096:
            seed_result_cache[seed_cache_key] = (comp_out, fixed_all)
        return (
            comp_out,
            tuple(state for state in fixed_all if state[2] != int(gi)),
        )

    @staticmethod
    def _clearance_frontier_coordinates(
        rect: Rect,
        offset: int,
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Exact centre-coordinate events of one rect + clearance footprint."""
        x0, y0, x1, y1 = _norm(rect)
        offset = int(offset)
        return (x0 - offset, x1 + offset), (y0 - offset, y1 + offset)

    @staticmethod
    def _integer_interval_representatives(
        lo: int,
        hi: int,
        events: Sequence[int],
    ) -> Tuple[int, ...]:
        """Every integer event and one representative of each open event interval."""
        lo, hi = sorted((int(lo), int(hi)))
        boundaries = sorted({
            lo,
            hi,
            *(int(value) for value in events if lo <= int(value) <= hi),
        })
        representatives = set(boundaries)
        for left, right in zip(boundaries, boundaries[1:]):
            if right - left > 1:
                representatives.add((left + right) // 2)
        return tuple(sorted(representatives))

    def _owned_atom_via_event_coordinates(
        self,
        atom: _OwnedTerminalAtom,
        net: str,
        model_n: Optional[int] = None,
    ) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        """Complete via-legality event partition for one pad-touch domain.

        ``_via_legal`` changes truth only when a candidate pad/cut crosses an exact
        blocker clearance frontier (plus the isolated identical-cut point).  A via
        centre need not lie inside the atom: electrical contact requires its LOWER
        pad to closed-touch the atom, so the exact centre domain is the atom grown
        by that model's lower-pad half.  Take every event and one integer
        representative between adjacent events in that full domain.  This is event
        enumeration, not continuous nm sampling. Immutable events are cached;
        committed geometry and the current attempt-cut ledger are overlaid for
        each endpoint access.
        """
        x0, y0, x1, y1 = _norm(atom.rect)
        bh, ah = (
            self._via_pad_half1[atom.gi]
            if model_n == 1
            else self._via_pad_half[atom.gi]
        )
        cx0, cy0, cx1, cy1 = (
            x0 - int(bh),
            y0 - int(bh),
            x1 + int(bh),
            y1 + int(bh),
        )
        centre_domain = (cx0, cy0, cx1, cy1)
        cache_key = (
            _owned_terminal_atom_key(atom),
            str(net),
            1 if model_n == 1 else 0,
        )
        static_cache = getattr(
            self, "_owned_atom_static_via_events", None
        )
        if not isinstance(static_cache, dict):
            static_cache = {}
            self._owned_atom_static_via_events = static_cache
        cached_static = static_cache.get(cache_key)
        if cached_static is None:
            x_events: Set[int] = {cx0, cx1}
            y_events: Set[int] = {cy0, cy1}
        else:
            x_events = set(cached_static[0])
            y_events = set(cached_static[1])
        work = 0

        def add_frontiers(rect: Rect, offset: int, *, cut_ext: Optional[int] = None):
            nonlocal work
            work += 1
            if work % 256 == 0:
                self._watchdog_check(net, "component_access_events")
            x_frontiers, y_frontiers = self._clearance_frontier_coordinates(
                rect, offset
            )
            expanded = (
                x_frontiers[0],
                y_frontiers[0],
                x_frontiers[1],
                y_frontiers[1],
            )
            if not _closed_rects_touch(centre_domain, expanded):
                return
            x_events.update(x_frontiers)
            y_events.update(y_frontiers)
            if cut_ext is not None:
                rx0, ry0, rx1, ry1 = _norm(rect)
                if rx1 - rx0 == 2 * cut_ext and ry1 - ry0 == 2 * cut_ext:
                    x_events.add((rx0 + rx1) // 2)
                    y_events.add((ry0 + ry1) // 2)

        ext = self._via_cut_extent(atom.gi, model_n)
        cut_offset = ext + int(self._via_space[atom.gi])
        cut_query = (
            cx0 - cut_offset,
            cy0 - cut_offset,
            cx1 + cut_offset,
            cy1 + cut_offset,
        )
        if cached_static is None:
            for layer_gi, half in ((atom.gi, bh), (atom.gi + 1, ah)):
                idx = self._obs_idx.get(layer_gi)
                if idx is None:
                    continue
                reach = int(half) + int(self._space[layer_gi])
                query = (
                    cx0 - reach,
                    cy0 - reach,
                    cx1 + reach,
                    cy1 + reach,
                )
                for owner, rect in sorted(
                    idx.query(query),
                    key=lambda item: (
                        "" if item[0] is None else str(item[0]),
                        tuple(_norm(item[1])),
                    ),
                ):
                    if owner == net:
                        continue
                    add_frontiers(rect, reach)
            for rect in self._local_fixed_cuts(atom.gi, cut_query):
                add_frontiers(rect, cut_offset, cut_ext=ext)
            static_cache[cache_key] = (
                tuple(sorted(x_events)),
                tuple(sorted(y_events)),
            )

        # Foreign committed wires and via pads are hard via obligations too.
        # They were soft when this event constructor was introduced; leaving
        # their later hard contact/spacing checks out of the domain can hide an
        # otherwise legal landing in a narrow committed-metal opening. Keep
        # these events outside the immutable cache and query only the exact
        # model-grown atom neighbourhood. Both kinds of conductor already live
        # in the canonical metal index; its owner semantics remain unchanged.
        spacing_partners = getattr(self, "_intent_spacing_rules", {}).get(net, {})
        for layer_gi, half in ((atom.gi, bh), (atom.gi + 1, ah)):
            idx = self._metal_idx.get(layer_gi)
            if idx is None:
                continue
            base_offsets = {
                int(half) + 1,  # Foreign closed contact is always forbidden.
                int(half) + int(self._space[layer_gi]),
            }
            # The named-net oracle may require more than the PDK halo. Reuse
            # its compiled pair distances, including in the local index query.
            reach = max(
                max(base_offsets),
                int(half) + max(spacing_partners.values(), default=0),
            )
            query = (
                cx0 - reach,
                cy0 - reach,
                cx1 + reach,
                cy1 + reach,
            )
            for owner, rect in sorted(
                idx.query(query),
                key=lambda item: (
                    "" if item[0] is None else str(item[0]),
                    tuple(_norm(item[1])),
                ),
            ):
                if owner == net:
                    continue
                offsets = set(base_offsets)
                declared_space = spacing_partners.get(owner, 0)
                if declared_space:
                    offsets.add(int(half) + int(declared_space))
                for offset in sorted(offsets):
                    add_frontiers(rect, offset)

        attempt_cuts = tuple(sorted(
            tuple(_norm(rect))
            for rect in self._attempt_cuts.get(atom.gi, ())
            if _closed_rects_touch(_norm(rect), cut_query)
        ))
        for rect in attempt_cuts:
            add_frontiers(rect, cut_offset, cut_ext=ext)

        # Every committed cut contributes its net-blind spacing frontier. The
        # exact identical-cut event is an allowed reuse only for this net;
        # foreign cuts instead retain their strict contact boundary. The via
        # oracle remains the authority for both cases.
        cut_idx = self._cut_idx.get(atom.gi)
        if cut_idx is not None:
            committed_reach = max(cut_offset, ext + 1)
            committed_query = (
                cx0 - committed_reach,
                cy0 - committed_reach,
                cx1 + committed_reach,
                cy1 + committed_reach,
            )
            for owner, rect in sorted(
                cut_idx.query(committed_query),
                key=lambda item: (
                    "" if item[0] is None else str(item[0]),
                    tuple(_norm(item[1])),
                ),
            ):
                add_frontiers(
                    rect, cut_offset, cut_ext=ext if owner == net else None,
                )
                if owner != net:
                    add_frontiers(rect, ext + 1)

        return (
            tuple(self._on_manufacturing_grid_lines(
                self._integer_interval_representatives(
                    cx0, cx1, tuple(x_events)
                ),
                cx0,
                cx1,
                preserve_frontier_sides=True,
            )),
            tuple(self._on_manufacturing_grid_lines(
                self._integer_interval_representatives(
                    cy0, cy1, tuple(y_events)
                ),
                cy0,
                cy1,
                preserve_frontier_sides=True,
            )),
        )

    def _component_access_via_legality_cache(
        self,
        net: str,
    ) -> ComponentAccessSession:
        """Bind the partition world and return its query owner.

        The broad context still invalidates the event/throat partition because
        local attempt/self-cut frontiers can add candidate coordinates.  Full via
        verdicts carry their own exact local attempt-cut signature, so a remote
        cut does not discard proofs for every unrelated terminal landing.
        """
        attempt_cuts = tuple(
            (
                int(lo),
                tuple(sorted(tuple(_norm(rect)) for rect in rects)),
            )
            for lo, rects in sorted(self._attempt_cuts.items())
        )
        context = PartitionContext(
            int(getattr(self, "_committed_index_generation", 0)),
            str(net),
            attempt_cuts,
            tuple(sorted(getattr(
                self, "_search_forbidden_via_sites", set()
            ))),
        )
        self._component_access.bind_partition(context)
        return self._component_access

    def _merge_component_access_rejections(
        self,
        rejections: Sequence[Tuple[Tuple[str, str], int]],
        samples: Sequence[
            Tuple[Tuple[Any, ...], int, Tuple[int, int, int, int]]
        ],
    ) -> None:
        table = getattr(self, "_hard_rejections", None)
        if not isinstance(table, dict):
            table = {}
            self._hard_rejections = table
        for key, count in rejections:
            table[key] = table.get(key, 0) + int(count)
        sample_table = getattr(self, "_hard_rejection_samples", None)
        if not isinstance(sample_table, dict):
            sample_table = {}
            self._hard_rejection_samples = sample_table
        for cause, count, extent in samples:
            previous = sample_table.get(cause)
            if previous is None:
                sample_table[cause] = (int(count), tuple(extent))
            else:
                old_count, old_extent = previous
                sample_table[cause] = (
                    int(old_count) + int(count),
                    (
                        min(old_extent[0], extent[0]),
                        min(old_extent[1], extent[1]),
                        max(old_extent[2], extent[2]),
                        max(old_extent[3], extent[3]),
                    ),
                )

    def _cached_component_access_via_legal(
        self,
        x: int,
        y: int,
        lo: int,
        net: str,
        model_n: Optional[int],
        cache: ComponentAccessSession,
        *,
        attempt_cut_pool: Optional[Sequence[Rect]] = None,
    ) -> bool:
        """Replay exact via legality plus its causal evidence in one access world."""
        _rects_clear = _eg_attr("_rects_clear", _base_rects_clear)
        raw_pool = (
            self._attempt_cuts.get(lo, ())
            if attempt_cut_pool is None else attempt_cut_pool
        )
        if raw_pool:
            ext = self._via_cut_extent(lo, model_n)
            cut = (x - ext, y - ext, x + ext, y + ext)
            vsp = self._via_space[lo]
            local_attempt_cuts = tuple(sorted(
                tuple(_norm(other))
                for other in raw_pool
                if other == cut or not _rects_clear(cut, other, vsp)
            ))
        else:
            local_attempt_cuts = ()
        forb_sites = getattr(self, "_search_forbidden_via_sites", None)
        is_forb = (
            bool(forb_sites)
            and self._via_search_identity(x, y, lo, model_n) in forb_sites
        )
        key = SiteLegalityKey(
            int(getattr(self, "_committed_index_generation", 0)),
            str(net),
            int(x),
            int(y),
            int(lo),
            1 if model_n == 1 else 0,
            local_attempt_cuts,
            is_forb,
            self._via_static_legality_key(x, y, lo, net, model_n),
        )
        cached = cache.lookup_site(key)
        if cached is not None:
            legal, rejections, samples = cached
            self._merge_component_access_rejections(rejections, samples)
            return bool(legal)

        outer_rejections = getattr(self, "_hard_rejections", {})
        outer_samples = getattr(self, "_hard_rejection_samples", {})
        self._hard_rejections = {}
        self._hard_rejection_samples = {}
        completed = False
        legal = False
        try:
            legal = self._via_legal(x, y, lo, net, n=model_n)
            completed = True
        finally:
            local_rejections = tuple(sorted(
                (cause, int(count))
                for cause, count in self._hard_rejections.items()
            ))
            local_samples = tuple(
                (cause, int(count), tuple(extent))
                for cause, (count, extent) in sorted(
                    self._hard_rejection_samples.items()
                )
            )
            self._hard_rejections = outer_rejections
            self._hard_rejection_samples = outer_samples
            self._merge_component_access_rejections(
                local_rejections, local_samples
            )
        if completed:
            cache.store_site(
                key, ViaLegalityProof(bool(legal), local_rejections, local_samples),
                max_expansions=self.max_expansions,
            )
        return bool(legal)

    def _owned_atom_mim_carve_blocker(self, atom: _OwnedTerminalAtom):
        """Prove an entire lower-pad access domain empty using an existing carve.

        Every admitted landing must touch the exact owned atom. If that atom is
        wholly inside one immutable MIM-carved blocker, its lower pad therefore
        touches the blocker too, which the unchanged via oracle always rejects.
        Partial coverage grants no shortcut; neither candidates nor checker
        rules are weakened, and existing native stack transitions remain intact.
        """
        x0, y0, x1, y1 = _norm(atom.rect)
        for owner, blocker, kind in getattr(
            self, "_mim_carve_bottom_blockers_by_gi", {}
        ).get(atom.gi, ()):
            bx0, by0, bx1, by1 = _norm(blocker)
            if bx0 <= x0 and by0 <= y0 and x1 <= bx1 and y1 <= by1:
                return owner, blocker, kind
        return None

    def _owned_atom_via_states(
        self,
        atom: _OwnedTerminalAtom,
        net: str,
        model_n: Optional[int] = None,
        *,
        legality_cache: Optional[ComponentAccessSession] = None,
    ) -> Tuple[
        Tuple[Tuple[Tuple[int, int, int], Optional[_ViaSite]], ...],
        int,
    ]:
        """Every event-distinct legal via state with a clean atom landing.

        Via legality is partitioned by ``_owned_atom_via_event_coordinates``.
        Junction-width truth adds a Euclidean boundary inside those rectangular
        cells.  Sample every existing frontier on both integer sides, the exact
        atom/pad overlap breakpoints, and the integer ceil-sqrt throat boundary.
        This keeps the enumeration finite and deterministic while preventing one
        forbidden frontier point from hiding the only legal interior landing.
        """
        if model_n is not None and model_n != self.via_redundancy and not (
            model_n == 1 and self._single_via_escape_allowed()
        ):
            return (), 0
        states = []
        candidate_count = 0
        bh, _ah = (
            self._via_pad_half1[atom.gi]
            if model_n == 1
            else self._via_pad_half[atom.gi]
        )
        carved_domain = self._owned_atom_mim_carve_blocker(atom)
        if carved_domain is not None:
            owner, blocker, kind = carved_domain
            self._record_hard_rejection(
                "mim_carve_bottom_interaction", owner, gi=atom.gi,
                candidate_rect=_norm(atom.rect), candidate_half=int(bh),
                required_space=0, blocker_rect=blocker, blocker_kind=kind,
            )
            return (), 0
        x0, y0, x1, y1 = _norm(atom.rect)
        cx0, cy0, cx1, cy1 = (
            x0 - int(bh),
            y0 - int(bh),
            x1 + int(bh),
            y1 + int(bh),
        )
        attempt_cut_pool = None
        cut_reach = self._via_cut_extent(atom.gi, model_n) + self._via_space[atom.gi]
        cut_query = (cx0 - cut_reach, cy0 - cut_reach, cx1 + cut_reach, cy1 + cut_reach)
        if legality_cache is not None:
            # Every tested centre lies in this atom's lower-pad domain. Only
            # cuts touching its cut/spacing-grown domain can enter any site's
            # exact key. Preserve raw rectangles and duplicates; the existing
            # per-site predicate and canonical via oracle remain authoritative.
            local_cuts = []
            for index, rect in enumerate(self._attempt_cuts.get(atom.gi, ())):
                if index % 1024 == 0:
                    self._watchdog_check(net, "component_access_cut_pool")
                if _closed_rects_touch(_norm(rect), cut_query):
                    local_cuts.append(rect)
            attempt_cut_pool = tuple(local_cuts)
        atom_memo = getattr(self, "_transactional_atom_escape_memo", None)
        atom_memo_key = None
        if (
            isinstance(atom_memo, OrderedDict)
            and getattr(self, "_active_mirror_legality", None) is None
            and getattr(self, "_active_wide_repair_route", None) is None
            and not getattr(self, "_group_route_global_interfaces_by_net", {})
            and not getattr(self, "_layered_portal_wire_states", ())
        ):
            ext_m = self._via_cut_extent(atom.gi, model_n)
            vsp_m = int(self._via_space[atom.gi])
            c_reach = ext_m + max(vsp_m, ext_m + 1) + 2
            cut_idx_lo = self._cut_idx.get(atom.gi)
            comm_cuts_sig = (
                ()
                if cut_idx_lo is None
                else tuple(sorted(
                    ("" if o is None else str(o), tuple(_norm(r)))
                    for o, r in cut_idx_lo.query(
                        (cx0 - c_reach, cy0 - c_reach, cx1 + c_reach, cy1 + c_reach)
                    )
                ))
            )
            raw_att_cuts = (
                attempt_cut_pool
                if attempt_cut_pool is not None
                else tuple(
                    r
                    for r in self._attempt_cuts.get(atom.gi, ())
                    if _closed_rects_touch(_norm(r), cut_query)
                )
            )
            att_cuts_sig = tuple(sorted(tuple(_norm(r)) for r in raw_att_cuts))
            forb_sig = tuple(sorted(
                s
                for s in getattr(self, "_search_forbidden_via_sites", ())
                if len(s) >= 3
                and s[2] == atom.gi
                and cx0 <= s[0] <= cx1
                and cy0 <= s[1] <= cy1
            ))
            sp_partners = getattr(self, "_intent_spacing_rules", {}).get(net, {})
            max_intent_sp = max((int(v) for v in sp_partners.values()), default=0)
            wide_by_gi = self._wide_indexes(True)
            layer_sigs = []
            for l_gi, pad_h in ((atom.gi, int(bh)), (atom.gi + 1, int( _ah ))):
                pbox = (cx0 - pad_h, cy0 - pad_h, cx1 + pad_h, cy1 + pad_h)
                base_sp = int(self._space[l_gi]) if l_gi < len(self._space) else 0
                max_wide_sp = max(
                    (
                        int(b.spacing.value)
                        for b in self._wide_geometry_entries_by_gi.get(l_gi, ())
                    ),
                    default=0,
                )
                m_reach = max(base_sp, max_intent_sp, max_wide_sp) + 2
                midx = self._metal_idx.get(l_gi)
                vp_claims = getattr(self, "_via_pad_provenance", {}).get(
                    l_gi, frozenset()
                )
                m_rows = (
                    ()
                    if midx is None
                    else tuple(sorted(
                        (
                            "" if o is None else str(o),
                            tuple(_norm(r)),
                            (o, tuple(_norm(r))) in vp_claims,
                        )
                        for o, r in midx.query((
                            pbox[0] - m_reach,
                            pbox[1] - m_reach,
                            pbox[2] + m_reach,
                            pbox[3] + m_reach,
                        ))
                        if o != net
                    ))
                )
                w_rows = []
                for w_sp, core_idx, win_idx, att_idx in wide_by_gi.get(l_gi, ()):
                    wq = (
                        pbox[0] - w_sp,
                        pbox[1] - w_sp,
                        pbox[2] + w_sp,
                        pbox[3] + w_sp,
                    )
                    c_hit = tuple(sorted(
                        ("" if o is None else str(o), tuple(_norm(r)))
                        for o, r in core_idx.query(wq)
                        if o != net
                    ))
                    win_hit = tuple(sorted(
                        tuple(_norm(r)) for _o, r in win_idx.query(pbox)
                    ))
                    a_hit = (
                        tuple(sorted(
                            ("" if o is None else str(o), tuple(_norm(r)))
                            for o, r in att_idx.query(wq)
                            if o != net
                        ))
                        if win_hit
                        else ()
                    )
                    if c_hit or win_hit:
                        w_rows.append((int(w_sp), c_hit, win_hit, a_hit))
                layer_sigs.append((l_gi, m_rows, tuple(w_rows)))
            atom_memo_key = (
                _owned_terminal_atom_key(atom),
                str(net),
                1 if model_n == 1 else 0,
                int(self.via_redundancy),
                self._via_static_legality_key(0, 0, atom.gi, str(net), model_n)[5:],
                tuple(sorted(getattr(self, "_active_half_width_overrides", {}).items())),
                bool(getattr(self, "_active_group_route_local_closure", False)),
                bool(getattr(self, "_negotiation_port_legalize", False)),
                att_cuts_sig,
                forb_sig,
                comm_cuts_sig,
                tuple(layer_sigs),
            )
            cached_atom = atom_memo.pop(atom_memo_key, None)
            if cached_atom is not None:
                atom_memo[atom_memo_key] = cached_atom
                c_states, c_count, c_rej, c_samp, c_rej_total = cached_atom
                self._merge_component_access_rejections(c_rej, c_samp)
                if c_rej_total:
                    self._hard_rejection_generation = (
                        int(getattr(self, "_hard_rejection_generation", 0))
                        + c_rej_total
                    )
                return c_states, c_count
        partition_key = (
            _owned_terminal_atom_key(atom),
            str(net),
            1 if model_n == 1 else 0,
        )
        cached_partition = (
            None if legality_cache is None
            else legality_cache.lookup_partition(partition_key)
        )
        if cached_partition is not None:
            xs, ys, cached_thresholds = cached_partition
            threshold_candidates = set(cached_thresholds)
        else:
            base_xs, base_ys = self._owned_atom_via_event_coordinates(
                atom, net, model_n
            )
            min_width = int(self._tech.gi_min_width(atom.gi))
            cpp_throat = load_native("_gridless_cpp")
            if (
                cpp_throat is not None
                and "_watchdog_check" not in self.__dict__
                and "_on_manufacturing_grid_lines" not in self.__dict__
            ):
                self._watchdog_check(net, "component_access_throat_events")
                grid_t = max(1, int(getattr(self, "_manufacturing_grid_nm", 1)))
                xs, ys, sorted_thresholds = cpp_throat.compute_throat_events_cpp(
                    base_xs, base_ys, x0, y0, x1, y1, int(bh), min_width, grid_t
                )
                threshold_candidates = set(sorted_thresholds)
                if legality_cache is not None:
                    legality_cache.store_partition(partition_key, (
                        xs,
                        ys,
                        sorted_thresholds,
                    ))
            else:
                axis_work = 0

                def axis_events(
                    values: Sequence[int],
                    domain_lo: int,
                    domain_hi: int,
                    atom_lo: int,
                    atom_hi: int,
                ) -> Tuple[int, ...]:
                    nonlocal axis_work
                    # Touch boundaries plus the two slope/containment breakpoints of
                    # interval overlap.  +/-1 samples both integer sides of strict
                    # blocker and exact-conflict frontiers without a per-nm scan.
                    seeds = {
                        atom_lo - int(bh),
                        atom_lo + int(bh),
                        atom_hi - int(bh),
                        atom_hi + int(bh),
                    }
                    for value in values:
                        if axis_work % 1024 == 0:
                            self._watchdog_check(
                                net, "component_access_throat_events"
                            )
                        axis_work += 1
                        seeds.add(int(value))
                    expanded = set()
                    for seed in sorted(seeds):
                        if axis_work % 1024 == 0:
                            self._watchdog_check(
                                net, "component_access_throat_events"
                            )
                        axis_work += 1
                        for delta in (-1, 0, 1):
                            value = seed + delta
                            if domain_lo <= value <= domain_hi:
                                expanded.add(value)
                    return tuple(self._on_manufacturing_grid_lines(
                        expanded,
                        domain_lo,
                        domain_hi,
                        preserve_frontier_sides=True,
                    ))

                xs = axis_events(base_xs, cx0, cx1, x0, x1)
                ys = axis_events(base_ys, cy0, cy1, y0, y1)
                width_sq = min_width * min_width

                def overlap_at(centre: int, lo: int, hi: int) -> int:
                    return max(
                        0,
                        min(hi, centre + int(bh)) - max(lo, centre - int(bh)),
                    )

                def ceil_sqrt(value: int) -> int:
                    root = math.isqrt(max(0, int(value)))
                    return root if root * root == value else root + 1

                threshold_candidates: Set[Tuple[int, int]] = set()

                def add_threshold_candidate(x: int, y: int) -> None:
                    for grid_x in self._on_manufacturing_grid_lines(
                        (x,), cx0, cx1, preserve_frontier_sides=True
                    ):
                        for grid_y in self._on_manufacturing_grid_lines(
                            (y,), cy0, cy1, preserve_frontier_sides=True
                        ):
                            threshold_candidates.add((grid_x, grid_y))

                # For each representative on one axis, solve the exact minimum
                # overlap required on the other: ox^2 + oy^2 >= min_width^2.
                # Include the threshold and its integer neighbours so equality and
                # strict frontier semantics are both represented.
                for index, y in enumerate(ys):
                    if index % 1024 == 0:
                        self._watchdog_check(
                            net, "component_access_throat_events"
                        )
                    overlap_y = overlap_at(y, y0, y1)
                    required_x = ceil_sqrt(width_sq - overlap_y * overlap_y)
                    if required_x <= min(x1 - x0, 2 * int(bh)):
                        for boundary in (
                            x0 - int(bh) + required_x,
                            x1 + int(bh) - required_x,
                        ):
                            for delta in (-1, 0, 1):
                                add_threshold_candidate(boundary + delta, y)
                for index, x in enumerate(xs):
                    if index % 1024 == 0:
                        self._watchdog_check(
                            net, "component_access_throat_events"
                        )
                    overlap_x = overlap_at(x, x0, x1)
                    required_y = ceil_sqrt(width_sq - overlap_x * overlap_x)
                    if required_y <= min(y1 - y0, 2 * int(bh)):
                        for boundary in (
                            y0 - int(bh) + required_y,
                            y1 + int(bh) - required_y,
                        ):
                            for delta in (-1, 0, 1):
                                add_threshold_candidate(x, boundary + delta)
                if legality_cache is not None:
                    legality_cache.store_partition(partition_key, (
                        tuple(xs),
                        tuple(ys),
                        tuple(sorted(threshold_candidates)),
                    ))

        atom_min_width = int(self._tech.gi_min_width(atom.gi))
        bh_int = int(bh)
        ah_int = int(_ah)
        atom_rect = atom.rect
        model_flag = 1 if model_n == 1 else 0

        def _try_cpp_batch_module():
            cpp = load_native("_gridless_cpp")
            if cpp is None:
                return None
            inst_dict = self.__dict__
            for name in _BATCH_VIA_FORBIDDEN_OVERRIDES:
                if name in inst_dict:
                    return None
            if (
                _eg_attr("_rects_clear", _base_rects_clear) is not _base_rects_clear
                or _eg_attr("_sep", _base_sep) is not _base_sep
            ):
                return None
            if not bool(getattr(self, "_active_group_route_local_closure", False)):
                if (
                    getattr(self, "_group_route_global_interfaces_by_net", {}).get(str(net))
                    or getattr(self, "_layered_portal_wire_states", ())
                ):
                    return None
            env = getattr(self, "_routing_envelope", None)
            if env is not None and type(env).__module__ != "clay.layout.route.floorplan_containment":
                return None
            lo_idx = atom.gi
            obs_cut_map = getattr(self, "_obs_cut_idx", None)
            for idx in (
                self._obs_idx.get(lo_idx),
                self._obs_idx.get(lo_idx + 1),
                obs_cut_map.get(lo_idx) if obs_cut_map is not None else None,
                self._metal_idx.get(lo_idx),
                self._metal_idx.get(lo_idx + 1),
                self._cut_idx.get(lo_idx),
            ):
                if idx is not None and (type(idx) is not _SpatialIndex or idx._native is None):
                    return None
            return cpp

        def _merge_cpp_batch_rejections(agg_rejs, cpp) -> None:
            if not agg_rejs:
                return
            table = getattr(self, "_hard_rejections", None)
            if table is None:
                return
            samples = getattr(self, "_hard_rejection_samples", None)
            obs_prov = getattr(self, "_obs_provenance", {})
            cut_prov = getattr(self, "_obs_cut_provenance", {})
            vp_prov = getattr(self, "_via_pad_provenance", {})
            total_gen = 0
            c_conflict = cpp.REJ_CONFLICT_BRANCH_VIA
            c_grid = cpp.REJ_MANUFACTURING_GRID
            c_loc_env = cpp.REJ_GROUP_ROUTE_LOCAL_ENVELOPE
            c_env = cpp.REJ_ROUTING_ENVELOPE
            c_fixed_m = cpp.REJ_FIXED_METAL
            c_fixed_sn = cpp.REJ_FIXED_SAME_NET_VIA_PAD
            c_fixed_c = cpp.REJ_FIXED_CUT
            c_comm_mc = cpp.REJ_COMMITTED_METAL_CONTACT
            c_comm_ms = cpp.REJ_COMMITTED_METAL_SPACING
            c_comm_cc = cpp.REJ_COMMITTED_CUT_CONTACT
            c_comm_cs = cpp.REJ_COMMITTED_CUT_SPACING
            c_comm_sc = cpp.REJ_COMMITTED_SELF_CUT
            c_att_c = cpp.REJ_ATTEMPT_CUT
            for cat, owner, gi, chalf, req_sp, blk, cnt, extent in agg_rejs:
                tref = None
                bkind = None
                if cat == c_fixed_m:
                    category = "fixed_metal"
                    bkind, tref = obs_prov.get((gi, owner, blk), (None, None))
                elif cat == c_fixed_sn:
                    category = "fixed_same_net_via_pad_spacing"
                    bkind, tref = obs_prov.get((gi, owner, blk), (None, None))
                elif cat == c_fixed_c:
                    category = "fixed_cut"
                    owner, bkind, tref = cut_prov.get((gi, blk), (None, None, None))
                elif cat == c_comm_mc:
                    is_vp = (owner, blk) in vp_prov.get(gi, ())
                    category = "committed_via_pad_contact" if is_vp else "committed_metal_contact"
                    bkind = "via_pad" if is_vp else "route_metal"
                elif cat == c_comm_ms:
                    is_vp = (owner, blk) in vp_prov.get(gi, ())
                    category = "committed_via_pad_spacing" if is_vp else "committed_metal_spacing"
                    bkind = "via_pad" if is_vp else "route_metal"
                elif cat == c_comm_cc:
                    category = "committed_cut_contact"
                    bkind = "via_cut"
                elif cat == c_comm_cs:
                    category = "committed_cut_spacing"
                    bkind = "via_cut"
                elif cat == c_comm_sc:
                    category = "committed_self_cut"
                elif cat == c_att_c:
                    category = "attempt_cut"
                elif cat == c_env:
                    category = "routing_envelope"
                    bkind = "floorplan_boundary"
                elif cat == c_grid:
                    category = "manufacturing_grid"
                    bkind = "via_manufacturing_grid"
                elif cat == c_loc_env:
                    category = "group_route_local_envelope"
                    bkind = "certified_group_envelope"
                elif cat == c_conflict:
                    category = "conflict_branch_via"
                else:
                    continue
                owner_id = str(owner) if owner is not None else "<unowned>"
                rkey = (category, owner_id)
                table[rkey] = table.get(rkey, 0) + cnt
                total_gen += cnt
                if samples is not None:
                    cause = (
                        category,
                        owner_id,
                        int(gi),
                        int(chalf),
                        req_sp,
                        blk,
                        None if bkind is None else str(bkind),
                        None if tref is None else str(tref),
                    )
                    prev = samples.get(cause)
                    if prev is None:
                        samples[cause] = (cnt, extent)
                    else:
                        old_cnt, old_ext = prev
                        samples[cause] = (
                            old_cnt + cnt,
                            (
                                min(old_ext[0], extent[0]),
                                min(old_ext[1], extent[1]),
                                max(old_ext[2], extent[2]),
                                max(old_ext[3], extent[3]),
                            ),
                        )
            if total_gen:
                self._hard_rejection_generation = (
                    int(getattr(self, "_hard_rejection_generation", 0)) + total_gen
                )

        def _run_cpp_batch(cpp) -> None:
            nonlocal candidate_count
            lo_idx = atom.gi
            ext_i = int(self._via_cut_extent(lo_idx, model_n))
            sp_lo = int(self._space[lo_idx])
            sp_hi = int(self._space[lo_idx + 1])
            vsp_i = int(self._via_space[lo_idx])
            grid_i = max(1, int(getattr(self, "_manufacturing_grid_nm", 1)))
            pbox_lo = (cx0 - bh_int, cy0 - bh_int, cx1 + bh_int, cy1 + bh_int)
            pbox_hi = (cx0 - ah_int, cy0 - ah_int, cx1 + ah_int, cy1 + ah_int)
            cbox = (cx0 - ext_i, cy0 - ext_i, cx1 + ext_i, cy1 + ext_i)
            pbox_lo_q = (pbox_lo[0] - sp_lo, pbox_lo[1] - sp_lo, pbox_lo[2] + sp_lo, pbox_lo[3] + sp_lo)
            pbox_hi_q = (pbox_hi[0] - sp_hi, pbox_hi[1] - sp_hi, pbox_hi[2] + sp_hi, pbox_hi[3] + sp_hi)

            fh_lo: List[Rect] = []
            fh_hi: List[Rect] = []
            fh_cut: List[Rect] = []

            # Wide-metal fallback halos (empty on 99%+ of atoms)
            self._ensure_fixed_wide_geometry_indexes()
            wide_by_gi = self._wide_indexes(True)
            min_wide_map = getattr(self, "_min_wide_ge_by_gi", {})
            for l_gi, pad_h, pbox, fh_list in (
                (lo_idx, bh_int, pbox_lo, fh_lo),
                (lo_idx + 1, ah_int, pbox_hi, fh_hi),
            ):
                min_w = min_wide_map.get(l_gi)
                if min_w is not None:
                    if 2 * pad_h >= min_w:
                        fh_list.append(pbox)
                    else:
                        for w_sp, core_idx, _win_idx, att_idx in wide_by_gi.get(l_gi, ()):
                            wq = (pbox[0] - w_sp, pbox[1] - w_sp, pbox[2] + w_sp, pbox[3] + w_sp)
                            for w_owner, w_rect in core_idx.query(wq):
                                if w_owner != net:
                                    nr = _norm(w_rect)
                                    fh_list.append((nr[0] - w_sp, nr[1] - w_sp, nr[2] + w_sp, nr[3] + w_sp))
                            for w_owner, w_rect in att_idx.query(wq):
                                if w_owner != net:
                                    nr = _norm(w_rect)
                                    fh_list.append((nr[0] - w_sp, nr[1] - w_sp, nr[2] + w_sp, nr[3] + w_sp))

            # MIM fallback halos
            mim_bot_map = getattr(self, "_mim_bottom_routing_clearance_by_gi", {})
            mim_der_map = (
                getattr(self, "_mim_derived_unrelated_clearance_by_gi", {})
                if self._mim_derived_unrelated_static_active()
                else {}
            )
            mim_crv_map = getattr(self, "_mim_carve_bottom_blockers_by_gi", {})
            for l_gi, pbox, fh_list in (
                (lo_idx, pbox_lo, fh_lo),
                (lo_idx + 1, pbox_hi, fh_hi),
            ):
                for _mo, mblk, mclr in mim_bot_map.get(l_gi, ()):
                    mc = int(mclr)
                    nb = _norm(mblk)
                    g = (nb[0] - mc, nb[1] - mc, nb[2] + mc, nb[3] + mc)
                    if _closed_rects_touch(pbox, g):
                        fh_list.append(g)
                for mo, mblk, mclr, _mm, _mr in mim_der_map.get(l_gi, ()):
                    if mo == net:
                        continue
                    mc = int(mclr)
                    nb = _norm(mblk)
                    g = (nb[0] - mc, nb[1] - mc, nb[2] + mc, nb[3] + mc)
                    if _closed_rects_touch(pbox, g):
                        fh_list.append(g)
                for _mo, mblk, _mk in mim_crv_map.get(l_gi, ()):
                    nb = _norm(mblk)
                    if _closed_rects_touch(pbox, nb):
                        fh_list.append(nb)
            for _mo, mblk in getattr(self, "_mim_forbidden_cut_touch_by_lo", {}).get(lo_idx, ()):
                nb = _norm(mblk)
                if _closed_rects_touch(cbox, nb):
                    fh_cut.append(nb)
            for _mo, mblk, mclr in getattr(self, "_mim_top_via_marker_clearance_by_lo", {}).get(lo_idx, ()):
                mc = int(mclr)
                nb = _norm(mblk)
                g = (nb[0] - mc, nb[1] - mc, nb[2] + mc, nb[3] + mc)
                if _closed_rects_touch(cbox, g):
                    fh_cut.append(g)

            # Named-net intent spacing fallback halos
            partners = getattr(self, "_intent_spacing_rules", {}).get(net, {})
            if partners:
                margin = max(partners.values(), default=0)
                if margin > 0:
                    for l_gi, pbox, fh_list in (
                        (lo_idx, pbox_lo, fh_lo),
                        (lo_idx + 1, pbox_hi, fh_hi),
                    ):
                        iq = (pbox[0] - margin, pbox[1] - margin, pbox[2] + margin, pbox[3] + margin)
                        for idx in (self._obs_idx.get(l_gi), self._metal_idx.get(l_gi)):
                            if idx is None:
                                continue
                            for p_owner, p_rect in idx.query(iq):
                                dist = partners.get(p_owner, 0)
                                if dist:
                                    nr = _norm(p_rect)
                                    fh_list.append((nr[0] - dist, nr[1] - dist, nr[2] + dist, nr[3] + dist))

            obs_lo = self._obs_idx.get(lo_idx)
            obs_hi = self._obs_idx.get(lo_idx + 1)
            obs_cut_map = getattr(self, "_obs_cut_idx", None)
            obs_cut_lo = obs_cut_map.get(lo_idx) if obs_cut_map is not None else None
            fixed_cuts_fb = (
                None
                if obs_cut_lo is not None
                else [
                    _norm(r)
                    for r in getattr(self, "_obs_cuts", {}).get(lo_idx, ())
                    if _closed_rects_touch(_norm(r), cut_query)
                ]
            )
            metal_lo = self._metal_idx.get(lo_idx)
            metal_hi = self._metal_idx.get(lo_idx + 1)
            cut_lo = self._cut_idx.get(lo_idx)

            local_att_cuts = (
                attempt_cut_pool
                if attempt_cut_pool is not None
                else [
                    r
                    for r in self._attempt_cuts.get(lo_idx, ())
                    if _closed_rects_touch(_norm(r), cut_query)
                ]
            )
            att_m_lo = [
                r
                for r in self._attempt_metal.get(lo_idx, ())
                if _closed_rects_touch(_norm(r), pbox_lo_q)
            ]
            att_m_hi = [
                r
                for r in self._attempt_metal.get(lo_idx + 1, ())
                if _closed_rects_touch(_norm(r), pbox_hi_q)
            ]
            abs_lo = [
                r
                for r in self._abstract_rects.get(lo_idx, ())
                if _closed_rects_touch(_norm(r), pbox_lo_q)
            ]
            abs_hi = [
                r
                for r in self._abstract_rects.get(lo_idx + 1, ())
                if _closed_rects_touch(_norm(r), pbox_hi_q)
            ]

            env = getattr(self, "_routing_envelope", None)
            env_bbox = tuple(int(v) for v in env.bbox_nm) if env is not None else None
            local_env_bbox = None
            if bool(getattr(self, "_active_group_route_local_closure", False)):
                local_iface = getattr(self, "_active_group_route_local_interface", None)
                if local_iface is not None:
                    local_env_bbox = tuple(int(v) for v in local_iface[0])

            forb_sites = getattr(self, "_search_forbidden_via_sites", None)
            forb_xy = (
                [
                    (int(s[0]), int(s[1]))
                    for s in forb_sites
                    if len(s) >= 4 and s[2] == lo_idx and s[3] == model_flag
                ]
                if forb_sites
                else None
            )
            mirror = getattr(self, "_active_mirror_legality", None)
            mirror_active = mirror is not None and str(net) == str(mirror.net_a)

            extra_sorted = sorted(threshold_candidates) if threshold_candidates else None
            self._watchdog_check(net, "component_access_landings")

            # Chunk huge Cartesian grids so wall-clock watchdogs remain responsive
            ny = len(ys)
            chunk_cols = max(1, 100_000 // max(1, ny))
            all_fallbacks: List[Tuple[int, int]] = []
            xs_list = list(xs)
            n_xs = len(xs_list)
            if n_xs <= chunk_cols:
                legal_xy, fb_xy, cnt, agg_rejs = cpp.batch_filter_via_sites_cpp(
                    xs=xs_list,
                    ys=ys,
                    extra_pts=extra_sorted,
                    sites=None,
                    atom_rect=atom_rect,
                    atom_min_width=atom_min_width,
                    lo=lo_idx,
                    bh=bh_int,
                    ah=ah_int,
                    ext=ext_i,
                    sp_lo=sp_lo,
                    sp_hi=sp_hi,
                    vsp=vsp_i,
                    grid=grid_i,
                    net=str(net),
                    obs_lo=obs_lo._native if obs_lo is not None else None,
                    obs_hi=obs_hi._native if obs_hi is not None else None,
                    obs_cut_lo=obs_cut_lo._native if obs_cut_lo is not None else None,
                    fixed_cuts_fallback=fixed_cuts_fb,
                    metal_lo=metal_lo._native if metal_lo is not None else None,
                    metal_hi=metal_hi._native if metal_hi is not None else None,
                    cut_lo=cut_lo._native if cut_lo is not None else None,
                    attempt_cuts=local_att_cuts,
                    attempt_metal_lo=att_m_lo,
                    attempt_metal_hi=att_m_hi,
                    abstract_lo=abs_lo,
                    abstract_hi=abs_hi,
                    envelope_bbox=env_bbox,
                    local_envelope_bbox=local_env_bbox,
                    forbidden_xy=forb_xy,
                    fallback_halos_lo=fh_lo or None,
                    fallback_halos_hi=fh_hi or None,
                    fallback_halos_cut=fh_cut or None,
                    mirror_active=mirror_active,
                )
                candidate_count += cnt
                _merge_cpp_batch_rejections(agg_rejs, cpp)
                for lx, ly in legal_xy:
                    states.append(((lx, ly, lo_idx + 1), _ViaSite(lx, ly, lo_idx, model_flag)))
                if fb_xy:
                    all_fallbacks.extend(fb_xy)
            else:
                for start in range(0, n_xs, chunk_cols):
                    if start:
                        self._watchdog_check(net, "component_access_landings")
                    sub_xs = xs_list[start : start + chunk_cols]
                    legal_xy, fb_xy, cnt, agg_rejs = cpp.batch_filter_via_sites_cpp(
                        xs=sub_xs,
                        ys=ys,
                        extra_pts=None,
                        sites=None,
                        atom_rect=atom_rect,
                        atom_min_width=atom_min_width,
                        lo=lo_idx,
                        bh=bh_int,
                        ah=ah_int,
                        ext=ext_i,
                        sp_lo=sp_lo,
                        sp_hi=sp_hi,
                        vsp=vsp_i,
                        grid=grid_i,
                        net=str(net),
                        obs_lo=obs_lo._native if obs_lo is not None else None,
                        obs_hi=obs_hi._native if obs_hi is not None else None,
                        obs_cut_lo=obs_cut_lo._native if obs_cut_lo is not None else None,
                        fixed_cuts_fallback=fixed_cuts_fb,
                        metal_lo=metal_lo._native if metal_lo is not None else None,
                        metal_hi=metal_hi._native if metal_hi is not None else None,
                        cut_lo=cut_lo._native if cut_lo is not None else None,
                        attempt_cuts=local_att_cuts,
                        attempt_metal_lo=att_m_lo,
                        attempt_metal_hi=att_m_hi,
                        abstract_lo=abs_lo,
                        abstract_hi=abs_hi,
                        envelope_bbox=env_bbox,
                        local_envelope_bbox=local_env_bbox,
                        forbidden_xy=forb_xy,
                        fallback_halos_lo=fh_lo or None,
                        fallback_halos_hi=fh_hi or None,
                        fallback_halos_cut=fh_cut or None,
                        mirror_active=mirror_active,
                    )
                    candidate_count += cnt
                    _merge_cpp_batch_rejections(agg_rejs, cpp)
                    for lx, ly in legal_xy:
                        states.append(((lx, ly, lo_idx + 1), _ViaSite(lx, ly, lo_idx, model_flag)))
                    if fb_xy:
                        all_fallbacks.extend(fb_xy)
                if extra_sorted:
                    xs_set_local = set(xs_list)
                    ys_set_local = set(ys)
                    rem_extra = [
                        (ex, ey)
                        for ex, ey in extra_sorted
                        if not (ex in xs_set_local and ey in ys_set_local)
                    ]
                    if rem_extra:
                        self._watchdog_check(net, "component_access_landings")
                        legal_xy, fb_xy, cnt, agg_rejs = cpp.batch_filter_via_sites_cpp(
                            xs=None,
                            ys=None,
                            extra_pts=None,
                            sites=rem_extra,
                            atom_rect=atom_rect,
                            atom_min_width=atom_min_width,
                            lo=lo_idx,
                            bh=bh_int,
                            ah=ah_int,
                            ext=ext_i,
                            sp_lo=sp_lo,
                            sp_hi=sp_hi,
                            vsp=vsp_i,
                            grid=grid_i,
                            net=str(net),
                            obs_lo=obs_lo._native if obs_lo is not None else None,
                            obs_hi=obs_hi._native if obs_hi is not None else None,
                            obs_cut_lo=obs_cut_lo._native if obs_cut_lo is not None else None,
                            fixed_cuts_fallback=fixed_cuts_fb,
                            metal_lo=metal_lo._native if metal_lo is not None else None,
                            metal_hi=metal_hi._native if metal_hi is not None else None,
                            cut_lo=cut_lo._native if cut_lo is not None else None,
                            attempt_cuts=local_att_cuts,
                            attempt_metal_lo=att_m_lo,
                            attempt_metal_hi=att_m_hi,
                            abstract_lo=abs_lo,
                            abstract_hi=abs_hi,
                            envelope_bbox=env_bbox,
                            local_envelope_bbox=local_env_bbox,
                            forbidden_xy=forb_xy,
                            fallback_halos_lo=fh_lo or None,
                            fallback_halos_hi=fh_hi or None,
                            fallback_halos_cut=fh_cut or None,
                            mirror_active=mirror_active,
                        )
                        candidate_count += cnt
                        _merge_cpp_batch_rejections(agg_rejs, cpp)
                        for lx, ly in legal_xy:
                            states.append(((lx, ly, lo_idx + 1), _ViaSite(lx, ly, lo_idx, model_flag)))
                        if fb_xy:
                            all_fallbacks.extend(fb_xy)

            for fb_i, (fx, fy) in enumerate(all_fallbacks):
                if fb_i % 1024 == 0:
                    self._watchdog_check(net, "component_access_landings")
                legal = (
                    self._via_legal(fx, fy, lo_idx, net, n=model_n)
                    if legality_cache is None
                    else self._cached_component_access_via_legal(
                        fx,
                        fy,
                        lo_idx,
                        net,
                        model_n,
                        legality_cache,
                        attempt_cut_pool=attempt_cut_pool,
                    )
                )
                if legal:
                    states.append(((fx, fy, lo_idx + 1), _ViaSite(fx, fy, lo_idx, model_flag)))
            self._watchdog_check(net, "component_access_landings")

        def consider(x: int, y: int) -> None:
            nonlocal candidate_count
            if candidate_count % 1024 == 0:
                self._watchdog_check(net, "component_access_landings")
            candidate_count += 1
            lower_pad = (
                x - bh_int,
                y - bh_int,
                x + bh_int,
                y + bh_int,
            )
            if not _closed_rects_touch(lower_pad, atom_rect):
                return
            if not self._owned_landing_has_minimum_throat(
                atom_rect,
                lower_pad,
                atom_min_width,
            ):
                return
            legal = (
                self._via_legal(x, y, atom.gi, net, n=model_n)
                if legality_cache is None
                else self._cached_component_access_via_legal(
                    x,
                    y,
                    atom.gi,
                    net,
                    model_n,
                    legality_cache,
                    attempt_cut_pool=attempt_cut_pool,
                )
            )
            if not legal:
                return
            site = _ViaSite(
                x,
                y,
                atom.gi,
                model_flag,
            )
            states.append(((x, y, atom.gi + 1), site))

        cpp_batch = _try_cpp_batch_module()

        # Stream the potentially large Cartesian base domain so the typed
        # watchdog runs before memory or sort work can scale with |xs|*|ys|.
        if atom_memo_key is not None and isinstance(atom_memo, OrderedDict):
            outer_atom_rej = getattr(self, "_hard_rejections", {})
            outer_atom_samp = getattr(self, "_hard_rejection_samples", {})
            outer_atom_gen = int(getattr(self, "_hard_rejection_generation", 0))
            self._hard_rejections = {}
            self._hard_rejection_samples = {}
            atom_completed = False
            try:
                if cpp_batch is not None:
                    _run_cpp_batch(cpp_batch)
                else:
                    for x in xs:
                        for y in ys:
                            consider(x, y)
                    xs_set = set(xs)
                    ys_set = set(ys)
                    for x, y in sorted(threshold_candidates):
                        if x in xs_set and y in ys_set:
                            continue  # already visited by the streamed Cartesian domain
                        consider(x, y)
                atom_completed = True
            finally:
                atom_rej = tuple(sorted(
                    (cause, int(cnt))
                    for cause, cnt in self._hard_rejections.items()
                ))
                atom_samp = tuple(
                    (cause, int(cnt), tuple(ext))
                    for cause, (cnt, ext) in sorted(
                        self._hard_rejection_samples.items()
                    )
                )
                atom_rej_total = sum(int(cnt) for _c, cnt in atom_rej)
                self._hard_rejections = outer_atom_rej
                self._hard_rejection_samples = outer_atom_samp
                self._hard_rejection_generation = outer_atom_gen + atom_rej_total
                self._merge_component_access_rejections(atom_rej, atom_samp)
            unique = {self._access_state_key(item): item for item in states}
            ordered_states = tuple(unique[key] for key in sorted(unique))
            if atom_completed:
                atom_memo[atom_memo_key] = (
                    ordered_states,
                    candidate_count,
                    atom_rej,
                    atom_samp,
                    atom_rej_total,
                )
                while len(atom_memo) > 4096:
                    atom_memo.popitem(last=False)
            return ordered_states, candidate_count
        if cpp_batch is not None:
            _run_cpp_batch(cpp_batch)
        else:
            for x in xs:
                for y in ys:
                    consider(x, y)
            xs_set = set(xs)
            ys_set = set(ys)
            for x, y in sorted(threshold_candidates):
                if x in xs_set and y in ys_set:
                    continue  # already visited by the streamed Cartesian domain
                consider(x, y)
        unique = {self._access_state_key(item): item for item in states}
        return tuple(unique[key] for key in sorted(unique)), candidate_count

    def _record_component_escape_diagnostic(
        self,
        *,
        status: str,
        component: Sequence[_OwnedTerminalAtom] = (),
        anchor_count: int = 0,
        legal_count: int = 0,
    ) -> None:
        """Attach stable component-access facts to an existing typed STOP context."""
        component = tuple(component)
        if component:
            bbox = (
                min(atom.rect[0] for atom in component),
                min(atom.rect[1] for atom in component),
                max(atom.rect[2] for atom in component),
                max(atom.rect[3] for atom in component),
            )
        else:
            bbox = ()
        refs = tuple(sorted({
            atom.terminal_conductor_ref
            for atom in component
            if type(atom.terminal_conductor_ref) is str
            and atom.terminal_conductor_ref
        }))
        updates = {
            "component_escape_status": str(status),
            "component_escape_terminal_refs": refs,
            "component_escape_bbox_nm": bbox,
            "component_escape_atom_count": len(component),
            "component_escape_anchor_count": int(anchor_count),
            "component_escape_legal_count": int(legal_count),
        }
        details = dict(getattr(self, "_route_failure_details", ()))
        details.update(updates)
        self._route_failure_details = tuple(sorted(details.items()))

    def _component_escape_state_world_key(
        self,
        net: str,
        px: int,
        py: int,
        gi: int,
    ) -> EscapeWorldKey:
        """Exact dynamic authority for one solve-local component state domain."""
        attempt_cuts = tuple(
            (
                int(lo),
                tuple(sorted(tuple(_norm(rect)) for rect in rects)),
            )
            for lo, rects in sorted(self._attempt_cuts.items())
        )
        attempt_metal = tuple(
            (
                int(layer_gi),
                tuple(sorted(tuple(_norm(rect)) for rect in rects)),
            )
            for layer_gi, rects in sorted(self._attempt_metal.items())
        )
        return EscapeWorldKey(
            str(net),
            int(px),
            int(py),
            int(gi),
            id(self._owned_terminal_atoms),
            int(getattr(self, "_committed_index_generation", 0)),
            attempt_cuts,
            attempt_metal,
            tuple(sorted(getattr(
                self, "_search_forbidden_via_sites", set()
            ))),
            int(getattr(self, "_history_generation", 0)),
            float(getattr(self, "_pres_fac", 0.0)),
        )

    def _replay_component_escape_state_snapshot(
        self,
        snapshot: EscapeProof,
    ) -> List[Tuple[Tuple[int, int, int], Optional[_ViaSite]]]:
        states, rejections, samples, component_details = snapshot
        self._merge_component_access_rejections(rejections, samples)
        details = dict(getattr(self, "_route_failure_details", ()))
        details.update(component_details)
        self._route_failure_details = tuple(sorted(details.items()))
        return list(states)

    def _owned_component_escape_states(
        self,
        net: str,
        px: int,
        py: int,
        gi: int,
    ) -> List[Tuple[Tuple[int, int, int], Optional[_ViaSite]]]:
        """Replay one complete access proof inside an unchanged rail-drop world."""
        cache = self._component_access
        if not cache.escape_scope_active:
            return self._compute_owned_component_escape_states(net, px, py, gi)
        key = self._component_escape_state_world_key(net, px, py, gi)
        cached = cache.lookup_escape(key)
        if cached is not None:
            return self._replay_component_escape_state_snapshot(cached)

        outer_rejections = getattr(self, "_hard_rejections", {})
        outer_samples = getattr(self, "_hard_rejection_samples", {})
        outer_details = tuple(getattr(self, "_route_failure_details", ()))
        self._hard_rejections = {}
        self._hard_rejection_samples = {}
        self._route_failure_details = ()
        try:
            states = self._compute_owned_component_escape_states(
                net, px, py, gi
            )
            rejections = tuple(sorted(
                (cause, int(count))
                for cause, count in self._hard_rejections.items()
            ))
            samples = tuple(
                (cause, int(count), tuple(extent))
                for cause, (count, extent) in sorted(
                    self._hard_rejection_samples.items()
                )
            )
            component_details = tuple(
                (name, value)
                for name, value in self._route_failure_details
                if name.startswith("component_escape_")
            )
        finally:
            self._hard_rejections = outer_rejections
            self._hard_rejection_samples = outer_samples
            self._route_failure_details = outer_details
        snapshot = EscapeProof(
            tuple(states),
            rejections,
            samples,
            component_details,
        )
        cache.store_escape(key, snapshot)
        return self._replay_component_escape_state_snapshot(snapshot)

    def _merge_cpp_via_batch_rejections(self, agg_rejs, cpp) -> None:
        if not agg_rejs:
            return
        table = getattr(self, "_hard_rejections", None)
        if table is None:
            return
        samples = getattr(self, "_hard_rejection_samples", None)
        obs_prov = getattr(self, "_obs_provenance", {})
        cut_prov = getattr(self, "_obs_cut_provenance", {})
        vp_prov = getattr(self, "_via_pad_provenance", {})
        total_gen = 0
        c_conflict = cpp.REJ_CONFLICT_BRANCH_VIA
        c_grid = cpp.REJ_MANUFACTURING_GRID
        c_loc_env = cpp.REJ_GROUP_ROUTE_LOCAL_ENVELOPE
        c_env = cpp.REJ_ROUTING_ENVELOPE
        c_fixed_m = cpp.REJ_FIXED_METAL
        c_fixed_sn = cpp.REJ_FIXED_SAME_NET_VIA_PAD
        c_fixed_c = cpp.REJ_FIXED_CUT
        c_comm_mc = cpp.REJ_COMMITTED_METAL_CONTACT
        c_comm_ms = cpp.REJ_COMMITTED_METAL_SPACING
        c_comm_cc = cpp.REJ_COMMITTED_CUT_CONTACT
        c_comm_cs = cpp.REJ_COMMITTED_CUT_SPACING
        c_comm_sc = cpp.REJ_COMMITTED_SELF_CUT
        c_att_c = cpp.REJ_ATTEMPT_CUT
        c_mim_bot = getattr(cpp, "REJ_MIM_BOTTOM_ROUTING", -1)
        c_mim_der = getattr(cpp, "REJ_MIM_DERIVED_UNRELATED", -1)
        c_mim_crv = getattr(cpp, "REJ_MIM_CARVE_BOTTOM", -1)
        c_mim_fct = getattr(cpp, "REJ_MIM_FORBIDDEN_CUT_TOUCH", -1)
        c_mim_tvm = getattr(cpp, "REJ_MIM_TOP_VIA_MARKER", -1)
        c_mir_m = getattr(cpp, "REJ_MIRROR_PAIR_METAL", -1)
        c_mir_c = getattr(cpp, "REJ_MIRROR_PAIR_CUT", -1)
        for rec in agg_rejs:
            if len(rec) == 9:
                cat, owner, gi, chalf, req_sp, blk, cnt, extent, extra_kind = rec
            else:
                cat, owner, gi, chalf, req_sp, blk, cnt, extent = rec
                extra_kind = None
            tref = None
            bkind = None
            if cat == c_fixed_m:
                category = "fixed_metal"
                bkind, tref = obs_prov.get((gi, owner, blk), (None, None))
            elif cat == c_fixed_sn:
                category = "fixed_same_net_via_pad_spacing"
                bkind, tref = obs_prov.get((gi, owner, blk), (None, None))
            elif cat == c_fixed_c:
                category = "fixed_cut"
                owner, bkind, tref = cut_prov.get((gi, blk), (None, None, None))
            elif cat == c_comm_mc:
                is_vp = (owner, blk) in vp_prov.get(gi, ())
                category = "committed_via_pad_contact" if is_vp else "committed_metal_contact"
                bkind = "via_pad" if is_vp else "route_metal"
            elif cat == c_comm_ms:
                is_vp = (owner, blk) in vp_prov.get(gi, ())
                category = "committed_via_pad_spacing" if is_vp else "committed_metal_spacing"
                bkind = "via_pad" if is_vp else "route_metal"
            elif cat == c_comm_cc:
                category = "committed_cut_contact"
                bkind = "via_cut"
            elif cat == c_comm_cs:
                category = "committed_cut_spacing"
                bkind = "via_cut"
            elif cat == c_comm_sc:
                category = "committed_self_cut"
            elif cat == c_att_c:
                category = "attempt_cut"
            elif cat == c_env:
                category = "routing_envelope"
                bkind = "floorplan_boundary"
            elif cat == c_grid:
                category = "manufacturing_grid"
                bkind = "via_manufacturing_grid"
            elif cat == c_loc_env:
                category = "group_route_local_envelope"
                bkind = "certified_group_envelope"
            elif cat == c_conflict:
                category = "conflict_branch_via"
            elif cat == c_mim_bot:
                category = "mim_bottom_routing_clearance"
                bkind = "mim_bottom_component"
            elif cat == c_mim_der:
                category = "mim_metal_clearance"
                bkind = extra_kind
            elif cat == c_mim_crv:
                category = "mim_carve_bottom_interaction"
                bkind = extra_kind
            elif cat == c_mim_fct:
                category = "mim_forbidden_cut_touch"
                bkind = "mim_bottom_component"
            elif cat == c_mim_tvm:
                category = "mim_top_via_marker_clearance"
                bkind = "mim_top_marker"
            elif cat == c_mir_m:
                category = "mirror_pair_in_transaction_conflict"
                bkind = "mirror_primary_attempt_metal"
            elif cat == c_mir_c:
                category = "mirror_pair_in_transaction_conflict"
                bkind = "mirror_primary_attempt_cut"
            else:
                continue
            owner_id = str(owner) if owner is not None else "<unowned>"
            rkey = (category, owner_id)
            table[rkey] = table.get(rkey, 0) + cnt
            total_gen += cnt
            if samples is not None:
                cause = (
                    category,
                    owner_id,
                    int(gi),
                    int(chalf),
                    req_sp,
                    blk,
                    None if bkind is None else str(bkind),
                    None if tref is None else str(tref),
                )
                prev = samples.get(cause)
                if prev is None:
                    samples[cause] = (cnt, extent)
                else:
                    old_cnt, old_ext = prev
                    samples[cause] = (
                        old_cnt + cnt,
                        (
                            min(old_ext[0], extent[0]),
                            min(old_ext[1], extent[1]),
                            max(old_ext[2], extent[2]),
                            max(old_ext[3], extent[3]),
                        ),
                    )
        if total_gen:
            self._hard_rejection_generation = (
                int(getattr(self, "_hard_rejection_generation", 0)) + total_gen
            )

    def _can_use_cpp_batch_congestion(self) -> Optional[Any]:
        cpp = load_native("_gridless_cpp")
        if cpp is None:
            return None
        inst_dict = self.__dict__
        for name in (
            "_via_congestion",
            "_box_congestion",
            "_via_cut_congestion",
            "_history_congestion_units",
            "_scan_committed_metal",
            "_scan_committed_cuts",
            "_static_or_direct_committed_metal_scan",
            "_static_or_direct_committed_cut_scan",
            "_wide_spacing_neighbors",
            "_wide_indexes",
        ):
            if name in inst_dict:
                return None
        if (
            _eg_attr("_rects_clear", _base_rects_clear) is not _base_rects_clear
            or _eg_attr("_sep", _base_sep) is not _base_sep
        ):
            return None
        return cpp

    def _batch_via_congestion_for_sites(
        self,
        sites_xy: Sequence[Tuple[int, int]],
        lo: int,
        net: str,
        model_n: Optional[int],
        cpp: Any,
        hist_by_rgi: Dict[int, List[Tuple[int, int, int]]],
    ) -> List[int]:
        if not sites_xy:
            return []
        metal_lo = self._metal_idx.get(lo)
        metal_hi = self._metal_idx.get(lo + 1)
        obs_lo = self._obs_idx.get(lo)
        obs_hi = self._obs_idx.get(lo + 1)
        att_lo = self._attempt_idx.get(lo)
        att_hi = self._attempt_idx.get(lo + 1)
        cut_lo = self._cut_idx.get(lo)
        for idx in (metal_lo, metal_hi, obs_lo, obs_hi, att_lo, att_hi, cut_lo):
            if idx is not None and (type(idx) is not _SpatialIndex or idx._native is None):
                return [self._via_congestion(x, y, lo, net, model_n) for x, y in sites_xy]

        bh, ah = self._via_pad_half1[lo] if model_n == 1 else self._via_pad_half[lo]
        bh_i, ah_i = int(bh), int(ah)
        ext_i = int(self._via_cut_extent(lo, model_n))
        sp_lo = int(self._space[lo])
        sp_hi = int(self._space[lo + 1])
        vsp_i = int(self._via_space[lo])
        rc_i = int(self._res_cell)
        pres_i = int(round(self._pres_fac * _COST_FACTOR_SCALE))
        mul_scale = rc_i * _STEP_COST_PER_NM * _CONG_WEIGHT

        xs_min = min(p[0] for p in sites_xy)
        xs_max = max(p[0] for p in sites_xy)
        ys_min = min(p[1] for p in sites_xy)
        ys_max = max(p[1] for p in sites_xy)
        pbox_lo = (xs_min - bh_i, ys_min - bh_i, xs_max + bh_i, ys_max + bh_i)
        pbox_hi = (xs_min - ah_i, ys_min - ah_i, xs_max + ah_i, ys_max + ah_i)

        # Wide-metal halos for congestion (ALL owners, since _box_congestion does not filter owner)
        self._ensure_fixed_wide_geometry_indexes()
        wide_by_gi = self._wide_indexes(True)
        min_wide_map = getattr(self, "_min_wide_ge_by_gi", {})
        wh_lo: List[Rect] = []
        wh_hi: List[Rect] = []
        for l_gi, pad_h, pbox, wh_list in (
            (lo, bh_i, pbox_lo, wh_lo),
            (lo + 1, ah_i, pbox_hi, wh_hi),
        ):
            min_w = min_wide_map.get(l_gi)
            if min_w is not None:
                if 2 * pad_h >= min_w:
                    wh_list.append(pbox)
                else:
                    for w_sp, core_idx, _win_idx, att_idx in wide_by_gi.get(l_gi, ()):
                        wq = (pbox[0] - w_sp, pbox[1] - w_sp, pbox[2] + w_sp, pbox[3] + w_sp)
                        for _wo, w_rect in core_idx.query(wq):
                            nr = _norm(w_rect)
                            wh_list.append((nr[0] - w_sp, nr[1] - w_sp, nr[2] + w_sp, nr[3] + w_sp))
                        for _wo, w_rect in att_idx.query(wq):
                            nr = _norm(w_rect)
                            wh_list.append((nr[0] - w_sp, nr[1] - w_sp, nr[2] + w_sp, nr[3] + w_sp))

        vals = cpp.batch_via_congestion_cpp(
            sites=sites_xy,
            bh=bh_i,
            ah=ah_i,
            ext=ext_i,
            sp_lo=sp_lo,
            sp_hi=sp_hi,
            vsp=vsp_i,
            rc=rc_i,
            pres=pres_i,
            mul_scale=mul_scale,
            cost_factor_scale=_COST_FACTOR_SCALE,
            net=str(net),
            metal_lo=metal_lo._native if metal_lo is not None else None,
            metal_hi=metal_hi._native if metal_hi is not None else None,
            obs_lo=obs_lo._native if obs_lo is not None else None,
            obs_hi=obs_hi._native if obs_hi is not None else None,
            attempt_lo=att_lo._native if att_lo is not None else None,
            attempt_hi=att_hi._native if att_hi is not None else None,
            cut_lo=cut_lo._native if cut_lo is not None else None,
            hist_lo=hist_by_rgi.get(lo),
            hist_hi=hist_by_rgi.get(lo + 1),
            hist_cut=hist_by_rgi.get(self._n_layers + lo),
            wide_halos_lo=wh_lo or None,
            wide_halos_hi=wh_hi or None,
        )
        for idx_i, val in enumerate(vals):
            if val < 0:
                sx, sy = sites_xy[idx_i]
                vals[idx_i] = self._via_congestion(sx, sy, lo, net, model_n)
        return vals

    def _batch_via_legal_for_sites_cpp(
        self,
        sites_xy: Sequence[Tuple[int, int]],
        lo: int,
        net: str,
        model_n: Optional[int],
        cpp: Any,
    ) -> Optional[Set[Tuple[int, int]]]:
        """Batch-check exact _via_legal(x, y, lo, net, n=model_n) with full rejection evidence."""
        if not sites_xy:
            return set()
        inst_dict = self.__dict__
        for name in _BATCH_VIA_FORBIDDEN_OVERRIDES:
            if name in inst_dict:
                return None
        if (
            _eg_attr("_rects_clear", _base_rects_clear) is not _base_rects_clear
            or _eg_attr("_sep", _base_sep) is not _base_sep
        ):
            return None
        if not bool(getattr(self, "_active_group_route_local_closure", False)):
            if (
                getattr(self, "_group_route_global_interfaces_by_net", {}).get(str(net))
                or getattr(self, "_layered_portal_wire_states", ())
            ):
                return None
        env = getattr(self, "_routing_envelope", None)
        if env is not None and type(env).__module__ != "clay.layout.route.floorplan_containment":
            return None
        obs_lo = self._obs_idx.get(lo)
        obs_hi = self._obs_idx.get(lo + 1)
        obs_cut_map = getattr(self, "_obs_cut_idx", None)
        obs_cut_lo = obs_cut_map.get(lo) if obs_cut_map is not None else None
        metal_lo = self._metal_idx.get(lo)
        metal_hi = self._metal_idx.get(lo + 1)
        cut_lo = self._cut_idx.get(lo)
        for idx in (obs_lo, obs_hi, obs_cut_lo, metal_lo, metal_hi, cut_lo):
            if idx is not None and (type(idx) is not _SpatialIndex or idx._native is None):
                return None

        bh, ah = self._via_pad_half1[lo] if model_n == 1 else self._via_pad_half[lo]
        bh_i, ah_i = int(bh), int(ah)
        ext_i = int(self._via_cut_extent(lo, model_n))
        sp_lo = int(self._space[lo])
        sp_hi = int(self._space[lo + 1])
        vsp_i = int(self._via_space[lo])
        grid_i = max(1, int(getattr(self, "_manufacturing_grid_nm", 1)))
        model_flag = 1 if model_n == 1 else 0

        xs_min = min(p[0] for p in sites_xy)
        xs_max = max(p[0] for p in sites_xy)
        ys_min = min(p[1] for p in sites_xy)
        ys_max = max(p[1] for p in sites_xy)
        pbox_lo = (xs_min - bh_i, ys_min - bh_i, xs_max + bh_i, ys_max + bh_i)
        pbox_hi = (xs_min - ah_i, ys_min - ah_i, xs_max + ah_i, ys_max + ah_i)
        cbox = (xs_min - ext_i, ys_min - ext_i, xs_max + ext_i, ys_max + ext_i)
        pbox_lo_q = (pbox_lo[0] - sp_lo, pbox_lo[1] - sp_lo, pbox_lo[2] + sp_lo, pbox_lo[3] + sp_lo)
        pbox_hi_q = (pbox_hi[0] - sp_hi, pbox_hi[1] - sp_hi, pbox_hi[2] + sp_hi, pbox_hi[3] + sp_hi)
        cut_q = (cbox[0] - vsp_i, cbox[1] - vsp_i, cbox[2] + vsp_i, cbox[3] + vsp_i)

        fh_lo: List[Rect] = []
        fh_hi: List[Rect] = []
        fh_cut: List[Rect] = []
        self._ensure_fixed_wide_geometry_indexes()
        wide_by_gi = self._wide_indexes(True)
        min_wide_map = getattr(self, "_min_wide_ge_by_gi", {})
        for l_gi, pad_h, pbox, fh_list in (
            (lo, bh_i, pbox_lo, fh_lo),
            (lo + 1, ah_i, pbox_hi, fh_hi),
        ):
            min_w = min_wide_map.get(l_gi)
            if min_w is not None:
                if 2 * pad_h >= min_w:
                    fh_list.append(pbox)
                else:
                    for w_sp, core_idx, _win_idx, att_idx in wide_by_gi.get(l_gi, ()):
                        wq = (pbox[0] - w_sp, pbox[1] - w_sp, pbox[2] + w_sp, pbox[3] + w_sp)
                        for w_owner, w_rect in core_idx.query(wq):
                            if w_owner != net:
                                nr = _norm(w_rect)
                                fh_list.append((nr[0] - w_sp, nr[1] - w_sp, nr[2] + w_sp, nr[3] + w_sp))
                        for w_owner, w_rect in att_idx.query(wq):
                            if w_owner != net:
                                nr = _norm(w_rect)
                                fh_list.append((nr[0] - w_sp, nr[1] - w_sp, nr[2] + w_sp, nr[3] + w_sp))

        mim_bot_map = getattr(self, "_mim_bottom_routing_clearance_by_gi", {})
        mim_der_map = (
            getattr(self, "_mim_derived_unrelated_clearance_by_gi", {})
            if self._mim_derived_unrelated_static_active()
            else {}
        )
        mim_crv_map = getattr(self, "_mim_carve_bottom_blockers_by_gi", {})
        for l_gi, pbox, fh_list in ((lo, pbox_lo, fh_lo), (lo + 1, pbox_hi, fh_hi)):
            for _mo, mblk, mclr in mim_bot_map.get(l_gi, ()):
                mc = int(mclr)
                nb = _norm(mblk)
                g = (nb[0] - mc, nb[1] - mc, nb[2] + mc, nb[3] + mc)
                if _closed_rects_touch(pbox, g):
                    fh_list.append(g)
            for mo, mblk, mclr, _mm, _mr in mim_der_map.get(l_gi, ()):
                if mo == net:
                    continue
                mc = int(mclr)
                nb = _norm(mblk)
                g = (nb[0] - mc, nb[1] - mc, nb[2] + mc, nb[3] + mc)
                if _closed_rects_touch(pbox, g):
                    fh_list.append(g)
            for _mo, mblk, _mk in mim_crv_map.get(l_gi, ()):
                nb = _norm(mblk)
                if _closed_rects_touch(pbox, nb):
                    fh_list.append(nb)
        for _mo, mblk in getattr(self, "_mim_forbidden_cut_touch_by_lo", {}).get(lo, ()):
            nb = _norm(mblk)
            if _closed_rects_touch(cbox, nb):
                fh_cut.append(nb)
        for _mo, mblk, mclr in getattr(self, "_mim_top_via_marker_clearance_by_lo", {}).get(lo, ()):
            mc = int(mclr)
            nb = _norm(mblk)
            g = (nb[0] - mc, nb[1] - mc, nb[2] + mc, nb[3] + mc)
            if _closed_rects_touch(cbox, g):
                fh_cut.append(g)

        partners = getattr(self, "_intent_spacing_rules", {}).get(net, {})
        if partners:
            margin = max(partners.values(), default=0)
            if margin > 0:
                for l_gi, pbox, fh_list in ((lo, pbox_lo, fh_lo), (lo + 1, pbox_hi, fh_hi)):
                    iq = (pbox[0] - margin, pbox[1] - margin, pbox[2] + margin, pbox[3] + margin)
                    for idx in (self._obs_idx.get(l_gi), self._metal_idx.get(l_gi)):
                        if idx is None:
                            continue
                        for p_owner, p_rect in idx.query(iq):
                            dist = partners.get(p_owner, 0)
                            if dist:
                                nr = _norm(p_rect)
                                fh_list.append((nr[0] - dist, nr[1] - dist, nr[2] + dist, nr[3] + dist))

        fixed_cuts_fb = (
            None
            if obs_cut_lo is not None
            else [
                _norm(r)
                for r in getattr(self, "_obs_cuts", {}).get(lo, ())
                if _closed_rects_touch(_norm(r), cut_q)
            ]
        )
        local_att_cuts = [
            r for r in self._attempt_cuts.get(lo, ())
            if _closed_rects_touch(_norm(r), cut_q)
        ]
        att_m_lo = [
            r for r in self._attempt_metal.get(lo, ())
            if _closed_rects_touch(_norm(r), pbox_lo_q)
        ]
        att_m_hi = [
            r for r in self._attempt_metal.get(lo + 1, ())
            if _closed_rects_touch(_norm(r), pbox_hi_q)
        ]
        abs_lo = [
            r for r in self._abstract_rects.get(lo, ())
            if _closed_rects_touch(_norm(r), pbox_lo_q)
        ]
        abs_hi = [
            r for r in self._abstract_rects.get(lo + 1, ())
            if _closed_rects_touch(_norm(r), pbox_hi_q)
        ]
        env_bbox = tuple(int(v) for v in env.bbox_nm) if env is not None else None
        local_env_bbox = None
        if bool(getattr(self, "_active_group_route_local_closure", False)):
            local_iface = getattr(self, "_active_group_route_local_interface", None)
            if local_iface is not None:
                local_env_bbox = tuple(int(v) for v in local_iface[0])
        forb_sites = getattr(self, "_search_forbidden_via_sites", None)
        forb_xy = (
            [
                (int(s[0]), int(s[1]))
                for s in forb_sites
                if len(s) >= 4 and s[2] == lo and s[3] == model_flag
            ]
            if forb_sites
            else None
        )
        mirror = getattr(self, "_active_mirror_legality", None)
        mirror_active = mirror is not None and str(net) == str(mirror.net_a)

        legal_xy, fb_xy, _cnt, agg_rejs = cpp.batch_filter_via_sites_cpp(
            xs=None,
            ys=None,
            extra_pts=None,
            sites=sites_xy,
            atom_rect=None,
            atom_min_width=0,
            lo=lo,
            bh=bh_i,
            ah=ah_i,
            ext=ext_i,
            sp_lo=sp_lo,
            sp_hi=sp_hi,
            vsp=vsp_i,
            grid=grid_i,
            net=str(net),
            obs_lo=obs_lo._native if obs_lo is not None else None,
            obs_hi=obs_hi._native if obs_hi is not None else None,
            obs_cut_lo=obs_cut_lo._native if obs_cut_lo is not None else None,
            fixed_cuts_fallback=fixed_cuts_fb,
            metal_lo=metal_lo._native if metal_lo is not None else None,
            metal_hi=metal_hi._native if metal_hi is not None else None,
            cut_lo=cut_lo._native if cut_lo is not None else None,
            attempt_cuts=local_att_cuts,
            attempt_metal_lo=att_m_lo,
            attempt_metal_hi=att_m_hi,
            abstract_lo=abs_lo,
            abstract_hi=abs_hi,
            envelope_bbox=env_bbox,
            local_envelope_bbox=local_env_bbox,
            forbidden_xy=forb_xy,
            fallback_halos_lo=fh_lo or None,
            fallback_halos_hi=fh_hi or None,
            fallback_halos_cut=fh_cut or None,
            mirror_active=mirror_active,
        )
        self._merge_cpp_via_batch_rejections(agg_rejs, cpp)
        legal_set: Set[Tuple[int, int]] = set(legal_xy)
        for fx, fy in fb_xy:
            if self._via_legal(fx, fy, lo, net, n=model_n):
                legal_set.add((fx, fy))
        return legal_set

    def _compute_owned_component_escape_states(
        self,
        net: str,
        px: int,
        py: int,
        gi: int,
    ) -> List[Tuple[Tuple[int, int, int], Optional[_ViaSite]]]:
        """Upper-layer via states landed on exact atoms of the pin's component."""
        component, fixed_states = self._owned_terminal_stack_access(
            net, px, py, gi
        )
        if not component:
            self._record_component_escape_diagnostic(status="NO_AUTHORIZED_SEED")
            return []

        # Component-level shared memo across pins on the same multi-pin conductor
        # component (e.g. 40-pin GND). All atom via scans, array/single congestion,
        # hard-only site screening, and rejection evidence depend ONLY on the
        # component and immutable world — ONLY the 3 final tie-broken single-via
        # selections depend on (px, py) and fixed_states filtering on gi!
        inst_dict = self.__dict__
        can_share_comp = (
            len(component) >= 1
            and not bool(self._attempt_idx)
            and not bool(self._attempt_cuts)
            and not bool(self._attempt_metal)
            and type(getattr(self, "_history", None)) is dict
            and not any(
                name in inst_dict
                for name in (
                    "_via_legal",
                    "_via_congestion",
                    "_via_pad_attempt_conflicts",
                    "_owned_atom_via_states",
                    "_watchdog_check",
                    "_record_hard_rejection",
                    "_access_state_key",
                )
            )
            and getattr(self, "_active_mirror_legality", None) is None
        )
        comp_shared_key = None
        comp_cache: Optional[OrderedDict] = None
        if can_share_comp:
            env = getattr(self, "_routing_envelope", None)
            history_d = self._history
            world_sig = (
                id(self._obs_idx),
                id(self._metal_idx),
                id(self._cut_idx),
                id(getattr(self, "_obs_cut_idx", None)),
                id(self._abstract_rects),
                id(getattr(self, "_obs_provenance", None)),
                id(getattr(self, "_obs_cut_provenance", None)),
                id(getattr(self, "_via_pad_provenance", None)),
                None if env is None else tuple(env.bbox_nm),
                float(getattr(self, "_pres_fac", 0.0)),
                float(getattr(self, "hist_fac", 0.0)),
                id(history_d),
                len(history_d),
                int(getattr(self, "_history_generation", 0)),
                bool(self._single_via_escape_allowed()),
                int(self.via_redundancy),
                frozenset(getattr(self, "_search_forbidden_via_sites", ()) or ()),
                bool(getattr(self, "_active_group_route_local_closure", False)),
                getattr(self, "_active_group_route_local_interface", None),
                id(getattr(self, "_group_route_global_interfaces_by_net", None)),
                id(getattr(self, "_intent_spacing_rules", None)),
                id(getattr(self, "_mim_bottom_routing_clearance_by_gi", None)),
                id(getattr(self, "_mim_carve_bottom_blockers_by_gi", None)),
                id(getattr(self, "_mim_derived_unrelated_clearance_by_gi", None)),
                bool(self._mim_derived_unrelated_static_active()),
                id(getattr(self, "_mim_forbidden_cut_touch_by_lo", None)),
                id(getattr(self, "_mim_top_via_marker_clearance_by_lo", None)),
            )
            cached_w = _COMPONENT_ESCAPE_SHARED_CACHE.get(self)
            if cached_w is not None and cached_w[0] == world_sig:
                comp_cache = cached_w[1]
            else:
                comp_cache = OrderedDict()
                _COMPONENT_ESCAPE_SHARED_CACHE[self] = (world_sig, comp_cache)
            comp_shared_key = (str(net), component)
            shared = comp_cache.get(comp_shared_key)
            if shared is not None:
                comp_cache.move_to_end(comp_shared_key)
                (
                    base_states_ordered,
                    base_keys_ordered,
                    base_keys_set,
                    base_cand_count,
                    improving_data,
                    pressured_data,
                    comp_rej,
                    comp_samp,
                    comp_rej_total,
                ) = shared
                self._merge_component_access_rejections(comp_rej, comp_samp)
                if comp_rej_total:
                    self._hard_rejection_generation = int(
                        getattr(self, "_hard_rejection_generation", 0)
                    ) + comp_rej_total
                extra_by_key: Dict[Tuple[int, ...], Tuple[Tuple[int, int, int], Optional[_ViaSite]]] = {}
                if improving_data:
                    # pressured_data: tuple of (x, y, lo, key)
                    p_best = min(
                        pressured_data,
                        key=lambda it: (
                            abs(it[0] - px) + abs(it[1] - py),
                            abs(it[1] - py),
                            abs(it[0] - px),
                            it[3],
                        ),
                    )
                    pr_x, pr_y = p_best[0], p_best[1]
                    # improving_data: tuple of (state, via, x, y, cong, key)
                    s_cong = min(
                        improving_data,
                        key=lambda it: (
                            it[4],
                            abs(it[2] - px) + abs(it[3] - py),
                            abs(it[3] - py),
                            abs(it[2] - px),
                            it[5],
                        ),
                    )
                    s_pin = min(
                        improving_data,
                        key=lambda it: (
                            abs(it[2] - px) + abs(it[3] - py),
                            abs(it[3] - py),
                            abs(it[2] - px),
                            it[4],
                            it[5],
                        ),
                    )
                    s_pres = min(
                        improving_data,
                        key=lambda it: (
                            abs(it[2] - pr_x) + abs(it[3] - pr_y),
                            abs(it[3] - pr_y),
                            abs(it[2] - pr_x),
                            it[4],
                            it[5],
                        ),
                    )
                    for sel in (s_cong, s_pin, s_pres):
                        if sel[5] not in base_keys_set:
                            extra_by_key[sel[5]] = (sel[0], sel[1])
                for fst in fixed_states:
                    fk = (*fst, 0, 0, 0, 0, 0)
                    if fk not in base_keys_set:
                        extra_by_key[fk] = (fst, None)
                if not extra_by_key:
                    ordered = list(base_states_ordered)
                else:
                    ordered = list(base_states_ordered)
                    # Insert <= ~5 extra keys in descending position order into pre-sorted list
                    inserts = sorted(
                        ((bisect.bisect_left(base_keys_ordered, ek), ek, ev) for ek, ev in extra_by_key.items()),
                        key=lambda x: (x[0], x[1]),
                        reverse=True,
                    )
                    for pos, _ek, ev in inserts:
                        ordered.insert(pos, ev)
                self._record_component_escape_diagnostic(
                    status="LEGAL_LANDING" if ordered else "NO_LEGAL_LANDING",
                    component=component,
                    anchor_count=base_cand_count + len(fixed_states),
                    legal_count=len(ordered),
                )
                return ordered

        outer_comp_rej = None
        outer_comp_samp = None
        outer_comp_gen = 0
        if comp_shared_key is not None:
            outer_comp_rej = getattr(self, "_hard_rejections", None)
            outer_comp_samp = getattr(self, "_hard_rejection_samples", None)
            outer_comp_gen = int(getattr(self, "_hard_rejection_generation", 0))
            if outer_comp_rej is not None:
                self._hard_rejections = {}
                self._hard_rejection_samples = {}

        improving_cache_data: Tuple[Any, ...] = ()
        pressured_cache_data: Tuple[Any, ...] = ()
        base_states_for_cache: Optional[List[Tuple[Tuple[int, int, int], Optional[_ViaSite]]]] = None
        try:
            legality_cache = self._component_access_via_legality_cache(net)
            states = []
            candidate_count = 0
            for atom in component:
                if atom.gi >= self._n_layers - 1:
                    continue
                atom_states, atom_candidate_count = self._owned_atom_via_states(
                    atom, net, None, legality_cache=legality_cache
                )
                states.extend(atom_states)
                candidate_count += atom_candidate_count
            check_attempt = (
                bool(self._attempt_idx)
                or "_via_pad_attempt_conflicts" in self.__dict__
            )
            if check_attempt:
                states = [
                    item for item in states
                    if item[1] is not None
                    and not self._via_pad_attempt_conflicts(net, item[1])
                ]
            else:
                states = [item for item in states if item[1] is not None]
            if self._single_via_escape_allowed():
                arrays_exist = bool(states)
                single_domain = []
                for atom in component:
                    if atom.gi >= self._n_layers - 1:
                        continue
                    atom_states, atom_candidate_count = self._owned_atom_via_states(
                        atom, net, 1, legality_cache=legality_cache
                    )
                    candidate_count += atom_candidate_count
                    single_domain.extend(atom_states)
                if check_attempt:
                    single_domain = [
                        item for item in single_domain
                        if item[1] is not None
                        and not self._via_pad_attempt_conflicts(net, item[1])
                    ]
                else:
                    single_domain = [item for item in single_domain if item[1] is not None]
                if not arrays_exist:
                    single_alternatives = single_domain
                    base_states_for_cache = list(states) + list(single_alternatives)
                else:
                    unique_single = {
                        self._access_state_key(item): item
                        for item in single_domain
                    }
                    sorted_single_keys = sorted(unique_single)
                    cpp_cong = self._can_use_cpp_batch_congestion()
                    if cpp_cong is not None and len(states) > 1:
                        states = list({item[1]: item for item in states}.values())
                    hist_by_rgi: Dict[int, List[Tuple[int, int, int]]] = {}
                    if cpp_cong is not None and self._history:
                        hfac = self.hist_fac * _COST_FACTOR_SCALE
                        for (hcx, hcy, hrgi), hv in self._history.items():
                            if hv:
                                hist_by_rgi.setdefault(int(hrgi), []).append(
                                    (int(hcx), int(hcy), int(round(hv * hfac)))
                                )

                    scored_single = []
                    if cpp_cong is not None and len(sorted_single_keys) >= 4:
                        single_items = [unique_single[key] for key in sorted_single_keys]
                        by_lo_s: Dict[int, List[int]] = {}
                        for idx_i, (_st, via) in enumerate(single_items):
                            assert via is not None and via.n == 1
                            by_lo_s.setdefault(via.lo, []).append(idx_i)
                        single_congs = [0] * len(single_items)
                        for lo_k, idxs in by_lo_s.items():
                            pts = [(single_items[i][1].x, single_items[i][1].y) for i in idxs]
                            vals = self._batch_via_congestion_for_sites(
                                pts, lo_k, net, 1, cpp_cong, hist_by_rgi
                            )
                            for i_pos, val in zip(idxs, vals):
                                single_congs[i_pos] = val
                        for key, (state, via), congestion in zip(
                            sorted_single_keys, single_items, single_congs
                        ):
                            scored_single.append((state, via, congestion, key))
                    else:
                        for key in sorted_single_keys:
                            state, via = unique_single[key]
                            assert via is not None and via.n == 1
                            congestion = self._via_congestion(
                                via.x, via.y, via.lo, net, 1
                            )
                            scored_single.append((state, via, congestion, key))

                    scored_arrays = []
                    array_congestion_by_site = {}
                    if cpp_cong is not None and len(states) >= 4:
                        by_lo_a: Dict[int, List[int]] = {}
                        for idx_i, (_ast, avia) in enumerate(states):
                            assert avia is not None and avia.n == 0
                            by_lo_a.setdefault(avia.lo, []).append(idx_i)
                        array_congs = [0] * len(states)
                        for lo_k, idxs in by_lo_a.items():
                            pts = [(states[i][1].x, states[i][1].y) for i in idxs]
                            vals = self._batch_via_congestion_for_sites(
                                pts, lo_k, net, None, cpp_cong, hist_by_rgi
                            )
                            for i_pos, val in zip(idxs, vals):
                                array_congs[i_pos] = val
                        for (array_state, array_via), array_congestion in zip(states, array_congs):
                            scored_arrays.append((
                                array_state,
                                array_via,
                                array_congestion,
                                self._access_state_key((array_state, array_via)),
                            ))
                            site_key = (array_via.x, array_via.y, array_via.lo)
                            array_congestion_by_site[site_key] = max(
                                array_congestion,
                                array_congestion_by_site.get(site_key, 0),
                            )
                    else:
                        for array_state, array_via in states:
                            assert array_via is not None and array_via.n == 0
                            array_congestion = self._via_congestion(
                                array_via.x, array_via.y, array_via.lo, net, None
                            )
                            scored_arrays.append((
                                array_state,
                                array_via,
                                array_congestion,
                                self._access_state_key((array_state, array_via)),
                            ))
                            site_key = (array_via.x, array_via.y, array_via.lo)
                            array_congestion_by_site[site_key] = max(
                                array_congestion,
                                array_congestion_by_site.get(site_key, 0),
                            )
                    same_site_alternatives = [
                        item[:2] for item in scored_single
                        if (
                            (item[1].x, item[1].y, item[1].lo)
                            in array_congestion_by_site
                            and item[2] < array_congestion_by_site[
                                (item[1].x, item[1].y, item[1].lo)
                            ]
                        )
                    ]
                    cand_hard_items = [
                        item for item in scored_single
                        if (item[1].x, item[1].y, item[1].lo) not in array_congestion_by_site
                    ]
                    hard_only_sites = set()
                    if cand_hard_items:
                        cpp_leg = load_native("_gridless_cpp")
                        used_cpp_hard = False
                        if cpp_leg is not None and len(cand_hard_items) >= 4:
                            by_lo_h: Dict[int, List[Any]] = {}
                            for item in cand_hard_items:
                                by_lo_h.setdefault(item[1].lo, []).append(item)
                            all_ok = True
                            illegal_by_lo: Dict[int, List[Any]] = {}
                            for lo_k, items_k in by_lo_h.items():
                                pts_k = [(it[1].x, it[1].y) for it in items_k]
                                legal_set_k = self._batch_via_legal_for_sites_cpp(
                                    pts_k, lo_k, net, None, cpp_leg
                                )
                                if legal_set_k is None:
                                    all_ok = False
                                    break
                                illegal_by_lo[lo_k] = [
                                    it for it in items_k
                                    if (it[1].x, it[1].y) not in legal_set_k
                                ]
                            if all_ok:
                                used_cpp_hard = True
                                for lo_k, ill_items in illegal_by_lo.items():
                                    if not ill_items:
                                        continue
                                    if cpp_cong is not None:
                                        pts_i = [(it[1].x, it[1].y) for it in ill_items]
                                        congs_i = self._batch_via_congestion_for_sites(
                                            pts_i, lo_k, net, None, cpp_cong, hist_by_rgi
                                        )
                                        for it, arr_c in zip(ill_items, congs_i):
                                            if arr_c > it[2]:
                                                hard_only_sites.add((it[1].x, it[1].y, lo_k))
                                    else:
                                        for it in ill_items:
                                            if self._via_congestion(it[1].x, it[1].y, lo_k, net) > it[2]:
                                                hard_only_sites.add((it[1].x, it[1].y, lo_k))
                        if not used_cpp_hard:
                            hard_only_sites = {
                                (item[1].x, item[1].y, item[1].lo)
                                for item in cand_hard_items
                                if (
                                    not self._via_legal(
                                        item[1].x, item[1].y, item[1].lo, net
                                    )
                                    and self._via_congestion(
                                        item[1].x, item[1].y, item[1].lo, net
                                    ) > item[2]
                                )
                            }
                    pressure = max(item[2] for item in scored_arrays)
                    improving = [
                        item for item in scored_single
                        if (
                            (item[1].x, item[1].y, item[1].lo) in hard_only_sites
                            or item[2] < pressure
                        )
                    ]
                    base_states_for_cache = list(states) + list(same_site_alternatives)
                    single_alternatives = list(same_site_alternatives)
                    if improving:
                        pressured = [
                            item for item in scored_arrays if item[2] == pressure
                        ]
                        if comp_shared_key is not None:
                            pressured_cache_data = tuple(
                                (it[1].x, it[1].y, it[1].lo, it[3]) for it in pressured
                            )
                            improving_cache_data = tuple(
                                (it[0], it[1], it[1].x, it[1].y, it[2], it[3]) for it in improving
                            )
                        _pressure_state, pressure_via, _cost, _key = min(
                            pressured,
                            key=lambda item: (
                                abs(item[1].x - px) + abs(item[1].y - py),
                                abs(item[1].y - py),
                                abs(item[1].x - px),
                                item[3],
                            ),
                        )

                        def pin_key(item):
                            _state, single_via, congestion, key = item
                            dx = single_via.x - px
                            dy = single_via.y - py
                            return (abs(dx) + abs(dy), abs(dy), abs(dx), congestion, key)

                        def pressure_key(item):
                            _state, single_via, congestion, key = item
                            dx = single_via.x - pressure_via.x
                            dy = single_via.y - pressure_via.y
                            return (abs(dx) + abs(dy), abs(dy), abs(dx), congestion, key)

                        def congestion_key(item):
                            _state, single_via, congestion, key = item
                            dx = single_via.x - px
                            dy = single_via.y - py
                            return (congestion, abs(dx) + abs(dy), abs(dy), abs(dx), key)

                        for selected in (
                            min(improving, key=congestion_key),
                            min(improving, key=pin_key),
                            min(improving, key=pressure_key),
                        ):
                            single_alternatives.append(selected[:2])
                states.extend(single_alternatives)
            else:
                base_states_for_cache = list(states)
        finally:
            if comp_shared_key is not None and outer_comp_rej is not None:
                comp_rej = tuple(sorted(
                    (cause, int(cnt))
                    for cause, cnt in self._hard_rejections.items()
                ))
                comp_samp = tuple(
                    (cause, int(cnt), tuple(ext))
                    for cause, (cnt, ext) in sorted(
                        self._hard_rejection_samples.items()
                    )
                )
                comp_rej_total = sum(int(cnt) for _c, cnt in comp_rej)
                self._hard_rejections = outer_comp_rej
                self._hard_rejection_samples = outer_comp_samp
                self._hard_rejection_generation = outer_comp_gen + comp_rej_total
                self._merge_component_access_rejections(comp_rej, comp_samp)
                if comp_cache is not None and base_states_for_cache is not None:
                    base_u = {
                        (
                            (*s, 0, 0, 0, 0, 0)
                            if v is None
                            else (*s, 1, v.x, v.y, v.lo, v.n)
                        ): (s, v)
                        for s, v in base_states_for_cache
                    }
                    base_keys_ord = tuple(sorted(base_u))
                    base_ord = tuple(base_u[k] for k in base_keys_ord)
                    comp_cache[comp_shared_key] = (
                        base_ord,
                        base_keys_ord,
                        frozenset(base_u),
                        candidate_count,
                        improving_cache_data,
                        pressured_cache_data,
                        comp_rej,
                        comp_samp,
                        comp_rej_total,
                    )
                    while len(comp_cache) > 64:
                        comp_cache.popitem(last=False)

        candidate_count += len(fixed_states)
        states.extend((state, None) for state in fixed_states)
        unique = {
            self._access_state_key(item): item
            for item in states
        }
        ordered = [unique[key] for key in sorted(unique)]
        self._record_component_escape_diagnostic(
            status="LEGAL_LANDING" if ordered else "NO_LEGAL_LANDING",
            component=component,
            anchor_count=candidate_count,
            legal_count=len(ordered),
        )
        return ordered

