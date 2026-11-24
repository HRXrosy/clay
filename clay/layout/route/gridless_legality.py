"""DRC legality, static proof caching, MIM/wide clearances, and congestion mixin for GridlessEngine."""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple
from weakref import WeakKeyDictionary

from clay._native import load_native
from clay.layout.route.constraints import Length, Mirror
from clay.layout.route.gridless_types import (
    Rect,
    _CONG_WEIGHT,
    _COST_FACTOR_SCALE,
    _OwnedTerminalAtom,
    _RoutedNet as _base_RoutedNet,
    _STEP_COST_PER_NM,
    _SpatialIndex,
    _ViaSite,
    _WireSeg,
    _closed_rects_touch,
    _deck_open_route_nets,
    _eg_attr,
    _norm,
    _rects_clear as _base_rects_clear,
    _seg_bbox,
    _sep as _base_sep,
)
from clay.layout.route.spacing_intent import (
    check_spacing_geometry,
    fixed_spacing_conflicts,
    spacing_blockers,
)
from clay.layout.route.via_model import subtract_rects


_NATIVE_HISTORY_GRID_CACHE: "WeakKeyDictionary[Any, Tuple[Any, Any]]" = WeakKeyDictionary()


class _GridlessLegalityMixin:
    """Static/dynamic wire & via legality, wide/MIM clearances, and congestion."""

    def _try_complete_world_legalization(
        self,
        committed: Dict[str, _RoutedNet],
        allowed_layers_by_net: Mapping[str, Sequence[int]],
        ordered: Sequence[str],
        visible_spacing: Sequence[Tuple[str, str]],
        *,
        mirrors: Sequence[Mirror] = (),
        length_constraints: Sequence[Length] = (),
    ) -> Optional[Dict[str, _RoutedNet]]:
        """Try existing final-world fills without publishing an incomplete trial.

        The caller has just proved a complete isolated world. Keep its original
        primitive objects, and accept only a strict reduction of visible pairs
        with the same complete deck and hard intent obligations. This stage
        never searches, retunes lengths, reconstructs mirrors, or renews a
        budget. The final emitted-world closure remains independent.
        """
        _RoutedNet = _eg_attr("_RoutedNet", _base_RoutedNet)
        if not ordered or not any(a == b for a, b in visible_spacing):
            return None
        world_names = (
            "_book", "_metal_idx", "_via_pad_idx", "_via_pad_provenance",
            "_cut_idx", "_last_deck_spacing_history_cells",
        )
        diagnostic_names = (
            "_hard_rejections", "_hard_rejection_samples",
            "_hard_rejection_generation", "_route_watchdog_details",
            "_route_failure_details", "_access_watchdog_details",
            "_astar_watchdog_details", "_mirror_failure_details",
        )
        missing = object()
        saved = {
            name: getattr(self, name, missing)
            for name in (*world_names, *diagnostic_names)
        }
        accepted = False
        try:
            # Fill vetoes may record diagnostics. Trial dictionaries must not
            # mutate an alias belonging to the preceding route transaction.
            for name in ("_hard_rejections", "_hard_rejection_samples"):
                previous = saved[name]
                setattr(self, name, {} if previous is missing else dict(previous))
            details = dict(getattr(self, "_route_watchdog_details", ()))
            details["legalization_stage"] = "before_ripup"
            self._route_watchdog_details = tuple(sorted(details.items()))
            self._watchdog_check(ordered[0], "pre_ripup_legalize")
            previous_lengths = tuple(
                self._length_report_for_committed(constraint, committed)
                for constraint in length_constraints
            )
            trial = {
                net: _RoutedNet(
                    name=route.name, segs=list(route.segs), vias=list(route.vias),
                    connected=route.connected,
                )
                for net, route in committed.items()
            }
            self._legalize_committed_for_layers(trial, allowed_layers_by_net)
            self._watchdog_check(ordered[0], "pre_ripup_legalize")
            # Additive repair preserves original endpoints, widths and selected
            # via models. New fills retain the actual lowered layer permission.
            changed = False
            if trial.keys() != committed.keys():
                return None
            for net, original in committed.items():
                candidate = trial[net]
                if (
                    candidate.name != original.name
                    or candidate.connected != original.connected
                    or len(candidate.segs) < len(original.segs)
                    or any(current is not previous for current, previous in
                           zip(candidate.segs, original.segs))
                    or len(candidate.vias) != len(original.vias)
                    or any(current is not previous for current, previous in
                           zip(candidate.vias, original.vias))
                ):
                    return None
                additions = candidate.segs[len(original.segs):]
                if any(segment.gi not in allowed_layers_by_net.get(net, ())
                       for segment in additions):
                    return None
                changed = changed or bool(additions)
            if not changed:
                return None
            _advisory, candidate_spacing = self._space_residual_scan(trial)
            self._watchdog_check(ordered[0], "pre_ripup_spacing")
            if not set(candidate_spacing) < set(visible_spacing):
                return None
            deck = self._deck_negotiation_snapshot(trial)
            self._watchdog_check(ordered[0], "pre_ripup_deck")
            if (
                not deck.report.cross_net_isolated
                or not deck.report.probe_complete
                or _deck_open_route_nets(deck.report, ordered)
                or any(not trial.get(net, _RoutedNet(name=net)).connected
                       for net in ordered)
            ):
                return None
            for mirror in mirrors:
                primary, partner = trial.get(mirror.net_a), trial.get(mirror.net_b)
                if primary is None or partner is None:
                    return None
                report = self._mirror_report(mirror, primary, partner)
                self._watchdog_check(ordered[0], "pre_ripup_intent")
                if not report["satisfied"]:
                    return None
            for constraint, before in zip(length_constraints, previous_lengths):
                after = self._length_report_for_committed(constraint, trial)
                self._watchdog_check(ordered[0], "pre_ripup_intent")
                if (
                    (before["satisfied"] and not after["satisfied"])
                    or (not before["satisfied"] and
                        after["relative_imbalance"] > before["relative_imbalance"])
                ):
                    return None
            # Native region/deck kernels can cross a deadline between their
            # own checkpoints. Never publish a candidate after such a return.
            self._watchdog_check(ordered[0], "pre_ripup_commit")
            accepted = True
            return trial
        finally:
            restore_names = diagnostic_names if accepted else (*world_names, *diagnostic_names)
            for name in restore_names:
                previous = saved[name]
                if previous is missing:
                    self.__dict__.pop(name, None)
                else:
                    setattr(self, name, previous)
            if not accepted:
                # Restored indexes still describe the original world; keep the
                # generation monotonic and discard any trial-dependent reuse.
                self._component_access.committed_world_changed()

    def _complete_world_wire_displacement_candidates(
        self,
        committed: Mapping[str, _RoutedNet],
        allowed_layers_by_net: Mapping[str, Sequence[int]],
        visible_spacing: Sequence[Tuple[str, str]],
    ) -> Iterable[Tuple[str, _RoutedNet]]:
        """Derive ordinary-wire chain shifts from exact foreign wire gaps.

        Only an endpoint-connected collinear chain of one actual minimum width
        is supported. It translates to the nearest outward rule/grid frontier;
        incident perpendicular endpoints follow without moving their far ends.
        Vias, fixed metal, rails and certified group conductors never move.
        These are proposals only: the complete-world transaction proves every
        physical access and rule before one may replace committed geometry.
        """
        _RoutedNet = _eg_attr("_RoutedNet", _base_RoutedNet)
        _sep = _eg_attr("_sep", _base_sep)
        def direction(segment: _WireSeg) -> Optional[int]:
            if segment.ay == segment.by and segment.ax != segment.bx:
                return 0
            if segment.ax == segment.bx and segment.ay != segment.by:
                return 1
            return None

        def endpoints(segment: _WireSeg):
            return ((segment.ax, segment.ay), (segment.bx, segment.by))

        protected = set(getattr(self, "_rail_by_net", {}))
        for name in (
            "_group_route_landing_baselines",
            "_group_route_local_segment_keys_by_net",
            "_group_route_local_via_keys_by_net",
            "_group_route_global_interfaces_by_net",
        ):
            protected.update(getattr(self, name, {}))
        wire_rows = {}
        indexes = {}
        for net in sorted({name for pair in visible_spacing for name in pair}):
            route = committed.get(net)
            if route is None or not route.connected:
                continue
            rows = {}
            for index, segment in enumerate(route.segs):
                if index % 1024 == 0:
                    self._watchdog_check(net, "wire_displacement_domain")
                gi = segment.gi
                if (
                    direction(segment) is None
                    or self._segment_half_width(net, segment) != self._half_w[gi]
                ):
                    continue
                rectangle = _norm(_seg_bbox(
                    segment.ax, segment.ay, segment.bx, segment.by,
                    self._segment_half_width(net, segment),
                ))
                rows[index] = rectangle
                indexes.setdefault((net, gi), _SpatialIndex()).add(str(index), rectangle)
            wire_rows[net] = rows

        chain_cache = {}

        def chain(net: str, seed_index: int) -> Tuple[int, ...]:
            key = (net, seed_index)
            if key in chain_cache:
                return chain_cache[key]
            segments = committed[net].segs
            seed = segments[seed_index]
            along = direction(seed)
            normal = 1 - along
            coordinate = endpoints(seed)[0][normal]
            half = self._segment_half_width(net, seed)
            eligible = {
                index for index, item in enumerate(segments)
                if item.gi == seed.gi and direction(item) == along
                and endpoints(item)[0][normal] == coordinate
                and self._segment_half_width(net, item) == half
            }
            selected = {seed_index}
            points = set(endpoints(seed))
            while True:
                self._watchdog_check(net, "wire_displacement_domain")
                additions = {
                    index for index in eligible - selected
                    if points.intersection(endpoints(segments[index]))
                }
                if not additions:
                    break
                selected.update(additions)
                for index in additions:
                    points.update(endpoints(segments[index]))
            intervals = sorted(
                tuple(sorted(point[along] for point in endpoints(segments[index])))
                for index in selected
            )
            supported = all(
                left[1] == right[0] for left, right in zip(intervals, intervals[1:])
            )
            low, high = intervals[0][0], intervals[-1][1]
            for index, item in enumerate(segments):
                if index in selected or item.gi != seed.gi:
                    continue
                item_direction = direction(item)
                item_points = endpoints(item)
                if item_direction == along and item_points[0][normal] == coordinate:
                    a, b = sorted(point[along] for point in item_points)
                    # An overlapping interval or a different-width continuation
                    # needs a richer topology proof than this chain primitive.
                    if max(low, a) <= min(high, b):
                        supported = False
                elif item_direction == normal:
                    a, b = sorted(point[normal] for point in item_points)
                    if low <= item_points[0][along] <= high and a <= coordinate <= b:
                        # An existing perpendicular crossing can remain fixed.
                        # Its exact full-width support at BOTH axes is checked
                        # when the candidate displacement is known below.
                        if self._segment_half_width(net, item) != half:
                            supported = False
                elif points.intersection(item_points):
                    supported = False
            result = tuple(sorted(selected)) if supported else ()
            for index in selected:
                chain_cache[(net, index)] = result
            return result

        proposals = set()
        grid = max(1, int(getattr(self, "_manufacturing_grid_nm", 1)))
        for first, second in sorted(set(tuple(sorted(pair)) for pair in visible_spacing)):
            if first == second:
                continue
            for net, other in ((first, second), (second, first)):
                if net in protected or net not in wire_rows or other not in wire_rows:
                    continue
                route = committed[net]
                for index, rectangle in sorted(wire_rows[net].items()):
                    self._watchdog_check(net, "wire_displacement_domain")
                    segment = route.segs[index]
                    gi = segment.gi
                    if gi not in allowed_layers_by_net.get(net, ()):
                        continue
                    other_index = indexes.get((other, gi))
                    if other_index is None:
                        continue
                    space = self._space[gi]
                    query = (rectangle[0] - space, rectangle[1] - space,
                             rectangle[2] + space, rectangle[3] + space)
                    for _other_index, blocker in sorted(other_index.query(query)):
                        sx, sy = _sep(rectangle, blocker)
                        if not ((sx == 0 < sy < space) or (sy == 0 < sx < space)):
                            continue
                        normal = 0 if sx else 1
                        along = 1 - normal
                        if direction(segment) != along or (
                            min(rectangle[along + 2], blocker[along + 2])
                            <= max(rectangle[along], blocker[along])
                        ):
                            continue
                        selected = chain(net, index)
                        if not selected:
                            continue
                        half = self._segment_half_width(net, segment)
                        if (2 * half) % grid:
                            continue
                        residue = half % grid
                        before = endpoints(segment)[0][normal]
                        if rectangle[normal + 2] <= blocker[normal]:
                            boundary = blocker[normal] - space - half
                            after = residue + ((boundary - residue) // grid) * grid
                        else:
                            boundary = blocker[normal + 2] + space + half
                            after = residue - ((residue - boundary) // grid) * grid
                        if after != before:
                            proposals.add((abs(after - before), net, gi, selected, normal, after))

        for _distance, net, gi, selected, normal, after in sorted(proposals):
            self._watchdog_check(net, "wire_displacement_candidate")
            original = committed[net]
            selected_set = set(selected)
            points = {point for index in selected for point in endpoints(original.segs[index])}
            before = endpoints(original.segs[selected[0]])[0][normal]
            shift = after - before
            along = 1 - normal
            low, high = min(point[along] for point in points), max(point[along] for point in points)
            replacements = list(original.segs)
            supported = True
            for index, segment in enumerate(original.segs):
                if segment.gi != gi:
                    continue
                a, b = (list(point) for point in endpoints(segment))
                if index in selected_set:
                    a[normal] += shift
                    b[normal] += shift
                elif direction(segment) == normal and points.intersection(endpoints(segment)):
                    for point in (a, b):
                        if tuple(point) in points:
                            point[normal] += shift
                    old_delta = endpoints(segment)[1][normal] - endpoints(segment)[0][normal]
                    if old_delta * (b[normal] - a[normal]) <= 0:
                        supported = False
                        break
                elif direction(segment) == normal:
                    near, far = sorted(point[normal] for point in endpoints(segment))
                    if low <= a[along] <= high and near <= before <= far:
                        # Equal effective widths and a crossing centre inside
                        # both centreline spans prove a complete rule-width
                        # square at each intersection. Keep this conductor
                        # unchanged; do not drag an unrelated branch endpoint.
                        if not near <= after <= far:
                            supported = False
                            break
                    continue
                else:
                    continue
                replacements[index] = _WireSeg(
                    gi, *a, *b, segment.half_width_nm,
                )
            if supported:
                yield net, _RoutedNet(
                    name=net, segs=replacements, vias=list(original.vias),
                    connected=original.connected,
                )

    def _try_complete_world_wire_displacement(
        self,
        committed: Dict[str, _RoutedNet],
        allowed_layers_by_net: Mapping[str, Sequence[int]],
        ordered: Sequence[str],
        visible_spacing: Sequence[Tuple[str, str]],
        *,
        net_nodes_by_net: Mapping[str, Sequence[Tuple[int, int, int]]],
        mirrors: Sequence[Mirror] = (),
        length_constraints: Sequence[Length] = (),
    ) -> Optional[Dict[str, _RoutedNet]]:
        """Admit one fully checked chain displacement before whole-net rip-up.

        A positive foreign wire band remains soft during search. This distinct
        complete-world transaction may replace a chain only after real emitted
        geometry, every original physical pin/via, and hard intent prove that
        the visible conflict set strictly shrank. Unsupported proposals fall
        back to ordinary negotiation without renewing any work or wall budget.
        """
        _RoutedNet = _eg_attr("_RoutedNet", _base_RoutedNet)
        if not ordered or not any(a != b for a, b in visible_spacing):
            return None
        world_names = (
            "_book", "_metal_idx", "_via_pad_idx", "_via_pad_provenance",
            "_cut_idx", "_last_deck_spacing_history_cells",
        )
        diagnostic_names = (
            "_hard_rejections", "_hard_rejection_samples",
            "_hard_rejection_generation", "_route_watchdog_details",
            "_route_failure_details", "_access_watchdog_details",
            "_astar_watchdog_details", "_mirror_failure_details",
        )
        missing = object()
        saved = {name: getattr(self, name, missing)
                 for name in (*world_names, *diagnostic_names)}
        accepted = False
        try:
            for name in ("_hard_rejections", "_hard_rejection_samples"):
                previous = saved[name]
                setattr(self, name, {} if previous is missing else dict(previous))
            details = dict(getattr(self, "_route_watchdog_details", ()))
            details["legalization_stage"] = "wire_displacement"
            self._route_watchdog_details = tuple(sorted(details.items()))
            self._watchdog_check(ordered[0], "wire_displacement_entry")
            previous_lengths = tuple(
                self._length_report_for_committed(item, committed)
                for item in length_constraints
            )
            protected = set(getattr(self, "_rail_by_net", {}))
            for name in (
                "_group_route_landing_baselines",
                "_group_route_local_segment_keys_by_net",
                "_group_route_local_via_keys_by_net",
                "_group_route_global_interfaces_by_net",
            ):
                protected.update(getattr(self, name, {}))
            grid = max(1, int(getattr(self, "_manufacturing_grid_nm", 1)))
            bounds = _norm(self._bounds)
            for net, candidate in self._complete_world_wire_displacement_candidates(
                committed, allowed_layers_by_net, visible_spacing,
            ):
                self._watchdog_check(net, "wire_displacement_candidate")
                original = committed.get(net)
                probes = tuple(tuple(map(int, state)) for state in net_nodes_by_net.get(net, ()))
                if (
                    original is None or net in protected or not probes
                    or candidate.name != net or not original.connected or not candidate.connected
                    or len(candidate.segs) != len(original.segs)
                    or len(candidate.vias) != len(original.vias)
                    or any(current is not previous for current, previous in
                           zip(candidate.vias, original.vias))
                ):
                    continue
                changed = [(old, new) for old, new in zip(original.segs, candidate.segs)
                           if old is not new]
                if not changed:
                    continue
                if any(
                    old.gi != new.gi or old.half_width_nm != new.half_width_nm
                    or new.gi not in allowed_layers_by_net.get(net, ())
                    or self._segment_half_width(net, old) != self._segment_half_width(net, new)
                    or self._segment_half_width(net, new) != self._half_w[new.gi]
                    or not ((new.ax == new.bx) ^ (new.ay == new.by))
                    or (old.ax == old.bx) != (new.ax == new.bx)
                    for old, new in changed
                ):
                    continue
                required = {tuple(state) for owner, state in
                            getattr(self, "_required_wire_endpoints", ()) if owner == net}
                endpoints = {(x, y, segment.gi) for segment in candidate.segs
                             for x, y in ((segment.ax, segment.ay), (segment.bx, segment.by))}
                if not required <= endpoints:
                    continue
                boxes_by_gi: Dict[int, List[Rect]] = {}
                physical = True
                for _old, segment in changed:
                    half = self._segment_half_width(net, segment)
                    box = _norm(_seg_bbox(segment.ax, segment.ay, segment.bx, segment.by, half))
                    if (
                        any(value % grid for value in box)
                        or not (bounds[0] <= box[0] and bounds[1] <= box[1]
                                and box[2] <= bounds[2] and box[3] <= bounds[3])
                    ):
                        physical = False
                        break
                    # Static geometry includes group interiors, MIM carving and
                    # bottom-metal clearance. It does not consume a previous
                    # MST attempt's forbidden edges, mirror or access state.
                    with self._half_width_scope({(net, segment.gi): half}):
                        if not self._compute_wire_static_legality(
                            segment.gi, segment.ax, segment.ay,
                            segment.bx, segment.by, net,
                        ):
                            physical = False
                            break
                    boxes_by_gi.setdefault(segment.gi, []).append(box)
                if not physical:
                    continue
                trial = dict(committed)
                trial[net] = candidate
                self._rebuild_committed_state(trial)
                self._watchdog_check(net, "wire_displacement_geometry")
                # This exact-footprint predicate supplies foreign, envelope and
                # named spacing checks. Own width/notch is proved from the full
                # canonical merged layer below, never waived on pairwise hope.
                if any(not self._fill_legal(net, gi, boxes, check_own_junctions=False)
                       for gi, boxes in sorted(boxes_by_gi.items())):
                    continue
                if not self._repair_preserves_world(
                    net, self._net_metal(original)[0], self._net_metal(candidate)[0],
                ):
                    continue
                _advisory, candidate_spacing = self._space_residual_scan(trial)
                self._watchdog_check(net, "wire_displacement_spacing")
                if not set(candidate_spacing) < set(visible_spacing):
                    continue
                deck = self._deck_negotiation_snapshot(trial)
                self._watchdog_check(net, "wire_displacement_deck")
                if (
                    not deck.report.cross_net_isolated or not deck.report.probe_complete
                    or _deck_open_route_nets(deck.report, ordered)
                    or any(not trial.get(name, _RoutedNet(name=name)).connected for name in ordered)
                    or self._detect_shorts(trial)
                ):
                    continue
                import klayout.db as kdb

                layer_clear = True
                for gi in sorted(boxes_by_gi):
                    layer = tuple(self._tech.gi_to_metal_key(gi))
                    region = kdb.Region()
                    for literal in deck.literal_rects:
                        if tuple(literal.layer) == layer:
                            region.insert(kdb.Box(*literal.rect))
                    region.merge()
                    if not (
                        region.width_check(2 * self._half_w[gi]).is_empty()
                        and region.space_check(self._space[gi]).is_empty()
                        and region.notch_check(self._space[gi]).is_empty()
                    ):
                        layer_clear = False
                    self._watchdog_check(net, "wire_displacement_layer")
                if not layer_clear:
                    continue
                # Literal-centre deck probes alone can miss a moved virtual pin.
                # Pair each actual original node and unchanged via landing with
                # one retained physical anchor through the exact PDK connect stack.
                all_probes = sorted(set(probes) | {
                    (via.x, via.y, gi) for via in original.vias
                    for gi in (via.lo, via.lo + 1)
                })
                connected = True
                for probe in all_probes:
                    if not self._candidate_deck_connects_probes(
                        net, candidate.segs, candidate.vias, (probes[0], probe),
                        include_fixed=True,
                    ):
                        connected = False
                        break
                    self._watchdog_check(net, "wire_displacement_access")
                if not connected:
                    continue
                intent_clear = True
                for mirror in mirrors:
                    primary, partner = trial.get(mirror.net_a), trial.get(mirror.net_b)
                    if primary is None or partner is None or not self._mirror_report(
                        mirror, primary, partner,
                    )["satisfied"]:
                        intent_clear = False
                        break
                    self._watchdog_check(net, "wire_displacement_intent")
                for constraint, before in zip(length_constraints, previous_lengths):
                    after = self._length_report_for_committed(constraint, trial)
                    if (
                        (before["satisfied"] and not after["satisfied"])
                        or (not before["satisfied"] and
                            after["relative_imbalance"] > before["relative_imbalance"])
                    ):
                        intent_clear = False
                    self._watchdog_check(net, "wire_displacement_intent")
                if not intent_clear:
                    continue
                self._watchdog_check(net, "wire_displacement_commit")
                accepted = True
                return trial
            return None
        finally:
            restore_names = diagnostic_names if accepted else (*world_names, *diagnostic_names)
            for name in restore_names:
                previous = saved[name]
                if previous is missing:
                    self.__dict__.pop(name, None)
                else:
                    setattr(self, name, previous)
            if not accepted:
                self._component_access.committed_world_changed()

    # ------------------------------------------------------- via ARRAY geometry
    def _site_n(self, v: _ViaSite) -> Optional[int]:
        """Resolve the stored site model consistently for geometry consumers.

        A single-cut model is authorized only when the selected policy is one;
        policy admission is enforced at candidate and emission boundaries.
        """
        return 1 if getattr(v, "n", 0) == 1 else None

    def _single_via_escape_allowed(self) -> bool:
        """Whether a single-cut fallback is authorized under a redundant via policy."""
        return self.via_redundancy > 1 and bool(
            getattr(self, "allow_single_via_escape", False)
        )

    def _require_policy_via(self, via: _ViaSite) -> None:
        """Reject route-owned via models outside the selected array policy."""
        model = getattr(via, "n", 0)
        allowed_models = (
            (0, 1, self.via_redundancy)
            if self._single_via_escape_allowed()
            else (0, self.via_redundancy)
        )
        if type(model) is not int or model not in allowed_models:
            raise ValueError("router-owned via differs from declared redundancy")

    @staticmethod
    def _via_search_identity(
        x: int,
        y: int,
        lo: int,
        n: Optional[int],
    ) -> Tuple[int, int, int, int]:
        """Exact site/model identity used only by conflict-directed search."""
        return (int(x), int(y), int(lo), 1 if n == 1 else 0)

    def _via_pad_half_for(self, v: _ViaSite) -> Tuple[int, int]:
        """Landing-pad halves shared by legality, occupancy, and emission."""
        if getattr(v, "n", 0) == 1 and self.via_redundancy > 1:
            return self._via_pad_half1[v.lo]
        return self._via_pad_half[v.lo]

    def _via_cut_extent(self, lo: int, n: Optional[int] = None) -> int:
        """Half-side (nm) of the via ARRAY's OUTER cut bounding box at level ``lo``:
        ``(N-1)*pitch/2 + cut/2``. The legality / occupancy / residual MODEL treats
        the whole redundant array as ONE conservative cut blob of this extent (it
        OVER-reserves the internal array gaps -- never under-reserves -- so it can
        never mask a cross-net short; the emitted artwork draws the real discrete
        cuts, ``_via_polys``). N=1 -> cut/2 (a single cut), byte-identical, and it
        never touches ``_via_pitch`` so hand-built engine states stay valid.
        ``n`` overrides the count for a single-via escape site (None -> default)."""
        n = self.via_redundancy if n is None else n
        ch = (self._via_cut[lo] + 1) // 2
        if n <= 1:
            return ch
        arr = (n - 1) * self._via_pitch[lo] // 2
        return arr + ch

    def _via_cut_centers(self, x: int, y: int, lo: int,
                         n: Optional[int] = None) -> List[Tuple[int, int]]:
        """The N x N cut centres of the redundant via array centred on site
        ``(x, y)`` at level ``lo``: integer, symmetric about the site (even pitch),
        adjacent centres exactly ``via{k}_space + cut`` apart so cut edges clear the
        via-space rule. N=1 -> ``[(x, y)]`` (a single via), byte-identical.
        ``n`` overrides the count for a single-via escape site (None -> default)."""
        N = self.via_redundancy if n is None else n
        if N <= 1:
            return [(x, y)]
        p = self._via_pitch[lo]
        base = -((N - 1) * p) // 2
        offs = [base + k * p for k in range(N)]
        return [(x + ox, y + oy) for ox in offs for oy in offs]

    # ------------------------------------------------------------- commit / rip
    def _commit(self, book: _Book, committed: Dict[str, _RoutedNet],
                rn: _RoutedNet) -> None:
        _RoutedNet = _eg_attr("_RoutedNet", _base_RoutedNet)
        metal_by_gi, cuts = self._net_metal(rn)
        book.add(rn.name, metal_by_gi, cuts, self._net_via_pads(rn))
        committed[rn.name] = rn

    def _net_via_pads(self, rn: _RoutedNet, *, checkpoint=None) -> Dict[int, List[Rect]]:
        """Lower and upper landing-pad rectangles of ``rn``'s emitted vias."""
        _RoutedNet = _eg_attr("_RoutedNet", _base_RoutedNet)
        pads: Dict[int, List[Rect]] = {}
        for index, v in enumerate(rn.vias):
            if checkpoint is not None and index % 256 == 0:
                checkpoint()
            self._require_policy_via(v)
            bh, ah = self._via_pad_half_for(v)
            pads.setdefault(v.lo, []).append(
                (v.x - bh, v.y - bh, v.x + bh, v.y + bh))
            pads.setdefault(v.lo + 1, []).append(
                (v.x - ah, v.y - ah, v.x + ah, v.y + ah))
        return pads

    def _net_metal(self, rn: _RoutedNet, *, checkpoint=None) -> Tuple[Dict[int, List[Rect]], List[Tuple[Rect, int]]]:
        """The committed clearance rects (per gi) + via cuts for a routed net."""
        _RoutedNet = _eg_attr("_RoutedNet", _base_RoutedNet)
        metal: Dict[int, List[Rect]] = {}
        cuts: List[Tuple[Rect, int]] = []
        for index, s in enumerate(rn.segs):
            if checkpoint is not None and index % 256 == 0:
                checkpoint()
            metal.setdefault(s.gi, []).append(
                _seg_bbox(
                    s.ax,
                    s.ay,
                    s.bx,
                    s.by,
                    self._segment_half_width(rn.name, s),
                )
            )
        for index, v in enumerate(rn.vias):
            if checkpoint is not None and index % 256 == 0:
                checkpoint()
            bh, ah = self._via_pad_half_for(v)
            metal.setdefault(v.lo, []).append(
                (v.x - bh, v.y - bh, v.x + bh, v.y + bh))
            metal.setdefault(v.lo + 1, []).append(
                (v.x - ah, v.y - ah, v.x + ah, v.y + ah))
            # the redundant array's OUTER cut bbox (a single cut when N=1 or a
            # single-via escape site): the conservative net-blind cut-spacing
            # participant (see _via_cut_extent).
            ext = self._via_cut_extent(v.lo, self._site_n(v))
            cuts.append(((v.x - ext, v.y - ext, v.x + ext, v.y + ext), v.lo))
        return metal, cuts

    def _occupancy(self, committed: Dict[str, _RoutedNet],
                   ) -> Dict[Tuple[int, int, int], Set[str]]:
        """Rasterise every CONNECTED net's metal (grown by half min-space) into the
        coarse resource grid -> {resource_cell: set(owning nets)}. Two nets sharing a
        cell are within min-space on that layer: a resource cell with >= 2 owners is
        OVER-USED (a short OR a sub-min gap) and drives history escalation. This is
        the negotiation's own advisory over-use signal (the independent KLayout
        oracle is the real verdict)."""
        _RoutedNet = _eg_attr("_RoutedNet", _base_RoutedNet)
        rc = self._res_cell
        occ: Dict[Tuple[int, int, int], Set[str]] = {}
        for name, rn in committed.items():
            self._watchdog_check(name, "occupancy")
            if not rn.connected:
                continue
            metal, _cuts = self._net_metal(rn)
            for gi, rects in metal.items():
                g = self._space[gi] // 2
                for (x0, y0, x1, y1) in rects:
                    for bx in range((x0 - g) // rc, (x1 + g) // rc + 1):
                        for by in range((y0 - g) // rc, (y1 + g) // rc + 1):
                            c = (bx, by, gi)
                            s = occ.get(c)
                            if s is None:
                                occ[c] = s = set()
                            s.add(name)
            # Via cuts are capacity resources too.  Give each via level a
            # disjoint resource layer above the routing-metal indices and grow
            # each cut by half the deck's net-blind cut-space requirement.  Two
            # cuts violating that requirement therefore share at least one
            # resource cell and accumulate the SAME persistent history consumed
            # by ``_via_cut_congestion`` on the next round.
            for cut, lo in _cuts:
                g = self._via_space[lo] // 2
                x0, y0, x1, y1 = _norm(cut)
                resource_gi = self._n_layers + lo
                for bx in range((x0 - g) // rc, (x1 + g) // rc + 1):
                    for by in range((y0 - g) // rc, (y1 + g) // rc + 1):
                        c = (bx, by, resource_gi)
                        owners = occ.get(c)
                        if owners is None:
                            occ[c] = owners = set()
                        owners.add(name)
        return occ

    # ------------------------------------------------------------- legality (HARD)
    def _record_hard_rejection(
        self, category: str, owner: Optional[str] = None,
        *,
        gi: Optional[int] = None,
        candidate_rect: Optional[Rect] = None,
        candidate_half: Optional[int] = None,
        required_space: Optional[int] = None,
        blocker_rect: Optional[Rect] = None,
        blocker_kind: Optional[str] = None,
        terminal_conductor_ref: Optional[str] = None,
    ) -> None:
        """Record deterministic causal diagnostics; never alter route choice.

        Rejected candidate rectangles are aggregated by their exact causal
        blocker identity.  The complete blocker set is therefore retained
        without a search-size-dependent witness list or an arbitrary slice.
        """
        table = getattr(self, "_hard_rejections", None)
        if table is None:
            return
        owner_id = str(owner) if owner is not None else "<unowned>"
        key = (str(category), owner_id)
        table[key] = table.get(key, 0) + 1
        self._hard_rejection_generation = (
            getattr(self, "_hard_rejection_generation", 0) + 1
        )
        samples = getattr(self, "_hard_rejection_samples", None)
        if samples is None or gi is None or candidate_rect is None:
            return
        cause = (
            str(category),
            owner_id,
            int(gi),
            None if candidate_half is None else int(candidate_half),
            None if required_space is None else int(required_space),
            None if blocker_rect is None else tuple(_norm(blocker_rect)),
            None if blocker_kind is None else str(blocker_kind),
            (
                None
                if terminal_conductor_ref is None
                else str(terminal_conductor_ref)
            ),
        )
        candidate = tuple(_norm(candidate_rect))
        previous = samples.get(cause)
        if previous is None:
            samples[cause] = (1, candidate)
            return
        count, extent = previous
        samples[cause] = (
            int(count) + 1,
            (
                min(extent[0], candidate[0]),
                min(extent[1], candidate[1]),
                max(extent[2], candidate[2]),
                max(extent[3], candidate[3]),
            ),
        )

    def _get_native_history_grid(self, cpp: Any, *, force_fresh: bool = False) -> Any:
        """Build or fetch an exact NativeHistoryGrid when self._history is a plain dict."""
        history = getattr(self, "_history", None)
        if not history or type(history) is not dict:
            return None
        hist_fac = float(getattr(self, "hist_fac", 1.0))
        if not force_fresh:
            key = (
                id(history),
                len(history),
                int(getattr(self, "_history_generation", 0)),
                hist_fac,
            )
            cached = _NATIVE_HISTORY_GRID_CACHE.get(self)
            if cached is not None and cached[0] == key:
                return cached[1]
        try:
            hgrid = cpp.NativeHistoryGrid(history, hist_fac, _COST_FACTOR_SCALE)
        except (OverflowError, TypeError, ValueError):
            return None
        if not force_fresh:
            _NATIVE_HISTORY_GRID_CACHE[self] = (key, hgrid)
        return hgrid

    def _flush_active_edge_ctx(self) -> None:
        """Drain buffered C++ NativeEdgeContext rejections into Python evidence tables."""
        ctx = getattr(self, "_active_edge_ctx", None)
        if ctx is None:
            return
        if ctx.take_mirror_conflict_seen():
            self._mirror_pair_conflict_seen = True
        bumped = int(getattr(self, "_active_edge_ctx_gen_bumped", 0))
        if ctx.pending_rej_count <= 0 and bumped == 0:
            return
        agg_rejs = ctx.flush_rejections()
        self._active_edge_ctx_gen_bumped = 0
        if bumped and getattr(self, "_hard_rejections", None) is not None:
            self._hard_rejection_generation = max(
                0,
                int(getattr(self, "_hard_rejection_generation", 0)) - bumped,
            )
        if agg_rejs:
            cpp = getattr(self, "_active_edge_ctx_cpp", None) or load_native("_gridless_cpp")
            if cpp is not None:
                self._merge_cpp_via_batch_rejections(agg_rejs, cpp)

    def _hard_rejection_evidence(self) -> Tuple[Tuple[str, str, int], ...]:
        if getattr(self, "_active_edge_ctx", None) is not None:
            self._flush_active_edge_ctx()
        table = getattr(self, "_hard_rejections", {})
        return tuple(
            (category, owner, int(count))
            for (category, owner), count in sorted(table.items())
        )

    def _hard_rejection_sample_evidence(
        self,
    ) -> Tuple[Tuple[Tuple[str, Any], ...], ...]:
        """Complete causal-blocker aggregates for the failing edge/drop."""
        if getattr(self, "_active_edge_ctx", None) is not None:
            self._flush_active_edge_ctx()
        samples = getattr(self, "_hard_rejection_samples", {})
        evidence = []
        for cause, (count, candidate_extent) in sorted(samples.items()):
            (
                category,
                owner,
                gi,
                candidate_half,
                required_space,
                blocker_rect,
                blocker_kind,
                terminal_ref,
            ) = cause
            evidence.append(
                (
                    ("category", category),
                    ("owner", owner),
                    ("gi", int(gi)),
                    ("candidate_rect", tuple(candidate_extent)),
                    ("candidate_half", candidate_half),
                    ("rejection_count", int(count)),
                    ("required_space", required_space),
                    ("blocker_rect", blocker_rect),
                    ("blocker_kind", blocker_kind),
                    ("terminal_conductor_ref", terminal_ref),
                )
            )
        return tuple(evidence)

    def _replay_static_legality_snapshot(
        self,
        snapshot: Tuple[Any, ...],
    ) -> bool:
        # A proof's evidence is immutable.  In large frontiers, re-summing it
        # on every cache hit costs more than the cached legality query itself.
        # Preserve compatibility with snapshots in older diagnostic pickles.
        if len(snapshot) == 3:
            legal, rejections, samples = snapshot
            rejection_count = _eg_attr("sum", sum)(int(count) for _cause, count in rejections)
        else:
            legal, rejections, samples, rejection_count = snapshot
        self._merge_component_access_rejections(rejections, samples)
        if rejection_count:
            self._hard_rejection_generation = int(getattr(
                self, "_hard_rejection_generation", 0
            )) + rejection_count
        return bool(legal)

    # Search-state budgets are not memory allowances.  These two proof LRUs
    # may otherwise retain millions of detailed causal snapshots each.  An
    # eviction only requires the same exact legality computation again; no
    # geometry, rejection evidence, or search state is omitted.
    _STATIC_LEGALITY_CACHE_MAX_ENTRIES = 250_000

    def _cached_static_legality(
        self,
        cache: OrderedDict,
        key: Tuple[Any, ...],
        compute,
        *,
        hit_counter: str,
        miss_counter: str,
    ) -> bool:
        """Cache one immutable DRC proof and replay its causal evidence exactly."""
        cached = cache.pop(key, None)
        if cached is not None:
            cache[key] = cached
            setattr(self, hit_counter, int(getattr(self, hit_counter, 0)) + 1)
            return self._replay_static_legality_snapshot(cached)

        setattr(self, miss_counter, int(getattr(self, miss_counter, 0)) + 1)
        outer_rejections = getattr(self, "_hard_rejections", {})
        outer_samples = getattr(self, "_hard_rejection_samples", {})
        outer_generation = int(getattr(
            self, "_hard_rejection_generation", 0
        ))
        self._hard_rejections = {}
        self._hard_rejection_samples = {}
        try:
            legal = bool(compute())
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
        finally:
            self._hard_rejections = outer_rejections
            self._hard_rejection_samples = outer_samples
            self._hard_rejection_generation = outer_generation
        rejection_count = _eg_attr("sum", sum)(int(count) for _cause, count in rejections)
        snapshot = (legal, rejections, samples, rejection_count)
        cache[key] = snapshot
        capacity = max(1, min(
            int(self.max_expansions),
            int(self._STATIC_LEGALITY_CACHE_MAX_ENTRIES),
        ))
        while len(cache) > capacity:
            cache.popitem(last=False)
        return self._replay_static_legality_snapshot(snapshot)

    def _clear_obstacles(
        self,
        gi: int,
        box: Rect,
        sp: int,
        net: str,
        *,
        candidate_half: Optional[int] = None,
    ) -> bool:
        """True iff ``box`` keeps ``sp`` clearance from every FOREIGN device OBSTACLE
        (fixed artwork) on ``gi``. Solve-local committed metal is checked separately:
        exact electrical contact is hard in
        ``_reject_foreign_committed_metal_contact``, while only its positive
        min-space band remains a soft ``_box_congestion`` cost.

        OWN-net artwork is deliberately NOT hard-checked. The deck's net-blind
        space() makes the OPEN band (0, min_space) next to own artwork a REAL
        violation (the measured R7a M2.S.1/M3.S.1 storey-by-its-own-tab family),
        but HARD-banding it SEALS real pin accesses -- the R6 gnd/vout edges
        exhausted their whole search graph (1-2M states, no legal path) because
        every approach to a pin offset from its own pad crossed the band. The
        band is therefore SOFT: priced in ``_box_congestion`` (escalating with
        ``pres_fac``) so the touch-or-clear alternative wins under negotiation,
        and judged EXACTLY at convergence (``_detect_space_residuals``) -- never
        a silent pass, never a seal.

        A rect outside ``box`` grown by ``sp`` is clear on some axis by
        construction (spatial-index local query)."""
        _rects_clear = _eg_attr("_rects_clear", _base_rects_clear)
        x0, y0, x1, y1 = _norm(box)
        clear = True
        oidx = self._obs_idx.get(gi)
        if (
            _rects_clear is _base_rects_clear
            and "_static_or_direct_obstacle_scan" not in self.__dict__
            and type(self)._static_or_direct_obstacle_scan
            is _GridlessLegalityMixin._static_or_direct_obstacle_scan
            and type(oidx) is _SpatialIndex
            and oidx._native is not None
        ):
            conflicts = oidx._native.scan_obstacle_conflicts(
                x0, y0, x1, y1, sp, net
            )
            if conflicts:
                obs_prov = getattr(self, "_obs_provenance", {})
                for owner, r in conflicts:
                    kind, terminal_ref = obs_prov.get(
                        (gi, owner, _norm(r)), (None, None)
                    )
                    self._record_hard_rejection(
                        "fixed_metal",
                        owner,
                        gi=gi,
                        candidate_rect=box,
                        candidate_half=candidate_half,
                        required_space=sp,
                        blocker_rect=r,
                        blocker_kind=kind,
                        terminal_conductor_ref=terminal_ref,
                    )
                    clear = False
        else:
            q = (x0 - sp, y0 - sp, x1 + sp, y1 + sp)
            for (owner, r) in self._static_or_direct_obstacle_scan(gi, q):
                if owner == net:
                    continue
                if not _rects_clear(box, r, sp):
                    kind, terminal_ref = getattr(
                        self, "_obs_provenance", {}
                    ).get((gi, owner, _norm(r)), (None, None))
                    self._record_hard_rejection(
                        "fixed_metal",
                        owner,
                        gi=gi,
                        candidate_rect=box,
                        candidate_half=candidate_half,
                        required_space=sp,
                        blocker_rect=r,
                        blocker_kind=kind,
                        terminal_conductor_ref=terminal_ref,
                    )
                    clear = False
        for owner, blocker, required, kind in self._wide_spacing_conflicts(
            gi, box, net, include_committed=False,
        ):
            self._record_hard_rejection(
                "fixed_wide_metal", owner, gi=gi, candidate_rect=box,
                candidate_half=candidate_half, required_space=required,
                blocker_rect=blocker, blocker_kind=kind,
            )
            clear = False
        return clear

    def _ensure_fixed_wide_geometry_indexes(self) -> None:
        """Derive once from the immutable literal world, including replay callers."""
        tech = getattr(self, "_tech", None)
        key = (id(tech), id(getattr(self, "_fixed_deck_literal_rects", None)),
               id(getattr(self, "_abstract_rects", None)),
               id(getattr(self, "_literal_planar_idx", None)))
        if getattr(self, "_fixed_wide_source_key", None) == key:
            return
        from .wide_metal_spacing import geometry_spacing_entries
        self._wide_geometry_entries_by_gi = (
            {
                gi: geometry_spacing_entries(tech.pdk, tech.metal_names[tech._gi_to_pdk[gi]])
                for gi in range(tech.n_layers)
            }
            if tech is not None and getattr(tech, "pdk", None) is not None
            else {}
        )
        self._min_wide_ge_by_gi = {
            gi: min(int(b.wide_ge) for b in bands)
            for gi, bands in self._wide_geometry_entries_by_gi.items()
            if bands
        }
        self._fixed_wide_geometry_indexes = self._build_wide_geometry_indexes({})
        self._fixed_wide_source_key = key
        self._wide_index_metal_index = None

    def _fixed_literal_metal_rows(self):
        """Use the canonical claim-specific literal index, or captured literals.

        The planar input API accepts named layers as well as GDS keys; its
        literal index has already resolved both. The deck capture deliberately
        retains only GDS-key literals and is the fallback for historical replay.
        """
        indexes = getattr(self, "_literal_planar_idx", {})
        if indexes:
            return {gi: tuple(sorted({(owner, tuple(rectangle))
                    for bucket in index.buckets.values() for owner, rectangle in bucket},
                    key=lambda row: (str(row[0]), row[1])))
                    for gi, index in indexes.items()}
        tech = getattr(self, "_tech", None)
        result = {}
        if tech is not None:
            abstract_rects = getattr(self, "_abstract_rects", {})
            for layer, owner, rectangle in getattr(self, "_fixed_deck_literal_rects", ()):
                gi = tech.resolve_layer_to_gi(tuple(layer))
                if gi is not None and rectangle not in abstract_rects.get(gi, ()):
                    result.setdefault(gi, []).append((owner, rectangle))
        return result

    def _build_wide_geometry_indexes(self, committed_rows):
        """Full merge/opening only at fixed-world or committed-index boundaries."""
        import klayout.db as kdb
        from .wide_metal_spacing import exact_region_rectangles, wide_metal_regions

        tech = getattr(self, "_tech", None)
        if tech is None:
            return {}
        fixed_rows_by_gi = self._fixed_literal_metal_rows()
        fixed_indexes = (
            getattr(self, "_fixed_wide_geometry_indexes", None)
            if committed_rows
            else None
        )
        wide_entries_by_gi = getattr(self, "_wide_geometry_entries_by_gi", None)
        all_gis = sorted(set(fixed_rows_by_gi) | set(committed_rows))
        result = {}
        for gi in all_gis:
            if wide_entries_by_gi is not None and not wide_entries_by_gi.get(gi):
                continue
            c_rows = committed_rows.get(gi, ())
            if fixed_indexes is not None and not c_rows:
                if gi in fixed_indexes:
                    result[gi] = fixed_indexes[gi]
                continue
            rows = [*fixed_rows_by_gi.get(gi, ()), *c_rows]
            if not rows:
                continue
            by_owner = {}
            drawn = kdb.Region()
            for owner, rectangle in rows:
                box = kdb.Box(*_norm(rectangle))
                by_owner.setdefault(owner, kdb.Region()).insert(box)
                drawn.insert(box)
            drawn.merge()
            layer_name = tech.metal_names[tech._gi_to_pdk[gi]]
            bands = wide_metal_regions(drawn, tech.pdk, layer_name)
            entries = []
            for band, core, window, attached in bands:
                if core.is_empty():
                    continue
                core_index, window_index, attached_index = (_SpatialIndex(), _SpatialIndex(), _SpatialIndex())
                for rectangle in exact_region_rectangles(window):
                    window_index.add(None, rectangle)
                for owner, metal in sorted(by_owner.items(), key=lambda row: (row[0] is not None, str(row[0]))):
                    for rectangle in exact_region_rectangles(core & metal):
                        core_index.add(owner, rectangle)
                    for rectangle in exact_region_rectangles(attached & metal):
                        attached_index.add(owner, rectangle)
                entries.append((int(band.spacing.value), core_index, window_index, attached_index))
            if entries:
                result[gi] = tuple(entries)
        self._wide_geometry_index_builds = int(getattr(self, "_wide_geometry_index_builds", 0)) + 1
        return result

    def _wide_indexes(self, include_committed: bool):
        self._ensure_fixed_wide_geometry_indexes()
        if not include_committed:
            return self._fixed_wide_geometry_indexes
        # The ordinary metal-index object is replaced on each rebuild and
        # restored by existing transactions. Identity therefore invalidates a
        # stale derived view after either a commit or a rollback, without a
        # full-world operation on subsequent edge queries.
        metal_index = getattr(self, "_metal_idx", None)
        if self._wide_index_metal_index is not metal_index:
            book = getattr(self, "_book", None)
            rows = getattr(book, "metal", {})
            self._wide_geometry_indexes = (self._build_wide_geometry_indexes(rows)
                if any(rows.values()) else self._fixed_wide_geometry_indexes)
            self._wide_index_metal_index = metal_index
        return self._wide_geometry_indexes

    def _wide_spacing_neighbors(self, gi: int, box: Rect, *, include_committed: bool):
        """Local literal pieces and exact canonical spacing, no region kernels."""
        self._ensure_fixed_wide_geometry_indexes()
        min_w = self._min_wide_ge_by_gi.get(gi)
        if min_w is None:
            return ()
        entries = (
            self._fixed_wide_geometry_indexes
            if not include_committed
            else self._wide_indexes(True)
        ).get(gi, ())
        candidate = (
            box
            if (box[0] <= box[2] and box[1] <= box[3])
            else _norm(box)
        )
        width = min(candidate[2] - candidate[0], candidate[3] - candidate[1])
        if not entries and width < min_w:
            return ()
        found = set()
        for space, core_index, window_index, attached_index in entries:
            q = (candidate[0] - space, candidate[1] - space,
                 candidate[2] + space, candidate[3] + space)
            for owner, blocker in core_index.query(q):
                found.add((owner, blocker, space, "wide_core", candidate))
            # Magic's attached rule applies only to actual candidate material
            # inside a wide-core halo, not to the bounding box of that core.
            for _owner, window in window_index.query(candidate):
                part = (max(candidate[0], window[0]), max(candidate[1], window[1]),
                        min(candidate[2], window[2]), min(candidate[3], window[3]))
                if part[0] >= part[2] or part[1] >= part[3]:
                    continue
                local = (part[0] - space, part[1] - space, part[2] + space, part[3] + space)
                for owner, blocker in attached_index.query(local):
                    found.add((owner, blocker, space, "wide_attached_halo", part))
            # Magic widespacing also extends up to `space` along any attached
            # corner step flush with an outer wide-core boundary face.
            for owner, blocker in attached_index.query(q):
                bx0, by0, bx1, by1 = blocker
                for co, (cx0, cy0, cx1, cy1) in core_index.query((bx0 - 1, by0 - 1, bx1 + 1, by1 + 1)):
                    if co != owner:
                        continue
                    flush_specs = []
                    if (bx1 == cx0 or bx0 == cx1) and by0 >= cy0 and by1 <= cy1 and (bx1 - bx0) <= space:
                        flush_specs.extend((
                            (by1 == cy1, (bx0, by1 + 1, bx1, by1 + 2), (bx0, by1, bx1, by1),
                             (candidate[0], max(candidate[1], by1), candidate[2], candidate[3])),
                            (by0 == cy0, (bx0, by0 - 2, bx1, by0 - 1), (bx0, by0, bx1, by0),
                             (candidate[0], candidate[1], candidate[2], min(candidate[3], by0))),
                        ))
                    if (by1 == cy0 or by0 == cy1) and bx0 >= cx0 and bx1 <= cx1 and (by1 - by0) <= space:
                        flush_specs.extend((
                            (bx0 == cx0, (bx0 - 2, by0, bx0 - 1, by1), (bx0, by0, bx0, by1),
                             (candidate[0], candidate[1], min(candidate[2], bx0), candidate[3])),
                            (bx1 == cx1, (bx1 + 1, by0, bx1 + 2, by1), (bx1, by0, bx1, by1),
                             (max(candidate[0], bx1), candidate[1], candidate[2], candidate[3])),
                        ))
                    for is_flush, probe, edge_box, out in flush_specs:
                        if not is_flush or out[0] >= out[2] or out[1] >= out[3]:
                            continue
                        if not any(co2 == owner for co2, _ in core_index.query(probe)):
                            found.add((owner, edge_box, space, "wide_core", out))
        # A candidate that itself is literally a wide rectangle owes the wider
        # clearance in the reverse routing order as well. This is a rectangle's
        # known dimension, never a width inferred from a merged component bbox.
        if width >= min_w:
            for band in self._wide_geometry_entries_by_gi[gi]:
                if width < band.wide_ge:
                    continue
                space = int(band.spacing.value)
                query = (candidate[0] - space, candidate[1] - space,
                         candidate[2] + space, candidate[3] + space)
                indexes = [self._literal_planar_idx.get(gi)]
                if include_committed:
                    indexes.append(getattr(self, "_metal_idx", {}).get(gi))
                for index in indexes:
                    if index is not None:
                        for owner, blocker in index.query(query):
                            found.add((owner, blocker, space, "wide_candidate", candidate))
        return tuple(sorted(found, key=lambda row: (str(row[0]), row[1], row[2], row[3], row[4])))

    def _wide_spacing_conflicts(self, gi, box, net, *, include_committed):
        _rects_clear = _eg_attr("_rects_clear", _base_rects_clear)
        return tuple((owner, blocker, space, kind)
            for owner, blocker, space, kind, part in self._wide_spacing_neighbors(
                gi, box, include_committed=include_committed)
            if owner != net and not _rects_clear(part, blocker, space))

    def _wide_repair_preserves_world(self, net, before_metal, after_metal) -> bool:
        """Judge a complete actual repair against fixed and foreign metal.

        The owning route is supplied explicitly: the committed book may still
        contain its pre-repair geometry while several legalization phases stage
        fills. New errors on inherited fixed edges count too. Existing unrelated
        errors are retained as obligations, not used to reject every repair.
        """
        import klayout.db as kdb
        from .wide_metal_spacing import geometry_spacing_entries, wide_spacing_checks

        tech = getattr(self, "_tech", None)
        if tech is None:
            return True
        fixed = self._fixed_literal_metal_rows()
        book_metal = getattr(getattr(self, "_book", None), "metal", {})
        for gi in sorted(set(before_metal) | set(after_metal)):
            name = tech.metal_names[tech._gi_to_pdk[gi]]
            if not geometry_spacing_entries(tech.pdk, name):
                continue
            self._watchdog_check(net, "wide_repair_acceptance")
            shared = kdb.Region()
            for _owner, rectangle in fixed.get(gi, ()):
                shared.insert(kdb.Box(*_norm(rectangle)))
            for owner, rectangle in book_metal.get(gi, ()):
                if owner != net:
                    shared.insert(kdb.Box(*_norm(rectangle)))
            before, after = shared.dup(), shared.dup()
            for rectangle in before_metal.get(gi, ()):
                before.insert(kdb.Box(*_norm(rectangle)))
            for rectangle in after_metal.get(gi, ()):
                after.insert(kdb.Box(*_norm(rectangle)))
            before.merge()
            after.merge()
            if (before ^ after).is_empty():
                continue
            old_checks = wide_spacing_checks(before, tech.pdk, name)
            new_checks = wide_spacing_checks(after, tech.pdk, name)
            if any(not (new.errors.polygons(0) - old.errors.polygons(0)).is_empty()
                   for old, new in zip(old_checks, new_checks)):
                return False
        return True

    def _mim_metal_changes(self, net, before_metal, after_metal):
        """New MIM errors in one complete proposed route/world replacement.

        Ownership is used only to replace the old current-net ledger.  The
        geometric predicate receives the complete literal union and no labels.
        There is deliberately no partial-via or static-cache version: a real
        bridge in this completed candidate can discharge a local obligation.
        """
        import klayout.db as kdb
        from .mim_metal_spacing import mim_metal_spacing_checks

        tech = getattr(getattr(self, "_tech", None), "pdk", None)
        query = getattr(tech, "mim_metal_geometry_rules", None)
        rules = tuple(query()) if callable(query) else ()
        if not rules:
            return ()
        changed_gis = set(before_metal) | set(after_metal)
        metal_keys = {tuple(self._tech.gi_to_metal_key(gi)): gi for gi in changed_gis}
        active_rules = tuple(rule for rule in rules if rule.metal_layer in metal_keys)
        if not active_rules:
            return ()
        fixed_rects = getattr(self, "_fixed_deck_literal_rects", ())
        cache = getattr(self, "_mim_fixed_regions_cache", None)
        if cache is None or cache[0] != id(fixed_rects) or len(cache) < 3:
            all_keys = {k for r in rules for k in (r.metal_layer, r.marker_layer)}
            fixed_map = {k: kdb.Region() for k in all_keys}
            for layer, _owner, rect in fixed_rects:
                t_layer = tuple(layer)
                if t_layer in fixed_map:
                    fixed_map[t_layer].insert(kdb.Box(*_norm(rect)))
            for reg in fixed_map.values():
                reg.merge()
            rule_halos = {}
            for r in rules:
                rad = int(getattr(r, "grow_nm", 0) or 0) + int(r.spacing_nm) + 500
                boxes = []
                for poly in fixed_map[r.marker_layer].each():
                    pb = poly.bbox()
                    boxes.append((pb.left - rad, pb.bottom - rad, pb.right + rad, pb.top + rad))
                rule_halos[r.rule_id] = tuple(boxes)
            cache = (id(fixed_rects), fixed_map, rule_halos)
            self._mim_fixed_regions_cache = cache
        fixed_map = cache[1]
        rule_halos = cache[2]
        if all(fixed_map[rule.marker_layer].is_empty() for rule in active_rules):
            return ()
        pure_addition = not any(before_metal.values())
        if pure_addition:
            near_rules = []
            for rule in active_rules:
                halos = rule_halos.get(rule.rule_id, ())
                if not halos:
                    continue
                m_gi = metal_keys[rule.metal_layer]
                prop_boxes = after_metal.get(m_gi, ())
                hit = False
                for raw_b in prop_boxes:
                    bx0, by0, bx1, by1 = _norm(raw_b)
                    for hx0, hy0, hx1, hy1 in halos:
                        if bx0 <= hx1 and bx1 >= hx0 and by0 <= hy1 and by1 >= hy0:
                            hit = True
                            break
                    if hit:
                        break
                if hit:
                    near_rules.append(rule)
            if not near_rules:
                return ()
            active_rules = tuple(near_rules)
        self._watchdog_check(net, "mim_metal_acceptance")
        keys = {key for rule in active_rules for key in (rule.metal_layer, rule.marker_layer)}
        book_metal_map = getattr(getattr(self, "_book", None), "metal", {})
        base_cache = getattr(self, "_mim_baseline_checks_cache", None)
        if base_cache is None:
            base_cache = OrderedDict()
            self._mim_baseline_checks_cache = base_cache
        base_key = None
        if pure_addition:
            book_sig = tuple(
                (int(gi), len(rows), rows[-1] if rows else None)
                for gi, rows in sorted(book_metal_map.items())
                if tuple(self._tech.gi_to_metal_key(gi)) in keys
            )
            base_key = (
                id(fixed_rects),
                str(net),
                tuple(sorted(keys)),
                book_sig,
                int(getattr(self, "_committed_index_generation", 0)),
            )
        cached_base = base_cache.get(base_key) if base_key is not None else None
        if cached_base is not None:
            before, old_checks = cached_base
            after = {key: region.dup() for key, region in before.items()}
            for gi, boxes in after_metal.items():
                key = tuple(self._tech.gi_to_metal_key(gi))
                if key in after:
                    for box in boxes:
                        after[key].insert(kdb.Box(*_norm(box)))
            for region in after.values():
                region.merge()
        else:
            shared = {key: fixed_map[key].dup() for key in keys}
            for gi, rows in book_metal_map.items():
                key = tuple(self._tech.gi_to_metal_key(gi))
                if key not in shared:
                    continue
                for owner, rect in rows:
                    if owner != net:
                        shared[key].insert(kdb.Box(*_norm(rect)))
            before = {key: region.dup() for key, region in shared.items()}
            after = {key: region.dup() for key, region in shared.items()}
            for world, route_metal in ((before, before_metal), (after, after_metal)):
                for gi, boxes in route_metal.items():
                    key = tuple(self._tech.gi_to_metal_key(gi))
                    if key in world:
                        for box in boxes:
                            world[key].insert(kdb.Box(*_norm(box)))
                for region in world.values():
                    region.merge()
            old_checks = None
        if all((before[rule.metal_layer] ^ after[rule.metal_layer]).is_empty()
               for rule in active_rules):
            return ()
        if old_checks is None:
            old_checks = mim_metal_spacing_checks(before, tech)
            if base_key is not None:
                base_cache[base_key] = (before, old_checks)
                while len(base_cache) > 8:
                    base_cache.popitem(last=False)
        new_checks = mim_metal_spacing_checks(after, tech)
        changes = []
        for old, new in zip(old_checks, new_checks):
            extra = (new.errors.polygons(0) - old.errors.polygons(0)).merged()
            if not extra.is_empty():
                changes.append((new, extra))
        self._watchdog_check(net, "mim_metal_acceptance")
        return tuple(changes)

    def _repair_preserves_world(self, net, before_metal, after_metal) -> bool:
        """Check complete repair unions against both wide-metal and MIM rules."""
        return (self._wide_repair_preserves_world(net, before_metal, after_metal)
                and not self._mim_metal_changes(net, before_metal, after_metal))

    def _ordinary_repair_preserves_world(self, net, before_metal, after_metal) -> bool:
        """Keep final staged fills from adding ordinary merged-metal defects.

        Intermediate fills may be repaired by a later phase. Judge only the
        completed per-net batch against its original literal same-owner union;
        existing errors remain obligations rather than vetoing every repair.
        Foreign metal must not supply support for an owning-net junction.
        """
        import klayout.db as kdb

        if getattr(self, "_tech", None) is None:
            return True
        fixed = self._fixed_literal_metal_rows()
        for gi in sorted(set(before_metal) | set(after_metal)):
            if before_metal.get(gi, ()) == after_metal.get(gi, ()):
                continue
            self._watchdog_check(net, "ordinary_repair_acceptance")
            shared = kdb.Region()
            for owner, rectangle in fixed.get(gi, ()):
                if owner == net:
                    shared.insert(kdb.Box(*_norm(rectangle)))
            before, after = shared.dup(), shared.dup()
            for rectangle in before_metal.get(gi, ()):
                before.insert(kdb.Box(*_norm(rectangle)))
            for rectangle in after_metal.get(gi, ()):
                after.insert(kdb.Box(*_norm(rectangle)))
            before.merge()
            after.merge()
            if (before ^ after).is_empty():
                continue
            for check, distance in (
                ("width_check", 2 * self._half_w[gi]),
                ("space_check", self._space[gi]),
            ):
                # space_check covers both separated islands and notches. A
                # bridge can change an old gap into a smaller notch without
                # creating a new spacing defect; preserve that improvement.
                old = getattr(before, check)(int(distance)).polygons(0)
                new = getattr(after, check)(int(distance)).polygons(0)
                self._watchdog_check(net, "ordinary_repair_acceptance")
                if not (new - old).is_empty():
                    return False
        return True

    def _candidate_mim_metal_failure(self, net, segs, vias, *, prior=None):
        """Return causal MIM evidence for a complete candidate, never a model."""
        _RoutedNet = _eg_attr("_RoutedNet", _base_RoutedNet)
        if prior is None:
            prior = getattr(self, "_active_access_routed_net", None)
        before = (self._net_metal(prior)[0] if prior is not None else
                  {gi: tuple(boxes) for gi, boxes in getattr(self, "_attempt_metal", {}).items()})
        proposed = self._net_metal(_RoutedNet(str(net), segs=list(segs), vias=list(vias)))[0]
        after = {gi: tuple(boxes) for gi, boxes in before.items()}
        for gi, boxes in proposed.items():
            after[gi] = (*after.get(gi, ()), *boxes)
        changes = self._mim_metal_changes(net, before, after)
        for check, extra in changes:
            box = extra.bbox()
            self._record_hard_rejection(
                "mim_metal_clearance", None,
                gi=self._tech.resolve_layer_to_gi(check.metal_layer),
                candidate_rect=(box.left, box.bottom, box.right, box.top),
                required_space=check.spacing_nm, blocker_kind=check.rule_id,
            )
        return changes

    def _mim_candidate_retry_identities(self, net, segs, vias, changes):
        """Scope retry diversification to actual candidate material at the gap.

        These identities only populate the existing access-domain branch.  They
        are not cached physical prohibitions: complete geometry is judged again
        when another proposal retains a site but supplies a valid local bridge.
        """
        _RoutedNet = _eg_attr("_RoutedNet", _base_RoutedNet)
        import klayout.db as kdb
        affected = {}
        for check, extra in changes:
            gi = self._tech.resolve_layer_to_gi(check.metal_layer)
            affected.setdefault(gi, kdb.Region())
            # Include the actual facing edge, whose intersection with a metal
            # rectangle otherwise has zero area.  This one-nm diagnostic probe
            # never adds routing geometry or changes the rule distance.
            affected[gi] += extra.sized(1)
        via_identity = None
        for via in vias:
            metal = self._net_metal(_RoutedNet(str(net), vias=[via]))[0]
            if any(gi in affected and any(not (kdb.Region(kdb.Box(*box)) & affected[gi]).is_empty()
                                         for box in boxes) for gi, boxes in metal.items()):
                via_identity = self._via_search_identity(via.x, via.y, via.lo, self._site_n(via))
                break
        wire_identity = None
        for seg in segs:
            if seg.gi not in affected:
                continue
            box = _seg_bbox(seg.ax, seg.ay, seg.bx, seg.by, self._segment_half_width(net, seg))
            if not (kdb.Region(kdb.Box(*box)) & affected[seg.gi]).is_empty():
                wire_identity = self._wire_search_identity(net, seg)
                break
        return via_identity, wire_identity

    @contextmanager
    def _wide_repair_transaction(self, net: str, rn: _RoutedNet):
        """Allow staged phases, then publish only their exact checked union."""
        _RoutedNet = _eg_attr("_RoutedNet", _base_RoutedNet)
        missing = object()
        previous = getattr(self, "_active_wide_repair_route", missing)
        before_metal, _cuts = self._net_metal(rn)
        count = len(rn.segs)
        self._active_wide_repair_route = rn
        accepted = False
        try:
            yield
            after_metal, _cuts = self._net_metal(rn)
            accepted = (
                self._repair_preserves_world(net, before_metal, after_metal)
                and self._ordinary_repair_preserves_world(net, before_metal, after_metal)
            )
        finally:
            if not accepted:
                del rn.segs[count:]
            if previous is missing:
                self.__dict__.pop("_active_wide_repair_route", None)
            else:
                self._active_wide_repair_route = previous

    def _static_or_direct_obstacle_scan(
        self,
        gi: int,
        query: Rect,
    ) -> Tuple[Tuple[Optional[str], Rect], ...]:
        """Reuse immutable fixed-obstacle rows inside one atomic via check."""
        key = (int(gi), tuple(_norm(query)))
        active = getattr(self, "_active_static_obstacle_scans", None)
        if active is not None and key in active:
            return active[key]
        index = self._obs_idx.get(gi)
        rows = tuple(index.query(query)) if index is not None else ()
        if active is not None:
            active[key] = rows
        return rows

    def _reject_foreign_committed_metal_contact(
        self,
        gi: int,
        box: Rect,
        net: str,
        *,
        candidate_half: Optional[int] = None,
    ) -> bool:
        """Record and reject exact contact with solve-local foreign conductors.

        The committed-metal book contains ordinary wire and every via landing pad.
        Canonical via-pad provenance built with that book classifies a contact
        without a second spatial query. Only closed-rectangle contact
        (``_sep == (0, 0)``) is hard; the positive ``(0, min_space)`` band remains
        in ``_box_congestion`` for Pathfinder to price. The index and provenance
        set are immutable for one ``_route_edge`` search.
        """
        candidate = _norm(box)
        _present, violations = self._static_or_direct_committed_metal_scan(
            gi, candidate, net
        )
        contacts = tuple(
            item for item in violations if item[2].endswith("_contact")
        )
        for owner, blocker, category, blocker_kind in contacts:
            self._record_hard_rejection(
                category,
                owner,
                gi=gi,
                candidate_rect=candidate,
                candidate_half=candidate_half,
                required_space=0,
                blocker_rect=blocker,
                blocker_kind=blocker_kind,
            )
        return bool(contacts)

    def _reject_foreign_committed_metal_spacing(
        self,
        gi: int,
        metal: Rect,
        net: str,
        *,
        candidate_half: int,
        require_supra_minimum_endpoint: bool = False,
    ) -> bool:
        """Reject one metal atom against solve-local foreign spacing bands.

        When ``require_supra_minimum_endpoint`` is true, retain the negotiated
        soft band for an ordinary minimum-width planar pair. Canonical committed
        via pads and declared supra-minimum carriers retain their full clearance
        in either route order; a pad's owner wire width cannot waive pad spacing.
        """
        candidate = _norm(metal)
        sp = self._space[gi]
        rejected = False
        for owner, blocker, required, kind in self._wide_spacing_conflicts(
            gi, candidate, net, include_committed=True,
        ):
            self._record_hard_rejection(
                "committed_wide_metal_spacing", owner, gi=gi,
                candidate_rect=candidate, candidate_half=candidate_half,
                required_space=required, blocker_rect=blocker, blocker_kind=kind,
            )
            rejected = True
        _present, committed_violations = (
            self._static_or_direct_committed_metal_scan(gi, candidate, net)
        )
        for owner, blocker, category, blocker_kind in committed_violations:
            if category.endswith("_spacing"):
                if (
                    require_supra_minimum_endpoint
                    and blocker_kind != "via_pad"
                    and not self._persistent_metal_requires_hard_space(
                        gi, owner, blocker
                    )
                    and candidate_half <= self._half_w[gi]
                    and (
                        owner is None
                        or self._hw(str(owner), gi) <= self._half_w[gi]
                    )
                ):
                    continue
                self._record_hard_rejection(
                    category,
                    owner,
                    gi=gi,
                    candidate_rect=candidate,
                    candidate_half=candidate_half,
                    required_space=sp,
                    blocker_rect=blocker,
                    blocker_kind=blocker_kind,
                )
                rejected = True
        return rejected

    def _persistent_metal_requires_hard_space(
        self,
        gi: int,
        owner: Optional[str],
        rect: Rect,
    ) -> bool:
        """Return True when ``rect`` belongs to frozen persistent group metal."""
        book = getattr(self, "_book", None)
        if book is None or owner is None or owner not in book.persistent:
            return False
        if bool(getattr(self, "_active_group_route_local_closure", False)) or bool(
            getattr(self, "_negotiation_port_legalize", False)
        ):
            return True
        is_pers = getattr(book, "is_persistent_metal", None)
        return bool(is_pers(gi, owner, rect)) if callable(is_pers) else False

    def _reject_committed_via_pad_spacing(
        self,
        gi: int,
        pad: Rect,
        net: str,
        *,
        candidate_half: int,
    ) -> bool:
        """Reject one landing pad against solve-local committed route metal."""
        return self._reject_foreign_committed_metal_spacing(
            gi, pad, net, candidate_half=candidate_half
        )

    def _reject_fixed_same_net_via_pad_spacing(
        self,
        gi: int,
        pad: Rect,
        net: str,
        *,
        candidate_half: int,
    ) -> bool:
        """Reject one landing pad against immutable own-net artwork spacing.

        Planar wire proximity remains a negotiated PathFinder price.  A via model
        is a bottom/cut/top transaction, so it follows anaroute's via-selector
        pattern: its immutable own-net landing relation is proved once per route
        world, independently of committed route geometry.
        """
        _sep = _eg_attr("_sep", _base_sep)
        candidate = _norm(pad)
        sp = self._space[gi]
        rejected = False
        abstract = self._abstract_rects.get(gi, ())
        oidx = self._obs_idx.get(gi)
        if (
            _sep is _base_sep
            and "_static_or_direct_obstacle_scan" not in self.__dict__
            and type(self)._static_or_direct_obstacle_scan
            is _GridlessLegalityMixin._static_or_direct_obstacle_scan
            and type(oidx) is _SpatialIndex
            and oidx._native is not None
        ):
            hits = oidx._native.scan_same_net_band_rects(
                candidate[0], candidate[1], candidate[2], candidate[3], sp, net
            )
            if hits:
                obs_prov = getattr(self, "_obs_provenance", {})
                for owner, blocker_raw, blocker, sx, sy in hits:
                    if blocker_raw in abstract:
                        continue
                    if self._raw_gap_metallised(
                        gi,
                        candidate,
                        blocker,
                        sx,
                        sy,
                        {gi: (candidate,)},
                    ):
                        continue
                    kind, terminal_ref = obs_prov.get(
                        (gi, owner, blocker), (None, None)
                    )
                    self._record_hard_rejection(
                        "fixed_same_net_via_pad_spacing",
                        owner,
                        gi=gi,
                        candidate_rect=candidate,
                        candidate_half=candidate_half,
                        required_space=sp,
                        blocker_rect=blocker,
                        blocker_kind=kind,
                        terminal_conductor_ref=terminal_ref,
                    )
                    rejected = True
            return rejected
        spsq = sp * sp
        query = (
            candidate[0] - sp,
            candidate[1] - sp,
            candidate[2] + sp,
            candidate[3] + sp,
        )
        for owner, blocker_raw in self._static_or_direct_obstacle_scan(gi, query):
            if owner != net or blocker_raw in abstract:
                continue
            blocker = _norm(blocker_raw)
            sx, sy = _sep(candidate, blocker)
            if not (sx or sy) or sx * sx + sy * sy >= spsq:
                continue
            if self._raw_gap_metallised(
                gi,
                candidate,
                blocker,
                sx,
                sy,
                {gi: (candidate,)},
            ):
                continue
            kind, terminal_ref = getattr(
                self, "_obs_provenance", {}
            ).get((gi, owner, blocker), (None, None))
            self._record_hard_rejection(
                "fixed_same_net_via_pad_spacing",
                owner,
                gi=gi,
                candidate_rect=candidate,
                candidate_half=candidate_half,
                required_space=sp,
                blocker_rect=blocker,
                blocker_kind=kind,
                terminal_conductor_ref=terminal_ref,
            )
            rejected = True
        return rejected

    def _reject_mim_bottom_routing_clearance(
        self,
        gi: int,
        box: Rect,
        net: str,
        *,
        candidate_half: int,
    ) -> bool:
        """Reject route metal inside a PDK-declared MIM bottom-plate band.

        The rule is deliberately net-blind.  A same-net M2 wire joined to the
        MIM plate becomes part of ``metal.interacting(marker)`` and can turn a
        later lower via into a foundry violation; only a genuinely detached
        landing reached from the higher escape metal is legal.
        """

        _rects_clear = _eg_attr("_rects_clear", _base_rects_clear)
        rejected = False
        for owner, blocker, clearance in getattr(
            self, "_mim_bottom_routing_clearance_by_gi", {}
        ).get(gi, ()):
            if _rects_clear(box, blocker, int(clearance)):
                continue
            self._record_hard_rejection(
                "mim_bottom_routing_clearance",
                owner,
                gi=gi,
                candidate_rect=_norm(box),
                candidate_half=candidate_half,
                required_space=int(clearance),
                blocker_rect=blocker,
                blocker_kind="mim_bottom_component",
            )
            rejected = True
        return rejected

    def _mim_derived_unrelated_static_active(self) -> bool:
        return bool(
            getattr(self, "_negotiation_port_legalize", False)
        ) or (
            not bool(getattr(self, "_active_group_route_local_closure", False))
            and bool(getattr(self, "_group_route_global_interfaces_by_net", None))
        )

    def _reject_mim_derived_unrelated_clearance(
        self,
        gi: int,
        box: Rect,
        net: str,
        *,
        candidate_half: int,
    ) -> bool:
        """Reject unrelated-net metal inside a PDK-declared derived MIM halo.

        When ordinary metal spacing strictly exceeds ``grow_nm``, no foreign net
        can ever touch ``(marker & metal).sized(grow_nm)`` without shorting or
        violating spacing against the plate owner; every complete union containing
        ``box`` is therefore guaranteed to remain in ``not_interacting(derived)``
        and fail ``separation_check(derived, spacing_nm)``.  Same-owner metal is
        never rejected here because a local bridge can legally attach to the plate.
        """
        _rects_clear = _eg_attr("_rects_clear", _base_rects_clear)
        _sep = _eg_attr("_sep", _base_sep)
        if not self._mim_derived_unrelated_static_active():
            return False
        rejected = False
        nb = _norm(box)
        for owner, blocker, clearance, metric, rule_id in getattr(
            self, "_mim_derived_unrelated_clearance_by_gi", {}
        ).get(gi, ()):
            if owner == net:
                continue
            req = int(clearance)
            if metric == "euclidian":
                sx, sy = _sep(nb, blocker)
                if sx * sx + sy * sy >= req * req:
                    continue
            elif _rects_clear(nb, blocker, req):
                continue
            self._record_hard_rejection(
                "mim_metal_clearance",
                owner,
                gi=gi,
                candidate_rect=nb,
                candidate_half=candidate_half,
                required_space=req,
                blocker_rect=blocker,
                blocker_kind=rule_id,
            )
            rejected = True
        return rejected

    def _reject_mim_carve_bottom_interaction(
        self,
        gi: int,
        box: Rect,
        net: str,
        *,
        candidate_half: int,
    ) -> bool:
        """Reject metal that the exact MIM carve would remove from routing."""

        rejected = False
        for owner, blocker, blocker_kind in getattr(
            self, "_mim_carve_bottom_blockers_by_gi", {}
        ).get(gi, ()):
            if not _closed_rects_touch(box, blocker):
                continue
            self._record_hard_rejection(
                "mim_carve_bottom_interaction",
                owner,
                gi=gi,
                candidate_rect=_norm(box),
                candidate_half=candidate_half,
                required_space=0,
                blocker_rect=blocker,
                blocker_kind=blocker_kind,
            )
            rejected = True
        return rejected

    def _wire_static_legality_key(
        self,
        gi: int,
        ax: int,
        ay: int,
        bx: int,
        by: int,
        net: str,
    ) -> Tuple[Any, ...]:
        first, second = sorted(((int(ax), int(ay)), (int(bx), int(by))))
        envelope = getattr(self, "_routing_envelope", None)
        return (
            int(gi),
            first,
            second,
            str(net),
            int(self._hw(net, gi)),
            getattr(self, "_obs_idx", {}).get(int(gi)),
            None if envelope is None else tuple(envelope.bbox_nm),
            id(getattr(self, "_abstract_rects", None)),
            id(getattr(self, "_mim_carve_bottom_blockers_by_gi", None)),
            id(getattr(self, "_mim_bottom_routing_clearance_by_gi", None)),
            id(getattr(self, "_mim_derived_unrelated_clearance_by_gi", None)),
            bool(
                int(gi) in (getattr(self, "_mim_derived_unrelated_clearance_by_gi", None) or ())
                and self._mim_derived_unrelated_static_active()
            ),
            id(getattr(self, "_obs_provenance", None)),
            bool(getattr(self, "_active_group_route_local_closure", False)),
            (
                getattr(self, "_active_group_route_local_interface", None)
                if bool(getattr(self, "_active_group_route_local_closure", False))
                else None
            ),
            (
                None
                if bool(getattr(self, "_active_group_route_local_closure", False))
                else tuple(sorted(
                    (getattr(self, "_group_route_local_segment_keys_by_net", None) or {}).get(str(net), ())
                ))
            ),
            (
                None
                if bool(getattr(self, "_active_group_route_local_closure", False))
                else tuple(
                    (getattr(self, "_group_route_global_interfaces_by_net", None) or {}).get(str(net), ())
                )
            ),
            (
                None
                if bool(getattr(self, "_active_group_route_local_closure", False))
                else tuple(sorted(
                    state
                    for owner, state in (
                        getattr(self, "_layered_portal_wire_states", None) or ()
                    )
                    if owner == str(net)
                ))
            ),
        )

    def _wire_static_legality(
        self,
        gi: int,
        ax: int,
        ay: int,
        bx: int,
        by: int,
        net: str,
    ) -> bool:
        return self._cached_static_legality(
            self._wire_static_legality_cache,
            self._wire_static_legality_key(gi, ax, ay, bx, by, net),
            lambda: self._compute_wire_static_legality(
                gi, ax, ay, bx, by, net
            ),
            hit_counter="_wire_static_legality_cache_hits_total",
            miss_counter="_wire_static_legality_cache_misses_total",
        )

    def _compute_wire_static_legality(
        self,
        gi: int,
        ax: int,
        ay: int,
        bx: int,
        by: int,
        net: str,
    ) -> bool:
        wb = _seg_bbox(ax, ay, bx, by, self._hw(net, gi))
        if self._reject_group_route_local_wire(
            gi, ax, ay, bx, by, net, candidate_box=wb
        ):
            return False
        if self._reject_group_route_global_wire(
            gi, ax, ay, bx, by, net, candidate_box=wb
        ):
            return False
        envelope = getattr(self, "_routing_envelope", None)
        if envelope is not None and not envelope.contains_bbox(_norm(wb)):
            self._record_hard_rejection(
                "routing_envelope",
                net,
                gi=gi,
                candidate_rect=_norm(wb),
                candidate_half=self._hw(net, gi),
                blocker_rect=envelope.bbox_nm,
                blocker_kind="floorplan_boundary",
            )
            return False
        if self._reject_mim_bottom_routing_clearance(
            gi,
            wb,
            net,
            candidate_half=self._hw(net, gi),
        ):
            return False
        if self._reject_mim_derived_unrelated_clearance(
            gi,
            wb,
            net,
            candidate_half=self._hw(net, gi),
        ):
            return False
        if self._reject_mim_carve_bottom_interaction(
            gi,
            wb,
            net,
            candidate_half=self._hw(net, gi),
        ):
            return False
        if not self._clear_obstacles(
            gi,
            wb,
            self._space[gi],
            net,
            candidate_half=self._hw(net, gi),
        ):
            return False
        return True

    def _reject_intent_spacing(
        self, gi: int, box: Rect, net: str, *, candidate_half: Optional[int] = None,
    ) -> bool:
        """Enforce named-net clearance outside static PDK query-cache bounds."""
        rules = getattr(self, "_intent_spacing_rules", {})
        if not rules.get(net):
            return False
        rejected = False
        for owner, blocker, distance, kind in spacing_blockers(
            rules, net, _norm(box),
            getattr(self, "_obs_idx", {}).get(gi),
            getattr(self, "_metal_idx", {}).get(gi),
            getattr(self, "_abstract_rects", {}).get(gi, ()),
        ):
            # Preserve the fixed/committed distinction used by graph refinement
            # and rip-up. Fixed terminals cannot be repaired by rerouting their
            # owner; committed wires can. The declared distance replaces only
            # the clearance, never the physical blocker identity.
            self._record_hard_rejection(
                "fixed_metal" if kind == "fixed_metal" else "committed_metal_spacing",
                owner, gi=gi, candidate_rect=_norm(box),
                candidate_half=self._hw(net, gi) if candidate_half is None else candidate_half,
                required_space=distance, blocker_rect=blocker,
                blocker_kind="declared_net_spacing",
            )
            rejected = True
        return rejected

    def _wire_search_identity(self, net: str, seg: _WireSeg) -> Tuple:
        """Exact emitted wire identity owned by one conflict-search branch."""
        a, b = sorted(((int(seg.ax), int(seg.ay)),
                       (int(seg.bx), int(seg.by))))
        return (str(net), int(seg.gi), *a, *b,
                int(self._segment_half_width(net, seg)))

    def _candidate_forbidden_wire(
        self, net: str, segs: Sequence[_WireSeg],
    ) -> Optional[Tuple]:
        """Find a prohibited whole edge, including collinear subdivisions.

        Partial overlap alone is allowed. The restriction is branch state, not
        an immutable physical blockage or a prohibition on the whole track.
        """
        forbidden = getattr(self, "_search_forbidden_wire_edges", ())
        if not forbidden:
            return None
        for key in sorted(forbidden):
            owner, gi, ax, ay, bx, by, half = key
            if owner != str(net):
                continue
            horizontal = ay == by
            if not horizontal and ax != bx:
                continue
            low, high = (ax, bx) if horizontal else (ay, by)
            intervals = []
            for seg in segs:
                if (int(seg.gi) != gi
                        or self._segment_half_width(net, seg) != half):
                    continue
                if horizontal and seg.ay == seg.by == ay:
                    intervals.append(tuple(sorted((seg.ax, seg.bx))))
                elif not horizontal and seg.ax == seg.bx == ax:
                    intervals.append(tuple(sorted((seg.ay, seg.by))))
            cursor = low
            for left, right in sorted(intervals):
                if right < cursor:
                    continue
                if left > cursor:
                    break
                cursor = max(cursor, right)
                if cursor >= high:
                    return key
        return None

    def _wire_legal_single(self, gi: int, ax: int, ay: int, bx: int, by: int,
                           net: str) -> bool:
        ctx = getattr(self, "_active_edge_ctx", None)
        if (
            ctx is not None
            and not getattr(self, "_active_edge_ctx_bypass", False)
            and str(net) == getattr(self, "_active_edge_ctx_net", None)
        ):
            res = ctx.check_wire_legal_single(gi, ax, ay, bx, by)
            if res is not None:
                if res >= 0:
                    active = getattr(self, "_active_static_metal_scans", None)
                    if active is not None:
                        wb_fast = _seg_bbox(ax, ay, bx, by, self._hw(net, gi))
                        active[(int(gi), wb_fast, str(net))] = (res, ())
                    return True
                self._hard_rejection_generation = int(
                    getattr(self, "_hard_rejection_generation", 0)
                ) + 1
                self._active_edge_ctx_gen_bumped = int(
                    getattr(self, "_active_edge_ctx_gen_bumped", 0)
                ) + 1
                return False
        hw = self._hw(net, gi)
        wb = _seg_bbox(ax, ay, bx, by, hw)
        cpp = load_native("_gridless_cpp")
        if (
            cpp is not None
            and not getattr(self, "_search_forbidden_wire_edges", ())
            and not bool(getattr(self, "_active_group_route_local_closure", False))
            and not getattr(self, "_group_route_global_interfaces_by_net", {}).get(str(net))
            and not getattr(self, "_intent_spacing_rules", {}).get(net)
            and not getattr(self, "_mim_bottom_routing_clearance_by_gi", {}).get(gi)
            and not getattr(self, "_mim_carve_bottom_blockers_by_gi", {}).get(gi)
            and not (
                getattr(self, "_mim_derived_unrelated_clearance_by_gi", {}).get(gi)
                and self._mim_derived_unrelated_static_active()
            )
            and _eg_attr("_rects_clear", _base_rects_clear) is _base_rects_clear
            and _eg_attr("_sep", _base_sep) is _base_sep
        ):
            inst_dict = self.__dict__
            if not (
                "_wire_static_legality" in inst_dict
                or "_compute_wire_static_legality" in inst_dict
                or "_static_or_direct_obstacle_scan" in inst_dict
                or "_clear_obstacles" in inst_dict
                or "_reject_intent_spacing" in inst_dict
                or "_reject_foreign_committed_metal_spacing" in inst_dict
                or "_reject_foreign_committed_metal_contact" in inst_dict
                or "_scan_committed_metal" in inst_dict
                or "_static_or_direct_committed_metal_scan" in inst_dict
                or "_wide_spacing_conflicts" in inst_dict
                or "_wide_spacing_neighbors" in inst_dict
                or "_wide_indexes" in inst_dict
            ):
                env = getattr(self, "_routing_envelope", None)
                env_ok = (
                    env is None
                    or type(env).__module__ == "clay.layout.route.floorplan_containment"
                )
                oidx = self._obs_idx.get(gi)
                midx = self._metal_idx.get(gi)
                idx_ok = (
                    (oidx is None or (type(oidx) is _SpatialIndex and oidx._native is not None))
                    and (midx is None or (type(midx) is _SpatialIndex and midx._native is not None))
                )
                if env_ok and idx_ok:
                    self._ensure_fixed_wide_geometry_indexes()
                    min_w = self._min_wide_ge_by_gi.get(gi)
                    wide_ok = True
                    wide_arg = None
                    if min_w is not None:
                        if 2 * hw >= min_w:
                            wide_ok = False
                        else:
                            w_entries = self._wide_indexes(True).get(gi, ())
                            if w_entries:
                                w_list = []
                                for w_sp, c_idx, _w_idx, a_idx in w_entries:
                                    if (
                                        type(c_idx) is not _SpatialIndex
                                        or c_idx._native is None
                                        or type(a_idx) is not _SpatialIndex
                                        or a_idx._native is None
                                    ):
                                        wide_ok = False
                                        break
                                    w_list.append((int(w_sp), c_idx._native, a_idx._native))
                                if wide_ok:
                                    wide_arg = tuple(w_list)
                    if wide_ok:
                        present = cpp.check_wire_legal_fast_cpp(
                            wb[0], wb[1], wb[2], wb[3],
                            self._space[gi],
                            self._res_cell,
                            int(round(self._pres_fac * _COST_FACTOR_SCALE)),
                            net,
                            oidx._native if oidx is not None else None,
                            midx._native if midx is not None else None,
                            env.bbox_nm if env is not None else None,
                            wide_arg,
                        )
                        if present is not None:
                            active = getattr(self, "_active_static_metal_scans", None)
                            if active is not None:
                                active[(int(gi), wb, str(net))] = (present, ())
                            return True
        if getattr(self, "_search_forbidden_wire_edges", ()) and (
            self._candidate_forbidden_wire(
                net, (_WireSeg(gi, ax, ay, bx, by, hw),)
            ) is not None
        ):
            self._record_hard_rejection(
                "candidate_wire_branch", net, gi=gi,
                candidate_rect=wb, candidate_half=hw,
                blocker_kind="exact_candidate_wire",
            )
            return False
        if not self._wire_static_legality(gi, ax, ay, bx, by, net):
            return False
        if self._reject_intent_spacing(gi, wb, net):
            return False
        # A declared supra-minimum carrier has a much larger emitted footprint
        # than an ordinary access wire.  Treat its exact positive foreign
        # spacing band as a search wall in either route order: leaving it soft
        # allowed both two 4um trunks and a later narrow signal beside an
        # already-committed wide carrier topology to reproduce deck residuals.
        # Minimum-width pairs retain the historical soft band so dense pins are
        # not sealed.
        if (
            self._reject_foreign_committed_metal_spacing(
                gi,
                wb,
                net,
                candidate_half=hw,
                require_supra_minimum_endpoint=True,
            )
        ):
            return False
        return not self._reject_foreign_committed_metal_contact(
            gi,
            wb,
            net,
            candidate_half=hw,
        )

    def _reject_group_route_local_wire(
        self,
        gi: int,
        ax: int,
        ay: int,
        bx: int,
        by: int,
        net: str,
        *,
        candidate_box: Rect,
    ) -> bool:
        """Confine local group-closure wires inside ``envelope_nm`` except at the active portal stub."""
        if not bool(getattr(self, "_active_group_route_local_closure", False)):
            return False
        local_iface = getattr(self, "_active_group_route_local_interface", None)
        if local_iface is None:
            return False
        envelope, portal = local_iface
        x0, y0, x1, y1 = (
            int(envelope[0]),
            int(envelope[1]),
            int(envelope[2]),
            int(envelope[3]),
        )
        px, py = int(portal[0]), int(portal[1])
        ax_i, ay_i, bx_i, by_i = int(ax), int(ay), int(bx), int(by)
        candidate = _norm(candidate_box)
        if not (
            x0 <= ax_i <= x1
            and y0 <= ay_i <= y1
            and x0 <= bx_i <= x1
            and y0 <= by_i <= y1
        ):
            self._record_hard_rejection(
                "group_route_local_envelope",
                net,
                gi=gi,
                candidate_rect=candidate,
                candidate_half=self._hw(net, gi),
                blocker_rect=(x0, y0, x1, y1),
                blocker_kind="certified_group_envelope",
            )
            return True
        cx0, cy0, cx1, cy1 = candidate
        if cx0 >= x0 and cy0 >= y0 and cx1 <= x1 and cy1 <= y1:
            return False
        over_l = cx0 < x0
        over_r = cx1 > x1
        over_b = cy0 < y0
        over_t = cy1 > y1
        portal_l = px == x0
        portal_r = px == x1
        portal_b = py == y0
        portal_t = py == y1
        covers_portal = cx0 <= px <= cx1 and cy0 <= py <= cy1
        rides_boundary = (ax_i, ay_i) != (bx_i, by_i) and (
            (over_l and ax_i == bx_i == x0)
            or (over_r and ax_i == bx_i == x1)
            or (over_b and ay_i == by_i == y0)
            or (over_t and ay_i == by_i == y1)
        )
        if (
            not covers_portal
            or rides_boundary
            or (over_l and not portal_l)
            or (over_r and not portal_r)
            or (over_b and not portal_b)
            or (over_t and not portal_t)
        ):
            self._record_hard_rejection(
                "group_route_local_envelope",
                net,
                gi=gi,
                candidate_rect=candidate,
                candidate_half=self._hw(net, gi),
                blocker_rect=(x0, y0, x1, y1),
                blocker_kind="certified_group_envelope",
            )
            return True
        return False

    def _reject_group_route_local_via(
        self,
        x: int,
        y: int,
        lo: int,
        net: str,
        n: Optional[int],
    ) -> bool:
        """Confine local group-closure vias strictly inside ``envelope_nm``."""
        if not bool(getattr(self, "_active_group_route_local_closure", False)):
            return False
        local_iface = getattr(self, "_active_group_route_local_interface", None)
        if local_iface is None:
            return False
        envelope, _portal = local_iface
        x0, y0, x1, y1 = (
            int(envelope[0]),
            int(envelope[1]),
            int(envelope[2]),
            int(envelope[3]),
        )
        below, above = (
            self._via_pad_half1[lo]
            if n == 1
            else self._via_pad_half[lo]
        )
        extent = max(int(below), int(above), int(self._via_cut_extent(lo, n)))
        candidate = (
            int(x) - extent,
            int(y) - extent,
            int(x) + extent,
            int(y) + extent,
        )
        if (
            candidate[0] < x0
            or candidate[1] < y0
            or candidate[2] > x1
            or candidate[3] > y1
        ):
            self._record_hard_rejection(
                "group_route_local_envelope",
                net,
                gi=lo,
                candidate_rect=candidate,
                candidate_half=extent,
                blocker_rect=(x0, y0, x1, y1),
                blocker_kind="certified_group_envelope",
            )
            return True
        return False

    def _reject_group_route_global_wire(
        self,
        gi: int,
        ax: int,
        ay: int,
        bx: int,
        by: int,
        net: str,
        *,
        candidate_box: Rect,
    ) -> bool:
        """Keep full emitted global wires outside certified group interiors."""

        if bool(getattr(self, "_active_group_route_local_closure", False)):
            return False
        local_key = (
            min((int(ax), int(ay)), (int(bx), int(by))),
            max((int(ax), int(ay)), (int(bx), int(by))),
            int(gi),
            int(self._hw(net, gi)),
        )
        if local_key in getattr(
            self, "_group_route_local_segment_keys_by_net", {}
        ).get(str(net), frozenset()):
            return False
        candidate = _norm(candidate_box)
        for envelope, portal in getattr(
            self, "_group_route_global_interfaces_by_net", {}
        ).get(str(net), ()):
            x0, y0, x1, y1 = envelope
            enters = (
                max(candidate[0], x0) <= min(candidate[2], x1)
                and max(candidate[1], y0) <= min(candidate[3], y1)
            )
            if not enters:
                continue
            px, py = int(portal[0]), int(portal[1])
            ax_i, ay_i = int(ax), int(ay)
            bx_i, by_i = int(bx), int(by)
            if (ax_i, ay_i) != (bx_i, by_i):
                exits_at_portal = (
                    (px == x0 and ay_i == by_i == py and min(ax_i, bx_i) < x0 and max(ax_i, bx_i) <= x0)
                    or (px == x1 and ay_i == by_i == py and max(ax_i, bx_i) > x1 and min(ax_i, bx_i) >= x1)
                    or (py == y0 and ax_i == bx_i == px and min(ay_i, by_i) < y0 and max(ay_i, by_i) <= y0)
                    or (py == y1 and ax_i == bx_i == px and max(ay_i, by_i) > y1 and min(ay_i, by_i) >= y1)
                )
                if exits_at_portal:
                    if (px, py) in ((ax_i, ay_i), (bx_i, by_i)):
                        continue
                    layered_gis = [
                        int(lgi)
                        for owner, (lx, ly, lgi) in (
                            getattr(self, "_layered_portal_wire_states", None) or ()
                        )
                        if owner == str(net) and (int(lx), int(ly)) == (px, py)
                    ]
                    if not layered_gis or int(gi) in layered_gis:
                        continue
            self._record_hard_rejection(
                "group_route_global_interior",
                net,
                gi=gi,
                candidate_rect=candidate,
                candidate_half=self._hw(net, gi),
                blocker_rect=envelope,
                blocker_kind="certified_group_envelope",
            )
            return True
        return False

    def _reject_group_route_local_fill(
        self,
        gi: int,
        candidate_box: Rect,
        net: str,
    ) -> bool:
        """Keep local closure fills inside ``envelope_nm`` except right at the portal."""
        if not bool(getattr(self, "_active_group_route_local_closure", False)):
            return False
        local_iface = getattr(self, "_active_group_route_local_interface", None)
        if local_iface is None:
            return False
        envelope, portal = local_iface
        x0, y0, x1, y1 = (int(v) for v in envelope)
        px, py = int(portal[0]), int(portal[1])
        cx0, cy0, cx1, cy1 = _norm(candidate_box)
        if cx0 >= x0 and cy0 >= y0 and cx1 <= x1 and cy1 <= y1:
            return False
        over_l = cx0 < x0
        over_r = cx1 > x1
        over_b = cy0 < y0
        over_t = cy1 > y1
        covers_portal = cx0 <= px <= cx1 and cy0 <= py <= cy1
        valid_portal_overhang = (
            covers_portal
            and (not over_l or px == x0)
            and (not over_r or px == x1)
            and (not over_b or py == y0)
            and (not over_t or py == y1)
        )
        if valid_portal_overhang:
            max_half = int(self._hw(net, int(gi)))
            for lo in (int(gi) - 1, int(gi)):
                pads = getattr(self, "_via_pad_half", {}).get(lo)
                if pads is not None:
                    max_half = max(max_half, int(pads[int(gi) - lo]))
            if (over_l or over_r) and cy0 >= py - max_half and cy1 <= py + max_half:
                return False
            if (over_b or over_t) and cx0 >= px - max_half and cx1 <= px + max_half:
                return False
        self._record_hard_rejection(
            "group_route_local_envelope",
            net,
            gi=gi,
            candidate_rect=(cx0, cy0, cx1, cy1),
            candidate_half=self._hw(net, gi),
            blocker_rect=(x0, y0, x1, y1),
            blocker_kind="certified_group_envelope",
        )
        return True

    def _reject_group_route_global_box(
        self,
        gi: int,
        candidate_box: Rect,
        net: str,
    ) -> bool:
        """Keep global and post-route fills off certified group boundaries outside portals."""
        if bool(getattr(self, "_active_group_route_local_closure", False)):
            return False
        cx0, cy0, cx1, cy1 = _norm(candidate_box)
        bridge_margin = 2 * (int(self._hw(net, int(gi))) + int(self._space[int(gi)]))
        for envelope, portal in getattr(
            self, "_group_route_global_interfaces_by_net", {}
        ).get(str(net), ()):
            x0, y0, x1, y1 = (int(v) for v in envelope)
            if not (max(cx0, x0) <= min(cx1, x1) and max(cy0, y0) <= min(cy1, y1)):
                continue
            if x0 < cx0 and cx1 < x1 and y0 < cy0 and cy1 < y1:
                continue
            px, py = int(portal[0]), int(portal[1])
            if len(portal) >= 3:
                layer_ok = int(gi) == int(portal[2])
            else:
                layered_gis = [
                    int(lgi)
                    for owner, (lx, ly, lgi) in (
                        getattr(self, "_layered_portal_wire_states", None) or ()
                    )
                    if owner == str(net) and (int(lx), int(ly)) == (px, py)
                ]
                layer_ok = not layered_gis or int(gi) in layered_gis
            if (
                layer_ok
                and cx0 <= px <= cx1
                and cy0 <= py <= cy1
                and (
                    (px == x0 and cx1 <= x0 and y0 < cy0 and cy1 < y1)
                    or (px == x1 and cx0 >= x1 and y0 < cy0 and cy1 < y1)
                    or (py == y0 and cy1 <= y0 and x0 < cx0 and cx1 < x1)
                    or (py == y1 and cy0 >= y1 and x0 < cx0 and cx1 < x1)
                )
            ):
                continue
            if layer_ok:
                touch_l = cx0 <= x0 <= cx1
                touch_r = cx0 <= x1 <= cx1
                touch_b = cy0 <= y0 <= cy1
                touch_t = cy0 <= y1 <= cy1
                only_portal_lr = (
                    ((px == x0 and touch_l and not touch_r) or (px == x1 and touch_r and not touch_l))
                    and not touch_b
                    and not touch_t
                    and y0 < cy0 and cy1 < y1
                    and max(cy0 - py, py - cy1, 0) <= bridge_margin
                )
                only_portal_bt = (
                    ((py == y0 and touch_b and not touch_t) or (py == y1 and touch_t and not touch_b))
                    and not touch_l
                    and not touch_r
                    and x0 < cx0 and cx1 < x1
                    and max(cx0 - px, px - cx1, 0) <= bridge_margin
                )
                if only_portal_lr or only_portal_bt:
                    continue
            self._record_hard_rejection(
                "group_route_global_interior",
                net,
                gi=gi,
                candidate_rect=(cx0, cy0, cx1, cy1),
                candidate_half=self._hw(net, gi),
                blocker_rect=(x0, y0, x1, y1),
                blocker_kind="certified_group_envelope",
            )
            return True
        return False

    def _reject_group_route_global_via(
        self,
        x: int,
        y: int,
        lo: int,
        net: str,
        n: Optional[int],
    ) -> bool:
        """Forbid full emitted global via geometry inside a routed group."""

        if bool(getattr(self, "_active_group_route_local_closure", False)):
            return False
        via_key = (
            int(x),
            int(y),
            int(lo),
            int(0 if n is None else n),
        )
        local_vias = getattr(
            self, "_group_route_local_via_keys_by_net", {}
        ).get(str(net), frozenset())
        # A layered interface is certified by an actual global wire on the
        # portal layer.  Do not climb or descend at that precise endpoint.
        if any(
            owner == str(net)
            and (px, py) == (x, y)
            and gi in (lo, lo + 1)
            for owner, (px, py, gi) in getattr(
                self, "_layered_portal_wire_states", ()
            )
        ):
            return True
        if via_key in local_vias:
            return False
        below, above = (
            self._via_pad_half1[lo]
            if n == 1
            else self._via_pad_half[lo]
        )
        extent = max(below, above, self._via_cut_extent(lo, n))
        candidate = (x - extent, y - extent, x + extent, y + extent)
        for envelope, portal in getattr(
            self, "_group_route_global_interfaces_by_net", {}
        ).get(str(net), ()):
            x0, y0, x1, y1 = envelope
            enters = (
                max(candidate[0], x0) <= min(candidate[2], x1)
                and max(candidate[1], y0) <= min(candidate[3], y1)
            )
            if not enters or (int(x), int(y)) == (
                int(portal[0]), int(portal[1])
            ):
                continue
            self._record_hard_rejection(
                "group_route_global_interior",
                net,
                gi=lo,
                candidate_rect=candidate,
                candidate_half=extent,
                blocker_rect=envelope,
                blocker_kind="certified_group_envelope",
            )
            return True
        return False

    @staticmethod
    def _reflect_box_x(box: Rect, axis2: int) -> Rect:
        x0, y0, x1, y1 = _norm(box)
        return (int(axis2) - x1, y0, int(axis2) - x0, y1)

    def _reject_active_mirror_pair_box(
        self,
        gi: int,
        primary_box: Rect,
        partner_box: Rect,
        partner_net: str,
        *,
        candidate_half: int,
    ) -> bool:
        """Enforce foreign-net spacing inside one uncommitted mirror pair.

        ``_attempt_metal`` contains earlier MST edges of the primary.  Their
        partner geometry is the exact reflection, so testing the new reflected
        box against those primary rectangles covers both cross-directions by
        symmetry.  The current primary/reflected atom is checked explicitly.
        """
        _rects_clear = _eg_attr("_rects_clear", _base_rects_clear)
        required = int(self._space[gi])
        mirror = getattr(self, "_active_mirror_legality", None)
        if mirror is not None:
            required = max(required, getattr(self, "_intent_spacing_rules", {}).get(
                str(mirror.net_a), {}).get(partner_net, 0))
        blockers = [tuple(_norm(primary_box))]
        blockers.extend(
            tuple(_norm(rect))
            for rect in self._attempt_metal.get(gi, ())
        )
        rejected = False
        for blocker in blockers:
            if _rects_clear(partner_box, blocker, required):
                continue
            self._record_hard_rejection(
                "mirror_pair_in_transaction_conflict",
                partner_net,
                gi=gi,
                candidate_rect=partner_box,
                candidate_half=candidate_half,
                required_space=required,
                blocker_rect=blocker,
                blocker_kind="mirror_primary_attempt_metal",
            )
            rejected = True
        if rejected:
            self._mirror_pair_conflict_seen = True
        return rejected

    def _wire_legal(self, gi: int, ax: int, ay: int, bx: int, by: int,
                    net: str) -> bool:
        """Judge one edge and, for a hard pair, its reflected partner edge.

        The symmetric corridor is a search-time obligation.  Checking it only
        after committing the primary makes the shortest primary path mask legal
        detours whenever the partner side contains fixed artwork.  This mirrors
        anaroute's DRC-driven symmetric A* pattern while keeping Clay's exact
        final reflection oracle unchanged.
        """
        ctx = getattr(self, "_active_edge_ctx", None)
        if (
            ctx is not None
            and not getattr(self, "_active_edge_ctx_bypass", False)
            and str(net) == getattr(self, "_active_edge_ctx_net", None)
        ):
            res = ctx.check_wire_legal(gi, ax, ay, bx, by)
            if res is not None:
                if res >= 0:
                    active = getattr(self, "_active_static_metal_scans", None)
                    if active is not None:
                        wb_fast = _seg_bbox(ax, ay, bx, by, self._hw(net, gi))
                        active[(int(gi), wb_fast, str(net))] = (res, ())
                    return True
                self._hard_rejection_generation = int(
                    getattr(self, "_hard_rejection_generation", 0)
                ) + 1
                self._active_edge_ctx_gen_bumped = int(
                    getattr(self, "_active_edge_ctx_gen_bumped", 0)
                ) + 1
                return False
        if not self._wire_legal_single(gi, ax, ay, bx, by, net):
            return False
        mirror = getattr(self, "_active_mirror_legality", None)
        if mirror is None or str(net) != str(mirror.net_a):
            return True
        axis2 = self._mirror_axis_twice_nm(mirror)
        partner_box = _seg_bbox(
            axis2 - int(ax),
            int(ay),
            axis2 - int(bx),
            int(by),
            self._hw(str(mirror.net_b), gi),
        )
        if self._reject_active_mirror_pair_box(
            gi,
            _seg_bbox(ax, ay, bx, by, self._hw(net, gi)),
            partner_box,
            str(mirror.net_b),
            candidate_half=self._hw(str(mirror.net_b), gi),
        ):
            return False
        return self._wire_legal_single(
            gi,
            axis2 - int(ax),
            int(ay),
            axis2 - int(bx),
            int(by),
            str(mirror.net_b),
        )

    def _scan_committed_cuts(
        self, x: int, y: int, lo: int, net: str, n: Optional[int] = None,
    ) -> Tuple[
        int,
        Tuple[Tuple[Optional[str], Rect], ...],
        Tuple[Tuple[Optional[str], Rect], ...],
    ]:
        """One local cut scan for contact/spacing legality and present price."""
        _rects_clear = _eg_attr("_rects_clear", _base_rects_clear)
        _sep = _eg_attr("_sep", _base_sep)
        ext = self._via_cut_extent(lo, n)
        cut = (x - ext, y - ext, x + ext, y + ext)
        vsp = self._via_space[lo]
        half = vsp // 2
        ebx = (
            cut[0] - half,
            cut[1] - half,
            cut[2] + half,
            cut[3] + half,
        )
        rc = self._res_cell
        total = 0
        foreign_contacts = []
        spacing_violations = []
        cut_idx = self._cut_idx.get(lo)
        if cut_idx is not None:
            pres = int(round(self._pres_fac * _COST_FACTOR_SCALE))
            if (
                _rects_clear is _base_rects_clear
                and _sep is _base_sep
                and type(cut_idx) is _SpatialIndex
                and cut_idx._native is not None
            ):
                return cut_idx._native.scan_committed_cuts(
                    cut[0], cut[1], cut[2], cut[3], vsp, rc, pres, net
                )
            nearby = cut_idx.query((
                cut[0] - vsp,
                cut[1] - vsp,
                cut[2] + vsp,
                cut[3] + vsp,
            ))
            for owner, other in nearby:
                other = _norm(other)
                if owner == net:
                    if other != cut and not _rects_clear(cut, other, vsp):
                        spacing_violations.append((owner, other))
                    continue
                if _sep(cut, other) == (0, 0):
                    foreign_contacts.append((owner, other))
                elif not _rects_clear(cut, other, vsp):
                    spacing_violations.append((owner, other))
                ox0, oy0, ox1, oy1 = other
                other_box = (
                    ox0 - half,
                    oy0 - half,
                    ox1 + half,
                    oy1 + half,
                )
                ix0 = max(ebx[0], other_box[0])
                iy0 = max(ebx[1], other_box[1])
                ix1 = min(ebx[2], other_box[2])
                iy1 = min(ebx[3], other_box[3])
                if ix0 >= ix1 or iy0 >= iy1:
                    continue
                total += pres * max(1, (ix1 - ix0) // rc) * max(
                    1, (iy1 - iy0) // rc
                )
        key = lambda item: (
            "" if item[0] is None else str(item[0]),
            item[1],
        )
        foreign_contacts.sort(key=key)
        spacing_violations.sort(key=key)
        return (
            total,
            tuple(foreign_contacts),
            tuple(spacing_violations),
        )

    def _static_or_direct_committed_cut_scan(
        self, x: int, y: int, lo: int, net: str, n: Optional[int] = None,
    ) -> Tuple[
        int,
        Tuple[Tuple[Optional[str], Rect], ...],
        Tuple[Tuple[Optional[str], Rect], ...],
    ]:
        ext = self._via_cut_extent(lo, n)
        key = (
            int(lo),
            (x - ext, y - ext, x + ext, y + ext),
            str(net),
        )
        active = getattr(self, "_active_static_cut_scans", None)
        if active is None:
            return self._scan_committed_cuts(x, y, lo, net, n)
        cached = active.get(key)
        if cached is not None:
            return cached
        scanned = self._scan_committed_cuts(x, y, lo, net, n)
        active[key] = scanned
        return scanned

    def _via_static_legality_key(
        self,
        x: int,
        y: int,
        lo: int,
        net: str,
        n: Optional[int],
    ) -> Tuple[Any, ...]:
        """Name one immutable route-world via obligation exactly.

        The index objects keep direct helper/tests that replace a fixed world
        from observing a stale proof.  Canonical ``route()`` additionally clears
        the whole cache whenever it ingests new artwork.
        """
        lo_i = int(lo)
        n_key = 1 if n == 1 else 0
        net_s = str(net)
        local_iface = (
            getattr(self, "_active_group_route_local_interface", None)
            if bool(getattr(self, "_active_group_route_local_closure", False))
            else None
        )
        active = getattr(self, "_active_static_obstacle_scans", None)
        tail_cache_key = ("__via_static_tail__", lo_i, n_key, net_s, local_iface)
        if active is not None:
            cached_tail = active.get(tail_cache_key)
            if cached_tail is not None:
                return (int(x), int(y), *cached_tail)
        cut_index = getattr(self, "_obs_cut_idx", {}).get(lo_i)
        fixed_cut_world: Any = cut_index
        if cut_index is None:
            fixed_cut_world = tuple(sorted(
                tuple(_norm(rect))
                for rect in getattr(self, "_obs_cuts", {}).get(lo_i, ())
            ))
        envelope = getattr(self, "_routing_envelope", None)
        tail = (
            lo_i,
            n_key,
            net_s,
            max(1, int(getattr(self, "_manufacturing_grid_nm", 1))),
            getattr(self, "_obs_idx", {}).get(lo_i),
            getattr(self, "_obs_idx", {}).get(lo_i + 1),
            fixed_cut_world,
            None if envelope is None else tuple(envelope.bbox_nm),
            id(getattr(self, "_abstract_rects", None)),
            id(getattr(self, "_mim_carve_bottom_blockers_by_gi", None)),
            id(getattr(self, "_mim_bottom_routing_clearance_by_gi", None)),
            id(getattr(self, "_mim_derived_unrelated_clearance_by_gi", None)),
            bool(
                (
                    lo_i in (getattr(self, "_mim_derived_unrelated_clearance_by_gi", None) or ())
                    or (lo_i + 1) in (getattr(self, "_mim_derived_unrelated_clearance_by_gi", None) or ())
                )
                and self._mim_derived_unrelated_static_active()
            ),
            id(getattr(self, "_mim_forbidden_cut_touch_by_lo", None)),
            id(getattr(self, "_mim_top_via_marker_clearance_by_lo", None)),
            id(getattr(self, "_obs_provenance", None)),
            id(getattr(self, "_obs_cut_provenance", None)),
            local_iface,
        )
        if active is not None:
            active[tail_cache_key] = tail
        return (int(x), int(y), *tail)

    def _evaluate_via_static_legality(
        self,
        x: int,
        y: int,
        lo: int,
        net: str,
        n: Optional[int],
    ) -> bool:
        return self._cached_static_legality(
            self._via_static_legality_cache,
            self._via_static_legality_key(x, y, lo, net, n),
            lambda: self._compute_via_static_legality(x, y, lo, net, n),
            hit_counter="_via_static_legality_cache_hits_total",
            miss_counter="_via_static_legality_cache_misses_total",
        )

    def _compute_via_static_legality(
        self,
        x: int,
        y: int,
        lo: int,
        net: str,
        n: Optional[int],
    ) -> bool:
        """Judge only immutable floorplan/artwork obligations for one via."""
        _rects_clear = _eg_attr("_rects_clear", _base_rects_clear)
        bh, ah = (self._via_pad_half1[lo] if n == 1 else self._via_pad_half[lo])
        grid = max(1, int(getattr(self, "_manufacturing_grid_nm", 1)))
        if int(x) % grid or int(y) % grid:
            # Native conductor centroids may lie between manufacturing sites.
            # Reject that via before pattern-first rail access commits it; the
            # ordinary covering-pad escape selects an on-grid point without
            # moving the source device or silently snapping emitted polygons.
            self._record_hard_rejection(
                "manufacturing_grid",
                net,
                gi=lo,
                candidate_rect=(x - bh, y - bh, x + bh, y + bh),
                candidate_half=bh,
                required_space=grid,
                blocker_kind="via_manufacturing_grid",
            )
            return False
        if self._reject_group_route_local_via(x, y, lo, net, n):
            return False
        legal = True
        envelope = getattr(self, "_routing_envelope", None)
        for gi, half in ((lo, bh), (lo + 1, ah)):
            pad = (x - half, y - half, x + half, y + half)
            if envelope is not None and not envelope.contains_bbox(pad):
                self._record_hard_rejection(
                    "routing_envelope",
                    net,
                    gi=gi,
                    candidate_rect=pad,
                    candidate_half=half,
                    blocker_rect=envelope.bbox_nm,
                    blocker_kind="floorplan_boundary",
                )
                legal = False
            if self._reject_mim_bottom_routing_clearance(
                gi,
                pad,
                net,
                candidate_half=half,
            ):
                legal = False
            if self._reject_mim_derived_unrelated_clearance(
                gi,
                pad,
                net,
                candidate_half=half,
            ):
                legal = False
            if self._reject_mim_carve_bottom_interaction(
                gi,
                pad,
                net,
                candidate_half=half,
            ):
                legal = False
            if not self._clear_obstacles(
                gi,
                pad,
                self._space[gi],
                net,
                candidate_half=half,
            ):
                legal = False
            if self._reject_fixed_same_net_via_pad_spacing(
                gi,
                pad,
                net,
                candidate_half=half,
            ):
                legal = False

        ext = self._via_cut_extent(lo, n)
        cut = (x - ext, y - ext, x + ext, y + ext)
        if envelope is not None and not envelope.contains_bbox(cut):
            self._record_hard_rejection(
                "routing_envelope",
                net,
                gi=lo,
                candidate_rect=cut,
                candidate_half=ext,
                blocker_rect=envelope.bbox_nm,
                blocker_kind="floorplan_boundary",
            )
            legal = False
        for owner, blocker in getattr(
            self, "_mim_forbidden_cut_touch_by_lo", {}
        ).get(lo, ()):
            if not _closed_rects_touch(cut, blocker):
                continue
            self._record_hard_rejection(
                "mim_forbidden_cut_touch",
                owner,
                gi=lo,
                candidate_rect=cut,
                candidate_half=ext,
                required_space=0,
                blocker_rect=blocker,
                blocker_kind="mim_bottom_component",
            )
            legal = False
        for owner, blocker, clearance in getattr(
            self, "_mim_top_via_marker_clearance_by_lo", {}
        ).get(lo, ()):
            if _rects_clear(cut, blocker, int(clearance)):
                continue
            self._record_hard_rejection(
                "mim_top_via_marker_clearance",
                owner,
                gi=lo,
                candidate_rect=cut,
                candidate_half=ext,
                required_space=int(clearance),
                blocker_rect=blocker,
                blocker_kind="mim_top_marker",
            )
            legal = False
        vsp = self._via_space[lo]
        obs_cut_map = getattr(self, "_obs_cut_idx", None)
        obs_cut_idx = obs_cut_map.get(lo) if obs_cut_map is not None else None
        if (
            _rects_clear is _base_rects_clear
            and "_local_fixed_cuts" not in self.__dict__
            and type(obs_cut_idx) is _SpatialIndex
            and obs_cut_idx._native is not None
        ):
            fixed_conflicts = obs_cut_idx._native.scan_fixed_cut_conflicts(
                cut[0], cut[1], cut[2], cut[3], vsp
            )
            if fixed_conflicts:
                cut_prov = getattr(self, "_obs_cut_provenance", {})
                for vb in fixed_conflicts:
                    owner, kind, terminal_ref = cut_prov.get(
                        (lo, vb), (None, None, None)
                    )
                    self._record_hard_rejection(
                        "fixed_cut",
                        owner,
                        gi=lo,
                        candidate_rect=cut,
                        candidate_half=ext,
                        required_space=vsp,
                        blocker_rect=vb,
                        blocker_kind=kind,
                        terminal_conductor_ref=terminal_ref,
                    )
                    legal = False
        else:
            fixed_query = (
                cut[0] - vsp,
                cut[1] - vsp,
                cut[2] + vsp,
                cut[3] + vsp,
            )
            for vb in self._local_fixed_cuts(lo, fixed_query):
                if vb == cut:
                    continue
                if not _rects_clear(cut, vb, vsp):
                    owner, kind, terminal_ref = getattr(
                        self, "_obs_cut_provenance", {}
                    ).get((lo, vb), (None, None, None))
                    self._record_hard_rejection(
                        "fixed_cut",
                        owner,
                        gi=lo,
                        candidate_rect=cut,
                        candidate_half=ext,
                        required_space=vsp,
                        blocker_rect=vb,
                        blocker_kind=kind,
                        terminal_conductor_ref=terminal_ref,
                    )
                    legal = False
        return legal

    def _via_legal(self, x: int, y: int, lo: int, net: str,
                   n: Optional[int] = None) -> bool:
        """Judge one policy-authorized via and any reflected partner atomically."""
        # A router-owned inter-metal transition must use the selected engine's
        # declared N-by-N array.  Fixed device stacks have their own artwork
        # authority and do not enter this candidate-via predicate.
        _rects_clear = _eg_attr("_rects_clear", _base_rects_clear)
        if n is not None and n != self.via_redundancy and not (
            n == 1 and self._single_via_escape_allowed()
        ):
            return False
        ctx = getattr(self, "_active_edge_ctx", None)
        if (
            ctx is not None
            and not getattr(self, "_active_edge_ctx_bypass", False)
            and str(net) == getattr(self, "_active_edge_ctx_net", None)
        ):
            n_flag = 1 if n == 1 else 0
            c_res = ctx.check_via_legal(x, y, lo, n_flag)
            if c_res is not None:
                if c_res is False:
                    self._hard_rejection_generation = int(
                        getattr(self, "_hard_rejection_generation", 0)
                    ) + 1
                    self._active_edge_ctx_gen_bumped = int(
                        getattr(self, "_active_edge_ctx_gen_bumped", 0)
                    ) + 1
                    return False
                p_lo, p_hi, p_cut = c_res
                active_m = getattr(self, "_active_static_metal_scans", None)
                if active_m is not None:
                    bh_f, ah_f = (
                        self._via_pad_half1[lo]
                        if n_flag == 1
                        else self._via_pad_half[lo]
                    )
                    net_s = str(net)
                    active_m[(int(lo), (x - bh_f, y - bh_f, x + bh_f, y + bh_f), net_s)] = (p_lo, ())
                    active_m[(int(lo) + 1, (x - ah_f, y - ah_f, x + ah_f, y + ah_f), net_s)] = (p_hi, ())
                active_c = getattr(self, "_active_static_cut_scans", None)
                if active_c is not None:
                    ext_f = self._via_cut_extent(lo, n)
                    active_c[(int(lo), (x - ext_f, y - ext_f, x + ext_f, y + ext_f), str(net))] = (p_cut, (), ())
                return True
        previous_metal_scans = self._active_static_metal_scans
        previous_cut_scans = self._active_static_cut_scans
        previous_obstacle_scans = self._active_static_obstacle_scans
        if previous_metal_scans is None:
            self._active_static_metal_scans = {}
        if previous_cut_scans is None:
            self._active_static_cut_scans = {}
        if previous_obstacle_scans is None:
            self._active_static_obstacle_scans = {}
        try:
            if not self._via_legal_with_active_scans(x, y, lo, net, n):
                return False
            mirror = getattr(self, "_active_mirror_legality", None)
            if mirror is None or str(net) != str(mirror.net_a):
                return True
            axis2 = self._mirror_axis_twice_nm(mirror)
            reflected_x = axis2 - int(x)
            if not self._via_legal_with_active_scans(
                reflected_x,
                int(y),
                int(lo),
                str(mirror.net_b),
                n,
            ):
                return False
            primary_below, primary_above = (
                self._via_pad_half1[lo]
                if n == 1
                else self._via_pad_half[lo]
            )
            partner_below, partner_above = primary_below, primary_above
            for gi, primary_half, partner_half in (
                (lo, primary_below, partner_below),
                (lo + 1, primary_above, partner_above),
            ):
                primary_pad = (
                    int(x) - primary_half,
                    int(y) - primary_half,
                    int(x) + primary_half,
                    int(y) + primary_half,
                )
                partner_pad = (
                    reflected_x - partner_half,
                    int(y) - partner_half,
                    reflected_x + partner_half,
                    int(y) + partner_half,
                )
                if self._reject_active_mirror_pair_box(
                    gi,
                    primary_pad,
                    partner_pad,
                    str(mirror.net_b),
                    candidate_half=partner_half,
                ):
                    return False
            primary_ext = self._via_cut_extent(lo, n)
            primary_cut = (
                int(x) - primary_ext,
                int(y) - primary_ext,
                int(x) + primary_ext,
                int(y) + primary_ext,
            )
            partner_cut = (
                reflected_x - primary_ext,
                int(y) - primary_ext,
                reflected_x + primary_ext,
                int(y) + primary_ext,
            )
            if not _rects_clear(
                primary_cut, partner_cut, int(self._via_space[lo])
            ):
                self._record_hard_rejection(
                    "mirror_pair_in_transaction_conflict",
                    str(mirror.net_b),
                    gi=lo,
                    candidate_rect=partner_cut,
                    candidate_half=primary_ext,
                    required_space=int(self._via_space[lo]),
                    blocker_rect=primary_cut,
                    blocker_kind="mirror_primary_attempt_cut",
                )
                self._mirror_pair_conflict_seen = True
                return False
            return True
        finally:
            self._active_static_metal_scans = previous_metal_scans
            self._active_static_cut_scans = previous_cut_scans
            self._active_static_obstacle_scans = previous_obstacle_scans

    def _via_legal_with_active_scans(
        self,
        x: int,
        y: int,
        lo: int,
        net: str,
        n: Optional[int] = None,
    ) -> bool:
        ctx = getattr(self, "_active_edge_ctx", None)
        if (
            ctx is not None
            and not getattr(self, "_active_edge_ctx_bypass", False)
            and str(net) == getattr(self, "_active_edge_ctx_net", None)
        ):
            n_flag = 1 if n == 1 else 0
            res = ctx.check_via_legal_single(x, y, lo, n_flag)
            if res is not None:
                if res is False:
                    self._hard_rejection_generation = int(
                        getattr(self, "_hard_rejection_generation", 0)
                    ) + 1
                    self._active_edge_ctx_gen_bumped = int(
                        getattr(self, "_active_edge_ctx_gen_bumped", 0)
                    ) + 1
                    return False
                p_lo, p_hi, p_cut = res
                active_m = getattr(self, "_active_static_metal_scans", None)
                if active_m is not None:
                    bh_f, ah_f = (
                        self._via_pad_half1[lo]
                        if n_flag == 1
                        else self._via_pad_half[lo]
                    )
                    net_s = str(net)
                    active_m[(int(lo), (x - bh_f, y - bh_f, x + bh_f, y + bh_f), net_s)] = (p_lo, ())
                    active_m[(int(lo) + 1, (x - ah_f, y - ah_f, x + ah_f, y + ah_f), net_s)] = (p_hi, ())
                active_c = getattr(self, "_active_static_cut_scans", None)
                if active_c is not None:
                    ext_f = self._via_cut_extent(lo, n)
                    active_c[(int(lo), (x - ext_f, y - ext_f, x + ext_f, y + ext_f), str(net))] = (p_cut, (), ())
                return True
        if ctx is None:
            cpp = load_native("_gridless_cpp")
            if (
                cpp is not None
                and not getattr(self, "_search_forbidden_via_sites", ())
                and not bool(getattr(self, "_active_group_route_local_closure", False))
                and not getattr(self, "_group_route_global_interfaces_by_net", {}).get(str(net))
                and not getattr(self, "_layered_portal_wire_states", ())
                and not getattr(self, "_intent_spacing_rules", {}).get(net)
                and not getattr(self, "_mim_bottom_routing_clearance_by_gi", {}).get(lo)
                and not getattr(self, "_mim_bottom_routing_clearance_by_gi", {}).get(lo + 1)
                and not getattr(self, "_mim_carve_bottom_blockers_by_gi", {}).get(lo)
                and not getattr(self, "_mim_carve_bottom_blockers_by_gi", {}).get(lo + 1)
                and not getattr(self, "_mim_forbidden_cut_touch_by_lo", {}).get(lo)
                and not getattr(self, "_mim_top_via_marker_clearance_by_lo", {}).get(lo)
                and not (
                    (
                        getattr(self, "_mim_derived_unrelated_clearance_by_gi", {}).get(lo)
                        or getattr(self, "_mim_derived_unrelated_clearance_by_gi", {}).get(lo + 1)
                    )
                    and self._mim_derived_unrelated_static_active()
                )
                and _eg_attr("_rects_clear", _base_rects_clear) is _base_rects_clear
                and _eg_attr("_sep", _base_sep) is _base_sep
            ):
                inst_dict = self.__dict__
                if not (
                    "_evaluate_via_static_legality" in inst_dict
                    or "_compute_via_static_legality" in inst_dict
                    or "_static_or_direct_obstacle_scan" in inst_dict
                    or "_clear_obstacles" in inst_dict
                    or "_reject_fixed_same_net_via_pad_spacing" in inst_dict
                    or "_reject_intent_spacing" in inst_dict
                    or "_reject_foreign_committed_metal_contact" in inst_dict
                    or "_reject_committed_via_pad_spacing" in inst_dict
                    or "_reject_foreign_committed_metal_spacing" in inst_dict
                    or "_scan_committed_metal" in inst_dict
                    or "_static_or_direct_committed_metal_scan" in inst_dict
                    or "_scan_committed_cuts" in inst_dict
                    or "_static_or_direct_committed_cut_scan" in inst_dict
                    or "_local_fixed_cuts" in inst_dict
                    or "_raw_gap_metallised" in inst_dict
                    or "_wide_spacing_conflicts" in inst_dict
                    or "_wide_spacing_neighbors" in inst_dict
                    or "_wide_indexes" in inst_dict
                ):
                    env = getattr(self, "_routing_envelope", None)
                    env_ok = (
                        env is None
                        or type(env).__module__ == "clay.layout.route.floorplan_containment"
                    )
                    obs_lo = self._obs_idx.get(lo)
                    obs_hi = self._obs_idx.get(lo + 1)
                    obs_cut_map = getattr(self, "_obs_cut_idx", None)
                    obs_cut_lo = obs_cut_map.get(lo) if obs_cut_map is not None else None
                    has_fallback_fixed_cuts = obs_cut_lo is None and bool(
                        getattr(self, "_obs_cuts", {}).get(lo)
                    )
                    metal_lo = self._metal_idx.get(lo)
                    metal_hi = self._metal_idx.get(lo + 1)
                    cut_lo = self._cut_idx.get(lo)
                    idx_ok = (
                        not has_fallback_fixed_cuts
                        and all(
                            idx is None or (type(idx) is _SpatialIndex and idx._native is not None)
                            for idx in (obs_lo, obs_hi, obs_cut_lo, metal_lo, metal_hi, cut_lo)
                        )
                    )
                    if env_ok and idx_ok:
                        bh_f, ah_f = (
                            self._via_pad_half1[lo] if n == 1 else self._via_pad_half[lo]
                        )
                        self._ensure_fixed_wide_geometry_indexes()
                        wide_ok = True
                        wide_args: List[Any] = [None, None]
                        for wi, (wgi, whalf) in enumerate(((lo, bh_f), (lo + 1, ah_f))):
                            min_w = self._min_wide_ge_by_gi.get(wgi)
                            if min_w is not None:
                                if 2 * whalf >= min_w:
                                    wide_ok = False
                                    break
                                w_entries = self._wide_indexes(True).get(wgi, ())
                                if w_entries:
                                    w_list = []
                                    for w_sp, c_idx, _w_idx, a_idx in w_entries:
                                        if (
                                            type(c_idx) is not _SpatialIndex
                                            or c_idx._native is None
                                            or type(a_idx) is not _SpatialIndex
                                            or a_idx._native is None
                                        ):
                                            wide_ok = False
                                            break
                                        w_list.append((int(w_sp), c_idx._native, a_idx._native))
                                    if not wide_ok:
                                        break
                                    wide_args[wi] = tuple(w_list)
                        if wide_ok:
                            ext_f = self._via_cut_extent(lo, n)
                            fast_res = cpp.check_via_legal_fast_cpp(
                                x, y, bh_f, ah_f, ext_f,
                                self._space[lo], self._space[lo + 1], self._via_space[lo],
                                max(1, int(getattr(self, "_manufacturing_grid_nm", 1))),
                                self._res_cell,
                                int(round(self._pres_fac * _COST_FACTOR_SCALE)),
                                net,
                                obs_lo._native if obs_lo is not None else None,
                                obs_hi._native if obs_hi is not None else None,
                                obs_cut_lo._native if obs_cut_lo is not None else None,
                                metal_lo._native if metal_lo is not None else None,
                                metal_hi._native if metal_hi is not None else None,
                                cut_lo._native if cut_lo is not None else None,
                                env.bbox_nm if env is not None else None,
                                wide_args[0],
                                wide_args[1],
                                self._attempt_cuts.get(lo) or None,
                                self._attempt_metal.get(lo) or None,
                                self._attempt_metal.get(lo + 1) or None,
                                self._abstract_rects.get(lo) or None,
                                self._abstract_rects.get(lo + 1) or None,
                            )
                            if fast_res is not None:
                                p_lo, p_hi, p_cut = fast_res
                                active_m = getattr(self, "_active_static_metal_scans", None)
                                if active_m is not None:
                                    net_s = str(net)
                                    active_m[(int(lo), (x - bh_f, y - bh_f, x + bh_f, y + bh_f), net_s)] = (p_lo, ())
                                    active_m[(int(lo) + 1, (x - ah_f, y - ah_f, x + ah_f, y + ah_f), net_s)] = (p_hi, ())
                                active_c = getattr(self, "_active_static_cut_scans", None)
                                if active_c is not None:
                                    active_c[(int(lo), (x - ext_f, y - ext_f, x + ext_f, y + ext_f), str(net))] = (p_cut, (), ())
                                return True
        _rects_clear = _eg_attr("_rects_clear", _base_rects_clear)
        bh, ah = (self._via_pad_half1[lo] if n == 1 else self._via_pad_half[lo])
        if self._reject_group_route_global_via(x, y, lo, net, n):
            return False
        if self._via_search_identity(x, y, lo, n) in getattr(
            self, "_search_forbidden_via_sites", set()
        ):
            ext = self._via_cut_extent(lo, n)
            self._record_hard_rejection(
                "conflict_branch_via",
                net,
                gi=lo,
                candidate_rect=(
                    x - ext,
                    y - ext,
                    x + ext,
                    y + ext,
                ),
                candidate_half=ext,
            )
            return False
        legal = self._evaluate_via_static_legality(x, y, lo, net, n)
        # An immutable floorplan/artwork rejection cannot be repaired by any
        # committed-route state.  Stop before querying solve-local metal/cuts:
        # those shapes are not causal blockers for a site that the fixed world
        # already forbids, while the cached static proof above has replayed every
        # named fixed rejection.
        if not legal:
            return False
        for (gi, half) in ((lo, bh), (lo + 1, ah)):
            pad = (x - half, y - half, x + half, y + half)
            if self._reject_intent_spacing(gi, pad, net, candidate_half=half):
                legal = False
            if self._reject_foreign_committed_metal_contact(
                gi,
                pad,
                net,
                candidate_half=half,
            ):
                legal = False
            if self._reject_committed_via_pad_spacing(
                gi,
                pad,
                net,
                candidate_half=half,
            ):
                legal = False
        # The deck's via space() is net-blind (the measured VIA1.S.1 x3 were
        # SAME-net cuts 70-120nm apart -- a pin's escape via next to a direct climb
        # via -- and their 5nm-offset pads unioned into the M2.W.1 sliver). Every
        # candidate via model therefore passes its full cut-spacing rule before
        # selection, just like its bottom and top landing pads.
        ext = self._via_cut_extent(lo, n)
        cut = (x - ext, y - ext, x + ext, y + ext)
        vsp = self._via_space[lo]
        # Exact foreign committed-cut contact is electrical connectivity and is
        # absent from the search graph.  A non-identical cut inside the rule is hard
        # for every owner; normal whole-net rip-up removes same-owner cuts before
        # routing, but the helper contract remains exact.
        _present, foreign_contacts, spacing_violations = (
            self._static_or_direct_committed_cut_scan(x, y, lo, net, n)
        )
        for owner, vb in foreign_contacts:
            self._record_hard_rejection(
                "committed_cut_contact",
                owner,
                gi=lo,
                candidate_rect=cut,
                candidate_half=ext,
                required_space=0,
                blocker_rect=vb,
                blocker_kind="via_cut",
            )
            legal = False
        for owner, vb in spacing_violations:
            foreign = owner != net
            self._record_hard_rejection(
                "committed_cut_spacing" if foreign else "committed_self_cut",
                owner,
                gi=lo,
                candidate_rect=cut,
                candidate_half=ext,
                required_space=vsp,
                blocker_rect=vb,
                blocker_kind="via_cut" if foreign else None,
            )
            legal = False
        for vb in self._attempt_cuts.get(lo, ()):
            if vb == cut:
                continue
            if not _rects_clear(cut, vb, vsp):
                self._record_hard_rejection(
                    "attempt_cut",
                    net,
                    gi=lo,
                    candidate_rect=cut,
                    candidate_half=ext,
                    required_space=vsp,
                    blocker_rect=vb,
                )
                legal = False
        return legal

    def _local_fixed_cuts(self, lo: int, query: Rect) -> Tuple[Rect, ...]:
        """Canonical fixed cuts near ``query``, with an exact legacy fallback."""
        index_by_lo = getattr(self, "_obs_cut_idx", None)
        index = index_by_lo.get(lo) if index_by_lo is not None else None
        if index is None:
            rects = self._obs_cuts.get(lo, ())
        else:
            rects = (rect for _owner, rect in index.query(query))
        return tuple(sorted(tuple(_norm(rect)) for rect in rects))

    # ------------------------------------------------------------- congestion (SOFT)
    def _scan_committed_metal(
        self, gi: int, box: Rect, net: str,
    ) -> Tuple[
        int,
        Tuple[Tuple[Optional[str], Rect, str, str], ...],
    ]:
        """One local committed-metal scan for via DRC evidence and present price.

        The integer present contribution is byte-for-byte the historical
        ``_box_congestion`` calculation. Contact and positive-spacing violations
        are returned canonically so planar/via callers can consume the relevant
        class and publish deterministic causal evidence without rescanning.
        """
        _sep = _eg_attr("_sep", _base_sep)
        sp = self._space[gi]
        half = sp // 2
        x0, y0, x1, y1 = _norm(box)
        ebx = (x0 - half, y0 - half, x1 + half, y1 + half)
        rc = self._res_cell
        total = 0
        violations = []
        via_pad_claims = getattr(self, "_via_pad_provenance", {}).get(
            gi, frozenset()
        )
        midx = self._metal_idx.get(gi)
        if midx is not None:
            pres = int(round(self._pres_fac * _COST_FACTOR_SCALE))
            if (
                _sep is _base_sep
                and type(midx) is _SpatialIndex
                and midx._native is not None
            ):
                total, raw_violations = midx._native.scan_committed_metal(
                    x0, y0, x1, y1, sp, rc, pres, net
                )
                if not raw_violations:
                    return total, ()
                for owner, blocker, is_contact in raw_violations:
                    is_via_pad = (owner, blocker) in via_pad_claims
                    if is_contact:
                        violations.append((
                            owner,
                            blocker,
                            (
                                "committed_via_pad_contact"
                                if is_via_pad
                                else "committed_metal_contact"
                            ),
                            "via_pad" if is_via_pad else "route_metal",
                        ))
                    else:
                        violations.append((
                            owner,
                            blocker,
                            (
                                "committed_via_pad_spacing"
                                if is_via_pad
                                else "committed_metal_spacing"
                            ),
                            "via_pad" if is_via_pad else "route_metal",
                        ))
                violations.sort(key=lambda item: (
                    "" if item[0] is None else str(item[0]),
                    item[1],
                    item[2],
                    item[3],
                ))
                return total, tuple(violations)
            q = (x0 - sp, y0 - sp, x1 + sp, y1 + sp)
            for owner, rect in midx.query(q):
                if owner == net:
                    continue
                blocker = _norm(rect)
                sx, sy = _sep((x0, y0, x1, y1), blocker)
                if (sx, sy) == (0, 0):
                    is_via_pad = (owner, blocker) in via_pad_claims
                    violations.append((
                        owner,
                        blocker,
                        (
                            "committed_via_pad_contact"
                            if is_via_pad
                            else "committed_metal_contact"
                        ),
                        "via_pad" if is_via_pad else "route_metal",
                    ))
                elif sx * sx + sy * sy < sp * sp:
                    is_via_pad = (owner, blocker) in via_pad_claims
                    violations.append((
                        owner,
                        blocker,
                        (
                            "committed_via_pad_spacing"
                            if is_via_pad
                            else "committed_metal_spacing"
                        ),
                        "via_pad" if is_via_pad else "route_metal",
                    ))
                rx0, ry0, rx1, ry1 = blocker
                rbx = (rx0 - half, ry0 - half, rx1 + half, ry1 + half)
                ox0 = max(ebx[0], rbx[0])
                oy0 = max(ebx[1], rbx[1])
                ox1 = min(ebx[2], rbx[2])
                oy1 = min(ebx[3], rbx[3])
                if ox0 >= ox1 or oy0 >= oy1:
                    continue
                ncx = max(1, (ox1 - ox0) // rc)
                ncy = max(1, (oy1 - oy0) // rc)
                total += pres * ncx * ncy
        violations.sort(key=lambda item: (
            "" if item[0] is None else str(item[0]),
            item[1],
            item[2],
            item[3],
        ))
        return total, tuple(violations)

    def _static_or_direct_committed_metal_scan(
        self, gi: int, box: Rect, net: str,
    ) -> Tuple[
        int,
        Tuple[Tuple[Optional[str], Rect, str, str], ...],
    ]:
        key = (int(gi), tuple(_norm(box)), str(net))
        active = getattr(self, "_active_static_metal_scans", None)
        if active is None:
            return self._scan_committed_metal(gi, box, net)
        cached = active.get(key)
        if cached is not None:
            return cached
        scanned = self._scan_committed_metal(gi, box, net)
        active[key] = scanned
        return scanned

    def _box_congestion(self, gi: int, box: Rect, net: str) -> int:
        """Congestion of drawing ``box`` on ``gi`` for ``net``:

          * PRESENT congestion -- for every FOREIGN committed-metal band the box's
            min-space band overlaps, ``pres_fac`` per shared resource cell (this
            round's already-routed nets repel later ones).
          * HISTORY -- the box's OWN resource cells carry a PERSISTENT per-cell
            penalty accumulated every round they were over-used, charged REGARDLESS
            of whether a foreign net is currently there. This is the load-bearing
            PathFinder mechanism: it steers a net off a chronically-contended cell
            even when the OTHER net has not been re-routed yet (the exact case here --
            the widest-span net routes FIRST every round, so present congestion alone
            never sees its future contender; only persistent history can teach it to
            vacate the shared column / cross on another layer).

        Scaled to wirelength (pitch * per-nm step) so it is commensurate with detour
        cost and PDK-scale-free -- edges a full min-space apart (bands just touch)
        overlap 0 present cells, and an uncontended cell has 0 history -> ZERO cost."""
        ctx = getattr(self, "_active_edge_ctx", None)
        if (
            ctx is not None
            and not getattr(self, "_active_edge_ctx_bypass", False)
            and str(net) == getattr(self, "_active_edge_ctx_net", None)
        ):
            nx0, ny0, nx1, ny1 = _norm(box)
            active_m = getattr(self, "_active_static_metal_scans", None)
            phint = -1
            if active_m is not None:
                cached_m = active_m.get((int(gi), (nx0, ny0, nx1, ny1), str(net)))
                if cached_m is not None:
                    phint = int(cached_m[0])
            half_min = (min(nx1 - nx0, ny1 - ny0) + 1) // 2
            c_fast = ctx.box_congestion(gi, nx0, ny0, nx1, ny1, half_min, phint)
            if c_fast is not None:
                return c_fast
        _sep = _eg_attr("_sep", _base_sep)
        sp = self._space[gi]
        half = sp // 2
        x0, y0, x1, y1 = _norm(box)
        ebx = (x0 - half, y0 - half, x1 + half, y1 + half)
        rc = self._res_cell
        pres = int(round(self._pres_fac * _COST_FACTOR_SCALE))
        # PRESENT: foreign committed metal within min-space. The A* static seam
        # precomputes this together with hard contact; direct callers scan here.
        total, _contacts = self._static_or_direct_committed_metal_scan(
            gi, (x0, y0, x1, y1), net
        )
        # OWN-net ARTWORK + OWN in-attempt METAL band (R7a): the deck's space() is
        # net-blind, so a box in the OPEN band (0, min_space) of the net's OWN
        # artwork -- or of its own EARLIER MST edges' metal -- is a real deck
        # violation unless the gap is metallised. It must not HARD-seal access
        # (hard-banding exhausted whole route graphs: the R6 gnd/vout failure), so
        # it is PRICED like present congestion -- escalating with ``pres_fac``
        # until the touch/clear (merge) alternative wins; ``_detect_space_residuals``
        # + victims close the loop. Euclidean band, consistent with the detector.
        q2 = (x0 - sp, y0 - sp, x1 + sp, y1 + sp)
        spsq = sp * sp
        oidx = self._obs_idx.get(gi)
        if oidx is not None:
            if (
                _sep is _base_sep
                and type(oidx) is _SpatialIndex
                and oidx._native is not None
            ):
                total += pres * oidx._native.scan_same_net_band_count(
                    x0, y0, x1, y1, sp, net, True
                )
            else:
                for (owner, r) in oidx.query(q2):
                    if owner != net:
                        continue
                    sx, sy = _sep((x0, y0, x1, y1), _norm(r))
                    if (sx or sy) and (sx * sx + sy * sy) < spsq:
                        total += pres  # one banded own rect ~ one shared resource cell
        aidx = self._attempt_idx.get(gi)
        if aidx is not None:
            if (
                _sep is _base_sep
                and type(aidx) is _SpatialIndex
                and aidx._native is not None
            ):
                total += pres * aidx._native.scan_same_net_band_count(
                    x0, y0, x1, y1, sp, None, False
                )
            else:
                for (_own, r) in aidx.query(q2):
                    sx, sy = _sep((x0, y0, x1, y1), _norm(r))
                    if (sx or sy) and (sx * sx + sy * sy) < spsq:
                        total += pres  # jittered near-own-track: pay; overlap: free
        for owner, blocker, required, _kind, part in self._wide_spacing_neighbors(
            gi, (x0, y0, x1, y1), include_committed=True,
        ):
            sx, sy = _sep(part, blocker)
            if (sx or sy) and sx * sx + sy * sy < required * required:
                total += pres
        # HISTORY: query the integer-addressed cells under THIS edge.  Provisional
        # conflicts can populate a large design-wide history table; scanning that
        # table for every A* successor multiplies bounded search work by global
        # congestion.  Direct local lookup preserves the exact overlap semantics.
        total += self._history_congestion_units(gi, ebx)
        numerator = total * rc * _STEP_COST_PER_NM * _CONG_WEIGHT
        return (numerator + _COST_FACTOR_SCALE // 2) // _COST_FACTOR_SCALE

    def _history_congestion_units(self, resource_gi: int, box: Rect) -> int:
        """Return fixed-point history units for resource cells intersecting box."""
        history = self._history
        if not history:
            return 0
        bx0, by0, bx1, by1 = _norm(box)
        rc = self._res_cell
        if type(history) is dict:
            hgrid = getattr(self, "_active_history_grid", None)
            if hgrid is not None:
                return hgrid.eval_units(resource_gi, bx0, by0, bx1, by1, rc)
        total = 0
        # Include one conservative neighbour at the low side, then retain the
        # previous strict rectangle-overlap predicate.  The extra boundary probes
        # are constant work and cannot change the charged cell set.
        for cx in range((bx0 - rc) // rc, bx1 // rc + 2):
            cxl = cx * rc
            if not (cxl < bx1 and cxl + rc > bx0):
                continue
            for cy in range((by0 - rc) // rc, by1 // rc + 2):
                cyl = cy * rc
                if not (cyl < by1 and cyl + rc > by0):
                    continue
                hv = history.get((cx, cy, resource_gi), 0.0)
                if hv:
                    total += int(round(
                        hv * self.hist_fac * _COST_FACTOR_SCALE
                    ))
        return total

    def _edge_congestion(self, gi: int, ax: int, ay: int, bx: int, by: int,
                         net: str) -> int:
        wb = _seg_bbox(ax, ay, bx, by, self._hw(net, gi))
        return self._box_congestion(gi, wb, net)

    def _via_cut_congestion(
        self, x: int, y: int, lo: int, net: str, n: Optional[int] = None,
    ) -> int:
        """Fixed-point present/history price for one provisional cut resource."""
        ext = self._via_cut_extent(lo, n)
        x0, y0, x1, y1 = (x - ext, y - ext, x + ext, y + ext)
        vsp = self._via_space[lo]
        half = vsp // 2
        ebx = (x0 - half, y0 - half, x1 + half, y1 + half)
        rc = self._res_cell
        total, _contacts, _spacing_violations = (
            self._static_or_direct_committed_cut_scan(x, y, lo, net, n)
        )
        resource_gi = self._n_layers + lo
        total += self._history_congestion_units(resource_gi, ebx)
        numerator = total * rc * _STEP_COST_PER_NM * _CONG_WEIGHT
        return (numerator + _COST_FACTOR_SCALE // 2) // _COST_FACTOR_SCALE

    def _via_congestion(
        self, x: int, y: int, lo: int, net: str, n: Optional[int] = None,
    ) -> int:
        bh, ah = (self._via_pad_half1[lo] if n == 1 else self._via_pad_half[lo])
        c = self._box_congestion(lo, (x - bh, y - bh, x + bh, y + bh), net)
        c += self._box_congestion(lo + 1, (x - ah, y - ah, x + ah, y + ah), net)
        c += self._via_cut_congestion(x, y, lo, net, n)
        return c

