"""Same-net legalization, GDS polygon emission, and DRC space/short residual scanning mixin."""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from clay.layout.backend import LayerPolygon
from clay.layout.minimum_area import minimum_area_square_side_nm
from clay.layout.route.engine import NetRoute
from clay.layout.route.engine_maze import (
    _NM_PER_UM as _base_NM_PER_UM,
    _via_name_for,
)
from clay.layout.route.gridless_types import (
    Rect,
    _DeckNegotiationSnapshot,
    _EmittedNetGeometry,
    _RoutedNet as _base_RoutedNet,
    _SpatialIndex,
    _ViaSite,
    _WireSeg,
    _closed_rects_touch,
    _contact_line_end_repair_rects,
    _eg_attr,
    _marker_selected_conductor_components,
    _norm,
    _pop,
    _rect,
    _rects_clear as _base_rects_clear,
    _seg_bbox,
    _sep as _base_sep,
)
from clay.layout.route.metal_junction_legalization import legalize_island_junctions
from clay.layout.route.spacing_intent import check_spacing_geometry, spacing_nm
from clay.layout.route.via_model import landing_rect_for_cut, subtract_rects


class _GridlessEmitDrcMixin:
    """Same-net fill legalization, polygon emission, and deck-aligned residual scans."""

    def _legalize_committed_for_layers(
        self, committed: Dict[str, _RoutedNet],
        allowed_layers_by_net: Mapping[str, Sequence[int]],
    ) -> None:
        """Transport actual lowered layer permissions to final fill creation."""
        _RoutedNet = _eg_attr("_RoutedNet", _base_RoutedNet)
        missing = object()
        previous = getattr(self, "_final_legalize_allowed_layers", missing)
        self._final_legalize_allowed_layers = {
            str(net): tuple(layers) for net, layers in allowed_layers_by_net.items()
        }
        try:
            self._legalize_committed(committed)
        finally:
            if previous is missing:
                del self._final_legalize_allowed_layers
            else:
                self._final_legalize_allowed_layers = previous

    def _legalize_final_emission(
        self, committed: Dict[str, _RoutedNet],
        allowed_layers_by_net: Mapping[str, Sequence[int]],
    ) -> None:
        """Finalize the emitted world, then close supported source junctions."""
        _RoutedNet = _eg_attr("_RoutedNet", _base_RoutedNet)
        from clay.layout.route.source_junction_kernel import (
            committed_world_matches_book, repair_source_junctions,
        )

        self._legalize_committed_for_layers(committed, allowed_layers_by_net)
        if not getattr(self, '_fixed_deck_literal_rects', ()):
            return
        if not committed_world_matches_book(self, committed):
            return

        def publish_junction(net, candidate):
            previous_route = committed[net]
            try:
                committed[net] = candidate
                self._rebuild_committed_state(committed)
            except Exception:
                committed[net] = previous_route
                self._rebuild_committed_state(committed)
                raise

        repair_source_junctions(
            self, committed, allowed_layers_by_net, publish_junction,
        )

    def _legalize_committed(self, committed: Dict[str, _RoutedNet]) -> None:
        """FINAL merge-or-clear materialization on the solution actually EMITTED.

        The per-attempt ``_legalize_same_net`` judges its fills against the
        world AS OF that attempt; rip-up then moves foreign metal, and the BEST
        snapshot the engine emits can retain same-net band gaps whose fill was
        (then) illegal or not yet needed -- the measured residual plateau
        (M2.S.1 / M3.S.1 on the real deck: a via pad 232nm from its own stamp
        riser pad). This pass re-runs the SAME materialization per net against
        the FINAL committed world, so every gap that is legally closable in the
        artifact the caller receives IS closed, and what remains is a true
        foreign-blocked residual (honest ``success=False``). The book / metal
        index are rebuilt from ``committed`` after each net so later nets'
        cover / foreign judgments see earlier nets' fills. Deterministic net
        order; no wall-clock participation."""
        _RoutedNet = _eg_attr("_RoutedNet", _base_RoutedNet)
        ordered_names = tuple(sorted(committed))

        def _checkpoint(
            name: str,
            net_index: int,
            rn: _RoutedNet,
            subphase: str,
        ) -> None:
            details = dict(getattr(self, "_route_watchdog_details", ()))
            details.update({
                "final_legalize_net_count": len(ordered_names),
                "final_legalize_net_index": int(net_index),
                "final_legalize_segment_count": len(rn.segs),
                "final_legalize_subphase": str(subphase),
                "final_legalize_via_count": len(rn.vias),
            })
            self._route_watchdog_details = tuple(sorted(details.items()))
            self._watchdog_check(name, f"final_legalize_{subphase}")

        def _run_phase(
            name: str,
            net_index: int,
            rn: _RoutedNet,
            subphase: str,
            callback,
        ):
            _checkpoint(name, net_index, rn, subphase)
            result = callback()
            # Region/index kernels do not all carry inner watchdog hooks.  Keep
            # the active subphase until after they return so a crossed deadline
            # is attributed to the work just performed, never the next net.
            _checkpoint(name, net_index, rn, subphase)
            return result

        if ordered_names:
            first_name = ordered_names[0]
            first_rn = committed[first_name]
            _run_phase(
                first_name,
                0,
                first_rn,
                "committed_state_rebuild",
                lambda: self._rebuild_committed_state(committed),
            )
        else:
            self._rebuild_committed_state(committed)
        book = self._book
        index_dirty = False
        for net_index, name in enumerate(ordered_names):
            rn = committed[name]
            _checkpoint(name, net_index, rn, "net_entry")
            if not (rn.segs or rn.vias):
                continue
            # ``_rebuild_committed_state`` already built the exact final-world
            # index.  Rebuild only after an earlier net actually added geometry;
            # clean nets leave the same book/index truth valid for every later
            # foreign-metal query.  The old unconditional rebuild made a clean
            # N-net final snapshot rescan the whole T-rectangle book N+2 times.
            if index_dirty:
                _run_phase(
                    name,
                    net_index,
                    rn,
                    "index_rebuild",
                    self._rebuild_metal_index,
                )
                index_dirty = False
            n_before = len(rn.segs)
            allowed = getattr(self, "_final_legalize_allowed_layers", {}).get(name, ())
            overrides = self._rail_local_half_width_overrides(
                name, allowed, self._rail_by_net.get(name, ()),
            )
            # Only new fill creation receives local width. Original implicit
            # wide wires must retain their width while geometry is measured.
            with self._wide_repair_transaction(name, rn), self._half_width_scope(
                overrides, attribute="_active_fill_half_width_overrides",
            ):
                _run_phase(
                    name,
                    net_index,
                    rn,
                    "same_net",
                    lambda: self._legalize_same_net(name, rn),
                )
                _run_phase(
                    name, net_index, rn, "wide_same_net",
                    lambda: self._legalize_wide_same_net(name, rn),
                )
                width_additions = _run_phase(
                    name,
                    net_index,
                    rn,
                    "merged_width_1",
                    lambda: self._legalize_merged_width_necks(name, rn),
                )
                contact_additions = _run_phase(
                    name,
                    net_index,
                    rn,
                    "contact_line_ends",
                    lambda: self._legalize_contact_line_ends(name, rn),
                )
                # A first neck repair can itself expose a second-generation island;
                # contact-line-end metal can do the same.  Repeat only when either
                # phase changed geometry.  With two zero additions the second scan
                # would be the identical region fixed point.
                if width_additions or contact_additions:
                    _run_phase(
                        name,
                        net_index,
                        rn,
                        "merged_width_2",
                        lambda: self._legalize_merged_width_necks(name, rn),
                    )
                _run_phase(
                    name,
                    net_index,
                    rn,
                    "min_hole_areas",
                    lambda: self._legalize_min_hole_areas(name, rn),
                )
            if len(rn.segs) != n_before:
                book.remove(name)
                metal_by_gi, cuts = self._net_metal(rn)
                book.add(name, metal_by_gi, cuts, self._net_via_pads(rn))
                index_dirty = True
        if index_dirty:
            last_index = len(ordered_names) - 1
            last_name = ordered_names[last_index]
            _run_phase(
                last_name,
                last_index,
                committed[last_name],
                "index_rebuild",
                self._rebuild_metal_index,
            )

    def _legalize_min_hole_areas(self, net: str, rn: _RoutedNet) -> int:
        """Close sub-min-hole-area enclosed loops formed by single-net metal."""
        tech = getattr(self, "_tech", None)
        pdk = getattr(tech, "pdk", None) if tech is not None else None
        if tech is None or pdk is None or not hasattr(pdk, "min_hole_area"):
            return 0
        import klayout.db as kdb

        added = 0
        metal_names = getattr(tech, "metal_names", {}) or {}
        gi_to_pdk = getattr(tech, "_gi_to_pdk", None)
        for gi in range(len(self._half_w)):
            pdk_n = gi_to_pdk[gi] if gi_to_pdk is not None else gi + 1
            layer_name = metal_names.get(pdk_n, f"metal{pdk_n}")
            min_hole_area = pdk.min_hole_area(layer_name)
            if min_hole_area is None or int(min_hole_area) <= 0:
                continue
            min_hole_area = int(min_hole_area)
            metal, _ = self._net_metal(rn)
            routed_boxes = [_norm(b) for b in metal.get(gi, ())]
            if not routed_boxes:
                continue
            layer_key = tuple(tech.gi_to_metal_key(gi))
            abstract = self._abstract_rects.get(gi, ())
            fixed_boxes: List[Rect] = []
            if getattr(self, "_fixed_deck_literal_rects", ()):
                fixed_boxes.extend(
                    _norm(rect)
                    for drawn_layer, owner, rect in self._fixed_deck_literal_rects
                    if tuple(drawn_layer) == layer_key and owner == net and rect not in abstract
                )
            elif gi in getattr(self, "_obs_by_gi", ()):
                fixed_boxes.extend(
                    _norm(rect) for owner, rect in self._obs_by_gi[gi] if owner == net
                )
            routed_reg = kdb.Region([kdb.Box(*b) for b in routed_boxes]).merged()
            full_reg = kdb.Region([kdb.Box(*b) for b in (*routed_boxes, *fixed_boxes)]).merged()
            bad_holes = [
                poly
                for poly in full_reg.holes().each()
                if 0 < int(poly.area()) < min_hole_area
                and not (kdb.Region(poly).sized(1) & routed_reg).is_empty()
            ]
            if not bad_holes:
                continue
            own_idx = _SpatialIndex()
            for box in routed_boxes:
                own_idx.add(net, box)
            cur_bad_count = len(bad_holes)
            for hole_poly in bad_holes:
                cand_segs = []
                cand_boxes: List[Rect] = []
                seen_rects = set(routed_boxes)
                for tile in kdb.Region(hole_poly).decompose_trapezoids():
                    tb = tile.bbox()
                    for seg in self._fill_segs(net, gi, (tb.left, tb.bottom, tb.right, tb.top)):
                        hw = int(self._segment_half_width(net, seg))
                        actual_box = _norm((
                            min(seg.ax, seg.bx) - hw,
                            min(seg.ay, seg.by) - hw,
                            max(seg.ax, seg.bx) + hw,
                            max(seg.ay, seg.by) + hw,
                        ))
                        if actual_box in seen_rects:
                            continue
                        seen_rects.add(actual_box)
                        cand_segs.append(seg)
                        cand_boxes.append(actual_box)
                if not cand_boxes:
                    continue
                if not self._fill_legal(
                    net,
                    gi,
                    cand_boxes,
                    own_idx={gi: own_idx},
                    check_own_junctions=False,
                    current_route=rn,
                ):
                    continue
                before_metal, _ = self._net_metal(rn)
                after_metal = {k: list(v) for k, v in before_metal.items()}
                after_metal[gi] = [*after_metal.get(gi, ()), *cand_boxes]
                if not self._repair_preserves_world(net, before_metal, after_metal):
                    continue
                after_reg = kdb.Region(
                    [kdb.Box(*b) for b in (*after_metal[gi], *fixed_boxes)]
                ).merged()
                after_bad_count = sum(
                    1
                    for poly in after_reg.holes().each()
                    if 0 < int(poly.area()) < min_hole_area
                    and not (kdb.Region(poly).sized(1) & (routed_reg + kdb.Region([kdb.Box(*b) for b in cand_boxes])).merged()).is_empty()
                )
                if after_bad_count >= cur_bad_count:
                    continue
                rn.segs.extend(cand_segs)
                book = getattr(self, "_book", None)
                if book is not None:
                    for actual_box in cand_boxes:
                        book.metal[gi].append((net, actual_box))
                    self._rebuild_metal_index()
                for actual_box in cand_boxes:
                    own_idx.add(net, actual_box)
                routed_boxes.extend(cand_boxes)
                routed_reg = (routed_reg + kdb.Region([kdb.Box(*b) for b in cand_boxes])).merged()
                cur_bad_count = after_bad_count
                added += len(cand_segs)
        return added

    def _legalize_wide_same_net(self, net: str, rn: _RoutedNet) -> int:
        """Allocate only exact wide-rule obligations through the owned fill path."""
        _RoutedNet = _eg_attr("_RoutedNet", _base_RoutedNet)
        import klayout.db as kdb
        from .wide_metal_spacing import (
            exact_region_rectangles, geometry_spacing_entries,
            wide_spacing_additions, wide_spacing_checks,
        )

        tech = getattr(self, "_tech", None)
        if tech is None:
            return 0
        metal, _cuts = self._net_metal(rn)
        total = 0
        for gi, route_rectangles in sorted(metal.items()):
            self._watchdog_check(net, "wide_same_net_legalize")
            name = tech.metal_names[tech._gi_to_pdk[gi]]
            if not geometry_spacing_entries(tech.pdk, name):
                continue
            own, foreign = kdb.Region(), kdb.Region()
            for rectangle in route_rectangles:
                own.insert(kdb.Box(*_norm(rectangle)))
            for owner, rectangle in self._fixed_literal_metal_rows().get(gi, ()):
                (own if owner == net else foreign).insert(kdb.Box(*_norm(rectangle)))
            for owner, rectangle in self._book.metal.get(gi, ()):
                if owner != net:
                    foreign.insert(kdb.Box(*_norm(rectangle)))
            own.merge()
            foreign.merge()
            extra = wide_spacing_additions(own, foreign, tech.pdk, name)
            if extra.is_empty():
                continue
            segments = []
            for rectangle in exact_region_rectangles(extra):
                segments.extend(self._fill_segs(net, gi, rectangle))
            actual_boxes = tuple(_seg_bbox(s.ax, s.ay, s.bx, s.by,
                                          self._segment_half_width(net, s)) for s in segments)
            actual = kdb.Region()
            for rectangle in actual_boxes:
                actual.insert(kdb.Box(*rectangle))
            # Existing min-width tiling can overlap already-owned metal, but
            # cannot enlarge this exact justified repair into another region.
            if not (extra - actual).is_empty() or not (actual - own - extra).is_empty():
                continue
            if not self._fill_legal(net, gi, actual_boxes, check_own_junctions=False,
                                    current_route=rn):
                continue
            before_world = (own + foreign).merged()
            after_world = (before_world + actual).merged()
            old_checks = wide_spacing_checks(before_world, tech.pdk, name)
            new_checks = wide_spacing_checks(after_world, tech.pdk, name)
            if any(not (new.errors.polygons(0) - old.errors.polygons(0)).is_empty()
                   for old, new in zip(old_checks, new_checks)):
                continue
            rn.segs.extend(segments)
            total += len(segments)
        return total

    def _legalize_contact_line_ends(self, net: str, rn: _RoutedNet) -> int:
        """Allocate PDK M1-over-contact EOL repairs into the routed net.

        The exact emitter is merge-only and must never invent this material.
        Work from the final route union plus literally drawn, ownership-tagged
        device M1, then append only foreign/floorplan-legal wire rectangles to
        ``rn`` so the ordinary allocated-route manifest accounts for them.
        """
        _RoutedNet = _eg_attr("_RoutedNet", _base_RoutedNet)
        import klayout.db as kdb

        tech_adapter = getattr(self, "_tech", None)
        if tech_adapter is None:
            return 0
        pdk = tech_adapter.pdk
        raw_line_end = pdk.line_end_ext("metal1")
        if raw_line_end is None:
            return 0
        min_width = pdk.min_width("metal1")
        if (
            type(raw_line_end) is not int
            or raw_line_end <= 0
            or type(min_width) is not int
            or min_width <= 0
        ):
            raise ValueError("metal1 line-end legalization lacks exact PDK rules")
        line_end = raw_line_end
        m1_key = tuple(pdk.metal_key(1))
        contact_key = tuple(pdk.contact_key())
        if (
            len(m1_key) != 2
            or len(contact_key) != 2
            or any(type(value) is not int for value in (*m1_key, *contact_key))
        ):
            raise ValueError("metal1 line-end legalization lacks canonical layers")
        gi = tech_adapter.resolve_layer_to_gi(m1_key)
        if gi is None:
            return 0

        route_metal, _cuts = self._net_metal(rn)
        route_rects = tuple(route_metal.get(gi, ()))
        if not route_rects:
            return 0
        route_region = kdb.Region()
        for rect in route_rects:
            route_region.insert(kdb.Box(*_norm(rect)))

        own_artwork = kdb.Region()
        abstract = set(self._abstract_rects.get(gi, ()))
        for owner, raw_rect in self._obs_by_gi.get(gi, ()):
            rect = _norm(raw_rect)
            if owner == net and rect not in abstract:
                own_artwork.insert(kdb.Box(*rect))
        own_artwork.merge()
        joined = (route_region + own_artwork).merged()
        if joined.is_empty():
            return 0

        contacts = kdb.Region()
        for raw_rect in getattr(self, "_fixed_contact_rects", ()):
            cut = kdb.Region(kdb.Box(*_norm(raw_rect)))
            if not (cut & joined).is_empty():
                contacts += cut
        contacts.merge()
        if contacts.is_empty():
            return 0

        # Keep the exact edge threshold mechanically identical to the deck
        # generator: 2.5x parsed minimum width (3x EOL only when width is absent;
        # the latter cannot occur here because min_width is required above).
        eol_edge_max = int(round(min_width * 2.5))
        repairs = _contact_line_end_repair_rects(
            joined,
            contacts,
            line_end_nm=line_end,
            eol_edge_max_nm=eol_edge_max,
        )
        added: List[_WireSeg] = []
        for repair in repairs:
            segments = self._fill_segs(net, gi, repair)
            boxes = tuple(
                _seg_bbox(
                    segment.ax,
                    segment.ay,
                    segment.bx,
                    segment.by,
                    self._segment_half_width(net, segment),
                )
                for segment in segments
            )
            materialized = kdb.Region()
            for box in boxes:
                materialized.insert(kdb.Box(*box))
            if (kdb.Region(kdb.Box(*repair)) - materialized).is_empty() and (
                self._fill_legal(
                    net,
                    gi,
                    boxes,
                    check_own_junctions=False,
                    current_route=rn,
                )
            ):
                added.extend(segments)
        if added and getattr(self, "_active_wide_repair_route", None) is not rn:
            trial = _RoutedNet(name=net, segs=[*rn.segs, *added], vias=list(rn.vias),
                               connected=rn.connected)
            if not self._repair_preserves_world(
                net, route_metal, self._net_metal(trial)[0],
            ):
                return 0
        rn.segs.extend(added)
        return len(added)

    def _legalize_merged_width_necks(self, net: str, rn: _RoutedNet) -> int:
        """Materialize route-owned fixes for emergent full-island necks.

        Pairwise rectangle contact is necessary but not sufficient for the
        deck's width/notch checks: merging several exact route rectangles with
        a device terminal component can create a reflex-corner neck visible
        only on the full polygon union.  The emitter audits that union but may
        not change it.  This final router transaction applies the shared exact
        region kernel, attributes only islands touched by ``rn`` to this net,
        checks every resulting rectangle against the router's normal foreign
        and floorplan predicates, and appends the accepted metal to ``rn`` so
        it is covered by the allocated route manifest.
        """
        _RoutedNet = _eg_attr("_RoutedNet", _base_RoutedNet)
        import klayout.db as kdb

        route_metal, _cuts = self._net_metal(rn)
        added_segment_count = 0
        for gi in sorted(route_metal):
            min_width = 2 * int(self._half_w[gi])
            min_space = int(self._space[gi])
            if min_width <= 0 or min_space <= 0:
                continue

            route_region = kdb.Region()
            for rect in route_metal.get(gi, ()):
                route_region.insert(kdb.Box(*_norm(rect)))
            route_region.merge()
            if route_region.is_empty():
                continue

            own = route_region.dup()
            blockers = kdb.Region()
            abstract = set(self._abstract_rects.get(gi, ()))
            for owner, raw_rect in self._obs_by_gi.get(gi, ()):
                rect = _norm(raw_rect)
                if owner == net and rect not in abstract:
                    own.insert(kdb.Box(*rect))
                else:
                    blockers.insert(kdb.Box(*rect))
            for owner, raw_rect in self._book.metal.get(gi, ()):
                if owner != net:
                    blockers.insert(kdb.Box(*_norm(raw_rect)))
            own.merge()
            blockers.merge()

            own_idx: Dict[int, _SpatialIndex] = {gi: _SpatialIndex()}
            for shape in own.decompose_trapezoids():
                bbox = shape.polygon.bbox()
                own_idx[gi].add(
                    net,
                    (bbox.left, bbox.bottom, bbox.right, bbox.top),
                )

            islands = sorted(
                (kdb.Region(polygon) for polygon in own.each()),
                key=lambda region: (
                    region.bbox().left,
                    region.bbox().bottom,
                    region.bbox().right,
                    region.bbox().top,
                ),
            )
            for island in islands:
                if (island & route_region).is_empty():
                    continue
                _repaired, generations = legalize_island_junctions(
                    island,
                    (blockers + (own - island)).sized(min_space),
                    min_width_nm=min_width,
                    min_space_nm=min_space,
                    manufacturing_grid_nm=getattr(
                        self, "_manufacturing_grid_nm", 1
                    ),
                )
                for generation in generations:
                    new_segments: List[_WireSeg] = []
                    exact_boxes: List[Rect] = []
                    for shape in generation.decompose_trapezoids():
                        polygon = kdb.Polygon(shape.polygon)
                        bbox = polygon.bbox()
                        box = kdb.Box(
                            bbox.left, bbox.bottom, bbox.right, bbox.top
                        )
                        # Router currency is axis-aligned wire rectangles.  A
                        # Euclidean opening can leave a shallow trapezoid; its
                        # bounding box is the minimal rectilinear cover.  The
                        # complete cover is checked below against foreign metal,
                        # floorplan containment and the required repair region.
                        rect = (box.left, box.bottom, box.right, box.top)
                        segments = self._fill_segs(net, gi, rect)
                        new_segments.extend(segments)
                        exact_boxes.extend(
                            _seg_bbox(
                                segment.ax,
                                segment.ay,
                                segment.bx,
                                segment.by,
                                self._segment_half_width(net, segment),
                            )
                            for segment in segments
                        )
                    if not new_segments:
                        break
                    materialized = kdb.Region()
                    for rect in exact_boxes:
                        materialized.insert(kdb.Box(*rect))
                    materialized.merge()
                    if not (generation - materialized).is_empty():
                        break
                    if not self._fill_legal(
                        net,
                        gi,
                        exact_boxes,
                        own_idx=own_idx,
                        check_own_junctions=False,
                    ):
                        break
                    trial_island = (island + materialized).merged()
                    _trial_repaired, trial_additions = legalize_island_junctions(
                        trial_island,
                        (blockers + (own - island)).sized(min_space),
                        min_width_nm=min_width,
                        min_space_nm=min_space,
                        manufacturing_grid_nm=getattr(
                            self, "_manufacturing_grid_nm", 1
                        ),
                    )
                    if trial_additions:
                        break
                    rn.segs.extend(new_segments)
                    added_segment_count += len(new_segments)
                    route_region += materialized
                    route_region.merge()
                    island = trial_island
                    own += materialized
                    own.merge()
                    for rect in exact_boxes:
                        own_idx[gi].add(net, rect)
        return added_segment_count

    def _legalize_group_port_baseline(
        self,
        net: str,
        baseline: _RoutedNet,
        envelope_nm: Tuple[int, int, int, int],
    ) -> None:
        _RoutedNet = _eg_attr("_RoutedNet", _base_RoutedNet)
        import klayout.db as kdb

        metal_by_gi, _cuts = self._net_metal(baseline)
        needs_repair = False
        for gi, rects in metal_by_gi.items():
            if not rects:
                continue
            sp = int(self._space[gi])
            oidx = self._obs_idx.get(gi)
            abs_gi = self._abstract_rects.get(gi, ())
            fixed_own: Set[Rect] = set()
            if oidx is not None:
                for r in rects:
                    nr = _norm(r)
                    q = (nr[0] - sp, nr[1] - sp, nr[2] + sp, nr[3] + sp)
                    for owner, obs_r in oidx.query(q):
                        if owner == net and obs_r not in abs_gi:
                            fixed_own.add(_norm(obs_r))
            before_reg = kdb.Region()
            for fr in fixed_own:
                before_reg.insert(kdb.Box(*fr))
            before_reg.merge()
            after_reg = before_reg.dup()
            for r in rects:
                after_reg.insert(kdb.Box(*_norm(r)))
            after_reg.merge()
            before_err = kdb.Edges()
            for ep in before_reg.space_check(sp):
                before_err.insert(ep.first)
                before_err.insert(ep.second)
            for ep in before_reg.notch_check(sp):
                before_err.insert(ep.first)
                before_err.insert(ep.second)
            after_err = kdb.Edges()
            for ep in after_reg.space_check(sp):
                after_err.insert(ep.first)
                after_err.insert(ep.second)
            for ep in after_reg.notch_check(sp):
                after_err.insert(ep.first)
                after_err.insert(ep.second)
            if not (after_err - before_err).is_empty():
                needs_repair = True
                break
        if not needs_repair:
            return
        saved_closure = getattr(self, "_active_group_route_local_closure", False)
        saved_bounds = self._bounds
        self._active_group_route_local_closure = True
        self._bounds = tuple(envelope_nm)
        try:
            self._legalize_same_net(net, baseline, allow_flush_ear_fills=True)
        finally:
            self._active_group_route_local_closure = saved_closure
            self._bounds = saved_bounds

    def _legalize_same_net(
        self, net: str, rn: _RoutedNet, *,
        active_metal: Optional[Set[Tuple[int, Rect]]] = None,
        allow_flush_ear_fills: bool = False,
    ) -> None:
        """SNAP-TO-MERGE: materialise the TOUCH-OR-CLEAR obligation on this
        attempt's committed geometry.

        The deck's net-blind ``space()`` makes any gap in the OPEN band
        (0, min_space) between SAME-net rects (routed metal x routed metal, or
        routed metal x own device artwork) a real M{n}.S.1 violation. Pricing
        alone cannot close these: the plateaued residual family (a via storey
        beside its own cap access tab; staggered same-column stubs at 25nm
        jitter) re-appears near-identically every rip-up round. So after a net
        routes, every remaining same-net band gap is MERGED: the gap box is
        fully metallised -- the deck's merged space() then has no facing edge
        pair left to flag (exactly the convergence detector's ``_gap_covered``
        semantics) -- with the fill extended one min-width INTO each rect so
        the union has no sub-min-width neck and no kissing corner.

        A fill is drawn ONLY when it is hard-legal against foreign artwork and
        min-space clear of foreign committed metal: an unpatchable gap stays an
        honest exact-detector residual for negotiation (never a silent pass,
        never a new short). Fully PDK-derived (per-layer min-width/min-space),
        deterministic scan order, no per-net/per-circuit branch, no wall-clock.

        GENERATION sweep (R7 junction round): every candidate pair of one
        generation is collected in ONE scan and processed in deterministic
        order (coverage re-judged against the fills already drawn this
        generation), then the grown geometry is re-scanned. The old
        first-hit-restart loop under a flat 32-pass cap silently TRUNCATED a
        large net's legalization -- the measured r7b M2.S.1: a 13-pin gnd
        burned the cap before ever reaching a gap whose fill was legal and
        waiting -- and a scaled per-gap budget made the restart scan
        quadratic. A bounded generation count keeps fill-on-fill cascades
        finite; whatever a cascade leaves open stays an honest exact-detector
        residual."""
        _RoutedNet = _eg_attr("_RoutedNet", _base_RoutedNet)
        active_metal = None if active_metal is None else set(active_metal)
        for _gen in range(4):
            self._watchdog_check(net, "same_net_legalize")
            pairs = self._same_net_gap_pairs(net, rn)
            if not pairs:
                return
            # A veto can depend on the current own-metal boundary.  Another
            # fill later in this generation may absorb that sliver, making the
            # same candidate legal on the next fixed-point generation.  Memoize
            # only within the immutable snapshot being scanned here.
            skipped: Set[Tuple[int, Rect, Rect]] = set()
            # Per-generation spatial index of the net's OWN metal -- the hole
            # source for coverage judgments. Rebuilding the net's whole rect
            # list per pair made one generation O(pairs x rects): the
            # measured 13-pin gnd carries ~62k banded pairs (jitter piles,
            # ~99.9% already covered) and burned ~284s/generation proving
            # coverage. Indexed judgment is O(pairs x local). Fills drawn
            # this generation are inserted incrementally so later pairs are
            # judged against them.
            metal, _cuts = self._net_metal(rn)
            own_idx: Dict[int, _SpatialIndex] = {}
            for gi2, rects2 in metal.items():
                idx2 = _SpatialIndex()
                for r2 in rects2:
                    idx2.add(net, _norm(r2))
                own_idx[gi2] = idx2
            progressed = False
            for pair_index, (gi, n1, n2, sx, sy) in enumerate(pairs):
                if pair_index % 1024 == 0:
                    self._watchdog_check(net, "same_net_legalize")
                if active_metal is not None and not (
                    (gi, n1) in active_metal or (gi, n2) in active_metal
                ):
                    continue
                if (gi, n1, n2) in skipped:
                    continue
                if self._gap_metallised(net, rn, gi, n1, n2, sx, sy,
                                        own_idx=own_idx):
                    continue
                committed_fill = False
                fill_candidates = (
                    self._merge_fill_rect_candidates(gi, n1, n2, sx, sy)
                    if allow_flush_ear_fills
                    else ((f,) if (f := self._merge_fill_rect(gi, n1, n2, sx, sy)) is not None else ())
                )
                for fill in fill_candidates:
                    new_segs = self._fill_segs(net, gi, fill)
                    actual_boxes = [
                        _seg_bbox(
                            segment.ax,
                            segment.ay,
                            segment.bx,
                            segment.by,
                            self._segment_half_width(net, segment),
                        )
                        for segment in new_segs
                    ]
                    if not self._fill_legal(
                        net,
                        gi,
                        actual_boxes,
                        own_idx=own_idx,
                        include_fixed_junction_support=allow_flush_ear_fills,
                    ):
                        continue
                    rn.segs.extend(new_segs)
                    for s2, actual_box in zip(new_segs, actual_boxes):
                        own_idx.setdefault(s2.gi, _SpatialIndex()).add(
                            net,
                            actual_box,
                        )
                        if active_metal is not None:
                            active_metal.add((s2.gi, _norm(actual_box)))
                    progressed = True
                    committed_fill = True
                    break
                if not committed_fill:
                    skipped.add((gi, n1, n2))
                    continue
            if not progressed:
                return

    def _same_net_gap_pairs(
        self, net: str, rn: _RoutedNet,
    ) -> List[Tuple[int, Rect, Rect, int, int]]:
        """ALL (deterministic order) same-net pairs of THIS attempt's metal
        against itself or against own artwork that violate touch-or-clear:
        either the edge-to-edge separation is in the open euclidean band
        (0, min_space), or the pair TOUCHES/OVERLAPS with a contact region
        whose diagonal is below one min-width (``ox^2 + oy^2 < w^2`` -- the
        JUNCTION-WIDTH obligation: the deck merges the pair into one polygon
        and a corner point-touch / sliver contact leaves a sub-min-width neck,
        the measured M4.W.1/M4.S.1 kiss family). Model-only keep-outs (bloated
        cap-plate collars) are never partners: they are not drawn. Coverage
        (``_gap_metallised``) is deliberately NOT judged here -- the caller
        re-judges each pair against the fills already drawn, so one collected
        generation stays valid while the geometry grows. Materialised list
        (snapshot): safe against ``rn.segs`` growing during processing."""
        _RoutedNet = _eg_attr("_RoutedNet", _base_RoutedNet)
        _sep = _eg_attr("_sep", _base_sep)
        out: List[Tuple[int, Rect, Rect, int, int]] = []
        self._watchdog_check(net, "same_net_gap_scan")
        metal, _cuts = self._net_metal(rn)
        for gi in sorted(metal):
            sp = self._space[gi]
            spsq = sp * sp
            w = 2 * self._half_w[gi]
            rects = [_norm(r) for r in metal[gi]]
            # Exact local candidate index.  A disjoint pair whose x- or
            # y-separation is at least ``sp`` cannot satisfy
            # ``sx**2 + sy**2 < sp**2``; only rectangles intersecting ``n1``
            # grown by ``sp`` can therefore be a spacing candidate.  Index by
            # sequence position (not geometry identity) so duplicate emitted
            # rectangles remain distinct claims and the historical i1<i2
            # ordering/output is preserved exactly.  The previous
            # ``rects[i1 + 1:]`` scan was O(n**2) on large multi-pin nets even
            # when every rectangle was remote -- the measured negotiation tail
            # expired inside same_net_legalize after routing 40/46 nets.
            rect_idx = _SpatialIndex()
            for rect_index, rect in enumerate(rects):
                rect_idx.add(str(rect_index), rect)
            oidx = self._obs_idx.get(gi)
            abs_gi = self._abstract_rects.get(gi, ())
            for i1, n1 in enumerate(rects):
                if i1 % 1024 == 0:
                    self._watchdog_check(net, "same_net_gap_scan")
                cands: List[Rect] = []
                if oidx is not None:
                    q = (n1[0] - sp, n1[1] - sp, n1[2] + sp, n1[3] + sp)
                    cands.extend(sorted(
                        _norm(r2) for (ow, r2) in oidx.query(q)
                        if ow == net and r2 not in abs_gi))
                q = (n1[0] - sp, n1[1] - sp,
                     n1[2] + sp, n1[3] + sp)
                local_later = []
                for owner, n2 in rect_idx.query(q):
                    assert owner is not None
                    i2 = int(owner)
                    if i2 > i1:
                        local_later.append((i2, _norm(n2)))
                cands.extend(
                    n2 for _i2, n2 in sorted(local_later)
                )
                for n2 in cands:
                    sx, sy = _sep(n1, n2)
                    if not (sx or sy):
                        ox = min(n1[2], n2[2]) - max(n1[0], n2[0])
                        oy = min(n1[3], n2[3]) - max(n1[1], n2[1])
                        if ox * ox + oy * oy >= w * w:
                            continue  # contact >= min-width: merged-clean
                    elif sx * sx + sy * sy >= spsq:
                        continue
                    out.append((gi, n1, n2, sx, sy))
        return out

    @staticmethod
    def _gap_box(n1: Rect, n2: Rect, sx: int, sy: int) -> Rect:
        """The open axis-aligned box between two banded rects (possibly zero-
        extent on one axis for corner-to-edge pairs)."""
        ax0, ay0, ax1, ay1 = n1
        bx0, by0, bx1, by1 = n2
        gx0, gx1 = ((max(ax0, bx0), min(ax1, bx1)) if sx == 0
                    else (min(ax1, bx1), max(ax0, bx0)))
        gy0, gy1 = ((max(ay0, by0), min(ay1, by1)) if sy == 0
                    else (min(ay1, by1), max(ay0, by0)))
        return (gx0, gy0, gx1, gy1)

    def _gap_metallised(self, net: str, rn: _RoutedNet, gi: int,
                        n1: Rect, n2: Rect, sx: int, sy: int,
                        own_idx: Optional[Dict[int, _SpatialIndex]] = None,
                        ) -> bool:
        """True when the band gap is already FULLY covered by same-layer metal
        (this net's own metal, device artwork of any owner, or committed
        foreign metal) -- the deck's net-blind merge leaves nothing to flag.
        Projection-thin gap boxes are judged over the box grown to the layer
        min-width on the narrow axis, same as the convergence detector's
        ``_gap_covered`` (a zero-area box can never be covered raw; the
        min-width swath is what the euclidean space oracle certifies). A
        JUNCTION pair (sx == sy == 0: sub-min-width contact) is judged over
        the SAME min-width swath centred on its contact box -- judging it
        over the full corner-fill footprint over-demanded cover the deck
        does not (measured: dense multi-rect merge zones read deck-clean
        while the 2w footprint stuck out past the outermost rect edge).
        The corner fill itself stays 2w (``_merge_fill_rect``: it must be
        junction-clean with BOTH rects) and covers the swath, so fill and
        judgment still converge. Model-only keep-outs (``_abstract_rects``)
        never count as cover: metal that exists only in the routing model
        merges nothing in the emitted artifact.

        ``own_idx`` (optional) is a caller-maintained spatial index of the
        net's own metal: the legalization sweep passes it so one generation's
        ~10^4-pair coverage judgment stays O(local) per pair instead of
        rebuilding the whole net rect list per call (the measured 284s/gen
        hang on a 13-pin gnd)."""
        _RoutedNet = _eg_attr("_RoutedNet", _base_RoutedNet)
        gx0, gy0, gx1, gy1 = self._gap_box(n1, n2, sx, sy)
        w = 2 * self._half_w[gi]
        if gx1 - gx0 < w:
            cx = (gx0 + gx1) // 2
            gx0, gx1 = cx - w // 2, cx - w // 2 + w
        if gy1 - gy0 < w:
            cy = (gy0 + gy1) // 2
            gy0, gy1 = cy - w // 2, cy - w // 2 + w
        gap = (gx0, gy0, gx1, gy1)
        abs_gi = self._abstract_rects.get(gi, ())
        holes: List[Rect] = []
        if own_idx is not None:
            oi = own_idx.get(gi)
            if oi is not None:
                for (_ow, r) in oi.query(gap):
                    rr = _norm(r)
                    if rr[0] < gx1 and gx0 < rr[2] and rr[1] < gy1 and gy0 < rr[3]:
                        holes.append(rr)
        else:
            metal, _cuts = self._net_metal(rn)
            for r in metal.get(gi, ()):
                rr = _norm(r)
                if rr[0] < gx1 and gx0 < rr[2] and rr[1] < gy1 and gy0 < rr[3]:
                    holes.append(rr)
        oidx = self._obs_idx.get(gi)
        if oidx is not None:
            for (_ow, r) in oidx.query(gap):
                if r in abs_gi:
                    continue
                rr = _norm(r)
                if rr[0] < gx1 and gx0 < rr[2] and rr[1] < gy1 and gy0 < rr[3]:
                    holes.append(rr)
        midx = self._metal_idx.get(gi)
        if midx is not None:
            for (_ow, r) in midx.query(gap):
                rr = _norm(r)
                if rr[0] < gx1 and gx0 < rr[2] and rr[1] < gy1 and gy0 < rr[3]:
                    holes.append(rr)
        # exact fast path: ONE hole containing the whole swath covers it --
        # the dominant case in dense same-net piles (a sibling rect spans the
        # gap); skips the expensive rectangle subtraction (~ms) per pair.
        for rr in holes:
            if (rr[0] <= gx0 and rr[1] <= gy0
                    and rr[2] >= gx1 and rr[3] >= gy1):
                return True
        return not subtract_rects(gap, holes)

    def _merge_fill_rect(self, gi: int, n1: Rect, n2: Rect,
                         sx: int, sy: int) -> Optional[Rect]:
        """The fill rect that merges a banded same-net pair legally: covers the
        WHOLE gap box (no facing edges left across it) and penetrates each rect
        by one min-width perpendicular to the gap (no sub-min-width neck, no
        kissing corner). A pair whose gap box cannot host a min-width-wide fill
        crossing (thin projection overlap / pure corner pair) gets the grown
        corner fill instead -- still full-cover + min-width penetration."""
        w = 2 * self._half_w[gi]
        gx0, gy0, gx1, gy1 = self._gap_box(n1, n2, sx, sy)
        if sx == 0 and sy > 0 and (gx1 - gx0) >= w:
            return (gx0, gy0 - w, gx1, gy1 + w)
        if sy == 0 and sx > 0 and (gy1 - gy0) >= w:
            return (gx0 - w, gy0, gx1 + w, gy1)
        # diagonal corner pair or thin projection overlap: grown corner fill.
        return (gx0 - w, gy0 - w, gx1 + w, gy1 + w)

    def _merge_fill_rect_candidates(
        self, gi: int, n1: Rect, n2: Rect, sx: int, sy: int,
    ) -> Tuple[Rect, ...]:
        primary = self._merge_fill_rect(gi, n1, n2, sx, sy)
        if primary is None:
            return ()
        w = 2 * self._half_w[gi]
        gx0, gy0, gx1, gy1 = self._gap_box(n1, n2, sx, sy)
        candidates = [primary]
        if sx == 0 and sy > 0 and 0 < (gx1 - gx0) < w:
            candidates.append((gx1 - w, gy0 - w, gx1, gy1 + w))
            candidates.append((gx0, gy0 - w, gx0 + w, gy1 + w))
        elif sy == 0 and sx > 0 and 0 < (gy1 - gy0) < w:
            candidates.append((gx0 - w, gy1 - w, gx1 + w, gy1))
            candidates.append((gx0 - w, gy0, gx1 + w, gy0 + w))
        return tuple(candidates)

    @staticmethod
    def _proposed_fill_rectangle(fills: Sequence[Rect]) -> Optional[Rect]:
        """Prove an exact rectangular union of one emitted strip batch.

        ``_fill_segs`` emits strips sharing an entire x- or y-span. Their
        remaining intervals must cover without a gap; their enclosing box alone
        is never evidence. The iterative proof also handles long strip batches
        without recursive rectangle subtraction. Mixed-span tilings remain
        unsupported here, even if their union might form a rectangle.
        """
        rectangles = tuple(_norm(rect) for rect in fills)
        if not rectangles or any(
            rect[0] >= rect[2] or rect[1] >= rect[3] for rect in rectangles
        ):
            return None
        first = rectangles[0]
        for shared in (0, 1):
            if not all(rect[shared] == first[shared]
                       and rect[shared + 2] == first[shared + 2]
                       for rect in rectangles):
                continue
            varying = 1 - shared
            intervals = sorted((rect[varying], rect[varying + 2])
                               for rect in rectangles)
            start, end = intervals[0]
            for low, high in intervals[1:]:
                if low > end:
                    return None
                end = max(end, high)
            rectangle = list(first)
            rectangle[varying], rectangle[varying + 2] = start, end
            return tuple(rectangle)
        return None

    @staticmethod
    def _atomic_fill_covers_junction(
        fill: Rect, other: Rect, fills: Sequence[Rect], minimum_width: int,
        *, support_rects: Sequence[Rect] = (),
    ) -> bool:
        """Prove that exact same-layer metal supports a small contact.

        A pairwise sliver is not a new neck if its entire intersection lies
        inside another proposed fill with a full-width overlap to BOTH shapes.
        Requiring a rectangular overlap dimension >= the PDK width is stronger
        than the usual diagonal junction test. A witness may be a proposed fill
        tile or an already materialized same-net rectangle from the caller's
        exact local metal index. No un-emitted metal or inferred component bbox
        is admissible. Existing support is essential when a new tile touches a
        segmented wire at a short edge already backed by its perpendicular leg.
        A full-rule-width shared edge is a valid connection even with zero
        overlap area; a shorter edge, corner kiss, or positive gap is not.
        """
        contact = (
            max(fill[0], other[0]), max(fill[1], other[1]),
            min(fill[2], other[2]), min(fill[3], other[3]),
        )
        for raw_bridge in (*fills, *support_rects):
            bridge = _norm(raw_bridge)
            if bridge == fill or not (
                bridge[0] <= contact[0] <= contact[2] <= bridge[2]
                and bridge[1] <= contact[1] <= contact[3] <= bridge[3]
            ):
                continue
            supported = True
            for shape in (fill, other):
                ox = min(bridge[2], shape[2]) - max(bridge[0], shape[0])
                oy = min(bridge[3], shape[3]) - max(bridge[1], shape[1])
                if min(ox, oy) < 0 or max(ox, oy) < minimum_width:
                    supported = False
                    break
            if supported:
                return True
        return False

    def _fill_legal(
        self,
        net: str,
        gi: int,
        fills: Sequence[Rect],
        own_idx: Optional[Dict[int, _SpatialIndex]] = None,
        *,
        check_own_junctions: bool = True,
        current_route: Optional[_RoutedNet] = None,
        include_fixed_junction_support: bool = False,
    ) -> bool:
        """Fill segment footprints may only be drawn when they create no defect.

        ``fills`` are the actual emitted ``_seg_bbox`` rectangles at
        ``_hw(net, gi)``, never the nominal PDK-minimum merge box.  A wide net
        must therefore pass foreign and own geometry checks at the exact width
        that legality, congestion, occupancy, and emit consume.

          * clear of foreign device artwork AND of committed foreign metal by
            the deck's OWN metric -- the EUCLIDEAN min-space band, exactly what
            the residual detector's ``_band`` judges (a fill that merely priced
            foreign metal would trade a same-net residual for a cross-net one
            -- or, at sep (0,0), a short). Chebyshev (either-axis) clearance
            here was STRICTER than the deck and vetoed fills the deck accepts
            (a diagonal foreign corner at euclid >= min_space), leaving
            closable same-net residuals open;
          * JUNCTION-CLEAN against the net's OWN metal (``own_idx``) and own
            artwork: a fill edge landing a SLIVER contact over a third own
            rect (contact diagonal < min-width) trades the closed gap for a
            NEW merged-width neck -- the measured cascade M2.W.1 (a gen-3 pile
            fill slivered 118nm over a neighbour and the sweep ended). Such a
            fill is vetoed; the gap stays an honest residual."""
        _RoutedNet = _eg_attr("_RoutedNet", _base_RoutedNet)
        _sep = _eg_attr("_sep", _base_sep)
        sp = self._space[gi]
        spsq = sp * sp
        w = 2 * self._half_w[gi]
        wsq = w * w
        abs_gi = self._abstract_rects.get(gi, ())
        oidx = self._obs_idx.get(gi)
        midx = self._metal_idx.get(gi)
        oi = own_idx.get(gi) if own_idx is not None else None
        proposed_rectangle_checked = False
        proposed_rectangle: Optional[Rect] = None

        def junction_supported(fill: Rect, other: Rect,
                               support: Sequence[Rect]) -> bool:
            nonlocal proposed_rectangle_checked, proposed_rectangle
            if self._atomic_fill_covers_junction(
                fill, other, fills, w, support_rects=support,
            ):
                return True
            # Only this atomic, same-net/same-layer proposed batch may supply
            # collective support. Existing metal is still judged one exact
            # witness at a time. Prove the batch once, only when needed, and
            # keep every later tile's foreign/envelope/MIM/intent checks.
            if not proposed_rectangle_checked:
                proposed_rectangle = self._proposed_fill_rectangle(fills)
                proposed_rectangle_checked = True
            return proposed_rectangle is not None and self._atomic_fill_covers_junction(
                fill, other, (), w, support_rects=(proposed_rectangle,),
            )

        needs_ordinary_union_proof = False
        for fill in fills:
            nf = _norm(fill)
            if getattr(self, "_active_group_route_local_closure", False):
                if include_fixed_junction_support:
                    b = getattr(self, "_bounds", None)
                    if (
                        isinstance(b, tuple)
                        and len(b) == 4
                        and not (b[0] <= nf[0] and nf[2] <= b[2] and b[1] <= nf[1] and nf[3] <= b[3])
                    ):
                        return False
                if self._reject_group_route_local_fill(gi, nf, net):
                    return False
            elif self._reject_group_route_global_box(gi, nf, net):
                return False
            envelope = getattr(self, "_routing_envelope", None)
            if envelope is not None and not envelope.contains_bbox(nf):
                self._record_hard_rejection(
                    "routing_envelope",
                    net,
                    gi=gi,
                    candidate_rect=nf,
                    candidate_half=self._hw(net, gi),
                    blocker_rect=envelope.bbox_nm,
                    blocker_kind="floorplan_boundary",
                )
                return False
            if self._reject_mim_bottom_routing_clearance(
                gi,
                nf,
                net,
                candidate_half=self._hw(net, gi),
            ):
                return False
            if self._reject_mim_derived_unrelated_clearance(
                gi,
                nf,
                net,
                candidate_half=self._hw(net, gi),
            ):
                return False
            if self._reject_intent_spacing(gi, nf, net):
                return False
            for owner, blocker, required, _kind, part in self._wide_spacing_neighbors(
                gi, nf, include_committed=True,
            ):
                if owner != net:
                    sx, sy = _sep(part, blocker)
                    if sx * sx + sy * sy < required * required:
                        return False
            q = (nf[0] - sp, nf[1] - sp, nf[2] + sp, nf[3] + sp)
            junction_support = (
                *(
                    (_norm(rect) for owner, rect in oi.query(q) if owner == net)
                    if oi is not None and check_own_junctions
                    else ()
                ),
                *(
                    (_norm(rect) for owner, rect in oidx.query(q)
                     if owner == net and rect not in abs_gi)
                    if include_fixed_junction_support and oidx is not None and check_own_junctions
                    else ()
                ),
            )
            if oidx is not None:
                for (owner, r) in oidx.query(q):
                    if owner == net:
                        if r in abs_gi:
                            continue
                        if not check_own_junctions:
                            continue
                        sx, sy = _sep(nf, _norm(r))
                        if not (sx or sy):
                            rr = _norm(r)
                            ox = min(nf[2], rr[2]) - max(nf[0], rr[0])
                            oy = min(nf[3], rr[3]) - max(nf[1], rr[1])
                            if ox * ox + oy * oy < wsq and not junction_supported(
                                nf, rr, junction_support
                            ):
                                if not include_fixed_junction_support:
                                    return False  # would neck against own artwork
                                needs_ordinary_union_proof = True
                        continue
                    sx, sy = _sep(nf, _norm(r))
                    if sx * sx + sy * sy < spsq:
                        return False
            if midx is not None:
                for (owner, r) in midx.query(q):
                    if owner == net:
                        continue
                    sx, sy = _sep(nf, _norm(r))
                    if sx * sx + sy * sy < spsq:
                        return False
            if oi is not None and check_own_junctions:
                for (_ow, r) in oi.query(q):
                    rr = _norm(r)
                    sx, sy = _sep(nf, rr)
                    if not (sx or sy):
                        ox = min(nf[2], rr[2]) - max(nf[0], rr[0])
                        oy = min(nf[3], rr[3]) - max(nf[1], rr[1])
                        if ox * ox + oy * oy < wsq and not junction_supported(
                            nf, rr, junction_support
                        ):
                            if not include_fixed_junction_support:
                                return False  # would neck against own metal
                            needs_ordinary_union_proof = True
        active = getattr(self, "_active_wide_repair_route", None)
        if active is not None and active.name == net:
            # The complete live route is checked after all phases. A temporary
            # own-wide gap may be repaired by a later phase in this transaction.
            return True
        if current_route is not None:
            current = self._net_metal(current_route)[0].get(gi, ())
        elif oi is not None:
            current = tuple({tuple(rectangle) for bucket in oi.buckets.values()
                             for owner, rectangle in bucket if owner == net})
        else:
            current = tuple(rectangle for owner, rectangle in
                            getattr(getattr(self, "_book", None), "metal", {}).get(gi, ())
                            if owner == net)
        if needs_ordinary_union_proof and not self._ordinary_repair_preserves_world(
            net, {gi: current}, {gi: (*current, *fills)},
        ):
            return False
        return self._repair_preserves_world(
            net, {gi: current}, {gi: (*current, *fills)},
        )

    def _fill_segs(self, net: str, gi: int, fill: Rect) -> List[_WireSeg]:
        """Freeze proven local fill widths without changing old wire geometry."""
        with self._half_width_scope(
            getattr(self, "_active_fill_half_width_overrides", {}),
        ):
            return self._fill_segs_impl(net, gi, fill)

    def _fill_segs_impl(self, net: str, gi: int, fill: Rect) -> List[_WireSeg]:
        """Tile ``fill`` with the net's actual emitted-width wire segments.

        The merge obligation itself remains the PDK minimum-width swath built by
        ``_merge_fill_rect``.  Segment centres and end points, however, consume
        ``_hw(net, gi)`` so subsequent emit does not expand a nominal min-width
        strip beyond the geometry that was checked.  When the structural swath
        is narrower than the intent width, one centred segment is emitted and
        its exact expanded bbox is judged by ``_fill_legal``.
        """
        x0, y0, x1, y1 = fill
        hw = self._hw(net, gi)
        w = 2 * hw
        grid = max(1, int(getattr(self, "_manufacturing_grid_nm", 1)))

        # Segment centres live on the lattice that makes both emitted edges
        # manufacturing-grid aligned.  This is normally the grid origin, but
        # expressing the offset explicitly also covers technologies whose
        # legal half-width is a non-zero residue of the manufacturing grid.
        centre_offset = hw % grid

        def centre_floor(value: int) -> int:
            return ((int(value) - centre_offset) // grid) * grid + centre_offset

        def centre_ceil(value: int) -> int:
            lower = centre_floor(value)
            return lower if lower == int(value) else lower + grid

        def grid_cover(values: Sequence[int]) -> List[int]:
            projected: Set[int] = set()
            for value in values:
                projected.add(centre_floor(value))
                projected.add(centre_ceil(value))
            return sorted(projected)

        def centres(lo: int, hi: int) -> List[int]:
            if hi - lo <= w:
                # A grid-aligned narrow swath can have an off-grid midpoint
                # (for example a 3995 nm GF180 repair on a 5 nm grid).  Keep
                # both neighbouring legal centres so their min-width boxes
                # cover the original swath without any emitter-time snap.
                return grid_cover(((lo + hi) // 2,))
            values = [lo + hw]
            while values[-1] + hw < hi:
                next_value = min(values[-1] + w, hi - hw)
                if next_value == values[-1]:
                    break
                values.append(next_value)
            return grid_cover(values)

        def span(lo: int, hi: int) -> Tuple[int, int]:
            if hi - lo <= w:
                centre = (lo + hi) // 2
                return centre_floor(centre), centre_ceil(centre)
            # Snap outwards in segment-centre space.  The emitted bbox then
            # contains the requested repair interval and stays on-grid.
            return centre_floor(lo + hw), centre_ceil(hi - hw)

        segs: List[_WireSeg] = []
        if (x1 - x0) >= (y1 - y0):
            ax, bx = span(x0, x1)
            for cy in centres(y0, y1):
                segs.append(self._wire_segment(net, gi, ax, cy, bx, cy))
        else:
            ay, by = span(y0, y1)
            for cx in centres(x0, x1):
                segs.append(self._wire_segment(net, gi, cx, ay, cx, by))
        return segs

    # ------------------------------------------------------------- emit
    @staticmethod
    def _candidate_access_deck_witness_required(
        access_states: Sequence[Tuple[int, int, int]],
    ) -> bool:
        """Whether this edge must emit geometry between distinct access states."""

        states = tuple(tuple(int(value) for value in row) for row in access_states)
        return len(states) == 2 and states[0] != states[1]

    def _candidate_access_deck_connected(
        self,
        net: str,
        segs: Sequence[_WireSeg],
        vias: Sequence[_ViaSite],
        access_states: Sequence[Tuple[int, int, int]],
    ) -> bool:
        """Whether one emitted edge joins both selected states in the PDK deck."""
        return self._candidate_deck_connects_probes(
            net,
            segs,
            vias,
            access_states,
            include_fixed=False,
        )

    def _candidate_terminal_deck_connected(
        self,
        net: str,
        segs: Sequence[_WireSeg],
        vias: Sequence[_ViaSite],
        access_states: Sequence[Tuple[int, int, int]],
        *,
        materialized_access_states: Sequence[Tuple[int, int, int]] = (),
    ) -> bool:
        """Whether one emitted edge joins both canonical terminal components."""
        probes = self._candidate_terminal_probe_states(
            net,
            access_states,
            materialized_access_states=materialized_access_states,
        )
        if probes is None:
            # Direct helper/unit callers without a canonical route context retain
            # the access-state physical check above.  Production signal edges
            # always publish both immutable endpoint identities.
            self._last_candidate_terminal_probes = None
            return True
        if len(probes) != 2:
            self._last_candidate_terminal_probes = None
            return False
        connected = self._candidate_deck_connects_probes(
            net,
            segs,
            vias,
            probes,
            include_fixed=True,
        )
        self._last_candidate_terminal_probes = probes if connected else None
        return connected

    def _candidate_terminal_deck_connected_after_group_resolution(
        self,
        net: str,
        prior_segs: Sequence[_WireSeg],
        current_segs: Sequence[_WireSeg],
        resolved_vias: Sequence[_ViaSite],
        terminal_probes: Sequence[Tuple[int, int, int]],
    ) -> bool:
        """Validate terminals in the complete resolved local route world."""

        return self._candidate_deck_connects_probes(
            net,
            tuple(prior_segs) + tuple(current_segs),
            tuple(resolved_vias),
            terminal_probes,
            include_fixed=True,
        )

    def _candidate_terminal_probe_states(
        self,
        net: str,
        access_states: Sequence[Tuple[int, int, int]],
        *,
        materialized_access_states: Sequence[Tuple[int, int, int]] = (),
    ) -> Optional[Tuple[Tuple[int, int, int], ...]]:
        """Canonical probes, preserving explicitly materialized MST junctions."""
        states = tuple(
            tuple(int(value) for value in state) for state in access_states
        )
        if len(states) != 2:
            return ()
        materialized = {
            tuple(int(value) for value in state)
            for state in materialized_access_states
        }
        details = dict(getattr(self, "_route_failure_details", ()) or ())
        endpoint_details = []
        for role in ("source", "goal"):
            raw = details.get(role)
            try:
                endpoint = dict(raw)
                xy = tuple(int(value) for value in endpoint["xy_nm"])
                gi = int(endpoint["gi"])
            except (KeyError, TypeError, ValueError):
                return None
            if len(xy) != 2:
                return None
            endpoint_details.append((xy[0], xy[1], gi))
        if len(endpoint_details) != 2:
            return ()
        probes = []
        for endpoint, fallback_state in zip(endpoint_details, states):
            if fallback_state in materialized:
                # This state is not an inferred substitute for the terminal.  A
                # prior MST edge already emitted and deck-validated the junction;
                # the current edge must attach to that exact physical state.
                probes.append(fallback_state)
                continue
            component = tuple(self._owned_terminal_component(net, *endpoint))
            if not component:
                probes.append(fallback_state)
                continue
            # The original pin seed is the canonical terminal obligation.  A
            # component member selected by lexical/geometry order may be joined
            # only in the router's closed-touch abstraction; substituting that
            # remote member would reject an edge that is physically attached at
            # the actual pin.  The seed is already mechanically authorized by
            # ``_owned_terminal_component`` above.
            probes.append(endpoint)
        return tuple(probes)

    def _fixed_deck_components_cache(self, stack, known_layers):
        """Pre-extract exact fixed-only connected components once per fixed world.

        By graph monotonicity, two probes are connected in ``fixed U candidate``
        iff they are connected in ``candidate`` plus ONLY the connected components
        of ``fixed`` that touch ``candidate`` or a probe (together with any MIM
        carve markers overlapping included bottom-metal shapes).
        """
        import klayout.db as kdb

        fixed_rects = self._fixed_deck_literal_rects
        dbu_um = float(self._tech.pdk.rules.dbu_nm) / 1000.0
        cache_key = (id(fixed_rects), len(fixed_rects), stack, dbu_um)
        cached = getattr(self, "_fixed_deck_comp_cache", None)
        if cached is not None and cached[0] == cache_key:
            return cached[1]

        layout = kdb.Layout()
        layout.dbu = dbu_um
        cell = layout.create_cell("FIXED_COMPONENTS")
        layer_handles: Dict[Tuple[int, int], int] = {}
        carve_marker_layers = {carve.top_marker for carve in stack.mim_carves}
        carve_markers_by_layer: Dict[Tuple[int, int], List[Rect]] = {}

        for raw_layer, _owner, raw_rect in fixed_rects:
            layer = tuple(raw_layer)
            if layer not in known_layers:
                continue
            box = _norm(raw_rect)
            li = layer_handles.get(layer)
            if li is None:
                li = layout.layer(*layer)
                layer_handles[layer] = li
            cell.shapes(li).insert(kdb.Box(*box))
            if layer in carve_marker_layers:
                carve_markers_by_layer.setdefault(layer, []).append(box)

        l2n = kdb.LayoutToNetlist(kdb.RecursiveShapeIterator(layout, cell, []))
        l2n.threads = 1
        handles: Dict[Tuple[int, int], Any] = {}
        for position, layer in enumerate(sorted(layer_handles)):
            handles[layer] = l2n.make_polygon_layer(
                layer_handles[layer], f"FC_{position}_{layer[0]}_{layer[1]}"
            )
        for layer in stack.conductors:
            h = handles.get(layer)
            if h is not None:
                l2n.connect(h)
        carves_by_conn = {
            (c.bottom, c.riser): c for c in stack.mim_carves
        }
        for conn in stack.connections:
            cut = handles.get(conn.cut)
            lower = handles.get(conn.lower)
            upper = handles.get(conn.upper)
            if cut is None:
                continue
            carve = carves_by_conn.get((conn.lower, conn.cut))
            marker = handles.get(carve.top_marker) if carve is not None else None
            if lower is not None and marker is not None and carve is not None:
                interacting = lower.interacting(marker)
                if carve.marker_only:
                    mim_virtual = marker & interacting
                elif carve.oversize_um > 0:
                    oversize_dbu = int(round(carve.oversize_um / dbu_um))
                    mim_virtual = marker.sized(oversize_dbu) & interacting
                else:
                    mim_virtual = interacting
                metal_ncap = lower - mim_virtual
                suffix = f"{conn.lower[0]}_{conn.lower[1]}"
                l2n.register(mim_virtual, f"FC_MIM_VIRTUAL_{suffix}")
                l2n.register(metal_ncap, f"FC_METAL_NCAP_{suffix}")
                l2n.connect(lower, mim_virtual)
                l2n.connect(metal_ncap, mim_virtual)
                l2n.connect(metal_ncap, lower)
                l2n.connect(metal_ncap, cut)
                l2n.connect(marker, cut)
            elif lower is not None:
                l2n.connect(lower, cut)
            if upper is not None:
                l2n.connect(cut, upper)
        l2n.extract_netlist()

        # Map each fixed shape (on conductor, cut, or MIM marker layers) to its
        # connected net cluster id.
        comp_shapes: Dict[int, List[Tuple[Tuple[int, int], Any]]] = {}
        comp_layer_idx: Dict[Tuple[int, int], _SpatialIndex] = {}
        next_cid = 0
        netlist = l2n.netlist()
        top_circuit = netlist.circuit_by_name("FIXED_COMPONENTS") if netlist is not None else None
        if top_circuit is not None:
            for net_obj in top_circuit.each_net():
                cid = next_cid
                next_cid += 1
                shapes_list: List[Tuple[Tuple[int, int], Any]] = []
                for layer, handle in handles.items():
                    reg = l2n.shapes_of_net(net_obj, handle, False)
                    if reg is None or reg.is_empty():
                        continue
                    lidx = comp_layer_idx.get(layer)
                    if lidx is None:
                        lidx = _SpatialIndex()
                        comp_layer_idx[layer] = lidx
                    for poly in reg.each():
                        bb = poly.bbox()
                        b = (int(bb.left), int(bb.bottom), int(bb.right), int(bb.top))
                        if poly.is_box():
                            shapes_list.append((layer, b))
                        else:
                            shapes_list.append((layer, poly))
                        lidx.add(cid, b)
                if shapes_list:
                    comp_shapes[cid] = shapes_list

        # Also include any unextracted cut shapes (isolated fixed cuts not touching
        # any fixed metal) so a candidate wire landing on an isolated cut is seen.
        isolated_cuts: Dict[Tuple[int, int], _SpatialIndex] = {}
        cut_layers = {c.cut for c in stack.connections}
        for raw_layer, _owner, raw_rect in fixed_rects:
            layer = tuple(raw_layer)
            if layer in cut_layers:
                lidx = isolated_cuts.get(layer)
                if lidx is None:
                    lidx = _SpatialIndex()
                    isolated_cuts[layer] = lidx
                isolated_cuts[layer].add(0, _norm(raw_rect))

        # Carve marker spatial index + oversize_dbu per carve bottom layer
        carve_info: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], int, List[Rect]]]] = {}
        for carve in stack.mim_carves:
            mboxes = carve_markers_by_layer.get(carve.top_marker)
            if mboxes:
                ov = int(round(carve.oversize_um / dbu_um)) if carve.oversize_um > 0 else 0
                carve_info.setdefault(carve.bottom, []).append((carve.top_marker, ov, mboxes))

        payload = (comp_shapes, comp_layer_idx, isolated_cuts, carve_info)
        self._fixed_deck_comp_cache = (cache_key, payload)
        return payload

    def _candidate_deck_connects_probes(
        self,
        net: str,
        segs: Sequence[_WireSeg],
        vias: Sequence[_ViaSite],
        probe_states: Sequence[Tuple[int, int, int]],
        *,
        include_fixed: bool,
        additional_fixed_metal: Sequence[Tuple[int, Rect]] = (),
    ) -> bool:
        """Run the exact PDK connect stack over one edge and two probe points."""
        _RoutedNet = _eg_attr("_RoutedNet", _base_RoutedNet)
        import klayout.db as kdb

        from clay.layout.route.deck_isolation import (
            DeckConnectStack,
            NetProbe,
            check_deck_net_isolation,
        )

        states = tuple(
            tuple(int(value) for value in state) for state in probe_states
        )
        if len(states) != 2:
            return False
        stack = getattr(self, "_deck_connect_stack", None)
        if stack is None:
            stack = DeckConnectStack.from_tech(self._tech.pdk)
        known_layers = set(stack.conductors) | {
            connection.cut for connection in stack.connections
        } | {carve.top_marker for carve in stack.mim_carves}
        dbu_um = float(self._tech.pdk.rules.dbu_nm) / 1000.0

        emitted = self._emit_net(_RoutedNet(
            name=str(net),
            segs=list(segs),
            vias=list(vias),
        ))
        cand_by_layer: Dict[Tuple[int, int], List[Any]] = {}
        cand_boxes_by_layer: Dict[Tuple[int, int], List[Rect]] = {}
        for gi, rect in additional_fixed_metal:
            layer = tuple(self._tech.gi_to_metal_key(gi))
            if layer not in known_layers:
                raise ValueError(f"declared rail lies outside the deck stack: {layer}")
            b = _norm(rect)
            cand_by_layer.setdefault(layer, []).append(kdb.Box(*b))
            cand_boxes_by_layer.setdefault(layer, []).append(b)
        for polygon in emitted.wires:
            layer = (int(polygon.layer), int(polygon.datatype))
            if layer not in known_layers:
                raise ValueError(
                    "candidate route rectangle lies outside the complete "
                    f"deck stack: {layer}"
                )
            verts = polygon.polygon.vertices
            if len(verts) == 4:
                xs_v = [int(p.x) for p in verts]
                ys_v = [int(p.y) for p in verts]
                b = (min(xs_v), min(ys_v), max(xs_v), max(ys_v))
                if (
                    all(int(p.x) in (b[0], b[2]) and int(p.y) in (b[1], b[3]) for p in verts)
                ):
                    cand_by_layer.setdefault(layer, []).append(kdb.Box(*b))
                    cand_boxes_by_layer.setdefault(layer, []).append(b)
                    continue
            pts = [kdb.Point(int(point.x), int(point.y)) for point in verts]
            poly = kdb.Polygon(pts)
            bb = poly.bbox()
            cand_by_layer.setdefault(layer, []).append(poly)
            cand_boxes_by_layer.setdefault(layer, []).append(
                (int(bb.left), int(bb.bottom), int(bb.right), int(bb.top))
            )

        probes = tuple(
            NetProbe(
                "CANDIDATE_ACCESS",
                tuple(self._tech.gi_to_metal_key(gi)),
                x,
                y,
            )
            for x, y, gi in states
        )

        # Build localized layout: when include_fixed is True and fixed rects are
        # large, include ONLY fixed components touching candidate shapes/probes
        # plus any overlapping MIM carve markers (100% graph-exact).
        layout = kdb.Layout()
        layout.dbu = dbu_um
        cell = layout.create_cell("GRIDLESS_CANDIDATE_ACCESS")
        li_cache: Dict[Tuple[int, int], int] = {}

        def _li(layer: Tuple[int, int]) -> int:
            idx = li_cache.get(layer)
            if idx is None:
                idx = layout.layer(*layer)
                li_cache[layer] = idx
            return idx

        if include_fixed:
            fixed_rects = self._fixed_deck_literal_rects
            if len(fixed_rects) <= 64:
                for raw_layer, _owner, raw_rect in fixed_rects:
                    layer = tuple(raw_layer)
                    if layer not in known_layers:
                        continue
                    cell.shapes(_li(layer)).insert(kdb.Box(*_norm(raw_rect)))
            else:
                comp_shapes, comp_layer_idx, isolated_cuts, carve_info = (
                    self._fixed_deck_components_cache(stack, known_layers)
                )
                # Build adjacency map of layers connected via cuts or MIM top markers
                adj_layers: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
                for conn in stack.connections:
                    adj_layers.setdefault(conn.lower, []).append(conn.cut)
                    adj_layers.setdefault(conn.upper, []).append(conn.cut)
                    adj_layers.setdefault(conn.cut, []).extend((conn.lower, conn.upper))
                for carve in stack.mim_carves:
                    adj_layers.setdefault(carve.riser, []).append(carve.top_marker)
                    adj_layers.setdefault(carve.top_marker, []).append(carve.riser)

                query_boxes: Dict[Tuple[int, int], List[Rect]] = {
                    layer: list(boxes) for layer, boxes in cand_boxes_by_layer.items()
                }
                for p in probes:
                    query_boxes.setdefault(p.layer, []).append((p.x, p.y, p.x, p.y))

                selected_cids: Set[int] = set()
                selected_cuts: Dict[Tuple[int, int], Set[Rect]] = {}
                for layer, boxes in query_boxes.items():
                    check_layers = [layer, *adj_layers.get(layer, ())]
                    for cl in check_layers:
                        cidx = comp_layer_idx.get(cl)
                        if cidx is not None:
                            for bx in boxes:
                                for cid, r in cidx.query(bx):
                                    if _closed_rects_touch(bx, r):
                                        selected_cids.add(cid)
                        icidx = isolated_cuts.get(cl)
                        if icidx is not None:
                            for bx in boxes:
                                for _o, r in icidx.query(bx):
                                    if _closed_rects_touch(bx, r):
                                        selected_cuts.setdefault(cl, set()).add(r)

                # If an isolated cut was touched, also pull any fixed component touching that cut
                for cl, cut_set in selected_cuts.items():
                    for target_l in adj_layers.get(cl, ()):
                        cidx = comp_layer_idx.get(target_l)
                        if cidx is None:
                            continue
                        for cb in cut_set:
                            for cid, r in cidx.query(cb):
                                if _closed_rects_touch(cb, r):
                                    selected_cids.add(cid)

                included_bottom_boxes: Dict[Tuple[int, int], List[Rect]] = {}
                for bl in carve_info:
                    if bl in cand_boxes_by_layer:
                        included_bottom_boxes.setdefault(bl, []).extend(cand_boxes_by_layer[bl])
                for cid in selected_cids:
                    for layer, shp in comp_shapes.get(cid, ()):
                        if isinstance(shp, tuple):
                            if layer in carve_info:
                                included_bottom_boxes.setdefault(layer, []).append(shp)
                        else:
                            if layer in carve_info:
                                bb = shp.bbox()
                                included_bottom_boxes.setdefault(layer, []).append(
                                    (int(bb.left), int(bb.bottom), int(bb.right), int(bb.top))
                                )

                # Pull any fixed component containing a MIM carve top_marker overlapping
                # an included bottom-metal shape (preserves exact carving AND top-plate connectivity).
                added_markers: Dict[Tuple[int, int], Set[Rect]] = {}
                if carve_info and included_bottom_boxes:
                    for bl, b_list in included_bottom_boxes.items():
                        for m_layer, ov, mboxes in carve_info.get(bl, ()):
                            mcidx = comp_layer_idx.get(m_layer)
                            for mb in mboxes:
                                grown = (mb[0] - ov, mb[1] - ov, mb[2] + ov, mb[3] + ov)
                                if any(_closed_rects_touch(b, grown) for b in b_list):
                                    added_markers.setdefault(m_layer, set()).add(mb)
                                    if mcidx is not None:
                                        for cid, r in mcidx.query(mb):
                                            if _closed_rects_touch(mb, r):
                                                selected_cids.add(cid)

                for cid in selected_cids:
                    for layer, shp in comp_shapes.get(cid, ()):
                        if isinstance(shp, tuple):
                            cell.shapes(_li(layer)).insert(kdb.Box(*shp))
                        else:
                            cell.shapes(_li(layer)).insert(shp)
                for cl, cut_set in selected_cuts.items():
                    sh_cl = cell.shapes(_li(cl))
                    for cb in cut_set:
                        sh_cl.insert(kdb.Box(*cb))
                for m_layer, mset in added_markers.items():
                    sh_m = cell.shapes(_li(m_layer))
                    for mb in mset:
                        sh_m.insert(kdb.Box(*mb))

        for layer, shapes in cand_by_layer.items():
            sh_l = cell.shapes(_li(layer))
            for shp in shapes:
                sh_l.insert(shp)

        return check_deck_net_isolation(
            layout=layout,
            cell=cell,
            stack=stack,
            probes=probes,
        ).ok

    def _emit_net(self, rn: _RoutedNet, *, checkpoint=None) -> _EmittedNetGeometry:
        _NM_PER_UM = _eg_attr("_NM_PER_UM", _base_NM_PER_UM)
        _RoutedNet = _eg_attr("_RoutedNet", _base_RoutedNet)
        segment_wires: List[LayerPolygon] = []
        length = 0.0
        for index, s in enumerate(rn.segs):
            if checkpoint is not None and index % 256 == 0:
                checkpoint()
            key = self._tech.gi_to_metal_key(s.gi)
            half = self._segment_half_width(rn.name, s)
            if s.ay == s.by:  # horizontal
                lo, hi = (s.ax, s.bx) if s.ax <= s.bx else (s.bx, s.ax)
                segment_wires.append(
                    _rect(key, lo - half, s.ay - half, hi + half, s.ay + half)
                )
            else:  # vertical
                lo, hi = (s.ay, s.by) if s.ay <= s.by else (s.by, s.ay)
                segment_wires.append(
                    _rect(key, s.ax - half, lo - half, s.ax + half, hi + half)
                )
            length += (abs(s.bx - s.ax) + abs(s.by - s.ay)) / _NM_PER_UM
        polys = list(segment_wires)
        for index, v in enumerate(rn.vias):
            if checkpoint is not None and index % 256 == 0:
                checkpoint()
            polys.extend(self._via_polys(v))
        return _EmittedNetGeometry(
            segment_wires=tuple(segment_wires),
            wires=tuple(polys),
            length_um=length,
        )

    def _deck_negotiation_snapshot(
        self,
        committed: Mapping[str, _RoutedNet],
    ) -> _DeckNegotiationSnapshot:
        """Extract one canonical deck view from exact fixed and emit geometry.

        Route literals come only from ``_emit_net`` so redundant arrays retain
        their individual cuts and landing pads exactly as emitted.  The complete
        deck stack classifies conductor ownership versus unowned cuts/markers;
        it also supplies the KLayout ``connect()`` and MIM-carve semantics.
        """
        _RoutedNet = _eg_attr("_RoutedNet", _base_RoutedNet)
        from clay.layout.route.deck_isolation import (
            DeckConnectStack,
            DeckLiteralRect,
            check_deck_literal_isolation,
        )

        stack = getattr(self, "_deck_connect_stack", None)
        if stack is None:
            stack = DeckConnectStack.from_tech(self._tech.pdk)
        conductor_layers = set(stack.conductors)
        cut_layers = {connection.cut for connection in stack.connections}
        marker_layers = {carve.top_marker for carve in stack.mim_carves}
        known_layers = conductor_layers | cut_layers | marker_layers
        literal_rects = [
            DeckLiteralRect(tuple(layer), tuple(rect), owner)
            for layer, owner, rect in self._fixed_deck_literal_rects
        ]
        for net in sorted(committed):
            routed = committed[net]
            for polygon in self._emit_net(routed).wires:
                layer = (int(polygon.layer), int(polygon.datatype))
                if layer not in known_layers:
                    raise ValueError(
                        "emitted route rectangle lies outside the complete "
                        f"deck stack: {layer}"
                    )
                points = tuple(
                    (int(point.x), int(point.y))
                    for point in polygon.polygon.vertices
                )
                if len(points) != 4:
                    raise ValueError(
                        "deck negotiation snapshots require emitted rectangles"
                    )
                xs = tuple(point[0] for point in points)
                ys = tuple(point[1] for point in points)
                rect = (min(xs), min(ys), max(xs), max(ys))
                corners = {
                    (rect[0], rect[1]),
                    (rect[2], rect[1]),
                    (rect[2], rect[3]),
                    (rect[0], rect[3]),
                }
                if set(points) != corners:
                    raise ValueError(
                        "deck negotiation snapshots require exact Manhattan "
                        "emitted rectangles"
                    )
                literal_rects.append(DeckLiteralRect(
                    layer,
                    rect,
                    str(routed.name) if layer in conductor_layers else None,
                ))

        raw_literals = tuple(literal_rects)
        report = check_deck_literal_isolation(
            tech=self._tech.pdk,
            stack=stack,
            shapes=raw_literals,
        )
        canonical_literals = tuple(sorted(
            (
                DeckLiteralRect(
                    literal.layer,
                    literal.rect,
                    None if literal.net is None else literal.net.upper(),
                )
                for literal in raw_literals
            ),
            key=lambda literal: (
                literal.layer,
                "" if literal.net is None else literal.net,
                literal.rect,
            ),
        ))
        return _DeckNegotiationSnapshot(canonical_literals, report)

    def _via_polys(self, v: _ViaSite) -> List[LayerPolygon]:
        """The emitted polygons for one via: the cut square + the two landing pads
        at EXACTLY the halves the legality/congestion/occupancy model used
        (``_via_pad_half``: min-width-floored, deck-enclosure-grown, resolved once
        in ``route()`` through the PDK contract surface). Emit and legality share
        ONE pad geometry -- re-deriving enclosures here from a different source is
        how the R7a flush-pad family (VIA1.EN.1 / M2.W.1 / M3.W.1) slipped out."""
        self._require_policy_via(v)
        tech = self._tech
        lo = v.lo
        cut = self._via_cut[lo]
        bh, ah = self._via_pad_half_for(v)
        polys: List[LayerPolygon] = []
        polys.append(_rect(tech.gi_to_metal_key(lo),
                           v.x - bh, v.y - bh, v.x + bh, v.y + bh))
        polys.append(_rect(tech.gi_to_metal_key(lo + 1),
                           v.x - ah, v.y - ah, v.x + ah, v.y + ah))
        vk = tech.via_key_for(lo)
        if vk is not None:
            # ONE cut per array centre (a single centre when via_redundancy=1):
            # the N x N redundant array the grown pad above
            # encloses, cuts at the via{k}_space rule pitch so the drawn artwork is
            # DRC-legal.
            for (cx, cy) in self._via_cut_centers(v.x, v.y, lo, self._site_n(v)):
                rx0, ry0, rx1, ry1 = landing_rect_for_cut(cx, cy, cut, 0)
                polys.append(_rect(vk, rx0, ry0, rx1, ry1))
        return polys

    # ------------------------------------------------------- spacing residuals
    def _detect_space_residuals(self, committed: Dict[str, _RoutedNet],
                                ) -> List[Tuple[str, str]]:
        """The STRICT advisory residual pairs -- first element of
        ``_space_residual_scan`` (detection semantics unchanged; see that
        method for the full two-tier contract)."""
        _RoutedNet = _eg_attr("_RoutedNet", _base_RoutedNet)
        return self._space_residual_scan(committed)[0]

    @staticmethod
    def _merged_same_net_metal_is_rule_clear(
        rectangles: Sequence[Rect], *, minimum_width: int, minimum_space: int,
        checkpoint=None,
    ) -> bool:
        """Prove that literal same-net output has no visible merged-rule error.

        Internal rectangle edges are not drawn polygon boundaries. A diagonal
        pair can therefore look sub-minimum even when another actual wire hides
        its nearest edge. Only the full union of supplied, already materialized
        metal may discharge that conservative pair witness. No native proxy,
        inferred bounding-box fill, or foreign-net geometry participates.
        """
        import klayout.db as kdb

        drawn = kdb.Region()
        for index, rectangle in enumerate(rectangles):
            if checkpoint is not None and index % 256 == 0:
                checkpoint()
            drawn.insert(kdb.Box(*_norm(rectangle)))
        if checkpoint is not None:
            checkpoint()
        drawn.merge()
        if checkpoint is not None:
            checkpoint()
        if not drawn.space_check(int(minimum_space)).is_empty():
            return False
        if checkpoint is not None:
            checkpoint()
        if not drawn.notch_check(int(minimum_space)).is_empty():
            return False
        if checkpoint is not None:
            checkpoint()
        result = drawn.width_check(int(minimum_width)).is_empty()
        if checkpoint is not None:
            checkpoint()
        return result

    def _space_residual_scan(self, committed: Dict[str, _RoutedNet],
                             ) -> Tuple[List[Tuple[str, str]],
                                        List[Tuple[str, str]]]:
        """``(strict_pairs, deck_visible_pairs)`` -- the engine's OWN spacing
        judgment over the emitted-model geometry, split into its two honest
        tiers. The independent KLayout deck stays the oracle for BOTH.

        STRICT (ADVISORY; the historical ``_detect_space_residuals`` verdict,
        byte-identical): two same-layer rects whose edge-to-edge gap lies in
        the OPEN band (0, min_space) -- NOT touching (touching cross-net metal
        is a SHORT, ``_detect_shorts``' account) and NOT min-space clear --
        or a TOUCHING/OVERLAPPING same-net pair failing the JUNCTION-WIDTH
        obligation (contact diagonal < min-width). Judged over:

          * committed nets pairwise (cross-net positive spacing-band residuals
            tolerated by soft congestion), INCLUDING a net against its own
            disjoint branches;
          * committed metal against the device artwork, ANY owner (own-net
            included -- guarded hard in ``_clear_obstacles`` since R7a).

        STRICT is deliberately SUFFICIENT-not-necessary for deck cleanliness
        (gap coverage is demanded over a min-width-GROWN swath; the junction-
        width obligation models the deck's WIDTH family on merged contacts).

        DECK-VISIBLE (the CONVERGENCE gate): the subset of strict pairs that
        would fire the deck's ACTUAL space rule on MERGED drawn geometry --
        merged, net-blind, EUCLIDEAN, at the SAME parsed rule values
        (``self._space`` / ``self._via_space``; no constants of its own):

          * a TOUCHING/OVERLAPPING pair with a POSITIVE-measure contact is
            ABSORBED by the drawn union (space() has no gap left to flag)
            -> deck-invisible; its junction-width obligation stays a strict
            advisory. A zero-measure corner KISS does NOT merge clean (the
            measured M4.W.1/M4.S.1 kiss family) -> stays deck-visible.
          * an OPEN pair euclidean-clear at the rule value -> deck-invisible
            (the cut model's either-axis clearance is stricter on diagonals).
          * an OPEN sub-min pair whose RAW facing band is FULLY metallised
            (merged union continuous across the whole facing region: nothing
            faces across empty space) -> deck-invisible. Any slit a partial
            cover leaves is a scanned pair of its own and stays visible; a
            degenerate zero-extent band cannot be certified absorbed ->
            visible unless the exact same-owner union proof below clears it.
          * a SAME-owner fixed-artwork witness may be discharged only when
            the target is a canonical fixed literal and the actual routed
            metal plus same-layer, same-owner non-abstract fixed literals
            pass merged width, space, and notch checks. Ownership of an
            obstacle alone never establishes that it is drawn metal.

        The negotiation loop's STOP consumes ONLY the deck-visible set (the
        measured burn: gates spent the whole route budget negotiating strict
        residuals the real deck never flags -- success=False, residuals>0,
        deck DRC total=0); the strict set is reported as
        ``advisory_residuals`` -- never silently dropped. Pure function of
        geometry + parsed rules: deterministic, no wall-clock.

        The rasterised ``_occupancy`` over-approximates on pitch cells.  This
        exact detector additionally publishes resource cells touched by each
        DECK-VISIBLE witness so DRC-driven rip-up retains local history even for
        same-net residuals (one owner cannot make an occupancy cell overused).
        Advisory-only geometry publishes no history.  Returns sorted unique
        net-name pairs (an artwork residual reports the routed net against itself
        -- artwork is immovable, the net is the only victim)."""
        _RoutedNet = _eg_attr("_RoutedNet", _base_RoutedNet)
        _rects_clear = _eg_attr("_rects_clear", _base_rects_clear)
        _sep = _eg_attr("_sep", _base_sep)
        spacing_history_cells: Set[Tuple[int, int, int]] = set()
        self._last_deck_spacing_history_cells = frozenset()

        def _remember_rect(gi: int, rect: Rect, grow: int) -> None:
            x0, y0, x1, y1 = _norm(rect)
            rc = self._res_cell
            for bx in range((x0 - grow) // rc, (x1 + grow) // rc + 1):
                for by in range((y0 - grow) // rc, (y1 + grow) // rc + 1):
                    spacing_history_cells.add((bx, by, gi))

        per_net: Dict[str, Dict[int, List[Rect]]] = {}
        cuts_by_lo: Dict[int, List[Tuple[str, Rect]]] = {}
        for name, rn in committed.items():
            self._watchdog_check(name, "spacing_scan")
            if not rn.segs and not rn.vias:
                continue
            metal, cuts = self._net_metal(rn)
            per_net[name] = metal
            for (r, lo) in cuts:
                cuts_by_lo.setdefault(lo, []).append((name, r))
        nets = sorted(per_net)
        # ONE spatial index over every committed net's metal per gi -- the
        # coverage judgment's hole source. The old all-nets LINEAR scan made
        # each covered-pair judgment O(total rects); a dense same-net pile
        # carries ~6*10^4 banded pairs, so the detector burned minutes per
        # call proving coverage. O(local) per query, byte-identical verdicts.
        all_metal_idx: Dict[int, _SpatialIndex] = {}
        per_net_metal_idx: Dict[Tuple[str, int], _SpatialIndex] = {}
        merged_same_net_clear: Dict[Tuple[str, int], bool] = {}
        for name in nets:
            for gi2, rects2 in per_net[name].items():
                idx2 = all_metal_idx.get(gi2)
                if idx2 is None:
                    all_metal_idx[gi2] = idx2 = _SpatialIndex()
                net_idx = _SpatialIndex()
                per_net_metal_idx[(name, gi2)] = net_idx
                for ordinal, r2 in enumerate(rects2):
                    idx2.add(name, _norm(r2))
                    net_idx.add(str(ordinal), _norm(r2))

        def _same_net_union_clear(name: str, gi: int) -> bool:
            key = (name, gi)
            if key not in merged_same_net_clear:
                merged_same_net_clear[key] = self._merged_same_net_metal_is_rule_clear(
                    per_net[name][gi],
                    minimum_width=2 * self._half_w[gi],
                    minimum_space=self._space[gi],
                )
            return merged_same_net_clear[key]

        same_net_fixed_literals: Dict[Tuple[str, int], Tuple[Rect, ...]] = {}
        merged_same_net_with_fixed_clear: Dict[Tuple[str, int], bool] = {}

        def _same_net_fixed_literals(name: str, gi: int) -> Tuple[Rect, ...]:
            # Direct spacing callers may have no deck/technology world yet.
            # Empty literal authority cannot discharge any fixed witness.
            if not self._fixed_deck_literal_rects:
                return ()
            key = (name, gi)
            if key not in same_net_fixed_literals:
                layer = tuple(self._tech.gi_to_metal_key(gi))
                abstract = self._abstract_rects.get(gi, ())
                same_net_fixed_literals[key] = tuple(
                    rect for drawn_layer, owner, rect in self._fixed_deck_literal_rects
                    if tuple(drawn_layer) == layer and owner == name and rect not in abstract
                )
            return same_net_fixed_literals[key]

        def _same_net_with_fixed_union_clear(name: str, gi: int) -> bool:
            key = (name, gi)
            if key not in merged_same_net_with_fixed_clear:
                merged_same_net_with_fixed_clear[key] = self._merged_same_net_metal_is_rule_clear(
                    [*per_net[name][gi], *_same_net_fixed_literals(name, gi)],
                    minimum_width=2 * self._half_w[gi],
                    minimum_space=self._space[gi],
                )
            return merged_same_net_with_fixed_clear[key]

        def _band(n1: Rect, n2: Rect, sp: int, wj: int = 0,
                  ) -> Optional[Tuple[int, int]]:
            """The (sx, sy) separation when the pair sits in the OPEN sub-min
            band, else None. EUCLIDEAN metric -- the deck's space() default --
            so a diagonal corner pair at sqrt(sx^2+sy^2) >= sp is NOT flagged
            (the Chebyshev legality predicate stays conservative for search;
            the convergence judgment must not demand what the deck does not).

            ``wj`` > 0 additionally applies the JUNCTION-WIDTH obligation to
            TOUCHING/OVERLAPPING pairs (same-net or own-artwork callers): the
            deck merges them into one polygon, which is width-clean only when
            the contact diagonal >= min-width (``ox^2 + oy^2 >= wj^2``). A
            corner point-touch (ox = oy = 0) or sliver contact merges into a
            sub-min-width neck -- the measured M4.W.1 x2 + M4.S.1 x2 at ONE
            kiss point (67.038, 71.038). Reported as (0, 0). ``wj = 0``
            (cross-net callers) keeps touch = short = ``_detect_shorts``'
            account."""
            sx, sy = _sep(n1, n2)
            if not (sx or sy):
                if wj:
                    ox = min(n1[2], n2[2]) - max(n1[0], n2[0])
                    oy = min(n1[3], n2[3]) - max(n1[1], n2[1])
                    if ox * ox + oy * oy < wj * wj:
                        return (0, 0)
                return None  # merged-clean contact (or a short)
            return (sx, sy) if (sx * sx + sy * sy) < sp * sp else None

        def _gap_covered(gi: int, n1: Rect, n2: Rect, sx: int, sy: int) -> bool:
            """True when the open band between ``n1`` and ``n2`` is FULLY
            metallised by other same-layer geometry (committed metal of any net
            + device artwork -- net-blind, exactly the deck's merge): the merged
            polygon is continuous across the gap, so no boundary edges face each
            other there and the deck's space() has nothing to flag (a staircase
            whose connector fills the step; a junction pad bridging two arms).

            A projection-THIN gap box (down to zero width: a colinear-edge
            corner-touch pair) is judged over the box GROWN to the layer
            min-width on the narrow axis: the raw box can never be called
            covered (zero area), and a filament-thin cover would leave a real
            deck notch anyway -- the min-width swath is the geometry the
            euclidean space oracle certifies clean (micro-oracle, R7). A
            JUNCTION pair (sx == sy == 0) is judged over the SAME min-width
            swath centred on its contact box (a 2w footprint over-demanded
            cover the deck does not). Model-only keep-outs never count as
            cover."""
            ax0, ay0, ax1, ay1 = n1
            bx0, by0, bx1, by1 = n2
            gx0, gx1 = ((max(ax0, bx0), min(ax1, bx1)) if sx == 0
                        else (min(ax1, bx1), max(ax0, bx0)))
            gy0, gy1 = ((max(ay0, by0), min(ay1, by1)) if sy == 0
                        else (min(ay1, by1), max(ay0, by0)))
            w = 2 * self._half_w[gi]
            if gx1 - gx0 < w:
                cx = (gx0 + gx1) // 2
                gx0, gx1 = cx - w // 2, cx - w // 2 + w
            if gy1 - gy0 < w:
                cy = (gy0 + gy1) // 2
                gy0, gy1 = cy - w // 2, cy - w // 2 + w
            return _box_metallised(gi, (gx0, gy0, gx1, gy1))

        def _box_metallised(gi: int, gap: Rect) -> bool:
            """True when ``gap`` is FULLY covered by drawn same-layer geometry
            (committed metal of any net + non-abstract device artwork --
            net-blind, exactly the deck's merge)."""
            gx0, gy0, gx1, gy1 = gap
            abs_gi = self._abstract_rects.get(gi, ())
            holes: List[Rect] = []
            midx2 = all_metal_idx.get(gi)
            if midx2 is not None:
                for (_ow, r) in midx2.query(gap):
                    rr = _norm(r)
                    if rr[0] < gx1 and gx0 < rr[2] and rr[1] < gy1 and gy0 < rr[3]:
                        holes.append(rr)
            oidx = self._obs_idx.get(gi)
            if oidx is not None:
                for (_ow, r) in oidx.query(gap):
                    if r in abs_gi:
                        continue
                    rr = _norm(r)
                    if rr[0] < gx1 and gx0 < rr[2] and rr[1] < gy1 and gy0 < rr[3]:
                        holes.append(rr)
            # exact fast path: one containing hole covers the swath (the
            # dominant dense-pile case); else exact subtraction.
            for rr in holes:
                if (rr[0] <= gx0 and rr[1] <= gy0
                        and rr[2] >= gx1 and rr[3] >= gy1):
                    return True
            return not subtract_rects(gap, holes)

        def _deck_visible(gi: int, n1: Rect, n2: Rect, sx: int, sy: int,
                          sp: int) -> bool:
            """Would the deck's space rule fire on this STRICT-flagged pair in
            MERGED drawn geometry? (merged, net-blind, euclidean, parsed rule
            values -- the classification contract in the method docstring.)"""
            if sx == 0 and sy == 0:
                # touching/overlapping: the union absorbs a POSITIVE-measure
                # contact (junction-width stays a WIDTH-family advisory); a
                # zero-measure corner kiss does not merge clean (measured
                # M4.W.1/M4.S.1 kiss family) -> visible.
                ox = min(n1[2], n2[2]) - max(n1[0], n2[0])
                oy = min(n1[3], n2[3]) - max(n1[1], n2[1])
                return ox == 0 and oy == 0
            if sx * sx + sy * sy >= sp * sp:
                return False  # euclidean-clear at the parsed rule (deck metric)
            ax0, ay0, ax1, ay1 = n1
            bx0, by0, bx1, by1 = n2
            gx0, gx1 = ((max(ax0, bx0), min(ax1, bx1)) if sx == 0
                        else (min(ax1, bx1), max(ax0, bx0)))
            gy0, gy1 = ((max(ay0, by0), min(ay1, by1)) if sy == 0
                        else (min(ay1, by1), max(ay0, by0)))
            if gx1 <= gx0 or gy1 <= gy0:
                return True  # degenerate facing band: absorption not certifiable
            # RAW facing band (no min-width growth -- that growth is the
            # strict tier's sufficient-not-necessary demand): fully
            # metallised -> the merged union is continuous across the gap.
            return not _box_metallised(gi, (gx0, gy0, gx1, gy1))

        def _cut_visible(r1: Rect, r2: Rect, vsp: int) -> bool:
            """Deck-visibility for a cut pair: EUCLIDEAN at the parsed
            via-space rule (``_rects_clear`` is either-axis -- stricter on
            diagonal pairs). Non-identical touching/overlapping cuts stay
            visible: never engine-legal by construction, and a merged cut
            blob is a cut exact-size defect -- not worth relaxing."""
            sx, sy = _sep(_norm(r1), _norm(r2))
            if sx == 0 and sy == 0:
                return True
            return sx * sx + sy * sy < vsp * vsp

        strict: Set[Tuple[str, str]] = set()
        vis: Set[Tuple[str, str]] = set()
        # committed vs device artwork (net-blind, like the deck; model-only
        # keep-outs excluded -- they are not drawn). Junction-width applies:
        # foreign/unowned artwork is hard-banded so only OWN artwork can ever
        # touch, and a sub-min-width contact with it is a real merged neck.
        # A pair keeps scanning past deck-INVISIBLE hits until a visible one
        # (or exhaustion): the strict set is any-hit (unchanged), visibility
        # is any-visible-hit.
        for a in nets:
            self._watchdog_check(a, "spacing_scan")
            hit = False
            hit_vis = False
            for gi, rects in per_net[a].items():
                sp = self._space[gi]
                wj = 2 * self._half_w[gi]
                oidx = self._obs_idx.get(gi)
                if oidx is None:
                    continue
                abs_gi = self._abstract_rects.get(gi, ())
                for r1 in rects:
                    n1 = _norm(r1)
                    q = (n1[0] - sp, n1[1] - sp, n1[2] + sp, n1[3] + sp)
                    for (_owner, r2) in oidx.query(q):
                        if r2 in abs_gi:
                            continue
                        n2 = _norm(r2)
                        b = _band(n1, n2, sp, wj)
                        if b is not None and not _gap_covered(gi, n1, n2, *b):
                            hit = True
                            if (
                                _deck_visible(gi, n1, n2, b[0], b[1], sp)
                                and (
                                    _owner != a
                                    or n2 not in _same_net_fixed_literals(a, gi)
                                    or not _same_net_with_fixed_union_clear(a, gi)
                                )
                            ):
                                hit_vis = True
                                _remember_rect(gi, n1, sp // 2)
                                break
                    if hit_vis:
                        break
                if hit_vis:
                    break
            if hit:
                strict.add((a, a))
            if hit_vis:
                vis.add((a, a))
        # committed vs committed: cross-net pairs + a net's own disjoint
        # branches. Junction-width applies to SAME-net pairs only (a touching
        # cross-net pair is a SHORT -- ``_detect_shorts``' account).
        for i in range(len(nets)):
            self._watchdog_check(nets[i], "spacing_scan")
            for j in range(i, len(nets)):
                a, b = nets[i], nets[j]
                ma, mb = per_net[a], per_net[b]
                hit = False
                hit_vis = False
                for gi, ra_ in ma.items():
                    rb_ = mb.get(gi)
                    if not rb_:
                        continue
                    sp = self._space[gi]
                    wj = 2 * self._half_w[gi] if a == b else 0
                    rb_idx = per_net_metal_idx[(b, gi)]
                    for i1, r1 in enumerate(ra_):
                        n1 = _norm(r1)
                        query = (
                            n1[0] - sp,
                            n1[1] - sp,
                            n1[2] + sp,
                            n1[3] + sp,
                        )
                        nearby = sorted(
                            (
                                (int(ordinal), r2)
                                for ordinal, r2 in rb_idx.query(query)
                                if ordinal is not None
                                and (a != b or int(ordinal) > i1)
                            ),
                            key=lambda item: item[0],
                        )
                        for _ordinal, r2 in nearby:
                            n2 = _norm(r2)
                            bd = _band(n1, n2, sp, wj)
                            if bd is not None and not _gap_covered(gi, n1, n2, *bd):
                                hit = True
                                if (
                                    _deck_visible(gi, n1, n2, bd[0], bd[1], sp)
                                    and (a != b or not _same_net_union_clear(a, gi))
                                ):
                                    hit_vis = True
                                    _remember_rect(gi, n1, sp // 2)
                                    _remember_rect(gi, n2, sp // 2)
                                    break
                        if hit_vis:
                            break
                    if hit_vis:
                        break
                if hit:
                    strict.add((a, b))
                if hit_vis:
                    vis.add((a, b))
        # cut-to-cut residuals (net-blind, per via level): IDENTICAL cuts merge
        # (the shared escape via -- emit dedups); anything else within the
        # via-space rule is a real VIA{k}.S.1, same-net pairs included (the
        # measured x3 were same-net).
        for lo, entries in cuts_by_lo.items():
            vsp = self._via_space.get(lo, 0)
            if not vsp:
                continue
            for i1 in range(len(entries)):
                a, r1 = entries[i1]
                self._watchdog_check(a, "via_spacing_scan")
                for j1 in range(i1 + 1, len(entries)):
                    b, r2 = entries[j1]
                    if r1 == r2:
                        continue
                    if not _rects_clear(r1, r2, vsp):
                        pr = (a, b) if a <= b else (b, a)
                        strict.add(pr)
                        if _cut_visible(r1, r2, vsp):
                            vis.add(pr)
                            resource_gi = self._n_layers + lo
                            _remember_rect(resource_gi, r1, vsp // 2)
                            _remember_rect(resource_gi, r2, vsp // 2)
                # ARTWORK cuts (stamp risers): net-blind, identical merges;
                # the routed net is the movable party -> reported (a, a).
                for r2 in self._obs_cuts.get(lo, ()):
                    if r1 == r2:
                        continue
                    if not _rects_clear(r1, r2, vsp):
                        strict.add((a, a))
                        if _cut_visible(r1, r2, vsp):
                            vis.add((a, a))
                            _remember_rect(
                                self._n_layers + lo,
                                r1,
                                vsp // 2,
                            )
        # Width/halo rules classify the entire literal union. They cannot be
        # discharged by the base-only pair proof above, nor by a shape bbox.
        # This runs once at convergence, never once per search edge.
        for pair, gi, rectangle, required in self._wide_spacing_residuals(
            per_net, all_metal_idx,
        ):
            strict.add(pair)
            vis.add(pair)
            _remember_rect(gi, rectangle, required // 2)
        self._last_deck_spacing_history_cells = frozenset(
            spacing_history_cells
        )
        return sorted(strict), sorted(vis)

    def _wide_spacing_residuals(self, per_net, route_indexes):
        import klayout.db as kdb
        from .wide_metal_spacing import wide_spacing_checks

        self._ensure_fixed_wide_geometry_indexes()
        if not any(self._wide_geometry_entries_by_gi.values()):
            return ()
        tech = self._tech
        by_gi = {}
        for net, layers in per_net.items():
            self._watchdog_check(net, "wide_spacing_scan")
            for gi, rectangles in layers.items():
                if not self._wide_geometry_entries_by_gi.get(gi):
                    continue
                drawn = by_gi.setdefault(gi, kdb.Region())
                for rectangle in rectangles:
                    drawn.insert(kdb.Box(*_norm(rectangle)))
        fixed_by_gi = {}
        for gi, rows in self._fixed_literal_metal_rows().items():
            if gi in by_gi:
                fixed = fixed_by_gi.setdefault(gi, kdb.Region())
                for _owner, rectangle in rows:
                    box = kdb.Box(*_norm(rectangle))
                    by_gi[gi].insert(box)
                    fixed.insert(box)
        result = set()
        for gi, drawn in sorted(by_gi.items()):
            name = tech.metal_names[tech._gi_to_pdk[gi]]
            index = route_indexes.get(gi)
            if index is None:
                continue
            drawn.merge()
            checks = wide_spacing_checks(drawn, tech.pdk, name)
            fixed_checks = wide_spacing_checks(fixed_by_gi.get(gi, kdb.Region()), tech.pdk, name)
            components = None
            for check_number, check in enumerate(checks):
                for error in check.errors.each():
                    owners = []
                    for edge in (error.first, error.second):
                        box = edge.bbox()
                        rectangle = (box.left, box.bottom, box.right, box.top)
                        query = (box.left - 1, box.bottom - 1, box.right + 1, box.top + 1)
                        owners.append({str(owner) for owner, metal in index.query(query)
                                       if owner is not None and _closed_rects_touch(rectangle, metal)})
                    if not (owners[0] or owners[1]):
                        # Route additions can make inherited fixed edges wide
                        # without touching the violating edges themselves. First
                        # exclude unchanged fixed-only errors, then attribute the
                        # new obligation to actual route material in the affected
                        # merged components, never their bounding boxes.
                        marker = kdb.EdgePairs()
                        marker.insert(error)
                        if (marker.polygons(0) - fixed_checks[check_number].errors.polygons(0)).is_empty():
                            continue
                        if components is None:
                            components = tuple(kdb.Region(polygon) for polygon in drawn.each())
                        for side, edge in enumerate((error.first, error.second)):
                            neighborhood = kdb.Region(edge.bbox().enlarged(1))
                            affected = [component for component in components
                                        if not (component & neighborhood).is_empty()]
                            for route_owner, layers in per_net.items():
                                if gi not in layers:
                                    continue
                                actual = kdb.Region()
                                for rectangle in layers[gi]:
                                    actual.insert(kdb.Box(*_norm(rectangle)))
                                if any(not (actual & component).is_empty() for component in affected):
                                    owners[side].add(str(route_owner))
                    if not (owners[0] or owners[1]):
                        continue
                    pairs = ({tuple(sorted((left, right)))
                              for left in owners[0] for right in owners[1]}
                             if all(owners) else {(owner, owner) for owner in owners[0] | owners[1]})
                    box = error.bbox()
                    rectangle = (box.left, box.bottom, box.right, box.top)
                    for pair in pairs:
                        result.add((pair, gi, rectangle, check.spacing_nm))
        return tuple(sorted(result))

    # ------------------------------------------------------------- short oracle
    def _detect_shorts(self, committed: Dict[str, _RoutedNet]) -> List[Tuple[str, str]]:
        """Exact same-layer electrical contact in the emitted-model geometry.

        Both committed-route vs committed-route and committed-route vs literally
        drawn fixed artwork participate.  Search legality normally keeps foreign
        fixed artwork clear, but final convergence is an independent judgment of
        the geometry it will actually emit: a construction/legalization defect
        cannot turn green merely because the other electrical owner is immovable.
        Routing abstractions are not drawn and therefore never participate; own-net
        artwork contact is the intended terminal connection.  Unowned fixed artwork
        is a unary obligation encoded as ``(net, net)``: no magic owner can collide
        with the circuit namespace, and victim selection naturally reroutes only
        that declared net.

        ``route_cell`` still runs the independent KLayout-region union-find; this is
        the engine's deterministic native safety signal, not a signoff substitute.
        """
        _RoutedNet = _eg_attr("_RoutedNet", _base_RoutedNet)
        _sep = _eg_attr("_sep", _base_sep)
        per_net: Dict[str, Dict[int, List[Rect]]] = {}
        for name, rn in committed.items():
            self._watchdog_check(name, "short_scan")
            metal, _cuts = self._net_metal(rn)
            per_net[name] = metal
        nets = sorted(per_net)
        merged: Set[Tuple[str, str]] = set()
        for i in range(len(nets)):
            self._watchdog_check(nets[i], "short_scan")
            for j in range(i + 1, len(nets)):
                a, b = nets[i], nets[j]
                ma, mb = per_net[a], per_net[b]
                hit = False
                for gi, ra in ma.items():
                    rb = mb.get(gi)
                    if not rb:
                        continue
                    for r1 in ra:
                        for r2 in rb:
                            sx, sy = _sep(r1, r2)
                            if sx == 0 and sy == 0:
                                hit = True
                                break
                        if hit:
                            break
                    if hit:
                        break
                if hit:
                    merged.add((a, b))
        # Committed route metal against the immutable, literally drawn artwork.
        # A positive-area overlap, edge abutment, or corner kiss is electrical
        # contact under the same closed-rectangle authority used by component
        # connectivity.  Query only local spatial-index buckets, then judge exact
        # rectangles; no bbox envelope or clearance-band approximation enters.
        for a in nets:
            self._watchdog_check(a, "short_scan_fixed_artwork")
            candidate_count = 0
            for gi, route_rects in per_net[a].items():
                oidx = self._literal_planar_idx.get(gi)
                if oidx is None:
                    continue
                for route_index, route_rect in enumerate(route_rects):
                    if route_index % 1024 == 0:
                        self._watchdog_check(a, "short_scan_fixed_artwork")
                    nroute = _norm(route_rect)
                    for owner, artwork_rect in oidx.query(nroute):
                        if candidate_count % 1024 == 0:
                            self._watchdog_check(
                                a,
                                "short_scan_fixed_artwork_candidates",
                            )
                        candidate_count += 1
                        nart = _norm(artwork_rect)
                        if owner == a:
                            continue
                        if not _closed_rects_touch(nroute, nart):
                            continue
                        fixed_owner = (
                            str(a)
                            if owner is None
                            else str(owner)
                        )
                        merged.add(tuple(sorted((str(a), fixed_owner))))
        return sorted(merged)

