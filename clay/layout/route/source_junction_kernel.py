"""Source/route junction repair for complete committed routing worlds."""
from collections import Counter


def _intersection(a, b):
    r = (max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))
    return r if r[0] < r[2] and r[1] < r[3] else None


def _edge_touches(edge, box):
    """Closed intersection with the actual Manhattan edge, not its pair bbox."""
    x0,y0,x1,y1=box
    a,b=edge.p1,edge.p2
    if a.x == b.x:
        return x0 <= a.x <= x1 and max(y0,min(a.y,b.y)) <= min(y1,max(a.y,b.y))
    if a.y == b.y:
        return y0 <= a.y <= y1 and max(x0,min(a.x,b.x)) <= min(x1,max(a.x,b.x))
    return False


def _junction_candidate_evidence(width_nm, grid_nm, edge_pairs, source_rects, route_rects, checkpoint=None):
    """Return candidate squares and all identified source/route marker IDs.

    Both edges of an owned width pair touch the actual positive-area intersection
    of a source and a route rectangle, and the candidate touches at least one
    of those edges. A bounding box alone never authorizes a candidate.
    """
    if type(width_nm) is not int or width_nm <= 0 or type(grid_nm) is not int or grid_nm <= 0:
        raise ValueError('positive integer rule and manufacturing grid required')
    width = -(-width_nm // grid_nm) * grid_nm
    result = {}
    identified = set()
    def align(values):
        return {v // grid_nm * grid_nm for value in values for v in (value, value + grid_nm - 1)}
    for source in source_rects:
        if checkpoint is not None: checkpoint()
        for index, route in enumerate(route_rects):
            if checkpoint is not None and index % 256 == 0: checkpoint()
            overlap = _intersection(source, route)
            if not overlap: continue
            relevant=[]
            for pair_index,p in enumerate(edge_pairs):
                if checkpoint is not None and pair_index % 256 == 0: checkpoint()
                if _edge_touches(p.first,overlap) and _edge_touches(p.second,overlap):
                    relevant.append(p)
            if not relevant: continue
            identified.update(str(pair) for pair in relevant)
            xs = align((overlap[0], overlap[2], overlap[0]-width, overlap[2]-width))
            ys = align((overlap[1], overlap[3], overlap[1]-width, overlap[3]-width))
            for x in xs:
                for y in ys:
                    candidate = (x, y, x+width, y+width)
                    if not _intersection(candidate, overlap): continue
                    associated={str(p) for p in relevant if _edge_touches(p.first,candidate)
                                or _edge_touches(p.second,candidate)}
                    if associated: result.setdefault(candidate,set()).update(associated)
    return (tuple((box,tuple(sorted(ids))) for box,ids in sorted(result.items())),
            frozenset(identified))


def generate_junction_candidates(width_nm, grid_nm, edge_pairs, source_rects, route_rects, checkpoint=None):
    """Return ``(square, associated marker IDs)`` for exact edge geometry."""
    return _junction_candidate_evidence(
        width_nm, grid_nm, edge_pairs, source_rects, route_rects, checkpoint,
    )[0]


def _rects(emitted, layer):
    rows = []
    for p in emitted.wires:
        if (int(p.layer), int(p.datatype)) != layer:
            continue
        xy = tuple((int(v.x),int(v.y)) for v in p.polygon.vertices)
        if len(xy) != 4:
            raise ValueError('emitted metal must be an exact rectangle')
        rect=(min(x for x,y in xy), min(y for x,y in xy),
              max(x for x,y in xy), max(y for x,y in xy))
        if set(xy) != {(rect[0],rect[1]),(rect[0],rect[3]),
                       (rect[2],rect[1]),(rect[2],rect[3])}:
            raise ValueError('emitted metal must be Manhattan')
        rows.append(rect)
    return tuple(rows)


def _exact_ref_union_overlaps(engine, net, gi, source, drawn, checkpoint):
    """Decompose only exact source/route intersection, preserving holes/ref."""
    import klayout.db as kdb
    from clay.layout.route.wide_metal_spacing import exact_region_rectangles

    source_set=set(source)
    refs={}
    for index,atom in enumerate(getattr(engine,'_owned_terminal_atoms',())):
        if index % 256 == 0: checkpoint()
        rect=tuple(atom.rect)
        ref=atom.terminal_conductor_ref
        if (atom.owner==net and int(atom.gi)==gi and rect in source_set
                and type(ref) is str and ref):
            refs.setdefault(ref,set()).add(rect)
    route=kdb.Region()
    for index,rect in enumerate(drawn):
        if index % 256 == 0: checkpoint()
        route.insert(kdb.Box(*rect))
    route.merge();checkpoint()
    result=[]
    for ref,rectangles in sorted(refs.items()):
        if len(rectangles)<2: continue
        source_region=kdb.Region()
        for index,rect in enumerate(sorted(rectangles)):
            if index % 256 == 0: checkpoint()
            source_region.insert(kdb.Box(*rect))
        overlap=(source_region.merged() & route).merged();checkpoint()
        for rect in exact_region_rectangles(overlap):
            result.append((ref,rect))
        checkpoint()
    return tuple(result)


def _markers(engine, snapshot, checkpoint):
    import klayout.db as kdb
    marks, edgepairs = {}, {}
    for gi in range(engine._n_layers):
        checkpoint()
        layer=tuple(engine._tech.gi_to_metal_key(gi))
        region=kdb.Region()
        for index,literal in enumerate(snapshot.literal_rects):
            if index % 256 == 0: checkpoint()
            if tuple(literal.layer)==layer:
                region.insert(kdb.Box(*literal.rect))
        region.merge()
        width=int(engine._tech.gi_min_width(gi))
        space=int(engine._space[gi])
        if width <= 0 or space <= 0: raise ValueError('missing physical rule')
        edges=list(region.width_check(width).each())
        edgepairs[gi]=tuple(edges)
        marks[gi]={'width':{str(p) for p in edges},
                   'space':{str(p) for p in region.space_check(space).each()},
                   'notch':{str(p) for p in region.notch_check(space).each()}}
        checkpoint()
    return marks,edgepairs


def committed_world_matches_book(engine, routes):
    """Fail closed unless all booked metal, via pads and cut extents are present.

    Book entries may duplicate persistent group baselines already included in
    the routes; compare their geometric sets. Exact individual emitted cuts
    are subsequently supplied by the normal deck snapshot, not by this book.
    """
    book=getattr(engine,'_book',None)
    if book is None or not routes: return False
    metal=set();pads=set();cuts=set()
    for net,route in sorted(routes.items()):
        engine._watchdog_check(net,'source_junction_world_binding')
        if route.name != net: return False
        route_metal,route_cuts=engine._net_metal(route)
        route_pads=engine._net_via_pads(route)
        metal.update((int(gi),net,tuple(rect)) for gi,rects in route_metal.items() for rect in rects)
        pads.update((int(gi),net,tuple(rect)) for gi,rects in route_pads.items() for rect in rects)
        cuts.update((net,tuple(rect),int(lo)) for rect,lo in route_cuts)
    actual_metal={(int(gi),net,tuple(rect)) for gi,rows in book.metal.items() for net,rect in rows}
    actual_pads={(int(gi),net,tuple(rect)) for gi,rows in book.via_pads.items() for net,rect in rows}
    actual_cuts={(net,tuple(rect),int(lo)) for net,rect,lo in book.cuts}
    engine._watchdog_check(next(iter(sorted(routes))),'source_junction_world_binding')
    if metal!=actual_metal or pads!=actual_pads or cuts!=actual_cuts:
        return False
    # Book cuts are conservative array bboxes. Prove every persistent via's
    # complete, individually emitted cut and pad rectangles is in this exact
    # output world too. Compare emitted geometry so default n=0 and explicit
    # n=2 are accepted when they really materialize the same 2x2 array.
    def via_geometry(via):
        polygons=engine._via_polys(via)
        return tuple(sorted((int(p.layer),int(p.datatype),
                             tuple(sorted((int(v.x),int(v.y)) for v in p.polygon.vertices)))
                            for p in polygons))
    for net,baseline in sorted(getattr(engine,'_group_route_landing_baselines',{}).items()):
        engine._watchdog_check(net,'source_junction_via_binding')
        route=routes.get(net)
        if route is None:
            return False
        emitted={via_geometry(v) for v in route.vias}
        for via in baseline.vias:
            engine._watchdog_check(net,'source_junction_via_binding')
            if via_geometry(via) not in emitted:
                return False
    engine._watchdog_check(next(iter(sorted(routes))),'source_junction_via_binding')
    return True


def _candidate_book_indices_match(engine, checkpoint):
    """Verify the existing spatial indexes exactly reflect current Book rows.

    ``_fill_legal`` consumes the live metal index, while via pads must also be
    represented in that metal ledger. Reuse the engine's own spatial-index
    insertion semantics to detect stale or misplaced buckets without mutating
    the caller's Book, index generation, or caches.
    """
    from clay.layout.route.engine_gridless import _SpatialIndex, _norm

    book = getattr(engine, '_book', None)
    if book is None:
        return False

    def claims(rows):
        return Counter((owner, tuple(_norm(rect))) for owner, rect in rows)

    for gi in set(book.metal) | set(book.via_pads):
        checkpoint()
        if claims(book.via_pads.get(gi, ())) - claims(book.metal.get(gi, ())):
            return False

    for rows_by_layer, indexes in (
        (book.metal, getattr(engine, '_metal_idx', {})),
        (book.via_pads, getattr(engine, '_via_pad_idx', {})),
    ):
        for gi in set(rows_by_layer) | set(indexes):
            checkpoint()
            actual = indexes.get(gi)
            rows = rows_by_layer.get(gi, ())
            if actual is None:
                if rows:
                    return False
                continue
            if not isinstance(actual, _SpatialIndex):
                return False
            expected = _SpatialIndex(actual.cell)
            for owner, rect in rows:
                checkpoint()
                expected.add(owner, rect)
            if set(actual.buckets) != set(expected.buckets):
                return False
            for bucket in expected.buckets:
                checkpoint()
                if claims(actual.buckets[bucket]) != claims(expected.buckets[bucket]):
                    return False
    return True


def classify_candidate_source_junction(
    engine, net, prior, candidate_segs, candidate_vias, allowed_layers, probe_states,
):
    """Classify and, when possible, repair a candidate source-width junction.

    The caller retains ownership of publication, selected-access validation and
    final full-world checks. Return ``(NO_NEW_SOURCE_WIDTH, None)`` for a
    certified candidate without an added source/route width marker,
    ``(REPAIRED, (segs, vias))`` for a guarded additive repair, or
    ``(UNKNOWN, None)`` when the world or required repair is not established.
    Ordinary candidate spacing remains the caller's existing obligation. Book
    cut envelopes are never emitted as individual cuts.
    """
    import klayout.db as kdb
    from clay.layout.route.engine_gridless import _RoutedNet, _SpatialIndex, _seg_bbox

    if not isinstance(prior, _RoutedNet) or prior.name != net:
        return 'UNKNOWN', None
    if len(probe_states) != 2 or not getattr(engine, '_fixed_deck_literal_rects', ()):
        return 'UNKNOWN', None
    def checkpoint():
        engine._watchdog_check(net, 'candidate_source_junction')

    from clay.layout.route.group_fallback_checkpoint import _canonical
    checkpoint()
    if not _candidate_book_indices_match(engine, checkpoint):
        return 'UNKNOWN', None
    book = engine._book
    book_signature = _canonical((book.metal, book.via_pads, book.cuts, book.persistent), set())
    checkpoint()
    candidate_segs, candidate_vias = tuple(candidate_segs), tuple(candidate_vias)
    combined = _RoutedNet(net, segs=list(prior.segs) + list(candidate_segs),
                          vias=list(prior.vias) + list(candidate_vias))
    baseline_metal = engine._authenticated_owned_book_metal(net, combined)
    if baseline_metal is None:
        return 'UNKNOWN', None
    names = {str(owner) for _layer, owner, _rect in engine._fixed_deck_literal_rects
             if owner is not None}
    for index in (book.metal, book.via_pads):
        names.update(str(owner) for rows in index.values() for owner, _rect in rows
                     if owner is not None)
    names.update(str(owner) for owner, _rect, _lo in book.cuts if owner is not None)
    names.add(str(net))
    if len({name.upper() for name in names}) != len(names):
        return 'UNKNOWN', None

    baseline = getattr(engine, '_active_group_route_union_baseline', None)
    if baseline is None:
        baseline = getattr(engine, '_group_route_landing_baselines', {}).get(net)
    if net in book.persistent:
        if not isinstance(baseline, _RoutedNet) or baseline.name != net:
            return 'UNKNOWN', None
        base_segs, base_vias = tuple(baseline.segs), tuple(baseline.vias)
    else:
        base_segs, base_vias = (), ()

    previous = _RoutedNet(net, segs=list(base_segs) + list(prior.segs),
                          vias=list(base_vias) + list(prior.vias))
    current = _RoutedNet(net, segs=list(previous.segs) + list(candidate_segs),
                         vias=list(previous.vias) + list(candidate_vias))
    before_metal = engine._net_metal(previous)[0]
    current_metal = engine._net_metal(current)[0]
    def world_unchanged():
        checkpoint()
        result = (engine._book is book
                  and _candidate_book_indices_match(engine, checkpoint)
                  and _canonical((book.metal, book.via_pads, book.cuts, book.persistent), set()) == book_signature
                  and engine._authenticated_owned_book_metal(net, combined) == baseline_metal)
        checkpoint()
        return result

    if engine._ordinary_repair_preserves_world(net, before_metal, current_metal):
        return ('NO_NEW_SOURCE_WIDTH', None) if world_unchanged() else ('UNKNOWN', None)

    checkpoint()
    previous_snapshot = engine._deck_negotiation_snapshot({net: previous})
    candidate_snapshot = engine._deck_negotiation_snapshot({net: current})
    before, before_edges = _markers(engine, previous_snapshot, checkpoint)
    candidate_marks, candidate_edges = _markers(engine, candidate_snapshot, checkpoint)

    missing = object()
    saved = {name: getattr(engine, name, missing) for name in
             ('_hard_rejections', '_hard_rejection_samples', '_hard_rejection_generation')}
    engine._hard_rejections = ({} if saved['_hard_rejections'] is missing
                               else dict(saved['_hard_rejections']))
    engine._hard_rejection_samples = ({} if saved['_hard_rejection_samples'] is missing
                                      else dict(saved['_hard_rejection_samples']))
    try:
        identified = set()
        unresolved_touch = False
        permitted = set(int(value) for value in allowed_layers)
        for gi in range(engine._n_layers):
            checkpoint()
            new_markers = candidate_marks[gi]['width'] - before[gi]['width']
            if not new_markers:
                continue
            layer = tuple(engine._tech.gi_to_metal_key(gi))
            source = tuple(tuple(rect) for raw_layer, owner, rect in engine._fixed_deck_literal_rects
                           if owner == net and tuple(raw_layer) == layer)
            if not source:
                continue
            drawn = _rects(engine._emit_net(current), layer)
            own = kdb.Region()
            for index, rect in enumerate((*source, *drawn)):
                if index % 256 == 0:
                    checkpoint()
                own.insert(kdb.Box(*rect))
            own.merge()
            own_pairs = set()
            for index, pair in enumerate(own.width_check(engine._tech.gi_min_width(gi)).each()):
                if index % 256 == 0:
                    checkpoint()
                own_pairs.add(str(pair))
            old_width_polygons = kdb.Region()
            for index, old_pair in enumerate(before_edges[gi]):
                if index % 256 == 0:
                    checkpoint()
                old_width_polygons.insert(old_pair.polygon(0))
            old_width_polygons.merge()
            pairs = []
            for index, pair in enumerate(candidate_edges[gi]):
                if index % 256 == 0:
                    checkpoint()
                if (str(pair) in new_markers and str(pair) in own_pairs
                        and not (kdb.Region(pair.polygon(0)) - old_width_polygons).is_empty()):
                    pairs.append(pair)
            if not pairs:
                continue
            # The established constructive generator only authorizes positive
            # area source/route intersections. A real zero-area edge or corner
            # contact can nevertheless participate in an added width marker.
            # Do not invent a fill from that contact or classify it as clean.
            route_index = _SpatialIndex()
            for index, route_box in enumerate(drawn):
                if index % 256 == 0:
                    checkpoint()
                route_index.add(None, route_box)
            contact_work = 0
            for source_box in source:
                checkpoint()
                if unresolved_touch:
                    break
                for _owner, route_box in route_index.query(source_box):
                    if contact_work % 256 == 0:
                        checkpoint()
                    contact_work += 1
                    x0 = max(source_box[0], route_box[0])
                    y0 = max(source_box[1], route_box[1])
                    x1 = min(source_box[2], route_box[2])
                    y1 = min(source_box[3], route_box[3])
                    if x0 > x1 or y0 > y1 or (x0 < x1 and y0 < y1):
                        continue
                    contact = (x0, y0, x1, y1)
                    for index, pair in enumerate(pairs):
                        if index % 256 == 0:
                            checkpoint()
                        if (_edge_touches(pair.first, contact)
                                or _edge_touches(pair.second, contact)):
                            unresolved_touch = True
                            break
                    if unresolved_touch:
                        break
            width = int(engine._tech.gi_min_width(gi))
            grid = int(engine._manufacturing_grid_nm)
            direct, direct_identified = _junction_candidate_evidence(
                width, grid, pairs, source, drawn, checkpoint=checkpoint,
            )
            identified.update(direct_identified)
            options = {box: set(ids) for box, ids in direct}
            for _ref, overlap in _exact_ref_union_overlaps(
                engine, net, gi, source, drawn, checkpoint,
            ):
                union_options, union_identified = _junction_candidate_evidence(
                    width, grid, pairs, (overlap,), (overlap,), checkpoint=checkpoint,
                )
                identified.update(union_identified)
                for box, ids in union_options:
                    options.setdefault(box, set()).update(ids)
            if not (direct_identified or any(associated for associated in options.values())):
                continue
            if gi not in permitted:
                continue
            if (not candidate_snapshot.report.cross_net_isolated
                    or not engine._candidate_deck_connects_probes(
                        net, current.segs, current.vias, probe_states, include_fixed=True,
                    )):
                return 'UNKNOWN', None
            for box, associated in sorted(options.items()):
                checkpoint()
                overrides = engine._rail_local_half_width_overrides(
                    net, allowed_layers, engine._rail_by_net.get(net, ()),
                )
                with engine._half_width_scope(overrides, attribute='_active_fill_half_width_overrides'):
                    fills = engine._fill_segs(net, gi, box)
                if not fills or any(seg.gi != gi for seg in fills):
                    continue
                boxes = [_seg_bbox(seg.ax, seg.ay, seg.bx, seg.by,
                                   engine._segment_half_width(net, seg)) for seg in fills]
                if not engine._fill_legal(net, gi, boxes, check_own_junctions=False,
                                          current_route=current):
                    continue
                trial = _RoutedNet(net, segs=list(current.segs) + list(fills),
                                   vias=list(current.vias))
                trial_metal = engine._net_metal(trial)[0]
                if (not engine._ordinary_repair_preserves_world(net, current_metal, trial_metal)
                        or not engine._repair_preserves_world(net, current_metal, trial_metal)
                        or not engine._ordinary_repair_preserves_world(net, before_metal, trial_metal)
                        or not engine._repair_preserves_world(net, before_metal, trial_metal)):
                    continue
                snapshot = engine._deck_negotiation_snapshot({net: trial})
                after, _ = _markers(engine, snapshot, checkpoint)
                if (not snapshot.report.cross_net_isolated
                        or not set(snapshot.report.unresolved_probes) <= set(candidate_snapshot.report.unresolved_probes)
                        or not set(snapshot.report.ambiguous_nets) <= set(candidate_snapshot.report.ambiguous_nets)
                        or any(after[g][kind] - before[g][kind] for g in before for kind in before[g])
                        or not (associated & (candidate_marks[gi]['width'] - after[gi]['width']))
                        or not engine._candidate_deck_connects_probes(
                            net, trial.segs, trial.vias, probe_states, include_fixed=True,
                        )):
                    continue
                # A successful fill must still refer to the very same Book
                # and complete current indexes that were authenticated above.
                if not world_unchanged():
                    return 'UNKNOWN', None
                return 'REPAIRED', (list(candidate_segs) + list(fills), list(candidate_vias))
        if identified or unresolved_touch or not world_unchanged():
            return 'UNKNOWN', None
        return 'NO_NEW_SOURCE_WIDTH', None
    finally:
        for name, value in saved.items():
            if value is missing:
                engine.__dict__.pop(name, None)
            else:
                setattr(engine, name, value)


def repair_group_edge_union(engine, net, prior, candidate_segs, candidate_vias,
                            allowed_layers, probe_states, selected_states):
    """Try one authenticated, additive group-edge repair without publishing it.

    The caller provides the base cut ledger and retains all routing state. A
    return of ``None`` is unresolved, not proof that the access pair is illegal.
    """
    from clay.layout.route.engine_gridless import _RoutedNet
    from clay.layout.route.group_fallback_checkpoint import _canonical

    if not isinstance(prior, _RoutedNet) or prior.name != net:
        return None
    book = getattr(engine, '_book', None)
    baseline = getattr(engine, '_active_group_route_union_baseline', None)
    if book is None or not isinstance(baseline, _RoutedNet) or baseline.name != net or net not in book.persistent:
        return None
    checkpoint = lambda: engine._watchdog_check(net, 'group_edge_source_union')
    if not _candidate_book_indices_match(engine, checkpoint):
        return None
    candidate_segs, candidate_vias = tuple(candidate_segs), tuple(candidate_vias)
    current = _RoutedNet(net, segs=list(prior.segs) + list(candidate_segs),
                         vias=list(prior.vias) + list(candidate_vias))
    if engine._authenticated_owned_book_metal(net, current) is None:
        return None
    original_book = _canonical((book.metal, book.via_pads, book.cuts, book.persistent), set())
    original_prior = _canonical(prior, set())
    original_baseline = _canonical(baseline, set())
    original_candidate = _canonical((candidate_segs, candidate_vias), set())
    previous = _RoutedNet(net, segs=list(baseline.segs) + list(prior.segs),
                          vias=list(baseline.vias) + list(prior.vias))
    before = engine._net_metal(previous)[0]
    original = _RoutedNet(net, segs=list(previous.segs) + list(candidate_segs),
                          vias=list(previous.vias) + list(candidate_vias))
    original_metal = engine._net_metal(original)[0]
    # Preserve the fast path when the complete source-owned union is already
    # ordinary-clean. The existing caller retains its other physical checks.
    if engine._ordinary_repair_preserves_world(net, before, original_metal):
        return list(candidate_segs), list(candidate_vias)
    repaired = engine._resolve_candidate_same_net_fills(
        net, previous, candidate_segs, candidate_vias, mutable_indices=set())
    gap_segs, gap_vias = repaired if repaired is not None else (list(candidate_segs), list(candidate_vias))
    if (len(gap_segs) < len(candidate_segs) or len(gap_vias) != len(candidate_vias)
            or any(a is not b for a, b in zip(gap_segs, candidate_segs))
            or any(a is not b for a, b in zip(gap_vias, candidate_vias))):
        return None
    status, repair = classify_candidate_source_junction(
        engine, net, prior, gap_segs, gap_vias, allowed_layers, probe_states)
    if status == 'REPAIRED':
        final_segs, final_vias = repair
    elif status == 'NO_NEW_SOURCE_WIDTH':
        final_segs, final_vias = gap_segs, gap_vias
    else:
        return None
    if (len(final_segs) < len(candidate_segs) or len(final_vias) != len(candidate_vias)
            or any(a is not b for a, b in zip(final_segs, candidate_segs))
            or any(a is not b for a, b in zip(final_vias, candidate_vias))):
        return None
    final = _RoutedNet(net, segs=list(previous.segs) + list(final_segs),
                       vias=list(previous.vias) + list(final_vias))
    final_metal = engine._net_metal(final)[0]
    if (not engine._ordinary_repair_preserves_world(net, before, final_metal)
            or not engine._repair_preserves_world(net, before, final_metal)
            or not engine._ordinary_repair_preserves_world(net, original_metal, final_metal)
            or not engine._repair_preserves_world(net, original_metal, final_metal)):
        return None
    for via in final.vias:
        engine._require_policy_via(via)
    old_snapshot = engine._deck_negotiation_snapshot({net: original}).report
    final_snapshot = engine._deck_negotiation_snapshot({net: final}).report
    if (not final_snapshot.cross_net_isolated
            or not set(final_snapshot.unresolved_probes) <= set(old_snapshot.unresolved_probes)
            or not set(final_snapshot.ambiguous_nets) <= set(old_snapshot.ambiguous_nets)
            or not engine._candidate_deck_connects_probes(net, final.segs, final.vias, probe_states, include_fixed=True)
            or not engine._candidate_deck_connects_probes(net, final.segs, final.vias, selected_states, include_fixed=True)):
        return None
    if (engine._book is not book or getattr(engine, '_active_group_route_union_baseline', None) is not baseline
            or _canonical((book.metal, book.via_pads, book.cuts, book.persistent), set()) != original_book
            or _canonical(prior, set()) != original_prior or _canonical(baseline, set()) != original_baseline
            or _canonical((candidate_segs, candidate_vias), set()) != original_candidate
            or not _candidate_book_indices_match(engine, checkpoint)
            or engine._authenticated_owned_book_metal(net, current) is None):
        return None
    return list(final_segs), list(final_vias)


def repair_candidate_source_junction(
    engine, net, prior, candidate_segs, candidate_vias, allowed_layers, probe_states,
):
    """Compatibility wrapper returning only an established additive repair."""
    status, repaired = classify_candidate_source_junction(
        engine, net, prior, candidate_segs, candidate_vias, allowed_layers, probe_states,
    )
    return repaired if status == 'REPAIRED' else None


def repair_source_junctions(engine, routes, allowed_layers_by_net, on_commit):
    """Try only legal additive square fills, preserving existing world issues.

    ``on_commit(net, candidate)`` must publish that route into the real world
    and rebuild indices before returning. It must not change cut geometry or
    drop any other net. The existing caller still performs its full intent,
    exact pin and signoff gates. Timeout and policy are inherited unchanged.
    """
    from clay.layout.route.engine_gridless import _RoutedNet, _seg_bbox
    if not getattr(engine, '_fixed_deck_literal_rects', ()) or not routes:
        return ()
    import klayout.db as kdb
    owners=Counter(str(n).upper() for n in routes)
    fixed_owners={str(owner) for _layer,owner,_rect in engine._fixed_deck_literal_rects
                  if owner is not None}
    all_names=fixed_owners|{str(net) for net in routes}
    if (len({name.upper() for name in all_names}) != len(all_names)
            or any(route.name != net for net,route in routes.items())):
        return ()
    visited=set(); accepted=[]
    first=next(iter(sorted(routes)))
    def checkpoint(): engine._watchdog_check(first,'source_junction_world')
    while True:
        checkpoint()
        baseline=engine._deck_negotiation_snapshot(routes)
        checkpoint()
        if not baseline.report.cross_net_isolated:
            break
        before,edges=_markers(engine,baseline,checkpoint)
        found=False
        for net in sorted(routes):
            if (owners[str(net).upper()] != 1 or routes[net].name != net
                    or any(owner != net and owner.upper() == str(net).upper()
                           for owner in fixed_owners)):
                continue
            route=routes[net]
            for gi in sorted(set(allowed_layers_by_net.get(net, ()))):
                gi=int(gi)
                if not 0 <= gi < engine._n_layers or not edges[gi]:
                    continue
                engine._watchdog_check(net,'source_junction_candidates')
                layer=tuple(engine._tech.gi_to_metal_key(gi))
                source=tuple(tuple(rect) for raw_layer,owner,rect in engine._fixed_deck_literal_rects
                             if owner == net
                             and tuple(raw_layer)==layer)
                if not source: continue
                drawn=_rects(engine._emit_net(route),layer)
                width=int(engine._tech.gi_min_width(gi))
                own=kdb.Region()
                for index,rect in enumerate((*source,*drawn)):
                    if index % 256 == 0:
                        engine._watchdog_check(net,'source_junction_owned_markers')
                    own.insert(kdb.Box(*rect))
                own.merge()
                owned={str(pair) for pair in own.width_check(width).each()}
                engine._watchdog_check(net,'source_junction_owned_markers')
                pairs=tuple(pair for pair in edges[gi] if str(pair) in owned)
                if not pairs: continue
                options=generate_junction_candidates(
                    width,int(engine._manufacturing_grid_nm),pairs,source,drawn,
                    checkpoint=lambda: engine._watchdog_check(net,'source_junction_candidates'))
                # Preserve source-declared terminal references while joining
                # split exact artwork. Decompose the actual intersection, never
                # replace a nonrectangular union or a hole by its bbox.
                combined={box:set(ids) for box,ids in options}
                for _ref,overlap in _exact_ref_union_overlaps(
                    engine,net,gi,source,drawn,
                    lambda:engine._watchdog_check(net,'source_junction_union')):
                    for box,ids in generate_junction_candidates(
                        width,int(engine._manufacturing_grid_nm),pairs,(overlap,),(overlap,),
                        checkpoint=lambda:engine._watchdog_check(net,'source_junction_candidates')):
                        combined.setdefault(box,set()).update(ids)
                options=tuple((box,tuple(sorted(ids))) for box,ids in sorted(combined.items()))
                for box,associated in options:
                    key=(net,gi,box)
                    if key in visited: continue
                    visited.add(key)
                    engine._watchdog_check(net,'source_junction_trial')
                    overrides=engine._rail_local_half_width_overrides(
                        net,allowed_layers_by_net.get(net,()),engine._rail_by_net.get(net,()))
                    with engine._half_width_scope(overrides,attribute='_active_fill_half_width_overrides'):
                        segs=engine._fill_segs(net,gi,box)
                    engine._watchdog_check(net,'source_junction_fill')
                    if not segs or any(s.gi != gi for s in segs): continue
                    boxes=[_seg_bbox(s.ax,s.ay,s.bx,s.by,engine._segment_half_width(net,s)) for s in segs]
                    legal=engine._fill_legal(net,gi,boxes,check_own_junctions=False,current_route=route)
                    engine._watchdog_check(net,'source_junction_fill_legal')
                    if not legal: continue
                    trial=_RoutedNet(name=route.name,segs=list(route.segs)+segs,
                                     vias=list(route.vias),connected=route.connected)
                    oldmetal,_=engine._net_metal(route); newmetal,_=engine._net_metal(trial)
                    preserved=engine._repair_preserves_world(net,oldmetal,newmetal)
                    engine._watchdog_check(net,'source_junction_world_guard')
                    if preserved:
                        preserved=engine._ordinary_repair_preserves_world(net,oldmetal,newmetal)
                    engine._watchdog_check(net,'source_junction_guards')
                    if not preserved: continue
                    world=dict(routes); world[net]=trial
                    snapshot=engine._deck_negotiation_snapshot(world)
                    engine._watchdog_check(net,'source_junction_deck')
                    after,_=_markers(engine,snapshot,checkpoint)
                    if (not snapshot.report.cross_net_isolated
                        or not set(snapshot.report.unresolved_probes) <= set(baseline.report.unresolved_probes)
                        or not set(snapshot.report.ambiguous_nets) <= set(baseline.report.ambiguous_nets)
                        or any(after[g][kind]-before[g][kind] for g in before for kind in before[g])
                        or not (set(associated) & (before[gi]['width']-after[gi]['width']))):
                        continue
                    # The callback publishes to the caller's real book and routes.
                    engine._watchdog_check(net,'source_junction_publish')
                    on_commit(net,trial)
                    if routes.get(net) is not trial:
                        raise ValueError('commit callback did not publish candidate route')
                    accepted.append(key);found=True
                    break
                if found: break
            if found: break
        if not found: break
    return tuple(accepted)
