"""Run a source-policy router on a separately supplied single-layer scene.

This local experiment does not materialize, route, or sign off the source cell.
The witness is measured separately and is never inserted into the routing world.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time


@dataclass(frozen=True)
class LocalRoutingSceneRequest:
    path: Path


def _region(rectangles):
    import klayout.db as db
    region = db.Region()
    for row in rectangles:
        if (len(row) != 4 or any(type(v) is not int for v in row)
                or row[0] >= row[2] or row[1] >= row[3]):
            raise ValueError('scene rectangle must be a non-empty integer-nm box')
        region.insert(db.Box(*row))
    return region.merged()


def _measure(region, obstacles, pins, width_nm, spacing_nm):
    import klayout.db as db
    merged = region.merged()
    connected = any(all(poly.inside(db.Point(*point)) for point in pins)
                    for poly in merged.each())
    return {
        'connected': connected,
        'cross_overlap_area_nm2': (merged & obstacles).area(),
        'cross_spacing_violations': merged.separation_check(obstacles, spacing_nm).count(),
        'width_violations': merged.width_check(width_nm).count(),
    }


def _between_nets(regions, spacing_nm):
    names = sorted(regions)
    return [{'nets': [a, b], 'overlap_area_nm2': (regions[a] & regions[b]).area(),
             'spacing_violations': regions[a].separation_check(regions[b], spacing_nm).count()}
            for i, a in enumerate(names) for b in names[i + 1:]]


def run_local_routing_scene(request, *, source_path, policy, pdk_id, work_dir, source_intent=None):
    import klayout.db as db
    from clay.layout.route.canonical_input_capture import route_engine_configuration
    from clay.layout.route.constraints import LayerSet, Width
    from clay.layout.route.engine import Obstacle, Pin, RoutedCell
    from clay.layout.route.engine_gridless import GridlessEngine
    from clay.layout.route.engine_registry import route_engine_from_policy
    from clay.layout.route.emit_gds import emit_routed_gds
    from clay.layout.route.metal_min_spacing_check import check_metal_min_spacing
    from clay.layout.tech import LayoutTech

    path = Path(request.path)
    raw = path.read_bytes()
    scene = json.loads(raw)
    schema = scene.get('schema')
    if schema not in ('clay.local-routing-scene/v1', 'clay.local-routing-scene/v2') or scene.get('pdk') != pdk_id:
        raise ValueError('local routing scene schema or PDK differs')
    source_bytes = Path(source_path).read_bytes()
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    declared_source = scene.get('provenance', {}).get('source')
    if declared_source is not None and declared_source.get('sha256') != source_sha:
        raise ValueError('local scene refers to a different source')
    layer = tuple(scene['layer'])
    bounds = tuple(scene['bounds_nm'])
    rows = [scene] if schema.endswith('/v1') else scene['nets']
    if type(rows) is not list or not rows or len(layer) != 2 or any(type(v) is not int for v in layer):
        raise ValueError('invalid local routing scene pins, layer, net or width')
    nets = {}
    for row in rows:
        net, width = row['net'], row['width_nm']
        pins = tuple(tuple(p) for p in row['pins_nm'])
        if (type(net) is not str or not net or net in nets or type(width) is not int or width <= 0
                or len(pins) < 2 or any(len(p) != 2 or any(type(v) is not int for v in p) for p in pins)):
            raise ValueError('invalid local routing scene pins, layer, net or width')
        nets[net] = {'width': width, 'pins': pins,
                     'witness': _region(row.get('witness_rectangles_nm', []))}
    from clay.layout.route.floorplan_containment import RoutingEnvelope
    RoutingEnvelope.coerce(bounds)
    tech = LayoutTech.resolve(pdk_id)
    layer_n = next((n for n in range(1, 7) if tuple(tech.metal_key(n)) == layer), None)
    if layer_n is None:
        raise ValueError('local scene layer is not a PDK routing metal')
    spacing = tech.min_space(f'metal{layer_n}')
    if type(spacing) is not int or spacing <= 0:
        raise ValueError('PDK metal spacing is unavailable')
    obstacles_region = _region(scene['obstacles_nm'])
    pins_by_net = {net: [Pin(net=net, device='__local_scene__', terminal=str(i),
                             xy_um=(x / 1000, y / 1000), xy_nm=(x, y), layer=layer)
                         for i, (x, y) in enumerate(value['pins'])]
                   for net, value in nets.items()}
    obstacles = [Obstacle(*(v / 1000 for v in row), layer=layer)
                 for row in scene['obstacles_nm']]
    engine = route_engine_from_policy(policy)
    if type(engine) is not GridlessEngine:
        raise ValueError('local routing scene needs source-selected Gridless')
    configuration = route_engine_configuration(engine)
    constraints = [item for net, value in nets.items()
                   for item in (LayerSet(net, frozenset((layer,))), Width(net, value['width'] / 1000))]
    source_trunks = []
    direction_weight = 0.0
    if schema.endswith('/v2') and source_intent is not None:
        from clay.layout.route.intent import RoutingIntent
        from clay.layout.route.lower import conventional_axes_for, lower_intent
        from clay.layout.route.route_cell import _HVH_DIRECTION_WEIGHT, _trunk_layer_preferences
        preferences = _trunk_layer_preferences(source_intent, tech)
        selected_trunks = {net: pref for net, pref in preferences.items() if net in nets}
        direction_weight = _HVH_DIRECTION_WEIGHT if getattr(source_intent, 'trunks', ()) else 0.0
        constraints.extend(lower_intent(RoutingIntent(trunk_layers=selected_trunks),
                           direction_weight=direction_weight, axes=conventional_axes_for(tech)))
        source_trunks = [{'net':net, 'layer':pref.value.layer, 'orient':pref.value.orient}
                         for net, pref in sorted(selected_trunks.items())]
    started = time.perf_counter()
    routed = engine.route(pins_by_net, obstacles, constraints, tech, routing_envelope_nm=bounds)
    elapsed = time.perf_counter() - started
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    route_path = work_dir / 'local-route.gds'
    emit_routed_gds(routed, str(route_path))
    layout = db.Layout()
    layout.read(str(route_path))
    top = layout.top_cell()
    index = layout.layer(*layer)
    regions = {}
    per_net = {}
    witness_per_net = {}
    for i, (net, value) in enumerate(sorted(nets.items())):
        target = routed.nets.get(net)
        net_path = work_dir / f'local-net-{i:04d}.gds'
        emit_routed_gds(RoutedCell(nets={} if target is None else {net:target}), str(net_path))
        net_layout = db.Layout(); net_layout.read(str(net_path))
        net_layer = net_layout.find_layer(*layer)
        # Materialize polygons before releasing this net's Layout. A Region
        # constructed directly from its shape iterator otherwise retains a lazy
        # view of that Layout and can lose its geometry on the next iteration.
        net_region = db.Region()
        if net_layer is not None:
            for polygon in db.Region(net_layout.top_cell().begin_shapes_rec(net_layer)).each():
                net_region.insert(polygon)
        regions[net] = net_region.merged()
        per_net[net] = _measure(regions[net], obstacles_region, value['pins'], value['width'], spacing)
        per_net[net]['gds_path'] = str(net_path)
        witness_per_net[net] = _measure(value['witness'], obstacles_region, value['pins'], value['width'], spacing)
    between = _between_nets(regions, spacing)
    witness_between = _between_nets({net:value['witness'] for net, value in nets.items()}, spacing)
    physical = {'connected': all(row['connected'] for row in per_net.values()),
                'cross_overlap_area_nm2': sum(row['cross_overlap_area_nm2'] for row in per_net.values())
                                          + sum(row['overlap_area_nm2'] for row in between),
                'cross_spacing_violations': sum(row['cross_spacing_violations'] for row in per_net.values())
                                            + sum(row['spacing_violations'] for row in between),
                'width_violations': sum(row['width_violations'] for row in per_net.values())}
    net_union = db.Region()
    for region in regions.values():
        net_union += region
    combined_route = db.Region(top.begin_shapes_rec(index)).merged()
    physical['emission_union_xor_area_nm2'] = (combined_route ^ net_union.merged()).area()
    top.shapes(index).insert(obstacles_region)
    combined_path = work_dir / 'local-route-and-background.gds'
    layout.write(str(combined_path))
    independent = check_metal_min_spacing(str(combined_path), tech)
    physical.update(gds_path=str(combined_path), metal_min_spacing_status=independent.status,
                    metal_min_spacing_count=independent.count)
    search_nets = {net: {'connected':bool(routed.nets.get(net) and routed.nets[net].connected),
                         'wires':len(routed.nets[net].wires) if net in routed.nets else 0}
                   for net in nets}
    result = {
        'scope': 'local_single_layer_routing_diagnostic', 'physical_admission': False,
        'scene_path': str(path), 'scene_sha256': hashlib.sha256(raw).hexdigest(),
        'source_path': str(source_path),
        'source_sha256': source_sha,
        'selected_engine': configuration, 'pdk': pdk_id, 'layer': layer,
        'search': {'connected': all(item['connected'] for item in search_nets.values()),
                   'wires': sum(item['wires'] for item in search_nets.values()),
                   'elapsed_seconds': elapsed,
                   'success': bool(routed.success), 'shorts': routed.shorts,
                   'unsatisfied': [type(item).__name__ for item in routed.unsatisfied]},
        'physical': physical,
        'witness': next(iter(witness_per_net.values())) if schema.endswith('/v1') else
                   {'nets': witness_per_net, 'between_nets': witness_between},
        'provenance': scene.get('provenance', {}),
    }
    if schema.endswith('/v2'):
        result['search']['nets'] = search_nets
        physical['nets'] = per_net
        physical['between_nets'] = between
        result['source_constraints'] = {
            'consumed_trunks': source_trunks,
            'direction_weight': direction_weight,
            'local_constraints': 'single-layer restriction and per-net scene width',
            'not_evaluated': ['full-source endpoint inventory and other nets', 'source rails and placement',
                              'other source routing and matching constraints', 'other routing layers'],
        }
    return result
