"""Source-bound N-well spacing inside native matched-device groups.

Only split, independently materialized PMOS matching members participate. A
shared diffusion row or an explicit abut remains under its existing owner and
physical replay; sharing a bulk net alone does not authorize merging wells.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class NativeMatchingWellPair:
    names: tuple[str, str]
    layer: tuple[int, int]
    spacing_nm: int
    member_gap_nm: int
    well_shapes: tuple[tuple, tuple]


def _ring_signature(vertices, origin):
    points = tuple((x - origin[0], y - origin[1]) for x, y in vertices)
    if len(points) < 3 or any(type(v) is not int for point in points for v in point):
        raise ValueError('native N-well polygon is malformed')
    if points[0] == points[-1]:
        points = points[:-1]
    if len(points) < 3:
        raise ValueError('native N-well polygon is malformed')
    backward = tuple(reversed(points))
    return min(*(points[i:] + points[:i] for i in range(len(points))),
               *(backward[i:] + backward[:i] for i in range(len(backward))))


def _device_well_signature(pd, layer):
    origin = pd.bbox[:2]
    shapes = tuple(sorted(_ring_signature(tuple((v.x, v.y) for v in lp.polygon.vertices), origin)
                          for lp in pd.polys or () if (lp.layer, lp.datatype) == layer))
    if not shapes:
        raise ValueError('native matching PMOS lacks N-well artwork')
    return shapes


def _box(pd, layer):
    xs, ys = [], []
    for lp in pd.polys or ():
        if (lp.layer, lp.datatype) == layer:
            for v in lp.polygon.vertices:
                if type(v.x) is not int or type(v.y) is not int:
                    raise ValueError('native well vertices must be exact integer nm')
                xs.append(v.x); ys.append(v.y)
    if not xs:
        raise ValueError('native matching PMOS lacks N-well artwork')
    box = min(xs), min(ys), max(xs), max(ys)
    if box[0] >= box[2] or box[1] >= box[3]:
        raise ValueError('native N-well artwork is degenerate')
    return box


def select_native_matching_well_pairs(base, cell, intent, tech):
    bundle = getattr(base, 'native_geometry', None)
    if bundle is None:
        return ()
    rows = getattr(base, 'row_program', None)
    row_members = {n for e in getattr(rows, 'entries', ()) for n in e.consumed_devices}
    abut = {frozenset((a, b)) for a, b in getattr(intent, 'abut_pairs', ())}
    axes = getattr(base, 'matching_pair_axes', {}) or {}
    pairs = []
    for raw in axes:
        if type(raw) is not tuple or len(raw) != 2 or raw[0] == raw[1]:
            continue
        a, b = tuple(sorted(raw))
        if (a in row_members or b in row_members or frozenset((a, b)) in abut
                or a not in base.devices or b not in base.devices):
            continue
        left, right = base.devices[a], base.devices[b]
        if left.kind != 'pmos' or right.kind != 'pmos':
            continue
        if left.geometry_owner or right.geometry_owner:
            raise ValueError('native matching PMOS has unexpected geometry owner')
        pairs.append((a, b))
    if not pairs:
        return ()
    from clay.layout.tech import LayoutTech
    bundle.require_current(cell)
    bundle._require_individual_placements(base, cell, {n for p in pairs for n in p})
    if not isinstance(tech, LayoutTech) or tech.pdk_id != 'sky130':
        raise ValueError('native matching PMOS requires the bound SKY130 PDK authority')
    canonical_tech = LayoutTech.resolve('sky130')
    # The native bundle validates the source models and copied PDK producer;
    # the placement technology must expose the same registered layer/rule.
    # Accessing rules also invokes the adapter's authority validator.
    _ = tech.rules, canonical_tech.rules
    layer = tech.build_layer('nwell')
    if type(layer) is not tuple or len(layer) != 2 or any(type(v) is not int for v in layer):
        raise ValueError('native matching PMOS requires a bound N-well layer')
    spacing = tech.min_space('nwell')
    if type(spacing) is not int or spacing <= 0:
        raise ValueError('native matching PMOS requires a positive N-well rule')
    if layer != canonical_tech.build_layer('nwell') or spacing != canonical_tech.min_space('nwell'):
        raise ValueError('native matching PMOS N-well rule differs from bound PDK authority')
    out = []
    for a, b in sorted(set(pairs)):
        ap, bp = base.devices[a], base.devices[b]
        aw, bw = _box(ap, layer), _box(bp, layer)
        for pd, well in ((ap, aw), (bp, bw)):
            x0, y0, x1, y1 = pd.bbox
            if not (x0 <= well[0] < well[2] <= x1 and y0 <= well[1] < well[3] <= y1):
                raise ValueError('native N-well lies outside its source footprint')
        # Use the more restrictive of either possible left/right ordering.
        margins = min(ap.bbox[2] - aw[2] + bw[0] - bp.bbox[0],
                      bp.bbox[2] - bw[2] + aw[0] - ap.bbox[0])
        out.append(NativeMatchingWellPair((a, b), layer, spacing, max(0, spacing - margins),
            (_device_well_signature(ap, layer), _device_well_signature(bp, layer))))
    return tuple(out)


def verify_native_matching_group_wells(pairs, devices, result):
    owners = dict(getattr(result, 'matching_pair_owners', {}) or {})
    for pair in pairs:
        owner = owners.get(pair.names) or owners.get(tuple(reversed(pair.names)))
        if not owner or not str(owner[1]).startswith('G_'):
            continue
        for name, expected in zip(pair.names, pair.well_shapes):
            if _device_well_signature(devices[name], pair.layer) != expected:
                raise ValueError('native matching PMOS N-well artwork changed')
        boxes = sorted((_box(devices[n], pair.layer) for n in pair.names))
        if boxes[1][0] - boxes[0][2] < pair.spacing_nm:
            raise ValueError('native matching PMOS N-well spacing is below PDK rule')


def verify_native_matching_emitted_wells(pairs, geometry, positions):
    from clay.layout.owned_artwork import _stamp_records
    for pair in pairs:
        boxes = []
        for name, expected in zip(pair.names, pair.well_shapes):
            if name not in geometry or name not in positions:
                raise ValueError('native matching PMOS source stamp is missing')
            records = _stamp_records(geometry[name], positions[name])
            points = [p for layer, dt, ring in records if (layer, dt) == pair.layer for p in ring]
            if not points:
                raise ValueError('native matching PMOS emitted well is missing')
            all_points = [p for _layer, _dt, ring in records for p in ring]
            origin = (min(p[0] for p in all_points), min(p[1] for p in all_points))
            actual = tuple(sorted(_ring_signature(ring, origin)
                                  for layer, dt, ring in records if (layer, dt) == pair.layer))
            if actual != expected:
                raise ValueError('native matching PMOS emitted N-well artwork changed')
            xs, ys = zip(*points)
            boxes.append((min(xs), min(ys), max(xs), max(ys)))
        a, b = boxes
        dx = max(0, a[0] - b[2], b[0] - a[2])
        dy = max(0, a[1] - b[3], b[1] - a[3])
        # Bounding boxes conservatively enclose actual polygons. Thus their
        # Euclidean separation is a lower bound on the real mask separation.
        if dx * dx + dy * dy < pair.spacing_nm * pair.spacing_nm:
            raise ValueError('native matching PMOS emitted N-well spacing is below PDK rule')
