"""Fresh GDS layer-region observations; no logical ownership or quality verdict.

PDK identity, snapshot, and layer semantics are caller declarations. The host must
authenticate those joins. Geometry uses KLayout's integer layer-region semantics;
DBU conversion is rational and hierarchy transforms must preserve the native grid.
Odd-width GDS square caps, round paths, and non-Manhattan paths fail closed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import json
from math import gcd, isfinite, lcm
from pathlib import Path
import re


Rational = tuple[int, int]
RationalBbox = tuple[Rational, Rational, Rational, Rational]
_COORD_LIMIT = (1 << 31) - 1
_MODULE_PATH = Path(__file__).resolve()
_LOADED_SHA256 = hashlib.sha256(_MODULE_PATH.read_bytes()).hexdigest()


class ReferenceGeometryObservationError(ValueError):
    """Invalid declaration, unsupported geometry, or changing input evidence."""


@dataclass(frozen=True, slots=True)
class ReferenceLayerGeometry:
    layer: int
    datatype: int
    semantic_tokens: tuple[str, ...]
    semantic_status: str
    physical_shape_count: int
    text_shape_count: int
    area_nm2: Rational
    bbox_nm: RationalBbox | None


@dataclass(frozen=True, slots=True)
class ReferenceGeometryObservation:
    gds_sha256: str
    top_cell: str
    cell_count: int
    dbu_um: Rational
    region_lattice_nm: Rational
    reference_pdk_id: str
    pdk_source_snapshot_sha256: str
    pdk_layermap_sha256: str
    declared_layermap: tuple[tuple[str, int, int], ...]
    implementation_sha256: tuple[tuple[str, str], ...]
    layers: tuple[ReferenceLayerGeometry, ...]
    bbox_nm: RationalBbox | None
    schema: str = "clay.reference-geometry-observation/v1"
    geometry_substrate: str = "gds_layer_regions"
    pdk_provenance: str = "caller_declared"

    def canonical_data(self) -> dict[str, object]:
        return json.loads(json.dumps(asdict(self), sort_keys=True))


@dataclass(frozen=True, slots=True)
class ReferenceLayerDifference:
    layer: int
    datatype: int
    xor_area_nm2: Rational
    equivalent: bool


@dataclass(frozen=True, slots=True)
class ReferenceLayerEquivalenceObservation:
    geometry: ReferenceGeometryObservation
    auxiliary: ReferenceGeometryObservation
    comparison_lattice_nm: Rational
    layers: tuple[ReferenceLayerDifference, ...]
    equivalent: bool
    schema: str = "clay.reference-layer-equivalence/v1"

    def canonical_data(self) -> dict[str, object]:
        return json.loads(json.dumps(asdict(self), sort_keys=True))


def _rational(value: Fraction) -> Rational:
    return value.numerator, value.denominator


def _declarations(reference_pdk_id, pdk_source_snapshot_sha256, pdk_layermap):
    if (type(reference_pdk_id) is not str or not reference_pdk_id
            or reference_pdk_id != reference_pdk_id.strip()):
        raise ReferenceGeometryObservationError("PDK_ID: expected a nonempty exact identifier")
    if (type(pdk_source_snapshot_sha256) is not str
            or re.fullmatch(r"[0-9a-f]{64}", pdk_source_snapshot_sha256) is None):
        raise ReferenceGeometryObservationError("PDK_SNAPSHOT: expected lowercase SHA-256")
    if type(pdk_layermap) is not dict or not pdk_layermap:
        raise ReferenceGeometryObservationError("LAYERMAP: expected a nonempty exact dict")
    rows = []
    for token, pair in pdk_layermap.items():
        if (type(token) is not str or not token or token != token.strip()
                or type(pair) is not tuple or len(pair) != 2
                or any(type(item) is not int or item < 0 for item in pair)):
            raise ReferenceGeometryObservationError("LAYERMAP: expected token -> (layer, datatype)")
        rows.append((token, *pair))
    rows = tuple(sorted(rows))
    digest = hashlib.sha256(json.dumps(rows, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()
    return reference_pdk_id, pdk_source_snapshot_sha256, digest, rows


def _capture(paths):
    sources = tuple(Path(path).resolve() for path in paths)
    hashes = tuple(hashlib.sha256(path.read_bytes()).hexdigest() for path in sources)
    implementation = hashlib.sha256(_MODULE_PATH.read_bytes()).hexdigest()
    if implementation != _LOADED_SHA256:
        raise ReferenceGeometryObservationError("IMPLEMENTATION_DRIFT: reload the observer")
    return sources, hashes, implementation


def _guard(sources, hashes, implementation):
    for path, expected in zip(sources, hashes):
        try:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            raise ReferenceGeometryObservationError("INPUT_DRIFT: source unavailable after observation") from exc
        if actual != expected:
            raise ReferenceGeometryObservationError("INPUT_DRIFT: source bytes changed during observation")
    if hashlib.sha256(_MODULE_PATH.read_bytes()).hexdigest() != implementation:
        raise ReferenceGeometryObservationError("IMPLEMENTATION_DRIFT: observer bytes changed")


def _validate_raw_paths(path):
    # GDS PATHTYPE=2 implies half-width end extensions. KLayout's integer
    # Path cannot retain those when width is odd, so reject before parsing.
    data = path.read_bytes()
    offset, in_path, path_type, width = 0, False, 0, 0
    while offset < len(data):
        size = int.from_bytes(data[offset:offset + 2], "big")
        if size < 4 or size % 2 or offset + size > len(data):
            raise ReferenceGeometryObservationError("GDS_RECORD: malformed raw record framing")
        kind = data[offset + 2]
        if kind == 0x04:  # ENDLIB; standard file padding may follow.
            return
        if kind == 0x09:  # PATH
            in_path, path_type, width = True, 0, 0
        elif in_path and kind in (0x21, 0x0F):  # PATHTYPE, WIDTH
            expected_size = 6 if kind == 0x21 else 8
            if size != expected_size:
                raise ReferenceGeometryObservationError("GDS_RECORD: malformed path scalar")
            value = int.from_bytes(data[offset + 4:offset + size], "big", signed=True)
            if kind == 0x21:
                path_type = value
            else:
                width = value
        elif in_path and kind == 0x11:  # ENDEL
            if path_type == 2 and width % 2:
                raise ReferenceGeometryObservationError("PATH: odd-width square caps require unsupported half-grid end extensions")
            in_path = False
        offset += size


def _read_layout(path, expected_top_cell):
    import klayout.db as kdb
    if type(expected_top_cell) is not str or not expected_top_cell:
        raise ReferenceGeometryObservationError("TOP_CELL: expected an explicit cell name")
    _validate_raw_paths(path)
    layout = kdb.Layout()
    layout.read(str(path))
    top = layout.cell(expected_top_cell)
    if top is None or top.cell_index() not in {cell.cell_index() for cell in layout.top_cells()}:
        raise ReferenceGeometryObservationError("TOP_CELL: declared top is missing or is not a top cell")
    return layout, top


def _coordinate(value):
    if not -_COORD_LIMIT <= value <= _COORD_LIMIT:
        raise ReferenceGeometryObservationError("OVERFLOW: coordinate exceeds the exact integer region range")
    return value


def _map_polygon(polygon, mapping):
    import klayout.db as kdb
    def points(sequence):
        return [kdb.Point(_coordinate(x), _coordinate(y))
                for x, y in (mapping(point.x, point.y) for point in sequence)]
    result = kdb.Polygon(points(polygon.each_point_hull()))
    for index in range(polygon.holes()):
        result.insert_hole(points(polygon.each_point_hole(index)))
    return result


def _native_matrix(transform):
    angle = transform.angle
    if transform.mag != 1.0 or angle not in (0.0, 90.0, 180.0, 270.0):
        raise ReferenceGeometryObservationError("TRANSFORM: hierarchy must preserve the integer grid")
    a, b, c, d = ((1, 0, 0, 1), (0, -1, 1, 0),
                  (-1, 0, 0, -1), (0, 1, -1, 0))[int(angle) // 90]
    if transform.is_mirror():
        b, d = -b, -d
    return a, b, c, d, transform.disp.x, transform.disp.y


def _compose(parent, child, offset):
    a, b, c, d, x, y = parent
    e, f, g, h, u, v = child
    u, v = u + offset[0], v + offset[1]
    return (a * e + b * g, a * f + b * h, c * e + d * g, c * f + d * h,
            _coordinate(a * u + b * v + x), _coordinate(c * u + d * v + y))


def _cell_occurrences(top):
    # Read only each native instance transform; never compose in KLayout integers.
    stack = [(top, (1, 0, 0, 1, 0, 0), frozenset())]
    result = []
    while stack:
        cell, matrix, ancestors = stack.pop()
        if cell.cell_index() in ancestors:
            raise ReferenceGeometryObservationError("HIERARCHY: cyclic cell reference")
        ancestors = ancestors | {cell.cell_index()}
        result.append((cell, matrix))
        for instance in cell.each_inst():
            child = _native_matrix(instance.cplx_trans)
            for i in range(max(instance.na, 1)):
                for j in range(max(instance.nb, 1)):
                    offset = (i * instance.a.x + j * instance.b.x,
                              i * instance.a.y + j * instance.b.y)
                    stack.append((instance.cell, _compose(matrix, child, offset), ancestors))
    return tuple(result)


def _region_scale(occurrences, layer_indices):
    visited = set()
    for cell, _ in occurrences:
        if cell.cell_index() in visited:
            continue
        visited.add(cell.cell_index())
        for index in layer_indices:
            for shape in cell.shapes(index).each():
                if shape.is_path() and shape.path.width % 2:
                    return 2
    return 1


def _path_polygons(path, scale):
    """Expand Manhattan paths on a lattice that retains half-width edges."""
    import klayout.db as kdb
    points = [(point.x * scale, point.y * scale) for point in path.each_point()]
    points = [point for index, point in enumerate(points) if not index or point != points[index - 1]]
    if path.is_round() or path.width <= 0 or len(points) < 2:
        raise ReferenceGeometryObservationError("PATH: requires a positive-width non-round Manhattan spine")
    half = path.width * scale // 2
    for index, ((x1, y1), (x2, y2)) in enumerate(zip(points, points[1:])):
        if x1 != x2 and y1 != y2:
            raise ReferenceGeometryObservationError("PATH: non-Manhattan edges do not have exact rational offsets")
        begin = path.bgn_ext * scale if index == 0 else half
        end = path.end_ext * scale if index == len(points) - 2 else half
        if x1 == x2:
            sign = 1 if y2 > y1 else -1
            start, stop = y1 - sign * begin, y2 + sign * end
            box = x1 - half, min(start, stop), x1 + half, max(start, stop)
        else:
            sign = 1 if x2 > x1 else -1
            start, stop = x1 - sign * begin, x2 + sign * end
            box = min(start, stop), y1 - half, max(start, stop), y1 + half
        if sign * (stop - start) < 0:
            raise ReferenceGeometryObservationError("PATH: extensions reverse a path segment")
        yield kdb.Polygon(kdb.Box(*(_coordinate(value) for value in box)))


def _layer_region(occurrences, layer_index, scale):
    import klayout.db as kdb
    region = kdb.Region()
    physical = texts = 0
    for cell, (a, b, c, d, dx, dy) in occurrences:
        def transform(x, y):
            return a * x + b * y + dx * scale, c * x + d * y + dy * scale
        for shape in cell.shapes(layer_index).each():
            if shape.is_text():
                texts += 1
                continue
            if shape.is_box():
                polygon = kdb.Polygon(shape.box)
            elif shape.is_polygon():
                polygon = shape.polygon
            elif shape.is_simple_polygon():
                polygon = kdb.Polygon(shape.simple_polygon)
            elif shape.is_path():
                polygon = None
            else:
                raise ReferenceGeometryObservationError("SHAPE: unsupported non-text GDS shape")
            polygons = (_path_polygons(shape.path, scale) if polygon is None
                        else (_map_polygon(polygon, lambda x, y: (x * scale, y * scale)),))
            for polygon in polygons:
                region.insert(_map_polygon(polygon, transform))
            physical += 1
    region.merge()
    return region, physical, texts


def _area_nm2(region, unit_nm):
    def twice_area(points):
        points = [(point.x, point.y) for point in points]
        return abs(sum(x1 * y2 - x2 * y1
                       for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1])))
    area2 = sum(twice_area(polygon.each_point_hull())
                - sum(twice_area(polygon.each_point_hole(index)) for index in range(polygon.holes()))
                for polygon in region.each())
    return _rational(Fraction(area2, 2) * unit_nm * unit_nm)


def _bbox_nm(region, unit_nm):
    if region.is_empty():
        return None
    bbox = region.bbox()
    return tuple(_rational(value * unit_nm) for value in (bbox.left, bbox.bottom, bbox.right, bbox.top))


def _observe(path, top_name, declaration, source_hash, implementation):
    layout, top = _read_layout(path, top_name)
    if not isfinite(layout.dbu) or layout.dbu <= 0:
        raise ReferenceGeometryObservationError("DBU: expected a finite positive database unit")
    dbu_um = Fraction(str(layout.dbu))
    indices = sorted(layout.layer_indices(), key=lambda index: (layout.get_info(index).layer, layout.get_info(index).datatype))
    occurrences = _cell_occurrences(top)
    scale = _region_scale(occurrences, indices)
    unit_nm = 1000 * dbu_um / scale
    tokens = {}
    for token, layer, datatype in declaration[3]:
        tokens.setdefault((layer, datatype), []).append(token)
    regions, rows, boxes = {}, [], []
    for index in indices:
        info = layout.get_info(index)
        pair = info.layer, info.datatype
        region, physical, texts = _layer_region(occurrences, index, scale)
        if pair in regions:
            raise ReferenceGeometryObservationError("LAYERMAP: duplicate numeric GDS layer index")
        regions[pair] = region
        bbox = _bbox_nm(region, unit_nm)
        if bbox is not None:
            boxes.append(bbox)
        semantic_tokens = tuple(tokens.get(pair, ()))
        rows.append(ReferenceLayerGeometry(*pair, semantic_tokens,
                    "declared" if semantic_tokens else "unknown", physical, texts,
                    _area_nm2(region, unit_nm), bbox))
    bbox = None
    if boxes:
        values = [tuple(Fraction(*value) for value in box) for box in boxes]
        bbox = tuple(_rational(operation(box[i] for box in values))
                     for i, operation in enumerate((min, min, max, max)))
    observation = ReferenceGeometryObservation(
        source_hash, top.name, layout.cells(), _rational(dbu_um), _rational(unit_nm), *declaration,
        (("clay/frontend/reference_geometry_observation.py", implementation),), tuple(rows), bbox)
    return observation, regions, unit_nm


def observe_reference_geometry(gds_path, *, expected_top_cell, reference_pdk_id,
                               pdk_source_snapshot_sha256, pdk_layermap):
    """Observe selected geometry freshly; declared PDK joins remain unauthenticated."""
    declaration = _declarations(reference_pdk_id, pdk_source_snapshot_sha256, pdk_layermap)
    sources, hashes, implementation = _capture((gds_path,))
    try:
        result, _, _ = _observe(sources[0], expected_top_cell, declaration, hashes[0], implementation)
    finally:
        _guard(sources, hashes, implementation)
    return result


def _scaled_region(region, factor):
    import klayout.db as kdb
    if factor.denominator != 1 or factor <= 0:
        raise ReferenceGeometryObservationError("DBU: comparison requires an exact positive integer scale")
    factor = factor.numerator
    result = kdb.Region()
    for polygon in region.each():
        result.insert(_map_polygon(polygon, lambda x, y: (x * factor, y * factor)))
    return result


def compare_reference_geometry(*, geometry_gds_path, geometry_top_cell,
                               auxiliary_gds_path, auxiliary_top_cell,
                               reference_pdk_id, pdk_source_snapshot_sha256, pdk_layermap):
    """Prove layer-region equality in physical units, including unknown raw layers."""
    import klayout.db as kdb
    declaration = _declarations(reference_pdk_id, pdk_source_snapshot_sha256, pdk_layermap)
    sources, hashes, implementation = _capture((geometry_gds_path, auxiliary_gds_path))
    try:
        geometry, left, left_nm = _observe(sources[0], geometry_top_cell, declaration, hashes[0], implementation)
        auxiliary, right, right_nm = _observe(sources[1], auxiliary_top_cell, declaration, hashes[1], implementation)
        denominator = lcm(left_nm.denominator, right_nm.denominator)
        lattice = Fraction(gcd(int(left_nm * denominator), int(right_nm * denominator)), denominator)
        rows = []
        for pair in sorted(left.keys() | right.keys()):
            lhs = _scaled_region(left.get(pair, kdb.Region()), left_nm / lattice)
            rhs = _scaled_region(right.get(pair, kdb.Region()), right_nm / lattice)
            difference = lhs ^ rhs
            rows.append(ReferenceLayerDifference(*pair, _area_nm2(difference, lattice), difference.is_empty()))
        result = ReferenceLayerEquivalenceObservation(geometry, auxiliary, _rational(lattice),
                                                      tuple(rows), all(row.equivalent for row in rows))
    finally:
        _guard(sources, hashes, implementation)
    return result
