"""Immutable local artwork and access variants, without occurrence ownership.

These records carry supplied data, not source, extraction, accessibility or DRC
authority. Integer translation removes the intrinsic bounding-box origin only;
native axes, holes and contact variants remain distinct. No source net, logical
owner, placement coordinate or routable DeviceGeom is constructed here.

Polygon constructors validate the shared immutable schema. Call
``validate_unit_master_regions`` separately for actual backend roundtrip and
local site coverage. Even that check does not compare a source GDS or establish
terminal roles: the source reader must independently bind those observations.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json

from clay.layout.physical_terminal_inventory import PhysicalTerminalPolygonNM

LayerKey = tuple[int, int]
LayerGeometry = tuple[tuple[LayerKey, tuple[PhysicalTerminalPolygonNM, ...]], ...]
_MAX_ITEMS = 262_144
_MAX_VERTICES = 1_048_576


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError('unit master ' + message)


def _text(value: object, field: str) -> None:
    _require(type(value) is str and 0 < len(value) <= 4096, field + ' must be a bounded exact string')


def _tuple(value: object, field: str, *, nonempty: bool = False) -> None:
    _require(type(value) is tuple and len(value) <= _MAX_ITEMS and (bool(value) or not nonempty),
             field + ' must be a bounded exact tuple')


def _sha(value: object, field: str) -> None:
    _require(type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value),
             field + ' must be SHA256')


def _geometry(value: LayerGeometry) -> int:
    _tuple(value, 'geometry', nonempty=True)
    layers, vertices = [], 0
    for row in value:
        _require(type(row) is tuple and len(row) == 2, 'layer record must be an exact pair')
        layer, shapes = row
        _require(type(layer) is tuple and len(layer) == 2 and
                 all(type(x) is int and 0 <= x <= 65535 for x in layer), 'layer must be an exact integer pair')
        layers.append(layer)
        _tuple(shapes, 'polygons', nonempty=True)
        for shape in shapes:
            _require(type(shape) is PhysicalTerminalPolygonNM, 'polygon requires the exact immutable schema')
            shape.__post_init__()
            vertices += len(shape.hull_nm) + sum(map(len, shape.holes_nm))
            _require(vertices <= _MAX_VERTICES, 'geometry exceeds the total vertex bound')
        _require(shapes == tuple(sorted(set(shapes))), 'polygons must be canonical and unique')
    _require(layers == sorted(set(layers)), 'geometry layers must be canonical and unique')
    return vertices


def _extent(geometry: LayerGeometry) -> tuple[int, int, int, int]:
    points = tuple(point for _layer, shapes in geometry for shape in shapes for point in shape.hull_nm)
    return (min(x for x, _ in points), min(y for _, y in points),
            max(x for x, _ in points), max(y for _, y in points))


def _payload(geometry: LayerGeometry) -> tuple:
    return tuple((layer, tuple((shape.hull_nm, shape.holes_nm) for shape in shapes)) for layer, shapes in geometry)


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=True, allow_nan=False).encode('ascii')).hexdigest()


@dataclass(frozen=True, slots=True, order=True)
class UnitMasterSource:
    auxiliary_gds_sha256: str
    master_cell: str
    observation_sha256: str

    def __post_init__(self) -> None:
        _sha(self.auxiliary_gds_sha256, 'auxiliary GDS identity')
        _text(self.master_cell, 'master cell alias')
        _sha(self.observation_sha256, 'observation identity')


@dataclass(frozen=True, slots=True, order=True)
class UnitTerminalSite:
    site_id: str
    role_group: str
    geometry_by_layer_nm: LayerGeometry

    def __post_init__(self) -> None:
        _text(self.site_id, 'site identity')
        _text(self.role_group, 'site role group')
        _geometry(self.geometry_by_layer_nm)


@dataclass(frozen=True, slots=True)
class UnitMasterGeometry:
    pdk_id: str
    parameter_class: str
    sources: tuple[UnitMasterSource, ...]
    geometry_by_layer_nm: LayerGeometry
    terminal_sites: tuple[UnitTerminalSite, ...]
    unresolved_access_obligations: tuple[str, ...]

    def __post_init__(self) -> None:
        _text(self.pdk_id, 'PDK identity')
        _text(self.parameter_class, 'parameter class')
        _tuple(self.sources, 'sources', nonempty=True)
        for source in self.sources:
            _require(type(source) is UnitMasterSource, 'source requires the exact immutable schema')
            source.__post_init__()
        _require(self.sources == tuple(sorted(set(self.sources))), 'sources must be canonical and unique')
        vertices = _geometry(self.geometry_by_layer_nm)
        _require(_extent(self.geometry_by_layer_nm)[:2] == (0, 0), 'artwork must use its intrinsic lower-left origin')
        _tuple(self.terminal_sites, 'terminal sites')
        ids = []
        for site in self.terminal_sites:
            _require(type(site) is UnitTerminalSite, 'site requires the exact immutable schema')
            site.__post_init__()
            ids.append(site.site_id)
            vertices += _geometry(site.geometry_by_layer_nm)
            _require(vertices <= _MAX_VERTICES, 'artwork and sites exceed the total vertex bound')
        _require(ids == sorted(set(ids)), 'site identities must be canonical and unique')
        _tuple(self.unresolved_access_obligations, 'access obligations')
        for obligation in self.unresolved_access_obligations:
            _text(obligation, 'access obligation')
        _require(self.unresolved_access_obligations == tuple(sorted(set(self.unresolved_access_obligations))),
                 'access obligations must be canonical and unique')

    @property
    def extent_nm(self) -> tuple[int, int, int, int]:
        return _extent(self.geometry_by_layer_nm)

    @property
    def artwork_sha256(self) -> str:
        """Canonical local polygon records; a digest alone proves no geometry."""
        return _digest(('clay.unit-master-artwork/v1', _payload(self.geometry_by_layer_nm)))

    @property
    def variant_sha256(self) -> str:
        return _digest(('clay.unit-master-variant/v1', self.pdk_id, self.parameter_class,
            self.artwork_sha256, tuple((site.site_id, site.role_group, _payload(site.geometry_by_layer_nm))
                for site in self.terminal_sites), self.unresolved_access_obligations))


def normalize_unit_master_geometry(*, pdk_id: str, parameter_class: str,
        sources: tuple[UnitMasterSource, ...], geometry_by_layer_nm: LayerGeometry,
        terminal_sites: tuple[UnitTerminalSite, ...],
        unresolved_access_obligations: tuple[str, ...]) -> UnitMasterGeometry:
    """Translate intrinsic artwork and sites together, preserving native axes."""
    _geometry(geometry_by_layer_nm)
    _tuple(terminal_sites, 'terminal sites')
    for site in terminal_sites:
        _require(type(site) is UnitTerminalSite, 'site requires the exact immutable schema')
        site.__post_init__()
    x0, y0, _x1, _y1 = _extent(geometry_by_layer_nm)

    def translate(geometry: LayerGeometry) -> LayerGeometry:
        def ring(points: tuple) -> tuple:
            return tuple((x - x0, y - y0) for x, y in points)
        return tuple((layer, tuple(PhysicalTerminalPolygonNM(ring(shape.hull_nm),
            tuple(ring(hole) for hole in shape.holes_nm)) for shape in shapes)) for layer, shapes in geometry)

    return UnitMasterGeometry(pdk_id, parameter_class, sources, translate(geometry_by_layer_nm),
        tuple(UnitTerminalSite(site.site_id, site.role_group, translate(site.geometry_by_layer_nm))
            for site in terminal_sites), unresolved_access_obligations)


def canonical_unit_master_library(masters: tuple[UnitMasterGeometry, ...]) -> tuple[UnitMasterGeometry, ...]:
    """Deduplicate exact variants while retaining every independent source alias."""
    _tuple(masters, 'library', nonempty=True)
    aliases, variants = {}, {}
    for master in masters:
        _require(type(master) is UnitMasterGeometry, 'library requires exact typed masters')
        master.__post_init__()
        key = master.variant_sha256
        for source in master.sources:
            alias = source.auxiliary_gds_sha256, source.master_cell
            _require(alias not in aliases or aliases[alias] == key, 'source alias binds contradictory variants')
            aliases[alias] = key
        prior = variants.get(key)
        if prior is None:
            variants[key] = master
        else:
            _require(replace(prior, sources=master.sources) == master, 'variant digest collision')
            variants[key] = replace(prior, sources=tuple(sorted(set(prior.sources + master.sources))))
    return tuple(variants[key] for key in sorted(variants))


def validate_unit_master_regions(master: UnitMasterGeometry) -> None:
    """Replay polygons and site containment, without a source/access PASS token.

    Each layer must already be the canonical merged polygon inventory. This
    makes the artwork digest unambiguous after validation while constructors
    and hashing remain free of native geometry work.
    """
    import klayout.db as kdb
    from clay.layout.occurrence_channels import _shape_keys

    _require(type(master) is UnitMasterGeometry, 'region replay requires an exact typed master')
    master.__post_init__()

    def regions(geometry: LayerGeometry) -> dict:
        result = {}
        for layer, shapes in geometry:
            region = kdb.Region()
            for shape in shapes:
                region += shape.to_region()
            _require(_shape_keys(region) == tuple((shape.hull_nm, shape.holes_nm) for shape in shapes),
                     'layer geometry must be canonical merged polygons')
            result[layer] = region
        return result

    artwork = regions(master.geometry_by_layer_nm)
    for site in master.terminal_sites:
        for layer, region in regions(site.geometry_by_layer_nm).items():
            _require(layer in artwork and (region - artwork[layer]).is_empty(),
                     'site geometry is outside its own master layer')


__all__ = ['UnitMasterSource', 'UnitTerminalSite', 'UnitMasterGeometry',
    'normalize_unit_master_geometry', 'canonical_unit_master_library', 'validate_unit_master_regions']
