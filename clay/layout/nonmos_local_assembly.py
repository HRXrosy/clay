"""Captured source operands for coordinate-free, occupied non-MOS assemblies.

Only the BJT fixed-unit capability is implemented. This capture binds canonical
source bytes and replays supplied structural evidence; it does not authenticate
reference extraction, electrical parameters, body tracing, DRC or placement.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from decimal import Decimal
import hashlib
import json
from pathlib import Path

from clay.layout.unit_grid_pattern import UnitGridPattern

_SEAL = object()
_ROLES = {'c': 'c', 'collector': 'c', 'b': 'b', 'base': 'b', 'e': 'e', 'emitter': 'e'}


def require(condition, message):
    if not condition:
        raise ValueError('nonmos source '+message)


def checkpoint():
    from clay.flow.execution_session import current_session
    session = current_session()
    if session is not None:
        session.checkpoint()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest()


def _deep(value):
    """Reconstruct typed schema validators, including their nested children."""
    if type(value) is tuple:
        return tuple(_deep(row) for row in value)
    if is_dataclass(value) and not isinstance(value, type):
        return replace(value, **{field.name: _deep(getattr(value, field.name))
                                 for field in fields(value) if field.init})
    return value


def _source(cell):
    from clay.layout.route.exact_artifact_evidence import (
        _canonical_source_member_inventory, _replayed_source_cell_authority)
    checkpoint()
    sha, inventory_sha, _classes, canonical = _replayed_source_cell_authority(cell)
    require(canonical is not None, 'canonical source is missing')
    return sha, inventory_sha, _canonical_source_member_inventory(canonical), canonical


def technology_identity(tech):
    from clay.layout.tech import LayoutTech
    from clay.layout.route.resource_world import pdk_provenance_digest
    require(type(tech) is LayoutTech, 'technology type differs')
    checkpoint()
    tech.rules
    snapshot = tech.adapter.source_snapshot(refresh=True)
    # Repo-pinned open PDK data can have no external installation. None remains
    # explicit; the fixed-GDS bytes and complete rule provenance are still bound.
    bindings = []
    from clay.layout.bjt_device_cell import _REPO_ROOT
    for kind, binding in sorted(tech.adapter.bjt_bindings.items()):
        require(binding.geometry_provider == 'fixed_gds', 'BJT geometry provider is unsupported')
        path = (_REPO_ROOT / binding.geometry_path).resolve(strict=True)
        before = path.read_bytes()
        require(hashlib.sha256(before).hexdigest() == binding.geometry_sha256, 'current PDK raw master digest differs')
        bindings.append((kind, digest(asdict(binding)), binding.geometry_sha256))
    checkpoint()
    return (tech.pdk_id, pdk_provenance_digest(tech),
        None if snapshot is None else snapshot.snapshot_sha256,
        None if snapshot is None else snapshot.installed_authority_sha256, tuple(bindings))


@dataclass(frozen=True, slots=True)
class NonmosAssemblyMember:
    owner: str
    kind: str
    source_model: str
    unit_count: int
    pins: tuple[tuple[str, str], ...]
    tap: tuple[str, str, str] | None
    unit_pattern: UnitGridPattern
    origin_rank: tuple[int, int]
    raw_master_sha256: str


@dataclass(frozen=True, slots=True, init=False)
class NonmosAssemblySource:
    source_sha256: str
    source_inventory_sha256: str
    source_inventory: tuple
    source_cell_name: str
    technology_identity: tuple
    reference_overlay: object
    reference_asir: object
    mosaic: object
    members: tuple[NonmosAssemblyMember, ...]
    array_declarations: tuple
    _content_sha256: str
    _seal: object

    def __init__(self, *args, **kwargs):
        raise TypeError('NonmosAssemblySource requires source capture')

    @property
    def pdk_id(self):
        return self.technology_identity[0]

    @property
    def intent(self):
        from clay.layout.intent import layout_intent_from_declarations
        return replace(layout_intent_from_declarations(self.source_cell_name, ()),
                       arrays=_deep(self.array_declarations))

    def construction_data(self):
        return {'schema': 'clay.nonmos-local-assembly-source/v1',
            'source_sha256': self.source_sha256, 'source_inventory_sha256': self.source_inventory_sha256,
            'technology_identity': self.technology_identity, 'logical_owners': tuple(m.owner for m in self.members),
            'members': tuple(asdict(m) for m in self.members), 'mosaic': self.mosaic.canonical_data()}

    def _payload(self):
        return {**self.construction_data(), 'source_inventory': self.source_inventory,
            'source_cell_name': self.source_cell_name, 'overlay': self.reference_overlay.canonical_data(),
            'asir': self.reference_asir.canonical_data(),
            'arrays': tuple(a.canonical_data() for a in self.array_declarations)}

    @property
    def sha256(self):
        require(self._seal is _SEAL and digest(self._payload()) == self._content_sha256, 'captured source identity changed')
        return self._content_sha256


def _members(cell, intent, mosaic, overlay, tech):
    from clay.layout.annotations import Tap
    from clay.layout.source_bjt_materialization import _bjt_array_declarations
    arrays = _bjt_array_declarations(cell, intent)
    devices = {device.instance_name: device for device in cell.devices}
    annotations = getattr(cell, '_layout_annotations', {}) or {}
    columns = {(owner, ordinal): rank for owner, ordinal, rank in mosaic.column_ranks}
    slots = {occ.node_id: (row, columns[(occ.logical_owner, occ.occurrence_ordinal)], occ.transform)
             for row, occurrences in enumerate(mosaic.rows) for occ in occurrences}
    observed = {occ.node_id: occ for occ in overlay.occurrences}
    terminals = {}
    for row in overlay.terminals:
        terminals.setdefault(row.occurrence_id, []).append(row)
    result = []
    for owner in mosaic.members:
        checkpoint()
        require(owner in devices, 'selected source owner is absent')
        device = devices[owner]
        kind, pins = device.device_class, tuple(sorted(device.pins.items()))
        require(kind in ('pnp', 'npn') and set(dict(pins)) == {'c', 'b', 'e'}, 'BJT source kind or terminal roles differ')
        binding = tech.adapter.bjt_bindings.get(kind)
        require(binding is not None and device.source_model == binding.lvs_model, 'source model differs from current PDK')
        sizing = device.sizing
        count = Decimal(str(sizing.get('area', 1))) * Decimal(str(sizing.get('M', 1)))
        require(count.is_finite() and count > 0 and count == int(count) and count <= 4096,
                'source unit count is not a bounded positive integer')
        occurrences = tuple(o for o in mosaic.occurrences if o.logical_owner == owner)
        require(tuple(sorted(o.occurrence_ordinal for o in occurrences)) == tuple(range(int(count))),
                'source occurrence count or ordinal cover differs')
        ordered = tuple(sorted(occurrences, key=lambda o: o.occurrence_ordinal))
        ranks = tuple(slots[o.node_id] for o in ordered)
        require(all(pose == 'R0' for _r, _c, pose in ranks), 'BJT orientation supports only R0')
        r0, c0 = min(r for r, _c, _p in ranks), min(c for _r, c, _p in ranks)
        pattern = UnitGridPattern(tuple((r-r0, c-c0, pose) for r, c, pose in ranks))
        declared = arrays.get(owner)
        require((declared is not None and declared.unit_pattern == pattern)
                or (declared is None and count == 1), 'effective source array pattern differs from projected occupied cells')
        for occurrence in ordered:
            raw = observed[occurrence.node_id]
            require(raw.device_class == kind and raw.body_well_id is None, 'source BJT semantic class/body differs')
            roles = tuple((_ROLES.get(t.role), t.net) for t in terminals.get(raw.node_id, ()))
            require(len(roles) == 3 and tuple(sorted(roles)) == pins, 'source semantic terminal role/net cover differs')
        taps = tuple(a for a in annotations.get(owner, ()) if type(a) is Tap)
        require(len(taps) <= 1, 'source tap declaration is duplicated')
        tap = None
        if taps:
            declared_tap = taps[0]
            tb = binding.tap_binding
            require(tb is not None and tb.mode == 'intrinsic_terminal', 'assembly external companion tap capability is unsupported')
            require(declared_tap.where in tb.where and declared_tap.net == dict(pins)[tb.terminal_role],
                    'source intrinsic tap net/domain differs')
            tap = (declared_tap.net, declared_tap.where, 'declared_not_traced')
        result.append(NonmosAssemblyMember(owner, kind, device.source_model, int(count), pins, tap,
                                           pattern, (r0, c0), binding.geometry_sha256))
    require(len({(m.kind, m.source_model, m.raw_master_sha256) for m in result}) == 1,
            'assembly requires one current fixed-unit model/master')
    return tuple(sorted(result, key=lambda m: m.owner))


def capture_nonmos_assembly_source(source_cell, *, tech, intent, reference_overlay,
        reference_asir, mosaic_id, logical_owners):
    from clay.layout.intent import LayoutIntent
    from clay.layout.spatial import AbstractSpatialIR, derive_abstract_spatial_ir
    from clay.layout.spatial.semantic import SemanticOverlay
    from clay.layout.spatial.lattice_mosaic_projection import project_lattice_mosaic
    checkpoint()
    sha, inventory_sha, inventory, canonical = _source(source_cell)
    identity = technology_identity(tech)
    require(type(intent) is LayoutIntent, 'effective intent requires exact type')
    require(type(logical_owners) is tuple and logical_owners == tuple(sorted(set(logical_owners))) and logical_owners,
            'owner scope must be canonical and nonempty')
    require(type(reference_overlay) is SemanticOverlay and reference_overlay.source_kind == 'reference_gds',
            'reference overlay type/scope differs')
    require(type(reference_asir) is AbstractSpatialIR, 'ASIR type differs')
    overlay, supplied = _deep(reference_overlay), _deep(reference_asir)
    asir = derive_abstract_spatial_ir(overlay)
    require(asir.canonical_bytes == supplied.canonical_bytes, 'ASIR replay differs from complete overlay')
    mosaic = project_lattice_mosaic(asir, mosaic_id=mosaic_id, logical_owners=logical_owners)
    members = _members(canonical, intent, mosaic, overlay, tech)
    selected_arrays = tuple(a for a in intent.arrays if a.group in logical_owners)
    values = dict(source_sha256=sha, source_inventory_sha256=inventory_sha, source_inventory=inventory,
        source_cell_name=canonical.local_name, technology_identity=identity, reference_overlay=overlay,
        reference_asir=asir, mosaic=mosaic, members=members, array_declarations=_deep(selected_arrays))
    result = object.__new__(NonmosAssemblySource)
    for key, value in values.items():
        object.__setattr__(result, key, value)
    object.__setattr__(result, '_seal', _SEAL)
    object.__setattr__(result, '_content_sha256', digest(result._payload()))
    require(_source(source_cell)[:3] == (sha, inventory_sha, inventory)
            and technology_identity(tech) == identity, 'source or PDK changed during capture')
    return result


def replay_nonmos_assembly_source(value, *, source_cell, tech, intent=None):
    require(type(value) is NonmosAssemblySource and value._seal is _SEAL, 'source capture is absent or foreign')
    old = value.sha256
    intent = value.intent if intent is None else intent
    actual = capture_nonmos_assembly_source(source_cell, tech=tech, intent=intent,
        reference_overlay=value.reference_overlay, reference_asir=value.reference_asir,
        mosaic_id=value.mosaic.mosaic_id, logical_owners=tuple(m.owner for m in value.members))
    require(actual.sha256 == old, 'source/PDK inventory replay differs')
    return value
