"""Source-bound local MIM construction and its exact native replay operands.

Association means a source declaration corresponds to an observed physical
master. It is not an electrical capacitance measurement, reference acceptance,
or a general source-authority registration API. Native consumption reopens the
source/evidence and rematerializes the explicit recipe in the current PDK.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
import hashlib
import json
import math
from pathlib import Path
from typing import ClassVar

import klayout.db as kdb

from clay.frontend.reference_occurrence_binding import (
    LogicalOwner, PhysicalOccurrence, solve_reference_occurrence_binding,
)
from clay.frontend.unit_master_geometry import UnitMasterGeometry
from clay.layout.bank_interconnect import replay_bank_interconnect_plan
from clay.layout.device_terminal import extract_device_nodes
from clay.layout.mim_unit_materializer import (
    _contained_box, _regions, _retained, _transform_regions, materialize_mim_unit,
)
from clay.layout.route.exact_artifact_evidence import (
    _canonical_source_member_inventory, _replayed_source_cell_authority,
)

_SEAL = object()
_D4 = {"R0", "R90", "R180", "R270", "MR0", "MR90", "MR180", "MR270"}


def _require(condition, message):
    if not condition:
        raise ValueError("source MIM " + message)


def _immutable(value):
    if type(value) is tuple:
        return tuple(_immutable(item) for item in value)
    if value is None or type(value) in (str, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise TypeError("source MIM requires immutable exact scalar/tuple operands")


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _source(cell):
    digest, inventory, _classes, canonical = _replayed_source_cell_authority(cell)
    _require(canonical is not None, "canonical source is missing")
    return digest, inventory, canonical


def _member(cell, name):
    rows = [row for row in _canonical_source_member_inventory(cell) if row[0] == name]
    _require(len(rows) == 1 and rows[0][1] == "cap", "source capacitor member is missing")
    row = rows[0]
    sizing, pins = dict(row[5]), dict(row[4])
    value, count = sizing.get("C", sizing.get("value")), sizing.get("M", 1)
    _require(type(count) in (int, float) and math.isfinite(count) and count > 0
             and count == int(count), "source multiplicity must be an exact positive integer")
    _require(type(value) in (int, float) and math.isfinite(value) and value > 0,
             "source capacitor value must be finite and positive")
    _require(set(pins) == {"p", "n"}, "source requires complete p/n pins")
    return row, float(value), int(count), pins


def _technology(tech):
    from clay.layout.tech import LayoutTech
    _require(type(tech) is LayoutTech, "technology must be exact LayoutTech")
    tech.rules
    snapshot = tech.adapter.source_snapshot(refresh=True)
    _require(snapshot is not None and snapshot.installed_authority_sha256 is not None,
             "PDK installed source snapshot is missing")
    return tech.pdk_id, snapshot.snapshot_sha256, snapshot.installed_authority_sha256


def _files(paths):
    _require(type(paths) is tuple and bool(paths), "evidence file denominator is missing")
    resolved = tuple(sorted(str(Path(path).resolve(strict=True)) for path in paths))
    _require(len(set(resolved)) == len(resolved), "evidence paths are duplicated")
    return tuple((path, hashlib.sha256(Path(path).read_bytes()).hexdigest()) for path in resolved)


def _reopen(rows):
    _require(_files(tuple(path for path, _sha in rows)) == rows, "evidence files changed digest")


@dataclass(frozen=True, slots=True, init=False)
class MimSourceAssociation:
    source_member: tuple
    source_sha256: str
    source_path: str
    source_cell_name: str
    tech_identity: tuple
    master_variant_sha256: str
    occurrence_ids: tuple[str, ...]
    parameter_class: str
    evidence_files: tuple
    graph_input_sha256: str
    _seal: object
    scope: ClassVar[str] = "source_declaration_observed_master_correspondence_not_electrical_value_verification"

    def __init__(self, *args, **kwargs):
        raise TypeError("MIM association requires actual source/graph capture")

    def __deepcopy__(self, memo):
        return self


def _capture(record_type, **values):
    _require(set(values) == {field.name for field in fields(record_type)} - {"_seal"},
             "capture field denominator differs")
    result = object.__new__(record_type)
    for name, value in values.items():
        object.__setattr__(result, name, value)
    object.__setattr__(result, "_seal", _SEAL)
    return result


def capture_mim_source_association(source_cell, *, logical_id, master, source_member,
        physical_occurrences, logical_owners, port_anchors, observed_variants, evidence_files, tech):
    """Replay supplied graph uniqueness and capture its explicit source/master join.

The host supplying real observations must bind source parameters to its actual
reference evidence. This function does not infer C from artwork or a class name.
"""
    before, _inventory, canonical = _source(source_cell)
    member, _value, count, pins = _member(canonical, logical_id)
    _require(_immutable(source_member) == member, "source parameter declaration differs")
    _require(type(master) is UnitMasterGeometry, "master must be exact UnitMasterGeometry")
    master.__post_init__()
    identity, bound = _technology(tech), _files(evidence_files)
    _require(master.pdk_id == identity[0], "master PDK differs")
    _require(type(physical_occurrences) is tuple and all(type(row) is PhysicalOccurrence for row in physical_occurrences),
             "physical occurrence inventory must be immutable typed data")
    _require(type(logical_owners) is tuple and all(type(row) is LogicalOwner for row in logical_owners),
             "logical owner inventory must be immutable typed data")
    owners = [row for row in logical_owners if row.owner_id == logical_id]
    _require(len(owners) == 1, "source owner correspondence is missing")
    owner = owners[0]
    _require(owner.multiplicity == count and owner.terminals == tuple(sorted((role, (net,)) for role, net in pins.items())),
             "source owner pins or multiplicity differs from binding")
    binding = solve_reference_occurrence_binding(physical_occurrences, logical_owners,
        port_anchors=port_anchors, prove_net_mapping=True)
    _require(binding.status == "UNIQUE" and binding.net_mapping_proof.status == "UNIQUE",
             "owner/net correspondence is not uniquely proven")
    ids = tuple(occurrence for occurrence, owner_id in binding.bindings if owner_id == logical_id)
    _require(len(ids) == count, "source occurrence denominator differs")
    _require(_immutable(observed_variants) == tuple((name, master.variant_sha256) for name in ids),
             "observed occurrence/variant denominator differs")
    _reopen(bound)
    _require(_source(source_cell)[0] == before and _technology(tech) == identity,
             "source or PDK changed during association capture")
    return _capture(MimSourceAssociation, source_member=member, source_sha256=before,
        source_path=str(Path(source_cell.source_span["file"]).resolve()), source_cell_name=canonical.local_name,
        tech_identity=identity, master_variant_sha256=master.variant_sha256, occurrence_ids=ids,
        parameter_class=owner.parameter_class, evidence_files=bound, graph_input_sha256=binding.witness.input_sha256)


@dataclass(frozen=True, slots=True)
class MimSourceRecipe:
    association: MimSourceAssociation
    master: UnitMasterGeometry
    site_roles: tuple[tuple[str, str], ...]
    rows: int = 1
    unit_transforms: tuple[str, ...] = ()
    orient: str = "R0"
    bank_align: str = "bottom"
    bank_interconnect: object = None
    quality_plate_site: str | None = None

    def __post_init__(self):
        _require(type(self.association) is MimSourceAssociation and self.association._seal is _SEAL,
                 "recipe requires captured source association")
        _require(type(self.master) is UnitMasterGeometry, "recipe requires exact master")
        self.master.__post_init__()
        _require(self.master.variant_sha256 == self.association.master_variant_sha256, "recipe master variant differs")
        _require(type(self.site_roles) is tuple and all(type(row) is tuple and len(row) == 2
                 and all(type(item) is str for item in row) for row in self.site_roles), "site roles must be exact immutable pairs")
        _require(_immutable(self.site_roles) == self.site_roles and tuple(row[0] for row in self.site_roles)
                 == tuple(site.site_id for site in self.master.terminal_sites)
                 and {row[1] for row in self.site_roles} == {"p", "n"}, "complete site role denominator differs")
        count = len(self.association.occurrence_ids)
        _require(type(self.rows) is int and 0 < self.rows <= count and count % self.rows == 0,
                 "recipe rows must divide the complete unit count")
        _require(type(self.unit_transforms) is tuple and len(self.unit_transforms) == count
                 and all(type(item) is str and item in _D4 for item in self.unit_transforms),
                 "recipe transforms require every actual unit")
        _require(type(self.orient) is str and self.orient in _D4, "recipe orientation must be D4")
        _require(type(self.bank_align) is str and self.bank_align in ("bottom", "center", "top"), "recipe alignment differs")
        if self.bank_interconnect is not None:
            replay_bank_interconnect_plan(self.bank_interconnect)
            _require(count > 1, "bank interconnect requires multiple units")
        if self.quality_plate_site is not None:
            _require(type(self.quality_plate_site) is str and any(
                site.site_id == self.quality_plate_site and site.role_group == 'plate'
                for site in self.master.terminal_sites), "recipe quality plate site is missing")


@dataclass(frozen=True, slots=True, init=False)
class SourceMimMaterialization:
    source_member: tuple
    source_sha256: str
    source_inventory_sha256: str
    recipe: MimSourceRecipe
    geometry: object
    site_ledger: tuple
    terminal_layers: tuple
    terminal_accesses: tuple
    geometry_snapshot: bytes
    _seal: object

    def __init__(self, *args, **kwargs):
        raise TypeError("source MIM requires native materialization capture")

    def __deepcopy__(self, memo):
        return self


def _geometry_snapshot(geometry):
    from clay.layout.passive_terminal_binding import _polygon_rows
    return _json((_polygon_rows(geometry.polygons), tuple(sorted((role, terminal.role, terminal.net, terminal.bbox)
        for role, terminal in geometry.terminals.items())), geometry.kind, geometry.finger_count, geometry.orient,
        geometry.extent(), geometry.capacitor_unit_occurrences.canonical_data(),
        getattr(geometry, "parallel_mim_bank", None), getattr(geometry, "bank_interconnect", None)))


def _check_recipe_source(cell, recipe, tech):
    _require(type(recipe) is MimSourceRecipe, "requires exact construction recipe")
    recipe.__post_init__()
    association = recipe.association
    digest, inventory, canonical = _source(cell)
    member, value, count, nets = _member(canonical, association.source_member[0])
    _require(digest == association.source_sha256 and member == association.source_member,
             "current source digest or source parameter declaration differs")
    _require(_technology(tech) == association.tech_identity, "current PDK identity differs")
    from clay.layout.generic_placement import _resistor_orient_map
    _require(_resistor_orient_map(canonical).get(member[0], "R0") == recipe.orient,
             "recipe orientation differs from canonical source")
    _reopen(association.evidence_files)
    return digest, inventory, member, value, count, nets


def _native_geometry(recipe, tech, value, count, nets):
    from clay.layout.capacitor_unit_occurrences import _compose
    from clay.layout.passive_device_cell import _build_parallel_mim_bank
    from clay.layout.route.pins import _mim_spec_for_passive
    from types import SimpleNamespace
    spec = _mim_spec_for_passive(tech, SimpleNamespace(model=recipe.association.source_member[3]))
    _require(spec is not None, "PDK MIM model is unavailable")
    def unit(orient):
        return materialize_mim_unit(recipe.master, site_roles=recipe.site_roles,
            nets=tuple(sorted(nets.items())), tech=tech, mim_spec=spec, orient=orient,
            quality_plate_site=recipe.quality_plate_site)
    if count == 1:
        pose = _compose(recipe.orient, recipe.unit_transforms[0])
        units, offsets = (unit(pose),), ((0, 0),)
        geometry = replace(units[0].geometry, orient=recipe.orient,
            capacitor_unit_occurrences=replace(units[0].geometry.capacitor_unit_occurrences,
                                               bank_orientation=recipe.orient))
    else:
        base = unit("R0")
        geometry = _build_parallel_mim_bank(tech, value, nets=nets, m=count,
            unit_builder=lambda *_args, **_kwargs: base.geometry, unit_kw={"orient": recipe.orient},
            bank_rows=recipe.rows, unit_transforms=recipe.unit_transforms, bank_align=recipe.bank_align,
            bank_interconnect_plan=recipe.bank_interconnect)
        _require(recipe.orient == "R0", "multiunit source MIM currently requires outer orientation R0")
        units = tuple(unit(pose) for pose in recipe.unit_transforms)
        offsets = tuple(geometry.parallel_mim_bank["r0_replica_offsets_nm"])
    return geometry, units, offsets, spec


def _site_ledger(recipe, geometry, units, offsets, tech, spec, nets):
    from clay.layout.passive_terminal_binding import MimMaterializedSite
    from clay.layout.mim_row_bus import _artwork
    nodes = extract_device_nodes(geometry, tech=tech, mim_spec=spec)
    _require(len(nodes) == 2, "native bank must retain two plate nodes")
    expected, ledger, accesses = {}, [], {"p": [], "n": []}
    layers = dict(units[0].terminal_layers)
    for index, (unit, offset) in enumerate(zip(units, offsets)):
        pose = unit.geometry.capacitor_unit_occurrences.unit_transforms[0]
        local, transform, _shift = _transform_regions(recipe.master, pose)
        transform = kdb.Trans(*offset) * transform
        for layer, region in local.items():
            expected.setdefault(layer, kdb.Region())
            expected[layer] += region.transformed(kdb.Trans(*offset))
        for site, access in zip(recipe.master.terminal_sites, unit.site_accesses):
            masks = {layer: region.transformed(transform) for layer, region in _regions(site.geometry_by_layer_nm).items()}
            routing = {layer: region.transformed(kdb.Trans(*offset))
                       for layer, region in _regions(access.routing_geometry_by_layer_nm).items()}
            ledger.append(MimMaterializedSite(index, site.site_id, access.role, nets[access.role], pose,
                (transform.disp.x, transform.disp.y), _retained(masks), _retained(routing)))
            _require(dict(unit.terminal_layers)[access.role] == layers[access.role], "unit routing layer changed across array")
            accesses[access.role].append(_contained_box(routing[layers[access.role]]))
    # The old tiler emits every original unit polygon before added interconnect.
    size = len(units[0].geometry.polygons) * len(units)
    copied = _artwork(geometry.polygons[:size])
    _require(set(copied) == set(expected) and all((copied[layer] ^ region).is_empty()
             for layer, region in expected.items()), "actual copied unit artwork differs from exact D4 recipe")
    for row in ledger:
        row.verify_nodes(nodes)
    return tuple(ledger), tuple(sorted(layers.items())), tuple((role, tuple(boxes)) for role, boxes in sorted(accesses.items()))


def materialize_source_mim(source_cell, *, recipe, tech):
    before, inventory, member, value, count, nets = _check_recipe_source(source_cell, recipe, tech)
    geometry, units, offsets, spec = _native_geometry(recipe, tech, value, count, nets)
    ledger, layers, accesses = _site_ledger(recipe, geometry, units, offsets, tech, spec, nets)
    _require(_check_recipe_source(source_cell, recipe, tech)[:2] == (before, inventory),
             "source changed during native construction")
    return _capture(SourceMimMaterialization, source_member=member, source_sha256=before,
        source_inventory_sha256=inventory, recipe=recipe, geometry=geometry, site_ledger=ledger,
        terminal_layers=layers, terminal_accesses=accesses, geometry_snapshot=_geometry_snapshot(geometry))


def replay_source_mim_materialization(value, *, source_cell=None, record=None, tech):
    _require(type(value) is SourceMimMaterialization and value._seal is _SEAL,
             "override requires exact native source MIM capture")
    if source_cell is None:
        from clay.parser import parse_clay_file
        association = value.recipe.association
        matches = [cell for cell in parse_clay_file(association.source_path).cell_defs
                   if cell.local_name == association.source_cell_name]
        _require(len(matches) == 1, "source cell identity changed")
        source_cell = matches[0]
    _require(_geometry_snapshot(value.geometry) == value.geometry_snapshot, "materialized geometry changed")
    fresh = materialize_source_mim(source_cell, recipe=value.recipe, tech=tech)
    for name in ("source_member", "source_sha256", "source_inventory_sha256", "geometry_snapshot",
                 "site_ledger", "terminal_layers", "terminal_accesses"):
        _require(getattr(value, name) == getattr(fresh, name), "native materialization replay differs: " + name)
    if record is not None:
        _member_row, capacitance, count, pins = _member(source_cell, value.source_member[0])
        _require((record.get("kind"), record.get("value"), record.get("M"), record.get("pins"), record.get("model", ""),
                  record.get("orient", "R0")) == ("cap", capacitance, count, pins, value.source_member[3] or "", value.recipe.orient),
                 "native record differs from exact source declaration")
    return value.geometry


def source_mim_record(record, value):
    """Retain explicit construction choices without replacing electrical fields."""
    recipe = value.recipe
    choices = {"array_rows": recipe.rows, "array_transforms": recipe.unit_transforms,
               "array_align": recipe.bank_align, "bank_interconnect_plan": recipe.bank_interconnect}
    result = dict(record)
    for name, actual in choices.items():
        if name in result:
            _require(result[name] == actual, "source array declaration conflicts with construction recipe")
        if actual is not None:
            result[name] = actual
    result["_geometry_override"] = value
    return result


def mim_source_recipes_from_capture(*args, **kwargs):
    from clay.layout.source_mim_reference import mim_source_recipes_from_capture as capture
    return capture(*args, **kwargs)


__all__ = ["MimSourceAssociation", "MimSourceRecipe", "SourceMimMaterialization",
           "capture_mim_source_association", "materialize_source_mim", "replay_source_mim_materialization", "source_mim_record",
           "mim_source_recipes_from_capture"]
