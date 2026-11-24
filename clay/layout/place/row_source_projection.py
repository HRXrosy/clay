"""Retained source operands for rows compiled from a checked discrete projection.

The private capture boundary is called by the fresh source producer, after its
source adapter succeeds. Public data, reconstructed JSON, and materialization
hashes cannot register that origin. Placement separately seals the actual row
compiler objects in its existing physical-macro scope.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
from weakref import WeakKeyDictionary

from clay.layout.intent import layout_intent_from_cell_def
from clay.layout.intent_authority_composition import CompositeIntentFactAuthority
from clay.layout.spatial import AbstractSpatialIR, derive_abstract_spatial_ir, project_discrete_mosaic
from clay.layout.spatial.derive import AggregateAlignmentSpec, MacroDagLevelSpec
from clay.layout.spatial.intent_projection import replay_discrete_mosaic_inventory
from clay.layout.spatial.mosaic_guest_slot import (
    derive_mosaic_guest_slot,
    replay_mosaic_guest_slot_inventory,
)
from clay.layout.spatial.semantic import SemanticOverlay
from clay.layout.spatial.mosaic_boundary_joins import replay_mosaic_boundary_join_inventory


class RowSourceProjectionError(ValueError):
    """A projected row lost its checked source or original compiler binding."""


def _bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def _require(condition, code):
    if not condition:
        raise RowSourceProjectionError(f"ROW_SOURCE_{code}")


_SOURCE_AUTHORITY_CACHE: dict[tuple[str, str, str, int], tuple[str, dict, object]] = {}
_SOURCE_AUTHORITY_CACHE_MAX = 32


def _source(cell):
    from clay import parser as _clay_parser
    from clay.frontend.source_revision import electrical_source_data
    from clay.ir import CellDef
    from clay.layout.route.exact_artifact_evidence import _replayed_source_cell_authority

    if type(cell) is CellDef:
        local_name = getattr(cell, "local_name", None)
        span = getattr(cell, "source_span", None)
        raw_path = span.get("file") if isinstance(span, Mapping) else None
        if (
            type(local_name) is str
            and local_name
            and local_name == local_name.strip()
            and type(raw_path) is str
            and raw_path
        ):
            try:
                path = Path(raw_path).resolve()
                before_bytes = path.read_bytes()
            except OSError:
                before_bytes = None
            if before_bytes is not None:
                live_sha = hashlib.sha256(before_bytes).hexdigest()
                cache_key = (str(path), live_sha, local_name, id(_clay_parser.parse_clay_file))
                cached = _SOURCE_AUTHORITY_CACHE.get(cache_key)
                if cached is not None:
                    inv_sha, expected_elec, canonical_tpl = cached
                    try:
                        same_bytes = path.read_bytes() == before_bytes
                        same_elec = same_bytes and (electrical_source_data(cell) == expected_elec)
                    except Exception:
                        same_elec = False
                    if same_elec:
                        return live_sha, inv_sha, deepcopy(canonical_tpl)

    digest, inventory, _classes, canonical = _replayed_source_cell_authority(cell)
    _require(canonical is not None, "SOURCE_MISSING")
    if type(canonical) is CellDef:
        span = getattr(canonical, "source_span", None)
        raw_path = span.get("file") if isinstance(span, Mapping) else None
        if type(raw_path) is str and raw_path:
            try:
                resolved = str(Path(raw_path).resolve())
                elec = electrical_source_data(canonical)
                cache_key = (resolved, digest, canonical.local_name, id(_clay_parser.parse_clay_file))
                if len(_SOURCE_AUTHORITY_CACHE) >= _SOURCE_AUTHORITY_CACHE_MAX:
                    _SOURCE_AUTHORITY_CACHE.pop(next(iter(_SOURCE_AUTHORITY_CACHE)))
                _SOURCE_AUTHORITY_CACHE[cache_key] = (inventory, elec, deepcopy(canonical))
            except Exception:
                pass
    return digest, inventory, canonical


def _tech_identity(tech):
    from clay.layout.tech import LayoutTech

    _require(type(tech) is LayoutTech, "TECH_TYPE")
    # The accessor checks the installed layout authority before compilation.
    tech.rules
    snapshot = tech.adapter.source_snapshot()
    return (tech.pdk_id, None if snapshot is None else snapshot.snapshot_sha256,
            None if snapshot is None else snapshot.installed_authority_sha256)


@dataclass(frozen=True, eq=False, init=False)
class RowSourceProjection:
    """Producer-captured replay operands; construction grants no public authority."""

    source_sha256: str
    source_inventory_sha256: str
    authoring_source_sha256: str
    authoring_source_cell: object
    source_cell_name: str
    source_cir: str | None
    topology_sha256: str
    device_records_bytes: bytes
    tech_identity: tuple
    reference_overlay: SemanticOverlay
    reference_asir: AbstractSpatialIR
    scope_by_logical: tuple
    aggregate_alignment_specs: tuple
    macro_level_specs: tuple
    macro_alignment_tolerance_ppm: int
    mosaic_requests: tuple
    discrete_mosaics: tuple
    mosaic_guest_slots: tuple
    mosaic_boundary_joins: tuple
    fact_authority: CompositeIntentFactAuthority | None
    projected_program: object
    mosaic_scope_by_logical: tuple = ()
    native_witness: object = None

    def __init__(self, *args, **kwargs):
        raise TypeError("RowSourceProjection requires private producer capture")

    def payload(self):
        authority = self.fact_authority
        payload = {
            "schema": "clay.row-source-projection.v1",
            "source_sha256": self.source_sha256,
            "source_inventory_sha256": self.source_inventory_sha256,
            "authoring_source_sha256": self.authoring_source_sha256,
            "source_cell_name": self.source_cell_name,
            "topology_sha256": self.topology_sha256,
            "device_records": json.loads(self.device_records_bytes),
            "tech_identity": self.tech_identity,
            "reference_overlay": self.reference_overlay.canonical_data(),
            "reference_asir": self.reference_asir.canonical_data(),
            "scope_by_logical": self.scope_by_logical,
            "aggregate_alignment_specs": tuple(asdict(row) for row in self.aggregate_alignment_specs),
            "macro_level_specs": tuple(asdict(row) for row in self.macro_level_specs),
            "macro_alignment_tolerance_ppm": self.macro_alignment_tolerance_ppm,
            "mosaic_requests": self.mosaic_requests,
            "discrete_mosaics": tuple(row.canonical_data() for row in self.discrete_mosaics),
            "mosaic_guest_slots": tuple(row.canonical_data() for row in self.mosaic_guest_slots),
            "fact_authority": None if authority is None else {
                "fact_set": authority.fact_set.canonical_data(),
                "source_inventory": authority.source_inventory.canonical_data(),
                "entailments": tuple(asdict(row) for row in authority.entailments),
                "components": authority.component_authority_sha256,
                "receipt": authority.receipt.canonical_data(),
            },
            "program": self.projected_program.payload(),
            # RowProgram serialization intentionally omits unconsumed members.
            # This source binding also preserves the complete compiler inventory.
            "program_members": tuple(row.payload() for row in self.projected_program.members),
        }
        if self.mosaic_boundary_joins:
            payload['schema'] = 'clay.row-source-projection.v2'
            payload['mosaic_boundary_joins'] = tuple(row.canonical_data() for row in self.mosaic_boundary_joins)
        if self.mosaic_scope_by_logical:
            payload['schema'] = 'clay.row-source-projection.v3'
            payload['mosaic_scope_by_logical'] = self.mosaic_scope_by_logical
        if self.native_witness is not None:
            payload['schema'] = 'clay.row-source-projection.v4'
            payload['native_pcell_witness_sha256'] = self.native_witness.sha256
        return payload

    @property
    def sha256(self):
        return hashlib.sha256(_bytes(self.payload())).hexdigest()


@dataclass(frozen=True)
class _CapturedSource:
    content: bytes
    overlay: object
    asir: object
    program: object
    fact_authority: object
    authoring_source_cell: object
    source_cir: str | None
    native_witness: object


_CAPTURED_SOURCES: WeakKeyDictionary = WeakKeyDictionary()


def require_row_source_projection(value, *, program=None):
    """Check captured origin/content without accepting a reconstructed record."""
    from clay.layout.place.row_program import RowProgram

    _require(type(value) is RowSourceProjection, "ORIGIN_TYPE")
    captured = _CAPTURED_SOURCES.get(value)
    _require(type(captured) is _CapturedSource, "ORIGIN_MISSING")
    _require(value.reference_overlay is captured.overlay
             and value.reference_asir is captured.asir
             and value.projected_program is captured.program
             and value.fact_authority is captured.fact_authority
             and value.authoring_source_cell is captured.authoring_source_cell
             and value.native_witness is captured.native_witness
             and value.source_cir == captured.source_cir, "ORIGIN_IDENTITY")
    _require(_bytes(value.payload()) == captured.content, "CONTENT_CHANGED")
    if program is not None:
        _require(type(program) is RowProgram
                 and program.members == value.projected_program.members
                 and program.to_canonical_bytes() == value.projected_program.to_canonical_bytes(),
                 "PROGRAM_MISMATCH")
    return value


def _replay_source_operands(value, canonical_source_cell, tech, *, trust_fresh_witness=None):
    from clay.layout.place.row_program import _group_map, compile_row_program

    source_sha, inventory_sha, canonical = _source(canonical_source_cell)
    _require((source_sha, inventory_sha, canonical.local_name)
             == (value.source_sha256, value.source_inventory_sha256, value.source_cell_name),
             "SOURCE_MISMATCH")
    authoring_sha, authoring_inventory, authoring = _source(value.authoring_source_cell)
    _require(authoring_sha == value.authoring_source_sha256
             and authoring_inventory == inventory_sha, "AUTHORING_SOURCE_MISMATCH")
    _require(_group_map(canonical) == _group_map(authoring), "AUTHORING_GROUP_MISMATCH")
    _require(_tech_identity(tech) == value.tech_identity, "TECH_MISMATCH")
    records = None
    if value.source_cir is not None:
        from clay.layout.generic_placement import parse_cir_mos_records

        path = Path(value.source_cir)
        before = path.read_bytes()
        records = parse_cir_mos_records(path)
        _require(before == path.read_bytes()
                 and hashlib.sha256(before).hexdigest() == value.topology_sha256,
                 "TOPOLOGY_MISMATCH")
    _require(_bytes(records) == value.device_records_bytes, "TOPOLOGY_RECORDS_MISMATCH")
    authority = value.fact_authority
    if authority is not None:
        authority.__post_init__()
        _require(authority.fact_set.source_sha256 == authoring_sha
                 and authority.fact_set.topology_sha256 == value.topology_sha256
                 and authority.fact_set.pdk_snapshot_sha256 == value.tech_identity[1]
                 and authority.fact_set.pdk_installed_authority_sha256 == value.tech_identity[2],
                 "FACT_AUTHORITY_MISMATCH")
    overlay, asir = value.reference_overlay, value.reference_asir
    _require(type(overlay) is SemanticOverlay and overlay.source_kind in ("reference_gds", "native_pcell"),
             "OVERLAY_AUTHORITY")
    if overlay.source_kind == 'native_pcell':
        if trust_fresh_witness is not None and value.native_witness is trust_fresh_witness:
            from clay.layout.native_diffusion_source import _require_native_pcell_witness_operands
            _require_native_pcell_witness_operands(value.native_witness, cell=canonical, tech=tech,
                                                   overlay=overlay, source_cir=value.source_cir)
        else:
            from clay.layout.native_diffusion_source import require_native_pcell_witness
            require_native_pcell_witness(value.native_witness, cell=canonical, tech=tech,
                                         overlay=overlay, source_cir=value.source_cir)
    else:
        _require(value.native_witness is None, 'NATIVE_WITNESS_UNEXPECTED')
    _require(type(asir) is AbstractSpatialIR and overlay.topology_sha256 == value.topology_sha256,
             "ASIR_AUTHORITY")
    replayed_asir = derive_abstract_spatial_ir(overlay,
        scope_by_logical=value.scope_by_logical,
        mosaic_scope_by_logical=value.mosaic_scope_by_logical,
        aggregate_alignment_specs=value.aggregate_alignment_specs,
        macro_level_specs=value.macro_level_specs,
        macro_alignment_tolerance_ppm=value.macro_alignment_tolerance_ppm)
    _require(replayed_asir.canonical_bytes == asir.canonical_bytes, "ASIR_REPLAY_MISMATCH")
    mosaics = replay_discrete_mosaic_inventory(tuple(project_discrete_mosaic(
        replayed_asir, mosaic_id=name, logical_owners=owners)
        for name, owners in value.mosaic_requests))
    _require(mosaics == replay_discrete_mosaic_inventory(value.discrete_mosaics),
             "MOSAIC_REPLAY_MISMATCH")
    by_digest = {row.sha256: row for row in mosaics}
    slots = replay_mosaic_guest_slot_inventory(value.mosaic_guest_slots)
    for slot in slots:
        mosaic = by_digest.get(slot.source_mosaic_sha256)
        _require(mosaic is not None, "SLOT_MOSAIC_MISMATCH")
        guests = tuple(row.logical_owner for row in overlay.occurrences if row.node_id == slot.guest_node)
        _require(len(guests) == 1, "SLOT_GUEST_MISMATCH")
        _require(derive_mosaic_guest_slot(overlay=overlay, asir=replayed_asir,
                 mosaic=mosaic, guest_owner=guests[0]) == slot, "SLOT_REPLAY_MISMATCH")
    intent = layout_intent_from_cell_def(canonical)
    authoring_intent = layout_intent_from_cell_def(authoring)
    for field in ("common_centroids", "arrays", "dummy_flanks", "matching_intents"):
        _require(getattr(intent, field) == getattr(authoring_intent, field),
                 "AUTHORING_DECLARATION_MISMATCH")
    # Source declarations remain intact; only independently derived additions enter.
    boundary_joins = replay_mosaic_boundary_join_inventory(value.mosaic_boundary_joins, mosaics=mosaics)
    projected = replace(intent, discrete_mosaics=mosaics, mosaic_guest_slots=slots,
                        mosaic_boundary_joins=boundary_joins)
    program = compile_row_program(canonical, projected, tech, device_records=records)
    _require(not program.residuals, "PROGRAM_RESIDUALS")
    _require(program.members == value.projected_program.members
             and program.to_canonical_bytes() == value.projected_program.to_canonical_bytes(),
             "PROGRAM_REPLAY_MISMATCH")
    return program


def replay_row_source_projection(value, canonical_source_cell, tech, *, trust_fresh_witness=None):
    require_row_source_projection(value)
    return _replay_source_operands(value, canonical_source_cell, tech, trust_fresh_witness=trust_fresh_witness)


def require_actual_row_source_topology(value, source_cir):
    """Bind the actual compiler operand, including records outside row ownership."""
    require_row_source_projection(value)
    if value.source_cir is None:
        _require(source_cir is None, "ACTUAL_TOPOLOGY_PRESENCE")
        return None
    _require(source_cir is not None, "ACTUAL_TOPOLOGY_PRESENCE")
    try:
        path = Path(source_cir).resolve(strict=True)
        _require(str(path) == value.source_cir, "ACTUAL_TOPOLOGY_PATH")
        before = path.read_bytes()
        from clay.layout.generic_placement import parse_cir_mos_records

        records = parse_cir_mos_records(path)
        _require(before == path.read_bytes()
                 and hashlib.sha256(before).hexdigest() == value.topology_sha256,
                 "ACTUAL_TOPOLOGY_BYTES")
        _require(_bytes(records) == value.device_records_bytes, "ACTUAL_TOPOLOGY_RECORDS")
    except (OSError, TypeError, UnicodeError) as exc:
        raise RowSourceProjectionError("ROW_SOURCE_ACTUAL_TOPOLOGY_UNREADABLE") from exc
    return records


def _capture_row_source_projection(*, canonical_source_cell, authoring_source_cell,
    tech, reference_overlay, reference_asir, mosaic_requests, discrete_mosaics,
    source_cir=None, scope_by_logical=(), aggregate_alignment_specs=(),
    macro_level_specs=(), macro_alignment_tolerance_ppm=5_000,
    mosaic_guest_slots=(), fact_authority=None, mosaic_boundary_joins=(), mosaic_scope_by_logical=(),
    native_witness=None):
    """Capture only at the trusted producer after its fresh source adapter succeeds.

    This internal boundary is deliberately not a generic adapter callback or a
    deserialization API. Its caller owns the actual original-source extraction.
    """
    from clay.layout.place.row_program import compile_row_program

    _require(type(reference_overlay) is SemanticOverlay and type(reference_asir) is AbstractSpatialIR,
             "CAPTURE_TYPES")
    _require(type(mosaic_requests) is tuple and bool(mosaic_requests)
             and all(type(row) is tuple and len(row) == 2
                     and type(row[0]) is str and row[0]
                     and type(row[1]) is tuple and bool(row[1])
                     and all(type(owner) is str and owner for owner in row[1])
                     and len(set(row[1])) == len(row[1]) for row in mosaic_requests)
             and len({row[0] for row in mosaic_requests}) == len(mosaic_requests), "MOSAIC_REQUESTS")
    _require(type(scope_by_logical) is tuple
             and all(type(row) is tuple and len(row) == 2
                     and all(type(item) is str and item for item in row) for row in scope_by_logical),
             "SCOPE_ARGUMENTS")
    _require(type(aggregate_alignment_specs) is tuple
             and all(type(row) is AggregateAlignmentSpec for row in aggregate_alignment_specs)
             and type(macro_level_specs) is tuple
             and all(type(row) is MacroDagLevelSpec for row in macro_level_specs), "DERIVATION_ARGUMENTS")
    _require(type(macro_alignment_tolerance_ppm) is int
             and 0 <= macro_alignment_tolerance_ppm <= 1_000_000, "DERIVATION_TOLERANCE")
    _require(fact_authority is None or type(fact_authority) is CompositeIntentFactAuthority,
             "FACT_AUTHORITY_TYPE")
    source_sha, inventory_sha, canonical = _source(canonical_source_cell)
    authoring_sha, _authoring_inventory, authoring = _source(authoring_source_cell)
    mosaics = replay_discrete_mosaic_inventory(discrete_mosaics)
    slots = replay_mosaic_guest_slot_inventory(mosaic_guest_slots)
    boundary_joins = replay_mosaic_boundary_join_inventory(mosaic_boundary_joins, mosaics=mosaics)
    records, cir_path = None, None
    topology_sha = reference_overlay.topology_sha256
    if source_cir is not None:
        from clay.layout.generic_placement import parse_cir_mos_records

        path = Path(source_cir).resolve(strict=True)
        before = path.read_bytes()
        records = parse_cir_mos_records(path)
        _require(before == path.read_bytes(), "TOPOLOGY_CHANGED_DURING_CAPTURE")
        topology_sha = hashlib.sha256(before).hexdigest()
        cir_path = str(path)
    projected_intent = replace(layout_intent_from_cell_def(canonical),
        discrete_mosaics=mosaics, mosaic_guest_slots=slots, mosaic_boundary_joins=boundary_joins)
    program = compile_row_program(canonical, projected_intent, tech, device_records=records)
    value = object.__new__(RowSourceProjection)
    fields = dict(source_sha256=source_sha, source_inventory_sha256=inventory_sha,
        authoring_source_sha256=authoring_sha, authoring_source_cell=authoring,
        source_cell_name=canonical.local_name, source_cir=cir_path,
        topology_sha256=topology_sha, device_records_bytes=_bytes(records), tech_identity=_tech_identity(tech),
        reference_overlay=reference_overlay, reference_asir=reference_asir,
        scope_by_logical=scope_by_logical, aggregate_alignment_specs=aggregate_alignment_specs,
        mosaic_scope_by_logical=mosaic_scope_by_logical,
        macro_level_specs=macro_level_specs, macro_alignment_tolerance_ppm=macro_alignment_tolerance_ppm,
        mosaic_requests=mosaic_requests, discrete_mosaics=mosaics, mosaic_guest_slots=slots,
        mosaic_boundary_joins=boundary_joins,
        fact_authority=fact_authority, projected_program=program, native_witness=native_witness)
    for name, item in fields.items():
        object.__setattr__(value, name, item)
    _replay_source_operands(value, canonical, tech)
    _CAPTURED_SOURCES[value] = _CapturedSource(_bytes(value.payload()),
        reference_overlay, reference_asir, program, fact_authority, authoring, cir_path, native_witness)
    return value


def verify_row_source_fact_authority(value, *, fact_set, source_inventory):
    """Join the retained authoring authority to the current final-flow denominator."""
    require_row_source_projection(value)
    authority = value.fact_authority
    if authority is None:
        _require(fact_set is None and source_inventory is None, "FACT_AUTHORITY_UNEXPECTED")
        return
    _require(fact_set is not None and source_inventory is not None, "FACT_AUTHORITY_MISSING")
    _require(fact_set.canonical_data() == authority.fact_set.canonical_data()
             and source_inventory.canonical_data() == authority.source_inventory.canonical_data(),
             "FACT_AUTHORITY_MISMATCH")


def authenticate_serialized_row_source_projection(
    value,
    *,
    canonical_source_cell,
    tech,
    expected_source_sha256: str,
    expected_topology_sha256: str,
    expected_pdk_provenance_sha256: str,
    expected_payload_sha256: str,
    authoring_source_cell=None,
    source_cir: str | Path | None = None,
    native_witness=None,
):
    """Dual-admission authenticator for a serialized RowSourceProjection.

    Re-verifies live .clay SHA-256, live .cir topology SHA-256, live LayoutTech
    provenance digest, canonical projection payload SHA-256, and the complete
    _replay_source_operands compiler derivation before admitting the instance.
    """
    from clay.layout.route.resource_world import pdk_provenance_digest

    _require(type(value) is RowSourceProjection, "ORIGIN_TYPE")
    source_sha, inventory_sha, canonical = _source(canonical_source_cell)
    _require(
        source_sha == expected_source_sha256
        and source_sha == value.source_sha256
        and inventory_sha == value.source_inventory_sha256,
        "SOURCE_MISMATCH",
    )
    target_authoring_input = (
        authoring_source_cell
        if authoring_source_cell is not None
        else (
            canonical
            if value.authoring_source_sha256 == source_sha
            else value.authoring_source_cell
        )
    )
    authoring_sha, authoring_inventory, authoring = _source(target_authoring_input)
    _require(
        authoring_sha == value.authoring_source_sha256
        and authoring_inventory == inventory_sha,
        "AUTHORING_SOURCE_MISMATCH",
    )
    object.__setattr__(value, "authoring_source_cell", authoring)

    _require(_tech_identity(tech) == value.tech_identity, "TECH_MISMATCH")
    _require(
        pdk_provenance_digest(tech) == expected_pdk_provenance_sha256,
        "PDK_PROVENANCE_MISMATCH",
    )

    resolved_cir: str | None = value.source_cir
    if source_cir is not None:
        resolved_cir = str(Path(source_cir).resolve(strict=True))
        object.__setattr__(value, "source_cir", resolved_cir)
    if resolved_cir is not None:
        cir_bytes = Path(resolved_cir).read_bytes()
        cir_sha = hashlib.sha256(cir_bytes).hexdigest()
        _require(
            cir_sha == expected_topology_sha256 and cir_sha == value.topology_sha256,
            "TOPOLOGY_MISMATCH",
        )
    else:
        _require(
            value.topology_sha256 == expected_topology_sha256,
            "TOPOLOGY_MISMATCH",
        )

    if native_witness is not None:
        _require(
            type(value.reference_overlay) is SemanticOverlay
            and value.reference_overlay.canonical_data()
            == native_witness.overlay.canonical_data(),
            "OVERLAY_AUTHORITY",
        )
        object.__setattr__(value, "reference_overlay", native_witness.overlay)
        object.__setattr__(value, "native_witness", native_witness)

    _require(value.sha256 == expected_payload_sha256, "PAYLOAD_DIGEST_MISMATCH")
    _replay_source_operands(value, canonical, tech, trust_fresh_witness=native_witness)
    _CAPTURED_SOURCES[value] = _CapturedSource(
        _bytes(value.payload()),
        value.reference_overlay,
        value.reference_asir,
        value.projected_program,
        value.fact_authority,
        value.authoring_source_cell,
        value.source_cir,
        value.native_witness,
    )
    return require_row_source_projection(value)


__all__ = ["RowSourceProjection", "RowSourceProjectionError",
           "authenticate_serialized_row_source_projection",
           "replay_row_source_projection",
           "require_row_source_projection", "require_actual_row_source_topology",
           "verify_row_source_fact_authority"]
