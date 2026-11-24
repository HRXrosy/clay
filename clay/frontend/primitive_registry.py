"""Canonical primitive taxonomy for the three-reference convergence domain.

This module is deliberately a registry, not an optimizer.  It freezes the
coordinate-free vocabulary and the minimum human-reference structure patterns
that must eventually be materialized and independently checked on exact GDS.
Nothing in this file is permission to call a pattern covered: Phase 2 coverage
requires separate, artifact-bound ``mapped && materialized && checked`` rows.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, Sequence, Tuple


DIMENSIONS: Tuple[str, ...] = (
    "device_geometry",
    "relative_placement",
    "grouping_partition",
    "density",
    "power_shielding",
    "thermal_stress",
)

TaxonomyStatus = Literal["existing", "gap"]
MaterializerStatus = Literal["materialized", "partial", "missing"]
CheckerStatus = Literal["exact_gds", "model_only", "missing"]
ReferenceScope = Literal[
    "complete_exact_gds",
    "complete_unmapped_gds",
    "oa_spatial_only",
]
PatternBasis = Literal[
    "complete_gds_observation",
    "oa_spatial_observation",
    "topology_declaration",
    "audit_gap",
]


class PrimitiveRegistryError(ValueError):
    """The taxonomy or its reference bindings are not closed."""


_MAX_TEXT = 4_096
_MAX_EVIDENCE = 64
_PRIMITIVE_ID_RE = re.compile(r"[a-z][a-z0-9_]*\Z")
_PATTERN_ID_RE = re.compile(r"[a-z][a-z0-9_.]*\Z")


def _exact_text(value: object, *, field: str, pattern: re.Pattern[str] | None = None) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise PrimitiveRegistryError(f"{field} must be an exact non-empty string")
    if len(value) > _MAX_TEXT:
        raise PrimitiveRegistryError(f"{field} exceeds {_MAX_TEXT} characters")
    if pattern is not None and pattern.fullmatch(value) is None:
        raise PrimitiveRegistryError(f"{field} is not canonical")
    return value


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_relative_evidence(path: str) -> None:
    if type(path) is not str or not path or path != path.strip() or "\\" in path:
        raise PrimitiveRegistryError("evidence paths must be non-empty repository-relative POSIX paths")
    if len(path) > _MAX_TEXT:
        raise PrimitiveRegistryError("evidence path is unbounded")
    pure = PurePosixPath(path)
    if pure.is_absolute() or ".." in pure.parts:
        raise PrimitiveRegistryError("evidence paths must stay inside the repository")
    if pure.suffix.lower() == ".png":
        raise PrimitiveRegistryError("PNG is forbidden as primitive or reference evidence")


def _exact_evidence(value: object, *, field: str = "evidence") -> Tuple[str, ...]:
    if type(value) is not tuple:
        raise PrimitiveRegistryError(f"{field} must be an exact tuple")
    if not value or len(value) > _MAX_EVIDENCE:
        raise PrimitiveRegistryError(f"{field} must be non-empty and bounded")
    for path in value:
        _validate_relative_evidence(path)
    if len(set(value)) != len(value):
        raise PrimitiveRegistryError(f"{field} contains duplicate paths")
    return value


@dataclass(frozen=True)
class PrimitiveDefinition:
    primitive_id: str
    dimensions: Tuple[str, ...]
    surface: str
    taxonomy_status: TaxonomyStatus
    materializer_status: MaterializerStatus
    checker_status: CheckerStatus
    gap: str | None
    evidence: Tuple[str, ...]

    def __post_init__(self) -> None:
        _exact_text(
            self.primitive_id,
            field="primitive id",
            pattern=_PRIMITIVE_ID_RE,
        )
        if (
            type(self.dimensions) is not tuple
            or not self.dimensions
            or len(self.dimensions) > len(DIMENSIONS)
            or len(set(self.dimensions)) != len(self.dimensions)
            or any(type(item) is not str or item not in DIMENSIONS for item in self.dimensions)
        ):
            raise PrimitiveRegistryError("primitive dimensions must be an exact unique tuple")
        _exact_text(self.surface, field="primitive surface")
        if type(self.taxonomy_status) is not str or self.taxonomy_status not in (
            "existing", "gap"
        ):
            raise PrimitiveRegistryError("unknown primitive taxonomy status")
        if type(self.materializer_status) is not str or self.materializer_status not in (
            "materialized", "partial", "missing"
        ):
            raise PrimitiveRegistryError("unknown primitive materializer status")
        if type(self.checker_status) is not str or self.checker_status not in (
            "exact_gds", "model_only", "missing"
        ):
            raise PrimitiveRegistryError("unknown primitive checker status")
        if self.gap is not None:
            _exact_text(self.gap, field="primitive gap")
        if self.taxonomy_status == "gap" and self.gap is None:
            raise PrimitiveRegistryError("gap primitive lacks a gap statement")
        _exact_evidence(self.evidence)


@dataclass(frozen=True)
class BoundReferenceFile:
    path: str
    sha256: str

    def __post_init__(self) -> None:
        _validate_relative_evidence(self.path)
        if not _is_sha256(self.sha256):
            raise PrimitiveRegistryError("bound reference file requires a lowercase SHA-256")


@dataclass(frozen=True)
class ReferenceScan:
    circuit_id: str
    evidence_scope: ReferenceScope
    gds: BoundReferenceFile
    manifest: BoundReferenceFile
    topology: BoundReferenceFile
    source: BoundReferenceFile
    source_cell_id: str
    scan_method: str
    coordinate_free_spatial_sha256: str | None = None
    oa_spatial_anchor_gds: BoundReferenceFile | None = None

    def __post_init__(self) -> None:
        _exact_text(
            self.circuit_id,
            field="reference scan circuit id",
            pattern=_PRIMITIVE_ID_RE,
        )
        if type(self.evidence_scope) is not str or self.evidence_scope not in (
            "complete_exact_gds", "complete_unmapped_gds", "oa_spatial_only"
        ):
            raise PrimitiveRegistryError("unknown reference scan evidence scope")
        if any(
            type(bound) is not BoundReferenceFile
            for bound in (self.gds, self.manifest, self.topology, self.source)
        ):
            raise PrimitiveRegistryError("reference scan files must be exact bound-file rows")
        if (
            self.oa_spatial_anchor_gds is not None
            and type(self.oa_spatial_anchor_gds) is not BoundReferenceFile
        ):
            raise PrimitiveRegistryError("reference scan OA anchor must be a bound-file row")
        _exact_text(
            self.source_cell_id,
            field="reference scan source cell id",
            pattern=_PRIMITIVE_ID_RE,
        )
        _exact_text(self.scan_method, field="reference scan method")
        if self.coordinate_free_spatial_sha256 is not None and not _is_sha256(
            self.coordinate_free_spatial_sha256
        ):
            raise PrimitiveRegistryError("reference scan spatial digest must be a lowercase SHA-256")


@dataclass(frozen=True)
class ReferencePattern:
    pattern_id: str
    circuit_id: str
    dimension: str
    description: str
    primitive_id: str
    basis: PatternBasis
    evidence: Tuple[str, ...]

    def __post_init__(self) -> None:
        _exact_text(self.pattern_id, field="pattern id", pattern=_PATTERN_ID_RE)
        _exact_text(
            self.circuit_id,
            field="pattern circuit id",
            pattern=_PRIMITIVE_ID_RE,
        )
        if type(self.dimension) is not str or self.dimension not in DIMENSIONS:
            raise PrimitiveRegistryError("pattern dimension is not canonical")
        _exact_text(self.description, field="pattern description")
        _exact_text(
            self.primitive_id,
            field="pattern primitive id",
            pattern=_PRIMITIVE_ID_RE,
        )
        if type(self.basis) is not str or self.basis not in (
            "complete_gds_observation",
            "oa_spatial_observation",
            "topology_declaration",
            "audit_gap",
        ):
            raise PrimitiveRegistryError("pattern basis is not canonical")
        _exact_evidence(self.evidence)


def _primitive(
    primitive_id: str,
    dimensions: Sequence[str],
    surface: str,
    *,
    taxonomy_status: TaxonomyStatus = "existing",
    materializer_status: MaterializerStatus = "materialized",
    checker_status: CheckerStatus = "model_only",
    gap: str | None = None,
    evidence: Sequence[str] = (),
) -> PrimitiveDefinition:
    return PrimitiveDefinition(
        primitive_id=primitive_id,
        dimensions=tuple(dimensions),
        surface=surface,
        taxonomy_status=taxonomy_status,
        materializer_status=materializer_status,
        checker_status=checker_status,
        gap=gap,
        evidence=tuple(evidence),
    )


PRIMITIVES: Tuple[PrimitiveDefinition, ...] = (
    # 1. Device geometry.
    _primitive(
        "fold", ("device_geometry",), "@fold",
        evidence=("clay/layout/annotate_to_solver.py",),
    ),
    _primitive(
        "array", ("device_geometry", "grouping_partition"), "@array",
        evidence=("clay/layout/common_centroid_array.py", "tests/test_array_consumption.py"),
    ),
    _primitive(
        "equal_size", ("device_geometry",), "@equal_size",
        checker_status="exact_gds",
        evidence=("clay/layout/completeness/gds_invariants.py",),
    ),
    _primitive(
        "dense_contacts", ("device_geometry", "power_shielding"), "@dense_contacts",
        evidence=("clay/layout/intent.py",),
    ),
    _primitive(
        "split", ("device_geometry", "grouping_partition"), "@split",
        evidence=("clay/layout/intent.py", "tests/test_passive_split_materialization.py"),
    ),
    _primitive(
        "orient", ("device_geometry",), "@orient",
        evidence=("clay/layout/annotations.py", "tests/test_device_orient.py"),
    ),
    _primitive(
        "bulk", ("device_geometry", "power_shielding"), "@bulk",
        evidence=("clay/layout/layout_plan.py", "tests/test_layout_plan_reconcile.py"),
    ),
    # 2. Relative placement.
    _primitive(
        "matching", ("relative_placement",), "@matching",
        checker_status="exact_gds",
        evidence=("clay/layout/intent_obligations.py", "clay/layout/completeness/gds_invariants.py"),
    ),
    _primitive(
        "mirror", ("relative_placement",), "@mirror",
        evidence=("clay/layout/intent.py",),
    ),
    _primitive(
        "symmetric", ("relative_placement",), "@symmetric",
        evidence=("clay/layout/annotate_to_solver.py",),
    ),
    _primitive(
        "abut", ("relative_placement",), "@abut",
        evidence=("clay/layout/annotate_to_solver.py", "tests/test_abut_shared_od_placement.py"),
    ),
    _primitive(
        "order", ("relative_placement",), "@order",
        evidence=("clay/layout/intent_obligations.py",),
    ),
    _primitive(
        "interdigitation", ("relative_placement",), "@interdigitated",
        checker_status="exact_gds",
        evidence=(
            "clay/layout/common_centroid_array.py",
            "clay/layout/completeness/gds_invariants.py",
            "tests/test_final_analog_intent.py",
        ),
    ),
    _primitive(
        "common_centroid", ("relative_placement",), "@common_centroid",
        checker_status="exact_gds",
        evidence=(
            "clay/layout/common_centroid_array.py",
            "clay/layout/completeness/gds_invariants.py",
            "tests/test_final_analog_intent.py",
        ),
    ),
    _primitive(
        "adjacent", ("relative_placement",), "@adjacent",
        checker_status="exact_gds",
        evidence=(
            "clay/layout/intent.py",
            "clay/layout/place_blocks_2d.py",
            "clay/layout/completeness/adjacent_check.py",
            "tests/test_adjacent_exact_intent.py",
            "tests/test_final_analog_intent.py",
        ),
    ),
    _primitive(
        "rail_affinity", ("relative_placement", "power_shielding"), "@rail_affinity",
        checker_status="exact_gds",
        evidence=(
            "clay/layout/place/rail_affinity_lower.py",
            "clay/layout/completeness/gds_witness.py",
            "clay/layout/completeness/gds_invariants.py",
            "clay/layout/final_layout_intent.py",
            "tests/test_final_analog_intent.py",
        ),
    ),
    _primitive(
        "match_length", ("relative_placement", "power_shielding"), "@match_length",
        evidence=("clay/layout/intent_obligations.py",),
    ),
    _primitive(
        "finger_array_order",
        ("relative_placement", "grouping_partition"),
        "@array(style=shared_diffusion, rows=1, order=ratio)",
        checker_status="exact_gds",
        evidence=(
            "clay/layout/intent.py",
            "clay/layout/place/row_program.py",
            "clay/layout/completeness/gds_invariants.py",
            "tests/test_final_analog_intent.py",
            "tests/test_p2_analog_macro_synthesis.py",
        ),
    ),
    # 3. Grouping and partition.
    _primitive(
        "well_island", ("grouping_partition",), "@well_island",
        evidence=("clay/layout/intent.py", "clay/layout/extract/well_islands.py"),
    ),
    _primitive(
        "band", ("grouping_partition", "power_shielding"), "@band",
        checker_status="exact_gds",
        evidence=(
            "clay/layout/place_blocks_2d.py",
            "clay/layout/completeness/gds_invariants.py",
            "tests/test_final_analog_intent.py",
        ),
    ),
    _primitive(
        "bank", ("grouping_partition",), "@place_group / bank role",
        checker_status="model_only",
        evidence=("clay/frontend/reference_spatial_intent.py", "clay/layout/intent_obligations.py"),
    ),
    _primitive(
        "place_group", ("grouping_partition",), "@place_group",
        evidence=("clay/layout/intent.py", "clay/layout/intent_obligations.py"),
    ),
    _primitive(
        "dummy_flank", ("grouping_partition",), "@dummy_flank",
        evidence=("clay/layout/common_centroid_array.py",),
    ),
    _primitive(
        "passive_bank_partition", ("grouping_partition",), "@passive_bank_partition",
        checker_status="exact_gds",
        evidence=(
            "clay/layout/intent.py",
            "clay/layout/physical_macro_graph.py",
            "clay/layout/nonrow_macro_plan.py",
            "clay/layout/completeness/gds_invariants.py",
            "tests/test_passive_bank_partition.py",
        ),
    ),
    _primitive(
        "well_island_sharing", ("grouping_partition",), "@well_island_share",
        checker_status="exact_gds",
        evidence=(
            "clay/layout/intent.py",
            "clay/layout/well_island_sharing.py",
            "clay/layout/route/lvs_gate.py",
            "clay/layout/completeness/gds_invariants.py",
            "tests/test_well_island_sharing.py",
        ),
    ),
    _primitive(
        "well_isolated", ("grouping_partition", "power_shielding"), "@well_isolated",
        evidence=("clay/layout/intent.py",),
    ),
    # 4. Density.
    _primitive(
        "density_whitespace_target", ("density",), "@density",
        checker_status="exact_gds",
        evidence=(
            "clay/layout/intent.py",
            "clay/layout/flow_placement.py",
            "clay/layout/completeness/gds_witness.py",
            "clay/layout/completeness/gds_invariants.py",
            "clay/layout/final_layout_intent.py",
            "tests/test_density_whitespace_intent.py",
            "tests/test_final_analog_intent.py",
        ),
    ),
    _primitive(
        "od_fill_target", ("density",), "@od_fill",
        checker_status="exact_gds",
        evidence=(
            "clay/layout/intent.py",
            "clay/layout/flow_placement.py",
            "clay/layout/place/final_placement_state.py",
            "clay/layout/completeness/gds_witness.py",
            "clay/layout/completeness/gds_invariants.py",
            "clay/layout/final_layout_intent.py",
            "tests/test_od_fill_intent.py",
            "tests/test_final_analog_intent.py",
        ),
    ),
    # 5. Power and shielding.
    _primitive(
        "rail", ("power_shielding",), "@rail / edge @trunk",
        evidence=("clay/layout/route/rail_plan.py",),
    ),
    _primitive(
        "trunk", ("power_shielding",), "@trunk",
        evidence=("clay/layout/route/route_cell.py",),
    ),
    _primitive(
        "power_plane", ("power_shielding",), "@power_plane",
        checker_status="exact_gds",
        evidence=(
            "clay/layout/route/power_plane.py",
            "clay/layout/completeness/power_plane_check.py",
            "tests/test_power_plane_materializer.py",
            "tests/test_final_analog_intent.py",
        ),
    ),
    _primitive(
        "wire_width", ("power_shielding",), "@wire_width",
        evidence=("clay/layout/intent_obligations.py",),
    ),
    _primitive(
        "guard_ring", ("power_shielding", "grouping_partition"),
        "@guard_ring(group, well, tie)",
        evidence=("clay/layout/route/guard_ring_check.py", "clay/layout/intent_elv.py"),
    ),
    _primitive(
        "guard_ring_tight", ("power_shielding", "grouping_partition"),
        "@guard_ring(hug=tight)", checker_status="exact_gds",
        evidence=(
            "clay/layout/route/route_cell.py",
            "clay/layout/completeness/gds_invariants.py",
            "tests/test_final_analog_intent.py",
        ),
    ),
    _primitive(
        "tap", ("power_shielding",), "@tap",
        evidence=("clay/layout/bjt_device_cell.py",),
    ),
    _primitive(
        "shield", ("power_shielding",), "@shield",
        checker_status="exact_gds",
        evidence=(
            "clay/layout/route/shield.py",
            "clay/layout/completeness/gds_witness.py",
            "clay/layout/completeness/shield_check.py",
            "clay/layout/final_layout_intent.py",
            "tests/test_shield_materialization.py",
            "tests/test_final_analog_intent.py",
            "tests/test_reference_ldo_shield.py",
        ),
    ),
    _primitive(
        "kelvin", ("power_shielding",), "@kelvin(force, sense, target)",
        checker_status="exact_gds",
        evidence=(
            "clay/layout/intent.py",
            "clay/layout/route/topology_compile.py",
            "clay/layout/completeness/kelvin_check.py",
            "tests/test_kelvin_intent.py",
            "tests/test_topology_compile.py",
            "tests/test_final_analog_intent.py",
        ),
    ),
    _primitive(
        "channel", ("power_shielding",),
        "@channel(orient=vertical, common-centroid owner)",
        checker_status="exact_gds",
        evidence=(
            "clay/layout/place_blocks_2d.py",
            "clay/layout/completeness/gds_invariants.py",
            "tests/test_final_analog_intent.py",
        ),
    ),
    _primitive(
        "anchor", ("power_shielding", "relative_placement"), "@anchor",
        checker_status="exact_gds",
        evidence=(
            "clay/layout/place_blocks_2d.py",
            "clay/layout/completeness/gds_invariants.py",
            "tests/test_final_analog_intent.py",
        ),
    ),
    _primitive(
        "power_sensitive_net_budget", ("power_shielding",),
        "@power_budget / @sensitive_net_budget",
        checker_status="exact_gds",
        evidence=(
            "clay/layout/intent.py",
            "clay/layout/route/topology_compile.py",
            "clay/layout/pex/electrical_budget.py",
            "clay/flow/step.py",
            "tests/test_power_sensitive_net_budget.py",
            "tests/test_electrical_budget_pex.py",
        ),
    ),
    # 6. Thermal and stress.
    _primitive(
        "thermal_stress_neighborhood", ("thermal_stress",),
        "@thermal_neighborhood / @stress_neighborhood",
        taxonomy_status="gap", materializer_status="missing", checker_status="missing",
        gap="No coordinate-free thermal/stress neighborhood primitive or exact checker exists.",
        evidence=("docs/audits/2026-08-11-three-reference-physical-design-gap-audit.md",),
    ),
    _primitive(
        "equal_well_edge", ("thermal_stress",), "@equal_well_edge",
        evidence=("clay/layout/intent_certify.py", "tests/test_lde_certify_e2e.py"),
    ),
    _primitive(
        "equal_od", ("thermal_stress",), "@equal_od",
        evidence=("clay/layout/intent_certify.py", "tests/test_lde_certify_e2e.py"),
    ),
    _primitive(
        "equal_od_spacing", ("thermal_stress",), "@equal_od_spacing",
        evidence=("clay/layout/intent_certify.py", "tests/test_lde_certify_e2e.py"),
    ),
)


REFERENCE_SCANS: Tuple[ReferenceScan, ...] = (
    ReferenceScan(
        circuit_id="two_stage_ota",
        evidence_scope="complete_exact_gds",
        gds=BoundReferenceFile(
            "ota_dev/ref/reference_layout.gds",
            "51d1c73b0f934885e9c5db84bb6884be7f8b5516f587720a9f271ad9d10b8d78",
        ),
        manifest=BoundReferenceFile(
            "ota_dev/ref/reference_manifest.json",
            "14f56602bdfb292a78c9534dd6fe4adf6a6049596486d5d5012c48cf18f9baaf",
        ),
        topology=BoundReferenceFile(
            "ota_dev/ref/reference_extracted_clean.cir",
            "8b25a0a2da701af2b7fa1ac4c5a3fc9e67c8997c81235868e8237065848472be",
        ),
        source=BoundReferenceFile(
            "ota_dev/source/ref_ota_tsmc.clay",
            "97ad40acc62a7bd5e6a71294d85da96c5699693f5fbde45c17c17d30e35c7b6f",
        ),
        source_cell_id="ota_cs_ref",
        scan_method="complete GDS plus instance map and extracted netlist; coordinate-free craft classification",
        coordinate_free_spatial_sha256="947e930dd387b6bb89e604aceefe0f917f9a57461ee7887460177b7ab69a8714",
    ),
    ReferenceScan(
        circuit_id="reference_ldo",
        evidence_scope="complete_unmapped_gds",
        gds=BoundReferenceFile(
            "ldo_dev/ref/reference_layout_flat.gds",
            "cb8a06166a3eb6005a5538b8596e1cc38df7482f334d97d5ae56327afacd887e",
        ),
        manifest=BoundReferenceFile(
            "ldo_dev/ref/reference_manifest.json",
            "bfea51ff58bcdff69470bb249aa7e2c2e84ea3e429f6c22561f50c5790490b1f",
        ),
        topology=BoundReferenceFile(
            "ldo_dev/ref/reference_topology.cir",
            "a0bac29cb5026ece0c627c136c76b34cbd5c3375228f0f2fee4ba20bb5d24939",
        ),
        source=BoundReferenceFile(
            "ldo_dev/source/reference_ldo.clay",
            "a0009ebd045fe9db7779822b8660a5a438821c632c77b4bd9f309d128e3b0b6d",
        ),
        source_cell_id="ldo1",
        scan_method="complete GDS master-family relations plus normalized topology; logical instance map unmeasurable",
        coordinate_free_spatial_sha256="2c88e515a214ffd3822d18118064993bfb64184f8719be6ca1abbf739dc13474",
        oa_spatial_anchor_gds=BoundReferenceFile(
            "ldo_dev/ref/reference_layout_full.gds",
            "c11e703aa5f29e2d6bc0b69ba2a160b3a625454207db83ff33686d0daaaca629",
        ),
    ),
    ReferenceScan(
        circuit_id="bandgap",
        evidence_scope="complete_unmapped_gds",
        gds=BoundReferenceFile(
            "bandgap_dev/ref/reference_layout_flat.gds",
            "05612d321ffde7d43d13379bffd1b9fccce3c9fdfed3eca695f5fa7f51f09824",
        ),
        manifest=BoundReferenceFile(
            "bandgap_dev/ref/reference_manifest.json",
            "15b15acd010ce06a2dca2dc06ccc7d2f3bb82d03e400ab5c98930b2f2dc2ed44",
        ),
        topology=BoundReferenceFile(
            "bandgap_dev/ref/reference_topology.cir",
            "cbe7bee15543c907f65937759dd6dd3ac6e6e7fb88446e8ddf9abaacf68b02c3",
        ),
        source=BoundReferenceFile(
            "bandgap_dev/source/reference_bandgap.clay",
            "8514cb1b69d9a80c3a6df211f2babec4bbe084f6f1ee7de338639be4ee7a1b7b",
        ),
        source_cell_id="bandgap",
        scan_method="flattened layer geometry plus OA-anchor master-family relations; logical instance map unavailable",
        coordinate_free_spatial_sha256="20d931323058db60b87480552ba5516209d4d2d55ff03eca6dc8bab2dd0d2a9a",
        oa_spatial_anchor_gds=BoundReferenceFile(
            "bandgap_dev/ref/reference_layout_full.gds",
            "c4a73be27d1b50a3cbc8054219a874cb121820c5882d6f703e8c086d5dc4ab37",
        ),
    ),
)


def _pattern(
    pattern_id: str,
    circuit_id: str,
    dimension: str,
    description: str,
    primitive_id: str,
    *evidence: str,
) -> ReferencePattern:
    if circuit_id == "two_stage_ota":
        basis: PatternBasis = "complete_gds_observation"
    elif primitive_id in {
        "matching", "place_group", "rail_affinity", "tap", "trunk", "well_isolated"
    } or pattern_id in {
        "bandgap.vref_edge_access",
        "ldo.reference_feedback_shield",
        "ldo.reference_kelvin",
    }:
        basis = "topology_declaration"
    elif primitive_id in {
        "channel",
        "finger_array_order",
        "passive_bank_partition",
        "power_sensitive_net_budget",
        "shield",
        "thermal_stress_neighborhood",
        "well_island_sharing",
    }:
        basis = "audit_gap"
    else:
        basis = "oa_spatial_observation"
    return ReferencePattern(
        pattern_id=pattern_id,
        circuit_id=circuit_id,
        dimension=dimension,
        description=description,
        primitive_id=primitive_id,
        basis=basis,
        evidence=tuple(evidence),
    )


_OTA_EVIDENCE = (
    "ota_dev/ref/reference_manifest.json",
    "ota_dev/analysis/reference_craft_spec.md",
)
_LDO_EVIDENCE = (
    "ldo_dev/ref/reference_manifest.json",
    "docs/audits/2026-08-11-three-reference-physical-design-gap-audit.md",
)
_BANDGAP_EVIDENCE = (
    "bandgap_dev/ref/reference_manifest.json",
    "docs/audits/2026-08-11-three-reference-physical-design-gap-audit.md",
)


PHASE1_PATTERNS: Tuple[ReferencePattern, ...] = (
    # OTA: complete GDS and instance map support structure-level classification.
    _pattern("ota.input_pair_centroid", "two_stage_ota", "relative_placement",
             "Input pair uses a point-reflected two-row common-centroid arrangement.",
             "common_centroid", *_OTA_EVIDENCE),
    _pattern("ota.input_pair_interleave", "two_stage_ota", "relative_placement",
             "Input-pair fingers interleave in complementary row sequences.",
             "interdigitation", *_OTA_EVIDENCE),
    _pattern("ota.functional_array_order", "two_stage_ota", "relative_placement",
             "Mirror and output arrays preserve topology-derived member and finger order.",
             "finger_array_order", *_OTA_EVIDENCE),
    _pattern("ota.shared_diffusion_arrays", "two_stage_ota", "device_geometry",
             "Multi-finger functional groups form shared-diffusion arrays.",
             "array", *_OTA_EVIDENCE),
    _pattern("ota.passive_unit_splits", "two_stage_ota", "device_geometry",
             "Miller capacitors and nulling or degeneration resistors use matched physical unit splits.",
             "split", *_OTA_EVIDENCE),
    _pattern("ota.dense_device_contacts", "two_stage_ota", "device_geometry",
             "Device terminals use dense contact columns instead of sparse point contacts.",
             "dense_contacts", *_OTA_EVIDENCE),
    _pattern("ota.dummy_end_flanks", "two_stage_ota", "grouping_partition",
             "Matched arrays carry topology-derived dummy flanks at their row ends.",
             "dummy_flank", *_OTA_EVIDENCE),
    _pattern("ota.private_input_well", "two_stage_ota", "grouping_partition",
             "The source-tied input pair occupies a private body-domain well island.",
             "well_island", *_OTA_EVIDENCE),
    _pattern("ota.cluster_well_sharing", "two_stage_ota", "grouping_partition",
             "Compatible devices share functional-cluster well islands without cross-domain merging.",
             "well_island_sharing", *_OTA_EVIDENCE),
    _pattern("ota.tight_cluster_rings", "two_stage_ota", "power_shielding",
             "Each functional cluster has a PDK-minimum tight tap or isolation ring.",
             "guard_ring_tight", *_OTA_EVIDENCE),
    _pattern("ota.passive_subbanks", "two_stage_ota", "grouping_partition",
             "Miller capacitors and split resistor pairs retain separate compact banks.",
             "passive_bank_partition", *_OTA_EVIDENCE),
    _pattern("ota.passive_bank_identity", "two_stage_ota", "grouping_partition",
             "Passive master families form identifiable compact bank structures.",
             "bank", *_OTA_EVIDENCE),
    _pattern("ota.passive_adjacency", "two_stage_ota", "relative_placement",
             "Passive sub-banks remain adjacent to their topology-defined analog neighbors.",
             "adjacent", *_OTA_EVIDENCE),
    _pattern("ota.rail_side_bands", "two_stage_ota", "power_shielding",
             "Supply-tied device groups and passives occupy ordered rail-side bands.",
             "band", *_OTA_EVIDENCE),
    _pattern("ota.signal_trunks", "two_stage_ota", "power_shielding",
             "High-fanout bias nets use declared horizontal signal trunks.",
             "trunk", *_OTA_EVIDENCE),
    _pattern("ota.signal_trunk_widths", "two_stage_ota", "power_shielding",
             "Bias trunks retain a wider routing class than local signal wires.",
             "wire_width", *_OTA_EVIDENCE),
    _pattern("ota.upper_power_planes", "two_stage_ota", "power_shielding",
             "Global supply distribution uses upper-metal plane structures above the device bands.",
             "power_plane", *_OTA_EVIDENCE),
    _pattern("ota.edge_power_rails", "two_stage_ota", "power_shielding",
             "Power and ground use continuous edge rail structures.",
             "rail", *_OTA_EVIDENCE),
    _pattern("ota.output_edge_access", "two_stage_ota", "power_shielding",
             "The output-driving group preserves right-edge access.",
             "anchor", *_OTA_EVIDENCE),
    _pattern("ota.cross_band_corridor", "two_stage_ota", "power_shielding",
             "A device-free cross-band routing corridor is reserved.",
             "channel", *_OTA_EVIDENCE),
    _pattern("ota.whitespace_efficiency", "two_stage_ota", "density",
             "Occupied-area and whitespace remain within a normalized reference-derived band.",
             "density_whitespace_target", *_OTA_EVIDENCE),
    _pattern("ota.active_fill", "two_stage_ota", "density",
             "Active-diffusion fill preserves the human reference packing regime.",
             "od_fill_target", *_OTA_EVIDENCE),

    # LDO: spatial-only GDS permits relative floorplan and density claims, not
    # missing-PCell device artwork claims.
    _pattern("ldo.mim_left", "reference_ldo", "relative_placement",
             "The MIM bank remains on the left side of the lateral power floorplan.",
             "anchor", *_LDO_EVIDENCE),
    _pattern("ldo.analog_core_center", "reference_ldo", "grouping_partition",
             "The analog core remains in the central floorplan band.",
             "band", *_LDO_EVIDENCE),
    _pattern("ldo.functional_macro_groups", "reference_ldo", "grouping_partition",
             "MIM, analog-core, resistor, and pass-device families remain distinct functional groups.",
             "place_group", *_LDO_EVIDENCE),
    _pattern("ldo.resistor_bank_top", "reference_ldo", "grouping_partition",
             "The resistor bank remains a distinct upper partition.",
             "band", *_LDO_EVIDENCE),
    _pattern("ldo.resistor_bank_identity", "reference_ldo", "grouping_partition",
             "The resistor master family forms an identifiable compact bank structure.",
             "bank", *_LDO_EVIDENCE),
    _pattern("ldo.resistor_subbank_partition", "reference_ldo", "grouping_partition",
             "The audit records a missing topology-defined partition within the resistor bank.",
             "passive_bank_partition", *_LDO_EVIDENCE),
    _pattern("ldo.pass_array_right", "reference_ldo", "relative_placement",
             "The wide pass-device array remains on the right side of the lateral floorplan.",
             "anchor", *_LDO_EVIDENCE),
    _pattern("ldo.pass_array_order", "reference_ldo", "relative_placement",
             "The audit records missing exact functional-order proof for the lateral pass-device fingers.",
             "finger_array_order", *_LDO_EVIDENCE),
    _pattern("ldo.matched_control_pairs", "reference_ldo", "relative_placement",
             "The topology declares matched control-device pair identities.",
             "matching", *_LDO_EVIDENCE),
    _pattern("ldo.well_grouping", "reference_ldo", "grouping_partition",
             "The audit records missing proof that compatible control and pass devices share only declared well islands.",
             "well_island_sharing", *_LDO_EVIDENCE),
    _pattern("ldo.routing_corridor", "reference_ldo", "power_shielding",
             "The audit records a missing cross-block routing-corridor primitive for the lateral macro arrangement.",
             "channel", *_LDO_EVIDENCE),
    _pattern("ldo.reference_feedback_shield", "reference_ldo", "power_shielding",
             "The topology declares GNDA shielding between the VREF aggressor and VFB victim domains.",
             "shield", *_LDO_EVIDENCE, "ldo_dev/source/reference_ldo.clay"),
    _pattern("ldo.reference_kelvin", "reference_ldo", "power_shielding",
             "The topology declares RR19 as the VOUT force and VFB sense termination.",
             "kelvin", *_LDO_EVIDENCE, "ldo_dev/source/reference_ldo.clay"),
    _pattern("ldo.pass_current_budget", "reference_ldo", "power_shielding",
             "The audit records missing current and voltage-drop budgets for pass-array power routes.",
             "power_sensitive_net_budget", *_LDO_EVIDENCE),
    _pattern("ldo.power_trunks", "reference_ldo", "power_shielding",
             "The topology declares shared supply and output trunks across the functional groups.",
             "trunk", *_LDO_EVIDENCE),
    _pattern("ldo.whitespace_efficiency", "reference_ldo", "density",
             "Whitespace remains within a normalized human-floorplan target.",
             "density_whitespace_target", *_LDO_EVIDENCE),
    _pattern("ldo.pass_thermal_neighborhood", "reference_ldo", "thermal_stress",
             "The audit records a missing thermal and stress neighborhood obligation for the pass array.",
             "thermal_stress_neighborhood", *_LDO_EVIDENCE),

    # Bandgap: source topology proves matched identities; the OA spatial GDS
    # supplies only relative bank/well/floorplan observations until PCells exist.
    _pattern("bandgap.pnp_matching", "bandgap", "relative_placement",
             "The reference PNP pair is matched as one physical comparison group.",
             "matching", *_BANDGAP_EVIDENCE),
    _pattern("bandgap.resistor_matching", "bandgap", "relative_placement",
             "Reference resistor pairs preserve their declared matching relationships.",
             "matching", *_BANDGAP_EVIDENCE),
    _pattern("bandgap.mos_array_order", "bandgap", "relative_placement",
             "The audit records missing exact member and finger-order proof for matched MOS groups.",
             "finger_array_order", *_BANDGAP_EVIDENCE),
    _pattern("bandgap.resistor_partition", "bandgap", "grouping_partition",
             "The audit records a missing topology-defined resistor sub-bank partition.",
             "passive_bank_partition", *_BANDGAP_EVIDENCE),
    _pattern("bandgap.passive_bank_identity", "bandgap", "grouping_partition",
             "Reference resistor and MIM master families form identifiable compact banks.",
             "bank", *_BANDGAP_EVIDENCE),
    _pattern("bandgap.functional_groups", "bandgap", "grouping_partition",
             "PNP, resistor, MIM, PMOS, and NMOS families remain distinct functional groups.",
             "place_group", *_BANDGAP_EVIDENCE),
    _pattern("bandgap.well_island_sharing", "bandgap", "grouping_partition",
             "The audit records missing proof for shared compatible well islands and separated body domains.",
             "well_island_sharing", *_BANDGAP_EVIDENCE),
    _pattern("bandgap.private_well_domains", "bandgap", "grouping_partition",
             "The topology declares private isolation for devices that require separate well domains.",
             "well_isolated", *_BANDGAP_EVIDENCE),
    _pattern("bandgap.pnp_taps", "bandgap", "power_shielding",
             "The topology declares substrate taps for the PNP comparison group.",
             "tap", *_BANDGAP_EVIDENCE),
    _pattern("bandgap.rail_side_bands", "bandgap", "power_shielding",
             "PMOS, NMOS, resistor, and MIM master families retain ordered rail-side bands.",
             "band", *_BANDGAP_EVIDENCE),
    _pattern("bandgap.group_rail_affinity", "bandgap", "power_shielding",
             "The topology assigns polarity and passive groups to supply-relative rail sides.",
             "rail_affinity", *_BANDGAP_EVIDENCE),
    _pattern("bandgap.supply_trunks", "bandgap", "power_shielding",
             "The topology declares shared supply trunks across the functional bands.",
             "trunk", *_BANDGAP_EVIDENCE),
    _pattern("bandgap.vref_edge_access", "bandgap", "power_shielding",
             "The reference output retains declared edge access.",
             "anchor", *_BANDGAP_EVIDENCE),
    _pattern("bandgap.escape_corridor", "bandgap", "power_shielding",
             "The audit records a missing routing escape-corridor obligation for well and matching groups.",
             "channel", *_BANDGAP_EVIDENCE),
    _pattern("bandgap.whitespace_efficiency", "bandgap", "density",
             "Whitespace remains within a normalized human-floorplan target.",
             "density_whitespace_target", *_BANDGAP_EVIDENCE),
    _pattern("bandgap.pnp_resistor_thermal", "bandgap", "thermal_stress",
             "The audit records a missing controlled thermal and stress neighborhood for PNP and resistor groups.",
             "thermal_stress_neighborhood", *_BANDGAP_EVIDENCE),
)


_COORDINATE_RE = re.compile(
    r"(?:^|[^A-Za-z0-9_])(?:x|y)\s*=|\(\s*[-+]?\d+(?:\.\d+)?\s*,\s*[-+]?\d+(?:\.\d+)?\s*\)",
    re.IGNORECASE,
)

def phase1_minimum_primitive_ids(
    patterns: Sequence[ReferencePattern] = PHASE1_PATTERNS,
) -> Tuple[str, ...]:
    """The exact Phase-1 vocabulary demanded by the scanned target patterns."""

    if type(patterns) is not tuple or patterns != PHASE1_PATTERNS:
        raise PrimitiveRegistryError("Phase-1 pattern denominator cannot change")
    return tuple(sorted({pattern.primitive_id for pattern in patterns}))


def canonical_registry_data() -> dict[str, object]:
    return {
        "schema": "clay.primitive-taxonomy/v1",
        "dimensions": list(DIMENSIONS),
        "primitives": [asdict(item) for item in PRIMITIVES],
        "reference_scans": [asdict(item) for item in REFERENCE_SCANS],
        "phase1_patterns": [asdict(item) for item in PHASE1_PATTERNS],
        "phase1_minimum_primitive_ids": list(phase1_minimum_primitive_ids()),
    }


def registry_sha256() -> str:
    payload = json.dumps(
        canonical_registry_data(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _manifest_artifact(payload: dict[str, object], role: str) -> dict[str, object]:
    artifacts = payload.get("reference_artifacts")
    if type(artifacts) is not list or len(artifacts) > _MAX_EVIDENCE:
        raise PrimitiveRegistryError("reference manifest artifacts are not bounded")
    matches = tuple(
        item
        for item in artifacts
        if type(item) is dict and item.get("role") == role
    )
    if len(matches) != 1:
        raise PrimitiveRegistryError(f"reference manifest must name one {role!r}")
    return matches[0]


def validate_reference_scan(
    repo_root: Path,
    scan: ReferenceScan,
) -> None:
    """Replay one scan's manifest scope and optional coordinate-free extraction."""

    if type(scan) is not ReferenceScan:
        raise PrimitiveRegistryError("reference scan must be an exact typed row")
    scan.__post_init__()
    root = Path(repo_root).resolve()
    bound_files = (
        scan.gds,
        scan.manifest,
        scan.topology,
        scan.source,
        *((scan.oa_spatial_anchor_gds,) if scan.oa_spatial_anchor_gds else ()),
    )
    for bound in bound_files:
        path = root / PurePosixPath(bound.path)
        if not path.is_file():
            raise PrimitiveRegistryError(f"bound reference file is missing: {bound.path}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != bound.sha256:
            raise PrimitiveRegistryError(f"bound reference file changed: {bound.path}")

    if PurePosixPath(scan.source.path).suffix.lower() != ".clay":
        raise PrimitiveRegistryError("reference source must be a Clay source artifact")
    try:
        from clay.parser import parse_clay_file

        parsed_source = parse_clay_file(root / PurePosixPath(scan.source.path))
    except Exception as exc:
        raise PrimitiveRegistryError("reference source is not valid Clay source") from exc
    source_cells = tuple(
        str(getattr(cell, "local_name", "")).lower()
        for cell in getattr(parsed_source, "cell_defs", ())
    )
    if source_cells != (scan.source_cell_id,):
        raise PrimitiveRegistryError("reference source cell identity does not match scan")

    manifest_path = root / PurePosixPath(scan.manifest.path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise PrimitiveRegistryError("reference manifest is not canonical JSON") from exc
    if type(payload) is not dict or payload.get("schema") != "clay.reference-manifest/v1":
        raise PrimitiveRegistryError("reference manifest schema is not canonical")
    if payload.get("circuit_id") != scan.circuit_id:
        raise PrimitiveRegistryError("reference manifest circuit identity does not match scan")
    if payload.get("topology_sha256") != scan.topology.sha256:
        raise PrimitiveRegistryError("reference manifest topology digest does not match scan")
    layout_artifact = _manifest_artifact(payload, "layout_gds")
    if (
        layout_artifact.get("repository_path") != scan.gds.path
        or layout_artifact.get("sha256") != scan.gds.sha256
    ):
        raise PrimitiveRegistryError("reference manifest GDS binding does not match scan")
    if scan.oa_spatial_anchor_gds is not None:
        anchor_artifact = _manifest_artifact(payload, "oa_spatial_anchor_gds")
        if (
            anchor_artifact.get("repository_path")
            != scan.oa_spatial_anchor_gds.path
            or anchor_artifact.get("sha256")
            != scan.oa_spatial_anchor_gds.sha256
        ):
            raise PrimitiveRegistryError(
                "reference manifest OA-anchor binding does not match scan"
            )
    topology_artifact = _manifest_artifact(payload, "canonical_netlist")
    if (
        topology_artifact.get("repository_path") != scan.topology.path
        or topology_artifact.get("sha256") != scan.topology.sha256
    ):
        raise PrimitiveRegistryError("reference manifest topology artifact does not match scan")

    scope = payload.get("scope")
    if type(scope) is not dict:
        raise PrimitiveRegistryError("reference manifest lacks an exact scope")
    empty_pcell_count = scope.get("empty_pcell_count")
    if type(empty_pcell_count) is not int or empty_pcell_count < 0:
        raise PrimitiveRegistryError("reference manifest empty-PCell count is invalid")
    if scan.evidence_scope == "complete_exact_gds":
        if not _is_sha256(scan.coordinate_free_spatial_sha256):
            raise PrimitiveRegistryError(
                "complete reference spatial scan requires a replayable digest"
            )
        if not (
            scope.get("device_geometry") == "pass"
            and scope.get("floorplan") == "pass"
            and scope.get("routing_geometry") == "pass"
            and empty_pcell_count == 0
        ):
            raise PrimitiveRegistryError("complete reference scan contradicts manifest scope")
    elif scan.evidence_scope == "complete_unmapped_gds":
        if not (
            scope.get("device_geometry") == "measurable_layer_level"
            and scope.get("floorplan") == "advisory"
            and scope.get("routing_geometry") == "advisory"
            and empty_pcell_count == 0
            and _is_sha256(scan.coordinate_free_spatial_sha256)
        ):
            raise PrimitiveRegistryError(
                "complete unmapped reference scan contradicts manifest scope"
            )
    else:
        if not (
            scope.get("device_geometry") == "unmeasurable"
            and scope.get("floorplan") == "advisory"
            and scope.get("routing_geometry") == "advisory"
            and empty_pcell_count > 0
            and _is_sha256(scan.coordinate_free_spatial_sha256)
        ):
            raise PrimitiveRegistryError("OA spatial scan contradicts manifest scope")
    from clay.frontend.reference_spatial_intent import (
        extract_reference_spatial_intent,
        require_reference_structure_primitives,
    )

    spatial_anchor = scan.oa_spatial_anchor_gds or scan.gds
    replayed = extract_reference_spatial_intent(
        str(root / PurePosixPath(spatial_anchor.path)),
        source_netlist=(root / PurePosixPath(scan.topology.path)).read_text(
            encoding="utf-8", errors="ignore"
        ),
    )
    if replayed.reference_scope_sha256 != scan.coordinate_free_spatial_sha256:
        raise PrimitiveRegistryError("coordinate-free spatial scan digest does not replay")
    require_reference_structure_primitives(replayed)


def validate_registry(
    repo_root: Path,
    *,
    primitives: Sequence[PrimitiveDefinition] = PRIMITIVES,
    scans: Sequence[ReferenceScan] = REFERENCE_SCANS,
    patterns: Sequence[ReferencePattern] = PHASE1_PATTERNS,
) -> None:
    """Fail closed on taxonomy drift, leakage, missing evidence, or stale SHA bindings."""

    if any(type(value) is not tuple for value in (primitives, scans, patterns)):
        raise PrimitiveRegistryError("registry row inventories must be exact tuples")
    root = Path(repo_root).resolve()
    primitive_rows = tuple(primitives)
    scan_rows = tuple(scans)
    pattern_rows = tuple(patterns)
    if any(type(row) is not PrimitiveDefinition for row in primitive_rows):
        raise PrimitiveRegistryError("primitive inventory contains a non-typed row")
    if any(type(row) is not ReferenceScan for row in scan_rows):
        raise PrimitiveRegistryError("reference scan inventory contains a non-typed row")
    if any(type(row) is not ReferencePattern for row in pattern_rows):
        raise PrimitiveRegistryError("pattern denominator contains a non-typed row")
    if tuple(row.primitive_id for row in primitive_rows) != tuple(
        row.primitive_id for row in PRIMITIVES
    ):
        raise PrimitiveRegistryError("primitive taxonomy denominator cannot change")
    if tuple(row.circuit_id for row in scan_rows) != tuple(
        row.circuit_id for row in REFERENCE_SCANS
    ):
        raise PrimitiveRegistryError("reference scan denominator cannot change")
    if tuple(row.pattern_id for row in pattern_rows) != tuple(
        row.pattern_id for row in PHASE1_PATTERNS
    ):
        raise PrimitiveRegistryError("Phase-1 pattern denominator cannot change")
    by_id = {row.primitive_id: row for row in primitive_rows}
    if len(by_id) != len(primitive_rows):
        raise PrimitiveRegistryError("primitive ids must be unique")
    for row in primitive_rows:
        row.__post_init__()
        if (
            type(row.primitive_id) is not str
            or not row.primitive_id
            or type(row.dimensions) is not tuple
            or not row.dimensions
            or any(dimension not in DIMENSIONS for dimension in row.dimensions)
            or len(set(row.dimensions)) != len(row.dimensions)
        ):
            raise PrimitiveRegistryError(f"invalid primitive definition {row!r}")
        if row.taxonomy_status == "gap" and not row.gap:
            raise PrimitiveRegistryError(f"gap primitive {row.primitive_id!r} lacks a gap statement")
        if row.checker_status == "exact_gds" and row.materializer_status != "materialized":
            raise PrimitiveRegistryError(f"{row.primitive_id!r} claims exact checking without materialization")
        for path in row.evidence:
            _validate_relative_evidence(path)
            if not (root / PurePosixPath(path)).is_file():
                raise PrimitiveRegistryError(f"primitive evidence is missing: {path}")

    scan_by_circuit = {row.circuit_id: row for row in scan_rows}
    if len(scan_by_circuit) != len(scan_rows) or set(scan_by_circuit) != {
        "two_stage_ota", "reference_ldo", "bandgap"
    }:
        raise PrimitiveRegistryError("reference scans must cover exactly OTA, LDO, and Bandgap")
    for scan in scan_rows:
        validate_reference_scan(root, scan)
        if scan.gds.path.lower().endswith(".gds") is False:
            raise PrimitiveRegistryError(f"reference authority must be GDS: {scan.gds.path}")
        if (
            scan.oa_spatial_anchor_gds is not None
            and not scan.oa_spatial_anchor_gds.path.lower().endswith(".gds")
        ):
            raise PrimitiveRegistryError(
                f"reference OA anchor must be GDS: {scan.oa_spatial_anchor_gds.path}"
            )

    pattern_ids = {row.pattern_id for row in pattern_rows}
    if len(pattern_ids) != len(pattern_rows):
        raise PrimitiveRegistryError("reference pattern ids must be unique")
    if {row.circuit_id for row in pattern_rows} != set(scan_by_circuit):
        raise PrimitiveRegistryError("Phase-1 patterns must cover all three reference scans")
    for row in pattern_rows:
        row.__post_init__()
        scan_scope = scan_by_circuit[row.circuit_id].evidence_scope
        if row.basis == "complete_gds_observation" and scan_scope != "complete_exact_gds":
            raise PrimitiveRegistryError(f"pattern {row.pattern_id!r} overclaims complete GDS scope")
        if row.basis == "oa_spatial_observation" and scan_scope not in {
            "oa_spatial_only",
            "complete_unmapped_gds",
        }:
            raise PrimitiveRegistryError(f"pattern {row.pattern_id!r} misstates OA spatial scope")
        if row.basis == "audit_gap" and "audit" not in row.description.lower():
            raise PrimitiveRegistryError(f"pattern {row.pattern_id!r} does not label its audit gap")
        primitive = by_id.get(row.primitive_id)
        if primitive is None:
            raise PrimitiveRegistryError(f"pattern {row.pattern_id!r} names an unknown primitive")
        if row.dimension not in primitive.dimensions:
            raise PrimitiveRegistryError(f"pattern {row.pattern_id!r} uses the primitive in the wrong dimension")
        if not row.description.strip():
            raise PrimitiveRegistryError(f"pattern {row.pattern_id!r} lacks a description")
        if _COORDINATE_RE.search(row.description):
            raise PrimitiveRegistryError(f"pattern {row.pattern_id!r} leaks a reference coordinate")
        if type(row.evidence) is not tuple or not row.evidence:
            raise PrimitiveRegistryError(f"pattern {row.pattern_id!r} lacks evidence")
        for path in row.evidence:
            _validate_relative_evidence(path)
            if not (root / PurePosixPath(path)).is_file():
                raise PrimitiveRegistryError(f"pattern evidence is missing: {path}")
    if primitive_rows != PRIMITIVES:
        raise PrimitiveRegistryError("primitive taxonomy contents differ from the frozen registry")
    if scan_rows != REFERENCE_SCANS:
        raise PrimitiveRegistryError("reference scan identity differs from the frozen registry")
    if pattern_rows != PHASE1_PATTERNS:
        raise PrimitiveRegistryError("Phase-1 denominator contents differ from the frozen registry")


def render_taxonomy_markdown() -> str:
    """Render the checked Phase-0 reference frame from this registry only."""

    lines = [
        "# Coordinate-free primitive taxonomy for OTA, LDO, and Bandgap",
        "",
        "This is the Phase-0 reference frame. Completeness is not inferred from placer",
        "quality or from the number of implemented primitives. A structure pattern is",
        "covered only when a later coverage row is mapped, materialized, and independently",
        "checked on the exact emitted GDS. Rows that are unsupported or unmeasurable block.",
        "",
        f"Registry schema: `clay.primitive-taxonomy/v1`; SHA-256: `{registry_sha256()}`.",
        "",
        "## Bound human-reference scans",
        "",
        "| Circuit | Scope | Bound evidence | Coordinate-free spatial SHA-256 |",
        "|---|---|---|---|",
    ]
    for scan in REFERENCE_SCANS:
        bound_rows = [
            ("layer GDS", scan.gds),
            ("manifest", scan.manifest),
            ("topology", scan.topology),
            ("intent", scan.source),
        ]
        if scan.oa_spatial_anchor_gds is not None:
            bound_rows.append(("OA anchor GDS", scan.oa_spatial_anchor_gds))
        bindings = "<br>".join(
            f"`{label}: {bound.path}` `{bound.sha256}`"
            for label, bound in bound_rows
        )
        lines.append(
            f"| `{scan.circuit_id}` | `{scan.evidence_scope}` | {bindings}<br>"
            f"method: {scan.scan_method} | "
            f"`{scan.coordinate_free_spatial_sha256 or 'not-applicable'}` |"
        )
    lines.extend([
        "",
        "LDO and Bandgap have complete flattened layer geometry but no full logical-instance",
        "polygon ownership map. Layer/region geometry is measurable without promoting it to",
        "exact logical-instance geometry truth; topology-bound relative floorplan observations",
        "use separate OA anchor GDS files. OTA has a complete GDS and instance map. No reference",
        "coordinate is stored in this registry.",
        "",
        "## Six-dimensional taxonomy",
    ])
    for dimension in DIMENSIONS:
        lines.extend([
            "",
            f"### {dimension.replace('_', ' ')}",
            "",
            "| Primitive | Surface | Taxonomy | Materializer | Checker | Open gap |",
            "|---|---|---|---|---|---|",
        ])
        for primitive in PRIMITIVES:
            if dimension not in primitive.dimensions:
                continue
            lines.append(
                f"| `{primitive.primitive_id}` | `{primitive.surface}` | "
                f"`{primitive.taxonomy_status}` | `{primitive.materializer_status}` | "
                f"`{primitive.checker_status}` | {primitive.gap or ''} |"
            )
    lines.extend([
        "",
        "## Phase-1 minimum-necessary patterns",
        "",
        "These rows are the real-scan denominator. They are requirements, not coverage",
        "claims. The primitive column names the vocabulary that must be closed in Phase 1.",
        "",
        "| Pattern | Circuit | Basis | Dimension | Structure | Required primitive | Evidence |",
        "|---|---|---|---|---|---|---|",
    ])
    for pattern in PHASE1_PATTERNS:
        evidence = "<br>".join(f"`{path}`" for path in pattern.evidence)
        lines.append(
            f"| `{pattern.pattern_id}` | `{pattern.circuit_id}` | `{pattern.basis}` | "
            f"`{pattern.dimension}` | {pattern.description} | "
            f"`{pattern.primitive_id}` | {evidence} |"
        )
    lines.extend([
        "",
        "## Phase-1 implementation set",
        "",
    ])
    for primitive_id in phase1_minimum_primitive_ids():
        primitive = next(item for item in PRIMITIVES if item.primitive_id == primitive_id)
        lines.append(
            f"- `{primitive_id}`: materializer=`{primitive.materializer_status}`, "
            f"checker=`{primitive.checker_status}`"
            + (f"; gap: {primitive.gap}" if primitive.gap else "")
        )
    lines.extend([
        "",
        "`anchor` closure is deliberately scoped to the minimum necessary target-domain",
        "variants: `left` and `right`. The parser preserves the existing `top` and `bottom`",
        "surface, but both variants remain explicit `unsupported` obligations in the",
        "materializer/checker path; they are not counted as covered until that path is",
        "implemented and independently checked on exact GDS.",
        "",
        "Phase 1 must turn every item used by the denominator into a real materializer",
        "and an independent exact-artifact checker, with a dropped-geometry negative test.",
        "Phase 2 may then recompute SHA-bound coverage rows. Admission and search remain",
        "frozen until both phases are green.",
        "",
    ])
    return "\n".join(lines)


__all__ = [
    "BoundReferenceFile",
    "DIMENSIONS",
    "PHASE1_PATTERNS",
    "PRIMITIVES",
    "PatternBasis",
    "PrimitiveDefinition",
    "PrimitiveRegistryError",
    "REFERENCE_SCANS",
    "ReferencePattern",
    "ReferenceScan",
    "canonical_registry_data",
    "phase1_minimum_primitive_ids",
    "registry_sha256",
    "render_taxonomy_markdown",
    "validate_reference_scan",
    "validate_registry",
]
