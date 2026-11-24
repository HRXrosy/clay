"""Canonical Phase-2 primitive coverage for the three human references.

Coverage is derived only from the frozen reference-pattern denominator and the
machine-checked primitive registry.  It is not a placement-quality score and it
does not promote model-only evidence to exact-GDS coverage.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping, Tuple

from clay.frontend.primitive_registry import (
    DIMENSIONS,
    PHASE1_PATTERNS,
    PRIMITIVES,
    REFERENCE_SCANS,
    registry_sha256,
    validate_registry,
)


_SCHEMA = "clay.primitive-coverage/v3"
_LOWER_BOUND_MATRIX_SCHEMA = "clay.pdk-legal-lower-bound-matrix/v3"
_LOWER_BOUND_MATRIX_PATH = "docs/audits/pdk-legal-lower-bounds.json"
_LOWER_BOUND_MATRIX_ARTIFACT_SHA256 = (
    "bc70589d9b030327d67d02036f918e3388ccccb3f5f51e93401320ff207f07fc"
)
_MAX_ROWS = 256
_MAX_TEXT = 4_096
_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")
_ID_RE = re.compile(r"[a-z][a-z0-9_.]*\Z")
_BLOCKER_CODE_RE = re.compile(r"[A-Z][A-Z0-9_]*\Z")
_PARITY_SCOPES = ("oa_spatial_parity", "device_geometry_parity")
_GEOMETRY_UNMEASURABLE_REASONS: Mapping[str, tuple[str, str]] = MappingProxyType({
    "complete_unmapped_gds": (
        "REFERENCE_LOGICAL_INSTANCE_GEOMETRY_MAP_UNAVAILABLE",
        "The active complete GDS has no complete logical-instance geometry map.",
    ),
    "oa_spatial_only": (
        "REFERENCE_PCELLS_EMPTY",
        "reference PCells empty",
    ),
})
_COORDINATE_RE = re.compile(
    r"(?:^|[^A-Za-z0-9_])(?:x|y)\s*=|"
    r"\(\s*[-+]?\d+(?:\.\d+)?\s*,\s*[-+]?\d+(?:\.\d+)?\s*\)",
    re.IGNORECASE,
)


COVERAGE_OUTPUTS: Mapping[str, str] = MappingProxyType({
    "two_stage_ota": "docs/audits/primitive-coverage-two-stage-ota.json",
    "reference_ldo": "docs/audits/primitive-coverage-reference-ldo.json",
    "bandgap": "docs/audits/primitive-coverage-bandgap.json",
})

VERDICT_OUTPUTS: Mapping[str, str] = MappingProxyType({
    "two_stage_ota": "docs/audits/convergence-verdict-two-stage-ota.md",
    "reference_ldo": "docs/audits/convergence-verdict-reference-ldo.md",
    "bandgap": "docs/audits/convergence-verdict-bandgap.md",
})

_VERDICT_CLASSIFICATION: Mapping[str, str] = MappingProxyType({
    "two_stage_ota": "(b) algorithm/domain gap",
    "reference_ldo": "(c) data/PDK hard constraint",
    "bandgap": "(c) data/PDK hard constraint",
})

_REACHABILITY_EVIDENCE: Mapping[str, str] = MappingProxyType({
    "two_stage_ota": "ota_dev/signoff/four_pdk_signoff_ledger.json",
    "reference_ldo": "ldo_dev/analysis/reference_physical_targets.json",
    "bandgap": "bandgap_dev/signoff/four_pdk_ledger.json",
})


class CoverageArtifactError(ValueError):
    """A coverage row or artifact does not replay the frozen registry."""


def _text(value: object, *, field: str, pattern: re.Pattern[str] | None = None) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or len(value) > _MAX_TEXT
    ):
        raise CoverageArtifactError(f"{field} must be an exact bounded string")
    if pattern is not None and pattern.fullmatch(value) is None:
        raise CoverageArtifactError(f"{field} is not canonical")
    return value


def _digest(value: object, *, field: str) -> str:
    if type(value) is not str or _DIGEST_RE.fullmatch(value) is None:
        raise CoverageArtifactError(f"{field} must be a lowercase SHA-256")
    return value


def _evidence_path(value: object) -> str:
    path = _text(value, field="coverage evidence")
    if "\\" in path:
        raise CoverageArtifactError("coverage evidence must use POSIX separators")
    pure = PurePosixPath(path)
    if pure.is_absolute() or ".." in pure.parts:
        raise CoverageArtifactError("coverage evidence must stay inside the repository")
    if pure.suffix.lower() == ".png":
        raise CoverageArtifactError("PNG is forbidden as coverage evidence")
    return path


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise CoverageArtifactError("coverage artifact is not canonical JSON") from exc


def _load_reference_lower_bound_row(
    root: Path,
    *,
    circuit_id: str,
    source_path: str,
    manifest_path: str,
) -> tuple[dict[str, object], dict[str, object], str]:
    """Load the replayed row for the PDK bound by the reference manifest."""

    matrix_path = root / PurePosixPath(_LOWER_BOUND_MATRIX_PATH)
    try:
        raw_matrix = matrix_path.read_bytes()
        payload = json.loads(
            raw_matrix.decode("ascii"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except CoverageArtifactError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CoverageArtifactError(
            f"PDK-legal lower-bound matrix cannot be read: {_LOWER_BOUND_MATRIX_PATH}"
        ) from exc
    if type(payload) is not dict or set(payload) != {
        "aggregation",
        "artifact_sha256",
        "circuit_order",
        "frozen_inputs",
        "pdk_order",
        "rows",
        "schema",
    }:
        raise CoverageArtifactError("PDK-legal lower-bound matrix schema is not exact")
    canonical_matrix = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True).encode("ascii")
        + b"\n"
    )
    if raw_matrix != canonical_matrix:
        raise CoverageArtifactError("PDK-legal lower-bound matrix bytes are not canonical")
    if payload.get("schema") != _LOWER_BOUND_MATRIX_SCHEMA:
        raise CoverageArtifactError("PDK-legal lower-bound matrix schema is not canonical")
    artifact_sha256 = _digest(
        payload.get("artifact_sha256"), field="lower-bound matrix artifact SHA-256"
    )
    unsigned = dict(payload)
    unsigned.pop("artifact_sha256")
    if hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest() != artifact_sha256:
        raise CoverageArtifactError("PDK-legal lower-bound matrix digest is invalid")
    if artifact_sha256 != _LOWER_BOUND_MATRIX_ARTIFACT_SHA256:
        raise CoverageArtifactError(
            "PDK-legal lower-bound matrix differs from the code-bound artifact authority"
        )

    frozen = payload.get("frozen_inputs")
    if type(frozen) is not dict or set(frozen) != {"path", "sha256"}:
        raise CoverageArtifactError("lower-bound matrix frozen-input binding is not exact")
    frozen_path = _evidence_path(frozen.get("path"))
    frozen_sha256 = _digest(
        frozen.get("sha256"), field="lower-bound matrix frozen-input SHA-256"
    )
    try:
        frozen_bytes = (root / PurePosixPath(frozen_path)).read_bytes()
    except OSError as exc:
        raise CoverageArtifactError(
            f"lower-bound matrix frozen input is missing: {frozen_path}"
        ) from exc
    if hashlib.sha256(frozen_bytes).hexdigest() != frozen_sha256:
        raise CoverageArtifactError("lower-bound matrix frozen-input digest is stale")

    from clay.flow.reference_evidence import load_reference_manifest

    manifest = load_reference_manifest(
        root / PurePosixPath(manifest_path), repo_root=root
    )
    reference_pdk = _text(
        manifest.reference_pdk_id,
        field="reference manifest PDK",
        pattern=_ID_RE,
    )
    pdk_order = payload.get("pdk_order")
    if type(pdk_order) is not list or reference_pdk not in pdk_order:
        raise CoverageArtifactError("reference PDK is absent from lower-bound matrix")
    rows = payload.get("rows")
    if type(rows) is not list or not rows or len(rows) > _MAX_ROWS:
        raise CoverageArtifactError("lower-bound matrix rows are not exact bounded rows")
    matches = tuple(
        row
        for row in rows
        if type(row) is dict
        and row.get("circuit_id") == circuit_id
        and row.get("pdk_id") == reference_pdk
    )
    if len(matches) != 1:
        raise CoverageArtifactError(
            "lower-bound matrix lacks one row for the circuit's reference PDK"
        )
    row = matches[0]
    if set(row) != {
        "blockers",
        "circuit_id",
        "components",
        "input_sha256",
        "lower_bound_nm2",
        "netlist",
        "normalized_connectivity_sha256",
        "pdk_id",
        "pdk_installed_authority_sha256",
        "pdk_source_snapshot_sha256",
        "receipt_sha256",
        "reference_manifest",
        "resources",
        "source",
        "status",
    }:
        raise CoverageArtifactError("lower-bound matrix row schema is not exact")
    manifest_locator = row.get("reference_manifest")
    if type(manifest_locator) is not dict or manifest_locator != {
        "path": manifest_path,
        "sha256": hashlib.sha256(
            (root / PurePosixPath(manifest_path)).read_bytes()
        ).hexdigest(),
    }:
        raise CoverageArtifactError("lower-bound matrix manifest binding is stale")
    if row.get("normalized_connectivity_sha256") != (
        manifest.normalized_connectivity_sha256
    ):
        raise CoverageArtifactError(
            "lower-bound matrix normalized connectivity binding is stale"
        )
    canonical_netlists = tuple(
        item
        for item in manifest.reference_artifacts
        if item.role == "canonical_netlist"
    )
    if len(canonical_netlists) != 1 or canonical_netlists[0].repository_path is None:
        raise CoverageArtifactError("reference manifest lacks one canonical netlist")
    netlist_locator = row.get("netlist")
    if type(netlist_locator) is not dict or netlist_locator != {
        "path": canonical_netlists[0].repository_path,
        "sha256": canonical_netlists[0].sha256,
    }:
        raise CoverageArtifactError("lower-bound matrix netlist binding is stale")
    source = row.get("source")
    if type(source) is not dict or source != {
        "path": source_path,
        "sha256": hashlib.sha256(
            (root / PurePosixPath(source_path)).read_bytes()
        ).hexdigest(),
    }:
        raise CoverageArtifactError("lower-bound matrix source binding is stale")
    status = row.get("status")
    blockers = row.get("blockers")
    if status == "supported":
        lower_bound_nm2 = row.get("lower_bound_nm2")
        if type(lower_bound_nm2) is not int or lower_bound_nm2 <= 0:
            raise CoverageArtifactError("supported legal lower bound is not positive")
        if blockers != []:
            raise CoverageArtifactError("supported legal lower bound retains blockers")
        _digest(row.get("input_sha256"), field="lower-bound input SHA-256")
        _digest(row.get("receipt_sha256"), field="lower-bound receipt SHA-256")
        _digest(
            row.get("pdk_source_snapshot_sha256"),
            field="lower-bound PDK snapshot SHA-256",
        )
        _digest(
            row.get("pdk_installed_authority_sha256"),
            field="lower-bound installed PDK authority SHA-256",
        )
    elif status == "unmeasurable":
        if any(row.get(field) is not None for field in (
            "input_sha256",
            "lower_bound_nm2",
            "pdk_installed_authority_sha256",
            "pdk_source_snapshot_sha256",
            "receipt_sha256",
        )):
            raise CoverageArtifactError(
                "unmeasurable legal lower bound contains fabricated numeric evidence"
            )
        if type(blockers) is not list or not blockers:
            raise CoverageArtifactError(
                "unmeasurable legal lower bound lacks a typed blocker"
            )
        for blocker in blockers:
            if type(blocker) is not dict or set(blocker) != {"code", "detail"}:
                raise CoverageArtifactError("lower-bound blocker schema is not exact")
            _text(
                blocker.get("code"),
                field="lower-bound blocker code",
                pattern=_BLOCKER_CODE_RE,
            )
            _text(blocker.get("detail"), field="lower-bound blocker detail")
    else:
        raise CoverageArtifactError("lower-bound matrix row status is unknown")
    return payload, row, reference_pdk


def _reference_scan(circuit_id: str) -> Any:
    matches = tuple(
        scan for scan in REFERENCE_SCANS if scan.circuit_id == circuit_id
    )
    if len(matches) != 1:
        raise CoverageArtifactError(
            f"coverage circuit {circuit_id!r} lacks one frozen reference scan"
        )
    return matches[0]


def _parity_scope_for_pattern(pattern: Any) -> str:
    if pattern.dimension == "device_geometry":
        if pattern.basis in {"oa_spatial_observation", "topology_declaration"}:
            raise CoverageArtifactError(
                "device-geometry pattern has an OA/topology-only evidence basis"
            )
        return "device_geometry_parity"
    return "oa_spatial_parity"


@dataclass(frozen=True, slots=True, kw_only=True)
class CoverageUnmeasurableReason:
    code: str
    detail: str
    evidence: str

    def __post_init__(self) -> None:
        _text(self.code, field="coverage unmeasurable reason code", pattern=_BLOCKER_CODE_RE)
        expected_details = {
            code: detail for code, detail in _GEOMETRY_UNMEASURABLE_REASONS.values()
        }
        if self.code not in expected_details or self.detail != expected_details[self.code]:
            raise CoverageArtifactError("coverage unmeasurable reason is not canonical")
        _evidence_path(self.evidence)

    def canonical_data(self) -> dict[str, str]:
        return {
            "code": self.code,
            "detail": self.detail,
            "evidence": self.evidence,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class CoverageParityScope:
    scope_id: str
    circuit_id: str
    availability: str
    covered_patterns: int | None
    identifiable_patterns: int | None
    reason: CoverageUnmeasurableReason | None

    def __post_init__(self) -> None:
        self.assert_authoritative()

    def assert_authoritative(self) -> None:
        if self.scope_id not in _PARITY_SCOPES:
            raise CoverageArtifactError("coverage parity scope is not canonical")
        scan = _reference_scan(self.circuit_id)
        if self.scope_id == "oa_spatial_parity":
            expected_availability = "measurable"
            expected_reason = None
        elif scan.evidence_scope == "complete_exact_gds":
            expected_availability = "measurable"
            expected_reason = None
        elif scan.evidence_scope == "complete_unmapped_gds":
            expected_availability = "measurable_layer_level"
            reason_spec = _GEOMETRY_UNMEASURABLE_REASONS[scan.evidence_scope]
            expected_reason = CoverageUnmeasurableReason(
                code=reason_spec[0],
                detail=reason_spec[1],
                evidence=scan.manifest.path,
            )
        else:
            expected_availability = "unmeasurable"
            reason_spec = _GEOMETRY_UNMEASURABLE_REASONS.get(scan.evidence_scope)
            if reason_spec is None:
                raise CoverageArtifactError(
                    "reference scan lacks a typed device-geometry availability contract"
                )
            expected_reason = CoverageUnmeasurableReason(
                code=reason_spec[0],
                detail=reason_spec[1],
                evidence=scan.manifest.path,
            )
        if self.availability != expected_availability:
            raise CoverageArtifactError(
                "coverage scope availability contradicts the reference authority"
            )
        if self.availability in {"measurable", "measurable_layer_level"}:
            for field in ("covered_patterns", "identifiable_patterns"):
                value = getattr(self, field)
                if type(value) is not int or value < 0 or value > _MAX_ROWS:
                    raise CoverageArtifactError(
                        f"measurable scope {field} must be an exact bounded integer"
                    )
            if self.availability == "measurable" and self.reason is not None:
                raise CoverageArtifactError("measurable coverage scope cannot have a reason")
            if (
                self.availability == "measurable_layer_level"
                and self.reason != expected_reason
            ):
                raise CoverageArtifactError(
                    "layer-level measurable scope lost instance-attribution reason"
                )
        else:
            if self.covered_patterns is not None or self.identifiable_patterns is not None:
                raise CoverageArtifactError(
                    "unmeasurable coverage scope must have null counts"
                )
            if type(self.reason) is not CoverageUnmeasurableReason:
                raise CoverageArtifactError(
                    "unmeasurable coverage scope requires a typed reason"
                )
            if self.reason != expected_reason:
                raise CoverageArtifactError(
                    "coverage unmeasurable reason contradicts the reference authority"
                )

    def canonical_data(self) -> dict[str, object]:
        self.assert_authoritative()
        return {
            "availability": self.availability,
            "covered_patterns": self.covered_patterns,
            "identifiable_patterns": self.identifiable_patterns,
            "reason": None if self.reason is None else self.reason.canonical_data(),
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class CoverageRow:
    pattern_id: str
    basis: str
    parity_scope: str
    dimension: str
    description: str
    primitive: str | None
    mapped: bool
    materialized: bool
    checked: bool
    gap: str | None
    evidence: str

    def __post_init__(self) -> None:
        self.assert_authoritative()

    def assert_authoritative(self) -> None:
        """Replay the exact row schema after any hostile object mutation."""

        _text(self.pattern_id, field="coverage pattern id", pattern=_ID_RE)
        if type(self.basis) is not str or self.basis not in {
            "complete_gds_observation",
            "oa_spatial_observation",
            "topology_declaration",
            "audit_gap",
        }:
            raise CoverageArtifactError("coverage basis is not canonical")
        if self.parity_scope not in _PARITY_SCOPES:
            raise CoverageArtifactError("coverage parity scope is not canonical")
        if type(self.dimension) is not str or self.dimension not in DIMENSIONS:
            raise CoverageArtifactError("coverage dimension is not canonical")
        _text(self.description, field="coverage description")
        if _COORDINATE_RE.search(self.description):
            raise CoverageArtifactError("coverage description leaks a reference coordinate")
        for field in ("mapped", "materialized", "checked"):
            if type(getattr(self, field)) is not bool:
                raise CoverageArtifactError(f"{field} must be an exact boolean")
        if self.mapped:
            _text(self.primitive, field="mapped coverage primitive", pattern=_ID_RE)
        elif self.primitive is not None:
            raise CoverageArtifactError("unmapped coverage row cannot name a primitive")
        if not self.mapped and (self.materialized or self.checked):
            raise CoverageArtifactError("unmapped coverage row cannot claim implementation")
        if self.checked and not self.materialized:
            raise CoverageArtifactError("checked coverage must also be materialized")
        if self.covered:
            if self.gap is not None:
                raise CoverageArtifactError("covered row cannot retain a gap")
        else:
            _text(self.gap, field="uncovered coverage gap")
        _evidence_path(self.evidence)

    @property
    def covered(self) -> bool:
        return self.mapped and self.materialized and self.checked

    def canonical_data(self) -> dict[str, object]:
        self.assert_authoritative()
        return {
            "pattern_id": self.pattern_id,
            "basis": self.basis,
            "parity_scope": self.parity_scope,
            "dimension": self.dimension,
            "description": self.description,
            "primitive": self.primitive,
            "mapped": self.mapped,
            "materialized": self.materialized,
            "checked": self.checked,
            "gap": self.gap,
            "evidence": self.evidence,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class CoverageArtifact:
    circuit_id: str
    registry_sha256: str
    reference_scan_sha256: str
    covered_patterns: int
    identifiable_patterns: int
    scopes: Tuple[CoverageParityScope, ...]
    rows: Tuple[CoverageRow, ...]

    def __post_init__(self) -> None:
        self.assert_authoritative()

    def assert_authoritative(self) -> None:
        """Bind public artifact construction to the frozen global denominator."""

        _text(self.circuit_id, field="coverage circuit id", pattern=_ID_RE)
        if self.circuit_id not in COVERAGE_OUTPUTS:
            raise CoverageArtifactError("coverage circuit id is not in the frozen domain")
        _digest(self.registry_sha256, field="coverage registry SHA-256")
        _digest(self.reference_scan_sha256, field="coverage reference scan SHA-256")
        for field in ("covered_patterns", "identifiable_patterns"):
            value = getattr(self, field)
            if type(value) is not int or value < 0 or value > _MAX_ROWS:
                raise CoverageArtifactError(f"{field} must be an exact bounded integer")
        if type(self.rows) is not tuple or not self.rows or len(self.rows) > _MAX_ROWS:
            raise CoverageArtifactError("coverage rows must be a non-empty exact bounded tuple")
        if any(type(item) is not CoverageRow for item in self.rows):
            raise CoverageArtifactError("coverage rows must contain exact typed rows")
        if len({item.pattern_id for item in self.rows}) != len(self.rows):
            raise CoverageArtifactError("coverage pattern ids must be unique")
        if self.identifiable_patterns != len(self.rows):
            raise CoverageArtifactError("coverage denominator differs from row inventory")
        if self.covered_patterns != sum(item.covered for item in self.rows):
            raise CoverageArtifactError("covered pattern count does not replay rows")
        if type(self.scopes) is not tuple or any(
            type(scope) is not CoverageParityScope for scope in self.scopes
        ):
            raise CoverageArtifactError(
                "coverage scopes must be the exact typed parity-scope inventory"
            )
        if tuple(scope.scope_id for scope in self.scopes) != _PARITY_SCOPES:
            raise CoverageArtifactError(
                "coverage scopes must be the exact typed parity-scope inventory"
            )
        for scope in self.scopes:
            scope.assert_authoritative()
            if scope.circuit_id != self.circuit_id:
                raise CoverageArtifactError("coverage scope circuit binding is stale")
            scoped_rows = tuple(
                row for row in self.rows if row.parity_scope == scope.scope_id
            )
            if scope.availability == "measurable":
                if (
                    scope.covered_patterns != sum(row.covered for row in scoped_rows)
                    or scope.identifiable_patterns != len(scoped_rows)
                ):
                    raise CoverageArtifactError(
                        "measurable coverage scope counts do not replay rows"
                    )
        patterns = tuple(
            item for item in PHASE1_PATTERNS if item.circuit_id == self.circuit_id
        )
        if tuple(item.pattern_id for item in self.rows) != tuple(
            item.pattern_id for item in patterns
        ):
            raise CoverageArtifactError(
                "coverage denominator differs from the frozen pattern inventory"
            )
        if self.registry_sha256 != registry_sha256():
            raise CoverageArtifactError("coverage registry SHA-256 is stale")
        if self.reference_scan_sha256 != _reference_scan_sha256(self.circuit_id):
            raise CoverageArtifactError("coverage reference scan SHA-256 is stale")
        primitive_by_id = {item.primitive_id: item for item in PRIMITIVES}
        for pattern, row in zip(patterns, self.rows, strict=True):
            primitive = primitive_by_id.get(pattern.primitive_id)
            if primitive is None:
                expected = {
                    "pattern_id": pattern.pattern_id,
                    "basis": pattern.basis,
                    "parity_scope": _parity_scope_for_pattern(pattern),
                    "dimension": pattern.dimension,
                    "description": pattern.description,
                    "primitive": None,
                    "mapped": False,
                    "materialized": False,
                    "checked": False,
                    "gap": f"No primitive maps pattern {pattern.pattern_id}.",
                    "evidence": pattern.evidence[-1],
                }
            else:
                expected = {
                    "pattern_id": pattern.pattern_id,
                    "basis": pattern.basis,
                    "parity_scope": _parity_scope_for_pattern(pattern),
                    "dimension": pattern.dimension,
                    "description": pattern.description,
                    "primitive": primitive.primitive_id,
                    "mapped": True,
                    "materialized": primitive.materializer_status == "materialized",
                    "checked": primitive.checker_status == "exact_gds",
                    "gap": _gap_for(primitive),
                    "evidence": primitive.evidence[-1],
                }
            if row.canonical_data() != expected:
                raise CoverageArtifactError(
                    "coverage row does not replay the frozen registry"
                )

    @property
    def schema(self) -> str:
        return _SCHEMA

    def canonical_data(self) -> dict[str, object]:
        self.assert_authoritative()
        return {
            "schema": _SCHEMA,
            "circuit_id": self.circuit_id,
            "registry_sha256": self.registry_sha256,
            "reference_scan_sha256": self.reference_scan_sha256,
            "pattern_inventory": {
                "covered_patterns": self.covered_patterns,
                "identifiable_patterns": self.identifiable_patterns,
            },
            "scopes": {
                scope.scope_id: scope.canonical_data() for scope in self.scopes
            },
            "rows": [item.canonical_data() for item in self.rows],
        }

    @property
    def artifact_sha256(self) -> str:
        return hashlib.sha256(_canonical_json_bytes(self.canonical_data())).hexdigest()

    def document_data(self) -> dict[str, object]:
        return {**self.canonical_data(), "artifact_sha256": self.artifact_sha256}


def _reference_scan_sha256(circuit_id: str) -> str:
    matches = tuple(
        scan for scan in REFERENCE_SCANS if scan.circuit_id == circuit_id
    )
    if len(matches) != 1:
        raise CoverageArtifactError(
            f"coverage circuit {circuit_id!r} lacks one frozen reference scan"
        )
    return hashlib.sha256(_canonical_json_bytes(asdict(matches[0]))).hexdigest()


def _gap_for(primitive: Any) -> str | None:
    if (
        primitive.materializer_status == "materialized"
        and primitive.checker_status == "exact_gds"
    ):
        return None
    if primitive.gap is not None:
        return primitive.gap
    return (
        f"Primitive {primitive.primitive_id} is not exact-GDS closed: "
        f"materializer={primitive.materializer_status}, "
        f"checker={primitive.checker_status}."
    )


def _build_coverage_scopes(
    circuit_id: str, rows: Tuple[CoverageRow, ...]
) -> Tuple[CoverageParityScope, ...]:
    scan = _reference_scan(circuit_id)
    scopes = []
    for scope_id in _PARITY_SCOPES:
        scoped_rows = tuple(row for row in rows if row.parity_scope == scope_id)
        if (
            scope_id == "device_geometry_parity"
            and scan.evidence_scope == "oa_spatial_only"
        ):
            reason_spec = _GEOMETRY_UNMEASURABLE_REASONS.get(scan.evidence_scope)
            if reason_spec is None:
                raise CoverageArtifactError(
                    "reference scan lacks a typed device-geometry availability contract"
                )
            scopes.append(CoverageParityScope(
                scope_id=scope_id,
                circuit_id=circuit_id,
                availability="unmeasurable",
                covered_patterns=None,
                identifiable_patterns=None,
                reason=CoverageUnmeasurableReason(
                    code=reason_spec[0],
                    detail=reason_spec[1],
                    evidence=scan.manifest.path,
                ),
            ))
        elif (
            scope_id == "device_geometry_parity"
            and scan.evidence_scope == "complete_unmapped_gds"
        ):
            reason_spec = _GEOMETRY_UNMEASURABLE_REASONS[scan.evidence_scope]
            scopes.append(CoverageParityScope(
                scope_id=scope_id,
                circuit_id=circuit_id,
                availability="measurable_layer_level",
                covered_patterns=sum(row.covered for row in scoped_rows),
                identifiable_patterns=len(scoped_rows),
                reason=CoverageUnmeasurableReason(
                    code=reason_spec[0],
                    detail=reason_spec[1],
                    evidence=scan.manifest.path,
                ),
            ))
        else:
            scopes.append(CoverageParityScope(
                scope_id=scope_id,
                circuit_id=circuit_id,
                availability="measurable",
                covered_patterns=sum(row.covered for row in scoped_rows),
                identifiable_patterns=len(scoped_rows),
                reason=None,
            ))
    return tuple(scopes)


def build_coverage_artifact(repo_root: Path, circuit_id: str) -> CoverageArtifact:
    """Recompute one circuit's coverage from the frozen registry denominator."""

    root = Path(repo_root).resolve()
    validate_registry(root)
    if type(circuit_id) is not str or circuit_id not in COVERAGE_OUTPUTS:
        raise CoverageArtifactError("coverage circuit id is not in the frozen domain")
    primitive_by_id = {item.primitive_id: item for item in PRIMITIVES}
    patterns = tuple(
        item for item in PHASE1_PATTERNS if item.circuit_id == circuit_id
    )
    if not patterns:
        raise CoverageArtifactError("coverage circuit has an empty pattern denominator")
    rows = []
    for pattern in patterns:
        primitive = primitive_by_id.get(pattern.primitive_id)
        if primitive is None:
            rows.append(CoverageRow(
                pattern_id=pattern.pattern_id,
                basis=pattern.basis,
                parity_scope=_parity_scope_for_pattern(pattern),
                dimension=pattern.dimension,
                description=pattern.description,
                primitive=None,
                mapped=False,
                materialized=False,
                checked=False,
                gap=f"No primitive maps pattern {pattern.pattern_id}.",
                evidence=pattern.evidence[-1],
            ))
            continue
        materialized = primitive.materializer_status == "materialized"
        checked = primitive.checker_status == "exact_gds"
        rows.append(CoverageRow(
            pattern_id=pattern.pattern_id,
            basis=pattern.basis,
            parity_scope=_parity_scope_for_pattern(pattern),
            dimension=pattern.dimension,
            description=pattern.description,
            primitive=primitive.primitive_id,
            mapped=True,
            materialized=materialized,
            checked=checked,
            gap=_gap_for(primitive),
            evidence=primitive.evidence[-1],
        ))
    typed_rows = tuple(rows)
    return CoverageArtifact(
        circuit_id=circuit_id,
        registry_sha256=registry_sha256(),
        reference_scan_sha256=_reference_scan_sha256(circuit_id),
        covered_patterns=sum(item.covered for item in typed_rows),
        identifiable_patterns=len(typed_rows),
        scopes=_build_coverage_scopes(circuit_id, typed_rows),
        rows=typed_rows,
    )


def render_coverage_json(repo_root: Path, circuit_id: str) -> str:
    artifact = build_coverage_artifact(repo_root, circuit_id)
    return _canonical_json_bytes(artifact.document_data()).decode("ascii") + "\n"


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise CoverageArtifactError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def verify_coverage_document(repo_root: Path, path: Path) -> CoverageArtifact:
    """Rebuild every row and require exact equality with one on-disk artifact."""

    candidate = Path(path)
    try:
        raw_bytes = candidate.read_bytes()
    except OSError as exc:
        raise CoverageArtifactError(f"coverage artifact cannot be read: {candidate}") from exc
    if not raw_bytes or len(raw_bytes) > 1_000_000:
        raise CoverageArtifactError("coverage artifact bytes are empty or unbounded")
    try:
        payload = json.loads(
            raw_bytes.decode("ascii"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                CoverageArtifactError(f"non-finite JSON constant {value!r}")
            ),
        )
    except CoverageArtifactError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CoverageArtifactError("coverage artifact is not canonical JSON") from exc
    if type(payload) is not dict:
        raise CoverageArtifactError("coverage artifact root must be an exact object")
    expected_keys = {
        "schema",
        "circuit_id",
        "registry_sha256",
        "reference_scan_sha256",
        "pattern_inventory",
        "scopes",
        "rows",
        "artifact_sha256",
    }
    if set(payload) != expected_keys:
        raise CoverageArtifactError("coverage artifact keys are not exact")
    if payload.get("schema") != _SCHEMA:
        raise CoverageArtifactError("coverage artifact schema is not canonical")
    circuit_id = payload.get("circuit_id")
    if type(circuit_id) is not str or circuit_id not in COVERAGE_OUTPUTS:
        raise CoverageArtifactError("coverage artifact circuit id is not canonical")
    expected = build_coverage_artifact(Path(repo_root), circuit_id)
    if payload != expected.document_data():
        raise CoverageArtifactError(
            "coverage artifact does not replay the frozen registry and reference scan"
        )
    if raw_bytes != render_coverage_json(Path(repo_root), circuit_id).encode("ascii"):
        raise CoverageArtifactError("coverage artifact bytes are not canonical")
    return expected


def _unmeasurable_geometry_scope_explanation(reason_code: str) -> str:
    if reason_code == "REFERENCE_LOGICAL_INSTANCE_GEOMETRY_MAP_UNAVAILABLE":
        return (
            "Layer/region geometry: measurable under the complete GDS polygon "
            "inventory. Logical-instance polygon ownership: typed `unmeasurable` "
            "because shared or abutted geometry has no complete instance map."
        )
    return (
        "OA spatial parity remains measurable, while device-geometry parity is "
        "explicitly unmeasurable under the bound human-reference authority."
    )


def render_convergence_verdict_markdown(repo_root: Path, circuit_id: str) -> str:
    """Render one bounded verdict from replayed coverage and lower-bound evidence."""

    root = Path(repo_root).resolve()
    artifact = build_coverage_artifact(root, circuit_id)
    scan = next(
        item for item in REFERENCE_SCANS if item.circuit_id == circuit_id
    )
    classification = _VERDICT_CLASSIFICATION[circuit_id]
    evidence_path = _REACHABILITY_EVIDENCE[circuit_id]
    if not (root / PurePosixPath(evidence_path)).is_file():
        raise CoverageArtifactError(
            f"convergence evidence is missing: {evidence_path}"
        )
    scope_by_id = {scope.scope_id: scope for scope in artifact.scopes}
    measurable_scope_ids = {
        scope.scope_id
        for scope in artifact.scopes
        if scope.availability in {"measurable", "measurable_layer_level"}
    }
    gaps = tuple(
        row.pattern_id
        for row in artifact.rows
        if row.parity_scope in measurable_scope_ids and not row.covered
    )
    if not gaps:
        gap_text = "none"
    else:
        gap_text = ", ".join(f"`{item}`" for item in gaps)
    coverage_path = COVERAGE_OUTPUTS[circuit_id]
    lower_bound_matrix, lower_bound_row, reference_pdk = (
        _load_reference_lower_bound_row(
            root,
            circuit_id=circuit_id,
            source_path=scan.source.path,
            manifest_path=scan.manifest.path,
        )
    )
    frozen_inputs = lower_bound_matrix["frozen_inputs"]
    if type(frozen_inputs) is not dict:
        raise CoverageArtifactError("lower-bound frozen-input binding is not an object")
    if lower_bound_row["status"] == "supported":
        lower_bound_nm2 = lower_bound_row["lower_bound_nm2"]
        if type(lower_bound_nm2) is not int:
            raise CoverageArtifactError("supported lower-bound row lost its numeric value")
        lower_bound_whole, lower_bound_fraction = divmod(
            lower_bound_nm2, 1_000_000
        )
        lower_bound_text = (
            f"- PDK-legal lower bound: "
            f"`{lower_bound_whole}.{lower_bound_fraction:06d} um2` for reference PDK "
            f"`{reference_pdk}` (matrix `{_LOWER_BOUND_MATRIX_PATH}`, artifact SHA-256 "
            f"`{lower_bound_matrix['artifact_sha256']}`, input SHA-256 "
            f"`{lower_bound_row['input_sha256']}`, receipt SHA-256 "
            f"`{lower_bound_row['receipt_sha256']}`). The matrix is bound to frozen inputs "
            f"`{frozen_inputs['path']}` (`{frozen_inputs['sha256']}`)."
        )
    else:
        blockers = lower_bound_row["blockers"]
        if type(blockers) is not list:
            raise CoverageArtifactError("unmeasurable lower-bound blockers are not a list")
        blocker_text = "; ".join(
            f"`{blocker['code']}`: {blocker['detail']}"
            for blocker in blockers
        )
        lower_bound_text = (
            f"- PDK-legal lower bound: typed `unmeasurable` for reference PDK "
            f"`{reference_pdk}`. Blocker: {blocker_text}. Matrix "
            f"`{_LOWER_BOUND_MATRIX_PATH}`, artifact SHA-256 "
            f"`{lower_bound_matrix['artifact_sha256']}`, frozen inputs "
            f"`{frozen_inputs['path']}` (`{frozen_inputs['sha256']}`). No numeric "
            "lower bound is fabricated for this row."
        )
    geometry_scope = scope_by_id["device_geometry_parity"]
    if geometry_scope.availability in {"unmeasurable", "measurable_layer_level"}:
        if type(geometry_scope.reason) is not CoverageUnmeasurableReason:
            raise CoverageArtifactError(
                "unmeasurable device-geometry scope lost its typed reason"
            )
        classification_reason = _unmeasurable_geometry_scope_explanation(
            geometry_scope.reason.code
        )
    else:
        classification_reason = (
            "A complete exact human-reference GDS exists, so the remaining blockers "
            "are the uncovered materializer/checker and domain-definition rows below."
        )
    oa_scope = scope_by_id["oa_spatial_parity"]
    if (
        oa_scope.availability != "measurable"
        or type(oa_scope.covered_patterns) is not int
        or type(oa_scope.identifiable_patterns) is not int
    ):
        raise CoverageArtifactError("OA spatial parity unexpectedly became unmeasurable")
    oa_scope_text = (
        f"- OA spatial parity: measurable "
        f"`{oa_scope.covered_patterns}/{oa_scope.identifiable_patterns}`."
    )
    if geometry_scope.availability == "measurable":
        if (
            type(geometry_scope.covered_patterns) is not int
            or type(geometry_scope.identifiable_patterns) is not int
        ):
            raise CoverageArtifactError("measurable device-geometry scope lost its counts")
        geometry_scope_text = (
            f"- Device geometry parity: measurable "
            f"`{geometry_scope.covered_patterns}/{geometry_scope.identifiable_patterns}`."
        )
    elif geometry_scope.availability == "measurable_layer_level":
        if (
            type(geometry_scope.covered_patterns) is not int
            or type(geometry_scope.identifiable_patterns) is not int
            or type(geometry_scope.reason) is not CoverageUnmeasurableReason
        ):
            raise CoverageArtifactError(
                "layer-level measurable device geometry lost counts or attribution reason"
            )
        geometry_scope_text = (
            "- Device geometry parity: measurable at flattened layer/region level "
            f"`{geometry_scope.covered_patterns}/{geometry_scope.identifiable_patterns}`; "
            "logical-instance polygon attribution remains unavailable. Reason "
            f"`{geometry_scope.reason.code}`: {geometry_scope.reason.detail} "
            f"Evidence: `{geometry_scope.reason.evidence}`."
        )
    else:
        if type(geometry_scope.reason) is not CoverageUnmeasurableReason:
            raise CoverageArtifactError(
                "unmeasurable device-geometry scope lost its typed reason"
            )
        reason_detail = geometry_scope.reason.detail
        if reason_detail[-1] not in ".!?":
            reason_detail += "."
        geometry_scope_text = (
            "- Device geometry parity: typed `unmeasurable`. Reason "
            f"`{geometry_scope.reason.code}`: {reason_detail} "
            f"Evidence: `{geometry_scope.reason.evidence}`."
        )
    lines = [
        f"# Convergence verdict: {circuit_id}",
        "",
        f"Classification: **{classification}**.",
        "",
        classification_reason,
        "",
        "## Completeness",
        "",
        f"Pattern inventory: `{artifact.covered_patterns}/{artifact.identifiable_patterns}`. "
        f"The inventory is partitioned exactly once across the two parity scopes. "
        f"The canonical table is `{coverage_path}` with artifact SHA-256 "
        f"`{artifact.artifact_sha256}` and registry SHA-256 "
        f"`{artifact.registry_sha256}`.",
        "",
        oa_scope_text,
        geometry_scope_text,
        "",
        f"Uncovered patterns: {gap_text}.",
        "",
        "## Reachability instruments",
        "",
        lower_bound_text,
        "",
        "- Exact-domain search classification: `best_observed`; "
        "`domain_exhausted=false`; `optimal=false`. Existing exact or signed output "
        "is retained only as the current observation. No post-Phase-2 full-denominator "
        "receipt exists, so this verdict does not claim convergence or optimality.",
        "",
        "## Bound evidence",
        "",
        f"- Reference scan: `{scan.gds.path}` (`{scan.gds.sha256}`), scope "
        f"`{scan.evidence_scope}`.",
        f"- Current physical/search evidence: `{evidence_path}`.",
        f"- Coverage evidence: `{coverage_path}`.",
        "",
        "Physical legality remains a required independent gate. Post-PEX electrical "
        "claims require a calibrated extractor receipt; an uncalibrated geometric "
        "artifact remains `physical_pex_extracted` and is not electrical verification.",
        "",
    ]
    return "\n".join(lines)


__all__ = [
    "COVERAGE_OUTPUTS",
    "VERDICT_OUTPUTS",
    "CoverageArtifact",
    "CoverageArtifactError",
    "CoverageParityScope",
    "CoverageRow",
    "CoverageUnmeasurableReason",
    "build_coverage_artifact",
    "render_coverage_json",
    "render_convergence_verdict_markdown",
    "verify_coverage_document",
]
