"""Strict non-signoff target DATA for compact physical-search admission."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Mapping

from clay.layout.compact_intent import CompactIntentCandidate


PHYSICAL_SEARCH_TARGET_SCHEMA = "clay.physical-search-target/v1"
PHYSICAL_SEARCH_ADMISSION_SCHEMA = "clay.physical-search-admission/v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER = re.compile(r"[a-z][a-z0-9_]*(?:[.-][a-z0-9_]+)*")
_CELL_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_MAX_MEASURE = 1.0e18
_ROOT_KEYS = frozenset(
    {
        "schema",
        "circuit_id",
        "top_cell",
        "pdk_id",
        "source_sha256",
        "reference_scope",
        "baseline",
        "acceptance",
        "regions",
        "relations",
    }
)
_BASELINE_KEYS = frozenset(
    {
        "candidate_sha256",
        "gds_sha256",
        "bbox_area_um2",
        "whitespace_um2",
        "wire_length_um",
        "via_count",
    }
)
_ACCEPTANCE_KEYS = frozenset(
    {
        "require_derived",
        "max_bbox_area_ratio",
        "max_whitespace_fraction",
        "min_aspect",
        "max_aspect",
        "max_wire_length_ratio",
        "max_via_count_ratio",
    }
)
_REGION_KEYS = frozenset({"region_id", "functional_groups"})
_RELATION_KEYS = frozenset({"kind", "subject", "object"})
_RELATION_ORDER = {"left_of": 0, "above": 1}
_ADMISSION_BLOCKER_ORDER = {
    code: index
    for index, code in enumerate(
        (
            "AUTHORING_CANDIDATE",
            "PLACEMENT_STATE_MISSING",
            "BBOX_AREA_LIMIT",
            "WHITESPACE_LIMIT",
            "ASPECT_LIMIT",
            "WIRE_LENGTH_LIMIT",
            "VIA_COUNT_LIMIT",
            "REGION_MISSING",
            "REFERENCE_RELATION_MISMATCH",
        )
    )
}
_ADMISSION_METRIC_KEYS = frozenset(
    {
        "bbox_nm",
        "bbox_width_nm",
        "bbox_height_nm",
        "bbox_area_um2",
        "whitespace_um2",
        "wire_length_um",
        "via_count",
        "ring_perimeter_um",
        "ring_max_perimeter_um",
        "ring_span_waste_um2",
        "physical_signature_sha256",
        "macro_consumption_sha256",
        "placement_state_sha256",
        "functional_group_bboxes_nm",
    }
)


class PhysicalSearchTargetError(ValueError):
    """The committed pre-exact target is malformed, stale, or over-authoritative."""


def _identifier(value: object, *, label: str) -> str:
    if type(value) is not str or _IDENTIFIER.fullmatch(value) is None:
        raise PhysicalSearchTargetError(f"{label} must be a canonical identifier")
    return value


def _cell_identifier(value: object, *, label: str) -> str:
    if type(value) is not str or _CELL_IDENTIFIER.fullmatch(value) is None:
        raise PhysicalSearchTargetError(f"{label} must be a canonical cell identifier")
    return value


def _sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise PhysicalSearchTargetError(f"{label} must be a lowercase SHA-256")
    return value


def _measure(
    value: object,
    *,
    label: str,
    positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PhysicalSearchTargetError(f"{label} must be an exact finite number")
    try:
        measured = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise PhysicalSearchTargetError(
            f"{label} must be a bounded finite "
            f"{'positive' if positive else 'non-negative'} number"
        ) from exc
    lower_ok = measured > 0.0 if positive else measured >= 0.0
    if not math.isfinite(measured) or not lower_ok or measured > _MAX_MEASURE:
        raise PhysicalSearchTargetError(
            f"{label} must be a bounded finite {'positive' if positive else 'non-negative'} number"
        )
    return 0.0 if measured == 0.0 else measured


def _count(value: object, *, label: str) -> int:
    if type(value) is not int or not 0 <= value <= int(_MAX_MEASURE):
        raise PhysicalSearchTargetError(f"{label} must be a bounded non-negative integer")
    return value


def _scaled_count(count: int, ratio: float) -> int:
    scaled = Decimal(count) * Decimal(str(ratio))
    return int(scaled.to_integral_value(rounding=ROUND_FLOOR))


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise PhysicalSearchTargetError(f"{label} must be a JSON object")
    return value


def _exact_keys(
    value: Mapping[str, object],
    expected: frozenset[str],
    *,
    label: str,
) -> None:
    actual = frozenset(value)
    unknown = sorted(actual - expected)
    missing = sorted(expected - actual)
    if unknown:
        raise PhysicalSearchTargetError(f"{label} has unknown keys: {unknown}")
    if missing:
        raise PhysicalSearchTargetError(f"{label} is missing keys: {missing}")


def _contains_png(value: object) -> bool:
    if isinstance(value, str):
        normalized = value.replace("\\", "/")
        return PurePosixPath(normalized).suffix.casefold() == ".png"
    if isinstance(value, Mapping):
        return any(_contains_png(key) or _contains_png(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_png(item) for item in value)
    return False


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class PhysicalSearchRegion:
    """One coordinate-free generated-layout region named by functional groups."""

    region_id: str
    functional_groups: tuple[str, ...]

    def __post_init__(self) -> None:
        _identifier(self.region_id, label="physical-search region ID")
        if type(self.functional_groups) is not tuple or not self.functional_groups:
            raise PhysicalSearchTargetError(
                f"region {self.region_id} functional_groups must be a non-empty tuple"
            )
        groups = tuple(
            _identifier(item, label=f"region {self.region_id} functional group")
            for item in self.functional_groups
        )
        if len(set(groups)) != len(groups):
            raise PhysicalSearchTargetError(
                f"region {self.region_id} repeats a functional group"
            )
        if _contains_png((self.region_id, groups)):
            raise PhysicalSearchTargetError("PNG cannot appear in a physical-search region")
        object.__setattr__(self, "functional_groups", tuple(sorted(groups)))

    def canonical_data(self) -> dict[str, object]:
        return {
            "region_id": self.region_id,
            "functional_groups": list(self.functional_groups),
        }


@dataclass(frozen=True)
class PhysicalSearchRelation:
    """A translation-free ordering relation between two generated regions."""

    kind: str
    subject: str
    object: str

    def __post_init__(self) -> None:
        if type(self.kind) is not str or self.kind not in _RELATION_ORDER:
            raise PhysicalSearchTargetError(
                "physical-search relation kind must be left_of or above"
            )
        _identifier(self.subject, label="physical-search relation subject")
        _identifier(self.object, label="physical-search relation object")
        if _contains_png((self.subject, self.object)):
            raise PhysicalSearchTargetError(
                "PNG cannot appear in a physical-search relation"
            )
        if self.subject == self.object:
            raise PhysicalSearchTargetError("physical-search relation cannot be reflexive")

    def canonical_data(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "subject": self.subject,
            "object": self.object,
        }


@dataclass(frozen=True)
class PhysicalSearchTarget:
    """Committed staging thresholds; never an exact-GDS or signoff receipt."""

    schema: str
    circuit_id: str
    top_cell: str
    pdk_id: str
    source_sha256: str
    reference_scope: str
    baseline_candidate_sha256: str
    baseline_gds_sha256: str
    baseline_bbox_area_um2: float
    baseline_whitespace_um2: float
    baseline_wire_length_um: float
    baseline_via_count: int
    require_derived: bool
    max_bbox_area_ratio: float
    max_whitespace_fraction: float
    min_aspect: float
    max_aspect: float
    max_wire_length_ratio: float
    max_via_count_ratio: float
    regions: tuple[PhysicalSearchRegion, ...]
    relations: tuple[PhysicalSearchRelation, ...]

    def __post_init__(self) -> None:
        if self.schema != PHYSICAL_SEARCH_TARGET_SCHEMA:
            raise PhysicalSearchTargetError("physical-search target schema is unsupported")
        _identifier(self.circuit_id, label="physical-search circuit ID")
        _cell_identifier(self.top_cell, label="physical-search top cell")
        _identifier(self.pdk_id, label="physical-search PDK ID")
        if _contains_png((self.circuit_id, self.top_cell, self.pdk_id)):
            raise PhysicalSearchTargetError("PNG cannot appear in a physical-search target")
        _sha256(self.source_sha256, label="physical-search source SHA-256")
        if self.reference_scope != "spatial_advisory":
            raise PhysicalSearchTargetError(
                "pre-exact target reference_scope must be spatial_advisory"
            )
        _sha256(
            self.baseline_candidate_sha256,
            label="physical-search baseline candidate SHA-256",
        )
        _sha256(
            self.baseline_gds_sha256,
            label="physical-search baseline GDS SHA-256",
        )
        area = _measure(
            self.baseline_bbox_area_um2,
            label="baseline bbox_area_um2",
            positive=True,
        )
        whitespace = _measure(
            self.baseline_whitespace_um2,
            label="baseline whitespace_um2",
        )
        wire = _measure(
            self.baseline_wire_length_um,
            label="baseline wire_length_um",
        )
        via = _count(self.baseline_via_count, label="baseline via_count")
        if whitespace > area:
            raise PhysicalSearchTargetError("baseline whitespace exceeds bbox area")
        if type(self.require_derived) is not bool:
            raise PhysicalSearchTargetError("require_derived must be an exact bool")
        area_ratio = _measure(
            self.max_bbox_area_ratio,
            label="max_bbox_area_ratio",
            positive=True,
        )
        whitespace_fraction = _measure(
            self.max_whitespace_fraction,
            label="max_whitespace_fraction",
        )
        min_aspect = _measure(self.min_aspect, label="min_aspect", positive=True)
        max_aspect = _measure(self.max_aspect, label="max_aspect", positive=True)
        wire_ratio = _measure(
            self.max_wire_length_ratio,
            label="max_wire_length_ratio",
            positive=True,
        )
        via_ratio = _measure(
            self.max_via_count_ratio,
            label="max_via_count_ratio",
            positive=True,
        )
        derived_wire_threshold = wire * wire_ratio
        if (
            not math.isfinite(derived_wire_threshold)
            or derived_wire_threshold > _MAX_MEASURE
        ):
            raise PhysicalSearchTargetError(
                "derived wire-length threshold must remain finite and bounded"
            )
        derived_via_threshold = _scaled_count(via, via_ratio)
        if derived_via_threshold > int(_MAX_MEASURE):
            raise PhysicalSearchTargetError(
                "derived via-count threshold must remain finite and bounded"
            )
        if area_ratio > 1.0:
            raise PhysicalSearchTargetError("max_bbox_area_ratio cannot exceed 1")
        if whitespace_fraction > 1.0:
            raise PhysicalSearchTargetError("max_whitespace_fraction cannot exceed 1")
        if min_aspect > max_aspect:
            raise PhysicalSearchTargetError("aspect range is inverted")
        if type(self.regions) is not tuple or not self.regions or any(
            type(item) is not PhysicalSearchRegion for item in self.regions
        ):
            raise PhysicalSearchTargetError("physical-search regions must be typed")
        region_ids = tuple(item.region_id for item in self.regions)
        if len(set(region_ids)) != len(region_ids):
            raise PhysicalSearchTargetError("physical-search target repeats a region")
        owned_groups = tuple(
            group
            for region in self.regions
            for group in region.functional_groups
        )
        if len(set(owned_groups)) != len(owned_groups):
            raise PhysicalSearchTargetError(
                "a functional group is owned by multiple regions"
            )
        if type(self.relations) is not tuple or not self.relations or any(
            type(item) is not PhysicalSearchRelation for item in self.relations
        ):
            raise PhysicalSearchTargetError("physical-search relations must be typed")
        relation_keys = tuple(
            (item.kind, item.subject, item.object) for item in self.relations
        )
        if len(set(relation_keys)) != len(relation_keys):
            raise PhysicalSearchTargetError("physical-search target has a duplicate relation")
        known_regions = set(region_ids)
        if any(
            item.subject not in known_regions or item.object not in known_regions
            for item in self.relations
        ):
            raise PhysicalSearchTargetError(
                "physical-search relation names an unknown region"
            )
        object.__setattr__(self, "baseline_bbox_area_um2", area)
        object.__setattr__(self, "baseline_whitespace_um2", whitespace)
        object.__setattr__(self, "baseline_wire_length_um", wire)
        object.__setattr__(self, "baseline_via_count", via)
        object.__setattr__(self, "max_bbox_area_ratio", area_ratio)
        object.__setattr__(self, "max_whitespace_fraction", whitespace_fraction)
        object.__setattr__(self, "min_aspect", min_aspect)
        object.__setattr__(self, "max_aspect", max_aspect)
        object.__setattr__(self, "max_wire_length_ratio", wire_ratio)
        object.__setattr__(self, "max_via_count_ratio", via_ratio)
        object.__setattr__(self, "regions", tuple(sorted(self.regions, key=lambda x: x.region_id)))
        object.__setattr__(
            self,
            "relations",
            tuple(
                sorted(
                    self.relations,
                    key=lambda x: (
                        _RELATION_ORDER[x.kind],
                        x.subject,
                        x.object,
                    ),
                )
            ),
        )

    @property
    def max_bbox_area_um2(self) -> float:
        return self.baseline_bbox_area_um2 * self.max_bbox_area_ratio

    @property
    def max_wire_length_um(self) -> float:
        return self.baseline_wire_length_um * self.max_wire_length_ratio

    @property
    def max_via_count(self) -> int:
        return _scaled_count(self.baseline_via_count, self.max_via_count_ratio)

    @property
    def aspect_range(self) -> tuple[float, float]:
        return (self.min_aspect, self.max_aspect)

    def canonical_data(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "circuit_id": self.circuit_id,
            "top_cell": self.top_cell,
            "pdk_id": self.pdk_id,
            "source_sha256": self.source_sha256,
            "reference_scope": self.reference_scope,
            "baseline": {
                "candidate_sha256": self.baseline_candidate_sha256,
                "gds_sha256": self.baseline_gds_sha256,
                "bbox_area_um2": self.baseline_bbox_area_um2,
                "whitespace_um2": self.baseline_whitespace_um2,
                "wire_length_um": self.baseline_wire_length_um,
                "via_count": self.baseline_via_count,
            },
            "acceptance": {
                "require_derived": self.require_derived,
                "max_bbox_area_ratio": self.max_bbox_area_ratio,
                "max_whitespace_fraction": self.max_whitespace_fraction,
                "min_aspect": self.min_aspect,
                "max_aspect": self.max_aspect,
                "max_wire_length_ratio": self.max_wire_length_ratio,
                "max_via_count_ratio": self.max_via_count_ratio,
            },
            "regions": [item.canonical_data() for item in self.regions],
            "relations": [item.canonical_data() for item in self.relations],
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.canonical_data(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )

    @property
    def canonical_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("ascii")).hexdigest()


@dataclass(frozen=True)
class PhysicalSearchAdmission:
    """Typed, non-signoff receipt for admission to one expensive exact replay."""

    target_sha256: str
    candidate_sha256: str
    materialization_mode: str
    placement_state_sha256: str
    require_derived: bool
    admitted: bool
    blocker_codes: tuple[str, ...]
    reference_relation_residual_count: int
    route_proxy_overflow_count: int
    normalized_bbox_gap: float
    whitespace_fraction: float
    aspect: float
    wire_length_um: float
    via_count: int
    signoff_authority: bool = False
    schema: str = PHYSICAL_SEARCH_ADMISSION_SCHEMA

    def __post_init__(self) -> None:
        if type(self.schema) is not str or self.schema != PHYSICAL_SEARCH_ADMISSION_SCHEMA:
            raise PhysicalSearchTargetError("physical-search admission schema is unsupported")
        _sha256(self.target_sha256, label="admission target SHA-256")
        _sha256(self.candidate_sha256, label="admission candidate SHA-256")
        if type(self.materialization_mode) is not str or self.materialization_mode not in {
            "authoring",
            "derived",
        }:
            raise PhysicalSearchTargetError("admission materialization mode is invalid")
        if type(self.require_derived) is not bool:
            raise PhysicalSearchTargetError(
                "admission require_derived policy must be an exact bool"
            )
        if type(self.placement_state_sha256) is not str or (
            self.placement_state_sha256
            and _SHA256.fullmatch(self.placement_state_sha256) is None
        ):
            raise PhysicalSearchTargetError(
                "admission placement state must be empty or a lowercase SHA-256"
            )
        if self.materialization_mode == "authoring" and self.placement_state_sha256:
            raise PhysicalSearchTargetError(
                "authoring admission cannot carry derived placement evidence"
            )
        if type(self.admitted) is not bool:
            raise PhysicalSearchTargetError("admission admitted flag must be exact bool")
        if type(self.blocker_codes) is not tuple or any(
            type(code) is not str or code not in _ADMISSION_BLOCKER_ORDER
            for code in self.blocker_codes
        ):
            raise PhysicalSearchTargetError("admission blockers must be an exact typed tuple")
        expected_codes = tuple(
            sorted(set(self.blocker_codes), key=_ADMISSION_BLOCKER_ORDER.__getitem__)
        )
        if self.blocker_codes != expected_codes:
            raise PhysicalSearchTargetError("admission blockers must be unique and canonical")
        if self.admitted != (not self.blocker_codes):
            raise PhysicalSearchTargetError("admission verdict disagrees with blockers")
        if (
            self.require_derived and self.materialization_mode == "authoring"
        ) != (
            "AUTHORING_CANDIDATE" in self.blocker_codes
        ):
            raise PhysicalSearchTargetError(
                "admission authoring blocker disagrees with materialization mode"
            )
        placement_state_required = self.require_derived or (
            self.materialization_mode == "derived"
        )
        if (placement_state_required and not self.placement_state_sha256) != (
            "PLACEMENT_STATE_MISSING" in self.blocker_codes
        ):
            raise PhysicalSearchTargetError(
                "admission placement blocker disagrees with placement evidence"
            )
        relation_residuals = _count(
            self.reference_relation_residual_count,
            label="admission reference relation residual count",
        )
        if (relation_residuals > 0) != (
            "REFERENCE_RELATION_MISMATCH" in self.blocker_codes
        ):
            raise PhysicalSearchTargetError(
                "admission relation blocker disagrees with residual count"
            )
        route_overflow = _count(
            self.route_proxy_overflow_count,
            label="admission route proxy overflow count",
        )
        expected_route_overflow = sum(
            code in self.blocker_codes
            for code in ("WIRE_LENGTH_LIMIT", "VIA_COUNT_LIMIT")
        )
        if route_overflow != expected_route_overflow:
            raise PhysicalSearchTargetError(
                "admission route blockers disagree with overflow count"
            )
        normalized_gap = _measure(
            self.normalized_bbox_gap,
            label="admission normalized bbox gap",
        )
        whitespace_fraction = _measure(
            self.whitespace_fraction,
            label="admission whitespace fraction",
        )
        if whitespace_fraction > 1.0:
            raise PhysicalSearchTargetError(
                "admission whitespace fraction cannot exceed one"
            )
        aspect = _measure(self.aspect, label="admission aspect", positive=True)
        wire = _measure(self.wire_length_um, label="admission wire length")
        via = _count(self.via_count, label="admission via count")
        if self.signoff_authority is not False:
            raise PhysicalSearchTargetError(
                "pre-exact admission cannot carry signoff authority"
            )
        object.__setattr__(self, "reference_relation_residual_count", relation_residuals)
        object.__setattr__(self, "route_proxy_overflow_count", route_overflow)
        object.__setattr__(self, "normalized_bbox_gap", normalized_gap)
        object.__setattr__(self, "whitespace_fraction", whitespace_fraction)
        object.__setattr__(self, "aspect", aspect)
        object.__setattr__(self, "wire_length_um", wire)
        object.__setattr__(self, "via_count", via)

    def rank_key(self) -> tuple[int | float | str, ...]:
        return (
            self.reference_relation_residual_count,
            self.normalized_bbox_gap,
            self.wire_length_um,
            self.via_count,
            self.candidate_sha256,
        )

    def canonical_data(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "target_sha256": self.target_sha256,
            "candidate_sha256": self.candidate_sha256,
            "materialization_mode": self.materialization_mode,
            "placement_state_sha256": self.placement_state_sha256,
            "require_derived": self.require_derived,
            "admitted": self.admitted,
            "blocker_codes": list(self.blocker_codes),
            "staging_rank": {
                "reference_relation_residual_count": (
                    self.reference_relation_residual_count
                ),
                "route_proxy_overflow_count": self.route_proxy_overflow_count,
                "normalized_bbox_gap": self.normalized_bbox_gap,
                "whitespace_fraction": self.whitespace_fraction,
                "aspect": self.aspect,
                "wire_length_um": self.wire_length_um,
                "via_count": self.via_count,
            },
            "authority": {
                "kind": "pre_exact_staging_only",
                "signoff_authority": False,
            },
        }

    @property
    def sha256(self) -> str:
        raw = json.dumps(
            self.canonical_data(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        return hashlib.sha256(raw.encode("ascii")).hexdigest()


def evaluate_physical_search_admission(
    target: PhysicalSearchTarget,
    candidate: CompactIntentCandidate,
    metrics: Mapping[str, object],
) -> PhysicalSearchAdmission:
    """Evaluate one generated placement against coordinate-free staging DATA."""

    if type(target) is not PhysicalSearchTarget:
        raise PhysicalSearchTargetError("admission target must be exact typed")
    if type(candidate) is not CompactIntentCandidate:
        raise PhysicalSearchTargetError("admission candidate must be exact typed")
    if type(metrics) is not dict or set(metrics) != _ADMISSION_METRIC_KEYS:
        raise PhysicalSearchTargetError("admission metrics must use the exact cheap schema")
    if _contains_png(metrics):
        raise PhysicalSearchTargetError("PNG cannot appear in admission metrics")

    raw_bbox = metrics.get("bbox_nm")
    if type(raw_bbox) is not list or len(raw_bbox) != 4 or any(
        type(value) is not int or abs(value) > int(_MAX_MEASURE)
        for value in (raw_bbox if type(raw_bbox) is list else ())
    ):
        raise PhysicalSearchTargetError(
            "admission bbox_nm must contain four bounded exact integers"
        )
    bbox = tuple(raw_bbox)
    width_nm = bbox[2] - bbox[0]
    height_nm = bbox[3] - bbox[1]
    if width_nm <= 0 or height_nm <= 0:
        raise PhysicalSearchTargetError("admission bbox_nm must have positive extent")
    if metrics.get("bbox_width_nm") != width_nm or type(
        metrics.get("bbox_width_nm")
    ) is not int:
        raise PhysicalSearchTargetError("admission bbox width disagrees with bbox_nm")
    if metrics.get("bbox_height_nm") != height_nm or type(
        metrics.get("bbox_height_nm")
    ) is not int:
        raise PhysicalSearchTargetError("admission bbox height disagrees with bbox_nm")
    area = _measure(
        metrics.get("bbox_area_um2"),
        label="admission bbox area",
        positive=True,
    )
    expected_area = width_nm * height_nm / 1_000_000.0
    if area != expected_area:
        raise PhysicalSearchTargetError("admission bbox area disagrees with bbox_nm")
    whitespace = _measure(
        metrics.get("whitespace_um2"),
        label="admission whitespace",
    )
    if whitespace > area:
        raise PhysicalSearchTargetError("admission whitespace exceeds bbox area")
    wire = _measure(metrics.get("wire_length_um"), label="admission wire length")
    via = _count(metrics.get("via_count"), label="admission via count")
    ring_perimeter = _measure(
        metrics.get("ring_perimeter_um"),
        label="admission ring perimeter",
    )
    ring_max_perimeter = _measure(
        metrics.get("ring_max_perimeter_um"),
        label="admission maximum ring perimeter",
    )
    _measure(
        metrics.get("ring_span_waste_um2"),
        label="admission ring span waste",
    )
    if ring_max_perimeter > ring_perimeter + 1e-9:
        raise PhysicalSearchTargetError(
            "admission maximum ring perimeter exceeds total perimeter"
        )
    _sha256(
        metrics.get("physical_signature_sha256"),
        label="admission physical signature SHA-256",
    )
    placement_state = metrics.get("placement_state_sha256")
    if type(placement_state) is not str or (
        placement_state and _SHA256.fullmatch(placement_state) is None
    ):
        raise PhysicalSearchTargetError(
            "admission placement state must be empty or a lowercase SHA-256"
        )
    macro_consumption = metrics.get("macro_consumption_sha256")
    if type(macro_consumption) is not str or (
        macro_consumption and _SHA256.fullmatch(macro_consumption) is None
    ):
        raise PhysicalSearchTargetError(
            "admission macro consumption must be empty or a lowercase SHA-256"
        )
    if bool(macro_consumption) != bool(placement_state):
        raise PhysicalSearchTargetError(
            "admission macro and placement evidence must be present together"
        )
    if candidate.materialization_mode == "authoring" and placement_state:
        raise PhysicalSearchTargetError(
            "authoring candidate cannot carry derived admission evidence"
        )

    raw_groups = metrics.get("functional_group_bboxes_nm")
    if type(raw_groups) is not list or len(raw_groups) > 4096:
        raise PhysicalSearchTargetError(
            "admission functional groups must be a bounded exact list"
        )
    group_boxes: dict[str, tuple[int, int, int, int]] = {}
    target_group_ids = {
        group_id for region in target.regions for group_id in region.functional_groups
    }
    for index, raw_group in enumerate(raw_groups):
        if type(raw_group) is not dict or set(raw_group) != {"group_id", "bbox_nm"}:
            raise PhysicalSearchTargetError(
                f"admission functional group {index} is malformed"
            )
        group_id = _identifier(
            raw_group["group_id"],
            label=f"admission functional group {index} ID",
        )
        if (
            _contains_png(group_id)
            or group_id in group_boxes
            or group_id not in target_group_ids
        ):
            raise PhysicalSearchTargetError(
                "admission functional groups are duplicate, foreign or PNG-owned"
            )
        raw_group_bbox = raw_group["bbox_nm"]
        if type(raw_group_bbox) is not list or len(raw_group_bbox) != 4 or any(
            type(value) is not int or abs(value) > int(_MAX_MEASURE)
            for value in (raw_group_bbox if type(raw_group_bbox) is list else ())
        ):
            raise PhysicalSearchTargetError(
                f"admission functional group {index} bbox is invalid"
            )
        group_bbox = tuple(raw_group_bbox)
        if (
            group_bbox[2] <= group_bbox[0]
            or group_bbox[3] <= group_bbox[1]
            or group_bbox[0] < bbox[0]
            or group_bbox[1] < bbox[1]
            or group_bbox[2] > bbox[2]
            or group_bbox[3] > bbox[3]
        ):
            raise PhysicalSearchTargetError(
                f"admission functional group {index} bbox is outside candidate"
            )
        group_boxes[group_id] = group_bbox
    if tuple(group_boxes) != tuple(sorted(group_boxes)):
        raise PhysicalSearchTargetError("admission functional groups are not canonical")

    blockers: set[str] = set()
    if target.require_derived and candidate.materialization_mode != "derived":
        blockers.add("AUTHORING_CANDIDATE")
    placement_state_required = target.require_derived or (
        candidate.materialization_mode == "derived"
    )
    if placement_state_required and not placement_state:
        blockers.add("PLACEMENT_STATE_MISSING")
    if area > target.max_bbox_area_um2:
        blockers.add("BBOX_AREA_LIMIT")
    whitespace_fraction = whitespace / area
    if whitespace_fraction > target.max_whitespace_fraction:
        blockers.add("WHITESPACE_LIMIT")
    aspect = width_nm / height_nm
    if not target.min_aspect <= aspect <= target.max_aspect:
        blockers.add("ASPECT_LIMIT")
    if wire > target.max_wire_length_um:
        blockers.add("WIRE_LENGTH_LIMIT")
    if via > target.max_via_count:
        blockers.add("VIA_COUNT_LIMIT")

    region_boxes: dict[str, tuple[int, int, int, int]] = {}
    for region in target.regions:
        boxes = [group_boxes.get(group_id) for group_id in region.functional_groups]
        if any(box is None for box in boxes):
            blockers.add("REGION_MISSING")
            continue
        present = tuple(box for box in boxes if box is not None)
        region_boxes[region.region_id] = (
            min(box[0] for box in present),
            min(box[1] for box in present),
            max(box[2] for box in present),
            max(box[3] for box in present),
        )

    relation_residuals = 0
    for relation in target.relations:
        subject = region_boxes.get(relation.subject)
        object_bbox = region_boxes.get(relation.object)
        if subject is None or object_bbox is None:
            continue
        satisfied = (
            subject[2] <= object_bbox[0]
            if relation.kind == "left_of"
            else subject[1] >= object_bbox[3]
        )
        if not satisfied:
            relation_residuals += 1
    if relation_residuals:
        blockers.add("REFERENCE_RELATION_MISMATCH")

    blocker_codes = tuple(sorted(blockers, key=_ADMISSION_BLOCKER_ORDER.__getitem__))
    route_overflow = sum(
        code in blockers for code in ("WIRE_LENGTH_LIMIT", "VIA_COUNT_LIMIT")
    )
    return PhysicalSearchAdmission(
        target_sha256=target.canonical_sha256,
        candidate_sha256=candidate.digest,
        materialization_mode=candidate.materialization_mode,
        placement_state_sha256=placement_state,
        require_derived=target.require_derived,
        admitted=not blocker_codes,
        blocker_codes=blocker_codes,
        reference_relation_residual_count=relation_residuals,
        route_proxy_overflow_count=route_overflow,
        normalized_bbox_gap=area / target.max_bbox_area_um2,
        whitespace_fraction=whitespace_fraction,
        aspect=aspect,
        wire_length_um=wire,
        via_count=via,
    )


def _pairs_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise PhysicalSearchTargetError(f"target JSON repeats key {key!r}")
        result[key] = value
    return result


def load_physical_search_target(
    path: Path | str,
    *,
    source_path: Path | str,
    expected_pdk: str,
) -> PhysicalSearchTarget:
    """Load and byte-bind one committed pre-exact target."""

    target_path = Path(path).resolve()
    source = Path(source_path).resolve()
    try:
        payload = json.loads(
            target_path.read_text(encoding="utf-8"),
            object_pairs_hook=_pairs_object,
        )
    except PhysicalSearchTargetError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PhysicalSearchTargetError(
            f"cannot load physical-search target: {type(exc).__name__}: {exc}"
        ) from exc
    root = _mapping(payload, label="physical-search target")
    if _contains_png(root):
        raise PhysicalSearchTargetError("PNG cannot appear in a physical-search target")
    _exact_keys(root, _ROOT_KEYS, label="physical-search target")
    baseline = _mapping(root["baseline"], label="physical-search baseline")
    acceptance = _mapping(root["acceptance"], label="physical-search acceptance")
    _exact_keys(baseline, _BASELINE_KEYS, label="physical-search baseline")
    _exact_keys(acceptance, _ACCEPTANCE_KEYS, label="physical-search acceptance")
    raw_regions = root["regions"]
    if type(raw_regions) is not list or not raw_regions:
        raise PhysicalSearchTargetError("physical-search regions must be a non-empty list")
    regions: list[PhysicalSearchRegion] = []
    for index, raw in enumerate(raw_regions):
        item = _mapping(raw, label=f"physical-search region {index}")
        _exact_keys(item, _REGION_KEYS, label=f"physical-search region {index}")
        groups = item["functional_groups"]
        if type(groups) is not list:
            raise PhysicalSearchTargetError(
                f"physical-search region {index} functional_groups must be a list"
            )
        regions.append(
            PhysicalSearchRegion(
                region_id=item["region_id"],
                functional_groups=tuple(groups),
            )
        )
    raw_relations = root["relations"]
    if type(raw_relations) is not list or not raw_relations:
        raise PhysicalSearchTargetError(
            "physical-search relations must be a non-empty list"
        )
    relations: list[PhysicalSearchRelation] = []
    for index, raw in enumerate(raw_relations):
        item = _mapping(raw, label=f"physical-search relation {index}")
        _exact_keys(item, _RELATION_KEYS, label=f"physical-search relation {index}")
        relations.append(
            PhysicalSearchRelation(
                kind=item["kind"],
                subject=item["subject"],
                object=item["object"],
            )
        )
    target = PhysicalSearchTarget(
        schema=root["schema"],
        circuit_id=root["circuit_id"],
        top_cell=root["top_cell"],
        pdk_id=root["pdk_id"],
        source_sha256=root["source_sha256"],
        reference_scope=root["reference_scope"],
        baseline_candidate_sha256=baseline["candidate_sha256"],
        baseline_gds_sha256=baseline["gds_sha256"],
        baseline_bbox_area_um2=baseline["bbox_area_um2"],
        baseline_whitespace_um2=baseline["whitespace_um2"],
        baseline_wire_length_um=baseline["wire_length_um"],
        baseline_via_count=baseline["via_count"],
        require_derived=acceptance["require_derived"],
        max_bbox_area_ratio=acceptance["max_bbox_area_ratio"],
        max_whitespace_fraction=acceptance["max_whitespace_fraction"],
        min_aspect=acceptance["min_aspect"],
        max_aspect=acceptance["max_aspect"],
        max_wire_length_ratio=acceptance["max_wire_length_ratio"],
        max_via_count_ratio=acceptance["max_via_count_ratio"],
        regions=tuple(regions),
        relations=tuple(relations),
    )
    try:
        measured_source_sha256 = _file_sha256(source)
    except OSError as exc:
        raise PhysicalSearchTargetError(
            f"cannot hash physical-search source: {type(exc).__name__}: {exc}"
        ) from exc
    if target.source_sha256 != measured_source_sha256:
        raise PhysicalSearchTargetError(
            "physical-search target source SHA-256 mismatch: "
            f"declared={target.source_sha256}, measured={measured_source_sha256}"
        )
    if type(expected_pdk) is not str or target.pdk_id != expected_pdk:
        raise PhysicalSearchTargetError(
            f"physical-search target differs from expected PDK {expected_pdk!r}"
        )
    return target


__all__ = [
    "PHYSICAL_SEARCH_ADMISSION_SCHEMA",
    "PHYSICAL_SEARCH_TARGET_SCHEMA",
    "PhysicalSearchAdmission",
    "PhysicalSearchRegion",
    "PhysicalSearchRelation",
    "PhysicalSearchTarget",
    "PhysicalSearchTargetError",
    "evaluate_physical_search_admission",
    "load_physical_search_target",
]
