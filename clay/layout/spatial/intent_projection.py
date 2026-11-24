"""Fail-closed projection from ASIR facts to coordinate-free physical intent."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import json
import re

from clay.layout.spatial.abstract_ir import (
    AbstractSpatialIR,
    HardInvariant,
    HierarchicalMacroDag,
    MacroDagLevel,
    MacroDagNode,
    SpatialFactKind,
)
from clay.layout.spatial.mosaic_dag_queries import compile_dag_queries


_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_NODE_RE = re.compile(r"(.+)\[(\d{3,})\]\Z")
_ORIENTATIONS = frozenset(
    {"R0", "R90", "R180", "R270", "MR0", "MR90", "MR180", "MR270"}
)


class SpatialIntentProjectionError(ValueError):
    """ASIR facts do not prove one unique coordinate-free intent projection."""


@dataclass(frozen=True, slots=True, order=True)
class AggregateAlignmentIntent:
    """One evidence-bound aggregate scope-edge alignment tolerance band."""

    alignment_id: str
    first_scope: str
    second_scope: str
    axis: str
    edge: str
    tolerance_ppm: int
    source_asir_sha256: str
    source_fact_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for value, label in (
            (self.alignment_id, "aggregate alignment id"),
            (self.first_scope, "aggregate alignment first scope"),
            (self.second_scope, "aggregate alignment second scope"),
        ):
            _text(value, label)
        if self.first_scope == self.second_scope:
            raise SpatialIntentProjectionError(
                "aggregate alignment requires two distinct scopes"
            )
        first_scope, second_scope = sorted((self.first_scope, self.second_scope))
        object.__setattr__(self, "first_scope", first_scope)
        object.__setattr__(self, "second_scope", second_scope)
        if self.axis not in {"horizontal", "vertical"}:
            raise SpatialIntentProjectionError(
                "aggregate alignment axis must be horizontal or vertical"
            )
        if self.edge not in {"leading", "trailing", "center"}:
            raise SpatialIntentProjectionError(
                "aggregate alignment edge must be leading, trailing, or center"
            )
        if (
            type(self.tolerance_ppm) is not int
            or not 0 <= self.tolerance_ppm <= 1_000_000
        ):
            raise SpatialIntentProjectionError(
                "aggregate alignment tolerance must be exact ppm in [0, 1000000]"
            )
        if (
            type(self.source_asir_sha256) is not str
            or _SHA256_RE.fullmatch(self.source_asir_sha256) is None
        ):
            raise SpatialIntentProjectionError(
                "aggregate alignment source ASIR identity is malformed"
            )
        object.__setattr__(
            self,
            "source_fact_ids",
            _sorted_texts(
                self.source_fact_ids,
                "aggregate alignment source fact ids",
            ),
        )
        object.__setattr__(
            self,
            "evidence_ids",
            _sorted_texts(
                self.evidence_ids,
                "aggregate alignment evidence ids",
            ),
        )

    def canonical_data(self) -> dict[str, object]:
        return {
            "schema": "clay.aggregate-alignment-intent/v1",
            "alignment_id": self.alignment_id,
            "first_scope": self.first_scope,
            "second_scope": self.second_scope,
            "axis": self.axis,
            "edge": self.edge,
            "tolerance_ppm": self.tolerance_ppm,
            "source_asir_sha256": self.source_asir_sha256,
            "source_fact_ids": list(self.source_fact_ids),
            "evidence_ids": list(self.evidence_ids),
        }

    @property
    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.canonical_data(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()


@dataclass(frozen=True, slots=True, order=True)
class AggregateOrderIntent:
    """One strict aggregate scope-envelope order projected without geometry."""

    order_id: str
    before_scope: str
    after_scope: str
    axis: str
    source_asir_sha256: str
    source_fact_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for value, label in (
            (self.order_id, "aggregate order id"),
            (self.before_scope, "aggregate order before scope"),
            (self.after_scope, "aggregate order after scope"),
        ):
            _text(value, label)
        if self.before_scope == self.after_scope:
            raise SpatialIntentProjectionError(
                "aggregate order requires two distinct scopes"
            )
        if self.axis not in {"horizontal", "vertical"}:
            raise SpatialIntentProjectionError(
                "aggregate order axis must be horizontal or vertical"
            )
        if (
            type(self.source_asir_sha256) is not str
            or _SHA256_RE.fullmatch(self.source_asir_sha256) is None
        ):
            raise SpatialIntentProjectionError(
                "aggregate order source ASIR identity is malformed"
            )
        object.__setattr__(
            self,
            "source_fact_ids",
            _sorted_texts(self.source_fact_ids, "aggregate order source fact ids"),
        )
        object.__setattr__(
            self,
            "evidence_ids",
            _sorted_texts(self.evidence_ids, "aggregate order evidence ids"),
        )

    def canonical_data(self) -> dict[str, object]:
        return {
            "schema": "clay.aggregate-order-intent/v1",
            "order_id": self.order_id,
            "before_scope": self.before_scope,
            "after_scope": self.after_scope,
            "axis": self.axis,
            "source_asir_sha256": self.source_asir_sha256,
            "source_fact_ids": list(self.source_fact_ids),
            "evidence_ids": list(self.evidence_ids),
        }

    @property
    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.canonical_data(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()


_REFERENCE_WITNESS_STRENGTH_ORDER = (
    "member_footprint_nonoverlap",
    "center_order",
    "bounded_overlap_envelope_order",
    "no_relation",
)


@dataclass(frozen=True, slots=True, order=True)
class ReferenceWitnessCandidateResult:
    """One strength candidate evaluated only on an observable reference surface."""

    strength: str
    status: str
    observable_surface: str | None
    metric: str
    observed_nm: int | None
    limit_nm: int | None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.strength not in _REFERENCE_WITNESS_STRENGTH_ORDER:
            raise SpatialIntentProjectionError(
                "reference witness candidate strength differs"
            )
        if self.status not in {"PASS", "FAIL", "UNMEASURABLE"}:
            raise SpatialIntentProjectionError(
                "reference witness candidate status differs"
            )
        _text(self.metric, "reference witness candidate metric")
        if self.status == "UNMEASURABLE":
            if (
                self.observable_surface is not None
                or self.observed_nm is not None
                or self.limit_nm is not None
                or type(self.reason) is not str
                or not self.reason
            ):
                raise SpatialIntentProjectionError(
                    "unmeasurable reference witness candidate is not typed"
                )
            return
        if self.observable_surface != "reference_oa_spatial_placement":
            raise SpatialIntentProjectionError(
                "reference witness candidate escapes observable OA placement"
            )
        if type(self.observed_nm) is not int or self.observed_nm < 0:
            raise SpatialIntentProjectionError(
                "reference witness candidate observation must be exact nm"
            )
        if self.limit_nm is not None and (
            type(self.limit_nm) is not int or self.limit_nm < 0
        ):
            raise SpatialIntentProjectionError(
                "reference witness candidate limit must be exact nm"
            )
        if self.reason is not None:
            raise SpatialIntentProjectionError(
                "measurable reference witness candidate carries unmeasurable reason"
            )

    def canonical_data(self) -> dict[str, object]:
        return {
            "strength": self.strength,
            "status": self.status,
            "observable_surface": self.observable_surface,
            "metric": self.metric,
            "observed_nm": self.observed_nm,
            "limit_nm": self.limit_nm,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True, order=True)
class ReferenceWitnessValidationReceipt:
    """Ordered witness decision that cannot use generated device geometry."""

    relation_id: str
    before_scope: str
    after_scope: str
    axis: str
    status: str
    selected_strength: str
    observable_surface: str
    reference_receipt_sha256: str
    reference_artifact_sha256: str
    candidates: tuple[ReferenceWitnessCandidateResult, ...]
    device_geometry_status: str = "UNMEASURABLE"
    device_geometry_reason: str = "reference PCells empty"

    def __post_init__(self) -> None:
        for value, label in (
            (self.relation_id, "reference witness relation id"),
            (self.before_scope, "reference witness before scope"),
            (self.after_scope, "reference witness after scope"),
        ):
            _text(value, label)
        if self.before_scope == self.after_scope:
            raise SpatialIntentProjectionError(
                "reference witness validation requires distinct scopes"
            )
        if self.axis not in {"horizontal", "vertical"}:
            raise SpatialIntentProjectionError(
                "reference witness validation axis differs"
            )
        if self.status not in {"PASS", "NO_RELATION"}:
            raise SpatialIntentProjectionError(
                "reference witness validation status differs"
            )
        if self.observable_surface != "reference_oa_spatial_placement":
            raise SpatialIntentProjectionError(
                "reference witness validation escapes observable OA placement"
            )
        for value, label in (
            (self.reference_receipt_sha256, "reference witness source receipt"),
            (self.reference_artifact_sha256, "reference witness artifact"),
        ):
            if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
                raise SpatialIntentProjectionError(f"{label} SHA-256 is malformed")
        if (
            type(self.candidates) is not tuple
            or tuple(candidate.strength for candidate in self.candidates)
            != _REFERENCE_WITNESS_STRENGTH_ORDER
            or any(
                type(candidate) is not ReferenceWitnessCandidateResult
                for candidate in self.candidates
            )
        ):
            raise SpatialIntentProjectionError(
                "reference witness candidates do not follow strength order"
            )
        passing = tuple(
            candidate.strength
            for candidate in self.candidates
            if candidate.status == "PASS"
        )
        expected = passing[0] if passing else "no_relation"
        if self.selected_strength != expected:
            raise SpatialIntentProjectionError(
                "reference witness selected strength is not the strongest passing candidate"
            )
        if (self.status == "NO_RELATION") != (
            self.selected_strength == "no_relation"
        ):
            raise SpatialIntentProjectionError(
                "reference witness status differs from selected strength"
            )
        if (
            self.device_geometry_status != "UNMEASURABLE"
            or self.device_geometry_reason != "reference PCells empty"
        ):
            raise SpatialIntentProjectionError(
                "reference witness device geometry surface must remain unmeasurable"
            )

    def canonical_data(self) -> dict[str, object]:
        return {
            "schema": "clay.reference-witness-validation/v1",
            "relation_id": self.relation_id,
            "before_scope": self.before_scope,
            "after_scope": self.after_scope,
            "axis": self.axis,
            "status": self.status,
            "selected_strength": self.selected_strength,
            "observable_surface": self.observable_surface,
            "reference_receipt_sha256": self.reference_receipt_sha256,
            "reference_artifact_sha256": self.reference_artifact_sha256,
            "candidates": [candidate.canonical_data() for candidate in self.candidates],
            "device_geometry": {
                "status": self.device_geometry_status,
                "reason": self.device_geometry_reason,
            },
        }

    @property
    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.canonical_data(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()


@dataclass(frozen=True, slots=True, order=True)
class ReferenceConditionedPlacementObligation:
    """One coordinate-free placement strength bound to reference evidence."""

    obligation_id: str
    before_scope: str
    after_scope: str
    axis: str
    strength: str
    source_relation_id: str
    source_relation_sha256: str
    reference_receipt_sha256: str
    reference_artifact_sha256: str
    evidence_ids: tuple[str, ...]
    witness_validation_status: str = "UNVALIDATED"
    witness_validation_receipt_sha256: str | None = None
    witness_validation_observable_surface: str | None = None
    witness_validated_strength: str | None = None
    relation_kind: str = "ordered_pair"
    host_scope: str | None = None
    guest_scopes: tuple[str, ...] = ()
    lower_boundary: str | None = None
    upper_boundary: str | None = None

    def __post_init__(self) -> None:
        for value, label in (
            (self.obligation_id, "reference-conditioned obligation id"),
            (self.before_scope, "reference-conditioned before scope"),
            (self.after_scope, "reference-conditioned after scope"),
            (self.source_relation_id, "reference-conditioned source relation id"),
        ):
            _text(value, label)
        if self.before_scope == self.after_scope:
            raise SpatialIntentProjectionError(
                "reference-conditioned placement requires two distinct scopes"
            )
        if self.relation_kind not in {"ordered_pair", "row_bounded_cavity"}:
            raise SpatialIntentProjectionError(
                "reference-conditioned placement relation kind differs"
            )
        if self.axis not in {"horizontal", "vertical"}:
            raise SpatialIntentProjectionError(
                "reference-conditioned placement axis must be horizontal or vertical"
            )
        if self.relation_kind == "ordered_pair":
            if self.strength not in {
                "envelope_separation",
                "member_footprint_nonoverlap",
                "center_order",
                "bounded_overlap_envelope_order",
            }:
                raise SpatialIntentProjectionError(
                    "reference-conditioned placement strength differs"
                )
            if any(
                value is not None
                for value in (
                    self.host_scope,
                    self.lower_boundary,
                    self.upper_boundary,
                )
            ) or self.guest_scopes:
                raise SpatialIntentProjectionError(
                    "ordered-pair placement carries cavity fields"
                )
        else:
            if self.strength != "row_bounded_cavity" or self.axis != "vertical":
                raise SpatialIntentProjectionError(
                    "row-bounded cavity placement strength differs"
                )
            _text(self.host_scope, "row-bounded cavity host scope")
            guests = _sorted_texts(
                self.guest_scopes,
                "row-bounded cavity guest scopes",
            )
            if not guests:
                raise SpatialIntentProjectionError(
                    "row-bounded cavity requires guest scopes"
                )
            if self.host_scope in guests:
                raise SpatialIntentProjectionError(
                    "row-bounded cavity host cannot also be a guest"
                )
            if self.before_scope != self.host_scope:
                raise SpatialIntentProjectionError(
                    "row-bounded cavity before scope differs from host"
                )
            if self.after_scope != f"guest_set:{','.join(guests)}":
                raise SpatialIntentProjectionError(
                    "row-bounded cavity guest-set scope differs"
                )
            if (
                self.lower_boundary != "maximal_span_host_atom_row_top_envelope"
                or self.upper_boundary != "host_envelope_top"
            ):
                raise SpatialIntentProjectionError(
                    "row-bounded cavity intrinsic boundary selectors differ"
                )
            object.__setattr__(self, "guest_scopes", guests)
        for value, label in (
            (self.source_relation_sha256, "source relation"),
            (self.reference_receipt_sha256, "reference receipt"),
            (self.reference_artifact_sha256, "reference artifact"),
        ):
            if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
                raise SpatialIntentProjectionError(
                    f"reference-conditioned {label} SHA-256 is malformed"
                )
        evidence = _sorted_texts(
            self.evidence_ids,
            "reference-conditioned placement evidence ids",
        )
        if not evidence:
            raise SpatialIntentProjectionError(
                "reference-conditioned placement requires evidence ids"
            )
        object.__setattr__(self, "evidence_ids", evidence)
        if self.witness_validation_status not in {
            "UNVALIDATED",
            "PASS",
            "FAIL",
            "UNMEASURABLE",
        }:
            raise SpatialIntentProjectionError(
                "reference-conditioned witness validation status differs"
            )
        witness_fields = (
            self.witness_validation_receipt_sha256,
            self.witness_validation_observable_surface,
            self.witness_validated_strength,
        )
        if self.witness_validation_status == "UNVALIDATED":
            if any(value is not None for value in witness_fields):
                raise SpatialIntentProjectionError(
                    "unvalidated reference-conditioned placement carries witness proof"
                )
        else:
            if (
                type(self.witness_validation_receipt_sha256) is not str
                or _SHA256_RE.fullmatch(
                    self.witness_validation_receipt_sha256
                ) is None
            ):
                raise SpatialIntentProjectionError(
                    "reference-conditioned witness validation receipt SHA-256 is malformed"
                )
            if (
                self.witness_validation_observable_surface
                != "reference_oa_spatial_placement"
            ):
                raise SpatialIntentProjectionError(
                    "reference-conditioned witness validation escapes observable OA placement"
                )
            if self.witness_validated_strength != self.strength:
                raise SpatialIntentProjectionError(
                    "reference-conditioned witness validation strength differs"
                )

    @property
    def provenance_status(self) -> str:
        return (
            "reference_proven"
            if self.witness_validation_status == "PASS"
            else "reference_candidate"
        )

    def canonical_data(self) -> dict[str, object]:
        payload = {
            "schema": "clay.reference-conditioned-placement-obligation/v2",
            "obligation_id": self.obligation_id,
            "before_scope": self.before_scope,
            "after_scope": self.after_scope,
            "axis": self.axis,
            "strength": self.strength,
            "source_relation_id": self.source_relation_id,
            "source_relation_sha256": self.source_relation_sha256,
            "reference_receipt_sha256": self.reference_receipt_sha256,
            "reference_artifact_sha256": self.reference_artifact_sha256,
            "evidence_ids": list(self.evidence_ids),
            "provenance_status": self.provenance_status,
            "witness_validation": {
                "status": self.witness_validation_status,
                "receipt_sha256": self.witness_validation_receipt_sha256,
                "observable_surface": self.witness_validation_observable_surface,
                "validated_strength": self.witness_validated_strength,
            },
        }
        if self.relation_kind == "row_bounded_cavity":
            payload["schema"] = (
                "clay.reference-conditioned-placement-obligation/v3"
            )
            payload["relation_kind"] = self.relation_kind
            payload["cavity"] = {
                "host_scope": self.host_scope,
                "guest_scopes": list(self.guest_scopes),
                "lower_boundary": self.lower_boundary,
                "upper_boundary": self.upper_boundary,
            }
        return payload

    @property
    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.canonical_data(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()


@dataclass(frozen=True, slots=True)
class HierarchicalMacroDagIntent:
    """Logical-owner macro topology projected from one occurrence-level ASIR."""

    hierarchy: HierarchicalMacroDag
    source_asir_sha256: str
    source_hierarchy_sha256: str
    source_relation_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.hierarchy) is not HierarchicalMacroDag:
            raise TypeError("macro DAG intent hierarchy must be exact")
        for value, label in (
            (self.source_asir_sha256, "macro DAG source ASIR identity"),
            (self.source_hierarchy_sha256, "macro DAG source hierarchy identity"),
        ):
            if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
                raise SpatialIntentProjectionError(f"{label} is malformed")
        if self.source_hierarchy_sha256 == self.hierarchy.sha256:
            raise SpatialIntentProjectionError(
                "macro DAG intent must distinguish occurrence and logical hierarchies"
            )
        object.__setattr__(
            self,
            "source_relation_ids",
            _sorted_texts(
                self.source_relation_ids,
                "macro DAG source relation ids",
            ),
        )
        object.__setattr__(
            self,
            "evidence_ids",
            _sorted_texts(self.evidence_ids, "macro DAG evidence ids"),
        )

    def canonical_data(self) -> dict[str, object]:
        return {
            "schema": "clay.hierarchical-macro-dag-intent/v1",
            "hierarchy": self.hierarchy.canonical_data(),
            "source_asir_sha256": self.source_asir_sha256,
            "source_hierarchy_sha256": self.source_hierarchy_sha256,
            "source_relation_ids": list(self.source_relation_ids),
            "evidence_ids": list(self.evidence_ids),
        }

    @property
    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.canonical_data(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise SpatialIntentProjectionError(f"{label} must be a canonical string")
    return value


def _sorted_texts(value: object, label: str, *, nonempty: bool = True) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{label} must be an exact tuple")
    rows = tuple(sorted(_text(item, label) for item in value))
    if len(set(rows)) != len(rows) or (nonempty and not rows):
        raise SpatialIntentProjectionError(f"{label} must be unique and non-empty")
    return rows


def derive_reference_conditioned_strength_from_oa_witness(
    *,
    relation_id: str,
    before_scope: str,
    after_scope: str,
    axis: str,
    before_center_envelope_nm: tuple[int, int, int, int],
    after_center_envelope_nm: tuple[int, int, int, int],
    reference_receipt_sha256: str,
    reference_artifact_sha256: str,
    member_footprint_observable: bool,
    member_footprint_unmeasurable_reason: str | None,
) -> ReferenceWitnessValidationReceipt:
    """Select the strongest placement relation supported by observable OA positions."""

    for value, label in (
        (relation_id, "reference witness relation id"),
        (before_scope, "reference witness before scope"),
        (after_scope, "reference witness after scope"),
    ):
        _text(value, label)
    if before_scope == after_scope:
        raise SpatialIntentProjectionError(
            "reference witness validation requires distinct scopes"
        )
    if axis not in {"horizontal", "vertical"}:
        raise SpatialIntentProjectionError(
            "reference witness validation axis differs"
        )
    for value, label in (
        (reference_receipt_sha256, "reference witness source receipt"),
        (reference_artifact_sha256, "reference witness artifact"),
    ):
        if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
            raise SpatialIntentProjectionError(f"{label} SHA-256 is malformed")
    if type(member_footprint_observable) is not bool:
        raise TypeError("member footprint observability must be an exact bool")
    if member_footprint_observable:
        raise SpatialIntentProjectionError(
            "observable member footprints require member-level witness geometry"
        )
    if (
        type(member_footprint_unmeasurable_reason) is not str
        or not member_footprint_unmeasurable_reason
    ):
        raise SpatialIntentProjectionError(
            "unobservable member footprints require a typed reason"
        )

    def _bbox(value: object, label: str) -> tuple[int, int, int, int]:
        if (
            type(value) is not tuple
            or len(value) != 4
            or any(type(item) is not int for item in value)
        ):
            raise TypeError(f"{label} must be an exact four-integer tuple")
        x0, y0, x1, y1 = value
        if x0 > x1 or y0 > y1:
            raise SpatialIntentProjectionError(f"{label} is inverted")
        return x0, y0, x1, y1

    before = _bbox(before_center_envelope_nm, "before OA center envelope")
    after = _bbox(after_center_envelope_nm, "after OA center envelope")
    lower_index, upper_index = (0, 2) if axis == "horizontal" else (1, 3)
    before_center_twice = before[lower_index] + before[upper_index]
    after_center_twice = after[lower_index] + after[upper_index]
    center_delta_twice = after_center_twice - before_center_twice
    if center_delta_twice % 2:
        raise SpatialIntentProjectionError(
            "reference OA center-order margin is not integral nanometers"
        )
    center_delta_nm = center_delta_twice // 2
    overlap_nm = max(0, before[upper_index] - after[lower_index])
    bounded_overlap_passes = before[lower_index] <= after[lower_index]

    candidates = (
        ReferenceWitnessCandidateResult(
            strength="member_footprint_nonoverlap",
            status="UNMEASURABLE",
            observable_surface=None,
            metric="member_footprint_overlap_nm",
            observed_nm=None,
            limit_nm=None,
            reason=member_footprint_unmeasurable_reason,
        ),
        ReferenceWitnessCandidateResult(
            strength="center_order",
            status="PASS" if center_delta_nm > 0 else "FAIL",
            observable_surface="reference_oa_spatial_placement",
            metric="aggregate_center_order_margin_nm",
            observed_nm=abs(center_delta_nm),
            limit_nm=0,
        ),
        ReferenceWitnessCandidateResult(
            strength="bounded_overlap_envelope_order",
            status="PASS" if bounded_overlap_passes else "FAIL",
            observable_surface="reference_oa_spatial_placement",
            metric="observable_envelope_overlap_nm",
            observed_nm=overlap_nm,
            limit_nm=overlap_nm,
        ),
        ReferenceWitnessCandidateResult(
            strength="no_relation",
            status="PASS",
            observable_surface="reference_oa_spatial_placement",
            metric="declared_relation_count",
            observed_nm=0,
            limit_nm=0,
        ),
    )
    selected_strength = next(
        candidate.strength for candidate in candidates if candidate.status == "PASS"
    )
    return ReferenceWitnessValidationReceipt(
        relation_id=relation_id,
        before_scope=before_scope,
        after_scope=after_scope,
        axis=axis,
        status="NO_RELATION" if selected_strength == "no_relation" else "PASS",
        selected_strength=selected_strength,
        observable_surface="reference_oa_spatial_placement",
        reference_receipt_sha256=reference_receipt_sha256,
        reference_artifact_sha256=reference_artifact_sha256,
        candidates=candidates,
    )


@dataclass(frozen=True, slots=True, order=True)
class MosaicOccurrence:
    """One logical unit in a row-major mosaic; deliberately no geometry fields."""

    logical_owner: str
    occurrence_ordinal: int
    transform: str

    def __post_init__(self) -> None:
        _text(self.logical_owner, "mosaic logical owner")
        if type(self.occurrence_ordinal) is not int or self.occurrence_ordinal < 0:
            raise SpatialIntentProjectionError(
                "mosaic occurrence ordinal must be a non-negative exact integer"
            )
        if self.transform not in _ORIENTATIONS:
            raise SpatialIntentProjectionError("mosaic transform is unsupported")

    @property
    def node_id(self) -> str:
        return f"{self.logical_owner}[{self.occurrence_ordinal:03d}]"

    def canonical_data(self) -> dict[str, object]:
        return {
            "logical_owner": self.logical_owner,
            "occurrence_ordinal": self.occurrence_ordinal,
            "transform": self.transform,
        }


@dataclass(frozen=True, slots=True)
class DiscreteMosaicIntent:
    """Evidence-bound exact occurrence cover and row order without coordinates."""

    mosaic_id: str
    rows: tuple[tuple[MosaicOccurrence, ...], ...]
    source_asir_sha256: str
    source_fact_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    rank_intervals: tuple[tuple[str, int, int, int], ...] = ()
    column_ranks: tuple[tuple[str, int, int], ...] = ()

    def __post_init__(self) -> None:
        _text(self.mosaic_id, "mosaic id")
        if type(self.rows) is not tuple or not self.rows:
            raise SpatialIntentProjectionError("mosaic rows must be a non-empty exact tuple")
        rows: list[tuple[MosaicOccurrence, ...]] = []
        for row in self.rows:
            if (
                type(row) is not tuple
                or not row
                or any(type(item) is not MosaicOccurrence for item in row)
            ):
                raise TypeError(
                    "mosaic rows must contain non-empty exact MosaicOccurrence tuples"
                )
            rows.append(row)
        flat = tuple(item for row in rows for item in row)
        if len({item.node_id for item in flat}) != len(flat):
            raise SpatialIntentProjectionError("mosaic occurrence cover contains duplicates")
        for owner in sorted({item.logical_owner for item in flat}):
            ordinals = tuple(sorted(
                item.occurrence_ordinal for item in flat if item.logical_owner == owner
            ))
            if ordinals != tuple(range(len(ordinals))):
                raise SpatialIntentProjectionError(
                    f"mosaic owner {owner!r} lacks a contiguous exact occurrence cover"
                )
        if type(self.rank_intervals) is not tuple:
            raise TypeError("mosaic rank intervals must be an exact tuple")
        if type(self.column_ranks) is not tuple:
            raise TypeError("mosaic column ranks must be an exact tuple")
        if self.column_ranks:
            if self.rank_intervals:
                raise SpatialIntentProjectionError("mosaic column ranks cannot mix with rank intervals")
            ranks = {}
            for value in self.column_ranks:
                if type(value) is not tuple or len(value) != 3:
                    raise TypeError("mosaic column rank must be an exact triple")
                owner, ordinal, rank = value
                _text(owner, "mosaic column rank owner")
                if type(ordinal) is not int or ordinal < 0 or type(rank) is not int or rank < 0:
                    raise SpatialIntentProjectionError("mosaic column rank must use nonnegative exact integers")
                node = f"{owner}[{ordinal:03d}]"
                if node in ranks:
                    raise SpatialIntentProjectionError("mosaic column rank occurrence is duplicated")
                ranks[node] = rank
            if set(ranks) != {item.node_id for item in flat}:
                raise SpatialIntentProjectionError("mosaic column rank occurrence cover is incomplete")
            if set(ranks.values()) != set(range(len(set(ranks.values())))):
                raise SpatialIntentProjectionError("mosaic column ranks are not contiguous")
            if any(any(ranks[a.node_id] >= ranks[b.node_id] for a,b in zip(row,row[1:])) for row in rows):
                raise SpatialIntentProjectionError("mosaic row column ranks must strictly increase")
            object.__setattr__(self, "column_ranks", tuple(sorted(self.column_ranks)))
        maximum_columns = max(len(row) for row in rows)
        short_nodes = {
            item.node_id
            for row in rows
            if len(row) < maximum_columns
            for item in row
        }
        intervals = []
        for raw in self.rank_intervals:
            if type(raw) is not tuple or len(raw) != 4:
                raise TypeError("mosaic rank interval must be one exact quadruple")
            owner, ordinal, left_rank, right_rank = raw
            _text(owner, "mosaic rank interval owner")
            if (
                type(ordinal) is not int
                or ordinal < 0
                or type(left_rank) is not int
                or type(right_rank) is not int
                or left_rank < 0
                or right_rank != left_rank + 1
                or right_rank >= maximum_columns
            ):
                raise SpatialIntentProjectionError("mosaic rank interval is malformed")
            intervals.append((owner, ordinal, left_rank, right_rank))
        intervals_tuple = tuple(sorted(intervals))
        if (
            len(set(intervals_tuple)) != len(intervals_tuple)
            or (not self.column_ranks and {
                f"{owner}[{ordinal:03d}]"
                for owner, ordinal, _left, _right in intervals_tuple
            }
            != short_nodes)
        ):
            raise SpatialIntentProjectionError(
                "mosaic ragged-row rank interval cover is incomplete"
            )
        if (
            type(self.source_asir_sha256) is not str
            or _SHA256_RE.fullmatch(self.source_asir_sha256) is None
        ):
            raise SpatialIntentProjectionError("mosaic source ASIR identity is malformed")
        object.__setattr__(self, "rows", tuple(rows))
        object.__setattr__(self, "rank_intervals", intervals_tuple)
        object.__setattr__(
            self,
            "source_fact_ids",
            _sorted_texts(self.source_fact_ids, "mosaic source fact ids"),
        )
        object.__setattr__(
            self,
            "evidence_ids",
            _sorted_texts(self.evidence_ids, "mosaic evidence ids"),
        )

    @property
    def members(self) -> tuple[str, ...]:
        return tuple(sorted({item.logical_owner for row in self.rows for item in row}))

    @property
    def occurrences(self) -> tuple[MosaicOccurrence, ...]:
        return tuple(item for row in self.rows for item in row)

    def canonical_data(self) -> dict[str, object]:
        payload = {
            "schema": "clay.discrete-mosaic-intent/v1",
            "mosaic_id": self.mosaic_id,
            "rows": [
                [item.canonical_data() for item in row]
                for row in self.rows
            ],
            "source_asir_sha256": self.source_asir_sha256,
            "source_fact_ids": list(self.source_fact_ids),
            "evidence_ids": list(self.evidence_ids),
        }
        if self.rank_intervals:
            payload["rank_intervals"] = [list(row) for row in self.rank_intervals]
        if self.column_ranks:
            payload["schema"] = "clay.discrete-mosaic-intent/v2"
            payload["column_ranks"] = [list(row) for row in self.column_ranks]
        return payload

    @property
    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.canonical_data(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()


@dataclass(frozen=True, slots=True, order=True)
class MosaicCenterAlignmentIntent:
    """Coordinate-free proof that a mosaic occupies an anchor lattice's centre."""

    alignment_id: str
    mosaic_id: str
    anchor_owner: str
    axis: str
    anchor_rank_count: int
    overlap_ranks: tuple[int, ...]
    source_mosaic_sha256: str
    source_asir_sha256: str
    source_fact_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _text(self.alignment_id, "mosaic center alignment id")
        _text(self.mosaic_id, "mosaic center alignment mosaic id")
        _text(self.anchor_owner, "mosaic center alignment anchor owner")
        if self.axis not in {"horizontal", "vertical"}:
            raise SpatialIntentProjectionError(
                "mosaic center alignment axis must be horizontal or vertical"
            )
        if type(self.anchor_rank_count) is not int or self.anchor_rank_count < 3:
            raise SpatialIntentProjectionError(
                "mosaic center alignment requires at least three anchor ranks"
            )
        if (
            type(self.overlap_ranks) is not tuple
            or not self.overlap_ranks
            or any(type(rank) is not int for rank in self.overlap_ranks)
            or self.overlap_ranks != tuple(sorted(set(self.overlap_ranks)))
            or self.overlap_ranks
            != tuple(range(self.overlap_ranks[0], self.overlap_ranks[-1] + 1))
            or self.overlap_ranks[0] < 0
            or self.overlap_ranks[-1] >= self.anchor_rank_count
        ):
            raise SpatialIntentProjectionError(
                "mosaic center alignment overlap ranks must be one bounded interval"
            )
        lower = self.overlap_ranks[0]
        upper = self.anchor_rank_count - self.overlap_ranks[-1] - 1
        if lower != upper:
            raise SpatialIntentProjectionError(
                "mosaic center alignment overlap interval is not central"
            )
        for value, label in (
            (self.source_mosaic_sha256, "source mosaic"),
            (self.source_asir_sha256, "source ASIR"),
        ):
            if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
                raise SpatialIntentProjectionError(
                    f"mosaic center alignment {label} identity is malformed"
                )
        object.__setattr__(
            self,
            "source_fact_ids",
            _sorted_texts(
                self.source_fact_ids,
                "mosaic center alignment source fact ids",
            ),
        )
        object.__setattr__(
            self,
            "evidence_ids",
            _sorted_texts(
                self.evidence_ids,
                "mosaic center alignment evidence ids",
            ),
        )

    def canonical_data(self) -> dict[str, object]:
        return {
            "schema": "clay.mosaic-center-alignment-intent/v1",
            "alignment_id": self.alignment_id,
            "mosaic_id": self.mosaic_id,
            "anchor_owner": self.anchor_owner,
            "axis": self.axis,
            "relation": "central_overlap",
            "anchor_rank_count": self.anchor_rank_count,
            "overlap_ranks": list(self.overlap_ranks),
            "source_mosaic_sha256": self.source_mosaic_sha256,
            "source_asir_sha256": self.source_asir_sha256,
            "source_fact_ids": list(self.source_fact_ids),
            "evidence_ids": list(self.evidence_ids),
        }

    @property
    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.canonical_data(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()


def replay_discrete_mosaic_intent(value: object) -> DiscreteMosaicIntent:
    if type(value) is not DiscreteMosaicIntent:
        raise TypeError("discrete mosaic declaration must be exact")
    return DiscreteMosaicIntent(
        mosaic_id=value.mosaic_id,
        rows=tuple(
            tuple(
                MosaicOccurrence(
                    logical_owner=item.logical_owner,
                    occurrence_ordinal=item.occurrence_ordinal,
                    transform=item.transform,
                )
                for item in row
            )
            for row in value.rows
        ),
        source_asir_sha256=value.source_asir_sha256,
        source_fact_ids=value.source_fact_ids,
        evidence_ids=value.evidence_ids,
        rank_intervals=value.rank_intervals,
        column_ranks=value.column_ranks,
    )


def replay_discrete_mosaic_inventory(
    value: object,
) -> tuple[DiscreteMosaicIntent, ...]:
    if type(value) is not tuple:
        raise TypeError("discrete mosaic inventory must be an exact tuple")
    rows = tuple(replay_discrete_mosaic_intent(item) for item in value)
    if len({item.mosaic_id for item in rows}) != len(rows):
        raise SpatialIntentProjectionError("discrete mosaic ids must be unique")
    claimed: set[str] = set()
    for row in rows:
        overlap = claimed & set(row.members)
        if overlap:
            raise SpatialIntentProjectionError(
                f"discrete mosaic member {min(overlap)!r} is claimed more than once"
            )
        claimed.update(row.members)
    return tuple(sorted(rows, key=lambda item: item.mosaic_id))


def replay_mosaic_center_alignment_intent(
    value: object,
) -> MosaicCenterAlignmentIntent:
    if type(value) is not MosaicCenterAlignmentIntent:
        raise TypeError("mosaic center alignment declaration must be exact")
    return MosaicCenterAlignmentIntent(
        alignment_id=value.alignment_id,
        mosaic_id=value.mosaic_id,
        anchor_owner=value.anchor_owner,
        axis=value.axis,
        anchor_rank_count=value.anchor_rank_count,
        overlap_ranks=value.overlap_ranks,
        source_mosaic_sha256=value.source_mosaic_sha256,
        source_asir_sha256=value.source_asir_sha256,
        source_fact_ids=value.source_fact_ids,
        evidence_ids=value.evidence_ids,
    )


def replay_mosaic_center_alignment_inventory(
    value: object,
) -> tuple[MosaicCenterAlignmentIntent, ...]:
    if type(value) is not tuple:
        raise TypeError("mosaic center alignment inventory must be an exact tuple")
    rows = tuple(replay_mosaic_center_alignment_intent(item) for item in value)
    if len({item.alignment_id for item in rows}) != len(rows):
        raise SpatialIntentProjectionError(
            "mosaic center alignment ids must be unique"
        )
    return tuple(sorted(rows, key=lambda item: item.alignment_id))


def replay_aggregate_order_intent(value: object) -> AggregateOrderIntent:
    if type(value) is not AggregateOrderIntent:
        raise TypeError("aggregate order declaration must be exact")
    return AggregateOrderIntent(
        order_id=value.order_id,
        before_scope=value.before_scope,
        after_scope=value.after_scope,
        axis=value.axis,
        source_asir_sha256=value.source_asir_sha256,
        source_fact_ids=value.source_fact_ids,
        evidence_ids=value.evidence_ids,
    )


def replay_aggregate_order_inventory(
    value: object,
) -> tuple[AggregateOrderIntent, ...]:
    if type(value) is not tuple:
        raise TypeError("aggregate order inventory must be an exact tuple")
    rows = tuple(replay_aggregate_order_intent(item) for item in value)
    if len({item.order_id for item in rows}) != len(rows):
        raise SpatialIntentProjectionError("aggregate order ids must be unique")
    return tuple(sorted(rows, key=lambda item: item.order_id))


def replay_reference_conditioned_placement_obligation(
    value: object,
) -> ReferenceConditionedPlacementObligation:
    if type(value) is not ReferenceConditionedPlacementObligation:
        raise TypeError("reference-conditioned placement declaration must be exact")
    return ReferenceConditionedPlacementObligation(
        obligation_id=value.obligation_id,
        before_scope=value.before_scope,
        after_scope=value.after_scope,
        axis=value.axis,
        strength=value.strength,
        source_relation_id=value.source_relation_id,
        source_relation_sha256=value.source_relation_sha256,
        reference_receipt_sha256=value.reference_receipt_sha256,
        reference_artifact_sha256=value.reference_artifact_sha256,
        evidence_ids=value.evidence_ids,
        witness_validation_status=value.witness_validation_status,
        witness_validation_receipt_sha256=(
            value.witness_validation_receipt_sha256
        ),
        witness_validation_observable_surface=(
            value.witness_validation_observable_surface
        ),
        witness_validated_strength=value.witness_validated_strength,
        relation_kind=value.relation_kind,
        host_scope=value.host_scope,
        guest_scopes=value.guest_scopes,
        lower_boundary=value.lower_boundary,
        upper_boundary=value.upper_boundary,
    )


def reference_conditioned_placement_obligation_from_data(
    value: object,
) -> ReferenceConditionedPlacementObligation:
    """Replay one canonical receipt payload into its exact typed obligation."""

    if type(value) is not dict:
        raise TypeError(
            "reference-conditioned placement payload must be an exact dict"
        )
    legacy_fields = {
        "schema",
        "obligation_id",
        "before_scope",
        "after_scope",
        "axis",
        "strength",
        "source_relation_id",
        "source_relation_sha256",
        "reference_receipt_sha256",
        "reference_artifact_sha256",
        "evidence_ids",
    }
    current_fields = {
        *legacy_fields,
        "provenance_status",
        "witness_validation",
    }
    cavity_fields = {
        *current_fields,
        "relation_kind",
        "cavity",
    }
    if frozenset(value) not in {
        frozenset(legacy_fields),
        frozenset(current_fields),
        frozenset(cavity_fields),
    }:
        raise SpatialIntentProjectionError(
            "reference-conditioned placement payload fields differ"
        )
    if value["schema"] not in {
        "clay.reference-conditioned-placement-obligation/v1",
        "clay.reference-conditioned-placement-obligation/v2",
        "clay.reference-conditioned-placement-obligation/v3",
    }:
        raise SpatialIntentProjectionError(
            "reference-conditioned placement payload schema differs"
        )
    expected_fields = {
        "clay.reference-conditioned-placement-obligation/v1": legacy_fields,
        "clay.reference-conditioned-placement-obligation/v2": current_fields,
        "clay.reference-conditioned-placement-obligation/v3": cavity_fields,
    }[value["schema"]]
    if set(value) != expected_fields:
        raise SpatialIntentProjectionError(
            "reference-conditioned placement payload schema/fields differ"
        )
    raw_evidence = value["evidence_ids"]
    if type(raw_evidence) is not list:
        raise TypeError(
            "reference-conditioned placement evidence payload must be a list"
        )
    witness = value.get("witness_validation")
    if witness is not None and (
        type(witness) is not dict
        or set(witness)
        != {"status", "receipt_sha256", "observable_surface", "validated_strength"}
    ):
        raise SpatialIntentProjectionError(
            "reference-conditioned witness validation payload fields differ"
        )
    cavity = value.get("cavity")
    if cavity is not None and (
        type(cavity) is not dict
        or set(cavity)
        != {
            "host_scope",
            "guest_scopes",
            "lower_boundary",
            "upper_boundary",
        }
        or type(cavity["guest_scopes"]) is not list
    ):
        raise SpatialIntentProjectionError(
            "reference-conditioned cavity payload fields differ"
        )
    result = ReferenceConditionedPlacementObligation(
        obligation_id=value["obligation_id"],
        before_scope=value["before_scope"],
        after_scope=value["after_scope"],
        axis=value["axis"],
        strength=value["strength"],
        source_relation_id=value["source_relation_id"],
        source_relation_sha256=value["source_relation_sha256"],
        reference_receipt_sha256=value["reference_receipt_sha256"],
        reference_artifact_sha256=value["reference_artifact_sha256"],
        evidence_ids=tuple(raw_evidence),
        witness_validation_status=(
            "UNVALIDATED" if witness is None else witness["status"]
        ),
        witness_validation_receipt_sha256=(
            None if witness is None else witness["receipt_sha256"]
        ),
        witness_validation_observable_surface=(
            None if witness is None else witness["observable_surface"]
        ),
        witness_validated_strength=(
            None if witness is None else witness["validated_strength"]
        ),
        relation_kind=value.get("relation_kind", "ordered_pair"),
        host_scope=None if cavity is None else cavity["host_scope"],
        guest_scopes=(
            () if cavity is None else tuple(cavity["guest_scopes"])
        ),
        lower_boundary=None if cavity is None else cavity["lower_boundary"],
        upper_boundary=None if cavity is None else cavity["upper_boundary"],
    )
    if value.get("provenance_status", result.provenance_status) != (
        result.provenance_status
    ):
        raise SpatialIntentProjectionError(
            "reference-conditioned provenance status differs from witness validation"
        )
    return result


def replay_reference_conditioned_placement_obligation_inventory(
    value: object,
) -> tuple[ReferenceConditionedPlacementObligation, ...]:
    if type(value) is not tuple:
        raise TypeError(
            "reference-conditioned placement inventory must be an exact tuple"
        )
    rows = tuple(
        replay_reference_conditioned_placement_obligation(item) for item in value
    )
    if len({item.obligation_id for item in rows}) != len(rows):
        raise SpatialIntentProjectionError(
            "reference-conditioned placement obligation ids must be unique"
        )
    semantic_identities = tuple(
        (
            item.relation_kind,
            item.host_scope if item.relation_kind == "row_bounded_cavity" else item.before_scope,
            item.guest_scopes if item.relation_kind == "row_bounded_cavity" else item.after_scope,
            item.axis,
            item.strength,
        )
        for item in rows
    )
    if len(set(semantic_identities)) != len(semantic_identities):
        raise SpatialIntentProjectionError(
            "reference-conditioned placement semantic identities must be unique"
        )
    return tuple(sorted(rows, key=lambda item: item.obligation_id))


def project_reference_conditioned_row_bounded_cavity(
    declaration: object,
    *,
    witness_validation_payload: object,
    witness_validation_receipt_sha256: str,
    source_relation_id: str,
    reference_receipt_sha256: str,
    reference_artifact_sha256: str,
    evidence_ids: tuple[str, ...],
) -> ReferenceConditionedPlacementObligation:
    """Bind one declared cavity to an exact passing OA witness receipt."""

    from clay.layout.intent import RowBoundedCavityIntent

    if type(declaration) is not RowBoundedCavityIntent:
        raise TypeError("reference-conditioned cavity declaration must be exact")
    if type(witness_validation_payload) is not dict:
        raise TypeError("row-bounded cavity witness receipt must be an exact dict")
    required_fields = {
        "schema",
        "status",
        "strength",
        "observable_surface",
        "host_scope",
        "guest_scopes",
        "lower_boundary",
        "upper_boundary",
        "reference_receipt_sha256",
        "reference_artifact_sha256",
        "evidence_ids",
    }
    if set(witness_validation_payload) != required_fields:
        raise SpatialIntentProjectionError(
            "row-bounded cavity witness receipt fields differ"
        )
    if (
        witness_validation_payload["schema"]
        != "clay.reference-row-bounded-cavity-witness-validation/v1"
        or witness_validation_payload["status"] != "PASS"
        or witness_validation_payload["strength"] != "row_bounded_cavity"
        or witness_validation_payload["observable_surface"]
        != "reference_oa_spatial_placement"
        or witness_validation_payload["host_scope"] != declaration.host
        or witness_validation_payload["guest_scopes"] != list(declaration.guests)
        or witness_validation_payload["lower_boundary"]
        != declaration.LOWER_BOUNDARY
        or witness_validation_payload["upper_boundary"]
        != declaration.UPPER_BOUNDARY
        or witness_validation_payload["reference_receipt_sha256"]
        != reference_receipt_sha256
        or witness_validation_payload["reference_artifact_sha256"]
        != reference_artifact_sha256
        or witness_validation_payload["evidence_ids"] != list(evidence_ids)
    ):
        raise SpatialIntentProjectionError(
            "row-bounded cavity witness receipt does not prove the declaration"
        )
    for value, label in (
        (witness_validation_receipt_sha256, "witness validation receipt"),
        (reference_receipt_sha256, "reference receipt"),
        (reference_artifact_sha256, "reference artifact"),
    ):
        if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
            raise SpatialIntentProjectionError(
                f"row-bounded cavity {label} SHA-256 is malformed"
            )
    receipt_bytes = (
        json.dumps(
            witness_validation_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        + "\n"
    ).encode("ascii")
    if hashlib.sha256(receipt_bytes).hexdigest() != witness_validation_receipt_sha256:
        raise SpatialIntentProjectionError(
            "row-bounded cavity witness receipt digest differs"
        )
    return ReferenceConditionedPlacementObligation(
        obligation_id=f"reference_row_bounded_cavity_{declaration.host}",
        before_scope=declaration.host,
        after_scope=f"guest_set:{','.join(declaration.guests)}",
        axis="vertical",
        strength="row_bounded_cavity",
        source_relation_id=source_relation_id,
        source_relation_sha256=declaration.sha256,
        reference_receipt_sha256=reference_receipt_sha256,
        reference_artifact_sha256=reference_artifact_sha256,
        evidence_ids=evidence_ids,
        witness_validation_status="PASS",
        witness_validation_receipt_sha256=witness_validation_receipt_sha256,
        witness_validation_observable_surface="reference_oa_spatial_placement",
        witness_validated_strength="row_bounded_cavity",
        relation_kind="row_bounded_cavity",
        host_scope=declaration.host,
        guest_scopes=declaration.guests,
        lower_boundary=declaration.LOWER_BOUNDARY,
        upper_boundary=declaration.UPPER_BOUNDARY,
    )


def replay_aggregate_alignment_inventory(
    value: object,
) -> tuple[AggregateAlignmentIntent, ...]:
    if type(value) is not tuple:
        raise TypeError("aggregate alignment inventory must be an exact tuple")
    rows = []
    for item in value:
        if type(item) is not AggregateAlignmentIntent:
            raise TypeError("aggregate alignment declaration must be exact")
        rows.append(
            AggregateAlignmentIntent(
                alignment_id=item.alignment_id,
                first_scope=item.first_scope,
                second_scope=item.second_scope,
                axis=item.axis,
                edge=item.edge,
                tolerance_ppm=item.tolerance_ppm,
                source_asir_sha256=item.source_asir_sha256,
                source_fact_ids=item.source_fact_ids,
                evidence_ids=item.evidence_ids,
            )
        )
    if len({item.alignment_id for item in rows}) != len(rows):
        raise SpatialIntentProjectionError("aggregate alignment ids must be unique")
    semantic_identities = tuple(
        (item.axis, item.edge, item.first_scope, item.second_scope)
        for item in rows
    )
    if len(set(semantic_identities)) != len(semantic_identities):
        raise SpatialIntentProjectionError(
            "aggregate alignment semantic identities must be unique"
        )
    return tuple(sorted(rows, key=lambda item: item.alignment_id))


def replay_hierarchical_macro_dag_intent(
    value: object,
) -> HierarchicalMacroDagIntent:
    if type(value) is not HierarchicalMacroDagIntent:
        raise TypeError("hierarchical macro DAG declaration must be exact")
    return HierarchicalMacroDagIntent(
        hierarchy=value.hierarchy,
        source_asir_sha256=value.source_asir_sha256,
        source_hierarchy_sha256=value.source_hierarchy_sha256,
        source_relation_ids=value.source_relation_ids,
        evidence_ids=value.evidence_ids,
    )


def _attributes(fact: HardInvariant) -> dict[str, str]:
    return dict(fact.attributes)


def _fact_nodes(fact: HardInvariant) -> tuple[str, ...]:
    return tuple(binding.node for binding in fact.bindings)


def _unique_fact(
    facts: tuple[HardInvariant, ...], *, label: str
) -> HardInvariant:
    if len(facts) != 1:
        raise SpatialIntentProjectionError(
            f"{label} must resolve to one fact, observed {len(facts)}"
        )
    return facts[0]


def _node_ordinal(node: str, owner: str) -> int:
    match = _NODE_RE.fullmatch(node)
    if match is None or match.group(1) != owner:
        raise SpatialIntentProjectionError(
            f"partition node {node!r} does not bind owner {owner!r}"
        )
    return int(match.group(2))


def project_aggregate_orders(
    ir: AbstractSpatialIR,
) -> tuple[AggregateOrderIntent, ...]:
    """Project every exact aggregate scope order carried by one ASIR."""

    if type(ir) is not AbstractSpatialIR:
        raise TypeError("aggregate order projection requires an exact AbstractSpatialIR")
    rows = []
    for fact in ir.hard_facts:
        if fact.kind is not SpatialFactKind.AGGREGATE_ORDER:
            continue
        attributes = _attributes(fact)
        if set(attributes) != {"after_scope", "axis", "before_scope"}:
            raise SpatialIntentProjectionError(
                "aggregate order fact has an incomplete semantic signature"
            )
        before_scope = _text(
            attributes["before_scope"], "aggregate order before scope"
        )
        after_scope = _text(
            attributes["after_scope"], "aggregate order after scope"
        )
        axis = attributes["axis"]
        if axis not in {"horizontal", "vertical"}:
            raise SpatialIntentProjectionError("aggregate order fact axis is invalid")
        roles = tuple(binding.role for binding in fact.bindings)
        if (
            not any(role.startswith("before_member_") for role in roles)
            or not any(role.startswith("after_member_") for role in roles)
        ):
            raise SpatialIntentProjectionError(
                "aggregate order fact lacks exact scope membership"
            )
        rows.append(
            AggregateOrderIntent(
                order_id=(
                    f"aggregate_order_{axis}_{before_scope}_{after_scope}"
                ),
                before_scope=before_scope,
                after_scope=after_scope,
                axis=axis,
                source_asir_sha256=ir.sha256,
                source_fact_ids=(fact.fact_id,),
                evidence_ids=fact.evidence_ids,
            )
        )
    return replay_aggregate_order_inventory(tuple(rows))


def project_aggregate_alignments(
    ir: AbstractSpatialIR,
) -> tuple[AggregateAlignmentIntent, ...]:
    """Project every proved aggregate scope-edge tolerance band in one ASIR."""

    if type(ir) is not AbstractSpatialIR:
        raise TypeError(
            "aggregate alignment projection requires an exact AbstractSpatialIR"
        )
    rows = []
    for fact in ir.hard_facts:
        if fact.kind is not SpatialFactKind.AGGREGATE_ALIGNMENT:
            continue
        attributes = _attributes(fact)
        if set(attributes) != {
            "axis",
            "edge",
            "first_scope",
            "second_scope",
            "tolerance_ppm",
        }:
            raise SpatialIntentProjectionError(
                "aggregate alignment fact has an incomplete semantic signature"
            )
        first_scope = _text(
            attributes["first_scope"], "aggregate alignment first scope"
        )
        second_scope = _text(
            attributes["second_scope"], "aggregate alignment second scope"
        )
        if first_scope == second_scope:
            raise SpatialIntentProjectionError(
                "aggregate alignment fact requires two distinct scopes"
            )
        first_scope, second_scope = sorted((first_scope, second_scope))
        axis = attributes["axis"]
        if axis not in {"horizontal", "vertical"}:
            raise SpatialIntentProjectionError(
                "aggregate alignment fact axis is invalid"
            )
        edge = attributes["edge"]
        if edge not in {"leading", "trailing", "center"}:
            raise SpatialIntentProjectionError(
                "aggregate alignment fact edge is invalid"
            )
        raw_tolerance = _text(
            attributes["tolerance_ppm"],
            "aggregate alignment tolerance",
        )
        if (
            not raw_tolerance.isascii()
            or not raw_tolerance.isdecimal()
            or str(int(raw_tolerance)) != raw_tolerance
        ):
            raise SpatialIntentProjectionError(
                "aggregate alignment tolerance is not canonical ppm"
            )
        tolerance_ppm = int(raw_tolerance)
        if tolerance_ppm > 1_000_000:
            raise SpatialIntentProjectionError(
                "aggregate alignment tolerance exceeds one layout span"
            )

        first_bindings = tuple(
            binding
            for binding in fact.bindings
            if binding.role.startswith("first_member_")
        )
        second_bindings = tuple(
            binding
            for binding in fact.bindings
            if binding.role.startswith("second_member_")
        )
        expected_roles = tuple(
            f"first_member_{index:03d}" for index in range(len(first_bindings))
        ) + tuple(
            f"second_member_{index:03d}" for index in range(len(second_bindings))
        )
        if (
            not first_bindings
            or not second_bindings
            or tuple(binding.role for binding in fact.bindings) != expected_roles
            or set(binding.node for binding in first_bindings)
            & set(binding.node for binding in second_bindings)
        ):
            raise SpatialIntentProjectionError(
                "aggregate alignment fact lacks exact disjoint scope membership"
            )
        rows.append(
            AggregateAlignmentIntent(
                alignment_id=(
                    f"aggregate_alignment_{axis}_{edge}_"
                    f"{first_scope}_{second_scope}"
                ),
                first_scope=first_scope,
                second_scope=second_scope,
                axis=axis,
                edge=edge,
                tolerance_ppm=tolerance_ppm,
                source_asir_sha256=ir.sha256,
                source_fact_ids=(fact.fact_id,),
                evidence_ids=fact.evidence_ids,
            )
        )
    return replay_aggregate_alignment_inventory(tuple(rows))


def project_hierarchical_macro_dag(
    ir: AbstractSpatialIR,
) -> HierarchicalMacroDagIntent:
    """Collapse occurrence ownership to logical owners without weakening topology."""

    if type(ir) is not AbstractSpatialIR:
        raise TypeError("macro DAG projection requires an exact AbstractSpatialIR")
    source = ir.macro_hierarchy
    if source is None:
        raise SpatialIntentProjectionError("ASIR carries no hierarchical macro DAG")
    levels = []
    relation_ids = []
    for level in source.levels:
        owner_sibling: dict[str, str] = {}
        projected_nodes = []
        for node in level.nodes:
            owners = []
            for occurrence_node in node.member_nodes:
                match = _NODE_RE.fullmatch(occurrence_node)
                if match is None:
                    raise SpatialIntentProjectionError(
                        "macro DAG occurrence node has no logical-owner form"
                    )
                owner = match.group(1)
                prior = owner_sibling.get(owner)
                if prior is not None and prior != node.node_id:
                    raise SpatialIntentProjectionError(
                        "macro DAG splits one logical owner across siblings"
                    )
                owner_sibling[owner] = node.node_id
                owners.append(owner)
            projected_nodes.append(
                MacroDagNode(
                    node_id=node.node_id,
                    member_nodes=tuple(sorted(set(owners))),
                )
            )
            relation_ids.append(f"macro_partition:{level.scope_id}:{node.node_id}")
        for order in level.orders:
            relation_ids.append(
                "macro_order:"
                f"{level.scope_id}:{order.axis}:{order.before}:{order.after}:"
                f"{order.strength.value}"
            )
        for alignment in level.alignments:
            relation_ids.append(
                "macro_alignment:"
                f"{level.scope_id}:{alignment.axis}:"
                + (
                    f"{alignment.edge}:"
                    if alignment.edge != "center"
                    else ""
                )
                + (
                    f"tol{alignment.tolerance_ppm}:"
                    if alignment.tolerance_ppm
                    else ""
                )
                + "+".join(alignment.nodes)
            )
        levels.append(
            MacroDagLevel(
                scope_id=level.scope_id,
                parent_node=level.parent_node,
                nodes=tuple(projected_nodes),
                orders=level.orders,
                alignments=level.alignments,
            )
        )
    logical_hierarchy = HierarchicalMacroDag(levels=tuple(levels))
    evidence = tuple(
        sorted(
            {
                evidence_id
                for fact in ir.hard_facts
                for evidence_id in fact.evidence_ids
            }
        )
    )
    if not evidence:
        raise SpatialIntentProjectionError(
            "macro DAG projection has no semantic occurrence evidence"
        )
    return HierarchicalMacroDagIntent(
        hierarchy=logical_hierarchy,
        source_asir_sha256=ir.sha256,
        source_hierarchy_sha256=source.sha256,
        source_relation_ids=tuple(sorted(relation_ids)),
        evidence_ids=evidence,
    )


def _total_order(nodes: tuple[str, ...], *, ir: AbstractSpatialIR, axis: str,
                 reaches: Callable[[str, str], bool] | None = None) -> tuple[str, ...]:
    if reaches is None:
        reaches = compile_dag_queries(ir.horizontal if axis == "horizontal" else ir.vertical)
    for index, left in enumerate(nodes):
        for right in nodes[index + 1 :]:
            if reaches(left, right) == reaches(right, left):
                raise SpatialIntentProjectionError(
                    f"mosaic {axis} order is not a strict total order"
                )
    return tuple(
        sorted(
            nodes,
            key=lambda node: sum(
                1 for other in nodes if other != node and reaches(other, node)
            ),
        )
    )


def _mosaic_row_order(
    nodes: tuple[str, ...],
    *,
    ir: AbstractSpatialIR,
    reaches: Callable[[str, str], bool] | None = None,
) -> tuple[str, ...]:
    """Order one row by envelope DAG, owner lattice, or semantic-class ranks."""

    try:
        return _total_order(nodes, ir=ir, axis="horizontal", reaches=reaches)
    except SpatialIntentProjectionError as envelope_error:
        owners = {
            match.group(1)
            for node in nodes
            if (match := _NODE_RE.fullmatch(node)) is not None
        }
        if len(owners) == 1:
            owner = next(iter(owners))
            lattice = _unique_fact(
                tuple(
                    fact
                    for fact in ir.hard_facts
                    if fact.kind is SpatialFactKind.LATTICE_SEQUENCE
                    and all(node.startswith(f"{owner}[") for node in _fact_nodes(fact))
                ),
                label=f"mosaic row lattice {owner!r}",
            )
            ranks = _lattice_axis_ranks(lattice, owner=owner, axis="horizontal")
            if any(node not in ranks for node in nodes) or len(
                {ranks[node] for node in nodes}
            ) != len(nodes):
                raise envelope_error
            return tuple(sorted(nodes, key=ranks.__getitem__))

        class_rank = _unique_fact(
            tuple(
                fact
                for fact in ir.hard_facts
                if fact.kind is SpatialFactKind.CLASS_RANK_SEQUENCE
                and _attributes(fact).get("axis") == "horizontal"
                and set(nodes) <= set(_fact_nodes(fact))
            ),
            label="mosaic row class-rank sequence",
        )
        ranks = {
            node: rank
            for rank, group in enumerate(
                _attributes(class_rank)["sequence"].split(";")
            )
            for node in group.split(",")
        }
        if any(node not in ranks for node in nodes) or len(
            {ranks[node] for node in nodes}
        ) != len(nodes):
            raise envelope_error
        return tuple(sorted(nodes, key=ranks.__getitem__))


def project_discrete_mosaic(
    ir: AbstractSpatialIR,
    *,
    mosaic_id: str,
    logical_owners: tuple[str, ...],
) -> DiscreteMosaicIntent:
    """Project one exact multi-owner row mosaic from hard ASIR facts only."""

    if type(ir) is not AbstractSpatialIR:
        raise TypeError("mosaic projection requires an exact AbstractSpatialIR")
    mosaic_id = _text(mosaic_id, "mosaic id")
    owners = _sorted_texts(logical_owners, "mosaic logical owners")

    partitions: dict[str, HardInvariant] = {}
    selected_nodes: set[str] = set()
    for owner in owners:
        fact = _unique_fact(
            tuple(
                row
                for row in ir.hard_facts
                if row.kind is SpatialFactKind.EXACT_PARTITION
                and _attributes(row).get("partition_id") == owner
                and _attributes(row).get("complete") == "true"
            ),
            label=f"mosaic partition {owner!r}",
        )
        partitions[owner] = fact
        nodes = _fact_nodes(fact)
        ordinals = tuple(sorted(_node_ordinal(node, owner) for node in nodes))
        if ordinals != tuple(range(len(nodes))):
            raise SpatialIntentProjectionError(
                f"mosaic partition {owner!r} is not an exact ordinal cover"
            )
        overlap = selected_nodes & set(nodes)
        if overlap:
            raise SpatialIntentProjectionError(
                f"mosaic partitions overlap at {min(overlap)!r}"
            )
        selected_nodes.update(nodes)

    transforms: dict[str, str] = {}
    lattice_facts: list[HardInvariant] = []
    for owner in owners:
        owner_nodes = set(_fact_nodes(partitions[owner]))
        lattice = _unique_fact(
            tuple(
                row
                for row in ir.hard_facts
                if row.kind is SpatialFactKind.LATTICE_SEQUENCE
                and set(_fact_nodes(row)) == owner_nodes
            ),
            label=f"mosaic lattice {owner!r}",
        )
        lattice_facts.append(lattice)
        ordered_nodes = tuple(sorted(owner_nodes, key=lambda node: _node_ordinal(node, owner)))
        classes = tuple(_attributes(lattice).get("transform_class", "").split(","))
        if len(classes) != len(ordered_nodes) or any(item not in _ORIENTATIONS for item in classes):
            raise SpatialIntentProjectionError(
                f"mosaic lattice {owner!r} has an incomplete transform cover"
            )
        transforms.update(zip(ordered_nodes, classes))

    if len(owners) == 1:
        owner, lattice = owners[0], lattice_facts[0]
        xs = _lattice_axis_ranks(lattice, owner=owner, axis="horizontal")
        ys = _lattice_axis_ranks(lattice, owner=owner, axis="vertical")
        if len({(xs[node], ys[node]) for node in selected_nodes}) != len(selected_nodes):
            raise SpatialIntentProjectionError("mosaic lattice repeats a row/column slot")
        ordered = tuple(sorted(selected_nodes, key=lambda node: (ys[node], xs[node])))
        used_facts = (partitions[owner], lattice)
        return DiscreteMosaicIntent(mosaic_id=mosaic_id, rows=tuple(
            tuple(MosaicOccurrence(owner, _node_ordinal(node, owner), transforms[node])
                  for node in ordered if ys[node] == row)
            for row in range(len(set(ys.values())))), source_asir_sha256=ir.sha256,
            source_fact_ids=tuple(sorted(fact.fact_id for fact in used_facts)),
            evidence_ids=tuple(sorted({evidence for fact in used_facts for evidence in fact.evidence_ids})),
            column_ranks=tuple(sorted((owner, _node_ordinal(node, owner), xs[node]) for node in ordered)))

    if any(fact.kind is SpatialFactKind.SCOPE_RANK_SEQUENCE
           and set(_fact_nodes(fact)) == selected_nodes for fact in ir.hard_facts):
        from clay.layout.spatial.lattice_mosaic_projection import project_lattice_mosaic
        return project_lattice_mosaic(ir, mosaic_id=mosaic_id, logical_owners=owners)

    horizontal_reaches = compile_dag_queries(ir.horizontal)
    vertical_reaches = compile_dag_queries(ir.vertical)
    vertical_class_ranks = tuple(
        fact
        for fact in ir.hard_facts
        if fact.kind is SpatialFactKind.CLASS_RANK_SEQUENCE
        and _attributes(fact).get("axis") == "vertical"
        and selected_nodes <= set(_fact_nodes(fact))
    )
    row_class_fact: HardInvariant | None = None
    if vertical_class_ranks:
        row_class_fact = _unique_fact(
            vertical_class_ranks,
            label="mosaic vertical class-rank sequence",
        )
        row_rows = tuple(
            (
                row_class_fact,
                tuple(
                    node
                    for node in raw_group.split(",")
                    if node in selected_nodes
                ),
            )
            for raw_group in _attributes(row_class_fact)["sequence"].split(";")
            if any(node in selected_nodes for node in raw_group.split(","))
        )
    else:
        row_rows = tuple(
            (row, selected)
            for row in ir.hard_facts
            if row.kind is SpatialFactKind.ALIGNMENT
            and _attributes(row).get("axis") == "horizontal"
            and len(
                selected := tuple(
                    node for node in _fact_nodes(row) if node in selected_nodes
                )
            )
            > 1
        )
    row_counts = {node: 0 for node in selected_nodes}
    for _fact, nodes in row_rows:
        for node in nodes:
            row_counts[node] += 1
    if not row_rows or any(count != 1 for count in row_counts.values()):
        raise SpatialIntentProjectionError(
            "mosaic row cover must bind every selected occurrence exactly once"
        )
    column_rows = []
    for fact in ir.hard_facts:
        if (
            fact.kind is not SpatialFactKind.ALIGNMENT
            or _attributes(fact).get("axis") != "vertical"
        ):
            continue
        selected = tuple(node for node in _fact_nodes(fact) if node in selected_nodes)
        rank_match = re.fullmatch(r"alignment:vertical:(\d+)", fact.fact_id)
        if len(selected) > 1 and rank_match is not None:
            column_rows.append((int(rank_match.group(1)), fact, selected))
    column_counts = {node: 0 for node in selected_nodes}
    for _rank, _fact, nodes in column_rows:
        for node in nodes:
            column_counts[node] += 1
    maximum_row_width = max(len(nodes) for _fact, nodes in row_rows)
    maximum_row_nodes = {
        node
        for _fact, nodes in row_rows
        if len(nodes) == maximum_row_width
        for node in nodes
    }
    column_facts: tuple[HardInvariant, ...] = ()
    if (
        column_rows
        and all(column_counts[node] == 1 for node in maximum_row_nodes)
        and all(
            column_counts[node] == 0
            for node in selected_nodes - maximum_row_nodes
        )
        and len({rank for rank, _fact, _nodes in column_rows}) == len(column_rows)
    ):
        rank_by_node = {
            node: rank
            for rank, _fact, nodes in column_rows
            for node in nodes
        }
        ordered_within_rows = tuple(
            (
                tuple(sorted(nodes, key=rank_by_node.__getitem__))
                if set(nodes) <= set(rank_by_node)
                else _mosaic_row_order(nodes, ir=ir, reaches=horizontal_reaches)
            )
            for _fact, nodes in row_rows
        )
        if any(
            set(row) <= set(rank_by_node)
            and len({rank_by_node[node] for node in row}) != len(row)
            for row in ordered_within_rows
        ):
            raise SpatialIntentProjectionError(
                "mosaic column cover assigns two row members to one column"
            )
        column_facts = tuple(
            fact for _rank, fact, _nodes in sorted(column_rows)
        )
    else:
        ordered_within_rows = tuple(
            _mosaic_row_order(nodes, ir=ir, reaches=horizontal_reaches)
            for _fact, nodes in row_rows
        )

    if row_class_fact is not None:
        ordered_rows = ordered_within_rows
    else:
        row_predecessors: dict[int, int] = {}
        for left_index, left in enumerate(ordered_within_rows):
            predecessors = 0
            for right_index, right in enumerate(ordered_within_rows):
                if left_index == right_index:
                    continue
                right_before_left = all(
                    vertical_reaches(right_node, left_node)
                    for right_node in right
                    for left_node in left
                )
                left_before_right = all(
                    vertical_reaches(left_node, right_node)
                    for left_node in left
                    for right_node in right
                )
                if right_before_left == left_before_right:
                    raise SpatialIntentProjectionError(
                        "mosaic vertical row order is not uniquely proven"
                    )
                predecessors += int(right_before_left)
            row_predecessors[left_index] = predecessors
        if set(row_predecessors.values()) != set(range(len(ordered_within_rows))):
            raise SpatialIntentProjectionError("mosaic vertical row order is cyclic")
        ordered_rows = tuple(
            ordered_within_rows[index]
            for index in sorted(row_predecessors, key=row_predecessors.__getitem__)
        )

    maximum_columns = max(len(row) for row in ordered_rows)
    base_rows = tuple(row for row in ordered_rows if len(row) == maximum_columns)
    rank_intervals = []
    class_rank_facts = tuple(
        fact
        for fact in ir.hard_facts
        if fact.kind is SpatialFactKind.CLASS_RANK_SEQUENCE
        and _attributes(fact).get("axis") == "horizontal"
        and selected_nodes <= set(_fact_nodes(fact))
    )
    if len(base_rows) != len(ordered_rows):
        base_columns = tuple(
            tuple(row[column] for row in base_rows)
            for column in range(maximum_columns)
        )
        class_rank = _unique_fact(
            class_rank_facts,
            label="mosaic horizontal class-rank sequence",
        )
        rank_by_node = {
            node: rank
            for rank, group in enumerate(
                _attributes(class_rank)["sequence"].split(";")
            )
            for node in group.split(",")
        }
        if not selected_nodes <= set(rank_by_node):
            raise SpatialIntentProjectionError(
                "mosaic class-rank sequence has an incomplete node cover"
            )
        base_column_ranks = []
        for column in base_columns:
            values = {rank_by_node[node] for node in column}
            if len(values) != 1:
                raise SpatialIntentProjectionError(
                    "mosaic base column lacks one class anchor rank"
                )
            base_column_ranks.append(next(iter(values)))
        for row in ordered_rows:
            if len(row) == maximum_columns:
                continue
            for node in row:
                matches = tuple(
                    rank
                    for rank in range(maximum_columns - 1)
                    if base_column_ranks[rank]
                    < rank_by_node[node]
                    < base_column_ranks[rank + 1]
                )
                if len(matches) != 1:
                    raise SpatialIntentProjectionError(
                        "mosaic ragged-row occurrence lacks one adjacent rank interval"
                    )
                parsed_node = _NODE_RE.fullmatch(node)
                assert parsed_node is not None
                rank_intervals.append(
                    (
                        parsed_node.group(1),
                        int(parsed_node.group(2)),
                        matches[0],
                        matches[0] + 1,
                    )
                )

    row_authority_facts = (
        (row_class_fact,)
        if row_class_fact is not None
        else tuple(fact for fact, _nodes in row_rows)
    )
    used_facts = tuple((
        *partitions.values(),
        *lattice_facts,
        *row_authority_facts,
        *column_facts,
        *class_rank_facts,
    ))
    return DiscreteMosaicIntent(
        mosaic_id=mosaic_id,
        rows=tuple(
            tuple(
                MosaicOccurrence(
                    logical_owner=_NODE_RE.fullmatch(node).group(1),  # type: ignore[union-attr]
                    occurrence_ordinal=int(_NODE_RE.fullmatch(node).group(2)),  # type: ignore[union-attr]
                    transform=transforms[node],
                )
                for node in row
            )
            for row in ordered_rows
        ),
        source_asir_sha256=ir.sha256,
        source_fact_ids=tuple(sorted(fact.fact_id for fact in used_facts)),
        evidence_ids=tuple(sorted({
            evidence for fact in used_facts for evidence in fact.evidence_ids
        })),
        rank_intervals=tuple(sorted(rank_intervals)),
    )


def _lattice_axis_ranks(
    fact: HardInvariant,
    *,
    owner: str,
    axis: str,
) -> dict[str, int]:
    sequence = _attributes(fact).get("sequence", "")
    axis_index = 0 if axis == "horizontal" else 1
    ranks: dict[str, int] = {}
    for raw in sequence.split(";"):
        match = re.fullmatch(r"(\d+):(\d+),(\d+)", raw)
        if match is None:
            raise SpatialIntentProjectionError(
                f"anchor lattice {owner!r} has a malformed rank sequence"
            )
        ordinal = int(match.group(1))
        node = f"{owner}[{ordinal:03d}]"
        if node in ranks:
            raise SpatialIntentProjectionError(
                f"anchor lattice {owner!r} repeats an occurrence ordinal"
            )
        ranks[node] = int(match.group(2 + axis_index))
    if set(ranks) != set(_fact_nodes(fact)):
        raise SpatialIntentProjectionError(
            f"anchor lattice {owner!r} lacks an exact rank cover"
        )
    observed = tuple(sorted(set(ranks.values())))
    if observed != tuple(range(len(observed))):
        raise SpatialIntentProjectionError(
            f"anchor lattice {owner!r} ranks are not contiguous"
        )
    return ranks


def project_mosaic_center_alignment(
    ir: AbstractSpatialIR,
    *,
    alignment_id: str,
    mosaic: DiscreteMosaicIntent,
    anchor_owner: str,
    axis: str,
) -> MosaicCenterAlignmentIntent:
    """Project a symmetric central-overlap relation from one ASIR axis DAG."""

    if type(ir) is not AbstractSpatialIR:
        raise TypeError("mosaic center projection requires an exact AbstractSpatialIR")
    mosaic = replay_discrete_mosaic_intent(mosaic)
    alignment_id = _text(alignment_id, "mosaic center alignment id")
    anchor_owner = _text(anchor_owner, "mosaic center alignment anchor owner")
    if axis not in {"horizontal", "vertical"}:
        raise SpatialIntentProjectionError(
            "mosaic center projection axis must be horizontal or vertical"
        )
    if mosaic.source_asir_sha256 != ir.sha256:
        raise SpatialIntentProjectionError(
            "mosaic center projection ASIR differs from its mosaic authority"
        )
    partition = _unique_fact(
        tuple(
            fact
            for fact in ir.hard_facts
            if fact.kind is SpatialFactKind.EXACT_PARTITION
            and _attributes(fact).get("partition_id") == anchor_owner
            and _attributes(fact).get("complete") == "true"
        ),
        label=f"mosaic center anchor partition {anchor_owner!r}",
    )
    anchor_nodes = set(_fact_nodes(partition))
    if anchor_nodes & {item.node_id for item in mosaic.occurrences}:
        raise SpatialIntentProjectionError(
            "mosaic center anchor overlaps mosaic occurrence ownership"
        )
    lattice = _unique_fact(
        tuple(
            fact
            for fact in ir.hard_facts
            if fact.kind is SpatialFactKind.LATTICE_SEQUENCE
            and set(_fact_nodes(fact)) == anchor_nodes
        ),
        label=f"mosaic center anchor lattice {anchor_owner!r}",
    )
    ranks = _lattice_axis_ranks(lattice, owner=anchor_owner, axis=axis)
    by_rank = {
        rank: tuple(sorted(node for node, value in ranks.items() if value == rank))
        for rank in sorted(set(ranks.values()))
    }
    target_nodes = tuple(item.node_id for item in mosaic.occurrences)
    dag = ir.horizontal if axis == "horizontal" else ir.vertical
    reaches = compile_dag_queries(dag)
    relation_by_rank: dict[int, str] = {}
    for rank, nodes in by_rank.items():
        before = all(reaches(node, target) for node in nodes for target in target_nodes)
        after = all(reaches(target, node) for node in nodes for target in target_nodes)
        if before and after:
            raise SpatialIntentProjectionError(
                "mosaic center projection contains a cyclic anchor rank"
            )
        relation_by_rank[rank] = "before" if before else "after" if after else "overlap"
    overlap = tuple(
        rank for rank in sorted(relation_by_rank)
        if relation_by_rank[rank] == "overlap"
    )
    expected = tuple(
        "before" if rank < overlap[0] else "after" if rank > overlap[-1] else "overlap"
        for rank in sorted(relation_by_rank)
    ) if overlap else ()
    observed = tuple(relation_by_rank[rank] for rank in sorted(relation_by_rank))
    if not overlap or observed != expected:
        raise SpatialIntentProjectionError(
            "mosaic center projection lacks one monotone overlap interval"
        )
    used_facts = tuple(
        fact
        for fact in ir.hard_facts
        if fact.fact_id in set(mosaic.source_fact_ids) | {partition.fact_id, lattice.fact_id}
    )
    return MosaicCenterAlignmentIntent(
        alignment_id=alignment_id,
        mosaic_id=mosaic.mosaic_id,
        anchor_owner=anchor_owner,
        axis=axis,
        anchor_rank_count=len(by_rank),
        overlap_ranks=overlap,
        source_mosaic_sha256=mosaic.sha256,
        source_asir_sha256=ir.sha256,
        source_fact_ids=tuple(sorted({*mosaic.source_fact_ids, partition.fact_id, lattice.fact_id})),
        evidence_ids=tuple(sorted({
            *mosaic.evidence_ids,
            *(evidence for fact in used_facts for evidence in fact.evidence_ids),
        })),
    )


__all__ = [
    "AggregateAlignmentIntent",
    "AggregateOrderIntent",
    "ReferenceConditionedPlacementObligation",
    "DiscreteMosaicIntent",
    "HierarchicalMacroDagIntent",
    "MosaicCenterAlignmentIntent",
    "MosaicOccurrence",
    "SpatialIntentProjectionError",
    "project_discrete_mosaic",
    "project_aggregate_alignments",
    "project_aggregate_orders",
    "project_hierarchical_macro_dag",
    "project_mosaic_center_alignment",
    "replay_discrete_mosaic_intent",
    "replay_discrete_mosaic_inventory",
    "replay_mosaic_center_alignment_intent",
    "replay_mosaic_center_alignment_inventory",
    "replay_aggregate_alignment_inventory",
    "replay_aggregate_order_intent",
    "replay_aggregate_order_inventory",
    "replay_reference_conditioned_placement_obligation",
    "replay_reference_conditioned_placement_obligation_inventory",
    "reference_conditioned_placement_obligation_from_data",
    "project_reference_conditioned_row_bounded_cavity",
    "replay_hierarchical_macro_dag_intent",
]
