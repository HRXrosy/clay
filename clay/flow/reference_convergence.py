"""Layered, digest-bound convergence claims for the three reference circuits."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from types import MappingProxyType
from typing import Mapping

from clay.core_contract import VerificationStatus
from clay.flow.portfolio_signoff import ArtifactLocator


REFERENCE_CONVERGENCE_ROW_SCHEMA = "clay.reference-convergence-row/v1"
REFERENCE_CONVERGENCE_LEDGER_SCHEMA = "clay.three-circuit-reference-convergence/v2"
INTENT_MEASUREMENT_RECEIPT_SCHEMA = "clay.intent-measurement-receipt/v1"
FROZEN_INPUTS_SCHEMA = "clay.reference-convergence-frozen-inputs/v3"
FROZEN_INPUT_REFERENCE_SCHEMA = "clay.frozen-input-reference/v1"
FROZEN_INPUTS_RELATIVE = "ldo_dev/run/frozen_inputs.json"
FROZEN_CODE_BASELINE_COMMIT = "bf7106a2f10cd1f513c2b9ec4c76b910892d5859"
FROZEN_CODE_BASELINE_ARC = "arc-18"
FROZEN_CODE_BASELINE_REQUIRED_PATHS = (
    "clay/flow/reference_convergence.py",
    "clay/layout/extract/compaction.py",
    "clay/layout/flow_placement.py",
    "clay/layout/intent.py",
    "clay/layout/place/final_placement_state.py",
    "clay/layout/place_blocks_2d.py",
    "clay/layout/placement_infeasibility.py",
    "clay/layout/spatial/__init__.py",
    "clay/layout/spatial/intent_projection.py",
    "ldo_dev/run/attribution/reference_obligation_rederive.py",
    "ldo_dev/run/attribution/reference_witness_replay.py",
    "ldo_dev/run/canonical_spatial_model.py",
    "ldo_dev/run/placement_similarity_poc.py",
    "ldo_dev/run/run_macro_role_place.py",
)
FROZEN_DESIGN_CLAY_RELATIVE = "ldo_dev/source/reference_ldo.clay"
FROZEN_DESIGN_CLAY_SHA256 = (
    "a0009ebd045fe9db7779822b8660a5a438821c632c77b4bd9f309d128e3b0b6d"
)
FROZEN_REFERENCE_GDS_RELATIVE = "ldo_dev/ref/reference_layout_flat.gds"
FROZEN_REFERENCE_GDS_SHA256 = (
    "cb8a06166a3eb6005a5538b8596e1cc38df7482f334d97d5ae56327afacd887e"
)
FROZEN_OA_SPATIAL_ANCHOR_GDS_RELATIVE = "ldo_dev/ref/reference_layout_full.gds"
FROZEN_OA_SPATIAL_ANCHOR_GDS_SHA256 = (
    "c11e703aa5f29e2d6bc0b69ba2a160b3a625454207db83ff33686d0daaaca629"
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
_HUMAN_GAP_IDENTIFIER = re.compile(r"[a-z][a-z0-9_]*(?:[.-][a-z0-9_]+)*")


REFERENCE_DIMENSIONS = (
    "physical_legality",
    "topology_fidelity",
    "intent_coverage",
    "geometric_quality",
    "pex_confidence",
    "electrical_function",
    "robustness",
)

# P2 optimization axes are intentionally separate from the publication maturity
# ladder above.  They are a vector of exact-GDS observations, not another scalar
# signoff score.
HUMAN_GAP_DIMENSIONS = (
    "legality",
    "intent",
    "macro_structure",
    "floorplan",
    "compactness",
    "routing_topology",
    "parasitics",
)

HUMAN_GAP_VECTOR_SCHEMA = "clay.human-gap-vector/v1"


class HumanGapAuthority(str, Enum):
    """The evidence authority allowed to set one human-gap field."""

    COMPLETE_REFERENCE = "complete_reference"
    SPATIAL_ADVISORY = "spatial_advisory"
    SAME_PDK_BASELINE = "same_pdk_baseline"
    PDK_LOWER_BOUND = "pdk_lower_bound"


class HumanGapLevel(str, Enum):
    """Granularity of one observation; device authority is checked explicitly."""

    ARTIFACT = "artifact"
    DEVICE = "device"
    STRUCTURE = "structure"
    FLOORPLAN = "floorplan"
    ROUTING = "routing"
    ELECTRICAL = "electrical"


_HUMAN_GAP_AUTHORITY_EVIDENCE = MappingProxyType(
    {
        HumanGapAuthority.COMPLETE_REFERENCE: ("reference_scope", "reference_gds"),
        HumanGapAuthority.SPATIAL_ADVISORY: ("reference_scope",),
        HumanGapAuthority.SAME_PDK_BASELINE: ("same_pdk_baseline",),
        HumanGapAuthority.PDK_LOWER_BOUND: ("pdk_rules",),
    }
)

_SPATIAL_ADVISORY_MEASUREMENT_KEYS = frozenset(
    {
        "reference_available",
        "region_tier_ordinals",
        "region_column_ordinals",
        "left_of_relations",
        "above_relations",
        "adjacency_relations",
        "banked_regions",
        "wide_block_regions",
        "required_relation_count",
        "satisfied_relation_count",
        "area_ratio",
        "aspect_relative_error",
        "whitespace_fraction",
        "routing_density_ratio",
        "macro_adjacency_jaccard",
    }
)

_SPATIAL_ADVISORY_THRESHOLD_KEYS = frozenset(
    {
        "advisory_only",
        "max_area_ratio",
        "max_aspect_relative_error",
        "max_whitespace_fraction",
        "min_routing_density_ratio",
        "max_routing_density_ratio",
        "min_macro_adjacency_jaccard",
    }
)

_COMPLETE_REFERENCE_MEASUREMENT_KEYS = _SPATIAL_ADVISORY_MEASUREMENT_KEYS | frozenset(
    {
        "od_fill_fraction",
        "shared_od_fraction",
        "dummy_adjacency_satisfied",
        "dummy_adjacency_required",
        "centroid_residual_um",
        "isolated_tap_count",
        "largest_empty_rectangle_um2",
        "m2_vertical_edge_fraction",
        "member_conservation",
        "matching_residual",
    }
)

_COMPLETE_REFERENCE_THRESHOLD_KEYS = (
    _SPATIAL_ADVISORY_THRESHOLD_KEYS - {"advisory_only"}
) | frozenset(
    {
        "min_od_fill_fraction",
        "min_shared_od_fraction",
        "required_dummy_adjacency",
        "max_centroid_residual_um",
        "max_isolated_tap_count",
        "max_largest_empty_rectangle_um2",
        "max_m2_vertical_edge_fraction",
        "require_member_conservation",
        "max_matching_residual",
    }
)


class Maturity(str, Enum):
    NONE = "NONE"
    PHYSICAL_LEGAL = "PHYSICAL_LEGAL"
    REFERENCE_AWARE = "REFERENCE_AWARE"
    ELECTRICALLY_CHARACTERIZED = "ELECTRICALLY_CHARACTERIZED"
    ROBUST = "ROBUST"


MATURITY_REQUIREMENTS = MappingProxyType(
    {
        Maturity.PHYSICAL_LEGAL: REFERENCE_DIMENSIONS[:1],
        Maturity.REFERENCE_AWARE: REFERENCE_DIMENSIONS[:4],
        Maturity.ELECTRICALLY_CHARACTERIZED: REFERENCE_DIMENSIONS[:6],
        Maturity.ROBUST: REFERENCE_DIMENSIONS,
    }
)


class ReferenceConvergenceError(ValueError):
    """A convergence claim is incomplete, ambiguous, or stale."""


class HumanGapVectorError(ReferenceConvergenceError):
    """A P2 quality observation is cross-bound or exceeds its authority."""


def validate_runner_neutral_measurement_payload(payload: object) -> object:
    """Reject aggregate verdict/maturity authority anywhere in a runner receipt."""

    def visit(value: object, *, path: str) -> None:
        if isinstance(value, Mapping):
            for raw_key, child in value.items():
                if type(raw_key) is not str or not raw_key:
                    raise ReferenceConvergenceError(
                        f"runner-neutral measurement key is invalid at {path}"
                    )
                normalized = raw_key.casefold().replace("_", "")
                if (
                    "verdict" in normalized
                    or "maturity" in normalized
                    or normalized in {"dimension", "dimensions"}
                ):
                    raise ReferenceConvergenceError(
                        f"runner-neutral measurement uses reserved aggregate authority key: {raw_key}"
                    )
                visit(child, path=f"{path}.{raw_key}")
            return
        if isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, path=f"{path}[{index}]")
            return
        if value is None or type(value) in {str, bool, int}:
            return
        if type(value) is float and math.isfinite(value):
            return
        raise ReferenceConvergenceError(
            f"runner-neutral measurement contains non-canonical JSON data at {path}"
        )

    visit(payload, path="$receipt")
    return payload


def _require_string_tuple(value: object, *, label: str) -> tuple[str, ...]:
    if type(value) is not tuple or any(
        type(item) is not str or not item for item in value
    ):
        raise ReferenceConvergenceError(f"{label} must be a tuple of non-empty strings")
    return value


def _require_measurement_number(
    value: object, *, label: str, allow_none: bool = False
) -> float | None:
    if value is None and allow_none:
        return None
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        raise ReferenceConvergenceError(f"{label} must be a finite number")
    return float(value)


@dataclass(frozen=True)
class IntentObligationMeasurement:
    """One raw exact-GDS intent observation, not an aggregate verdict."""

    kind: str
    pair: tuple[str, str]
    measured_imbalance: float | None
    rel_tol: float
    measurement_status: VerificationStatus
    over_budget: float
    missing_nets: tuple[str, ...]
    declaration_ids: tuple[str, ...]
    measurements: Mapping[str, float | None]
    intent_relation: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.kind, label="intent measurement kind")
        if (
            type(self.pair) is not tuple
            or len(self.pair) != 2
            or any(type(item) is not str or not item for item in self.pair)
        ):
            raise ReferenceConvergenceError(
                "intent measurement pair must contain two non-empty member IDs"
            )
        _require_measurement_number(
            self.measured_imbalance,
            label="intent measured imbalance",
            allow_none=True,
        )
        tolerance = _require_measurement_number(
            self.rel_tol, label="intent relative tolerance"
        )
        if tolerance is None or tolerance < 0.0:
            raise ReferenceConvergenceError(
                "intent relative tolerance must be nonnegative"
            )
        over_budget = _require_measurement_number(
            self.over_budget, label="intent over-budget measurement"
        )
        if over_budget is None or over_budget < 0.0:
            raise ReferenceConvergenceError(
                "intent over-budget measurement must be nonnegative"
            )
        try:
            status = VerificationStatus(self.measurement_status)
        except (TypeError, ValueError) as exc:
            raise ReferenceConvergenceError(
                "intent measurement status is invalid"
            ) from exc
        object.__setattr__(self, "measurement_status", status)
        missing_nets = _require_string_tuple(
            self.missing_nets, label="intent missing nets"
        )
        closed_statuses = {
            VerificationStatus.PASS,
            VerificationStatus.FAIL,
            VerificationStatus.UNMEASURABLE,
            VerificationStatus.UNSUPPORTED,
        }
        if status not in closed_statuses:
            raise ReferenceConvergenceError(
                "intent measurement requires a closed pass/fail/unmeasurable/unsupported status"
            )
        measured = self.measured_imbalance
        if status in {VerificationStatus.PASS, VerificationStatus.FAIL}:
            if measured is None:
                raise ReferenceConvergenceError(
                    "intent measured pass/fail requires finite measured imbalance"
                )
            expected_status = (
                VerificationStatus.PASS
                if float(measured) <= float(tolerance)
                else VerificationStatus.FAIL
            )
            if status is not expected_status:
                raise ReferenceConvergenceError(
                    "intent measurement status conflicts with measured imbalance and tolerance"
                )
            expected_over_budget = max(
                0.0, float(measured) - float(tolerance)
            )
            if float(over_budget) != expected_over_budget:
                raise ReferenceConvergenceError(
                    "intent over-budget differs from measured imbalance and tolerance"
                )
            if missing_nets:
                raise ReferenceConvergenceError(
                    "measured pass/fail obligation cannot carry missing nets"
                )
        else:
            if measured is not None:
                raise ReferenceConvergenceError(
                    f"intent {status.value} obligation must not carry measured imbalance"
                )
            if float(over_budget) != 0.0:
                raise ReferenceConvergenceError(
                    f"intent {status.value} obligation must have zero over-budget"
                )
            if not missing_nets:
                raise ReferenceConvergenceError(
                    f"intent {status.value} obligation must name its missing capability/evidence"
                )
            if status is VerificationStatus.UNSUPPORTED and self.kind != "matching":
                raise ReferenceConvergenceError(
                    "intent unsupported status is not a Task4 capability outcome for this obligation kind"
                )
        declarations = _require_string_tuple(
            self.declaration_ids, label="intent declaration IDs"
        )
        if len(set(declarations)) != len(declarations):
            raise ReferenceConvergenceError(
                "intent obligation has duplicate declaration IDs"
            )
        relation = self.intent_relation
        if relation is not None and (
            type(relation) is not str or relation not in {"left_of", "above"}
        ):
            raise ReferenceConvergenceError(
                "intent order relation must be left_of, above, or null"
            )
        if self.kind == "order":
            if relation is None:
                raise ReferenceConvergenceError(
                    "intent order measurement must carry its authored relation"
                )
        elif relation is not None:
            raise ReferenceConvergenceError(
                "only an order measurement may carry an intent relation"
            )
        if not isinstance(self.measurements, Mapping) or any(
            type(name) is not str or not name
            for name in self.measurements
        ):
            raise ReferenceConvergenceError(
                "intent detailed measurements are invalid"
            )
        normalized_measurements: dict[str, float | None] = {}
        for name, value in self.measurements.items():
            normalized_measurements[name] = _require_measurement_number(
                value,
                label=f"intent measurement {name}",
                allow_none=True,
            )
        object.__setattr__(
            self, "measurements", MappingProxyType(normalized_measurements)
        )
        try:
            from clay.layout.intent import LayoutObligation

            reconstructed = LayoutObligation(
                kind=self.kind,
                pair=self.pair,
                measured_imbalance=self.measured_imbalance,
                rel_tol=self.rel_tol,
                missing_nets=self.missing_nets,
                verification_status=status,
                declaration_ids=self.declaration_ids,
                measurements={
                    name: value
                    for name, value in normalized_measurements.items()
                    if value is not None
                },
            )
        except (TypeError, ValueError) as exc:
            raise ReferenceConvergenceError(
                f"intent measurement does not reconstruct a closed LayoutObligation: {exc}"
            ) from exc
        if reconstructed.status != status.value:
            raise ReferenceConvergenceError(
                "intent measurement status differs from reconstructed LayoutObligation"
            )
        if float(reconstructed.over_budget) != float(over_budget):
            raise ReferenceConvergenceError(
                "intent over-budget differs from reconstructed LayoutObligation"
            )

    def canonical_data(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "pair": list(self.pair),
            "measured_imbalance": self.measured_imbalance,
            "rel_tol": self.rel_tol,
            "measurement_status": self.measurement_status.value,
            "over_budget": self.over_budget,
            "missing_nets": list(self.missing_nets),
            "declaration_ids": list(self.declaration_ids),
            "intent_relation": self.intent_relation,
            "measurements": dict(self.measurements),
        }


@dataclass(frozen=True)
class IntentMeasurementReceipt:
    """Strict runner-neutral inventory of intent observations on one exact GDS."""

    circuit_id: str
    topology_id: str
    pdk_id: str
    exact_gds_sha256: str
    intent_digest: str
    declaration_inventory: tuple[str, ...]
    obligation_rows: tuple[IntentObligationMeasurement, ...]

    def __post_init__(self) -> None:
        from clay.flow.pdk_matrix import FOUR_PDK_IDS

        _require_text(self.circuit_id, label="intent receipt circuit ID")
        _require_text(self.topology_id, label="intent receipt topology ID")
        if self.pdk_id not in FOUR_PDK_IDS:
            raise ReferenceConvergenceError("intent receipt PDK is invalid")
        _require_sha256(self.exact_gds_sha256, label="intent receipt exact GDS")
        if (
            type(self.intent_digest) is not str
            or len(self.intent_digest) != 16
            or any(character not in "0123456789abcdef" for character in self.intent_digest)
        ):
            raise ReferenceConvergenceError("intent receipt digest is invalid")
        inventory = _require_string_tuple(
            self.declaration_inventory,
            label="intent declaration inventory",
        )
        if inventory != tuple(sorted(set(inventory))):
            raise ReferenceConvergenceError(
                "intent declaration inventory must be sorted and unique"
            )
        if type(self.obligation_rows) is not tuple or any(
            not isinstance(item, IntentObligationMeasurement)
            for item in self.obligation_rows
        ):
            raise ReferenceConvergenceError("intent obligation rows are not typed")
        row_keys = tuple(
            (item.kind, item.pair, item.intent_relation or "", item.declaration_ids)
            for item in self.obligation_rows
        )
        if row_keys != tuple(sorted(row_keys)) or len(set(row_keys)) != len(row_keys):
            raise ReferenceConvergenceError(
                "intent obligation declaration rows must be canonical and unique"
            )
        declared = tuple(
            declaration
            for item in self.obligation_rows
            for declaration in item.declaration_ids
        )
        if len(set(declared)) != len(declared) or tuple(sorted(declared)) != inventory:
            raise ReferenceConvergenceError(
                "intent obligation rows do not exactly cover the declaration inventory"
            )
        validate_runner_neutral_measurement_payload(self.canonical_data())

    def canonical_data(self) -> dict[str, object]:
        return {
            "schema": INTENT_MEASUREMENT_RECEIPT_SCHEMA,
            "circuit_id": self.circuit_id,
            "topology_id": self.topology_id,
            "pdk_id": self.pdk_id,
            "exact_gds_sha256": self.exact_gds_sha256,
            "intent_digest": self.intent_digest,
            "declaration_inventory": list(self.declaration_inventory),
            "obligation_rows": [item.canonical_data() for item in self.obligation_rows],
        }


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ReferenceConvergenceError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _strict_fields(
    value: object, *, expected: set[str], label: str
) -> Mapping[str, object]:
    if type(value) is not dict:
        raise ReferenceConvergenceError(f"{label} must be an object")
    if set(value) != expected:
        raise ReferenceConvergenceError(
            f"{label} fields differ: missing={sorted(expected - set(value))}, "
            f"unknown={sorted(set(value) - expected)}"
        )
    return value


def _intent_obligation_from_mapping(value: object) -> IntentObligationMeasurement:
    raw = _strict_fields(
        value,
        expected={
            "kind",
            "pair",
            "measured_imbalance",
            "rel_tol",
            "measurement_status",
            "over_budget",
            "missing_nets",
            "declaration_ids",
            "intent_relation",
            "measurements",
        },
        label="intent obligation measurement",
    )
    if type(raw["pair"]) is not list:
        raise ReferenceConvergenceError("intent measurement pair must be an array")
    if type(raw["missing_nets"]) is not list or type(raw["declaration_ids"]) is not list:
        raise ReferenceConvergenceError(
            "intent missing nets and declaration IDs must be arrays"
        )
    try:
        return IntentObligationMeasurement(
            kind=raw["kind"],  # type: ignore[arg-type]
            pair=tuple(raw["pair"]),  # type: ignore[arg-type]
            measured_imbalance=raw["measured_imbalance"],  # type: ignore[arg-type]
            rel_tol=raw["rel_tol"],  # type: ignore[arg-type]
            measurement_status=raw["measurement_status"],  # type: ignore[arg-type]
            over_budget=raw["over_budget"],  # type: ignore[arg-type]
            missing_nets=tuple(raw["missing_nets"]),  # type: ignore[arg-type]
            declaration_ids=tuple(raw["declaration_ids"]),  # type: ignore[arg-type]
            measurements=raw["measurements"],  # type: ignore[arg-type]
            intent_relation=raw["intent_relation"],  # type: ignore[arg-type]
        )
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ReferenceConvergenceError):
            raise
        raise ReferenceConvergenceError(
            f"intent obligation measurement is invalid: {exc}"
        ) from exc


def intent_measurement_receipt_from_mapping(
    value: object,
) -> IntentMeasurementReceipt:
    raw = _strict_fields(
        value,
        expected={
            "schema",
            "circuit_id",
            "topology_id",
            "pdk_id",
            "exact_gds_sha256",
            "intent_digest",
            "declaration_inventory",
            "obligation_rows",
        },
        label="intent measurement receipt",
    )
    if raw["schema"] != INTENT_MEASUREMENT_RECEIPT_SCHEMA:
        raise ReferenceConvergenceError(
            "intent measurement receipt schema is unsupported"
        )
    if type(raw["declaration_inventory"]) is not list or type(
        raw["obligation_rows"]
    ) is not list:
        raise ReferenceConvergenceError(
            "intent declaration inventory/obligation rows must be arrays"
        )
    validate_runner_neutral_measurement_payload(raw)
    return IntentMeasurementReceipt(
        circuit_id=raw["circuit_id"],  # type: ignore[arg-type]
        topology_id=raw["topology_id"],  # type: ignore[arg-type]
        pdk_id=raw["pdk_id"],  # type: ignore[arg-type]
        exact_gds_sha256=raw["exact_gds_sha256"],  # type: ignore[arg-type]
        intent_digest=raw["intent_digest"],  # type: ignore[arg-type]
        declaration_inventory=tuple(raw["declaration_inventory"]),  # type: ignore[arg-type]
        obligation_rows=tuple(
            _intent_obligation_from_mapping(item)
            for item in raw["obligation_rows"]  # type: ignore[union-attr]
        ),
    )


def _required_declared_intents(parsed_intent: object) -> tuple[object, ...]:
    """Independently enumerate every Task4-enforced exact-Clay declaration."""

    from clay.layout.intent_completeness import (
        _UNENFORCED_FAMILIES,
        _obligation_family,
        enumerate_declared_intents,
    )

    return tuple(
        declared
        for declared in enumerate_declared_intents(parsed_intent)  # type: ignore[arg-type]
        if _obligation_family(declared.kind) not in _UNENFORCED_FAMILIES
    )


def _required_declaration_inventory(
    required_intents: tuple[object, ...],
) -> tuple[str, ...]:
    inventory = tuple(
        sorted(
            str(declaration)
            for declared in required_intents
            for declaration in declared.declaration_ids  # type: ignore[attr-defined]
        )
    )
    if len(set(inventory)) != len(inventory):
        raise ReferenceConvergenceError(
            "exact Clay declaration inventory contains duplicates"
        )
    return inventory


def _strict_required_intent_row_signatures(
    parsed_intent: object,
    required_intents: tuple[object, ...],
) -> tuple[tuple[str, tuple[str, str], tuple[str, ...]], ...]:
    """Bind matching and place-group rows to their exact Task4 producer identity."""

    from clay.layout.intent_completeness import _obligation_family

    signatures: list[tuple[str, tuple[str, str], tuple[str, ...]]] = []
    declared_place_groups = tuple(
        declared
        for declared in required_intents
        if _obligation_family(declared.kind) == "place_group"  # type: ignore[attr-defined]
    )
    groups_by_targets: dict[frozenset[str], object] = {}
    for group in tuple(getattr(parsed_intent, "place_groups", ()) or ()):
        targets = tuple(str(member) for member in group.devices)
        if group.group:
            targets = (*targets, str(group.group))
        elif not targets:
            targets = (str(group.group),)
        target_set = frozenset(targets)
        if target_set in groups_by_targets:
            raise ReferenceConvergenceError(
                "exact Clay place-group declaration identities collide"
            )
        groups_by_targets[target_set] = group
    if (
        len(groups_by_targets) != len(declared_place_groups)
        or set(groups_by_targets)
        != {declared.targets for declared in declared_place_groups}  # type: ignore[attr-defined]
    ):
        raise ReferenceConvergenceError(
            "exact Clay place-group declarations cannot be mapped to LayoutObligations"
        )

    for declared in required_intents:
        family = _obligation_family(declared.kind)  # type: ignore[attr-defined]
        if family == "matching":
            pair = tuple(sorted(str(item) for item in declared.targets))  # type: ignore[attr-defined]
            declaration_ids = tuple(
                str(item) for item in declared.declaration_ids  # type: ignore[attr-defined]
            )
            if len(pair) != 2 or not declaration_ids:
                raise ReferenceConvergenceError(
                    "exact Clay declared matching must have two targets and declaration IDs"
                )
            signatures.append(("matching", pair, declaration_ids))  # type: ignore[arg-type]
        elif family == "place_group":
            group = groups_by_targets[declared.targets]  # type: ignore[attr-defined]
            members = tuple(str(member) for member in group.devices)
            if len(members) >= 2:
                pair = (members[0], members[1])
            elif len(members) == 1:
                pair = (members[0], str(group.group))
            else:
                pair = (
                    str(group.group),
                    f"row{group.row_min}_col{group.col}",
                )
            signatures.append(("place_group_realized", pair, ()))

    normalized = tuple(sorted(signatures))
    if len(set(normalized)) != len(normalized):
        raise ReferenceConvergenceError(
            "exact Clay required intent row identities are not unique"
        )
    return normalized


def _exact_unordered_pair(value: object, *, label: str) -> tuple[str, str]:
    try:
        pair = tuple(str(item) for item in value)  # type: ignore[union-attr]
    except TypeError as exc:
        raise ReferenceConvergenceError(f"{label} must be a pair") from exc
    if (
        len(pair) != 2
        or any(not item for item in pair)
        or pair[0] == pair[1]
    ):
        raise ReferenceConvergenceError(
            f"{label} must contain exactly two distinct members"
        )
    return tuple(sorted(pair))  # type: ignore[return-value]


def _generic_required_intent_row_signatures(
    parsed_intent: object,
    required_intents: tuple[object, ...],
) -> tuple[
    tuple[str, tuple[str, str], str | None, tuple[str, ...], float], ...
]:
    """Build exact family-specific identities without completeness subset rules."""

    from clay.layout.intent_completeness import _obligation_family
    from clay.layout.intent_obligations import (
        _DEFAULT_ABUT_TOL,
        _DEFAULT_ORDER_TOL,
    )

    signatures: list[
        tuple[str, tuple[str, str], str | None, tuple[str, ...], float]
    ] = []
    for budget in tuple(getattr(parsed_intent, "budgets", ()) or ()):
        signatures.append(
            (
                str(budget.kind),
                _exact_unordered_pair(
                    budget.pair, label="exact Clay parasitic-budget identity"
                ),
                None,
                (),
                float(budget.rel_tol),
            )
        )
    for budget in tuple(getattr(parsed_intent, "lde_budgets", ()) or ()):
        signatures.append(
            (
                f"lde:{budget.metric}",
                _exact_unordered_pair(
                    budget.pair, label="exact Clay LDE-budget identity"
                ),
                None,
                (),
                float(budget.rel_tol),
            )
        )
    for raw in tuple(getattr(parsed_intent, "order_constraints", ()) or ()):
        if type(raw) is not tuple or len(raw) != 3:
            raise ReferenceConvergenceError(
                "exact Clay order identity must contain a directed pair and relation"
            )
        a, b, relation = (str(item) for item in raw)
        if not a or not b or a == b or relation not in {"left_of", "above"}:
            raise ReferenceConvergenceError(
                "exact Clay order identity has invalid direction or relation"
            )
        signatures.append(
            ("order", (a, b), relation, (), float(_DEFAULT_ORDER_TOL))
        )
    for pair in tuple(getattr(parsed_intent, "abut_pairs", ()) or ()):
        signatures.append(
            (
                "abut",
                _exact_unordered_pair(pair, label="exact Clay abut identity"),
                None,
                (),
                float(_DEFAULT_ABUT_TOL),
            )
        )

    generic_declared = tuple(
        declared
        for declared in required_intents
        if _obligation_family(declared.kind)  # type: ignore[attr-defined]
        not in {"matching", "place_group"}
    )
    declared_base = {
        (
            str(declared.kind),  # type: ignore[attr-defined]
            tuple(sorted(str(item) for item in declared.targets)),  # type: ignore[attr-defined]
        )
        for declared in generic_declared
    }
    expected_base = {
        (kind, tuple(sorted(pair))) for kind, pair, _relation, _ids, _tol in signatures
    }
    if declared_base != expected_base:
        raise ReferenceConvergenceError(
            "exact Clay generic intent vocabulary lacks a closed family-specific identity"
        )
    identity_keys = tuple(
        (kind, pair, relation, declaration_ids)
        for kind, pair, relation, declaration_ids, _tolerance in signatures
    )
    if len(set(identity_keys)) != len(identity_keys):
        raise ReferenceConvergenceError(
            "exact Clay generic intent identities collide"
        )
    return tuple(sorted(signatures, key=repr))


def _require_authoritative_row_tolerance(
    item: IntentObligationMeasurement,
    expected: float,
) -> None:
    if float(item.rel_tol) != float(expected):
        raise ReferenceConvergenceError(
            "intent receipt rel_tol differs from exact Clay/Task4 authoritative tolerance"
        )


def _validate_required_intent_rows(
    parsed_intent: object,
    rows: tuple[IntentObligationMeasurement, ...],
) -> tuple[object, ...]:
    """Require a one-to-one map from all preserved runner rows to exact Clay."""

    from clay.layout.intent_completeness import _obligation_family

    required_intents = _required_declared_intents(parsed_intent)
    strict_expected = _strict_required_intent_row_signatures(
        parsed_intent, required_intents
    )
    strict_families = {"matching", "place_group"}
    strict_observed = tuple(
        sorted(
            (item.kind, item.pair, item.declaration_ids)
            for item in rows
            if _obligation_family(item.kind) in strict_families
        )
    )
    if strict_observed != strict_expected:
        raise ReferenceConvergenceError(
            "intent obligation rows do not map one-to-one to exact Clay required intents"
        )
    from clay.layout.intent_obligations import (
        _DEFAULT_CC_TOL,
        _DEFAULT_PLACE_GROUP_TOL,
    )

    for item in rows:
        family = _obligation_family(item.kind)
        if family == "matching":
            _require_authoritative_row_tolerance(item, _DEFAULT_CC_TOL)
        elif family == "place_group":
            _require_authoritative_row_tolerance(item, _DEFAULT_PLACE_GROUP_TOL)

    generic_expected = _generic_required_intent_row_signatures(
        parsed_intent, required_intents
    )
    generic_rows = tuple(
        item
        for item in rows
        if _obligation_family(item.kind) not in strict_families
    )
    generic_observed: list[
        tuple[str, tuple[str, str], str | None, tuple[str, ...], float]
    ] = []
    for item in generic_rows:
        if item.kind == "order":
            if item.pair[0] == item.pair[1]:
                raise ReferenceConvergenceError(
                    "intent order identity requires two distinct directed members"
                )
            pair = item.pair
        else:
            pair = _exact_unordered_pair(
                item.pair, label="intent generic obligation identity"
            )
        generic_observed.append(
            (
                item.kind,
                pair,
                item.intent_relation,
                item.declaration_ids,
                float(item.rel_tol),
            )
        )
    if tuple(sorted(generic_observed, key=repr)) != generic_expected:
        raise ReferenceConvergenceError(
            "intent generic obligation identity/threshold does not map one-to-one "
            "to exact Clay required intents"
        )
    return required_intents


def _runner_row_intent_relation(
    parsed_intent: object,
    value: Mapping[str, object],
) -> str | None:
    """Bind an order row to exact Clay when the legacy runner omitted relation."""

    supplied = value.get("intent_relation")
    if value.get("kind") != "order" or supplied is not None:
        return supplied  # type: ignore[return-value]
    raw_pair = value.get("pair", ())
    if type(raw_pair) not in {list, tuple} or len(raw_pair) != 2:
        raise ReferenceConvergenceError(
            "intent order measurement pair is invalid"
        )
    pair = tuple(str(item) for item in raw_pair)
    candidates = tuple(
        str(relation)
        for a, b, relation in tuple(
            getattr(parsed_intent, "order_constraints", ()) or ()
        )
        if (str(a), str(b)) == pair
    )
    if len(candidates) != 1:
        raise ReferenceConvergenceError(
            "intent order measurement cannot infer one exact authored relation"
        )
    return candidates[0]


def build_intent_measurement_receipt(
    *,
    circuit_id: str,
    topology_id: str,
    pdk_id: str,
    exact_gds_sha256: str,
    intent_source: str,
    intent_result: Mapping[str, object],
) -> IntentMeasurementReceipt:
    """Project an intent-step result into a runner-neutral measurement receipt."""

    if type(intent_source) is not str:
        raise ReferenceConvergenceError("intent source must be exact Clay text")
    try:
        from clay.layout.intent import parse_layout_intent
        parsed_intent = parse_layout_intent(intent_source)
        required_intents = _required_declared_intents(parsed_intent)
        required_inventory = _required_declaration_inventory(
            required_intents
        )
    except (TypeError, ValueError) as exc:
        raise ReferenceConvergenceError(
            f"exact Clay intent source is invalid: {exc}"
        ) from exc
    if not isinstance(intent_result, Mapping):
        raise ReferenceConvergenceError("intent step result must be an object")
    raw_rows = intent_result.get("obligations", [])
    if type(raw_rows) is not list:
        raise ReferenceConvergenceError("intent step obligations must be an array")
    rows: list[IntentObligationMeasurement] = []
    for index, value in enumerate(raw_rows):
        if type(value) is not dict:
            raise ReferenceConvergenceError(
                f"intent step obligation {index} must be an object"
            )
        status = value.get("status")
        verification_status = value.get("verification_status", status)
        if status is not None and verification_status != status:
            raise ReferenceConvergenceError(
                f"intent step obligation {index} has conflicting measurement statuses"
            )
        try:
            row = IntentObligationMeasurement(
                kind=value.get("kind"),  # type: ignore[arg-type]
                pair=tuple(value.get("pair", ())),  # type: ignore[arg-type]
                measured_imbalance=value.get("measured_imbalance"),  # type: ignore[arg-type]
                rel_tol=value.get("rel_tol"),  # type: ignore[arg-type]
                measurement_status=verification_status,  # type: ignore[arg-type]
                over_budget=value.get("over_budget"),  # type: ignore[arg-type]
                missing_nets=tuple(value.get("missing_nets", ())),  # type: ignore[arg-type]
                declaration_ids=tuple(value.get("declaration_ids", ())),  # type: ignore[arg-type]
                measurements=value.get("measurements", {}),  # type: ignore[arg-type]
                intent_relation=_runner_row_intent_relation(parsed_intent, value),
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, ReferenceConvergenceError):
                raise
            raise ReferenceConvergenceError(
                f"intent step obligation {index} is invalid: {exc}"
            ) from exc
        rows.append(row)
    ordered = tuple(
        sorted(
            rows,
            key=lambda item: (
                item.kind,
                item.pair,
                item.intent_relation or "",
                item.declaration_ids,
            ),
        )
    )
    observed_inventory = tuple(
        sorted(
            declaration
            for item in ordered
            for declaration in item.declaration_ids
        )
    )
    if observed_inventory != required_inventory:
        raise ReferenceConvergenceError(
            "intent obligation rows do not exactly cover the exact Clay declaration inventory"
        )
    _validate_required_intent_rows(parsed_intent, ordered)
    return IntentMeasurementReceipt(
        circuit_id=circuit_id,
        topology_id=topology_id,
        pdk_id=pdk_id,
        exact_gds_sha256=exact_gds_sha256,
        intent_digest=parsed_intent.digest,
        declaration_inventory=required_inventory,
        obligation_rows=ordered,
    )


def load_intent_measurement_receipt(
    path: str | Path,
    *,
    expected_receipt_sha256: str,
    expected_circuit_id: str,
    expected_topology_id: str,
    expected_pdk_id: str,
    expected_exact_gds_sha256: str,
) -> IntentMeasurementReceipt:
    """Load canonical receipt bytes and enforce their complete row identity."""

    source = Path(path)
    try:
        raw_bytes = source.read_bytes()
        payload = json.loads(
            raw_bytes.decode("utf-8"), object_pairs_hook=_strict_json_object
        )
    except ReferenceConvergenceError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReferenceConvergenceError(
            f"intent measurement receipt is missing or invalid: {source}"
        ) from exc
    canonical = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    if raw_bytes != canonical:
        raise ReferenceConvergenceError(
            "intent measurement receipt bytes are non-canonical"
        )
    _require_sha256(
        expected_receipt_sha256, label="expected intent measurement receipt"
    )
    if hashlib.sha256(raw_bytes).hexdigest() != expected_receipt_sha256:
        raise ReferenceConvergenceError(
            "intent measurement receipt SHA-256 differs from physical row locator"
        )
    receipt = intent_measurement_receipt_from_mapping(payload)
    expected = (
        expected_circuit_id,
        expected_topology_id,
        expected_pdk_id,
        expected_exact_gds_sha256,
    )
    actual = (
        receipt.circuit_id,
        receipt.topology_id,
        receipt.pdk_id,
        receipt.exact_gds_sha256,
    )
    if actual != expected:
        raise ReferenceConvergenceError(
            "intent measurement receipt circuit/topology/PDK/exact-GDS binding differs"
        )
    return receipt


def _validate_receipt_against_exact_clay_source(
    receipt: IntentMeasurementReceipt,
    *,
    physical_row: object,
    repo_root: str | Path,
) -> None:
    """Independently reparse the one physical-row Clay source authority."""

    source_locators = tuple(
        item
        for item in getattr(physical_row, "artifacts", ())
        if getattr(item, "logical_id", None) == "source"
    )
    if len(source_locators) != 1:
        raise ReferenceConvergenceError(
            "physical row must bind exactly one exact Clay source"
        )
    source_path = Path(repo_root).resolve() / source_locators[0].path
    try:
        source_text = source_path.read_text(encoding="utf-8")
        from clay.layout.intent import parse_layout_intent
        parsed = parse_layout_intent(source_text)
        required_intents = _required_declared_intents(parsed)
        inventory = _required_declaration_inventory(
            required_intents
        )
    except (OSError, UnicodeError, TypeError, ValueError) as exc:
        raise ReferenceConvergenceError(
            f"physical row exact Clay source is invalid: {exc}"
        ) from exc
    if receipt.intent_digest != parsed.digest:
        raise ReferenceConvergenceError(
            "intent receipt digest differs from the exact Clay source"
        )
    if receipt.declaration_inventory != inventory:
        raise ReferenceConvergenceError(
            "intent receipt declaration inventory differs from the exact Clay source"
        )
    _validate_required_intent_rows(parsed, receipt.obligation_rows)


@dataclass(frozen=True)
class DimensionVerdict:
    """One aggregate-owned verdict over runner-neutral measurement evidence."""

    dimension: str
    status: VerificationStatus
    evidence: tuple[ArtifactLocator, ...]
    measurements: Mapping[str, object]
    thresholds: Mapping[str, object]
    blocker_reason: str | None

    def __post_init__(self) -> None:
        if type(self.dimension) is not str or self.dimension not in REFERENCE_DIMENSIONS:
            raise ReferenceConvergenceError(
                f"reference convergence dimension is invalid: {self.dimension!r}"
            )
        try:
            status = VerificationStatus(self.status)
        except (TypeError, ValueError) as exc:
            raise ReferenceConvergenceError(
                f"{self.dimension} verification status is invalid"
            ) from exc
        object.__setattr__(self, "status", status)
        if type(self.evidence) is not tuple or any(
            not isinstance(item, ArtifactLocator) for item in self.evidence
        ):
            raise ReferenceConvergenceError(
                f"{self.dimension} evidence locators are invalid"
            )
        if not isinstance(self.measurements, Mapping) or not isinstance(
            self.thresholds, Mapping
        ):
            raise ReferenceConvergenceError(
                f"{self.dimension} measurements/thresholds are invalid"
            )
        object.__setattr__(self, "measurements", MappingProxyType(dict(self.measurements)))
        object.__setattr__(self, "thresholds", MappingProxyType(dict(self.thresholds)))
        if status is VerificationStatus.PASS:
            if self.blocker_reason is not None:
                raise ReferenceConvergenceError(
                    f"pass dimension {self.dimension} cannot carry a blocker"
                )
        elif type(self.blocker_reason) is not str or not self.blocker_reason.strip():
            raise ReferenceConvergenceError(
                f"non-pass dimension {self.dimension} requires a blocker reason"
            )

    def canonical_data(self) -> dict[str, object]:
        return {
            "dimension": self.dimension,
            "status": self.status.value,
            "evidence": [item.canonical_data() for item in self.evidence],
            "measurements": dict(self.measurements),
            "thresholds": dict(self.thresholds),
            "blocker_reason": self.blocker_reason,
        }


def _ordered_verdicts(
    verdicts: tuple[DimensionVerdict, ...],
) -> tuple[DimensionVerdict, ...]:
    if type(verdicts) is not tuple or tuple(
        verdict.dimension for verdict in verdicts
    ) != REFERENCE_DIMENSIONS:
        raise ReferenceConvergenceError(
            "dimension verdicts must contain the exact canonical seven-dimension order"
        )
    return verdicts


def achieved_maturity(verdicts: tuple[DimensionVerdict, ...]) -> Maturity:
    """Return the highest prefix policy whose every required status is pass."""

    ordered = _ordered_verdicts(verdicts)
    by_dimension = {item.dimension: item for item in ordered}
    achieved = Maturity.NONE
    for maturity in (
        Maturity.PHYSICAL_LEGAL,
        Maturity.REFERENCE_AWARE,
        Maturity.ELECTRICALLY_CHARACTERIZED,
        Maturity.ROBUST,
    ):
        if all(
            by_dimension[dimension].status is VerificationStatus.PASS
            for dimension in MATURITY_REQUIREMENTS[maturity]
        ):
            achieved = maturity
        else:
            break
    return achieved


def _dimension_verdict(
    dimension: str,
    status: VerificationStatus,
    *,
    evidence: tuple[ArtifactLocator, ...],
    measurements: Mapping[str, object],
    thresholds: Mapping[str, object],
    blocker: str,
) -> DimensionVerdict:
    return DimensionVerdict(
        dimension=dimension,
        status=status,
        evidence=evidence,
        measurements=measurements,
        thresholds=thresholds,
        blocker_reason=(None if status is VerificationStatus.PASS else blocker),
    )


def initial_dimension_verdicts(
    *,
    physical_row: object,
    reference_manifest: object,
    physical_ledger: ArtifactLocator,
    exact_gds: ArtifactLocator,
    reference_manifest_locator: ArtifactLocator,
    intent_measurement_receipt: IntentMeasurementReceipt,
    intent_measurement_locator: ArtifactLocator,
    repo_root: str | Path,
) -> tuple[DimensionVerdict, ...]:
    """Map stable Task1/2/4 facts into initial aggregate-owned verdicts.

    A physical PEX artifact is deliberately recorded only as a physical fact.  It
    cannot satisfy calibrated ``pex_confidence`` or any electrical dimension.
    """

    from clay.flow.portfolio_signoff import PortfolioSignoffRow
    from clay.flow.reference_evidence import ReferenceManifest

    if not isinstance(physical_row, PortfolioSignoffRow):
        raise ReferenceConvergenceError("initial dimensions require a typed v2 physical row")
    if not isinstance(reference_manifest, ReferenceManifest):
        raise ReferenceConvergenceError("initial dimensions require a typed reference manifest")
    if (
        physical_row.physical_status is not VerificationStatus.PASS
        or physical_row.physical_signed_off is not True
    ):
        raise ReferenceConvergenceError("initial dimensions require physical legality pass")
    if (
        physical_row.circuit_id != reference_manifest.circuit_id
        or physical_row.topology_id != reference_manifest.topology_id
        or physical_row.normalized_connectivity_sha256
        != reference_manifest.normalized_connectivity_sha256
    ):
        raise ReferenceConvergenceError("physical/reference identity differs")
    _validate_locator_shape(
        physical_ledger, logical_id="physical_ledger", label="physical ledger"
    )
    _validate_locator_shape(exact_gds, logical_id="gds", label="exact GDS")
    _validate_locator_shape(
        reference_manifest_locator,
        logical_id="reference_manifest",
        label="reference manifest",
    )
    _validate_locator_shape(
        intent_measurement_locator,
        logical_id="intent_measurement_receipt",
        label="intent measurement receipt",
    )
    physical_gds = tuple(
        item for item in physical_row.artifacts if item.logical_id == "gds"
    )
    physical_pex = tuple(
        item for item in physical_row.artifacts if item.logical_id == "pex"
    )
    if physical_gds != (exact_gds,) or len(physical_pex) != 1:
        raise ReferenceConvergenceError(
            "initial dimensions require unique exact GDS and physical PEX locators"
        )
    if (
        intent_measurement_receipt.circuit_id != physical_row.circuit_id
        or intent_measurement_receipt.topology_id != physical_row.topology_id
        or intent_measurement_receipt.pdk_id != physical_row.pdk_id
        or intent_measurement_receipt.exact_gds_sha256 != exact_gds.sha256
    ):
        raise ReferenceConvergenceError(
            "intent measurement receipt differs from its physical row/exact GDS"
        )

    topology_status = VerificationStatus(reference_manifest.scope.topology)
    intent_rows = intent_measurement_receipt.obligation_rows
    nonpass_intent = next(
        (
            item.measurement_status
            for item in intent_rows
            if item.measurement_status is not VerificationStatus.PASS
        ),
        None,
    )
    if not intent_rows:
        intent_status = VerificationStatus.UNMEASURABLE
    elif nonpass_intent is None:
        intent_status = VerificationStatus.PASS
    else:
        intent_status = nonpass_intent
    intent_counts = {
        status.value: sum(
            item.measurement_status is status for item in intent_rows
        )
        for status in VerificationStatus
    }
    intent_counts = {key: value for key, value in intent_counts.items() if value}

    from clay.flow.reference_quality import geometric_quality_verdict

    geometric_verdict = geometric_quality_verdict(
        physical_row=physical_row,
        reference_manifest=reference_manifest,
        exact_gds=exact_gds,
        reference_manifest_locator=reference_manifest_locator,
        repo_root=repo_root,
    )

    return (
        _dimension_verdict(
            "physical_legality",
            VerificationStatus.PASS,
            evidence=(physical_ledger, exact_gds),
            measurements={
                "physical_signed_off": True,
                "drc_violations": physical_row.drc_violations,
                "lvs_matched": physical_row.lvs_matched,
            },
            thresholds={"required_status": "pass"},
            blocker="physical artifact signoff is not pass",
        ),
        _dimension_verdict(
            "topology_fidelity",
            topology_status,
            evidence=(reference_manifest_locator,),
            measurements={
                "normalized_connectivity_sha256": (
                    reference_manifest.normalized_connectivity_sha256
                )
            },
            thresholds={"required_status": "pass"},
            blocker=f"registered reference topology status is {topology_status.value}",
        ),
        _dimension_verdict(
            "intent_coverage",
            intent_status,
            evidence=(intent_measurement_locator, exact_gds),
            measurements={
                "declaration_count": len(
                    intent_measurement_receipt.declaration_inventory
                ),
                "obligation_count": len(intent_rows),
                "measurement_status_counts": intent_counts,
            },
            thresholds={
                "required_status": "pass",
                "complete_declaration_coverage": True,
            },
            blocker=(
                "intent measurement receipt has no obligations"
                if not intent_rows
                else f"intent measurement status is {intent_status.value}"
            ),
        ),
        geometric_verdict,
        _dimension_verdict(
            "pex_confidence",
            VerificationStatus.UNMEASURABLE,
            evidence=(physical_pex[0], exact_gds),
            measurements={"physical_pex_artifact_present": True},
            thresholds={"calibrated_extractor_receipt_required": True},
            blocker="physical PEX artifact has no calibrated confidence receipt",
        ),
        _dimension_verdict(
            "electrical_function",
            VerificationStatus.UNMEASURABLE,
            evidence=(physical_pex[0], physical_ledger),
            measurements={"post_pex_result_receipt_present": False},
            thresholds={"digest_bound_post_pex_result_required": True},
            blocker="physical PEX artifact is not an electrical result receipt",
        ),
        _dimension_verdict(
            "robustness",
            VerificationStatus.UNMEASURABLE,
            evidence=(physical_ledger,),
            measurements={"robustness_result_receipt_present": False},
            thresholds={"complete_robustness_matrix_required": True},
            blocker="no digest-bound robustness result receipt",
        ),
    )


def _require_text(value: object, *, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ReferenceConvergenceError(f"{label} must be a non-empty string")
    return value


def _require_sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ReferenceConvergenceError(f"{label} must be a lowercase SHA-256")
    return value


def _validate_locator_shape(
    locator: ArtifactLocator, *, logical_id: str, label: str
) -> None:
    if not isinstance(locator, ArtifactLocator) or locator.logical_id != logical_id:
        raise ReferenceConvergenceError(f"{label} locator identity is invalid")
    _require_sha256(locator.sha256, label=f"{label} locator digest")
    if locator.required is not True or locator.artifact_type not in {"file", "manifest"}:
        raise ReferenceConvergenceError(f"{label} locator scope is invalid")
    if type(locator.path) is not str or not locator.path:
        raise ReferenceConvergenceError(f"{label} locator path is empty")
    path = PurePosixPath(locator.path)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != locator.path:
        raise ReferenceConvergenceError(
            f"{label} locator path must be canonical repository-relative"
        )
    prefix = ("signoff", "artifacts", "sha256", locator.sha256)
    if path.parts[:4] != prefix or len(path.parts) < 5:
        raise ReferenceConvergenceError(
            f"{label} locator must be canonical content-addressed evidence"
        )


def _require_canonical_json(value: object, *, label: str) -> object:
    """Require a deterministic JSON value without accepting NaN or custom types."""

    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        return json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise HumanGapVectorError(f"{label} is not canonical JSON") from exc


def _payload_contains_png(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(
            "png" in str(key).casefold() or _payload_contains_png(child)
            for key, child in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_payload_contains_png(child) for child in value)
    return type(value) is str and ".png" in value.casefold()


def _freeze_canonical_json(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_canonical_json(child) for key, child in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_canonical_json(child) for child in value)
    return value


def _thaw_canonical_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _thaw_canonical_json(child) for key, child in value.items()
        }
    if isinstance(value, tuple):
        return [_thaw_canonical_json(child) for child in value]
    return value


def _require_human_gap_identifier(value: object, *, label: str) -> str:
    if type(value) is not str or _HUMAN_GAP_IDENTIFIER.fullmatch(value) is None:
        raise HumanGapVectorError(f"{label} must be a canonical identifier")
    return value


def _validate_spatial_advisory_payload(
    measurements: Mapping[str, object],
    thresholds: Mapping[str, object],
    *,
    authority: HumanGapAuthority,
) -> None:
    if authority is HumanGapAuthority.SPATIAL_ADVISORY:
        allowed_measurements = _SPATIAL_ADVISORY_MEASUREMENT_KEYS
        allowed_thresholds = _SPATIAL_ADVISORY_THRESHOLD_KEYS
    elif authority is HumanGapAuthority.COMPLETE_REFERENCE:
        allowed_measurements = _COMPLETE_REFERENCE_MEASUREMENT_KEYS
        allowed_thresholds = _COMPLETE_REFERENCE_THRESHOLD_KEYS
    else:  # pragma: no cover - private contract
        raise HumanGapVectorError("reference-derived authority is invalid")
    unknown_measurements = set(measurements) - allowed_measurements
    unknown_thresholds = set(thresholds) - allowed_thresholds
    if unknown_measurements or unknown_thresholds:
        raise HumanGapVectorError(
            f"{authority.value} payload exceeds its closed coordinate-free schema: "
            f"measurement={sorted(unknown_measurements)}, "
            f"threshold={sorted(unknown_thresholds)}"
        )
    if "reference_available" in measurements and type(
        measurements["reference_available"]
    ) is not bool:
        raise HumanGapVectorError(
            "spatial advisory reference_available must be boolean"
        )
    ordinal_maps: dict[str, Mapping[str, object]] = {}
    for key in ("region_tier_ordinals", "region_column_ordinals"):
        if key not in measurements:
            continue
        value = measurements[key]
        if not isinstance(value, Mapping) or any(
            _HUMAN_GAP_IDENTIFIER.fullmatch(str(region)) is None
            or type(ordinal) is not int
            or ordinal < 0
            for region, ordinal in value.items()
        ):
            raise HumanGapVectorError(
                f"spatial advisory {key} must contain canonical region ordinals"
            )
        if not value:
            raise HumanGapVectorError(
                f"spatial advisory {key} cannot be empty"
            )
        ordinals = tuple(int(item) for item in value.values())
        if key == "region_column_ordinals":
            compact = sorted(ordinals) == list(range(len(ordinals)))
        else:
            compact = set(ordinals) == set(range(max(ordinals) + 1))
        if not compact:
            raise HumanGapVectorError(
                f"spatial advisory {key} must use compact ordinals"
            )
        ordinal_maps[key] = value
    if len(ordinal_maps) == 2 and set(
        ordinal_maps["region_tier_ordinals"]
    ) != set(ordinal_maps["region_column_ordinals"]):
        raise HumanGapVectorError(
            "spatial advisory tier/column ordinal region sets differ"
        )
    for key in ("left_of_relations", "above_relations", "adjacency_relations"):
        if key not in measurements:
            continue
        value = measurements[key]
        if not isinstance(value, (list, tuple)) or any(
            not isinstance(pair, (list, tuple))
            or len(pair) != 2
            or any(
                type(region) is not str
                or _HUMAN_GAP_IDENTIFIER.fullmatch(region) is None
                for region in pair
            )
            for pair in value
        ):
            raise HumanGapVectorError(
                f"spatial advisory {key} must contain canonical region pairs"
            )
        if key == "left_of_relations":
            columns = ordinal_maps.get("region_column_ordinals")
            if columns is None or any(
                pair[0] not in columns
                or pair[1] not in columns
                or columns[pair[0]] >= columns[pair[1]]
                for pair in value
            ):
                raise HumanGapVectorError(
                    "spatial advisory left_of relation contradicts column ordinals"
                )
        elif key == "above_relations":
            tiers = ordinal_maps.get("region_tier_ordinals")
            if tiers is None or any(
                pair[0] not in tiers
                or pair[1] not in tiers
                or tiers[pair[0]] <= tiers[pair[1]]
                for pair in value
            ):
                raise HumanGapVectorError(
                    "spatial advisory above relation contradicts tier ordinals"
                )
        else:
            tiers = ordinal_maps.get("region_tier_ordinals")
            columns = ordinal_maps.get("region_column_ordinals")
            if tiers is None or columns is None or any(
                pair[0] not in tiers
                or pair[1] not in tiers
                or pair[0] not in columns
                or pair[1] not in columns
                or abs(tiers[pair[0]] - tiers[pair[1]]) > 1
                or abs(columns[pair[0]] - columns[pair[1]]) > 1
                for pair in value
            ):
                raise HumanGapVectorError(
                    "spatial advisory adjacency contradicts region ordinals"
                )
        measurements[key] = sorted([list(pair) for pair in value])
    for key in ("banked_regions", "wide_block_regions"):
        if key not in measurements:
            continue
        value = measurements[key]
        if not isinstance(value, (list, tuple)) or any(
            type(region) is not str
            or _HUMAN_GAP_IDENTIFIER.fullmatch(region) is None
            for region in value
        ):
            raise HumanGapVectorError(
                f"spatial advisory {key} must contain canonical regions"
            )
        measurements[key] = sorted(value)
    for key in ("required_relation_count", "satisfied_relation_count"):
        if key in measurements and (
            type(measurements[key]) is not int or measurements[key] < 0
        ):
            raise HumanGapVectorError(
                f"spatial advisory {key} must be a nonnegative integer"
            )
    numeric_keys = (
        allowed_measurements
        - {
            "reference_available",
            "region_tier_ordinals",
            "region_column_ordinals",
            "left_of_relations",
            "above_relations",
            "adjacency_relations",
            "banked_regions",
            "wide_block_regions",
            "required_relation_count",
            "satisfied_relation_count",
            "member_conservation",
        }
    )
    for key in numeric_keys:
        if key in measurements and (
            type(measurements[key]) not in {int, float}
            or not math.isfinite(float(measurements[key]))
        ):
            raise HumanGapVectorError(
                f"spatial advisory {key} must be a finite number"
            )
    if "member_conservation" in measurements and type(
        measurements["member_conservation"]
    ) is not bool:
        raise HumanGapVectorError(
            "complete reference member_conservation must be boolean"
        )
    if "advisory_only" in thresholds and thresholds["advisory_only"] is not True:
        raise HumanGapVectorError(
            "spatial advisory advisory_only threshold must be true"
        )
    for key, value in thresholds.items():
        if key == "advisory_only":
            continue
        if key == "require_member_conservation":
            if type(value) is not bool:
                raise HumanGapVectorError(
                    "complete reference require_member_conservation must be boolean"
                )
            continue
        if type(value) not in {int, float} or not math.isfinite(float(value)):
            raise HumanGapVectorError(
                f"spatial advisory threshold {key} must be a finite number"
            )


@dataclass(frozen=True)
class HumanGapBinding:
    """Identity shared by every measurement in one exact-GDS quality vector."""

    topology_id: str
    source_sha256: str
    reference_scope_sha256: str
    generated_gds_sha256: str

    def __post_init__(self) -> None:
        _require_human_gap_identifier(
            self.topology_id, label="human-gap topology ID"
        )
        _require_sha256(self.source_sha256, label="human-gap source digest")
        _require_sha256(
            self.reference_scope_sha256,
            label="human-gap reference-scope digest",
        )
        _require_sha256(
            self.generated_gds_sha256,
            label="human-gap generated GDS digest",
        )

    def canonical_data(self) -> dict[str, str]:
        return {
            "topology_id": self.topology_id,
            "source_sha256": self.source_sha256,
            "reference_scope_sha256": self.reference_scope_sha256,
            "generated_gds_sha256": self.generated_gds_sha256,
        }


@dataclass(frozen=True)
class HumanGapObservation:
    """One field-level observation with explicit authority and byte binding."""

    dimension: str
    name: str
    authority: HumanGapAuthority
    level: HumanGapLevel
    status: VerificationStatus
    binding: HumanGapBinding
    evidence: tuple[ArtifactLocator, ...]
    measurements: Mapping[str, object]
    thresholds: Mapping[str, object]
    blocker_reason: str | None

    def __post_init__(self) -> None:
        if self.dimension not in HUMAN_GAP_DIMENSIONS:
            raise HumanGapVectorError(
                f"human-gap dimension is invalid: {self.dimension!r}"
            )
        _require_human_gap_identifier(
            self.name, label=f"{self.dimension} observation name"
        )
        try:
            authority = HumanGapAuthority(self.authority)
            level = HumanGapLevel(self.level)
            status = VerificationStatus(self.status)
        except (TypeError, ValueError) as exc:
            raise HumanGapVectorError(
                f"{self.dimension}/{self.name} authority, level, or status is invalid"
            ) from exc
        object.__setattr__(self, "authority", authority)
        object.__setattr__(self, "level", level)
        object.__setattr__(self, "status", status)
        if status not in {
            VerificationStatus.PASS,
            VerificationStatus.FAIL,
            VerificationStatus.UNMEASURABLE,
            VerificationStatus.ADVISORY,
        }:
            raise HumanGapVectorError(
                f"{self.dimension}/{self.name} status is not a closed P2 observation"
            )
        if not isinstance(self.binding, HumanGapBinding):
            raise HumanGapVectorError(
                f"{self.dimension}/{self.name} binding is not typed"
            )
        if authority is HumanGapAuthority.SPATIAL_ADVISORY and status not in {
            VerificationStatus.ADVISORY,
            VerificationStatus.UNMEASURABLE,
        }:
            raise HumanGapVectorError(
                "spatial advisory cannot issue a binding pass or fail"
            )
        if type(self.evidence) is not tuple or any(
            not isinstance(item, ArtifactLocator) for item in self.evidence
        ):
            raise HumanGapVectorError(
                f"{self.dimension}/{self.name} evidence is invalid"
            )
        if any(PurePosixPath(item.path).suffix.casefold() == ".png" for item in self.evidence):
            raise HumanGapVectorError("PNG cannot be human-gap evidence")
        for item in self.evidence:
            try:
                _require_text(item.logical_id, label="human-gap evidence logical ID")
                _validate_locator_shape(
                    item,
                    logical_id=item.logical_id,
                    label=f"human-gap {item.logical_id} evidence",
                )
            except ReferenceConvergenceError as exc:
                raise HumanGapVectorError(str(exc)) from exc
        ordered_evidence = tuple(
            sorted(
                self.evidence,
                key=lambda item: (
                    item.logical_id,
                    item.sha256,
                    item.path,
                    item.artifact_type,
                    item.required,
                ),
            )
        )
        object.__setattr__(self, "evidence", ordered_evidence)
        evidence_by_id = {
            logical_id: tuple(
                item for item in ordered_evidence if item.logical_id == logical_id
            )
            for logical_id in {item.logical_id for item in ordered_evidence}
        }
        required_authority_evidence = _HUMAN_GAP_AUTHORITY_EVIDENCE[authority]
        if any(
            len(evidence_by_id.get(logical_id, ())) != 1
            for logical_id in required_authority_evidence
        ):
            raise HumanGapVectorError(
                f"{authority.value} authority evidence is incomplete or ambiguous"
            )
        exact_gds = tuple(item for item in self.evidence if item.logical_id == "gds")
        if len(exact_gds) != 1:
            raise HumanGapVectorError(
                f"{self.dimension}/{self.name} requires one exact GDS locator"
            )
        try:
            _validate_locator_shape(
                exact_gds[0], logical_id="gds", label="human-gap exact GDS"
            )
        except ReferenceConvergenceError as exc:
            raise HumanGapVectorError(str(exc)) from exc
        if exact_gds[0].sha256 != self.binding.generated_gds_sha256:
            raise HumanGapVectorError(
                f"{self.dimension}/{self.name} generated GDS binding differs"
            )
        if not isinstance(self.measurements, Mapping) or not isinstance(
            self.thresholds, Mapping
        ):
            raise HumanGapVectorError(
                f"{self.dimension}/{self.name} measurements/thresholds are invalid"
            )
        measurements = _require_canonical_json(
            dict(self.measurements),
            label=f"{self.dimension}/{self.name} measurements",
        )
        thresholds = _require_canonical_json(
            dict(self.thresholds),
            label=f"{self.dimension}/{self.name} thresholds",
        )
        if not isinstance(measurements, dict) or not isinstance(thresholds, dict):
            raise HumanGapVectorError(
                f"{self.dimension}/{self.name} payloads must be JSON objects"
            )
        if _payload_contains_png(measurements) or _payload_contains_png(thresholds):
            raise HumanGapVectorError("PNG cannot be human-gap measurement authority")
        if authority in {
            HumanGapAuthority.COMPLETE_REFERENCE,
            HumanGapAuthority.SPATIAL_ADVISORY,
        }:
            _validate_spatial_advisory_payload(
                measurements,
                thresholds,
                authority=authority,
            )
        object.__setattr__(self, "measurements", _freeze_canonical_json(measurements))
        object.__setattr__(self, "thresholds", _freeze_canonical_json(thresholds))
        if status is VerificationStatus.PASS:
            if self.blocker_reason is not None:
                raise HumanGapVectorError(
                    f"pass observation {self.dimension}/{self.name} cannot carry a blocker"
                )
        elif type(self.blocker_reason) is not str or not self.blocker_reason.strip():
            raise HumanGapVectorError(
                f"non-pass observation {self.dimension}/{self.name} requires a blocker"
            )
        if _payload_contains_png(self.blocker_reason):
            raise HumanGapVectorError("PNG cannot appear in a human-gap blocker")

    def canonical_data(self) -> dict[str, object]:
        return {
            "dimension": self.dimension,
            "name": self.name,
            "authority": self.authority.value,
            "level": self.level.value,
            "status": self.status.value,
            "binding": self.binding.canonical_data(),
            "evidence": [item.canonical_data() for item in self.evidence],
            "measurements": _thaw_canonical_json(self.measurements),
            "thresholds": _thaw_canonical_json(self.thresholds),
            "blocker_reason": self.blocker_reason,
        }


@dataclass(frozen=True)
class HumanGapDimension:
    """One non-averaged P2 axis containing independently authorized fields."""

    dimension: str
    observations: tuple[HumanGapObservation, ...]

    def __post_init__(self) -> None:
        if self.dimension not in HUMAN_GAP_DIMENSIONS:
            raise HumanGapVectorError(
                f"human-gap dimension is invalid: {self.dimension!r}"
            )
        if type(self.observations) is not tuple or not self.observations:
            raise HumanGapVectorError(
                f"human-gap dimension {self.dimension} has no observations"
            )
        if any(
            not isinstance(item, HumanGapObservation)
            or item.dimension != self.dimension
            for item in self.observations
        ):
            raise HumanGapVectorError(
                f"human-gap dimension {self.dimension} contains cross-axis observations"
            )
        names = tuple(item.name for item in self.observations)
        if len(set(names)) != len(names):
            raise HumanGapVectorError(
                f"human-gap dimension {self.dimension} repeats an observation name"
            )
        object.__setattr__(
            self,
            "observations",
            tuple(sorted(self.observations, key=lambda item: item.name)),
        )

    def canonical_data(self) -> dict[str, object]:
        return {
            "dimension": self.dimension,
            "observations": [item.canonical_data() for item in self.observations],
        }


@dataclass(frozen=True)
class HumanGapVector:
    """Seven exact-GDS quality axes with no aggregate score or hidden averaging."""

    pdk_id: str
    binding: HumanGapBinding
    source: ArtifactLocator
    reference_scope: ArtifactLocator
    exact_gds: ArtifactLocator
    dimensions: tuple[HumanGapDimension, ...]

    def __post_init__(self) -> None:
        _require_human_gap_identifier(self.pdk_id, label="human-gap PDK ID")
        if not isinstance(self.binding, HumanGapBinding):
            raise HumanGapVectorError("human-gap vector binding is not typed")
        if any(
            PurePosixPath(locator.path).suffix.casefold() == ".png"
            for locator in (self.source, self.reference_scope, self.exact_gds)
        ):
            raise HumanGapVectorError("PNG cannot be a human-gap vector artifact")
        try:
            _validate_locator_shape(
                self.source, logical_id="source", label="human-gap source"
            )
            _validate_locator_shape(
                self.reference_scope,
                logical_id="reference_scope",
                label="human-gap reference scope",
            )
            _validate_locator_shape(
                self.exact_gds, logical_id="gds", label="human-gap exact GDS"
            )
        except ReferenceConvergenceError as exc:
            raise HumanGapVectorError(str(exc)) from exc
        if self.source.sha256 != self.binding.source_sha256:
            raise HumanGapVectorError("human-gap source locator differs from its binding")
        if self.reference_scope.sha256 != self.binding.reference_scope_sha256:
            raise HumanGapVectorError(
                "human-gap reference scope locator differs from its binding"
            )
        if self.exact_gds.sha256 != self.binding.generated_gds_sha256:
            raise HumanGapVectorError(
                "human-gap exact GDS locator differs from its binding"
            )
        if type(self.dimensions) is not tuple or tuple(
            item.dimension if isinstance(item, HumanGapDimension) else None
            for item in self.dimensions
        ) != HUMAN_GAP_DIMENSIONS:
            raise HumanGapVectorError(
                "human-gap dimensions must use the exact canonical seven-axis order"
            )
        for dimension in self.dimensions:
            for observation in dimension.observations:
                if observation.binding != self.binding:
                    raise HumanGapVectorError(
                        f"{dimension.dimension}/{observation.name} binding differs from vector"
                    )
                if observation.authority in {
                    HumanGapAuthority.COMPLETE_REFERENCE,
                    HumanGapAuthority.SPATIAL_ADVISORY,
                }:
                    scopes = tuple(
                        item
                        for item in observation.evidence
                        if item.logical_id == "reference_scope"
                    )
                    if scopes != (self.reference_scope,):
                        raise HumanGapVectorError(
                            f"{dimension.dimension}/{observation.name} reference scope differs from vector"
                        )
                gds = tuple(
                    item for item in observation.evidence if item.logical_id == "gds"
                )
                if gds != (self.exact_gds,):
                    raise HumanGapVectorError(
                        f"{dimension.dimension}/{observation.name} exact GDS differs from vector"
                    )

    def canonical_data(self) -> dict[str, object]:
        return {
            "schema": HUMAN_GAP_VECTOR_SCHEMA,
            "pdk_id": self.pdk_id,
            "binding": self.binding.canonical_data(),
            "source": self.source.canonical_data(),
            "reference_scope": self.reference_scope.canonical_data(),
            "exact_gds": self.exact_gds.canonical_data(),
            "dimensions": [item.canonical_data() for item in self.dimensions],
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.canonical_data(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )


@dataclass(frozen=True)
class ReferenceConvergenceRow:
    """One circuit/PDK claim bound to physical and reference evidence bytes."""

    circuit_id: str
    pdk_id: str
    topology_id: str
    normalized_connectivity_sha256: str
    reference_topology_sha256: str
    physical_evidence_commit: str
    physical_ledger: ArtifactLocator
    physical_row_sha256: str
    exact_gds: ArtifactLocator
    reference_manifest: ArtifactLocator
    dimensions: tuple[DimensionVerdict, ...]
    achieved_maturity: Maturity

    def __post_init__(self) -> None:
        from clay.flow.pdk_matrix import FOUR_PDK_IDS
        from clay.flow.portfolio_signoff import PORTFOLIO_CIRCUIT_IDS

        if self.circuit_id not in PORTFOLIO_CIRCUIT_IDS:
            raise ReferenceConvergenceError(
                f"reference convergence circuit is invalid: {self.circuit_id!r}"
            )
        if self.pdk_id not in FOUR_PDK_IDS:
            raise ReferenceConvergenceError(
                f"reference convergence PDK is invalid: {self.pdk_id!r}"
            )
        _require_text(self.topology_id, label="reference convergence topology ID")
        _require_sha256(
            self.normalized_connectivity_sha256,
            label="reference convergence connectivity digest",
        )
        _require_sha256(
            self.reference_topology_sha256,
            label="reference topology digest",
        )
        if (
            type(self.physical_evidence_commit) is not str
            or _COMMIT.fullmatch(self.physical_evidence_commit) is None
        ):
            raise ReferenceConvergenceError(
                "physical evidence commit must be a lowercase full object ID"
            )
        _require_sha256(self.physical_row_sha256, label="physical row digest")
        _validate_locator_shape(
            self.physical_ledger,
            logical_id="physical_ledger",
            label="physical ledger",
        )
        _validate_locator_shape(self.exact_gds, logical_id="gds", label="exact GDS")
        _validate_locator_shape(
            self.reference_manifest,
            logical_id="reference_manifest",
            label="reference manifest",
        )
        ordered = _ordered_verdicts(self.dimensions)
        try:
            maturity = Maturity(self.achieved_maturity)
        except (TypeError, ValueError) as exc:
            raise ReferenceConvergenceError("achieved maturity is invalid") from exc
        object.__setattr__(self, "achieved_maturity", maturity)
        if achieved_maturity(ordered) is not maturity:
            raise ReferenceConvergenceError(
                "stored achieved maturity differs from derived maturity"
            )

    def canonical_data(self) -> dict[str, object]:
        return {
            "schema": REFERENCE_CONVERGENCE_ROW_SCHEMA,
            "circuit_id": self.circuit_id,
            "pdk_id": self.pdk_id,
            "topology_id": self.topology_id,
            "normalized_connectivity_sha256": self.normalized_connectivity_sha256,
            "reference_topology_sha256": self.reference_topology_sha256,
            "physical_evidence_commit": self.physical_evidence_commit,
            "physical_ledger": self.physical_ledger.canonical_data(),
            "physical_row_sha256": self.physical_row_sha256,
            "exact_gds": self.exact_gds.canonical_data(),
            "reference_manifest": self.reference_manifest.canonical_data(),
            "dimensions": [item.canonical_data() for item in self.dimensions],
            "achieved_maturity": self.achieved_maturity.value,
        }


def validate_reference_convergence_rows(
    rows: tuple[ReferenceConvergenceRow, ...],
) -> tuple[ReferenceConvergenceRow, ...]:
    """Require the exact three-circuit x four-PDK canonical row sequence."""

    from clay.flow.pdk_matrix import FOUR_PDK_IDS
    from clay.flow.portfolio_signoff import PORTFOLIO_CIRCUIT_IDS

    if type(rows) is not tuple or any(
        not isinstance(item, ReferenceConvergenceRow) for item in rows
    ):
        raise ReferenceConvergenceError(
            "convergence matrix rows must be a typed tuple"
        )
    expected = tuple(
        (circuit, pdk)
        for circuit in PORTFOLIO_CIRCUIT_IDS
        for pdk in FOUR_PDK_IDS
    )
    actual = tuple((item.circuit_id, item.pdk_id) for item in rows)
    if actual != expected or len(set(actual)) != len(expected):
        raise ReferenceConvergenceError(
            "convergence matrix must equal the canonical twelve-row Cartesian order"
        )
    return rows


def blocker_dimensions(
    row: ReferenceConvergenceRow, requested: Maturity
) -> tuple[str, ...]:
    """Derive requested-policy blockers without storing a request-dependent field."""

    if not isinstance(row, ReferenceConvergenceRow):
        raise ReferenceConvergenceError("maturity query row is not typed")
    try:
        policy = Maturity(requested)
        required = MATURITY_REQUIREMENTS[policy]
    except (TypeError, ValueError, KeyError) as exc:
        raise ReferenceConvergenceError(
            "requested maturity must be one of the four publication policies"
        ) from exc
    by_dimension = {item.dimension: item for item in row.dimensions}
    return tuple(
        dimension
        for dimension in required
        if by_dimension[dimension].status is not VerificationStatus.PASS
    )


def meets_maturity(row: ReferenceConvergenceRow, requested: Maturity) -> bool:
    return not blocker_dimensions(row, requested)


def _artifact_locator_from_mapping(value: object, *, label: str) -> ArtifactLocator:
    raw = _strict_fields(
        value,
        expected={"logical_id", "path", "sha256", "artifact_type", "required"},
        label=label,
    )
    try:
        return ArtifactLocator(**raw)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ReferenceConvergenceError(f"{label} is invalid: {exc}") from exc


def _dimension_verdict_from_mapping(value: object) -> DimensionVerdict:
    raw = _strict_fields(
        value,
        expected={
            "dimension",
            "status",
            "evidence",
            "measurements",
            "thresholds",
            "blocker_reason",
        },
        label="dimension verdict",
    )
    if type(raw["evidence"]) is not list:
        raise ReferenceConvergenceError("dimension verdict evidence must be an array")
    return DimensionVerdict(
        dimension=raw["dimension"],  # type: ignore[arg-type]
        status=raw["status"],  # type: ignore[arg-type]
        evidence=tuple(
            _artifact_locator_from_mapping(item, label="dimension evidence locator")
            for item in raw["evidence"]
        ),
        measurements=raw["measurements"],  # type: ignore[arg-type]
        thresholds=raw["thresholds"],  # type: ignore[arg-type]
        blocker_reason=raw["blocker_reason"],  # type: ignore[arg-type]
    )


def reference_convergence_row_from_mapping(value: object) -> ReferenceConvergenceRow:
    raw = _strict_fields(
        value,
        expected={
            "schema",
            "circuit_id",
            "pdk_id",
            "topology_id",
            "normalized_connectivity_sha256",
            "reference_topology_sha256",
            "physical_evidence_commit",
            "physical_ledger",
            "physical_row_sha256",
            "exact_gds",
            "reference_manifest",
            "dimensions",
            "achieved_maturity",
        },
        label="reference convergence row",
    )
    if raw["schema"] != REFERENCE_CONVERGENCE_ROW_SCHEMA:
        raise ReferenceConvergenceError("reference convergence row schema is unsupported")
    if type(raw["dimensions"]) is not list:
        raise ReferenceConvergenceError("reference convergence dimensions must be an array")
    return ReferenceConvergenceRow(
        circuit_id=raw["circuit_id"],  # type: ignore[arg-type]
        pdk_id=raw["pdk_id"],  # type: ignore[arg-type]
        topology_id=raw["topology_id"],  # type: ignore[arg-type]
        normalized_connectivity_sha256=raw["normalized_connectivity_sha256"],  # type: ignore[arg-type]
        reference_topology_sha256=raw["reference_topology_sha256"],  # type: ignore[arg-type]
        physical_evidence_commit=raw["physical_evidence_commit"],  # type: ignore[arg-type]
        physical_ledger=_artifact_locator_from_mapping(
            raw["physical_ledger"], label="physical ledger locator"
        ),
        physical_row_sha256=raw["physical_row_sha256"],  # type: ignore[arg-type]
        exact_gds=_artifact_locator_from_mapping(
            raw["exact_gds"], label="exact GDS locator"
        ),
        reference_manifest=_artifact_locator_from_mapping(
            raw["reference_manifest"], label="reference manifest locator"
        ),
        dimensions=tuple(
            _dimension_verdict_from_mapping(item) for item in raw["dimensions"]
        ),
        achieved_maturity=raw["achieved_maturity"],  # type: ignore[arg-type]
    )


def _maturity_counts(rows: tuple[ReferenceConvergenceRow, ...]) -> dict[str, int]:
    return {
        maturity.value: sum(item.achieved_maturity is maturity for item in rows)
        for maturity in Maturity
        if maturity is not Maturity.NONE
        and any(item.achieved_maturity is maturity for item in rows)
    }


def _validate_ledger_bindings(
    rows: tuple[ReferenceConvergenceRow, ...],
    *,
    physical_rows: Mapping[tuple[str, str], object],
    physical_evidence_commit: str,
    physical_ledger_path: str | Path,
    reference_manifests: Mapping[str, object],
    reference_manifest_paths: Mapping[str, str | Path],
    topology_registry: ArtifactLocator,
    topology_registry_path: str | Path,
    repo_root: str | Path,
) -> None:
    from clay.flow.pdk_matrix import FOUR_PDK_IDS
    from clay.flow.portfolio_signoff import PORTFOLIO_CIRCUIT_IDS
    from clay.flow.signoff_artifacts import SignoffArtifactError, verify_artifact_locator

    expected_keys = {
        (circuit, pdk)
        for circuit in PORTFOLIO_CIRCUIT_IDS
        for pdk in FOUR_PDK_IDS
    }
    if set(physical_rows) != expected_keys:
        raise ReferenceConvergenceError(
            "physical binding contexts do not cover the exact twelve-row matrix"
        )
    circuits = set(PORTFOLIO_CIRCUIT_IDS)
    if set(reference_manifests) != circuits or set(reference_manifest_paths) != circuits:
        raise ReferenceConvergenceError(
            "reference manifest binding contexts do not cover the three circuits"
        )
    _validate_locator_shape(
        topology_registry,
        logical_id="topology_registry",
        label="topology registry",
    )
    try:
        verify_artifact_locator(repo_root, topology_registry)
    except SignoffArtifactError as exc:
        raise ReferenceConvergenceError(
            f"topology registry packaged evidence is invalid: {exc}"
        ) from exc
    if _file_sha256(
        topology_registry_path, label="registered topology registry"
    ) != topology_registry.sha256:
        raise ReferenceConvergenceError(
            "topology registry locator differs from actual registered bytes"
        )
    physical_locators = {
        (
            item.physical_ledger.path,
            item.physical_ledger.sha256,
            item.physical_ledger.artifact_type,
            item.physical_ledger.required,
        )
        for item in rows
    }
    if len(physical_locators) != 1:
        raise ReferenceConvergenceError(
            "convergence rows do not bind one common physical ledger"
        )
    for row in rows:
        validate_reference_convergence_row(
            row,
            physical_row=physical_rows[(row.circuit_id, row.pdk_id)],
            physical_evidence_commit=physical_evidence_commit,
            physical_ledger_path=physical_ledger_path,
            reference_manifest=reference_manifests[row.circuit_id],
            reference_manifest_path=reference_manifest_paths[row.circuit_id],
            repo_root=repo_root,
        )


def publish_reference_convergence_ledger(
    destination: str | Path,
    rows: tuple[ReferenceConvergenceRow, ...],
    *,
    physical_rows: Mapping[tuple[str, str], object],
    physical_evidence_commit: str,
    physical_ledger_path: str | Path,
    reference_manifests: Mapping[str, object],
    reference_manifest_paths: Mapping[str, str | Path],
    topology_registry: ArtifactLocator,
    topology_registry_path: str | Path,
    repo_root: str | Path,
) -> dict[str, object]:
    """Validate every binding, then atomically publish the semantic ledger."""

    from clay.flow.pdk_matrix import FOUR_PDK_IDS
    from clay.flow.portfolio_signoff import PORTFOLIO_CIRCUIT_IDS

    ordered = validate_reference_convergence_rows(rows)
    _validate_ledger_bindings(
        ordered,
        physical_rows=physical_rows,
        physical_evidence_commit=physical_evidence_commit,
        physical_ledger_path=physical_ledger_path,
        reference_manifests=reference_manifests,
        reference_manifest_paths=reference_manifest_paths,
        topology_registry=topology_registry,
        topology_registry_path=topology_registry_path,
        repo_root=repo_root,
    )
    payload: dict[str, object] = {
        "schema": REFERENCE_CONVERGENCE_LEDGER_SCHEMA,
        "claim_scope": "reference_convergence",
        "circuit_order": list(PORTFOLIO_CIRCUIT_IDS),
        "pdk_order": list(FOUR_PDK_IDS),
        "frozen_inputs": _frozen_input_reference(repo_root),
        "physical_ledger": ordered[0].physical_ledger.canonical_data(),
        "topology_registry": topology_registry.canonical_data(),
        "maturity_counts": _maturity_counts(ordered),
        "signoffs": [item.canonical_data() for item in ordered],
    }
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return payload


def verify_reference_convergence_ledger(
    source: str | Path,
    *,
    physical_rows: Mapping[tuple[str, str], object],
    physical_evidence_commit: str,
    physical_ledger_path: str | Path,
    reference_manifests: Mapping[str, object],
    reference_manifest_paths: Mapping[str, str | Path],
    topology_registry_path: str | Path,
    repo_root: str | Path,
) -> dict[str, object]:
    """Read and re-hash a ledger without invoking any design or simulation tool."""

    from clay.flow.pdk_matrix import FOUR_PDK_IDS
    from clay.flow.portfolio_signoff import PORTFOLIO_CIRCUIT_IDS

    path = Path(source)
    try:
        raw_bytes = path.read_bytes()
        payload = json.loads(
            raw_bytes.decode("utf-8"), object_pairs_hook=_strict_json_object
        )
    except ReferenceConvergenceError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReferenceConvergenceError(
            f"reference convergence ledger is missing or invalid: {path}"
        ) from exc
    canonical = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    if raw_bytes != canonical:
        raise ReferenceConvergenceError(
            "reference convergence ledger bytes are non-canonical"
        )
    raw = _strict_fields(
        payload,
        expected={
            "schema",
            "claim_scope",
            "circuit_order",
            "pdk_order",
            "frozen_inputs",
            "physical_ledger",
            "topology_registry",
            "maturity_counts",
            "signoffs",
        },
        label="reference convergence ledger",
    )
    if (
        raw["schema"] != REFERENCE_CONVERGENCE_LEDGER_SCHEMA
        or raw["claim_scope"] != "reference_convergence"
        or raw["circuit_order"] != list(PORTFOLIO_CIRCUIT_IDS)
        or raw["pdk_order"] != list(FOUR_PDK_IDS)
        or type(raw["signoffs"]) is not list
    ):
        raise ReferenceConvergenceError(
            "reference convergence ledger identity/order is invalid"
        )
    rows = tuple(
        reference_convergence_row_from_mapping(item)
        for item in raw["signoffs"]  # type: ignore[union-attr]
    )
    validate_reference_convergence_rows(rows)
    if raw["maturity_counts"] != _maturity_counts(rows):
        raise ReferenceConvergenceError(
            "reference convergence maturity counts are not derived"
        )
    expected_frozen_inputs = _frozen_input_reference(repo_root)
    stored_frozen_inputs = _strict_fields(
        raw["frozen_inputs"],
        expected={"schema", "path", "sha256"},
        label="frozen input reference",
    )
    if dict(stored_frozen_inputs) != expected_frozen_inputs:
        raise ReferenceConvergenceError(
            "frozen input reference differs from the current frozen authority"
        )
    physical_locator = _artifact_locator_from_mapping(
        raw["physical_ledger"], label="physical ledger locator"
    )
    if physical_locator.canonical_data() != rows[0].physical_ledger.canonical_data():
        raise ReferenceConvergenceError(
            "top-level physical ledger differs from convergence rows"
        )
    topology_locator = _artifact_locator_from_mapping(
        raw["topology_registry"], label="topology registry locator"
    )
    _validate_ledger_bindings(
        rows,
        physical_rows=physical_rows,
        physical_evidence_commit=physical_evidence_commit,
        physical_ledger_path=physical_ledger_path,
        reference_manifests=reference_manifests,
        reference_manifest_paths=reference_manifest_paths,
        topology_registry=topology_locator,
        topology_registry_path=topology_registry_path,
        repo_root=repo_root,
    )
    return payload


def _file_sha256(path: str | Path, *, label: str) -> str:
    source = Path(path)
    try:
        raw = source.read_bytes()
    except OSError as exc:
        raise ReferenceConvergenceError(f"{label} is missing: {source}") from exc
    if not raw:
        raise ReferenceConvergenceError(f"{label} is empty: {source}")
    return hashlib.sha256(raw).hexdigest()


def _frozen_input_reference(repo_root: str | Path) -> dict[str, str]:
    """Validate the frozen authority and return its receipt-level byte binding."""

    root = Path(repo_root).resolve()
    frozen_path = root.joinpath(*PurePosixPath(FROZEN_INPUTS_RELATIVE).parts)
    try:
        raw_bytes = frozen_path.read_bytes()
        payload = json.loads(
            raw_bytes.decode("utf-8"), object_pairs_hook=_strict_json_object
        )
    except ReferenceConvergenceError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReferenceConvergenceError(
            f"frozen input authority is missing or invalid: {frozen_path}"
        ) from exc
    canonical = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    if raw_bytes != canonical:
        raise ReferenceConvergenceError("frozen input authority bytes are non-canonical")
    frozen = _strict_fields(
        payload,
        expected={
            "schema",
            "code_baseline",
            "design_clay",
            "reference_gds",
            "oa_spatial_anchor_gds",
        },
        label="frozen input authority",
    )
    if frozen["schema"] != FROZEN_INPUTS_SCHEMA:
        raise ReferenceConvergenceError("frozen input authority schema is invalid")
    baseline = _strict_fields(
        frozen["code_baseline"],
        expected={"arc", "commit", "files"},
        label="frozen input code baseline",
    )
    if (
        baseline["arc"] != FROZEN_CODE_BASELINE_ARC
        or baseline["commit"] != FROZEN_CODE_BASELINE_COMMIT
    ):
        raise ReferenceConvergenceError(
            "frozen input code baseline differs from this evidence arc"
        )
    raw_code_files = baseline["files"]
    if type(raw_code_files) is not list:
        raise ReferenceConvergenceError(
            "frozen input code baseline files are invalid"
        )
    code_files = []
    for raw in raw_code_files:
        item = _strict_fields(
            raw,
            expected={"path", "sha256"},
            label="frozen input code baseline file",
        )
        relative = item["path"]
        digest = item["sha256"]
        if type(relative) is not str or type(digest) is not str:
            raise ReferenceConvergenceError(
                "frozen input code baseline file is invalid"
            )
        code_files.append((relative, digest))
    code_paths = tuple(relative for relative, _digest in code_files)
    if code_paths != tuple(sorted(set(code_paths))) or not set(
        FROZEN_CODE_BASELINE_REQUIRED_PATHS
    ).issubset(code_paths):
        raise ReferenceConvergenceError(
            "frozen input code baseline paths are unordered, duplicated, or incomplete"
        )
    for relative, digest in code_files:
        pure = PurePosixPath(relative)
        if (
            _SHA256.fullmatch(digest) is None
            or pure.is_absolute()
            or ".." in pure.parts
            or pure.as_posix() != relative
        ):
            raise ReferenceConvergenceError(
                "frozen input code baseline binding is invalid"
            )
        code_path = root.joinpath(*pure.parts).resolve()
        try:
            code_path.relative_to(root)
        except ValueError as exc:
            raise ReferenceConvergenceError(
                "frozen input code baseline resolves outside the repository"
            ) from exc
        if _file_sha256(code_path, label="frozen input code baseline") != digest:
            raise ReferenceConvergenceError(
                f"frozen input code baseline changed: {relative}"
            )
    expected_inputs = {
        "design_clay": (
            FROZEN_DESIGN_CLAY_RELATIVE,
            FROZEN_DESIGN_CLAY_SHA256,
        ),
        "reference_gds": (
            FROZEN_REFERENCE_GDS_RELATIVE,
            FROZEN_REFERENCE_GDS_SHA256,
        ),
        "oa_spatial_anchor_gds": (
            FROZEN_OA_SPATIAL_ANCHOR_GDS_RELATIVE,
            FROZEN_OA_SPATIAL_ANCHOR_GDS_SHA256,
        ),
    }
    for field in ("design_clay", "reference_gds", "oa_spatial_anchor_gds"):
        item = _strict_fields(
            frozen[field],
            expected={"path", "sha256"},
            label=f"frozen input {field}",
        )
        relative = item["path"]
        digest = item["sha256"]
        if type(relative) is not str:
            raise ReferenceConvergenceError(f"frozen input {field} path is invalid")
        expected_path, expected_digest = expected_inputs[field]
        if relative != expected_path:
            raise ReferenceConvergenceError(
                f"frozen input {field} path differs from this evidence arc"
            )
        pure = PurePosixPath(relative)
        if (
            not relative
            or pure.is_absolute()
            or ".." in pure.parts
            or pure.as_posix() != relative
        ):
            raise ReferenceConvergenceError(
                f"frozen input {field} path must be canonical repository-relative"
            )
        if type(digest) is not str or _SHA256.fullmatch(digest) is None:
            raise ReferenceConvergenceError(f"frozen input {field} SHA-256 is invalid")
        if digest != expected_digest:
            raise ReferenceConvergenceError(
                f"frozen input {field} digest differs from this evidence arc"
            )
        input_path = root.joinpath(*pure.parts).resolve()
        try:
            input_path.relative_to(root)
        except ValueError as exc:
            raise ReferenceConvergenceError(
                f"frozen input {field} resolves outside the repository"
            ) from exc
        if _file_sha256(input_path, label=f"frozen input {field}") != digest:
            raise ReferenceConvergenceError(
                f"frozen input {field} digest differs from current bytes"
            )
    return {
        "schema": FROZEN_INPUT_REFERENCE_SCHEMA,
        "path": FROZEN_INPUTS_RELATIVE,
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
    }


def validate_reference_convergence_row(
    row: ReferenceConvergenceRow,
    *,
    physical_row: object,
    physical_evidence_commit: str,
    physical_ledger_path: str | Path,
    reference_manifest: object,
    reference_manifest_path: str | Path,
    repo_root: str | Path,
) -> ReferenceConvergenceRow:
    """Re-hash and cross-bind one aggregate row to verified upstream objects."""

    from clay.flow.portfolio_signoff import (
        PortfolioSignoffError,
        PortfolioSignoffRow,
        canonical_portfolio_row_sha256,
        validate_portfolio_row_artifacts,
    )
    from clay.flow.reference_evidence import ReferenceManifest
    from clay.flow.signoff_artifacts import (
        SignoffArtifactError,
        verify_artifact_locator,
    )

    if not isinstance(row, ReferenceConvergenceRow):
        raise ReferenceConvergenceError("reference convergence row is not typed")
    if not isinstance(physical_row, PortfolioSignoffRow):
        raise ReferenceConvergenceError("physical row is not a verified v2 row")
    if not isinstance(reference_manifest, ReferenceManifest):
        raise ReferenceConvergenceError("reference manifest is not typed")
    try:
        validate_portfolio_row_artifacts(
            physical_row,
            current_commit=physical_row.commit,
            repo_root=repo_root,
        )
        for locator in (
            row.physical_ledger,
            row.exact_gds,
            row.reference_manifest,
            *(item for verdict in row.dimensions for item in verdict.evidence),
        ):
            verify_artifact_locator(repo_root, locator)
    except (PortfolioSignoffError, SignoffArtifactError, OSError, ValueError) as exc:
        raise ReferenceConvergenceError(f"bound evidence is invalid: {exc}") from exc

    if (
        row.circuit_id != physical_row.circuit_id
        or row.pdk_id != physical_row.pdk_id
        or row.topology_id != physical_row.topology_id
        or row.normalized_connectivity_sha256
        != physical_row.normalized_connectivity_sha256
    ):
        raise ReferenceConvergenceError(
            "reference convergence identity differs from its physical row"
        )
    if row.physical_evidence_commit != physical_evidence_commit:
        raise ReferenceConvergenceError(
            "reference convergence physical evidence commit differs"
        )
    if canonical_portfolio_row_sha256(physical_row) != row.physical_row_sha256:
        raise ReferenceConvergenceError("physical row canonical SHA-256 differs")

    physical_gds = tuple(
        locator for locator in physical_row.artifacts if locator.logical_id == "gds"
    )
    if len(physical_gds) != 1:
        raise ReferenceConvergenceError(
            "physical row must bind one unique exact GDS locator"
        )
    if physical_gds[0].canonical_data() != row.exact_gds.canonical_data():
        raise ReferenceConvergenceError("exact GDS differs from the physical row")

    physical_path = Path(physical_ledger_path)
    if _file_sha256(physical_path, label="physical ledger") != row.physical_ledger.sha256:
        raise ReferenceConvergenceError("physical ledger locator digest differs")
    try:
        physical_bytes = physical_path.read_bytes()
        physical_payload = json.loads(physical_bytes.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReferenceConvergenceError("physical ledger is not valid JSON") from exc
    canonical_physical = (
        json.dumps(physical_payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )
    if physical_bytes != canonical_physical:
        raise ReferenceConvergenceError("physical ledger bytes are non-canonical")
    if (
        type(physical_payload) is not dict
        or physical_payload.get("schema")
        != "clay.three-circuit-four-pdk-signoff/v2"
        or type(physical_payload.get("signoffs")) is not list
    ):
        raise ReferenceConvergenceError("physical ledger is not verified v2 data")
    matching_rows = [
        item
        for item in physical_payload["signoffs"]
        if type(item) is dict
        and item.get("circuit_id") == row.circuit_id
        and item.get("pdk_id") == row.pdk_id
    ]
    if matching_rows != [physical_row.canonical_data()]:
        raise ReferenceConvergenceError(
            "physical ledger does not contain the exact unique physical row"
        )

    if (
        row.circuit_id != reference_manifest.circuit_id
        or row.topology_id != reference_manifest.topology_id
        or row.normalized_connectivity_sha256
        != reference_manifest.normalized_connectivity_sha256
        or row.reference_topology_sha256 != reference_manifest.topology_sha256
    ):
        raise ReferenceConvergenceError(
            "reference convergence identity differs from its registered manifest"
        )
    manifest_sha = _file_sha256(
        reference_manifest_path, label="registered reference manifest"
    )
    if manifest_sha != row.reference_manifest.sha256:
        raise ReferenceConvergenceError(
            "reference manifest locator differs from actual registered manifest bytes"
        )

    intent_locators = tuple(
        locator
        for locator in physical_row.artifacts
        if locator.logical_id == "intent_measurement_receipt"
    )
    if len(intent_locators) != 1:
        raise ReferenceConvergenceError(
            "physical row must bind exactly one intent measurement receipt"
        )
    intent_locator = intent_locators[0]
    receipt = load_intent_measurement_receipt(
        Path(repo_root).resolve() / intent_locator.path,
        expected_receipt_sha256=intent_locator.sha256,
        expected_circuit_id=physical_row.circuit_id,
        expected_topology_id=physical_row.topology_id,
        expected_pdk_id=physical_row.pdk_id,
        expected_exact_gds_sha256=row.exact_gds.sha256,
    )
    _validate_receipt_against_exact_clay_source(
        receipt, physical_row=physical_row, repo_root=repo_root
    )
    expected_dimensions = initial_dimension_verdicts(
        physical_row=physical_row,
        reference_manifest=reference_manifest,
        physical_ledger=row.physical_ledger,
        exact_gds=row.exact_gds,
        reference_manifest_locator=row.reference_manifest,
        intent_measurement_receipt=receipt,
        intent_measurement_locator=intent_locator,
        repo_root=repo_root,
    )
    if [item.canonical_data() for item in row.dimensions] != [
        item.canonical_data() for item in expected_dimensions
    ]:
        raise ReferenceConvergenceError(
            "stored dimensions differ from re-evaluated bound measurements"
        )
    # Task 3 integration TODO: re-evaluate aggregate-authorized exact-GDS
    # geometric receipts here once that schema is stable.  A physical exact
    # artifact alone must never promote PEX or electrical dimensions.
    return row


__all__ = [
    "DimensionVerdict",
    "HUMAN_GAP_DIMENSIONS",
    "HUMAN_GAP_VECTOR_SCHEMA",
    "HumanGapAuthority",
    "HumanGapBinding",
    "HumanGapDimension",
    "HumanGapLevel",
    "HumanGapObservation",
    "HumanGapVector",
    "HumanGapVectorError",
    "INTENT_MEASUREMENT_RECEIPT_SCHEMA",
    "IntentMeasurementReceipt",
    "IntentObligationMeasurement",
    "MATURITY_REQUIREMENTS",
    "REFERENCE_DIMENSIONS",
    "Maturity",
    "ReferenceConvergenceError",
    "ReferenceConvergenceRow",
    "REFERENCE_CONVERGENCE_ROW_SCHEMA",
    "REFERENCE_CONVERGENCE_LEDGER_SCHEMA",
    "achieved_maturity",
    "blocker_dimensions",
    "build_intent_measurement_receipt",
    "intent_measurement_receipt_from_mapping",
    "initial_dimension_verdicts",
    "load_intent_measurement_receipt",
    "meets_maturity",
    "publish_reference_convergence_ledger",
    "reference_convergence_row_from_mapping",
    "validate_runner_neutral_measurement_payload",
    "validate_reference_convergence_row",
    "validate_reference_convergence_rows",
    "verify_reference_convergence_ledger",
]
