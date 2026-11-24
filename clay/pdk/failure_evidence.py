"""Typed, digest-bound PDK failure knowledge for physical-intent search.

The library records failed trajectories without turning them into circuit
recipes.  Predicates are deliberately small and geometry-only.  An observation
may become rejecting only after positive and negative regressions exist and the
deck plus implementation identities still match the current evaluator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Literal, Mapping, Sequence


_SHA256 = re.compile(r"[0-9a-f]{64}")
_OPERATORS = frozenset({"eq", "ne", "lt", "le", "gt", "ge"})
_GEOMETRY_FEATURES = frozenset(
    {
        "bbox_height_nm",
        "bbox_width_nm",
        "body_well_compatible",
        "corridor_capacity_nm",
        "corridor_demand_nm",
        "manufacturing_grid_remainder_nm",
        "rail_width_nm",
        "ring_axis_clearance_nm",
        "via_cut_size_nm",
        "via_enclosure_margin_nm",
        "via_landing_height_nm",
        "via_landing_width_nm",
        "well_tie_present",
        "whitespace_nm2",
    }
)
_FORBIDDEN_SELECTOR_TOKENS = frozenset(
    {"circuit", "cell", "net", "device", "instance", "group", "name"}
)


def _canonical_sha256(value: str, *, label: str) -> str:
    digest = str(value).strip().casefold()
    if _SHA256.fullmatch(digest) is None:
        raise ValueError(f"{label} must be one canonical SHA-256")
    return digest


def _canonical_digest(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class GeometryPredicate:
    """One conservative comparison over an approved geometry fact."""

    feature: str
    operator: Literal["eq", "ne", "lt", "le", "gt", "ge"]
    value: int | float | bool

    def __post_init__(self) -> None:
        feature = str(self.feature).strip().casefold()
        tokens = set(filter(None, re.split(r"[^a-z0-9]+", feature)))
        if feature not in _GEOMETRY_FEATURES or tokens & _FORBIDDEN_SELECTOR_TOKENS:
            raise ValueError(
                "PDK failure predicates must be geometry-only approved features; "
                f"got {self.feature!r}"
            )
        operator = str(self.operator).strip().casefold()
        if operator not in _OPERATORS:
            raise ValueError(
                f"unsupported geometry predicate operator {self.operator!r}"
            )
        value = self.value
        if not isinstance(value, (int, float, bool)):
            raise TypeError("geometry predicate value must be numeric or boolean")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("geometry predicate value must be finite")
        object.__setattr__(self, "feature", feature)
        object.__setattr__(self, "operator", operator)

    def matches(self, geometry: Mapping[str, object]) -> tuple[bool, str]:
        if self.feature not in geometry:
            return False, f"missing geometry feature {self.feature!r}"
        actual = geometry[self.feature]
        if not isinstance(actual, (int, float, bool)):
            return False, f"geometry feature {self.feature!r} is non-numeric"
        if isinstance(actual, float) and not math.isfinite(actual):
            return False, f"geometry feature {self.feature!r} is non-finite"
        operations = {
            "eq": lambda left, right: left == right,
            "ne": lambda left, right: left != right,
            "lt": lambda left, right: left < right,
            "le": lambda left, right: left <= right,
            "gt": lambda left, right: left > right,
            "ge": lambda left, right: left >= right,
        }
        try:
            matched = bool(operations[self.operator](actual, self.value))
        except TypeError:
            return False, f"geometry feature {self.feature!r} has incompatible type"
        return matched, "predicate matched" if matched else "predicate is false"


FailureRuleStatus = Literal["observed", "active", "retired"]


@dataclass(frozen=True)
class PdkFailureRule:
    rule_id: str
    pdk_id: str
    deck_digest: str
    implementation_digest: str
    stage: str
    geometry_predicate: GeometryPredicate
    failure_signature: str
    evidence_artifact_digest: str
    repair_class: str
    positive_regressions: tuple[str, ...]
    negative_counterexamples: tuple[str, ...]
    status: FailureRuleStatus = "observed"

    def __post_init__(self) -> None:
        for field_name in (
            "rule_id",
            "pdk_id",
            "stage",
            "failure_signature",
            "repair_class",
        ):
            value = str(getattr(self, field_name)).strip()
            if not value:
                raise ValueError(f"{field_name} must be non-empty")
            object.__setattr__(self, field_name, value)
        if not isinstance(self.geometry_predicate, GeometryPredicate):
            raise TypeError("geometry_predicate must be GeometryPredicate")
        if self.status not in {"observed", "active", "retired"}:
            raise ValueError("failure rule status must be observed, active, or retired")
        object.__setattr__(
            self,
            "deck_digest",
            _canonical_sha256(self.deck_digest, label="deck_digest"),
        )
        object.__setattr__(
            self,
            "implementation_digest",
            _canonical_sha256(
                self.implementation_digest, label="implementation_digest"
            ),
        )
        object.__setattr__(
            self,
            "evidence_artifact_digest",
            _canonical_sha256(
                self.evidence_artifact_digest, label="evidence_artifact_digest"
            ),
        )
        positives = tuple(
            sorted(
                {
                    _canonical_sha256(value, label="positive regression")
                    for value in self.positive_regressions
                }
            )
        )
        negatives = tuple(
            sorted(
                {
                    _canonical_sha256(value, label="negative counterexample")
                    for value in self.negative_counterexamples
                }
            )
        )
        if self.status == "active" and (not positives or not negatives):
            raise ValueError(
                "active failure rule requires positive and negative regressions"
            )
        object.__setattr__(self, "positive_regressions", positives)
        object.__setattr__(self, "negative_counterexamples", negatives)

    def canonical_data(self) -> dict[str, object]:
        return {
            "schema": "clay.pdk-failure-rule/v1",
            "rule_id": self.rule_id,
            "pdk_id": self.pdk_id,
            "deck_digest": self.deck_digest,
            "implementation_digest": self.implementation_digest,
            "stage": self.stage,
            "geometry_predicate": asdict(self.geometry_predicate),
            "failure_signature": self.failure_signature,
            "evidence_artifact_digest": self.evidence_artifact_digest,
            "repair_class": self.repair_class,
            "positive_regressions": list(self.positive_regressions),
            "negative_counterexamples": list(self.negative_counterexamples),
            "status": self.status,
        }

    @property
    def sha256(self) -> str:
        return _canonical_digest(self.canonical_data())


FailureEvidenceOutcome = Literal[
    "match",
    "miss",
    "observed",
    "retired",
    "stale_deck",
    "stale_implementation",
    "pdk_mismatch",
]


@dataclass(frozen=True)
class PdkFailureEvidence:
    rule_id: str
    rule_sha256: str
    pdk_id: str
    outcome: FailureEvidenceOutcome
    reject: bool
    reason: str

    def canonical_data(self) -> dict[str, object]:
        return {
            "schema": "clay.pdk-failure-evidence/v1",
            "rule_id": self.rule_id,
            "rule_sha256": self.rule_sha256,
            "pdk_id": self.pdk_id,
            "outcome": self.outcome,
            "reject": self.reject,
            "reason": self.reason,
        }

    @property
    def sha256(self) -> str:
        return _canonical_digest(self.canonical_data())


def evaluate_pdk_failure_rules(
    rules: Sequence[PdkFailureRule],
    *,
    pdk_id: str,
    deck_digest: str,
    implementation_digest: str,
    geometry: Mapping[str, object],
) -> tuple[PdkFailureEvidence, ...]:
    """Evaluate without repair; only a current active match may reject."""

    current_deck = _canonical_sha256(deck_digest, label="current deck digest")
    current_implementation = _canonical_sha256(
        implementation_digest, label="current implementation digest"
    )
    evidence: list[PdkFailureEvidence] = []
    for rule in sorted(rules, key=lambda item: (item.rule_id, item.sha256)):
        outcome: FailureEvidenceOutcome
        reject = False
        if rule.pdk_id != str(pdk_id):
            outcome, reason = "pdk_mismatch", "rule belongs to a different PDK"
        elif rule.deck_digest != current_deck:
            outcome, reason = "stale_deck", "rule deck digest is stale"
        elif rule.implementation_digest != current_implementation:
            outcome, reason = (
                "stale_implementation",
                "rule implementation digest is stale",
            )
        elif rule.status == "observed":
            outcome, reason = "observed", "rule is observation-only"
        elif rule.status == "retired":
            outcome, reason = "retired", "rule is retired"
        else:
            matched, reason = rule.geometry_predicate.matches(geometry)
            outcome = "match" if matched else "miss"
            reject = matched
        evidence.append(
            PdkFailureEvidence(
                rule_id=rule.rule_id,
                rule_sha256=rule.sha256,
                pdk_id=rule.pdk_id,
                outcome=outcome,
                reject=reject,
                reason=reason,
            )
        )
    return tuple(evidence)


def _digest_files(paths: Sequence[Path], *, fallback: Mapping[str, object]) -> str:
    digest = hashlib.sha256()
    measured = False
    for path in sorted({Path(path).resolve() for path in paths}, key=str):
        if not path.is_file():
            continue
        measured = True
        label = path.name.encode("utf-8")
        digest.update(len(label).to_bytes(8, "big"))
        digest.update(label)
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest() if measured else _canonical_digest(fallback)


def current_failure_rule_context(adapter: Any) -> tuple[str, str]:
    """Return current deck and shared-materializer implementation identities."""

    decks: list[Path] = []
    try:
        decks.extend(
            Path(path)
            for path in (adapter.klayout_decks() or {}).values()
            if path is not None
        )
    except Exception:  # unavailable proprietary/generated deck: bind parsed facts
        decks = []
    deck_fallback: dict[str, object] = {
        "pdk_id": str(adapter.pdk_id),
        "drc_rules": {},
        "layermap": {},
    }
    try:
        deck_fallback["drc_rules"] = adapter.drc_rules()
    except Exception:
        pass
    try:
        deck_fallback["layermap"] = adapter.layermap()
    except Exception:
        pass
    deck_digest = _digest_files(decks, fallback=deck_fallback)

    clay_root = Path(__file__).resolve().parents[1]
    implementation_digest = _digest_files(
        (
            Path(__file__),
            clay_root / "layout" / "device_cell.py",
            clay_root / "layout" / "guard_ring.py",
            clay_root / "layout" / "route" / "emit_gds.py",
            clay_root / "layout" / "route" / "engine_gridless.py",
        ),
        fallback={"implementation": "unavailable"},
    )
    return deck_digest, implementation_digest


def _fixture_digest(
    pdk_id: str, rule_id: str, label: str, geometry: Mapping[str, object]
) -> str:
    return _canonical_digest(
        {
            "schema": "clay.pdk-failure-regression-fixture/v1",
            "pdk_id": pdk_id,
            "rule_id": rule_id,
            "label": label,
            "geometry": dict(geometry),
        }
    )


def validated_failure_rules(adapter: Any) -> tuple[PdkFailureRule, ...]:
    """PDK-scoped generic rules backed by paired geometry regressions."""

    pdk_id = str(adapter.pdk_id)
    deck_digest, implementation_digest = current_failure_rule_context(adapter)
    specifications = (
        (
            "via_landing_enclosure",
            "route",
            GeometryPredicate("via_enclosure_margin_nm", "lt", 0),
            "VIA_LANDING_ENCLOSURE",
            "via_selector",
            {"via_enclosure_margin_nm": -1},
            {"via_enclosure_margin_nm": 0},
        ),
        (
            "body_well_compatibility",
            "materialize",
            GeometryPredicate("body_well_compatible", "eq", False),
            "BODY_WELL_TIE_MISMATCH",
            "body_tie_materializer",
            {"body_well_compatible": False},
            {"body_well_compatible": True},
        ),
        (
            "manufacturing_grid",
            "emit",
            GeometryPredicate("manufacturing_grid_remainder_nm", "gt", 0),
            "MANUFACTURING_GRID_MISALIGNMENT",
            "exact_geometry_materializer",
            {"manufacturing_grid_remainder_nm": 1},
            {"manufacturing_grid_remainder_nm": 0},
        ),
    )
    rules = []
    for (
        stem,
        stage,
        predicate,
        signature,
        repair_class,
        positive,
        negative,
    ) in specifications:
        rule_id = f"{pdk_id}.{stem}.v1"
        positive_digest = _fixture_digest(pdk_id, rule_id, "positive", positive)
        negative_digest = _fixture_digest(pdk_id, rule_id, "negative", negative)
        rules.append(
            PdkFailureRule(
                rule_id=rule_id,
                pdk_id=pdk_id,
                deck_digest=deck_digest,
                implementation_digest=implementation_digest,
                stage=stage,
                geometry_predicate=predicate,
                failure_signature=signature,
                evidence_artifact_digest=_canonical_digest(
                    {
                        "positive": positive_digest,
                        "negative": negative_digest,
                        "source": "typed_regression_pair",
                    }
                ),
                repair_class=repair_class,
                positive_regressions=(positive_digest,),
                negative_counterexamples=(negative_digest,),
                status="active",
            )
        )
    return tuple(rules)


__all__ = [
    "GeometryPredicate",
    "PdkFailureEvidence",
    "PdkFailureRule",
    "current_failure_rule_context",
    "evaluate_pdk_failure_rules",
    "validated_failure_rules",
]
