"""Circuit-neutral preparation policy for compact physical-intent search.

The module ranks materialized target-PDK evidence.  It does not own a circuit,
PDK, placer, router, or signoff oracle.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Literal, Mapping

from clay.flow.optimized_clay import (
    declared_compact_candidate,
    require_resolved_compact_intent,
)
from clay.layout.compact_intent import CompactIntentCandidate
from clay.layout.compact_intent_domain import compact_structure_fingerprint
from clay.pdk.failure_evidence import (
    PdkFailureEvidence,
    PdkFailureRule,
    current_failure_rule_context,
    evaluate_pdk_failure_rules,
)
from clay.parser import parse_clay_file

from .intent_domain import (
    build_compact_intent_domain,
    compact_intent_domain_spec,
    recombine_compact_intent_candidates,
    reconstruct_compact_intent_candidate,
)
from .physical_intent import (
    PhysicalIntentInfeasible,
    PhysicalIntentCheapResult,
    PhysicalIntentFunctionalGroupBBox,
    PhysicalIntentProbeBudget,
    PhysicalIntentProbeResult,
    deduplicate_physical_intent_candidates,
    evaluate_compact_intent_candidate,
    evaluate_compact_intent_probe,
    load_physical_intent_cache,
    load_physical_intent_probe_cache,
    physical_intent_probe_budget,
    physical_intent_probe_rank,
)
from .physical_search_target import (
    PhysicalSearchAdmission,
    PhysicalSearchTarget,
    evaluate_physical_search_admission,
)


CheapOutcome = Literal["pass", "infeasible", "unmeasurable"]
CheapPhysicalEvaluator = Callable[..., PhysicalIntentCheapResult]
ProbePhysicalEvaluator = Callable[..., PhysicalIntentProbeResult]


class _SearchInputChanged(ValueError):
    pass


_MAX_SEARCH_COUNT = 2**31 - 1
_MAX_SEARCH_MEASURE = 1.0e18
_MAX_SEARCH_CANDIDATES = 4096
_MAX_CHEAP_COORDINATE_NM = 10**15
_CHEAP_METRIC_KEYS = frozenset(
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
_BOUNDED_OUTCOME_KEYS = frozenset(
    {
        "schema",
        "input_sha256",
        "status",
        "outcome",
        "candidate",
        "candidate_sha256",
        "error_type",
        "error",
        "signoff_authority",
    }
)


def _search_count(value: object, *, label: str) -> int:
    if type(value) is not int or not 0 <= value <= _MAX_SEARCH_COUNT:
        raise ValueError(
            f"{label} must be an exact bounded non-negative integer"
        )
    return value


def _search_measure(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a bounded finite non-negative number")
    try:
        measured = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{label} must be a bounded finite non-negative number"
        ) from exc
    if not math.isfinite(measured) or not 0.0 <= measured <= _MAX_SEARCH_MEASURE:
        raise ValueError(f"{label} must be a bounded finite non-negative number")
    return measured


@dataclass(frozen=True)
class PhysicalSearchVector:
    """Closed lexicographic vector for exact physical candidate evidence."""

    hard_gate_failures: int
    intent_failures: int
    reference_structure_gap: float
    route_resource_overflow: int
    bbox_lower_bound_gap: float
    wire_length: float
    via_count: int
    uncalibrated_parasitic_observation: float

    def __post_init__(self) -> None:
        for field in (
            "hard_gate_failures",
            "intent_failures",
            "route_resource_overflow",
            "via_count",
        ):
            object.__setattr__(
                self,
                field,
                _search_count(getattr(self, field), label=field),
            )
        for field in (
            "reference_structure_gap",
            "bbox_lower_bound_gap",
            "wire_length",
            "uncalibrated_parasitic_observation",
        ):
            object.__setattr__(
                self,
                field,
                _search_measure(getattr(self, field), label=field),
            )

    def rank_key(self) -> tuple[int | float, ...]:
        return (
            self.hard_gate_failures,
            self.intent_failures,
            self.reference_structure_gap,
            self.route_resource_overflow,
            self.bbox_lower_bound_gap,
            self.wire_length,
            self.via_count,
            self.uncalibrated_parasitic_observation,
        )

    def canonical_data(self) -> dict[str, object]:
        return {
            "hard_gate_failures": self.hard_gate_failures,
            "intent_failures": self.intent_failures,
            "reference_structure_gap": self.reference_structure_gap,
            "route_resource_overflow": self.route_resource_overflow,
            "bbox_lower_bound_gap": self.bbox_lower_bound_gap,
            "wire_length": self.wire_length,
            "via_count": self.via_count,
            "uncalibrated_parasitic": {
                "calibrated": False,
                "observation": self.uncalibrated_parasitic_observation,
                "signoff_authority": False,
            },
        }

    @property
    def declared_gap_zero(self) -> bool:
        return (
            self.hard_gate_failures == 0
            and self.intent_failures == 0
            and self.reference_structure_gap == 0.0
            and self.route_resource_overflow == 0
            and self.bbox_lower_bound_gap == 0.0
        )


@dataclass(frozen=True)
class ExactCandidateFeedback:
    """One typed, digest-bound serial exact replay outcome."""

    candidate_sha256: str
    structure_fingerprint_sha256: str
    outcome: Literal["passed", "blocked"]
    vector: PhysicalSearchVector
    evidence_sha256: str
    blocker_code: str
    gds_sha256: str | None = None

    def __post_init__(self) -> None:
        if type(self.vector) is not PhysicalSearchVector:
            raise ValueError("exact feedback vector must be PhysicalSearchVector")
        for field in (
            "candidate_sha256",
            "structure_fingerprint_sha256",
            "evidence_sha256",
        ):
            if type(getattr(self, field)) is not str:
                raise ValueError(f"{field} must be an exact string")
            object.__setattr__(
                self,
                field,
                _canonical_sha256(getattr(self, field), label=field),
            )
        if type(self.outcome) is not str or self.outcome not in {"passed", "blocked"}:
            raise ValueError("exact feedback outcome must be passed or blocked")
        if type(self.blocker_code) is not str:
            raise ValueError("blocker_code must be an exact string")
        blocker = str(self.blocker_code).strip()
        if self.outcome == "blocked" and not blocker:
            raise ValueError("blocked exact feedback requires a typed blocker_code")
        if self.outcome == "passed" and blocker:
            raise ValueError("passed exact feedback cannot carry blocker_code")
        object.__setattr__(self, "blocker_code", blocker)
        if self.gds_sha256 is None:
            if self.outcome == "passed":
                raise ValueError("passed exact feedback requires canonical GDS SHA-256")
        else:
            if type(self.gds_sha256) is not str:
                raise ValueError("gds_sha256 must be an exact string or None")
            object.__setattr__(
                self,
                "gds_sha256",
                _canonical_sha256(self.gds_sha256, label="gds_sha256"),
            )
        if self.outcome == "passed" and not self.vector.declared_gap_zero:
            raise ValueError(
                "passed exact feedback requires zero hard/intent/resource/bbox gap"
            )

    def canonical_data(self) -> dict[str, object]:
        return {
            "schema": "clay.exact-candidate-feedback.v1",
            "candidate_sha256": self.candidate_sha256,
            "structure_fingerprint_sha256": self.structure_fingerprint_sha256,
            "outcome": self.outcome,
            "vector": self.vector.canonical_data(),
            "evidence_sha256": self.evidence_sha256,
            "blocker_code": self.blocker_code,
            "gds_sha256": self.gds_sha256,
        }


@dataclass(frozen=True)
class PhysicalIntentCandidateEvidence:
    candidate: CompactIntentCandidate
    search_stage: str
    outcome: CheapOutcome
    candidate_source: Path | None = None
    candidate_source_sha256: str = ""
    metrics: Mapping[str, object] | None = None
    error_type: str = ""
    error: str = ""
    pdk_failure_evidence: tuple[PdkFailureEvidence, ...] = ()

    def canonical_data(self) -> dict[str, object]:
        data: dict[str, object] = {
            "candidate": self.candidate.canonical_data(),
            "candidate_sha256": self.candidate.digest,
            "structure_fingerprint_sha256": compact_structure_fingerprint(
                self.candidate
            ),
            "search_stage": self.search_stage,
            "outcome": self.outcome,
            # Compatibility for existing plan readers during runner migration.
            "status": "pass" if self.outcome == "pass" else "fail",
        }
        if self.candidate_source is not None:
            data["candidate_source"] = str(self.candidate_source)
            data["candidate_source_sha256"] = _canonical_sha256(
                self.candidate_source_sha256,
                label="candidate_source_sha256",
            )
        if self.metrics is not None:
            data["metrics"] = dict(self.metrics)
        if self.error_type:
            data["error_type"] = self.error_type
        if self.error:
            data["error"] = self.error
        data["pdk_failure_evidence"] = [
            item.canonical_data() for item in self.pdk_failure_evidence
        ]
        return data


@dataclass(frozen=True)
class PhysicalIntentSearchCheckpoint:
    """Canonical, non-signoff progress receipt for one bounded cheap search."""

    target_sha256: str
    candidate_denominator_sha256: tuple[str, ...]
    structure_denominator_sha256: tuple[str, ...]
    evaluated_candidate_sha256: tuple[str, ...]
    remaining_candidate_sha256: tuple[str, ...]
    admitted_candidate_sha256: tuple[str, ...]
    outcome_by_candidate: tuple[tuple[str, CheapOutcome], ...]
    new_evaluation_count: int
    domain_exhausted: bool
    status: Literal["best_observed", "no_admitted_candidate"]
    signoff_authority: bool = False
    schema: str = "clay.physical-intent-search-checkpoint/v1"

    def __post_init__(self) -> None:
        if (
            type(self.schema) is not str
            or self.schema != "clay.physical-intent-search-checkpoint/v1"
        ):
            raise ValueError("physical-intent checkpoint schema is unsupported")
        if type(self.target_sha256) is not str:
            raise ValueError("checkpoint target_sha256 must be an exact string")
        _canonical_sha256(self.target_sha256, label="checkpoint target_sha256")

        def digest_tuple(value: object, *, label: str) -> tuple[str, ...]:
            if type(value) is not tuple or len(value) > _MAX_SEARCH_CANDIDATES:
                raise ValueError(f"{label} must be a bounded exact tuple")
            if any(type(item) is not str for item in value):
                raise ValueError(f"{label} must contain exact strings")
            result = tuple(
                _canonical_sha256(item, label=label)
                for item in value
            )
            if len(set(result)) != len(result):
                raise ValueError(f"{label} contains a duplicate")
            return result

        denominator = digest_tuple(
            self.candidate_denominator_sha256,
            label="checkpoint candidate denominator",
        )
        if not denominator or denominator != tuple(sorted(denominator)):
            raise ValueError(
                "checkpoint candidate denominator must be non-empty and canonical"
            )
        structures = digest_tuple(
            self.structure_denominator_sha256,
            label="checkpoint structure denominator",
        )
        if not structures or structures != tuple(sorted(structures)):
            raise ValueError(
                "checkpoint structure denominator must be non-empty and canonical"
            )
        if len(structures) > len(denominator):
            raise ValueError(
                "checkpoint structure denominator exceeds candidate denominator"
            )
        evaluated = digest_tuple(
            self.evaluated_candidate_sha256,
            label="checkpoint evaluated candidates",
        )
        remaining = digest_tuple(
            self.remaining_candidate_sha256,
            label="checkpoint remaining candidates",
        )
        if evaluated != denominator[: len(evaluated)] or remaining != denominator[
            len(evaluated) :
        ]:
            raise ValueError(
                "checkpoint evaluated/remaining candidates are not one exact prefix"
            )
        admitted = digest_tuple(
            self.admitted_candidate_sha256,
            label="checkpoint admitted candidates",
        )
        admitted_set = set(admitted)
        if admitted != tuple(item for item in evaluated if item in admitted_set):
            raise ValueError(
                "checkpoint admitted candidates must be an evaluated canonical subset"
            )
        if type(self.outcome_by_candidate) is not tuple or len(
            self.outcome_by_candidate
        ) != len(evaluated):
            raise ValueError("checkpoint outcomes must cover the evaluated prefix")
        outcome_digests: list[str] = []
        for item in self.outcome_by_candidate:
            if (
                type(item) is not tuple
                or len(item) != 2
                or type(item[0]) is not str
                or type(item[1]) is not str
                or item[1] not in {"pass", "infeasible", "unmeasurable"}
            ):
                raise ValueError("checkpoint outcome is not exact typed")
            outcome_digests.append(
                _canonical_sha256(item[0], label="checkpoint outcome candidate")
            )
        if tuple(outcome_digests) != evaluated:
            raise ValueError("checkpoint outcomes disagree with evaluated candidates")
        outcome_by_digest = dict(self.outcome_by_candidate)
        if any(outcome_by_digest[digest] != "pass" for digest in admitted):
            raise ValueError("checkpoint can admit only cheap-pass candidates")
        new_count = _search_count(
            self.new_evaluation_count,
            label="checkpoint new evaluation count",
        )
        if new_count > len(evaluated):
            raise ValueError("checkpoint new evaluation count exceeds evaluated prefix")
        if type(self.domain_exhausted) is not bool:
            raise ValueError("checkpoint domain_exhausted must be an exact bool")
        if self.domain_exhausted and remaining:
            raise ValueError("checkpoint cannot exhaust a domain with remaining candidates")
        expected_status = (
            "best_observed" if admitted else "no_admitted_candidate"
        )
        if type(self.status) is not str or self.status != expected_status:
            raise ValueError("checkpoint status disagrees with admitted candidates")
        if self.signoff_authority is not False:
            raise ValueError("cheap-search checkpoint cannot carry signoff authority")
        object.__setattr__(self, "new_evaluation_count", new_count)

    def canonical_data(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "target_sha256": self.target_sha256,
            "candidate_denominator_sha256": list(
                self.candidate_denominator_sha256
            ),
            "structure_denominator_sha256": list(
                self.structure_denominator_sha256
            ),
            "evaluated_candidate_sha256": list(self.evaluated_candidate_sha256),
            "remaining_candidate_sha256": list(self.remaining_candidate_sha256),
            "admitted_candidate_sha256": list(self.admitted_candidate_sha256),
            "outcomes": [
                {"candidate_sha256": digest, "outcome": outcome}
                for digest, outcome in self.outcome_by_candidate
            ],
            "new_evaluation_count": self.new_evaluation_count,
            "domain_exhausted": self.domain_exhausted,
            "status": self.status,
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


@dataclass(frozen=True)
class PhysicalIntentSearchResult:
    pdk: str
    top_cell: str
    state_sha256: str
    authoring_source_sha256: str
    cheap_evaluator_sha256: str
    probe_evaluator_sha256: str | None
    seed: int
    domain_budget: int
    refinement_budget: int
    maximum_signoffs: int
    domain_exhausted: bool
    domain_total_combinations: int
    base_candidate_count: int
    refinement_count: int
    cheap_evaluation_scope: str
    cheap_evaluated_count: int
    pdk_failure_rule_deck_sha256: str
    pdk_failure_rule_implementation_sha256: str
    pdk_failure_rule_sha256: tuple[str, ...]
    rows: tuple[PhysicalIntentCandidateEvidence, ...]
    candidates: tuple[CompactIntentCandidate, ...]
    cohort: tuple[CompactIntentCandidate, ...]
    candidate_source_by_digest: Mapping[str, Path]
    probes_by_digest: Mapping[str, PhysicalIntentProbeResult]
    admissions_by_digest: Mapping[str, PhysicalSearchAdmission]
    physical_aliases: Mapping[str, str]
    probe_budget: PhysicalIntentProbeBudget | None
    checkpoint: PhysicalIntentSearchCheckpoint | None
    ledger_path: Path

    @property
    def candidate_count(self) -> int:
        return len(self.candidates)

    @property
    def cheap_pass_count(self) -> int:
        return sum(row.outcome == "pass" for row in self.rows)

    @property
    def probe_count(self) -> int:
        return len(self.probes_by_digest)

    @property
    def structure_candidate_count(self) -> int:
        return len({
            compact_structure_fingerprint(candidate)
            for candidate in self.candidates
        })

    @property
    def cohort_structure_count(self) -> int:
        return len({
            compact_structure_fingerprint(candidate)
            for candidate in self.cohort
        })

    @property
    def cheap_new_evaluated_count(self) -> int:
        return (
            self.cheap_evaluated_count
            if self.checkpoint is None
            else self.checkpoint.new_evaluation_count
        )

    def cheap_metrics(self, candidate_sha256: str) -> Mapping[str, object]:
        matches = [
            row.metrics
            for row in self.rows
            if row.candidate.digest == candidate_sha256 and row.outcome == "pass"
        ]
        if len(matches) != 1 or matches[0] is None:
            raise KeyError(f"candidate {candidate_sha256} has no unique cheap metrics")
        return matches[0]

    def canonical_data(self) -> dict[str, object]:
        data: dict[str, object] = {
            "schema": (
                "clay.physical-intent-search/v3"
                if self.checkpoint is not None
                else "clay.physical-intent-search/v2"
            ),
            "binding_truth": False,
            "evidence_scope": "pre_signoff_candidate_preparation",
            "pdk": self.pdk,
            "top_cell": self.top_cell,
            "state_sha256": self.state_sha256,
            "authoring_source_sha256": self.authoring_source_sha256,
            "cheap_evaluator_sha256": self.cheap_evaluator_sha256,
            "probe_evaluator_sha256": self.probe_evaluator_sha256,
            "seed": self.seed,
            "domain_budget": self.domain_budget,
            "refinement_budget": self.refinement_budget,
            "maximum_signoffs": self.maximum_signoffs,
            "domain_exhausted": self.domain_exhausted,
            "domain_total_combinations": self.domain_total_combinations,
            "base_candidate_count": self.base_candidate_count,
            "refinement_count": self.refinement_count,
            "candidate_count": self.candidate_count,
            "structure_candidate_count": self.structure_candidate_count,
            "cheap_evaluation_scope": self.cheap_evaluation_scope,
            "cheap_evaluated_count": self.cheap_evaluated_count,
            "cheap_pass_count": self.cheap_pass_count,
            "pdk_failure_rule_deck_sha256": self.pdk_failure_rule_deck_sha256,
            "pdk_failure_rule_implementation_sha256": (
                self.pdk_failure_rule_implementation_sha256
            ),
            "pdk_failure_rule_sha256": list(self.pdk_failure_rule_sha256),
            "probe_budget": (
                None
                if self.probe_budget is None
                else self.probe_budget.canonical_data()
            ),
            "probe_count": self.probe_count,
            "search_outcome": (
                "best_observed" if self.cohort else "no_convergence"
            ),
            "selection_policy": "structure_fingerprint_round_robin.v1",
            "exact_candidate_denominator": len(self.cohort),
            "cohort_structure_count": self.cohort_structure_count,
            "probes_by_digest": {
                digest: probe.metrics()
                for digest, probe in sorted(self.probes_by_digest.items())
            },
            "cohort_sha256": [candidate.digest for candidate in self.cohort],
            "physical_aliases": dict(sorted(self.physical_aliases.items())),
            "candidates": [row.canonical_data() for row in self.rows],
        }
        if self.checkpoint is not None:
            data.update(
                cheap_new_evaluated_count=self.cheap_new_evaluated_count,
                search_outcome=self.checkpoint.status,
                checkpoint=self.checkpoint.canonical_data(),
                checkpoint_sha256=self.checkpoint.sha256,
                admissions_by_digest={
                    digest: admission.canonical_data()
                    for digest, admission in sorted(
                        self.admissions_by_digest.items()
                    )
                },
            )
        return data


def _write_json(path: Path, payload: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"bounded cheap outcome repeats key {key!r}")
        result[key] = value
    return result


def _load_bounded_cheap_outcome(
    path: Path,
    *,
    input_sha256: str,
    candidate: CompactIntentCandidate,
) -> dict[str, str] | None:
    """Load one durable typed non-measurement for deterministic batch progress."""

    if not path.is_file():
        return None
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_json_object,
        )
        if type(payload) is not dict or set(payload) != _BOUNDED_OUTCOME_KEYS:
            return None
        if (
            payload.get("schema") != "clay.physical-compact-batch-outcome/v1"
            or payload.get("input_sha256") != input_sha256
            or payload.get("status") != "fail"
            or payload.get("outcome") != "unmeasurable"
            or payload.get("candidate") != candidate.canonical_data()
            or payload.get("candidate_sha256") != candidate.digest
            or payload.get("signoff_authority") is not False
        ):
            return None
        error_type = payload.get("error_type")
        error = payload.get("error")
        if (
            type(error_type) is not str
            or not error_type
            or len(error_type) > 256
            or type(error) is not str
            or len(error) > 4096
        ):
            return None
        return {"error_type": error_type, "error": error}
    except (OSError, UnicodeError, ValueError, TypeError, KeyError):
        return None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_search_input_sha256(
    path: Path,
    expected_sha256: str,
    *,
    label: str,
) -> None:
    try:
        observed_sha256 = _file_sha256(path)
    except OSError as exc:
        raise _SearchInputChanged(
            f"{label} changed during physical intent search: {path}"
        ) from exc
    if observed_sha256 != expected_sha256:
        raise _SearchInputChanged(
            f"{label} changed during physical intent search: {path}"
        )


def physical_intent_state_sha256(source_path: Path | str) -> str:
    """Bind caches to source plus every implementation layer they consume."""

    source = Path(source_path).resolve()
    repo_root = Path(__file__).resolve().parents[2]
    roots = (
        source,
        repo_root / "clay" / "layout",
        repo_root / "clay" / "pdk",
        repo_root / "clay" / "pdks",
        repo_root / "clay" / "flow" / "optimized_clay.py",
        repo_root / "clay" / "flow" / "optimized_layout_style.py",
        Path(__file__).resolve().with_name("intent_domain.py"),
        Path(__file__).resolve().with_name("physical_intent.py"),
        Path(__file__).resolve(),
    )
    labelled: dict[str, Path] = {}
    for root in roots:
        if root.is_file():
            files = (root,)
        elif root.is_dir():
            files = tuple(
                path
                for path in root.rglob("*")
                if path.is_file()
                and path.suffix.casefold() in {".py", ".yaml", ".yml", ".json", ".toml"}
                and "__pycache__" not in path.parts
            )
        else:
            continue
        for path in files:
            resolved = path.resolve()
            if resolved == source:
                label = "authoring_source"
            else:
                try:
                    label = resolved.relative_to(repo_root).as_posix()
                except ValueError:
                    label = f"implementation/{resolved.name}"
            labelled[label] = resolved
    digest = hashlib.sha256()
    for label, path in sorted(labelled.items()):
        encoded = label.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(_file_sha256(path).encode("ascii"))
    return digest.hexdigest()


def _input_sha256(payload: Mapping[str, object]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _canonical_sha256(value: str | None, *, label: str) -> str:
    digest = str(value or "").casefold()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{label} must be one canonical SHA-256")
    return digest


def _pdk_failure_context(
    pdk: str,
) -> tuple[tuple[PdkFailureRule, ...], str, str]:
    from clay.pdk import get_pdk

    adapter = get_pdk(str(pdk))
    rules = tuple(adapter.pdk_failure_rules())
    deck_digest, implementation_digest = current_failure_rule_context(adapter)
    for rule in rules:
        if not isinstance(rule, PdkFailureRule):
            raise TypeError("PDK failure-rule resolver returned a non-rule value")
        if rule.pdk_id != str(pdk):
            raise ValueError("PDK failure rule is owned by a different adapter")
    return rules, deck_digest, implementation_digest


def _candidate_failure_geometry(
    candidate: CompactIntentCandidate,
    metrics: Mapping[str, object] | None,
    pdk: str,
) -> dict[str, object]:
    """Project only name-free candidate/artifact geometry into rule features."""

    from clay.layout.route.emit_gds import pdk_manufacturing_grid_nm
    from clay.layout.tech import LayoutTech

    geometry: dict[str, object] = {}
    grid_nm = float(pdk_manufacturing_grid_nm(LayoutTech.resolve(str(pdk))))
    dimensions_um = tuple(
        value
        for value in (
            candidate.edge_rail_width_um,
            candidate.mos_fold_height_um,
            candidate.block_gap_um,
        )
        if value is not None
    )
    if dimensions_um and grid_nm > 0:
        geometry["manufacturing_grid_remainder_nm"] = max(
            abs(
                (float(value) * 1000.0) / grid_nm
                - round((float(value) * 1000.0) / grid_nm)
            )
            * grid_nm
            for value in dimensions_um
        )
    if candidate.edge_rail_width_um is not None:
        geometry["rail_width_nm"] = float(candidate.edge_rail_width_um) * 1000.0
    if metrics is not None:
        for key in ("bbox_width_nm", "bbox_height_nm"):
            if key in metrics:
                geometry[key] = metrics[key]
        if "whitespace_um2" in metrics:
            geometry["whitespace_nm2"] = float(metrics["whitespace_um2"]) * 1e6
        macro_sha = str(metrics.get("macro_consumption_sha256", ""))
        if len(macro_sha) == 64:
            geometry["body_well_compatible"] = True
            geometry["well_tie_present"] = True
    return geometry


def _default_evaluator_sha256(evaluator: Callable[..., object]) -> str:
    module_path = Path(inspect.getfile(evaluator)).resolve()
    payload = {
        "module_sha256": _file_sha256(module_path),
        "module": str(getattr(evaluator, "__module__", "")),
        "qualname": str(getattr(evaluator, "__qualname__", "")),
    }
    return _input_sha256(payload)


def _evaluator_sha256(
    evaluator: Callable[..., object],
    default: Callable[..., object],
    declared: str | None,
    *,
    label: str,
) -> str:
    if evaluator is default:
        measured = _default_evaluator_sha256(default)
        if (
            declared is not None
            and _canonical_sha256(declared, label=label) != measured
        ):
            raise ValueError(f"{label} disagrees with the default evaluator")
        return measured
    if declared is None:
        raise ValueError(f"injected evaluator requires explicit {label} provenance")
    return _canonical_sha256(declared, label=label)


def _validate_cheap_metrics(
    raw: Mapping[str, object],
    *,
    materialization_mode: str,
    expected_functional_group_ids: tuple[str, ...],
) -> dict[str, object]:
    if type(raw) is not dict or set(raw) != _CHEAP_METRIC_KEYS:
        raise ValueError("physical cheap metrics must use the exact v2 schema")
    if type(materialization_mode) is not str or materialization_mode not in {
        "authoring",
        "derived",
    }:
        raise ValueError("physical cheap materialization mode is invalid")
    if (
        type(expected_functional_group_ids) is not tuple
        or any(type(item) is not str for item in expected_functional_group_ids)
        or tuple(sorted(expected_functional_group_ids))
        != expected_functional_group_ids
        or len(set(expected_functional_group_ids))
        != len(expected_functional_group_ids)
    ):
        raise ValueError("expected functional-group IDs must be an exact sorted tuple")
    metrics = dict(raw)
    raw_bbox = metrics["bbox_nm"]
    if type(raw_bbox) is not list or len(raw_bbox) != 4 or any(
        type(value) is not int or abs(value) > _MAX_CHEAP_COORDINATE_NM
        for value in (raw_bbox if type(raw_bbox) is list else ())
    ):
        raise ValueError("physical cheap bbox_nm must contain four bounded integers")
    bbox = tuple(raw_bbox)
    width_nm = bbox[2] - bbox[0]
    height_nm = bbox[3] - bbox[1]
    if width_nm <= 0 or height_nm <= 0:
        raise ValueError("physical cheap bbox_nm must have positive extent")
    if type(metrics["bbox_width_nm"]) is not int or metrics["bbox_width_nm"] != width_nm:
        raise ValueError("physical cheap bbox_width_nm disagrees with bbox_nm")
    if (
        type(metrics["bbox_height_nm"]) is not int
        or metrics["bbox_height_nm"] != height_nm
    ):
        raise ValueError("physical cheap bbox_height_nm disagrees with bbox_nm")
    for key in (
        "bbox_area_um2",
        "whitespace_um2",
        "wire_length_um",
        "ring_perimeter_um",
        "ring_max_perimeter_um",
        "ring_span_waste_um2",
    ):
        metrics[key] = _search_measure(metrics.get(key), label=f"physical cheap {key}")
    if float(metrics["bbox_area_um2"]) <= 0.0:
        raise ValueError("physical cheap bbox_area_um2 must be positive")
    expected_area_um2 = width_nm * height_nm / 1_000_000.0
    if float(metrics["bbox_area_um2"]) != expected_area_um2:
        raise ValueError("physical cheap bbox_area_um2 disagrees with bbox_nm")
    if float(metrics["whitespace_um2"]) > expected_area_um2:
        raise ValueError("physical cheap whitespace_um2 exceeds bbox area")
    via_count = metrics.get("via_count")
    if type(via_count) is not int or not 0 <= via_count <= _MAX_SEARCH_COUNT:
        raise ValueError("physical cheap via_count must be a bounded non-negative integer")
    if (
        float(metrics["ring_max_perimeter_um"])
        > float(metrics["ring_perimeter_um"]) + 1e-9
    ):
        raise ValueError("maximum ring perimeter exceeds total ring perimeter")
    signature = metrics.get("physical_signature_sha256")
    if type(signature) is not str:
        raise ValueError("physical_signature_sha256 must be an exact string")
    metrics["physical_signature_sha256"] = _canonical_sha256(
        signature,
        label="physical_signature_sha256",
    )
    placement_state = metrics.get("placement_state_sha256")
    if type(placement_state) is not str:
        raise ValueError("physical cheap placement_state_sha256 must be an exact string")
    if placement_state:
        metrics["placement_state_sha256"] = _canonical_sha256(
            placement_state,
            label="placement_state_sha256",
        )
    macro_consumption = metrics.get("macro_consumption_sha256")
    if type(macro_consumption) is not str:
        raise ValueError(
            "physical cheap macro_consumption_sha256 must be an exact string"
        )
    if macro_consumption:
        metrics["macro_consumption_sha256"] = _canonical_sha256(
            macro_consumption,
            label="macro_consumption_sha256",
        )
    if materialization_mode == "derived" and (
        not placement_state or not macro_consumption
    ):
        raise ValueError("derived cheap metrics require placement and macro evidence")
    if materialization_mode == "authoring" and (
        placement_state or macro_consumption
    ):
        raise ValueError("authoring cheap metrics cannot claim derived evidence")
    groups = metrics.get("functional_group_bboxes_nm")
    if type(groups) is not list or len(groups) > _MAX_SEARCH_CANDIDATES:
        raise ValueError(
            "physical cheap functional_group_bboxes_nm must be a bounded list"
        )
    typed_groups: list[PhysicalIntentFunctionalGroupBBox] = []
    for index, raw_group in enumerate(groups):
        if type(raw_group) is not dict or set(raw_group) != {"group_id", "bbox_nm"}:
            raise ValueError(f"physical cheap functional group {index} is malformed")
        raw_bbox = raw_group["bbox_nm"]
        if type(raw_bbox) is not list:
            raise ValueError(f"physical cheap functional group {index} bbox must be a list")
        try:
            typed_groups.append(
                PhysicalIntentFunctionalGroupBBox(
                    group_id=raw_group["group_id"],
                    bbox_nm=tuple(raw_bbox),
                )
            )
        except ValueError as exc:
            raise ValueError(
                f"physical cheap functional group {index} is invalid: {exc}"
            ) from exc
    group_ids = tuple(item.group_id for item in typed_groups)
    if len(set(group_ids)) != len(group_ids):
        raise ValueError("physical cheap functional groups repeat a group ID")
    if group_ids != tuple(sorted(group_ids)):
        raise ValueError("physical cheap functional groups are not canonical")
    if group_ids != expected_functional_group_ids:
        raise ValueError("physical cheap functional groups disagree with source inventory")
    x0, y0, x1, y1 = bbox
    if any(
        item.bbox_nm[0] < x0
        or item.bbox_nm[1] < y0
        or item.bbox_nm[2] > x1
        or item.bbox_nm[3] > y1
        for item in typed_groups
    ):
        raise ValueError("physical cheap functional group lies outside bbox_nm")
    metrics["bbox_nm"] = list(bbox)
    metrics["functional_group_bboxes_nm"] = [
        item.metrics_data() for item in typed_groups
    ]
    return metrics


def _validate_probe_result(
    probe: PhysicalIntentProbeResult,
    *,
    candidate_sha256: str,
) -> PhysicalIntentProbeResult:
    if not isinstance(probe, PhysicalIntentProbeResult):
        raise ValueError("probe evaluator must return PhysicalIntentProbeResult")
    if probe.candidate_sha256 != candidate_sha256:
        raise ValueError("probe candidate digest disagrees with request")
    _canonical_sha256(probe.evidence_sha256, label="probe evidence_sha256")
    if probe.status not in {"complete", "stopped"}:
        raise ValueError("probe status must be complete or stopped")
    if not probe.phase or not probe.state or not probe.reason:
        raise ValueError("probe phase, state and reason must be named")
    if probe.status == "complete" and (
        probe.phase != "ROUTE_COMPLETE" or probe.reason != "NONE"
    ):
        raise ValueError("complete probe requires ROUTE_COMPLETE/NONE evidence")
    return probe


def _exact_candidate_cheap_rank(
    candidate: CompactIntentCandidate,
    cheap_metrics: Mapping[str, Mapping[str, object]],
) -> tuple[float | int | str, ...]:
    metrics = cheap_metrics[candidate.digest]

    def number(key: str) -> float:
        value = metrics.get(key, math.inf)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f"exact replay cheap rank requires non-negative {key}")
        return float(value)

    via_count = metrics.get("via_count", _MAX_SEARCH_COUNT)
    _search_count(via_count, label="via_count")
    return (
        number("bbox_area_um2"),
        number("wire_length_um"),
        int(via_count),
        candidate.digest,
    )


def select_exact_replay_candidates(
    candidates: Iterable[CompactIntentCandidate],
    cheap_metrics: Mapping[str, Mapping[str, object]],
    *,
    feedback: tuple[ExactCandidateFeedback, ...] = (),
    maximum: int,
    admissions: Mapping[str, PhysicalSearchAdmission] | None = None,
    target: PhysicalSearchTarget | None = None,
    target_sha256: str | None = None,
) -> tuple[CompactIntentCandidate, ...]:
    """Select serial exact replays, visiting untried structures first."""

    if type(maximum) is not int or maximum <= 0:
        raise ValueError("exact replay maximum must be a positive integer")
    bounded: list[CompactIntentCandidate] = []
    for index, candidate in enumerate(candidates):
        if index >= _MAX_SEARCH_CANDIDATES:
            raise ValueError("exact replay candidate denominator exceeds bound")
        if type(candidate) is not CompactIntentCandidate:
            raise ValueError("exact replay candidates contain an invalid value")
        bounded.append(candidate)
    if len({candidate.digest for candidate in bounded}) != len(bounded):
        raise ValueError("exact replay candidate denominator contains a duplicate")
    ordered = tuple(sorted(bounded, key=lambda item: item.digest))
    if len(ordered) > _MAX_SEARCH_CANDIDATES:
        raise ValueError("exact replay candidate denominator exceeds bound")
    by_digest = {candidate.digest: candidate for candidate in ordered}
    if any(candidate.digest not in cheap_metrics for candidate in ordered):
        raise ValueError("exact replay candidates require complete cheap metrics")
    target_aware = (
        admissions is not None or target is not None or target_sha256 is not None
    )
    admission_by_digest: dict[str, PhysicalSearchAdmission] = {}
    if target_aware:
        if admissions is None or target is None or target_sha256 is None:
            raise ValueError(
                "target-aware exact replay requires admissions, target and target_sha256"
            )
        if type(target) is not PhysicalSearchTarget:
            raise ValueError("target-aware exact replay target must be exact typed")
        if type(target_sha256) is not str:
            raise ValueError("target_sha256 must be an exact string")
        canonical_target = _canonical_sha256(
            target_sha256,
            label="target_sha256",
        )
        if canonical_target != target_sha256:
            raise ValueError("target_sha256 must be canonical")
        if target.canonical_sha256 != canonical_target:
            raise ValueError("target-aware exact replay has stale target identity")
        if type(admissions) is not dict or set(admissions) != set(by_digest):
            raise ValueError(
                "target-aware exact replay requires complete admission receipts"
            )
        for digest in sorted(admissions):
            if type(digest) is not str:
                raise ValueError("admission mapping keys must be exact strings")
            admission = admissions[digest]
            if type(admission) is not PhysicalSearchAdmission:
                raise ValueError("admission mapping contains an invalid receipt")
            if admission.candidate_sha256 != digest:
                raise ValueError("admission receipt names a different candidate")
            if admission.target_sha256 != canonical_target:
                raise ValueError("admission receipt has stale target identity")
            replayed = evaluate_physical_search_admission(
                target,
                by_digest[digest],
                cheap_metrics[digest],
            )
            if admission != replayed:
                raise ValueError("admission receipt disagrees with target/metrics replay")
            admission_by_digest[digest] = admission
    elif admissions is not None:
        raise ValueError("admissions require target_sha256")
    if type(feedback) is not tuple or len(feedback) > _MAX_SEARCH_CANDIDATES:
        raise ValueError("exact replay feedback must be a bounded exact tuple")
    seen: set[str] = set()
    attempted_structures: set[str] = set()
    for item in feedback:
        if type(item) is not ExactCandidateFeedback:
            raise ValueError("exact replay feedback contains an invalid value")
        candidate = by_digest.get(item.candidate_sha256)
        if candidate is None:
            raise ValueError("exact replay feedback names a foreign candidate")
        if target_aware and not admission_by_digest[item.candidate_sha256].admitted:
            raise ValueError("exact replay feedback names a non-admitted candidate")
        if item.candidate_sha256 in seen:
            raise ValueError("exact replay feedback repeats one candidate")
        fingerprint = compact_structure_fingerprint(candidate)
        if item.structure_fingerprint_sha256 != fingerprint:
            raise ValueError("exact replay feedback has stale structure identity")
        seen.add(item.candidate_sha256)
        attempted_structures.add(fingerprint)
    passed_indices = tuple(
        index for index, item in enumerate(feedback) if item.outcome == "passed"
    )
    if len(passed_indices) > 1 or (
        passed_indices and passed_indices[0] != len(feedback) - 1
    ):
        raise ValueError("serial exact trajectory cannot continue after a pass")
    if passed_indices:
        return ()

    buckets: dict[str, list[CompactIntentCandidate]] = {}
    for candidate in ordered:
        if candidate.digest in seen:
            continue
        if target_aware and not admission_by_digest[candidate.digest].admitted:
            continue
        buckets.setdefault(
            compact_structure_fingerprint(candidate), []
        ).append(candidate)

    def replay_rank(candidate: CompactIntentCandidate) -> tuple[object, ...]:
        if target_aware:
            return admission_by_digest[candidate.digest].rank_key()
        return _exact_candidate_cheap_rank(candidate, cheap_metrics)

    for values in buckets.values():
        values.sort(key=replay_rank)
    keys = sorted(
        buckets,
        key=lambda fingerprint: (
            fingerprint in attempted_structures,
            replay_rank(buckets[fingerprint][0]),
            fingerprint,
        ),
    )
    selected: list[CompactIntentCandidate] = []
    while len(selected) < maximum and any(buckets[key] for key in keys):
        for key in keys:
            if buckets[key]:
                selected.append(buckets[key].pop(0))
                if len(selected) == maximum:
                    break
    return tuple(selected)


def classify_exact_search_outcome(
    feedback: tuple[ExactCandidateFeedback, ...],
    *,
    domain_exhausted: bool,
    denominator: int,
) -> Literal["no_convergence", "best_observed", "optimal"]:
    """Calibrate an unprivileged bounded-run claim.

    ``domain_exhausted`` and ``denominator`` are caller-supplied progress
    metadata.  They are useful for consistency checks, but cannot prove that
    the supplied denominator is the complete declared domain.  Consequently
    this API may report a passing candidate only as ``best_observed``.  An
    ``optimal`` claim requires the replay-bound Task-11 optimality receipt.
    """

    if type(feedback) is not tuple or any(
        type(item) is not ExactCandidateFeedback for item in feedback
    ):
        raise ValueError("exact search outcome requires typed feedback tuple")
    if type(domain_exhausted) is not bool:
        raise ValueError("domain_exhausted must be an exact bool")
    if type(denominator) is not int or not 0 < denominator <= _MAX_SEARCH_COUNT:
        raise ValueError("exact search denominator must be positive and bounded")
    if len(feedback) > denominator or len({
        item.candidate_sha256 for item in feedback
    }) != len(feedback):
        raise ValueError("exact search feedback disagrees with denominator")
    passed_indices = tuple(
        index for index, item in enumerate(feedback) if item.outcome == "passed"
    )
    if len(passed_indices) > 1 or (
        passed_indices and passed_indices[0] != len(feedback) - 1
    ):
        raise ValueError("serial exact trajectory cannot continue after a pass")
    passed = tuple(item for item in feedback if item.outcome == "passed")
    if not passed:
        return "no_convergence"
    return "best_observed"


def exact_search_trajectory_data(
    candidates: Iterable[CompactIntentCandidate],
    feedback: tuple[ExactCandidateFeedback, ...],
    *,
    domain_exhausted: bool,
) -> dict[str, object]:
    """Return a closed denominator/trajectory receipt for serial exact replay."""

    bounded: list[CompactIntentCandidate] = []
    for index, candidate in enumerate(candidates):
        if index >= _MAX_SEARCH_CANDIDATES:
            raise ValueError("exact trajectory candidate denominator exceeds bound")
        if type(candidate) is not CompactIntentCandidate:
            raise ValueError("exact trajectory contains an invalid candidate")
        bounded.append(candidate)
    if len({candidate.digest for candidate in bounded}) != len(bounded):
        raise ValueError("exact trajectory candidate denominator contains a duplicate")
    ordered = tuple(sorted(bounded, key=lambda item: item.digest))
    if not ordered:
        raise ValueError("exact trajectory denominator must be non-empty")
    if type(feedback) is not tuple:
        raise ValueError("exact trajectory feedback must be an exact tuple")
    by_digest = {candidate.digest: candidate for candidate in ordered}
    seen: set[str] = set()
    for index, item in enumerate(feedback):
        if type(item) is not ExactCandidateFeedback:
            raise ValueError("exact trajectory contains invalid feedback")
        candidate = by_digest.get(item.candidate_sha256)
        if candidate is None or item.candidate_sha256 in seen:
            raise ValueError("exact trajectory feedback disagrees with denominator")
        if item.structure_fingerprint_sha256 != compact_structure_fingerprint(
            candidate
        ):
            raise ValueError("exact trajectory feedback has stale structure identity")
        if item.outcome == "passed" and index != len(feedback) - 1:
            raise ValueError("serial exact trajectory cannot continue after a pass")
        seen.add(item.candidate_sha256)
    outcome = classify_exact_search_outcome(
        feedback,
        domain_exhausted=domain_exhausted,
        denominator=len(ordered),
    )
    passed = tuple(item for item in feedback if item.outcome == "passed")
    best = (
        None
        if not passed
        else min(passed, key=lambda item: (
            item.vector.rank_key(), item.candidate_sha256
        )).candidate_sha256
    )
    return {
        "schema": "clay.exact-search-trajectory.v1",
        "binding_truth": False,
        "signoff_authority": False,
        "domain_exhausted": domain_exhausted,
        "denominator": len(ordered),
        "structure_denominator": len({
            compact_structure_fingerprint(candidate) for candidate in ordered
        }),
        "evaluated_count": len(feedback),
        "outcome": outcome,
        "best_observed_candidate_sha256": best,
        "candidate_denominator": [
            {
                "candidate_sha256": candidate.digest,
                "structure_fingerprint_sha256": compact_structure_fingerprint(
                    candidate
                ),
            }
            for candidate in ordered
        ],
        "remaining_candidate_sha256": [
            candidate.digest for candidate in ordered if candidate.digest not in seen
        ],
        "trajectory": [item.canonical_data() for item in feedback],
        "electrical_optimality": (
            "not_claimed_uncalibrated_observation_only"
        ),
    }


def select_routability_cohort(
    candidates: Iterable[CompactIntentCandidate],
    cheap_metrics: Mapping[str, Mapping[str, object]],
    *,
    maximum: int,
) -> tuple[CompactIntentCandidate, ...]:
    """Round-robin exact structure classes before continuous variants."""

    if maximum <= 0:
        raise ValueError("routability cohort maximum must be positive")
    measured = [
        candidate for candidate in candidates if candidate.digest in cheap_metrics
    ]
    if not measured:
        return ()

    def rank(candidate: CompactIntentCandidate) -> tuple[object, ...]:
        metrics = cheap_metrics[candidate.digest]
        return (
            float(metrics["ring_max_perimeter_um"]),
            float(metrics["ring_span_waste_um2"]),
            len(candidate.guard_ring_groups),
            float(metrics["ring_perimeter_um"]),
            float(metrics["bbox_area_um2"]),
            float(metrics["wire_length_um"]),
            candidate.digest,
        )

    measured = list(
        deduplicate_physical_intent_candidates(
            measured,
            cheap_metrics,
            rank=rank,
        )
    )
    buckets: dict[str, list[CompactIntentCandidate]] = {}
    for candidate in measured:
        buckets.setdefault(compact_structure_fingerprint(candidate), []).append(
            candidate
        )
    for values in buckets.values():
        values.sort(key=rank)
    keys = sorted(
        buckets,
        key=lambda fingerprint: (rank(buckets[fingerprint][0]), fingerprint),
    )
    selected: list[CompactIntentCandidate] = []
    while len(selected) < maximum and any(buckets[key] for key in keys):
        for key in keys:
            if buckets[key]:
                selected.append(buckets[key].pop(0))
                if len(selected) == maximum:
                    break
    return tuple(selected)


def _physical_aliases(
    candidates: Iterable[CompactIntentCandidate],
    cheap_metrics: Mapping[str, Mapping[str, object]],
) -> dict[str, str]:
    measured = tuple(
        candidate for candidate in candidates if candidate.digest in cheap_metrics
    )
    if not measured:
        return {}
    representatives = select_routability_cohort(
        measured,
        cheap_metrics,
        maximum=len(measured),
    )
    representative_by_signature = {
        str(cheap_metrics[candidate.digest]["physical_signature_sha256"]): candidate
        for candidate in representatives
    }
    aliases: dict[str, str] = {}
    for candidate in measured:
        signature = str(cheap_metrics[candidate.digest]["physical_signature_sha256"])
        representative = representative_by_signature[signature]
        if candidate.digest != representative.digest:
            aliases[candidate.digest] = representative.digest
    return aliases


def refine_measured_candidates(
    candidates: Iterable[CompactIntentCandidate],
    cheap_metrics: Mapping[str, Mapping[str, object]],
    *,
    parent_count: int,
    refinement_budget: int,
) -> tuple[CompactIntentCandidate, ...]:
    """Recombine measured, routing-diverse parents within a declared budget."""

    if refinement_budget <= 0:
        return ()
    measured = tuple(
        candidate for candidate in candidates if candidate.digest in cheap_metrics
    )
    primary = select_routability_cohort(
        measured,
        cheap_metrics,
        maximum=max(2, int(parent_count)),
    )
    graph_buckets: dict[tuple[object, ...], list[CompactIntentCandidate]] = {}
    for candidate in measured:
        graph_key = (
            candidate.guard_ring_groups,
            candidate.passive_bank_partitions,
            candidate.well_island_partitions,
            candidate.common_centroid_rows,
        )
        graph_buckets.setdefault(graph_key, []).append(candidate)
    graph_representatives = tuple(
        select_routability_cohort(bucket, cheap_metrics, maximum=1)[0]
        for bucket in graph_buckets.values()
    )
    parents = tuple(set(primary).union(graph_representatives))
    return recombine_compact_intent_candidates(
        parents,
        maximum=int(refinement_budget),
    )


def select_exact_candidate(
    candidates: Iterable[CompactIntentCandidate],
    candidate_sha256: str | None,
) -> tuple[CompactIntentCandidate, ...]:
    """Return a full measurement slice or one explicitly named candidate."""

    ordered = tuple(candidates)
    if candidate_sha256 is None:
        return ordered
    matches = tuple(
        candidate for candidate in ordered if candidate.digest == str(candidate_sha256)
    )
    if len(matches) != 1:
        raise ValueError(
            f"exact candidate {candidate_sha256!r} is not present in the "
            "measured compact-intent domain"
        )
    return matches


def prepare_physical_intent_search(
    *,
    source_path: Path | str,
    source_cir: Path | str | None = None,
    top_cell: str,
    pdk: str,
    work_dir: Path | str,
    maximum_signoffs: int,
    candidate_sha256: str | None = None,
    maximum_new_cheap_evaluations: int | None = None,
    physical_search_target: PhysicalSearchTarget | None = None,
    cheap_evaluator: CheapPhysicalEvaluator = evaluate_compact_intent_candidate,
    probe_evaluator: ProbePhysicalEvaluator = evaluate_compact_intent_probe,
    cheap_evaluator_sha256: str | None = None,
    probe_evaluator_sha256: str | None = None,
    logger: Callable[[str], None] | None = None,
) -> PhysicalIntentSearchResult:
    """Prepare one declared target-PDK cohort without claiming signoff truth."""

    if type(maximum_signoffs) is not int or maximum_signoffs <= 0:
        raise ValueError("maximum_signoffs must be positive")
    bounded_batch = (
        maximum_new_cheap_evaluations is not None
        or physical_search_target is not None
    )
    if bounded_batch:
        if (
            type(maximum_new_cheap_evaluations) is not int
            or maximum_new_cheap_evaluations <= 0
            or maximum_new_cheap_evaluations > _MAX_SEARCH_CANDIDATES
        ):
            raise ValueError(
                "maximum_new_cheap_evaluations must be an exact bounded "
                "positive integer"
            )
        if type(physical_search_target) is not PhysicalSearchTarget:
            raise ValueError(
                "bounded cheap search requires an exact physical_search_target"
            )
    source = Path(source_path).resolve()
    source_cir_path: Path | None = None
    source_cir_sha = ""
    if source_cir is None:
        if cheap_evaluator is evaluate_compact_intent_candidate:
            raise ValueError("default physical intent search requires source_cir")
    else:
        source_cir_path = Path(source_cir).resolve()
        if not source_cir_path.is_file():
            raise ValueError(
                f"physical intent search source_cir is missing: {source_cir_path}"
            )
        source_cir_sha = _file_sha256(source_cir_path)
    root = Path(work_dir).resolve()
    authoring_source_sha = _file_sha256(source)
    if bounded_batch:
        assert physical_search_target is not None
        if (
            physical_search_target.top_cell != str(top_cell)
            or physical_search_target.pdk_id != str(pdk)
            or physical_search_target.source_sha256 != authoring_source_sha
        ):
            raise ValueError(
                "bounded cheap search target disagrees with source, cell or PDK"
            )
    cheap_evaluator_sha = _evaluator_sha256(
        cheap_evaluator,
        evaluate_compact_intent_candidate,
        cheap_evaluator_sha256,
        label="cheap_evaluator_sha256",
    )
    parsed = parse_clay_file(source)
    cells = [
        cell
        for cell in (getattr(parsed, "cell_defs", ()) or ())
        if getattr(cell, "local_name", getattr(cell, "name", None)) == top_cell
    ]
    if len(cells) != 1:
        raise ValueError(
            f"physical intent search requires exactly one {top_cell!r} cell; "
            f"found {len(cells)}"
        )
    cell = cells[0]
    from clay.layout.functional_roles import functional_group_records_from_cell

    expected_functional_group_ids = tuple(
        sorted(group.name for group in functional_group_records_from_cell(cell))
    )
    if len(set(expected_functional_group_ids)) != len(expected_functional_group_ids):
        raise ValueError("source cell repeats a functional-group ID")
    domain = build_compact_intent_domain(parsed, cell, str(pdk))
    domain_spec = compact_intent_domain_spec(cell)
    base_by_digest = {candidate.digest: candidate for candidate in domain.candidates}
    source_text = source.read_text(encoding="utf-8-sig")
    materialized_candidate = declared_compact_candidate(source_text)
    if materialized_candidate is not None:
        materialized_candidate = require_resolved_compact_intent(cell, source_text)
        if (
            candidate_sha256 is not None
            and candidate_sha256 != materialized_candidate.digest
        ):
            raise ValueError(
                "exact candidate digest disagrees with the candidate already "
                "materialized in the Clay source"
            )
    selected_sha256 = (
        materialized_candidate.digest
        if materialized_candidate is not None
        else candidate_sha256
    )
    if bounded_batch and selected_sha256 is not None:
        raise ValueError(
            "bounded cheap search cannot be combined with one exact candidate"
        )
    exact_candidate = materialized_candidate
    if exact_candidate is None and selected_sha256 is not None:
        exact_candidate = reconstruct_compact_intent_candidate(
            domain.candidates,
            selected_sha256,
        )
    exact_is_base = selected_sha256 in base_by_digest
    evaluation_candidates = (
        tuple(domain.candidates) if exact_candidate is None else (exact_candidate,)
    )
    if bounded_batch:
        evaluation_candidates = tuple(
            sorted(evaluation_candidates, key=lambda candidate: candidate.digest)
        )
    state_sha = physical_intent_state_sha256(source)
    failure_rules, failure_deck_sha, failure_implementation_sha = _pdk_failure_context(
        str(pdk)
    )
    failure_rule_shas = tuple(rule.sha256 for rule in failure_rules)
    metrics_by_digest: dict[str, dict[str, object]] = {}
    sources_by_digest: dict[str, Path] = {}
    rows: list[PhysicalIntentCandidateEvidence] = []
    evaluated_count = 0
    new_evaluated_count = 0
    pending_cache_writes: list[tuple[Path, Mapping[str, object]]] = []

    def require_search_inputs() -> None:
        _require_search_input_sha256(
            source,
            authoring_source_sha,
            label="authoring source",
        )
        if source_cir_path is not None:
            _require_search_input_sha256(
                source_cir_path,
                source_cir_sha,
                label="source_cir",
            )

    def queue_cache_write(path: Path, payload: Mapping[str, object]) -> None:
        pending_cache_writes.append((path, payload))

    def log(message: str) -> None:
        if logger is not None:
            logger(message)

    def measure(
        candidates: Iterable[CompactIntentCandidate],
        *,
        stage: str,
    ) -> None:
        nonlocal evaluated_count, new_evaluated_count
        ordered = tuple(candidates)
        for index, candidate in enumerate(ordered, start=1):
            failure_evidence = evaluate_pdk_failure_rules(
                failure_rules,
                pdk_id=str(pdk),
                deck_digest=failure_deck_sha,
                implementation_digest=failure_implementation_sha,
                geometry=_candidate_failure_geometry(candidate, None, str(pdk)),
            )
            rejecting = tuple(item for item in failure_evidence if item.reject)
            if rejecting:
                evaluated_count += 1
                rows.append(
                    PhysicalIntentCandidateEvidence(
                        candidate=candidate,
                        search_stage=stage,
                        outcome="infeasible",
                        error_type="PdkFailureRuleMatch",
                        error=", ".join(item.rule_id for item in rejecting),
                        pdk_failure_evidence=failure_evidence,
                    )
                )
                continue
            candidate_dir = root / candidate.digest
            candidate_source = candidate_dir / "design.clay"
            metrics_path = candidate_dir / "cheap_metrics.json"
            bounded_outcome_path = candidate_dir / "bounded_outcome.json"
            input_sha = _input_sha256(
                {
                    "state_sha256": state_sha,
                    "top_cell": str(top_cell),
                    "pdk": str(pdk),
                    "candidate_sha256": candidate.digest,
                    "source_cir_sha256": source_cir_sha,
                    "cheap_evaluator_sha256": cheap_evaluator_sha,
                    "pdk_failure_rule_sha256": list(failure_rule_shas),
                    "pdk_failure_rule_deck_sha256": failure_deck_sha,
                    "pdk_failure_rule_implementation_sha256": (
                        failure_implementation_sha
                    ),
                }
            )
            require_search_inputs()
            cached = load_physical_intent_cache(
                metrics_path,
                candidate_source=candidate_source,
                input_sha256=input_sha,
                candidate=candidate,
                expected_functional_group_ids=expected_functional_group_ids,
            )
            bounded_outcome = (
                _load_bounded_cheap_outcome(
                    bounded_outcome_path,
                    input_sha256=input_sha,
                    candidate=candidate,
                )
                if bounded_batch and cached is None
                else None
            )
            if (
                cached is None
                and bounded_outcome is None
                and bounded_batch
                and new_evaluated_count >= int(maximum_new_cheap_evaluations)
            ):
                log(
                    f"{pdk} CHEAP {stage} STOP new-budget="
                    f"{maximum_new_cheap_evaluations} next={candidate.digest[:12]}"
                )
                break
            log(
                f"{pdk} CHEAP {stage} {index}/{len(ordered)} "
                f"{'CACHE' if cached is not None or bounded_outcome is not None else 'START'} "
                f"candidate={candidate.digest[:12]}"
            )
            if bounded_outcome is not None:
                evaluated_count += 1
                rows.append(
                    PhysicalIntentCandidateEvidence(
                        candidate=candidate,
                        search_stage=stage,
                        outcome="unmeasurable",
                        error_type=bounded_outcome["error_type"],
                        error=bounded_outcome["error"],
                        pdk_failure_evidence=failure_evidence,
                    )
                )
                continue
            if cached is not None and cached["status"] == "fail":
                evaluated_count += 1
                rows.append(
                    PhysicalIntentCandidateEvidence(
                        candidate=candidate,
                        search_stage=stage,
                        outcome="infeasible",
                        error_type=str(cached["error_type"]),
                        error=str(cached["error"]),
                        pdk_failure_evidence=failure_evidence,
                    )
                )
                continue
            try:
                if cached is None:
                    new_evaluated_count += 1
                    evaluator_arguments = {
                        "source_path": source,
                        "output_path": candidate_source,
                        "top_cell": top_cell,
                        "pdk": str(pdk),
                        "candidate": candidate,
                        "seed": int(domain_spec.seed),
                    }
                    if source_cir_path is not None:
                        evaluator_arguments["source_cir"] = source_cir_path
                    measured = cheap_evaluator(
                        **evaluator_arguments,
                    )
                    require_search_inputs()
                    if measured.candidate_sha256 != candidate.digest:
                        raise RuntimeError(
                            "cheap evaluator candidate digest disagrees with request"
                        )
                    if (
                        Path(measured.candidate_source).resolve()
                        != candidate_source.resolve()
                    ):
                        raise RuntimeError(
                            "cheap evaluator candidate source disagrees with request"
                        )
                    if not candidate_source.is_file():
                        raise RuntimeError(
                            "cheap evaluator produced no candidate source"
                        )
                    metrics = _validate_cheap_metrics(
                        measured.metrics(),
                        materialization_mode=candidate.materialization_mode,
                        expected_functional_group_ids=expected_functional_group_ids,
                    )
                    queue_cache_write(
                        metrics_path,
                        {
                            "schema": "clay.physical-compact-cheap/v2",
                            "input_sha256": input_sha,
                            "status": "pass",
                            "candidate": candidate.canonical_data(),
                            "candidate_sha256": candidate.digest,
                            "candidate_source": str(candidate_source),
                            "candidate_source_sha256": _file_sha256(candidate_source),
                            "metrics": metrics,
                        },
                    )
                else:
                    metrics = _validate_cheap_metrics(
                        cached["metrics"],
                        materialization_mode=candidate.materialization_mode,
                        expected_functional_group_ids=expected_functional_group_ids,
                    )
                failure_evidence = evaluate_pdk_failure_rules(
                    failure_rules,
                    pdk_id=str(pdk),
                    deck_digest=failure_deck_sha,
                    implementation_digest=failure_implementation_sha,
                    geometry=_candidate_failure_geometry(candidate, metrics, str(pdk)),
                )
                rejecting = tuple(item for item in failure_evidence if item.reject)
                if rejecting:
                    raise PhysicalIntentInfeasible(
                        "PDK_FAILURE_RULE: "
                        + ", ".join(item.rule_id for item in rejecting)
                    )
            except _SearchInputChanged:
                raise
            except Exception as exc:  # candidate red is evidence, not domain abort
                require_search_inputs()
                deterministic = isinstance(exc, PhysicalIntentInfeasible)
                outcome: CheapOutcome = (
                    "infeasible" if deterministic else "unmeasurable"
                )
                error_type = type(exc).__name__
                error_message = str(exc)
                if bounded_batch and not deterministic:
                    if len(error_type) > 256:
                        error_type = "ExceptionTypeSha256:" + hashlib.sha256(
                            error_type.encode("utf-8")
                        ).hexdigest()
                    error_message = error_message[:4096]
                if deterministic:
                    queue_cache_write(
                        metrics_path,
                        {
                            "schema": "clay.physical-compact-cheap/v2",
                            "input_sha256": input_sha,
                            "status": "fail",
                            "outcome": outcome,
                            "candidate": candidate.canonical_data(),
                            "candidate_sha256": candidate.digest,
                            "error_type": error_type,
                            "error": error_message,
                        },
                    )
                elif bounded_batch:
                    queue_cache_write(
                        bounded_outcome_path,
                        {
                            "schema": "clay.physical-compact-batch-outcome/v1",
                            "input_sha256": input_sha,
                            "status": "fail",
                            "outcome": "unmeasurable",
                            "candidate": candidate.canonical_data(),
                            "candidate_sha256": candidate.digest,
                            "error_type": error_type,
                            "error": error_message,
                            "signoff_authority": False,
                        },
                    )
                rows.append(
                    PhysicalIntentCandidateEvidence(
                        candidate=candidate,
                        search_stage=stage,
                        outcome=outcome,
                        error_type=error_type,
                        error=error_message,
                        pdk_failure_evidence=failure_evidence,
                    )
                )
                evaluated_count += 1
                continue
            metrics_by_digest[candidate.digest] = metrics
            sources_by_digest[candidate.digest] = candidate_source
            rows.append(
                PhysicalIntentCandidateEvidence(
                    candidate=candidate,
                    search_stage=stage,
                    outcome="pass",
                    candidate_source=candidate_source,
                    candidate_source_sha256=_file_sha256(candidate_source),
                    metrics=metrics,
                    pdk_failure_evidence=failure_evidence,
                )
            )
            evaluated_count += 1

    measure(evaluation_candidates, stage="base")
    refinements: tuple[CompactIntentCandidate, ...] = ()
    if selected_sha256 is None and not bounded_batch:
        refinements = refine_measured_candidates(
            domain.candidates,
            metrics_by_digest,
            parent_count=maximum_signoffs,
            refinement_budget=int(domain_spec.refinement_budget),
        )
        measure(refinements, stage="refinement")
    elif selected_sha256 is not None and not exact_is_base:
        assert exact_candidate is not None
        refinements = (exact_candidate,)

    require_search_inputs()

    candidate_by_digest = {
        candidate.digest: candidate for candidate in (*domain.candidates, *refinements)
    }
    candidates = tuple(
        candidate_by_digest[digest] for digest in sorted(candidate_by_digest)
    )
    admissions_by_digest: dict[str, PhysicalSearchAdmission] = {}
    checkpoint: PhysicalIntentSearchCheckpoint | None = None
    if bounded_batch:
        assert physical_search_target is not None
        measured_candidates = tuple(
            candidate
            for candidate in candidates
            if candidate.digest in metrics_by_digest
        )
        admissions_by_digest = {
            candidate.digest: evaluate_physical_search_admission(
                physical_search_target,
                candidate,
                metrics_by_digest[candidate.digest],
            )
            for candidate in measured_candidates
        }
        cohort = (
            select_exact_replay_candidates(
                measured_candidates,
                metrics_by_digest,
                maximum=maximum_signoffs,
                admissions=admissions_by_digest,
                target=physical_search_target,
                target_sha256=physical_search_target.canonical_sha256,
            )
            if measured_candidates
            else ()
        )
        denominator = tuple(candidate.digest for candidate in evaluation_candidates)
        evaluated = tuple(row.candidate.digest for row in rows)
        remaining = denominator[len(evaluated) :]
        admitted_set = {
            digest
            for digest, admission in admissions_by_digest.items()
            if admission.admitted
        }
        admitted = tuple(digest for digest in evaluated if digest in admitted_set)
        checkpoint = PhysicalIntentSearchCheckpoint(
            target_sha256=physical_search_target.canonical_sha256,
            candidate_denominator_sha256=denominator,
            structure_denominator_sha256=tuple(
                sorted(
                    {
                        compact_structure_fingerprint(candidate)
                        for candidate in evaluation_candidates
                    }
                )
            ),
            evaluated_candidate_sha256=evaluated,
            remaining_candidate_sha256=remaining,
            admitted_candidate_sha256=admitted,
            outcome_by_candidate=tuple(
                (row.candidate.digest, row.outcome) for row in rows
            ),
            new_evaluation_count=new_evaluated_count,
            domain_exhausted=bool(domain.exhausted) and not remaining,
            status=("best_observed" if admitted else "no_admitted_candidate"),
        )
    else:
        cohort = select_routability_cohort(
            candidates,
            metrics_by_digest,
            maximum=maximum_signoffs,
        )
    if not cohort and not bounded_batch:
        raise ValueError(f"{pdk} compact search produced no measurable candidate")
    if selected_sha256 is not None:
        cohort = select_exact_candidate(candidates, selected_sha256)

    budget = physical_intent_probe_budget(cell)
    probe_evaluator_sha = (
        None
        if budget is None
        else _evaluator_sha256(
            probe_evaluator,
            evaluate_compact_intent_probe,
            probe_evaluator_sha256,
            label="probe_evaluator_sha256",
        )
    )
    probes_by_digest: dict[str, PhysicalIntentProbeResult] = {}
    if budget is not None:
        for index, candidate in enumerate(cohort, start=1):
            signature = str(
                metrics_by_digest[candidate.digest].get("physical_signature_sha256", "")
            )
            if not signature:
                raise ValueError(
                    f"{pdk} candidate {candidate.digest} has no physical signature"
                )
            input_sha = _input_sha256(
                {
                    "state_sha256": state_sha,
                    "top_cell": str(top_cell),
                    "pdk": str(pdk),
                    "physical_signature_sha256": signature,
                    "probe_budget": budget.canonical_data(),
                    "probe_evaluator_sha256": probe_evaluator_sha,
                }
            )
            probe_path = root / "route_probe" / signature / "probe.json"
            require_search_inputs()
            cached_probe = load_physical_intent_probe_cache(
                probe_path,
                input_sha256=input_sha,
                candidate_sha256=candidate.digest,
            )
            log(
                f"{pdk} PROBE {index}/{len(cohort)} "
                f"{'CACHE' if cached_probe is not None else 'START'} "
                f"candidate={candidate.digest[:12]}"
            )
            if cached_probe is None:
                try:
                    probe = probe_evaluator(
                        candidate_source=sources_by_digest[candidate.digest],
                        candidate_sha256=candidate.digest,
                        top_cell=top_cell,
                        pdk=str(pdk),
                        budget=budget,
                    )
                    probe = _validate_probe_result(
                        probe,
                        candidate_sha256=candidate.digest,
                    )
                except Exception as exc:
                    require_search_inputs()
                    error = f"{type(exc).__name__}: {exc}"
                    probe = PhysicalIntentProbeResult(
                        candidate_sha256=candidate.digest,
                        status="stopped",
                        phase="PROBE_ERROR",
                        state=type(exc).__name__,
                        reason="UNMEASURABLE",
                        net="",
                        evidence_sha256=hashlib.sha256(
                            error.encode("utf-8")
                        ).hexdigest(),
                        details=(("error", error),),
                    )
                else:
                    require_search_inputs()
                    queue_cache_write(
                        probe_path,
                        {
                            "schema": "clay.physical-intent-route-probe/v1",
                            "input_sha256": input_sha,
                            "physical_signature_sha256": signature,
                            "probe_budget": budget.canonical_data(),
                            "probe": probe.metrics(),
                        },
                    )
            else:
                try:
                    probe = _validate_probe_result(
                        cached_probe,
                        candidate_sha256=candidate.digest,
                    )
                except ValueError as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    probe = PhysicalIntentProbeResult(
                        candidate_sha256=candidate.digest,
                        status="stopped",
                        phase="PROBE_ERROR",
                        state=type(exc).__name__,
                        reason="UNMEASURABLE",
                        net="",
                        evidence_sha256=hashlib.sha256(
                            error.encode("utf-8")
                        ).hexdigest(),
                        details=(("error", error),),
                    )
            probes_by_digest[candidate.digest] = probe
        cohort = tuple(
            sorted(
                cohort,
                key=lambda candidate: physical_intent_probe_rank(
                    probes_by_digest[candidate.digest],
                    cheap_metrics=metrics_by_digest[candidate.digest],
                ),
            )
        )

    require_search_inputs()
    ledger_path = (
        root / "physical_intent_search_ledger.json"
        if selected_sha256 is None
        else root / "exact_candidate_ledgers" / f"{selected_sha256}.json"
    )
    result = PhysicalIntentSearchResult(
        pdk=str(pdk),
        top_cell=str(top_cell),
        state_sha256=state_sha,
        authoring_source_sha256=authoring_source_sha,
        cheap_evaluator_sha256=cheap_evaluator_sha,
        probe_evaluator_sha256=probe_evaluator_sha,
        seed=int(domain_spec.seed),
        domain_budget=int(domain_spec.budget),
        refinement_budget=int(domain_spec.refinement_budget),
        maximum_signoffs=int(maximum_signoffs),
        domain_exhausted=(
            bool(domain.exhausted)
            if checkpoint is None
            else checkpoint.domain_exhausted
        ),
        domain_total_combinations=int(domain.total_combinations),
        base_candidate_count=len(domain.candidates),
        refinement_count=len(refinements),
        cheap_evaluation_scope=(
            "bounded_target_prefix"
            if bounded_batch
            else (
                "materialized_candidate_replay"
                if materialized_candidate is not None
                else (
                    "full"
                    if selected_sha256 is None
                    else (
                        "exact_candidate"
                        if exact_is_base
                        else "exact_refinement_reconstruction"
                    )
                )
            )
        ),
        cheap_evaluated_count=evaluated_count,
        pdk_failure_rule_deck_sha256=failure_deck_sha,
        pdk_failure_rule_implementation_sha256=failure_implementation_sha,
        pdk_failure_rule_sha256=failure_rule_shas,
        rows=tuple(rows),
        candidates=candidates,
        cohort=cohort,
        candidate_source_by_digest=dict(sources_by_digest),
        probes_by_digest=dict(probes_by_digest),
        admissions_by_digest=dict(admissions_by_digest),
        physical_aliases=_physical_aliases(candidates, metrics_by_digest),
        probe_budget=budget,
        checkpoint=checkpoint,
        ledger_path=ledger_path,
    )
    require_search_inputs()
    written_cache_paths: list[Path] = []
    prior_ledger = ledger_path.read_bytes() if ledger_path.is_file() else None
    ledger_written = False
    try:
        for cache_path, payload in pending_cache_writes:
            _write_json(cache_path, payload)
            written_cache_paths.append(cache_path)
        require_search_inputs()
        _write_json(ledger_path, result.canonical_data())
        ledger_written = True
        require_search_inputs()
    except Exception:
        for cache_path in reversed(written_cache_paths):
            cache_path.unlink(missing_ok=True)
        if ledger_written:
            ledger_path.unlink(missing_ok=True)
            if prior_ledger is not None:
                temporary = ledger_path.with_name(ledger_path.name + ".rollback")
                try:
                    temporary.write_bytes(prior_ledger)
                    temporary.replace(ledger_path)
                finally:
                    temporary.unlink(missing_ok=True)
        raise
    return result


__all__ = [
    "CheapOutcome",
    "ExactCandidateFeedback",
    "PhysicalSearchVector",
    "PhysicalIntentCandidateEvidence",
    "PhysicalIntentInfeasible",
    "PhysicalIntentSearchCheckpoint",
    "PhysicalIntentSearchResult",
    "classify_exact_search_outcome",
    "exact_search_trajectory_data",
    "physical_intent_state_sha256",
    "prepare_physical_intent_search",
    "refine_measured_candidates",
    "select_exact_candidate",
    "select_exact_replay_candidates",
    "select_routability_cohort",
]
