"""Typed exact admission identity and fail-closed G6 result validation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import TypeAlias

from clay.flow.intent_experiment import IntentExperimentSpec
from clay.layout.intent_basis import MinimalityScope


_REQUIRED_EXACT_GATES = (
    "compile", "drc", "intent", "layout", "lvs", "pex", "sign"
)
_PRE_EXACT_ORACLE_STAGES = frozenset(f"G{index}" for index in range(6))
_MINIMALITY_SCOPES = frozenset(scope.value for scope in MinimalityScope)
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_DIRECT_SCHEMA = "clay.direct-exact-admission/v1"
_DIRECT_RESULT_SCHEMA = "clay.direct-current-source-exact/v1"
_DIRECT_SELECTION_KIND = "direct_current_source"


def _selection_sha256(selected_atom_ids: tuple[str, ...]) -> str:
    payload = json.dumps(
        list(selected_atom_ids), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ExactAdmission:
    spec: IntentExperimentSpec
    selected_atom_ids: tuple[str, ...]
    selected_basis_sha256: str
    selected_source_path: Path
    selected_source_sha256: str
    selected_candidate_sha256: str
    area_nm2: int
    bbox_nm: tuple[int, int, int, int]
    placement_receipt_sha256: str
    g5_receipt_sha256: str
    fact_set_sha256: str
    fact_authority_receipt_sha256: str
    fact_closure_receipt_sha256: str
    minimality_scope: str = MinimalityScope.UNPROVEN.value
    search_receipt_sha256: str | None = None
    basis_search_oracle_max_stage: str | None = None
    proposal_rank: int | None = None
    pareto_frontier_row: tuple[int, int, int, str] | None = None

    def __post_init__(self) -> None:
        for field in (
            "selected_basis_sha256",
            "selected_source_sha256",
            "selected_candidate_sha256",
            "placement_receipt_sha256",
            "g5_receipt_sha256",
            "fact_set_sha256",
            "fact_authority_receipt_sha256",
            "fact_closure_receipt_sha256",
        ):
            value = getattr(self, field)
            if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
                raise ValueError(f"{field} must be a lowercase SHA-256")
        if (
            type(self.minimality_scope) is not str
            or self.minimality_scope not in _MINIMALITY_SCOPES
        ):
            raise ValueError("minimality scope is not a declared MinimalityScope value")
        if (
            type(self.selected_atom_ids) is not tuple
            or not self.selected_atom_ids
            or any(type(item) is not str or not item for item in self.selected_atom_ids)
            or tuple(sorted(set(self.selected_atom_ids))) != self.selected_atom_ids
        ):
            raise ValueError("selected atom IDs must be a sorted unique non-empty tuple")
        search_fields = (
            self.search_receipt_sha256,
            self.basis_search_oracle_max_stage,
            self.proposal_rank,
            self.pareto_frontier_row,
        )
        if all(value is None for value in search_fields):
            if self.minimality_scope != MinimalityScope.UNPROVEN.value:
                raise ValueError(
                    "exact admission without search proof must remain unproven"
                )
            return
        if any(value is None for value in search_fields):
            raise ValueError("exact admission search proof must be complete or absent")
        if (
            type(self.search_receipt_sha256) is not str
            or _SHA256_RE.fullmatch(self.search_receipt_sha256) is None
        ):
            raise ValueError("search_receipt_sha256 must be a lowercase SHA-256")
        if (
            type(self.basis_search_oracle_max_stage) is not str
            or self.basis_search_oracle_max_stage not in _PRE_EXACT_ORACLE_STAGES
        ):
            raise ValueError("basis-search oracle max stage must be one of G0 through G5")
        if type(self.proposal_rank) is not int or self.proposal_rank < 0:
            raise ValueError("proposal rank must be an exact non-negative integer")
        if (
            self.proposal_rank > 0
            and self.minimality_scope != MinimalityScope.UNPROVEN.value
        ):
            raise ValueError("ranked Pareto proposals beyond rank zero must be unproven")
        row = self.pareto_frontier_row
        if type(row) is not tuple or len(row) != 4:
            raise ValueError("Pareto frontier row must be an exact canonical 4-tuple")
        cost_bits, atom_count, area_nm2, selection_sha256 = row
        if (
            type(cost_bits) is not int
            or cost_bits < 0
            or type(atom_count) is not int
            or atom_count <= 0
            or type(area_nm2) is not int
            or area_nm2 <= 0
            or type(selection_sha256) is not str
            or _SHA256_RE.fullmatch(selection_sha256) is None
        ):
            raise ValueError("Pareto frontier row is not canonical")
        if (
            atom_count != len(self.selected_atom_ids)
            or area_nm2 != self.area_nm2
            or selection_sha256 != _selection_sha256(self.selected_atom_ids)
        ):
            raise ValueError("Pareto frontier row does not bind the admitted candidate")

    @property
    def has_search_proof(self) -> bool:
        """Whether admission carries a complete bounded-search/Pareto proof."""

        return self.search_receipt_sha256 is not None


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _require_sha256(value: object, *, field: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be one lowercase SHA-256")
    return value


def _require_absolute_file_path(
    value: object, *, field: str, suffix: str
) -> Path:
    if not isinstance(value, Path) or not value.is_absolute():
        raise TypeError(f"{field} must be one absolute Path")
    if value.suffix.casefold() != suffix or value.resolve(strict=False) != value:
        raise ValueError(f"{field} must be one canonical absolute {suffix} path")
    return value


@dataclass(frozen=True, slots=True)
class DirectExactAdmission:
    """Exact admission for one current source without a basis-search claim.

    This type is deliberately separate from :class:`ExactAdmission`: a circuit
    that has no typed G0--G5 basis receipts must not manufacture fact, placement,
    route, or search digests merely to enter the exact flow.  It binds the two
    real design inputs and leaves minimality explicitly unproven.
    """

    circuit_id: str
    pdk_id: str
    source_path: Path
    source_sha256: str
    source_input_path: Path
    source_input_sha256: str
    selected_candidate_sha256: str
    acceptance_policy_sha256: str
    implementation_sha256: tuple[tuple[str, str], ...]
    pdk_snapshot_sha256: str
    pdk_installed_authority_sha256: str
    budget_s: int
    output_root: str
    forbidden_side_effects: tuple[str, ...]
    selection_kind: str = _DIRECT_SELECTION_KIND
    minimality_scope: str = MinimalityScope.UNPROVEN.value

    def __post_init__(self) -> None:
        for field in ("circuit_id", "pdk_id"):
            value = getattr(self, field)
            if type(value) is not str or not value or value != value.strip():
                raise TypeError(f"{field} must be one canonical string")
        _require_absolute_file_path(
            self.source_path, field="source_path", suffix=".clay"
        )
        if (
            not isinstance(self.source_input_path, Path)
            or not self.source_input_path.is_absolute()
            or self.source_input_path.resolve(strict=False) != self.source_input_path
            or self.source_input_path.suffix.casefold() not in {".cir", ".spice"}
        ):
            raise ValueError(
                "source_input_path must be one canonical absolute CIR/SPICE path"
            )
        for field in (
            "source_sha256",
            "source_input_sha256",
            "selected_candidate_sha256",
            "acceptance_policy_sha256",
            "pdk_snapshot_sha256",
            "pdk_installed_authority_sha256",
        ):
            _require_sha256(getattr(self, field), field=field)
        rows = self.implementation_sha256
        if (
            type(rows) is not tuple
            or not rows
            or rows != tuple(sorted(set(rows)))
        ):
            raise ValueError(
                "implementation_sha256 must be one sorted unique non-empty tuple"
            )
        names: set[str] = set()
        for row in rows:
            if type(row) is not tuple or len(row) != 2:
                raise TypeError("implementation rows must be exact 2-tuples")
            name, digest = row
            if (
                type(name) is not str
                or not name
                or name.startswith(("/", "\\"))
                or "\\" in name
                or ".." in Path(name).parts
                or name in names
            ):
                raise ValueError("implementation path is not canonical repository-relative")
            names.add(name)
            _require_sha256(digest, field="implementation digest")
        if type(self.budget_s) is not int or self.budget_s <= 0:
            raise TypeError("direct exact budget must be one positive integer")
        output = Path(self.output_root)
        if (
            type(self.output_root) is not str
            or not output.is_absolute()
            or str(output.resolve(strict=False)) != self.output_root
        ):
            raise ValueError("direct exact output_root must be canonical and absolute")
        if (
            type(self.forbidden_side_effects) is not tuple
            or tuple(sorted(set(self.forbidden_side_effects)))
            != self.forbidden_side_effects
            or any(type(item) is not str or not item for item in self.forbidden_side_effects)
        ):
            raise ValueError("forbidden side effects must be a sorted unique tuple")
        if self.selection_kind != _DIRECT_SELECTION_KIND:
            raise ValueError("direct exact selection kind must be direct_current_source")
        if self.minimality_scope != MinimalityScope.UNPROVEN.value:
            raise ValueError("direct exact minimality must remain unproven")

    @property
    def output_path(self) -> Path:
        return Path(self.output_root)

    def require_fresh_output_root(self) -> None:
        if self.output_path.exists():
            raise FileExistsError(
                f"refusing to reuse direct exact output root: {self.output_root}"
            )

    def canonical_data(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema": _DIRECT_SCHEMA,
            "circuit_id": self.circuit_id,
            "pdk_id": self.pdk_id,
            "source_path": str(self.source_path),
            "source_sha256": self.source_sha256,
            "source_input_path": str(self.source_input_path),
            "source_input_sha256": self.source_input_sha256,
            "selected_candidate_sha256": self.selected_candidate_sha256,
            "acceptance_policy_sha256": self.acceptance_policy_sha256,
            "implementation_sha256": [list(row) for row in self.implementation_sha256],
            "pdk_snapshot_sha256": self.pdk_snapshot_sha256,
            "pdk_installed_authority_sha256": self.pdk_installed_authority_sha256,
            "budget_s": self.budget_s,
            "output_root": self.output_root,
            "forbidden_side_effects": list(self.forbidden_side_effects),
            "selection_kind": self.selection_kind,
            "minimality_scope": self.minimality_scope,
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical_bytes(self.canonical_data())).hexdigest()

    @classmethod
    def from_data(cls, data: object) -> "DirectExactAdmission":
        expected = {
            "schema",
            "circuit_id",
            "pdk_id",
            "source_path",
            "source_sha256",
            "source_input_path",
            "source_input_sha256",
            "selected_candidate_sha256",
            "acceptance_policy_sha256",
            "implementation_sha256",
            "pdk_snapshot_sha256",
            "pdk_installed_authority_sha256",
            "budget_s",
            "output_root",
            "forbidden_side_effects",
            "selection_kind",
            "minimality_scope",
        }
        if type(data) is not dict or set(data) != expected:
            raise ValueError("direct exact admission canonical fields differ")
        if data.get("schema") != _DIRECT_SCHEMA:
            raise ValueError("direct exact admission schema differs")
        rows = data.get("implementation_sha256")
        effects = data.get("forbidden_side_effects")
        if (
            type(rows) is not list
            or any(type(row) is not list or len(row) != 2 for row in rows)
            or type(effects) is not list
        ):
            raise TypeError("direct exact tuple fields are not canonical lists")
        result = cls(
            circuit_id=data["circuit_id"],  # type: ignore[arg-type]
            pdk_id=data["pdk_id"],  # type: ignore[arg-type]
            source_path=Path(data["source_path"]),  # type: ignore[arg-type]
            source_sha256=data["source_sha256"],  # type: ignore[arg-type]
            source_input_path=Path(data["source_input_path"]),  # type: ignore[arg-type]
            source_input_sha256=data["source_input_sha256"],  # type: ignore[arg-type]
            selected_candidate_sha256=data["selected_candidate_sha256"],  # type: ignore[arg-type]
            acceptance_policy_sha256=data["acceptance_policy_sha256"],  # type: ignore[arg-type]
            implementation_sha256=tuple((row[0], row[1]) for row in rows),  # type: ignore[misc]
            pdk_snapshot_sha256=data["pdk_snapshot_sha256"],  # type: ignore[arg-type]
            pdk_installed_authority_sha256=data["pdk_installed_authority_sha256"],  # type: ignore[arg-type]
            budget_s=data["budget_s"],  # type: ignore[arg-type]
            output_root=data["output_root"],  # type: ignore[arg-type]
            forbidden_side_effects=tuple(effects),
            selection_kind=data["selection_kind"],  # type: ignore[arg-type]
            minimality_scope=data["minimality_scope"],  # type: ignore[arg-type]
        )
        if result.canonical_data() != data:
            raise ValueError("direct exact admission replay is not canonical")
        return result


ExactAdmissionLike: TypeAlias = ExactAdmission | DirectExactAdmission


def _validate_exact_physical_result(
    result: dict[str, object], admission: ExactAdmission | None
) -> str:
    """Validate the hard G6 physical gates for the selected candidate."""

    gates = result.get("gates")
    if type(gates) is not dict or any(
        gates.get(name) != "pass" for name in _REQUIRED_EXACT_GATES
    ):
        raise ValueError("exact hard gate set is not all pass")
    drc = result.get("drc")
    if (
        type(drc) is not dict
        or drc.get("status") != "pass"
        or drc.get("error_count") != 0
    ):
        raise ValueError("DRC is not zero-clean")
    lvs = result.get("lvs")
    if type(lvs) is not dict or lvs != {
        "status": "pass",
        "verdict": "MATCHED",
        "scope": "full",
    }:
        raise ValueError("LVS is not a full MATCHED result")
    pex = result.get("pex")
    if type(pex) is not dict or (
        pex.get("status") != "pass"
        or pex.get("claim_scope") != "physical_only"
        or pex.get("calibrated") is not False
        or pex.get("field_solver") is not False
    ):
        raise ValueError("PEX scope is not honest physical-only evidence")
    intent = result.get("intent")
    if type(intent) is not dict or (
        intent.get("status") != "pass"
        or type(intent.get("n_total")) is not int
        or intent["n_total"] <= 0
        or intent.get("n_pass") != intent["n_total"]
        or any(
            intent.get(name) != 0
            for name in ("n_fail", "n_unmeasurable", "n_unsupported")
        )
        or intent.get("signed") is not True
    ):
        raise ValueError("intent obligations are not exhaustive and signed")
    artifact = result.get("exact_artifact")
    if type(artifact) is not dict or (
        artifact.get("status") != "pass"
        or artifact.get("exhaustive") is not True
        or artifact.get("containment") != "pass"
        or type(artifact.get("route_manifest_entry_count")) is not int
        or artifact["route_manifest_entry_count"] <= 0
    ):
        raise ValueError("exact artifact or containment scope failed")
    side_effects = result.get("side_effects")
    if type(side_effects) is not dict or side_effects != {
        "png_files": 0,
        "png_json_references": 0,
        "publisher": "not_run",
        "search_validator": "not_run",
        "transaction": "not_run",
    }:
        raise ValueError("forbidden side effect was observed")
    if (
        admission is not None
        and admission.has_search_proof
        and result.get("area_nm2") != admission.area_nm2
    ):
        raise ValueError("exact GDS area differs from admitted placement")
    if admission is not None and not admission.has_search_proof:
        _exact_result_geometry(result)
    optimality = result.get("placement_optimality")
    if optimality not in {"optimal", "unproven_feasible"}:
        raise ValueError("placement optimality status is malformed")
    return optimality


def _exact_result_geometry(
    result: dict[str, object],
) -> tuple[int, list[int]]:
    area = result.get("area_nm2")
    bbox = result.get("bbox_nm")
    if (
        type(area) is not int
        or area <= 0
        or type(bbox) is not list
        or len(bbox) != 4
        or any(type(value) is not int for value in bbox)
        or bbox[0] >= bbox[2]
        or bbox[1] >= bbox[3]
        or area != (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    ):
        raise ValueError("exact GDS area or bbox is not canonical")
    return area, bbox


def validate_exact_result(
    result: dict[str, object], admission: ExactAdmission
) -> dict[str, object]:
    """Validate G6 without conflating physical pass and minimality scope."""

    optimality = _validate_exact_physical_result(result, admission)

    if admission.has_search_proof:
        area_nm2 = admission.area_nm2
        bbox_nm = list(admission.bbox_nm)
    else:
        area_nm2, bbox_nm = _exact_result_geometry(result)

    receipt: dict[str, object] = {
        "schema": "clay.bandgap-minimal-intent-exact/v3",
        "status": "pass",
        "physical_verdict": "pass",
        "placement_optimality": optimality,
        "minimality_scope": admission.minimality_scope,
        "selected_candidate_validation_max_stage": "G6",
        "basis_sha256": admission.selected_basis_sha256,
        "selected_atom_ids": list(admission.selected_atom_ids),
        "source_sha256": admission.spec.source_sha256,
        "materialized_source_sha256": admission.selected_source_sha256,
        "candidate_sha256": admission.selected_candidate_sha256,
        "acceptance_policy_sha256": admission.spec.acceptance_policy_sha256,
        "spec_sha256": admission.spec.sha256,
        "area_nm2": area_nm2,
        "bbox_nm": bbox_nm,
        "placement_receipt_sha256": admission.placement_receipt_sha256,
        "g5_receipt_sha256": admission.g5_receipt_sha256,
        "fact_set_sha256": admission.fact_set_sha256,
        "fact_authority_receipt_sha256": (
            admission.fact_authority_receipt_sha256
        ),
        "fact_closure_receipt_sha256": admission.fact_closure_receipt_sha256,
        "gates": result,
    }
    if admission.has_search_proof:
        assert admission.basis_search_oracle_max_stage is not None
        assert admission.proposal_rank is not None
        assert admission.pareto_frontier_row is not None
        assert admission.search_receipt_sha256 is not None
        receipt.update(
            {
                "basis_search_oracle_max_stage": (
                    admission.basis_search_oracle_max_stage
                ),
                "minimality_oracle_max_stage": (
                    admission.basis_search_oracle_max_stage
                ),
                "proposal_rank": admission.proposal_rank,
                "pareto_frontier_row": list(admission.pareto_frontier_row),
                "search_receipt_sha256": admission.search_receipt_sha256,
            }
        )
    else:
        receipt["selection_proof"] = {
            "kind": "direct",
            "minimality_scope": MinimalityScope.UNPROVEN.value,
        }
    return receipt


def _validate_direct_exact_result(
    result: dict[str, object], admission: DirectExactAdmission
) -> dict[str, object]:
    optimality = _validate_exact_physical_result(result, None)
    area, bbox = _exact_result_geometry(result)
    return {
        "schema": _DIRECT_RESULT_SCHEMA,
        "status": "pass",
        "physical_verdict": "pass",
        "placement_optimality": optimality,
        "minimality_scope": MinimalityScope.UNPROVEN.value,
        "selected_candidate_validation_max_stage": "G6",
        "selection_proof": {
            "kind": _DIRECT_SELECTION_KIND,
            "minimality_scope": MinimalityScope.UNPROVEN.value,
        },
        "admission_sha256": admission.sha256,
        "circuit_id": admission.circuit_id,
        "pdk_id": admission.pdk_id,
        "source_sha256": admission.source_sha256,
        "materialized_source_sha256": admission.source_sha256,
        "source_input_sha256": admission.source_input_sha256,
        "candidate_sha256": admission.selected_candidate_sha256,
        "acceptance_policy_sha256": admission.acceptance_policy_sha256,
        "area_nm2": area,
        "bbox_nm": bbox,
        "gates": result,
    }


def normalize_exact_result(
    result: dict[str, object], admission: ExactAdmissionLike
) -> dict[str, object]:
    """Normalize one exact physical result without upgrading its proof scope."""

    if type(admission) is ExactAdmission:
        return validate_exact_result(result, admission)
    if type(admission) is DirectExactAdmission:
        return _validate_direct_exact_result(result, admission)
    raise TypeError("exact result requires one typed exact admission")


def persist_exact_admission(admission: ExactAdmissionLike) -> Path:
    """Persist the typed admission inside its already-created fresh root."""

    if type(admission) is ExactAdmission:
        root = admission.spec.output_path
        name = "experiment_spec.json"
        data = admission.spec.canonical_data()
    elif type(admission) is DirectExactAdmission:
        root = admission.output_path
        name = "direct_exact_admission.json"
        data = admission.canonical_data()
    else:
        raise TypeError("cannot persist an untyped exact admission")
    if not root.is_dir():
        raise ValueError("exact output root must exist before admission persistence")
    path = root / name
    if path.exists():
        raise FileExistsError(f"refusing to overwrite exact admission: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_canonical_bytes(data) + b"\n")
    temporary.replace(path)
    return path


def normalize_and_persist_exact_result(
    result: dict[str, object], admission: ExactAdmissionLike
) -> tuple[dict[str, object], Path]:
    """Validate, normalize, and atomically persist one focused exact receipt."""

    receipt = normalize_exact_result(result, admission)
    root = (
        admission.spec.output_path
        if type(admission) is ExactAdmission
        else admission.output_path
    )
    path = root / "focused_validation_receipt.json"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite exact receipt: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_canonical_bytes(receipt) + b"\n")
    temporary.replace(path)
    return receipt, path


__all__ = [
    "DirectExactAdmission",
    "ExactAdmission",
    "ExactAdmissionLike",
    "normalize_and_persist_exact_result",
    "normalize_exact_result",
    "persist_exact_admission",
    "validate_exact_result",
]
