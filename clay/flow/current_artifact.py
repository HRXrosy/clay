"""Immutable identity for one freshly generated exact GDS artifact.

The receipt is deliberately downstream of exact admission.  Its factory replays
the typed admission against the focused exact result, the canonical flow summary,
the emitter-minted exact-artifact receipt, the current source/implementation
bytes, and a captured PDK snapshot.  Renderers may consume this receipt; they may
not replace it with a path or an untyped JSON mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

from clay.flow.exact_intent_result import (
    DirectExactAdmission,
    ExactAdmission,
    ExactAdmissionLike,
    normalize_exact_result,
    validate_exact_result,
)
from clay.flow.intent_experiment import ExperimentStage, IntentExperimentSpec
from clay.pdk import get_pdk
from clay.pdk._source_snapshot import PdkSourceSnapshot


_SCHEMA = "clay.current-artifact/v1"
_EXACT_ADMISSION_SCHEMA = "clay.exact-admission/v1"
_EXACT_ARTIFACT_SCHEMA = "clay.exact-layout-artifact/v1"
_CIRCUIT_RE = re.compile(r"[a-z][a-z0-9_.-]*\Z")
_RECEIPT_SCHEMA_RE = re.compile(r"clay\.[a-z][a-z0-9_.-]*/v[1-9][0-9]*\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_REQUIRED_FLOW_STEPS = frozenset(
    {"compile", "layout", "drc", "lvs", "pex", "intent", "sign"}
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be one lowercase SHA-256")
    return value


def _require_absolute_path(value: object, *, field: str, suffix: str) -> Path:
    if type(value) is not str or not value or value != value.strip():
        raise TypeError(f"{field} must be one exact non-empty path string")
    path = Path(value)
    if (
        not path.is_absolute()
        or path.suffix.casefold() != suffix
        or str(path.resolve(strict=False)) != value
    ):
        raise ValueError(f"{field} must be one canonical absolute {suffix} path")
    return path


def _require_absolute_artifact_path(value: object, *, field: str) -> Path:
    if type(value) is not str or not value or value != value.strip():
        raise TypeError(f"{field} must be one exact non-empty path string")
    path = Path(value)
    if not path.is_absolute() or str(path.resolve(strict=False)) != value:
        raise ValueError(f"{field} must be one canonical absolute artifact path")
    return path


def _require_absolute_directory(value: object, *, field: str) -> Path:
    if type(value) is not str or not value or value != value.strip():
        raise TypeError(f"{field} must be one exact non-empty path string")
    path = Path(value)
    if not path.is_absolute() or str(path.resolve(strict=False)) != value:
        raise ValueError(f"{field} must be one canonical absolute directory path")
    return path


def _read_json(path: Path, *, field: str) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"{field} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{field} is not readable canonical JSON") from exc
    if type(value) is not dict:
        raise TypeError(f"{field} must contain one exact JSON object")
    return value


def _confined(path: Path, root: Path, *, field: str) -> Path:
    try:
        resolved_root = root.resolve(strict=True)
        resolved = path.resolve(strict=True)
        resolved.relative_to(resolved_root)
    except (OSError, ValueError) as exc:
        raise ValueError(f"{field} escapes the fresh output root") from exc
    if not resolved.is_file():
        raise ValueError(f"{field} is not one regular file")
    return resolved


def _implementation_rows(value: object) -> tuple[tuple[str, str], ...]:
    if type(value) is not tuple or not value:
        raise TypeError("implementation_sha256 must be one non-empty exact tuple")
    rows: list[tuple[str, str]] = []
    for raw in value:
        if type(raw) is not tuple or len(raw) != 2:
            raise TypeError("implementation rows must be exact 2-tuples")
        path, digest = raw
        resolved = _require_absolute_artifact_path(path, field="implementation path")
        rows.append((str(resolved), _require_sha256(digest, field="implementation digest")))
    result = tuple(rows)
    if result != tuple(sorted(set(result))) or len({path for path, _ in result}) != len(
        result
    ):
        raise ValueError("implementation rows must be sorted with unique paths")
    return result


def _implementation_rows_from_data(value: object) -> tuple[tuple[str, str], ...]:
    if type(value) is not list or any(
        type(row) is not list or len(row) != 2 for row in value
    ):
        raise TypeError("implementation canonical data must be a list of 2-lists")
    return tuple((row[0], row[1]) for row in value)  # type: ignore[misc]


def _exact_admission_data(admission: ExactAdmission) -> dict[str, object]:
    """Return the complete canonical exact-admission identity."""

    if type(admission) is not ExactAdmission:
        raise TypeError("exact admission must be one ExactAdmission")
    admission.__post_init__()
    spec = admission.spec
    if type(spec) is not IntentExperimentSpec or spec.stage is not ExperimentStage.EXACT:
        raise TypeError("exact admission must carry one exact experiment spec")
    _require_absolute_path(
        spec.source_path, field="exact admission source_path", suffix=".clay"
    )
    _require_absolute_directory(
        spec.output_root, field="exact admission output_root"
    )
    selected_source = _require_absolute_path(
        str(admission.selected_source_path),
        field="selected_source_path",
        suffix=".clay",
    )
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
        _require_sha256(getattr(admission, field), field=field)
    if admission.selected_basis_sha256 != spec.basis_sha256:
        raise ValueError("exact admission basis differs from its experiment spec")
    if (
        type(admission.area_nm2) is not int
        or admission.area_nm2 <= 0
        or type(admission.bbox_nm) is not tuple
        or len(admission.bbox_nm) != 4
        or any(type(value) is not int for value in admission.bbox_nm)
    ):
        raise ValueError("exact admission area or bbox is not canonical")
    result: dict[str, object] = {
        "schema": _EXACT_ADMISSION_SCHEMA,
        "spec": spec.canonical_data(),
        "selected_atom_ids": list(admission.selected_atom_ids),
        "selected_basis_sha256": admission.selected_basis_sha256,
        "selected_source_path": str(selected_source),
        "selected_source_sha256": admission.selected_source_sha256,
        "selected_candidate_sha256": admission.selected_candidate_sha256,
        "area_nm2": admission.area_nm2,
        "bbox_nm": list(admission.bbox_nm),
        "minimality_scope": admission.minimality_scope,
        "placement_receipt_sha256": admission.placement_receipt_sha256,
        "g5_receipt_sha256": admission.g5_receipt_sha256,
        "fact_set_sha256": admission.fact_set_sha256,
        "fact_authority_receipt_sha256": (
            admission.fact_authority_receipt_sha256
        ),
        "fact_closure_receipt_sha256": admission.fact_closure_receipt_sha256,
    }
    if admission.has_search_proof:
        assert admission.search_receipt_sha256 is not None
        assert admission.basis_search_oracle_max_stage is not None
        assert admission.proposal_rank is not None
        assert admission.pareto_frontier_row is not None
        result.update(
            {
                "search_receipt_sha256": admission.search_receipt_sha256,
                "basis_search_oracle_max_stage": (
                    admission.basis_search_oracle_max_stage
                ),
                "proposal_rank": admission.proposal_rank,
                "pareto_frontier_row": list(admission.pareto_frontier_row),
            }
        )
    else:
        result["selection_proof"] = {
            "kind": "direct",
            "minimality_scope": "unproven",
        }
    return result


def _exact_admission_from_data(value: object) -> ExactAdmission:
    common = {
        "schema",
        "spec",
        "selected_atom_ids",
        "selected_basis_sha256",
        "selected_source_path",
        "selected_source_sha256",
        "selected_candidate_sha256",
        "area_nm2",
        "bbox_nm",
        "minimality_scope",
        "placement_receipt_sha256",
        "g5_receipt_sha256",
        "fact_set_sha256",
        "fact_authority_receipt_sha256",
        "fact_closure_receipt_sha256",
    }
    search_fields = {
        "search_receipt_sha256",
        "basis_search_oracle_max_stage",
        "proposal_rank",
        "pareto_frontier_row",
    }
    direct_fields = {"selection_proof"}
    allowed_fields = {
        frozenset(common | search_fields),
        frozenset(common | direct_fields),
    }
    if type(value) is not dict or frozenset(value) not in allowed_fields:
        raise ValueError("exact admission canonical fields differ")
    if value.get("schema") != _EXACT_ADMISSION_SCHEMA:
        raise ValueError("exact admission schema is not canonical")
    atoms = value.get("selected_atom_ids")
    bbox = value.get("bbox_nm")
    has_search = "search_receipt_sha256" in value
    frontier = value.get("pareto_frontier_row")
    if (
        type(atoms) is not list
        or type(bbox) is not list
        or len(bbox) != 4
        or (
            has_search
            and (type(frontier) is not list or len(frontier) != 4)
        )
    ):
        raise TypeError("exact admission tuple fields are not canonical lists")
    if not has_search and value.get("selection_proof") != {
        "kind": "direct",
        "minimality_scope": "unproven",
    }:
        raise ValueError("direct exact admission selection proof differs")
    result = ExactAdmission(
        spec=IntentExperimentSpec.from_data(value["spec"]),
        selected_atom_ids=tuple(atoms),
        selected_basis_sha256=value["selected_basis_sha256"],  # type: ignore[arg-type]
        selected_source_path=Path(value["selected_source_path"]),  # type: ignore[arg-type]
        selected_source_sha256=value["selected_source_sha256"],  # type: ignore[arg-type]
        selected_candidate_sha256=value["selected_candidate_sha256"],  # type: ignore[arg-type]
        area_nm2=value["area_nm2"],  # type: ignore[arg-type]
        bbox_nm=tuple(bbox),  # type: ignore[arg-type]
        minimality_scope=value["minimality_scope"],  # type: ignore[arg-type]
        search_receipt_sha256=value.get("search_receipt_sha256"),  # type: ignore[arg-type]
        placement_receipt_sha256=value["placement_receipt_sha256"],  # type: ignore[arg-type]
        g5_receipt_sha256=value["g5_receipt_sha256"],  # type: ignore[arg-type]
        fact_set_sha256=value["fact_set_sha256"],  # type: ignore[arg-type]
        fact_authority_receipt_sha256=value[
            "fact_authority_receipt_sha256"
        ],  # type: ignore[arg-type]
        fact_closure_receipt_sha256=value[
            "fact_closure_receipt_sha256"
        ],  # type: ignore[arg-type]
        basis_search_oracle_max_stage=value.get("basis_search_oracle_max_stage"),  # type: ignore[arg-type]
        proposal_rank=value.get("proposal_rank"),  # type: ignore[arg-type]
        pareto_frontier_row=(
            tuple(frontier) if has_search else None
        ),  # type: ignore[arg-type]
    )
    if _exact_admission_data(result) != value:
        raise ValueError("exact admission replay data is not canonical")
    return result


def _admission_data(admission: ExactAdmissionLike) -> dict[str, object]:
    if type(admission) is ExactAdmission:
        return _exact_admission_data(admission)
    if type(admission) is DirectExactAdmission:
        return admission.canonical_data()
    raise TypeError("current artifact requires one typed exact admission")


def _admission_from_data(value: object) -> ExactAdmissionLike:
    if type(value) is not dict:
        raise TypeError("exact admission canonical data must be one object")
    if value.get("schema") == "clay.direct-exact-admission/v1":
        return DirectExactAdmission.from_data(value)
    return _exact_admission_from_data(value)


def _live_pdk_snapshot(pdk_id: str) -> PdkSourceSnapshot:
    try:
        adapter = get_pdk(pdk_id)
    except Exception as exc:
        raise ValueError(f"live PDK adapter is unavailable: {pdk_id}") from exc
    snapshot = adapter.source_snapshot()
    if (
        type(snapshot) is not PdkSourceSnapshot
        or snapshot.pdk_id != pdk_id
        or snapshot.installed_authority_sha256 is None
    ):
        raise ValueError("live PDK snapshot lacks exact installed authority")
    return snapshot


def _require_current_file(path: Path, digest: str, *, field: str) -> None:
    if not path.is_file():
        raise ValueError(f"{field} is missing: {path}")
    if _sha256_file(path) != digest:
        raise ValueError(f"{field} SHA-256 differs from the receipt")


def _layout_step(
    summary: dict[str, object], *, flow_spec_key: str
) -> dict[str, object]:
    if set(summary) != {"spec_key", "verdict", "work_dir", "steps"}:
        raise ValueError("run summary schema is not exact")
    if summary["spec_key"] != flow_spec_key:
        raise ValueError(
            "run summary circuit/flow spec differs from current artifact"
        )
    if summary["verdict"] != "pass":
        raise ValueError("run summary is not a physical pass")
    raw_steps = summary["steps"]
    if type(raw_steps) is not list or not raw_steps:
        raise TypeError("run summary steps must be one non-empty exact list")
    by_name: dict[str, dict[str, object]] = {}
    expected_keys = {"name", "status", "summary", "blocking", "data", "artifacts"}
    for raw in raw_steps:
        if type(raw) is not dict or set(raw) != expected_keys:
            raise ValueError("run summary step rows are not exact")
        name = raw["name"]
        if type(name) is not str or not name or name in by_name:
            raise ValueError("run summary step names are not unique")
        by_name[name] = raw
    if not _REQUIRED_FLOW_STEPS <= set(by_name) or any(
        by_name[name]["status"] != "pass" for name in _REQUIRED_FLOW_STEPS
    ):
        raise ValueError("run summary exact gate set is not all pass")
    return by_name["layout"]


def _summary_artifacts(
    summary: dict[str, object],
    *,
    flow_spec_key: str,
    output_root: Path,
    run_summary_path: Path,
) -> tuple[Path, Path, dict[str, object]]:
    work_dir = Path(summary.get("work_dir", ""))
    if (
        not work_dir.is_absolute()
        or work_dir.resolve(strict=False) != run_summary_path.parent
    ):
        raise ValueError("run summary work_dir differs from its current path")
    try:
        work_dir.resolve(strict=True).relative_to(output_root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise ValueError("run summary work_dir escapes the fresh output root") from exc
    layout = _layout_step(summary, flow_spec_key=flow_spec_key)
    data = layout["data"]
    artifacts = layout["artifacts"]
    if type(data) is not dict or type(artifacts) is not dict:
        raise TypeError("layout summary lacks exact data/artifact objects")
    if set(artifacts) < {"layout_gds", "exact_artifact_receipt"}:
        raise ValueError("layout summary lacks GDS or exact-artifact paths")
    raw_gds = artifacts["layout_gds"]
    raw_exact = artifacts["exact_artifact_receipt"]
    if type(raw_gds) is not str or type(raw_exact) is not str:
        raise TypeError("layout summary artifact paths must be exact strings")
    gds_path = _confined(Path(raw_gds), output_root, field="GDS artifact")
    exact_artifact_path = _confined(
        Path(raw_exact), output_root, field="exact artifact receipt"
    )
    exact_artifact = data.get("exact_artifact")
    if type(exact_artifact) is not dict:
        raise TypeError("layout summary lacks exact artifact evidence")
    return gds_path, exact_artifact_path, exact_artifact


def _verify_exact_artifact(
    artifact: dict[str, object],
    *,
    path: Path,
    gds_path: Path,
    gds_sha256: str,
    candidate_sha256: str,
) -> None:
    current = _read_json(path, field="exact artifact receipt")
    if current != artifact:
        raise ValueError("run summary exact artifact differs from its receipt")
    scope = artifact.get("scope")
    if (
        artifact.get("schema") != _EXACT_ARTIFACT_SCHEMA
        or artifact.get("status") != "pass"
        or type(scope) is not dict
        or scope.get("exhaustive") is not True
        or type(artifact.get("route_manifest_entry_count")) is not int
        or artifact["route_manifest_entry_count"] <= 0
        or artifact.get("candidate_sha256") != candidate_sha256
        or artifact.get("exact_gds") != str(gds_path)
        or artifact.get("exact_gds_sha256") != gds_sha256
    ):
        raise ValueError("exact artifact identity or exhaustive scope differs")


def _verify_exact_result(
    result: dict[str, object],
    *,
    receipt: "CurrentArtifactReceipt",
    admission: ExactAdmissionLike,
    summary_artifact: dict[str, object],
) -> None:
    gates = result.get("gates")
    if type(gates) is not dict:
        raise TypeError("exact result lacks one physical gate object")
    _verify_exact_result_core(result, admission=admission)
    if type(admission) is ExactAdmission:
        admission_identity_ok = result.get("spec_sha256") == admission.spec.sha256
    else:
        admission_identity_ok = (
            result.get("admission_sha256") == admission.sha256
            and result.get("circuit_id") == admission.circuit_id
        )
    if (
        result.get("status") != "pass"
        or result.get("physical_verdict") != "pass"
        or result.get("selected_candidate_validation_max_stage") != "G6"
        or not admission_identity_ok
        or result.get("source_sha256") != receipt.source_sha256
        or result.get("materialized_source_sha256")
        != receipt.materialized_source_sha256
    ):
        raise ValueError("exact admission identity differs from current artifact")
    artifact = gates.get("exact_artifact")
    if (
        type(artifact) is not dict
        or artifact.get("status") != "pass"
        or artifact.get("exhaustive") is not True
        or artifact.get("containment") != "pass"
        or artifact.get("exact_gds") != receipt.gds_path
        or artifact.get("exact_gds_sha256") != receipt.gds_sha256
        or summary_artifact.get("exact_gds") != receipt.gds_path
        or summary_artifact.get("exact_gds_sha256") != receipt.gds_sha256
    ):
        raise ValueError("exact result GDS or artifact scope differs")


def _verify_exact_result_core(
    result: dict[str, object], *, admission: ExactAdmissionLike
) -> None:
    """Replay the generic G6 core while preserving circuit receipt extensions."""

    gates = result.get("gates")
    if type(gates) is not dict:
        raise TypeError("exact result lacks one physical gate object")
    replayed = normalize_exact_result(gates, admission)
    schema = result.get("schema")
    if type(schema) is not str or _RECEIPT_SCHEMA_RE.fullmatch(schema) is None:
        raise ValueError("focused exact receipt schema is not canonical")
    # ``normalize_exact_result`` defines the circuit-independent semantic core.
    # Circuit runners may name that receipt and add typed closure fields of
    # their own; the complete focused receipt is still hash-bound below.
    replayed_core = {key: value for key, value in replayed.items() if key != "schema"}
    observed_core = {key: result.get(key) for key in replayed_core}
    if observed_core != replayed_core:
        raise ValueError("focused exact receipt differs from exact admission replay")


@dataclass(frozen=True, slots=True)
class CurrentArtifactReceipt:
    circuit_id: str
    flow_spec_key: str
    repository_root: str
    pdk_id: str
    source_sha256: str
    materialized_source_path: str
    materialized_source_sha256: str
    implementation_sha256: tuple[tuple[str, str], ...]
    pdk_snapshot_sha256: str
    pdk_installed_authority_sha256: str
    exact_receipt_path: str
    exact_receipt_sha256: str
    run_summary_path: str
    run_summary_sha256: str
    gds_path: str
    gds_sha256: str
    exact_admission: ExactAdmissionLike

    def __post_init__(self) -> None:
        if type(self.circuit_id) is not str or _CIRCUIT_RE.fullmatch(self.circuit_id) is None:
            raise ValueError("current artifact circuit identity is not canonical")
        if (
            type(self.flow_spec_key) is not str
            or _CIRCUIT_RE.fullmatch(self.flow_spec_key) is None
        ):
            raise ValueError("current artifact flow spec key is not canonical")
        if type(self.pdk_id) is not str or _CIRCUIT_RE.fullmatch(self.pdk_id) is None:
            raise ValueError("current artifact PDK identity is not canonical")
        root = _require_absolute_directory(
            self.repository_root, field="repository_root"
        )
        for field in (
            "source_sha256",
            "materialized_source_sha256",
            "pdk_snapshot_sha256",
            "pdk_installed_authority_sha256",
            "exact_receipt_sha256",
            "run_summary_sha256",
            "gds_sha256",
        ):
            _require_sha256(getattr(self, field), field=field)
        _implementation_rows(self.implementation_sha256)
        materialized = _require_absolute_path(
            self.materialized_source_path,
            field="materialized_source_path",
            suffix=".clay",
        )
        _require_absolute_path(
            self.exact_receipt_path, field="exact_receipt_path", suffix=".json"
        )
        _require_absolute_path(
            self.run_summary_path, field="run_summary_path", suffix=".json"
        )
        _require_absolute_path(self.gds_path, field="gds_path", suffix=".gds")
        if len({self.exact_receipt_path, self.run_summary_path, self.gds_path}) != 3:
            raise ValueError("current artifact paths must be distinct")
        admission = _admission_from_data(_admission_data(self.exact_admission))
        if type(admission) is ExactAdmission:
            circuit_id = admission.spec.circuit_id
            source_sha256 = admission.spec.source_sha256
            selected_source = admission.selected_source_path
            selected_source_sha256 = admission.selected_source_sha256
            pdk_snapshot_sha256 = admission.spec.pdk_snapshot_sha256
            installed_authority_sha256 = (
                admission.spec.pdk_installed_authority_sha256
            )
            output_path = admission.spec.output_path
            implementation = admission.spec.implementation_sha256
        else:
            circuit_id = admission.circuit_id
            source_sha256 = admission.source_sha256
            selected_source = admission.source_path
            selected_source_sha256 = admission.source_sha256
            pdk_snapshot_sha256 = admission.pdk_snapshot_sha256
            installed_authority_sha256 = admission.pdk_installed_authority_sha256
            output_path = admission.output_path
            implementation = admission.implementation_sha256
        expected_implementation = []
        for relative, digest in implementation:
            pure = PurePosixPath(relative)
            if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != relative:
                raise ValueError("implementation closure path is not repository relative")
            expected_implementation.append(
                (str((root / Path(*pure.parts)).resolve(strict=False)), digest)
            )
        if tuple(sorted(expected_implementation)) != self.implementation_sha256:
            raise ValueError("implementation closure path binding differs from exact admission")
        if (
            circuit_id != self.circuit_id
            or source_sha256 != self.source_sha256
            or str(selected_source) != str(materialized)
            or selected_source_sha256 != self.materialized_source_sha256
            or pdk_snapshot_sha256 != self.pdk_snapshot_sha256
            or installed_authority_sha256 != self.pdk_installed_authority_sha256
            or output_path.resolve(strict=False) != self.output_root
        ):
            raise ValueError("current artifact fields differ from exact admission")

    @property
    def output_root(self) -> Path:
        return Path(self.exact_receipt_path).parent

    def canonical_data(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema": _SCHEMA,
            "circuit_id": self.circuit_id,
            "flow_spec_key": self.flow_spec_key,
            "repository_root": self.repository_root,
            "pdk_id": self.pdk_id,
            "source_sha256": self.source_sha256,
            "materialized_source_path": self.materialized_source_path,
            "materialized_source_sha256": self.materialized_source_sha256,
            "implementation_sha256": [list(row) for row in self.implementation_sha256],
            "pdk_snapshot_sha256": self.pdk_snapshot_sha256,
            "pdk_installed_authority_sha256": self.pdk_installed_authority_sha256,
            "exact_receipt_path": self.exact_receipt_path,
            "exact_receipt_sha256": self.exact_receipt_sha256,
            "run_summary_path": self.run_summary_path,
            "run_summary_sha256": self.run_summary_sha256,
            "gds_path": self.gds_path,
            "gds_sha256": self.gds_sha256,
            "exact_admission": _admission_data(self.exact_admission),
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.canonical_data()).encode("ascii")).hexdigest()

    @classmethod
    def from_data(cls, data: object) -> "CurrentArtifactReceipt":
        if type(data) is not dict:
            raise TypeError("current artifact replay data must be one exact dict")
        expected = {
            "schema",
            "circuit_id",
            "flow_spec_key",
            "repository_root",
            "pdk_id",
            "source_sha256",
            "materialized_source_path",
            "materialized_source_sha256",
            "implementation_sha256",
            "pdk_snapshot_sha256",
            "pdk_installed_authority_sha256",
            "exact_receipt_path",
            "exact_receipt_sha256",
            "run_summary_path",
            "run_summary_sha256",
            "gds_path",
            "gds_sha256",
            "exact_admission",
        }
        keys = set(data)
        if keys != expected:
            raise ValueError(
                "current artifact canonical fields differ: "
                f"missing={sorted(expected - keys)}, unknown={sorted(keys - expected)}"
            )
        if data["schema"] != _SCHEMA:
            raise ValueError(f"current artifact schema must be {_SCHEMA!r}")
        result = cls(
            circuit_id=data["circuit_id"],  # type: ignore[arg-type]
            flow_spec_key=data["flow_spec_key"],  # type: ignore[arg-type]
            repository_root=data["repository_root"],  # type: ignore[arg-type]
            pdk_id=data["pdk_id"],  # type: ignore[arg-type]
            source_sha256=data["source_sha256"],  # type: ignore[arg-type]
            materialized_source_path=data["materialized_source_path"],  # type: ignore[arg-type]
            materialized_source_sha256=data["materialized_source_sha256"],  # type: ignore[arg-type]
            implementation_sha256=_implementation_rows_from_data(
                data["implementation_sha256"]
            ),
            pdk_snapshot_sha256=data["pdk_snapshot_sha256"],  # type: ignore[arg-type]
            pdk_installed_authority_sha256=data[
                "pdk_installed_authority_sha256"
            ],  # type: ignore[arg-type]
            exact_receipt_path=data["exact_receipt_path"],  # type: ignore[arg-type]
            exact_receipt_sha256=data["exact_receipt_sha256"],  # type: ignore[arg-type]
            run_summary_path=data["run_summary_path"],  # type: ignore[arg-type]
            run_summary_sha256=data["run_summary_sha256"],  # type: ignore[arg-type]
            gds_path=data["gds_path"],  # type: ignore[arg-type]
            gds_sha256=data["gds_sha256"],  # type: ignore[arg-type]
            exact_admission=_admission_from_data(data["exact_admission"]),
        )
        if result.canonical_data() != data:
            raise ValueError("current artifact replay data is not canonical")
        return result

    @classmethod
    def from_exact_receipt(
        cls,
        exact_receipt_path: str | Path,
        *,
        admission: ExactAdmissionLike,
        run_summary_path: str | Path,
        repository_root: str | Path,
        pdk_snapshot: PdkSourceSnapshot,
        flow_spec_key: str | None = None,
    ) -> "CurrentArtifactReceipt":
        if type(admission) not in {ExactAdmission, DirectExactAdmission}:
            raise TypeError("current artifact factory requires one typed exact admission")
        admission.__post_init__()
        if type(admission) is ExactAdmission:
            spec = admission.spec
            circuit_id = spec.circuit_id
            source_path = Path(spec.source_path)
            source_sha256 = spec.source_sha256
            selected_source_path = admission.selected_source_path
            selected_source_sha256 = admission.selected_source_sha256
            implementation = spec.implementation_sha256
            snapshot_sha256 = spec.pdk_snapshot_sha256
            installed_authority_sha256 = spec.pdk_installed_authority_sha256
            output_root = spec.output_path.resolve(strict=True)
            admission_file = output_root / "experiment_spec.json"
        else:
            spec = None
            circuit_id = admission.circuit_id
            source_path = admission.source_path
            source_sha256 = admission.source_sha256
            selected_source_path = admission.source_path
            selected_source_sha256 = admission.source_sha256
            implementation = admission.implementation_sha256
            snapshot_sha256 = admission.pdk_snapshot_sha256
            installed_authority_sha256 = admission.pdk_installed_authority_sha256
            output_root = admission.output_path.resolve(strict=True)
            admission_file = output_root / "direct_exact_admission.json"
        resolved_flow_spec_key = (
            circuit_id if flow_spec_key is None else flow_spec_key
        )
        if (
            type(resolved_flow_spec_key) is not str
            or _CIRCUIT_RE.fullmatch(resolved_flow_spec_key) is None
        ):
            raise ValueError("current artifact flow spec key is not canonical")
        if type(admission) is ExactAdmission and (
            type(spec) is not IntentExperimentSpec
            or spec.stage is not ExperimentStage.EXACT
        ):
            raise TypeError("current artifact factory requires one exact experiment spec")
        if type(pdk_snapshot) is not PdkSourceSnapshot:
            raise TypeError("current artifact factory requires one captured PDK snapshot")
        live_snapshot = _live_pdk_snapshot(pdk_snapshot.pdk_id)
        if live_snapshot != pdk_snapshot:
            raise ValueError("caller PDK snapshot differs from live PDK capture")
        if (
            pdk_snapshot.snapshot_sha256 != snapshot_sha256
            or pdk_snapshot.installed_authority_sha256
            != installed_authority_sha256
        ):
            raise ValueError("PDK snapshot or installed authority differs from admission")

        root = Path(repository_root).resolve(strict=True)
        if not root.is_dir():
            raise ValueError("repository root is not one directory")
        source = _confined(source_path, root, field="canonical source")
        if _sha256_file(source) != source_sha256:
            raise ValueError("canonical source differs from exact admission")
        if type(admission) is DirectExactAdmission:
            source_input = _confined(
                admission.source_input_path, root, field="direct source input"
            )
            if _sha256_file(source_input) != admission.source_input_sha256:
                raise ValueError("direct source input differs from exact admission")
        selected_source = _confined(
            selected_source_path, root, field="materialized source"
        )
        if _sha256_file(selected_source) != selected_source_sha256:
            raise ValueError("materialized source differs from exact admission")

        absolute_implementation: list[tuple[str, str]] = []
        for relative, digest in implementation:
            pure = PurePosixPath(relative)
            if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != relative:
                raise ValueError("implementation closure path is not repository relative")
            implementation_path = _confined(
                root / Path(*pure.parts), root, field="implementation closure"
            )
            if _sha256_file(implementation_path) != digest:
                raise ValueError("implementation closure differs from exact admission")
            absolute_implementation.append((str(implementation_path), digest))
        implementation_rows = tuple(sorted(absolute_implementation))

        if not output_root.is_dir():
            raise ValueError("exact output root is not one directory")
        exact_path = _confined(
            Path(exact_receipt_path), output_root, field="focused exact receipt"
        )
        summary_path = _confined(
            Path(run_summary_path), output_root, field="run summary"
        )
        admission_data = _read_json(admission_file, field="exact admission")
        if type(admission) is ExactAdmission:
            if IntentExperimentSpec.from_data(admission_data) != spec:
                raise ValueError("experiment spec differs from exact admission")
        else:
            replayed_admission = DirectExactAdmission.from_data(admission_data)
            if replayed_admission != admission:
                raise ValueError("direct exact admission differs from persisted identity")

        exact_result = _read_json(exact_path, field="focused exact receipt")
        _verify_exact_result_core(exact_result, admission=admission)

        summary = _read_json(summary_path, field="run summary")
        gds_path, exact_artifact_path, exact_artifact = _summary_artifacts(
            summary,
            flow_spec_key=resolved_flow_spec_key,
            output_root=output_root,
            run_summary_path=summary_path,
        )
        gds_sha256 = _sha256_file(gds_path)
        layout = _layout_step(summary, flow_spec_key=resolved_flow_spec_key)
        data = layout["data"]
        if type(data) is not dict or data.get("gds_sha256") != gds_sha256:
            raise ValueError("run summary GDS SHA differs from current artifact")
        _verify_exact_artifact(
            exact_artifact,
            path=exact_artifact_path,
            gds_path=gds_path,
            gds_sha256=gds_sha256,
            candidate_sha256=admission.selected_candidate_sha256,
        )

        result = cls(
            circuit_id=circuit_id,
            flow_spec_key=resolved_flow_spec_key,
            repository_root=str(root),
            pdk_id=pdk_snapshot.pdk_id,
            source_sha256=source_sha256,
            materialized_source_path=str(selected_source),
            materialized_source_sha256=selected_source_sha256,
            implementation_sha256=implementation_rows,
            pdk_snapshot_sha256=snapshot_sha256,
            pdk_installed_authority_sha256=installed_authority_sha256,
            exact_receipt_path=str(exact_path),
            exact_receipt_sha256=_sha256_file(exact_path),
            run_summary_path=str(summary_path),
            run_summary_sha256=_sha256_file(summary_path),
            gds_path=str(gds_path),
            gds_sha256=gds_sha256,
            exact_admission=admission,
        )
        result.verify()
        return result

    def verify(self) -> "CurrentArtifactReceipt":
        self.__post_init__()
        admission = _admission_from_data(_admission_data(self.exact_admission))
        if type(admission) is ExactAdmission:
            implementation = admission.spec.implementation_sha256
            source_path = Path(admission.spec.source_path)
            selected_source_path = admission.selected_source_path
            identity_file = "experiment_spec.json"
        else:
            implementation = admission.implementation_sha256
            source_path = admission.source_path
            selected_source_path = admission.source_path
            identity_file = "direct_exact_admission.json"
        root = Path(self.repository_root).resolve(strict=True)
        if not root.is_dir():
            raise ValueError("repository root is not one current directory")
        exact_path = Path(self.exact_receipt_path)
        summary_path = Path(self.run_summary_path)
        gds_path = Path(self.gds_path)
        _require_current_file(
            exact_path, self.exact_receipt_sha256, field="focused exact receipt"
        )
        _require_current_file(
            summary_path, self.run_summary_sha256, field="run summary"
        )
        _require_current_file(gds_path, self.gds_sha256, field="GDS artifact")
        expected_implementation = []
        for relative, digest in implementation:
            pure = PurePosixPath(relative)
            implementation = _confined(
                root / Path(*pure.parts), root, field="implementation closure"
            )
            expected_implementation.append((str(implementation), digest))
            _require_current_file(
                implementation, digest, field="implementation closure"
            )
        if tuple(sorted(expected_implementation)) != self.implementation_sha256:
            raise ValueError("implementation closure path binding differs")

        output_root = self.output_root.resolve(strict=True)
        _confined(summary_path, output_root, field="run summary")
        _confined(gds_path, output_root, field="GDS artifact")
        identity_data = _read_json(
            output_root / identity_file, field="exact admission identity"
        )
        if type(admission) is ExactAdmission:
            spec = IntentExperimentSpec.from_data(identity_data)
            identity_ok = (
                spec == admission.spec
                and spec.stage is ExperimentStage.EXACT
                and spec.circuit_id == self.circuit_id
                and spec.output_path.resolve(strict=True) == output_root
                and spec.source_sha256 == self.source_sha256
                and spec.pdk_snapshot_sha256 == self.pdk_snapshot_sha256
                and spec.pdk_installed_authority_sha256
                == self.pdk_installed_authority_sha256
            )
        else:
            direct = DirectExactAdmission.from_data(identity_data)
            identity_ok = (
                direct == admission
                and direct.circuit_id == self.circuit_id
                and direct.output_path.resolve(strict=True) == output_root
                and direct.source_sha256 == self.source_sha256
                and direct.pdk_snapshot_sha256 == self.pdk_snapshot_sha256
                and direct.pdk_installed_authority_sha256
                == self.pdk_installed_authority_sha256
            )
        if not identity_ok:
            raise ValueError("exact admission identity differs from current artifact")
        source = _confined(source_path, root, field="canonical source")
        _require_current_file(source, self.source_sha256, field="canonical source")
        if type(admission) is DirectExactAdmission:
            source_input = _confined(
                admission.source_input_path, root, field="direct source input"
            )
            _require_current_file(
                source_input,
                admission.source_input_sha256,
                field="direct source input",
            )
        materialized = _confined(
            Path(self.materialized_source_path), root, field="materialized source"
        )
        if materialized != selected_source_path.resolve(strict=True):
            raise ValueError("materialized source path differs from exact admission")
        _require_current_file(
            materialized,
            self.materialized_source_sha256,
            field="materialized source",
        )
        live_snapshot = _live_pdk_snapshot(self.pdk_id)
        if (
            live_snapshot.snapshot_sha256 != self.pdk_snapshot_sha256
            or live_snapshot.installed_authority_sha256
            != self.pdk_installed_authority_sha256
        ):
            raise ValueError("live PDK snapshot differs from current artifact")
        summary = _read_json(summary_path, field="run summary")
        summary_gds, exact_artifact_path, exact_artifact = _summary_artifacts(
            summary,
            flow_spec_key=self.flow_spec_key,
            output_root=output_root,
            run_summary_path=summary_path,
        )
        if summary_gds != gds_path:
            raise ValueError("run summary GDS path differs from current artifact")
        exact_result = _read_json(exact_path, field="focused exact receipt")
        candidate = exact_result.get("candidate_sha256")
        if type(candidate) is not str:
            raise TypeError("focused exact receipt lacks candidate identity")
        _verify_exact_artifact(
            exact_artifact,
            path=exact_artifact_path,
            gds_path=gds_path,
            gds_sha256=self.gds_sha256,
            candidate_sha256=candidate,
        )
        layout = _layout_step(summary, flow_spec_key=self.flow_spec_key)
        data = layout["data"]
        if type(data) is not dict or data.get("gds_sha256") != self.gds_sha256:
            raise ValueError("run summary GDS SHA differs from current artifact")
        _verify_exact_result(
            exact_result,
            receipt=self,
            admission=admission,
            summary_artifact=exact_artifact,
        )
        return self


def load_current_artifact_receipt(path: str | Path) -> CurrentArtifactReceipt:
    """Load and verify one persisted current-artifact receipt."""

    receipt = CurrentArtifactReceipt.from_data(
        _read_json(Path(path), field="current artifact receipt")
    )
    return receipt.verify()


def persist_current_artifact_receipt(
    receipt: CurrentArtifactReceipt, path: str | Path
) -> Path:
    """Verify and atomically persist one current-artifact receipt once."""

    if type(receipt) is not CurrentArtifactReceipt:
        raise TypeError("current artifact persistence requires one typed receipt")
    receipt.verify()
    destination = Path(path).resolve(strict=False)
    if destination.exists():
        raise FileExistsError(
            f"refusing to overwrite current artifact receipt: {destination}"
        )
    if destination.parent.resolve(strict=True) != receipt.output_root.resolve(strict=True):
        raise ValueError("current artifact receipt must stay in its fresh output root")
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        _canonical_json(receipt.canonical_data()) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)
    return destination


__all__ = [
    "CurrentArtifactReceipt",
    "load_current_artifact_receipt",
    "persist_current_artifact_receipt",
]
