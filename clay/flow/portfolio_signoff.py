"""Portable physical-artifact signoff rows for the three-circuit portfolio.

The v2 row deliberately certifies physical legality and replayable artifacts only.
Reference fidelity and electrical characterization belong to separate ledgers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile
from typing import Callable, Iterable, Literal, Mapping, Sequence, TypeAlias

from clay.core_contract import VerificationStatus
from clay.flow.pdk_matrix import FOUR_PDK_IDS


PORTFOLIO_CIRCUIT_IDS = ("two_stage_ota", "reference_ldo", "bandgap")
PHYSICAL_CLAIM_SCOPE = "physical_artifact_signoff"
PORTFOLIO_ROW_V1_SCHEMA = "clay.portfolio-signoff-row/v1"
PORTFOLIO_ROW_V2_SCHEMA = "clay.portfolio-signoff-row/v2"
PORTFOLIO_LEDGER_V1_SCHEMA = "clay.three-circuit-four-pdk-signoff/v1"
PORTFOLIO_LEDGER_V2_SCHEMA = "clay.three-circuit-four-pdk-signoff/v2"
_COMMIT = re.compile(r"[0-9a-f]{40,64}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
BASELINE_GDS_AREA_ABS_EPSILON_UM2 = 1e-9
_BASELINE_LEDGER_PATHS = {
    "bandgap": "bandgap_dev/signoff/four_pdk_ledger.json",
}
_BASELINE_GDS_TOP_CELLS = {"bandgap": "Bandgap"}
_BASELINE_RECEIPT_POLICIES = {
    "bandgap": {
        "comparator": "lt",
        "threshold": 1.0,
        "algorithm": "bbox-area-ratio",
        "algorithm_version": "1",
        "epsilon": 0.0,
    }
}
_SEARCH_STATUSES = frozenset(
    {
        "not_run",
        "baseline_only",
        "best_observed",
        "domain_exhausted",
        "optimal",
        "no_convergence",
    }
)
_CANDIDATE_ORIGINS = frozenset({"authoring", "derived"})
_REQUIRED_ARTIFACT_IDS = frozenset(
    {
        "source",
        "candidate",
        "physical_macro_graph",
        "gds",
        "drc_deck",
        "lvs_deck",
        "drc_report",
        "lvs_report",
        "pex",
        "exact_artifact",
        "containment",
    }
)


class PortfolioSignoffError(ValueError):
    """The portfolio cannot be published from the supplied evidence."""


@dataclass(frozen=True)
class ArtifactLocator:
    """Repository-relative locator for one digest-bound evidence object."""

    logical_id: str
    path: str
    sha256: str
    artifact_type: Literal["file", "manifest"] = "file"
    required: bool = True

    def canonical_data(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DeckScopeReceipt:
    """Logical and rule-coverage identity of a DRC or LVS deck."""

    logical_origin: str
    version: str
    sha256: str
    coverage: Literal["partial", "foundry"]
    foundry_signoff: bool

    def canonical_data(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SearchReceipt:
    """Bounded-search accounting attached to a physical candidate."""

    evaluated_count: int
    domain_count: int
    full_exact_count: int
    budget: int
    stop_reason: str
    lower_bound: float | None
    optimality_gap: float | None

    def canonical_data(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PortfolioSignoffRow:
    """Current physical-legality row; it owns no reference/electrical verdict."""

    circuit_id: str
    pdk_id: str
    commit: str
    claim_scope: Literal["physical_artifact_signoff"]
    physical_status: VerificationStatus
    physical_signed_off: bool
    topology_id: str
    normalized_connectivity_sha256: str
    candidate_origin: Literal["authoring", "derived"]
    search_status: Literal[
        "not_run",
        "baseline_only",
        "best_observed",
        "domain_exhausted",
        "optimal",
        "no_convergence",
    ]
    search_receipt: SearchReceipt
    drc_deck: DeckScopeReceipt
    lvs_deck: DeckScopeReceipt
    artifacts: tuple[ArtifactLocator, ...]
    drc_violations: int
    lvs_matched: bool
    lvs_full_extraction: bool
    body_measured: bool
    pex_status: str
    exact_artifact_status: str
    containment_status: str
    promoted: bool
    baseline_comparison_receipt: BaselineComparisonReceipt | None = None

    def canonical_data(self) -> dict[str, object]:
        payload = {
            "schema": PORTFOLIO_ROW_V2_SCHEMA,
            "circuit_id": self.circuit_id,
            "pdk_id": self.pdk_id,
            "commit": self.commit,
            "claim_scope": self.claim_scope,
            "physical_status": self.physical_status.value,
            "physical_signed_off": self.physical_signed_off,
            "topology_id": self.topology_id,
            "normalized_connectivity_sha256": self.normalized_connectivity_sha256,
            "candidate_origin": self.candidate_origin,
            "search_status": self.search_status,
            "search_receipt": self.search_receipt.canonical_data(),
            "drc_deck": self.drc_deck.canonical_data(),
            "lvs_deck": self.lvs_deck.canonical_data(),
            "artifacts": [item.canonical_data() for item in self.artifacts],
            "drc_violations": self.drc_violations,
            "lvs_matched": self.lvs_matched,
            "lvs_full_extraction": self.lvs_full_extraction,
            "body_measured": self.body_measured,
            "pex_status": self.pex_status,
            "exact_artifact_status": self.exact_artifact_status,
            "containment_status": self.containment_status,
            "promoted": self.promoted,
        }
        if self.baseline_comparison_receipt is not None:
            payload["baseline_comparison_receipt"] = (
                self.baseline_comparison_receipt.canonical_data()
            )
        return payload


@dataclass(frozen=True)
class LegacyPortfolioSignoffRow:
    """Read-only v1 representation normalized to the physical claim scope."""

    circuit_id: str
    pdk_id: str
    commit: str
    source_path: str
    source_sha256: str
    candidate_path: str
    candidate_sha256: str
    physical_macro_graph_path: str
    physical_macro_graph_sha256: str
    gds_path: str
    gds_sha256: str
    drc_deck_path: str
    drc_deck_sha256: str
    lvs_deck_path: str
    lvs_deck_sha256: str
    drc_report_path: str
    drc_report_sha256: str
    drc_status: str
    drc_violations: int
    lvs_report_path: str
    lvs_report_sha256: str
    lvs_status: str
    lvs_matched: bool
    lvs_full_extraction: bool
    body_status: str
    body_measured: bool
    pex_path: str
    pex_sha256: str
    pex_status: str
    exact_artifact_path: str
    exact_artifact_sha256: str
    exact_artifact_status: str
    containment_path: str
    containment_sha256: str
    containment_status: str
    promoted: bool

    @property
    def claim_scope(self) -> str:
        return PHYSICAL_CLAIM_SCOPE

    @property
    def physical_signed_off(self) -> bool:
        return (
            self.drc_status == "pass"
            and self.drc_violations == 0
            and self.lvs_status == "pass"
            and self.lvs_matched is True
            and self.lvs_full_extraction is True
            and self.body_status == "pass"
            and self.body_measured is True
            and self.pex_status == "pass"
            and self.exact_artifact_status == "pass"
            and self.containment_status == "pass"
            and self.promoted is True
        )

    @property
    def physical_status(self) -> VerificationStatus:
        return (
            VerificationStatus.PASS
            if self.physical_signed_off
            else VerificationStatus.FAIL
        )

    def canonical_data(self) -> dict[str, object]:
        return {"schema": PORTFOLIO_ROW_V1_SCHEMA, **asdict(self)}


PortfolioReadableRow: TypeAlias = PortfolioSignoffRow | LegacyPortfolioSignoffRow


@dataclass(frozen=True)
class BaselineComparisonReceipt:
    """Immutable binding for a same-topology, same-PDK relative comparison."""

    circuit_id: str
    topology_id: str
    pdk_id: str
    evidence_commit: str
    physical_row_sha256: str
    baseline_gds_sha256: str
    baseline_candidate_sha256: str
    baseline_area_um2: float
    new_gds_sha256: str
    new_candidate_sha256: str
    new_area_um2: float
    comparator: Literal["lt", "le", "gt", "ge"]
    threshold: float
    algorithm: str
    algorithm_version: str
    epsilon: float

    def canonical_data(self) -> dict[str, object]:
        return {
            "schema": "clay.baseline-comparison-receipt/v1",
            **asdict(self),
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.canonical_data())).hexdigest()


_LEGACY_ARTIFACT_PAIRS = (
    ("source_path", "source_sha256"),
    ("candidate_path", "candidate_sha256"),
    ("physical_macro_graph_path", "physical_macro_graph_sha256"),
    ("gds_path", "gds_sha256"),
    ("drc_deck_path", "drc_deck_sha256"),
    ("lvs_deck_path", "lvs_deck_sha256"),
    ("drc_report_path", "drc_report_sha256"),
    ("lvs_report_path", "lvs_report_sha256"),
    ("pex_path", "pex_sha256"),
    ("exact_artifact_path", "exact_artifact_sha256"),
    ("containment_path", "containment_sha256"),
)


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def normalized_connectivity_sha256(cell_def: object) -> str:
    """Hash only the declared topology graph, excluding physical sizing knobs.

    Device records are canonicalized by instance identity, class, and pin-to-net
    connectivity.  Composed cells bind their sub-cell identity and port map.  A
    stage-style cell with neither form binds its topology declaration because
    that declaration is the only connectivity authority present in the source.
    """

    devices = tuple(getattr(cell_def, "devices", ()) or ())
    instances = tuple(getattr(cell_def, "instances", ()) or ())
    projection: dict[str, object] = {
        "cell": str(getattr(cell_def, "local_name", "")),
        # Port position is part of the subcircuit interface contract.  Preserve
        # declaration order; sorting would silently accept a reordered pinout.
        "ports": [
            str(port) for port in (getattr(cell_def, "ports", ()) or ())
        ],
    }
    if devices:
        projection["devices"] = [
            {
                "instance": str(getattr(device, "instance_name", "")),
                "class": str(getattr(device, "device_class", "")),
                "pins": sorted(
                    (str(pin), str(net))
                    for pin, net in dict(getattr(device, "pins", {}) or {}).items()
                ),
            }
            for device in sorted(
                devices, key=lambda item: str(getattr(item, "instance_name", ""))
            )
        ]
    elif instances:
        projection["instances"] = [
            {
                "instance": str(getattr(instance, "name", "")),
                "cell_path": str(getattr(instance, "cell_path", "")),
                "port_map": sorted(
                    (str(port), str(net))
                    for port, net in dict(
                        getattr(instance, "port_map", {}) or {}
                    ).items()
                ),
            }
            for instance in sorted(
                instances, key=lambda item: str(getattr(item, "name", ""))
            )
        ]
    else:
        topology = dict(getattr(cell_def, "topology", {}) or {})
        projection["topology"] = {
            str(key): topology[key]
            for key in sorted(topology)
            if str(key) != "rationale"
        }
    return hashlib.sha256(_canonical_json(projection)).hexdigest()


def normalized_clay_file_connectivity_sha256(
    source: str | Path, *, top_cell: str
) -> str:
    """Parse one Clay authority and hash the uniquely named top-cell graph."""

    from clay.parser import parse_clay_file

    parsed = parse_clay_file(Path(source))
    matches = [
        cell for cell in parsed.cell_defs if str(cell.local_name) == str(top_cell)
    ]
    if len(matches) != 1:
        raise PortfolioSignoffError(
            f"connectivity authority requires one {top_cell!r} cell; "
            f"found {len(matches)}"
        )
    return normalized_connectivity_sha256(matches[0])


def canonical_portfolio_row_sha256(row: PortfolioReadableRow) -> str:
    return hashlib.sha256(_canonical_json(row.canonical_data())).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha(value: object, *, label: str) -> str:
    normalized = str(value).casefold()
    if _SHA256.fullmatch(normalized) is None:
        raise PortfolioSignoffError(f"{label} is not a SHA-256")
    return normalized


def _validate_search_receipt(receipt: SearchReceipt, *, key: str) -> None:
    counts = (
        receipt.evaluated_count,
        receipt.domain_count,
        receipt.full_exact_count,
        receipt.budget,
    )
    if any(type(value) is not int or value < 0 for value in counts):
        raise PortfolioSignoffError(f"{key} search counts/budget are invalid")
    if receipt.evaluated_count > receipt.domain_count:
        raise PortfolioSignoffError(f"{key} evaluated count exceeds domain count")
    if receipt.full_exact_count > receipt.evaluated_count:
        raise PortfolioSignoffError(f"{key} full-exact count exceeds evaluated count")
    if receipt.evaluated_count > receipt.budget:
        raise PortfolioSignoffError(f"{key} evaluated count exceeds search budget")
    if not receipt.stop_reason.strip():
        raise PortfolioSignoffError(f"{key} search stop reason is empty")
    for label, value in (
        ("lower bound", receipt.lower_bound),
        ("optimality gap", receipt.optimality_gap),
    ):
        if value is not None and (
            type(value) not in {int, float}
            or not math.isfinite(value)
            or value < 0.0
        ):
            raise PortfolioSignoffError(f"{key} search {label} is invalid")


def _validate_legacy_row(
    row: LegacyPortfolioSignoffRow, *, current_commit: str
) -> None:
    key = f"{row.circuit_id}/{row.pdk_id}"
    if row.commit != current_commit:
        raise PortfolioSignoffError(f"{key} commit differs from current HEAD")
    for path_field, digest_field in _LEGACY_ARTIFACT_PAIRS:
        raw_digest = _require_sha(getattr(row, digest_field), label=f"{key} {digest_field}")
        path = Path(str(getattr(row, path_field))).resolve()
        if not path.is_file() or path.stat().st_size <= 0:
            raise PortfolioSignoffError(f"{key} artifact is missing: {path_field}")
        if _sha256(path) != raw_digest:
            raise PortfolioSignoffError(f"{key} {path_field} digest mismatch")
    if not row.physical_signed_off:
        if row.drc_status != "pass" or row.drc_violations != 0:
            gate = "DRC"
        elif (
            row.lvs_status != "pass"
            or row.lvs_matched is not True
            or row.lvs_full_extraction is not True
        ):
            gate = "LVS"
        elif row.body_status != "pass" or row.body_measured is not True:
            gate = "body"
        elif row.pex_status != "pass":
            gate = "PEX"
        elif row.exact_artifact_status != "pass":
            gate = "exact-artifact"
        elif row.containment_status != "pass":
            gate = "containment"
        else:
            gate = "promotion"
        raise PortfolioSignoffError(f"{key} {gate} gate is not pass")


def _validate_v2_row(
    row: PortfolioSignoffRow, *, current_commit: str, repo_root: str | Path
) -> None:
    key = f"{row.circuit_id}/{row.pdk_id}"
    if row.commit != current_commit:
        raise PortfolioSignoffError(f"{key} commit differs from current HEAD")
    if row.claim_scope != PHYSICAL_CLAIM_SCOPE:
        raise PortfolioSignoffError(f"{key} claim scope is not physical artifact signoff")
    if row.physical_status is not VerificationStatus.PASS:
        raise PortfolioSignoffError(f"{key} physical status is not pass")
    if row.physical_signed_off is not True:
        raise PortfolioSignoffError(f"{key} physical signoff is withheld")
    if not row.topology_id.strip():
        raise PortfolioSignoffError(f"{key} topology id is empty")
    _require_sha(
        row.normalized_connectivity_sha256,
        label=f"{key} normalized connectivity digest",
    )
    if row.candidate_origin not in _CANDIDATE_ORIGINS:
        raise PortfolioSignoffError(f"{key} candidate origin is unknown")
    if row.search_status not in _SEARCH_STATUSES:
        raise PortfolioSignoffError(f"{key} search status is unknown")
    _validate_search_receipt(row.search_receipt, key=key)
    if row.drc_deck.coverage not in {"partial", "foundry"}:
        raise PortfolioSignoffError(f"{key} DRC deck coverage is unknown")
    if row.lvs_deck.coverage not in {"partial", "foundry"}:
        raise PortfolioSignoffError(f"{key} LVS deck coverage is unknown")
    for label, receipt in (("DRC", row.drc_deck), ("LVS", row.lvs_deck)):
        if not receipt.logical_origin.strip() or not receipt.version.strip():
            raise PortfolioSignoffError(f"{key} {label} deck identity is incomplete")
        _require_sha(receipt.sha256, label=f"{key} {label} deck digest")
        if type(receipt.foundry_signoff) is not bool:
            raise PortfolioSignoffError(
                f"{key} {label} deck foundry signoff flag is not boolean"
            )
        if receipt.foundry_signoff and receipt.coverage != "foundry":
            raise PortfolioSignoffError(
                f"{key} {label} partial deck cannot claim foundry signoff"
            )
    by_id: dict[str, ArtifactLocator] = {}
    for locator in row.artifacts:
        if locator.logical_id in by_id:
            raise PortfolioSignoffError(
                f"{key} duplicate artifact logical id: {locator.logical_id}"
            )
        by_id[locator.logical_id] = locator
    missing = _REQUIRED_ARTIFACT_IDS - set(by_id)
    if missing or any(not by_id[item].required for item in _REQUIRED_ARTIFACT_IDS - missing):
        raise PortfolioSignoffError(
            f"{key} required artifact locators are incomplete: {sorted(missing)}"
        )
    if row.drc_deck.sha256 != by_id["drc_deck"].sha256:
        raise PortfolioSignoffError(f"{key} DRC deck receipt digest mismatch")
    if row.lvs_deck.sha256 != by_id["lvs_deck"].sha256:
        raise PortfolioSignoffError(f"{key} LVS deck receipt digest mismatch")
    if row.baseline_comparison_receipt is not None:
        repository = Path(repo_root).resolve()
        try:
            ledger_path = _BASELINE_LEDGER_PATHS[row.circuit_id]
            gds_top_cell = _BASELINE_GDS_TOP_CELLS[row.circuit_id]
        except KeyError as exc:
            raise PortfolioSignoffError(
                f"{key} has no strict baseline evidence contract"
            ) from exc
        validate_baseline_comparison_evidence(
            row.baseline_comparison_receipt,
            current_commit=current_commit,
            repo_root=repository,
            current_row=row,
            evidence_ledger_path=ledger_path,
            gds_top_cell=gds_top_cell,
            expected_bindings=_BASELINE_RECEIPT_POLICIES[row.circuit_id],
        )
    try:
        from clay.flow.signoff_artifacts import verify_artifact_graph

        verify_artifact_graph(repo_root, row.artifacts)
    except (OSError, ValueError) as exc:
        raise PortfolioSignoffError(f"{key} artifact verification failed: {exc}") from exc
    if type(row.drc_violations) is not int:
        raise PortfolioSignoffError(f"{key} DRC violations is not an integer")
    gates = (
        (row.drc_violations == 0, "DRC"),
        (row.lvs_matched is True and row.lvs_full_extraction is True, "LVS"),
        (row.body_measured is True, "body"),
        (row.pex_status == "pass", "PEX"),
        (row.exact_artifact_status == "pass", "exact-artifact"),
        (row.containment_status == "pass", "containment"),
        (row.promoted is True, "promotion"),
    )
    failed = next((label for passed, label in gates if not passed), None)
    if failed is not None:
        raise PortfolioSignoffError(f"{key} {failed} gate is not pass")


def _strict_dataclass_payload(
    data: Mapping[str, object], cls: type, *, ignored: Sequence[str] = ("schema",)
) -> dict[str, object]:
    expected = set(cls.__dataclass_fields__)
    optional = (
        {"baseline_comparison_receipt"}
        if cls is PortfolioSignoffRow
        else set()
    )
    unknown = set(data) - expected - set(ignored)
    missing = expected - set(data) - optional
    if unknown or missing:
        raise PortfolioSignoffError(
            f"portfolio row fields differ: missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )
    return {
        key: data[key] if key in data else None
        for key in expected
    }


def _locator_from_mapping(data: object) -> ArtifactLocator:
    if not isinstance(data, Mapping):
        raise PortfolioSignoffError("artifact locator is not an object")
    return ArtifactLocator(**_strict_dataclass_payload(data, ArtifactLocator, ignored=()))  # type: ignore[arg-type]


def _deck_from_mapping(data: object) -> DeckScopeReceipt:
    if not isinstance(data, Mapping):
        raise PortfolioSignoffError("deck scope receipt is not an object")
    return DeckScopeReceipt(**_strict_dataclass_payload(data, DeckScopeReceipt, ignored=()))  # type: ignore[arg-type]


def _search_from_mapping(data: object) -> SearchReceipt:
    if not isinstance(data, Mapping):
        raise PortfolioSignoffError("search receipt is not an object")
    return SearchReceipt(**_strict_dataclass_payload(data, SearchReceipt, ignored=()))  # type: ignore[arg-type]


def portfolio_row_from_mapping(data: Mapping[str, object]) -> PortfolioReadableRow:
    """Strictly parse a v2 row or normalize a legacy v1 physical row."""

    if not isinstance(data, Mapping):
        raise PortfolioSignoffError("portfolio row is not an object")
    schema = data.get("schema", PORTFOLIO_ROW_V1_SCHEMA)
    if schema == PORTFOLIO_ROW_V1_SCHEMA:
        scope = data.get("claim_scope", PHYSICAL_CLAIM_SCOPE)
        if scope != PHYSICAL_CLAIM_SCOPE:
            raise PortfolioSignoffError(f"unknown portfolio claim scope: {scope!r}")
        normalized = dict(data)
        normalized.pop("claim_scope", None)
        fields = _strict_dataclass_payload(normalized, LegacyPortfolioSignoffRow)
        return LegacyPortfolioSignoffRow(**fields)  # type: ignore[arg-type]
    if schema != PORTFOLIO_ROW_V2_SCHEMA:
        raise PortfolioSignoffError(f"portfolio row schema is not supported: {schema!r}")
    if data.get("claim_scope") != PHYSICAL_CLAIM_SCOPE:
        raise PortfolioSignoffError(
            f"unknown portfolio claim scope: {data.get('claim_scope')!r}"
        )
    fields = _strict_dataclass_payload(data, PortfolioSignoffRow)
    try:
        fields["physical_status"] = VerificationStatus(fields["physical_status"])
        fields["search_receipt"] = _search_from_mapping(fields["search_receipt"])
        fields["drc_deck"] = _deck_from_mapping(fields["drc_deck"])
        fields["lvs_deck"] = _deck_from_mapping(fields["lvs_deck"])
        raw_baseline_receipt = fields["baseline_comparison_receipt"]
        if raw_baseline_receipt is not None:
            if not isinstance(raw_baseline_receipt, Mapping):
                raise PortfolioSignoffError(
                    "baseline comparison receipt is not an object"
                )
            fields["baseline_comparison_receipt"] = (
                baseline_comparison_receipt_from_mapping(raw_baseline_receipt)
            )
        raw_artifacts = fields["artifacts"]
        if not isinstance(raw_artifacts, Sequence) or isinstance(
            raw_artifacts, (str, bytes)
        ):
            raise PortfolioSignoffError("artifact locators are not a sequence")
        fields["artifacts"] = tuple(_locator_from_mapping(item) for item in raw_artifacts)
        return PortfolioSignoffRow(**fields)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        if isinstance(exc, PortfolioSignoffError):
            raise
        raise PortfolioSignoffError(f"portfolio v2 row is invalid: {exc}") from exc


def validate_portfolio_row_artifacts(
    row: PortfolioReadableRow,
    *,
    current_commit: str,
    repo_root: str | Path | None = None,
) -> PortfolioReadableRow:
    commit = str(current_commit).casefold()
    if _COMMIT.fullmatch(commit) is None:
        raise PortfolioSignoffError("current commit is not a full Git object id")
    if isinstance(row, PortfolioSignoffRow):
        if repo_root is None:
            raise PortfolioSignoffError("v2 artifact validation requires a repo root")
        _validate_v2_row(row, current_commit=commit, repo_root=repo_root)
    else:
        _validate_legacy_row(row, current_commit=commit)
    return row


def validate_portfolio_rows(
    rows: Iterable[PortfolioReadableRow],
    *,
    current_commit: str,
    repo_root: str | Path | None = None,
) -> tuple[PortfolioReadableRow, ...]:
    commit = str(current_commit).casefold()
    if _COMMIT.fullmatch(commit) is None:
        raise PortfolioSignoffError("current commit is not a full Git object id")
    normalized = tuple(rows)
    expected = tuple(
        (circuit, pdk) for circuit in PORTFOLIO_CIRCUIT_IDS for pdk in FOUR_PDK_IDS
    )
    keys = tuple((row.circuit_id, row.pdk_id) for row in normalized)
    if len(keys) != len(expected) or set(keys) != set(expected) or len(set(keys)) != len(keys):
        raise PortfolioSignoffError(
            "portfolio rows must equal the exact Cartesian product of three circuits and four PDKs"
        )
    by_key = {key: row for key, row in zip(keys, normalized)}
    ordered = tuple(by_key[key] for key in expected)
    if any(not isinstance(row, PortfolioSignoffRow) for row in ordered) and any(
        isinstance(row, PortfolioSignoffRow) for row in ordered
    ):
        raise PortfolioSignoffError("portfolio row schemas may not be mixed")
    for row in ordered:
        validate_portfolio_row_artifacts(
            row, current_commit=commit, repo_root=repo_root
        )
    return ordered


def publish_portfolio_signoff(
    destination: str | Path,
    rows: Iterable[PortfolioSignoffRow],
    *,
    current_commit: str,
    worktree_dirty: bool,
    repo_root: str | Path | None = None,
) -> dict[str, object]:
    """Validate v2 rows before atomic replacement; preserve old bytes on failure."""

    if worktree_dirty:
        raise PortfolioSignoffError("portfolio signoff refuses a dirty worktree")
    if repo_root is None:
        raise PortfolioSignoffError("v2 publication requires an explicit repo root")
    ordered = validate_portfolio_rows(
        rows, current_commit=current_commit, repo_root=repo_root
    )
    if any(not isinstance(row, PortfolioSignoffRow) for row in ordered):
        raise PortfolioSignoffError("v1 rows are read-only and cannot be republished")
    payload: dict[str, object] = {
        "schema": PORTFOLIO_LEDGER_V2_SCHEMA,
        "claim_scope": PHYSICAL_CLAIM_SCOPE,
        "physical_status": VerificationStatus.PASS.value,
        "physical_signed_off": True,
        "commit": str(current_commit).casefold(),
        "circuit_order": list(PORTFOLIO_CIRCUIT_IDS),
        "pdk_order": list(FOUR_PDK_IDS),
        "signoffs": [row.canonical_data() for row in ordered],
    }
    _write_atomic_json(Path(destination).resolve(), payload)
    return payload


def _write_atomic_json(path: Path, payload: Mapping[str, object]) -> None:
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


def verify_portfolio_signoff(
    source: str | Path,
    *,
    current_commit: str,
    worktree_dirty: bool,
    implementation_unchanged: bool = False,
    repo_root: str | Path | None = None,
) -> dict[str, object]:
    """Revalidate legacy/v2 ledger bytes and bound artifacts without rerunning."""

    if worktree_dirty:
        raise PortfolioSignoffError(
            "portfolio verification refuses a dirty implementation worktree"
        )
    path = Path(source).resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PortfolioSignoffError(
            f"portfolio ledger is missing or invalid: {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise PortfolioSignoffError("portfolio ledger root is not an object")
    schema = payload.get("schema")
    if schema not in {PORTFOLIO_LEDGER_V1_SCHEMA, PORTFOLIO_LEDGER_V2_SCHEMA}:
        raise PortfolioSignoffError("portfolio ledger schema is not supported")
    if schema == PORTFOLIO_LEDGER_V1_SCHEMA:
        if payload.get("signed_off") is not True:
            raise PortfolioSignoffError("legacy physical portfolio is not signed off")
    else:
        if payload.get("claim_scope") != PHYSICAL_CLAIM_SCOPE:
            raise PortfolioSignoffError("portfolio claim scope is unknown")
        if payload.get("physical_status") != VerificationStatus.PASS.value:
            raise PortfolioSignoffError("portfolio physical status is not pass")
        if payload.get("physical_signed_off") is not True:
            raise PortfolioSignoffError("portfolio physical signoff is withheld")
        if "signed_off" in payload:
            raise PortfolioSignoffError("v2 portfolio contains ambiguous signed_off")
    ledger_commit = str(payload.get("commit", "")).casefold()
    current = str(current_commit).casefold()
    if _COMMIT.fullmatch(current) is None:
        raise PortfolioSignoffError("current commit is not a full Git object id")
    if ledger_commit != current and not implementation_unchanged:
        raise PortfolioSignoffError(
            "portfolio ledger belongs to a different implementation commit"
        )
    if payload.get("circuit_order") != list(PORTFOLIO_CIRCUIT_IDS):
        raise PortfolioSignoffError("portfolio circuit order differs")
    if payload.get("pdk_order") != list(FOUR_PDK_IDS):
        raise PortfolioSignoffError("portfolio PDK order differs")
    raw_rows = payload.get("signoffs")
    if not isinstance(raw_rows, list):
        raise PortfolioSignoffError("portfolio ledger has no signoff rows")
    rows = tuple(portfolio_row_from_mapping(row) for row in raw_rows)
    ordered = validate_portfolio_rows(
        rows,
        current_commit=ledger_commit,
        repo_root=repo_root,
    )
    if [row.canonical_data() for row in ordered] != raw_rows:
        raise PortfolioSignoffError("portfolio signoff rows are not canonical")
    return payload


def baseline_comparison_receipt_from_mapping(
    data: Mapping[str, object],
) -> BaselineComparisonReceipt:
    if data.get("schema") != "clay.baseline-comparison-receipt/v1":
        raise PortfolioSignoffError("baseline comparison receipt schema is unsupported")
    fields = _strict_dataclass_payload(data, BaselineComparisonReceipt)
    try:
        return BaselineComparisonReceipt(**fields)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise PortfolioSignoffError(f"baseline comparison receipt is invalid: {exc}") from exc


def validate_baseline_comparison_receipt(
    receipt: BaselineComparisonReceipt,
    *,
    current_commit: str,
    is_ancestor: Callable[[str, str], bool],
    expected_bindings: Mapping[str, object] | None = None,
) -> BaselineComparisonReceipt:
    """Reject stale, future, cross-candidate, or mutated comparison evidence."""

    current = str(current_commit).casefold()
    evidence = str(receipt.evidence_commit).casefold()
    if _COMMIT.fullmatch(current) is None or _COMMIT.fullmatch(evidence) is None:
        raise PortfolioSignoffError("baseline comparison commit is invalid")
    if evidence == current or not is_ancestor(evidence, current):
        raise PortfolioSignoffError(
            "baseline evidence commit must be a strict ancestor of current commit"
        )
    for label, digest in (
        ("physical row", receipt.physical_row_sha256),
        ("baseline GDS", receipt.baseline_gds_sha256),
        ("baseline candidate", receipt.baseline_candidate_sha256),
        ("new GDS", receipt.new_gds_sha256),
        ("new candidate", receipt.new_candidate_sha256),
    ):
        _require_sha(digest, label=f"baseline comparison {label} digest")
    if receipt.baseline_candidate_sha256 == receipt.new_candidate_sha256:
        raise PortfolioSignoffError("baseline comparison uses the same candidate")
    if receipt.baseline_gds_sha256 == receipt.new_gds_sha256:
        raise PortfolioSignoffError("baseline comparison uses the same GDS")
    if not receipt.circuit_id or not receipt.topology_id or not receipt.pdk_id:
        raise PortfolioSignoffError("baseline comparison identity is incomplete")
    if receipt.comparator not in {"lt", "le", "gt", "ge"}:
        raise PortfolioSignoffError("baseline comparison comparator is unsupported")
    for label, value in (
        ("baseline area", receipt.baseline_area_um2),
        ("new area", receipt.new_area_um2),
        ("threshold", receipt.threshold),
        ("epsilon", receipt.epsilon),
    ):
        if type(value) not in {int, float} or not math.isfinite(float(value)):
            raise PortfolioSignoffError(f"baseline comparison {label} is invalid")
    if (
        receipt.baseline_area_um2 <= 0
        or receipt.new_area_um2 <= 0
        or receipt.threshold <= 0
        or receipt.epsilon < 0
    ):
        raise PortfolioSignoffError(
            "baseline comparison area/threshold/epsilon is invalid"
        )
    if not receipt.algorithm.strip() or not receipt.algorithm_version.strip():
        raise PortfolioSignoffError("baseline comparison algorithm identity is incomplete")
    if receipt.algorithm != "bbox-area-ratio":
        raise PortfolioSignoffError(
            "baseline comparison algorithm is unsupported"
        )
    observed = float(receipt.new_area_um2) / float(receipt.baseline_area_um2)
    threshold = float(receipt.threshold)
    epsilon = float(receipt.epsilon)
    comparison_passed = {
        "lt": observed < threshold - epsilon,
        "le": observed <= threshold + epsilon,
        "gt": observed > threshold + epsilon,
        "ge": observed >= threshold - epsilon,
    }[receipt.comparator]
    if not comparison_passed:
        raise PortfolioSignoffError(
            "baseline comparison does not satisfy its comparator/threshold"
        )
    if expected_bindings is not None:
        actual = asdict(receipt)
        unknown = set(expected_bindings) - set(actual)
        if unknown:
            raise PortfolioSignoffError(
                f"baseline comparison expected bindings are unknown: {sorted(unknown)}"
            )
        for field, expected in expected_bindings.items():
            if actual[field] != expected:
                raise PortfolioSignoffError(
                    f"baseline comparison {field.replace('_', ' ')} was mutated"
                )
    return receipt


def _strict_repo_relative_path(value: object, *, label: str) -> str:
    if type(value) is not str or not value or "\\" in value:
        raise PortfolioSignoffError(f"{label} path is not canonical repo-relative")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise PortfolioSignoffError(f"{label} path is not canonical repo-relative")
    return path.as_posix()


def _git_tree_blob(
    repository: Path,
    commit: str,
    relative_path: str,
    *,
    label: str,
) -> bytes:
    path = _strict_repo_relative_path(relative_path, label=label)
    try:
        completed = subprocess.run(
            ("git", "cat-file", "blob", f"{commit}:{path}"),
            cwd=repository,
            check=False,
            capture_output=True,
        )
    except OSError as exc:
        raise PortfolioSignoffError(f"{label} Git-tree blob is unavailable") from exc
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        suffix = f": {detail}" if detail else ""
        raise PortfolioSignoffError(f"{label} Git-tree blob is unavailable{suffix}")
    data = completed.stdout
    if data.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise PortfolioSignoffError(f"{label} is an unresolved Git LFS pointer")
    if not data:
        raise PortfolioSignoffError(f"{label} Git-tree blob is empty")
    return data


def _git_is_ancestor(repository: Path, ancestor: str, descendant: str) -> bool:
    try:
        return (
            subprocess.run(
                ("git", "merge-base", "--is-ancestor", ancestor, descendant),
                cwd=repository,
                check=False,
                capture_output=True,
            ).returncode
            == 0
        )
    except OSError:
        return False


def _strict_ancestor_baseline_row(
    receipt: BaselineComparisonReceipt,
    *,
    repository: Path,
    ledger_path: str,
) -> tuple[PortfolioSignoffRow, ArtifactLocator, ArtifactLocator]:
    ledger_relative = _strict_repo_relative_path(
        ledger_path, label="baseline physical ledger"
    )
    ledger_bytes = _git_tree_blob(
        repository,
        receipt.evidence_commit,
        ledger_relative,
        label="baseline physical ledger",
    )
    try:
        payload = json.loads(ledger_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PortfolioSignoffError("baseline physical ledger is invalid JSON") from exc
    if not isinstance(payload, dict):
        raise PortfolioSignoffError("baseline physical ledger root is invalid")
    canonical_ledger = (
        json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )
    if ledger_bytes != canonical_ledger:
        raise PortfolioSignoffError("baseline physical ledger bytes are non-canonical")
    required_fields = {
        "schema",
        "claim_scope",
        "physical_status",
        "physical_signed_off",
        "commit",
        "source_state_sha256",
        "intent",
        "summary",
        "signoffs",
        "portfolio_rows",
    }
    if set(payload) != required_fields:
        raise PortfolioSignoffError("baseline physical ledger fields are non-canonical")
    if (
        payload.get("schema") != "clay.physical-pdk-matrix/v2"
        or payload.get("claim_scope") != PHYSICAL_CLAIM_SCOPE
        or payload.get("physical_status") != "pass"
        or payload.get("physical_signed_off") is not True
    ):
        raise PortfolioSignoffError("baseline physical ledger is not signed physical evidence")
    ledger_commit = payload.get("commit")
    if type(ledger_commit) is not str or _COMMIT.fullmatch(ledger_commit) is None:
        raise PortfolioSignoffError("baseline physical ledger commit is invalid")
    if ledger_commit != receipt.evidence_commit and not _git_is_ancestor(
        repository, ledger_commit, receipt.evidence_commit
    ):
        raise PortfolioSignoffError(
            "baseline physical ledger implementation is not an evidence ancestor"
        )
    _require_sha(
        payload.get("source_state_sha256"),
        label="baseline physical ledger source-state digest",
    )
    raw_rows = payload.get("portfolio_rows")
    if (
        type(raw_rows) is not list
        or raw_rows != payload.get("signoffs")
        or len(raw_rows) != len(FOUR_PDK_IDS)
    ):
        raise PortfolioSignoffError("baseline physical ledger rows are non-canonical")
    parsed_rows: list[PortfolioSignoffRow] = []
    for expected_pdk, raw_row in zip(FOUR_PDK_IDS, raw_rows):
        if not isinstance(raw_row, Mapping):
            raise PortfolioSignoffError("baseline physical ledger row is invalid")
        row = portfolio_row_from_mapping(raw_row)
        if (
            not isinstance(row, PortfolioSignoffRow)
            or row.pdk_id != expected_pdk
            or row.commit != ledger_commit
            or row.canonical_data() != raw_row
            or row.claim_scope != PHYSICAL_CLAIM_SCOPE
            or row.physical_status is not VerificationStatus.PASS
            or row.physical_signed_off is not True
        ):
            raise PortfolioSignoffError("baseline physical ledger row is non-canonical")
        parsed_rows.append(row)
    digest_matches = [
        row
        for row in parsed_rows
        if canonical_portfolio_row_sha256(row) == receipt.physical_row_sha256
    ]
    if len(digest_matches) != 1:
        raise PortfolioSignoffError(
            "baseline physical row digest does not identify exactly one ledger row"
        )
    row = digest_matches[0]
    if (
        row.circuit_id != receipt.circuit_id
        or row.topology_id != receipt.topology_id
        or row.pdk_id != receipt.pdk_id
    ):
        raise PortfolioSignoffError("baseline physical row identity differs from receipt")
    by_id: dict[str, ArtifactLocator] = {}
    for locator in row.artifacts:
        if locator.logical_id in by_id:
            raise PortfolioSignoffError(
                f"baseline physical row has duplicate {locator.logical_id} locator"
            )
        by_id[locator.logical_id] = locator
    if _REQUIRED_ARTIFACT_IDS - set(by_id):
        raise PortfolioSignoffError("baseline physical row artifact locators are incomplete")
    return row, by_id["candidate"], by_id["gds"]


def _git_cas_artifact_bytes(
    repository: Path,
    commit: str,
    locator: ArtifactLocator,
    *,
    label: str,
) -> bytes:
    digest = _require_sha(locator.sha256, label=f"{label} locator digest")
    path = _strict_repo_relative_path(locator.path, label=label)
    parts = PurePosixPath(path).parts
    prefix = ("signoff", "artifacts", "sha256")
    if (
        locator.artifact_type != "file"
        or locator.required is not True
        or len(parts) != len(prefix) + 2
        or parts[: len(prefix)] != prefix
        or parts[len(prefix)] != digest
    ):
        raise PortfolioSignoffError(f"{label} locator is not canonical content-addressed data")
    data = _git_tree_blob(repository, commit, path, label=label)
    if hashlib.sha256(data).hexdigest() != digest:
        raise PortfolioSignoffError(f"{label} Git-tree blob digest mismatch")
    return data


def _current_baseline_locators(
    current_row: PortfolioSignoffRow | Mapping[str, object],
    receipt: BaselineComparisonReceipt,
) -> tuple[ArtifactLocator, ArtifactLocator]:
    if isinstance(current_row, PortfolioSignoffRow):
        if (
            current_row.circuit_id != receipt.circuit_id
            or current_row.topology_id != receipt.topology_id
            or current_row.pdk_id != receipt.pdk_id
        ):
            raise PortfolioSignoffError("current physical row identity differs from receipt")
        by_id = {locator.logical_id: locator for locator in current_row.artifacts}
        if len(by_id) != len(current_row.artifacts):
            raise PortfolioSignoffError("current physical row has duplicate artifact locators")
        try:
            return by_id["candidate"], by_id["gds"]
        except KeyError as exc:
            raise PortfolioSignoffError(
                "current physical row candidate/GDS locators are incomplete"
            ) from exc
    if not isinstance(current_row, Mapping):
        raise PortfolioSignoffError("current physical row is invalid")
    pdk = current_row.get("pdk", current_row.get("pdk_id"))
    topology = current_row.get("topology_id")
    circuit = current_row.get("circuit_id", receipt.circuit_id)
    if (
        pdk != receipt.pdk_id
        or topology != receipt.topology_id
        or circuit != receipt.circuit_id
    ):
        raise PortfolioSignoffError("current physical row identity differs from receipt")

    def raw_locator(logical_id: str) -> ArtifactLocator:
        raw = current_row.get(logical_id)
        if not isinstance(raw, Mapping):
            raise PortfolioSignoffError(
                f"current physical row {logical_id} locator is incomplete"
            )
        path = raw.get("path")
        digest = raw.get("sha256")
        if type(path) is not str:
            raise PortfolioSignoffError(
                f"current physical row {logical_id} path is invalid"
            )
        return ArtifactLocator(
            logical_id=logical_id,
            path=path,
            sha256=_require_sha(
                digest, label=f"current physical row {logical_id} digest"
            ),
        )

    return raw_locator("candidate"), raw_locator("gds")


def _current_artifact_bytes(
    repository: Path,
    locator: ArtifactLocator,
    *,
    label: str,
) -> bytes:
    raw_path = Path(locator.path)
    path = raw_path if raw_path.is_absolute() else repository / raw_path
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(repository)
    except (OSError, ValueError) as exc:
        raise PortfolioSignoffError(f"{label} is outside the current repository") from exc
    if not resolved.is_file():
        raise PortfolioSignoffError(f"{label} is not a regular file")
    try:
        data = resolved.read_bytes()
    except OSError as exc:
        raise PortfolioSignoffError(f"{label} bytes are unavailable") from exc
    if data.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise PortfolioSignoffError(f"{label} is an unresolved Git LFS pointer")
    if not data or hashlib.sha256(data).hexdigest() != locator.sha256:
        raise PortfolioSignoffError(f"{label} digest differs from current row locator")
    return data


def _gds_bbox_area_from_bytes(data: bytes, *, top_cell: str, label: str) -> float:
    from clay.flow.physical_matrix import gds_geometry_metrics

    try:
        with tempfile.TemporaryDirectory(prefix="clay-baseline-gds-") as temporary:
            path = Path(temporary) / "evidence.gds"
            path.write_bytes(data)
            area = gds_geometry_metrics(path, top_cell=top_cell)["bbox_area_um2"]
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise PortfolioSignoffError(f"{label} geometry is unmeasurable") from exc
    if type(area) not in {int, float} or not math.isfinite(float(area)) or area <= 0:
        raise PortfolioSignoffError(f"{label} bbox area is invalid")
    return float(area)


def validate_baseline_comparison_evidence(
    receipt: BaselineComparisonReceipt,
    *,
    current_commit: str,
    repo_root: str | Path,
    current_row: PortfolioSignoffRow | Mapping[str, object],
    evidence_ledger_path: str,
    gds_top_cell: str,
    is_ancestor: Callable[[str, str], bool] | None = None,
    area_epsilon_um2: float = BASELINE_GDS_AREA_ABS_EPSILON_UM2,
    expected_bindings: Mapping[str, object] | None = None,
) -> BaselineComparisonReceipt:
    """Bind a receipt to ancestor Git blobs and current-row artifact bytes."""

    repository = Path(repo_root).resolve()
    if not repository.is_dir():
        raise PortfolioSignoffError("baseline comparison repository is unavailable")
    if (
        type(area_epsilon_um2) not in {int, float}
        or not math.isfinite(float(area_epsilon_um2))
        or area_epsilon_um2 <= 0
    ):
        raise PortfolioSignoffError("baseline comparison area epsilon is invalid")
    ancestor_check = (
        (lambda ancestor, current: _git_is_ancestor(repository, ancestor, current))
        if is_ancestor is None
        else is_ancestor
    )
    validate_baseline_comparison_receipt(
        receipt,
        current_commit=current_commit,
        is_ancestor=ancestor_check,
        expected_bindings=expected_bindings,
    )
    baseline_row, baseline_candidate, baseline_gds = _strict_ancestor_baseline_row(
        receipt,
        repository=repository,
        ledger_path=evidence_ledger_path,
    )
    baseline_candidate_bytes = _git_cas_artifact_bytes(
        repository,
        receipt.evidence_commit,
        baseline_candidate,
        label="baseline candidate",
    )
    baseline_gds_bytes = _git_cas_artifact_bytes(
        repository,
        receipt.evidence_commit,
        baseline_gds,
        label="baseline GDS",
    )
    current_candidate, current_gds = _current_baseline_locators(current_row, receipt)
    current_candidate_bytes = _current_artifact_bytes(
        repository, current_candidate, label="current candidate"
    )
    current_gds_bytes = _current_artifact_bytes(
        repository, current_gds, label="current GDS"
    )
    baseline_area = _gds_bbox_area_from_bytes(
        baseline_gds_bytes, top_cell=gds_top_cell, label="baseline GDS"
    )
    current_area = _gds_bbox_area_from_bytes(
        current_gds_bytes, top_cell=gds_top_cell, label="current GDS"
    )
    epsilon = float(area_epsilon_um2)
    if not math.isclose(
        float(receipt.baseline_area_um2), baseline_area, rel_tol=0.0, abs_tol=epsilon
    ):
        raise PortfolioSignoffError("baseline comparison baseline area was not remeasured")
    if not math.isclose(
        float(receipt.new_area_um2), current_area, rel_tol=0.0, abs_tol=epsilon
    ):
        raise PortfolioSignoffError("baseline comparison new area was not remeasured")
    artifact_bindings = {
        "circuit_id": baseline_row.circuit_id,
        "topology_id": baseline_row.topology_id,
        "pdk_id": baseline_row.pdk_id,
        "physical_row_sha256": canonical_portfolio_row_sha256(baseline_row),
        "baseline_candidate_sha256": hashlib.sha256(
            baseline_candidate_bytes
        ).hexdigest(),
        "baseline_gds_sha256": hashlib.sha256(baseline_gds_bytes).hexdigest(),
        "new_candidate_sha256": hashlib.sha256(current_candidate_bytes).hexdigest(),
        "new_gds_sha256": hashlib.sha256(current_gds_bytes).hexdigest(),
    }
    combined_bindings = dict(expected_bindings or {})
    combined_bindings.update(artifact_bindings)
    return validate_baseline_comparison_receipt(
        receipt,
        current_commit=current_commit,
        is_ancestor=ancestor_check,
        expected_bindings=combined_bindings,
    )


__all__ = [
    "ArtifactLocator",
    "BASELINE_GDS_AREA_ABS_EPSILON_UM2",
    "BaselineComparisonReceipt",
    "DeckScopeReceipt",
    "LegacyPortfolioSignoffRow",
    "PHYSICAL_CLAIM_SCOPE",
    "PORTFOLIO_CIRCUIT_IDS",
    "PortfolioReadableRow",
    "PortfolioSignoffError",
    "PortfolioSignoffRow",
    "SearchReceipt",
    "baseline_comparison_receipt_from_mapping",
    "canonical_portfolio_row_sha256",
    "normalized_clay_file_connectivity_sha256",
    "normalized_connectivity_sha256",
    "portfolio_row_from_mapping",
    "publish_portfolio_signoff",
    "validate_baseline_comparison_evidence",
    "validate_baseline_comparison_receipt",
    "validate_portfolio_row_artifacts",
    "validate_portfolio_rows",
    "verify_portfolio_signoff",
]
