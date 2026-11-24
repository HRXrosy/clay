"""Typed, fail-closed release contract for claim-bearing pytest nodes."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import uuid
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Final, Iterable


SCHEMA: Final = "clay.test-contract-manifest/v2"
OWNERS: Final = frozenset({"ota", "ldo", "bandgap", "cross_pdk", "shared"})
GATES: Final = frozenset(
    {
        "intent",
        "placement",
        "routing",
        "drc",
        "lvs",
        "body",
        "pex",
        "artifact",
        "ledger",
    }
)
REQUIRED_OWNERS: Final = ("ota", "ldo", "bandgap", "cross_pdk", "shared")
REQUIRED_GATES: Final = (
    "intent",
    "placement",
    "routing",
    "drc",
    "lvs",
    "body",
    "pex",
    "artifact",
    "ledger",
)
PATH_CLASSES: Final = frozenset({"canonical", "legacy"})
TIERS: Final = frozenset({"unit", "integration", "bringup", "signoff"})
CLAIM_SCOPES: Final = frozenset(
    {
        "contract_integrity",
        "reference_identity_contract",
        "reference_quality_observation",
        "physical_artifact_signoff",
        "diagnostic_only",
    }
)
REQUIRED_STATES: Final = frozenset({"pass"})
EXPECTED_VERDICTS: Final = frozenset({"pass", "fail", "unmeasurable"})
REFERENCE_SCOPES: Final = frozenset(
    {"reference_identity_contract", "reference_quality_observation"}
)
PHYSICAL_ARTIFACT_SIGNOFF_NODEIDS: Final = frozenset(
    {
        "tests/test_portfolio_signoff.py::"
        "test_v2_row_is_explicitly_physical_and_cannot_carry_other_verdicts"
    }
)

_TOP_LEVEL_FIELDS: Final = frozenset(
    {"schema", "required_gates", "required_owners", "contracts"}
)
_ROW_FIELDS: Final = frozenset(
    {
        "id",
        "nodeid",
        "owner",
        "gate",
        "circuit_id",
        "topology_id",
        "reference_id",
        "path_class",
        "tier",
        "claim_scope",
        "required_state",
        "expected_verdict",
    }
)
_ID_RE: Final = re.compile(r"[a-z][a-z0-9_]*\Z")


class SuiteContractError(ValueError):
    """The suite contract is ambiguous, incomplete, or over-claims evidence."""


@dataclass(frozen=True)
class ReferenceNodeContract:
    circuit_id: str
    topology_id: str
    reference_id: str
    expected_verdict: str


REFERENCE_QUALITY_NODE_CONTRACTS: Final[
    Mapping[str, ReferenceNodeContract]
] = MappingProxyType(
    {
        (
            "tests/test_canonical_reference_fidelity.py::"
            "test_portable_exact_gds_reports_digest_bound_metric_inventory"
        ): ReferenceNodeContract(
            circuit_id="two_stage_ota",
            topology_id="ota_cs_ref_extracted_r1",
            reference_id="ota_dev/ref/reference_manifest.json",
            expected_verdict="fail",
        )
    }
)
REFERENCE_IDENTITY_NODE_CONTRACTS: Final[
    Mapping[str, ReferenceNodeContract]
] = MappingProxyType(
    {
        (
            "tests/test_reference_ldo_source.py::"
            "test_tracked_netlist_is_bound_to_typed_canonical_reference"
        ): ReferenceNodeContract(
            circuit_id="reference_ldo",
            topology_id="ldo1_human_r1",
            reference_id="ldo_dev/ref/reference_manifest.json",
            expected_verdict="pass",
        ),
        (
            "tests/test_reference_bandgap_workspace.py::"
            "test_bandgap_physical_rows_use_typed_reference_manifest_identity"
        ): ReferenceNodeContract(
            circuit_id="bandgap",
            topology_id="bandgap_human_r1",
            reference_id="bandgap_dev/ref/reference_manifest.json",
            expected_verdict="pass",
        ),
    }
)


@dataclass(frozen=True)
class SuiteContractRow:
    id: str
    nodeid: str
    owner: str
    gate: str
    circuit_id: str | None
    topology_id: str | None
    reference_id: str | None
    path_class: str
    tier: str
    claim_scope: str
    required_state: str
    expected_verdict: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class SuiteContractManifest:
    schema: str
    required_gates: tuple[str, ...]
    required_owners: tuple[str, ...]
    contracts: tuple[SuiteContractRow, ...]

    @property
    def by_nodeid(self) -> dict[str, SuiteContractRow]:
        return {row.nodeid: row for row in self.contracts}


@dataclass(frozen=True)
class SuiteContractVerificationReport:
    rows: tuple[SuiteContractRow, ...]
    passed: int
    skipped: int
    xfailed: int


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise SuiteContractError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise SuiteContractError(f"non-finite JSON constant is forbidden: {value}")


def _object(value: object, *, label: str) -> dict[str, object]:
    if type(value) is not dict:
        raise SuiteContractError(f"{label} must be a JSON object")
    return value


def _closed_fields(
    value: dict[str, object], expected: frozenset[str], *, label: str
) -> None:
    unknown = sorted(set(value) - expected)
    missing = sorted(expected - set(value))
    if unknown:
        raise SuiteContractError(f"{label} has unknown field(s): {', '.join(unknown)}")
    if missing:
        raise SuiteContractError(f"{label} is missing field(s): {', '.join(missing)}")


def _string(value: object, *, label: str) -> str:
    if type(value) is not str or not value:
        raise SuiteContractError(f"{label} must be a non-empty string")
    return value


def _nullable_string(value: object, *, label: str) -> str | None:
    if value is None:
        return None
    return _string(value, label=label)


def _closed_string_list(
    value: object, *, label: str, allowed: frozenset[str]
) -> tuple[str, ...]:
    if type(value) is not list or not value:
        raise SuiteContractError(f"{label} must be a non-empty array")
    items = tuple(_string(item, label=f"{label} item") for item in value)
    if len(set(items)) != len(items):
        raise SuiteContractError(f"{label} contains duplicate values")
    unknown = sorted(set(items) - allowed)
    if unknown:
        raise SuiteContractError(
            f"{label} contains unknown value(s): {', '.join(unknown)}"
        )
    return items


def _enum(value: object, *, label: str, allowed: frozenset[str]) -> str:
    result = _string(value, label=label)
    if result not in allowed:
        choices = ", ".join(sorted(allowed))
        raise SuiteContractError(f"{label} must be one of: {choices}; got {result!r}")
    return result


def _validate_nodeid(nodeid: str, *, repo_root: Path | None) -> None:
    if "\\" in nodeid or nodeid.count("::") < 1:
        raise SuiteContractError(
            f"contract nodeid must be an exact forward-slash pytest node: {nodeid!r}"
        )
    file_name, selector = nodeid.split("::", 1)
    path = PurePosixPath(file_name)
    selector_parts = selector.split("::")
    if (
        path.is_absolute()
        or not path.parts
        or path.parts[0] != "tests"
        or any(part in {"", ".", ".."} for part in path.parts)
        or not path.name.startswith("test_")
        or path.suffix != ".py"
        or not selector
        or selector.startswith("[")
        or any(
            part in {"", ".", ".."} or "/" in part or "\\" in part
            for part in selector_parts
        )
    ):
        raise SuiteContractError(f"invalid contract nodeid: {nodeid!r}")
    if repo_root is None:
        return
    root = repo_root.resolve()
    target = (root / Path(*path.parts)).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise SuiteContractError(
            f"contract nodeid escapes repository: {nodeid!r}"
        ) from exc
    if not target.is_file():
        raise SuiteContractError(f"contract nodeid file does not exist: {nodeid!r}")


def _row(value: object, *, index: int, repo_root: Path | None) -> SuiteContractRow:
    raw = _object(value, label=f"contracts[{index}]")
    _closed_fields(raw, _ROW_FIELDS, label=f"contracts[{index}]")
    contract_id = _string(raw["id"], label=f"contracts[{index}].id")
    if not _ID_RE.fullmatch(contract_id):
        raise SuiteContractError(
            f"contracts[{index}].id must match [a-z][a-z0-9_]*"
        )
    nodeid = _string(raw["nodeid"], label=f"contracts[{index}].nodeid")
    _validate_nodeid(nodeid, repo_root=repo_root)
    owner = _enum(raw["owner"], label=f"contracts[{index}].owner", allowed=OWNERS)
    gate = _enum(raw["gate"], label=f"contracts[{index}].gate", allowed=GATES)
    circuit_id = _nullable_string(
        raw["circuit_id"], label=f"contracts[{index}].circuit_id"
    )
    topology_id = _nullable_string(
        raw["topology_id"], label=f"contracts[{index}].topology_id"
    )
    reference_id = _nullable_string(
        raw["reference_id"], label=f"contracts[{index}].reference_id"
    )
    path_class = _enum(
        raw["path_class"],
        label=f"contracts[{index}].path_class",
        allowed=PATH_CLASSES,
    )
    tier = _enum(raw["tier"], label=f"contracts[{index}].tier", allowed=TIERS)
    claim_scope = _enum(
        raw["claim_scope"],
        label=f"contracts[{index}].claim_scope",
        allowed=CLAIM_SCOPES,
    )
    required_state = _enum(
        raw["required_state"],
        label=f"contracts[{index}].required_state",
        allowed=REQUIRED_STATES,
    )
    expected_verdict = _enum(
        raw["expected_verdict"],
        label=f"contracts[{index}].expected_verdict",
        allowed=EXPECTED_VERDICTS,
    )
    if claim_scope == "physical_artifact_signoff" and path_class == "legacy":
        raise SuiteContractError("physical_artifact_signoff cannot use a legacy path")
    if claim_scope == "physical_artifact_signoff" and tier == "bringup":
        raise SuiteContractError("physical_artifact_signoff cannot use bringup")
    if path_class == "legacy" and claim_scope != "diagnostic_only":
        raise SuiteContractError("legacy contracts are diagnostic_only")
    if tier == "bringup" and claim_scope != "diagnostic_only":
        raise SuiteContractError("bringup contracts are diagnostic_only")
    identity = (circuit_id, topology_id, reference_id)
    if claim_scope == "physical_artifact_signoff":
        if expected_verdict != "pass":
            raise SuiteContractError(
                "physical_artifact_signoff expected_verdict must be pass"
            )
        if nodeid not in PHYSICAL_ARTIFACT_SIGNOFF_NODEIDS:
            raise SuiteContractError(
                "physical_artifact_signoff must bind a code-owned portfolio "
                "physical legality node"
            )
        if any(item is not None for item in identity):
            raise SuiteContractError(
                "aggregate physical_artifact_signoff identity fields must be null"
            )
    elif claim_scope in REFERENCE_SCOPES and any(item is None for item in identity):
        raise SuiteContractError(
            f"{claim_scope} requires circuit_id/topology_id/reference_id"
        )
    elif claim_scope not in REFERENCE_SCOPES and any(
        item is not None for item in identity
    ):
        raise SuiteContractError(
            "non-reference contract circuit_id/topology_id/reference_id must be null"
        )
    if claim_scope == "reference_identity_contract" and expected_verdict != "pass":
        raise SuiteContractError(
            "reference_identity_contract expected_verdict must be pass"
        )
    if claim_scope in {"contract_integrity", "diagnostic_only"} and (
        expected_verdict != "pass"
    ):
        raise SuiteContractError(
            "non-reference contract expected_verdict must be pass"
        )
    if nodeid in REFERENCE_QUALITY_NODE_CONTRACTS:
        reference_contract = REFERENCE_QUALITY_NODE_CONTRACTS[nodeid]
        reference_scope = "reference_quality_observation"
    elif nodeid in REFERENCE_IDENTITY_NODE_CONTRACTS:
        reference_contract = REFERENCE_IDENTITY_NODE_CONTRACTS[nodeid]
        reference_scope = "reference_identity_contract"
    elif claim_scope in REFERENCE_SCOPES:
        raise SuiteContractError(
            "reference claim must bind a code-owned exact reference node contract"
        )
    else:
        reference_contract = None
        reference_scope = None
    if reference_contract is not None and (
        claim_scope != reference_scope
        or circuit_id != reference_contract.circuit_id
        or topology_id != reference_contract.topology_id
        or reference_id != reference_contract.reference_id
        or expected_verdict != reference_contract.expected_verdict
    ):
        raise SuiteContractError(
            "code-owned reference node contract requires exact claim_scope, "
            "circuit_id, topology_id, reference_id, and expected_verdict"
        )
    return SuiteContractRow(
        id=contract_id,
        nodeid=nodeid,
        owner=owner,
        gate=gate,
        circuit_id=circuit_id,
        topology_id=topology_id,
        reference_id=reference_id,
        path_class=path_class,
        tier=tier,
        claim_scope=claim_scope,
        required_state=required_state,
        expected_verdict=expected_verdict,
    )


def load_suite_contract_manifest(
    path: str | Path, *, repo_root: str | Path | None = None
) -> SuiteContractManifest:
    """Load one v2 manifest without accepting JSON/schema ambiguity."""

    manifest_path = Path(path)
    try:
        payload = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SuiteContractError(
            f"cannot load suite contract {manifest_path}: {exc}"
        ) from exc
    raw = _object(payload, label="manifest")
    _closed_fields(raw, _TOP_LEVEL_FIELDS, label="manifest")
    if raw["schema"] != SCHEMA:
        raise SuiteContractError(f"manifest schema must be {SCHEMA!r}")
    required_gates = _closed_string_list(
        raw["required_gates"], label="required_gates", allowed=GATES
    )
    required_owners = _closed_string_list(
        raw["required_owners"], label="required_owners", allowed=OWNERS
    )
    if required_gates != REQUIRED_GATES:
        raise SuiteContractError("required_gates must equal the code-owned exact tuple")
    if required_owners != REQUIRED_OWNERS:
        raise SuiteContractError("required_owners must equal the code-owned exact tuple")
    contracts_value = raw["contracts"]
    if type(contracts_value) is not list or not contracts_value:
        raise SuiteContractError("contracts must be a non-empty array")
    root = Path(repo_root) if repo_root is not None else None
    contracts = tuple(
        _row(value, index=index, repo_root=root)
        for index, value in enumerate(contracts_value)
    )
    ids = Counter(row.id for row in contracts)
    nodeids = Counter(row.nodeid for row in contracts)
    for label, counts in (("id", ids), ("nodeid", nodeids)):
        duplicates = sorted(key for key, count in counts.items() if count != 1)
        if duplicates:
            raise SuiteContractError(
                f"duplicate contract {label}: {', '.join(duplicates)}"
            )
    missing_gates = sorted(set(required_gates) - {row.gate for row in contracts})
    missing_owners = sorted(set(required_owners) - {row.owner for row in contracts})
    if missing_gates:
        raise SuiteContractError(
            f"required gate has no exact contract node: {', '.join(missing_gates)}"
        )
    if missing_owners:
        raise SuiteContractError(
            f"required owner has no exact contract node: {', '.join(missing_owners)}"
        )
    return SuiteContractManifest(
        schema=SCHEMA,
        required_gates=required_gates,
        required_owners=required_owners,
        contracts=contracts,
    )


def validate_contract_collection(
    manifest: SuiteContractManifest, collected_nodeids: Iterable[str]
) -> None:
    """Require every declared node to be collected exactly once."""

    counts = Counter(nodeid.replace("\\", "/") for nodeid in collected_nodeids)
    missing = sorted(
        row.nodeid for row in manifest.contracts if counts[row.nodeid] == 0
    )
    duplicate = sorted(
        row.nodeid for row in manifest.contracts if counts[row.nodeid] > 1
    )
    if missing:
        raise SuiteContractError(f"missing contract nodeid: {', '.join(missing)}")
    if duplicate:
        raise SuiteContractError(
            f"contract nodeid collected more than once: {', '.join(duplicate)}"
        )


def validate_reference_claims(
    manifest: SuiteContractManifest, *, repo_root: str | Path
) -> None:
    """Cross-bind every reference claim to the registry and typed manifest."""

    from clay.flow.reference_evidence import (
        load_registered_reference_manifest,
        load_topology_registry,
    )

    reference_rows = tuple(
        row
        for row in manifest.contracts
        if row.claim_scope in REFERENCE_SCOPES
    )
    if not reference_rows:
        return
    root = Path(repo_root).resolve()
    registry_path = root / "signoff" / "topology_registry.json"
    registry = load_topology_registry(registry_path, repo_root=root)
    entries = registry.by_id()
    for row in reference_rows:
        assert row.topology_id is not None
        assert row.circuit_id is not None
        assert row.reference_id is not None
        entry = entries.get(row.topology_id)
        if entry is None or entry.circuit_id != row.circuit_id:
            raise SuiteContractError(
                f"suite contract registry identity differs for {row.id!r}"
            )
        locator = entry.reference_manifest
        if locator is None or locator.repository_path != row.reference_id:
            raise SuiteContractError(
                f"suite contract typed reference identity differs for {row.id!r}"
            )
        typed = load_registered_reference_manifest(
            registry_path,
            topology_id=row.topology_id,
            repo_root=root,
        )
        if typed.circuit_id != row.circuit_id or typed.topology_id != row.topology_id:
            raise SuiteContractError(
                f"suite contract typed reference manifest differs for {row.id!r}"
            )


def _collect_exact_node(repo_root: Path, nodeid: str) -> None:
    result = subprocess.run(
        [sys.executable, "-B", "-m", "pytest", "--collect-only", "-q", nodeid],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=240,
    )
    collected = [
        line.strip().replace("\\", "/")
        for line in result.stdout.splitlines()
        if line.strip().startswith("tests/") and "::" in line
    ]
    if result.returncode != 0 or collected != [nodeid]:
        detail = result.stdout + "\n" + result.stderr
        if len(collected) > 1:
            reason = "collected more than once; use an exact parameter ID"
        else:
            reason = "did not collect exactly once or node was not found"
        raise SuiteContractError(f"{nodeid}: {reason}\n{detail}")


def _execute_required_pass(repo_root: Path, row: SuiteContractRow) -> None:
    if "CLAY_SUITE_CONTRACT_TMP" in os.environ:
        raise SuiteContractError(
            "suite-contract temp root override is forbidden; basetemp must be "
            "repository-local"
        )
    root = repo_root.resolve()
    # The parent creates a unique gitignored leaf, then the child pytest process
    # deletes it from `pytest_unconfigure` while it still owns the Windows ACL.
    # The cleanup plugin cross-checks pytest's invocation root and the exact
    # `--basetemp` before removing anything.
    basetemp = root / (
        f".pytest-suite-contract-{os.getpid()}-{uuid.uuid4().hex}"
    )
    basetemp.mkdir(parents=False, exist_ok=False)
    child_env = os.environ.copy()
    child_env.pop("CLAY_SUITE_CONTRACT_BASETEMP", None)
    child_env.pop("CLAY_SUITE_CONTRACT_REPO_ROOT", None)
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "pytest",
            "-q",
            "-rA",
            "-p",
            "no:cacheprovider",
            "-p",
            "clay.testing.suite_contract_cleanup",
            "--basetemp",
            str(basetemp),
            row.nodeid,
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=900,
        env=child_env,
    )
    output = result.stdout + "\n" + result.stderr
    lowered = output.lower()
    forbidden = (" skipped", " xfailed", " xpassed", " failed", " error")
    if result.returncode != 0 or any(token in lowered for token in forbidden):
        raise SuiteContractError(
            f"contract {row.id!r} required PASS without skip/xfail:\n{output}"
        )
    if " passed" not in lowered:
        raise SuiteContractError(
            f"contract {row.id!r} required PASS but no pass was reported:\n{output}"
        )


def verify_suite_contract(
    manifest: SuiteContractManifest, *, repo_root: str | Path
) -> SuiteContractVerificationReport:
    """Independent release gate: collect and execute every exact row."""

    root = Path(repo_root).resolve()
    validate_reference_claims(manifest, repo_root=root)
    for row in manifest.contracts:
        _collect_exact_node(root, row.nodeid)
        _execute_required_pass(root, row)
    return SuiteContractVerificationReport(
        rows=manifest.contracts,
        passed=len(manifest.contracts),
        skipped=0,
        xfailed=0,
    )


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Verify the exact Clay suite contract")
    parser.add_argument(
        "manifest", nargs="?", default="tests/suite_contract_manifest.json"
    )
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)
    root = Path(args.repo_root).resolve()
    manifest = load_suite_contract_manifest(root / args.manifest, repo_root=root)
    report = verify_suite_contract(manifest, repo_root=root)
    print(f"suite contract: {report.passed}/{len(report.rows)} PASS; 0 skip; 0 xfail")
    return 0


__all__ = [
    "SCHEMA",
    "REFERENCE_IDENTITY_NODE_CONTRACTS",
    "REFERENCE_QUALITY_NODE_CONTRACTS",
    "REQUIRED_GATES",
    "REQUIRED_OWNERS",
    "SuiteContractError",
    "SuiteContractManifest",
    "SuiteContractRow",
    "SuiteContractVerificationReport",
    "load_suite_contract_manifest",
    "validate_contract_collection",
    "validate_reference_claims",
    "verify_suite_contract",
]


if __name__ == "__main__":
    raise SystemExit(main())
