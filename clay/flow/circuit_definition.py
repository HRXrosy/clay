"""Source-derived design identity for an explicit local case package.

This layer resolves data only. It neither imports a case's layout module nor
executes manifest commands. Execution readiness and physical admission are
separate contracts.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import re
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from clay.casepkg import load_case_manifest
from clay.flow.spec import CircuitSpec, DEFAULT_STEPS, compile_flow_spec
from clay.parser import parse_clay_source

INTENT_FLOW_SCHEMA = "clay.intent-flow/v1"
MODES = ("generate", "structure_transfer", "reference_reconstruction")


class DesignInputError(ValueError):
    def __init__(self, code: str, message: str, *, location: dict | None = None):
        self.code = code
        self.location = location or {}
        super().__init__(message)


def artifact_path(root: Path, locator: str) -> Path:
    """Resolve a POSIX input locator under the caller-selected project root."""
    if (type(locator) is not str or not locator or "\\" in locator
            or PurePosixPath(locator).is_absolute() or PureWindowsPath(locator).drive
            or ".." in PurePosixPath(locator).parts
            or PurePosixPath(locator).as_posix() != locator):
        raise DesignInputError("INPUT_PATH_OUTSIDE_PROJECT", f"invalid project-relative input: {locator!r}")
    path = (root / locator).resolve()
    if not path.is_relative_to(root):
        raise DesignInputError("INPUT_PATH_OUTSIDE_PROJECT", f"input leaves project: {locator}")
    if not path.is_file():
        raise DesignInputError("INPUT_MISSING", f"input file missing: {locator}")
    return path


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _mapping(value: object, label: str, fields: set[str]) -> dict:
    if type(value) is not dict:
        raise DesignInputError("CASE_INPUT_INVALID", f"{label} must be a mapping")
    unknown = set(value) - fields
    if unknown:
        raise DesignInputError("UNKNOWN_FIELD", f"unknown {label} fields: {sorted(unknown)}")
    return value


def _string(value: object, label: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise DesignInputError("CASE_INPUT_INVALID", f"{label} must be a non-empty string")
    return value


@dataclass(frozen=True)
class ReferenceDefinition:
    manifest_artifact: str | None
    geometry_artifact: str
    representation: str
    observation_artifact: str | None = None
    observation_role: str = "primary"
    macro_grammar_artifact: str | None = None
    auxiliary_artifacts: tuple[str, ...] = ()


@dataclass(frozen=True)
class CircuitDefinition:
    key: str
    variant: str
    mode: str
    top_cell: str
    source_artifact: str
    generated_pdk_id: str
    reference: ReferenceDefinition | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InputBinding:
    artifact: str
    sha256: str


@dataclass(frozen=True)
class PreparedDesign:
    definition: CircuitDefinition
    project_root: Path
    case_dir: Path
    source_bytes: bytes
    source_sha256: str
    inputs: tuple[InputBinding, ...]
    flow_spec: CircuitSpec

    def require_unchanged(self) -> None:
        for binding in self.inputs:
            try:
                path = artifact_path(self.project_root, binding.artifact)
                if file_sha256(path) != binding.sha256:
                    raise DesignInputError("INPUT_CHANGED", f"input changed after preparation: {binding.artifact}", location={"artifact": binding.artifact})
            except OSError as exc:
                raise DesignInputError("INPUT_UNREADABLE", "bound input can no longer be read", location={"artifact": binding.artifact}) from exc


def _reference(value: object) -> ReferenceDefinition | None:
    if value is None:
        return None
    row = _mapping(value, "reference", {"manifest", "geometry", "representation", "observation",
                                       "observation_role", "macro_grammar", "auxiliary"})
    geometry = _string(row.get("geometry"), "reference.geometry")
    representation = row.get("representation")
    if representation not in ("flat", "hierarchical"):
        raise DesignInputError("UNSUPPORTED_REFERENCE_FORMAT", "reference representation must be flat or hierarchical")
    role = row.get("observation_role", "primary")
    if role not in ("primary", "auxiliary"):
        raise DesignInputError("CASE_INPUT_INVALID", "observation_role must be primary or auxiliary")
    auxiliary = row.get("auxiliary", [])
    if type(auxiliary) is not list or any(type(p) is not str for p in auxiliary):
        raise DesignInputError("CASE_INPUT_INVALID", "reference.auxiliary must be a list of paths")
    if len(set(auxiliary)) != len(auxiliary) or geometry in auxiliary:
        raise DesignInputError("REFERENCE_ROLE_CONFLICT", "primary and auxiliary geometry roles overlap")
    optional = {key: None if row.get(key) is None else _string(row[key], f"reference.{key}")
                for key in ("manifest", "observation", "macro_grammar")}
    return ReferenceDefinition(optional["manifest"], geometry, representation,
                               optional["observation"], role, optional["macro_grammar"], tuple(auxiliary))


def prepare_design(case_dir: str | Path, *, project_root: str | Path | None = None) -> PreparedDesign:
    """Bind one source/variant from case metadata and compile its source flow.

Inputs are relative to case_dir by default. Repository migration samples use
an explicitly supplied project_root, never an authority selected by case data.
"""
    case = Path(case_dir).resolve()
    root = Path(project_root).resolve() if project_root is not None else case
    if not case.is_relative_to(root):
        raise DesignInputError("INPUT_PATH_OUTSIDE_PROJECT", "case directory leaves explicit project root")
    paths = [p for p in (case / "case.yaml", case / "case.yml", case / "case.json") if p.is_file()]
    if len(paths) != 1:
        raise DesignInputError("CASE_INPUT_INVALID", "case must contain exactly one case.yaml/yml/json")
    manifest_path = paths[0]
    if not manifest_path.resolve().is_relative_to(root):
        raise DesignInputError("INPUT_PATH_OUTSIDE_PROJECT", "case manifest leaves project root")
    initial_sha = file_sha256(manifest_path)
    try:
        manifest = load_case_manifest(case)
    except (ValueError, OSError) as exc:
        raise DesignInputError("CASE_INPUT_INVALID", str(exc)) from exc
    if manifest.get("schema") != "clay.case/v1":
        raise DesignInputError("UNSUPPORTED_SCHEMA", "expected clay.case/v1")
    metadata = manifest.get("metadata", {})
    if type(metadata) is not dict:
        raise DesignInputError("CASE_INPUT_INVALID", "metadata must be a mapping")
    config = _mapping(metadata.get("intent_flow"), "intent_flow", {"schema", "variant", "mode", "reference"})
    if config.get("schema") != INTENT_FLOW_SCHEMA:
        raise DesignInputError("UNSUPPORTED_SCHEMA", f"expected {INTENT_FLOW_SCHEMA}")
    if config.get("mode") not in MODES:
        raise DesignInputError("UNSUPPORTED_MODE", f"mode must be one of {MODES}")
    source = _mapping(manifest.get("source"), "source", {"clay", "top_cell"})
    locator = _string(source.get("clay"), "source.clay")
    top = _string(source.get("top_cell"), "source.top_cell")
    source_bytes = artifact_path(root, locator).read_bytes()
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    try:
        parsed = parse_clay_source(source_bytes.decode("utf-8-sig"))
    except Exception as exc:
        raise DesignInputError("SOURCE_PARSE_FAILED", str(exc), location={
            "artifact": locator, "line": getattr(exc, "line", 1), "column": getattr(exc, "column", 1)}) from exc
    cells = [c for c in parsed.cell_defs if c.local_name == top]
    if len(cells) != 1:
        raise DesignInputError("SOURCE_TOP_MISMATCH", f"source must define exactly one selected cell {top!r}")
    cell = cells[0]
    flow = cell.flow_decl
    if flow is None or cell.quality_decl is None:
        raise DesignInputError("FLOW_DECLARATION_REQUIRED", "selected source requires flow and quality declarations")
    location = dict(flow.source_span or {}, artifact=locator)
    if set(flow.steps) - set(DEFAULT_STEPS):
        raise DesignInputError("UNKNOWN_FLOW_STAGE", f"unknown stages: {sorted(set(flow.steps) - set(DEFAULT_STEPS))}", location=location)
    pdk = _string(flow.pdk or parsed.pdk_id, "source PDK")
    if parsed.pdk_id and pdk != parsed.pdk_id:
        raise DesignInputError("PDK_IDENTITY_MISMATCH", "source and flow PDK identities differ", location=location)
    key = _string(metadata.get("key", manifest.get("id")), "case key")
    if re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", key) is None:
        raise DesignInputError("CASE_ID_INVALID", "case key must be an identifier, not a path")
    # Existing spec compilation is reused, with only non-executable identity
    # metadata. User artifact hints cannot redirect its inspection outside root.
    try:
        spec = compile_flow_spec(cell, case_dir=case, verify_jobs=parsed.verify_jobs,
                                 policies=parsed.policies, discover_layout=False,
                                 inspect_artifacts=False,
                                 metadata={"key": key, "id": manifest.get("id", key), "pdk": pdk})
    except Exception as exc:
        raise DesignInputError("FLOW_COMPILATION_FAILED", str(exc), location=location) from exc
    reference = _reference(config.get("reference"))
    definition = CircuitDefinition(key, _string(config.get("variant"), "variant"), config["mode"],
                                   top, locator, pdk, reference)
    bindings = {manifest_path.relative_to(root).as_posix(): initial_sha, locator: source_sha}
    if reference:
        for relative in (reference.manifest_artifact, reference.geometry_artifact, reference.observation_artifact,
                         reference.macro_grammar_artifact, *reference.auxiliary_artifacts):
            if relative is not None:
                bindings[relative] = file_sha256(artifact_path(root, relative))
    prepared = PreparedDesign(definition, root, case, source_bytes, source_sha,
                               tuple(InputBinding(k, v) for k, v in sorted(bindings.items())), spec)
    prepared.require_unchanged()
    return prepared
