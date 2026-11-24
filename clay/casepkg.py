"""Standard Clay case-package discovery, validation, and dry-run support."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from pathlib import Path
import re
import shlex
import subprocess
from typing import Any


CASE_SCHEMA = "clay.case/v1"
CATALOG_SCHEMA = "clay.case_catalog/v1"

MATURITIES: tuple[str, ...] = (
    "L0_SPEC",
    "L1_PARSE",
    "L2_COMPILE",
    "L3_SIM",
    "L4_LAYOUT_INTENT",
    "L5_SIGNOFF",
)
MATURITY_RANK: dict[str, int] = {name: i for i, name in enumerate(MATURITIES)}

_MANIFEST_NAMES = ("case.yaml", "case.yml", "case.json")


@dataclass(frozen=True)
class CaseCheck:
    name: str
    status: str
    message: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "status": self.status,
            "message": self.message,
        }
        if self.detail:
            payload["detail"] = self.detail
        return payload


@dataclass(frozen=True)
class CaseValidation:
    path: str
    case_id: str | None
    maturity: str | None
    passed: bool
    checks: tuple[CaseCheck, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "case_id": self.case_id,
            "maturity": self.maturity,
            "passed": self.passed,
            "checks": [check.to_dict() for check in self.checks],
        }


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    title: str
    maturity: str
    origin: str = ""
    category: str = ""
    package_path: str | None = None
    source_clay: str | None = None
    source_aliases: tuple[str, ...] = ()
    evidence_tests: tuple[str, ...] = ()
    verify_commands: tuple[str, ...] = ()
    artifacts: tuple[str, ...] = ()
    gaps: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.case_id,
            "title": self.title,
            "maturity": self.maturity,
        }
        for key, value in (
            ("origin", self.origin),
            ("category", self.category),
            ("package_path", self.package_path),
            ("source_clay", self.source_clay),
        ):
            if value:
                payload[key] = value
        if self.source_aliases:
            payload["source_aliases"] = list(self.source_aliases)
        for key, value in (
            ("evidence_tests", self.evidence_tests),
            ("verify_commands", self.verify_commands),
            ("artifacts", self.artifacts),
            ("gaps", self.gaps),
        ):
            if value:
                payload[key] = list(value)
        return payload


@dataclass(frozen=True)
class CaseInventory:
    summary: dict[str, Any]
    records: tuple[CaseRecord, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "clay.case_inventory/v1",
            "summary": self.summary,
            "records": [record.to_dict() for record in self.records],
        }


@dataclass(frozen=True)
class MaturityAssessment:
    """Evidence-computed maturity for one case package.

    ``computed`` is the highest level whose prerequisite evidence is
    actually present on disk. ``declared`` is the hand-written
    ``case.yaml: maturity`` value (or ``None`` when absent / unreadable).
    ``evidence`` records, per level, whether its evidence was found, so a
    consumer can show *why* a level was (not) reached.
    """

    case_dir: str
    case_id: str | None
    declared: str | None
    computed: str
    declared_valid: bool
    drift: bool
    overclaimed: bool
    evidence: dict[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_dir": self.case_dir,
            "case_id": self.case_id,
            "declared": self.declared,
            "computed": self.computed,
            "declared_valid": self.declared_valid,
            "drift": self.drift,
            "overclaimed": self.overclaimed,
            "evidence": self.evidence,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class MaturityAudit:
    """Repo-wide computed-vs-declared maturity audit."""

    summary: dict[str, Any]
    packages: tuple[MaturityAssessment, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "clay.case_maturity_audit/v1",
            "summary": self.summary,
            "packages": [pkg.to_dict() for pkg in self.packages],
        }


@dataclass(frozen=True)
class CaseCommand:
    name: str
    argv: tuple[str, ...]
    cwd: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "argv": list(self.argv),
            "cwd": self.cwd,
        }


def load_case_manifest(case_dir: str | Path) -> dict[str, Any]:
    """Load a case package manifest from a directory."""
    manifest_path = _find_manifest(Path(case_dir))
    if manifest_path is None:
        raise FileNotFoundError(
            f"no case manifest found in {case_dir}; expected one of "
            f"{', '.join(_MANIFEST_NAMES)}"
        )
    data = _load_structured_file(manifest_path)
    if not isinstance(data, dict):
        raise ValueError(f"{manifest_path} must contain a mapping")
    return data


def validate_case_package(case_dir: str | Path) -> CaseValidation:
    """Validate a standard case package without running expensive tools."""
    case_path = Path(case_dir)
    checks: list[CaseCheck] = []
    manifest: dict[str, Any] | None = None

    manifest_path = _find_manifest(case_path)
    if manifest_path is None:
        checks.append(CaseCheck(
            "manifest.exists",
            "fail",
            f"missing manifest; expected one of {', '.join(_MANIFEST_NAMES)}",
        ))
        return _validation(case_path, None, None, checks)

    try:
        loaded = _load_structured_file(manifest_path)
    except Exception as exc:  # pragma: no cover - exact parser text varies
        checks.append(CaseCheck(
            "manifest.load",
            "fail",
            f"could not load {manifest_path.name}: {exc}",
        ))
        return _validation(case_path, None, None, checks)

    if not isinstance(loaded, dict):
        checks.append(CaseCheck(
            "manifest.type",
            "fail",
            "case manifest must be a mapping",
        ))
        return _validation(case_path, None, None, checks)
    manifest = loaded

    case_id = _optional_str(manifest.get("id"))
    maturity = _optional_str(manifest.get("maturity"))
    checks.append(CaseCheck(
        "manifest.load",
        "pass",
        f"loaded {manifest_path.name}",
    ))

    missing = _missing_required_fields(manifest)
    if missing:
        checks.append(CaseCheck(
            "manifest.required_fields",
            "fail",
            "missing required field(s): " + ", ".join(missing),
        ))
    else:
        checks.append(CaseCheck(
            "manifest.required_fields",
            "pass",
            "required manifest fields are present",
        ))

    if manifest.get("schema") != CASE_SCHEMA:
        checks.append(CaseCheck(
            "manifest.schema",
            "fail",
            f"schema must be {CASE_SCHEMA!r}",
        ))
    else:
        checks.append(CaseCheck(
            "manifest.schema",
            "pass",
            f"schema is {CASE_SCHEMA}",
        ))

    if maturity not in MATURITY_RANK:
        checks.append(CaseCheck(
            "manifest.maturity",
            "fail",
            f"maturity must be one of {', '.join(MATURITIES)}",
        ))
        return _validation(case_path, case_id, maturity, checks)
    checks.append(CaseCheck(
        "manifest.maturity",
        "pass",
        f"maturity is {maturity}",
    ))

    rank = MATURITY_RANK[maturity]
    _check_declared_package_paths(manifest, case_path, checks)

    source = manifest.get("source") if isinstance(manifest.get("source"), dict) else {}
    clay_rel = _optional_str(source.get("clay")) if source else None
    clay_file = (case_path / clay_rel).resolve() if clay_rel else None
    cf = None

    if rank >= MATURITY_RANK["L1_PARSE"]:
        if not clay_rel:
            checks.append(CaseCheck(
                "source.clay",
                "fail",
                "source.clay is required for L1_PARSE and above",
            ))
        elif clay_file is not None and not _is_within(case_path.resolve(), clay_file):
            checks.append(CaseCheck(
                "source.clay",
                "fail",
                f"source.clay must stay inside the case package: {clay_rel}",
            ))
        elif clay_file is None or not clay_file.exists():
            checks.append(CaseCheck(
                "source.clay",
                "fail",
                f"source clay file does not exist: {clay_rel}",
            ))
        else:
            checks.append(CaseCheck(
                "source.clay",
                "pass",
                f"source exists: {clay_rel}",
            ))
            try:
                from .parser import parse_clay_file

                cf = parse_clay_file(clay_file)
            except Exception as exc:
                checks.append(CaseCheck(
                    "source.parse",
                    "fail",
                    f"parse failed: {exc}",
                ))
            else:
                checks.append(CaseCheck(
                    "source.parse",
                    "pass",
                    (
                        f"parsed {len(cf.cell_defs)} cell(s) and "
                        f"{len(cf.verify_jobs)} verify job(s)"
                    ),
                ))
                top_cell = _optional_str(source.get("top_cell"))
                if top_cell:
                    _check_top_cell(top_cell, cf, checks)

    if rank >= MATURITY_RANK["L2_COMPILE"] and cf is not None:
        _check_compile(cf, source, checks)

    if rank >= MATURITY_RANK["L3_SIM"]:
        _check_has_sim_evidence(manifest, checks)

    if rank >= MATURITY_RANK["L4_LAYOUT_INTENT"]:
        _check_has_layout_intent_evidence(manifest, cf, checks)

    if rank >= MATURITY_RANK["L5_SIGNOFF"]:
        _check_has_signoff_evidence(manifest, checks)

    return _validation(case_path, case_id, maturity, checks)


def assess_maturity(
    case_dir: str | Path,
    flow_result_dir: str | Path | None = None,
    *,
    repo_root: str | Path | None = None,
) -> str:
    """Compute the case package's maturity from REAL on-disk evidence.

    Returns the highest level whose prerequisite evidence is verifiable,
    never trusting the hand-written ``case.yaml: maturity`` value. The
    promotion ladder is monotone: a level is only reached when every level
    below it is also satisfied (a signed token without a parsing source is
    not L5 — it stays at the highest contiguous level).

    Evidence per level:

    * ``L0_SPEC``         — a manifest exists and loads.
    * ``L1_PARSE``        — ``source.clay`` parses via ``parse_clay_file``.
    * ``L2_COMPILE``      — the (top) cell compiles via ``compile_cell_def``.
    * ``L3_SIM``          — a real sim json with ``"measured": true`` exists
      (resolved from ``metadata.source_spice`` dir, a sibling
      ``presim_results.json``/post results, or ``flow_result_dir``).
    * ``L4_LAYOUT_INTENT``— a ``layout_intent``/``annotate`` block is present
      in the parsed ``.clay`` AND a GDS layout artifact exists.
    * ``L5_SIGNOFF``      — a REAL signed signoff token (``signed: true`` +
      ``trace_digest``) with its report/record artifact present on disk, the
      same signed proof ``validate_case_package`` requires. A DRC-clean
      ``.lyrdb`` and post-layout results are recorded as supporting evidence
      but do not, on their own, constitute signoff.
    """
    assessment = assess_maturity_detail(
        case_dir, flow_result_dir, repo_root=repo_root
    )
    return assessment.computed


def assess_maturity_detail(
    case_dir: str | Path,
    flow_result_dir: str | Path | None = None,
    *,
    repo_root: str | Path | None = None,
) -> MaturityAssessment:
    """Like :func:`assess_maturity` but returns the full evidence record."""
    case_path = Path(case_dir).resolve()
    root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else _infer_repo_root(case_path)
    )
    flow_dir = Path(flow_result_dir).resolve() if flow_result_dir else None

    evidence: dict[str, Any] = {}
    notes: list[str] = []

    manifest: dict[str, Any] | None = None
    declared: str | None = None
    case_id: str | None = None
    manifest_path = _find_manifest(case_path)
    if manifest_path is not None:
        try:
            loaded = _load_structured_file(manifest_path)
        except Exception as exc:  # pragma: no cover - parser text varies
            notes.append(f"manifest load failed: {exc}")
            loaded = None
        if isinstance(loaded, dict):
            manifest = loaded
            declared = _optional_str(manifest.get("maturity"))
            case_id = _optional_str(manifest.get("id"))

    # L0: a manifest is present + loads.
    has_manifest = manifest is not None
    evidence["L0_SPEC"] = {"manifest": has_manifest}

    source = (
        manifest.get("source")
        if manifest and isinstance(manifest.get("source"), dict)
        else {}
    )
    metadata = (
        manifest.get("metadata")
        if manifest and isinstance(manifest.get("metadata"), dict)
        else {}
    )

    # L1: source.clay parses.
    cf = None
    clay_rel = _optional_str(source.get("clay")) if source else None
    clay_file = (case_path / clay_rel).resolve() if clay_rel else None
    parsed_ok = False
    if clay_file is not None and clay_file.exists():
        try:
            from .parser import parse_clay_file

            cf = parse_clay_file(clay_file)
            parsed_ok = True
        except Exception as exc:
            notes.append(f"parse failed: {exc}")
    evidence["L1_PARSE"] = {
        "clay": clay_rel,
        "exists": bool(clay_file and clay_file.exists()),
        "parsed": parsed_ok,
    }

    # L2: top cell compiles. The compile path is style-agnostic: a
    # stage-template / primitive / composition cell compiles directly,
    # and a functional-group intent OVERLAY (`design.clay` whose
    # topology is role blocks listing device names, with the sized
    # devices in the bound `metadata.source_spice` `.cir`) compiles by
    # lowering that source netlist and validating every overlay device
    # reference resolves. We resolve `source_spice` here and hand it to
    # the compiler so the overlay path does not have to rediscover it.
    compiled_ok = False
    if parsed_ok and cf is not None:
        compiled_ok = _try_compile(cf, source, metadata, root)
    evidence["L2_COMPILE"] = {"compiled": compiled_ok}

    # L3: real measurement evidence — either a measured ngspice sim json
    # (`"measured": true`) or a real calibrated PEX/parasitics extraction
    # (Magic ext2spice, not a geometric estimate). Both are "real" data the
    # design was characterized against; a geometric guess does NOT qualify.
    sim_dirs = _sim_evidence_dirs(case_path, metadata, flow_dir, root)
    sim_json = _find_measured_sim(sim_dirs)
    pex_json = _find_real_extraction(sim_dirs) if sim_json is None else None
    sim_evidence = sim_json or pex_json
    evidence["L3_SIM"] = {
        "measured_json": _relative_display(str(sim_json), root) if sim_json else None,
        "extraction_json": _relative_display(str(pex_json), root) if pex_json else None,
        "searched": [_relative_display(str(d), root) for d in sim_dirs],
    }

    # L4: source layout intent block AND a GDS artifact.
    has_source_intent = _has_source_layout_intent(cf)
    gds = _find_gds_artifact(case_path, manifest, sim_dirs)
    evidence["L4_LAYOUT_INTENT"] = {
        "source_intent": has_source_intent,
        "gds": _relative_display(str(gds), root) if gds else None,
    }

    # L5: a REAL signed signoff token is required. L5_SIGNOFF means "signed
    # off" — the same signed proof artifact `validate_case_package`'s
    # `_has_required_signoff_token` gate demands (a signoff record/token
    # carrying `signed: true` + `trace_digest`, with its report/record on
    # disk). DRC-clean lyrdb + measured post results are recorded as
    # supporting evidence but, on their own, do NOT constitute signoff: an
    # unsigned design that merely passed DRC and post-layout sim has not been
    # signed off. This keeps the evidence-computed view aligned with the
    # stricter declared-check gate so the two maturity models cannot diverge.
    drc_clean = _find_drc_clean_lyrdb(sim_dirs)
    post_json = _find_post_results(sim_dirs)
    signed_token = _has_signed_token_on_disk(manifest, case_path) if manifest else False
    signoff_ok = bool(signed_token)
    evidence["L5_SIGNOFF"] = {
        "drc_clean_lyrdb": _relative_display(str(drc_clean), root) if drc_clean else None,
        "post_results": _relative_display(str(post_json), root) if post_json else None,
        "signed_token": signed_token,
    }

    # Monotone promotion: stop at the first level whose evidence is missing.
    reached = {
        "L0_SPEC": has_manifest,
        "L1_PARSE": parsed_ok,
        "L2_COMPILE": compiled_ok,
        "L3_SIM": sim_evidence is not None,
        "L4_LAYOUT_INTENT": has_source_intent and gds is not None,
        "L5_SIGNOFF": signoff_ok,
    }
    computed = "L0_SPEC"
    for level in MATURITIES:
        if reached.get(level):
            computed = level
        else:
            break

    declared_valid = declared in MATURITY_RANK
    if declared is not None and not declared_valid:
        notes.append(
            f"declared maturity {declared!r} is not one of "
            f"{', '.join(MATURITIES)}"
        )

    declared_rank = MATURITY_RANK.get(declared, -1) if declared else -1
    computed_rank = MATURITY_RANK[computed]
    drift = declared is not None and declared != computed
    overclaimed = declared_valid and declared_rank > computed_rank

    return MaturityAssessment(
        case_dir=_relative_display(str(case_path), root) or str(case_path),
        case_id=case_id,
        declared=declared,
        computed=computed,
        declared_valid=declared_valid,
        drift=bool(drift),
        overclaimed=bool(overclaimed),
        evidence=evidence,
        notes=tuple(notes),
    )


def audit_maturity(
    root: str | Path = ".",
    *,
    cases_root: str | Path = "cases",
) -> MaturityAudit:
    """Audit every case package: computed vs declared maturity + drift.

    Surfaces invalid declared values (e.g. the ``L2_LAYOUT``-style typo)
    via ``declared_valid=False`` and flags both drift (declared != computed)
    and overclaim (declared rank > computed rank).
    """
    root_path = Path(root).resolve()
    cases_dir = _resolve_under(root_path, cases_root)

    assessments: list[MaturityAssessment] = []
    if cases_dir.exists():
        manifest_dirs: set[Path] = set()
        for name in _MANIFEST_NAMES:
            for manifest_path in cases_dir.rglob(name):
                manifest_dirs.add(manifest_path.parent)
        for case_path in sorted(manifest_dirs):
            assessments.append(
                assess_maturity_detail(case_path, repo_root=root_path)
            )

    invalid = [a for a in assessments if a.declared is not None and not a.declared_valid]
    drifted = [a for a in assessments if a.drift]
    overclaimed = [a for a in assessments if a.overclaimed]
    summary = {
        "total": len(assessments),
        "invalid_declared": len(invalid),
        "drift": len(drifted),
        "overclaimed": len(overclaimed),
        "invalid_ids": [a.case_id or a.case_dir for a in invalid],
        "overclaimed_ids": [a.case_id or a.case_dir for a in overclaimed],
    }
    return MaturityAudit(summary=summary, packages=tuple(assessments))


def _infer_repo_root(case_path: Path) -> Path:
    """Walk up from ``case_path`` to the repo root (the dir with ``circuits``).

    Falls back to the parent of the top-level ``cases`` directory, then to
    ``case_path`` itself.
    """
    for parent in [case_path, *case_path.parents]:
        if (parent / "circuits").is_dir() and (parent / "clay").is_dir():
            return parent
    # Fallback: the directory containing the top-level `cases` segment.
    parts = case_path.parts
    if "cases" in parts:
        idx = parts.index("cases")
        return Path(*parts[:idx]) if idx else case_path
    return case_path


def _try_compile(
    cf: Any,
    source: dict[str, Any],
    metadata: dict[str, Any] | None = None,
    root: Path | None = None,
) -> bool:
    """L2_COMPILE evidence: does the (top) cell compile to a real
    device netlist?

    Style-template / primitive / composition cells compile from the
    `.clay` alone. A functional-group intent overlay compiles by
    lowering the bound source netlist (`metadata.source_spice`) and
    validating that every `functional_group` device reference resolves
    against it — so the W/L stay in the `.cir`, not duplicated in the
    `.clay`. We resolve `source_spice` (when present) and pass it to
    `compile_design`; the overlay path also auto-discovers it from the
    case manifest when no explicit path is supplied.
    """
    try:
        from .compiler import compile_design
    except Exception:  # pragma: no cover - import guard
        return False
    top_cell = _optional_str(source.get("top_cell")) if source else None
    cell_defs = list(getattr(cf, "cell_defs", []) or [])
    if top_cell:
        filtered = [c for c in cell_defs if c.local_name == top_cell]
        cell_defs = filtered or cell_defs
    if not cell_defs:
        return False

    source_cir: Path | None = None
    source_spice = _optional_str(metadata.get("source_spice")) if metadata else None
    if source_spice:
        spice_path = Path(source_spice)
        if not spice_path.is_absolute() and root is not None:
            spice_path = root / spice_path
        if spice_path.exists():
            source_cir = spice_path.resolve()

    try:
        compile_design(cell_defs[0], source_cir=source_cir)
    except Exception:
        return False
    return True


def _sim_evidence_dirs(
    case_path: Path,
    metadata: dict[str, Any],
    flow_dir: Path | None,
    root: Path,
) -> list[Path]:
    """Directories that may hold real sim / signoff evidence, in priority order."""
    dirs: list[Path] = []

    def add(candidate: Path | None) -> None:
        if candidate is None:
            return
        resolved = candidate.resolve()
        if resolved.is_dir() and resolved not in dirs:
            dirs.append(resolved)

    if flow_dir is not None:
        add(flow_dir)
    # In-package evidence dirs.
    add(case_path)
    add(case_path / "final")
    # The circuit dir behind metadata.source_spice (repo-relative).
    source_spice = _optional_str(metadata.get("source_spice")) if metadata else None
    if source_spice:
        spice_path = Path(source_spice)
        if not spice_path.is_absolute():
            spice_path = root / spice_path
        add(spice_path.parent)
    # layout_guide also points into the circuit dir.
    layout_guide = _optional_str(metadata.get("layout_guide")) if metadata else None
    if layout_guide:
        guide_path = Path(layout_guide)
        if not guide_path.is_absolute():
            guide_path = root / guide_path
        add(guide_path.parent)
    return dirs


def _find_measured_sim(dirs: list[Path]) -> Path | None:
    """Find a sim-results json carrying ``"measured": true`` evidence."""
    candidate_names = (
        "presim_results.json",
        "post_results.json",
        "postlayout_sim.json",
    )
    for directory in dirs:
        for name in candidate_names:
            path = directory / name
            if path.exists() and _json_is_measured(path):
                return path
        # Any *results*.json in the dir as a fallback.
        for path in sorted(directory.glob("*results*.json")):
            if _json_is_measured(path):
                return path
    return None


def _find_real_extraction(dirs: list[Path]) -> Path | None:
    """Find a real calibrated PEX/parasitics extraction artifact.

    Conservative: only a parasitics json that declares a calibrated 2.5D
    extraction (Magic ext2spice) AND carries non-empty extracted cap data
    qualifies. A geometric estimate is explicitly rejected.
    """
    for directory in dirs:
        for name in ("parasitics.json", "post_results.json", "postlayout_sim.json"):
            path = directory / name
            if path.exists() and _json_is_real_extraction(path):
                return path
        # Magic calibrated extractions are commonly named after the cell, e.g.
        # `comparator.parasitics.json`; accept any `*.parasitics.json` that
        # passes the calibrated-extraction gate (geometric estimates rejected).
        for path in sorted(directory.glob("*.parasitics.json")):
            if path.exists() and _json_is_real_extraction(path):
                return path
    return None


def _json_is_real_extraction(path: Path) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(data, dict):
        return False
    # Post-layout results with a real pre/deltas characterization.
    if isinstance(data.get("pre"), dict) and isinstance(data.get("deltas"), dict):
        return True
    if data.get("calibrated_2p5d") is True:
        caps = data.get("net_total_cap_aF")
        if isinstance(caps, dict) and caps:
            return True
    # Magic-sourced extraction with populated cap data.
    source = str(data.get("source", "")).lower()
    if "magic" in source or "ext2spice" in source:
        caps = data.get("net_total_cap_aF")
        if isinstance(caps, dict) and caps:
            return True
    return False


def _json_is_measured(path: Path) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(data, dict):
        return False
    if data.get("measured") is True:
        return True
    # Post-layout results embed a `pre` block with real numbers; treat the
    # presence of post KPI keys + a pre/deltas block as measured evidence.
    if isinstance(data.get("pre"), dict) and isinstance(data.get("deltas"), dict):
        return True
    # A named simulator/tool plus a populated measurement array (e.g. a real
    # ngspice dc/ac sweep) is genuine sim evidence even without a `measured`
    # flag. An empty array does NOT qualify (no real data captured).
    tool = data.get("simulator") or data.get("tool") or data.get("ngspice_binary")
    if tool:
        for key in (
            "dc_transfer",
            "results",
            "measurements",
            "ac_sweep",
            "sweep",
            # A post-layout sim json may store its swept rows under `points`
            # (e.g. a DC compliance sweep): a populated list is real sim data.
            "points",
        ):
            value = data.get(key)
            if isinstance(value, list) and value:
                return True
            if isinstance(value, dict) and value:
                return True
        # A populated operating-point / metrics block also counts.
        for key in ("operating_point", "metrics"):
            value = data.get(key)
            if isinstance(value, dict) and value:
                return True
    return False


def _has_source_layout_intent(cf: Any) -> bool:
    if cf is None:
        return False
    for cell in getattr(cf, "cell_defs", []) or []:
        if getattr(cell, "layout_intents", None):
            return True
        if cell.__dict__.get("_layout_annotations"):
            return True
    return False


def _find_gds_artifact(
    case_path: Path,
    manifest: dict[str, Any] | None,
    sim_dirs: list[Path],
) -> Path | None:
    # Manifest-declared artifacts first (in-package, validated path).
    if manifest is not None:
        for raw in _flatten_artifact_strings(manifest.get("artifacts")):
            if raw.lower().endswith(".gds") and not _is_url(raw):
                candidate = (case_path / raw).resolve()
                if candidate.exists():
                    return candidate
    # Then any GDS in the candidate evidence dirs.
    for directory in sim_dirs:
        for path in sorted(directory.glob("*.gds")):
            if path.exists():
                return path
    return None


def _find_drc_clean_lyrdb(dirs: list[Path]) -> Path | None:
    """Return a KLayout DRC report (.lyrdb) with zero violations, if any."""
    for directory in dirs:
        for path in sorted(directory.glob("*.lyrdb")):
            if _lyrdb_is_clean(path):
                return path
    return None


def _lyrdb_is_clean(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    if "<report-database" not in text:
        return False
    return "<item>" not in text


def _find_post_results(dirs: list[Path]) -> Path | None:
    for directory in dirs:
        for name in ("post_results.json", "postlayout_sim.json"):
            path = directory / name
            if path.exists() and _json_is_measured(path):
                return path
    return None


def _has_signed_token_on_disk(manifest: dict[str, Any], case_path: Path) -> bool:
    """A real signoff token: signed + digest AND the report artifact on disk."""
    signoff = manifest.get("signoff") if isinstance(manifest.get("signoff"), dict) else {}
    if not signoff:
        return False
    if not _has_required_signoff_token(signoff) or not _has_explicit_signoff_verdict(signoff):
        return False
    token = signoff.get("token")
    signed = False
    if isinstance(token, dict):
        digest = _optional_str(token.get("trace_digest")) or _optional_str(token.get("digest"))
        signed = token.get("signed") is True and digest is not None
    elif isinstance(token, str):
        signed = bool(token.strip())
    if not signed:
        return False
    # Require at least one referenced report/record artifact to exist on disk.
    for key in ("report", "record"):
        rel = _optional_str(signoff.get(key))
        if rel and not _is_url(rel):
            if (case_path / rel).resolve().exists():
                return True
    return False


def discover_case_packages(root: str | Path) -> list[CaseRecord]:
    """Discover manifest-backed case packages under ``root``."""
    root_path = Path(root)
    if not root_path.exists():
        return []
    manifest_paths: list[Path] = []
    for name in _MANIFEST_NAMES:
        manifest_paths.extend(root_path.rglob(name))

    records: list[CaseRecord] = []
    seen: set[Path] = set()
    for manifest_path in sorted(manifest_paths):
        if manifest_path in seen:
            continue
        seen.add(manifest_path)
        try:
            data = _load_structured_file(manifest_path)
        except Exception:
            continue
        if isinstance(data, dict) and data.get("schema") == CASE_SCHEMA:
            records.append(_record_from_manifest(data, manifest_path.parent))
    return records


def discover_example_records(examples_dir: str | Path) -> list[CaseRecord]:
    """Return conservative virtual records for committed ``examples/*.clay``."""
    base = Path(examples_dir)
    if not base.exists():
        return []
    records: list[CaseRecord] = []
    for clay_path in sorted(base.rglob("*.clay")):
        rel = clay_path.relative_to(base)
        slug = _slugify(rel.with_suffix("").as_posix())
        records.append(CaseRecord(
            case_id=f"example-{slug}",
            title=rel.with_suffix("").as_posix().replace("/", " "),
            origin="legacy_example",
            category=_infer_example_category(rel),
            maturity="L1_PARSE",
            source_clay=str(clay_path),
            gaps=(
                "virtual record only; migrate into a self-contained case "
                "package before raising maturity",
            ),
        ))
    return records


def load_seed_catalog(path: str | Path) -> list[CaseRecord]:
    """Load a curated catalog of mature legacy cases."""
    catalog_path = Path(path)
    data = _load_structured_file(catalog_path)
    if not isinstance(data, dict):
        raise ValueError(f"{catalog_path} must contain a mapping")
    if data.get("schema") != CATALOG_SCHEMA:
        raise ValueError(
            f"{catalog_path} schema must be {CATALOG_SCHEMA!r}"
        )
    records = data.get("records")
    if not isinstance(records, list):
        raise ValueError(f"{catalog_path} must contain a records list")
    return [
        _record_from_catalog_item(item)
        for item in records
        if isinstance(item, dict)
    ]


def build_case_inventory(
    repo_root: str | Path = ".",
    *,
    cases_root: str | Path = "cases",
    catalog_path: str | Path = "cases/catalog.yaml",
    examples_root: str | Path = "examples",
) -> CaseInventory:
    """Build the de-duplicated inventory of all known Clay cases.

    Curated catalog records overlay their corresponding virtual example
    records, so a mature legacy example appears once at its evidence-backed
    maturity rather than twice as both L1 and L3/L4/L5.
    """
    root = Path(repo_root).resolve()
    cases_dir = _resolve_under(root, cases_root)
    catalog_file = _resolve_under(root, catalog_path)
    examples_dir = _resolve_under(root, examples_root)

    package_records = [
        _relativize_record(record, root)
        for record in discover_case_packages(cases_dir)
    ]
    package_ids = {record.case_id for record in package_records}
    all_catalog_records: list[CaseRecord] = []
    if catalog_file.exists():
        all_catalog_records = [
            _relativize_record(record, root)
            for record in load_seed_catalog(catalog_file)
        ]
    catalog_records = [
        record for record in all_catalog_records
        if record.case_id not in package_ids
    ]

    package_source_keys = {
        key
        for record in package_records
        for key in _record_source_keys(record, root)
        if key is not None
    }
    catalog_source_keys = {
        _source_key(record.source_clay, root)
        for record in catalog_records
        if record.source_clay
    }
    suppressed_source_keys = package_source_keys | catalog_source_keys
    example_records: list[CaseRecord] = []
    if examples_dir.exists():
        for record in discover_example_records(examples_dir):
            rel_record = _relativize_record(record, root)
            if _source_key(rel_record.source_clay, root) not in suppressed_source_keys:
                example_records.append(rel_record)

    records = sorted(
        [*package_records, *catalog_records, *example_records],
        key=lambda record: (
            MATURITY_RANK.get(record.maturity, -1),
            record.origin,
            record.case_id,
        ),
    )
    return CaseInventory(
        summary=_inventory_summary(records),
        records=tuple(records),
    )


def collect_run_commands(case_dir: str | Path) -> list[CaseCommand]:
    manifest = load_case_manifest(case_dir)
    return _commands_from_manifest(manifest, Path(case_dir))


def run_case_package(
    case_dir: str | Path,
    *,
    execute: bool = False,
) -> dict[str, Any]:
    """Validate a case package and return or execute its declared commands."""
    validation = validate_case_package(case_dir)
    commands = collect_run_commands(case_dir) if validation.passed else []
    payload: dict[str, Any] = {
        "case_id": validation.case_id,
        "passed": validation.passed,
        "execute": execute,
        "commands": [command.to_dict() for command in commands],
        "validation": validation.to_dict(),
    }
    if not execute or not validation.passed:
        return payload

    results: list[dict[str, Any]] = []
    for command in commands:
        completed = subprocess.run(
            list(command.argv),
            cwd=command.cwd,
            capture_output=True,
            text=True,
            timeout=600,
        )
        results.append({
            "name": command.name,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        })
    payload["results"] = results
    payload["passed"] = all(item["returncode"] == 0 for item in results)
    return payload


def records_to_json(records: list[CaseRecord]) -> list[dict[str, Any]]:
    return [record.to_dict() for record in records]


def _resolve_under(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (root / candidate).resolve()


def _source_key(value: str | None, root: Path) -> str | None:
    if not value:
        return None
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        return candidate.resolve().relative_to(root).as_posix().lower()
    except ValueError:
        return candidate.resolve().as_posix().lower()


def _record_source_keys(record: CaseRecord, root: Path) -> tuple[str | None, ...]:
    return tuple(
        _source_key(value, root)
        for value in (record.source_clay, *record.source_aliases)
    )


def _relativize_record(record: CaseRecord, root: Path) -> CaseRecord:
    package_path = _relative_display(record.package_path, root)

    def path_for_inventory(value: str | None) -> str | None:
        if (
            record.origin == "self_contained_package"
            and value
            and package_path
            and not Path(value).is_absolute()
            and not _is_url(value)
        ):
            normalized = value.replace("\\", "/")
            return f"{package_path}/{normalized}"
        return _relative_display(value, root)

    return replace(
        record,
        package_path=package_path,
        source_clay=path_for_inventory(record.source_clay),
        source_aliases=tuple(
            _relative_display(path, root) or path
            for path in record.source_aliases
        ),
        artifacts=tuple(
            path_for_inventory(path) or path
            for path in record.artifacts
        ),
        evidence_tests=tuple(
            path_for_inventory(path) or path
            for path in record.evidence_tests
        ),
    )


def _relative_display(value: str | None, root: Path) -> str | None:
    if not value or _is_url(value):
        return value
    candidate = Path(value)
    if not candidate.is_absolute():
        return value.replace("\\", "/")
    try:
        return candidate.resolve().relative_to(root).as_posix()
    except ValueError:
        return candidate.resolve().as_posix()


def _inventory_summary(records: list[CaseRecord]) -> dict[str, Any]:
    maturity_counts = {maturity: 0 for maturity in MATURITIES}
    origin_counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    for record in records:
        maturity_counts[record.maturity] = maturity_counts.get(record.maturity, 0) + 1
        origin = record.origin or "unknown"
        origin_counts[origin] = origin_counts.get(origin, 0) + 1
        category = record.category or "uncategorized"
        category_counts[category] = category_counts.get(category, 0) + 1
    return {
        "total": len(records),
        "maturity": maturity_counts,
        "origins": dict(sorted(origin_counts.items())),
        "categories": dict(sorted(category_counts.items())),
        "self_contained": origin_counts.get("self_contained_package", 0),
        "curated": origin_counts.get("curated_catalog", 0),
        "legacy_unmigrated": origin_counts.get("legacy_example", 0),
    }


def _validation(
    case_path: Path,
    case_id: str | None,
    maturity: str | None,
    checks: list[CaseCheck],
) -> CaseValidation:
    passed = all(check.status != "fail" for check in checks)
    return CaseValidation(
        path=str(case_path),
        case_id=case_id,
        maturity=maturity,
        passed=passed,
        checks=tuple(checks),
    )


def _find_manifest(case_path: Path) -> Path | None:
    for name in _MANIFEST_NAMES:
        candidate = case_path / name
        if candidate.exists():
            return candidate
    return None


def _check_declared_package_paths(
    manifest: dict[str, Any],
    case_path: Path,
    checks: list[CaseCheck],
) -> None:
    root = case_path.resolve()
    grouped_paths: list[tuple[str, tuple[str, ...]]] = [
        ("artifact", _flatten_artifact_strings(manifest.get("artifacts"))),
    ]
    evidence = manifest.get("evidence") if isinstance(manifest.get("evidence"), dict) else {}
    if evidence:
        grouped_paths.append(("evidence test", _as_str_tuple(evidence.get("tests"))))
        grouped_paths.append(("evidence report", _as_str_tuple(evidence.get("reports"))))
    signoff = manifest.get("signoff") if isinstance(manifest.get("signoff"), dict) else {}
    if signoff:
        signoff_paths: list[str] = []
        for key in ("report", "record", "token", "drc", "lvs", "pex"):
            value = signoff.get(key)
            if isinstance(value, str):
                signoff_paths.append(value)
        grouped_paths.append(("signoff artifact", tuple(signoff_paths)))

    for label, paths in grouped_paths:
        for raw_path in paths:
            if not raw_path or _is_url(raw_path):
                continue
            candidate = (case_path / raw_path).resolve()
            if not _is_within(root, candidate):
                checks.append(CaseCheck(
                    "manifest.paths",
                    "fail",
                    f"{label} path must stay inside the case package: {raw_path}",
                ))
            elif not candidate.exists():
                checks.append(CaseCheck(
                    "manifest.paths",
                    "fail",
                    f"{label} path does not exist: {raw_path}",
                ))
            else:
                checks.append(CaseCheck(
                    "manifest.paths",
                    "pass",
                    f"{label} path exists: {raw_path}",
                ))


def _is_within(root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def _is_url(value: str) -> bool:
    return "://" in value


def _load_structured_file(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "YAML case manifests require PyYAML; use case.json or install "
            "PyYAML>=6.0"
        ) from exc
    return yaml.safe_load(text)


def _missing_required_fields(manifest: dict[str, Any]) -> list[str]:
    missing = [
        key
        for key in ("schema", "id", "title", "maturity")
        if not _optional_str(manifest.get(key))
    ]
    maturity = _optional_str(manifest.get("maturity"))
    if maturity and MATURITY_RANK.get(maturity, 0) >= MATURITY_RANK["L1_PARSE"]:
        source = manifest.get("source") if isinstance(manifest.get("source"), dict) else {}
        if not source or not _optional_str(source.get("clay")):
            missing.append("source.clay")
    return missing


def _optional_str(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _check_top_cell(top_cell: str, cf: Any, checks: list[CaseCheck]) -> None:
    cell_names = {cell.local_name for cell in getattr(cf, "cell_defs", [])}
    if top_cell in cell_names:
        checks.append(CaseCheck(
            "source.top_cell",
            "pass",
            f"top cell {top_cell!r} is defined in source",
        ))
    else:
        checks.append(CaseCheck(
            "source.top_cell",
            "fail",
            f"top cell {top_cell!r} not found in source",
        ))


def _check_compile(cf: Any, source: dict[str, Any], checks: list[CaseCheck]) -> None:
    from .compiler import compile_cell_def

    top_cell = _optional_str(source.get("top_cell"))
    cell_defs = list(getattr(cf, "cell_defs", []) or [])
    if top_cell:
        cell_defs = [cell for cell in cell_defs if cell.local_name == top_cell]
    if not cell_defs:
        checks.append(CaseCheck(
            "source.compile",
            "fail",
            "no local cell definition available to compile",
        ))
        return
    try:
        compiled = compile_cell_def(cell_defs[0])
    except Exception as exc:
        checks.append(CaseCheck(
            "source.compile",
            "fail",
            f"compile failed: {exc}",
        ))
    else:
        checks.append(CaseCheck(
            "source.compile",
            "pass",
            f"compiled {compiled.name}",
        ))


def _has_declared_criteria(manifest: dict[str, Any]) -> bool:
    for key in ("criteria", "metrics", "kpis"):
        value = manifest.get(key)
        if isinstance(value, dict) and value:
            return True
        if isinstance(value, list) and value:
            return True
    return False


def _has_explicit_signoff_verdict(signoff: dict[str, Any]) -> bool:
    if isinstance(signoff.get("passed"), bool):
        return True
    verdict = _optional_str(signoff.get("verdict")) or _optional_str(signoff.get("status"))
    if verdict is None:
        return False
    return verdict.lower() in {"pass", "passed", "fail", "failed"}


def _has_required_signoff_token(signoff: dict[str, Any]) -> bool:
    passed = signoff.get("passed") is True
    verdict = _optional_str(signoff.get("verdict")) or _optional_str(signoff.get("status"))
    if verdict is not None:
        passed = passed or verdict.lower() in {"pass", "passed"}
    if not passed:
        return True

    token = signoff.get("token")
    if isinstance(token, str):
        return bool(token.strip())
    if not isinstance(token, dict):
        return False
    if token.get("signed") is not True:
        return False
    digest = _optional_str(token.get("trace_digest")) or _optional_str(token.get("digest"))
    return digest is not None


def _check_has_sim_evidence(
    manifest: dict[str, Any],
    checks: list[CaseCheck],
) -> None:
    commands = _commands_from_manifest(manifest, Path("."))
    evidence = manifest.get("evidence") if isinstance(manifest.get("evidence"), dict) else {}
    has_tests = bool(evidence.get("tests")) if evidence else False
    if commands or has_tests:
        checks.append(CaseCheck(
            "evidence.sim",
            "pass",
            "simulation evidence or command is declared",
        ))
    else:
        checks.append(CaseCheck(
            "evidence.sim",
            "fail",
            "L3_SIM and above require verify.commands or evidence.tests",
        ))
    if _has_declared_criteria(manifest):
        checks.append(CaseCheck(
            "evidence.criteria",
            "pass",
            "explicit pass/fail criteria are declared",
        ))
    else:
        checks.append(CaseCheck(
            "evidence.criteria",
            "fail",
            "L3_SIM and above require explicit criteria or KPI thresholds",
        ))


def _check_has_layout_intent_evidence(
    manifest: dict[str, Any],
    cf: Any,
    checks: list[CaseCheck],
) -> None:
    has_source_intent = False
    if cf is not None:
        for cell in getattr(cf, "cell_defs", []) or []:
            if getattr(cell, "layout_intents", None):
                has_source_intent = True
            if cell.__dict__.get("_layout_annotations"):
                has_source_intent = True
    artifacts = _flatten_artifact_strings(manifest.get("artifacts"))
    has_artifact_intent = any(
        token in artifact.lower()
        for artifact in artifacts
        for token in ("layout", "gds", "pex", "intent")
    )
    if has_source_intent or has_artifact_intent:
        checks.append(CaseCheck(
            "evidence.layout_intent",
            "pass",
            "layout-intent evidence is declared or present in source",
        ))
    else:
        checks.append(CaseCheck(
            "evidence.layout_intent",
            "fail",
            "L4_LAYOUT_INTENT requires source intent or layout artifacts",
        ))


def _check_has_signoff_evidence(
    manifest: dict[str, Any],
    checks: list[CaseCheck],
) -> None:
    artifacts = _flatten_artifact_strings(manifest.get("artifacts"))
    evidence = manifest.get("evidence") if isinstance(manifest.get("evidence"), dict) else {}
    signoff = manifest.get("signoff") if isinstance(manifest.get("signoff"), dict) else {}
    has_signoff = bool(signoff) or any(
        token in artifact.lower()
        for artifact in artifacts
        for token in ("signoff", "drc", "lvs", "pex")
    )
    has_tests = any("signoff" in item.lower() for item in _as_str_tuple(evidence.get("tests")))
    if has_signoff or has_tests:
        checks.append(CaseCheck(
            "evidence.signoff",
            "pass",
            "signoff evidence is declared",
        ))
    else:
        checks.append(CaseCheck(
            "evidence.signoff",
            "fail",
            "L5_SIGNOFF requires signoff artifacts, signoff block, or signoff tests",
        ))
    if _has_explicit_signoff_verdict(signoff):
        checks.append(CaseCheck(
            "evidence.signoff_verdict",
            "pass",
            "explicit signoff verdict is declared",
        ))
    else:
        checks.append(CaseCheck(
            "evidence.signoff_verdict",
            "fail",
            "L5_SIGNOFF requires signoff.verdict or signoff.passed",
        ))
    if _has_required_signoff_token(signoff):
        checks.append(CaseCheck(
            "evidence.signoff_token",
            "pass",
            "required signoff token metadata is declared",
        ))
    else:
        checks.append(CaseCheck(
            "evidence.signoff_token",
            "fail",
            "passed L5_SIGNOFF requires signed token metadata",
        ))


def _record_from_manifest(data: dict[str, Any], package_path: Path) -> CaseRecord:
    evidence = data.get("evidence") if isinstance(data.get("evidence"), dict) else {}
    source = data.get("source") if isinstance(data.get("source"), dict) else {}
    return CaseRecord(
        case_id=str(data.get("id", package_path.name)),
        title=str(data.get("title", package_path.name)),
        origin="self_contained_package",
        category=str(data.get("category", "")),
        maturity=str(data.get("maturity", "L0_SPEC")),
        package_path=str(package_path),
        source_clay=_optional_str(source.get("clay")) if source else None,
        source_aliases=_source_aliases_from_manifest(source),
        evidence_tests=_as_str_tuple(evidence.get("tests")) if evidence else (),
        verify_commands=tuple(command.raw for command in _raw_commands(data)),
        artifacts=_flatten_artifact_strings(data.get("artifacts")),
        gaps=_as_str_tuple(data.get("gaps")),
    )


def _record_from_catalog_item(item: dict[str, Any]) -> CaseRecord:
    evidence = item.get("evidence") if isinstance(item.get("evidence"), dict) else {}
    source = item.get("source") if isinstance(item.get("source"), dict) else {}
    return CaseRecord(
        case_id=str(item.get("id", item.get("case_id", ""))),
        title=str(item.get("title", item.get("id", ""))),
        origin="curated_catalog",
        category=str(item.get("category", "")),
        maturity=str(item.get("maturity", "L0_SPEC")),
        package_path=_optional_str(item.get("package_path")),
        source_clay=_optional_str(source.get("clay")) if source else None,
        source_aliases=_as_str_tuple(source.get("aliases")) if source else (),
        evidence_tests=_as_str_tuple(evidence.get("tests")) if evidence else (),
        verify_commands=_as_str_tuple(item.get("verify_commands")),
        artifacts=_flatten_artifact_strings(item.get("artifacts")),
        gaps=_as_str_tuple(item.get("gaps")),
    )


@dataclass(frozen=True)
class _RawCommand:
    name: str
    raw: str
    argv: tuple[str, ...]
    cwd: str | None = None


def _commands_from_manifest(
    manifest: dict[str, Any],
    case_dir: Path,
) -> list[CaseCommand]:
    commands: list[CaseCommand] = []
    for raw in _raw_commands(manifest):
        cwd = str((case_dir / raw.cwd).resolve()) if raw.cwd else str(case_dir.resolve())
        commands.append(CaseCommand(raw.name, raw.argv, cwd))
    return commands


def _raw_commands(manifest: dict[str, Any]) -> list[_RawCommand]:
    verify = manifest.get("verify")
    if not isinstance(verify, dict):
        return []
    commands = verify.get("commands")
    if not commands:
        return []
    out: list[_RawCommand] = []
    if isinstance(commands, dict):
        for name, value in commands.items():
            out.extend(_coerce_command_group(str(name), value))
    elif isinstance(commands, list):
        for index, value in enumerate(commands):
            out.extend(_coerce_command_group(f"cmd{index + 1}", value))
    elif isinstance(commands, str):
        out.extend(_coerce_command_group("default", commands))
    return out


def _coerce_command_group(name: str, value: Any) -> list[_RawCommand]:
    if isinstance(value, str):
        return [_command_from_value(name, value)]
    if isinstance(value, list):
        commands = []
        for index, item in enumerate(value):
            item_name = name if len(value) == 1 else f"{name}[{index}]"
            commands.append(_command_from_value(item_name, item))
        return commands
    if isinstance(value, dict):
        return [_command_from_value(name, value)]
    return []


def _command_from_value(name: str, value: Any) -> _RawCommand:
    cwd: str | None = None
    if isinstance(value, list):
        argv = tuple(str(part) for part in value)
        raw = " ".join(argv)
    elif isinstance(value, dict):
        cwd = _optional_str(value.get("cwd"))
        if isinstance(value.get("argv"), list):
            argv = tuple(str(part) for part in value["argv"])
            raw = " ".join(argv)
        else:
            raw = str(value.get("cmd", ""))
            argv = tuple(shlex.split(raw, posix=True))
    else:
        raw = str(value)
        argv = tuple(shlex.split(raw, posix=True))
    return _RawCommand(name=name, raw=raw, argv=argv, cwd=cwd)


def _flatten_artifact_strings(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list):
        items: list[str] = []
        for item in value:
            items.extend(_flatten_artifact_strings(item))
        return tuple(items)
    if isinstance(value, dict):
        items = []
        for item in value.values():
            items.extend(_flatten_artifact_strings(item))
        return tuple(items)
    return ()


def _as_str_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list):
        return tuple(str(item) for item in value)
    return ()


def _source_aliases_from_manifest(source: dict[str, Any]) -> tuple[str, ...]:
    aliases: list[str] = []
    upstream = _optional_str(source.get("upstream_clay"))
    if upstream:
        aliases.append(upstream)
    aliases.extend(_as_str_tuple(source.get("aliases")))
    return tuple(aliases)


def _slugify(value: str) -> str:
    value = value.replace("\\", "/")
    value = re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-")
    return value.lower()


def _infer_example_category(rel_path: Path) -> str:
    parts = [part.lower() for part in rel_path.parts]
    stem = rel_path.stem.lower()
    if parts and parts[0] == "imports":
        return "fixture_import"
    if "pll_subcells" in parts:
        return "pll_subcell"
    if "pll_ring_subcells" in parts:
        return "pll_ring_subcell"
    if parts and parts[0].startswith("gf180mcu"):
        return "gf180_cohort"
    if any(token in stem for token in ("layout_aware", "corouted", "pdk_dims")):
        return "layout_intent_candidate"
    if stem.startswith("analoggym_"):
        return "analoggym"
    if "benchmark" in stem:
        return "benchmark"
    if "functional" in stem:
        return "functional"
    if stem.endswith("_demo") or stem in {"showcase", "two_stage_miller"}:
        return "demo"
    return "legacy_example"
