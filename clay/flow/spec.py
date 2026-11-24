"""Frozen data contracts for the Clay flow engine.

These dataclasses are the COMPILED PROJECTION of the source-level ``flow {}`` /
``quality {}`` Clay blocks. They are the single interface the language layer
(WS-D: ``compile_flow_spec`` / ``discover_registry``) and the engine
(WS-A: ``clay/flow/step.py`` + ``engine.py``) compose against. Do not add
``if circuit == "..."`` branches anywhere: circuit specifics are DATA on a
:class:`CircuitSpec`, discovered from the case packages.

Public signatures here are FROZEN per docs/superpowers/2026-05-29-clay-flow-engine-contract.md.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)
from clay.flow.execution_session import ExecutionSession

if TYPE_CHECKING:  # avoid a hard import cycle with the parser at module load
    from clay.ir import CellDef, ParsedPolicyDecl, SpecField, VerifyJob

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Canonical universal analog flow, in dependency order. A design may select a
#: subset / reorder via its ``flow { steps (...) }`` block. These names are the
#: identities the engine's built-in steps register under.
DEFAULT_STEPS: tuple[str, ...] = (
    "compile",
    "sizing",
    "layout",
    "drc",
    "lvs",
    "pex",
    "post",
    "intent",
    "sign",
)


# --------------------------------------------------------------------------- #
# Acceptance contract (the "quality" block, compiled)                         #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class KpiSpec:
    """One measurable acceptance criterion, lifted from ``@kpi(...)``.

    ``key`` is the field name read from the circuit's ``post_results.json``
    (e.g. ``"dc_gain_db_post"``). ``bound`` is the pass condition, never a
    per-circuit code branch:
        ("ge", x)   -> value >= x
        ("le", x)   -> value <= x
        ("near", (target, tol)) -> abs(value - target) <= tol
    """

    key: str
    label: str = ""
    unit: str = ""
    scale: float = 1.0  # display = raw * scale (e.g. 1e-6 for Hz->MHz)
    target: str = ""
    bound: Optional[tuple[str, Any]] = None

    def evaluate(self, raw: Any) -> str:
        """Return one honest KPI obligation state.

        Required, measured obligations are ``pass``/``fail``.  Missing or
        non-finite evidence is ``unmeasurable``; an unknown comparison is
        ``unsupported``; and a metric without a bound is explicitly
        ``advisory`` rather than silently passing.
        """
        if raw is None:
            return "unmeasurable"
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return "unmeasurable"
        if not math.isfinite(value):
            return "unmeasurable"
        if self.bound is None:
            return "advisory"
        op, ref = self.bound
        if op == "ge":
            return "pass" if value >= float(ref) else "fail"
        if op == "le":
            return "pass" if value <= float(ref) else "fail"
        if op == "near":
            target, tol = ref
            return "pass" if abs(value - float(target)) <= float(tol) else "fail"
        return "unsupported"


@dataclass(frozen=True)
class ConsistencySpec:
    """Pre/post relative-consistency criterion, lifted from ``@consistency(...)``.

    RELATIVE only (never absolute deltas) per the project rule.
    """

    pre_key: str
    post_key: str
    rel_tol: float
    label: str = ""

    def evaluate(self, pre: Any, post: Any) -> str:
        try:
            p = float(pre)
            q = float(post)
        except (TypeError, ValueError):
            return "pending"
        if p == 0:
            return "pass" if abs(q) <= self.rel_tol else "warn"
        return "pass" if abs(q - p) / abs(p) <= self.rel_tol else "warn"


@dataclass(frozen=True)
class GateSpec:
    """A required hard gate lifted from ``@require(...)``.

    ``kind`` is one of: ``drc_clean``, ``lvs_match``, ``pex_extracted``,
    ``intent_signed``, ``post_consistent``. The engine's SignStep maps each to
    the corresponding step result; a failing required gate fails the verdict.
    """

    kind: str


@dataclass(frozen=True)
class OptimizationVariableSpec:
    """One free sizing degree of freedom lowered from ``sizing {}``."""

    name: str
    default: float
    low: float
    high: float
    bindings: tuple[str, ...] = ()
    step: Optional[float] = None
    binding_scales: tuple[tuple[str, float], ...] = ()

    @property
    def bounds(self) -> tuple[float, float]:
        return (self.low, self.high)


@dataclass(frozen=True)
class OptimizationObjectiveSpec:
    """Registry key and direction for the declared optimization objective."""

    name: str
    direction: str
    cl_f: float = 1.0


@dataclass(frozen=True)
class OptimizationSpec:
    """Compiled projection of optimization DATA carried by ``sizing {}``.

    This is deliberately not a language AST node.  The existing numeric
    sizing surface uses reserved ``opt_*`` keys for orchestration metadata and
    ``NAME``/``NAME_min``/``NAME_max`` triplets for free variables.
    """

    engine: str
    seed: int
    budget: int
    post_every: int
    normalized_domain: tuple[float, float]
    truth: str
    variables: tuple[OptimizationVariableSpec, ...]
    objective: OptimizationObjectiveSpec
    degradation_limits: tuple[tuple[str, float], ...] = ()


# --------------------------------------------------------------------------- #
# The compiled circuit spec (the "flow" block + identity + bindings)          #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CircuitSpec:
    """Compiled projection of a design's ``flow {}`` + ``quality {}`` blocks.

    Built by :func:`compile_flow_spec` from a parsed ``CellDef``; discovered
    across case packages by :func:`discover_registry`. The hand-written
    ``circuit_case_archive.CIRCUITS`` table is being retired in favour of this.
    """

    key: str
    design_id: str
    display_name: str
    cell_type: str
    top_cell: str
    pdk: str
    vdd: float

    case_dir: Path
    source_cir: Path
    layout_guide: Path
    presim_results: Path
    post_results: Path

    # layout generator binding (module:function), referenced by name (no import here)
    layout_module: str
    build_function: str
    emit_function: Optional[str]
    gds_cell_name: str
    fallback_gds: Path

    pex_spice: Optional[Path] = None
    drc_report: Optional[Path] = None
    presim_module: Optional[str] = None
    presim_function: str = "run_candidate"
    postsim_module: Optional[str] = None
    postsim_function: Optional[str] = None
    lvs_ref: Optional[str] = None

    corners: tuple[str, ...] = ("tt",)
    temps_c: tuple[float, ...] = (27.0,)
    steps: tuple[str, ...] = DEFAULT_STEPS
    kpis: tuple[KpiSpec, ...] = ()
    consistency: tuple[ConsistencySpec, ...] = ()
    gates: tuple[GateSpec, ...] = ()
    optimization: Optional[OptimizationSpec] = None

    # File-level layout policy selected by ``flow { policy = NAME; }``.
    # Keep the declaration on the compiled projection: a CircuitSpec may be
    # executed from a canonical source whose case-package design.clay is only
    # an import wrapper, so runtime path re-parsing cannot be the authority.
    policy_ref: Optional[str] = None
    policy_decl: Optional["ParsedPolicyDecl"] = None

    source_contract_note: str = ""
    artifacts_resolved: bool = True
    # Explicit native tool/PDK authority survives per-candidate case staging.
    native_runtime_manifest: Optional[Path] = None
    native_runtime_sha256: Optional[str] = None


# --------------------------------------------------------------------------- #
# Step protocol + result + context (the engine's plug-in surface)             #
# --------------------------------------------------------------------------- #
@dataclass
class StepResult:
    name: str
    status: str  # "pass" | "warn" | "fail" | "skipped"
    summary: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    blocking: bool = False  # a failing blocking step stops the flow + fails the verdict

    @property
    def ok(self) -> bool:
        return self.status in ("pass", "skipped", "warn")


@dataclass
class FlowContext:
    spec: CircuitSpec
    work_dir: Path
    recorder: Any = None  # clay.runtime.recorder.RunRecorder | None
    force: bool = False
    results: dict[str, StepResult] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    execution: Optional["ExecutionSession"] = None


@runtime_checkable
class FlowStep(Protocol):
    name: str

    def inputs(self, spec: CircuitSpec, ctx: FlowContext) -> list[Path]:
        """Files whose content-hash gates the incremental-skip decision."""
        ...

    def run(self, spec: CircuitSpec, ctx: FlowContext) -> StepResult:
        ...


# --------------------------------------------------------------------------- #
# Language -> spec compilation + registry discovery (WS-D fills these in).     #
# Stubs raise NotImplementedError so the frozen contract imports cleanly and   #
# the engine (WS-A) can be built and unit-tested against fixtures meanwhile.   #
# --------------------------------------------------------------------------- #
import importlib
import json

# Map the @kpi op surface token -> the frozen ``KpiSpec.bound`` op tag.
_OP_TO_BOUND: dict[str, str] = {">=": "ge", "<=": "le"}

# Default relative tolerance applied to a ``near`` KPI when the source
# block does not (cannot, per the frozen grammar) carry an explicit
# tolerance. RELATIVE-only per the project rule; the absolute tol is
# derived as ``frac * |target|`` at compile time.
_NEAR_REL_TOL: float = 0.10

# Units whose RAW value is in base SI but the human display divides /
# multiplies by a power of ten. ``scale`` = display / raw.
_UNIT_DISPLAY_SCALE: dict[str, float] = {
    "hz": 1e-6,   # raw Hz -> display MHz
    "a": 1e6,     # raw A  -> display uA
    # Regulation units: the raw value is the SI ratio (V/V, V/A); the display
    # unit is milli- (mV/V, mV/A), so display = raw * 1e3. This both preserves
    # the unit label AND keeps the displayed number consistent with it (a raw
    # 0.05 V/V regulation shows as 50 mV/V).
    "mv/v": 1e3,
    "mv/a": 1e3,
}

# Human-readable display unit for a raw unit, after applying scale.
_UNIT_DISPLAY_LABEL: dict[str, str] = {
    "hz": "MHz",
    "a": "uA",
    "mv/v": "mV/V",
    "mv/a": "mV/A",
}

# Map the verify/spec ``spec_field`` dimension token onto the same raw-unit
# string the quality ``@kpi(... unit=...)`` surface uses, so a verify-sourced
# KPI lands in exactly the same display-scale / display-label tables as a
# quality-sourced one. ``real`` carries no physical unit -> empty string.
_DIMENSION_TO_UNIT: dict[str, str] = {
    "freq": "Hz",
    "current": "A",
    "volt": "V",
    "cap": "F",
    "time": "s",
    "angle": "deg",
    "db": "dB",
    "real": "",
    # Regulation / ratio dimensions: a regulation figure (e.g.
    # ``line_regulation : real <= 50 mV/V``) carries a composite display
    # unit. Without an entry here the unit string is dropped from the KPI
    # label; these preserve it (display scale stays 1.0 — the value is a raw
    # ratio, not a power-of-ten-shifted SI quantity).
    "reg_line": "mV/V",
    "reg_load": "mV/A",
    "ratio": "",
}

# Map the verify/spec ``spec_field`` operator onto the ``@kpi`` op surface
# token consumed by :func:`_kpi_from_entry`. ``=`` lowers to ``near`` (a
# RELATIVE band, per the project rule); the strict ``>`` / ``<`` collapse to
# the same inclusive bound the engine evaluates (``ge`` / ``le``).
_SPEC_OP_TO_KPI_OP: dict[str, str] = {
    ">=": ">=",
    ">": ">=",
    "<=": "<=",
    "<": "<=",
    "=": "near",
}


def _read_yaml_metadata(case_dir: Path) -> dict[str, Any]:
    """Read ``case.yaml`` (best-effort) and return its ``metadata`` mapping.

    Uses PyYAML if importable; otherwise falls back to an empty mapping so
    path inference proceeds from conventions alone. The top-level ``id`` and
    ``source.top_cell`` are folded into the returned dict for convenience.
    """
    path = case_dir / "case.yaml"
    if not path.is_file():
        return {}
    try:
        import yaml  # type: ignore

        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001 - YAML missing/broken -> conventions only
        return {}
    if not isinstance(doc, dict):
        return {}
    meta = dict(doc.get("metadata") or {})
    meta.setdefault("id", doc.get("id"))
    src = doc.get("source") or {}
    if isinstance(src, dict):
        meta.setdefault("top_cell", src.get("top_cell"))
    return meta


def _resolve_repo_path(value: Any) -> Optional[Path]:
    """Resolve a metadata path value (repo-relative or absolute) to a Path."""
    if not value:
        return None
    p = Path(str(value))
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p


def _short_key_from_layout_module(layout_module: Optional[str], fallback: str) -> str:
    """Derive a short registry key from the layout generator binding.

    Only ``circuits.<key>.<...>`` bindings carry the circuit identity in the
    module path (``circuits.fc_ota.fc_ota_layout`` -> ``fc_ota``); the
    in-repo ``circuits/<key>/`` artifact directory follows that key. For a
    ``clay.layout.<router>`` binding (a shared router, NOT a per-circuit
    package — e.g. ``clay.layout.current_mirror_router``) the module path does
    NOT name the circuit, so the caller's ``fallback`` (the case-directory
    name) is the sane key instead.
    """
    if layout_module:
        parts = layout_module.split(".")
        if len(parts) >= 2 and parts[0] == "circuits":
            return parts[1]
    return fallback


# Case-directory name suffixes that mark a packaging/maturity variant rather
# than the circuit identity. Stripped (longest-first) when deriving a default
# registry key from the case directory, so ``current_mirror/gf180_postlayout``
# and ``fc_ota_current`` collapse to ``current_mirror`` / ``fc_ota``.
_CASE_DIR_VARIANT_SUFFIXES: tuple[str, ...] = (
    "_gf180_postlayout",
    "_smic18mse_3v3",
    "_signoff_gf180",
    "_layout_aware",
    "_postlayout",
    "_multi_role",
    "_fullswing",
    "_current",
    "_signoff",
    "_gf180",
    "_sky130",
    "_demo",
)

# Leaf directory names that carry NO circuit identity — they are whole-leaf
# packaging/maturity descriptors. When the leaf (after suffix stripping) is
# one of these (or empty), the PARENT directory name supplies the circuit key
# (``cases/current_mirror/gf180_postlayout`` -> ``current_mirror``).
_CASE_NONIDENTITY_LEAVES: frozenset[str] = frozenset(
    {
        "gf180_postlayout",
        "gf180",
        "postlayout",
        "signoff",
        "source",
        "final",
    }
)

# Parent directory names that are pure category buckets, never a circuit
# identity. They are NOT used as a fallback key.
_CASE_CATEGORY_DIRS: frozenset[str] = frozenset(
    {
        "opamp",
        "regulator",
        "reference",
        "oscillator",
        "pll",
        "layout_intent",
        "migration",
        "viewer",
        "quickstart",
        "cases",
    }
)


def _strip_variant_suffix(name: str) -> str:
    lowered = name.lower()
    # Longest suffix first so ``_signoff_gf180`` wins over a bare ``_gf180``.
    for suffix in sorted(_CASE_DIR_VARIANT_SUFFIXES, key=len, reverse=True):
        if lowered.endswith(suffix) and len(lowered) > len(suffix):
            return lowered[: -len(suffix)]
    return lowered


def _key_from_case_dir(case_dir: Path) -> str:
    """Derive a clean default registry key from the case directory.

    The leaf directory name has packaging/maturity suffixes stripped
    (``fc_ota_current`` -> ``fc_ota``). If the leaf is purely a packaging
    descriptor (``gf180_postlayout``) or strips to empty, the PARENT
    directory name is used instead — UNLESS the parent is a category bucket
    (``opamp`` / ``regulator`` / ...), in which case the un-stripped leaf is
    the most specific identity available. Falls back to the bare leaf name.
    """
    leaf = case_dir.name
    stripped = _strip_variant_suffix(leaf)
    if stripped and stripped not in _CASE_NONIDENTITY_LEAVES:
        return stripped
    parent = case_dir.parent.name
    parent_stripped = _strip_variant_suffix(parent)
    if parent_stripped and parent_stripped not in _CASE_CATEGORY_DIRS:
        return parent_stripped
    return leaf.lower()


def _discover_layout_functions(layout_module: Optional[str]) -> tuple[str, Optional[str]]:
    """Find the ``build_routed_*`` / ``emit_routed_*`` functions in a module.

    Returns ``(build_function, emit_function|None)``. The names are
    discovered by import + introspection (no per-circuit branch). If the
    module cannot be imported, falls back to the canonical default names so
    the spec still compiles for offline path inspection.
    """
    build_fn = "build_routed"
    emit_fn: Optional[str] = None
    if not layout_module:
        return build_fn, emit_fn
    try:
        module = importlib.import_module(layout_module)
    except Exception:  # noqa: BLE001 - keep spec compilable without the generator
        return build_fn, emit_fn
    builds = sorted(
        name for name in dir(module)
        if name.startswith("build_routed_") and callable(getattr(module, name, None))
    )
    emits = sorted(
        name for name in dir(module)
        if name.startswith("emit_routed_") and callable(getattr(module, name, None))
    )
    if builds:
        build_fn = builds[0]
    if emits:
        emit_fn = emits[0]
    return build_fn, emit_fn


def _first_existing(*candidates: Optional[Path]) -> Optional[Path]:
    for cand in candidates:
        if cand is not None and cand.is_file():
            return cand
    return None


def _kpi_from_entry(key: str, op: str, value: float, unit: Optional[str]) -> KpiSpec:
    """Lower one parsed ``@kpi(key, op, value, unit=...)`` into a KpiSpec.

    The quality block names the BASE metric (``dc_gain_db``); the field read
    from ``post_results.json`` is ``<key>_post`` (the post-layout result).
    """
    post_key = f"{key}_post"
    unit_norm = (unit or "").strip()
    scale = _UNIT_DISPLAY_SCALE.get(unit_norm.lower(), 1.0)
    display_unit = _UNIT_DISPLAY_LABEL.get(unit_norm.lower(), unit_norm)
    if op == "near":
        tol = abs(value) * _NEAR_REL_TOL
        bound: tuple[str, Any] = ("near", (value, tol))
        target = f"~= {value}"
    else:
        bound = (_OP_TO_BOUND[op], value)
        target = f"{op} {value}"
    label = key.replace("_", " ").strip().title()
    return KpiSpec(
        key=post_key,
        label=label,
        unit=display_unit,
        scale=scale,
        target=target,
        bound=bound,
    )


def _consistency_from_entry(key: str, rel_tol: float) -> ConsistencySpec:
    """Lower one ``@consistency(key, rel <= tol)`` into a ConsistencySpec.

    ``pre_key`` is the base metric name (read from the ``pre`` block of
    ``post_results.json``); ``post_key`` is ``<key>_post``.
    """
    return ConsistencySpec(
        pre_key=key,
        post_key=f"{key}_post",
        rel_tol=rel_tol,
        label=f"{key.replace('_', ' ').strip()} pre/post",
    )


def _kpi_from_spec_field(sf: "SpecField") -> KpiSpec:
    """Lower one verify ``spec { name : dimension op value; }`` field to a KpiSpec.

    The verify/spec assertion names the SAME base metric the quality block
    names (e.g. ``dc_gain_db``), with the bound value already in SI units
    (the parser resolves ``10MHz`` -> ``1e7``). The dimension token supplies
    the display unit; the operator maps onto the ``@kpi`` op surface so this
    routes through the identical :func:`_kpi_from_entry` lowering as a
    quality-sourced KPI (same ``_post`` key, same display scale/label).
    """
    unit = _DIMENSION_TO_UNIT.get((sf.dimension or "").strip().lower(), "")
    if not unit:
        # A regulation figure is declared with the dimensionless ``real`` token
        # plus a composite unit suffix on the VALUE (``line_regulation : real
        # <= 50 mV/V``). The parser folds that suffix into the SI value and
        # does not retain it on SpecField, so the dimension alone yields no
        # display unit. Recover the regulation unit from the metric NAME so
        # the display label is preserved (data-driven, no per-circuit branch).
        unit = _regulation_unit_for_name(sf.name)
    kpi_op = _SPEC_OP_TO_KPI_OP.get(sf.op, sf.op)
    return _kpi_from_entry(sf.name, kpi_op, float(sf.value), unit)


# Metric-name -> regulation display unit. A ``real``-dimension spec field whose
# name signals a regulation figure carries its unit (``mV/V`` / ``mV/A``) only
# on the value suffix, which the parser discards; this recovers the label.
_REGULATION_NAME_UNIT: tuple[tuple[str, str], ...] = (
    ("line_regulation", "mV/V"),
    ("load_regulation", "mV/A"),
)


def _regulation_unit_for_name(name: str) -> str:
    lowered = (name or "").strip().lower()
    for marker, unit in _REGULATION_NAME_UNIT:
        if marker in lowered:
            return unit
    return ""


# Nested container keys, in lookup order, where a post-layout-schema results
# JSON parks its measured values. The in-repo ``circuits/*/post_results.json``
# shape is FLAT (``dc_gain_db_post`` at top level); a case-package
# ``final/postlayout_sim.json`` nests under ``metrics`` / ``summary`` / ``op``
# / ``results`` with NO ``_post`` suffix. :func:`resolve_kpi_value` tries both
# layouts so neither schema shows a spurious "pending".
_POST_NESTED_CONTAINERS: tuple[str, ...] = ("metrics", "summary", "op", "results")


def resolve_kpi_value(kpi: KpiSpec, post_results: Optional[Mapping[str, Any]]) -> Any:
    """Resolve a KPI's measured value out of a post-results document.

    The flow's KPI keys follow the quality@kpi ``<base>_post`` convention
    (matching the in-repo flat ``circuits/*/post_results.json``). A
    case-package ``final/postlayout_sim.json`` instead nests its measured
    values under ``metrics`` / ``summary`` / ``op`` / ``results`` keyed by the
    BARE base metric (no ``_post`` suffix). This resolver tries, in order:

      1. the flat top-level ``kpi.key`` (``dc_gain_db_post``);
      2. the flat top-level base key (``dc_gain_db`` — ``_post`` stripped);
      3. each nested container by ``kpi.key`` then by the base key.

    Returns the first non-None hit, else ``None`` (KPI stays "pending"). This
    is the read consumers (the sign step) must route KPI lookups through so
    post-layout-schema circuits resolve real values instead of pending.
    """
    if not post_results:
        return None
    key = kpi.key
    base = key[: -len("_post")] if key.endswith("_post") else key
    # 1 + 2: flat top-level, suffixed then bare base.
    for candidate in (key, base):
        value = post_results.get(candidate)
        if value is not None:
            return value
    # 3: nested containers.
    for container_name in _POST_NESTED_CONTAINERS:
        container = post_results.get(container_name)
        if not isinstance(container, Mapping):
            continue
        for candidate in (key, base):
            value = container.get(candidate)
            if value is not None:
                return value
    return None


def _kpi_tables_disagree(
    primary: Sequence[KpiSpec], shadowed: Sequence[KpiSpec]
) -> bool:
    """True if two KPI tables differ in their keys or any shared key's bound.

    Used to decide whether verify/spec silently overriding a quality@kpi
    table is a benign duplicate (identical targets -> no diagnostic) or a
    real authoring conflict worth a warning.
    """
    p_by_key = {k.key: k.bound for k in primary}
    s_by_key = {k.key: k.bound for k in shadowed}
    if set(p_by_key) != set(s_by_key):
        return True
    return any(p_by_key[key] != s_by_key.get(key) for key in p_by_key)


def _verify_specs_for_cell(
    cell_def: "CellDef",
    verify_jobs: Optional[Sequence["VerifyJob"]],
) -> tuple[KpiSpec, ...]:
    """Collect the verify/spec-sourced KPIs that target ``cell_def``.

    A ``verify <qualified_name> { ... }`` block parses to a top-level
    :class:`clay.ir.VerifyJob` keyed by ``cell_path`` (``pkg::CellName``),
    NOT onto the CellDef directly. We match a job to this cell by comparing
    the trailing ``::CellName`` segment (or the bare path) against
    ``cell_def.local_name``.

    Both the job's top-level ``spec`` and every per-setup ``spec`` (multi-
    setup verify, v0.0.29) contribute assertion fields. Duplicate field
    names (same base metric asserted in multiple setups) collapse to the
    first occurrence so the KPI table stays one-row-per-metric.
    """
    if not verify_jobs:
        return ()
    local_name = str(getattr(cell_def, "local_name", "") or "")
    if not local_name:
        return ()

    fields: list["SpecField"] = []
    for job in verify_jobs:
        cell_path = str(getattr(job, "cell_path", "") or "")
        target = cell_path.rsplit("::", 1)[-1] if "::" in cell_path else cell_path
        if target != local_name:
            continue
        spec = getattr(job, "spec", None)
        if spec is not None:
            fields.extend(getattr(spec, "fields", []) or [])
        for setup in getattr(job, "setups", []) or []:
            setup_spec = getattr(setup, "spec", None)
            if setup_spec is not None:
                fields.extend(getattr(setup_spec, "fields", []) or [])

    kpis: list[KpiSpec] = []
    seen: set[str] = set()
    for sf in fields:
        if sf.name in seen:
            continue
        seen.add(sf.name)
        kpis.append(_kpi_from_spec_field(sf))
    return tuple(kpis)


def compile_flow_spec(
    cell_def: "CellDef",
    *,
    case_dir: Path,
    verify_jobs: Optional[Sequence["VerifyJob"]] = None,
    policies: Optional[Mapping[str, "ParsedPolicyDecl"]] = None,
    discover_layout: bool = True,
    metadata: Optional[Mapping[str, Any]] = None,
    inspect_artifacts: bool = True,
) -> CircuitSpec:
    """Lower a parsed CellDef's ``flow {}``/``quality {}`` blocks into a CircuitSpec.

    The frozen ``CircuitSpec`` is the COMPILED PROJECTION of the source-level
    ``flow`` + ``quality`` blocks. Circuit-specific paths are inferred from
    ``case_dir`` / ``case.yaml`` metadata and PDK conventions (no per-circuit
    code branch).

    KPI source precedence (the language has ONE canonical KPI target): if the
    cell carries ``verify { ... spec { ... } }`` assertions (passed via
    ``verify_jobs`` — typically ``ClayFile.verify_jobs``), those ``spec_field``
    assertions are the PRIMARY KPI table. The ``quality { @kpi(...) }`` entries
    are used only when the cell has NO verify/spec KPIs. The other quality
    dimensions — ``@consistency`` (pre/post relative band) and ``@require``
    (drc_clean / lvs_match / intent_signed gates) — are not expressible in
    verify/spec and are always sourced from the quality block.
    """
    flow = getattr(cell_def, "flow_decl", None)
    quality = getattr(cell_def, "quality_decl", None)
    if flow is None or quality is None:
        raise ValueError(
            "compile_flow_spec requires a CellDef carrying BOTH a flow{} and "
            "a quality{} block (the compiled CircuitSpec is their projection)."
        )

    policy_ref = str(getattr(flow, "policy_ref", "") or "") or None
    policy_decl = None
    if policy_ref is not None and policies is not None:
        policy_decl = policies.get(policy_ref)
        if policy_decl is None:
            raise ValueError(
                f"flow for cell {getattr(cell_def, 'local_name', '')!r} selects "
                f"unknown policy {policy_ref!r}; available policies: "
                f"{sorted(policies)}"
            )

    top_cell = str(getattr(cell_def, "local_name", "") or "")
    meta = dict(metadata) if metadata is not None else _read_yaml_metadata(case_dir)

    layout_module = flow.layout_module or ""
    # Registry-key resolution precedence:
    #   1. an explicit case.yaml ``metadata.key`` / ``metadata.flow_key``
    #      override (lets a package name itself when neither the binding nor
    #      the directory name is ideal);
    #   2. the ``circuits.<key>.<...>`` binding (key embedded in the path);
    #   3. the CASE DIRECTORY name (``current_mirror`` from
    #      ``cases/current_mirror/gf180_postlayout``) — the sane default for a
    #      shared ``clay.layout.<router>`` binding or a LAYOUT-LESS package,
    #      where the binding does NOT name the circuit. Falls back through the
    #      case parent then the bare dir so a nested ``gf180_postlayout`` leaf
    #      does not become the key.
    key_override = meta.get("key") or meta.get("flow_key")
    dir_default = _key_from_case_dir(case_dir)
    if key_override:
        short_key = str(key_override)
    else:
        short_key = _short_key_from_layout_module(layout_module, fallback=dir_default)
    design_id = str(meta.get("id") or case_dir.name)

    # --- path inference: circuit artifact directory ------------------- #
    # A ``circuits.<key>.<...>`` layout binding (or an explicit
    # ``metadata.source_spice``) means the design's artifacts live in the
    # in-repo ``circuits/<key>/`` directory. A shared ``clay.layout.<router>``
    # binding or a LAYOUT-LESS package has NO such directory — its artifacts
    # (PEX deck, post-layout sim) ship inside the case package and are
    # surfaced via explicit case.yaml metadata paths. In that situation the
    # in-repo guess is wrong, so identity paths resolve to the case package
    # (final/) when present, else honest None / a case-relative default rather
    # than a fabricated ``circuits/<key>`` path.
    binding_is_in_repo = inspect_artifacts and bool(layout_module) and layout_module.split(".")[0] == "circuits"
    # Pure compilation never discovers artifact paths or reads old results.
    # These conventional paths remain unresolved, and engine admission refuses
    # such a spec until an explicit artifact resolver has bound its inputs.
    resolve_path = _resolve_repo_path if inspect_artifacts else lambda _value: None
    first_existing = _first_existing if inspect_artifacts else lambda *_paths: None
    source_cir = resolve_path(meta.get("source_spice"))
    if source_cir is not None:
        circuit_dir = source_cir.parent
    elif binding_is_in_repo:
        circuit_dir = REPO_ROOT / "circuits" / short_key
    else:
        circuit_dir = case_dir
    if source_cir is None:
        source_cir = first_existing(
            circuit_dir / f"{top_cell}.cir",
            circuit_dir / f"{short_key}_gf180.cir",
            circuit_dir / f"{short_key}.cir",
            case_dir / "source" / f"{top_cell}.cir",
        )
        if source_cir is None:
            # No on-disk .cir. ``CircuitSpec.source_cir`` is a non-optional
            # ``Path`` (frozen field), so point at the conventional location
            # rather than fabricating an in-repo ``circuits/<key>`` path for a
            # shared-router / layout-less package: those resolve INSIDE the
            # case package (the PEX deck there is the sim source of truth).
            if binding_is_in_repo:
                source_cir = circuit_dir / f"{short_key}.cir"
            else:
                source_cir = case_dir / "source" / f"{top_cell}.cir"

    layout_guide = resolve_path(meta.get("layout_guide")) or (circuit_dir / "layout_guide.txt")
    presim_results = resolve_path(meta.get("pre_results")) or (circuit_dir / "presim_results.json")
    post_results = resolve_path(meta.get("post_results")) or (circuit_dir / "post_results.json")
    pex_spice = resolve_path(meta.get("pex_spice")) or first_existing(
        circuit_dir / f"{top_cell}.pex.spice",
        circuit_dir / f"{short_key}.pex.spice",
    )
    drc_report = resolve_path(meta.get("drc_report")) or first_existing(
        circuit_dir / f"{short_key}_current.lyrdb",
        circuit_dir / f"{top_cell}.lyrdb",
    )

    # --- layout generator binding ------------------------------------- #
    build_function, emit_function = (
        _discover_layout_functions(layout_module or None)
        if discover_layout and inspect_artifacts else ("build_routed", None)
    )
    gds_cell_name = str(meta.get("gds_cell_name") or top_cell)
    fallback_gds = resolve_path(meta.get("layout_gds")) or first_existing(
        circuit_dir / f"{gds_cell_name}.gds",
        circuit_dir / f"{short_key}_current.gds",
        circuit_dir / f"{top_cell}.gds",
    ) or (circuit_dir / f"{gds_cell_name}.gds")

    # postsim_module binding. An explicit ``metadata.postsim_module`` always
    # wins — this lets a LAYOUT-LESS (sim-only) package bind a post-sim
    # entrypoint even though it has no ``layout = ...`` binding to derive one
    # from. With no override, fall back to the sibling-package convention for
    # an in-repo ``circuits.<key>`` design (where ``circuits.<key>.postsim``
    # is the established location). A shared-router / layout-less package with
    # no override stays unbound (None) — its post-sim is the captured
    # post_results.json rather than a live module.
    postsim_module = meta.get("postsim_module")
    if postsim_module is None and binding_is_in_repo:
        # convention: sibling postsim module in the same in-repo package
        postsim_module = f"circuits.{short_key}.postsim"
    presim_module = meta.get("presim_module")
    presim_function = str(meta.get("presim_function") or "run_candidate")
    postsim_function = (
        str(meta.get("postsim_function")) if meta.get("postsim_function") else None
    )

    # --- supply / pdk: prefer the flow block, fall back to results ---- #
    pdk = flow.pdk or str(meta.get("pdk") or "gf180mcu")
    vdd = _infer_vdd(post_results, presim_results, meta) if inspect_artifacts else float(meta.get("vdd", 3.3))

    # --- KPI source: verify/spec is canonical, quality@kpi is fallback - #
    # The language has ONE canonical KPI target. When the cell carries
    # verify/spec assertions they are the PRIMARY table; quality@kpi only
    # fills in when there are none (the current 4 production designs have
    # no verify blocks -> they keep sourcing from quality@kpi).
    verify_kpis = _verify_specs_for_cell(cell_def, verify_jobs)
    quality_kpis = tuple(
        _kpi_from_entry(key, op, value, unit)
        for (key, op, value, unit) in quality.kpis
    )
    if verify_kpis:
        kpis = verify_kpis
        # The quality@kpi table is SHADOWED by verify/spec. Silently dropping
        # it hides a real authoring conflict (a quality@kpi the author thinks
        # is in effect but is not). Emit a diagnostic when both surfaces carry
        # KPIs AND they disagree (different keys or different bounds), so the
        # shadowing is visible rather than silent.
        if quality_kpis and _kpi_tables_disagree(verify_kpis, quality_kpis):
            import warnings

            v_keys = sorted({k.key for k in verify_kpis})
            q_keys = sorted({k.key for k in quality_kpis})
            warnings.warn(
                f"cell {top_cell!r}: verify/spec KPIs {v_keys} SHADOW the "
                f"quality {{@kpi}} table {q_keys}; the quality@kpi bounds are "
                f"ignored. Reconcile the two surfaces (verify/spec is the "
                f"canonical KPI source) or drop the redundant @kpi entries.",
                stacklevel=2,
            )
    else:
        kpis = quality_kpis
    consistency = tuple(
        _consistency_from_entry(key, rel_tol)
        for (key, rel_tol) in quality.consistencies
    )
    gates = tuple(GateSpec(kind) for kind in quality.requires)

    cell_corners = tuple(getattr(cell_def, "_corners", ()) or ())
    corners = cell_corners or (tuple(flow.corners) if flow.corners else ("tt",))
    temps_c = tuple(float(value) for value in (getattr(cell_def, "_temps", ()) or (27.0,)))
    steps = tuple(flow.steps) if flow.steps else DEFAULT_STEPS
    optimization = _optimization_from_sizing(cell_def)

    native_runtime_manifest = _resolve_repo_path(meta.get("native_runtime_manifest"))
    native_runtime_sha256 = None
    if native_runtime_manifest is not None and inspect_artifacts:
        import hashlib

        native_runtime_manifest = native_runtime_manifest.resolve()
        native_runtime_sha256 = hashlib.sha256(native_runtime_manifest.read_bytes()).hexdigest()

    return CircuitSpec(
        key=short_key,
        design_id=design_id,
        display_name=str(meta.get("title") or top_cell),
        cell_type=str(meta.get("category") or "circuit"),
        top_cell=top_cell,
        pdk=pdk,
        vdd=vdd,
        case_dir=case_dir,
        source_cir=source_cir,
        layout_guide=layout_guide,
        presim_results=presim_results,
        post_results=post_results,
        layout_module=layout_module,
        build_function=build_function,
        emit_function=emit_function,
        gds_cell_name=gds_cell_name,
        fallback_gds=fallback_gds,
        pex_spice=pex_spice,
        drc_report=drc_report,
        presim_module=str(presim_module) if presim_module else None,
        presim_function=presim_function,
        postsim_module=postsim_module,
        postsim_function=postsim_function,
        lvs_ref="effective_w",
        corners=corners,
        temps_c=temps_c,
        steps=steps,
        kpis=kpis,
        consistency=consistency,
        gates=gates,
        optimization=optimization,
        policy_ref=policy_ref,
        policy_decl=policy_decl,
        source_contract_note=(
            f"compiled from source-level flow{{}}/quality{{}} of cell "
            f"{top_cell} in {case_dir.name}/source/design.clay"
        ),
        artifacts_resolved=inspect_artifacts,
        native_runtime_manifest=native_runtime_manifest,
        native_runtime_sha256=native_runtime_sha256,
    )


def _optimization_from_sizing(cell_def: "CellDef") -> Optional[OptimizationSpec]:
    """Lower the reserved optimization convention in existing sizing data."""
    sizing = {
        str(key): float(value)
        for key, value in (getattr(cell_def, "sizing_kwargs", {}) or {}).items()
    }
    engine_keys = sorted(key for key in sizing if key.startswith("opt_engine_") and sizing[key] != 0)
    if not engine_keys:
        return None
    if len(engine_keys) != 1:
        raise ValueError(f"sizing optimization must enable exactly one engine: {engine_keys}")
    engine = engine_keys[0][len("opt_engine_"):]

    objective_keys = sorted(
        key for key in sizing
        if (key.startswith("opt_objective_maximize_") or key.startswith("opt_objective_minimize_"))
        and sizing[key] != 0
    )
    if len(objective_keys) != 1:
        raise ValueError(
            "sizing optimization must enable exactly one "
            f"opt_objective_{{maximize|minimize}}_* key: {objective_keys}"
        )
    objective_key = objective_keys[0]
    prefix = (
        "opt_objective_maximize_"
        if objective_key.startswith("opt_objective_maximize_")
        else "opt_objective_minimize_"
    )
    direction = "maximize" if "maximize" in prefix else "minimize"

    required = (
        "opt_seed", "opt_budget", "opt_post_every",
        "opt_normalized_low", "opt_normalized_high", "opt_truth_post_pex",
    )
    missing = [key for key in required if key not in sizing]
    if missing:
        raise ValueError(f"sizing optimization is missing required fields: {missing}")

    def positive_int(key: str) -> int:
        value = sizing[key]
        if value <= 0 or not value.is_integer():
            raise ValueError(f"sizing {key} must be a positive integer, got {value!r}")
        return int(value)

    normalized_domain = (sizing["opt_normalized_low"], sizing["opt_normalized_high"])
    if not normalized_domain[0] < normalized_domain[1]:
        raise ValueError("sizing normalized domain must have low < high")
    if sizing["opt_truth_post_pex"] != 1:
        raise ValueError("sizing optimization truth must enable post_pex")

    binding_map: dict[str, list[str]] = {}
    for key, enabled in sizing.items():
        if not key.startswith("opt_bind__"):
            continue
        if enabled != 1:
            raise ValueError(f"sizing binding {key!r} must equal 1")
        parts = key[len("opt_bind__"):].split("__")
        if len(parts) < 3 or not all(parts):
            raise ValueError(
                f"sizing binding {key!r} must be "
                "opt_bind__VARIABLE__TARGET_KIND__TARGET..."
            )
        variable, target = parts[0], ".".join(parts[1:])
        binding_map.setdefault(variable, []).append(target)

    scale_map: dict[str, dict[str, float]] = {}
    for key, scale in sizing.items():
        if not key.startswith("opt_bind_scale__"):
            continue
        parts = key[len("opt_bind_scale__"):].split("__")
        if len(parts) < 3 or not all(parts) or not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"invalid positive dimensionless binding scale {key!r}")
        variable, target = parts[0], ".".join(parts[1:])
        if target not in binding_map.get(variable, ()):
            raise ValueError(f"binding scale {key!r} has no matching opt_bind declaration")
        scale_map.setdefault(variable, {})[target] = scale

    variables: list[OptimizationVariableSpec] = []
    base_names = sorted(
        key for key in sizing
        if not key.startswith("opt_") and not key.endswith(("_min", "_max"))
    )
    for name in base_names:
        low_key, high_key = f"{name}_min", f"{name}_max"
        has_low, has_high = low_key in sizing, high_key in sizing
        if has_low != has_high:
            raise ValueError(f"sizing variable {name!r} must declare both _min and _max")
        if not has_low:
            continue
        low, high, default = sizing[low_key], sizing[high_key], sizing[name]
        if not low < high:
            raise ValueError(f"sizing variable {name!r} requires min < max")
        if not low <= default <= high:
            raise ValueError(f"sizing variable {name!r} default lies outside its bounds")
        bindings = tuple(binding_map.pop(name, ()))
        if not bindings:
            raise ValueError(f"sizing variable {name!r} has no opt_bind declaration")
        step = sizing.get(f"opt_step__{name}")
        if step is not None:
            if not math.isfinite(step) or step <= 0:
                raise ValueError(f"sizing variable {name!r} requires a positive finite step")
            if not math.isclose(default / step, round(default / step), rel_tol=0, abs_tol=1e-7):
                raise ValueError(f"sizing variable {name!r} default is not on its declared step")
        scales = tuple(sorted(scale_map.pop(name, {}).items()))
        variables.append(OptimizationVariableSpec(name, default, low, high, bindings, step, scales))
    if not variables:
        raise ValueError("sizing optimization declares no bounded variables")
    if binding_map:
        raise ValueError(
            "sizing bindings reference variables without bounds: "
            f"{sorted(binding_map)}"
        )
    unknown_steps = {key[len("opt_step__"):] for key in sizing if key.startswith("opt_step__")} - {v.name for v in variables}
    if unknown_steps:
        raise ValueError(f"sizing steps reference variables without bounds: {sorted(unknown_steps)}")

    degradation = tuple(sorted(
        (key[len("opt_degradation_"):], value)
        for key, value in sizing.items()
        if key.startswith("opt_degradation_")
    ))
    if any(value < 0 for _name, value in degradation):
        raise ValueError("sizing optimization degradation limits must be non-negative")

    return OptimizationSpec(
        engine=engine,
        seed=positive_int("opt_seed"),
        budget=positive_int("opt_budget"),
        post_every=positive_int("opt_post_every"),
        normalized_domain=normalized_domain,
        truth="post_pex",
        variables=tuple(variables),
        objective=OptimizationObjectiveSpec(
            name=objective_key[len(prefix):],
            direction=direction,
            cl_f=sizing.get("opt_objective_cl_f", 1.0),
        ),
        degradation_limits=degradation,
    )


def _infer_vdd(
    post_results: Path,
    presim_results: Path,
    meta: Mapping[str, Any],
) -> float:
    """Use explicit case intent first, then infer legacy cases from results."""
    try:
        explicit = float(meta.get("vdd"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        explicit = None
    if explicit is not None:
        return explicit
    for path in (post_results, presim_results):
        if path is not None and path.is_file():
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            for field_name in ("vdd", "vin"):
                value = doc.get(field_name)
                if isinstance(value, (int, float)):
                    return float(value)
    return 3.3


import re

_EXCLUDED_DISCOVERY_DIRS: frozenset[str] = frozenset(
    {
        "three_circuit_comparison",
        "joint_access",
        "_flow_run",
        "runs",
        "__pycache__",
    }
)
_FLOW_BLOCK_PRECHECK_RE = re.compile(r"(?m)^\s*flow\s+[A-Za-z_]")
_QUALITY_BLOCK_PRECHECK_RE = re.compile(r"(?m)^\s*quality\s+[A-Za-z_]")
_REGISTRY_CACHE: dict[
    tuple[str, bool],
    tuple[tuple[tuple[str, int, int], ...], dict[str, CircuitSpec]],
] = {}


def clear_registry_cache() -> None:
    """Clear the in-memory mtime/size-validated ``discover_registry`` cache."""
    _REGISTRY_CACHE.clear()


def _candidate_design_files(root: Path) -> list[Path]:
    """Enumerate ``**/source/design.clay`` under ``root``, pruning scratch/quarantined subtrees."""
    import os

    out: list[Path] = []
    if not root.is_dir():
        return out
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in _EXCLUDED_DISCOVERY_DIRS)
        if os.path.basename(dirpath) == "source" and "design.clay" in filenames:
            out.append(Path(dirpath) / "design.clay")
    out.sort()
    return out


def _registry_snapshot_signature(designs: list[Path]) -> tuple[tuple[str, int, int], ...]:
    sig: list[tuple[str, int, int]] = []
    for design in designs:
        try:
            st = design.stat()
            sig.append((str(design), int(st.st_mtime_ns), int(st.st_size)))
        except OSError:
            continue
        case_yaml = design.parent.parent / "case.yaml"
        if case_yaml.is_file():
            try:
                cst = case_yaml.stat()
                sig.append((str(case_yaml), int(cst.st_mtime_ns), int(cst.st_size)))
            except OSError:
                pass
    return tuple(sig)


def discover_registry(cases_root: Optional[Path] = None, *, strict: bool = False) -> dict[str, CircuitSpec]:
    """Discover every case package carrying ``flow {}``/``quality {}`` and compile it.

    Scans ``cases/**/source/design.clay`` (excluding scratch ``_flow_run`` / ``runs``
    and quarantined ``three_circuit_comparison`` / ``joint_access`` trees unless
    explicitly targeted via ``cases_root``), parses each candidate, and for every
    cell carrying BOTH a ``flow`` and a ``quality`` block compiles a
    :class:`CircuitSpec`. Results are cached per ``(root, strict)`` and validated
    against live ``(path, st_mtime_ns, st_size)`` so any on-disk edit invalidates
    immediately.
    """
    root = Path(cases_root) if cases_root is not None else (REPO_ROOT / "cases")
    designs = _candidate_design_files(root)
    cache_key = (str(root.resolve()), bool(strict))
    sig = _registry_snapshot_signature(designs)
    cached = _REGISTRY_CACHE.get(cache_key)
    if cached is not None and cached[0] == sig:
        return dict(cached[1])

    from clay.parser import parse_clay_file  # local import: avoid load-time cycle

    registry: dict[str, CircuitSpec] = {}
    for design in designs:
        case_dir = design.parent.parent
        if not strict:
            try:
                raw_text = design.read_text(encoding="utf-8-sig", errors="ignore")
            except OSError:
                continue
            if not (
                _FLOW_BLOCK_PRECHECK_RE.search(raw_text)
                and _QUALITY_BLOCK_PRECHECK_RE.search(raw_text)
            ):
                continue
        try:
            cf = parse_clay_file(design)
        except Exception as exc:  # noqa: BLE001 - legacy discovery is best effort
            if strict:
                raise ValueError(f"cannot parse selected case {design}: {exc}") from exc
            continue
        for cell in getattr(cf, "cell_defs", []) or []:
            if getattr(cell, "flow_decl", None) is None:
                continue
            if getattr(cell, "quality_decl", None) is None:
                continue
            try:
                spec = compile_flow_spec(
                    cell,
                    case_dir=case_dir,
                    verify_jobs=getattr(cf, "verify_jobs", None),
                    policies=getattr(cf, "policies", None),
                    discover_layout=not strict,
                    inspect_artifacts=not strict,
                )
                if strict and set(spec.steps) - set(DEFAULT_STEPS):
                    raise ValueError(f"unknown flow steps: {set(spec.steps) - set(DEFAULT_STEPS)}")
            except Exception as exc:  # noqa: BLE001
                if strict:
                    raise ValueError(f"cannot compile selected case {design}: {exc}") from exc
                continue
            if strict and spec.key in registry:
                raise ValueError(f"duplicate flow registry key {spec.key!r}: {registry[spec.key].case_dir}, {case_dir}")
            registry[spec.key] = spec
    _REGISTRY_CACHE[cache_key] = (sig, dict(registry))
    return registry
