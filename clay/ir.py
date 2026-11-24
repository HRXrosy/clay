"""
Clay simulator-neutral IR.

Anything below this layer (drivers/) translates IR to simulator-specific
text. The IR itself never carries dialect-specific syntax.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Mapping, Optional, Sequence


# ─── Spec field ─────────────────────────────────────────────

@dataclass
class SpecField:
    name: str
    dimension: str                # 'real' | 'freq' | 'volt' | 'current' | 'cap' | 'time' | 'angle' | 'db'
    op: str                       # '>=' | '<=' | '=' | '>' | '<'
    value: float                  # in SI units
    soft: bool = False
    # v0.0.30 (G11 closeout): per-assertion corner scope. None means the
    # spec applies at every corner the parent VerifyJob is running at.
    # When a non-empty list is supplied, the field is checked ONLY at
    # the listed corners — at other corners the field's assertion is
    # skipped (still recorded as a pass for that point so the AND-of-
    # corners aggregate stays well-defined). The parent VerifyJob's
    # corner sweep is augmented (UNION) so every listed corner actually
    # gets a sim run.
    across_corners: Optional[list[str]] = None
    # v0.0.30 (G11 closeout): per-assertion margin as a percentage
    # (e.g. 20.0 means 20%). None means no margin tightening. The
    # margin direction depends on the operator — see verifier docstring.
    margin_pct: Optional[float] = None
    # Proof-driven Clay core: stable source-obligation identity. Existing
    # specs may omit this; the verifier derives a deterministic id from
    # cell name, field position, field name, and operator.
    source_obligation_id: Optional[str] = None
    # Optional parser provenance. Kept None for constructed/test IR and
    # populated by parse_clay_file/parse_clay_source when available.
    source_span: Optional[dict[str, Any]] = None

    def source_obligation_id_for(
        self,
        cell_name: str,
        index: int,
        *,
        scope: Optional[str] = None,
    ) -> str:
        if self.source_obligation_id:
            return self.source_obligation_id
        scope_part = f":{_obligation_scope(scope)}" if scope else ""
        return (
            f"spec:{_obligation_atom(cell_name)}{scope_part}:{index}:"
            f"{_obligation_atom(self.name)}:{_op_obligation_token(self.op)}"
        )


def _obligation_atom(value: str) -> str:
    atom = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return atom or "unnamed"


def _obligation_scope(value: str) -> str:
    return ":".join(_obligation_atom(part) for part in str(value).split(":"))


def _op_obligation_token(op: str) -> str:
    return {
        ">=": "gte",
        "<=": "lte",
        ">": "gt",
        "<": "lt",
        "=": "eq",
    }.get(op, _obligation_atom(op))


@dataclass
class Spec:
    fields: list[SpecField]

    def get(self, name: str) -> SpecField:
        for f in self.fields:
            if f.name == name:
                return f
        raise KeyError(name)


# ─── Device-level IR ────────────────────────────────────────

@dataclass
class Device:
    """One concrete sized device after sizing pass."""
    instance_name: str            # 'M1', 'Mb1', etc.
    device_class: str             # 'nmos', 'pmos', 'cap', 'res', 'cur_src', 'ind'
    pdk_variant: Optional[str]    # 'sky130_fd_pr__nfet_01v8' or None for primitive
    pins: dict[str, str]          # {'d': 'd1', 'g': 'in_p', 's': 'tail', 'b': 'vdd'}
    sizing: dict[str, float]      # {'W': 10e-6, 'L': 0.5e-6, 'M': 1}
    # Source-netlist model identity. The semantic device_class remains portable;
    # a PDK adapter owns the target model/geometry binding.
    source_model: Optional[str] = None
    # Only explicit ``device ... : native`` declarations carry these original
    # native X-card parameter tokens. Ordered immutable pairs retain expression
    # spelling and distinguish nf, m, mult, MF and vm without normalization.
    native_parameters: tuple[tuple[str, str], ...] = ()


# ─── Sources and loads ──────────────────────────────────────

@dataclass
class Source:
    """A simulator-neutral source IR.

    `kind` selects the source shape; the legal `params` keys depend on
    the kind.

    Voltage/current DC and AC kinds:
      * `voltage_dc`   — params: {'dc': <volts>}.
      * `voltage_ac`   — params: {'dc': <volts>, 'ac': <volts>}.
      * `current_dc`   — params: {'dc': <amps>}.

    Voltage/current PULSE kind (v0.0.34):
      * `voltage_pulse` / `current_pulse` — params:
          'v_low'    — low value (volts or amps)
          'v_high'   — high value (volts or amps)
          'td'       — delay before the first transition (seconds);
                       defaults to 0 if absent.
          't_rise'   — rise time (seconds); defaults to 10n if absent.
          't_fall'   — fall time (seconds); defaults to `t_rise` if absent.
          'pulse_width' — pulse width (seconds); defaults to t_end/2.
          'period'   — pulse period (seconds); defaults to t_end (one cycle).
        Matches ngspice `PULSE(v_low v_high td t_rise t_fall pulse_width period)`
        syntax. Names use the canonical `v_*` / `t_*` prefixes even for
        current sources so the IR vocabulary stays uniform.

    Voltage/current PWL kind (v0.0.34):
      * `voltage_pwl` / `current_pwl` — params:
          'pwl_points' — a sequence of alternating time/value floats
                         (t0, v0, t1, v1, t2, v2, ...). Stored on the
                         params map as a single key with a list value
                         (the IR layer is dataclass-flat, so the typed
                         `dict[str, float]` annotation is widened to
                         `dict[str, Any]` for params at the impl level).
                         The ngspice driver emits this as
                         `PWL(t0 v0 t1 v1 ...)`. Times must be
                         monotonically increasing and the list must
                         have an even length.
    """
    kind: str                     # see docstring above for the full set
    instance_name: str            # 'V1', 'Iref', etc.
    pins: dict[str, str]          # {'p': 'vdd', 'n': '0'}
    params: dict[str, Any]        # see kind-specific docs above


# ─── Analyses ────────────────────────────────────────────────

@dataclass
class AcAnalysis:
    sweep_kind: str = "dec"       # 'dec' | 'lin' | 'oct'
    points_per_decade: int = 10
    f_start: float = 1.0
    f_stop: float = 1e9


@dataclass
class DcAnalysis:
    sweep_source: Optional[str] = None   # source instance name
    start: float = 0.0
    stop: float = 0.0
    step: float = 0.0


@dataclass
class TranAnalysis:
    step: float = 1e-9
    t_end: float = 1e-6


@dataclass
class OpAnalysis:
    pass


Analysis = AcAnalysis | DcAnalysis | TranAnalysis | OpAnalysis


# ─── Measurements ────────────────────────────────────────────

@dataclass
class Measure:
    """A simulator-neutral measurement request."""
    name: str                     # 'gain_dc', 'gbw', 'phase_at_gbw'
    analysis_ref: str             # 'ac', 'tran', etc.
    expr: str                     # neutral expression, e.g. "vdb(out) @ freq=1"
    # The driver translates expr to its dialect's .meas / extract.


# ─── Cell IR ─────────────────────────────────────────────────

@dataclass
class Cell:
    name: str
    ports: list[str]              # ['inn', 'inp', 'out', 'vdd', 'vss']
    spec: Spec
    devices: list[Device] = field(default_factory=list)   # filled by sizing pass
    pdk_id: str = "sky130"
    functional_roles: list["ParsedFunctionalRole"] = field(
        default_factory=list,
        kw_only=True,
        repr=False,
        compare=False,
    )
    layout_intents: list["ParsedLayoutIntentDecl"] = field(
        default_factory=list,
        kw_only=True,
        repr=False,
        compare=False,
    )


@dataclass
class Testbench:
    """A complete simulation scenario: DUT instance + sources + loads + analyses + measures."""
    cell: Cell
    sources: list[Source] = field(default_factory=list)
    extra_devices: list[Device] = field(default_factory=list)   # AC source via VCVS, load caps, etc.
    analyses: list[Analysis] = field(default_factory=list)
    measures: list[Measure] = field(default_factory=list)
    pdk_corner: str = "tt"
    supply_volt: float = 1.8
    # v0.0.24 (G11): nominal simulation temperature in Celsius. Drivers
    # emit a `.option temp = <value>` line so the sweep result reflects
    # the requested point. Default 27 °C matches the historical
    # ngspice default (no temp override emitted before this version).
    temp: float = 27.0


# ─── Verification result ─────────────────────────────────────

@dataclass
class AssertionResult:
    """One spec-field assertion outcome.

    The `category` field distinguishes:
      * 'spec_violation'    — circuit ran and produced a number that
                              does not meet the spec
      * 'measure_missing'   — circuit ran but the simulator log does
                              not contain the measure for this field
                              (usually means a .meas failed to extract
                              — convergence, range mismatch, etc.)
      * 'simulator_error'   — the simulator crashed / did not produce
                              a log at all (driver-level failure)
      * 'pass'              — everything OK

    Code review #6 (2026-05-15): without this distinction, a failed
    sim looks indistinguishable from underperforming circuit.
    """
    name: str
    op: str
    expected: float
    actual: Optional[float]
    passed: bool
    message: str = ""
    category: str = "pass"     # 'pass' / 'spec_violation' / 'measure_missing' / 'simulator_error'
    source_obligation_id: Optional[str] = None


@dataclass
class VerifyResult:
    cell_name: str
    measures: Mapping[str, float]
    assertions: list[AssertionResult]
    log_path: str = ""
    netlist_path: str = ""
    simulator_error: Optional[str] = None      # set if the driver failed
                                                # to produce any log at all

    @property
    def all_passed(self) -> bool:
        return self.simulator_error is None and all(
            a.passed for a in self.assertions
        )

    @property
    def simulator_ok(self) -> bool:
        """True if the simulator completed (log produced) — even if some
        assertions failed."""
        return self.simulator_error is None

    @property
    def spec_violations(self) -> list[AssertionResult]:
        """Assertions where the circuit ran but underperformed."""
        return [a for a in self.assertions if a.category == "spec_violation"]

    @property
    def measure_missing(self) -> list[AssertionResult]:
        """Assertions where simulator ran but did not emit this measure."""
        return [a for a in self.assertions if a.category == "measure_missing"]


# ─── Verify setup (multi-setup verify, v0.0.29 / G4) ─────────────


@dataclass
class SetupSpec:
    """One test configuration within a multi-setup verify (v0.0.29, G4).

    A multi-setup verify lets a single cell be exercised under several
    drive conditions (differential-mode for gain_dc / common-mode for
    CMRR / supply-AC for PSRR) in one verify run. Each setup gets its
    OWN testbench, its own ngspice simulation, and its own measure
    dictionary. Top-level derived measures (declared with
    `measure NAME := <expr>;`) are computed in Python AFTER every
    setup completes.

    Fields:
      * name       — local setup handle (`adm`, `acm`, `psrr_p`, ...).
                     Unique within the parent VerifyJob.
      * drive_mode — single keyword identifying the stimulus pattern:
                       'ac_diff'      — VCVS pair, opposite phase (the
                                        historical default)
                       'ac_common'    — same source on inn/inp (CMRR)
                       'ac_supply_p'  — vdd carries 1V AC (PSRR+)
                       'ac_supply_n'  — vss carries 1V AC (PSRR−)
                     Future drive keywords will land here without
                     grammar changes.
      * drive_value — the RHS of the drive field (e.g. `1` for
                     `ac_diff = 1;`). Reserved — v0 only uses presence.
      * spec       — optional per-setup spec block. Assertions on the
                     measures emitted by THIS setup's ngspice run only.
                     A setup with no spec block still contributes its
                     measures to the top-level derived expressions.
    """
    name: str
    drive_mode: str
    drive_value: float = 1.0
    spec: Optional[Spec] = None


# ─── Verify job (parsed from .clay file's `verify { ... }` block) ─

@dataclass
class VerifyJob:
    cell_path: str                                # 'pkg.path::CellName'
    spec: Spec
    sizing_overrides: dict[str, float] = field(default_factory=dict)
    # v0.0.21: declarative corner sweep. Default ["tt"] keeps every
    # pre-existing .clay file's behavior identical (single typical run).
    # The CLI iterates this list and AND's the per-corner pass/fail.
    corners: list[str] = field(default_factory=lambda: ["tt"])
    # v0.0.24 (G11): declarative discrete temperature sweep, in
    # Celsius. Default [27.0] preserves pre-G11 single-run behavior.
    # When combined with `corners` the CLI does a cartesian sweep
    # (corners × temps) — all combinations must pass.
    temps: list[float] = field(default_factory=lambda: [27.0])
    # v0.0.29 (G4 multi-setup verify): list of test configurations.
    # Empty list (default) means "single-setup verify" — the historical
    # behavior driven by the top-level `spec { ... }` block and the
    # cell's default testbench. A non-empty list flips the CLI into
    # multi-setup mode: per-setup tb build + sim + measure collection
    # + cross-setup derived measure computation.
    setups: list[SetupSpec] = field(default_factory=list)
    # v0.0.29 (G4): top-level derived measure declarations, captured
    # as `name → expression_string`. The expression is a small scalar
    # arithmetic expression over `<setup>.<measure>` references — see
    # `clay.verifier._eval_derived_expr` for the supported subset.
    # Computed AFTER every setup's ngspice run completes.
    derived_measures: dict[str, str] = field(default_factory=dict)
    source_span: Optional[dict[str, Any]] = None
    # Optional proof-source declarations for production layout signoff
    # rows that must be present before the file-level proof certificate
    # may emit a token. Entries are signoff kinds such as "drc", "lvs",
    # or explicit "kind:check" strings such as "drc:gf180mcu".
    required_signoff: list[str] = field(default_factory=list)


@dataclass
class BiasNode:
    """One declarative bias-tree node (v0.0.26 / G2).

    A `bias_tree { ... }` block inside a `cell_def` declares the bias
    chain explicitly — instead of the stage template hardcoding it. Each
    declaration becomes one `BiasNode`. Five `kind` values are supported
    in the MVP:

      * 'external'    — an injected reference current
                        (`name: external = 10uA;`)
      * 'pmos_diode'  — PMOS diode-connected to vdd, drain receives
                        current from `src`
      * 'nmos_diode'  — NMOS diode-connected to vss, drain receives
                        current from `src`
      * 'pmos_mirror' — PMOS mirror of `src` with width multiplier `by_M`
      * 'nmos_mirror' — NMOS mirror of `src` with width multiplier `by_M`

    Field meaning depends on `kind`:
      * external:    `value` set (SI amperes); `src`/`by_M` are None.
      * *_diode:     `src` set (name of upstream bias node);
                     `value`/`by_M` are None.
      * *_mirror:    `src` set, `by_M` set (positive integer);
                     `value` is None.

    Stage templates that accept a `bias_tree` kwarg consume the list to
    materialize their bias chain. Templates that DO NOT accept it
    receive nothing — backward compatible with every pre-G2 stage.
    """
    name: str
    kind: str
    src: Optional[str] = None
    value: Optional[float] = None
    by_M: Optional[int] = None


@dataclass
class CellInstance:
    """One sub-cell instantiation inside an outer cell's topology block
    (v0.0.28, cell composition).

    Surface form:

        instance amp_a = clay.stdlib.opamps::TwoStageMillerPMOS {
            inn = in_a;     // sub-cell port `inn` <- outer net `in_a`
            inp = vbias_a;
            out = out_a;
            vdd = vdd;
            vss = vss;
        };

    Semantics:
      * `name`       — local instance handle ("amp_a"); used to prefix
                       internal nets of this sub-cell during flatten so
                       sibling instances don't collide on shared names
                       like `tail`, `d1`, `d2`, `pbias`.
      * `cell_path`  — `'pkg.path::CellName'` reference to the sub-cell
                       being instantiated. Resolved via the same
                       allow-list as v0.0.11 (`clay.stdlib.*` /
                       `clay.pdk.*`) — see `module.resolve()`.
      * `port_map`   — `{sub_cell_port: outer_net}` map. Every port on
                       the sub-cell MUST be bound explicitly (no
                       positional / implicit fallback). Sub-cell ports
                       not in the map → reject; map keys not in the
                       sub-cell's port list → reject. Empty map is
                       rejected at parse time too (always reject; even
                       a zero-port sub-cell would be uninteresting in
                       this version).
    """
    name: str
    cell_path: str
    port_map: dict[str, str] = field(default_factory=dict)


@dataclass
class CellDef:
    """A cell defined entirely in `.clay` text (v0.0.19).

    Unlike `CellAlias` which parameterizes an existing stdlib cell,
    a `CellDef` carries everything needed to compile a fresh cell:
    port list, topology choice, spec, sizing intent. The compiler
    resolves `topology.style` to a registered stage template, then
    passes the spec + sizing kwargs to materialize a Cell IR.

    .clay surface form:

        cell MyAmp(inn, inp, out, vdd, vss) {
            topology { style = 'two_stage_miller_pmos'; rationale = '...'; }
            spec     { gain_dc : real >= 25; gbw : freq >= 1MHz; }
            sizing   { gm_id_diff = 15; gm_id_load1 = 10; }
        }

    v0.0.22 (G10): ports can carry an optional `: <role>` annotation
    (`input` / `output` / `inout` / `power_p` / `power_n` / `bias` /
    `ground`). The role string is exposed via the `port_roles`
    dict — the bare `ports` list keeps shape `list[str]` for backward
    compatibility with the compiler's port-match check. Ports
    without an explicit role have value `None` in `port_roles`.

    v0.0.26 (G2): optional `bias_tree { ... }` block lets the .clay
    source declare bias nodes explicitly. Stored as `list[BiasNode]`.
    Stage templates that accept the `bias_tree` kwarg consume the
    list to build their bias chain; templates that do not accept it
    ignore it (backward-compatible with all pre-G2 stages).
    """
    local_name: str
    ports: list[str]
    topology: dict[str, object]       # style + rationale + free-form fields
    spec: Spec
    sizing_kwargs: dict[str, float] = field(default_factory=dict)
    # v0.0.22 (G10): port name → role, with role ∈
    # {'input', 'output', 'inout', 'power_p', 'power_n', 'bias',
    # 'ground'} ∪ {None}. Defaults to an all-None map when ports are
    # bare. Informational only — compile_cell_def uses port *names*
    # for the shape check.
    port_roles: dict[str, Optional[str]] = field(default_factory=dict)
    # v0.0.26 (G2): declarative bias chain. Empty list means "no
    # bias_tree block was present" — the stage template uses its
    # built-in bias chain (pre-G2 behavior preserved).
    bias_tree: list[BiasNode] = field(default_factory=list)
    # v0.0.28 (cell composition): list of `instance NAME = pkg::Cell { ... };`
    # declarations inside the topology block. Empty list (default)
    # means the cell is a primitive cell driven by a stage template
    # — i.e. pre-v0.0.28 behavior. A non-empty list flips the
    # compiler into "compose from sub-cells" mode: every sub-cell is
    # resolved, instantiated, its devices net-renamed via the port
    # map, and the union becomes this cell's device list. Stage-style
    # builds (`topology.style = '...'`) and composition (`instances`)
    # are mutually exclusive in a given cell — the compiler raises an
    # error if both are present.
    instances: list[CellInstance] = field(default_factory=list)
    # v0.0.28 (device primitives): list of explicit device-level
    # declarations parsed from `device M1 : pmos { ... }`,
    # `cap C1 { ... }`, and `res R1 { ... }` entries inside the
    # topology block. Empty list (default) means the cell falls back
    # to whichever build path `topology.style` / `instances` selects
    # — i.e. pre-v0.0.28 behavior. A non-empty list flips the
    # compiler into "primitive cell" mode: it builds the `Cell` IR
    # directly from these devices + the bias_tree, skipping any
    # registered stage builder. Mutually exclusive with both
    # `topology.style` and `instances` — the compiler raises if any
    # combination is mixed.
    devices: list[Device] = field(default_factory=list)
    # v0.0.31 (G3 closeout): list of lean role declarations parsed
    # from `pmos_pair NAME { ... }` / `pmos_mirror NAME { ... }` /
    # `pmos_cs NAME { ... }` (and their nmos siblings) inside the
    # topology block. Roles are expanded to concrete `Device` entries
    # in the parser transformer; the expanded devices land in the
    # `devices` list above. This `role_decls` list keeps the role-
    # level intent visible to downstream tooling (and for tests) but
    # the compiler dispatch logic only looks at `devices`. Empty list
    # (default) means no roles were declared — pre-v0.0.31 behavior
    # preserved exactly.
    role_decls: list["RoleDecl"] = field(default_factory=list)
    # v0.0.62 (functional roles): list of functional-role declarations
    # parsed from the six functional keywords (`input_transconductor`,
    # `output_buffer`, `feedforward_path`, `dominant_pole_cap`,
    # `compensation_zero`, `tail_current_source`). These are kept as
    # parsed surface intent — the compiler / lowering layer is
    # responsible for translating them into underlying structural
    # roles or device primitives. Empty list (default) means no
    # functional roles were declared — pre-v0.0.62 behavior preserved.
    functional_roles: list["ParsedFunctionalRole"] = field(default_factory=list)
    # Explicit net-level layout-intent declarations parsed from a
    # sibling `layout_intent <CellName> { ... }` block. These are the
    # designer-authored overrides/supplements for the auto-derived
    # layout intent. Empty list preserves pre-layout-intent-source
    # behavior.
    layout_intents: list["ParsedLayoutIntentDecl"] = field(default_factory=list)
    # Source-level flow-automation + acceptance contract, parsed from
    # sibling top-level `flow <CellName> { ... }` / `quality <CellName>
    # { ... }` blocks. None (default) preserves pre-flow-block behavior.
    # `clay.flow.spec.compile_flow_spec` lowers these into the frozen
    # `CircuitSpec` (the compiled projection); the flow registry is
    # discovered from cells carrying BOTH blocks.
    flow_decl: Optional["ParsedFlowDecl"] = None
    quality_decl: Optional["ParsedQualityDecl"] = None
    # v0.0.92 (combinator-in-source, Round 68): optional combinator
    # block parsed from the cell body. When present, the cell is built
    # by lowering the combinator algebra expression to a `ComposedStage`
    # via `clay.stage_combinators.*` and forwarding the resulting builder
    # to the cell. Mutually exclusive with `topology.style` (stage
    # template), `instances` (composition), and `devices` (primitive) —
    # the compiler raises `CellCompileError` if more than one is
    # supplied. None (default) preserves pre-v0.0.92 behavior.
    combinator_block: Optional["ParsedCombinatorBlock"] = None
    source_span: Optional[dict[str, Any]] = None


@dataclass
class RoleDecl:
    """One LEAN role-typed device declaration (v0.0.31 / G3 closeout).

    Surface form (`.clay` text):

        pmos_pair diff_pair {
            gm_id = 15;
            W = 20u; L = 0.5u;
            gates = (inn, inp);     // optional explicit pin override
            drains = (d1, d2);
            tail = tail;
        }

    Semantics: the role kind (`pmos_pair`, `nmos_pair`, `pmos_mirror`,
    `nmos_mirror`, `pmos_cs`, `nmos_cs`) determines how many concrete
    `Device` IR entries the role expands into, what their device_class
    is, and which auto-named pins they wire to. The role expansion runs
    in the parser transformer (after surface validation) so by the time
    the IR reaches the compiler, every role has already been replaced
    by its concrete Device list inside `CellDef.devices`.

    Field meaning:
      * `name`       — local role handle (`diff_pair`, `first_load`).
                       Used to prefix auto-named pins (`<name>_tail`,
                       `<name>_d1`, etc.) so multiple roles in one
                       cell don't collide on shared default net names.
      * `kind`       — one of the six role keywords (above). Determines
                       expansion arity and connectivity convention.
      * `sizing`     — `{key: value}` map of size fields applied to
                       every device in the expansion. Standard keys
                       `W`, `L`, `M`, `nf` plus the optional `gm_id`
                       intent (forwarded to every contained device's
                       sizing dict — sizing pass picks it up just like
                       it would on a stage-built cell).
      * `pin_overrides` — `{pin_label: net_name}` map of single-net
                       pin overrides. Pin labels are role-specific:
                       `tail` (pair), `in`/`out` (mirror),
                       `gate`/`drain` (cs).
      * `pair_overrides` — `{pin_label: (net_a, net_b)}` map of
                       two-net pin overrides. Pin labels are
                       `gates`/`drains` for pair roles.

    The IR keeps the role-level intent visible AFTER expansion (via
    `CellDef.role_decls`) so tooling that wants to read "this cell has
    a differential pair" can see it without re-deriving from the device
    list. The expanded `Device` entries also carry a `role_origin`
    string in their sizing dict (`role_origin: <role_name>`) so the
    sizing pass / inspector can trace each device back to its role.
    """
    name: str
    kind: str
    sizing: dict[str, float] = field(default_factory=dict)
    pin_overrides: dict[str, str] = field(default_factory=dict)
    pair_overrides: dict[str, tuple[str, str]] = field(default_factory=dict)


@dataclass
class ParsedFunctionalRole:
    """One parsed functional-role declaration (v0.0.62 / functional layer).

    Surface form (`.clay` text):

        input_transconductor Mdp {
            gm_target = 2mA/V,
            intent = 'differential_drive',
            polarity = pmos
        }

        dominant_pole_cap Cc {
            pole_freq_target = 1MHz,
            at_node = stage2_out
        }

        feedforward_path Pff1 {
            from = stage1,
            to = stage3,
            gain = 0.5
        }

    Semantics: the functional-role keyword names a behavioral noun
    rather than a structural one. Where `role_decl` (v0.0.31) lets the
    LLM write at the transistor-bundle level (`pmos_pair`, `nmos_mirror`),
    `functional_role_decl` lets the LLM write at the analog-function
    level (input transconductor, output buffer, compensation zero, ...).
    The lowering from functional roles to underlying structural roles
    or device primitives is the responsibility of the compiler layer
    — this dataclass only carries the parsed surface intent.

    Field meaning:
      * `name`       — local role handle (`Mdp`, `Cc`, `Pff1`). Used by
                       downstream tooling to refer back to the role.
      * `kind`       — one of the six functional-role keywords:
                       `input_transconductor`, `output_buffer`,
                       `feedforward_path`, `dominant_pole_cap`,
                       `compensation_zero`, `tail_current_source`.
      * `attributes` — `{key: value}` map of attribute fields. Values
                       are one of:
                         - `float`   (number with optional unit suffix)
                         - `str`     (quoted string literal)
                         - `str`     (bare identifier — for `polarity =
                                      pmos`, `at_node = stage2_out`)
                         - `list[str]` (parenthesized identifier list —
                                      for `from = (a, b)`-style values)
                       The parser does NOT distinguish bare identifiers
                       from quoted strings at the dataclass level; both
                       store as `str`. Downstream consumers infer the
                       distinction from the attribute key (e.g.
                       `at_node` is always a net name, `intent` is a
                       free-form tag).
    """
    kind: str
    name: str
    attributes: dict[str, Any] = field(default_factory=dict)
    source_span: Optional[dict[str, Any]] = None


@dataclass
class ParsedLayoutIntentDecl:
    """One parsed net-level layout-intent declaration.

    Surface forms live in a top-level `layout_intent <CellName> { ... }`
    block:

        @match(net0, net1);
        @symmetric(crossn, crossp);
        @shield(clk -> crossn, gnd);
        @budget(clk_regen, crossn, crossp, 0.10);
        @mirror(outp, outn);
        @equal_well_edge(mp_a, mp_b, 0.10);
        @equal_od(mn_a, mn_b);
        @equal_od_spacing(mp_a, mp_b, 0.15);

    The parser records the source form without deciding merge policy.
    `clay.layout.intent` lowers this typed IR into the stable
    `LayoutIntent` dataclasses and merges it with automatic derivation.

    The three `@equal_*` device-environment (LDE) forms lower to a single
    `kind="lde_budget"` declaration carrying a `metric` in
    `{well_edge|od_length|od_neighbor_spacing}` (set on `metric`); `nets`
    names the two DEVICE labels whose layout-dependent-effect metric must
    match and `rel_tol` is the relative tolerance (default applied at
    lowering when omitted). A parser may equivalently emit
    `kind="equal_well_edge"` / `"equal_od"` / `"equal_od_spacing"` (the
    surface keyword); `clay.layout.intent` maps either form to the same
    obligation.
    """
    kind: str
    nets: tuple[str, str] = ()
    budget_kind: Optional[str] = None
    rel_tol: Optional[float] = None
    # Device-environment (LDE) metric selector for the `@equal_*` forms:
    # one of "well_edge" | "od_length" | "od_neighbor_spacing". None for every
    # non-LDE declaration kind. Lowered in clay.layout.intent.
    metric: Optional[str] = None
    aggressor: Optional[str] = None
    victim: Optional[str] = None
    shield_net: str = "gnd"
    target: Optional[str] = None
    goal: Optional[str] = None
    limit_kind: Optional[str] = None
    limit_value: Optional[float] = None
    group: Optional[str] = None
    devices: tuple[str, ...] = ()
    row_min: Optional[int] = None
    row_max: Optional[int] = None
    priority: Optional[str] = None
    attributes: dict[str, Any] = field(default_factory=dict)
    source_span: Optional[dict[str, Any]] = None


@dataclass
class ParsedFlowDecl:
    """One parsed `flow <CellName> { ... }` block.

    Surface form lives in a top-level block sibling to `layout_intent`:

        flow folded_cascode_ota {
          pdk "gf180mcu";
          corners (tt);
          steps (compile, layout, drc, lvs, pex, post, intent, sign);
          layout = circuits.fc_ota.fc_ota_layout;
        }

    The parser records the surface form; `clay.flow.spec.compile_flow_spec`
    lowers it (with case.yaml path inference) into the frozen `CircuitSpec`.
    `corners`/`steps` are stored in source order; `layout_module` is the
    dotted generator-module binding.
    """
    pdk: Optional[str] = None
    corners: tuple[str, ...] = ()
    steps: tuple[str, ...] = ()
    layout_module: Optional[str] = None
    # v2-lang: the name from `flow { policy = NAME; }`. Resolved against
    # the file-level `ClayFile.policies` registry by the layout engine.
    policy_ref: Optional[str] = None
    source_span: Optional[dict[str, Any]] = None


@dataclass
class ParsedPolicyDecl:
    """One parsed `policy NAME { ... }` block (v2-lang).

    A reusable, file-level place/route policy: the GENERAL place/route
    strategy + convergence defaults a cell can select via
    `flow { policy = NAME; }`. Surface form:

        policy analog_default {
          @place_policy(symmetry = mirror_x, pack = rows, area = minimize);
          @route_policy(layers = (M1_riser, M2_trunk), tracks = net_disjoint, shield = on_request);
          @converge(place_iters = 6, route_iters = 4, tol = 0.10);
          @learn(on);
        }

    `place_policy` / `route_policy` / `converge` are stored as plain dicts
    (tuple values for parenthesised id-lists, e.g. `layers`). `converge`
    keys are coerced to numbers (place_iters/route_iters -> int, tol ->
    float). `learn` is a bool from `@learn(on|off)`. Honest scope: only
    `converge` is actively consumed downstream today; the strategy maps are
    recorded provenance. Registered in `ClayFile.policies` keyed by `name`
    (NOT per-cell).
    """
    name: str
    place_policy: dict[str, Any] = field(default_factory=dict)
    route_policy: dict[str, Any] = field(default_factory=dict)
    converge: dict[str, Any] = field(default_factory=dict)
    learn: bool = False
    source_span: Optional[dict[str, Any]] = None


@dataclass
class ParsedQualityDecl:
    """One parsed `quality <CellName> { ... }` block.

    Surface form (the acceptance contract / "type" the design satisfies):

        quality folded_cascode_ota {
          @kpi(dc_gain_db, ">=", 80, unit="dB");
          @consistency(dc_gain_db, rel <= 0.05);
          @require(drc_clean);
        }

    `kpis` is a list of `(key, op, value, unit)` where `op` is one of
    ">=", "<=", "near". `consistencies` is a list of `(key, rel_tol)`
    (RELATIVE thresholds only). `requires` is a list of gate-kind names.
    `clay.flow.spec.compile_flow_spec` lowers these into `KpiSpec` /
    `ConsistencySpec` / `GateSpec` on the frozen `CircuitSpec`.
    """
    kpis: list[tuple[str, str, float, Optional[str]]] = field(default_factory=list)
    consistencies: list[tuple[str, float]] = field(default_factory=list)
    requires: list[str] = field(default_factory=list)
    source_span: Optional[dict[str, Any]] = None


@dataclass
class CellAlias:
    """A locally-named cell that aliases / parameterizes a stdlib cell.

    v0.0.17: lets `.clay` source declare a local name + spec/sizing
    overrides without an inline verify. Subsequent `verify NAME;`
    statements resolve via this table.

    Example .clay:
        cell MyAmp = clay.stdlib.opamps::TwoStageMillerPMOS {
          spec { gain_dc : real >= 25; }
          sizing_override { gm_id_diff = 15; }
        };
        verify MyAmp;
    """
    local_name: str
    parent_path: str                              # 'pkg.path::CellName'
    spec: Spec
    sizing_overrides: dict[str, float] = field(default_factory=dict)


@dataclass
class FunctionDef:
    """A user-defined function declared at file scope (v0.0.37).

    Surface form (.clay text):

        fn compute_W(gm_id, I_d) {
          return gm_id * I_d / 250e-6;
        }

    Semantics:
      * Pure function: no side effects, no I/O.
      * Args and return value are scalar floats (Python `float`).
      * Body is a single `return EXPR;` (no local variables, no
        control flow in this MVP).
      * Expressions support `+ - * / **`, parentheses, number
        literals with units, argument references, and calls into
        other functions DEFINED EARLIER in the same file.

    The parser stores the body as an opaque AST node (`body_expr`) —
    a recursive `dict` with keys `kind` and (depending on kind)
    `args`, `op`, `name`, `value`. The evaluator (`_eval_expr` in
    `clay.parser`) walks this AST to produce a float given a
    `{argname: value}` environment and the file's accumulated
    function table.

    Scope: file-scope only — no function-in-function nesting. A
    function can call any function defined earlier in the same .clay
    file (or in an imported file processed before this one). Forward
    references and recursive self-calls are rejected at parse time.
    """
    name: str
    args: list[str]
    # AST node — small dict-shaped tree. See `clay.parser._eval_expr`
    # for the supported kinds and shapes. Kept as `Any` so the IR
    # dataclass surface doesn't grow a public AST type before the
    # expression sublanguage stabilizes.
    body_expr: Any


@dataclass
class ParsedCombinatorBlock:
    """A `combinator { ... }` block declared inside a cell body (v0.0.92).

    Surface form (`.clay` text):

        cell three_stage(in, out, vdd, vss) {
          combinator three_stage {
            stages = [common_source_pmos_input, common_source_pmos_input,
                      common_source_pmos_input];
            build = stages[0] >> stages[1] >> stages[2];
          }
          spec { gain_dc : real >= 60; }
        }

    Semantics: the block lets the LLM write combinator algebra
    (`>>`, `feedforward(...)`, `compensate(...)`, `mirror_match(...)`,
    `cascade_chain([...])`) directly in `.clay` source instead of
    needing to generate Python pseudo-code first. At compile time
    (in `compile_cell_def`) the block lowers to a `ComposedStage` via
    the Python combinator API (`clay.stage_combinators.*`) and the
    resulting builder replaces the stage-template builder.

    Field meaning:
      * `name`        — local block handle (`three_stage_combo`,
                        `nmcf_combo`). Used for diagnostics only.
      * `stages`      — declared stage template names referenced by
                        `stages[0]` / `stages[1]` / etc. inside `build`.
                        Each must be a registered StageInterface
                        template_name (validated at compile time).
      * `build_expr`  — AST node for the build expression. The AST is
                        kept as nested dicts (`{kind: ..., children: [...]}`)
                        so the parser doesn't grow a public AST type
                        before the combinator surface stabilizes.
                        Supported `kind` values:
                          * `'stage_ref'`   — `stages[i]` reference;
                                              `index: int`
                          * `'stage_name'`  — bare template name;
                                              `name: str`
                          * `'cascade'`     — `a >> b`; `left`/`right` AST
                          * `'feedforward'` — `feedforward(src, sink, gain=k)`;
                                              `source`/`sink` AST,
                                              `kwargs: dict[str, float]`
                          * `'compensate'`  — `compensate(stage, miller=Cc, ...)`;
                                              `inner` AST,
                                              `kwargs: dict[str, float]`
                          * `'mirror_match'`— `mirror_match(a, b)`;
                                              `left`/`right` AST
                          * `'cascade_chain'`— `cascade_chain([s1, s2, ...])`;
                                              `stages: list[AST]`

    Lowering details (handled in `clay.compiler._lower_combinator_block`):
      1. Walk the AST, resolving every `stage_ref` / `stage_name` to a
         registered `StageInterface` via
         `clay.stage_interface.get_stage_interface`.
      2. Apply combinator operators bottom-up:
         `cascade` → `clay.stage_combinators.cascade`,
         `feedforward` → `clay.stage_combinators.feedforward`,
         `compensate` → `clay.stage_combinators.compensate`,
         `mirror_match` → `clay.stage_combinators.mirror_match`,
         `cascade_chain` → `clay.stage_combinators.cascade_chain`.
      3. The resulting `ComposedStage` is lowered via
         `clay.stage_combinators.lower_to_template` to a builder, and
         the cell's spec + sizing kwargs are forwarded verbatim.
    """
    name: str
    stages: list[str]
    build_expr: Any   # nested dict AST; see docstring


@dataclass
class ClayFile:
    package_name: Optional[str]                   # from `package X;`
    pdk_id: Optional[str]                         # from `pdk 'sky130';`
    imports: list[str] = field(default_factory=list)
    verify_jobs: list[VerifyJob] = field(default_factory=list)
    cell_aliases: list[CellAlias] = field(default_factory=list)
    cell_defs: list[CellDef] = field(default_factory=list)
    # v0.0.37 (user functions): file-scope function definitions. Each
    # entry is one `fn NAME(args) { return EXPR; }` declaration in
    # source order. The list is consulted at parse time to resolve
    # function calls inside `sizing { ... }` blocks; downstream
    # consumers (compiler, drivers, verifier) never see this list
    # because the resolution happens before the IR leaves the parser.
    functions: list[FunctionDef] = field(default_factory=list)
    # v2-lang: file-level registry of `policy NAME { ... }` blocks, keyed
    # by policy name. Policies are NOT per-cell; a cell selects one via
    # `flow { policy = NAME; }` (recorded on `ParsedFlowDecl.policy_ref`).
    policies: dict[str, "ParsedPolicyDecl"] = field(default_factory=dict)
