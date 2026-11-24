"""
Stage typed interface — P2 first slice (v0.0.86).

Provides the explicit `StageInterface` typed surface that algebraic
spatial combinators (`stage_a >> stage_b`, `feedforward(stage_a, stage_b)`)
will need at composition time. Combinator runtime is NOT in this slice;
this module ships only the typed declaration vocabulary and a
type-check stub function whose error format is forward-fixed so the
combinator step can rely on it without further surface churn.

Vocabulary (one dataclass per concept; flat, no inheritance):

  * `PortRole`     — one of {`signal_in`, `signal_out`, `bias`, `supply`}.
                     Drives which list a port lands in.
  * `PortPolarity` — `differential_pos` / `differential_neg` /
                     `single_ended` / `none` (for supplies / unipolar bias).
                     A combinator that composes two stages compares
                     output_ports[*].polarity against next stage's
                     input_ports[*].polarity to catch direction errors
                     (e.g. piping a single-ended stage into a diff-input
                     stage without an explicit splitter).
  * `PortDecl`     — `(name, role, polarity)`.
  * `SpecTarget`   — `(kpi, dimension, op, target_value)` — what KPI
                     the template's builder PROMISES it can hit when its
                     gm/Id intent kwargs are routed sanely. Used by the
                     type-check stub to surface "this template cannot
                     deliver gain_dc >= 80 dB on a 1-stage cell" before
                     the combinator wires up a sim.
  * `StageInterface` — the full surface declaration for one
                       stage_template.

This module is intentionally Python-only. The Rust mirror lands in a
later round when combinator runtime is wired into `clay_parser` — at
that point a `PyStageInterface` PyO3 wrapper falls out naturally.
For now the type-check stub is pure Python and runs in `clay.compiler`
ahead of any builder call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

# ─── Vocabulary ─────────────────────────────────────────────

# Port role buckets. The four legal values are mutually exclusive — a
# given port lands in exactly one of {input_ports, output_ports,
# bias_ports, supply_ports} on a `StageInterface`. The split exists so
# combinator type-check can reject "wire stage_a.vdd into stage_b.inn"
# at the surface level rather than at netlist flatten.
PortRole = Literal["signal_in", "signal_out", "bias", "supply"]

# Port polarity for combinator alignment checks. A combinator that
# composes `stage_a >> stage_b` requires that the multiset of
# `stage_a.output_ports[*].polarity` matches the multiset of
# `stage_b.input_ports[*].polarity` — otherwise the composition is
# structurally invalid (you'd be wiring a single-ended output into a
# differential input, etc.).
#
# `none` is used for supplies (vdd/vss) and unipolar bias rails — they
# don't participate in differential-vs-single-ended checks.
PortPolarity = Literal[
    "differential_pos",      # e.g. `inp` on a diff-input stage
    "differential_neg",      # e.g. `inn` on a diff-input stage
    "single_ended",          # e.g. `out` on a 1-stage CS amplifier
    "none",                  # supplies, unipolar bias rails
]


@dataclass(frozen=True)
class PortDecl:
    """One typed port declaration on a stage_template's surface.

    `name`     — the .clay-surface port name (`inn`, `inp`, `out`,
                 `vdd`, `vss`, etc.). Must match the string emitted by
                 the builder in `Cell.ports`.
    `role`     — one of {`signal_in`, `signal_out`, `bias`, `supply`}.
                 Determines which list the port lives in.
    `polarity` — see `PortPolarity` above. For supplies and unipolar
                 bias the default `none` is correct.
    """
    name: str
    role: PortRole
    polarity: PortPolarity = "none"


@dataclass(frozen=True)
class SpecTarget:
    """One KPI promise from a stage_template's builder.

    A `SpecTarget` is a hint, not a contract — the builder still has
    to produce a circuit that meets the target under sim. The combinator
    type-check uses these to triage obviously-wrong asks early (e.g.
    asking a 1-stage CS amplifier for `gain_dc >= 80 dB`).

    Field meaning matches the SpecField surface in `clay.ir`:
      * `kpi`          — measure name (`gain_dc`, `gbw`, `pm`,
                         `phase_at_gbw`, ...).
      * `dimension`    — `real` / `freq` / `volt` / `db` / `angle` / etc.
      * `op`           — `>=` / `<=` / `=` / `>` / `<`.
      * `target_value` — SI-unit number, same convention as SpecField.
    """
    kpi: str
    dimension: str
    op: str
    target_value: float


@dataclass(frozen=True)
class StageInterface:
    """Typed surface of one stage_template.

    Combinator slice (later round) will compose two StageInterfaces by
    inspecting their port lists, polarities, and KPI promises. This
    slice ships the declaration vocabulary plus a type-check stub
    (`check_compose`) so the wire-format is forward-fixed.

    Invariants enforced by `__post_init__`:
      * `template_name` non-empty.
      * Port names unique across all four port lists (a port can't
        be both `signal_in` and `signal_out`).
      * Every `PortDecl.role` matches the list it lives in.
      * `vdd` / `vss` MUST be in `supply_ports` if present (catches the
        common error of registering them as `signal_in`).
    """
    template_name: str
    input_ports: List[PortDecl] = field(default_factory=list)
    output_ports: List[PortDecl] = field(default_factory=list)
    bias_ports: List[PortDecl] = field(default_factory=list)
    supply_ports: List[PortDecl] = field(default_factory=list)
    spec_targets: Dict[str, SpecTarget] = field(default_factory=dict)
    compatible_compensation: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.template_name:
            raise ValueError("StageInterface.template_name must be non-empty")

        # role consistency
        for p in self.input_ports:
            if p.role != "signal_in":
                raise ValueError(
                    f"port {p.name!r} on {self.template_name!r} is in "
                    f"input_ports but role={p.role!r} (expected 'signal_in')"
                )
        for p in self.output_ports:
            if p.role != "signal_out":
                raise ValueError(
                    f"port {p.name!r} on {self.template_name!r} is in "
                    f"output_ports but role={p.role!r} (expected 'signal_out')"
                )
        for p in self.bias_ports:
            if p.role != "bias":
                raise ValueError(
                    f"port {p.name!r} on {self.template_name!r} is in "
                    f"bias_ports but role={p.role!r} (expected 'bias')"
                )
        for p in self.supply_ports:
            if p.role != "supply":
                raise ValueError(
                    f"port {p.name!r} on {self.template_name!r} is in "
                    f"supply_ports but role={p.role!r} (expected 'supply')"
                )

        # uniqueness across all four lists
        all_names: List[str] = (
            [p.name for p in self.input_ports]
            + [p.name for p in self.output_ports]
            + [p.name for p in self.bias_ports]
            + [p.name for p in self.supply_ports]
        )
        seen: set[str] = set()
        for n in all_names:
            if n in seen:
                raise ValueError(
                    f"port {n!r} appears twice in {self.template_name!r}; "
                    "every port must live in exactly one of "
                    "{input,output,bias,supply}_ports"
                )
            seen.add(n)

        # canonical supply guard: vdd / vss MUST be supplies if present
        supply_names = {p.name for p in self.supply_ports}
        for canonical in ("vdd", "vss"):
            if canonical in all_names and canonical not in supply_names:
                raise ValueError(
                    f"port {canonical!r} on {self.template_name!r} must be "
                    "declared in supply_ports (not input/output/bias)"
                )

    @property
    def all_port_names(self) -> List[str]:
        """Every declared port name, preserving canonical order
        (input → output → bias → supply)."""
        return (
            [p.name for p in self.input_ports]
            + [p.name for p in self.output_ports]
            + [p.name for p in self.bias_ports]
            + [p.name for p in self.supply_ports]
        )

    def __rshift__(self, other: Any) -> "ComposedStage":
        """`self >> other` — cascade combinator entry point.

        Delegates to `clay.stage_combinators.cascade`. Allows
        `stage_a >> stage_b` syntax directly on `StageInterface`
        instances without first wrapping into `ComposedStage`.
        """
        from .stage_combinators import cascade  # noqa: WPS433
        return cascade(self, other)


# ─── Registry ───────────────────────────────────────────────

_INTERFACE_REGISTRY: Dict[str, StageInterface] = {}


def register_stage_interface(iface: StageInterface) -> None:
    """Register a StageInterface under its `template_name`.

    Idempotent re-registration of the EXACT same StageInterface is OK.
    Conflicting interfaces for the same template_name raise — caller
    must remove the old registration first if they really mean to
    replace it.
    """
    existing = _INTERFACE_REGISTRY.get(iface.template_name)
    if existing is not None:
        if existing == iface:
            return
        raise RuntimeError(
            f"stage_interface for {iface.template_name!r} already "
            f"registered with a different declaration; refusing to "
            "overwrite"
        )
    _INTERFACE_REGISTRY[iface.template_name] = iface


def get_stage_interface(template_name: str) -> StageInterface:
    """Return the StageInterface for a registered template; raise on miss.

    Note: this is keyed on the same `style` string used by
    `clay.stage_templates.register_stage`. A template that has a builder
    but NO interface declaration will raise here — combinator code is
    expected to use `has_stage_interface` first if it wants to degrade
    gracefully for legacy templates.
    """
    if template_name not in _INTERFACE_REGISTRY:
        raise KeyError(
            f"no StageInterface registered for {template_name!r}. "
            f"Registered: {sorted(_INTERFACE_REGISTRY)}. "
            f"Register via clay.stage_interface.register_stage_interface()."
        )
    return _INTERFACE_REGISTRY[template_name]


def has_stage_interface(template_name: str) -> bool:
    """True if a StageInterface is registered for `template_name`."""
    return template_name in _INTERFACE_REGISTRY


def list_stage_interfaces() -> List[str]:
    """Sorted list of every template that has a StageInterface declared."""
    return sorted(_INTERFACE_REGISTRY)


# ─── Type-check stub (forward-fixes the error format for combinator) ──

@dataclass(frozen=True)
class ComposeError:
    """One reason a `stage_a >> stage_b` composition is structurally invalid.

    The combinator slice (later round) raises `ComposeTypeError` carrying
    a list of these; this slice's stub produces them so test fixtures
    can pin the exact message format ahead of time.

    `kind` is a stable identifier the combinator runtime will match on:
      * `port_count_mismatch`     — stage_a outputs n, stage_b inputs m, n != m
      * `polarity_mismatch`       — multisets differ
      * `bias_unsatisfied`        — stage_b has a bias_port that stage_a
                                    cannot supply (placeholder for now)
      * `incompatible_compensation` — explicit reject; reserved.
    """
    kind: str
    message: str


def check_compose(
    stage_a: StageInterface,
    stage_b: StageInterface,
) -> List[ComposeError]:
    """Type-check whether `stage_a >> stage_b` would be structurally valid.

    Returns the list of ComposeErrors describing every structural
    incompatibility; empty list means the composition is valid at the
    type level (the combinator runtime, when it lands, MAY still reject
    on KPI promise grounds).

    Implementation is intentionally minimal — combinator runtime in a
    later round will extend this with bias-port matching, fanout rules,
    and KPI compatibility. This stub fixes the error message wire
    format so test fixtures can pin it now.

    Cases handled in this slice:
      1. `port_count_mismatch` — len(stage_a.output_ports) !=
         len(stage_b.input_ports).
      2. `polarity_mismatch` — sorted polarity multisets differ.
    """
    errors: List[ComposeError] = []

    out_ports = stage_a.output_ports
    in_ports = stage_b.input_ports

    if len(out_ports) != len(in_ports):
        errors.append(ComposeError(
            kind="port_count_mismatch",
            message=(
                f"cannot compose {stage_a.template_name!r} >> "
                f"{stage_b.template_name!r}: {stage_a.template_name!r} "
                f"emits {len(out_ports)} output port(s) "
                f"({[p.name for p in out_ports]}) but "
                f"{stage_b.template_name!r} expects "
                f"{len(in_ports)} input port(s) "
                f"({[p.name for p in in_ports]})"
            ),
        ))
        # Polarity check is meaningless when counts differ — bail.
        return errors

    out_pol = sorted(p.polarity for p in out_ports)
    in_pol = sorted(p.polarity for p in in_ports)
    if out_pol != in_pol:
        errors.append(ComposeError(
            kind="polarity_mismatch",
            message=(
                f"cannot compose {stage_a.template_name!r} >> "
                f"{stage_b.template_name!r}: output polarity "
                f"{out_pol} does not match input polarity {in_pol}"
            ),
        ))

    return errors


# ─── Combinator runtime (v0.0.88, P2 second slice) ──────────────────
#
# `ComposedStage` is the algebraic surface a combinator expression
# returns. It is itself a `StageInterface` (so cascades chain naturally
# — `(a >> b) >> c` is well-typed), plus a `kind` tag and a `parts`
# tuple recording the constituent stages or sub-compositions.
#
# Six `kind` values cover the surface:
#   * `'leaf'`       — wraps a real stage_template; `parts` is the
#                      template_name (str). Created via
#                      `clay.stage_combinators.stage(name)` or
#                      automatically when a bare `StageInterface` enters
#                      a combinator expression.
#   * `'cascade'`    — produced by `stage_a >> stage_b`. `parts` is a
#                      `(left, right)` tuple of `ComposedStage`s.
#   * `'feedforward'`— produced by `feedforward(stage_a, stage_b, gain)`.
#                      `parts` is `(source, sink)`; `meta['gain']`
#                      carries the FF gain.
#   * `'compensate'` — produced by `compensate(stage, miller=Cc,
#                      zero=Rz)`. `parts` is a single-element tuple
#                      `(inner,)`; `meta` carries the compensation
#                      parameters.
#   * `'mirror'`     — produced by `mirror_match(stage_a, stage_b)`.
#                      `parts` is `(a, b)`. `meta` reserved.
#   * `'decouple'`   — produced by `decouple(stage, decoupling_cap=Cd,
#                      isolation_device_W=W, target_psrr_db=PSRR)`.
#                      `parts` is `(inner,)`; `meta` carries the
#                      decoupling cap value, pass-device W, and (for
#                      auto-sizing) the PSRR target with derived values
#                      under `derived_cap` / `derived_W` /
#                      `auto_psrr_trace`. v0.0.95 R72 D.
#
# `meta` is a frozen, flat dict of small scalar parameters (no nested
# structures). It is intentionally NOT exposed via PyO3 — keep it Python
# side until the combinator surface stabilises.


@dataclass(frozen=True)
class ComposedStage:
    """Algebraic surface of a combinator expression.

    A `ComposedStage` IS-A `StageInterface` (it carries the same
    declared surface fields so it can chain with `>>` further). The
    additional `kind` / `parts` / `meta` describe HOW it was composed,
    so `lower_to_template` can walk the tree and dispatch to the right
    netlist composition routine.

    Construction by combinator helpers — users should not instantiate
    `ComposedStage` directly. The combinator helpers
    (`>>`, `feedforward`, `compensate`, `mirror_match`) live in
    `clay.stage_combinators`.

    Invariants:
      * `kind` is one of `{'leaf', 'cascade', 'feedforward',
        'compensate', 'mirror', 'decouple', 'sample_and_hold'}`.
      * `parts` is non-empty for non-leaf kinds.
      * `interface` is the canonical typed surface for the composition —
        downstream combinator chains and `check_compose` operate on it.
    """
    kind: Literal[
        "leaf", "cascade", "feedforward", "compensate", "mirror", "decouple",
        "sample_and_hold",
    ]
    interface: "StageInterface"
    parts: Tuple[Any, ...] = ()
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        legal = {
            "leaf", "cascade", "feedforward", "compensate", "mirror",
            "decouple", "sample_and_hold",
        }
        if self.kind not in legal:
            raise ValueError(
                f"ComposedStage.kind={self.kind!r} not in {sorted(legal)}"
            )
        if self.kind != "leaf" and len(self.parts) == 0:
            raise ValueError(
                f"ComposedStage(kind={self.kind!r}) must carry non-empty "
                "parts; got ()"
            )

    @property
    def template_name(self) -> str:
        """The composed surface's template_name (delegates to interface)."""
        return self.interface.template_name

    @property
    def input_ports(self) -> List[PortDecl]:
        return self.interface.input_ports

    @property
    def output_ports(self) -> List[PortDecl]:
        return self.interface.output_ports

    @property
    def bias_ports(self) -> List[PortDecl]:
        return self.interface.bias_ports

    @property
    def supply_ports(self) -> List[PortDecl]:
        return self.interface.supply_ports

    @property
    def spec_targets(self) -> Dict[str, SpecTarget]:
        return self.interface.spec_targets

    @property
    def compatible_compensation(self) -> List[str]:
        return self.interface.compatible_compensation

    @property
    def all_port_names(self) -> List[str]:
        return self.interface.all_port_names

    def __rshift__(self, other: Any) -> "ComposedStage":  # type: ignore[override]
        """`self >> other` — cascade combinator.

        Resolves to `clay.stage_combinators.cascade(self, other)`. The
        helper lives in the combinators module to avoid a hard import
        cycle (this file is imported by stage_templates very early
        during package init).
        """
        # Lazy import — combinators module imports from this file.
        from .stage_combinators import cascade  # noqa: WPS433
        return cascade(self, other)


class ComposeTypeError(TypeError):
    """Raised when a combinator expression fails `check_compose`.

    Carries the structured `ComposeError` list verbatim so call sites
    can match on `kind` rather than parsing the message.
    """

    def __init__(self, errors: List[ComposeError]) -> None:
        self.errors = list(errors)
        if not errors:
            super().__init__("ComposeTypeError raised with no errors (bug)")
        else:
            joined = "; ".join(e.message for e in errors)
            super().__init__(joined)


def _interface_to_composed_leaf(iface: StageInterface) -> ComposedStage:
    """Wrap a bare `StageInterface` as a `kind='leaf'` ComposedStage.

    Combinator entry points accept either a registered template name
    (str), a `StageInterface`, or an existing `ComposedStage`; this
    helper normalises the second case. The leaf's `parts` carries the
    original template_name (str) so the lowering pass can find the
    builder.
    """
    return ComposedStage(
        kind="leaf",
        interface=iface,
        parts=(iface.template_name,),
        meta={},
    )
