"""
Clay refinement types for spec contracts (v0.1.02 / R79 Agent C / BL-086).

Per the LLM-extrapolation philosophy audit (Tier 0, §10 minimum
four-piece kit), this module lifts the existing ``SpecField`` runtime
check to a *refinement type* — i.e. a type-level predicate that can be
checked DECLARATIVELY at elaboration time (before any simulator runs)
and EMPIRICALLY at verify time (after simulator emits measures).

Refinement types in EDALLM
==========================

A refinement type is the pair *(base type, predicate)* where the
predicate carves out a subset of the base type's inhabitants. We pick
the dialect closest to Flux (PLDI 2023) / Liquid Haskell:

  type GainDc      = { x : f64 | x >= 60.0 }
  type PhaseMargin = { x : f64 | x >= 60.0 }     -- degrees
  type Noise       = { x : f64 | x <= 5e-9 }     -- V/√Hz

A ``SpecContract`` is a list of refinements over the measure namespace;
"`cell C` satisfies `SpecContract S`" iff every refinement in S is
satisfied by the measurement of its corresponding metric.

Why a separate file (rather than extending ``SpecField``)
---------------------------------------------------------

The existing ``SpecField`` dataclass has a stable byte-exact contract
used by the legacy ``verify()`` path, the multi-setup verifier, the
Rust IR bridge, and the LLM authoring tools. Refinement types are an
ADDITIVE layer on top — every refinement carries the same operator /
target / dimension fields as ``SpecField`` but adds the ELABORATION-TIME
inhabitability check, the LLM-facing negative-signal rendering, and the
compose-with-distance-to-sat plumbing.

Importing this module does NOT alter ``clay.verifier.verify`` — that
entry point keeps byte-exact behavior. A new ``verify_with_refinement``
entry point is added to ``clay.verifier`` (re-exports the implementation
from this module).

Three concrete refinement types ship in this round
--------------------------------------------------

* ``RefinedGain`` — gain in dB. Predicate is ``gain_dc >= min_db``
  by default; supports ``<=`` for "loop transfer gain must be bounded"
  ceilings and an ``in_range`` band for tightly toleranced gains.
* ``RefinedPM`` — phase margin in degrees. Predicate is
  ``pm >= min_deg`` by default; supports an ``in_range`` band for
  "PM between 60° and 80° to avoid both ringing and slow settling".
* ``RefinedNoise`` — input-referred noise in V/√Hz. Predicate is
  ``noise <= max_nv_per_sqrt_hz`` (a ceiling).

Each one is a thin subclass of ``Refinement`` that:

  * Knows its canonical metric name (``"gain_dc"`` / ``"pm"`` /
    ``"noise"``) for the measure-namespace lookup.
  * Knows its SI unit and how to format the negative signal text.
  * Implements ``check_inhabitable(other_refinements_in_contract)``
    so the elaboration-stage check can detect contradictions like
    ``gain_dc >= 60 AND gain_dc <= 40``.

Static inhabitability check (the type-level signal)
---------------------------------------------------

Before any simulator runs we walk the SpecContract and run two passes:

  1. Per-refinement self-check: is the predicate by itself satisfiable?
     A single ``>= 60`` is trivially satisfiable; a refinement that
     carries ``in_range`` with ``lower > upper`` is degenerate.
  2. Cross-refinement contradiction check: do any two refinements on
     the SAME metric impose mutually unsatisfiable predicates?

     - ``gain_dc >= 60 AND gain_dc <= 40``    → contradiction
     - ``gain_dc >= 60 AND gain_dc >= 80``    → 80 is the binding floor
     - ``gain_dc <= 40 AND gain_dc <= 30``    → 30 is the binding ceiling
     - ``gain_dc >= 60 AND gain_dc in [40,80]`` → tightens to [60, 80]
     - ``gain_dc >= 60 AND gain_dc = 50``      → contradiction

When step 2 reports a contradiction, the contract is *not inhabitable*
— no circuit can satisfy it — and ``validate_refinement_inhabitable``
returns ``inhabitable=False`` with a structured ``contradiction``
record. The caller (typically ``verify_with_refinement``) refuses to
simulate and surfaces the contradiction to the LLM directly. This is
the Type-Constrained Code Generation 2025 "compilation error halved"
signal: the LLM sees a compile-time rejection rather than a 30-second
ngspice round trip producing a meaningless number.

Compatibility with R78 A distance-to-sat
----------------------------------------

Refinement types and distance-to-sat are complementary signals:

  * Refinement types answer "what is the contract" at the TYPE level —
    a static predicate the LLM can read AT GENERATION TIME.
  * Distance-to-sat answers "how far from satisfied" at the VALUE level
    — a continuous gradient the LLM (or its RL trainer) can read AT
    TRAINING TIME.

This module reuses ``clay.distance_to_sat.compute_spec_distance`` and
``evaluate_spec_distance`` for the empirical check (no duplication of
the distance kernel). The contract is:

  * Every refinement is materialized as a ``SpecField`` (for the
    distance kernel) AND carries the refinement-level metadata.
  * When the contract is inhabitable AND every measure satisfies its
    refinement, the distance kernel reports ``distance=0`` for that
    metric — i.e. the two signals agree on satisfied vs unsatisfied.
  * When the contract is NOT inhabitable, the distance kernel never
    runs; the contradiction signal is surfaced directly.

API surface (additive — every existing path stays byte-exact)
-------------------------------------------------------------

  * ``Refinement``                        — abstract base class
  * ``RefinedGain``, ``RefinedPM``,
    ``RefinedNoise``                      — three concrete refinements
  * ``SpecContract``                      — type alias `list[Refinement]`
  * ``RefinementValidation``              — inhabitability result record
  * ``RefinementVerifyResult``            — empirical-verify result record
  * ``validate_refinement_inhabitable``  — pre-simulation static check
  * ``verify_with_refinement``           — pre-simulation static check
                                            then simulator + empirical
                                            check + distance signal
  * ``format_refinement_violation_for_llm``
                                          — LLM-friendly text rendering

References
----------
* Flux (PLDI 2023): "Refinement Types for Rust" — production-grade
  refinement-type checker for Rust; we borrow the same predicate
  language (``{ x : T | P(x) }``).
* Liquid Haskell — the canonical interactive refinement-type system;
  we borrow the binary contradiction check structure (same metric ⇒
  fold predicates into a normalized closed-interval form, then check
  emptiness).
* Type-Constrained Code Generation (NeurIPS 2025 D&B / Karras et al.):
  reports that LLM compilation-error rate drops ~50% when the generator
  sees compile-time TYPE rejection signals instead of only runtime
  test failures. Refinement types are our analog for circuit specs.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Optional, Sequence


# ─── Tolerances and constants ─────────────────────────────────────


# Relative tolerance used by ``RefinedGain`` / ``RefinedPM`` / ...
# when computing inhabitability of an equality refinement against a
# bound. Matches ``clay.distance_to_sat.EQ_RELATIVE_TOLERANCE`` so the
# two surfaces agree on "is this satisfied" / "are these compatible".
EQ_RELATIVE_TOLERANCE: float = 0.05


# ─── Refinement abstract base ─────────────────────────────────────


@dataclass
class Refinement:
    """A single refinement-type predicate over one measure name.

    The base class carries the operator (``>=`` / ``<=`` / ``=`` /
    ``>`` / ``<`` / ``in_range``) and the predicate's bound values.
    Concrete subclasses (``RefinedGain``, ``RefinedPM``, ``RefinedNoise``)
    bind the metric name + SI unit so the LLM-facing text rendering
    stays domain-specific.

    Fields:
      * ``metric_name``   — the measure-namespace key (e.g.
                            ``"gain_dc"``). Matches the corresponding
                            ``SpecField.name`` so refinement and
                            existing ``verify()`` path agree on which
                            measure to pull from the simulator.
      * ``op``            — comparison operator. ``in_range`` requires
                            both ``lower`` and ``upper`` to be set.
      * ``lower``         — lower bound for ``in_range``; None
                            otherwise. For ``>=``/``>`` this is the
                            target; for ``<=``/``<``/``=`` this is
                            ignored.
      * ``upper``         — upper bound for ``in_range`` AND target
                            for ``<=``/``<``. For ``>=``/``>``/``=``
                            this is the target.
      * ``unit``          — SI unit string (``"dB"`` / ``"deg"`` /
                            ``"V/sqrt(Hz)"`` / ...). Surfaced verbatim
                            in the LLM-prompt rendering.

    Concrete subclasses bind ``metric_name`` and ``unit`` so the caller
    only supplies the bound values.

    Inhabitability self-check
    -------------------------
    ``self_inhabitable()`` returns True iff the predicate itself is
    satisfiable in isolation. ``in_range`` with ``lower > upper`` is
    the only single-refinement contradiction we detect at this layer;
    every other shape is trivially satisfiable in isolation.
    """

    # NOTE: every field has a default so subclasses (RefinedGain /
    # RefinedPM / RefinedNoise) can override any subset without
    # tripping Python's "non-default after default" dataclass rule.
    # Construction without explicit values is meaningless — callers
    # supply ``metric_name``, ``op``, and ``upper`` at minimum.
    metric_name: str = ""
    op: str = ">="
    upper: float = 0.0
    lower: Optional[float] = None
    unit: str = ""

    # ─── Sub-class hook: metric → SpecField/Measure conversion ─────

    def to_spec_field(self) -> "Any":
        """Materialize this refinement as a ``SpecField``.

        Centralizes the conversion so callers do not have to
        ``from clay.ir import SpecField`` themselves. The resulting
        ``SpecField`` is the value the legacy ``verify()`` path AND
        the distance-to-sat kernel both consume. ``in_range`` ops
        flatten to a ``>=``+``<=`` PAIR via
        ``SpecContract.to_spec_fields`` — a single ``SpecField`` cannot
        carry both bounds in the current IR.
        """
        # Lazy import to keep this module's import surface light.
        from .ir import SpecField

        if self.op == "in_range":
            # The caller asked for a single ``SpecField`` from an
            # ``in_range`` refinement. Surface the upper bound as the
            # field's value; the LOWER bound is dropped at this
            # conversion. Use ``SpecContract.to_spec_fields`` to get
            # both ``>=`` and ``<=`` fields paired up.
            return SpecField(
                name=self.metric_name,
                dimension=_dimension_for_unit(self.unit),
                op="<=",
                value=float(self.upper),
            )
        return SpecField(
            name=self.metric_name,
            dimension=_dimension_for_unit(self.unit),
            op=self.op,
            value=float(self.upper),
        )

    def self_inhabitable(self) -> tuple[bool, Optional[str]]:
        """Single-refinement self-check.

        Returns ``(True, None)`` if the predicate is satisfiable in
        isolation. Returns ``(False, reason)`` for degenerate shapes:

          * ``in_range`` with ``lower > upper`` (empty interval).
          * ``in_range`` with non-finite bounds (NaN / inf).
          * any op with a non-finite ``upper`` (NaN / inf target).

        Note: a single ``>= 1e308`` refinement is technically
        satisfiable (the float can hold the value) so we don't
        flag astronomically large bounds at this layer. The
        cross-refinement check below catches the "no circuit can do
        this" case when paired against an upper-bound refinement.
        """
        if not math.isfinite(self.upper):
            return (
                False,
                f"refinement {self.metric_name!r} carries non-finite "
                f"bound {self.upper!r}",
            )
        if self.op == "in_range":
            if self.lower is None:
                return (
                    False,
                    f"refinement {self.metric_name!r} op=in_range "
                    f"requires both lower and upper; lower is None",
                )
            if not math.isfinite(self.lower):
                return (
                    False,
                    f"refinement {self.metric_name!r} carries non-finite "
                    f"lower bound {self.lower!r}",
                )
            if self.lower > self.upper:
                return (
                    False,
                    f"refinement {self.metric_name!r} op=in_range has "
                    f"lower ({self.lower}) > upper ({self.upper}); "
                    f"the interval is empty",
                )
        return (True, None)

    def to_predicate_str(self) -> str:
        """Render the predicate in Liquid-Haskell-style notation.

        Used in the LLM-facing rendering so the model sees the same
        predicate language across all refinements:

          * ``gain_dc >= 60.0 dB``
          * ``pm in [60.0, 80.0] deg``
          * ``noise <= 5e-9 V/sqrt(Hz)``
        """
        unit_str = f" {self.unit}" if self.unit else ""
        if self.op == "in_range":
            lo_s = _format_number(self.lower)
            hi_s = _format_number(self.upper)
            return f"{self.metric_name} in [{lo_s}, {hi_s}]{unit_str}"
        return (
            f"{self.metric_name} {self.op} "
            f"{_format_number(self.upper)}{unit_str}"
        )


# ─── Three concrete refinement types ──────────────────────────────


def _dimension_for_unit(unit: str) -> str:
    """Map a unit string to the IR's ``SpecField.dimension`` namespace.

    The IR keeps a small enumerated dimension tag (``"db"`` / ``"freq"``
    / ``"volt"`` / ``"current"`` / ``"cap"`` / ``"time"`` / ``"angle"``
    / ``"real"``). The refinement-level unit string is more flexible
    (``"V/sqrt(Hz)"`` carries the spectral-density semantics) so the
    conversion folds the flexible string back to the enumerated set
    for storage on the ``SpecField``.
    """
    u = unit.lower()
    if u in ("db", "dbv"):
        return "db"
    if u in ("hz", "khz", "mhz", "ghz", "freq"):
        return "freq"
    if u in ("v", "mv", "uv", "nv", "volt"):
        return "volt"
    if u in ("a", "ma", "ua", "na", "current"):
        return "current"
    if u in ("deg", "degrees", "rad", "angle"):
        return "angle"
    if u in ("s", "ms", "us", "ns", "ps", "time"):
        return "time"
    if u in ("f", "pf", "nf", "uf", "cap"):
        return "cap"
    if u in ("v/sqrt(hz)", "nv/sqrt(hz)", "uv/sqrt(hz)"):
        return "real"
    return "real"


@dataclass
class RefinedGain(Refinement):
    """Refinement on DC gain ``gain_dc`` (units: dB).

    Typical declaration::

        RefinedGain.at_least(60.0)         # gain_dc >= 60 dB
        RefinedGain.at_most(80.0)          # gain_dc <= 80 dB
        RefinedGain.in_range(60.0, 80.0)   # gain_dc in [60, 80] dB

    The metric name (``"gain_dc"``) and SI unit (``"dB"``) are bound
    by the subclass; callers only supply the bound values.
    """

    # Dataclass field defaults so the subclass can be constructed
    # without supplying ``metric_name`` / ``unit`` explicitly.
    metric_name: str = "gain_dc"
    unit: str = "dB"

    @classmethod
    def at_least(cls, min_db: float) -> "RefinedGain":
        """``gain_dc >= min_db`` (floor)."""
        return cls(
            metric_name="gain_dc",
            op=">=",
            upper=float(min_db),
            unit="dB",
        )

    @classmethod
    def at_most(cls, max_db: float) -> "RefinedGain":
        """``gain_dc <= max_db`` (ceiling)."""
        return cls(
            metric_name="gain_dc",
            op="<=",
            upper=float(max_db),
            unit="dB",
        )

    @classmethod
    def in_range(cls, min_db: float, max_db: float) -> "RefinedGain":
        """``gain_dc in [min_db, max_db]`` (band)."""
        return cls(
            metric_name="gain_dc",
            op="in_range",
            lower=float(min_db),
            upper=float(max_db),
            unit="dB",
        )


@dataclass
class RefinedPM(Refinement):
    """Refinement on phase margin ``pm`` (units: degrees).

    Typical declaration::

        RefinedPM.at_least(60.0)           # pm >= 60 deg
        RefinedPM.in_range(60.0, 80.0)     # pm in [60, 80] deg

    Empirically, phase margin under 45° rings; over 90° the loop is
    over-damped and settling slows. The ``in_range`` band is the more
    realistic "engineering-good-PM" predicate.
    """

    metric_name: str = "pm"
    unit: str = "deg"

    @classmethod
    def at_least(cls, min_deg: float) -> "RefinedPM":
        """``pm >= min_deg`` (floor)."""
        return cls(
            metric_name="pm",
            op=">=",
            upper=float(min_deg),
            unit="deg",
        )

    @classmethod
    def at_most(cls, max_deg: float) -> "RefinedPM":
        """``pm <= max_deg`` (ceiling)."""
        return cls(
            metric_name="pm",
            op="<=",
            upper=float(max_deg),
            unit="deg",
        )

    @classmethod
    def in_range(cls, min_deg: float, max_deg: float) -> "RefinedPM":
        """``pm in [min_deg, max_deg]`` (band)."""
        return cls(
            metric_name="pm",
            op="in_range",
            lower=float(min_deg),
            upper=float(max_deg),
            unit="deg",
        )


@dataclass
class RefinedNoise(Refinement):
    """Refinement on input-referred noise ``noise``.

    Typical declaration::

        RefinedNoise.at_most(5e-9)                 # noise <= 5e-9 V/sqrt(Hz)
        RefinedNoise.at_most(5.0, unit="nV/sqrt(Hz)")
                                                    # 5 nV/sqrt(Hz)

    Noise is by convention an upper-bound (lower noise is always
    better) so ``at_most`` is the default constructor; ``at_least``
    is supported for completeness (e.g. testing instrumentation noise
    floor) but rarely used in practice.
    """

    metric_name: str = "noise"
    unit: str = "V/sqrt(Hz)"

    @classmethod
    def at_most(
        cls, max_value: float, *, unit: str = "V/sqrt(Hz)",
    ) -> "RefinedNoise":
        """``noise <= max_value`` (ceiling)."""
        return cls(
            metric_name="noise",
            op="<=",
            upper=float(max_value),
            unit=unit,
        )

    @classmethod
    def at_least(
        cls, min_value: float, *, unit: str = "V/sqrt(Hz)",
    ) -> "RefinedNoise":
        """``noise >= min_value`` (floor — rarely used)."""
        return cls(
            metric_name="noise",
            op=">=",
            upper=float(min_value),
            unit=unit,
        )

    @classmethod
    def in_range(
        cls,
        min_value: float,
        max_value: float,
        *,
        unit: str = "V/sqrt(Hz)",
    ) -> "RefinedNoise":
        """``noise in [min_value, max_value]`` (band)."""
        return cls(
            metric_name="noise",
            op="in_range",
            lower=float(min_value),
            upper=float(max_value),
            unit=unit,
        )


# ─── SpecContract: a collection of refinements ─────────────────────


# Type alias used in public function signatures. We keep it as a
# bare list so callers can construct one with no ceremony:
#
#   contract: SpecContract = [
#       RefinedGain.at_least(60.0),
#       RefinedPM.at_least(60.0),
#       RefinedNoise.at_most(5e-9),
#   ]
SpecContract = list  # type: ignore[assignment]


def contract_to_spec_fields(contract: Sequence[Refinement]) -> list:
    """Convert a SpecContract to a list of ``clay.ir.SpecField``.

    Every refinement materializes as one SpecField, except for
    ``in_range`` which flattens to a pair (``>= lower`` and
    ``<= upper``). The resulting list can be dropped straight into a
    ``Spec.fields`` slot or fed to ``evaluate_spec_distance``.

    Field order in the output mirrors the refinement order in the
    input, with ``in_range`` expansions kept adjacent (lower bound
    first, upper bound second) so callers see a stable layout.
    """
    from .ir import SpecField

    out: list = []
    for r in contract:
        if r.op == "in_range":
            if r.lower is not None:
                out.append(SpecField(
                    name=r.metric_name,
                    dimension=_dimension_for_unit(r.unit),
                    op=">=",
                    value=float(r.lower),
                ))
            out.append(SpecField(
                name=r.metric_name,
                dimension=_dimension_for_unit(r.unit),
                op="<=",
                value=float(r.upper),
            ))
        else:
            out.append(r.to_spec_field())
    return out


# ─── Inhabitability validation ────────────────────────────────────


@dataclass
class Contradiction:
    """One detected refinement-contract contradiction.

    Surfaces structurally so an LLM consumer keys off the metric name,
    the contradicting refinements, and the structured reason.

    Fields:
      * ``metric_name``     — measure-namespace key.
      * ``reason``          — short canonical description (one of
                              "empty_interval", "lower_above_upper",
                              "equality_outside_band", "non_finite_bound").
      * ``message``         — human-readable one-line summary.
      * ``refinements``     — the source refinements (as dicts) so a
                              caller can re-render them.
      * ``effective_lower`` — the binding floor across all refinements
                              on this metric (or None if no floor).
      * ``effective_upper`` — the binding ceiling across all
                              refinements on this metric (or None).
    """

    metric_name: str
    reason: str
    message: str
    refinements: list[dict] = field(default_factory=list)
    effective_lower: Optional[float] = None
    effective_upper: Optional[float] = None

    def to_json_dict(self) -> dict[str, Any]:
        """JSON-serializable representation."""
        d = asdict(self)
        for key in ("effective_lower", "effective_upper"):
            v = d.get(key)
            if isinstance(v, float) and not math.isfinite(v):
                d[key] = None
        return d


@dataclass
class RefinementValidation:
    """Result of the elaboration-stage inhabitability check.

    Fields:
      * ``inhabitable``    — True iff every metric's combined
                             refinements are mutually satisfiable.
      * ``contradictions`` — one Contradiction per metric whose
                             refinements clash. Empty when
                             ``inhabitable=True``.
      * ``self_failures``  — list of (refinement, reason) tuples for
                             refinements that fail their own self-check
                             (e.g. ``in_range`` with empty interval).
                             Empty in the happy path.
      * ``per_metric_bounds`` — ``{metric_name: (lower, upper)}`` of
                                the binding floor and ceiling derived
                                from the contract. Useful for the LLM
                                rendering even when satisfiable.
    """

    inhabitable: bool
    contradictions: list[Contradiction] = field(default_factory=list)
    self_failures: list[tuple[Refinement, str]] = field(
        default_factory=list
    )
    per_metric_bounds: dict[str, tuple[
        Optional[float], Optional[float]
    ]] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "inhabitable": self.inhabitable,
            "contradictions": [
                c.to_json_dict() for c in self.contradictions
            ],
            "self_failures": [
                {
                    "refinement": _refinement_as_dict(r),
                    "reason": reason,
                }
                for r, reason in self.self_failures
            ],
            "per_metric_bounds": {
                k: list(v) for k, v in self.per_metric_bounds.items()
            },
        }


def _refinement_as_dict(r: Refinement) -> dict[str, Any]:
    """Plain-dict form of a Refinement suitable for JSON.

    We can't use ``dataclasses.asdict`` directly because the field set
    is identical across all concrete subclasses; we want the class name
    in the output so the LLM keeps the type identity.
    """
    return {
        "type": r.__class__.__name__,
        "metric_name": r.metric_name,
        "op": r.op,
        "upper": float(r.upper),
        "lower": (
            None if r.lower is None else float(r.lower)
        ),
        "unit": r.unit,
    }


def _floor_from_refinement(r: Refinement) -> Optional[float]:
    """Return the floor (lower bound) implied by a refinement.

    For ``>=`` / ``>`` the floor is ``upper`` (we use ``upper`` as the
    target-bound storage even for floor refinements — see Refinement
    docstring). For ``in_range`` it's ``lower``. Otherwise None.
    """
    if r.op in (">=", ">"):
        return float(r.upper)
    if r.op == "in_range" and r.lower is not None:
        return float(r.lower)
    return None


def _ceiling_from_refinement(r: Refinement) -> Optional[float]:
    """Return the ceiling (upper bound) implied by a refinement."""
    if r.op in ("<=", "<"):
        return float(r.upper)
    if r.op == "in_range":
        return float(r.upper)
    return None


def _equality_value(r: Refinement) -> Optional[float]:
    """Return the equality target if this refinement is ``=`` op."""
    if r.op == "=":
        return float(r.upper)
    return None


def validate_refinement_inhabitable(
    cell: Any,
    contract: Sequence[Refinement],
    *,
    eq_rel_tol: float = EQ_RELATIVE_TOLERANCE,
) -> RefinementValidation:
    """Static elaboration-stage check: is the contract satisfiable?

    The check is composed of two passes:

      1. Per-refinement self-check. Each refinement implements
         ``self_inhabitable()``; failures land in ``self_failures``.

      2. Cross-refinement contradiction check. For each metric_name
         present in the contract, fold the per-refinement bounds into
         a single (effective_floor, effective_ceiling) interval and
         check whether it's empty. Equality refinements are
         intersected with the closed-interval form by checking whether
         the target value lies inside the interval (with the EQ
         tolerance band applied).

    The ``cell`` argument is currently unused but reserved for future
    refinements that depend on cell-level structural knowledge (e.g.
    "every input transconductor's W must satisfy refinement Y given
    L=N"). We keep the parameter in the signature so future extensions
    don't break callers.

    Args:
      cell: a ``clay.ir.Cell`` (reserved — used only for future
        cell-aware refinements; the v0.1.02 ship-set is cell-blind).
      contract: a list of ``Refinement`` instances.
      eq_rel_tol: relative-tolerance band applied to ``=`` refinements
        when checking whether the equality target lies inside the
        bracketed interval implied by the rest of the contract.

    Returns:
      A ``RefinementValidation`` with ``inhabitable`` set and (if
      False) one or more Contradiction records.
    """
    _ = cell  # reserved
    contradictions: list[Contradiction] = []
    self_failures: list[tuple[Refinement, str]] = []
    per_metric_bounds: dict[
        str, tuple[Optional[float], Optional[float]]
    ] = {}

    # Pass 1: per-refinement self-check.
    valid_refinements: list[Refinement] = []
    for r in contract:
        ok, reason = r.self_inhabitable()
        if not ok:
            self_failures.append((r, reason or "self-check failed"))
            continue
        valid_refinements.append(r)

    # Group surviving refinements by metric name.
    by_metric: dict[str, list[Refinement]] = {}
    for r in valid_refinements:
        by_metric.setdefault(r.metric_name, []).append(r)

    # Pass 2: per-metric contradiction check.
    for metric, group in by_metric.items():
        # Build the effective interval from the group.
        effective_lo: Optional[float] = None
        effective_hi: Optional[float] = None
        eq_targets: list[float] = []
        for r in group:
            lo = _floor_from_refinement(r)
            hi = _ceiling_from_refinement(r)
            if lo is not None:
                effective_lo = (
                    lo if effective_lo is None else max(effective_lo, lo)
                )
            if hi is not None:
                effective_hi = (
                    hi if effective_hi is None else min(effective_hi, hi)
                )
            eq_v = _equality_value(r)
            if eq_v is not None:
                eq_targets.append(eq_v)

        per_metric_bounds[metric] = (effective_lo, effective_hi)

        # Check for an empty interval.
        if (
            effective_lo is not None
            and effective_hi is not None
            and effective_lo > effective_hi
        ):
            contradictions.append(Contradiction(
                metric_name=metric,
                reason="empty_interval",
                message=(
                    f"refinements on {metric!r} produce an empty "
                    f"interval: effective floor={effective_lo:g} > "
                    f"effective ceiling={effective_hi:g}; no value "
                    f"satisfies the contract"
                ),
                refinements=[_refinement_as_dict(r) for r in group],
                effective_lower=effective_lo,
                effective_upper=effective_hi,
            ))
            continue

        # Check equality targets fall inside the (lo, hi) interval.
        contradicted = False
        for eq_v in eq_targets:
            tol = eq_rel_tol * max(abs(eq_v), 1.0)
            below = (
                effective_lo is not None
                and eq_v < effective_lo - tol
            )
            above = (
                effective_hi is not None
                and eq_v > effective_hi + tol
            )
            if below or above:
                contradictions.append(Contradiction(
                    metric_name=metric,
                    reason="equality_outside_band",
                    message=(
                        f"refinements on {metric!r} include "
                        f"equality target {eq_v:g} but other "
                        f"refinements require value in "
                        f"[{effective_lo}, {effective_hi}]; the "
                        f"equality target lies outside that band"
                    ),
                    refinements=[
                        _refinement_as_dict(r) for r in group
                    ],
                    effective_lower=effective_lo,
                    effective_upper=effective_hi,
                ))
                contradicted = True
                break

        # Check two equality targets aren't mutually contradictory.
        if not contradicted and len(eq_targets) >= 2:
            seen_value = eq_targets[0]
            for next_v in eq_targets[1:]:
                tol = eq_rel_tol * max(
                    abs(seen_value), abs(next_v), 1.0,
                )
                if abs(next_v - seen_value) > tol:
                    contradictions.append(Contradiction(
                        metric_name=metric,
                        reason="equality_outside_band",
                        message=(
                            f"refinements on {metric!r} include "
                            f"multiple incompatible equality targets: "
                            f"{seen_value:g} and {next_v:g} "
                            f"(differ by more than {tol:g})"
                        ),
                        refinements=[
                            _refinement_as_dict(r) for r in group
                        ],
                        effective_lower=effective_lo,
                        effective_upper=effective_hi,
                    ))
                    break

    inhabitable = (
        not contradictions
        and not self_failures
    )
    return RefinementValidation(
        inhabitable=inhabitable,
        contradictions=contradictions,
        self_failures=self_failures,
        per_metric_bounds=per_metric_bounds,
    )


# ─── Empirical verify: bridge to simulator + distance kernel ──────


@dataclass
class RefinementVerifyResult:
    """Aggregate result for the refinement-aware verify path.

    Combines the static inhabitability check, the simulator-driven
    boolean check (from ``clay.verifier.verify``), and the continuous
    distance signal (from ``clay.distance_to_sat.evaluate_spec_distance``).

    Fields:
      * ``cell_name``           — echoed from the underlying VerifyResult.
      * ``validation``          — the elaboration-stage inhabitability
                                  result. When ``validation.inhabitable``
                                  is False, the simulator never runs.
      * ``simulated``           — True if the cell was actually
                                  simulated (i.e. the contract was
                                  inhabitable AND the testbench was
                                  provided). False when the contract
                                  was rejected pre-simulation.
      * ``satisfied``           — True iff every refinement is
                                  satisfied by the measured values
                                  (or by definition False when not
                                  simulated).
      * ``violations``          — list of ``SpecViolation`` records
                                  from the distance kernel (empty when
                                  not simulated). Reuses R78 A's
                                  SpecViolation shape verbatim.
      * ``satisfaction_loss``   — aggregate continuous loss from
                                  ``compute_satisfaction_loss``;
                                  ``math.inf`` when not simulated.
      * ``verify_result``       — the underlying ``VerifyResult`` from
                                  ``clay.verifier.verify``, or None
                                  when not simulated.
    """

    cell_name: str
    validation: RefinementValidation
    simulated: bool
    satisfied: bool
    violations: list = field(default_factory=list)
    satisfaction_loss: float = math.inf
    verify_result: Any = None

    @property
    def inhabitable(self) -> bool:
        """Shortcut for ``self.validation.inhabitable``."""
        return self.validation.inhabitable

    @property
    def llm_text(self) -> str:
        """LLM-friendly multi-line rendering.

        Combines:
          * The contract listing (as predicate strings).
          * Either the contradiction report (when not inhabitable) OR
            the distance report (when simulated).
        """
        return format_refinement_violation_for_llm(self)


def verify_with_refinement(
    cell: Any,
    contract: Sequence[Refinement],
    *,
    work_dir: Any = "tmp_clay_rundir",
    driver: Any = None,
    testbench: Any = None,
    weights: Optional[Mapping[str, float]] = None,
    eq_rel_tol: float = EQ_RELATIVE_TOLERANCE,
    treat_inf_as: Optional[float] = None,
) -> RefinementVerifyResult:
    """One-call verifier with refinement-type contract.

    Pipeline:

      1. Run the static inhabitability check via
         ``validate_refinement_inhabitable``. If the contract is not
         inhabitable, RETURN IMMEDIATELY with ``simulated=False``,
         ``satisfied=False``, and the structured contradiction record.
         The simulator does NOT run.
      2. Otherwise materialize the contract as a list of ``SpecField``
         and run the legacy ``clay.verifier.verify`` to drive the
         simulator. The cell's existing spec is REPLACED with the
         refinement-derived spec for the verify run (a deepcopy avoids
         mutating the caller's cell).
      3. Build SpecViolation records from the simulator's measures
         and compose the satisfaction_loss via the distance kernel.

    Args:
      cell: a ``clay.ir.Cell``. The cell's ``.spec`` is replaced with
        the contract-derived ``Spec`` for the verify call.
      contract: a list of ``Refinement`` instances.
      work_dir / driver: forwarded to ``clay.verifier.verify``.
      testbench: optional pre-built ``Testbench``; if None, the caller
        is expected to wire one up itself OR rely on the cell having
        a default testbench (the function uses a synthesized minimal
        testbench by default; see ``_build_minimal_testbench``).
      weights: optional ``{metric_name: weight}`` map (uniform if None).
      eq_rel_tol: relative-tolerance band for ``=`` ops (forwarded to
        both the inhabitability check AND the distance kernel).
      treat_inf_as: replacement value for ``math.inf`` in the
        aggregate satisfaction loss; None propagates inf.

    Returns:
      A ``RefinementVerifyResult``.
    """
    # Stage 1: static inhabitability check.
    validation = validate_refinement_inhabitable(
        cell, contract, eq_rel_tol=eq_rel_tol,
    )
    if not validation.inhabitable:
        return RefinementVerifyResult(
            cell_name=getattr(cell, "name", "<unknown>"),
            validation=validation,
            simulated=False,
            satisfied=False,
            violations=[],
            satisfaction_loss=math.inf,
            verify_result=None,
        )

    # Stage 2: simulate and check empirically. We materialize the
    # contract as a list of SpecField, swap the cell's spec for the
    # verify run, and rely on R78 A's distance kernel for the
    # SpecViolation records.
    from copy import deepcopy

    from .ir import Spec, Testbench
    from .verifier import verify as _verify
    from .distance_to_sat import (
        compute_satisfaction_loss,
        evaluate_spec_distance,
    )

    spec_fields = contract_to_spec_fields(contract)
    # Build a per-call cell with the refinement-derived spec. The
    # caller's cell is NOT mutated.
    cell_for_verify = deepcopy(cell)
    cell_for_verify.spec = Spec(fields=spec_fields)

    # Build or take a testbench. If the caller didn't supply one we
    # synthesize a minimal one (cell + empty source/measure list) —
    # the legacy ``verify()`` path will then surface a
    # ``simulator_error`` if the cell genuinely needs a testbench
    # the caller forgot to wire up. This keeps the API ergonomic for
    # unit tests where the cell is the only artifact under test.
    if testbench is None:
        tb = Testbench(cell=cell_for_verify)
    else:
        tb = deepcopy(testbench)
        tb.cell = cell_for_verify

    verify_result = _verify(tb, work_dir=work_dir, driver=driver)
    measures = dict(verify_result.measures) if verify_result.measures else {}

    # Stage 3: build distance records over the refinement-derived
    # spec_fields. We DON'T just reuse VerifyResult.assertions because
    # ``in_range`` refinements expand to a ``>=``+``<=`` PAIR in
    # spec_fields, but the LLM-facing rendering should keep the
    # ``in_range`` view. So we build per-REFINEMENT SpecViolation
    # records (one per refinement, not one per SpecField) by
    # consulting the measures directly.
    violations = _evaluate_contract_distance(
        contract, measures,
        weights=weights, eq_rel_tol=eq_rel_tol,
    )
    loss = compute_satisfaction_loss(
        violations, treat_inf_as=treat_inf_as,
    )
    satisfied = all(v.satisfied for v in violations)

    return RefinementVerifyResult(
        cell_name=verify_result.cell_name,
        validation=validation,
        simulated=True,
        satisfied=satisfied,
        violations=violations,
        satisfaction_loss=loss,
        verify_result=verify_result,
    )


def _evaluate_contract_distance(
    contract: Sequence[Refinement],
    measures: Mapping[str, Optional[float]],
    *,
    weights: Optional[Mapping[str, float]] = None,
    eq_rel_tol: float = EQ_RELATIVE_TOLERANCE,
) -> list:
    """Build SpecViolation records, one per Refinement.

    Unlike ``clay.distance_to_sat.evaluate_spec_distance`` (which is
    one-per-SpecField), this walks the contract itself so an
    ``in_range`` refinement produces a SINGLE record (preserving the
    in_range view). For each refinement we compute the distance via
    ``compute_spec_distance`` with the refinement's full bounds.
    """
    from .distance_to_sat import (
        DEFAULT_WEIGHT,
        SpecViolation,
        compute_spec_distance,
    )

    weights = dict(weights or {})
    out: list = []
    for r in contract:
        # Resolve the measured value (None / NaN / inf all collapse
        # to the "missing measure" path, matching the distance kernel).
        raw = measures.get(r.metric_name) if measures else None
        measured: Optional[float] = None
        if raw is not None and isinstance(raw, (int, float)):
            f = float(raw)
            if math.isfinite(f):
                measured = f
        try:
            distance = compute_spec_distance(
                measured if measured is not None else None,
                float(r.upper),
                r.op,
                lower=r.lower,
                eq_rel_tol=eq_rel_tol,
            )
        except ValueError:
            distance = math.inf
        weight = float(weights.get(r.metric_name, DEFAULT_WEIGHT))
        satisfied = (measured is not None and distance == 0.0)
        out.append(SpecViolation(
            spec_name=r.metric_name,
            op=r.op,
            target=float(r.upper),
            lower=r.lower,
            measured=measured,
            weight=weight,
            distance=distance,
            satisfied=satisfied,
            unit=_dimension_for_unit(r.unit),
            extras={
                "refinement_type": r.__class__.__name__,
                "refinement_unit": r.unit,
                "measure_was_missing": measured is None,
            },
        ))
    return out


# ─── LLM-prompt-friendly formatting ───────────────────────────────


def _format_number(v: Optional[float]) -> str:
    """Compact numeric rendering (same conventions as distance_to_sat)."""
    if v is None:
        return "n/a"
    if not math.isfinite(v):
        return "inf" if v > 0 else "-inf"
    av = abs(v)
    if av >= 1e3 or (av < 1e-2 and v != 0.0):
        return f"{v:.3e}"
    return f"{v:.4g}"


def format_refinement_violation_for_llm(
    result: RefinementVerifyResult,
    *,
    include_satisfied: bool = True,
) -> str:
    """Build LLM-readable text for a ``RefinementVerifyResult``.

    Layout differs based on the validation outcome:

      * NOT INHABITABLE (compile-time negative signal):

            Refinement contract: NOT INHABITABLE (pre-simulation reject)
              Contradictions:
                gain_dc: empty_interval — effective floor=60 dB >
                  effective ceiling=40 dB. Refinements:
                    - gain_dc >= 60.0 dB
                    - gain_dc <= 40.0 dB
              Per-metric effective bounds:
                gain_dc: [60.0, 40.0] (empty)
              Action: design cannot satisfy this contract — no simulator
              call was issued. Either relax the floor or raise the
              ceiling so the interval is non-empty.

      * INHABITABLE + SIMULATED:

            Refinement contract: INHABITABLE
              Contract:
                - gain_dc >= 60.0 dB
                - pm in [60.0, 80.0] deg
                - noise <= 5e-9 V/sqrt(Hz)
              Empirical check:
                gain_dc:  measured 55.0 dB,  required >= 60.0 dB  -> DISTANCE 5.0 dB
                pm:       measured 70.0 deg, required in [60, 80] -> SATISFIED
                noise:    measured 6e-9,     required <= 5e-9     -> DISTANCE 1e-9
              Total satisfaction loss: 6.0 (1 of 3 specs satisfied)

    Args:
      result: a ``RefinementVerifyResult`` (returned by
        ``verify_with_refinement``).
      include_satisfied: when False, satisfied empirical checks
        collapse to a tail line. Default True (the rendering is short
        — usually a few specs — and an LLM prompt benefits from full
        evidence).
    """
    lines: list[str] = []
    if not result.validation.inhabitable:
        lines.append(
            "Refinement contract: NOT INHABITABLE "
            "(pre-simulation reject)"
        )
        if result.validation.self_failures:
            lines.append("  Self-failures:")
            for r, reason in result.validation.self_failures:
                lines.append(f"    - {r.to_predicate_str()}: {reason}")
        if result.validation.contradictions:
            lines.append("  Contradictions:")
            for c in result.validation.contradictions:
                lines.append(f"    {c.metric_name}: {c.reason}")
                lines.append(f"      {c.message}")
                if c.refinements:
                    lines.append("      Refinements:")
                    for rd in c.refinements:
                        # Render each refinement in predicate notation.
                        if rd.get("op") == "in_range":
                            lo_s = _format_number(rd.get("lower"))
                            hi_s = _format_number(rd.get("upper"))
                            unit_s = (
                                f" {rd.get('unit')}"
                                if rd.get("unit") else ""
                            )
                            lines.append(
                                f"        - {rd.get('metric_name')} "
                                f"in [{lo_s}, {hi_s}]{unit_s}"
                            )
                        else:
                            unit_s = (
                                f" {rd.get('unit')}"
                                if rd.get("unit") else ""
                            )
                            lines.append(
                                f"        - {rd.get('metric_name')} "
                                f"{rd.get('op')} "
                                f"{_format_number(rd.get('upper'))}"
                                f"{unit_s}"
                            )
        if result.validation.per_metric_bounds:
            lines.append("  Per-metric effective bounds:")
            for m, (lo, hi) in result.validation.per_metric_bounds.items():
                lo_s = _format_number(lo)
                hi_s = _format_number(hi)
                empty_tag = ""
                if (
                    lo is not None and hi is not None
                    and lo > hi
                ):
                    empty_tag = " (empty)"
                lines.append(
                    f"    {m}: [{lo_s}, {hi_s}]{empty_tag}"
                )
        lines.append(
            "  Action: design cannot satisfy this contract — no "
            "simulator call was issued. Either relax the floor or "
            "raise the ceiling so the interval is non-empty."
        )
        return "\n".join(lines)

    # Inhabitable branch.
    lines.append("Refinement contract: INHABITABLE")
    if not result.simulated:
        lines.append(
            "  (Simulation was not requested; contract validated "
            "as type-level satisfiable.)"
        )
        return "\n".join(lines)
    lines.append("  Empirical check:")
    sat_count = 0
    missing_count = 0
    for v in result.violations:
        if v.satisfied:
            sat_count += 1
        if v.measured is None or (
            isinstance(v.distance, float) and math.isinf(v.distance)
        ):
            missing_count += 1
        # Render unit using the stored canonical dimension token
        # (matches distance_to_sat formatter; we re-implement the
        # unit-folding locally to avoid coupling).
        unit_str = v.extras.get("refinement_unit", "") if v.extras else ""
        unit_suffix = f" {unit_str}" if unit_str else ""
        m_str = _format_number(v.measured) + unit_suffix
        if v.op == "in_range":
            lo_s = _format_number(v.lower)
            hi_s = _format_number(v.target)
            req_str = f"in [{lo_s}, {hi_s}]{unit_suffix}"
        else:
            req_str = (
                f"{v.op} {_format_number(v.target)}{unit_suffix}"
            )
        if v.satisfied:
            if include_satisfied:
                lines.append(
                    f"    {v.spec_name}: measured {m_str}, "
                    f"required {req_str}  -> SATISFIED"
                )
            continue
        if v.measured is None or (
            isinstance(v.distance, float) and math.isinf(v.distance)
        ):
            lines.append(
                f"    {v.spec_name}: measured {m_str}, "
                f"required {req_str}  -> DISTANCE inf (measure missing)"
            )
            continue
        d_str = _format_number(v.distance) + unit_suffix
        lines.append(
            f"    {v.spec_name}: measured {m_str}, "
            f"required {req_str}  -> DISTANCE {d_str}"
        )
    n = len(result.violations)
    loss = result.satisfaction_loss
    loss_str = (
        "inf" if (isinstance(loss, float) and math.isinf(loss))
        else f"{loss:.4g}"
    )
    tail_bits = [f"{sat_count} of {n} specs satisfied"]
    if missing_count > 0:
        tail_bits.append(f"{missing_count} measure missing")
    lines.append(
        f"  Total satisfaction loss: {loss_str} "
        f"({', '.join(tail_bits)})"
    )
    return "\n".join(lines)


def format_contract_for_llm(contract: Sequence[Refinement]) -> str:
    """Render a SpecContract as a list of predicate strings.

    Useful as a prompt fragment: "Generate a cell satisfying the
    following refinement contract" + ``format_contract_for_llm(...)``.
    """
    lines = ["Refinement contract:"]
    for r in contract:
        lines.append(f"  - {r.to_predicate_str()}")
    return "\n".join(lines)
