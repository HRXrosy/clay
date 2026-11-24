"""
Clay distance-to-satisfaction loss (v0.1.01, Round 78 §5.2 Tier 1).

Per the LLM-extrapolation-philosophy audit (`docs/audits/
2026-05-19-clay-llm-extrapolation-philosophy.html`, §5.2):

The MDP env Clay exposes to an LLM agent currently reports spec
satisfaction as a BOOLEAN per assertion (`pass` vs `spec_violation`).
That is a classic sparse-reward setup — the LLM cannot tell whether
its trial circuit is 1 dB short or 20 dB short, so it has no
gradient-like signal to guide refinement.

This module expresses refinement-type violations as a CONTINUOUS
distance-to-satisfaction value:

    type GainSpec = { gain_dc : f64 | gain_dc >= 60.0 }

    violation(GainSpec)(measured) =
        max(0, 60.0 - measured)     # for >= ops
        max(0, measured - target)   # for <= ops
        |measured - target|         # for == ops
        max(measured - hi, 0) +     # for range ops (lo, hi)
        max(lo - measured, 0)

The Clay adjoint network (`crates/clay_ir/src/adjoint.rs`, exposed
via PyO3 through `clay._ir_bridge.gradient_on_functional_target`)
already returns `∂(measured)/∂(W)` for sizing parameters that flow
into the AC / OP analysis. The distance-to-satisfaction loss is
piecewise-differentiable in `measured`:

    ∂violation/∂measured =
        −1   if op is `>=` and violated
        +1   if op is `<=` and violated
        sign(measured − target)   if op is `=` and violated
        0    if op is satisfied

Compose with the adjoint's `∂measured/∂W` via the chain rule to get
`∂violation/∂W`, then `∂satisfaction_loss/∂W` is the weighted sum.
The LLM RL reward becomes `−satisfaction_loss` (a smooth, dense
signal) instead of the v0.0.x 0/1 step.

API surface (additive — every existing `verify`/`check_*` path keeps
byte-exact behavior):

  * `SpecViolation`                   — one per-spec violation record
  * `compute_spec_distance(...)`      — scalar distance for one spec
  * `compute_satisfaction_loss(...)`  — weighted sum across specs
  * `compute_satisfaction_gradient(...)` — ∂loss/∂param via adjoint
  * `format_violation_for_llm(...)`   — text rendering for prompts
  * `evaluate_spec_distance(...)`     — top-level: build SpecViolations
                                        from a `Spec` + measure dict
  * `verify_with_distance(...)`       — verifier-shaped one-call entry

The module is import-light: it only depends on `clay.ir.SpecField` for
the spec dataclass, the standard library, and on `clay._ir_bridge` for
the (optional) adjoint chain. Importing `clay.distance_to_sat` never
triggers a Rust extension load; the bridge call is lazy and soft-fails
when `_HAS_CLAY_IR` is False.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence


# ─── Public constants ─────────────────────────────────────────────


# Tolerance applied to `=` (equality) spec comparisons before the
# violation is considered non-zero. Matches `clay.verifier._check`'s
# 5% relative-tolerance band on `=` so the distance-aware path agrees
# with the boolean-aware path on what counts as "satisfied".
EQ_RELATIVE_TOLERANCE: float = 0.05

# A sentinel weight for specs that don't carry an explicit weight in
# the IR. Equal weighting is the safest default for an LLM-RL reward;
# downstream callers that want a non-uniform schedule pass an explicit
# `weights` dict to `compute_satisfaction_loss`.
DEFAULT_WEIGHT: float = 1.0


# ─── Data classes ─────────────────────────────────────────────────


@dataclass
class SpecViolation:
    """One spec assertion expressed as a continuous distance.

    Fields:
      * `spec_name`      — the SpecField's name (e.g. `gain_dc`).
      * `op`             — the comparison operator (`>=`, `<=`, `=`,
                           `>`, `<`, `in_range`).
      * `target`         — the spec's target value (in SI units). For
                           range specs this is the upper bound.
      * `lower`          — the lower bound for `in_range` specs;
                           None otherwise. Stored so the LLM-facing
                           text rendering can show the full band.
      * `measured`       — what the simulator returned. None when the
                           measure was missing (no continuous signal
                           is meaningful — we surface a sentinel
                           `distance=math.inf` so a downstream weighted
                           sum still flags the cell as failing without
                           injecting a magic finite number).
      * `weight`         — non-negative scalar applied to `distance`
                           when aggregating into the satisfaction loss.
      * `distance`       — `≥ 0`. `0.0` means satisfied (within the
                           operator's tolerance band); `> 0` means
                           that's how far the measurement is from
                           satisfaction in the units of `measured`.
      * `satisfied`      — boolean shortcut. Always `distance == 0.0`
                           AND `measured is not None`. The boolean
                           path is kept so callers that already key
                           off `passed` need a small adapter.
      * `unit`            — optional unit string for LLM rendering
                            (e.g. `"dB"`, `"Hz"`, `"V"`). Carried
                            verbatim from the SpecField's
                            `dimension` field when available.

    Every SpecViolation is JSON-serializable through `to_json_dict()`.
    Non-finite values (NaN, ±inf) are coerced to None so the surface
    matches the v0.0.x InvariantViolation contract.
    """

    spec_name: str
    op: str
    target: float
    measured: Optional[float]
    weight: float
    distance: float
    satisfied: bool
    lower: Optional[float] = None
    unit: Optional[str] = None
    # Free-form extras for tooling that wants extra context (e.g. which
    # SpecField object was consulted, whether a margin was applied).
    extras: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        """Plain-dict form suitable for `json.dumps`.

        Non-finite floats (NaN / ±inf) become `None` so a downstream
        prompt template doesn't render `Infinity` literals (which are
        not valid JSON anyway). `distance == math.inf` collapses to
        `None` with `extras["_distance_is_infinite"] = True` so the
        receiver can still detect the missing-measure case.
        """
        d = asdict(self)
        # Coerce non-finite floats to None for JSON safety.
        for key in ("measured", "target", "lower", "distance"):
            v = d.get(key)
            if isinstance(v, float) and not math.isfinite(v):
                if key == "distance" and v == math.inf:
                    d["extras"] = dict(d.get("extras") or {})
                    d["extras"]["_distance_is_infinite"] = True
                d[key] = None
        return d


# ─── Per-spec distance computation ────────────────────────────────


_SUPPORTED_OPS: frozenset[str] = frozenset({
    ">=", ">", "<=", "<", "=", "==", "in_range",
    # Unicode forms accepted in the IR (parser produces these for the
    # `≥` / `≤` source glyphs).
    "≥", "≤",
})


def _normalize_op(op: str) -> str:
    """Fold Unicode and English `==` variants down to the canonical
    set the dispatch table understands. Unknown ops raise ValueError so
    callers see a precise error rather than a silent zero distance."""
    if op == "≥":
        return ">="
    if op == "≤":
        return "<="
    if op == "==":
        return "="
    if op in _SUPPORTED_OPS:
        return op
    raise ValueError(
        f"unsupported distance op {op!r}. Supported: "
        f"{sorted(_SUPPORTED_OPS)}"
    )


def compute_spec_distance(
    measured: float,
    target: float,
    op: str,
    *,
    lower: Optional[float] = None,
    eq_rel_tol: float = EQ_RELATIVE_TOLERANCE,
) -> float:
    """Return the non-negative distance from `measured` to satisfaction.

    Semantics by operator (all return values are ≥ 0):

      * `>=` / `>`  : `max(0, target − measured)`. Satisfied when
                       `measured >= target`. Strict `>` collapses to
                       the same distance — at the precision Clay's
                       simulator pipeline operates at the `>` vs `>=`
                       distinction is below the noise floor and we
                       prefer a continuous (non-Heaviside) signal.
      * `<=` / `<`  : `max(0, measured − target)`. Satisfied when
                       `measured <= target`. Same `<` vs `<=`
                       collapse rationale.
      * `=`         : `max(0, |measured − target| − eq_rel_tol *
                       max(|target|, 1.0))`. Satisfied when within
                       the relative-tolerance band; outside the band
                       the distance is the excess deviation. Using
                       `max(|target|, 1.0)` in the denominator avoids
                       a degenerate zero band when `target == 0`.
      * `in_range`  : `lower` and `target` are the band bounds. The
                       distance is `0` inside the band, and equal to
                       the closer-bound deficit outside (`max(0, lower
                       − measured) + max(0, measured − target)`).
                       If `lower is None` the function falls back to
                       `<=` semantics (target is the upper bound).

    `measured` of NaN or ±inf returns `math.inf` (the measurement is
    not usable — same surface as the `measured is None` path). A
    `target` that is itself NaN raises ValueError; spec targets must
    be finite real numbers.
    """
    if not math.isfinite(target):
        raise ValueError(
            f"spec target must be finite, got {target!r}"
        )
    if measured is None or (
        isinstance(measured, float) and not math.isfinite(measured)
    ):
        return math.inf
    canon = _normalize_op(op)
    if canon in (">=", ">"):
        return max(0.0, float(target) - float(measured))
    if canon in ("<=", "<"):
        return max(0.0, float(measured) - float(target))
    if canon == "=":
        deviation = abs(float(measured) - float(target))
        band = eq_rel_tol * max(abs(float(target)), 1.0)
        return max(0.0, deviation - band)
    if canon == "in_range":
        # `target` is the upper bound; `lower` is the lower bound.
        if lower is None:
            return max(0.0, float(measured) - float(target))
        lo = float(lower)
        hi = float(target)
        if lo > hi:
            # Defensive: a degenerate band can happen if the IR
            # produced an inverted (lower > upper) tuple. Treat as
            # equality at the midpoint with the half-span as
            # tolerance.
            mid = 0.5 * (lo + hi)
            span = 0.5 * abs(hi - lo)
            return max(0.0, abs(float(measured) - mid) - span)
        return (
            max(0.0, lo - float(measured))
            + max(0.0, float(measured) - hi)
        )
    raise ValueError(f"unsupported op {op!r}")


def compute_distance_gradient_dmeasured(
    measured: float,
    target: float,
    op: str,
    *,
    lower: Optional[float] = None,
    eq_rel_tol: float = EQ_RELATIVE_TOLERANCE,
) -> float:
    """Return `∂distance/∂measured` at the current measurement point.

    Piecewise constant — the distance function is a hinge / V-shape /
    chevron in `measured`. Saturated (satisfied) regions return `0`
    so the per-spec gradient zero-suppresses naturally in the chain
    rule. The sub-gradient at the kink is taken as 0 (the left/right
    derivatives bracket 0 and -1/+1 respectively; 0 is the safest
    choice for a sparse-reward signal).

    Used internally by `compute_satisfaction_gradient` to lift the
    adjoint's `∂measured/∂W` to `∂loss/∂W` via the chain rule.

    Returns 0.0 when `measured` is unusable (None / NaN / inf) — the
    caller already routes those to the `measure_missing` reporting
    path and the gradient signal carries no information.
    """
    if measured is None or (
        isinstance(measured, float) and not math.isfinite(measured)
    ):
        return 0.0
    if not math.isfinite(target):
        return 0.0
    canon = _normalize_op(op)
    m = float(measured)
    t = float(target)
    if canon in (">=", ">"):
        # distance = max(0, t - m); ∂/∂m = -1 in the violated region,
        # 0 in the satisfied region.
        return -1.0 if m < t else 0.0
    if canon in ("<=", "<"):
        # distance = max(0, m - t); ∂/∂m = +1 in the violated region.
        return 1.0 if m > t else 0.0
    if canon == "=":
        band = eq_rel_tol * max(abs(t), 1.0)
        deviation = m - t
        if abs(deviation) <= band:
            return 0.0
        return 1.0 if deviation > 0 else -1.0
    if canon == "in_range":
        if lower is None:
            return 1.0 if m > t else 0.0
        lo = float(lower)
        hi = float(t)
        if lo > hi:
            mid = 0.5 * (lo + hi)
            return 1.0 if m > mid else (-1.0 if m < mid else 0.0)
        if m < lo:
            return -1.0
        if m > hi:
            return 1.0
        return 0.0
    return 0.0


# ─── Aggregate loss ───────────────────────────────────────────────


def compute_satisfaction_loss(
    violations: Sequence[SpecViolation],
    *,
    treat_inf_as: Optional[float] = None,
) -> float:
    """Return the weighted sum of per-spec distances.

    The natural aggregation is `Σᵢ weightᵢ · distanceᵢ`. When any
    violation carries `distance = math.inf` (the measurement was
    missing or non-finite), the sum is `math.inf` by default — this
    correctly flags the cell as "I cannot evaluate the reward; treat
    as catastrophically failing". Callers building an RL reward
    typically clamp this with `treat_inf_as=large_finite`; we don't
    impose a value because what counts as "large finite" depends on
    the units of the unsatisfied spec.

    Args:
      violations: sequence of SpecViolation records (the output of
        `evaluate_spec_distance`).
      treat_inf_as: when not None, every `distance == math.inf`
        contributes this value (weighted) to the sum instead. Useful
        for RL training where an infinite reward gradient is
        undefined; e.g. pass `treat_inf_as=1e9` to use a very large
        finite penalty.

    Returns:
      Non-negative float. `0.0` iff every violation is satisfied.
    """
    total = 0.0
    for v in violations:
        d = v.distance
        if isinstance(d, float) and math.isinf(d):
            if treat_inf_as is None:
                return math.inf
            d = treat_inf_as
        total += float(v.weight) * float(d)
    return total


# ─── Top-level entry: build SpecViolation records from a Spec ─────


def evaluate_spec_distance(
    spec_fields: Sequence,
    measures: Mapping[str, Optional[float]],
    *,
    weights: Optional[Mapping[str, float]] = None,
    eq_rel_tol: float = EQ_RELATIVE_TOLERANCE,
) -> list[SpecViolation]:
    """Walk every SpecField and produce a SpecViolation record.

    Args:
      spec_fields: an iterable of `clay.ir.SpecField` (the verifier
        feeds `tb.cell.spec.fields`). Each field's `name` is used
        to look up the corresponding measured value in `measures`.
      measures: `{measure_name: float | None}` map. A missing key
        OR an explicit `None` value is treated as "measure missing"
        and surfaces `distance = math.inf`. NaN / ±inf measures are
        ALSO treated as missing (matches the verifier's
        `measure_missing` category for post-processed NaNs).
      weights: optional `{spec_name: weight}` override. Missing
        names fall back to `DEFAULT_WEIGHT`.
      eq_rel_tol: relative-tolerance band for `=` ops (default
        matches `EQ_RELATIVE_TOLERANCE`).

    Returns:
      `list[SpecViolation]` — one per SpecField, in declaration order.
    """
    weights = dict(weights or {})
    out: list[SpecViolation] = []
    for sf in spec_fields:
        name = getattr(sf, "name", None)
        if name is None:
            # Defensive: skip an unnamed field rather than crashing
            # — the parser should never produce one but we want this
            # surface to be robust to ad-hoc Spec(fields=[...]) tests.
            continue
        op = getattr(sf, "op", ">=")
        target = float(getattr(sf, "value", 0.0))
        # Range-op metadata lives on the SpecField as `lower`/`upper`
        # in a future-compatible spot. For v0.1.01 the IR's SpecField
        # carries only `value` (single number), so range support is
        # only reachable through explicit calls to
        # `compute_spec_distance(..., op="in_range", lower=...)`.
        # We surface the optional `lower` attribute lookup so future
        # IR extensions don't need to change this function.
        lower = getattr(sf, "lower", None)
        unit = getattr(sf, "dimension", None)
        raw_measured = measures.get(name) if measures else None
        # Treat NaN / ±inf as "missing" — the verifier already routes
        # those to `measure_missing` and the LLM should see a single
        # consistent surface.
        measured: Optional[float] = None
        if raw_measured is not None and isinstance(
            raw_measured, (int, float)
        ):
            f = float(raw_measured)
            if math.isfinite(f):
                measured = f
        try:
            distance = compute_spec_distance(
                measured if measured is not None else None,
                target,
                op,
                lower=lower,
                eq_rel_tol=eq_rel_tol,
            )
        except ValueError:
            # Unknown op: surface as infinite distance so an LLM
            # consumer sees a failure rather than a silent pass.
            distance = math.inf
        weight = float(weights.get(name, DEFAULT_WEIGHT))
        satisfied = (
            measured is not None and distance == 0.0
        )
        out.append(SpecViolation(
            spec_name=name,
            op=op,
            target=target,
            lower=lower,
            measured=measured,
            weight=weight,
            distance=distance,
            satisfied=satisfied,
            unit=unit,
            extras={
                "spec_field_dimension": unit,
                "measure_was_missing": measured is None,
            },
        ))
    return out


# ─── Gradient via Clay adjoint ────────────────────────────────────


# Type alias for the adjoint-call function the caller supplies. The
# signature matches `clay._ir_bridge.gradient_on_functional_target`
# minus the bridge-specific positional arguments — a small adapter
# function (one line of code) is all the caller needs to write.
#
# The contract:
#   * Takes a spec name and a sizing-parameter name.
#   * Returns the partial derivative `∂(measured value for spec)/
#     ∂(sizing param)` at the current OP. None or NaN signals
#     "adjoint unavailable" (the bridge's `_HAS_CLAY_IR=False` path,
#     pass-through attrs, or a degenerate OP solve).
AdjointFn = Callable[[str, str], Optional[float]]


def compute_satisfaction_gradient(
    violations: Sequence[SpecViolation],
    sizing_params: Sequence[str],
    adjoint_fn: AdjointFn,
    *,
    eq_rel_tol: float = EQ_RELATIVE_TOLERANCE,
) -> dict[str, float]:
    """Compute `∂satisfaction_loss/∂param` for each sizing parameter.

    Chain rule across the violations:

        loss = Σᵢ wᵢ · distanceᵢ(measuredᵢ)
        ∂loss/∂param = Σᵢ wᵢ · (∂distanceᵢ/∂measuredᵢ)
                            · (∂measuredᵢ/∂param)

    `(∂distanceᵢ/∂measuredᵢ)` is the piecewise sign function from
    `compute_distance_gradient_dmeasured` (-1, 0, or +1). The
    `(∂measuredᵢ/∂param)` term is what the Clay adjoint emits — the
    caller supplies it via `adjoint_fn(spec_name, param_name)`.

    Args:
      violations: SpecViolation records (typically from
        `evaluate_spec_distance`).
      sizing_params: list of sizing-parameter names. The returned
        dict has one entry per param name (with value 0.0 when no
        violation contributes).
      adjoint_fn: caller-supplied callback returning the partial
        `∂measured/∂param`. Returning None / NaN / inf for a
        (spec, param) pair zero-suppresses that term (the
        contribution is treated as 0). The callback is invoked at
        most once per (spec_name, param_name) pair.
      eq_rel_tol: relative-tolerance band for `=` ops.

    Returns:
      `{param_name: float}` — the partial derivatives. Missing
      gradients (no contribution from any violation) land as 0.0.
      Non-finite per-spec contributions zero-suppress, so a single
      missing adjoint never poisons the whole gradient.
    """
    grads: dict[str, float] = {p: 0.0 for p in sizing_params}
    for v in violations:
        # Saturated-region violations have ∂distance/∂measured = 0
        # already; we skip the adjoint call entirely to avoid the
        # PyO3 round trip on satisfied specs.
        if v.distance == 0.0:
            continue
        dD_dm = compute_distance_gradient_dmeasured(
            v.measured,
            v.target,
            v.op,
            lower=v.lower,
            eq_rel_tol=eq_rel_tol,
        )
        if dD_dm == 0.0:
            continue
        for param in sizing_params:
            try:
                dm_dp = adjoint_fn(v.spec_name, param)
            except Exception:  # noqa: BLE001 — soft fail per param
                dm_dp = None
            if dm_dp is None:
                continue
            if not isinstance(dm_dp, (int, float)):
                continue
            f = float(dm_dp)
            if not math.isfinite(f):
                continue
            grads[param] += float(v.weight) * dD_dm * f
    return grads


# ─── LLM-prompt-friendly formatting ───────────────────────────────


def _format_number(v: Optional[float]) -> str:
    """Compact rendering for prompt strings.

    None becomes `"n/a"`. Numbers outside `[1e-2, 1e3)` are rendered
    in scientific notation with 3 significant digits; in-range numbers
    use the `%.4g` fixed-point form so the eye reads `40` instead of
    `4.000e+01`.
    """
    if v is None:
        return "n/a"
    if not math.isfinite(v):
        return "inf" if v > 0 else "-inf"
    av = abs(v)
    if av >= 1e3 or (av < 1e-2 and v != 0.0):
        return f"{v:.3e}"
    return f"{v:.4g}"


def _format_unit(unit: Optional[str]) -> str:
    """Tag an SI unit suffix onto a value rendering when one is set.

    Empty / None / `"real"` map to "" (the IR's default dimension is
    `"real"` for unitless quantities). Other dimension tokens (`"db"`,
    `"freq"`, `"volt"`, ...) render verbatim, with the special case
    `"db"` upcased to `"dB"` to match scientific convention.
    """
    if not unit or unit == "real":
        return ""
    if unit == "db":
        return " dB"
    return f" {unit}"


def format_violation_for_llm(
    violations: Sequence[SpecViolation],
    *,
    include_satisfied: bool = False,
    title: str = "Spec satisfaction report",
) -> str:
    """Build an LLM-readable text rendering of a violation list.

    Layout (deliberately short — meant to be embedded in a prompt
    template, not viewed as a standalone report):

        Spec satisfaction report (3 of 5 satisfied):
          gain_dc:  measured 55.0 dB, target >= 60.0 dB → DISTANCE 5.0 dB
          gbw:      measured 1.5e+07 Hz, target >= 2e+07 Hz → DISTANCE 5e+06 Hz
          phase_margin: measured 65.0 deg, target >= 60.0 deg → SATISFIED
          power:    measured n/a, target <= 1e-03 W → DISTANCE inf (measure missing)
          icmr_lo:  measured 0.2 V, target in_range [0.1, 0.5] V → SATISFIED

      Total satisfaction loss: 5.0 (3 of 5 specs satisfied, 1 measure missing)

    The text is intentionally redundant about "DISTANCE" vs
    "SATISFIED" so an LLM reading the prompt sees the categorical
    signal at a glance AND the continuous magnitude.

    Args:
      violations: SpecViolation records to render.
      include_satisfied: when False (default), specs already at
        `distance == 0` are collapsed to a single tail line
        `"... and N more satisfied"`. When True, every spec is
        rendered. The default keeps prompt budget tight on cells
        with many trivial passes.
      title: header line (lets a caller emit "Setup A:" instead of
        the default).
    """
    sat_count = sum(1 for v in violations if v.satisfied)
    missing_count = sum(
        1 for v in violations
        if v.measured is None or (
            isinstance(v.distance, float) and math.isinf(v.distance)
        )
    )
    total_loss = compute_satisfaction_loss(
        violations, treat_inf_as=None,
    )
    total_loss_str = (
        "inf" if (isinstance(total_loss, float) and math.isinf(total_loss))
        else f"{total_loss:.4g}"
    )
    lines: list[str] = [
        f"{title} ({sat_count} of {len(violations)} satisfied):"
    ]
    extra_satisfied = 0
    for v in violations:
        unit = _format_unit(v.unit)
        measured_s = _format_number(v.measured) + unit
        if v.op == "in_range":
            target_s = (
                f"in_range [{_format_number(v.lower)}, "
                f"{_format_number(v.target)}]{unit}"
            )
        else:
            target_s = f"{v.op} {_format_number(v.target)}{unit}"
        if v.satisfied:
            if include_satisfied:
                lines.append(
                    f"  {v.spec_name}: measured {measured_s}, "
                    f"target {target_s} -> SATISFIED"
                )
            else:
                extra_satisfied += 1
            continue
        if v.measured is None or (
            isinstance(v.distance, float) and math.isinf(v.distance)
        ):
            lines.append(
                f"  {v.spec_name}: measured {measured_s}, "
                f"target {target_s} -> DISTANCE inf (measure missing)"
            )
            continue
        dist_s = _format_number(v.distance) + unit
        lines.append(
            f"  {v.spec_name}: measured {measured_s}, "
            f"target {target_s} -> DISTANCE {dist_s}"
        )
    if extra_satisfied > 0 and not include_satisfied:
        lines.append(f"  ... and {extra_satisfied} more satisfied")
    tail_bits = [
        f"{sat_count} of {len(violations)} specs satisfied",
    ]
    if missing_count > 0:
        tail_bits.append(f"{missing_count} measure missing")
    lines.append(
        f"  Total satisfaction loss: {total_loss_str} "
        f"({', '.join(tail_bits)})"
    )
    return "\n".join(lines)


# ─── verify_with_distance: verifier-shaped one-call ───────────────


@dataclass
class DistanceVerifyResult:
    """Aggregate result for the distance-aware verify path.

    Holds both the boolean satisfaction summary (so callers that
    previously keyed off `all_passed` still work without change) AND
    the continuous distance signal.

    Fields:
      * `satisfied`             — True iff every violation has
                                  `satisfied=True`.
      * `violations`            — full list of SpecViolation.
      * `satisfaction_loss`     — weighted Σ distance (inf if any
                                  measure was missing).
      * `cell_name`             — echoed back from the underlying
                                  VerifyResult.
      * `verify_result`         — the underlying `VerifyResult` from
                                  `clay.verifier.verify`, for callers
                                  that want the per-assertion text
                                  and category fields.
      * `gradient`              — `{param_name: ∂satisfaction_loss/
                                  ∂param}` when the Clay adjoint bridge
                                  is available, else `None`. This is the
                                  NATIVE "which direction reduces the
                                  violation" signal — a caller gets it
                                  without re-wiring
                                  `compute_satisfaction_gradient`. It is
                                  best-effort: a missing/unbuilt bridge,
                                  a degenerate cell, or any failure in
                                  the adjoint chain leaves it `None` and
                                  nothing else changes.
    """

    satisfied: bool
    violations: list[SpecViolation]
    satisfaction_loss: float
    cell_name: str
    verify_result: Any = None
    gradient: Optional[dict[str, float]] = None

    @property
    def llm_text(self) -> str:
        """Convenience: `format_violation_for_llm(self.violations)`."""
        return format_violation_for_llm(self.violations)


# Canonical spec-name → adjoint target-attr mapping. The bridge's
# `gradient_on_functional_target` exposes a small set of functional-role
# target attributes; this default lets a caller get a NATIVE gradient on
# the common analog specs without hand-writing the mapping. A spec not in
# this map simply contributes no gradient term (the gradient surface
# zero-suppresses it), so the default is safe — it never invents a
# derivative for a spec the adjoint cannot express.
_DEFAULT_SPEC_TO_TARGET_ATTR: dict[str, str] = {
    "gain_dc": "gain_target",
    "gain": "gain_target",
    "gbw": "pole_freq_target",
    "gm": "gm_target",
    "iref": "iref",
}


def _best_effort_native_gradient(
    tb,
    violations: Sequence[SpecViolation],
    *,
    eq_rel_tol: float,
) -> Optional[dict[str, float]]:
    """Compute `∂satisfaction_loss/∂param` via the Clay adjoint bridge.

    Best-effort and NEVER raises: any failure (no functional roles, no
    bridge / unbuilt PyO3 extension, a degenerate OP solve, a missing
    target-attr mapping) returns ``None`` and the caller leaves the
    verdict's ``gradient`` field unset. When the bridge IS available the
    return is ``{role_instance_name: partial_derivative}`` over the
    cell's functional-role instance names.

    The adjoint chain itself lives in
    :func:`compute_satisfaction_gradient` + :func:`make_adjoint_fn_from_bridge`;
    this helper only assembles the inputs (sizing-param names from the
    cell's functional roles, the canonical spec→target-attr mapping) so
    a generic ``verify_with_distance`` caller gets the direction signal
    without re-wiring the gradient surface.
    """
    try:
        cell = getattr(tb, "cell", None)
        if cell is None:
            return None
        roles = list(getattr(cell, "functional_roles", []) or [])
        sizing_params = [
            str(getattr(r, "name", ""))
            for r in roles
            if getattr(r, "name", "")
        ]
        if not sizing_params:
            # No declared functional roles → no param axis to
            # differentiate along. An honest None (no signal), not a
            # zero vector that could read as "flat gradient".
            return None
        # Only attempt the (expensive) adjoint round trip when at least
        # one violation has a non-zero gradient w.r.t. measured — a fully
        # satisfied cell has a zero gradient by construction.
        if all(v.distance == 0.0 for v in violations):
            return {p: 0.0 for p in sizing_params}
        adjoint_fn = make_adjoint_fn_from_bridge(
            cell,
            spec_to_target_attr=_DEFAULT_SPEC_TO_TARGET_ATTR,
            pdk_id=str(getattr(cell, "pdk_id", "sky130")),
        )
        grad = compute_satisfaction_gradient(
            violations, sizing_params, adjoint_fn, eq_rel_tol=eq_rel_tol,
        )
        # If every component is exactly 0.0 the bridge produced no usable
        # derivative (unbuilt extension / no matching target attr). Return
        # None so the field reads as "no signal" rather than "flat".
        if all(g == 0.0 for g in grad.values()):
            return None
        return grad
    except Exception:  # noqa: BLE001 — best-effort, never breaks verify
        return None


def verify_with_distance(
    tb,
    *,
    work_dir: Any = "tmp_clay_rundir",
    driver: Any = None,
    weights: Optional[Mapping[str, float]] = None,
    eq_rel_tol: float = EQ_RELATIVE_TOLERANCE,
    treat_inf_as: Optional[float] = None,
) -> DistanceVerifyResult:
    """One-call verifier surface that returns both boolean + distance.

    This is the v0.1.01 §5.2 entry point — call it instead of
    `clay.verifier.verify` when an LLM RL loop needs a continuous
    reward signal.

    The v0.0.x `verify()` API is preserved byte-exact: this function
    calls into `clay.verifier.verify` to drive the simulator, picks
    the resulting `measures` dict apart, and builds SpecViolation
    records against `tb.cell.spec.fields`.

    Args:
      tb: a `clay.ir.Testbench` (same input shape as `verify`).
      work_dir / driver: forwarded to `verify` verbatim.
      weights: optional `{spec_name: weight}` map (uniform if None).
      eq_rel_tol: relative-tolerance band for `=` ops.
      treat_inf_as: replacement value for `math.inf` in the
        aggregate loss (default None: inf propagates).

    Returns:
      `DistanceVerifyResult` carrying both the boolean and the
      continuous signal.
    """
    # Lazy import keeps this module importable on environments where
    # the verifier's transitive deps are heavier than needed for the
    # distance-only consumers (e.g. RL prompt-formatter unit tests).
    from .verifier import verify as _verify

    verify_result = _verify(tb, work_dir=work_dir, driver=driver)
    measures = dict(verify_result.measures) if verify_result.measures else {}
    spec_fields = list(tb.cell.spec.fields) if tb.cell.spec else []
    violations = evaluate_spec_distance(
        spec_fields, measures,
        weights=weights, eq_rel_tol=eq_rel_tol,
    )
    loss = compute_satisfaction_loss(
        violations, treat_inf_as=treat_inf_as,
    )
    satisfied = all(v.satisfied for v in violations)
    # Native adjoint gradient (best-effort; None when the bridge / a
    # functional-role axis is unavailable). This gives a caller the
    # "which direction reduces the violation" signal as a verdict field
    # without re-wiring `compute_satisfaction_gradient`.
    gradient = _best_effort_native_gradient(
        tb, violations, eq_rel_tol=eq_rel_tol,
    )
    result = DistanceVerifyResult(
        satisfied=satisfied,
        violations=violations,
        satisfaction_loss=loss,
        cell_name=tb.cell.name,
        verify_result=verify_result,
        gradient=gradient,
    )
    # Episode-spine hook: append an EpisodeRecord to the process-level
    # store (default-ON, env-gated by CLAY_RECORD_EPISODES). This is
    # contractually side-effect-free on the verdict — `record_episode`
    # swallows every failure and returns None, so the verdict object
    # returned below is byte-identical whether or not recording ran.
    try:
        from .episode import record_episode as _record_episode

        _record_episode(result)
    except Exception:  # noqa: BLE001 — recording must never break verify
        pass
    return result


# ─── Adjoint adapter (optional, lazy-imported bridge) ─────────────


def make_adjoint_fn_from_bridge(
    cell_or_handle: Any,
    spec_to_target_attr: Mapping[str, str],
    *,
    in_node: str = "in",
    out_node: str = "out",
    supply_vdd: float = 1.8,
    pdk_id: str = "sky130",
    step_rel: float = 0.05,
) -> AdjointFn:
    """Build an `AdjointFn` over Clay's `_ir_bridge` adjoint API.

    The bridge's `gradient_on_functional_target(role_name, target_attr,
    ...)` returns `∂(DC gain)/∂(target_attr)` — i.e. the measured
    quantity is *fixed at DC gain*, and the parameter is one of the
    canonical functional-role target attributes (`gm_target`,
    `gain_target`, `pole_freq_target`, `iref`). To plug this into the
    spec-distance gradient surface we need to:

      1. Map a spec name (e.g. `gain_dc`) to the target attribute the
         adjoint actually exposes. The caller provides this mapping
         via `spec_to_target_attr` — e.g.
         `{"gain_dc": "gain_target"}` for a cell whose output_buffer
         role carries the gain target.
      2. Map the sizing-param name to the role name (the function
         expects the param to be the role's instance name, e.g.
         `"Mdp"` for an `input_transconductor Mdp`).

    Returns a callable suitable for `compute_satisfaction_gradient`.
    A missing entry in `spec_to_target_attr` (or any failure in the
    bridge call) returns None — the gradient surface zero-suppresses
    that term so a missing adjoint doesn't poison the whole sum.

    The bridge is lazy-imported so `clay.distance_to_sat` can be
    imported on extension-less Python environments without errors.
    """
    try:
        from . import _ir_bridge
    except ImportError:
        # Bridge unavailable — return a constant-None callback so the
        # gradient surface yields a zero vector. The caller's LLM RL
        # loop sees "no gradient signal" instead of a hard crash.
        def _zero_fn(_spec: str, _param: str) -> Optional[float]:
            return None
        return _zero_fn

    def _fn(spec_name: str, param_name: str) -> Optional[float]:
        target_attr = spec_to_target_attr.get(spec_name)
        if target_attr is None:
            return None
        try:
            grad_dict = _ir_bridge.gradient_on_functional_target(
                cell_or_handle,
                param_name,
                target_attr,
                in_node,
                out_node,
                supply_vdd=supply_vdd,
                pdk_id=pdk_id,
                step_rel=step_rel,
            )
        except Exception:  # noqa: BLE001
            return None
        if grad_dict is None:
            return None
        if not grad_dict.get("success", False):
            return None
        g = grad_dict.get("gradient")
        if g is None or not isinstance(g, (int, float)):
            return None
        gf = float(g)
        if not math.isfinite(gf):
            return None
        return gf

    return _fn


# ─── v0.1.05 Round 81 Agent C: distance × counterfactual composition ─


def _violation_distance_map(
    violations: Sequence[SpecViolation],
) -> dict[str, float]:
    """Return ``{spec_name: distance}`` for a list of SpecViolation.

    Helper used by :func:`compose_with_counterfactual`.  The
    counterfactual side already builds a similar map (with case-folded
    and canonical-bridge-metric aliases) — this one is intentionally
    minimal and returns only the verbatim ``spec_name`` keys so a
    caller wiring a custom distance source has a deterministic surface.

    Non-finite distances pass through (``math.inf`` from a missing
    measure stays ``math.inf``); the counterfactual side handles the
    clamp via its ``inf_replacement`` kwarg.
    """
    out: dict[str, float] = {}
    for v in violations:
        if not isinstance(v, SpecViolation):
            continue
        out[str(v.spec_name)] = float(v.distance) if v.distance is not None else math.inf
    return out


# ─── v0.1.15 BL-124 R92 F: distance weight lookup for trace sorter ───


def compute_metric_weight(
    metric_name: str,
    violations: Sequence[SpecViolation],
    *,
    inf_replacement: float = 1e9,
) -> float:
    """Return the distance-to-sat weight for a metric, with alias resolution.

    This is the helper :mod:`clay.llm_counterfactual_trace` calls when it
    needs to sort counterfactual-trace entries by "how out of spec is the
    metric this entry helps".  The sorter cares about magnitude — entries
    with metric_name in violations get the max distance over violations
    matching that metric (after R78 B alias normalisation
    ``gain ↔ gain_dc`` etc.).

    The function does NOT compute a new distance — it reads the
    pre-computed :attr:`SpecViolation.distance` already on the input
    records.  That keeps the trace-sorter path cheap (no re-evaluation
    of spec satisfaction) and guarantees byte-exact agreement with
    :func:`evaluate_spec_distance` upstream.

    Parameters
    ----------
    metric_name
        The spec metric the trace entry talks about (e.g. ``"gain_dc"``,
        ``"pm"``, ``"gbw"``, or a spec-side alias like ``"gain"``).
        Case-insensitive after alias resolution.
    violations
        Sequence of :class:`SpecViolation` records (typically the output
        of :func:`evaluate_spec_distance`).
    inf_replacement
        Finite replacement for ``math.inf`` distance (a missing measure
        produces ``math.inf``).  Default ``1e9`` — large enough to dwarf
        any finite distance, small enough to keep Python sort orders
        well-defined.  Set to ``math.inf`` to let the raw inf dominate.

    Returns
    -------
    float
        Non-negative weight.  ``0.0`` means "metric not in violations OR
        the spec is satisfied".  Larger values mean "metric is farther
        from satisfaction; surface this row first in a trace".

    Tie-breaker policy
    ------------------
    When the same canonical metric appears in multiple violations (e.g.
    a multi-corner Spec where ``gain_dc@tt`` and ``gain_dc@ss`` both
    map to ``gain_dc``), the function returns the MAX distance — the
    worst corner drives the priority.  Sum / average / median are
    reasonable alternatives; max is the conservative one (RL-style
    "worst-case credit assignment").

    Byte-exact note
    ---------------
    Does NOT modify :func:`compose_with_counterfactual`,
    :func:`evaluate_spec_distance`, or any R78 A surface.  Pure
    look-up; never re-computes a SpecViolation field.
    """
    # Lazy import dodges a circular import on the rare path where a
    # downstream module imports this helper from within
    # clay.llm_counterfactual_trace at module-load time.
    from .llm_counterfactual_trace import (
        _build_distance_map,
        _lookup_distance,
        _normalize_distance_for_score,
    )

    distance_map = _build_distance_map(violations)
    raw = _lookup_distance(distance_map, metric_name)
    return _normalize_distance_for_score(
        raw, inf_replacement=inf_replacement,
    )


def compose_with_counterfactual(
    violations: Sequence[SpecViolation],
    counterfactual_entries: Sequence[Any],
    *,
    top_k: int = 8,
    abs_floor: float = 0.0,
    weight_mode: str = "distance",
    inf_replacement: float = 1e9,
) -> list[Any]:
    """Rank ``counterfactual_entries`` by ``|gradient| × distance_to_sat``.

    The Tier 2 LLM-extrapolation §5 spec-algebra seed: compose the
    R78 A distance signal (this module) with the R78 B counterfactual
    trace (sister module :mod:`clay.llm_counterfactual_trace`).  The
    counterfactual side ranks knobs by gradient magnitude alone — "this
    knob has the most leverage".  This composition adds the distance
    factor so the LLM sees "this knob has the most leverage on the
    metric that is *farthest from satisfaction*" — i.e. the most
    actionable knob given the current spec gap.

    Parameters
    ----------
    violations
        Sequence of :class:`SpecViolation` (the output of
        :func:`evaluate_spec_distance` or any downstream filter).  Only
        the ``spec_name`` and ``distance`` attributes are read; the
        function does NOT re-evaluate spec satisfaction.
    counterfactual_entries
        Sequence of :class:`clay.llm_counterfactual_trace.CounterfactualEntry`
        (the output of :func:`compute_counterfactual_trace`).  Treated
        as a duck-typed object to avoid a circular import — the
        composition reads only ``reason`` / ``derivative_value`` /
        ``spec_metric`` / ``param_name``.
    top_k / abs_floor / weight_mode / inf_replacement
        Forwarded verbatim to
        :func:`clay.llm_counterfactual_trace.select_top_k_by_distance`.

    Returns
    -------
    list
        The same shape :func:`select_top_k_by_distance` returns — a
        list of :class:`CounterfactualEntry`, descending by composite
        score.  Empty list when no actionable entry remains.

    Byte-exact note
    ---------------
    Does NOT modify :func:`compute_satisfaction_loss`,
    :func:`evaluate_spec_distance`, :func:`compute_spec_distance`, or
    any other R78 A surface.  Consume-only: the function reads the
    SpecViolation records but never mutates them, and the
    counterfactual entries are passed through to the sister module
    untouched.
    """
    # Lazy import dodges the (currently impossible) circular import if
    # a future refactor moves SpecViolation into llm_counterfactual_trace.
    from .llm_counterfactual_trace import select_top_k_by_distance

    return select_top_k_by_distance(
        counterfactual_entries,
        violations,
        top_k=top_k,
        abs_floor=abs_floor,
        weight_mode=weight_mode,
        inf_replacement=inf_replacement,
    )
