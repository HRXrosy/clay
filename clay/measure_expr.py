"""
Clay neutral measure expression — AST + parser + emitter helpers.

A Clay `Measure(name, analysis_ref, expr)` carries a SIMULATOR-NEUTRAL
expression string. Each driver translates this expression to its own
simulator dialect (ngspice `.meas`, hspice `.measure`, spectre
`expression`, etc.). Before v0.0.18, the ngspice driver used 3 regex
patterns to match a handful of fixed expression forms — fragile and not
extensible.

v0.0.18 replaced that with a proper grammar-based parser. The post-v0.0.23
extension closes AnalogGym gap G5 by adding aggregate operators.

  Supported expression forms:

    SIGNAL                       — measure the value of SIGNAL
    SIGNAL @ AXIS = VALUE        — value of SIGNAL when AXIS hits VALUE
    SIGNAL when SIGNAL = VALUE   — value of LHS when RHS equals VALUE
    AXIS when SIGNAL = VALUE     — value of AXIS where SIGNAL crosses VALUE
    OP(SIGNAL)                   — aggregate of SIGNAL over the analysis
                                   axis (OP ∈ max/min/avg/pp/rms)
    OP(SIGNAL) over AXIS from X to Y
                                 — aggregate of SIGNAL restricted to
                                   AXIS ∈ [X, Y]; X / Y may carry an SI
                                   suffix (Hz/kHz/MHz/GHz/ s/ms/us/ns/ps).

  SIGNAL can be:
    vdb(node)                    — node voltage in dB
    v(node)                      — node voltage (real)
    vp(node)                     — node phase
    i(name)                      — branch current

  AXIS is a special keyword for the analysis sweep axis:
    freq   — AC analysis frequency
    time   — transient time
    dc_sweep — DC sweep variable

  Aggregate operators map to ngspice `.meas` primitives:

    MAX / MIN / AVG / PP / RMS  — native `.meas` primitives, applied
                                   along the analysis axis with optional
                                   FROM=/TO= clauses.

    SUM                          — without `over AXIS from X to Y`,
                                   interpreted as identity (read the
                                   operand value at axis=0). With an
                                   `over` clause, emitted as ngspice's
                                   `INTEG <signal> FROM=X TO=Y` — the
                                   integral over the sweep axis is the
                                   continuous analogue of a discrete
                                   sum, and is what batch-mode ngspice
                                   actually computes for SUM-like
                                   queries.

    SQRT                         — emitted as the bare operand (no
                                   aggregation at the `.meas` level),
                                   then post-processed in Python by
                                   `math.sqrt`. A negative raw value
                                   yields `nan` and a warning; the
                                   downstream verifier reports this as
                                   `measure_missing` so the LLM sees a
                                   clear "this branch is invalid"
                                   signal rather than silent NaN
                                   propagation.

  Both SUM (without `over`) and SQRT need a Python post-processing
  pass after the simulator returns. The post-processing function
  `apply_post_processing` below takes the raw value + the AST and
  returns the final scalar; the verifier calls it once per measure.

  Nested aggregates (`sqrt(sum(...))`, `max(sqrt(...))`, etc.) are
  REJECTED in this version with a clear "not supported" message.
  Adding nesting requires the emitter to plumb a temporary node into
  the netlist (B-source for sqrt, INTEG vector for sum) before the
  outer aggregate can read it — out of scope for v0.0.30.

The parsed AST (a small tree of dataclasses) is what drivers consume.
Drivers translate the AST to their dialect — no regex required.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Union


# ─── AST nodes ─────────────────────────────────────────────────

@dataclass
class Signal:
    """A measurable signal: vdb(out), v(net1), vp(out), i(vdd)."""
    func: str       # 'vdb' | 'v' | 'vp' | 'i'
    arg: str        # node or device name


@dataclass
class Axis:
    """The analysis sweep axis (freq / time / dc_sweep)."""
    name: str


@dataclass
class AtCondition:
    """`Expr @ axis = value` — Expr's value where the sweep axis = value."""
    target: Signal              # the signal/quantity being measured
    axis: Axis
    value: float


@dataclass
class WhenCondition:
    """`Expr when Signal = value` — Expr's value where Signal crosses value."""
    target: Union[Signal, Axis]   # the quantity to read off (signal OR axis)
    when_signal: Signal           # the signal we're matching on
    value: float


# Native ngspice `.meas` aggregate operators. These are emitted as
# `meas <analysis> NAME OP <signal> [FROM=lo TO=hi]` and ngspice handles
# the aggregation itself. String values match the ngspice tokens.
_NATIVE_AGGREGATE_OPS: frozenset[str] = frozenset(
    {"MAX", "MIN", "AVG", "PP", "RMS"}
)

# Post-processed aggregate operators. ngspice has no direct `.meas`
# primitive for these, so the driver emits a different `.meas` (or the
# bare operand) and the verifier applies `apply_post_processing` after
# the raw value comes back from the log parser.
#
#   SUM   — emitted as INTEG when an `over AXIS from X to Y` clause is
#           present (ngspice's INTEG line shape: `meas <analysis> NAME
#           INTEG <signal> [FROM=lo TO=hi]`). Without the `over`
#           clause, the operand is read at axis=0 and returned as-is
#           (sum of a single point ≡ that point's value).
#   SQRT  — emitted as the bare operand (`find <signal> at=0`); Python
#           takes the sqrt afterwards.
_POST_PROCESSED_AGGREGATE_OPS: frozenset[str] = frozenset({"SUM", "SQRT"})

# All aggregate operators the parser recognizes (native ∪ post-processed).
# The parser uses this set to decide what `OP(...)` heads are aggregates.
AGGREGATE_OPS: frozenset[str] = _NATIVE_AGGREGATE_OPS | _POST_PROCESSED_AGGREGATE_OPS


@dataclass
class Aggregate:
    """`op(SIGNAL)` — aggregate of SIGNAL over an analysis axis.

    `op` is one of MAX / MIN / AVG / PP / RMS (uppercase canonical form).
    `axis` and `from_val`/`to_val` are populated when the source carries
    an explicit `over <axis> from X to Y` clause; otherwise None, and
    the emitter aggregates over the full analysis sweep.
    """
    op: str                       # canonical uppercase: MAX|MIN|AVG|PP|RMS
    operand: Signal               # the signal to aggregate
    axis: Optional[Axis] = None         # explicit constraint axis (or None)
    from_val: Optional[float] = None    # lower bound on axis (SI value)
    to_val: Optional[float] = None      # upper bound on axis (SI value)


# Top-level expression is one of these
MeasureExpr = Union[Signal, AtCondition, WhenCondition, Aggregate]


# ─── Parser ────────────────────────────────────────────────────

# Regex helpers for tokenization
_RE_SIGNAL  = re.compile(r"^(vdb|v|vp|i)\s*\(\s*([A-Za-z_][\w.]*)\s*\)$")
_RE_AXIS    = re.compile(r"^(freq|time|dc_sweep)$")
_RE_FLOAT   = re.compile(r"^-?[\d.]+(?:[eE][-+]?\d+)?$")

# Aggregate operators: case-insensitive name token followed by '('.
# We can't use a single non-greedy regex here because the inner operand
# may itself contain parentheses (e.g. `max(vdb(out))`), so we use a
# tiny custom matcher (_split_agg_head) that tracks paren nesting.
# The head regex just captures the OP name and confirms the leading
# `OP (` shape.
_RE_AGG_NAME = re.compile(r"^([A-Za-z_]+)\s*\(")

# `over <axis> from X to Y` trailing clause. The numbers may carry one
# of the SI suffixes listed in _MEAS_UNITS below.
_RE_OVER_CLAUSE = re.compile(
    r"^over\s+(\w+)\s+from\s+(\S+)\s+to\s+(\S+)\s*$",
    re.IGNORECASE,
)

# SI suffix multipliers recognized by the aggregate parser's range
# bounds. This is a compact subset of the spec-layer unit table —
# enough for the axes aggregates can run over (freq, time). Keeping
# the table local avoids importing clay.parser (and lark) from this
# self-contained sublanguage module.
_MEAS_UNITS: dict[str, float] = {
    "":    1.0,
    # Frequency
    "Hz":  1.0,    "kHz": 1e3,  "MHz": 1e6,  "GHz": 1e9,
    # Time
    "s":   1.0,    "ms":  1e-3, "us":  1e-6, "ns":  1e-9, "ps": 1e-12,
}
# Match a number with optional unit suffix (e.g. `1Hz`, `1.5kHz`, `5e-9s`, `100`).
_RE_NUM_UNIT = re.compile(
    r"^(?P<num>-?[\d.]+(?:[eE][-+]?\d+)?)(?P<unit>[A-Za-z]*)$"
)


class MeasureParseError(ValueError):
    """Raised when a Clay measure expression cannot be parsed."""


def _parse_si_number(text: str) -> float:
    """Parse a numeric literal with optional SI suffix. Used by the
    aggregate `from X to Y` clause. Raises MeasureParseError on garbage."""
    s = text.strip()
    m = _RE_NUM_UNIT.match(s)
    if not m:
        raise MeasureParseError(
            f"expected number (optionally with SI suffix), got: {text!r}"
        )
    num_part = m.group("num")
    unit = m.group("unit")
    mult = _MEAS_UNITS.get(unit)
    if mult is None:
        raise MeasureParseError(
            f"unknown SI suffix {unit!r} in measure range bound {text!r}. "
            f"Allowed: {sorted(k for k in _MEAS_UNITS if k)}."
        )
    try:
        return float(num_part) * mult
    except ValueError:
        raise MeasureParseError(
            f"could not parse number in measure range bound: {text!r}"
        )


def _parse_token(token: str) -> Union[Signal, Axis, float]:
    token = token.strip()
    m = _RE_SIGNAL.match(token)
    if m:
        return Signal(func=m.group(1), arg=m.group(2))
    if _RE_AXIS.match(token):
        return Axis(name=token)
    if _RE_FLOAT.match(token):
        return float(token)
    raise MeasureParseError(f"unrecognized measure token: {token!r}")


def _split_agg_head(s: str) -> Optional[tuple[str, str, str]]:
    """Split `OP ( ...balanced... ) tail` into (op, inner, tail).

    Returns None if the string doesn't even start with an identifier-
    followed-by-`(`. Raises MeasureParseError if the leading `(` is
    never balanced — that means a malformed aggregate.
    """
    m = _RE_AGG_NAME.match(s)
    if not m:
        return None
    op_raw = m.group(1)
    # Position just past the opening `(`.
    i = m.end()
    depth = 1
    n = len(s)
    while i < n and depth > 0:
        c = s[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                inner = s[m.end():i]
                tail = s[i + 1:]
                return op_raw, inner, tail
        i += 1
    raise MeasureParseError(
        f"unbalanced parentheses in expression: {s!r}"
    )


def _try_parse_aggregate(s: str) -> Optional[Aggregate]:
    """Try to parse `OP(SIGNAL)` or `OP(SIGNAL) over AXIS from X to Y`.

    Returns the Aggregate AST on success, None if the expression does
    not start with a recognized aggregate operator. Raises
    MeasureParseError if the expression starts with `OP(` but the rest
    is malformed.
    """
    split = _split_agg_head(s)
    if split is None:
        return None
    op_raw, inner, tail = split
    inner = inner.strip()
    tail = tail.strip()

    op_upper = op_raw.upper()
    # If the head token is not an aggregate name, this isn't an aggregate
    # — fall back so the caller can try Signal / AtCondition / WhenCondition.
    if op_upper not in AGGREGATE_OPS:
        # Could be a Signal token like `v(out)` — let the caller fall
        # through. `vdb`, `v`, `vp`, `i` would hit this branch.
        return None

    # Reject nested aggregates (`sqrt(max(...))`, `sum(rms(...))`, etc.).
    # v0.0.30 only handles single-level aggregates: the inner must parse
    # as a plain Signal. Allowing nesting needs the emitter to materialize
    # an intermediate vector in the netlist before the outer op can read
    # it — out of scope for this release.
    inner_split = _split_agg_head(inner)
    if inner_split is not None and inner_split[0].upper() in AGGREGATE_OPS:
        raise MeasureParseError(
            f"nested aggregate {op_raw}({inner_split[0]}(...)) is not "
            f"supported. v0.0.30 only handles single-level aggregates; "
            f"nesting would require materializing the inner result as a "
            f"netlist signal first. Rewrite as two separate measures or "
            f"wait for nested-aggregate support."
        )

    # The inner text must parse as a Signal.
    sig = _parse_token(inner)
    if not isinstance(sig, Signal):
        raise MeasureParseError(
            f"aggregate {op_raw}() operand must be a signal like "
            f"vdb(out) / v(out) / vp(out) / i(name), got: {inner!r}"
        )

    if not tail:
        return Aggregate(op=op_upper, operand=sig)

    # Optional `over AXIS from X to Y` clause.
    om = _RE_OVER_CLAUSE.match(tail)
    if not om:
        raise MeasureParseError(
            f"aggregate {op_raw}() has unrecognized trailing text: "
            f"{tail!r}. Expected `over <axis> from <num> to <num>`."
        )
    axis_text = om.group(1)
    axis_tok = _parse_token(axis_text)
    if not isinstance(axis_tok, Axis):
        raise MeasureParseError(
            f"aggregate range axis must be freq/time/dc_sweep, got: {axis_text!r}"
        )
    lo = _parse_si_number(om.group(2))
    hi = _parse_si_number(om.group(3))
    return Aggregate(op=op_upper, operand=sig, axis=axis_tok,
                     from_val=lo, to_val=hi)


def parse(expr: str) -> MeasureExpr:
    """Parse a Clay measure expression into the AST.

    Examples accepted:
      'vdb(out) @ freq = 1'
      'freq when vdb(out) = 0'
      'vp(out) when vdb(out) = 0'
      'v(out) @ time = 100e-9'
      'i(vdd) when time = 5e-6'
      'vdb(out)'                            # plain signal (no condition)
      'max(vdb(out))'                       # aggregate over analysis axis
      'pp(v(out))'                          # peak-to-peak over analysis axis
      'max(vdb(out)) over freq from 1Hz to 1kHz'  # constrained aggregate
    """
    s = expr.strip()
    if not s:
        raise MeasureParseError("empty measure expression")

    # Try aggregate form first — `OP(SIGNAL) [over AXIS from X to Y]`.
    # This must come before the @/when checks so an aggregate's trailing
    # `over ... from ... to ...` clause is not misread.
    agg = _try_parse_aggregate(s)
    if agg is not None:
        return agg

    # Try `Expr when SIGNAL = value`
    if " when " in s:
        left, _, right = s.partition(" when ")
        # right: SIGNAL = value
        if "=" not in right:
            raise MeasureParseError(
                f"`when` clause missing '=': {expr!r}"
            )
        sig_text, _, val_text = right.partition("=")
        sig = _parse_token(sig_text)
        val = _parse_token(val_text)
        if not isinstance(sig, Signal):
            raise MeasureParseError(
                f"`when` RHS must be a signal like vdb(node), got: {sig_text!r}"
            )
        if not isinstance(val, float):
            raise MeasureParseError(
                f"`when` value must be a number, got: {val_text!r}"
            )
        target = _parse_token(left)
        if not isinstance(target, (Signal, Axis)):
            raise MeasureParseError(
                f"`when` LHS must be a signal or axis, got: {left!r}"
            )
        return WhenCondition(target=target, when_signal=sig, value=val)

    # Try `SIGNAL @ AXIS = value`
    if "@" in s:
        left, _, right = s.partition("@")
        target = _parse_token(left)
        if not isinstance(target, Signal):
            raise MeasureParseError(
                f"`@` LHS must be a signal, got: {left!r}"
            )
        if "=" not in right:
            raise MeasureParseError(
                f"`@` clause missing '=': {expr!r}"
            )
        axis_text, _, val_text = right.partition("=")
        axis = _parse_token(axis_text)
        val = _parse_token(val_text)
        if not isinstance(axis, Axis):
            raise MeasureParseError(
                f"`@` axis must be one of freq/time/dc_sweep, got: {axis_text!r}"
            )
        if not isinstance(val, float):
            raise MeasureParseError(
                f"`@` value must be a number, got: {val_text!r}"
            )
        return AtCondition(target=target, axis=axis, value=val)

    # Plain signal
    sig = _parse_token(s)
    if isinstance(sig, Signal):
        return sig
    raise MeasureParseError(f"expected a signal or conditional, got: {expr!r}")


# Backwards-compat alias. The ngspice driver and the AnalogGym-facing
# example code both import this name; we also expose `parse` itself for
# legacy callers.
parse_measure_expr = parse


# ─── ngspice emitter ──────────────────────────────────────────

def emit_ngspice_meas(name: str, analysis_ref: str, ast: MeasureExpr) -> str:
    """Convert the neutral AST to one ngspice `.meas` line.

    name: target variable name for .meas output
    analysis_ref: which analysis to draw from ('ac', 'dc', 'tran', 'noise')
    ast: parsed neutral expression
    """
    head = f"meas {analysis_ref} {name}"
    if isinstance(ast, Signal):
        return f"{head} find {_signal_to_ngspice(ast)} at=0"
    if isinstance(ast, AtCondition):
        axis_param = _axis_at(ast.axis)
        return (
            f"{head} find {_signal_to_ngspice(ast.target)} "
            f"{axis_param}={ast.value:g}"
        )
    if isinstance(ast, WhenCondition):
        # find <target> when <sig> = <val>
        # If the target is an Axis (e.g. freq), ngspice doesn't take
        # `find freq` — it implicitly returns the axis on a `when` clause:
        #     meas ac gbw when vdb(out)=0
        sig_s = _signal_to_ngspice(ast.when_signal)
        if isinstance(ast.target, Axis):
            return f"{head} when {sig_s}={ast.value:g}"
        # target is a Signal: `meas ac name find vp(out) when vdb(out)=0`
        return f"{head} find {_signal_to_ngspice(ast.target)} when {sig_s}={ast.value:g}"
    if isinstance(ast, Aggregate):
        # Native aggregates (MAX/MIN/AVG/PP/RMS) emit the ngspice .meas
        # aggregate form directly:
        #     meas <analysis> <name> <OP> <signal> [FROM=<lo> TO=<hi>]
        # Bounds (when present) are applied along the analysis's natural
        # axis. ngspice itself doesn't accept an axis keyword on the
        # aggregate line — the axis is implicit to the analysis (freq
        # for ac, time for tran, the sweep variable for dc).
        sig_s = _signal_to_ngspice(ast.operand)
        if ast.op in _NATIVE_AGGREGATE_OPS:
            line = f"{head} {ast.op} {sig_s}"
            if ast.from_val is not None and ast.to_val is not None:
                line += f" FROM={ast.from_val:g} TO={ast.to_val:g}"
            return line
        # Post-processed aggregates.
        if ast.op == "SUM":
            # SUM with `over` clause → INTEG (integral is the continuous
            # analogue of a discrete sum; ngspice computes it natively).
            # Without `over` it degenerates to identity → emit a `find`
            # at axis=0. The verifier's post-processing then returns
            # the raw value unchanged.
            if ast.from_val is not None and ast.to_val is not None:
                return (
                    f"{head} INTEG {sig_s} "
                    f"FROM={ast.from_val:g} TO={ast.to_val:g}"
                )
            return f"{head} find {sig_s} at=0"
        if ast.op == "SQRT":
            # SQRT measures the bare operand at axis=0. The verifier's
            # post-processing applies math.sqrt(raw). An `over` clause
            # on sqrt is not meaningful (sqrt is a pointwise op, not
            # an axis aggregate); the parser already accepts it
            # syntactically, so we honour the bound by using the
            # operand's value at the lower bound `FROM` if provided.
            if ast.from_val is not None:
                return f"{head} find {sig_s} at={ast.from_val:g}"
            return f"{head} find {sig_s} at=0"
        raise NotImplementedError(
            f"emit_ngspice_meas: aggregate op {ast.op!r} is in "
            f"AGGREGATE_OPS but has no emitter branch"
        )
    raise NotImplementedError(f"emit_ngspice_meas: unhandled AST node {type(ast).__name__}")


def _signal_to_ngspice(sig: Signal) -> str:
    # ngspice accepts vdb(), v(), vp(), i() directly
    return f"{sig.func}({sig.arg})"


def _axis_at(axis: Axis) -> str:
    """ngspice's @ axis kwarg name."""
    if axis.name == "freq":
        return "at"          # `at=f` for ac
    if axis.name == "time":
        return "at"          # `at=t` for tran
    if axis.name == "dc_sweep":
        return "at"
    return "at"


# ─── Post-processing for SUM / SQRT (v0.0.30) ──────────────────

import math as _math
import warnings as _warnings


def needs_post_processing(ast: MeasureExpr) -> bool:
    """Return True if `ast` requires a Python post-processing pass after
    the raw `.meas` value is collected from the simulator log.

    Currently the only post-processed AST shape is an Aggregate whose
    `op` is in `_POST_PROCESSED_AGGREGATE_OPS` (SUM / SQRT). For SUM the
    post-processing is identity when an `over` clause was present (the
    emitter already used INTEG so ngspice did the work), and identity
    when it was absent (sum of a single value ≡ that value). For SQRT
    the post-processing is `math.sqrt(raw)` with NaN handling for
    negative inputs.
    """
    return isinstance(ast, Aggregate) and ast.op in _POST_PROCESSED_AGGREGATE_OPS


def apply_post_processing(ast: MeasureExpr, raw: float, *,
                          measure_name: str | None = None) -> float:
    """Apply the Python-side post-processing step for `ast` to `raw`.

    `raw` is the scalar value the driver pulled out of the simulator
    log for the emitted `.meas` line. For native aggregates / non-
    aggregates this function returns `raw` unchanged. For SUM / SQRT
    it applies the documented transformation:

      * SUM, no `over` clause   → identity (sum of one value = value)
      * SUM, with `over` clause → identity (INTEG already done)
      * SQRT, raw >= 0          → math.sqrt(raw)
      * SQRT, raw <  0          → NaN, with a UserWarning. The verifier
                                  treats NaN as `measure_missing` so the
                                  LLM sees a clear "invalid branch"
                                  signal rather than silently rolling
                                  forward with a NaN.

    `measure_name` is used only for the warning message — it lets the
    log line point at the right .clay measure declaration. Pass None
    when calling from a context that doesn't know the name (rare;
    every driver-fed call has it).
    """
    if not isinstance(ast, Aggregate):
        return raw
    if ast.op not in _POST_PROCESSED_AGGREGATE_OPS:
        return raw
    if ast.op == "SUM":
        # Both forms (with and without `over`) are identity at the
        # post-processing layer — the emitter took care of the actual
        # work (INTEG when `over` was present; `find at=0` otherwise).
        return raw
    if ast.op == "SQRT":
        if raw < 0.0 or _math.isnan(raw):
            tag = f"{measure_name!r} " if measure_name else ""
            _warnings.warn(
                f"measure {tag}— sqrt({raw!r}) is not a real number; "
                f"returning NaN. The verifier will mark this assertion "
                f"as measure_missing.",
                category=UserWarning,
                stacklevel=2,
            )
            return float("nan")
        return _math.sqrt(raw)
    # Unreachable — both ops covered above.
    raise NotImplementedError(
        f"apply_post_processing: aggregate op {ast.op!r} is in "
        f"_POST_PROCESSED_AGGREGATE_OPS but has no implementation"
    )
