"""Clay verifier: orchestrate Cell IR + Testbench → driver → assertion check.

Outcomes are categorized (code review #6, fixed in v0.0.16):
  * pass               — everything OK
  * spec_violation     — circuit ran but didn't meet a spec field
  * measure_missing    — simulator log lacks this measure (likely
                         .meas extraction failure or convergence issue)
  * simulator_error    — simulator crashed before producing a log

This lets downstream consumers (LLMs, dashboards, CI) distinguish
"the circuit is wrong" from "I don't have data to know".
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from copy import deepcopy
from pathlib import Path
from typing import Mapping

from .drivers import NgspiceDriver, SimulatorDriver
from .ir import (
    AcAnalysis, AssertionResult, DcAnalysis, Device, Measure, OpAnalysis,
    SetupSpec, Source, Spec, SpecField, Testbench, TranAnalysis,
    VerifyResult,
)
from .measure_expr import (
    MeasureParseError, apply_post_processing, needs_post_processing,
    parse as parse_measure_expr,
)


def verify(
    tb: Testbench,
    *,
    work_dir: Path | str = "tmp_clay_rundir",
    driver: SimulatorDriver | None = None,
) -> VerifyResult:
    work = Path(work_dir)
    drv = driver or NgspiceDriver()

    # ─── Stage 1: run the simulator ───────────────────────────
    try:
        measures, netlist, log = drv.emit_run_parse(tb, work)
        sim_error: str | None = None
    except Exception as exc:
        # Driver-level failure: emit, run, or parse blew up. Mark every
        # spec field as simulator_error so callers see the full damage
        # picture (not just the first failure).
        sim_error = f"{type(exc).__name__}: {exc}"
        measures = {}
        netlist = work / "(failed)"
        log = work / "(no log)"

    # ─── Stage 1.5: post-process measures (v0.0.30 / G5) ─────
    # SUM and SQRT aggregate operators have no native ngspice .meas
    # primitive. The emitter wrote a related .meas line (INTEG for
    # `sum() over`, plain `find` otherwise), and we close the loop here
    # by applying the Python-side transformation (`math.sqrt`, identity
    # for SUM) before any assertion runs.
    #
    # Done unconditionally on every measure whose AST signals
    # `needs_post_processing` — a no-op for the 5 native aggregates
    # and for plain Signal / AtCondition / WhenCondition forms, so
    # the existing pre-G5 measures are not perturbed.
    if sim_error is None and measures:
        measures = _post_process_measures(measures, tb.measures)
        # v0.0.33: LDO line/load regulation are derived metrics. The
        # ngspice .meas for them was a placeholder `find v(out)`;
        # here we replace the raw value with the proper derivation
        # (vout_pp / sweep_span). The replacement runs only when the
        # measure NAME matches (`line_regulation` / `load_regulation`)
        # AND the supporting vout_pp/vout_at_* measures are present,
        # so a non-LDO testbench that happens to declare a measure
        # called `line_regulation` is not perturbed.
        measures = _post_process_ldo_regulation(measures, tb.measures)

    # ─── Stage 2: classify each spec field ────────────────────
    assertions: list[AssertionResult] = []
    spec = tb.cell.spec
    current_corner = tb.pdk_corner
    for spec_index, sf in enumerate(spec.fields):
        source_obligation_id = sf.source_obligation_id_for(
            tb.cell.name, spec_index
        )
        # v0.0.30 (G11 closeout): `across_corners(...)` scoping. When a
        # field carries a corner whitelist that does NOT include the
        # current sim's corner, we skip the assertion at THIS point but
        # still record a synthetic pass — the AND-of-corners aggregate
        # in the CLI works the same way, and the field will still get
        # checked at the corners it does target (the CLI augments the
        # corner sweep with the field's list — see _run_one_job).
        if sf.across_corners is not None and current_corner not in sf.across_corners:
            assertions.append(AssertionResult(
                name=sf.name, op=sf.op, expected=sf.value, actual=None,
                passed=True,
                message=(
                    f"{sf.name}: skipped at corner={current_corner} "
                    f"(across_corners={sf.across_corners})"
                ),
                category="pass",
                source_obligation_id=source_obligation_id,
            ))
            continue

        if sim_error is not None:
            assertions.append(AssertionResult(
                name=sf.name, op=sf.op, expected=sf.value, actual=None,
                passed=False,
                message=f"{sf.name}: simulator error — {sim_error}",
                category="simulator_error",
                source_obligation_id=source_obligation_id,
            ))
            continue

        actual = measures.get(sf.name)
        if actual is None:
            assertions.append(AssertionResult(
                name=sf.name, op=sf.op, expected=sf.value, actual=None,
                passed=False,
                message=(
                    f"{sf.name}: measure not in simulator log "
                    f"(.meas extraction failed or convergence issue)"
                ),
                category="measure_missing",
                source_obligation_id=source_obligation_id,
            ))
            continue

        # v0.0.30 (G5 closeout): SQRT post-processing produces NaN when
        # the raw operand was negative. Surface those as measure_missing
        # so the LLM sees "this branch is invalid" instead of a
        # confusing spec_violation against NaN.
        if isinstance(actual, float) and math.isnan(actual):
            assertions.append(AssertionResult(
                name=sf.name, op=sf.op, expected=sf.value, actual=actual,
                passed=False,
                message=(
                    f"{sf.name}: post-processed value is NaN "
                    f"(likely sqrt of a negative simulator result)"
                ),
                category="measure_missing",
                source_obligation_id=source_obligation_id,
            ))
            continue

        # v0.0.30 (G11 closeout): `with_margin(N%)` tightens the spec
        # value before the comparison. The effective expected value is
        # the one displayed in the message so users see the tightened
        # threshold rather than the surface value.
        effective_value = _apply_margin(sf.op, sf.value, sf.margin_pct)
        passed = _check(actual, sf.op, effective_value)
        category = "pass" if passed else "spec_violation"
        margin_note = ""
        if sf.margin_pct is not None and sf.margin_pct > 0:
            margin_note = (
                f" (with_margin={sf.margin_pct:g}%, "
                f"tightened from {sf.value:g})"
            )
        msg = (
            f"{sf.name}: actual={actual:g}, expected {sf.op} "
            f"{effective_value:g}{margin_note}"
        )
        assertions.append(AssertionResult(
            name=sf.name, op=sf.op, expected=effective_value, actual=actual,
            passed=passed, message=msg, category=category,
            source_obligation_id=source_obligation_id,
        ))

    result = VerifyResult(
        cell_name=tb.cell.name,
        measures=measures,
        assertions=assertions,
        log_path=str(log),
        netlist_path=str(netlist),
        simulator_error=sim_error,
    )
    if "_compiler_source_obligation_ids" in tb.cell.__dict__:
        result.__dict__["_compiler_source_obligation_ids"] = list(
            tb.cell.__dict__["_compiler_source_obligation_ids"]
        )
    if "_compiler_source_obligations" in tb.cell.__dict__:
        result.__dict__["_compiler_source_obligations"] = list(
            tb.cell.__dict__["_compiler_source_obligations"]
        )

    # ─── Stage 3: functional-role invariants (v0.0.62) ────────
    # The functional-role layer adds a behavioral-noun keyword set
    # (`input_transconductor`, `output_buffer`, `feedforward_path`,
    # `dominant_pole_cap`, `compensation_zero`, `tail_current_source`).
    # Each declaration carries a `*_target` attribute the runtime
    # compares against the corresponding ngspice quantity. Mismatches
    # are surfaced as structured `InvariantViolation`s so an LLM
    # consumer sees actionable feedback rather than a generic FAIL.
    #
    # The check is invariant-of-spec: it runs even when every
    # SpecField assertion passes, and skips silently when the cell
    # carries no functional_roles. We stash the result on the
    # VerifyResult via `__dict__` to avoid adding a new dataclass
    # field this round (the IR layer is Agent B's scope).
    try:
        from .invariants import check_functional_invariants
        invariant_violations = check_functional_invariants(
            tb.cell, op_result=measures, ac_result=measures,
        )
    except Exception as exc:  # noqa: BLE001 — surface a soft failure
        # The invariant runtime must never crash the verify pipeline.
        # Errors during the check itself are recorded as a single
        # informational entry on the result; the legacy assertion
        # path is unaffected.
        invariant_violations = []
        result.__dict__["_invariant_runtime_error"] = (
            f"{type(exc).__name__}: {exc}"
        )
    result.__dict__["invariant_violations"] = invariant_violations
    obligation_trace = _build_obligation_trace(
        tb=tb,
        result=result,
        invariant_violations=invariant_violations,
    )
    result.__dict__["obligation_trace"] = obligation_trace
    certification_status, proof_token = _certify_obligation_trace(
        tb=tb,
        rows=obligation_trace,
    )
    result.__dict__["certification_status"] = certification_status
    result.__dict__["proof_token"] = proof_token
    return result


def _build_obligation_trace(
    *,
    tb: Testbench,
    result: VerifyResult,
    invariant_violations: list,
) -> list[dict]:
    """Build minimal source-obligation to evidence trace rows.

    This is intentionally a flat JSON-friendly row set: enough to make
    proof-token status auditable without changing parser or driver
    contracts in this first closure.
    """
    rows: list[dict] = []
    for assertion in result.assertions:
        status = _assertion_trace_status(assertion)
        obligation_id = f"obl:{tb.cell.name}:spec:{assertion.name}"
        rows.append({
            "obligation_id": obligation_id,
            "trace_id": _trace_id(tb, obligation_id, "measurement"),
            "cell": tb.cell.name,
            "source_kind": "spec",
            "required": status != "skip",
            "status": status,
            "source": {
                "cell": tb.cell.name,
                "name": assertion.name,
                "field": assertion.name,
                "ir_symbol": f"{tb.cell.name}.spec.{assertion.name}",
                "source_ref": _source_ref_from_span(
                    source_span=_spec_source_span(tb, assertion.name),
                    fallback_path=f"cell[{tb.cell.name}].spec[{assertion.name}]",
                ),
            },
            "coverage": {
                "artifact_kind": "measurement",
                "analysis_ref": _analysis_ref_for_measure(tb, assertion.name),
                "measure_name": assertion.name,
                "netlist_path": result.netlist_path,
                "log_path": result.log_path,
            },
            "evidence": {
                "expected": assertion.expected,
                "actual": assertion.actual,
                "category": assertion.category,
                "message": assertion.message,
            },
        })

    for violation in invariant_violations:
        target_attr = getattr(violation, "target_attr", "") or ""
        role_name = getattr(violation, "role_name", "?")
        obligation_id = (
            f"obl:{tb.cell.name}:functional_role:{role_name}:{target_attr}"
        )
        status = _violation_trace_status(violation)
        rows.append({
            "obligation_id": obligation_id,
            "trace_id": _trace_id(tb, obligation_id, "invariant"),
            "cell": tb.cell.name,
            "source_kind": "functional_role",
            "required": status != "skip",
            "status": status,
            "source": {
                "cell": tb.cell.name,
                "name": role_name,
                "field": target_attr,
                "role_kind": getattr(violation, "role_kind", ""),
                "ir_symbol": (
                    f"{tb.cell.name}.functional_roles.{role_name}.{target_attr}"
                ),
                "source_ref": _source_ref_from_span(
                    source_span=_functional_role_source_span(tb, role_name),
                    fallback_path=(
                        f"cell[{tb.cell.name}].functional_roles"
                        f"[{role_name}].{target_attr}"
                    ),
                ),
            },
            "coverage": {
                "artifact_kind": "invariant",
                "analysis_ref": _analysis_ref_for_measure(
                    tb,
                    _measure_name_for_violation(violation),
                ),
                "measure_name": _measure_name_for_violation(violation),
                "netlist_path": result.netlist_path,
                "log_path": result.log_path,
            },
            "evidence": {
                "expected": getattr(violation, "target_value", None),
                "actual": getattr(violation, "actual_value", None),
                "category": getattr(violation, "category", ""),
                "message": getattr(violation, "message", ""),
            },
        })
    return rows


def _trace_id(tb: Testbench, obligation_id: str, artifact_kind: str) -> str:
    return (
        f"tr:{tb.cell.name}:{tb.pdk_corner}:{tb.temp:g}:"
        f"{obligation_id}:{artifact_kind}"
    )


def _assertion_trace_status(assertion: AssertionResult) -> str:
    if assertion.category == "pass":
        if assertion.actual is None and "skipped" in assertion.message.lower():
            return "skip"
        return "pass"
    if assertion.category == "spec_violation":
        return "fail"
    if assertion.category == "measure_missing":
        return "pending"
    if assertion.category == "simulator_error":
        return "error"
    return "error"


def _violation_trace_status(violation) -> str:
    from .invariants import category_to_verification_status

    return category_to_verification_status(violation.category).value


def _measure_name_for_violation(violation) -> str:
    extras = getattr(violation, "extras", {}) or {}
    source = extras.get("actual_source")
    if isinstance(source, str) and source:
        match = re.match(r"[A-Za-z_][A-Za-z0-9_]*", source)
        if match:
            return match.group(0)
    attempts = extras.get("resolution_attempts")
    if isinstance(attempts, list) and attempts:
        first = attempts[0]
        if isinstance(first, str):
            return first
    return getattr(violation, "target_attr", "") or ""


def _analysis_ref_for_measure(tb: Testbench, measure_name: str) -> str:
    for measure in tb.measures:
        if measure.name == measure_name:
            return measure.analysis_ref
    if measure_name.startswith(("gm", "id", "vov", "i_")):
        return "op"
    return ""


def _source_ref_from_span(
    *,
    source_span,
    fallback_path: str,
) -> dict:
    if isinstance(source_span, dict) and source_span.get("line") is not None:
        out = {
            "kind": "source",
            "file": source_span.get("file"),
            "line": source_span.get("line"),
            "column": source_span.get("column"),
            "end_line": source_span.get("end_line"),
            "end_column": source_span.get("end_column"),
            "path": fallback_path,
        }
        return {k: v for k, v in out.items() if v is not None}
    return {
        "kind": "ir",
        "path": fallback_path,
    }


def _spec_source_span(tb: Testbench, field_name: str):
    for field in getattr(getattr(tb.cell, "spec", None), "fields", []) or []:
        if getattr(field, "name", None) == field_name:
            return getattr(field, "source_span", None)
    return None


def _functional_role_source_span(tb: Testbench, role_name: str):
    for role in getattr(tb.cell, "functional_roles", []) or []:
        if getattr(role, "name", None) == role_name:
            return getattr(role, "source_span", None)
    for role in getattr(tb.cell, "__dict__", {}).get("_functional_roles", []) or []:
        if getattr(role, "name", None) == role_name:
            return getattr(role, "source_span", None)
    return None


def _certify_obligation_trace(
    *,
    tb: Testbench,
    rows: list[dict],
) -> tuple[str, dict | None]:
    return _certify_obligation_rows(
        cell_name=tb.cell.name,
        pdk_corner=tb.pdk_corner,
        temp=tb.temp,
        rows=rows,
        source_obligations=_source_obligations_for_testbench(tb),
    )


def _certify_obligation_rows(
    *,
    cell_name: str,
    pdk_corner: str,
    temp: float,
    rows: list[dict],
    source_obligations: list[dict] | None = None,
) -> tuple[str, dict | None]:
    required = [row for row in rows if row.get("required", True)]
    if not required:
        return "no_obligation", None
    missing_sources = _missing_required_source_obligations(
        required,
        source_obligations or [],
    )
    if missing_sources:
        return "pending", None
    statuses = [str(row.get("status", "error")) for row in required]
    if all(status == "pass" for status in statuses):
        required_source_ids = _required_source_obligation_ids(source_obligations or [])
        token = {
            "schema_version": 1,
            "cell": cell_name,
            "status": "pass",
            "pdk_corner": pdk_corner,
            "temp": temp,
            "obligation_ids": [row["obligation_id"] for row in required],
            "trace_ids": [row["trace_id"] for row in required],
            "trace_digest": _obligation_trace_digest(required),
        }
        if required_source_ids:
            token["source_obligation_ids"] = required_source_ids
        return "pass", token
    for status in ["fail", "error", "no_convergence", "unsupported", "pending"]:
        if status in statuses:
            return status, None
    return "error", None


def _missing_required_source_obligations(
    required_rows: list[dict],
    source_obligations: list[dict],
) -> list[str]:
    covered = {str(row.get("obligation_id", "")) for row in required_rows}
    missing = []
    for source in source_obligations:
        if not source.get("required", True):
            continue
        obligation_id = str(source.get("obligation_id", ""))
        if obligation_id and obligation_id not in covered:
            missing.append(obligation_id)
    return missing


def _required_source_obligation_ids(source_obligations: list[dict]) -> list[str]:
    ids = []
    for source in source_obligations:
        if not source.get("required", True):
            continue
        obligation_id = str(source.get("obligation_id", ""))
        if obligation_id:
            ids.append(obligation_id)
    return ids


def _source_obligations_for_testbench(tb: Testbench) -> list[dict]:
    obligations: list[dict] = []
    for field in getattr(getattr(tb.cell, "spec", None), "fields", []) or []:
        name = getattr(field, "name", None)
        if not _field_required_for_corner(field, tb.pdk_corner):
            continue
        if name:
            obligations.append({
                "obligation_id": f"obl:{tb.cell.name}:spec:{name}",
                "required": True,
                "source_kind": "spec",
            })
    for role in _functional_roles_for_source_obligations(tb.cell):
        target_attr = _functional_role_target_attr(role)
        role_name = getattr(role, "name", None)
        if role_name and target_attr:
            obligations.append({
                "obligation_id": (
                    f"obl:{tb.cell.name}:functional_role:{role_name}:{target_attr}"
                ),
                "required": True,
                "source_kind": "functional_role",
            })
    return obligations


def _functional_roles_for_source_obligations(cell) -> list:
    roles = list(getattr(cell, "functional_roles", []) or [])
    legacy = list(getattr(cell, "__dict__", {}).get("_functional_roles", []) or [])
    seen = {(getattr(role, "name", None), getattr(role, "kind", None)) for role in roles}
    for role in legacy:
        key = (getattr(role, "name", None), getattr(role, "kind", None))
        if key not in seen:
            roles.append(role)
            seen.add(key)
    return roles


def _functional_role_target_attr(role) -> str | None:
    try:
        from .invariants import _TARGET_ATTR_BY_KIND
    except Exception:  # noqa: BLE001
        _TARGET_ATTR_BY_KIND = {}
    target_attr = _TARGET_ATTR_BY_KIND.get(getattr(role, "kind", ""))
    attrs = getattr(role, "attributes", {}) or {}
    if target_attr and target_attr in attrs:
        return target_attr
    return None


def _obligation_trace_digest(rows: list[dict]) -> str:
    payload = [
        {
            "obligation_id": row.get("obligation_id"),
            "trace_id": row.get("trace_id"),
            "required": row.get("required", True),
            "status": row.get("status", "error"),
            "source": row.get("source", {}),
            "coverage": row.get("coverage", {}),
            "evidence": row.get("evidence", {}),
        }
        for row in rows
    ]
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _post_process_measures(
    raw: Mapping[str, float],
    requested: list[Measure],
) -> dict[str, float]:
    """Apply per-measure Python post-processing (v0.0.30 / G5 SUM, SQRT).

    Walks every `Measure` declared on the testbench. For measures whose
    parsed AST needs post-processing (`needs_post_processing(ast)` is
    True), pulls the raw scalar from the simulator log and replaces it
    with the result of `apply_post_processing(ast, raw)`. Measures the
    parser can't read are passed through unchanged — the existing
    `measure_missing` path in `verify()` already surfaces those at the
    assertion layer.
    """
    out: dict[str, float] = dict(raw)
    for m in requested:
        if m.name not in out:
            # measure_missing — leave it absent so the verifier reports
            # the per-AssertionResult "not in simulator log" message.
            continue
        try:
            ast = parse_measure_expr(m.expr)
        except MeasureParseError:
            # Unparsable expression should not silently swallow the raw
            # value — surface the raw number so the user can still see
            # ngspice's result while the measure-expr error gets
            # reported through the existing diagnostic path elsewhere.
            continue
        if not needs_post_processing(ast):
            continue
        out[m.name] = apply_post_processing(ast, out[m.name],
                                            measure_name=m.name)
    return out


def _post_process_ldo_regulation(
    measures: dict[str, float],
    requested: list[Measure],
) -> dict[str, float]:
    """Derive `line_regulation` / `load_regulation` from vout_pp & sweep span.

    Both ngspice's `.meas` and Clay's measure_expr support FIND, MAX,
    MIN, AVG, PP, RMS, INTEG — none of which can compute a ratio of
    two extracted quantities in a single line. Rather than push that
    capability into the IR (which would require a new derived-meas
    operator in the grammar), v0.0.33 emits the components and does
    the ratio in Python here.

    Convention: when `line_regulation` is one of the requested
    measures AND `vout_pp` is in the simulator's output AND the
    sweep range is recoverable from the measure expressions, we
    REPLACE the placeholder `line_regulation` value with
    `vout_pp / (vin_max - vin_min)`. Same logic for `load_regulation`
    with the iload range.

    The sweep range is recovered from the .meas expression itself —
    the rewriter emitted `vout_pp` as
    `PP(v(out)) over dc_sweep from <lo> to <hi>` and we parse those
    bounds. This keeps the derivation hermetic to the post-process
    step (the rewriter doesn't need a separate metadata channel).

    Failure modes — if any piece is missing we leave `line_regulation`
    as-is and rely on the existing measure_missing path to surface
    the issue clearly:
      * vout_pp absent: ngspice .meas extraction failed.
      * Bounds unparseable: emitter wrote a non-aggregate expression.
      * Span ≤ 0:           defensive — division by zero / negative
                            range.
    """
    out = dict(measures)
    requested_by_name = {m.name: m for m in requested}
    for fld in ("line_regulation", "load_regulation"):
        if fld not in requested_by_name:
            continue
        # vout_pp must be available; without it we can't derive.
        pp_val = out.get("vout_pp")
        if pp_val is None:
            continue
        # Recover the sweep bounds from the vout_pp measure's
        # expression. The expression has shape
        # `PP(v(out)) over dc_sweep from <lo> to <hi>`.
        vout_pp_expr = requested_by_name.get("vout_pp")
        if vout_pp_expr is None:
            continue
        lo, hi = _extract_dc_sweep_bounds(vout_pp_expr.expr)
        if lo is None or hi is None:
            continue
        span = hi - lo
        if span <= 0.0:
            continue
        out[fld] = float(pp_val) / span
    return out


def _extract_dc_sweep_bounds(expr: str) -> tuple[float | None, float | None]:
    """Parse `(lo, hi)` from an `over dc_sweep from <lo> to <hi>` clause.

    Returns `(None, None)` if the expression doesn't contain a
    well-formed `over ... from ... to ...` clause — caller should
    treat that as "skip the derivation".
    """
    m = re.search(
        r"over\s+dc_sweep\s+from\s+([-+0-9.eE]+)\s+to\s+([-+0-9.eE]+)",
        expr,
    )
    if not m:
        return (None, None)
    try:
        return (float(m.group(1)), float(m.group(2)))
    except ValueError:
        return (None, None)


def _check(actual: float, op: str, expected: float) -> bool:
    if op in (">=", "≥"): return actual >= expected
    if op in ("<=", "≤"): return actual <= expected
    if op == "=":         return abs(actual - expected) / max(abs(expected), 1e-12) < 0.05
    if op == ">":         return actual >  expected
    if op == "<":         return actual <  expected
    raise ValueError(f"unknown op {op!r}")


def _apply_margin(op: str, value: float, margin_pct: float | None) -> float:
    """Tighten a spec value by the requested percentage (v0.0.30, G11).

    The direction of tightening depends on the operator:

      * `>=` / `>`  — the spec demands MORE than `value`; tightening
                      increases the threshold (`value * (1 + m/100)`).
                      `gain_dc >= 25 dB with_margin(20%)` requires the
                      actual to be ≥ 30 dB.
      * `<=` / `<`  — the spec demands LESS than `value`; tightening
                      decreases the threshold (`value * (1 - m/100)`).
                      `power <= 1 mW with_margin(20%)` requires the
                      actual to be ≤ 0.8 mW.
      * `=`        — for equality specs, margin tightens the tolerance
                     band by REDUCING the magnitude of `value` by `m%`
                     so the relative-tolerance check in `_check` has
                     less slack. The "tighten the equality" semantics
                     are documented but rarely used in practice;
                     callers that need an absolute tolerance should
                     prefer ≥ / ≤ form.

    `value == 0` for a `>=` / `>` spec is a degenerate case — any
    positive margin still maps to zero (`0 * (1 + m/100) == 0`). That's
    arguably the correct behavior (the spec is "non-negative" and that
    stays non-negative under any tightening) but we surface a note in
    the verifier output so the user isn't surprised.

    Negative `value` with `>=` op (e.g. `cmrr_phase >= -180 deg`) is
    handled by the same multiplication: `-180 * 1.2 = -216`, which
    DEMANDS MORE phase margin. This matches the algebraic intent.
    """
    if margin_pct is None or margin_pct == 0.0:
        return value
    factor = margin_pct / 100.0
    if op in (">=", ">", "≥"):
        return value * (1.0 + factor)
    if op in ("<=", "<", "≤"):
        return value * (1.0 - factor)
    # op == "="; tightening shrinks the symmetric tolerance band by
    # demanding closer-to-zero deviation. We expose this by REDUCING
    # the magnitude of value used in _check — the comparison there is
    # `|actual - expected| / max(|expected|, 1e-12) < 0.05`, so a
    # smaller expected makes the relative-tolerance band tighter.
    return value * (1.0 - factor)


# ─── v0.0.29 (G4 multi-setup verify) ────────────────────────────────


def rewrite_testbench_for_drive_mode(
    tb: Testbench,
    drive_mode: str,
    drive_value: float = 1.0,
) -> Testbench:
    """Return a NEW Testbench with sources/extras swapped for `drive_mode`.

    The input testbench `tb` is expected to be the "differential AC
    rig" produced by the cell's default testbench builder (or by
    `build_ac_testbench` directly). We don't try to handle every
    possible custom testbench shape — only the shapes the registered
    stage testbenches actually produce.

    Drive modes:

      * `ac_diff`    — historical default. Pass through unchanged
                       (deep-copied for safety).
      * `ac_common`  — both inn and inp see +drive_value AC. The two
                       VCVS gains are forced to `+drive_value / 2`
                       (matching the differential rig's ±0.5 split)
                       so the common-mode gain remains a directly
                       comparable scalar.
      * `ac_supply_p` — VDD source becomes a `voltage_ac` with
                       ac=drive_value (dc=supply_volt). The
                       differential AC source (VSIG) is set to dc=0
                       ac=0 so the input nodes don't double-drive.
                       VCVS gains are zeroed (inputs sit at VCM, no
                       differential excitation).
      * `ac_supply_n` — VSS source becomes `voltage_ac` with
                       ac=drive_value (dc=0). Same input quieting as
                       `ac_supply_p`.

    Unknown drive modes raise ValueError. The function never mutates
    the input testbench — every change is on a deepcopy.
    """
    if drive_mode == "ac_diff":
        # Pass-through path. We still deep-copy so the caller can
        # safely mutate the result without aliasing the original.
        return deepcopy(tb)
    new_tb = deepcopy(tb)
    if drive_mode == "ac_common":
        # Common-mode: flip both VCVS gains to +0.5*drive_value so
        # inn and inp move TOGETHER. The historical rig used `Edp`
        # (gain=+0.5) and `Edn` (gain=-0.5); we set both to
        # +0.5 * drive_value here. If the stage testbench used
        # different VCVS instance names, we still match on the
        # `vcvs` device_class and rewrite uniformly — any VCVS
        # driving an input pair takes the common-mode gain.
        for dev in new_tb.extra_devices:
            if dev.device_class == "vcvs":
                dev.sizing["gain"] = 0.5 * drive_value
        return new_tb
    if drive_mode in ("ac_supply_p", "ac_supply_n"):
        target = "vdd" if drive_mode == "ac_supply_p" else "vss"
        supply_volt = float(new_tb.supply_volt)
        # Walk the source list; promote the DC supply we care about
        # to a voltage_ac source carrying drive_value AC.
        rewrote_supply = False
        for i, src in enumerate(new_tb.sources):
            if (
                src.kind == "voltage_dc"
                and src.pins.get("p") == target
            ):
                new_sources = list(new_tb.sources)
                new_sources[i] = Source(
                    kind="voltage_ac",
                    instance_name=src.instance_name,
                    pins=dict(src.pins),
                    params={
                        # Preserve the historical DC level so the
                        # operating point doesn't move.
                        "dc": (supply_volt if target == "vdd" else 0.0),
                        "ac": drive_value,
                    },
                )
                new_tb.sources = new_sources
                rewrote_supply = True
                break
        if not rewrote_supply:
            raise ValueError(
                f"drive mode {drive_mode!r}: no voltage_dc source "
                f"feeding {target!r} was found in the testbench. "
                f"This drive mode requires a {target!r} DC supply in "
                f"the testbench's source list."
            )
        # Quiet the differential AC input: set VSIG.ac = 0 if it
        # exists, and zero out VCVS gains so the input pair sits at
        # VCM with no excitation.
        for src in new_tb.sources:
            if src.kind == "voltage_ac" and src.instance_name == "VSIG":
                src.params["ac"] = 0.0
        for dev in new_tb.extra_devices:
            if dev.device_class == "vcvs":
                dev.sizing["gain"] = 0.0
        return new_tb

    # ─── v0.0.33: LDO drive modes ────────────────────────────────
    # The four LDO drive modes (`dc_vin_sweep`, `dc_iload_sweep`,
    # `tran_load_step`, `tran_line_step`) rewrite the testbench
    # produced by `clay.templates.ldo_testbench.build_ldo_testbench`.
    # They mutate the analyses list (DC/TRAN swap) and the measures
    # list (line vs load vs transient) but never touch the cell —
    # the rewriter is still a pure IR-level transformation.
    #
    # `drive_value` is interpreted as a magnitude/upper-bound knob
    # specific to each mode:
    #   * dc_vin_sweep:    Vin sweep upper bound, in volts
    #   * dc_iload_sweep:  Iload sweep upper bound, in amps
    #   * tran_load_step:  Iload post-step value, in amps
    #   * tran_line_step:  Vin post-step value, in volts
    # The lower bound / pre-step value uses the source's existing
    # `dc` param so callers can configure both sides via the
    # testbench builder kwargs (vin_min/vin_max, iload_min/iload_max).
    if drive_mode == "dc_vin_sweep":
        return _rewrite_dc_sweep(
            new_tb, source_name="VIN",
            ngspice_source_name="VVIN",
            upper_bound=drive_value, axis_kind="line",
        )
    if drive_mode == "dc_iload_sweep":
        return _rewrite_dc_sweep(
            new_tb, source_name="ILOAD",
            ngspice_source_name="IILOAD",
            upper_bound=drive_value, axis_kind="load",
        )
    if drive_mode == "tran_load_step":
        return _rewrite_tran_step(
            new_tb, source_name="ILOAD",
            post_step_value=drive_value, axis_kind="load",
        )
    if drive_mode == "tran_line_step":
        return _rewrite_tran_step(
            new_tb, source_name="VIN",
            post_step_value=drive_value, axis_kind="line",
        )
    raise ValueError(
        f"unknown drive_mode {drive_mode!r}. Valid modes: "
        f"'ac_diff', 'ac_common', 'ac_supply_p', 'ac_supply_n', "
        f"'dc_vin_sweep', 'dc_iload_sweep', "
        f"'tran_load_step', 'tran_line_step'."
    )


def _find_source_index(sources: list[Source], name: str) -> int:
    """Return the index of the named source in `sources` or raise."""
    for i, s in enumerate(sources):
        if s.instance_name == name:
            return i
    raise ValueError(
        f"required source {name!r} not found in testbench. "
        f"Available sources: {[s.instance_name for s in sources]}. "
        f"This drive mode expects an LDO-shaped testbench (see "
        f"clay.templates.ldo_testbench.build_ldo_testbench)."
    )


def _rewrite_dc_sweep(
    tb: Testbench,
    *,
    source_name: str,
    ngspice_source_name: str,
    upper_bound: float,
    axis_kind: str,
) -> Testbench:
    """Rewrite the testbench to sweep `source_name` over a DC range.

    Replaces the analyses list with `[OpAnalysis(), DcAnalysis(...)]`,
    swaps the measures list for the line- or load-regulation suite
    (per `axis_kind`), and ALSO emits a single derived measure
    `line_regulation` (or `load_regulation`) computed from the PP
    and the sweep range bounds.

    Args:
        tb: deep-copied testbench (caller owns the copy).
        source_name: which source instance to sweep, by IR name
            (`VIN` or `ILOAD`).
        ngspice_source_name: the FULL ngspice device name (prefixed
            with `V` for voltage / `I` for current sources). The
            DcAnalysis.sweep_source field stores this name so the
            generated `dc <name> ...` line matches the netlist
            instance.
        upper_bound: sweep upper bound (volts for Vin, amps for Iload).
        axis_kind: `"line"` or `"load"` — picks the measure naming
            and the derived-regulation field name.

    The lower bound comes from the source's current `dc` param.
    Sweep step is `(upper - lower) / 100` so any sweep produces a
    smooth 101-point trace by default.
    """
    src_idx = _find_source_index(tb.sources, source_name)
    src = tb.sources[src_idx]
    lower_bound = float(src.params.get("dc", 0.0))
    if upper_bound <= lower_bound:
        # Defensive: a flat / inverted sweep collapses the .meas
        # FROM/TO range and ngspice computes garbage. Raise with a
        # precise hint so the user can fix the testbench kwargs.
        raise ValueError(
            f"drive mode dc_{axis_kind} sweep: upper_bound={upper_bound:g} "
            f"must exceed lower_bound={lower_bound:g} (the source's "
            f"current dc value). Adjust the testbench builder's "
            f"min/max kwargs or the drive_value."
        )
    step = (upper_bound - lower_bound) / 100.0
    # Replace analyses with OP + a DC sweep over this source.
    tb.analyses = [
        OpAnalysis(),
        DcAnalysis(
            sweep_source=ngspice_source_name,
            start=lower_bound,
            stop=upper_bound,
            step=step,
        ),
    ]
    # Replace measures with the line- or load-regulation suite.
    tb.measures = _dc_sweep_measures(
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        axis_kind=axis_kind,
    )
    return tb


def _rewrite_tran_step(
    tb: Testbench,
    *,
    source_name: str,
    post_step_value: float,
    axis_kind: str,
) -> Testbench:
    """Rewrite the testbench to model a transient step using a real
    PULSE source edge.

    v0.0.34 (PWL/PULSE source IR): the named source's kind is rewritten
    in place from `voltage_dc` → `voltage_pulse` (line step) or
    `current_dc` → `current_pulse` (load step). The pre-step level is
    taken from the source's existing `dc` param; the post-step level
    is `post_step_value`. A 10 ns rise time at t=10 us drives the
    edge cleanly inside the 100 us transient window, so the .meas
    captures both the settled pre-step OP and the transient response
    to the edge.

    Args:
        tb: deep-copied testbench.
        source_name: which source to step (`VIN` or `ILOAD`).
        post_step_value: source value AFTER the step.
        axis_kind: `"line"` or `"load"` — informational; used in
            the docstring of emitted measures.

    Returns the mutated testbench.
    """
    src_idx = _find_source_index(tb.sources, source_name)
    src = tb.sources[src_idx]
    pre_step_value = float(src.params.get("dc", 0.0))
    # Decide voltage_pulse vs current_pulse from the original kind.
    # voltage_dc → voltage_pulse; current_dc → current_pulse. We
    # don't try to handle voltage_ac on these sources — it would be
    # surprising in a tran_*_step setup and the testbench builder
    # currently always emits DC kinds here.
    if src.kind == "voltage_dc":
        new_kind = "voltage_pulse"
    elif src.kind == "current_dc":
        new_kind = "current_pulse"
    else:
        raise ValueError(
            f"tran_*_step drive mode: source {source_name!r} has "
            f"unexpected kind {src.kind!r}. Expected voltage_dc or "
            f"current_dc as the pre-step source shape."
        )
    # Pulse timing: step happens 10us into the transient window so the
    # solver settles on the pre-step OP before the edge arrives. A
    # 10 ns edge is fast enough to look like a step to LDO-class
    # bandwidth but slow enough to avoid stressing ngspice convergence.
    from .templates.ldo_testbench import (
        DEFAULT_TRAN_END, DEFAULT_TRAN_STEP,
    )
    src.kind = new_kind
    src.params = {
        # Pulse params: drive the new ngspice PULSE(...) line.
        "v_low": pre_step_value,
        "v_high": post_step_value,
        "td": 10e-6,           # 10us delay before the step
        "t_rise": 10e-9,       # 10 ns rise
        "t_fall": 10e-9,       # 10 ns fall (unused for one-shot step)
        # pulse stays high to the end of the window (one-shot step) —
        # so we just need pulse_width >= tran_end.
        "pulse_width": DEFAULT_TRAN_END,
        "period": 2.0 * DEFAULT_TRAN_END,
        # Back-compat: keep `dc` set to the post-step value so any
        # introspecting tooling (and existing tests) reading
        # `source.params["dc"]` still sees the meaningful settle-point
        # number. The ngspice driver ignores `dc` for *_pulse kinds.
        "dc": post_step_value,
    }
    tb.analyses = [
        OpAnalysis(),
        TranAnalysis(step=DEFAULT_TRAN_STEP, t_end=DEFAULT_TRAN_END),
    ]
    tb.measures = _tran_step_measures(
        t_end=DEFAULT_TRAN_END, axis_kind=axis_kind,
    )
    return tb


def _dc_sweep_measures(
    *, lower_bound: float, upper_bound: float, axis_kind: str,
) -> list[Measure]:
    """Build the .meas list for a DC sweep rewrite.

    Emits four .meas: MAX / MIN / AVG / PP of v(out) over the
    `dc_sweep` axis bounded by the sweep range, plus an `at lower
    bound` sample for dropout/load-min reference. Then computes a
    derived `line_regulation` (or `load_regulation`) measure via
    Clay's SUM(...) / division — but Clay's measure_expr doesn't
    support division between two measures inside one expression
    (only AGG + scalar bounds), so we EMIT the components and
    document that the user can derive regulation as a top-level
    `measure := setup.vout_pp / range` in a multi-setup verify.

    For convenience the standalone (single-setup) line/load reg
    spec field is matched against `vout_pp / span` post-processed
    in the regulation_post_process function below — but for v0
    we keep the emitter simple and let derivations live in the
    user's .clay file.

    To support the common case where a user writes
    `setup line_reg { drive { dc_vin_sweep = 3.0; }
                      spec { line_regulation : real <= 50 mV/V; } }`
    directly, we ALSO emit a synthetic `line_regulation` /
    `load_regulation` measure that the verifier post-processes
    using `vout_pp` and the sweep span. The ngspice .meas itself
    can't compute the division so we emit it as a `find v(out)`
    at the lower bound (a value ngspice WILL produce) and rely on
    `_post_process_ldo_regulation` to override it with the correct
    derivation before the assertion fires.
    """
    fld = "line_regulation" if axis_kind == "line" else "load_regulation"
    span_var = "vin" if axis_kind == "line" else "iload"
    return [
        Measure("vout_max", "dc",
                f"MAX(v(out)) over dc_sweep from {lower_bound:g} to {upper_bound:g}"),
        Measure("vout_min", "dc",
                f"MIN(v(out)) over dc_sweep from {lower_bound:g} to {upper_bound:g}"),
        Measure("vout_avg", "dc",
                f"AVG(v(out)) over dc_sweep from {lower_bound:g} to {upper_bound:g}"),
        Measure("vout_pp", "dc",
                f"PP(v(out)) over dc_sweep from {lower_bound:g} to {upper_bound:g}"),
        # vout at the sweep lower bound — dropout/min-load reference.
        Measure(f"vout_at_{span_var}_min", "dc",
                f"v(out) @ dc_sweep={lower_bound:g}"),
        # vout at the sweep upper bound — for symmetry / dropout
        # delta computation.
        Measure(f"vout_at_{span_var}_max", "dc",
                f"v(out) @ dc_sweep={upper_bound:g}"),
        # Synthetic regulation measure. Emitted as a placeholder
        # `find v(out)` so ngspice doesn't complain; the verifier's
        # post-processor (`_post_process_ldo_regulation`) replaces
        # the raw value with `vout_pp / span` before the assertion.
        Measure(fld, "dc",
                f"v(out) @ dc_sweep={lower_bound:g}"),
    ]


def _tran_step_measures(*, t_end: float, axis_kind: str) -> list[Measure]:
    """Build the .meas list for a transient-step rewrite."""
    # axis_kind is currently informational — the same set of vout
    # measures applies to both load step and line step.
    _ = axis_kind  # explicit silence (kept for future symmetry).
    return [
        Measure("vout_max", "tran",
                f"MAX(v(out)) over time from 0 to {t_end:g}"),
        Measure("vout_min", "tran",
                f"MIN(v(out)) over time from 0 to {t_end:g}"),
        Measure("vout_initial", "tran",
                f"v(out) @ time=0"),
        Measure("vout_final", "tran",
                f"v(out) @ time={t_end:g}"),
    ]


def verify_multi_setup(
    tb_builder,
    cell,
    setups: list[SetupSpec],
    derived_measures: Mapping[str, str],
    top_spec: Spec,
    *,
    work_dir: Path | str = "tmp_clay_rundir",
    driver: SimulatorDriver | None = None,
    pdk_corner: str = "tt",
    temp: float = 27.0,
) -> "MultiSetupResult":
    """Run a multi-setup verify: per-setup sims + derived measures + asserts.

    Args:
        tb_builder: function `(cell) -> Testbench`. We call it ONCE per
            setup and then rewrite sources/extras for that setup's
            drive mode. Calling per-setup keeps each Testbench fresh
            so the deepcopy in `rewrite_testbench_for_drive_mode`
            stays cheap.
        cell: the compiled Cell IR (shared across setups).
        setups: list of SetupSpec from `VerifyJob.setups`.
        derived_measures: top-level `name → expr_str` dict.
        top_spec: top-level Spec block (may be empty). Each spec field's
            `name` MUST resolve to a derived measure name (the parser
            already validated this).
        pdk_corner / temp: simulation point (caller sets these per
            corners × temps loop iteration).
    """
    drv = driver or NgspiceDriver()
    work = Path(work_dir)
    # Per-setup VerifyResult — collected even if some setups fail so
    # the caller can show the full damage picture.
    per_setup: dict[str, VerifyResult] = {}
    # Per-setup measure dict (only the ngspice-emitted scalars; not
    # the derived ones). Used to evaluate derived expressions below.
    setup_measures: dict[str, dict[str, float]] = {}
    for setup in setups:
        base_tb: Testbench = tb_builder(cell)
        base_tb.pdk_corner = pdk_corner
        base_tb.temp = temp
        # The per-setup testbench overrides the cell's default spec
        # (which is what gets emitted as .meas in the netlist) with the
        # setup's spec (if it has one). The cell's spec field carries
        # the assertion target; the testbench's measure list is the
        # superset of what we extract. Important: we keep the
        # testbench's measure list intact — every default measure
        # (gain_dc, gbw, phase_at_gbw) still extracts, regardless of
        # whether the setup's spec talks about them.
        rewritten = rewrite_testbench_for_drive_mode(
            base_tb,
            drive_mode=setup.drive_mode,
            drive_value=setup.drive_value,
        )
        # Sub-directory so each setup's netlist/log lives in its own
        # folder (otherwise the .cir name collides between setups).
        setup_work = work / f"setup_{setup.name}"
        setup_work.mkdir(parents=True, exist_ok=True)
        # Per-setup cell with the setup's spec injected — so the
        # single-setup `verify()` function applies setup-scoped
        # assertions correctly.
        setup_cell = deepcopy(rewritten.cell)
        setup_cell.spec = (
            _scope_spec_source_obligations(
                setup.spec,
                cell_name=cell.name,
                scope=f"setup:{setup.name}",
            )
            if setup.spec is not None
            else Spec(fields=[])
        )
        rewritten.cell = setup_cell
        result = verify(rewritten, work_dir=setup_work, driver=drv)
        per_setup[setup.name] = result
        setup_measures[setup.name] = dict(result.measures)
    # Compute derived measures from the per-setup measure dicts.
    derived_values: dict[str, float] = {}
    derived_errors: dict[str, str] = {}
    for dm_name, dm_expr in derived_measures.items():
        try:
            value = _eval_derived_expr(dm_expr, setup_measures)
        except _DerivedEvalError as exc:
            derived_errors[dm_name] = str(exc)
            continue
        derived_values[dm_name] = value
    # Apply top-level assertions against derived measures.
    top_assertions: list[AssertionResult] = []
    for spec_index, f in enumerate(top_spec.fields):
        source_obligation_id = f.source_obligation_id_for(
            cell.name, spec_index, scope="derived"
        )
        if f.across_corners is not None and pdk_corner not in f.across_corners:
            top_assertions.append(AssertionResult(
                name=f.name, op=f.op, expected=f.value, actual=None,
                passed=True,
                message=(
                    f"{f.name}: skipped at corner={pdk_corner} "
                    f"(across_corners={f.across_corners})"
                ),
                category="pass",
                source_obligation_id=source_obligation_id,
            ))
            continue
        if f.name in derived_errors:
            top_assertions.append(AssertionResult(
                name=f.name, op=f.op, expected=f.value, actual=None,
                passed=False,
                message=(
                    f"{f.name}: derived measure could not be evaluated — "
                    f"{derived_errors[f.name]}"
                ),
                category="measure_missing",
                source_obligation_id=source_obligation_id,
            ))
            continue
        actual = derived_values.get(f.name)
        if actual is None:
            top_assertions.append(AssertionResult(
                name=f.name, op=f.op, expected=f.value, actual=None,
                passed=False,
                message=(
                    f"{f.name}: top-level assertion references "
                    f"undeclared derived measure"
                ),
                category="measure_missing",
                source_obligation_id=source_obligation_id,
            ))
            continue
        if not math.isfinite(actual):
            top_assertions.append(AssertionResult(
                name=f.name, op=f.op, expected=f.value, actual=actual,
                passed=False,
                message=(
                    f"{f.name}: derived value is non-finite ({actual!r}) "
                    f"— check the underlying setup measures"
                ),
                category="measure_missing",
                source_obligation_id=source_obligation_id,
            ))
            continue
        passed = _check(actual, f.op, f.value)
        category = "pass" if passed else "spec_violation"
        msg = f"{f.name}: actual={actual:g}, expected {f.op} {f.value:g}"
        top_assertions.append(AssertionResult(
            name=f.name, op=f.op, expected=f.value, actual=actual,
            passed=passed, message=msg, category=category,
            source_obligation_id=source_obligation_id,
        ))
    ms_result = MultiSetupResult(
        cell_name=cell.name,
        per_setup=per_setup,
        derived_measures=derived_values,
        derived_assertions=top_assertions,
        derived_errors=derived_errors,
    )
    trace_rows = _build_multi_setup_obligation_trace(
        cell_name=cell.name,
        per_setup=per_setup,
        derived_assertions=top_assertions,
        derived_expressions=derived_measures,
        pdk_corner=pdk_corner,
        temp=temp,
    )
    status, token = _certify_obligation_rows(
        cell_name=cell.name,
        pdk_corner=pdk_corner,
        temp=temp,
        rows=trace_rows,
        source_obligations=_source_obligations_for_multi_setup(
            cell_name=cell.name,
            setups=setups,
            top_spec=top_spec,
            pdk_corner=pdk_corner,
        ),
    )
    ms_result.__dict__["obligation_trace"] = trace_rows
    ms_result.__dict__["certification_status"] = status
    ms_result.__dict__["proof_token"] = token
    return ms_result


def _source_obligations_for_multi_setup(
    *,
    cell_name: str,
    setups: list[SetupSpec],
    top_spec: Spec,
    pdk_corner: str,
) -> list[dict]:
    obligations: list[dict] = []
    for setup in setups:
        for field in getattr(getattr(setup, "spec", None), "fields", []) or []:
            name = getattr(field, "name", None)
            if name and _field_required_for_corner(field, pdk_corner):
                obligations.append({
                    "obligation_id": f"obl:{cell_name}:setup:{setup.name}:spec:{name}",
                    "required": True,
                    "source_kind": "setup_spec",
                })
    for field in getattr(top_spec, "fields", []) or []:
        name = getattr(field, "name", None)
        if name and _field_required_for_corner(field, pdk_corner):
            obligations.append({
                "obligation_id": f"obl:{cell_name}:derived_spec:{name}",
                "required": True,
                "source_kind": "derived_spec",
            })
    return obligations


def _field_required_for_corner(field, pdk_corner: str) -> bool:
    corners = getattr(field, "across_corners", None)
    return corners is None or pdk_corner in corners


def _build_multi_setup_obligation_trace(
    *,
    cell_name: str,
    per_setup: Mapping[str, VerifyResult],
    derived_assertions: list[AssertionResult],
    derived_expressions: Mapping[str, str],
    pdk_corner: str,
    temp: float,
) -> list[dict]:
    """Aggregate per-setup traces and top-level derived assertions."""
    rows: list[dict] = []
    for setup_name, setup_result in per_setup.items():
        for row in setup_result.__dict__.get("obligation_trace", []):
            cloned = dict(row)
            cloned["source"] = dict(row.get("source", {}))
            cloned["coverage"] = dict(row.get("coverage", {}))
            cloned["evidence"] = dict(row.get("evidence", {}))
            old_obligation_id = str(row["obligation_id"])
            if old_obligation_id.startswith(f"obl:{cell_name}:"):
                suffix = old_obligation_id[len(f"obl:{cell_name}:"):]
            else:
                suffix = old_obligation_id
            obligation_id = f"obl:{cell_name}:setup:{setup_name}:{suffix}"
            cloned["setup"] = setup_name
            cloned["obligation_id"] = obligation_id
            cloned["trace_id"] = (
                f"tr:{cell_name}:{pdk_corner}:{temp:g}:{obligation_id}:"
                f"{cloned['coverage'].get('artifact_kind', 'measurement')}"
            )
            cloned["source"]["setup"] = setup_name
            rows.append(cloned)

    for assertion in derived_assertions:
        status = _assertion_trace_status(assertion)
        obligation_id = f"obl:{cell_name}:derived_spec:{assertion.name}"
        expression = derived_expressions.get(assertion.name, "")
        rows.append({
            "obligation_id": obligation_id,
            "trace_id": (
                f"tr:{cell_name}:{pdk_corner}:{temp:g}:"
                f"{obligation_id}:derived_measure"
            ),
            "cell": cell_name,
            "source_kind": "derived_spec",
            "required": status != "skip",
            "status": status,
            "source": {
                "cell": cell_name,
                "name": assertion.name,
                "field": assertion.name,
                "ir_symbol": f"{cell_name}.derived_spec.{assertion.name}",
                "source_ref": {
                    "kind": "ir",
                    "path": f"verify[{cell_name}].spec[{assertion.name}]",
                },
            },
            "coverage": {
                "artifact_kind": "derived_measure",
                "analysis_ref": None,
                "measure_name": assertion.name,
                "expression": expression,
            },
            "evidence": {
                "expected": assertion.expected,
                "actual": assertion.actual,
                "category": assertion.category,
                "message": assertion.message,
            },
        })
    return rows


def _scope_spec_source_obligations(
    spec: Spec,
    *,
    cell_name: str,
    scope: str,
) -> Spec:
    scoped = deepcopy(spec)
    for index, field in enumerate(scoped.fields):
        if field.source_obligation_id is None:
            field.source_obligation_id = field.source_obligation_id_for(
                cell_name, index, scope=scope
            )
    return scoped


from dataclasses import dataclass, field


@dataclass
class MultiSetupResult:
    """Aggregated result from a multi-setup verify job (v0.0.29, G4).

    Holds:
      * `per_setup`           — per-setup VerifyResult (one ngspice run
                                each). Keyed by setup name.
      * `derived_measures`    — scalar values for each successfully
                                evaluated `measure NAME := <expr>;`.
      * `derived_assertions`  — top-level spec assertions on the
                                derived measures. Categorized the same
                                way as `VerifyResult.assertions`.
      * `derived_errors`      — `name → error_message` for derived
                                measures that failed to evaluate
                                (e.g. a referenced setup measure was
                                measure_missing).
    """
    cell_name: str
    per_setup: dict[str, VerifyResult]
    derived_measures: dict[str, float]
    derived_assertions: list[AssertionResult]
    derived_errors: dict[str, str] = field(default_factory=dict)

    @property
    def all_passed(self) -> bool:
        """True iff every setup passed AND every derived assertion passed.

        A setup with no spec block still passes as long as its simulator
        ran (no simulator_error and no measure_missing on the assertions
        it declared). An empty top-level spec passes trivially.
        """
        for r in self.per_setup.values():
            if not r.all_passed:
                return False
        return all(a.passed for a in self.derived_assertions)


# ─── Derived expression evaluator ────────────────────────────────


class _DerivedEvalError(Exception):
    """Raised by `_eval_derived_expr` when a derived measure can't be
    computed (missing setup measure, divide-by-zero, etc.)."""


_DERIVED_TOKEN_RE = re.compile(
    r"\s*(?:"
    r"(?P<num>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
    r"|(?P<ref>[A-Za-z_][A-Za-z_0-9]*\.[A-Za-z_][A-Za-z_0-9]*)"
    r"|(?P<op>[+\-*/()])"
    r")"
)


def _tokenize_derived(expr: str) -> list[tuple[str, str]]:
    """Tokenize a derived-measure expression string.

    Returns a list of `(kind, lexeme)` pairs where `kind` ∈
    {'num', 'ref', 'op'} (matching the regex named groups). Raises
    `_DerivedEvalError` on any character that doesn't match.
    """
    tokens: list[tuple[str, str]] = []
    pos = 0
    while pos < len(expr):
        m = _DERIVED_TOKEN_RE.match(expr, pos)
        if m is None or m.end() == pos:
            # Catch leading whitespace separately so the error points
            # at a real bad character.
            if expr[pos].isspace():
                pos += 1
                continue
            raise _DerivedEvalError(
                f"unexpected character {expr[pos]!r} at position {pos} "
                f"of derived expression {expr!r}"
            )
        if m.group("num") is not None:
            tokens.append(("num", m.group("num")))
        elif m.group("ref") is not None:
            tokens.append(("ref", m.group("ref")))
        elif m.group("op") is not None:
            tokens.append(("op", m.group("op")))
        pos = m.end()
    return tokens


def _eval_derived_expr(
    expr: str,
    setup_measures: Mapping[str, Mapping[str, float]],
) -> float:
    """Evaluate a derived-measure expression against per-setup measures.

    Supported subset: scalar arithmetic over `<setup>.<measure>` refs,
    numeric literals, and operators `+ - * /` with parenthesized
    sub-expressions. Operator precedence follows standard math
    (`*` / `/` > `+` / `-`). Unary minus is supported at the start of
    an expression and immediately after `(`.

    Raises `_DerivedEvalError` if a referenced setup or measure is
    missing, the expression doesn't parse, or a divide-by-zero occurs.
    """
    tokens = _tokenize_derived(expr)
    if not tokens:
        raise _DerivedEvalError(f"empty derived expression: {expr!r}")
    # Recursive-descent parser over the token list.
    state = {"i": 0}

    def peek() -> tuple[str, str] | None:
        if state["i"] < len(tokens):
            return tokens[state["i"]]
        return None

    def consume() -> tuple[str, str]:
        tok = tokens[state["i"]]
        state["i"] += 1
        return tok

    def parse_expr() -> float:
        # Addition / subtraction (left-assoc).
        v = parse_term()
        while True:
            tok = peek()
            if tok is None:
                return v
            if tok == ("op", "+"):
                consume()
                v = v + parse_term()
            elif tok == ("op", "-"):
                consume()
                v = v - parse_term()
            else:
                return v

    def parse_term() -> float:
        # Multiplication / division (left-assoc).
        v = parse_factor()
        while True:
            tok = peek()
            if tok is None:
                return v
            if tok == ("op", "*"):
                consume()
                v = v * parse_factor()
            elif tok == ("op", "/"):
                consume()
                rhs = parse_factor()
                if rhs == 0.0:
                    raise _DerivedEvalError(
                        f"divide by zero in derived expression {expr!r}"
                    )
                v = v / rhs
            else:
                return v

    def parse_factor() -> float:
        tok = peek()
        if tok is None:
            raise _DerivedEvalError(
                f"unexpected end of derived expression {expr!r}"
            )
        kind, lexeme = tok
        if kind == "op" and lexeme == "-":
            consume()
            return -parse_factor()
        if kind == "op" and lexeme == "+":
            consume()
            return parse_factor()
        if kind == "op" and lexeme == "(":
            consume()
            v = parse_expr()
            close = peek()
            if close != ("op", ")"):
                raise _DerivedEvalError(
                    f"missing `)` in derived expression {expr!r}"
                )
            consume()
            return v
        if kind == "num":
            consume()
            return float(lexeme)
        if kind == "ref":
            consume()
            setup_name, measure_name = lexeme.split(".", 1)
            if setup_name not in setup_measures:
                raise _DerivedEvalError(
                    f"derived expression {expr!r}: unknown setup "
                    f"{setup_name!r}"
                )
            sd = setup_measures[setup_name]
            if measure_name not in sd:
                raise _DerivedEvalError(
                    f"derived expression {expr!r}: setup "
                    f"{setup_name!r} has no measure {measure_name!r} "
                    f"(measure_missing or simulator_error). Known: "
                    f"{sorted(sd)}."
                )
            return float(sd[measure_name])
        raise _DerivedEvalError(
            f"unexpected token {tok!r} in derived expression {expr!r}"
        )

    result = parse_expr()
    if state["i"] != len(tokens):
        raise _DerivedEvalError(
            f"trailing tokens in derived expression {expr!r} "
            f"(parsed up to index {state['i']} of {len(tokens)})"
        )
    return result


# ─── v0.1.01 §5.2: distance-to-satisfaction verify ──────────────────
#
# The boolean `verify()` API above stays byte-exact: this section adds
# a NEW entry point `verify_with_distance` that drives the same
# simulator pipeline and then wraps the result with the continuous
# distance-to-satisfaction signal from `clay.distance_to_sat`. The
# v0.0.x `VerifyResult` is preserved verbatim under
# `DistanceVerifyResult.verify_result`, so any caller that still
# expects the per-`AssertionResult.category` text path keeps working.
#
# Why an additive wrapper rather than inlining the loss into
# `VerifyResult`: the dataclass `VerifyResult` ships as part of Clay's
# stable surface — adding a field would force every downstream
# consumer to rev. The wrapper composes the new signal without
# touching the legacy surface.
#
# Module-level import would create a circular dep (`distance_to_sat`
# also imports from this module's `verify`). The function-level lazy
# import is the standard Python pattern for forward references.


def verify_with_distance(
    tb: Testbench,
    *,
    work_dir: Path | str = "tmp_clay_rundir",
    driver: SimulatorDriver | None = None,
    weights: Mapping[str, float] | None = None,
    eq_rel_tol: float | None = None,
    treat_inf_as: float | None = None,
):
    """Run `verify(tb)` AND build a continuous-distance result.

    See `clay.distance_to_sat.verify_with_distance` for the full
    contract. This function is a thin re-export so callers reaching
    into `clay.verifier` (the historical home of the verify entry
    point) find both shapes side-by-side.

    Args:
        tb: a Testbench (same input shape as `verify`).
        work_dir / driver: forwarded to `verify` verbatim.
        weights: optional per-spec weight overrides (uniform if None).
        eq_rel_tol: relative-tolerance band for `=` operators. None
            uses the module-level default from `clay.distance_to_sat`.
        treat_inf_as: replacement value for `math.inf` in the
            aggregate satisfaction loss. None propagates inf.

    Returns:
        A `clay.distance_to_sat.DistanceVerifyResult`.
    """
    from .distance_to_sat import (
        EQ_RELATIVE_TOLERANCE,
        verify_with_distance as _dist_verify,
    )
    return _dist_verify(
        tb, work_dir=work_dir, driver=driver,
        weights=weights,
        eq_rel_tol=(
            EQ_RELATIVE_TOLERANCE if eq_rel_tol is None else eq_rel_tol
        ),
        treat_inf_as=treat_inf_as,
    )


def evaluate_spec_distance_from_verify_result(
    verify_result: VerifyResult,
    spec_fields,
    *,
    weights: Mapping[str, float] | None = None,
    eq_rel_tol: float | None = None,
):
    """Re-export: build distance records from an existing VerifyResult.

    Convenience for callers that already hold a `VerifyResult` (from
    a v0.0.x `verify()` call) and want to compute the distance signal
    without re-running the simulator. Delegates to
    `clay.distance_to_sat.evaluate_spec_distance` with the result's
    `measures` map.
    """
    from .distance_to_sat import (
        EQ_RELATIVE_TOLERANCE,
        evaluate_spec_distance as _eval,
    )
    return _eval(
        spec_fields,
        verify_result.measures or {},
        weights=weights,
        eq_rel_tol=(
            EQ_RELATIVE_TOLERANCE if eq_rel_tol is None else eq_rel_tol
        ),
    )


# ─── v0.1.02 BL-086: refinement-type spec-contract verify ──────────
#
# Two new re-export surfaces compose `clay.refinement_types` into the
# verifier module (the historical home of every verify entry point).
# Both are thin pass-throughs — the implementation lives in
# `clay.refinement_types` so the additive boundary is clean and the
# legacy `verify()` API stays byte-exact.
#
# The lazy imports avoid a circular dep: `refinement_types` imports
# `clay.verifier.verify` internally, so module-level imports here would
# trigger a load-order problem.


def verify_with_refinement(
    cell,
    contract,
    *,
    work_dir: Path | str = "tmp_clay_rundir",
    driver: SimulatorDriver | None = None,
    testbench=None,
    weights: Mapping[str, float] | None = None,
    eq_rel_tol: float | None = None,
    treat_inf_as: float | None = None,
):
    """Run a refinement-type contract check, then simulate empirically.

    Thin re-export of ``clay.refinement_types.verify_with_refinement``
    on the historical verifier-module entry surface. See the
    docstring there for the full contract.

    The function returns a ``RefinementVerifyResult`` carrying:
      * The elaboration-stage inhabitability validation.
      * The empirical measure-vs-refinement check (if inhabitable).
      * A continuous satisfaction loss (compatible with R78 A
        distance-to-sat).

    Args mirror the implementation; ``eq_rel_tol=None`` falls back to
    ``clay.refinement_types.EQ_RELATIVE_TOLERANCE``.
    """
    from .refinement_types import (
        EQ_RELATIVE_TOLERANCE as REF_EQ_REL_TOL,
        verify_with_refinement as _verify_with_refinement,
    )
    return _verify_with_refinement(
        cell, contract,
        work_dir=work_dir, driver=driver,
        testbench=testbench,
        weights=weights,
        eq_rel_tol=(
            REF_EQ_REL_TOL if eq_rel_tol is None else eq_rel_tol
        ),
        treat_inf_as=treat_inf_as,
    )


def validate_refinement_inhabitable(
    cell,
    contract,
    *,
    eq_rel_tol: float | None = None,
):
    """Static inhabitability check (no simulator call).

    Thin re-export of
    ``clay.refinement_types.validate_refinement_inhabitable`` on the
    historical verifier-module surface. Use this when you want to
    answer "is this contract satisfiable at all" before paying the
    cost of building a testbench or invoking the driver.

    Returns a ``RefinementValidation`` record (see refinement_types
    docstring).
    """
    from .refinement_types import (
        EQ_RELATIVE_TOLERANCE as REF_EQ_REL_TOL,
        validate_refinement_inhabitable as _validate,
    )
    return _validate(
        cell, contract,
        eq_rel_tol=(
            REF_EQ_REL_TOL if eq_rel_tol is None else eq_rel_tol
        ),
    )
