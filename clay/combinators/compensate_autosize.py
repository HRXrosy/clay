"""
Compensate auto-sizing — Newton + bisection + Adjoint search (v0.1.12 split).

Hosts the v0.0.91 / v0.0.92 auto-sizing machinery for
`compensate(target_pm_deg=...)`. Public-ish entry point is
`_autosize_compensate`; everything else here is internal scaffolding
the auto-sizer needs (probe-build, Newton-step, bisection, Adjoint
gradient consult, PM measurement, I/O node resolution).

The split keeps the user-facing `compensate.py` light (< 300 LOC) so
the API surface and the search/probe machinery can be read
independently.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..ir import Cell, Spec
from ..stage_interface import ComposedStage
from ._common import (
    TemplateBuilder,
    _is_finite,
    _resolve_autosize_testbench,
)


# Initial seed for the auto-sizing search. Picked to be a reasonable
# starting point for sky130-style 2-stage opamps; the search will
# diverge from these values as it bisects.
_AUTOSIZE_SEED_CC: float = 1.0e-12     # 1 pF
_AUTOSIZE_SEED_RZ: float = 10.0e3      # 10 kΩ

# Bounds for the bisection / Newton clip. These keep the search inside
# physically realistic device sizes (no femtofarad caps that disappear
# in parasitic noise, no nanofarad caps that take the entire amplifier
# off-chip).
_AUTOSIZE_CC_MIN: float = 5.0e-14      # 0.05 pF
_AUTOSIZE_CC_MAX: float = 1.0e-10      # 100 pF

# Convergence band: ±5 % of the target PM (so target_pm=60 means stop
# once measured PM is in [57, 63]).
_AUTOSIZE_PM_TOL: float = 0.05

# Maximum number of ngspice probes the bisection-fallback path runs.
# The Newton path has its own cap (`_AUTOSIZE_NEWTON_MAX_ITERS`); when
# Newton hands off to bisection the combined trace is still bounded by
# `_AUTOSIZE_MAX_ITERS` so the overall worst case matches the v0.0.91
# guarantee of "≤ 5 ngspice probes per compensate call".
_AUTOSIZE_MAX_ITERS: int = 5

# v0.0.92 Newton-step constants — Newton tries to land target PM in at
# most this many iterations on TOP of the seed probe. So the Newton
# happy path costs 1 + 2 = 3 ngspice probes worst case; bisection
# (which costs 5 probes worst case on its own) only fires when Newton
# can't converge inside its own clip band.
_AUTOSIZE_NEWTON_MAX_ITERS: int = 2

# Newton clip band: each Newton candidate Cc must stay inside
# [seed / clip_factor, seed × clip_factor]. Keeps a single divergent
# step from blowing past the bisection's physical bounds.
_AUTOSIZE_NEWTON_CLIP_FACTOR: float = 10.0

# Empirical PM-vs-log(Cc) slope for the canonical 2-stage Miller
# topology on sky130 1.8 V. Used as the initial Newton-step slope
# fallback when the Adjoint bridge can't supply a non-zero ∂PM/∂Cc.
#
# v0.0.93 lifecycle:
#   * Primary slope source: v0.0.93 R69 E's
#     ``adjoint_gradient_pm_op_resolve`` PyO3 method. On cells with
#     a properly-normalized Miller cap on the AC stamp, that method
#     returns a real ∂PM/∂Cc (deg / F) which the Newton step uses
#     directly. The trace records ``slope_source = "adjoint"``.
#   * Fallback (this constant): kicks in when the adjoint surface
#     returns 0.0 / None (e.g. when the bridge Cap-field
#     normalization fix from Round 70 Agent A hasn't landed yet on
#     stdlib cells, or when ``clay_ir`` isn't built). The trace
#     records ``slope_source = "empirical"`` so callers can tell
#     which path was taken.
#
# Sign convention: PM DECREASES as log10(Cc) increases for this
# topology (larger Cc → more phase shift at GBW → smaller PM).
#
# Trend from the v0.0.91 empirical table (Cc=0.05 pF → PM≈96°,
# Cc=1 pF → PM≈89.4°, Cc=30 pF → PM≈88.6°):
#   slope_over_decade ≈ (89.4 - 96) / log10(1pF / 0.05pF) ≈ -5.0 deg/decade
# We use a conservative magnitude (−4.0 deg/decade) so the first
# Newton step is intentionally undersized — overshooting Cc by an
# order of magnitude is a worse outcome than taking one extra probe.
_AUTOSIZE_NEWTON_EMPIRICAL_SLOPE_DEG_PER_DECADE: float = -4.0


def _autosize_compensate(
    inner: ComposedStage,
    cell_name: str,
    spec: Spec,
    kwargs: Dict[str, Any],
    target_pm_deg: float,
) -> Tuple[Optional[float], Optional[float], List[Dict[str, Any]]]:
    """Search for Cc / Rz that land PM near `target_pm_deg`.

    Two-tier strategy (v0.0.92):

      Tier 1 — Newton step on Cc (DEFAULT):
        Probe seed Cc, read measured PM, take a Newton step in
        log-Cc space using `∂PM/∂log10(Cc)`. The slope is sourced
        from v0.0.93's R69 E ``adjoint_gradient_pm_op_resolve`` PyO3
        method (real ∂PM/∂Cc via one DC OP re-solve per knob); when
        that returns zero / unavailable, falls back to a conservative
        empirical slope (−4 deg / decade). The trace's
        ``slope_source`` field records which path was used. At most
        `_AUTOSIZE_NEWTON_MAX_ITERS` Newton iterations; each candidate
        is clipped to `[seed / N, seed × N]` where N =
        `_AUTOSIZE_NEWTON_CLIP_FACTOR`. Empirically converges in 1-2
        probes (vs the bisection's 4-5) on canonical 2-stage Miller
        targets in [60°, 110°].

      Tier 2 — Bisection (FALLBACK):
        Same bracket-and-bisect as v0.0.91. Activates when Newton
        couldn't converge inside its clip band OR a Newton step
        landed outside band but the next step pointed outside the
        clip. The bisection inherits the trace + bracket state from
        Newton, so the total ngspice probe count is bounded by
        `_AUTOSIZE_MAX_ITERS` (5).

    The Adjoint gradient (∂PM/∂Cc) is consulted on the SEED probe to
    seed the Newton slope. v0.0.93 primary path: R69 E's
    ``CellGraphHandle.adjoint_gradient_pm_op_resolve``. The fallback
    empirical slope kicks in transparently if that returns zero (e.g.
    when Round 70 Agent A's stdlib Cap-field normalization fix hasn't
    landed yet — gradient is then structurally zero); the trace
    records which slope source each Newton step used.

    Sign convention on every topology this auto-sizer ships against
    (canonical 2-stage Miller on sky130): LARGER Cc → LOWER PM.
    Newton respects that sign automatically because `∂PM/∂log10(Cc)`
    carries the sign.

    Returns:
      `(cc, rz, trace)` — derived Cc / Rz values + per-iteration trace
      records. Each record has at minimum `{iter, Cc, Rz, PM, status}`.
      Newton records also carry `method="newton"`, `slope`,
      `slope_source ∈ {"adjoint","empirical","secant"}`, and `step_log10`.
      Bisection records carry `method="bisection"` and the v0.0.91
      bracket fields. The trace is preserved on
      `meta['auto_search_trace']` so callers can inspect the search.

      Returns `(seed_cc, seed_rz, [])` and an empty trace when probing
      isn't possible (e.g. ngspice not on PATH, the template has no
      testbench module). The empty-trace return is the documented
      "degrade gracefully" surface — callers can detect it via
      `len(meta['auto_search_trace']) == 0`.
    """
    trace: List[Dict[str, Any]] = []

    # Stage 1: try to find a testbench builder for the wrapped template.
    # The compensate combinator can wrap leaf templates OR composed
    # cascades. For composed surfaces we walk to the FIRST leaf (the
    # left-most input stage in a cascade chain) since that's the
    # canonical testbench template for an opamp's AC characterization.
    tb_builder, supply = _resolve_autosize_testbench(inner)
    if tb_builder is None:
        # Couldn't find a testbench module — degrade to seed values.
        return _AUTOSIZE_SEED_CC, _AUTOSIZE_SEED_RZ, trace

    # Stage 2: try Newton step path FIRST. Newton's seed probe is
    # iter 0; subsequent Newton iterations are iter 1, 2, …, up to
    # `_AUTOSIZE_NEWTON_MAX_ITERS`. If Newton converges, we're done.
    # Otherwise we hand off the seed bracket to bisection.
    newton_result = _newton_step_compensate(
        inner, cell_name, spec, kwargs,
        target_pm_deg, tb_builder, supply, trace,
    )
    if newton_result is not None:
        # Newton converged or returned the best-known value within its
        # own clip band — trace is final.
        return newton_result

    # Stage 3: Newton didn't converge inside its clip band. Hand off to
    # bisection, which resumes from the last Newton probe and uses any
    # bracketing probes already in `trace`. The combined trace length
    # is bounded by `_AUTOSIZE_MAX_ITERS` (the v0.0.91 hard ceiling).
    return _bisection_compensate(
        inner, cell_name, spec, kwargs,
        target_pm_deg, tb_builder, supply, trace,
    )


def _newton_step_compensate(
    inner: ComposedStage,
    cell_name: str,
    spec: Spec,
    kwargs: Dict[str, Any],
    target_pm_deg: float,
    tb_builder: Callable,
    supply: Optional[float],
    trace: List[Dict[str, Any]],
) -> Optional[Tuple[float, float, List[Dict[str, Any]]]]:
    """Newton-step auto-sizing on Cc using ∂PM/∂log10(Cc).

    Adjoint-driven Newton step convergence path (v0.0.92):

      1. Seed Cc / Rz value.
      2. Run ONE ngspice probe + read measured PM.
      3. If in band → return (1 ngspice probe total).
      4. Consult Adjoint gradient ∂PM/∂Cc via `compute_gradients`. If
         non-zero, convert to slope-per-decade. Otherwise fall back to
         the empirical slope constant.
      5. Newton step in log10(Cc) space:
           Cc_new = Cc × 10 ^ ((PM_target − PM_meas) / slope_per_decade)
      6. Clip to [seed / CLIP, seed × CLIP]. If clipped at the boundary
         AND the boundary candidate equals a previously-probed Cc,
         bail (Newton can't make progress).
      7. Re-probe. If in band → return (2 probes total). Otherwise
         use the secant slope from the last two probes for iter 2.
      8. At most `_AUTOSIZE_NEWTON_MAX_ITERS` Newton steps. If not
         converged, return None so the caller hands off to bisection.

    Trace records added per Newton step:
      `{iter, Cc, Rz, PM, status, method='newton', slope, slope_source,
        step_log10, clipped}`. Status uses the same lexicon as bisection
      (`in_band` / `over_compensated` / `under_compensated`) plus
      `newton_clipped` (Newton step clipped to bounds) and
      `newton_no_slope` (Newton couldn't pick a slope at all — both
      adjoint and empirical returned zero, which means the toplogy is
      effectively flat in PM-vs-Cc and Newton can't help).
    """
    rz = _AUTOSIZE_SEED_RZ
    cc = _AUTOSIZE_SEED_CC
    cc_clip_lo = cc / _AUTOSIZE_NEWTON_CLIP_FACTOR
    cc_clip_hi = cc * _AUTOSIZE_NEWTON_CLIP_FACTOR

    # Seed probe (iter 0).
    pm_meas = _probe_pm_with_trace(
        inner, cell_name, spec, kwargs, cc, rz,
        tb_builder, supply, trace, iter_idx=0,
        method_tag="newton",
    )
    if pm_meas is None:
        # Probe failed at seed — Newton can't proceed. Caller falls
        # back to bisection (which retries the same seed probe; the
        # repeated work is fine because bisection caches via its own
        # try/except — and a seed probe failure usually means the
        # whole template is unsimulatable so bisection will also bail).
        return None

    rel_err = (pm_meas - target_pm_deg) / target_pm_deg
    seed_status = "in_band" if abs(rel_err) <= _AUTOSIZE_PM_TOL else (
        "over_compensated" if pm_meas > target_pm_deg
        else "under_compensated"
    )
    if trace:
        trace[-1]["status"] = seed_status
        trace[-1]["rel_err"] = rel_err
    if abs(rel_err) <= _AUTOSIZE_PM_TOL:
        # Seed already in band — 1 probe and we're done.
        return cc, rz, trace

    # v0.0.93: try the Adjoint gradient for slope-per-decade. The
    # primary path is R69 E's ``adjoint_gradient_pm_op_resolve`` PyO3
    # method on the bridged handle (real ∂PM/∂Cc on cells with a
    # Miller cap on the AC stamp); the v0.0.91 ``compute_gradients``
    # path is the fallback for older binaries. When the adjoint
    # surface can't produce a non-zero slope (bridge Cap field not yet
    # normalized for stdlib cells, or pre-v0.0.93 build), the helper
    # returns None and Newton falls through to the empirical slope —
    # the slope_source field in the trace records which path was
    # taken.
    adjoint_slope_per_decade = _adjoint_slope_pm_per_log10_cc(
        inner, cell_name, spec, kwargs, cc,
    )

    last_cc: float = cc
    last_pm: float = pm_meas

    for newton_iter in range(_AUTOSIZE_NEWTON_MAX_ITERS):
        # Pick the slope for this Newton iteration.
        # - Iter 0 (first Newton step): adjoint (if non-zero) else empirical.
        # - Iter 1+: prefer secant slope from the last two probes
        #   (much more accurate on the actual PM(Cc) curve).
        slope_per_decade: Optional[float] = None
        slope_source: str = ""
        if newton_iter == 0:
            if adjoint_slope_per_decade is not None \
                    and abs(adjoint_slope_per_decade) > 1e-12:
                slope_per_decade = adjoint_slope_per_decade
                slope_source = "adjoint"
            else:
                slope_per_decade = (
                    _AUTOSIZE_NEWTON_EMPIRICAL_SLOPE_DEG_PER_DECADE
                )
                slope_source = "empirical"
        else:
            # Secant from the last two probes — need at least 2 trace
            # entries with non-None PM. Walk back through trace for
            # the most recent two.
            recent: List[Tuple[float, float]] = []
            for entry in reversed(trace):
                pm_e = entry.get("PM")
                cc_e = entry.get("Cc")
                if pm_e is None or cc_e is None:
                    continue
                if not _is_finite(float(pm_e)):
                    continue
                if cc_e <= 0:
                    continue
                recent.append((float(cc_e), float(pm_e)))
                if len(recent) >= 2:
                    break
            if len(recent) >= 2 \
                    and abs(math.log10(recent[0][0]) - math.log10(recent[1][0])) > 1e-9:
                slope_per_decade = (
                    (recent[0][1] - recent[1][1])
                    / (math.log10(recent[0][0]) - math.log10(recent[1][0]))
                )
                slope_source = "secant"
            else:
                # Can't form secant — fall back to whichever first-iter
                # slope we used.
                if adjoint_slope_per_decade is not None \
                        and abs(adjoint_slope_per_decade) > 1e-12:
                    slope_per_decade = adjoint_slope_per_decade
                    slope_source = "adjoint"
                else:
                    slope_per_decade = (
                        _AUTOSIZE_NEWTON_EMPIRICAL_SLOPE_DEG_PER_DECADE
                    )
                    slope_source = "empirical"

        if slope_per_decade is None or abs(slope_per_decade) < 1e-12:
            # No usable slope — Newton can't propose a step.
            trace.append({
                "iter": len(trace),
                "Cc": last_cc,
                "Rz": rz,
                "PM": last_pm,
                "status": "newton_no_slope",
                "method": "newton",
                "slope": slope_per_decade,
                "slope_source": slope_source or "none",
            })
            # Hand off to bisection.
            return None

        # Newton step in log10 space: solve PM_target = PM_meas + slope * step.
        pm_gap = target_pm_deg - last_pm
        step_log10 = pm_gap / slope_per_decade
        cc_candidate = last_cc * (10.0 ** step_log10)

        # Clip to Newton band [seed / CLIP, seed × CLIP]. Honesty:
        # we clip to BOTH the Newton band AND the physical absolute
        # bounds; the Newton band is the tighter one in normal use.
        cc_new = max(min(cc_candidate, cc_clip_hi), cc_clip_lo)
        cc_new = max(min(cc_new, _AUTOSIZE_CC_MAX), _AUTOSIZE_CC_MIN)
        clipped = (cc_new != cc_candidate)

        # If the clipped candidate equals a previously-probed Cc to
        # within numerical noise, Newton can't make progress.
        already_probed = any(
            abs(math.log10(max(float(e.get("Cc") or cc_new), 1e-30))
                - math.log10(cc_new)) < 1e-6
            for e in trace
            if e.get("Cc") is not None and float(e.get("Cc") or 0) > 0
        )
        if already_probed:
            # Record the attempted step before bailing so the trace
            # explains why Newton stopped.
            trace.append({
                "iter": len(trace),
                "Cc": cc_new,
                "Rz": rz,
                "PM": None,
                "status": "newton_clipped",
                "method": "newton",
                "slope": slope_per_decade,
                "slope_source": slope_source,
                "step_log10": step_log10,
                "clipped": True,
            })
            return None

        # Probe at the Newton-stepped Cc.
        pm_meas_new = _probe_pm_with_trace(
            inner, cell_name, spec, kwargs, cc_new, rz,
            tb_builder, supply, trace, iter_idx=len(trace),
            method_tag="newton",
            extra={
                "slope": slope_per_decade,
                "slope_source": slope_source,
                "step_log10": step_log10,
                "clipped": clipped,
            },
        )
        if pm_meas_new is None:
            # Probe failed at the Newton step — return what we had.
            return last_cc, rz, trace

        last_cc = cc_new
        last_pm = pm_meas_new

        rel_err = (pm_meas_new - target_pm_deg) / target_pm_deg
        step_status = "in_band" if abs(rel_err) <= _AUTOSIZE_PM_TOL else (
            "over_compensated" if pm_meas_new > target_pm_deg
            else "under_compensated"
        )
        if trace:
            trace[-1]["status"] = step_status
            trace[-1]["rel_err"] = rel_err
        if abs(rel_err) <= _AUTOSIZE_PM_TOL:
            # Newton converged.
            return last_cc, rz, trace

    # Newton ran out of iterations without converging — fall back to
    # bisection. Return None to signal the caller.
    return None


def _adjoint_slope_pm_per_log10_cc(
    inner: ComposedStage,
    cell_name: str,
    spec: Spec,
    kwargs: Dict[str, Any],
    cc: float,
) -> Optional[float]:
    """Try to get ∂PM/∂log10(Cc) from the Adjoint bridge.

    v0.0.93 flip: the primary path now consults the v0.0.93 R69 E PyO3
    method ``adjoint_gradient_pm_op_resolve`` on the cell's hypergraph
    handle. That method runs one DC OP re-solve per knob and returns a
    real ∂PM/∂Cc (deg / F) on cells with a Miller cap — closing the gap
    the v0.0.91 W-flavoured ``adjoint_gradient_pm_w`` left open. The
    Newton step picks ``slope_source = "adjoint"`` automatically when
    the value is non-zero.

    The legacy v0.0.91 ``compute_gradients(metric="pm",
    sizing_params=["Cmiller"])`` call is retained as a fallback for
    binaries that predate v0.0.93 (the method probe via ``hasattr``
    keeps the surface forward-compatible). Both paths converge on the
    same return contract.

    Honest-scope graceful-degradation contract:
      * Returns None when ``clay_ir`` is unavailable, when the seed
        cell cannot be built / lowered, when the I/O nodes can't be
        resolved, when the Cmiller instance isn't on the bridged
        handle, or when the gradient returns 0.0 / non-finite. The
        Newton-step caller treats None as "no slope available" and
        falls back to ``_AUTOSIZE_NEWTON_EMPIRICAL_SLOPE_DEG_PER_DECADE``
        with ``slope_source = "empirical"``.
      * In particular, if Round 70 Agent A's bridge Cap-field
        normalization fix isn't merged yet, the bridged Cell's Cap
        node may store the value under the legacy ``"value"`` key
        while the Rust AC stamp reads ``"Cc"`` / ``"C"``. In that case
        the adjoint gradient comes back 0.0 and we degrade to the
        empirical slope — the slope_source tag tells the caller which
        path was taken.

    Returns ∂PM/∂log10(Cc) in deg/decade (the unit the Newton step
    expects), or None when no usable slope is available.
    """
    try:
        from .. import _ir_bridge
        if not _ir_bridge.has_rust_ir():
            return None
        seed_cell = _build_compensate_probe_cell(
            inner, cell_name + "_grad_seed", spec, kwargs,
            miller=cc, zero=_AUTOSIZE_SEED_RZ,
        )
        in_node, out_node = _autosize_io_nodes(seed_cell)
        if in_node is None or out_node is None:
            return None

        # v0.0.93 primary path: call R69 E's PyO3 method on the handle
        # directly. We bridge the seed cell to a handle, locate the
        # Cmiller node id, then call adjoint_gradient_pm_op_resolve
        # with the (nid, "Cc") knob. The op-resolve path returns a
        # real ∂PM/∂Cc on cells with a Miller cap on the AC stamp.
        dpm_dcc = _adjoint_pm_cc_via_op_resolve(
            _ir_bridge, seed_cell, in_node, out_node,
        )

        # v0.0.91 legacy fallback: if the v0.0.93 method isn't
        # available (e.g. on a pre-v0.0.93 clay_ir binary) OR returned
        # 0.0 (e.g. unbridged Cap fields), fall back to the W-flavoured
        # ``compute_gradients`` path. On cap knobs that returns 0.0
        # too, which propagates to the None / empirical-slope fallback.
        if dpm_dcc is None or abs(dpm_dcc) < 1e-30:
            grads = _ir_bridge.compute_gradients(
                seed_cell, in_node, out_node,
                sizing_params=["Cmiller"],
                supply_vdd=1.8, pdk_id=seed_cell.pdk_id, metric="pm",
            )
            if grads is None:
                return None
            dpm_dcc = float(grads.get("Cmiller", 0.0))

        if dpm_dcc is None or abs(dpm_dcc) < 1e-30:
            return None
        if not _is_finite(dpm_dcc):
            return None
        # ∂PM/∂Cc in deg / F. Convert to deg / decade of Cc via
        # ∂PM/∂log10(Cc) = (∂PM/∂Cc) × Cc × ln(10).
        slope_per_decade = dpm_dcc * cc * math.log(10.0)
        if not _is_finite(slope_per_decade):
            return None
        return slope_per_decade
    except Exception:  # noqa: BLE001
        return None


def _adjoint_pm_cc_via_op_resolve(
    bridge,
    seed_cell: Cell,
    in_node: str,
    out_node: str,
) -> Optional[float]:
    """Call the v0.0.93 R69 E PyO3 method ``adjoint_gradient_pm_op_resolve``
    on a bridged handle and return ∂PM/∂Cc (deg / F) on the Cmiller node.

    Returns None when:
      * ``clay_ir`` cannot bridge the cell to a handle.
      * The handle doesn't expose ``adjoint_gradient_pm_op_resolve``
        (pre-v0.0.93 binary).
      * The handle has no ``Cmiller`` instance (cell didn't ship the
        compensate cap on this build).
      * The method call raises (caller catches and falls through to
        the v0.0.91 legacy path).

    Returns 0.0 when the bridge does have a Cmiller node but the
    op-resolve gradient is structurally zero (e.g. the Cap value field
    didn't normalize through to the Rust AC stamp — the Round 70 Agent
    A bridge-fix scope). The caller treats 0.0 the same as None and
    falls back to the legacy / empirical path.
    """
    try:
        handle = bridge.cell_to_hypergraph(seed_cell)
        if handle is None:
            return None
        # Pre-v0.0.93 binaries don't ship this method; forward-compat
        # via hasattr means a stale Rust build won't crash the Python
        # side — we just fall through to the v0.0.91 fallback.
        if not hasattr(handle, "adjoint_gradient_pm_op_resolve"):
            return None
        # Locate the Cmiller node id by walking the handle. Mirrors
        # the name-to-nid scan in bridge.compute_gradients.
        cmiller_nid: Optional[int] = None
        try:
            n_nodes = handle.node_count()
        except Exception:  # noqa: BLE001
            return None
        for nid in range(n_nodes):
            try:
                inst_name = handle.node_instance_name(nid)
            except Exception:  # noqa: BLE001
                continue
            if inst_name == "Cmiller":
                cmiller_nid = nid
                break
        if cmiller_nid is None:
            return None
        # One knob: ∂PM/∂Cc on the Cmiller node.
        triples = handle.adjoint_gradient_pm_op_resolve(
            in_node, out_node, [(cmiller_nid, "Cc")],
            1.8, seed_cell.pdk_id, 0.05,
        )
        if not triples:
            return None
        # Shape: [(nid, kind, gradient)].
        _nid, _kind, grad = triples[0]
        return float(grad)
    except Exception:  # noqa: BLE001
        return None


def _probe_pm_with_trace(
    inner: ComposedStage,
    cell_name: str,
    spec: Spec,
    kwargs: Dict[str, Any],
    cc: float,
    rz: float,
    tb_builder: Callable,
    supply: Optional[float],
    trace: List[Dict[str, Any]],
    *,
    iter_idx: int,
    method_tag: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Optional[float]:
    """Build a probe cell, run one ngspice measurement, append a trace
    record, and return the measured PM (or None on failure).

    Shared by both the Newton and bisection loops so trace format
    stays uniform. The caller passes `method_tag="newton"` or
    `method_tag="bisection"`; the field lands on every record.
    """
    try:
        probe_cell = _build_compensate_probe_cell(
            inner, cell_name + f"_probe{iter_idx}", spec, kwargs,
            miller=cc, zero=rz,
        )
        pm_meas = _measure_pm(probe_cell, tb_builder, supply)
    except Exception as exc:  # noqa: BLE001
        rec: Dict[str, Any] = {
            "iter": iter_idx,
            "Cc": cc,
            "Rz": rz,
            "PM": None,
            "status": "probe_failed",
            "method": method_tag,
            "error": f"{type(exc).__name__}: {exc}",
        }
        if extra:
            rec.update(extra)
        trace.append(rec)
        return None

    if pm_meas is None or not _is_finite(pm_meas):
        rec = {
            "iter": iter_idx,
            "Cc": cc,
            "Rz": rz,
            "PM": pm_meas,
            "status": "pm_not_resolved",
            "method": method_tag,
        }
        if extra:
            rec.update(extra)
        trace.append(rec)
        return None

    # Compute status against the target. We can't read target here
    # without plumbing it through; instead, the helper takes whatever
    # status the caller wants, derived from `pm_meas`. We append a
    # pre-populated entry and let the caller patch `status` if needed.
    rec = {
        "iter": iter_idx,
        "Cc": cc,
        "Rz": rz,
        "PM": pm_meas,
        "method": method_tag,
    }
    if extra:
        rec.update(extra)
    trace.append(rec)
    return pm_meas


def _bisection_compensate(
    inner: ComposedStage,
    cell_name: str,
    spec: Spec,
    kwargs: Dict[str, Any],
    target_pm_deg: float,
    tb_builder: Callable,
    supply: Optional[float],
    trace: List[Dict[str, Any]],
) -> Tuple[Optional[float], Optional[float], List[Dict[str, Any]]]:
    """Bracket-and-bisect on Cc — the v0.0.91 fallback path.

    Reuses any probes already in `trace` (from a prior Newton attempt)
    to seed cc_lo / cc_hi when possible. The total trace length is
    bounded by `_AUTOSIZE_MAX_ITERS` regardless of where the loop
    starts; Newton probes count against the budget.

    Strategy is unchanged from v0.0.91:
      * If measured PM > target+5%: PM too high → INCREASE Cc.
      * If measured PM < target-5%: PM too low  → DECREASE Cc.
      * Once cc_lo and cc_hi both pinned, bisect geometrically
        (sqrt of the bracket) and probe again.

    PM is monotonic-but-saturating in Cc for the canonical 2-stage
    Miller topology — LARGER Cc → LOWER PM (the second pole moves
    into the GBW band as GBW drops, adding phase shift).
    """
    # Seed cc_lo / cc_hi from prior trace if possible.
    cc_lo: Optional[float] = None
    cc_hi: Optional[float] = None
    for entry in trace:
        cc_e = entry.get("Cc")
        pm_e = entry.get("PM")
        if cc_e is None or pm_e is None:
            continue
        try:
            cc_f = float(cc_e)
            pm_f = float(pm_e)
        except (TypeError, ValueError):
            continue
        if not _is_finite(cc_f) or not _is_finite(pm_f) or cc_f <= 0:
            continue
        if pm_f > target_pm_deg:
            cc_lo = cc_f if cc_lo is None else max(cc_lo, cc_f)
        else:
            cc_hi = cc_f if cc_hi is None else min(cc_hi, cc_f)

    # Start from the seed (or the last Newton probe) — whichever
    # matters less because the bracket determines the path forward.
    cc = _AUTOSIZE_SEED_CC
    rz = _AUTOSIZE_SEED_RZ
    last_cc: Optional[float] = None
    last_rz: Optional[float] = None
    if trace:
        last = trace[-1]
        if last.get("Cc") is not None:
            try:
                cc = float(last["Cc"])
                last_cc = cc
            except (TypeError, ValueError):
                pass

    while len(trace) < _AUTOSIZE_MAX_ITERS:
        iter_idx = len(trace)
        # Need a probe at the candidate Cc unless the previous trace
        # entry IS at this Cc with a resolved PM (then we can reuse it).
        prior_pm: Optional[float] = None
        if trace and trace[-1].get("Cc") is not None \
                and trace[-1].get("PM") is not None:
            try:
                if abs(math.log10(float(trace[-1]["Cc"])) - math.log10(cc)) < 1e-9:
                    prior_pm = float(trace[-1]["PM"])
            except (TypeError, ValueError):
                prior_pm = None

        if prior_pm is None:
            pm_meas = _probe_pm_with_trace(
                inner, cell_name, spec, kwargs, cc, rz,
                tb_builder, supply, trace, iter_idx=iter_idx,
                method_tag="bisection",
                extra={"cc_lo": cc_lo, "cc_hi": cc_hi},
            )
            if pm_meas is None:
                # Probe failed — return last-known-good.
                if last_cc is not None:
                    return last_cc, last_rz, trace
                return _AUTOSIZE_SEED_CC, _AUTOSIZE_SEED_RZ, trace
        else:
            pm_meas = prior_pm

        last_cc, last_rz = cc, rz
        rel_err = (pm_meas - target_pm_deg) / target_pm_deg
        within_band = abs(rel_err) <= _AUTOSIZE_PM_TOL

        # Patch status onto the just-added trace entry (or the prior
        # one if we reused it).
        status_str = "in_band" if within_band else (
            "over_compensated" if pm_meas > target_pm_deg
            else "under_compensated"
        )
        if trace and trace[-1].get("Cc") is not None:
            trace[-1]["status"] = status_str
            trace[-1]["rel_err"] = rel_err

        if within_band:
            return cc, rz, trace

        if pm_meas > target_pm_deg:
            cc_lo = cc if cc_lo is None else max(cc_lo, cc)
            if cc_hi is None:
                cc_next = min(cc * 2.0, _AUTOSIZE_CC_MAX)
                if cc_next >= _AUTOSIZE_CC_MAX * 0.999:
                    return cc_next, rz, trace
                cc = cc_next
            else:
                cc = math.sqrt(cc_lo * cc_hi)
        else:
            cc_hi = cc if cc_hi is None else min(cc_hi, cc)
            if cc_lo is None:
                cc_next = max(cc / 2.0, _AUTOSIZE_CC_MIN)
                if cc_next <= _AUTOSIZE_CC_MIN * 1.001:
                    return cc_next, rz, trace
                cc = cc_next
            else:
                cc = math.sqrt(cc_lo * cc_hi)

    # Hit the iteration cap without convergence — return last probed.
    return (
        last_cc if last_cc is not None else _AUTOSIZE_SEED_CC,
        last_rz if last_rz is not None else _AUTOSIZE_SEED_RZ,
        trace,
    )


def _build_compensate_probe_cell(
    inner: ComposedStage,
    cell_name: str,
    spec: Spec,
    kwargs: Dict[str, Any],
    *,
    miller: float,
    zero: float,
) -> Cell:
    """Build a `Cell` with the candidate Miller cap + nulling resistor
    attached. Used by the auto-sizing probe loop.

    Internally constructs a one-shot `compensate(inner, miller=miller,
    zero=zero)` ComposedStage (BYPASSING auto-sizing, since we want
    explicit numerics here), lowers it, and calls the builder.

    Note: this constructs the composition INSIDE the probe loop on every
    iteration. The composition object is lightweight (a dataclass plus
    refs); the heavy work is in the builder call which the loop has to
    do anyway.
    """
    from ._lower import lower_to_template
    # Strip kwargs that would re-trigger auto_bias side effects in the
    # probe build — we want a "clean" build identical to a normal
    # explicit-compensate call.
    probe_kwargs = dict(kwargs)
    # The probe builds the same inner template — the kwargs DO pass through.

    probe_composed = ComposedStage(
        kind="compensate",
        interface=inner.interface,
        parts=(inner,),
        meta={"miller": float(miller), "zero": float(zero)},
    )
    probe_builder = lower_to_template(probe_composed)
    return probe_builder(cell_name, spec, **probe_kwargs)


def _measure_pm(
    cell: Cell,
    tb_builder: Callable,
    supply: Optional[float],
) -> Optional[float]:
    """Run a single ngspice probe on `cell` and extract PM (degrees).

    The standard `build_ac_testbench` family reports `phase_at_gbw` in
    radians (because the inline `.control` block does not set
    `set units=degrees` — it uses the testbench module's own measure
    formula). We convert the measured phase to degrees and compute
    `pm = 180 - |phase_deg|` here.

    Returns None when:
      * The simulator errored out.
      * `phase_at_gbw` is absent from the result measures.
      * `phase_at_gbw` is NaN or infinite.
    """
    # Lazy imports — the auto-sizing path is opt-in, so the verifier
    # / driver modules should not be imported unconditionally.
    from ..verifier import verify

    if supply is None:
        tb = tb_builder(cell)
    else:
        tb = tb_builder(cell, supply=supply)

    # Use a tmp work dir per probe so probes don't collide on disk.
    import tempfile
    with tempfile.TemporaryDirectory(prefix="clay_autosize_") as tmp:
        result = verify(tb, work_dir=tmp)
    if result.simulator_error is not None:
        return None
    phase = result.measures.get("phase_at_gbw")
    if phase is None:
        return None
    try:
        phase_f = float(phase)
    except (TypeError, ValueError):
        return None
    if not _is_finite(phase_f):
        return None
    # The phase is in RADIANS (the testbench formula uses ngspice's
    # default phase units). Convert to degrees + compute PM.
    phase_deg = phase_f * 180.0 / math.pi
    pm = 180.0 - abs(phase_deg)
    if not _is_finite(pm):
        return None
    return pm


def _adjoint_direction_hint(
    inner: ComposedStage,
    cell_name: str,
    spec: Spec,
    kwargs: Dict[str, Any],
) -> Optional[Dict[str, float]]:
    """Best-effort Adjoint gradient call on the seed probe cell.

    Returns a dict mapping `{ 'Cc': ∂PM/∂Cc, 'Rz': ∂PM/∂Rz }` when the
    `clay_ir` extension is available AND a build at the seed values
    succeeds. Returns None otherwise. The auto-sizing loop currently
    only uses this for telemetry — bisection direction is chosen from
    the probe PM itself, not from the gradient sign. The hook is here
    so a future round can switch to a Newton step on PM target without
    the bisection fallback.
    """
    try:
        from .. import _ir_bridge
        if not _ir_bridge.has_rust_ir():
            return None
        # We build the seed cell once to feed the bridge. The cell shape
        # is the same as the bisection's iter-0 probe.
        seed_cell = _build_compensate_probe_cell(
            inner, cell_name + "_grad_seed", spec, kwargs,
            miller=_AUTOSIZE_SEED_CC, zero=_AUTOSIZE_SEED_RZ,
        )
        in_node, out_node = _autosize_io_nodes(seed_cell)
        if in_node is None or out_node is None:
            return None
        # Query the gradient on Cmiller's W-knob (the cap's "value" knob).
        # The Adjoint bridge's W-flavoured gradient kind doesn't natively
        # cover caps + resistors yet — we attempt the call but accept
        # None as the documented degrade case.
        grads = _ir_bridge.compute_gradients(
            seed_cell, in_node, out_node,
            sizing_params=["Cmiller", "Rnullz"],
            supply_vdd=1.8, pdk_id=seed_cell.pdk_id, metric="pm",
        )
        if grads is None:
            return None
        return {
            "Cc": float(grads.get("Cmiller", 0.0)),
            "Rz": float(grads.get("Rnullz", 0.0)),
        }
    except Exception:  # noqa: BLE001
        return None


def _autosize_io_nodes(cell: Cell) -> Tuple[Optional[str], Optional[str]]:
    """Pick the (input, output) ports the Adjoint bridge should use.

    Heuristic: the input is the first port whose name is in
    {`in`, `inp`} (single-ended OR positive-differential input); the
    output is `out` if present, else the first signal_out-shaped port.
    Returns `(None, None)` when the cell doesn't have a recognisable
    pair.
    """
    in_candidates = ["in", "inp"]
    out_candidates = ["out", "op", "vout"]
    in_node: Optional[str] = None
    out_node: Optional[str] = None
    for p in cell.ports:
        if in_node is None and p in in_candidates:
            in_node = p
        if out_node is None and p in out_candidates:
            out_node = p
    return in_node, out_node
