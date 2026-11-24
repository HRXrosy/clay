"""Unified symmetry-quality gate: placement + routing + parasitic symmetry in ONE
objective verdict.

The whole point of this lane (user, 2026-05-27): a layout that passes the
geometric placement-symmetry score is NOT electrically symmetric unless its
ROUTING mirrors the differential nets AND the extracted PARASITICS balance. The
v183 comparator is the proof: R172 symmetry_placement = 1.0, DRC 0, LVS match,
but 18.4% differential-net parasitic imbalance (real Magic). So placement symmetry
alone is a FALSE PASS.

This gate composes the three levels into one verdict, each optional (scored only
if its evidence is supplied):

  * PLACEMENT  -- the R172 ``symmetry_placement`` score (device-centroid mirror),
    passes at ``placement_tol`` (default 0.95).
  * ROUTING    -- every differential-net pair's routed geometry is a true mirror
    with balanced wirelength (clay.layout.symmetry_route.RouteBalance).
  * PARASITIC  -- every differential-net pair's extracted cap balances within tol
    (clay.layout.pex.symmetry_quality_gate.ParasiticSymmetryVerdict).

overall_passed is True iff EVERY supplied check passes. A perfect placement score
can NOT carry a failing parasitic check -- the gate fails the v183-class layout.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence


@dataclass(frozen=True)
class SymmetryQualityReport:
    """Composite PASS/FAIL across the placement / routing / parasitic levels."""

    placement_score: Optional[float]
    placement_passed: Optional[bool]
    routing_balanced: Optional[bool]
    routing_worst_pair: Optional[str]
    parasitic_passed: Optional[bool]
    parasitic_worst_imbalance: Optional[float]
    overall_passed: bool
    summary: str


def evaluate_symmetry_quality(
    *,
    placement_score: Optional[float] = None,
    placement_tol: float = 0.95,
    route_balances: Optional[Sequence] = None,
    parasitic_verdict: Optional[object] = None,
) -> SymmetryQualityReport:
    """Compose the three symmetry levels into one verdict.

    Supply whichever evidence exists; ``overall_passed`` requires every supplied
    level to pass. At least one level must be supplied (else the verdict is
    vacuous and fails).
    """
    checks: List[bool] = []
    parts: List[str] = []

    pl_pass: Optional[bool] = None
    if placement_score is not None:
        pl_pass = placement_score >= placement_tol
        checks.append(pl_pass)
        parts.append(f"placement {placement_score:.3f}{'>=' if pl_pass else '<'}{placement_tol} "
                     f"[{'PASS' if pl_pass else 'FAIL'}]")

    rt_bal: Optional[bool] = None
    rt_worst: Optional[str] = None
    if route_balances is not None:
        rbs = list(route_balances)
        rt_bal = all(rb.balanced for rb in rbs)
        checks.append(rt_bal)
        bad = [rb for rb in rbs if not rb.balanced]
        if bad:
            w = max(bad, key=lambda rb: rb.len_rel_imbalance)
            rt_worst = f"{w.left}/{w.right}"
        parts.append(f"routing {len(rbs)} pair(s) [{'PASS' if rt_bal else 'FAIL'}"
                     f"{'' if rt_bal else ' worst ' + str(rt_worst)}]")

    pa_pass: Optional[bool] = None
    pa_worst: Optional[float] = None
    if parasitic_verdict is not None:
        pa_pass = bool(parasitic_verdict.passed)
        pa_worst = float(parasitic_verdict.worst_imbalance)
        checks.append(pa_pass)
        parts.append(f"parasitic worst {pa_worst:.3f} [{'PASS' if pa_pass else 'FAIL'}]")

    overall = bool(checks) and all(checks)
    summary = ("symmetry-quality " + ("PASS" if overall else "FAIL") + ": "
               + " | ".join(parts) if parts else "symmetry-quality FAIL: no evidence supplied")

    return SymmetryQualityReport(
        placement_score=placement_score,
        placement_passed=pl_pass,
        routing_balanced=rt_bal,
        routing_worst_pair=rt_worst,
        parasitic_passed=pa_pass,
        parasitic_worst_imbalance=pa_worst,
        overall_passed=overall,
        summary=summary,
    )
