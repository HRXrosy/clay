"""Normalized layout-intent QUALITY SCORE: obligations -> one benchmarkable number.

The obligation checker (:mod:`clay.layout.intent_obligations`) turns a cell's
declared parasitic budgets into a list of pass/fail
:class:`~clay.layout.intent.LayoutObligation`s, and
:func:`~clay.layout.intent_obligations.layout_intent_verdict` collapses them into
a binary pass/fail. A binary verdict cannot RANK two passing layouts, nor say how
close a layout sits to its budgets. This module adds a single scalar quality
score in ``[0, 1]`` so layouts can be compared on a common axis (e.g. a Clay
layout vs a TED one), without ever reading absolute capacitances.

Why a separate scalar
---------------------
The verdict answers "is this layout signed off?"; the score answers "how good is
it?" A layout that passes every budget with a hair of headroom and one that
passes with the imbalance near zero are both ``passed=True`` in the verdict, yet
the second is the better layout -- the score distinguishes them. Because the
inputs are RELATIVE ratios (``measured_imbalance / rel_tol``), the score is
backend-neutral and scale-invariant: multiply every extracted cap by a constant
and the score does not move.

Scoring formula (RELATIVE quantities only)
------------------------------------------
For each obligation define a per-obligation quality ``q in [0, 1]`` purely from
the ratio ``r = measured_imbalance / rel_tol`` (never absolute aF). ``q`` is one
CONTINUOUS, strictly-decreasing function of ``r`` over the whole domain
``[0, inf)``, split at the budget edge ``r = 1`` by the constant ``_AT_BUDGET_Q``
(the quality of an obligation sitting exactly on its budget):

  * PASS (``measured <= rel_tol``, i.e. ``r <= 1``): a *headroom* reward that
    interpolates the band ``[_AT_BUDGET_Q, 1]`` by headroom ``h = 1 - r``:
        q = _AT_BUDGET_Q + (1 - _AT_BUDGET_Q) * h
    so ``q = 1`` at zero imbalance (full headroom) and ``q = _AT_BUDGET_Q``
    exactly at the budget. A layout that merely scrapes under budget earns little
    credit -- it is one perturbation away from failing.

  * FAIL (``measured > rel_tol``, i.e. ``r > 1``): a penalty in the band
    ``(0, _AT_BUDGET_Q]`` that decays with the relative over-budget
    ``over = r - 1`` and joins the pass branch continuously at the edge:
        q = _AT_BUDGET_Q / (1 + over)
    A just-over-budget fail (``over -> 0``) sits at ``_AT_BUDGET_Q`` (matching the
    at-budget pass) and a far-over fail decays toward 0.

  Because the pass band is ``[_AT_BUDGET_Q, 1]`` and the fail band is
  ``(0, _AT_BUDGET_Q]``, the curve is continuous and strictly decreasing in
  ``r``: a worse obligation ALWAYS scores lower, and (except for the measure-zero
  boundary ``r = 1`` itself, where they meet) every fail scores below every pass.

  * UNMEASURABLE (``measured_imbalance is None``): the relationship was never
    actually extracted, so it is unverified. It contributes a fixed
    ``q = _UNMEASURABLE_Q`` (< 1), which CAPS the aggregate strictly below 1 --
    an unverified obligation can never look as good as a verified one with full
    headroom. We do not treat it as a pass (mirrors the verdict, which blocks).

The aggregate ``score`` is the arithmetic mean of the per-obligation ``q`` (every
obligation weighted equally). Mean (not min) so that improving any single
obligation always raises the score -- useful as an optimization/benchmark target
-- while the ``weakest`` field still surfaces the binding obligation. Empty
obligation list -> ``score = 1.0`` with ``empty=True`` (vacuous, mirrors
:func:`layout_intent_verdict`).

The returned dict additionally reports ``pass_fraction`` and counts, the
``worst_margin`` (smallest pass headroom), the ``weakest`` obligation
(least-margin pass, else worst over-budget fail, else an unmeasurable one), and a
``per_kind`` breakdown (mean ``q`` + counts per budget kind). Deterministic: no
randomness, ties broken by obligation order.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from .intent import LayoutObligation

# Quality of an obligation sitting EXACTLY on its budget (r == 1): the seam
# between the pass band [_AT_BUDGET_Q, 1] and the fail band (0, _AT_BUDGET_Q].
# Small (an at-budget layout is fragile) but > 0 so a true pass always beats a
# fail of any overshoot.
_AT_BUDGET_Q = 0.25
# An UNMEASURABLE (unverified) obligation contributes this fixed quality, which
# holds the aggregate strictly below 1. It is NOT a pass (mirrors the verdict);
# placed at 0.5 -- unchecked is worse than a comfortable pass yet not as bad as
# a measured failure.
_UNMEASURABLE_Q = 0.5


def _is_exact_constraint(o: LayoutObligation) -> bool:
    """True when the obligation is a binary exact predicate.

    Relational placement claims such as ``@order`` intentionally lower with a
    zero tolerance: zero violation passes and every positive violation fails.
    Such a predicate has no finite ``measured / tolerance`` ratio, so it must be
    scored on that binary contract rather than sent through the ratio curve.
    """
    return o.rel_tol == 0.0


def _obligation_quality(o: LayoutObligation) -> float:
    """Per-obligation quality ``q in [0, 1]`` from the RELATIVE ratio alone.

    See the module docstring for the piecewise formula. Reads only
    ``measured_imbalance / rel_tol`` (a relative ratio) -- never an absolute cap.
    """
    if o.measured_imbalance is None:           # unmeasurable / unverified
        return _UNMEASURABLE_Q
    if _is_exact_constraint(o):
        return 1.0 if o.measured_imbalance <= 0.0 else 0.0
    r = o.measured_imbalance / o.rel_tol       # relative ratio in [0, inf)
    if r <= 1.0:                               # pass: reward the headroom band
        h = 1.0 - r                            # headroom in [0, 1]
        return _AT_BUDGET_Q + (1.0 - _AT_BUDGET_Q) * h
    over = r - 1.0                             # fail: relative over-budget
    return _AT_BUDGET_Q / (1.0 + over)


def _pass_headroom(o: LayoutObligation) -> Optional[float]:
    """Headroom ``1 - measured/rel_tol`` for a PASSING obligation, else None."""
    if o.status != "pass":
        return None
    if _is_exact_constraint(o):
        # Exact predicates have no continuous tolerance band.  A measured pass
        # is full satisfaction; a positive measurement is already a fail.
        return 1.0
    return 1.0 - (o.measured_imbalance / o.rel_tol)


def _weakest_key(o: LayoutObligation):
    """Sort key putting the WEAKEST obligation first (lowest per-obligation ``q``).

    Ordering by the per-obligation quality ``q`` (lower == weaker), which gives:
    measured FAILs (decaying from the at-budget seam toward 0, worst overshoot
    weakest), then near/at-budget PASSes, then UNMEASURABLE obligations
    (``_UNMEASURABLE_Q``), then comfortable PASSes -- i.e. the binding obligation
    is whatever is closest to (or furthest past) its budget, with an unverified
    obligation treated as middling rather than the worst. ``min`` is stable, so
    on an exact ``q`` tie the earliest obligation wins (deterministic).
    """
    return _obligation_quality(o)


def _label(o: LayoutObligation) -> str:
    """Human-readable ``kind(a~b)`` identifier for an obligation."""
    a, b = o.pair
    return f"{o.kind}({a}~{b})"


def layout_intent_score(
    obligations: Sequence[LayoutObligation],
) -> Dict[str, object]:
    """Aggregate obligations into a single normalized quality score in ``[0, 1]``.

    See the module docstring for the full formula. Operates purely on a sequence
    of :class:`LayoutObligation` (relative ratios only), so it is fast, PEX/Magic
    independent, and unit-testable. Deterministic.

    Returns a dict with:

      * ``score`` -- mean per-obligation quality in ``[0, 1]`` (1 = every budget
        passes with full headroom; drops as obligations approach/exceed budget;
        any unmeasurable obligation caps it strictly below 1). Empty -> 1.0.
      * ``pass_fraction`` -- fraction of obligations that are a measured pass
        (0.0 for the empty case).
      * ``n_pass`` / ``n_fail`` / ``n_unmeasurable`` / ``n_unsupported`` /
        ``n_total`` -- disjoint verification-state counts.
      * ``worst_margin`` -- the SMALLEST pass headroom (``1 - measured/rel_tol``)
        among passing obligations; ``None`` when none pass.
      * ``weakest`` -- a dict describing the least-margin obligation
        (``kind``, ``pair``, ``label``, ``status``, ``quality``, and either the
        pass ``headroom`` or the fail ``over_budget`` relative overshoot);
        ``None`` only for the empty case.
      * ``per_kind`` -- per-budget-kind breakdown: ``{kind: {mean_quality,
        n_pass, n_fail, n_unmeasurable, n_unsupported, n_total}}``.
      * ``empty`` -- True iff there are no obligations (then ``score == 1.0``,
        mirroring :func:`~clay.layout.intent_obligations.layout_intent_verdict`).
    """
    obs: List[LayoutObligation] = list(obligations)
    n_total = len(obs)

    if n_total == 0:
        return {
            "score": 1.0,
            "pass_fraction": 0.0,
            "n_pass": 0,
            "n_fail": 0,
            "n_unmeasurable": 0,
            "n_unsupported": 0,
            "n_total": 0,
            "worst_margin": None,
            "weakest": None,
            "per_kind": {},
            "empty": True,
        }

    qualities = [_obligation_quality(o) for o in obs]
    score = sum(qualities) / n_total

    n_pass = sum(1 for o in obs if o.status == "pass")
    n_fail = sum(1 for o in obs if o.status == "fail")
    n_unmeasurable = sum(1 for o in obs if o.status == "unmeasurable")
    n_unsupported = sum(1 for o in obs if o.status == "unsupported")
    pass_fraction = n_pass / n_total

    headrooms = [h for h in (_pass_headroom(o) for o in obs) if h is not None]
    worst_margin = min(headrooms) if headrooms else None

    # Weakest = lowest per-obligation quality. ``min`` is stable, so on exact ties
    # the earliest obligation wins -> deterministic.
    weakest_ob = min(obs, key=_weakest_key)
    weakest: Dict[str, object] = {
        "kind": weakest_ob.kind,
        "pair": weakest_ob.pair,
        "label": _label(weakest_ob),
        "status": weakest_ob.status,
        "quality": _obligation_quality(weakest_ob),
    }
    if weakest_ob.status == "pass":
        weakest["headroom"] = _pass_headroom(weakest_ob)
    elif weakest_ob.status == "fail":
        if _is_exact_constraint(weakest_ob):
            # Relative overshoot is undefined against a zero denominator.  Keep
            # the JSON evidence finite and explicit instead of emitting Inf.
            weakest["over_budget"] = None
            weakest["exact_constraint"] = True
        else:
            # Relative over-budget (r - 1); matches the formula's ``over``.
            weakest["over_budget"] = (
                weakest_ob.measured_imbalance / weakest_ob.rel_tol
            ) - 1.0
    # unmeasurable: neither headroom nor over_budget is meaningful.

    per_kind: Dict[str, Dict[str, object]] = {}
    for o, q in zip(obs, qualities):
        bucket = per_kind.setdefault(
            o.kind,
            {"_q_sum": 0.0, "n_pass": 0, "n_fail": 0,
             "n_unmeasurable": 0, "n_unsupported": 0, "n_total": 0},
        )
        bucket["_q_sum"] += q
        bucket["n_total"] += 1
        if o.status == "pass":
            bucket["n_pass"] += 1
        elif o.status == "fail":
            bucket["n_fail"] += 1
        elif o.status == "unsupported":
            bucket["n_unsupported"] += 1
        else:
            bucket["n_unmeasurable"] += 1
    for bucket in per_kind.values():
        q_sum = bucket.pop("_q_sum")
        bucket["mean_quality"] = q_sum / bucket["n_total"]

    return {
        "score": score,
        "pass_fraction": pass_fraction,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_unmeasurable": n_unmeasurable,
        "n_unsupported": n_unsupported,
        "n_total": n_total,
        "worst_margin": worst_margin,
        "weakest": weakest,
        "per_kind": per_kind,
        "empty": False,
    }
