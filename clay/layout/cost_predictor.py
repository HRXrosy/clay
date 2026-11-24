"""Combined layout cost score.

Aggregates four independently-measured layout quality components into a
single scalar suitable for ranking candidate layouts:

  * Phase 1 mismatch sigma (mV) -- from ``mismatch.py``
  * Phase 3a area (um^2)        -- from ``symbolic_layout.py``
  * Phase 7 parasitic load (fF) -- from ``parasitic.py``
  * Phase 8d congestion crossings (count) -- from ``congestion.py``

Each component already has its own canonical computation path; this module
is a pure-Python aggregator -- it does not re-measure anything and does
not import from the upstream modules so that the existing APIs remain
untouched. Callers gather the four numbers themselves and hand them in
via :class:`LayoutCostComponents`.

The total cost is a weighted linear combination

    total = sigma * w_sigma + area * w_area
          + parasitic * w_parasitic + congestion * w_congestion

with default weights chosen so each term contributes O(1) for a "typical"
analog cell (a few mV mismatch, a few hundred um^2 area, a few hundred fF
load, a handful of crossings). Lower is better.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass(frozen=True)
class CostWeights:
    """Linear weights applied to each cost component.

    Defaults are tuned so that, for a typical analog cell, each weighted
    term lands in the same O(1) ballpark and no single component dominates
    the ranking by accident.
    """

    sigma_weight: float = 1.0
    area_weight: float = 0.01
    parasitic_weight: float = 0.001
    congestion_weight: float = 0.1


@dataclass
class LayoutCostComponents:
    """The four raw measurements gathered from upstream phases."""

    sigma_mv: float
    area_um2: float
    parasitic_fF: float
    congestion_crossings: int


@dataclass
class LayoutCostScore:
    """Result of :func:`compute_layout_cost` -- carries inputs + total."""

    components: LayoutCostComponents
    weights: CostWeights
    total: float


def compute_layout_cost(
    components: LayoutCostComponents,
    weights: CostWeights = CostWeights(),
) -> LayoutCostScore:
    """Combine the four components into a single scalar cost.

    ``total = sigma * w1 + area * w2 + parasitic * w3 + congestion * w4``

    The function is intentionally a thin wrapper so the formula is
    auditable from a single line of code. Lower totals are better.
    """

    total = (
        float(components.sigma_mv) * weights.sigma_weight
        + float(components.area_um2) * weights.area_weight
        + float(components.parasitic_fF) * weights.parasitic_weight
        + float(components.congestion_crossings) * weights.congestion_weight
    )
    return LayoutCostScore(components=components, weights=weights, total=total)


def compare_layouts(
    components_a: LayoutCostComponents,
    components_b: LayoutCostComponents,
    weights: CostWeights = CostWeights(),
) -> str:
    """Compare two candidate layouts by total cost.

    Returns the string ``"a_better"``, ``"b_better"``, or ``"tied"``.
    Equal totals (down to bit-exact float compare) are reported as
    ``"tied"`` rather than picking an arbitrary winner.
    """

    cost_a = compute_layout_cost(components_a, weights).total
    cost_b = compute_layout_cost(components_b, weights).total
    if cost_a < cost_b:
        return "a_better"
    if cost_b < cost_a:
        return "b_better"
    return "tied"


def rank_layouts(
    candidates: List[LayoutCostComponents],
    weights: CostWeights = CostWeights(),
) -> List[Tuple[int, float]]:
    """Rank candidates ascending by total cost.

    Returns a list of ``(original_index, total_cost)`` pairs sorted so
    that the cheapest (best) layout is first. Original indices are
    preserved so callers can map back to their input array. Ties keep
    Python's stable-sort order (lower index first).
    """

    scored = [
        (idx, compute_layout_cost(comp, weights).total)
        for idx, comp in enumerate(candidates)
    ]
    scored.sort(key=lambda pair: pair[1])
    return scored
