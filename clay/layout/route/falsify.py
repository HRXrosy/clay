"""Migration falsification for routing widths.

The routing analogue of the placement A/B leakage proof (:func:`migrate_smic.leakage_ab`):
a width *table* MEASURED from the reference layout is keyed by the reference net names, so it
does not survive transfer; a DESIGN-DERIVED model (:func:`width_model.derive_width_intent`)
re-derives correct widths from any netlist. :func:`width_leakage_ab` quantifies the gap.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from clay.layout.route.intent import RoutingIntent
from clay.layout.route.verify import _width_class
from clay.layout.route.width_model import (
    DEFAULT_POWER_BASES,
    WidthPolicy,
    classify_net_role,
    derive_width_intent,
    reference_width_intent,
    _all_nets,
    _norm,
    _pdk_floor_um,
)


def _expected_class(net: str, intent, *, policy: WidthPolicy) -> str:
    """The design-intended width class of a net = its netlist role."""
    return classify_net_role(net, intent, policy=policy)


def width_leakage_ab(
    primary_intent,
    primary_ref_widths: Dict[str, float],
    holdout_intent,
    tech,
    *,
    layer: str = "metal1",
    policy: Optional[WidthPolicy] = None,
) -> dict:
    """Compare the LEAKED reference-width table (A) vs the DESIGN-derived model (B) on a
    held-out design realised in ``tech``.

    Returns a dict with, for each arm, the per-net width-class error count (predicted class
    vs the net's design-intended role class), the table coverage on the held-out nets, and
    the leakage count.
    """
    pol = policy or WidthPolicy()
    holdout_nets = _all_nets(holdout_intent)
    n = len(holdout_nets) or 1
    floor_um = _pdk_floor_um(tech, layer)

    # ground-truth: each net's design-intended class (from the held-out netlist).
    expected = {net: _expected_class(net, holdout_intent, policy=pol) for net in holdout_nets}

    # --- A) leaked reference-width table, applied by net name -----------------
    leaked = {k: float(v) for k, v in primary_ref_widths.items()}
    leaked_norm = {_norm(k): v for k, v in leaked.items()}
    covered = 0
    leaked_errors = 0
    for net in holdout_nets:
        key = _norm(net)
        if key in leaked_norm:
            covered += 1
            w = leaked_norm[key]
        else:
            w = floor_um  # uncovered -> the router falls back to the min-width floor
        pred = _width_class(w, is_power=False,
                            cap_threshold_um=pol.cap_min_um,
                            trunk_threshold_um=pol.power_min_um)
        if pred != expected[net]:
            leaked_errors += 1

    # --- B) design-derived width model on the held-out netlist ----------------
    design = derive_width_intent(holdout_intent, tech, layer=layer, policy=pol)
    design_errors = 0
    for net in holdout_nets:
        w = design[net].value
        pred = _width_class(w, is_power=False,
                            cap_threshold_um=pol.cap_min_um,
                            trunk_threshold_um=pol.power_min_um)
        if pred != expected[net]:
            design_errors += 1

    leaked_intent = RoutingIntent(cell="leaked", widths=reference_width_intent(leaked))
    design_intent = RoutingIntent(cell="design", widths=design)

    return {
        "holdout_n_nets": len(holdout_nets),
        "leaked_coverage": covered / n,
        "design_coverage": 1.0,  # the model derives a width for every net by construction
        "leaked_class_errors": leaked_errors,
        "design_class_errors": design_errors,
        "leaked_leakage_count": leaked_intent.leakage_count(),
        "design_leakage_count": design_intent.leakage_count(),
    }


def falsify_sweep(
    primary_intent,
    primary_ref_widths: Dict[str, float],
    holdouts: Sequence[Tuple[str, object]],
    tech,
    *,
    layer: str = "metal1",
    policy: Optional[WidthPolicy] = None,
) -> dict:
    """Run :func:`width_leakage_ab` over a SET of held-out designs and aggregate.

    The falsification is robust (not anecdotal) iff, on EVERY held-out, the design model has
    0 class-errors + 0 leakage and the leaked table is overall strictly worse. ``holdouts``
    is a sequence of ``(name, ExtractedIntent)``.
    """
    rows: List[dict] = []
    d_err = d_leak = l_err = l_leak = 0
    for name, ho in holdouts:
        r = width_leakage_ab(primary_intent, primary_ref_widths, ho, tech,
                             layer=layer, policy=policy)
        r["name"] = name
        rows.append(r)
        d_err += r["design_class_errors"]
        d_leak += r["design_leakage_count"]
        l_err += r["leaked_class_errors"]
        l_leak += r["leaked_leakage_count"]
    design_wins_all = all(
        r["design_class_errors"] == 0
        and r["design_leakage_count"] == 0
        and (r["leaked_class_errors"] > 0 or r["leaked_coverage"] < r["design_coverage"])
        for r in rows
    )
    return {
        "n_holdouts": len(rows),
        "design_total_class_errors": d_err,
        "design_total_leakage": d_leak,
        "leaked_total_class_errors": l_err,
        "leaked_total_leakage": l_leak,
        "design_wins_all": design_wins_all,
        "rows": rows,
    }
