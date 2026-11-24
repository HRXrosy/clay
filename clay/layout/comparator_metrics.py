"""Coupling / kickback / shielding metrics for the dynamic comparator layout.

These are the objective functions the layout iterates against. A dynamic
comparator has large-swing AGGRESSOR nets (the clock, the regenerative cross
nodes, the rail-to-rail outputs) that capacitively couple into SENSITIVE VICTIM
nets (the analog inputs, the pre-amp outputs). Two electrical failure modes
follow directly:

  * **Kickback** — coupling from the switching aggressors back onto the input
    nodes ``net0``/``net1``. The damage is not the absolute coupling but its
    IMBALANCE between the two inputs: a balanced kick is a common-mode glitch the
    differential pair rejects; an imbalanced kick injects a differential offset.
  * **Regeneration / cross-coupling skew** — the clock must couple EQUALLY into
    the two regenerative nodes ``crossn``/``crossp`` (and the outputs), or the
    latch resolves with a built-in decision offset.

Every metric is computed from a :class:`~clay.layout.pex.base.PexReport`, so the
whole harness is testable WITHOUT Magic/Docker by passing a synthetic report.
The geometric ``shielding_between`` reads the routed cell's ``tagged`` bboxes.

Per the project-wide rule, every imbalance reported here is RELATIVE (a
fraction), never an absolute aF delta -- so the numbers are process- and
scale-neutral.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

from .pex.base import PexReport
from .pex.symmetry_balance import matched_net_parasitic_balance

# Large-swing nets that capacitively assault their neighbours during a decision.
DYNAMIC_AGGRESSORS: Tuple[str, ...] = ("clk", "crossn", "crossp", "outm", "outp")
# Quiet analog nets whose voltage we do not want disturbed.
SENSITIVE_VICTIMS: Tuple[str, ...] = ("net0", "net1", "intern", "interp")
# Schematic-derived differential-net pairs (must stay parasitically matched).
DIFF_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("intern", "interp"),
    ("crossn", "crossp"),
    ("outm", "outp"),
    ("net0", "net1"),
)
# Inputs that suffer kickback.
DEFAULT_INPUTS: Tuple[str, str] = ("net0", "net1")
# Common-mode / supply rails for coupling-subset purposes.
DEFAULT_GROUND_NETS: Tuple[str, ...] = ("gnd", "vdd", "0", "vss", "vsubs", "VSUBS")


def _rel_imbalance(a: float, b: float) -> float:
    """Relative imbalance between two non-negative magnitudes, in [0, 1].

    Uses the MEAN as the denominator so a (90, 110) pair reports 0.20 -- the
    convention the kickback/regeneration checks want (deviation as a fraction of
    the typical value). ``differential_balance`` uses max-denominator via the
    shared ``symmetry_balance`` helper; both are documented where used.
    """
    mean = (a + b) / 2.0
    if mean <= 0.0:
        return 0.0
    return abs(a - b) / mean


def coupling_cap_aF(report: PexReport, net_a: str, net_b: str) -> float:
    """Total coupling capacitance (aF) directly between ``net_a`` and ``net_b``.

    Sums every parasitic cap whose two terminals are EXACTLY the unordered pair
    ``{net_a, net_b}``. A self-cap (``net_a == net_b``) is meaningless coupling
    and returns 0.0.
    """
    if net_a == net_b:
        return 0.0
    want = frozenset((net_a, net_b))
    return sum(c.cap_aF for c in report.caps if frozenset((c.a, c.b)) == want)


def cc_matrix(
    report: PexReport,
    aggressors: Sequence[str] = DYNAMIC_AGGRESSORS,
    victims: Sequence[str] = SENSITIVE_VICTIMS,
) -> Dict[str, float]:
    """Aggressor x victim coupling-cap matrix, keyed ``"agg~vic"`` (aF).

    Self-pairs (an aggressor that is also a victim) are skipped.
    """
    out: Dict[str, float] = {}
    for agg in aggressors:
        for vic in victims:
            if agg == vic:
                continue
            out[f"{agg}~{vic}"] = coupling_cap_aF(report, agg, vic)
    return out


def kickback_cc(
    report: PexReport,
    aggressors: Sequence[str] = DYNAMIC_AGGRESSORS,
    inputs: Tuple[str, str] = DEFAULT_INPUTS,
) -> Dict[str, object]:
    """Total dynamic-aggressor coupling onto each input + their RELATIVE imbalance.

    For each input net, sums the coupling from every aggressor (excluding the
    input itself if it appears in ``aggressors``). ``imbalance`` is the relative
    difference between the two inputs' totals, ``|c0 - c1| / mean`` -- balanced
    kickback is common-mode (rejected); imbalanced kickback is a differential
    offset.
    """
    a, b = inputs
    ca = sum(coupling_cap_aF(report, agg, a) for agg in aggressors if agg != a)
    cb = sum(coupling_cap_aF(report, agg, b) for agg in aggressors if agg != b)
    return {
        "inputs": (a, b),
        f"{a}_total_cc_aF": ca,
        f"{b}_total_cc_aF": cb,
        "imbalance": _rel_imbalance(ca, cb),
    }


def clk_regen_balance(
    report: PexReport,
    *,
    clock_net: str = "clk",
    pair: Tuple[str, str] = ("crossn", "crossp"),
) -> Dict[str, object]:
    """Clock-into-regenerative-pair coupling balance (clock_net~pair[0] vs ~pair[1]).

    The clock/strobe edge must couple EQUALLY into the two regenerative nodes or
    the regeneration starts with a built-in decision offset. Reports the two
    coupling caps and their RELATIVE imbalance.

    ``clock_net`` and ``pair`` default to the dynamic-comparator convention
    (``clk`` against ``(crossn, crossp)``) so legacy call sites stay byte-identical,
    but a fast-strobe / differential-clock comparator can override them so the
    SAME metric serves any cell category whose regenerative pair is aggressed by a
    non-``clk`` net. The keys in the returned dict are templated on the actual
    nets so a downstream consumer can recover which clock/pair was measured.
    """
    a, b = pair
    cn = coupling_cap_aF(report, clock_net, a)
    cp = coupling_cap_aF(report, clock_net, b)
    return {
        f"{clock_net}_{a}_aF": cn,
        f"{clock_net}_{b}_aF": cp,
        "clock_net": clock_net,
        "pair": (a, b),
        "imbalance": _rel_imbalance(cn, cp),
    }


def differential_balance(
    report: PexReport,
    pairs: Sequence[Tuple[str, str]] = DIFF_PAIRS,
) -> Dict[str, object]:
    """Relative total-cap imbalance per differential pair + the worst pair.

    Reuses :func:`matched_net_parasitic_balance` (max-denominator relative
    imbalance over each net's TOTAL parasitic cap) by wrapping ``pairs`` in the
    minimal ``differential_nets`` shape that helper consumes.
    """
    group = _DiffNetGroup(tuple(pairs))
    bal = matched_net_parasitic_balance(group, report)
    per_pair = {
        f"{p.net_a}~{p.net_b}": {
            "cap_a_aF": p.cap_a_aF,
            "cap_b_aF": p.cap_b_aF,
            "imbalance": p.rel_imbalance,
        }
        for p in bal.pairs
    }
    worst = max(bal.pairs, key=lambda p: p.rel_imbalance, default=None)
    return {
        "pairs": per_pair,
        "worst_imbalance": bal.worst_imbalance,
        "worst_pair": (worst.net_a, worst.net_b) if worst else None,
    }


class _DiffNetGroup:
    """Minimal duck-typed stand-in for a SymmetryGroup (only ``differential_nets``).

    ``matched_net_parasitic_balance`` reads exactly one attribute, so the metrics
    module stays decoupled from ``clay.layout.symmetry_derive``.
    """

    __slots__ = ("differential_nets",)

    def __init__(self, differential_nets: Tuple[Tuple[str, str], ...]) -> None:
        self.differential_nets = differential_nets


def _bbox_of(tagged_entry) -> Tuple[int, int, int, int]:
    """(x0, y0, x1, y1) of a ``(net, layer_key, bbox)`` tagged entry."""
    return tagged_entry[2]


def _net_bbox(cell, net: str) -> Optional[Tuple[int, int, int, int]]:
    """Union bbox of all shapes tagged with ``net`` in a routed cell, or None."""
    xs0, ys0, xs1, ys1 = [], [], [], []
    for entry in cell.tagged:
        if entry[0] == net:
            b = _bbox_of(entry)
            xs0.append(b[0]); ys0.append(b[1]); xs1.append(b[2]); ys1.append(b[3])
    if not xs0:
        return None
    return (min(xs0), min(ys0), max(xs1), max(ys1))


def shielding_between(
    cell,
    aggressor_net: str,
    victim_net: str,
    gnd_net: str = "gnd",
) -> Dict[str, object]:
    """Detect whether a grounded shape lies spatially between two signal nets.

    Heuristic (Manhattan, channel routing): take the union bbox of every shape
    tagged ``aggressor_net``, of ``victim_net``, and of every ``gnd_net`` shape.
    The aggressor and victim are separated along whichever axis their bboxes do
    NOT overlap (the gap axis); the perpendicular axis is the overlap span where
    a shield would have to sit to block the field line. A gnd shape SHIELDS iff:

      * its span on the gap axis falls strictly inside the aggressor->victim gap
        (it is geometrically between them), AND
      * it overlaps the aggressor/victim shared span on the perpendicular axis
        (it actually stands in the way, not off to the side).

    Returns a dict with ``shielded`` (bool) and the supporting geometry so the
    result is inspectable. If either signal net is absent, or their bboxes overlap
    (no clean gap to shield), ``shielded`` is False with a ``reason``.
    """
    agg = _net_bbox(cell, aggressor_net)
    vic = _net_bbox(cell, victim_net)
    if agg is None or vic is None:
        return {"shielded": False, "reason": "aggressor or victim net not present",
                "aggressor_bbox": agg, "victim_bbox": vic}

    # Decide the separation axis: the one on which the two bboxes do not overlap.
    x_gap = max(agg[0], vic[0]) - min(agg[2], vic[2])   # >0 => disjoint in x
    y_gap = max(agg[1], vic[1]) - min(agg[3], vic[3])   # >0 => disjoint in y

    if x_gap > 0 and (y_gap <= 0 or x_gap >= y_gap):
        axis = "x"
        gap_lo, gap_hi = min(agg[2], vic[2]), max(agg[0], vic[0])
        perp_lo, perp_hi = max(agg[1], vic[1]), min(agg[3], vic[3])
        gi0, gi1, pi0, pi1 = 0, 2, 1, 3
    elif y_gap > 0:
        axis = "y"
        gap_lo, gap_hi = min(agg[3], vic[3]), max(agg[1], vic[1])
        perp_lo, perp_hi = max(agg[0], vic[0]), min(agg[2], vic[2])
        gi0, gi1, pi0, pi1 = 1, 3, 0, 2
    else:
        return {"shielded": False, "reason": "aggressor and victim overlap (no gap)",
                "aggressor_bbox": agg, "victim_bbox": vic}

    for entry in cell.tagged:
        if entry[0] != gnd_net:
            continue
        g = _bbox_of(entry)
        between = g[gi0] >= gap_lo and g[gi1] <= gap_hi
        blocks = g[pi1] > perp_lo and g[pi0] < perp_hi
        if between and blocks:
            return {"shielded": True, "axis": axis, "shield_bbox": g,
                    "aggressor_bbox": agg, "victim_bbox": vic,
                    "gap": (gap_lo, gap_hi)}

    return {"shielded": False, "reason": "no gnd shape in the gap",
            "axis": axis, "aggressor_bbox": agg, "victim_bbox": vic,
            "gap": (gap_lo, gap_hi)}


def comparator_metrics_report(
    report: Optional[PexReport] = None,
    cell=None,
) -> Dict[str, object]:
    """Unified metrics report. PEX metrics need ``report``; geometric need ``cell``.

    Returns a plain JSON-serializable dict. PEX-derived sections (coupling matrix,
    kickback, clk regeneration balance, differential balance) appear iff a
    :class:`PexReport` is supplied; the geometric shielding survey appears iff a
    routed ``cell`` is supplied. Either may be omitted.
    """
    out: Dict[str, object] = {}
    if report is not None:
        out["cell_name"] = report.cell_name
        out["calibrated"] = report.calibrated
        out["cc_matrix_aF"] = cc_matrix(report)
        out["kickback"] = kickback_cc(report)
        out["clk_regen_balance"] = clk_regen_balance(report)
        out["differential_balance"] = differential_balance(report)
    if cell is not None:
        shielding: Dict[str, object] = {}
        for agg in DYNAMIC_AGGRESSORS:
            for vic in SENSITIVE_VICTIMS:
                if agg == vic:
                    continue
                res = shielding_between(cell, agg, vic)
                shielding[f"{agg}~{vic}"] = bool(res.get("shielded"))
        out["shielding"] = shielding
    return out
