"""Matched-net parasitic balance — the invariant that makes a symmetric layout's
post-layout sim track its pre-layout sim on DIFFERENTIAL metrics.

The user's third requirement (2026-05-27): after parasitic extraction, post-layout
and pre-layout performance must be close. For a SYMMETRIC circuit (a diff pair,
a comparator), the differential performance (offset, CMRR, differential GBW) is
preserved through layout ONLY IF the two halves' parasitics are BALANCED — the
cap/res on ``inn`` equals that on ``inp``, on ``outn`` equals ``outp``, etc. A
geometric-mirror placement is necessary but NOT sufficient: routing or coupling
imbalance breaks the match (the v178 router's own ``known limitation`` note).

This module measures that balance directly from a :class:`PexReport` (the
calibrated Magic extraction, or the geometric estimate) over a circuit-derived
:class:`SymmetryGroup`'s differential-net pairs. The imbalance is RELATIVE
(per project rule: no absolute thresholds), so it is process- and scale-neutral.
A small ``worst_imbalance`` is the layout-level proof that the schematic symmetry
survived into the parasitics — which is what keeps post-sim ≈ pre-sim.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence, Tuple

from .base import PexReport

# Nets that are common-mode / supply rails and therefore NOT expected to mirror.
DEFAULT_GROUND_NETS: Tuple[str, ...] = ("0", "gnd", "gnd!", "vss", "vss!", "vsubs",
                                        "vsubs!", "vgnd", "vdd", "vdd!", "VSUBS")


@dataclass(frozen=True)
class NetPairBalance:
    """Parasitic balance for one differential-net pair (relative imbalance)."""

    net_a: str
    net_b: str
    cap_a_aF: float
    cap_b_aF: float
    rel_imbalance: float  # |a-b| / max(a,b), in [0, 1]


@dataclass(frozen=True)
class BalanceReport:
    """Matched-net parasitic balance over a symmetry group's differential nets."""

    cell_name: str
    pairs: Tuple[NetPairBalance, ...]
    rel_tol: float = 0.05
    backend: str = ""
    calibrated: bool = False
    field_solver: bool = False

    @property
    def worst_imbalance(self) -> float:
        return max((p.rel_imbalance for p in self.pairs), default=0.0)

    @property
    def balanced(self) -> bool:
        return self.worst_imbalance <= self.rel_tol

    @property
    def score(self) -> float:
        """1.0 = perfectly balanced; 0.0 = at/over the tolerance."""
        if not self.pairs:
            return 1.0
        return max(0.0, 1.0 - self.worst_imbalance / self.rel_tol) if self.rel_tol > 0 else 0.0


def matched_net_parasitic_balance(
    group,
    report: PexReport,
    *,
    rel_tol: float = 0.05,
) -> BalanceReport:
    """Measure parasitic balance of a SymmetryGroup's differential-net pairs.

    For each ``(net_a, net_b)`` differential pair the group declares, compares the
    total parasitic capacitance the :class:`PexReport` attributes to each net
    (substrate + coupling), as a RELATIVE imbalance ``|Ca-Cb| / max(Ca,Cb)``. A
    balanced symmetric layout keeps every pair under ``rel_tol``.
    """
    pairs: List[NetPairBalance] = []
    for (na, nb) in getattr(group, "differential_nets", ()):
        ca = report.net_total_cap_aF(na)
        cb = report.net_total_cap_aF(nb)
        denom = max(ca, cb, 1e-30)
        pairs.append(NetPairBalance(na, nb, ca, cb, abs(ca - cb) / denom))
    return BalanceReport(
        cell_name=report.cell_name,
        pairs=tuple(pairs),
        rel_tol=rel_tol,
        backend=report.backend,
        calibrated=report.calibrated,
        field_solver=report.field_solver,
    )
