"""Parasitic-symmetry quality gate — the objective rule a density-optimized layout
like v183 FAILS even at symmetry_score 1.0.

Measured fact (2026-05-27, real Magic on the v183 dense-router comparator GDS):
the device-placement symmetry score is 1.0, DRC 0, LVS matches -- yet the 10
signal nets' extracted parasitic caps, best-case paired, are 6.3% / 12.1% / 18.4%
IMBALANCED on 3 of 5 pairs. The dense router optimizes density/co-clean, routes
the differential nets without mirror constraint or parasitic balance, and the GDS
does not even LABEL the signal nets. So a comparator can pass every existing gate
and still be electrically broken (input offset, CMRR loss) from parasitic skew.

This gate makes matched-net parasitic balance an ENFORCEABLE objective rule: given
the circuit-derived differential-net pairs (clay.layout.symmetry_derive) and a
PexReport (calibrated Magic, or geometric), every pair must balance within
``rel_tol`` (relative, per project rule). It FAILS the v183-class layout and is
meant to sit in the layout-quality flywheel beside DRC/LVS/symmetry_score so
"passes the gates" finally implies "electrically symmetric".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from .base import PexReport
from .symmetry_balance import NetPairBalance, matched_net_parasitic_balance


@dataclass(frozen=True)
class ParasiticSymmetryVerdict:
    """PASS/FAIL of the matched-net parasitic-balance gate over a symmetry group."""

    passed: bool
    rel_tol: float
    pairs: Tuple[NetPairBalance, ...]
    worst_imbalance: float
    worst_pair: Optional[Tuple[str, str]]
    backend: str
    calibrated: bool
    field_solver: bool

    @property
    def detail(self) -> str:
        if not self.pairs:
            return "no differential-net pairs to check"
        wp = f"{self.worst_pair[0]}/{self.worst_pair[1]}" if self.worst_pair else "-"
        verdict = "PASS" if self.passed else "FAIL"
        return (
            f"parasitic-symmetry {verdict}: worst {wp} "
            f"rel_imbalance={self.worst_imbalance:.3f} vs tol {self.rel_tol} "
            f"({self.backend}, calibrated={self.calibrated}, field_solver={self.field_solver})"
        )


def parasitic_symmetry_gate(
    group,
    report: PexReport,
    *,
    rel_tol: float = 0.05,
) -> ParasiticSymmetryVerdict:
    """Gate a layout's parasitics against the circuit's differential-net symmetry.

    PASS iff every differential-net pair's relative cap imbalance <= ``rel_tol``.
    An empty group (no differential nets) PASSes vacuously (nothing to balance).
    """
    bal = matched_net_parasitic_balance(group, report, rel_tol=rel_tol)
    worst = max(bal.pairs, key=lambda p: p.rel_imbalance, default=None)
    return ParasiticSymmetryVerdict(
        passed=bal.balanced,
        rel_tol=rel_tol,
        pairs=bal.pairs,
        worst_imbalance=bal.worst_imbalance,
        worst_pair=(worst.net_a, worst.net_b) if worst else None,
        backend=report.backend,
        calibrated=report.calibrated,
        field_solver=report.field_solver,
    )
