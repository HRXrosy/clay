"""Circuit symmetry -> symmetric routing (the routing half of "map symmetry to
layout consistently").

R2 bridged the circuit-derived :class:`SymmetryGroup` to the MILP PLACER. This is
the routing parallel: each differential-net pair the circuit declares becomes a
router :class:`SymPair`, so the two halves of a differential net are routed as
exact x-mirror images about the cell axis (the
``gridless_over_cell_router_v178`` already mirrors a SymPair's geometry — here we
DRIVE it from the circuit, not a hand-listed pair). Plus a routed-balance metric:
the routing-level analogue of R3's parasitic balance — two mirror-routed nets
must carry equal wirelength and be true geometric mirrors, else their parasitics
diverge and the differential match breaks.

NOTE (deferred, router-internals): the v178 router documents a via-bridge-short
"known limitation" where two tightly-pitched mirror trunks can spuriously merge
through a via cut. Fixing that union-find inside the router is a separate,
higher-risk change; this module supplies the circuit-driven SymPairs + the
balance metric that DETECTS such imbalance, leaving the internals fix as a
follow-up.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple


@dataclass(frozen=True)
class SymPair:
    """A topology-declared pair of nets that share one mirror axis."""

    left: str
    right: str


@dataclass(frozen=True)
class WireSeg:
    """One routed straight segment in integer database coordinates."""

    layer: int
    ax: int
    ay: int
    bx: int
    by: int


def symmetry_group_to_sym_pairs(group) -> List[SymPair]:
    """Map a circuit-derived SymmetryGroup's differential-net pairs to router
    SymPairs, so routing mirrors the SAME nets the circuit declares differential.
    """
    return [SymPair(left=a, right=b) for (a, b) in getattr(group, "differential_nets", ())]


def _seg_len(s: WireSeg) -> int:
    return abs(s.bx - s.ax) + abs(s.by - s.ay)


def _mirror_seg(s: WireSeg, axis_x: int) -> Tuple[int, int, int, int, int]:
    """x-reflect a segment about ``axis_x`` (y unchanged); key for mirror-match."""
    return (s.layer, 2 * axis_x - s.ax, s.ay, 2 * axis_x - s.bx, s.by)


def _seg_key(s: WireSeg) -> Tuple[int, int, int, int, int]:
    return (s.layer, s.ax, s.ay, s.bx, s.by)


@dataclass(frozen=True)
class RouteBalance:
    """Routed-geometry balance for one mirror net pair (the routing R3 analogue)."""

    left: str
    right: str
    left_len: int
    right_len: int
    len_rel_imbalance: float        # |L-R| / max(L,R)
    mirror_match_frac: float        # fraction of left segs whose x-mirror exists in right
    rel_tol: float = 0.05

    @property
    def balanced(self) -> bool:
        return self.len_rel_imbalance <= self.rel_tol and self.mirror_match_frac >= 1.0 - 1e-9


def routed_mirror_balance(
    left: str,
    right: str,
    left_segs: Sequence[WireSeg],
    right_segs: Sequence[WireSeg],
    axis_x: int,
    *,
    rel_tol: float = 0.05,
) -> RouteBalance:
    """Check two mirror-routed nets carry equal wirelength and are true x-mirrors.

    ``len_rel_imbalance`` is relative (process/scale neutral). ``mirror_match_frac``
    is the fraction of left segments whose exact x-reflection about ``axis_x``
    appears in the right net — 1.0 means perfect geometric mirror.
    """
    ll = sum(_seg_len(s) for s in left_segs)
    rl = sum(_seg_len(s) for s in right_segs)
    denom = max(ll, rl, 1)
    len_imb = abs(ll - rl) / denom

    right_keys = {_seg_key(s) for s in right_segs}
    if left_segs:
        matched = sum(1 for s in left_segs if _mirror_seg(s, axis_x) in right_keys)
        frac = matched / len(left_segs)
    else:
        frac = 1.0

    return RouteBalance(
        left=left, right=right, left_len=ll, right_len=rl,
        len_rel_imbalance=len_imb, mirror_match_frac=frac, rel_tol=rel_tol,
    )
