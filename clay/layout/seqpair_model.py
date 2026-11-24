"""Frozen interface shared by the constraint-graph placer (Agent A) and the
layout style-transfer (Agent B) — round seqpair-placement-and-style-transfer.

Inspired by LaSTPlace (TCAD): formulate analog placement as RELATIVE ORDER on
horizontal/vertical constraint graphs (a sequence-pair). A legal placement is
overlap-free BY CONSTRUCTION (every device pair is ordered in H or V) and is
solved in O(V+E) by longest-path, so it scales past the exact MILP. The placer
(A) turns Clay's declared INTENT (symmetry pairs, common-centroid groups, rail
rows) into a :class:`RelativeOrder` and solves it; style-transfer (B) produces a
:class:`RelativeOrder` for a NEW cell by graph-aligning it to a reference layout
and transferring the reference's relative order. They compose through this type:
``solve_placement(transfer_style(target, reference_order))``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class RelativeOrder:
    """A sequence-pair-style relative-order description of a placement.

    ``left_of`` / ``below`` are the strict relations of the horizontal / vertical
    constraint graphs (``(u, v)`` means u is strictly left-of / below v). A valid
    order makes every device pair comparable in H OR V (overlap-free). ``rows``
    groups devices into rail-banded rows (each tuple is one row, bottom-to-top);
    ``mirror_pairs`` are the device pairs to center symmetrically about the
    vertical axis. ``positions`` is filled by the longest-path solver (device ->
    placed bbox in nm), empty before solving."""

    cell: str
    devices: Tuple[str, ...]
    left_of: Tuple[Tuple[str, str], ...] = ()
    below: Tuple[Tuple[str, str], ...] = ()
    rows: Tuple[Tuple[str, ...], ...] = ()
    mirror_pairs: Tuple[Tuple[str, str], ...] = ()
    axis_x: Optional[int] = None
    positions: Dict[str, Tuple[int, int, int, int]] = field(default_factory=dict)
    # True when the rows came from a DECLARED 2D ``@place_group`` floorplan (the
    # band-free, intent-driven path). The solver then applies the kind-aware
    # opposite-type well/implant gap (NMOS can sit next to PMOS). Default False
    # keeps the historical banded behaviour byte-stable.
    band_free: bool = False
    # Per-device DECLARED column index from a 2D ``@place_group ... col=N`` floorplan
    # (``{label: col}``). Drives the cross-column x-banding (every col-c device is
    # x-ordered strictly left of every higher-column device) and the PER-COLUMN
    # symmetry reconciliation (a mirror pair mirrors about ITS OWN column's axis, so
    # the symmetry snap never drags a member across a column boundary). Empty when no
    # floorplan declares columns -> the solver keeps the single-axis whole-cell
    # symmetry, so every non-floorplan placement is byte-identical.
    col_of: Dict[str, int] = field(default_factory=dict)
    # Declared ADJACENCY pairs from ``@abut(a, b)`` (``((a, b), ...)``). The solver
    # tightens the facing-edge gap between each pair to the DRC-MINIMUM (instead of
    # the default inter-device gap) on whichever axis already orders the pair, so the
    # two devices pack edge-adjacent. Empty by default so a placement that declares no
    # ``@abut`` keeps the default gaps -> byte-identical.
    abut_pairs: Tuple[Tuple[str, str], ...] = ()

    def relation_set(self) -> set:
        """The set of directed relations ``{('H',u,v)} ∪ {('V',u,v)}`` — the basis
        for the constraint-graph distance D(G,G') used to score style similarity."""
        s = set()
        for u, v in self.left_of:
            s.add(("H", u, v))
        for u, v in self.below:
            s.add(("V", u, v))
        return s


@dataclass(frozen=True)
class StyleTransferResult:
    """Output of graph-alignment style transfer: the target's transferred
    :class:`RelativeOrder` plus the node alignment + a similarity score."""

    order: RelativeOrder
    matched: Tuple[Tuple[str, str], ...]   # (target_device, reference_device)
    coverage: float                         # fraction of target devices matched
    note: str = ""


def constraint_graph_distance(a: RelativeOrder, b: RelativeOrder) -> float:
    """RELATIVE distance D(G,G') in [0,1] between two relative orders over their
    SHARED device pairs: the fraction of co-present directed relations that differ
    (LaSTPlace Eq. 17, normalised). 0.0 = identical style, 1.0 = fully opposed."""
    sa, sb = a.relation_set(), b.relation_set()
    keys_a = {(u, v) for _, u, v in sa} | {(u, v) for _, u, v in sb}
    devs = set(a.devices) & set(b.devices)
    shared = [(u, v) for (u, v) in keys_a if u in devs and v in devs]
    if not shared:
        return 0.0
    diff = 0
    for u, v in shared:
        ra = {ax for ax, x, y in sa if x == u and y == v}
        rb = {ax for ax, x, y in sb if x == u and y == v}
        if ra != rb:
            diff += 1
    return diff / len(shared)


__all__ = ["RelativeOrder", "StyleTransferResult", "constraint_graph_distance"]
