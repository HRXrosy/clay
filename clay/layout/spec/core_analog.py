"""SP2 -- the ``core_analog`` spec pack (well-supported matching / symmetry).

This pack carries the analog matching and symmetry rules whose physical basis
is established (common-centroid against process gradients, interdigitation of
current-mirror pairs, orientation matching of matched devices). Every rule
DETECTS its object purely TOPOLOGICALLY -- from the DESIGN's declared diff /
mirror / matched pairs (netlist-sourced data on ``intent``) -- and never from a
device name match or a coordinate. Every rule EMITS only categorical relations
into the zero-coordinate QSR network (``RelationNetwork``): no micron, no bbox.

The pack registers itself on import (the spec tests rely on import triggering
``register_pack``). Importing this module is the only side effect.

Rule taxonomy tuple per rule: (metric, effect, object, hardness, checker).
The matching rules all measure ``offset`` (the mismatch the layout must keep
below the matched-pair budget) -- the central analog-matching metric.
"""
from __future__ import annotations

from typing import Iterable, List, Tuple

from .packs import SpecPack, SpecRule, register_pack


# --------------------------------------------------------------------------- #
# Topological detectors (DESIGN data only -- no name branch, no coordinate).   #
# Each returns a deterministic, deduplicated list of member tuples.            #
# --------------------------------------------------------------------------- #
def _pairs(intent, *attrs: str) -> List[Tuple[str, ...]]:
    """Deduplicated, deterministically ordered member tuples drawn from the
    named pair attributes of ``intent`` (each a list of (a, b) tuples).

    Dedup is by unordered membership (a diff pair that is also a mirror pair is
    emitted once); the canonical emitted tuple is the name-sorted pair so the
    output is order-stable regardless of the source pair's internal order.
    Detection is topological: these attributes are netlist-derived design data,
    not coordinates and not name matches."""
    out: List[Tuple[str, ...]] = []
    seen: set = set()
    for attr in attrs:
        for pair in (getattr(intent, attr, None) or []):
            members = tuple(str(m) for m in pair)
            key = frozenset(members)
            if len(key) < 2 or key in seen:
                continue
            seen.add(key)
            out.append(tuple(sorted(members)))
    out.sort()
    return out


def _detect_diff_or_mirror(intent) -> List[Tuple[str, ...]]:
    """Common-centroid candidates: every declared differential or current-
    mirror pair (the pairs whose offset is dominated by the process gradient)."""
    return _pairs(intent, "diff_pairs", "mirror_pairs")


def _detect_mirror(intent) -> List[Tuple[str, ...]]:
    """Interdigitation candidates: declared current-mirror pairs."""
    return _pairs(intent, "mirror_pairs")


def _detect_matched(intent) -> List[Tuple[str, ...]]:
    """Same-orientation candidates: declared matched (equal-size) pairs."""
    return _pairs(intent, "matched_pairs")


# --------------------------------------------------------------------------- #
# Emitters (zero-coordinate: only categorical hyperedges into the network).    #
# --------------------------------------------------------------------------- #
def _emit_common_centroid(intent, group: Iterable[str], net) -> None:
    """Emit a common-centroid hyperedge over the pair's members."""
    net.add_group("common_centroid", tuple(group))


def _emit_interdigitate(intent, group: Iterable[str], net) -> None:
    """Emit an interdigitation hyperedge over the mirror pair's members."""
    net.add_group("interdigitate", tuple(group))


def _emit_same_orientation(intent, group: Iterable[str], net) -> None:
    """Emit a same-orientation hyperedge over the matched pair's members."""
    net.add_group("same_orientation", tuple(group))


# --------------------------------------------------------------------------- #
# The pack.                                                                    #
# --------------------------------------------------------------------------- #
CORE_ANALOG = SpecPack(
    name="core_analog",
    applies_to=["*"],  # universal: matching/symmetry serve every analog block.
    rules=[
        SpecRule(
            name="diffpair_common_centroid",
            metric="offset",
            effect="process_gradient",
            object="diff_pair",
            hardness="quantified_soft",
            detect=_detect_diff_or_mirror,
            emit=_emit_common_centroid,
            checker="centroid_coincidence",
        ),
        SpecRule(
            name="mirror_interdigitate",
            metric="offset",
            effect="process_gradient",
            object="mirror_pair",
            hardness="quantified_soft",
            detect=_detect_mirror,
            emit=_emit_interdigitate,
            checker="interdigitation_balance",
        ),
        SpecRule(
            name="matched_same_orientation",
            metric="offset",
            effect="orientation_mismatch",
            object="matched_pair",
            hardness="quantified_soft",
            detect=_detect_matched,
            emit=_emit_same_orientation,
            checker="orientation_match",
        ),
    ],
)


# Register on import (the spec tests rely on import triggering registration).
register_pack(CORE_ANALOG)


__all__ = ["CORE_ANALOG"]
