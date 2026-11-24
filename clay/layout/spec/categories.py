"""SP4 -- per-circuit-category spec packs (opamp / comparator / bandgap).

Each circuit category loads the universal ``core_analog`` pack (SP2, applies_to
``["*"]``) PLUS a category-specific pack registered here. The category tag is a
LOAD-SELECTION key (``packs_for_category`` filters by ``applies_to``); it is NOT
a per-device special case. Every rule below DETECTS its object purely
TOPOLOGICALLY -- from the DESIGN's declared diff / mirror / matched pairs and
passives (netlist-sourced data on ``intent``) -- never from a device-name match
and never from a coordinate. Every rule EMITS only categorical relations into
the zero-coordinate QSR network (``RelationNetwork``): no micron, no bbox.

Each pack registers itself on import; importing this module is the only side
effect (the spec tests rely on import triggering ``register_pack``).

Content (not just name) distinguishes the packs:
  * opamp      -- compensation-node parasitic minimization + output-stage
                  (diff/mirror) symmetry. Metrics: bandwidth / cmrr.
  * comparator -- clock-net shielding against kickback coupling + regenerative-
                  pair symmetry. Carries a kickback/coupling rule the opamp pack
                  does NOT (metric=noise, effect=coupling).
  * bandgap    -- reference-device common-centroid matching + reference-branch
                  symmetry. Metrics: offset / psrr.

Real geometric metrics for the new checkers are deferred to ``register_checker``
(SP3). Until then their named checkers resolve to honest ``advisory`` /
``unsupported`` via the SP3 verdict layer -- never a fake pass.
"""
from __future__ import annotations

from typing import Iterable, List, Tuple

from .packs import SpecPack, SpecRule, register_pack


# --------------------------------------------------------------------------- #
# Shared topological detectors (DESIGN data only -- no name branch, no coord). #
# Each returns a deterministic, deduplicated list of member tuples.            #
# --------------------------------------------------------------------------- #
def _pairs(intent, *attrs: str) -> List[Tuple[str, ...]]:
    """Deduplicated, deterministically ordered member tuples drawn from the
    named pair attributes of ``intent`` (each a list of (a, b) tuples).

    Dedup is by unordered membership; the canonical emitted tuple is the
    name-sorted pair so output is order-stable regardless of source order.
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


def _detect_diff(intent) -> List[Tuple[str, ...]]:
    """Differential-pair candidates from the declared diff pairs."""
    return _pairs(intent, "diff_pairs")


def _detect_diff_or_mirror(intent) -> List[Tuple[str, ...]]:
    """Output-stage / branch symmetry candidates: declared diff or mirror
    pairs (the topology whose balance the category cares about)."""
    return _pairs(intent, "diff_pairs", "mirror_pairs")


def _detect_matched_or_passive_pairs(intent) -> List[Tuple[str, ...]]:
    """Reference-device matching candidates for bandgap: declared matched pairs
    plus same-kind passive pairs (resistor/BJT ratios). Passive pairing is
    topological -- it groups consecutive same-kind passives from the netlist,
    never by name and never by coordinate."""
    out = _pairs(intent, "matched_pairs", "mirror_pairs")
    seen = {frozenset(t) for t in out}
    # Same-kind passive pairs (e.g. the PTAT/CTAT resistor pair, BJT pair):
    # group passives by their declared kind, pair them in declaration order.
    by_kind: dict = {}
    for p in (getattr(intent, "passives", None) or []):
        kind = str(getattr(p, "kind", None) or "") or None
        by_kind.setdefault(kind, []).append(str(p.name))
    for kind in sorted(k for k in by_kind if k is not None):
        names = by_kind[kind]
        for i in range(0, len(names) - 1, 2):
            members = tuple(sorted((names[i], names[i + 1])))
            key = frozenset(members)
            if len(key) < 2 or key in seen:
                continue
            seen.add(key)
            out.append(members)
    out.sort()
    return out


# --------------------------------------------------------------------------- #
# Shared emitters (zero-coordinate: only categorical hyperedges).              #
# --------------------------------------------------------------------------- #
def _emit_group(kind: str):
    def _emit(intent, group: Iterable[str], net) -> None:
        net.add_group(kind, tuple(group))
    return _emit


_emit_compensation_min = _emit_group("compensation_parasitic_min")
_emit_output_symmetry = _emit_group("output_stage_symmetry")
_emit_clk_shield = _emit_group("clk_sensitive_shield")
_emit_regen_symmetry = _emit_group("regen_pair_symmetry")
_emit_ref_centroid = _emit_group("reference_common_centroid")
_emit_ref_branch_symmetry = _emit_group("reference_branch_symmetry")


# --------------------------------------------------------------------------- #
# opamp pack.                                                                  #
# --------------------------------------------------------------------------- #
OPAMP = SpecPack(
    name="opamp",
    applies_to=["opamp"],
    rules=[
        SpecRule(
            name="compensation_node_parasitic_min",
            metric="bandwidth",
            effect="parasitic_load",
            object="diff_pair",
            hardness="qualitative_soft",
            detect=_detect_diff,
            emit=_emit_compensation_min,
            checker="compensation_parasitic_min_check",
        ),
        SpecRule(
            name="output_stage_symmetry",
            metric="cmrr",
            effect="branch_imbalance",
            object="diff_pair",
            hardness="quantified_soft",
            detect=_detect_diff_or_mirror,
            emit=_emit_output_symmetry,
            checker="output_symmetry_check",
        ),
    ],
)


# --------------------------------------------------------------------------- #
# comparator pack -- carries the kickback/clock-shield rule opamp lacks.       #
# --------------------------------------------------------------------------- #
COMPARATOR = SpecPack(
    name="comparator",
    applies_to=["comparator"],
    rules=[
        SpecRule(
            # Comparator-only: shield the clock/sensitive net so the dynamic
            # clock edge does not couple kickback into the input pair.
            name="clk_sensitive_shield",
            metric="noise",
            effect="coupling",
            object="sensitive_net",
            hardness="qualitative_soft",
            detect=_detect_diff,
            emit=_emit_clk_shield,
            checker="shield_check",
        ),
        SpecRule(
            name="regen_pair_symmetry",
            metric="offset",
            effect="branch_imbalance",
            object="diff_pair",
            hardness="quantified_soft",
            detect=_detect_diff_or_mirror,
            emit=_emit_regen_symmetry,
            checker="regen_symmetry_check",
        ),
    ],
)


# --------------------------------------------------------------------------- #
# bandgap pack.                                                                #
# --------------------------------------------------------------------------- #
BANDGAP = SpecPack(
    name="bandgap",
    applies_to=["bandgap"],
    rules=[
        SpecRule(
            name="reference_device_common_centroid",
            metric="offset",
            effect="process_gradient",
            object="matched_pair",
            hardness="quantified_soft",
            detect=_detect_matched_or_passive_pairs,
            emit=_emit_ref_centroid,
            checker="centroid_coincidence",
        ),
        SpecRule(
            name="reference_branch_symmetry",
            metric="psrr",
            effect="branch_imbalance",
            object="mirror_pair",
            hardness="quantified_soft",
            detect=_detect_diff_or_mirror,
            emit=_emit_ref_branch_symmetry,
            checker="reference_branch_symmetry_check",
        ),
    ],
)


# Register on import (the spec tests rely on import triggering registration).
register_pack(OPAMP)
register_pack(COMPARATOR)
register_pack(BANDGAP)


__all__ = ["OPAMP", "COMPARATOR", "BANDGAP"]
