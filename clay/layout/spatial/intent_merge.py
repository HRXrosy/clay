"""Conflict-preserving unions for independently projected spatial declarations."""
from typing import Dict, Tuple
from .intent_projection import (
    AggregateAlignmentIntent, ReferenceConditionedPlacementObligation,
    replay_aggregate_alignment_inventory, replay_reference_conditioned_placement_obligation_inventory,
)

def _merge_aggregate_alignment_inventories(
    first: Tuple[AggregateAlignmentIntent, ...],
    second: Tuple[AggregateAlignmentIntent, ...],
) -> Tuple[AggregateAlignmentIntent, ...]:
    """Union projected alignments without letting one semantic id overwrite another."""

    by_id: Dict[str, AggregateAlignmentIntent] = {}
    for item in (
        *replay_aggregate_alignment_inventory(first),
        *replay_aggregate_alignment_inventory(second),
    ):
        prior = by_id.get(item.alignment_id)
        if prior is not None and prior != item:
            raise ValueError(
                f"conflicting aggregate alignment id {item.alignment_id!r}"
            )
        by_id[item.alignment_id] = item
    return replay_aggregate_alignment_inventory(tuple(by_id.values()))


def _merge_reference_conditioned_placement_obligations(
    first: Tuple[ReferenceConditionedPlacementObligation, ...],
    second: Tuple[ReferenceConditionedPlacementObligation, ...],
) -> Tuple[ReferenceConditionedPlacementObligation, ...]:
    """Union typed reference obligations without silent evidence replacement."""

    by_id: Dict[str, ReferenceConditionedPlacementObligation] = {}
    for item in (
        *replay_reference_conditioned_placement_obligation_inventory(first),
        *replay_reference_conditioned_placement_obligation_inventory(second),
    ):
        prior = by_id.get(item.obligation_id)
        if prior is not None and prior != item:
            raise ValueError(
                "conflicting reference-conditioned placement obligation id "
                f"{item.obligation_id!r}"
            )
        by_id[item.obligation_id] = item
    return replay_reference_conditioned_placement_obligation_inventory(
        tuple(by_id.values())
    )


