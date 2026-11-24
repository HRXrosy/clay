"""Fail-closed certificate that selected macro partitions reached placement.

The certificate is intentionally independent of any placer implementation.  It
checks the existing ``PlaceGroup``/guard-ring intent against the graph-bound
candidate and then proves that every selected member exists in the final route
placement and geometry maps.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

from clay.flow.optimized_layout_style import macro_partition_place_group_name
from clay.layout.compact_intent import CompactIntentCandidate, MacroPartitionChoice
from clay.layout.physical_macro_graph import PhysicalMacroGraph


ConsumptionStatus = Literal["pass", "fail", "unmeasurable"]


@dataclass(frozen=True)
class MacroConsumptionCertificate:
    graph_sha256: str
    candidate_sha256: str
    consumed_macro_ids: tuple[str, ...]
    device_to_macro_blocks: tuple[tuple[str, tuple[str, ...]], ...]
    status: ConsumptionStatus
    reason: str

    def canonical_data(self) -> dict[str, object]:
        return {
            "schema": "clay.macro-consumption/v1",
            "graph_sha256": self.graph_sha256,
            "candidate_sha256": self.candidate_sha256,
            "consumed_macro_ids": list(self.consumed_macro_ids),
            "device_to_macro_blocks": [
                [device, list(blocks)] for device, blocks in self.device_to_macro_blocks
            ],
            "status": self.status,
            "reason": self.reason,
        }

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.canonical_data(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def _certificate(
    graph: PhysicalMacroGraph,
    candidate: CompactIntentCandidate,
    *,
    choices: Sequence[MacroPartitionChoice],
    member_blocks: Mapping[str, set[str]],
    status: ConsumptionStatus,
    reason: str,
) -> MacroConsumptionCertificate:
    return MacroConsumptionCertificate(
        graph_sha256=graph.sha256,
        candidate_sha256=candidate.digest,
        consumed_macro_ids=tuple(sorted(choice.macro_id for choice in choices)),
        device_to_macro_blocks=tuple(
            (device, tuple(sorted(blocks)))
            for device, blocks in sorted(member_blocks.items())
        ),
        status=status,
        reason=reason,
    )


def certify_macro_consumption(
    graph: PhysicalMacroGraph,
    candidate: CompactIntentCandidate,
    intent: object,
    placement: Mapping[str, object],
    geometry: Mapping[str, object],
) -> MacroConsumptionCertificate:
    """Prove selected macro DATA was consumed by the final shared placement."""

    choices = tuple(
        sorted(
            (*candidate.passive_bank_partitions, *candidate.well_island_partitions),
            key=lambda choice: choice.macro_id,
        )
    )
    member_blocks: dict[str, set[str]] = {}

    def finish(status: ConsumptionStatus, reason: str) -> MacroConsumptionCertificate:
        return _certificate(
            graph,
            candidate,
            choices=choices,
            member_blocks=member_blocks,
            status=status,
            reason=reason,
        )

    graph_binding_selected = bool(choices) or (
        candidate.physical_macro_graph_sha256 is not None
    )
    if graph_binding_selected and candidate.physical_macro_graph_sha256 != graph.sha256:
        return finish(
            "fail",
            "STALE_PHYSICAL_MACRO_GRAPH: candidate and effective graph differ",
        )

    macros = {macro.macro_id: macro for macro in graph.macros}
    selected_passive_groups = tuple(
        sorted(
            members
            for choice in candidate.passive_bank_partitions
            for members in choice.member_groups
        )
    )
    effective_passive_groups = tuple(
        sorted(
            macro.members
            for macro in graph.macros
            if macro.kind == "passive_bank"
        )
    )
    if candidate.passive_bank_partitions and (
        selected_passive_groups != effective_passive_groups
    ):
        return finish(
            "fail",
            "PARTITION_MEMBER_LOSS: selected passive groups differ from "
            "the effective physical macro graph",
        )
    place_groups = tuple(getattr(intent, "place_groups", ()) or ())
    by_group: dict[str, object] = {}
    for group in place_groups:
        name = str(getattr(group, "group", ""))
        if name in by_group:
            return finish(
                "fail", f"UNCONSUMED_MACRO_PARTITION: duplicate group {name!r}"
            )
        by_group[name] = group

    for choice in choices:
        if choice in candidate.well_island_partitions:
            macro = macros.get(choice.macro_id)
            if macro is None:
                return finish(
                    "fail",
                    f"PARTITION_MEMBER_LOSS: unknown macro {choice.macro_id!r}",
                )
            flattened = tuple(
                member for members in choice.member_groups for member in members
            )
            if set(flattened) != set(macro.members) or len(flattened) != len(
                macro.members
            ):
                return finish(
                    "fail",
                    f"PARTITION_MEMBER_LOSS: {choice.macro_id!r} membership differs",
                )
        for index, expected_members in enumerate(choice.member_groups):
            block = macro_partition_place_group_name(choice, index)
            declared = by_group.get(block)
            actual_members = tuple(sorted(getattr(declared, "devices", ()) or ()))
            if declared is None or actual_members != expected_members:
                return finish(
                    "fail",
                    "UNCONSUMED_MACRO_PARTITION: "
                    f"{block!r} expected {expected_members!r}, got "
                    f"{actual_members!r}",
                )
            for member in expected_members:
                member_blocks.setdefault(member, set()).add(block)

    missing = tuple(
        sorted(
            member
            for member in member_blocks
            if member not in placement or member not in geometry
        )
    )
    if missing:
        return finish(
            "unmeasurable",
            f"MISSING_PLACED_MEMBER: final route geometry lacks {missing!r}",
        )

    for macro in graph.macros:
        if macro.kind not in {"matched_array", "matching_relation"}:
            continue
        blocks = {
            block for member in macro.members for block in member_blocks.get(member, ())
        }
        if len(blocks) > 1:
            return finish(
                "fail",
                "MATCHING_COMPONENT_SPLIT: "
                f"{macro.macro_id!r} spans {tuple(sorted(blocks))!r}",
            )

    rings = tuple(getattr(intent, "guard_rings", ()) or ())
    for choice in candidate.well_island_partitions:
        macro = macros[choice.macro_id]
        attributes = dict(macro.attributes)
        expected_well = attributes.get("well", "")
        expected_body = attributes.get("body", "")
        for members in choice.member_groups:
            matching = tuple(
                ring
                for ring in rings
                if tuple(sorted(getattr(ring, "members", ()) or ())) == members
            )
            if len(matching) != 1 or any(
                str(getattr(ring, "well", "")) != expected_well
                or str(getattr(ring, "tie_net", "")) != expected_body
                for ring in matching
            ):
                return finish(
                    "fail",
                    "BODY_WELL_TIE_MISMATCH: "
                    f"{choice.macro_id!r} group {members!r} requires "
                    f"well={expected_well!r}, body={expected_body!r}",
                )

    return finish("pass", "all selected macro partitions were consumed exactly once")


__all__ = [
    "ConsumptionStatus",
    "MacroConsumptionCertificate",
    "certify_macro_consumption",
]
