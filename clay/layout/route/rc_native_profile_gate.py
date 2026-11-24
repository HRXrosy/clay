"""Isolated fixed-stack native output profile gate (quarantined for Gate G3 retirement)."""
from __future__ import annotations

from collections import Counter
from typing import Any


_NATIVE_440_OUTPUT_PROFILES = (
    "sky130_lvt_single_440_output_m12_v1",
    "sky130_lvt_single_440_output_input_m12_v1",
)


def _check_source_engine_before_group_fallback(engine: Any, source_record: Any) -> None:
    """Check the selected source engine before or after temporary method installation."""
    if source_record is not None:
        from clay.layout.route.source_engine_record import require_source_engine

        require_source_engine(source_record, engine)


def _maybe_freeze_native_output_profile_engine(
    placement: Any,
    intent: Any,
    route_engine: Any,
) -> Any:
    """Freeze source engine identity when legacy 440 output profiles or row candidates are present."""
    initial_row_materialization = getattr(placement, "row_materialization", None)
    initial_projection = getattr(initial_row_materialization, "source_projection", None)
    initial_native_witness = getattr(initial_projection, "native_witness", None)
    if (
        any(
            getattr(row, "generation_profile", "") in _NATIVE_440_OUTPUT_PROFILES
            for row in getattr(intent, "arrays", ())
        )
        or getattr(initial_native_witness, "row_candidates", ())
    ):
        from clay.layout.route.source_engine_record import freeze_source_engine

        return freeze_source_engine(route_engine)
    return None


def _verify_native_output_profile_fixed_stacks(
    *,
    cell_def: Any,
    tech: Any,
    route_engine: Any,
    output_profile_engine_record: Any,
    geometry: Any,
    placement_um: Any,
    final_row_placement: Any,
    intent: Any,
    layout_model: Any,
    obstacles: Any,
    pins_by_net: Any,
) -> None:
    """Verify legacy 440 output-profile fixed-stack obligations when declared."""
    output_profile_groups = {
        row.group: (
            9
            if row.generation_profile == "sky130_lvt_single_440_output_input_m12_v1"
            else 4
        )
        for row in getattr(intent, "arrays", ())
        if getattr(row, "generation_profile", "") in _NATIVE_440_OUTPUT_PROFILES
    }
    final_materialization = getattr(final_row_placement, "row_materialization", None)
    final_projection = getattr(final_materialization, "source_projection", None)
    final_witness = getattr(final_projection, "native_witness", None)
    if getattr(final_witness, "row_candidates", ()):
        from clay.layout.intent import layout_intent_from_cell_def
        from clay.layout.native_diffusion_source import (
            _require_native_pcell_witness_operands,
            declared_native_chains,
        )
        from clay.layout.native_row_candidate import (
            BUFFER_COMPACT_SOURCE,
            PRECHARGE_SPLIT_GATE,
            witness_implementation,
        )

        _require_native_pcell_witness_operands(
            final_witness,
            cell=cell_def,
            tech=tech,
            overlay=final_witness.overlay,
            source_cir=final_projection.source_cir,
        )
        chains = dict(
            declared_native_chains(cell_def, layout_intent_from_cell_def(cell_def))
        )
        for choice in final_witness.row_candidates:
            if choice.group in output_profile_groups or choice.group not in chains:
                raise ValueError("candidate output source group differs")
            profile, _ = witness_implementation(
                cell_def, chains[choice.group], final_witness, choice.group
            )
            if profile in (BUFFER_COMPACT_SOURCE, PRECHARGE_SPLIT_GATE):
                continue
            if profile != "sky130_lvt_single_440_output_input_m12_v1":
                raise ValueError("candidate output source profile differs")
            output_profile_groups[choice.group] = 9
    if output_profile_groups:
        from clay.layout.route.native_output_world_proof import (
            verify_source_fixed_stack_world,
        )
        from clay.layout.route.source_engine_record import require_source_engine

        require_source_engine(output_profile_engine_record, route_engine)
        stack_proofs = verify_source_fixed_stack_world(
            cell=cell_def,
            tech=tech,
            engine=route_engine,
            geometry=geometry,
            placement_um=placement_um,
            row_program=final_row_placement.row_program,
            row_materialization=final_row_placement.row_materialization,
            row_final_certificate=final_row_placement.row_final_certificate,
            intent=intent,
            layout_model=layout_model,
            obstacles=obstacles,
            pins_by_net=pins_by_net,
        )
        if Counter(proof.stamp_id for proof in stack_proofs) != output_profile_groups:
            raise ValueError("source output world fixed-stack obligations differ")
