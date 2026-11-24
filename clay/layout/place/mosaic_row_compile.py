"""Compile exact mosaic occurrence covers into immutable row calculus."""
from __future__ import annotations

from typing import Mapping, Protocol

from .row_calculus import (
    BoundaryTerminal, DiffusionBoundary, RowAtom, RowAtomKind,
    RowMacro, RowString, RowStructureKind, compile_row_macro,
)
from .row_program_primitives import MAX_DIMENSION_NM, RowProgramError


class MosaicRowMember(Protocol):
    """Read-only source properties needed to compile a mosaic row."""

    kind: str
    oxide: str
    body_net: str
    drain_net: str
    source_net: str
    width_nm: int
    length_nm: int


def _mosaic_gap_nm(tech: object) -> int:
    """Resolve one installed-PDK separation for non-sharing mosaic neighbours."""

    from clay.layout.device_cell import _active_name, _rules_from_tech, _space_or

    resolver = getattr(tech, "min_space", None)
    if not callable(resolver):
        raise RowProgramError(
            "discrete mosaic requires an installed-PDK min_space resolver"
        )
    layer = _active_name(tech)
    rules = _rules_from_tech(tech)
    value = _space_or(tech, layer, rules.channel_min_width)
    if type(value) is int and 0 < value <= MAX_DIMENSION_NM:
        return value
    raise RowProgramError(
        f"discrete mosaic cannot resolve a positive {layer!r} active-layer spacing"
    )


def _compile_mosaic_macro(
    declaration: object,
    *,
    by_name: Mapping[str, MosaicRowMember],
    tech: object,
    guest_slot: object = None,
    boundary_intent: object = None,
) -> RowMacro:
    """Compile a projected row-major exact cover without inferring diffusion joins."""

    from clay.layout.spatial.intent_projection import replay_discrete_mosaic_intent

    mosaic = replay_discrete_mosaic_intent(declaration)
    from clay.layout.spatial.mosaic_boundary_joins import (
        compile_mosaic_row_boundaries, replay_mosaic_boundary_join_intent,
    )
    if boundary_intent is not None:
        replay_mosaic_boundary_join_intent(boundary_intent, mosaic=mosaic, members=by_name)
    gap_nm = _mosaic_gap_nm(tech)
    refs = tuple(sorted((
        f"discrete_mosaic:{mosaic.mosaic_id}",
        mosaic.sha256,
        f"asir:{mosaic.source_asir_sha256}",
        *(f"fact:{fact_id}" for fact_id in mosaic.source_fact_ids),
    )))
    if mosaic.column_ranks:
        refs = tuple(sorted((*refs, f"mosaic_channel_lattice:{mosaic.sha256}")))
    if guest_slot is not None:
        from clay.layout.spatial.mosaic_guest_slot import MosaicGuestSlot
        if type(guest_slot) is not MosaicGuestSlot or guest_slot.source_mosaic_sha256 != mosaic.sha256:
            raise RowProgramError("mosaic guest slot source identity differs")
        refs = tuple(sorted((*refs, f"mosaic_guest_slot:{guest_slot.sha256}")))
    if boundary_intent is not None:
        refs = tuple(sorted((*refs, f"mosaic_boundary_joins:{boundary_intent.sha256}")))
    rows = []
    for row_index, projected_row in enumerate(mosaic.rows):
        atoms = []
        for sequence_index, occurrence in enumerate(projected_row):
            member = by_name.get(occurrence.logical_owner)
            if member is None:
                raise RowProgramError(
                    f"discrete mosaic names unknown MOS owner {occurrence.logical_owner!r}"
                )
            width_nm = max(2, member.length_nm)
            boundary_span_nm = min(member.width_nm, width_nm - 1)
            ref = f"discrete_mosaic:{mosaic.mosaic_id}"
            atoms.append(RowAtom(
                atom_id=(
                    f"{mosaic.mosaic_id}:r{row_index}:active:"
                    f"{occurrence.logical_owner}:{occurrence.node_id}"
                ),
                sequence_index=sequence_index,
                width_nm=width_nm,
                height_nm=member.width_nm,
                left=DiffusionBoundary(
                    net=member.source_net,
                    terminal=BoundaryTerminal.SOURCE,
                    share_span_nm=boundary_span_nm,
                    declaration_ref=ref,
                ),
                right=DiffusionBoundary(
                    net=member.drain_net,
                    terminal=BoundaryTerminal.DRAIN,
                    share_span_nm=boundary_span_nm,
                    declaration_ref=ref,
                ),
                match_key=(
                    f"mosaic:{occurrence.node_id}:{occurrence.transform}"
                ),
                kind=RowAtomKind.ACTIVE,
                declaration_refs=refs,
            ))
        rows.append(RowString(
            row_id=f"{mosaic.mosaic_id}:row:{row_index}",
            row_index=row_index,
            atoms=tuple(atoms),
            declaration_refs=refs,
            inter_atom_gap_nm=gap_nm if boundary_intent is None else None,
            boundary_joins=None if boundary_intent is None else compile_mosaic_row_boundaries(
                boundary_intent, row=projected_row, atoms=atoms, gap_nm=gap_nm),
        ))
    macro = compile_row_macro(
        macro_id=f"row:{mosaic.mosaic_id}",
        structure_kind=RowStructureKind.DISCRETE_MOSAIC,
        rows=tuple(rows),
        row_pitch_nm=max(by_name[name].width_nm for name in mosaic.members) + gap_nm,
        declaration_refs=refs,
    )
    assert macro is not None
    return macro


