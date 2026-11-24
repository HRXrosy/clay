"""Compile, solve, materialize, and certify one complete rigid placement."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
from decimal import Decimal
import os
from typing import Any, Dict, List, Mapping, Optional, Tuple
from clay.layout.compact_blocks import _NM_PER_UM, apply_block_deltas, extract_blocks
from clay.layout.intent_basis import IntentBasis, IntentConsumptionCertificate
from clay.layout.intent_closure import ConsumerProjection
from clay.layout.intent_facts import IntentFactSet, SourceInventory
from clay.layout.place.solver_execution_policy import incumbent_replay_enabled, scoped_incumbent_replay
from clay.layout.place.materialized_rail_policy import (
    materialized_rail_kwargs, materialized_replay_seed_positions,
)
from clay.layout.spatial.intent_projection import replay_reference_conditioned_placement_obligation_inventory
from clay.layout.place.constraints_hierarchical import (
    _layered_component_refinement_owners,
)
from clay.layout.place.constraints_spatial import (
    _assert_no_physical_block_overlap,
    _canonical_flexible_envelope_constraints,
    _canonical_rigid_compound_footprints,
    _landscape_place_enabled,
    _placement_declaration_refs,
    _placement_objective_from_intent,
    _strict_band_order_required,
    _vertical_channel_nm,
    assign_bands,
    common_centroid_devices,
    floorplan_column_groups,
    identify_left_column,
)
from clay.layout.place.constraints_well_guard import (
    _adjacency_compatible_bottom_exemptions,
    _bottom_edge_keepout_clear_blocks,
    _compiled_edge_rail_keepouts_nm,
    _compiled_edge_rail_specs_for_placement,
    _effective_block_gap_nm,
    _exact_member_ring_padding_rows,
    _hard_placement_compatible_bottom_exemptions,
    _layer_disjoint_overlap_compatible_pairs,
    _materialize_same_ring_passive_adjacency,
    _merge_pair_clearances_nm,
    _project_pair_clearances_through_composites,
    _mim_pair_clearances_nm,
    _mim_native_access_pair_clearances_nm,
    _project_native_access_pair_clearances_to_masters,
    _pdk_layer_clearances_nm,
    _project_pair_clearances_to_masters,
    _replay_ring_padding_on_positioned_master,
    _resolve_block_member_devices,
    _resolved_adjacency_limits_nm,
    _resolved_device_master_rows,
    _ring_pair_clearances_nm,
    _same_bulk_mergeable_pair_layers,
    _same_bulk_nwell_pair_clearances_nm,
    _top_edge_keepout_clear_blocks,
    _well_share_ring_cohesion_groups,
    bundle_passive_composites,
    replay_member_ring_padding,
)
from clay.layout.place.cp_model_builder import (
    _DeterministicBudgetLedger,
    _place_blocks_2d_with_alignment_scale_source,
    _replace_flexible_envelope_member_bboxes,
    place_blocks_2d_with_certificate,
)

_DEFAULT_REPLAY_PLACER = _place_blocks_2d_with_alignment_scale_source
from clay.layout.place.placement_certificates import (
    FinalPlacementCertificate,
    PlacementSolveCutResidualError,
    RepositionPum2DResult,
    _assert_materialized_diff_axis_certificate,
    _bind_certificate_to_materialized_blocks,
    placement_certificate_is_signable,
)
from clay.layout.place.placement_constants import (
    _BAND_CENTER,
    _BAND_TOP,
    _Bbox,
)
from clay.layout.place.placement_cuts import (
    _exact_cut_input_blocks,
)
from clay.layout.place.layered_placement_boundary import materialized_source_snapshot
from clay.layout.place.placement_geometry import (
    _bundle_declared_edge_alignment_composites,
    _bundle_mos_matching_composites,
    _bundle_rigid_nonmos_composites,
    _canonical_materialized_block_source_members,
    _canonical_mos_matching_axis_rows,
    _diff_pair_axis_alignment_groups,
    _diff_axis_positions_in_solver_frame,
    _discrete_mosaic_local_order_proofs,
    _edge_alignment_bbox_device_inventory,
    _feedback_final_master_inventory,
    _fixed_pcell_compound_footprints_from_geometry,
    _layered_collision_footprints_from_geometry,
    _matching_axis_cumulative_authority,
    _matching_axis_master_authority,
    _materialized_adjacency_scope_bboxes_nm,
    _materialized_source_ownership_requested,
    _project_mos_matching_axes_through_block_deltas,
    _project_source_device_bboxes_to_solver_masters,
    _resistor_adjacency_terms,
    _unbound_adjacency_band_overrides,
    _rigid_compound_footprints_from_members,
    _source_bbox_refinement_rows,
    _source_device_bboxes_nm,
    _source_device_occurrence_envelopes_nm,
)
from clay.layout.place.placement_relations import (
    _aggregate_envelope_orders_from_intent,
    _hierarchical_macro_cohesion_groups_from_intent,
    _hierarchical_macro_constraints_from_intent,
    _horizontal_order_is_materializable,
    _left_column_after_explicit_edge_anchor,
    _mosaic_center_alignment_pairs,
    _nonrigid_place_group_blocks,
    _reference_conditioned_row_bounded_cavities_from_intent,
    _relational_place_group_envelope_edges,
    _relational_place_group_master_constraints,
)


def _admitted_group_spacing_pairs(ir, pair_clearances, native_access_clearances):
    from clay.layout.place.native_feol_reservation import sky130_group_spacing_master_pairs
    return tuple(pair for pair in sky130_group_spacing_master_pairs(ir)
                 if pair not in pair_clearances and pair not in native_access_clearances)


def _source_cross_row_bottom_alignments(alignments, inventory):
    """Select source bottom alignments on the same two matching-owned rows."""
    if inventory is None:
        return ()
    row_for_member = {}
    for row, members in inventory.row_source_members:
        for member in members:
            if member in row_for_member:
                raise ValueError('cross-row alignment source member has multiple row owners')
            row_for_member[member] = row
    matching_rows = {frozenset(endpoint.row_owner for endpoint in pair.endpoints)
                     for pair in inventory.pairs}
    return tuple(declaration for declaration in tuple(alignments or ())
        if declaration.edge == 'bottom' and declaration.offset_nm == 0
        and row_for_member.get(declaration.first) is not None
        and row_for_member.get(declaration.second) is not None
        and row_for_member[declaration.first] != row_for_member[declaration.second]
        and frozenset((row_for_member[declaration.first], row_for_member[declaration.second])) in matching_rows)


def _has_one_row_occurrence(geometry, program, row, member):
    entries = [entry for entry in program.entries if entry.entry_id == row]
    if len(entries) != 1 or member not in entries[0].consumed_devices:
        return False
    stamp = geometry.get(entries[0].group)
    rows = getattr(stamp, 'physical_occurrences_by_member', ())
    if type(rows) is not tuple:
        return False
    matches = [occurrences for owner, occurrences in rows if owner == member]
    return len(matches) == 1 and type(matches[0]) in (tuple, list) and len(matches[0]) == 1


def _require_cross_row_source_alignment_authority(alignments, inventory, projection, program):
    from clay.layout.place.row_source_projection import require_row_source_projection
    from clay.layout.intent import layout_intent_from_cell_def
    require_row_source_projection(projection, program=program)
    authoring = layout_intent_from_cell_def(projection.authoring_source_cell)
    canonical = layout_intent_from_cell_def(projection.native_witness.cell)
    def relevant(rows):
        return frozenset(_source_cross_row_bottom_alignments(rows, inventory))
    if (relevant(alignments) != relevant(authoring.edge_alignments)
            or relevant(authoring.edge_alignments) != relevant(canonical.edge_alignments)):
        raise ValueError('cross-row alignment consumer and source authorities differ')


def _partition_cross_row_bottom_alignments(alignments, inventory, program,
        source_occurrences, source_centres, source_axes, live_master_devices,
        master_members, block_stamps, source_device_names, source_geometry):
    """Keep proven matching rows independent while retaining their source edge.

    Only a singleton source occurrence on each of two matching-owned rows can
    use the existing exact same-phase cross-row center constraint.  All other
    source alignments keep their ordinary rigid-bundling path.
    """
    declarations = tuple(alignments or ())
    if inventory is None or not declarations:
        return declarations, ()
    row_for_member = {}
    for row, members in inventory.row_source_members:
        for member in members:
            if member in row_for_member:
                raise ValueError('cross-row alignment source member has multiple row owners')
            row_for_member[member] = row
    candidates = []
    for declaration in _source_cross_row_bottom_alignments(declarations, inventory):
        first, second = declaration.first, declaration.second
        a, b = source_occurrences.get(first), source_occurrences.get(second)
        first_row, second_row = row_for_member.get(first), row_for_member.get(second)
        if (a is not None and b is not None
                and _has_one_row_occurrence(source_geometry, program, first_row, first)
                and _has_one_row_occurrence(source_geometry, program, second_row, second)
                and a[2]-a[0] == a[3]-a[1] == 1
                and b[2]-b[0] == b[3]-b[1] == 1 and a[1] == b[1]):
            candidates.append(declaration)
    if not candidates:
        return declarations, ()
    from clay.layout.place.cross_row_matching import (
        native_cross_row_center_alignments, native_cross_row_solver_owners,
    )
    owners = native_cross_row_solver_owners(inventory, program, live_master_devices,
        master_members, block_stamps, source_device_names)
    supported = set(native_cross_row_center_alignments(inventory, source_centres, source_axes, owners))
    deferred = tuple(declaration for declaration in candidates
        if (*sorted((owners[row_for_member[declaration.first]], owners[row_for_member[declaration.second]])),
            'vertical') in supported)
    return tuple(declaration for declaration in declarations if declaration not in deferred), deferred


def _replay_cross_row_bottom_alignments(declarations, geometry, placement_um):
    if not declarations:
        return
    from clay.layout.place.placement_geometry import _source_device_occurrence_envelopes_nm
    names = tuple(sorted({member for row in declarations for member in (row.first, row.second)}))
    occurrences = _source_device_occurrence_envelopes_nm(geometry, placement_um, names)
    if set(occurrences) != set(names):
        raise ValueError('cross-row alignment final occurrence inventory differs')
    for name in names:
        matches = [rows for stamp in geometry.values()
                   for owner, rows in (getattr(stamp, 'physical_occurrences_by_member', ()) or ())
                   if owner == name]
        if len(matches) != 1 or type(matches[0]) not in (tuple, list) or len(matches[0]) != 1:
            raise ValueError('cross-row alignment final source occurrence count differs')
    for row in declarations:
        first, second = occurrences[row.first], occurrences[row.second]
        if (first[2]-first[0] != 1 or first[3]-first[1] != 1
                or second[2]-second[0] != 1 or second[3]-second[1] != 1
                or second[1]-first[1] != row.offset_nm):
            raise ValueError('cross-row source bottom alignment differs after materialization')


def _replay_cross_row_source_bottom_alignments(alignments, inventory, geometry, placement_um):
    """Replay current source obligations using the actual downstream artwork."""
    declarations = _source_cross_row_bottom_alignments(alignments, inventory)
    if not declarations:
        return
    from clay.layout.place.placement_geometry import _source_device_occurrence_envelopes_nm
    names = tuple(sorted({member for row in declarations for member in (row.first, row.second)}))
    occurrences = _source_device_occurrence_envelopes_nm(geometry, placement_um, names)
    if set(occurrences) != set(names):
        raise ValueError('cross-row source alignment occurrence inventory differs')
    for row in declarations:
        if occurrences[row.second][1] - occurrences[row.first][1] != row.offset_nm:
            raise ValueError('cross-row source bottom alignment differs in final route placement')


def _native_row_left_offset_alignments(intent, program):
    """Source non-abutting left offsets between independently owned rows."""
    if intent is None or program is None:
        return ()
    owners = {}
    for entry in program.entries:
        for member in entry.consumed_devices:
            if member in owners:
                raise ValueError('source row alignment member has multiple owners')
            owners[member] = entry.entry_id
    abuts = {frozenset(map(str, pair)) for pair in (getattr(intent, 'abut_pairs', ()) or ())}
    return tuple(row for row in (getattr(intent, 'edge_alignments', ()) or ())
        if row.edge == 'left' and row.offset_nm != 0
        and row.first in owners and row.second in owners
        and owners[row.first] != owners[row.second]
        and frozenset((row.first, row.second)) not in abuts)


def _require_native_left_alignment_authority(intent, projection, program):
    from clay.layout.place.row_source_projection import require_row_source_projection
    from clay.layout.intent import layout_intent_from_cell_def
    require_row_source_projection(projection, program=program)
    authoring = layout_intent_from_cell_def(projection.authoring_source_cell)
    canonical = layout_intent_from_cell_def(projection.native_witness.cell)
    if (frozenset(_native_row_left_offset_alignments(intent, program))
            != frozenset(_native_row_left_offset_alignments(authoring, program))
            or frozenset(_native_row_left_offset_alignments(authoring, program))
            != frozenset(_native_row_left_offset_alignments(canonical, program))):
        raise ValueError('native offset alignment source authorities differ')
    return authoring


def _witness_singleton_native_left_alignments(intent, program, witness):
    by_group = {name:geometry for name,geometry,_folder in witness.rows}
    entries = {member:entry for entry in program.entries for member in entry.consumed_devices}
    def one(member):
        entry=entries.get(member)
        geometry=None if entry is None else by_group.get(entry.group)
        binding=getattr(geometry, 'joint_row_binding', None)
        return binding is not None and sum(item.owner == member for item in binding.occurrences) == 1
    return tuple(row for row in _native_row_left_offset_alignments(intent, program)
                 if one(row.first) and one(row.second))


def _partition_native_left_alignments(alignments, intent, program, geometry,
        occurrences, blocks, master_members, block_stamps, source_device_names, grid):
    """Defer only source rows with complete independent pre-bundle masters."""
    declarations = tuple(alignments)
    if program is None:
        return declarations, ()
    source = set(source_device_names)
    boxes = dict(blocks)
    rows = {member: entry for entry in program.entries for member in entry.consumed_devices}
    owner = {}
    for master in boxes:
        for member in set(master_members[master]) & source:
            if member in owner:
                raise ValueError('native offset alignment member has duplicate masters')
            owner[member] = master
    candidates = set(_native_row_left_offset_alignments(intent, program))
    deferred = []
    for declaration in declarations:
        if declaration not in candidates:
            continue
        a, b = declaration.first, declaration.second
        ma, mb = owner.get(a), owner.get(b)
        if ma is None or mb is None or ma == mb:
            continue
        if (set(master_members[ma]) & source != set(rows[a].consumed_devices)
                or set(master_members[mb]) & source != set(rows[b].consumed_devices)
                or tuple(block_stamps.get(ma, ())) != (rows[a].group,)
                or tuple(block_stamps.get(mb, ())) != (rows[b].group,)
                or not _has_one_row_occurrence(geometry, program, rows[a].entry_id, a)
                or not _has_one_row_occurrence(geometry, program, rows[b].entry_id, b)):
            continue
        oa, ob = occurrences.get(a), occurrences.get(b)
        if (oa is None or ob is None or oa[2]-oa[0] != 1 or oa[3]-oa[1] != 1
                or ob[2]-ob[0] != 1 or ob[3]-ob[1] != 1
                or type(grid) is not int or grid <= 0 or declaration.offset_nm % grid
                or oa[0] % grid or ob[0] % grid):
            continue
        axis_a = oa[0] - boxes[ma][0] + declaration.offset_nm
        axis_b = ob[0] - boxes[mb][0]
        if not (0 <= axis_a <= boxes[ma][2]-boxes[ma][0]
                and 0 <= axis_b <= boxes[mb][2]-boxes[mb][0]):
            continue
        deferred.append(declaration)
    return tuple(row for row in declarations if row not in deferred), tuple(deferred)


def _native_left_alignment_axis_groups(declarations, program, blocks, block_members, geometry, placement_um):
    if not declarations:
        return ()
    from clay.layout.place.placement_geometry import _source_device_occurrence_envelopes_nm
    from clay.layout.place.constraints_spatial import _canonical_axis_alignment_groups
    names = tuple(sorted({member for row in declarations for member in (row.first,row.second)}))
    occurrences = _source_device_occurrence_envelopes_nm(geometry, placement_um, names)
    boxes = dict(blocks)
    row_for = {member:entry for entry in program.entries for member in entry.consumed_devices}
    owners = {}
    for master in boxes:
        for member in set(block_members[master]) & set(names):
            if member in owners:
                raise ValueError('native offset alignment member has duplicate live owners')
            owners[member] = master
    if set(owners) != set(names) or set(occurrences) != set(names):
        raise ValueError('native offset alignment occurrence or owner is missing')
    for member in names:
        if not _has_one_row_occurrence(geometry, program, row_for[member].entry_id, member):
            raise ValueError('native offset alignment occurrence count differs')
    groups=[]
    for row in declarations:
        a,b=owners[row.first],owners[row.second]
        pa,pb=occurrences[row.first],occurrences[row.second]
        if any((box[2]-box[0],box[3]-box[1]) != (1,1) for box in (pa,pb)):
            raise ValueError('native offset alignment occurrence frame differs')
        first=2*(pa[0]-boxes[a][0]+row.offset_nm)
        second=2*(pb[0]-boxes[b][0])
        if a==b:
            if first!=second:
                raise ValueError('one native offset alignment master violates source')
        else:
            groups.append(((a,first),(b,second)))
    dims={name:(box[2]-box[0],box[3]-box[1]) for name,box in blocks}
    return _canonical_axis_alignment_groups(tuple(groups),block_ids=tuple(boxes),dims=dims)


def _replay_native_left_alignments(declarations, program, geometry, placement_um):
    if not declarations:
        return
    from clay.layout.place.placement_geometry import _source_device_occurrence_envelopes_nm
    names=tuple(sorted({member for row in declarations for member in (row.first,row.second)}))
    rows={member:entry for entry in program.entries for member in entry.consumed_devices}
    observed=_source_device_occurrence_envelopes_nm(geometry,placement_um,names)
    for member in names:
        if member not in rows or not _has_one_row_occurrence(geometry,program,rows[member].entry_id,member):
            raise ValueError('native offset alignment final occurrence count differs')
    for row in declarations:
        if observed[row.second][0]-observed[row.first][0] != row.offset_nm:
            raise ValueError('native source left offset differs after materialization')


def reposition_pum_2d(
    geom: Dict[str, Any],
    pum: Dict[str, Tuple[float, float]],
    intent: Any,
    tech: Any,
    tech_id: str,
    gap_um: float,
    bi: Any = None,
    aspect_target: float = 1.67,
    aspect_bounds: Optional[Tuple[float, float]] = None,
    aspect_obligation: bool = True,
    outline_width_um: Optional[float] = None,
    max_time_s: float = 30.0,
    adjacency: object = (),
    pack_mode: str = "rows",
    left_column_group: str = "",
    channel_strategy: str = "declared",
    solve_cuts: Any = (),
    solve_cut_proofs: Any = (),
    solve_cut_source_certificate: Any = None,
    source_member_ring_pad_top_nm: Any = (),
    source_member_ring_pad_bottom_nm: Any = (),
    rigid_nonmos_macro_plans: Any = None,
    rigid_nonmos_placed_devices: Any = None,
    matching_pair_axes: Any = None,
    well_island_share_plans: Any = (),
    shared_well_guard_envelope_plans: Any = (),
    shared_well_guard_child_plans: Any = (),
    flexible_envelope_constraints: Any = (),
    flexible_content_bboxes_nm: Any = (),
    minimum_structural_occupancy: Optional[float] = None,
    *,
    intent_basis: IntentBasis | None = None,
    intent_fact_set: IntentFactSet | None = None,
    source_inventory: SourceInventory | None = None,
    generated_incumbent: Any = None,
    cell_def: Any = None,
    row_program: Any = None,
    row_materialization: Any = None,
    native_cross_row_cp_proof: Any = None,
    area_target: Any = None,
    layered_placement_assemblies: tuple = (),
    native_geometry: Any = None,
) -> RepositionPum2DResult:
    """2-D RE-PLACE bridged blocks and preserve the exact final-master certificate.

    ``geom`` / ``pum`` are the bridged geometry + placement (WITH the ``GRING`` ring
    stamps already materialized). Steps:

      1. derive the rigid blocks with the SHARED :func:`extract_blocks`;
      2. preserve each materializer-certified matching atom, bundle passive banks,
         then materialize declared shared device edges as flat rigid composites;
      3. assign each (possibly composite) block a BAND ordinal from device polarity /
         passive-ness (realising the declared ``@band``; polarity fallback otherwise);
      4. build the SOFT position intent -- right-align every passive composite (the
         cap bank -> OUTPUT/right side, default); for each declared
         ``@adjacent(<cap>, <mos>)`` pull the cap bank toward that mos block's x
         (plain band stack only); and pull each split-resistor guard-ring ISLAND
         toward its declared ``@adjacent`` neighbour -- a resistor adjacent to the
         cap bank hugs it in the passive centre channel, a resistor adjacent only
         to a MOS island MOVES into that MOS's band (the pull overrides the passive
         centre band for a small passive block);
      4b. identify the LEFT-COLUMN block from the ``@common_centroid`` role (the
         tall folded diff pair anchors the cell's full-height left column, pulled
         OUT of the top pmos band) via :func:`identify_left_column`;
      4c. lower ``@order(a, b, left_of)`` to HARD same-band block x-order pairs
         (endpoints resolved to blocks by device / group-tile name);
      4d. ROUTING-OWED placement (P4): resolve ``@anchor(net|group, edge=right)``
         to a SOFT right-pull on the target's lower-region output block (the twin
         of the input LEFT column), ``@channel(orient=vertical, width=W)`` to the
         reserved column -> lower-region separation, and ``@trunk`` signal nets to
         SOFT same-band x-proximity pulls between the clusters that share each gate
         net (so one straight horizontal M2 trunk spans them). Absent intent ->
         empty additions (byte-identical);
      5. 2-D re-place with :func:`place_blocks_2d` -- the REFERENCE floorplan
         (full-width pmos top strip / left diff-pair column / nmos+cap lower-right)
         when a left column is identified, the plain band stack otherwise;
      6. distribute each composite's placement back to its member blocks, compute per-
         ORIGINAL-block deltas, HARD-ASSERT no overlap, and shift stamps + RE-
         MATERIALIZE the rings via the SHARED :func:`apply_block_deltas`.

    ``adjacency`` is the exact typed inventory lowered from device-level
    ``@adjacent`` declarations. Its ratio is consumed against the installed PDK
    poly pitch to form a hard maximum edge gap. An empty tuple leaves the cap bank
    right-aligned (the default). Banding is always
    active for this cell (``@band`` declared, else polarity fallback); the pure
    :func:`place_blocks_2d` stays byte-identical when called without ``bands``.

    Returns ``(new_pum, new_geom)`` -- the same pair every emit/verify consumer reads.
    """
    from clay.layout.intent import (
        replay_adjacent_inventory,
        replay_passive_bank_partition_intent,
    )
    if native_geometry is not None:
        from clay.layout.native_sky130 import NativeGeometryBundle
        if type(native_geometry) is not NativeGeometryBundle or cell_def is None:
            raise ValueError('native placement requires its source-bound geometry bundle')
        native_geometry.require_current(cell_def)
    if intent_basis is not None:
        if type(intent_basis) is not IntentBasis:
            raise TypeError("intent_basis must be an exact IntentBasis or None")
        if IntentBasis.from_data(intent_basis.canonical_data()) != intent_basis:
            raise ValueError("intent_basis does not replay byte-equivalently")
    if (intent_fact_set is None) != (source_inventory is None):
        raise TypeError(
            "intent_fact_set and source_inventory must be provided together"
        )
    if intent_fact_set is not None and type(intent_fact_set) is not IntentFactSet:
        raise TypeError("intent_fact_set must be an exact IntentFactSet")
    if source_inventory is not None and type(source_inventory) is not SourceInventory:
        raise TypeError("source_inventory must be an exact SourceInventory")
    reference_conditioned_placement_obligations = (
        replay_reference_conditioned_placement_obligation_inventory(
            getattr(intent, "reference_conditioned_placement_obligations", ())
        )
    )

    placement_objective = _placement_objective_from_intent(intent)
    fixed_outline_width_nm = None
    if outline_width_um is not None:
        if type(outline_width_um) not in {int, float}:
            raise TypeError("outline_width_um must be an exact number or None")
        width_nm = Decimal(str(outline_width_um)) * Decimal(_NM_PER_UM)
        if (
            not width_nm.is_finite()
            or width_nm <= 0
            or width_nm != width_nm.to_integral_value()
        ):
            raise ValueError(
                "outline_width_um must be positive, finite, and exactly representable "
                "in integer nm"
            )
        fixed_outline_width_nm = int(width_nm)
    adjacency_declarations = replay_adjacent_inventory(adjacency)
    raw_passive_partitions = getattr(intent, "passive_bank_partitions", ())
    if type(raw_passive_partitions) is not tuple:
        raise TypeError(
            "passive bank partition inventory must be an exact tuple"
        )
    passive_partition_groups = tuple(
        replay_passive_bank_partition_intent(declaration).members
        for declaration in raw_passive_partitions
    )
    adjacency_pairs = tuple(
        (item.source, item.target) for item in adjacency_declarations
    )
    declared_gap_nm = int(round(float(gap_um) * _NM_PER_UM))
    gap_nm = _effective_block_gap_nm(intent, tech, declared_gap_nm)
    from clay.layout.route.emit_gds import pdk_manufacturing_grid_nm

    manufacturing_grid_nm = pdk_manufacturing_grid_nm(tech)
    # A PDK assembly outline describes cell extent, not a physical mask that
    # prevents two devices from sharing XY.  Keep the original polygons in the
    # stamps and exclude it only from the placement collision projection.
    assembly_boundary = getattr(tech, "assembly_boundary_key", lambda: None)()
    collision_excluded_layers = (assembly_boundary,) if assembly_boundary is not None else ()
    edge_rail_keepouts_nm, edge_rail_nets = _compiled_edge_rail_keepouts_nm(
        intent, tech, bi
    )
    guard_ring_declarations = tuple(getattr(intent, "guard_rings", ()) or ())
    from clay.layout.extract.realize_render import _ring_extension_nm

    guard_ring_outer_clearances_nm = tuple(
        _ring_extension_nm(tech, str(declaration.well))
        for declaration in guard_ring_declarations
    )
    bottom_center_keepout_nm = edge_rail_keepouts_nm.get(("bottom", "H"), 0)
    top_edge_keepout_nm = edge_rail_keepouts_nm.get(("top", "H"), 0)
    top_edge_rail_nets = set(edge_rail_nets.get(("top", "H"), ()))
    bottom_edge_rail_nets = set(edge_rail_nets.get(("bottom", "H"), ()))
    placement_declaration_refs = _placement_declaration_refs(intent)
    try:
        frozen_solve_cuts = tuple(solve_cuts or ())
    except TypeError:
        frozen_solve_cuts = (solve_cuts,)
    feedback_repair = bool(frozen_solve_cuts)
    solve_labels = (
        ("feedback_final_master",)
        if feedback_repair
        else ("initial_master", "exact_materialization_replay")
    )
    solve_ledger = _DeterministicBudgetLedger.split(max_time_s, solve_labels)
    def current_blocks(current_geometry, current_positions):
        from clay.layout.layered_placement_assembly import project_current_layered_placement

        blocks, stamps, rings, ring_names = extract_blocks(current_geometry, current_positions, intent)
        observed, projection = project_current_layered_placement(layered_placement_assemblies,
            source_cell=cell_def, tech=tech, row_program=row_program, row_materialization=row_materialization,
            geometry=current_geometry, placement_um=current_positions, blocks=blocks,
            block_stamps=stamps, block_gring=rings, gring_names=ring_names)
        if projection is not None:
            blocks, stamps, rings, ring_names = (list(projection.blocks), projection.block_stamps,
                                               projection.block_gring, projection.gring_names)
        from clay.layout.place.nonmos_emitter_ownership import canonical_current_nonmos_block_members
        canonical_current_nonmos_block_members(rigid_nonmos_macro_plans,
            block_member_devices=_resolve_block_member_devices(stamps, current_geometry),
            geometry=current_geometry, placement_um=current_positions, block_stamps=stamps)
        return blocks, stamps, rings, ring_names, observed, projection

    # Fold physical accessories before sampling any source ownership or bbox.
    (blocks, block_stamps, block_gring, gring_names,
     layered_replay, layered_projection) = current_blocks(geom, pum)
    canonical_flexible_envelopes = _canonical_flexible_envelope_constraints(
        flexible_envelope_constraints,
        block_ids=tuple(block_id for block_id, _bbox in blocks),
    )
    if canonical_flexible_envelopes:
        blocks = list(
            _replace_flexible_envelope_member_bboxes(
                tuple(blocks),
                canonical_flexible_envelopes,
                flexible_content_bboxes_nm,
            )
        )
    elif flexible_content_bboxes_nm != ():
        raise ValueError("flexible content bboxes require envelope authority")
    # device kind (pmos/nmos/cap/res) from the SHARED bridged intent (no re-derive)
    device_kind: Dict[str, str] = {}
    device_nets: Dict[str, Dict[str, str]] = {}
    for d in getattr(bi, "devices", ()) or []:
        name = str(getattr(d, "name", ""))
        device_kind[name] = str(getattr(d, "kind", ""))
        device_nets[name] = dict(getattr(d, "nets", {}) or {})
    for p in getattr(bi, "passives", ()) or []:
        device_kind[str(getattr(p, "name", ""))] = str(getattr(p, "kind", ""))
    for q in getattr(bi, "bjts", ()) or []:
        device_kind[str(getattr(q, "name", ""))] = str(getattr(q, "kind", "bjt"))
    canonical_device_names = tuple(sorted(device_kind))
    raw_route_groups = getattr(intent, "route_groups", ())
    if type(raw_route_groups) is not tuple:
        raise TypeError("route group inventory must be an exact tuple")
    group_route_requested = bool(raw_route_groups)

    block_member_devices = _resolve_block_member_devices(block_stamps, geom)
    adjusted_geom, adjusted_pum = _materialize_same_ring_passive_adjacency(
        geom,
        pum,
        intent=intent,
        tech=tech,
        tech_id=tech_id,
        bridged_intent=bi,
        declarations=adjacency_declarations,
        device_kind=device_kind,
        block_gring=block_gring,
        gring_names=gring_names,
        block_member_devices=block_member_devices,
        rigid_nonmos_macro_plans=rigid_nonmos_macro_plans,
    )
    if adjusted_geom is not geom or adjusted_pum is not pum:
        geom, pum = adjusted_geom, adjusted_pum
        (blocks, block_stamps, block_gring, gring_names,
         layered_replay, layered_projection) = current_blocks(geom, pum)
        if canonical_flexible_envelopes:
            blocks = list(
                _replace_flexible_envelope_member_bboxes(
                    tuple(blocks),
                    canonical_flexible_envelopes,
                    flexible_content_bboxes_nm,
                )
            )
        block_member_devices = _resolve_block_member_devices(block_stamps, geom)
    if canonical_flexible_envelopes and not feedback_repair and bi is not None:
        # A floating/private MOS can inherit its hard neighbour's packing band.
        # Its foreign-tied ring then owes the existing top-rail reserve. Compile
        # that deterministic primitive artwork before any master dimensions or
        # collision factors are sampled, rather than growing it after Solve.
        source_ids=tuple(block_id for block_id,_bbox in blocks)
        declared_source_bands:set[str]=set()
        source_bands=assign_bands(source_ids,block_member_devices,device_kind,
            device_nets=device_nets,place_groups=tuple(getattr(intent,'place_groups',()) or ()),
            rail_affinities=tuple(getattr(intent,'rail_affinities',()) or ()),
            band_intents=tuple(getattr(intent,'bands',()) or ()),
            _declared_member_blocks=declared_source_bands)
        source_owners={device:block for block in source_ids
                       for device in block_member_devices.get(block,())}
        source_bands.update(_unbound_adjacency_band_overrides(
            tuple((source_owners[left],source_owners[right]) for left,right in adjacency_pairs
                  if left in source_owners and right in source_owners),
            bands=source_bands,declared_band_blocks=declared_source_bands,
            movable_blocks={block for block in source_ids if block_member_devices.get(block)
                and all(device_kind.get(device) in {'nmos','pmos'}
                        for device in block_member_devices[block])}))
        source_left=identify_left_column(source_ids,block_member_devices,
            common_centroid_devices(intent,geom),
            declared_left_groups=(str(left_column_group),) if left_column_group else (),
            require_unique=bool(getattr(intent,'channels',()) or ()))
        source_top,_source_bottom,_source_bottom_blocks=replay_member_ring_padding(
            blocks=tuple(blocks),block_stamps=block_stamps,block_member_devices=block_member_devices,
            bands=source_bands,left_column=source_left,guard_rings=guard_ring_declarations,
            guard_ring_outer_clearances_nm=guard_ring_outer_clearances_nm,
            top_edge_keepout_nm=top_edge_keepout_nm,top_edge_rail_nets=tuple(sorted(top_edge_rail_nets)),
            bottom_edge_keepout_nm=0,bottom_edge_rail_nets=())
        if source_top:
            pum,geom=apply_block_deltas(geom,pum,intent,tech,tech_id,
                {block:(0,0) for block in source_ids},block_stamps,block_gring,gring_names,bi,
                member_ring_pad_top_nm=source_top)
            (blocks,block_stamps,block_gring,gring_names,
             layered_replay,layered_projection)=current_blocks(geom,pum)
            blocks=list(_replace_flexible_envelope_member_bboxes(
                tuple(blocks),canonical_flexible_envelopes,flexible_content_bboxes_nm))
            block_member_devices=_resolve_block_member_devices(block_stamps,geom)
    source_block_member_devices = {
        str(block_id): set(members)
        for block_id, members in block_member_devices.items()
    }
    cross_inventory = None
    cross_directions = {}
    cross_initial_centres = {}
    cross_initial_axes = {}
    legacy_matching_pair_axes = matching_pair_axes
    same_inventory = None
    same_initial_centres = {}
    same_initial_axes = {}
    same_logical_axes = {}
    # Reuse only one invocation-owned source/PDK row replay while building
    # the initial cross/same matching relations. Final pose and certificate
    # replay remain outside this scope. Unknown or mixed row currencies retain
    # the independent standalone validation path.
    matching_scope = nullcontext()
    projection = getattr(row_materialization, "source_projection", None)
    witness = getattr(projection, "native_witness", None)
    native_rows_requested = any(getattr(row, "style", None) == "native_diffusion"
                                for row in tuple(getattr(intent, "arrays", ()) or ()))
    if (witness is not None and row_program is not None and getattr(tech, "pdk_id", None) == "sky130"
            and (native_cross_row_cp_proof is not None or native_rows_requested)):
        from clay.layout.common_centroid_array import CommonCentroidArray
        from clay.layout.mosaic_geometry import MosaicGeometry
        from clay.layout.joint_row_geometry import JointRowGeometryBinding
        from clay.layout.place.row_replay_session import native_matching_row_scope
        realizations = getattr(row_materialization, "realizations", ())
        if realizations and all(
            type(row.geometry) in (CommonCentroidArray, MosaicGeometry)
            and type(getattr(row.geometry, "joint_row_binding", None)) is JointRowGeometryBinding
            for row in realizations
        ):
            matching_scope = native_matching_row_scope(
                program=row_program, materialization=row_materialization, tech=tech)
    with matching_scope:
        if native_cross_row_cp_proof is not None:
            if feedback_repair:
                raise ValueError("native cross-row matching feedback repair is unsupported")
            from clay.layout.place.cross_row_matching import (
                native_cross_row_inventory_from_projection, native_cross_row_stamp_centres,
                replay_native_cross_row_cp_proof,
            )
            cross_inventory = native_cross_row_inventory_from_projection(row_program, row_materialization, tech)
            if cross_inventory is None or type(matching_pair_axes) is not dict:
                raise ValueError("native cross-row source matching inventory is absent")
            pairs = {pair.members for pair in cross_inventory.pairs}
            if not pairs <= set(matching_pair_axes):
                raise ValueError("native cross-row matching axes are incomplete")
            cross_directions = {pair: matching_pair_axes[pair][0] for pair in pairs}
            cross_initial_centres, cross_initial_axes = replay_native_cross_row_cp_proof(
                cross_inventory, native_cross_row_cp_proof, cross_directions, row_program=row_program,
                row_materialization=row_materialization, tech=tech)
            if native_cross_row_stamp_centres(cross_inventory, row_program, row_materialization,
                                              geom, pum) != cross_initial_centres:
                raise ValueError("native cross-row initial stamp poses differ from CP")
            if any(matching_pair_axes[pair] != (axis, coordinate / 4.0)
                   for pair, (axis, coordinate) in cross_initial_axes.items()):
                raise ValueError("native cross-row initial axis differs from CP")
            legacy_matching_pair_axes = {pair: axis for pair, axis in matching_pair_axes.items()
                                         if pair not in pairs}
        if native_rows_requested:
            if cell_def is None or row_program is None or row_materialization is None:
                raise ValueError("native same-row matching requires complete source row authority")
            from clay.layout.place.native_same_row_matching import (
                native_same_row_inventory_from_projection, native_same_row_stamp_centres,
                native_same_row_occurrence_axes, verify_native_same_row_geometry,
            )
            same_inventory = native_same_row_inventory_from_projection(row_program, row_materialization, tech)
            if same_inventory is not None:
                pairs = {pair.members for pair in same_inventory.pairs}
                if type(legacy_matching_pair_axes) is not dict or not pairs <= set(legacy_matching_pair_axes):
                    raise ValueError("native same-row matching axes are incomplete")
                same_logical_axes = {pair: (axis, coordinate4)
                    for pair, axis, coordinate4 in _canonical_mos_matching_axis_rows(
                        legacy_matching_pair_axes, device_kind) if pair in pairs}
                if set(same_logical_axes) != pairs:
                    raise ValueError("native same-row matching source MOS axes are incomplete")
                same_initial_centres = native_same_row_stamp_centres(
                    same_inventory, row_program, row_materialization, geom, pum)
                same_initial_axes = native_same_row_occurrence_axes(
                    same_inventory, same_initial_centres, same_logical_axes)
                verify_native_same_row_geometry(same_inventory, same_initial_centres, same_initial_axes)
                legacy_matching_pair_axes = {pair: axis for pair, axis in legacy_matching_pair_axes.items()
                                             if pair not in pairs}
    mos_matching_rows = _canonical_mos_matching_axis_rows(
        legacy_matching_pair_axes, device_kind
    )
    mos_matching_devices = tuple(sorted({
        member
        for pair, _axis, _coordinate4 in mos_matching_rows
        for member in pair
    }))
    source_mos_matching_bboxes = (
        _source_device_bboxes_nm(geom, pum, mos_matching_devices)
        if mos_matching_devices
        else {}
    )
    alignment_devices = _edge_alignment_bbox_device_inventory(
        intent,
        known_device_names=canonical_device_names,
    )
    source_alignment_bboxes = (
        _source_device_bboxes_nm(geom, pum, alignment_devices)
        if alignment_devices
        else {}
    )
    source_alignment_artwork_bboxes = dict(source_alignment_bboxes)
    source_alignment_occurrence_envelopes = (
        _source_device_occurrence_envelopes_nm(geom, pum, alignment_devices)
        if alignment_devices
        else {}
    )
    for declaration in tuple(getattr(intent, "edge_alignments", ()) or ()):
        occurrence_endpoints = tuple(
            member in source_alignment_occurrence_envelopes
            for member in (declaration.first, declaration.second)
        )
        if occurrence_endpoints[0] != occurrence_endpoints[1]:
            raise ValueError(
                "edge alignment cannot mix occurrence and artwork bbox frames"
            )
    source_alignment_bboxes.update(source_alignment_occurrence_envelopes)
    deferred_cross_row_alignments = ()
    deferred_native_left_alignments = ()
    island_block_ids = set(block_gring.keys())  # guard-ring islands: never bundled
    materialized_mim_pair_clearances_nm = _mim_pair_clearances_nm(
        tech,
        block_stamps=block_stamps,
        geometry=geom,
        base_gap_nm=gap_nm,
    )
    materialized_native_access_pair_clearances_nm = _mim_native_access_pair_clearances_nm(
        tech, block_stamps=block_stamps, geometry=geom, base_gap_nm=gap_nm,
        device_nets=device_nets,
        rail_specs=_compiled_edge_rail_specs_for_placement(intent, tech, bi),
        placement_um=pum,
    )

    # ---- 2. expose every precompiled aggregate as one exact solver atom, then
    #         bundle any remaining legacy all-cap place group.
    raw_place_groups = getattr(intent, "place_groups", ())
    if type(raw_place_groups) is not tuple:
        raise TypeError("place group inventory must be an exact tuple")
    relational_group_edges: Tuple[Tuple[str, str], ...] = ()
    if intent_fact_set is not None:
        relational_group_edges = _relational_place_group_envelope_edges(
            intent_fact_set,
            raw_place_groups,
        )
    place_groups = raw_place_groups
    locally_consumed_order_constraints: set[Tuple[str, str, str]] = set(
        _discrete_mosaic_local_order_proofs(
            tuple(getattr(intent, "discrete_mosaics", ()) or ()),
            tuple(getattr(intent, "order_constraints", ()) or ()),
        )
    )
    feedback_master_members: Optional[Dict[str, Tuple[str, ...]]] = None
    # Pair evidence is produced only by the native edge-alignment pre-bundle.
    alignment_native_feol_pairs = ()
    alignment_native_feol_clearances = {}
    if feedback_repair:
        try:
            (
                new_blocks,
                composite_members,
                member_offsets,
                block_member_devices,
                feedback_master_members,
            ) = _feedback_final_master_inventory(
                blocks,
                block_member_devices,
                solve_cut_source_certificate,
            )
        except (AttributeError, TypeError, ValueError):
            if isinstance(solve_cut_source_certificate, FinalPlacementCertificate):
                raise PlacementSolveCutResidualError(
                    replace(
                        solve_cut_source_certificate,
                        status="RESIDUAL_STALE_SOLVE_CUT",
                        geometry_legal=False,
                        cuts_satisfied=False,
                        applied_cut_hashes=(),
                    )
                )
            raise
    else:
        (
            matching_blocks,
            matching_composites,
            matching_offsets,
            matching_consumed_orders,
        ) = (
            _bundle_mos_matching_composites(
                blocks,
                block_member_devices,
                matching_pair_axes=legacy_matching_pair_axes,
                device_kind=device_kind,
                guard_ring_owner_ids=tuple(sorted(island_block_ids)),
                flexible_owner_ids=tuple(
                    sorted(
                        {
                            member
                            for constraint in canonical_flexible_envelopes
                            for member in constraint.members
                        }
                    )
                ),
                place_groups=place_groups,
                order_constraints=tuple(
                    getattr(intent, "order_constraints", ()) or ()
                ),
                gap_nm=gap_nm,
            )
        )
        locally_consumed_order_constraints.update(matching_consumed_orders)
        for comp_id, parts in matching_composites.items():
            block_member_devices[comp_id] = set().union(
                *(block_member_devices.get(part, set()) for part in parts)
            )
        rigid_composites: Dict[str, List[str]] = {}
        rigid_offsets: Dict[str, Tuple[str, Tuple[int, int]]] = {}
        rigid_blocks = list(matching_blocks)
        if rigid_nonmos_macro_plans is not None:
            if not isinstance(rigid_nonmos_placed_devices, Mapping):
                raise TypeError(
                    "rigid non-MOS compaction requires placed-device authority"
                )
            for plan in getattr(rigid_nonmos_macro_plans, "plans", ()):
                plan.translation_nm(rigid_nonmos_placed_devices)
                relation_refs = {
                    projection.declaration_ref
                    for projection in tuple(
                        getattr(plan, "local_relation_projections", ()) or ()
                    )
                }
                locally_consumed_order_constraints.update(
                    tuple(str(value) for value in raw)
                    for raw in tuple(
                        getattr(intent, "order_constraints", ()) or ()
                    )
                    if f"@order({raw[0]},{raw[1]},{raw[2]})" in relation_refs
                )
            from clay.layout.extract.realize_render import _placed_stamp_bboxes_nm

            stamp_bboxes = _placed_stamp_bboxes_nm(geom, pum)
            rigid_members = {
                member
                for plan in rigid_nonmos_macro_plans.plans
                for member in plan.members
            }
            route_member_bboxes: Dict[str, _Bbox] = {}
            for stamp_name, stamp in geom.items():
                owners = tuple(getattr(stamp, "composite_members", ()) or ())
                if not owners and stamp_name in rigid_members:
                    owners = (stamp_name,)
                for owner in owners:
                    if owner not in rigid_members:
                        continue
                    if owner in route_member_bboxes or stamp_name not in stamp_bboxes:
                        raise ValueError(
                            f"rigid non-MOS member {owner!r} has ambiguous route artwork"
                        )
                    route_member_bboxes[owner] = stamp_bboxes[stamp_name]
            rigid_blocks, rigid_composites, rigid_offsets = (
                _bundle_rigid_nonmos_composites(
                    rigid_blocks,
                    block_member_devices,
                    rigid_nonmos_macro_plans,
                    route_member_bboxes,
                )
            )
            for comp_id, parts in rigid_composites.items():
                block_member_devices[comp_id] = set().union(
                    *(block_member_devices.get(part, set()) for part in parts)
                )
        new_blocks, passive_composites, passive_offsets = bundle_passive_composites(
            rigid_blocks,
            block_member_devices,
            device_kind,
            place_groups,
            gap_nm,
            island_block_ids,
            order_constraints=tuple(getattr(intent, "order_constraints", ()) or ()),
            pair_clearances_nm=_project_pair_clearances_through_composites(
                materialized_mim_pair_clearances_nm,
                live_ids=tuple(name for name, _bbox in rigid_blocks),
                composites={**matching_composites, **rigid_composites},
                base_gap_nm=gap_nm,
            ),
            matching_pair_axes=matching_pair_axes or {},
            partitioned_passive_groups=passive_partition_groups,
            edge_alignments=tuple(getattr(intent, "edge_alignments", ()) or ()),
        )
        composite_members = {
            **matching_composites,
            **rigid_composites,
            **passive_composites,
        }
        member_offsets = {
            **matching_offsets,
            **rigid_offsets,
            **passive_offsets,
        }
        for comp_id, parts in composite_members.items():
            devs: set = set()
            for p in parts:
                devs |= block_member_devices.get(p, set())
            block_member_devices[comp_id] = devs
        for raw in tuple(getattr(intent, "order_constraints", ()) or ()):
            left, right, relation = (str(value) for value in raw)
            if relation != "left_of":
                continue
            owners = tuple(
                comp_id
                for comp_id in passive_composites
                if {left, right} <= block_member_devices.get(comp_id, set())
            )
            if len(owners) == 1:
                locally_consumed_order_constraints.add((left, right, relation))
        # Retain the PDK-backed pre-merge pair evidence for an independent
        # replay against the actual stamps after the merged master is placed.
        if alignment_devices:
            projected_alignment_bboxes = (
                _project_source_device_bboxes_to_solver_masters(
                    source_alignment_bboxes,
                    source_blocks=tuple(blocks),
                    source_block_member_devices=source_block_member_devices,
                    solver_blocks=tuple(new_blocks),
                    solver_block_member_devices={
                        block_id: set(block_member_devices[block_id])
                        for block_id, _bbox in new_blocks
                    },
                    member_offsets=member_offsets,
                )
            )
            projected_alignment_artwork_bboxes = (
                _project_source_device_bboxes_to_solver_masters(
                    source_alignment_artwork_bboxes,
                    source_blocks=tuple(blocks),
                    source_block_member_devices=source_block_member_devices,
                    solver_blocks=tuple(new_blocks),
                    solver_block_member_devices={
                        block_id: set(block_member_devices[block_id])
                        for block_id, _bbox in new_blocks
                    },
                    member_offsets=member_offsets,
                )
            )
            pre_alignment_master_members = {
                block_id: tuple(composite_members.get(block_id, (block_id,)))
                for block_id, _bbox in new_blocks
            }
            pre_alignment_block_member_devices = {
                block_id: set(block_member_devices[block_id])
                for block_id, _bbox in new_blocks
            }
            pre_alignment_layered_footprints = (
                _layered_collision_footprints_from_geometry(
                    excluded_layers=collision_excluded_layers,
                    solver_blocks=tuple(new_blocks),
                    materialized_blocks=tuple(blocks),
                    block_stamps=block_stamps,
                    block_gring=block_gring,
                    geometry=geom,
                    placement_um=pum,
                    master_members=pre_alignment_master_members,
                    member_offsets_nm=member_offsets,
                    flexible_envelope_constraints=canonical_flexible_envelopes,
                )
            )
            native_feol_pairs = ()
            pair_layer_clearances = _same_bulk_nwell_pair_clearances_nm(
                block_member_devices=pre_alignment_block_member_devices,
                device_nets=device_nets, tech=tech)
            if native_geometry is not None and tech_id == 'sky130':
                from clay.layout.intent import replay_edge_alignment_inventory
                from clay.layout.place.edge_rail_geometry import _bind_inventory
                from clay.layout.place.native_feol_reservation import (
                    compile_sky130_native_feol_pairs, sky130_native_row_kinds,
                )

                owner_by_device = {device: owner for owner, members in pre_alignment_block_member_devices.items()
                                   for device in members}
                declared_abuts = {frozenset(pair) for pair in (getattr(intent, 'abut_pairs', ()) or ())}
                row_by_member = {}
                if row_program is not None:
                    for entry in row_program.entries:
                        for member in entry.consumed_devices:
                            if member in row_by_member:
                                raise ValueError('native FEOL row source owner is duplicated')
                            row_by_member[member] = entry
                source_names=set(canonical_device_names)
                def complete_source_owner(master, member):
                    owned=pre_alignment_block_member_devices[master]
                    if owned == {member}:
                        return True
                    entry=row_by_member.get(member)
                    return (entry is not None
                            and pre_alignment_master_members[master] == (master,)
                            and set(owned) & source_names == set(entry.consumed_devices)
                            and set(owned) - source_names <= {entry.group}
                            and tuple(block_stamps.get(master, ())) == (entry.group,))
                eligible = set()
                for declaration in replay_edge_alignment_inventory(
                        tuple(getattr(intent, 'edge_alignments', ()) or ())):
                    a, b = declaration.first, declaration.second
                    if frozenset((a, b)) not in declared_abuts:
                        continue
                    if a not in native_geometry.geometries or b not in native_geometry.geometries:
                        continue
                    if {native_geometry.geometries[a].kind, native_geometry.geometries[b].kind} not in ({'nmos', 'pmos'}, {'nmos'}):
                        continue
                    first, second = owner_by_device.get(a), owner_by_device.get(b)
                    if (first is not None and second is not None and first != second
                            and complete_source_owner(first, a)
                            and complete_source_owner(second, b)):
                        eligible.add(tuple(sorted((first, second))))
                if eligible:
                    stamps, _atoms = _bind_inventory(
                        solver_blocks=tuple(new_blocks), materialized_blocks=tuple(blocks),
                        block_stamps=block_stamps, block_gring=block_gring,
                        geometry=geom, placement_um=pum,
                        master_members=pre_alignment_master_members, member_offsets_nm=member_offsets,
                        relevant_layers=frozenset(), conductors=(),
                        flexible_envelope_constraints=canonical_flexible_envelopes)
                    row_kinds = sky130_native_row_kinds(bundle=native_geometry, cell=cell_def,
                        stamps=stamps, geometry=geom, row_program=row_program,
                        row_materialization=row_materialization, tech=tech)
                    native_feol_pairs, native_clearances = compile_sky130_native_feol_pairs(
                        bundle=native_geometry, cell=cell_def, stamps=stamps,
                        geometry=geom, placement_um=pum, eligible_master_pairs=tuple(sorted(eligible)),
                        row_kinds=row_kinds, row_program=row_program,
                        row_materialization=row_materialization, tech=tech)
                    pair_layer_clearances.update(native_clearances)
                    alignment_native_feol_pairs = tuple(
                        pair for pair in native_feol_pairs
                        if pair.left_stamp in row_kinds or pair.right_stamp in row_kinds)
                    row_master_pairs={tuple(sorted((pair.left_master,pair.right_master)))
                                      for pair in alignment_native_feol_pairs}
                    alignment_native_feol_clearances = {
                        key:value for key,value in native_clearances.items() if key[:2] in row_master_pairs}
            bundled_alignments, deferred_cross_row_alignments = _partition_cross_row_bottom_alignments(
                tuple(getattr(intent, 'edge_alignments', ()) or ()), cross_inventory, row_program,
                source_alignment_occurrence_envelopes, cross_initial_centres, cross_initial_axes,
                pre_alignment_block_member_devices, pre_alignment_master_members, block_stamps,
                canonical_device_names, geom)
            if deferred_cross_row_alignments:
                _require_cross_row_source_alignment_authority(
                    tuple(getattr(intent, 'edge_alignments', ()) or ()), cross_inventory,
                    row_materialization.source_projection, row_program)
            if native_geometry is not None and row_program is not None:
                bundled_alignments, deferred_native_left_alignments = _partition_native_left_alignments(
                    bundled_alignments, intent, row_program, geom, projected_alignment_bboxes,
                    tuple(new_blocks), pre_alignment_block_member_devices, block_stamps,
                    canonical_device_names, manufacturing_grid_nm)
                if deferred_native_left_alignments:
                    _require_native_left_alignment_authority(intent, row_materialization.source_projection, row_program)
            (
                new_blocks,
                composite_members,
                member_offsets,
                block_member_devices,
                alignment_consumed_orders,
            ) = _bundle_declared_edge_alignment_composites(
                new_blocks,
                block_member_devices,
                alignments=bundled_alignments,
                order_constraints=tuple(
                    getattr(intent, "order_constraints", ()) or ()
                ),
                abut_pairs=tuple(getattr(intent, "abut_pairs", ()) or ()),
                abut_device_bboxes_nm=projected_alignment_artwork_bboxes,
                device_bboxes_nm=projected_alignment_bboxes,
                composite_members=composite_members,
                member_offsets=member_offsets,
                gap_nm=gap_nm,
                manufacturing_grid_nm=manufacturing_grid_nm,
                layered_collision_footprints=pre_alignment_layered_footprints,
                layer_clearances_nm=_pdk_layer_clearances_nm(tech),
                pair_layer_clearances_nm=pair_layer_clearances,
                native_feol_pairs=native_feol_pairs,
                overlap_compatible_pair_layers=(
                    _same_bulk_mergeable_pair_layers(
                        block_member_devices=pre_alignment_block_member_devices,
                        device_nets=device_nets,
                        tech=tech,
                    )
                ),
                forbidden_owner_ids=tuple(sorted({
                    *island_block_ids,
                    *(
                        member
                        for constraint in canonical_flexible_envelopes
                        for member in constraint.members
                    ),
                })),
            )
            locally_consumed_order_constraints.update(alignment_consumed_orders)
    new_ids = [bid for bid, _ in new_blocks]
    # Every proposed carrier is pinned to y=0 by the placement master.  The
    # exemption therefore cannot authorize a same-net ring that CP later stacks
    # above the physical boundary.
    same_net_ring_blocks = tuple(
        sorted(
            f"ISL{index}"
            for index, ring in enumerate(
                tuple(getattr(intent, "guard_rings", ()) or ())
            )
            if f"ISL{index}" in new_ids
            and str(getattr(ring, "tie_net", "") or "")
            in bottom_edge_rail_nets
        )
    )
    bottom_edge_keepout_exempt_blocks = same_net_ring_blocks
    # Feedback consumes the same canonical tie-net proof.  The final exact rail
    # planner, not a prior synthetic pad token, decides whether the intrinsic
    # boundary carrier is compatible with the complete same-layer conductor scene.
    spacing_master_members = (
        feedback_master_members
        if feedback_master_members is not None
        else {
            block_id: tuple(composite_members.get(block_id, (block_id,)))
            for block_id in new_ids
        }
    )
    if feedback_master_members is not None:
        rigid_compound_owner_ids = tuple(sorted(
            master
            for master, members in spacing_master_members.items()
            if master.startswith("A_nonrow:") and len(tuple(members)) > 1
        ))
    else:
        rigid_member_sets = {
            frozenset(parts) for parts in rigid_composites.values()
        }
        rigid_compound_owner_ids = tuple(sorted(
            master
            for master, members in spacing_master_members.items()
            if frozenset(members) in rigid_member_sets
        ))
    member_rigid_compound_footprints = _rigid_compound_footprints_from_members(
        solver_blocks=tuple(new_blocks),
        materialized_blocks=tuple(blocks),
        master_members=spacing_master_members,
        compound_owner_ids=rigid_compound_owner_ids,
    )
    fixed_pcell_compound_footprints = (
        _fixed_pcell_compound_footprints_from_geometry(
            solver_blocks=tuple(new_blocks),
            block_stamps=block_stamps,
            geometry=geom,
            master_members=spacing_master_members,
        )
    )
    fixed_pcell_compound_owners = {
        owner for owner, _rectangles in fixed_pcell_compound_footprints
    }
    rigid_compound_footprints = _canonical_rigid_compound_footprints(
        tuple(sorted(
            (
                *(
                    row
                    for row in member_rigid_compound_footprints
                    if row[0] not in fixed_pcell_compound_owners
                ),
                *fixed_pcell_compound_footprints,
            )
        )),
        dims={
            owner: (bbox[2] - bbox[0], bbox[3] - bbox[1])
            for owner, bbox in new_blocks
        },
    )
    live_master_devices = {
        block_id: set(block_member_devices.get(block_id, ()))
        for block_id in sorted(new_ids)
    }
    row_bounded_cavity_projections = (
        _reference_conditioned_row_bounded_cavities_from_intent(
            intent,
            live_master_devices,
            rigid_compound_footprints=rigid_compound_footprints,
            dims={
                owner: (bbox[2] - bbox[0], bbox[3] - bbox[1])
                for owner, bbox in new_blocks
            },
        )
    )
    aggregate_envelope_orders = _aggregate_envelope_orders_from_intent(
        intent,
        live_master_devices,
        known_device_names=tuple(sorted(device_kind)),
        composite_alias_members={name: tuple(stamp.composite_members)
                                 for name, stamp in geom.items()
                                 if getattr(stamp, "composite_members", ())},
    )
    canonical_device_names = tuple(sorted(device_kind))
    from clay.layout.mosaic_guest_slot_stamp import build_mosaic_guest_slot_projections
    mosaic_guest_slot_projections = build_mosaic_guest_slot_projections(
        intent, solver_blocks=tuple(new_blocks), block_stamps=block_stamps,
        master_members=spacing_master_members,
        owner_logicals={owner: tuple(sorted(set(members) & set(canonical_device_names)))
                        for owner, members in live_master_devices.items()},
        geometry=geom, placement_um=pum, minimum_clearance_nm=gap_nm, tech=tech,
    )
    from clay.layout.route.group_ports import (
        build_explicit_route_group_envelope_constraints,
    )

    route_group_envelopes = build_explicit_route_group_envelope_constraints(
        intent,
        blocks=tuple(sorted(new_blocks)),
        block_member_devices={
            block_id: set(block_member_devices[block_id])
            for block_id in sorted(new_ids)
        },
        known_device_names=canonical_device_names,
    )
    canonical_flexible_envelopes = _canonical_flexible_envelope_constraints(
        tuple(sorted((*canonical_flexible_envelopes, *route_group_envelopes))),
        block_ids=tuple(sorted(new_ids)),
    )
    relaxed_ring_pair_clearances_nm = _ring_pair_clearances_nm(
        intent,
        tech,
        block_ids=new_ids,
        block_member_devices=block_member_devices,
        known_device_names=canonical_device_names,
        well_island_share_plans=well_island_share_plans,
        base_gap_nm=gap_nm,
    )
    well_share_seed_cohesion_groups = _well_share_ring_cohesion_groups(
        intent,
        tech,
        block_ids=tuple(sorted(new_ids)),
        block_member_devices=block_member_devices,
        known_device_names=canonical_device_names,
        well_island_share_plans=well_island_share_plans,
    )
    pair_clearances_nm = _merge_pair_clearances_nm(
        relaxed_ring_pair_clearances_nm,
        _project_pair_clearances_to_masters(
            materialized_mim_pair_clearances_nm,
            master_members=spacing_master_members,
            base_gap_nm=gap_nm,
        ),
    )
    native_access_pair_clearances_nm = _project_native_access_pair_clearances_to_masters(
        materialized_native_access_pair_clearances_nm,
        master_members=spacing_master_members, base_gap_nm=gap_nm,
    )
    derived_overlap_compatible_pairs = tuple(
        pair
        for pair in _layer_disjoint_overlap_compatible_pairs(
            excluded_layers=collision_excluded_layers,
            block_stamps=block_stamps,
            geometry=geom,
            master_members=spacing_master_members,
        )
        if pair not in pair_clearances_nm and pair not in native_access_pair_clearances_nm
    )

    # ---- 3. band ordinals (realise @band; polarity fallback). Banding is active
    #         whenever @band is declared OR polarity is derivable (always here).
    declared_band_order = bool(getattr(intent, "bands", ()))
    band_active = declared_band_order or bool(device_kind)
    strict_band_member_blocks: set[str] = set()
    bands_map = (
        assign_bands(
            new_ids,
            block_member_devices,
            device_kind,
            device_nets=device_nets,
            place_groups=place_groups,
            rail_affinities=tuple(getattr(intent, "rail_affinities", ()) or ()),
            band_intents=tuple(getattr(intent, "bands", ()) or ()),
            _declared_member_blocks=strict_band_member_blocks,
        )
        if band_active
        else {}
    )
    component_refinement_owners = _layered_component_refinement_owners(
        aggregate_envelope_orders,
        bands=bands_map,
        fixed_pcell_compound_footprints=fixed_pcell_compound_footprints,
        layered_placement_projection=layered_projection,
        master_members=spacing_master_members,
        rigid_nonmos_macro_plans=rigid_nonmos_macro_plans,
        rigid_nonmos_placed_devices=rigid_nonmos_placed_devices,
        block_member_devices=block_member_devices,
        geometry=geom, placement_um=pum, block_stamps=block_stamps,
    )
    from clay.layout.canonical_collision_projection import (
        build_canonical_collision_projection_scope,
    )

    collision_projection_scope = build_canonical_collision_projection_scope(
        master_members=spacing_master_members,
        component_refinement_owners=component_refinement_owners,
    )
    from clay.layout.place.route_escape_reservation import (
        native_row_escape_pair_clearances_nm,
    )
    _early_dev_block = {
        d: bid for bid in new_ids for d in block_member_devices.get(bid, ())
    }
    _early_adjacency_limits_nm = _resolved_adjacency_limits_nm(
        adjacency_declarations,
        device_blocks=_early_dev_block,
        tech=tech,
        minimum_gap_nm=gap_nm,
    )
    _guest_masters = (
        {
            str(g)
            for row in row_bounded_cavity_projections
            for g in getattr(row, "guest_masters", ())
        }
        | {
            str(getattr(slot, "guest_master", ""))
            for slot in mosaic_guest_slot_projections
            if getattr(slot, "guest_master", None)
        }
    )
    _row_escape_clearances_nm = native_row_escape_pair_clearances_nm(
        intent=intent,
        tech=tech,
        cell_def=cell_def,
        row_program=row_program,
        solver_blocks=tuple(new_blocks),
        materialized_blocks=tuple(blocks),
        master_members=spacing_master_members,
        member_offsets_nm=member_offsets,
        block_stamps=block_stamps,
        block_gring=block_gring,
        block_member_devices=block_member_devices,
        geometry=geom,
        placement_um=pum,
        collision_excluded_layers=collision_excluded_layers,
        refined_collision_owners=set(collision_projection_scope.master_owners),
        base_gap_nm=gap_nm,
        adjacency_limits_nm=_early_adjacency_limits_nm,
        guest_masters=_guest_masters,
    )
    if _row_escape_clearances_nm:
        pair_clearances_nm = _merge_pair_clearances_nm(
            pair_clearances_nm,
            _row_escape_clearances_nm,
        )
        derived_overlap_compatible_pairs = tuple(
            pair
            for pair in derived_overlap_compatible_pairs
            if pair not in pair_clearances_nm
        )
    layered_collision_footprints = _layered_collision_footprints_from_geometry(
        excluded_layers=collision_excluded_layers,
        solver_blocks=tuple(new_blocks),
        materialized_blocks=tuple(blocks),
        block_stamps=block_stamps,
        block_gring=block_gring,
        geometry=geom,
        placement_um=pum,
        master_members=spacing_master_members,
        member_offsets_nm=member_offsets,
        component_refinement_owners=collision_projection_scope.master_owners,
        flexible_envelope_constraints=canonical_flexible_envelopes,
    )
    canonical_flexible_envelopes = _canonical_flexible_envelope_constraints(
        canonical_flexible_envelopes,
        block_ids=tuple(sorted(new_ids)),
        bands=bands_map if band_active else None,
    )
    strict_band_order = _strict_band_order_required(
        source_declares_band=declared_band_order,
        block_ids=new_ids,
        bands=bands_map,
        declared_member_blocks=strict_band_member_blocks,
    )
    bottom_edge_master_keepout_nm = int(bottom_center_keepout_nm)
    bottom_edge_keepout_clear_candidates = (
        _bottom_edge_keepout_clear_blocks(
            intent,
            tech,
            bi,
            geometry=geom,
            block_stamps=block_stamps,
            block_gring=block_gring,
            master_members=spacing_master_members,
            bands=bands_map,
        )
        if bottom_edge_master_keepout_nm > 0
        else ()
    )
    top_edge_keepout_clear_candidates = (
        _top_edge_keepout_clear_blocks(
            intent,
            tech,
            bi,
            geometry=geom,
            block_stamps=block_stamps,
            block_gring=block_gring,
            master_members=spacing_master_members,
            bands=bands_map,
        )
        if top_edge_keepout_nm > 0
        and fixed_outline_width_nm is not None
        and canonical_flexible_envelopes
        else ()
    )

    # ---- 4. soft position intent. Two GENERAL @adjacent consumers keyed on block
    #         role (never a device / circuit name):
    #   (a) the cap-bank COMPOSITE right-aligns (output side) and, when NO left
    #       column pulls it, is pulled toward each declared @adjacent mos block's x;
    #   (b) each small-passive GUARD-RING ISLAND (a split-resistor pair) is pulled
    #       toward its declared @adjacent neighbour. A resistor whose neighbour is
    #       the cap-bank composite HUGS the cap in the passive centre channel (no
    #       band change); a resistor whose neighbours are only MOS islands MOVES to
    #       that MOS's band (a degeneration resistor drops into the device band next
    #       to its device), the @adjacent pull overriding the passive centre band.
    cap_composites = {
        comp_id
        for comp_id in composite_members
        if block_member_devices.get(comp_id)
        and all(
            device_kind.get(device) == "cap"
            for device in block_member_devices[comp_id]
        )
    }
    right_pull = set(cap_composites)
    dev_block = {d: bid for bid in new_ids for d in block_member_devices.get(bid, ())}
    axis_solver_positions = _diff_axis_positions_in_solver_frame(
        source_blocks=tuple(blocks), source_members=source_block_member_devices,
        solver_blocks=tuple(new_blocks), solver_members=block_member_devices,
        member_offsets=member_offsets, placement_um=pum,
    )
    axis_alignment_groups = _diff_pair_axis_alignment_groups(
        new_blocks,
        block_member_devices,
        geom,
        axis_solver_positions,
        intent,
        bi,
    )
    from clay.layout.place.placement_geometry import _source_channel_alignment_groups
    if getattr(intent, 'channel_alignments', ()):
        from clay.layout.joint_placement_terminals import build_placement_terminal_face
        build_placement_terminal_face(tech=tech, geometry=geom, placement_um=axis_solver_positions,
            row_program=row_program, row_materialization=row_materialization,
            intent=intent, source_cell=cell_def)
    axis_alignment_groups = tuple(sorted(set((*axis_alignment_groups,
        *_source_channel_alignment_groups(new_blocks, block_member_devices,
                                          geom, axis_solver_positions, intent)))))
    native_left_axis_groups = _native_left_alignment_axis_groups(
        deferred_native_left_alignments, row_program, new_blocks, block_member_devices,
        geom, axis_solver_positions)
    axis_alignment_groups = tuple(sorted(set((*axis_alignment_groups, *native_left_axis_groups))))
    center_alignment_pairs = _mosaic_center_alignment_pairs(
        intent,
        live_master_devices,
    )
    cross_solver_owners = {}
    cross_center_alignment_pairs = ()
    if cross_inventory is not None:
        from clay.layout.place.cross_row_matching import (
            native_cross_row_center_alignments, native_cross_row_solver_owners,
        )
        solver_boxes = dict(new_blocks)
        realizations = {row.entry_id: row for row in row_materialization.realizations}
        cross_solver_owners = native_cross_row_solver_owners(
            cross_inventory, row_program, live_master_devices, spacing_master_members,
            block_stamps, canonical_device_names)
        for row, members in cross_inventory.row_source_members:
            master = cross_solver_owners[row]
            bbox = solver_boxes[master]
            source_box = realizations[row].local_bbox_nm
            if (bbox[2] - bbox[0], bbox[3] - bbox[1]) != (
                    source_box[2] - source_box[0], source_box[3] - source_box[1]):
                raise ValueError("native cross-row solver master does not exactly frame row")
        cross_center_alignment_pairs = native_cross_row_center_alignments(
            cross_inventory, cross_initial_centres, cross_initial_axes, cross_solver_owners)
        center_alignment_pairs = tuple(sorted(set((*center_alignment_pairs, *cross_center_alignment_pairs))))
    (
        hierarchical_macro_orders,
        hierarchical_macro_alignments,
    ) = _hierarchical_macro_constraints_from_intent(
        intent,
        live_master_devices,
        known_device_names=canonical_device_names,
        tech=tech,
        cell_def=cell_def,
        base_gap_nm=gap_nm,
    )
    hierarchical_macro_cohesion_groups = (
        _hierarchical_macro_cohesion_groups_from_intent(
            intent,
            live_master_devices,
            known_device_names=canonical_device_names,
        )
    )
    cap_devs: set = set()
    for comp_id in cap_composites:
        cap_devs |= block_member_devices.get(comp_id, set())

    # (a) cap-bank composite -> @adjacent(mos) pulls (used on the plain band stack;
    #     the floorplan path keeps the cap on the right-align alone).
    cap_pull_pairs: List[Tuple[str, str]] = []
    for s, t in adjacency_pairs:
        if s not in cap_devs or device_kind.get(t) not in ("pmos", "nmos"):
            continue
        tb = dev_block.get(t)
        if tb is None:
            continue
        for comp_id, parts_devs in (
            (c, block_member_devices.get(c, set())) for c in cap_composites
        ):
            if (
                s in parts_devs
                and comp_id != tb
                and (comp_id, tb) not in cap_pull_pairs
            ):
                cap_pull_pairs.append((comp_id, tb))

    # (b) split-resistor ISLAND -> @adjacent(neighbour) pulls + optional band move.
    res_islands = {
        bid
        for bid in new_ids
        if bid in island_block_ids
        and (lambda ks: bool(ks) and ks <= {"res", "cap"})(
            {
                device_kind.get(d)
                for d in block_member_devices.get(bid, ())
                if device_kind.get(d)
            }
        )
    }
    # gather each resistor island's distinct-block @adjacent targets
    res_targets: Dict[str, set] = {bid: set() for bid in res_islands}
    for s, t in adjacency_pairs:
        sb = dev_block.get(s)
        tb = dev_block.get(t)
        if sb in res_islands and tb is not None and tb != sb:
            res_targets[sb].add(tb)
    cap_blocks = set(cap_composites)
    res_pull_rows, band_override = _resistor_adjacency_terms(
        res_targets,
        cap_blocks=cap_blocks,
        bands=bands_map,
        declared_band_blocks=strict_band_member_blocks,
    )
    res_pull_pairs = list(res_pull_rows)
    adjacency_limits_nm = _resolved_adjacency_limits_nm(
        adjacency_declarations,
        device_blocks=dev_block,
        tech=tech,
        minimum_gap_nm=gap_nm,
    )

    # Without an explicit @band, adjacency may still choose the neighbour's
    # inferred polarity band.  A declared center band is a hard structural fact.
    for rid, bnd in band_override.items():
        bands_map[rid] = bnd
    unbound_mos_blocks = {
        bid for bid in new_ids
        if block_member_devices.get(bid)
        and all(device_kind.get(device) in {"nmos", "pmos"}
                for device in block_member_devices[bid])
    }
    bands_map.update(_unbound_adjacency_band_overrides(
        tuple((dev_block[s], dev_block[t]) for s, t in adjacency_pairs
              if s in dev_block and t in dev_block),
        bands=bands_map,
        declared_band_blocks=strict_band_member_blocks,
        movable_blocks=unbound_mos_blocks,
    ))
    bottom_edge_keepout_exempt_blocks = (
        _adjacency_compatible_bottom_exemptions(
            bottom_edge_keepout_exempt_blocks,
            adjacency_limits_nm=adjacency_limits_nm,
            bands=bands_map,
        )
    )
    bottom_edge_keepout_exempt_blocks = (
        _hard_placement_compatible_bottom_exemptions(
            bottom_edge_keepout_exempt_blocks,
            axis_alignment_groups=axis_alignment_groups,
            bands=bands_map,
        )
    )

    # ---- 4b. identify the LEFT-COLUMN block (the common-centroid diff pair). The
    #          @common_centroid role -> the tall diff-pair block is pulled OUT of
    #          the top pmos band and anchored as the cell's full-height left
    #          column (topology-derived; see identify_left_column's language-gap
    #          note). None when no @common_centroid is declared -> plain band stack.
    cc_devs = common_centroid_devices(intent, geom)
    declared_left_group = str(left_column_group or "")
    left_column = identify_left_column(
        new_ids,
        block_member_devices,
        cc_devs,
        declared_left_groups=(declared_left_group,) if declared_left_group else (),
        require_unique=bool(getattr(intent, "channels", ()) or ()),
    )
    (
        relational_order_pairs,
        relational_lateral_blocks,
        left_column,
    ) = _relational_place_group_master_constraints(
        relational_group_edges,
        tuple(place_groups),
        dev_block,
        left_column,
    )
    relational_lateral_blocks = tuple(
        sorted(
            set(relational_lateral_blocks)
            | set(_nonrigid_place_group_blocks(tuple(place_groups), dev_block))
        )
    )

    # ---- Wave D (flag-gated, OFF by default): LANDSCAPE-DENSE general 2D pack.
    #      The reference-skeleton floorplan parks the cap bank in the EXCLUSIVE
    #      @band(center) height tier (~33um) + adds a dead OD band above the short
    #      @common_centroid left column -> a portrait cell (root-cause 2026-07-17).
    #      When CLAY_LANDSCAPE_PLACE is set, DROP the band-stack skeleton entirely
    #      (no per-band ordinal, no inferred left column) so place_blocks_2d runs its
    #      GENERAL aspect-optimised 2D pack (the FINE path, spec D1): every block --
    #      the cap COMPOSITE included -- is a free element packed toward aspect_target
    #      with the @adjacent net-pulls, so the cap fills LATERAL SLACK instead of an
    #      exclusive tier. OFF -> byte-identical to the reference-skeleton floorplan.
    if _landscape_place_enabled():
        bands_map = {}
        left_column = None

    bottom_edge_keepout_clear_blocks = (
        tuple(
            sorted(
                set(bottom_edge_keepout_clear_candidates)
                - set(bottom_edge_keepout_exempt_blocks)
            )
        )
        if left_column is not None
        else ()
    )
    top_edge_master_keepout_nm = (
        int(top_edge_keepout_nm)
        if fixed_outline_width_nm is not None
        and left_column is None
        and canonical_flexible_envelopes
        else 0
    )
    top_edge_keepout_clear_blocks = (
        tuple(sorted(top_edge_keepout_clear_candidates))
        if top_edge_master_keepout_nm > 0
        else ()
    )

    # ---- 4c. HARD directional order. The endpoints may
    #          be device names OR a materialized array/CC GROUP-tile name (both are
    #          members of their block, so dev_block resolves either). Horizontal
    #          pairs stay same-band; vertical pairs remain local relations and do
    #          not promote unrelated blocks into a new global band.
    order_pairs: List[Tuple[str, str]] = []
    vertical_order_pairs: List[Tuple[str, str]] = []
    for a, b, rel in getattr(intent, "order_constraints", ()) or ():
        ba = dev_block.get(str(a))
        bb = dev_block.get(str(b))
        if ba is None or bb is None:
            continue
        if ba == bb:
            source_order = (str(a), str(b), str(rel))
            if source_order not in locally_consumed_order_constraints:
                raise ValueError(
                    "same-composite order lacks a local materialization proof: "
                    f"@order({source_order[0]},{source_order[1]},{source_order[2]})"
                )
            continue
        relation = str(rel)
        if (
            relation == "left_of"
            and _horizontal_order_is_materializable(
                ba,
                bb,
                bands_map,
                tuple(relational_lateral_blocks),
            )
            and (ba, bb) not in order_pairs
        ):
            order_pairs.append((ba, bb))
        elif relation == "below" and (ba, bb) not in vertical_order_pairs:
            vertical_order_pairs.append((ba, bb))
        elif relation == "above" and (bb, ba) not in vertical_order_pairs:
            vertical_order_pairs.append((bb, ba))
    for pair in relational_order_pairs:
        if pair not in order_pairs:
            order_pairs.append(pair)
    # ---- 4d. ROUTING-OWED placement (P4): edge access (@anchor) + reserved routing
    #          channel (@channel) + shared-gate trunk alignment (@trunk). Direct
    #          canonical member anchors lower to HARD block-edge equalities. Net and
    #          group anchors retain their below-envelope cluster pull; a net anchor is
    #          materialized exactly by the router's PDK-derived edge pin. Absent intent
    #          -> empty additions (the placement is byte-identical to before). Net /
    #          gate connectivity comes from the SHARED bridged intent `bi` (no
    #          re-derive); all targets are topology names, never coordinates.
    net_devs: Dict[str, set] = {}
    dev_gate: Dict[str, str] = {}
    for d in getattr(bi, "devices", ()) or []:
        for role, net in (getattr(d, "nets", {}) or {}).items():
            if net:
                net_devs.setdefault(str(net), set()).add(str(getattr(d, "name", "")))
            if role == "gate" and net:
                dev_gate[str(getattr(d, "name", ""))] = str(net)
    for p in getattr(bi, "passives", ()) or []:
        for net in (getattr(p, "nets", {}) or {}).values():
            if net:
                net_devs.setdefault(str(net), set()).add(str(getattr(p, "name", "")))
    pg_devs = {
        str(getattr(pg, "group", "")): set(getattr(pg, "devices", ()) or ())
        for pg in place_groups
    }
    canonical_member_names = {
        str(getattr(member, "name", ""))
        for field in ("devices", "passives", "bjts")
        for member in tuple(getattr(bi, field, ()) or ())
        if str(getattr(member, "name", ""))
    }

    def _blocks_for_target(target: str) -> set:
        """Resolve an @anchor target -> the blocks that materialize it: a NET ->
        its connected devices; else a @place_group name -> its devices; else a bare
        device name. Never a coordinate."""
        devs = set(net_devs.get(target, ())) or set(pg_devs.get(target, ())) or {target}
        return {bid for bid in new_ids if block_member_devices.get(bid, set()) & devs}

    # (C1) @anchor(target, edge=right): retain the historical SOFT pull for the
    #      target cluster. A direct canonical member additionally gets a HARD exact
    #      block-edge equality. Net anchors do not hard-pin every connected member:
    #      their exact edge access belongs to the route materializer.
    anchor_right_blocks: set = set()
    anchor_left_blocks: set = set()
    hard_anchor_right_blocks: set = set()
    hard_anchor_left_blocks: set = set()
    for a in getattr(intent, "anchors", ()) or []:
        edge = str(getattr(a, "edge", "")).strip().lower()
        if edge not in {"left", "right"}:
            continue
        target = str(getattr(a, "target", ""))
        target_blocks = _blocks_for_target(target)
        for bid in target_blocks:
            if bands_map.get(bid) == _BAND_TOP and bid != left_column:
                continue
            if edge == "right":
                anchor_right_blocks.add(bid)
            else:
                anchor_left_blocks.add(bid)
        if target in canonical_member_names:
            if edge == "right":
                hard_anchor_right_blocks.update(target_blocks)
            else:
                hard_anchor_left_blocks.update(target_blocks)
    inferred_left_column = left_column
    left_column = _left_column_after_explicit_edge_anchor(
        left_column,
        tuple(sorted(hard_anchor_left_blocks)),
    )
    if inferred_left_column is not None and left_column is None:
        bottom_edge_keepout_clear_blocks = ()
        top_edge_master_keepout_nm = (
            int(top_edge_keepout_nm)
            if fixed_outline_width_nm is not None and canonical_flexible_envelopes
            else 0
        )
        top_edge_keepout_clear_blocks = (
            tuple(sorted(top_edge_keepout_clear_candidates))
            if top_edge_master_keepout_nm > 0
            else ()
        )
    # An explicit left anchor overrides the cap composite's historical default
    # right pull and any net-wide right anchor that also reaches that composite.
    left_pull = set(anchor_left_blocks)
    right_pull = (set(right_pull) | anchor_right_blocks) - left_pull

    # (C3) @channel(orient=vertical, width=W): reserve the column -> lower-region
    #      separation (a device-free vertical corridor for the cross-band M3 straps).
    channel_nm = _vertical_channel_nm(intent, channel_strategy)

    # (C2) @trunk shared-gate alignment: the clusters that SHARE a gate/bias trunk
    #      net AND live in the SAME band should sit x-adjacent so one straight
    #      horizontal M2 trunk spans them (a short, straight run + clean vertical M3
    #      drops). Derive the shared-gate block sets from the netlist gate
    #      connectivity + the declared SIGNAL @trunk nets (the edge rails vdd/gnd,
    #      declared with edge=, are excluded -- a power rail is not a gate to align),
    #      and add each same-band pair as a SOFT x-proximity pull (the same below-
    #      envelope pull the cap / @adjacent uses). Cross-band trunk nets are served
    #      by the reserved M3 channel (C3), not a placement pull.
    trunk_nets = {
        str(getattr(t, "net", ""))
        for t in (getattr(intent, "trunks", ()) or [])
        if not str(getattr(t, "edge", "")).strip()
    }
    trunk_align_pairs: List[Tuple[str, str]] = []
    for tnet in sorted(trunk_nets):
        gate_blocks = {
            dev_block[d] for d, g in dev_gate.items() if g == tnet and d in dev_block
        }
        by_band: Dict[int, List[str]] = {}
        for bid in sorted(gate_blocks):
            by_band.setdefault(bands_map.get(bid, _BAND_CENTER), []).append(bid)
        for blks in by_band.values():
            for i in range(len(blks)):
                for j in range(i + 1, len(blks)):
                    pair = (blks[i], blks[j])
                    if pair not in trunk_align_pairs:
                        trunk_align_pairs.append(pair)

    column_devices = tuple(sorted({
        member
        for group in place_groups
        if getattr(group, "col", None) is not None
        for member in tuple(getattr(group, "devices", ()) or ())
    }))
    source_floorplan_device_bboxes = (
        _source_device_bboxes_nm(geom, pum, column_devices)
        if column_devices
        else {}
    )
    solver_floorplan_device_bboxes = (
        source_floorplan_device_bboxes
        if feedback_repair or not column_devices
        else _project_source_device_bboxes_to_solver_masters(
            source_floorplan_device_bboxes,
            source_blocks=tuple(blocks),
            source_block_member_devices=source_block_member_devices,
            solver_blocks=tuple(new_blocks),
            solver_block_member_devices={
                block_id: set(block_member_devices[block_id])
                for block_id in new_ids
            },
            member_offsets=member_offsets,
        )
    )
    projected_floorplan_columns: Tuple[Tuple[str, int, int, int], ...] = ()
    projected_floorplan_groups = (
        floorplan_column_groups(
            tuple(new_blocks),
            block_member_devices,
            place_groups,
            solver_floorplan_device_bboxes,
        )
        if column_devices
        else ()
    )
    projected_consumption_certificate = None
    if intent_basis is not None:
        from clay.layout.place.intent_consumption import bind_floorplan_consumption

        projected_consumption_certificate = bind_floorplan_consumption(
            intent_basis,
            stage=(
                "exact_materialization_replay"
                if feedback_repair
                else "initial_master"
            ),
            floorplan_column_groups=projected_floorplan_groups,
        )
    projected_fact_projection = None
    if intent_fact_set is not None and source_inventory is not None:
        from clay.layout.place.intent_consumption import (
            build_placement_fact_projection,
        )

        projected_fact_projection = build_placement_fact_projection(
            consumer=(
                "placement_replay" if feedback_repair else "placement_initial"
            ),
            fact_set=intent_fact_set,
            source_inventory=source_inventory,
            place_groups=place_groups,
            route_groups=raw_route_groups,
            floorplan_column_groups=projected_floorplan_groups,
            envelope_order_edges=relational_group_edges,
            resolved_device_masters=_resolved_device_master_rows(
                block_member_devices,
                tuple(sorted(new_ids)),
            ),
            rigid_nonmos_macro_plans=rigid_nonmos_macro_plans,
            well_island_share_plans=well_island_share_plans,
            shared_well_guard_envelope_plans=(
                shared_well_guard_envelope_plans
            ),
            shared_well_guard_child_plans=shared_well_guard_child_plans,
            flexible_envelope_constraints=canonical_flexible_envelopes,
        )

    # ---- 5. band-constrained 2-D re-place (reference floorplan when a left column
    #         is identified; plain band stack otherwise). On the floorplan path the
    #         cap keeps its right-align (its @adjacent-mos pulls fight the right edge
    #         and are dropped there); the resistor pulls + trunk-align + @order apply.
    if left_column is not None:
        pull_pairs = list(res_pull_pairs) + list(trunk_align_pairs)
    else:
        pull_pairs = (
            list(cap_pull_pairs) + list(res_pull_pairs) + list(trunk_align_pairs)
        )
    overlap_compatible_pairs = (
        derived_overlap_compatible_pairs if left_column is None else ()
    )
    initial_incumbent_scope = nullcontext()
    initial_incumbent_kwargs = {}
    if generated_incumbent is not None:
        from clay.layout.place.generated_incumbent import GeneratedIncumbentSeed, scoped_generated_incumbent_frame
        if type(generated_incumbent) is not GeneratedIncumbentSeed or not incumbent_replay_enabled():
            raise ValueError("generated incumbent requires its exact native request and scoped replay mode")
        initial_owner_logicals = {
            owner: tuple(sorted(set(live_master_devices.get(owner, ())) & set(canonical_device_names)))
            for owner, _bbox in new_blocks
        }
        generated_incumbent.require_owner_partition(initial_owner_logicals)
        initial_incumbent_kwargs = {"incumbent_seed_positions": generated_incumbent.positions}
        initial_incumbent_scope = scoped_generated_incumbent_frame(
            generated_incumbent, initial_owner_logicals, phase="initial_master",
        )
    from clay.layout.place.current_owner_stage import capture_candidate_owners
    capture_candidate_owners(solver_blocks=tuple(new_blocks),
        solver_block_member_devices=block_member_devices, logical_kinds=device_kind)
    from clay.layout.place.reference_quality_geometry import (
        LiveQualityGeometryFailure,
        capture_live_quality_frame,
        wants_live_quality_geometry,
    )
    if wants_live_quality_geometry(tuple(owner for owner, _bbox in new_blocks)):
        try:
            capture_live_quality_frame(
                source_blocks=tuple(blocks), solver_blocks=tuple(new_blocks),
                source_block_member_devices=source_block_member_devices,
                solver_block_member_devices=block_member_devices,
                member_offsets=member_offsets, geometry=geom, placement_um=pum,
                logical_kinds=device_kind,
                source_device_bboxes=_source_device_bboxes_nm(geom, pum, canonical_device_names),
            )
        except Exception as exc:
            raise LiveQualityGeometryFailure("current native geometry observation failed: " + str(exc)) from exc
    from clay.layout.place.edge_rail_geometry import compile_edge_rail_reservation_from_geometry

    edge_rail_reservation = compile_edge_rail_reservation_from_geometry(
        intent=intent, tech=tech, bridged_intent=bi, cell_def=cell_def,
        row_program=row_program, row_materialization=row_materialization,
        native_geometry=native_geometry,
        solver_blocks=tuple(new_blocks), materialized_blocks=tuple(blocks),
        block_stamps=block_stamps, block_gring=block_gring,
        geometry=geom, placement_um=pum, master_members=spacing_master_members,
        member_offsets_nm=member_offsets,
        flexible_envelope_constraints=canonical_flexible_envelopes,
        bands=bands_map,
    )
    group_spacing_pairs = _admitted_group_spacing_pairs(
        edge_rail_reservation, pair_clearances_nm, native_access_pair_clearances_nm)
    if left_column is None:
        overlap_compatible_pairs = tuple(sorted(set(overlap_compatible_pairs) | set(group_spacing_pairs)))
    initial_rail_kwargs = materialized_rail_kwargs(
        edge_rail_reservation=edge_rail_reservation,
        bottom_center_keepout_nm=bottom_center_keepout_nm,
        bottom_edge_keepout_nm=bottom_edge_master_keepout_nm,
        bottom_edge_keepout_exempt=bottom_edge_keepout_exempt_blocks,
        bottom_edge_keepout_clear=bottom_edge_keepout_clear_blocks,
        top_edge_keepout_nm=top_edge_master_keepout_nm,
        top_edge_keepout_clear=top_edge_keepout_clear_blocks,
    )
    initial_search_scope = nullcontext()
    if canonical_flexible_envelopes and generated_incumbent is None and not feedback_repair:
        from clay.layout.place.solver_execution_policy import (
            AreaSearchExecutionPolicy, scoped_area_search_execution_policy,
        )
        # Preserve the objective, hard model and deterministic phase allowance.
        # Single-worker execution + pinned random seed + canonical coordinate
        # tie-breaking lock deterministic coordinates across runs on identical
        # .clay inputs; exact materialization is independently replayed below.
        initial_search_scope = scoped_area_search_execution_policy(
            AreaSearchExecutionPolicy(
                placement_objective != "area",
                60.0,
                num_search_workers=1,
                max_deterministic_time=8.0,
                random_seed=0,
                canonical_tie_break=(placement_objective == "area"),
            ),
        )
    with initial_incumbent_scope, initial_search_scope:
        placement_result = place_blocks_2d_with_certificate(
        new_blocks,
        gap_nm,
        aspect_target=aspect_target,
        max_time_s=solve_ledger.for_phase(
            "feedback_final_master" if feedback_repair else "initial_master"
        ),
        bands=bands_map,
        floorplan_column_classes=projected_floorplan_columns,
        floorplan_column_groups=projected_floorplan_groups,
        right_pull=right_pull,
        right_edge=hard_anchor_right_blocks,
        pull_pairs=pull_pairs,
        adjacency_limits_nm=adjacency_limits_nm,
        left_pull=left_pull,
        left_edge=hard_anchor_left_blocks,
        left_column=left_column,
        lateral_blocks=tuple(sorted(relational_lateral_blocks)),
        order_pairs=order_pairs,
        vertical_order_pairs=vertical_order_pairs,
        channel_nm=channel_nm,
        **initial_rail_kwargs,
        pack_mode=pack_mode,
        strict_band_order=strict_band_order,
        strict_band_members=tuple(sorted(strict_band_member_blocks)),
        pair_clearances_nm=pair_clearances_nm,
        **({"native_access_pair_clearances_nm": native_access_pair_clearances_nm}
           if native_access_pair_clearances_nm else {}),
        axis_alignment_groups=axis_alignment_groups,
        center_alignment_pairs=center_alignment_pairs,
        aggregate_envelope_orders=aggregate_envelope_orders,
        hierarchical_macro_orders=hierarchical_macro_orders,
        hierarchical_macro_alignments=hierarchical_macro_alignments,
        hierarchical_cohesion_groups=hierarchical_macro_cohesion_groups,
        solve_cuts=frozen_solve_cuts,
        solve_cut_proofs=solve_cut_proofs,
        solve_cut_source_certificate=solve_cut_source_certificate,
        aspect_bounds=aspect_bounds,
        aspect_obligation=aspect_obligation,
        manufacturing_grid_nm=manufacturing_grid_nm,
        minimum_structural_occupancy=minimum_structural_occupancy,
        placement_objective=placement_objective,
        fixed_outline_width_nm=fixed_outline_width_nm,
        area_target=area_target,
        flexible_envelope_constraints=canonical_flexible_envelopes,
        rigid_compound_footprints=(
            rigid_compound_footprints if left_column is None else ()
        ),
        layered_collision_footprints=(
            layered_collision_footprints if left_column is None else ()
        ),
        row_bounded_cavity_projections=row_bounded_cavity_projections,
        overlap_compatible_pairs=overlap_compatible_pairs,
        **(
            {
                "seed_cohesion_groups": well_share_seed_cohesion_groups,
            }
            if fixed_outline_width_nm is not None
            and well_share_seed_cohesion_groups
            else {}
        ),
        **({"mosaic_guest_slot_projections": mosaic_guest_slot_projections}
           if mosaic_guest_slot_projections else {}),
        **initial_incumbent_kwargs,
        **({"edge_rail_reservation": edge_rail_reservation}
           if edge_rail_reservation is not None else {}),
        )
    if generated_incumbent is not None:
        generated_incumbent.require_final_positions(placement_result.positions)
    if feedback_repair and not (
        placement_certificate_is_signable(placement_result.certificate)
        and placement_result.certificate.applied_cut_hashes
        == tuple(sorted(cut.cut_hash for cut in frozen_solve_cuts))
    ):
        raise PlacementSolveCutResidualError(placement_result.certificate)
    positions = placement_result.positions

    # ---- 6. distribute composite placement back to members -> absolute per-block
    abs_pos: Dict[str, Tuple[int, int]] = {}
    for bid, _bb in new_blocks:
        if bid in composite_members:
            continue
        abs_pos[bid] = positions[bid]
    for comp_id, parts in composite_members.items():
        X, Y = positions[comp_id]
        for p in parts:
            _cid, (ox, oy) = member_offsets[p]
            abs_pos[p] = (X + ox, Y + oy)

    # ---- Replay only PDK/ownership-required ring padding.  Ring frames retain
    #      their intrinsic materialized heights; no source intent requires a common
    #      top-band ceiling.  A same-net boundary ring needs no speculative rail
    #      cavity because the final rail planner checks exact M2 ownership and
    #      spacing.  Foreign/ambiguous carriers keep the compiled reservation.
    ring_pad_top_nm: Dict[str, int] = {}
    ring_pad_bottom_nm: Dict[str, int] = {}
    bottom_edge_padded_master_blocks: set[str] = set()
    if feedback_repair:
        ring_pad_top_nm = _exact_member_ring_padding_rows(
            source_member_ring_pad_top_nm,
            "source_member_ring_pad_top_nm",
        )
        ring_pad_bottom_nm = _exact_member_ring_padding_rows(
            source_member_ring_pad_bottom_nm,
            "source_member_ring_pad_bottom_nm",
        )
    if feedback_repair:
        (
            ring_pad_top_nm,
            ring_pad_bottom_nm,
            bottom_edge_padded_master_blocks,
        ) = _replay_ring_padding_on_positioned_master(
            materialized_blocks=blocks,
            master_blocks=new_blocks,
            master_members=feedback_master_members or {},
            positions=positions,
            ring_block_ids=set(block_gring),
            block_stamps=block_stamps,
            block_member_devices=source_block_member_devices,
            prior_top_padding=ring_pad_top_nm,
            prior_bottom_padding=ring_pad_bottom_nm,
            bands=bands_map,
            left_column=left_column,
            guard_rings=guard_ring_declarations,
            guard_ring_outer_clearances_nm=guard_ring_outer_clearances_nm,
            top_edge_keepout_nm=top_edge_keepout_nm,
            top_edge_rail_nets=tuple(sorted(top_edge_rail_nets)),
            bottom_edge_keepout_nm=bottom_center_keepout_nm,
            bottom_edge_rail_nets=tuple(sorted(bottom_edge_rail_nets)),
        )
    else:
        positioned_new_blocks = tuple(
            (
                block_id,
                (
                    positions[block_id][0],
                    positions[block_id][1],
                    positions[block_id][0] + (bbox[2] - bbox[0]),
                    positions[block_id][1] + (bbox[3] - bbox[1]),
                ),
            )
            for block_id, bbox in new_blocks
        )
        (
            ring_pad_top_nm,
            ring_pad_bottom_nm,
            bottom_edge_padded_master_blocks,
        ) = replay_member_ring_padding(
            blocks=positioned_new_blocks,
            block_stamps=block_stamps,
            block_member_devices=block_member_devices,
            bands=bands_map,
            left_column=left_column,
            guard_rings=guard_ring_declarations,
            guard_ring_outer_clearances_nm=guard_ring_outer_clearances_nm,
            top_edge_keepout_nm=top_edge_keepout_nm,
            top_edge_rail_nets=tuple(sorted(top_edge_rail_nets)),
            bottom_edge_keepout_nm=bottom_center_keepout_nm,
            bottom_edge_rail_nets=tuple(sorted(bottom_edge_rail_nets)),
        )

    # per-ORIGINAL-block delta from old bbox origin to new origin
    deltas: Dict[str, Tuple[int, int]] = {}
    for bid, bb in blocks:
        nx0, ny0 = abs_pos[bid]
        deltas[bid] = (nx0 - bb[0], ny0 - bb[1])

    # Hard-assert the re-placed ORIGINAL physical collision atoms do not overlap.
    # A fixed mosaic may intentionally expose an owner-envelope void; treating
    # that owner as one solid rectangle would reject a placement the solver and
    # exact cell-local geometry both proved legal.
    replaced = [
        (
            bid,
            (
                abs_pos[bid][0],
                abs_pos[bid][1],
                abs_pos[bid][0] + (bb[2] - bb[0]),
                abs_pos[bid][1] + (bb[3] - bb[1]),
            ),
        )
        for bid, bb in blocks
    ]
    source_fixed_pcell_footprints = (
        _fixed_pcell_compound_footprints_from_geometry(
            solver_blocks=tuple(replaced),
            block_stamps=block_stamps,
            geometry=geom,
            master_members={block_id: (block_id,) for block_id, _bbox in replaced},
        )
    )
    source_overlap_compatible_pairs = _layer_disjoint_overlap_compatible_pairs(
        excluded_layers=collision_excluded_layers,
        block_stamps=block_stamps,
        geometry=geom,
        master_members={block_id: (block_id,) for block_id, _bbox in replaced},
    )
    source_layered_collision_footprints = (
        _layered_collision_footprints_from_geometry(
            excluded_layers=collision_excluded_layers,
            solver_blocks=tuple(replaced),
            materialized_blocks=tuple(blocks),
            block_stamps=block_stamps,
            block_gring=block_gring,
            geometry=geom,
            placement_um=pum,
            master_members={
                block_id: (block_id,) for block_id, _bbox in replaced
            },
            component_refinement_owners=(
                collision_projection_scope.source_members
            ),
        )
    )
    _assert_no_physical_block_overlap(
        tuple(replaced),
        rigid_compound_footprints=source_fixed_pcell_footprints,
        layered_collision_footprints=source_layered_collision_footprints,
        overlap_compatible_pairs=source_overlap_compatible_pairs,
        overlap_compatible_pair_layers=_same_bulk_mergeable_pair_layers(
            block_member_devices=source_block_member_devices,
            device_nets=device_nets,
            tech=tech,
        ),
    )

    # Shift stamps and rebuild rings through the shared canonical materializer.
    compact_um, compact_geometry = apply_block_deltas(
        geom,
        pum,
        intent,
        tech,
        tech_id,
        deltas,
        block_stamps,
        block_gring,
        gring_names,
        bi,
        member_ring_pad_top_nm=(ring_pad_top_nm or None),
        member_ring_pad_bottom_nm=(ring_pad_bottom_nm or None),
    )
    (
        materialized_blocks,
        materialized_stamps,
        materialized_rings,
        materialized_ring_names,
        materialized_layered_replay,
        materialized_layered_projection,
    ) = current_blocks(compact_geometry, compact_um)
    if alignment_native_feol_pairs:
        # The alignment pre-bundle can merge two original row masters into one
        # solver master. Re-bind the actual post-placement stamps to the
        # original physical blocks and recheck their pair predicates here;
        # the ordinary final rail IR intentionally excludes same-master pairs.
        from clay.layout.place.edge_rail_geometry import _bind_inventory
        from clay.layout.place.native_feol_reservation import (
            compile_sky130_native_feol_pairs, native_feol_rows_satisfied,
            sky130_native_row_kinds,
        )
        from clay.layout.place.constraints_spatial import _seed_geometry_legal
        actual_blocks=dict(materialized_blocks)
        master_pairs=tuple(sorted({tuple(sorted((row.left_master,row.right_master)))
                                   for row in alignment_native_feol_pairs}))
        relevant={name for pair in master_pairs for name in pair}
        if not relevant <= set(actual_blocks):
            raise ValueError('native row abut final source block ownership changed')
        actual_stamps,_ = _bind_inventory(
            solver_blocks=tuple(materialized_blocks), materialized_blocks=tuple(materialized_blocks),
            block_stamps=materialized_stamps, block_gring=materialized_rings,
            geometry=compact_geometry, placement_um=compact_um,
            master_members={name:(name,) for name,_bbox in materialized_blocks},
            member_offsets_nm={}, relevant_layers=frozenset(), conductors=())
        actual_kinds=sky130_native_row_kinds(bundle=native_geometry, cell=cell_def,
            stamps=actual_stamps, geometry=compact_geometry, row_program=row_program,
            row_materialization=row_materialization, tech=tech)
        actual_feol, actual_clearances=compile_sky130_native_feol_pairs(
            bundle=native_geometry, cell=cell_def, stamps=actual_stamps,
            geometry=compact_geometry, placement_um=compact_um,
            eligible_master_pairs=master_pairs, row_kinds=actual_kinds, row_program=row_program,
            row_materialization=row_materialization, tech=tech)
        if actual_feol != alignment_native_feol_pairs or actual_clearances != alignment_native_feol_clearances:
            raise ValueError('native row abut final FEOL authority or artwork changed')
        actual_positions={name:actual_blocks[name][:2] for name in relevant}
        if not native_feol_rows_satisfied(actual_feol, actual_positions):
            raise ValueError('native row abut final FEOL geometry failed replay')
        actual_layers=_layered_collision_footprints_from_geometry(
            excluded_layers=collision_excluded_layers,
            solver_blocks=tuple(materialized_blocks), materialized_blocks=tuple(materialized_blocks),
            block_stamps=materialized_stamps, block_gring=materialized_rings,
            geometry=compact_geometry, placement_um=compact_um,
            master_members={name:(name,) for name,_bbox in materialized_blocks})
        layer_by_owner=dict(actual_layers)
        if not relevant <= set(layer_by_owner):
            raise ValueError('native row abut final layered artwork is incomplete')
        layer_clearances=_pdk_layer_clearances_nm(tech)
        for pair in master_pairs:
            pair_dims={name:(actual_blocks[name][2]-actual_blocks[name][0],
                             actual_blocks[name][3]-actual_blocks[name][1]) for name in pair}
            if not _seed_geometry_legal({name:actual_positions[name] for name in pair},
                    dims=pair_dims,gap_nm=gap_nm,pair_clearances_nm={},
                    layered_collision_footprints=tuple((name,layer_by_owner[name]) for name in pair),
                    layer_clearances_nm=layer_clearances,
                    pair_layer_clearances_nm=actual_clearances):
                raise ValueError('native row abut final same-layer geometry failed replay')
    master_members = (
        feedback_master_members
        if feedback_master_members is not None
        else {
            block_id: tuple(composite_members.get(block_id, (block_id,)))
            for block_id, _bbox in new_blocks
        }
    )
    replay_native_access_pair_clearances_nm = _project_native_access_pair_clearances_to_masters(
        _mim_native_access_pair_clearances_nm(
            tech, block_stamps=materialized_stamps, geometry=compact_geometry,
            base_gap_nm=gap_nm, device_nets=device_nets,
            rail_specs=_compiled_edge_rail_specs_for_placement(intent, tech, bi),
            placement_um=compact_um,
        ),
        master_members=master_members, base_gap_nm=gap_nm,
    )
    materialized_overlap_compatible_pairs = tuple(
        pair
        for pair in _layer_disjoint_overlap_compatible_pairs(
            excluded_layers=collision_excluded_layers,
            block_stamps=materialized_stamps,
            geometry=compact_geometry,
            master_members=master_members,
        )
        if pair not in pair_clearances_nm and pair not in replay_native_access_pair_clearances_nm
    )
    if group_spacing_pairs:
        group_replay = compile_edge_rail_reservation_from_geometry(
            intent=intent, tech=tech, bridged_intent=bi, cell_def=cell_def,
            row_program=row_program, row_materialization=row_materialization, native_geometry=native_geometry,
            solver_blocks=placement_result.certificate.blocks, materialized_blocks=tuple(materialized_blocks),
            block_stamps=materialized_stamps, block_gring=materialized_rings,
            geometry=compact_geometry, placement_um=compact_um, master_members=master_members,
            member_offsets_nm={}, flexible_envelope_constraints=canonical_flexible_envelopes,
            bands=bands_map)
        if _admitted_group_spacing_pairs(group_replay, pair_clearances_nm,
                                         replay_native_access_pair_clearances_nm) != group_spacing_pairs:
            raise ValueError('native group source spacing changed across materialization')
        materialized_overlap_compatible_pairs = tuple(sorted(set(materialized_overlap_compatible_pairs) | set(group_spacing_pairs)))
    materialized_overlap_compatible_pairs = (
        materialized_overlap_compatible_pairs if left_column is None else ()
    )
    if materialized_overlap_compatible_pairs != overlap_compatible_pairs:
        raise ValueError(
            "layer-disjoint overlap compatibility changed across materialization"
        )
    materialized_collision_footprints = (
        _fixed_pcell_compound_footprints_from_geometry(
            solver_blocks=tuple(materialized_blocks),
            block_stamps=materialized_stamps,
            geometry=compact_geometry,
            master_members={
                block_id: (block_id,) for block_id, _bbox in materialized_blocks
            },
        )
    )
    materialized_layered_collision_footprints = (
        _layered_collision_footprints_from_geometry(
            excluded_layers=collision_excluded_layers,
            solver_blocks=placement_result.certificate.blocks,
            materialized_blocks=tuple(materialized_blocks),
            block_stamps=materialized_stamps,
            block_gring=materialized_rings,
            geometry=compact_geometry,
            placement_um=compact_um,
            master_members=master_members,
            component_refinement_owners=collision_projection_scope.master_owners,
            flexible_envelope_constraints=canonical_flexible_envelopes,
        )
        if left_column is None
        else ()
    )

    if feedback_repair:
        materialized_source_ownership, materialized_source_geometry = materialized_source_snapshot(
            required=bool(layered_placement_assemblies) or _materialized_source_ownership_requested(
                group_route_requested, intent_fact_set), blocks=materialized_blocks,
            stamps=materialized_stamps, geometry=compact_geometry, placement_um=compact_um,
            canonical_device_names=canonical_device_names)
        final_certificate = _bind_certificate_to_materialized_blocks(
            placement_result.certificate,
            materialized_blocks=materialized_blocks,
            master_members=master_members,
            materialized_block_source_members=materialized_source_ownership,
            materialized_source_member_bboxes=materialized_source_geometry,
            materialized_collision_footprints=materialized_collision_footprints,
            layered_collision_footprints=(
                materialized_layered_collision_footprints
            ),
            overlap_compatible_pairs=materialized_overlap_compatible_pairs,
            gap_nm=gap_nm,
            pair_clearances_nm=pair_clearances_nm,
            pair_clearance_scope="solver",
            solve_cuts=frozen_solve_cuts,
            placement_declaration_refs=placement_declaration_refs,
            aspect_obligation=aspect_obligation,
            layered_placement_assemblies=layered_placement_assemblies,
            layered_placement_replay=materialized_layered_replay,
            layered_placement_projection=materialized_layered_projection,
        )
        intent_consumption_certificates: Tuple[
            IntentConsumptionCertificate, ...
        ] = ()
        if intent_basis is not None:
            from clay.layout.place.intent_consumption import require_consumer_parity

            if projected_consumption_certificate is None:
                raise ValueError("feedback replay lacks its intent consumption")
            prior_consumption = getattr(
                solve_cut_source_certificate,
                "intent_consumption_certificates",
                (),
            )
            if type(prior_consumption) is not tuple:
                raise TypeError(
                    "feedback source intent consumption must be an exact tuple"
                )
            prior_initial = tuple(
                row for row in prior_consumption if row.stage == "initial_master"
            )
            intent_consumption_certificates = tuple(
                sorted(
                    (*prior_initial, projected_consumption_certificate),
                    key=lambda row: row.stage,
                )
            )
            require_consumer_parity(
                intent_basis,
                intent_consumption_certificates,
            )
            final_certificate = replace(
                final_certificate,
                intent_consumption_certificates=intent_consumption_certificates,
            )
        fact_projections: Tuple[ConsumerProjection, ...] = ()
        if intent_fact_set is not None:
            if projected_fact_projection is None:
                raise ValueError("feedback replay lacks its fact projection")
            prior_projections = getattr(
                solve_cut_source_certificate,
                "fact_projections",
                (),
            )
            if type(prior_projections) is not tuple:
                raise TypeError("feedback source fact projections must be exact")
            prior_initial = tuple(
                row
                for row in prior_projections
                if type(row) is ConsumerProjection
                and row.consumer == "placement_initial"
            )
            if len(prior_initial) != 1:
                raise ValueError("feedback source lacks one initial fact projection")
            fact_projections = (prior_initial[0], projected_fact_projection)
            final_certificate = replace(
                final_certificate,
                fact_projections=fact_projections,
            )
        if mos_matching_devices:
            matching_ownership, matching_deltas = (
                _matching_axis_master_authority(
                    matching_pair_axes=legacy_matching_pair_axes,
                    device_kind=device_kind,
                    live_master_ids=tuple(sorted(master_members)),
                    block_member_devices=block_member_devices,
                    master_members=master_members,
                    member_deltas_nm=deltas,
                )
            )
            final_certificate = replace(
                final_certificate,
                matching_pair_axis_projections=(
                    _project_mos_matching_axes_through_block_deltas(
                        matching_pair_axes=legacy_matching_pair_axes,
                        source_device_bboxes=source_mos_matching_bboxes,
                        final_device_bboxes=_source_device_bboxes_nm(
                            compact_geometry,
                            compact_um,
                            mos_matching_devices,
                        ),
                        source_block_member_devices=matching_ownership,
                        device_kind=device_kind,
                        block_deltas_nm=matching_deltas,
                    )
                ),
            )
        if cross_inventory is not None:
            from clay.layout.place.cross_row_matching import native_cross_row_final_proof, native_cross_row_stamp_centres
            final_centres = native_cross_row_stamp_centres(
                cross_inventory, row_program, row_materialization, compact_geometry, compact_um)
            final_certificate = replace(final_certificate, native_cross_row_final_proof=
                native_cross_row_final_proof(cross_inventory, native_cross_row_cp_proof, cross_directions,
                    final_centres, manufacturing_grid_nm, row_program=row_program,
                    row_materialization=row_materialization, tech=tech))
        if same_inventory is not None:
            from clay.layout.place.native_same_row_matching import native_same_row_final_proof, native_same_row_stamp_centres
            final_centres = native_same_row_stamp_centres(
                same_inventory, row_program, row_materialization, compact_geometry, compact_um)
            final_certificate = replace(final_certificate, native_same_row_final_proof=
                native_same_row_final_proof(same_inventory, same_initial_centres, same_initial_axes,
                                            final_centres, manufacturing_grid_nm, logical_axes=same_logical_axes))
        _assert_materialized_diff_axis_certificate(
            final_certificate,
            block_member_devices,
            compact_geometry,
            compact_um,
            intent,
            bi,
            row_program=row_program, row_materialization=row_materialization,
            tech=tech, source_cell=cell_def,
            native_left_alignment_declarations=deferred_native_left_alignments,
        )
        final_certificate = replace(
            final_certificate,
            reference_conditioned_placement_obligations=(
                reference_conditioned_placement_obligations
            ),
            collision_exclusion_pdk_id=tech_id if collision_excluded_layers else '',
            collision_excluded_layers=collision_excluded_layers,
        )
        _replay_cross_row_bottom_alignments(deferred_cross_row_alignments, compact_geometry, compact_um)
        _replay_native_left_alignments(deferred_native_left_alignments, row_program, compact_geometry, compact_um)
        return RepositionPum2DResult(
            placement_um=compact_um,
            geometry=compact_geometry,
            final_placement_certificate=final_certificate,
            member_ring_pad_top_nm=tuple(sorted(ring_pad_top_nm.items())),
            member_ring_pad_bottom_nm=tuple(sorted(ring_pad_bottom_nm.items())),
            intent_consumption_certificates=intent_consumption_certificates,
            fact_projections=fact_projections,
        )

    # Ring re-materialization is allowed to change a rigid block's exact bbox
    # (notably when top-band padding changes a guard ring's protected inner
    # height).  The first solve therefore cannot certify the geometry that now
    # exists.  Replay the SAME final master once with these exact post-
    # materialization sizes, then re-materialize every ring at the replayed
    # member positions.  This is the same operation route_cell will perform
    # after stripping the temporary placement rings, so the certificate binds
    # the sole materializer's exact bbox rather than a rigidly translated
    # approximation of its prior artwork.
    # This is a placement legality replay, not a router-side legalizer: all
    # original band/order/channel constraints remain in the one CP master.
    materialized_by_name = dict(_exact_cut_input_blocks(materialized_blocks))
    flexible_member_names = {
        member
        for constraint in canonical_flexible_envelopes
        for member in constraint.members
    }
    solved_content_by_name = dict(placement_result.certificate.blocks)
    materialized_master_blocks: List[Tuple[str, _Bbox]] = []
    materialized_master_bbox: Dict[str, _Bbox] = {}
    for master, members in sorted(master_members.items()):
        if master in flexible_member_names:
            if master not in solved_content_by_name:
                raise ValueError(
                    "flexible envelope member changed final-master ownership"
                )
            solved_bbox = solved_content_by_name[master]
            constraint = next(
                row
                for row in canonical_flexible_envelopes
                if master in row.members
            )
            solved_width = solved_bbox[2] - solved_bbox[0]
            solved_height = solved_bbox[3] - solved_bbox[1]
            full_master_authority = dict(
                constraint.member_physical_offsets_nm
            )[master] == (0, 0, solved_width, solved_height)
            if full_master_authority:
                boxes = [materialized_by_name[member] for member in members]
                bbox = (
                    min(box[0] for box in boxes),
                    min(box[1] for box in boxes),
                    max(box[2] for box in boxes),
                    max(box[3] for box in boxes),
                )
                if (
                    bbox[2] - bbox[0] != solved_width
                    or bbox[3] - bbox[1] != solved_height
                ):
                    raise ValueError(
                        "flexible full-master envelope changed exact dimensions"
                    )
            else:
                if tuple(members) != (master,):
                    raise ValueError(
                        "virtualized flexible member changed final-master ownership"
                    )
                bbox = solved_bbox
        else:
            boxes = [materialized_by_name[member] for member in members]
            bbox = (
                min(box[0] for box in boxes),
                min(box[1] for box in boxes),
                max(box[2] for box in boxes),
                max(box[3] for box in boxes),
            )
        materialized_master_blocks.append((master, bbox))
        materialized_master_bbox[master] = bbox
    replay_axis_alignment_groups = _diff_pair_axis_alignment_groups(
        materialized_master_blocks,
        block_member_devices,
        compact_geometry,
        compact_um,
        intent,
        bi,
    )
    if getattr(intent, 'channel_alignments', ()):
        from clay.layout.joint_placement_terminals import build_placement_terminal_face
        build_placement_terminal_face(tech=tech, geometry=compact_geometry, placement_um=compact_um,
            row_program=row_program, row_materialization=row_materialization,
            intent=intent, source_cell=cell_def)
    replay_axis_alignment_groups = tuple(sorted(set((*replay_axis_alignment_groups,
        *_source_channel_alignment_groups(materialized_master_blocks, block_member_devices,
                                          compact_geometry, compact_um, intent)))))
    replay_native_left_axis_groups = _native_left_alignment_axis_groups(
        deferred_native_left_alignments, row_program, materialized_master_blocks,
        block_member_devices, compact_geometry, compact_um)
    if (tuple(tuple(name for name, _ in group) for group in replay_native_left_axis_groups)
            != tuple(tuple(name for name, _ in group) for group in native_left_axis_groups)):
        raise ValueError('native offset alignment master ownership changed across replay')
    replay_axis_alignment_groups = tuple(sorted(set((*replay_axis_alignment_groups, *replay_native_left_axis_groups))))
    adjacency_scope_master_blocks = tuple(
        (
            master,
            (
                min(materialized_by_name[member][0] for member in members),
                min(materialized_by_name[member][1] for member in members),
                max(materialized_by_name[member][2] for member in members),
                max(materialized_by_name[member][3] for member in members),
            ),
        )
        for master, members in sorted(master_members.items())
    )
    replay_adjacency_projection_evidence = (
        [] if generated_incumbent is not None and adjacency_limits_nm else None
    )
    replay_adjacency_scopes = (
        _materialized_adjacency_scope_bboxes_nm(
            master_blocks=adjacency_scope_master_blocks,
            master_members=master_members,
            materialized_stamps=materialized_stamps,
            materialized_rings=materialized_rings,
            geometry=compact_geometry,
            placement_um=compact_um,
            tech=tech,
            source_members=tuple(sorted({
                str(member.name)
                for field in ("devices", "passives", "bjts")
                for member in tuple(getattr(bi, field, ()) or ())
            })),
            rigid_nonmos_macro_plans=rigid_nonmos_macro_plans,
            **({
                "projection_evidence_sink": replay_adjacency_projection_evidence.append,
                "projection_collision_inputs": {
                    "materialized_blocks": tuple(materialized_blocks),
                    "component_refinement_owners": collision_projection_scope.master_owners,
                },
            } if replay_adjacency_projection_evidence is not None else {}),
        )
        if adjacency_limits_nm
        else ()
    )
    replay_adjacency_projection = None
    if replay_adjacency_projection_evidence is not None:
        if len(replay_adjacency_projection_evidence) != 1:
            raise ValueError("exact adjacency projection lacks its single native observation")
        replay_adjacency_projection = replay_adjacency_projection_evidence[0]
    replay_bottom_edge_master_keepout_nm = int(bottom_center_keepout_nm)
    replay_bottom_edge_keepout_clear_blocks = (
        tuple(
            sorted(
                set(
                    _bottom_edge_keepout_clear_blocks(
                        intent,
                        tech,
                        bi,
                        geometry=compact_geometry,
                        block_stamps=materialized_stamps,
                        block_gring=materialized_rings,
                        master_members=master_members,
                        bands=bands_map,
                    )
                )
                - set(bottom_edge_padded_master_blocks)
            )
        )
        if replay_bottom_edge_master_keepout_nm > 0 and left_column is not None
        else ()
    )
    replay_top_edge_keepout_clear_blocks = (
        _top_edge_keepout_clear_blocks(
            intent,
            tech,
            bi,
            geometry=compact_geometry,
            block_stamps=materialized_stamps,
            block_gring=materialized_rings,
            master_members=master_members,
            bands=bands_map,
        )
        if top_edge_master_keepout_nm > 0
        else ()
    )
    materialized_member_devices = _resolve_block_member_devices(
        materialized_stamps, compact_geometry
    )
    replay_master_devices = {
        master: set().union(
            *(materialized_member_devices.get(member, set()) for member in members)
        )
        for master, members in master_members.items()
    }
    replay_center_alignment_pairs = _mosaic_center_alignment_pairs(
        intent,
        replay_master_devices,
    )
    if cross_inventory is not None:
        replay_cross_owners = native_cross_row_solver_owners(
            cross_inventory, row_program, replay_master_devices, master_members,
            materialized_stamps, canonical_device_names)
        if replay_cross_owners != cross_solver_owners:
            raise ValueError("native cross-row replay master ownership changed")
        for row, members in cross_inventory.row_source_members:
            bbox = materialized_master_bbox[cross_solver_owners[row]]
            source_box = next(item.local_bbox_nm for item in row_materialization.realizations if item.entry_id == row)
            if (bbox[2] - bbox[0], bbox[3] - bbox[1]) != (
                    source_box[2] - source_box[0], source_box[3] - source_box[1]):
                raise ValueError("native cross-row replay master does not exactly frame row")
        replay_center_alignment_pairs = tuple(sorted(set((*replay_center_alignment_pairs,
                                                          *cross_center_alignment_pairs))))
    if replay_center_alignment_pairs != center_alignment_pairs:
        raise ValueError(
            "mosaic center alignment ownership changed across exact materialization replay"
        )
    replay_aggregate_envelope_orders = _aggregate_envelope_orders_from_intent(
        intent,
        replay_master_devices,
        known_device_names=canonical_device_names,
        composite_alias_members={name: tuple(stamp.composite_members)
                                 for name, stamp in geom.items()
                                 if getattr(stamp, "composite_members", ())},
    )
    if replay_aggregate_envelope_orders != aggregate_envelope_orders:
        raise ValueError(
            "aggregate envelope order ownership changed across exact materialization replay"
        )
    (
        replay_hierarchical_macro_orders,
        replay_hierarchical_macro_alignments,
    ) = _hierarchical_macro_constraints_from_intent(
        intent,
        replay_master_devices,
        known_device_names=canonical_device_names,
        tech=tech,
        cell_def=cell_def,
        base_gap_nm=gap_nm,
    )
    replay_hierarchical_macro_cohesion_groups = (
        _hierarchical_macro_cohesion_groups_from_intent(
            intent,
            replay_master_devices,
            known_device_names=canonical_device_names,
        )
    )
    if (
        replay_hierarchical_macro_orders != hierarchical_macro_orders
        or replay_hierarchical_macro_alignments
        != hierarchical_macro_alignments
        or replay_hierarchical_macro_cohesion_groups
        != hierarchical_macro_cohesion_groups
    ):
        raise ValueError(
            "hierarchical macro ownership changed across exact materialization replay"
        )
    replay_route_group_envelopes = (
        build_explicit_route_group_envelope_constraints(
            intent,
            blocks=tuple(sorted(materialized_master_blocks)),
            block_member_devices={
                master: set(replay_master_devices[master])
                for master in sorted(replay_master_devices)
            },
            known_device_names=canonical_device_names,
        )
    )
    if replay_route_group_envelopes != route_group_envelopes:
        raise ValueError(
            "explicit route-group envelope ownership changed across exact "
            "materialization replay"
        )
    replay_well_share_seed_cohesion_groups = _well_share_ring_cohesion_groups(
        intent,
        tech,
        block_ids=tuple(sorted(replay_master_devices)),
        block_member_devices=replay_master_devices,
        known_device_names=canonical_device_names,
        well_island_share_plans=well_island_share_plans,
    )
    if (
        replay_well_share_seed_cohesion_groups
        != well_share_seed_cohesion_groups
    ):
        raise ValueError(
            "shared well ring ownership changed across exact materialization replay"
        )
    replay_floorplan_columns: Tuple[Tuple[str, int, int, int], ...] = ()
    replay_floorplan_groups = (
        floorplan_column_groups(
            tuple(materialized_master_blocks),
            replay_master_devices,
            place_groups,
            _source_device_bboxes_nm(
                compact_geometry,
                compact_um,
                column_devices,
            ),
        )
        if column_devices
        else ()
    )
    intent_consumption_certificates = ()
    if intent_basis is not None:
        from clay.layout.place.intent_consumption import (
            bind_floorplan_consumption,
            require_consumer_parity,
        )

        if projected_consumption_certificate is None:
            raise ValueError("initial master lacks its intent consumption")
        replay_consumption_certificate = bind_floorplan_consumption(
            intent_basis,
            stage="exact_materialization_replay",
            floorplan_column_groups=replay_floorplan_groups,
        )
        intent_consumption_certificates = tuple(
            sorted(
                (
                    projected_consumption_certificate,
                    replay_consumption_certificate,
                ),
                key=lambda row: row.stage,
            )
        )
        require_consumer_parity(
            intent_basis,
            intent_consumption_certificates,
        )
    fact_projections: Tuple[ConsumerProjection, ...] = ()
    if intent_fact_set is not None and source_inventory is not None:
        from clay.layout.place.intent_consumption import (
            build_placement_fact_projection,
        )

        if projected_fact_projection is None:
            raise ValueError("initial master lacks its fact projection")
        replay_fact_projection = build_placement_fact_projection(
            consumer="placement_replay",
            fact_set=intent_fact_set,
            source_inventory=source_inventory,
            place_groups=place_groups,
            route_groups=raw_route_groups,
            floorplan_column_groups=replay_floorplan_groups,
            envelope_order_edges=relational_group_edges,
            resolved_device_masters=_resolved_device_master_rows(
                replay_master_devices,
                tuple(sorted(replay_master_devices)),
            ),
            rigid_nonmos_macro_plans=rigid_nonmos_macro_plans,
            well_island_share_plans=well_island_share_plans,
            shared_well_guard_envelope_plans=(
                shared_well_guard_envelope_plans
            ),
            shared_well_guard_child_plans=shared_well_guard_child_plans,
            flexible_envelope_constraints=canonical_flexible_envelopes,
        )
        fact_projections = (
            projected_fact_projection,
            replay_fact_projection,
        )
        if (
            projected_fact_projection.fact_ids
            != replay_fact_projection.fact_ids
            or projected_fact_projection.normalized_projection_sha256
            != replay_fact_projection.normalized_projection_sha256
        ):
            raise ValueError("placement fact projection changed across replay")
    replay_edge_rail_reservation = compile_edge_rail_reservation_from_geometry(
        intent=intent, tech=tech, bridged_intent=bi, cell_def=cell_def,
        row_program=row_program, row_materialization=row_materialization,
        native_geometry=native_geometry,
        solver_blocks=tuple(materialized_master_blocks), materialized_blocks=tuple(materialized_blocks),
        block_stamps=materialized_stamps, block_gring=materialized_rings,
        geometry=compact_geometry, placement_um=compact_um, master_members=master_members,
        member_offsets_nm={},
        flexible_envelope_constraints=canonical_flexible_envelopes,
        bands=bands_map,
    )
    replay_rail_kwargs = materialized_rail_kwargs(
        edge_rail_reservation=replay_edge_rail_reservation,
        bottom_center_keepout_nm=bottom_center_keepout_nm,
        bottom_edge_keepout_nm=replay_bottom_edge_master_keepout_nm,
        bottom_edge_keepout_exempt=tuple(sorted(bottom_edge_padded_master_blocks)),
        bottom_edge_keepout_clear=replay_bottom_edge_keepout_clear_blocks,
        top_edge_keepout_nm=top_edge_master_keepout_nm,
        top_edge_keepout_clear=replay_top_edge_keepout_clear_blocks,
    )
    replay_incumbent_seed_positions = materialized_replay_seed_positions(
        materialized_master_blocks, bands_map, replay_rail_kwargs,
        flexible_envelope_constraints=canonical_flexible_envelopes,
    )
    replay_member_rigid_compound_footprints = (
        _rigid_compound_footprints_from_members(
            solver_blocks=tuple(materialized_master_blocks),
            materialized_blocks=tuple(materialized_blocks),
            master_members=master_members,
            compound_owner_ids=rigid_compound_owner_ids,
        )
    )
    replay_fixed_pcell_compound_footprints = (
        _fixed_pcell_compound_footprints_from_geometry(
            solver_blocks=tuple(materialized_master_blocks),
            block_stamps=materialized_stamps,
            geometry=compact_geometry,
            master_members=master_members,
        )
    )
    replay_fixed_pcell_compound_owners = {
        owner for owner, _rectangles in replay_fixed_pcell_compound_footprints
    }
    replay_rigid_compound_footprints = _canonical_rigid_compound_footprints(
        tuple(sorted((
            *(
                row
                for row in replay_member_rigid_compound_footprints
                if row[0] not in replay_fixed_pcell_compound_owners
            ),
            *replay_fixed_pcell_compound_footprints,
        ))),
        dims={
            owner: (bbox[2] - bbox[0], bbox[3] - bbox[1])
            for owner, bbox in materialized_master_blocks
        },
    )
    replay_layered_collision_footprints = (
        _layered_collision_footprints_from_geometry(
            excluded_layers=collision_excluded_layers,
            solver_blocks=tuple(materialized_master_blocks),
            materialized_blocks=tuple(materialized_blocks),
            block_stamps=materialized_stamps,
            block_gring=materialized_rings,
            geometry=compact_geometry,
            placement_um=compact_um,
            master_members=master_members,
            component_refinement_owners=collision_projection_scope.master_owners,
            flexible_envelope_constraints=canonical_flexible_envelopes,
        )
    )
    replay_row_bounded_cavity_projections = (
        _reference_conditioned_row_bounded_cavities_from_intent(
            intent,
            {
                master: set(block_member_devices.get(master, ()))
                for master, _bbox in materialized_master_blocks
            },
            rigid_compound_footprints=replay_rigid_compound_footprints,
            dims={
                owner: (bbox[2] - bbox[0], bbox[3] - bbox[1])
                for owner, bbox in materialized_master_blocks
            },
        )
    )
    replay_mosaic_guest_slot_projections = build_mosaic_guest_slot_projections(
        intent, solver_blocks=tuple(materialized_master_blocks), block_stamps=materialized_stamps,
        master_members=master_members,
        owner_logicals={master: tuple(sorted(set(block_member_devices.get(master, ())) & set(canonical_device_names)))
                        for master, _bbox in materialized_master_blocks},
        geometry=compact_geometry, placement_um=compact_um, minimum_clearance_nm=gap_nm, tech=tech,
    )
    exact_incumbent_scope = nullcontext()
    if canonical_flexible_envelopes and generated_incumbent is None:
        # The first CP solve has placed the content masters. Rebuilding their
        # enclosing rings does not create a new positioning problem. Certify
        # the freshly measured incumbent in the complete original native CP
        # and hard geometry audit, rather than asking search to rediscover it.
        exact_incumbent_scope = scoped_incumbent_replay(indexed_geometry=True)
    if generated_incumbent is not None:
        replay_owner_logicals = {
            owner: tuple(sorted(set(replay_master_devices.get(owner, ())) & set(canonical_device_names)))
            for owner, _bbox in materialized_master_blocks
        }
        generated_incumbent.require_owner_partition(replay_owner_logicals)
        # First rollout deliberately requires unchanged physical frame. The
        # exact seed and geometry above were freshly measured, never copied.
        generated_incumbent.require_final_positions(replay_incumbent_seed_positions)
        exact_incumbent_scope = scoped_generated_incumbent_frame(
            generated_incumbent, replay_owner_logicals, phase="exact_materialization_replay",
            adjacency_projection=replay_adjacency_projection,
        )
    _replay_live_quality = wants_live_quality_geometry(
        tuple(owner for owner, _bbox in materialized_master_blocks)
    )
    if _replay_live_quality:
        capture_live_quality_frame(
            source_blocks=tuple(materialized_blocks), solver_blocks=tuple(materialized_master_blocks),
            source_block_member_devices=materialized_member_devices,
            solver_block_member_devices=replay_master_devices,
            member_offsets={member: (master, (materialized_by_name[member][0]-materialized_master_bbox[master][0],
                materialized_by_name[member][1]-materialized_master_bbox[master][1]))
                for master, members in master_members.items() for member in members if member != master},
            geometry=compact_geometry, placement_um=compact_um, logical_kinds=device_kind,
            source_device_bboxes=_source_device_bboxes_nm(compact_geometry, compact_um, canonical_device_names))
    _replay_graph = placement_result.certificate.full2d_compaction_graph
    _can_reuse_proven_optimal_replay = (
        os.environ.get("CLAY_EXACT_OPTIMAL_REPLAY_REUSE", "1") != "0"
        and generated_incumbent is None
        and not incumbent_replay_enabled()
        and not feedback_repair
        and not frozen_solve_cuts
        and not canonical_flexible_envelopes
        and not _replay_live_quality
        and _place_blocks_2d_with_alignment_scale_source is _DEFAULT_REPLAY_PLACER
        and placement_result.certificate.status == "OPTIMAL"
        and placement_result.certificate.primary_status in (None, "OPTIMAL")
        and not placement_result.certificate.used_fallback
        and replay_incumbent_seed_positions == placement_result.positions
        and tuple(sorted(placement_result.certificate.blocks))
        == tuple(sorted(materialized_master_blocks))
        and tuple(sorted((k, (b[2] - b[0], b[3] - b[1])) for k, b in materialized_master_blocks))
        == tuple(sorted((k, (b[2] - b[0], b[3] - b[1])) for k, b in new_blocks))
        and replay_floorplan_columns == projected_floorplan_columns
        and replay_floorplan_groups == projected_floorplan_groups
        and replay_rail_kwargs == initial_rail_kwargs
        and replay_native_access_pair_clearances_nm == native_access_pair_clearances_nm
        and replay_axis_alignment_groups == axis_alignment_groups
        and replay_center_alignment_pairs == center_alignment_pairs
        and replay_aggregate_envelope_orders == aggregate_envelope_orders
        and replay_hierarchical_macro_orders == hierarchical_macro_orders
        and replay_hierarchical_macro_alignments == hierarchical_macro_alignments
        and replay_hierarchical_macro_cohesion_groups == hierarchical_macro_cohesion_groups
        and (replay_rigid_compound_footprints if left_column is None else ())
        == (rigid_compound_footprints if left_column is None else ())
        and (replay_layered_collision_footprints if left_column is None else ())
        == (layered_collision_footprints if left_column is None else ())
        and replay_row_bounded_cavity_projections == row_bounded_cavity_projections
        and replay_mosaic_guest_slot_projections == mosaic_guest_slot_projections
        and replay_edge_rail_reservation == edge_rail_reservation
        and materialized_overlap_compatible_pairs == overlap_compatible_pairs
        and not replay_adjacency_scopes
        and replay_well_share_seed_cohesion_groups == well_share_seed_cohesion_groups
        and (
            _replay_graph is None
            or _replay_graph.replay(dict(materialized_master_blocks))
        )
    )
    if _can_reuse_proven_optimal_replay:
        _ = solve_ledger.for_phase("exact_materialization_replay")
        replay_result = placement_result
    else:
        with exact_incumbent_scope:
            replay_result = _place_blocks_2d_with_alignment_scale_source(
            placement_result.certificate,
            materialized_master_blocks,
            gap_nm,
            aspect_target=aspect_target,
            max_time_s=solve_ledger.for_phase("exact_materialization_replay"),
            bands=bands_map,
            floorplan_column_classes=replay_floorplan_columns,
            floorplan_column_groups=replay_floorplan_groups,
            right_pull=right_pull,
            right_edge=hard_anchor_right_blocks,
            left_pull=left_pull,
            left_edge=hard_anchor_left_blocks,
            pull_pairs=pull_pairs,
            adjacency_limits_nm=adjacency_limits_nm,
            adjacency_scope_bboxes_nm=replay_adjacency_scopes,
            left_column=left_column,
            lateral_blocks=tuple(sorted(relational_lateral_blocks)),
            order_pairs=order_pairs,
            vertical_order_pairs=vertical_order_pairs,
            channel_nm=channel_nm,
            **replay_rail_kwargs,
            pack_mode=pack_mode,
            strict_band_order=strict_band_order,
            strict_band_members=tuple(sorted(strict_band_member_blocks)),
            pair_clearances_nm=pair_clearances_nm,
            **({"native_access_pair_clearances_nm": replay_native_access_pair_clearances_nm}
               if replay_native_access_pair_clearances_nm else {}),
            axis_alignment_groups=replay_axis_alignment_groups,
            center_alignment_pairs=replay_center_alignment_pairs,
            aggregate_envelope_orders=replay_aggregate_envelope_orders,
            hierarchical_macro_orders=replay_hierarchical_macro_orders,
            hierarchical_macro_alignments=replay_hierarchical_macro_alignments,
            hierarchical_cohesion_groups=(
                replay_hierarchical_macro_cohesion_groups
            ),
            aspect_bounds=aspect_bounds,
            aspect_obligation=aspect_obligation,
            manufacturing_grid_nm=manufacturing_grid_nm,
            minimum_structural_occupancy=minimum_structural_occupancy,
            placement_objective=placement_objective,
            fixed_outline_width_nm=fixed_outline_width_nm,
            area_target=area_target,
            incumbent_seed_positions=replay_incumbent_seed_positions,
            flexible_envelope_constraints=canonical_flexible_envelopes,
            rigid_compound_footprints=(
                replay_rigid_compound_footprints if left_column is None else ()
            ),
            layered_collision_footprints=(
                replay_layered_collision_footprints if left_column is None else ()
            ),
            row_bounded_cavity_projections=(
                replay_row_bounded_cavity_projections
            ),
            **({"mosaic_guest_slot_projections": replay_mosaic_guest_slot_projections}
               if replay_mosaic_guest_slot_projections else {}),
            **({"edge_rail_reservation": replay_edge_rail_reservation}
               if replay_edge_rail_reservation is not None else {}),
            overlap_compatible_pairs=materialized_overlap_compatible_pairs,
            **(
                {
                    "seed_cohesion_groups": replay_well_share_seed_cohesion_groups,
                }
                if fixed_outline_width_nm is not None
                and replay_well_share_seed_cohesion_groups
                else {}
            ),
            )
    if generated_incumbent is not None:
        generated_incumbent.require_final_positions(replay_result.positions)
    if (
        replay_result.certificate.hierarchical_alignment_tolerance_scales_nm
        != placement_result.certificate.hierarchical_alignment_tolerance_scales_nm
    ):
        raise ValueError(
            "hierarchical alignment tolerance scale changed across exact replay"
        )
    (
        ring_pad_top_nm,
        ring_pad_bottom_nm,
        _final_bottom_padded_blocks,
    ) = _replay_ring_padding_on_positioned_master(
        materialized_blocks=materialized_blocks,
        master_blocks=materialized_master_blocks,
        master_members=master_members,
        positions=replay_result.positions,
        ring_block_ids=set(materialized_rings),
        block_stamps=materialized_stamps,
        block_member_devices=materialized_member_devices,
        prior_top_padding=ring_pad_top_nm,
        prior_bottom_padding=ring_pad_bottom_nm,
        bands=bands_map,
        left_column=left_column,
        guard_rings=guard_ring_declarations,
        guard_ring_outer_clearances_nm=guard_ring_outer_clearances_nm,
        top_edge_keepout_nm=top_edge_keepout_nm,
        top_edge_rail_nets=tuple(sorted(top_edge_rail_nets)),
        bottom_edge_keepout_nm=bottom_center_keepout_nm,
        bottom_edge_rail_nets=tuple(sorted(bottom_edge_rail_nets)),
    )
    replay_deltas: Dict[str, Tuple[int, int]] = {}
    for master, members in master_members.items():
        old_x0, old_y0, _old_x1, _old_y1 = materialized_master_bbox[master]
        new_x0, new_y0 = replay_result.positions[master]
        delta = (new_x0 - old_x0, new_y0 - old_y0)
        for member in members:
            replay_deltas[member] = delta
    compact_um, compact_geometry = apply_block_deltas(
        compact_geometry,
        compact_um,
        intent,
        tech,
        tech_id,
        replay_deltas,
        materialized_stamps,
        materialized_rings,
        materialized_ring_names,
        bi=bi,
        member_ring_pad_top_nm=(ring_pad_top_nm or None),
        member_ring_pad_bottom_nm=(ring_pad_bottom_nm or None),
    )
    _replay_cross_row_bottom_alignments(deferred_cross_row_alignments, compact_geometry, compact_um)
    _replay_native_left_alignments(deferred_native_left_alignments, row_program, compact_geometry, compact_um)
    (materialized_blocks, final_stamps, final_rings, _ring_names,
     final_layered_replay, final_layered_projection) = current_blocks(compact_geometry, compact_um)
    materialized_source_ownership, materialized_source_geometry = materialized_source_snapshot(
        required=bool(layered_placement_assemblies) or _materialized_source_ownership_requested(
            group_route_requested, intent_fact_set), blocks=materialized_blocks,
        stamps=final_stamps, geometry=compact_geometry, placement_um=compact_um,
        canonical_device_names=canonical_device_names)
    final_materialized_collision_footprints = (
        _fixed_pcell_compound_footprints_from_geometry(
            solver_blocks=tuple(materialized_blocks),
            block_stamps=final_stamps,
            geometry=compact_geometry,
            master_members={
                block_id: (block_id,) for block_id, _bbox in materialized_blocks
            },
        )
    )
    final_native_access_pair_clearances_nm = _project_native_access_pair_clearances_to_masters(
        _mim_native_access_pair_clearances_nm(
            tech, block_stamps=final_stamps, geometry=compact_geometry,
            base_gap_nm=gap_nm, device_nets=device_nets,
            rail_specs=_compiled_edge_rail_specs_for_placement(intent, tech, bi),
            placement_um=compact_um,
        ),
        master_members=master_members, base_gap_nm=gap_nm,
    )
    if final_native_access_pair_clearances_nm != replay_native_access_pair_clearances_nm:
        raise ValueError("native-access obligations changed across exact replay")
    final_overlap_compatible_pairs = tuple(
        pair
        for pair in _layer_disjoint_overlap_compatible_pairs(
            excluded_layers=collision_excluded_layers,
            block_stamps=final_stamps,
            geometry=compact_geometry,
            master_members=master_members,
        )
        if pair not in pair_clearances_nm and pair not in final_native_access_pair_clearances_nm
    )
    if group_spacing_pairs:
        if _admitted_group_spacing_pairs(replay_edge_rail_reservation, pair_clearances_nm,
                                         final_native_access_pair_clearances_nm) != group_spacing_pairs:
            raise ValueError('native group source spacing changed across final replay')
        final_overlap_compatible_pairs = tuple(sorted(set(final_overlap_compatible_pairs) | set(group_spacing_pairs)))
    final_overlap_compatible_pairs = (
        final_overlap_compatible_pairs if left_column is None else ()
    )
    final_layered_collision_footprints = (
        _layered_collision_footprints_from_geometry(
            excluded_layers=collision_excluded_layers,
            solver_blocks=replay_result.certificate.blocks,
            materialized_blocks=tuple(materialized_blocks),
            block_stamps=final_stamps,
            block_gring=final_rings,
            geometry=compact_geometry,
            placement_um=compact_um,
            master_members=master_members,
            component_refinement_owners=collision_projection_scope.master_owners,
            flexible_envelope_constraints=canonical_flexible_envelopes,
        )
        if left_column is None
        else ()
    )
    if final_overlap_compatible_pairs != materialized_overlap_compatible_pairs:
        raise ValueError(
            "layer-disjoint overlap compatibility changed across exact replay"
        )
    final_certificate = _bind_certificate_to_materialized_blocks(
        replay_result.certificate,
        materialized_blocks=materialized_blocks,
        master_members=master_members,
        materialized_block_source_members=materialized_source_ownership,
        materialized_source_member_bboxes=materialized_source_geometry,
        materialized_collision_footprints=final_materialized_collision_footprints,
        layered_collision_footprints=final_layered_collision_footprints,
        overlap_compatible_pairs=final_overlap_compatible_pairs,
        gap_nm=gap_nm,
        pair_clearances_nm=pair_clearances_nm,
        pair_clearance_scope="solver",
        placement_declaration_refs=placement_declaration_refs,
        aspect_obligation=aspect_obligation,
        layered_placement_assemblies=layered_placement_assemblies,
        layered_placement_replay=final_layered_replay,
        layered_placement_projection=final_layered_projection,
    )
    if intent_consumption_certificates:
        final_certificate = replace(
            final_certificate,
            intent_consumption_certificates=intent_consumption_certificates,
        )
    if fact_projections:
        final_certificate = replace(
            final_certificate,
            fact_projections=fact_projections,
        )
    if mos_matching_devices:
        matching_ownership, matching_deltas = (
            _matching_axis_cumulative_authority(
                matching_pair_axes=legacy_matching_pair_axes,
                device_kind=device_kind,
                live_master_ids=tuple(sorted(master_members)),
                block_member_devices=block_member_devices,
                master_members=master_members,
                first_member_deltas_nm=deltas,
                replay_member_deltas_nm=replay_deltas,
            )
        )
        final_certificate = replace(
            final_certificate,
            matching_pair_axis_projections=(
                _project_mos_matching_axes_through_block_deltas(
                    matching_pair_axes=legacy_matching_pair_axes,
                    source_device_bboxes=source_mos_matching_bboxes,
                    final_device_bboxes=_source_device_bboxes_nm(
                        compact_geometry,
                        compact_um,
                        mos_matching_devices,
                    ),
                    source_block_member_devices=matching_ownership,
                    device_kind=device_kind,
                    block_deltas_nm=matching_deltas,
                )
            ),
        )
    if cross_inventory is not None:
        from clay.layout.place.cross_row_matching import native_cross_row_final_proof, native_cross_row_stamp_centres
        final_centres = native_cross_row_stamp_centres(
            cross_inventory, row_program, row_materialization, compact_geometry, compact_um)
        final_certificate = replace(final_certificate, native_cross_row_final_proof=
            native_cross_row_final_proof(cross_inventory, native_cross_row_cp_proof, cross_directions,
                final_centres, manufacturing_grid_nm, row_program=row_program,
                row_materialization=row_materialization, tech=tech))
    if same_inventory is not None:
        from clay.layout.place.native_same_row_matching import native_same_row_final_proof, native_same_row_stamp_centres
        final_centres = native_same_row_stamp_centres(
            same_inventory, row_program, row_materialization, compact_geometry, compact_um)
        final_certificate = replace(final_certificate, native_same_row_final_proof=
            native_same_row_final_proof(same_inventory, same_initial_centres, same_initial_axes,
                                        final_centres, manufacturing_grid_nm, logical_axes=same_logical_axes))
    _assert_materialized_diff_axis_certificate(
        final_certificate,
        block_member_devices,
        compact_geometry,
        compact_um,
        intent,
        bi,
        row_program=row_program, row_materialization=row_materialization,
        tech=tech, source_cell=cell_def,
        native_left_alignment_declarations=deferred_native_left_alignments,
    )
    final_certificate = replace(
        final_certificate,
        reference_conditioned_placement_obligations=(
            reference_conditioned_placement_obligations
        ),
        collision_exclusion_pdk_id=tech_id if collision_excluded_layers else '',
        collision_excluded_layers=collision_excluded_layers,
    )
    return RepositionPum2DResult(
        placement_um=compact_um,
        geometry=compact_geometry,
        final_placement_certificate=final_certificate,
        member_ring_pad_top_nm=tuple(sorted(ring_pad_top_nm.items())),
        member_ring_pad_bottom_nm=tuple(sorted(ring_pad_bottom_nm.items())),
        intent_consumption_certificates=intent_consumption_certificates,
        fact_projections=fact_projections,
    )
