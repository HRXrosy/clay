"""Append the native outer obligations absent from the X-repair legalizer.

This compiler consumes the *actual* legalizer model and its complete outer
operands. It preserves every original variable, constraint and assumption;
collision, floorplan, symmetry, hierarchy, adjacency, aspect and source-specific
projections remain obligations of that prefix. It is not a replacement native
model, a source/PDK admission, or a certificate for an arbitrary supplied model.
The host must still certify the original CP and replay the complete outer check.

The additional domain includes physical flexible extents, legacy band order,
rail keepouts, normalized bottom origin, full solve cuts and declared outline
anchors. Horizontal order adds the outer base gap while retaining the native
pair-specific gap. No incumbent-derived group span or reference placement is read.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import inspect
import math

from ortools.sat.python import cp_model

from clay.flow.execution_session import current_session
from clay.layout.place import constraints_hierarchical as hierarchy
from clay.layout.place import constraints_spatial as spatial
from clay.layout.place.generated_cohesion import _coordinates, _pairs, _require_prefix
from clay.layout.place.generated_incumbent import canonical_bytes
from clay.layout.place.placement_constants import _BAND_CENTER
from clay.layout.place.placement_cuts import (
    _PreparedSolveCut, _add_solve_cut_constraints, _is_canonical_sha256,
)
from clay.layout.place.placement_diagnostics import _diagnostic_canonical_value
from clay.layout.place.placement_replay import (
    _banded_seed_satisfies_hard_constraints, _require_edge_rail_reservation,
)


_OUTER_FIELDS = frozenset('''ids dims gap_nm width_bound_nm height_bound_nm bands
adjacency_limits_nm adjacency_scope_offsets_nm order_pairs vertical_order_pairs
floorplan_column_classes floorplan_column_groups bottom_center_keepout_nm
bottom_edge_keepout_nm bottom_edge_keepout_exempt top_edge_keepout_nm
top_edge_keepout_clear solve_cuts pair_clearances_nm axis_alignment_groups
center_alignment_pairs aggregate_envelope_orders hierarchical_macro_orders
hierarchical_macro_alignments hierarchical_alignment_tolerance_scales_nm
aspect_target aspect_bounds aspect_obligation manufacturing_grid_nm right_edge
left_edge placement_objective strict_width_cap exact_outline_width lateral_blocks
flexible_envelope_constraints rigid_compound_footprints layered_collision_footprints
row_bounded_cavity_projections mosaic_guest_slot_projections edge_rail_reservation
overlap_compatible_pairs enforce_legacy_band_order'''.split())

_ADDITIONAL_FAMILIES = (
    'outline_bounds', 'physical_outline_anchors', 'right_edge', 'bottom_keepout',
    'flexible_bottom_keepout', 'bottom_origin', 'top_edge_keepout', 'band_order',
    'flexible_band_order', 'flexible_foreign_clearance', 'solve_cuts',
    'horizontal_order_base_gap',
)
_INHERITED_FAMILIES = (
    'native_coordinate_domains', 'manufacturing_grid', 'physical_collision',
    'pair_clearance', 'axis_alignment', 'center_alignment', 'horizontal_order',
    'vertical_order', 'floorplan_columns', 'aggregate_envelope_order',
    'hierarchical_orders_alignments', 'adjacency', 'aspect', 'left_edge',
    'row_cavity', 'mosaic_guest_slot', 'edge_rail_reservation',
)


def _checkpoint():
    session = current_session()
    if session is not None:
        session.checkpoint()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, field, *, minimum=0):
    _require(type(value) is int and minimum <= value < 2**60,
             f'{field} requires a bounded exact integer >= {minimum}')
    return value


def _owners(value, ids, field):
    _require(type(value) in (tuple, list) and all(type(name) is str for name in value)
             and len(value) == len(set(value)) and set(value) <= set(ids),
             f'{field} has a foreign, duplicated or malformed owner')
    return set(value)


def _validate_cuts(cuts, ids):
    _require(type(cuts) is tuple, 'solve_cuts must be the complete prepared tuple')
    for cut in cuts:
        _require(type(cut) is _PreparedSolveCut, 'solve_cuts contains an unsupported type')
        _require(all(_is_canonical_sha256(value) for value in
                     (cut.cut_hash, cut.placement_hash, cut.candidate_pool_hash)),
                 'prepared solve cut hashes differ')
        _require(not cut.placement_basis_sha256 or _is_canonical_sha256(cut.placement_basis_sha256),
                 'prepared solve cut basis hash differs')
        _integer(cut.required_clearance_nm, 'solve cut clearance')
        _require(type(cut.incumbent_origins) is tuple and type(cut.guarded_coordinates) is tuple,
                 'solve cut coordinate ledgers must be exact tuples')
        origins, guards = set(), set()
        for row in cut.incumbent_origins:
            _require(type(row) is tuple and len(row) == 3 and row[0] in ids and row[0] not in origins,
                     'solve cut incumbent owner inventory differs')
            origins.add(row[0])
            _integer(row[1], 'solve cut incumbent x')
            _integer(row[2], 'solve cut incumbent y')
        for row in cut.guarded_coordinates:
            _require(type(row) is tuple and len(row) == 3 and row[0] in ids
                     and row[1] in ('x', 'y') and row[:2] not in guards,
                     'solve cut guarded coordinate inventory differs')
            guards.add(row[:2])
            _integer(row[2], 'solve cut guarded coordinate')
        if not origins:
            _require(cut.owner_a in ids and cut.owner_b in ids and cut.owner_a != cut.owner_b
                     and cut.corridor_relation in ('horizontal_separation', 'vertical_separation'),
                     'unsupported solve cut relation or owner')


def _validate_inherited(values, ids, dims):
    """Validate existing typed currencies without inventing new native defaults."""
    blocks = tuple((name, (0, 0, *dims[name])) for name in ids)
    for field in ('order_pairs', 'vertical_order_pairs'):
        _require(type(values[field]) in (tuple, list), f'{field} must be a finite sequence')
        for row in values[field]:
            _require(type(row) in (tuple, list) and len(row) == 2
                     and row[0] in ids and row[1] in ids and row[0] != row[1],
                     f'{field} requires two distinct current owners')
    _require(isinstance(values['pair_clearances_nm'], Mapping), 'pair clearances require a mapping')
    for pair, gap in values['pair_clearances_nm'].items():
        _require(type(pair) is tuple and len(pair) == 2 and pair[0] in ids
                 and pair[1] in ids and pair[0] < pair[1], 'pair clearance owners differ')
        _integer(gap, 'pair clearance')
    spatial._canonical_floorplan_column_classes(values['floorplan_column_classes'], block_ids=dict(blocks))
    spatial._canonical_floorplan_column_groups(values['floorplan_column_groups'], block_ids=dict(blocks))
    spatial._canonical_axis_alignment_groups(values['axis_alignment_groups'], block_ids=ids, dims=dims)
    spatial._canonical_center_alignment_pairs(values['center_alignment_pairs'], block_ids=ids, dims=dims)
    for field, validator in (
        ('aggregate_envelope_orders', hierarchy._canonical_aggregate_envelope_orders),
        ('hierarchical_macro_orders', hierarchy._canonical_hierarchical_macro_orders),
        ('hierarchical_macro_alignments', hierarchy._canonical_hierarchical_macro_alignments),
    ):
        validator(values[field], block_ids=ids)
    hierarchy._require_hierarchical_alignment_tolerance_scales_nm(
        values['hierarchical_alignment_tolerance_scales_nm'], alignments=values['hierarchical_macro_alignments'])
    for field, validator in (
        ('rigid_compound_footprints', spatial._canonical_rigid_compound_footprints),
        ('layered_collision_footprints', spatial._canonical_layered_collision_footprints),
        ('row_bounded_cavity_projections', spatial._canonical_row_bounded_cavity_projections),
    ):
        validator(values[field], dims=dims)
    spatial._canonical_overlap_compatible_pairs(values['overlap_compatible_pairs'], block_ids=ids,
                                                pair_clearances_nm=values['pair_clearances_nm'])
    spatial._canonical_mosaic_guest_slot_projections(values['mosaic_guest_slot_projections'], dims=dims,
                                                     minimum_clearance_nm=values['gap_nm'])
    _require_edge_rail_reservation(values['edge_rail_reservation'], dims=dims)
    limits = spatial._canonical_adjacency_limits_nm(values['adjacency_limits_nm'], block_ids=ids,
                                                    base_gap_nm=values['gap_nm'])
    offsets = values['adjacency_scope_offsets_nm']
    _require(isinstance(offsets, Mapping) and set(offsets) <= set(ids), 'adjacency scope owners differ')
    _require({name for row in limits for name in row[:2]} <= set(offsets), 'adjacency scope is incomplete')
    for box in offsets.values():
        _require(type(box) is tuple and len(box) == 4 and all(type(v) is int for v in box)
                 and box[0] < box[2] and box[1] < box[3], 'adjacency scope must be an exact physical box')


def _validate(values):
    declared = {name for name in inspect.signature(_banded_seed_satisfies_hard_constraints).parameters
                if name != 'seed' and not name.endswith('_out')}
    _require(declared == _OUTER_FIELDS, 'unsupported current outer signature; compiler coverage must be updated')
    _require(isinstance(values, Mapping) and set(values) == _OUTER_FIELDS, 'complete outer fields differ')
    ids = values['ids']
    _require(type(ids) in (tuple, list) and ids and all(type(n) is str and n and n == n.strip() for n in ids)
             and len(ids) == len(set(ids)), 'complete outer owner inventory is malformed')
    dims = _pairs(values['dims'], minimum=1, label='outer dimensions')
    _require(isinstance(values['bands'], Mapping) and set(dims) == set(ids) == set(values['bands']),
             'outer dimension/band owner inventories differ')
    _require(all(type(band) is int for band in values['bands'].values()), 'outer bands must be exact integers')
    for field in ('gap_nm', 'bottom_center_keepout_nm', 'bottom_edge_keepout_nm', 'top_edge_keepout_nm'):
        _integer(values[field], field)
    for field in ('width_bound_nm', 'height_bound_nm', 'manufacturing_grid_nm'):
        _integer(values[field], field, minimum=1)
    for field in ('aspect_obligation', 'strict_width_cap', 'exact_outline_width', 'enforce_legacy_band_order'):
        _require(type(values[field]) is bool, f'{field} must be an exact bool')
    _require(type(values['placement_objective']) is str and values['placement_objective'] in ('area', 'height'),
             'unsupported placement objective')
    target, bounds = values['aspect_target'], values['aspect_bounds']
    _require(type(target) in (int, float) and math.isfinite(target) and target > 0, 'aspect target must be finite positive')
    if bounds is not None:
        _require(type(bounds) in (tuple, list) and len(bounds) == 2
                 and all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in bounds)
                 and bounds[0] <= bounds[1], 'aspect bounds must be finite, positive and ordered')
    if values['aspect_obligation']:
        spatial._scaled_aspect_bounds(target, bounds)
    for field in ('lateral_blocks', 'right_edge', 'left_edge', 'bottom_edge_keepout_exempt', 'top_edge_keepout_clear'):
        _owners(values[field], ids, field)
    flexible = spatial._canonical_flexible_envelope_constraints(
        values['flexible_envelope_constraints'], block_ids=ids, bands=values['bands'])
    for row in flexible:
        row.__post_init__()
    spatial._flexible_seed_physical_views({name: (0, 0) for name in ids}, dims=dims, constraints=flexible)
    _validate_inherited(values, ids, dims)
    _validate_cuts(values['solve_cuts'], ids)
    return tuple(ids), dims, flexible


def _bounds(model, boxes, width, height, name):
    result = tuple(model.NewIntVar(0, (width, height)[axis % 2], f'outer:{name}:{axis}') for axis in range(4))
    for axis, target in enumerate(result):
        method = model.AddMinEquality if axis < 2 else model.AddMaxEquality
        method(target, [box[axis] for box in boxes])
    return result


def _separate(model, left, right, required, name):
    terms = (right[0] - left[2], left[0] - right[2], right[1] - left[3], left[1] - right[3])
    directions = tuple(model.NewBoolVar(f'outer:separation:{name}:{axis}') for axis in range(4))
    for term, direction in zip(terms, directions):
        model.Add(term >= required).OnlyEnforceIf(direction)
    model.AddBoolOr(directions)


def _physical_views(model, values, ids, dims, flexible, xs, ys):
    rigid = {name: (xs[name], ys[name], xs[name] + dims[name][0], ys[name] + dims[name][1]) for name in ids}
    physical, groups = dict(rigid), {}
    width, height = values['width_bound_nm'], values['height_bound_nm']
    for row in flexible:
        _checkpoint()
        for member, offset in row.member_physical_offsets_nm:
            physical[member] = tuple((xs[member], ys[member])[axis % 2] + value for axis, value in enumerate(offset))
        groups[row.group] = _bounds(model, [physical[member] for member in row.members], width, height, f'flex:{row.group}')
        for foreign in ids:
            _checkpoint()
            if foreign in row.members:
                continue
            required = {member: max(row.foreign_clearance_nm,
                values['pair_clearances_nm'].get(tuple(sorted((member, foreign))), values['gap_nm'])) for member in row.members}
            if row.foreign_exclusion_mode == 'envelope':
                _separate(model, groups[row.group], rigid[foreign], max(required.values()), f'{row.group}:{foreign}')
            else:
                for member in row.members:
                    _separate(model, physical[member], rigid[foreign], required[member], f'{member}:{foreign}')
    all_boxes = (*rigid.values(), *groups.values())
    for box in all_boxes:
        model.Add(box[0] >= 0)
        model.Add(box[1] >= 0)
        model.Add(box[2] <= width)
        model.Add(box[3] <= height)
    outline = _bounds(model, all_boxes, width, height, 'outline')
    if values['exact_outline_width']:
        model.Add(outline[0] == 0)
        model.Add(outline[2] == width)
    if values['placement_objective'] == 'area':
        model.Add(outline[0] == 0)
    for name in values['right_edge']:
        model.Add(xs[name] + dims[name][0] == width)
    return rigid, physical, groups, outline


def _bands_and_keepouts(model, values, ids, flexible, ys, rigid, physical, groups, outline):
    bands, lateral = values['bands'], set(values['lateral_blocks'])
    exempt = set(values['bottom_edge_keepout_exempt'])
    edge, center = values['bottom_edge_keepout_nm'], values['bottom_center_keepout_nm']
    spatial._add_bottom_edge_keepout_constraints(model, ys, edge, exempt)
    spatial._add_top_edge_keepout_constraints(model, {name: box[3] for name, box in physical.items()},
                                              outline[3], values['top_edge_keepout_nm'], values['top_edge_keepout_clear'])
    slacks = []
    for name in ids:
        lower = max(0 if name in exempt else edge, center if name not in lateral and bands[name] == _BAND_CENTER else 0)
        model.Add(ys[name] >= lower)
        slacks.append(ys[name] - lower)
    for row in flexible:
        lower = max(0 if set(row.members) <= exempt else edge, center if bands[row.members[0]] == _BAND_CENTER else 0)
        model.Add(groups[row.group][1] >= lower)
        slacks.append(groups[row.group][1] - lower)
    model.AddMinEquality(0, slacks)
    if values['enforce_legacy_band_order']:
        core = tuple(name for name in ids if name not in lateral)
        present = sorted({bands[name] for name in core})
        for previous, current in zip(present, present[1:]):
            tops = [rigid[name][3] for name in core if bands[name] == previous]
            tops += [groups[row.group][3] for row in flexible if bands[row.members[0]] == previous]
            starts = [ys[name] for name in core if bands[name] == current]
            starts += [groups[row.group][1] for row in flexible if bands[row.members[0]] == current]
            top = model.NewIntVar(0, values['height_bound_nm'], f'outer:band_top:{previous}')
            model.AddMaxEquality(top, tops)
            for start in starts:
                model.Add(start >= top + values['gap_nm'])


def append_current_outer_domain(model, *, coordinate_indices, outer_kwargs):
    """Return a cloned model plus descriptive identity data; never call Solve.

    The caller must supply the complete actual legalizer prefix associated with
    ``outer_kwargs`` and retain independent native replay. This data-level API
    cannot prove that association for an arbitrary caller-supplied CpModel.
    A contradictory explicit slice stays INFEASIBLE; no row or bound is relaxed.
    """
    _checkpoint()
    _require(type(model) is cp_model.CpModel, 'outer compiler requires an exact current CpModel')
    before = model.Proto().SerializeToString(deterministic=True)
    input_bytes = canonical_bytes(_diagnostic_canonical_value(outer_kwargs))
    ids, dims, flexible = _validate(outer_kwargs)
    indices = _pairs(coordinate_indices, minimum=0, label='outer coordinate indices')
    flat = [index for pair in indices.values() for index in pair]
    _require(set(indices) == set(ids) and all(index < len(model.Proto().variables) for index in flat),
             'outer coordinate indices differ from current owner inventory')
    points = {name: tuple(model.Proto().variables[index].domain[0] for index in pair) for name, pair in indices.items()}
    _coordinates(model.Proto(), indices, dims, points)
    result = model.Clone()
    xs = {name: result.GetIntVarFromProtoIndex(indices[name][0]) for name in ids}
    ys = {name: result.GetIntVarFromProtoIndex(indices[name][1]) for name in ids}
    rigid, physical, groups, outline = _physical_views(result, outer_kwargs, ids, dims, flexible, xs, ys)
    _bands_and_keepouts(result, outer_kwargs, ids, flexible, ys, rigid, physical, groups, outline)
    for left, right in outer_kwargs['order_pairs']:
        _checkpoint()
        result.Add(xs[right] >= xs[left] + dims[left][0] + outer_kwargs['gap_nm'])
    _add_solve_cut_constraints(result, xs, ys, dims, outer_kwargs['solve_cuts'])
    _require_prefix(model.Proto(), result.Proto())
    _require(not result.Validate(), 'invalid augmented outer model: ' + result.Validate())
    _checkpoint()
    _require(model.Proto().SerializeToString(deterministic=True) == before
             and canonical_bytes(_diagnostic_canonical_value(outer_kwargs)) == input_bytes
             and _pairs(coordinate_indices, minimum=0, label='outer coordinate indices') == indices,
             'outer compiler inputs changed during construction')
    return result, dict(schema='clay.current-outer-domain/v1', outer_fields=tuple(sorted(_OUTER_FIELDS)),
        coordinate_indices=indices, outer_input_sha256=hashlib.sha256(input_bytes).hexdigest(),
        original_model_sha256=hashlib.sha256(before).hexdigest(),
        scope='SUPPLEMENT_TO_ACTUAL_LEGALIZER_REQUIRES_FULL_NATIVE_REPLAY',
        inherited_families=_INHERITED_FAMILIES, appended_families=_ADDITIONAL_FAMILIES,
        horizontal_order_semantics=dict(inherited='pair_specific_gap', appended='base_gap', combination='AND'),
        added_variable_count=len(result.Proto().variables) - len(model.Proto().variables),
        added_constraint_count=len(result.Proto().constraints) - len(model.Proto().constraints))
