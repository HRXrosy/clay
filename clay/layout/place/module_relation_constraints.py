"""Exact module relations from current logical artwork union midpoints.

Only relations are lowered. The explicit contained-frame domain makes integer
floor-midpoint ordering equivalent to the native normalized-ppm ordering. This
does not verify module locations, source authority, structure, or signoff.
"""
from collections import Counter
import sys

from ortools.sat.python import cp_model

from clay.layout.reference_quality_contract import ReferenceQualityContract, RigidQualityGeometryIR


_PREFIX = "reference_quality_module_relation:"
_PPM = 1_000_000
_SAFE = (1 << 62) - 1


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _bytes(model):
    return model.Proto().SerializeToString(deterministic=True)


def _activity(*terms):
    _require(sum(abs(value) for value in terms) <= _SAFE,
             "module relation affine activity exceeds int64 safety bound")


def _normalization_domain(frame):
    _require(type(frame) is tuple and len(frame) == 4 and all(type(value) is int for value in frame),
             "module relation frame must be an exact integer bbox")
    extents = [frame[axis+2] - frame[axis] for axis in (0, 1)]
    _require(all(0 < value <= _PPM for value in extents),
             "module relation normalization frame extents must be in 1..1000000 nm")
    _require(sys.implementation.name == "cpython" and sys.version_info >= (3, 10)
             and (sys.float_info.radix, sys.float_info.mant_dig, sys.float_info.rounds) == (2, 53, 1),
             "module relation normalization requires nearest-binary64 CPython arithmetic")
    for axis in (0, 1):
        _activity(4 * max(abs(frame[axis]), abs(frame[axis+2])), 1)
    # The metric multiplies integer n by PPM before its one binary64 division.
    # For 0 <= n <= D < PPM, adjacent exact quotients differ by at least
    # 1 + 1/D. Each division error is <2^-32; their combined error is <1/D.
    # Computed quotients therefore remain >1 apart, so round preserves strict
    # order. D==PPM gives exact integers. Equal inputs give equal outputs;
    # no equivalence to mathematical rational ties-to-even is needed here.
    return {"arithmetic": "CPYTHON_NEAREST_BINARY64", "frame_extents_nm": extents,
        "maximum_frame_extent_nm": _PPM, "normalized_integer_numerator_bound": _PPM * _PPM,
        "division_error_strict_bound_ppm": [1, 1 << 32],
        "submaximum_adjacent_spacing_excess_lower_bound_ppm": [1, _PPM - 1],
        "adjacent_combined_division_error_strict_bound_ppm": [1, 1 << 31],
        "maximum_extent_normalization": "EXACT_INTEGER_QUOTIENT",
        "contained_integer_midpoint_order": "STRICTLY_PRESERVED_BY_NATIVE_NORMALIZATION"}


def _coordinate_bounds(model, indices, owners):
    _require(type(indices) is dict and set(indices) == owners,
             "module relation coordinate owner inventory differs")
    proto = model.Proto()
    seen, bounds = set(), {}
    for owner, pair in sorted(indices.items()):
        _require(type(pair) is tuple and len(pair) == 2,
                 "module relation coordinate indices must be exact axis pairs")
        axes = []
        for axis, index in enumerate(pair):
            _require(type(index) is int and 0 <= index < len(proto.variables) and index not in seen,
                     "module relation coordinate indices are missing, aliased, or malformed")
            seen.add(index)
            variable = proto.variables[index]
            _require(variable.name == f"{'xy'[axis]}_{owner}",
                     "module relation coordinate axis/owner identity differs")
            domain = tuple(variable.domain)
            _require(domain and len(domain) % 2 == 0
                     and all(low <= high for low, high in zip(domain[::2], domain[1::2]))
                     and all(domain[i-1] < domain[i] for i in range(2, len(domain), 2)),
                     "module relation coordinate domain is malformed")
            axes.append((int(domain[0]), int(domain[-1])))
        bounds[owner] = tuple(axes)
    return bounds


def _module_inventory(geometry, contract, bounds, frame):
    expected = Counter({row.logical_id: len(row.occurrence_ids) for row in contract.logical_population})
    _require(Counter(row.logical_id for row in geometry.occurrences) == expected,
             "module relation complete logical occurrence population/count differs")
    modules = []
    for target in contract.modules:
        logicals = set(target.logical_ids)
        selected = [row for row in geometry.materialized_blocks if logicals.intersection(row.logical_ids)]
        _require(selected and all(set(row.logical_ids) <= logicals for row in selected)
                 and {name for row in selected for name in row.logical_ids} == logicals,
                 "module relation artwork ownership differs from complete module partition")
        blocks = []
        for row in selected:
            for axis in (0, 1):
                maximum_coordinate = max(map(abs, bounds[row.owner_id][axis]))
                maximum_frame = max(abs(frame[axis]), abs(frame[axis+2]))
                for index in (axis, axis+2):
                    _activity(maximum_coordinate, row.bbox_offset_nm[index], maximum_frame)
            blocks.append({"block_id": row.block_id, "owner_id": row.owner_id,
                "logical_ids": list(row.logical_ids), "bbox_offset_nm": list(row.bbox_offset_nm)})
        modules.append({"module_id": target.module_id, "logical_ids": list(target.logical_ids),
            "occurrence_count": len(target.occurrence_ids), "blocks": blocks})
    return modules


def _append_module(candidate, item, indices, frame, *, prefix=_PREFIX):
    first = len(candidate.Proto().constraints)
    bbox_indices, center_indices = [None] * 4, []
    for axis in (0, 1):
        name = f"{prefix}{item['module_id']}:{'xy'[axis]}"
        low = candidate.NewIntVar(frame[axis], frame[axis+2], name + ":minimum")
        high = candidate.NewIntVar(frame[axis], frame[axis+2], name + ":maximum")
        center = candidate.NewIntVar(frame[axis], frame[axis+2], name + ":midpoint")
        starts, ends = [], []
        for row in item["blocks"]:
            coordinate = candidate.GetIntVarFromProtoIndex(indices[row["owner_id"]][axis])
            starts.append(coordinate + row["bbox_offset_nm"][axis])
            ends.append(coordinate + row["bbox_offset_nm"][axis+2])
        # Bounding both true extrema by the frame is complete union containment.
        candidate.AddMinEquality(low, starts).WithName(name + ":union_minimum")
        candidate.AddMaxEquality(high, ends).WithName(name + ":union_maximum")
        candidate.Add(2 * center <= low + high).WithName(name + ":floor_lower")
        candidate.Add(low + high <= 2 * center + 1).WithName(name + ":floor_upper")
        bbox_indices[axis], bbox_indices[axis+2] = low.Index(), high.Index()
        center_indices.append(center.Index())
    item.update(bbox_variable_indices=bbox_indices, midpoint_variable_indices=center_indices,
                constraint_indices=list(range(first, len(candidate.Proto().constraints))))
    return tuple(candidate.GetIntVarFromProtoIndex(index) for index in center_indices)


def append_module_relation_constraints(model, *, coordinate_indices, geometry, contract, frame_nm):
    """Append all module pair/axis relations in the explicit bounded-frame domain."""
    if type(model) is not cp_model.CpModel:
        raise TypeError("module relation compiler requires an exact CP model")
    if type(geometry) is not RigidQualityGeometryIR or type(contract) is not ReferenceQualityContract:
        raise TypeError("module relation compiler requires exact current geometry and public contract")
    geometry.__post_init__()
    contract.__post_init__()
    normalization = _normalization_domain(frame_nm)
    before = _bytes(model)
    _require(not any(row.name.startswith(_PREFIX) for row in model.Proto().variables)
             and not any(row.name.startswith(_PREFIX) for row in model.Proto().constraints),
             "module relation model is already augmented")
    bounds = _coordinate_bounds(model, coordinate_indices, {row[0] for row in geometry.owners})
    modules = _module_inventory(geometry, contract, bounds, frame_nm)
    metadata = {"schema": "clay.module-relation-constraints/v1",
        "scope": "EXACT_MODULE_RELATIONS_WITHIN_BOUNDED_FRAME", "complete_quality_equivalence": False,
        "frame_nm": list(frame_nm), "normalization": normalization,
        "containment": "ACTUAL_UNION_EXTREMA_DOMAINS_ARE_THE_DECLARED_FRAME",
        "module_count": len(modules), "relation_count": len(contract.module_relations),
        "logical_count": len(contract.logical_population), "occurrence_count": len(geometry.occurrences),
        "contract_sha256": contract.sha256, "geometry_sha256": geometry.sha256,
        "modules": modules, "relations": []}
    candidate = model.Clone()
    centers = {item["module_id"]: _append_module(candidate, item, coordinate_indices, frame_nm) for item in modules}
    for target in contract.module_relations:
        axis = 0 if target.axis == "horizontal" else 1
        first, second = centers[target.first_module][axis], centers[target.second_module][axis]
        index = len(candidate.Proto().constraints)
        if target.relation == "before":
            constraint = candidate.Add(first + 1 <= second)
        elif target.relation == "after":
            constraint = candidate.Add(second + 1 <= first)
        else:
            constraint = candidate.Add(first == second)
        constraint.WithName(f"{_PREFIX}{target.first_module}:{target.second_module}:{target.axis}")
        metadata["relations"].append({"first_module": target.first_module, "second_module": target.second_module,
            "axis": target.axis, "relation": target.relation, "constraint_index": index})
    prefix = candidate.Clone()
    del prefix.Proto().variables[len(model.Proto().variables):]
    del prefix.Proto().constraints[len(model.Proto().constraints):]
    _require(_bytes(prefix) == before and _bytes(model) == before,
             "module relation compilation changed the original CP prefix identity")
    model.Proto().CopyFrom(candidate.Proto())
    return metadata


__all__ = ["append_module_relation_constraints"]
