"""Complete envelopes and sufficient true-module centroid constraints.

Envelope inverses use the actual native normalization, in an explicitly
contained integer frame. Module centroids use current artwork union floor
midpoints and the existing certified inner polygon. This grants no source,
artwork, consumption or full-domain feasibility/optimality authority.
"""
from collections import Counter
from fractions import Fraction

from ortools.sat.python import cp_model

from clay.layout.place.module_relation_constraints import (
    _activity, _append_module, _bytes, _coordinate_bounds, _module_inventory,
    _normalization_domain, _require,
)
from clay.layout.reference_quality_contract import ReferenceQualityContract, RigidQualityGeometryIR
from clay.layout.reference_quality_metrics import normalize_point


_PREFIX = "reference_quality_location:"
_PPM = 1_000_000
_SAFE = (1 << 62) - 1


def _native_interval(low, high, minimum_ppm, maximum_ppm):
    """Exact inverse interval of the native integer normalizer on its frame."""
    _normalization_domain((low, 0, high, 1))
    _require(type(minimum_ppm) is int and type(maximum_ppm) is int and minimum_ppm <= maximum_ppm,
             "native normalized interval limits must be ordered exact integers")
    def first_at_least(level):
        left, right = low, high+1
        while left < right:
            middle = (left+right)//2
            value = normalize_point((middle, 0), (low, 0, high, 1))[0]
            if value >= level:
                right = middle
            else:
                left = middle+1
        return left
    first, last = first_at_least(minimum_ppm), first_at_least(maximum_ppm+1)-1
    return None if first > last else (first, last)


def _interval_evidence(frame, axis, interval):
    if interval is None:
        return []
    low, high = interval
    points = {low, high}
    if low > frame[axis]:
        points.add(low-1)
    if high < frame[axis+2]:
        points.add(high+1)
    return [[value, normalize_point((value, 0), (frame[axis], 0, frame[axis+2], 1))[0]]
            for value in sorted(points)]


def _population_extrema(candidate, target, occurrences, indices, bounds, frame):
    variables = [None]*4
    for axis in (0, 1):
        name = f"{_PREFIX}{target.scope}:{target.target_id}:{'xy'[axis]}"
        points = []
        for row in occurrences:
            _activity(max(map(abs, bounds[row.owner_id][axis])), row.center_offset_nm[axis],
                      max(abs(frame[axis]), abs(frame[axis+2])))
            points.append(candidate.GetIntVarFromProtoIndex(indices[row.owner_id][axis]) + row.center_offset_nm[axis])
        low = candidate.NewIntVar(frame[axis], frame[axis+2], name+":minimum")
        high = candidate.NewIntVar(frame[axis], frame[axis+2], name+":maximum")
        candidate.AddMinEquality(low, points).WithName(name+":actual_minimum")
        candidate.AddMaxEquality(high, points).WithName(name+":actual_maximum")
        variables[axis], variables[axis+2] = low.Index(), high.Index()
    return variables


def _append_envelope(candidate, scope, target, variable_indices, occurrence_count, frame, limit):
    edges = []
    for edge, (index, reference) in enumerate(zip(variable_indices, target.envelope_ppm)):
        axis, start = edge % 2, len(candidate.Proto().constraints)
        minimum, maximum = reference-limit, reference+limit
        interval = _native_interval(frame[axis], frame[axis+2], minimum, maximum)
        variable = candidate.GetIntVarFromProtoIndex(index)
        name = f"{_PREFIX}{scope}:{target.target_id if scope != 'module' else target.module_id}:edge:{edge}"
        if interval is None:
            # A false linear constraint retains a replayable contradiction;
            # Add(False) would introduce an unsupported empty BoolOr primitive.
            candidate.Add(variable >= frame[axis+2]+1).WithName(name+":empty_native_interval")
        else:
            candidate.Add(variable >= interval[0]).WithName(name+":lower")
            candidate.Add(variable <= interval[1]).WithName(name+":upper")
        edges.append({"edge": edge, "variable_index": index, "reference_ppm": reference,
            "accepted_ppm": [minimum, maximum], "interval_nm": None if interval is None else list(interval),
            "native_boundary_evaluations": _interval_evidence(frame, axis, interval),
            "constraint_indices": list(range(start, len(candidate.Proto().constraints)))})
    return {"scope": scope, "target_id": target.target_id if scope != "module" else target.module_id,
            "occurrence_count": occurrence_count, "edges": edges}


def _append_module_centroid(candidate, target, module, frame, polygon):
    from clay.layout.place.generated_reference_quality import _integer_halfplane, _rational
    axes = []
    for axis, index in enumerate(module["midpoint_variable_indices"]):
        coefficient = Fraction(_PPM, frame[axis+2]-frame[axis])
        constant = -coefficient*frame[axis]-target.centroid_ppm[axis]
        axes.append({"constant": _rational(constant), "terms": [
            {"coordinate_index": index, "coefficient": _rational(coefficient)}]})
    planes = []
    for number, facet in enumerate(polygon["facets"]):
        coefficients, rhs = Counter(), Fraction(facet["c"])
        for factor, axis in zip((facet["a"], facet["b"]), axes):
            rhs -= factor*Fraction(*axis["constant"])
            for term in axis["terms"]:
                coefficients[term["coordinate_index"]] += factor*Fraction(*term["coefficient"])
        plane = _integer_halfplane(dict(coefficients), rhs)
        activity = abs(plane["upper_bound"])
        for index, coefficient in zip(plane["variable_indices"], plane["coefficients"]):
            domain = candidate.Proto().variables[index].domain
            activity += abs(coefficient)*max(abs(int(domain[0])), abs(int(domain[-1])))
        _require(activity <= _SAFE, "module centroid facet activity exceeds int64 safety")
        expression = sum(coefficient*candidate.GetIntVarFromProtoIndex(index)
            for index, coefficient in zip(plane["variable_indices"], plane["coefficients"]))
        constraint_index = len(candidate.Proto().constraints)
        candidate.Add(expression <= plane["upper_bound"]).WithName(
            f"{_PREFIX}module:{target.module_id}:centroid_facet:{number}")
        planes.append({**plane, "constraint_index": constraint_index, "activity_bound": activity})
    return {"scope": "module", "target_id": target.module_id, "occurrence_count": len(target.occurrence_ids),
            "reference_centroid_ppm": list(target.centroid_ppm), "axes": axes, "facet_planes": planes}


def append_quality_location_constraints(model, *, coordinate_indices, geometry, contract, frame_nm):
    """Atomically append every envelope and the remaining module centroid facets."""
    from clay.layout.place.generated_reference_quality import _AXIS_ERROR, _inner_polygon, _rational
    if type(model) is not cp_model.CpModel:
        raise TypeError("quality location compiler requires an exact CP model")
    if type(geometry) is not RigidQualityGeometryIR or type(contract) is not ReferenceQualityContract:
        raise TypeError("quality location compiler requires exact geometry and public contract")
    geometry.__post_init__()
    contract.__post_init__()
    normalization = _normalization_domain(frame_nm)
    before = _bytes(model)
    _require(not any(row.name.startswith(_PREFIX) for row in model.Proto().variables)
             and not any(row.name.startswith(_PREFIX) for row in model.Proto().constraints),
             "quality location model is already augmented")
    bounds = _coordinate_bounds(model, coordinate_indices, {row[0] for row in geometry.owners})
    modules = _module_inventory(geometry, contract, bounds, frame_nm)
    candidate, polygon = model.Clone(), _inner_polygon()
    for module in modules:
        _append_module(candidate, module, coordinate_indices, frame_nm, prefix=_PREFIX+"module:")
    envelopes, centroids = [], []
    populations = {("logical", row.logical_id): (row.logical_id,) for row in contract.logical_population}
    populations.update({("class", row.class_id): row.logical_ids for row in contract.class_population})
    for target in contract.location_targets:
        logicals = populations[target.scope, target.target_id]
        occurrences = tuple(row for row in geometry.occurrences if row.logical_id in logicals)
        variables = _population_extrema(candidate, target, occurrences, coordinate_indices, bounds, frame_nm)
        envelopes.append(_append_envelope(candidate, target.scope, target, variables,
            len(occurrences), frame_nm, contract.envelope_limit_ppm))
    for target, module in zip(contract.modules, modules):
        _require(target.module_id == module["module_id"], "location module inventory order differs")
        envelopes.append(_append_envelope(candidate, "module", target, module["bbox_variable_indices"],
            len(target.occurrence_ids), frame_nm, contract.envelope_limit_ppm))
        centroids.append(_append_module_centroid(candidate, target, module, frame_nm, polygon))
    prefix = candidate.Clone()
    del prefix.Proto().variables[len(model.Proto().variables):]
    del prefix.Proto().constraints[len(model.Proto().constraints):]
    _require(_bytes(prefix) == before and _bytes(model) == before,
             "quality location compilation changed original CP prefix identity")
    model.Proto().CopyFrom(candidate.Proto())
    return {"schema": "clay.quality-location-constraints/v1",
        "scope": "EXACT_CONTAINED_ENVELOPES_AND_SUFFICIENT_MODULE_CENTROIDS", "complete_quality_equivalence": False,
        "frame_nm": list(frame_nm), "normalization": normalization,
        "containment": "ALL_ACTUAL_MODULE_UNIONS_AND_OCCURRENCE_EXTREMA_INSIDE_FRAME",
        "envelope_count": len(envelopes), "module_centroid_count": len(centroids),
        "occurrence_count": len(geometry.occurrences), "logical_count": len(contract.logical_population),
        "class_count": len(contract.class_population), "contract_sha256": contract.sha256,
        "geometry_sha256": geometry.sha256, "envelopes": envelopes, "modules": modules,
        "metrics": centroids, "polygon": polygon, "axis_affine_error_strict_bound": _rational(_AXIS_ERROR)}


def replay_location_search(report, centroid_search, location_search, solution, contract):
    """Recheck actual metric output against both certified centroid populations."""
    from clay.layout.place.generated_reference_quality import _AXIS_ERROR, _RADIUS
    native = {("logical", row["logical_instance"]): row for row in report["logical_devices"]}
    native.update({("class", row["reference_master"]): row for row in report["location_classes"]})
    native.update({("module", row["group_id"]): row for row in report["module_locations"]})
    _require({(row["scope"], row["target_id"]) for row in location_search["envelopes"]} == set(native),
             "complete location replay denominator differs")
    for search in (centroid_search, location_search):
        for metric in search["metrics"]:
            affine = tuple(Fraction(*axis["constant"]) + sum(Fraction(*term["coefficient"])*solution[term["coordinate_index"]]
                for term in axis["terms"]) for axis in metric["axes"])
            row = native[metric["scope"], metric["target_id"]]
            delta = tuple(row["current_centroid_normalized_ppm"][axis]-metric["reference_centroid_ppm"][axis] for axis in (0, 1))
            _require(all(abs(Fraction(delta[axis])-affine[axis]) < _AXIS_ERROR for axis in (0, 1))
                     and all(facet["a"]*affine[0]+facet["b"]*affine[1] <= facet["c"] for facet in search["polygon"]["facets"])
                     and sum(value*value for value in affine) <= _RADIUS*_RADIUS
                     and row["centroid_gap_ppm"] <= contract.centroid_limit_ppm,
                     "actual native centroid differs from certified sufficient affine search")
    _require(all(row["envelope_gap_ppm"] <= contract.envelope_limit_ppm for row in native.values()),
             "actual native envelope differs from complete compiled inverse intervals")


__all__ = ["append_quality_location_constraints", "replay_location_search"]
