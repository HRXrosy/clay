"""Exact greedy class rows conditional on the full labelled X-column mapping.

Every row anchor is the actual minimum of all its assigned occurrences. This
does not authenticate source, artwork or consumption, and imposes no X equality
by itself; the shared quality compiler separately enforces the exact X mapping.
"""
from collections import Counter

from ortools.sat.python import cp_model

from clay.layout.place.class_x_rank_constraints import _prepare_classes
from clay.layout.place.module_relation_constraints import _coordinate_bounds
from clay.layout.reference_quality_contract import ReferenceQualityContract, RigidQualityGeometryIR
from clay.layout.reference_quality_metrics import rank_pattern


_PREFIX = "reference_quality_class_y:"
_SAFE = (1 << 62) - 1


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _bytes(model):
    return model.Proto().SerializeToString(deterministic=True)


def _activity(*terms):
    _require(sum(abs(value) for value in terms) <= _SAFE,
             "class Y affine/tolerance activity exceeds int64 safety bound")


def _records(geometry, contract):
    if type(geometry) is not RigidQualityGeometryIR or type(contract) is not ReferenceQualityContract:
        raise TypeError("class Y requires exact current geometry and public contract")
    geometry.__post_init__()
    contract.__post_init__()
    expected = Counter({row.logical_id: len(row.occurrence_ids) for row in contract.logical_population})
    _require(Counter(row.logical_id for row in geometry.occurrences) == expected,
             "class Y complete logical occurrence population/count differs")


def validate_rigid_quality_patterns(geometry, contract):
    """Reject all translation-invariant logical-pattern and D4 count conflicts."""
    _records(geometry, contract)
    logicals, orientations = [], {}
    for target in contract.logical_patterns:
        selected = tuple(row for row in geometry.occurrences if row.logical_id == target.logical_id)
        pattern = rank_pattern(tuple(row.center_offset_nm for row in selected))
        counts = tuple(sorted(Counter(row.orientation for row in selected).items()))
        _require(pattern == target.rank_pattern and counts == target.orientation_counts,
                 f"fixed logical pattern/orientation invariant differs for {target.logical_id}: "
                 f"expected={(target.rank_pattern, target.orientation_counts)!r}, current={(pattern, counts)!r}")
        orientations[target.logical_id] = Counter(dict(counts))
        logicals.append({"logical_id": target.logical_id, "rank_pattern": [list(row) for row in pattern],
                        "orientation_counts": [list(row) for row in counts], "occurrence_count": len(selected)})
    classes = []
    targets = {row.class_id: row for row in contract.class_patterns}
    for population in contract.class_population:
        counts = Counter()
        for logical in population.logical_ids:
            counts.update(orientations[logical])
        counts = tuple(sorted(counts.items()))
        _require(counts == targets[population.class_id].orientation_counts,
                 f"fixed class orientation invariant differs for {population.class_id}")
        classes.append({"class_id": population.class_id, "orientation_counts": [list(row) for row in counts]})
    return {"scope": "FIXED_RIGID_LOGICAL_PATTERNS_AND_COMPLETE_D4_COUNTS", "complete_quality_equivalence": False,
        "geometry_sha256": geometry.sha256, "contract_sha256": contract.sha256,
        "logical_count": len(logicals), "class_count": len(classes), "occurrence_count": len(geometry.occurrences),
        "logical_patterns": logicals, "class_orientations": classes}


def _ranked_classes(geometry, contract, bounds):
    columns = _prepare_classes(geometry, contract, {owner: axes[0] for owner, axes in bounds.items()})
    targets = {row.class_id: row for row in contract.class_patterns}
    classes = []
    for item in columns:
        target = targets[item["class_id"]]
        tolerance, assigned = target.row_stagger_tolerance_nm, []
        _activity(tolerance, 1)
        for column in item["logical_columns"]:
            logical, owner, x = column["logical_id"], column["owner_id"], column["local_x_nm"]
            actual = sorted((row for row in geometry.occurrences if row.logical_id == logical
                and row.center_offset_nm[0] == x), key=lambda row: (row.center_offset_nm[1], row.orientation))
            ranks = sorted(y for target_x, y, label in target.labelled_rank_pattern
                           if label == logical and target_x == column["class_rank"])
            _require(len(actual) == len(ranks) == column["occurrence_count"],
                     "class Y full column occurrence/rank count differs")
            same_y = {}
            for ordinal, (row, rank) in enumerate(zip(actual, ranks)):
                y = row.center_offset_nm[1]
                _require(y not in same_y or same_y[y] == rank,
                         f"class Y equal native centers map to different row ranks for {logical}")
                same_y[y] = rank
                low, high = (value+y for value in bounds[owner][1])
                _activity(max(abs(low), abs(high)), y, tolerance, 1)
                assigned.append({"logical_id": logical, "owner_id": owner, "local_x_nm": x, "local_y_nm": y,
                    "orientation": row.orientation, "column_occurrence_ordinal": ordinal,
                    "x_rank": column["class_rank"], "y_rank": rank, "affine_domain_nm": [low, high]})
        _require(tuple(sorted((row["x_rank"], row["y_rank"], row["logical_id"]) for row in assigned))
                 == target.labelled_rank_pattern, "class Y complete labelled XY assignment differs")
        rows = []
        for rank in range(len({row[1] for row in target.labelled_rank_pattern})):
            selected = [row for row in assigned if row["y_rank"] == rank]
            _require(bool(selected), "class Y full rank has no occurrence minimum witness")
            low = min(row["affine_domain_nm"][0] for row in selected)
            high = max(row["affine_domain_nm"][1] for row in selected)
            for row in selected:
                _activity(max(abs(low), abs(high)), max(map(abs, bounds[row["owner_id"]][1])),
                          row["local_y_nm"], tolerance)
            rows.append({"rank": rank, "domain_nm": [low, high], "occurrence_count": len(selected)})
        for first, second in zip(rows, rows[1:]):
            _activity(max(map(abs, first["domain_nm"])), max(map(abs, second["domain_nm"])), tolerance, 1)
        classes.append({"class_id": item["class_id"], "row_stagger_tolerance_nm": tolerance,
            "occurrence_count": len(assigned), "rows": rows, "assigned_occurrences": assigned})
    return classes


def append_class_y_rank_constraints(model, *, coordinate_indices, geometry, contract):
    """Append exact full-class greedy Y rows under the separately enforced X mapping."""
    if type(model) is not cp_model.CpModel:
        raise TypeError("class Y compiler requires an exact CP model")
    _records(geometry, contract)
    before = _bytes(model)
    _require(not any(row.name.startswith(_PREFIX) for row in model.Proto().variables)
             and not any(row.name.startswith(_PREFIX) for row in model.Proto().constraints),
             "class Y model is already augmented")
    bounds = _coordinate_bounds(model, coordinate_indices, {row[0] for row in geometry.owners})
    classes = _ranked_classes(geometry, contract, bounds)
    candidate = model.Clone()
    for item in classes:
        start, anchors = len(candidate.Proto().constraints), []
        tolerance = item["row_stagger_tolerance_nm"]
        for row in item["rows"]:
            name = f"{_PREFIX}{item['class_id']}:{row['rank']}"
            anchor = candidate.NewIntVar(*row["domain_nm"], name + ":minimum")
            row["variable_index"] = anchor.Index()
            anchors.append(anchor)
            points = [candidate.GetIntVarFromProtoIndex(coordinate_indices[point["owner_id"]][1]) + point["local_y_nm"]
                      for point in item["assigned_occurrences"] if point["y_rank"] == row["rank"]]
            candidate.AddMinEquality(anchor, points).WithName(name + ":actual_minimum")
            for index, point in enumerate(points):
                candidate.Add(point <= anchor+tolerance).WithName(name + f":span:{index}")
        for rank in range(len(anchors)-1):
            candidate.Add(anchors[rank+1] >= anchors[rank]+tolerance+1).WithName(
                f"{_PREFIX}{item['class_id']}:next_minimum:{rank}")
        item["constraint_indices"] = list(range(start, len(candidate.Proto().constraints)))
    prefix = candidate.Clone()
    del prefix.Proto().variables[len(model.Proto().variables):]
    del prefix.Proto().constraints[len(model.Proto().constraints):]
    _require(_bytes(prefix) == before and _bytes(model) == before, "class Y compilation changed original CP prefix identity")
    model.Proto().CopyFrom(candidate.Proto())
    return {"schema": "clay.class-y-rank-constraints/v1", "scope": "EXACT_GREEDY_Y_CONDITIONAL_ON_FULL_CLASS_X",
        "requires_exact_class_x_mapping": True, "complete_quality_equivalence": False,
        "class_count": len(classes), "occurrence_count": len(geometry.occurrences),
        "contract_sha256": contract.sha256, "geometry_sha256": geometry.sha256, "classes": classes}


__all__ = ["append_class_y_rank_constraints", "validate_rigid_quality_patterns"]
