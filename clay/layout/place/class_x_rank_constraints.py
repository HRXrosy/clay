"""Append exact class X occupancy to an unchanged native CP model.

This compiler binds actual owner-local columns to labelled public rank columns.
It does not lower Y clustering, orientation, location, structure, or signoff.
Reference coordinates and physical occurrence IDs never select native ordinals.
"""
from collections import Counter

from ortools.sat.python import cp_model

from clay.layout.reference_quality_contract import ReferenceQualityContract, RigidQualityGeometryIR


_PREFIX = "reference_quality_class_x:"
_SAFE = (1 << 62) - 1


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _bytes(model):
    return model.Proto().SerializeToString(deterministic=True)


def _activity(*terms):
    _require(sum(abs(value) for value in terms) <= _SAFE,
             "class X affine activity exceeds int64 safety bound")


def _coordinate_bounds(model, indices, owners):
    _require(type(indices) is dict and set(indices) == set(owners),
             "class X coordinate owner inventory differs")
    proto = model.Proto()
    seen, bounds = set(), {}
    for owner, pair in sorted(indices.items()):
        _require(type(pair) is tuple and len(pair) == 2,
                 "class X coordinate indices must be exact axis pairs")
        for axis, index in enumerate(pair):
            _require(type(index) is int and 0 <= index < len(proto.variables) and index not in seen,
                     "class X coordinate indices are missing, aliased, or malformed")
            seen.add(index)
            variable = proto.variables[index]
            _require(variable.name == f"{'xy'[axis]}_{owner}",
                     "class X coordinate axis/owner identity differs")
            domain = tuple(variable.domain)
            _require(domain and len(domain) % 2 == 0
                     and all(low <= high for low, high in zip(domain[::2], domain[1::2]))
                     and all(domain[i-1] < domain[i] for i in range(2, len(domain), 2)),
                     "class X coordinate domain is malformed")
            if axis == 0:
                bounds[owner] = (int(domain[0]), int(domain[-1]))
    return bounds


def _prepare_classes(geometry, contract, bounds):
    actual = {}
    for row in geometry.occurrences:
        actual.setdefault(row.logical_id, []).append(row)
    expected = {row.logical_id: len(row.occurrence_ids) for row in contract.logical_population}
    _require({name: len(rows) for name, rows in actual.items()} == expected,
             "class X complete logical occurrence population/count differs")
    patterns = {row.class_id: row for row in contract.class_patterns}
    classes = []
    for population in contract.class_population:
        target = patterns[population.class_id].labelled_rank_pattern
        mappings = []
        for logical in population.logical_ids:
            rows = actual[logical]
            owner_ids = {row.owner_id for row in rows}
            _require(len(owner_ids) == 1, "class X logical rigid owner differs")
            owner = next(iter(owner_ids))
            local_counts = Counter(row.center_offset_nm[0] for row in rows)
            target_counts = Counter(rank for rank, _row_rank, name in target if name == logical)
            _require(len(local_counts) == len(target_counts),
                     f"class X column rank count differs for {logical}")
            for (local_x, count), (rank, target_count) in zip(sorted(local_counts.items()), sorted(target_counts.items())):
                _require(count == target_count,
                         f"class X column occurrence multiplicity differs for {logical}")
                low, high = (value + local_x for value in bounds[owner])
                _activity(max(abs(low), abs(high)), local_x)
                mappings.append({"logical_id": logical, "owner_id": owner,
                    "local_x_nm": local_x, "class_rank": rank, "occurrence_count": count,
                    "affine_domain_nm": [low, high]})
        ranks = sorted({row[0] for row in target})
        _require(ranks == list(range(len(ranks))), "class X normalized rank inventory differs")
        columns = []
        for rank in ranks:
            selected = [row for row in mappings if row["class_rank"] == rank]
            _require(bool(selected), "class X rank lacks its full occurrence column")
            # A union is a sound bound even when intersections are empty. The
            # equalities retain that contradiction for the actual CP solver.
            low = min(row["affine_domain_nm"][0] for row in selected)
            high = max(row["affine_domain_nm"][1] for row in selected)
            for row in selected:
                original = bounds[row["owner_id"]]
                _activity(max(abs(low), abs(high)), max(map(abs, original)), row["local_x_nm"])
            counts = Counter(name for column, _row, name in target if column == rank)
            _require(Counter({row["logical_id"]: row["occurrence_count"] for row in selected}) == counts,
                     "class X labelled column population differs")
            columns.append({"rank": rank, "domain_nm": [low, high],
                            "logical_counts": [list(item) for item in sorted(counts.items())]})
        for first, second in zip(columns, columns[1:]):
            _activity(max(map(abs, first["domain_nm"])), max(map(abs, second["domain_nm"])), 1)
        classes.append({"class_id": population.class_id, "column_count": len(columns),
            "occurrence_count": len(population.occurrence_ids), "columns": columns,
            "logical_columns": mappings})
    return classes


def append_class_x_rank_constraints(model, *, coordinate_indices, geometry, contract):
    """Validate every class, then atomically append its exact X-only constraints."""
    if type(model) is not cp_model.CpModel:
        raise TypeError("class X compiler requires an exact CP model")
    if type(geometry) is not RigidQualityGeometryIR or type(contract) is not ReferenceQualityContract:
        raise TypeError("class X compiler requires exact current geometry and public contract")
    geometry.__post_init__()
    contract.__post_init__()
    before = _bytes(model)
    _require(not any(row.name.startswith(_PREFIX) for row in model.Proto().variables)
             and not any(row.name.startswith(_PREFIX) for row in model.Proto().constraints),
             "class X model is already augmented")
    bounds = _coordinate_bounds(model, coordinate_indices, {row[0] for row in geometry.owners})
    classes = _prepare_classes(geometry, contract, bounds)
    metadata = {"schema": "clay.class-x-rank-constraints/v1", "scope": "EXACT_CLASS_X_MULTIPLICITY_ONLY",
        "complete_quality_equivalence": False, "class_count": len(classes),
        "occurrence_count": len(geometry.occurrences), "contract_sha256": contract.sha256,
        "geometry_sha256": geometry.sha256, "classes": classes}
    candidate = model.Clone()
    for item in classes:
        start = len(candidate.Proto().constraints)
        columns = []
        for column in item["columns"]:
            variable = candidate.NewIntVar(*column["domain_nm"],
                f"{_PREFIX}{item['class_id']}:{column['rank']}")
            column["variable_index"] = variable.Index()
            columns.append(variable)
        for number, row in enumerate(item["logical_columns"]):
            index = coordinate_indices[row["owner_id"]][0]
            candidate.Add(columns[row["class_rank"]]
                == candidate.GetIntVarFromProtoIndex(index) + row["local_x_nm"]).WithName(
                    f"{_PREFIX}{item['class_id']}:binding:{number}")
        for rank in range(len(columns)-1):
            candidate.Add(columns[rank+1] - columns[rank] >= 1).WithName(
                f"{_PREFIX}{item['class_id']}:order:{rank}")
        item["constraint_indices"] = list(range(start, len(candidate.Proto().constraints)))
    prefix = candidate.Clone()
    del prefix.Proto().variables[len(model.Proto().variables):]
    del prefix.Proto().constraints[len(model.Proto().constraints):]
    _require(_bytes(prefix) == before and _bytes(model) == before,
             "class X compilation changed the original CP prefix identity")
    model.Proto().CopyFrom(candidate.Proto())
    return metadata


__all__ = ["append_class_x_rank_constraints"]
