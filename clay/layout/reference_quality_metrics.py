"""Complete quality measurements for an explicitly declared reference contract.

This module has no filesystem, candidate receipt, or solver authority. Callers
independently establish original-source and current-materialization provenance.
Arithmetic follows native per-occurrence normalization and Python ties-to-even
rounding, including the final Euclidean centroid distance.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import math

from clay.layout.reference_quality_contract import ReferenceQualityContract, RigidQualityGeometryIR


def _point(point: object) -> tuple[int, int]:
    if type(point) is not tuple or len(point) != 2 or any(type(value) is not int for value in point):
        raise TypeError("point must be an exact pair of integer coordinates")
    return point


def _points(points: object) -> tuple[tuple[int, int], ...]:
    if type(points) is not tuple or not points:
        raise ValueError("occurrence point multiset must be a nonempty exact tuple")
    return tuple(_point(point) for point in points)


def _frame(box: object) -> tuple[int, int, int, int]:
    if (type(box) is not tuple or len(box) != 4 or any(type(value) is not int for value in box)
            or box[2] <= box[0] or box[3] <= box[1]):
        raise ValueError("normalization frame must be an exact positive integer bbox")
    return box


def normalize_point(point: tuple[int, int], bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    _point(point)
    _frame(bbox)
    return (int(round((point[0] - bbox[0]) * 1_000_000 / (bbox[2] - bbox[0]))),
            int(round((point[1] - bbox[1]) * 1_000_000 / (bbox[3] - bbox[1]))))


def point_average(points: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    _points(points)
    return tuple(int(round(sum(point[axis] for point in points) / len(points))) for axis in (0, 1))


def point_envelope(points: tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    _points(points)
    return (min(point[0] for point in points), min(point[1] for point in points),
            max(point[0] for point in points), max(point[1] for point in points))


def normalized_population(points: tuple[tuple[int, int], ...], bbox: tuple[int, int, int, int]):
    """Normalize each occurrence before taking its weighted mean and envelope."""
    normalized = tuple(normalize_point(point, bbox) for point in _points(points))
    return point_average(normalized), point_envelope(normalized)


def centroid_gap_ppm(first: tuple[int, int], second: tuple[int, int]) -> int:
    _point(first)
    _point(second)
    return int(round(math.hypot(first[0] - second[0], first[1] - second[1])))


def rank_pattern(points: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    _points(points)
    xs = {value: index for index, value in enumerate(sorted({point[0] for point in points}))}
    ys = {value: index for index, value in enumerate(sorted({point[1] for point in points}))}
    return tuple(sorted((xs[x], ys[y]) for x, y in points))


def _cluster_axis_ranks(values: Sequence[int], *, tolerance_nm: int) -> dict[int, int]:
    if type(tolerance_nm) is not int or tolerance_nm < 0:
        raise ValueError("axis rank tolerance must be an exact nonnegative integer")
    unique = tuple(sorted(set(values)))
    if not unique or any(type(value) is not int for value in unique):
        raise ValueError("axis rank values must be nonempty integers")
    ranks = {}
    cluster_min = unique[0]
    rank = 0
    for value in unique:
        if value - cluster_min > tolerance_nm:
            rank += 1
            cluster_min = value
        ranks[value] = rank
    return ranks


def physical_class_pattern(points_by_logical: dict[str, tuple[tuple[int, int], ...]], *, row_stagger_tolerance_nm: int):
    """Labelled exact-X and bounded-span clustered-Y occupancy multiset."""
    if type(points_by_logical) is not dict or not points_by_logical:
        raise ValueError("class pattern requires a nonempty exact mapping")
    rows = []
    for logical, points in sorted(points_by_logical.items()):
        if type(logical) is not str or not logical:
            raise TypeError("class pattern logical identity is malformed")
        rows.extend((logical, x, y) for x, y in _points(points))
    xs = _cluster_axis_ranks(tuple(x for _logical, x, _y in rows), tolerance_nm=0)
    ys = _cluster_axis_ranks(tuple(y for _logical, _x, y in rows), tolerance_nm=row_stagger_tolerance_nm)
    pattern = tuple(sorted((xs[x], ys[y], logical) for logical, x, y in rows))
    return pattern, len(set(ys.values())), len(xs)


def _box_center(box):
    return ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)


def _normalize_box(box, frame):
    return (*normalize_point((box[0], box[1]), frame), *normalize_point((box[2], box[3]), frame))


def _box_union(boxes):
    if not boxes:
        raise ValueError("module has no materialized blocks")
    return (min(box[0] for box in boxes), min(box[1] for box in boxes),
            max(box[2] for box in boxes), max(box[3] for box in boxes))


def _location(target, current_center, current_envelope, contract):
    centroid_gap = centroid_gap_ppm(target.centroid_ppm, current_center)
    envelope_gap = max(abs(first - second) for first, second in zip(target.envelope_ppm, current_envelope))
    return {
        "reference_centroid_normalized_ppm": list(target.centroid_ppm),
        "current_centroid_normalized_ppm": list(current_center),
        "reference_envelope_normalized_ppm": list(target.envelope_ppm),
        "current_envelope_normalized_ppm": list(current_envelope),
        "centroid_gap_ppm": centroid_gap, "envelope_gap_ppm": envelope_gap,
        "location_status": "pass" if centroid_gap <= contract.centroid_limit_ppm
            and envelope_gap <= contract.envelope_limit_ppm else "fail",
    }


def _pattern_fields(reference_pattern, current_pattern, reference_orientations, current_orientations):
    reference_rows = len({row[1] for row in reference_pattern})
    reference_columns = len({row[0] for row in reference_pattern})
    current_rows = len({row[1] for row in current_pattern})
    current_columns = len({row[0] for row in current_pattern})
    return {
        "reference_occurrence_count": len(reference_pattern),
        "current_occurrence_count": len(current_pattern),
        "reference_rows": reference_rows, "reference_columns": reference_columns,
        "current_rows": current_rows, "current_columns": current_columns,
        "reference_rank_pattern": [list(row) for row in reference_pattern],
        "current_rank_pattern": [list(row) for row in current_pattern],
        "reference_orientation_class_counts": [list(row) for row in reference_orientations],
        "current_orientation_class_counts": [list(row) for row in current_orientations],
        "pattern_status": "pass" if reference_pattern == current_pattern
            and reference_orientations == current_orientations else "fail",
    }


def evaluate_reference_quality(*, positions: Mapping[str, tuple[int, int]], geometry: RigidQualityGeometryIR,
        contract: ReferenceQualityContract, frame_nm: tuple[int, int, int, int],
        required_structure_status: str = "unverified", current_structure_status: str = "unverified") -> dict[str, object]:
    """Measure the full declared quality population; never issue physical signoff.

Structure statuses describe separately checked fixed artwork and current stamp
consumption. Omitting either leaves a blocker. The production caller must bind
those conditions to its actual materialization and subsequently run real gates.
"""
    if type(contract) is not ReferenceQualityContract or type(geometry) is not RigidQualityGeometryIR:
        raise TypeError("quality evaluation requires exact contract and live geometry IR")
    contract.__post_init__()
    geometry.__post_init__()
    _frame(frame_nm)
    if any(type(value) is not str or value not in {"pass", "fail", "unverified"}
           for value in (required_structure_status, current_structure_status)):
        raise ValueError("structure conditions must be pass, fail, or unverified")
    if not isinstance(positions, Mapping) or set(positions) != {row[0] for row in geometry.owners}:
        raise ValueError("current positions must exactly cover live owners")
    for owner, point in positions.items():
        if type(owner) is not str:
            raise TypeError("current owner identity must be a string")
        _point(point)
    expected_counts = {row.logical_id: len(row.occurrence_ids) for row in contract.logical_population}
    counts = Counter(row.logical_id for row in geometry.occurrences)
    if dict(counts) != expected_counts:
        raise ValueError("current occurrence logical/multiplicity population differs from reference contract")
    current_points = {name: [] for name in expected_counts}
    current_orientations = {name: Counter() for name in expected_counts}
    for row in geometry.occurrences:
        x, y = positions[row.owner_id]
        current_points[row.logical_id].append((x + row.center_offset_nm[0], y + row.center_offset_nm[1]))
        current_orientations[row.logical_id][row.orientation] += 1
    current_points = {name: tuple(sorted(points)) for name, points in current_points.items()}
    orientations = {name: tuple(sorted(counts.items())) for name, counts in current_orientations.items()}
    targets = {(row.scope, row.target_id): row for row in contract.location_targets}
    logical_patterns = {row.logical_id: row for row in contract.logical_patterns}
    logical_rows = []
    for population in contract.logical_population:
        name = population.logical_id
        pattern = logical_patterns[name]
        centroid, envelope = normalized_population(current_points[name], frame_nm)
        row = {"logical_instance": name, "reference_master": population.class_id,
               "logical_mapping_status": "declared_source_occurrence_mapping"}
        row.update(_location(targets[("logical", name)], centroid, envelope, contract))
        row.update(_pattern_fields(pattern.rank_pattern, rank_pattern(current_points[name]),
                                   pattern.orientation_counts, orientations[name]))
        logical_rows.append(row)
    class_patterns = {row.class_id: row for row in contract.class_patterns}
    class_rows = []
    for population in contract.class_population:
        pattern = class_patterns[population.class_id]
        points = tuple(point for name in population.logical_ids for point in current_points[name])
        centroid, envelope = normalized_population(points, frame_nm)
        current_pattern, _rows, _columns = physical_class_pattern(
            {name: current_points[name] for name in population.logical_ids},
            row_stagger_tolerance_nm=pattern.row_stagger_tolerance_nm)
        orientation_counts = Counter()
        for name in population.logical_ids:
            orientation_counts.update(current_orientations[name])
        row = {"reference_master": population.class_id, "logical_candidates": list(population.logical_ids),
               "logical_mapping_status": "unique" if len(population.logical_ids) == 1 else "set_ambiguous",
               "row_stagger_tolerance_nm": pattern.row_stagger_tolerance_nm}
        row.update(_location(targets[("class", population.class_id)], centroid, envelope, contract))
        row.update(_pattern_fields(pattern.labelled_rank_pattern, current_pattern,
                                   pattern.orientation_counts, tuple(sorted(orientation_counts.items()))))
        class_rows.append(row)
    placed_blocks = []
    for row in geometry.materialized_blocks:
        x, y = positions[row.owner_id]
        x0, y0, x1, y1 = row.bbox_offset_nm
        placed_blocks.append((row.logical_ids, (x + x0, y + y0, x + x1, y + y1)))
    module_rows = []
    module_centers = {}
    for target in contract.modules:
        logicals = set(target.logical_ids)
        selected = [(members, box) for members, box in placed_blocks if logicals.intersection(members)]
        if (any(not set(members) <= logicals for members, _box in selected)
                or {name for members, _box in selected for name in members} != logicals):
            raise ValueError("current materialized module ownership differs from reference contract")
        box = _box_union([box for _members, box in selected])
        center = normalize_point(_box_center(box), frame_nm)
        envelope = _normalize_box(box, frame_nm)
        module_centers[target.module_id] = center
        row = {"group_id": target.module_id, "reference_occurrence_count": len(target.occurrence_ids),
               "current_logical_count": len(target.logical_ids)}
        row.update(_location(target, center, envelope, contract))
        module_rows.append(row)
    relation_rows = []
    for target in contract.module_relations:
        axis = 0 if target.axis == "horizontal" else 1
        first = module_centers[target.first_module][axis]
        second = module_centers[target.second_module][axis]
        relation = "before" if first < second else "after" if first > second else "coincident"
        relation_rows.append({"first_group": target.first_module, "second_group": target.second_module,
            "axis": target.axis, "reference_relation": target.relation, "current_relation": relation,
            "status": "pass" if relation == target.relation else "fail"})
    relation_blockers = [f"module_relation:{row['first_group']}:{row['second_group']}:{row['axis']}"
                         for row in relation_rows if row["status"] != "pass"]
    blockers = ([f"logical_pattern:{row['logical_instance']}" for row in logical_rows if row["pattern_status"] != "pass"]
        + [f"logical_location:{row['logical_instance']}" for row in logical_rows if row["location_status"] != "pass"]
        + [f"parameter_pattern:{row['reference_master']}" for row in class_rows if row["pattern_status"] != "pass"]
        + ([] if required_structure_status == "pass" else ["required_structure:fixed_artwork"])
        + ([] if current_structure_status == "pass" else ["current_structure:materialized_consumption"])
        + [f"parameter_location:{row['reference_master']}" for row in class_rows if row["location_status"] != "pass"]
        + [f"module_location:{row['group_id']}" for row in module_rows if row["location_status"] != "pass"]
        + relation_blockers)
    return {"schema": "clay.reference-quality-evaluation/v1", "strict_status": "pass" if not blockers else "reject",
        "strict_blockers": blockers, "relaxed_status": "pass" if not relation_blockers else "reject",
        "relaxed_blockers": relation_blockers,
        "check_denominator": 2 * len(logical_rows) + 2 * len(class_rows) + len(module_rows) + len(relation_rows) + 2,
        "structure_condition": {"required_structure_status": required_structure_status,
                                "current_structure_status": current_structure_status},
        "logical_devices": logical_rows, "location_classes": class_rows,
        "module_locations": module_rows, "module_relations": relation_rows}
