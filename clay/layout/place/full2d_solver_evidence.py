"""Semantic validation for diagnostic full-2D CP-SAT solve observations."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
from typing import Any


_OBJECTIVE_NAMES = ("x_max", "y_max", "movement_sum")
_DIRECTION_NAMES = (
    "first_before_second_horizontal",
    "first_before_second_vertical",
    "second_before_first_horizontal",
    "second_before_first_vertical",
)
_ACTIVE_STATUS_CODES = (
    (0, "UNAVAILABLE_NO_INCUMBENT"),
    (1, "ACTIVE_ZERO_SLACK"),
    (2, "SATISFIED_POSITIVE_SLACK"),
    (3, "ENFORCED_POSITIVE_SLACK"),
)
_NECESSITY_STATUS_CODES = ((0, "UNMEASURABLE_SINGLE_INCUMBENT"),)
_ROW_SCHEMA = (
    "pair_constraint_index",
    "active_status_code",
    "selected_literal_mask",
    "directional_slacks_nm",
    "necessity_status_code",
)
_INCUMBENT_STATUSES = frozenset(("OPTIMAL", "FEASIBLE"))
_NO_INCUMBENT_STATUSES = frozenset(("INFEASIBLE", "UNKNOWN", "MODEL_INVALID"))
_COVER_ROW_SCHEMA = (*_ROW_SCHEMA[:2], "solver_literal_mask_or_replay_sentinel", *_ROW_SCHEMA[3:])


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_full2d_solver_evidence(
    values: object,
    *,
    graph_sha256: str,
    pair_constraint_modes: Sequence[str],
    graph: object | None = None,
) -> list[dict[str, Any]]:
    """Validate observations; with a typed graph, rebuild v3 cover and replay.

    Failure diagnostics can omit the graph and receive structural validation
    only. They confer no candidate authority. Final placement always supplies
    its actual graph. Original v3 disjunction masks are -3 (geometric replay,
    no original solver literal); actual literals belong to cover_constraints.
    """

    if not _is_sha256(graph_sha256):
        raise ValueError("full-2D solver evidence graph digest is malformed")
    modes = tuple(pair_constraint_modes)
    if any(mode not in {"fixed_edge", "four_way_disjunction"} for mode in modes):
        raise ValueError("full-2D solver evidence pair modes are malformed")
    if not isinstance(values, (tuple, list)):
        raise TypeError("full-2D solver evidence must be one row inventory")
    try:
        canonical = json.loads(json.dumps(
            values,
            sort_keys=True,
            separators=(",", ":"),
        ))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("full-2D solver evidence is not canonical JSON") from exc
    if type(canonical) is not list or len(canonical) > len(_OBJECTIVE_NAMES):
        raise ValueError("full-2D solver evidence solve inventory differs")

    expected_cover = None
    if graph is not None:
        from clay.layout.extract.compaction import Full2DCompactionGraph

        if (
            type(graph) is not Full2DCompactionGraph
            or graph.sha256 != graph_sha256
            or tuple(row.mode for row in graph.pair_constraints) != modes
        ):
            raise ValueError("full-2D solver evidence typed graph differs")
        if any(phase.get("schema") == "clay.full2d-solve-evidence/v3"
               for phase in canonical if type(phase) is dict):
            from clay.layout.place.full2d_pair_cover import build_full2d_pair_cover

            expected_cover = build_full2d_pair_cover(graph)

    terminal_no_incumbent_seen = False
    evidence_schema = None
    nonpair_identity = None
    cover_identity = None
    for expected_index, phase in enumerate(canonical):
        if terminal_no_incumbent_seen:
            raise ValueError(
                "full-2D solver evidence continues after a terminal no-incumbent phase"
            )
        common_fields = {
            "active_status_codes",
            "basis_summary",
            "candidate_positions_sha256",
            "constraint_row_schema",
            "constraint_scope",
            "constraints",
            "direction_order",
            "graph_sha256",
            "necessity_status_codes",
            "objective_name",
            "schema",
            "solve_index",
            "solver_status",
        }
        phase_schema = phase.get("schema") if type(phase) is dict else None
        expected_fields = set(common_fields)
        compressed = phase_schema == "clay.full2d-solve-evidence/v3"
        if phase_schema in {"clay.full2d-solve-evidence/v2", "clay.full2d-solve-evidence/v3"}:
            expected_fields.add("nonpair_constraint_registry")
        if compressed:
            expected_fields.update(("collision_encoding", "cover_constraints", "candidate_positions_nm"))
        if (
            type(phase) is not dict
            or set(phase) != expected_fields
            or phase_schema not in {
                "clay.full2d-solve-evidence/v1",
                "clay.full2d-solve-evidence/v2",
                "clay.full2d-solve-evidence/v3",
            }
        ):
            raise ValueError("full-2D solver evidence phase schema differs")
        if evidence_schema is None:
            evidence_schema = phase_schema
        elif phase_schema != evidence_schema:
            raise ValueError("full-2D solver evidence mixes schema versions")
        status = phase["solver_status"]
        if (
            phase["solve_index"] != expected_index
            or phase["objective_name"] != _OBJECTIVE_NAMES[expected_index]
            or phase["constraint_scope"] != "full2d_graph_pair_constraints"
            or phase["constraint_row_schema"] != list(_COVER_ROW_SCHEMA if compressed else _ROW_SCHEMA)
            or phase["direction_order"] != list(_DIRECTION_NAMES)
            or phase["active_status_codes"]
            != [list(row) for row in _ACTIVE_STATUS_CODES]
            or phase["necessity_status_codes"]
            != [list(row) for row in _NECESSITY_STATUS_CODES]
            or phase["graph_sha256"] != graph_sha256
            or status not in _INCUMBENT_STATUSES | _NO_INCUMBENT_STATUSES
            or phase["basis_summary"] != {
                "reason": "CP-SAT exposes no LP basis or constraint duals",
                "status": "UNAVAILABLE_CP_SAT_NO_BASIS",
            }
        ):
            raise ValueError("full-2D solver evidence phase authority differs")

        if phase_schema in {"clay.full2d-solve-evidence/v2", "clay.full2d-solve-evidence/v3"}:
            from clay.layout.place.full2d_nonpair_registry import (
                validate_full2d_nonpair_constraint_registry,
            )

            registry = validate_full2d_nonpair_constraint_registry(
                phase["nonpair_constraint_registry"]
            )
            identity = tuple(
                (
                    row["row_id"],
                    row["family"],
                    row["model_row_id"],
                    # An adjacency is one four-way obligation. Its chosen
                    # horizontal/vertical literal is an incumbent observation,
                    # not a change to the constraint across objective phases.
                    "joint" if row["family"] == "adjacency" else row["axis"],
                    tuple(row["subjects"]),
                    tuple(row["source_constraint_ids"]),
                    row["declaration_source"],
                    row["provenance_status"],
                    row["projection_source"],
                )
                for row in registry["rows"]
            )
            if nonpair_identity is None:
                nonpair_identity = identity
            elif identity != nonpair_identity:
                raise ValueError(
                    "full-2D solver evidence non-pair inventory changes by phase"
                )
            registry_has_incumbent = registry["incumbent_status"] == "AVAILABLE"
            if registry_has_incumbent != (status in _INCUMBENT_STATUSES):
                raise ValueError(
                    "full-2D solver evidence non-pair incumbent state differs"
                )

        has_incumbent = status in _INCUMBENT_STATUSES
        positions_sha256 = phase["candidate_positions_sha256"]
        if has_incumbent:
            if not _is_sha256(positions_sha256):
                raise ValueError("full-2D solver evidence lacks its incumbent")
        elif positions_sha256 is not None:
            raise ValueError(
                "full-2D solver evidence no-incumbent phase has a candidate digest"
            )

        replay_slacks = None
        if compressed:
            if type(phase["solve_index"]) is not int:
                raise ValueError("full-2D solver evidence solve index is malformed")
            encoding = phase["collision_encoding"]
            _validate_cover_inventory(encoding, graph_sha256=graph_sha256, modes=modes)
            if expected_cover is not None and encoding != expected_cover:
                raise ValueError("full-2D solver evidence cover differs from actual graph")
            if cover_identity is None:
                cover_identity = encoding
            elif encoding != cover_identity:
                raise ValueError("full-2D solver evidence cover changes by phase")
            positions = _validate_cover_observation(phase, has_incumbent=has_incumbent)
            if graph is not None and positions is not None:
                replay_slacks = _original_graph_slacks(graph, positions)

        rows = phase["constraints"]
        if type(rows) is not list or len(rows) != len(modes):
            raise ValueError("full-2D solver evidence does not cover graph pair rows")
        for pair_index, (row, mode) in enumerate(zip(rows, modes)):
            if (
                type(row) is not list
                or len(row) != len(_ROW_SCHEMA)
                or row[0] != pair_index
                or (compressed and type(row[0]) is not int)
                or type(row[1]) is not int
                or type(row[2]) is not int
                or type(row[3]) is not list
                or any(type(slack) is not int for slack in row[3])
                or row[4] != 0
                or (compressed and type(row[4]) is not int)
            ):
                raise ValueError("full-2D solver evidence constraint schema differs")
            active_code, selected_mask, slacks_nm = row[1:4]
            expected_sentinel = -1 if mode == "fixed_edge" else -3 if compressed else -2
            if not has_incumbent:
                if (
                    active_code != 0
                    or selected_mask != expected_sentinel
                    or slacks_nm
                ):
                    raise ValueError(
                        "full-2D solver evidence no-incumbent row is inconsistent"
                    )
                continue
            if mode == "fixed_edge":
                if selected_mask != -1 or len(slacks_nm) != 1 or slacks_nm[0] < 0:
                    raise ValueError(
                        "full-2D solver evidence fixed row is inconsistent"
                    )
                expected_active = 1 if slacks_nm[0] == 0 else 2
            elif compressed:
                if selected_mask != -3 or len(slacks_nm) != 4 or max(slacks_nm) < 0:
                    raise ValueError("full-2D solver evidence original disjunction replay differs")
                expected_active = 1 if max(slacks_nm) == 0 else 2
            else:
                if selected_mask not in range(1, 16) or len(slacks_nm) != 4:
                    raise ValueError(
                        "full-2D solver evidence disjunction row is inconsistent"
                    )
                selected_slacks = tuple(
                    slack
                    for bit, slack in enumerate(slacks_nm)
                    if selected_mask & (1 << bit)
                )
                if not selected_slacks or any(slack < 0 for slack in selected_slacks):
                    raise ValueError(
                        "full-2D solver evidence selected literal violates its row"
                    )
                expected_active = 1 if 0 in selected_slacks else 3
            if active_code != expected_active:
                raise ValueError(
                    "full-2D solver evidence active status contradicts selected slack"
                )
            if replay_slacks is not None and slacks_nm != replay_slacks[pair_index]:
                raise ValueError("full-2D solver evidence original slack does not replay graph")
        terminal_no_incumbent_seen = not has_incumbent
    return canonical


def _validate_cover_inventory(encoding: object, *, graph_sha256: str, modes: tuple[str, ...]) -> None:
    geometry_compressed = (type(encoding) is dict
                           and encoding.get("schema") == "clay.full2d-pair-cover/v2")
    fields = {"schema", "graph_sha256", "groups"}
    if geometry_compressed:
        fields.update(("original_atom_pair_count", "scope"))
    if (
        type(encoding) is not dict
        or set(encoding) != fields
        or encoding["schema"] not in {"clay.full2d-pair-cover/v1", "clay.full2d-pair-cover/v2"}
        or encoding["graph_sha256"] != graph_sha256
        or type(encoding["groups"]) is not list
    ):
        raise ValueError("full-2D solver evidence cover inventory schema differs")
    if geometry_compressed and (
        type(encoding["original_atom_pair_count"]) is not int
        or encoding["original_atom_pair_count"] < len(modes)
        or encoding["scope"] != "exact union of every interacting original atom pair"
    ):
        raise ValueError("full-2D solver evidence compressed geometry inventory differs")
    covered = []
    owner_pairs = []
    for group in encoding["groups"]:
        if type(group) is not dict or set(group) != {"owners", "pair_constraint_indices", "forbidden_boxes_nm"}:
            raise ValueError("full-2D solver evidence cover group schema differs")
        owners = group["owners"]
        indices = group["pair_constraint_indices"]
        boxes = group["forbidden_boxes_nm"]
        if (
            type(owners) is not list or len(owners) != 2
            or any(type(name) is not str or not name for name in owners)
            or owners[0] >= owners[1]
            or type(indices) is not list or not indices
            or any(type(index) is not int or index < 0 or index >= len(modes) for index in indices)
            or indices != sorted(set(indices))
            or any(modes[index] != "four_way_disjunction" for index in indices)
            or type(boxes) is not list or not boxes
        ):
            raise ValueError("full-2D solver evidence cover ownership differs")
        canonical_boxes = []
        for box in boxes:
            if (
                type(box) is not list or len(box) != 4
                or any(type(value) is not int for value in box)
                or box[0] >= box[2] or box[1] >= box[3]
            ):
                raise ValueError("full-2D solver evidence forbidden box is malformed")
            canonical_boxes.append(tuple(box))
        if canonical_boxes != sorted(set(canonical_boxes)):
            raise ValueError("full-2D solver evidence forbidden boxes are not canonical")
        owner_pairs.append(tuple(owners))
        covered.extend(indices)
    if (
        owner_pairs != sorted(set(owner_pairs))
        or sorted(covered) != [i for i, mode in enumerate(modes) if mode == "four_way_disjunction"]
    ):
        raise ValueError("full-2D solver evidence cover does not partition original rows")


def _validate_cover_observation(phase: dict, *, has_incumbent: bool) -> dict | None:
    from clay.layout.place.full2d_pair_cover import cover_directional_slacks

    raw_positions = phase["candidate_positions_nm"]
    positions = None
    if has_incumbent:
        if type(raw_positions) is not list or not raw_positions:
            raise ValueError("full-2D solver evidence cover lacks candidate coordinates")
        names = []
        positions = {}
        for row in raw_positions:
            if (
                type(row) is not list or len(row) != 2
                or type(row[0]) is not str or not row[0]
                or type(row[1]) is not list or len(row[1]) != 2
                or any(type(value) is not int or value < 0 for value in row[1])
            ):
                raise ValueError("full-2D solver evidence candidate coordinate is malformed")
            names.append(row[0])
            positions[row[0]] = row[1]
        if names != sorted(set(names)):
            raise ValueError("full-2D solver evidence candidate coordinates are not canonical")
        digest = hashlib.sha256(json.dumps(raw_positions, sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest()
        if digest != phase["candidate_positions_sha256"]:
            raise ValueError("full-2D solver evidence candidate coordinates differ from digest")
    elif raw_positions is not None:
        raise ValueError("full-2D solver evidence no-incumbent phase has coordinates")
    expected_rows = []
    for group_index, group in enumerate(phase["collision_encoding"]["groups"]):
        owners = group["owners"]
        if positions is not None and any(owner not in positions for owner in owners):
            raise ValueError("full-2D solver evidence cover owner lacks coordinates")
        for box_index, box in enumerate(group["forbidden_boxes_nm"]):
            expected_rows.append((group_index, box_index, owners, box))
    rows = phase["cover_constraints"]
    if type(rows) is not list or len(rows) != len(expected_rows):
        raise ValueError("full-2D solver evidence actual cover literal inventory differs")
    for row, (group_index, box_index, owners, box) in zip(rows, expected_rows):
        if (
            type(row) is not list or len(row) != 4
            or any(type(value) is not int for value in row[:3])
            or row[:2] != [group_index, box_index]
            or type(row[3]) is not list or any(type(value) is not int for value in row[3])
        ):
            raise ValueError("full-2D solver evidence actual cover row schema differs")
        mask, slacks = row[2:]
        if not has_incumbent:
            if mask != -2 or slacks:
                raise ValueError("full-2D solver evidence unavailable cover row has literals")
            continue
        delta = tuple(positions[owners[1]][axis] - positions[owners[0]][axis] for axis in (0, 1))
        expected_slacks = list(cover_directional_slacks(box, delta))
        if mask not in range(1, 16) or slacks != expected_slacks:
            raise ValueError("full-2D solver evidence actual cover slacks differ")
        if any(slack < 0 for bit, slack in enumerate(slacks) if mask & (1 << bit)):
            raise ValueError("full-2D solver evidence selected cover literal violates its box")
    return positions


def _original_graph_slacks(graph: Any, positions: dict) -> list[list[int]]:
    sizes = dict(graph.sizes_nm)
    masters = dict(graph.master_sizes_nm)
    if set(positions) != set(masters):
        raise ValueError("full-2D solver evidence candidate master inventory differs")
    boxes = {
        name: (point[0], point[1], point[0] + masters[name][0], point[1] + masters[name][1])
        for name, point in positions.items()
    }
    if not graph.replay(boxes):
        raise ValueError("full-2D solver evidence candidate does not replay its original graph")
    points = {
        node: (positions[owner][0] + offset[0], positions[owner][1] + offset[1])
        for node, owner, offset in graph.node_owner_offsets_nm
    }
    slacks = []
    for row in graph.pair_constraints:
        if row.mode == "fixed_edge":
            axis = 0 if row.axis == "horizontal" else 1
            slacks.append([points[row.upper][axis] - points[row.lower][axis] - sizes[row.lower][axis] - row.clearance_nm])
        else:
            first, second = points[row.first], points[row.second]
            first_size, second_size = sizes[row.first], sizes[row.second]
            gap = row.clearance_nm
            slacks.append([
                second[0] - first[0] - first_size[0] - gap,
                second[1] - first[1] - first_size[1] - gap,
                first[0] - second[0] - second_size[0] - gap,
                first[1] - second[1] - second_size[1] - gap,
            ])
    return slacks


__all__ = ["validate_full2d_solver_evidence"]
