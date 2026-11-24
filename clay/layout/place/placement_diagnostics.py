"""Canonical placement replay rows and preserved full-2D failure diagnostics."""
from __future__ import annotations

import hashlib
import json
import re
from itertools import combinations
from dataclasses import asdict, is_dataclass
from typing import Dict, Mapping, Sequence, Tuple


class Full2DCompactionDiagnosticError(ValueError):
    """A fail-closed full-2D compaction exit with preserved replay evidence."""

    def __init__(self, message: str, *, diagnostic: Mapping[str, object]) -> None:
        self.diagnostic = diagnostic
        self.diagnostic_preservation = diagnostic
        super().__init__(message)

def _full_hard_replay_row(
    row_id: str,
    constraint_name: str,
    *,
    source_constraint_ids: Tuple[str, ...] = (),
) -> Dict[str, object]:
    """Build one canonical hard-replay audit row."""

    return {
        "constraint_names": (constraint_name,),
        "row_id": row_id,
        "source_constraint_ids": source_constraint_ids,
    }

def _diagnostic_canonical_value(value: object) -> object:
    """Project model inputs into deterministic JSON-compatible values."""

    if is_dataclass(value) and not isinstance(value, type):
        return _diagnostic_canonical_value(asdict(value))
    if isinstance(value, Mapping):
        rows = (
            (
                _diagnostic_canonical_value(key),
                _diagnostic_canonical_value(item),
            )
            for key, item in value.items()
        )
        return tuple(sorted(rows, key=lambda row: repr(row[0])))
    if isinstance(value, (tuple, list)):
        return tuple(_diagnostic_canonical_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted(
            (_diagnostic_canonical_value(item) for item in value),
            key=repr,
        ))
    if value is None or type(value) in {bool, float, int, str}:
        return value
    raise TypeError(
        f"diagnostic model input has unsupported type {type(value).__name__}"
    )

def _diagnostic_safe_canonical_value(value: object) -> object:
    """Never let an invalid model input mask the exception being preserved."""

    try:
        return _diagnostic_canonical_value(value)
    except Exception as error:  # noqa: BLE001 - diagnostic boundary
        return {
            "capture_error_type": type(error).__name__,
            "input_type": type(value).__name__,
        }

def _full2d_compaction_position_rows(
    positions: object,
) -> Tuple[Tuple[str, Tuple[int, int]], ...]:
    """Best-effort canonical coordinate snapshot that never masks an exit."""

    try:
        if not isinstance(positions, Mapping):
            return ()
        return tuple(sorted(
            (
                str(name),
                (int(point[0]), int(point[1])),
            )
            for name, point in positions.items()
        ))
    except Exception:  # noqa: BLE001 - diagnostic capture must not mask the exit
        return ()

def _canonical_full_hard_failure_rows(
    values: object,
) -> Tuple[Dict[str, object], ...]:
    """Canonicalize named hard-checker rows without inventing source IDs."""

    source_prefix_constraints = {
        "aggregate_envelope_order": "aggregate_envelope_order",
        "hierarchical_alignment": "hierarchical_macro_alignment",
        "hierarchical_order": "hierarchical_macro_order",
    }
    if values is None:
        return ()
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError("full-hard failure rows require one exact sequence")
    rows = []
    for value in values:
        if isinstance(value, Mapping):
            if set(value) != {
                "constraint_names",
                "row_id",
                "source_constraint_ids",
            }:
                raise ValueError("full-hard failure row schema differs")
            row_id = value["row_id"]
            constraint_names = value["constraint_names"]
            source_constraint_ids = value["source_constraint_ids"]
        elif type(value) is str and value:
            prefix = value.partition(":")[0]
            if prefix in source_prefix_constraints and ":" in value:
                row_id = value
                constraint_names = (source_prefix_constraints[prefix],)
                source_constraint_ids = (value,)
            else:
                raise ValueError("full-hard failure string is not canonical")
        else:
            raise TypeError("full-hard failure row is malformed")
        if type(row_id) is not str or not row_id:
            raise TypeError("full-hard failure row ID must be one exact string")
        if (
            type(constraint_names) is not tuple
            or not constraint_names
            or constraint_names != tuple(sorted(set(constraint_names)))
            or any(type(name) is not str or not name for name in constraint_names)
        ):
            raise ValueError("full-hard constraint names must be canonical")
        if (
            type(source_constraint_ids) is not tuple
            or source_constraint_ids
            != tuple(sorted(set(source_constraint_ids)))
            or any(
                type(source_id) is not str or not source_id
                for source_id in source_constraint_ids
            )
        ):
            raise ValueError("full-hard source IDs must be canonical")
        rows.append({
            "constraint_names": constraint_names,
            "row_id": row_id,
            "source_constraint_ids": source_constraint_ids,
        })
    if len({str(row["row_id"]) for row in rows}) != len(rows):
        raise ValueError("full-hard failure row IDs must be unique")
    return tuple(sorted(rows, key=lambda row: str(row["row_id"])))

def _full2d_compaction_failure_diagnostic(
    *,
    error: Exception,
    state: Mapping[str, object],
) -> Dict[str, object]:
    seed_rows = tuple(state.get("seed_positions_nm", ()))
    candidate_rows = tuple(state.get("candidate_positions_nm", seed_rows))
    dimension_rows = tuple(state.get("dimensions_nm", ()))
    seed_by_name = dict(seed_rows)
    candidate_delta = tuple(
        (
            name,
            (
                point[0] - seed_by_name[name][0],
                point[1] - seed_by_name[name][1],
            ),
        )
        for name, point in candidate_rows
        if name in seed_by_name
    )

    graph = state.get("graph")
    graph_failed_rows: Tuple[Dict[str, object], ...] = ()
    graph_status = "NOT_RUN"
    graph_sha256 = None
    pair_coverage = ()
    pair_constraint_mode_codes = ""
    pair_constraint_inventory = ()
    fixed_edges = ()
    live_disjunctions = ()
    from clay.layout.extract.fixed_incumbent_geometry import FixedIncumbentGeometryProof
    fixed_geometry_summary = None
    if type(graph) is FixedIncumbentGeometryProof:
        fixed_geometry_summary = {
            "schema": graph.payload()["schema"], "sha256": graph.sha256,
            "owner_count": len(graph.master_nodes), "atom_count": len(graph.nodes),
            "replay_kind": "complete_indexed_geometry",
        }
    if graph is not None and fixed_geometry_summary is None:
        graph_sha256 = graph.sha256
        pair_coverage = graph.pair_coverage
        pair_constraint_mode_codes = "".join(
            "F" if row.mode == "fixed_edge" else "D"
            for row in graph.pair_constraints
        )
        pair_constraint_inventory = tuple(
            (
                row.first,
                row.second,
                row.clearance_nm,
                row.mode,
                row.axis,
            )
            for row in graph.pair_constraints
        )
        fixed_edges = tuple(sorted(
            (
                {
                    "authority_kind": row.authority_kind,
                    "row_id": row.edge_id,
                    "source_constraint_ids": row.source_constraint_ids,
                }
                for row in graph.pair_constraints
                if row.mode == "fixed_edge"
            ),
            key=lambda row: str(row["row_id"]),
        ))
        live_disjunctions = tuple(sorted(
            (
                {
                    "authority_kind": row.authority_kind,
                    "row_id": (
                        f"four_way_disjunction:{row.first}<->{row.second}"
                    ),
                    "seed_hint": row.seed_hint or None,
                    "source_constraint_ids": row.source_constraint_ids,
                }
                for row in graph.pair_constraints
                if row.mode == "four_way_disjunction"
            ),
            key=lambda row: str(row["row_id"]),
        ))
    if graph is not None:
        graph_sha256 = graph.sha256
        dims = dict(dimension_rows)
        if "graph_status" in state:
            graph_failed_rows = tuple(state.get("graph_failed_rows", ()))
            graph_status = str(state["graph_status"])
        elif candidate_rows and set(dims) == {
            name for name, _ in candidate_rows
        }:
            boxes = {
                name: (
                    point[0],
                    point[1],
                    point[0] + dims[name][0],
                    point[1] + dims[name][1],
                )
                for name, point in candidate_rows
            }
            graph_failed_rows = graph.replay_diagnostic(boxes)
            graph_status = "FAIL" if graph_failed_rows else "PASS"
        else:
            graph_failed_rows = ({
                "constraint_names": ("master_inventory",),
                "row_id": "graph_input:master_inventory",
                "source_constraint_ids": (),
            },)
            graph_status = "FAIL"
    else:
        input_provenance = state.get("input_provenance", {})
        if isinstance(input_provenance, Mapping):
            fixed_edges = tuple(input_provenance.get("fixed_edges", ()))
            live_disjunctions = tuple(
                input_provenance.get("live_disjunctions", ())
            )
            input_pair_coverage = []
            for row in fixed_edges:
                match = re.fullmatch(
                    r"fixed_axis:(.+)->(.+):(horizontal|vertical)",
                    str(row.get("row_id", "")),
                )
                if match is not None:
                    first, second = sorted(match.group(1, 2))
                    input_pair_coverage.append((
                        first, second, match.group(3)
                    ))
            for row in live_disjunctions:
                match = re.fullmatch(
                    r"four_way_disjunction:([^<]+)<->(.+)",
                    str(row.get("row_id", "")),
                )
                if match is not None:
                    input_pair_coverage.append((
                        match.group(1),
                        match.group(2),
                        "four_way_disjunction",
                    ))
            pair_coverage = tuple(sorted(input_pair_coverage))
            pair_constraint_mode_codes = "".join(
                "D" if row[2] == "four_way_disjunction" else "F"
                for row in pair_coverage
            )
        failure_stage = str(state.get("failure_stage", "input_capture"))
        error_edge_ids = getattr(error, "edge_ids", ())
        if (
            failure_stage == "graph_build"
            and type(error_edge_ids) is tuple
            and error_edge_ids
            and error_edge_ids == tuple(sorted(set(error_edge_ids)))
            and all(type(edge_id) is str and edge_id for edge_id in error_edge_ids)
        ):
            graph_failed_rows = tuple(
                {
                    "constraint_names": ("axis_provenance",),
                    "row_id": edge_id,
                    "source_constraint_ids": (),
                }
                for edge_id in error_edge_ids
            )
        else:
            graph_failed_rows = ({
                "constraint_names": (
                    ("diagnostic_contract_gap",)
                    if not seed_rows or not candidate_rows
                    else (f"full2d_graph_{failure_stage}",)
                ),
                "row_id": f"full2d_stage:{failure_stage}",
                "source_constraint_ids": (),
            },)
        graph_status = "FAIL"

    hard_status = str(state.get("full_hard_status", "NOT_RUN"))
    hard_failed_rows = tuple(state.get("full_hard_failed_rows", ()))
    if hard_status == "FAIL":
        if not hard_failed_rows:
            hard_failed_rows = ({
                "constraint_names": ("diagnostic_contract_gap",),
                "row_id": "full_hard_diagnostic:missing_named_rows",
                "source_constraint_ids": (),
            },)
    elif hard_status == "ERROR":
        hard_failed_rows = ({
            "constraint_names": ("diagnostic_contract_gap",),
            "row_id": "full_hard_diagnostic:callback_error",
            "source_constraint_ids": (),
        },)
        hard_status = "FAIL"

    stage_input_payload = dict(state.get("stage_input", {}))
    stage_input_payload.update({
        "dimensions_nm": dimension_rows,
        "graph_sha256": graph_sha256,
        "hard_replay_inventory": tuple(
            state.get("full_hard_inventory", ())
        ),
        "hard_replay_spec": _diagnostic_safe_canonical_value(
            state.get("full_hard_spec", {})
        ),
        "seed_positions_nm": seed_rows,
    })
    stage_input_sha256 = hashlib.sha256(json.dumps(
        stage_input_payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")).hexdigest()
    diagnostic = {
        "schema": "clay.full2d-compaction-failure/v3",
        "failure_candidate_positions_nm": candidate_rows,
        "candidate_delta_from_seed_nm": candidate_delta,
        "rejected_candidate_positions_nm": tuple(
            state.get("rejected_candidate_positions_nm", ())
        ),
        "graph_replay": {
            "failed_rows": graph_failed_rows,
            "status": graph_status,
        },
        "full_hard_replay": {
            "failed_rows": hard_failed_rows,
            "status": hard_status,
        },
        "stage_input": stage_input_payload,
        "stage_input_sha256": stage_input_sha256,
        "semantic_snapshot": {
            "graph_sha256": graph_sha256,
            "pair_coverage": pair_coverage,
            "pair_constraint_inventory": pair_constraint_inventory,
            "pair_constraint_inventory_sha256": hashlib.sha256(json.dumps(
                pair_constraint_inventory,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")).hexdigest(),
            "pair_constraint_mode_codes": pair_constraint_mode_codes,
        },
        "provenance_audit": {
            "fixed_edges": fixed_edges,
            "live_disjunctions": live_disjunctions,
        },
        "solver_evidence": tuple(state.get("solver_evidence", ())),
    }
    if fixed_geometry_summary is not None:
        diagnostic["semantic_snapshot"]["fixed_incumbent_geometry"] = fixed_geometry_summary
    return diagnostic

def _full2d_input_provenance(
    positions: Mapping[str, Tuple[int, int]],
    *,
    dims: Mapping[str, Tuple[int, int]],
    minimum_clearance_nm: int,
    pair_axis_authority: object,
    pair_axis_preference: object,
    overlap_compatible_pairs: object,
) -> Dict[str, object]:
    """Preserve declaration-side pair status before graph canonicalization."""

    if not isinstance(positions, Mapping) or not isinstance(dims, Mapping):
        return {"fixed_edges": (), "live_disjunctions": ()}
    names = tuple(sorted(
        str(name) for name in positions if str(name) in dims
    ))
    known = set(names)
    fixed_edge_sources: Dict[str, set[str]] = {}
    if isinstance(pair_axis_authority, tuple):
        for row in pair_axis_authority:
            if (
                type(row) is tuple
                and len(row) == 6
                and type(row[0]) is str
                and type(row[1]) is str
                and row[0] < row[1]
                and set(row[:2]) <= known
                and row[2] in {"horizontal", "vertical"}
                and type(row[3]) is str
                and type(row[4]) is str
                and set(row[3:5]) == set(row[:2])
                and row[3] != row[4]
                and type(row[5]) is tuple
                and row[5]
                and all(type(source_id) is str and source_id for source_id in row[5])
            ):
                first, second, axis, lower, upper, source_ids = row
                row_id = f"fixed_axis:{lower}->{upper}:{axis}"
                fixed_edge_sources.setdefault(row_id, set()).update(source_ids)
    compatible = {
        pair
        for pair in overlap_compatible_pairs
        if (
            type(pair) is tuple
            and len(pair) == 2
            and type(pair[0]) is str
            and type(pair[1]) is str
            and pair[0] < pair[1]
            and set(pair) <= known
        )
    } if isinstance(overlap_compatible_pairs, tuple) else set()
    preference_by_pair = {
        row[:2]: row[2]
        for row in pair_axis_preference
        if (
            type(row) is tuple
            and len(row) == 3
            and type(row[0]) is str
            and type(row[1]) is str
            and row[0] < row[1]
            and set(row[:2]) <= known
            and row[2] in {"horizontal", "vertical"}
        )
    } if isinstance(pair_axis_preference, tuple) else {}
    live_disjunctions = []
    for first, second in combinations(names, 2):
        if (first, second) in compatible:
            continue
        seed_hint = None
        try:
            ax, ay = positions[first]
            bx, by = positions[second]
            aw, ah = dims[first]
            bw, bh = dims[second]
            clearance = int(minimum_clearance_nm)
            x_candidates = []
            if ax + aw + clearance <= bx:
                x_candidates.append((first, second))
            if bx + bw + clearance <= ax:
                x_candidates.append((second, first))
            y_candidates = []
            if ay + ah + clearance <= by:
                y_candidates.append((first, second))
            if by + bh + clearance <= ay:
                y_candidates.append((second, first))
            preferred = preference_by_pair.get((first, second))
            if preferred == "vertical" and y_candidates:
                lower, upper = y_candidates[0]
                axis = "vertical"
            elif preferred == "horizontal" and x_candidates:
                lower, upper = x_candidates[0]
                axis = "horizontal"
            elif x_candidates:
                lower, upper = x_candidates[0]
                axis = "horizontal"
            elif y_candidates:
                lower, upper = y_candidates[0]
                axis = "vertical"
            else:
                lower = upper = axis = None
            if axis is not None:
                seed_hint = (
                    lower,
                    upper,
                    axis,
                    f"implicit_seed_axis:{lower}->{upper}:{axis}",
                )
        except (KeyError, TypeError, ValueError):
            seed_hint = None
        live_disjunctions.append({
            "authority_kind": "physical_nonoverlap",
            "row_id": f"four_way_disjunction:{first}<->{second}",
            "seed_hint": seed_hint,
            "source_constraint_ids": ("physical_nonoverlap",),
        })
    return {
        "fixed_edges": tuple(
            {
                "authority_kind": "declared_hard",
                "row_id": row_id,
                "source_constraint_ids": tuple(sorted(source_ids)),
            }
            for row_id, source_ids in sorted(fixed_edge_sources.items())
        ),
        "live_disjunctions": tuple(sorted(
            live_disjunctions, key=lambda row: str(row["row_id"])
        )),
    }
