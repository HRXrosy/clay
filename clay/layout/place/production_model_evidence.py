"""Read-only evidence adapter for the production fixed-outline legalizer.

The input projection, emitted assumption guards, and live proto are separate
views.  This is a checker for a captured incumbent, never another placement
solve.  Unknown CP primitives fail closed instead of acquiring replay coverage.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import math
from typing import Any

from .constraint_model_promotion import (
    _data_sha256,
    build_constraint_model_promotion_receipt,
    build_production_constraint_inventory,
)


def _proto_sha(proto: Any) -> str:
    return hashlib.sha256(proto.SerializeToString(deterministic=True)).hexdigest()


def _validated_native_edge_rail(inputs: Mapping[str, Any]) -> Any:
    """Validate declared rail data against the current native operand frame.

    This is a data projection, not an authority constructor. The production
    acquisition still binds these complete operands to its fresh source frame.
    """
    reservation = inputs.get("edge_rail_reservation")
    if reservation is None:
        return None
    from .edge_rail_reservation import EdgeRailReservationIR

    if type(reservation) is not EdgeRailReservationIR:
        raise ValueError("native edge rail projection requires an exact typed reservation")
    reservation.payload()  # Revalidate complete declaration and nested inventory.
    if any(row.declaration_sha256 != hashlib.sha256(row.declaration_ref.encode("utf-8")).hexdigest()
           for row in reservation.demands):
        raise ValueError("native edge rail declaration digest differs from its complete reference")
    dimensions = {name: (width, height) for name, width, height in reservation.masters}
    if (dict(inputs["dims"]) != dimensions or set(inputs["seed"]) != set(dimensions)
            or any(type(value) is not tuple or len(value) != 2
                   or any(type(item) is not int for item in value)
                   for value in inputs["dims"].values())):
        raise ValueError("native edge rail master dimension inventory differs")
    grid = inputs["manufacturing_grid_nm"]
    if type(grid) is not int or grid != reservation.manufacturing_grid_nm:
        raise ValueError("native edge rail manufacturing grid differs")
    return reservation


def _native_assumption_row_ids(inputs: Mapping[str, Any]) -> set[str]:
    """Project presence without enumerating collision boxes or CP primitives."""
    identities: set[str] = set()
    for family, field in (
        ("order_pair", "order_pairs"), ("center_alignment", "center_alignment_pairs"),
        ("aggregate_order", "aggregate_envelope_orders"), ("adjacency", "adjacency_limits_nm"),
        ("axis_alignment", "axis_alignment_groups"), ("hierarchy_order", "hierarchical_macro_orders"),
        ("hierarchy_alignment", "hierarchical_macro_alignments"),
        ("row_bounded_cavity", "row_bounded_cavity_projections"),
    ):
        identities.update(f"{family}:{index}" for index, _ in enumerate(inputs[field]))
    for family, field in (
        ("vertical_order_pair", "vertical_order_pairs"),
        ("mosaic_guest_slot", "mosaic_guest_slot_projections"),
    ):
        identities.update(f"{family}:{index}" for index, _ in enumerate(inputs.get(field, ())))
    reservation = _validated_native_edge_rail(inputs)
    if reservation is not None:
        identities.update(f"edge_rail:{row.demand_id}" for row in reservation.demands)
    owners = set(inputs["seed"])
    for side in ("right", "left"):
        identities.update(f"horizontal_edge:{side}:{name}"
                          for name in set(inputs[f"{side}_edge"]) & owners)
    if inputs["aspect_obligation"]:
        identities.add("aspect_obligation")
    columns = inputs["floorplan_column_classes"]
    if any(left[1] < right[1] for left in columns for right in columns):
        identities.add("floorplan_column")
    cells = {(row[1], row[2]) for row in inputs["floorplan_column_groups"]}
    if any(a[0] == b[0] and a[1] < b[1] for a in cells for b in cells):
        identities.add("floorplan_column_group")
    refined = {row[0] for field in ("rigid_compound_footprints", "layered_collision_footprints")
               for row in inputs[field]}
    if any(a not in refined and b not in refined for a, b in inputs["pair_clearances_nm"]):
        identities.add("pair_clearance")
    compatible = set(inputs["overlap_compatible_pairs"])
    ordered_owners = sorted(owners)
    if ((refined or compatible)
            and any((a, b) not in compatible
                    for index, a in enumerate(ordered_owners) for b in ordered_owners[index + 1:])):
        identities.add("layered_collision")
    return identities


def project_native_obligation_metadata(inputs: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    """Return the complete input-derived native row denominator and provenance.

    The five existing metadata fields retain their historical serialized form.
    Neither emitted row labels nor caller-selected evidence domains are inputs.
    """
    projected = _metadata_for_native_row_ids(inputs, _native_assumption_row_ids(inputs))
    for name in _hard_background_names(inputs):
        row_id = f"hard_background:{name}"
        projected[row_id] = {
            "row_kind": "HARD_BACKGROUND", "row_id": row_id, "family": "hard_background",
            "obligation_id": name, "declaration_source": "clay/layout/place_blocks_2d.py",
        }
    return projected


def project_native_obligation_domains(inputs: Mapping[str, Any]) -> dict[str, str]:
    """Project applicability from validated native declarations, never row text."""
    metadata = project_native_obligation_metadata(inputs)
    return {
        row_id: (
            "GENERATED_BACKGROUND" if row["row_kind"] == "HARD_BACKGROUND"
            else "SOURCE_DECLARATION_AND_GENERATED_PHYSICAL" if row["family"] == "edge_rail"
            else "GENERATED_PACKING" if row_id == "layered_collision"
            else "REFERENCE_AND_GENERATED"
        )
        for row_id, row in metadata.items()
    }


def _expected_assumption_counts(inputs: Mapping[str, Any]) -> dict[str, int]:
    """Project obligations from geometry/declarations before examining CP rows."""
    from clay.layout import place_blocks_2d as placer

    ids = tuple(sorted(inputs["seed"]))
    expected: dict[str, int] = {}
    for family, field in (
        ("order_pair", "order_pairs"),
        ("center_alignment", "center_alignment_pairs"),
        ("aggregate_order", "aggregate_envelope_orders"),
        ("adjacency", "adjacency_limits_nm"),
    ):
        expected.update({f"{family}:{i}": 1 for i, _ in enumerate(inputs[field])})
    expected.update({f"vertical_order_pair:{i}": 1
                     for i, _ in enumerate(inputs.get("vertical_order_pairs", ()))})
    expected.update({
        f"axis_alignment:{i}": len(group) - 1
        for i, group in enumerate(inputs["axis_alignment_groups"])
    })
    expected.update({
        f"hierarchy_order:{i}": (
            len(row[5]) if row[4] in {
                "member_before_aggregate_center",
                "member_before_aggregate_center_with_track_floor",
            } else 1
        )
        for i, row in enumerate(inputs["hierarchical_macro_orders"])
    })
    expected.update({
        f"hierarchy_alignment:{i}": len(
            placer._hierarchical_macro_alignment_parts(row)[4]
        ) - 1
        for i, row in enumerate(inputs["hierarchical_macro_alignments"])
    })
    expected.update({
        f"row_bounded_cavity:{i}": 4 * len(row.guest_masters)
        for i, row in enumerate(inputs["row_bounded_cavity_projections"])
    })
    expected.update({f"mosaic_guest_slot:{i}": 4
                     for i, _ in enumerate(inputs.get("mosaic_guest_slot_projections", ()))})
    reservation = _validated_native_edge_rail(inputs)
    if reservation is not None:
        _total, guarded = reservation.lowering_counts()
        expected.update({f"edge_rail:{name}": count for name, count in guarded})
    for side in ("right", "left"):
        expected.update({
            f"horizontal_edge:{side}:{name}": 1
            for name in set(inputs[f"{side}_edge"]) & set(ids)
        })
    if inputs["aspect_obligation"]:
        expected["aspect_obligation"] = 2
    columns = inputs["floorplan_column_classes"]
    count = sum(left[1] < right[1] for left in columns for right in columns)
    if count:
        expected["floorplan_column"] = count
    cells = {(row[1], row[2]) for row in inputs["floorplan_column_groups"]}
    count = sum(a[0] == b[0] and a[1] < b[1] for a in cells for b in cells)
    if count:
        expected["floorplan_column_group"] = count
    refined_owners = {
        row[0] for field in ("rigid_compound_footprints", "layered_collision_footprints")
        for row in inputs[field]
    }
    count = sum(
        a not in refined_owners and b not in refined_owners
        for a, b in inputs["pair_clearances_nm"]
    )
    if count:
        expected["pair_clearance"] = count
    compatible = set(inputs["overlap_compatible_pairs"])
    if refined_owners or compatible:
        atoms = placer._collision_atoms_by_owner(
            dims=inputs["dims"],
            rigid_compound_footprints=inputs["rigid_compound_footprints"],
            layered_collision_footprints=inputs["layered_collision_footprints"],
        )
        pairs = [(a, b) for i, a in enumerate(ids) for b in ids[i + 1:]
                 if (a, b) not in compatible]
        if pairs:
            expected["layered_collision"] = sum(
                len(placer._pair_collision_forbidden_box_cover(
                    atoms[a], atoms[b],
                    required_nm=inputs["pair_clearances_nm"].get((a, b), inputs["gap_nm"]),
                )[0]) for a, b in pairs
            )
    return expected


def _hard_background_names(inputs: Mapping[str, Any]) -> tuple[str, ...]:
    refined = any(inputs[field] for field in (
        "rigid_compound_footprints", "layered_collision_footprints",
        "overlap_compatible_pairs",
    ))
    return (
        "derived_envelopes", "fixed_domain",
        *(("fixed_y",) if set(inputs["seed"]) - set(inputs["movable_y"]) else ()),
        "manufacturing_grid", "physical_origin",
        *(("bottom_origin",) if inputs["bottom_origin_lower_bounds_nm"] else ()),
        *(("exact_outline",) if inputs["exact_outline"] else ()),
        *(("no_overlap2d",) if not refined else ()),
    )


def _expected_obligation_metadata(
    inputs: Mapping[str, Any], expected_counts: Mapping[str, int],
) -> dict[str, dict[str, str]]:
    """Join independently counted primitives to the shared native denominator."""
    projected = {
        row_id: row for row_id, row in project_native_obligation_metadata(inputs).items()
        if row["row_kind"] == "ASSUMPTION"
    }
    if set(expected_counts) != set(projected):
        raise ValueError("independent production primitive row inventory differs from native denominator")
    return projected


def _metadata_for_native_row_ids(
    inputs: Mapping[str, Any], row_ids: set[str],
) -> dict[str, dict[str, str]]:
    """Project semantic provenance without consulting emitted trace records."""
    from clay.layout import place_blocks_2d as placer

    placement_source = "clay/layout/place_blocks_2d.py"
    intent_source = "clay/layout/intent.py"
    projection_source = "clay/layout/spatial/intent_projection.py"
    physical = {
        "layered_collision": ("physical_nonoverlap:canonical_layered_atoms", placement_source),
        "pair_clearance": ("physical_pair_clearance:canonical_master_pairs", placement_source),
        "floorplan_column": ("floorplan_column_order", placement_source),
        "floorplan_column_group": ("floorplan_column_group_order", placement_source),
        "aspect_obligation": ("declared_aspect_ratio", "clay/layout/policy.py"),
    }
    projected = {}
    for row_id in sorted(row_ids):
        family, _separator, raw_index = row_id.partition(":")
        if row_id in physical:
            obligation_id, source = physical[row_id]
        elif family == "horizontal_edge":
            obligation_id, source = row_id, intent_source
        elif family == "edge_rail":
            row = next(row for row in inputs["edge_rail_reservation"].demands
                       if row.demand_id == raw_index)
            obligation_id, source = row.demand_id, row.declaration_ref
        elif family == "axis_alignment":
            group = inputs["axis_alignment_groups"][int(raw_index)]
            obligation_id = "axis_alignment:" + ":".join(name for name, _offset in group)
            source = intent_source
        elif family == "center_alignment":
            left, right, axis = inputs["center_alignment_pairs"][int(raw_index)]
            obligation_id, source = f"center_alignment:{left}:{right}:{axis}", projection_source
        elif family == "hierarchy_order":
            obligation_id = placer._hierarchical_macro_order_obligation_id(
                inputs["hierarchical_macro_orders"][int(raw_index)]
            )
            source = projection_source
        elif family == "hierarchy_alignment":
            scope, axis, edge, _tolerance, _groups = placer._hierarchical_macro_alignment_parts(
                inputs["hierarchical_macro_alignments"][int(raw_index)]
            )
            prefix = placer.SUPPLEMENTAL_AGGREGATE_ALIGNMENT_SCOPE_PREFIX
            obligation_id = (scope.removeprefix(prefix) if scope.startswith(prefix)
                             else f"hierarchical_alignment:{scope}:{axis}:{edge}")
            source = projection_source
        elif family == "aggregate_order":
            before, after, axis, _left, _right = inputs["aggregate_envelope_orders"][int(raw_index)]
            obligation_id, source = f"aggregate_envelope_order:{before}:{after}:{axis}", projection_source
        elif family == "row_bounded_cavity":
            obligation_id = inputs["row_bounded_cavity_projections"][int(raw_index)].obligation_id
            source = projection_source
        elif family == "mosaic_guest_slot":
            obligation_id = inputs["mosaic_guest_slot_projections"][int(raw_index)].obligation_id
            source = "clay/layout/spatial/mosaic_guest_slot.py"
        elif family == "adjacency":
            left, right, maximum = inputs["adjacency_limits_nm"][int(raw_index)]
            obligation_id, source = f"adjacent:{left}:{right}:maximum_gap_nm={maximum}", intent_source
        elif family == "order_pair":
            left, right = inputs["order_pairs"][int(raw_index)]
            obligation_id, source = f"horizontal_order:{left}:{right}", intent_source
        elif family == "vertical_order_pair":
            lower, upper = inputs["vertical_order_pairs"][int(raw_index)]
            obligation_id, source = f"vertical_order:{lower}:{upper}", intent_source
        else:
            raise ValueError(f"unsupported production obligation metadata family: {family}")
        projected[row_id] = {
            "row_kind": "ASSUMPTION", "row_id": row_id, "family": family,
            "obligation_id": obligation_id, "declaration_source": source,
        }
    return projected


def _expected_primitive_count(inputs: Mapping[str, Any], assumptions: Mapping[str, int]) -> int:
    """Count lowering primitives from each input family, including auxiliaries.

    Shared hierarchy envelopes are counted by their semantic key. A dropped
    branch implication or envelope definition must not disappear unnoticed
    merely because its top-level assumption literal still exists.
    """
    from clay.layout.place_blocks_2d import _hierarchical_macro_alignment_parts

    count = sum(assumptions.values())
    reservation = inputs.get("edge_rail_reservation")
    if reservation is not None:
        rail_total, rail_guarded = reservation.lowering_counts()
        # A demand-pair clause is one primitive with both demand guards.
        count += rail_total - sum(value for _name, value in rail_guarded)
    n = len(inputs["seed"])
    movable = set(inputs["movable_y"])
    count += 3 + int(inputs["exact_outline"])  # physical min/max and origin
    count += n + len(movable)  # movement absolute-value definitions
    if inputs["bottom_origin_lower_bounds_nm"]:
        count += n + 2  # per-owner bounds, minimum slack, zero slack
    if "no_overlap2d" in _hard_background_names(inputs):
        count += 2 * n + 1
    else:
        count += 4 * assumptions.get("layered_collision", 0)
    count += 2 * n if inputs["manufacturing_grid_nm"] > 1 else 0
    count += 6 if inputs["aspect_obligation"] else 0
    count += 4 * assumptions.get("pair_clearance", 0)
    count += 16 * len(inputs["adjacency_limits_nm"])
    count += 4 * len(inputs["aggregate_envelope_orders"])
    envelopes = set()
    for row in inputs["hierarchical_macro_orders"]:
        _scope, before, after, axis, _strength, left, right = row
        envelopes.update(((axis, before, tuple(left)), (axis, after, tuple(right))))
    for row in inputs["hierarchical_macro_alignments"]:
        _scope, axis, _edge, tolerance, groups = _hierarchical_macro_alignment_parts(row)
        envelopes.update((axis, node, tuple(members)) for node, members in groups)
        if tolerance:
            count += 2 * (len(groups) - 1)  # delta and magnitude definitions
    count += 2 * len(envelopes)
    for group in inputs["hierarchical_cohesion_groups"]:
        count += 3 + (3 if set(group) & movable else 0)
    return count


def _expression_value(expression: Any, solution: tuple[int, ...]) -> int:
    return int(expression.offset) + sum(
        coefficient * (solution[index] if index >= 0 else -solution[-index - 1])
        for index, coefficient in zip(expression.vars, expression.coeffs)
    )


def _check_proto_solution(proto: Any, solution: tuple[int, ...]) -> list[dict[str, object]]:
    """Evaluate only the finite primitive vocabulary emitted by this adapter."""
    if type(solution) is not tuple or any(type(value) is not int for value in solution):
        raise TypeError("solver solution must contain exact integer values")
    if len(solution) != len(proto.variables):
        raise ValueError("solver solution variable inventory differs from model")

    def literal(value: int) -> bool:
        return solution[value] == 1 if value >= 0 else solution[-value - 1] == 0

    def active(row: Any) -> bool:
        return all(literal(value) for value in row.enforcement_literal)

    def in_domain(value: int, domain: Any) -> bool:
        return any(a <= value <= b for a, b in zip(domain[::2], domain[1::2]))

    checks = [{
        "row_id": f"cp_domain:{i}", "status": "PASS" if in_domain(value, var.domain) else "FAIL",
    } for i, (var, value) in enumerate(zip(proto.variables, solution))]
    checks.extend({
        "row_id": f"cp_assumption:{index}", "status": "PASS" if literal(index) else "FAIL",
    } for index in proto.assumptions)
    supported = {"linear", "lin_max", "int_mod", "int_prod", "bool_or", "bool_and",
                 "exactly_one", "at_most_one", "interval", "no_overlap_2d"}
    for index, row in enumerate(proto.constraints):
        kind = row.WhichOneof("constraint")
        status = "PASS"
        if kind not in supported:
            status = f"UNSUPPORTED_CP_PRIMITIVE:{kind}"
        elif active(row):
            body = getattr(row, kind)
            if kind == "linear":
                value = sum(c * solution[v] for v, c in zip(body.vars, body.coeffs))
                valid = in_domain(value, body.domain)
            elif kind == "lin_max":
                valid = _expression_value(body.target, solution) == max(
                    _expression_value(expr, solution) for expr in body.exprs
                )
            elif kind == "int_prod":
                # All affine factors and the target are exact signed integers.
                # Unbounded intermediate arithmetic prevents float rounding or
                # machine overflow from turning a wrong product into a match.
                valid = bool(body.exprs) and _expression_value(body.target, solution) == math.prod(
                    _expression_value(expr, solution) for expr in body.exprs
                )
            elif kind == "int_mod":
                numerator, denominator = (
                    _expression_value(expr, solution) for expr in body.exprs
                )
                quotient = (abs(numerator) // abs(denominator)) if denominator else 0
                if numerator * denominator < 0:
                    quotient = -quotient
                valid = bool(denominator) and _expression_value(body.target, solution) == (
                    numerator - denominator * quotient
                )
            elif kind in {"bool_or", "bool_and", "exactly_one", "at_most_one"}:
                values = [literal(value) for value in body.literals]
                valid = {"bool_or": any(values), "bool_and": all(values),
                         "exactly_one": sum(values) == 1, "at_most_one": sum(values) <= 1}[kind]
            elif kind == "interval":
                valid = (
                    _expression_value(body.size, solution) >= 0
                    and _expression_value(body.start, solution) + _expression_value(body.size, solution)
                    == _expression_value(body.end, solution)
                )
            else:
                rectangles = []
                for xi, yi in zip(body.x_intervals, body.y_intervals):
                    xrow, yrow = proto.constraints[xi], proto.constraints[yi]
                    if active(xrow) and active(yrow):
                        rectangles.append(tuple(_expression_value(expr, solution) for expr in (
                            xrow.interval.start, yrow.interval.start,
                            xrow.interval.end, yrow.interval.end,
                        )))
                valid = all(
                    a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]
                    for i, a in enumerate(rectangles) for b in rectangles[i + 1:]
                )
            if not valid:
                status = "FAIL"
        checks.append({"row_id": f"cp_primitive:{index}", "kind": kind, "status": status})
    return checks


class FixedOutlineProductionEvidenceContext:
    """Runtime capture reusable for a solved, complete, objective-free clone."""

    def __init__(self, *, model: Any, stage_inputs: Mapping[str, Any],
                 assumption_rows: tuple[Mapping[str, Any], ...],
                 coordinate_indices: Mapping[str, tuple[int, int]]) -> None:
        from clay.layout.place_blocks_2d import _diagnostic_canonical_value

        # MappingProxyType defaults are normalized by the caller before copying.
        self._inputs = deepcopy(dict(stage_inputs))
        self.stage_input_data = _diagnostic_canonical_value(self._inputs)
        self.stage_input_sha256 = _data_sha256(self.stage_input_data)
        self._model = model.Clone()
        self._indices = dict(coordinate_indices)
        self._expected_counts = _expected_assumption_counts(self._inputs)
        self._expected_metadata = _expected_obligation_metadata(self._inputs, self._expected_counts)
        self._expected_primitive_count = _expected_primitive_count(self._inputs, self._expected_counts)
        self._assumption_rows = tuple(dict(row) for row in assumption_rows)
        self._background = _hard_background_names(self._inputs)
        # Capturing a model must not prevent an independent UNSAT diagnostic.
        # Completeness is mandatory only when a caller requests promotion.
        self._bindings: dict[str, list[str]] | None = None

    def _materialized_bindings(self, proto: Any) -> dict[str, list[str]]:
        if set(self._indices) != set(self._inputs["seed"]):
            raise ValueError("stage geometry coordinate inventory differs")
        for name, (x_index, y_index) in self._indices.items():
            width, height = self._inputs["dims"][name]
            x_domain = [0, self._inputs["width_nm"] - width]
            y_domain = (
                [self._inputs["movable_y_lower_bounds_nm"].get(name, 0),
                 self._inputs["height_nm"] - height]
                if name in self._inputs["movable_y"]
                else [self._inputs["seed"][name][1]] * 2
            )
            if (
                proto.variables[x_index].name != f"x_{name}"
                or list(proto.variables[x_index].domain) != x_domain
                or list(proto.variables[y_index].domain) != y_domain
            ):
                raise ValueError(f"stage geometry coordinate domain differs: {name}")
        emitted = {str(row["row_id"]): [] for row in self._assumption_rows}
        by_index = {int(row["assumption_index"]): str(row["row_id"])
                    for row in self._assumption_rows}
        if set(emitted) != set(self._expected_counts):
            raise ValueError("expected and materialized production obligations differ")
        for row in self._assumption_rows:
            metadata = self._expected_metadata[row["row_id"]]
            if any(row.get(field) != value for field, value in metadata.items()):
                raise ValueError(f"production obligation metadata differs: {row['row_id']}")
        for i, row in enumerate(proto.constraints):
            for literal in row.enforcement_literal:
                if literal in by_index:
                    emitted[by_index[literal]].append(f"cp_primitive:{i}")
        for row_id, count in self._expected_counts.items():
            if count <= 0 or len(emitted[row_id]) != count:
                raise ValueError(f"production obligation primitive coverage differs: {row_id}")
        if len(proto.constraints) != self._expected_primitive_count:
            raise ValueError(
                "expected CP primitive count differs from emitted model "
                f"(derived_envelopes/branch definitions: expected={self._expected_primitive_count}, "
                f"actual={len(proto.constraints)})"
            )
        for index, row_id in by_index.items():
            emitted[row_id].append(f"cp_assumption:{index}")

        variables = {var.name: i for i, var in enumerate(proto.variables) if var.name}
        domains = [f"cp_domain:{i}" for i in range(len(proto.variables))]
        emitted["hard_background:fixed_domain"] = domains
        frozen_y = sorted(set(self._inputs["seed"]) - set(self._inputs["movable_y"]))
        if frozen_y:
            emitted["hard_background:fixed_y"] = [
                f"cp_domain:{self._indices[name][1]}" for name in frozen_y
            ]
        for name in self._background:
            key = f"hard_background:{name}"
            if key in emitted:
                continue
            indices = []
            for i, row in enumerate(proto.constraints):
                kind = row.WhichOneof("constraint")
                if name == "derived_envelopes":
                    selected = not any(v in by_index for v in row.enforcement_literal)
                elif name == "manufacturing_grid":
                    selected = kind == "int_mod"
                elif name == "no_overlap2d":
                    selected = kind in {"no_overlap_2d", "interval"}
                else:
                    variable = variables.get({"physical_origin": "physical_x_min",
                                              "exact_outline": "physical_x_max",
                                              "bottom_origin": "physical_bottom_slack"}[name])
                    selected = (kind == "linear" and variable is not None
                                and list(row.linear.vars) == [variable])
                if selected:
                    indices.append(f"cp_primitive:{i}")
            if name == "manufacturing_grid" and self._inputs["manufacturing_grid_nm"] == 1:
                indices = sorted({f"cp_domain:{i}" for pair in self._indices.values() for i in pair})
            if not indices:
                raise ValueError(f"missing materialized hard background: {name}")
            emitted[key] = indices
        return emitted

    def positions_from_solver(self, solver: Any) -> dict[str, tuple[int, int]]:
        solution = tuple(solver.ResponseProto().solution)
        if len(solution) != len(self._model.Proto().variables):
            raise ValueError("candidate/model variable inventory differs")
        return {name: (int(solution[x]), int(solution[y]))
                for name, (x, y) in sorted(self._indices.items())}

    def _production_inventory(self, model: Any, *,
                              incumbent_replay_positions: Mapping[str, tuple[int, int]] | None = None) -> dict[str, object]:
        from clay.layout.place_blocks_2d import _diagnostic_canonical_value

        if _data_sha256(_diagnostic_canonical_value(self._inputs)) != self.stage_input_sha256:
            raise ValueError("captured production stage inputs changed")
        captured = self._model.Clone()
        live = model.Clone()
        if incumbent_replay_positions is not None:
            from clay.layout.place.solver_execution_policy import prepare_incumbent_replay_model

            # This admits one exact mathematical certification transformation.
            # The caller separately proves the candidate's generation/source
            # origin. A coordinate dictionary alone cannot establish that origin.
            if (captured.Proto().objective.domain
                    or captured.Proto().HasField("floating_point_objective")):
                raise ValueError("incumbent certification cannot remove an objective domain or floating objective")
            coordinates = {name: tuple(captured.GetIntVarFromProtoIndex(i) for i in pair)
                           for name, pair in self._indices.items()}
            captured = prepare_incumbent_replay_model(captured,
                coordinate_variables=coordinates, positions=incumbent_replay_positions)
        else:
            for candidate in (captured, live):
                candidate.ClearObjective()
                ordered = sorted(candidate.Proto().assumptions)
                candidate.ClearAssumptions()
                candidate.Proto().assumptions.extend(ordered)
        if _proto_sha(captured.Proto()) != _proto_sha(live.Proto()):
            raise ValueError("live model differs from captured complete production model")
        materialized_bindings = self._materialized_bindings(self._model.Proto())
        bindings = self._materialized_bindings(model.Proto())
        if bindings != materialized_bindings:
            raise ValueError("materialized and active production primitive bindings differ")
        self._bindings = materialized_bindings
        by_id = {row["row_id"]: row for row in self._assumption_rows}
        expected = []
        for row_id, metadata in sorted(self._expected_metadata.items()):
            emitted = by_id[row_id]
            index = int(emitted["assumption_index"])
            expected.append({
                **metadata,
                "assumption_literal": model.Proto().variables[index].name,
                "assumption_index": index,
                "witness_status": emitted["witness_status"],
            })
        backgrounds = [{
            "row_kind": "HARD_BACKGROUND", "row_id": f"hard_background:{name}",
            "family": "hard_background", "obligation_id": name,
            "declaration_source": "clay/layout/place_blocks_2d.py",
        } for name in self._background]
        expected.extend(backgrounds)
        materialized = [dict(by_id[row_id]) for row_id in sorted(bindings) if row_id in by_id]
        materialized.extend(row for row in backgrounds if row["row_id"] in bindings)
        by_index = {int(row["assumption_index"]): row for row in self._assumption_rows}
        active = []
        for index in model.Proto().assumptions:
            if index not in by_index:
                raise ValueError("unknown or disabled live production assumption")
            row = by_index[index]
            if model.Proto().variables[index].name != row["assumption_literal"]:
                raise ValueError("live assumption identity differs")
            active.append(dict(row))
        active.extend(row for row in backgrounds if row["row_id"] in bindings)
        return build_production_constraint_inventory(
            stage_input_sha256=self.stage_input_sha256,
            model_proto_sha256=_proto_sha(model.Proto()),
            projected_rows=expected, materialized_rows=materialized, active_solver_rows=active,
        )

    def _native_replay(self, positions: Mapping[str, tuple[int, int]]) -> dict[str, object]:
        from clay.layout.place_blocks_2d import _banded_seed_satisfies_hard_constraints

        inputs = self._inputs
        failures: list[dict[str, object]] = []
        inventory: list[dict[str, object]] = []
        replay_spec: dict[str, object] = {}
        fields = (
            "dims", "bands", "gap_nm", "adjacency_limits_nm", "adjacency_scope_offsets_nm",
            "order_pairs", "floorplan_column_classes", "floorplan_column_groups",
            "pair_clearances_nm", "axis_alignment_groups", "center_alignment_pairs",
            "aggregate_envelope_orders", "hierarchical_macro_orders", "hierarchical_macro_alignments",
            "hierarchical_alignment_tolerance_scales_nm", "aspect_target", "aspect_bounds",
            "aspect_obligation", "manufacturing_grid_nm", "right_edge", "left_edge",
            "rigid_compound_footprints", "layered_collision_footprints",
            "row_bounded_cavity_projections", "overlap_compatible_pairs",
        )
        actual_width = max(positions[n][0] + inputs["dims"][n][0] for n in positions)
        passed = _banded_seed_satisfies_hard_constraints(
            positions, **{key: inputs[key] for key in fields}, ids=tuple(sorted(inputs["seed"])),
            width_bound_nm=inputs["width_nm"] if inputs["exact_outline"] else actual_width,
            height_bound_nm=inputs["height_nm"],
            vertical_order_pairs=inputs.get("vertical_order_pairs", ()),
            **({"mosaic_guest_slot_projections": inputs["mosaic_guest_slot_projections"]}
               if inputs.get("mosaic_guest_slot_projections", ()) else {}),
            **({"edge_rail_reservation": inputs["edge_rail_reservation"]}
               if inputs.get("edge_rail_reservation") is not None else {}),
            bottom_center_keepout_nm=0, bottom_edge_keepout_nm=0,
            bottom_edge_keepout_exempt=(), solve_cuts=(), placement_objective="area",
            strict_width_cap=True, exact_outline_width=inputs["exact_outline"],
            enforce_legacy_band_order=False, diagnostic_out=failures,
            hard_inventory_out=inventory, hard_spec_out=replay_spec,
        )
        # The legalizer also fixes a Y subset and can declare per-owner bottom
        # bounds. These are not declarations of the banded replay interface.
        for name, (x, y) in positions.items():
            if name not in inputs["movable_y"] and y != inputs["seed"][name][1]:
                failures.append({"row_id": f"fixed_y:{name}"})
            if y < inputs["movable_y_lower_bounds_nm"].get(name, 0):
                failures.append({"row_id": f"movable_y_bound:{name}"})
        bottom = inputs["bottom_origin_lower_bounds_nm"]
        if bottom and (any(positions[n][1] < bottom[n] for n in positions)
                       or min(positions[n][1] - bottom[n] for n in positions) != 0):
            failures.append({"row_id": "bottom_origin"})
        # The native checker reports individual source rows for relational
        # declarations. Its other checks are early-return branches; reaching
        # its successful return proves those branches were all evaluated.
        background_checks = [{
            "row_id": f"native:{name}", "status": "PASS" if passed and not failures else "FAIL",
        } for name in (*self._background, "physical_geometry", "floorplan_column",
                       "floorplan_column_group", "horizontal_edge", "aspect_obligation",
                       *(("edge_rail",) if inputs.get("edge_rail_reservation") is not None else ()))]
        return {"status": "PASS" if passed and not failures else "FAIL",
                "failed_rows": failures, "inventory": inventory,
                "background_checks": background_checks, "replay_spec": replay_spec}

    def build_receipt(self, *, model: Any, solver: Any, solver_status: Any,
                      candidate_positions: Mapping[str, tuple[int, int]] | None = None,
                      incumbent_replay_positions: Mapping[str, tuple[int, int]] | None = None) -> dict[str, object]:
        """Bind an existing full-model solution; never invoke a solver here."""
        from ortools.sat.python import cp_model

        inventory = self._production_inventory(model, incumbent_replay_positions=incumbent_replay_positions)
        response = solver.ResponseProto()
        status_name = solver.StatusName(solver_status) if type(solver_status) is not str else solver_status
        if status_name not in {"OPTIMAL", "FEASIBLE"} or response.status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise ValueError("production evidence requires an actual SAT incumbent")
        if status_name != solver.StatusName(response.status):
            raise ValueError("reported solver status differs from actual response")
        positions = self.positions_from_solver(solver)
        if incumbent_replay_positions is not None and dict(incumbent_replay_positions) != positions:
            raise ValueError("incumbent certification positions differ from the actual solver response")
        if candidate_positions is not None and dict(candidate_positions) != positions:
            raise ValueError("candidate positions differ from captured solver solution")
        candidate_sha = _data_sha256(sorted(positions.items()))
        solution = tuple(int(value) for value in response.solution)
        primitive_checks = _check_proto_solution(model.Proto(), solution)
        native = self._native_replay(positions)
        check_by_id = {row["row_id"]: row for row in primitive_checks}
        native_by_source = {
            source_id: row["row_id"] for row in native["inventory"]
            for source_id in row.get("source_constraint_ids", ())
        }
        native_background = {
            row["row_id"] for row in native.get("background_checks", ())
            if row["status"] == "PASS"
        }
        uncovered = []
        coverage = []
        for row in inventory["rows"]:
            row_id = row["row_id"]
            checked = self._bindings.get(row_id, [])
            family = row["family"]
            if row["row_kind"] == "HARD_BACKGROUND":
                native_id = f"native:{row['obligation_id']}"
            elif family in {"pair_clearance", "layered_collision"}:
                native_id = "native:physical_geometry"
            elif family in {"floorplan_column", "floorplan_column_group", "horizontal_edge", "aspect_obligation", "edge_rail"}:
                native_id = f"native:{family}"
            else:
                native_id = native_by_source.get(row["obligation_id"])
            native_covered = native_id is not None and (
                native_id in native_background or native_id in native_by_source.values()
            )
            if not checked or not native_covered or any(
                name not in check_by_id or check_by_id[name]["status"] != "PASS" for name in checked
            ):
                uncovered.append(row_id)
            coverage.append({"production_row_id": row_id, "primitive_check_ids": checked,
                             "native_replay_row_ids": [] if native_id is None else [native_id],
                             "native_checker": "_banded_seed_satisfies_hard_constraints"})
        failed = [row for row in primitive_checks if row["status"] != "PASS"] + native["failed_rows"]
        passed = native["status"] == "PASS" and not failed and not uncovered
        full_model = {
            "claim": "FULL_MODEL_SAT", "model_kind": "PRODUCTION_FULL_MODEL",
            "solver_status": status_name, "stage_input_sha256": self.stage_input_sha256,
            "model_proto_sha256": inventory["model_proto_sha256"],
            "captured_model_proto_sha256": _proto_sha(self._model.Proto()),
            "candidate_positions_sha256": candidate_sha,
            "production_row_count": inventory["row_count"],
            "production_row_ids_sha256": inventory["row_ids_sha256"],
            "production_rows_sha256": inventory["rows_sha256"],
            "evidence_adapter": "clay.fixed-outline-production-evidence/v1",
            "expected_primitive_count": self._expected_primitive_count,
            "materialized_primitive_count": len(self._model.Proto().constraints),
            "active_primitive_count": len(model.Proto().constraints),
            "variable_count": len(model.Proto().variables),
            "assumption_indices": list(model.Proto().assumptions),
            "primitive_bindings": self._bindings,
            "primitive_bindings_sha256": _data_sha256(self._bindings),
            "stage_inputs": self.stage_input_data,
            "candidate_positions": sorted(positions.items()),
        }
        replay = {
            "claim": "FULL_HARD_REPLAY_PASS" if passed else "FULL_HARD_REPLAY_FAIL",
            "status": "PASS" if passed else "FAIL", "failed_rows": failed,
            "uncovered_production_row_ids": uncovered,
            "candidate_positions_sha256": candidate_sha,
            "model_proto_sha256": inventory["model_proto_sha256"],
            "stage_input_sha256": self.stage_input_sha256,
            "checked_production_row_ids_sha256": inventory["row_ids_sha256"],
            "native_replay_inventory_sha256": _data_sha256(native),
            "native_replay": native, "primitive_checks": primitive_checks,
            "coverage_rows": coverage, "coverage_map_sha256": _data_sha256(coverage),
        }
        return build_constraint_model_promotion_receipt(
            reduced_model=None, production_inventory=inventory,
            full_model=full_model, full_hard_replay=replay, required_adapter=True,
        )


__all__ = ["FixedOutlineProductionEvidenceContext"]
