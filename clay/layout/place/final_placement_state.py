"""Cycle-free canonical identity for the exact final placement master.

This module owns both the static placement-causality basis and the canonical
versioned placement-state payload.  Placement and route-feedback code import
this module; it imports neither of them, so there is one hash and validation
implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any
from clay.layout.place.group_separation_strength import separation_gap_nm
from clay.layout.place.layered_assembly_certificate import with_layered_assembly_payload


_BBoxNm = tuple[int, int, int, int]
_GLOBAL_FLOORPLAN_COLUMN_SCOPE = -1


@dataclass(frozen=True, order=True)
class MatchingPairAxisProjection:
    """Exact rigid-block transport of one canonical MOS matching axis.

    Coordinates use four-times-nm integers so a midpoint between two half-nm
    device centres remains lossless.  The logical solver view and physical PCell
    footprint view have independent coordinates but share one axis direction and
    one rigid ``delta_nm``.  Final coordinate authority cannot be inferred from
    final artwork alone.
    """

    pair: tuple[str, str]
    axis: str
    source_logical_coordinate4_nm: int
    source_coordinate4_nm: int
    source_master: str
    delta_nm: tuple[int, int]
    final_logical_coordinate4_nm: int
    final_coordinate4_nm: int
    source_member_bboxes_nm: tuple[_BBoxNm, _BBoxNm]
    final_member_bboxes_nm: tuple[_BBoxNm, _BBoxNm]

    def __post_init__(self) -> None:
        if (
            type(self.pair) is not tuple
            or len(self.pair) != 2
            or any(type(member) is not str or not member for member in self.pair)
            or self.pair[0] >= self.pair[1]
        ):
            raise TypeError("matching-axis projection pair must be canonical strings")
        if type(self.axis) is not str or self.axis not in {"x", "y"}:
            raise TypeError("matching-axis projection axis must be exact x or y")
        if type(self.source_logical_coordinate4_nm) is not int:
            raise TypeError(
                "matching-axis projection source logical coordinate must be exact"
            )
        if type(self.source_coordinate4_nm) is not int:
            raise TypeError(
                "matching-axis projection source physical coordinate must be exact"
            )
        if type(self.source_master) is not str or not self.source_master:
            raise TypeError("matching-axis projection source master must be named")
        if (
            type(self.delta_nm) is not tuple
            or len(self.delta_nm) != 2
            or any(type(value) is not int for value in self.delta_nm)
        ):
            raise TypeError("matching-axis projection delta must be exact integer nm")
        if type(self.final_logical_coordinate4_nm) is not int:
            raise TypeError(
                "matching-axis projection final logical coordinate must be exact"
            )
        if type(self.final_coordinate4_nm) is not int:
            raise TypeError(
                "matching-axis projection final physical coordinate must be exact"
            )
        axis_index = 0 if self.axis == "x" else 1
        if self.final_logical_coordinate4_nm != (
            self.source_logical_coordinate4_nm + 4 * self.delta_nm[axis_index]
        ):
            raise ValueError(
                "matching-axis projection logical coordinate does not replay delta"
            )
        if self.final_coordinate4_nm != (
            self.source_coordinate4_nm + 4 * self.delta_nm[axis_index]
        ):
            raise ValueError("matching-axis projection coordinate does not replay delta")
        canonical_boxes = []
        for label, raw_boxes in (
            ("source", self.source_member_bboxes_nm),
            ("final", self.final_member_bboxes_nm),
        ):
            if type(raw_boxes) is not tuple or len(raw_boxes) != 2:
                raise TypeError(
                    f"matching-axis projection {label} bboxes must be an exact pair"
                )
            boxes = []
            for raw_bbox in raw_boxes:
                if (
                    type(raw_bbox) is not tuple
                    or len(raw_bbox) != 4
                    or any(type(value) is not int for value in raw_bbox)
                ):
                    raise TypeError(
                        f"matching-axis projection {label} bbox must be exact integer nm"
                    )
                if raw_bbox[2] <= raw_bbox[0] or raw_bbox[3] <= raw_bbox[1]:
                    raise ValueError(
                        f"matching-axis projection {label} bbox must have positive area"
                    )
                boxes.append(raw_bbox)
            sizes = tuple(
                (bbox[2] - bbox[0], bbox[3] - bbox[1]) for bbox in boxes
            )
            if sizes[0] != sizes[1]:
                raise ValueError(
                    f"matching-axis projection {label} pair size does not match"
                )
            perpendicular = 1 - axis_index
            if (
                boxes[0][perpendicular] + boxes[0][perpendicular + 2]
                != boxes[1][perpendicular] + boxes[1][perpendicular + 2]
            ):
                raise ValueError(
                    f"matching-axis projection {label} pair is not perpendicularly aligned"
                )
            coordinate4 = sum(
                bbox[axis_index] + bbox[axis_index + 2] for bbox in boxes
            )
            expected_coordinate4 = (
                self.source_coordinate4_nm
                if label == "source"
                else self.final_coordinate4_nm
            )
            if coordinate4 != expected_coordinate4:
                raise ValueError(
                    f"matching-axis projection {label} bboxes do not replay axis"
                )
            canonical_boxes.append(tuple(boxes))
        dx, dy = self.delta_nm
        translated = tuple(
            (x0 + dx, y0 + dy, x1 + dx, y1 + dy)
            for x0, y0, x1, y1 in canonical_boxes[0]
        )
        if translated != canonical_boxes[1]:
            raise ValueError(
                "matching-axis projection final bboxes do not replay rigid delta"
            )


def _matching_declared_source_index(certificate, blocks):
    """Replay declared source -> physical block -> solver master ownership.

    Composite master envelopes can contain empty space occupied by another
    legal master. Their bbox containment is a geometry check, not an ownership
    relation. Legacy certificates without any source partition retain the
    separate unique-bbox fallback in the caller.
    """
    sources = getattr(certificate, "materialized_block_source_members", ())
    source_boxes = getattr(certificate, "materialized_source_member_bboxes", ())
    if type(sources) is not tuple or type(source_boxes) is not tuple:
        raise TypeError("matching-axis source ownership and geometry must be exact tuples")
    if sources == () and source_boxes == ():
        return None

    def pairs(raw, label, *, boxes=False):
        if type(raw) is not tuple or not raw:
            raise TypeError(f"matching-axis {label} must be a nonempty exact tuple")
        result = {}
        for row in raw:
            if (type(row) is not tuple or len(row) != 2 or type(row[0]) is not str
                    or not row[0] or type(row[1]) is not tuple or not row[1]):
                raise TypeError(f"matching-axis {label} has a malformed row")
            name, values = row
            if boxes:
                valid = (len(values) == 4 and all(type(v) is int for v in values)
                         and values[0] < values[2] and values[1] < values[3])
            else:
                valid = (all(type(v) is str and v for v in values)
                         and values == tuple(sorted(set(values))))
            if not valid or name in result:
                raise ValueError(f"matching-axis {label} is malformed or repeated")
            result[name] = values
        if tuple(result) != tuple(sorted(result)):
            raise ValueError(f"matching-axis {label} must be canonical")
        return result

    members = pairs(getattr(certificate, "materialized_block_members", ()), "master partition")
    physical = pairs(getattr(certificate, "materialized_block_bboxes", ()), "physical bboxes", boxes=True)
    logicals = pairs(sources, "source partition")
    parent_of = {}
    for master, children in members.items():
        for child in children:
            if child in parent_of:
                raise ValueError("matching-axis physical block has repeated source master ownership")
            parent_of[child] = master
    if set(members) != set(blocks) or set(parent_of) != set(physical) or set(logicals) != set(physical):
        raise ValueError("matching-axis source master ownership partition is incomplete")

    def contains(outer, inner):
        return outer[0] <= inner[0] < inner[2] <= outer[2] and outer[1] <= inner[1] < inner[3] <= outer[3]

    from clay.layout.place.physical_master_frames import physical_master_boxes
    projections = getattr(certificate, "flexible_envelope_projections", ())
    flexible_physical = physical_master_boxes(blocks, projections)

    logical_index = {}
    for child, names in logicals.items():
        master = parent_of[child]
        if not contains(blocks[master], physical[child]):
            if child != master or physical[child] != flexible_physical.get(master):
                raise ValueError("matching-axis physical block escapes its source master")
        for name in names:
            if name in logical_index:
                raise ValueError("matching-axis logical source has repeated ownership")
            logical_index[name] = master, physical[child]
    measured = {} if source_boxes == () else pairs(source_boxes, "source bboxes", boxes=True)
    if not set(measured) <= set(logical_index):
        raise ValueError("matching-axis source bbox has foreign ownership")
    for names in logicals.values():
        covered = set(names).intersection(measured)
        if covered and covered != set(names):
            raise ValueError("matching-axis source bboxes incompletely cover a physical owner")
    for name, box in measured.items():
        if not contains(logical_index[name][1], box):
            raise ValueError("matching-axis source bbox escapes its physical owner")
    return logical_index, measured


def verify_matching_pair_projection_owners(certificate: object) -> bool:
    """Replay every MOS pair against one unique signed final-master owner.

    The producer names the source master before solving.  Its identity survives
    in ``certificate.blocks`` while the block bbox moves by the row's exact
    delta. The materializer's source partition identifies that master even when
    another compound envelope contains the pair in a cavity. Pair footprints
    must stay inside their declared physical/master owners and agree with any
    retained source bboxes. Certificates without a source partition retain the
    legacy unique-containment requirement. Reverse translation replays source
    containment without deriving either matching axis from final coordinates.
    """

    raw_rows = getattr(certificate, "matching_pair_axis_projections", ())
    if type(raw_rows) is not tuple:
        raise TypeError("matching-axis projection inventory must be an exact tuple")
    if not raw_rows:
        return True
    if any(type(row) is not MatchingPairAxisProjection for row in raw_rows):
        raise TypeError("matching-axis projection inventory contains a foreign row")
    raw_blocks = getattr(certificate, "blocks", ())
    if type(raw_blocks) is not tuple:
        raise TypeError("matching-axis projection owner blocks must be an exact tuple")
    blocks: dict[str, _BBoxNm] = {}
    for raw in raw_blocks:
        if (
            type(raw) is not tuple
            or len(raw) != 2
            or type(raw[0]) is not str
            or not raw[0]
            or type(raw[1]) is not tuple
            or len(raw[1]) != 4
            or any(type(value) is not int for value in raw[1])
            or raw[1][2] <= raw[1][0]
            or raw[1][3] <= raw[1][1]
            or raw[0] in blocks
        ):
            raise TypeError(
                "matching-axis projection owner blocks are malformed"
            )
        blocks[raw[0]] = raw[1]

    def contains(outer: _BBoxNm, inner: _BBoxNm) -> bool:
        return (
            outer[0] <= inner[0]
            and outer[1] <= inner[1]
            and outer[2] >= inner[2]
            and outer[3] >= inner[3]
        )

    declared = _matching_declared_source_index(certificate, blocks)
    for row in raw_rows:
        if declared is None:
            final_owners = tuple(sorted(name for name, bbox in blocks.items()
                if all(contains(bbox, member) for member in row.final_member_bboxes_nm)))
        else:
            logical_index, measured = declared
            if any(name not in logical_index for name in row.pair):
                raise ValueError("matching-axis pair lacks declared source master ownership")
            final_owners = tuple(sorted({logical_index[name][0] for name in row.pair}))
            for name, box in zip(row.pair, row.final_member_bboxes_nm):
                if not contains(logical_index[name][1], box) or (name in measured and measured[name] != box):
                    raise ValueError("matching-axis pair geometry differs from its materialized source owner")
        final_owner = blocks.get(row.source_master)
        if (final_owner is None or final_owners != (row.source_master,)
                or not all(contains(final_owner, box) for box in row.final_member_bboxes_nm)):
            raise ValueError(
                "matching-axis projection source master ownership does not replay"
            )
        dx, dy = row.delta_nm
        source_owner = (
            final_owner[0] - dx,
            final_owner[1] - dy,
            final_owner[2] - dx,
            final_owner[3] - dy,
        )
        if not all(
            contains(source_owner, member)
            for member in row.source_member_bboxes_nm
        ):
            raise ValueError(
                "matching-axis projection source master delta does not replay"
            )
    return True


def _sha256(payload: object) -> str:
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def canonical_minimum_structural_occupancy(value: object) -> float | None:
    """Replay the sole exact schema for a declared structural occupancy target."""

    if value is None:
        return None
    if type(value) is not float:
        raise TypeError("minimum_structural_occupancy must be an exact float")
    if not math.isfinite(value) or not 0.0 < value <= 1.0:
        raise ValueError("minimum_structural_occupancy must be finite and in (0, 1]")
    return value


def structural_occupancy_policy_binding(
    value: object,
) -> tuple[float | None, str]:
    """Bind one optional occupancy policy without relying on placement geometry."""

    canonical = canonical_minimum_structural_occupancy(value)
    return canonical, _sha256(
        {
            "minimum_structural_occupancy": canonical,
            "schema": "clay.structural-occupancy-policy.v1",
        }
    )


def functional_od_policy_binding(
    minimum_fraction: object,
    functional_area_nm2: object,
    active_layer_keys: object,
) -> tuple[float | None, int, tuple[tuple[int, int], ...], str]:
    """Canonical materializer evidence for source-owned functional active."""

    minimum = canonical_minimum_structural_occupancy(minimum_fraction)
    if (
        type(functional_area_nm2) is not int
        or functional_area_nm2 < 0
        or functional_area_nm2 > 10**18
    ):
        raise TypeError("functional_od_area_nm2 must be a non-negative exact int")
    if type(active_layer_keys) is not tuple:
        raise TypeError("functional OD layer keys must be an exact tuple")
    if len(active_layer_keys) > 4096:
        raise ValueError("functional OD layer keys must contain at most 4096 items")
    layers = []
    for raw in active_layer_keys:
        if (
            type(raw) is not tuple
            or len(raw) != 2
            or any(
                type(value) is not int or not 0 <= value <= 65_535
                for value in raw
            )
        ):
            raise TypeError("functional OD layer keys must be exact integer pairs")
        layers.append(raw)
    canonical_layers = tuple(sorted(layers))
    if tuple(active_layer_keys) != canonical_layers or len(canonical_layers) != len(
        set(canonical_layers)
    ):
        raise ValueError("functional OD layer keys must be sorted and unique")
    if minimum is None:
        if functional_area_nm2 != 0 or canonical_layers:
            raise ValueError("absent functional OD policy cannot carry area or layers")
    elif functional_area_nm2 <= 0 or not canonical_layers:
        raise ValueError("functional OD policy requires positive area and active layers")
    digest = _sha256(
        {
            "active_layer_keys": [list(key) for key in canonical_layers],
            "functional_area_nm2": functional_area_nm2,
            "minimum_fraction": minimum,
            "schema": "clay.functional-od-policy.v1",
        }
    )
    return minimum, functional_area_nm2, canonical_layers, digest


def _sorted_unique_strings(values: object, field_name: str) -> tuple[str, ...]:
    frozen = tuple(values or ())
    if any(not isinstance(value, str) or not value for value in frozen):
        raise ValueError(f"{field_name} must contain non-empty strings")
    if len(frozen) != len(set(frozen)):
        raise ValueError(f"{field_name} must be unique")
    return tuple(sorted(frozen))


def _canonical_block_bboxes(values: object) -> tuple[tuple[str, _BBoxNm], ...]:
    rows = []
    for raw in tuple(values or ()):
        if not isinstance(raw, (tuple, list)) or len(raw) != 2:
            raise ValueError("placement blocks must contain exact typed rows")
        name, raw_bbox = raw
        bbox = tuple(raw_bbox) if isinstance(raw_bbox, (tuple, list)) else ()
        if (
            not isinstance(name, str)
            or not name
            or len(bbox) != 4
            or any(type(value) is not int for value in bbox)
            or bbox[2] < bbox[0]
            or bbox[3] < bbox[1]
        ):
            raise ValueError(
                "placement blocks must contain ordered exact integer bboxes"
            )
        rows.append((name, bbox))
    canonical = tuple(sorted(rows))
    if len({name for name, _bbox in canonical}) != len(canonical):
        raise ValueError("placement block ids must be unique")
    return canonical


def _canonical_axis_alignment_groups(
    values: object,
    blocks: tuple[tuple[str, _BBoxNm], ...],
) -> tuple[tuple[tuple[str, int], ...], ...]:
    """Validate the exact differential axes signed by the final master."""

    by_name = dict(blocks)
    groups = []
    for raw_group in tuple(values or ()):
        entries = []
        for raw_entry in tuple(raw_group or ()):
            if not isinstance(raw_entry, (tuple, list)) or len(raw_entry) != 2:
                raise ValueError("placement axis alignment entry is malformed")
            name, offset2 = raw_entry
            if (
                not isinstance(name, str)
                or name not in by_name
                or type(offset2) is not int
                or offset2 < 0
                or offset2 > 2 * (by_name[name][2] - by_name[name][0])
            ):
                raise ValueError("placement axis alignment entry is invalid")
            entries.append((name, offset2))
        canonical = tuple(sorted(entries))
        if len(canonical) < 2 or len({name for name, _offset in canonical}) != len(
            canonical
        ):
            raise ValueError("placement axis alignment group is not a relation")
        axes = {
            2 * by_name[name][0] + offset2 for name, offset2 in canonical
        }
        if len(axes) != 1:
            raise ValueError("placement axis alignment is not satisfied")
        groups.append(canonical)
    canonical_groups = tuple(sorted(groups))
    if len(canonical_groups) != len(set(canonical_groups)):
        raise ValueError("placement axis alignment groups must be unique")
    return canonical_groups


def _canonical_center_alignment_pairs(
    values: object,
    blocks: tuple[tuple[str, _BBoxNm], ...],
) -> tuple[tuple[str, str, str], ...]:
    """Validate exact symmetric centre relations signed by the final master."""

    by_name = dict(blocks)
    pairs = []
    for raw_pair in tuple(values or ()):
        if not isinstance(raw_pair, (tuple, list)) or len(raw_pair) != 3:
            raise ValueError("placement center alignment entry is malformed")
        left, right = sorted((raw_pair[0], raw_pair[1]))
        axis = raw_pair[2]
        if (
            not isinstance(left, str)
            or not isinstance(right, str)
            or left == right
            or left not in by_name
            or right not in by_name
            or axis not in {"horizontal", "vertical"}
        ):
            raise ValueError("placement center alignment entry is invalid")
        coordinate = 0 if axis == "horizontal" else 1
        dimension = 2 if axis == "horizontal" else 3
        left_center2 = 2 * by_name[left][coordinate] + (
            by_name[left][dimension] - by_name[left][coordinate]
        )
        right_center2 = 2 * by_name[right][coordinate] + (
            by_name[right][dimension] - by_name[right][coordinate]
        )
        if left_center2 != right_center2:
            raise ValueError("placement center alignment is not satisfied")
        pairs.append((left, right, axis))
    canonical = tuple(sorted(pairs))
    if len(canonical) != len(set(canonical)):
        raise ValueError("placement center alignment pairs must be unique")
    return canonical


def _canonical_aggregate_envelope_orders(
    values: object,
    blocks: tuple[tuple[str, _BBoxNm], ...],
) -> tuple[tuple[str, str, str, tuple[str, ...], tuple[str, ...]], ...]:
    by_name = dict(blocks)
    rows = []
    for raw in tuple(values or ()):
        if not isinstance(raw, (tuple, list)) or len(raw) != 5:
            raise ValueError("placement aggregate envelope order is malformed")
        before_scope, after_scope, axis, before, after = raw
        before = tuple(before)
        after = tuple(after)
        from .constraints_hierarchical import _is_rail_halfplane_order
        rail_halfplane = _is_rail_halfplane_order(
            (before_scope, after_scope, axis, before, after)
        )
        if (
            not isinstance(before_scope, str)
            or not isinstance(after_scope, str)
            or before_scope == after_scope
            or axis not in {"horizontal", "vertical"}
            or not before
            or not after
            or before != tuple(sorted(before))
            or after != tuple(sorted(after))
            or (set(before) & set(after) and not rail_halfplane)
            or not set((*before, *after)) <= set(by_name)
            or (rail_halfplane and set((*before, *after)) != set(by_name))
        ):
            raise ValueError("placement aggregate envelope order is invalid")
        lower = 0 if axis == "horizontal" else 1
        upper = 2 if axis == "horizontal" else 3
        before_center2 = min(by_name[name][lower] for name in before) + max(
            by_name[name][upper] for name in before
        )
        after_center2 = min(by_name[name][lower] for name in after) + max(
            by_name[name][upper] for name in after
        )
        if before_center2 > after_center2 or (
            not rail_halfplane and before_center2 == after_center2
        ):
            raise ValueError("placement aggregate envelope order is not satisfied")
        rows.append((before_scope, after_scope, axis, before, after))
    canonical = tuple(sorted(rows))
    if len(canonical) != len(set(canonical)):
        raise ValueError("placement aggregate envelope orders must be unique")
    return canonical


def _canonical_hierarchical_macro_orders(
    values: object,
    blocks: tuple[tuple[str, tuple[int, int, int, int]], ...],
    *,
    gap_nm: int,
) -> tuple[tuple[object, ...], ...]:
    if type(values) is not tuple:
        raise TypeError("hierarchical macro orders must be an exact tuple")
    by_bbox = dict(blocks)
    rows = []
    for raw in values:
        if type(raw) is not tuple or len(raw) != 7:
            raise TypeError("hierarchical macro order row is malformed")
        scope, before_node, after_node, axis, strength, before, after = raw
        if (
            any(
                type(value) is not str or not value
                for value in (scope, before_node, after_node)
            )
            or before_node == after_node
            or axis not in {"horizontal", "vertical"}
            or (strength not in {
                "center_order",
                "center_order_with_track_floor",
                "envelope_separation",
                "member_before_aggregate_center",
                "member_before_aggregate_center_with_track_floor",
                "leading_edge_order",
                "leading_edge_equal",
            } and separation_gap_nm(strength) is None)
            or type(before) is not tuple
            or type(after) is not tuple
            or not before
            or not after
            or before != tuple(sorted(before))
            or after != tuple(sorted(after))
            or set(before) & set(after)
            or not set((*before, *after)) <= set(by_bbox)
        ):
            raise ValueError("hierarchical macro order signature differs")
        lower = 0 if axis == "horizontal" else 1
        upper = 2 if axis == "horizontal" else 3
        before_low = min(by_bbox[name][lower] for name in before)
        before_high = max(by_bbox[name][upper] for name in before)
        after_low = min(by_bbox[name][lower] for name in after)
        after_high = max(by_bbox[name][upper] for name in after)
        if separation_gap_nm(strength) is not None:
            satisfied = before_high + separation_gap_nm(strength) <= after_low
        elif strength == "leading_edge_order":
            satisfied = before_low < after_low
        elif strength == "leading_edge_equal":
            satisfied = before_low == after_low
        elif strength == "member_before_aggregate_center":
            satisfied = max(
                by_bbox[name][lower] + by_bbox[name][upper]
                for name in before
            ) < after_low + after_high
        elif strength == "member_before_aggregate_center_with_track_floor":
            satisfied = max(
                by_bbox[name][lower] + by_bbox[name][upper]
                for name in before
            ) + 2 * gap_nm <= after_low + after_high
        elif strength == "center_order_with_track_floor":
            satisfied = (
                before_low + before_high + 2 * gap_nm
                <= after_low + after_high
            )
        else:
            satisfied = (before_low + before_high) // 2 < (
                after_low + after_high
            ) // 2
        if not satisfied:
            raise ValueError("hierarchical macro order is not satisfied")
        rows.append(raw)
    canonical = tuple(sorted(rows))
    if canonical != values or len(canonical) != len(set(canonical)):
        raise ValueError("hierarchical macro orders must be canonical and unique")
    return canonical


def _canonical_hierarchical_macro_alignments(
    values: object,
    blocks: tuple[tuple[str, tuple[int, int, int, int]], ...],
    *,
    tolerance_scales_nm: object = (),
) -> tuple[tuple[object, ...], ...]:
    if type(values) is not tuple:
        raise TypeError("hierarchical macro alignments must be an exact tuple")
    if type(tolerance_scales_nm) is not tuple:
        raise TypeError(
            "hierarchical alignment tolerance scales must be an exact tuple"
        )
    scales = tuple(tolerance_scales_nm)
    if any(
        type(row) is not tuple
        or len(row) != 2
        or row[0] not in {"horizontal", "vertical"}
        or type(row[1]) is not int
        or row[1] <= 0
        for row in scales
    ):
        raise ValueError("hierarchical alignment tolerance scale is malformed")
    if scales != tuple(sorted(set(scales))):
        raise ValueError(
            "hierarchical alignment tolerance scales must be canonical and unique"
        )
    scales_by_axis = dict(scales)
    by_bbox = dict(blocks)
    rows = []
    required_tolerance_axes = set()
    for raw in values:
        if type(raw) is not tuple or len(raw) not in {3, 4, 5}:
            raise TypeError("hierarchical macro alignment row is malformed")
        if len(raw) == 3:
            scope, axis, groups = raw
            edge = "center"
            tolerance_ppm = 0
        elif len(raw) == 4:
            scope, axis, edge, groups = raw
            tolerance_ppm = 0
        else:
            scope, axis, edge, tolerance_ppm, groups = raw
        if (
            type(scope) is not str
            or not scope
            or axis not in {"horizontal", "vertical"}
            or edge not in {"center", "leading", "trailing"}
            or type(tolerance_ppm) is not int
            or not 0 <= tolerance_ppm <= 1_000_000
            or type(groups) is not tuple
            or len(groups) < 2
        ):
            raise ValueError("hierarchical macro alignment signature differs")
        lower = 0 if axis == "horizontal" else 1
        upper = 2 if axis == "horizontal" else 3
        if tolerance_ppm > 0:
            required_tolerance_axes.add(axis)
        alignment_values = []
        claimed = set()
        for group in groups:
            if type(group) is not tuple or len(group) != 2:
                raise TypeError("hierarchical macro alignment group is malformed")
            node, members = group
            if (
                type(node) is not str
                or not node
                or type(members) is not tuple
                or not members
                or members != tuple(sorted(members))
                or claimed & set(members)
                or not set(members) <= set(by_bbox)
            ):
                raise ValueError("hierarchical macro alignment ownership differs")
            claimed.update(members)
            low = min(by_bbox[name][lower] for name in members)
            high = max(by_bbox[name][upper] for name in members)
            alignment_values.append(
                low if edge == "leading" else high if edge == "trailing" else low + high
            )
        normalization_nm = scales_by_axis.get(axis)
        if normalization_nm is None:
            normalization_nm = max(box[upper] for box in by_bbox.values()) - min(
                box[lower] for box in by_bbox.values()
            )
        scale = 2 if edge == "center" else 1
        base = alignment_values[0]
        satisfied = (
            len(set(alignment_values)) == 1
            if tolerance_ppm == 0
            else all(
                abs(base - value) * 1_000_000
                <= scale * normalization_nm * tolerance_ppm
                for value in alignment_values[1:]
            )
        )
        if groups != tuple(sorted(groups)) or not satisfied:
            raise ValueError("hierarchical macro alignment is not satisfied")
        rows.append(raw)

    def alignment_key(row: tuple[object, ...]) -> tuple[object, ...]:
        if len(row) == 3:
            scope, axis, groups = row
            return scope, axis, "center", 0, groups
        if len(row) == 4:
            scope, axis, edge, groups = row
            return scope, axis, edge, 0, groups
        scope, axis, edge, tolerance_ppm, groups = row
        return scope, axis, edge, tolerance_ppm, groups

    canonical = tuple(sorted(rows, key=alignment_key))
    if canonical != values or len(canonical) != len(set(canonical)):
        raise ValueError(
            "hierarchical macro alignments must be canonical and unique"
        )
    if tolerance_scales_nm:
        if tuple(axis for axis, _scale_nm in scales) != tuple(
            sorted(required_tolerance_axes)
        ):
            raise ValueError(
                "hierarchical alignment tolerance scales differ from required axes"
            )
    return canonical


@dataclass(frozen=True, order=True)
class PlacementCorridorBoundary:
    """One visible pair of final-master faces that can bound a route corridor."""

    relation: str
    owner_block_ids: tuple[str, str]
    run_interval_nm: tuple[int, int]
    observed_clearance_nm: int

    def __post_init__(self) -> None:
        if self.relation not in ("horizontal_separation", "vertical_separation"):
            raise ValueError("placement corridor boundary has unknown relation")
        owners = tuple(sorted(self.owner_block_ids))
        if (
            len(owners) != 2
            or owners[0] == owners[1]
            or any(not isinstance(owner, str) or not owner for owner in owners)
        ):
            raise ValueError("placement corridor boundary requires two block ids")
        interval = tuple(self.run_interval_nm)
        if (
            len(interval) != 2
            or any(type(value) is not int for value in interval)
            or interval[1] <= interval[0]
        ):
            raise ValueError("placement corridor boundary needs a non-empty run interval")
        if type(self.observed_clearance_nm) is not int or self.observed_clearance_nm < 0:
            raise ValueError("placement corridor clearance must be non-negative integer nm")
        object.__setattr__(self, "owner_block_ids", owners)
        object.__setattr__(self, "run_interval_nm", interval)


@dataclass(frozen=True)
class PlacementCausalityBasis:
    """Static, route-candidate-free facts signed by the final placement master."""

    blocks: tuple[tuple[str, _BBoxNm, tuple[str, ...]], ...]
    corridor_boundaries: tuple[PlacementCorridorBoundary, ...]
    landing_owner_block_ids: tuple[str, ...]
    declaration_refs: tuple[str, ...] = ()
    schema: str = "clay.placement-causality-basis.v1"

    def __post_init__(self) -> None:
        if self.schema != "clay.placement-causality-basis.v1":
            raise ValueError("unknown placement causality basis schema")
        rows = []
        for raw in tuple(self.blocks):
            if not isinstance(raw, (tuple, list)) or len(raw) != 3:
                raise ValueError("placement causality basis has malformed block row")
            name, raw_bbox, raw_members = raw
            bbox = tuple(raw_bbox) if isinstance(raw_bbox, (tuple, list)) else ()
            members = tuple(sorted(raw_members))
            if (
                not isinstance(name, str)
                or not name
                or len(bbox) != 4
                or any(type(value) is not int for value in bbox)
                or bbox[2] < bbox[0]
                or bbox[3] < bbox[1]
                or not members
                or any(not isinstance(member, str) or not member for member in members)
                or len(members) != len(set(members))
            ):
                raise ValueError("placement causality basis has malformed ownership row")
            rows.append((name, bbox, members))
        canonical = tuple(sorted(rows))
        names = tuple(name for name, _bbox, _members in canonical)
        members = tuple(
            member for _name, _bbox, row_members in canonical for member in row_members
        )
        if len(names) != len(set(names)) or len(members) != len(set(members)):
            raise ValueError("placement causality basis ownership is not a partition")
        boundaries = tuple(sorted(self.corridor_boundaries))
        if any(
            not isinstance(boundary, PlacementCorridorBoundary)
            or not set(boundary.owner_block_ids) <= set(names)
            for boundary in boundaries
        ) or len(boundaries) != len(set(boundaries)):
            raise ValueError("placement causality basis has invalid corridor boundaries")
        landing_owners = _sorted_unique_strings(
            self.landing_owner_block_ids, "landing_owner_block_ids"
        )
        if set(landing_owners) != set(names):
            raise ValueError("placement causality basis landing ownership is incomplete")
        object.__setattr__(self, "blocks", canonical)
        object.__setattr__(self, "corridor_boundaries", boundaries)
        object.__setattr__(self, "landing_owner_block_ids", landing_owners)
        object.__setattr__(
            self,
            "declaration_refs",
            _sorted_unique_strings(self.declaration_refs, "declaration_refs"),
        )

    @property
    def sha256(self) -> str:
        return _sha256({
            "blocks": [
                [name, list(bbox), list(members)]
                for name, bbox, members in self.blocks
            ],
            "corridor_boundaries": [
                {
                    "observed_clearance_nm": boundary.observed_clearance_nm,
                    "owner_block_ids": list(boundary.owner_block_ids),
                    "relation": boundary.relation,
                    "run_interval_nm": list(boundary.run_interval_nm),
                }
                for boundary in self.corridor_boundaries
            ],
            "declaration_refs": list(self.declaration_refs),
            "landing_owner_block_ids": list(self.landing_owner_block_ids),
            "schema": self.schema,
        })


def build_placement_causality_basis(
    blocks: object,
    *,
    master_members: object = (),
    declaration_refs: object = (),
) -> PlacementCausalityBasis:
    """Compile deterministic static causes from one exact final-master geometry."""

    raw_blocks = tuple(blocks or ())
    if not raw_blocks:
        if tuple(master_members or ()):
            raise ValueError("empty placement cannot name master ownership")
        return PlacementCausalityBasis(
            blocks=(), corridor_boundaries=(), landing_owner_block_ids=(),
            declaration_refs=tuple(declaration_refs or ()),
        )
    canonical_blocks = _canonical_block_bboxes(raw_blocks)
    by_name = dict(canonical_blocks)
    raw_members = tuple(master_members or ())
    if raw_members:
        member_map: dict[str, tuple[str, ...]] = {}
        for raw in raw_members:
            if not isinstance(raw, (tuple, list)) or len(raw) != 2:
                raise ValueError("placement causality master ownership is malformed")
            name, members = raw
            frozen = tuple(sorted(members))
            if (
                not isinstance(name, str) or not name or not frozen
                or any(not isinstance(item, str) or not item for item in frozen)
                or name in member_map
            ):
                raise ValueError("placement causality master ownership is malformed")
            member_map[name] = frozen
        if set(member_map) != set(by_name):
            raise ValueError("placement causality master ownership is incomplete")
    else:
        member_map = {name: (name,) for name in by_name}

    def visible_gap(
        left_name: str,
        right_name: str,
        relation: str,
        run_interval: tuple[int, int],
    ) -> bool:
        left, right = by_name[left_name], by_name[right_name]
        if relation == "vertical_separation":
            if left[3] <= right[1]:
                gap = (left[3], right[1])
            elif right[3] <= left[1]:
                gap = (right[3], left[1])
            else:
                return False
            prism = (run_interval[0], gap[0], run_interval[1], gap[1])
        else:
            if left[2] <= right[0]:
                gap = (left[2], right[0])
            elif right[2] <= left[0]:
                gap = (right[2], left[0])
            else:
                return False
            prism = (gap[0], run_interval[0], gap[1], run_interval[1])
        return not any(
            other not in (left_name, right_name)
            and max(prism[0], bbox[0]) < min(prism[2], bbox[2])
            and max(prism[1], bbox[1]) < min(prism[3], bbox[3])
            for other, bbox in canonical_blocks
        )

    boundaries = []
    names = tuple(name for name, _bbox in canonical_blocks)
    for index, left_name in enumerate(names):
        left = by_name[left_name]
        for right_name in names[index + 1:]:
            right = by_name[right_name]
            x_overlap = (max(left[0], right[0]), min(left[2], right[2]))
            if (
                x_overlap[0] < x_overlap[1]
                and (left[3] <= right[1] or right[3] <= left[1])
                and visible_gap(left_name, right_name, "vertical_separation", x_overlap)
            ):
                boundaries.append(PlacementCorridorBoundary(
                    relation="vertical_separation",
                    owner_block_ids=(left_name, right_name),
                    run_interval_nm=x_overlap,
                    observed_clearance_nm=max(
                        0, left[1] - right[3], right[1] - left[3]
                    ),
                ))
            y_overlap = (max(left[1], right[1]), min(left[3], right[3]))
            if (
                y_overlap[0] < y_overlap[1]
                and (left[2] <= right[0] or right[2] <= left[0])
                and visible_gap(left_name, right_name, "horizontal_separation", y_overlap)
            ):
                boundaries.append(PlacementCorridorBoundary(
                    relation="horizontal_separation",
                    owner_block_ids=(left_name, right_name),
                    run_interval_nm=y_overlap,
                    observed_clearance_nm=max(
                        0, left[0] - right[2], right[0] - left[2]
                    ),
                ))
    return PlacementCausalityBasis(
        blocks=tuple((name, bbox, member_map[name]) for name, bbox in canonical_blocks),
        corridor_boundaries=tuple(boundaries),
        landing_owner_block_ids=names,
        declaration_refs=tuple(declaration_refs or ()),
    )


def verify_placement_causality_basis(certificate: object) -> bool:
    basis = getattr(certificate, "placement_causality_basis", None)
    if not isinstance(basis, PlacementCausalityBasis):
        raise ValueError("final placement certificate lacks a causality basis")
    replayed = build_placement_causality_basis(
        getattr(certificate, "blocks", ()),
        master_members=tuple(
            getattr(certificate, "materialized_block_members", ()) or ()
        ),
        declaration_refs=tuple(
            getattr(certificate, "placement_declaration_refs", ()) or ()
        ),
    )
    if replayed != basis or replayed.sha256 != basis.sha256:
        raise ValueError("final placement causality basis does not replay")
    return True


def _canonical_full2d_solver_evidence(
    values: object,
    *,
    graph_sha256: str,
    pair_constraint_modes: tuple[str, ...],
    graph: object | None = None,
) -> list[dict[str, Any]]:
    """Validate diagnostic CP-SAT observations without promoting them to a basis."""

    if type(values) is not tuple:
        raise TypeError("full-2D solver evidence must be one exact tuple")
    from clay.layout.place.full2d_solver_evidence import (
        validate_full2d_solver_evidence,
    )

    return validate_full2d_solver_evidence(
        values,
        graph_sha256=graph_sha256,
        pair_constraint_modes=pair_constraint_modes,
        graph=graph,
    )


def final_placement_state_payload(certificate: object) -> dict[str, Any]:
    """Validate and return the canonical versioned placement-state payload."""

    blocks = _canonical_block_bboxes(getattr(certificate, "blocks", ()))
    from clay.layout.place.physical_master_frames import physical_master_boxes, projection_payload
    flexible_projections = getattr(certificate, "flexible_envelope_projections", ())
    physical_master_boxes(dict(blocks), flexible_projections)
    bbox_nm = tuple(getattr(certificate, "bbox_nm", ()))
    if (
        len(bbox_nm) != 4
        or any(type(value) is not int for value in bbox_nm)
        or bbox_nm[2] < bbox_nm[0]
        or bbox_nm[3] < bbox_nm[1]
    ):
        raise ValueError("placement bbox must be an ordered exact integer tuple")
    bands = tuple(getattr(certificate, "bands", ()))
    if any(
        not isinstance(name, str) or not name or type(band) is not int
        for name, band in bands
    ) or len({name for name, _band in bands}) != len(bands):
        raise ValueError("placement bands must name unique exact integer ordinals")
    declared_channel_nm = getattr(certificate, "declared_channel_nm", None)
    if type(declared_channel_nm) is not int:
        raise ValueError("declared channel must be exact integer nm")
    payload: dict[str, Any] = {
        "bands": [[name, band] for name, band in sorted(bands)],
        "bbox_nm": list(bbox_nm),
        "blocks": [[name, list(bbox)] for name, bbox in blocks],
        "declared_channel_nm": declared_channel_nm,
        "schema": "clay.final-placement-state.v2",
    }
    raw_reference_obligations = getattr(
        certificate,
        "reference_conditioned_placement_obligations",
        (),
    )
    if type(raw_reference_obligations) is not tuple:
        raise TypeError(
            "reference-conditioned placement obligations must be an exact tuple"
        )
    if raw_reference_obligations:
        from clay.layout.spatial.intent_projection import (
            replay_reference_conditioned_placement_obligation_inventory,
        )

        reference_obligations = (
            replay_reference_conditioned_placement_obligation_inventory(
                raw_reference_obligations
            )
        )
        if reference_obligations != raw_reference_obligations:
            raise ValueError(
                "reference-conditioned placement obligations are not canonical"
            )
        payload["reference_conditioned_placement_obligations"] = [
            row.canonical_data() for row in reference_obligations
        ]
    raw_cavity_projections = getattr(
        certificate,
        "row_bounded_cavity_projections",
        (),
    )
    if type(raw_cavity_projections) is not tuple:
        raise TypeError("row-bounded cavity projections must be an exact tuple")
    if raw_cavity_projections:
        from clay.layout.canonical_collision_projection import (
            CanonicalRowBoundedCavityProjection,
        )

        if any(
            type(row) is not CanonicalRowBoundedCavityProjection
            for row in raw_cavity_projections
        ):
            raise TypeError("row-bounded cavity projection has a foreign type")
        if tuple(
            sorted(raw_cavity_projections, key=lambda row: row.obligation_id)
        ) != raw_cavity_projections:
            raise ValueError("row-bounded cavity projections are not canonical")
        block_map = dict(blocks)
        cavity_clearance_nm = int(getattr(certificate, "declared_gap_nm", 0))
        if cavity_clearance_nm < 0:
            raise ValueError("row-bounded cavity clearance is negative")
        for row in raw_cavity_projections:
            row.__post_init__()
            host = block_map[row.host_master]
            if row.upper_boundary_offset_nm != host[3] - host[1]:
                raise ValueError("row-bounded cavity host height differs")
            for guest_name in row.guest_masters:
                guest = block_map[guest_name]
                if not (
                    host[0] <= guest[0]
                    and guest[2] <= host[2]
                    and guest[1]
                    >= host[1] + row.lower_boundary_offset_nm + cavity_clearance_nm
                    and guest[3] + cavity_clearance_nm
                    <= host[1] + row.upper_boundary_offset_nm
                ):
                    raise ValueError("row-bounded cavity projection is not satisfied")
        payload["row_bounded_cavity_projections"] = [
            row.canonical_data() for row in raw_cavity_projections
        ]
    raw_guest_slot_projections = getattr(certificate, "mosaic_guest_slot_projections", ())
    if type(raw_guest_slot_projections) is not tuple:
        raise TypeError("guest-slot projections must be an exact tuple")
    if raw_guest_slot_projections:
        from clay.layout.mosaic_guest_slot_projection import (
            canonical_mosaic_guest_slot_projections, mosaic_guest_slot_satisfied,
        )
        guest_slot_dimensions = {name: (box[2]-box[0], box[3]-box[1]) for name, box in blocks}
        guest_slot_positions = {name: (box[0], box[1]) for name, box in blocks}
        canonical_mosaic_guest_slot_projections(raw_guest_slot_projections, dims=guest_slot_dimensions)
        guest_slot_gap = getattr(certificate, "declared_gap_nm", 0)
        if (type(guest_slot_gap) is not int or guest_slot_gap < 0
                or any(row.minimum_clearance_nm != guest_slot_gap for row in raw_guest_slot_projections)):
            raise ValueError("guest-slot clearance differs from declared source gap")
        if any(not mosaic_guest_slot_satisfied(guest_slot_positions, row) for row in raw_guest_slot_projections):
            raise ValueError("guest-slot projection is not satisfied")
        payload["mosaic_guest_slot_projections"] = [row.canonical_data() for row in raw_guest_slot_projections]
    raw_consumption = getattr(
        certificate,
        "intent_consumption_certificates",
        (),
    )
    if type(raw_consumption) is not tuple:
        raise TypeError("intent consumption certificates must be an exact tuple")
    if raw_consumption:
        from clay.layout.intent_basis import IntentConsumptionCertificate

        if any(type(row) is not IntentConsumptionCertificate for row in raw_consumption):
            raise TypeError("intent consumption certificate has a foreign type")
        stages = tuple(row.stage for row in raw_consumption)
        if stages != tuple(sorted(stages)) or len(set(stages)) != len(stages):
            raise ValueError(
                "intent consumption certificates must have sorted unique stages"
            )
        payload["intent_consumption_certificates"] = [
            row.canonical_data() for row in raw_consumption
        ]
    raw_fact_projections = getattr(certificate, "fact_projections", ())
    if type(raw_fact_projections) is not tuple:
        raise TypeError("placement fact projections must be an exact tuple")
    if raw_fact_projections:
        from clay.layout.intent_closure import ConsumerProjection

        if any(type(row) is not ConsumerProjection for row in raw_fact_projections):
            raise TypeError("placement fact projection has a foreign type")
        consumers = tuple(row.consumer for row in raw_fact_projections)
        if consumers != ("placement_initial", "placement_replay"):
            raise ValueError(
                "placement fact projections require initial then replay order"
            )
        if (
            raw_fact_projections[0].fact_ids
            != raw_fact_projections[1].fact_ids
            or raw_fact_projections[0].normalized_projection_sha256
            != raw_fact_projections[1].normalized_projection_sha256
        ):
            raise ValueError("placement fact projections violate replay parity")
        payload["fact_projections"] = [
            row.canonical_data() for row in raw_fact_projections
        ]
    declared_bottom_edge_keepout_nm = getattr(
        certificate, "declared_bottom_edge_keepout_nm", 0
    )
    if type(declared_bottom_edge_keepout_nm) is not int:
        raise TypeError(
            "declared bottom-edge keepout must be an exact integer nm value"
        )
    if declared_bottom_edge_keepout_nm < 0:
        raise ValueError("declared bottom-edge keepout must be non-negative")
    bottom_edge_inventories: dict[str, tuple[str, ...]] = {}
    for field, label in (
        ("bottom_edge_keepout_carrier_blocks", "carrier"),
        ("bottom_edge_keepout_clear_blocks", "clear"),
    ):
        raw_names = getattr(certificate, field, ())
        if type(raw_names) is not tuple:
            raise TypeError(
                f"bottom-edge keepout {label} blocks must be one exact tuple"
            )
        if any(type(name) is not str or not name for name in raw_names):
            raise TypeError(
                f"bottom-edge keepout {label} blocks must be exact named strings"
            )
        if raw_names != tuple(sorted(raw_names)) or len(set(raw_names)) != len(
            raw_names
        ):
            raise ValueError(
                f"bottom-edge keepout {label} blocks must be canonical and unique"
            )
        if any(name not in dict(blocks) for name in raw_names):
            raise ValueError(
                f"bottom-edge keepout {label} blocks name a foreign master"
            )
        bottom_edge_inventories[field] = raw_names
    bottom_carriers = bottom_edge_inventories[
        "bottom_edge_keepout_carrier_blocks"
    ]
    bottom_clear = bottom_edge_inventories["bottom_edge_keepout_clear_blocks"]
    if set(bottom_carriers).intersection(bottom_clear):
        raise ValueError(
            "bottom-edge keepout carrier and clear ownership must be disjoint"
        )
    if declared_bottom_edge_keepout_nm == 0 and (bottom_carriers or bottom_clear):
        raise ValueError(
            "bottom-edge keepout ownership requires a non-zero declaration"
        )
    if declared_bottom_edge_keepout_nm > 0:
        allowed = set(bottom_carriers) | set(bottom_clear)
        if bbox_nm[1] != 0 or any(
            bbox[1] < declared_bottom_edge_keepout_nm
            for name, bbox in blocks
            if name not in allowed
        ):
            raise ValueError(
                "bottom-edge rail reserve does not replay the placement bbox"
            )
        payload["declared_bottom_edge_keepout_nm"] = (
            declared_bottom_edge_keepout_nm
        )
        payload["bottom_edge_keepout_carrier_blocks"] = list(bottom_carriers)
        payload["bottom_edge_keepout_clear_blocks"] = list(bottom_clear)
    raw_columns = getattr(certificate, "floorplan_column_classes", ())
    if type(raw_columns) is not tuple:
        raise TypeError("floorplan column classes must be one exact tuple")
    block_by_name = dict(blocks)
    column_rows = []
    for raw in raw_columns:
        if type(raw) is not tuple or len(raw) != 4:
            raise TypeError("floorplan column classes contain a malformed row")
        name, column, min_center_offset2, max_center_offset2 = raw
        if (
            type(name) is not str
            or name not in block_by_name
            or type(column) is not int
            or column < 0
            or type(min_center_offset2) is not int
            or type(max_center_offset2) is not int
            or min_center_offset2 < 0
            or max_center_offset2 < min_center_offset2
            or max_center_offset2
            > 2 * (block_by_name[name][2] - block_by_name[name][0])
        ):
            raise ValueError("floorplan column class is not bound to a final block")
        column_rows.append(
            (name, column, min_center_offset2, max_center_offset2)
        )
    canonical_columns = tuple(sorted(column_rows))
    if tuple(raw_columns) != canonical_columns or len(
        {row[0] for row in canonical_columns}
    ) != len(canonical_columns):
        raise ValueError("floorplan column classes are not canonical and unique")
    for left, left_column, _left_min2, left_max2 in canonical_columns:
        for right, right_column, right_min2, _right_max2 in canonical_columns:
            if left_column >= right_column:
                continue
            left_bbox = block_by_name[left]
            right_bbox = block_by_name[right]
            if 2 * left_bbox[0] + left_max2 >= 2 * right_bbox[0] + right_min2:
                raise ValueError("placement blocks violate floorplan column order")
    if canonical_columns:
        payload["floorplan_column_classes"] = [
            list(row) for row in canonical_columns
        ]
    raw_column_groups = getattr(certificate, "floorplan_column_groups", ())
    if type(raw_column_groups) is not tuple:
        raise TypeError("floorplan column groups must be one exact tuple")
    column_group_rows = []
    for raw in raw_column_groups:
        if type(raw) is not tuple or len(raw) != 5:
            raise TypeError("floorplan column groups contain a malformed row")
        group, row, column, member_count, raw_terms = raw
        if (
            type(group) is not str
            or not group
            or type(row) is not int
            or row < _GLOBAL_FLOORPLAN_COLUMN_SCOPE
            or type(column) is not int
            or column < 0
            or type(member_count) is not int
            or member_count <= 0
            or type(raw_terms) is not tuple
            or not raw_terms
        ):
            raise ValueError("floorplan column group has malformed authority")
        terms = []
        for term in raw_terms:
            if type(term) is not tuple or len(term) != 3:
                raise TypeError("floorplan column group contains a malformed term")
            owner, count, center_offset2_sum = term
            if (
                type(owner) is not str
                or owner not in block_by_name
                or type(count) is not int
                or count <= 0
                or type(center_offset2_sum) is not int
                or center_offset2_sum < 0
                or center_offset2_sum
                > 2 * count * (block_by_name[owner][2] - block_by_name[owner][0])
            ):
                raise ValueError("floorplan column group term is not bound to a block")
            terms.append((owner, count, center_offset2_sum))
        canonical_terms = tuple(sorted(terms))
        if tuple(raw_terms) != canonical_terms or len(
            {owner for owner, _count, _offset in terms}
        ) != len(terms):
            raise ValueError("floorplan column group terms are not canonical and unique")
        if sum(count for _owner, count, _offset in terms) != member_count:
            raise ValueError("floorplan column group does not conserve member count")
        column_group_rows.append((group, row, column, member_count, canonical_terms))
    canonical_column_groups = tuple(sorted(column_group_rows))
    if tuple(raw_column_groups) != canonical_column_groups or len(
        {row[0] for row in canonical_column_groups}
    ) != len(canonical_column_groups):
        raise ValueError("floorplan column groups are not canonical and unique")

    cell_counts: dict[tuple[int, int], int] = {}
    cell_sums: dict[tuple[int, int], int] = {}
    for _group, row, column, member_count, terms in canonical_column_groups:
        cell = (row, column)
        cell_counts[cell] = cell_counts.get(cell, 0) + member_count
        cell_sums[cell] = cell_sums.get(cell, 0) + sum(
            2 * count * block_by_name[owner][0] + center_offset2_sum
            for owner, count, center_offset2_sum in terms
        )
    for left_cell, left_count in cell_counts.items():
        for right_cell, right_count in cell_counts.items():
            if left_cell[0] != right_cell[0] or left_cell[1] >= right_cell[1]:
                continue
            if (
                right_count * cell_sums[left_cell]
                >= left_count * cell_sums[right_cell]
            ):
                raise ValueError(
                    "placement blocks violate floorplan functional-group order"
                )
    if canonical_column_groups:
        payload["floorplan_column_groups"] = [
            [group, row, column, count, [list(term) for term in terms]]
            for group, row, column, count, terms in canonical_column_groups
        ]
    minimum_occupancy = canonical_minimum_structural_occupancy(
        getattr(certificate, "declared_minimum_structural_occupancy", None)
    )
    if minimum_occupancy is not None:
        payload["declared_minimum_structural_occupancy"] = minimum_occupancy
    minimum_od, od_area, od_layers, _od_policy_sha = functional_od_policy_binding(
        getattr(certificate, "declared_minimum_functional_od_occupancy", None),
        getattr(certificate, "functional_od_area_nm2", 0),
        getattr(certificate, "functional_od_layer_keys", ()),
    )
    if minimum_od is not None:
        payload["declared_minimum_functional_od_occupancy"] = minimum_od
        payload["functional_od_area_nm2"] = od_area
        payload["functional_od_layer_keys"] = [list(key) for key in od_layers]
    declared_outline_width_nm = getattr(
        certificate, "declared_outline_width_nm", None
    )
    if declared_outline_width_nm is not None:
        if type(declared_outline_width_nm) is not int or declared_outline_width_nm <= 0:
            raise ValueError("declared outline width must be a positive exact integer nm")
        if bbox_nm[2] - bbox_nm[0] != declared_outline_width_nm:
            raise ValueError("placement bbox does not satisfy declared outline width")
        payload["declared_outline_width_nm"] = declared_outline_width_nm
    axis_alignment_groups = _canonical_axis_alignment_groups(
        getattr(certificate, "axis_alignment_groups", ()),
        blocks,
    )
    if axis_alignment_groups:
        payload["axis_alignment_groups"] = [
            [[name, offset2] for name, offset2 in group]
            for group in axis_alignment_groups
        ]
        payload["schema"] = "clay.final-placement-state.v4"
    center_alignment_pairs = _canonical_center_alignment_pairs(
        getattr(certificate, "center_alignment_pairs", ()),
        blocks,
    )
    if center_alignment_pairs:
        payload["center_alignment_pairs"] = [
            [left, right, axis]
            for left, right, axis in center_alignment_pairs
        ]
    aggregate_envelope_orders = _canonical_aggregate_envelope_orders(
        getattr(certificate, "aggregate_envelope_orders", ()),
        blocks,
    )
    if aggregate_envelope_orders:
        payload["aggregate_envelope_orders"] = [
            [before, after, axis, list(before_blocks), list(after_blocks)]
            for before, after, axis, before_blocks, after_blocks
            in aggregate_envelope_orders
        ]
    declared_gap_nm = getattr(certificate, "declared_gap_nm", 0)
    if type(declared_gap_nm) is not int or declared_gap_nm < 0:
        raise ValueError("declared placement gap must be an exact non-negative nm")
    hierarchical_macro_orders = _canonical_hierarchical_macro_orders(
        getattr(certificate, "hierarchical_macro_orders", ()),
        blocks,
        gap_nm=declared_gap_nm,
    )
    has_track_floor = any(
        strength in {
            "center_order_with_track_floor",
            "member_before_aggregate_center_with_track_floor",
        }
        for _scope, _before, _after, _axis, strength, _left, _right
        in hierarchical_macro_orders
    )
    if has_track_floor:
        if declared_gap_nm <= 0:
            raise ValueError("hierarchical track floor requires a positive declared gap")
        payload["declared_gap_nm"] = declared_gap_nm
    if hierarchical_macro_orders:
        payload["hierarchical_macro_orders"] = [
            [
                scope,
                before_node,
                after_node,
                axis,
                strength,
                list(before_blocks),
                list(after_blocks),
            ]
            for (
                scope,
                before_node,
                after_node,
                axis,
                strength,
                before_blocks,
                after_blocks,
            ) in hierarchical_macro_orders
        ]
    hierarchical_alignment_tolerance_scales_nm = tuple(
        getattr(
            certificate,
            "hierarchical_alignment_tolerance_scales_nm",
            (),
        )
        or ()
    )
    hierarchical_macro_alignments = _canonical_hierarchical_macro_alignments(
        getattr(certificate, "hierarchical_macro_alignments", ()),
        blocks,
        tolerance_scales_nm=hierarchical_alignment_tolerance_scales_nm,
    )
    if hierarchical_macro_alignments:
        payload["hierarchical_macro_alignments"] = [
            (
                [scope, axis, [[node, list(members)] for node, members in groups]]
                if len(row) == 3
                else [
                    scope,
                    axis,
                    row[2],
                    row[3],
                    [[node, list(members)] for node, members in row[4]],
                ]
                if len(row) == 5
                else [
                    scope,
                    axis,
                    row[2],
                    [[node, list(members)] for node, members in row[3]],
                ]
            )
            for row in hierarchical_macro_alignments
            for scope, axis, groups in [
                (
                    row[0],
                    row[1],
                    row[2]
                    if len(row) == 3
                    else row[3]
                    if len(row) == 4
                    else row[4],
                )
            ]
        ]
    if hierarchical_alignment_tolerance_scales_nm:
        payload["hierarchical_alignment_tolerance_scales_nm"] = [
            [axis, scale_nm]
            for axis, scale_nm in hierarchical_alignment_tolerance_scales_nm
        ]
    excluded = getattr(certificate, 'collision_excluded_layers', ())
    exclusion_pdk = getattr(certificate, 'collision_exclusion_pdk_id', '')
    if excluded:
        payload['collision_exclusion'] = {'pdk_id': exclusion_pdk,
                                          'layers': [list(layer) for layer in excluded]}
    full2d_compaction_graph = getattr(
        certificate, "full2d_compaction_graph", None
    )
    raw_full2d_solver_evidence = getattr(
        certificate, "full2d_solver_evidence", ()
    )
    if type(raw_full2d_solver_evidence) is not tuple:
        raise TypeError("full-2D solver evidence must be one exact tuple")
    raw_layered_collision_footprints = getattr(
        certificate, "layered_collision_footprints", ()
    )
    if type(raw_layered_collision_footprints) is not tuple:
        raise TypeError("layered collision footprints must be one exact tuple")
    block_dimensions = {
        name: (bbox[2] - bbox[0], bbox[3] - bbox[1]) for name, bbox in blocks
    }
    layered_collision_footprints = []
    for row in raw_layered_collision_footprints:
        if (
            type(row) is not tuple
            or len(row) != 2
            or type(row[0]) is not str
            or row[0] not in block_dimensions
            or type(row[1]) is not tuple
            or not row[1]
        ):
            raise ValueError("layered collision footprint ownership is malformed")
        owner, atoms = row
        width, height = block_dimensions[owner]
        for atom in atoms:
            if (
                type(atom) is not tuple
                or len(atom) != 6
                or any(type(value) is not int for value in atom)
                or atom[0] < 0
                or atom[1] < 0
                or not (
                    0 <= atom[2] < atom[4] <= width
                    and 0 <= atom[3] < atom[5] <= height
                )
            ):
                raise ValueError("layered collision footprint atom is malformed")
        if atoms != tuple(sorted(set(atoms))):
            raise ValueError("layered collision footprint atoms are not canonical")
        layered_collision_footprints.append((owner, atoms))
    layered_collision_footprints = tuple(layered_collision_footprints)
    if (
        layered_collision_footprints
        != tuple(sorted(layered_collision_footprints))
        or len({owner for owner, _atoms in layered_collision_footprints})
        != len(layered_collision_footprints)
    ):
        raise ValueError("layered collision footprints must be canonical and unique")
    if layered_collision_footprints:
        payload["layered_collision_footprints"] = [
            [owner, [list(atom) for atom in atoms]]
            for owner, atoms in layered_collision_footprints
        ]
    raw_overlap_compatible_pairs = getattr(
        certificate, "overlap_compatible_pairs", ()
    )
    if type(raw_overlap_compatible_pairs) is not tuple:
        raise TypeError("overlap-compatible pairs must be one exact tuple")
    overlap_compatible_pairs = tuple(raw_overlap_compatible_pairs)
    block_names = {name for name, _bbox in blocks}
    if (
        overlap_compatible_pairs != tuple(sorted(set(overlap_compatible_pairs)))
        or any(
            type(pair) is not tuple
            or len(pair) != 2
            or type(pair[0]) is not str
            or type(pair[1]) is not str
            or pair[0] >= pair[1]
            or pair[0] not in block_names
            or pair[1] not in block_names
            for pair in overlap_compatible_pairs
        )
    ):
        raise ValueError("overlap-compatible pairs must be canonical and owned")
    if overlap_compatible_pairs:
        payload["overlap_compatible_pairs"] = [
            list(pair) for pair in overlap_compatible_pairs
        ]
    if full2d_compaction_graph is not None:
        from clay.layout.place.full2d_geometry_boundary import require_full2d_geometry_proof
        require_full2d_geometry_proof(full2d_compaction_graph, blocks,
            overlap_compatible_pairs=overlap_compatible_pairs,
            layered_footprints_nm=layered_collision_footprints, solver_evidence=raw_full2d_solver_evidence)
        payload["full2d_compaction_graph"] = full2d_compaction_graph.payload()
        if raw_full2d_solver_evidence:
            payload["full2d_solver_evidence"] = _canonical_full2d_solver_evidence(
                raw_full2d_solver_evidence,
                graph_sha256=full2d_compaction_graph.sha256,
                pair_constraint_modes=tuple(
                    row.mode for row in full2d_compaction_graph.pair_constraints
                ),
                graph=full2d_compaction_graph,
            )
    elif raw_full2d_solver_evidence:
        raise ValueError("full-2D solver evidence requires its canonical graph")

    raw_members = tuple(
        getattr(certificate, "materialized_block_members", ()) or ()
    )
    member_rows = []
    for row in raw_members:
        if not isinstance(row, (tuple, list)) or len(row) != 2:
            raise ValueError("materialized block members must contain exact typed rows")
        master, members = row
        frozen = tuple(members) if isinstance(members, (tuple, list)) else ()
        if (
            not isinstance(master, str) or not master or not frozen
            or any(not isinstance(member, str) or not member for member in frozen)
        ):
            raise ValueError("materialized block members must contain exact typed rows")
        member_rows.append((master, tuple(sorted(frozen))))
    materialized_members = tuple(sorted(member_rows))
    raw_materialized_bboxes = tuple(
        getattr(certificate, "materialized_block_bboxes", ()) or ()
    )
    materialized_bboxes = (
        _canonical_block_bboxes(raw_materialized_bboxes)
        if raw_materialized_bboxes else ()
    )
    if bool(materialized_members) != bool(materialized_bboxes):
        raise ValueError("materialized block members and bboxes must be signed together")
    if materialized_members:
        masters = [master for master, _members in materialized_members]
        members = [
            member
            for _master, row_members in materialized_members
            for member in row_members
        ]
        if (
            len(masters) != len(set(masters))
            or set(masters) != {name for name, _bbox in blocks}
            or len(members) != len(set(members))
            or {name for name, _bbox in materialized_bboxes} != set(members)
            or any(len(row) != len(set(row)) for _master, row in materialized_members)
        ):
            raise ValueError("materialized block members must be an exact master partition")
        payload["materialized_block_members"] = [
            [master, list(members)] for master, members in materialized_members
        ]
        payload["materialized_block_bboxes"] = [
            [name, list(bbox)] for name, bbox in materialized_bboxes
        ]

    raw_source_members = getattr(
        certificate, "materialized_block_source_members", ()
    )
    if type(raw_source_members) is not tuple:
        raise TypeError("materialized block source ownership must be an exact tuple")
    source_rows = []
    all_sources = []
    for row in raw_source_members:
        if type(row) is not tuple or len(row) != 2:
            raise TypeError(
                "materialized block source ownership row must be an exact pair"
            )
        owner, sources = row
        if (
            type(owner) is not str
            or not owner
            or type(sources) is not tuple
            or not sources
            or any(type(source) is not str or not source for source in sources)
        ):
            raise TypeError("materialized block source ownership row is malformed")
        if sources != tuple(sorted(sources)) or len(set(sources)) != len(sources):
            raise ValueError(
                "materialized block source ownership must be canonical and unique"
            )
        source_rows.append((owner, sources))
        all_sources.extend(sources)
    canonical_source_rows = tuple(source_rows)
    if canonical_source_rows:
        if (
            canonical_source_rows != tuple(sorted(canonical_source_rows))
            or len({owner for owner, _sources in canonical_source_rows})
            != len(canonical_source_rows)
            or {owner for owner, _sources in canonical_source_rows}
            != {name for name, _bbox in materialized_bboxes}
            or len(all_sources) != len(set(all_sources))
        ):
            raise ValueError(
                "materialized block source ownership must be an exact partition"
            )
        payload["materialized_block_source_members"] = [
            [owner, list(sources)] for owner, sources in canonical_source_rows
        ]

    raw_source_bboxes = getattr(
        certificate, "materialized_source_member_bboxes", ()
    )
    if type(raw_source_bboxes) is not tuple:
        raise TypeError("materialized source member bboxes must be an exact tuple")
    source_bbox_rows = []
    for row in raw_source_bboxes:
        if type(row) is not tuple or len(row) != 2:
            raise TypeError(
                "materialized source member bbox row must be an exact pair"
            )
        source, bbox = row
        if (
            type(source) is not str
            or not source
            or type(bbox) is not tuple
            or len(bbox) != 4
            or any(type(value) is not int for value in bbox)
            or bbox[2] <= bbox[0]
            or bbox[3] <= bbox[1]
        ):
            raise TypeError("materialized source member bbox row is malformed")
        source_bbox_rows.append((source, bbox))
    canonical_source_bboxes = tuple(source_bbox_rows)
    if canonical_source_bboxes:
        if not canonical_source_rows:
            raise ValueError(
                "materialized source member bboxes require source ownership"
            )
        source_owner = {
            source: owner
            for owner, sources in canonical_source_rows
            for source in sources
        }
        physical_bboxes = dict(materialized_bboxes)
        if (
            canonical_source_bboxes != tuple(sorted(canonical_source_bboxes))
            or len({source for source, _bbox in canonical_source_bboxes})
            != len(canonical_source_bboxes)
        ):
            raise ValueError(
                "materialized source member bboxes must be canonical and unique"
            )
        bbox_sources = {source for source, _bbox in canonical_source_bboxes}
        if not bbox_sources <= set(source_owner):
            raise ValueError(
                "materialized source member bboxes name foreign source ownership"
            )
        for owner, sources in canonical_source_rows:
            covered = set(sources).intersection(bbox_sources)
            if covered and covered != set(sources):
                raise ValueError(
                    "materialized source member bboxes must cover exact source "
                    f"ownership per physical owner {owner!r}"
                )
        for source, bbox in canonical_source_bboxes:
            owner_bbox = physical_bboxes[source_owner[source]]
            if not (
                owner_bbox[0] <= bbox[0] < bbox[2] <= owner_bbox[2]
                and owner_bbox[1] <= bbox[1] < bbox[3] <= owner_bbox[3]
            ):
                raise ValueError(
                    "materialized source member bbox must stay inside its physical owner"
                )
        payload["materialized_source_member_bboxes"] = [
            [source, list(bbox)] for source, bbox in canonical_source_bboxes
        ]

    raw_axis_projections = getattr(
        certificate, "matching_pair_axis_projections", ()
    )
    if type(raw_axis_projections) is not tuple:
        raise TypeError("matching-axis projection inventory must be an exact tuple")
    if any(type(row) is not MatchingPairAxisProjection for row in raw_axis_projections):
        raise TypeError("matching-axis projection inventory contains a foreign row")
    canonical_axis_projections = tuple(sorted(raw_axis_projections))
    if (
        raw_axis_projections != canonical_axis_projections
        or len({row.pair for row in canonical_axis_projections})
        != len(canonical_axis_projections)
    ):
        raise ValueError("matching-axis projections must be canonical and unique")
    if canonical_axis_projections:
        materialized_names = {
            master for master, _members in materialized_members
        }
        if not materialized_names or any(
            row.source_master not in materialized_names
            for row in canonical_axis_projections
        ):
            raise ValueError(
                "matching-axis projection source master lacks materialized ownership"
            )
        payload["matching_pair_axis_projections"] = [
            {
                "axis": row.axis,
                "delta_nm": list(row.delta_nm),
                "final_coordinate4_nm": row.final_coordinate4_nm,
                "final_logical_coordinate4_nm": row.final_logical_coordinate4_nm,
                "final_member_bboxes_nm": [
                    list(bbox) for bbox in row.final_member_bboxes_nm
                ],
                "pair": list(row.pair),
                "source_coordinate4_nm": row.source_coordinate4_nm,
                "source_logical_coordinate4_nm": row.source_logical_coordinate4_nm,
                "source_member_bboxes_nm": [
                    list(bbox) for bbox in row.source_member_bboxes_nm
                ],
                "source_master": row.source_master,
            }
            for row in canonical_axis_projections
        ]

    basis = getattr(certificate, "placement_causality_basis", None)
    if basis is not None:
        if not isinstance(basis, PlacementCausalityBasis):
            raise ValueError("final placement certificate has a foreign causality basis")
        verify_placement_causality_basis(certificate)
        payload["placement_causality_basis_sha256"] = basis.sha256
        payload["placement_declaration_refs"] = list(
            getattr(certificate, "placement_declaration_refs", ()) or ()
        )
        if not axis_alignment_groups:
            payload["schema"] = "clay.final-placement-state.v3"
    if minimum_occupancy is not None:
        payload["schema"] = "clay.final-placement-state.v5"
    if minimum_od is not None:
        payload["schema"] = "clay.final-placement-state.v6"
    if canonical_columns:
        payload["schema"] = "clay.final-placement-state.v7"
    if canonical_column_groups:
        payload["schema"] = "clay.final-placement-state.v9"
    if canonical_axis_projections:
        payload["schema"] = "clay.final-placement-state.v12"
    if canonical_source_rows:
        payload["schema"] = "clay.final-placement-state.v14"
    raw_group_route_placement = getattr(
        certificate, "group_route_placement_certificate", None
    )
    if raw_group_route_placement is not None:
        from clay.layout.route.group_ports import GroupRoutePlacementCertificate

        if type(raw_group_route_placement) is not GroupRoutePlacementCertificate:
            raise TypeError("group-route placement certificate has a foreign type")
        payload["group_route_placement_certificate"] = (
            raw_group_route_placement.canonical_data()
        )
        payload["schema"] = (
            "clay.final-placement-state.v14"
            if canonical_source_rows
            else "clay.final-placement-state.v13"
        )
    if declared_bottom_edge_keepout_nm > 0:
        payload["schema"] = "clay.final-placement-state.v15"
    if raw_fact_projections:
        payload["schema"] = "clay.final-placement-state.v16"
    if canonical_source_bboxes:
        payload["schema"] = "clay.final-placement-state.v17"
    if center_alignment_pairs:
        payload["schema"] = "clay.final-placement-state.v18"
    if aggregate_envelope_orders:
        payload["schema"] = "clay.final-placement-state.v19"
    if hierarchical_macro_orders or hierarchical_macro_alignments:
        payload["schema"] = "clay.final-placement-state.v20"
    if has_track_floor:
        payload["schema"] = "clay.final-placement-state.v21"
    if full2d_compaction_graph is not None:
        payload["schema"] = "clay.final-placement-state.v22"
    if overlap_compatible_pairs:
        payload["schema"] = "clay.final-placement-state.v23"
    if layered_collision_footprints:
        payload["schema"] = "clay.final-placement-state.v24"
        if any(
            len(tuple(atom[:2] for atom in atoms))
            != len(set(atom[:2] for atom in atoms))
            for _owner, atoms in layered_collision_footprints
        ):
            payload["schema"] = "clay.final-placement-state.v25"
    if any(len(row) == 4 for row in hierarchical_macro_alignments):
        payload["schema"] = "clay.final-placement-state.v26"
    if any(len(row) == 5 for row in hierarchical_macro_alignments):
        payload["schema"] = "clay.final-placement-state.v27"
    if hierarchical_alignment_tolerance_scales_nm:
        payload["schema"] = "clay.final-placement-state.v28"
    if raw_reference_obligations:
        payload["schema"] = "clay.final-placement-state.v29"
    if raw_full2d_solver_evidence:
        payload["schema"] = "clay.final-placement-state.v30"
    if raw_cavity_projections:
        payload["schema"] = "clay.final-placement-state.v31"
    if raw_guest_slot_projections:
        payload["schema"] = "clay.final-placement-state.v32"
    edge_rail_reservation = getattr(certificate, "edge_rail_reservation", None)
    if edge_rail_reservation is not None:
        from clay.layout.place.edge_rail_reservation import EdgeRailReservationIR

        if type(edge_rail_reservation) is not EdgeRailReservationIR:
            raise TypeError("edge rail reservation must be an exact EdgeRailReservationIR")
        rail_blocks = dict(certificate.blocks)
        if not edge_rail_reservation.replay(
            dims={name: (box[2] - box[0], box[3] - box[1]) for name, box in rail_blocks.items()},
            positions={name: box[:2] for name, box in rail_blocks.items()},
            physical_bbox_nm=certificate.bbox_nm,
        ):
            raise ValueError("edge rail reservation does not replay final placement")
        payload["edge_rail_reservation"] = edge_rail_reservation.canonical_data()
        payload["schema"] = "clay.final-placement-state.v33"
    payload = with_layered_assembly_payload(certificate, payload)
    if flexible_projections:
        payload["flexible_envelope_projections"] = projection_payload(flexible_projections)
        payload["schema"] = "clay.final-placement-state.v35"
    cross_row = getattr(certificate, "native_cross_row_final_proof", None)
    if cross_row is not None:
        from clay.layout.place.cross_row_matching import NativeCrossRowFinalProof
        if type(cross_row) is not NativeCrossRowFinalProof:
            raise TypeError("final native cross-row proof is foreign")
        payload["native_cross_row_final_proof_sha256"] = cross_row.sha256
        payload["schema"] = "clay.final-placement-state.v36"
    same_row = getattr(certificate, "native_same_row_final_proof", None)
    if same_row is not None:
        from clay.layout.place.native_same_row_matching import NativeSameRowFinalProof
        if type(same_row) is not NativeSameRowFinalProof:
            raise TypeError("final native same-row proof is foreign")
        same_row.__post_init__()
        payload["native_same_row_final_proof_sha256"] = same_row.sha256
        payload["schema"] = "clay.final-placement-state.v37"
    return payload


def final_placement_state_digest(certificate: object) -> str:
    return _sha256(final_placement_state_payload(certificate))


__all__ = [
    "MatchingPairAxisProjection",
    "PlacementCausalityBasis",
    "PlacementCorridorBoundary",
    "build_placement_causality_basis",
    "canonical_minimum_structural_occupancy",
    "final_placement_state_digest",
    "final_placement_state_payload",
    "functional_od_policy_binding",
    "structural_occupancy_policy_binding",
    "verify_placement_causality_basis",
]
