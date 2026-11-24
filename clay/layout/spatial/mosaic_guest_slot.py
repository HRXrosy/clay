"""Coordinate-free source slot topology and current-geometry rank planning.

Reference envelopes are consumed only by the source topology projector. The
generated planner accepts current native rank anchors and envelopes separately.
These types support local materialization; they do not provide a solver result,
placement admission, or a physical signoff certificate.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re

from clay.layout.spatial.abstract_ir import AbstractSpatialIR, SpatialFactKind
from clay.layout.spatial.derive import derive_abstract_spatial_ir
from clay.layout.spatial.intent_projection import DiscreteMosaicIntent, project_discrete_mosaic
from clay.layout.spatial.semantic import SemanticOverlay


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()


def _sha(value):
    _require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value), "invalid authority SHA")


def _text(value):
    _require(type(value) is str and value and value == value.strip(), "invalid node identity")


@dataclass(frozen=True, slots=True)
class MosaicSlotBoundary:
    row_index: int
    left_rank: int
    right_rank: int
    left_node: str
    right_node: str

    def __post_init__(self):
        _require(all(type(value) is int and value >= 0
                     for value in (self.row_index, self.left_rank, self.right_rank)),
                 "slot row/rank must be nonnegative integers")
        _require(self.right_rank == self.left_rank + 1, "slot boundary ranks must be adjacent")
        _text(self.left_node)
        _text(self.right_node)
        _require(self.left_node != self.right_node, "slot boundary nodes coincide")

    def canonical_data(self):
        return {"row_index": self.row_index, "left_rank": self.left_rank,
                "right_rank": self.right_rank, "left_node": self.left_node, "right_node": self.right_node}


@dataclass(frozen=True, slots=True)
class MosaicGuestSlot:
    source_overlay_sha256: str
    source_asir_sha256: str
    source_mosaic_sha256: str
    host_nodes: tuple[str, ...]
    guest_node: str
    boundaries: tuple[MosaicSlotBoundary, ...]

    def __post_init__(self):
        for sha in (self.source_overlay_sha256, self.source_asir_sha256, self.source_mosaic_sha256):
            _sha(sha)
        _require(type(self.host_nodes) is tuple and self.host_nodes
                 and tuple(sorted(set(self.host_nodes))) == self.host_nodes, "slot host cover differs")
        for node in (*self.host_nodes, self.guest_node):
            _text(node)
        _require(self.guest_node not in self.host_nodes, "guest belongs to the host")
        _require(type(self.boundaries) is tuple and len(self.boundaries) >= 2
                 and all(type(row) is MosaicSlotBoundary for row in self.boundaries),
                 "slot requires exact boundaries spanning at least two full rows")
        _require(tuple(row.row_index for row in self.boundaries)
                 == tuple(sorted({row.row_index for row in self.boundaries})), "duplicate or unordered slot rows")
        _require(len({row.left_rank for row in self.boundaries}) == 1, "slot seam rank differs between rows")
        _require(all(row.left_node in self.host_nodes and row.right_node in self.host_nodes
                     for row in self.boundaries), "slot boundary is outside host cover")

    def topology_data(self):
        return {"axis": "horizontal", "host_nodes": list(self.host_nodes), "guest_node": self.guest_node,
                "boundaries": [row.canonical_data() for row in self.boundaries]}

    def canonical_data(self):
        return {"schema": "clay.mosaic-guest-slot-intent/v1", **self.topology_data(),
                "source_overlay_sha256": self.source_overlay_sha256,
                "source_asir_sha256": self.source_asir_sha256,
                "source_mosaic_sha256": self.source_mosaic_sha256,
                "source_authority": "reference_projected_topology",
                "scope": "local_materialization_only",
                "outer_guest_constraint_required": True, "promotion": "REJECT"}

    @property
    def sha256(self):
        return _hash(self.canonical_data())


def derive_mosaic_guest_slot(*, overlay, asir, mosaic, guest_owner):
    """Prove measured positive envelope relations over a complete native ASIR."""
    if type(overlay) is not SemanticOverlay or type(asir) is not AbstractSpatialIR:
        raise TypeError("slot observation requires exact SemanticOverlay and AbstractSpatialIR")
    if type(mosaic) is not DiscreteMosaicIntent:
        raise TypeError("slot observation requires exact DiscreteMosaicIntent")
    _text(guest_owner)
    _require(overlay.source_kind == "reference_gds", "slot observation requires reference authority")
    _require(asir.source_clm_sha256 == overlay.sha256, "slot CLM authority differs")
    _require(mosaic.source_asir_sha256 == asir.sha256, "slot mosaic ASIR authority differs")
    _require(asir.nodes == overlay.node_ids, "slot ASIR node cover is incomplete")
    fresh = derive_abstract_spatial_ir(overlay)
    _require(asir.horizontal == fresh.horizontal and asir.vertical == fresh.vertical,
             "slot requires complete freshly derived ASIR separation DAGs")
    basic_kinds = {SpatialFactKind.EXACT_PARTITION, SpatialFactKind.LATTICE_SEQUENCE,
                   SpatialFactKind.ALIGNMENT, SpatialFactKind.CLASS_RANK_SEQUENCE}
    _require(tuple(row for row in asir.hard_facts if row.kind in basic_kinds)
             == tuple(row for row in fresh.hard_facts if row.kind in basic_kinds),
             "slot ASIR basic structural fact cover differs from fresh derivation")
    native_mosaic = project_discrete_mosaic(asir, mosaic_id=mosaic.mosaic_id, logical_owners=mosaic.members)
    _require(native_mosaic == mosaic, "slot mosaic differs from native structural projection")
    occurrences = {row.node_id: row for row in overlay.occurrences}
    host_nodes = tuple(sorted(row.node_id for row in mosaic.occurrences))
    _require(host_nodes == tuple(row.node_id for row in overlay.occurrences
                                 if row.logical_owner in mosaic.members), "slot host exact occurrence cover differs")
    _require(all(row.transform == occurrences[row.node_id].physical.orientation
                 for row in mosaic.occurrences), "slot host occurrence orientation differs")
    guests = tuple(row for row in overlay.occurrences if row.logical_owner == guest_owner)
    _require(len(guests) == 1 and guest_owner not in mosaic.members, "slot requires one foreign guest occurrence")
    guest = guests[0]
    _require(all(occurrences[node].master_artwork_sha256 != guest.master_artwork_sha256 for node in host_nodes),
             "slot guest must be a foreign artwork class")
    g = guest.physical.envelope_nm
    full_width = max(map(len, mosaic.rows))
    boundaries = []
    for row_index, row in enumerate(mosaic.rows):
        if len(row) != full_width:
            continue
        choices = []
        for rank, (left, right) in enumerate(zip(row, row[1:])):
            l = occurrences[left.node_id].physical.envelope_nm
            r = occurrences[right.node_id].physical.envelope_nm
            if (l[2] <= g[0] and g[2] <= r[0]
                    and max(l[1], g[1]) < min(l[3], g[3])
                    and max(r[1], g[1]) < min(r[3], g[3])):
                _require(asir.horizontal.reaches(left.node_id, guest.node_id)
                         and asir.horizontal.reaches(guest.node_id, right.node_id),
                         "slot positive boundary order is absent from complete ASIR")
                choices.append(MosaicSlotBoundary(row_index, rank, rank + 1, left.node_id, right.node_id))
        _require(len(choices) == 1, "slot seam lacks unique positive boundary order and vertical overlap")
        boundaries.append(choices[0])
    return MosaicGuestSlot(overlay.sha256, asir.sha256, mosaic.sha256, host_nodes,
                           guest.node_id, tuple(boundaries))


@dataclass(frozen=True, slots=True)
class MosaicGuestSlotPlan:
    rank_positions_nm: tuple[int, ...]
    node_dx_nm: tuple[tuple[str, int], ...]
    node_target_x_nm: tuple[tuple[str, int], ...]
    guest_origin_x_interval_nm: tuple[int, int]
    expansion_nm: int
    source_slot_sha256: str
    geometry_inputs_sha256: str
    outer_guest_constraint_required: bool = True
    promotion: str = "REJECT"

    def canonical_data(self):
        return {"schema": "clay.generated-mosaic-slot-plan/v1", "scope": "local_materialization_only",
                "rank_positions_nm": list(self.rank_positions_nm),
                "node_dx_nm": [list(row) for row in self.node_dx_nm],
                "node_target_x_nm": [list(row) for row in self.node_target_x_nm],
                "guest_origin_x_interval_nm": list(self.guest_origin_x_interval_nm),
                "expansion_nm": self.expansion_nm, "source_slot_sha256": self.source_slot_sha256,
                "geometry_inputs_sha256": self.geometry_inputs_sha256,
                "outer_guest_constraint_required": self.outer_guest_constraint_required, "promotion": self.promotion}

    @property
    def sha256(self):
        return _hash(self.canonical_data())


def _box(box):
    if type(box) is not tuple or len(box) != 4 or any(type(value) is not int for value in box):
        raise TypeError("generated envelope must be an exact four-integer tuple")
    _require(box[0] < box[2] and box[1] < box[3], "generated envelope must have positive area")


def plan_mosaic_guest_slot(*, slot, mosaic, rank_origins_nm, host_boxes_nm,
                          guest_box_nm, source_gap_nm, grid_nm):
    """Grow one rank seam from current envelopes, retaining independent guest y.

`rank_origins_nm` are the caller's actual native target_x anchors, not envelope
centres. Their initial grid phase is preserved; all new translations are grid
multiples. The result supplies necessary horizontal outer-CP bounds only.
    """
    if type(slot) is not MosaicGuestSlot or type(mosaic) is not DiscreteMosaicIntent:
        raise TypeError("slot planning requires exact typed slot and mosaic")
    _require(slot.source_mosaic_sha256 == mosaic.sha256 and slot.source_asir_sha256 == mosaic.source_asir_sha256,
             "generated slot mosaic authority differs")
    host_nodes = tuple(sorted(row.node_id for row in mosaic.occurrences))
    _require(slot.host_nodes == host_nodes, "generated slot host exact cover differs")
    width = max(map(len, mosaic.rows))
    full_indices = tuple(index for index, row in enumerate(mosaic.rows) if len(row) == width)
    _require(tuple(row.row_index for row in slot.boundaries) == full_indices, "generated slot full-row cover differs")
    for row in slot.boundaries:
        _require(row.right_rank < width and mosaic.rows[row.row_index][row.left_rank].node_id == row.left_node
                 and mosaic.rows[row.row_index][row.right_rank].node_id == row.right_node,
                 "generated slot boundary identity differs")
    if (type(rank_origins_nm) is not tuple or len(rank_origins_nm) != width
            or any(type(value) is not int for value in rank_origins_nm)):
        raise TypeError("native rank origins must be a complete exact integer tuple")
    _require(all(a < b for a, b in zip(rank_origins_nm, rank_origins_nm[1:])), "native rank origins must increase")
    if type(source_gap_nm) is not int or type(grid_nm) is not int:
        raise TypeError("source gap and manufacturing grid must be exact integers")
    _require(source_gap_nm > 0 and grid_nm > 0, "source gap and manufacturing grid must be positive")
    _box(guest_box_nm)
    if type(host_boxes_nm) is not tuple:
        raise TypeError("generated host envelopes must be an exact tuple")
    boxes = {}
    for row in host_boxes_nm:
        if type(row) is not tuple or len(row) != 2:
            raise TypeError("generated host envelope row must be one exact pair")
        node, box = row
        _text(node)
        _box(box)
        _require(node not in boxes, "duplicate generated host envelope")
        boxes[node] = box
    _require(tuple(sorted(boxes)) == host_nodes, "generated host envelope exact cover differs")
    down = lambda value: (value // grid_nm) * grid_nm
    up = lambda value: -((-value) // grid_nm) * grid_nm
    lower = up(max(boxes[row.left_node][2] + source_gap_nm - guest_box_nm[0] for row in slot.boundaries))
    upper_before = min(boxes[row.right_node][0] - source_gap_nm - guest_box_nm[2] for row in slot.boundaries)
    quantum = 2 * grid_nm
    needed = max(0, lower - upper_before)
    expansion = ((needed + quantum - 1) // quantum) * quantum
    right_rank = slot.boundaries[0].right_rank
    positions = tuple(value + (expansion if rank >= right_rank else 0)
                      for rank, value in enumerate(rank_origins_nm))
    intervals = {f"{owner}[{ordinal:03d}]": (left, right)
                 for owner, ordinal, left, right in mosaic.rank_intervals}
    translations, targets = [], []
    for row in mosaic.rows:
        for rank, occurrence in enumerate(row):
            if len(row) == width:
                old, new = rank_origins_nm[rank], positions[rank]
            else:
                left, right = intervals[occurrence.node_id]
                old = (rank_origins_nm[left] + rank_origins_nm[right]) // 2
                new = (positions[left] + positions[right]) // 2
            _require((new - old) % grid_nm == 0, "generated rank translation leaves manufacturing grid")
            translations.append((occurrence.node_id, new - old))
            targets.append((occurrence.node_id, new))
    upper = down(upper_before + expansion)
    _require(lower <= upper, "generated slot has no grid-aligned horizontal guest domain")
    geometry_hash = _hash({"rank_origins_nm": rank_origins_nm,
                           "host_boxes_nm": sorted(boxes.items()), "guest_box_nm": guest_box_nm,
                           "source_gap_nm": source_gap_nm, "grid_nm": grid_nm})
    return MosaicGuestSlotPlan(positions, tuple(sorted(translations)), tuple(sorted(targets)),
                               (lower, upper), expansion, slot.sha256, geometry_hash)


# The source materializer constructs these records from current unit polygons;
# arbitrary caller plans do not establish current source or placement authority.


def replay_mosaic_guest_slot_inventory(values):
    if type(values) is not tuple or any(type(row) is not MosaicGuestSlot for row in values):
        raise TypeError("mosaic guest slots require an exact typed tuple")
    for row in values:
        row.__post_init__()
    _require(len({row.source_mosaic_sha256 for row in values}) == len(values),
             "one mosaic cannot receive two guest slot declarations")
    _require(len({row.guest_node for row in values}) == len(values),
             "one guest cannot receive two slot declarations")
    return tuple(sorted(values, key=lambda row: row.source_mosaic_sha256))


def _polygon_records(polygons):
    rows = []
    for polygon in polygons:
        layer, datatype = polygon.layer, polygon.datatype
        points = tuple((point.x, point.y) for point in polygon.polygon.vertices)
        _require(type(layer) is int and type(datatype) is int and layer >= 0 and datatype >= 0,
                 "slot polygon layer/datatype differs")
        _require(len(points) >= 3 and all(type(value) is int for point in points for value in point),
                 "slot polygon vertices differ")
        rows.append((layer, datatype, points))
    _require(rows and len(rows) <= 262144, "slot polygon denominator is empty or unbounded")
    return tuple(rows)


def _records_box(records):
    _require(type(records) is tuple and records and len(records) <= 262144,
             "slot polygon records require one bounded nonempty exact tuple")
    vertex_count = 0
    for row in records:
        _require(type(row) is tuple and len(row) == 3, "slot polygon record differs")
        layer, datatype, vertices = row
        _require(type(layer) is int and type(datatype) is int and 0 <= layer <= 65535
                 and 0 <= datatype <= 65535, "slot polygon layer/datatype differs")
        _require(type(vertices) is tuple and 3 <= len(vertices) <= 4096,
                 "slot polygon vertex count differs")
        _require(all(type(point) is tuple and len(point) == 2
                     and all(type(value) is int for value in point) for point in vertices),
                 "slot polygon vertices must be exact integers")
        vertex_count += len(vertices)
    _require(vertex_count <= 1048576, "slot unit vertices exceed replay bound")
    points = tuple(point for _layer, _datatype, vertices in records for point in vertices)
    _require(points, "slot unit geometry is empty")
    return (min(point[0] for point in points), min(point[1] for point in points),
            max(point[0] for point in points), max(point[1] for point in points))


@dataclass(frozen=True, slots=True)
class MosaicGuestSlotGeometryBinding:
    """Source projection plus actual generated operands, with replayable shifts.

This is a local materialization binding. Its hash is not an independent claim
that the caller supplied current native PCells. The LDO entry rebuilds the
source projection and obtains these operands from the native source inventory;
fresh production admission must rebuild that entry and compare the result.
    """
    slot: MosaicGuestSlot
    mosaic: DiscreteMosaicIntent
    rank_origins_nm: tuple[int, ...]
    # Per node: original native target x/y and complete placed unit artwork.
    baseline_units: tuple[tuple[str, int, int, tuple], ...]
    guest_extent_nm: tuple[int, int, int, int]
    guest_polygon_records: tuple
    source_gap_nm: int
    grid_nm: int
    plan: MosaicGuestSlotPlan
    grid_authority: str = "explicit_generated_unit_translation_grid"

    def __post_init__(self):
        if type(self.slot) is not MosaicGuestSlot or type(self.mosaic) is not DiscreteMosaicIntent:
            raise TypeError("slot geometry requires exact source projection types")
        _text(self.grid_authority)
        _require(type(self.baseline_units) is tuple and self.baseline_units,
                 "slot geometry unit inventory differs")
        expected = self.slot.host_nodes
        _require(tuple(row[0] for row in self.baseline_units) == expected,
                 "slot geometry baseline exact node cover differs")
        for row in self.baseline_units:
            _require(type(row) is tuple and len(row) == 4
                     and type(row[1]) is int and type(row[2]) is int and type(row[3]) is tuple,
                     "slot geometry baseline unit is malformed")
        _require(sum(len(row[3]) for row in self.baseline_units) <= 262144,
                 "slot baseline polygon inventory is unbounded")
        _box(self.guest_extent_nm)
        _require(_records_box(self.guest_polygon_records) == self.guest_extent_nm,
                 "guest current drawn extent differs from its polygon authority")
        rebuilt = plan_mosaic_guest_slot(
            slot=self.slot, mosaic=self.mosaic, rank_origins_nm=self.rank_origins_nm,
            host_boxes_nm=tuple((node, _records_box(records)) for node, _x, _y, records in self.baseline_units),
            guest_box_nm=self.guest_extent_nm, source_gap_nm=self.source_gap_nm, grid_nm=self.grid_nm)
        _require(type(self.plan) is MosaicGuestSlotPlan and self.plan == rebuilt,
                 "slot geometry plan is not the replay of generated operands")
        shifts, targets = dict(rebuilt.node_dx_nm), dict(rebuilt.node_target_x_nm)
        _require(all(x + shifts[node] == targets[node] for node, x, _y, _records in self.baseline_units),
                 "slot rank anchors differ from original native unit target_x")

    def canonical_data(self):
        return {"schema": "clay.mosaic-guest-slot-generated-binding/v1",
                "slot": self.slot.canonical_data(), "mosaic": self.mosaic.canonical_data(),
                "rank_origins_nm": self.rank_origins_nm, "baseline_units": self.baseline_units,
                "guest_extent_nm": self.guest_extent_nm, "guest_polygon_records": self.guest_polygon_records,
                "source_gap_nm": self.source_gap_nm, "grid_nm": self.grid_nm,
                "grid_authority": self.grid_authority,
                "plan": self.plan.canonical_data(), "outer_guest_constraint_required": True}

    @property
    def sha256(self):
        return _hash(self.canonical_data())


def replay_mosaic_guest_slot_geometry(geometry, *, expected_slot_sha256=None):
    """Bind every emitted host polygon/finger to one current unit and its plan."""
    binding = getattr(geometry, "mosaic_guest_slot_binding", None)
    if binding is None:
        _require(expected_slot_sha256 is None, "declared slot geometry binding is missing")
        return None
    if type(binding) is not MosaicGuestSlotGeometryBinding:
        raise TypeError("foreign slot geometry binding")
    binding.__post_init__()
    if expected_slot_sha256 is not None:
        _require(binding.slot.sha256 == expected_slot_sha256, "slot geometry/source declaration differs")
    shifts = dict(binding.plan.node_dx_nm)
    units = {node: (x, y, records) for node, x, y, records in binding.baseline_units}
    expected_polygons = tuple(
        (layer, datatype, tuple((x + shifts[node], y) for x, y in vertices))
        for node, _x, _y, records in binding.baseline_units
        for layer, datatype, vertices in records)
    _require(tuple(sorted(expected_polygons)) == tuple(sorted(_polygon_records(geometry.polygons))),
             "slot emitted polygons differ from complete translated native units")
    fingers = tuple(geometry.fingers)
    _require(len(fingers) == len(binding.mosaic.occurrences), "slot emitted finger denominator differs")
    for occurrence, finger in zip(binding.mosaic.occurrences, fingers):
        x, y, _records = units[occurrence.node_id]
        _require(finger.member == occurrence.logical_owner and finger.orientation == occurrence.transform
                 and (finger.x0 + finger.x1) // 2 == x + shifts[occurrence.node_id] and finger.cy == y,
                 "slot emitted finger identity/target differs")
    return binding.canonical_data()
