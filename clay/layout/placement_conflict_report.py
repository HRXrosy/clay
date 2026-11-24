"""Typed attribution for placement consumers that disagree on one geometry witness."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Mapping


_CAPTURE_SCHEMA = "clay.placement-projection-split-capture/v1"
_REPORT_SCHEMA = "clay.placement-conflict-report/v1"
_SHA_RE = re.compile(r"[0-9a-f]{64}\Z")


class PlacementConflictReportError(ValueError):
    """Projection evidence cannot support one typed placement attribution."""


def _mapping(value: object, *, field: str, keys: set[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise PlacementConflictReportError(f"{field} schema is not exact")
    return value


def _text(value: object, *, field: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise PlacementConflictReportError(f"{field} must be one exact string")
    return value


def _digest(value: object, *, field: str) -> str:
    if type(value) is not str or _SHA_RE.fullmatch(value) is None:
        raise PlacementConflictReportError(f"{field} must be one lowercase digest")
    return value


def _pair(value: object, *, field: str) -> tuple[str, str]:
    if (
        type(value) is not list
        or len(value) != 2
        or any(type(item) is not str or not item for item in value)
        or value[0] == value[1]
    ):
        raise PlacementConflictReportError(f"{field} must be one distinct string pair")
    return value[0], value[1]


def _point(value: object, *, field: str) -> tuple[int, int]:
    if (
        type(value) is not list
        or len(value) != 2
        or any(type(item) is not int for item in value)
    ):
        raise PlacementConflictReportError(f"{field} must be one exact integer point")
    return value[0], value[1]


def _bbox(value: object, *, field: str) -> tuple[int, int, int, int]:
    if (
        type(value) is not list
        or len(value) != 4
        or any(type(item) is not int for item in value)
        or value[2] <= value[0]
        or value[3] <= value[1]
    ):
        raise PlacementConflictReportError(f"{field} must be one positive integer bbox")
    return value[0], value[1], value[2], value[3]


def _binding(value: object, *, field: str) -> tuple[str, str]:
    if type(value) is not list or len(value) != 2:
        raise PlacementConflictReportError(f"{field} binding must be one path/digest pair")
    return _text(value[0], field=f"{field} path"), _digest(
        value[1], field=f"{field} digest"
    )


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _projection_digest(value: Mapping[str, object], *, field: str) -> str:
    claimed = _digest(value.get("projection_sha256"), field=f"{field} digest")
    payload = {key: item for key, item in value.items() if key != "projection_sha256"}
    observed = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    if claimed != observed:
        raise PlacementConflictReportError(f"{field} digest does not replay geometry")
    return claimed


def _owner_boxes(
    value: object,
    *,
    field: str,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    if type(value) is not list or len(value) != 2:
        raise PlacementConflictReportError(f"{field} must contain two owner bboxes")
    return (
        _bbox(value[0], field=f"{field} left"),
        _bbox(value[1], field=f"{field} right"),
    )


def _collision_atoms(
    value: object,
    *,
    field: str,
    owners: tuple[str, str],
    owner_boxes: tuple[tuple[int, int, int, int], tuple[int, int, int, int]],
) -> tuple[
    tuple[tuple[tuple[int, int] | None, tuple[int, int, int, int]], ...],
    tuple[tuple[tuple[int, int] | None, tuple[int, int, int, int]], ...],
]:
    if type(value) is not list or len(value) != 2:
        raise PlacementConflictReportError(f"{field} must contain two owner rows")
    parsed: list[
        tuple[tuple[tuple[int, int] | None, tuple[int, int, int, int]], ...]
    ] = []
    for index, row in enumerate(value):
        if (
            type(row) is not list
            or len(row) != 2
            or row[0] != owners[index]
            or type(row[1]) is not list
            or not row[1]
        ):
            raise PlacementConflictReportError(f"{field} ownership is not exact")
        atoms = []
        for raw_atom in row[1]:
            if type(raw_atom) is not list or len(raw_atom) != 2:
                raise PlacementConflictReportError(f"{field} atom is malformed")
            raw_layer, raw_bbox = raw_atom
            if raw_layer is None:
                layer = None
            elif (
                type(raw_layer) is list
                and len(raw_layer) == 2
                and all(type(item) is int and item >= 0 for item in raw_layer)
            ):
                layer = (raw_layer[0], raw_layer[1])
            else:
                raise PlacementConflictReportError(f"{field} atom layer is malformed")
            bbox = _bbox(raw_bbox, field=f"{field} atom bbox")
            owner_bbox = owner_boxes[index]
            if not (
                owner_bbox[0] <= bbox[0] < bbox[2] <= owner_bbox[2]
                and owner_bbox[1] <= bbox[1] < bbox[3] <= owner_bbox[3]
            ):
                raise PlacementConflictReportError(f"{field} atom escapes its owner")
            atoms.append((layer, bbox))
        atom_rows = tuple(atoms)
        canonical_atoms = tuple(
            sorted(
                set(atom_rows),
                key=lambda atom: (
                    (0, 0, 0, *atom[1])
                    if atom[0] is None
                    else (1, atom[0][0], atom[0][1], *atom[1])
                ),
            )
        )
        if atom_rows != canonical_atoms:
            raise PlacementConflictReportError(
                f"{field} atoms must be canonical and unique"
            )
        parsed.append(tuple(atoms))
    return parsed[0], parsed[1]


def _overlap(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> tuple[tuple[int, int, int, int], int] | None:
    intersection = (
        max(left[0], right[0]),
        max(left[1], right[1]),
        min(left[2], right[2]),
        min(left[3], right[3]),
    )
    if intersection[2] <= intersection[0] or intersection[3] <= intersection[1]:
        return None
    area = (intersection[2] - intersection[0]) * (
        intersection[3] - intersection[1]
    )
    return intersection, area


def _positive_layered_overlaps(
    left: tuple[tuple[tuple[int, int] | None, tuple[int, int, int, int]], ...],
    right: tuple[tuple[tuple[int, int] | None, tuple[int, int, int, int]], ...],
) -> tuple[
    tuple[
        tuple[int, int, int, int],
        tuple[int, int, int, int],
        tuple[int, int, int, int],
        int,
    ],
    ...,
]:
    rows = []
    for left_layer, left_bbox in left:
        for right_layer, right_bbox in right:
            if (
                left_layer is not None
                and right_layer is not None
                and left_layer != right_layer
            ):
                continue
            overlap = _overlap(left_bbox, right_bbox)
            if overlap is not None:
                intersection, area = overlap
                rows.append((left_bbox, right_bbox, intersection, area))
    return tuple(rows)


@dataclass(frozen=True, slots=True, kw_only=True)
class PlacementConflictReport:
    frozen_inputs_path: str
    frozen_inputs_sha256: str
    source_path: str
    source_sha256: str
    runner_path: str
    runner_sha256: str
    reference_gds_path: str
    reference_gds_sha256: str
    failure_receipt_path: str
    failure_receipt_sha256: str
    replay_receipt_path: str
    replay_receipt_sha256: str
    original_classification: str
    error_type: str
    failure_message_sha256: str
    replay_message_sha256: str
    replay_message_exact: bool
    status: str
    category: str
    conflict_pair: tuple[str, str]
    solver_owner_pair: tuple[str, str]
    post_compaction_member_pair: tuple[str, str]
    solver_owner_boxes_nm: tuple[
        tuple[int, int, int, int], tuple[int, int, int, int]
    ]
    post_compaction_owner_boxes_nm: tuple[
        tuple[int, int, int, int], tuple[int, int, int, int]
    ]
    violating_atom_boxes_nm: tuple[
        tuple[int, int, int, int], tuple[int, int, int, int]
    ]
    overlap_bbox_nm: tuple[int, int, int, int]
    overlap_area_nm2: int
    solver_projection_sha256: str
    post_compaction_projection_sha256: str
    mechanical_split_point: str
    solver_compatibility_exempt: bool
    certificate_geometry_legal: bool
    declaration_obligations: tuple[tuple[str, str], ...]
    declaration_resolution: str
    suggested_action: str
    explanation: str

    def __post_init__(self) -> None:
        for field in (
            "frozen_inputs_sha256",
            "source_sha256",
            "runner_sha256",
            "reference_gds_sha256",
            "failure_receipt_sha256",
            "replay_receipt_sha256",
            "failure_message_sha256",
            "replay_message_sha256",
            "solver_projection_sha256",
            "post_compaction_projection_sha256",
        ):
            _digest(getattr(self, field), field=field)
        for field in (
            "frozen_inputs_path",
            "source_path",
            "runner_path",
            "reference_gds_path",
            "failure_receipt_path",
            "replay_receipt_path",
            "original_classification",
            "error_type",
            "explanation",
        ):
            _text(getattr(self, field), field=field)
        if self.status != "ATTRIBUTED_CONFLICT":
            raise PlacementConflictReportError("report status is not ATTRIBUTED_CONFLICT")
        if type(self.replay_message_exact) is not bool:
            raise PlacementConflictReportError("replay message identity is not exact")
        if self.category != "MODEL_SPLIT":
            raise PlacementConflictReportError("report category is not MODEL_SPLIT")
        if self.mechanical_split_point not in {
            "COMPOSITE_MEMBER_TRANSPORT_MISMATCH",
            "SOURCE_LAYER_ENVELOPE_COARSENING",
            "MASTER_LAYERED_ATOM_LOSS",
            "SOLVER_COMPATIBILITY_SCOPE_MISMATCH",
            "SOLVER_CERTIFICATE_GEOMETRY_ILLEGAL",
            "COLLISION_REPLAY_DIVERGENCE",
        }:
            raise PlacementConflictReportError("mechanical split point is unknown")
        if (
            type(self.solver_compatibility_exempt) is not bool
            or type(self.certificate_geometry_legal) is not bool
        ):
            raise PlacementConflictReportError("model-split booleans are not exact")
        if (
            type(self.conflict_pair) is not tuple
            or len(self.conflict_pair) != 2
            or any(type(item) is not str or not item for item in self.conflict_pair)
        ):
            raise PlacementConflictReportError("conflict obligation pair is malformed")
        if (
            type(self.solver_owner_pair) is not tuple
            or type(self.post_compaction_member_pair) is not tuple
            or len(self.solver_owner_pair) != 2
            or len(self.post_compaction_member_pair) != 2
        ):
            raise PlacementConflictReportError("conflict actor pairs are malformed")
        for boxes, field in (
            (self.solver_owner_boxes_nm, "solver owner boxes"),
            (self.post_compaction_owner_boxes_nm, "post-compaction owner boxes"),
            (self.violating_atom_boxes_nm, "violating atom boxes"),
        ):
            if type(boxes) is not tuple or len(boxes) != 2:
                raise PlacementConflictReportError(f"{field} are malformed")
            for bbox in boxes:
                _bbox(list(bbox), field=field)
        _bbox(list(self.overlap_bbox_nm), field="conflict overlap bbox")
        expected_area = (
            (self.overlap_bbox_nm[2] - self.overlap_bbox_nm[0])
            * (self.overlap_bbox_nm[3] - self.overlap_bbox_nm[1])
        )
        if type(self.overlap_area_nm2) is not int or self.overlap_area_nm2 != expected_area:
            raise PlacementConflictReportError("conflict overlap area does not replay bbox")
        if (
            type(self.declaration_obligations) is not tuple
            or not self.declaration_obligations
            or any(
                type(row) is not tuple
                or len(row) != 2
                or any(type(item) is not str or not item for item in row)
                for row in self.declaration_obligations
            )
        ):
            raise PlacementConflictReportError("declaration context is malformed")
        if self.declaration_resolution != "none":
            raise PlacementConflictReportError(
                "a model split cannot authorize declaration resolution"
            )
        if self.suggested_action != "UNIFY_CANONICAL_GEOMETRY_PROJECTION":
            raise PlacementConflictReportError("model-split action is not canonical")

    def canonical_data(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema": _REPORT_SCHEMA,
            "status": self.status,
            "category": self.category,
            "bindings": {
                "frozen_inputs": [self.frozen_inputs_path, self.frozen_inputs_sha256],
                "source": [self.source_path, self.source_sha256],
                "runner": [self.runner_path, self.runner_sha256],
                "reference_gds": [self.reference_gds_path, self.reference_gds_sha256],
                "failure_receipt": [
                    self.failure_receipt_path,
                    self.failure_receipt_sha256,
                ],
                "replay_receipt": [
                    self.replay_receipt_path,
                    self.replay_receipt_sha256,
                ],
            },
            "failure": {
                "original_classification": self.original_classification,
                "normalized_classification": self.status,
                "error_type": self.error_type,
                "message_sha256": self.failure_message_sha256,
                "replay_message_sha256": self.replay_message_sha256,
                "replay_message_exact": self.replay_message_exact,
            },
            "conflict": {
                "obligation_pair": list(self.conflict_pair),
                "solver_owner_pair": list(self.solver_owner_pair),
                "post_compaction_member_pair": list(
                    self.post_compaction_member_pair
                ),
                "solver_owner_boxes_nm": [
                    list(bbox) for bbox in self.solver_owner_boxes_nm
                ],
                "post_compaction_owner_boxes_nm": [
                    list(bbox) for bbox in self.post_compaction_owner_boxes_nm
                ],
                "violating_atom_boxes_nm": [
                    list(bbox) for bbox in self.violating_atom_boxes_nm
                ],
                "intersection_bbox_nm": list(self.overlap_bbox_nm),
                "overlap_area_nm2": self.overlap_area_nm2,
            },
            "model_split": {
                "mechanical_split_point": self.mechanical_split_point,
                "solver_compatibility_exempt": self.solver_compatibility_exempt,
                "certificate_geometry_legal": self.certificate_geometry_legal,
                "solver_projection_sha256": self.solver_projection_sha256,
                "post_compaction_projection_sha256": (
                    self.post_compaction_projection_sha256
                ),
            },
            "declaration_context": [
                {"obligation_id": obligation_id, "source": source}
                for obligation_id, source in self.declaration_obligations
            ],
            "decision": {
                "declaration_resolution": self.declaration_resolution,
                "suggested_action": self.suggested_action,
                "explanation": self.explanation,
            },
        }

    @property
    def artifact_sha256(self) -> str:
        return hashlib.sha256(_canonical_bytes(self.canonical_data())).hexdigest()

    def document_data(self) -> dict[str, object]:
        return {**self.canonical_data(), "artifact_sha256": self.artifact_sha256}


def attribute_projection_split(capture: object) -> PlacementConflictReport:
    """Require a solver/checker contradiction and emit its named obligation pair."""

    root = _mapping(
        capture,
        field="projection split capture",
        keys={
            "schema",
            "bindings",
            "failure",
            "solver_projection",
            "post_compaction_projection",
            "split_diagnosis",
            "declaration_context",
        },
    )
    if root.get("schema") != _CAPTURE_SCHEMA:
        raise PlacementConflictReportError("projection split capture schema is unknown")
    bindings = _mapping(
        root.get("bindings"),
        field="projection split bindings",
        keys={
            "frozen_inputs",
            "source",
            "runner",
            "reference_gds",
            "failure_receipt",
            "replay_receipt",
        },
    )
    frozen_path, frozen_sha = _binding(
        bindings.get("frozen_inputs"), field="frozen inputs"
    )
    source_path, source_sha = _binding(bindings.get("source"), field="source")
    runner_path, runner_sha = _binding(bindings.get("runner"), field="runner")
    reference_path, reference_sha = _binding(
        bindings.get("reference_gds"), field="reference GDS"
    )
    receipt_path, receipt_sha = _binding(
        bindings.get("failure_receipt"), field="failure receipt"
    )
    replay_receipt_path, replay_receipt_sha = _binding(
        bindings.get("replay_receipt"), field="replay receipt"
    )
    failure = _mapping(
        root.get("failure"),
        field="projection split failure",
        keys={
            "classification",
            "error_type",
            "message_sha256",
            "replay_message_sha256",
            "replay_message_exact",
        },
    )
    original_classification = _text(
        failure.get("classification"), field="original failure classification"
    )
    error_type = _text(failure.get("error_type"), field="failure error type")
    if error_type != "DeclaredCompactionError":
        raise PlacementConflictReportError(
            "projection split attribution requires DeclaredCompactionError"
        )
    message_sha = _digest(
        failure.get("message_sha256"), field="failure message digest"
    )
    replay_message_sha = _digest(
        failure.get("replay_message_sha256"), field="replay failure message digest"
    )
    replay_message_exact = failure.get("replay_message_exact")
    if type(replay_message_exact) is not bool or replay_message_exact is not (
        replay_message_sha == message_sha
    ):
        raise PlacementConflictReportError("replay message identity does not replay")

    solver = _mapping(
        root.get("solver_projection"),
        field="solver projection",
        keys={
            "status",
            "grain",
            "owner_pair",
            "owner_members",
            "owner_boxes_nm",
            "collision_atoms_nm",
            "collision_obligation_present",
            "overlap_area_nm2",
            "projection_sha256",
        },
    )
    if solver.get("status") not in {"OPTIMAL", "FEASIBLE"}:
        raise PlacementConflictReportError("solver projection lacks a feasible witness")
    if solver.get("grain") != "master":
        raise PlacementConflictReportError("solver projection is not master-grain")
    solver_pair = _pair(solver.get("owner_pair"), field="solver owner pair")
    solver_owner_boxes = _owner_boxes(
        solver.get("owner_boxes_nm"), field="solver owner boxes"
    )
    solver_atoms = _collision_atoms(
        solver.get("collision_atoms_nm"),
        field="solver collision atoms",
        owners=solver_pair,
        owner_boxes=solver_owner_boxes,
    )
    if solver.get("collision_obligation_present") is not True:
        raise PlacementConflictReportError(
            "solver projection lacks the named collision obligation"
        )
    solver_overlaps = _positive_layered_overlaps(*solver_atoms)
    if solver.get("overlap_area_nm2") != 0 or solver_overlaps:
        raise PlacementConflictReportError(
            "solver projection did not accept a non-overlap witness"
        )
    solver_projection_sha = _projection_digest(solver, field="solver projection")
    raw_owner_members = solver.get("owner_members")
    if type(raw_owner_members) is not list or len(raw_owner_members) != 2:
        raise PlacementConflictReportError("solver owner membership is not exact")
    owner_members: dict[str, tuple[str, ...]] = {}
    for row in raw_owner_members:
        if (
            type(row) is not list
            or len(row) != 2
            or type(row[0]) is not str
            or not row[0]
            or type(row[1]) is not list
            or not row[1]
            or any(type(member) is not str or not member for member in row[1])
        ):
            raise PlacementConflictReportError("solver owner membership row is malformed")
        owner_members[row[0]] = tuple(row[1])
    if tuple(owner_members) != solver_pair:
        raise PlacementConflictReportError("solver owner membership does not bind its pair")

    post = _mapping(
        root.get("post_compaction_projection"),
        field="post-compaction projection",
        keys={
            "status",
            "grain",
            "owner_pair",
            "owner_boxes_nm",
            "collision_atoms_nm",
            "violating_atom_boxes_nm",
            "intersection_bbox_nm",
            "overlap_area_nm2",
            "projection_sha256",
        },
    )
    if post.get("status") != "REJECTED" or post.get("grain") != "source_member":
        raise PlacementConflictReportError(
            "post-compaction projection is not a rejected source-member witness"
        )
    member_pair = _pair(post.get("owner_pair"), field="post-compaction member pair")
    post_owner_boxes = _owner_boxes(
        post.get("owner_boxes_nm"), field="post-compaction owner boxes"
    )
    post_atoms = _collision_atoms(
        post.get("collision_atoms_nm"),
        field="post-compaction collision atoms",
        owners=member_pair,
        owner_boxes=post_owner_boxes,
    )
    violating_boxes = _owner_boxes(
        post.get("violating_atom_boxes_nm"),
        field="post-compaction violating atom boxes",
    )
    overlaps = _positive_layered_overlaps(*post_atoms)
    matching = tuple(row for row in overlaps if row[:2] == violating_boxes)
    if len(matching) != 1:
        raise PlacementConflictReportError(
            "post-compaction violating atoms do not replay collision authority"
        )
    _left_atom, _right_atom, intersection, overlap_area = matching[0]
    declared_intersection = _bbox(
        post.get("intersection_bbox_nm"), field="post-compaction intersection"
    )
    if intersection != declared_intersection:
        raise PlacementConflictReportError("post-compaction overlap bbox does not replay")
    if post.get("overlap_area_nm2") != overlap_area or overlap_area <= 0:
        raise PlacementConflictReportError("post-compaction overlap area does not replay")
    post_projection_sha = _projection_digest(
        post, field="post-compaction projection"
    )
    if (
        member_pair[0] not in owner_members[solver_pair[0]]
        or member_pair[1] not in owner_members[solver_pair[1]]
    ):
        raise PlacementConflictReportError(
            "post-compaction members are not owned by the solver pair"
        )
    if solver_projection_sha == post_projection_sha:
        raise PlacementConflictReportError(
            "distinct projection witnesses cannot share one digest"
        )

    split = _mapping(
        root.get("split_diagnosis"),
        field="split diagnosis",
        keys={
            "mechanical_split_point",
            "solver_compatibility_exempt",
            "certificate_geometry_legal",
            "actors",
        },
    )
    compatibility_exempt = split.get("solver_compatibility_exempt")
    geometry_legal = split.get("certificate_geometry_legal")
    if type(compatibility_exempt) is not bool or type(geometry_legal) is not bool:
        raise PlacementConflictReportError("split diagnosis booleans are not exact")
    raw_actors = split.get("actors")
    if type(raw_actors) is not list or len(raw_actors) != 2:
        raise PlacementConflictReportError("split diagnosis requires two actors")
    origin_mismatch = False
    atom_loss = False
    all_missing_are_component_covers = True
    for index, raw_actor in enumerate(raw_actors):
        actor = _mapping(
            raw_actor,
            field="split diagnosis actor",
            keys={
                "source_member",
                "solver_owner",
                "member_offset_nm",
                "source_owner_origin_nm",
                "solver_owner_origin_nm",
                "expected_source_origin_nm",
                "origin_transport_consistent",
                "source_local_atom_bbox_nm",
                "expected_solver_local_atom_bbox_nm",
                "solver_local_atom_present",
            },
        )
        if (
            actor.get("source_member") != member_pair[index]
            or actor.get("solver_owner") != solver_pair[index]
        ):
            raise PlacementConflictReportError("split actor ownership differs")
        offset = _point(actor.get("member_offset_nm"), field="member offset")
        source_origin = _point(
            actor.get("source_owner_origin_nm"), field="source owner origin"
        )
        solver_origin = _point(
            actor.get("solver_owner_origin_nm"), field="solver owner origin"
        )
        expected_origin = _point(
            actor.get("expected_source_origin_nm"), field="expected source origin"
        )
        if source_origin != post_owner_boxes[index][:2] or solver_origin != (
            solver_owner_boxes[index][:2]
        ):
            raise PlacementConflictReportError("split actor origins differ from projection")
        replayed_expected_origin = (
            solver_origin[0] + offset[0],
            solver_origin[1] + offset[1],
        )
        consistent = source_origin == replayed_expected_origin
        if (
            expected_origin != replayed_expected_origin
            or actor.get("origin_transport_consistent") is not consistent
        ):
            raise PlacementConflictReportError("split actor transport does not replay")
        origin_mismatch = origin_mismatch or not consistent
        source_local = _bbox(
            actor.get("source_local_atom_bbox_nm"), field="source local atom"
        )
        replayed_source_local = (
            violating_boxes[index][0] - source_origin[0],
            violating_boxes[index][1] - source_origin[1],
            violating_boxes[index][2] - source_origin[0],
            violating_boxes[index][3] - source_origin[1],
        )
        if source_local != replayed_source_local:
            raise PlacementConflictReportError("source local atom does not replay")
        expected_local = _bbox(
            actor.get("expected_solver_local_atom_bbox_nm"),
            field="expected solver local atom",
        )
        replayed_expected_local = (
            source_local[0] + offset[0],
            source_local[1] + offset[1],
            source_local[2] + offset[0],
            source_local[3] + offset[1],
        )
        if expected_local != replayed_expected_local:
            raise PlacementConflictReportError("expected solver atom does not replay")
        source_layers = tuple(
            layer for layer, bbox in post_atoms[index] if bbox == violating_boxes[index]
        )
        if len(source_layers) != 1:
            raise PlacementConflictReportError("violating source atom layer is ambiguous")
        present = any(
            layer == source_layers[0]
            and (
                bbox[0] - solver_origin[0],
                bbox[1] - solver_origin[1],
                bbox[2] - solver_origin[0],
                bbox[3] - solver_origin[1],
            )
            == expected_local
            for layer, bbox in solver_atoms[index]
        )
        if actor.get("solver_local_atom_present") is not present:
            raise PlacementConflictReportError("solver atom presence does not replay")
        atom_loss = atom_loss or not present
        if not present:
            expected_absolute = (
                solver_origin[0] + expected_local[0],
                solver_origin[1] + expected_local[1],
                solver_origin[0] + expected_local[2],
                solver_origin[1] + expected_local[3],
            )
            component_cover = tuple(
                bbox
                for layer, bbox in solver_atoms[index]
                if layer == source_layers[0]
                and expected_absolute[0] <= bbox[0] < bbox[2] <= expected_absolute[2]
                and expected_absolute[1] <= bbox[1] < bbox[3] <= expected_absolute[3]
            )
            cover_envelope = (
                (
                    min(bbox[0] for bbox in component_cover),
                    min(bbox[1] for bbox in component_cover),
                    max(bbox[2] for bbox in component_cover),
                    max(bbox[3] for bbox in component_cover),
                )
                if component_cover
                else None
            )
            all_missing_are_component_covers = (
                all_missing_are_component_covers
                and len(component_cover) >= 2
                and cover_envelope == expected_absolute
            )
    mechanical_split_point = _text(
        split.get("mechanical_split_point"), field="mechanical split point"
    )
    expected_split_point = (
        "COMPOSITE_MEMBER_TRANSPORT_MISMATCH"
        if origin_mismatch
        else "SOURCE_LAYER_ENVELOPE_COARSENING"
        if atom_loss and all_missing_are_component_covers
        else "MASTER_LAYERED_ATOM_LOSS"
        if atom_loss
        else "SOLVER_COMPATIBILITY_SCOPE_MISMATCH"
        if compatibility_exempt
        else "SOLVER_CERTIFICATE_GEOMETRY_ILLEGAL"
        if not geometry_legal
        else "COLLISION_REPLAY_DIVERGENCE"
    )
    if mechanical_split_point != expected_split_point:
        raise PlacementConflictReportError("mechanical split point does not replay")

    context = _mapping(
        root.get("declaration_context"),
        field="declaration context",
        keys={"alignment", "adjacency"},
    )
    raw_alignment = _mapping(
        context.get("alignment"),
        field="alignment declaration",
        keys={"obligation_id", "source"},
    )
    raw_adjacency = context.get("adjacency")
    if type(raw_adjacency) is not list or not raw_adjacency:
        raise PlacementConflictReportError("adjacency declaration context is empty")
    declarations = [
        (
            _text(raw_alignment.get("obligation_id"), field="alignment obligation id"),
            _text(raw_alignment.get("source"), field="alignment source"),
        )
    ]
    for raw in raw_adjacency:
        row = _mapping(
            raw,
            field="adjacency declaration",
            keys={"obligation_id", "source"},
        )
        declarations.append((
            _text(row.get("obligation_id"), field="adjacency obligation id"),
            _text(row.get("source"), field="adjacency source"),
        ))

    return PlacementConflictReport(
        frozen_inputs_path=frozen_path,
        frozen_inputs_sha256=frozen_sha,
        source_path=source_path,
        source_sha256=source_sha,
        runner_path=runner_path,
        runner_sha256=runner_sha,
        reference_gds_path=reference_path,
        reference_gds_sha256=reference_sha,
        failure_receipt_path=receipt_path,
        failure_receipt_sha256=receipt_sha,
        replay_receipt_path=replay_receipt_path,
        replay_receipt_sha256=replay_receipt_sha,
        original_classification=original_classification,
        error_type=error_type,
        failure_message_sha256=message_sha,
        replay_message_sha256=replay_message_sha,
        replay_message_exact=replay_message_exact,
        status="ATTRIBUTED_CONFLICT",
        category="MODEL_SPLIT",
        conflict_pair=(
            f"layered_collision:{solver_pair[0]}:{solver_pair[1]}",
            f"source_member_nonoverlap:{member_pair[0]}:{member_pair[1]}",
        ),
        solver_owner_pair=solver_pair,
        post_compaction_member_pair=member_pair,
        solver_owner_boxes_nm=solver_owner_boxes,
        post_compaction_owner_boxes_nm=post_owner_boxes,
        violating_atom_boxes_nm=violating_boxes,
        overlap_bbox_nm=intersection,
        overlap_area_nm2=overlap_area,
        solver_projection_sha256=solver_projection_sha,
        post_compaction_projection_sha256=post_projection_sha,
        mechanical_split_point=mechanical_split_point,
        solver_compatibility_exempt=compatibility_exempt,
        certificate_geometry_legal=geometry_legal,
        declaration_obligations=tuple(declarations),
        declaration_resolution="none",
        suggested_action="UNIFY_CANONICAL_GEOMETRY_PROJECTION",
        explanation=(
            "The solver accepted the master-grain collision obligation, while "
            "post-compaction rejected a positive-area overlap between members owned "
            "by those masters. The consumers did not evaluate the same geometry "
            "projection, so no declaration-layer resolution is justified until one "
            "canonical projection is shared."
        ),
    )


def render_placement_conflict_report(capture: object) -> str:
    report = attribute_projection_split(capture)
    return json.dumps(
        report.document_data(),
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise PlacementConflictReportError(f"duplicate conflict-report key {key!r}")
        result[key] = value
    return result


def verify_placement_conflict_report_document(
    capture: object, path: Path
) -> PlacementConflictReport:
    candidate = Path(path)
    try:
        raw = candidate.read_bytes()
        payload = json.loads(
            raw.decode("ascii"), object_pairs_hook=_reject_duplicate_keys
        )
    except PlacementConflictReportError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PlacementConflictReportError(
            "conflict report document cannot be replayed"
        ) from exc
    report = attribute_projection_split(capture)
    if payload != report.document_data():
        raise PlacementConflictReportError(
            "conflict report document does not replay projection evidence"
        )
    if raw != render_placement_conflict_report(capture).encode("ascii"):
        raise PlacementConflictReportError("conflict report bytes are not canonical")
    return report


__all__ = [
    "PlacementConflictReport",
    "PlacementConflictReportError",
    "attribute_projection_split",
    "render_placement_conflict_report",
    "verify_placement_conflict_report_document",
]
