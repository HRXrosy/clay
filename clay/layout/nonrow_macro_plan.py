"""Canonical pre-placement plans for rigid aggregate non-MOS macros."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from typing import Mapping
import weakref

from clay.layout.physical_macro_contracts import (
    PhysicalMacroRouteDemand,
    PhysicalMacroTerminalAccess,
    _artwork_sha256,
    _canonical_ring,
)


_HEX = frozenset("0123456789abcdef")
_MAX_ITEMS = 4096
_MAX_COORDINATE_NM = 10**12
_KINDS = frozenset({"passive_bank", "bjt_cluster"})
_ORTHOGONAL_TRANSFORMS = frozenset({
    "R0", "R90", "R180", "R270",
    "MR0", "MR90", "MR180", "MR270",
})


class RigidNonmosMacroPlanError(ValueError):
    """A declared aggregate non-MOS macro cannot be planned exactly."""


class RigidNonmosMacroProjectionError(RuntimeError):
    """A placed aggregate no longer replays its rigid local member plan."""

    def __init__(self, code: str) -> None:
        self.code = str(code)
        super().__init__(self.code)


def _text(value: object, field: str) -> str:
    if type(value) is not str or not value or len(value) > 4096:
        raise RigidNonmosMacroPlanError(f"{field} must be one bounded exact string")
    return value


def _digest(value: object, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(char not in _HEX for char in value)
    ):
        raise RigidNonmosMacroPlanError(f"{field} must be one canonical sha256")
    return value


def _bounded_tuple(value: object, field: str, *, allow_empty: bool = True) -> tuple:
    if type(value) is not tuple:
        raise RigidNonmosMacroPlanError(f"{field} must be an exact tuple")
    if len(value) > _MAX_ITEMS:
        raise RigidNonmosMacroPlanError(f"{field} exceeds the physical item cap")
    if not allow_empty and not value:
        raise RigidNonmosMacroPlanError(f"{field} must not be empty")
    return value


def _coordinate(value: object, field: str) -> int:
    if (
        type(value) is not int
        or value < -_MAX_COORDINATE_NM
        or value > _MAX_COORDINATE_NM
    ):
        raise RigidNonmosMacroPlanError(
            f"{field} must be one bounded exact integer"
        )
    return value


def _bbox(value: object, field: str) -> tuple[int, int, int, int]:
    if type(value) is not tuple or len(value) != 4:
        raise RigidNonmosMacroPlanError(f"{field} must be one exact bbox tuple")
    result = tuple(_coordinate(item, field) for item in value)
    if result[2] <= result[0] or result[3] <= result[1]:
        raise RigidNonmosMacroPlanError(f"{field} must have positive area")
    return result  # type: ignore[return-value]


def _positive_overlap(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> bool:
    return (
        min(left[2], right[2]) > max(left[0], right[0])
        and min(left[3], right[3]) > max(left[1], right[1])
    )


def _sha(payload: object) -> str:
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True, kw_only=True, order=True)
class RigidNonmosMacroMemberPlan:
    member: str
    local_bbox_nm: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "member", _text(self.member, "macro member"))
        object.__setattr__(self, "local_bbox_nm", _bbox(
            self.local_bbox_nm, "member local bbox"
        ))

    @property
    def width_nm(self) -> int:
        return self.local_bbox_nm[2] - self.local_bbox_nm[0]

    @property
    def height_nm(self) -> int:
        return self.local_bbox_nm[3] - self.local_bbox_nm[1]

    def payload(self) -> tuple[object, ...]:
        return (self.member, self.local_bbox_nm)


@dataclass(frozen=True, slots=True, kw_only=True, order=True)
class RigidNonmosLocalRelationProjection:
    """Replayable proof that one authored relation became local geometry."""

    macro_id: str
    declaration_ref: str
    members: tuple[str, str]
    relation: str
    gap_nm: int
    pre_local_bboxes_nm: tuple[tuple[str, tuple[int, int, int, int]], ...]
    post_local_bboxes_nm: tuple[tuple[str, tuple[int, int, int, int]], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "macro_id", _text(self.macro_id, "relation macro id"))
        object.__setattr__(
            self,
            "declaration_ref",
            _text(self.declaration_ref, "relation declaration reference"),
        )
        if (
            type(self.members) is not tuple
            or len(self.members) != 2
            or self.members[0] == self.members[1]
            or any(type(item) is not str or not item for item in self.members)
        ):
            raise RigidNonmosMacroPlanError(
                "relation projection requires two directed members"
            )
        if self.relation not in {"left_of", "below"}:
            raise RigidNonmosMacroPlanError(
                "relation projection must be normalized"
            )
        if type(self.gap_nm) is not int or self.gap_nm <= 0:
            raise RigidNonmosMacroPlanError(
                "relation projection gap must be a positive exact integer"
            )
        rebuilt = {}
        for field in ("pre_local_bboxes_nm", "post_local_bboxes_nm"):
            raw = _bounded_tuple(getattr(self, field), field, allow_empty=False)
            rows = []
            for item in raw:
                if type(item) is not tuple or len(item) != 2:
                    raise RigidNonmosMacroPlanError(
                        "relation projection bboxes require exact pairs"
                    )
                rows.append((
                    _text(item[0], "relation projection member"),
                    _bbox(item[1], field),
                ))
            if raw != tuple(sorted(rows)) or {row[0] for row in rows} != set(
                self.members
            ):
                raise RigidNonmosMacroPlanError(
                    "relation projection bboxes must canonically cover its members"
                )
            rebuilt[field] = dict(rows)
        pre = rebuilt["pre_local_bboxes_nm"]
        post = rebuilt["post_local_bboxes_nm"]
        for member in self.members:
            if pre[member][0:2] != (0, 0) or (
                pre[member][2] - pre[member][0],
                pre[member][3] - pre[member][1],
            ) != (
                post[member][2] - post[member][0],
                post[member][3] - post[member][1],
            ):
                raise RigidNonmosMacroPlanError(
                    "relation projection changed member geometry"
                )
        left, right = (post[member] for member in self.members)
        satisfied = (
            left[2] + self.gap_nm <= right[0]
            if self.relation == "left_of"
            else left[3] + self.gap_nm <= right[1]
        )
        if not satisfied:
            raise RigidNonmosMacroPlanError(
                "relation projection does not satisfy its local geometry"
            )

    def payload(self) -> dict[str, object]:
        return {
            "schema": "clay.rigid-nonmos-local-relation-projection/v1",
            "macro_id": self.macro_id,
            "declaration_ref": self.declaration_ref,
            "members": self.members,
            "relation": self.relation,
            "gap_nm": self.gap_nm,
            "pre_local_bboxes_nm": self.pre_local_bboxes_nm,
            "post_local_bboxes_nm": self.post_local_bboxes_nm,
        }


@dataclass(frozen=True, slots=True, kw_only=True, order=True)
class RigidNonmosLocalAlignmentProjection:
    """Replayable proof that one authored edge equality became local geometry."""

    macro_id: str
    declaration_ref: str
    members: tuple[str, str]
    edge: str
    pre_local_bboxes_nm: tuple[tuple[str, tuple[int, int, int, int]], ...]
    post_local_bboxes_nm: tuple[tuple[str, tuple[int, int, int, int]], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "macro_id", _text(self.macro_id, "alignment macro id"))
        object.__setattr__(
            self,
            "declaration_ref",
            _text(self.declaration_ref, "alignment declaration reference"),
        )
        if (
            type(self.members) is not tuple
            or len(self.members) != 2
            or self.members != tuple(sorted(set(self.members)))
            or any(type(item) is not str or not item for item in self.members)
        ):
            raise RigidNonmosMacroPlanError(
                "alignment projection requires one canonical member pair"
            )
        if self.edge not in {"left", "right", "bottom", "top"}:
            raise RigidNonmosMacroPlanError(
                "alignment projection edge is unsupported"
            )
        rebuilt = {}
        for field in ("pre_local_bboxes_nm", "post_local_bboxes_nm"):
            raw = _bounded_tuple(getattr(self, field), field, allow_empty=False)
            rows = []
            for item in raw:
                if type(item) is not tuple or len(item) != 2:
                    raise RigidNonmosMacroPlanError(
                        "alignment projection bboxes require exact pairs"
                    )
                rows.append((
                    _text(item[0], "alignment projection member"),
                    _bbox(item[1], field),
                ))
            if raw != tuple(sorted(rows)) or {row[0] for row in rows} != set(
                self.members
            ):
                raise RigidNonmosMacroPlanError(
                    "alignment projection bboxes must canonically cover its members"
                )
            rebuilt[field] = dict(rows)
        pre = rebuilt["pre_local_bboxes_nm"]
        post = rebuilt["post_local_bboxes_nm"]
        for member in self.members:
            if (
                pre[member][2] - pre[member][0],
                pre[member][3] - pre[member][1],
            ) != (
                post[member][2] - post[member][0],
                post[member][3] - post[member][1],
            ):
                raise RigidNonmosMacroPlanError(
                    "alignment projection changed member geometry"
                )
        coordinate = {"left": 0, "bottom": 1, "right": 2, "top": 3}[self.edge]
        first, second = (post[member] for member in self.members)
        if first[coordinate] != second[coordinate]:
            raise RigidNonmosMacroPlanError(
                "alignment projection does not satisfy its local geometry"
            )

    def payload(self) -> dict[str, object]:
        return {
            "schema": "clay.rigid-nonmos-local-alignment-projection/v1",
            "macro_id": self.macro_id,
            "declaration_ref": self.declaration_ref,
            "members": self.members,
            "edge": self.edge,
            "pre_local_bboxes_nm": self.pre_local_bboxes_nm,
            "post_local_bboxes_nm": self.post_local_bboxes_nm,
        }


@dataclass(frozen=True, slots=True, kw_only=True, order=True)
class RigidNonmosMacroPlan:
    macro_id: str
    kind: str
    members: tuple[str, ...]
    member_plans: tuple[RigidNonmosMacroMemberPlan, ...]
    local_bbox_nm: tuple[int, int, int, int]
    matching_pairs: tuple[tuple[str, str], ...]
    local_relation_projections: tuple[RigidNonmosLocalRelationProjection, ...] = ()
    local_alignment_projections: tuple[RigidNonmosLocalAlignmentProjection, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "macro_id", _text(self.macro_id, "macro id"))
        if self.kind not in _KINDS:
            raise RigidNonmosMacroPlanError(
                f"unsupported aggregate non-MOS macro kind {self.kind!r}"
            )
        members = _bounded_tuple(self.members, "macro members", allow_empty=False)
        if (
            members != tuple(sorted(set(members)))
            or any(type(item) is not str or not item for item in members)
        ):
            raise RigidNonmosMacroPlanError("macro members must be canonical")
        plans = _bounded_tuple(
            self.member_plans, "macro member plans", allow_empty=False
        )
        if any(type(item) is not RigidNonmosMacroMemberPlan for item in plans):
            raise RigidNonmosMacroPlanError(
                "macro member plans must contain exact typed values"
            )
        if plans != tuple(sorted(plans)) or tuple(item.member for item in plans) != members:
            raise RigidNonmosMacroPlanError(
                "macro member plans must cover members in canonical order"
            )
        local_bbox = _bbox(self.local_bbox_nm, "macro local bbox")
        if local_bbox[0:2] != (0, 0):
            raise RigidNonmosMacroPlanError(
                "macro local bbox must start at the origin"
            )
        for item in plans:
            x0, y0, x1, y1 = item.local_bbox_nm
            if x0 < 0 or y0 < 0 or x1 > local_bbox[2] or y1 > local_bbox[3]:
                raise RigidNonmosMacroPlanError(
                    "member local bbox escapes its rigid macro"
                )
        for index, left in enumerate(plans):
            for right in plans[index + 1:]:
                if _positive_overlap(left.local_bbox_nm, right.local_bbox_nm):
                    raise RigidNonmosMacroPlanError(
                        "rigid macro member bboxes overlap"
                    )
        pairs = _bounded_tuple(self.matching_pairs, "macro matching pairs")
        rebuilt = []
        for pair in pairs:
            if (
                type(pair) is not tuple
                or len(pair) != 2
                or any(type(item) is not str or not item for item in pair)
                or pair[0] >= pair[1]
                or not set(pair) <= set(members)
            ):
                raise RigidNonmosMacroPlanError(
                    "macro matching pairs must be canonical owned pairs"
                )
            rebuilt.append(pair)
        if pairs != tuple(sorted(set(rebuilt))):
            raise RigidNonmosMacroPlanError("macro matching pairs must be canonical")
        pair_members = [member for pair in pairs for member in pair]
        if len(pair_members) != len(set(pair_members)):
            raise RigidNonmosMacroPlanError("macro matching pairs overlap")
        projections = _bounded_tuple(
            self.local_relation_projections, "macro local relation projections"
        )
        if any(
            type(item) is not RigidNonmosLocalRelationProjection
            for item in projections
        ):
            raise RigidNonmosMacroPlanError(
                "macro local relation projections require exact typed values"
            )
        if projections != tuple(sorted(projections)) or len(set(projections)) != len(
            projections
        ):
            raise RigidNonmosMacroPlanError(
                "macro local relation projections must be canonical"
            )
        plan_bboxes = {item.member: item.local_bbox_nm for item in plans}
        if any(
            item.macro_id != self.macro_id
            or not set(item.members) <= set(members)
            or dict(item.post_local_bboxes_nm) != {
                member: plan_bboxes[member] for member in item.members
            }
            for item in projections
        ):
            raise RigidNonmosMacroPlanError(
                "macro local relation projection differs from member plans"
            )
        alignments = _bounded_tuple(
            self.local_alignment_projections,
            "macro local alignment projections",
        )
        if any(
            type(item) is not RigidNonmosLocalAlignmentProjection
            for item in alignments
        ):
            raise RigidNonmosMacroPlanError(
                "macro local alignment projections require exact typed values"
            )
        if alignments != tuple(sorted(alignments)) or len(set(alignments)) != len(
            alignments
        ):
            raise RigidNonmosMacroPlanError(
                "macro local alignment projections must be canonical"
            )
        if any(
            item.macro_id != self.macro_id
            or not set(item.members) <= set(members)
            or dict(item.post_local_bboxes_nm) != {
                member: plan_bboxes[member] for member in item.members
            }
            for item in alignments
        ):
            raise RigidNonmosMacroPlanError(
                "macro local alignment projection differs from member plans"
            )

    def payload(self) -> dict[str, object]:
        payload = {
            "schema": (
                "clay.rigid-nonmos-macro-plan/v3"
                if self.local_alignment_projections
                else "clay.rigid-nonmos-macro-plan/v2"
            ),
            "macro_id": self.macro_id,
            "kind": self.kind,
            "members": self.members,
            "member_plans": tuple(item.payload() for item in self.member_plans),
            "local_bbox_nm": self.local_bbox_nm,
            "matching_pairs": self.matching_pairs,
            "local_relation_projections": tuple(
                item.payload() for item in self.local_relation_projections
            ),
        }
        if self.local_alignment_projections:
            payload["local_alignment_projections"] = tuple(
                item.payload() for item in self.local_alignment_projections
            )
        return payload

    @property
    def sha256(self) -> str:
        return _sha(self.payload())

    def translation_nm(self, devices: Mapping[str, object]) -> tuple[int, int]:
        translations = set()
        for item in self.member_plans:
            view = devices.get(item.member)
            if view is None:
                raise RigidNonmosMacroProjectionError(
                    f"RIGID_NONMOS_MEMBER_MISSING:{item.member}"
                )
            try:
                bbox = _bbox(getattr(view, "bbox", None), "placed member bbox")
            except RigidNonmosMacroPlanError as exc:
                raise RigidNonmosMacroProjectionError(
                    f"RIGID_NONMOS_MEMBER_VIEW_INVALID:{item.member}"
                ) from exc
            local = item.local_bbox_nm
            if (bbox[2] - bbox[0], bbox[3] - bbox[1]) != (
                local[2] - local[0], local[3] - local[1]
            ):
                raise RigidNonmosMacroProjectionError(
                    f"RIGID_NONMOS_MEMBER_FOOTPRINT_MISMATCH:{item.member}"
                )
            translations.add((bbox[0] - local[0], bbox[1] - local[1]))
        if len(translations) != 1:
            raise RigidNonmosMacroProjectionError(
                f"RIGID_NONMOS_MEMBER_OFFSET_MISMATCH:{self.macro_id}"
            )
        return next(iter(translations))

    def placed_bbox(self, devices: Mapping[str, object]) -> tuple[int, int, int, int]:
        dx, dy = self.translation_nm(devices)
        x0, y0, x1, y1 = self.local_bbox_nm
        return (x0 + dx, y0 + dy, x1 + dx, y1 + dy)


@dataclass(frozen=True, slots=True, kw_only=True)
class RigidNonmosMacroPlanSet:
    graph_sha256: str
    tech_id: str
    plans: tuple[RigidNonmosMacroPlan, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "graph_sha256", _digest(
            self.graph_sha256, "plan graph_sha256"
        ))
        object.__setattr__(self, "tech_id", _text(self.tech_id, "plan tech_id"))
        plans = _bounded_tuple(self.plans, "rigid non-MOS plans")
        from clay.layout.source_bound_nonmos_plan import SourceBoundNonmosMacroPlan
        if any(type(item) not in (RigidNonmosMacroPlan, SourceBoundNonmosMacroPlan) for item in plans):
            raise RigidNonmosMacroPlanError(
                "plan set requires exact rigid non-MOS plans"
            )
        if plans != tuple(sorted(plans, key=lambda p: p.macro_id)) or len({item.macro_id for item in plans}) != len(plans):
            raise RigidNonmosMacroPlanError("rigid non-MOS plans must be canonical")
        consumed = [member for plan in plans for member in plan.members]
        if len(consumed) != len(set(consumed)):
            raise RigidNonmosMacroPlanError(
                "one non-MOS member may belong to only one rigid aggregate"
            )

    def payload(self) -> dict[str, object]:
        return {
            "schema": "clay.rigid-nonmos-macro-plan-set/v1",
            "graph_sha256": self.graph_sha256,
            "tech_id": self.tech_id,
            "plans": tuple(item.payload() for item in self.plans),
        }

    @property
    def sha256(self) -> str:
        return _sha(self.payload())

    def placed_bbox(
        self, devices: Mapping[str, object], macro_id: str
    ) -> tuple[int, int, int, int]:
        selected = [item for item in self.plans if item.macro_id == macro_id]
        if len(selected) != 1:
            raise RigidNonmosMacroProjectionError(
                f"RIGID_NONMOS_MACRO_UNKNOWN:{macro_id}"
            )
        return selected[0].placed_bbox(devices)


@dataclass(frozen=True, slots=True, kw_only=True, order=True)
class RigidNonmosMacroPlacementBinding:
    macro_id: str
    plan_sha256: str
    translation_nm: tuple[int, int]
    bbox_nm: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "macro_id", _text(self.macro_id, "binding macro id"))
        object.__setattr__(self, "plan_sha256", _digest(
            self.plan_sha256, "binding plan_sha256"
        ))
        if type(self.translation_nm) is not tuple or len(self.translation_nm) != 2:
            raise RigidNonmosMacroPlanError(
                "binding translation_nm must be one exact tuple"
            )
        object.__setattr__(self, "translation_nm", tuple(
            _coordinate(value, "binding translation_nm")
            for value in self.translation_nm
        ))
        object.__setattr__(self, "bbox_nm", _bbox(
            self.bbox_nm, "binding bbox_nm"
        ))

    def payload(self) -> tuple[object, ...]:
        return (
            self.macro_id,
            self.plan_sha256,
            self.translation_nm,
            self.bbox_nm,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class RigidNonmosPlacementCertificate:
    graph_sha256: str
    plans_sha256: str
    interfaces_sha256: str
    bindings: tuple[RigidNonmosMacroPlacementBinding, ...]
    device_bboxes: tuple[tuple[str, tuple[int, int, int, int]], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "graph_sha256", _digest(
            self.graph_sha256, "certificate graph_sha256"
        ))
        object.__setattr__(self, "plans_sha256", _digest(
            self.plans_sha256, "certificate plans_sha256"
        ))
        object.__setattr__(self, "interfaces_sha256", _digest(
            self.interfaces_sha256, "certificate interfaces_sha256"
        ))
        bindings = _bounded_tuple(self.bindings, "certificate bindings")
        if any(type(item) is not RigidNonmosMacroPlacementBinding for item in bindings):
            raise RigidNonmosMacroPlanError(
                "certificate bindings must contain exact typed values"
            )
        if bindings != tuple(sorted(bindings)):
            raise RigidNonmosMacroPlanError(
                "certificate bindings must use canonical order"
            )
        if len({item.macro_id for item in bindings}) != len(bindings):
            raise RigidNonmosMacroPlanError(
                "certificate macro bindings must be unique"
            )
        raw_bboxes = _bounded_tuple(
            self.device_bboxes, "certificate device bboxes"
        )
        rebuilt = []
        for item in raw_bboxes:
            if type(item) is not tuple or len(item) != 2:
                raise RigidNonmosMacroPlanError(
                    "certificate device bbox entries must be exact tuples"
                )
            name, bbox = item
            rebuilt.append((_text(name, "certificate device"), _bbox(
                bbox, "certificate device bbox"
            )))
        if raw_bboxes != tuple(sorted(rebuilt)):
            raise RigidNonmosMacroPlanError(
                "certificate device bboxes must use canonical order"
            )
        if len({name for name, _bbox_nm in rebuilt}) != len(rebuilt):
            raise RigidNonmosMacroPlanError(
                "certificate device bboxes must be unique"
            )

    @classmethod
    def build(
        cls,
        plans: RigidNonmosMacroPlanSet,
        interfaces: "RigidNonmosMacroInterfaceSet",
        devices: Mapping[str, object],
    ) -> "RigidNonmosPlacementCertificate":
        if type(plans) is not RigidNonmosMacroPlanSet:
            raise RigidNonmosMacroPlanError(
                "certificate requires one exact plan set"
            )
        if type(interfaces) is not RigidNonmosMacroInterfaceSet:
            raise RigidNonmosMacroPlanError(
                "certificate requires one exact interface set"
            )
        if (
            interfaces.graph_sha256 != plans.graph_sha256
            or interfaces.plans_sha256 != plans.sha256
        ):
            raise RigidNonmosMacroPlanError(
                "certificate interface set differs from its rigid plans"
            )
        if not isinstance(devices, Mapping):
            raise RigidNonmosMacroPlanError(
                "certificate devices must be one mapping"
            )
        device_bboxes = []
        for name, view in devices.items():
            exact_name = _text(name, "placed device name")
            try:
                bbox = _bbox(getattr(view, "bbox", None), "placed device bbox")
            except RigidNonmosMacroPlanError as exc:
                raise RigidNonmosMacroProjectionError(
                    f"RIGID_NONMOS_MEMBER_VIEW_INVALID:{exact_name}"
                ) from exc
            device_bboxes.append((exact_name, bbox))
        bindings = []
        for plan in plans.plans:
            translation = plan.translation_nm(devices)
            bindings.append(RigidNonmosMacroPlacementBinding(
                macro_id=plan.macro_id,
                plan_sha256=plan.sha256,
                translation_nm=translation,
                bbox_nm=plan.placed_bbox(devices),
            ))
        return cls(
            graph_sha256=plans.graph_sha256,
            plans_sha256=plans.sha256,
            interfaces_sha256=interfaces.sha256,
            bindings=tuple(sorted(bindings)),
            device_bboxes=tuple(sorted(device_bboxes)),
        )

    def payload(self) -> dict[str, object]:
        return {
            "schema": "clay.rigid-nonmos-placement-certificate/v2",
            "graph_sha256": self.graph_sha256,
            "plans_sha256": self.plans_sha256,
            "interfaces_sha256": self.interfaces_sha256,
            "bindings": tuple(item.payload() for item in self.bindings),
            "device_bboxes": self.device_bboxes,
        }

    @property
    def sha256(self) -> str:
        return _sha(self.payload())


@dataclass(frozen=True, slots=True, kw_only=True, order=True)
class RigidNonmosMemberSize:
    member: str
    width_nm: int
    height_nm: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "member", _text(self.member, "size member"))
        for field in ("width_nm", "height_nm"):
            value = getattr(self, field)
            if type(value) is not int or not 0 < value <= _MAX_COORDINATE_NM:
                raise RigidNonmosMacroPlanError(
                    f"{field} must be one bounded positive exact integer"
                )

    def payload(self) -> tuple[object, ...]:
        return (self.member, self.width_nm, self.height_nm)


@dataclass(frozen=True, slots=True, kw_only=True, order=True)
class RigidNonmosMemberRelation:
    """One authored directional relation consumed inside a rigid macro."""

    left: str
    right: str
    relation: str
    declaration_ref: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "left", _text(self.left, "relation left member"))
        object.__setattr__(self, "right", _text(self.right, "relation right member"))
        if self.left == self.right:
            raise RigidNonmosMacroPlanError(
                "rigid member relation endpoints must be distinct"
            )
        if self.relation not in {"left_of", "below"}:
            raise RigidNonmosMacroPlanError(
                "rigid member relation must be normalized to left_of or below"
            )
        object.__setattr__(
            self,
            "declaration_ref",
            _text(self.declaration_ref, "relation declaration reference"),
        )

    def payload(self) -> tuple[object, ...]:
        return (self.left, self.right, self.relation, self.declaration_ref)


@dataclass(frozen=True, slots=True, kw_only=True, order=True)
class RigidNonmosMemberAlignment:
    """One authored edge equality consumed inside a rigid macro."""

    first: str
    second: str
    edge: str
    declaration_ref: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "first", _text(self.first, "alignment first member"))
        object.__setattr__(
            self, "second", _text(self.second, "alignment second member")
        )
        if self.first >= self.second:
            raise RigidNonmosMacroPlanError(
                "rigid member alignment endpoints must be canonical and distinct"
            )
        if self.edge not in {"left", "right", "bottom", "top"}:
            raise RigidNonmosMacroPlanError(
                "rigid member alignment edge is unsupported"
            )
        object.__setattr__(
            self,
            "declaration_ref",
            _text(self.declaration_ref, "alignment declaration reference"),
        )

    def payload(self) -> tuple[object, ...]:
        return (self.first, self.second, self.edge, self.declaration_ref)


@dataclass(frozen=True, slots=True, kw_only=True, order=True)
class RigidNonmosMemberAdjacency:
    """One source-owned maximum edge gap inside an indivisible macro."""

    first: str
    second: str
    maximum_nm: int
    declaration_ref: str

    def __post_init__(self) -> None:
        _text(self.first, "adjacency first member")
        _text(self.second, "adjacency second member")
        _text(self.declaration_ref, "adjacency declaration reference")
        if self.first >= self.second:
            raise RigidNonmosMacroPlanError("internal adjacency members must be canonical and distinct")
        if type(self.maximum_nm) is not int or not 0 < self.maximum_nm <= _MAX_COORDINATE_NM:
            raise RigidNonmosMacroPlanError("internal adjacency maximum must be a bounded positive exact integer")

    def payload(self) -> tuple[object, ...]:
        return (self.first, self.second, self.maximum_nm, self.declaration_ref)


def _require_internal_adjacencies(boxes, declarations):
    # Use the same signed edge-gap metric as final exact-GDS acceptance.
    # Source-bound PCells can share artwork, so logical ownership envelopes
    # may overlap. Independent tiles are still checked for native collisions.
    from clay.layout.completeness.adjacent_check import _edge_gap_nm

    for item in declarations:
        left, right = boxes[item.first], boxes[item.second]
        gap = _edge_gap_nm(left, right)
        if gap is None or gap > item.maximum_nm:
            raise RigidNonmosMacroPlanError(
                "rigid macro cannot realize internal adjacency: " + item.declaration_ref)


@dataclass(frozen=True, slots=True, kw_only=True, order=True)
class RigidNonmosMemberInterfaceSnapshot:
    """Exact member-local PDK interface captured before placement."""

    member: str
    local_bbox_nm: tuple[int, int, int, int]
    terminal_accesses: tuple[PhysicalMacroTerminalAccess, ...]
    body_tie_domain: tuple[str, ...]
    legal_orientations: tuple[str, ...]
    artwork_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "member", _text(self.member, "interface member"))
        local_bbox = _bbox(self.local_bbox_nm, "interface member local bbox")
        if local_bbox[0:2] != (0, 0):
            raise RigidNonmosMacroPlanError(
                "interface member local bbox must start at the origin"
            )
        object.__setattr__(self, "local_bbox_nm", local_bbox)
        terminals = _bounded_tuple(
            self.terminal_accesses, "interface member terminals", allow_empty=False
        )
        if any(type(item) is not PhysicalMacroTerminalAccess for item in terminals):
            raise RigidNonmosMacroPlanError(
                "interface member terminals must contain exact typed values"
            )
        if terminals != tuple(sorted(terminals)) or any(
            item.member != self.member for item in terminals
        ):
            raise RigidNonmosMacroPlanError(
                "interface member terminals must be canonical and owned"
            )
        body = _bounded_tuple(self.body_tie_domain, "interface body tie domain")
        if body != tuple(sorted(set(body))) or any(
            type(item) is not str or not item for item in body
        ):
            raise RigidNonmosMacroPlanError(
                "interface body tie domain must be canonical"
            )
        orientations = _bounded_tuple(
            self.legal_orientations,
            "interface legal orientations",
            allow_empty=False,
        )
        if orientations != tuple(sorted(set(orientations))) or any(
            item not in _ORTHOGONAL_TRANSFORMS
            for item in orientations
        ):
            raise RigidNonmosMacroPlanError(
                "interface legal orientations are invalid"
            )
        object.__setattr__(self, "artwork_sha256", _digest(
            self.artwork_sha256, "interface artwork_sha256"
        ))

    def payload(self) -> dict[str, object]:
        return {
            "member": self.member,
            "local_bbox_nm": self.local_bbox_nm,
            "terminal_accesses": tuple(
                item.payload() for item in self.terminal_accesses
            ),
            "body_tie_domain": self.body_tie_domain,
            "legal_orientations": self.legal_orientations,
            "artwork_sha256": self.artwork_sha256,
        }


@dataclass(frozen=True, slots=True, kw_only=True, order=True)
class RigidNonmosLocalSymmetryAxis:
    members: tuple[str, str]
    axis: str
    coordinate_2nm: int

    def __post_init__(self) -> None:
        if (
            type(self.members) is not tuple
            or len(self.members) != 2
            or self.members != tuple(sorted(set(self.members)))
            or any(type(item) is not str or not item for item in self.members)
        ):
            raise RigidNonmosMacroPlanError(
                "interface symmetry axis requires one canonical member pair"
            )
        if self.axis not in {"x", "y"}:
            raise RigidNonmosMacroPlanError(
                "interface symmetry axis must be x or y"
            )
        object.__setattr__(self, "coordinate_2nm", _coordinate(
            self.coordinate_2nm, "interface doubled-nm symmetry coordinate"
        ))

    def payload(self) -> tuple[object, ...]:
        return (self.members, self.axis, self.coordinate_2nm)


@dataclass(frozen=True, slots=True, kw_only=True, order=True)
class RigidNonmosMacroInterface:
    macro_id: str
    plan_sha256: str
    members: tuple[str, ...]
    local_bbox_nm: tuple[int, int, int, int]
    terminal_accesses: tuple[PhysicalMacroTerminalAccess, ...]
    local_symmetry_axes_2nm: tuple[RigidNonmosLocalSymmetryAxis, ...]
    body_tie_domain: tuple[str, ...]
    member_orientations: tuple[tuple[str, tuple[str, ...]], ...]
    route_demand: tuple[PhysicalMacroRouteDemand, ...]
    artwork_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "macro_id", _text(self.macro_id, "interface macro id"))
        object.__setattr__(self, "plan_sha256", _digest(
            self.plan_sha256, "interface plan_sha256"
        ))
        members = _bounded_tuple(self.members, "interface macro members", allow_empty=False)
        if members != tuple(sorted(set(members))) or any(
            type(item) is not str or not item for item in members
        ):
            raise RigidNonmosMacroPlanError("interface macro members must be canonical")
        local_bbox = _bbox(self.local_bbox_nm, "interface macro local bbox")
        if local_bbox[0:2] != (0, 0):
            raise RigidNonmosMacroPlanError(
                "interface macro local bbox must start at the origin"
            )
        object.__setattr__(self, "local_bbox_nm", local_bbox)
        terminals = _bounded_tuple(
            self.terminal_accesses, "interface macro terminals", allow_empty=False
        )
        if (
            any(type(item) is not PhysicalMacroTerminalAccess for item in terminals)
            or terminals != tuple(sorted(terminals))
            or any(item.member not in members for item in terminals)
            or len({(item.member, item.role) for item in terminals}) != len(terminals)
        ):
            raise RigidNonmosMacroPlanError(
                "interface macro terminals must be canonical, unique, and owned"
            )
        if set(members) - {item.member for item in terminals}:
            raise RigidNonmosMacroPlanError(
                "interface macro lacks a member terminal view"
            )
        axes = _bounded_tuple(
            self.local_symmetry_axes_2nm, "interface macro symmetry axes"
        )
        if any(type(item) is not RigidNonmosLocalSymmetryAxis for item in axes):
            raise RigidNonmosMacroPlanError(
                "interface symmetry axes must contain exact typed values"
            )
        if axes != tuple(sorted(axes)) or any(
            not set(item.members) <= set(members) for item in axes
        ):
            raise RigidNonmosMacroPlanError(
                "interface symmetry axes must be canonical and owned"
            )
        body = _bounded_tuple(self.body_tie_domain, "interface macro body domain")
        if body != tuple(sorted(set(body))) or any(
            type(item) is not str or not item for item in body
        ):
            raise RigidNonmosMacroPlanError(
                "interface macro body domain must be canonical"
            )
        orientations = _bounded_tuple(
            self.member_orientations, "interface member orientations", allow_empty=False
        )
        rebuilt_orientations = []
        for item in orientations:
            if type(item) is not tuple or len(item) != 2:
                raise RigidNonmosMacroPlanError(
                    "interface member orientation entry must be exact"
                )
            member, allowed = item
            if member not in members or type(allowed) is not tuple or not allowed:
                raise RigidNonmosMacroPlanError(
                    "interface member orientation is not owned"
                )
            if allowed != tuple(sorted(set(allowed))) or any(
                value not in _ORTHOGONAL_TRANSFORMS
                for value in allowed
            ):
                raise RigidNonmosMacroPlanError(
                    "interface member orientation is invalid"
                )
            rebuilt_orientations.append((member, allowed))
        if orientations != tuple(sorted(rebuilt_orientations)) or {
            item[0] for item in orientations
        } != set(members):
            raise RigidNonmosMacroPlanError(
                "interface member orientations must cover the macro"
            )
        demand = _bounded_tuple(
            self.route_demand, "interface macro route demand", allow_empty=False
        )
        if any(type(item) is not PhysicalMacroRouteDemand for item in demand) or demand != tuple(sorted(demand)):
            raise RigidNonmosMacroPlanError(
                "interface route demand must be canonical exact values"
            )
        derived = tuple(
            PhysicalMacroRouteDemand(
                net=net,
                members=tuple(sorted({item.member for item in selected})),
                terminal_count=len(selected),
                access_count=sum(len(item.accesses_nm) for item in selected),
            )
            for net in sorted({item.net for item in terminals})
            if (selected := tuple(item for item in terminals if item.net == net))
        )
        if demand != derived:
            raise RigidNonmosMacroPlanError(
                "interface route demand differs from terminal ownership"
            )
        object.__setattr__(self, "artwork_sha256", _digest(
            self.artwork_sha256, "interface macro artwork_sha256"
        ))

    def payload(self) -> dict[str, object]:
        return {
            "schema": "clay.rigid-nonmos-macro-interface/v1",
            "macro_id": self.macro_id,
            "plan_sha256": self.plan_sha256,
            "members": self.members,
            "local_bbox_nm": self.local_bbox_nm,
            "terminal_accesses": tuple(
                item.payload() for item in self.terminal_accesses
            ),
            "local_symmetry_axes_2nm": tuple(
                item.payload() for item in self.local_symmetry_axes_2nm
            ),
            "body_tie_domain": self.body_tie_domain,
            "member_orientations": self.member_orientations,
            "route_demand": tuple(item.payload() for item in self.route_demand),
            "artwork_sha256": self.artwork_sha256,
        }

    @property
    def structure_fingerprint(self) -> str:
        return _sha(self.payload())


@dataclass(frozen=True, slots=True, kw_only=True)
class RigidNonmosMacroInterfaceSet:
    graph_sha256: str
    plans_sha256: str
    tech_id: str
    macros: tuple[RigidNonmosMacroInterface, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "graph_sha256", _digest(
            self.graph_sha256, "interface set graph_sha256"
        ))
        object.__setattr__(self, "plans_sha256", _digest(
            self.plans_sha256, "interface set plans_sha256"
        ))
        object.__setattr__(self, "tech_id", _text(self.tech_id, "interface set tech_id"))
        macros = _bounded_tuple(self.macros, "rigid non-MOS interfaces")
        if any(type(item) is not RigidNonmosMacroInterface for item in macros):
            raise RigidNonmosMacroPlanError(
                "interface set requires exact typed macros"
            )
        if macros != tuple(sorted(macros)) or len({item.macro_id for item in macros}) != len(macros):
            raise RigidNonmosMacroPlanError("rigid non-MOS interfaces must be canonical")

    def payload(self) -> dict[str, object]:
        return {
            "schema": "clay.rigid-nonmos-macro-interface-set/v1",
            "graph_sha256": self.graph_sha256,
            "plans_sha256": self.plans_sha256,
            "tech_id": self.tech_id,
            "macros": tuple(item.payload() for item in self.macros),
        }

    @property
    def sha256(self) -> str:
        return _sha(self.payload())


@dataclass(frozen=True, slots=True, kw_only=True, weakref_slot=True)
class RigidNonmosEmissionCertificate:
    """Aggregate ownership joined to the exact emitter device snapshot."""

    graph_sha256: str
    plans_sha256: str
    compile_inputs_sha256: str
    interfaces_sha256: str
    placement_certificate_sha256: str
    emitted_device_snapshot_sha256: str
    terminal_pin_snapshot_sha256: str
    macro_structure_fingerprints: tuple[tuple[str, str], ...]
    owned_member_artwork: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        for field in (
            "graph_sha256",
            "plans_sha256",
            "compile_inputs_sha256",
            "interfaces_sha256",
            "placement_certificate_sha256",
            "emitted_device_snapshot_sha256",
            "terminal_pin_snapshot_sha256",
        ):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        for field in ("macro_structure_fingerprints", "owned_member_artwork"):
            values = _bounded_tuple(getattr(self, field), field)
            rebuilt = []
            for item in values:
                if type(item) is not tuple or len(item) != 2:
                    raise RigidNonmosMacroPlanError(
                        f"{field} entries must be exact pairs"
                    )
                rebuilt.append((_text(item[0], field), _digest(item[1], field)))
            if values != tuple(sorted(rebuilt)) or len({item[0] for item in values}) != len(values):
                raise RigidNonmosMacroPlanError(f"{field} must be canonical")

    def payload(self) -> dict[str, object]:
        return {
            "schema": "clay.rigid-nonmos-emission-certificate/v2",
            "graph_sha256": self.graph_sha256,
            "plans_sha256": self.plans_sha256,
            "compile_inputs_sha256": self.compile_inputs_sha256,
            "interfaces_sha256": self.interfaces_sha256,
            "placement_certificate_sha256": self.placement_certificate_sha256,
            "emitted_device_snapshot_sha256": self.emitted_device_snapshot_sha256,
            "terminal_pin_snapshot_sha256": self.terminal_pin_snapshot_sha256,
            "macro_structure_fingerprints": self.macro_structure_fingerprints,
            "owned_member_artwork": self.owned_member_artwork,
        }

    @property
    def sha256(self) -> str:
        return _sha(self.payload())

    def assert_authoritative(self) -> None:
        authority = _RIGID_EMISSION_AUTHORITIES.get(id(self))
        if (
            authority is None
            or authority[0]() is not self
            or authority[1] != self.sha256
        ):
            raise RigidNonmosMacroProjectionError(
                "RIGID_NONMOS_EMISSION_CERTIFICATE_UNAUTHORITATIVE"
            )

    def bind_exact_artifact(self, artifact: object) -> None:
        from clay.layout.route.exact_artifact_evidence import ExactEmittedArtifact

        self.assert_authoritative()
        if type(artifact) is not ExactEmittedArtifact:
            raise RigidNonmosMacroProjectionError(
                "RIGID_NONMOS_EXACT_ARTIFACT_TYPE_MISMATCH"
            )
        artifact.validate()
        if (
            artifact.scope.emitted_device_snapshot_sha256
            != self.emitted_device_snapshot_sha256
        ):
            raise RigidNonmosMacroProjectionError(
                "RIGID_NONMOS_EMITTER_SNAPSHOT_MISMATCH"
            )


_RIGID_EMISSION_AUTHORITIES: dict[
    int, tuple[weakref.ReferenceType[RigidNonmosEmissionCertificate], str]
] = {}


def _seal_rigid_nonmos_emission_certificate(
    certificate: RigidNonmosEmissionCertificate,
) -> RigidNonmosEmissionCertificate:
    key = id(certificate)

    def discard(reference: object) -> None:
        authority = _RIGID_EMISSION_AUTHORITIES.get(key)
        if authority is not None and authority[0] is reference:
            _RIGID_EMISSION_AUTHORITIES.pop(key, None)

    _RIGID_EMISSION_AUTHORITIES[key] = (
        weakref.ref(certificate, discard),
        certificate.sha256,
    )
    return certificate


@dataclass(frozen=True, slots=True, kw_only=True)
class RigidNonmosMacroCompileInputs:
    graph_sha256: str
    tech_id: str
    gap_nm: int
    member_sizes: tuple[RigidNonmosMemberSize, ...]
    member_relations: tuple[RigidNonmosMemberRelation, ...] = ()
    member_alignments: tuple[RigidNonmosMemberAlignment, ...] = ()
    member_interfaces: tuple[RigidNonmosMemberInterfaceSnapshot, ...] = ()
    native_assemblies: tuple = ()
    member_adjacencies: tuple[RigidNonmosMemberAdjacency, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "graph_sha256", _digest(
            self.graph_sha256, "compile input graph_sha256"
        ))
        object.__setattr__(self, "tech_id", _text(
            self.tech_id, "compile input tech_id"
        ))
        if type(self.gap_nm) is not int or not 0 < self.gap_nm <= _MAX_COORDINATE_NM:
            raise RigidNonmosMacroPlanError(
                "compile input gap_nm must be a bounded positive exact integer"
            )
        sizes = _bounded_tuple(
            self.member_sizes, "compile input member sizes"
        )
        if any(type(item) is not RigidNonmosMemberSize for item in sizes):
            raise RigidNonmosMacroPlanError(
                "compile input sizes must contain exact typed values"
            )
        if sizes != tuple(sorted(sizes)) or len({item.member for item in sizes}) != len(sizes):
            raise RigidNonmosMacroPlanError(
                "compile input member sizes must be canonical"
            )
        relations = _bounded_tuple(
            self.member_relations, "compile input member relations"
        )
        if any(type(item) is not RigidNonmosMemberRelation for item in relations):
            raise RigidNonmosMacroPlanError(
                "compile input relations must contain exact typed values"
            )
        if relations != tuple(sorted(relations)) or len(set(relations)) != len(relations):
            raise RigidNonmosMacroPlanError(
                "compile input member relations must be canonical"
            )
        sized_members = {item.member for item in sizes}
        if any(
            item.left not in sized_members or item.right not in sized_members
            for item in relations
        ):
            raise RigidNonmosMacroPlanError(
                "compile input member relation names a foreign member"
            )
        alignments = _bounded_tuple(
            self.member_alignments, "compile input member alignments"
        )
        if any(type(item) is not RigidNonmosMemberAlignment for item in alignments):
            raise RigidNonmosMacroPlanError(
                "compile input alignments must contain exact typed values"
            )
        if (
            alignments != tuple(sorted(alignments))
            or len(set(alignments)) != len(alignments)
        ):
            raise RigidNonmosMacroPlanError(
                "compile input member alignments must be canonical"
            )
        if any(
            item.first not in sized_members or item.second not in sized_members
            for item in alignments
        ):
            raise RigidNonmosMacroPlanError(
                "compile input member alignment names a foreign member"
            )
        adjacencies = _bounded_tuple(self.member_adjacencies, "compile input member adjacencies")
        if any(type(item) is not RigidNonmosMemberAdjacency for item in adjacencies):
            raise RigidNonmosMacroPlanError("compile input adjacencies require exact typed values")
        if adjacencies != tuple(sorted(set(adjacencies))):
            raise RigidNonmosMacroPlanError("compile input adjacencies must be canonical")
        if any(item.first not in sized_members or item.second not in sized_members for item in adjacencies):
            raise RigidNonmosMacroPlanError("compile input adjacency names a foreign member")
        interfaces = _bounded_tuple(
            self.member_interfaces, "compile input member interfaces"
        )
        if any(
            type(item) is not RigidNonmosMemberInterfaceSnapshot
            for item in interfaces
        ):
            raise RigidNonmosMacroPlanError(
                "compile input interfaces require exact typed snapshots"
            )
        if interfaces != tuple(sorted(interfaces)) or len({item.member for item in interfaces}) != len(interfaces):
            raise RigidNonmosMacroPlanError(
                "compile input interfaces must be canonical"
            )
        if interfaces and {item.member for item in interfaces} != {
            item.member for item in sizes
        }:
            raise RigidNonmosMacroPlanError(
                "compile input interfaces differ from member sizes"
            )

        if self.native_assemblies:
            from clay.layout.source_bound_nonmos_plan import validate_native_assemblies
            validate_native_assemblies(self.native_assemblies, tech_id=self.tech_id, member_sizes=self.member_sizes)
        elif type(self.native_assemblies) is not tuple:
            raise RigidNonmosMacroPlanError('native assemblies must be an exact tuple')

    def payload(self) -> dict[str, object]:
        payload = {
            "schema": (
                "clay.rigid-nonmos-macro-compile-inputs/v4"
                if self.member_alignments
                else "clay.rigid-nonmos-macro-compile-inputs/v3"
            ),
            "graph_sha256": self.graph_sha256,
            "tech_id": self.tech_id,
            "gap_nm": self.gap_nm,
            "member_sizes": tuple(item.payload() for item in self.member_sizes),
            "member_relations": tuple(
                item.payload() for item in self.member_relations
            ),
            "member_interfaces": tuple(
                item.payload() for item in self.member_interfaces
            ),
        }
        if self.member_alignments:
            payload["member_alignments"] = tuple(
                item.payload() for item in self.member_alignments
            )
        if self.native_assemblies:
            payload['schema'] = 'clay.rigid-nonmos-macro-compile-inputs/v5'
            payload['native_assemblies'] = tuple(a.sha256 for a in self.native_assemblies)
        if getattr(self, 'member_adjacencies', ()):
            payload['schema'] = 'clay.rigid-nonmos-macro-compile-inputs/v6'
            payload['member_adjacencies'] = tuple(item.payload() for item in self.member_adjacencies)
        return payload

    @property
    def sha256(self) -> str:
        return _sha(self.payload())


def _stable_atom_topological_order(
    atoms: tuple[tuple[str, ...], ...],
    edges: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...],
    *,
    axis: str,
) -> tuple[tuple[str, ...], ...]:
    """Return one canonical linear extension or reject a local relation cycle."""

    outgoing = {atom: set() for atom in atoms}
    incoming = {atom: 0 for atom in atoms}
    for left, right in edges:
        if right not in outgoing[left]:
            outgoing[left].add(right)
            incoming[right] += 1
    def ready_key(atom: tuple[str, ...]) -> tuple[object, ...]:
        return (-len(outgoing[atom]), atom)

    ready = sorted(
        (atom for atom in atoms if incoming[atom] == 0), key=ready_key
    )
    ordered = []
    while ready:
        current = ready.pop(0)
        ordered.append(current)
        for follower in sorted(outgoing[current]):
            incoming[follower] -= 1
            if incoming[follower] == 0:
                ready.append(follower)
                ready.sort(key=ready_key)
    if len(ordered) != len(atoms):
        raise RigidNonmosMacroPlanError(
            f"aggregate macro has a cyclic local {axis} relation"
        )
    return tuple(ordered)


def _apply_rigid_member_alignments(
    member_boxes: Mapping[str, tuple[int, int, int, int]],
    relations: tuple[RigidNonmosMemberRelation, ...],
    alignments: tuple[RigidNonmosMemberAlignment, ...],
    *,
    gap_nm: int,
) -> dict[str, tuple[int, int, int, int]]:
    """Compact one full-2D local topology while preserving exact edge equalities."""

    from itertools import combinations

    from clay.layout.extract.dcsolve import _solve_axis

    names = sorted(member_boxes)
    sizes = {
        name: (
            member_boxes[name][2] - member_boxes[name][0],
            member_boxes[name][3] - member_boxes[name][1],
        )
        for name in names
    }
    triples: dict[str, list[tuple[str, str, float]]] = {"x": [], "y": []}
    for alignment in alignments:
        first_width, first_height = sizes[alignment.first]
        second_width, second_height = sizes[alignment.second]
        if alignment.edge == "left":
            axis, offset = "x", 0
        elif alignment.edge == "right":
            axis, offset = "x", first_width - second_width
        elif alignment.edge == "bottom":
            axis, offset = "y", 0
        else:
            axis, offset = "y", first_height - second_height
        triples[axis].append((alignment.second, alignment.first, float(offset)))

    solutions = {
        axis: _solve_axis(names, triples[axis], names[0])
        for axis in ("x", "y")
    }
    if any(solution.conflicts for solution in solutions.values()):
        raise RigidNonmosMacroPlanError(
            "aggregate macro has inconsistent local edge alignments"
        )

    # Each explicit relation remains an inequality. Other pairs retain exactly one
    # separation edge selected from the already legal 2-D seed, never a fake row.
    separations: dict[str, list[tuple[str, str]]] = {"x": [], "y": []}
    related_pairs = set()
    for relation in relations:
        axis = "x" if relation.relation == "left_of" else "y"
        separations[axis].append((relation.left, relation.right))
        related_pairs.add(frozenset((relation.left, relation.right)))
    for first, second in combinations(names, 2):
        if frozenset((first, second)) in related_pairs:
            continue
        left = member_boxes[first]
        right = member_boxes[second]
        candidates = []
        if left[2] + gap_nm <= right[0]:
            candidates.append((right[0] - left[2] - gap_nm, "x", first, second))
        if right[2] + gap_nm <= left[0]:
            candidates.append((left[0] - right[2] - gap_nm, "x", second, first))
        if left[3] + gap_nm <= right[1]:
            candidates.append((right[1] - left[3] - gap_nm, "y", first, second))
        if right[3] + gap_nm <= left[1]:
            candidates.append((left[1] - right[3] - gap_nm, "y", second, first))
        if not candidates:
            raise RigidNonmosMacroPlanError(
                "aggregate macro seed lacks full-2D pair separation: "
                f"{first},{second}"
            )
        _, axis, lower, upper = min(candidates)
        separations[axis].append((lower, upper))

    def solve_coordinate(axis: str) -> dict[str, int]:
        solution = solutions[axis]
        size_index = 0 if axis == "x" else 1
        roots = sorted({solution.root_of[name] for name in names})
        base = {
            root: max(
                -solution.pot[name]
                for name in names
                if solution.root_of[name] == root
            )
            for root in roots
        }
        edges = []
        for lower, upper in sorted(set(separations[axis])):
            lower_root = solution.root_of[lower]
            upper_root = solution.root_of[upper]
            weight = (
                solution.pot[lower]
                + sizes[lower][size_index]
                + gap_nm
                - solution.pot[upper]
            )
            if lower_root == upper_root:
                if weight > 0:
                    raise RigidNonmosMacroPlanError(
                        "aggregate macro alignment conflicts with local separation: "
                        f"{lower},{upper},{axis}"
                    )
                continue
            edges.append((lower_root, upper_root, weight))
        for iteration in range(len(roots) + 1):
            changed = False
            for lower_root, upper_root, weight in edges:
                required = base[lower_root] + weight
                if base[upper_root] < required:
                    base[upper_root] = required
                    changed = True
            if not changed:
                break
            if iteration == len(roots):
                raise RigidNonmosMacroPlanError(
                    f"aggregate macro has a cyclic local {axis} separation graph"
                )
        coordinates = {}
        for name in names:
            value = base[solution.root_of[name]] + solution.pot[name]
            rounded = round(value)
            if abs(value - rounded) > 1e-9:
                raise RigidNonmosMacroPlanError(
                    "aggregate macro alignment escaped the integer nm grid"
                )
            coordinates[name] = int(rounded)
        shift = min(coordinates.values())
        return {name: value - shift for name, value in coordinates.items()}

    x = solve_coordinate("x")
    y = solve_coordinate("y")
    rebuilt = {
        name: (x[name], y[name], x[name] + sizes[name][0], y[name] + sizes[name][1])
        for name in names
    }
    for alignment in alignments:
        coordinate = {"left": 0, "bottom": 1, "right": 2, "top": 3}[
            alignment.edge
        ]
        if rebuilt[alignment.first][coordinate] != rebuilt[alignment.second][coordinate]:
            raise RigidNonmosMacroPlanError(
                "aggregate macro failed to materialize local alignment: "
                f"{alignment.declaration_ref}"
            )
    for relation in relations:
        left = rebuilt[relation.left]
        right = rebuilt[relation.right]
        satisfied = (
            left[2] + gap_nm <= right[0]
            if relation.relation == "left_of"
            else left[3] + gap_nm <= right[1]
        )
        if not satisfied:
            raise RigidNonmosMacroPlanError(
                "aggregate macro failed to materialize local relation: "
                f"{relation.declaration_ref}"
            )
    for first, second in combinations(names, 2):
        left = rebuilt[first]
        right = rebuilt[second]
        if not (
            left[2] + gap_nm <= right[0]
            or right[2] + gap_nm <= left[0]
            or left[3] + gap_nm <= right[1]
            or right[3] + gap_nm <= left[1]
        ):
            raise RigidNonmosMacroPlanError(
                "aggregate macro alignment introduced a local overlap: "
                f"{first},{second}"
            )
    return rebuilt


def compile_rigid_nonmos_macro_plans_from_inputs(
    graph: object,
    inputs: RigidNonmosMacroCompileInputs,
) -> RigidNonmosMacroPlanSet:
    """Deterministically tile aggregate non-MOS macros from exact size inputs."""

    from math import ceil, sqrt
    from clay.layout.physical_macro_graph import PhysicalMacroGraph

    if type(graph) is not PhysicalMacroGraph:
        raise RigidNonmosMacroPlanError(
            "graph must be an exact PhysicalMacroGraph"
        )
    if type(inputs) is not RigidNonmosMacroCompileInputs:
        raise RigidNonmosMacroPlanError(
            "compile inputs must be exact RigidNonmosMacroCompileInputs"
        )
    if inputs.graph_sha256 != graph.sha256:
        raise RigidNonmosMacroPlanError(
            "compile inputs do not bind the current physical macro graph"
        )
    sizes = {
        item.member: (item.width_nm, item.height_nm)
        for item in inputs.member_sizes
    }
    structural = tuple(
        macro for macro in graph.macros
        if macro.kind in _KINDS
    )
    expected = {member for macro in structural for member in macro.members}
    if set(sizes) != expected:
        raise RigidNonmosMacroPlanError(
            "compile input size members differ from aggregate graph members"
        )
    for relation in inputs.member_relations:
        owners = tuple(
            macro.macro_id
            for macro in structural
            if {relation.left, relation.right} <= set(macro.members)
        )
        if len(owners) != 1:
            raise RigidNonmosMacroPlanError(
                "compile input member relation lacks one rigid macro owner: "
                f"{relation.declaration_ref}"
            )
    for alignment in inputs.member_alignments:
        owners = tuple(
            macro.macro_id
            for macro in structural
            if {alignment.first, alignment.second} <= set(macro.members)
        )
        if len(owners) != 1:
            raise RigidNonmosMacroPlanError(
                "compile input member alignment lacks one rigid macro owner: "
                f"{alignment.declaration_ref}"
            )
    adjacencies = getattr(inputs, 'member_adjacencies', ())
    for adjacency in adjacencies:
        owners = tuple(macro for macro in structural
                       if {adjacency.first, adjacency.second} <= set(macro.members))
        if len(owners) != 1:
            raise RigidNonmosMacroPlanError("internal adjacency lacks one rigid macro owner")
    matched = tuple(
        macro.members for macro in graph.macros
        if macro.kind == "matched_array" and len(macro.members) == 2
    )
    plans = []
    from clay.layout.source_bound_nonmos_plan import compile_source_bound_plans
    native_plans = compile_source_bound_plans(graph, inputs) if inputs.native_assemblies else {}
    for macro in structural:
        local_adjacencies = tuple(item for item in adjacencies
                                  if {item.first, item.second} <= set(macro.members))
        # The blanket well-to-well gap is only a packing default.  An authored
        # local maximum constrains it; the producer independently checks every
        # actual native layer before sealing these plans for placement.
        gap = min((inputs.gap_nm, *(item.maximum_nm for item in local_adjacencies)))
        if macro.macro_id in native_plans:
            _require_internal_adjacencies(
                {item.member: item.local_bbox_nm for item in native_plans[macro.macro_id].member_plans},
                local_adjacencies)
            plans.append(native_plans[macro.macro_id])
            continue
        pairs = tuple(sorted(
            pair for pair in matched if set(pair) <= set(macro.members)
        ))
        paired_members = {member for pair in pairs for member in pair}
        if len(paired_members) != 2 * len(pairs):
            raise RigidNonmosMacroPlanError(
                f"aggregate macro has overlapping matching pairs: {macro.macro_id}"
            )
        atoms = [tuple(pair) for pair in pairs]
        atoms.extend((member,) for member in macro.members if member not in paired_members)
        atoms = sorted(atoms, key=lambda item: item[0])
        canonical_atoms = tuple(atoms)
        member_atom = {
            member: atom for atom in canonical_atoms for member in atom
        }
        local_relations = tuple(
            relation
            for relation in inputs.member_relations
            if {relation.left, relation.right} <= set(macro.members)
        )
        local_alignments = tuple(
            alignment
            for alignment in inputs.member_alignments
            if {alignment.first, alignment.second} <= set(macro.members)
        )
        for relation in local_relations:
            if member_atom[relation.left] == member_atom[relation.right]:
                raise RigidNonmosMacroPlanError(
                    "aggregate macro relation would split a matched atom: "
                    f"{relation.declaration_ref}"
                )
        for alignment in local_alignments:
            if member_atom[alignment.first] == member_atom[alignment.second]:
                raise RigidNonmosMacroPlanError(
                    "aggregate macro alignment would split a matched atom: "
                    f"{alignment.declaration_ref}"
                )
        attributes = dict(macro.attributes)
        raw_pack_rows = attributes.get("pack_rows")
        if raw_pack_rows is None:
            ncol = max(1, ceil(sqrt(len(atoms))))
            atom_rows = tuple(
                tuple(atoms[start:start + ncol])
                for start in range(0, len(atoms), ncol)
            )
        else:
            try:
                selected_rows = int(raw_pack_rows)
            except (TypeError, ValueError) as exc:
                raise RigidNonmosMacroPlanError(
                    f"aggregate macro pack_rows is malformed: {macro.macro_id}"
                ) from exc
            if (
                str(selected_rows) != raw_pack_rows
                or not 1 <= selected_rows <= len(atoms)
            ):
                raise RigidNonmosMacroPlanError(
                    f"aggregate macro pack_rows is unrealizable: {macro.macro_id}"
                )
            quotient, remainder = divmod(len(atoms), selected_rows)
            row_sizes = tuple(
                quotient + (1 if index < remainder else 0)
                for index in range(selected_rows)
            )
            atom_rows_list = []
            start = 0
            for row_size in row_sizes:
                atom_rows_list.append(tuple(atoms[start:start + row_size]))
                start += row_size
            atom_rows = tuple(atom_rows_list)
        if local_relations:
            baseline_row = {
                atom: row_index
                for row_index, row in enumerate(atom_rows)
                for atom in row
            }
            x_edges = tuple(sorted({
                (member_atom[relation.left], member_atom[relation.right])
                for relation in local_relations
                if relation.relation == "left_of"
            }))
            y_edges = tuple(sorted({
                (member_atom[relation.left], member_atom[relation.right])
                for relation in local_relations
                if relation.relation == "below"
            }))
            x_order = _stable_atom_topological_order(
                canonical_atoms, x_edges, axis="horizontal"
            )
            y_order = _stable_atom_topological_order(
                canonical_atoms, y_edges, axis="vertical"
            )
            y_members = {
                atom for edge in y_edges for atom in edge
            }
            y_rank = {atom: 0 for atom in canonical_atoms}
            y_followers = {atom: set() for atom in canonical_atoms}
            for lower, upper in y_edges:
                y_followers[lower].add(upper)
            for atom in y_order:
                for upper in y_followers[atom]:
                    y_rank[upper] = max(y_rank[upper], y_rank[atom] + 1)
            assigned_row = {
                atom: (
                    y_rank[atom] if atom in y_members else baseline_row[atom]
                )
                for atom in canonical_atoms
            }
            if raw_pack_rows is not None and max(assigned_row.values()) >= int(
                raw_pack_rows
            ):
                raise RigidNonmosMacroPlanError(
                    f"aggregate macro local relations exceed pack_rows: {macro.macro_id}"
                )
            x_rank = {atom: index for index, atom in enumerate(x_order)}
            row_ids = tuple(sorted(set(assigned_row.values())))
            atom_rows = tuple(
                tuple(sorted(
                    (atom for atom in canonical_atoms if assigned_row[atom] == row),
                    key=lambda atom: (x_rank[atom], atom),
                ))
                for row in row_ids
            )
            row_edges = tuple(
                (left, right)
                for row in atom_rows
                for left, right in zip(row, row[1:])
            )
            all_x_edges = tuple(sorted(set(x_edges) | set(row_edges)))
            x_order = _stable_atom_topological_order(
                canonical_atoms, all_x_edges, axis="horizontal"
            )
            atom_width = {
                atom: sum(sizes[member][0] for member in atom)
                + gap * (len(atom) - 1)
                for atom in canonical_atoms
            }
            x_followers = {atom: set() for atom in canonical_atoms}
            for left, right in all_x_edges:
                x_followers[left].add(right)
            atom_x = {atom: 0 for atom in canonical_atoms}
            for atom in x_order:
                required = atom_x[atom] + atom_width[atom] + gap
                for right in x_followers[atom]:
                    atom_x[right] = max(atom_x[right], required)
        else:
            atom_x = {}
        member_boxes: dict[str, tuple[int, int, int, int]] = {}
        max_width = 0
        cursor_y = 0
        for row_index, row in enumerate(atom_rows):
            row_height = max(sizes[member][1] for atom in row for member in atom)
            cursor_x = 0
            for atom in row:
                atom_height = max(sizes[member][1] for member in atom)
                if local_relations:
                    cursor_x = atom_x[atom]
                for index, member in enumerate(atom):
                    width, height = sizes[member]
                    y0 = cursor_y + (atom_height - height) // 2
                    member_boxes[member] = (
                        cursor_x, y0, cursor_x + width, y0 + height
                    )
                    cursor_x += width
                    if index + 1 < len(atom):
                        cursor_x += gap
                cursor_x += gap
            row_width = cursor_x - gap
            max_width = max(max_width, row_width)
            cursor_y += row_height
            if row_index + 1 < len(atom_rows):
                cursor_y += gap
        if local_alignments:
            member_boxes = _apply_rigid_member_alignments(
                member_boxes,
                local_relations,
                local_alignments,
                gap_nm=gap,
            )
            max_width = max(box[2] for box in member_boxes.values())
            cursor_y = max(box[3] for box in member_boxes.values())
        for relation in local_relations:
            left = member_boxes[relation.left]
            right = member_boxes[relation.right]
            satisfied = (
                left[2] + gap <= right[0]
                if relation.relation == "left_of"
                else left[3] + gap <= right[1]
            )
            if not satisfied:
                raise RigidNonmosMacroPlanError(
                    "aggregate macro failed to materialize local relation: "
                    f"{relation.declaration_ref}"
                )
        _require_internal_adjacencies(member_boxes, local_adjacencies)
        plans.append(RigidNonmosMacroPlan(
            macro_id=macro.macro_id,
            kind=macro.kind,
            members=macro.members,
            member_plans=tuple(
                RigidNonmosMacroMemberPlan(
                    member=member, local_bbox_nm=member_boxes[member]
                )
                for member in macro.members
            ),
            local_bbox_nm=(0, 0, max_width, cursor_y),
            matching_pairs=pairs,
            local_relation_projections=tuple(sorted(
                RigidNonmosLocalRelationProjection(
                    macro_id=macro.macro_id,
                    declaration_ref=relation.declaration_ref,
                    members=(relation.left, relation.right),
                    relation=relation.relation,
                    gap_nm=gap,
                    pre_local_bboxes_nm=tuple(sorted((
                        (
                            member,
                            (0, 0, sizes[member][0], sizes[member][1]),
                        )
                        for member in (relation.left, relation.right)
                    ))),
                    post_local_bboxes_nm=tuple(sorted((
                        (member, member_boxes[member])
                        for member in (relation.left, relation.right)
                    ))),
                )
                for relation in local_relations
            )),
            local_alignment_projections=tuple(sorted(
                RigidNonmosLocalAlignmentProjection(
                    macro_id=macro.macro_id,
                    declaration_ref=alignment.declaration_ref,
                    members=(alignment.first, alignment.second),
                    edge=alignment.edge,
                    pre_local_bboxes_nm=tuple(sorted((
                        (
                            member,
                            (0, 0, sizes[member][0], sizes[member][1]),
                        )
                        for member in (alignment.first, alignment.second)
                    ))),
                    post_local_bboxes_nm=tuple(sorted((
                        (member, member_boxes[member])
                        for member in (alignment.first, alignment.second)
                    ))),
                )
                for alignment in local_alignments
            )),
        ))
    return RigidNonmosMacroPlanSet(
        graph_sha256=graph.sha256,
        tech_id=inputs.tech_id,
        plans=tuple(sorted(plans, key=lambda p: p.macro_id)),
    )


def _terminal_center(terminal: object, field: str) -> tuple[int, int]:
    raw_bbox = getattr(terminal, "bbox", None)
    if type(raw_bbox) is not tuple or len(raw_bbox) != 4:
        raise RigidNonmosMacroPlanError(
            f"{field} must expose one exact bbox tuple"
        )
    bbox = tuple(_coordinate(item, field) for item in raw_bbox)
    if bbox[2] < bbox[0] or bbox[3] < bbox[1]:
        raise RigidNonmosMacroPlanError(f"{field} bbox is inverted")
    return ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)


def _terminal_entry_layer(
    terminal: object,
    polygons: object,
    metal_keys: tuple[tuple[int, int], ...],
) -> tuple[int, int] | None:
    """Highest declared routing metal covering the terminal marker centre."""

    cx, cy = _terminal_center(terminal, "interface terminal marker")
    covering = set()
    for polygon in tuple(polygons or ()):
        key = (getattr(polygon, "layer", None), getattr(polygon, "datatype", None))
        if key not in metal_keys:
            continue
        vertices = tuple(
            getattr(getattr(polygon, "polygon", None), "vertices", ()) or ()
        )
        if not vertices:
            continue
        xs = tuple(getattr(point, "x", None) for point in vertices)
        ys = tuple(getattr(point, "y", None) for point in vertices)
        if any(type(value) is not int for value in (*xs, *ys)):
            raise RigidNonmosMacroPlanError(
                "interface artwork contains a non-integer vertex"
            )
        if min(xs) <= cx <= max(xs) and min(ys) <= cy <= max(ys):
            covering.add(key)
    return next((key for key in reversed(metal_keys) if key in covering), None)


def compile_rigid_nonmos_macro_interfaces(
    graph: object,
    plans: RigidNonmosMacroPlanSet,
    inputs: RigidNonmosMacroCompileInputs,
    *,
    geometries: Mapping[str, object],
    device_pins: Mapping[str, Mapping[str, str]],
    canonical_terminal_accesses: Mapping[
        str, tuple[PhysicalMacroTerminalAccess, ...]
    ] | None = None,
    tech: object,
) -> tuple[RigidNonmosMacroInterfaceSet, RigidNonmosMacroCompileInputs]:
    """Freeze exact PDK artwork and routing interfaces for every rigid member."""

    if type(plans) is not RigidNonmosMacroPlanSet:
        raise RigidNonmosMacroPlanError("interface compiler requires exact plans")
    if type(inputs) is not RigidNonmosMacroCompileInputs:
        raise RigidNonmosMacroPlanError("interface compiler requires exact inputs")
    if (
        plans.graph_sha256 != inputs.graph_sha256
        or plans.tech_id != inputs.tech_id
        or str(getattr(tech, "pdk_id", "")) != inputs.tech_id
    ):
        raise RigidNonmosMacroPlanError(
            "interface compiler graph/plan/technology scope differs"
        )
    if not isinstance(geometries, Mapping) or not isinstance(device_pins, Mapping):
        raise RigidNonmosMacroPlanError(
            "interface compiler geometry and pins must be mappings"
        )
    if canonical_terminal_accesses is None:
        canonical_terminal_accesses = {}
    if not isinstance(canonical_terminal_accesses, Mapping):
        raise RigidNonmosMacroPlanError(
            "canonical terminal accesses must be a mapping"
        )
    expected_members = {item.member for item in inputs.member_sizes}
    if not set(canonical_terminal_accesses) <= expected_members:
        raise RigidNonmosMacroPlanError(
            "canonical terminal accesses name a foreign member"
        )
    metal_keys = []
    for level in range(1, 17):
        try:
            key = tuple(int(item) for item in tech.metal_key(level))
        except Exception:
            continue
        if len(key) != 2 or key in metal_keys:
            continue
        metal_keys.append(key)
    if not metal_keys:
        raise RigidNonmosMacroPlanError(
            "interface compiler technology exposes no routable metal1"
        )
    metal_keys_tuple = tuple(metal_keys)
    default_layer = metal_keys_tuple[0]
    snapshots = []
    for size in inputs.member_sizes:
        member = size.member
        geometry = geometries.get(member)
        pins = device_pins.get(member)
        if geometry is None or not isinstance(pins, Mapping):
            raise RigidNonmosMacroPlanError(
                f"rigid interface member lacks geometry or pins: {member}"
            )
        raw_bbox = _bbox(geometry.extent(), "interface geometry bbox")
        ox, oy = raw_bbox[0], raw_bbox[1]
        local_bbox = (0, 0, raw_bbox[2] - ox, raw_bbox[3] - oy)
        if (local_bbox[2], local_bbox[3]) != (size.width_nm, size.height_nm):
            raise RigidNonmosMacroPlanError(
                f"rigid interface geometry size changed: {member}"
            )
        raw_terminals = getattr(geometry, "terminals", None)
        if not isinstance(raw_terminals, Mapping) or not raw_terminals:
            raise RigidNonmosMacroPlanError(
                f"rigid interface member lacks terminals: {member}"
            )
        raw_accesses = getattr(geometry, "terminal_accesses", {}) or {}
        raw_layers = getattr(geometry, "terminal_layers", {}) or {}
        supplied_terminals = canonical_terminal_accesses.get(member)
        if supplied_terminals is not None:
            if type(supplied_terminals) is not tuple or any(
                type(item) is not PhysicalMacroTerminalAccess
                for item in supplied_terminals
            ):
                raise RigidNonmosMacroPlanError(
                    f"canonical terminal accesses are invalid: {member}"
                )
            expected_roles = tuple(sorted(
                (
                    member,
                    _text(raw_role, "interface terminal role"),
                    pins.get(raw_role),
                )
                for raw_role in raw_terminals
            ))
            if tuple(
                (item.member, item.role, item.net)
                for item in supplied_terminals
            ) != expected_roles:
                raise RigidNonmosMacroPlanError(
                    f"canonical terminal accesses differ from topology: {member}"
                )
            terminals = list(supplied_terminals)
        else:
            terminals = []
            for raw_role, terminal in sorted(raw_terminals.items()):
                role = _text(raw_role, "interface terminal role")
                net = pins.get(role)
                if type(net) is not str or not net:
                    raise RigidNonmosMacroPlanError(
                        f"rigid interface terminal lacks a net: {member}.{role}"
                    )
                access_terms = tuple(raw_accesses.get(role, ()) or (terminal,))
                accesses = tuple(sorted((
                    point[0] - ox,
                    point[1] - oy,
                ) for point in (
                    _terminal_center(item, f"interface terminal {member}.{role}")
                    for item in access_terms
                )))
                raw_layer = raw_layers.get(role)
                if raw_layer is None:
                    raw_layer = _terminal_entry_layer(
                        terminal,
                        getattr(geometry, "polygons", ()),
                        metal_keys_tuple,
                    ) or default_layer
                layer = tuple(raw_layer)
                terminals.append(PhysicalMacroTerminalAccess(
                    member=member,
                    role=role,
                    net=net,
                    layer=layer,
                    accesses_nm=accesses,
                ))
        orientation = str(getattr(geometry, "orient", "R0") or "R0").upper()
        body = tuple(sorted({
            str(value) for value in (pins.get("body_tie"),)
            if type(value) is str and value
        }))
        snapshots.append(RigidNonmosMemberInterfaceSnapshot(
            member=member,
            local_bbox_nm=local_bbox,
            terminal_accesses=tuple(sorted(terminals)),
            body_tie_domain=body,
            legal_orientations=(orientation,),
            artwork_sha256=_artwork_sha256(
                getattr(geometry, "polygons", ()), origin_nm=(ox, oy)
            ),
        ))
    closed_inputs = replace(inputs, member_interfaces=tuple(sorted(snapshots)))
    interfaces = replay_rigid_nonmos_macro_interfaces(graph, plans, closed_inputs)
    return interfaces, closed_inputs


def replay_rigid_nonmos_macro_interfaces(
    graph: object,
    plans: RigidNonmosMacroPlanSet,
    inputs: RigidNonmosMacroCompileInputs,
) -> RigidNonmosMacroInterfaceSet:
    """Replay aggregate interfaces solely from graph, plan, and frozen inputs."""

    replayed_plans = compile_rigid_nonmos_macro_plans_from_inputs(graph, inputs)
    if replayed_plans.sha256 != plans.sha256 or replayed_plans.payload() != plans.payload():
        raise RigidNonmosMacroPlanError(
            "interface inputs do not replay the rigid plan set"
        )
    snapshots = {item.member: item for item in inputs.member_interfaces}
    expected_members = {member for plan in plans.plans for member in plan.members}
    if set(snapshots) != expected_members:
        raise RigidNonmosMacroPlanError(
            "interface snapshots differ from rigid plan members"
        )
    macros = []
    for plan in plans.plans:
        member_plans = {item.member: item for item in plan.member_plans}
        terminals = []
        for member in plan.members:
            snapshot = snapshots[member]
            member_plan = member_plans[member]
            expected_size = (
                member_plan.local_bbox_nm[2] - member_plan.local_bbox_nm[0],
                member_plan.local_bbox_nm[3] - member_plan.local_bbox_nm[1],
            )
            if expected_size != (
                snapshot.local_bbox_nm[2], snapshot.local_bbox_nm[3]
            ):
                raise RigidNonmosMacroPlanError(
                    f"interface member footprint differs from plan: {member}"
                )
            dx, dy = member_plan.local_bbox_nm[0:2]
            terminals.extend(
                PhysicalMacroTerminalAccess(
                    member=item.member,
                    role=item.role,
                    net=item.net,
                    layer=item.layer,
                    accesses_nm=tuple(
                        (x + dx, y + dy) for x, y in item.accesses_nm
                    ),
                )
                for item in snapshot.terminal_accesses
            )
        terminal_tuple = tuple(sorted(terminals))
        demands = tuple(
            PhysicalMacroRouteDemand(
                net=net,
                members=tuple(sorted({item.member for item in selected})),
                terminal_count=len(selected),
                access_count=sum(len(item.accesses_nm) for item in selected),
            )
            for net in sorted({item.net for item in terminal_tuple})
            if (selected := tuple(item for item in terminal_tuple if item.net == net))
        )
        axes = []
        for pair in plan.matching_pairs:
            left = member_plans[pair[0]].local_bbox_nm
            right = member_plans[pair[1]].local_bbox_nm
            numerator = left[0] + left[2] + right[0] + right[2]
            if numerator % 2:
                raise RigidNonmosMacroPlanError(
                    f"matching axis is not representable in doubled nm: {pair}"
                )
            axes.append(RigidNonmosLocalSymmetryAxis(
                members=pair,
                axis="x",
                coordinate_2nm=numerator // 2,
            ))
        macros.append(RigidNonmosMacroInterface(
            macro_id=plan.macro_id,
            plan_sha256=plan.sha256,
            members=plan.members,
            local_bbox_nm=plan.local_bbox_nm,
            terminal_accesses=terminal_tuple,
            local_symmetry_axes_2nm=tuple(sorted(axes)),
            body_tie_domain=tuple(sorted({
                net
                for member in plan.members
                for net in snapshots[member].body_tie_domain
            })),
            member_orientations=tuple(
                (member, snapshots[member].legal_orientations)
                for member in plan.members
            ),
            route_demand=demands,
            artwork_sha256=_sha(tuple(
                (
                    member,
                    member_plans[member].local_bbox_nm,
                    snapshots[member].artwork_sha256,
                )
                for member in plan.members
            )),
        ))
    return RigidNonmosMacroInterfaceSet(
        graph_sha256=plans.graph_sha256,
        plans_sha256=plans.sha256,
        tech_id=inputs.tech_id,
        macros=tuple(sorted(macros)),
    )


def _route_stamp_artwork_sha256(
    stamp: object,
    position_um: object,
    bbox_nm: tuple[int, int, int, int],
) -> str:
    from clay.layout.placed_geometry import placed_point_nm

    if type(position_um) not in (tuple, list) or len(position_um) != 2:
        raise RigidNonmosMacroProjectionError(
            "RIGID_NONMOS_ROUTE_PLACEMENT_INVALID"
        )
    layers = getattr(stamp, "polys_by_layer", None)
    if not isinstance(layers, Mapping):
        raise RigidNonmosMacroProjectionError(
            "RIGID_NONMOS_ROUTE_STAMP_INVALID"
        )
    rows = []
    for raw_layer, rings in layers.items():
        if (
            type(raw_layer) is not tuple
            or len(raw_layer) != 2
            or any(type(item) is not int or not 0 <= item <= 65_535 for item in raw_layer)
            or type(rings) not in (tuple, list)
        ):
            raise RigidNonmosMacroProjectionError(
                "RIGID_NONMOS_ROUTE_STAMP_INVALID"
            )
        for ring in rings:
            try:
                points = tuple(
                    (
                        placed_point_nm(float(x), float(y), tuple(position_um))[0]
                        - bbox_nm[0],
                        placed_point_nm(float(x), float(y), tuple(position_um))[1]
                        - bbox_nm[1],
                    )
                    for x, y in tuple(ring)
                )
                rows.append((raw_layer[0], raw_layer[1], _canonical_ring(points)))
            except Exception as exc:
                raise RigidNonmosMacroProjectionError(
                    "RIGID_NONMOS_ROUTE_STAMP_INVALID"
                ) from exc
    if not rows:
        raise RigidNonmosMacroProjectionError(
            "RIGID_NONMOS_ROUTE_STAMP_INVALID"
        )
    return _sha(tuple(sorted(rows)))


def build_rigid_nonmos_emission_certificate(
    plans: RigidNonmosMacroPlanSet,
    interfaces: RigidNonmosMacroInterfaceSet,
    placement_certificate: RigidNonmosPlacementCertificate,
    *,
    scope_token: object,
    compile_inputs: RigidNonmosMacroCompileInputs,
    geometry: Mapping[str, object],
    placement_um: Mapping[str, object],
    pins_by_net: Mapping[str, object],
) -> RigidNonmosEmissionCertificate:
    """Prove exact route/emitter inputs preserve every aggregate member stamp."""

    if (
        type(plans) is not RigidNonmosMacroPlanSet
        or type(compile_inputs) is not RigidNonmosMacroCompileInputs
        or type(interfaces) is not RigidNonmosMacroInterfaceSet
        or type(placement_certificate) is not RigidNonmosPlacementCertificate
    ):
        raise RigidNonmosMacroProjectionError(
            "RIGID_NONMOS_EMISSION_METADATA_INCOMPLETE"
        )
    from clay.layout.generic_placement import (
        verify_rigid_nonmos_compiler_authority,
    )

    verify_rigid_nonmos_compiler_authority(
        scope_token,
        plans.graph_sha256,
        plans=plans,
        compile_inputs=compile_inputs,
        interfaces=interfaces,
    )
    if (
        interfaces.graph_sha256 != plans.graph_sha256
        or interfaces.plans_sha256 != plans.sha256
        or placement_certificate.graph_sha256 != plans.graph_sha256
        or placement_certificate.plans_sha256 != plans.sha256
        or placement_certificate.interfaces_sha256 != interfaces.sha256
    ):
        raise RigidNonmosMacroProjectionError(
            "RIGID_NONMOS_EMISSION_METADATA_MISMATCH"
        )
    if (
        not isinstance(geometry, Mapping)
        or not isinstance(placement_um, Mapping)
        or not isinstance(pins_by_net, Mapping)
    ):
        raise RigidNonmosMacroProjectionError(
            "RIGID_NONMOS_EMISSION_INPUT_INVALID"
    )
    bboxes = dict(placement_certificate.device_bboxes)
    # Member-local artwork authority lives in the compile-input snapshots, while
    # each macro interface binds their aggregate digest.  Recover the per-member
    # digest from the exact macro payload encoded during replay.
    expected_artwork = {}
    expected_pins = []
    for macro in interfaces.macros:
        plan = next(item for item in plans.plans if item.macro_id == macro.macro_id)
        translation = next(
            item.translation_nm
            for item in placement_certificate.bindings
            if item.macro_id == macro.macro_id
        )
        expected_pins.extend(
            (
                terminal.member,
                terminal.role,
                terminal.net,
                terminal.layer,
                (
                    access[0] + translation[0],
                    access[1] + translation[1],
                ),
            )
            for terminal in macro.terminal_accesses
            for access in terminal.accesses_nm
        )
        for member_plan in plan.member_plans:
            # The route stamp is checked below against the placed-device artwork
            # digest carried by the macro interface's aggregate fingerprint.  Keep
            # a canonical member digest alongside the route input certificate.
            expected_artwork[member_plan.member] = _route_stamp_artwork_sha256(
                geometry.get(member_plan.member),
                placement_um.get(member_plan.member),
                bboxes[member_plan.member],
            )
        aggregate = _sha(tuple(
            (
                member,
                next(
                    item.local_bbox_nm
                    for item in plan.member_plans
                    if item.member == member
                ),
                expected_artwork[member],
            )
            for member in plan.members
        ))
        if aggregate != macro.artwork_sha256:
            raise RigidNonmosMacroProjectionError(
                f"RIGID_NONMOS_ROUTE_ARTWORK_MISMATCH:{macro.macro_id}"
            )
    owned = set(expected_artwork)
    observed_pins = []
    for raw_net, raw_pins in pins_by_net.items():
        if type(raw_pins) not in (tuple, list):
            raise RigidNonmosMacroProjectionError(
                "RIGID_NONMOS_ROUTE_PIN_SNAPSHOT_INVALID"
            )
        for pin in raw_pins:
            member = str(getattr(pin, "device", "")).split("@", 1)[0]
            if member not in owned:
                continue
            net = getattr(pin, "net", None)
            terminal = getattr(pin, "terminal", None)
            layer = getattr(pin, "layer", None)
            point = getattr(pin, "xy_nm", None)
            if (
                type(net) is not str
                or net != str(raw_net)
                or type(terminal) is not str
                or type(layer) is not tuple
                or len(layer) != 2
                or any(type(value) is not int for value in layer)
                or type(point) is not tuple
                or len(point) != 2
                or any(type(value) is not int for value in point)
            ):
                raise RigidNonmosMacroProjectionError(
                    "RIGID_NONMOS_ROUTE_PIN_SNAPSHOT_INVALID"
                )
            observed_pins.append((member, terminal, net, layer, point))
    if tuple(sorted(observed_pins)) != tuple(sorted(expected_pins)):
        raise RigidNonmosMacroProjectionError(
            "RIGID_NONMOS_ROUTE_PIN_SNAPSHOT_MISMATCH"
        )
    from clay.layout.route.exact_artifact_evidence import (
        _emitted_device_snapshot_sha256,
    )

    return _seal_rigid_nonmos_emission_certificate(RigidNonmosEmissionCertificate(
        graph_sha256=plans.graph_sha256,
        plans_sha256=plans.sha256,
        compile_inputs_sha256=compile_inputs.sha256,
        interfaces_sha256=interfaces.sha256,
        placement_certificate_sha256=placement_certificate.sha256,
        emitted_device_snapshot_sha256=_emitted_device_snapshot_sha256(
            geometry, placement_um
        ),
        terminal_pin_snapshot_sha256=_sha(tuple(sorted(observed_pins))),
        macro_structure_fingerprints=tuple(sorted(
            (item.macro_id, item.structure_fingerprint)
            for item in interfaces.macros
        )),
        owned_member_artwork=tuple(sorted(expected_artwork.items())),
    ))


def verify_rigid_nonmos_emission_certificate(
    certificate: RigidNonmosEmissionCertificate,
    plans: RigidNonmosMacroPlanSet,
    interfaces: RigidNonmosMacroInterfaceSet,
    placement_certificate: RigidNonmosPlacementCertificate,
    *,
    scope_token: object,
    compile_inputs: RigidNonmosMacroCompileInputs,
    geometry: Mapping[str, object],
    placement_um: Mapping[str, object],
    pins_by_net: Mapping[str, object],
) -> bool:
    """Replay the sealed witness from the publisher's current emitter inputs."""

    if type(certificate) is not RigidNonmosEmissionCertificate:
        raise RigidNonmosMacroProjectionError(
            "RIGID_NONMOS_EMISSION_CERTIFICATE_MISSING"
        )
    certificate.assert_authoritative()
    replayed = build_rigid_nonmos_emission_certificate(
        plans,
        interfaces,
        placement_certificate,
        scope_token=scope_token,
        compile_inputs=compile_inputs,
        geometry=geometry,
        placement_um=placement_um,
        pins_by_net=pins_by_net,
    )
    if replayed.sha256 != certificate.sha256 or replayed.payload() != certificate.payload():
        raise RigidNonmosMacroProjectionError(
            "RIGID_NONMOS_EMISSION_CERTIFICATE_REPLAY_MISMATCH"
        )
    return True


__all__ = [
    "RigidNonmosLocalAlignmentProjection",
    "RigidNonmosLocalSymmetryAxis",
    "RigidNonmosLocalRelationProjection",
    "RigidNonmosEmissionCertificate",
    "RigidNonmosMacroInterface",
    "RigidNonmosMacroInterfaceSet",
    "RigidNonmosMacroMemberPlan",
    "RigidNonmosMacroCompileInputs",
    "RigidNonmosMemberAlignment",
    "RigidNonmosMemberAdjacency",
    "RigidNonmosMemberRelation",
    "RigidNonmosMemberInterfaceSnapshot",
    "RigidNonmosMemberSize",
    "RigidNonmosMacroPlan",
    "RigidNonmosMacroPlanError",
    "RigidNonmosMacroPlanSet",
    "RigidNonmosMacroPlacementBinding",
    "RigidNonmosPlacementCertificate",
    "RigidNonmosMacroProjectionError",
    "build_rigid_nonmos_emission_certificate",
    "compile_rigid_nonmos_macro_interfaces",
    "compile_rigid_nonmos_macro_plans_from_inputs",
    "replay_rigid_nonmos_macro_interfaces",
    "verify_rigid_nonmos_emission_certificate",
]
