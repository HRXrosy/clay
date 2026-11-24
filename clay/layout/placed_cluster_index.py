"""Deterministic ownership index for the final placed geometry stamps.

Logical devices can disappear as individual stamps when an array or matched
cluster materializer fuses them.  This index is the shared translation from
those declaration-level member names to the one final stamp that owns their
physical footprint.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from types import MappingProxyType
from typing import Iterable, Mapping, Optional, Sequence, Tuple

from clay.layout.provenance import DeclarationRef, ProvenanceKind

BBoxNm = Tuple[int, int, int, int]


class PlacedClusterConflict(ValueError):
    """One logical member is claimed by more than one final stamp."""

    def __init__(self, member: str, stamps: Sequence[str]):
        ordered = tuple(sorted(set(str(stamp) for stamp in stamps)))
        self.member = str(member)
        self.stamps = ordered
        super().__init__(
            f"placed member {self.member!r} is ambiguous across stamps "
            f"{', '.join(ordered)}"
        )


@dataclass(frozen=True)
class PlacedCluster:
    stamp: str
    bbox_nm: BBoxNm
    members: Tuple[str, ...]
    source_refs: Tuple[DeclarationRef, ...] = ()


def _integer_bbox(stamp: str, box: Sequence[int]) -> BBoxNm:
    if len(box) != 4 or any(
        isinstance(value, bool) or not isinstance(value, Integral) for value in box
    ):
        raise TypeError(
            f"stamp {stamp!r} bbox must contain four integer nm coordinates"
        )
    x0, y0, x1, y1 = (int(value) for value in box)
    if x1 < x0 or y1 < y0:
        raise ValueError(f"stamp {stamp!r} has inverted bbox {(x0, y0, x1, y1)}")
    return (x0, y0, x1, y1)


def _ordered_refs(refs: Iterable[DeclarationRef]) -> Tuple[DeclarationRef, ...]:
    values = tuple(refs)
    if any(not isinstance(ref, DeclarationRef) for ref in values):
        raise TypeError("source_refs entries must be DeclarationRef values")
    kind_order = {kind: index for index, kind in enumerate(ProvenanceKind)}
    return tuple(
        sorted(set(values), key=lambda ref: (kind_order[ref.kind], ref.identifier))
    )


@dataclass(frozen=True)
class PlacedClusterIndex:
    """Frozen final-stamp inventory plus logical-member ownership."""

    clusters: Tuple[PlacedCluster, ...]
    member_to_stamp: Mapping[str, str]
    _by_stamp: Mapping[str, PlacedCluster] = field(repr=False)

    @classmethod
    def build(
        cls,
        *,
        boxes: Mapping[str, Sequence[int]],
        composite_members: Optional[Mapping[str, Sequence[str]]] = None,
        source_refs: Optional[Mapping[str, Sequence[DeclarationRef]]] = None,
    ) -> "PlacedClusterIndex":
        """Build from absolute integer-nm final stamp boxes.

        Every stamp owns its own name.  A composite additionally owns each
        logical member listed for it.  Input ordering and absolute translation
        do not affect that ownership relation.
        """
        composites = composite_members or {}
        refs_by_stamp = source_refs or {}
        names = tuple(sorted(boxes))
        if any(not isinstance(name, str) or not name for name in names):
            raise TypeError("placed stamp names must be non-empty strings")

        unknown_composites = sorted(set(composites) - set(names))
        if unknown_composites:
            raise ValueError(
                "composite provenance names unplaced stamps: "
                + ", ".join(unknown_composites)
            )
        unknown_refs = sorted(set(refs_by_stamp) - set(names))
        if unknown_refs:
            raise ValueError(
                "source provenance names unplaced stamps: " + ", ".join(unknown_refs)
            )

        clusters = []
        claims: dict[str, list[str]] = {}
        for stamp in names:
            members = {stamp}
            for member in composites.get(stamp, ()):
                if not isinstance(member, str) or not member:
                    raise TypeError("composite member names must be non-empty strings")
                members.add(member)
            ordered_members = tuple(sorted(members))
            cluster = PlacedCluster(
                stamp=stamp,
                bbox_nm=_integer_bbox(stamp, boxes[stamp]),
                members=ordered_members,
                source_refs=_ordered_refs(refs_by_stamp.get(stamp, ())),
            )
            clusters.append(cluster)
            for member in ordered_members:
                claims.setdefault(member, []).append(stamp)

        conflicts = [
            (member, tuple(sorted(stamps)))
            for member, stamps in sorted(claims.items())
            if len(set(stamps)) != 1
        ]
        if conflicts:
            member, stamps = conflicts[0]
            raise PlacedClusterConflict(member, stamps)

        member_to_stamp = {
            member: stamps[0] for member, stamps in sorted(claims.items())
        }
        by_stamp = {cluster.stamp: cluster for cluster in clusters}
        return cls(
            clusters=tuple(clusters),
            member_to_stamp=MappingProxyType(member_to_stamp),
            _by_stamp=MappingProxyType(by_stamp),
        )

    def cluster(self, stamp: str) -> PlacedCluster:
        return self._by_stamp[stamp]

    def stamp_for_member(self, member: str) -> Optional[str]:
        return self.member_to_stamp.get(str(member))

    def union_for_members(self, members: Sequence[str]) -> Optional[BBoxNm]:
        stamps = sorted(
            {
                stamp
                for member in members
                if (stamp := self.stamp_for_member(str(member))) is not None
            }
        )
        if not stamps:
            return None
        boxes = [self._by_stamp[stamp].bbox_nm for stamp in stamps]
        return (
            min(box[0] for box in boxes),
            min(box[1] for box in boxes),
            max(box[2] for box in boxes),
            max(box[3] for box in boxes),
        )


__all__ = [
    "BBoxNm",
    "PlacedCluster",
    "PlacedClusterConflict",
    "PlacedClusterIndex",
]
