"""Deterministic pure IR for shared-diffusion placement rows.

This module is the first, deliberately unintegrated RG5 row-calculus slice.  It
describes shared-OD rows, point-symmetric common-centroid row macros, and explicit
dummy end caps.  It does not emit geometry, call a solver, or choose a circuit-
specific layout.  Existing array/cluster builders remain the rigid materializers;
the contract exposed here is intended for a later final-master integration.

All physical values are exact integer nanometres.  Collection inputs are consumed
with a finite ``cap + 1`` probe, and every semantically unordered collection is
canonicalized before it participates in equality or hashing.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import hashlib
from itertools import islice
import json
from typing import Any, Iterable


BBoxNm = tuple[int, int, int, int]

MAX_ROW_ATOMS = 4096
MAX_MACRO_ROWS = 256
MAX_DECLARATION_REFS = 4096
MAX_DUMMY_ENDCAP_OBLIGATIONS = 4096
MAX_TEXT_CHARS = 4096


class RowCalculusError(ValueError):
    """Base class for invalid row-calculus declarations."""


class RowCalculusLimitError(RowCalculusError):
    """An input exceeded a finite collection or scalar bound."""


class DiffusionCompatibilityError(RowCalculusError):
    """Two adjacent atoms cannot share their touching diffusion boundary."""

    def __init__(
        self,
        *,
        left_atom_id: str,
        right_atom_id: str,
        left_boundary: "DiffusionBoundary",
        right_boundary: "DiffusionBoundary",
        reason: str,
    ) -> None:
        self.left_atom_id = left_atom_id
        self.right_atom_id = right_atom_id
        self.left_net = left_boundary.net
        self.right_net = right_boundary.net
        self.left_terminal = left_boundary.terminal.value
        self.right_terminal = right_boundary.terminal.value
        self.reason = reason
        super().__init__(
            "diffusion conflict at "
            f"{left_atom_id}.right({self.left_terminal}={self.left_net!r}, "
            f"span={left_boundary.share_span_nm}nm) / "
            f"{right_atom_id}.left({self.right_terminal}={self.right_net!r}, "
            f"span={right_boundary.share_span_nm}nm): {reason}"
        )


class CommonCentroidSymmetryError(RowCalculusError):
    """A declared common-centroid row macro lacks strict point symmetry."""


class DummyEndcapError(RowCalculusError):
    """Dummy end-cap declarations and row atoms do not consume each other."""


class ProvenanceClosureError(RowCalculusError):
    """A surviving row request contains a non-live nested declaration ref."""


class BoundaryTerminal(str, Enum):
    SOURCE = "source"
    DRAIN = "drain"


class RowAtomKind(str, Enum):
    ACTIVE = "active"
    DUMMY = "dummy"


class RowStructureKind(str, Enum):
    SHARED_OD = "shared_od"
    COMMON_CENTROID = "common_centroid"
    DISCRETE_MOSAIC = "discrete_mosaic"


class DummyEndcapSide(str, Enum):
    LEFT = "left"
    RIGHT = "right"


def _bounded_tuple(values: Iterable[Any], *, limit: int, field_name: str) -> tuple[Any, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise RowCalculusError(f"{field_name} must be an iterable of items, not text")
    try:
        items = tuple(islice(iter(values), limit + 1))
    except TypeError as exc:
        raise RowCalculusError(f"{field_name} must be iterable") from exc
    if len(items) > limit:
        raise RowCalculusLimitError(
            f"{field_name} exceeds its {limit}-item limit (cap+1 probe)"
        )
    return items


def _text(value: Any, field_name: str) -> str:
    if type(value) is not str:
        raise RowCalculusError(f"{field_name} must be an exact str")
    normalized = value.strip()
    if not normalized:
        raise RowCalculusError(f"{field_name} must be a non-empty exact str")
    if len(normalized) > MAX_TEXT_CHARS:
        raise RowCalculusLimitError(
            f"{field_name} exceeds its {MAX_TEXT_CHARS}-character limit"
        )
    return normalized


def _non_negative_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise RowCalculusError(f"{field_name} must be a non-negative integer")
    return value


def _positive_int(value: Any, field_name: str) -> int:
    value = _non_negative_int(value, field_name)
    if value == 0:
        raise RowCalculusError(f"{field_name} must be positive")
    return value


def _sorted_unique_texts(values: Iterable[Any], field_name: str) -> tuple[str, ...]:
    items = _bounded_tuple(
        values, limit=MAX_DECLARATION_REFS, field_name=field_name
    )
    normalized = tuple(_text(value, field_name) for value in items)
    if len(set(normalized)) != len(normalized):
        raise RowCalculusError(f"{field_name} must not contain duplicates")
    return tuple(sorted(normalized))


def _terminal(value: BoundaryTerminal | str) -> BoundaryTerminal:
    if type(value) is BoundaryTerminal:
        return value
    if type(value) is not str:
        raise RowCalculusError("terminal must be an exact str or BoundaryTerminal")
    aliases = {"s": "source", "d": "drain"}
    stripped = value.strip().lower()
    normalized = aliases.get(stripped, stripped)
    try:
        return BoundaryTerminal(normalized)
    except ValueError as exc:
        raise RowCalculusError(
            "terminal must be 'source'/'s' or 'drain'/'d'"
        ) from exc


def _atom_kind(value: RowAtomKind | str) -> RowAtomKind:
    if type(value) is RowAtomKind:
        return value
    if type(value) is not str:
        raise RowCalculusError("kind must be an exact str or RowAtomKind")
    try:
        return RowAtomKind(value.strip().lower())
    except ValueError as exc:
        raise RowCalculusError("kind must be 'active' or 'dummy'") from exc


def _structure_kind(value: RowStructureKind | str) -> RowStructureKind:
    if type(value) is RowStructureKind:
        return value
    if type(value) is not str:
        raise RowCalculusError(
            "structure_kind must be an exact str or RowStructureKind"
        )
    try:
        return RowStructureKind(value.strip().lower())
    except ValueError as exc:
        raise RowCalculusError(
            "structure_kind must be shared_od, common_centroid, or discrete_mosaic"
        ) from exc


def _dummy_side(value: DummyEndcapSide | str) -> DummyEndcapSide:
    if type(value) is DummyEndcapSide:
        return value
    if type(value) is not str:
        raise DummyEndcapError(
            "dummy end-cap side must be an exact str or DummyEndcapSide"
        )
    try:
        return DummyEndcapSide(value.strip().lower())
    except ValueError as exc:
        raise DummyEndcapError("dummy end-cap side must be 'left' or 'right'") from exc


@dataclass(frozen=True, slots=True, kw_only=True)
class DiffusionBoundary:
    """One electrically named vertical diffusion edge of a row atom."""

    net: str
    terminal: BoundaryTerminal | str
    share_span_nm: int
    declaration_ref: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "net", _text(self.net, "net"))
        object.__setattr__(self, "terminal", _terminal(self.terminal))
        object.__setattr__(
            self,
            "share_span_nm",
            _positive_int(self.share_span_nm, "share_span_nm"),
        )
        object.__setattr__(
            self,
            "declaration_ref",
            _text(self.declaration_ref, "declaration_ref"),
        )

    def is_compatible_with(self, other: "DiffusionBoundary") -> bool:
        """Return whether the two edges can become one physical diffusion."""

        if not isinstance(other, DiffusionBoundary):
            return False
        # D-D, S-S and S-D abutments all occur in legal MOS rows.  Electrical
        # identity and equal physical sharing span, not terminal spelling, decide
        # whether the two polygons may be one diffusion island.
        return self.net == other.net and self.share_span_nm == other.share_span_nm


@dataclass(frozen=True, slots=True, order=True, kw_only=True)
class DummyEndcapObligation:
    """Typed request for an exact number of dummy atoms on one macro side."""

    side: DummyEndcapSide | str
    declaration_ref: str
    count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "side", _dummy_side(self.side))
        object.__setattr__(
            self,
            "declaration_ref",
            _text(self.declaration_ref, "dummy declaration_ref"),
        )
        object.__setattr__(self, "count", _positive_int(self.count, "dummy count"))


@dataclass(frozen=True, slots=True, kw_only=True)
class RowAtom:
    """One ordered, still-unmaterialized unit of a shared-diffusion row."""

    atom_id: str
    sequence_index: int
    width_nm: int
    height_nm: int
    left: DiffusionBoundary
    right: DiffusionBoundary
    match_key: str = ""
    kind: RowAtomKind | str = RowAtomKind.ACTIVE
    declaration_refs: tuple[str, ...] = ()
    dummy_obligation_ref: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "atom_id", _text(self.atom_id, "atom_id"))
        object.__setattr__(
            self,
            "sequence_index",
            _non_negative_int(self.sequence_index, "sequence_index"),
        )
        object.__setattr__(self, "width_nm", _positive_int(self.width_nm, "width_nm"))
        object.__setattr__(
            self, "height_nm", _positive_int(self.height_nm, "height_nm")
        )
        if not isinstance(self.left, DiffusionBoundary):
            raise RowCalculusError("left must be a DiffusionBoundary")
        if not isinstance(self.right, DiffusionBoundary):
            raise RowCalculusError("right must be a DiffusionBoundary")
        if self.left.share_span_nm >= self.width_nm:
            raise RowCalculusError("left share_span_nm must be smaller than width_nm")
        if self.right.share_span_nm >= self.width_nm:
            raise RowCalculusError("right share_span_nm must be smaller than width_nm")
        if type(self.match_key) is not str:
            raise RowCalculusError("match_key must be an exact str")
        match_key = self.match_key.strip()
        if len(match_key) > MAX_TEXT_CHARS:
            raise RowCalculusLimitError(
                f"match_key exceeds its {MAX_TEXT_CHARS}-character limit"
            )
        object.__setattr__(self, "match_key", match_key)
        kind = _atom_kind(self.kind)
        object.__setattr__(self, "kind", kind)
        refs = _sorted_unique_texts(self.declaration_refs, "declaration_refs")
        if not refs:
            raise ProvenanceClosureError(
                f"row atom {self.atom_id!r} requires declaration provenance"
            )
        object.__setattr__(
            self,
            "declaration_refs",
            refs,
        )
        dummy_ref = self.dummy_obligation_ref
        if kind is RowAtomKind.ACTIVE:
            if dummy_ref is not None:
                raise DummyEndcapError(
                    f"active atom {self.atom_id!r} cannot consume a dummy obligation"
                )
        else:
            if dummy_ref is None:
                raise DummyEndcapError(
                    f"dummy atom {self.atom_id!r} requires dummy_obligation_ref"
                )
            dummy_ref = _text(dummy_ref, "dummy_obligation_ref")
            if dummy_ref not in refs:
                raise DummyEndcapError(
                    f"dummy atom {self.atom_id!r} obligation ref must be included "
                    "in declaration_refs"
                )
        object.__setattr__(self, "dummy_obligation_ref", dummy_ref)

    @property
    def bbox_nm(self) -> BBoxNm:
        return (0, 0, self.width_nm, self.height_nm)


@dataclass(frozen=True, slots=True, kw_only=True)
class SharedDiffusionRequest:
    """Unrealized request to abut two compatible same-net diffusion edges."""

    left_atom_id: str
    right_atom_id: str
    net: str
    share_span_nm: int
    declaration_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "left_atom_id", _text(self.left_atom_id, "left_atom_id")
        )
        object.__setattr__(
            self, "right_atom_id", _text(self.right_atom_id, "right_atom_id")
        )
        object.__setattr__(self, "net", _text(self.net, "net"))
        object.__setattr__(
            self,
            "share_span_nm",
            _positive_int(self.share_span_nm, "share_span_nm"),
        )
        object.__setattr__(
            self,
            "declaration_refs",
            _sorted_unique_texts(self.declaration_refs, "declaration_refs"),
        )


def _shared_diffusion_request(
    left: RowAtom, right: RowAtom
) -> SharedDiffusionRequest:
    lb = left.right
    rb = right.left
    if lb.net != rb.net:
        raise DiffusionCompatibilityError(
            left_atom_id=left.atom_id,
            right_atom_id=right.atom_id,
            left_boundary=lb,
            right_boundary=rb,
            reason="touching diffusion nets are distinct",
        )
    if lb.share_span_nm != rb.share_span_nm:
        raise DiffusionCompatibilityError(
            left_atom_id=left.atom_id,
            right_atom_id=right.atom_id,
            left_boundary=lb,
            right_boundary=rb,
            reason="touching diffusion share spans differ",
        )
    return SharedDiffusionRequest(
        left_atom_id=left.atom_id,
        right_atom_id=right.atom_id,
        net=lb.net,
        share_span_nm=lb.share_span_nm,
        declaration_refs=tuple(sorted({lb.declaration_ref, rb.declaration_ref})),
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class RowString:
    """An ordered row requesting compatible same-net diffusion abutments."""

    row_id: str
    row_index: int
    atoms: tuple[RowAtom, ...]
    declaration_refs: tuple[str, ...] = ()
    inter_atom_gap_nm: int | None = None
    boundary_joins: tuple | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "row_id", _text(self.row_id, "row_id"))
        object.__setattr__(
            self, "row_index", _non_negative_int(self.row_index, "row_index")
        )
        atoms = _bounded_tuple(
            self.atoms, limit=MAX_ROW_ATOMS, field_name="atoms"
        )
        if not atoms:
            raise RowCalculusError("atoms must not be empty")
        if any(not isinstance(atom, RowAtom) for atom in atoms):
            raise RowCalculusError("atoms must contain only RowAtom values")
        atoms = tuple(sorted(atoms, key=lambda atom: atom.sequence_index))
        indices = tuple(atom.sequence_index for atom in atoms)
        if len(set(indices)) != len(indices):
            raise RowCalculusError("atom sequence_index values must be unique in a row")
        expected = tuple(range(len(atoms)))
        if indices != expected:
            raise RowCalculusError(
                f"atom sequence_index values must be contiguous from zero; got {indices}"
            )
        atom_ids = tuple(atom.atom_id for atom in atoms)
        if len(set(atom_ids)) != len(atom_ids):
            raise RowCalculusError("atom_id values must be unique in a row")
        gap = self.inter_atom_gap_nm
        if self.boundary_joins is not None:
            from .row_boundary_joins import checked_boundary_joins
            if gap is not None:
                raise RowCalculusError('explicit boundary joins and legacy gap modes are mutually exclusive')
            checked_boundary_joins(self.boundary_joins, atom_ids, self.declaration_refs)
        elif gap is not None:
            gap = _non_negative_int(gap, "inter_atom_gap_nm")
        else:
            for left, right in zip(atoms, atoms[1:]):
                _shared_diffusion_request(left, right)
        object.__setattr__(self, "atoms", atoms)
        object.__setattr__(self, "inter_atom_gap_nm", gap)
        object.__setattr__(
            self,
            "declaration_refs",
            _sorted_unique_texts(self.declaration_refs, "declaration_refs"),
        )

    @property
    def sequence(self) -> tuple[str, ...]:
        return tuple(atom.atom_id for atom in self.atoms)

    @property
    def shared_diffusion_requests(self) -> tuple[SharedDiffusionRequest, ...]:
        if self.inter_atom_gap_nm is not None or self.boundary_joins is not None:
            return ()
        return tuple(
            _shared_diffusion_request(left, right)
            for left, right in zip(self.atoms, self.atoms[1:])
        )

    @property
    def requested_atom_bboxes_nm(self) -> tuple[tuple[str, BBoxNm], ...]:
        placed: list[tuple[str, BBoxNm]] = []
        shared_requests = self.shared_diffusion_requests
        x = 0
        for index, atom in enumerate(self.atoms):
            placed.append((atom.atom_id, (x, 0, x + atom.width_nm, atom.height_nm)))
            if index < len(self.atoms) - 1:
                if self.boundary_joins is not None:
                    x += atom.width_nm + self.boundary_joins[index].gap_nm
                elif self.inter_atom_gap_nm is None:
                    x += atom.width_nm - shared_requests[index].share_span_nm
                else:
                    x += atom.width_nm + self.inter_atom_gap_nm
        return tuple(placed)

    @property
    def bbox_nm(self) -> BBoxNm:
        """Requested row-local envelope; it is not emitted geometry evidence."""

        placed = self.requested_atom_bboxes_nm
        return (
            0,
            0,
            max(box[2] for _atom_id, box in placed),
            max(box[3] for _atom_id, box in placed),
        )

    def concatenate(self, other: "RowString", *, row_id: str | None = None) -> "RowString":
        """Return one row, refusing a named conflict at the new join."""

        if not isinstance(other, RowString):
            raise RowCalculusError("other must be a RowString")
        if self.boundary_joins is not None or other.boundary_joins is not None:
            raise RowCalculusError('concatenating explicit boundary joins requires a new edge declaration')
        if self.row_index != other.row_index:
            raise RowCalculusError(
                "only RowString values with the same row_index can concatenate"
            )
        if self.inter_atom_gap_nm != other.inter_atom_gap_nm:
            raise RowCalculusError(
                "concatenated rows must use the same inter-atom join contract"
            )
        combined = tuple(self.atoms) + tuple(other.atoms)
        renumbered = tuple(
            replace(atom, sequence_index=index) for index, atom in enumerate(combined)
        )
        return RowString(
            row_id=row_id or f"{self.row_id}+{other.row_id}",
            row_index=self.row_index,
            atoms=renumbered,
            declaration_refs=tuple(
                sorted(set(self.declaration_refs) | set(other.declaration_refs))
            ),
            inter_atom_gap_nm=self.inter_atom_gap_nm,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class RowMacro:
    """A finite collection of solver-visible rows and their materializer contract."""

    macro_id: str
    structure_kind: RowStructureKind | str
    rows: tuple[RowString, ...]
    row_pitch_nm: int
    declaration_refs: tuple[str, ...]
    dummy_endcap_obligations: tuple[DummyEndcapObligation, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "macro_id", _text(self.macro_id, "macro_id"))
        object.__setattr__(
            self, "structure_kind", _structure_kind(self.structure_kind)
        )
        rows = _bounded_tuple(
            self.rows, limit=MAX_MACRO_ROWS, field_name="rows"
        )
        if not rows:
            raise RowCalculusError("rows must not be empty")
        if any(not isinstance(row, RowString) for row in rows):
            raise RowCalculusError("rows must contain only RowString values")
        rows = tuple(sorted(rows, key=lambda row: row.row_index))
        indices = tuple(row.row_index for row in rows)
        if len(set(indices)) != len(indices):
            raise RowCalculusError("row_index values must be unique in a macro")
        if indices != tuple(range(len(rows))):
            raise RowCalculusError(
                f"row_index values must be contiguous from zero; got {indices}"
            )
        row_ids = tuple(row.row_id for row in rows)
        if len(set(row_ids)) != len(row_ids):
            raise RowCalculusError("row_id values must be unique in a macro")
        pitch = _non_negative_int(self.row_pitch_nm, "row_pitch_nm")
        if len(rows) > 1 and pitch == 0:
            raise RowCalculusError("row_pitch_nm must be positive for multiple rows")
        for left_row, right_row in zip(rows, rows[1:]):
            required_pitch = max(left_row.bbox_nm[3], right_row.bbox_nm[3])
            if pitch < required_pitch:
                raise RowCalculusError(
                    f"row_pitch_nm {pitch} is smaller than adjacent row height "
                    f"{required_pitch} for {left_row.row_id!r}/{right_row.row_id!r}"
                )
        refs = _sorted_unique_texts(self.declaration_refs, "declaration_refs")
        if not refs:
            raise RowCalculusError(
                "a RowMacro requires at least one live structural declaration"
            )
        obligations = _bounded_tuple(
            self.dummy_endcap_obligations,
            limit=MAX_DUMMY_ENDCAP_OBLIGATIONS,
            field_name="dummy_endcap_obligations",
        )
        if any(not isinstance(item, DummyEndcapObligation) for item in obligations):
            raise DummyEndcapError(
                "dummy_endcap_obligations must contain DummyEndcapObligation values"
            )
        obligations = tuple(sorted(obligations))
        obligation_refs = tuple(item.declaration_ref for item in obligations)
        if len(obligation_refs) != len(set(obligation_refs)):
            raise DummyEndcapError(
                "one declaration_ref cannot name both or multiple dummy obligations"
            )
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "row_pitch_nm", pitch)
        object.__setattr__(self, "declaration_refs", refs)
        object.__setattr__(self, "dummy_endcap_obligations", obligations)
        self._validate_provenance_closure()
        self._validate_dummy_endcaps()
        if (any(row.boundary_joins is not None for row in rows)
                and self.structure_kind is not RowStructureKind.DISCRETE_MOSAIC):
            raise RowCalculusError('explicit boundary joins require a discrete mosaic')
        if self.structure_kind is RowStructureKind.COMMON_CENTROID:
            self._validate_common_centroid()

    @property
    def sequence(self) -> tuple[tuple[str, ...], ...]:
        return tuple(row.sequence for row in self.rows)

    @property
    def bbox_nm(self) -> BBoxNm:
        """Requested macro-local envelope; it is not a realized placement."""

        width = max(row.bbox_nm[2] for row in self.rows)
        height = max(
            row.row_index * self.row_pitch_nm + row.bbox_nm[3]
            for row in self.rows
        )
        return (0, 0, width, height)

    @property
    def requested_atom_bboxes_nm(
        self,
    ) -> tuple[tuple[str, str, BBoxNm], ...]:
        placed: list[tuple[str, str, BBoxNm]] = []
        for row in self.rows:
            dy = row.row_index * self.row_pitch_nm
            for atom_id, (x0, y0, x1, y1) in row.requested_atom_bboxes_nm:
                placed.append((row.row_id, atom_id, (x0, y0 + dy, x1, y1 + dy)))
        return tuple(placed)

    @property
    def consumed_dummy_endcaps(self) -> tuple[tuple[str, str, int], ...]:
        counts = {item.declaration_ref: 0 for item in self.dummy_endcap_obligations}
        for row in self.rows:
            for atom in row.atoms:
                if atom.kind is RowAtomKind.DUMMY and atom.dummy_obligation_ref in counts:
                    counts[atom.dummy_obligation_ref] += 1
        return tuple(
            (item.side.value, item.declaration_ref, counts[item.declaration_ref])
            for item in self.dummy_endcap_obligations
        )

    def _validate_provenance_closure(self) -> None:
        live = set(self.declaration_refs)
        nested: list[tuple[str, str]] = []
        for row in self.rows:
            nested.extend((f"row {row.row_id}", ref) for ref in row.declaration_refs)
            for atom in row.atoms:
                nested.extend((f"atom {atom.atom_id}", ref) for ref in atom.declaration_refs)
                nested.append((f"atom {atom.atom_id} left boundary", atom.left.declaration_ref))
                nested.append((f"atom {atom.atom_id} right boundary", atom.right.declaration_ref))
                if atom.dummy_obligation_ref is not None:
                    nested.append((f"atom {atom.atom_id} dummy obligation", atom.dummy_obligation_ref))
        nested.extend(
            (f"dummy obligation {item.side.value}", item.declaration_ref)
            for item in self.dummy_endcap_obligations
        )
        for owner, ref in nested:
            if ref not in live:
                raise ProvenanceClosureError(
                    f"non-live declaration {ref!r} survives in {owner}"
                )

    def _validate_dummy_endcaps(self) -> None:
        obligations = {
            item.declaration_ref: item for item in self.dummy_endcap_obligations
        }
        dummy_atoms = [
            atom for row in self.rows for atom in row.atoms
            if atom.kind is RowAtomKind.DUMMY
        ]
        if not dummy_atoms and not obligations:
            return
        sides = {item.side for item in obligations.values()}
        if sides != {DummyEndcapSide.LEFT, DummyEndcapSide.RIGHT}:
            raise DummyEndcapError(
                "dummy end caps require both left and right typed obligations"
            )
        obligation_ref_set = set(obligations)
        for row in self.rows:
            phase = DummyEndcapSide.LEFT
            saw_active = False
            for atom in row.atoms:
                if atom.kind is RowAtomKind.ACTIVE:
                    if phase is DummyEndcapSide.RIGHT:
                        raise DummyEndcapError(
                            f"active atom {atom.atom_id!r} follows the right dummy suffix"
                        )
                    saw_active = True
                    continue
                matching_refs = obligation_ref_set.intersection(atom.declaration_refs)
                if len(matching_refs) > 1:
                    raise DummyEndcapError(
                        f"dummy atom {atom.atom_id!r} names multiple dummy obligations"
                    )
                obligation = obligations.get(atom.dummy_obligation_ref or "")
                if obligation is None or matching_refs != {obligation.declaration_ref}:
                    raise DummyEndcapError(
                        f"dummy atom {atom.atom_id!r} lacks one exact typed obligation"
                    )
                if obligation.side is DummyEndcapSide.LEFT:
                    if saw_active or phase is DummyEndcapSide.RIGHT:
                        raise DummyEndcapError(
                            f"left dummy atom {atom.atom_id!r} is outside the prefix"
                        )
                else:
                    if not saw_active:
                        raise DummyEndcapError(
                            f"right dummy atom {atom.atom_id!r} precedes the active row body"
                        )
                    phase = DummyEndcapSide.RIGHT
            if not saw_active:
                raise DummyEndcapError(f"row {row.row_id!r} has no active body")
        actual = {ref: count for _side, ref, count in self.consumed_dummy_endcaps}
        mismatches = tuple(
            (item.side.value, item.declaration_ref, item.count, actual[item.declaration_ref])
            for item in self.dummy_endcap_obligations
            if actual[item.declaration_ref] != item.count
        )
        if mismatches:
            raise DummyEndcapError(
                f"dummy end-cap obligation counts are not consumed exactly: {mismatches}"
            )

    def _validate_common_centroid(self) -> None:
        row_lengths = {len(row.atoms) for row in self.rows}
        if len(row_lengths) != 1:
            raise CommonCentroidSymmetryError(
                "common-centroid rows must have equal atom counts"
            )
        row_widths = {row.bbox_nm[2] for row in self.rows}
        if len(row_widths) != 1:
            raise CommonCentroidSymmetryError(
                "common-centroid rows must have equal widths"
            )
        macro_width = self.bbox_nm[2]
        macro_height = self.bbox_nm[3]
        placements = {
            (row_id, atom_id): bbox
            for row_id, atom_id, bbox in self.requested_atom_bboxes_nm
        }
        row_count = len(self.rows)
        column_count = len(self.rows[0].atoms)
        dummy_obligation_refs = {
            item.side.value: item.declaration_ref
            for item in self.dummy_endcap_obligations
        }
        has_unique_flank_obligations = tuple(
            sorted(item.side.value for item in self.dummy_endcap_obligations)
        ) == ("left", "right")
        for row_pos, row in enumerate(self.rows):
            opposite_row = self.rows[row_count - 1 - row_pos]
            for column_pos, atom in enumerate(row.atoms):
                opposite = opposite_row.atoms[column_count - 1 - column_pos]
                if not atom.match_key or atom.match_key != opposite.match_key:
                    raise CommonCentroidSymmetryError(
                        "common-centroid match-key mismatch between "
                        f"{atom.atom_id!r} and {opposite.atom_id!r}"
                    )
                if atom.kind is not opposite.kind:
                    raise CommonCentroidSymmetryError(
                        "common-centroid atom-kind mismatch between "
                        f"{atom.atom_id!r} and {opposite.atom_id!r}"
                    )
                if (atom.width_nm, atom.height_nm) != (
                    opposite.width_nm,
                    opposite.height_nm,
                ):
                    raise CommonCentroidSymmetryError(
                        "common-centroid dimensions mismatch between "
                        f"{atom.atom_id!r} and {opposite.atom_id!r}"
                    )
                allow_distinct_dummy_drains = (
                    row_count == 1
                    and has_unique_flank_obligations
                    and atom.kind is RowAtomKind.DUMMY
                    and opposite.kind is RowAtomKind.DUMMY
                    and atom.match_key.startswith("cc:dummy-flank-pair:")
                    and {
                        atom.dummy_obligation_ref,
                        opposite.dummy_obligation_ref,
                    }
                    == {
                        dummy_obligation_refs.get("left"),
                        dummy_obligation_refs.get("right"),
                    }
                )
                if not _reflected_boundaries_match(
                    atom,
                    opposite,
                    allow_distinct_dummy_drains=allow_distinct_dummy_drains,
                ):
                    raise CommonCentroidSymmetryError(
                        "common-centroid boundary mismatch between "
                        f"{atom.atom_id!r} and {opposite.atom_id!r}"
                    )
                box = placements[(row.row_id, atom.atom_id)]
                opposite_box = placements[(opposite_row.row_id, opposite.atom_id)]
                if not _boxes_are_point_reflections(
                    box, opposite_box, macro_width, macro_height
                ):
                    raise CommonCentroidSymmetryError(
                        "common-centroid bbox mismatch between "
                        f"{atom.atom_id!r} and {opposite.atom_id!r}"
                    )

    @property
    def final_master_contract(self) -> dict[str, Any]:
        """Stable, unrealized request payload for later final-master integration."""

        row_contracts: list[dict[str, Any]] = []
        for row in self.rows:
            atom_bboxes = dict(row.requested_atom_bboxes_nm)
            row_contract = {
                    "row_id": row.row_id,
                    "row_index": row.row_index,
                    "requested_local_envelope_nm": row.bbox_nm,
                    "sequence": row.sequence,
                    "atoms": tuple(
                        {
                            "atom_id": atom.atom_id,
                            "sequence_index": atom.sequence_index,
                            "kind": atom.kind.value,
                            "match_key": atom.match_key,
                            "requested_local_bbox_nm": atom_bboxes[atom.atom_id],
                            "left": _boundary_payload(atom.left),
                            "right": _boundary_payload(atom.right),
                            "declaration_refs": atom.declaration_refs,
                            "dummy_obligation_ref": atom.dummy_obligation_ref,
                        }
                        for atom in row.atoms
                    ),
                    "shared_diffusion_requests": tuple(
                        {
                            "left_atom_id": shared.left_atom_id,
                            "right_atom_id": shared.right_atom_id,
                            "net": shared.net,
                            "share_span_nm": shared.share_span_nm,
                            "declaration_refs": shared.declaration_refs,
                        }
                        for shared in row.shared_diffusion_requests
                    ),
                }
            if row.inter_atom_gap_nm is not None:
                row_contract["inter_atom_gap_nm"] = row.inter_atom_gap_nm
            if row.boundary_joins is not None:
                row_contract["boundary_joins"] = tuple(item.payload() for item in row.boundary_joins)
            row_contracts.append(row_contract)
        return {
            "schema": ("clay.row-macro-request.v2" if any(row.boundary_joins is not None for row in self.rows)
                       else "clay.row-macro-request.v1"),
            "realized": False,
            "macro_id": self.macro_id,
            "structure_kind": self.structure_kind.value,
            "requested_local_envelope_nm": self.bbox_nm,
            "declaration_refs": self.declaration_refs,
            "dummy_endcap_obligations": tuple(
                {
                    "side": item.side.value,
                    "declaration_ref": item.declaration_ref,
                    "count": item.count,
                }
                for item in self.dummy_endcap_obligations
            ),
            "rows": tuple(row_contracts),
        }

    def to_canonical_bytes(self) -> bytes:
        return json.dumps(
            self.final_master_contract,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_canonical_bytes()).hexdigest()


def _reflected_boundaries_match(
    left: RowAtom,
    right: RowAtom,
    *,
    allow_distinct_dummy_drains: bool = False,
) -> bool:
    pairs = ((left.left, right.right), (left.right, right.left))
    paired_dummy_drains = (
        allow_distinct_dummy_drains
        and left.kind is RowAtomKind.DUMMY
        and right.kind is RowAtomKind.DUMMY
        and left.match_key.startswith("cc:dummy-flank-pair:")
        and left.match_key == right.match_key
    )
    return all(
        (
            a.net == b.net
            or (
                paired_dummy_drains
                and a.terminal is BoundaryTerminal.DRAIN
                and b.terminal is BoundaryTerminal.DRAIN
            )
        )
        and a.terminal is b.terminal
        and a.share_span_nm == b.share_span_nm
        for a, b in pairs
    )


def _boxes_are_point_reflections(
    left: BBoxNm,
    right: BBoxNm,
    macro_width: int,
    macro_height: int,
) -> bool:
    return (
        left[0] + right[2] == macro_width
        and left[2] + right[0] == macro_width
        and left[1] + right[3] == macro_height
        and left[3] + right[1] == macro_height
    )


def _boundary_payload(boundary: DiffusionBoundary) -> dict[str, Any]:
    return {
        "net": boundary.net,
        "terminal": boundary.terminal.value,
        "share_span_nm": boundary.share_span_nm,
        "declaration_ref": boundary.declaration_ref,
    }


def _rebuild_boundary_exact(
    boundary: object, *, field_name: str
) -> DiffusionBoundary:
    if type(boundary) is not DiffusionBoundary:
        raise RowCalculusError(f"{field_name} must be exact DiffusionBoundary")
    return DiffusionBoundary(
        net=boundary.net,
        terminal=boundary.terminal,
        share_span_nm=boundary.share_span_nm,
        declaration_ref=boundary.declaration_ref,
    )


def _rebuild_atom_exact(atom: object) -> RowAtom:
    if type(atom) is not RowAtom:
        raise RowCalculusError("row atoms must be exact RowAtom values")
    refs = _bounded_tuple(
        atom.declaration_refs,
        limit=MAX_DECLARATION_REFS,
        field_name="atom declaration_refs",
    )
    return RowAtom(
        atom_id=atom.atom_id,
        sequence_index=atom.sequence_index,
        width_nm=atom.width_nm,
        height_nm=atom.height_nm,
        left=_rebuild_boundary_exact(atom.left, field_name="atom left boundary"),
        right=_rebuild_boundary_exact(atom.right, field_name="atom right boundary"),
        match_key=atom.match_key,
        kind=atom.kind,
        declaration_refs=refs,
        dummy_obligation_ref=atom.dummy_obligation_ref,
    )


def _rebuild_row_exact(row: object) -> RowString:
    if type(row) is not RowString:
        raise RowCalculusError("rows must contain exact RowString values")
    raw_atoms = _bounded_tuple(
        row.atoms,
        limit=MAX_ROW_ATOMS,
        field_name="row atoms",
    )
    refs = _bounded_tuple(
        row.declaration_refs,
        limit=MAX_DECLARATION_REFS,
        field_name="row declaration_refs",
    )
    return RowString(
        row_id=row.row_id,
        row_index=row.row_index,
        atoms=tuple(_rebuild_atom_exact(atom) for atom in raw_atoms),
        declaration_refs=refs,
        inter_atom_gap_nm=row.inter_atom_gap_nm,
        boundary_joins=row.boundary_joins,
    )


def _rebuild_dummy_obligation_exact(
    obligation: object,
) -> DummyEndcapObligation:
    if type(obligation) is not DummyEndcapObligation:
        raise DummyEndcapError(
            "dummy obligations must be exact DummyEndcapObligation values"
        )
    return DummyEndcapObligation(
        side=obligation.side,
        declaration_ref=obligation.declaration_ref,
        count=obligation.count,
    )


def compile_row_macro(
    *,
    macro_id: str,
    structure_kind: RowStructureKind | str,
    rows: Iterable[RowString],
    row_pitch_nm: int,
    declaration_refs: Iterable[str],
    dummy_endcap_obligations: Iterable[DummyEndcapObligation] = (),
) -> RowMacro | None:
    """Compile live row declarations, or return ``None`` after their deletion.

    Explicit row/atom indices carry physical order, so callers may present rows and
    atom declarations in any iteration order without changing the canonical bytes.
    ``declaration_refs`` is the complete live-ref set.  Atoms with no live refs are
    deleted; a partially live atom or any surviving nested non-live ref is rejected
    because silently trimming a jointly-owned atom would invent semantics.  Empty
    live refs, or filtering every atom, returns ``None`` without residual structure.
    """

    bounded_rows = _bounded_tuple(rows, limit=MAX_MACRO_ROWS, field_name="rows")
    rebuilt_rows = tuple(_rebuild_row_exact(row) for row in bounded_rows)
    refs = _sorted_unique_texts(declaration_refs, "declaration_refs")
    raw_obligations = _bounded_tuple(
        dummy_endcap_obligations,
        limit=MAX_DUMMY_ENDCAP_OBLIGATIONS,
        field_name="dummy_endcap_obligations",
    )
    obligations = tuple(
        _rebuild_dummy_obligation_exact(item) for item in raw_obligations
    )
    if not refs:
        return None
    live = set(refs)
    surviving_rows: list[RowString] = []
    for row in sorted(rebuilt_rows, key=lambda item: item.row_index):
        surviving_atoms: list[RowAtom] = []
        for atom in row.atoms:
            atom_refs = set(atom.declaration_refs)
            overlap = atom_refs & live
            if not overlap:
                continue
            dead_atom_refs = atom_refs - live
            if dead_atom_refs:
                raise ProvenanceClosureError(
                    f"atom {atom.atom_id!r} partially survives with non-live "
                    f"declarations {tuple(sorted(dead_atom_refs))}"
                )
            nested_refs = {
                atom.left.declaration_ref,
                atom.right.declaration_ref,
            }
            if atom.dummy_obligation_ref is not None:
                nested_refs.add(atom.dummy_obligation_ref)
            dead_nested_refs = nested_refs - live
            if dead_nested_refs:
                raise ProvenanceClosureError(
                    f"atom {atom.atom_id!r} retains non-live nested declaration "
                    f"{tuple(sorted(dead_nested_refs))}"
                )
            surviving_atoms.append(
                replace(atom, sequence_index=len(surviving_atoms))
            )
        if not surviving_atoms:
            continue
        if row.boundary_joins is not None and len(surviving_atoms) != len(row.atoms):
            raise RowCalculusError('partial atom deletion requires new explicit boundary join declarations')
        surviving_rows.append(RowString(
            row_id=row.row_id,
            row_index=len(surviving_rows),
            atoms=tuple(surviving_atoms),
            declaration_refs=tuple(
                sorted(set(row.declaration_refs).intersection(live))
            ),
            inter_atom_gap_nm=row.inter_atom_gap_nm,
            boundary_joins=row.boundary_joins,
        ))
    if not surviving_rows:
        return None
    surviving_obligations = tuple(
        item for item in obligations if item.declaration_ref in live
    )
    return RowMacro(
        macro_id=macro_id,
        structure_kind=structure_kind,
        rows=tuple(surviving_rows),
        row_pitch_nm=row_pitch_nm,
        declaration_refs=refs,
        dummy_endcap_obligations=surviving_obligations,
    )


__all__ = [
    "BBoxNm",
    "BoundaryTerminal",
    "CommonCentroidSymmetryError",
    "DiffusionBoundary",
    "DiffusionCompatibilityError",
    "DummyEndcapObligation",
    "DummyEndcapSide",
    "DummyEndcapError",
    "MAX_DECLARATION_REFS",
    "MAX_DUMMY_ENDCAP_OBLIGATIONS",
    "MAX_MACRO_ROWS",
    "MAX_ROW_ATOMS",
    "ProvenanceClosureError",
    "RowAtom",
    "RowAtomKind",
    "RowCalculusError",
    "RowCalculusLimitError",
    "RowMacro",
    "RowString",
    "RowStructureKind",
    "SharedDiffusionRequest",
    "compile_row_macro",
]
