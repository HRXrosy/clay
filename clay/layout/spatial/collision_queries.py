"""Immutable rectangle index for exact clearance queries at one placement.

An enclosing node can prove separation, while a possible conflict descends to
the original rectangles. Empty space inside an envelope is never occupied.
"""
from dataclasses import dataclass

from clay._native import cpp_pnr_disabled, load_native

Box = tuple[int, int, int, int]


def clearance(left: Box, right: Box) -> int:
    return max(right[0]-left[2], left[0]-right[2],
               right[1]-left[3], left[1]-right[3])


@dataclass(frozen=True, slots=True)
class _Node:
    bbox: Box
    boxes: tuple[Box, ...]
    children: tuple['_Node', ...]

    def conflicts(self, box: Box, gap: int, poll=None) -> bool:
        if poll:
            poll()
        if clearance(self.bbox, box) >= gap:
            return False
        if self.children:
            return any(child.conflicts(box, gap, poll) for child in self.children)
        return any(clearance(candidate, box) < gap for candidate in self.boxes)

    def intersecting(self, box: Box, *, include_touching: bool):
        distance = clearance(self.bbox, box)
        if distance > 0 or (distance == 0 and not include_touching):
            return
        if self.children:
            for child in self.children:
                yield from child.intersecting(box, include_touching=include_touching)
        else:
            for candidate in self.boxes:
                distance = clearance(candidate, box)
                if distance < 0 or (include_touching and distance == 0):
                    yield candidate


def _build(boxes: tuple[Box, ...], checkpoint=None) -> _Node:
    if checkpoint:
        checkpoint()
    x0, y0, x1, y1 = boxes[0]
    for index, box in enumerate(boxes):
        if checkpoint and index % 256 == 0:
            checkpoint()
        x0, y0 = min(x0, box[0]), min(y0, box[1])
        x1, y1 = max(x1, box[2]), max(y1, box[3])
    bbox = x0, y0, x1, y1
    if len(boxes) <= 8:
        return _Node(bbox, boxes, ())
    axis = 0 if bbox[2]-bbox[0] >= bbox[3]-bbox[1] else 1
    keyed = 0
    def sort_key(box):
        nonlocal keyed
        keyed += 1
        if checkpoint and keyed % 256 == 0:
            checkpoint()
        return box[axis]+box[axis+2], box
    ordered = sorted(boxes, key=sort_key)
    if checkpoint:
        checkpoint()
    middle = len(ordered) // 2
    return _Node(bbox, (), (_build(tuple(ordered[:middle]), checkpoint),
                            _build(tuple(ordered[middle:]), checkpoint)))


_ORIG_CLEARANCE = clearance
_ORIG_BUILD = _build


@dataclass(frozen=True, slots=True, init=False)
class RectangleIndex:
    root: _Node | None
    _native: object | None
    _rows: tuple[Box, ...]

    def __init__(self, boxes: tuple[Box, ...], *, checkpoint=None):
        native_mod = load_native("_place_cpp")
        if native_mod is not None and _build is _ORIG_BUILD:
            if checkpoint and type(boxes) not in (tuple, list):
                checkpoint()
            rows = boxes if type(boxes) is tuple else tuple(boxes)
            try:
                native = native_mod.NativeRectangleIndex(rows, checkpoint)
            except OverflowError:
                pass
            else:
                object.__setattr__(self, 'root', None)
                object.__setattr__(self, '_native', native)
                object.__setattr__(self, '_rows', rows)
                return
        if checkpoint:
            checkpoint()
        rows = tuple(boxes)
        for index, row in enumerate(rows):
            if checkpoint and index % 256 == 0:
                checkpoint()
            if len(row) != 4 or any(type(v) is not int for v in row):
                raise TypeError('collision index requires four exact integers')
            if row[0] >= row[2] or row[1] >= row[3]:
                raise ValueError('collision index requires positive rectangles')
        object.__setattr__(self, 'root', _build(rows, checkpoint) if rows else None)
        object.__setattr__(self, '_native', None)
        object.__setattr__(self, '_rows', rows)
        if checkpoint:
            checkpoint()

    def _ensure_py_root(self) -> _Node | None:
        root = self.root
        if root is None and self._rows:
            root = _build(self._rows)
            object.__setattr__(self, 'root', root)
        return root

    def conflicts(self, boxes: tuple[Box, ...], *, clearance_nm: int, checkpoint=None) -> bool:
        # The callback belongs to this operation, never to the immutable index.
        if (
            self._native is not None
            and clearance is _ORIG_CLEARANCE
            and not cpp_pnr_disabled()
        ):
            rows_q = boxes
            if type(rows_q) not in (tuple, list):
                if checkpoint:
                    checkpoint()
                rows_q = tuple(rows_q)
            try:
                return self._native.conflicts(rows_q, clearance_nm, checkpoint)
            except (OverflowError, TypeError):
                boxes = rows_q
        if checkpoint:
            checkpoint()
        root = self._ensure_py_root()
        if root is None:
            return False
        visited = 0
        def poll():
            nonlocal visited
            visited += 1
            if visited % 256 == 0:
                checkpoint()
        result = any(root.conflicts(box, clearance_nm, poll if checkpoint else None)
                     for box in boxes)
        if checkpoint:
            checkpoint()
        return result

    def intersecting(self, box: Box, *, include_touching: bool = False) -> tuple[Box, ...]:
        """Return candidate rectangles; callers still own exact polygon queries.

        Traversal order is spatial, not input order. Repeated rectangles remain
        repeated, and edge/corner contacts are included only when requested.
        """
        if (
            self._native is not None
            and clearance is _ORIG_CLEARANCE
            and not cpp_pnr_disabled()
        ):
            try:
                return self._native.intersecting(box, include_touching)
            except (OverflowError, TypeError):
                pass
        root = self._ensure_py_root()
        if root is None:
            return ()
        return tuple(root.intersecting(box, include_touching=include_touching))
