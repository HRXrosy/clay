"""Pure integer displacement-domain geometry for layered rigid rectangles."""
from heapq import merge, nlargest
from itertools import chain

from clay._native import load_native
from clay.layout.spatial.rectangle_union import rectangle_union_cover

_ORIG_RECTANGLE_UNION_COVER = rectangle_union_cover


def _translated(left, right, required):
    # Bind each left operand in its own generator frame, before heap merge
    # advances any streams. A loop-scoped generator expression would alias it.
    lx0, ly0, lx1, ly1 = left
    for rx0, ry0, rx1, ry1 in right:
        yield (lx0-rx1-required+1, ly0-ry1-required+1,
               lx1+required-rx0, ly1+required-ry0)


def _ordered_pairs(left_atoms, right_atoms, required, checkpoint):
    """Sort operands once; merge translated streams with O(L+R) state."""
    right_by_layer = {}
    all_right = []
    for index, (layer, rectangle) in enumerate(right_atoms):
        if checkpoint and index % 256 == 0:
            checkpoint()
        right_by_layer.setdefault(layer, []).append(rectangle)
        all_right.append(rectangle)
    keyed = 0
    def order(box):
        nonlocal keyed
        keyed += 1
        if checkpoint and keyed % 256 == 0:
            checkpoint()
        x0, y0, x1, y1 = box
        return -x1, -y1, -x0, -y0
    for layer, rows in right_by_layer.items():
        right_by_layer[layer] = sorted(rows, key=order)
    all_right.sort(key=order)
    streams = []
    for index, (layer, rectangle) in enumerate(left_atoms):
        if checkpoint and index % 256 == 0:
            checkpoint()
        groups = ((all_right,) if layer is None else
                  (right_by_layer.get(None, ()), right_by_layer.get(layer, ())))
        for group in groups:
            if group:
                streams.append(_translated(rectangle, group, required))
    for index, box in enumerate(merge(*streams)):
        if checkpoint and index % 256 == 0:
            checkpoint()
        yield box


def _batched_cover(boxes, checkpoint):
    """Union bounded batches, merging equal-sized runs in a binary tree.

    Temporary storage follows the input batch and intermediate union covers,
    never a retained set of every atom pair. A sparse final cover necessarily
    still needs storage proportional to its actual output geometry.
    """
    levels = []
    batch = []
    witnesses = ()

    def add_run(rows):
        nonlocal witnesses
        carry = rectangle_union_cover(rows, checkpoint=checkpoint)
        level = 0
        while level < len(levels) and levels[level] is not None:
            carry = rectangle_union_cover((*levels[level], *carry), checkpoint=checkpoint)
            levels[level] = None
            level += 1
        if level == len(levels):
            levels.append(carry)
        else:
            levels[level] = carry
        retained = set(witnesses)
        witnesses = tuple(nlargest(8,
            chain(witnesses, (box for box in carry if box not in retained)),
            key=lambda box: (box[2]-box[0])*(box[3]-box[1])))

    for box in boxes:
        # A single actual union rectangle is a sufficient containment witness.
        # Never use its envelope across holes. Counts are advanced upstream
        # even for discarded geometry, as are raw-stream cancellation polls.
        x0, y0, x1, y1 = box
        if any(a <= x0 and b <= y0 and x1 <= c and y1 <= d for a,b,c,d in witnesses):
            continue
        batch.append(box)
        if len(batch) == 2048:
            add_run(batch)
            batch.clear()
    if batch:
        add_run(batch)
    result = ()
    for cover in levels:
        if cover is not None:
            result = rectangle_union_cover((*result, *cover), checkpoint=checkpoint)
    return result


def _forbidden_box_cover_py(left_atoms, right_atoms, *, required_nm, checkpoint=None):
    """Preserve every compatible atom-pair exclusion, including integer edges.

    Returned raw/unique counts retain the original atom denominator. The
    geometric union only changes its representation, not the legal domain.
    No reference, PDK, placement seed or solver state is consumed here.
    """
    if type(required_nm) is not int or required_nm < 0:
        raise ValueError('pair collision clearance must be one non-negative integer')
    if checkpoint:
        checkpoint()
    left_atoms, right_atoms = tuple(left_atoms), tuple(right_atoms)
    raw_count = unique_count = 0
    wide = False

    def unique_boxes():
        nonlocal raw_count, unique_count, wide
        previous = None
        for box in _ordered_pairs(left_atoms, right_atoms, required_nm, checkpoint):
            raw_count += 1
            if box != previous:
                unique_count += 1
                wide |= any(v < -(2**31) or v > 2**31-1 for v in box)
                yield box
                previous = box

    exact = _batched_cover(unique_boxes(), checkpoint)
    if unique_count and not exact:
        raise ValueError('pair collision forbidden union lost all geometry')
    # Keep the old representation choice, including the wide-coordinate
    # compatibility branch. Only an actual output can require this storage;
    # compact unions do not retain the pair population just for these counts.
    if wide or len(exact) >= unique_count:
        previous = None
        canonical = []
        for box in _ordered_pairs(left_atoms, right_atoms, required_nm, checkpoint):
            if box != previous:
                canonical.append(box)
                previous = box
        exact = tuple(canonical)
    if checkpoint:
        checkpoint()
    return exact, raw_count, unique_count


def forbidden_box_cover(left_atoms, right_atoms, *, required_nm, checkpoint=None):
    """Preserve every compatible atom-pair exclusion, including integer edges."""
    if type(required_nm) is not int or required_nm < 0:
        raise ValueError('pair collision clearance must be one non-negative integer')
    native = load_native("_place_cpp")
    if native is not None and rectangle_union_cover is _ORIG_RECTANGLE_UNION_COVER:
        if type(left_atoms) in (tuple, list) and type(right_atoms) in (tuple, list):
            try:
                return native.forbidden_box_cover_cpp(
                    left_atoms, right_atoms, required_nm, checkpoint
                )
            except OverflowError:
                return _forbidden_box_cover_py(
                    left_atoms, right_atoms, required_nm=required_nm, checkpoint=checkpoint
                )
        if checkpoint:
            checkpoint()
        left_rows, right_rows = tuple(left_atoms), tuple(right_atoms)
        try:
            return native.forbidden_box_cover_cpp(
                left_rows, right_rows, required_nm, checkpoint
            )
        except OverflowError:
            return _forbidden_box_cover_py(
                left_rows, right_rows, required_nm=required_nm, checkpoint=checkpoint
            )
    return _forbidden_box_cover_py(
        left_atoms, right_atoms, required_nm=required_nm, checkpoint=checkpoint
    )
