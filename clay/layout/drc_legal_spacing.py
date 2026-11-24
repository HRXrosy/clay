"""R137 (BL-137B) — DRC-legal inter-device spacing post-process.

R135's ``connectivity_placement`` spaces adjacent devices by ``4 * fin_pitch``
(192 nm on sky130). R136's ``run_layout_drc`` then finds that the resulting
inter-device diff-diff gaps (192 nm) are below the sky130 diff-diff min-spacing
rule (270 nm), so the routed 5T-OTA carries 6 inter-device diff-diff spacing
violations. (It also carries 5 poly-poly and 2 M1 violations — see the honest
scope note below.)

This module is a *placement post-process*: given a ``DevicePlacement`` produced
by ``connectivity_placement.compute_placement`` plus the per-device local
bounding boxes and the PDK rules, it widens the gaps BETWEEN devices so that
every pair of adjacent devices is separated by at least the relevant
min-spacing rule (the max of poly-poly and diff-diff, applied to whichever
axis already separates the two devices). It returns a NEW ``DevicePlacement``
with adjusted offsets. Topology / connectivity is untouched: only the
per-device translation offsets change, never the device set or any per-device
geometry, so a re-applied placement keeps the same netlist.

Honest scope (INTER vs INTRA device).
--------------------------------------
This post-process can ONLY fix INTER-device spacing — gaps *between* two
distinct devices' poly/diff bounding boxes. It widens those gaps by sliding the
two devices apart along whichever axis already separates them.

It CANNOT fix INTRA-device spacing — the gap between the poly fingers (or the
diff slices) *within a single device*. Those finger gaps are a property of the
base device lowering (the fingers sit at the fin pitch, 48 nm on sky130), and
moving a whole device does not change the spacing between its own fingers.
Fixing those requires changing the lowering (a wider drawn poly-to-poly finger
pitch), which is out of this lane's scope.

On the R136 5T-OTA the 13 min-spacing violations split as:
  * 5 poly-poly (66/20) @ 48 nm — every one is between the two fingers WITHIN a
    single device. INTRA-device. NOT fixable here (lowering-level).
  * 6 diff-diff (65/20) @ 192 nm — every one is between two distinct devices'
    diff bboxes. INTER-device. Fixable here -> widened to >= 270 nm.
  * 2 M1 (67/20) @ 52 nm — routed-wire spacing, not device placement at all.
    Out of this lane (router-level), reported for completeness.

So this lane targets the 6 inter-device diff-diff violations; the 5 intra-device
poly and 2 routing-wire violations are documented as out-of-scope, not papered
over.

No PDK number is fabricated: the min-spacing rules come from
``layout_drc.min_spacing_nm_for_layer`` (which reads the committed
``layer_pitch.py`` / ``drc.py`` tables). The geometry comes from the real
per-device bounding boxes. This module imports ``connectivity_placement``,
``layout_drc`` and ``backend`` read-only and modifies none of them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .annotations import AnnotatedCell
from .backend import LayerMap, LogicalLayer, ParameterizedLibrary
from .connectivity_placement import (
    DevicePlacement,
    _device_local_bboxes,
    find_overlaps,
)
from .layout_drc import min_spacing_nm_for_layer

# (layer, datatype) keys we care about for inter-device spacing: the poly and
# the diff layers. Resolved per-PDK from the LayerMap so no constant is hard
# coded here.
_INTER_DEVICE_LOGICAL_LAYERS = (
    LogicalLayer.POLY,
    LogicalLayer.NDIFF,
    LogicalLayer.PDIFF,
)


Bbox = Tuple[int, int, int, int]


@dataclass
class SpacingAdjustment:
    """Audit record for the inter-device spacing widening.

    Attributes:
        required_spacing_nm: the inter-device min-spacing target actually used
            (max of the poly-poly and diff-diff rules for the PDK).
        original_min_inter_gap: the smallest inter-device bbox gap (nm) before
            widening (None if there were fewer than two devices / no gaps).
        adjusted_min_inter_gap: the smallest inter-device bbox gap (nm) after
            widening.
        moved_devices: device names whose offset changed.
    """

    required_spacing_nm: int = 0
    original_min_inter_gap: Optional[int] = None
    adjusted_min_inter_gap: Optional[int] = None
    moved_devices: List[str] = field(default_factory=list)


# --- PDK rule resolution (real values only) ---------------------------------


def required_inter_device_spacing_nm(pdk_layers: LayerMap) -> int:
    """The inter-device min-spacing target for a PDK (nm).

    Adjacent devices place poly *and* diff next to each other, so a legal
    inter-device gap must satisfy BOTH the poly-poly and the diff-diff rule.
    The binding constraint is the larger of the two. Values come from
    ``layout_drc.min_spacing_nm_for_layer`` (real committed PDK tables).

    Raises:
        ValueError: if neither a poly nor a diff min-spacing rule is available
            for the PDK (so we never silently widen by a fabricated number).
    """
    rules: List[int] = []
    for logical in _INTER_DEVICE_LOGICAL_LAYERS:
        g = pdk_layers.gds_layer(logical)
        if g is None:
            continue
        s = min_spacing_nm_for_layer(pdk_layers, g.layer, g.datatype)
        if s is not None:
            rules.append(int(s))
    if not rules:
        raise ValueError(
            "PDK provides no poly/diff min-spacing rule; cannot compute a "
            "legal inter-device spacing without a real rule value"
        )
    return max(rules)


# --- geometry helpers --------------------------------------------------------


def _placed_bbox(local: Bbox, off: Tuple[int, int]) -> Bbox:
    dx, dy = off
    x0, y0, x1, y1 = local
    return (x0 + dx, y0 + dy, x1 + dx, y1 + dy)


def _axis_gap(a: Bbox, b: Bbox) -> Tuple[int, int]:
    """Signed gaps (x_gap, y_gap) between two placed bboxes.

    A positive value is the clear separation on that axis; <= 0 means the
    projections overlap / touch on that axis.
    """
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    x_gap = max(bx0 - ax1, ax0 - bx1)
    y_gap = max(by0 - ay1, ay0 - by1)
    return x_gap, y_gap


def inter_device_min_gap(
    placed: Dict[str, Bbox],
) -> Optional[Tuple[int, str, str, str]]:
    """Smallest separating inter-device gap across all device pairs.

    Returns ``(gap_nm, dev_a, dev_b, axis)`` for the tightest pair that is
    separated on at least one axis (axis in {"x","y"}), or None if there are
    no separated pairs (e.g. fewer than two devices). Pairs that overlap on
    both axes are skipped (they are an overlap problem, not a spacing one).
    """
    names = sorted(placed.keys())
    best: Optional[Tuple[int, str, str, str]] = None
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            x_gap, y_gap = _axis_gap(placed[a], placed[b])
            # The separating gap is the max of the two axis gaps (the axis on
            # which the boxes do NOT overlap). If both are <= 0 the boxes
            # overlap -> not a spacing pair.
            sep = max(x_gap, y_gap)
            if sep <= 0:
                continue
            axis = "x" if x_gap >= y_gap else "y"
            if best is None or sep < best[0]:
                best = (sep, a, b, axis)
    return best


# --- core widening -----------------------------------------------------------


def _widen_offsets(
    local_bboxes: Dict[str, Bbox],
    offsets: Dict[str, Tuple[int, int]],
    required_nm: int,
) -> Dict[str, Tuple[int, int]]:
    """Scale each device's center-to-center separation so every adjacent-device
    gap reaches ``required_nm``.

    Strategy: keep each device's local size, keep the relative *ordering* of
    device origins on each axis, and uniformly grow the inter-device gaps. We
    do this per axis independently:

      * Sort devices by their placed-bbox lower edge on the axis.
      * Walk the sorted order; whenever consecutive (in that order) device
        bboxes are separated by a positive gap on this axis that is below
        ``required_nm``, shift the current device (and everything after it in
        the order) further along the axis by the deficit.

    This only ever *increases* separations, so it can never create a new
    overlap, and it preserves the connectivity-aware ordering (tail below
    inputs, loads above). Devices whose bboxes overlap on the axis are not
    shifted relative to each other on that axis (their separation lives on the
    other axis), which keeps stacked rows aligned.
    """
    new_off = dict(offsets)

    for axis in (0, 1):  # 0 = x, 1 = y
        def lo(dev: str) -> int:
            return _placed_bbox(local_bboxes[dev], new_off[dev])[axis]

        def hi(dev: str) -> int:
            return _placed_bbox(local_bboxes[dev], new_off[dev])[axis + 2]

        order = sorted(local_bboxes.keys(), key=lambda d: (lo(d), d))
        # Cumulative shift to apply to the current device and all later ones.
        for idx in range(1, len(order)):
            cur = order[idx]
            # Find the deficit against the nearest preceding device that
            # actually separates from `cur` on this axis (positive gap).
            deficit = 0
            cur_lo = lo(cur)
            for k in range(idx):
                prev = order[k]
                gap = cur_lo - hi(prev)
                if gap <= 0:
                    # overlap/touch on this axis -> separation is on the other
                    # axis; do not force a gap here.
                    continue
                if gap < required_nm:
                    need = required_nm - gap
                    if need > deficit:
                        deficit = need
            if deficit > 0:
                # Shift cur and every later device by `deficit` on this axis.
                for later in order[idx:]:
                    dx, dy = new_off[later]
                    if axis == 0:
                        new_off[later] = (dx + deficit, dy)
                    else:
                        new_off[later] = (dx, dy + deficit)

    return new_off


# --- public API --------------------------------------------------------------


def legalize_inter_device_spacing(
    lib: ParameterizedLibrary,
    cell: AnnotatedCell,
    placement: DevicePlacement,
    pdk_layers: LayerMap,
) -> Tuple[DevicePlacement, SpacingAdjustment]:
    """Return a placement with DRC-legal inter-device poly/diff spacing.

    Args:
        lib: a ``lower_*`` output (supplies per-device local-origin geometry).
        cell: the annotated cell (selects the per-device cells).
        placement: a ``DevicePlacement`` from ``compute_placement`` to widen.
        pdk_layers: the PDK LayerMap (supplies the real min-spacing rules).

    Returns:
        ``(new_placement, adjustment)``. ``new_placement`` keeps the same
        ``strategy`` (with a ``+drc_legal_spacing`` suffix) and the same device
        set; only the offsets change. ``adjustment`` records the rule used and
        the before/after tightest inter-device gap.

    The returned placement is overlap-free whenever the input was, since the
    widening only ever increases device separations.
    """
    local = _device_local_bboxes(lib, cell)
    # Only consider devices that have geometry AND an offset.
    devs = [d for d in placement.offsets if d in local]
    placed_before = {d: _placed_bbox(local[d], placement.offsets[d]) for d in devs}

    required = required_inter_device_spacing_nm(pdk_layers)
    before = inter_device_min_gap(placed_before)

    new_off = _widen_offsets(
        {d: local[d] for d in devs},
        {d: placement.offsets[d] for d in devs},
        required,
    )
    # Devices not in ``local`` (no geometry) keep their original offset.
    for d in placement.offsets:
        if d not in new_off:
            new_off[d] = placement.offsets[d]

    placed_after = {d: _placed_bbox(local[d], new_off[d]) for d in devs}
    after = inter_device_min_gap(placed_after)

    moved = sorted(
        d for d in devs if new_off[d] != placement.offsets[d]
    )
    new_placement = DevicePlacement(
        offsets=new_off,
        strategy=placement.strategy + "+drc_legal_spacing",
    )
    adjustment = SpacingAdjustment(
        required_spacing_nm=required,
        original_min_inter_gap=None if before is None else before[0],
        adjusted_min_inter_gap=None if after is None else after[0],
        moved_devices=moved,
    )
    return new_placement, adjustment


def inter_device_spacing_legal(
    lib: ParameterizedLibrary,
    cell: AnnotatedCell,
    placement: DevicePlacement,
    pdk_layers: LayerMap,
) -> bool:
    """True iff every separated adjacent-device pair meets the inter-device
    min-spacing rule (and the placement is overlap-free)."""
    if find_overlaps(lib, cell, placement):
        return False
    local = _device_local_bboxes(lib, cell)
    devs = [d for d in placement.offsets if d in local]
    placed = {d: _placed_bbox(local[d], placement.offsets[d]) for d in devs}
    required = required_inter_device_spacing_nm(pdk_layers)
    tightest = inter_device_min_gap(placed)
    if tightest is None:
        return True  # no separated pairs -> vacuously legal
    return tightest[0] >= required


__all__ = [
    "SpacingAdjustment",
    "required_inter_device_spacing_nm",
    "inter_device_min_gap",
    "legalize_inter_device_spacing",
    "inter_device_spacing_legal",
]
