"""Intrinsic alignment of complete shared contact access between two raw MOS.

The full contact lattice and its complete raw M1 component determine the only
translation. No observed pitch, clipping window or overlap width is accepted.
This checks local geometry; source nets, PDK identity and the complete terminal
graphs remain the caller's independent replay obligations.
"""
from dataclasses import dataclass

import klayout.db as kdb

from clay.layout.fixed_artwork_abutment import _region, _require
from clay.layout.joint_terminal_inventory import _geometry


@dataclass(frozen=True, slots=True)
class FixedArtworkSharedAccessAlignment:
    translation_nm: tuple[int, int]
    overlap_area_nm2: int
    scope: str = 'complete_intrinsic_contact_and_metal_coincidence_only'


def _access(artwork, site, contact_layer, metal_layer, checkpoint):
    _geometry(artwork, 'complete shared access raw artwork')
    layers = dict(artwork)
    _require(contact_layer in layers and metal_layer in layers, 'complete shared contact or metal layer is absent')
    cuts = kdb.Region()
    for shape in layers[contact_layer]:
        checkpoint()
        polygon = shape.to_region()
        if (polygon & site).is_empty():
            continue
        _require((polygon-site).is_empty(), 'complete contact polygon crosses the raw SD boundary')
        cuts += polygon
    cuts.merge()
    _require(not cuts.is_empty(), 'complete contact lattice is empty')
    metal = _region(layers[metal_layer], 'complete metal artwork', checkpoint).interacting(cuts)
    _require(metal.size() == 1 and (cuts-metal).is_empty(),
             'complete metal access must contain every contact in one raw component')
    return cuts, metal


def align_fixed_artwork_shared_access(*, left_site, right_site, left_channel, right_channel,
        left_artwork, right_artwork, contact_layer, metal_layer, checkpoint):
    """Align full shared access masks and prove the resulting positive SD overlap."""
    _require(callable(checkpoint), 'checkpoint must be callable')
    left = _region(left_site, 'left site', checkpoint)
    right = _region(right_site, 'right site', checkpoint)
    lc = _region(left_channel, 'left channel', checkpoint)
    rc = _region(right_channel, 'right channel', checkpoint)
    lb, rb = lc.bbox(), rc.bbox()
    _require(left.bbox().left >= lb.right and right.bbox().right <= rb.left,
             'shared raw site is not on its declared facing side')
    _require((left & lc).is_empty() and (right & rc).is_empty(), 'shared raw site overlaps its own channel')
    lcuts, lmetal = _access(left_artwork, left, contact_layer, metal_layer, checkpoint)
    rcuts, rmetal = _access(right_artwork, right, contact_layer, metal_layer, checkpoint)
    a, b = lcuts.bbox(), rcuts.bbox()
    dx, dy = a.left-b.left, a.bottom-b.bottom
    _require(abs(dx) <= 1_000_000_000 and abs(dy) <= 1_000_000_000, 'shared access translation exceeds bounds')
    move = kdb.Trans(dx, dy)
    _require((lcuts ^ rcuts.transformed(move)).is_empty(), 'complete contact lattices do not coincide')
    _require((lmetal ^ rmetal.transformed(move)).is_empty(), 'complete metal access components do not coincide')
    moved_channel = rc.transformed(move)
    rb = moved_channel.bbox()
    _require((lb.bottom, lb.top) == (rb.bottom, rb.top) and lb.right < rb.left
             and (lc & moved_channel).is_empty(), 'shared access channel order or full baselines differ')
    overlap = (left & right.transformed(move)).merged()
    bounds = overlap.bbox()
    _require(not overlap.is_empty() and overlap.area() == bounds.width()*bounds.height()
             and bounds.bottom <= lb.bottom and bounds.top >= lb.top,
             'shared SD overlap must cover the complete channel baseline span')
    _require((lcuts-overlap).is_empty(), 'complete shared contacts are not contained in both raw SD sites')
    checkpoint()
    return FixedArtworkSharedAccessAlignment((dx, dy), int(overlap.area()))
