"""Shared immutable placement vocabulary and the alignment-authority context."""
from __future__ import annotations

from contextvars import ContextVar
from typing import Tuple, Union
from clay.layout.canonical_collision_projection import CanonicalRowBoundedCavityProjection


_Bbox = Tuple[int, int, int, int]  # (x0, y0, x1, y1) nm

# Dense-pack fill estimate used only to seed the FIRST strip width; the aspect-
# correcting loop then adjusts the width from the measured aspect, so the exact value
# is not load-bearing (it only sets where the search starts).
_FILL_SEED = 0.72

ASPECT_RELATIVE_TOLERANCE = 0.05

# Backward-compatible private name used throughout this solver.  Artifact-side
# intent verification imports the public constant so both boundaries apply the
# exact same declared-aspect contract.
_ASPECT_TOL = ASPECT_RELATIVE_TOLERANCE

_MAX_ITERS = 6

_ASPECT_CONSTRAINT_SCALE = 1_000_000

_ALIGNMENT_SCALE_SOURCE_CERTIFICATE: ContextVar[object] = ContextVar(
    "clay_alignment_scale_source_certificate",
    default=None,
)

_FloorplanColumnTerm = Tuple[str, int, int]

# Internal scope tag for ``col=N`` declarations that intentionally omit an
# exact ``row=N``.  It participates only in horizontal centroid comparisons;
# it is never a Y coordinate or an electrical-band ordinal.
_GLOBAL_FLOORPLAN_COLUMN_SCOPE = -1

_FloorplanColumnGroup = Tuple[
    str,
    int,
    int,
    int,
    Tuple[_FloorplanColumnTerm, ...],
]

_AggregateEnvelopeOrder = Tuple[
    str,
    str,
    str,
    Tuple[str, ...],
    Tuple[str, ...],
]

_RigidCompoundFootprint = Tuple[str, Tuple[_Bbox, ...]]

_LayeredCollisionAtom = Tuple[int, int, int, int, int, int]

_LayeredCollisionFootprint = Tuple[str, Tuple[_LayeredCollisionAtom, ...]]

_RowBoundedCavityProjection = CanonicalRowBoundedCavityProjection

_HierarchicalMacroOrder = Tuple[
    str,
    str,
    str,
    str,
    str,
    Tuple[str, ...],
    Tuple[str, ...],
]

_HierarchicalMacroAlignmentGroups = Tuple[Tuple[str, Tuple[str, ...]], ...]

_HierarchicalMacroAlignment = Union[
    Tuple[str, str, _HierarchicalMacroAlignmentGroups],
    Tuple[str, str, str, _HierarchicalMacroAlignmentGroups],
    Tuple[str, str, str, int, _HierarchicalMacroAlignmentGroups],
]

_HierarchicalAlignmentToleranceScalesNm = Tuple[Tuple[str, int], ...]

_SECONDARY_AXIS_REPAIRABLE_BLOCKERS = frozenset(
    {
        "axis_alignment",
        "center_alignment",
        "aggregate_envelope_order",
        "hierarchical_order",
        "hierarchical_alignment",
        "mosaic_guest_slot",
    }
)

# --------------------------------------------------------------------------- #
# BAND-CONSTRAINED strip pack (opt-in): stack clusters bottom->center->top by a
# per-block band ordinal, pack horizontally within each band, keep a passive
# composite (the cap bank) on the OUTPUT (right) side of the center channel.
# --------------------------------------------------------------------------- #
# band ordinals: 0 = bottom (nmos, gnd side), 1 = center (passives), 2 = top
# (pmos, vdd side). The re-placer only reads the ORDINAL order (stack low->high),
# so the mapping is fully general (any N ordered bands).
_BAND_BOTTOM = 0

_BAND_CENTER = 1

_BAND_TOP = 2
