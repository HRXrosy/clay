"""Intent-completeness measurement (the INVARIANCE HARNESS).

This subpackage is a pure MEASUREMENT INSTRUMENT. It does NOT place anything and
it never edits a placer. It CONSUMES placement results (the drop-in
:class:`clay.layout.generic_placement.Placement` both engines return) and
quantifies how much of the final layout is *pinned by the declared intent* vs
*left to the placer to decide*.

The operational definition:

  A declared layout intent is COMPLETE (up to the global don't-cares: absolute
  origin + an optional global mirror) iff ANY faithful placer produces the SAME
  layout. The operational test is PLACER-INVARIANCE: run the SAME declared intent
  through different faithful placers / deterministic input perturbations; every
  invariant that matters must be identical. Whatever VARIES is a layout decision
  NOT pinned by any primitive -- the incompleteness to close later (by adding
  GENERAL language primitives, never by tuning a placer per-circuit).

Public API:

  * :func:`extract_invariants` -- canonical (sorted, deterministic) dict over the
    8 shared invariant categories, computed off one placement result.
  * :func:`run_placement` -- thin adapter onto the EXISTING placement entry point
    (``place_to_intent(engine=...)``), forcing ``"milp"`` or ``"seqpair"``.
  * :func:`incompleteness_report` -- the disagreement set per category across runs
    + a scalar ``incompleteness_score`` (0 = complete up to global don't-cares).
"""
from .invariance import (
    INVARIANT_CATEGORIES,
    extract_invariants,
    incompleteness_report,
    run_placement,
)
from .route_invariance import (
    ROUTE_INVARIANT_CATEGORIES,
    NetCompletenessGap,
    RouteIncompletenessReport,
    extract_route_invariants,
    route_incompleteness_report,
    run_route,
)
from .gds_invariants import (
    GdsAnchorObligation,
    GdsBandObligation,
    GdsChannelObligation,
    GdsGuardRingObligation,
    MIM_PLATE_LAYER,
    extract_invariants_from_gds,
    geometric_placement_from_gds,
    verify_obligations_on_gds,
    verify_anchors_on_gds,
    verify_bands_on_gds,
    verify_channels_on_gds,
    verify_guard_rings_on_gds,
)

__all__ = [
    "INVARIANT_CATEGORIES",
    "extract_invariants",
    "run_placement",
    "incompleteness_report",
    # routing analog
    "ROUTE_INVARIANT_CATEGORIES",
    "extract_route_invariants",
    "run_route",
    "route_incompleteness_report",
    "RouteIncompletenessReport",
    "NetCompletenessGap",
    # GDS-truth obligation measurement (realization soundness)
    "MIM_PLATE_LAYER",
    "GdsAnchorObligation",
    "GdsBandObligation",
    "GdsChannelObligation",
    "GdsGuardRingObligation",
    "geometric_placement_from_gds",
    "extract_invariants_from_gds",
    "verify_obligations_on_gds",
    "verify_anchors_on_gds",
    "verify_bands_on_gds",
    "verify_channels_on_gds",
    "verify_guard_rings_on_gds",
]
