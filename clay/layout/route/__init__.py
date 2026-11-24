"""Clay routing: intent + a single generic FAITHFUL router + verification.

A clean namespace, deliberately OUTSIDE the legacy ``clay/layout/*_vNNN`` router
pile (which is per-circuit-hardcoded and superseded). The architecture (see
``docs/superpowers/specs/2026-06-07-clay-routing-architecture.html``) is three
separable halves:

  * INTENT  (:mod:`clay.layout.route.intent`) -- the declarative, design-derived,
    leakage-free spec of HOW nets are routed (net->layer, symmetry, width, shield,
    length-match, track-order, access, via). Open vocabulary.
  * a single generic FAITHFUL ROUTER (:mod:`clay.layout.route.engine`) -- takes a
    placement + pins + the lowered intent + PDK rules and produces DRC-legal
    geometry that SATISFIES every hard constraint (or honestly reports UNSAT); its
    own secondary objectives (short wire, few vias) act ONLY on the don't-care
    degrees of freedom and never override the intent. ZERO per-circuit branches.
  * the INTENT lowers to a small CLOSED set of generic constraints
    (:mod:`clay.layout.route.constraints`) the engine consumes -- so adding an
    intent primitive is a new lowering, not an engine change.

Verification (LVS connectivity + real DRC + route-invariance + per-constraint
obligations) judges the router's output against the intent.
"""
from __future__ import annotations

from clay.layout.route.constraints import (
    RouteConstraint,
    LayerSet,
    Width,
    Mirror,
    Spacing,
    Shield,
    Length,
    TrackOrder,
    Access,
    Via,
    CONSTRAINT_KINDS,
)
from clay.layout.route.intent import RoutingIntent, IntentSource
from clay.layout.route.engine import (
    Pin,
    Obstacle,
    NetRoute,
    RoutedCell,
    RouteEngine,
    RouteUnsat,
)
from clay.layout.route.lower import lower_intent

__all__ = [
    "RouteConstraint",
    "LayerSet",
    "Width",
    "Mirror",
    "Spacing",
    "Shield",
    "Length",
    "TrackOrder",
    "Access",
    "Via",
    "CONSTRAINT_KINDS",
    "RoutingIntent",
    "IntentSource",
    "Pin",
    "Obstacle",
    "NetRoute",
    "RoutedCell",
    "RouteEngine",
    "RouteUnsat",
    "lower_intent",
]
