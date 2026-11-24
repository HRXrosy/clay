"""Intent Implementation ABC -- the registered adapter over ONE layout-intent KIND.

The re-arch's keystone (spec ``docs/superpowers/specs/2026-07-17-clay-intent-ir-engine-
rearch-p0p3.html`` §3.1, Wave B/A5): every declarative ``LayoutIntent`` field becomes a
REGISTERED :class:`IntentImplementation`, dispatched by intent KIND -- never by a
circuit / device / cell / OTA name. The engine (placer / router) keeps its CLOSED generic
vocabulary; the circuit-semantics -> generic-physical-object translation lives HERE, one
implementation per kind. Adding a new dialect construct is a new implementation, not an
engine branch.

Each implementation exposes the six-method surface the program names:

* ``applicable(decl, pdk) -> bool``     -- does this impl handle ``decl`` (looks at
  ``decl.kind`` + an optional PDK capability probe, NEVER a circuit name)?
* ``lower(decl, ctx) -> LowerResult``   -- one declaration -> generic physical objects
  (route constraints + placement structure requests); a pure function, no coordinates.
* ``materialize(decl, ctx) -> CompositeMacro | None`` -- optional rigid macro (default
  ``None`` = the impl only adds constraints, does not build a composite tile).
* ``obligations(decl, ctx) -> [SourceObligation]`` -- the observable evidence the intent
  owes on the emitted artifact.
* ``capabilities(decl, backend) -> CheckerVerdict`` -- the PRE-handshake half: a 5-state
  verdict of whether ``backend`` can satisfy the generic kinds this decl lowers to. It
  NEVER infers a substitute and NEVER silently passes; an unsatisfiable required kind
  surfaces ``UNSUPPORTED`` and feeds the capability handshake (Wave A).
* ``provenance(decl) -> Provenance``    -- USER / TOPOLOGY / PDK source (default USER).

This module is DATA + interface only; the concrete craft adapters (guard_rings, arrays,
common_centroids, rail_affinities, ...) live in
:mod:`clay.layout.intent_impls_craft` and register through
:mod:`clay.layout.intent_registry`.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, FrozenSet, Optional, Sequence, Tuple

from clay.core_contract import SourceObligation, VerificationStatus
from clay.signoff_verdict import CheckerVerdict

__all__ = [
    "Provenance",
    "StructureRequest",
    "LowerResult",
    "PLACEMENT_REQUEST_KINDS",
    "IntentImplementation",
]


class Provenance(str, Enum):
    """Where a physical object's authority comes from (invariant I8: provenance is
    explicit). USER = the designer declared it; TOPOLOGY = derived from the netlist
    structure; PDK = a process rule / strategy (e.g. preferred routing axis, via
    redundancy) -- PDK-provenance objects live in the backend, never as an intent."""

    USER = "user"
    TOPOLOGY = "topology"
    PDK = "pdk"


# The CLOSED placement-side request vocabulary (the placement dual of
# ``route.constraints.CONSTRAINT_KINDS``). An intent lowers ONLY to these generic
# physical objects -- there is no "input pair" / "OTA output stage" request. Kept in
# lock-step with :data:`clay.layout.place.constraints.PLACEMENT_CONSTRAINT_KINDS` (a test
# asserts the equality), so this module and the placement closed set cannot drift.
PLACEMENT_REQUEST_KINDS: FrozenSet[str] = frozenset(
    {
        "rigid_macro",
        "relative_order",
        "abutment",
        "symmetry",
        "region_affinity",
        "keep_together",
        "alignment",
        "edge_affinity",
        "channel",
        "keepout",
        "spacing",
    }
)


@dataclass(frozen=True)
class StructureRequest:
    """A placement request in the CLOSED generic vocabulary (the placement dual of a
    :class:`~clay.layout.route.constraints.RouteConstraint`).

    ``kind`` is a member of :data:`PLACEMENT_REQUEST_KINDS` -- a generic physical
    object (rigid_macro / symmetry / relative_order / region_affinity / edge_affinity /
    channel / keepout / ...), NEVER a circuit role. ``spec`` is a tuple of device
    LABELS + generic geometric knobs; it carries NO coordinate and NO role name. Frozen
    + hashable so a placement graph can dedup / order requests deterministically."""

    kind: str
    spec: Tuple[Any, ...] = ()
    hardness: str = "hard"  # HARD (gates the handshake) | SOFT (advisory bias)

    def __post_init__(self) -> None:
        if self.kind not in PLACEMENT_REQUEST_KINDS:
            raise ValueError(
                f"StructureRequest.kind {self.kind!r} is not in the CLOSED placement "
                f"vocabulary {sorted(PLACEMENT_REQUEST_KINDS)} -- a new generic kind is a "
                "deliberate CONSTRAINT_KINDS extension + one placer branch, never an "
                "ad-hoc string."
            )


@dataclass(frozen=True)
class LowerResult:
    """The generic physical objects one declaration lowers to: route constraints
    (:mod:`clay.layout.route.constraints`) + placement structure requests (above).
    Both default empty so an impl that lowers to neither is a legal, honest no-op."""

    route_constraints: Tuple[Any, ...] = ()
    structure_requests: Tuple[StructureRequest, ...] = ()

    def request_kinds(self) -> FrozenSet[str]:
        return frozenset(r.kind for r in self.structure_requests)

    def route_kinds(self) -> FrozenSet[str]:
        return frozenset(
            str(getattr(c, "kind", "?")) for c in self.route_constraints
        )


class IntentImplementation(abc.ABC):
    """The registered adapter for ONE intent KIND. Subclasses set ``kind`` to a
    declarative ``LayoutIntent`` field name (validated by the registry against the
    intent-kind universe); dispatch is NOMINAL (keyed on the declared kind), never
    structural, and never on a circuit name."""

    #: the intent KIND this implements -- MUST equal a declarative LayoutIntent field
    #: name (registry-enforced). NEVER a cell / device / OTA name.
    kind: str = ""
    #: the term's honest E/L/V tier (PASS = verified obligation, ADVISORY = drives the
    #: engine but never gates, UNSUPPORTED = parsed only / no spine consumer yet). This
    #: is the backend-INDEPENDENT classification the ELV台账 reads.
    status: VerificationStatus = VerificationStatus.UNSUPPORTED
    #: is the obligation REQUIRED (a required UNSUPPORTED/UNMEASURABLE blocks signoff)?
    required: bool = False

    @abc.abstractmethod
    def applicable(self, decl: Any, pdk: Any) -> bool:
        """Does this impl handle ``decl``? Looks ONLY at ``decl.kind`` (+ an optional
        PDK capability probe). Raising or reading a circuit name here is banned."""

    @abc.abstractmethod
    def lower(self, decl: Any, ctx: Any) -> LowerResult:
        """One declaration -> generic physical objects. Pure; no coordinates."""

    def materialize(self, decl: Any, ctx: Any) -> Optional[Any]:
        """Optional rigid ``CompositeMacro``. Default ``None`` = constraints only."""
        return None

    @abc.abstractmethod
    def obligations(self, decl: Any, ctx: Any) -> Sequence[SourceObligation]:
        """The observable evidence this intent owes on the emitted artifact."""

    @abc.abstractmethod
    def capabilities(self, decl: Any, backend: Any) -> CheckerVerdict:
        """PRE-handshake: 5-state verdict of whether ``backend`` can satisfy the
        generic kinds ``decl`` lowers to. NEVER infers a substitute; an unsatisfiable
        required kind surfaces UNSUPPORTED (never a silent pass / silent drop)."""

    def provenance(self, decl: Any) -> Provenance:
        return Provenance.USER
