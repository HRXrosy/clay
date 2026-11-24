"""The CLOSED set of generic routing constraints the faithful router consumes.

This is the extensibility keystone. Every routing-INTENT primitive (open vocabulary
-- @route_layer / @mirror_route / @shield / @wire_width / @match_length / ...) lowers
(see :mod:`clay.layout.route.lower`) to one or more of THESE constraint types, and the
engine satisfies the constraint types -- not the primitives. So:

  * adding a new intent primitive that maps onto existing constraint types
    => a new lowering only; the engine does NOT change.
  * adding a genuinely-new constraint type (rare) => add one frozen dataclass here
    AND teach the engine one new "how to satisfy it" branch -- a single, well-defined
    place, never a per-primitive or per-circuit fork.

Each constraint is a frozen dataclass (hashable, comparable, no behaviour). Layers are
``(layer, datatype)`` GDS tuples OR role names ('M1','M2',...) -- the engine resolves
role names through :class:`clay.layout.tech.LayoutTech`; never hardcode PDK numbers in a
constraint.

NOTE on connectivity: "this net connects these pins" is the MANDATORY base requirement
and travels as the ``pins_by_net`` input to the engine (see :mod:`clay.layout.route.engine`),
NOT as an optional constraint here -- a route that does not connect the netlist is simply
wrong (LVS), not an intent choice. The constraints below are the declarative OVERLAY on
top of mandatory connectivity.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Tuple, Union

# A layer is either a GDS (layer, datatype) pair or a PDK role name resolved via tech.
Layer = Union[Tuple[int, int], str]

# --- hardness axis (A0) ------------------------------------------------------
# Every constraint carries a HARDNESS: HARD (a required obligation -- the route is
# WRONG if it is not satisfiable, so an unsupported HARD kind must STOP compilation
# before emit) or SOFT (an advisory preference -- a bias the engine honours where it
# can, never a hard block, never a stop). This is the axis the capability handshake
# (the PRE-gate in route_cell) keys on: only HARD constraints of an unsupported KIND
# gate; SOFT ones are advisory. It is a FIELD on every constraint (below), so an
# intent lowers a constraint at the right hardness WITHOUT the engine/gate ever
# branching on the kind -- exactly the per-kind special-case the architecture bans.
# Defaults preserve current behaviour: everything currently enforced defaults HARD;
# Direction (a preferred-direction bias, never a hard block -- see its docstring)
# defaults SOFT, so a design's HVH discipline can never collapse into a hard stop.
HARD = "hard"
SOFT = "soft"


@dataclass(frozen=True)
class LayerSet:
    """Net ``net`` may use ONLY these metal layers (e.g. force a rail onto M5, keep a
    sensitive signal off a noisy layer). From ``@route_layer`` / ``@on_layer``."""
    net: str
    layers: FrozenSet[Layer]
    kind: str = "layer_set"
    hardness: str = HARD


@dataclass(frozen=True)
class Width:
    """Net ``net``'s wire width = ``width_um`` (clamped UP to the PDK metal-min by the
    engine, never down). From ``@wire_width`` or an EM / current-driven width rule.
    ``width_um`` is process knowledge (current/role), NOT a memorised reference value."""
    net: str
    width_um: float
    kind: str = "width"
    hardness: str = HARD


@dataclass(frozen=True)
class Mirror:
    """The route of ``net_b`` must be the mirror image of ``net_a`` about the vertical
    axis ``axis_x_um`` (differential / matched pair). The engine routes one and reflects
    the other (and verifies the reflection is obstacle-free). From ``@mirror_route`` /
    a circuit ``SymmetryGroup`` (via :mod:`clay.layout.symmetry_route`)."""
    net_a: str
    net_b: str
    axis_x_um: float
    kind: str = "mirror"
    hardness: str = HARD


@dataclass(frozen=True)
class Spacing:
    """Minimum edge-to-edge metal spacing ``min_um`` between two named nets
    on each shared routing layer, in addition to the PDK rules. Via landing pads
    count as metal; different-layer crossing is not prohibited by this rule.
    From ``@shield`` (the inter-net leg) or an explicit spacing intent."""
    net_a: str
    net_b: str
    min_um: float
    kind: str = "spacing"
    hardness: str = HARD


@dataclass(frozen=True)
class Shield:
    """Route a grounded shield net ``by`` between aggressor->victim ``net`` and keep
    ``min_um`` to it. Lowers ALSO to a Spacing; kept as its own type so the engine can
    synthesise the shield wire. From ``@shield(net, by=, min=)``."""
    net: str
    by: str
    min_um: float
    kind: str = "shield"
    hardness: str = HARD


@dataclass(frozen=True)
class Length:
    """Matched-length: ``|len(net_a) - len(net_b)| <= tol_rel * max(len)``. From
    ``@match_length``. Symmetric pairs get this for free via :class:`Mirror`; this is
    for non-mirror matched nets."""
    net_a: str
    net_b: str
    tol_rel: float = 0.1
    kind: str = "length"
    hardness: str = HARD


@dataclass(frozen=True)
class TrackOrder:
    """In channel ``channel`` the listed nets occupy tracks in THIS relative order
    (coordinate-quotiented). From ``@route_order``."""
    channel: str
    nets: Tuple[str, ...]
    kind: str = "track_order"
    hardness: str = HARD


@dataclass(frozen=True)
class Access:
    """Net ``net`` enters device ``device`` terminal ``terminal`` from side ``side``
    ('L'/'R'/'U'/'D'/'on'). From ``@access``."""
    net: str
    device: str
    terminal: str
    side: str
    kind: str = "access"
    hardness: str = HARD


@dataclass(frozen=True)
class Direction:
    """Per-layer preferred routing direction (the HVH layer discipline): an in-plane
    routing step on metal ``layer_n`` along the NON-preferred axis pays an EXTRA cost
    of ``weight`` x the base step cost. A SOFT preference, never a hard block -- a net
    that must jog against the grain still may (it just pays for it), so connectivity
    is never sacrificed to direction style.

    ``layer_n`` is the 1-based PDK metal index (1 = metal1); ``axis`` is ``'H'`` or
    ``'V'``; ``weight`` >= 0 (0 = disabled preference -- explicitly inert, byte-stable).
    A later Direction for the same layer overrides an earlier one, so an explicit
    designer Direction wins over the conventional table the lowering emits. A
    Direction for a metal outside the engine's routing stack is vacuously satisfied
    (there is nothing to bias). From the HVH process-convention table in
    :mod:`clay.layout.route.lower` (``CONVENTIONAL_AXES``) or an explicit
    ``@route_direction`` intent."""
    layer_n: int
    axis: str  # 'H' | 'V'
    weight: float
    kind: str = "direction"
    # SOFT by construction: Direction is a preference, never a hard block (see the
    # docstring above). It must NEVER gate the capability handshake -- a design's
    # HVH discipline lowered onto a backend that does not consume Direction is an
    # advisory no-op, not an UNSUPPORTED stop.
    hardness: str = SOFT


@dataclass(frozen=True)
class TrunkLayer:
    """Prefer long runs of ``net`` on/above ``layer`` without capping transport.

    This is the soft routing meaning of ``@trunk(net, layer=...)``: the declared
    layer biases the net's main run, while upper metals remain available for a
    legal cross-band detour.  A hard allowed-layer restriction is represented
    separately by :class:`LayerSet`.
    """
    net: str
    layer: Layer
    orient: str = ""
    kind: str = "trunk_layer"
    hardness: str = SOFT

    def __post_init__(self) -> None:
        axis = str(self.orient).strip().upper()
        if axis not in {"", "H", "V"}:
            raise TrunkLayerDeclarationError(
                f"trunk orientation must be H, V, or blank; got {self.orient!r}"
            )
        object.__setattr__(self, "orient", axis)


class TrunkLayerDeclarationError(ValueError):
    """A lowered trunk-layer advisory is malformed or unresolved."""


@dataclass(frozen=True)
class Via:
    """Via policy for net ``net`` ('stack' = straight stack, 'array' = N-cut array,
    'max:<n>' = cap). From ``@via_stack`` / ``@max_vias``."""
    net: str
    policy: str
    kind: str = "via"
    hardness: str = HARD


@dataclass(frozen=True)
class ChannelRoute:
    """Hard ordered channel bound to actual placed row edges and a shared axis."""
    declaration: object
    y_low_nm: int
    y_high_nm: int
    axis_x2_nm: int | None = None
    kind: str = "channel_route"
    hardness: str = HARD

    def __post_init__(self):
        from clay.layout.channel_intent import ChannelRouteIntent
        if type(self.declaration) is not ChannelRouteIntent:
            raise TypeError("channel constraint requires an exact source declaration")
        self.declaration.assert_authoritative()
        if (type(self.y_low_nm) is not int or type(self.y_high_nm) is not int
                or self.y_low_nm >= self.y_high_nm):
            raise ValueError("channel requires two distinct placed row edges")
        if self.declaration.centered_nets and type(self.axis_x2_nm) is not int:
            raise ValueError("centered channel spans require a bound shared placement axis")


RouteConstraint = Union[
    LayerSet, Width, Mirror, Spacing, Shield, Length, TrackOrder, Access,
    Direction, TrunkLayer, Via, ChannelRoute
]

# The CLOSED set. Adding to this set is the rare, deliberate act that also requires an
# engine change; everything else (new primitives) lowers onto these.
CONSTRAINT_KINDS: FrozenSet[str] = frozenset(
    {
        "layer_set",
        "width",
        "mirror",
        "spacing",
        "shield",
        "length",
        "track_order",
        "access",
        "direction",
        "trunk_layer",
        "via",
        "channel_route",
    }
)


def is_hard(c: RouteConstraint) -> bool:
    """Is constraint ``c`` HARD (a required obligation the capability handshake
    gates on) as opposed to SOFT (an advisory bias that never gates)?

    Reads the ``hardness`` FIELD (never the kind), so the gate/engine classify a
    constraint by its declared hardness without a per-kind branch. A hand-built
    or legacy constraint with no ``hardness`` attribute is treated HARD (the
    conservative default -- an unrecognised constraint is an obligation, not a
    preference)."""
    return str(getattr(c, "hardness", HARD)) == HARD
