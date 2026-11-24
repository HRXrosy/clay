"""RoutingIntent -- the declarative, design-derived, leakage-free routing spec.

Pure data (no behaviour). The OPEN vocabulary of routing-intent primitives lives here
as fields; each field lowers (see :mod:`clay.layout.route.lower`) to the CLOSED set of
:mod:`clay.layout.route.constraints`. The faithful router consumes the constraints, so a
new primitive = a new field + a new lowering, NOT an engine change.

LEAKAGE DISCIPLINE (the layout leakage incident, transposed to routing): a routing-intent
value is legitimate only if it is DESIGN-DERIVED (net role / current / matching group /
netlist topology) or an explicit don't-care. A value MEASURED from the reference layout
and replayed (a specific wire width in um, a specific track index, a wire coordinate) is
LEAKAGE -- the routing analogue of memorising reference device coordinates. Every
declaration therefore carries an :class:`IntentSource` so the leakage gate can count
memorised-reference values (target: 0 in the final intent; the leakage-full ANCHOR is
allowed to be all 'reference' as the degradation START, then must degrade to 0).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, FrozenSet, List, Optional, Tuple

from clay.layout.route.constraints import Layer


class IntentSource(Enum):
    """Provenance of a routing-intent value -- drives the leakage gate."""
    DESIGN = "design"        # derived from netlist / net role / current / matching -> legitimate
    DONT_CARE = "dont_care"  # the router is free here -> legitimate (stores no value)
    REFERENCE = "reference"  # MEASURED from the reference layout -> LEAKAGE unless it is the anchor


@dataclass(frozen=True)
class Declared:
    """A single intent value with its provenance (so leakage is countable)."""
    value: object
    source: IntentSource = IntentSource.DESIGN


class TrunkPreferenceError(ValueError):
    """A declared soft trunk preference is malformed."""


@dataclass(frozen=True)
class TrunkPreference:
    """Unambiguous value object for one soft trunk layer/orientation preference.

    A GDS ``Layer`` is itself a two-item tuple, so representing
    ``(layer, orient)`` with another tuple is ambiguous.  This named value keeps
    the public ``RoutingIntent`` surface compatible with both role-name and GDS
    tuple layers.
    """

    layer: Layer
    orient: str = ""

    def __post_init__(self) -> None:
        axis = str(self.orient).strip().upper()
        if axis not in {"", "H", "V"}:
            raise TrunkPreferenceError(
                f"trunk orientation must be H, V, or blank; got {self.orient!r}"
            )
        object.__setattr__(self, "orient", axis)


@dataclass
class RoutingIntent:
    """Declarative routing spec for one cell. Every field is optional; an absent field
    means 'the router is free' (don't-care). Each per-net value is a :class:`Declared`
    so provenance/leakage is tracked.

    Open vocabulary -- add a primitive by adding a field here + a lowering in
    :mod:`clay.layout.route.lower` + an invariant in route_invariance. The engine and the
    closed constraint set do not change for primitives that map onto existing constraints.
    """
    cell: str = ""

    # @route_layer / @on_layer : net -> allowed layer set
    net_layers: Dict[str, Declared] = field(default_factory=dict)        # value: FrozenSet[Layer]
    # @wire_width / EM width : net -> width (um)
    widths: Dict[str, Declared] = field(default_factory=dict)            # value: float
    # @mirror_route / diff pairs : (net_a, net_b, axis_x_um)
    mirror_pairs: List[Declared] = field(default_factory=list)           # value: (str,str,float)
    # @shield : (net, by, min_um)
    shields: List[Declared] = field(default_factory=list)                # value: (str,str,float)
    # @match_length : (net_a, net_b, tol_rel)
    length_pairs: List[Declared] = field(default_factory=list)           # value: (str,str,float)
    # @route_order : (channel, (nets...))
    track_orders: List[Declared] = field(default_factory=list)           # value: (str, tuple)
    # @access : (net, device, terminal, side)
    accesses: List[Declared] = field(default_factory=list)               # value: (str,str,str,str)
    # @via_stack / @max_vias : net -> policy
    vias: Dict[str, Declared] = field(default_factory=dict)              # value: str
    # @route_direction (declared face): the design-side HVH direction-discipline
    # weight, wired to lower_intent(direction_weight=...). A step-COST knob (how
    # much direction style is worth, in base-step-cost units), NEVER a measured
    # reference value -- which is why it is a plain float, not a Declared, and
    # does not enter leakage_count. 0.0 (default) lowers to NO Direction
    # constraint: byte-stable. Per-layer style-EXTRACTED axes do not ride this
    # knob; they travel as explicit Direction constraints (the documented
    # override face of constraints.Direction) on the self-declaring style path.
    direction_weight: float = 0.0
    # @trunk : net -> soft preferred long-run layer (upper transport stays legal).
    # Appended after the original fields to preserve RoutingIntent's positional ABI.
    trunk_layers: Dict[str, Declared] = field(default_factory=dict)      # value: Layer | TrunkPreference
    # Named-net same-layer metal clearance; no shield conductor is implied.
    spacing_pairs: List[Declared] = field(default_factory=list)         # value: (str,str,float)

    def leakage_count(self) -> int:
        """How many intent values are memorised-from-reference (IntentSource.REFERENCE).

        The leakage gate: this must be 0 in the FINAL (degraded) intent. The anchor
        (self-consistent reproduction) is allowed to be non-zero as the degradation start.
        """
        n = 0
        for d in self.widths.values():
            if d.source is IntentSource.REFERENCE:
                n += 1
        for d in self.net_layers.values():
            if d.source is IntentSource.REFERENCE:
                n += 1
        for d in self.trunk_layers.values():
            if d.source is IntentSource.REFERENCE:
                n += 1
        for seq in (self.mirror_pairs, self.shields, self.spacing_pairs, self.length_pairs,
                    self.track_orders, self.accesses):
            for d in seq:
                if d.source is IntentSource.REFERENCE:
                    n += 1
        for d in self.vias.values():
            if d.source is IntentSource.REFERENCE:
                n += 1
        return n

    def is_leakage_free(self) -> bool:
        return self.leakage_count() == 0
