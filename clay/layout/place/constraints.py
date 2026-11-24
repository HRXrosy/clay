"""The CLOSED set of generic PLACEMENT constraints the faithful placer consumes.

The placement dual of :mod:`clay.layout.route.constraints` (spec §4.2, P1). Before this
module the placement side had NO closed vocabulary -- it drove the solver through two
divergent loose forms (``place_envelope_min`` kwargs + ``annotate_to_solver.
SolverConstraints``). This is the single closed set: every placement-organizational
intent (@array / @common_centroid / @guard_ring / @anchor / @rail_affinity / @band /
@order / @abut / ...) lowers to one or more of THESE, and the placer satisfies the
CONSTRAINT TYPES -- never the intents, never a circuit role.

  * adding a new intent primitive that maps onto existing constraint types
    => a new lowering only; the placer does NOT change.
  * a genuinely-new constraint type (rare) => one frozen dataclass here AND one new
    "how to satisfy it" placer branch -- a single, well-defined place.

Each constraint is a frozen dataclass (hashable, no behaviour). Device / group / net
labels are DATA (topology identifiers); a constraint NEVER carries a coordinate -- the
reference-coupled ``relgraph.ConstraintGraph`` (which stores exact dx/dy) is a leakage
BACK-derived graph and must NOT be reused as this forward contract.

Hardness axis (mirrors route.constraints): HARD constraints are obligations the
capability handshake gates on (an unsupported HARD kind STOPS before emit); SOFT
constraints are advisory biases (a pull the placer honours where it can, never a stop).
Defaults preserve current behaviour: structural constraints HARD; the edge/channel pulls
SOFT (an @anchor / @rail_affinity on a backend that cannot satisfy it is a reported
UNSUPPORTED / advisory, never a hard crash).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Mapping, Optional, Tuple

# hardness axis (identical semantics to route.constraints.HARD/SOFT).
HARD = "hard"
SOFT = "soft"


@dataclass(frozen=True)
class RigidMacro:
    """Group ``id`` is an OPAQUE rigid tile of extent ``w_um`` x ``h_um`` -- the placer
    translates it but never reshapes it. Wraps the existing ``Group`` / ``ArrayTile``
    (a shared-diffusion array is a frozen-interior rigid macro). ``interior`` is an
    opaque tag map (``{}`` = an atomic leaf); it carries NO coordinate. From @array /
    a compact matched MOS group."""
    id: str
    w_um: float
    h_um: float
    interior: Mapping[str, object] = field(default_factory=dict)
    kind: str = "rigid_macro"
    hardness: str = HARD


@dataclass(frozen=True)
class RelativeOrder:
    """Group ``a`` sits ``rel`` of group ``b`` in the realised placement -- ``rel`` in
    {``left_of``, ``above``}. From @order(a, b, rel). Names are DATA."""
    a: str
    b: str
    rel: str
    kind: str = "relative_order"
    hardness: str = HARD


@dataclass(frozen=True)
class Abutment:
    """Groups ``a`` and ``b`` are placed edge-adjacent, the facing gap tightened to the
    DRC-minimum. From @abut(a, b)."""
    a: str
    b: str
    kind: str = "abutment"
    hardness: str = HARD


@dataclass(frozen=True)
class Symmetry:
    """The listed ``pairs`` of groups are mirror-symmetric about one shared axis on
    equal rows (cross-group reflection); ``axis_offset_by_id`` binds each hosting
    group's within-group axis; ``mode`` is the axis-binding mode
    (``shared_hard`` / ``centered_soft``). From @common_centroid / a circuit
    SymmetryGroup. The most mature placement constraint (``symmetry_pairs`` /
    ``axis_groups`` today)."""
    pairs: Tuple[Tuple[str, str], ...]
    axis_offset_by_id: Tuple[Tuple[str, float], ...] = ()
    mode: str = "shared_hard"
    kind: str = "symmetry"
    hardness: str = HARD


@dataclass(frozen=True)
class KeepTogether:
    """Members ``members`` must stay one contiguous region (a bounding hull disjoint
    from every other cluster), padded ``pad_um`` at the island boundary. The
    @guard_ring keep-together obligation (``disjoint_clusters`` + ``island_pads``
    today)."""
    members: Tuple[str, ...]
    pad_um: float = 0.0
    kind: str = "keep_together"
    hardness: str = HARD


@dataclass(frozen=True)
class RegionAffinity:
    """Group ``id`` belongs to floorplan region / band ordinal ``region`` (a group in a
    lower band sits entirely below every group in a higher band). Unifies
    ``band_of_group`` (@place_group ROWS / @band). ``region`` is a topology ordinal,
    never a coordinate."""
    id: str
    region: int
    kind: str = "region_affinity"
    hardness: str = HARD


@dataclass(frozen=True)
class Alignment:
    """Groups ``a`` and ``b`` share an ``axis`` (``x`` = same vertical centreline,
    ``y`` = same horizontal centreline); ``ref`` selects the reference edge / centre
    (``center`` default). NEW forward primitive (P8): today only a reference-coupled
    ``align:x/y`` exists in the back-derived relgraph -- there is no forward alignment
    term reaching the placer."""
    a: str
    b: str
    axis: str
    ref: str = "center"
    kind: str = "alignment"
    hardness: str = HARD


@dataclass(frozen=True)
class EdgeAffinity:
    """Group ``id`` is pulled toward cell ``edge`` (``left`` / ``right`` / ``top`` /
    ``bottom``) -- edge-access placement. Unifies @anchor (routing-owed edge pin) and
    @rail_affinity (pin a polarity cluster to its supply-rail edge). A SOFT pull (never
    a hard block): a backend that cannot satisfy ``edge_affinity`` reports UNSUPPORTED,
    it does not crash. From @anchor / @rail_affinity."""
    id: str
    edge: str
    kind: str = "edge_affinity"
    hardness: str = SOFT


@dataclass(frozen=True)
class Channel:
    """Reserve a device-free routing corridor of ``width_um`` along ``orient``
    (``vertical`` / ``horizontal``), placed ``after`` a named group (``""`` = the
    cell's reserved corridor). A SOFT reservation. From @channel."""
    orient: str
    width_um: float
    after: str = ""
    kind: str = "channel"
    hardness: str = SOFT


@dataclass(frozen=True)
class Keepout:
    """A generic rectangular device-exclusion region in cell coordinates
    ``(x0, y0)-(x1, y1)`` (um). The one place a placement constraint may carry a
    rectangle -- an explicit designer keepout, never a back-derived reference box."""
    x0: float
    y0: float
    x1: float
    y1: float
    kind: str = "keepout"
    hardness: str = HARD


@dataclass(frozen=True)
class Spacing:
    """Minimum centre-clear spacing ``min_um`` between groups ``a`` and ``b`` (above the
    always-on NoOverlap gap). The placement dual of route ``Spacing``. NEW."""
    a: str
    b: str
    min_um: float
    kind: str = "spacing"
    hardness: str = HARD


PlacementConstraint = (
    RigidMacro
    | RelativeOrder
    | Abutment
    | Symmetry
    | KeepTogether
    | RegionAffinity
    | Alignment
    | EdgeAffinity
    | Channel
    | Keepout
    | Spacing
)

# The CLOSED placement set. Kept in lock-step with
# :data:`clay.layout.intent_impl.PLACEMENT_REQUEST_KINDS` (a test asserts equality) --
# adding to this set is the rare, deliberate act that also requires a placer branch.
PLACEMENT_CONSTRAINT_KINDS: FrozenSet[str] = frozenset(
    {
        "rigid_macro",
        "relative_order",
        "abutment",
        "symmetry",
        "keep_together",
        "region_affinity",
        "alignment",
        "edge_affinity",
        "channel",
        "keepout",
        "spacing",
    }
)


def is_hard(c: object) -> bool:
    """Is placement constraint ``c`` HARD (the handshake gates on it) rather than SOFT
    (an advisory bias)? Reads the ``hardness`` FIELD, never the kind, so the gate
    classifies a constraint without a per-kind branch. A hand-built constraint with no
    ``hardness`` attribute is treated HARD (the conservative default)."""
    return str(getattr(c, "hardness", HARD)) == HARD
