"""Clay layout-intent contract (v1): net-level parasitic intent.

Declares the net-level layout intent for a cell: which nets are large-swing
(``dynamic``) aggressors, which are offset-critical (``sensitive``) victims, the
differential pairs that must stay matched, the parasitic budgets (RELATIVE
imbalance tolerances, never absolute), and shielding requirements. Most of this
is derived from the circuit's functional structure; designer annotations
override/extend it.

This is the INPUT layer of the contract:
  * ``intent_obligations.check_layout_intent`` turns the budgets into
    verification obligations against a real PEX extraction;
  * the ``clay.repair`` ``LAYOUT_AWARE`` operators repair a violated budget.

It never changes circuit topology -- it is intent about the layout, not the
netlist.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import ClassVar, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from clay.layout.spatial.mosaic_guest_slot import MosaicGuestSlot, replay_mosaic_guest_slot_inventory
from clay.layout.spatial.mosaic_boundary_joins import (
    MosaicBoundaryJoinIntent, replay_mosaic_boundary_join_inventory, merge_mosaic_boundary_join_inventories,
)
from clay.core_contract import VerificationStatus
from clay.layout.bank_interconnect import (
    BankInterconnectPlan, bank_interconnect_from_attributes,
    replay_bank_interconnect_plan,
)
from clay.layout.mim_row_bus import MimRowBusPlan
from clay.layout.unit_grid_pattern import UnitGridPattern, validate_array_unit_pattern
from clay.layout.net_match_scope import (
    NetMatchScope, match_scopes_from_declarations, merge_match_scopes,
    placement_diff_pairs as _placement_diff_pairs, replay_match_scopes, scope_digest_data,
)
from clay.layout.net_spacing import NetSpacingIntent, canonical_net_spacings
from clay.layout.channel_intent import (
    ChannelRouteIntent, canonical_channel_routes, channel_routes_from_declarations,
)
from clay.layout.imported_route_intent import (
    ImportedLayoutIntent, RouteTailIntent, canonical_imported_routes,
    imported_routes_from_declarations, merge_imported_routes,
)
from clay.ir import CellDef, ParsedLayoutIntentDecl
from clay.layout.spatial.intent_projection import (
    AggregateAlignmentIntent,
    AggregateOrderIntent,
    ReferenceConditionedPlacementObligation,
    DiscreteMosaicIntent,
    HierarchicalMacroDagIntent,
    MosaicCenterAlignmentIntent,
    MosaicOccurrence,
    replay_discrete_mosaic_inventory,
    replay_aggregate_alignment_inventory,
    replay_aggregate_order_inventory,
    replay_reference_conditioned_placement_obligation_inventory,
    replay_hierarchical_macro_dag_intent,
    replay_mosaic_center_alignment_inventory,
)

# Default RELATIVE-imbalance budgets (relative thresholds only -- never absolute,
# per the project-wide consistency rule).
_DEFAULT_MATCH_TOL = 0.10  # matched-net total-cap balance, per diff pair
_DEFAULT_KICKBACK_TOL = 0.15  # dynamic-line -> input kickback imbalance
_DEFAULT_CLKREGEN_TOL = 0.10  # clk coupling balance to the regen pair

_BUDGET_KINDS = frozenset(
    {"matched_cc_balance", "kickback", "clk_regen", "parasitic_cap"}
)

# Supply / ground RAIL net names (lower-cased). A `@budget(parasitic_cap, net,
# rail)` whose second net is one of these is a SINGLE-ENDED budget against a
# reference rail (e.g. `parasitic_cap(vout, vss)`), NOT a differential node-cap
# balance between two symmetric signal nets. A single-ended budget must never be
# scored with the differential pair-imbalance metric (a rail and a signal node
# are asymmetric by design -- their total caps differ by orders of magnitude, so
# pair-imbalance would report a meaningless ~1.0 false-fail). See
# `clay.layout.intent_obligations` for how the single-ended case is measured.
_RAIL_NETS = frozenset(
    {
        "gnd",
        "vdd",
        "vddd",
        "vdda",
        "0",
        "vss",
        "vssd",
        "vssa",
        "vsub",
        "vsubs",
        "vnw",
        "vpw",
        "avdd",
        "avss",
        "dvdd",
        "dvss",
    }
)


def _is_rail(net: str) -> bool:
    """True iff ``net`` names a supply/ground RAIL (case-insensitive)."""
    return str(net).lower() in _RAIL_NETS


def parasitic_cap_is_single_ended(pair: Tuple[str, str]) -> bool:
    """True iff a ``parasitic_cap`` budget over ``pair`` is SINGLE-ENDED.

    A ``parasitic_cap`` budget is single-ended when exactly one of its two nets
    is a supply/ground rail: the budget then constrains how the (single) signal
    net references that rail, NOT a symmetry between two signal nets. When BOTH
    nets are signal nets it is a genuine differential node-cap balance (measured
    by the shared per-pair imbalance metric). When BOTH are rails the budget is
    degenerate and also treated single-ended (no symmetry obligation).
    """
    a, b = pair
    return _is_rail(a) or _is_rail(b)


# Source-level `@budget(...)` kinds Clay has NO PEX measurement for yet (no
# resistance-imbalance metric exists in `comparator_metrics`). These are
# recorded on the intent for provenance (`LayoutIntent.resistance_budgets`) but
# are NOT turned into a measured obligation -- a budget with no honest metric
# would otherwise become a permanent UNMEASURABLE block on every sign-off, which
# is worse than recording it as a declared-but-unverified intent.
_RESISTANCE_BUDGET_KINDS = frozenset({"series_res"})


@dataclass(frozen=True)
class ParasiticBudget:
    """A relative-imbalance budget on a coupling relationship.

    ``kind`` is one of matched_cc_balance / kickback / clk_regen; ``pair`` is the
    two nets whose coupling must balance; ``rel_tol`` is the relative-imbalance
    budget in [0, 1].

    Comparator-specific parameters (kickback aggressor list, clk_regen clock net)
    were originally hardcoded in :mod:`clay.layout.comparator_metrics`. As of v2
    those drivers move ONTO the budget itself so the same budget kind can be
    instantiated for any cell category whose dynamic aggressor is not ``clk`` and
    whose kickback victims are not ``net0``/``net1``:

      * ``aggressors`` -- (kickback only) which nets to SUM as aggressors against
        the pair. ``None`` means "use the owning intent's ``dynamic_nets``"
        (resolved at obligation-check time); an empty tuple means "no aggressors"
        (so the budget is intentionally vacuous, not silently using a default).
      * ``victim_set`` -- (kickback only) victim nets to score. ``None`` means
        ``pair`` itself; supply a longer tuple to score every pair-position
        against the same aggressor set.
      * ``clock_net`` -- (clk_regen only) the aggressor net coupling into the
        regenerative pair. Defaults to ``"clk"`` ONLY for backward compatibility
        of the comparator deriver; a fast-strobe or differential-clock comparator
        should set ``"strobe"`` / ``"clkp"`` etc.
      * ``source`` / ``profile_id`` -- provenance ("auto", "category-profile",
        "declared", "designer-override", ...) so a signoff record can tell where
        a budget came from. They never influence the verdict; they ARE part of
        ``intent_digest`` so a record's provenance is hash-stable.

    All defaults are picked so every existing constructor call
    (``ParasiticBudget(kind, pair, rel_tol)``) stays valid and the legacy
    behaviour is byte-identical when the new fields are not set.
    """

    kind: str
    pair: Tuple[str, str]
    rel_tol: float
    # ── v2: per-budget metric drivers (back-compat defaults below) ──────────
    aggressors: Optional[Tuple[str, ...]] = field(default=None, compare=False)
    victim_set: Optional[Tuple[str, ...]] = field(default=None, compare=False)
    clock_net: Optional[str] = field(default=None, compare=False)
    source: str = field(default="auto", compare=False)
    profile_id: Optional[str] = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if self.kind not in _BUDGET_KINDS:
            raise ValueError(
                f"unknown budget kind {self.kind!r}; "
                f"expected one of {sorted(_BUDGET_KINDS)}"
            )
        if not (0.0 < self.rel_tol < 1.0):
            raise ValueError(
                f"rel_tol must be a RELATIVE fraction in (0,1), got {self.rel_tol!r}"
            )
        a, b = self.pair
        if a == b:
            raise ValueError(
                f"a budget pair must name two distinct nets, got {self.pair!r}"
            )


@dataclass(frozen=True)
class Shield:
    """Keep ``shield_net`` between ``aggressor`` and ``victim`` routing."""

    aggressor: str
    victim: str
    shield_net: str = "gnd"

    def __post_init__(self) -> None:
        names = (self.aggressor, self.victim, self.shield_net)
        if any(
            type(name) is not str
            or not name
            or name != name.strip()
            or len(name) > 256
            for name in names
        ):
            raise TypeError("shield nets must be exact bounded strings")
        if len(set(names)) != 3:
            raise ValueError("shield aggressor, victim, and shield net must be distinct")

    def canonical_data(self) -> dict[str, str]:
        self.__post_init__()
        return {
            "aggressor": self.aggressor,
            "victim": self.victim,
            "shield_net": self.shield_net,
        }

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.canonical_data(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        return hashlib.sha256(payload).hexdigest()


def replay_shield_intent(value: object) -> Shield:
    if type(value) is not Shield:
        raise TypeError("shield declaration must be an exact Shield")
    return Shield(
        aggressor=value.aggressor,
        victim=value.victim,
        shield_net=value.shield_net,
    )


def replay_shield_inventory(value: object) -> Tuple[Shield, ...]:
    if type(value) is not tuple:
        raise TypeError("shield inventory must be an exact tuple")
    if len(value) > 256:
        raise ValueError("shield inventory is too large")
    declarations = tuple(replay_shield_intent(item) for item in value)
    if len({item.sha256 for item in declarations}) != len(declarations):
        raise ValueError("duplicate shield declaration")
    return declarations


_MAX_ADJACENT_RATIO_TERM = 1_000_000


@dataclass(frozen=True)
class AdjacentIntent:
    """One directed, coordinate-free maximum edge-separation declaration."""

    source: str
    target: str
    max_separation_multiplier: Fraction

    def __post_init__(self) -> None:
        for name in (self.source, self.target):
            if (
                type(name) is not str
                or not name
                or name != name.strip()
                or len(name) > 256
            ):
                raise TypeError("adjacent endpoints must be exact bounded strings")
        if self.source == self.target:
            raise ValueError("adjacent endpoints must be distinct")
        if type(self.max_separation_multiplier) is not Fraction:
            raise TypeError("adjacent separation multiplier must be an exact Fraction")
        if not 0 < self.max_separation_multiplier <= 256:
            raise ValueError("adjacent separation multiplier must be in (0, 256]")
        if (
            self.max_separation_multiplier.numerator > _MAX_ADJACENT_RATIO_TERM
            or self.max_separation_multiplier.denominator > _MAX_ADJACENT_RATIO_TERM
        ):
            raise ValueError("adjacent separation multiplier terms must be bounded")

    def canonical_data(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "source": self.source,
            "target": self.target,
            "max_separation_numerator": self.max_separation_multiplier.numerator,
            "max_separation_denominator": self.max_separation_multiplier.denominator,
        }

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.canonical_data(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        return hashlib.sha256(payload).hexdigest()


def replay_adjacent_intent(value: object) -> AdjacentIntent:
    if type(value) is not AdjacentIntent:
        raise TypeError("adjacent declaration must be an exact AdjacentIntent")
    return AdjacentIntent(
        source=value.source,
        target=value.target,
        max_separation_multiplier=value.max_separation_multiplier,
    )


def replay_adjacent_inventory(value: object) -> Tuple[AdjacentIntent, ...]:
    if type(value) is not tuple:
        raise TypeError("adjacent inventory must be an exact tuple")
    if len(value) > 4096:
        raise ValueError("adjacent inventory is too large")
    declarations = tuple(replay_adjacent_intent(item) for item in value)
    if declarations != tuple(
        sorted(declarations, key=lambda item: (item.source, item.target))
    ):
        raise ValueError("adjacent inventory must be in canonical order")
    if len({(item.source, item.target) for item in declarations}) != len(declarations):
        raise ValueError("duplicate directed adjacent declaration")
    return declarations


# The three layout-dependent-effect (LDE) metrics an ``@equal_*`` device-
# environment budget can constrain, measured by ``clay.layout.device_environment``:
#   * ``well_edge``           -- WPE: device active-bbox distance to enclosing-well edge
#   * ``od_length``           -- LOD/STI stress: diffusion extent along the channel axis
#   * ``od_neighbor_spacing`` -- OD-spacing stress: gap to the nearest other device's OD
_LDE_METRICS = frozenset({"well_edge", "od_length", "od_neighbor_spacing"})

# Surface keyword -> LDE metric. ``@equal_well_edge`` / ``@equal_od`` /
# ``@equal_od_spacing`` are the user-facing forms; each selects one metric.
_LDE_KIND_TO_METRIC = {
    "equal_well_edge": "well_edge",
    "equal_od": "od_length",
    "equal_od_spacing": "od_neighbor_spacing",
}

# Default RELATIVE tolerance for an LDE matching budget when the declaration
# omits one. A matched device pair holds its WPE / LOD / OD-spacing metric to
# well within this; an asymmetrically-placed pair blows past it. Relative only.
_DEFAULT_LDE_TOL = 0.10


@dataclass(frozen=True)
class LdeBudget:
    """A relative-imbalance budget on a per-device LAYOUT-DEPENDENT-EFFECT metric.

    Where :class:`ParasiticBudget` constrains a coupling-CAP relationship measured
    from a PEX extraction, an ``LdeBudget`` constrains a DEVICE-GEOMETRY metric
    (well-edge distance / OD length / OD-neighbour spacing) measured straight off
    the cell's terminal bboxes + layer polygons by
    :func:`clay.layout.device_environment.extract_device_environment`.

    ``pair`` names the two DEVICE labels (not nets) whose ``metric`` must match;
    ``metric`` is one of ``well_edge`` / ``od_length`` / ``od_neighbor_spacing``;
    ``rel_tol`` is the relative-imbalance budget in (0, 1). The obligation checker
    reduces the two devices' metric to ``|va - vb| / max(va, vb)`` (RELATIVE, so
    the dbu unit cancels and scaling the layout cannot change the verdict). When
    the metric is ``None`` for either device (the required layer was absent from
    the extraction) the obligation is UNMEASURABLE and withholds the proof token,
    never silently passing.
    """

    pair: Tuple[str, str]
    metric: str
    rel_tol: float
    source: str = field(default="declared", compare=False)

    def __post_init__(self) -> None:
        if self.metric not in _LDE_METRICS:
            raise ValueError(
                f"unknown LDE metric {self.metric!r}; "
                f"expected one of {sorted(_LDE_METRICS)}"
            )
        if not (0.0 < self.rel_tol < 1.0):
            raise ValueError(
                f"rel_tol must be a RELATIVE fraction in (0,1), got {self.rel_tol!r}"
            )
        a, b = self.pair
        if a == b:
            raise ValueError(
                f"an LDE budget pair must name two distinct devices, got {self.pair!r}"
            )


@dataclass(frozen=True)
class PlaceGroup:
    """A declared device-placement group from a `@place_group(...)` intent.

    Membership, priority and a ``rows=A..B`` packing-count range are generation
    directives.  Exact ``row=N`` / ``col=N`` values additionally make a final
    geometric claim and therefore become :class:`LayoutObligation` values.  All
    forms remain on :class:`LayoutIntent` so the placer and intent digest consume
    the same declaration.
    """

    group: str
    devices: Tuple[str, ...] = ()
    row_min: Optional[int] = None
    row_max: Optional[int] = None
    priority: Optional[str] = None
    # Optional single-column hint (the horizontal slot within the floorplan row).
    # When set, the placer orders this group's devices left-to-right by ``col``.
    # ``None`` leaves column ordering to the placer (label order). Sourced from a
    # ``col=N`` attribute on the ``@place_group`` declaration.
    col: Optional[int] = None
    # Optional selected internal packing row count.  Unlike ``row=N`` this is
    # not a floorplan coordinate: it is one concrete choice inside the
    # ``rows=A..B`` domain and is consumed by aggregate macro construction.
    pack_rows: Optional[int] = None
    # Shape contract for the group envelope.  ``None`` preserves the historical
    # rigid aggregate behaviour. ``deformable`` keeps member devices independently
    # placeable and permits another physical group to occupy whitespace inside their
    # union. ``relational`` likewise retains member-level placement, but asserts only
    # the authored membership and relations rather than a deformable envelope. This
    # is a discrete structural relation, never an observed coordinate.
    packing: Optional[str] = None

    def __post_init__(self) -> None:
        if self.packing not in {None, "rigid", "deformable", "relational"}:
            raise ValueError(
                "place_group packing must be exactly rigid, deformable, or relational"
            )


@dataclass(frozen=True, slots=True, order=True)
class RowBoundedCavityIntent:
    """Bind guest owners into one host's coordinate-free generated cavity.

    The boundary selectors are part of the type, not authored values.  A host
    contributes its own measurable atom-row footprints and envelope; no rail,
    reference coordinate, or numeric spacing can enter this declaration.
    """

    host: str
    guests: Tuple[str, ...]

    LOWER_BOUNDARY: ClassVar[str] = "maximal_span_host_atom_row_top_envelope"
    UPPER_BOUNDARY: ClassVar[str] = "host_envelope_top"

    def __post_init__(self) -> None:
        if type(self.host) is not str:
            raise TypeError("row-bounded cavity host must be an exact string")
        if not self.host or self.host != self.host.strip() or len(self.host) > 256:
            raise ValueError("row-bounded cavity host must be canonical")
        if type(self.guests) is not tuple:
            raise TypeError("row-bounded cavity guests must be an exact tuple")
        if not self.guests or len(self.guests) > 256:
            raise ValueError("row-bounded cavity requires a bounded guest set")
        if any(
            type(guest) is not str
            or not guest
            or guest != guest.strip()
            or len(guest) > 256
            for guest in self.guests
        ):
            raise ValueError("row-bounded cavity guests must be canonical strings")
        canonical_guests = tuple(sorted(self.guests))
        if len(set(canonical_guests)) != len(canonical_guests):
            raise ValueError("row-bounded cavity guests must be unique")
        if self.host in canonical_guests:
            raise ValueError("row-bounded cavity host cannot also be a guest")
        object.__setattr__(self, "guests", canonical_guests)

    def canonical_data(self) -> Dict[str, object]:
        return {
            "host": self.host,
            "guests": list(self.guests),
            "lower_boundary": self.LOWER_BOUNDARY,
            "upper_boundary": self.UPPER_BOUNDARY,
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self.canonical_data(),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("ascii")
        ).hexdigest()


def replay_row_bounded_cavity_inventory(
    value: object,
) -> Tuple[RowBoundedCavityIntent, ...]:
    """Replay a bounded, non-overlapping generated-cavity declaration set."""

    if type(value) is not tuple:
        raise TypeError("row-bounded cavity inventory must be an exact tuple")
    if len(value) > 256:
        raise ValueError("row-bounded cavity inventory is too large")
    rows = []
    for item in value:
        if type(item) is not RowBoundedCavityIntent:
            raise TypeError("row-bounded cavity inventory contains a foreign row")
        rows.append(RowBoundedCavityIntent(host=item.host, guests=item.guests))
    canonical = tuple(sorted(rows, key=lambda item: (item.host, item.guests)))
    if len({(item.host, item.guests) for item in canonical}) != len(canonical):
        raise ValueError("duplicate row-bounded cavity declarations are forbidden")
    guest_hosts = [
        (guest, item.host) for item in canonical for guest in item.guests
    ]
    if len({guest for guest, _host in guest_hosts}) != len(guest_hosts):
        raise ValueError("one guest cannot occupy two row-bounded cavities")
    return canonical


@dataclass(frozen=True)
class EdgeAlignmentIntent:
    """Second materialized edge is offset from the first, in nanometres."""

    first: str
    second: str
    edge: str
    offset_nm: int = 0

    def __post_init__(self) -> None:
        if any(type(value) is not str for value in (self.first, self.second, self.edge)):
            raise TypeError("edge alignment fields must be exact strings")
        if any(
            not value or value != value.strip() or len(value) > 256
            for value in (self.first, self.second)
        ):
            raise ValueError("edge alignment endpoints must be canonical")
        if self.first == self.second:
            raise ValueError("edge alignment endpoints must be distinct")
        if self.edge not in {"left", "right", "top", "bottom", "center_x", "center_y"}:
            raise ValueError(
                "edge alignment must name left, right, top, bottom, center_x, or center_y"
            )
        if type(self.offset_nm) is not int:
            raise TypeError("edge alignment offset must be integer nanometres")
        if self.second < self.first:
            first, second = self.second, self.first
            object.__setattr__(self, "first", first)
            object.__setattr__(self, "second", second)
            object.__setattr__(self, "offset_nm", -self.offset_nm)

    def canonical_data(self) -> Dict[str, str]:
        result = {"first": self.first, "second": self.second, "edge": self.edge}
        if self.offset_nm:
            result["offset_nm"] = self.offset_nm
        return result


@dataclass(frozen=True)
class GroupCenterAlignment:
    """Align one place-group center with the union of other place groups."""
    target: str
    groups: Tuple[str, ...]

    def __post_init__(self):
        if not self.target or not self.groups or self.target in self.groups:
            raise ValueError("group center alignment requires distinct place groups")
        object.__setattr__(self, "groups", tuple(sorted(set(self.groups))))

    def canonical_data(self):
        return {"target": self.target, "groups": list(self.groups)}


@dataclass(frozen=True)
class GroupEdgeAlignment:
    """Align a named edge of two independently owned place-group envelopes."""
    first: str
    second: str
    edge: str

    def __post_init__(self):
        if any(type(v) is not str or not v or v != v.strip() for v in (self.first, self.second)):
            raise ValueError("group alignment requires canonical group names")
        if self.first == self.second or self.edge not in {"left", "right", "bottom", "top"}:
            raise ValueError("group alignment requires distinct groups and a valid edge")
        if self.second < self.first:
            first, second = self.second, self.first
            object.__setattr__(self, "first", first)
            object.__setattr__(self, "second", second)

    def canonical_data(self):
        return {"first": self.first, "second": self.second, "edge": self.edge}


@dataclass(frozen=True)
class GroupSeparation:
    """Order two complete group envelopes on a declared axis."""
    before: str
    after: str
    axis: str
    gap_nm: int = 0

    def __post_init__(self):
        if any(type(v) is not str or not v or v != v.strip() for v in (self.before, self.after)):
            raise ValueError("group separation requires canonical group names")
        if self.before == self.after or self.axis not in {"horizontal", "vertical"}:
            raise ValueError("group separation requires distinct groups and a valid axis")
        if type(self.gap_nm) is not int or not 0 <= self.gap_nm <= 1_000_000_000:
            raise ValueError("group separation gap must be a bounded nonnegative integer")

    def canonical_data(self):
        result = {"before": self.before, "after": self.after, "axis": self.axis}
        if self.gap_nm:
            result["gap_nm"] = self.gap_nm
        return result


@dataclass(frozen=True)
class ChannelAlignment:
    """Align the x centers of two source devices' single physical channels."""
    first: str
    second: str

    def __post_init__(self):
        if any(type(v) is not str or not v or v != v.strip() for v in (self.first, self.second)):
            raise ValueError("channel alignment requires canonical device names")
        if self.first == self.second:
            raise ValueError("channel alignment requires distinct devices")
        if self.second < self.first:
            first, second = self.second, self.first
            object.__setattr__(self, "first", first)
            object.__setattr__(self, "second", second)

    def canonical_data(self):
        return {"first": self.first, "second": self.second}


def replay_edge_alignment_inventory(
    value: object,
) -> Tuple[EdgeAlignmentIntent, ...]:
    if type(value) is not tuple:
        raise TypeError("edge alignment inventory must be an exact tuple")
    rows: List[EdgeAlignmentIntent] = []
    for item in value:
        if type(item) is not EdgeAlignmentIntent:
            raise TypeError("edge alignment inventory contains a foreign row")
        rows.append(EdgeAlignmentIntent(item.first, item.second, item.edge, item.offset_nm))
    replayed = tuple(rows)
    identities = tuple((item.first, item.second, item.edge) for item in replayed)
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate edge alignment declarations are forbidden")
    return replayed


@dataclass(frozen=True)
class TerminalAccessIntent:
    """Expose one logical terminal on a device-local envelope side."""

    device: str
    terminal: str
    side: str

    def __post_init__(self) -> None:
        if any(type(value) is not str for value in (self.device, self.terminal, self.side)):
            raise TypeError("terminal access fields must be exact strings")
        if (
            not self.device
            or self.device != self.device.strip()
            or len(self.device) > 256
        ):
            raise ValueError("terminal access device must be canonical")
        if self.terminal not in {"d", "g", "s", "b", "p", "n"}:
            raise ValueError("terminal access role is unsupported")
        if self.side not in {"left", "right", "top", "bottom"}:
            raise ValueError("terminal access side is unsupported")

    def canonical_data(self) -> Dict[str, str]:
        return {
            "device": self.device,
            "terminal": self.terminal,
            "side": self.side,
        }


def replay_terminal_access_inventory(
    value: object,
) -> Tuple[TerminalAccessIntent, ...]:
    if type(value) is not tuple:
        raise TypeError("terminal access inventory must be an exact tuple")
    rows = []
    for item in value:
        if type(item) is not TerminalAccessIntent:
            raise TypeError("terminal access inventory contains a foreign row")
        rows.append(TerminalAccessIntent(item.device, item.terminal, item.side))
    replayed = tuple(rows)
    identities = tuple((item.device, item.terminal) for item in replayed)
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate terminal access declarations are forbidden")
    return replayed


@dataclass(frozen=True)
class RouteGroupIntent:
    """Bind explicit physical placement ownership to hierarchical route ports.

    ``ports=external`` carries no named net or physical coordinate.  The exact
    port denominator is derived later from the cell topology: a member terminal
    is external exactly when its net is a top-cell port or reaches a device
    outside the group. A legacy declaration opts the physical ``@place_group``
    named by ``group`` into that derivation. ``place_groups`` instead carries an
    explicit, canonical union of two or more physical placement groups; semantic
    ``functional_group`` labels never gain physical authority implicitly.
    """

    group: str
    ports: str = "external"
    place_groups: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.group) is not str or type(self.ports) is not str:
            raise TypeError("route group fields must be exact strings")
        if (
            not self.group
            or self.group != self.group.strip()
            or len(self.group) > 256
        ):
            raise ValueError("route group name must be canonical and non-empty")
        if self.ports != "external":
            raise ValueError("route group ports must be exactly 'external'")
        if type(self.place_groups) is not tuple:
            raise TypeError("route group place_groups must be an exact tuple")
        if self.place_groups:
            if len(self.place_groups) < 2:
                raise ValueError(
                    "explicit route group place_groups requires at least two groups"
                )
            if any(type(name) is not str for name in self.place_groups):
                raise TypeError(
                    "route group place_groups entries must be exact strings"
                )
            if any(
                not name or name != name.strip() or len(name) > 256
                for name in self.place_groups
            ):
                raise ValueError(
                    "route group place_groups entries must be canonical"
                )
            if self.place_groups != tuple(sorted(set(self.place_groups))):
                raise ValueError(
                    "route group place_groups must be sorted and unique"
                )

    def canonical_data(self) -> Dict[str, object]:
        data: Dict[str, object] = {"group": self.group, "ports": self.ports}
        if self.place_groups:
            data["place_groups"] = list(self.place_groups)
        return data

    @property
    def sha256(self) -> str:
        encoded = json.dumps(
            self.canonical_data(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        return hashlib.sha256(encoded).hexdigest()


def replay_route_group_inventory(
    value: object,
) -> Tuple[RouteGroupIntent, ...]:
    """Replay the exact, bounded and unique route-group denominator."""

    if type(value) is not tuple:
        raise TypeError("route group inventory must be an exact tuple")
    if len(value) > 4096:
        raise ValueError("route group inventory is too large")
    replayed = []
    for item in value:
        if type(item) is not RouteGroupIntent:
            raise TypeError("route group inventory contains a foreign row")
        replayed.append(
            RouteGroupIntent(
                group=item.group,
                ports=item.ports,
                place_groups=item.place_groups,
            )
        )
    groups = tuple(item.group for item in replayed)
    if len(set(groups)) != len(groups):
        raise ValueError("duplicate route group declarations are forbidden")
    return tuple(replayed)


def place_group_floorplan_row(group: object) -> Optional[int]:
    """Return the ordinal ``row=N`` pin, never a ``rows=A..B`` count bound.

    Both surface forms share ``row_min``/``row_max`` in the historical IR.  An
    exact equal pair is the floorplan-coordinate form consumed by row-order
    placement/checking.  A non-degenerate range instead bounds how many rows the
    group's area packer may use; interpreting its lower bound as a global row
    index fabricates cross-group ordering obligations.
    """

    row_min = getattr(group, "row_min", None)
    row_max = getattr(group, "row_max", None)
    if row_min is None or row_max is None:
        return None
    if type(row_min) is not int or type(row_max) is not int:
        raise TypeError("floorplan row must be an exact integer")
    if row_min < 0 or row_max < row_min:
        raise ValueError("floorplan row domain must be ordered and non-negative")
    return row_min if row_min == row_max else None


@dataclass(frozen=True)
class ResistanceBudget:
    """A declared `@budget(series_res, a, b, tol)` recorded for provenance.

    Clay has no PEX resistance-imbalance metric, so a series-resistance budget
    cannot be honestly MEASURED yet. Rather than emit a permanently-unmeasurable
    obligation that would block every sign-off, the lowering records it here as a
    declared-but-unverified intent. ``rel_tol`` is a RELATIVE fraction.
    """

    pair: Tuple[str, str]
    rel_tol: float
    kind: str = "series_res"


@dataclass(frozen=True)
class LayoutObligation:
    """The result of checking one parasitic budget against an extraction.

    Shared contract between the obligation checker (which produces these) and the
    repair operators (which propose fixes for failing ones). ``measured_imbalance``
    and ``rel_tol`` are RELATIVE fractions.

    Four closed verification states, exposed through :attr:`status`:

      * ``"pass"`` / ``"fail"`` -- the budget WAS measured against the extraction
        and the measured RELATIVE imbalance is within / over ``rel_tol``. These
        are the only states the v1 checker produced and stay byte-compatible.
      * ``"unmeasurable"`` -- the extraction does not carry the net(s) this budget
        needs, so no honest imbalance exists. ``measured_imbalance`` is ``None``
        (NOT a misleading ``0.0``) and ``missing_nets`` names the absent nets.
        An unmeasurable obligation must NOT be read as a pass: an absent net means
        the layout was not even checked for this relationship, which the verdict
        treats as blocking (see :func:`~clay.layout.intent_obligations.layout_intent_verdict`).
      * ``"unsupported"`` -- the checker/materializer lacks the required device-
        class capability.  This is distinct from missing artifact geometry and is
        likewise blocking; it never certifies a pass.

    ``missing_nets`` is empty for a measured obligation. A measured obligation
    always carries a finite ``measured_imbalance``; an unmeasurable/unsupported
    one carries ``None``.
    """

    kind: str
    pair: Tuple[str, str]
    measured_imbalance: Optional[float]
    rel_tol: float
    missing_nets: Tuple[str, ...] = ()
    # Optional closed-set status for capability outcomes which cannot be inferred
    # from the legacy measured/None encoding.  Existing constructors leave this
    # unset and retain byte-compatible pass/fail/unmeasurable behaviour.
    verification_status: Optional[VerificationStatus] = None
    # Exact source declarations covered by this obligation.  Matching coverage
    # requires equality of this complete set; a one-direction proxy is invalid.
    declaration_ids: Tuple[str, ...] = ()
    # Named mechanical sub-measurements (centroid/orientation/footprint for a
    # matching obligation).  A dict is used only as immutable result evidence;
    # callers receive a fresh mapping from each checker.
    measurements: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.verification_status is not None and not isinstance(
            self.verification_status, VerificationStatus
        ):
            object.__setattr__(
                self,
                "verification_status",
                VerificationStatus(str(self.verification_status)),
            )
        if self.kind == "matching" and self.verification_status is not None:
            status = self.verification_status
            closed = {
                VerificationStatus.PASS,
                VerificationStatus.FAIL,
                VerificationStatus.UNMEASURABLE,
                VerificationStatus.UNSUPPORTED,
            }
            if status not in closed:
                raise ValueError(
                    f"matching obligation requires a closed status, got {status.value!r}"
                )
            measured = self.measured_imbalance
            if status in (VerificationStatus.PASS, VerificationStatus.FAIL):
                if measured is None or not math.isfinite(float(measured)):
                    raise ValueError(
                        "measured matching pass/fail requires finite "
                        "measured_imbalance"
                    )
                inferred = (
                    VerificationStatus.PASS
                    if float(measured) <= float(self.rel_tol)
                    else VerificationStatus.FAIL
                )
                if status is not inferred:
                    raise ValueError(
                        "matching verification_status conflicts with "
                        "measured_imbalance/rel_tol"
                    )
            elif measured is not None:
                raise ValueError(
                    f"matching {status.value} must not carry measured_imbalance"
                )
        if len(set(self.declaration_ids)) != len(self.declaration_ids):
            raise ValueError("LayoutObligation declaration_ids must be unique")

    @property
    def unmeasurable(self) -> bool:
        """True iff this budget could not be measured (a net was absent)."""
        return self.status == VerificationStatus.UNMEASURABLE.value

    @property
    def status(self) -> str:
        if self.verification_status is not None:
            return self.verification_status.value
        if self.measured_imbalance is None:
            return VerificationStatus.UNMEASURABLE.value
        return (
            VerificationStatus.PASS.value
            if self.measured_imbalance <= self.rel_tol
            else VerificationStatus.FAIL.value
        )

    @property
    def over_budget(self) -> float:
        """How far the imbalance exceeds the budget (0 when within budget).

        An unmeasurable obligation has no imbalance to exceed the budget, so it
        reports ``0.0`` here -- ``over_budget`` ranks the SEVERITY of measured
        overshoot for repair, and is not the channel that flags unmeasurability
        (``status`` / :attr:`unmeasurable` is). The verdict blocks on it anyway.
        """
        if self.measured_imbalance is None:
            return 0.0
        return max(0.0, self.measured_imbalance - self.rel_tol)


@dataclass(frozen=True)
class GuardRingIntent:
    """A declared ``@guard_ring(group=(...), well=<p|n>, tie=<net>)`` cluster ring.

    A first-class, net/cluster-level intent: wrap the named DEVICE cluster in ONE
    PDK-rule-driven tap ring, strapped to ``tie_net``. ``members`` are the device
    instance labels the ring encloses; ``well`` is the declared ring polarity
    (``"p"`` substrate-tap / ``"n"`` nwell-tap) the materialized ring MUST draw;
    ``tie_net`` is the supply/body net the ring is tied to (``""`` when a designer
    declares no tie -- the checker then reads the drawn tap net without an equality
    obligation).

    This is the DECLARED obligation; the drawn ring is the EVIDENCE. ``route_cell``
    materializes it through ``clay.layout.extract.realize_render.group_guard_rings``
    (which resolves the drawn polarity from the device KIND and the drawn tap net
    from the device body net -- pure design data, no reference coordinate), and the
    independent ``clay.layout.route.guard_ring_check`` checker verifies the drawn
    polarity == ``well`` and the union-find tap net == ``tie_net``.
    """

    members: Tuple[str, ...]
    well: str
    tie_net: str = ""
    hug: str = ""
    role: str = ""
    enclose: str = ""
    pack: str = ""

    def __post_init__(self) -> None:
        if type(self.members) is not tuple:
            raise TypeError("guard-ring members must be an exact tuple")
        if (
            not self.members
            or any(
                type(member) is not str or not member or member != member.strip()
                for member in self.members
            )
            or len(set(self.members)) != len(self.members)
        ):
            raise ValueError("guard-ring members must be non-empty and unique")
        for field_name in ("well", "tie_net", "hug", "role", "enclose", "pack"):
            value = getattr(self, field_name)
            if type(value) is not str:
                raise TypeError(f"guard-ring {field_name} must be an exact string")
            if value != value.strip():
                raise ValueError(f"guard-ring {field_name} must be canonical")
        if self.well not in ("p", "n"):
            raise ValueError("guard-ring well must be 'p' or 'n'")
        if self.hug != self.hug.lower() or self.role != self.role.lower():
            raise ValueError("guard-ring hug/role must be lowercase canonical")
        if self.pack not in ("", "hierarchical"):
            raise ValueError("guard-ring pack must be empty or 'hierarchical'")
        if self.hug == "tight" and not self.tie_net:
            raise ValueError("tight guard-ring declaration requires a tie net")
        if self.hug == "tight" and self.role not in ("tap", "isolation"):
            raise ValueError(
                "tight guard-ring declaration requires role='tap' or 'isolation'"
            )

    def canonical_data(self) -> Dict[str, object]:
        """Exact declaration payload consumed by materializer and checker."""

        payload = {
            "members": list(self.members),
            "well": self.well,
            "tie_net": self.tie_net,
            "hug": self.hug,
            "role": self.role,
            "enclose": self.enclose,
        }
        if self.pack:
            payload["pack"] = self.pack
        return payload

    @property
    def sha256(self) -> str:
        encoded = json.dumps(
            self.canonical_data(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Stage-1 first-class analog-layout CRAFT intent (coordinate-free organizational
# intent inferred from netlist topology). Each is a declared obligation the
# materializer will later consume; at this stage they PARSE + LOWER only (no
# geometry / placer / router behaviour). Every field is topology-derived (group
# label / device kind / M / shared nets / port role) -- never a coordinate.
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ArrayIntent:
    """`@array(..., transforms=..., align=..., pin_facing=...)`.

    The named ``group`` is materialized as one continuous shared-OD array (or
    discrete fingers), with ``rows`` row(s) and the given ``order`` interleave
    (``ABBA`` / ``AABB`` / ``blocked`` / ``ratio``). Finger counts come from each
    member's ``M`` (not stored here). ``style`` / ``order`` are DATA.
    """

    group: str
    style: str = ""
    rows: int = 1
    order: str = ""
    transforms: Tuple[str, ...] = ()
    align: str = ""
    pin_facing: str = ""
    bank_interconnect: BankInterconnectPlan | MimRowBusPlan | None = None
    unit_pattern: UnitGridPattern | None = None
    # Source-order ledger in the untransformed native row frame. Repeated
    # source owners receive consecutive physical-occurrence ordinals.
    occurrences: Tuple[MosaicOccurrence, ...] = ()
    # Explicit PDK contact coverage for a supported native diffusion row.
    # None preserves the PDK's original unmodified generation invocation.
    coverage_profile: Optional[Tuple[str, str]] = None
    access_profile: str = ""
    generation_profile: str = ""
    synthesis: str = ""
    tap_rail: str = ""

    def __post_init__(self) -> None:
        if type(self.synthesis) is not str:
            raise TypeError("array synthesis must be an exact string")
        if self.synthesis not in ("", "parameterized", "cmos_gate"):
            raise ValueError(f"unsupported array synthesis {self.synthesis!r}")
        if type(self.tap_rail) is not str:
            raise TypeError("array tap_rail must be an exact string")
        if self.tap_rail not in ("", "top", "bottom", "guard_ring"):
            raise ValueError(f"unsupported array tap_rail {self.tap_rail!r}")
        if self.synthesis and self.style != "native_diffusion":
            raise ValueError("array synthesis requires style=native_diffusion")
        if self.tap_rail and (self.style != "native_diffusion" or not self.synthesis):
            raise ValueError("array tap_rail requires parameterized native_diffusion")
        if self.synthesis and (self.coverage_profile is not None or self.access_profile or self.generation_profile):
            raise ValueError("parameterized native_diffusion cannot combine legacy access/generation profiles")
        if type(self.generation_profile) is not str:
            raise TypeError("native generation profile must be an exact string")
        if self.generation_profile not in ("", "sky130_lvt_single_440_v1", "sky130_lvt_single_440_output_m12_v1", "sky130_lvt_single_440_output_input_m12_v1"):
            raise ValueError("unsupported native generation profile")
        if self.generation_profile and (self.style != "native_diffusion" or
                self.coverage_profile is not None or self.access_profile or not self.occurrences):
            raise ValueError("native generation profile requires an explicit uncombined native occurrence ledger")
        if type(self.access_profile) is not str:
            raise TypeError("native diffusion access profile must be an exact string")
        if self.access_profile not in ("", "shared_source_li_m1", "shared_source_li_m1_far", "shared_cross_coupled_m1_v1"):
            raise ValueError("unsupported native diffusion access profile")
        if self.access_profile and self.style != "native_diffusion":
            raise ValueError("native access profile requires native_diffusion")
        if self.coverage_profile is not None:
            if (type(self.coverage_profile) is not tuple or len(self.coverage_profile) != 2
                    or any(type(value) is not str for value in self.coverage_profile)):
                raise TypeError("native diffusion coverage profile must be an exact pair of strings")
            if self.style != "native_diffusion":
                raise ValueError("contact coverage requires native_diffusion")
            if self.coverage_profile not in (("+20", "-20"), ("-20", "+20")):
                raise ValueError("unsupported native diffusion contact coverage profile")
        for field_name in ("group", "style", "order", "pin_facing"):
            value = getattr(self, field_name)
            if type(value) is not str:
                raise TypeError(f"array {field_name} must be an exact string")
            if value != value.strip():
                raise ValueError(f"array {field_name} must be canonical")
            if len(value) > 256:
                raise ValueError(f"array {field_name} is too long")
        if not self.group:
            raise ValueError("array group must be non-empty")
        if self.style not in {"", "shared_diffusion", "discrete", "native_diffusion", "native_cmos"}:
            raise ValueError(f"unsupported array style {self.style!r}")
        if type(self.rows) is not int:
            raise TypeError("array rows must be an exact integer")
        if not 1 <= self.rows <= 256:
            raise ValueError("array rows must be bounded in [1, 256]")
        if self.order not in {"", "ratio", "ABBA", "BAAB", "AABB", "blocked"}:
            raise ValueError(f"unsupported array order {self.order!r}")
        if self.style == "native_diffusion" and (self.rows != 1 or self.order != "ratio"):
            raise ValueError("native diffusion requires rows=1 and order=ratio")
        if self.style == "native_cmos" and (self.rows != 1 or self.order not in {"", "ratio"}):
            raise ValueError("native_cmos requires rows=1 and empty or ratio order")
        if type(self.transforms) is not tuple or any(
            type(value) is not str for value in self.transforms
        ):
            raise TypeError("array transforms must be one exact tuple of strings")
        allowed_transforms = {
            "R0", "R90", "R180", "R270",
            "MR0", "MR90", "MR180", "MR270",
        }
        if any(value not in allowed_transforms for value in self.transforms):
            raise ValueError("array transforms contain an unsupported transform")
        if len(self.transforms) > 256:
            raise ValueError("array transforms must be bounded")
        if type(self.align) is not str:
            raise TypeError("array align must be an exact string")
        if self.align not in {"", "bottom", "center", "top"}:
            raise ValueError("array align must be bottom, center, or top")
        if self.pin_facing not in {
            "", "outward", "inward", "top", "bottom", "left", "right"
        }:
            raise ValueError("array pin_facing is not a supported discrete relation")
        if self.style in {"native_diffusion", "native_cmos"} and self.transforms not in ((), ("R0",), ("MR0",)):
            raise ValueError("native diffusion/cmos supports only a single R0 or MR0 transform")
        if ((self.align or self.pin_facing) and self.style != "discrete") or (
                self.transforms and self.style not in {"discrete", "native_diffusion", "native_cmos"}):
            raise ValueError(
                "array transforms/alignment/pin_facing require style=discrete"
            )
        if self.bank_interconnect is not None:
            replay_bank_interconnect_plan(self.bank_interconnect)
            if self.style != "discrete" or self.order != "ratio":
                raise ValueError("bank interconnect requires style=discrete and order=ratio")
        if self.unit_pattern is not None:
            validate_array_unit_pattern(self.unit_pattern, style=self.style, order=self.order,
                                        rows=self.rows, transforms=self.transforms)
            if self.bank_interconnect is not None:
                raise ValueError("array unit pattern cannot use a rectangular bank interconnect")
        if type(self.occurrences) is not tuple:
            raise TypeError("array occurrences must be one exact tuple")
        if len(self.occurrences) > 4096:
            raise ValueError("array occurrences exceed the bounded source inventory")
        if self.occurrences and self.style != "native_diffusion":
            raise ValueError("array occurrences require native_diffusion")
        next_ordinal: Dict[str, int] = {}
        for occurrence in self.occurrences:
            if type(occurrence) is not MosaicOccurrence:
                raise TypeError("array occurrences must contain exact MosaicOccurrence values")
            occurrence.__post_init__()
            expected = next_ordinal.get(occurrence.logical_owner, 0)
            if occurrence.occurrence_ordinal != expected or occurrence.transform != "R0":
                raise ValueError("array occurrence ordinal/transform is duplicate or noncanonical")
            next_ordinal[occurrence.logical_owner] = expected + 1

    def canonical_data(self) -> Dict[str, object]:
        self.__post_init__()
        data: Dict[str, object] = {
            "group": self.group,
            "style": self.style,
            "rows": self.rows,
            "order": self.order,
        }
        if self.transforms:
            data["transforms"] = list(self.transforms)
        if self.align:
            data["align"] = self.align
        if self.pin_facing:
            data["pin_facing"] = self.pin_facing
        if self.bank_interconnect is not None:
            data["bank_interconnect"] = self.bank_interconnect.canonical_data()
        if self.unit_pattern is not None:
            data["unit_pattern"] = self.unit_pattern.canonical_data()
        if self.occurrences:
            data["occurrences"] = [item.canonical_data() for item in self.occurrences]
        if self.coverage_profile is not None:
            data["coverage_profile"] = list(self.coverage_profile)
        if self.access_profile:
            data["access_profile"] = self.access_profile
        if self.generation_profile:
            data["generation_profile"] = self.generation_profile
        if self.synthesis:
            data["synthesis"] = self.synthesis
        if self.tap_rail:
            data["tap_rail"] = self.tap_rail
        return data

    @property
    def sha256(self) -> str:
        encoded = json.dumps(
            self.canonical_data(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def replay_array_intent(value: object) -> ArrayIntent:
    """Re-run the exact array schema at every evidence-authority boundary."""

    if type(value) is not ArrayIntent:
        raise TypeError("array declaration must be an exact ArrayIntent")
    try:
        return ArrayIntent(
            group=value.group,
            style=value.style,
            rows=value.rows,
            order=value.order,
            transforms=value.transforms,
            align=value.align,
            pin_facing=value.pin_facing,
            bank_interconnect=value.bank_interconnect,
            unit_pattern=value.unit_pattern,
            occurrences=value.occurrences,
            coverage_profile=value.coverage_profile,
            access_profile=value.access_profile,
            generation_profile=value.generation_profile,
            synthesis=value.synthesis,
            tap_rail=value.tap_rail,
        )
    except AttributeError as exc:
        raise TypeError("array declaration lacks canonical fields") from exc


@dataclass(frozen=True)
class PassiveBankPartitionIntent:
    """One topology-defined sub-bank of a passive-bank functional group."""

    parent: str
    bank: str
    members: Tuple[str, ...]

    _IDENTIFIER: ClassVar[re.Pattern[str]] = re.compile(
        r"[A-Za-z_][A-Za-z0-9_]*\Z"
    )

    def __post_init__(self) -> None:
        for field_name in ("parent", "bank"):
            value = getattr(self, field_name)
            if type(value) is not str:
                raise TypeError(
                    f"passive bank partition {field_name} must be an exact string"
                )
            if not value or value != value.strip():
                raise ValueError(
                    f"passive bank partition {field_name} must be canonical and non-empty"
                )
            if len(value) > 256:
                raise ValueError(f"passive bank partition {field_name} is too long")
            if self._IDENTIFIER.fullmatch(value) is None:
                raise ValueError(
                    f"passive bank partition {field_name} must be a Clay identifier"
                )
        if type(self.members) is not tuple:
            raise TypeError("passive bank partition members must be an exact tuple")
        if not 1 <= len(self.members) <= 256:
            raise ValueError(
                "passive bank partition members must contain 1..256 entries"
            )
        for member in self.members:
            if type(member) is not str:
                raise TypeError(
                    "passive bank partition members must be exact strings"
                )
            if not member or member != member.strip() or len(member) > 256:
                raise ValueError(
                    "passive bank partition members must be canonical bounded names"
                )
            if self._IDENTIFIER.fullmatch(member) is None:
                raise ValueError(
                    "passive bank partition members must be Clay identifiers"
                )
        if self.members != tuple(sorted(set(self.members))):
            raise ValueError(
                "passive bank partition members must be unique and canonical"
            )

    def canonical_data(self) -> Dict[str, object]:
        return {
            "parent": self.parent,
            "bank": self.bank,
            "members": self.members,
        }

    @property
    def sha256(self) -> str:
        encoded = json.dumps(
            self.canonical_data(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def replay_passive_bank_partition_intent(
    value: object,
) -> PassiveBankPartitionIntent:
    """Re-run the exact partition schema at each authority boundary."""

    if type(value) is not PassiveBankPartitionIntent:
        raise TypeError(
            "passive bank partition declaration must be an exact "
            "PassiveBankPartitionIntent"
        )
    try:
        return PassiveBankPartitionIntent(
            parent=value.parent,
            bank=value.bank,
            members=value.members,
        )
    except AttributeError as exc:
        raise TypeError(
            "passive bank partition declaration lacks canonical fields"
        ) from exc


@dataclass(frozen=True)
class CommonCentroidIntent:
    """Common-centroid topology plus coordinate-free artwork transforms.

    Row 2 is the point mirror of Row 1 so even-order gradients cancel in both
    axes (the input differential pair). ``rows`` defaults to 2 and the complete
    PDK-rigid macro defaults to the unrotated ``R0`` orientation.  The optional
    ``first_member`` is coordinate-free physical phase: it selects which member
    starts the interleave without changing the functional group's electrical order.
    ``member_transforms`` gives the two members' artwork transforms in that same
    ordered-member frame for the left half; the opposite half is its vertical
    reflection.  It carries no pitch, coordinate, or reference placement.
    """

    group: str
    rows: int = 2
    orientation: str = "R0"
    first_member: str = ""
    member_transforms: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.group) is not str:
            raise TypeError("common-centroid group must be an exact string")
        if (
            not self.group
            or self.group != self.group.strip()
            or len(self.group) > 256
            or PassiveBankPartitionIntent._IDENTIFIER.fullmatch(self.group) is None
        ):
            raise ValueError(
                "common-centroid group must be a canonical bounded Clay identifier"
            )
        if type(self.rows) is not int:
            raise TypeError("common-centroid rows must be an exact integer")
        if self.rows not in (1, 2):
            raise ValueError("common-centroid rows must be 1 or 2")
        if type(self.orientation) is not str:
            raise TypeError("common-centroid orientation must be an exact string")
        if self.orientation not in ("R0", "R90", "R180", "R270"):
            raise ValueError(
                "common-centroid orientation must be one of "
                "('R0', 'R90', 'R180', 'R270')"
            )
        if type(self.first_member) is not str:
            raise TypeError("common-centroid first_member must be an exact string")
        if self.first_member and (
            self.first_member != self.first_member.strip()
            or len(self.first_member) > 256
            or PassiveBankPartitionIntent._IDENTIFIER.fullmatch(self.first_member)
            is None
        ):
            raise ValueError(
                "common-centroid first_member must be empty or a canonical bounded "
                "Clay identifier"
            )
        if type(self.member_transforms) is not tuple or any(
            type(value) is not str for value in self.member_transforms
        ):
            raise TypeError(
                "common-centroid member transforms must be one exact tuple of strings"
            )
        allowed_transforms = {
            "R0", "R90", "R180", "R270",
            "MR0", "MR90", "MR180", "MR270",
        }
        if self.member_transforms and (
            len(self.member_transforms) != 2
            or any(value not in allowed_transforms for value in self.member_transforms)
        ):
            raise ValueError(
                "common-centroid member transforms must be empty or contain exactly "
                "two supported orthogonal transforms"
            )

    def canonical_data(self) -> Dict[str, object]:
        payload = {
            "group": self.group,
            "rows": self.rows,
            "orientation": self.orientation,
        }
        if self.first_member:
            payload["first_member"] = self.first_member
        if self.member_transforms:
            payload["member_transforms"] = list(self.member_transforms)
        return payload

    @property
    def sha256(self) -> str:
        encoded = json.dumps(
            self.canonical_data(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def replay_common_centroid_intent(value: object) -> CommonCentroidIntent:
    """Re-run the exact common-centroid schema at every authority boundary."""

    if type(value) is not CommonCentroidIntent:
        raise TypeError(
            "common-centroid declaration must be an exact CommonCentroidIntent"
        )
    try:
        return CommonCentroidIntent(
            group=value.group,
            rows=value.rows,
            orientation=value.orientation,
            first_member=value.first_member,
            member_transforms=value.member_transforms,
        )
    except AttributeError as exc:
        raise TypeError(
            "common-centroid declaration lacks canonical fields"
        ) from exc


@dataclass(frozen=True)
class DummyFlankIntent:
    """`@dummy_flank(target, per_end=N, tie=gate_off|rail, share_diffusion=true)`.

    Generative edge dummies materialized at both ends of every row of ``target``
    WITHOUT a separate device card. ``tie`` is ``gate_off`` (gate/source/body on
    the array's source net) or ``rail`` (all terminals on a supply). Electrically
    off; a matched edge condition for the outermost active fingers.
    """

    target: str
    per_end: int = 1
    tie: str = ""
    share_diffusion: bool = True


@dataclass(frozen=True)
class WellIslandIntent:
    """`@well_island(group, body=net)` -- force ``group`` into its OWN well.

    The well is tied to ``body``; the placer must separate the island (e.g. the
    source-tied-body input pair whose body is a signal net, not the supply well).
    """

    group: str
    body: str = ""


@dataclass(frozen=True)
class WellIslandShareIntent:
    """``@well_island_share(group, body=net)`` shared-well contract.

    The members of ``group`` must occupy one connected well island whose body
    domain is ``body``.  Coordinates are deliberately absent: membership and
    body ownership come from canonical source topology, while materialization
    and verification operate on emitted well artwork.
    """

    group: str
    body: str

    def __post_init__(self) -> None:
        for field_name in ("group", "body"):
            value = getattr(self, field_name)
            if type(value) is not str:
                raise TypeError(
                    f"well island share {field_name} must be an exact string"
                )
            if (
                not value
                or value != value.strip()
                or len(value) > 256
                or PassiveBankPartitionIntent._IDENTIFIER.fullmatch(value) is None
            ):
                raise ValueError(
                    f"well island share {field_name} must be a canonical bounded "
                    "Clay identifier"
                )

    def canonical_data(self) -> Dict[str, str]:
        return {"group": self.group, "body": self.body}

    @property
    def sha256(self) -> str:
        encoded = json.dumps(
            self.canonical_data(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def replay_well_island_share_intent(value: object) -> WellIslandShareIntent:
    """Re-run the exact shared-well schema at every authority boundary."""

    if type(value) is not WellIslandShareIntent:
        raise TypeError(
            "well island share declaration must be an exact WellIslandShareIntent"
        )
    try:
        return WellIslandShareIntent(group=value.group, body=value.body)
    except AttributeError as exc:
        raise TypeError(
            "well island share declaration lacks canonical fields"
        ) from exc


@dataclass(frozen=True)
class BandIntent:
    """`@band(top=<supply>, bottom=<supply>, center=passives)` -- rail-side banding.

    P current sources hug the ``top`` supply rail, N sources hug the ``bottom``
    supply rail, passives occupy the ``center`` inter-rail channel. Inferred from
    source-to-rail connectivity, never a coordinate.
    """

    top: str = ""
    bottom: str = ""
    center: str = ""

    def __post_init__(self) -> None:
        for field_name in ("top", "bottom", "center"):
            value = getattr(self, field_name)
            if type(value) is not str:
                raise TypeError("band top/bottom/center must be exact strings")
            if not value or value != value.strip():
                raise ValueError(f"band requires a canonical non-empty {field_name}")
            if len(value) > 256:
                raise ValueError(f"band {field_name} is too long")
        if self.top == self.bottom:
            raise ValueError("band top and bottom nets must be distinct")

    def canonical_data(self) -> Dict[str, str]:
        return {"top": self.top, "bottom": self.bottom, "center": self.center}

    @property
    def sha256(self) -> str:
        encoded = json.dumps(
            self.canonical_data(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class RailAffinityIntent:
    """`@rail_affinity(group, side=top|bottom)` -- pin ``group`` to a supply band."""

    group: str
    side: str = ""

    def __post_init__(self) -> None:
        self.assert_authoritative()

    def assert_authoritative(self) -> None:
        if type(self.group) is not str or type(self.side) is not str:
            raise TypeError("rail affinity group/side must be exact strings")
        if (
            not self.group
            or self.group != self.group.strip()
            or len(self.group) > 256
        ):
            raise ValueError("rail affinity requires a canonical non-empty group")
        if self.side not in {"top", "bottom"}:
            raise ValueError("rail affinity side must be top or bottom")

    def canonical_data(self) -> Dict[str, str]:
        self.assert_authoritative()
        return {"group": self.group, "side": self.side}

    @property
    def sha256(self) -> str:
        encoded = json.dumps(
            self.canonical_data(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def replay_rail_affinity_intent(value: object) -> RailAffinityIntent:
    """Re-run the exact rail-affinity schema at every authority boundary."""

    if type(value) is not RailAffinityIntent:
        raise TypeError(
            "rail affinity declaration must be an exact RailAffinityIntent"
        )
    try:
        return RailAffinityIntent(group=value.group, side=value.side)
    except AttributeError as exc:
        raise TypeError(
            "rail affinity declaration lacks canonical fields"
        ) from exc


def replay_rail_affinity_inventory(
    value: object,
) -> Tuple[RailAffinityIntent, ...]:
    """Replay one bounded, unique rail-affinity declaration denominator."""

    if type(value) is not tuple:
        raise TypeError("rail affinity inventory must be an exact tuple")
    if len(value) > 64:
        raise ValueError("rail affinity declaration inventory is too large")
    replayed = tuple(replay_rail_affinity_intent(item) for item in value)
    if len({item.sha256 for item in replayed}) != len(replayed):
        raise ValueError("duplicate rail affinity declarations are forbidden")
    if len({item.group for item in replayed}) != len(replayed):
        raise ValueError("rail affinity requires exactly one side per group")
    return replayed


@dataclass(frozen=True)
class TrunkIntent:
    """`@trunk(net, layer=M2, orient=H [, width=W] [, edge=top|bottom])` -- signal-
    trunk layer + orientation (and, for a supply edge rail, width + cell edge).

    Assigns the routing ``layer`` and ``orient`` (H/V) for a signal trunk (e.g.
    the high-fan-out mirror-gate net as a wide horizontal M2 trunk) under the
    HVH discipline. ``width`` (um, optional) pins the trunk/rail wire width; ``edge``
    (``top`` / ``bottom``, optional) materializes the trunk as a full-cell-width M2
    EDGE RAIL along that cell edge (the reference power-rail structure) instead of a
    demand-routed trunk. ``offset`` (um, default 0) moves an edge rail inward
    by that exact distance from its named boundary. Inferred from net role /
    declared in the source.
    """

    net: str
    layer: str = ""
    orient: str = ""
    width_um: float = 0.0
    edge: str = ""
    offset_um: float = 0.0


@dataclass(frozen=True)
class PowerPlaneIntent:
    """`@power_plane(net, layer=M5|M6)` -- upper-metal power plane.

    A power/ground net distributed as an upper-metal plane (power-as-structure),
    not a thin wire. Inferred from port role.
    """

    net: str
    layer: str = ""
    _layer_index: int = field(init=False, repr=False, compare=False)
    _declaration_sha256: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        for field_name in ("net", "layer"):
            value = getattr(self, field_name)
            if type(value) is not str:
                raise TypeError("power plane net/layer must be exact strings")
            if not value or value != value.strip() or len(value) > 256:
                raise ValueError(
                    f"power plane requires a canonical non-empty {field_name}"
                )
        match = re.fullmatch(r"(?:M|metal)([1-9][0-9]?)", self.layer)
        if match is None:
            raise ValueError("power plane layer must be M<number> or metal<number>")
        object.__setattr__(self, "_layer_index", int(match.group(1)))
        object.__setattr__(
            self,
            "_declaration_sha256",
            hashlib.sha256(
                json.dumps(
                    self.canonical_data(),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                ).encode("ascii")
            ).hexdigest(),
        )

    def canonical_data(self) -> Dict[str, object]:
        match = re.fullmatch(r"(?:M|metal)([1-9][0-9]?)", self.layer)
        if match is None:
            raise ValueError("power plane layer must be M<number> or metal<number>")
        return {"net": self.net, "layer_index": int(match.group(1))}

    def assert_authoritative(self) -> None:
        replayed = PowerPlaneIntent(net=self.net, layer=self.layer)
        if (
            replayed._layer_index != self._layer_index
            or replayed._declaration_sha256 != self._declaration_sha256
        ):
            raise ValueError("power plane declaration is not canonical")

    @property
    def layer_index(self) -> int:
        self.assert_authoritative()
        return self._layer_index

    @property
    def sha256(self) -> str:
        self.assert_authoritative()
        return self._declaration_sha256


def replay_power_plane_inventory(
    value: object,
) -> Tuple[PowerPlaneIntent, ...]:
    """Replay the exact bounded power-plane declaration denominator."""

    if type(value) is not tuple:
        raise TypeError("power plane inventory must be an exact tuple")
    if len(value) > 64:
        raise ValueError("power plane declaration inventory is too large")
    replayed = []
    for item in value:
        if type(item) is not PowerPlaneIntent:
            raise TypeError("power plane inventory contains a foreign row")
        item.assert_authoritative()
        replayed.append(item)
    if len({(item.net, item.layer_index) for item in replayed}) != len(replayed):
        raise ValueError("duplicate power plane declarations are forbidden")
    return tuple(replayed)


@dataclass(frozen=True)
class AnchorIntent:
    """`@anchor(target, edge=left|right|top|bottom)` -- edge-access placement.

    Pin the cluster carrying net ``target`` (or a named group / device) to a cell
    EDGE so that net's port enters/exits there -- the routing-owed twin of the
    diff-pair LEFT column (e.g. ``@anchor(vout, edge=right)`` drives the output
    stage toward the right edge so VOUT exits right). ``target`` is a net / group /
    device label (topology reference, never a coordinate); ``edge`` is one of
    ``left`` / ``right`` / ``top`` / ``bottom``. Consumed by
    :func:`clay.layout.place_blocks_2d.reposition_pum_2d`: a direct canonical
    member becomes a HARD block-edge equality, while net/group clusters retain
    the legacy SOFT edge pull below envelope minimization.  Net edge access is
    materialized exactly by the router from PDK rules.
    """

    target: str
    edge: str = ""

    def __post_init__(self) -> None:
        for field_name in ("target", "edge"):
            value = getattr(self, field_name)
            if type(value) is not str:
                raise TypeError("anchor target/edge must be exact strings")
            if not value or value != value.strip():
                raise ValueError(f"anchor requires a canonical non-empty {field_name}")
            if len(value) > 256:
                raise ValueError(f"anchor {field_name} is too long")
        if self.edge not in {"left", "right", "top", "bottom"}:
            raise ValueError("anchor edge must be left/right/top/bottom")

    def canonical_data(self) -> Dict[str, str]:
        return {"target": self.target, "edge": self.edge}

    @property
    def sha256(self) -> str:
        encoded = json.dumps(
            self.canonical_data(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ChannelIntent:
    """`@channel(orient=vertical|horizontal, width=Wum)` -- reserve a routing channel.

    Reserve a corridor of ``width_um`` (a strip free of device geometry) along the
    given ``orient`` axis for cross-band routing straps (the reference's M3 vertical
    straps). Consumed by :func:`clay.layout.place_blocks_2d` as the reserved
    left-column -> lower-region separation (a vertical channel just right of the
    input column). Width is DESIGN intent (the strap corridor class), never a
    reference coordinate.
    """

    orient: str = ""
    width_um: float = 0.0

    def __post_init__(self) -> None:
        if type(self.orient) is not str:
            raise TypeError("channel orient must be an exact string")
        if self.orient not in {"vertical", "horizontal"}:
            raise ValueError("channel orient must be vertical or horizontal")
        if type(self.width_um) is not float:
            raise TypeError("channel width_um must be an exact float")
        if not (
            math.isfinite(self.width_um)
            and 0.0 < self.width_um <= 1_000_000.0
            and int(round(self.width_um * 1000.0)) >= 1
        ):
            raise ValueError(
                "channel width_um must be positive finite, quantize to at least "
                "1nm, and remain bounded at 1000000um"
            )

    def canonical_data(self) -> Dict[str, object]:
        return {"orient": self.orient, "width_um": self.width_um}

    @property
    def sha256(self) -> str:
        encoded = json.dumps(
            self.canonical_data(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def replay_channel_intent(value: object) -> ChannelIntent:
    """Re-run the exact channel schema at every evidence-authority boundary."""

    if type(value) is not ChannelIntent:
        raise TypeError("channel declaration must be an exact ChannelIntent")
    try:
        return ChannelIntent(orient=value.orient, width_um=value.width_um)
    except AttributeError as exc:
        raise TypeError("channel declaration lacks canonical fields") from exc


@dataclass(frozen=True)
class DensityIntent:
    """Coordinate-free maximum whitespace fraction for a structural envelope."""

    max_whitespace_fraction: float
    scope: str = "structural"

    def __post_init__(self) -> None:
        if type(self.max_whitespace_fraction) is not float:
            raise TypeError(
                "density max_whitespace_fraction must be an exact float"
            )
        if not (
            math.isfinite(self.max_whitespace_fraction)
            and 0.0 <= self.max_whitespace_fraction < 1.0
            and not (
                self.max_whitespace_fraction == 0.0
                and math.copysign(1.0, self.max_whitespace_fraction) < 0.0
            )
        ):
            raise ValueError(
                "density max_whitespace_fraction must be finite and in [0, 1)"
            )
        if type(self.scope) is not str:
            raise TypeError("density scope must be an exact string")
        if self.scope != "structural":
            raise ValueError("density scope must be structural")

    def canonical_data(self) -> Dict[str, object]:
        canonical = DensityIntent(
            max_whitespace_fraction=self.max_whitespace_fraction,
            scope=self.scope,
        )
        return {
            "max_whitespace_fraction": canonical.max_whitespace_fraction,
            "scope": canonical.scope,
        }

    @property
    def sha256(self) -> str:
        encoded = json.dumps(
            self.canonical_data(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def replay_density_intent(value: object) -> DensityIntent:
    """Re-run the exact density schema at every evidence-authority boundary."""

    if type(value) is not DensityIntent:
        raise TypeError("density declaration must be an exact DensityIntent")
    try:
        return DensityIntent(
            max_whitespace_fraction=value.max_whitespace_fraction,
            scope=value.scope,
        )
    except AttributeError as exc:
        raise TypeError("density declaration lacks canonical fields") from exc


def _replay_density_inventory(value: object) -> Tuple[DensityIntent, ...]:
    if type(value) is not tuple:
        raise TypeError("density inventory must be an exact tuple")
    if len(value) > 1:
        raise ValueError("density intent supports exactly one declaration")
    return tuple(replay_density_intent(item) for item in value)


@dataclass(frozen=True)
class OdFillIntent:
    """Minimum source-owned functional active occupancy of the core envelope."""

    min_fraction: float
    scope: str = "core"

    def __post_init__(self) -> None:
        if type(self.min_fraction) is not float:
            raise TypeError("od_fill min_fraction must be an exact float")
        if not (
            math.isfinite(self.min_fraction)
            and 0.0 < self.min_fraction <= 1.0
        ):
            raise ValueError("od_fill min_fraction must be finite and in (0, 1]")
        if type(self.scope) is not str:
            raise TypeError("od_fill scope must be an exact string")
        if self.scope != "core":
            raise ValueError("od_fill scope must be core")

    def canonical_data(self) -> Dict[str, object]:
        canonical = OdFillIntent(
            min_fraction=self.min_fraction,
            scope=self.scope,
        )
        return {
            "min_fraction": canonical.min_fraction,
            "scope": canonical.scope,
        }

    @property
    def sha256(self) -> str:
        encoded = json.dumps(
            self.canonical_data(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def replay_od_fill_intent(value: object) -> OdFillIntent:
    """Re-run the exact functional OD schema at every authority boundary."""

    if type(value) is not OdFillIntent:
        raise TypeError("od_fill declaration must be an exact OdFillIntent")
    try:
        return OdFillIntent(
            min_fraction=value.min_fraction,
            scope=value.scope,
        )
    except AttributeError as exc:
        raise TypeError("od_fill declaration lacks canonical fields") from exc


def _replay_od_fill_inventory(value: object) -> Tuple[OdFillIntent, ...]:
    if type(value) is not tuple:
        raise TypeError("od_fill inventory must be an exact tuple")
    if len(value) > 1:
        raise ValueError("od_fill intent supports exactly one declaration")
    return tuple(replay_od_fill_intent(item) for item in value)


_ELECTRICAL_BUDGET_NET_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def _validate_electrical_budget_net(value: object) -> str:
    if type(value) is not str:
        raise TypeError("electrical budget net must be an exact string")
    if len(value) > 256 or _ELECTRICAL_BUDGET_NET_RE.fullmatch(value) is None:
        raise ValueError("electrical budget net must be a canonical net name")
    return value


def _positive_finite_budget_value(name: str, value: object) -> float:
    if type(value) is not float:
        raise TypeError(f"{name} must be an exact float")
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return value


@dataclass(frozen=True)
class PowerBudgetIntent:
    """IR-drop limit for one power/output net, verified by calibrated PEX."""

    net: str
    current_a: float
    max_drop_v: float

    def __post_init__(self) -> None:
        _validate_electrical_budget_net(self.net)
        _positive_finite_budget_value("power budget current_a", self.current_a)
        _positive_finite_budget_value("power budget max_drop_v", self.max_drop_v)

    def canonical_data(self) -> Dict[str, object]:
        canonical = replay_power_budget_intent(self)
        return {
            "net": canonical.net,
            "current_a": canonical.current_a,
            "max_drop_v": canonical.max_drop_v,
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self.canonical_data(),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()


def replay_power_budget_intent(value: object) -> PowerBudgetIntent:
    if type(value) is not PowerBudgetIntent:
        raise TypeError("power budget declaration must be an exact PowerBudgetIntent")
    try:
        return PowerBudgetIntent(
            net=value.net,
            current_a=value.current_a,
            max_drop_v=value.max_drop_v,
        )
    except AttributeError as exc:
        raise TypeError("power budget declaration lacks canonical fields") from exc


@dataclass(frozen=True)
class SensitiveNetBudgetIntent:
    """Maximum extracted R/C for one sensitive net."""

    net: str
    max_resistance_ohm: float
    max_capacitance_f: float

    def __post_init__(self) -> None:
        _validate_electrical_budget_net(self.net)
        _positive_finite_budget_value(
            "sensitive net max_resistance_ohm", self.max_resistance_ohm
        )
        _positive_finite_budget_value(
            "sensitive net max_capacitance_f", self.max_capacitance_f
        )

    def canonical_data(self) -> Dict[str, object]:
        canonical = replay_sensitive_net_budget_intent(self)
        return {
            "net": canonical.net,
            "max_resistance_ohm": canonical.max_resistance_ohm,
            "max_capacitance_f": canonical.max_capacitance_f,
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self.canonical_data(),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()


def replay_sensitive_net_budget_intent(value: object) -> SensitiveNetBudgetIntent:
    if type(value) is not SensitiveNetBudgetIntent:
        raise TypeError(
            "sensitive net budget declaration must be an exact "
            "SensitiveNetBudgetIntent"
        )
    try:
        return SensitiveNetBudgetIntent(
            net=value.net,
            max_resistance_ohm=value.max_resistance_ohm,
            max_capacitance_f=value.max_capacitance_f,
        )
    except AttributeError as exc:
        raise TypeError(
            "sensitive net budget declaration lacks canonical fields"
        ) from exc


def _replay_electrical_budget_inventory(
    value: object, *, power: bool
) -> tuple[PowerBudgetIntent, ...] | tuple[SensitiveNetBudgetIntent, ...]:
    label = "power budget" if power else "sensitive net budget"
    if type(value) is not tuple:
        raise TypeError(f"{label} inventory must be an exact tuple")
    if len(value) > 256:
        raise ValueError(f"{label} inventory is too large")
    replay = replay_power_budget_intent if power else replay_sensitive_net_budget_intent
    canonical = tuple(replay(item) for item in value)
    nets = tuple(item.net for item in canonical)
    if len(set(nets)) != len(nets):
        raise ValueError(f"duplicate {label} net declaration")
    return tuple(sorted(canonical, key=lambda item: item.net))


def replay_power_budget_inventory(
    value: object,
) -> tuple[PowerBudgetIntent, ...]:
    return tuple(_replay_electrical_budget_inventory(value, power=True))


def replay_sensitive_net_budget_inventory(
    value: object,
) -> tuple[SensitiveNetBudgetIntent, ...]:
    return tuple(_replay_electrical_budget_inventory(value, power=False))


@dataclass(frozen=True, order=True)
class KelvinIntent:
    """Coordinate-free force/sense pair terminated by one sensing device."""

    force_net: str
    sense_net: str
    target_device: str

    def __post_init__(self) -> None:
        _validate_electrical_budget_net(self.force_net)
        _validate_electrical_budget_net(self.sense_net)
        _validate_electrical_budget_net(self.target_device)
        if self.force_net == self.sense_net:
            raise ValueError("Kelvin force and sense nets must be distinct")

    def canonical_data(self) -> Dict[str, str]:
        canonical = replay_kelvin_intent(self)
        return {
            "force_net": canonical.force_net,
            "sense_net": canonical.sense_net,
            "target_device": canonical.target_device,
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self.canonical_data(),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()


def replay_kelvin_intent(value: object) -> KelvinIntent:
    if type(value) is not KelvinIntent:
        raise TypeError("Kelvin declaration must be an exact KelvinIntent")
    try:
        return KelvinIntent(
            force_net=value.force_net,
            sense_net=value.sense_net,
            target_device=value.target_device,
        )
    except AttributeError as exc:
        raise TypeError("Kelvin declaration lacks canonical fields") from exc


def replay_kelvin_inventory(value: object) -> tuple[KelvinIntent, ...]:
    if type(value) is not tuple:
        raise TypeError("Kelvin inventory must be an exact tuple")
    if len(value) > 256:
        raise ValueError("Kelvin inventory is too large")
    canonical = tuple(replay_kelvin_intent(item) for item in value)
    identities = tuple(
        (item.force_net, item.sense_net, item.target_device) for item in canonical
    )
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate Kelvin declaration")
    return tuple(sorted(canonical))


@dataclass(frozen=True)
class SplitIntent:
    """`@split(passive, n=N, match=series|parallel)` -- matched split of a passive.

    The passive's value is realized as ``n`` equal ``series``/``parallel`` units
    laid adjacent (matched); the materializer generates the units + a shared
    isolation ring. Makes the MATCHING intent first-class.
    """

    passive: str
    n: int = 1
    match: str = ""


@dataclass(frozen=True)
class MatchingIntent:
    """One canonical unordered device-matching claim.

    ``members`` is the full unordered member set (currently exactly two labels).
    ``declaration_ids`` retains every directed source declaration which collapsed
    into this one obligation, so reciprocal ``A -> B`` / ``B -> A`` authorship is
    neither duplicated nor lost.  ``device_classes`` is source provenance used by
    the materializer/capability gate; it never changes the geometric comparator.
    """

    members: Tuple[str, str]
    declaration_ids: Tuple[str, ...]
    device_classes: Tuple[str, str] = ("", "")
    requires_interdigitation: bool = False
    interdigitation_declaration_ids: Tuple[str, ...] = ()
    interdigitation_pattern: str = ""
    interdigitation_finger_pairs: int = 0

    def __post_init__(self) -> None:
        if type(self.members) is not tuple:
            raise TypeError("matching members must be an exact tuple")
        if type(self.declaration_ids) is not tuple:
            raise TypeError("matching declaration_ids must be an exact tuple")
        if type(self.device_classes) is not tuple:
            raise TypeError("matching device_classes must be an exact tuple")
        if type(self.interdigitation_declaration_ids) is not tuple:
            raise TypeError(
                "interdigitation_declaration_ids must be an exact tuple"
            )
        if type(self.requires_interdigitation) is not bool:
            raise TypeError("requires_interdigitation must be an exact bool")
        if type(self.interdigitation_pattern) is not str:
            raise TypeError("interdigitation_pattern must be an exact string")
        if type(self.interdigitation_finger_pairs) is not int:
            raise TypeError("interdigitation_finger_pairs must be an exact integer")
        if any(
            type(item) is not str
            for item in self.interdigitation_declaration_ids
        ):
            raise TypeError(
                "interdigitation_declaration_ids must contain exact strings"
            )
        interdigitation_ids = tuple(sorted(self.interdigitation_declaration_ids))
        if (
            len(interdigitation_ids) > 8
            or len(interdigitation_ids) != len(set(interdigitation_ids))
            or any(
                not item or item != item.strip() or len(item) > 512
                for item in interdigitation_ids
            )
        ):
            raise ValueError(
                "interdigitation declaration ids must be unique canonical values"
            )
        object.__setattr__(
            self, "interdigitation_declaration_ids", interdigitation_ids
        )
        if self.requires_interdigitation:
            if not interdigitation_ids:
                raise ValueError(
                    "interdigitated matching requires declaration authority"
                )
            if self.interdigitation_pattern not in {"abab", "abba"}:
                raise ValueError(
                    "interdigitated matching requires pattern abab or abba"
                )
            if not 1 <= self.interdigitation_finger_pairs <= 256:
                raise ValueError(
                    "interdigitation_finger_pairs must be bounded in [1, 256]"
                )
        elif (
            self.interdigitation_pattern
            or interdigitation_ids
            or self.interdigitation_finger_pairs != 0
        ):
            raise ValueError(
                "non-interdigitated matching must not carry interdigitation fields"
            )
        if any(type(member) is not str for member in self.members):
            raise TypeError("matching members must be exact strings")
        if any(type(item) is not str for item in self.device_classes):
            raise TypeError("matching device_classes must be exact strings")
        if any(type(item) is not str for item in self.declaration_ids):
            raise TypeError("matching declaration_ids must be exact strings")
        source_members = self.members
        classes = self.device_classes
        if len(source_members) != 2 or len(classes) != 2:
            raise ValueError(
                "matching members and device_classes must contain two entries"
            )
        members = tuple(sorted(source_members))
        if members[0] == members[1]:
            raise ValueError(
                f"matching members must be two distinct labels, got {self.members!r}"
            )
        object.__setattr__(self, "members", members)
        ids = tuple(sorted(self.declaration_ids))
        if (
            not ids
            or len(ids) > 8
            or len(ids) != len(set(ids))
            or any(
                not item or item != item.strip() or len(item) > 512
                for item in ids
            )
        ):
            raise ValueError("matching declaration_ids must be non-empty and unique")
        object.__setattr__(self, "declaration_ids", ids)
        class_by_member = dict(zip(source_members, classes))
        object.__setattr__(
            self,
            "device_classes",
            tuple(class_by_member[member] for member in members),
        )

    def canonical_data(self) -> Dict[str, object]:
        return {
            "members": self.members,
            "declaration_ids": self.declaration_ids,
            "device_classes": self.device_classes,
            "requires_interdigitation": self.requires_interdigitation,
            "interdigitation_declaration_ids": (
                self.interdigitation_declaration_ids
            ),
            "interdigitation_pattern": self.interdigitation_pattern,
            "interdigitation_finger_pairs": self.interdigitation_finger_pairs,
        }

    @property
    def sha256(self) -> str:
        encoded = json.dumps(
            self.canonical_data(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def replay_matching_intent(value: object) -> MatchingIntent:
    """Re-run the exact matching/interdigitation schema at evidence boundaries."""

    if type(value) is not MatchingIntent:
        raise TypeError("matching declaration must be an exact MatchingIntent")
    try:
        return MatchingIntent(
            members=value.members,
            declaration_ids=value.declaration_ids,
            device_classes=value.device_classes,
            requires_interdigitation=value.requires_interdigitation,
            interdigitation_declaration_ids=(
                value.interdigitation_declaration_ids
            ),
            interdigitation_pattern=value.interdigitation_pattern,
            interdigitation_finger_pairs=value.interdigitation_finger_pairs,
        )
    except AttributeError as exc:
        raise TypeError("matching declaration lacks canonical fields") from exc


@dataclass(frozen=True)
class LayoutIntent:
    """The declared net-level layout intent for one cell.

    ``diff_pairs`` are genuinely DIFFERENTIAL net pairs (symmetric roles -- input
    transconductor, regenerative latch, differential output, differential cascode
    branches). Each becomes a ``matched_cc_balance`` node-cap budget: the two
    nodes must stay parasitically balanced.

    ``mirror_pairs`` are single-ended-output CURRENT-MIRROR node pairs (one node
    the diode/reference leg, the other the mirrored output). A mirror's two NODES
    are asymmetric by design, so a mirror pair carries NO ``matched_cc_balance``
    budget; its matching is a DEVICE placement/sizing concern, recorded here so
    :mod:`clay.layout.placement_symmetry` can consume it. Default ``()`` keeps
    every existing constructor call valid (additive, back-compatible).

    v2 provenance / three-layer composition fields (additive, back-compatible):

      * ``profile_id`` -- the category profile this intent specialises (e.g.
        ``"dynamic_comparator_v1"``, ``"5t_ota_v1"``), or ``None`` for a hand-built
        intent. The full signoff record stores it so a benchmark across cells can
        group by category.
      * ``source`` -- where this intent came from ("auto" for the per-cell
        deriver, "category-profile" for the profile-driven composer, "declared"
        for a designer-authored intent merged via :func:`merge_layout_intent`).
      * ``digest`` (property) -- a stable SHA-256 of the intent's user-visible
        contents (cell + classes + pairs + each budget's kind/pair/tol/drivers/
        source + each shield), 16 hex chars. Stored alongside a signed token in
        the record so a verdict carries the EXACT intent it certified, not just
        the cell name.
    """

    cell: str
    dynamic_nets: Tuple[str, ...]
    sensitive_nets: Tuple[str, ...]
    diff_pairs: Tuple[Tuple[str, str], ...]
    budgets: Tuple[ParasiticBudget, ...]
    shields: Tuple[Shield, ...]
    mirror_pairs: Tuple[Tuple[str, str], ...] = ()
    profile_id: Optional[str] = None
    source: str = "auto"
    # ── Placement / area directives (NOT parasitic budgets — never gate the ──
    #    proof token). Lowered from `@objective` / `@area` / `@place_group`.
    #    All default empty/None so every existing constructor call stays valid.
    objective: Optional[Tuple[str, str]] = (
        None  # (target, goal) e.g. ("area","minimize")
    )
    area_max_um2: Optional[float] = None
    place_groups: Tuple[PlaceGroup, ...] = ()
    # Declared-but-unverified resistance budgets (no PEX res metric yet).
    resistance_budgets: Tuple[ResistanceBudget, ...] = ()
    # Device-environment (LDE) budgets: per-device well-edge / OD-length /
    # OD-spacing matching, measured over DEVICE GEOMETRY (not the PEX caps). Each
    # becomes a proof-token-gating obligation exactly like a cap budget. Default
    # empty so every existing constructor call stays valid (additive).
    lde_budgets: Tuple[LdeBudget, ...] = ()
    # Per-net DECLARED routing WIDTH (nm), keyed by net name. A `@wire_width(net,
    # 2u)` declaration the designer authors so the generic router routes that net's
    # trunk / risers / rail at >= the declared width (parasitic / IR-drop / EM
    # intent) instead of the PDK minimum. The router consumes this map (no net-name
    # branch in the engine — the net name is DATA from here); a `wire_width`
    # geometric obligation verifies the REALISED width >= the declared width. Empty
    # by default so every existing intent / constructor call is byte-identical and a
    # cell that declares no width routes exactly as before.
    wire_widths: Dict[str, int] = field(default_factory=dict)
    # Matched-net WIRE-LENGTH balance pairs. A `@match_length(netA, netB)`
    # declaration the designer authors so the two nets' ROUTED wire-length
    # parasitic is balanced (differential / matched signals). Each entry is an
    # ordered `(netA, netB)` pair. The production router consumes it as a hard
    # symmetric Length constraint; a separate `match_length` geometric
    # obligation MEASURES each net's routed metal run length on the final layout
    # and verifies the relative imbalance |La - Lb| / max(La, Lb) is within a
    # relative tolerance. Empty by default so every existing intent /
    # constructor call is byte-identical. Net names are DATA (no engine branch).
    match_length_pairs: Tuple[Tuple[str, str], ...] = ()
    # Declared RELATIVE-ORDER constraints between two placed devices. Each entry is
    # an ordered `(a, b, rel)` triple from an `@order(a, b, rel)` declaration, where
    # `rel` is "left_of" (a's centroid sits left of b's) or "above" (a's centroid
    # sits above b's) in the realised placement. Unlike the CHECK-only
    # `match_length_pairs`, this DRIVES the placer: the seqpair engine adds the
    # corresponding directed edge to its H (left_of) / V (above) constraint graph so
    # the realised order respects it BY CONSTRUCTION, and a `check_order` obligation
    # verifies it on the real placement. Empty by default so every existing intent /
    # constructor call is byte-identical and a cell that declares no `@order` places
    # exactly as before. Device names are DATA (no engine branch).
    order_constraints: Tuple[Tuple[str, str, str], ...] = ()
    # Declared ADJACENCY pairs. Each entry is an ordered `(a, b)` pair from an
    # `@abut(a, b)` declaration: a and b must be placed edge-adjacent, the facing
    # inter-device gap tightened to the DRC-minimum (not the default routing
    # channel). Like `order_constraints` this DRIVES the placer (tight spacing +
    # adjacency preference in the packing order); a `check_abut` obligation verifies
    # the realised facing gap is within a small abs/rel tolerance. Empty by default
    # (byte-parity); device names are DATA.
    abut_pairs: Tuple[Tuple[str, str], ...] = ()
    # Declared PRIORITY-axis annotations (intent-priority axis). The priority axis
    # is orthogonal to hardness: hardness is a property of ONE obligation (does it
    # gate? observable? supported?), priority is a RELATION BETWEEN obligations
    # (when two conflict, which yields?). Both fields are lowered to / consumed by
    # the resolver ``clay.layout.spec.priority:resolve_priority`` (Intent ->
    # Obligation enumeration -> *resolver* -> Verdict). Empty by default so every
    # existing intent / constructor call is byte-identical and an existing design's
    # digest / verdict is unchanged (back-compat: the priority axis must not alter
    # any existing verdict).
    #
    #   ``priorities`` -- explicit ``@priority(target, tier)`` tier overrides. Each
    #     entry is an ordered ``(target, tier)`` pair: ``target`` is the obligation
    #     key (kind / member-pair label) the designer pins to ``tier`` (one of
    #     ``clay.layout.spec.priority.VALID_TIER``), overriding the default-table
    #     derivation. Names are DATA (no engine branch).
    priorities: Tuple[Tuple[str, str], ...] = ()
    #   ``superiority`` -- intra-tier defeasible ``A > B`` declarations. Each entry
    #     is an ordered ``(a, b)`` pair meaning ``a`` defeats ``b`` when the two
    #     conflict at the SAME tier (the resolver's declared-superiority branch).
    #     The relation is a PARTIAL order and must be ACYCLIC: a declared cycle
    #     (``A > B > C > A``) raises at lowering (contract violation, not silent
    #     drop) -- defeasible logic requires ``>`` acyclic. Names are DATA.
    superiority: Tuple[Tuple[str, str], ...] = ()
    # Declared GUARD-RING cluster intents. Each entry is a `@guard_ring(group=(...),
    # well=<p|n>, tie=<net>)` declaration lowered to a `GuardRingIntent` (members /
    # well / tie_net). `route_cell` materializes each into the ONE shared geometry
    # dict via `group_guard_rings` (the same builder anchor_ota uses) + injects a
    # ring-tap Pin on the tie net so the ring is TIED, not dead artwork; the
    # independent `native.guard_ring_materialized` checker verifies the drawn ring
    # against these declarations. Empty by default so every existing intent /
    # constructor call is byte-identical and a cell that declares no ring routes /
    # emits exactly as before. Device/net names are DATA (no engine branch).
    guard_rings: Tuple[GuardRingIntent, ...] = ()
    # Existing ``@well_isolated`` annotation targets.  This is executable
    # topology intent: a supporting PDK lowers each target to a private p-well
    # inside deep N-well; an unsupported PDK fails closed.
    well_isolated_devices: Tuple[str, ...] = ()
    # ── Stage-1 first-class CRAFT intent (coordinate-free organizational intent
    #    inferred from netlist topology). Each lowers from a `@array` /
    #    `@common_centroid` / `@dummy_flank` / `@well_island` / `@band` /
    #    `@rail_affinity` / `@trunk` / `@power_plane` / `@split` declaration. All
    #    default EMPTY so a `.clay` that declares no craft intent is BYTE-IDENTICAL
    #    (unchanged digest / verdict). PARSE + LOWER only at this stage -- no
    #    materializer / placer / router consumes them yet. Names / styles / layers
    #    are DATA (no engine branch).
    arrays: Tuple[ArrayIntent, ...] = ()
    common_centroids: Tuple[CommonCentroidIntent, ...] = ()
    dummy_flanks: Tuple[DummyFlankIntent, ...] = ()
    well_islands: Tuple[WellIslandIntent, ...] = ()
    well_island_shares: Tuple[WellIslandShareIntent, ...] = ()
    bands: Tuple[BandIntent, ...] = ()
    rail_affinities: Tuple[RailAffinityIntent, ...] = ()
    trunks: Tuple[TrunkIntent, ...] = ()
    power_planes: Tuple[PowerPlaneIntent, ...] = ()
    splits: Tuple[SplitIntent, ...] = ()
    passive_bank_partitions: Tuple[PassiveBankPartitionIntent, ...] = ()
    # Direct device-geometry matching claims from the ``annotate`` block.  Each
    # reciprocal pair is canonicalized once while retaining all directed IDs.
    matching_intents: Tuple[MatchingIntent, ...] = ()
    # Routing-owed PLACEMENT intent (P4). `anchors` pins a net's cluster / a group
    # to a cell edge (edge-access); `channels` reserves a device-free routing
    # corridor for cross-band straps. Both default EMPTY so a `.clay` that declares
    # neither is BYTE-IDENTICAL (unchanged digest / verdict). Consumed by
    # `place_blocks_2d` (direct-member hard edge / cluster soft pull / reserved
    # column separation); no engine branch (topology names + the axis are DATA).
    anchors: Tuple[AnchorIntent, ...] = ()
    channels: Tuple[ChannelIntent, ...] = ()
    densities: Tuple[DensityIntent, ...] = ()
    od_fills: Tuple[OdFillIntent, ...] = ()
    power_budgets: Tuple[PowerBudgetIntent, ...] = ()
    sensitive_net_budgets: Tuple[SensitiveNetBudgetIntent, ...] = ()
    kelvins: Tuple[KelvinIntent, ...] = ()
    adjacencies: Tuple[AdjacentIntent, ...] = ()
    # Explicit hierarchy seam between placement groups and routing.  Appended
    # to preserve LayoutIntent's positional ABI.  A route group derives every
    # external member terminal from topology and carries no reference data.
    route_groups: Tuple[RouteGroupIntent, ...] = ()
    edge_alignments: Tuple[EdgeAlignmentIntent, ...] = ()
    terminal_accesses: Tuple[TerminalAccessIntent, ...] = ()
    # Evidence-bound row-major occurrence cover projected from canonical ASIR.
    # This carries only owner/ordinal/transform topology and provenance; all
    # reference coordinates remain outside the production intent payload.
    discrete_mosaics: Tuple[DiscreteMosaicIntent, ...] = ()
    mosaic_center_alignments: Tuple[MosaicCenterAlignmentIntent, ...] = ()
    aggregate_orders: Tuple[AggregateOrderIntent, ...] = ()
    reference_conditioned_placement_obligations: Tuple[
        ReferenceConditionedPlacementObligation, ...
    ] = ()
    hierarchical_macro_dag: Optional[HierarchicalMacroDagIntent] = None
    aggregate_alignments: Tuple[AggregateAlignmentIntent, ...] = ()
    # Cross-owner generated-cavity placement declaration. Appended to preserve
    # LayoutIntent's positional ABI. Boundaries are intrinsic coordinate-free
    # selectors over host geometry; reference evidence is attached only through
    # ReferenceConditionedPlacementObligation during projection.
    row_bounded_cavities: Tuple[RowBoundedCavityIntent, ...] = ()
    # Appended to preserve the positional ABI. Empty emits no new payload key.
    mosaic_guest_slots: Tuple[MosaicGuestSlot, ...] = ()
    mosaic_boundary_joins: Tuple[MosaicBoundaryJoinIntent, ...] = ()
    match_scopes: Tuple[NetMatchScope, ...] = ()
    # Explicit same-layer metal clearance. Empty keeps the historical digest.
    net_spacings: Tuple[NetSpacingIntent, ...] = ()
    # Hard source-bound routing channels; absent preserves historical digests.
    channel_routes: Tuple[ChannelRouteIntent, ...] = ()
    imported_layout: Optional[ImportedLayoutIntent] = None
    route_tails: Tuple[RouteTailIntent, ...] = ()
    group_center_alignments: Tuple[GroupCenterAlignment, ...] = ()
    group_edge_alignments: Tuple[GroupEdgeAlignment, ...] = ()
    group_separations: Tuple[GroupSeparation, ...] = ()
    channel_alignments: Tuple[ChannelAlignment, ...] = ()

    def __post_init__(self) -> None:
        for name, cls in (("group_edge_alignments", GroupEdgeAlignment),
                          ("group_separations", GroupSeparation),
                          ("channel_alignments", ChannelAlignment)):
            rows = getattr(self, name)
            if type(rows) is not tuple or any(type(row) is not cls for row in rows):
                raise TypeError(name + " must contain exact typed declarations")
            if len(set(rows)) != len(rows):
                raise ValueError("duplicate " + name + " declarations")
            for row in rows:
                row.__post_init__()
        object.__setattr__(self, "net_spacings", canonical_net_spacings(self.net_spacings))
        object.__setattr__(self, "channel_routes", canonical_channel_routes(self.channel_routes))
        canonical_imported_routes(self.imported_layout, self.route_tails)
        replay_match_scopes(self.match_scopes, self.diff_pairs)
        object.__setattr__(self, "shields", replay_shield_inventory(self.shields))
        object.__setattr__(
            self,
            "route_groups",
            replay_route_group_inventory(self.route_groups),
        )
        object.__setattr__(
            self,
            "edge_alignments",
            replay_edge_alignment_inventory(self.edge_alignments),
        )
        object.__setattr__(
            self,
            "terminal_accesses",
            replay_terminal_access_inventory(self.terminal_accesses),
        )
        object.__setattr__(
            self,
            "discrete_mosaics",
            replay_discrete_mosaic_inventory(self.discrete_mosaics),
        )
        object.__setattr__(self, "mosaic_guest_slots", replay_mosaic_guest_slot_inventory(self.mosaic_guest_slots))
        object.__setattr__(self, "mosaic_boundary_joins", replay_mosaic_boundary_join_inventory(
            self.mosaic_boundary_joins, mosaics=self.discrete_mosaics))
        if any(slot.source_mosaic_sha256 not in {row.sha256 for row in self.discrete_mosaics}
               for slot in self.mosaic_guest_slots):
            raise ValueError("mosaic guest slot source declaration is absent")
        object.__setattr__(
            self,
            "mosaic_center_alignments",
            replay_mosaic_center_alignment_inventory(
                self.mosaic_center_alignments
            ),
        )
        object.__setattr__(
            self,
            "aggregate_orders",
            replay_aggregate_order_inventory(self.aggregate_orders),
        )
        object.__setattr__(
            self,
            "reference_conditioned_placement_obligations",
            replay_reference_conditioned_placement_obligation_inventory(
                self.reference_conditioned_placement_obligations
            ),
        )
        object.__setattr__(
            self,
            "aggregate_alignments",
            replay_aggregate_alignment_inventory(self.aggregate_alignments),
        )
        object.__setattr__(
            self,
            "row_bounded_cavities",
            replay_row_bounded_cavity_inventory(self.row_bounded_cavities),
        )
        if self.hierarchical_macro_dag is not None:
            object.__setattr__(
                self,
                "hierarchical_macro_dag",
                replay_hierarchical_macro_dag_intent(self.hierarchical_macro_dag),
            )
        replay_rail_affinity_inventory(self.rail_affinities)
        object.__setattr__(
            self,
            "power_planes",
            replay_power_plane_inventory(self.power_planes),
        )
        _replay_density_inventory(self.densities)
        _replay_od_fill_inventory(self.od_fills)
        object.__setattr__(
            self,
            "power_budgets",
            _replay_electrical_budget_inventory(self.power_budgets, power=True),
        )
        object.__setattr__(
            self,
            "sensitive_net_budgets",
            _replay_electrical_budget_inventory(
                self.sensitive_net_budgets, power=False
            ),
        )
        object.__setattr__(self, "kelvins", replay_kelvin_inventory(self.kelvins))
        object.__setattr__(
            self,
            "adjacencies",
            replay_adjacent_inventory(self.adjacencies),
        )

    def net_class(self, net: str) -> frozenset:
        """Return the set of classes a net carries ('dynamic' and/or 'sensitive')."""
        out = set()
        if net in self.dynamic_nets:
            out.add("dynamic")
        if net in self.sensitive_nets:
            out.add("sensitive")
        return frozenset(out)

    @property
    def placement_diff_pairs(self) -> Tuple[Tuple[str, str], ...]:
        """Net matches allowed to influence placement; route scope stays elsewhere."""
        return _placement_diff_pairs(self)

    @property
    def digest(self) -> str:
        """Stable 16-hex SHA-256 of the intent's user-visible contents.

        Stable across runs / Python versions (encoded as JSON with sorted keys).
        Changing ANY field that affects the verdict (a budget's pair, tol,
        aggressors, clock_net, victim_set, source; a shield; a diff/mirror pair;
        the cell name) changes the digest. The digest is what a signoff record
        and a signed token embed to bind the verdict to the exact intent that
        produced it.
        """
        payload = {
            "cell": self.cell,
            "profile_id": self.profile_id,
            "source": self.source,
            "dynamic_nets": list(self.dynamic_nets),
            "sensitive_nets": list(self.sensitive_nets),
            "diff_pairs": [list(p) for p in self.diff_pairs],
            "mirror_pairs": [list(p) for p in self.mirror_pairs],
            "budgets": [
                {
                    "kind": b.kind,
                    "pair": list(b.pair),
                    "rel_tol": float(b.rel_tol),
                    "aggressors": None if b.aggressors is None else list(b.aggressors),
                    "victim_set": None if b.victim_set is None else list(b.victim_set),
                    "clock_net": b.clock_net,
                    "source": b.source,
                    "profile_id": b.profile_id,
                }
                for b in self.budgets
            ],
            "shields": [
                s.canonical_data() for s in replay_shield_inventory(self.shields)
            ],
            "objective": None if self.objective is None else list(self.objective),
            "area_max_um2": self.area_max_um2,
            "place_groups": [
                {
                    "group": g.group,
                    "devices": list(g.devices),
                    "row_min": g.row_min,
                    "row_max": g.row_max,
                    "priority": g.priority,
                    "col": g.col,
                    **(
                        {"pack_rows": g.pack_rows}
                        if g.pack_rows is not None
                        else {}
                    ),
                    **(
                        {"packing": g.packing}
                        if g.packing is not None
                        else {}
                    ),
                }
                for g in self.place_groups
            ],
            "resistance_budgets": [
                {"pair": list(r.pair), "rel_tol": float(r.rel_tol), "kind": r.kind}
                for r in self.resistance_budgets
            ],
            "lde_budgets": [
                {
                    "pair": list(b.pair),
                    "metric": b.metric,
                    "rel_tol": float(b.rel_tol),
                    "source": b.source,
                }
                for b in self.lde_budgets
            ],
            "wire_widths": {k: int(v) for k, v in sorted(self.wire_widths.items())},
            "match_length_pairs": [list(p) for p in self.match_length_pairs],
            "order_constraints": [list(t) for t in self.order_constraints],
            "abut_pairs": [list(p) for p in self.abut_pairs],
        }
        if self.net_spacings:
            payload["net_spacings"] = [
                [c.net_a, c.net_b, c.minimum_nm] for c in self.net_spacings
            ]
        if self.channel_routes:
            payload["channel_routes"] = [
                channel.canonical_data()
                for channel in canonical_channel_routes(self.channel_routes)
            ]
        if self.imported_layout is not None:
            canonical_imported_routes(self.imported_layout, self.route_tails)
            payload["imported_layout"] = self.imported_layout.canonical_data()
            payload["route_tails"] = [tail.canonical_data() for tail in self.route_tails]
        if scoped_pairs := scope_digest_data(self.match_scopes):
            payload["match_scopes"] = scoped_pairs
        if self.route_groups:
            payload["route_groups"] = [
                item.canonical_data()
                for item in replay_route_group_inventory(self.route_groups)
            ]
        if self.edge_alignments:
            payload["edge_alignments"] = [
                item.canonical_data()
                for item in replay_edge_alignment_inventory(self.edge_alignments)
            ]
        if self.terminal_accesses:
            payload["terminal_accesses"] = [
                item.canonical_data()
                for item in replay_terminal_access_inventory(self.terminal_accesses)
            ]
        # Priority-axis fields are added to the digest payload ONLY when present so
        # an existing design that declares no priority keeps a BYTE-IDENTICAL
        # digest (back-compat: the priority axis must not change any existing
        # verdict / signed token). A declared priority IS user-visible, so when
        # present it changes the digest.
        if self.priorities:
            payload["priorities"] = [list(p) for p in self.priorities]
        if self.superiority:
            payload["superiority"] = [list(p) for p in self.superiority]
        # Guard-ring intents are added to the digest ONLY when present so a cell
        # that declares none keeps a BYTE-IDENTICAL digest / signed token (the
        # feature must not change any existing verdict). A declared ring IS
        # user-visible, so when present it changes the digest.
        if self.guard_rings:
            payload["guard_rings"] = [
                g.canonical_data()
                for g in self.guard_rings
            ]
        if self.well_isolated_devices:
            payload["well_isolated_devices"] = list(self.well_isolated_devices)
        if self.adjacencies:
            payload["adjacencies"] = [
                item.canonical_data()
                for item in replay_adjacent_inventory(self.adjacencies)
            ]
        # Stage-1 craft-intent fields are added to the digest ONLY when present so
        # a cell that declares no craft intent keeps a BYTE-IDENTICAL digest /
        # signed token (the vocabulary must not change any existing verdict). A
        # declared craft intent IS user-visible, so when present it changes the
        # digest.
        if self.arrays:
            payload["arrays"] = [
                {
                    "group": a.group,
                    "style": a.style,
                    "rows": a.rows,
                    "order": a.order,
                    **(
                        {"pin_facing": a.pin_facing}
                        if a.pin_facing
                        else {}
                    ),
                    **(
                        {"occurrences": [item.canonical_data() for item in a.occurrences]}
                        if a.occurrences else {}
                    ),
                }
                for a in self.arrays
            ]
        if self.common_centroids:
            payload["common_centroids"] = [
                {
                    **{"group": c.group, "rows": c.rows},
                    **(
                        {"orientation": c.orientation}
                        if c.orientation != "R0"
                        else {}
                    ),
                    **(
                        {"first_member": c.first_member}
                        if c.first_member
                        else {}
                    ),
                    **(
                        {"member_transforms": list(c.member_transforms)}
                        if c.member_transforms
                        else {}
                    ),
                }
                for c in self.common_centroids
            ]
        if self.dummy_flanks:
            payload["dummy_flanks"] = [
                {
                    "target": d.target,
                    "per_end": d.per_end,
                    "tie": d.tie,
                    "share_diffusion": d.share_diffusion,
                }
                for d in self.dummy_flanks
            ]
        if self.discrete_mosaics:
            payload["discrete_mosaics"] = [
                item.canonical_data() for item in self.discrete_mosaics
            ]
        if self.mosaic_guest_slots:
            payload["mosaic_guest_slots"] = [row.canonical_data() for row in self.mosaic_guest_slots]
        if self.mosaic_boundary_joins:
            payload["mosaic_boundary_joins"] = [row.canonical_data() for row in self.mosaic_boundary_joins]
        if self.mosaic_center_alignments:
            payload["mosaic_center_alignments"] = [
                item.canonical_data()
                for item in self.mosaic_center_alignments
            ]
        if self.aggregate_orders:
            payload["aggregate_orders"] = [
                item.canonical_data() for item in self.aggregate_orders
            ]
        if self.reference_conditioned_placement_obligations:
            payload["reference_conditioned_placement_obligations"] = [
                item.canonical_data()
                for item in self.reference_conditioned_placement_obligations
            ]
        if self.aggregate_alignments:
            payload["aggregate_alignments"] = [
                item.canonical_data() for item in self.aggregate_alignments
            ]
        if self.group_center_alignments:
            payload["group_center_alignments"] = [
                item.canonical_data() for item in self.group_center_alignments
            ]
        if self.group_edge_alignments:
            payload["group_edge_alignments"] = [item.canonical_data() for item in self.group_edge_alignments]
        if self.group_separations:
            payload["group_separations"] = [item.canonical_data() for item in self.group_separations]
        if self.channel_alignments:
            payload["channel_alignments"] = [item.canonical_data() for item in self.channel_alignments]
        if self.row_bounded_cavities:
            payload["row_bounded_cavities"] = [
                item.canonical_data() for item in self.row_bounded_cavities
            ]
        if self.hierarchical_macro_dag is not None:
            payload["hierarchical_macro_dag"] = (
                self.hierarchical_macro_dag.canonical_data()
            )
        if self.well_islands:
            payload["well_islands"] = [
                {"group": w.group, "body": w.body} for w in self.well_islands
            ]
        if self.well_island_shares:
            payload["well_island_shares"] = [
                item.canonical_data() for item in self.well_island_shares
            ]
        if self.bands:
            payload["bands"] = [
                {"top": b.top, "bottom": b.bottom, "center": b.center}
                for b in self.bands
            ]
        rail_affinities = replay_rail_affinity_inventory(self.rail_affinities)
        if rail_affinities:
            payload["rail_affinities"] = [
                {"group": r.group, "side": r.side} for r in rail_affinities
            ]
        if self.trunks:
            payload["trunks"] = [
                {
                    "net": t.net,
                    "layer": t.layer,
                    "orient": t.orient,
                    "width_um": t.width_um,
                    "edge": t.edge,
                }
                for t in self.trunks
            ]
        if self.power_planes:
            payload["power_planes"] = [
                p.canonical_data()
                for p in replay_power_plane_inventory(self.power_planes)
            ]
        if self.splits:
            payload["splits"] = [
                {"passive": s.passive, "n": s.n, "match": s.match} for s in self.splits
            ]
        if self.passive_bank_partitions:
            payload["passive_bank_partitions"] = [
                item.canonical_data() for item in self.passive_bank_partitions
            ]
        if self.anchors:
            payload["anchors"] = [
                {"target": a.target, "edge": a.edge} for a in self.anchors
            ]
        if self.channels:
            payload["channels"] = [
                {"orient": c.orient, "width_um": c.width_um} for c in self.channels
            ]
        densities = _replay_density_inventory(self.densities)
        if densities:
            payload["densities"] = [
                replay_density_intent(item).canonical_data()
                for item in densities
            ]
        od_fills = _replay_od_fill_inventory(self.od_fills)
        if od_fills:
            payload["od_fills"] = [
                replay_od_fill_intent(item).canonical_data()
                for item in od_fills
            ]
        power_budgets = _replay_electrical_budget_inventory(
            self.power_budgets, power=True
        )
        if power_budgets:
            payload["power_budgets"] = [
                replay_power_budget_intent(item).canonical_data()
                for item in power_budgets
            ]
        sensitive_net_budgets = _replay_electrical_budget_inventory(
            self.sensitive_net_budgets, power=False
        )
        if sensitive_net_budgets:
            payload["sensitive_net_budgets"] = [
                replay_sensitive_net_budget_intent(item).canonical_data()
                for item in sensitive_net_budgets
            ]
        kelvins = replay_kelvin_inventory(self.kelvins)
        if kelvins:
            payload["kelvins"] = [item.canonical_data() for item in kelvins]
        if self.matching_intents:
            payload["matching_intents"] = [
                {
                    "members": list(item.members),
                    "declaration_ids": list(item.declaration_ids),
                    "device_classes": list(item.device_classes),
                    "requires_interdigitation": item.requires_interdigitation,
                    **(
                        {"interdigitation_pattern": item.interdigitation_pattern}
                        if item.requires_interdigitation
                        else {}
                    ),
                    **(
                        {
                            "interdigitation_declaration_ids": list(
                                item.interdigitation_declaration_ids
                            ),
                            "interdigitation_finger_pairs": (
                                item.interdigitation_finger_pairs
                            ),
                        }
                        if item.requires_interdigitation
                        else {}
                    ),
                }
                for item in self.matching_intents
            ]
        text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ─────────────────────────────────────────────────────────────────────────────
# Generic, reusable builder: net-classification -> LayoutIntent
# ─────────────────────────────────────────────────────────────────────────────


def _normalize_pairs(
    pairs: Iterable[Sequence[str]],
) -> Tuple[Tuple[str, str], ...]:
    """Coerce an iterable of 2-tuples into a deduplicated tuple of net pairs.

    Preserves the FIRST occurrence's net order (so ``(net0, net1)`` is not
    silently reordered) but collapses an exact-duplicate pair regardless of net
    order, since a diff pair is unordered. Each pair must name two DISTINCT nets.
    """
    out: List[Tuple[str, str]] = []
    seen = set()
    for p in pairs:
        a, b = tuple(p)
        if a == b:
            raise ValueError(
                f"a differential pair must name two distinct nets, got {p!r}"
            )
        key = frozenset((a, b))
        if key in seen:
            continue
        seen.add(key)
        out.append((a, b))
    return tuple(out)


def _budget_key(budget: ParasiticBudget) -> Tuple[str, frozenset[str]]:
    return (budget.kind, frozenset(budget.pair))


def _shield_key(shield: Shield) -> Tuple[str, str]:
    return (shield.aggressor, shield.victim)


# NOTE (intent-priority axis): these `_dedupe_*_prefer_later` helpers collapse
# EXACT same-key duplicates only -- a budget keyed on (kind, frozenset(pair)), a
# shield on (aggressor, victim). "Later wins" therefore fires only when the
# designer RESTATES the identical obligation (same kind + same members); it is
# redundancy collapse, NOT a cross-obligation priority decision. The priority
# resolver (clay.layout.spec.priority:resolve_priority) subsumes the cross-
# obligation case: it adjudicates DIFFERENT, conflicting obligations by tier /
# superiority / specificity and emits an explicit CONFLICT for incomparable ties
# (no arrival-order tie-break). The two never overlap, so the dedupe's recency
# rule is kept (collapsing identical restatements is legitimate and back-compat;
# retiring it would change every existing intent's byte-parity for no semantic
# gain). See docs/research/intent-priority-construction-plan-2026-06-15.html S5.
def _dedupe_budgets_prefer_later(
    budgets: Iterable[ParasiticBudget],
) -> Tuple[ParasiticBudget, ...]:
    out: List[ParasiticBudget] = []
    index: Dict[Tuple[str, frozenset[str]], int] = {}
    for budget in budgets:
        key = _budget_key(budget)
        if key in index:
            out[index[key]] = budget
        else:
            index[key] = len(out)
            out.append(budget)
    return tuple(out)


def _dedupe_shields_prefer_later(shields: Iterable[Shield]) -> Tuple[Shield, ...]:
    out: List[Shield] = []
    index: Dict[Tuple[str, str], int] = {}
    for shield in shields:
        key = _shield_key(shield)
        if key in index:
            out[index[key]] = shield
        else:
            index[key] = len(out)
            out.append(shield)
    return tuple(out)


def _lde_metric_for_decl(decl: ParsedLayoutIntentDecl) -> Optional[str]:
    """Resolve an LDE metric from a parsed decl, or ``None`` if it is not LDE.

    Accepts either surface form:
      * the user-facing keyword kind (``equal_well_edge`` / ``equal_od`` /
        ``equal_od_spacing``) -> mapped via :data:`_LDE_KIND_TO_METRIC`;
      * a lowered ``kind == "lde_budget"`` carrying an explicit ``metric``
        attribute (or a ``metric`` entry in ``attributes``).
    Returns the canonical metric name (one of :data:`_LDE_METRICS`) or ``None``.
    """
    kind = decl.kind
    if kind in _LDE_KIND_TO_METRIC:
        return _LDE_KIND_TO_METRIC[kind]
    if kind == "lde_budget":
        metric = getattr(decl, "metric", None)
        if metric is None:
            metric = (decl.attributes or {}).get("metric")
        if metric in _LDE_METRICS:
            return str(metric)
        raise ValueError(
            f"lde_budget declaration needs a metric in {sorted(_LDE_METRICS)}, "
            f"got {metric!r}"
        )
    return None


def _coerce_int(value: object, default: int) -> int:
    """Coerce a parsed kwarg value to ``int``, falling back to ``default``.

    Stage-1 craft-intent kwargs (``rows`` / ``per_end`` / ``n``) arrive as
    strings from the parser (``layout_place_arg`` stringifies the token). A
    missing or non-integer value falls back to ``default`` rather than raising --
    a bogus count never fabricates an out-of-range array, it just uses the
    default (defensive, mirroring the ``@place_group`` ``int(str(...))`` pattern).
    """
    if value is None:
        return default
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return default


def _coerce_float(value: object, default: float) -> float:
    """Coerce a parsed kwarg value to ``float``, falling back to ``default``.

    Layout-intent kwargs (e.g. ``@trunk(..., width=8)``) arrive as strings from the
    parser (``layout_place_arg`` stringifies the token). A missing or non-numeric
    value falls back to ``default`` rather than raising. The value is a plain number
    interpreted in MICRONS (the ``layout_place_arg`` grammar carries no unit suffix,
    so a trunk/rail width is authored as ``width=8`` = 8um, ``width=1.94`` = 1.94um).
    """
    if value is None:
        return default
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return default


def _coerce_bool(value: object, default: bool) -> bool:
    """Coerce a parsed kwarg value to ``bool``, falling back to ``default``.

    Accepts the surface spellings the grammar can carry (``true`` / ``false`` /
    ``1`` / ``0``, case-insensitive). An unrecognised value falls back to
    ``default`` rather than raising.
    """
    if value is None:
        return default
    token = str(value).strip().lower()
    if token in ("true", "1", "yes"):
        return True
    if token in ("false", "0", "no"):
        return False
    return default


def parasitic_budget_driver_attributes(
    declaration: ParsedLayoutIntentDecl,
) -> Dict[str, object]:
    """Validate source metric drivers before binding, lowering, or formatting.

    The source surface exposes the existing ``ParasiticBudget`` fields only on
    the metric that consumes them. Unknown or misplaced options cannot silently
    disappear, and repeated aggressors cannot count the same coupling twice.
    Empty attributes keep all legacy four-argument declarations unchanged.
    """
    attributes = declaration.attributes
    if type(attributes) is not dict:
        raise TypeError("@budget attributes must be an exact dict")
    unknown = set(attributes) - {"aggressors", "clock_net"}
    if unknown:
        raise ValueError(f"unknown @budget driver attributes: {sorted(unknown)!r}")
    result: Dict[str, object] = {}
    if "aggressors" in attributes:
        if declaration.budget_kind != "kickback":
            raise ValueError("@budget aggressors is only supported for kickback")
        values = attributes["aggressors"]
        if type(values) is not tuple or not values or any(
            type(value) is not str or not value or value != value.strip()
            for value in values
        ):
            raise TypeError("@budget aggressors must be a nonempty tuple of net names")
        if len(set(values)) != len(values):
            raise ValueError("@budget aggressors must not repeat a net")
        result["aggressors"] = values
    if "clock_net" in attributes:
        if declaration.budget_kind != "clk_regen":
            raise ValueError("@budget clock_net is only supported for clk_regen")
        value = attributes["clock_net"]
        if type(value) is not str or not value or value != value.strip():
            raise TypeError("@budget clock_net must be a nonempty net name")
        result["clock_net"] = value
    return result


def layout_intent_from_declarations(
    cell: str,
    declarations: Iterable[ParsedLayoutIntentDecl],
    *,
    default_match_tol: float = _DEFAULT_MATCH_TOL,
) -> LayoutIntent:
    """Lower parsed `.clay` layout-intent declarations to LayoutIntent."""
    declarations = tuple(declarations)
    match_scopes = match_scopes_from_declarations(declarations)
    diff_pairs: List[Tuple[str, str]] = []
    mirror_pairs: List[Tuple[str, str]] = []
    explicit_budgets: List[ParasiticBudget] = []
    shields: List[Shield] = []
    resistance_budgets: List[ResistanceBudget] = []
    place_groups: List[PlaceGroup] = []
    route_groups: List[RouteGroupIntent] = []
    lde_budgets: List[LdeBudget] = []
    wire_widths: Dict[str, int] = {}
    match_length_pairs: List[Tuple[str, str]] = []
    net_spacings = []
    order_constraints: List[Tuple[str, str, str]] = []
    edge_alignments: List[EdgeAlignmentIntent] = []
    group_center_alignments: List[GroupCenterAlignment] = []
    group_edge_alignments: List[GroupEdgeAlignment] = []
    group_separations: List[GroupSeparation] = []
    channel_alignments: List[ChannelAlignment] = []
    terminal_accesses: List[TerminalAccessIntent] = []
    abut_pairs: List[Tuple[str, str]] = []
    priorities: List[Tuple[str, str]] = []
    superiority: List[Tuple[str, str]] = []
    guard_rings: List[GuardRingIntent] = []
    arrays: List[ArrayIntent] = []
    common_centroids: List[CommonCentroidIntent] = []
    dummy_flanks: List[DummyFlankIntent] = []
    well_islands: List[WellIslandIntent] = []
    well_island_shares: List[WellIslandShareIntent] = []
    bands: List[BandIntent] = []
    rail_affinities: List[RailAffinityIntent] = []
    trunks: List[TrunkIntent] = []
    power_planes: List[PowerPlaneIntent] = []
    splits: List[SplitIntent] = []
    passive_bank_partitions: List[PassiveBankPartitionIntent] = []
    row_bounded_cavities: List[RowBoundedCavityIntent] = []
    anchors: List[AnchorIntent] = []
    channels: List[ChannelIntent] = []
    densities: List[DensityIntent] = []
    od_fills: List[OdFillIntent] = []
    power_budgets: List[PowerBudgetIntent] = []
    sensitive_net_budgets: List[SensitiveNetBudgetIntent] = []
    kelvins: List[KelvinIntent] = []
    objective: Optional[Tuple[str, str]] = None
    area_max_um2: Optional[float] = None

    for decl in declarations:
        kind = decl.kind
        # Device-environment (LDE) budgets: the surface keyword
        # (@equal_well_edge / @equal_od / @equal_od_spacing) OR a lowered
        # kind="lde_budget" carrying a `metric` attribute both route here.
        metric = _lde_metric_for_decl(decl)
        if metric is not None:
            rel_tol = (
                float(decl.rel_tol) if decl.rel_tol is not None else _DEFAULT_LDE_TOL
            )
            lde_budgets.append(
                LdeBudget(tuple(decl.nets), metric, rel_tol, source="declared")
            )
            continue
        if kind in {"match", "symmetric"}:
            diff_pairs.append(tuple(decl.nets))
        elif kind == "mirror":
            mirror_pairs.append(tuple(decl.nets))
        elif kind == "align":
            attrs = decl.attributes or {}
            edge_alignments.append(
                EdgeAlignmentIntent(
                    first=str(attrs.get("first") or decl.nets[0]),
                    second=str(attrs.get("second") or decl.nets[1]),
                    edge=str(attrs.get("edge") or ""),
                    offset_nm=int(attrs.get("offset_nm") or 0),
                )
            )
        elif kind == "center_group":
            attrs = decl.attributes or {}
            group_center_alignments.append(GroupCenterAlignment(
                target=str(attrs["target"]), groups=tuple(attrs["groups"])))
        elif kind == "group_align":
            attrs = decl.attributes or {}
            group_edge_alignments.append(GroupEdgeAlignment(attrs["first"], attrs["second"], attrs["edge"]))
        elif kind == "group_separate":
            attrs = decl.attributes or {}
            group_separations.append(GroupSeparation(attrs["before"], attrs["after"], attrs["axis"], int(attrs.get("gap_nm", "0"))))
        elif kind == "channel_align":
            attrs = decl.attributes or {}
            channel_alignments.append(ChannelAlignment(attrs["first"], attrs["second"]))
        elif kind == "terminal_access":
            attrs = decl.attributes or {}
            terminal_accesses.append(
                TerminalAccessIntent(
                    device=str(attrs.get("device") or decl.nets[0]),
                    terminal=str(attrs.get("terminal") or ""),
                    side=str(attrs.get("side") or ""),
                )
            )
        elif kind == "shield":
            if decl.aggressor is None or decl.victim is None:
                raise ValueError(f"shield intent missing aggressor/victim: {decl!r}")
            shields.append(Shield(decl.aggressor, decl.victim, decl.shield_net))
        elif kind == "budget":
            if decl.budget_kind is None or decl.rel_tol is None:
                raise ValueError(f"budget intent missing kind/tolerance: {decl!r}")
            raw_kind = decl.budget_kind
            driver_attributes = parasitic_budget_driver_attributes(decl)
            # Resistance budgets have no PEX metric yet -> record for provenance,
            # do NOT emit a (permanently-unmeasurable) parasitic obligation.
            if raw_kind in _RESISTANCE_BUDGET_KINDS:
                resistance_budgets.append(
                    ResistanceBudget(
                        tuple(decl.nets), float(decl.rel_tol), kind=raw_kind
                    )
                )
                continue
            # `parasitic_cap` keeps its own kind (no longer aliased onto
            # `matched_cc_balance`): the obligation checker measures a SINGLE-ENDED
            # `parasitic_cap(net, rail)` against the reference rail and a
            # two-signal-net `parasitic_cap(a, b)` as a differential node-cap
            # balance. A `matched_cc_balance` source kind stays as-is.
            budget = ParasiticBudget(
                raw_kind,
                tuple(decl.nets),
                float(decl.rel_tol),
                source="declared",
                **driver_attributes,
            )
            explicit_budgets.append(budget)
            # Only a genuinely-differential budget (matched_cc_balance, or a
            # parasitic_cap between two SIGNAL nets) seeds a diff pair. A
            # single-ended parasitic_cap(net, rail) is asymmetric by design and
            # must NOT become a differential pair / symmetry obligation.
            if budget.kind == "matched_cc_balance" or (
                budget.kind == "parasitic_cap"
                and not parasitic_cap_is_single_ended(budget.pair)
            ):
                diff_pairs.append(budget.pair)
        elif kind == "objective":
            # `@objective(target, goal)` e.g. (area, minimize). Last one wins.
            if decl.target is not None and decl.goal is not None:
                objective = (str(decl.target), str(decl.goal))
        elif kind == "area":
            # `@area(max_um2, <value>)`: record the area cap when given as max_um2.
            if decl.limit_kind == "max_um2" and decl.limit_value is not None:
                area_max_um2 = float(decl.limit_value)
        elif kind == "place_group":
            col_attr = (decl.attributes or {}).get("col")
            col_val: Optional[int] = None
            if col_attr is not None:
                try:
                    col_val = int(str(col_attr))
                except (TypeError, ValueError):
                    col_val = None
            pack_rows_attr = (decl.attributes or {}).get("pack_rows")
            pack_rows_val: Optional[int] = None
            if pack_rows_attr is not None:
                if type(pack_rows_attr) is bool:
                    raise ValueError("place_group pack_rows must be an exact integer")
                try:
                    pack_rows_val = int(str(pack_rows_attr))
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "place_group pack_rows must be an exact integer"
                    ) from exc
                if str(pack_rows_val) != str(pack_rows_attr) or pack_rows_val < 1:
                    raise ValueError(
                        "place_group pack_rows must be a positive exact integer"
                    )
                if pack_rows_val > len(tuple(decl.devices)):
                    raise ValueError(
                        "place_group pack_rows exceeds its member capacity"
                    )
            packing_attr = (decl.attributes or {}).get("packing")
            packing_val: Optional[str] = None
            if packing_attr is not None:
                if type(packing_attr) is not str:
                    raise ValueError(
                        "place_group packing must be exactly rigid, deformable, or relational"
                    )
                packing_val = packing_attr
            place_groups.append(
                PlaceGroup(
                    group=str(decl.group) if decl.group is not None else "",
                    devices=tuple(decl.devices),
                    row_min=decl.row_min,
                    row_max=decl.row_max,
                    priority=decl.priority,
                    col=col_val,
                    pack_rows=pack_rows_val,
                    packing=packing_val,
                )
            )
        elif kind == "route_group":
            attrs = decl.attributes
            if type(attrs) is not dict:
                raise TypeError("route_group attributes must be an exact dict")
            if set(attrs) not in (
                {"group", "ports"},
                {"group", "groups", "ports"},
            ):
                raise ValueError(
                    "route_group requires group, ports, and optional groups"
                )
            raw_groups = attrs.get("groups", ())
            if type(raw_groups) is not tuple:
                raise TypeError("route_group groups must be an exact tuple")
            route_groups.append(
                RouteGroupIntent(
                    group=attrs["group"],
                    ports=attrs["ports"],
                    place_groups=raw_groups,
                )
            )
        elif kind == "wire_width":
            # `@wire_width(net, width)`: the parser carried the net + the width in
            # NANOMETRES on `attributes`. Record the declared width per net (last
            # declaration wins). A non-positive width is ignored (a width <= 0 is not
            # a real widening; it would never bind against max(declared, PDK-min)).
            attrs = decl.attributes or {}
            net = str(attrs.get("net") or (decl.nets[0] if decl.nets else ""))
            width_nm = attrs.get("width_nm")
            if net and width_nm is not None:
                try:
                    w = int(width_nm)
                except (TypeError, ValueError):
                    w = 0
                if w > 0:
                    wire_widths[net] = w
        elif kind == "match_length":
            # `@match_length(netA, netB)`: the two nets whose ROUTED wire-length
            # parasitic must be balanced (differential / matched signals). The
            # parser carried the ordered pair on `attributes['nets']` (mirroring
            # `@wire_width`); fall back to `decl.nets`. The lowered pair drives
            # the router's hard Length constraint and a separate `match_length`
            # geometric obligation measures the final routed metal run length.
            attrs = decl.attributes or {}
            pair = attrs.get("nets")
            if not pair and decl.nets:
                pair = (decl.nets[0], decl.nets[1]) if len(decl.nets) >= 2 else None
            if pair and len(tuple(pair)) >= 2:
                a, b = str(pair[0]), str(pair[1])
                if a and b:
                    match_length_pairs.append((a, b))
        elif kind == "net_spacing":
            if len(decl.nets) != 2 or set(decl.attributes or {}) != {"minimum_nm"}:
                raise ValueError("net_spacing requires two nets and a minimum length")
            net_spacings.append(NetSpacingIntent(*decl.nets, decl.attributes["minimum_nm"]))
        elif kind == "order":
            # `@order(a, b, rel)`: a relative-order constraint the placer realises
            # by construction. The parser carried the ordered pair + relation on
            # `attributes` (a/b/rel); fall back to `decl.nets` for the pair. `rel`
            # must be one of {left_of, above}; any other value is dropped (the
            # grammar enum prevents it, but stay defensive). A self-pair (a == b) is
            # vacuous and dropped.
            attrs = decl.attributes or {}
            a = str(attrs.get("a") or (decl.nets[0] if decl.nets else ""))
            b = str(attrs.get("b") or (decl.nets[1] if len(decl.nets) > 1 else ""))
            rel = str(attrs.get("rel") or "")
            if a and b and a != b and rel in {"left_of", "above"}:
                order_constraints.append((a, b, rel))
        elif kind == "abut":
            # `@abut(a, b)`: an adjacency pair the placer packs tight. The parser
            # carried the ordered pair on `attributes['pair']`; fall back to
            # `decl.nets`. A self-pair (a == b) is vacuous and dropped.
            attrs = decl.attributes or {}
            pair = attrs.get("pair")
            if not pair and decl.nets:
                pair = (decl.nets[0], decl.nets[1]) if len(decl.nets) >= 2 else None
            if pair and len(tuple(pair)) >= 2:
                a, b = str(pair[0]), str(pair[1])
                if a and b and a != b:
                    abut_pairs.append((a, b))
        elif kind == "priority":
            # `@priority(target, tier)`: pin obligation `target` to an explicit
            # priority `tier`, overriding the default-table tier derivation in
            # clay.layout.spec.priority. The parser carried target + tier on
            # `attributes` (fall back to `decl.nets` for the target). `tier` must be
            # one of priority.VALID_TIER; any other value is dropped (the grammar
            # enum prevents it, but stay defensive -- a bogus tier never becomes a
            # fabricated priority). Names/tiers are DATA (no engine branch).
            attrs = decl.attributes or {}
            target = str(attrs.get("target") or (decl.nets[0] if decl.nets else ""))
            tier = str(attrs.get("tier") or "")
            if target and tier in _VALID_PRIORITY_TIER:
                priorities.append((target, tier))
        elif kind == "superiority":
            # `A > B` intra-tier superiority: a defeats b at the same tier. The
            # parser carried the ordered pair on `attributes` (a/b); fall back to
            # `decl.nets`. A self-pair (a == b) is vacuous and dropped. Cycle
            # detection runs AFTER all declarations are gathered (see below).
            attrs = decl.attributes or {}
            a = str(attrs.get("a") or (decl.nets[0] if decl.nets else ""))
            b = str(attrs.get("b") or (decl.nets[1] if len(decl.nets) > 1 else ""))
            if a and b and a != b:
                superiority.append((a, b))
        elif kind == "guard_ring":
            # `@guard_ring(group=(...), well=<p|n>, tie=<net>)`: a cluster tap ring.
            # The parser carried the member device labels on `attributes['members']`
            # (fall back to `decl.nets`), the declared polarity on `attributes['well']`
            # and the tie net on `attributes['tie']`. `well` must be one of {p, n};
            # any other value drops the ring (the grammar CNAME allows anything, so
            # stay defensive -- a bogus polarity never becomes a fabricated ring). An
            # empty member list is vacuous and dropped. Names are DATA.
            attrs = decl.attributes or {}
            members = tuple(str(m) for m in (attrs.get("members") or decl.nets or ()))
            well = str(attrs.get("well") or "").strip().lower()
            tie = str(attrs.get("tie") or "").strip()
            if members and well in ("p", "n"):
                guard_rings.append(
                    GuardRingIntent(
                        members=members,
                        well=well,
                        tie_net=tie,
                        hug=str(attrs.get("hug") or "").strip().lower(),
                        role=str(attrs.get("role") or "").strip().lower(),
                        enclose=str(attrs.get("enclose") or "").strip().lower(),
                        pack=str(attrs.get("pack") or "").strip().lower(),
                    )
                )
        elif kind == "array":
            # `@array(group, style=, rows=, order=)`. The parser carried the group
            # + kwargs on `attributes`. A missing group is vacuous and dropped
            # (the grammar requires a head, but stay defensive). `rows` defaults to
            # 1; a non-integer rows is dropped to the default. Names are DATA.
            attrs = decl.attributes or {}
            group = str(attrs.get("group") or "")
            if group:
                coverage_keys = ("source_contact_coverage", "drain_contact_coverage")
                present = tuple(key in attrs for key in coverage_keys)
                if any(present) and not all(present):
                    raise ValueError("source and drain contact coverage must be declared together")
                coverage_profile = tuple(attrs[key] for key in coverage_keys) if all(present) else None
                access_profile = attrs.get("access_profile", "")
                if type(access_profile) is not str:
                    raise TypeError("native access profile must be an exact string")
                generation_profile = attrs.get("generation_profile", "")
                if type(generation_profile) is not str:
                    raise TypeError("native generation profile must be an exact string")
                synthesis = attrs.get("synthesis", "")
                if type(synthesis) is not str:
                    raise TypeError("array synthesis must be an exact string")
                tap_rail = attrs.get("tap_rail", "")
                if type(tap_rail) is not str:
                    raise TypeError("array tap_rail must be an exact string")
                raw_occurrences = attrs.get("occurrences", ())
                if type(raw_occurrences) is not tuple:
                    raise ValueError("array occurrences must be a parenthesized owner list")
                ordinals: Dict[str, int] = {}
                occurrences = []
                for owner in raw_occurrences:
                    if type(owner) is not str:
                        raise TypeError("array occurrence owner must be an exact string")
                    ordinal = ordinals.get(owner, 0)
                    occurrences.append(MosaicOccurrence(owner, ordinal, "R0"))
                    ordinals[owner] = ordinal + 1
                style_str = str(attrs.get("style") or "")
                order_str = str(attrs.get("order") or "")
                if style_str == "native_diffusion" and synthesis and not order_str:
                    order_str = "ratio"
                if style_str in {"native_diffusion", "native_cmos"} and attrs.get("rows") is not None:
                    try:
                        native_rows = int(str(attrs["rows"]))
                    except (TypeError, ValueError) as exc:
                        raise ValueError("native diffusion rows must be an integer") from exc
                else:
                    native_rows = _coerce_int(attrs.get("rows"), 1)
                arrays.append(
                    ArrayIntent(
                        group=group,
                        style=style_str,
                        rows=native_rows,
                        order=order_str,
                        transforms=(
                            tuple(str(item) for item in attrs.get("transforms"))
                            if type(attrs.get("transforms")) is tuple
                            else ()
                        ),
                        align=str(attrs.get("align") or ""),
                        pin_facing=str(attrs.get("pin_facing") or ""),
                        bank_interconnect=bank_interconnect_from_attributes(attrs),
                        occurrences=tuple(occurrences),
                        coverage_profile=coverage_profile,
                        access_profile=access_profile,
                        generation_profile=generation_profile,
                        synthesis=synthesis,
                        tap_rail=tap_rail,
                    )
                )
        elif kind == "common_centroid":
            # `@common_centroid(group, rows=, orientation=, first_member=,
            # member_transforms=)`: one rigid CC macro with coordinate-free phase.
            # one rigid CC macro with an optional coordinate-free member phase.
            attrs = decl.attributes or {}
            group = str(attrs.get("group") or "")
            if group:
                common_centroids.append(
                    CommonCentroidIntent(
                        group=group,
                        rows=_coerce_int(attrs.get("rows"), 2),
                        orientation=str(attrs.get("orientation") or "R0"),
                        first_member=str(attrs.get("first_member") or ""),
                        member_transforms=(
                            tuple(str(item) for item in attrs.get("member_transforms"))
                            if type(attrs.get("member_transforms")) is tuple
                            else ()
                        ),
                    )
                )
        elif kind == "dummy_flank":
            # `@dummy_flank(target, per_end=, tie=, share_diffusion=)`.
            attrs = decl.attributes or {}
            target = str(attrs.get("target") or "")
            if target:
                dummy_flanks.append(
                    DummyFlankIntent(
                        target=target,
                        per_end=_coerce_int(attrs.get("per_end"), 1),
                        tie=str(attrs.get("tie") or ""),
                        share_diffusion=_coerce_bool(
                            attrs.get("share_diffusion"), True
                        ),
                    )
                )
        elif kind == "well_island":
            # `@well_island(group, body=)`: own well tied to `body`.
            attrs = decl.attributes or {}
            group = str(attrs.get("group") or "")
            if group:
                well_islands.append(
                    WellIslandIntent(group=group, body=str(attrs.get("body") or ""))
                )
        elif kind == "well_island_share":
            attrs = decl.attributes or {}
            well_island_shares.append(
                WellIslandShareIntent(
                    group=attrs.get("group"),
                    body=attrs.get("body"),
                )
            )
        elif kind == "band":
            # `@band(top=, bottom=, center=)`: rail-side banding (all-keyword).
            attrs = decl.attributes or {}
            top = str(attrs.get("top") or "")
            bottom = str(attrs.get("bottom") or "")
            center = str(attrs.get("center") or "")
            if top or bottom or center:
                bands.append(BandIntent(top=top, bottom=bottom, center=center))
        elif kind == "rail_affinity":
            # `@rail_affinity(group, side=top|bottom)`.
            attrs = decl.attributes or {}
            group = str(attrs.get("group") or "")
            if group:
                rail_affinities.append(
                    RailAffinityIntent(group=group, side=str(attrs.get("side") or ""))
                )
        elif kind == "trunk":
            # `@trunk(net, layer=, orient= [, width=] [, edge=])`: signal-trunk
            # layer + orientation, optional width (um) + cell edge for a supply rail.
            attrs = decl.attributes or {}
            net = str(attrs.get("net") or "")
            if net:
                trunks.append(
                    TrunkIntent(
                        net=net,
                        layer=str(attrs.get("layer") or ""),
                        orient=str(attrs.get("orient") or ""),
                        width_um=_coerce_float(attrs.get("width"), 0.0),
                        edge=str(attrs.get("edge") or ""),
                        offset_um=float(attrs.get("offset") or 0.0),
                    )
                )
        elif kind == "power_plane":
            # `@power_plane(net, layer=M5|M6)`: upper-metal power plane.
            attrs = decl.attributes or {}
            net = str(attrs.get("net") or "")
            if net:
                power_planes.append(
                    PowerPlaneIntent(net=net, layer=str(attrs.get("layer") or ""))
                )
        elif kind == "split":
            # `@split(passive, n=, match=series|parallel)`: matched split.
            attrs = decl.attributes or {}
            passive = str(attrs.get("passive") or "")
            if passive:
                splits.append(
                    SplitIntent(
                        passive=passive,
                        n=_coerce_int(attrs.get("n"), 1),
                        match=str(attrs.get("match") or ""),
                    )
                )
        elif kind == "passive_bank_partition":
            attrs = decl.attributes or {}
            passive_bank_partitions.append(
                PassiveBankPartitionIntent(
                    parent=attrs.get("parent"),
                    bank=attrs.get("bank"),
                    members=attrs.get("members"),
                )
            )
        elif kind == "row_bounded_cavity":
            attrs = decl.attributes
            if type(attrs) is not dict or set(attrs) != {"host", "guests"}:
                raise ValueError(
                    "row_bounded_cavity requires exactly host and guests"
                )
            row_bounded_cavities.append(
                RowBoundedCavityIntent(
                    host=attrs["host"],
                    guests=attrs["guests"],
                )
            )
        elif kind == "anchor":
            # `@anchor(target, edge=left|right|top|bottom)`: edge-access placement.
            # `edge` must be a valid cell edge; any other value drops the anchor (the
            # grammar CNAME allows anything, so stay defensive -- a bogus edge never
            # fabricates a placement pull). A missing target is vacuous and dropped.
            attrs = decl.attributes or {}
            target = str(attrs.get("target") or "")
            edge = str(attrs.get("edge") or "").strip().lower()
            if target and edge in ("left", "right", "top", "bottom"):
                anchors.append(AnchorIntent(target=target, edge=edge))
        elif kind == "channel":
            # `@channel(orient=vertical|horizontal, width=<um>)`: reserved routing
            # corridor. `orient` must be a valid axis and `width` positive; otherwise
            # the channel is vacuous and dropped (a bogus orient / non-positive width
            # never fabricates a reserved corridor).
            attrs = decl.attributes or {}
            orient = str(attrs.get("orient") or "").strip().lower()
            width = _coerce_float(attrs.get("width"), 0.0)
            if orient in ("vertical", "horizontal") and width > 0:
                channels.append(ChannelIntent(orient=orient, width_um=width))
        elif kind == "density":
            attrs = decl.attributes
            if type(attrs) is not dict:
                raise TypeError("density attributes must be an exact dict")
            required = {"max_whitespace_fraction", "scope"}
            if set(attrs) != required:
                raise ValueError(
                    "density requires exactly max_whitespace_fraction and scope"
                )
            if densities:
                raise ValueError("density intent supports exactly one declaration")
            raw_fraction = attrs["max_whitespace_fraction"]
            if type(raw_fraction) is not str:
                raise TypeError(
                    "density max_whitespace_fraction source value must be a string"
                )
            try:
                fraction = float(raw_fraction)
            except ValueError as exc:
                raise ValueError(
                    "density max_whitespace_fraction must be numeric"
                ) from exc
            densities.append(
                DensityIntent(
                    max_whitespace_fraction=fraction,
                    scope=attrs["scope"],
                )
            )
        elif kind == "od_fill":
            attrs = decl.attributes
            if type(attrs) is not dict:
                raise TypeError("od_fill attributes must be an exact dict")
            required = {"min_fraction", "scope"}
            if set(attrs) != required:
                raise ValueError("od_fill requires exactly min_fraction and scope")
            if od_fills:
                raise ValueError("od_fill intent supports exactly one declaration")
            raw_fraction = attrs["min_fraction"]
            if type(raw_fraction) is not str:
                raise TypeError("od_fill min_fraction source value must be a string")
            try:
                fraction = float(raw_fraction)
            except ValueError as exc:
                raise ValueError("od_fill min_fraction must be numeric") from exc
            od_fills.append(
                OdFillIntent(min_fraction=fraction, scope=attrs["scope"])
            )
        elif kind == "power_budget":
            attrs = decl.attributes
            if type(attrs) is not dict:
                raise TypeError("power_budget attributes must be an exact dict")
            required = {"net", "current_a", "max_drop_v"}
            if set(attrs) != required:
                raise ValueError(
                    "power_budget requires exactly net, current_a, and max_drop_v"
                )
            for name in ("current_a", "max_drop_v"):
                if type(attrs[name]) is not str:
                    raise TypeError(f"power_budget {name} source value must be a string")
            power_budgets.append(
                PowerBudgetIntent(
                    net=attrs["net"],
                    current_a=float(attrs["current_a"]),
                    max_drop_v=float(attrs["max_drop_v"]),
                )
            )
        elif kind == "sensitive_net_budget":
            attrs = decl.attributes
            if type(attrs) is not dict:
                raise TypeError(
                    "sensitive_net_budget attributes must be an exact dict"
                )
            required = {"net", "max_resistance_ohm", "max_capacitance_f"}
            if set(attrs) != required:
                raise ValueError(
                    "sensitive_net_budget requires exactly net, "
                    "max_resistance_ohm, and max_capacitance_f"
                )
            for name in ("max_resistance_ohm", "max_capacitance_f"):
                if type(attrs[name]) is not str:
                    raise TypeError(
                        f"sensitive_net_budget {name} source value must be a string"
                    )
            sensitive_net_budgets.append(
                SensitiveNetBudgetIntent(
                    net=attrs["net"],
                    max_resistance_ohm=float(attrs["max_resistance_ohm"]),
                    max_capacitance_f=float(attrs["max_capacitance_f"]),
                )
            )
        elif kind == "kelvin":
            attrs = decl.attributes
            if type(attrs) is not dict or set(attrs) != {
                "force_net",
                "sense_net",
                "target_device",
            }:
                raise ValueError("Kelvin intent requires force, sense, and target")
            kelvins.append(
                KelvinIntent(
                    force_net=attrs["force_net"],
                    sense_net=attrs["sense_net"],
                    target_device=attrs["target_device"],
                )
            )
        else:
            # Unrecognized future declaration kinds retain the legacy fallback.
            continue

    pairs = _normalize_pairs(diff_pairs)
    mpairs = _normalize_pairs(mirror_pairs)
    explicit_keys = {_budget_key(b) for b in explicit_budgets}
    # A two-signal-net parasitic_cap budget already measures the same pair's
    # node-cap balance, so it suppresses the default matched_cc_balance budget
    # for that pair (avoids two near-identical obligations on one pair).
    pc_pair_keys = {
        frozenset(b.pair)
        for b in explicit_budgets
        if b.kind == "parasitic_cap" and not parasitic_cap_is_single_ended(b.pair)
    }
    default_budgets = [
        ParasiticBudget(
            "matched_cc_balance", pair, default_match_tol, source="declared"
        )
        for pair in pairs
        if ("matched_cc_balance", frozenset(pair)) not in explicit_keys
        and frozenset(pair) not in pc_pair_keys
    ]

    return LayoutIntent(
        cell=cell,
        dynamic_nets=(), sensitive_nets=(),
        diff_pairs=pairs,
        budgets=_dedupe_budgets_prefer_later(default_budgets + explicit_budgets),
        shields=_dedupe_shields_prefer_later(shields),
        mirror_pairs=mpairs,
        source="declared", match_scopes=match_scopes,
        objective=objective,
        area_max_um2=area_max_um2,
        place_groups=tuple(place_groups),
        route_groups=tuple(route_groups),
        resistance_budgets=tuple(resistance_budgets),
        lde_budgets=_dedupe_lde_budgets_prefer_later(lde_budgets),
        wire_widths=dict(wire_widths),
        match_length_pairs=tuple(match_length_pairs),
        net_spacings=tuple(net_spacings),
        channel_routes=channel_routes_from_declarations(declarations, place_groups),
        imported_layout=imported_routes_from_declarations(declarations)[0],
        route_tails=imported_routes_from_declarations(declarations)[1],
        order_constraints=_dedupe_preserve_order(order_constraints),
        edge_alignments=replay_edge_alignment_inventory(tuple(edge_alignments)),
        group_center_alignments=tuple(group_center_alignments),
        group_edge_alignments=tuple(group_edge_alignments),
        group_separations=tuple(group_separations),
        channel_alignments=tuple(channel_alignments),
        terminal_accesses=replay_terminal_access_inventory(tuple(terminal_accesses)),
        abut_pairs=_dedupe_preserve_order(abut_pairs),
        priorities=_dedupe_preserve_order(priorities),
        superiority=_assert_acyclic_superiority(_dedupe_preserve_order(superiority)),
        guard_rings=tuple(guard_rings),
        arrays=tuple(arrays),
        common_centroids=tuple(common_centroids),
        dummy_flanks=tuple(dummy_flanks),
        well_islands=tuple(well_islands),
        well_island_shares=tuple(
            sorted(well_island_shares, key=lambda item: item.group)
        ),
        bands=tuple(bands),
        rail_affinities=tuple(rail_affinities),
        trunks=tuple(trunks),
        power_planes=tuple(power_planes),
        splits=tuple(splits),
        passive_bank_partitions=tuple(
            sorted(
                passive_bank_partitions,
                key=lambda item: (item.parent, item.bank),
            )
        ),
        row_bounded_cavities=replay_row_bounded_cavity_inventory(
            tuple(row_bounded_cavities)
        ),
        anchors=tuple(anchors),
        channels=tuple(channels),
        densities=tuple(densities),
        od_fills=tuple(od_fills),
        power_budgets=tuple(power_budgets),
        sensitive_net_budgets=tuple(sensitive_net_budgets),
        kelvins=tuple(kelvins),
    )


# The valid priority tiers, imported from the single source of truth in the
# priority module so the lowering and the resolver agree on the vocabulary (no
# second tier list). A bogus tier on a `@priority` declaration is dropped at
# lowering rather than fabricating an out-of-vocabulary priority.
from clay.layout.spec.priority import VALID_TIER as _VALID_PRIORITY_TIER


def _assert_acyclic_superiority(
    edges: Tuple[Tuple[str, str], ...],
) -> Tuple[Tuple[str, str], ...]:
    """Validate that a declared superiority partial order is ACYCLIC.

    Each ``(a, b)`` edge means ``a`` defeats ``b`` at the same tier (the
    resolver's declared-superiority branch). Defeasible logic requires ``>``
    acyclic: a declared cycle (``A > B > C > A``) makes the partial order
    ill-formed. A cycle RAISES ``ValueError`` at lowering -- a contract violation
    surfaced explicitly, never a silent drop or an arrival-order tie-break.

    Returns the edges unchanged when acyclic. Detection is a Kahn topological
    sort over the directed graph (no recursion / pure stdlib).
    """
    if not edges:
        return edges
    # build adjacency + in-degree over the node set the edges mention.
    succ: Dict[str, List[str]] = {}
    indeg: Dict[str, int] = {}
    for a, b in edges:
        succ.setdefault(a, [])
        succ.setdefault(b, [])
        indeg.setdefault(a, 0)
        indeg.setdefault(b, 0)
    for a, b in edges:
        succ[a].append(b)
        indeg[b] += 1
    # Kahn: repeatedly remove a zero-in-degree node; a remaining node => a cycle.
    queue = [n for n, d in indeg.items() if d == 0]
    removed = 0
    while queue:
        n = queue.pop()
        removed += 1
        for m in succ[n]:
            indeg[m] -= 1
            if indeg[m] == 0:
                queue.append(m)
    if removed != len(indeg):
        in_cycle = sorted(n for n, d in indeg.items() if d > 0)
        raise ValueError(
            "superiority relation must be acyclic (defeasible '>' is a partial "
            f"order); a declared cycle involves: {in_cycle}. Declared edges: "
            f"{sorted(edges)}."
        )
    return edges


def _dedupe_preserve_order(items: Iterable) -> Tuple:
    """Drop exact-duplicate tuples, keeping first-seen order (deterministic).

    Used for ``order_constraints`` / ``abut_pairs`` where a repeated identical
    declaration is redundant but not an error; the first occurrence wins.
    """
    out: List = []
    seen: set = set()
    for it in items:
        key = tuple(it)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return tuple(out)


def _lde_budget_key(budget: LdeBudget) -> Tuple[str, frozenset[str]]:
    return (budget.metric, frozenset(budget.pair))


def _dedupe_lde_budgets_prefer_later(
    budgets: Iterable[LdeBudget],
) -> Tuple[LdeBudget, ...]:
    """Collapse duplicate (metric, device-pair) LDE budgets, last declaration wins."""
    out: List[LdeBudget] = []
    index: Dict[Tuple[str, frozenset[str]], int] = {}
    for budget in budgets:
        key = _lde_budget_key(budget)
        if key in index:
            out[index[key]] = budget
        else:
            index[key] = len(out)
            out.append(budget)
    return tuple(out)


def _deriver_key_for_cell(cell_name: str) -> Optional[str]:
    """Resolve a `_DERIVERS` key for a parsed cell name, or None.

    The deriver registry is keyed by canonical category names
    (``comparator`` / ``diff_ota`` / ``five_t_ota`` / ``folded_cascode`` /
    ``current_mirror``); a parsed cell is named by its design id
    (``folded_cascode_ota``, ``ldo_gf180``, ...). Match by exact key first, then
    by a deriver key being a SUBSTRING of the (lower-cased) cell name so
    ``folded_cascode_ota`` resolves to ``folded_cascode``. Returns ``None`` when
    no deriver category applies (the common case for the gf180 circuits, which
    are fully parse-driven).
    """
    name = (cell_name or "").lower()
    if name in _DERIVERS:
        return name
    for key in _DERIVERS:
        if key in name:
            return key
    return None


def matching_intents_from_cell_def(cd: CellDef) -> Tuple[MatchingIntent, ...]:
    """Canonicalize device ``@matching`` declarations without losing direction.

    Source IDs are semantic and stable across parser/runtime instances:
    ``matching:<cell>:<target>-><other>``.  Reciprocal declarations collapse to
    one unordered pair whose ``declaration_ids`` contains both IDs.  Repeating
    the same directed declaration is rejected because it would make declaration
    coverage ambiguous rather than creating a second proof token.
    """
    from clay.layout.annotations import Interdigitated, Matching

    raw = getattr(cd, "__dict__", {}).get("_layout_annotations", {}) or {}
    classes = {
        str(getattr(device, "instance_name", "")): str(
            getattr(device, "device_class", "") or ""
        ).lower()
        for device in (getattr(cd, "devices", ()) or ())
        if getattr(device, "instance_name", None)
    }
    interdigitated: Dict[
        Tuple[str, str], List[Tuple[str, str, int]]
    ] = {}
    interdigitated_seen: set[str] = set()
    for target, annotations in raw.items():
        for annotation in annotations or ():
            if isinstance(annotation, Interdigitated):
                left = str(target)
                right = str(annotation.with_device)
                if right == "__pattern_only__":
                    raise ValueError(
                        "@interdigitated pattern-only form has no exact partner"
                    )
                pair = tuple(sorted((left, right)))
                if pair[0] == pair[1]:
                    raise ValueError(
                        f"self interdigitation declaration is invalid: {left!r}"
                    )
                declaration_id = (
                    f"interdigitated:{cd.local_name}:{left}->{right}"
                )
                if declaration_id in interdigitated_seen:
                    raise ValueError(
                        f"duplicate directed @interdigitated: {declaration_id}"
                    )
                interdigitated_seen.add(declaration_id)
                finger_pairs = getattr(annotation, "finger_pairs", None)
                if type(finger_pairs) is not int or not 1 <= finger_pairs <= 256:
                    raise ValueError(
                        "@interdigitated finger_pairs must be an exact integer "
                        "in [1, 256]"
                    )
                interdigitated.setdefault(pair, []).append(
                    (
                        declaration_id,
                        str(annotation.pattern.value),
                        finger_pairs,
                    )
                )

    by_pair: Dict[Tuple[str, str], List[str]] = {}
    directed_seen: set[str] = set()
    for target, annotations in raw.items():
        for annotation in annotations or ():
            if not isinstance(annotation, Matching):
                continue
            left = str(target)
            right = str(annotation.other)
            pair = tuple(sorted((left, right)))
            if pair[0] == pair[1]:
                raise ValueError(f"self matching declaration is invalid: {left!r}")
            declaration_id = f"matching:{cd.local_name}:{left}->{right}"
            if declaration_id in directed_seen:
                raise ValueError(f"duplicate directed @matching: {declaration_id}")
            directed_seen.add(declaration_id)
            by_pair.setdefault(pair, []).append(declaration_id)

    out: List[MatchingIntent] = []
    for pair in sorted(set(by_pair) | set(interdigitated)):
        interdigitation_records = tuple(
            sorted(interdigitated.get(pair, ()))
        )
        patterns = {record[1] for record in interdigitation_records}
        finger_pair_counts = {record[2] for record in interdigitation_records}
        if len(patterns) > 1 or len(finger_pair_counts) > 1:
            raise ValueError(
                f"conflicting @interdigitated declarations for pair {pair!r}: "
                f"patterns={tuple(sorted(patterns))!r}, "
                f"finger_pairs={tuple(sorted(finger_pair_counts))!r}"
            )
        interdigitation_ids = tuple(
            record[0] for record in interdigitation_records
        )
        matching_ids = tuple(sorted(by_pair.get(pair, ())))
        out.append(
            MatchingIntent(
                members=pair,
                declaration_ids=(matching_ids or interdigitation_ids),
                device_classes=(classes.get(pair[0], ""), classes.get(pair[1], "")),
                requires_interdigitation=bool(interdigitation_records),
                interdigitation_declaration_ids=interdigitation_ids,
                interdigitation_pattern=(next(iter(patterns)) if patterns else ""),
                interdigitation_finger_pairs=(
                    next(iter(finger_pair_counts)) if finger_pair_counts else 0
                ),
            )
        )
    return tuple(out)


def adjacent_intents_from_cell_def(cd: CellDef) -> Tuple[AdjacentIntent, ...]:
    """Replay device-level ``@adjacent`` annotations without dropping ratios."""

    from clay.layout.annotations import Adjacent

    raw = getattr(cd, "__dict__", {}).get("_layout_annotations", {}) or {}
    source_names = {
        str(device.instance_name)
        for device in tuple(getattr(cd, "devices", ()) or ())
        if getattr(device, "instance_name", None)
    }
    declarations: list[AdjacentIntent] = []
    for source, annotations in raw.items():
        for annotation in annotations or ():
            if not isinstance(annotation, Adjacent):
                continue
            target = annotation.to
            if target == "__unspecified__":
                raise ValueError("bare @adjacent has no materializable target")
            if str(source) not in source_names or str(target) not in source_names:
                raise ValueError(
                    f"@adjacent endpoint is absent from cell {cd.local_name!r}: "
                    f"{source!r}->{target!r}"
                )
            declarations.append(
                AdjacentIntent(
                    source=str(source),
                    target=str(target),
                    max_separation_multiplier=annotation.max_separation_multiplier,
                )
            )
    return replay_adjacent_inventory(
        tuple(sorted(declarations, key=lambda item: (item.source, item.target)))
    )


def layout_intent_from_cell_def(cd: CellDef) -> LayoutIntent:
    """Build a :class:`LayoutIntent` for a parsed cell — parse-driven first.

    The parsed ``.clay`` ``layout_intent`` block is the SOURCE OF TRUTH: when the
    cell declares ANY layout intent, that declared intent IS the result and the
    hardcoded ``_DERIVERS`` registry is NOT consulted. The derivers build their
    intent off SYNTHETIC template nets (``inp`` / ``outp`` / ``casn`` ...) that
    do not exist in a real design's extraction, so merging one over a declared
    intent would inject phantom, never-extractable nets and block sign-off; the
    deriver is therefore a fallback used ONLY when the cell declares no intent:

      * cell declares layout intent -> return the lowered declared intent;
      * cell declares NO layout intent AND a deriver category matches the cell
        name -> return that deriver's auto-derived intent, re-labelled to the
        parsed cell name;
      * cell declares NO layout intent and NO deriver matches -> return the empty
        declared intent (nothing to verify; never crashes).
    """
    from dataclasses import replace

    from clay.well_isolation import declared_well_isolated_devices

    isolated_devices = declared_well_isolated_devices(cd)
    matching_intents = matching_intents_from_cell_def(cd)
    adjacencies = adjacent_intents_from_cell_def(cd)

    def _attach_cell_annotations(intent: LayoutIntent) -> LayoutIntent:
        if any(row.style == "native_cmos" for row in intent.arrays):
            from clay.layout.native_cmos_gate_composer import expand_cmos_array_intents

            intent = expand_cmos_array_intents(cd, intent)
        if (
            intent.well_isolated_devices == isolated_devices
            and intent.matching_intents == matching_intents
            and intent.adjacencies == adjacencies
        ):
            return intent
        return replace(
            intent,
            well_isolated_devices=isolated_devices,
            matching_intents=matching_intents,
            adjacencies=adjacencies,
        )

    decls = getattr(cd, "layout_intents", []) or []
    if decls:
        # NEVER SILENT PASS: a declared @symmetric / @match / @mirror / @shield /
        # @budget that names a net (or an @equal_* / @place_group / @order /
        # @abut that names a device) absent from this cell's parsed topology
        # would otherwise be lowered into a diff-pair / budget over a net that is
        # never routed, mirrored, or extracted -- the intent silently no-ops with
        # no verdict. `assert_intent_net_binding` turns that silent drop into a
        # named RED verdict (raises IntentNetBindingError) for a cell whose net /
        # device universe is DETERMINED from the parse (explicit `device`
        # topology); a synthesis-driven cell (functional roles / stage style /
        # instances) is UNMEASURABLE and is not raised on, since its internal
        # nodes materialise only after compilation. Keyed purely on parsed
        # structure -- no circuit / PDK / net-label branch.
        from clay.layout.intent_net_binding import assert_intent_net_binding

        assert_intent_net_binding(cd)
        intent = layout_intent_from_declarations(cd.local_name, decls)
        from clay.layout.intent_alignment_binding import reconcile_source_alignments

        reconcile_source_alignments(decls, intent)
        return _attach_cell_annotations(intent)

    deriver_key = _deriver_key_for_cell(cd.local_name)
    if deriver_key is not None:
        # Re-label to the parsed cell name so the record/digest names the design,
        # not the deriver category.
        return _attach_cell_annotations(
            _relabel_intent(_DERIVERS[deriver_key](), cd.local_name)
        )
    return _attach_cell_annotations(
        layout_intent_from_declarations(cd.local_name, decls)
    )


def _relabel_intent(intent: LayoutIntent, cell: str) -> LayoutIntent:
    """Return a copy of ``intent`` with its ``cell`` field set to ``cell``."""
    from dataclasses import replace

    if intent.cell == cell:
        return intent
    return replace(intent, cell=cell)


def parse_layout_intent(source: str, *, cell: Optional[str] = None) -> LayoutIntent:
    """Parse a `.clay` source string and return its declared LayoutIntent.

    The source must contain exactly one cell with layout-intent declarations,
    unless `cell=` names the target cell explicitly.
    """
    from clay.parser import parse_clay_source

    cf = parse_clay_source(source)
    if cell is not None:
        matches = [cd for cd in cf.cell_defs if cd.local_name == cell]
        if not matches:
            raise ValueError(f"no cell named {cell!r} in source")
        cd = matches[0]
        if not getattr(cd, "layout_intents", None):
            raise ValueError(f"cell {cell!r} has no layout_intent declarations")
        return layout_intent_from_cell_def(cd)

    candidates = [cd for cd in cf.cell_defs if getattr(cd, "layout_intents", None)]
    if len(candidates) != 1:
        raise ValueError(
            "parse_layout_intent requires exactly one cell with "
            f"layout_intent declarations, got {[cd.local_name for cd in candidates]}"
        )
    return layout_intent_from_cell_def(candidates[0])


from clay.layout.spatial.intent_merge import (
    _merge_aggregate_alignment_inventories, _merge_reference_conditioned_placement_obligations,
)


def merge_layout_intent(
    auto: LayoutIntent, declared: Optional[LayoutIntent]
) -> LayoutIntent:
    """Merge designer-declared intent over auto-derived intent.

    Declared budgets replace auto budgets with the same `(kind, pair)` key.
    Declared shields replace auto shields with the same `(aggressor, victim)`
    key. Unmentioned auto-derived classes, pairs, budgets, and shields stay.
    """
    if declared is None:
        return auto
    if declared.cell != auto.cell:
        raise ValueError(
            f"cannot merge layout intent for cell {declared.cell!r} "
            f"into auto intent for {auto.cell!r}"
        )

    # The declared layer's source/profile_id wins when supplied (a designer
    # override is more authoritative than the auto-derived label); fall back to
    # the auto layer otherwise.
    declared_source = (
        declared.source if declared.source not in ("", "auto", None) else None
    )
    if (
        auto.hierarchical_macro_dag is not None
        and declared.hierarchical_macro_dag is not None
        and auto.hierarchical_macro_dag != declared.hierarchical_macro_dag
    ):
        raise ValueError("conflicting hierarchical macro DAG declarations")
    return LayoutIntent(
        cell=auto.cell,
        dynamic_nets=tuple(dict.fromkeys(auto.dynamic_nets + declared.dynamic_nets)),
        sensitive_nets=tuple(
            dict.fromkeys(auto.sensitive_nets + declared.sensitive_nets)
        ),
        diff_pairs=_normalize_pairs(auto.diff_pairs + declared.diff_pairs),
        match_scopes=merge_match_scopes(auto.match_scopes, declared.match_scopes),
        budgets=_dedupe_budgets_prefer_later(auto.budgets + declared.budgets),
        shields=_dedupe_shields_prefer_later(auto.shields + declared.shields),
        mirror_pairs=_normalize_pairs(auto.mirror_pairs + declared.mirror_pairs),
        profile_id=declared.profile_id or auto.profile_id,
        source=declared_source or auto.source,
        lde_budgets=_dedupe_lde_budgets_prefer_later(
            auto.lde_budgets + declared.lde_budgets
        ),
        # Declared per-net widths override the auto layer's for the same net; an
        # unmentioned auto width stays. (Auto derivers currently declare none.)
        wire_widths={**auto.wire_widths, **declared.wire_widths},
        net_spacings=auto.net_spacings + declared.net_spacings,
        channel_routes=canonical_channel_routes(tuple(dict.fromkeys(
            auto.channel_routes + declared.channel_routes))),
        imported_layout=merge_imported_routes(auto.imported_layout, auto.route_tails,
                                             declared.imported_layout, declared.route_tails)[0],
        route_tails=merge_imported_routes(auto.imported_layout, auto.route_tails,
                                         declared.imported_layout, declared.route_tails)[1],
        # Matched-length pairs drive routing and final verification; union the
        # auto + declared pairs (auto derivers currently declare none), deduped.
        match_length_pairs=tuple(
            dict.fromkeys(auto.match_length_pairs + declared.match_length_pairs)
        ),
        edge_alignments=replay_edge_alignment_inventory(
            tuple(dict.fromkeys(auto.edge_alignments + declared.edge_alignments))
        ),
        group_center_alignments=tuple(dict.fromkeys(
            auto.group_center_alignments + declared.group_center_alignments)),
        group_edge_alignments=tuple(dict.fromkeys(auto.group_edge_alignments + declared.group_edge_alignments)),
        group_separations=tuple(dict.fromkeys(auto.group_separations + declared.group_separations)),
        channel_alignments=tuple(dict.fromkeys(auto.channel_alignments + declared.channel_alignments)),
        terminal_accesses=replay_terminal_access_inventory(
            tuple(dict.fromkeys(auto.terminal_accesses + declared.terminal_accesses))
        ),
        passive_bank_partitions=tuple(
            sorted(
                {
                    (item.parent, item.bank): item
                    for item in (
                        auto.passive_bank_partitions
                        + declared.passive_bank_partitions
                    )
                }.values(),
                key=lambda item: (item.parent, item.bank),
            )
        ),
        well_island_shares=tuple(
            sorted(
                {
                    item.group: item
                    for item in (
                        auto.well_island_shares + declared.well_island_shares
                    )
                }.values(),
                key=lambda item: item.group,
            )
        ),
        densities=(
            _replay_density_inventory(declared.densities)
            or _replay_density_inventory(auto.densities)
        ),
        od_fills=(
            _replay_od_fill_inventory(declared.od_fills)
            or _replay_od_fill_inventory(auto.od_fills)
        ),
        power_budgets=tuple(
            {
                item.net: item
                for item in (
                    tuple(
                        _replay_electrical_budget_inventory(
                            auto.power_budgets, power=True
                        )
                    )
                    + tuple(
                        _replay_electrical_budget_inventory(
                            declared.power_budgets, power=True
                        )
                    )
                )
            }.values()
        ),
        sensitive_net_budgets=tuple(
            {
                item.net: item
                for item in (
                    tuple(
                        _replay_electrical_budget_inventory(
                            auto.sensitive_net_budgets, power=False
                        )
                    )
                    + tuple(
                        _replay_electrical_budget_inventory(
                            declared.sensitive_net_budgets, power=False
                        )
                    )
                )
            }.values()
        ),
        kelvins=replay_kelvin_inventory(auto.kelvins + declared.kelvins),
        discrete_mosaics=replay_discrete_mosaic_inventory(tuple(
            {
                item.mosaic_id: item
                for item in (*auto.discrete_mosaics, *declared.discrete_mosaics)
            }.values()
        )),
        mosaic_guest_slots=replay_mosaic_guest_slot_inventory(tuple(
            {row.sha256: row for row in (*auto.mosaic_guest_slots, *declared.mosaic_guest_slots)}.values()
        )),
        mosaic_boundary_joins=merge_mosaic_boundary_join_inventories(
            auto.mosaic_boundary_joins, declared.mosaic_boundary_joins),
        mosaic_center_alignments=replay_mosaic_center_alignment_inventory(tuple(
            {
                item.alignment_id: item
                for item in (
                    *auto.mosaic_center_alignments,
                    *declared.mosaic_center_alignments,
                )
            }.values()
        )),
        aggregate_orders=replay_aggregate_order_inventory(tuple(
            {
                item.order_id: item
                for item in (*auto.aggregate_orders, *declared.aggregate_orders)
            }.values()
        )),
        reference_conditioned_placement_obligations=(
            _merge_reference_conditioned_placement_obligations(
                auto.reference_conditioned_placement_obligations,
                declared.reference_conditioned_placement_obligations,
            )
        ),
        aggregate_alignments=_merge_aggregate_alignment_inventories(
            auto.aggregate_alignments,
            declared.aggregate_alignments,
        ),
        hierarchical_macro_dag=(
            declared.hierarchical_macro_dag or auto.hierarchical_macro_dag
        ),
        row_bounded_cavities=replay_row_bounded_cavity_inventory(tuple(
            {
                (item.host, item.guests): item
                for item in (
                    *auto.row_bounded_cavities,
                    *declared.row_bounded_cavities,
                )
            }.values()
        )),
    )


def derive_intent_from_roles(
    cell: str,
    *,
    role_to_nets: Optional[Dict[str, Sequence[str]]] = None,
    dynamic_nets: Iterable[str] = (),
    sensitive_nets: Iterable[str] = (),
    diff_pairs: Iterable[Sequence[str]] = (),
    mirror_pairs: Iterable[Sequence[str]] = (),
    shields: Iterable[Shield] = (),
    kickback_pair: Optional[Sequence[str]] = None,
    clk_regen_pair: Optional[Sequence[str]] = None,
    clock_net: Optional[str] = None,
    kickback_aggressors: Optional[Sequence[str]] = None,
    match_tol: float = _DEFAULT_MATCH_TOL,
    kickback_tol: float = _DEFAULT_KICKBACK_TOL,
    clkregen_tol: float = _DEFAULT_CLKREGEN_TOL,
    profile_id: Optional[str] = None,
    source: str = "auto",
) -> LayoutIntent:
    """GENERIC layout-intent builder from an already-classified netlist.

    This is the reusable core every per-cell deriver routes through, so the
    classification policy lives in ONE place. Given a net classification (which
    nets are large-swing ``dynamic`` aggressors, which are offset-critical
    ``sensitive`` victims, and the differential ``diff_pairs`` that must stay
    parasitically matched), it assembles the :class:`LayoutIntent`:

      * one ``matched_cc_balance`` budget (relative imbalance ``match_tol``) per
        DIFFERENTIAL pair -- the universal "keep the two halves matched" rule.
        A ``mirror_pairs`` entry (a single-ended-output current mirror) is NOT a
        differential pair: its two nodes are asymmetric by design, so it produces
        NO ``matched_cc_balance`` budget; it is recorded on the intent
        (``mirror_pairs``) for placement-symmetry to consume but carries no cap
        budget here;
      * an optional ``kickback`` budget on ``kickback_pair`` (the two
        externally-driven input nets whose imbalanced coupling becomes a
        differential offset) -- present iff a cell HAS such an input pair;
      * an optional ``clk_regen`` budget on ``clk_regen_pair`` (a clock/strobe
        must couple equally into the two regenerative nodes) -- present iff the
        cell HAS a clocked regenerative pair;
      * the shields verbatim.

    ``role_to_nets`` is accepted for documentation/provenance (the role->net map
    the classification came from) and is otherwise unused here; the heavy lifting
    of turning roles into the classification lives in :func:`classify_nets_by_role`.

    SINGLE-ENDED / no-diff-pair cells: an empty ``diff_pairs`` is a legitimate
    input (a single-ended amp, a bias cell, a non-differential block) -- it yields
    a valid intent with ZERO ``matched_cc_balance`` budgets rather than raising.
    A cell with no diff pairs AND no kickback/clk_regen pair declares no
    parasitic budgets at all; the obligation checker's verdict flags such an
    empty-budget cell as vacuously-but-explicitly passing (see
    :func:`~clay.layout.intent_obligations.layout_intent_verdict`) so a cell that
    declared no intent is never mistaken for a verified one.

    All tolerances are RELATIVE fractions in (0, 1) per the project-wide rule.
    """
    dynamic = tuple(dict.fromkeys(dynamic_nets))  # order-preserving dedupe
    sensitive = tuple(dict.fromkeys(sensitive_nets))
    pairs = _normalize_pairs(diff_pairs)
    mpairs = _normalize_pairs(mirror_pairs)

    agg_tuple = (
        None
        if kickback_aggressors is None
        else tuple(dict.fromkeys(kickback_aggressors))
    )
    budgets: List[ParasiticBudget] = [
        ParasiticBudget(
            "matched_cc_balance",
            p,
            match_tol,
            source=source,
            profile_id=profile_id,
        )
        for p in pairs
    ]
    if kickback_pair is not None:
        budgets.append(
            ParasiticBudget(
                "kickback",
                tuple(kickback_pair),
                kickback_tol,
                aggressors=agg_tuple,
                source=source,
                profile_id=profile_id,
            )
        )
    if clk_regen_pair is not None:
        budgets.append(
            ParasiticBudget(
                "clk_regen",
                tuple(clk_regen_pair),
                clkregen_tol,
                clock_net=clock_net,
                source=source,
                profile_id=profile_id,
            )
        )

    return LayoutIntent(
        cell=cell,
        dynamic_nets=dynamic,
        sensitive_nets=sensitive,
        diff_pairs=pairs,
        budgets=tuple(budgets),
        shields=tuple(shields),
        mirror_pairs=mpairs,
        profile_id=profile_id,
        source=source,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Topology / role-driven net classifier
# ─────────────────────────────────────────────────────────────────────────────

# Roles whose GATE net is an externally-driven analog input -> offset-critical
# (a disturbance on it is read as a differential input offset). The canonical
# member is the differential ``input_transconductor`` pair; ``diff_pair`` is the
# bare-device synonym used by the cell templates.
_SENSITIVE_INPUT_ROLES = frozenset({"input_transconductor", "diff_pair"})

# Roles whose OUTPUT net swings rail-to-rail and aggresses its neighbours ->
# dynamic. Covers the regenerative latch nodes, digital/output inverters, and any
# explicitly clocked node.
_DYNAMIC_OUTPUT_ROLES = frozenset(
    {
        "regenerative_latch",
        "regen",
        "cross_coupled_latch",
        "output_inverter",
        "output_buffer",
        "digital_output",
    }
)

# Roles that ARE a clock/strobe net (the whole net swings rail-to-rail) -> dynamic.
_CLOCK_ROLES = frozenset({"clock", "clk", "strobe", "tail_switch_clock"})


def classify_nets_with_mirrors(
    role_bindings: Sequence[Dict[str, object]],
) -> Tuple[
    Tuple[str, ...],
    Tuple[str, ...],
    Tuple[Tuple[str, str], ...],
    Tuple[Tuple[str, str], ...],
]:
    """Topology/role-driven net classifier (mirror-aware 4-tuple form).

    Consumes a list of ROLE BINDINGS and returns ``(dynamic_nets,
    sensitive_nets, diff_pairs, mirror_pairs)``. This is the full classifier;
    :func:`classify_nets_by_role` is the back-compatible 3-tuple wrapper that
    drops ``mirror_pairs`` (every legacy caller expects the 3-tuple).

    Each binding is a dict describing one functional-role instance:

      * ``kind``   -- the functional-role keyword (``input_transconductor``,
        ``regenerative_latch``, ``output_inverter``, ``current_mirror``,
        ``clock`` ...);
      * ``gate``   -- (optional) the net on the role's controlling/input gate;
      * ``output`` -- (optional) the net on the role's driven output/drain;
      * ``net``    -- (optional) the role's single defining net (used for clocks
        and other whole-net roles where gate/output don't apply);
      * ``diff_partner`` -- (optional) the ``name`` of the sibling role that is
        this role's differential mirror image. The two halves' like-nets (gate
        with gate, output with output) form the matched pairs.
      * ``match_kind`` -- (optional) ``"differential"`` (default) or ``"mirror"``.
        A ``"mirror"`` pair is a single-ended-output current mirror: its two
        NODES are asymmetric by design (one diode/reference leg, one mirrored
        output), so it goes to ``mirror_pairs`` (no cap budget) instead of
        ``diff_pairs``. Falls back to the binding ``kind`` when absent
        (``current_mirror`` -> mirror; else differential).
      * ``name``   -- (optional) a stable id used to resolve ``diff_partner``.

    Classification RULES (general; each documented inline):

      R1 SENSITIVE: the GATE net of an externally-driven input device
         (``input_transconductor`` / ``diff_pair``) is offset-critical -- a
         capacitive disturbance on it is indistinguishable from a real
         differential input, so it is a victim to be protected.
      R2 DYNAMIC: a CLOCK/strobe net (a ``clock``-family role) swings
         rail-to-rail and is the dominant aggressor.
      R3 DYNAMIC: the OUTPUT net of a regenerative-latch half, an output
         inverter/buffer, or a digital-output role swings rail-to-rail and
         aggresses its neighbours.
      R4 MATCHED PAIRS: two role bindings that name each other as ``diff_partner``
         (or each carries the other's ``name``) are mirror images; their like-nets
         pair up (gate<->gate, output<->output). A ``differential`` pair's net
         pair is a DIFF PAIR (must stay parasitically matched -> a cap budget); a
         ``mirror`` pair's net pair is a MIRROR PAIR (asymmetric nodes by design
         -> NO cap budget, recorded for placement symmetry).
      R5 MIRROR PRECEDENCE: a net pair claimed by a ``mirror`` matched pair is a
         mirror pair even if a ``differential`` role (e.g. the input pair whose
         drains ARE the single-ended-output mirror's nodes) also pairs the same
         two nets. The current-mirror load decides the node symmetry, so the
         coinciding differential output pairing is suppressed from ``diff_pairs``.

    Determinism: nets are emitted in first-seen order; pairs are deduplicated by
    unordered net membership while preserving the first occurrence's order. A net
    may legitimately end up in BOTH ``dynamic_nets`` and ``sensitive_nets`` (e.g.
    a regenerative node that both swings hard AND sets the decision) -- the
    classes are not mutually exclusive.
    """
    dynamic: List[str] = []
    sensitive: List[str] = []
    diff_set: List[Tuple[str, str]] = []
    mirror_set: List[Tuple[str, str]] = []

    def _add(bucket: List[str], net: Optional[object]) -> None:
        if net and net not in bucket:
            bucket.append(str(net))

    # Index by name so diff_partner can be resolved symmetrically.
    by_name: Dict[str, Dict[str, object]] = {}
    for b in role_bindings:
        nm = b.get("name")
        if nm is not None:
            by_name[str(nm)] = b

    # ── Per-binding net-class rules (R1-R3) ──────────────────────────────────
    for b in role_bindings:
        kind = str(b.get("kind", ""))
        gate = b.get("gate")
        output = b.get("output")
        net = b.get("net")

        # R1: input device gate -> sensitive victim.
        if kind in _SENSITIVE_INPUT_ROLES:
            _add(sensitive, gate)

        # R2: clock-family role -> dynamic aggressor (its defining net, or gate).
        if kind in _CLOCK_ROLES:
            _add(dynamic, net if net is not None else gate)

        # R3: rail-to-rail output role -> dynamic aggressor (its output net).
        if kind in _DYNAMIC_OUTPUT_ROLES:
            _add(dynamic, output if output is not None else net)
            # A regenerative node is ALSO sensitive: its balance sets the
            # decision, so an imbalanced disturbance is a built-in offset.
            if kind in {"regenerative_latch", "regen", "cross_coupled_latch"}:
                _add(sensitive, output if output is not None else net)

    # ── Matched-pair rule (R4) + mirror split (R5) ───────────────────────────
    # Collect every matched pair's net-pairs tagged differential/mirror, in
    # first-seen order, then resolve: a net pair is a mirror pair when ANY mirror
    # match claims it; otherwise it is a diff pair. This makes the mirror load
    # win over the input pair's coinciding drain pairing (R5).
    def _match_kind_of(binding: Dict[str, object]) -> str:
        mk = binding.get("match_kind")
        if mk is not None:
            return str(mk)
        # Fall back to the role kind so a hand-built binding need not carry the
        # tag: a current_mirror role is a mirror pair, everything else diff.
        return (
            "mirror"
            if str(binding.get("kind", "")) == "current_mirror"
            else "differential"
        )

    candidates: List[Tuple[Tuple[str, str], str]] = []  # (pair, match_kind)
    paired = set()
    for b in role_bindings:
        nm = b.get("name")
        partner_name = b.get("diff_partner")
        if partner_name is None:
            continue
        partner = by_name.get(str(partner_name))
        if partner is None:
            continue
        key = frozenset((str(nm), str(partner_name)))
        if key in paired:
            continue
        paired.add(key)

        mk = _match_kind_of(b)
        for fn in ("gate", "output"):
            na, nb = b.get(fn), partner.get(fn)
            if na and nb and na != nb:
                candidates.append(((str(na), str(nb)), mk))

    # Net pairs any mirror match claims (by unordered membership).
    mirror_keys = {frozenset(pair) for pair, mk in candidates if mk == "mirror"}

    seen: set = set()
    for pair, mk in candidates:
        fk = frozenset(pair)
        if fk in seen:
            continue
        if mk == "mirror" or fk in mirror_keys:
            # R5: a mirror-claimed net pair is a mirror pair (NO cap budget),
            # even when a differential role also pairs the same two nets.
            seen.add(fk)
            if pair not in mirror_set and (pair[1], pair[0]) not in mirror_set:
                mirror_set.append(pair)
        else:
            seen.add(fk)
            diff_set.append(pair)

    return tuple(dynamic), tuple(sensitive), tuple(diff_set), tuple(mirror_set)


def classify_nets_by_role(
    role_bindings: Sequence[Dict[str, object]],
) -> Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[Tuple[str, str], ...]]:
    """Back-compatible 3-tuple classifier: ``(dynamic, sensitive, diff_pairs)``.

    Thin wrapper over :func:`classify_nets_with_mirrors` that drops the
    ``mirror_pairs`` channel -- every legacy caller of this function expects the
    3-tuple. ``diff_pairs`` here are ONLY the genuinely-differential pairs;
    single-ended-output current-mirror node pairs are NOT in it (they live in the
    4-tuple form's ``mirror_pairs``). Callers that need the mirror pairs (the
    template-driven derivers) use :func:`classify_nets_with_mirrors` directly.
    """
    dynamic, sensitive, diff_pairs, _mirror = classify_nets_with_mirrors(role_bindings)
    return dynamic, sensitive, diff_pairs


# ─────────────────────────────────────────────────────────────────────────────
# Per-cell derivers (route through the generic core)
# ─────────────────────────────────────────────────────────────────────────────


def derive_comparator_intent(
    *,
    match_tol: float = _DEFAULT_MATCH_TOL,
    kickback_tol: float = _DEFAULT_KICKBACK_TOL,
    clkregen_tol: float = _DEFAULT_CLKREGEN_TOL,
) -> LayoutIntent:
    """Derive the dynamic-comparator layout intent from its functional structure.

    Net classes + differential pairs come from the comparator's functional-role
    layer (``comparator_metrics``): the large-swing nets (clk, regen pair,
    outputs) are dynamic aggressors; the input + pre-amp nets are offset-critical
    victims. The regen pair (crossn/crossp) is BOTH dynamic and sensitive -- it
    swings hard yet its balance sets the decision (designer note). Budgets: each
    diff pair stays matched; the inputs' kickback balances; clk's coupling to the
    regen pair balances. Shields: clk away from the regen pair (most damaging
    coupling).

    Routed through :func:`derive_intent_from_roles` (the generic core) so its
    output stays byte-identical to the hand-rolled v1 while the assembly logic is
    shared with every other cell.
    """
    from clay.layout.comparator_metrics import (
        DYNAMIC_AGGRESSORS,
        SENSITIVE_VICTIMS,
        DIFF_PAIRS,
    )

    # crossn/crossp are sensitive AND dynamic.
    sensitive = tuple(SENSITIVE_VICTIMS) + ("crossn", "crossp")
    # Ship the comparator-specific drivers ON the budgets so the measure layer
    # is parameterised by intent, not by hardcoded module constants. The clock
    # of a dynamic comparator is ``clk``; its dynamic aggressors are the full
    # large-swing set (the strobe + the regen pair + the rail-to-rail outputs).
    return derive_intent_from_roles(
        "comparator",
        dynamic_nets=tuple(DYNAMIC_AGGRESSORS),
        sensitive_nets=sensitive,
        diff_pairs=tuple(tuple(p) for p in DIFF_PAIRS),
        kickback_pair=("net0", "net1"),
        kickback_aggressors=tuple(DYNAMIC_AGGRESSORS),
        clk_regen_pair=("crossn", "crossp"),
        clock_net="clk",
        shields=(Shield("clk", "crossn"), Shield("clk", "crossp")),
        match_tol=match_tol,
        kickback_tol=kickback_tol,
        clkregen_tol=clkregen_tol,
        profile_id="dynamic_comparator_v1",
        source="auto",
    )


def _has_clock_aggressor(role_bindings: Sequence[Dict[str, object]]) -> bool:
    """True iff any role binding is a CLOCK/strobe net (a clock-family role).

    Kickback (clock charge-injection onto the sensitive inputs during a switching
    event) and clk_regen (clock coupling into the regenerative pair) are CLOCKED
    phenomena: they only exist when the cell has a clock/dynamic aggressor that
    switches. A continuous-time amplifier (a 5T-OTA, a diff-OTA, a folded cascode)
    has NO clock, so neither phenomenon applies -- its only parasitic-matching
    requirement is ``matched_cc_balance`` on the genuinely differential pairs.

    This mirrors the classifier's R2 rule (:data:`_CLOCK_ROLES` -> dynamic) but
    answers a different question: not "which nets are dynamic" but "is there a
    clock-class aggressor at all". The two cannot be conflated -- a regenerative
    latch node or an output inverter is a dynamic aggressor too, yet it is NOT a
    clock and does not by itself license a kickback budget.
    """
    return any(str(b.get("kind", "")) in _CLOCK_ROLES for b in role_bindings)


def _intent_from_template(
    cell: str,
    template,
    *,
    kickback_pair: Optional[Sequence[str]] = None,
    match_tol: float = _DEFAULT_MATCH_TOL,
    kickback_tol: float = _DEFAULT_KICKBACK_TOL,
) -> LayoutIntent:
    """Build a :class:`LayoutIntent` from a REAL cell template's annotations.

    This is the template-driven path that lets layout-intent derivation work on
    any cell whose factory follows the device-naming convention, instead of a
    hand-written role table per cell. It reads the template's matched-pair +
    device-name structure (:func:`clay.layout._template_roles.template_role_bindings`),
    runs the SAME :func:`classify_nets_by_role` policy every other deriver uses,
    then assembles through the generic core.

    ``kickback_pair`` is the input pair a caller WANTS protected against kickback;
    but kickback is a CLOCKED phenomenon, so this bridge emits a kickback budget
    only when the template actually carries a clock-class aggressor
    (:func:`_has_clock_aggressor`). A continuous-time amplifier template (no clock
    role) suppresses the kickback budget here even when ``kickback_pair`` is
    supplied -- its parasitic requirement is just the ``matched_cc_balance`` on
    its differential pairs. This is the clock-gating SEAM: the gate lives at the
    template-driven deriver, leaving the generic core's unconditional
    ``kickback_pair``/``clk_regen_pair`` contract (relied on by single-ended /
    synthetic callers) untouched.
    """
    from clay.layout._template_roles import template_role_bindings

    role_bindings = template_role_bindings(template)
    dynamic, sensitive, diff_pairs, mirror_pairs = classify_nets_with_mirrors(
        role_bindings
    )
    # Clock-gate the kickback budget: only a clocked cell suffers clock kickback.
    if kickback_pair is not None and not _has_clock_aggressor(role_bindings):
        kickback_pair = None
    return derive_intent_from_roles(
        cell,
        role_to_nets={
            b["name"]: (b.get("gate"), b.get("output"))
            for b in role_bindings
            if "name" in b
        },
        dynamic_nets=dynamic,
        sensitive_nets=sensitive,
        diff_pairs=diff_pairs,
        mirror_pairs=mirror_pairs,
        kickback_pair=kickback_pair,
        match_tol=match_tol,
        kickback_tol=kickback_tol,
    )


def derive_five_t_ota_intent(
    *,
    match_tol: float = _DEFAULT_MATCH_TOL,
    kickback_tol: float = _DEFAULT_KICKBACK_TOL,
) -> LayoutIntent:
    """Derive a 5T-OTA layout intent FROM ITS REAL CELL TEMPLATE.

    Unlike the hand-mapped derivers, this reads the actual
    ``build_5t_ota`` template's annotations: the ``M_in_p`` / ``M_in_n``
    ``Matching`` pair becomes the differential ``input_transconductor`` (gates
    ``inp`` / ``inn`` -> offset-critical victims), and the ``M_load_p`` /
    ``M_load_n`` pair becomes the single-ended-output current-mirror load whose
    drains are ``outp`` / ``outn``. ``M_tail`` is the on-axis
    ``tail_current_source``. Nets are synthesised from the factory's
    device-naming convention (the only net-level fact the template encodes);
    nothing is hardcoded per net the way the comparator constants are.

    DIFFERENTIAL vs MIRROR: the input gate pair ``(inp, inn)`` is the only
    genuinely DIFFERENTIAL pair, so it is the only ``diff_pairs`` entry and gets
    a ``matched_cc_balance`` budget. The output node pair ``(outp, outn)`` is the
    current-mirror LOAD -- ``outp`` is the diode/reference leg, ``outn`` the
    single-ended output, asymmetric BY DESIGN -- so it is recorded on
    ``mirror_pairs`` and carries NO cap budget (its matching is a device
    placement/sizing concern, handled by ``clay.layout.placement_symmetry``).

    Budgets: ``matched_cc_balance(inp, inn)`` ONLY. The 5T-OTA is a
    continuous-time amplifier with NO clock, so neither ``kickback`` (clock
    charge-injection onto the inputs during a switching event) nor ``clk_regen``
    (clock coupling into a regenerative pair) applies -- both are CLOCKED-
    comparator phenomena. ``_intent_from_template`` clock-gates the kickback
    budget on the presence of a clock-class aggressor, of which the OTA template
    has none.
    """
    from clay.layout.cell_templates import build_5t_ota

    template = build_5t_ota("five_t_ota", input_w_um=4.0, input_l_um=0.5)
    return _intent_from_template(
        "five_t_ota",
        template,
        match_tol=match_tol,
        kickback_tol=kickback_tol,
    )


def derive_diff_ota_intent(
    *,
    match_tol: float = _DEFAULT_MATCH_TOL,
    kickback_tol: float = _DEFAULT_KICKBACK_TOL,
) -> LayoutIntent:
    """Derive a differential-OTA (5T-OTA family) layout intent from its roles.

    Now reconciled onto the template-driven path: the ``diff_ota`` family IS the
    5T-OTA, so this builds from the real ``build_5t_ota`` template
    (:func:`derive_five_t_ota_intent`) and re-labels the cell ``"diff_ota"`` to
    keep the long-standing public name and its exact output. The classifier
    discovers the differential input gate pair ``(inp, inn)`` from the template's
    matched-pair structure; the current-mirror load drains ``(outp, outn)`` go to
    ``mirror_pairs`` (asymmetric nodes by design), NOT ``diff_pairs``.

    Net classes: the input gates are offset-critical victims. Budgets:
    ``matched_cc_balance(inp, inn)`` ONLY -- the mirror load carries no cap budget.
    The diff-OTA is continuous-time (no clock), so neither ``kickback`` nor
    ``clk_regen`` applies (both are CLOCKED-comparator phenomena); the kickback
    budget is clock-gated away in ``_intent_from_template``. The tail and the
    mirror reference are on-axis common nets (neither dynamic nor sensitive).
    """
    from clay.layout.cell_templates import build_5t_ota

    template = build_5t_ota("diff_ota", input_w_um=4.0, input_l_um=0.5)
    return _intent_from_template(
        "diff_ota",
        template,
        match_tol=match_tol,
        kickback_tol=kickback_tol,
    )


def derive_folded_cascode_intent(
    *,
    match_tol: float = _DEFAULT_MATCH_TOL,
    kickback_tol: float = _DEFAULT_KICKBACK_TOL,
) -> LayoutIntent:
    """Derive a folded-cascode layout intent FROM ITS REAL CELL TEMPLATE.

    The ``build_folded_cascode`` template carries three ``Matching`` pairs: the
    ``M_in_*`` differential input pair (gates ``inp`` / ``inn`` -> sensitive;
    drains ``outp`` / ``outn``), and the NMOS + PMOS cascode pairs
    (``M_n_cas_*`` / ``M_p_cas_*``) whose drains share the internal differential
    cascode nodes ``casp`` / ``casn``. All three are genuinely DIFFERENTIAL pairs
    (the input pair and the two differential cascode branches), so each becomes a
    ``matched_cc_balance`` budget. There is no single-ended-output current-mirror
    matched pair here, so ``mirror_pairs`` is empty -- the unmatched bias mirrors
    (``M_n_bias`` / ``M_p_bias``) and the tail are on-axis common legs that,
    having no matched partner, contribute no synthesised nets (the bridge's
    honesty boundary).

    Budgets: three ``matched_cc_balance`` (one per differential pair) ONLY. The
    folded cascode is a continuous-time amplifier with NO clock, so it carries
    neither ``kickback`` nor ``clk_regen`` (both are CLOCKED-comparator
    phenomena); the kickback budget is clock-gated away in
    ``_intent_from_template``.
    """
    from clay.layout.cell_templates import build_folded_cascode

    template = build_folded_cascode("folded_cascode", input_w_um=4.0, input_l_um=0.5)
    return _intent_from_template(
        "folded_cascode",
        template,
        match_tol=match_tol,
        kickback_tol=kickback_tol,
    )


def derive_current_mirror_intent(
    *,
    match_tol: float = _DEFAULT_MATCH_TOL,
    kickback_tol: float = _DEFAULT_KICKBACK_TOL,
) -> LayoutIntent:
    """Derive a current-mirror layout intent FROM ITS REAL CELL TEMPLATE.

    The ``build_current_mirror`` template is a single ``M_ref`` / ``M_out``
    ``Matching`` pair -- a single-ended-output current mirror, NOT a differential
    block. The bridge maps its two legs to the node pair ``outp`` / ``outn`` (the
    diode/reference leg + the mirrored output), tagged ``match_kind="mirror"``.
    Because the two NODES are asymmetric by design, this pair goes to
    ``mirror_pairs`` and carries NO ``matched_cc_balance`` cap budget -- its
    matching is a DEVICE placement/sizing concern (``clay.layout.placement_symmetry``),
    not a node-cap balance. There is no externally-driven input gate, so NO net is
    marked sensitive and there is no kickback budget either. A standalone mirror
    therefore declares ZERO parasitic node budgets (only ``mirror_pairs``); this
    exercises the no-diff-pair / no-sensitive-input path of the generic core.
    """
    from clay.layout.cell_templates import build_current_mirror

    template = build_current_mirror("current_mirror", w_um=4.0, l_um=1.0)
    return _intent_from_template(
        "current_mirror",
        template,
        kickback_pair=None,
        match_tol=match_tol,
        kickback_tol=kickback_tol,
    )


_DERIVERS = {
    "comparator": derive_comparator_intent,
    "diff_ota": derive_diff_ota_intent,
    "five_t_ota": derive_five_t_ota_intent,
    "folded_cascode": derive_folded_cascode_intent,
    "current_mirror": derive_current_mirror_intent,
}


def derive_layout_intent(
    cell: str = "comparator",
    *,
    declared: Optional[LayoutIntent] = None,
    **kw,
) -> LayoutIntent:
    """Dispatch to a per-cell derivation.

    Registered cells: ``comparator`` (hand-mapped from its functional-role
    layer), ``diff_ota`` (template-driven, the 5T-OTA family), ``five_t_ota`` /
    ``folded_cascode`` / ``current_mirror`` (template-driven, read straight off
    each ``build_*`` cell template's annotations).
    """
    try:
        auto = _DERIVERS[cell](**kw)
    except KeyError:
        if declared is not None and declared.cell == cell:
            return declared
        raise ValueError(f"no layout-intent derivation for cell={cell!r} yet")
    return merge_layout_intent(auto, declared)


# ─────────────────────────────────────────────────────────────────────────────
# Flow-engine arbiter hook (the frozen contract surface)
# ─────────────────────────────────────────────────────────────────────────────


def evaluate_intent_for_flow(
    cell_def, pex_report, placement_geometry=None, routed_geometry=None
) -> Dict[str, object]:
    """Arbiter hook the flow engine's IntentStep calls.

    ``placement_geometry`` is an OPTIONAL
    :class:`~clay.layout.geometric_intent_checks.GeometricPlacement` built from the
    real placed layout. When supplied, the DIRECT geometric obligations
    (place_group_realized / common_centroid / interdig) are MEASURED on the real
    geometry instead of surfacing as ``unchecked_intent`` from the completeness
    gate. When omitted (callers that have no placement, e.g. cap-only paths), the
    cap/LDE obligations are unchanged and a declared structural intent stays
    blocking-but-unmeasured -- never silently passed. Adding this trailing keyword
    keeps the original ``(cell_def, pex_report)`` positional contract intact.
    ``routed_geometry`` is the analogous OPTIONAL exact-artifact-bound view used
    by ``@wire_width`` and ``@match_length``. When absent, those declarations
    remain unmeasurable and block signing; it is never inferred from placement.

    Composes the parse-driven layout intent + the existing certify pipeline +
    the quality score into a single verdict dict the engine can act on:

      * the intent is taken from the parsed cell via
        :func:`layout_intent_from_cell_def` (the ``.clay`` block is the source of
        truth; the hardcoded ``_DERIVERS`` are a fallback only);
      * :func:`clay.layout.intent_certify.certify_layout_intent` runs the
        obligation check + the SAME proof-token gate DRC/LVS use, signing or
        WITHHOLDING a token;
      * a withheld token (``signed=False``) is a BLOCKING failure: ``verdict``
        is ``"fail"`` and ``blocking`` is non-empty (the failing/unmeasurable
        obligation kinds). ``SignStep`` treats this exactly like a DRC/LVS fail.

    ``pex_report`` is a :class:`~clay.layout.pex.base.PexReport`; ``cell_def`` is
    a parsed :class:`~clay.ir.CellDef`. Returns::

        {
          "signed": bool,                 # proof token signed (not withheld)
          "obligations": [ <serialized obligation>, ... ],
          "blocking": [ <kind>, ... ],    # failing/unmeasurable obligation kinds
          "verdict": "pass" | "fail",
          "score": { ...quality score block... },
          "cell": <cell name>,
          "intent_digest": <16-hex>,
          "intent_source": <"declared" | "auto" | ...>,
        }

    A cell that declared NO parasitic budgets certifies vacuously (no required
    row fails) and signs — but ``score["empty"]`` is True so a no-intent cell is
    never read as a verified one. The token is NEVER fabricated: a withheld token
    always produces ``verdict="fail"`` with a non-empty ``blocking`` list.
    """
    from .intent_certify import certify_layout_intent
    from .intent_score import layout_intent_score

    intent = layout_intent_from_cell_def(cell_def)
    certified = certify_layout_intent(
        intent,
        pex_report,
        intent=intent,
        placement_geometry=placement_geometry,
        routed_geometry=routed_geometry,
    )
    obligations = certified["obligations"]
    score = layout_intent_score(obligations)  # type: ignore[arg-type]

    signed = certified["token"] is not None
    blocking = list(certified["blocking"])  # failing / unmeasurable obligation kinds

    # Token-honesty invariant: a withheld token MUST fail with a non-empty
    # blocking list. If the gate withheld a token but reported no blocking kind
    # (should not happen), surface it explicitly rather than silently passing.
    if not signed and not blocking:
        blocking = ["intent_signed"]
    verdict = "pass" if signed else "fail"

    return {
        "signed": bool(signed),
        "obligations": [
            {
                "kind": str(o.kind),
                "pair": [str(o.pair[0]), str(o.pair[1])],
                "measured_imbalance": (
                    None
                    if o.measured_imbalance is None
                    else float(o.measured_imbalance)
                ),
                "rel_tol": float(o.rel_tol),
                "status": str(o.status),
                "verification_status": str(o.status),
                "over_budget": float(o.over_budget),
                "missing_nets": [str(n) for n in o.missing_nets],
                "declaration_ids": [
                    str(item) for item in (o.declaration_ids or ())
                ],
                "measurements": {
                    str(name): (
                        None if value is None else float(value)
                    )
                    for name, value in dict(o.measurements or {}).items()
                },
            }
            for o in obligations
        ],
        "blocking": [str(k) for k in blocking],
        "verdict": verdict,
        "score": score,
        "cell": str(certified["cell"]),
        "intent_digest": str(certified["intent_digest"]),
        "intent_source": str(certified["intent_source"]),
    }
