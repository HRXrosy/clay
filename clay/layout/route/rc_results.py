"""RouteCellResult, BringupDiagnosticRouteResult, and pre-route capability gate."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from clay.core_contract import VerificationStatus
from clay.layout.route import constraints as _C
from clay.layout.route.engine import NetRoute, Obstacle
from clay.layout.route.rc_connectivity import ShortsCheck, check_net_connectivity
from clay.layout.route.rc_devices import RouteExecutionScope, _XY
from clay.layout.route.rc_intent_constraints import _diagnostic_constraint_ref
from clay.layout.route.rc_placement_evidence import FinalPlacementResourceEvidence
from clay.layout.tech import LayoutTech


@dataclass
class RouteCellResult:
    """The general route's honest output bundle.

    ``routed`` is the engine :class:`~clay.layout.route.engine.RoutedCell` (its own
    ``success`` / ``shorts`` / ``unsatisfied`` / per-net ``connected``). ``shorts``
    here is the INDEPENDENT union-find oracle verdict (cross-net merged pairs) over
    the routed metal + device pads -- the authoritative no-short check. The pins /
    obstacles / geometry / placement are exposed so the caller can emit GDS + run
    the obligations / LVS gate without re-deriving them."""

    routed: Any
    pins_by_net: Dict[str, List[Any]]
    obstacles: List[Obstacle]
    geometry: Dict[str, Any]
    placement_um: Dict[str, _XY]
    constraints: List[Any]
    tech: LayoutTech
    shorts: Any                          # ShortsCheck (union-find merged_pairs)
    uncovered_terminals: List[Any] = field(default_factory=list)
    axis_x_um: float = 0.0
    # the bridged MOS device views the route consumed (.name/.kind/.W/.L/.M +
    # ROLE-keyed .nets) -- the canonical netlist binding, exposed so downstream
    # obligation checkers (bulk_tie_report) + the emit-side well merge read the
    # SAME device source the router did (no second netlist derivation).
    devices: Tuple[Any, ...] = ()
    # Same canonical source-compiled passive views.  The exact emitter replays
    # MOS/passive member classes from these two inventories; it never accepts a
    # caller-authored class tuple as evidence authority.
    passives: Tuple[Any, ...] = ()
    # the DECLARED `@guard_ring` cluster intents (GuardRingIntent tuples) the route
    # materialized -- exposed so the independent guard_ring_report checker can verify
    # the DRAWN rings against them. Empty when the cell declares no ring.
    guard_rings: Tuple[Any, ...] = ()
    shared_well_guard_envelope_plans: Tuple[Any, ...] = ()
    shared_well_guard_ring_inventory: Any = None
    # Canonical netlist class of every ring-addressable member.  The exact-GDS
    # checker combines this independently compiled source fact with its own
    # complete-artwork claim before adjudicating role=tap|isolation.
    guard_ring_member_classes: Tuple[Tuple[str, str], ...] = ()
    # Exact structural expansion used when a declared edge rail reserves space
    # inside a tight ring's protected inner region.  The final-placement
    # certificate binds the resulting materialized block geometry; the exact-GDS
    # checker replays these rows only together with that certificate.
    member_ring_pad_top_nm: Tuple[Tuple[str, int], ...] = ()
    member_ring_pad_bottom_nm: Tuple[Tuple[str, int], ...] = ()
    # A3 POST-gate (capability handshake, AFTER route): the HARD constraints the
    # engine reported it could NOT satisfy on THIS route (``routed.unsatisfied``
    # filtered to HARD -- a blocked exact Mirror, an unmaterialized Width, a
    # non-converged Spacing). SURFACED here so signoff never rests on
    # ``net_complete`` alone (which is same-layer/deck-unaware). These are of a
    # SUPPORTED kind (the PRE-gate already stopped unsupported KINDS before route),
    # so they are instance-level fidelity residuals, NOT capability gaps; the engine
    # already reflects them in ``routed.success``. Populated always; empty for a
    # route that satisfied every HARD constraint.
    required_unsatisfied: List[Any] = field(default_factory=list)
    # The materializer-derived final rectangular floorplan and the independent
    # post-legalizer verdict over the exact route polygons handed to emit.
    routing_envelope_nm: Tuple[int, int, int, int] = ()
    routing_envelope_certificate: Any = None
    route_containment_report: Any = None
    # Always-on, report-only observation of the exact FINAL placement consumed by
    # this route.  It is compiled before power-rail artwork exists and is never
    # forwarded into engine kwargs or candidate ordering.
    resource_evidence: Any = None
    # Deterministic net-class topology plan compiled from the FINAL pins,
    # materialized rings, rails, explicit declarations, and PDK facts. It is a
    # load-bearing candidate-compiler input for plan-consuming engines and remains
    # report-only evidence for non-consuming engines.
    topology_plan: Any = None
    # Exact final routing world used to validate a compiler-selected CandidatePool:
    # final pins/rings/rails, capability-projected obstacles, placement/topology
    # evidence, and PDK routing facts. Validation-only; never materializes geometry.
    routing_world_provenance: Any = None
    candidate_pool: Any = None
    resource_plan: Any = None
    rail_candidate_pool: Any = None
    rail_placement_plan: Any = None
    materialized_rail_certificate: Any = None
    # Canonical declaration/envelope facts for upper-metal power planes.  The
    # exact-GDS checker recompiles these from source + final placement; this field
    # is a carried materialization record, never its own authority.
    power_plane_facts: Tuple[Any, ...] = ()
    # The exact row-calculus inputs consumed by placement/routing.  Post-emission
    # structure checkers use these typed certificates as obligations while rereading
    # physical geometry from the emitted GDS; they never reuse pre-route metrics.
    row_program: Any = None
    row_materialization: Any = None
    row_final_placement_certificate: Any = None
    row_final_placement: Any = None
    final_placement_certificate: Any = None
    exact_structure_expectation: Any = None
    # Aggregate passive/BJT authority compiled before placement and replayed over
    # the exact geometry/placement mapping handed to the GDS emitter.
    physical_macro_scope_token: Any = None
    rigid_nonmos_macro_plans: Any = None
    rigid_nonmos_macro_inputs: Any = None
    rigid_nonmos_macro_interfaces: Any = None
    rigid_nonmos_placement_certificate: Any = None
    rigid_nonmos_emission_certificate: Any = None
    # Immutable RG4 proof that the final simultaneous plan has zero capacity/Hall
    # deficits in the same placement scope, including every applied Benders round.
    placement_resource_closure: Any = None
    # Actual engine consumed by this route, retained for source-bound replay.
    selected_route_engine: Any = field(default=None, repr=False, compare=False)

    @property
    def merged_pairs(self) -> List[Tuple[str, str]]:
        return list(getattr(self.shorts, "merged_pairs", []) or [])

    @property
    def n_connected(self) -> int:
        return int(getattr(self.routed, "n_connected", 0))

    @property
    def n_required(self) -> int:
        return len(self.pins_by_net)

    @property
    def bulk_tie_report(self) -> Any:
        """BULK-TIE FIDELITY (native, ADVISORY): per s!=b MOS device, is the
        declared body net REALLY tied (routed metal) to the drawn tap M1 island?
        See :mod:`clay.layout.route.bulk_tie_check` -- lazily computed + cached
        (the ``geom_unconnected_nets`` pattern). On a 3-terminal deck (gf180) the
        foundry LVS cannot measure bulk at all, so this is the ONLY bulk-tie
        oracle there; on a 4-terminal deck (external_pdk/external_pdk) the independent KLayout
        LVS remains the signoff authority and this is the native early-warning."""
        cached = self.__dict__.get("_bulk_tie_report")
        if cached is None:
            from clay.layout.route.bulk_tie_check import check_bulk_tie_fidelity

            cached = check_bulk_tie_fidelity(
                self.routed, self.tech, geometry=self.geometry,
                placement_um=self.placement_um, pins_by_net=self.pins_by_net,
                devices=self.devices,
            )
            self.__dict__["_bulk_tie_report"] = cached
        return cached

    @property
    def guard_ring_report(self) -> Any:
        """GUARD-RING MATERIALIZATION (native, ADVISORY): is each DECLARED
        ``@guard_ring`` cluster really drawn as a CLOSED tap ring, tied to the
        declared net, on the declared well polarity, in the EMITTED artifact? See
        :mod:`clay.layout.route.guard_ring_check` -- lazily computed + cached (the
        ``bulk_tie_report`` pattern). Reads the emitted geometry + routed metal +
        pins, NOT the builder record. On a 3-terminal deck (gf180) the foundry LVS
        cannot see bulk, so this is the ONLY bulk-ring oracle there; on a 4-terminal
        deck the independent KLayout LVS stays the signoff authority and this is the
        native early-warning. An empty report (no declared ring) is trivially clean."""
        cached = self.__dict__.get("_guard_ring_report")
        if cached is None:
            from clay.layout.route.guard_ring_check import (
                GuardRingReport,
                check_guard_ring_materialized,
            )

            if not self.guard_rings:
                cached = GuardRingReport(clean=True)
            else:
                cached = check_guard_ring_materialized(
                    self.routed, self.tech, geometry=self.geometry,
                    placement_um=self.placement_um, pins_by_net=self.pins_by_net,
                    guard_ring_intents=self.guard_rings,
                )
            self.__dict__["_guard_ring_report"] = cached
        return cached

    @property
    def geom_unconnected_nets(self) -> List[str]:
        """Nets whose pins do NOT all land in ONE via-joined component of the EMITTED
        metal -- the GEOMETRIC opens (e.g. an emit-time via-envelope SUPPRESSION, the
        source/bulk via-landing deadlock) that the engine's routing-graph ``connected``
        flag MISSES (the flag is set from the grid route, before the emit clip). The
        authoritative native connectivity verdict; agrees with the KLayout LVS oracle."""
        cached = self.__dict__.get("_geom_unconn")
        if cached is None:
            conn = check_net_connectivity(
                self.routed, self.tech, geometry=self.geometry,
                placement=self.placement_um, pins_by_net=self.pins_by_net)
            cached = sorted(n for n, ok in conn.items() if not ok)
            self.__dict__["_geom_unconn"] = cached
        return cached

    @property
    def unconnected_nets(self) -> List[str]:
        """The GEOMETRIC opens (emitted-metal union-find) -- the honest connectivity
        truth, not the engine's pre-emit-clip ``connected`` flag."""
        return list(self.geom_unconnected_nets)

    @property
    def net_complete(self) -> bool:
        """NATIVE (ADVISORY) net-completeness: every required net present + GEOMETRICALLY
        connected (emitted-metal union-find, NOT the engine's pre-emit-clip routing-graph
        flag) + no SAME-LAYER cross-net union-find short. NOT a signoff verdict.

        Reading the emitted geometry makes the CONNECTIVITY half honest: the engine sets
        each net's ``connected`` flag from the grid route, before ``_emit_wires`` may
        SUPPRESS a via whose landing pad cannot clear a foreign net -- so the flag
        over-reports an open (the LDO MM10 source on the no-ref placement). The geometric
        union-find here catches that open.

        But the SHORT half (``not self.merged_pairs``) inherits
        :func:`check_cross_net_shorts`'s same-layer-only / deck-unaware blind spot, so
        ``net_complete`` can be TRUE on a genuine cross-net short that closes through a
        device-cell via stack -- PROVEN on the LDO ``CC1`` MIM cap (``NET034<->GNDA`` via
        ``metal_ncap/via2/metal3``), where ``net_complete is True`` yet the KLayout LVS
        reports the short. So this is the dangerous native-clean / oracle-dirty direction:
        signoff MUST gate on the independent KLayout LVS, never on ``net_complete`` alone."""
        nets = getattr(self.routed, "nets", {}) or {}
        all_present = all(n in nets for n in self.pins_by_net)
        return bool(all_present and not self.geom_unconnected_nets
                    and not self.merged_pairs)

    def emit_gds(self, out_gds: str, *, cell_name: str = "routed_cell") -> str:
        """Write the routed metal to ``out_gds`` (per-net merged; no cross-net union)."""
        from clay.layout.route.emit_gds import emit_routed_gds
        return emit_routed_gds(self.routed, out_gds, cell_name=cell_name)

    def to_routed_cell(self) -> Any:
        """Adapt this result to the generic ``clay.layout.generic_router.RoutedCell``
        shape (``polys`` / ``tagged`` / ``labels`` / ``shorts`` / ``net_complete``) the
        production flow + the fault-intent diagnosis + the DRC-diagnosis join consume.

        The general router (PathFinder) speaks the engine ``RoutedCell`` shape
        (``.nets[*].wires``); the production flow + ``clay.layout.fault_intent`` read
        the generic ``RoutedCell`` surface. This is the SINGLE-canonical-result bridge
        so a route_cell result flows through the SAME downstream consumers as a
        legacy route_generic result -- never a second divergent geometry path.

          * ``polys``  -- the routed BEOL wires (every layer).
          * ``tagged`` -- per-wire ``(net, (layer, datatype), bbox)`` for the
            fault-intent contact scan + DRC-violation net-provenance join.
          * ``labels`` -- one on-metal1 anchor ``net -> (x, y)`` per net.
          * ``shorts`` -- the INDEPENDENT union-find verdict (distinct / merged_pairs).
          * ``net_complete`` -- left ``None`` (route_cell carries its own
            ``net_complete`` property; the generic gate object is not reconstructed).
        """
        from clay.layout.routed_cell_types import RoutedCell as _GRoutedCell
        from clay.layout.routed_cell_types import ShortsCheck as _GShortsCheck

        polys = list(self.routed.wires)
        tagged: List[Tuple[str, Tuple[int, int], Tuple[int, int, int, int]]] = []
        for net, nr in (getattr(self.routed, "nets", {}) or {}).items():
            for lp in getattr(nr, "wires", []) or []:
                xs = [int(v.x) for v in lp.polygon.vertices]
                ys = [int(v.y) for v in lp.polygon.vertices]
                if not xs:
                    continue
                tagged.append((str(net), (int(lp.layer), int(lp.datatype)),
                               (min(xs), min(ys), max(xs), max(ys))))
        sc = self.shorts
        gsc = _GShortsCheck(
            distinct=bool(getattr(sc, "distinct", True)),
            merged_pairs=list(getattr(sc, "merged_pairs", ()) or []),
            net_components={},
            shape_roots=(),
        )
        labels: Dict[str, Tuple[int, int]] = {}
        try:
            from clay.layout.route.emit_gds import net_label_points
            for n, (x, y) in net_label_points(
                self.routed, self.pins_by_net, tech=self.tech,
                geometry=self.geometry, placement=self.placement_um,
            ).items():
                labels[str(n)] = (int(x), int(y))
        except Exception:  # noqa: BLE001 - labels are advisory for the adapter surface
            labels = {}
        return _GRoutedCell(polys=polys, tagged=tagged, labels=labels,
                            shorts=gsc, net_complete=None)


@dataclass
class BringupDiagnosticRouteResult:
    """Non-signoff geometry from an explicitly non-signoff routing attempt.

    The production bring-up path is unplanned deterministic negotiation.  The
    older bounded candidate-frontier fields remain in the schema so archived
    diagnostic receipts stay readable; they are zero/empty when no candidate
    pool was constructed.
    """

    routed: Any
    pins_by_net: Dict[str, List[Any]]
    obstacles: List[Obstacle]
    geometry: Dict[str, Any]
    placement_um: Dict[str, _XY]
    constraints: List[Any]
    tech: LayoutTech
    shorts: Any
    source_planning_stop: Any
    family_count: int
    family_winner_count: int
    failed_family_refs: Tuple[str, ...]
    unrouted_nets: Tuple[str, ...]
    diagnostic_route_program_hash: str = ""
    selected_program_refs: Tuple[Tuple[str, str], ...] = ()
    routed_geometry_digest: str = ""
    diagnostic_mode: str = "CANDIDATE_FRONTIER"
    diagnostic_stops: Tuple[Any, ...] = ()
    uncovered_terminals: List[Any] = field(default_factory=list)
    axis_x_um: float = 0.0
    devices: Tuple[Any, ...] = ()
    passives: Tuple[Any, ...] = ()
    guard_rings: Tuple[Any, ...] = ()
    shared_well_guard_envelope_plans: Tuple[Any, ...] = ()
    shared_well_guard_ring_inventory: Any = None
    guard_ring_member_classes: Tuple[Tuple[str, str], ...] = ()
    member_ring_pad_top_nm: Tuple[Tuple[str, int], ...] = ()
    member_ring_pad_bottom_nm: Tuple[Tuple[str, int], ...] = ()
    required_unsatisfied: List[Any] = field(default_factory=list)
    routing_envelope_nm: Tuple[int, int, int, int] = ()
    routing_envelope_certificate: Any = None
    route_containment_report: Any = None
    suppressed_hard_constraints: Tuple[Any, ...] = ()
    resource_evidence: Any = None
    topology_plan: Any = None
    routing_world_provenance: Any = None
    rail_candidate_pool: Any = None
    rail_placement_plan: Any = None
    materialized_rail_certificate: Any = None
    power_plane_facts: Tuple[Any, ...] = ()
    row_program: Any = None
    row_materialization: Any = None
    row_final_placement_certificate: Any = None
    row_final_placement: Any = None
    final_placement_certificate: Any = None
    exact_structure_expectation: Any = None
    # Preserve the exact aggregate-macro projection consumed by routing so a
    # diagnostic result can still be audited.  Carrying these fields does not
    # confer signoff authority: this result type remains categorically
    # ``diagnostic_only`` and cannot be published as exact evidence.
    physical_macro_scope_token: Any = None
    rigid_nonmos_macro_plans: Any = None
    rigid_nonmos_macro_inputs: Any = None
    rigid_nonmos_macro_interfaces: Any = None
    rigid_nonmos_placement_certificate: Any = None
    rigid_nonmos_emission_certificate: Any = None
    selected_route_engine: Any = field(default=None, repr=False, compare=False)

    @property
    def diagnostic_only(self) -> bool:
        return True

    @property
    def suppressed_hard_constraint_refs(self) -> Tuple[str, ...]:
        """Content refs for HARD intent omitted from the diagnostic subset."""

        return tuple(sorted(
            _diagnostic_constraint_ref(constraint)
            for constraint in self.suppressed_hard_constraints
        ))

    @property
    def diagnostic_stop_refs(self) -> Tuple[str, ...]:
        """Content refs for typed per-net negotiated-congestion stops."""

        return tuple(sorted(
            str(stop.content_ref) for stop in self.diagnostic_stops
        ))

    @property
    def diagnostic_summary_digest(self) -> str:
        payload = {
            "diagnostic_mode": self.diagnostic_mode,
            "diagnostic_stop_refs": list(sorted(self.diagnostic_stop_refs)),
            "failed_family_refs": list(sorted(self.failed_family_refs)),
            "family_count": self.family_count,
            "family_winner_count": self.family_winner_count,
            "route_program_hash": self.diagnostic_route_program_hash,
            "routing_world": self.routing_world_provenance.sha256(),
            "schema": "clay.bringup-diagnostic-route-result.v1",
            "source_stop": (
                None
                if self.source_planning_stop is None
                else self.source_planning_stop.sha256()
            ),
            "selected_program_refs": [
                list(item) for item in sorted(self.selected_program_refs)
            ],
            "routed_geometry_digest": self.routed_geometry_digest,
            "suppressed_hard_constraint_refs": list(
                self.suppressed_hard_constraint_refs
            ),
            "unrouted_nets": list(sorted(self.unrouted_nets)),
        }
        return hashlib.sha256(json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest()

    merged_pairs = RouteCellResult.merged_pairs
    n_connected = RouteCellResult.n_connected
    n_required = RouteCellResult.n_required
    bulk_tie_report = RouteCellResult.bulk_tie_report
    guard_ring_report = RouteCellResult.guard_ring_report
    geom_unconnected_nets = RouteCellResult.geom_unconnected_nets
    unconnected_nets = RouteCellResult.unconnected_nets
    net_complete = RouteCellResult.net_complete
    emit_gds = RouteCellResult.emit_gds
    to_routed_cell = RouteCellResult.to_routed_cell


class NetIncompleteError(RuntimeError):
    """Raised by :func:`route_cell` (``require_complete=True``) when the route is NOT
    net-complete (a missing/disconnected net OR a cross-net union-find short)."""


class RouteUnsupportedError(RuntimeError):
    """Raised by the :func:`route_cell` capability handshake (PRE-gate) when the
    SELECTED :class:`~clay.layout.route.engine.RouteEngine` cannot satisfy a
    HARD/required routing constraint KIND.

    The compilation contract is ``parsed != lowered != backend-satisfied !=
    GDS-satisfied``; without this stop the router silently EMITS a GDS that
    ignores the unmaterializable constraint (a mirror that is never reflected, a
    shield that is never routed). This is the typed 5-state stop: it carries
    ``status == VerificationStatus.UNSUPPORTED`` and the offending constraint
    objects, so a caller/flow renders ONE honest verdict rather than crashing or
    silently emitting. It NEVER fires for a cell whose HARD constraint kinds the
    engine all satisfies (the production PathFinder OTA: every kind supported)."""

    def __init__(self, engine: Any, unsupported: Sequence[Any]) -> None:
        self.status = VerificationStatus.UNSUPPORTED
        self.engine_name = str(getattr(engine, "name", type(engine).__name__))
        self.unsupported = list(unsupported)
        kinds = sorted({str(getattr(c, "kind", "?")) for c in self.unsupported})
        self.unsupported_kinds = kinds
        super().__init__(
            f"route_cell capability handshake: engine {self.engine_name!r} does NOT "
            f"satisfy required constraint kind(s) {kinds} -- stopping before emit "
            f"(5-state UNSUPPORTED), never silently routing a GDS that ignores them. "
            f"Unsupported: {self.unsupported}"
        )


def _preroute_capability_gate(route_engine: Any, cons: Sequence[Any]) -> None:
    """A2 PRE-gate (capability handshake, BEFORE route/emit).

    For each HARD constraint (``constraints.is_hard`` -- the hardness FIELD, never
    a per-kind branch) whose ``kind`` is NOT in ``route_engine.satisfies()``,
    collect it; if any are collected, STOP by raising :class:`RouteUnsupportedError`
    (a typed 5-state UNSUPPORTED verdict listing them all) -- the router must not
    proceed to route/emit a GDS that silently ignores a required constraint. SOFT
    constraints (Direction) are advisory and NEVER gate. An engine that declares no
    ``satisfies()`` (a legacy strategy that opts out of the handshake) is not gated
    -- byte-identical fallback, never a crash. A NO-OP (returns None) whenever every
    HARD kind is satisfied, so the production PathFinder route is byte-identical."""
    satisfies = getattr(route_engine, "satisfies", None)
    if satisfies is None:
        return
    supported = frozenset(satisfies())
    unsupported = [
        c for c in cons
        if _C.is_hard(c) and str(getattr(c, "kind", None)) not in supported
    ]
    if unsupported:
        raise RouteUnsupportedError(route_engine, unsupported)


def _supports_deterministic_negotiation(route_engine: Any) -> bool:
    """Return whether an engine exposes the deterministic feasibility seam."""

    return (
        bool(getattr(route_engine, "supports_deterministic_negotiation", False))
        and callable(getattr(route_engine, "satisfies", None))
        and callable(getattr(route_engine, "route", None))
    )



def _build_bringup_diagnostic_route_result(
    *,
    common_result_fields: Mapping[str, Any],
    routed: Any,
    diagnostic_compile: Any,
    use_bringup_negotiation: bool,
    suppressed_hard_constraints: Tuple[Any, ...],
) -> BringupDiagnosticRouteResult:
    """Assemble BringupDiagnosticRouteResult with canonical stop normalization."""
    family_count = (
        diagnostic_compile.family_count
        if diagnostic_compile is not None
        else 0
    )
    emission_unrouted_nets = tuple(sorted(
        net
        for net, net_route in (getattr(routed, "nets", {}) or {}).items()
        if not bool(getattr(net_route, "connected", False))
    ))
    from clay.layout.route.engine_gridless import (
        NegotiationRouteStop,
        NegotiationStopReason,
    )

    route_stops = list(getattr(routed, "route_stops", ()) or ())
    if any(type(stop) is not NegotiationRouteStop for stop in route_stops):
        raise TypeError(
            "deterministic negotiation returned an untyped route stop"
        )
    has_deck_merge = any(
        stop.reason is NegotiationStopReason.DECK_NET_MERGE
        for stop in route_stops
    )
    if has_deck_merge:
        unrouted_nets = tuple(sorted({
            stop.net
            for stop in route_stops
            if stop.reason is not NegotiationStopReason.DECK_NET_MERGE
        }))
    else:
        unrouted_nets = emission_unrouted_nets
        stopped_nets = {stop.net for stop in route_stops}
        route_stops.extend(
            NegotiationRouteStop(
                NegotiationStopReason.NO_LEGAL_ROUTE,
                net,
                "route_cell_boundary",
            )
            for net in unrouted_nets
            if net not in stopped_nets
        )
    route_stops.sort(key=lambda stop: stop.canonical_json())
    result = BringupDiagnosticRouteResult(
        **common_result_fields,
        source_planning_stop=(
            None if diagnostic_compile is None
            else diagnostic_compile.source_unknown
        ),
        family_count=family_count,
        family_winner_count=(
            0 if diagnostic_compile is None
            else diagnostic_compile.family_winner_count
        ),
        failed_family_refs=(
            () if diagnostic_compile is None
            else diagnostic_compile.failed_family_refs
        ),
        unrouted_nets=(
            unrouted_nets if diagnostic_compile is None
            else diagnostic_compile.unrouted_nets
        ),
        diagnostic_route_program_hash=(
            ""
            if diagnostic_compile is None
            else diagnostic_compile.route_program.sha256()
        ),
        selected_program_refs=tuple(
            (item.net, item.candidate_id)
            for item in (
                ()
                if diagnostic_compile is None
                else diagnostic_compile.route_program.selected_nets
            )
        ),
        routed_geometry_digest=str(
            getattr(routed, "geometry_digest", "")
        ),
        diagnostic_mode=(
            "UNPLANNED_ENGINE_NEGOTIATION"
            if use_bringup_negotiation
            else "CANDIDATE_FRONTIER"
        ),
        diagnostic_stops=tuple(route_stops),
        suppressed_hard_constraints=suppressed_hard_constraints,
    )
    routed.diagnostic_summary_digest = result.diagnostic_summary_digest
    return result
