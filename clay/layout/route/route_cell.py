"""General intent-driven cell router: route ANY placed cell through the engine.
This is the GENERAL counterpart to
:func:`clay.layout.route.anchor_ota.route_ota_leakfree`. It reuses that path's
PROVEN engine wiring -- the negotiated-congestion rip-up/reroute
:class:`~clay.layout.route.engine_pathfinder.PathFinderEngine`, the leakage-free
:func:`~clay.layout.route.pins.extract_pins`, the unified
:func:`~clay.layout.model.build_layout_model` net-aware obstacles, and
:func:`~clay.layout.route.lower.lower_intent` + per-diff-pair
:class:`~clay.layout.route.constraints.Mirror` constraints -- but DROPS every
OTA-specific bit (MIM/cap-plate keep-outs, IO edge-stub anchors, rail corridor
preroutes, the reference-GDS comparison). It carries NO per-circuit logic and NO
circuit-name branch: the only inputs are a placed cell, its parsed ``CellDef``,
its declarative routing/layout intent, and the PDK.
The one piece of NEW machinery here is the ``place_generic <-> route/`` data-model
BRIDGE (:func:`_bridge_placement_to_route_inputs`): the generic placer
(:func:`clay.layout.generic_placement.place_generic`) returns a
:class:`~clay.layout.generic_placement.Placement` whose ``devices`` are
:class:`~clay.layout.generic_placement.PlacedDevice` keyed by label, in NM, with
NO net binding -- while ``extract_pins`` / ``build_layout_model`` consume an
``ExtractedIntent``-shaped object (``.devices`` with ``.name/.kind/.W/.L/.M`` +
a ``.nets`` role->net map) plus a ``placement`` dict ``{name -> (x,y) um}`` and a
``geometry`` dict of leakage-free :class:`~clay.layout.tech` ``CellGeometry``
stamps. The netlist binding + per-device W/L/M live UPSTREAM in the parsed
``CellDef.devices``; the bridge rekeys those into the shape the route subsystem
expects. No reference geometry, no foundry GDS.

The in-memory union-find short check is an early advisory only. Connectivity
signoff belongs to the exact emitted GDS judged by the real KLayout full-extraction
LVS path; native ``merged_pairs`` / ``net_complete`` and DRC cannot certify it.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field, fields, is_dataclass, replace
from enum import Enum
from fractions import Fraction
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from clay.core_contract import VerificationStatus
from clay.layout.extract.realize_render import (
    build_device_geometry,
    resolve_device_fold_target,
)
from clay.layout.model import build_layout_model
from clay.layout.nonmos_placement_inputs import replay_native_nonmos_placement
from clay.layout.placed_geometry import placed_point_nm
from clay.layout.route import constraints as _C
from clay.layout.route.engine import (
    NetRoute,
    Obstacle,
    RouteResourcePlanIncompleteError,
    require_resource_plan_support,
)
from clay.layout.route.engine_pathfinder import PathFinderEngine
from clay.layout.route.lower import conventional_axes_for, lower_intent
from clay.layout.route.pins import extract_pins
from clay.layout.route.via_pad_legalize import legalize_via_pads
from clay.layout.tech import LayoutTech

from clay.layout.route.rc_devices import (
    _NM_PER_UM,
    _XY,
    _HVH_DIRECTION_WEIGHT,
    _PIN_KEY_TO_ROLE,
    _ROLE_TO_TERM,
    _declared_ground_ports,
    RouteExecutionScope,
    _DiagnosticRouteCompilerProduct,
    _MOS_CLASSES,
    _MOS_THICK,
    _mos_kind_oxide,
    _BridgedDevice,
    _BridgedPassive,
    _BridgedBjt,
    _PlacedStamp,
    _PlacedBjtStamp,
    _PlacedPassiveStamp,
    _RecordDevice,
    _RecordPassive,
    _BridgedIntent,
    _bridge_devices_from_records,
    bridge_devices_from_cell_def,
    bridge_bjts_from_cell_def,
    _cell_bbox_um,
    _BringupRailConstructionEvidence,
    RoutePlanningStop,
    _metal_index_of,
)
from clay.layout.route.rc_bridge import (
    _ArrayConsumption,
    _resolve_rail_flank_dummies,
    _resolve_cc_flank_dummies,
    resolve_shared_diffusion_arrays,
    resolve_common_centroids,
    _ArrayExtentTranslation,
    _array_extent_translation,
    array_member_offsets_um,
    _array_cell_geometry,
    _stored_row_consumptions,
    _snap_matching_stamp_positions_to_manufacturing_grid,
    _normalize_materialized_geometry_to_manufacturing_grid,
    _manufacturing_grid_bbox_access,
    _bridge_placement_to_route_inputs,
    _bridge_placement_to_route_inputs_impl,
    _AdjacentNativeRowPins,
    _adjacent_native_row_snapshot,
    _adjacent_pin_facts,
    _extract_native_row_pin_report,
    _relocate_array_member_pins,
)
from clay.layout.route.rc_connectivity import (
    ShortsCheck,
    _via_keys,
    _metal_keys,
    _seeded_geometry_metal_regions,
    check_cross_net_shorts,
    check_net_connectivity,
)
from clay.layout.route.rc_placement_evidence import (
    FinalPlacementResourceEvidence,
    FinalPlacementBindingError,
    _certificate_routed_master_blocks,
    _final_placement_provenance_digest,
    _resource_cluster_boxes,
    _shared_guard_virtual_resource_boxes,
    _rectilinear_union_boxes,
    build_final_placement_resource_evidence,
    _build_measurable_final_placement_resource_evidence,
)
from clay.layout.route.rc_results import (
    RouteCellResult,
    BringupDiagnosticRouteResult,
    NetIncompleteError,
    RouteUnsupportedError,
    _preroute_capability_gate,
    _supports_deterministic_negotiation,
    _build_bringup_diagnostic_route_result,
)
from clay.layout.route.rc_rails_guards import (
    _validate_guard_ring_member_padding,
    _effective_guard_ring_intents,
    _append_guard_ring_tap_pins,
    _guard_ring_taps_from_stamps,
    _materialize_guard_rings,
    _routing_envelope_from_geometry,
    _routing_envelope_from_final_placement,
    _materialize_declared_anchor_edge_pins,
    _legacy_materialize_power_rails,
    _materialize_power_rails,
    _require_no_new_wide_rail_spacing,
    _PhysicalRailWriteFact,
    _write_physical_power_rails,
    _bringup_rail_construction_stop,
    _build_bringup_outside_edge_rail_facts,
    _materialize_bringup_power_rails,
)
from clay.layout.route.rc_planning import (
    _CompiledEdgeRailSpec,
    _compile_edge_rail_specs_from_intent,
    _canonical_rail_ast_value,
    _group_route_contract_for_engine,
    _canonical_rail_declaration_payload,
    _canonical_rail_source_ast,
    _parsed_edge_rail_receipts,
    _compile_edge_rail_demands_from_intent,
    _build_pre_rail_scene,
    _plan_rails,
    _compile_route_resource_plan,
    _verify_route_resource_plan_or_stop,
)
from clay.layout.route.rc_intent_constraints import (
    _diagnostic_canonical_value,
    _diagnostic_constraint_json,
    _diagnostic_constraint_ref,
    DiagnosticSuppressedHardConstraint,
    _diagnostic_materialization_inputs,
    _diagnostic_negotiation_constraints,
    _canonical_polygon_vertices,
    _routed_geometry_digest,
    _restore_diagnostic_open_nets,
    _unsatisfied_required_route_goals,
    _exact_mirror_axis_um,
    _require_exact_mirror_axis_um,
    _opportunistic_match_mirror,
    _layout_match_length_constraints,
    _layout_route_match_balance_constraints,
    InvalidTrunkIntentError,
    ConflictingTrunkIntentError,
    _validate_trunk_constraints_for_tech,
    _trunk_layer_preferences,
    _routing_intent_for,
)
from clay.layout.route.rc_native_profile_gate import (
    _check_source_engine_before_group_fallback,
    _maybe_freeze_native_output_profile_engine,
    _verify_native_output_profile_fixed_stacks,
)


def _resolve_route_engine(engine: Optional[Any], trunk_layers: bool):
    """The ONE engine-default seam: an explicit ``engine`` instance passes
    through untouched (a declared @route_policy strategy via the registry, or
    a caller-built instance); ``None`` constructs the byte-identical production
    default ``PathFinderEngine(widen=True, trunk_layers=...)``."""
    if engine is not None:
        return engine
    return PathFinderEngine(widen=True, trunk_layers=bool(trunk_layers))


def route_prepared_cell(
    pins_by_net: Mapping[str, Sequence[Any]],
    obstacles: Sequence[Any],
    constraints: Sequence[Any],
    tech: Any,
    *,
    trunk_layers: bool = False,
    over_active_penalty: float = 0.0,
    over_active_min_level: int = 3,
    over_active_rects_um: Optional[
        Sequence[Tuple[float, float, float, float]]
    ] = None,
    preroutes: Optional[Mapping[str, Sequence[Any]]] = None,
) -> Any:
    """Route already-lowered inputs through the canonical PathFinder seam.

    This narrow adapter exists for legacy callers that already own pin extraction
    and obstacle lowering.  Engine construction and execution remain in this
    module, so production routing has one runtime entry while those callers are
    migrated to the full :func:`route_cell` contract.
    """

    engine = PathFinderEngine(
        widen=True,
        trunk_layers=bool(trunk_layers),
        over_active_penalty=float(over_active_penalty),
        over_active_min_level=int(over_active_min_level),
        over_active_rects_um=over_active_rects_um,
    )
    return engine.route(
        dict(pins_by_net),
        list(obstacles),
        list(constraints),
        tech,
        preroutes=None if preroutes is None else dict(preroutes),
    )



def route_cell(
    placement: Any,
    cell_def: Any,
    intent: Any,
    pdk: Any,
    *,
    require_complete: bool = False,
    device_records: Optional[Dict[str, Dict[str, Any]]] = None,
    trunk_layers: bool = False,
    dense_contacts: bool = False,
    engine: Optional[Any] = None,
    route_power: bool = True,
    placement_override: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None,
    final_placement_certificate: Any = None,
    member_ring_pad_top_nm: Tuple[Tuple[str, int], ...] = (),
    member_ring_pad_bottom_nm: Tuple[Tuple[str, int], ...] = (),
    shared_well_guard_envelope_plans: tuple = (),
    shared_well_guard_ring_inventory: Any = None,
    execution_scope: RouteExecutionScope = RouteExecutionScope.SIGNOFF,
    intent_fact_set: Any = None,
    source_inventory: Any = None,
    prepared_input_sink: Any = None,
    group_fallback_evidence_dir: Any = None,
    diagnostic_observer: Any = None,
    pre_route_gate: Any = None,
    diagnostic_compact: Any = None,
    diagnostic_policy: Any = None,
    diagnostic_source_ticket: Any = None,
) -> RouteCellResult | BringupDiagnosticRouteResult:
    """Route a placed cell through the general PathFinder engine (NO per-circuit logic).

    Parameters
    ----------
    placement
        A :class:`clay.layout.generic_placement.Placement` (the generic placer
        output -- ``place_generic(cell_def, intent, tech=pdk)``).
    cell_def
        The parsed ``CellDef`` (carries ``devices`` with pins/sizing/device_class --
        the netlist binding the placer's ``Placement`` does not retain).
    intent
        The declarative :class:`clay.layout.intent.LayoutIntent` (its ``diff_pairs``
        drive the per-pair :class:`~clay.layout.route.constraints.Mirror` about the
        SOLVED placement axis). May also carry routing widths; when it is not a
        :class:`~clay.layout.route.intent.RoutingIntent`, design-derived widths are
        synthesised (:func:`derive_width_intent`).
    pdk
        A PDK id string (e.g. ``"gf180mcu"``) or a :class:`LayoutTech`.
    require_complete
        When True, raise :class:`NetIncompleteError` if the route is not
        net-complete (mirrors ``route_generic(require_complete=True)``).
    trunk_layers
        When True, forward ``trunk_layers=True`` to the engine: assign each net a
        TRUNK metal by pin-bbox span and bias its long run to climb there (spread
        metal across M1..M_top). This is the over-cell layer climb the OTA fold's
        ``route_ota_leakfree`` uses to route a long net over a compact pack WITHOUT
        extra room. DEFAULT False keeps the historical bare ``PathFinderEngine(widen
        =True)`` route -- so every existing caller is byte-identical. Layers are a
        demand-driven resource, not a KPI; opt in per cell (e.g. a compact pack whose
        long output net cannot close on M1/M2 alone).
    placement_override
        Optional ``(placement_um, geometry)`` pair that REPLACES the placer-derived
        device positions + stamps for this route (default ``None`` -> byte-identical:
        the bridge output is used verbatim, exactly as before). This is the seam that
        lets the production route consume a DIFFERENT floorplan than the generic
        placer produced -- e.g. the compact ``@anchor/@channel/@band`` craft-intent
        floorplan from :func:`clay.layout.place_blocks_2d.reposition_pum_2d` (pins
        near the edge rails -> short local supply drops). Only the geometric positions
        change: the ``bridged_intent`` (net binding, W/L/M, device kinds) is derived
        from the SAME ``cell_def`` bridge and is POSITION-INDEPENDENT, so it is kept as
        is (never overridden). route_cell remains the SOLE owner of guard-ring / power-
        rail materialization: any ``group_guard_ring`` / ``power_rail`` stamps that
        happen to be present in the override are DROPPED first, then the rings + rails
        are (re)materialized EXACTLY ONCE on the override positions through the ONE
        existing path (so the ring-tap pins are wired + the count == len(guard_rings),
        never 2x). The mirror axis is re-derived from the override device bbox (not the
        stale placer axis) so the per-diff-pair :class:`Mirror` constraints reference
        the compact placement. The caller is responsible for having verified the
        override is a LEGAL placement (reposition's own no-overlap check) before
        passing it -- route_cell does not re-solve placement.
    final_placement_certificate
        Optional typed certificate for ``placement_override``.  When present its
        stable placement-state digest, rather than mutable observation metadata,
        scopes the final routing world and any exact INFEASIBLE feedback context.
    member_ring_pad_top_nm
        Exact member-keyed top padding from the same final-master materialization.
        It is consumed only by the canonical guard-ring materializer.
    member_ring_pad_bottom_nm
        Exact member-keyed bottom padding from the same final-master materialization.
        It is consumed only by the canonical guard-ring materializer.
    Returns a :class:`RouteCellResult` (the engine result + the INDEPENDENT union-find
    short verdict + the pins/obstacles/geometry needed to emit + verify).
    """
    if type(execution_scope) is not RouteExecutionScope:
        raise TypeError("execution_scope must be an exact RouteExecutionScope")
    if prepared_input_sink is not None and not callable(prepared_input_sink):
        raise TypeError("prepared_input_sink must be callable")
    if pre_route_gate is not None:
        from clay.layout.route.live_diagnostic import PreRouteGate
        if type(pre_route_gate) is not PreRouteGate:
            raise TypeError('pre-route gate must be an exact PreRouteGate')
        if diagnostic_observer is not None or prepared_input_sink is not None:
            raise ValueError('pre-route gate cannot share another observation boundary')
        if diagnostic_source_ticket is not None:
            raise ValueError('source handoff requires its diagnostic observer')
    if (intent_fact_set is None) != (source_inventory is None):
        raise TypeError(
            "intent_fact_set and source_inventory must be provided together"
        )
    tech = pdk if isinstance(pdk, LayoutTech) else LayoutTech.resolve(str(pdk))
    tech_id = getattr(tech, "pdk_id", None) or str(pdk)
    source_intent = intent
    guard_ring_intents = _effective_guard_ring_intents(
        source_intent,
        shared_well_guard_envelope_plans=shared_well_guard_envelope_plans,
        shared_well_guard_ring_inventory=shared_well_guard_ring_inventory,
    )
    if guard_ring_intents != tuple(getattr(source_intent, "guard_rings", ())):
        intent = replace(source_intent, guard_rings=guard_ring_intents)

    # Replay the pre-solver macro authority before reading any member artwork or
    # pins into the route world.  A caller may translate a certified aggregate,
    # but cannot replace its same-graph compiler inputs/interfaces and then ask
    # the router/emitter to bless that self-consistent substitute.
    if getattr(placement, "physical_macro_graph", None) is not None:
        from clay.layout.generic_placement import (
            verify_physical_macro_projection,
            verify_rigid_nonmos_macro_projection,
        )

        verify_physical_macro_projection(placement)
        verify_rigid_nonmos_macro_projection(placement)

    # 1. BRIDGE placer output + CellDef -> the route subsystem's input triple. The
    #    net binding comes from CellDef.devices when materialised, else from the bound
    #    .cir records (device_records) -- the SAME netlist the placer + router consume.
    bridged_intent, placement_um, geometry = _bridge_placement_to_route_inputs(
        placement, cell_def, tech_id, device_records=device_records,
        dense_contacts=dense_contacts, intent=intent,
    )
    route_engine = _resolve_route_engine(engine, trunk_layers)
    output_profile_engine_record = _maybe_freeze_native_output_profile_engine(
        placement, intent, route_engine,
    )
    consumes_plan = bool(getattr(route_engine, "consumes_resource_plan", False))
    consumes_rail_plan = consumes_plan or bool(
        getattr(route_engine, "consumes_rail_placement_plan", False))
    use_bringup_negotiation = (
        execution_scope is RouteExecutionScope.BRINGUP_DIAGNOSTIC
    )
    use_deterministic_negotiation = _supports_deterministic_negotiation(
        route_engine
    )
    if use_bringup_negotiation and not use_deterministic_negotiation:
        engine_type = (
            "default:PathFinderEngine"
            if engine is None
            else f"{type(route_engine).__module__}."
            f"{type(route_engine).__qualname__}"
        )
        raise RoutePlanningStop(
            phase="DIAGNOSTIC_CAPABILITY",
            state="DETERMINISTIC_NEGOTIATION_ENGINE_REQUIRED",
            evidence=(
                "BRINGUP_DIAGNOSTIC requires an engine with deterministic "
                "negotiated-congestion capability",
                engine_type,
            ),
            status=VerificationStatus.UNSUPPORTED,
        )
    routing_intent = _routing_intent_for(intent, bridged_intent, tech)
    verified_candidate_pool: Optional[Any] = None
    verified_resource_plan: Optional[Any] = None
    diagnostic_compile: Optional[Any] = None
    rail_candidate_pool: Optional[Any] = None
    rail_placement_plan: Optional[Any] = None
    rail_placement_replay: Optional[Any] = None
    materialized_rail_facts: Tuple[Any, ...] = ()
    materialized_rail_certificate: Optional[Any] = None
    power_plane_facts: Tuple[Any, ...] = ()

    # 1a. PLACEMENT OVERRIDE (opt-in, default None -> byte-identical): swap the placer-
    #     derived device positions/stamps for a caller-supplied compact floorplan (the
    #     `@anchor/@channel/@band` re-placement). Net binding lives in `bridged_intent`
    #     (position-INDEPENDENT), so ONLY placement_um + geometry change. Any ring/rail
    #     stamps present in the override are dropped -- route_cell is the SOLE owner of
    #     guard-ring + power-rail materialization (step 1b + 2c below), so they are
    #     (re)drawn EXACTLY ONCE on the override positions (count == len(guard_rings),
    #     never doubled). The mirror axis is re-derived from the override device bbox so
    #     the diff-pair Mirror constraints reference the compact placement, not the
    #     stale placer axis.
    override_axis_x_um: Optional[float] = None
    final_row_placement = placement
    if placement_override is not None:
        ov_pum, ov_geom = placement_override
        _drop = {
            n for n, cg in (ov_geom or {}).items()
            if str(getattr(cg, "source", "")) in (
                "group_guard_ring", "power_rail", "power_plane"
            )
            or n.startswith("GRING")
            or n.startswith("PRAIL_")
            or n.startswith("PPLANE_")
        }
        placement_um = {n: p for n, p in dict(ov_pum).items() if n not in _drop}
        geometry = {n: g for n, g in dict(ov_geom).items() if n not in _drop}
        _ov_bbox = _cell_bbox_um(placement_um, geometry)
        if _ov_bbox is not None:
            override_axis_x_um = (_ov_bbox[0] + _ov_bbox[2]) / 2.0
    # 1b. GROUP GUARD RINGS (declared `@guard_ring` intents): materialize each into
    #     the ONE shared geometry dict as placement-absolute ring stamps, exactly as
    #     anchor_ota does -- built from the FINAL placement + the SAME reusable
    #     `group_guard_rings` core (nmos cluster -> p substrate-tap ring, pmos
    #     cluster -> n nwell-tap ring, every dimension PDK-rule-resolved). Everything
    #     downstream consumes them through the EXISTING single paths: the router sees
    #     the ring frames as unowned hard keep-outs (build_layout_model), the emit
    #     boundary writes them through the same device loop, and the LVS/DRC oracles
    #     judge the SAME artifact -- no second materialization path. The ring-tap Pin
    #     is injected AFTER the pins are extracted (below) so the tie net REACHES the
    #     ring (routed metal) and the ring is not dead artwork. Gated on a non-empty
    #     `intent.guard_rings`, so a cell that declares none is byte-identical.
    ring_taps: List[Dict[str, Any]] = []
    ring_pad_rows = _validate_guard_ring_member_padding(
        member_ring_pad_top_nm,
        placement_override=placement_override,
        final_placement_certificate=final_placement_certificate,
    )
    ring_bottom_pad_rows = _validate_guard_ring_member_padding(
        member_ring_pad_bottom_nm,
        placement_override=placement_override,
        final_placement_certificate=final_placement_certificate,
    )
    if guard_ring_intents:
        ring_taps = _materialize_guard_rings(
            guard_ring_intents,
            bridged_intent,
            placement_um,
            geometry,
            tech,
            tech_id,
            member_ring_pad_top_nm=dict(ring_pad_rows),
            member_ring_pad_bottom_nm=dict(ring_bottom_pad_rows),
        )

    if placement_override is not None:
        from clay.layout.flow_placement import project_final_route_placement

        # The native placement certificate covers the actual guard-ring hulls.
        # The override deliberately omits them, so its projection must wait for
        # this sole materializer, rather than certify a device-only partial world.
        final_row_placement = project_final_route_placement(
            placement, placement_um, geometry,
            final_placement_certificate=final_placement_certificate, intent=intent,
        )

    # Structure expectations describe the final device-mask assembly, including
    # declared guard rings.  Compile only after their exact stamps have entered the
    # shared geometry dictionary; emitter-owned well unioning is certified later at
    # the write boundary.
    exact_structure_expectation = None
    if getattr(final_row_placement, "row_program", None) is not None:
        from clay.layout.spec.exact_structure_profile import (
            compile_exact_structure_expectation,
        )

        exact_structure_expectation = compile_exact_structure_expectation(
            tech,
            geometry=geometry,
            placement_um=placement_um,
            row_program=final_row_placement.row_program,
            row_materialization=final_row_placement.row_materialization,
            row_final_certificate=final_row_placement.row_final_certificate,
            final_placement=final_row_placement,
            well_merge_devices=bridged_intent.devices,
            source_cell=cell_def,
        )

    # C4 SHADOW SNAPSHOT: observe the exact final override plus its re-materialized
    # rings now, before the PRAIL artwork below can be mistaken for a placed cluster.
    # This value is report-only and never enters route-engine inputs.
    resource_evidence = build_final_placement_resource_evidence(
        placement_um,
        geometry,
        # The physical route world may lower several source-local rings into one
        # proven outer ring.  Cluster identity remains the placement
        # certificate's source-owner partition; observing with the lowered
        # declaration would collapse those owners before the flexible-envelope
        # projection can replay them.  The exact artwork is unchanged here and
        # is still read from ``geometry``.
        source_intent,
        tech,
        final_placement_certificate=final_placement_certificate,
        shared_well_guard_ring_inventory=shared_well_guard_ring_inventory,
    )
    geometry_core_envelope = _routing_envelope_from_geometry(
        placement_um, geometry
    )
    geometry_envelope = _routing_envelope_from_final_placement(
        geometry_core_envelope,
        final_placement_certificate,
    )
    ordered_floorplan_nm = None
    if getattr(final_row_placement, "ordered_row_evidence", None) is not None:
        from clay.layout.place.ordered_mirror_rows import verify_ordered_row_floorplan
        from clay.layout.route.floorplan_containment import RoutingEnvelope
        ordered_floorplan_nm = verify_ordered_row_floorplan(
            final_row_placement, intent, tech,
            geometry=geometry, placement_um=placement_um,
        )
        if (final_placement_certificate is not None
                and tuple(ordered_floorplan_nm) != geometry_envelope.bbox_nm):
            raise FinalPlacementBindingError("ordered-row and certified floorplans disagree")
        geometry_envelope = RoutingEnvelope.coerce(ordered_floorplan_nm)

    # 2. PINS (leakage-free; include power so the rails are routed nets, not keep-outs).
    #    ``route_power`` (default True -> production byte-identical): when False, the
    #    power/supply device pins are NOT extracted, so the router does not build a
    #    point-to-point tree for vdd/gnd -- used to measure the SIGNAL routing
    #    direction discipline without the (slow, high-pin) supply-net solve. The
    #    supply EDGE RAILS are still materialized as geometry (see
    #    ``_materialize_power_rails``); this only skips the supply's routed tree.
    adjacent_row_pins = []
    if getattr(final_row_placement, "native_geometry", None) is not None:
        pin_report = _extract_native_row_pin_report(
            bridged_intent, placement_um, tech=tech, geometry=geometry,
            intent=intent, source_cell=cell_def, placement_obj=final_row_placement,
            route_engine=route_engine, include_power=bool(route_power),
            extra_power_nets=_declared_ground_ports(cell_def),
            _reuse_capture=adjacent_row_pins if consumes_rail_plan else None,
        )
    else:
        pin_report = extract_pins(
            bridged_intent, placement_um, tech=tech, return_report=True,
            include_power=bool(route_power), geometry=geometry, per_device_fold=True,
            extra_power_nets=_declared_ground_ports(cell_def),
        )
    pins_by_net = pin_report.pins_by_net

    # 2a. ARRAY PIN-LANDING (Stage-1 craft, S1): a shared-diffusion @array member was
    #     probed by extract_pins as a STANDALONE MOS at its sub-centroid (MOS-
    #     geometry-blind), so its drain landed OFF the interleaved merged-OD strap
    #     -> the per-member drain nets collapsed (routing 7/15, our MOS un-extractable
    #     under LVS). Replace each member's stale pins with its REAL strap pins (drain
    #     strap column / shared-source column / shared gate contact), derived from
    #     SharedDiffusionArray.member_pins + the array tile centre. Byte-identical for
    #     a cell with no shared-diffusion array (resolves to nothing -> no-op).
    _relocate_array_member_pins(
        pins_by_net, bridged_intent, cell_def, intent, tech, placement_um,
        device_records=device_records, placement_obj=final_row_placement,
        geometry=geometry,
        allow_single_component_joint=bool(
            getattr(route_engine, 'consumes_single_component_joint_terminals', False)),
        allow_fixed_multilayer_joint=bool(
            getattr(route_engine, 'consumes_fixed_multilayer_joint_terminals', False)),
        multilayer_engine=route_engine,
        _adjacent_native_row_pins=adjacent_row_pins[0] if adjacent_row_pins else None,
    )
    # Native rail terminal counts describe source components. Ring taps,
    # anchors and shield endpoints are separate, later route obligations.
    source_pins_by_net = {net: tuple(pins) for net, pins in pins_by_net.items()}

    # 2b. RING-TAP PINS: one extra same-net Pin per materialized ring, landing on the
    #     ring's tap M1 segment. The router wires it like any terminal (no engine
    #     change), so the tie net's trunk reaches the ring frame -> a REAL tie, not
    #     dead artwork. The tap net is the ring's DESIGN body net (group_guard_rings
    #     derived it from the members' body terminal); a ring with an empty tap net
    #     (an untied p-ring the PDK substrate biases globally) injects no pin.
    # Nets declared as SUPPLY EDGE RAILS (materialized as drawn M2 rails below). When
    # ``route_power`` is False these nets are NOT routed as trees, so their guard-ring
    # taps are not wired either (the rail is drawn geometry; a dangling tap pin would
    # only reintroduce the slow cell-spanning supply solve).
    edge_rail_nets = {
        str(getattr(t, "net", "") or "").strip().lower()
        for t in (getattr(intent, "trunks", ()) or ())
        if str(getattr(t, "edge", "") or "")
    }
    _append_guard_ring_tap_pins(
        pins_by_net, ring_taps, route_power=route_power, edge_rail_nets=edge_rail_nets)

    # 2b.1. DECLARED SIGNAL EDGE ACCESS: a net @anchor is a route obligation,
    # not merely a soft placement pull.  Add one PDK-derived upper-horizontal
    # endpoint whose actual emitted carrier reaches the certified edge. The
    # negotiated producer's square end cap differs from exact-plan flush ends.
    _materialize_declared_anchor_edge_pins(
        intent, pins_by_net, routing_intent, tech, geometry_envelope,
        square_wire_ends=bool(
            use_deterministic_negotiation
            and getattr(route_engine, "negotiated_wire_square_ends", False)
        ),
    )

    # 2b.1a. DECLARED SHIELD FENCES: derive one non-vacuous corridor from the
    # canonical aggressor/victim terminal domains and add two same-layer pins to
    # the declared shield net.  The ordinary router must connect through the
    # complete fence; no reference coordinate or per-circuit rule is involved.
    from clay.layout.intent import replay_shield_inventory

    shield_declarations = replay_shield_inventory(intent.shields)
    if shield_declarations:
        from clay.layout.route.shield import (
            installed_shield_layer_policy,
            materialize_declared_shield_fences,
        )

        layer_by_axis, grid_nm, minimum_width_nm = (
            installed_shield_layer_policy(tech)
        )
        source_member_names = frozenset(
            str(member.name)
            for collection in (
                bridged_intent.devices,
                bridged_intent.passives,
                getattr(bridged_intent, "bjts", ()),
            )
            for member in tuple(collection)
        )
        materialize_declared_shield_fences(
            shield_declarations,
            pins_by_net,
            layer_by_axis=layer_by_axis,
            manufacturing_grid_nm=grid_nm,
            minimum_width_nm=minimum_width_nm,
            domain_pin_owners=source_member_names,
            emitted_device_geometry=geometry,
            emitted_device_placement_um=placement_um,
            tech=tech,
            wire_widths_nm=intent.wire_widths,
        )

    # 2b.2. UPPER-METAL POWER PLANES: each declaration owns one complete
    # coordinate-free final-placement envelope on its PDK-resolved layer.  The
    # geometry and same-net access pin enter the same emitter/model dictionaries as
    # every other owned conductor; no reference coordinate or secondary GDS path is
    # involved.  A diagnostic that disables power routing may not emit a floating
    # plane and claim the declaration was consumed.
    from clay.layout.intent import replay_power_plane_inventory
    from clay.layout.route.power_plane import materialize_power_planes

    power_plane_declarations = replay_power_plane_inventory(intent.power_planes)
    if power_plane_declarations and not route_power:
        raise RoutePlanningStop(
            phase="POWER_PLANE_MATERIALIZATION",
            state="ROUTE_POWER_DISABLED",
            evidence=power_plane_declarations,
            status=VerificationStatus.UNSUPPORTED,
        )
    power_plane_facts = materialize_power_planes(
        power_plane_declarations,
        geometry_envelope.bbox_nm,
        tech,
        geometry=geometry,
        placement_um=placement_um,
        pins_by_net=pins_by_net,
    )

    # 2c. POWER EDGE RAILS: materialize each declared edge-rail @trunk (VDD top / GND
    #     bottom, 8um M2) into the ONE geometry dict + inject a same-net rail Pin so
    #     the rail seeds the gridless upper-artwork model as OWNED metal (foreign nets
    #     route around it; the supply net drops onto it -- a REAL tie, not artwork).
    #     Runs BEFORE build_layout_model so the rail is projected as an obstacle, and
    #     after the pins exist so the rail Pin joins the net. A cell with no edge-rail
    #     @trunk yields nothing -> byte-identical.
    edge_rail_demands = _compile_edge_rail_demands_from_intent(
        intent, routing_intent, tech, cell_def=cell_def
    )
    # A reused pin inventory is provisional until the same independent final
    # verifier used by rail planning replays the current source and artwork.
    if adjacent_row_pins and adjacent_row_pins[0].used and not (consumes_rail_plan and edge_rail_demands):
        from clay.layout.row_terminal_probes import require_routing_row_terminal_probes
        require_routing_row_terminal_probes(
            tech=tech, geometry=geometry, placement_um=placement_um, pins_by_net=pins_by_net,
            row_program=final_row_placement.row_program,
            row_materialization=final_row_placement.row_materialization,
            row_final_certificate=final_row_placement.row_final_certificate,
            intent=intent, source_cell=cell_def,
            allow_fixed_multilayer=bool(getattr(route_engine, 'consumes_fixed_multilayer_joint_terminals', False)), multilayer_engine=route_engine)
    plane_owner_by_layer = {
        fact.layer_key: fact.net for fact in power_plane_facts
    }
    for demand in edge_rail_demands:
        layer_key = tuple(getattr(demand, "layer_key", ()))
        if (
            layer_key in plane_owner_by_layer
            and plane_owner_by_layer[layer_key] != str(getattr(demand, "net", ""))
        ):
            raise ValueError(
                "power plane and edge rail place different nets on one physical layer"
            )
    if (
        (consumes_rail_plan or use_bringup_negotiation)
        and edge_rail_demands
        and not route_power
    ):
        raise RoutePlanningStop(
            phase="RAIL_PLACEMENT",
            state="ROUTE_POWER_DISABLED",
            evidence=edge_rail_demands,
            status=VerificationStatus.UNSUPPORTED,
        )
    if use_bringup_negotiation:
        # NON_SIGNOFF bring-up constructs stable, legal outside-edge physical
        # facts directly from the already-compiled demands and final bbox.  The
        # shared writer emits them, but this branch mints no candidate pool,
        # plan, optimality replay, or materialization certificate.
        power_rails = _materialize_bringup_power_rails(
            edge_rail_demands,
            placement_um,
            geometry,
            tech,
            pins_by_net=pins_by_net,
        )
    elif consumes_rail_plan and edge_rail_demands:
        # Planning observes a style-independent exact scene.  Device-active
        # projections are objective carriers only; they never become obstacles.
        from clay.layout.row_terminal_probes import require_routing_row_terminal_probes as require_row_terminal_probes

        require_row_terminal_probes(
            tech=tech, geometry=geometry, placement_um=placement_um, pins_by_net=pins_by_net,
            row_program=final_row_placement.row_program,
            row_materialization=final_row_placement.row_materialization,
            row_final_certificate=final_row_placement.row_final_certificate,
            intent=intent, source_cell=cell_def,
            allow_fixed_multilayer=bool(getattr(route_engine, 'consumes_fixed_multilayer_joint_terminals', False)),
            multilayer_engine=route_engine,
        )
        pre_rail_model = build_layout_model(
            bridged_intent,
            placement_um,
            geometry,
            tech=tech,
            pins_by_net=pins_by_net,
            include_upper_artwork=True,
            strict_device_m1_provenance=bool(
                getattr(route_engine, "strict_device_m1_provenance", False)
            ),
            include_device_projection=False,
        )
        from clay.layout.place.edge_rail_geometry import verify_final_edge_rail_geometry

        actual_shared_guard_edge_rail = verify_final_edge_rail_geometry(
            certificate=final_placement_certificate,
            geometry=geometry, placement_um=placement_um, layout_model=pre_rail_model,
            demands=edge_rail_demands, pins_by_net=pins_by_net, tech=tech,
            cell_def=cell_def, bridged_intent=bridged_intent,
            source_pins_by_net=source_pins_by_net,
            intent=intent, row_program=final_row_placement.row_program,
            row_materialization=final_row_placement.row_materialization,
            row_final_certificate=final_row_placement.row_final_certificate,
            native_geometry=getattr(final_row_placement, "native_geometry", None),
        )
        rail_scene = _build_pre_rail_scene(
            placement_um=placement_um,
            geometry=geometry,
            pins_by_net=pins_by_net,
            layout_model=pre_rail_model,
            placement_provenance_digest=resource_evidence.sha256,
            demands=edge_rail_demands,
            tech=tech,
            final_placement_certificate=final_placement_certificate,
            declared_floorplan_nm=ordered_floorplan_nm,
        )
        rail_candidate_pool, rail_placement_plan, rail_placement_replay = _plan_rails(
            rail_scene, edge_rail_demands, tech
        )
        power_rails, materialized_rail_facts = _materialize_power_rails(
            rail_candidate_pool,
            rail_placement_plan,
            rail_placement_replay,
            tech,
            placement_um,
            geometry,
            pins_by_net=pins_by_net,
        )
    elif consumes_rail_plan:
        power_rails = []
    else:
        # Historical engines retain their old edge-rail path.  With no edge
        # declaration this remains a literal no-op and no planning kwarg is sent.
        power_rails = _legacy_materialize_power_rails(
            tuple(getattr(intent, "trunks", ()) or ()),
            placement_um,
            geometry,
            tech,
            pins_by_net=pins_by_net,
        )
    # RAIL-SPINE hand-off: net -> [(rail_layer, rail_bbox_um)] for every materialized
    # supply rail, so an engine that consumes rails (declares ``consumes_rail_spine``)
    # drops each supply pin LOCALLY onto the owned rail metal instead of solving the
    # supply net as a whole-cell MST toward one hub pin. Keyed ONLY on the declared
    # edge-rail @trunk (via power_rails) + the engine capability -- no per-circuit
    # branch; passed to route() below only when the engine advertises the spine path
    # (the default PathFinder never receives it -> byte-identical).
    rails_by_net: Dict[str, List[Any]] = {}
    for plane in power_plane_facts:
        rails_by_net.setdefault(plane.net, []).append((
            plane.layer_key,
            tuple(value / _NM_PER_UM for value in plane.bbox_nm),
        ))
    if power_rails and route_power:
        from clay.layout.route.engine import Pin as _RailPin

        for pr in power_rails:
            net = str(pr["net"])
            if not net:
                continue
            rails_by_net.setdefault(net, []).append(
                (tuple(pr["layer"]), tuple(pr["bbox_um"])))
            cx_um, cy_um = pr["center_um"]
            # ONE rail-access Pin at the rail centre: seeds the gridless upper-artwork
            # model so the rail is OWNED metal for its net, and gives the supply a
            # connection node onto the rail. (An earlier per-supply-pin "drop" fan-out
            # ballooned the power net's pin count and made its point-to-point solve the
            # route's bottleneck -- the rail is drawn geometry, not a routed tree.)
            # Skipped entirely under route_power=False: the rail is then pure emitted
            # geometry (a keep-out signals route around), no supply tree at all.
            if not any(
                str(getattr(pin, "device", "")) == str(pr["rail"])
                and str(getattr(pin, "terminal", "")) == "rail"
                for net_pins in pins_by_net.values()
                for pin in net_pins
            ):
                pins_by_net.setdefault(net, []).append(
                    _RailPin(
                        net=net,
                        device=pr["rail"],
                        terminal="rail",
                        xy_um=(float(cx_um), float(cy_um)),
                        layer=tuple(pr["layer"]),
                        xy_nm=(
                            int(round(float(cx_um) * _NM_PER_UM)),
                            int(round(float(cy_um) * _NM_PER_UM)),
                        ),
                    )
                )

    # One canonical physical domain: the certified final placement rectangle.
    # Declared edge rails are exact bands inside that rectangle and may not enlarge
    # it; no route-owned geometry is allowed to escape the human floorplan.
    routing_envelope_certificate = None
    if final_placement_certificate is not None:
        from clay.layout.route.floorplan_containment import (
            RoutingEnvelope,
            certify_routing_envelope,
        )

        certified_core = RoutingEnvelope.coerce(
            getattr(final_placement_certificate, "bbox_nm", ())
        )
        if certified_core.bbox_nm != geometry_envelope.bbox_nm:
            raise FinalPlacementBindingError(
                "final placement certificate does not bind the exact routed "
                f"override: certificate={certified_core.bbox_nm}, "
                f"geometry={geometry_envelope.bbox_nm}"
            )
        rail_bboxes_nm = tuple(
            tuple(
                int(round(float(value) * _NM_PER_UM))
                for value in rail["bbox_um"]
            )
            for rail in power_rails
        )
        routing_envelope_certificate = certify_routing_envelope(
            final_placement_certificate,
            rail_bboxes_nm=rail_bboxes_nm,
        )
        routing_envelope = RoutingEnvelope.coerce(
            routing_envelope_certificate.envelope_nm
        )
        if routing_envelope.bbox_nm != geometry_envelope.bbox_nm:
            raise FinalPlacementBindingError(
                "final materialized geometry envelope differs from the exact "
                "placement floorplan: "
                f"certificate={routing_envelope.bbox_nm}, "
                f"geometry={geometry_envelope.bbox_nm}"
            )
    else:
        routing_envelope = geometry_envelope

    # 2d. GROUP-ROUTE HANDOFF: freeze every declared physical group's complete
    # topology-derived external port denominator against the final member boxes
    # and the exact Pin inventory.  A declaration is never silently ignored by
    # an engine that lacks the local-port consumer.
    group_route_contract = _group_route_contract_for_engine(
        cell_def,
        source_intent,
        final_placement_certificate,
        pins_by_net,
        route_engine,
        intent_fact_set=intent_fact_set,
        source_inventory=source_inventory,
    )

    # 2e. TOPOLOGY COMPILER: pins, ring taps, and edge rails are now final.
    # Compile their immutable net demands and PDK/declaration-derived topology
    # candidates at the canonical route_cell seam. Plan-consuming engines lower
    # this plan into their candidate pool below; non-consuming engines receive no
    # planner kwargs and expose the plan as report-only evidence.
    from clay.layout.route.topology_compile import compile_topology_plan

    topology_plan = compile_topology_plan(
        pins_by_net,
        cell_def=cell_def,
        intent=intent,
        routing_intent=routing_intent,
        tech=tech,
        guard_ring_taps=tuple(ring_taps),
    )

    # 3. OBSTACLES: net-aware device M1 pads via the unified layout model (drop the
    #    OTA-specific cap-plate / no-route-under conductors -- the bridged intent has
    #    no passives, so build_layout_model emits device pads only).
    #    The routing engine is resolved FIRST (pure argument default, no other
    #    dependency) because the model projection is capability-keyed: an engine
    #    that declares ``consumes_artwork_conductors`` (the gridless strategy)
    #    additionally receives every stamp's ABOVE-M1 riser pads + drawn via cuts
    #    as net-tagged conductors -- the artwork the deck judges net-blind
    #    (measured M2.S.1/M3.S.1 riser-gap + VIA2.S.1 riser-cut families). The
    #    default PathFinder path keeps the byte-identical M1+plates model.
    from clay.layout.row_terminal_probes import require_routing_row_terminal_probes as require_row_terminal_probes

    require_row_terminal_probes(
        tech=tech, geometry=geometry, placement_um=placement_um, pins_by_net=pins_by_net,
        row_program=final_row_placement.row_program,
        row_materialization=final_row_placement.row_materialization,
        row_final_certificate=final_row_placement.row_final_certificate,
        intent=intent, source_cell=cell_def,
        allow_fixed_multilayer=bool(getattr(route_engine, 'consumes_fixed_multilayer_joint_terminals', False)),
        multilayer_engine=route_engine,
    )
    include_upper_artwork = bool(
        getattr(route_engine, "consumes_artwork_conductors", False)
    )
    reserve_column_guards = bool(
        getattr(route_engine, "consumes_cap_column_guards", include_upper_artwork)
    )
    layout_model = build_layout_model(
        bridged_intent, placement_um, geometry, tech=tech, pins_by_net=pins_by_net,
        include_upper_artwork=include_upper_artwork,
        reserve_column_guards=reserve_column_guards,
        # Gridless declares this TRUE as an unconditional safety capability.
        strict_device_m1_provenance=bool(
            getattr(route_engine, "strict_device_m1_provenance", False)),
        include_device_projection=bool(
            getattr(route_engine, "consumes_device_active_projection", False)),
    )
    obstacles = layout_model.router_obstacles()
    _verify_native_output_profile_fixed_stacks(
        cell_def=cell_def,
        tech=tech,
        route_engine=route_engine,
        output_profile_engine_record=output_profile_engine_record,
        geometry=geometry,
        placement_um=placement_um,
        final_row_placement=final_row_placement,
        intent=intent,
        layout_model=layout_model,
        obstacles=obstacles,
        pins_by_net=pins_by_net,
    )

    # 3b. FINAL ROUTING-WORLD PROVENANCE: all facts that can change whether a
    # candidate pool applies are now fixed. A pool's internal replay proof is
    # necessary but not sufficient; only an exact required-net/placement/topology/
    # PDK/world match may reach the engine. No candidate geometry is looked up or
    # materialized here (selection-only plans remain a later C2 obligation).
    from clay.layout.route.resource_world import (
        build_routing_world_provenance,
        verify_candidate_pool_applicability,
    )

    placement_provenance_digest = _final_placement_provenance_digest(
        resource_evidence, final_placement_certificate
    )
    routing_world_provenance = build_routing_world_provenance(
        pins_by_net=pins_by_net,
        obstacles=obstacles,
        rails_by_net=rails_by_net,
        placement_provenance_digest=placement_provenance_digest,
        topology_plan=topology_plan,
        tech=tech,
    )
    if rail_candidate_pool is not None:
        from clay.layout.route.rail_plan import (
            MaterializedRailPlanCertificate,
            _verify_materialized_rail_plan_certificate_with_replay,
        )

        materialized_rail_certificate = MaterializedRailPlanCertificate(
            candidate_pool_hash=rail_candidate_pool.sha256(),
            rail_plan_hash=rail_placement_plan.sha256(),
            final_routing_world_hash=routing_world_provenance.sha256(),
            materialized_facts=materialized_rail_facts,
        )
        _verify_materialized_rail_plan_certificate_with_replay(
            rail_candidate_pool,
            rail_placement_plan,
            rail_placement_replay,
            materialized_rail_certificate,
            tech=tech,
            geometry=geometry,
            placement_um=placement_um,
            pins_by_net=pins_by_net,
            routing_world=routing_world_provenance,
        )
    # Engines with a deterministic feasibility seam allocate exact route
    # geometry themselves; forcing their certified SIGNOFF calls through the
    # monolithic CandidatePool duplicates that allocation domain and can reject
    # a routable placement before the engine sees it.  The final-placement
    # certificate remains load-bearing in ``routing_world_provenance`` above and
    # is carried into the result for exact emission/replay.  Engines without the
    # seam still require the globally allocated resource plan.
    if consumes_plan and not use_deterministic_negotiation:
        compile_kwargs = {}
        if final_placement_certificate is not None:
            compile_kwargs["final_placement_certificate"] = (
                final_placement_certificate
            )
        compiler_product, verified_resource_plan = _compile_route_resource_plan(
            topology_plan,
            routing_world_provenance,
            tech,
            **compile_kwargs,
        )
        # The private compiler seam is intentionally monkeypatchable for focused
        # tests, but its output is never trusted: route_cell remains the sole
        # applicability/replay authority at the engine boundary.
        if isinstance(compiler_product, _DiagnosticRouteCompilerProduct):
            diagnostic_compile = compiler_product
        else:
            verified_candidate_pool = compiler_product
            _verify_route_resource_plan_or_stop(
                verified_candidate_pool, verified_resource_plan
            )
            verify_candidate_pool_applicability(
                verified_candidate_pool, routing_world_provenance, tech=tech
            )

    # Preserve the solved/final placement centre as result metadata.  It is no
    # longer a fallback authority for hard Mirror constraints.
    axis_x_um = (
        override_axis_x_um
        if override_axis_x_um is not None
        else float(getattr(placement, "axis_x", 0)) / _NM_PER_UM
    )

    # 4. CONSTRAINTS: lower explicit routing intent first. A LayoutIntent diff
    #    pair is a placement/parasitic match, not automatically a hard whole-net
    #    route mirror. Add Mirror only when FINAL accesses prove one exact axis.
    #    An explicit RoutingIntent Mirror remains hard and is checked here.
    # Lower with the design's declared direction_weight and the PDK-DERIVED HVH axes
    # (conventional_axes_for reads the adapter's preferred_routing_axes, else the
    # generic alternation), so a declared HVH discipline reaches the engine as the
    # process-correct per-layer Direction preferences. weight 0 -> no Direction ->
    # byte-identical for a cell that declares no direction discipline.
    cons: List[Any] = list(lower_intent(
        routing_intent,
        direction_weight=float(getattr(routing_intent, "direction_weight", 0.0) or 0.0),
        axes=conventional_axes_for(tech),
    ))
    from clay.layout.route.channel_geometry import bind_channel_routes
    cons.extend(bind_channel_routes(
        intent, placement_um, geometry, tech, final_row_placement,
    ))
    present = set(pins_by_net)
    from clay.layout.net_spacing import lower_net_spacings
    cons.extend(lower_net_spacings(getattr(intent, "net_spacings", ()), present))
    declared_length_pairs = {
        frozenset((str(constraint.net_a), str(constraint.net_b)))
        for constraint in cons
        if isinstance(constraint, _C.Length)
    }
    for constraint in _layout_match_length_constraints(intent, present):
        pair = frozenset((constraint.net_a, constraint.net_b))
        if pair not in declared_length_pairs:
            cons.append(constraint)
            declared_length_pairs.add(pair)
    for constraint in _layout_route_match_balance_constraints(intent, present):
        pair = frozenset((constraint.net_a, constraint.net_b))
        if pair not in declared_length_pairs:
            cons.append(constraint)
            declared_length_pairs.add(pair)
    explicit_mirror_pairs: set[frozenset[str]] = set()
    for constraint in tuple(cons):
        if not isinstance(constraint, _C.Mirror):
            continue
        na, nb = str(constraint.net_a), str(constraint.net_b)
        explicit_mirror_pairs.add(frozenset((na, nb)))
        if na not in present or nb not in present or na == nb:
            raise RoutePlanningStop(
                phase="MIRROR_AXIS_BINDING",
                state="MIRROR_NET_ABSENT",
                status=VerificationStatus.FAIL,
                evidence={"net_a": na, "net_b": nb},
            )
        exact_axis = _require_exact_mirror_axis_um(
            pins_by_net[na], pins_by_net[nb], net_a=na, net_b=nb
        )
        if int(round(exact_axis * _NM_PER_UM)) != int(
            round(float(constraint.axis_x_um) * _NM_PER_UM)
        ):
            raise RoutePlanningStop(
                phase="MIRROR_AXIS_BINDING",
                state="DECLARED_AXIS_MISMATCH",
                status=VerificationStatus.FAIL,
                evidence={
                    "net_a": na,
                    "net_b": nb,
                    "declared_axis_nm": int(
                        round(float(constraint.axis_x_um) * _NM_PER_UM)
                    ),
                    "endpoint_axis_nm": int(round(exact_axis * _NM_PER_UM)),
                },
            )
    for pair in getattr(intent, "diff_pairs", ()) or ():
        na, nb = str(pair[0]), str(pair[1])
        if (
            na in present
            and nb in present
            and frozenset((na, nb)) not in explicit_mirror_pairs
        ):
            mirror = _opportunistic_match_mirror(
                pins_by_net[na], pins_by_net[nb], net_a=na, net_b=nb,
                route_constraints=cons,
                group_route_contract=group_route_contract,
            )
            if mirror is not None:
                cons.append(mirror)
    _validate_trunk_constraints_for_tech(cons, tech)

    engine_constraints = cons
    suppressed_hard_constraints: Tuple[Any, ...] = ()
    if use_bringup_negotiation:
        (
            engine_constraints,
            suppressed_hard_constraints,
        ) = _diagnostic_negotiation_constraints(route_engine, cons)

    # 4b. PRE-GATE (A2 -- capability handshake). ``cons`` is now fully assembled
    #     (lowered intent + the per-diff-pair Mirror). BEFORE routing, verify the
    #     SELECTED engine can materialize every HARD/required constraint KIND; a HARD
    #     kind the engine does not satisfy would otherwise be silently emitted as a
    #     GDS that ignores it (a mirror never reflected, a shield never routed). Stop
    #     here with a typed 5-state UNSUPPORTED verdict. SOFT constraints (Direction)
    #     are advisory and never gate. NO-OP + byte-identical for a cell all of whose
    #     HARD kinds the engine satisfies -- the production PathFinder OTA
    #     (layer_set/width/mirror all supported) compiles + routes exactly as before.
    _preroute_capability_gate(route_engine, engine_constraints)

    # 5. ROUTE through the faithful-router engine. The DEFAULT is the general
    #    negotiated-congestion PathFinder (widen=True). trunk_layers (opt-in, default
    #    False -> byte-identical) forwards the upper-layer climb the OTA fold uses:
    #    each net gets a trunk metal by pin-bbox span so a long net (e.g. a compact
    #    pack's output) climbs to an upper layer instead of contending on M1/M2.
    #
    #    ``engine`` (opt-in, default None -> byte-identical PathFinder) lets a caller
    #    supply an ALTERNATE :class:`~clay.layout.route.engine.RouteEngine` strategy --
    #    e.g. the gridless / shape-based over-the-cell engine that routes on continuous
    #    nm coordinates at DRC-min clearance (the lever for packing below the uniform
    #    grid's floor). The engine is a STRATEGY behind the SAME contract; everything
    #    downstream (pins / obstacles / constraints / the independent short + LVS
    #    oracles) is unchanged, so the alternate engine drops in without touching intent
    #    or verification. None keeps the production route byte-for-byte identical.
    #    (Resolved above, before the layout model, so the model projection can key on
    #    the engine's declared world-model capability.)
    route_kwargs: Dict[str, Any] = {}
    if group_route_contract is not None:
        route_kwargs["group_route_contract"] = group_route_contract
        # This separate opt-in consumes the plan only as a soft ranking input;
        # it does not claim the resource-plan capability or alter the policy.
        from clay.layout.route.engine_gridless import GridlessEngine
        if (type(route_engine) is GridlessEngine and
                getattr(route_engine, '_experimental_group_demand_guidance', False)):
            route_kwargs['group_demand_topology_plan'] = topology_plan
    if rails_by_net and getattr(route_engine, "consumes_rail_spine", False):
        route_kwargs["rails"] = rails_by_net
    if getattr(route_engine, "consumes_routing_envelope", False):
        route_kwargs["routing_envelope_nm"] = routing_envelope
    engine_pins_by_net = pins_by_net
    if diagnostic_compile is not None:
        (
            engine_pins_by_net,
            engine_constraints,
            suppressed_hard_constraints,
        ) = (
            _diagnostic_materialization_inputs(
                pins_by_net, cons, verified_resource_plan
            )
        )
        route_kwargs["diagnostic_resource_program"] = verified_resource_plan
        route_kwargs["diagnostic_full_pins_by_net"] = pins_by_net
    elif verified_resource_plan is not None:
        require_resource_plan_support(route_engine, verified_resource_plan)
        route_kwargs["resource_plan"] = verified_resource_plan
    fixed_index_seal_budget = getattr(route_engine, '_experimental_fixed_index_seal_budget', None)
    if fixed_index_seal_budget is not None and (
            pre_route_gate is not None or diagnostic_observer is not None
            or 'resource_plan' in route_kwargs or 'diagnostic_resource_program' in route_kwargs):
        from clay.layout.route.fixed_index_seal import FixedIndexSealStop
        raise FixedIndexSealStop('FIXED_INDEX_SEAL_UNSUPPORTED')
    post_group = None
    if (type(diagnostic_observer).__module__ == 'clay.layout.route.group_endpoint_observation'
            and type(diagnostic_observer).__name__ == '_BoundObserver'):
        from clay.layout.route.group_endpoint_observation import _BoundObserver
        if type(diagnostic_observer) is not _BoundObserver:
            raise TypeError('group endpoint observation has a foreign bound request')
        post_group = diagnostic_observer
    post_group_cap = None
    if post_group is not None:
        from clay.layout.route.live_diagnostic import LiveRoutingDiagnostic
        if (post_group.engine is not route_engine or post_group.request is not post_group.authority.request
                or diagnostic_source_ticket is not None
                or pre_route_gate is not None or diagnostic_compile is not None
                or diagnostic_compact is None
                or placement is not diagnostic_compact.logical_placement_basis
                or placement_override is not diagnostic_compact.override
                or final_placement_certificate is not diagnostic_compact.final_placement_certificate):
            raise ValueError('group endpoint source routing caller differs')
        post_group.authority.check()
        post_group_cap = LiveRoutingDiagnostic(compact=diagnostic_compact,
            effective_policy=diagnostic_policy, capture_endpoint_authority=True,
            route_context=dict(placement=placement, placement_override=placement_override,
                intent=intent, device_records=device_records, topology_plan=topology_plan,
                intent_fact_set=intent_fact_set, source_inventory=source_inventory),
            snapshot_inputs=dict(source_cell=cell_def, tech=tech, geometry=geometry,
                placement_um=placement_um, pins_by_net=engine_pins_by_net, obstacles=obstacles,
                constraints=engine_constraints, rails_by_net=rails_by_net,
                routing_envelope=routing_envelope, routing_world=routing_world_provenance,
                final_placement_certificate=final_placement_certificate, execution_scope=execution_scope,
                route_engine=route_engine, route_kwargs=route_kwargs,
                actual_shared_guard_edge_rail=locals().get('actual_shared_guard_edge_rail')))
    if diagnostic_observer is not None and post_group is None:
        from clay.layout.route.live_diagnostic import observe_prepared_routing
        if diagnostic_compact is not None and (
                placement is not diagnostic_compact.logical_placement_basis
                or placement_override is not diagnostic_compact.override
                or final_placement_certificate is not diagnostic_compact.final_placement_certificate):
            raise ValueError('diagnostic compact differs from actual route-cell caller')
        from contextlib import nullcontext
        if diagnostic_source_ticket is None:
            source_scope = nullcontext()
        else:
            from clay.layout.route.source_handoff import _route_producer_scope
            source_scope = _route_producer_scope(diagnostic_source_ticket,
                pin_report=pin_report, source_pins_by_net=source_pins_by_net,
                final_row_placement=final_row_placement, topology_plan=topology_plan,
                full_pins_by_net=pins_by_net, engine_pins_by_net=engine_pins_by_net,
                world=routing_world_provenance, source_cell=cell_def, tech=tech,
                geometry=geometry, placement_um=placement_um, intent=intent,
                engine=route_engine, diagnostic_compile=diagnostic_compile)
        with source_scope:
            observe_prepared_routing(diagnostic_observer, compact=diagnostic_compact,
            effective_policy=diagnostic_policy,
            route_context=dict(placement=placement, placement_override=placement_override,
                               intent=intent, device_records=device_records, topology_plan=topology_plan,
                               intent_fact_set=intent_fact_set, source_inventory=source_inventory),
            snapshot_inputs=dict(source_cell=cell_def, tech=tech, geometry=geometry,
                placement_um=placement_um, pins_by_net=engine_pins_by_net, obstacles=obstacles,
                constraints=engine_constraints, rails_by_net=rails_by_net,
                routing_envelope=routing_envelope, routing_world=routing_world_provenance,
                final_placement_certificate=final_placement_certificate, execution_scope=execution_scope,
                route_engine=route_engine, route_kwargs=route_kwargs,
                 actual_shared_guard_edge_rail=locals().get('actual_shared_guard_edge_rail')))
    if prepared_input_sink is not None:
        from clay.layout.route.canonical_input_capture import snapshot_prepared_routing_input

        prepared_input_sink(snapshot_prepared_routing_input(
            source_cell=cell_def, tech=tech, geometry=geometry, placement_um=placement_um,
            pins_by_net=engine_pins_by_net, obstacles=obstacles, constraints=engine_constraints,
            rails_by_net=rails_by_net, routing_envelope=routing_envelope,
            routing_world=routing_world_provenance,
            final_placement_certificate=final_placement_certificate,
            execution_scope=execution_scope, route_engine=route_engine,
            route_kwargs=route_kwargs,
            actual_shared_guard_edge_rail=locals().get('actual_shared_guard_edge_rail')))
    from contextlib import nullcontext
    if post_group is None:
        post_scope = nullcontext()
    else:
        from clay.layout.route.group_endpoint_observation import observation_scope
        post_scope = observation_scope(post_group, post_group_cap)
    with post_scope:
        # Install on the declared exact Gridless strategy only; the scene and
        # evidence directory are constructed only when feedback is needed.
        _check_source_engine_before_group_fallback(route_engine, output_profile_engine_record)
        restore_group_fallback = None
        if group_fallback_evidence_dir is not None and group_route_contract is not None:
            from clay.layout.route.engine_gridless import GridlessEngine
            if type(route_engine) is GridlessEngine:
                from clay.layout.route.group_fallback_integration import install_group_fallback
                restore_group_fallback = install_group_fallback(
                    route_engine, geometry=geometry, placement_um=placement_um,
                    pins_by_net=pins_by_net, tech=tech,
                    routing_world_sha256=routing_world_provenance.sha256(),
                    evidence_root=group_fallback_evidence_dir)
        try:
            if pre_route_gate is not None:
                from clay.layout.route.live_diagnostic import gate_prepared_routing
                if diagnostic_compact is not None and (
                        placement is not diagnostic_compact.logical_placement_basis
                        or placement_override is not diagnostic_compact.override
                        or final_placement_certificate is not diagnostic_compact.final_placement_certificate):
                    raise ValueError('pre-route compact differs from actual route-cell caller')
                gate_prepared_routing(pre_route_gate, compact=diagnostic_compact,
                    effective_policy=diagnostic_policy,
                    route_context=dict(placement=placement, placement_override=placement_override,
                        final_row_placement=final_row_placement,
                        intent=intent, device_records=device_records, topology_plan=topology_plan,
                        intent_fact_set=intent_fact_set, source_inventory=source_inventory),
                    snapshot_inputs=dict(source_cell=cell_def, tech=tech, geometry=geometry,
                        placement_um=placement_um, pins_by_net=engine_pins_by_net, obstacles=obstacles,
                        constraints=engine_constraints, rails_by_net=rails_by_net,
                        routing_envelope=routing_envelope, routing_world=routing_world_provenance,
                        final_placement_certificate=final_placement_certificate, execution_scope=execution_scope,
                        route_engine=route_engine, route_kwargs=route_kwargs,
                        actual_shared_guard_edge_rail=locals().get('actual_shared_guard_edge_rail')))
                from clay.flow.execution_session import current_session
                session = current_session()
                if session is not None:
                    session.checkpoint()
            if fixed_index_seal_budget is None:
                routed = route_engine.route(
                    engine_pins_by_net, obstacles, engine_constraints, tech, **route_kwargs,
                )
            else:
                from clay.layout.route.fixed_index_seal import fixed_index_seal_scope
                with fixed_index_seal_scope(route_engine) as seal_scope:
                    routed = route_engine.route(
                        engine_pins_by_net, obstacles, engine_constraints, tech, **route_kwargs,
                    )
                    if not seal_scope.published:
                        from clay.layout.route.fixed_index_seal import FixedIndexSealStop
                        raise FixedIndexSealStop('FIXED_INDEX_SEAL_UNSUPPORTED')
        except Exception as exc:
            from clay.layout.route.engine_gridless import NegotiationWatchdogStop

            if type(exc) is NegotiationWatchdogStop and post_group is None:
                stop_phase = str(getattr(exc.stop, "phase", "") or "").strip()
                raise RoutePlanningStop(
                    phase=stop_phase or "NEGOTIATED_CONGESTION",
                    state=exc.stop.reason.value,
                    evidence=exc.stop,
                    status=VerificationStatus.NO_CONVERGENCE,
                ) from exc
            if not isinstance(exc, RouteResourcePlanIncompleteError):
                raise
            if verified_resource_plan is None:
                raise
            raise RoutePlanningStop(
                phase="RESOURCE_MATERIALIZATION",
                state=exc.state,
                evidence=exc,
                status=exc.status,
            ) from exc
        finally:
            if restore_group_fallback is not None:
                import sys
                original = sys.exception()
                try:
                    restore_group_fallback()
                except BaseException as cleanup:
                    if original is None:
                        raise
                    from clay.flow.control_stop import record_unwind_failure
                    try:
                        record_unwind_failure(original, cleanup)
                    except BaseException:
                        if post_group is not None:
                            from clay.layout.route.group_endpoint_observation import record_observation_unwind
                            record_observation_unwind(original, cleanup)
    _check_source_engine_before_group_fallback(route_engine, output_profile_engine_record)
    # Carry required clearances through the final GDS write boundary, regardless
    # of routing strategy. The emitter checks its own final conductor geometry.
    routed.spacing_constraints = tuple(c for c in cons if isinstance(c, _C.Spacing))
    if group_route_contract is not None and intent_fact_set is not None:
        from clay.layout.route.group_ports import (
            bind_group_route_routing_fact_projection,
            build_group_route_port_inventory,
        )

        landing_certificate = getattr(
            routed,
            "group_route_landing_certificate",
            None,
        )
        topology = build_group_route_port_inventory(
            cell_def,
            source_intent,
            fact_set=intent_fact_set,
            source_inventory=source_inventory,
        )
        routed.group_route_landing_certificate = (
            bind_group_route_routing_fact_projection(
                inventory=topology,
                contract=group_route_contract,
                certificate=landing_certificate,
                fact_set=intent_fact_set,
                source_inventory=source_inventory,
            )
        )
    if execution_scope is RouteExecutionScope.BRINGUP_DIAGNOSTIC:
        selected_nets = set(engine_pins_by_net)
        _restore_diagnostic_open_nets(
            routed,
            pins_by_net,
            selected_nets,
            set(
                ()
                if diagnostic_compile is None
                else diagnostic_compile.unrouted_nets
            ),
        )
        routed.allocation_replay_verified = False
        routed.success = False
        routed.diagnostic_only = True
        routed.diagnostic_summary_digest = "pending-route-cell-summary"

    # 5b. VIA-PAD LEGALIZATION: DRC-by-pitch covers WIRE spacing but not the wider via
    #     landing pad, so a foreign net packs to the wire-class clearance from a pad and
    #     fires the foundry metal min-space (cross-net via-pad M2.2a). Trim no-cut widen
    #     overhangs + relocate real via stacks along their own trunk until each pad holds
    #     the PDK min_space from foreign metal -- general, net-identity + PDK-rule driven,
    #     each move kept on the net's own trunk (no open) + clear of foreign metal (no
    #     short). A pad with no admissible position is left UNCHANGED (honest residual --
    #     the mirror-pinned diff-pair via2 pad, a placement/track-assignment obligation).
    #     Runs BEFORE the short oracle so the canonical the checkers read is legalized.
    # Gridless always enters this strict branch, independent of over-cell planning.
    if verified_resource_plan is not None:
        # Exact selected geometry is already replay-certified.  Mutating it in a
        # legacy post-route legalizer would invalidate the allocation certificate.
        pass
    elif getattr(route_engine, "strict_device_m1_provenance", False):
        legalize_via_pads(
            routed, tech, device_obstacles=obstacles,
            enforce_device_m1_provenance=True)
    else:
        legalize_via_pads(routed, tech)

    # 5c. ASSEMBLED-METAL OWNERSHIP: Gridless' internal rectangle world can
    # prove its segment/via junctions but the final placed stamp + exact pin
    # component reconciliation lives at this route-cell seam.  Commit the
    # independent PDK-rule-derived terminal/EOL and assembled-metal closure to
    # the owning NetRoute BEFORE the
    # containment/short/manifest observers.  An exact resource plan is already
    # immutable and must have allocated its own complete seam geometry; mutating
    # that replay here would invalidate its certificate.
    if (
        verified_resource_plan is None
        and getattr(route_engine, "materializes_terminal_interface_m1", False)
    ):
        from clay.layout.route.owned_metal_finalization import (
            materialize_router_owned_metal_finalization,
        )

        materialize_router_owned_metal_finalization(
            routed,
            geometry,
            placement_um,
            pins_by_net,
            tech,
        )

    # A boundary-capable engine checks complete candidates before selection.
    # Re-read its exact polygons after the only geometry-mutating post-pass as an
    # independent emission gate; clipping is forbidden because it can create an
    # electrical open.  Historical engines receive the same observation as
    # report-only evidence until they explicitly advertise this hard capability.
    from clay.layout.route.floorplan_containment import (
        check_routed_containment,
        require_routed_containment,
    )

    if getattr(route_engine, "consumes_routing_envelope", False):
        route_containment_report = require_routed_containment(
            routed, routing_envelope
        )
    else:
        route_containment_report = check_routed_containment(
            routed, routing_envelope
        )
    routed.routing_envelope_nm = routing_envelope.bbox_nm
    routed.routing_envelope_certificate = routing_envelope_certificate

    # Bind the exact post-legalizer geometry handed to emit.  Negotiated routes do
    # not mint allocation/materialization certificates; this digest is only a
    # deterministic diagnostic receipt and carries no signoff authority.
    if use_bringup_negotiation:
        routed.geometry_digest = _routed_geometry_digest(routed)

    # 6. INDEPENDENT union-find short oracle over the routed metal + device pads.
    #    Cap names let the oracle attribute the two stacked MIM plates by terminal
    #    access layer (distinct-plate physics) instead of a bbox-contains-pin rule.
    cap_names = {
        str(getattr(p, "name", ""))
        for p in getattr(bridged_intent, "passives", []) or []
        if str(getattr(p, "kind", "")).lower().startswith("cap")
    }
    shorts = check_cross_net_shorts(
        routed, obstacles, tech,
        geometry=geometry, placement=placement_um, pins_by_net=pins_by_net,
        cap_names=cap_names,
    )

    # A3 POST-gate (capability handshake, AFTER route): SURFACE the HARD constraints
    # the engine could not satisfy on this route (``routed.unsatisfied`` filtered to
    # HARD -- never a per-kind branch). The PRE-gate already stopped unsupported
    # KINDS before routing, so anything here is a SUPPORTED kind the engine could not
    # materialize on THIS instance (a blocked exact Mirror, an unmaterialized Width).
    # We surface it on the result so a signoff verdict never rests on net_complete
    # alone; enforcing a STOP on these instance residuals is a later, separately
    # audited flip (per the migration plan) -- doing it now would collapse an
    # LVS-green cell whose only residual is e.g. an inexact-but-legal Mirror.
    required_unsat = [
        c for c in (getattr(routed, "unsatisfied", []) or []) if _C.is_hard(c)
    ]
    if execution_scope is RouteExecutionScope.BRINGUP_DIAGNOSTIC:
        required_unsat.extend(
            DiagnosticSuppressedHardConstraint(
                constraint_ref=_diagnostic_constraint_ref(constraint),
                source_kind=str(getattr(constraint, "kind", type(constraint).__name__)),
                reason=(
                    "NOT_EVALUATED_UNSUPPORTED_BY_NEGOTIATION_BRINGUP"
                    if use_bringup_negotiation
                    else "NOT_EVALUATED_REFERENCES_OMITTED_DIAGNOSTIC_NET"
                ),
            )
            for constraint in suppressed_hard_constraints
        )

    rigid_nonmos_emission_certificate = None
    if (
        final_row_placement.rigid_nonmos_macro_interfaces is not None
        and route_power
    ):
        from clay.layout.nonrow_macro_plan import (
            build_rigid_nonmos_emission_certificate,
        )

        rigid_nonmos_emission_certificate = (
            build_rigid_nonmos_emission_certificate(
                final_row_placement.rigid_nonmos_macro_plans,
                final_row_placement.rigid_nonmos_macro_interfaces,
                final_row_placement.rigid_nonmos_placement_certificate,
                scope_token=final_row_placement.physical_macro_scope_token,
                compile_inputs=final_row_placement.rigid_nonmos_macro_inputs,
                geometry=geometry,
                placement_um=placement_um,
                pins_by_net=pins_by_net,
            )
        )

    common_result_fields = dict(
        routed=routed,
        pins_by_net=pins_by_net,
        obstacles=obstacles,
        geometry=geometry,
        placement_um=placement_um,
        constraints=cons,
        tech=tech,
        shorts=shorts,
        selected_route_engine=output_profile_engine_record,
        uncovered_terminals=list(getattr(pin_report, "uncovered", []) or []),
        axis_x_um=axis_x_um,
        devices=tuple(getattr(bridged_intent, "devices", []) or []),
        passives=tuple(getattr(bridged_intent, "passives", []) or []),
        guard_rings=guard_ring_intents,
        shared_well_guard_envelope_plans=shared_well_guard_envelope_plans,
        shared_well_guard_ring_inventory=shared_well_guard_ring_inventory,
        guard_ring_member_classes=tuple(sorted(
            [
                (str(getattr(device, "name", "")), "mos")
                for device in tuple(getattr(bridged_intent, "devices", ()) or ())
            ]
            + [
                (str(getattr(passive, "name", "")), "passive")
                for passive in tuple(getattr(bridged_intent, "passives", ()) or ())
            ]
        )),
        member_ring_pad_top_nm=tuple(sorted(ring_pad_rows)),
        member_ring_pad_bottom_nm=tuple(sorted(ring_bottom_pad_rows)),
        required_unsatisfied=required_unsat,
        routing_envelope_nm=routing_envelope.bbox_nm,
        routing_envelope_certificate=routing_envelope_certificate,
        route_containment_report=route_containment_report,
        resource_evidence=resource_evidence,
        topology_plan=topology_plan,
        routing_world_provenance=routing_world_provenance,
        rail_candidate_pool=rail_candidate_pool,
        rail_placement_plan=rail_placement_plan,
        materialized_rail_certificate=materialized_rail_certificate,
        power_plane_facts=power_plane_facts,
        row_program=final_row_placement.row_program,
        row_materialization=final_row_placement.row_materialization,
        row_final_placement_certificate=final_row_placement.row_final_certificate,
        row_final_placement=final_row_placement,
        final_placement_certificate=final_placement_certificate,
        exact_structure_expectation=exact_structure_expectation,
        physical_macro_scope_token=final_row_placement.physical_macro_scope_token,
        rigid_nonmos_macro_plans=final_row_placement.rigid_nonmos_macro_plans,
        rigid_nonmos_macro_inputs=final_row_placement.rigid_nonmos_macro_inputs,
        rigid_nonmos_macro_interfaces=(
            final_row_placement.rigid_nonmos_macro_interfaces
        ),
        rigid_nonmos_placement_certificate=(
            final_row_placement.rigid_nonmos_placement_certificate
        ),
        rigid_nonmos_emission_certificate=rigid_nonmos_emission_certificate,
    )
    if execution_scope is RouteExecutionScope.SIGNOFF:
        result = RouteCellResult(
            **common_result_fields,
            candidate_pool=verified_candidate_pool,
            resource_plan=verified_resource_plan,
        )
    else:
        result = _build_bringup_diagnostic_route_result(
            common_result_fields=common_result_fields,
            routed=routed,
            diagnostic_compile=diagnostic_compile,
            use_bringup_negotiation=use_bringup_negotiation,
            suppressed_hard_constraints=suppressed_hard_constraints,
        )

    # Required topology goals remain named alongside existing hard-constraint
    # residuals.  This native connectivity result is a blocking construction
    # obligation, not an LVS claim; full-extraction LVS remains authoritative.
    result.required_unsatisfied.extend(_unsatisfied_required_route_goals(result))

    if require_complete and not result.net_complete:
        raise NetIncompleteError(
            f"route_cell is NOT net-complete: "
            f"{result.n_connected}/{result.n_required} nets connected, "
            f"unconnected={result.unconnected_nets}, "
            f"cross-net shorts={result.merged_pairs}, "
            f"required-unsatisfied={result.required_unsatisfied}"
        )
    return result
