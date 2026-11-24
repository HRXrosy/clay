"""Edge-rail AST/demand compilation, rail placement planning, and resource-plan compilation."""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from clay.core_contract import VerificationStatus
from clay.layout.placed_geometry import placed_point_nm
from clay.layout.route import constraints as _C
from clay.layout.route.engine import Obstacle
from clay.layout.route.lower import conventional_axes_for
from clay.layout.route.rc_devices import (
    RouteExecutionScope,
    RoutePlanningStop,
    _BringupRailConstructionEvidence,
    _DiagnosticRouteCompilerProduct,
    _NM_PER_UM,
    _XY,
    _cell_bbox_um,
    _declared_ground_ports,
    _metal_index_of,
    _rc_attr,
)
from clay.layout.route.rc_placement_evidence import FinalPlacementBindingError
from clay.layout.tech import LayoutTech


@dataclass(frozen=True)
class _CompiledEdgeRailSpec:
    declaration_ref: str
    demand_id: str
    net: str
    layer_n: int
    layer_key: Tuple[int, int]
    actual_axis: str
    axis_source: str
    current_preferred_axis: Optional[str]
    edge: str
    width_nm: int
    width_source: str
    declared_width_um: float
    offset_nm: int = 0

    @property
    def semantic_declaration_key(self) -> tuple[object, ...]:
        return (
            self.net,
            self.layer_n,
            self.actual_axis if self.axis_source == "explicit_intent" else "",
            self.declared_width_um,
            self.edge,
            self.offset_nm,
        )


def _compile_edge_rail_specs_from_intent(
    intent: Any,
    routing_intent: Any,
    tech: LayoutTech,
) -> Tuple[_CompiledEdgeRailSpec, ...]:
    """Lower exact edge geometry without minting an authorization capability."""

    widths = getattr(routing_intent, "widths", {}) or {}
    rows: List[Tuple[str, int, Any]] = []
    for trunk in tuple(getattr(intent, "trunks", ()) or ()):
        edge = str(getattr(trunk, "edge", "") or "").strip().lower()
        net = str(getattr(trunk, "net", "") or "")
        if not edge:
            if float(getattr(trunk, "offset_um", 0.0) or 0.0) != 0:
                raise ValueError("rail offset requires an edge declaration")
            continue
        layer_n = _metal_index_of(str(getattr(trunk, "layer", "") or ""), tech)
        if not net or layer_n is None:
            raise ValueError("edge rail requires exact net and PDK metal layer")
        try:
            layer_key = tuple(int(v) for v in tech.metal_key(layer_n))
        except Exception as exc:
            raise ValueError("edge rail PDK layer facts are unavailable") from exc
        declared_axis = str(getattr(trunk, "orient", "") or "").strip().upper()
        if declared_axis and declared_axis not in ("H", "V"):
            raise ValueError(
                f"edge rail {net!r} orientation must be 'H' or 'V'"
            )
        # An explicit @trunk orientation is design intent and supplies the actual
        # rail axis even when it is nonpreferred in the current foundry stack.
        # Without that explicit authority, use the PDK-aware shared default; the
        # rail verifier independently recomputes both the current preference and
        # their typed relation before planning, replay, or materialization.
        actual_axis = declared_axis or conventional_axes_for(
            tech, n_max=max(10, layer_n)
        )[layer_n]
        axis_source = "explicit_intent" if declared_axis else "pdk_default"
        try:
            raw_current_preferred_axis = tech.metal_preferred_axis_declared(layer_n)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError("edge rail PDK preferred-axis facts are unavailable") from exc
        current_preferred_axis = (
            None
            if raw_current_preferred_axis is None
            else str(raw_current_preferred_axis).upper()
        )
        if current_preferred_axis not in (None, "H", "V"):
            raise ValueError("edge rail PDK preferred axis must be H, V, or absent")
        edge_width_um = float(getattr(trunk, "width_um", 0.0) or 0.0)
        offset_um = float(getattr(trunk, "offset_um", 0.0) or 0.0)
        if not math.isfinite(offset_um) or offset_um < 0 or abs(offset_um * _NM_PER_UM - round(offset_um * _NM_PER_UM)) > 1e-7:
            raise ValueError("edge rail offset must be nonnegative whole nanometres")
        offset_nm = round(offset_um * _NM_PER_UM)
        layer_floor_nm = int(tech.min_width(f"metal{layer_n}") or 0)
        if edge_width_um > 0:
            raw_width_nm = round(edge_width_um * _NM_PER_UM)
            width_nm = max(
                raw_width_nm,
                layer_floor_nm,
            )
            width_source = json.dumps({
                "kind": "declared_edge_trunk",
                "pdk_min_width_nm": layer_floor_nm,
                "raw_width_nm": raw_width_nm,
            }, sort_keys=True, separators=(",", ":"))
        else:
            declared = widths.get(net)
            value_um = getattr(declared, "value", declared)
            try:
                raw_width_nm = round(float(value_um) * _NM_PER_UM)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"edge rail {net!r} has no unified routing width"
                ) from exc
            width_nm = max(raw_width_nm, layer_floor_nm)
            width_source = json.dumps({
                "kind": "routing_intent_fallback",
                "pdk_min_width_nm": layer_floor_nm,
                "raw_width_nm": raw_width_nm,
            }, sort_keys=True, separators=(",", ":"))
        if width_nm <= 0:
            raise ValueError(f"edge rail {net!r} width must be positive")
        descriptor = json.dumps(
            {
                "cell": str(getattr(intent, "cell", "") or ""),
                "edge": edge,
                "layer_key": layer_key,
                "layer_n": layer_n,
                "net": net,
                "actual_axis": actual_axis,
                "axis_source": axis_source,
                "current_preferred_axis": current_preferred_axis,
                "width_nm": width_nm,
                "width_source": width_source,
                **({"offset_nm": offset_nm} if offset_nm else {}),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        rows.append((descriptor, len(rows), (
            net,
            layer_n,
            layer_key,
            actual_axis,
            axis_source,
            current_preferred_axis,
            edge,
            width_nm,
            width_source,
            edge_width_um,
            offset_nm,
        )))
    occurrence: Dict[str, int] = {}
    specs = []
    for descriptor, _source_order, values in sorted(rows):
        ordinal = occurrence.get(descriptor, 0)
        occurrence[descriptor] = ordinal + 1
        (
            net,
            layer_n,
            layer_key,
            actual_axis,
            axis_source,
            current_preferred_axis,
            edge,
            width_nm,
            width_source,
            declared_width_um,
            offset_nm,
        ) = values
        declaration_ref = f"edge_trunk:{descriptor}|occurrence={ordinal:03d}"
        demand_id = "rail-demand:" + hashlib.sha256(
            declaration_ref.encode("utf-8")
        ).hexdigest()
        specs.append(_CompiledEdgeRailSpec(
            declaration_ref=declaration_ref,
            demand_id=demand_id,
            net=net,
            layer_n=layer_n,
            layer_key=layer_key,
            actual_axis=actual_axis,
            axis_source=axis_source,
            current_preferred_axis=current_preferred_axis,
            edge=edge,
            width_nm=width_nm,
            width_source=width_source,
            declared_width_um=declared_width_um,
            offset_nm=offset_nm,
        ))
    return tuple(specs)


def _canonical_rail_ast_value(value: Any) -> Any:
    if is_dataclass(value):
        return {
            item.name: _canonical_rail_ast_value(getattr(value, item.name))
            for item in fields(value)
        }


    if isinstance(value, dict):
        return {
            str(key): _canonical_rail_ast_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_canonical_rail_ast_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"rail source AST contains non-canonical value {value!r}")


def _group_route_contract_for_engine(
    cell_def: Any,
    intent: Any,
    final_placement_certificate: Any,
    pins_by_net: object,
    route_engine: Any,
    *,
    intent_fact_set: Any = None,
    source_inventory: Any = None,
) -> Any:
    """Compile the exact route-group handoff or stop before route emission."""

    raw_declarations = getattr(intent, "route_groups", None)
    if type(raw_declarations) is not tuple:
        raise TypeError("route-group route boundary requires an exact tuple")
    if not raw_declarations:
        return None
    from clay.layout.route.group_ports import (
        GroupRoutePlacementCertificate,
        build_group_route_port_inventory,
        build_group_route_routing_contract,
        verify_group_route_placement,
    )

    topology = build_group_route_port_inventory(
        cell_def,
        intent,
        fact_set=intent_fact_set,
        source_inventory=source_inventory,
    )
    placed = getattr(
        final_placement_certificate,
        "group_route_placement_certificate",
        None,
    )
    if type(placed) is not GroupRoutePlacementCertificate:
        raise RoutePlanningStop(
            phase="GROUP_ROUTE_PORTS",
            state="PLACEMENT_CERTIFICATE_MISSING",
            evidence=topology,
            status=VerificationStatus.FAIL,
        )
    verify_group_route_placement(topology, final_placement_certificate, placed)
    contract = build_group_route_routing_contract(
        topology,
        placed,
        pins_by_net,
    )
    if not bool(getattr(route_engine, "consumes_group_route_ports", False)):
        raise RoutePlanningStop(
            phase="GROUP_ROUTE_PORTS",
            state="ENGINE_UNSUPPORTED",
            evidence=contract,
            status=VerificationStatus.UNSUPPORTED,
        )
    return contract


def _canonical_rail_declaration_payload(declaration: Any) -> dict[str, Any]:
    payload = _canonical_rail_ast_value(declaration)
    if not isinstance(payload, dict):
        raise TypeError("parsed rail declaration must lower to a canonical object")
    span = payload.get("source_span")
    if isinstance(span, dict):
        # Absolute checkout paths are environment data, not source semantics.
        payload["source_span"] = {
            key: span[key]
            for key in ("line", "column", "end_line", "end_column")
            if key in span
        }
    payload["schema"] = "clay.parsed-layout-intent-declaration.v1"
    return payload


def _canonical_rail_source_ast(
    cell_def: Any,
) -> tuple[str, tuple[tuple[Any, dict[str, Any]], ...], str]:
    from clay.ir import CellDef, ParsedLayoutIntentDecl

    if type(cell_def) is not CellDef:
        raise TypeError("explicit rail authority requires an exact parsed CellDef")
    declarations = tuple(getattr(cell_def, "layout_intents", ()) or ())
    if not declarations or not all(
        type(item) is ParsedLayoutIntentDecl for item in declarations
    ):
        raise ValueError(
            "explicit rail authority requires parsed layout-intent source AST"
        )
    rows = tuple(
        (declaration, _canonical_rail_declaration_payload(declaration))
        for declaration in declarations
    )
    source_files = {
        str((getattr(item, "source_span", None) or {}).get("file") or "")
        for item in declarations
    }
    source_files.discard("")
    if len(source_files) != 1:
        raise ValueError(
            "explicit rail authority requires one real source file locator and bytes"
        )
    source_file = os.path.realpath(os.path.abspath(next(iter(source_files))))
    if not os.path.isfile(source_file):
        raise ValueError("parsed rail source file is unavailable for receipt")
    with open(source_file, "rb") as handle:
        source_bytes = handle.read()
    source_bytes_digest = hashlib.sha256(source_bytes).hexdigest()
    if len(source_files) == 1:
        # Bind the byte digest to the AST actually authorizing the rail.  A
        # caller holding an old CellDef after the source changed cannot combine
        # stale declarations with the new file digest.
        from clay.parser import parse_clay_file

        reparsed = parse_clay_file(source_file)
        matches = [
            item
            for item in reparsed.cell_defs
            if str(getattr(item, "local_name", "") or "")
            == str(getattr(cell_def, "local_name", "") or "")
        ]
        if len(matches) != 1:
            raise ValueError("rail source bytes do not resolve one authoritative cell")
        reparsed_payloads = [
            _canonical_rail_declaration_payload(item)
            for item in tuple(getattr(matches[0], "layout_intents", ()) or ())
        ]
        if reparsed_payloads != [payload for _item, payload in rows]:
            raise ValueError("rail source bytes and parsed source AST disagree")
    source_payload = {
        "canonical_source_bytes_digest": source_bytes_digest,
        "cell_name": str(getattr(cell_def, "local_name", "") or ""),
        "layout_intents": [payload for _declaration, payload in rows],
        "schema": "clay.parsed-layout-intent-source.v2",
    }
    return (
        json.dumps(source_payload, sort_keys=True, separators=(",", ":")),
        rows,
        source_file,
    )


def _parsed_edge_rail_receipts(
    cell_def: Any,
    specs: Tuple[_CompiledEdgeRailSpec, ...],
    tech: LayoutTech,
) -> dict[str, Any]:
    _canonical_rail_source_ast = _rc_attr("_canonical_rail_source_ast", globals()["_canonical_rail_source_ast"])
    from clay.layout.route.rail_plan import _issue_rail_axis_declaration_receipt

    canonical_source_ast_json, declarations, source_locator = (
        _canonical_rail_source_ast(cell_def)
    )
    source_bytes_digest = json.loads(canonical_source_ast_json)[
        "canonical_source_bytes_digest"
    ]
    by_semantics: Dict[tuple[object, ...], List[dict[str, Any]]] = {}
    for declaration, payload in declarations:
        if str(getattr(declaration, "kind", "") or "") != "trunk":
            continue
        attributes = getattr(declaration, "attributes", {}) or {}
        edge = str(attributes.get("edge") or "").strip().lower()
        if not edge:
            continue
        net = str(attributes.get("net") or "")
        layer_n = _metal_index_of(str(attributes.get("layer") or ""), tech)
        orient = str(attributes.get("orient") or "").strip().upper()
        try:
            width_um = float(attributes.get("width") or 0.0)
        except (TypeError, ValueError) as exc:
            raise ValueError("parsed edge-rail width is not numeric") from exc
        offset_nm = round(float(attributes.get("offset") or 0.0) * _NM_PER_UM)
        key = (net, layer_n, orient, width_um, edge, offset_nm)
        by_semantics.setdefault(key, []).append(payload)

    receipts: dict[str, Any] = {}
    for spec in specs:
        if spec.axis_source != "explicit_intent":
            continue
        matches = by_semantics.get(spec.semantic_declaration_key, [])
        if not matches:
            raise ValueError(
                "edge rail explicit intent is absent from parsed source/AST"
            )
        declaration_payload = matches.pop(0)
        receipts[spec.declaration_ref] = _issue_rail_axis_declaration_receipt(
            cell_name=str(getattr(cell_def, "local_name", "") or ""),
            declaration_ref=spec.declaration_ref,
            source_locator=source_locator,
            source_bytes_digest=source_bytes_digest,
            canonical_source_ast_json=canonical_source_ast_json,
            canonical_declaration_ast_json=json.dumps(
                declaration_payload, sort_keys=True, separators=(",", ":")
            ),
        )
    return receipts


def _compile_edge_rail_demands_from_intent(
    intent: Any,
    routing_intent: Any,
    tech: LayoutTech,
    *,
    cell_def: Any = None,
) -> Tuple[Any, ...]:
    """Issue exact edge demands only after matching explicit axes to parsed AST."""
    from clay.layout.route.rail_plan import (
        EdgeRailDemand,
        _issue_rail_axis_authority,
        compile_edge_rail_demands,
        make_rail_axis_authority,
    )

    specs = _compile_edge_rail_specs_from_intent(intent, routing_intent, tech)
    explicit = tuple(spec for spec in specs if spec.axis_source == "explicit_intent")
    receipts = (
        _parsed_edge_rail_receipts(cell_def, explicit, tech) if explicit else {}
    )
    demands = []
    for spec in specs:
        if spec.axis_source == "explicit_intent":
            axis_authority = _issue_rail_axis_authority(
                actual_axis=spec.actual_axis,
                axis_source=spec.axis_source,
                current_preferred_axis=spec.current_preferred_axis,
                declaration_ref=spec.declaration_ref,
                source_receipt=receipts[spec.declaration_ref],
            )
        else:
            axis_authority = make_rail_axis_authority(
                actual_axis=spec.actual_axis,
                axis_source=spec.axis_source,
                current_preferred_axis=spec.current_preferred_axis,
                declaration_ref=spec.declaration_ref,
            )
        demands.append(EdgeRailDemand(
            demand_id=spec.demand_id,
            declaration_ref=spec.declaration_ref,
            net=spec.net,
            layer_n=spec.layer_n,
            layer_key=spec.layer_key,
            axis_authority=axis_authority,
            edge=spec.edge,
            width_nm=spec.width_nm,
            width_source=spec.width_source,
            offset_nm=spec.offset_nm,
        ))
    return compile_edge_rail_demands(tuple(demands))


def _build_pre_rail_scene(
    *,
    placement_um: Dict[str, _XY],
    geometry: Dict[str, Any],
    pins_by_net: Dict[str, List[Any]],
    layout_model: Any,
    placement_provenance_digest: str,
    demands: Tuple[Any, ...],
    tech: LayoutTech,
    final_placement_certificate: Any = None,
    declared_floorplan_nm: Any = None,
) -> Any:
    from clay.layout.model import device_active_projection_conductors
    from clay.layout.route.emit_gds import (
        _nearest_on_grid,
        pdk_manufacturing_grid_nm,
    )
    from clay.layout.route.rail_plan import (
        RailPlacementScene,
        RailProjectionFact,
        RailTerminalFact,
    )
    from clay.layout.route.resource_world import pdk_provenance_digest

    bbox = _cell_bbox_um(placement_um, geometry)
    if bbox is None:
        raise ValueError("edge rail planning requires a non-empty final placement")
    # The rail run endpoints are emitted geometry, so their scene envelope must
    # be the SAME nearest-grid projection applied by the final GDS writer.  The
    # compact placer works in integer nanometres but device/passive dimensions
    # may legitimately contribute a non-manufacturing-grid extreme (for
    # example, an area-derived MIM side).  Requiring that raw analytic extent to
    # be on-grid either rejects a valid placement or tempts a later hidden snap.
    # Project the bbox here, explicitly and deterministically, before it becomes
    # content-addressed rail-planning evidence.
    grid_nm = pdk_manufacturing_grid_nm(tech)
    cell_bbox_nm = tuple(
        _nearest_on_grid(round(value * _NM_PER_UM), grid_nm) for value in bbox
    )
    if declared_floorplan_nm is not None:
        from clay.layout.route.floorplan_containment import RoutingEnvelope
        declared = RoutingEnvelope.coerce(declared_floorplan_nm).bbox_nm
        if (declared[0] > cell_bbox_nm[0] or declared[1] > cell_bbox_nm[1]
                or declared[2] < cell_bbox_nm[2] or declared[3] < cell_bbox_nm[3]):
            raise FinalPlacementBindingError("declared row floorplan excludes actual stamps")
        cell_bbox_nm = declared
    if final_placement_certificate is not None:
        raw_certified_bbox = getattr(
            final_placement_certificate, "bbox_nm", None
        )
        if (
            type(raw_certified_bbox) is not tuple
            or len(raw_certified_bbox) != 4
            or any(type(value) is not int for value in raw_certified_bbox)
            or raw_certified_bbox[0] >= raw_certified_bbox[2]
            or raw_certified_bbox[1] >= raw_certified_bbox[3]
        ):
            raise FinalPlacementBindingError(
                "pre-rail scene requires one exact certified floorplan bbox"
            )
        certified_bbox_nm = tuple(
            _nearest_on_grid(value, grid_nm) for value in raw_certified_bbox
        )
        if declared_floorplan_nm is not None and certified_bbox_nm != cell_bbox_nm:
            raise FinalPlacementBindingError("declared row and certified floorplans disagree")
        if not (
            certified_bbox_nm[0] <= cell_bbox_nm[0]
            and certified_bbox_nm[1] <= cell_bbox_nm[1]
            and cell_bbox_nm[2] <= certified_bbox_nm[2]
            and cell_bbox_nm[3] <= certified_bbox_nm[3]
        ):
            raise FinalPlacementBindingError(
                "pre-rail geometry leaves the certified floorplan bbox"
            )
        cell_bbox_nm = certified_bbox_nm
    declaration_digest = hashlib.sha256(json.dumps(
        [
            {
                "declaration_ref": item.declaration_ref,
                "axis_authority": {
                    "actual_axis": item.axis_authority.actual_axis,
                    "axis_source": item.axis_authority.axis_source,
                    "current_preferred_axis": (
                        item.axis_authority.current_preferred_axis
                    ),
                    "declaration_digest": (
                        item.axis_authority.declaration_digest
                    ),
                    "declaration_ref": item.axis_authority.declaration_ref,
                    "relation": item.axis_authority.relation,
                },
                "demand_id": item.demand_id,
                "edge": item.edge,
                "layer_key": list(item.layer_key),
                "net": item.net,
                "width_nm": item.width_nm,
            }
            for item in demands
        ],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    active = tuple(RailProjectionFact(
        bbox_nm=tuple(round(v * _NM_PER_UM) for v in (c.x0, c.y0, c.x1, c.y1)),
        layer_key=tuple(c.layer),
        owner_net="",
        kind="device_projection",
    ) for c in device_active_projection_conductors(
        placement_um, geometry, tech=tech
    ))
    # Rail placement is a same-layer spacing problem.  Keep its closed scene
    # exact but scoped: via cuts and conductors on layers no declared rail can
    # occupy cannot block a candidate, and the enumerator already compares
    # conductor facts only when ``fact.layer_key == demand.layer_key``.  Filter
    # before constructing the bounded scene so dense device via arrays do not
    # consume proof capacity with semantically irrelevant facts.
    rail_layer_keys = {
        tuple(int(v) for v in demand.layer_key) for demand in demands
    }
    conductors = tuple(RailProjectionFact(
        bbox_nm=tuple(round(v * _NM_PER_UM) for v in (c.x0, c.y0, c.x1, c.y1)),
        layer_key=tuple(c.layer),
        owner_net=None if c.net is None else str(c.net),
        kind=str(c.kind or "conductor"),
    ) for c in tuple(getattr(layout_model, "conductors", ()) or ())
        if str(getattr(c, "kind", "")) != "device_projection"
        and tuple(int(v) for v in c.layer) in rail_layer_keys)
    from clay.layout.route.terminal_identity import pin_provenance

    terminals = tuple(RailTerminalFact(
        net=str(net),
        xy_nm=tuple(round(float(v) * _NM_PER_UM) for v in pin.xy_um),
        layer_key=tuple(pin.layer),
        provenance=pin_provenance(pin),
    ) for net, pins in sorted(pins_by_net.items()) for pin in pins)
    return RailPlacementScene(
        placement_provenance_digest=placement_provenance_digest,
        pdk_provenance_digest=pdk_provenance_digest(tech),
        declaration_provenance_digest=declaration_digest,
        cell_bbox_nm=cell_bbox_nm,
        active_projection_facts=active,
        conductor_facts=conductors,
        terminal_facts=terminals,
    )


def _plan_rails(
    scene: Any, demands: Tuple[Any, ...], tech: LayoutTech
) -> Tuple[Any, Any, Any]:
    from clay.layout.route.rail_plan import (
        certify_rail_placement_plan_replay,
        enumerate_rail_placement_candidates,
        select_rail_placement_plan,
        verify_rail_placement_candidate_pool,
        verify_rail_placement_infeasibility,
    )

    pool = enumerate_rail_placement_candidates(scene, demands, tech)
    verify_rail_placement_candidate_pool(scene, demands, tech, pool)
    result = select_rail_placement_plan(pool)
    if result.status == "UNKNOWN":
        from clay.layout.route.rail_diagnostics import observe_rail_placement_stop

        raise RoutePlanningStop(
            phase="RAIL_PLACEMENT",
            state=result.solver_status,
            evidence=result,
            rail_observation=observe_rail_placement_stop(
                scene=scene, pool=pool, result=result, tech=tech
            ),
        )
    if result.status == "INFEASIBLE":
        verify_rail_placement_infeasibility(pool, result.certificate)
        from clay.layout.route.rail_diagnostics import observe_rail_placement_stop

        raise RoutePlanningStop(
            phase="RAIL_PLACEMENT",
            state="INFEASIBLE",
            evidence=result.certificate,
            status=VerificationStatus.FAIL,
            rail_observation=observe_rail_placement_stop(
                scene=scene, pool=pool, result=result, tech=tech
            ),
        )
    replay_receipt = certify_rail_placement_plan_replay(
        pool, result.plan, tech=tech
    )
    return pool, result.plan, replay_receipt


def _compile_route_resource_plan(
    topology_plan: Any,
    routing_world: Any,
    tech: LayoutTech,
    *,
    final_placement_certificate: Any = None,
    execution_scope: RouteExecutionScope = RouteExecutionScope.SIGNOFF,
) -> Tuple[Any, Any]:
    """Canonical post-world compile/select/replay seam for plan consumers."""
    _verify_route_resource_plan_or_stop = _rc_attr("_verify_route_resource_plan_or_stop", globals()["_verify_route_resource_plan_or_stop"])
    from clay.layout.route.pattern_compile import (
        DiagnosticDecomposedCompileResult,
        PatternCompileUnknown,
        PatternCompileScope,
        build_lane_incidence_index,
        compile_fixed_pattern_candidate_pool,
    )
    from clay.layout.route.resource_conductors import OwnedConductorComponentIndex
    from clay.layout.route.resource_plan import (
        CapacityConflict,
        HallDeficit,
        select_complete_net_candidates,
        verify_selection_certificate,
    )
    from clay.layout.route.resource_world import (
        enumerate_explicit_routing_resources,
        verify_candidate_pool_applicability,
        verify_explicit_routing_resource_ledger,
    )

    conductor_index = OwnedConductorComponentIndex.build_planar(
        routing_world, tech
    )
    ledger = enumerate_explicit_routing_resources(routing_world, tech)
    verify_explicit_routing_resource_ledger(ledger, routing_world, tech)
    incidence = build_lane_incidence_index(ledger, conductor_index, tech)
    if isinstance(incidence, PatternCompileUnknown):
        raise RoutePlanningStop(
            phase=incidence.phase,
            state=incidence.code,
            evidence=incidence,
        )
    compiled = compile_fixed_pattern_candidate_pool(
        topology_plan,
        routing_world,
        ledger,
        conductor_index,
        tech,
        incidence_index=incidence,
        scope=(
            PatternCompileScope.BRINGUP_DIAGNOSTIC
            if execution_scope is RouteExecutionScope.BRINGUP_DIAGNOSTIC
            else PatternCompileScope.SIGNOFF
        ),
    )
    if isinstance(compiled, DiagnosticDecomposedCompileResult):
        product = _DiagnosticRouteCompilerProduct(
            route_program=compiled.route_program,
            source_unknown=compiled.source_unknown,
            family_count=compiled.family_count,
            family_winner_count=compiled.family_winner_count,
            failed_family_refs=compiled.failed_family_refs,
            unrouted_nets=compiled.unrouted_nets,
        )
        return product, compiled.route_program
    if compiled.status == "UNKNOWN":
        raise RoutePlanningStop(
            phase=compiled.unknown.phase,
            state=compiled.unknown.code,
            evidence=compiled.unknown,
        )
    else:
        pool = compiled.candidate_pool
    if final_placement_certificate is not None:
        from clay.layout.place.resource_feedback import (
            attach_proven_landing_expansion_resources,
        )

        pool = attach_proven_landing_expansion_resources(
            pool, final_placement_certificate
        )
    verify_candidate_pool_applicability(pool, routing_world, tech=tech)
    selection = select_complete_net_candidates(pool)
    if selection.status == "UNKNOWN":
        raise RoutePlanningStop(
            phase="RESOURCE_SELECTION",
            state=selection.solver_status,
            evidence=selection,
        )
    if selection.status == "INFEASIBLE":
        verify_selection_certificate(pool, selection.certificate)
        feedback_context = None
        if (
            final_placement_certificate is not None
            and isinstance(selection.certificate, (CapacityConflict, HallDeficit))
        ):
            from clay.layout.place.resource_feedback import (
                RoutePlanningFeedbackContext,
            )

            feedback_context = RoutePlanningFeedbackContext(
                pool=pool,
                conflict=selection.certificate,
                routing_world=routing_world,
                final_placement_certificate=final_placement_certificate,
            )
        raise RoutePlanningStop(
            phase="RESOURCE_SELECTION",
            state="INFEASIBLE",
            evidence=selection.certificate,
            status=VerificationStatus.FAIL,
            feedback_context=feedback_context,
        )
    _verify_route_resource_plan_or_stop(
        pool,
        selection.plan,
        verification_receipt=getattr(selection, "verification_receipt", None),
    )
    verify_candidate_pool_applicability(pool, routing_world, tech=tech)
    if execution_scope is RouteExecutionScope.BRINGUP_DIAGNOSTIC:
        from dataclasses import replace

        from clay.layout.route.resource_plan import DiagnosticRouteProgramBundle

        plan = selection.plan
        frontier_digest = hashlib.sha256(json.dumps({
            "candidate_pool_hash": pool.sha256(),
            "route_plan_hash": plan.sha256(),
            "schema": "clay.complete-bringup-frontier.v1",
        }, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        index_certificate = plan.owned_conductor_index_certificate
        if index_certificate is None:
            raise RoutePlanningStop(
                phase="DIAGNOSTIC_MATERIALIZATION",
                state="OWNED_INDEX_REQUIRED",
                evidence=plan,
            )
        no_stop_digest = hashlib.sha256(
            b"clay.bringup-diagnostic.no-source-stop.v1"
        ).hexdigest()
        route_program = DiagnosticRouteProgramBundle(
            frontier_digest=frontier_digest,
            source_stop_digest=no_stop_digest,
            family_projection_digest=pool.sha256(),
            selected_nets=plan.selected_nets,
            allocated_programs=tuple(
                replace(program, candidate_pool_hash=frontier_digest)
                for program in plan.allocated_programs
            ),
            owned_conductor_index_certificate=index_certificate,
            source_witness_hashes=((
                "complete-global-frontier",
                pool.sha256(),
                plan.sha256(),
            ),),
            landing_site_conflict_domain_digest=(
                plan.landing_site_conflict_domain_digest
            ),
            lazy_landing_site_conflicts=plan.lazy_landing_site_conflicts,
        )
        family_count = (
            len(pool.candidate_families)
            + len(pool.factorized_candidate_families)
            or len(pool.candidates)
        )
        product = _DiagnosticRouteCompilerProduct(
            route_program=route_program,
            source_unknown=None,
            family_count=max(1, family_count),
            family_winner_count=len(plan.selected_nets),
            failed_family_refs=(),
            unrouted_nets=(),
        )
        return product, route_program
    return pool, selection.plan


def _verify_route_resource_plan_or_stop(
    pool: Any,
    plan: Any,
    *,
    verification_receipt: Any = None,
) -> None:
    """Translate an unproved objective replay into the canonical typed stop."""
    from clay.layout.route.resource_plan import (
        RouteResourcePlanVerificationBudgetExhausted,
        verify_route_plan_verification_receipt,
        verify_route_resource_plan,
    )

    try:
        if verification_receipt is None:
            # A directly supplied/external plan has no same-call-chain proof
            # transport and must independently complete the public deep replay.
            verify_route_resource_plan(pool, plan)
        else:
            # Selector-issued receipts are process-local, identity-registered,
            # and hash-bound.  Foreign or stale receipts fail closed here; they
            # never silently downgrade to an expensive or weaker fallback.
            verify_route_plan_verification_receipt(
                pool, plan, verification_receipt
            )
    except RouteResourcePlanVerificationBudgetExhausted as exc:
        raise RoutePlanningStop(
            phase="RESOURCE_PLAN_VERIFICATION",
            state=exc.state,
            evidence=exc,
        ) from exc
