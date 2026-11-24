"""Diagnostic constraint suppression, mirror/length lowering, trunk validation, and RoutingIntent synthesis."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from clay.core_contract import VerificationStatus
from clay.layout.placed_geometry import placed_point_nm
from clay.layout.route import constraints as _C
from clay.layout.route.engine import NetRoute
from clay.layout.route.lower import conventional_axes_for
from clay.layout.route.rc_devices import (
    RoutePlanningStop,
    _HVH_DIRECTION_WEIGHT,
    _NM_PER_UM,
    _XY,
    _metal_index_of,
)
from clay.layout.tech import LayoutTech


def _diagnostic_canonical_value(value: Any) -> Any:
    """Return an address-free JSON value for a closed route constraint."""

    if is_dataclass(value) and not isinstance(value, type):
        return {
            "fields": {
                item.name: _diagnostic_canonical_value(getattr(value, item.name))
                for item in fields(value)
            },
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
        }
    if isinstance(value, Enum):
        return _diagnostic_canonical_value(value.value)
    if isinstance(value, dict):
        return {
            str(key): _diagnostic_canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_diagnostic_canonical_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_diagnostic_canonical_value(item) for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(
                item, sort_keys=True, separators=(",", ":")
            ),
        )
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(
        "diagnostic constraint provenance requires a canonical dataclass value; "
        f"got {type(value).__module__}.{type(value).__qualname__}"
    )


def _diagnostic_constraint_json(constraint: Any) -> str:
    return json.dumps(
        _diagnostic_canonical_value(constraint),
        sort_keys=True,
        separators=(",", ":"),
    )


def _diagnostic_constraint_ref(constraint: Any) -> str:
    return hashlib.sha256(
        _diagnostic_constraint_json(constraint).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class DiagnosticSuppressedHardConstraint:
    """Explicit residual for HARD intent outside a bring-up route subset."""

    constraint_ref: str
    source_kind: str
    reason: str = "NOT_EVALUATED_REFERENCES_OMITTED_DIAGNOSTIC_NET"
    status: VerificationStatus = VerificationStatus.UNKNOWN
    hardness: str = _C.HARD


def _diagnostic_materialization_inputs(
    pins_by_net: Dict[str, List[Any]],
    constraints: Sequence[Any],
    route_program: Any,
) -> Tuple[Dict[str, List[Any]], List[Any], Tuple[Any, ...]]:
    """Project runtime inputs to the explicitly selected diagnostic subset."""

    selected_nets = {item.net for item in route_program.selected_nets}
    projected_pins = {
        net: net_pins
        for net, net_pins in pins_by_net.items()
        if net in selected_nets
    }

    def constraint_nets(constraint: Any) -> set[str]:
        refs = set()
        for name in ("net", "net_a", "net_b", "shield_net", "by"):
            value = getattr(constraint, name, None)
            if isinstance(value, str) and value:
                refs.add(value)
        values = getattr(constraint, "nets", ())
        if isinstance(values, (tuple, list, frozenset, set)):
            refs.update(
                value for value in values
                if isinstance(value, str) and value
            )
        return refs

    projected_constraints = []
    suppressed_hard = []
    for constraint in constraints:
        refs = constraint_nets(constraint)
        if not refs or refs <= selected_nets:
            projected_constraints.append(constraint)
        elif _C.is_hard(constraint):
            suppressed_hard.append(constraint)
    suppressed_hard.sort(key=_diagnostic_constraint_json)
    return projected_pins, projected_constraints, tuple(suppressed_hard)


def _diagnostic_negotiation_constraints(
    route_engine: Any,
    constraints: Sequence[Any],
) -> Tuple[List[Any], Tuple[Any, ...]]:
    """Project HARD intent to what the bring-up engine can actually evaluate.

    This projection is diagnostic-only.  Unsupported HARD declarations remain
    content-addressed residuals; they are never presented as satisfied and never
    enter a signoff result.  SOFT declarations remain advisory inputs.
    """

    supported = frozenset(route_engine.satisfies())
    forwarded: List[Any] = []
    suppressed: List[Any] = []
    for constraint in constraints:
        kind = str(getattr(constraint, "kind", None))
        if not _C.is_hard(constraint) or kind in supported:
            forwarded.append(constraint)
        else:
            suppressed.append(constraint)
    suppressed.sort(key=_diagnostic_constraint_json)
    return forwarded, tuple(suppressed)


def _canonical_polygon_vertices(vertices: Sequence[Any]) -> Tuple[Tuple[int, int], ...]:
    """Canonical cyclic/reversal representation of one emitted polygon ring."""

    ring = tuple((int(point.x), int(point.y)) for point in vertices)
    if not ring:
        return ()
    rotations = []
    for direction in (ring, tuple(reversed(ring))):
        rotations.extend(
            direction[index:] + direction[:index]
            for index in range(len(direction))
        )
    return min(rotations)


def _routed_geometry_digest(routed: Any) -> str:
    """Hash the canonical post-legalizer geometry received by emit.

    This is diagnostic provenance, not an LVS/DRC claim.  It deliberately binds
    connected state and exact integer polygon vertices while ignoring insertion
    order and polygon start-vertex choice.
    """

    records = []
    for net, net_route in sorted((getattr(routed, "nets", {}) or {}).items()):
        polygons = sorted(
            (
                int(wire.layer),
                int(wire.datatype),
                _canonical_polygon_vertices(wire.polygon.vertices),
            )
            for wire in (getattr(net_route, "wires", ()) or ())
        )
        records.append({
            "connected": bool(getattr(net_route, "connected", False)),
            "net": str(net),
            "polygons": polygons,
        })
    payload = {
        "nets": records,
        "schema": "clay.diagnostic-routed-geometry.v1",
    }
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def _restore_diagnostic_open_nets(
    routed: Any,
    pins_by_net: Dict[str, List[Any]],
    selected_nets: set[str],
    unrouted_required_nets: set[str],
) -> None:
    """Restore full-world nets with honest required/singleton connectivity."""

    for net, net_pins in sorted(pins_by_net.items()):
        if net in selected_nets:
            continue
        routed.nets[net] = NetRoute(
            net=net,
            pins=list(net_pins),
            wires=[],
            connected=net not in unrouted_required_nets and len(net_pins) <= 1,
        )


def _unsatisfied_required_route_goals(result: RouteCellResult) -> List[Any]:
    """Return named topology goals whose target/net/connectivity is incomplete."""

    plan = getattr(result, "topology_plan", None)
    goals = tuple(getattr(plan, "required_goals", ()) or ())
    if not goals:
        return []

    routed_nets = getattr(result.routed, "nets", {}) or {}
    failures: List[Any] = []
    connectivity_candidates: List[Any] = []
    for goal in goals:
        target = getattr(goal, "target", None)
        if target is None:
            failures.append(goal)
            continue
        if goal.net not in result.pins_by_net or goal.net not in routed_nets:
            failures.append(goal)
            continue
        try:
            demand = plan.by_net(goal.net)
        except KeyError:
            failures.append(goal)
            continue
        # Materializing a frame is insufficient: the access must also have been
        # injected into this exact net's final terminal set. Compare structural
        # point+layer facts, not pin/device/net spellings.
        if not any(
            terminal.xy_nm == target.xy_nm
            and terminal.layer == target.layer
            and terminal.provenance == target.provenance
            for terminal in demand.terminals
        ):
            failures.append(goal)
            continue
        goal_tap_sources = {
            other.target.provenance
            for other in demand.required_goals
            if other.target is not None
        }
        # A ring tap (or several ring taps) cannot prove that it is tied to the
        # circuit merely because the resulting one-source net is vacuously
        # connected. Require a distinct device/port/rail terminal source on the
        # same net; the connectivity oracle below then proves all of those pins
        # share one emitted-metal component with the frame access.
        if not any(
            terminal.provenance not in goal_tap_sources
            for terminal in demand.terminals
        ):
            failures.append(goal)
            continue
        connectivity_candidates.append(goal)

    if connectivity_candidates:
        unconnected = set(result.unconnected_nets)
        failures.extend(
            goal for goal in connectivity_candidates if goal.net in unconnected
        )
    return failures


def _exact_mirror_axis_um(
    primary_pins: Sequence[Any],
    partner_pins: Sequence[Any],
) -> Optional[float]:
    """Return the one exact vertical axis shared by two final pin sets.

    Compact placement can move rigid matched islands independently, so the
    generic placer's historical axis or the whole-cell bbox centre is not an
    electrical endpoint fact.  A hard route mirror is possible only when every
    final access on one net has a same-layer, same-y reflected partner.  Derive
    that axis directly and return ``None`` when no such exact bijection exists.
    """

    def grouped(pins: Sequence[Any]) -> Optional[Dict[Tuple[int, Tuple[int, int]], List[int]]]:
        out: Dict[Tuple[int, Tuple[int, int]], List[int]] = {}
        for pin in pins:
            xy_nm = getattr(pin, "xy_nm", None)
            if xy_nm is None:
                xy_um = getattr(pin, "xy_um", None)
                if xy_um is None:
                    return None
                try:
                    xy_nm = tuple(
                        int(round(float(value) * _NM_PER_UM)) for value in xy_um
                    )
                except (TypeError, ValueError):
                    return None
            try:
                x_nm, y_nm = int(xy_nm[0]), int(xy_nm[1])
                layer = tuple(int(value) for value in getattr(pin, "layer"))
            except (IndexError, TypeError, ValueError):
                return None
            if len(layer) != 2:
                return None
            out.setdefault((y_nm, layer), []).append(x_nm)
        return out

    left = grouped(primary_pins)
    right = grouped(partner_pins)
    if not left or not right or set(left) != set(right):
        return None
    axis2_values: set[int] = set()
    for key in sorted(left):
        left_x = sorted(left[key])
        right_x = sorted(right[key], reverse=True)
        if len(left_x) != len(right_x):
            return None
        axis2_values.update(a + b for a, b in zip(left_x, right_x))
    if len(axis2_values) != 1:
        return None
    return next(iter(axis2_values)) / (2.0 * _NM_PER_UM)


def _require_exact_mirror_axis_um(
    primary_pins: Sequence[Any],
    partner_pins: Sequence[Any],
    *,
    net_a: str,
    net_b: str,
) -> float:
    """Return the endpoint-proven axis for a declared hard net mirror.

    A placement or outline centre is not evidence that two final routed pin
    sets are reflections.  If the accesses do not admit one exact axis, stop
    before routing instead of turning a malformed hard intent into a long,
    impossible search around an unrelated fallback axis.
    """

    axis_x_um = _exact_mirror_axis_um(primary_pins, partner_pins)
    if axis_x_um is None:
        raise RoutePlanningStop(
            phase="MIRROR_AXIS_BINDING",
            state="NO_EXACT_ENDPOINT_AXIS",
            status=VerificationStatus.FAIL,
            evidence={
                "net_a": str(net_a),
                "net_b": str(net_b),
                "primary_pin_count": len(primary_pins),
                "partner_pin_count": len(partner_pins),
            },
        )
    return axis_x_um


def _opportunistic_match_mirror(
    primary_pins: Sequence[Any],
    partner_pins: Sequence[Any],
    *,
    net_a: str,
    net_b: str,
    route_constraints: Sequence[Any] = (),
    group_route_contract: Any = None,
) -> Optional[Any]:
    """Lower a matched net pair to Mirror only when final accesses prove it.

    ``LayoutIntent.diff_pairs`` expresses placement/parasitic matching. Exact
    whole-net route reflection is stronger and is sound only when every final
    endpoint admits one common axis. Returning ``None`` therefore declines an
    optional stronger lowering; explicit ``RoutingIntent`` Mirrors remain hard.
    Distinct source-declared horizontal tracks or group-route external port
    allocations also rule out this optional stronger lowering.
    """

    if group_route_contract is not None:
        group_nets = {
            str(getattr(port, "net", ""))
            for port in (getattr(group_route_contract, "ports", ()) or ())
        }
        if str(net_a) in group_nets or str(net_b) in group_nets:
            return None
    for constraint in route_constraints:
        if isinstance(constraint, _C.ChannelRoute):
            nets = constraint.declaration.nets
            if net_a != net_b and net_a in nets and net_b in nets:
                return None
    axis_x_um = _exact_mirror_axis_um(primary_pins, partner_pins)
    if axis_x_um is None:
        return None
    return _C.Mirror(net_a=str(net_a), net_b=str(net_b), axis_x_um=axis_x_um)


def _layout_match_length_constraints(
    intent: Any,
    present_nets: set[str],
) -> Tuple[Any, ...]:
    """Lower canonical layout match-length declarations for the real router.

    Layout-intent certification and route compilation must consume the same
    declaration.  Preserve the exact immutable declaration boundary here:
    mutable/coerced inventories and malformed pairs stop instead of becoming a
    different route program.  Length is symmetric, so pairs are canonicalized
    by net name and duplicate declarations collapse to one hard constraint.
    """

    from clay.layout.net_match_scope import routing_match_length_pairs
    return tuple(
        _C.Length(net_a=net_a, net_b=net_b, tol_rel=0.1)
        for net_a, net_b in routing_match_length_pairs(intent, present_nets)
    )


def _layout_route_match_balance_constraints(
    intent: Any,
    present_nets: set[str],
) -> Tuple[Any, ...]:
    """Lower soft balance constraints from @match(..., scope=route) and matched_cc_balance."""

    from clay.layout.net_match_scope import routing_match_length_pairs

    hard_pairs = {
        frozenset((str(a), str(b)))
        for a, b in routing_match_length_pairs(intent, present_nets)
    }
    pair_tol: Dict[frozenset, Tuple[Tuple[str, str], float]] = {}

    def _record(a: str, b: str, tol: float) -> None:
        key = frozenset((a, b))
        if (
            len(key) != 2
            or a not in present_nets
            or b not in present_nets
            or key in hard_pairs
        ):
            return
        clamped = min(0.06, max(0.01, float(tol)))
        canonical = (min(a, b), max(a, b))
        prev = pair_tol.get(key)
        if prev is None or clamped < prev[1]:
            pair_tol[key] = (canonical, clamped)

    for scope_item in getattr(intent, "match_scopes", ()) or ():
        if getattr(scope_item, "scope", None) == "route":
            nets = tuple(str(n) for n in getattr(scope_item, "nets", ()) or ())
            if len(nets) == 2:
                _record(nets[0], nets[1], 0.06)

    for budget in getattr(intent, "budgets", ()) or ():
        if getattr(budget, "kind", None) == "matched_cc_balance":
            pair = tuple(str(n) for n in getattr(budget, "pair", ()) or ())
            if len(pair) == 2:
                _record(pair[0], pair[1], float(getattr(budget, "rel_tol", 0.10)))

    return tuple(
        _C.Length(net_a=a, net_b=b, tol_rel=tol, hardness=_C.SOFT)
        for _k, ((a, b), tol) in sorted(pair_tol.items(), key=lambda kv: kv[1][0])
    )


class InvalidTrunkIntentError(ValueError):
    """A non-edge ``@trunk`` declaration is invalid for the target PDK."""


class ConflictingTrunkIntentError(InvalidTrunkIntentError):
    """One net declares incompatible non-edge trunk semantics."""


def _validate_trunk_constraints_for_tech(
    constraints: Sequence[Any], tech: LayoutTech,
) -> None:
    """Resolve every lowered trunk advisory before backend selection."""

    from clay.layout.route.engine_maze import _Tech

    route_tech = _Tech(tech)
    for constraint in constraints:
        if not isinstance(constraint, _C.TrunkLayer):
            continue
        if route_tech.resolve_layer_to_gi(constraint.layer) is None:
            raise InvalidTrunkIntentError(
                f"trunk layer for {constraint.net!r} is not a usable "
                f"routing layer in {tech.pdk_id}: {constraint.layer!r}"
            )


def _trunk_layer_preferences(
    intent: Any, tech: LayoutTech,
) -> Dict[str, "Declared"]:
    """Soft long-run layer preferences from declared non-edge ``@trunk`` intents.

    ``@trunk(net, layer=Mk)`` is not an allowed-layer restriction.  It biases the
    main run to Mk while leaving the PDK's upper routing stack available for legal
    cross-band transport.  Only explicit ``@route_layer`` / ``@on_layer`` intent
    lowers to a hard :class:`LayerSet`.  Edge trunks remain drawn rail geometry.
    """
    from clay.layout.route.intent import (
        Declared,
        IntentSource,
        TrunkPreference,
    )
    from clay.layout.route.engine_maze import _Tech

    # Edge rails carry their own explicit orientation and are materialized by a
    # separate compiler.  Do not demand a complete routing-world axis inventory
    # until an actual non-edge trunk needs graph-layer resolution.
    route_tech = None
    options: Dict[str, Dict[str, set[str]]] = {}
    for t in getattr(intent, "trunks", ()) or ():
        net = str(getattr(t, "net", "") or "")
        if not net or str(getattr(t, "edge", "") or ""):
            continue  # no net, or an edge rail (drawn as geometry, not a LayerSet)
        layer = str(getattr(t, "layer", "") or "")
        k = _metal_index_of(layer, tech)
        if k is None or k < 1:
            raise InvalidTrunkIntentError(
                f"unresolvable non-edge @trunk layer for {net!r}: {layer!r}"
            )
        try:
            tech.metal_key(k)
        except KeyError as exc:
            raise InvalidTrunkIntentError(
                f"non-edge @trunk layer for {net!r} is absent from "
                f"{tech.pdk_id}: {layer!r}"
            ) from exc
        if route_tech is None:
            route_tech = _Tech(tech)
        if route_tech.resolve_layer_to_gi(f"metal{k}") is None:
            raise InvalidTrunkIntentError(
                f"non-edge @trunk layer for {net!r} is not a usable "
                f"routing layer in {tech.pdk_id}: {layer!r}"
            )
        orient = str(getattr(t, "orient", "") or "").strip().upper()
        if orient not in {"H", "V"}:
            if orient:
                raise InvalidTrunkIntentError(
                    f"non-edge @trunk orientation for {net!r} must be "
                    f"H, V, or blank; got {orient!r}"
                )
            orient = ""
        options.setdefault(net, {}).setdefault(
            f"metal{k}", set()
        ).add(orient)
    conflicts: Dict[
        str, Tuple[Tuple[str, Tuple[str, ...]], ...]
    ] = {}
    for net, by_layer in sorted(options.items()):
        concrete_axes = {
            axis
            for axes in by_layer.values()
            for axis in axes
            if axis
        }
        if len(by_layer) > 1 or len(concrete_axes) > 1:
            conflicts[net] = tuple(
                (layer, tuple(sorted(axes)))
                for layer, axes in sorted(by_layer.items())
            )
    if conflicts:
        raise ConflictingTrunkIntentError(
            "conflicting non-edge @trunk declarations: "
            + ", ".join(
                f"{net}={values!r}"
                for net, values in sorted(conflicts.items())
            )
        )
    out: Dict[str, "Declared"] = {}
    for net, by_layer in sorted(options.items()):
        layer = next(iter(by_layer))
        concrete_axes = sorted(
            axis for axis in by_layer[layer] if axis
        )
        orient = concrete_axes[0] if concrete_axes else ""
        out[net] = Declared(
            value=TrunkPreference(layer=layer, orient=orient),
            source=IntentSource.DESIGN,
        )
    return out


def _routing_intent_for(intent: Any, bridged_intent: Any, tech: LayoutTech) -> Any:
    """Return a :class:`~clay.layout.route.intent.RoutingIntent` to lower.

    When ``intent`` is already a ``RoutingIntent`` it is used directly. Otherwise a
    design-derived width intent is synthesised from the bridged MOS devices (the
    leakage-free :func:`derive_width_intent` -- widths from netlist/PDK, never a
    reference), PLUS soft per-net trunk-layer preferences from declared ``@trunk``
    intents and a nonzero ``direction_weight`` when the design declares a routing
    direction discipline, so the HVH preferred-direction axes reach the engine."""
    from clay.layout.route.intent import RoutingIntent
    if isinstance(intent, RoutingIntent):
        return intent
    from clay.layout.route.width_model import WidthPolicy, derive_width_intent

    # DECLARED @wire_width override: the LayoutIntent carries per-net widths (NANOMETRES,
    # keyed by net) the designer authored in the .clay; convert to um and hand them to the
    # width model as the per-net override so an explicit @wire_width(net, w) wins over the
    # general design-derived rail-class default for that net (the .clay is the source of
    # truth). A cell that declares none passes an empty map -> the pure role-derived widths
    # (byte-identical). Net names are DATA -- no engine / per-circuit branch.
    declared_um = {
        str(n): float(w) / _NM_PER_UM
        for n, w in (getattr(intent, "wire_widths", {}) or {}).items()
        if float(w) > 0
    }
    edge_declared_um: Dict[str, float] = {}
    for trunk in tuple(getattr(intent, "trunks", ()) or ()):
        net = str(getattr(trunk, "net", "") or "")
        width_um = float(getattr(trunk, "width_um", 0.0) or 0.0)
        edge = str(getattr(trunk, "edge", "") or "").strip().lower()
        if net and width_um > 0 and not edge:
            declared_um[net] = max(declared_um.get(net, 0.0), width_um)
        elif net and width_um > 0 and edge:
            layer_n = _metal_index_of(str(getattr(trunk, "layer", "") or ""), tech)
            edge_um = width_um
            if layer_n is not None:
                try:
                    floor_nm = int(tech.min_width(f"metal{layer_n}") or 0)
                except Exception:
                    floor_nm = 0
                if floor_nm > 0:
                    edge_um = max(width_um, floor_nm / _NM_PER_UM)
            edge_declared_um[net] = max(edge_declared_um.get(net, 0.0), edge_um)
    policy = WidthPolicy()
    widths = derive_width_intent(
        bridged_intent, tech, layer="metal1", policy=policy,
        declared=declared_um,
    )
    if edge_declared_um:
        # An edge @trunk wider than the role-derived target (e.g. 8um rail over a
        # 4um power floor) stays local to the edge bar so demand-routed drops do
        # not inflate above 4um. Conversely, when no @wire_width or interior
        # @trunk overrides the net and an explicit edge @trunk narrows the rail
        # below the role-derived floor (e.g. 1.6um M4 edge rails), bind the
        # legalized edge width so lowered Width matches the drawn rail carrier
        # and activates local minimum-width rail-drop / group-landing scoping.
        declared_bases = {
            str(k).strip().rstrip("!").lower() for k in declared_um
        }
        widths_by_base = {
            str(k).strip().rstrip("!").lower(): float(v.value)
            for k, v in widths.items()
        }
        narrowed = False
        for net, edge_um in edge_declared_um.items():
            base = net.strip().rstrip("!").lower()
            if (
                base not in declared_bases
                and base in widths_by_base
                and edge_um < widths_by_base[base]
            ):
                declared_um[net] = edge_um
                narrowed = True
        if narrowed:
            widths = derive_width_intent(
                bridged_intent, tech, layer="metal1", policy=policy,
                declared=declared_um,
            )
    cell_name = str(getattr(intent, "cell", "") or getattr(intent, "name", "") or "cell")
    trunk_layers = _trunk_layer_preferences(intent, tech)
    # DIRECTION DISCIPLINE: enabling the per-layer HVH soft preference changes every
    # route, so it is gated on the design DECLARING it via @trunk (the .clay is the
    # source of truth). A cell with no @trunk -> weight 0 -> lower_intent emits NO
    # Direction constraint -> byte-identical. The gridless engine takes its direction
    # from the PDK preferred axis directly (Direction is a no-op there); the weight
    # drives the maze/PathFinder path.
    direction_weight = _HVH_DIRECTION_WEIGHT if (getattr(intent, "trunks", ()) or ()) else 0.0
    return RoutingIntent(
        cell=cell_name,
        widths=widths,
        trunk_layers=trunk_layers,
        direction_weight=direction_weight,
    )
