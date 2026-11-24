"""Compile declaration-derived net demands into a deterministic topology plan.

The compiler is deliberately downstream of pin, guard-ring, and rail
materialization.  It observes the final terminal set and explicit design
declarations, then publishes typed topology candidates.  Canonical plan-consuming
engines use the plan to compile their candidate pool; non-consuming engines retain
it as report-only evidence.  Classification uses port roles and declarations only;
circuit names and literal net spellings are not inputs.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any, Optional

from clay.layout.route.resource_plan import (
    MAX_TOPOLOGY_LAYERS,
    RequiredRouteGoal,
    RoutingDemand,
    TerminalDemand,
    TopologyCandidate,
    TopologyPlan,
)


_SUPPLY_PORT_ROLES = frozenset({"power_p", "power_n", "ground"})
_TOPOLOGY_CLASSES = frozenset(
    {"local_signal", "shared_signal_trunk", "power_spine_tree"}
)


class TopologyCompileError(ValueError):
    """The declarations or PDK facts cannot produce an honest topology plan."""


def guard_ring_declaration_ref(declaration: Any, *, occurrence: int = 0) -> str:
    """Return a readable, byte-stable identity for one guard-ring declaration."""

    members = ",".join(sorted(_members_of(declaration)))
    well = str(getattr(declaration, "well", "") or "")
    tie = str(
        getattr(declaration, "tie_net", getattr(declaration, "tie", "")) or ""
    )
    return (
        f"guard_ring:members={members}|well={well}|tie={tie}"
        f"|occurrence={int(occurrence):03d}"
    )


def compile_routing_demands(
    pins_by_net: Mapping[str, Sequence[Any]],
    *,
    cell_def: Any,
    intent: Any,
    routing_intent: Any = None,
    tech: Any,
    guard_ring_taps: Sequence[Mapping[str, Any]] = (),
) -> tuple[RoutingDemand, ...]:
    """Compile final pins and declarations into sorted immutable net demands."""

    pin_map = {
        str(net): tuple(pins or ())
        for net, pins in (pins_by_net or {}).items()
    }
    port_roles = {
        str(net): (None if role is None else str(role).strip().lower())
        for net, role in (getattr(cell_def, "port_roles", {}) or {}).items()
    }
    trunks = tuple(getattr(intent, "trunks", ()) or ())
    guard_rings = tuple(getattr(intent, "guard_rings", ()) or ())

    width_by_net = _declared_widths_nm(
        routing_intent if routing_intent is not None else intent,
        () if routing_intent is not None else trunks,
    )
    allowed_layers_by_net = _routing_intent_allowed_layers(
        routing_intent, tech
    )
    preferred_layers_by_net = _routing_intent_trunk_layers(
        routing_intent, tech
    )
    current_by_net = _numeric_declaration_map(
        intent,
        ("net_currents_ma", "current_demands_ma", "currents_ma", "net_currents"),
        cast=float,
    )
    max_drop_by_net: dict[str, float] = {}
    max_resistance_by_net: dict[str, float] = {}
    max_capacitance_by_net: dict[str, float] = {}
    from clay.layout.intent import (
        LayoutIntent,
        replay_kelvin_inventory,
        replay_power_plane_inventory,
        replay_power_budget_inventory,
        replay_sensitive_net_budget_inventory,
    )

    if type(intent) is LayoutIntent:
        stored_fields = vars(intent)
        if not {"power_budgets", "sensitive_net_budgets"}.issubset(stored_fields):
            raise TopologyCompileError(
                "routing intent lacks the electrical budget inventory"
            )
        if "kelvins" not in stored_fields:
            raise TopologyCompileError("routing intent lacks the Kelvin inventory")
        if "power_planes" not in stored_fields:
            raise TopologyCompileError("routing intent lacks the power-plane inventory")
        raw_power_planes = stored_fields["power_planes"]
        raw_power_budgets = stored_fields["power_budgets"]
        raw_sensitive_budgets = stored_fields["sensitive_net_budgets"]
        raw_kelvins = stored_fields["kelvins"]
    else:
        # Lightweight report/test intent surfaces predate the closed LayoutIntent
        # schema.  They may omit the feature entirely, but if they expose it the
        # exact inventory replay still applies.
        raw_power_planes = getattr(intent, "power_planes", ())
        raw_power_budgets = getattr(intent, "power_budgets", ())
        raw_sensitive_budgets = getattr(intent, "sensitive_net_budgets", ())
        raw_kelvins = getattr(intent, "kelvins", ())
    power_planes = replay_power_plane_inventory(raw_power_planes)
    kelvins = replay_kelvin_inventory(raw_kelvins)
    for budget in replay_power_budget_inventory(raw_power_budgets):
        current_by_net[budget.net] = budget.current_a * 1000.0
        max_drop_by_net[budget.net] = budget.max_drop_v
        max_resistance_by_net[budget.net] = budget.max_drop_v / budget.current_a
    for budget in replay_sensitive_net_budget_inventory(raw_sensitive_budgets):
        prior = max_resistance_by_net.get(budget.net)
        max_resistance_by_net[budget.net] = (
            budget.max_resistance_ohm
            if prior is None
            else min(prior, budget.max_resistance_ohm)
        )
        max_capacitance_by_net[budget.net] = budget.max_capacitance_f
    via_by_net = _numeric_declaration_map(
        intent,
        ("via_multiplicities", "via_multiplicity", "via_counts"),
        cast=int,
    )
    topology_by_net = _topology_declaration_map(intent)

    trunk_by_net: dict[str, list[Any]] = {}
    for trunk in trunks:
        trunk_by_net.setdefault(str(getattr(trunk, "net", "") or ""), []).append(trunk)
    plane_by_net: dict[str, list[Any]] = {}
    for plane in power_planes:
        plane_by_net.setdefault(str(getattr(plane, "net", "") or ""), []).append(plane)

    goals_by_net: dict[str, list[RequiredRouteGoal]] = {}
    sorted_taps = sorted(
        (dict(tap) for tap in guard_ring_taps),
        key=_tap_sort_key,
    )
    used_taps: set[int] = set()
    occurrence_by_identity: dict[tuple[tuple[str, ...], str, str], int] = {}
    for declaration in guard_rings:
        identity = _guard_ring_identity(declaration)
        occurrence = occurrence_by_identity.get(identity, 0)
        occurrence_by_identity[identity] = occurrence + 1
        declaration_ref = guard_ring_declaration_ref(
            declaration, occurrence=occurrence
        )
        target: Optional[TerminalDemand] = None
        for tap_index, tap in enumerate(sorted_taps):
            if tap_index in used_taps:
                continue
            if _guard_ring_tap_identity(tap) != identity:
                continue
            target = _terminal_from_guard_ring_tap(tap, declaration_ref)
            if target is not None:
                used_taps.add(tap_index)
                break
        tie = identity[2]
        goals_by_net.setdefault(tie, []).append(
            RequiredRouteGoal(
                kind="guard_ring_frame_tie",
                net=tie,
                target=target,
                declaration_ref=declaration_ref,
            )
        )

    for declaration in kelvins:
        for role, net in (
            ("force", declaration.force_net),
            ("sense", declaration.sense_net),
        ):
            target = _kelvin_target_terminal(
                pin_map,
                net=net,
                target_device=declaration.target_device,
            )
            goals_by_net.setdefault(net, []).append(
                RequiredRouteGoal(
                    kind=f"kelvin_{role}_terminal",
                    net=net,
                    target=target,
                    declaration_ref=f"kelvin:{declaration.sha256}:{role}",
                )
            )

    demand_nets = set(pin_map)
    demand_nets.update(net for net in trunk_by_net if net)
    demand_nets.update(net for net in plane_by_net if net)
    demand_nets.update(topology_by_net)
    demand_nets.update(goals_by_net)
    demand_nets.update(max_resistance_by_net)
    demand_nets.update(max_capacitance_by_net)

    demands: list[RoutingDemand] = []
    for net in sorted(demand_nets):
        net_trunks = tuple(trunk_by_net.get(net, ()))
        net_planes = tuple(plane_by_net.get(net, ()))
        declared_trunk_layers = tuple(sorted(set(tuple(
            layer_n
            for layer_n in (_parse_layer_n(getattr(t, "layer", "")) for t in net_trunks)
            if layer_n is not None
        ) + preferred_layers_by_net.get(net, ()))))
        allowed_layers = allowed_layers_by_net.get(net, ())
        trunk_layers = (
            tuple(
                layer_n
                for layer_n in declared_trunk_layers
                if layer_n in set(allowed_layers)
            )
            if allowed_layers
            else declared_trunk_layers
        )
        plane_layers = tuple(
            layer_n
            for layer_n in (_parse_layer_n(getattr(p, "layer", "")) for p in net_planes)
            if layer_n is not None
        )
        edge_trunks = tuple(
            str(getattr(t, "edge", "") or "").strip().lower()
            for t in net_trunks
            if str(getattr(t, "edge", "") or "").strip()
        )
        role = port_roles.get(net)
        functional_role = "supply" if role in _SUPPLY_PORT_ROLES else (role or "signal")
        terminals = tuple(
            _terminal_from_pin(pin) for pin in pin_map.get(net, ())
        )
        required_goals = tuple(goals_by_net.get(net, ()))
        _validate_terminal_provenance(
            terminals
            + tuple(
                goal.target
                for goal in required_goals
                if goal.target is not None
            )
        )
        # One labelled physical terminal is already the complete top-level
        # access for a leaf net; width/current/via constraints shape geometry
        # only after a connectivity obligation exists.  They must not fabricate
        # a second endpoint or silently promote topology.  Explicit trunks,
        # planes, topologies, and guard-ring goals remain structural demands.
        if (
            len(terminals) < 2
            and not net_trunks
            and not net_planes
            and not required_goals
            and topology_by_net.get(net) is None
            and net not in max_resistance_by_net
            and net not in max_capacitance_by_net
        ):
            continue
        demands.append(
            RoutingDemand(
                net=net,
                functional_role=functional_role,
                terminals=terminals,
                required_goals=required_goals,
                declared_trunk_layers=trunk_layers,
                declared_allowed_layers=allowed_layers,
                declared_edge_trunks=edge_trunks,
                declared_power_plane_layers=plane_layers,
                declared_topology=topology_by_net.get(net),
                declared_width_nm=width_by_net.get(net),
                declared_current_ma=current_by_net.get(net),
                declared_via_multiplicity=via_by_net.get(net),
                declared_max_drop_v=max_drop_by_net.get(net),
                declared_max_resistance_ohm=max_resistance_by_net.get(net),
                declared_max_capacitance_f=max_capacitance_by_net.get(net),
            )
        )
    return tuple(demands)


def compile_topology_candidates(
    demand: RoutingDemand,
    tech: Any,
) -> tuple[TopologyCandidate, ...]:
    """Return the preferred generic topology candidate for one net demand.

    A supply role, edge rail, or power plane is structural power intent and wins
    over pin-count heuristics.  For signals, three or more terminals require a
    branch point and therefore form a shared trunk; two-terminal connections
    remain local.  Only structural topology/trunk declarations promote a signal;
    width, current, and via constraints shape the selected topology without
    becoming hidden topology switches.
    """

    topology_class = _classify(demand)
    preferred_layers = tuple(sorted(set(
        demand.declared_trunk_layers + demand.declared_power_plane_layers
    )))
    if demand.declared_allowed_layers:
        compatible = tuple(
            layer_n
            for layer_n in preferred_layers
            if layer_n in set(demand.declared_allowed_layers)
        )
        layers = compatible or demand.declared_allowed_layers
    else:
        layers = preferred_layers or (1,)
    branch_terminals = list(demand.terminals)
    occupied_accesses = {
        (terminal.xy_nm, terminal.layer) for terminal in branch_terminals
    }
    for goal in demand.required_goals:
        target = goal.target
        if target is None or (target.xy_nm, target.layer) in occupied_accesses:
            continue
        branch_terminals.append(target)
        occupied_accesses.add((target.xy_nm, target.layer))

    branch_targets = tuple(sorted({
        terminal.xy_nm for terminal in branch_terminals
    }))
    pdk_min_width = max(_pdk_min_width_nm(tech, layer_n) for layer_n in layers)
    width_nm = max(
        pdk_min_width,
        int(demand.declared_width_nm or 0),
        _electrical_resistance_width_floor_nm(
            demand, tech=tech, layers=layers, branch_targets=branch_targets
        ),
    )

    if demand.declared_via_multiplicity is not None:
        via_multiplicity = demand.declared_via_multiplicity
    else:
        demanded_keys = {_metal_key(tech, layer_n) for layer_n in layers}
        terminal_layers = {
            terminal.layer for terminal in demand.terminals
        } | {
            goal.target.layer
            for goal in demand.required_goals
            if goal.target is not None
        }
        # No current-density or via-redundancy table is inferred.  A single cut
        # is only requested when the declared routing layer differs from an access
        # layer; otherwise the honest minimum demand is zero transitions.
        via_multiplicity = 1 if terminal_layers - demanded_keys else 0

    transport_layers = _pdk_transport_layer_closure(tech)
    if demand.declared_allowed_layers:
        transport_layers = tuple(
            layer_n
            for layer_n in transport_layers
            if layer_n in set(demand.declared_allowed_layers)
        )
    if not set(layers).issubset(transport_layers):
        missing = tuple(sorted(set(layers) - set(transport_layers)))
        raise TopologyCompileError(
            f"declared routing layers are absent from the PDK closure: {missing}"
        )

    return (
        TopologyCandidate(
            net=demand.net,
            topology_class=topology_class,
            branch_targets=branch_targets,
            layers=layers,
            width_nm=width_nm,
            via_multiplicity=via_multiplicity,
            branch_terminals=tuple(branch_terminals),
            required_goal_refs=tuple(
                goal.declaration_ref for goal in demand.required_goals
            ),
            transport_layers=transport_layers,
        ),
    )


def _electrical_resistance_width_floor_nm(
    demand: RoutingDemand,
    *,
    tech: Any,
    layers: Sequence[int],
    branch_targets: Sequence[tuple[int, int]],
) -> int:
    """Materialize a declared R/IR budget as a calibrated metal-width floor."""

    max_resistance = demand.declared_max_resistance_ohm
    if max_resistance is None:
        return 0
    if len(branch_targets) < 2:
        raise TopologyCompileError(
            f"electrical budget net {demand.net!r} has fewer than two route accesses"
        )
    sheet_reader = getattr(
        tech, "calibrated_metal_sheet_resistance_ohm_sq", None
    )
    sheet_values = (
        tuple(sheet_reader(layer_n) for layer_n in layers)
        if callable(sheet_reader)
        else ()
    )
    if not sheet_values or any(
        type(value) is not float
        or not math.isfinite(value)
        or value <= 0.0
        for value in sheet_values
    ):
        raise TopologyCompileError(
            f"electrical budget net {demand.net!r} requires calibrated sheet resistance"
        )
    xs = tuple(point[0] for point in branch_targets)
    ys = tuple(point[1] for point in branch_targets)
    span_nm = max(xs) - min(xs) + max(ys) - min(ys)
    width_nm = math.ceil(max(sheet_values) * span_nm / max_resistance)
    if width_nm > 1_000_000_000:
        raise TopologyCompileError(
            f"electrical budget net {demand.net!r} requires an unbounded route width"
        )
    return max(1, width_nm)


def compile_topology_plan(
    pins_by_net: Mapping[str, Sequence[Any]],
    *,
    cell_def: Any,
    intent: Any,
    routing_intent: Any = None,
    tech: Any,
    guard_ring_taps: Sequence[Mapping[str, Any]] = (),
) -> TopologyPlan:
    """Compile the complete deterministic demand/candidate collection."""

    demands = compile_routing_demands(
        pins_by_net,
        cell_def=cell_def,
        intent=intent,
        routing_intent=routing_intent,
        tech=tech,
        guard_ring_taps=guard_ring_taps,
    )
    return TopologyPlan(
        demands=demands,
        candidates=tuple(
            candidate
            for demand in demands
            for candidate in compile_topology_candidates(demand, tech)
        ),
    )


def _classify(demand: RoutingDemand) -> str:
    if (
        demand.functional_role == "supply"
        or demand.declared_edge_trunks
        or demand.declared_power_plane_layers
        or demand.declared_topology == "power_spine_tree"
    ):
        return "power_spine_tree"
    if demand.declared_topology is not None:
        return demand.declared_topology
    if (
        len(demand.terminals) > 2
        or demand.declared_trunk_layers
        or demand.declared_allowed_layers
    ):
        return "shared_signal_trunk"
    return "local_signal"


def _terminal_from_pin(pin: Any) -> TerminalDemand:
    from clay.layout.route.terminal_identity import pin_provenance

    xy_nm = getattr(pin, "xy_nm", None)
    if xy_nm is None:
        xy_um = getattr(pin, "xy_um", None)
        if xy_um is None:
            raise TopologyCompileError(f"pin has no xy_nm/xy_um point: {pin!r}")
        xy_nm = tuple(int(round(float(value) * 1000.0)) for value in xy_um)
    else:
        xy_nm = tuple(int(value) for value in xy_nm)
    layer = _layer_key(getattr(pin, "layer", None))
    return TerminalDemand(
        xy_nm=(xy_nm[0], xy_nm[1]),
        layer=layer,
        provenance=pin_provenance(pin),
    )


def _kelvin_target_terminal(
    pin_map: Mapping[str, Sequence[Any]],
    *,
    net: str,
    target_device: str,
) -> TerminalDemand:
    from clay.layout.route.terminal_identity import pin_matches_source

    selected_pins = tuple(pin for pin in pin_map.get(net, ())
                          if pin_matches_source(pin, target_device))
    candidates = tuple(
        _terminal_from_pin(pin)
        for pin in selected_pins
    )
    if not candidates:
        raise TopologyCompileError(
            f"Kelvin target access is missing: net={net!r}, "
            f"device={target_device!r}"
        )
    if (any(getattr(pin, "terminal_binding", None) is not None for pin in selected_pins)
            and len({terminal.provenance for terminal in candidates}) != 1):
        raise TopologyCompileError(
            f"Kelvin target resolves multiple physical components: net={net!r}, "
            f"device={target_device!r}"
        )
    return min(
        candidates,
        key=lambda terminal: (
            terminal.xy_nm,
            terminal.layer,
            terminal.provenance,
        ),
    )


def _terminal_from_guard_ring_tap(
    tap: Mapping[str, Any], declaration_ref: str
) -> Optional[TerminalDemand]:
    bbox = tap.get("bbox_nm")
    access_xy_nm = tap.get("access_xy_nm")
    layer = tap.get("layer")
    try:
        values = tuple(int(value) for value in bbox)
    except (TypeError, ValueError):
        return None
    try:
        access = tuple(int(value) for value in access_xy_nm)
    except (TypeError, ValueError):
        return None
    if len(values) != 4 or len(access) != 2 or layer is None:
        return None
    x0, y0, x1, y1 = values
    if x1 <= x0 or y1 <= y0:
        return None
    if not (x0 <= access[0] <= x1 and y0 <= access[1] <= y1):
        return None
    try:
        layer_key = _layer_key(layer)
    except TopologyCompileError:
        return None
    ring = str(tap.get("ring", "") or "")
    return TerminalDemand(
        xy_nm=(access[0], access[1]),
        layer=layer_key,
        provenance=(
            f"pin:{ring}:tap"
            if ring
            else f"{declaration_ref}:materialized_tap"
        ),
    )


def _layer_key(layer: Any) -> tuple[int, int]:
    try:
        values = tuple(layer)
    except TypeError as exc:
        raise TopologyCompileError(f"terminal layer is not a GDS key: {layer!r}") from exc
    if len(values) != 2:
        raise TopologyCompileError(f"terminal layer is not a GDS key: {layer!r}")
    try:
        return int(values[0]), int(values[1])
    except (TypeError, ValueError) as exc:
        raise TopologyCompileError(f"terminal layer is not a GDS key: {layer!r}") from exc


def _members_of(declaration: Any) -> tuple[str, ...]:
    return tuple(str(member) for member in (getattr(declaration, "members", ()) or ()))


def _guard_ring_identity(declaration: Any) -> tuple[tuple[str, ...], str, str]:
    return (
        tuple(sorted(_members_of(declaration))),
        str(getattr(declaration, "well", "") or ""),
        str(
            getattr(declaration, "tie_net", getattr(declaration, "tie", "")) or ""
        ),
    )


def _tap_members(tap: Mapping[str, Any]) -> tuple[str, ...]:
    values = tap.get("declared_members", tap.get("members", ())) or ()
    return tuple(str(member) for member in values)


def _guard_ring_tap_identity(
    tap: Mapping[str, Any],
) -> tuple[tuple[str, ...], str, str]:
    return (
        tuple(sorted(_tap_members(tap))),
        str(tap.get("well", "") or ""),
        str(tap.get("net", "") or ""),
    )


def _validate_terminal_provenance(terminals: Sequence[TerminalDemand]) -> None:
    exact_by_provenance: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {}
    for terminal in terminals:
        exact = (terminal.xy_nm, terminal.layer)
        prior = exact_by_provenance.setdefault(terminal.provenance, exact)
        if prior != exact:
            raise TopologyCompileError(
                "terminal provenance resolves to multiple exact accesses: "
                f"{terminal.provenance!r} -> {prior!r}, {exact!r}"
            )


def _tap_sort_key(tap: Mapping[str, Any]) -> tuple[object, ...]:
    bbox = tap.get("bbox_nm", ()) or ()
    layer = tap.get("layer", ()) or ()
    return (
        tuple(sorted(_tap_members(tap))),
        str(tap.get("well", "") or ""),
        str(tap.get("net", "") or ""),
        tuple(int(v) for v in bbox),
        tuple(int(v) for v in (tap.get("access_xy_nm", ()) or ())),
        tuple(int(v) for v in layer),
        str(tap.get("ring", "") or ""),
    )


def _declared_widths_nm(intent: Any, trunks: Sequence[Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for net, width in (getattr(intent, "wire_widths", {}) or {}).items():
        value = _unwrap_declared(width)
        if value is not None and int(value) > 0:
            out[str(net)] = int(value)
    for trunk in trunks:
        net = str(getattr(trunk, "net", "") or "")
        width_um = float(getattr(trunk, "width_um", 0.0) or 0.0)
        if net and width_um > 0.0:
            out[net] = max(out.get(net, 0), int(round(width_um * 1000.0)))
    # RoutingIntent widths carry micrometres in ``Declared.value``.
    for net, declared in (getattr(intent, "widths", {}) or {}).items():
        value = _unwrap_declared(declared)
        if value is not None and float(value) > 0.0:
            out[str(net)] = max(
                out.get(str(net), 0), int(round(float(value) * 1000.0))
            )
    return out


def _routing_intent_allowed_layers(
    routing_intent: Any,
    tech: Any,
) -> dict[str, tuple[int, ...]]:
    if routing_intent is None:
        return {}
    out: dict[str, tuple[int, ...]] = {}
    for net, declared in (getattr(routing_intent, "net_layers", {}) or {}).items():
        value = _unwrap_declared(declared)
        try:
            values = tuple(value)
        except TypeError:
            continue
        layers = tuple(sorted({
            _resolve_routing_layer_n(
                item, tech, label=f"hard layer for {net!r}"
            )
            for item in values
        }))
        if layers:
            out[str(net)] = layers
    return out


def _routing_intent_trunk_layers(
    routing_intent: Any,
    tech: Any,
) -> dict[str, tuple[int, ...]]:
    if routing_intent is None:
        return {}
    from clay.layout.route.intent import TrunkPreference

    out: dict[str, tuple[int, ...]] = {}
    for net, declared in (
        getattr(routing_intent, "trunk_layers", {}) or {}
    ).items():
        value = _unwrap_declared(declared)
        if isinstance(value, TrunkPreference):
            value = value.layer
        layer_n = _resolve_routing_layer_n(
            value, tech, label=f"trunk layer for {net!r}"
        )
        out[str(net)] = (layer_n,)
    return out


def _resolve_routing_layer_n(
    value: Any, tech: Any, *, label: str,
) -> int:
    """Resolve role-name, integer, or GDS layer through the active PDK."""

    closure = _pdk_transport_layer_closure(tech)
    if (
        isinstance(value, tuple)
        and len(value) == 2
        and all(isinstance(item, int) for item in value)
    ):
        matches = tuple(
            layer_n
            for layer_n in closure
            if tuple(tech.metal_key(layer_n)) == tuple(value)
        )
        if len(matches) == 1:
            return matches[0]
        raise TopologyCompileError(
            f"{label} is absent from the PDK routing closure: {value!r}"
        )
    layer_n = _parse_layer_n(value)
    if layer_n is None or layer_n not in closure:
        raise TopologyCompileError(
            f"{label} is absent from the PDK routing closure: {value!r}"
        )
    return layer_n


def _numeric_declaration_map(
    declarations: Any,
    names: Sequence[str],
    *,
    cast: Any,
) -> dict[str, Any]:
    for name in names:
        values = getattr(declarations, name, None)
        if not isinstance(values, Mapping):
            continue
        out: dict[str, Any] = {}
        for net, declared in values.items():
            value = _unwrap_declared(declared)
            try:
                converted = cast(value)
            except (TypeError, ValueError):
                continue
            if converted > 0:
                out[str(net)] = converted
        return out
    return {}


def _topology_declaration_map(declarations: Any) -> dict[str, str]:
    for name in ("net_topologies", "topologies"):
        values = getattr(declarations, name, None)
        if not isinstance(values, Mapping):
            continue
        out: dict[str, str] = {}
        for net, declared in values.items():
            topology = str(_unwrap_declared(declared) or "")
            if topology not in _TOPOLOGY_CLASSES:
                raise TopologyCompileError(
                    f"net {net!r} declares unknown topology {topology!r}"
                )
            out[str(net)] = topology
        return out
    return {}


def _unwrap_declared(value: Any) -> Any:
    return getattr(value, "value", value)


def _parse_layer_n(value: Any) -> Optional[int]:
    if isinstance(value, int) and not isinstance(value, bool):
        return value if value > 0 else None
    token = str(value or "").strip().lower()
    for prefix in ("metal", "met", "m"):
        if token.startswith(prefix) and token[len(prefix) :].isdigit():
            layer_n = int(token[len(prefix) :])
            return layer_n if layer_n > 0 else None
    return None


def _pdk_min_width_nm(tech: Any, layer_n: int) -> int:
    accessor = getattr(tech, "min_width", None)
    if not callable(accessor):
        raise TopologyCompileError("PDK has no min_width accessor")
    failures: list[str] = []
    for alias in (f"metal{layer_n}", f"met{layer_n}", f"m{layer_n}"):
        try:
            width = accessor(alias)
        except Exception as exc:
            failures.append(f"{alias}: {exc}")
            continue
        if width is None:
            failures.append(f"{alias}: absent")
            continue
        if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
            raise TopologyCompileError(
                f"PDK {alias} min-width must be positive integer nm, got {width!r}"
            )
        return width
    raise TopologyCompileError(
        f"PDK has no honest metal{layer_n} min-width fact ({'; '.join(failures)})"
    )


def _metal_key(tech: Any, layer_n: int) -> tuple[int, int]:
    accessor = getattr(tech, "metal_key", None)
    if not callable(accessor):
        raise TopologyCompileError("PDK has no metal_key accessor")
    try:
        return _layer_key(accessor(layer_n))
    except Exception as exc:
        if isinstance(exc, TopologyCompileError):
            raise
        raise TopologyCompileError(
            f"PDK has no metal{layer_n} layer key: {exc}"
        ) from exc


def _pdk_transport_layer_closure(tech: Any) -> tuple[int, ...]:
    """Return the bounded contiguous routing stack declared by the PDK.

    Patterns may use only this signed closure.  Stopping at the first absent
    metal keeps the stack adjacent-via connected and prevents a pattern from
    silently discovering a layer that the topology certificate never named.
    """

    result = []
    for layer_n in range(1, MAX_TOPOLOGY_LAYERS + 1):
        try:
            _metal_key(tech, layer_n)
        except TopologyCompileError:
            break
        result.append(layer_n)
    if not result:
        raise TopologyCompileError("PDK declares no contiguous routing-metal stack")
    return tuple(result)


__all__ = [
    "TopologyCompileError",
    "compile_routing_demands",
    "compile_topology_candidates",
    "compile_topology_plan",
    "guard_ring_declaration_ref",
]
