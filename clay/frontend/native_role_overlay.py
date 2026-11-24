"""Add structural intent to complete native declarations without re-emitting them.

This is the native adapter to the existing role graph and matching-safe role
partition. Supply and clock identities come from explicit source declarations;
model polarity comes from PDK bindings. A role is evidence about connectivity,
not permission to discard devices or to substitute portable device geometry.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import re

from clay.frontend.native_spice_import import (
    emit_native_spice, native_cell_signature, validate_native_device,
)
from clay.frontend.role_infer import _Graph, _infer_input_pairs
from clay.frontend.role_model import (
    DeviceCard, FunctionalRoleHit, RoleDecomposition, disjoint_role_partition,
)
from clay.ir import CellDef
from clay.pdk import get_pdk
from clay.pdk.native_spice import NativeSpiceFamily


def _signature_sha(cell, pdk_id):
    payload = native_cell_signature(cell, pdk_id=pdk_id)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True).encode()).hexdigest()


@dataclass(frozen=True)
class NativeRoleOverlay:
    pdk_id: str
    source_signature_sha256: str
    decomposition: RoleDecomposition
    groups: tuple[FunctionalRoleHit, ...]
    clock_ports: tuple[str, ...]
    diagnostics: tuple[str, ...]

    @property
    def unclassified_devices(self):
        return tuple(name for group in self.groups if group.role == "unclassified"
                     for name in group.devices)

    def to_dict(self):
        result = asdict(self)
        result["unclassified_devices"] = list(self.unclassified_devices)
        result["scope"] = "additive structural intent; no geometry or signoff claim"
        return result


@dataclass(frozen=True)
class _Gate:
    kind: str
    inputs: tuple[str, ...]
    output: str
    devices: tuple[str, ...]


def _native_view(cell, pdk_id):
    adapter = get_pdk(pdk_id)
    cards, infos, diagnostics = [], {}, []
    for device in cell.devices:
        info = validate_native_device(device, pdk_id=pdk_id)
        infos[device.instance_name] = info
        family = info.contract.family
        kind = "other"
        if family is NativeSpiceFamily.MOS:
            polarities = {
                name.split("_", 1)[0]
                for name, model in adapter.device_bindings.items()
                if model.casefold() == device.source_model.casefold()
                and name.split("_", 1)[0] in {"nmos", "pmos"}
            }
            if len(polarities) == 1:
                kind = polarities.pop()
            else:
                diagnostics.append(f"{device.instance_name}: no unique PDK MOS polarity binding")
        elif family is NativeSpiceFamily.MIM:
            kind = "cap"
        elif family is NativeSpiceFamily.VARACTOR:
            kind = "varactor"
        raw = {key.casefold(): value for key, value in device.native_parameters}
        # These role-view fields remain SOURCE tokens; native_parameters is the
        # complete authority and no portable emitter consumes this adapter.
        cards.append(DeviceCard(device.instance_name, kind, dict(device.pins),
                                raw.get("w", ""), raw.get("l", ""), raw.get("m", ""),
                                device.source_model, native_parameters=device.native_parameters))
    return cards, infos, diagnostics


def _logic_gates(active, positive, negative, grounds, ports):
    """Recognize complete static inverter/NOR pullup and pulldown networks."""
    channel = defaultdict(set)
    drains = defaultdict(list)
    for card in active:
        channel[card.terminals["d"]].add(card.name)
        channel[card.terminals["s"]].add(card.name)
        drains[card.terminals["d"]].append(card)
    gates = []
    for output, drivers in drains.items():
        if output in positive | negative:
            continue
        n = [c for c in drivers if c.kind == "nmos" and c.terminals["s"] in negative
             and c.terminals["b"] in negative | grounds]
        p = [c for c in drivers if c.kind == "pmos" and c.terminals["b"] in positive]
        if len(n) == 1 and len(p) == 1:
            left, right = n[0], p[0]
            members = {left.name, right.name}
            if (right.terminals["s"] in positive
                    and left.terminals["g"] == right.terminals["g"]
                    and left.terminals["g"] not in positive | negative | {output}
                    and channel[output] == members):
                gates.append(_Gate("inverter", (left.terminals["g"],), output,
                                   (left.name, right.name)))
        if len(n) != 2 or len(p) not in (1, 2):
            continue
        inputs = {c.terminals["g"] for c in n}
        if len(inputs) != 2 or inputs & (positive | negative | {output}):
            continue
        members, valid = {c.name for c in n}, True
        for lower in p:
            internal = lower.terminals["s"]
            upper = [c for c in drains.get(internal, ()) if c.kind == "pmos"
                     and c.terminals["s"] in positive and c.terminals["b"] in positive
                     and {c.terminals["g"], lower.terminals["g"]} == inputs]
            if (internal in ports or internal in positive | negative or len(upper) != 1
                    or channel[internal] != {lower.name, upper[0].name}):
                valid = False
                break
            members.update((lower.name, upper[0].name))
        expected_drivers = {c.name for c in n + p}
        if valid and channel[output] == expected_drivers and len(members) == 2 + 2 * len(p):
            gates.append(_Gate("nor2", tuple(sorted(inputs)), output,
                               tuple(c.name for c in active if c.name in members)))
    return gates, channel


def infer_native_role_overlay(cell: CellDef, *, pdk_id: str,
                              clock_ports: tuple[str, ...] = ()) -> NativeRoleOverlay:
    """Infer a complete additive role overlay from strict, flat native Clay.

    Inference is conservative: unknown structures remain explicit unclassified
    devices. Matching uses equal native model, dimensions, finger count and
    multiplicities, independently of whether a structural role was recognized.
    Missing multiplicity tokens are not silently replaced with model defaults.
    """
    source_sha = _signature_sha(cell, pdk_id)
    if len(set(clock_ports)) != len(clock_ports):
        raise ValueError("clock ports must be unique")
    if any(port not in cell.ports or cell.port_roles.get(port) != "input" for port in clock_ports):
        raise ValueError("clock ports must be explicit input ports of the source cell")
    cards, infos, diagnostics = _native_view(cell, pdk_id)
    by_name = {card.name: card for card in cards}
    order = {card.name: index for index, card in enumerate(cards)}
    positive = {p for p in cell.ports if cell.port_roles.get(p) == "power_p"}
    negative = {p for p in cell.ports if cell.port_roles.get(p) == "power_n"}
    grounds = {p for p in cell.ports if cell.port_roles.get(p) == "ground"}
    negative = negative or grounds
    rails = positive | negative | grounds
    inputs = {p for p in cell.ports if cell.port_roles.get(p) == "input"} - set(clock_ports)
    roles, matches = [], []

    def hit(role, names, evidence, confidence=1.0):
        names = tuple(sorted(set(names), key=order.__getitem__))
        if names and not any(old.role == role and old.devices == names for old in roles):
            roles.append(FunctionalRoleHit(role, names, evidence, confidence))

    def match(a, b):
        left, right = by_name[a], by_name[b]
        ln, rn = dict(infos[a].numeric_parameters), dict(infos[b].numeric_parameters)
        # W/L/nf semantics come from validated PDK contracts, never a portable
        # sizing map. Presence also matters for optional multiplicity tokens.
        keys = {"w", "l", "nf", "m", "mult"}
        keys.update(k for k in (infos[a].contract.total_width_parameter,
                                infos[a].contract.finger_parameter) if k)
        equal = (left.source_model.casefold() == right.source_model.casefold()
                 and left.kind == right.kind and left.kind in {"nmos", "pmos"}
                 and all((key in ln) == (key in rn) and ln.get(key) == rn.get(key) for key in keys))
        pair = tuple(sorted((a, b), key=order.__getitem__))
        if equal and pair not in matches:
            matches.append(pair)
        elif not equal:
            diagnostics.append(f"matching withheld for {a}, {b}: native model or W/L/nf/m/mult differs")

    dummy = [c for c in cards if c.kind in {"nmos", "pmos"}
             and c.terminals["d"] == c.terminals["s"]]
    for card in dummy:
        hit("dummy_mos", (card.name,), "drain equals source; retained with every gate, body and native parameter")
    active = [c for c in cards if c.kind in {"nmos", "pmos"} and c not in dummy]
    for card in cards:
        if card.kind == "cap":
            shorted = card.terminals["p"] == card.terminals["n"]
            hit("shorted_mim" if shorted else "mim_capacitor", (card.name,),
                "PDK MIM contract; both explicit terminals " + ("are the same net" if shorted else "remain distinct"))
        elif card.kind == "varactor":
            shunt = card.terminals["n"] == card.terminals["b"] and card.terminals["n"] in negative | grounds
            hit("shunt_varactor" if shunt else "native_varactor", (card.name,),
                "PDK three-terminal varactor contract; p, n and b connections retained explicitly")
    if not positive or not negative:
        diagnostics.append("missing explicit positive/negative supply roles; supply-dependent inference withheld")
        raw = RoleDecomposition(cell.local_name, tuple(cards), tuple(roles), (), None, tuple(cell.ports))
        return NativeRoleOverlay(pdk_id, source_sha, raw, disjoint_role_partition(raw),
                                 tuple(clock_ports), tuple(diagnostics))

    gates, channel = _logic_gates(active, positive, negative, grounds, set(cell.ports))
    inv = [g for g in gates if g.kind == "inverter"]
    nors = [g for g in gates if g.kind == "nor2"]
    clock = {port: {(port, 0)} for port in clock_ports}
    # A finite set of (explicit root, polarity) facts handles loops/conflicting
    # paths without arbitrarily choosing whichever inverter was visited first.
    for _ in range(2 * len(inv) + 1):
        changed = False
        for gate in inv:
            propagated = {(root, 1 - parity) for root, parity in clock.get(gate.inputs[0], ())}
            old = clock.setdefault(gate.output, set())
            if not propagated <= old:
                old.update(propagated)
                changed = True
        if not changed:
            break
    ambiguous = {net for net, facts in clock.items() if len(facts) > 1}
    for net in sorted(ambiguous):
        diagnostics.append(f"ambiguous clock origin or polarity at {net}; clock-specific inference withheld")
    clock = {net: next(iter(facts)) for net, facts in clock.items() if len(facts) == 1}
    logic_names = {name for gate in gates for name in gate.devices}
    for i, a in enumerate(nors):
        for b in nors[i + 1:]:
            if (a.output in b.inputs and b.output in a.inputs
                    and set(a.devices).isdisjoint(b.devices)
                    and (set(a.inputs) - {b.output}) != (set(b.inputs) - {a.output})):
                hit("nor_latch", a.devices + b.devices,
                    f"two complete static NOR networks with reciprocal feedback {a.output}, {b.output}")
    gate_loads = defaultdict(set)
    for card in active:
        gate_loads[card.terminals["g"]].add(card.name)
    for a in inv:
        if a.inputs[0] in clock:
            hit("clock_inverter", a.devices,
                f"complete complementary inverter propagating explicit clock {clock[a.inputs[0]][0]}")
            continue
        for b in inv:
            if (a.output == b.inputs[0] and a.output not in cell.ports
                    and gate_loads[a.output] == set(b.devices)
                    and set(a.devices).isdisjoint(b.devices)):
                hit("buffer", a.devices + b.devices,
                    f"two complete inverters in series with private intermediate net {a.output}")
    for gate in gates:
        hit(gate.kind, gate.devices, f"complete static {gate.kind} conduction network at {gate.output}")

    graph = _Graph(active, rails=rails)
    proposed, _, _ = _infer_input_pairs(graph)
    input_pairs, tail_controls = [], []
    for candidate in proposed:
        a, b = (by_name[name] for name in candidate.devices)
        if (not {a.terminals["g"], b.terminals["g"]} <= inputs
                or a.terminals["d"] == b.terminals["d"]
                or {a.terminals["d"], b.terminals["d"]} & rails
                or a.terminals["b"] != b.terminals["b"]):
            continue
        # A shared source with >2 active input branches is not a uniquely
        # identified pair; do not introduce arbitrary overlapping match atoms.
        peers = [c for c in active if c.kind == a.kind and c.terminals["s"] == a.terminals["s"]
                 and c.terminals["g"] in inputs]
        if len(peers) != 2:
            diagnostics.append(f"ambiguous input pair at shared source {a.terminals['s']}")
            continue
        hit("input_pair", candidate.devices, candidate.evidence + "; distinct explicit input ports and common body")
        match(a.name, b.name)
        input_pairs.append((a, b))
        supply = negative if a.kind == "nmos" else positive
        body_supply = supply | (grounds if a.kind == "nmos" else set())
        tails = [c for c in active if c.kind == a.kind and c.terminals["d"] == a.terminals["s"]
                 and c.terminals["s"] in supply and c.terminals["b"] in body_supply]
        if len(tails) != 1:
            continue
        tail = tails[0]
        control = tail.terminals["g"]
        tail_controls.append(control)
        hit("clocked_tail" if control in clock else "switched_tail", (tail.name,),
            f"unique rail-side device at input-pair common source; control {control}")
        opposite = "pmos" if a.kind == "nmos" else "nmos"
        opposite_supply = positive if a.kind == "nmos" else negative
        opposite_body = opposite_supply | (grounds if opposite == "nmos" else set())
        precharge = [c for c in active if c.kind == opposite
                     and c.terminals["d"] in {a.terminals["d"], b.terminals["d"]}
                     and c.terminals["g"] == control and c.terminals["s"] in opposite_supply
                     and c.terminals["b"] in opposite_body]
        if len(precharge) == 2 and len({c.terminals["d"] for c in precharge}) == 2:
            hit("clocked_precharge" if control in clock else "switched_precharge",
                (c.name for c in precharge), "paired opposite-polarity rail switches at input drains; same tail control")
            match(*(c.name for c in precharge))

    analog = [c for c in active if c.name not in logic_names]
    for i, a in enumerate(analog):
        for b in analog[i + 1:]:
            if (a.kind != b.kind or a.terminals["d"] == b.terminals["d"]
                    or a.terminals["g"] != b.terminals["d"] or b.terminals["g"] != a.terminals["d"]
                    or a.terminals["s"] != b.terminals["s"] or a.terminals["b"] != b.terminals["b"]):
                continue
            outputs = {a.terminals["d"], b.terminals["d"]}
            supply = negative if a.kind == "nmos" else positive
            body_supply = supply | (grounds if a.kind == "nmos" else set())
            if a.terminals["s"] not in supply or a.terminals["b"] not in body_supply:
                continue
            hit("cross_coupled_pair", (a.name, b.name),
                "reciprocal gate-to-opposite-drain connections with common source and body supply")
            match(a.name, b.name)
            resets = defaultdict(list)
            for card in analog:
                control = card.terminals["g"]
                if (card.kind == a.kind and card.terminals["d"] in outputs
                        and card.terminals["s"] in supply and card.terminals["b"] in body_supply
                        and control in clock and any(t in clock and clock[t][0] == clock[control][0]
                                                    and clock[t][1] != clock[control][1] for t in tail_controls)):
                    resets[control].append(card)
            for control, reset in resets.items():
                if len(reset) == 2 and {c.terminals["d"] for c in reset} == outputs:
                    hit("clocked_reset", (c.name for c in reset),
                        f"paired rail switches at regenerative outputs; {control} has opposite clock phase to tail")
                    match(*(c.name for c in reset))
            opposite = "pmos" if a.kind == "nmos" else "nmos"
            opposite_supply = positive if a.kind == "nmos" else negative
            opposite_body = opposite_supply | (grounds if opposite == "nmos" else set())
            branches = []
            for output in sorted(outputs):
                found = []
                for lower in analog:
                    if lower.kind != opposite or lower.terminals["d"] != output or lower.terminals["b"] not in opposite_body:
                        continue
                    internal = lower.terminals["s"]
                    for upper in analog:
                        if (upper.kind == opposite and upper.terminals["d"] == internal
                                and upper.terminals["s"] in opposite_supply and upper.terminals["b"] in opposite_body
                                and upper.terminals["g"] in outputs - {output}
                                and internal not in cell.ports and channel[internal] == {lower.name, upper.name}):
                            found.append((lower, upper))
                if len(found) == 1:
                    branches.append(found[0])
            if len(branches) == 2 and any(
                    {branch[0].terminals["g"] for branch in branches} == {left.terminals["d"], right.terminals["d"]}
                    for left, right in input_pairs):
                names = tuple(c.name for branch in branches for c in branch)
                hit("regenerative_load", names,
                    "two opposite-polarity series branches: input-drain controls and reciprocal output feedback")
                match(branches[0][0].name, branches[1][0].name)
                match(branches[0][1].name, branches[1][1].name)

    # Strict mirror detection retains the old diode-reference premise and does
    # not reinterpret generic tied clock gates as a current mirror.
    mirror_groups = defaultdict(list)
    for card in analog:
        supply = negative if card.kind == "nmos" else positive
        body_supply = supply | (grounds if card.kind == "nmos" else set())
        if (card.terminals["s"] in supply and card.terminals["b"] in body_supply
                and card.terminals["g"] not in rails):
            mirror_groups[(card.kind, card.terminals["s"], card.terminals["b"], card.terminals["g"])].append(card)
    for group in mirror_groups.values():
        if len(group) >= 2 and any(c.terminals["d"] == c.terminals["g"] for c in group):
            hit("current_mirror", (c.name for c in group), "common gate/source/body with an explicit diode-connected reference")
            if len(group) == 2:
                match(*(c.name for c in group))

    # The role graph may overlap; physical membership may not. Composite logic
    # hits precede their constituent gates. Matching components are indivisible
    # under the existing, shared disjoint partition adapter.
    raw = RoleDecomposition(cell.local_name, tuple(cards), tuple(roles), tuple(matches),
                            None, tuple(cell.ports))
    groups = disjoint_role_partition(raw)
    flat = [name for group in groups for name in group.devices]
    if len(flat) != len(cards) or set(flat) != set(by_name):
        raise ValueError("native role partition failed complete source inventory")
    return NativeRoleOverlay(pdk_id, source_sha, raw, groups, tuple(clock_ports), tuple(diagnostics))


def apply_native_role_overlay(source: str, overlay: NativeRoleOverlay) -> str:
    """Insert additive groups/matching only, then verify complete source fidelity.

    Existing intent is not overwritten. Callers must explicitly reconcile an
    already annotated source before applying a second inferred overlay.
    """
    from clay.flow.optimized_clay import _block_end, _cell_span
    from clay.frontend.clay_emit import _annotate_block, _emit_intent
    from clay.parser import parse_clay_source

    parsed = parse_clay_source(source)
    matches = [c for c in parsed.cell_defs if c.local_name == overlay.decomposition.cell]
    if len(matches) != 1:
        raise ValueError("native overlay requires one matching source cell")
    cell = matches[0]
    if parsed.pdk_id != overlay.pdk_id or _signature_sha(cell, overlay.pdk_id) != overlay.source_signature_sha256:
        raise ValueError("native overlay source signature or PDK changed")
    # Re-inference validates the evidence as well as its inventory, including
    # matching decisions; a user-constructed object cannot inject unsafe pairs.
    if infer_native_role_overlay(cell, pdk_id=overlay.pdk_id, clock_ports=overlay.clock_ports) != overlay:
        raise ValueError("native overlay does not match current structural inference")
    if cell.functional_roles or re.search(rf"\bannotate\s+{re.escape(cell.local_name)}\s*\{{", source):
        raise ValueError("native overlay requires reconciliation of existing functional roles or annotations")
    start, end = _cell_span(source, cell.local_name)
    topology = list(re.finditer(r"\btopology\s*\{", source[start:end]))
    if len(topology) != 1:
        raise ValueError("native overlay requires one explicit topology block")
    opening = start + topology[0].end() - 1
    closing = _block_end(source, opening) - 1
    addition = "\n" + "\n".join(_emit_intent(overlay.groups)) + "\n  "
    result = source[:closing] + addition + source[closing:]
    annotate = _annotate_block(overlay.decomposition)
    if annotate:
        result = result.rstrip() + "\n\n" + annotate + "\n"
    reparsed = parse_clay_source(result)
    after = next(c for c in reparsed.cell_defs if c.local_name == cell.local_name)
    if (_signature_sha(after, overlay.pdk_id) != overlay.source_signature_sha256
            or emit_native_spice(after, pdk_id=overlay.pdk_id) != emit_native_spice(cell, pdk_id=overlay.pdk_id)):
        raise ValueError("native overlay changed source topology or parameters")
    return result


__all__ = ["NativeRoleOverlay", "infer_native_role_overlay", "apply_native_role_overlay"]
