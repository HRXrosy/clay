"""Bind an untouched native RC extraction to same-GDS physical devices.

Positive resistor contraction is a proof view only. The geometry/terminal
correspondence follows Magic's complete .ext/.res.ext device records; no node
suffix or schematic-device substitution determines electrical ownership.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math
import re
import shlex

from clay.layout.pex.joint_extraction import _spice_logical_cards
from clay.layout.pex.magic_parser import _parse_spice_value
from clay.pdk.native_spice import native_spice_model_contract


class NativeEvidenceError(ValueError):
    """Incomplete or inconsistent native evidence cannot authorize a view."""


def _require(condition, code, detail=None):
    if not condition:
        raise NativeEvidenceError(f"{code}: {detail}" if detail is not None else code)


class _UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, node):
        node = node.casefold()
        self.parent.setdefault(node, node)
        if self.parent[node] != node:
            self.parent[node] = self.find(self.parent[node])
        return self.parent[node]

    def join(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[b] = a


def parse_native_extraction(text: str, *, top_cell: str, pdk_id: str) -> dict:
    """Read every flat native X/R/C card under explicit PDK model contracts."""
    devices, resistors, capacitors, ports = {}, [], [], None
    inside, ended, seen = False, False, set()
    for card in _spice_logical_cards(text):
        parts = card.split()
        head = parts[0].casefold()
        _require(not ended, "STATEMENT_AFTER_SUBCIRCUIT", card)
        if head == ".subckt":
            _require(ports is None and not inside and len(parts) >= 2
                     and parts[1].casefold() == top_cell.casefold(), "WRONG_NATIVE_TOP", card)
            ports = [p.casefold() for p in parts[2:]]
            inside = True
            continue
        if head == ".ends":
            _require(inside and len(parts) <= 2
                     and (len(parts) == 1 or parts[1].casefold() == top_cell.casefold()),
                     "UNEXPECTED_NATIVE_END", card)
            inside, ended = False, True
            continue
        _require(inside and head[0] in "xrc", "UNEXPECTED_NATIVE_CARD", card)
        _require(head not in seen, "DUPLICATE_NATIVE_CARD", head)
        seen.add(head)
        if head[0] in "rc":
            _require(len(parts) == 4, "NONLITERAL_PARASITIC", card)
            value = _parse_spice_value(parts[3])
            _require(math.isfinite(value) and (head[0] != "r" or value > 0),
                     "NONFINITE_C_OR_NONPOSITIVE_R", card)
            row = {"name": head, "nodes": [p.casefold() for p in parts[1:3]],
                   "value": value, "literal": parts[3]}
            (resistors if head[0] == "r" else capacitors).append(row)
            continue
        first_parameter = next((i for i, token in enumerate(parts) if "=" in token), len(parts))
        model_index = first_parameter - 1
        _require(model_index >= 3, "MALFORMED_NATIVE_DEVICE", card)
        model = parts[model_index].casefold()
        contract = native_spice_model_contract(pdk_id, model)
        nodes = [p.casefold() for p in parts[1:model_index]]
        _require(len(nodes) == len(contract.terminals), "NATIVE_MODEL_TERMINAL_ARITY", card)
        params = {}
        for token in parts[first_parameter:]:
            _require("=" in token, "NONLITERAL_NATIVE_PARAMETER", card)
            key, literal = token.split("=", 1)
            key = key.casefold()
            _require(key not in params and key in contract.parameters,
                     "UNKNOWN_OR_DUPLICATE_NATIVE_PARAMETER", token)
            value = _parse_spice_value(literal)
            _require(math.isfinite(value) and
                     (value > 0 if key in contract.positive_parameters else value >= 0),
                     "NONPHYSICAL_NATIVE_PARAMETER", token)
            if key in contract.integer_parameters:
                _require(float(value).is_integer(), "NONINTEGER_NATIVE_PARAMETER", token)
            params[key] = value
        _require(set(contract.required_parameters) <= params.keys(), "MISSING_NATIVE_PARAMETER", card)
        _require(all(params.get(key, 1) == 1 for key in ("m", "mult", "nf", "mf", "vm")),
                 "COMBINED_NATIVE_OCCURRENCE", card)
        devices[head] = {"name": head, "model": model, "nodes": nodes,
                         "parameters": params, "card": card,
                         "terminal_roles": [list(pair) for pair in contract.terminals]}
    _require(not inside and ended and ports is not None and len(ports) == len(set(ports))
             and bool(devices), "INCOMPLETE_NATIVE_SUBCIRCUIT")
    return {"devices": devices, "resistors": resistors, "capacitors": capacitors, "ports": ports}


def native_nets(report: dict) -> set[str]:
    return set(report["ports"]) | {
        node for row in [*report["devices"].values(), *report["resistors"], *report["capacitors"]]
        for node in row["nodes"]
    }


def parse_magic_ext(text: str) -> dict:
    """Retain complete physical coordinate/model keys and native terminal order."""
    devices, equivalences, nodes, scales = {}, [], set(), []
    for line in text.splitlines():
        parts = shlex.split(line)
        if not parts:
            continue
        if parts[0] == "scale":
            scales.append(parts)
        elif parts[0] == "equiv":
            _require(len(parts) == 3, "NONLITERAL_MAGIC_EQUIVALENCE")
            equivalences.append(tuple(p.casefold() for p in parts[1:]))
        elif parts[0] in {"node", "port"}:
            nodes.add(parts[1].casefold())
        elif parts[0] == "device":
            _require(len(parts) >= 7, "TRUNCATED_PHYSICAL_DEVICE")
            key = tuple(parts[1:7])
            _require(key not in devices, "DUPLICATE_PHYSICAL_DEVICE_COORDINATE", key)
            quoted = [n.casefold() for n in re.findall(r'"([^\"]*)"', line)]
            _require(bool(quoted), "MISSING_PHYSICAL_TERMINALS", key)
            devices[key] = {"key": key, "model": parts[2].casefold(),
                            "nodes": quoted, "tokens": parts}
    _require(len(scales) == 1 and bool(devices), "INCOMPLETE_MAGIC_EXT")
    return {"devices": devices, "equivalences": equivalences, "nodes": nodes, "scales": scales}


def bind_native_rc(*, raw: dict, rc: dict, raw_ext: dict, rc_ext: dict,
                   res_ext: dict, physical_to_source: dict[str, str]) -> dict:
    """Prove every native terminal and RC node before exposing source names."""
    _require(not raw["resistors"] and not raw["capacitors"], "LVS_EXTRACTION_HAS_PARASITICS")
    _require(raw["ports"] == rc["ports"], "TOP_PORT_ORDER_DIFFERS")
    _require(set(raw["devices"]) == set(rc["devices"]), "NATIVE_DEVICE_IDS_DIFFER")
    _require(raw_ext["scales"] == rc_ext["scales"] == res_ext["scales"], "EXT_SCALE_DIFFERS")
    _require(set(raw_ext["devices"]) == set(rc_ext["devices"]) == set(res_ext["devices"]),
             "PHYSICAL_DEVICE_COORDINATES_DIFFER")
    _require({key: row["tokens"] for key, row in raw_ext["devices"].items()} ==
             {key: row["tokens"] for key, row in rc_ext["devices"].items()},
             "SAME_GDS_RAW_EXT_DEVICE_EVIDENCE_DIFFERS")
    original_nets = native_nets(raw)
    _require(set(physical_to_source) == original_nets and
             len(set(physical_to_source.values())) == len(original_nets), "INCOMPLETE_SOURCE_NET_MAPPING")
    aliases = _UnionFind()
    for a, b in rc_ext["equivalences"]:
        aliases.join(a, b)
    source_roots = defaultdict(set)
    for node in original_nets:
        source_roots[aliases.find(node)].add(node)
    _require(all(len(values) == 1 for values in source_roots.values()), "LVS_NETS_MERGED_BY_EXT_EQUIVALENCES")

    def canonical(node):
        anchors = source_roots.get(aliases.find(node), set())
        _require(len(anchors) == 1, "RAW_TERMINAL_WITHOUT_UNIQUE_LVS_ANCHOR", node)
        return next(iter(anchors))

    graph = _UnionFind()
    for resistor in rc["resistors"]:
        graph.join(*resistor["nodes"])
    all_nodes = native_nets(rc)
    for node in all_nodes:
        graph.find(node)
    evidence, geometry_lookup = defaultdict(list), defaultdict(list)
    for key, distributed in res_ext["devices"].items():
        original = rc_ext["devices"][key]
        _require(len(original["nodes"]) == len(distributed["nodes"]), "PHYSICAL_TERMINAL_ARITY_DIFFERS", key)
        for ordinal, (native_node, original_node) in enumerate(zip(distributed["nodes"], original["nodes"])):
            if native_node == "none" and original_node == "none":
                continue
            _require(native_node in all_nodes, "RES_EXT_TERMINAL_MISSING_FROM_PEX", native_node)
            evidence[graph.find(native_node)].append({"native_node": native_node,
                "native_lvs_net": canonical(original_node), "physical_device": key,
                "terminal_ordinal": ordinal, "authority": "same-coordinate ext/res.ext terminals"})
        real_nodes = [n for n in distributed["nodes"] if n != "none"]
        geometry_lookup[(distributed["model"], tuple(sorted(real_nodes)))].append(key)
    explicit = rc_ext["nodes"] | original_nets | {n for pair in rc_ext["equivalences"] for n in pair}
    for node in explicit & all_nodes:
        if aliases.find(node) in source_roots:
            evidence[graph.find(node)].append({"native_node": node, "native_lvs_net": canonical(node),
                                               "authority": "explicit original node/equivalence"})
    components = defaultdict(list)
    for node in sorted(all_nodes):
        components[graph.find(node)].append(node)
    native_mapping, source_mapping, component_rows, source_components = {}, {}, [], defaultdict(list)
    for root, members in sorted(components.items()):
        anchors = sorted({row["native_lvs_net"] for row in evidence[root]})
        _require(len(anchors) == 1, "RC_COMPONENT_WITHOUT_UNIQUE_LVS_ANCHOR",
                 {"root": root, "anchors": anchors, "members": members})
        original = anchors[0]
        source_components[original].append(root)
        native_mapping.update({node: original for node in members})
        source_mapping.update({node: physical_to_source[original] for node in members})
        component_rows.append({"component": root, "members": members, "native_lvs_net": original,
                               "source_net": physical_to_source[original], "anchor_evidence": evidence[root]})
    _require(set(source_components) == original_nets, "RC_ANCHOR_NET_INVENTORY_DIFFERS")
    _require(all(len(roots) == 1 for roots in source_components.values()), "ORIGINAL_NET_SPLIT_IN_RC_GRAPH")
    device_rows, physical_seen, junction_differences = [], [], []
    for ident, device in rc["devices"].items():
        reference = raw["devices"][ident]
        keys = geometry_lookup.get((device["model"], tuple(sorted(device["nodes"]))), [])
        _require(len(keys) == 1, "NATIVE_CARD_WITHOUT_UNIQUE_PHYSICAL_DEVICE", ident)
        physical_seen.append(keys[0])
        _require(device["model"] == reference["model"], "NATIVE_MODEL_CHANGED", ident)
        contracted = [native_mapping[node] for node in device["nodes"]]
        _require(contracted == reference["nodes"], "ORDERED_NATIVE_DEVICE_TERMINALS_DIFFER", ident)
        differences = {key: {"lvs": reference["parameters"].get(key), "rc": device["parameters"].get(key)}
                       for key in set(reference["parameters"]) | set(device["parameters"])
                       if reference["parameters"].get(key) != device["parameters"].get(key)}
        _require(not (set(differences) - {"ad", "as", "pd", "ps"}), "NONJUNCTION_DEVICE_PARAMETER_CHANGED", ident)
        if differences:
            junction_differences.append({"name": ident, "physical_device": keys[0], "differences": differences})
        device_rows.append({"name": ident, "physical_device": keys[0], "model": device["model"],
                            "terminal_roles": device["terminal_roles"], "native_nodes": device["nodes"],
                            "contracted_nodes": contracted, "lvs_nodes": reference["nodes"],
                            "native_parameters": device["parameters"], "lvs_parameters": reference["parameters"]})
    _require(Counter(physical_seen) == Counter(raw_ext["devices"].keys()), "PHYSICAL_CHANNEL_COVERAGE_DIFFERS")
    external, internal, coupling = defaultdict(float), defaultdict(float), defaultdict(float)
    for cap in rc["capacitors"]:
        a, b = [source_mapping[node] for node in cap["nodes"]]
        if a == b:
            internal[a] += cap["value"]
        else:
            external[a] += cap["value"]
            external[b] += cap["value"]
            coupling[tuple(sorted((a, b)))] += cap["value"]
    return {"passed": True, "device_count": len(device_rows), "resistor_count": len(rc["resistors"]),
            "capacitor_count": len(rc["capacitors"]), "source_net_count": len(original_nets),
            "distributed_node_count": len(source_mapping), "negative_capacitor_count": sum(c["value"] < 0 for c in rc["capacitors"]),
            "native_rc_node_to_source_net": source_mapping,
            "source_net_to_native_lvs_net": {source: native for native, source in physical_to_source.items()},
            "rc_components": component_rows, "devices": device_rows,
            "junction_parameters_equal_to_nonRC_LVS": not junction_differences,
            "junction_parameter_differences": junction_differences,
            "external_signed_capacitance_fF": {net: value * 1e15 for net, value in sorted(external.items())},
            "internal_signed_capacitance_fF": {net: value * 1e15 for net, value in sorted(internal.items())},
            "coupling_signed_capacitance_fF": [{"nets": list(nets), "fF": value * 1e15}
                                               for nets, value in sorted(coupling.items())],
            "scope": "Complete same-GDS native physical terminals and positive-R components; signed C preserved. Junction redistribution is disclosed. Simulation bytes are never rewritten."}
