"""Strict hierarchical native SPICE import into explicit Clay declarations.

Every reachable X card must resolve either to a source subcircuit or to a
PDK-owned native model contract. This importer does not infer device families,
change source units, remove dummy devices, evaluate schematic expressions, or
associate schematic nets with GDS/extracted names. Parameterized hierarchy and
external lexical parameter scopes are rejected until they have a typed lowering.

Native parameter expressions may refer only to parameters on the same instance;
the original expression token is emitted unchanged. Physical W/L/nf/m values
must be finite literals. The resulting .clay source can be parsed and emitted
back to native SPICE without the import object or a sidecar file.
"""
from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import io
import json
import math
import re
import tokenize
from typing import Mapping

from clay.ir import CellDef, Device, Spec
from clay.pdk.native_spice import NativeSpiceModelContract, native_spice_model_contract
from clay.spice_analysis import _parse_spice_value, _tokenise

_CNAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SPICE_LITERAL = re.compile(r"^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?(?:[tTgGkKmMuUnNpPfF]|[mM][eE][gG])?$")
_ENCODE_PREFIX = "clayx_"
_MAX_DEPTH = 64
_MAX_DEVICES = 100000
_MAX_TEXT = 32 * 1024 * 1024


class NativeSpiceImportError(ValueError):
    """Unsupported or ambiguous native source semantics; no partial import."""


def _require(ok, message):
    if not ok:
        raise NativeSpiceImportError(message)


def _json(data):
    return json.dumps(data, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False)


def _sha(data):
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def encode_native_name(path: tuple[str, ...]) -> str:
    """Injective identifier encoding, including literal underscores and slashes.

    Ordinary single-component identifiers remain readable. The reserved prefix
    itself is escaped; other paths encode each UTF-8 component separately.
    """
    _require(type(path) is tuple and path and all(type(x) is str and x for x in path),
             "native name requires a nonempty tuple of nonempty path components")
    if len(path) == 1 and _CNAME.fullmatch(path[0]) and not path[0].casefold().startswith(_ENCODE_PREFIX):
        return path[0]
    return _ENCODE_PREFIX + "_".join(part.encode("utf-8").hex() for part in path)


def decode_native_name(name: str) -> tuple[str, ...]:
    _require(type(name) is str and bool(_CNAME.fullmatch(name)), "native identifier is not a Clay CNAME")
    if not name.startswith(_ENCODE_PREFIX):
        return (name,)
    try:
        result = tuple(bytes.fromhex(part).decode("utf-8") for part in name[len(_ENCODE_PREFIX):].split("_"))
    except (ValueError, UnicodeError) as exc:
        raise NativeSpiceImportError("native identifier has invalid reversible encoding") from exc
    _require(all(result) and encode_native_name(result) == name, "native identifier encoding is noncanonical")
    return result


def _native_net(name: str) -> str:
    # SPICE's absolute node zero is encoded only because Clay identifiers cannot
    # start with a digit. This exact reversible spelling is not a rail-name guess.
    return "0" if name == encode_native_name(("0",)) else name


def _quoted_source_token(token: str) -> str:
    _require(type(token) is str and token and not any(ch in token for ch in "\r\n\x00"),
             "native parameter token must be one nonempty logical-line value")
    if '"' not in token:
        return '"' + token + '"'
    if "'" not in token:
        return "'" + token + "'"
    raise NativeSpiceImportError("native parameter token contains both quote types; Clay string escaping is unsupported")


def _strict_tokens(card: str) -> list[str]:
    # Reuse the existing SPICE tokenizer after rejecting the tolerance it offers
    # for analysis-only parsing (notably implicit closing of an unmatched quote).
    quote = None
    for ch in card:
        if quote is not None:
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch in ";$":
            raise NativeSpiceImportError("inline SPICE comments are unsupported in native import")
    _require(quote is None, "unterminated native SPICE quote")
    _require("{" not in card and "}" not in card,
             "brace SPICE expressions require an explicit supported scope lowering")
    return _tokenise(card)


def _parameters(tokens: list[str]) -> tuple[tuple[str, str], ...]:
    result, seen = [], set()
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if "=" in token:
            key, value = token.split("=", 1)
            i += 1
            if not value:
                _require(i < len(tokens), f"missing native parameter value for {key!r}")
                value = tokens[i]
                i += 1
        else:
            key = token
            _require(i + 1 < len(tokens) and tokens[i + 1].startswith("="),
                     f"unexpected trailing native SPICE token {token!r}")
            value = tokens[i + 1][1:]
            i += 2
            if not value:
                _require(i < len(tokens), f"missing native parameter value for {key!r}")
                value = tokens[i]
                i += 1
        _require(bool(_CNAME.fullmatch(key)) and value, "malformed native parameter assignment")
        _require(key.casefold() not in seen, f"duplicate native parameter {key!r}")
        seen.add(key.casefold())
        result.append((key, value))
    return tuple(result)


@lru_cache(maxsize=4096)
def _expression_dependencies_cached(raw: str, known_frozen: frozenset[str]) -> frozenset[str]:
    _require(len(raw) >= 2 and raw[0] in "'\"" and raw[-1] == raw[0],
             f"unsupported native parameter expression {raw!r}")
    expression = raw[1:-1].strip()
    try:
        parsed = ast.parse(expression, mode="eval")
        # Python's AST is only a structural checker. Python-only numeric forms
        # such as 1_000 and 0x10 must not acquire SPICE semantics by accident.
        for token in tokenize.generate_tokens(io.StringIO(expression).readline):
            if token.type == tokenize.NUMBER:
                _require(bool(_SPICE_LITERAL.fullmatch(token.string)),
                         "native expression contains a non-SPICE numeric literal")
            _require(token.type != tokenize.COMMENT, "native expression contains an unsupported comment")
    except (SyntaxError, ValueError, tokenize.TokenError) as exc:
        raise NativeSpiceImportError(f"unsupported native parameter expression {raw!r}") from exc
    dependencies = set()
    called_names = {id(node.func) for node in ast.walk(parsed) if isinstance(node, ast.Call)}
    allowed_nodes = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub,
                     ast.Mult, ast.Div, ast.Pow, ast.UAdd, ast.USub, ast.Load)
    for node in ast.walk(parsed):
        if isinstance(node, allowed_nodes):
            continue
        if isinstance(node, ast.Constant):
            try:
                finite = type(node.value) in (int, float) and math.isfinite(node.value)
            except OverflowError:
                finite = False
            _require(finite, "native expression constants must be finite numbers")
        elif isinstance(node, ast.Call):
            _require(isinstance(node.func, ast.Name) and node.func.id.casefold() == "int"
                     and len(node.args) == 1 and not node.keywords,
                     "native expressions currently support only the PDK int() function")
        elif isinstance(node, ast.Name):
            if node.id.casefold() == "int":
                _require(id(node) in called_names, "native int token must be a function call")
                continue
            _require(node.id.casefold() in known_frozen,
                     f"native expression references external or missing parameter {node.id!r}")
            dependencies.add(node.id.casefold())
        else:
            raise NativeSpiceImportError(f"unsupported native expression construct {type(node).__name__}")
    return frozenset(dependencies)


def _expression_dependencies(raw: str, *, known: set[str]) -> set[str]:
    return set(_expression_dependencies_cached(raw, frozenset(known)))


@dataclass(frozen=True, slots=True)
class ValidatedNativeDevice:
    contract: NativeSpiceModelContract
    numeric_parameters: tuple[tuple[str, float], ...]
    expression_dependencies: tuple[tuple[str, tuple[str, ...]], ...]
    dimensions_si: tuple[tuple[str, float], ...]


@lru_cache(maxsize=4096)
def _validate_native_device_cached(
    pdk_id: str,
    instance_name: str,
    model: str,
    pdk_variant: str | None,
    pins_items: tuple[tuple[str, str], ...],
    pairs: tuple[tuple[str, str], ...],
) -> ValidatedNativeDevice:
    _require(bool(_CNAME.fullmatch(instance_name)), "native device name is not a Clay identifier")
    _require(bool(_CNAME.fullmatch(model)), "native device requires an explicit model identifier")
    _require(pdk_variant in (None, model), "native device model and PDK variant differ")
    contract = native_spice_model_contract(pdk_id, model)
    expected_pins = {role for _terminal, role in contract.terminals}
    pin_keys = {k for k, _ in pins_items}
    _require(pin_keys == expected_pins and len(pin_keys) == len(pins_items),
             f"native device {instance_name!r} pins must be exactly {sorted(expected_pins)}")
    _require(all(bool(_CNAME.fullmatch(net)) for _, net in pins_items),
             "native pin nets must be explicit Clay identifiers")
    params = {}
    for key, value in pairs:
        _require(bool(_CNAME.fullmatch(key)) and value and not any(ch in value for ch in "\r\n\x00;$="),
                 "native parameter token contains unsupported delimiters")
        low = key.casefold()
        _require(low not in params, f"duplicate native parameter {key!r}")
        _require(low in contract.parameters, f"native model {model!r} does not declare parameter {key!r}")
        params[low] = value
    _require(set(contract.required_parameters) <= params.keys(),
             f"native device {instance_name!r} missing required parameters "
             f"{sorted(set(contract.required_parameters) - params.keys())}")
    known_frozen = frozenset(params)
    numeric, dependencies = {}, {}
    for key, token in params.items():
        value = _parse_spice_value(token) if _SPICE_LITERAL.fullmatch(token) else None
        if value is None:
            _require(key in contract.expression_parameters,
                     f"native physical parameter {key!r} must be a literal, got {token!r}")
            dependencies[key] = _expression_dependencies_cached(token, known_frozen)
            continue
        _require(math.isfinite(value), f"native parameter {key!r} is nonfinite")
        _require(value > 0 if key in contract.positive_parameters else value >= 0,
                 f"native parameter {key!r} violates its PDK sign contract")
        if key in contract.integer_parameters:
            _require(float(value).is_integer(), f"native parameter {key!r} must be an integer")
        numeric[key] = value
    # Expressions are preserved, but moving a reference to an ambient lexical
    # scope or a self-referential definition is not a valid flattening operation.
    def visit(key, stack):
        _require(key not in stack, f"cyclic native parameter expression at {key!r}")
        for dependency in dependencies.get(key, ()):
            visit(dependency, (*stack, key))
    for key in dependencies:
        visit(key, ())
    dimensions = tuple((key, numeric[key] * factor) for key, factor in contract.length_scales if key in numeric)
    return ValidatedNativeDevice(
        contract,
        tuple(sorted(numeric.items())),
        tuple((key, tuple(sorted(value))) for key, value in sorted(dependencies.items())),
        dimensions,
    )


def validate_native_device(device: Device, *, pdk_id: str) -> ValidatedNativeDevice:
    """Validate the source declaration without substituting or evaluating values."""
    _require(type(device) is Device and device.device_class == "native", "expected an explicit native Device")
    _require(type(device.instance_name) is str, "native device name is not a Clay identifier")
    model = device.source_model
    _require(type(model) is str, "native device requires an explicit model identifier")
    _require(device.pdk_variant is None or type(device.pdk_variant) is str,
             "native device model and PDK variant differ")
    _require(type(device.pins) is dict and all(type(k) is str and type(v) is str for k, v in device.pins.items()),
             "native pin nets must be explicit Clay identifiers")
    pairs = device.native_parameters
    _require(type(pairs) is tuple and bool(pairs), "native device must preserve explicit parameter pairs")
    for pair in pairs:
        _require(type(pair) is tuple and len(pair) == 2 and all(type(x) is str for x in pair),
                 "native parameters require immutable string pairs")
    _require(not device.sizing,
             "native declarations use original native_param tokens; a parallel portable sizing map is ambiguous")
    return _validate_native_device_cached(
        str(pdk_id),
        device.instance_name,
        model,
        device.pdk_variant,
        tuple(sorted(device.pins.items())),
        pairs,
    )


def native_cell_signature(cell_def: CellDef, *, pdk_id: str) -> dict:
    """Complete native topology/parameter signature; no device-family filtering."""
    _require(isinstance(cell_def, CellDef) and bool(cell_def.devices), "native source must contain a nonempty CellDef")
    _require(not cell_def.instances and not cell_def.bias_tree and not cell_def.topology,
             "native cell emission requires explicit flat devices without implicit topology")
    _require(len({name.casefold() for name in cell_def.ports}) == len(cell_def.ports),
             "native cell ports collide under SPICE case folding")
    _require(all(bool(_CNAME.fullmatch(name)) for name in cell_def.ports), "native ports must be Clay identifiers")
    seen, records = set(), []
    for device in cell_def.devices:
        info = validate_native_device(device, pdk_id=pdk_id)
        _require(device.instance_name.casefold() not in seen, "native device names collide under SPICE case folding")
        seen.add(device.instance_name.casefold())
        records.append({"instance": device.instance_name, "family": info.contract.family.value,
                        "model": device.source_model,
                        "pins": [[role, device.pins[role]] for _, role in info.contract.terminals],
                        "parameters": [list(pair) for pair in device.native_parameters]})
    # Every node spelling is an electrical identity. Do not let SPICE silently
    # collapse two differently spelled Clay nodes.
    nets = set(cell_def.ports) | {net for device in cell_def.devices for net in device.pins.values()}
    _require(len({net.casefold() for net in nets}) == len(nets), "native net names collide under SPICE case folding")
    return {"pdk_id": pdk_id, "ports": list(cell_def.ports),
            "port_roles": {port: cell_def.port_roles.get(port) for port in cell_def.ports},
            "devices": records}


def emit_native_spice(cell_def: CellDef, *, pdk_id: str, top_name: str | None = None) -> str:
    """Emit solely from parsed native Clay devices, preserving all parameter tokens."""
    native_cell_signature(cell_def, pdk_id=pdk_id)
    top = top_name or cell_def.local_name
    _require(type(top) is str and bool(_CNAME.fullmatch(top)), "native SPICE top must be a Clay identifier")
    lines = [f".subckt {top} " + " ".join(_native_net(net) for net in cell_def.ports)]
    for device in cell_def.devices:
        contract = native_spice_model_contract(pdk_id, device.source_model)
        nodes = " ".join(_native_net(device.pins[role]) for _, role in contract.terminals)
        parameters = " ".join(f"{key}={value}" for key, value in device.native_parameters)
        lines.append(f"X{device.instance_name} {nodes} {device.source_model} {parameters}")
    lines.append(f".ends {top}")
    return "\n".join(lines) + "\n"


def emit_native_device_clay(device: Device, *, pdk_id: str, indent: str = "") -> str:
    """Emit one validated native declaration without changing parameter tokens."""
    info = validate_native_device(device, pdk_id=pdk_id)
    inner = indent + "  "
    lines = [f"{indent}device {device.instance_name} : native {{",
             f"{inner}model = {device.source_model};"]
    lines.append(inner + " ".join(f"{role} = {device.pins[role]};" for _, role in info.contract.terminals))
    lines.extend(f"{inner}native_param {key} = {_quoted_source_token(value)};"
                 for key, value in device.native_parameters)
    lines.append(indent + "}")
    return "\n".join(lines) + "\n"


def emit_native_clay(cell_def: CellDef, *, pdk_id: str, package: str = "native_import") -> str:
    native_cell_signature(cell_def, pdk_id=pdk_id)
    _require(bool(_CNAME.fullmatch(package)), "native Clay package must be an identifier")
    ports = []
    for port in cell_def.ports:
        role = cell_def.port_roles.get(port)
        _require(role in (None, "input", "output", "inout", "power_p", "power_n", "bias", "ground"),
                 f"unsupported declared source port role {role!r}")
        ports.append(port + (f": {role}" if role else ""))
    lines = [f"package {package};", f"pdk '{pdk_id}';", "",
             f"cell {cell_def.local_name}({', '.join(ports)}) {{", "  topology {"]
    for device in cell_def.devices:
        lines.extend(emit_native_device_clay(device, pdk_id=pdk_id, indent="    ").splitlines())
    lines.extend(["  }", "}"])
    return "\n".join(lines) + "\n"


@dataclass(frozen=True, slots=True)
class _SourceInstance:
    name: str
    nodes: tuple[str, ...]
    model: str
    parameters: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class _Subcircuit:
    name: str
    ports: tuple[str, ...]
    instances: tuple[_SourceInstance, ...]


def _read_subcircuits(text: str) -> dict[str, _Subcircuit]:
    # Importing source validation must not initialize the physical layout stack.
    # Only netlist ingestion needs the existing continuation-card reader.
    from clay.layout.pex.joint_extraction import _spice_logical_cards

    _require(type(text) is str and 0 < len(text.encode("utf-8")) <= _MAX_TEXT, "native source is empty or too large")
    definitions = {}
    current = None
    ports, instances, instance_names = (), [], set()
    ended = False
    for card in _spice_logical_cards(text):
        _require(not ended, "native source has statements after .end")
        tokens = _strict_tokens(card)
        _require(bool(tokens), "empty native source card")
        head = tokens[0].casefold()
        if head == ".subckt":
            _require(current is None and len(tokens) >= 2, "nested or empty .subckt declaration")
            _require(not any("=" in token or token.casefold() == "params:" for token in tokens[2:]),
                     "parameterized source subcircuits require explicit lexical-scope lowering")
            current, ports = tokens[1], tuple(tokens[2:])
            _require(current.casefold() not in definitions, f"duplicate source subcircuit {current!r}")
            _require(len({port.casefold() for port in ports}) == len(ports), "duplicate source subcircuit port")
            instances, instance_names = [], set()
        elif head == ".ends":
            _require(current is not None and len(tokens) <= 2, "unmatched or malformed .ends")
            _require(len(tokens) == 1 or tokens[1].casefold() == current.casefold(), ".ends names a different subcircuit")
            definitions[current.casefold()] = _Subcircuit(current, ports, tuple(instances))
            current = None
        elif head == ".end":
            _require(current is None and len(tokens) == 1, ".end inside source subcircuit")
            ended = True
        else:
            _require(current is not None, f"native source statement outside a subcircuit: {tokens[0]!r}")
            _require(head.startswith("x"), f"unsupported native source card {tokens[0]!r}; expected X subcircuit instance")
            _require(head not in instance_names, f"duplicate source instance {tokens[0]!r} in {current!r}")
            instance_names.add(head)
            param_start = next((i for i, token in enumerate(tokens[1:], 1)
                                if "=" in token or (i + 1 < len(tokens) and tokens[i + 1].startswith("="))), len(tokens))
            fixed = tokens[1:param_start]
            _require(len(fixed) >= 2, "native source instance lacks model or nodes")
            instances.append(_SourceInstance(tokens[0], tuple(fixed[:-1]), fixed[-1], _parameters(tokens[param_start:])))
    _require(current is None, "unterminated source subcircuit")
    _require(bool(definitions), "native source contains no subcircuits")
    return definitions


@dataclass(frozen=True)
class NativeSpiceImport:
    cell_def: CellDef
    clay_source: str
    name_map: dict
    audit: dict

    @property
    def native_net_by_clay(self) -> dict[str, str]:
        """Source hierarchical display names only; never inferred GDS/PEX labels."""
        return {row["clay_name"]: row["source_hierarchical_name"] for row in self.name_map["nets"]}


def import_native_spice(text: str, *, top_cell: str, pdk_id: str,
                        port_roles: Mapping[str, str] | None = None) -> NativeSpiceImport:
    definitions = _read_subcircuits(text)
    _require(type(top_cell) is str and top_cell.casefold() in definitions, f"missing native top subcircuit {top_cell!r}")
    top = definitions[top_cell.casefold()]
    net_paths, device_paths, net_aliases, device_aliases = {}, {}, {}, {}
    devices, occurrences, used_subcircuits = [], [], set()

    def alias(path, paths, aliases):
        name = encode_native_name(path)
        _require(decode_native_name(name) == path, "native name encoding is not reversible")
        previous = aliases.setdefault(name.casefold(), path)
        _require(previous == path, "native aliases collide under SPICE case folding")
        paths[name] = path
        return name

    def net_alias(path):
        return alias(path, net_paths, net_aliases)

    port_paths = {port.casefold(): (port,) for port in top.ports}
    for port in top.ports:
        net_alias((port,))

    def expand(subckt, path, environment, stack):
        _require(len(stack) < _MAX_DEPTH and subckt.name.casefold() not in stack,
                 f"cyclic or excessively deep native hierarchy at {subckt.name!r}")
        used_subcircuits.add(subckt.name)
        locals_by_case = {}

        def resolve(node):
            if node == "0":
                result = ("0",)
            elif node.casefold() in environment:
                result = environment[node.casefold()]
            else:
                spelling = locals_by_case.setdefault(node.casefold(), node)
                result = (*path, spelling)
            net_alias(result)
            return result

        for instance in subckt.instances:
            instance_path = (*path, instance.name)
            resolved_nodes = tuple(resolve(node) for node in instance.nodes)
            child = definitions.get(instance.model.casefold())
            if child is not None:
                _require(not instance.parameters, "parameterized hierarchy instances require explicit lexical-scope lowering")
                _require(len(resolved_nodes) == len(child.ports), f"source subcircuit {child.name!r} pin count differs")
                expand(child, instance_path, dict(zip((pin.casefold() for pin in child.ports), resolved_nodes)),
                       (*stack, subckt.name.casefold()))
                continue
            contract = native_spice_model_contract(pdk_id, instance.model)
            _require(len(instance.nodes) == len(contract.terminals), f"native model {instance.model!r} pin count differs")
            name = alias(instance_path, device_paths, device_aliases)
            pins = {role: net_alias(node) for (_, role), node in zip(contract.terminals, resolved_nodes)}
            device = Device(name, "native", instance.model, pins, {},
                            source_model=instance.model, native_parameters=instance.parameters)
            info = validate_native_device(device, pdk_id=pdk_id)
            devices.append(device)
            _require(len(devices) <= _MAX_DEVICES, "native source expanded device limit exceeded")
            occurrences.append({"clay_name": name, "source_path": list(instance_path),
                                "source_subcircuit": subckt.name, "source_instance": instance.name,
                                "source_model": instance.model, "family": contract.family.value,
                                "source_local_nodes": list(instance.nodes),
                                "resolved_node_paths": [list(node) for node in resolved_nodes],
                                "parameters": [list(pair) for pair in instance.parameters],
                                "dimensions_si": dict(info.dimensions_si),
                                "expression_dependencies": dict(info.expression_dependencies)})

    expand(top, (), port_paths, ())
    roles = dict(port_roles or {})
    _require(set(roles) <= set(top.ports), "source port-role declarations contain unknown or differently spelled ports")
    cell_def = CellDef(encode_native_name((top.name,)), [net_alias((port,)) for port in top.ports], {}, Spec([]),
                       port_roles={net_alias((port,)): roles.get(port) for port in top.ports}, devices=devices)
    signature = native_cell_signature(cell_def, pdk_id=pdk_id)
    clay_source = emit_native_clay(cell_def, pdk_id=pdk_id)
    # The text is the deliverable. Reparse independently now, before reporting
    # success, so native parameters cannot exist only on the producer's object.
    from clay.parser import parse_clay_source
    parsed = parse_clay_source(clay_source)
    _require(len(parsed.cell_defs) == 1 and parsed.pdk_id == pdk_id, "native Clay reparsing changed file identity")
    _require(native_cell_signature(parsed.cell_defs[0], pdk_id=pdk_id) == signature,
             "native Clay reparsing changed complete topology or parameter tokens")
    emitted = emit_native_spice(parsed.cell_defs[0], pdk_id=pdk_id)
    name_map = {
        "scheme": "readable-single-CNAME-or-reserved-prefix-UTF8-component-hex/v1",
        "nets": [{"clay_name": name, "source_path": list(path),
                  "source_hierarchical_name": "/".join(path), "emitted_spice_name": _native_net(name)}
                 for name, path in sorted(net_paths.items())],
        "instances": [{"clay_name": name, "source_path": list(path),
                       "source_hierarchical_name": "/".join(path), "emitted_spice_name": "X" + name}
                      for name, path in sorted(device_paths.items())],
        "physical_name_mapping": "Unresolved here: GDS labels and extracted net names require independent physical correspondence",
    }
    contracts = {device.source_model: native_spice_model_contract(pdk_id, device.source_model).canonical_data()
                 for device in devices}
    audit = {"schema": "clay.native-spice-import/v1", "source_sha256": _sha(text), "pdk_id": pdk_id,
             "top_cell": top.name, "ports": list(top.ports), "source_subcircuits": sorted(used_subcircuits),
             "unused_source_subcircuits": sorted(set(d.name for d in definitions.values()) - used_subcircuits),
             "device_count": len(devices), "net_count": len(net_paths),
             "models": dict(sorted(Counter(device.source_model for device in devices).items())),
             "occurrences": occurrences, "model_contracts": contracts,
             "model_contracts_sha256": _sha(_json(contracts)),
             "native_topology_sha256": _sha(_json(signature)),
             "clay_source_sha256": _sha(clay_source), "emitted_native_spice_sha256": _sha(emitted),
             "roundtrip": "all model identities, ordered pins and every native parameter token preserved after Clay reparsing",
             "expression_policy": "instance-local quoted arithmetic/int expressions retained verbatim, never evaluated or rewritten",
             "limitations": ["Only explicit X-device hierarchy with no parameterized lexical subcircuit scope is supported",
                             "No native device geometry, routing, physical correspondence or signoff is implied"]}
    return NativeSpiceImport(parsed.cell_defs[0], clay_source, name_map, audit)


__all__ = ["NativeSpiceImportError", "NativeSpiceImport", "ValidatedNativeDevice",
           "import_native_spice", "emit_native_spice", "emit_native_clay", "emit_native_device_clay", "native_cell_signature",
           "validate_native_device", "encode_native_name", "decode_native_name"]
