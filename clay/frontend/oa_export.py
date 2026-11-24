"""Fail-closed intake for authoritative ``oa2verilog`` + ``oa2def`` exports.

OpenAccess remains outside Clay's runtime.  The bundled OA CLI tools provide two
plain-text, independently useful views: structural Verilog owns connectivity and
DEF owns per-instance CDF properties.  This module joins them by instance name and
lowers the result through caller-supplied master bindings.  It contains no circuit
names, PDK master guesses, or coordinate recovery.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Mapping, Sequence


class OaExportError(ValueError):
    """The two OA exports cannot prove one exact structural design."""


@dataclass(frozen=True)
class OaInstance:
    name: str
    master: str
    pins: Mapping[str, str]
    properties: Mapping[str, str]


@dataclass(frozen=True)
class OaReference:
    top_cell: str
    ports: tuple[str, ...]
    port_directions: Mapping[str, str]
    instances: tuple[OaInstance, ...]

    def by_name(self) -> dict[str, OaInstance]:
        return {item.name: item for item in self.instances}

    def master_counts(self) -> dict[str, int]:
        return dict(sorted(Counter(item.master for item in self.instances).items()))


_MOS_PINS = {"D": "drain", "G": "gate", "S": "source", "B": "body"}
_BJT_PINS = {"C": "collector", "B": "base", "E": "emitter"}
_PASSIVE_PINS = {"PLUS": "p", "MINUS": "n"}


@dataclass(frozen=True)
class ClayMasterBinding:
    """One OA master-to-Clay semantic mapping supplied as process data."""

    surface: str
    clay_kind: str
    pin_map: Mapping[str, str]
    value_property: str = ""
    default_value: str = ""
    model_property: str = "model"
    default_model: str = ""
    unit_area_m2: float = 0.0
    default_properties: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def mos(
        cls, clay_kind: str, *, default_properties: Mapping[str, str] | None = None
    ) -> "ClayMasterBinding":
        return cls(
            surface="mos",
            clay_kind=clay_kind,
            pin_map=_MOS_PINS,
            default_properties=dict(default_properties or {}),
        )

    @classmethod
    def bjt(
        cls,
        clay_kind: str,
        *,
        default_model: str,
        unit_area_m2: float,
        default_properties: Mapping[str, str] | None = None,
    ) -> "ClayMasterBinding":
        if unit_area_m2 <= 0.0:
            raise ValueError("BJT unit_area_m2 must be positive")
        return cls(
            surface="bjt",
            clay_kind=clay_kind,
            pin_map=_BJT_PINS,
            default_model=default_model,
            unit_area_m2=float(unit_area_m2),
            default_properties=dict(default_properties or {}),
        )

    @classmethod
    def resistor(
        cls, *, default_model: str, default_value: str = ""
    ) -> "ClayMasterBinding":
        return cls(
            surface="res",
            clay_kind="res",
            pin_map=_PASSIVE_PINS,
            value_property="r",
            default_value=default_value,
            default_model=default_model,
        )

    @classmethod
    def capacitor(
        cls, *, default_model: str, default_value: str = ""
    ) -> "ClayMasterBinding":
        return cls(
            surface="cap",
            clay_kind="cap",
            pin_map=_PASSIVE_PINS,
            value_property="c",
            default_value=default_value,
            default_model=default_model,
        )


_INSTANCE_RE = re.compile(
    r"(?m)^\s*([A-Za-z_]\w*)\s+([A-Za-z_]\w*)\s*\((.*?)\);", re.S
)
_PIN_RE = re.compile(r"\.([A-Za-z_]\w*)\(([^)]*)\)")
_PSEUDO_MASTERS = frozenset({"ipin", "opin", "iopin"})


def _natural_key(value: str) -> tuple[object, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
    )


def _parse_verilog(
    path: Path, masters: set[str]
) -> tuple[str, tuple[str, ...], dict[str, str], dict[str, tuple[str, dict[str, str]]]]:
    text = path.read_text(encoding="utf-8")
    module = re.search(
        r"\bmodule\s+([A-Za-z_]\w*)\s*\((.*?)\);", text, re.S
    )
    if module is None:
        raise OaExportError(f"{path}: no structural Verilog module")
    top = module.group(1)
    ports = tuple(
        item.strip() for item in module.group(2).replace("\n", " ").split(",")
        if item.strip()
    )
    directions: dict[str, str] = {}
    for direction, names in re.findall(
        r"(?m)^\s*(input|output|inout)\s+([^;]+);", text
    ):
        for name in names.split(","):
            directions[name.strip()] = direction
    if set(directions) != set(ports):
        raise OaExportError(
            f"{path}: port direction mismatch missing={sorted(set(ports)-set(directions))} "
            f"extra={sorted(set(directions)-set(ports))}"
        )

    instances: dict[str, tuple[str, dict[str, str]]] = {}
    unknown: set[str] = set()
    for match in _INSTANCE_RE.finditer(text):
        master, name, body = match.groups()
        if master == "module":
            continue
        if master in _PSEUDO_MASTERS:
            continue
        if master not in masters:
            unknown.add(master)
            continue
        pins = dict(_PIN_RE.findall(body))
        if name in instances:
            raise OaExportError(f"{path}: duplicate instance {name!r}")
        instances[name] = (master, pins)
    if unknown:
        raise OaExportError(
            f"{path}: unknown structural master(s) {', '.join(sorted(unknown))}"
        )
    return top, ports, directions, instances


def _parse_def(path: Path) -> tuple[str, dict[str, tuple[str, dict[str, str]]]]:
    text = path.read_text(encoding="utf-8")
    design = re.search(r"(?m)^\s*DESIGN\s+(\w+)\s*;", text)
    if design is None:
        raise OaExportError(f"{path}: no DEF DESIGN")
    block = re.search(r"\bCOMPONENTS\s+\d+\s*;(.*?)\bEND\s+COMPONENTS", text, re.S)
    if block is None:
        raise OaExportError(f"{path}: no DEF COMPONENTS section")
    out: dict[str, tuple[str, dict[str, str]]] = {}
    for match in re.finditer(r"(?m)^\s*-\s+(\S+)\s+(\S+)\s*(.*?);", block.group(1), re.S):
        name, master, body = match.groups()
        props: dict[str, str] = {}
        for prop in re.finditer(
            r"\bPROPERTY\s+(\w+)\s+(?:\"([^\"]*)\"|([^\s;]+))", body
        ):
            props[prop.group(1)] = prop.group(2) if prop.group(2) is not None else prop.group(3)
        if name in out:
            raise OaExportError(f"{path}: duplicate DEF component {name!r}")
        out[name] = (master, props)
    return design.group(1), out


def load_oa_exports(
    verilog_path: str | Path,
    def_path: str | Path,
    *,
    masters: set[str],
) -> OaReference:
    """Join exact OA exports by instance name and reject any seam disagreement."""

    verilog = Path(verilog_path)
    deffile = Path(def_path)
    top, ports, directions, connectivity = _parse_verilog(verilog, set(masters))
    def_top, parameters = _parse_def(deffile)
    if top != def_top:
        raise OaExportError(f"OA export top mismatch: Verilog={top!r} DEF={def_top!r}")

    physical_def_names = {
        name for name, (master, _props) in parameters.items() if master in masters
    }
    conn_names = set(connectivity)
    if conn_names != physical_def_names:
        raise OaExportError(
            "OA export instance mismatch: "
            f"missing_in_def={sorted(conn_names-physical_def_names)} "
            f"missing_in_verilog={sorted(physical_def_names-conn_names)}"
        )

    rows: list[OaInstance] = []
    for name in sorted(conn_names, key=_natural_key):
        master, pins = connectivity[name]
        def_master, props = parameters[name]
        if master != def_master:
            raise OaExportError(
                f"OA export master mismatch for {name}: Verilog={master} DEF={def_master}"
            )
        rows.append(OaInstance(name, master, dict(pins), dict(props)))
    return OaReference(top, ports, directions, tuple(rows))


def _required(mapping: Mapping[str, str], key: str, *, where: str) -> str:
    value = str(mapping.get(key, "") or "")
    if not value:
        raise OaExportError(f"{where}: missing required {key!r}")
    return value


def _resistance_token(raw: str) -> str:
    match = re.fullmatch(r"([0-9.+-]+(?:e[+-]?\d+)?)([kKmM]?)", raw, re.I)
    if match is None:
        raise OaExportError(f"unsupported resistor value token {raw!r}")
    number, suffix = match.groups()
    if suffix.lower() == "k":
        return f"{number}k"
    if suffix in {"M", "m"}:
        return f"{number}MOhm"
    return number


def _capacitance_token(raw: str) -> str:
    match = re.fullmatch(r"([0-9.+-]+(?:e[+-]?\d+)?)([fpnu]?)", raw, re.I)
    if match is None:
        raise OaExportError(f"unsupported capacitor value token {raw!r}")
    number, suffix = match.groups()
    return f"{number}{suffix.lower()}"


def render_clay_topology(
    reference: OaReference,
    *,
    bindings: Mapping[str, ClayMasterBinding],
    package: str,
    pdk: str,
    port_roles: Mapping[str, str],
) -> str:
    """Render only authoritative topology/sizing; layout intent remains separate."""

    if set(port_roles) != set(reference.ports):
        raise OaExportError(
            f"port role mismatch missing={sorted(set(reference.ports)-set(port_roles))} "
            f"extra={sorted(set(port_roles)-set(reference.ports))}"
        )
    ports = ", ".join(f"{name}: {port_roles[name]}" for name in reference.ports)
    lines = [
        f"package {package};",
        f'pdk "{pdk}";',
        "",
        f"cell {reference.top_cell}({ports}) {{",
        "  topology {",
    ]

    for item in reference.instances:
        binding = bindings.get(item.master)
        if binding is None:
            raise OaExportError(f"no Clay binding for OA master {item.master!r}")
        props = dict(binding.default_properties)
        props.update(item.properties)
        where = f"{item.master} {item.name}"
        pins: dict[str, str] = {}
        if set(item.pins) != set(binding.pin_map):
            raise OaExportError(
                f"{where}: terminal mismatch got={sorted(item.pins)} "
                f"expected={sorted(binding.pin_map)}"
            )
        for source_pin, clay_pin in binding.pin_map.items():
            pins[clay_pin] = _required(item.pins, source_pin, where=where)

        if binding.surface == "mos":
            w = _required(props, "w", where=where)
            length = _required(props, "l", where=where)
            mult = int(float(props.get("m", "1") or "1"))
            lines.extend([
                f"    device {item.name} : {binding.clay_kind} {{",
                f"      drain = {pins['drain']}; gate = {pins['gate']}; "
                f"source = {pins['source']}; body = {pins['body']};",
                f"      W = {w}; L = {length}; M = {mult};",
                "    }",
            ])
        elif binding.surface == "bjt":
            model = str(props.get(binding.model_property, binding.default_model) or "")
            if not model:
                raise OaExportError(f"{where}: no source BJT model")
            area_m2 = float(props.get("area", binding.unit_area_m2) or binding.unit_area_m2)
            area_scale = area_m2 / binding.unit_area_m2
            mult = int(float(props.get("m", "1") or "1"))
            lines.extend([
                f"    device {item.name} : {binding.clay_kind} {{",
                f"      collector = {pins['collector']}; base = {pins['base']}; "
                f"emitter = {pins['emitter']};",
                f"      model = {model}; area = {area_scale:g}; M = {mult};",
                "    }",
            ])
        elif binding.surface in {"res", "cap"}:
            raw_value = str(props.get(binding.value_property, binding.default_value) or "")
            if not raw_value:
                raise OaExportError(
                    f"{where}: missing {binding.value_property!r} and no CDF default"
                )
            value = (
                _resistance_token(raw_value)
                if binding.surface == "res"
                else _capacitance_token(raw_value)
            )
            model = str(props.get(binding.model_property, binding.default_model) or "")
            mult = int(float(props.get("m", "1") or "1"))
            lines.extend([
                f"    {binding.surface} {item.name} {{",
                f"      p = {pins['p']}; n = {pins['n']};",
                f"      value = {value};",
                *([f"      model = {model};"] if model else []),
                *([f"      M = {mult};"] if mult != 1 else []),
                "    }",
            ])
        else:
            raise OaExportError(f"{where}: unsupported Clay surface {binding.surface!r}")

    lines.extend(["  }", "}", ""])
    return "\n".join(lines)


__all__ = [
    "ClayMasterBinding",
    "OaExportError",
    "OaInstance",
    "OaReference",
    "load_oa_exports",
    "render_clay_topology",
]
