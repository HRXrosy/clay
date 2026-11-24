"""
Canonical formatter for Clay .clay source files.

Walks a parsed `ClayFile` IR and re-emits a canonical text representation.
The output is idempotent: running the formatter twice produces the same
text as running it once. It is also semantically equivalent: parsing the
formatted output yields the same IR as parsing the input (modulo source
comments — see Known limitations).

Canonical form rules
--------------------
* 2-space indent (matches existing examples in `examples/*.clay`).
* Single quotes for all strings (per spec section 1.3; double quotes in
  source are normalized to single).
* No trailing whitespace, single trailing newline.
* Blank line between top-level statements.
* Imports sorted alphabetically.
* Within a `cell_def` body, blocks appear in canonical order:
  topology -> spec -> sizing -> corners -> temps.
* Numbers are emitted with a "nice" unit suffix when the spec field's
  dimension implies one (e.g. `freq` -> `100kHz`, `volt` -> `1.8V`).
  Dimensionless dimensions (`real`, `db`, `integer`) emit a plain number.
* `corners` and `temps` blocks are omitted when they match the default
  (`["tt"]` for corners, `[27.0]` for temps).

CLI
---
    python tools/clay_format.py FILE.clay              # print to stdout
    python tools/clay_format.py FILE.clay -i           # rewrite in place
    python tools/clay_format.py FILE.clay --check      # exit 0 if formatted

Known limitations
-----------------
* **Comments are dropped.** `lark` strips `//` comments at the tokenizer
  layer, so they never reach the IR. Preserving them would require a
  custom non-tree parser; out of scope for the initial formatter.
* **Number literal form is lossy.** The parser stores numbers as SI
  floats (e.g. `100kHz` becomes `100000.0`). The formatter applies a
  per-dimension heuristic to pick a nice unit, which generally restores
  the original surface form but is not guaranteed to be identical for
  unusual literals (e.g. `1.5MHz` round-trips to `1.5MHz`, but `0.5MHz`
  may emerge as `500kHz`).
* **`verify ALIAS;` vs inline `verify pkg::Cell { ... }`** with
  identical spec/sizing collapse to the same IR. The formatter emits
  the alias form whenever a matching alias exists in the same file.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clay.ir import (
    BiasNode,
    CellAlias,
    CellDef,
    CellInstance,
    ClayFile,
    Device,
    ParsedLayoutIntentDecl,
    ParsedPolicyDecl,
    RoleDecl,
    Spec,
    SpecField,
    VerifyJob,
)
from clay.parser import parse_clay_file, parse_clay_source


# 2-space indent — matches every existing example in `examples/*.clay`.
INDENT = "  "

# Default fields on a VerifyJob that should not be emitted when they
# match. Mirrors the dataclass defaults in `clay/ir.py`.
DEFAULT_CORNERS: list[str] = ["tt"]
DEFAULT_TEMPS: list[float] = [27.0]


# Per-dimension preferred-unit ladder. The formatter walks the list and
# picks the first unit whose displayed value falls in `[1, 1000)` (or
# the smallest unit if every result is sub-unit). Keys correspond to the
# `dimension` field on `SpecField`.
_UNIT_LADDERS: dict[str, list[tuple[str, float]]] = {
    "freq":    [("GHz", 1e9), ("MHz", 1e6), ("kHz", 1e3), ("Hz", 1.0)],
    "volt":    [("V", 1.0), ("mV", 1e-3), ("uV", 1e-6)],
    "current": [("A", 1.0), ("mA", 1e-3), ("uA", 1e-6),
                ("nA", 1e-9), ("pA", 1e-12)],
    "cap":     [("F", 1.0), ("uF", 1e-6), ("nF", 1e-9),
                ("pF", 1e-12), ("fF", 1e-15)],
    "res":     [("MOhm", 1e6), ("kOhm", 1e3), ("Ohm", 1.0)],
    "ind":     [("H", 1.0), ("mH", 1e-3), ("uH", 1e-6), ("nH", 1e-9)],
    "time":    [("s", 1.0), ("ms", 1e-3), ("us", 1e-6),
                ("ns", 1e-9), ("ps", 1e-12)],
    "length":  [("m", 1.0), ("mm", 1e-3), ("um", 1e-6), ("nm", 1e-9)],
    "angle":   [("deg", 1.0)],
    # `real`, `db`, `integer` — no unit, plain number.
}


# Sizing-override keys whose values are SI quantities (so they deserve
# a unit on emit). Everything else (gm_id_*, etc.) is dimensionless
# intent and emits as a bare number. The patterns below match the
# usages in `examples/*.clay` and `clay/stage_templates.py`.
_SIZING_UNIT_HINTS: dict[str, str] = {
    "I_ref":   "current",
    "I_bias":  "current",
    "Cc":      "cap",
    "CL":      "cap",
    "C_load":  "cap",
    "W_diff":  "length",
    "W":       "length",
    "L":       "length",
    "Rz":      "res",
}


def _format_number(value: float, dimension: str | None) -> str:
    """Render a SI-unit float back into a Clay literal.

    Picks the largest unit in the dimension's ladder whose displayed
    value is at least 1 (so `1e6 Hz` -> `1MHz`, not `1000kHz`). Falls
    back to scientific-ish notation when no ladder applies (dimension
    is `real` / `db` / `integer` / None) or when the value is exactly
    zero.
    """
    # Integer dimension always renders bare-integer.
    if dimension == "integer":
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)

    # Zero is a special case — every dimension renders it as `0`.
    if value == 0:
        return "0"

    ladder = _UNIT_LADDERS.get(dimension or "")
    if not ladder:
        # Dimensionless — strip trailing `.0` for integers, otherwise
        # use a precision-preserving general format so the parser
        # reconstructs the exact same float (round-trip-safe).
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return _g_or_repr(value)

    # Walk the ladder from largest unit downward; pick the first where
    # `value / multiplier` is at least 1. If every unit gives a sub-1
    # value, fall through to the smallest unit so the smallest unit's
    # name is used (e.g. 0.5 Hz -> `0.5Hz`, not "no unit fits").
    abs_value = abs(value)
    for suffix, mult in ladder:
        if abs_value >= mult:
            scaled = value / mult
            return _format_scaled_number(scaled) + suffix
    # Fall through: use the smallest unit.
    suffix, mult = ladder[-1]
    scaled = value / mult
    return _format_scaled_number(scaled) + suffix


def _g_or_repr(value: float) -> str:
    """Render a float losslessly, preferring `:g`'s clean form when safe.

    `f"{value:g}"` produces compact 6-significant-digit output, which is
    nice for human-readable canonical form but truncates >6-sig-fig
    sizings (e.g. `6.880182e-05` -> `6.88018e-05`) and breaks round-trip
    IR equality for examples whose `sizing` block carries extra
    precision. Falls back to `repr(value)` — Python's shortest
    round-trip-safe repr — when `:g` is lossy.
    """
    g = f"{value:g}"
    try:
        if float(g) == value:
            return g
    except ValueError:
        pass
    return repr(value)


def _format_scaled_number(value: float) -> str:
    """Render a scaled-into-unit value as a clean literal.

    `100.0` -> `100`, `1.5` -> `1.5`, `3.14159` -> `3.14159`.
    Avoids scientific notation in the common case (since the unit
    suffix already handles the order of magnitude). Uses
    `_g_or_repr` so values like `6.880182e-05 / 1e-6 = 68.8018...`
    keep enough precision to round-trip back to the original float.
    """
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return _g_or_repr(value)


def _format_string(s: str) -> str:
    """Quote a string with single quotes (canonical per spec section 1.3).

    Embedded single quotes are not currently supported by the grammar
    (`SINGLE_QUOTED_STRING: /'[^']*'/`). If one slips through we fall
    back to double quotes; round-trip will then be non-canonical and
    the user will see it on the next `--check`.
    """
    if "'" not in s:
        return f"'{s}'"
    return f'"{s}"'


def _format_object_literal(obj: dict[str, object], indent: str) -> str:
    """Emit a Clay object literal `{ key: value, ... }` across lines.

    Used for topology fields whose value is a dict (the v0.0.23
    `rationale = { ... }` map). Single-element maps are emitted
    inline; multi-element maps put one field per line. Trailing comma
    on the last field for stable diffs.
    """
    if not obj:
        return "{ }"
    items = list(obj.items())
    if len(items) == 1:
        key, val = items[0]
        return "{ " + key + ": " + _format_object_value(val) + " }"
    inner_indent = indent + INDENT
    lines = ["{"]
    for key, val in items:
        lines.append(inner_indent + key + ": " + _format_object_value(val) + ",")
    lines.append(indent + "}")
    return "\n".join(lines)


def _format_object_value(val: object) -> str:
    """Single value inside an object literal — string / number / bareword."""
    if isinstance(val, str):
        return _format_string(val)
    if isinstance(val, bool):
        # Bool is not currently a Clay value, but defensive: emit as
        # bareword identifier.
        return "true" if val else "false"
    if isinstance(val, (int, float)):
        return _format_number(float(val), None)
    # Fallback for unforeseen value types (e.g. nested dict).
    return str(val)


def _format_topology_value(val: object, indent: str) -> str:
    """Single value on the right of a topology field assignment."""
    if isinstance(val, dict):
        return _format_object_literal(val, indent)
    if isinstance(val, str):
        # Heuristic: a bare identifier-style string (e.g. style =
        # 'two_stage_miller_pmos') is the canonical form. We always
        # quote strings, matching existing examples.
        return _format_string(val)
    if isinstance(val, (int, float)):
        return _format_number(float(val), None)
    return str(val)


def _emit_spec_block(spec: Spec, indent: str) -> list[str]:
    """Emit a `spec { ... }` block as a list of indented lines."""
    if not spec.fields:
        return [indent + "spec { }"]
    lines = [indent + "spec {"]
    inner = indent + INDENT
    # Pad field names to the widest one for an aligned colon (matches
    # the existing examples' visual style).
    width = max(len(f.name) for f in spec.fields)
    for f in spec.fields:
        rhs = _format_number(f.value, f.dimension)
        soft_suffix = " [soft]" if f.soft else ""
        lines.append(
            f"{inner}{f.name.ljust(width)} : {f.dimension} {f.op} {rhs}"
            f"{soft_suffix};"
        )
    lines.append(indent + "}")
    return lines


def _emit_sizing_block(
    name: str,
    sizing: dict[str, float],
    indent: str,
) -> list[str]:
    """Emit a `sizing { ... }` or `sizing_override { ... }` block.

    `name` selects the block keyword (`sizing` for cell_def,
    `sizing_override` for cell_alias / verify_block). Empty sizing
    blocks are not emitted; the caller should skip them.
    """
    if not sizing:
        return []
    inner = indent + INDENT
    keys = list(sizing.keys())
    width = max(len(k) for k in keys)
    lines = [indent + name + " {"]
    for k in keys:
        v = sizing[k]
        dim = _SIZING_UNIT_HINTS.get(k)
        rhs = _format_number(v, dim)
        lines.append(f"{inner}{k.ljust(width)} = {rhs};")
    lines.append(indent + "}")
    return lines


# v0.0.28 device-primitive sizing keys, in canonical emit order. The
# parser stores sizes in a `dict[str, float]` whose insertion order is
# the source order — but we re-emit in canonical order so the formatter
# output stays deterministic regardless of how the source was written.
_DEVICE_SIZE_KEY_ORDER: tuple[str, ...] = ("W", "L", "M", "nf")

# v0.0.28 MOS pin emit order. The IR keys are short-form (`d`, `g`,
# `s`, `b`); we emit the long-form names (`drain`, `gate`, `source`,
# `body`) because that's the human/LLM-friendly surface form. Either
# spelling parses, but we pick the long form because every existing
# example in `examples/*.clay` uses long-form pin names for MOS devices.
_MOS_PIN_SHORT_TO_LONG: dict[str, str] = {
    "d": "drain",
    "g": "gate",
    "s": "source",
    "b": "body",
}
_MOS_PIN_EMIT_ORDER: tuple[str, ...] = ("d", "g", "s", "b")

# v0.0.28 / v0.0.34 passive pin emit order. Passives (`cap`, `res`,
# `ind`) use `p` / `n` on both surface and IR — no aliasing.
_PASSIVE_PIN_EMIT_ORDER: tuple[str, ...] = ("p", "n")

# v0.0.31 lean-role sizing emit order. Same canonical ordering as
# `_DEVICE_SIZE_KEY_ORDER` but with `gm_id` inserted first (it's the
# role-level intent — sizing pass consumes it). The remaining keys
# follow the device order.
_ROLE_SIZE_KEY_ORDER: tuple[str, ...] = ("gm_id", "W", "L", "M", "nf")

# v0.0.31 lean-role single-net pin label emit order, per role kind.
# Pair roles use `tail`; mirror roles use `in` then `out`; cs roles
# use `gate` then `drain`. The order matters only for visual
# consistency — the parser accepts any order.
_ROLE_PIN_LABEL_ORDER: dict[str, tuple[str, ...]] = {
    "pmos_pair":   ("tail",),
    "nmos_pair":   ("tail",),
    "pmos_mirror": ("in", "out"),
    "nmos_mirror": ("in", "out"),
    "pmos_cs":     ("gate", "drain"),
    "nmos_cs":     ("gate", "drain"),
}
# Pair-net (two-tuple) labels emitted in this order. Only pair roles
# carry these.
_ROLE_PAIR_LABEL_ORDER: dict[str, tuple[str, ...]] = {
    "pmos_pair":   ("gates", "drains"),
    "nmos_pair":   ("gates", "drains"),
}

# Device-class to passive-decl surface keyword. MOS-family device
# classes (`pmos`, `nmos`, plus any future variants like `pmos_lvt`)
# emit via the `device NAME : KIND { ... }` form; the three passive
# classes emit via their dedicated `cap`/`res`/`ind` keyword. Any
# unknown device_class falls back to the generic `device` form so
# the output is round-trip-safe even when new classes are added.
_PASSIVE_CLASSES: frozenset[str] = frozenset({"cap", "res", "ind"})


def _passive_decl_kind(dev: Device) -> str | None:
    """Return `'cap'`/`'res'`/`'ind'` for a passive Device, else None.

    Used by the topology emitter to dispatch between the four device
    surface forms: `cap`/`res`/`ind NAME { p = ...; n = ...; value = ...; }`
    for the three passive classes, and `device NAME : KIND { drain = ...;
    gate = ...; source = ...; body = ...; W = ...; L = ...; M = ...; }`
    for everything else.
    """
    return dev.device_class if dev.device_class in _PASSIVE_CLASSES else None


def _is_role_expanded_device(dev: Device, role_names: set[str]) -> bool:
    """True if `dev` was synthesized by a role_decl expansion in this cell.

    The parser's `_expand_role_to_devices` names every expanded device
    with the deterministic pattern `<role.name>_<suffix>` where the
    suffix is `M1` (pair, mirror, cs) or `M2` (pair, mirror). We match
    on that pattern so the formatter can skip role-expanded devices
    when emitting the topology block — otherwise the surface form
    would carry both the role and its expansion, and re-parsing would
    double-count.
    """
    if not role_names:
        return False
    name = dev.instance_name
    for r_name in role_names:
        prefix = r_name + "_"
        if name.startswith(prefix):
            suffix = name[len(prefix):]
            # v0.0.31 role expansion emits at most `M1` / `M2` suffixes.
            # The narrow check avoids false positives if a user happens
            # to name an explicit device with a colliding prefix
            # (`pmos_pair diff_pair { ... }` + `device diff_pair_helper`).
            if suffix in ("M1", "M2"):
                return True
    return False


def _emit_device_body(
    dev: Device,
    indent: str,
) -> list[str]:
    """Emit the body of a `device NAME : KIND { ... }` MOS block.

    Pin order: drain / gate / source / body. Size order: W / L / M / nf.
    Width-aligned `=` for visual consistency with hand-authored examples.
    """
    inner = indent + INDENT
    # Build the pin lines first so we know how wide to align the LHS.
    pin_pairs: list[tuple[str, str]] = []
    for short in _MOS_PIN_EMIT_ORDER:
        if short in dev.pins:
            pin_pairs.append((_MOS_PIN_SHORT_TO_LONG[short], dev.pins[short]))
    # Any non-standard pins the parser left in the dict (defensive: if
    # a future device kind grows a fifth pin like `bulk_well`, emit it
    # under its short-form name verbatim so round-trip works).
    for short, net in dev.pins.items():
        if short not in _MOS_PIN_EMIT_ORDER:
            pin_pairs.append((short, net))

    size_pairs: list[tuple[str, str]] = []
    for k in _DEVICE_SIZE_KEY_ORDER:
        if k in dev.sizing:
            size_pairs.append((k, _format_number(
                float(dev.sizing[k]),
                _SIZING_UNIT_HINTS.get(k),
            )))
    # Any non-canonical sizing keys (e.g. `gm_id` left over from a role
    # expansion, or a future field added by a sizing pass). Emit them
    # in source order after the canonical ones.
    for k, v in dev.sizing.items():
        if k in _DEVICE_SIZE_KEY_ORDER:
            continue
        # Skip non-numeric / sentinel values defensively — they'd
        # otherwise round-trip as garbage. The IR only stores floats
        # in this dict per the dataclass annotation, but Python
        # doesn't enforce it and a role pass may have stashed a
        # `gm_id = 15.0` we want to emit.
        if not isinstance(v, (int, float)):
            continue
        size_pairs.append((k, _format_number(
            float(v), _SIZING_UNIT_HINTS.get(k),
        )))

    lhs_pairs = pin_pairs + size_pairs
    width = max(len(k) for k, _ in lhs_pairs) if lhs_pairs else 0
    lines: list[str] = []
    for k, v in pin_pairs:
        lines.append(f"{inner}{k.ljust(width)} = {v};")
    for k, v in size_pairs:
        lines.append(f"{inner}{k.ljust(width)} = {v};")
    return lines


def _emit_passive_body(
    dev: Device,
    indent: str,
) -> list[str]:
    """Emit the body of a `cap`/`res`/`ind NAME { ... }` block.

    Pin order: p / n. Single size field: value. Emitted inline on one
    line when both pins + the value fit (matching `lc_tank_demo.clay`
    and the other passive examples), otherwise one field per line.
    """
    inner = indent + INDENT
    pin_pairs: list[tuple[str, str]] = []
    for short in _PASSIVE_PIN_EMIT_ORDER:
        if short in dev.pins:
            pin_pairs.append((short, dev.pins[short]))
    value = dev.sizing.get("value")
    size_pairs: list[tuple[str, str]] = []
    if value is not None:
        # Use the device-class dimension hint so the value renders with
        # a nice unit (`100fF`, `1kOhm`, `1nH`) instead of `1e-13`, etc.
        dim = dev.device_class  # 'cap' / 'res' / 'ind'
        size_pairs.append(("value", _format_number(float(value), dim)))
    lhs_pairs = pin_pairs + size_pairs
    width = max(len(k) for k, _ in lhs_pairs) if lhs_pairs else 0
    lines: list[str] = []
    for k, v in pin_pairs:
        lines.append(f"{inner}{k.ljust(width)} = {v};")
    for k, v in size_pairs:
        lines.append(f"{inner}{k.ljust(width)} = {v};")
    return lines


def _emit_device_decl_block(
    dev: Device,
    indent: str,
    *,
    pdk_id: str | None = None,
) -> list[str]:
    """Emit one device declaration block.

    Dispatches on `dev.device_class`:
      * `cap` / `res` / `ind`  → `cap NAME { ... }` form (no kind qualifier)
      * everything else (`pmos`, `nmos`, future variants) →
        `device NAME : KIND { ... }` form

    The header / body opens on a new line; the body is indented one
    level deeper; the closing brace lines up with the header.
    """
    if dev.device_class == "native":
        from clay.frontend.native_spice_import import emit_native_device_clay

        if not pdk_id:
            raise ValueError("native device formatting requires an explicit file-level PDK")
        return emit_native_device_clay(dev, pdk_id=pdk_id, indent=indent).splitlines()
    passive = _passive_decl_kind(dev)
    if passive is not None:
        header = f"{indent}{passive} {dev.instance_name} {{"
        body = _emit_passive_body(dev, indent)
    else:
        header = (
            f"{indent}device {dev.instance_name} : "
            f"{dev.device_class} {{"
        )
        body = _emit_device_body(dev, indent)
    return [header] + body + [f"{indent}}}"]


def _emit_instance_decl_block(
    inst: CellInstance,
    indent: str,
) -> list[str]:
    """Emit one `instance NAME = pkg::Cell { port = net; ... };` block.

    The port map is emitted in source order — the parser preserves
    insertion order on `dict[str, str]`. Pin names left-aligned with
    width padding for visual consistency.
    """
    inner = indent + INDENT
    lines = [f"{indent}instance {inst.name} = {inst.cell_path} {{"]
    items = list(inst.port_map.items())
    if items:
        width = max(len(k) for k, _ in items)
        for sub_port, outer_net in items:
            lines.append(f"{inner}{sub_port.ljust(width)} = {outer_net};")
    lines.append(f"{indent}}};")
    return lines


def _emit_role_decl_block(
    role: RoleDecl,
    indent: str,
) -> list[str]:
    """Emit one `<role_kw> NAME { ... }` LEAN role declaration block.

    Sizing keys emit in canonical order (gm_id, W, L, M, nf). Single-
    net pin overrides emit per the role-kind's allowed-label order.
    Two-net pair overrides emit after, in the role-kind's pair-label
    order (only pair roles have these).
    """
    inner = indent + INDENT
    lines = [f"{indent}{role.kind} {role.name} {{"]

    # Build all LHS pairs upfront so we can width-align them.
    size_pairs: list[tuple[str, str]] = []
    for k in _ROLE_SIZE_KEY_ORDER:
        if k in role.sizing:
            v = role.sizing[k]
            # `gm_id` is dimensionless intent; W/L are length; M/nf
            # are integer multipliers. Reuse the existing sizing hints.
            dim = _SIZING_UNIT_HINTS.get(k)
            size_pairs.append((k, _format_number(float(v), dim)))
    # Any non-canonical sizing keys land last in source order.
    for k, v in role.sizing.items():
        if k in _ROLE_SIZE_KEY_ORDER:
            continue
        if not isinstance(v, (int, float)):
            continue
        size_pairs.append((k, _format_number(
            float(v), _SIZING_UNIT_HINTS.get(k),
        )))

    pin_pairs: list[tuple[str, str]] = []
    for label in _ROLE_PIN_LABEL_ORDER.get(role.kind, ()):
        if label in role.pin_overrides:
            pin_pairs.append((label, role.pin_overrides[label]))
    # Any out-of-order labels (defensive — parser only emits the
    # canonical ones).
    for label, net in role.pin_overrides.items():
        if label not in _ROLE_PIN_LABEL_ORDER.get(role.kind, ()):
            pin_pairs.append((label, net))

    pair_pairs: list[tuple[str, str]] = []
    for label in _ROLE_PAIR_LABEL_ORDER.get(role.kind, ()):
        if label in role.pair_overrides:
            net_a, net_b = role.pair_overrides[label]
            pair_pairs.append((label, f"({net_a}, {net_b})"))
    for label, pair in role.pair_overrides.items():
        if label not in _ROLE_PAIR_LABEL_ORDER.get(role.kind, ()):
            net_a, net_b = pair
            pair_pairs.append((label, f"({net_a}, {net_b})"))

    all_pairs = size_pairs + pin_pairs + pair_pairs
    width = max(len(k) for k, _ in all_pairs) if all_pairs else 0
    for k, v in size_pairs:
        lines.append(f"{inner}{k.ljust(width)} = {v};")
    for k, v in pin_pairs:
        lines.append(f"{inner}{k.ljust(width)} = {v};")
    for k, v in pair_pairs:
        lines.append(f"{inner}{k.ljust(width)} = {v};")
    lines.append(f"{indent}}}")
    return lines


def _emit_topology_block(
    topology: dict[str, object],
    indent: str,
    *,
    instances: list[CellInstance] | None = None,
    devices: list[Device] | None = None,
    role_decls: list[RoleDecl] | None = None,
    pdk_id: str | None = None,
) -> list[str]:
    """Emit a `topology { ... }` block.

    Items inside the block come from four sibling axes on the parsed
    `CellDef`:
      * `topology` dict — the legacy `style = '...'; rationale = ...;`
        stage-form fields (v0.0.19+)
      * `instances`     — sub-cell instantiations (v0.0.28 composition)
      * `devices`       — device primitives, including caps/resistors/
        inductors and the expanded forms of digital primitives
        (v0.0.28, v0.0.34, v0.0.35)
      * `role_decls`    — LEAN role-typed device declarations
        (v0.0.31)

    Role-expanded devices (whose names start with `<role.name>_`) are
    filtered out when role_decls are present, because the role declaration
    itself will re-expand them at parse time. Emitting both forms would
    cause re-parse to produce twice as many devices as the original IR.

    v0.0.37 (clay_format.py fix): before this version, only the
    topology dict was emitted — devices, instances, and roles silently
    vanished from the formatter output. This regression is now caught
    by `tests/test_clay_format.py::test_device_primitives_round_trip`
    and its siblings.
    """
    instances = instances or []
    devices = devices or []
    role_decls = role_decls or []
    inner = indent + INDENT

    # Bail out fast on the all-empty case so the output stays compact.
    if not topology and not instances and not devices and not role_decls:
        return [indent + "topology { }"]

    lines = [indent + "topology {"]

    # 1. Stage-form key/value fields (style / rationale / freeform).
    if topology:
        keys = list(topology.keys())
        width = max(len(k) for k in keys)
        for k in keys:
            v = topology[k]
            rhs = _format_topology_value(v, inner)
            lines.append(f"{inner}{k.ljust(width)} = {rhs};")

    # 2. Sub-cell instances. Each instance is its own block — no
    #    width-alignment shared with the topology dict above.
    for inst in instances:
        lines.extend(_emit_instance_decl_block(inst, inner))

    # 3. Device primitives. Skip role-expanded ones — the role_decls
    #    re-emitted below will expand to the same devices at parse time.
    role_names = {r.name for r in role_decls}
    for dev in devices:
        if _is_role_expanded_device(dev, role_names):
            continue
        lines.extend(_emit_device_decl_block(dev, inner, pdk_id=pdk_id))

    # 4. LEAN role declarations.
    for role in role_decls:
        lines.extend(_emit_role_decl_block(role, inner))

    lines.append(indent + "}")
    return lines


def _emit_bias_tree_block(
    bias_tree: list[BiasNode],
    indent: str,
) -> list[str]:
    """Emit a `bias_tree { ... }` block (v0.0.26 / G2).

    Each `BiasNode` renders as one of:
      * `NAME: external = <value with current unit>;`
      * `NAME: pmos_diode from <SRC>;`     (or nmos_diode)
      * `NAME: pmos_mirror of <SRC> by <M>;` (or nmos_mirror)

    Nodes emit in source order — the parser preserves insertion order
    on `list[BiasNode]`. Width-aligned LHS for visual consistency with
    the existing `bias_tree_demo.clay` style.
    """
    if not bias_tree:
        return []
    inner = indent + INDENT
    width = max(len(n.name) for n in bias_tree)
    # The RHS keyword (`external` / `pmos_diode` / etc.) gets a
    # secondary width-align so the values / source-refs line up too.
    rhs_kw_width = max(len(n.kind) for n in bias_tree)
    lines = [indent + "bias_tree {"]
    for n in bias_tree:
        name_pad = (n.name + ":").ljust(width + 1)
        kind_pad = n.kind.ljust(rhs_kw_width)
        if n.kind == "external":
            val_str = _format_number(float(n.value), "current")
            lines.append(f"{inner}{name_pad} {kind_pad} = {val_str};")
        elif n.kind in ("pmos_diode", "nmos_diode"):
            lines.append(f"{inner}{name_pad} {kind_pad} from {n.src};")
        elif n.kind in ("pmos_mirror", "nmos_mirror"):
            lines.append(
                f"{inner}{name_pad} {kind_pad} of {n.src} by "
                f"{int(n.by_M)};"
            )
        else:
            # Defensive — parser only emits the five kinds above. Fall
            # through to a comment so the user sees the dropped node
            # in the diff instead of silently losing it.
            lines.append(
                f"{inner}// (unrenderable bias node "
                f"{n.name}: kind={n.kind!r})"
            )
    lines.append(indent + "}")
    return lines


def _emit_corners_block(corners: list[str], indent: str) -> list[str]:
    """Emit a `corners { tt; ss; ... }` block.

    Single-corner blocks stay on one line; multi-corner blocks use one
    corner per line for readability and for stable diffs.
    """
    if len(corners) <= 1:
        return [indent + "corners { " + "; ".join(corners) + "; }"]
    inner = indent + INDENT
    lines = [indent + "corners {"]
    for c in corners:
        lines.append(inner + c + ";")
    lines.append(indent + "}")
    return lines


def _emit_temps_block(temps: list[float], indent: str) -> list[str]:
    """Emit a `temps { 27C; 85C; ... }` block.

    Temperatures are always emitted with a `C` suffix (decorative —
    parser strips it). Integer values render bare-integer (`27C`);
    non-integer values render with general format (`27.5C`).
    """
    def fmt_temp(t: float) -> str:
        if float(t).is_integer():
            return f"{int(t)}C"
        return f"{t:g}C"
    if len(temps) <= 1:
        return [indent + "temps { " + "; ".join(fmt_temp(t) for t in temps) + "; }"]
    inner = indent + INDENT
    lines = [indent + "temps {"]
    for t in temps:
        lines.append(inner + fmt_temp(t) + ";")
    lines.append(indent + "}")
    return lines


def _emit_cell_def(
    cd: CellDef,
    cf: ClayFile,
) -> list[str]:
    """Emit a `cell NAME(ports) { ... }` block.

    Pulls corners / temps off the `_corners` / `_temps` private slot
    that `clay.parser._ClayTransformer` stashes when the cell_def's
    body had a `corners` / `temps` block. Default values are skipped.
    """
    # Render the port list: bare `name` or `name: role`.
    port_strs: list[str] = []
    for p in cd.ports:
        role = cd.port_roles.get(p)
        if role is None:
            port_strs.append(p)
        else:
            port_strs.append(f"{p}: {role}")
    # Multi-line port list if any port carries a role, to mirror the
    # convention used in `examples/port_roles_demo.clay` and `showcase.clay`.
    has_roles = any(r is not None for r in cd.port_roles.values())
    if has_roles and len(port_strs) > 1:
        port_block = "(\n" + ",\n".join(INDENT + s for s in port_strs) + "\n)"
    else:
        port_block = "(" + ", ".join(port_strs) + ")"

    header = f"cell {cd.local_name}{port_block} {{"
    lines = [header]
    inner = INDENT

    # Canonical body order: topology, spec, sizing, bias_tree, corners,
    # temps. The topology block carries four sibling axes:
    #   topology dict + instances + devices + role_decls.
    # Any of them being non-empty forces the block to emit.
    has_topology = bool(
        cd.topology or cd.instances or cd.devices or cd.role_decls
    )
    if has_topology:
        lines.extend(_emit_topology_block(
            cd.topology, inner,
            instances=cd.instances,
            devices=cd.devices,
            role_decls=cd.role_decls,
            pdk_id=cf.pdk_id,
        ))
    if cd.spec.fields:
        lines.extend(_emit_spec_block(cd.spec, inner))
    if cd.sizing_kwargs:
        lines.extend(_emit_sizing_block("sizing", cd.sizing_kwargs, inner))
    if cd.bias_tree:
        lines.extend(_emit_bias_tree_block(cd.bias_tree, inner))

    corners = cd.__dict__.get("_corners")
    if corners is not None and corners != DEFAULT_CORNERS:
        lines.extend(_emit_corners_block(corners, inner))
    temps = cd.__dict__.get("_temps")
    if temps is not None and temps != DEFAULT_TEMPS:
        lines.extend(_emit_temps_block(temps, inner))

    lines.append("}")
    return lines


def _emit_layout_intent_block(cd: CellDef) -> list[str]:
    """Emit a top-level `layout_intent <Cell> { ... }` block."""
    decls = list(getattr(cd, "layout_intents", []) or [])
    if not decls:
        return []
    lines = [f"layout_intent {cd.local_name} {{"]
    inner = INDENT
    for decl in decls:
        lines.append(inner + _format_layout_intent_decl(decl))
    lines.append("}")
    return lines


def _format_policy_value(value: object) -> str:
    """Preserve the grammar's identifier/number tokens and identifier lists."""
    if isinstance(value, tuple):
        if not value or any(not isinstance(item, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", item)
                            for item in value):
            raise ValueError("policy lists require nonempty identifier tuples")
        return "(" + ", ".join(value) + ")"
    if type(value) in (int, float):
        if not math.isfinite(value):
            raise ValueError("policy numbers must be finite")
        return repr(value)
    if isinstance(value, str) and re.fullmatch(
        r"(?:[A-Za-z_][A-Za-z_0-9]*|[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)", value
    ):
        return value
    raise ValueError(f"unsupported policy value {value!r}")


def _emit_policy_block(policy: ParsedPolicyDecl) -> list[str]:
    lines = [f"policy {policy.name} {{"]
    for name in ("place_policy", "route_policy", "converge"):
        fields = getattr(policy, name)
        if fields:
            entries = ", ".join(f"{key}={_format_policy_value(value)}" for key, value in fields.items())
            lines.append(f"{INDENT}@{name}({entries});")
    if type(policy.learn) is not bool:
        raise ValueError("policy learn must be a boolean")
    lines.append(f"{INDENT}@learn({'on' if policy.learn else 'off'});")
    lines.append("}")
    return lines


def _emit_flow_block(cd: CellDef) -> list[str]:
    flow = cd.flow_decl
    if flow is None:
        return []
    lines = [f"flow {cd.local_name} {{"]
    if flow.pdk is not None:
        lines.append(f"{INDENT}pdk {_format_string(flow.pdk)};")
    if flow.corners:
        lines.append(f"{INDENT}corners ({', '.join(flow.corners)});")
    if flow.steps:
        lines.append(f"{INDENT}steps ({', '.join(flow.steps)});")
    if flow.layout_module is not None:
        lines.append(f"{INDENT}layout = {flow.layout_module};")
    if flow.policy_ref is not None:
        lines.append(f"{INDENT}policy = {flow.policy_ref};")
    if len(lines) == 1:
        raise ValueError("cannot format an empty flow declaration")
    lines.append("}")
    return lines


def _format_layout_intent_decl(decl: ParsedLayoutIntentDecl) -> str:
    if decl.kind in {"imported_layout", "route_tail"}:
        import json
        from clay.layout.imported_route_intent import imported_routes_from_declarations, route_tail_from_declaration
        if decl.kind == "imported_layout":
            imported, _ = imported_routes_from_declarations((decl,))
            return (f"@imported_layout(manifest={json.dumps(imported.manifest)}, "
                    f"sha256={json.dumps(imported.sha256)});")
        tail = route_tail_from_declaration(decl)
        return f"@route_tail({tail.net}, resource={json.dumps(tail.resource)}, trim_nm={tail.trim_nm});"
    if decl.kind == "channel_route":
        attrs = decl.attributes
        fields = []
        for key in ("nets", "between", "trunk_layer", "branch_layer", "width", "pitch", "centered_nets"):
            if key in attrs:
                value = attrs[key]
                formatted = "(" + ", ".join(value) + ")" if isinstance(value, tuple) else str(value)
                fields.append(f"{key}={formatted}")
        if set(attrs) - {item.split("=", 1)[0] for item in fields}:
            raise ValueError("unknown channel_route field cannot be formatted")
        return "@channel_route(" + ", ".join(fields) + ");"
    if decl.kind == "channel_branch":
        return f"@channel_branch({decl.nets[0]}, layer={decl.attributes['layer']});"
    a, b = decl.nets if decl.nets else ("", "")
    if decl.kind == "match":
        scope = decl.attributes.get("scope")
        if scope is not None and scope != "route":
            raise ValueError("explicit @match scope must be route")
        return f"@match({a}, {b}{', scope=route' if scope else ''});"
    if decl.kind == "match_length":
        return f"@match_length({a}, {b});"
    if decl.kind == "symmetric":
        return f"@symmetric({a}, {b});"
    if decl.kind == "mirror":
        return f"@mirror({a}, {b});"
    if decl.kind == "budget":
        from clay.layout.intent import parasitic_budget_driver_attributes

        tol = _format_number(float(decl.rel_tol), None)
        drivers = parasitic_budget_driver_attributes(decl)
        options = []
        if "aggressors" in drivers:
            options.append(f"aggressors=({', '.join(drivers['aggressors'])})")
        if "clock_net" in drivers:
            options.append(f"clock_net={drivers['clock_net']}")
        suffix = ", " + ", ".join(options) if options else ""
        return f"@budget({decl.budget_kind}, {a}, {b}, {tol}{suffix});"
    if decl.kind == "shield":
        return (
            f"@shield({decl.aggressor} -> {decl.victim}, "
            f"{decl.shield_net});"
        )
    return f"// (unrenderable layout intent kind={decl.kind!r})"


def _emit_cell_alias(ca: CellAlias) -> list[str]:
    """Emit a `cell NAME = pkg::Cell { spec; sizing_override; };`."""
    lines = [f"cell {ca.local_name} = {ca.parent_path} {{"]
    inner = INDENT
    lines.extend(_emit_spec_block(ca.spec, inner))
    if ca.sizing_overrides:
        lines.extend(_emit_sizing_block(
            "sizing_override", ca.sizing_overrides, inner,
        ))
    lines.append("};")
    return lines


def _verify_job_matches_alias(vj: VerifyJob, ca: CellAlias) -> bool:
    """True if `vj` can be re-expressed as `verify ALIAS;` against `ca`.

    Matches require: same `cell_path`, structurally equivalent spec,
    and identical sizing_overrides. Corners / temps must also be at
    default (the alias form does not carry corners / temps — they
    would silently get the per-job default).
    """
    if vj.cell_path != ca.parent_path:
        return False
    if not _specs_equivalent(vj.spec, ca.spec):
        return False
    if vj.sizing_overrides != ca.sizing_overrides:
        return False
    if vj.corners != DEFAULT_CORNERS:
        return False
    if vj.temps != DEFAULT_TEMPS:
        return False
    return True


def _specs_equivalent(a: Spec, b: Spec) -> bool:
    """Order-insensitive structural equality on two Spec objects."""
    if len(a.fields) != len(b.fields):
        return False
    by_name_a = {f.name: f for f in a.fields}
    by_name_b = {f.name: f for f in b.fields}
    if set(by_name_a) != set(by_name_b):
        return False
    for name, fa in by_name_a.items():
        fb = by_name_b[name]
        if fa.dimension != fb.dimension or fa.op != fb.op:
            return False
        if not math.isclose(fa.value, fb.value, rel_tol=1e-12):
            return False
        if fa.soft != fb.soft:
            return False
    return True


def _emit_verify_job(
    vj: VerifyJob,
    cf: ClayFile,
) -> list[str]:
    """Emit either a `verify NAME;` reference or an inline `verify { ... }`.

    Resolution rules (order matters):
      1. `cell_path` of `__celldef__::NAME` with NO setups / derived
         measures -> `verify NAME;` (simple cell-def reference). The
         cell_def already carries the spec, so the shorthand form
         loses no information.
      2. `cell_path` of `__celldef__::NAME` WITH setups / derived
         measures -> inline `verify __celldef__::NAME { setup ... }`.
         The multi-setup body cannot be carried by the shorthand
         `verify NAME;` form, so we emit the full sentinel-qualified
         block. This was v0.0.33's enabling fix for cell_def + multi-
         setup verify (the cell-def shorthand drops setup info, so
         the formatter now preserves it).
      3. Matches a CellAlias's parent_path with equivalent spec /
         sizing_overrides AND non-default corners/temps -> `verify NAME;`.
      4. Otherwise -> inline verify block.
    """
    # 1. / 2. CellDef reference (sentinel form). Without setups /
    # derived measures we use the compact `verify NAME;` shorthand;
    # with them we MUST preserve the full sentinel-qualified inline
    # block so the multi-setup verify body survives a round-trip.
    if vj.cell_path.startswith("__celldef__::"):
        local_name = vj.cell_path[len("__celldef__::"):]
        if not vj.setups and not vj.derived_measures:
            return [f"verify {local_name};"]
        # Multi-setup verify against a cell_def: emit inline form
        # with the sentinel-qualified cell_path. Falls through to
        # the generic inline emitter below.
    else:
        # 3. CellAlias reference.
        for ca in cf.cell_aliases:
            if _verify_job_matches_alias(vj, ca):
                return [f"verify {ca.local_name};"]
    # 4. Inline verify block (the generic case).
    lines = [f"verify {vj.cell_path} {{"]
    inner = INDENT
    if vj.spec.fields:
        lines.extend(_emit_spec_block(vj.spec, inner))
    if vj.sizing_overrides:
        lines.extend(_emit_sizing_block(
            "sizing_override", vj.sizing_overrides, inner,
        ))
    if vj.corners != DEFAULT_CORNERS:
        lines.extend(_emit_corners_block(vj.corners, inner))
    if vj.temps != DEFAULT_TEMPS:
        lines.extend(_emit_temps_block(vj.temps, inner))
    # v0.0.33: multi-setup verify body. Each setup gets its own
    # `setup NAME { drive { ... } [spec { ... }] }` block, then
    # any `measure NAME := <expr>;` derived-measure declarations
    # land after the setups.
    for s in vj.setups:
        lines.extend(_emit_setup_block(s, inner))
    for dm_name, dm_expr in vj.derived_measures.items():
        lines.append(f"{inner}measure {dm_name} := {dm_expr};")
    lines.append("}")
    return lines


def _emit_setup_block(setup, indent: str) -> list[str]:
    """Emit a `setup NAME { drive { ... } [spec { ... }] }` block.

    Mirrors the canonical surface form: drive block first (required),
    optional per-setup spec block second. Drive values are emitted
    using the same numeric formatter as spec fields (no unit
    inflection because drive values are dimensionless to the
    grammar — they're just floats).
    """
    inner = indent + INDENT
    lines = [f"{indent}setup {setup.name} {{"]
    # drive block — single key = value. Drive values are bare floats
    # at the grammar level (no unit suffix carried through; the
    # rewriter interprets them per-drive-mode), so we use the
    # dimension-free formatting branch of `_format_number`.
    drive_val = _format_number(setup.drive_value, None)
    lines.append(f"{inner}drive {{ {setup.drive_mode} = {drive_val}; }}")
    if setup.spec is not None and setup.spec.fields:
        lines.extend(_emit_spec_block(setup.spec, inner))
    lines.append(f"{indent}}}")
    return lines


def format_clay_file(cf: ClayFile) -> str:
    """Walk a ClayFile IR and return a canonical formatted source string.

    The output ends with exactly one trailing newline. Statements are
    separated by exactly one blank line — no leading blank, no double
    blanks.
    """
    blocks: list[list[str]] = []

    # 1. File header block — package + pdk on adjacent lines, matching
    #    every existing example. No blank line between them; they are
    #    a single semantic unit (file identity + PDK pin).
    header_lines: list[str] = []
    if cf.package_name:
        header_lines.append(f"package {cf.package_name};")
    if cf.pdk_id:
        header_lines.append(f"pdk {_format_string(cf.pdk_id)};")
    if header_lines:
        blocks.append(header_lines)

    # 2. Imports — sorted alphabetically, one block (no blank between
    #    individual imports) for compactness.
    if cf.imports:
        import_lines = [f"import {name};" for name in sorted(cf.imports)]
        blocks.append(import_lines)

    for policy in cf.policies.values():
        blocks.append(_emit_policy_block(policy))

    # 4. Cell aliases — source order, blank between each.
    for ca in cf.cell_aliases:
        blocks.append(_emit_cell_alias(ca))

    # 5. Cell definitions — source order, blank between each.
    for cd in cf.cell_defs:
        blocks.append(_emit_cell_def(cd, cf))
        flow_block = _emit_flow_block(cd)
        if flow_block:
            blocks.append(flow_block)
        layout_block = _emit_layout_intent_block(cd)
        if layout_block:
            blocks.append(layout_block)

    # 6. Verify jobs — source order. Consecutive single-line `verify
    #    NAME;` references are grouped into one block (no intervening
    #    blank lines) for readability, matching the showcase example.
    #    Inline verify blocks always get their own block (blank lines
    #    around them).
    current_group: list[str] = []
    for vj in cf.verify_jobs:
        emitted = _emit_verify_job(vj, cf)
        if len(emitted) == 1:
            current_group.extend(emitted)
        else:
            if current_group:
                blocks.append(current_group)
                current_group = []
            blocks.append(emitted)
    if current_group:
        blocks.append(current_group)

    if not blocks:
        return "\n"
    return "\n\n".join("\n".join(block) for block in blocks) + "\n"


def format_clay_source(text: str) -> str:
    """Parse a Clay source string and return its canonical re-emission."""
    return format_clay_file(parse_clay_source(text))


# ─── CLI ─────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="clay_format",
        description=(
            "Canonical formatter for .clay files. Parses the input, "
            "then re-emits a canonical text representation. "
            "Comments are dropped (lark strips them at the tokenizer "
            "layer)."
        ),
    )
    p.add_argument("path", help="path to a .clay file")
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "-i", "--in-place",
        action="store_true",
        help="rewrite the file in place",
    )
    g.add_argument(
        "--check",
        action="store_true",
        help=(
            "exit 0 if the file is already canonically formatted, "
            "exit 1 if it would change (CI mode)"
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    path = Path(args.path)
    original = path.read_text(encoding="utf-8")
    formatted = format_clay_source(original)
    if args.check:
        if original == formatted:
            return 0
        print(f"would reformat: {path}", file=sys.stderr)
        return 1
    if args.in_place:
        if original != formatted:
            path.write_text(formatted, encoding="utf-8")
        return 0
    # Default: print to stdout.
    sys.stdout.write(formatted)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
