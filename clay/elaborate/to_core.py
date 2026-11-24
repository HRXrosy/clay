"""Desugar a clay.ir.Cell to clean-room core .clay text.

Pure, no I/O. ``emit_cell(cell)`` returns a string accepted by the clean-room
kernel's ``parseDesign`` (Clay/Compiler/Parser.lean). See package docstring
for the target surface.

Design contract (verified against Parser.lean):
  * Cell is FIRST and exactly one (leaf or node). constraints block optional,
    after the cell. No realize block emitted (C2 needs a drawing).
  * interface ports: ``<role> <name>``; role in {in,out,bias,supply,clock}.
  * composition: ``composition { nets { <ident,...> } <inst>* }``. nets MUST
    list every bind-RHS net AND every node interface port name (C1 boundary
    consistency + net scoping). This is the elaborator's hard obligation.
  * instance: ``inst <cell> bind { <port>=<net>, ... }``; child inlined.
  * model literal: bare ident if simple, else double-quoted.
  * constraints: ``constraint <stratum> <domain> <checker> "<pred>"``;
    stratum in {core,representation,oracle}; domain/checker idents; pred quoted.

Lossy projections (documented, C1-safe):
  * 7 EDALLM port roles -> 5 kernel roles (power_p/power_n/ground -> supply).
  * sizing / bias / functional-role / intent constraints carry [intent] text;
    they ROUTE but are not checked obligations (advisory).
  * orchestration (flow/quality/policy/verify) emits NOTHING structural.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


# ─────────────────────────────────────────────────────────────────────
# Deferred-record bookkeeping (honest out-of-core accounting).
# ─────────────────────────────────────────────────────────────────────


@dataclass
class DeferredRecord:
    """What the elaborator saw but did NOT emit into the structural core.

    Orchestration families (flow/quality/policy/verify/corners/temps/fn/
    topology.style/alias/import) are process, not structure. They are
    recorded here, never coerced into the 2-primitive core.
    """

    notes: list[str] = field(default_factory=list)

    def add(self, construct: str, reason: str) -> None:
        self.notes.append(f"{construct}: {reason}")


# ─────────────────────────────────────────────────────────────────────
# Lexical helpers — keep everything inside the kernel's ident/string grammar.
# Kernel idents: first char isAlpha|'_', body isAlpha|isDigit|'_'|'.'.
# Kernel strings: double-quoted, NO escapes (lexer reads body to next '"').
# ─────────────────────────────────────────────────────────────────────

_IDENT_BODY_OK = re.compile(r"[A-Za-z0-9_.]")
_SIMPLE_MODEL = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")


def _sanitize_ident(s: str) -> str:
    """Map an arbitrary EDALLM net/port name to a valid kernel ident.

    Kernel idents allow [A-Za-z0-9_.], first char must be alpha or '_'.
    Illegal chars -> '_'. Empty / digit-leading -> prefix 'n_'.
    """
    s = str(s)
    out = []
    for ch in s:
        out.append(ch if _IDENT_BODY_OK.match(ch) else "_")
    res = "".join(out)
    if not res:
        return "n_anon"
    first = res[0]
    if not (first.isalpha() or first == "_"):
        res = "n_" + res
    return res


def _esc_pred(s: str) -> str:
    """Render a predicate as a kernel double-quoted string body.

    The kernel lexer has no escape handling: a string is everything between
    the opening quote and the next ``"``. So internal double-quotes must be
    stripped, and newlines flattened. Spaces/brackets/operators are fine
    inside a quoted string.
    """
    s = str(s).replace('"', "'")
    s = s.replace("\r", " ").replace("\n", " ")
    return s


def _model_lit(s: str) -> str:
    """Emit a model literal: bare ident if simple, else double-quoted."""
    s = str(s)
    if _SIMPLE_MODEL.match(s):
        return s
    return '"' + _esc_pred(s) + '"'


# ─────────────────────────────────────────────────────────────────────
# Port-role projection: 7 EDALLM roles -> 5 kernel keywords.
# ─────────────────────────────────────────────────────────────────────

# Explicit EDALLM role -> kernel keyword. power_p/power_n/ground collapse to
# supply (lossy but sound: roles are advisory below C1).
_ROLE_MAP = {
    "input": "in",
    "inout": "in",
    "output": "out",
    "power_p": "supply",
    "power_n": "supply",
    "ground": "supply",
    "supply": "supply",
    "bias": "bias",
    "clock": "clock",
    # tolerate already-kernel keywords passing through
    "in": "in",
    "out": "out",
}

_SUPPLY_NAMES = {"vdd", "vcc", "vss", "vee", "gnd", "vbb", "vnn", "vpp", "agnd", "dgnd"}
_BIAS_NAMES = {"vbn", "vbp", "vb", "ibias", "iref", "vbias", "nbias", "pbias", "vcm"}
_CLOCK_NAMES = {"clk", "clock", "clkb", "clk_n", "clk_p", "phi", "phi1", "phi2"}


def _role_kw(port_name: str, role_str: Optional[str]) -> str:
    """Map an EDALLM port to a kernel role keyword.

    If an explicit role is given, use the projection table. Otherwise infer
    from the port name (supply rails / bias / clock / in* / out* / default in).
    """
    if role_str is not None:
        mapped = _ROLE_MAP.get(str(role_str).lower())
        if mapped is not None:
            return mapped
    nm = str(port_name).lower()
    if nm in _SUPPLY_NAMES:
        return "supply"
    if nm in _CLOCK_NAMES:
        return "clock"
    if nm in _BIAS_NAMES:
        return "bias"
    if nm.startswith("out"):
        return "out"
    if nm.startswith("in"):
        return "in"
    # default: treat unknown leaf nets as signal-in
    return "in"


# Device-pin -> kernel role for the synthesized leaf interface of one Device.
# MOS pins d/g/s/b ; passive pins p/n.
_PIN_ROLE = {
    "g": "in",
    "d": "out",
    "s": "supply",
    "b": "supply",
    "p": "in",
    "n": "out",
    "vb": "bias",
    "vbias": "bias",
}


def _pin_role(pin: str) -> str:
    return _PIN_ROLE.get(str(pin).lower(), "in")


# ─────────────────────────────────────────────────────────────────────
# Structural emission: leaf / device-inst / node / child-inst.
# ─────────────────────────────────────────────────────────────────────


def _port_roles_of(cell: Any) -> dict[str, Any]:
    """Sidecar dict[port_name -> role_str|None] stashed by the compiler."""
    return dict(getattr(cell, "_port_roles", None) or cell.__dict__.get("_port_roles", {}) or {})


def emit_leaf(cell: Any, model: Optional[str] = None) -> str:
    """A pure leaf device cell: ``leaf interface {...} model "<class>"``."""
    roles = _port_roles_of(cell)
    ports = list(getattr(cell, "ports", []) or [])
    port_lines = []
    for p in ports:
        kw = _role_kw(p, roles.get(p))
        port_lines.append(f"{kw} {_sanitize_ident(p)}")
    iface = ",\n    ".join(port_lines)
    model_name = model if model is not None else getattr(cell, "name", "leaf_cell")
    body = "interface {\n    " + iface + "\n  }" if port_lines else "interface {}"
    return f'leaf {body} model {_model_lit(str(model_name))}'


def _emit_device_inst(dev: Any) -> tuple[str, list[str]]:
    """One Device -> ``inst leaf interface {...} model "<class>" bind {...}``.

    Returns (text, nets_referenced). Bind order = dev.pins insertion order.
    """
    pins: dict[str, str] = dict(getattr(dev, "pins", {}) or {})
    cls = str(getattr(dev, "device_class", "device"))
    variant = getattr(dev, "pdk_variant", None)
    model = f"{cls}/{variant}" if variant else cls

    iface_parts = []
    bind_parts = []
    nets: list[str] = []
    for pin, net in pins.items():
        snet = _sanitize_ident(net)
        iface_parts.append(f"{_pin_role(pin)} {_sanitize_ident(pin)}")
        bind_parts.append(f"{_sanitize_ident(pin)} = {snet}")
        nets.append(snet)
    iface = "interface { " + ", ".join(iface_parts) + " }" if iface_parts else "interface {}"
    bind = "bind { " + ", ".join(bind_parts) + " }" if bind_parts else "bind {}"
    text = f"inst leaf {iface} model {_model_lit(model)} {bind}"
    return text, nets


def _emit_child_inst(
    inst: Any,
    resolve: Optional[Callable[[str], Any]],
    deferred: DeferredRecord,
) -> tuple[str, list[str]]:
    """A CellInstance -> ``inst <inlined child cell> bind { <subport>=<net> }``.

    The child cell is recursively emitted (inlined) so the kernel sees one
    self-contained Cell tree. port_map order is preserved.
    """
    port_map: dict[str, str] = dict(getattr(inst, "port_map", {}) or {})
    cell_path = getattr(inst, "cell_path", None)

    child_cell = None
    if resolve is not None and cell_path is not None:
        try:
            child_cell = resolve(cell_path)
        except Exception as exc:  # noqa: BLE001 — honest deferral on resolve fail
            deferred.add(
                f"instance {getattr(inst, 'name', '?')} ({cell_path})",
                f"child resolve failed: {exc}",
            )

    if child_cell is None:
        # Cannot inline an unresolved child; emit an opaque leaf placeholder
        # whose interface is the port_map keys (so binds are well-formed) and
        # record the loss honestly.
        deferred.add(
            f"instance {getattr(inst, 'name', '?')} ({cell_path})",
            "child cell IR not resolvable; emitted opaque leaf placeholder",
        )
        roles = {}
        iface_parts = [f"{_role_kw(p, None)} {_sanitize_ident(p)}" for p in port_map]
        iface = "interface { " + ", ".join(iface_parts) + " }" if iface_parts else "interface {}"
        child_text = f'leaf {iface} model {_model_lit(str(cell_path or "child"))}'
    else:
        child_text = _emit_cell_inner(child_cell, resolve, deferred, indent="  ")

    bind_parts = []
    nets: list[str] = []
    for subport, outernet in port_map.items():
        snet = _sanitize_ident(outernet)
        bind_parts.append(f"{_sanitize_ident(subport)} = {snet}")
        nets.append(snet)
    bind = "bind { " + ", ".join(bind_parts) + " }" if bind_parts else "bind {}"
    return f"inst {child_text} {bind}", nets


def _collect_nets(cell: Any, device_nets: list[str], child_nets: list[str]) -> list[str]:
    """Stable union of: all node interface port names + every bind-RHS net.

    Order: interface ports first (in cell.ports order), then any additional
    nets discovered in device/child binds, de-duplicated, stable.
    """
    seen: dict[str, None] = {}
    ordered: list[str] = []

    def push(n: str) -> None:
        if n not in seen:
            seen[n] = None
            ordered.append(n)

    for p in getattr(cell, "ports", []) or []:
        push(_sanitize_ident(p))
    for n in device_nets:
        push(n)
    for n in child_nets:
        push(n)
    return ordered


def emit_node(
    cell: Any,
    resolve: Optional[Callable[[str], Any]] = None,
    deferred: Optional[DeferredRecord] = None,
) -> str:
    """A cell with devices and/or sub-instances -> node + composition."""
    return _emit_node_inner(cell, resolve, deferred or DeferredRecord(), indent="")


def _emit_node_inner(
    cell: Any,
    resolve: Optional[Callable[[str], Any]],
    deferred: DeferredRecord,
    indent: str,
) -> str:
    roles = _port_roles_of(cell)
    ports = list(getattr(cell, "ports", []) or [])
    port_lines = [f"{_role_kw(p, roles.get(p))} {_sanitize_ident(p)}" for p in ports]
    iface = ",\n    ".join(port_lines)
    iface_block = "interface {\n    " + iface + "\n  }" if port_lines else "interface {}"

    inst_texts: list[str] = []
    device_nets: list[str] = []
    for dev in getattr(cell, "devices", []) or []:
        t, nets = _emit_device_inst(dev)
        inst_texts.append(t)
        device_nets.extend(nets)

    child_nets: list[str] = []
    for inst in getattr(cell, "instances", []) or []:
        t, nets = _emit_child_inst(inst, resolve, deferred)
        inst_texts.append(t)
        child_nets.extend(nets)

    nets = _collect_nets(cell, device_nets, child_nets)
    nets_block = "nets { " + ", ".join(nets) + " }"
    insts_joined = ",\n    ".join(inst_texts)
    comp = (
        "composition {\n    "
        + nets_block
        + ("\n    " + insts_joined if inst_texts else "")
        + "\n  }"
    )
    return f"node {iface_block} {comp}"


def _emit_cell_inner(
    cell: Any,
    resolve: Optional[Callable[[str], Any]],
    deferred: DeferredRecord,
    indent: str,
) -> str:
    """Structural part only (no constraints) — used for recursive inlining."""
    has_devices = bool(getattr(cell, "devices", []) or [])
    has_instances = bool(getattr(cell, "instances", []) or [])
    if has_devices or has_instances:
        return _emit_node_inner(cell, resolve, deferred, indent)
    # No internal structure -> opaque leaf.
    return emit_leaf(cell)


# ─────────────────────────────────────────────────────────────────────
# Constraint emission: spec / sizing / bias / annotate / intent / role.
# ─────────────────────────────────────────────────────────────────────


def _fmt_value(v: float) -> str:
    """SI float -> compact string (no trailing-zero noise, keep magnitude)."""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return str(v)
    f = float(v)
    if f == int(f) and abs(f) < 1e15:
        # keep integers integral but large/small magnitudes scientific
        if abs(f) >= 1e6 or (0 < abs(f) < 1e-3):
            return repr(f)
        return str(int(f))
    return repr(f)


def spec_constraints(cell: Any) -> list[str]:
    """Each SpecField -> ``constraint oracle electrical <name> "<pred>"``."""
    lines: list[str] = []
    spec = getattr(cell, "spec", None)
    fields = getattr(spec, "fields", None) or []
    for f in fields:
        name = _sanitize_ident(getattr(f, "name", "spec"))
        op = getattr(f, "op", ">=")
        val = _fmt_value(getattr(f, "value", 0.0))
        pred = f"{getattr(f, 'name', name)} {op} {val}"
        corners = getattr(f, "across_corners", None)
        if corners:
            pred += " @corners[" + ",".join(str(c) for c in corners) + "]"
        margin = getattr(f, "margin_pct", None)
        if margin is not None:
            pred += f" margin={_fmt_value(margin)}%"
        lines.append(
            f'constraint oracle electrical {name} "{_esc_pred(pred)}"'
        )
    return lines


def resource_feasibility_constraint(cell: Any) -> list[str]:
    """Resource fields (gbw/cl/vdd/power) in the Spec -> ONE C3 conservation line.

    Emits ``constraint oracle resource feasibility "gbw=.. cl=.. vdd=.. power=.."``
    -- the exact (stratum=oracle, domain=resource, checker=feasibility) locus the
    clean-room Main.lean ``isResourceFeasibility`` routes to the conservation kernel,
    and the exact ``key=val`` pred the ``clay.c3.adjudicate --spec`` parser consumes
    (whitespace-split, ``float(val)``). Only fields actually present are emitted;
    absent ones are omitted. Returns [] if no resource field is present.

    Field detection is by NAME keyword (not by dimension), so it works whether the
    source writes the type_id as ``freq``/``cap``/``volt`` or anything else. ``gbw``,
    ``vdd``, ``power`` use substring containment; ``cl`` requires an exact match
    (``nm == "cl"``) to avoid false positives from ``clock``/``recall``/``class``.
    """
    spec = getattr(cell, "spec", None)
    if not spec:
        return []
    fields = getattr(spec, "fields", None) or []
    # name-keyword -> canonical resource key (order = emission order)
    _RES_KEYS = ("gbw", "cl", "vdd", "power")
    found: dict[str, str] = {}
    for f in fields:
        nm = str(getattr(f, "name", "")).lower()
        for key in _RES_KEYS:
            if key in found:
                continue
            if key == "cl":
                hit = nm == "cl"
            else:
                hit = key in nm
            if hit:
                found[key] = _fmt_value(getattr(f, "value", 0.0))
                break
    if not found:
        return []
    pred = " ".join(f"{key}={found[key]}" for key in _RES_KEYS if key in found)
    return [f'constraint oracle resource feasibility "{_esc_pred(pred)}"']


# knobs that are electrical sizing intent vs structural multiplier intent
_STRUCT_SIZING = ("n_", "multiplier", "fingers", "m_", "nf_")


def sizing_constraints(sizing_kwargs: Optional[dict[str, Any]]) -> list[str]:
    """gm_id_*/Cc/Rz/Cz -> oracle electrical; N_*/multiplier -> core structural."""
    lines: list[str] = []
    if not sizing_kwargs:
        return lines
    for k, v in sizing_kwargs.items():
        kk = str(k)
        lk = kk.lower()
        val = _fmt_value(v) if isinstance(v, (int, float)) else _esc_pred(str(v))
        if lk.startswith(_STRUCT_SIZING):
            lines.append(
                f'constraint core structural device_multiplier '
                f'"{_esc_pred(kk)} = {val} [intent]"'
            )
        else:
            lines.append(
                f'constraint oracle electrical sizing_{_sanitize_ident(kk)} '
                f'"{_esc_pred(kk)} = {val} [intent]"'
            )
    return lines


def bias_tree_constraint(cell: Any) -> list[str]:
    """bias_tree -> one ``constraint core structural bias_topology "bias_chain(...)"``."""
    bt = getattr(cell, "bias_tree", None) or []
    if not bt:
        return []
    names = []
    for node in bt:
        nm = getattr(node, "name", None) or getattr(node, "out_net", None)
        if nm:
            names.append(str(nm))
    if not names:
        return []
    chain = ",".join(names)
    return [
        f'constraint core structural bias_topology "bias_chain({_esc_pred(chain)})"'
    ]


def _annotation_pred(device_id: str, ann: Any) -> Optional[tuple[str, str]]:
    """Map one LayoutAnnotation -> (kind, predicate_string). None if unknown."""
    cn = type(ann).__name__

    def _enumval(x: Any) -> str:
        return str(getattr(x, "value", x))

    if cn == "Matching":
        return ("matching", f"matching_pair({device_id},{getattr(ann,'other','?')},{_enumval(getattr(ann,'kind',''))})")
    if cn == "GuardRing":
        return ("guard_ring", f"has_guard_ring({device_id},{_enumval(getattr(ann,'well_type',''))})")
    if cn == "WellIsolated":
        return ("well_isolated", f"well_isolated({device_id},{getattr(ann,'from_device','?')})")
    if cn == "Symmetric":
        devs = getattr(ann, "devices", []) or [device_id]
        axis = _enumval(getattr(ann, "axis", "vertical"))
        return ("symmetry", f"symmetric([{','.join(str(d) for d in devs)}],{axis})")
    if cn == "Interdigitated":
        return ("interdigitated", f"interdigitated({device_id},{getattr(ann,'with_device','?')},{_enumval(getattr(ann,'pattern',''))})")
    if cn == "Adjacent":
        return ("adjacent", f"adjacent({device_id},{getattr(ann,'to','?')})")
    if cn == "Bulk":
        return ("bulk", f"bulk_net({device_id},{getattr(ann,'net','?')})")
    if cn == "Tap":
        return ("tap", f"tap({device_id},{getattr(ann,'net','?')},{getattr(ann,'where','?')})")
    return None


def annotate_constraints(cell: Any) -> list[str]:
    """device-id -> [LayoutAnnotation] -> representation geometric <kind>."""
    anns = cell.__dict__.get("_layout_annotations", None) or {}
    lines: list[str] = []
    for device_id, ann_list in anns.items():
        for ann in ann_list or []:
            res = _annotation_pred(str(device_id), ann)
            if res is None:
                continue
            kind, pred = res
            lines.append(
                f'constraint representation geometric {kind} "{_esc_pred(pred)}"'
            )
    return lines


def layout_intent_constraints(cell: Any) -> list[str]:
    """ParsedLayoutIntentDecl (@match/@symmetric/@shield/@budget/@mirror/...) ->
    representation geometric <kind>; @objective -> representation timing;
    @area -> representation geometric area_budget."""
    intents = getattr(cell, "layout_intents", None) or []
    lines: list[str] = []
    for it in intents:
        kind = str(getattr(it, "kind", "intent")).lstrip("@")
        nets = getattr(it, "nets", ()) or ()
        if kind in ("match", "symmetric", "mirror") and nets:
            pred = f"{kind}({','.join(str(n) for n in nets)})"
            lines.append(f'constraint representation geometric {kind} "{_esc_pred(pred)}"')
        elif kind == "shield":
            agg = getattr(it, "aggressor", None) or getattr(it, "victim", None)
            sh = getattr(it, "shield_net", "gnd")
            pred = f"shield({agg},{sh})"
            lines.append(f'constraint representation geometric shield "{_esc_pred(pred)}"')
        elif kind == "budget":
            bk = getattr(it, "budget_kind", "coupling")
            tol = getattr(it, "rel_tol", None)
            pred = f"budget({bk}," + ",".join(str(n) for n in nets) + (f",{tol}" if tol is not None else "") + ")"
            lines.append(f'constraint representation geometric budget "{_esc_pred(pred)}"')
        elif kind == "objective":
            goal = getattr(it, "goal", None) or getattr(it, "target", None) or kind
            lines.append(f'constraint representation timing objective "{_esc_pred(str(goal))} [intent]"')
        elif kind == "area":
            lv = getattr(it, "limit_value", None)
            pred = f"area_budget({lv})" if lv is not None else "area_budget"
            lines.append(f'constraint representation geometric area_budget "{_esc_pred(pred)} [intent]"')
        elif kind == "place_group":
            grp = getattr(it, "group", None) or "group"
            devs = getattr(it, "devices", ()) or ()
            pred = f"place_group({grp},[{','.join(str(d) for d in devs)}])"
            lines.append(f'constraint representation geometric place_group "{_esc_pred(pred)} [intent]"')
        else:
            # generic fallback: carry kind + nets as intent text
            tail = ",".join(str(n) for n in nets)
            pred = f"{kind}({tail})" if tail else kind
            lines.append(f'constraint representation geometric {_sanitize_ident(kind)} "{_esc_pred(pred)} [intent]"')
    return lines


def functional_role_constraints(cell: Any) -> list[str]:
    """ParsedFunctionalRole -> representation function <kind> "<name>: <attrs> [intent]"."""
    roles = getattr(cell, "functional_roles", None) or []
    lines: list[str] = []
    for r in roles:
        kind = _sanitize_ident(str(getattr(r, "kind", "role")))
        name = str(getattr(r, "name", "?"))
        attrs = getattr(r, "attributes", {}) or {}
        attr_str = ", ".join(f"{k}={v}" for k, v in attrs.items())
        pred = f"{name}: {attr_str} [intent]" if attr_str else f"{name}: [intent]"
        lines.append(f'constraint representation function {kind} "{_esc_pred(pred)}"')
    return lines


def emit_constraints(
    cell: Any,
    sizing_kwargs: Optional[dict[str, Any]] = None,
    deferred: Optional[DeferredRecord] = None,
) -> str:
    """Gather + dedupe all constraint lines, wrap in a constraints block.

    Returns '' (no block) if there are no constraints.
    """
    dref = deferred or DeferredRecord()
    sk = sizing_kwargs
    if sk is None:
        sk = cell.__dict__.get("sizing_kwargs", None) or cell.__dict__.get("_sizing_kwargs", None)

    lines: list[str] = []
    lines += spec_constraints(cell)
    lines += resource_feasibility_constraint(cell)   # C3 conservation feasibility
    lines += sizing_constraints(sk)
    lines += bias_tree_constraint(cell)
    lines += annotate_constraints(cell)
    lines += layout_intent_constraints(cell)
    lines += functional_role_constraints(cell)

    # Honest deferral: orchestration is never emitted as structure.
    for attr, label in (
        ("flow", "flow_decl"),
        ("flows", "flow_decl"),
        ("quality", "quality_decl"),
        ("qualities", "quality_decl"),
        ("policies", "policy_decl"),
        ("verify_blocks", "verify_block"),
    ):
        if cell.__dict__.get(attr):
            dref.add(label, "process orchestration -> out of structural core")

    # de-dupe, preserve order
    seen: dict[str, None] = {}
    deduped: list[str] = []
    for ln in lines:
        if ln not in seen:
            seen[ln] = None
            deduped.append(ln)

    if not deduped:
        return ""
    body = ",\n    ".join(deduped)
    return "constraints {\n    " + body + "\n  }"


# ─────────────────────────────────────────────────────────────────────
# Top dispatcher.
# ─────────────────────────────────────────────────────────────────────


def emit_cell(
    cell: Any,
    resolve: Optional[Callable[[str], Any]] = None,
    sizing_kwargs: Optional[dict[str, Any]] = None,
    deferred: Optional[DeferredRecord] = None,
    leaf_model: Optional[str] = None,
) -> str:
    """Desugar one clay.ir.Cell to parseDesign-accepted core text.

    Args:
      cell:          a post-compile clay.ir.Cell (devices populated) or CellDef.
      resolve:       optional ``cell_path -> child Cell`` for hierarchical inlining.
      sizing_kwargs: optional sizing-intent dict (else read from cell sidecar).
      deferred:      optional DeferredRecord to accumulate out-of-core notes.
      leaf_model:    optional model name to force a pure-leaf emission.

    Returns:
      core .clay text (cell first, then optional constraints block).
    """
    dref = deferred or DeferredRecord()
    has_devices = bool(getattr(cell, "devices", []) or [])
    has_instances = bool(getattr(cell, "instances", []) or [])

    if leaf_model is not None or (not has_devices and not has_instances):
        cell_text = emit_leaf(cell, model=leaf_model)
    else:
        cell_text = _emit_node_inner(cell, resolve, dref, indent="")

    constraints_text = emit_constraints(cell, sizing_kwargs=sizing_kwargs, deferred=dref)

    name = getattr(cell, "name", "cell")
    header = f"# desugared core for cell '{name}' (clay.elaborate.to_core)\n"
    parts = [cell_text]
    if constraints_text:
        parts.append(constraints_text)
    return header + "\n".join(parts) + "\n"
