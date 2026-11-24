"""Emit a ``design.clay`` from a :class:`RoleDecomposition`.

This is the CLAY-EMIT stage of the front-end paradigm mapper. A ``design.clay``
is **two distinct layers** and this emitter renders BOTH:

    Layer A -- the COMPLETE device-level netlist (a lossless projection of the
               source netlist). EVERY source device is declared with its kind,
               terminals -> nets, and W/L/M verbatim. This layer is an enforced
               INVARIANT (see :mod:`clay.frontend.clay_emit_check`): a device is
               NEVER dropped for lacking a role.
    Layer B -- the ADDITIVE, partial intent overlay (functional_group roles,
               @symmetric / @matching, layout_intent budgets). Best-effort
               inference; it annotates a subset of the already-declared devices.

Surface form::

    package <package>;
    pdk "<pdk>";
    cell <cell>(<port: kind, ...>) {
      topology {
        device <name> : <kind> { drain=..; gate=..; source=..; body=..; W=..; L=..; M=..; }  // Layer A
        res <name>  { p=..; n=..; value=..; }                                                 // Layer A
        cap <name>  { p=..; n=..; value=..; }                                                 // Layer A
        ...
        functional_group <role> { role/group/label/devices/intent }                           // Layer B
        ...
      }
    }
    layout_intent <cell> { ... }   // skeleton, only when derivable (Layer B)
    flow <cell> { ... }
    quality <cell> { @require(drc_clean); }
    annotate <cell> {
      @cell: @symmetric(vertical);   // when mirror_axis == "vertical" (Layer B)
      <devA>: @matching(<devB>);     // per matched pair (both directions)
    }

Iteration is over the DEVICE list for Layer A and over the ROLE list for Layer B
-- never the reverse. A device with no inferred role is still fully declared in
Layer A; it simply carries no Layer-B annotation.

The emitter NEVER fabricates KPI thresholds or parasitic budgets. Where a
concrete value is unknown it emits a ``// TODO`` marker for a human reviewer,
keeping optional blocks minimal.

The hard contract: the emitted string PARSES via ``clay.parser.parse_clay_file``,
``place_generic`` seats EVERY declared device, and the declared netlist is a
LOSSLESS projection of the source (``assert_design_clay_complete`` GREEN) with
faithful sizes (``assert_size_fidelity`` GREEN).
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Sequence, Tuple

from clay.frontend.role_model import (
    DeviceCard,
    FunctionalRoleHit,
    RoleDecomposition,
)
from clay.layout.extract.netlist_intent import model_oxide

if TYPE_CHECKING:
    from clay.frontend.craft_infer import AutoCraftPlan

__all__ = ["emit_design_clay", "infer_port_kind"]


# A small, generic net-name -> port-kind map. NOT a per-circuit table: it
# matches by net-name stem so it generalises across analog cells. The kinds
# are exactly the Clay `_VALID_PORT_ROLES` set the transformer enforces.
_PORT_KIND_RULES: Tuple[Tuple[Tuple[str, ...], str], ...] = (
    (("vinp", "vip", "inp", "vin_p", "vplus", "vp", "vref"), "input"),
    (("vinn", "vin", "inn", "vin_n", "vminus", "vm"), "input"),
    (("vout", "out", "vo", "outp", "outn"), "output"),
    (("vdd", "vcc", "avdd", "dvdd", "vddd", "vdda"), "power_p"),
    (("vss", "gnd", "gnda", "vgnd", "agnd", "dgnd"), "power_n"),
    (("sub", "substrate"), "ground"),
    (("ibias", "iref", "ibn", "ibp", "bias", "vbias", "vb"), "bias"),
)


def infer_port_kind(net: str) -> str:
    """Infer a Clay port kind from a net name.

    Generic, stem-based: exact-stem match first (so ``vin`` is treated as the
    negative input rather than swallowed by an ``in`` substring), then a
    substring fallback. Unknown nets default to ``inout`` (a valid port role)
    so the draft still parses; the human reviewer retypes it.
    """
    low = net.strip().lower()
    # Exact-stem pass.
    for stems, kind in _PORT_KIND_RULES:
        if low in stems:
            return kind
    # Substring fallback (e.g. "vinp_buf", "out_main", "vdd33").
    for stems, kind in _PORT_KIND_RULES:
        for stem in stems:
            if stem in low:
                return kind
    return "inout"


def _humanize(token: str) -> str:
    """Turn ``input_pair`` into ``Input pair`` for a readable label."""
    words = token.replace("-", "_").split("_")
    words = [w for w in words if w]
    if not words:
        return token
    first = words[0].capitalize()
    return " ".join([first] + words[1:])


def _escape(text: str) -> str:
    """Escape a string for a double-quoted Clay literal.

    Clay string literals are ``"[^"]*"`` — no escapes — so we strip embedded
    double quotes and collapse newlines rather than emit an unparseable token.
    """
    return text.replace('"', "'").replace("\n", " ").replace("\r", " ").strip()


def _intent_for(hit: FunctionalRoleHit) -> str:
    """A short, non-fabricated intent string for a role.

    Prefers the inferred ``evidence`` (the structural reason the role matched)
    so the draft carries real provenance rather than invented prose.
    """
    evidence = _escape(hit.evidence or "")
    if evidence:
        return evidence
    return f"{_humanize(hit.role)} (review intent)"


def _unique_group_names(roles: Sequence[FunctionalRoleHit]) -> List[str]:
    """Assign a unique functional_group identifier per role hit.

    Multiple hits of the same role token get a numeric suffix so the emitted
    ``functional_group <name>`` identifiers stay distinct (the grammar keys
    the block on a CNAME).
    """
    counts: Dict[str, int] = {}
    for hit in roles:
        counts[hit.role] = counts.get(hit.role, 0) + 1
    seen: Dict[str, int] = {}
    names: List[str] = []
    for hit in roles:
        base = hit.role
        if counts[base] == 1:
            names.append(base)
            continue
        seen[base] = seen.get(base, 0) + 1
        names.append(f"{base}_{seen[base]}")
    return names


# === Layer A: complete device-level netlist ==================================
#
# Iterate the FULL device list (every source device) and emit one parseable
# device-level block per device. Completeness is an INVARIANT here: a device is
# never gated on whether a role referenced it.

# Canonical SPICE engineering suffixes the Clay number grammar accepts (lower-
# case k/u/n/p/f plus the `meg` mega prefix). The source netlist writes mixed-
# case tokens (`54.2122K`, `534.05f`, `200u`); we normalise the suffix to the
# grammar's canonical lowercase form WITHOUT changing the magnitude, so the
# emitted size is byte-faithful to the source value (the fidelity checker re-
# parses both sides through the same SI scale).
_SUFFIX_NORM: Dict[str, str] = {
    "t": "T",   # tera -> the grammar has no bare 't'; scale explicitly below
    "g": "G",
    "meg": "meg",
    "x": "meg",
    "k": "k",
    "u": "u",
    "n": "n",
    "p": "p",
    "f": "f",
}

# Grammar-recognised suffixes (clay/parser_modules/_constants.py _UNITS). A token
# whose suffix is NOT in this set is rendered as a bare resolved magnitude so the
# draft still parses.
_GRAMMAR_SUFFIXES = {"k", "u", "n", "p", "f", "meg"}

# SI scale of each SPICE suffix, for the rare suffix the grammar cannot express
# directly (tera / giga): we fold it into a bare metres/farads/ohms magnitude.
_SI_SCALE: Dict[str, float] = {
    "t": 1e12, "g": 1e9, "meg": 1e6, "x": 1e6,
    "k": 1e3, "m": 1e-3, "u": 1e-6, "n": 1e-9, "p": 1e-12, "f": 1e-15, "a": 1e-18,
}

_SIZE_TOKEN_RE = re.compile(r"^\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*([a-zA-Z]*)\s*$")

# Parse-safe placeholder for a MANDATORY size field whose source token is
# symbolic / unresolved. It is intentionally NOT a plausible real geometry so the
# fidelity checker reports it as unfaithful rather than it masquerading as a real
# size; the device is still declared + placeable.
_SIZE_PLACEHOLDER = "1u"


def _norm_size_token(tok: str) -> str:
    """Normalise a SPICE size token to a grammar-parseable Clay number.

    ``54.2122K`` -> ``54.2122k``; ``534.05f`` -> ``534.05f``; ``200u`` -> ``200u``;
    ``40`` -> ``40``. A token with a suffix the Clay grammar does not list (tera /
    giga / atto) is folded into a bare SI magnitude. A symbolic / unparseable
    token returns ``""`` (the caller substitutes a parse-safe placeholder for a
    mandatory MOS field, which the fidelity checker then reports as unfaithful).
    """
    if tok is None:
        return ""
    s = str(tok).strip()
    if not s:
        return ""
    m = _SIZE_TOKEN_RE.match(s)
    if not m:
        # Symbolic token (a `.param` name like ``W_TAIL`` / ``{W_diff}``). The
        # Clay device_size_field grammar requires a NUMBER, so emitting
        # ``W = W_TAIL`` would mis-lex as a pin assignment. Return "" so the size
        # field is omitted (the device is still declared + placed; the fidelity
        # checker reports the unresolved size honestly rather than the draft
        # failing to parse). Literal-sized netlists (the LDO) are unaffected.
        return ""
    mant, suf = m.group(1), m.group(2)
    if not suf:
        return mant
    low = suf.lower()
    # `meg` (3-letter mega) and `x` are special-cased before single-letter logic.
    if low in ("meg", "x"):
        return f"{mant}meg"
    first = low[0]
    if first in _GRAMMAR_SUFFIXES:
        return f"{mant}{first}"
    # suffix the grammar cannot name -> fold to a bare SI magnitude.
    scale = _SI_SCALE.get(first)
    if scale is None:
        return mant
    try:
        return repr(float(mant) * scale)
    except ValueError:
        return mant


def _device_kind_token(card: DeviceCard) -> Optional[str]:
    """The Clay ``device_class`` token for a MOS card (generic ``n*``/``p*`` rule).

    The :class:`DeviceCard` kind is already resolved (``nmos`` / ``pmos``) by the
    composed MOS reader. This is the generic rule -- never a per-circuit branch.
    Passives (res / cap) and unsupported kinds return ``None`` (handled by their
    own emit paths).
    """
    k = (card.kind or "").strip().lower()
    base: Optional[str] = None
    if k.startswith("n"):
        base = "nmos"
    elif k.startswith("p") and k != "pnp":
        base = "pmos"
    if base is None:
        return None

    # ``kind`` is intentionally only the polarity.  Preserve the independent
    # voltage-domain fact from the source model so n18/p18 and n33/p33 do not
    # collapse to one primitive.  Hand-authored cards without a source model
    # retain an explicit ``*3v`` kind when supplied.
    if card.source_model:
        thick = model_oxide(card.source_model) == "thick"
    else:
        thick = "3v" in k or "3p3" in k
    if thick:
        return base + "3v"
    return base


def _resolve_token_with_params(
    tok: str,
    sizing_params: Optional[Mapping[str, str]],
    *,
    is_mos_dim: bool = False,
) -> str:
    """Resolve symbolic SPICE tokens via ``sizing_params``; keep literals untouched
    unless resolving a unitless micron parameter for a MOS W/L dimension."""
    norm = _norm_size_token(tok)
    if norm:
        if is_mos_dim and sizing_params is not None and re.fullmatch(r"\d+(?:\.\d+)?", norm):
            # Bare micron MOS width/length in netlists with `.option scale=1.0u`
            v = float(norm)
            if 0.08 <= v <= 10000.0:
                return f"{norm}u"
        return norm
    if not tok or not sizing_params:
        return ""
    s = str(tok).strip()
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1].strip()
    # Direct lookup (case-insensitive fallback)
    cand = sizing_params.get(s)
    if cand is None:
        s_low = s.lower()
        for k, v in sizing_params.items():
            if str(k).lower() == s_low:
                cand = v
                break
    if cand is not None:
        norm_cand = _norm_size_token(cand)
        if norm_cand:
            if is_mos_dim and re.fullmatch(r"\d+(?:\.\d+)?", norm_cand):
                v = float(norm_cand)
                if 0.08 <= v <= 10000.0:
                    return f"{norm_cand}u"
            return norm_cand
    # Expression fallback
    try:
        from clay.frontend.op_enrich import resolve_netlist_params

        ev = resolve_netlist_params(f".param _res = {s}\n", sizing_params).get("_res", "")
        res_tok = _norm_size_token(ev)
        if is_mos_dim and res_tok and re.fullmatch(r"\d+(?:\.\d+)?", res_tok):
            v = float(res_tok)
            if 0.08 <= v <= 10000.0:
                return f"{res_tok}u"
        return res_tok
    except Exception:  # noqa: BLE001
        return ""


def _emit_mos(
    card: DeviceCard,
    sizing_params: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    """One ``device <name> : <kind> { ... }`` block, or ``None`` if not a MOS."""
    kind = _device_kind_token(card)
    if kind is None:
        return None
    pins = card.terminals
    # long-form pin names (the human/LLM-friendly surface the grammar accepts).
    d = pins.get("d", pins.get("drain", ""))
    g = pins.get("g", pins.get("gate", ""))
    s = pins.get("s", pins.get("source", ""))
    b = pins.get("b", pins.get("body", pins.get("bulk", "")))
    fields = [
        f"      device {card.name} : {kind} {{",
        f"        drain = {d}; gate = {g}; source = {s}; body = {b};",
    ]
    w = _resolve_token_with_params(card.w, sizing_params, is_mos_dim=True)
    l = _resolve_token_with_params(card.l, sizing_params, is_mos_dim=True)
    m = _resolve_token_with_params(card.m, sizing_params, is_mos_dim=False)
    # W and L are MANDATORY for a MOS device decl. When the source carries a
    # literal size (the LDO), it is emitted verbatim. When the source size is
    # SYMBOLIC (a `.param` name the role model preserves, e.g. `W_TAIL`), and
    # cannot be resolved via sizing_params, a parse-safe placeholder is emitted.
    unresolved: List[str] = []
    if not w:
        unresolved.append(f"W({card.w})")
        w = _SIZE_PLACEHOLDER
    if not l:
        unresolved.append(f"L({card.l})")
        l = _SIZE_PLACEHOLDER
    sizes = [f"W = {w};", f"L = {l};"]
    if m:
        sizes.append(f"M = {m};")
    fields.append("        " + " ".join(sizes))
    if unresolved:
        fields.append(
            f"        // TODO: source size(s) {', '.join(unresolved)} are "
            f"symbolic/unresolved -- placeholder emitted; resolve to the real "
            f".param value (fidelity checker flags this)."
        )
    fields.append("      }")
    return "\n".join(fields)


def _emit_passive(
    card: DeviceCard,
    sizing_params: Optional[Mapping[str, str]] = None,
    *,
    inferred_res_model: Optional[str] = None,
) -> Optional[str]:
    """One ``res`` / ``cap`` block, or ``None`` if the card is not a 2-terminal
    passive. The value (resistance / capacitance) rides the source ``w`` slot
    (that is how the role model carries a passive's magnitude)."""
    k = (card.kind or "").strip().lower()
    if k in ("res", "resistor", "r"):
        kw = "res"
    elif k in ("cap", "capacitor", "c", "mim"):
        kw = "cap"
    else:
        return None
    pins = card.terminals
    p = pins.get("p", pins.get("plus", ""))
    n = pins.get("n", pins.get("minus", ""))
    value = _resolve_token_with_params(card.w, sizing_params)
    m = _resolve_token_with_params(card.m, sizing_params)
    lines = [
        f"      {kw} {card.name} {{",
        f"        p = {p}; n = {n};",
    ]
    if value:
        lines.append(f"        value = {value};")
    else:
        # No magnitude -> the draft still parses with a placeholder; the
        # completeness checker still sees the device declared.
        lines.append("        value = 0;")
        lines.append("        // TODO: source value missing for this passive")
    if kw == "res":
        res_model = str(card.source_model or "").strip() or str(inferred_res_model or "").strip()
        if res_model:
            lines.append(f"        model = {res_model};")
    # The SPICE inline multiplier ``m=N`` means N parallel unit devices. Carry
    # it on the passive (mirror the MOS path) so the value stays byte-faithful
    # to the source AND the realized magnitude is correct (cap C = value*M,
    # res R = value/M). ``m == 1`` is the identity and is left implicit.
    if m and m not in ("1", "1.0"):
        lines.append(f"        M = {m};")
    lines.append("      }")
    return "\n".join(lines)


def _emit_bjt(card: DeviceCard) -> Optional[str]:
    """Emit portable three-terminal BJT intent and source-model provenance."""
    kind = (card.kind or "").strip().lower()
    if kind == "bjt":
        model = str(card.source_model or "").lower()
        if "pnp" in model:
            kind = "pnp"
        elif "npn" in model:
            kind = "npn"
    if kind not in ("pnp", "npn"):
        return None
    pins = card.terminals
    c = pins.get("c", pins.get("collector", ""))
    b = pins.get("b", pins.get("base", ""))
    e = pins.get("e", pins.get("emitter", ""))
    model = str(card.source_model or "").strip()
    area = _norm_size_token(card.area) or "1"
    multiplier = _norm_size_token(card.m)
    lines = [
        f"      device {card.name} : {kind} {{",
        f"        collector = {c}; base = {b}; emitter = {e};",
    ]
    if model:
        lines.append(f"        model = {model};")
    lines.append(f"        area = {area};")
    if multiplier and multiplier not in ("1", "1.0"):
        lines.append(f"        M = {multiplier};")
    lines.append("      }")
    return "\n".join(lines)


def _emit_device(
    card: DeviceCard,
    sizing_params: Optional[Mapping[str, str]] = None,
    *,
    res_models_by_dev: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    """Emit one Layer-A device block for any card kind, or ``None`` if the kind
    has no Clay primitive (e.g. a BJT -- declared kinds the grammar cannot carry
    are deliberately NOT silently dropped here; they surface as a named RED in
    ``assert_design_clay_complete`` rather than a silent loss)."""
    block = _emit_mos(card, sizing_params=sizing_params)
    if block is not None:
        return block
    inferred_model = res_models_by_dev.get(card.name) if res_models_by_dev else None
    block = _emit_passive(card, sizing_params=sizing_params, inferred_res_model=inferred_model)
    if block is not None:
        return block
    return _emit_bjt(card)


def _device_block(
    devices: Sequence[DeviceCard],
    sizing_params: Optional[Mapping[str, str]] = None,
    *,
    res_models_by_dev: Optional[Mapping[str, str]] = None,
) -> List[str]:
    """Layer A: a parseable device declaration for EVERY device in the netlist."""
    lines: List[str] = []
    for card in devices:
        block = _emit_device(card, sizing_params=sizing_params, res_models_by_dev=res_models_by_dev)
        if block is not None:
            lines.append(block)
    return lines


def _emit_intent(roles: Sequence[FunctionalRoleHit]) -> List[str]:
    """Layer B: functional_group annotations over ALREADY-declared devices.

    Best-effort: a role that matched no device, or a device that matched no role,
    is simply unannotated. This NEVER affects which devices are declared.
    """
    if not roles:
        return []
    group_names = _unique_group_names(roles)
    lines: List[str] = []
    for name, hit in zip(group_names, roles):
        label = _humanize(hit.role)
        devices = ", ".join(hit.devices)
        intent = _intent_for(hit)
        lines.append(f"    functional_group {name} {{")
        lines.append(f"      role = {hit.role},")
        lines.append(f"      group = {hit.role},")
        lines.append(f'      label = "{_escape(label)}",')
        lines.append(f'      group_label = "{_escape(label)}",')
        lines.append(f"      devices = ({devices}),")
        lines.append(f'      intent = "{intent}"')
        lines.append("    }")
    return lines


def _topology_block(
    devices: Sequence[DeviceCard],
    roles: Sequence[FunctionalRoleHit],
    sizing_params: Optional[Mapping[str, str]] = None,
    *,
    res_models_by_dev: Optional[Mapping[str, str]] = None,
    craft_plan: Optional["AutoCraftPlan"] = None,
) -> str:
    """The cell's ``topology {}`` block: the COMPLETE device netlist (Layer A)
    followed by the ADDITIVE functional-group overlay (Layer B).

    Layer A alone satisfies the grammar's ``topology_item+`` requirement, so a
    cell with no inferred roles still parses (and places every device).
    """
    device_lines = _device_block(devices, sizing_params=sizing_params, res_models_by_dev=res_models_by_dev)
    intent_lines = _emit_intent(roles)
    if craft_plan is not None:
        declared_names = {d.name for d in devices}
        for arr in getattr(craft_plan, "shared_od_arrays", ()) or ():
            if not getattr(arr, "needs_functional_group", False):
                continue
            members = [m for m in arr.devices if m in declared_names]
            if len(members) < 2:
                continue
            intent_lines.append(f"    functional_group {arr.group_name} {{")
            intent_lines.append("      role = current_mirror,")
            intent_lines.append("      group = current_mirror,")
            intent_lines.append('      label = "Current mirror",')
            intent_lines.append('      group_label = "Current mirror",')
            intent_lines.append(f"      devices = ({', '.join(members)}),")
            intent_lines.append('      intent = "uniform-unit shared-diffusion mirror array"')
            intent_lines.append("    }")

    lines: List[str] = ["  topology {"]
    if device_lines:
        lines.extend(device_lines)
    if intent_lines:
        lines.extend(intent_lines)
    if not device_lines and not intent_lines:
        # Neither a declarable device nor a role -> a placeholder group so the
        # draft still parses and the reviewer sees the gap.
        lines.append("    // TODO: no declarable devices or roles for this cell")
        lines.append("    functional_group unclassified {")
        lines.append("      role = unclassified,")
        lines.append("      group = unclassified,")
        lines.append('      label = "Unclassified devices",')
        lines.append('      group_label = "Unclassified devices",')
        lines.append('      intent = "review: role inference produced no hits"')
        lines.append("    }")
    lines.append("  }")
    return "\n".join(lines)


def _annotate_block(
    decomp: RoleDecomposition,
    *,
    craft_plan: Optional["AutoCraftPlan"] = None,
) -> str:
    """Emit the annotate block: @cell symmetric axis + @matching + @fold + @well_isolated.

    Each matched pair is emitted in BOTH directions (``A: @matching(B);`` and
    ``B: @matching(A);``) to mirror the committed design.clay convention so the
    parsed ``_layout_annotations`` recover a symmetric matching relation.
    """
    entries: List[str] = []
    axis_raw = craft_plan.mirror_axis if craft_plan is not None else decomp.mirror_axis
    if axis_raw:
        axis = axis_raw.strip().lower()
        # The grammar's symmetric axis literal is `vertical` | `horizontal`.
        if axis in ("vertical", "horizontal"):
            entries.append(f"  @cell: @symmetric({axis});")

    pairs_src = craft_plan.matched_pairs if craft_plan is not None else decomp.matched_pairs
    dev_map = {d.name: d for d in decomp.devices}

    seen: set[Tuple[str, str]] = set()
    for a, b in pairs_src:
        if a not in dev_map or b not in dev_map:
            continue
        for x, y in ((a, b), (b, a)):
            if (x, y) in seen or not x or not y:
                continue
            seen.add((x, y))
            entries.append(f"  {x}: @matching({y});")

    if craft_plan is not None:
        seen_dense: set[str] = set()
        for dev_name in getattr(craft_plan, "dense_contact_devices", ()) or ():
            if dev_name in dev_map and dev_name not in seen_dense:
                seen_dense.add(dev_name)
                entries.append(f"  {dev_name}: @dense_contacts;")

        seen_fold: set[str] = set()
        for fh in craft_plan.folding_hints:
            if fh.device in dev_map and fh.device not in seen_fold and fh.h_max_um > 0:
                seen_fold.add(fh.device)
                h_str = (
                    str(int(fh.h_max_um))
                    if float(fh.h_max_um).is_integer()
                    else f"{fh.h_max_um:g}"
                )
                entries.append(f"  {fh.device}: @fold({h_str});")

        seen_iso: set[str] = set()
        for dev_name, partner in craft_plan.well_isolated:
            card = dev_map.get(dev_name)
            if (
                card is not None
                and partner in dev_map
                and dev_name not in seen_iso
                and (card.kind or "").strip().lower().startswith("n")
            ):
                seen_iso.add(dev_name)
                entries.append(f"  {dev_name}: @well_isolated({partner});")

        declared_nets = {n for d in decomp.devices for n in d.terminals.values() if n}
        seen_taps: set[str] = set()
        for dev_name, tap_net, where in getattr(craft_plan, "device_taps", ()) or ():
            if dev_name in dev_map and tap_net in declared_nets and dev_name not in seen_taps:
                seen_taps.add(dev_name)
                entries.append(f"  {dev_name}: @tap({tap_net}, where={where});")

        seen_rseg: set[str] = set()
        for rs in getattr(craft_plan, "resistor_segments", ()) or ():
            if rs.device in dev_map and rs.device not in seen_rseg:
                seen_rseg.add(rs.device)
                entries.append(
                    f"  {rs.device}: @resistor_segments(pdk={rs.pdk}, count={rs.count}, "
                    f"connection={rs.connection}, width_um={rs.width_um:g}, length_um={rs.length_um:g}, "
                    f"segment_resistance_ohms={rs.segment_resistance_ohms:g}, "
                    f"value_quantum_ohms={rs.value_quantum_ohms:g});"
                )

    if not entries:
        return ""
    body = "\n".join(entries)
    return f"annotate {decomp.cell} {{\n{body}\n}}"


def _reference_place_groups(
    decomp: RoleDecomposition,
    reference_gds: str,
    source_netlist: str,
):
    """The REFERENCE-driven floorplan, or ``None`` when extraction is unavailable.

    Reads the reference layout's ABSTRACT arrangement (region tier / column order /
    banking / wide-block -- relations only, no coordinate) and routes every device
    to its region by PDK master. Returns ``None`` (caller falls back to the role-only
    floorplan) when the GDS cannot be read or yields no usable arrangement, so the
    role-only path stays the default for any cell without a reference layout.
    """
    try:
        from clay.frontend.reference_spatial_intent import (
            extract_reference_spatial_intent,
            device_master_map,
        )
        from clay.frontend.floorplan_intent import (
            floorplan_place_groups_from_reference,
        )

        ref_intent = extract_reference_spatial_intent(
            reference_gds, source_netlist=source_netlist
        )
        if not getattr(ref_intent, "region_tier", None):
            return None
        master_map = device_master_map(source_netlist)
        return floorplan_place_groups_from_reference(decomp, ref_intent, master_map)
    except Exception:  # noqa: BLE001 - extraction is best-effort; fall back to roles
        return None


def _clean_domain_place_groups(
    decomp: RoleDecomposition,
    raw_groups: Sequence,
    *,
    craft_plan: Optional["AutoCraftPlan"] = None,
):
    """Synthesize polarity-pure, domain-clean 2-D ``(row, col)`` floorplan groups
    from schematic topology roles and device classes (zero hardcoded circuit or
    device literals, zero coordinates).

    Fires whenever ``craft_plan`` is supplied (auto-craft onboarding) OR whenever
    the raw role-only rows mix MOS with non-MOS, mix NMOS and PMOS in the same
    row, or invert NMOS above PMOS alongside BJTs.
    """
    from statistics import median
    from clay.frontend.floorplan_intent import PlaceGroupRelation
    from clay.frontend.op_enrich import _canon_kind, _resolve_token_value

    if not raw_groups:
        return raw_groups

    dev_cards = list(decomp.devices)
    dev_kind = {d.name: _canon_kind(d) for d in dev_cards}
    has_bjt = any(k in ("pnp", "npn", "bjt") for k in dev_kind.values())

    def _domain_of(kind: str) -> str:
        if kind in ("nmos", "pmos"):
            return kind
        if kind in ("pnp", "npn", "bjt"):
            return "bjt"
        if kind in ("res", "cap"):
            return kind
        return "other"

    mixed_mos_nonmos = any(
        ({"nmos", "pmos"} & {_domain_of(dev_kind.get(dn, "")) for dn in pg.devices})
        and ({"bjt", "res", "cap"} & {_domain_of(dev_kind.get(dn, "")) for dn in pg.devices})
        for pg in raw_groups
    )
    mixed_nmos_pmos = any(
        ({"nmos", "pmos"} <= {_domain_of(dev_kind.get(dn, "")) for dn in pg.devices})
        for pg in raw_groups
    )
    nmos_rows = [
        pg.row for pg in raw_groups
        if any(_domain_of(dev_kind.get(dn, "")) == "nmos" for dn in pg.devices)
    ]
    pmos_rows = [
        pg.row for pg in raw_groups
        if any(_domain_of(dev_kind.get(dn, "")) == "pmos" for dn in pg.devices)
    ]
    inverted_with_bjt = has_bjt and bool(nmos_rows) and bool(pmos_rows) and min(nmos_rows) > max(pmos_rows)
    inverted_mos = bool(nmos_rows) and bool(pmos_rows) and min(nmos_rows) > max(pmos_rows)

    if craft_plan is None and not (mixed_mos_nonmos or inverted_with_bjt or (mixed_nmos_pmos and inverted_mos)):
        return raw_groups

    # Collect all roles per device so multi-role devices (e.g. a pass-FET bank
    # tagged both load_mirror and pass_device, or a tail FET tagged tail_current
    # and current_mirror) keep their structural signals cleanly.
    roles_of: Dict[str, set[str]] = {}
    primary_role: Dict[str, str] = {}
    for hit in decomp.roles:
        for dn in hit.devices:
            roles_of.setdefault(dn, set()).add(hit.role)
            primary_role.setdefault(dn, hit.role)

    matched_set = {n for pair in decomp.matched_pairs for n in pair}
    nmos_devs = [d for d in dev_cards if dev_kind[d.name] == "nmos"]
    pmos_devs = [d for d in dev_cards if dev_kind[d.name] == "pmos"]
    bjt_devs = sorted(d.name for d in dev_cards if _domain_of(dev_kind[d.name]) == "bjt")
    res_cards = [d for d in dev_cards if dev_kind[d.name] == "res"]
    cap_cards = [d for d in dev_cards if dev_kind[d.name] == "cap"]
    other_devs = sorted(
        d.name for d in dev_cards
        if _domain_of(dev_kind[d.name]) == "other"
    )

    def _l_um(card: DeviceCard) -> float:
        v = _resolve_token_value(card.l, {}, default_scale_um=True)
        return float(v) * 1e6 if v else 1.0

    def _r_ohm(card: DeviceCard) -> float:
        raw = getattr(card, "value", None) or getattr(card, "w", None)
        v = _resolve_token_value(raw, {}, default_scale_um=False)
        return float(v) if v else 0.0

    groups: List[PlaceGroupRelation] = []
    row_idx = 0

    # 1. NMOS bottom rail band (row=0): never emit 1-device singleton rows;
    #    split into 2 balanced (col=0, col=1) groups when both halves have >= 2 devices.
    if nmos_devs:
        n_names = [d.name for d in nmos_devs]
        n_input = [d.name for d in nmos_devs if "input_pair" in roles_of.get(d.name, ())]
        n_cascode = [d.name for d in nmos_devs if "cascode" in roles_of.get(d.name, ())]
        n_bias_tagged = {
            d.name for d in nmos_devs
            if roles_of.get(d.name, set()) & {"tail_current", "bias_tree"}
        }
        if n_input and n_cascode:
            # Signal-path NMOS (input pair + cascodes + matched signal mirrors) vs bias tree
            n_core_set = set(n_input) | set(n_cascode) | {
                d.name for d in nmos_devs
                if d.name in matched_set and d.name not in n_bias_tagged
            }
            n_col1 = sorted(n_core_set)
            n_col0 = sorted(n for n in n_names if n not in n_core_set)
            if len(n_col0) >= 2 and len(n_col1) >= 2:
                groups.append(PlaceGroupRelation("nmos_bias", tuple(n_col0), row=row_idx, col=0))
                groups.append(PlaceGroupRelation("nmos_core", tuple(n_col1), row=row_idx, col=1))
            else:
                groups.append(PlaceGroupRelation("nmos_core", tuple(sorted(n_names)), row=row_idx))
        elif len(n_input) >= 2:
            n_col1 = sorted(n_input)
            n_col0 = sorted(n for n in n_names if n not in set(n_col1))
            if len(n_col0) >= 2:
                groups.append(PlaceGroupRelation("nmos_bias", tuple(n_col0), row=row_idx, col=0))
                groups.append(PlaceGroupRelation("nmos_diff", tuple(n_col1), row=row_idx, col=1))
            else:
                groups.append(PlaceGroupRelation("nmos_core", tuple(sorted(n_names)), row=row_idx))
        else:
            # Bandgap / PMOS-input style: matched/load mirror NMOS vs startup/bias core NMOS
            n_load = sorted(
                d.name for d in nmos_devs
                if (
                    d.name in matched_set
                    or ("load_mirror" in roles_of.get(d.name, ()) and (has_bjt or _l_um(d) >= 4.0))
                )
            )
            n_core = sorted(n for n in n_names if n not in set(n_load))
            if len(n_core) >= 2 and len(n_load) >= 2:
                groups.append(PlaceGroupRelation("nmos_core", tuple(n_core), row=row_idx, col=0))
                groups.append(PlaceGroupRelation("nmos_load", tuple(n_load), row=row_idx, col=1))
            else:
                groups.append(PlaceGroupRelation("nmos_core", tuple(sorted(n_names)), row=row_idx))
        row_idx += 1

    # 2. PMOS upper MOS rail band:
    #    - If PMOS has both a cascode tier (>= 2) and a VDD-rail load tier (>= 2),
    #      place cascode (+ cascode bias) on the lower PMOS row (adjacent to NMOS)
    #      and rail-connected load mirrors on the upper PMOS row (adjacent to VDD).
    #    - Otherwise pack ALL PMOS onto ONE 2-D row split cleanly into (col=0, col=1)
    #      by pass-FET bank (>= 4) or input pair (>= 2), eliminating singleton/extra rows.
    if pmos_devs:
        p_names = [d.name for d in pmos_devs]
        p_cascode = sorted(d.name for d in pmos_devs if "cascode" in roles_of.get(d.name, ()))
        p_load_only = sorted(
            d.name for d in pmos_devs
            if d.name not in set(p_cascode)
            and bool(roles_of.get(d.name, set()) & {"load_mirror", "pass_device"})
        )
        p_pass = sorted(d.name for d in pmos_devs if "pass_device" in roles_of.get(d.name, ()))
        p_diff = sorted(d.name for d in pmos_devs if "input_pair" in roles_of.get(d.name, ()))

        if len(p_cascode) >= 2 and len(p_load_only) >= 2:
            p_bias = sorted(n for n in p_names if n not in set(p_cascode) and n not in set(p_load_only))
            if len(p_bias) >= 2:
                groups.append(PlaceGroupRelation("pmos_bias", tuple(p_bias), row=row_idx, col=0))
                groups.append(PlaceGroupRelation("pmos_cascode", tuple(p_cascode), row=row_idx, col=1))
            elif p_bias:
                groups.append(PlaceGroupRelation("pmos_cascode", tuple(sorted(p_bias + p_cascode)), row=row_idx))
            else:
                groups.append(PlaceGroupRelation("pmos_cascode", tuple(p_cascode), row=row_idx))
            row_idx += 1
            groups.append(PlaceGroupRelation("pmos_load", tuple(p_load_only), row=row_idx))
            row_idx += 1
        elif len(p_pass) >= 4:
            p_core = sorted(n for n in p_names if n not in set(p_pass))
            if len(p_core) >= 2:
                groups.append(PlaceGroupRelation("pmos_core", tuple(p_core), row=row_idx, col=0))
                groups.append(PlaceGroupRelation("pmos_pass", tuple(p_pass), row=row_idx, col=1))
            else:
                groups.append(PlaceGroupRelation("pmos_core", tuple(sorted(p_names)), row=row_idx))
            row_idx += 1
        elif len(p_diff) >= 2:
            p_mirror = sorted(n for n in p_names if n not in set(p_diff))
            if len(p_mirror) >= 2:
                groups.append(PlaceGroupRelation("pmos_diff", tuple(p_diff), row=row_idx, col=0))
                groups.append(PlaceGroupRelation("pmos_mirror", tuple(p_mirror), row=row_idx, col=1))
            else:
                groups.append(PlaceGroupRelation("pmos_core", tuple(sorted(p_names)), row=row_idx))
            row_idx += 1
        else:
            groups.append(PlaceGroupRelation("pmos_core", tuple(sorted(p_names)), row=row_idx))
            row_idx += 1

    # 3. BJT substrate bank strictly above PMOS and strictly below passive banks.
    if bjt_devs:
        groups.append(PlaceGroupRelation("pnp_bank", tuple(bjt_devs), row=row_idx))
        row_idx += 1

    # 4. Compact passive tier: combine resistors & capacitors onto ONE 2-D row
    #    instead of stacking separate singleton resistor/cap rows.
    res_names = sorted(d.name for d in res_cards)
    cap_names = sorted(d.name for d in cap_cards)
    if res_names and cap_names:
        # When one or more high-value bias resistors dominate smaller feedback
        # resistors + cap, place the dominant resistor(s) in col=0 beside the
        # feedback+cap bank in col=1 on the same row.
        r_vals = {d.name: _r_ohm(d) for d in res_cards}
        pos_vals = [v for v in r_vals.values() if v > 0.0]
        r_med = float(median(pos_vals)) if len(pos_vals) >= 3 else 0.0
        seg_counts = {
            rs.device: int(rs.count)
            for rs in (getattr(craft_plan, "resistor_segments", ()) or ())
        } if craft_plan is not None else {}
        dominant_res = sorted(
            n for n in res_names
            if (r_med > 0.0 and r_vals.get(n, 0.0) >= 5.0 * r_med)
            or (seg_counts.get(n, 0) >= 16 and len(res_names) >= 2)
        )
        rest_passives = sorted([n for n in res_names if n not in set(dominant_res)] + cap_names)
        if dominant_res and len(rest_passives) >= 2:
            groups.append(PlaceGroupRelation("res_bias", tuple(dominant_res), row=row_idx, col=0))
            groups.append(PlaceGroupRelation("passive_fb", tuple(rest_passives), row=row_idx, col=1))
        else:
            groups.append(PlaceGroupRelation("passive_bank", tuple(sorted(res_names + cap_names)), row=row_idx))
        row_idx += 1
    elif res_names:
        groups.append(PlaceGroupRelation("res_bank", tuple(res_names), row=row_idx))
        row_idx += 1
    elif cap_names:
        groups.append(PlaceGroupRelation("cap_bank", tuple(cap_names), row=row_idx))
        row_idx += 1

    if other_devs:
        groups.append(PlaceGroupRelation("aux_bank", tuple(other_devs), row=row_idx))

    return tuple(groups)


def _layout_intent_block(
    decomp: RoleDecomposition,
    *,
    reference_gds: Optional[str] = None,
    source_netlist: Optional[str] = None,
    craft_plan: Optional["AutoCraftPlan"] = None,
    sizing_params: Optional[Mapping[str, str]] = None,
) -> str:
    """Emit the layout_intent block: the SPATIAL OVERLAY + topology/OP-aware craft.

    Emit the ABSTRACT FLOORPLAN -- either free-2D polarity-banded packing with
    ``@objective(area, minimize)`` for compact MOS / single-cap+resistor cells
    with shared-diffusion PCell arrays, or ``@place_group(..., row=R[, col=C])``
    per domain tier -- plus, when ``craft_plan`` is supplied, validated
    ``@array``, ``@guard_ring``, ``@trunk``, ``@wire_width``, ``@shield``, and
    ``@kelvin`` declarations derived without raw coordinates.
    """
    from collections import Counter
    from clay.frontend.floorplan_intent import floorplan_place_groups
    from clay.frontend.op_enrich import _canon_kind, _resolve_token_value

    declared_devs = {d.name: d for d in decomp.devices}
    declared_nets = {n for d in decomp.devices for n in d.terminals.values() if n}
    dev_kind = {d.name: _canon_kind(d) for d in decomp.devices}
    nmos_names = sorted(d.name for d in decomp.devices if dev_kind[d.name] == "nmos")
    pmos_names = sorted(d.name for d in decomp.devices if dev_kind[d.name] == "pmos")
    bjt_names = sorted(d.name for d in decomp.devices if dev_kind[d.name] in ("pnp", "npn", "bjt"))
    res_names = sorted(d.name for d in decomp.devices if dev_kind[d.name] == "res")
    cap_names = sorted(d.name for d in decomp.devices if dev_kind[d.name] == "cap")
    other_names = sorted(
        d.name for d in decomp.devices
        if dev_kind[d.name] not in ("nmos", "pmos", "pnp", "npn", "bjt", "res", "cap")
    )

    shared_od_arrays = (
        tuple(getattr(craft_plan, "shared_od_arrays", ()) or ())
        if craft_plan is not None
        else ()
    )
    has_clock = any(infer_port_kind(n) == "clock" for n in declared_nets)
    has_well_iso = bool(
        craft_plan is not None
        and (getattr(craft_plan, "well_islands", ()) or getattr(craft_plan, "well_isolated", ()))
    )
    n_passives = len(res_names) + len(cap_names)
    use_free_2d = bool(
        craft_plan is not None
        and not (reference_gds and source_netlist)
        and shared_od_arrays
        and not bjt_names
        and not other_names
        and not has_well_iso
        and not has_clock
        and len(nmos_names) >= 2
        and len(pmos_names) >= 2
        and (n_passives == 0 or (len(cap_names) >= 1 and len(res_names) <= 1))
    )

    place_groups = None
    if not use_free_2d:
        if reference_gds and source_netlist:
            place_groups = _reference_place_groups(decomp, reference_gds, source_netlist)
        if place_groups is None:
            place_groups = _clean_domain_place_groups(
                decomp, floorplan_place_groups(decomp), craft_plan=craft_plan
            )

    if not use_free_2d and not place_groups and craft_plan is None:
        return ""

    lines: List[str] = [f"layout_intent {decomp.cell} {{"]
    pg_by_name: Dict[str, set[str]] = {}
    pg_order: List[Tuple[str, set[str]]] = []

    if use_free_2d:
        lines.append("  @objective(area, minimize);")
        lines.append(f"  @place_group(nmos_band, devices=({', '.join(nmos_names)}));")
        lines.append(f"  @place_group(pmos_band, devices=({', '.join(pmos_names)}));")
        if n_passives > 0:
            passive_all = sorted(cap_names + res_names)
            lines.append(f"  @place_group(passive_bank, devices=({', '.join(passive_all)}));")
            p_bodies = Counter(
                declared_devs[m].terminals.get("b", "")
                for m in pmos_names
                if declared_devs[m].terminals.get("b", "")
            )
            n_bodies = Counter(
                declared_devs[m].terminals.get("b", "")
                for m in nmos_names
                if declared_devs[m].terminals.get("b", "")
            )
            top_rail = p_bodies.most_common(1)[0][0] if p_bodies else "vdd"
            bot_rail = n_bodies.most_common(1)[0][0] if n_bodies else "vss"
            if top_rail in declared_nets and bot_rail in declared_nets and top_rail != bot_rail:
                lines.append(f"  @band(top={top_rail}, bottom={bot_rail}, center=passive_bank);")
    else:
        for pg in (place_groups or ()):
            if not pg.devices:
                continue
            devs = ", ".join(pg.devices)
            col = f", col={pg.col}" if pg.col is not None else ""
            lines.append(
                f"  @place_group({pg.name}, devices=({devs}), row={pg.row}{col});"
            )
            pg_set = set(pg.devices)
            pg_by_name[pg.name] = pg_set
            pg_order.append((pg.name, pg_set))

    if craft_plan is not None:
        seen_arrays: set[str] = set()
        for arr in shared_od_arrays:
            gname = getattr(arr, "group_name", "")
            members = [m for m in getattr(arr, "devices", ()) if m in declared_devs]
            if not gname or len(members) < 2 or gname in seen_arrays:
                continue
            seen_arrays.add(gname)
            lines.append(f"  @array({gname}, style=shared_diffusion, order=ratio);")

        island_devs: set[str] = set()
        for grp, members, body_net in craft_plan.well_islands:
            valid_members = [m for m in members if m in declared_devs]
            if len(valid_members) < 2 or body_net not in declared_nets:
                continue
            # Only bind @well_island when an exact declared @place_group owns these devices
            exact_pg = next(
                (pg_name for pg_name, pg_set in pg_by_name.items() if pg_set == set(valid_members)),
                None,
            )
            if exact_pg is not None:
                lines.append(f"  @well_island({exact_pg}, body={body_net});")
                island_devs.update(valid_members)

        emitted_ring_sets: set[frozenset[str]] = set()
        nominated_rings = list(craft_plan.guard_rings)
        if use_free_2d:
            # Free-2D interlocking mode: emit tight per-cluster guard rings for
            # differential input pairs, multi-leg (>=3) mirrors/banks, and wide
            # (>=25um aggregate) mirrors/cascodes without wrapping entire polarity
            # bands into giant hollow rings.
            input_pair_devs = {
                dn
                for hit in decomp.roles
                if hit.role == "input_pair"
                for dn in hit.devices
            }
            def _card_w_eff_um(card: DeviceCard) -> float:
                w_m = _resolve_token_value(card.w, sizing_params or {}, default_scale_um=True)
                m_v = _resolve_token_value(card.m, sizing_params or {}, default_scale_um=False)
                m_int = max(1, int(round(float(m_v)))) if m_v else 1
                return (float(w_m) * 1e6 * m_int) if w_m else 0.0

            claimed_devs: set[str] = set()
            for gr in nominated_rings:
                raw_members = sorted(
                    m for m in gr.devices if m in declared_devs and m not in island_devs
                )
                if len(raw_members) < 2 or gr.tie not in declared_nets:
                    continue
                if set(raw_members) & claimed_devs:
                    continue
                bodies = {declared_devs[m].terminals.get("b", "") for m in raw_members}
                pols = {
                    "pmos" if dev_kind.get(m) == "pmos" else ("nmos" if dev_kind.get(m) == "nmos" else "")
                    for m in raw_members
                }
                if not (
                    bodies == {gr.tie}
                    and ((gr.well == "n" and pols == {"pmos"}) or (gr.well == "p" and pols == {"nmos"}))
                ):
                    continue
                is_input = bool(set(raw_members) <= input_pair_devs)
                sum_w_um = sum(_card_w_eff_um(declared_devs[m]) for m in raw_members)
                if not (is_input or len(raw_members) >= 3 or sum_w_um >= 25.0):
                    continue
                emitted_ring_sets.add(frozenset(raw_members))
                claimed_devs.update(raw_members)
                lines.append(
                    f"  @guard_ring(group=({', '.join(raw_members)}), well={gr.well}, "
                    f"tie={gr.tie}, hug=tight, role={gr.role});"
                )
        elif pg_order:
            # Align guard rings 1:1 with polarity-pure, uniform-body MOS place_groups
            # whenever a nominated ring covers at least half (or >= 2) of a group's
            # devices (or matches the group exactly), preventing strict-subset slicing
            # while avoiding dropping rings when small same-polarity helpers merge in.
            for _pg_name, pg_set in pg_order:
                if len(pg_set) < 2 or (pg_set & island_devs):
                    continue
                if frozenset(pg_set) in emitted_ring_sets:
                    continue
                pols = set()
                bodies = set()
                valid_mos = True
                for m in pg_set:
                    card = declared_devs.get(m)
                    if card is None:
                        valid_mos = False
                        break
                    k = (card.kind or "").strip().lower()
                    if k.startswith("p") and k != "pnp":
                        pols.add("pmos")
                    elif k.startswith("n") and k != "npn":
                        pols.add("nmos")
                    else:
                        valid_mos = False
                        break
                    bodies.add(card.terminals.get("b", ""))
                if not valid_mos or len(pols) != 1 or len(bodies) != 1:
                    continue
                pol = next(iter(pols))
                tie = next(iter(bodies))
                if not tie or tie not in declared_nets:
                    continue
                want_well = "n" if pol == "pmos" else "p"
                matching_nom = None
                second_stage_devs = {
                    dn
                    for hit in decomp.roles
                    if hit.role == "second_stage"
                    for dn in hit.devices
                }
                for gr in nominated_rings:
                    if gr.well != want_well or gr.tie != tie:
                        continue
                    gr_set = set(gr.devices)
                    overlap = [m for m in gr.devices if m in pg_set]
                    extra = pg_set - gr_set
                    # Never forcibly pull an unnominated high-drive second-stage
                    # output FET into a small-signal mirror's guard ring.
                    if extra & second_stage_devs:
                        continue
                    if set(overlap) == pg_set or len(overlap) >= 2:
                        matching_nom = gr
                        break
                if matching_nom is not None:
                    members_sorted = sorted(pg_set)
                    emitted_ring_sets.add(frozenset(pg_set))
                    lines.append(
                        f"  @guard_ring(group=({', '.join(members_sorted)}), well={want_well}, "
                        f"tie={tie}, hug=tight, role={matching_nom.role});"
                    )
        else:
            for gr in nominated_rings:
                raw_members = sorted(m for m in gr.devices if m in declared_devs and m not in island_devs)
                if len(raw_members) < 2 or gr.tie not in declared_nets:
                    continue
                bodies = {declared_devs[m].terminals.get("b", "") for m in raw_members}
                pols = {
                    "pmos" if (declared_devs[m].kind or "").lower().startswith("p") else "nmos"
                    for m in raw_members
                }
                if bodies == {gr.tie} and ((gr.well == "n" and pols == {"pmos"}) or (gr.well == "p" and pols == {"nmos"})):
                    lines.append(
                        f"  @guard_ring(group=({', '.join(raw_members)}), well={gr.well}, "
                        f"tie={gr.tie}, hug=tight, role={gr.role});"
                    )

        seen_trunks: set[str] = set()
        for tr in craft_plan.trunks:
            net = getattr(tr, "net", "")
            if not net or net not in declared_nets or net in seen_trunks:
                continue
            seen_trunks.add(net)
            layer = getattr(tr, "layer", "M2") or "M2"
            orient = getattr(tr, "orient", "H") or "H"
            extras: List[str] = [f"layer={layer}", f"orient={orient}"]
            w_um = float(getattr(tr, "width_um", 0.0) or 0.0)
            if w_um > 0:
                extras.append(f"width={w_um:g}")
            edge = str(getattr(tr, "edge", "") or "").strip().lower()
            if edge in ("top", "bottom", "left", "right"):
                extras.append(f"edge={edge}")
            lines.append(f"  @trunk({net}, {', '.join(extras)});")

        seen_widths: set[str] = set()
        for ww in craft_plan.wire_widths:
            if ww.net in declared_nets and ww.net not in seen_widths and ww.width_um > 0.36:
                seen_widths.add(ww.net)
                lines.append(f"  @wire_width({ww.net}, {ww.width_um:g}u);")

        seen_shields: set[Tuple[str, str]] = set()
        for sh in craft_plan.shields:
            if (
                {sh.aggressor, sh.victim, sh.shield_net} <= declared_nets
                and len({sh.aggressor, sh.victim, sh.shield_net}) == 3
                and (sh.aggressor, sh.victim) not in seen_shields
            ):
                seen_shields.add((sh.aggressor, sh.victim))
                lines.append(f"  @shield({sh.aggressor} -> {sh.victim}, {sh.shield_net});")

        for kv in craft_plan.kelvins:
            card = declared_devs.get(kv.target)
            if (
                card is not None
                and kv.force_net != kv.sense_net
                and {kv.force_net, kv.sense_net} <= set(card.terminals.values())
            ):
                lines.append(
                    f"  @kelvin({kv.force_net}, sense={kv.sense_net}, target={kv.target});"
                )

    if len(lines) == 1:
        return ""
    lines.append("  // TODO: parasitic-cap budgets / area cap need a target spec")
    lines.append("}")
    return "\n".join(lines)


def _policy_block(policy_name: str, *, gap_um: float = 2.0) -> str:
    return (
        f"policy {policy_name} {{\n"
        f"  @place_policy(gap = {gap_um:g});\n"
        f"  @route_policy(engine = pathfinder);\n"
        f"}}"
    )


def _flow_block(cell: str, pdk: str, *, policy_name: Optional[str] = None) -> str:
    pol_line = f"  policy = {policy_name};\n" if policy_name else ""
    return (
        f"flow {cell} {{\n"
        f'  pdk "{pdk}";\n'
        f"  steps (compile, layout, drc, lvs, pex, post, intent, sign);\n"
        f"  layout = generic;\n"
        f"{pol_line}"
        f"}}"
    )


def _fmt_kpi_num(v: float) -> str:
    """A grammar-safe SIGNED_NUMBER (plain int / decimal, never `1e+07`)."""
    return str(int(v)) if float(v) == int(v) else repr(float(v))


def _quality_block(cell: str, kpi_specs=None) -> str:
    """Emit ``quality {}``. Concrete ``@kpi`` lines when ``kpi_specs`` are derived
    (from the tb / Spec); only the structural ``@require(drc_clean)`` + a TODO when
    none (no fabricated thresholds)."""
    lines = [f"quality {cell} {{"]
    for s in (kpi_specs or []):
        unit = f', unit="{s.unit}"' if getattr(s, "unit", "") else ""
        lines.append(f'  @kpi({s.metric}, "{s.op}", {_fmt_kpi_num(s.value)}{unit});')
    lines.append("  @require(drc_clean);")
    if not kpi_specs:
        lines.append("  // TODO: add @kpi(...) acceptance thresholds for this design")
    lines.append("}")
    return "\n".join(lines)


def _disambiguate_port_kind(
    decomp: RoleDecomposition,
    net: str,
    kind: str,
) -> str:
    """Disambiguate ``input``-named ports using pure terminal topology:
    1) Promote supply rails named ``vin`` (touching MOS body+source with zero
       gate pins) to ``power_p`` (PMOS bulk) or ``power_n`` (NMOS bulk).
    2) Promote ``vref``-like outputs (driven solely by drain/emitter/collector
       pins with zero gate/source/body pins) to ``output``."""
    if kind != "input":
        return kind
    pins: set[Tuple[str, str]] = set()
    for dev in decomp.devices:
        dk = (dev.kind or "").strip().lower()
        for pk, pn in dev.terminals.items():
            if pn == net:
                pins.add((dk, pk.strip().lower()))
    has_gate = any(pk == "g" for _, pk in pins)
    if not has_gate:
        has_pmos_bs = any(
            dk.startswith(("pmos", "pfet")) and pk == "b" for dk, pk in pins
        ) and any(
            dk.startswith(("pmos", "pfet")) and pk == "s" for dk, pk in pins
        )
        has_nmos_bs = any(
            dk.startswith(("nmos", "nfet")) and pk == "b" for dk, pk in pins
        ) and any(
            dk.startswith(("nmos", "nfet")) and pk == "s" for dk, pk in pins
        )
        if has_pmos_bs and not has_nmos_bs:
            return "power_p"
        if has_nmos_bs and not has_pmos_bs:
            return "power_n"
    has_drive = any(
        (dk.startswith(("nmos", "pmos", "nfet", "pfet")) and pk == "d")
        or (dk in ("pnp", "npn", "bjt") and pk in ("c", "e"))
        for dk, pk in pins
    )
    has_ctrl_or_source = any(pk in ("g", "b", "s") for _, pk in pins)
    if has_drive and not has_ctrl_or_source:
        return "output"
    return kind


def _port_lines(
    decomp: RoleDecomposition,
    ports: Optional[Sequence[Tuple[str, str]]],
) -> List[Tuple[str, str]]:
    """Resolve the (net, kind) port list.

    If ``ports`` is supplied it is used verbatim. Otherwise the port nets are
    inferred from the device terminals: nets that touch a power/ground/bias/IO
    name (via :func:`infer_port_kind`) become ports. To keep the draft
    deterministic and reviewable we collect candidate port nets as the union of
    every device terminal net whose name maps to a non-``inout`` kind, in first-
    seen order.
    """
    if ports is not None:
        return [(net, kind) for net, kind in ports]

    if decomp.source_ports:
        return [
            (net, _disambiguate_port_kind(decomp, net, infer_port_kind(net)))
            for net in decomp.source_ports
        ]

    ordered: List[str] = []
    seen: set[str] = set()
    for dev in decomp.devices:
        for net in dev.terminals.values():
            if net in seen:
                continue
            seen.add(net)
            ordered.append(net)

    inferred: List[Tuple[str, str]] = []
    for net in ordered:
        kind = _disambiguate_port_kind(decomp, net, infer_port_kind(net))
        if kind != "inout":
            inferred.append((net, kind))
    return inferred


def emit_design_clay(
    decomp: RoleDecomposition,
    *,
    package: str,
    pdk: str = "gf180mcu",
    ports: Optional[Sequence[Tuple[str, str]]] = None,
    kpi_specs: Optional[Sequence] = None,
    reference_gds: Optional[str] = None,
    source_netlist: Optional[str] = None,
    craft_plan: Optional["AutoCraftPlan"] = None,
    sizing_params: Optional[Mapping[str, str]] = None,
) -> str:
    """Render a ``design.clay`` DRAFT from a :class:`RoleDecomposition`.

    Args:
        decomp: the typed functional-role decomposition to overlay.
        package: the dotted package name for the ``package ...;`` header.
        pdk: the target PDK token (default ``gf180mcu``).
        ports: explicit ``(net, kind)`` port list; when omitted, port kinds are
            inferred from device terminal net names via :func:`infer_port_kind`.
        craft_plan: optional :class:`AutoCraftPlan` enriching ``layout_intent``
            and ``annotate``.
        sizing_params: optional resolved ``.param`` map to substitute symbolic
            device dimensions cleanly.

    Returns:
        Valid Clay source (parses via ``clay.parser.parse_clay_file``).
    """
    resolved_ports = _port_lines(decomp, ports)
    if resolved_ports:
        port_src = ", ".join(f"{net}: {kind}" for net, kind in resolved_ports)
    else:
        # A cell with no ports cannot be expressed (port_list is non-optional in
        # the grammar). Fall back to a single TODO power port so the draft still
        # parses; the reviewer fills in the real interface.
        port_src = "vdd: power_p"

    parts: List[str] = []
    parts.append(f"package {package};")
    parts.append(f'pdk "{pdk}";')
    parts.append("")
    parts.append(f"cell {decomp.cell}({port_src}) {{")
    res_models = (
        {rs.device: rs.model for rs in getattr(craft_plan, "resistor_segments", ()) or () if rs.model}
        if craft_plan is not None
        else None
    )
    parts.append(
        _topology_block(
            decomp.devices,
            decomp.roles,
            sizing_params=sizing_params,
            res_models_by_dev=res_models,
            craft_plan=craft_plan,
        )
    )
    parts.append("}")

    layout_intent = _layout_intent_block(
        decomp,
        reference_gds=reference_gds,
        source_netlist=source_netlist,
        craft_plan=craft_plan,
        sizing_params=sizing_params,
    )
    if layout_intent:
        parts.append("")
        parts.append(layout_intent)

    compact_policy_name: Optional[str] = None
    if craft_plan is not None and (
        ", col=" in layout_intent or "@objective(area, minimize)" in layout_intent
    ):
        compact_policy_name = f"{decomp.cell.lower()}_compact_2d"
        has_bjt_cell = any(
            (d.kind or "").strip().lower() in ("pnp", "npn", "bjt")
            for d in decomp.devices
        )
        has_arrays = bool(getattr(craft_plan, "shared_od_arrays", ()))
        gap_um = (
            1.8
            if (not has_bjt_cell and (has_arrays or "@objective(area, minimize)" in layout_intent))
            else 2.0
        )
        parts.append("")
        parts.append(_policy_block(compact_policy_name, gap_um=gap_um))

    parts.append("")
    parts.append(_flow_block(decomp.cell, pdk, policy_name=compact_policy_name))
    parts.append("")
    parts.append(_quality_block(decomp.cell, kpi_specs))

    annotate = _annotate_block(decomp, craft_plan=craft_plan)
    if annotate:
        parts.append("")
        parts.append(annotate)

    return "\n".join(parts) + "\n"
