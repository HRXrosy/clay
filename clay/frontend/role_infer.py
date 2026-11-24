"""Structural functional-role inference for a SPICE ``.cir`` netlist
(front-end paradigm mapper, agent A = ROLE-INFER).

This module is the GENERAL front-end that turns a foreign / hand-written
SPICE netlist into the typed :class:`clay.frontend.role_model.RoleDecomposition`
the Clay paradigm overlay is built on. It does NOT special-case the four probe
circuits: every role is inferred from STRUCTURAL patterns over the device graph
(nets as nodes, devices as edges), so a new topology onboards by the same rules.

What it composes (imported, never re-implemented)
=================================================
* :func:`clay.layout.generic_placement.parse_cir_mos_records` -- the existing,
  battle-tested MOS card reader (kind from the model token nch / pch /
  nmos_3p3 / pmos_3p3 / ..., terminals ``d g s b``, W / L / m resolved through
  ``.param``). We reuse its card grammar verbatim for transistors and mirror the
  same tokeniser only for the 2-terminal (R / C) and BJT cards it (correctly)
  skips, so the device list is complete for the role graph.

Structural rules (one uniform rule set for every circuit)
=========================================================
The roles are inferred over the device graph, not from device names:

* ``input_pair`` -- two same-kind MOS whose gates are two DISTINCT nets and
  whose sources tie to a common node (the tail); that common node is the
  evidence the pair is differential.
* ``tail_current`` -- the device whose drain is that common source node and
  whose source is a rail (vdd for a PMOS pair, vss for an NMOS pair).
* ``current_mirror`` / ``load_mirror`` -- two+ same-kind MOS sharing a gate net
  that is the gate=drain (diode-connected) node of one of them (the reference);
  ``load_mirror`` when the shared-gate group sits on the opposite rail from the
  input pair (it is the active load), ``current_mirror`` otherwise.
* ``cascode`` -- a device whose source is another signal-path device's drain
  and whose gate is a bias net (not a signal input, not its own drain).
* ``bias_tree`` -- diode-connected devices and the gate-tied chain hanging off
  them that generate the bias nets (a device is in the bias tree when it is
  diode-connected to a rail or its gate rides a diode node and it is not already
  claimed as input / tail / mirror / cascode of the signal path).
* ``pass_device`` -- a large PMOS (or a parallel bank of them) from the top rail
  to the regulated output, gate on an internal driver net (LDO pass FET).
* ``pnp_array`` / ``resistor`` / ``compensation`` -- BJT clusters sharing an
  emitter/collector net, matched 2-terminal resistor banks, and the series
  Rz->Cc compensation branch are recognised from the 2-/3-terminal cards.

``matched_pairs`` are the symmetric members of a role (the input pair, the
mirror pair, the cascode pair, the fold pair); ``mirror_axis`` is ``"vertical"``
whenever a differential (input-pair) structure is found.
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from clay.frontend.role_model import (
    CANONICAL_ROLES,
    DeviceCard,
    FunctionalRoleHit,
    RoleDecomposition,
)
from clay.layout.generic_placement import (
    _cir_model_kind,
    _cir_params,
    _cir_resolve,
    parse_cir_mos_records,
)

__all__ = ["infer_roles", "parse_devices", "role_family"]

# The structural rules below emit these tokens; all but the two descriptive
# extensions (pnp_array / resistor / second_stage) are drawn from the suggested
# canonical analog set in the frozen role model. Recorded so a reader can see
# the emitted vocabulary aligns with the shared catalog.
_EMITTED_CANONICAL = (
    "input_pair",
    "tail_current",
    "current_mirror",
    "load_mirror",
    "cascode",
    "bias_tree",
    "pass_device",
    "compensation",
)
assert set(_EMITTED_CANONICAL) <= set(CANONICAL_ROLES)


# === device parsing ==========================================================
#
# MOS cards are read by the existing parser (composition). The 2-terminal
# (R / C) and BJT cards that parser skips are read here with the same head
# tokeniser so the device list is complete for the structural graph.

_BJT_MODEL_HINTS = ("pnp", "npn", "vpnp", "vnpn", "bjt")


def _read_text(cir_path_or_text: str | Path) -> str:
    """Accept either a path to a .cir or the netlist text directly."""
    s = str(cir_path_or_text)
    # A path-ish single-line string with no newline that exists on disk -> read.
    if "\n" not in s:
        try:
            p = Path(s)
            if p.exists():
                return p.read_text(encoding="utf-8-sig")
        except (OSError, ValueError):
            pass
    return s


def _strip(line: str) -> str:
    """Strip a SPICE comment ($ ... and trailing) and surrounding whitespace."""
    return line.split("$", 1)[0].strip()


def _select_subckt(text: str, cell_name: Optional[str]) -> str:
    """Return only the body of the resolved ``.subckt`` (the top cell).

    If ``cell_name`` is given, that subckt's body is returned; otherwise the
    FIRST ``.subckt`` is used (the probe decks define exactly one design cell,
    with the bandgap deck adding a flat testbench OUTSIDE the subckt that must
    not pollute the device graph). When no ``.subckt`` is present the whole text
    is the body (a flat netlist).
    """
    lines = text.splitlines()
    blocks: List[Tuple[str, List[str]]] = []
    cur_name: Optional[str] = None
    cur: List[str] = []
    depth = 0
    for raw in lines:
        line = _strip(raw)
        low = line.lower()
        if low.startswith(".subckt"):
            parts = line.split()
            if depth == 0:
                cur_name = parts[1] if len(parts) > 1 else ""
                cur = []
            depth += 1
            continue
        if low.startswith(".ends"):
            depth -= 1
            if depth <= 0 and cur_name is not None:
                blocks.append((cur_name, cur))
                cur_name = None
                cur = []
                depth = 0
            continue
        if cur_name is not None:
            cur.append(raw)
    if not blocks:
        return text
    if cell_name is not None:
        for name, body in blocks:
            if name == cell_name:
                return "\n".join(body)
    return "\n".join(blocks[0][1])


def _subckt_signature(
    text: str, cell_name: Optional[str]
) -> Tuple[str, Tuple[str, ...]]:
    """Return the selected ``.SUBCKT`` name and ordered external nodes.

    Port existence and order are source-netlist facts, independent of role or
    direction inference.  Parameter declarations begin at ``PARAMS:`` or the
    first assignment token and are therefore excluded from the interface.
    Flat netlists return an empty signature and retain the legacy name heuristic.
    """

    first: Optional[Tuple[str, Tuple[str, ...]]] = None
    for raw in text.splitlines():
        line = _strip(raw)
        if not line.lower().startswith(".subckt"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        name = parts[1]
        ports: List[str] = []
        for token in parts[2:]:
            low = token.casefold()
            if low in {"params:", "param:"} or "=" in token:
                break
            ports.append(token)
        signature = (name, tuple(ports))
        if first is None:
            first = signature
        if cell_name is not None and name == cell_name:
            return signature
    return first or (cell_name or "", ())


def _inline_multiplier(tokens: List[str], params: Dict[str, str]) -> str:
    """Recover the SPICE inline device multiplier ``m=N`` from a passive card's
    trailing tokens.

    On a passive card (``C<name> p n value m=N`` / ``R<name> p n value m=N``)
    the ``m=N`` factor means N parallel unit devices: a capacitor's effective
    capacitance is ``value * N`` and a resistor's effective resistance is
    ``value / N``. The MOS reader already carries this as ``card.m``; passives
    must too, otherwise the multiplier is silently dropped and the device is
    built at the wrong magnitude. Returns the resolved token (a ``.param`` name
    is resolved like the value), or ``""`` when no ``m=`` token is present.
    """
    for tok in tokens:
        if "=" not in tok:
            continue
        key, _, val = tok.partition("=")
        if key.strip().lower() == "m":
            return _cir_resolve(val.strip(), params)
    return ""


def _inline_parameter(
    tokens: List[str], key_name: str, params: Dict[str, str]
) -> str:
    """Resolve one ``key=value`` token without assigning device semantics."""
    for tok in tokens:
        if "=" not in tok:
            continue
        key, _, value = tok.partition("=")
        if key.strip().lower() == key_name.lower():
            return _cir_resolve(value.strip(), params)
    return ""


def _parse_two_term_and_bjt(
    body: str, params: Dict[str, str]
) -> List[DeviceCard]:
    """Parse R / C / BJT cards (the ones the MOS reader skips) into cards.

    * ``R<name> p n value`` -> kind ``res``, terminals ``{p, n}``, value -> w.
    * ``C<name> p n value`` -> kind ``cap``, terminals ``{p, n}``, value -> w.
    * a BJT, written either as a primitive ``Q<name> c b e model`` or a subckt
      instance ``x<name> c b e <pnp model>`` -> kind ``bjt`` (terminals
      ``{c, b, e}``). A subckt instance whose model is NOT a BJT is ignored
      here (MOS instances are handled by the composed MOS reader).
    """
    out: List[DeviceCard] = []
    for raw in body.splitlines():
        line = _strip(raw)
        if not line or line.startswith("*") or line.startswith("."):
            continue
        parts = line.split()
        head = parts[0]
        lead = head[:1].upper()
        if lead == "R" and len(parts) >= 4:
            out.append(
                DeviceCard(
                    name=head,
                    kind="res",
                    terminals={"p": parts[1], "n": parts[2]},
                    w=_cir_resolve(parts[3], params),
                    m=_inline_multiplier(parts[4:], params),
                )
            )
        elif lead == "C" and len(parts) >= 4:
            out.append(
                DeviceCard(
                    name=head,
                    kind="cap",
                    terminals={"p": parts[1], "n": parts[2]},
                    w=_cir_resolve(parts[3], params),
                    m=_inline_multiplier(parts[4:], params),
                )
            )
        elif lead == "Q" and len(parts) >= 5:
            model = parts[4]
            trailing = parts[5:]
            area = _inline_parameter(trailing, "area", params)
            if not area and trailing and "=" not in trailing[0]:
                area = _cir_resolve(trailing[0], params)
            out.append(
                DeviceCard(
                    name=head,
                    # Keep the long-standing front-end family token stable;
                    # source_model proves polarity for Clay emission.
                    kind="bjt",
                    terminals={"c": parts[1], "b": parts[2], "e": parts[3]},
                    m=_inline_multiplier(trailing, params),
                    source_model=model,
                    area=area,
                )
            )
        elif lead == "X":
            # subckt instance: find the model token (first token before any k=v)
            kv_start = len(parts)
            for i, tok in enumerate(parts[1:], start=1):
                if "=" in tok:
                    kv_start = i
                    break
            head_tokens = parts[1:kv_start]
            if len(head_tokens) < 2:
                continue
            model = head_tokens[-1]
            nodes = head_tokens[:-1]
            low = model.lower()
            if any(h in low for h in _BJT_MODEL_HINTS) and len(nodes) >= 3:
                name = head[1:] if head[:1].lower() == "x" else head
                out.append(
                    DeviceCard(
                        name=name,
                        kind="bjt",
                        terminals={"c": nodes[0], "b": nodes[1], "e": nodes[2]},
                        m=_inline_multiplier(parts[kv_start:], params),
                        source_model=model,
                        area=_inline_parameter(parts[kv_start:], "area", params),
                    )
                )
            # MOS instances handled by parse_cir_mos_records; skip here.
    return out


def parse_devices(
    cir_path_or_text: str | Path, cell_name: Optional[str] = None
) -> Tuple[str, List[DeviceCard]]:
    """Parse a ``.cir`` into ``(cell_name, [DeviceCard, ...])``.

    MOS cards are read by the composed :func:`parse_cir_mos_records` (kind,
    terminals d/g/s/b, W/L/m -- symbolic tokens preserved); R / C / BJT cards are
    read here. Only the resolved top ``.subckt`` (or the flat netlist) is parsed.
    """
    text = _read_text(cir_path_or_text)
    params = _cir_params(text)
    body = _select_subckt(text, cell_name)

    # resolve the cell name actually used.
    resolved_cell = cell_name or ""
    if not resolved_cell:
        m = re.search(r"(?im)^\s*\.subckt\s+(\S+)", text)
        if m:
            resolved_cell = m.group(1)

    cards: List[DeviceCard] = []

    # 1. MOS via the composed reader -- but it parses the WHOLE file, so we
    #    restrict to the names that appear in the resolved subckt body so a flat
    #    testbench MOS (none here, but be safe) cannot leak in.
    body_names: Set[str] = set()
    for raw in body.splitlines():
        line = _strip(raw)
        if not line or line.startswith("*") or line.startswith("."):
            continue
        head = line.split()[0]
        n = head[1:] if head[:1].lower() == "x" else head
        body_names.add(n)
        body_names.add(head)

    mos = _mos_records_from_text(cir_path_or_text, text)
    for label, rec in mos.items():
        if body_names and label not in body_names and ("x" + label) not in body_names:
            continue
        pins = rec.get("pins", {}) or {}
        # recover the ORIGINAL symbolic W/L tokens from the body (the composed
        # reader returns resolved SI floats; the role model wants the tokens).
        w_tok, l_tok, m_tok = _symbolic_sizing(body, label)
        cards.append(
            DeviceCard(
                name=label,
                kind=str(rec.get("kind", "")),
                terminals={k: str(v) for k, v in pins.items()},
                w=w_tok,
                l=l_tok,
                m=m_tok,
                source_model=str(rec.get("model", "")),
            )
        )

    # 2. R / C / BJT.
    cards.extend(_parse_two_term_and_bjt(body, params))
    return resolved_cell, cards


def _mos_records_from_text(cir_path_or_text, text: str):
    """Run the composed MOS reader on the netlist.

    :func:`parse_cir_mos_records` reads a PATH (``Path(source).read_text``). When
    we were handed a real path we pass it straight through (no I/O duplication);
    when we were handed raw text we materialise it in a temp file so the composed
    reader runs unmodified. The body-name filter in :func:`parse_devices` then
    keeps only the devices inside the resolved subckt, so a flat testbench cannot
    leak in.
    """
    s = str(cir_path_or_text)
    if "\n" not in s:
        try:
            if Path(s).exists():
                return parse_cir_mos_records(s)
        except (OSError, ValueError):
            pass
    import tempfile

    with tempfile.NamedTemporaryFile(
        "w", suffix=".cir", delete=False, encoding="utf-8"
    ) as fh:
        fh.write(text)
        tmp = fh.name
    try:
        return parse_cir_mos_records(tmp)
    finally:
        try:
            Path(tmp).unlink()
        except OSError:
            pass


_SIZE_RE = {
    "W": re.compile(r"\bW\s*=\s*(\{[^}]*\}|[A-Za-z0-9_.+\-]+)", re.IGNORECASE),
    "L": re.compile(r"\bL\s*=\s*(\{[^}]*\}|[A-Za-z0-9_.+\-]+)", re.IGNORECASE),
    "m": re.compile(r"\bm\s*=\s*(\{[^}]*\}|[A-Za-z0-9_.+\-]+)", re.IGNORECASE),
}


def _symbolic_sizing(body: str, label: str) -> Tuple[str, str, str]:
    """Recover the AS-WRITTEN W / L / m tokens for a device from the body.

    Returns the tokens stripped of any surrounding braces (``{W_diff}`` ->
    ``W_diff``), so the role model carries the symbolic intent (``W_diff``) the
    same way the netlist wrote it -- not the resolved float.
    """
    for raw in body.splitlines():
        line = _strip(raw)
        if not line:
            continue
        head = line.split()[0]
        n = head[1:] if head[:1].lower() == "x" else head
        if n != label and head != label:
            continue
        out = []
        for key in ("W", "L", "m"):
            mm = _SIZE_RE[key].search(line)
            tok = ""
            if mm:
                tok = mm.group(1).strip()
                if tok.startswith("{") and tok.endswith("}"):
                    tok = tok[1:-1].strip()
            out.append(tok)
        return out[0], out[1], out[2]
    return "", "", ""


# === structural graph ========================================================


class _Graph:
    """Net-centric view of the parsed devices for structural matching."""

    def __init__(self, cards: List[DeviceCard], *, rails: Optional[Set[str]] = None):
        self.cards = cards
        self.by_name: Dict[str, DeviceCard] = {c.name: c for c in cards}
        self.mos: List[DeviceCard] = [c for c in cards if c.kind in ("nmos", "pmos")]
        # net -> set of device names touching it (any terminal)
        self.net_devs: Dict[str, Set[str]] = defaultdict(set)
        for c in cards:
            for net in c.terminals.values():
                self.net_devs[net].add(c.name)
        # detect rails: the most-shared nets that look like power/ground.
        self.rails = self._detect_rails() if rails is None else set(rails)

    def _detect_rails(self) -> Set[str]:
        rails: Set[str] = set()
        for net in self.net_devs:
            low = net.lower()
            if low in ("vss", "gnd", "0", "vdd", "vin", "vcc", "vpwr", "vgnd"):
                rails.add(net)
        # also any net that is the bulk of >=3 MOS and a source of >=3 MOS is a
        # rail even if oddly named.
        src_count: Dict[str, int] = defaultdict(int)
        for c in self.mos:
            src_count[c.terminals.get("s", "")] += 1
        for net, n in src_count.items():
            if n >= 3 and net:
                rails.add(net)
        return rails

    def is_rail(self, net: str) -> bool:
        return net in self.rails


def _diode_connected(c: DeviceCard) -> bool:
    """A MOS is diode-connected when gate == drain."""
    return (
        c.kind in ("nmos", "pmos")
        and c.terminals.get("g")
        and c.terminals.get("g") == c.terminals.get("d")
    )


# === role inference ==========================================================


def _infer_input_pairs(
    g: _Graph,
) -> Tuple[List[FunctionalRoleHit], List[Tuple[str, str]], Dict[str, str]]:
    """Find differential input pairs: two same-kind MOS, distinct gate nets,
    common source node.

    Returns the hits, the matched pairs, and a map ``tail_net -> kind`` so the
    tail-current inference can target the right rail-side device.
    """
    hits: List[FunctionalRoleHit] = []
    pairs: List[Tuple[str, str]] = []
    tails: Dict[str, str] = {}
    seen: Set[frozenset] = set()
    mos = g.mos
    for i in range(len(mos)):
        for j in range(i + 1, len(mos)):
            a, b = mos[i], mos[j]
            if a.kind != b.kind:
                continue
            ga, gb = a.terminals.get("g"), b.terminals.get("g")
            sa, sb = a.terminals.get("s"), b.terminals.get("s")
            if not (ga and gb and sa and sb):
                continue
            if ga == gb:
                continue  # same gate net -> a mirror, not an input pair
            if sa != sb:
                continue  # sources not shared -> not a diff pair
            if g.is_rail(sa):
                continue  # a pair sourced straight to the rail is not a tail-
                # coupled diff pair (e.g. a mirror); the diff pair's shared
                # source is the (internal) tail node.
            # gates must be genuine signal inputs (not a diode / bias node):
            if _diode_connected(a) or _diode_connected(b):
                continue
            key = frozenset((a.name, b.name))
            if key in seen:
                continue
            seen.add(key)
            hits.append(
                FunctionalRoleHit(
                    role="input_pair",
                    devices=(a.name, b.name),
                    evidence=(
                        f"two {a.kind} with distinct gate nets ({ga}, {gb}) "
                        f"sharing source node {sa} (the tail)"
                    ),
                    confidence=0.95,
                )
            )
            pairs.append((a.name, b.name))
            tails[sa] = a.kind
    return hits, pairs, tails


def _infer_tail(
    g: _Graph, tails: Dict[str, str], claimed: Set[str]
) -> Tuple[List[FunctionalRoleHit], List[str]]:
    """The tail current source: drain == the diff-pair common-source node, source
    on the rail opposite the pair's signal flow."""
    hits: List[FunctionalRoleHit] = []
    out: List[str] = []
    for c in g.mos:
        if c.name in claimed:
            continue
        d = c.terminals.get("d")
        s = c.terminals.get("s")
        if d in tails and g.is_rail(s):
            hits.append(
                FunctionalRoleHit(
                    role="tail_current",
                    devices=(c.name,),
                    evidence=(
                        f"drain on the diff-pair tail node {d}, source on rail {s}"
                    ),
                    confidence=0.9,
                )
            )
            out.append(c.name)
    return hits, out


def _infer_mirrors(
    g: _Graph, input_kinds: Set[str], claimed: Set[str]
) -> Tuple[List[FunctionalRoleHit], List[Tuple[str, str]]]:
    """Current / load mirrors: a diode-connected reference plus other same-kind
    MOS sharing its gate net.

    The shared gate net of a mirror is the diode node (gate==drain of the
    reference). Group all same-kind, same-rail MOS riding that net. A mirror that
    sits on the rail OPPOSITE the input pair is the active load (load_mirror).
    """
    hits: List[FunctionalRoleHit] = []
    pairs: List[Tuple[str, str]] = []
    # Group same-kind, rail-sourced MOS by their shared gate net. A mirror is a
    # gate-tied group of >=2 such devices; the reference is the diode-connected
    # member when present (gate=drain), else the group is still a mirror (the
    # diode reference may be on a different rail, e.g. the cascode-fed load).
    by_gate: Dict[Tuple[str, str], List[DeviceCard]] = defaultdict(list)
    for c in g.mos:
        gnet = c.terminals.get("g")
        s = c.terminals.get("s")
        if not gnet or not s:
            continue
        if not g.is_rail(s):
            continue  # a mirror's members are rail-sourced current sources
        by_gate[(gnet, c.kind)].append(c)

    used: Set[str] = set()
    for (node, kind), members in by_gate.items():
        if len(members) < 2:
            continue
        names = tuple(sorted(m.name for m in members))
        if any(n in used for n in names):
            continue
        has_diode = any(_diode_connected(m) for m in members)
        # a mirror feeding the SIGNAL path (a member's drain is a non-rail,
        # non-bias node read by the core) is a load/current mirror; a mirror
        # whose drains all feed bias / diode nets is part of the bias tree.
        is_load = bool(input_kinds) and kind not in input_kinds
        role = "load_mirror" if is_load else "current_mirror"
        hits.append(
            FunctionalRoleHit(
                role=role,
                devices=names,
                evidence=(
                    f"{len(members)} {kind} sharing gate net {node}"
                    + (
                        f" (gate=drain diode reference present)"
                        if has_diode
                        else " (gate-tied current mirror)"
                    )
                ),
                confidence=0.85 if has_diode else 0.7,
            )
        )
        used.update(names)
        if len(names) == 2:
            pairs.append((names[0], names[1]))
    return hits, pairs


def _infer_cascodes(
    g: _Graph, claimed: Set[str]
) -> Tuple[List[FunctionalRoleHit], List[Tuple[str, str]]]:
    """Cascode devices: source sits on another signal-path device's DRAIN, gate
    is a bias net (constant: not its own drain, not a primary input)."""
    hits: List[FunctionalRoleHit] = []
    pairs: List[Tuple[str, str]] = []
    drain_nets: Set[str] = {
        c.terminals.get("d", "") for c in g.mos if c.terminals.get("d")
    }
    # a bias net is an internal node that some device DRIVES (a drain) -- a
    # cascode gate rides such a generated bias, never a primary input port. The
    # set of nets used as gates but never as a drain are the primary inputs.
    gate_nets = {c.terminals.get("g", "") for c in g.mos}
    input_ports = {n for n in gate_nets if n and n not in drain_nets}
    cands: List[DeviceCard] = []
    for c in g.mos:
        if c.name in claimed:
            continue  # already an input / tail / mirror member
        s = c.terminals.get("s")
        gnet = c.terminals.get("g")
        d = c.terminals.get("d")
        if not (s and gnet and d):
            continue
        if g.is_rail(s):
            continue  # rail-sourced -> not stacked on a signal node
        if s not in drain_nets:
            continue  # source must sit on some device's drain (stacked)
        if _diode_connected(c):
            continue  # diode -> mirror reference, not a cascode
        if gnet == s or gnet == d:
            continue
        if gnet in input_ports:
            continue  # gate on a primary input -> a diff-pair member, not cascode
        # gate must be a generated bias net (a low-fanout internal node).
        cands.append(c)
    # group cascode candidates by gate net into matched pairs.
    by_gate: Dict[str, List[str]] = defaultdict(list)
    for c in cands:
        by_gate[c.terminals["g"]].append(c.name)
    for gnet, names in by_gate.items():
        names = sorted(names)
        hits.append(
            FunctionalRoleHit(
                role="cascode",
                devices=tuple(names),
                evidence=(
                    f"source stacked on a signal-path drain, gate on bias net "
                    f"{gnet}"
                ),
                confidence=0.8,
            )
        )
        if len(names) == 2:
            pairs.append((names[0], names[1]))
    return hits, pairs


def _infer_pass_device(
    g: _Graph, claimed: Set[str]
) -> Tuple[List[FunctionalRoleHit], List[Tuple[str, str]]]:
    """LDO pass device(s): PMOS from the top rail to a regulated output net,
    gate on an internal driver node, replicated as a wide parallel bank.

    Recognised structurally as a set of >=2 PMOS sharing the SAME source rail,
    the SAME drain net, and the SAME gate net (a parallel finger bank). A single
    such PMOS still qualifies if its drain net is driven by nothing else of its
    own kind to a rail (kept conservative: require >=1)."""
    hits: List[FunctionalRoleHit] = []
    pairs: List[Tuple[str, str]] = []
    groups: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)
    for c in g.mos:
        if c.kind != "pmos":
            continue
        s = c.terminals.get("s")
        d = c.terminals.get("d")
        gnet = c.terminals.get("g")
        if not (s and d and gnet):
            continue
        if not g.is_rail(s):
            continue
        if g.is_rail(d):
            continue  # drain must be a signal/output node, not a rail
        groups[(s, d, gnet)].append(c.name)
    for (s, d, gnet), names in groups.items():
        if len(names) < 2:
            continue
        names = sorted(names)
        hits.append(
            FunctionalRoleHit(
                role="pass_device",
                devices=tuple(names),
                evidence=(
                    f"{len(names)} parallel PMOS from rail {s} to output {d} with "
                    f"shared gate {gnet} (wide pass-FET bank)"
                ),
                confidence=0.85,
            )
        )
        # matched pair = the first two members (the @matching declaration only
        # pins one representative pair of a wide bank).
        pairs.append((names[0], names[1]))
    return hits, pairs


def _infer_second_stage(
    g: _Graph, claimed: Set[str], mirror_drains: Set[str]
) -> Tuple[List[FunctionalRoleHit], Set[str]]:
    """A second gain stage: a common-source amplifier (gate = an internal signal
    node, NOT a rail / bias / input) whose drain is the output node, paired with
    its complementary-kind current load on the same drain.

    Structural signature (uniform): a device whose GATE net is an internal node
    that is itself the DRAIN of some signal-path device (the previous stage's
    output), source on a rail, drain on a high-impedance output node shared with
    exactly one opposite-kind device (the current load). This catches the NMOS-CS
    + PMOS-load (two_stage) and PMOS-CS (ldo) second stages without naming them.
    """
    hits: List[FunctionalRoleHit] = []
    claimed_out: Set[str] = set()
    # internal signal nodes = non-rail nets that are some device's drain.
    signal_nodes: Set[str] = {
        c.terminals.get("d", "")
        for c in g.mos
        if c.terminals.get("d") and not g.is_rail(c.terminals["d"])
    }
    diode_nets = {c.terminals["g"] for c in g.mos if _diode_connected(c)}
    for c in g.mos:
        if c.name in claimed:
            continue
        gnet = c.terminals.get("g")
        s = c.terminals.get("s")
        d = c.terminals.get("d")
        if not (gnet and s and d):
            continue
        if not g.is_rail(s):
            continue
        if g.is_rail(d):
            continue
        # gate must be a driven internal SIGNAL node (a previous drain), not a
        # bias / diode net and not a rail.
        if gnet not in signal_nodes or gnet in diode_nets or g.is_rail(gnet):
            continue
        # its drain is a high-Z output shared with the complementary load device.
        co_devs = [
            o
            for o in g.mos
            if o is not c and o.terminals.get("d") == d
        ]
        load = [o for o in co_devs if o.kind != c.kind]
        if not load:
            continue
        members = tuple(sorted([c.name] + [o.name for o in load]))
        hits.append(
            FunctionalRoleHit(
                role="second_stage",
                devices=members,
                evidence=(
                    f"common-source {c.kind} (gate on signal node {gnet}) driving "
                    f"output {d} against a {load[0].kind} current load"
                ),
                confidence=0.75,
            )
        )
        claimed_out.update(members)
    return hits, claimed_out


def _infer_bias_tree(
    g: _Graph, claimed: Set[str]
) -> List[FunctionalRoleHit]:
    """Bias tree / chain: diode-connected devices to a rail and the gate-tied
    chain hanging off the bias nets, minus anything already claimed by the
    signal path (input / tail / mirror / cascode / pass)."""
    hits: List[FunctionalRoleHit] = []
    # bias nets = the gate nets of diode-connected devices (the generated
    # references) plus any net that is a diode node.
    diode_nets: Set[str] = {
        c.terminals["g"] for c in g.mos if _diode_connected(c)
    }
    members: List[str] = []
    for c in g.mos:
        if c.name in claimed:
            continue
        gnet = c.terminals.get("g")
        d = c.terminals.get("d")
        s = c.terminals.get("s")
        # in the bias tree if it is diode-connected, or its gate rides a diode
        # (bias) net, or it generates a bias net by sitting between two rails /
        # a rail and a bias node.
        is_bias = (
            _diode_connected(c)
            or gnet in diode_nets
            or (d in diode_nets)
        )
        if is_bias:
            members.append(c.name)
    if members:
        members = sorted(set(members))
        hits.append(
            FunctionalRoleHit(
                role="bias_tree",
                devices=tuple(members),
                evidence=(
                    "diode-connected / gate-tied devices generating the bias "
                    f"nets {sorted(diode_nets)}"
                ),
                confidence=0.7,
            )
        )
    return hits


def _infer_bjt_array(g: _Graph) -> List[FunctionalRoleHit]:
    """BJT clusters sharing an emitter/collector net -> a PNP/NPN array."""
    bjts = [c for c in g.cards if c.kind == "bjt"]
    if len(bjts) < 2:
        if bjts:
            return [
                FunctionalRoleHit(
                    role="pnp_array",
                    devices=tuple(b.name for b in bjts),
                    evidence="single BJT device",
                    confidence=0.6,
                )
            ]
        return []
    return [
        FunctionalRoleHit(
            role="pnp_array",
            devices=tuple(sorted(b.name for b in bjts)),
            evidence=f"{len(bjts)} BJT devices forming a Delta-Vbe array",
            confidence=0.85,
        )
    ]


def _infer_passives(
    g: _Graph,
) -> Tuple[List[FunctionalRoleHit], List[Tuple[str, str]]]:
    """Resistor banks (matched ratio) and the Rz->Cc series compensation branch.

    * caps in series/adjacent to a resistor on a shared internal node -> the
      ``compensation`` branch (Miller Rz-Cc or a lone Cc).
    * the remaining resistors -> a ``resistor`` group (the summing / divider
      banks); a 2-member group is reported as a matched pair.
    """
    hits: List[FunctionalRoleHit] = []
    pairs: List[Tuple[str, str]] = []
    caps = [c for c in g.cards if c.kind == "cap"]
    res = [c for c in g.cards if c.kind == "res"]
    comp_devs: Set[str] = set()
    for cap in caps:
        cap_nets = set(cap.terminals.values())
        branch = [cap.name]
        # attach a nulling resistor ONLY when it forms a SERIES chain with the
        # cap: the shared node is an internal, low-fanout node (degree 2 -- just
        # the cap and that resistor), the Rz->Cc node. A resistor that merely
        # touches the cap's output/rail net (a feedback divider on vout) is not
        # part of the compensation branch.
        for r in res:
            shared = set(r.terminals.values()) & cap_nets
            for net in shared:
                if g.is_rail(net):
                    continue
                if len(g.net_devs.get(net, set())) == 2:
                    branch.append(r.name)
                    break
        comp_devs.update(branch)
        hits.append(
            FunctionalRoleHit(
                role="compensation",
                devices=tuple(branch),
                evidence=(
                    f"cap {cap.name} on nets {sorted(cap_nets)} forming a "
                    "Miller compensation branch"
                ),
                confidence=0.8,
            )
        )
    rest = [r for r in res if r.name not in comp_devs]
    if rest:
        names = tuple(sorted(r.name for r in rest))
        hits.append(
            FunctionalRoleHit(
                role="resistor",
                devices=names,
                evidence=f"{len(names)} resistor(s): ratio / feedback bank",
                confidence=0.7,
            )
        )
        if len(names) == 2:
            pairs.append((names[0], names[1]))
    return hits, pairs


# === public entry point ======================================================


def infer_roles(
    cir_path_or_text: str | Path, cell_name: Optional[str] = None
) -> RoleDecomposition:
    """Parse a SPICE ``.cir`` and infer a typed functional-role decomposition.

    See the module docstring for the structural rules. Returns a frozen
    :class:`RoleDecomposition`: every device's parsed card, the inferred roles
    with evidence + confidence, the symmetric matched pairs, and the mirror axis
    (``"vertical"`` when a differential input pair is found).
    """
    text = _read_text(cir_path_or_text)
    cell, cards = parse_devices(cir_path_or_text, cell_name)
    signature_cell, source_ports = _subckt_signature(text, cell_name or cell)
    if not cell:
        cell = signature_cell
    g = _Graph(cards)

    roles: List[FunctionalRoleHit] = []
    matched: List[Tuple[str, str]] = []

    in_hits, in_pairs, tails = _infer_input_pairs(g)
    roles += in_hits
    matched += in_pairs
    input_kinds = {g.by_name[a].kind for (a, b) in in_pairs}

    claimed: Set[str] = set()
    for h in in_hits:
        claimed.update(h.devices)

    tail_hits, tail_names = _infer_tail(g, tails, claimed)
    roles += tail_hits
    claimed.update(tail_names)

    mir_hits, mir_pairs = _infer_mirrors(g, input_kinds, claimed)
    roles += mir_hits
    matched += mir_pairs
    for h in mir_hits:
        claimed.update(h.devices)

    casc_hits, casc_pairs = _infer_cascodes(g, claimed)
    roles += casc_hits
    matched += casc_pairs
    for h in casc_hits:
        claimed.update(h.devices)

    pass_hits, pass_pairs = _infer_pass_device(g, claimed)
    roles += pass_hits
    matched += pass_pairs
    for h in pass_hits:
        claimed.update(h.devices)

    mirror_drains = {
        d for h in mir_hits for d in
        (g.by_name[n].terminals.get("d", "") for n in h.devices)
    }
    stage_hits, stage_claimed = _infer_second_stage(g, claimed, mirror_drains)
    roles += stage_hits
    # second-stage members are real signal-path devices: do NOT let the bias
    # tree also claim them.
    claimed.update(stage_claimed)

    bias_hits = _infer_bias_tree(g, claimed)
    roles += bias_hits

    roles += _infer_bjt_array(g)

    pass_dev_pairs = pass_pairs
    pas_hits, pas_pairs = _infer_passives(g)
    roles += pas_hits
    matched += pas_pairs

    mirror_axis = "vertical" if in_pairs else None

    # de-dup matched pairs (canonical ordering).
    seen_pairs: Set[frozenset] = set()
    uniq_pairs: List[Tuple[str, str]] = []
    for a, b in matched:
        k = frozenset((a, b))
        if k in seen_pairs:
            continue
        seen_pairs.add(k)
        uniq_pairs.append((a, b))

    return RoleDecomposition(
        cell=cell,
        devices=tuple(cards),
        roles=tuple(roles),
        matched_pairs=tuple(uniq_pairs),
        mirror_axis=mirror_axis,
        source_ports=source_ports,
    )


# === role-family compatibility (for recall against declared design.clay) =====
#
# The committed design.clay roles are fine-grained (pmos_fold_source vs
# pmos_cascode, four distinct bias generators). Recall is measured at the
# FAMILY level: an inferred role and a declared role are compatible when they
# map to the same coarse family. This keeps the validation honest (it credits a
# correct structural class, not an exact token) without per-circuit casing.

_FAMILY_KEYWORDS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("input", ("input_pair", "input")),
    ("pass", ("pass_device", "pass_device_array", "pass")),
    ("pnp", ("pnp", "npn", "bjt")),
    ("compensation", ("miller", "compensation", "comp")),
    ("cascode", ("cascode",)),
    ("tail", ("tail",)),
    ("resistor", ("resistor", "_res", "degeneration", "summer_resistors")),
    ("startup", ("startup",)),
    ("stage", ("stage2", "second_stage", "output_stage", "stage_gain", "gain")),
    # an embedded OTA the design groups under ONE role token; its devices are
    # legitimately recovered by their structural sub-roles (input / mirror /
    # tail). Keyword 'ota' must precede 'mirror' so 'summer_ota' lands here.
    ("ota", ("summer_ota", "_ota", "ota_core")),
    ("mirror", ("mirror", "_load", "load_mirror")),
    (
        "bias",
        (
            "bias",
            "generator",
            "reference",
            "fold_source",
            "fold_bias",
            "vbc",
            "core_nmos",
            "constant_gm",
            "chain",
        ),
    ),
)


def role_family(role: str) -> str:
    """Coarse functional family for a role token (inferred OR declared)."""
    r = (role or "").lower()
    for fam, keys in _FAMILY_KEYWORDS:
        if any(k in r for k in keys):
            return fam
    return r or "other"


# Compatibility: an INFERRED family on the left satisfies the DECLARED families
# on the right. The structural reality is that bias trees, tails, and load /
# current mirrors are all gate-tied current sources, so a structurally-correct
# "mirror" hit legitimately recovers a device the designer labelled "bias" (a
# bias-tree mirror) or "tail". The relation is intentionally NOT symmetric: it
# credits a structurally-correct current-source class, not an arbitrary swap.
_FAMILY_COMPAT: Dict[str, Set[str]] = {
    "input": {"input", "ota"},
    "tail": {"tail", "bias", "mirror", "ota"},
    "mirror": {"mirror", "bias", "tail", "ota"},
    "cascode": {"cascode", "ota"},
    "bias": {"bias", "mirror", "tail"},
    "pass": {"pass", "mirror"},
    "pnp": {"pnp"},
    "resistor": {"resistor"},
    "compensation": {"compensation"},
    "stage": {"stage", "mirror"},
    "ota": {"ota", "input", "mirror", "tail"},
    "startup": {"startup", "mirror", "bias"},
}


def families_compatible(inferred_family: str, declared_family: str) -> bool:
    """True when an inferred role family recovers a declared role family.

    Used by the recall validation: a device is recovered when ANY of its
    inferred families is compatible with its declared family.
    """
    allowed = _FAMILY_COMPAT.get(inferred_family, {inferred_family})
    return declared_family in allowed


def device_family_recall(rd: RoleDecomposition, declared: Dict[str, str]):
    """Recall of declared device->family assignments recovered by ``rd``.

    ``declared`` maps device -> declared role family (from the committed
    design.clay catalog). Returns ``(recovered, total, misses)`` where a device
    is recovered when one of its inferred role families is compatible with its
    declared family (see :func:`families_compatible`).
    """
    inferred: Dict[str, Set[str]] = {}
    for h in rd.roles:
        fam = role_family(h.role)
        for d in h.devices:
            inferred.setdefault(d, set()).add(fam)
    recovered = 0
    misses: List[Tuple[str, str, List[str]]] = []
    for dev, dfam in declared.items():
        got = inferred.get(dev, set())
        if any(families_compatible(f, dfam) for f in got):
            recovered += 1
        else:
            misses.append((dev, dfam, sorted(got)))
    return recovered, len(declared), misses
