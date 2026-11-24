"""Compose one simulatable PEX from same-GDS complementary extractors.

Magic contributes calibrated distributed wire R/C.  The full-extraction LVS
database contributes device-class topology and measured passive values that a
Magic technology file may omit or mis-bind.  Passive values always come from
the same GDS.  A declared resistor chain may be used only to invert a foundry
deck's proven series-reduction equivalence: the extracted total is preserved
against the actual unsimplified segments. Source nominal resistor values are
not measurements, particularly across process migration.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .base import PexReport


class JointPexIntegrityError(ValueError):
    """Fail-closed error when Magic extraction contains shorts or ground collapse."""

    def __init__(self, message: str, *, diagnostics: Mapping[str, object]) -> None:
        super().__init__(message)
        self.diagnostics = dict(diagnostics)


_GROUND_PORT_ALIASES = frozenset({
    "gnd", "gnda", "agnd", "dgnd", "vss", "vssa", "avss", "dvss",
    "sub", "vsub", "gnd!", "vss!",
})


@dataclass(frozen=True)
class FullExtractionCap:
    a: str
    b: str
    capacitance_f: float


@dataclass(frozen=True)
class FullExtractionResistor:
    a: str
    b: str
    bulk: str
    resistance_ohm: float
    model: str


@dataclass(frozen=True)
class FullExtractionPassives:
    caps: tuple[FullExtractionCap, ...]
    resistors: tuple[FullExtractionResistor, ...]


@dataclass(frozen=True)
class DeclaredResistorSegment:
    """One physical resistor segment declared by portable .clay intent."""

    name: str
    a: str
    b: str
    resistance_ohm: float


@dataclass(frozen=True)
class DeclaredResistorNetwork:
    """Resistor segmentation plus nets that forbid series reduction.

    protected_nets are touched by a non-resistor device.  They cannot be
    reconstructed as resistor-only degree-two midpoints even if two resistor
    declarations also meet there.
    """

    segments: tuple[DeclaredResistorSegment, ...]
    protected_nets: tuple[str, ...]


def declared_resistor_network_from_cell(cell: object) -> DeclaredResistorNetwork:
    """Project generic CellDef device intent into a resistor-chain proof.

    This deliberately keys only on portable device class/pin/value fields.  It
    contains no circuit, instance-name, net-name, or PDK special case.
    """

    segments: list[DeclaredResistorSegment] = []
    protected: set[str] = set()
    for device in tuple(getattr(cell, "devices", ()) or ()):
        kind = str(getattr(device, "device_class", "")).strip().casefold()
        pins = {
            str(name).strip().casefold(): str(net).strip().casefold()
            for name, net in dict(getattr(device, "pins", {}) or {}).items()
        }
        if kind in {"res", "resistor"}:
            a = pins.get("p", pins.get("a"))
            b = pins.get("n", pins.get("b"))
            sizing = dict(getattr(device, "sizing", {}) or {})
            value = sizing.get("R", sizing.get("value"))
            if not a or not b or value is None:
                raise ValueError(
                    f"declared resistor {getattr(device, 'instance_name', '')!r} "
                    "lacks two terminals or a value"
                )
            resistance = float(value)
            if not resistance > 0.0:
                raise ValueError(
                    f"declared resistor {getattr(device, 'instance_name', '')!r} "
                    "must have positive resistance"
                )
            segments.append(
                DeclaredResistorSegment(
                    str(getattr(device, "instance_name", "")),
                    a,
                    b,
                    resistance,
                )
            )
        else:
            protected.update(net for net in pins.values() if net)
    return DeclaredResistorNetwork(tuple(segments), tuple(sorted(protected)))


_SUBCKT_RE = re.compile(r"(?im)^\s*\.subckt\s+(\S+)\s+([^\r\n]+)$")
_MAGIC_TERMINAL_RE = re.compile(
    r"(?<![A-Za-z0-9_#$])(?P<node>[A-Za-z0-9_#$]+\.t\d+)(?![A-Za-z0-9_#$])",
    re.IGNORECASE,
)


def _subckt_ports(text: str, cell_name: str) -> tuple[str, ...]:
    # Magic wraps large internally-labelled cells over standard SPICE ``+``
    # continuation records.  Parsing only the physical ``.subckt`` line makes
    # every continued port look absent even though it is present in the raw
    # extracted artifact.  Consume the same logical-card representation used by
    # the full-extraction composer below.
    for card in _spice_logical_cards(text):
        tokens = card.split()
        if len(tokens) < 2 or tokens[0].casefold() != ".subckt":
            continue
        if tokens[1].casefold() != cell_name.casefold():
            continue
        ports: list[str] = []
        for token in tokens[2:]:
            if "=" in token:
                break
            ports.append(token)
        if ports:
            return tuple(ports)
    raise ValueError(f"Magic PEX has no .subckt {cell_name} port declaration")


def _canonicalize_magic_terminal_nodes(
    lines: list[str],
    *,
    cell_name: str,
) -> list[str]:
    """Rename unstable ``net.tN`` nodes from stable device-pin incidence.

    Magic's extresist output can assign different ``.tN`` ordinals to the same
    device terminals across identical extractions.  Device instance names and
    terminal positions remain stable, so use their incidence sets as the
    canonical labels.  This runs before technology resistor cards are removed:
    those cards are the only anchors for their three distributed-RC terminals.
    """

    target_indices: list[int] = []
    terminal_nodes: dict[str, str] = {}
    anchors: dict[str, list[tuple[str, int]]] = {}
    in_target = False
    for index, line in enumerate(lines):
        tokens = line.strip().split()
        if tokens and tokens[0].casefold() == ".subckt" and len(tokens) >= 2:
            in_target = tokens[1].casefold() == cell_name.casefold()
        if not in_target:
            continue
        target_indices.append(index)
        for match in _MAGIC_TERMINAL_RE.finditer(line):
            node = match.group("node")
            terminal_nodes.setdefault(node.casefold(), node)

        if tokens and tokens[0][:1].casefold() in {"x", "m", "d", "q"}:
            positional: list[str] = []
            for token in tokens[1:]:
                if "=" in token:
                    break
                positional.append(token)
            for pin_index, node in enumerate(positional[:-1]):
                if _MAGIC_TERMINAL_RE.fullmatch(node):
                    anchors.setdefault(node.casefold(), []).append(
                        (tokens[0].casefold(), pin_index)
                    )
        if tokens and tokens[0].casefold() == ".ends":
            in_target = False

    unanchored = sorted(set(terminal_nodes) - set(anchors))
    if unanchored:
        raise ValueError(
            f"Magic PEX {cell_name} has terminal node(s) without device-pin anchors: "
            + ", ".join(terminal_nodes[node] for node in unanchored)
        )

    by_base: dict[str, list[str]] = {}
    base_spelling: dict[str, str] = {}
    for node_key, node in terminal_nodes.items():
        base = node.rsplit(".", 1)[0]
        base_key = base.casefold()
        base_spelling.setdefault(base_key, base)
        by_base.setdefault(base_key, []).append(node_key)

    replacements: dict[str, str] = {}
    for base_key, nodes in by_base.items():
        signatures = [tuple(sorted(anchors[node])) for node in nodes]
        if len(set(signatures)) != len(signatures):
            raise ValueError(
                f"Magic PEX {cell_name} has ambiguous terminal anchors on "
                f"{base_spelling[base_key]}"
            )
        ordered = sorted(nodes, key=lambda node: tuple(sorted(anchors[node])))
        for ordinal, node in enumerate(ordered):
            replacements[node] = f"{base_spelling[base_key]}.t{ordinal}"

    rewritten = list(lines)
    for index in target_indices:
        rewritten[index] = _MAGIC_TERMINAL_RE.sub(
            lambda match: replacements[match.group("node").casefold()],
            rewritten[index],
        )
    return rewritten


def _canonicalize_magic_cap_cards(
    lines: list[str],
    *,
    cell_name: str,
) -> list[str]:
    """Sort symmetric Magic C cards and replace unstable instance ordinals."""

    slots: list[int] = []
    cards: list[tuple[str, str, tuple[str, ...]]] = []
    in_target = False
    for index, line in enumerate(lines):
        tokens = line.strip().split()
        if not tokens:
            continue
        if tokens[0].casefold() == ".subckt" and len(tokens) >= 2:
            in_target = tokens[1].casefold() == cell_name.casefold()
            continue
        if tokens[0].casefold() == ".ends" and in_target:
            in_target = False
            continue
        if not in_target or tokens[0][:1].casefold() != "c" or len(tokens) < 4:
            continue
        left, right = sorted((tokens[1], tokens[2]), key=str.casefold)
        slots.append(index)
        cards.append((left, right, tuple(tokens[3:])))

    ordered = sorted(
        cards,
        key=lambda card: (
            card[0].casefold(),
            card[1].casefold(),
            tuple(token.casefold() for token in card[2]),
        ),
    )
    rewritten = list(lines)
    for ordinal, (slot, card) in enumerate(zip(slots, ordered)):
        left, right, suffix = card
        rewritten[slot] = f"C{ordinal} {left} {right} {' '.join(suffix)}"
    return rewritten


def _resolve_semantic_ground_net(
    raw_ports: Sequence[str],
    external_ports: Sequence[str],
    *,
    ground_net: str | None = None,
    port_roles: Mapping[object, object] | None = None,
) -> tuple[str | None, tuple[str, ...]]:
    """Resolve the subckt's single semantic ground/substrate port without hardcoding."""
    by_fold = {str(port).casefold(): str(port) for port in raw_ports}
    if ground_net is not None:
        explicit = str(ground_net).strip()
        if not explicit:
            raise ValueError("Magic PEX explicit ground_net must be non-empty")
        if explicit == "0":
            return "0", ()
        key = explicit.casefold()
        if key not in by_fold:
            raise ValueError(
                f"Magic PEX declared ground net {ground_net!r} is absent from "
                "extracted subckt ports"
            )
        return by_fold[key], ()

    if port_roles:
        from clay.port_roles import declared_substrate_ports

        declared = declared_substrate_ports(port_roles)
        if declared:
            key = declared[0].casefold()
            if key not in by_fold:
                raise ValueError(
                    f"Magic PEX declared substrate port {declared[0]!r} is absent "
                    "from extracted subckt ports"
                )
            return by_fold[key], ()

    candidates = [
        str(port)
        for port in external_ports
        if str(port).strip().casefold() in _GROUND_PORT_ALIASES
    ]
    if not candidates:
        candidates = [
            str(port)
            for port in raw_ports
            if str(port).strip().casefold() in _GROUND_PORT_ALIASES
        ]
    seen: dict[str, str] = {}
    for item in candidates:
        seen.setdefault(item.casefold(), by_fold.get(item.casefold(), item))
    unique = tuple(seen.values())
    if len(unique) == 1:
        return unique[0], ()
    if len(unique) > 1:
        return None, unique
    return None, ()


def _parse_resistor_ohm_safe(token: str) -> float | None:
    try:
        from .magic_parser import _parse_spice_value

        value = float(_parse_spice_value(token))
    except Exception:
        return None
    if not math.isfinite(value) or value <= 0.0:
        return None
    return value


def _prune_isolated_resistor_stub_islands(
    lines: list[str],
    *,
    anchor_folds: set[str],
    r_edges: dict[str, set[str]],
    known: set[str],
    base_fn: Callable[[str], str],
) -> tuple[list[str], int]:
    """Drop DC-disconnected R+C stub islands left after stripping tech resistors."""
    visited: set[str] = set()
    prune_folds: set[str] = set()
    for start in sorted(r_edges):
        if start in visited:
            continue
        comp: set[str] = set()
        stack = [start]
        visited.add(start)
        while stack:
            cur = stack.pop()
            comp.add(cur)
            for nxt in r_edges.get(cur, ()):
                if nxt not in visited:
                    visited.add(nxt)
                    stack.append(nxt)
        if comp & anchor_folds:
            continue
        if any(base_fn(node) in known for node in comp):
            continue
        prune_folds.update(comp)
    if not prune_folds:
        return lines, 0

    filtered: list[str] = []
    removed_cards = 0
    for line in lines:
        tokens = line.strip().split()
        prefix = tokens[0][:1].casefold() if tokens else ""
        if (
            prefix in {"r", "c"}
            and len(tokens) >= 4
            and (tokens[1].casefold() in prune_folds or tokens[2].casefold() in prune_folds)
        ):
            removed_cards += 1
            continue
        filtered.append(line)
    return filtered, removed_cards


def _compute_dc_conductive_folds(
    anchor_folds: set[str],
    r_edges: dict[str, set[str]],
) -> set[str]:
    """Return nodes connected by kept resistors to at least one DC anchor."""
    conductive_folds = set(anchor_folds)
    queue = list(anchor_folds & r_edges.keys())
    while queue:
        cur = queue.pop()
        for nxt in r_edges.get(cur, ()):
            if nxt not in conductive_folds:
                conductive_folds.add(nxt)
                queue.append(nxt)
    return conductive_folds


def _condition_magic_rc(
    raw_text: str,
    *,
    cell_name: str,
    known_nets: Sequence[str],
    resistor_models: set[str],
    ground_net: str | None = None,
    ambiguous_ground_nets: tuple[str, ...] = (),
    external_ports: Sequence[str] = (),
    caps: Sequence[FullExtractionCap] = (),
    isolated_body_nets: Iterable[str] | None = None,
    low_resistance_short_ohm: float = 10.0,
    strict_cross_net: bool = False,
) -> str:
    """Keep within-net distributed RC and fail closed on shorts/ground collapse."""
    known = {str(net).casefold() for net in known_nets}
    external_set = {str(net).casefold() for net in external_ports}
    gnd_key = ground_net.casefold() if ground_net is not None else None
    iso_keys = (
        {str(net).casefold() for net in isolated_body_nets}
        if isolated_body_nets is not None
        else set()
    )

    def base(node: str) -> str:
        return str(node).split(".", 1)[0].casefold()

    canonical_lines = _canonicalize_magic_terminal_nodes(
        raw_text.splitlines(), cell_name=cell_name
    )
    kept: list[str] = []
    removed_cross_net = 0
    removed_resistor_devices = 0
    removed_cap_devices = 0
    cross_net_records: list[dict[str, object]] = []
    folded_models = {model.casefold() for model in resistor_models}
    has_lvsdb_caps = bool(caps)
    folded_cap_models = {
        str(getattr(cap, "model", "") or "").casefold()
        for cap in (caps or ())
        if getattr(cap, "model", None)
    }
    for line in canonical_lines:
        tokens = line.strip().split()
        if tokens and tokens[0][:1].casefold() == "r" and len(tokens) >= 4:
            left, right = base(tokens[1]), base(tokens[2])
            if left in known and right in known and left != right:
                removed_cross_net += 1
                cross_net_records.append(
                    {
                        "instance": tokens[0],
                        "left": left,
                        "right": right,
                        "resistance_ohm": _parse_resistor_ohm_safe(tokens[3]),
                    }
                )
                continue
        if tokens and tokens[0][:1].casefold() == "x":
            card_models = {token.casefold() for token in tokens[1:] if "=" not in token}
            if card_models & folded_models:
                removed_resistor_devices += 1
                continue
            if has_lvsdb_caps and (
                (card_models & folded_cap_models)
                or any(tok.startswith("cap_mim_") for tok in card_models)
            ):
                removed_cap_devices += 1
                continue
        kept.append(line)

    anchor_folds: set[str] = {
        str(port).casefold() for port in _subckt_ports(raw_text, cell_name)
    } | {"0"}
    raw_r_edges: dict[str, set[str]] = {}
    for line in kept:
        tokens = line.strip().split()
        if not tokens:
            continue
        prefix = tokens[0][:1].casefold()
        if prefix == "r" and len(tokens) >= 4:
            left_fold, right_fold = tokens[1].casefold(), tokens[2].casefold()
            raw_r_edges.setdefault(left_fold, set()).add(right_fold)
            raw_r_edges.setdefault(right_fold, set()).add(left_fold)
        elif prefix in {"x", "m", "d", "q"}:
            positional = []
            for token in tokens[1:]:
                if "=" in token:
                    break
                positional.append(token)
            term_tokens = positional[:-1] if len(positional) >= 2 else positional
            anchor_folds.update(token.casefold() for token in term_tokens)

    removed_isolated_rc = 0
    if (removed_resistor_devices > 0 or removed_cap_devices > 0) and raw_r_edges:
        kept, removed_isolated_rc = _prune_isolated_resistor_stub_islands(
            kept,
            anchor_folds=anchor_folds,
            r_edges=raw_r_edges,
            known=known,
            base_fn=base,
        )

    kept = _canonicalize_magic_cap_cards(kept, cell_name=cell_name)

    capacitive_nodes: dict[str, str] = {}
    r_edges: dict[str, set[str]] = {}
    r_nodes: dict[str, str] = {}
    active_ground_touches = 0
    nmos_bodies: list[str] = []
    nmos_sources: list[str] = []
    bjt_count = 0
    for line in kept:
        tokens = line.strip().split()
        if not tokens:
            continue
        prefix = tokens[0][:1].casefold()
        if prefix == "c" and len(tokens) >= 4:
            for tok in (tokens[1], tokens[2]):
                capacitive_nodes.setdefault(tok.casefold(), tok)
        elif prefix == "r" and len(tokens) >= 4:
            left_fold, right_fold = tokens[1].casefold(), tokens[2].casefold()
            r_edges.setdefault(left_fold, set()).add(right_fold)
            r_edges.setdefault(right_fold, set()).add(left_fold)
            r_nodes.setdefault(left_fold, tokens[1])
            r_nodes.setdefault(right_fold, tokens[2])
        elif prefix in {"x", "m", "d", "q"}:
            positional = []
            for token in tokens[1:]:
                if "=" in token:
                    break
                positional.append(token)
            if prefix in {"x", "m", "q"} and len(positional) >= 2:
                model = positional[-1].casefold()
                term_bases = [base(tok) for tok in positional[:-1]]
                if gnd_key is not None and gnd_key in term_bases:
                    active_ground_touches += 1
                if prefix == "q" or any(
                    tok in model for tok in ("pnp", "npn", "vpnp", "vnpn")
                ):
                    bjt_count += 1
                elif (
                    model.startswith("nfet")
                    or model.startswith("nmos")
                    or "__nfet" in model
                    or "_nfet" in model
                ) and len(term_bases) >= 4:
                    nmos_sources.append(term_bases[2])
                    nmos_bodies.append(term_bases[3])

    conductive_folds = _compute_dc_conductive_folds(anchor_folds, r_edges)

    cap_pairs = {
        frozenset((str(cap.a).casefold(), str(cap.b).casefold()))
        for cap in caps
    }
    ground_shorts = [
        item
        for item in cross_net_records
        if gnd_key is not None and gnd_key in (item["left"], item["right"])
    ]
    low_ohm_shorts = [
        item
        for item in cross_net_records
        if item["resistance_ohm"] is not None
        and float(item["resistance_ohm"]) <= low_resistance_short_ohm
    ]
    non_iso_nmos_bodies = [b for b in nmos_bodies if b not in iso_keys]
    collapsed_ground = False
    if gnd_key is not None and active_ground_touches == 0:
        if ground_shorts:
            collapsed_ground = True
        elif bjt_count > 0 or len(non_iso_nmos_bodies) > 1:
            collapsed_ground = True
        elif len(non_iso_nmos_bodies) == 1 and (
            isolated_body_nets is not None or non_iso_nmos_bodies[0] not in external_set
        ):
            collapsed_ground = True
    elif (
        gnd_key is not None
        and isolated_body_nets is not None
        and non_iso_nmos_bodies
        and gnd_key not in non_iso_nmos_bodies
    ):
        collapsed_ground = True

    failing_records = (
        list(cross_net_records)
        if (strict_cross_net and cross_net_records)
        else []
    )
    if not failing_records:
        seen_ids: set[int] = set()
        for item in ground_shorts + low_ohm_shorts:
            if id(item) not in seen_ids:
                seen_ids.add(id(item))
                failing_records.append(item)

    if failing_records or collapsed_ground:
        pair_summary: dict[tuple[str, str], dict[str, object]] = {}
        for item in failing_records or cross_net_records:
            pair = tuple(sorted((str(item["left"]), str(item["right"]))))
            entry = pair_summary.setdefault(
                pair, {"nets": pair, "count": 0, "min_resistance_ohm": None}
            )
            entry["count"] = int(entry["count"]) + 1
            r_val = item["resistance_ohm"]
            if r_val is not None:
                prev = entry["min_resistance_ohm"]
                entry["min_resistance_ohm"] = (
                    float(r_val) if prev is None else min(float(prev), float(r_val))
                )
        shorted_pairs = sorted(pair_summary.values(), key=lambda x: x["nets"])
        suspected: list[str] = []
        if any(frozenset(entry["nets"]) in cap_pairs for entry in shorted_pairs) or (
            cap_pairs and low_ohm_shorts
        ):
            suspected.append("gf180_mim_option_mismatch")
        if collapsed_ground or ground_shorts:
            suspected.append("collapsed_substrate_ground")
            suspected.append("isolated_substrate_island")
        if low_ohm_shorts:
            suspected.append("cross_net_low_resistance_short")
        collapsed_targets = sorted(
            set(non_iso_nmos_bodies)
            | {
                str(n)
                for item in ground_shorts
                for n in (item["left"], item["right"])
                if n != gnd_key
            }
        )
        causes_str = ", ".join(dict.fromkeys(suspected)) or "cross_net_short"
        pairs_str = "; ".join(
            f"{e['nets'][0]}<->{e['nets'][1]}"
            + (
                f" (min_R={e['min_resistance_ohm']:.4g} ohm, count={e['count']})"
                if e["min_resistance_ohm"] is not None
                else f" (count={e['count']})"
            )
            for e in shorted_pairs[:8]
        )
        diag = {
            "cell_name": cell_name,
            "ground_net": ground_net,
            "active_device_ground_count": active_ground_touches,
            "collapsed_ground_to": collapsed_targets,
            "removed_cross_net_resistors": removed_cross_net,
            "shorted_pairs": shorted_pairs,
            "suspected_causes": list(dict.fromkeys(suspected)),
        }
        raise JointPexIntegrityError(
            f"Magic PEX failed closed for {cell_name}: {causes_str}. "
            f"Ground {ground_net!r} active_touches={active_ground_touches}, "
            f"collapsed_to={collapsed_targets!r}; shorted_pairs=[{pairs_str}]. "
            "Refusing to silently strip cross-net shorts.",
            diagnostics=diag,
        )

    unanchored_nodes: dict[str, str] = dict(capacitive_nodes)
    for node_fold, node_name in r_nodes.items():
        unanchored_nodes.setdefault(node_fold, node_name)
    floating = sorted(
        node_name
        for node_fold, node_name in unanchored_nodes.items()
        if node_fold not in conductive_folds and node_name != "0"
    )
    if floating:
        if ground_net is None:
            if ambiguous_ground_nets:
                raise ValueError(
                    f"Magic PEX cell {cell_name!r} has {len(floating)} floating "
                    f"capacitive node(s) and ambiguous ground candidates "
                    f"{ambiguous_ground_nets!r}; pass explicit ground_net or port_roles"
                )
            raise ValueError(
                f"Magic PEX cell {cell_name!r} has {len(floating)} floating "
                "capacitive node(s) and no declared ground/substrate net; "
                "pass ground_net or port_roles"
            )
    additions = [
        f"* PEX net-isolation: removed {removed_cross_net} cross-net wire "
        f"resistor(s) and {removed_resistor_devices} technology resistor card(s)."
    ]
    if removed_isolated_rc:
        additions.append(
            f"* PEX isolated-stub cleanup: removed {removed_isolated_rc} orphaned RC card(s)."
        )
    if floating:
        assert ground_net is not None
        additions.extend(
            f"RPEX_FLOAT_{index} {node} {ground_net} 1e15"
            for index, node in enumerate(floating)
        )
    conditioned = "\n".join(kept)
    end_re = re.compile(r"(?im)^\s*\.ends(?:\s+\S+)?\s*$")
    conditioned, count = end_re.subn(
        "\n".join(additions) + "\n" + r"\g<0>", conditioned, count=1
    )
    if count != 1:
        raise ValueError(f"Magic PEX subckt {cell_name} has no .ends")
    return conditioned + "\n"


def _mos_alias(name: str, target_model: str) -> list[str]:
    """Return a four-terminal parameter-preserving Magic-model adapter."""
    return [
        f".subckt {name} d g s b w=1u l=0.28u ad=0 pd=0 as=0 ps=0",
        f"XCORE d g s b {target_model} W='w' L='l' "
        "ad={ad} pd={pd} as={as} ps={ps}",
        f".ends {name}",
    ]


def _bjt_alias(name: str, target_model: str) -> list[str]:
    """Return a three-terminal C/B/E Magic-BJT alias wrapper."""
    return [
        f".subckt {name} c b e",
        f"XCORE c b e {target_model}",
        f".ends {name}",
    ]


def _resolve_bjt_magic_aliases(
    conditioned_text: str,
    *,
    pdk: str | None,
) -> dict[str, str]:
    """Resolve raw Magic BJT subcircuit names (e.g. pnp_10p00x00p42) to PDK sim models."""
    instantiated_models: set[str] = set()
    for card in _spice_logical_cards(conditioned_text):
        tokens = card.split()
        if not tokens or tokens[0][:1].casefold() not in {"x", "q"}:
            continue
        positional = [tok for tok in tokens[1:] if "=" not in tok]
        if positional:
            instantiated_models.add(positional[-1].casefold())
    if not instantiated_models or not pdk:
        return {}
    from clay.pdk import get_pdk, load_all

    load_all()
    try:
        adapter = get_pdk(pdk)
    except Exception:
        return {}
    bjt_classes = adapter.extra.get("magic_bjt_device_classes", {})
    if not isinstance(bjt_classes, Mapping):
        return {}
    out: dict[str, str] = {}
    for raw_name, semantic in bjt_classes.items():
        if str(raw_name).casefold() not in instantiated_models:
            continue
        binding = (adapter.bjt_bindings or {}).get(str(semantic).casefold())
        if binding is not None and getattr(binding, "simulator_model", None):
            out[str(raw_name)] = str(binding.simulator_model)
    return out


def _resolve_dnw_magic_aliases(
    conditioned_text: str,
    *,
    pdk: str | None,
    magic_device_models: Mapping[str, str],
) -> dict[str, str]:
    """Supplement Magic alias wrappers when raw text instantiates DNW FET models."""
    merged = dict(magic_device_models)
    instantiated_models: set[str] = set()
    for card in _spice_logical_cards(conditioned_text):
        tokens = card.split()
        if not tokens or tokens[0][:1].casefold() != "x":
            continue
        positional = [tok for tok in tokens[1:] if "=" not in tok]
        if positional:
            instantiated_models.add(positional[-1].casefold())
    if not instantiated_models:
        return merged
    pdk_candidates = [pdk] if pdk else ["gf180mcu"]
    for pdk_id in pdk_candidates:
        try:
            all_aliases = magic_device_models_for_pdk(pdk_id, include_deep_nwell=True)
        except Exception:
            continue
        for raw_alias, sim_model in all_aliases.items():
            if raw_alias.casefold() in instantiated_models:
                merged.setdefault(raw_alias, sim_model)
    return merged


def compose_joint_pex_text(
    raw_magic_text: str,
    *,
    cell_name: str,
    external_ports: Sequence[str],
    required_layout_nets: Iterable[str],
    caps: Sequence[FullExtractionCap],
    resistors: Sequence[FullExtractionResistor],
    magic_device_models: Mapping[str, str],
    ground_net: str | None = None,
    port_roles: Mapping[object, object] | None = None,
    pdk: str | None = None,
    isolated_body_nets: Iterable[str] | None = None,
    low_resistance_short_ohm: float = 10.0,
    strict_cross_net: bool = False,
    declared_resistor_network: DeclaredResistorNetwork | None = None,
    uncombined_extracted_text: str | None = None,
    uncombined_net_mapping: Mapping[str, str] | None = None,
    physical_resistor_reference_text: str | None = None,
    resistor_evidence: dict | None = None,
) -> str:
    """Wrap Magic RC and add only passives measured by full extraction."""
    raw_ports = _subckt_ports(raw_magic_text, cell_name)
    by_fold = {port.casefold(): port for port in raw_ports}
    required = tuple(str(net) for net in required_layout_nets)
    missing = sorted(net for net in required if net.casefold() not in by_fold)
    if missing:
        raise ValueError(
            "Magic PEX missing required layout nets: " + ", ".join(missing)
        )
    missing_external = sorted(
        str(port) for port in external_ports if str(port).casefold() not in by_fold
    )
    if missing_external:
        raise ValueError(
            "Magic PEX missing external ports: " + ", ".join(missing_external)
        )
    evidence_nets = {
        net.casefold()
        for cap in caps
        for net in (cap.a, cap.b)
    } | {
        net.casefold()
        for resistor in resistors
        for net in (resistor.a, resistor.b, resistor.bulk)
    }
    lost_evidence = sorted(net for net in evidence_nets if net not in by_fold)
    if lost_evidence:
        raise ValueError(
            "Magic PEX lost full-extraction passive nets: "
            + ", ".join(lost_evidence)
        )
    for cap in caps:
        if not cap.capacitance_f > 0:
            raise ValueError("full-extraction capacitance must be positive")
    for resistor in resistors:
        if not resistor.resistance_ohm > 0:
            raise ValueError("full-extraction resistance must be positive")

    resolved_ground, ambiguous_grounds = _resolve_semantic_ground_net(
        raw_ports,
        external_ports,
        ground_net=ground_net,
        port_roles=port_roles,
    )

    conditioned = _condition_magic_rc(
        raw_magic_text,
        cell_name=cell_name,
        known_nets=required,
        resistor_models={resistor.model for resistor in resistors},
        ground_net=resolved_ground,
        ambiguous_ground_nets=ambiguous_grounds,
        external_ports=external_ports,
        caps=caps,
        isolated_body_nets=isolated_body_nets,
        low_resistance_short_ohm=low_resistance_short_ohm,
        strict_cross_net=strict_cross_net,
    )
    raw_name = f"{cell_name}__magic_raw"
    header_re = re.compile(
        rf"(?im)^(\s*\.subckt\s+){re.escape(cell_name)}(?=\s)"
    )
    renamed, count = header_re.subn(rf"\g<1>{raw_name}", conditioned, count=1)
    if count != 1:
        raise ValueError(f"could not rename raw Magic subckt {cell_name}")

    external = {str(port).casefold(): str(port) for port in external_ports}

    def wrapper_net(net: str) -> str:
        raw = by_fold.get(str(net).casefold(), str(net))
        return external.get(raw.casefold(), f"int_{raw}")

    effective_models = _resolve_dnw_magic_aliases(
        conditioned, pdk=pdk, magic_device_models=magic_device_models
    )
    bjt_aliases = _resolve_bjt_magic_aliases(conditioned, pdk=pdk)

    extracted_res_rows = [
        _ExtractedResistor(
            card_index=index,
            a=str(resistor.a).casefold(),
            b=str(resistor.b).casefold(),
            resistance_ohm=float(resistor.resistance_ohm),
            value_token=f"{resistor.resistance_ohm:.12g}",
            model=str(resistor.model).casefold(),
        )
        for index, resistor in enumerate(resistors)
    ]
    actual_resistor_cards = None
    series_evidence: dict[str, object] = {}
    if physical_resistor_reference_text is not None and uncombined_extracted_text is not None:
        from .uncombined_resistors import bind_actual_resistor_graph

        actual_resistor_cards, actual_proof = bind_actual_resistor_graph(
            uncombined_extracted_text,
            physical_resistor_reference_text,
            cell_name=cell_name,
            net_mapping=uncombined_net_mapping or {},
            aggregate_rows=tuple(extracted_res_rows),
        )
        if resistor_evidence is not None:
            resistor_evidence.update(actual_proof)
    refinements: dict[int, tuple[DeclaredResistorSegment, ...]] = {}
    if actual_resistor_cards is None and declared_resistor_network is not None:
        refinements = _series_refinements(
            tuple(extracted_res_rows),
            declared_resistor_network,
            external_nets={k.casefold() for k in external_ports},
            raw_text=uncombined_extracted_text,
            raw_net_mapping=uncombined_net_mapping or {},
            cell_name=cell_name,
            evidence=series_evidence,
        )
        if resistor_evidence is not None and series_evidence:
            resistor_evidence.update(series_evidence)

    lines = [
        "* Joint full-extraction + calibrated Magic-RC PEX.",
        "* Every injected passive value and terminal comes from the same GDS LVSDB.",
        "",
    ]
    for alias, model in sorted(effective_models.items()):
        lines.extend(_mos_alias(str(alias), str(model)))
    for alias, model in sorted(bjt_aliases.items()):
        lines.extend(_bjt_alias(str(alias), str(model)))
    lines.extend(
        (
            "",
            renamed.rstrip(),
            "",
            f".subckt {cell_name} {' '.join(str(p) for p in external_ports)}",
            f"XRAW {' '.join(wrapper_net(port) for port in raw_ports)} {raw_name}",
        )
    )
    if actual_resistor_cards is not None:
        for card in actual_resistor_cards:
            toks = card.split()
            lines.append(
                f"{toks[0]} {wrapper_net(toks[1])} {wrapper_net(toks[2])} {toks[3]} "
                "$ full-extraction resistor"
            )
    else:
        for index, resistor in enumerate(resistors):
            refinement = refinements.get(index)
            if refinement:
                for seg_index, segment in enumerate(refinement):
                    lines.append(
                        f"RPEX_DEVICE_{index}_{seg_index} {wrapper_net(segment.a)} "
                        f"{wrapper_net(segment.b)} {segment.resistance_ohm:.12g} "
                        "$ full-extraction resistor"
                    )
            else:
                lines.append(
                    f"RPEX_DEVICE_{index} {wrapper_net(resistor.a)} "
                    f"{wrapper_net(resistor.b)} {resistor.resistance_ohm:.12g} "
                    "$ full-extraction resistor"
                )
    for index, cap in enumerate(caps):
        lines.append(
            f"CPEX_DEVICE_{index} {wrapper_net(cap.a)} {wrapper_net(cap.b)} "
            f"{cap.capacitance_f:.12g} $ full-extraction capacitor"
        )
    lines.extend((f".ends {cell_name}", ""))
    return "\n".join(lines)


def read_full_extraction_passives(lvsdb_path: str | Path) -> FullExtractionPassives:
    """Read MIM and poly-resistor topology/value evidence from an LVSDB."""
    import klayout.db as kdb

    from clay.layout.route.full_extraction_body_check import _lvs_net_xref_mapping

    path = Path(lvsdb_path)
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"full-extraction LVSDB is absent or empty: {path}")
    database = kdb.LayoutVsSchematic()
    database.read(str(path))
    net_xref = {
        str(raw).strip().lstrip("$").casefold(): str(schematic).strip().casefold()
        for raw, schematic in _lvs_net_xref_mapping(database).items()
    }

    def canonical(net: object) -> str:
        token = str(net).rsplit(":", 1)[-1].strip()
        return net_xref.get(token.lstrip("$").casefold(), token.casefold())

    netlist = database.netlist()
    if netlist is None:
        raise ValueError(f"full-extraction LVSDB has no layout netlist: {path}")
    caps: list[FullExtractionCap] = []
    resistors: list[FullExtractionResistor] = []
    for circuit in netlist.each_circuit():
        for device in circuit.each_device():
            device_class = device.device_class()
            name = str(device_class.name)
            terminals = {
                definition.name: definition.id()
                for definition in device_class.terminal_definitions()
            }
            if "mim" in name.casefold():
                if not {"A", "B"} <= terminals.keys():
                    raise ValueError(f"full-extraction MIM class {name!r} lacks A/B")
                value = float(device.parameter("C"))
                caps.append(
                    FullExtractionCap(
                        canonical(device.net_for_terminal(terminals["A"])),
                        canonical(device.net_for_terminal(terminals["B"])),
                        value,
                    )
                )
            elif "poly" in name.casefold():
                if not {"A", "B", "W"} <= terminals.keys():
                    raise ValueError(
                        f"full-extraction resistor class {name!r} lacks A/B/W"
                    )
                value = float(device.parameter("R"))
                resistors.append(
                    FullExtractionResistor(
                        canonical(device.net_for_terminal(terminals["A"])),
                        canonical(device.net_for_terminal(terminals["B"])),
                        canonical(device.net_for_terminal(terminals["W"])),
                        value,
                        name,
                    )
                )
    return FullExtractionPassives(tuple(caps), tuple(resistors))


def write_joint_pex(
    raw_magic_path: str | Path,
    output_path: str | Path,
    *,
    lvsdb_path: str | Path,
    cell_name: str,
    external_ports: Sequence[str],
    required_layout_nets: Iterable[str],
    magic_device_models: Mapping[str, str],
    ground_net: str | None = None,
    port_roles: Mapping[object, object] | None = None,
    pdk: str | None = None,
    isolated_body_nets: Iterable[str] | None = None,
    low_resistance_short_ohm: float = 10.0,
    strict_cross_net: bool = False,
    declared_resistor_network: DeclaredResistorNetwork | None = None,
    uncombined_extracted_text: str | None = None,
    uncombined_net_mapping: Mapping[str, str] | None = None,
    physical_resistor_reference_text: str | None = None,
    resistor_evidence: dict | None = None,
) -> tuple[Path, FullExtractionPassives]:
    evidence = read_full_extraction_passives(lvsdb_path)
    raw_mapping = uncombined_net_mapping
    if raw_mapping is None and uncombined_extracted_text is not None:
        import klayout.db as kdb
        from clay.layout.route.full_extraction_body_check import _lvs_net_xref_mapping

        db = kdb.LayoutVsSchematic()
        db.read(str(lvsdb_path))
        raw_mapping = _lvs_net_xref_mapping(db)
    text = compose_joint_pex_text(
        Path(raw_magic_path).read_text(encoding="utf-8"),
        cell_name=cell_name,
        external_ports=external_ports,
        required_layout_nets=required_layout_nets,
        caps=evidence.caps,
        resistors=evidence.resistors,
        magic_device_models=magic_device_models,
        ground_net=ground_net,
        port_roles=port_roles,
        pdk=pdk,
        isolated_body_nets=isolated_body_nets,
        low_resistance_short_ohm=low_resistance_short_ohm,
        strict_cross_net=strict_cross_net,
        declared_resistor_network=declared_resistor_network,
        uncombined_extracted_text=uncombined_extracted_text,
        uncombined_net_mapping=raw_mapping,
        physical_resistor_reference_text=physical_resistor_reference_text,
        resistor_evidence=resistor_evidence,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    return output, evidence


def magic_device_models_for_pdk(
    pdk: str,
    *,
    include_deep_nwell: bool = False,
) -> dict[str, str]:
    """Resolve raw Magic aliases through one PDK adapter's model registry."""
    from clay.pdk import get_pdk

    adapter = get_pdk(pdk)
    classes = adapter.extra.get("magic_device_classes")
    if not isinstance(classes, Mapping) or not classes:
        raise ValueError(f"PDK {pdk!r} declares no Magic device-class mapping")
    canonical = adapter.lvs_reference_models()
    out: dict[str, str] = {}
    for raw_name, device_class in classes.items():
        target = canonical.get(str(device_class))
        if not target:
            raise ValueError(
                f"PDK {pdk!r} Magic model {raw_name!r} names unsupported "
                f"device class {device_class!r}"
            )
        out[str(raw_name)] = str(target)
    if include_deep_nwell:
        deep = adapter.deep_nwell()
        if deep is not None:
            dnw_classes = adapter.extra.get("magic_deep_nwell_device_classes", {})
            if isinstance(dnw_classes, Mapping):
                for raw_name, device_class in dnw_classes.items():
                    sim_target = deep.sim_models.get(str(device_class))
                    if not sim_target:
                        raise ValueError(
                            f"PDK {pdk!r} Magic DNW model {raw_name!r} has no "
                            f"deep-N-well simulator mapping for {device_class!r}"
                        )
                    out.setdefault(str(raw_name), str(sim_target))
            for semantic, lvs_name in deep.lvs_models.items():
                sim_target = deep.sim_models.get(str(semantic))
                if lvs_name and sim_target:
                    out.setdefault(str(lvs_name), str(sim_target))
    return out


def full_extraction_mos_models_for_pdk(
    pdk: str,
) -> dict[str, tuple[str, str]]:
    """Map full-extraction MOS classes to simulator model/card DATA."""
    from clay.layout.route.lvs_gate import _resolved_lvs_config
    from clay.pdk import get_pdk, load_all

    load_all()
    adapter = get_pdk(pdk)
    config = _resolved_lvs_config(pdk, full_extraction=True)
    extracted = dict(config.get("devices") or {})
    sim = adapter.extra.get("ngspice_device_bindings", {})
    if not isinstance(sim, Mapping):
        sim = {}

    def target(semantic: str) -> tuple[str, str]:
        base_model = str(adapter.device_bindings.get(semantic, ""))
        model = str(sim.get(semantic, base_model))
        if not model:
            raise ValueError(f"PDK {pdk!r} has no simulator {semantic} binding")
        # A sample-device bin cannot be imposed on every extracted geometry.
        # Let ngspice select the installed family's W/L bin for each MOS card.
        if base_model and re.fullmatch(re.escape(base_model) + r"\.\d+", model):
            model = base_model
        folded = model.casefold()
        prefix = adapter.extra.get("ngspice_instance_prefixes", {}).get(
            semantic, "X" if "_ckt" in folded else "M")
        if prefix not in {"M", "X"}:
            raise ValueError(f"PDK {pdk!r} has invalid simulator card {prefix!r}")
        return model, prefix

    out: dict[str, tuple[str, str]] = {}
    for polarity in ("n", "p"):
        source = extracted.get(f"{polarity}mos")
        if not source:
            raise ValueError(f"PDK {pdk!r} full extraction lacks {polarity}mos class")
        simulator_binding = target(f"{polarity}mos")
        out[str(source)] = simulator_binding
        # Some real decks emit the already-normalized simulator subcircuit name
        # while their config advertises the primitive extraction alias.  Both
        # spellings are PDK-owned DATA and lower to the same card; accepting the
        # target as an input alias avoids circuit-name branches without relaxing
        # the unknown-model failure for any other class.
        out.setdefault(simulator_binding[0], simulator_binding)
        for semantic in (f"{polarity}mos", f"{polarity}mos3v"):
            if semantic not in adapter.device_bindings and semantic not in sim:
                continue
            binding = target(semantic)
            for alias in (adapter.device_bindings.get(semantic), sim.get(semantic), binding[0]):
                if alias:
                    out.setdefault(str(alias), binding)

    deep = adapter.deep_nwell()
    if deep is not None:
        for semantic, source in sorted(deep.lvs_models.items()):
            if not str(semantic).startswith("nmos"):
                raise ValueError(
                    f"PDK {pdk!r} deep-N-well LVS class {semantic!r} is not NMOS"
                )
            sim_model = deep.sim_models.get(str(semantic))
            if not sim_model:
                raise ValueError(
                    f"PDK {pdk!r} deep-N-well LVS class {semantic!r} has no "
                    "simulator binding"
                )
            folded = str(sim_model).casefold()
            prefix = (
                "X"
                if "_ckt" in folded
                else "M"
            )
            # A PDK may use the same extracted class for ordinary and isolated
            # devices (external_pdk/external_pdk MMRF thin NMOS).  Preserve the established
            # circuit-domain binding in that ambiguous case; unique DNW classes
            # gain their exact PDK-owned simulator model here.
            out.setdefault(str(source), (str(sim_model), prefix))
            out.setdefault(str(sim_model), (str(sim_model), prefix))
    return out


def full_extraction_bjt_models_for_pdk(pdk: str) -> dict[str, str]:
    """Map extracted three-terminal BJT classes to simulator models.

    Both names come from the adapter's semantic BJT binding.  A conflicting
    duplicate LVS class is rejected because choosing either simulator model
    would make the PEX netlist ambiguous.
    """
    from clay.pdk import get_pdk, load_all

    load_all()
    adapter = get_pdk(pdk)
    out: dict[str, str] = {}
    for semantic, binding in sorted(adapter.bjt_bindings.items()):
        source = str(binding.lvs_model)
        target = str(binding.simulator_model)
        if not source or not target:
            raise ValueError(
                f"PDK {pdk!r} BJT binding {semantic!r} has no LVS/simulator model"
            )
        prior = out.get(source)
        if prior is not None and prior != target:
            raise ValueError(
                f"PDK {pdk!r} BJT LVS class {source!r} has conflicting "
                f"simulator models {prior!r} and {target!r}"
            )
        out[source] = target
    return out


def full_extraction_mos_instance_parameters_for_pdk(pdk: str) -> dict[str, str]:
    """Return adapter-owned simulator parameters appended to extracted MOS cards."""
    from clay.pdk import get_pdk, load_all

    load_all()
    raw = get_pdk(pdk).extra.get("ngspice_mos_instance_parameters", {})
    if not isinstance(raw, Mapping):
        raise ValueError(f"PDK {pdk!r} has invalid MOS instance parameters")
    return {str(name): str(value) for name, value in sorted(raw.items())}


def _spice_logical_cards(text: str) -> tuple[str, ...]:
    """Join standard ``+`` continuation records into fail-closed SPICE cards."""
    cards: list[str] = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("*"):
            continue
        if stripped.startswith("+"):
            if not cards:
                raise ValueError("orphan full-extraction SPICE continuation")
            payload = stripped[1:].strip()
            if not payload:
                raise ValueError("empty full-extraction SPICE continuation")
            cards[-1] += " " + payload
        else:
            cards.append(stripped)
    return tuple(cards)


@dataclass(frozen=True)
class _ExtractedResistor:
    card_index: int
    a: str
    b: str
    resistance_ohm: float
    value_token: str
    model: str | None = None
    length: float | None = None
    width: float | None = None


def _parse_extracted_resistor(
    tokens: Sequence[str],
    *,
    card_index: int,
    node,
) -> _ExtractedResistor:
    """Parse either two-terminal SPICE R or KLayout A/B/W bulk-R cards."""

    parameter_index = next(
        (index for index, token in enumerate(tokens) if "=" in token),
        len(tokens),
    )
    positional = tuple(tokens[:parameter_index])
    if len(positional) < 4:
        raise ValueError(f"malformed full-extraction resistor card {tokens[0]!r}")
    # Ordinary SPICE: name A B value [model].
    # KLayout resistor_with_bulk: name A B W value model.
    value_index = 4 if len(positional) >= 6 else 3
    if value_index >= len(positional):
        raise ValueError(f"malformed full-extraction resistor card {tokens[0]!r}")
    from .magic_parser import _parse_spice_value

    try:
        resistance = float(_parse_spice_value(positional[value_index]))
    except ValueError as exc:
        raise ValueError(
            f"unparseable full-extraction resistor value "
            f"{positional[value_index]!r}"
        ) from exc
    if not math.isfinite(resistance) or not resistance > 0.0:
        raise ValueError("full-extraction resistor value must be positive")
    # Resolve the substrate/well terminal too.  It is not emitted by an ideal
    # two-terminal post-simulation resistor, but an unbound bulk is still an
    # extraction-topology error and must not be hidden.
    if value_index == 4:
        node(positional[3])
    parameters = {key.casefold(): value for token in tokens[parameter_index:]
                  if "=" in token for key, value in (token.split("=", 1),)}
    geometry = {}
    for name in ("l", "w"):
        value = parameters.get(name)
        geometry[name] = None if value is None else float(_parse_spice_value(value))
        if geometry[name] is not None and (not math.isfinite(geometry[name]) or geometry[name] <= 0):
            raise ValueError("full-extraction resistor geometry must be finite and positive")
    return _ExtractedResistor(
        card_index=card_index,
        a=node(positional[1]),
        b=node(positional[2]),
        resistance_ohm=resistance,
        value_token=positional[value_index],
        model=(positional[value_index + 1].casefold() if len(positional) > value_index + 1 else None),
        length=geometry["l"],
        width=geometry["w"],
    )


def _declared_series_chains(
    network: DeclaredResistorNetwork,
    *,
    external_nets: set[str],
) -> tuple[tuple[DeclaredResistorSegment, ...], ...]:
    """Return maximal, unbranched resistor chains through reducible midpoints."""

    segments = tuple(
        DeclaredResistorSegment(
            str(segment.name),
            str(segment.a).casefold(),
            str(segment.b).casefold(),
            float(segment.resistance_ohm),
        )
        for segment in network.segments
    )
    for segment in segments:
        if not segment.a or not segment.b or not segment.resistance_ohm > 0.0:
            raise ValueError(
                f"declared resistor segment {segment.name!r} is incomplete"
            )
    incident: dict[str, list[int]] = {}
    for index, segment in enumerate(segments):
        incident.setdefault(segment.a, []).append(index)
        incident.setdefault(segment.b, []).append(index)
    protected = {str(net).casefold() for net in network.protected_nets}
    foldable = {
        net
        for net, indexes in incident.items()
        if len(indexes) == 2
        and net not in external_nets
        and net not in protected
    }
    neighbors: dict[int, set[int]] = {index: set() for index in range(len(segments))}
    for net in foldable:
        left, right = incident[net]
        neighbors[left].add(right)
        neighbors[right].add(left)

    chains: list[tuple[DeclaredResistorSegment, ...]] = []
    visited: set[int] = set()
    for seed in range(len(segments)):
        if seed in visited:
            continue
        stack = [seed]
        component: set[int] = set()
        while stack:
            index = stack.pop()
            if index in component:
                continue
            component.add(index)
            stack.extend(neighbors[index] - component)
        visited.update(component)
        if len(component) < 2:
            continue
        endpoints = sorted(
            {
                net
                for index in component
                for net in (segments[index].a, segments[index].b)
                if net not in foldable
            }
        )
        if len(endpoints) != 2:
            raise ValueError(
                "declared resistor reduction component is not an unbranched path"
            )
        current = endpoints[0]
        remaining = set(component)
        ordered: list[DeclaredResistorSegment] = []
        while remaining:
            touching = sorted(
                index
                for index in remaining
                if current in (segments[index].a, segments[index].b)
            )
            if len(touching) != 1:
                raise ValueError(
                    "declared resistor reduction path has ambiguous ordering"
                )
            index = touching[0]
            segment = segments[index]
            other = segment.b if segment.a == current else segment.a
            ordered.append(
                DeclaredResistorSegment(
                    segment.name,
                    current,
                    other,
                    segment.resistance_ohm,
                )
            )
            remaining.remove(index)
            current = other
        if current != endpoints[1]:
            raise ValueError("declared resistor reduction path does not terminate")
        chains.append(tuple(ordered))
    return tuple(
        sorted(
            chains,
            key=lambda chain: (
                chain[0].a,
                chain[-1].b,
                tuple(segment.name.casefold() for segment in chain),
            ),
        )
    )


def _series_refinements(
    extracted: Sequence[_ExtractedResistor],
    network: DeclaredResistorNetwork | None,
    *,
    external_nets: set[str],
    relative_tolerance: float = 0.02,
    raw_text: str | None = None,
    raw_net_mapping: Mapping[str, str] | None = None,
    cell_name: str | None = None,
    evidence: dict | None = None,
) -> dict[int, tuple[DeclaredResistorSegment, ...]]:
    """Form candidate chain labels; publication also requires the actual raw graph.

    Endpoint/total checks cannot establish physical midpoint existence. The
    caller must bind these candidate labels to pre-simplify extracted segments
    before returning any composed PEX.
    """

    if network is None or not network.segments:
        return {}
    chains = _declared_series_chains(network, external_nets=external_nets)
    extracted_by_ends: dict[frozenset[str], list[_ExtractedResistor]] = {}
    for resistor in extracted:
        extracted_by_ends.setdefault(frozenset((resistor.a, resistor.b)), []).append(
            resistor
        )
    chain_by_ends: dict[
        frozenset[str], list[tuple[DeclaredResistorSegment, ...]]
    ] = {}
    for chain in chains:
        chain_by_ends.setdefault(
            frozenset((chain[0].a, chain[-1].b)), []
        ).append(chain)

    raw_rows = raw_chains = None
    refinements: dict[int, tuple[DeclaredResistorSegment, ...]] = {}
    for ends, candidate_chains in sorted(
        chain_by_ends.items(), key=lambda item: tuple(sorted(item[0]))
    ):
        candidate_extracted = extracted_by_ends.get(ends, [])
        if not candidate_extracted:
            continue
        if len(candidate_chains) != 1 or len(candidate_extracted) != 1:
            raise ValueError(
                "series-resistor refinement is ambiguous for endpoints "
                + ", ".join(sorted(ends))
            )
        chain = candidate_chains[0]
        resistor = candidate_extracted[0]
        declared_total = sum(segment.resistance_ohm for segment in chain)
        if raw_text is None or raw_net_mapping is None or cell_name is None:
            raise ValueError(
                "series-resistor refinement requires actual unsimplified GDS segment evidence"
            )
        if raw_rows is None:
            from .uncombined_resistors import _anchored_chains, _resistors
            from .uncombined_mos import _name

            raw_rows = _resistors(raw_text, cell_name)
            raw_nodes = {node for row, *_ in raw_rows for node in (row.a, row.b)}
            targets = {node for row in extracted for node in (row.a, row.b)}
            mapping = {_name(key): _name(value) for key, value in raw_net_mapping.items()}
            for name in set(mapping.values()) & raw_nodes:
                mapping.setdefault(name, name)
            anchors = {key: value for key, value in mapping.items()
                       if key in raw_nodes and value in targets}
            raw_chains = _anchored_chains(raw_rows, anchors)
        actual_paths = raw_chains.get(tuple(sorted(ends)), ())
        if len(actual_paths) != 1:
            raise ValueError("Actual resistor chain mapping is missing or ambiguous at " + repr(tuple(sorted(ends))))
        actual_order, actual_nodes = actual_paths[0]
        if len(actual_order) != len(chain):
            raise ValueError("Actual resistor segment count differs from source topology at " + repr(tuple(sorted(ends))))
        measured = tuple(raw_rows[index][0] for index in actual_order)
        actual_total = sum(segment.resistance_ohm for segment in measured)
        if not math.isclose(actual_total, resistor.resistance_ohm, rel_tol=1e-6, abs_tol=1e-12):
            raise ValueError("Actual unsimplified resistor total differs from extracted aggregate at " + repr(tuple(sorted(ends))))
        for segment in measured:
            if resistor.model is not None and segment.model != resistor.model:
                raise ValueError("Actual resistor model differs from extracted aggregate")
            if all(value is not None for value in (resistor.length, resistor.width, segment.length, segment.width)):
                aggregate_sheet = resistor.resistance_ohm * resistor.width / resistor.length
                segment_sheet = segment.resistance_ohm * segment.width / segment.length
                if not math.isclose(aggregate_sheet, segment_sheet, rel_tol=1e-6, abs_tol=1e-12):
                    raise ValueError("Actual resistor segment geometry/model resistance differs from aggregate")
        refinements[resistor.card_index] = tuple(
            DeclaredResistorSegment(
                segment.name,
                segment.a,
                segment.b,
                actual.resistance_ohm,
            )
            for segment, actual in zip(chain, measured)
        )
        if evidence is not None:
            evidence.setdefault("series_realizations", []).append({
                "endpoints": sorted(ends), "source_nominal_total_ohm": declared_total,
                "actual_raw_total_ohm": actual_total,
                "extracted_aggregate_ohm": resistor.resistance_ohm,
                "source_nominal_used_as_measurement": False,
                "nominal_relative_difference": (actual_total - declared_total) / declared_total,
                "actual_geometry_and_aggregate_verified": True,
                "actual_raw_nodes": actual_nodes,
                "actual_segment_values_ohm": [segment.resistance_ohm for segment in measured],
                "actual_segment_models": [segment.model for segment in measured],
            })
    return refinements


def compose_full_extraction_geometric_pex_text(
    extracted_text: str,
    *,
    cell_name: str,
    external_ports: Sequence[str],
    net_mapping: Mapping[str, str],
    mos_models: Mapping[str, tuple[str, str]],
    report: PexReport,
    mos_instance_parameters: Mapping[str, str] | None = None,
    bjt_models: Mapping[str, str] | None = None,
    declared_resistor_network: DeclaredResistorNetwork | None = None,
    uncombined_extracted_text: str | None = None,
    uncombined_net_mapping: Mapping[str, str] | None = None,
    resistor_evidence: dict | None = None,
    physical_resistor_reference_text: str | None = None,
) -> str:
    """Compose simulatable PEX from full-LVS topology + uncalibrated route C.

    Device topology and passive values come from the same-run extracted
    netlist.  Declared resistor segmentation can invert a deck's series
    reduction only after same-extraction raw graph/node/per-segment proof;
    actual unsimplified segment values remain authoritative. The report may add
    capacitance only on explicitly named route nets; anonymous/unbound
    components fail closed.
    """
    if report.calibrated or report.backend != "geometric":
        raise ValueError("geometric full-extraction PEX requires calibrated=false")
    header = re.search(
        rf"(?im)^[ \t]*\.subckt[ \t]+{re.escape(cell_name)}[ \t]+([^\r\n]+)$",
        extracted_text,
    )
    if header is None:
        raise ValueError(f"extracted netlist has no .subckt {cell_name}")
    raw_ports = tuple(header.group(1).split())
    end = re.search(r"(?im)^\s*\.ends(?:\s+\S+)?\s*$", extracted_text[header.end():])
    if end is None:
        raise ValueError(f"extracted .subckt {cell_name} has no .ends")
    body = extracted_text[header.end(): header.end() + end.start()]

    mapping = {
        str(raw).strip().lstrip("$").casefold(): str(target).casefold()
        for raw, target in net_mapping.items()
    }
    for match in re.finditer(
        r"(?im)^\s*\*\s*net\s+(\S+)\s+(\S+)\s*$", extracted_text
    ):
        mapping[match.group(1).lstrip("$").casefold()] = match.group(2).casefold()
    external = {str(port).casefold() for port in external_ports}
    raw_port_targets: list[str] = []
    unbound_raw_ports: list[str] = []
    for raw_port in raw_ports:
        key = str(raw_port).strip().lstrip("$").casefold()
        target = mapping.get(key, key if key in external else None)
        if target is None:
            unbound_raw_ports.append(str(raw_port))
        else:
            raw_port_targets.append(target)
    if unbound_raw_ports:
        raise ValueError(
            "unbound extracted top-level ports: " + ", ".join(unbound_raw_ports)
        )
    missing_external = sorted(external - set(raw_port_targets))
    if missing_external:
        raise ValueError(
            "full-extraction net mapping misses external ports: "
            + ", ".join(missing_external)
        )
    duplicated_external = sorted(
        port for port in external if raw_port_targets.count(port) != 1
    )
    if duplicated_external:
        raise ValueError(
            "full-extraction external ports are not one-to-one: "
            + ", ".join(duplicated_external)
        )

    def node(raw: str) -> str:
        key = str(raw).strip().lstrip("$").casefold()
        value = mapping.get(key, key if key in external else None)
        if value is None:
            raise ValueError(f"unbound full-extraction net {raw!r}")
        return value

    model_map = {
        str(source).casefold(): (str(target), str(prefix).upper())
        for source, (target, prefix) in mos_models.items()
    }
    bjt_model_map = {
        str(source).casefold(): str(target)
        for source, target in (bjt_models or {}).items()
    }
    instance_parameters = {
        str(name): str(value)
        for name, value in sorted((mos_instance_parameters or {}).items())
    }
    instance_suffix = " ".join(
        f"{name}={value}" for name, value in instance_parameters.items()
    )
    lines = [
        "* Full-extraction topology + geometric route-capacitance PEX.",
        "* calibrated=false; field_solver=false; no foundry ITF/TLU+ claim.",
        f"* constants_source={report.constants_source}",
        "",
        f".subckt {cell_name} {' '.join(str(p) for p in external_ports)}",
    ]
    valid_nets = set(external)
    cards = _spice_logical_cards(body)
    extracted_resistors: dict[int, _ExtractedResistor] = {}
    for card_index, raw in enumerate(cards):
        tokens = raw.split()
        if tokens and tokens[0][:1].upper() == "R":
            extracted_resistors[card_index] = _parse_extracted_resistor(
                tokens,
                card_index=card_index,
                node=node,
            )
    actual_resistor_cards = None
    actual_graph_proof = None
    if physical_resistor_reference_text is not None and uncombined_extracted_text is not None:
        from .uncombined_resistors import bind_actual_resistor_graph

        actual_resistor_cards, actual_graph_proof = bind_actual_resistor_graph(
            uncombined_extracted_text, physical_resistor_reference_text,
            cell_name=cell_name,
            net_mapping=(mapping if uncombined_net_mapping is None else uncombined_net_mapping),
            aggregate_rows=tuple(extracted_resistors.values()),
        )
        actual_graph_proof['series_realizations'] = []
        if declared_resistor_network is not None:
            for chain in _declared_series_chains(declared_resistor_network, external_nets=external):
                ends = frozenset((chain[0].a, chain[-1].b))
                aggregate = [row for row in extracted_resistors.values() if frozenset((row.a, row.b)) == ends]
                if len(aggregate) != 1:
                    continue
                nominal = sum(row.resistance_ohm for row in chain)
                actual = aggregate[0].resistance_ohm
                actual_graph_proof['series_realizations'].append(dict(
                    endpoints=sorted(ends), source_nominal_total_ohm=nominal,
                    actual_equivalent_resistance_ohm=actual,
                    source_nominal_used_as_measurement=False,
                    nominal_relative_difference=(actual-nominal)/nominal,
                    actual_graph_and_aggregate_conductance_verified=True))
    series_evidence = {}
    resistor_refinements = {} if actual_resistor_cards is not None else _series_refinements(
        tuple(extracted_resistors.values()),
        declared_resistor_network,
        external_nets=external,
        raw_text=uncombined_extracted_text,
        raw_net_mapping=(mapping if uncombined_net_mapping is None else uncombined_net_mapping),
        cell_name=cell_name,
        evidence=series_evidence,
    )
    if resistor_refinements and uncombined_extracted_text is None:
        raise ValueError("series-resistor refinement requires actual unsimplified GDS segment evidence")
    mos_index = bjt_index = res_index = cap_index = 0
    for card_index, raw in enumerate(cards):
        tokens = raw.split()
        kind = tokens[0][:1].upper()
        if kind == "M" and len(tokens) >= 6:
            source_model = tokens[5].casefold()
            if source_model not in model_map:
                raise ValueError(f"unbound full-extraction MOS class {tokens[5]!r}")
            target_model, prefix = model_map[source_model]
            nodes = [node(value) for value in tokens[1:5]]
            valid_nets.update(nodes)
            lines.append(
                f"{prefix}PEX_{mos_index} {' '.join(nodes)} {target_model} "
                + " ".join(tokens[6:])
                + (f" {instance_suffix}" if instance_suffix else "")
            )
            mos_index += 1
        elif kind == "Q" and len(tokens) >= 5:
            source_model = tokens[4].casefold()
            if source_model not in bjt_model_map:
                raise ValueError(f"unbound full-extraction BJT class {tokens[4]!r}")
            nodes = [node(value) for value in tokens[1:4]]
            valid_nets.update(nodes)
            suffix = " ".join(tokens[5:])
            lines.append(
                f"QPEX_{bjt_index} {' '.join(nodes)} "
                f"{bjt_model_map[source_model]}"
                + (f" {suffix}" if suffix else "")
            )
            bjt_index += 1
        elif kind == "R" and len(tokens) >= 4:
            if actual_resistor_cards is not None:
                if res_index == 0:
                    lines.extend(actual_resistor_cards)
                    for actual_card in actual_resistor_cards:
                        valid_nets.update(actual_card.split()[1:3])
                res_index += 1
                continue
            extracted_resistor = extracted_resistors[card_index]
            refinement = resistor_refinements.get(card_index)
            if refinement:
                for segment_index, segment in enumerate(refinement):
                    valid_nets.update((segment.a, segment.b))
                    lines.append(
                        f"RPEX_DEVICE_{res_index}_{segment_index} "
                        f"{segment.a} {segment.b} "
                        f"{segment.resistance_ohm:.12g}"
                    )
            else:
                valid_nets.update(
                    (extracted_resistor.a, extracted_resistor.b)
                )
                lines.append(
                    f"RPEX_DEVICE_{res_index} {extracted_resistor.a} "
                    f"{extracted_resistor.b} {extracted_resistor.value_token}"
                )
            res_index += 1
        elif kind == "C" and len(tokens) >= 4:
            a, b = node(tokens[1]), node(tokens[2])
            valid_nets.update((a, b))
            lines.append(f"CPEX_DEVICE_{cap_index} {a} {b} {tokens[3]}")
            cap_index += 1
        else:
            raise ValueError(f"unsupported full-extraction card {tokens[0]!r}")

    parasitic_lines: list[str] = []
    # SPICE node 0 is the global reference, including inside a subcircuit.
    # It is a valid explicitly selected PEX reference even when the extracted
    # device-only subcircuit has no terminal connected to ground. All other
    # names still require an extracted/source binding.
    valid_nets.add("0")
    for index, cap in enumerate(report.caps):
        a, b = str(cap.a).casefold(), str(cap.b).casefold()
        if a not in valid_nets:
            raise ValueError(f"unbound geometric PEX net {cap.a!r}")
        if b not in valid_nets:
            raise ValueError(f"unbound geometric PEX net {cap.b!r}")
        if a == b or float(cap.cap_aF) <= 0.0:
            continue
        parasitic_lines.append(
            f"CPEX_WIRE_{index} {a} {b} {float(cap.cap_aF) * 1e-18:.12g}"
        )
    if not parasitic_lines:
        raise ValueError("geometric PEX produced no bound positive route capacitance")
    lines.extend(parasitic_lines)
    lines.extend((f".ends {cell_name}", ""))
    result = "\n".join(lines)
    if actual_graph_proof is not None:
        actual_graph_proof.update(route_cap_cards_unchanged=True,
                                  retained_cap_card_count=cap_index + len(parasitic_lines))
        if resistor_evidence is not None:
            resistor_evidence.update(actual_graph_proof)
    elif uncombined_extracted_text is not None:
        from .uncombined_resistors import restore_uncombined_resistors

        result, proof = restore_uncombined_resistors(
            result, uncombined_extracted_text, cell_name=cell_name,
            net_mapping=mapping if uncombined_net_mapping is None else uncombined_net_mapping)
        if resistor_evidence is not None:
            resistor_evidence.update(proof)
            resistor_evidence.update(series_evidence)
    return result


__all__ = [
    "DeclaredResistorNetwork",
    "DeclaredResistorSegment",
    "FullExtractionCap",
    "FullExtractionPassives",
    "FullExtractionResistor",
    "JointPexIntegrityError",
    "compose_joint_pex_text",
    "declared_resistor_network_from_cell",
    "compose_full_extraction_geometric_pex_text",
    "full_extraction_bjt_models_for_pdk",
    "full_extraction_mos_instance_parameters_for_pdk",
    "full_extraction_mos_models_for_pdk",
    "magic_device_models_for_pdk",
    "read_full_extraction_passives",
    "write_joint_pex",
]
