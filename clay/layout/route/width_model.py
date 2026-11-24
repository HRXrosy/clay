"""Design-derived, leakage-free routing-width model.

The leakage incident, transposed to routing widths: a per-net wire width MEASURED from the
reference layout and replayed is the routing analogue of memorising reference device
coordinates -- it is design-specific (keyed to the reference net names) and does not survive
transfer to a held-out design or PDK. This module DERIVES per-net width from the netlist
alone (net role + a relative drive-current proxy) plus a documented EM methodology policy,
floored at the target PDK's ``min_width``. Every derived width is tagged
:class:`IntentSource.DESIGN`, so :meth:`RoutingIntent.leakage_count` is 0.

HONEST SCOPE. The PDK adapters carry no current-density (J_max) table, so this model does
NOT claim to reproduce the reference's absolute micron widths (e.g. the 8um external_pdk rails). It
derives the width *class* (power/signal/cap) and a *relative* ranking (more incident drive
=> wider), and assigns class-appropriate widths by a stated policy. The claim it supports is
(a) reproduce the reference width-CLASS without reading reference widths, and (b) generalise
to held-out designs/PDKs where a memorised per-net table cannot. See
:mod:`clay.layout.route.falsify`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from clay.layout.route.intent import Declared, IntentSource

# Supply-net base names. The designer NAMES a supply rail (vdd/gnd/...); recognising those
# names is design knowledge, not a reference measurement.
DEFAULT_POWER_BASES = frozenset({
    "vdd", "vss", "gnd", "vcc", "vee", "vpwr", "vgnd", "vnb", "vpb",
    "avdd", "avss", "dvdd", "dvss", "vddd", "vssd", "vdda", "vssa",
    "vbat", "vref", "vbias",  # vbias/vref are rail-like (low-current); still floored as signal-ish below
})
# of the above, treat only true supplies as 'power' (wide); biases are signal-current.
_TRUE_SUPPLY = frozenset({
    "vdd", "vss", "gnd", "vcc", "vee", "vpwr", "vgnd",
    "avdd", "avss", "dvdd", "dvss", "vddd", "vssd", "vdda", "vssa", "vbat",
})


@dataclass(frozen=True)
class WidthPolicy:
    """EM/IR routing-width methodology policy (documented constants, NOT reference values).

    These are designer policy knobs applied uniformly to every design/PDK; they read nothing
    from any reference layout.

    RAIL-CLASS MULTIPLES (PDK-min-relative, re-floored). The rail-class widths below are the
    GENERAL design-derived default the router materialises for ANY cell through the normal
    ``route_cell`` path -- a net's width comes from its net ROLE / rail-class (+ a relative
    drive proxy), never a per-circuit switch. ``power_min_um`` / ``cap_min_um`` are absolute
    EM/IR methodology floors; ``rail_min_mult`` / ``global_min_mult`` express the same
    rail-class widening as a MULTIPLE of the target PDK's ``min_width`` so it transfers across
    PDKs. Every rail-class width is finally re-floored on ``min_width`` (never sub-min).

    BYTE-IDENTITY DEFAULT. ``global_min_mult`` (the high-fan-out / global SIGNAL-net widening
    multiplier) and ``rail_min_mult`` default to ``1.0`` -- i.e. INERT: a regular signal net
    stays at the PDK ``min_width`` and a power rail stays at its ``power_min_um`` floor, so a
    cell that neither declares ``@wire_width`` nor derives a wider rail-class (the comparator /
    LDO) is byte-identical to the pre-rail-class route. A design widens its power/global wiring
    either by DECLARING ``@wire_width`` (the ``declared`` override in :func:`derive_width_intent`)
    or by supplying a policy with ``global_min_mult`` / ``rail_min_mult`` > 1 -- both are
    design-derived / designer-authored, never a build flag and never a circuit name branch.
    """
    power_min_um: float = 4.0          # power-rail methodology floor (EM/IR headroom)
    cap_min_um: float = 10.0           # wide plate-strap floor for big caps
    signal_floor_um: float = 0.0       # 0 -> use PDK min_width as the signal floor
    em_um_per_proxy: float = 0.0       # extra width per unit drive-proxy (relative current)
    power_fanout_min: int = 4          # source of >= this many non-shorted channels is a rail
    #   (topological power detection -- catches non-standard-named supplies)
    cap_value_threshold_f: float = float("inf")  # cap-class ('wide plate-strap') is OPT-IN.
    #   A 2-terminal SPICE cap to ground is AMBIGUOUS: an output load cap (signal-routed,
    #   e.g. the OTA's 3pF vout->gnd) is topologically identical to a decoupling cap (wide
    #   strap). The netlist alone cannot tell them apart, so the DEFAULT never promotes a net
    #   to cap-class; pass a finite threshold to opt in for designs with known wide caps.
    rail_min_mult: float = 1.0         # power rail floor as a MULTIPLE of PDK min_width.
    #   power width = max(power_min_um, rail_min_mult * min_width). Default 1.0 -> the absolute
    #   power_min_um floor dominates (byte-identical); raise it for a PDK-relative rail width.
    global_min_mult: float = 1.0       # global (high-fan-out) SIGNAL-net width as a MULTIPLE of
    #   PDK min_width. A signal net wired to >= global_fanout_min device terminals routes
    #   BETWEEN functional groups (a bias / tail / output trunk) and wants a wider trunk than a
    #   purely-local node. Default 1.0 -> global signals stay at min_width (byte-identical); a
    #   design opts into general global widening with a value > 1 (or declares @wire_width).
    global_fanout_min: int = 6         # a SIGNAL net on >= this many device terminals is 'global'
    #   (routes across the cell). Threshold on TOPOLOGY (terminal count), never a net name.


def _norm(net: str) -> str:
    return net.strip().rstrip("!").lower()


def _cap_nets(intent, *, value_threshold_f: float) -> frozenset:
    """Nets that want a wide plate-strap ('cap' class).

    A cap qualifies only if it is large (>= ``value_threshold_f``) AND has at least one
    terminal on a true supply -- i.e. a decoupling/bypass cap, whose non-supply plate is
    routed as a wide strap. A cap floating between two signal nodes (a Miller compensation
    cap) is NOT a wide strap: its nets stay signal-class. This avoids promoting a Miller
    Cc node (e.g. the OTA ``vout``) to cap-class.
    """
    out = set()
    for p in getattr(intent, "passives", ()) or ():
        if getattr(p, "kind", None) != "cap":
            continue
        if float(getattr(p, "value", 0.0)) < value_threshold_f:
            continue
        terms = [_norm(str(n)) for n in (getattr(p, "nets", {}) or {}).values()]
        if not any(t in _TRUE_SUPPLY for t in terms):
            continue  # floating signal cap (Miller comp) -> not a wide strap
        for t in terms:
            if t not in _TRUE_SUPPLY:
                out.add(t)  # the non-supply plate is the wide strap
    return frozenset(out)


def classify_net_role(
    net: str,
    intent,
    *,
    power_bases=DEFAULT_POWER_BASES,
    policy: Optional[WidthPolicy] = None,
) -> str:
    """Classify a net as ``'power'`` / ``'cap'`` / ``'signal'`` from netlist topology alone."""
    pol = policy or WidthPolicy()
    base = _norm(net)
    if base in _TRUE_SUPPLY:
        return "power"
    if _rail_fanout(base, intent) >= pol.power_fanout_min:
        return "power"
    if base in _cap_nets(intent, value_threshold_f=pol.cap_value_threshold_f):
        return "cap"
    return "signal"


def _rail_fanout(base: str, intent) -> int:
    """How many devices draw channel current from this source rail.

    A body-only substrate or well bias has no drain/source current proxy.  Counting
    those ties as rail fan-out promotes a low-current bias to the power-width class
    merely because many devices share a well domain.  A device whose drain and
    source are the same net also contributes no channel-current demand; its real
    terminal loads remain counted separately by :func:`net_fanout`.
    """
    n = 0
    for d in getattr(intent, "devices", ()) or ():
        nets = getattr(d, "nets", {}) or {}
        source = _norm(str(nets.get("source", "")))
        drain = _norm(str(nets.get("drain", "")))
        if source != drain and source == base:
            n += 1
    return n


def net_drive_proxy(net: str, intent) -> float:
    """Relative current proxy: Sigma (W*M) over devices with drain/source on ``net``.

    A leakage-free stand-in for net current: nets that sink/source more channel width carry
    more current and want a wider wire. Gate-only nets and devices with shorted
    drain/source terminals carry no channel-current proxy here. This does not
    remove their junction/gate loads or any physical terminal.
    """
    base = _norm(net)
    total = 0.0
    for d in getattr(intent, "devices", ()) or ():
        nets = getattr(d, "nets", {}) or {}
        source = _norm(str(nets.get("source", "")))
        drain = _norm(str(nets.get("drain", "")))
        if source != drain and base in (drain, source):
            total += float(getattr(d, "W", 0.0)) * float(getattr(d, "M", 1) or 1)
    return total


def net_fanout(net: str, intent) -> int:
    """How many DEVICE TERMINALS attach to ``net`` (MOS drain/gate/source/body + passive
    p/n). A leakage-free topological measure of how GLOBAL a net is: a node wired to many
    device terminals routes ACROSS the cell (a bias / tail / output trunk that spans
    functional groups) and wants a wider trunk than a purely-local two-terminal node. Counts
    terminals, not distinct devices, so a net on both the source AND body of one device
    counts twice (it carries that much more connection)."""
    base = _norm(net)
    n = 0
    for d in getattr(intent, "devices", ()) or ():
        nets = getattr(d, "nets", {}) or {}
        for role in ("drain", "gate", "source", "body"):
            if _norm(str(nets.get(role, ""))) == base:
                n += 1
    for p in getattr(intent, "passives", ()) or ():
        for v in (getattr(p, "nets", {}) or {}).values():
            if _norm(str(v)) == base:
                n += 1
    return n


def _all_nets(intent) -> list:
    seen = {}
    for d in getattr(intent, "devices", ()) or ():
        for n in (getattr(d, "nets", {}) or {}).values():
            seen.setdefault(_norm(str(n)), str(n))
    for p in getattr(intent, "passives", ()) or ():
        for n in (getattr(p, "nets", {}) or {}).values():
            seen.setdefault(_norm(str(n)), str(n))
    # stable order by original name
    return list(dict.fromkeys(seen.values()))


def _pdk_floor_um(tech, layer: str) -> float:
    mw = None
    try:
        mw = tech.min_width(layer)
    except Exception:
        mw = None
    if not mw:
        # conservative generic fallback if the PDK declares no metal min-width
        return 0.2
    return float(mw) / 1000.0


def derive_width_intent(
    intent,
    tech,
    *,
    layer: str = "metal1",
    policy: Optional[WidthPolicy] = None,
    power_bases=DEFAULT_POWER_BASES,
    declared: Optional[Dict[str, float]] = None,
) -> Dict[str, Declared]:
    """Per-net width, DESIGN-sourced (leakage-free), class-floored, PDK-floored.

    Returns ``{net_name: Declared(width_um, IntentSource.DESIGN)}`` for every net in the
    netlist. The GENERAL, design-derived width per net comes from its net ROLE / rail-class
    (+ a relative drive proxy), a general capability shared by any cell on the ``route_cell``
    path -- never a per-circuit switch:

      * power rails reach ``max(policy.power_min_um, policy.rail_min_mult * min_width)``;
      * big decoupling caps reach ``policy.cap_min_um`` (opt-in, see ``cap_value_threshold_f``);
      * a GLOBAL signal net (wired to >= ``policy.global_fanout_min`` device terminals -- a
        bias / tail / output trunk that routes across the cell) reaches
        ``policy.global_min_mult * min_width``;
      * every other (local) signal is floored at the PDK ``min_width``.

    All widths are finally re-floored on ``min_width`` (never sub-min). With the default
    policy ``rail_min_mult == global_min_mult == 1.0`` the rail-class multiples are INERT
    (global signals stay at ``min_width``, power stays at ``power_min_um``), so a cell that
    neither declares ``@wire_width`` nor supplies a wider policy is byte-identical.

    DECLARED OVERRIDE (``declared``): a ``{net -> width_um}`` map of designer-authored
    ``@wire_width`` widths. A net present here takes the declared width (re-floored on
    ``min_width``) INSTEAD of its role-derived value -- the ``.clay`` is the source of truth,
    so an explicit ``@wire_width(net, w)`` wins over the general default for that net. A
    declared width is DESIGNER INTENT (authored in the source, not measured from a reference
    layout), so it is tagged :class:`IntentSource.DESIGN` and does not count as leakage.
    """
    pol = policy or WidthPolicy()
    floor_um = _pdk_floor_um(tech, layer)
    signal_floor = max(pol.signal_floor_um, floor_um)
    # declared @wire_width (um), normalised by net name -> the per-net override.
    decl_um = {_norm(str(k)): float(v) for k, v in (declared or {}).items()}
    out: Dict[str, Declared] = {}
    for net in _all_nets(intent):
        base = _norm(net)
        if base in decl_um:
            # DECLARED override: honour the .clay's @wire_width, re-floored on min_width.
            w = max(decl_um[base], floor_um)
            out[net] = Declared(value=round(float(w), 4), source=IntentSource.DESIGN)
            continue
        role = classify_net_role(net, intent, power_bases=power_bases, policy=pol)
        proxy = net_drive_proxy(net, intent)
        extra = pol.em_um_per_proxy * proxy
        if role == "power":
            w = max(pol.power_min_um, pol.rail_min_mult * floor_um, signal_floor + extra)
        elif role == "cap":
            w = max(pol.cap_min_um, signal_floor + extra)
        else:
            w = max(signal_floor, signal_floor + extra)
            # GLOBAL signal rail-class: a high-fan-out node routes across the cell and wants
            # a wider trunk (a PDK-min-relative multiple, re-floored). Inert at mult 1.0.
            if net_fanout(net, intent) >= pol.global_fanout_min:
                w = max(w, pol.global_min_mult * floor_um)
        out[net] = Declared(value=round(float(w), 4), source=IntentSource.DESIGN)
    return out


def reference_width_intent(per_net: Dict[str, float]) -> Dict[str, Declared]:
    """The LEAKED baseline: a per-net width TABLE measured from the reference layout.

    Tagged :class:`IntentSource.REFERENCE` so the leakage gate counts it. This is the
    degradation START / the 'A' arm of the migration-falsification A/B.
    """
    return {net: Declared(value=float(w), source=IntentSource.REFERENCE)
            for net, w in per_net.items()}
