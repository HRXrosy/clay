"""lower_intent: RoutingIntent -> [RouteConstraint].

The ONE place the open intent vocabulary maps onto the closed constraint set. Adding an
intent primitive adds a case HERE (and a field in :mod:`clay.layout.route.intent`), so the
engine -- which consumes only constraints -- does not change. Keep this a pure, total
function with no PDK access and no per-circuit logic.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from clay.layout.route import constraints as C
from clay.layout.route.intent import RoutingIntent, TrunkPreference

# Per-layer preferred-direction CONVENTION (process prior DATA, not a measured
# reference value): alternating axes by metal index -- ODD metals horizontal,
# EVEN metals vertical (M1=H, M2=V, M3=H, M4=V, ...). This HVH alternation is the
# standard preferred-direction discipline of multi-layer detailed routing: it is
# what LEF routing-layer ``DIRECTION HORIZONTAL|VERTICAL`` attributes encode and
# what the ISPD-2018/2019 initial-detailed-routing contest decks assign per layer
# (alternating per metal), and it matches the maze engine's even-graph-layer-
# horizontal soft prior. It is a DEFAULT (used when no PDK-derived table is
# supplied), overridable both by the PDK's own declared axes (see
# ``conventional_axes_for``) and by an explicit Direction constraint (the engine
# takes the LAST Direction per layer, so explicit wins). M1..M10 covers every
# supported PDK's routable stack; a table entry for a metal outside a PDK's stack
# is vacuously ignored by the engine.
CONVENTIONAL_AXES: Dict[int, str] = {
    n: ("H" if n % 2 == 1 else "V") for n in range(1, 11)
}


def conventional_axes_for(tech, *, n_max: int = 10) -> Dict[int, str]:
    """The per-metal preferred-direction table DERIVED from the PDK.

    Reads ``tech.metal_preferred_axis(n)`` (the adapter's ``preferred_routing_axes``
    process-convention table when declared, else the generic ODD=H/EVEN=V
    alternation) for metals 1..``n_max``. So a PDK that declares M2=H/M3=V yields
    that; an undeclared PDK yields exactly ``CONVENTIONAL_AXES`` (byte-identical).
    A ``tech`` without the accessor falls back to ``CONVENTIONAL_AXES``."""
    fn = getattr(tech, "metal_preferred_axis", None)
    if fn is None:
        return dict(CONVENTIONAL_AXES)
    out: Dict[int, str] = {}
    for n in range(1, n_max + 1):
        try:
            a = fn(n)
        except Exception:
            a = CONVENTIONAL_AXES.get(n, "H" if n % 2 == 1 else "V")
        out[n] = a if a in ("H", "V") else ("H" if n % 2 == 1 else "V")
    return out


def lower_intent(
    intent: RoutingIntent,
    *,
    direction_weight: float = 0.0,
    axes: Optional[Dict[int, str]] = None,
) -> List[C.RouteConstraint]:
    """Lower a declarative :class:`RoutingIntent` to the closed constraint set.

    Provenance (IntentSource) is intentionally NOT carried into constraints -- the
    constraint is the same regardless of how the value was sourced; leakage is gated on
    the intent (``RoutingIntent.leakage_count``) BEFORE lowering, not here.

    ``direction_weight`` (keyword-only, default 0.0): when > 0, ALSO emit the
    per-layer :class:`~clay.layout.route.constraints.Direction` soft preferences
    from the ``axes`` HVH table (``CONVENTIONAL_AXES`` when ``axes`` is None), each
    at this weight. The default 0 emits NONE -- the lowered constraint list (and
    therefore every existing route) is byte-unchanged. The weight is a design-side
    knob (how much direction style is worth in step-cost units), never a
    reference-measured value.

    ``axes`` (keyword-only, default None -> ``CONVENTIONAL_AXES``): the per-metal
    {index -> 'H'|'V'} table to emit Direction from. The caller passes the
    PDK-derived table (``conventional_axes_for(tech)``) so the HVH discipline is
    process-correct for the target stack; None keeps the generic default. Data, not
    the PDK, so this function stays PDK-access-free.
    """
    out: List[C.RouteConstraint] = []

    for net, d in intent.net_layers.items():
        out.append(C.LayerSet(net=net, layers=frozenset(d.value)))

    for net, d in intent.trunk_layers.items():
        value = d.value
        if isinstance(value, TrunkPreference):
            layer, orient = value.layer, value.orient
        else:
            layer, orient = value, ""
        out.append(
            C.TrunkLayer(
                net=net,
                layer=layer,
                orient=str(orient).upper(),
            )
        )

    for net, d in intent.widths.items():
        out.append(C.Width(net=net, width_um=float(d.value)))

    for d in intent.mirror_pairs:
        a, b, axis = d.value
        out.append(C.Mirror(net_a=a, net_b=b, axis_x_um=float(axis)))

    for d in intent.shields:
        net, by, mn = d.value
        out.append(C.Shield(net=net, by=by, min_um=float(mn)))
        out.append(C.Spacing(net_a=net, net_b=by, min_um=float(mn)))

    for d in intent.spacing_pairs:
        a, b, minimum = d.value
        out.append(C.Spacing(net_a=a, net_b=b, min_um=minimum))

    for d in intent.length_pairs:
        a, b, tol = d.value
        out.append(C.Length(net_a=a, net_b=b, tol_rel=float(tol)))

    for d in intent.track_orders:
        ch, nets = d.value
        out.append(C.TrackOrder(channel=ch, nets=tuple(nets)))

    for d in intent.accesses:
        net, dev, term, side = d.value
        out.append(C.Access(net=net, device=dev, terminal=term, side=side))

    for net, d in intent.vias.items():
        out.append(C.Via(net=net, policy=str(d.value)))

    if direction_weight > 0:
        table = axes if axes is not None else CONVENTIONAL_AXES
        for layer_n in sorted(table):
            out.append(
                C.Direction(
                    layer_n=layer_n,
                    axis=table[layer_n],
                    weight=float(direction_weight),
                )
            )

    return out
