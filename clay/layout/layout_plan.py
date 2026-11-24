"""Frozen LayoutPlan — the reconciled, PDK-free intent-IR for one cell.

PHASE P2 of the Clay intent-driven P&R plan. A :class:`LayoutPlan` is the
single, frozen object a placer (P3) and router (P4) consume INSTEAD of
re-walking annotations / re-deriving symmetry / re-computing connectivity each
on their own. It reconciles three sources of layout intent into one
authoritative shape:

  * DECLARED device-level annotations (``annotate <cell> { ... }``) — the
    designer's source-of-truth (``@symmetric`` / ``@matching`` device pairs,
    ``@bulk`` body nets, ``@tap`` substrate ties);
  * DECLARED net-level layout intent (``layout_intent <cell> { ... }``) — diff
    pairs, shields and parasitic budgets, already lowered to
    :class:`clay.layout.intent.LayoutIntent`;
  * CIRCUIT-DERIVED symmetry (:func:`clay.layout.symmetry_derive.derive_symmetry_groups`)
    read straight from the topology — used ONLY to fill pairs the designer did
    not declare.

Reconciliation rule (mirrors ``clay.layout.intent.layout_intent_from_cell_def``
and the ``layout_intent_from_declarations`` precedent): the DECLARED intent
wins; derivation only fills gaps; a deliberately-conflicting declaration is
FLAGGED (via :func:`crosscheck_against_annotations`) and the declared pair is
still kept (declared-over-derived). ``layout_plan_from_cell_def`` is pure
Python orchestration with NO PDK knowledge — the plan is the same on every
process node; the PDK only enters later, in the placer/router that consumes it.

Connectivity authority: ``net_terminals`` is computed ONCE here, directly from
the parsed CellDef pins, using the SAME pin->terminal mapping that
:func:`clay.layout.generic_router.net_terminals_from_netlist` uses. The router
filters that graph against a concrete Placement (drops body ties with no placed
terminal); at PLAN time there is no placement yet, so the plan carries the full
schematic net->[(device,term)] graph and the router narrows it when it places.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .annotations import (
    Bulk,
    Matching,
    Symmetric,
    SymmetryAxis,
    Tap,
)
from .intent import LayoutIntent, ParasiticBudget, Shield
from .net_match_scope import placement_net_subset
from .symmetry_derive import (
    crosscheck_against_annotations,
    derive_symmetry_groups,
)

# Sentinel used by the parser for `@cell:`-targeted annotations. Kept in sync
# with `clay.parser_modules._grammar._CELL_ANNOTATION_TARGET` (a local copy so
# this module does not import the parser package — avoids a layout->parser dep).
_CELL_ANNOTATION_TARGET = "__cell__"

# Pin-key -> terminal-key map. IDENTICAL to
# `clay.layout.generic_router._PIN_TO_TERM` (the single connectivity authority):
# drain/gate/source/body and their long spellings all collapse to d/g/s/b so the
# plan's `net_terminals` graph matches what the router builds.
_PIN_TO_TERM: Dict[str, str] = {
    "d": "d", "g": "g", "s": "s", "b": "b",
    "drain": "d", "gate": "g", "source": "s", "bulk": "b", "body": "b",
}

# Class -> rail side fallback. Pure class-inference (no PDK / no `@rail`): an
# NMOS source rail conventionally sits at the BOTTOM, a PMOS source rail at the
# TOP. P7 replaces this with the declared `@rail`; until then this records the
# inferred convention so a placer has a starting orientation. Devices of other
# classes (cap/res/...) get no entry (they carry no rail-side convention).
_CLASS_RAIL_SIDE: Dict[str, str] = {"nmos": "bottom", "pmos": "top"}


def _norm_pair(a: str, b: str) -> Tuple[str, str]:
    """Order a 2-tuple so (a,b) and (b,a) collapse to one canonical pair."""
    return (a, b) if a <= b else (b, a)


@dataclass(frozen=True)
class LayoutPlan:
    """The reconciled, frozen, PDK-free layout intent for one cell.

    Every field is the single authority for its concern; a placer / router
    consumes the plan and never re-derives any of these from scratch.

    Fields
    ------
    cell:
        The cell's local name (provenance; matches ``CellDef.local_name``).
    device_pairs:
        Canonical (norm-ordered) matched/symmetric DEVICE instance-name pairs.
        DECLARED ``@symmetric`` / ``@matching`` pairs come first (declared-over-
        derived); circuit-derived pairs that the designer did NOT declare fill
        the gaps. Each pair is ``(a, b)`` with ``a <= b``.
    axis:
        The symmetry axis for the cell ("vertical" / "horizontal"). Taken from a
        declared ``@symmetric(... , <axis>)`` when present, else the derived
        group axis, else "vertical".
    on_axis:
        Device instance names that sit ON the symmetry axis (a tail switch /
        shared cascode whose drain or source lands on a group common net). From
        the circuit-derived groups' ``self_symmetric`` sets.
    cc_groups:
        Common-centroid device GROUPS (each a tuple of >=2 instance names).
        EMPTY for now — P7 populates this from ``@common_centroid``. The field
        exists so P3/P4 can consume a stable shape.
    rail_sides:
        Per-device inferred rail side ("bottom" for nmos, "top" for pmos). A
        class-inference FALLBACK; P7's ``@rail`` overrides it. Devices with no
        rail convention are absent from the map.
    bulk_nets:
        Per-device DECLARED body/bulk net (from ``@bulk(net)``). This is the
        body-net AUTHORITY: where a device declares ``@bulk``, that net — not the
        schematic ``b`` pin — is the body connection a placer/tap-inserter uses.
    taps:
        Declared substrate/well taps (``@tap(net, where=...)``) as
        :class:`clay.layout.annotations.Tap` objects, in source order.
    diff_net_pairs:
        Canonical (norm-ordered) DIFFERENTIAL NET pairs from the net-level
        ``layout_intent`` (``@diff_pair`` / ``@match`` / ``@symmetric`` nets),
        i.e. ``LayoutIntent.diff_pairs``. Distinct from ``device_pairs`` (which
        names DEVICES); these name NETS.
    shields:
        Declared shields (``clay.layout.intent.Shield``) from the net-level
        intent, in declaration order.
    budgets:
        Declared parasitic budgets (``clay.layout.intent.ParasiticBudget``) from
        the net-level intent, in declaration order.
    net_terminals:
        ``net -> [(device_label, term_key), ...]`` — the single connectivity
        authority, computed once from the CellDef pins. ``term_key`` is one of
        d/g/s/b. Per-net device order is first-seen (deterministic).
    symmetry_report:
        The honest cross-check of derived-vs-declared device pairs
        (``clay.layout.symmetry_derive.ConsistencyReport``), or ``None`` when no
        derivation was run (``derive=False``). ``unbacked`` flags declared pairs
        with no circuit backing; ``underdeclared`` flags circuit-symmetric pairs
        the designer never declared (these were folded into ``device_pairs``).
    """

    cell: str
    device_pairs: Tuple[Tuple[str, str], ...]
    axis: str
    on_axis: Tuple[str, ...]
    cc_groups: Tuple[Tuple[str, ...], ...]
    rail_sides: Mapping[str, str]
    bulk_nets: Mapping[str, str]
    taps: Tuple[Tap, ...]
    diff_net_pairs: Tuple[Tuple[str, str], ...]
    shields: Tuple[Shield, ...]
    budgets: Tuple[ParasiticBudget, ...]
    net_terminals: Mapping[str, List[Tuple[str, str]]]
    symmetry_report: Optional[object] = None
    placement_diff_net_pairs: Optional[Tuple[Tuple[str, str], ...]] = None

    def __post_init__(self):
        placement_net_subset(self.diff_net_pairs, self.placement_diff_net_pairs)


# ── connectivity authority (mirrors generic_router.net_terminals_from_netlist) ─


def net_terminals_from_cell_def(cd) -> Dict[str, List[Tuple[str, str]]]:
    """Build ``net -> [(device_label, term_key), ...]`` from a parsed CellDef.

    The plan-time counterpart of
    :func:`clay.layout.generic_router.net_terminals_from_netlist`: SAME
    pin->terminal mapping, but NO placement filter (there is no placement yet at
    plan time), so EVERY schematic terminal — including body ties — is recorded.
    The router narrows this to placed terminals when it places the cell.

    Determinism: nets are emitted in first-seen device order; within a net the
    terminals follow device order then pin order.
    """
    nets: Dict[str, List[Tuple[str, str]]] = {}
    for dev in getattr(cd, "devices", []) or []:
        label = getattr(dev, "instance_name", None)
        pins = getattr(dev, "pins", None) or {}
        if label is None:
            continue
        for pin_key, net in pins.items():
            tk = _PIN_TO_TERM.get(str(pin_key).lower())
            if tk is None or net is None:
                continue
            nets.setdefault(str(net), []).append((label, tk))
    return nets


# ── annotation readers ────────────────────────────────────────────────────────


def _layout_annotations(cd) -> Dict[str, List[object]]:
    """Read the cell's ``{target -> [LayoutAnnotation]}`` map (or empty)."""
    return getattr(cd, "__dict__", {}).get("_layout_annotations", {}) or {}


def _declared_device_pairs_and_axis(
    cd,
) -> Tuple[List[Tuple[str, str]], Optional[str]]:
    """Extract DECLARED matched/symmetric DEVICE pairs + an explicit axis.

    Sources, in the order they are read (all DECLARED, so all source-of-truth):

      * ``target: @matching(other)`` — pairs ``(target, other)``.
      * ``@cell: @symmetric(d1, d2, ...)`` or ``target: @symmetric(d1, ...)`` —
        every adjacent (d0,d1),(d2,d3),... consecutive device couple is a pair;
        a lone device on a non-cell target pairs WITH the target.

    The axis is taken from the FIRST ``@symmetric`` annotation that carries one
    (they all default ``vertical``); ``None`` means "no declared axis".
    """
    pairs: List[Tuple[str, str]] = []
    axis: Optional[str] = None
    ann_map = _layout_annotations(cd)
    for target, anns in ann_map.items():
        for ann in anns:
            if isinstance(ann, Matching):
                pairs.append(_norm_pair(target, ann.other))
            elif isinstance(ann, Symmetric):
                if axis is None:
                    axis = ann.axis.value if isinstance(
                        ann.axis, SymmetryAxis
                    ) else str(ann.axis)
                devs = list(ann.devices)
                if not devs:
                    # `@symmetric(vertical)` on @cell carries only the axis; the
                    # device pairing comes from derivation, not here.
                    continue
                if target != _CELL_ANNOTATION_TARGET and len(devs) == 1:
                    # A device-targeted single-device symmetric pairs the target
                    # with the one named device.
                    pairs.append(_norm_pair(target, devs[0]))
                    continue
                # Pair consecutive devices (d0,d1),(d2,d3),...; a trailing odd
                # device is left unpaired (the spatial validator flags odd
                # counts separately).
                for i in range(0, len(devs) - 1, 2):
                    pairs.append(_norm_pair(devs[i], devs[i + 1]))
    return pairs, axis


def _declared_bulk_nets(cd) -> Dict[str, str]:
    """Per-device DECLARED body net from ``@bulk(net)`` (last wins per device)."""
    out: Dict[str, str] = {}
    for target, anns in _layout_annotations(cd).items():
        if target == _CELL_ANNOTATION_TARGET:
            continue
        for ann in anns:
            if isinstance(ann, Bulk):
                out[target] = ann.net
    return out


def _declared_taps(cd) -> List[Tap]:
    """All declared ``@tap(...)`` annotations, in stable (target, source) order."""
    out: List[Tap] = []
    for target in sorted(_layout_annotations(cd).keys()):
        for ann in _layout_annotations(cd)[target]:
            if isinstance(ann, Tap):
                out.append(ann)
    return out


def _rail_sides_by_class(cd) -> Dict[str, str]:
    """Inferred per-device rail side from device class (nmos->bottom/pmos->top)."""
    out: Dict[str, str] = {}
    for dev in getattr(cd, "devices", []) or []:
        label = getattr(dev, "instance_name", None)
        dclass = getattr(dev, "device_class", None)
        if label is None:
            continue
        side = _CLASS_RAIL_SIDE.get(str(dclass).lower())
        if side is not None:
            out[label] = side
    return out


# ── public orchestration ──────────────────────────────────────────────────────


def layout_plan_from_cell_def(cd, *, derive: bool = True) -> LayoutPlan:
    """Reconcile a parsed CellDef's intent into a frozen :class:`LayoutPlan`.

    Pure Python, NO PDK knowledge. Reconciliation:

      1. DECLARED ``@symmetric`` / ``@matching`` device pairs are read first —
         they are source-of-truth.
      2. When ``derive`` is True, :func:`derive_symmetry_groups` reads the
         topology and any circuit-symmetric pair the designer did NOT declare is
         APPENDED (declared pairs keep their position; derivation only fills
         gaps). The honest cross-check
         (:func:`crosscheck_against_annotations`) is stored on the plan so a
         caller can see which declared pairs are unbacked and which derived pairs
         were undeclared.
      3. The net-level ``layout_intent`` is lowered to a
         :class:`LayoutIntent` (declared-wins, via the existing
         ``layout_intent_from_cell_def`` precedent) and its diff pairs / shields
         / budgets are carried onto the plan.
      4. ``net_terminals`` is computed once from the CellDef pins.
    """
    cell_name = getattr(cd, "local_name", "") or ""

    # 1. declared device pairs + axis (source-of-truth).
    declared_pairs, declared_axis = _declared_device_pairs_and_axis(cd)
    # Dedupe declared pairs while preserving first-seen order.
    device_pairs: List[Tuple[str, str]] = []
    seen: set = set()
    for p in declared_pairs:
        if p not in seen:
            seen.add(p)
            device_pairs.append(p)

    # 2. circuit-derived symmetry fills UNdeclared pairs.
    derived_axis: Optional[str] = None
    on_axis: List[str] = []
    symmetry_report = None
    if derive:
        groups = derive_symmetry_groups(cd)
        if groups:
            derived_axis = groups[0].axis
            for g in groups:
                on_axis.extend(g.self_symmetric)
        for g in groups:
            for (a, b) in g.pairs:
                np = _norm_pair(a, b)
                if np not in seen:
                    seen.add(np)
                    device_pairs.append(np)
        # Honest cross-check of derived vs declared (declared wins; this only
        # FLAGS the divergence — it never drops a declared pair).
        symmetry_report = crosscheck_against_annotations(
            groups, list(declared_pairs)
        )

    axis = declared_axis or derived_axis or "vertical"

    # 3. net-level intent (declared-wins via the existing precedent).
    intent = _layout_intent_for(cd, cell_name)
    diff_net_pairs = tuple(_norm_pair(a, b) for (a, b) in intent.diff_pairs)
    shields = tuple(intent.shields)
    budgets = tuple(intent.budgets)

    # bulk / tap / rail.
    bulk_nets = _declared_bulk_nets(cd)
    taps = tuple(_declared_taps(cd))
    rail_sides = _rail_sides_by_class(cd)

    # 4. connectivity authority.
    net_terminals = net_terminals_from_cell_def(cd)

    return LayoutPlan(
        cell=cell_name,
        device_pairs=tuple(device_pairs),
        axis=axis,
        on_axis=tuple(dict.fromkeys(on_axis)),  # dedupe, preserve order
        cc_groups=(),  # P7 fills @common_centroid
        rail_sides=dict(rail_sides),
        bulk_nets=dict(bulk_nets),
        taps=taps,
        diff_net_pairs=diff_net_pairs,
        placement_diff_net_pairs=tuple(_norm_pair(a, b) for a, b in intent.placement_diff_pairs),
        shields=shields,
        budgets=budgets,
        net_terminals=net_terminals,
        symmetry_report=symmetry_report,
    )


def _layout_intent_for(cd, cell_name: str) -> LayoutIntent:
    """Lower the cell's net-level intent (declared-wins), never raising.

    Reuses :func:`clay.layout.intent.layout_intent_from_cell_def` so the
    declared-over-derived precedent is honoured. Returns an EMPTY declared
    intent when the cell carries no net-level layout intent and no deriver
    category matches — the common case for plain primitive cells.
    """
    from .intent import (
        layout_intent_from_cell_def,
        layout_intent_from_declarations,
    )

    return layout_intent_from_cell_def(cd)
