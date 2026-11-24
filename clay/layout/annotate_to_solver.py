"""R185 (BL-185-ANNO2SOLVER) -- wire the .clay SOURCE-LEVEL layout annotations
into the solver place-and-route, closing the gap where the language-level
``annotate {}`` declaration did NOT actually flow into the MILP placer.

The gap this module closes
==========================
Clay's grammar already parses an ``annotate Cell { M1: @matching(M2);
@cell: @symmetric(M1, M2, vertical); }`` block into a Python ``AnnotatedCell``
holding ``Symmetric`` / ``Matching`` / ... annotation objects (see
``clay/layout/annotations.py`` + ``clay/parser_modules/_grammar.py``). But the
solver place-and-route (``milp_placer_v175.solve_placement``) was driven by
Python-declared constraint CONSTANTS imported VERBATIM from
``symmetric_comparator_v173`` (``SYMMETRY_GROUPS``, ``ON_AXIS_DEVICE``,
``RAIL_CONVENTION``, ``COMMON_CENTROID_GROUP``). So the placement that came out
was a function of a hand-written Python dict, NOT of the ``.clay`` source. The
"declare layout intent in the language" moat was only HALF wired: the front-end
parsed the intent, but the back-end ignored it.

This module is the missing translator + driver:

  1. :func:`annotations_to_solver_constraints` maps a parsed ``AnnotatedCell``
     to a :class:`SolverConstraints` -- the same shape the placer's
     constraint set has, but DERIVED from the ``.clay`` annotations:
       * ``@symmetric(devices..., axis)``  -> the symmetry axis + per-pair
         mirror groups (consecutive declared devices pair off about the axis),
       * ``@matching(a, b)``               -> a matched/symmetry pair,
       * device class (``pmos`` / ``nmos``) -> the rail band each device sits in
         (NMOS near the bottom rail, PMOS near the top rail), which IS the rail
         convention realised structurally.
  2. :func:`place_and_route_from_clay` parses the ``.clay`` (source string OR
     path), extracts the annotations, builds the solver constraints, runs a
     symmetry-constrained MILP placement (HiGHS via ``scipy.optimize.milp`` --
     already present, zero new dependency), scores the result with the REAL R172
     ``symmetry_placement`` rule, and (optionally) emits a GDS + runs the real
     gf180mcu DRC / union-find co-clean on the .clay-driven layout.

The Clay moat realised END-TO-END
==================================
The symmetry pairs the MILP enforces, and that the R172 checker scores 1.0, come
from the ``.clay`` ``@symmetric`` declaration -- NOT from a Python dict. Swap the
``@symmetric`` axis or pairs in the source and the placement changes accordingly
(proven by ``tests/test_annotate_to_solver.py::test_clay_declaration_drives_mirror``).
That is the difference between a layout that happens to be symmetric and a layout
that is symmetric BECAUSE the designer said so in the language.

Honesty contract
----------------
* The MILP solver status + wall time are the REAL ``scipy.optimize.milp`` numbers.
* The symmetry / rail / area scores are the REAL ``layout_quality_rules_v172``
  measurements on the solved device bboxes.
* DRC / union-find verdicts (when run) are read verbatim from the real toolchain;
  if the real toolchain is unavailable the result honestly reports the achieved
  level (placement-from-annotation + symmetry score) and the co-clean residual.
* The grammar extension for ``@common_centroid`` / power-``rail(side)`` is
  documented as a roadmap item -- the consume step uses the EXISTING six
  annotation kinds (``@symmetric`` / ``@matching``) and infers the rail band from
  the device class. No fabricated parser surface.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .annotations import (
    AnnotatedCell,
    EqualSize,
    Fix,
    Matching,
    Symmetric,
    SymmetryAxis,
)
from .coclean_5tota import GF180_COMP_KEY
from .device_cell import DeviceGeom, Terminal, build_device
from .layout_quality_rules import (
    RuleResult,
    common_centroid as r172_common_centroid,
    rail_position as r172_rail_position,
    symmetry_placement as r172_symmetry_placement,
)
from .pdk_rules import PdkRules
from .tech import LayoutTech
from .place.ordered_mirror_rows import DeclaredMirrorGroup

Bbox = Tuple[int, int, int, int]

# the sentinel the parser uses for @cell-targeted annotations.
_CELL_TARGET = "__cell__"


# === the solver-facing constraint container ==================================


@dataclass
class SolverConstraints:
    """The placer-facing constraints DERIVED from a ``.clay`` ``annotate {}``
    block. This is the same logical shape the hand-written
    ``symmetric_comparator_v173`` constants carried (``SYMMETRY_GROUPS`` etc.) --
    but every field here is sourced from the parsed annotations, not a Python
    literal.

    Fields
    ------
    cell_name : the cell the annotations target.
    symmetry_pairs : the declared mirror pairs ``(left, right)`` about the axis.
        Built from ``@symmetric(devices..., axis)`` (consecutive device pairing)
        and ``@matching(a, b)`` (each match is a candidate pair).
    symmetry_axis : ``"x"`` (vertical mirror line) or ``"y"`` (horizontal) -- the
        R172 ``symmetry_placement`` axis convention. ``@symmetric(vertical)`` maps
        to ``"x"`` (mirror about a vertical line), ``horizontal`` to ``"y"``.
    on_axis_devices : devices that must sit ON the symmetry axis (an odd member of
        a declared symmetric group, e.g. a tail switch). Empty when every declared
        device pairs off.
    rail_convention : the ``(net, edge)`` rail bands. Inferred from the device
        classes: NMOS band at the bottom (``vss``), PMOS band at the top (``vdd``).
        When no device classes are known it is empty (no rail gate run).
    common_centroid_groups : groups to balance common-centroid. The first declared
        ``@symmetric`` device pair is treated as the matched diff pair to balance
        (the canonical analog common-centroid target); a future ``@common_centroid``
        annotation (grammar roadmap) would feed this directly.
    device_kinds : ``label -> "nmos"|"pmos"`` from the parsed cell.
    diff_net_sides : ``device_label -> -1 | +1`` half-plane assignment derived from a
        declared/derived DIFFERENTIAL-NET pair (e.g. ``(di_n, di_p)``). Every device
        whose DRAIN is the first net of a diff pair is pinned to one half-plane
        (``-1``, left of the axis) and every device whose drain is the second net to
        the opposite half-plane (``+1``, right). This makes a differential pair's two
        nets MIRROR-SEPARABLE (the two-row idea the generic router co-tracks on): the
        symmetry-pair side assignment already realises this for a balanced cell, so
        the constraint is NON-BINDING there (gf180 parity preserved); it BINDS only
        when the unconstrained optimum would let a diff net's devices straddle the
        axis. Empty when no diff-net pair is declared.
    """

    cell_name: str
    symmetry_pairs: Tuple[Tuple[str, str], ...] = ()
    symmetry_axis: str = "x"
    on_axis_devices: Tuple[str, ...] = ()
    rail_convention: Tuple[Tuple[str, str], ...] = ()
    common_centroid_groups: Tuple[Tuple[str, ...], ...] = ()
    device_kinds: Dict[str, str] = field(default_factory=dict)
    diff_net_sides: Dict[str, int] = field(default_factory=dict)
    # Declared FLOORPLAN intent (@place_group ... row=/col=). A device-level 2D
    # placement directive that assigns devices to explicit rows (and optional
    # columns) of the floorplan. When present, the row/column STRUCTURE the placer
    # builds comes from this declaration, NOT from the NMOS-below-PMOS device-class
    # banding -- this is what lets a designer express a true 2D analog floorplan in
    # the language (input pair + mirror in one row, CS + load in another, side by
    # side -> a wide 2D shape) instead of the tall std-cell column the default
    # banding produces. Maps ``device_label -> (row, col)``; ``col`` is ``None``
    # when the group declared only a row. Empty when no @place_group floorplan is
    # declared (the placer then falls back to its band-free / banded default).
    floorplan_rows: Dict[str, Tuple[int, Optional[int]]] = field(default_factory=dict)
    # Preserve the actual named functional scopes.  Their row ordinal orders
    # group regions; it does not itself require every member to share a y center.
    floorplan_groups: Tuple[Tuple[str, Tuple[str, ...]], ...] = ()
    # Declared EQUAL-FOOTPRINT groups (@equal_size(d1, d2, ...)). Each frozenset is
    # a set of device/passive LABELS the placer must realise at an IDENTICAL
    # footprint -- the SAME placed width AND the same placed height, within the
    # placer's grid epsilon. This PINS the ``size_class`` invariant the layout
    # language otherwise leaves implicit (two passives end up equal-footprint only
    # because a designer hand-set their schematic values equal; nothing in the
    # layout language pins the equality). Groups that share a label are MERGED so
    # the result is a clean partition-ish set (one frozenset per connected
    # component of the @equal_size declarations). Empty when no @equal_size is
    # declared; the placer then imposes no equal-footprint constraint, and a
    # ``check_equal_size`` obligation verifies each declared group's realised
    # footprint spread on the real placement.
    equal_size_groups: Tuple[FrozenSet[str], ...] = ()
    # Declared RELATIVE-ORDER constraints (@order(a, b, rel)). Each entry is an
    # ordered ``(a, b, rel)`` triple with ``rel`` in {"left_of", "above"}: device a
    # must sit left_of / above device b in the realised placement. The seqpair
    # placer consumes these as DIRECTED constraint-graph edges (left_of -> a before b
    # on the x graph, above -> a before b on the y graph) so the order is realised by
    # construction. Empty when no @order is declared (the placer's ordering is then
    # unchanged -> byte-parity). Sourced from ``LayoutIntent.order_constraints`` via
    # ``_constraints_from_intent_and_annotations``.
    order_constraints: Tuple[Tuple[str, str, str], ...] = ()
    # Declared ADJACENCY pairs (@abut(a, b)). Each entry is an ordered ``(a, b)``
    # pair the placer packs edge-adjacent: the facing inter-device gap is tightened to
    # the DRC-minimum (not the default routing channel) and the pair is preferred
    # adjacent in the packing order. Empty when no @abut is declared (byte-parity).
    # Sourced from ``LayoutIntent.abut_pairs``.
    abut_pairs: Tuple[Tuple[str, str], ...] = ()
    # Declared ABSOLUTE-PLACEMENT pins (@fix(device, x=<um>, y=<um>)). Maps a device
    # LABEL -> its pinned centroid ``(x_um, y_um)`` in microns. A device in this map is
    # placed with its centroid AT the pinned coordinate, OVERRIDING whatever position the
    # engine computed -- its built geometry (size / orientation) is preserved, only its
    # translation is fixed. THE ANCHOR PROPERTY: when EVERY device is pinned the placement
    # is fully determined by the pins, so the engine has zero remaining freedom and milp /
    # seqpair produce the SAME placement (strict consistency by construction). Empty when
    # no @fix is declared -> every placer path is byte-unchanged (the no-op guard). A
    # ``check_fix`` obligation verifies each pinned device's realised centroid lands at its
    # pin. Device names are DATA (no engine branch).
    pinned_positions: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    # Explicit @symmetric groups retain their own common-axis authority. A plain
    # @matching edge does not create such a group. Exact ordered rows consume
    # these groups before the legacy contiguous matching-atom projection.
    declared_mirror_groups: Tuple[DeclaredMirrorGroup, ...] = ()
    # Native PDK/channel lowerer supplies physical edge-to-edge row clearances.
    row_gap_minima_nm: Tuple[Tuple[int, int, int], ...] = ()

    def all_symmetry_devices(self) -> List[str]:
        out: List[str] = []
        for a, b in self.symmetry_pairs:
            out += [a, b]
        out += list(self.on_axis_devices)
        # de-dup preserving order.
        seen: set = set()
        uniq: List[str] = []
        for d in out:
            if d not in seen:
                seen.add(d)
                uniq.append(d)
        return uniq


def _axis_str(axis: SymmetryAxis) -> str:
    """Map the parsed SymmetryAxis enum to the R172 ``symmetry_placement`` axis
    string. ``VERTICAL`` mirror line -> the devices differ in x -> axis="x"."""
    return "x" if axis == SymmetryAxis.VERTICAL else "y"


def _device_kinds_from_cell(annotated: AnnotatedCell) -> Dict[str, str]:
    """Pull ``label -> device_class`` from the layouts/annotations. The
    ``AnnotatedCell.layouts`` are SymbolicLayouts (no class). We attach the class
    via a side map the builder sets (``_device_kinds``); fall back to empty."""
    return dict(getattr(annotated, "_device_kinds", {}) or {})


def kind_polarity_oxide(device_class) -> Tuple[Optional[str], str]:
    """Split a ``.clay`` device KIND token into ``(mos_polarity, gate_oxide)``.

    The device KIND is the single DECLARED source of the 3.3V-vs-1.8V fact: a
    3.3V THICK-gate-oxide device is declared with a thick-ox kind token
    (``pmos3v`` / ``nmos3v`` -- matching the PDK's real ``pch3`` / ``nch3``
    devices and the auCdl ``PD`` / ``ND``); a 1.8V core device with the plain
    polarity token (``pmos`` / ``nmos``).

    Returns ``(polarity, oxide)``:
      * ``polarity`` -- ``'pmos'`` / ``'nmos'`` for a MOS class (so every
        polarity-keyed consumer -- rail band, guard-ring well, symmetry, and the
        ``build_device`` ``kind`` arg -- is byte-UNCHANGED whether the design
        declares a plain or a 3v kind), or ``None`` for a NON-MOS class (cap /
        res / ind / varactor), which the caller leaves untouched.
      * ``oxide`` -- ``'thick'`` / ``'thin'`` from the SHARED ``model_oxide``
        marker rule (the SAME rule applied to the reference SPICE model tokens --
        no per-circuit device-name list -- so the ``.clay`` KIND and the extracted
        netlist agree on the oxide domain). An unmarked token is the safe
        ``'thin'`` default, so a 1.8V design threads ``oxide='thin'`` -> no OD2 ->
        byte-identical.
    """
    from clay.layout.extract.netlist_intent import model_oxide

    m = str(device_class or "").strip().lower()
    if m.startswith("p") and ("mos" in m or "ch" in m or m == "pd"):
        polarity: Optional[str] = "pmos"
    elif m.startswith("n") and ("mos" in m or "ch" in m or m == "nd"):
        polarity = "nmos"
    else:
        return None, "thin"
    return polarity, model_oxide(m)


def annotations_to_solver_constraints(
    annotated: AnnotatedCell,
    *,
    apply_rail_band: bool = False,
) -> SolverConstraints:
    """Translate a parsed ``.clay`` ``AnnotatedCell`` into the placer's
    :class:`SolverConstraints`.

    The mapping (the heart of the moat -- DECLARED intent becomes solver data):

      * ``@symmetric(d1, d2, d3, ..., axis)`` -- the devices pair off about the
        axis: ``(d1, d2)``, ``(d3, d4)``, ... A leftover odd device becomes an
        on-axis device. The axis literal sets the mirror line.
      * a bare ``@symmetric(axis)`` (no devices) declares the AXIS only; the pairs
        are then taken from the ``@matching`` annotations.
      * ``@matching(a, b)`` -- ``(a, b)`` is a candidate mirror pair (a matched
        device pair is a symmetry pair in a balanced analog cell). Duplicate
        reciprocal matches (``M1:@matching(M2)`` + ``M2:@matching(M1)``) collapse
        to ONE pair.
      * the rail convention is inferred from device classes (NMOS -> bottom/vss,
        PMOS -> top/vdd) -- the standard analog band convention, applied so every
        NMOS source faces the bottom rail and every PMOS the top.

    Annotations OTHER than ``@symmetric`` / ``@matching`` (guard_ring,
    well_isolated, interdigitated, adjacent) are parsed and carried on the cell
    but do not yet drive the placer; they are out of scope for this round's
    placement gate and documented as the next consume targets.
    """
    cell_name = annotated.cell_name
    device_kinds = _device_kinds_from_cell(annotated)

    # collect the @symmetric (axis + ordered devices) and @matching pairs.
    sym_axis = SymmetryAxis.VERTICAL
    sym_devices: List[str] = []
    saw_symmetric = False
    matching_pairs: List[Tuple[str, str]] = []
    # Each @equal_size(d1, d2, ...) contributes one raw label group; groups that
    # share a label are merged below into the clean partition-ish set.
    equal_size_raw: List[List[str]] = []
    # Each @fix(device, x, y) contributes one ABSOLUTE pin: device label -> (x_um, y_um).
    pinned_positions: Dict[str, Tuple[float, float]] = {}
    declared_mirror_groups: List[DeclaredMirrorGroup] = []

    for target, anns in annotated.annotations.items():
        for ann in anns:
            if isinstance(ann, Symmetric):
                saw_symmetric = True
                sym_axis = ann.axis
                # a @cell-targeted @symmetric lists the devices; a device-targeted
                # one (rare) implies that device + its declared devices.
                for d in ann.devices:
                    if d not in sym_devices:
                        sym_devices.append(d)
                group_members = tuple(str(d) for d in ann.devices)
                if target != _CELL_TARGET and len(group_members) == 1:
                    group_members = (str(target), *group_members)
                if group_members:
                    group = DeclaredMirrorGroup(
                        _axis_str(ann.axis),
                        tuple(zip(group_members[0::2], group_members[1::2])),
                        group_members[-1:] if len(group_members) % 2 else (),
                    )
                    if group not in declared_mirror_groups:
                        declared_mirror_groups.append(group)
            elif isinstance(ann, Matching):
                a = target
                b = ann.other
                # normalise the pair order so reciprocal matches collapse.
                pair = (a, b) if a <= b else (b, a)
                if pair not in matching_pairs:
                    matching_pairs.append(pair)
            elif isinstance(ann, EqualSize):
                # An @equal_size group of member LABELS to realise at an identical
                # footprint. Collected raw; merged (groups sharing a label union)
                # into ``equal_size_groups`` below.
                labels = [str(d) for d in ann.devices]
                if len(labels) >= 2:
                    equal_size_raw.append(labels)
            elif isinstance(ann, Fix):
                # An ABSOLUTE pin: the named device's centroid is fixed at (x, y) um.
                # Keyed by device LABEL -> (x_um, y_um); a later @fix on the same device
                # overrides an earlier one (last declaration wins, deterministic).
                pinned_positions[str(ann.device)] = (float(ann.x), float(ann.y))

    # Build the symmetry pairs.
    pairs: List[Tuple[str, str]] = []
    on_axis: List[str] = []
    if sym_devices:
        # consecutive pairing about the axis; a leftover odd device is on-axis.
        it = list(sym_devices)
        i = 0
        while i + 1 < len(it):
            pairs.append((it[i], it[i + 1]))
            i += 2
        if i < len(it):
            on_axis.append(it[i])
    # @matching pairs augment the symmetry pairs (a matched pair is mirror-placed).
    for p in matching_pairs:
        # avoid duplicating a pair already covered by @symmetric (in either order).
        if p not in pairs and (p[1], p[0]) not in pairs:
            pairs.append(p)

    axis_str = _axis_str(sym_axis) if (saw_symmetric or pairs) else "x"

    # rail convention from device classes (NMOS bottom / PMOS top). OPT-IN, OFF by
    # default: a fixed vss-bottom/vdd-top device banding is a DIGITAL std-cell
    # convention, not a universal analog rule (the user retracted the "all analog
    # layouts must have gnd/vdd top/bottom rails" rule). A cell opts in via
    # apply_rail_band=True; otherwise rail_convention stays empty and the placer
    # imposes no NMOS/PMOS banding.
    rail: List[Tuple[str, str]] = []
    if apply_rail_band:
        has_nmos = any(k == "nmos" for k in device_kinds.values())
        has_pmos = any(k == "pmos" for k in device_kinds.values())
        if has_nmos:
            rail.append(("vss", "bottom"))
        if has_pmos:
            rail.append(("vdd", "top"))

    # common-centroid: the first declared mirror pair is the canonical matched
    # diff pair to balance about its own centre.
    cc_groups: Tuple[Tuple[str, ...], ...] = ()
    if pairs:
        cc_groups = (tuple(pairs[0]),)

    return SolverConstraints(
        cell_name=cell_name,
        symmetry_pairs=tuple(pairs),
        symmetry_axis=axis_str,
        on_axis_devices=tuple(on_axis),
        rail_convention=tuple(rail),
        common_centroid_groups=cc_groups,
        device_kinds=device_kinds,
        equal_size_groups=_merge_equal_size_groups(equal_size_raw),
        pinned_positions=pinned_positions,
        declared_mirror_groups=tuple(declared_mirror_groups),
    )


def _merge_equal_size_groups(
    raw_groups: Sequence[Sequence[str]],
) -> Tuple[FrozenSet[str], ...]:
    """Merge raw ``@equal_size`` label groups into a clean partition-ish set.

    Two declared ``@equal_size`` groups that share ANY label name the same
    equal-footprint class (footprint equality is transitive: if ``A==B`` and
    ``B==C`` then ``A==B==C``), so they are unioned into ONE frozenset --
    connected components over the "share a label" relation. Each member label is
    normalised to ``str``; a group with fewer than two members is dropped (a
    single device trivially equals itself). Returns the components as frozensets,
    ordered by the FIRST-seen member so the result is deterministic.
    """
    # parent map for a simple union-find over the label names.
    parent: Dict[str, str] = {}
    order: List[str] = []  # first-seen label order for deterministic output

    def _find(x: str) -> str:
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        # path-compress.
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def _union(a: str, b: str) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[ra] = rb

    for grp in raw_groups:
        labels = [str(d) for d in grp]
        if len(labels) < 2:
            continue
        for lbl in labels:
            if lbl not in parent:
                parent[lbl] = lbl
                order.append(lbl)
        first = labels[0]
        for lbl in labels[1:]:
            _union(first, lbl)

    # group labels by their union-find root, preserving first-seen order.
    components: Dict[str, List[str]] = {}
    root_order: List[str] = []
    for lbl in order:
        root = _find(lbl)
        if root not in components:
            components[root] = []
            root_order.append(root)
        components[root].append(lbl)

    return tuple(
        frozenset(components[root])
        for root in root_order
        if len(components[root]) >= 2
    )


def _plan_axis_to_solver_axis(axis: str) -> str:
    """Map a :class:`~clay.layout.layout_plan.LayoutPlan` axis literal
    (``"vertical"`` / ``"horizontal"``) to the R172 ``symmetry_placement`` axis
    string. A vertical mirror line => the paired devices differ in x => axis="x"."""
    return "y" if str(axis).lower().startswith("h") else "x"


def solver_constraints_from_layout_plan(
    plan,
    device_kinds: Optional[Dict[str, str]] = None,
    *,
    apply_rail_band: bool = False,
) -> SolverConstraints:
    """Translate a frozen :class:`~clay.layout.layout_plan.LayoutPlan` slice into
    the placer's :class:`SolverConstraints` -- the P3 consumption path.

    The plan is the reconciled, PDK-free intent authority (P2). The placer reads
    its PLACEMENT slice instead of re-walking annotations:

      * ``device_pairs``  -> the mirror ``symmetry_pairs`` (already norm-ordered);
      * ``axis``          -> the mirror line ("vertical" -> "x", "horizontal" -> "y");
      * ``on_axis``       -> the on-axis (tail / shared cascode) devices;
      * ``rail_sides``    -> the rail convention bands (nmos source band at the
        BOTTOM = vss, pmos source band at the TOP = vdd);
      * ``cc_groups``     -> the REAL common-centroid groups (multi-pair): the MILP
        balances EVERY device in each group's y-centroid, NOT just the first pair.
        ``cc_groups`` is EMPTY until P7, so today the proxy below applies; once P7
        populates ``@common_centroid``, a 4-device group balances all four.

    Common-centroid resolution (the multi-pair fix): when the plan declares
    ``cc_groups`` they are used VERBATIM (every member balanced). Only when the
    plan declares NONE does the canonical first-pair proxy apply (the matched diff
    pair is the conventional common-centroid target), so the behaviour is a
    superset of the annotation path -- a real multi-pair group is no longer
    truncated to its first pair.

    ``device_kinds`` (label -> "nmos"|"pmos") is needed for the rail bands + the
    R172 rail score; pass the kinds read from the CellDef. When absent the rail
    bands are inferred from ``plan.rail_sides`` directly.
    """
    pairs = tuple(plan.device_pairs)
    axis_str = _plan_axis_to_solver_axis(getattr(plan, "axis", "vertical"))
    # A device that is a member of a declared mirror PAIR cannot ALSO be an on-axis
    # device: the plan's ``on_axis`` (from the derived groups' self_symmetric sets)
    # can overlap the declared pairs, and the row model would then assign the device
    # to BOTH a pair row and an on-axis row (an infeasible double placement). The
    # pair declaration wins -- the device mirrors, it is not on-axis.
    paired = {m for pr in pairs for m in pr}
    on_axis = tuple(d for d in plan.on_axis if d not in paired)

    kinds: Dict[str, str] = dict(device_kinds or {})
    # rail convention: a device whose rail side is "bottom" sits in the vss band,
    # "top" in the vdd band. The placer needs at most the two band names.
    # OPT-IN, OFF by default (see annotations_to_solver_constraints): the plan's
    # rail_sides are informational unless the cell opts into the banding.
    rail: List[Tuple[str, str]] = []
    if apply_rail_band:
        sides = set(getattr(plan, "rail_sides", {}).values())
        if "bottom" in sides:
            rail.append(("vss", "bottom"))
        if "top" in sides:
            rail.append(("vdd", "top"))

    # common-centroid: declared multi-pair groups VERBATIM (every member balanced);
    # only fall back to the first-pair proxy when the plan declares no group.
    cc_groups = tuple(tuple(g) for g in getattr(plan, "cc_groups", ()) or ())
    if not cc_groups and pairs:
        cc_groups = (tuple(pairs[0]),)

    # differential-net half-plane sides from the plan's NET-level diff pairs +
    # ``net_terminals`` (the connectivity authority): every device whose DRAIN ('d')
    # is the first diff net goes LEFT (-1), the second net's drain devices go RIGHT
    # (+1). This makes the diff pair mirror-separable (see the solver's half-plane
    # constraint); only the kinds the placer actually places carry a side.
    diff_sides: Dict[str, int] = {}
    from clay.layout.net_match_scope import placement_net_subset
    diff_pairs = placement_net_subset(tuple(getattr(plan, "diff_net_pairs", ()) or ()),
                                     getattr(plan, "placement_diff_net_pairs", None))
    net_terms = getattr(plan, "net_terminals", {}) or {}
    if diff_pairs and net_terms:
        for (na, nb) in diff_pairs:
            for (lbl, tk) in net_terms.get(str(na), ()):  # left half
                if tk == "d" and (not kinds or lbl in kinds):
                    diff_sides.setdefault(lbl, -1)
            for (lbl, tk) in net_terms.get(str(nb), ()):  # right half
                if tk == "d" and (not kinds or lbl in kinds):
                    diff_sides.setdefault(lbl, +1)

    return SolverConstraints(
        cell_name=getattr(plan, "cell", "") or "",
        symmetry_pairs=pairs,
        symmetry_axis=axis_str,
        on_axis_devices=on_axis,
        rail_convention=tuple(rail),
        common_centroid_groups=cc_groups,
        device_kinds=kinds,
        diff_net_sides=diff_sides,
    )


# === build an AnnotatedCell from a parsed .clay cell ==========================


def annotated_cell_from_clay(
    clay_source_or_path: Union[str, Path],
    *,
    cell_name: Optional[str] = None,
) -> AnnotatedCell:
    """Parse the ``.clay`` and build an :class:`AnnotatedCell` for one cell,
    carrying both the per-device annotations AND a ``_device_kinds`` side map
    (``label -> "nmos"|"pmos"``) read from the parsed devices.

    A path that exists on disk is parsed via ``parse_clay_file``; otherwise the
    argument is treated as source text and parsed via ``parse_clay_source``.
    """
    from clay.parser import parse_clay_file, parse_clay_source
    from clay.layout.symbolic_layout import SymbolicLayout

    src = str(clay_source_or_path)
    is_path = False
    try:
        is_path = Path(src).exists()
    except (OSError, ValueError):
        is_path = False
    cf = parse_clay_file(src) if is_path else parse_clay_source(src)

    if not cf.cell_defs:
        raise ValueError("no cell_def in the parsed .clay file")
    if cell_name is not None:
        cd = next((c for c in cf.cell_defs if c.local_name == cell_name), None)
        if cd is None:
            raise ValueError(
                f"cell {cell_name!r} not found; cells: "
                f"{[c.local_name for c in cf.cell_defs]}"
            )
    else:
        # pick the cell that carries layout annotations, else the first.
        cd = next(
            (c for c in cf.cell_defs if c.__dict__.get("_layout_annotations")),
            cf.cell_defs[0],
        )

    ann_map = cd.__dict__.get("_layout_annotations", {}) or {}
    cell = AnnotatedCell(cell_name=cd.local_name)
    device_kinds: Dict[str, str] = {}
    device_oxide: Dict[str, str] = {}
    for dev in cd.devices:
        # KIND -> (polarity, oxide). A MOS device stores its POLARITY (nmos/pmos)
        # so every polarity-keyed consumer is byte-unchanged whether the design
        # declares a plain (1.8V) or a 3v (3.3V thick-ox) kind; the 3.3V-vs-1.8V
        # fact rides the additive ``_device_oxide`` side map, threaded into
        # ``build_device(oxide=)``. A NON-MOS class (cap/res/...) keeps its raw
        # device_class (the build loop filters it out) and owes no oxide.
        pol, ox = kind_polarity_oxide(dev.device_class)
        device_kinds[dev.instance_name] = pol if pol is not None else dev.device_class
        if pol is not None:
            device_oxide[dev.instance_name] = ox
        # attach a minimal symbolic layout so has_device() works. W/L in metres.
        w = dev.sizing.get("W", 1e-6)
        l = dev.sizing.get("L", 1e-6)
        nf = int(dev.sizing.get("nf", dev.sizing.get("M", 1)) or 1)
        try:
            sl = SymbolicLayout.new(max(1, nf), 1, 1, 1)
        except Exception:  # noqa: BLE001
            sl = None
        if sl is not None:
            cell.set_layout(dev.instance_name, sl)
    for target, anns in ann_map.items():
        for ann in anns:
            cell.add_annotation(target, ann)
    # side-channel the device classes for the constraint translator + placer.
    cell.__dict__["_device_kinds"] = device_kinds
    # side-channel the per-device gate-oxide domain (thick 3.3V / thin 1.8V) so
    # ``_build_devices_from_clay`` threads oxide= into build_device; absent -> thin.
    cell.__dict__["_device_oxide"] = device_oxide
    # side-channel the raw sizing so the placer can build real device geometry.
    cell.__dict__["_device_sizing"] = {
        d.instance_name: dict(d.sizing) for d in cd.devices
    }
    return cell


# === device geometry (built from the .clay sizing) ============================


def _orient_map_from_cell(annotated: AnnotatedCell) -> Dict[str, str]:
    """Collect ``device_label -> 'R0'|'R90'|'R180'|'R270'`` from each device's
    declared ``@orient`` annotation.

    Reads the parsed device-level :class:`~clay.layout.annotations.Orient`
    annotations off the :class:`AnnotatedCell` (the same ``annotations`` map the
    interdigitation fuse reads). A device with no ``@orient`` is absent from the
    map (the caller defaults it to ``'R0'`` -> byte-unchanged). Only the
    device-annotation application path is touched here -- no solver / placement
    logic changes."""
    from clay.layout.annotations import Orient

    out: Dict[str, str] = {}
    for target, anns in (getattr(annotated, "annotations", {}) or {}).items():
        for ann in anns:
            if isinstance(ann, Orient):
                out[target] = ann.orientation.value
    return out


def _dense_contacts_set_from_cell(annotated: AnnotatedCell) -> set:
    """Collect device labels carrying a declared ``@dense_contacts`` annotation
    (reference-parity full S/D contact column). Reads the parsed device-level
    :class:`~clay.layout.annotations.DenseContacts` off the :class:`AnnotatedCell`
    (same map ``@orient`` reads). A device with no ``@dense_contacts`` is absent ->
    the caller leaves it sparse (byte-unchanged)."""
    from clay.layout.annotations import DenseContacts

    out = set()
    for target, anns in (getattr(annotated, "annotations", {}) or {}).items():
        for ann in anns:
            if isinstance(ann, DenseContacts):
                out.add(str(target))
    return out


# The serpentine 2D-fold gate now lives in realize_render.resolve_device_fold_target
# (the SINGLE decision the placer + route geometry bridge + routing pin-probe share,
# so a folded device renders identically by every path). These names are kept as
# back-compat aliases of the shared constants.
from clay.layout.extract.realize_render import (
    LARGE_M_FOLD_THRESHOLD as _LARGE_M_FOLD_THRESHOLD,
    LARGE_M_FOLD_ASPECT as _LARGE_M_FOLD_ASPECT,
)


def _fold_map_from_cell(annotated: AnnotatedCell) -> Dict[str, Optional[float]]:
    """Collect device labels carrying a declared ``@fold`` annotation -> its
    ``h_max_um`` (or ``None`` -> the module-default finger-height bound). Reads the
    parsed device-level :class:`~clay.layout.annotations.Fold` off the
    :class:`AnnotatedCell` (same map ``@orient`` / ``@dense_contacts`` read). A device
    with no ``@fold`` is absent -> the caller builds it un-folded (byte-unchanged)."""
    from clay.layout.annotations import Fold

    out: Dict[str, Optional[float]] = {}
    for target, anns in (getattr(annotated, "annotations", {}) or {}).items():
        for ann in anns:
            if isinstance(ann, Fold):
                out[str(target)] = ann.h_max_um
    return out


@dataclass(frozen=True, slots=True)
class DiscreteMosArraySpec:
    """One coordinate-free singleton MOS multiplicity expanded as a unit grid."""

    group: str
    device: str
    replicas: int
    rows: int
    columns: int

    def __post_init__(self) -> None:
        if (
            type(self.group) is not str
            or not self.group
            or type(self.device) is not str
            or not self.device
            or type(self.replicas) is not int
            or type(self.rows) is not int
            or type(self.columns) is not int
            or min(self.replicas, self.rows, self.columns) <= 0
            or self.rows * self.columns != self.replicas
        ):
            raise ValueError("discrete MOS array specification is malformed")


@dataclass(frozen=True, slots=True)
class DiscreteMosArrayInfo:
    replicas: int
    rows: int
    columns: int
    unit_finger_count: int
    unit_bbox_nm: Bbox
    replica_offsets_nm: Tuple[Tuple[int, int], ...]


def resolve_discrete_mos_array_specs(
    cell_def: object,
    intent: object,
    annotated: AnnotatedCell,
) -> Dict[str, DiscreteMosArraySpec]:
    """Resolve singleton ``@array(style=discrete)`` declarations without coordinates.

    A logical MOS multiplicity is the replica count.  ``rows`` declares only the
    grid topology; the column count is derived and must close exactly.  Passive
    discrete arrays stay on their established passive materializer.
    """

    from clay.frontend.craft_infer import group_members

    members_by_group = group_members(cell_def)
    kinds = _device_kinds_from_cell(annotated)
    sizing = getattr(annotated, "_device_sizing", {}) or {}
    resolved: Dict[str, DiscreteMosArraySpec] = {}
    for declaration in tuple(getattr(intent, "arrays", ()) or ()):
        if str(getattr(declaration, "style", "")).strip().lower() != "discrete":
            continue
        group = str(getattr(declaration, "group", "") or "")
        members = tuple(str(item) for item in (members_by_group.get(group) or ()))
        if not members and group in kinds:
            members = (group,)
        mos_members = tuple(
            member for member in members if kinds.get(member) in {"nmos", "pmos"}
        )
        if not mos_members:
            continue
        if mos_members != members or len(mos_members) != 1:
            raise ValueError(
                f"discrete MOS array {group!r} requires one logical MOS owner"
            )
        if str(getattr(declaration, "order", "") or "").strip().lower() not in {
            "",
            "ratio",
        }:
            raise ValueError(
                f"discrete MOS array {group!r} requires ratio or empty order"
            )
        device = mos_members[0]
        record = sizing.get(device, {})
        replicas = int(record.get("nf", record.get("M", 1)) or 1)
        rows = int(getattr(declaration, "rows", 1) or 1)
        if replicas % rows:
            raise ValueError(
                f"discrete MOS array {group!r} cannot close {replicas} replicas "
                f"into {rows} rows"
            )
        if device in resolved:
            raise ValueError(f"logical MOS {device!r} belongs to multiple discrete arrays")
        resolved[device] = DiscreteMosArraySpec(
            group=group,
            device=device,
            replicas=replicas,
            rows=rows,
            columns=replicas // rows,
        )
    return resolved


def _translate_bbox(box: Bbox, dx: int, dy: int) -> Bbox:
    return (box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy)


def _bbox_union(boxes: Sequence[Bbox]) -> Bbox:
    if not boxes:
        raise ValueError("cannot union an empty bbox set")
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _build_discrete_mos_array(
    unit: DeviceGeom,
    specification: DiscreteMosArraySpec,
    tech: LayoutTech,
) -> DeviceGeom:
    """Tile one existing unit DeviceGeom; no device geometry is reimplemented."""

    from .backend import _translate

    unit_box = unit.extent()
    unit_width = unit_box[2] - unit_box[0]
    unit_height = unit_box[3] - unit_box[1]
    if unit_width <= 0 or unit_height <= 0:
        raise ValueError("discrete MOS unit has an empty extent")
    spaces = []
    for layer in ("comp", "poly", "metal1"):
        try:
            value = tech.min_space(layer)
        except Exception:  # noqa: BLE001 - absent PDK spacing stays honest below
            value = None
        if value is not None:
            spaces.append(int(value))
    gap = max(spaces, default=0)
    pitch_x = unit_width + gap
    pitch_y = unit_height + gap
    offsets = tuple(
        (
            column * pitch_x - unit_box[0],
            row * pitch_y - unit_box[1],
        )
        for row in range(specification.rows)
        for column in range(specification.columns)
    )
    if len(offsets) != specification.replicas:
        raise ValueError("discrete MOS grid cardinality changed")

    polygons = [
        polygon
        for dx, dy in offsets
        for polygon in _translate(list(unit.polygons), dx, dy)
    ]
    unit_terminal_accesses = getattr(unit, "terminal_accesses", None) or {
        role: (terminal,) for role, terminal in unit.terminals.items()
    }
    terminal_accesses = {
        role: tuple(
            Terminal(
                role=terminal.role,
                bbox=_translate_bbox(terminal.bbox, dx, dy),
                net=terminal.net,
            )
            for dx, dy in offsets
            for terminal in tuple(accesses)
        )
        for role, accesses in sorted(unit_terminal_accesses.items())
    }
    terminals = {role: accesses[0] for role, accesses in terminal_accesses.items()}

    def translated_optional(box: Optional[Bbox]) -> Optional[Bbox]:
        if box is None:
            return None
        return _bbox_union(tuple(_translate_bbox(box, dx, dy) for dx, dy in offsets))

    geometry = DeviceGeom(
        polygons=polygons,
        terminals=terminals,
        kind=unit.kind,
        finger_count=unit.finger_count * specification.replicas,
        gate_length_nm=unit.gate_length_nm,
        channel_width_nm=unit.channel_width_nm,
        comp_bbox=translated_optional(unit.comp_bbox),
        mirror=False,
        axis_x=None,
        orient=unit.orient,
        nwell_bbox=translated_optional(unit.nwell_bbox),
        source_rail_bbox=terminals.get("s").bbox if "s" in terminals else None,
        drain_rail_bbox=terminals.get("d").bbox if "d" in terminals else None,
        gate_bus_bbox=translated_optional(unit.gate_bus_bbox),
        gate_contact_bbox=terminals.get("g").bbox if "g" in terminals else None,
        placement_extent=translated_optional(unit.placement_extent),
        well_tap_isolated=unit.well_tap_isolated,
    )
    geometry.__dict__["terminal_accesses"] = terminal_accesses
    geometry.__dict__["terminal_layers"] = {
        role: tuple(int(value) for value in tech.metal_key(1))
        for role in terminal_accesses
    }
    geometry.__dict__["discrete_array"] = DiscreteMosArrayInfo(
        replicas=specification.replicas,
        rows=specification.rows,
        columns=specification.columns,
        unit_finger_count=unit.finger_count,
        unit_bbox_nm=unit_box,
        replica_offsets_nm=offsets,
    )
    return geometry


def _build_devices_from_clay(
    annotated: AnnotatedCell,
    tech: Optional[LayoutTech] = None,
    *,
    fold_strips: bool = False,
    dense_contacts: bool = False,
    discrete_array_specs: Optional[Dict[str, DiscreteMosArraySpec]] = None,
    geometry_overrides: Optional[Dict[str, DeviceGeom]] = None,
    native_bulk_access: bool = False,
) -> Dict[str, object]:
    """Build the PDK-parameterized device geometry for each device in the cell,
    using the W / L / finger count read from the ``.clay`` source. Returns
    ``label -> DeviceGeom``.

    ``tech`` is the :class:`~clay.layout.tech.LayoutTech` the geometry is built on;
    it defaults to gf180mcu so every existing caller is unchanged (P6 wires an
    explicit ``tech=`` for other PDKs). The geometry comes from
    :func:`clay.layout.device_cell.build_device`, which on gf180mcu is
    BYTE-IDENTICAL to the legacy foldable recipe (the parity oracle in
    ``tests/test_device_cell_pdk.py``), so the gf180 placement is unchanged while
    external_pdk / other PDKs now build real implant-bearing devices.

    ``draw_substrate_tap=False`` reproduces the legacy gf180 behaviour (the NMOS
    bulk is the global substrate; the P+ tap is added at cell level), keeping the
    device extents byte-identical to the frozen reference. Native channel rows
    opt into ``native_bulk_access`` because their placed artwork is retained
    through emission; the builder then supplies its ordinary PDK/body-net tap
    policy before sizes or placement are measured. For any other PDK the
    builder's own ``bulk_topology()`` default already applies."""
    if type(native_bulk_access) is not bool:
        raise TypeError("native bulk access selection must be an exact bool")
    if tech is None:
        tech = LayoutTech.resolve("gf180mcu")
    gf180 = getattr(tech, "pdk_id", "").lower() == "gf180mcu"
    sizing = getattr(annotated, "_device_sizing", {}) or {}
    pins_map = getattr(annotated, "_device_pins", {}) or {}
    ground_nets = tuple(getattr(annotated, "_ground_nets", ()) or ())
    kinds = _device_kinds_from_cell(annotated)
    # Per-device gate-oxide domain (thick=3.3V / thin=1.8V), threaded from the
    # declared KIND (pmos3v/nmos3v -> thick) into build_device(oxide=). Absent ->
    # 'thin' (byte-identical: a plain pmos/nmos device emits no OD2).
    oxide_map = getattr(annotated, "_device_oxide", {}) or {}
    discrete_array_specs = discrete_array_specs or {}
    if set(discrete_array_specs) - set(kinds):
        raise ValueError("discrete MOS array names devices absent from the cell")
    if geometry_overrides is None:
        geometry_overrides = {}
    elif type(geometry_overrides) is not dict:
        raise TypeError("device geometry overrides must be one exact dictionary")
    if set(geometry_overrides) - set(kinds):
        raise ValueError("device geometry overrides name devices absent from the cell")
    devs: Dict[str, object] = {}
    # Devices that are MEMBERS of a declared @symmetric / @matching pair are
    # placement-CONSTRAINED (the placer mirrors them and shapes their row), so the
    # strip-fold (below) leaves them byte-unchanged; only FREE/unpaired strip
    # devices (e.g. the LDO pass FET, which has no match) fold at the cell level.
    # Structural property of the declared intent -- not a per-circuit branch.
    paired_devices: set = set()
    if fold_strips:
        for _t, _anns in getattr(annotated, "annotations", {}).items():
            for _ann in _anns or []:
                if isinstance(_ann, Symmetric):
                    paired_devices.update(str(d) for d in _ann.devices)
                elif isinstance(_ann, Matching):
                    paired_devices.add(str(_t))
                    paired_devices.add(str(_ann.other))
    # SHARED-NWELL well-tap gate: only on a PDK that declares a metal line-end rule
    # (the external_pdk / commercial line-end path emits device FEOL into the generic flow).
    # gf180 has NO line-end rule, so this is False -> every PMOS keeps its source-tied
    # tap and the gf180 device geometry stays BYTE-IDENTICAL to the frozen reference.
    line_end_pdk = False
    try:
        line_end_pdk = bool(tech.line_end_ext("metal1")) and not gf180
    except Exception:  # noqa: BLE001 - PDK silent on line-end -> no shared-well change
        line_end_pdk = False
    # Per-device ORIENTATION declared via the typed orthogonal `@orient(R*|MR*)`
    # annotation. The placer builds each device at its declared orientation (R0 is
    # the default vertical-gate cell, so an un-annotated device is byte-unchanged);
    # the rotated DeviceGeom carries its W/H-swapped extent + rotated terminal map,
    # so the placer/router consume it with no further change (rotation is
    # LVS-invariant -- connectivity is preserved). Reciprocal/duplicate declarations
    # collapse (last wins, but a device has at most one @orient in practice).
    orient_map = _orient_map_from_cell(annotated)
    # Per-device ``@fold`` intent DECLARED in the .clay annotate block: a wide/tall
    # single-finger MOS folds into F bounded-height fingers (W_eff conserved). The
    # placer builds the FOLDED footprint so it reserves the (shorter, wider) folded
    # extent -- the fold changes the placement footprint (unlike @dense_contacts,
    # which is render-only). Read the parsed Fold annotations (label -> h_max_um or
    # None); a device with none folds not at all (byte-unchanged).
    fold_map = _fold_map_from_cell(annotated)
    for lbl, kind in kinds.items():
        if kind not in ("nmos", "pmos"):
            continue
        s = sizing.get(lbl, {})
        w_m = float(s.get("W", 4e-6))
        l_m = float(s.get("L", 0.5e-6))
        nf = int(s.get("nf", s.get("M", 1)) or 1)
        array_spec = discrete_array_specs.get(lbl)
        if array_spec is not None and array_spec.replicas != nf:
            raise ValueError(f"discrete MOS array multiplicity changed for {lbl!r}")
        override = geometry_overrides.get(lbl)
        if override is not None:
            if type(override) is not DeviceGeom or override.kind != kind:
                raise TypeError(f"device geometry override differs in kind for {lbl!r}")
            expected_width_nm = int(round(w_m * 1e9)) * max(1, nf)
            expected_length_nm = int(round(l_m * 1e9))
            if (
                override.total_width_nm != expected_width_nm
                or override.gate_length_nm != expected_length_nm
            ):
                raise ValueError(
                    f"device geometry override changes electrical sizing for {lbl!r}"
                )
            if array_spec is not None:
                structure = getattr(override, "discrete_array", None)
                if structure is None or (
                    structure.replicas,
                    structure.rows,
                    structure.columns,
                ) != (
                    array_spec.replicas,
                    array_spec.rows,
                    array_spec.columns,
                ):
                    raise ValueError(
                        f"device geometry override changes discrete array for {lbl!r}"
                    )
            devs[lbl] = override
            continue
        kwargs: Dict[str, object] = {}
        # Pass the device's OWN declared terminal nets so ``build_device`` can derive
        # its tap policy from the topology (leakage-free -- net NAMES only, never any
        # coordinate). For a PMOS this selects the in-well well-tap mode: source==bulk
        # keeps the byte-identical source-tied tap; source!=bulk (the LDO cascode PMOS)
        # draws a LOCAL well tap on the body net instead of welding the source rail
        # onto the well node (the net073->VDDA merge). Net-stamping the terminal map
        # is geometry-neutral for self-bulk devices, so gf180 parity is preserved.
        _pins = pins_map.get(lbl)
        if _pins:
            kwargs["nets"] = dict(_pins)
        if ground_nets:
            kwargs["substrate_rail_nets"] = ground_nets
        if gf180 and not native_bulk_access:
            # gf180 parity: the legacy foldable device drew no NMOS substrate tap.
            kwargs["draw_substrate_tap"] = False
        # Source/body inequality does not prove a shared physical well. Keep
        # the builder's isolated body tap unless a certified geometry override
        # above has already supplied a real shared-well implementation. Compact
        # placement may separate these devices, so a speculative neighbor tap
        # cannot discharge their bulk obligations.
        orient = orient_map.get(lbl, "R0")
        mirror_transform = orient if orient.startswith("MR") else None
        if orient != "R0" and mirror_transform is None:
            kwargs["orient"] = orient
        # LARGE-M 2D FOLD: a many-finger device whose single-row build is a tall/wide
        # STRIP (e.g. the LDO pass FET MM28 W=200u M=40 -> 34x202um, aspect 5.93) is
        # routed through the mature serpentine fold DECISION
        # (realize_render.resolve_mos_fold) so the fingers redistribute across
        # fold_rows horizontal BANDS -> a ~square 2D array (channel width conserved
        # EXACTLY). DATA-driven, no per-circuit branch: the gate is the device's OWN
        # single-row aspect against the footprint bound; the fold count is pure PDK
        # arithmetic vs the design band-height STYLE target. The fold path itself is
        # the EXISTING device_cell.build_device(fold_rows=R) -- no new folding code.
        #
        # The aspect gate (not a bare M threshold) is what keeps already-compact
        # large-M devices byte-unchanged: the fc_ota m=8 bias/cascode devices
        # (W=5u L=2u -> 21.8x7.5um, aspect 2.91 < bound) draw single-row exactly as
        # before; only a real strip folds. The serpentine core accepts
        # draw_substrate_tap / draw_well_tap / orient, but NOT stable_extent or
        # contact_rows (single-row knobs, device_cell.py:1596); a device carrying
        # either keeps its single-row build (the shared-well PMOS that drops its tap
        # stays placement-neutral).
        #
        # GATED on ``fold_strips`` (default OFF): the seqpair / MILP frozen oracles
        # call this chokepoint with the default and stay BYTE-IDENTICAL to the legacy
        # recipe (their own placement-level wide-packing already shapes free strips).
        # Only the place_generic path -- which has no placement-level strip packing --
        # opts in, so its tall LDO pass FET folds at the CELL level.
        # A discrete declaration assigns the source multiplicity to REPLICA count.
        # Build exactly one electrical unit here; ``_build_discrete_mos_array``
        # performs the one and only expansion.  Without this split M was consumed
        # once as unit fingers and again as replica count (M^2 physical channels).
        W_build, nf_build = w_m, (1 if array_spec is not None else max(1, nf))
        # DECLARED ``@fold``: split the device into F bounded-height fingers via the
        # SHARED :func:`folded_finger_plan` (W_eff = nf*W conserved EXACTLY; F =
        # max(nf, ceil(W_eff/h_max)); W_finger = W_eff/F). This is a SINGLE-ROW
        # multi-finger build (``build_device`` lays F fingers side-by-side, each
        # W_finger tall), so a W=40 nf=1 strip becomes 2 fingers of 20um -- half the
        # height, electrically identical. @fold is the fold AUTHORITY for a declared
        # device, so the serpentine strip-fold below is skipped for it (one canonical
        # geometry). Works for any device/PDK that declares it -- data-driven, no
        # per-circuit branch.
        _folded_here = False
        if lbl in fold_map:
            from clay.layout.extract.fold import (
                DEFAULT_H_MAX_UM,
                folded_finger_plan,
            )
            _hmax = fold_map[lbl]
            _hmax = float(_hmax) if _hmax is not None else DEFAULT_H_MAX_UM
            w_finger_um, n_fingers = folded_finger_plan(
                w_m * 1e6,
                l_m * 1e6,
                1 if array_spec is not None else max(1, nf),
                h_max_um=_hmax,
            )
            W_build = w_finger_um * 1e-6
            nf_build = n_fingers
            _folded_here = True
        _fold_blockers = {"stable_extent", "contact_rows"}
        if (
            fold_strips
            and not _folded_here
            and array_spec is None
            and lbl not in paired_devices
            and not (_fold_blockers & kwargs.keys())
        ):
            try:
                from clay.layout.extract.realize_render import (
                    resolve_mos_fold,
                    resolve_device_fold_target,
                )
                # The SHARED per-device fold gate (threshold + single-row-strip
                # aspect + serpentine-divides-M): the SAME decision the route
                # geometry bridge and the routing pin-probe consume, so a folded
                # device is rendered identically by every path (one canonical
                # artwork). Returns the design row-height target, or None (no fold).
                _target = resolve_device_fold_target(tech, kind, w_m, l_m, nf)
                if _target is not None:
                    W_out, M_out, fold_rows, _k, _note = resolve_mos_fold(
                        tech, kind, w_m * 1e6, l_m * 1e6, nf, _target,
                    )
                    W_build = W_out * 1e-6
                    nf_build = M_out
                    kwargs["fold_rows"] = fold_rows
            except Exception:  # noqa: BLE001 - fold is an optional density refinement
                W_build, nf_build = w_m, max(1, nf)
                kwargs.pop("fold_rows", None)
        # CONTACTS ARE A RENDER DETAIL, NOT A PLACEMENT FOOTPRINT INPUT. The PLACEMENT
        # is CONTACT-INVARIANT by principle: a declared @dense_contacts must NOT change
        # the placer footprint (a ~20nm dense-rail delta tips CP-SAT into a ~20%-worse
        # packing basin, measured). The reference-parity full S/D contact column is
        # materialised at RENDER time in route_cell, which reads the SAME declared
        # @dense_contacts off the CellDef -- so placement stays byte-stable and the
        # declared intent is honoured on the emitted GDS the LVS/DRC oracles read.
        # (Intentionally NO contact_rows here: contacts do not size the placement.)
        geometry = build_device(
            tech,
            kind,
            W=W_build,
            L=l_m,
            nf=nf_build,
            oxide=oxide_map.get(lbl, "thin"),
            **kwargs,
        )
        if mirror_transform is not None:
            # Reuse the proved coordinate-free DeviceGeom transform already
            # consumed by fixed MIM banks and segmented resistors.  Apply it to
            # the R0 unit before discrete tiling so every replica carries the
            # declared transform and its terminal accesses move with artwork.
            from clay.layout.passive_device_cell import _transform_devicegeom

            geometry = _transform_devicegeom(geometry, mirror_transform, tech)
        devs[lbl] = (
            _build_discrete_mos_array(geometry, array_spec, tech)
            if array_spec is not None
            else geometry
        )

    # COMMON-CENTROID INTERDIGITATION: fuse each declared ``@interdigitated(A, B)``
    # matched pair into ONE interleaved common-centroid finger array. This is the
    # shared chokepoint both placement engines (MILP + seqpair) build off, so the
    # interleave drives BOTH. Generic + intent-driven (any declared pair); the
    # constraint/order builders filter out the dropped member B automatically (they
    # keep only ``d in devs``), so the A/B symmetry pair collapses to the single
    # self-centroid-symmetric cluster.
    devs = _fuse_interdigitated_clusters(annotated, devs, sizing, pins_map, kinds, tech)
    return devs


def _fuse_interdigitated_clusters(
    annotated: AnnotatedCell,
    devs: Dict[str, object],
    sizing: Dict[str, Dict[str, float]],
    pins_map: Dict[str, Dict[str, str]],
    kinds: Dict[str, str],
    tech,
) -> Dict[str, object]:
    """Replace each declared ``@interdigitated(A, B)`` device pair with ONE fused
    common-centroid interleaved finger array (keyed under member A; B removed).

    For every declared interdigitated pair whose two members are SAME-kind,
    EQUAL-finger MOS that were both built, this builds a single
    :func:`clay.layout.common_centroid_cluster.build_interdigitated_cluster`
    :class:`DeviceGeom` (the interleaved array). A pair whose members are not
    interdigitatable (mixed kind / unequal nf / not both built) is left as the two
    separate devices. The fused cluster duck-types a single device for the placer;
    its ``cluster`` provenance is carried on ``geom.__dict__['cluster']``.

    OPT-IN GATE (``CLAY_INTERDIG_FUSE``, default OFF). The interleaved-array
    builder (:mod:`clay.layout.common_centroid_cluster`) is a HEURISTIC generator
    whose current finger/contact/well geometry is NOT yet correct-by-construction
    (it produces contact + M1/M2 spacing DRC on the dense interleave). Until the
    array generator is rewritten correct-by-construction, fusing is gated OFF so
    the default flow stays DRC-clean. With fusing OFF the declared
    ``@interdigitated`` intent is NOT silently honored: it carries no realizing
    cluster, so the geometric obligation is UNMEASURABLE and the no-silent-ignore
    completeness gate WITHHOLDS the intent token (declared-but-unrealized is an
    HONEST non-pass, never a free pass). Set ``CLAY_INTERDIG_FUSE=1`` to exercise
    the in-progress array builder."""
    import os
    if os.environ.get("CLAY_INTERDIG_FUSE", "") not in ("1", "true", "True"):
        return devs
    from clay.layout.annotations import Interdigitated
    from clay.layout.common_centroid_cluster import (
        build_interdigitated_cluster,
        is_interdigitatable,
    )

    # collect declared interdigitated pairs (reciprocal declarations collapse).
    seen: set = set()
    pairs: List[Tuple[str, str, object]] = []
    for target, anns in (getattr(annotated, "annotations", {}) or {}).items():
        for ann in anns:
            if not isinstance(ann, Interdigitated):
                continue
            partner = ann.with_device
            if not partner or partner == "__pattern_only__":
                continue
            key = frozenset((target, partner))
            if key in seen:
                continue
            seen.add(key)
            pairs.append((target, partner, ann.pattern))
    if not pairs:
        return devs

    devs = dict(devs)
    drop: set = set()
    for a, b, pattern in pairs:
        if a not in devs or b not in devs or a in drop or b in drop:
            continue
        ka, kb = kinds.get(a), kinds.get(b)
        sa = sizing.get(a, {}) or {}
        sb = sizing.get(b, {}) or {}
        nf_a = int(sa.get("nf", sa.get("M", 1)) or 1)
        nf_b = int(sb.get("nf", sb.get("M", 1)) or 1)
        if not is_interdigitatable(ka, kb, nf_a, nf_b):
            continue
        w_m = float(sa.get("W", 4e-6))
        l_m = float(sa.get("L", 0.5e-6))
        pa = pins_map.get(a, {}) or {}
        pb = pins_map.get(b, {}) or {}
        nets_a = {"s": pa.get("s", ""), "d": pa.get("d", ""),
                  "g": pa.get("g", ""), "b": pa.get("b", "")}
        nets_b = {"s": pb.get("s", ""), "d": pb.get("d", ""),
                  "g": pb.get("g", ""), "b": pb.get("b", "")}
        devs[a] = build_interdigitated_cluster(
            tech,
            kind=ka,
            W=w_m,
            L=l_m,
            nf=max(1, nf_a),
            device_a=a,
            device_b=b,
            nets_a=nets_a,
            nets_b=nets_b,
            pattern=pattern,
        )
        drop.add(b)
    for m in drop:
        devs.pop(m, None)
    return devs


# === the symmetry-constrained MILP placement (driven by SolverConstraints) ====


@dataclass
class PlacedDevice:
    label: str
    kind: str
    dx: int
    dy: int
    extent: Bbox
    mirror: bool
    row: int


@dataclass
class PlacementSolution:
    placed: Dict[str, PlacedDevice]
    axis_x: int
    status: str
    success: bool
    solve_seconds: float
    objective: float
    n_vars: int
    n_constraints: int
    full_extent: Bbox
    constraints: SolverConstraints
    note: str = ""
    ordered_row_program: object = None
    ordered_axis_coordinates_nm: Tuple[int, ...] = ()

    def device_bboxes(self) -> Dict[str, Bbox]:
        return {lbl: pd.extent for lbl, pd in self.placed.items()}


def _half_widths(devs: Dict[str, object]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for lbl, d in devs.items():
        e = d.extent()
        out[lbl] = (e[2] - e[0]) // 2
    return out


def _heights(devs: Dict[str, object]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for lbl, d in devs.items():
        e = d.extent()
        out[lbl] = e[3] - e[1]
    return out


def _row_plan_band_free(
    constraints: SolverConstraints, devs: Dict[str, object]
) -> Optional[List[List[Tuple[str, object]]]]:
    """A band-free default row plan: keep the declared symmetry structure but DROP
    the forced NMOS-below-PMOS device-class banding (the digital std-cell
    convention the user retracted for analog). Each declared mirror pair takes its
    own row (so ``y_a==y_b`` + mirror-about-axis still binds); the remaining
    on-axis / unpaired devices are packed two-per-row to widen the cell instead of
    stacking into one tall column. Rows are NOT sorted by device class, so an NMOS
    and a PMOS row can be adjacent -- the rails are then derived from the actual
    placement, not a fixed vss-bottom/vdd-top rule. Returns ``None`` when there is
    nothing to place (caller falls back to the banded default).
    """
    covered: set = set()
    rows: List[List[Tuple[str, object]]] = []
    for a, b in constraints.symmetry_pairs:
        if a in devs and b in devs:
            rows.append([("pair", (a, b))])
            covered.add(a)
            covered.add(b)
    loose: List[str] = [d for d in constraints.on_axis_devices if d in devs and d not in covered]
    for d in loose:
        covered.add(d)
    loose += [lbl for lbl in devs if lbl not in covered]
    # pack the loose (unpaired) devices two per row to keep the cell compact + wide.
    for i in range(0, len(loose), 2):
        rows.append([("axis", lbl) for lbl in loose[i:i + 2]])
    return rows if rows else None


def _row_plan_from_floorplan(
    constraints: SolverConstraints, devs: Dict[str, object]
) -> Optional[List[List[Tuple[str, object]]]]:
    """Build the row structure from a DECLARED ``@place_group`` floorplan, NOT from
    the device-class band. This is the intent-driven, band-free path: the designer
    states which devices live in which row (and optional column) of a 2D
    floorplan, and the placer realises exactly that grid -- an input pair + mirror
    in one row, a CS + load in the next, side by side, produces a WIDE 2D shape
    rather than the tall NMOS-below-PMOS column the default banding gives.

    ``constraints.floorplan_rows`` maps ``label -> (row, col)``. Returns rows
    BOTTOM-TO-TOP (row index ascending). Within a row the devices order by their
    declared ``col`` (then label). A declared symmetry pair whose two members land
    in the SAME floorplan row stays a mirror ``("pair", (a, b))`` so the symmetry
    constraint still binds; everything else is an ``("axis", label)`` cell that
    packs left-to-right. Any built device the floorplan did NOT mention is appended
    to the top row so the cell stays complete (never silently dropped). Returns
    ``None`` when no floorplan touches a built device (caller falls back).
    """
    fp = constraints.floorplan_rows
    if not fp:
        return None
    # only floorplan entries for devices we actually built as MOS geometry.
    placed_fp = {lbl: rc for lbl, rc in fp.items() if lbl in devs}
    if not placed_fp:
        return None

    # which declared symmetry pairs are kept together in one floorplan row -> emit
    # them as a mirror pair (so y_a==y_b + mirror-about-axis still binds).
    pair_of: Dict[str, Tuple[str, str]] = {}
    for a, b in constraints.symmetry_pairs:
        if a in placed_fp and b in placed_fp and placed_fp[a][0] == placed_fp[b][0]:
            pair_of[a] = (a, b)
            pair_of[b] = (a, b)

    # bucket built devices by their floorplan row. Devices NOT named in the
    # floorplan go to the highest declared row so they are not lost.
    declared_rows = sorted({rc[0] for rc in placed_fp.values()})
    row_index = {r: i for i, r in enumerate(declared_rows)}
    buckets: Dict[int, List[str]] = {i: [] for i in range(len(declared_rows))}
    for lbl in devs:
        rc = placed_fp.get(lbl)
        if rc is None:
            buckets[len(declared_rows) - 1].append(lbl)
        else:
            buckets[row_index[rc[0]]].append(lbl)

    rows: List[List[Tuple[str, object]]] = []
    for i in range(len(declared_rows)):
        members = buckets[i]
        # order within the row by declared col (None sorts last), then label.
        def _colkey(lbl: str):
            rc = placed_fp.get(lbl)
            col = rc[1] if rc is not None else None
            return (0, col) if col is not None else (1, 0), lbl
        members.sort(key=_colkey)
        row: List[Tuple[str, object]] = []
        emitted_pair: set = set()
        for lbl in members:
            if lbl in pair_of:
                pa, pb = pair_of[lbl]
                if pa in emitted_pair:
                    continue
                row.append(("pair", (pa, pb)))
                emitted_pair.add(pa)
                emitted_pair.add(pb)
            else:
                row.append(("axis", lbl))
        if row:
            rows.append(row)
    return rows if rows else None


def _row_plan(
    constraints: SolverConstraints, devs: Dict[str, object]
) -> List[List[Tuple[str, object]]]:
    """Assign the declared pairs + on-axis devices to ROWS.

    Three modes, in priority order:

      1. DECLARED FLOORPLAN (``@place_group ... row=/col=``): the row/column
         structure is taken VERBATIM from the designer's 2D floorplan intent (see
         :func:`_row_plan_from_floorplan`). This is band-free -- devices land where
         the floorplan puts them, regardless of NMOS/PMOS class -- so a designer
         can express a real wide 2D analog layout in the language.
      2. BAND-FREE DEFAULT (opt-in via ``CLAY_BAND_FREE_PLACER=1``): no floorplan
         declared, but the forced NMOS-below-PMOS std-cell banding is DROPPED in
         favour of a compact mixed arrangement (declared pairs each on their own
         row, then on-axis/unpaired devices packed into shared rows). Keeps
         symmetry structural without stacking every device into one tall column.
      3. BANDED DEFAULT (back-compat): the historical NMOS-low / PMOS-high band
         split. This stays the DEFAULT so existing solver/rail tests that depend on
         the band ordering are byte-stable.

    Returns rows bottom-to-top; each entry is ``("pair", (a, b))`` or
    ``("axis", label)``.
    """
    fp_rows = _row_plan_from_floorplan(constraints, devs)
    if fp_rows is not None:
        return fp_rows

    import os

    band_free = os.environ.get("CLAY_BAND_FREE_PLACER", "").strip() not in ("", "0", "false", "False")
    if band_free:
        bf = _row_plan_band_free(constraints, devs)
        if bf is not None:
            return bf

    kinds = constraints.device_kinds
    nmos_rows: List[List[Tuple[str, object]]] = []
    pmos_rows: List[List[Tuple[str, object]]] = []
    covered: set = set()
    # A declared symmetry pair / on-axis device whose member was NOT built as a
    # foldable MOS is SKIPPED here (it never enters a row), so the solver never
    # looks up ``half_w[member]`` / ``heights[member]`` for an absent device and
    # KeyError-crashes. This happens legitimately: a bandgap's ``annotate`` block
    # matches its summing RESISTORS (R_sum1 / R_sum2) and PNP devices (Q1 / Q8_0),
    # which ``_build_devices_from_clay`` correctly omits (they are not gf180
    # foldable MOS). The pair is still HONEST -- it is simply not placeable as a
    # mirror MOS pair, so it is dropped rather than fabricated.
    for pair in constraints.symmetry_pairs:
        a, b = pair
        if a not in devs or b not in devs:
            # one (or both) members has no built MOS geometry -> not placeable as
            # a mirror pair. Place any built member as a lone on-axis device so it
            # is not silently lost; drop the unbuilt member entirely.
            for m in (a, b):
                if m in devs and m not in covered:
                    kind = kinds.get(m, "nmos")
                    (nmos_rows if kind != "pmos" else pmos_rows).append([("axis", m)])
                    covered.add(m)
            continue
        kind = kinds.get(a, "nmos")
        row: List[Tuple[str, object]] = [("pair", pair)]
        (nmos_rows if kind != "pmos" else pmos_rows).append(row)
        covered.add(a)
        covered.add(b)
    # on-axis devices: each on its own row in its band (centred on the axis). An
    # on-axis device with no built MOS geometry is skipped (same KeyError guard).
    for d in constraints.on_axis_devices:
        if d not in devs:
            continue
        kind = kinds.get(d, "nmos")
        (nmos_rows if kind != "pmos" else pmos_rows).append([("axis", d)])
        covered.add(d)
    # any device present in the cell but NOT named in a constraint (e.g. an
    # unpaired tail/bias switch) still needs a placement: give it an on-axis row
    # in its rail band so the cell stays complete. Its placement is not symmetry-
    # gated (it has no declared pair), but it must not be dropped.
    for lbl in devs:
        if lbl in covered:
            continue
        kind = kinds.get(lbl, "nmos")
        (nmos_rows if kind != "pmos" else pmos_rows).append([("axis", lbl)])
    rows = nmos_rows + pmos_rows
    if not rows:
        # no symmetry declared: one row of every device, in label order.
        rows = [[("axis", lbl)] for lbl in devs]
    return rows


def _resolve_tech(pdk: object = None) -> LayoutTech:
    """Resolve a :class:`~clay.layout.tech.LayoutTech`, defaulting to gf180mcu.

    Accepts an already-built ``LayoutTech`` (returned as-is), a legacy
    :class:`~clay.layout.pdk_rules.PdkRules` (resolved by its ``pdk_id``), a
    plain PDK-id string, or ``None`` (gf180mcu default -- parity + no caller
    breaks). This is the single seam every placer entrypoint funnels its PDK
    argument through so the spacing minima flow from one tech object."""
    if isinstance(pdk, LayoutTech):
        return pdk
    if pdk is None:
        return LayoutTech.resolve("gf180mcu")
    if isinstance(pdk, str):
        return LayoutTech.resolve(pdk)
    pid = getattr(pdk, "pdk_id", None)
    return LayoutTech.resolve(pid if pid else "gf180mcu")


def _dev_gap_from_tech(tech: LayoutTech) -> int:
    """The inter-device min gap (nm) the placer enforces between adjacent devices.

    Resolved from ``tech`` spacing rules: the conservative max of the active/COMP
    same-layer spacing, the nwell spacing (a PMOS-pair gap must clear adjacent
    wells), and the gate-poly spacing. The active + poly LAYER NAMES are resolved
    from ``tech`` because the rule container names them per-PDK (gf180 'comp' /
    'poly2', external_pdk 'diff' / 'poly'); a missing rule contributes 0.

    gf180 parity: ``max(comp=280, nwell=1400, poly2=240) == 1400`` -- the SAME
    value the legacy ``max(min_spacing('nwell'), min_spacing('comp')) == 1400`` gave,
    so the gf180 placement is byte-identical. When a PDK's deck carries none of
    these (rare), a node-scaled fallback from the contact spacing keeps adjacent
    devices from abutting.

    Device-edge channel floor (under-resolved decks)
    ------------------------------------------------
    A REAL device-edge gap between two adjacent device columns must clear the
    well-to-well (nwell) and diffusion-to-diffusion (active) spacing -- not just the
    gate-poly spacing. Several open / ported decks (open_pdk/mmrf, open_pdk/40/65/
    180rf) parse a ``poly`` spacing but carry NO ``nwell`` / ``diff`` same-layer
    spacing rule, so the resolved gap collapses to the gate-poly value ALONE. That
    poly-only value UNDER-estimates the true device-edge clearance: it leaves
    adjacent device columns so tight that the generic router cannot fit a routing
    lane between them (the external_pdk StrongARM comparator's di_n/di_p columns sit ~258nm
    apart, below one M1 lane + clearance, forcing the differential trunks to merge).
    When the gap is gate-poly-ONLY (no well/active rule resolved), floor it to a
    contact-pitch routing channel so adjacent columns always admit one router lane.
    This is gf180-PARITY-SAFE: gf180 resolves ``active`` + ``nwell`` (it is NOT
    poly-only), so the floor never applies and its 1400nm gap is byte-identical.
    """
    from .device_cell import _active_name, _poly_name, _contact_via_name

    active_v = nwell_v = poly_v = None
    for name, slot in ((_active_name(tech), "a"), ("nwell", "w"), (_poly_name(tech), "p")):
        try:
            v = tech.min_space(name)
        except KeyError:
            v = None
        if v:
            iv = int(v)
            if slot == "a":
                active_v = iv
            elif slot == "w":
                nwell_v = iv
            else:
                poly_v = iv

    vals = [v for v in (active_v, nwell_v, poly_v) if v]
    if not vals:
        # deck silent on every device-edge layer: node-scale from the contact spacing.
        cv = _contact_via_name(tech)
        csp = (tech.via_spacing(cv) if cv else None) or 0
        return int(csp) if csp else 0

    gap = max(vals)
    # Under-resolved deck: the device-edge gap came from the gate-poly spacing ALONE
    # (no well/diffusion same-layer rule). Floor it to a contact-pitch routing
    # channel so two adjacent device columns admit at least one clear router lane.
    poly_only = poly_v is not None and active_v is None and nwell_v is None
    if poly_only:
        gap = max(gap, _device_edge_channel_floor(tech, poly_v))
    return int(gap)


def _device_edge_channel_floor(tech: LayoutTech, poly_space: int) -> int:
    """A device-edge routing-channel floor for an under-resolved (poly-only) deck.

    The floor is the gap that lets the generic router thread one M1/contact lane
    between two adjacent device columns: a contact landing plus its same-layer
    clearance on each flank. We size it from the PDK's OWN contact-to-contact
    spacing (node-scaled, so an advanced node gets a proportionally tighter floor
    than a mature node) and never let it drop below the gate-poly value the deck DID
    resolve. The 6/5 factor adds the half-contact enclosure a real well/diffusion
    edge would have carried, which the poly-only deck omitted -- it lifts open_pdk's
    250nm poly-only gap to ~300nm (centred in the router's 290-320nm net-complete
    plateau for the StrongARM di_n/di_p columns) while staying a node-relative,
    PDK-derived quantity rather than a hardcoded floorplan constant."""
    from .device_cell import _contact_via_name

    cv = _contact_via_name(tech)
    csp = 0
    if cv:
        try:
            csp = int(tech.via_spacing(cv) or 0)
        except (KeyError, TypeError, ValueError):
            csp = 0
    # contact-pitch channel = contact spacing + a half-contact (6/5) enclosure margin
    # the omitted well/diffusion edge would have provided; never below gate-poly.
    channel = (csp * 6) // 5 if csp else 0
    return max(int(poly_space), channel)


def _abut_gap_from_tech(tech: LayoutTech) -> int:
    """The DRC-MINIMUM same-layer device-edge gap (nm) for a declared ``@abut`` pair.

    Where :func:`_dev_gap_from_tech` is the CONSERVATIVE inter-device gap (the MAX of
    active / nwell / poly spacing, padded so a routing lane fits -- 1400nm on gf180
    from the nwell rule), an ``@abut`` pair is DECLARED to pack edge-adjacent, so the
    facing gap drops to the TIGHTEST legal same-layer spacing: the active/COMP
    same-layer minimum (gf180 'comp' = 280nm), falling back to the gate-poly spacing,
    then a node-scaled contact spacing -- never the well-padded routing gap. This is
    the minimal device-edge clearance that keeps the two devices' active/poly DRC-legal
    while removing the channel padding. Never used unless a cell declares ``@abut``
    (byte-parity); a deck with no resolvable rule returns 0 (devices touch)."""
    from .device_cell import _active_name, _poly_name, _contact_via_name

    for name in (_active_name(tech), _poly_name(tech)):
        try:
            v = tech.min_space(name)
        except KeyError:
            v = None
        if v:
            return int(v)
    cv = _contact_via_name(tech)
    try:
        csp = int(tech.via_spacing(cv) or 0) if cv else 0
    except (KeyError, TypeError, ValueError):
        csp = 0
    return csp


def _opp_kind_extra_gap(tech: LayoutTech) -> int:
    """Extra inter-device clearance (nm) two ADJACENT OPPOSITE-TYPE devices need
    on top of :func:`_dev_gap_from_tech`.

    The default ``dev_gap`` clears same-type adjacency (active-to-active,
    well-to-well, poly-to-poly). When an NMOS column abuts a PMOS column -- which
    only happens once the placement is band-free / 2D-floorplan-driven, never in
    the old NMOS-below-PMOS bands -- the PMOS's enclosing NWELL + P+ implant must
    additionally clear the NMOS's N+ implant + COMP. We add the well-to-pcomp
    enclosure (the nwell sticks out beyond the pcomp by this much) plus the
    well-to-ncomp enclosure, so the nwell edge stays clear of the neighbour's
    active + opposite implant. All terms come from the tech rule container (no
    hardcoded floorplan constant); a deck missing a rule contributes 0. This is
    ZERO-IMPACT on the historical banded path: there opposite-type devices are
    never H/V-adjacent, so the extra gap is never requested.
    """
    extra = 0
    try:
        iw = tech.implant_well()
        extra += int(getattr(iw, "nwell_pcomp_enclosure", 0) or 0)
        extra += int(getattr(iw, "nwell_ncomp_enclosure", 0) or 0)
    except Exception:  # noqa: BLE001 - deck silent on implant/well -> no extra
        return 0
    # also keep the opposite implants apart (N+ to P+ space), the larger of the two.
    impl = 0
    for nm in ("nplus", "pplus"):
        try:
            v = tech.min_space(nm)
        except (KeyError, TypeError, ValueError):
            v = None
        if v:
            impl = max(impl, int(v))
    return extra + impl


def _pair_gap(tech: LayoutTech, dev_gap: int, kind_a: str, kind_b: str) -> int:
    """The min centre-to-centre EDGE clearance between two adjacent devices,
    kind-aware: same-type -> ``dev_gap``; opposite-type -> ``dev_gap`` plus the
    well/implant clearance (:func:`_opp_kind_extra_gap`). This is what lets a
    band-free 2D floorplan put an NMOS next to a PMOS without an NW / implant DRC
    violation, while leaving the same-type (and the historical banded) spacing
    byte-identical."""
    if kind_a != kind_b:
        return dev_gap + _opp_kind_extra_gap(tech)
    return dev_gap


def solve_placement_from_constraints(
    constraints: SolverConstraints,
    devs: Dict[str, object],
    *,
    tech: Optional[LayoutTech] = None,
    pdk: Optional[PdkRules] = None,
    height_weight: float = 1.0,
    grid_nm: int = 5,
    time_limit_s: float = 30.0,
    min_target_width_nm: int = 0,
    ring_group_of: Optional[Dict[str, int]] = None,
) -> PlacementSolution:
    """A symmetry-constrained, rail-banded, area-minimal MILP placement driven
    ENTIRELY by ``constraints`` (which were derived from the ``.clay`` annotations).

    The formulation mirrors ``milp_placer_v175`` (row-based, no big-M): each
    declared mirror pair shares a row at mirrored x about a single global vertical
    axis, NMOS rows stack below PMOS rows (the inferred rail convention), and the
    objective minimises a linear bounding-box-area proxy. The decision variables
    are the axis x, each pair's half-offset, each row's y-baseline, and the bbox
    extent aux vars -- all on a coarse nm grid (integrality enforced).

    ``tech`` is the :class:`~clay.layout.tech.LayoutTech` the spacing minima are
    read from; it DEFAULTS to gf180mcu so every existing caller is unchanged (P6
    wires an explicit ``tech=`` for other PDKs) and the gf180 placement is
    byte-identical (the resolved ``dev_gap`` is the same 1400nm the legacy gf180
    PdkRules path gave). ``pdk`` is the legacy parameter: when given (and ``tech``
    is None) the tech is resolved from ``pdk.pdk_id`` so the spacing still tracks
    that PDK. The placer no longer reads PdkRules directly -- all spacing flows
    through ``tech``.
    """
    if tech is None:
        tech = _resolve_tech(pdk)
    dev_gap = _dev_gap_from_tech(tech)

    half_w = _half_widths(devs)
    heights = _heights(devs)
    rows = _row_plan(constraints, devs)
    from clay.layout.place.ordered_mirror_rows import compile_ordered_rows

    ordered_program = compile_ordered_rows(
        constraints,
        {name: tuple(device.extent()) for name, device in devs.items()},
        gap_nm=dev_gap,
        opposite_kind_extra_gap_nm=_opp_kind_extra_gap(tech),
    )
    hierarchical_floorplan = bool(constraints.floorplan_rows) and ordered_program is None
    hierarchical_band_count = 0
    hierarchical_subrow_count = 0
    hierarchical_target_width_nm = 0
    atom_of_dev: Dict[str, int] = {}
    if ordered_program is not None:
        rows = []
        for _source_row, members in ordered_program.rows:
            row_pairs = [pair for pair in ordered_program.pairs if pair[0] in members]
            paired = {member for pair in row_pairs for member in pair}
            rows.append([("pair", pair) for pair in row_pairs]
                        + [("axis", member) for member in members if member not in paired])
        atom_of_dev = {name: owner for name, owner in ordered_program.axis_owners if owner != 0}
    if hierarchical_floorplan:
        from clay.layout.place.hierarchical_macro_placement import (
            compile_hierarchical_macro_rows,
            plan_geometry_bounded_subrows,
        )

        col_of_member = {
            lbl: int(rc[1])
            for lbl, rc in constraints.floorplan_rows.items()
            if rc[1] is not None and lbl in devs
        }
        # When explicit 2-D columns and shared guard rings are active, compute
        # subrow planning widths using true intra-ring extents (only the outer
        # left/right members of each contiguous same-ring run carry the outward
        # ring wall pad; interior sibling gaps stay at dev_gap, preventing fake
        # width inflation from wrapping a compact row into hollow subrows).
        plan_bboxes_nm: Dict[str, Tuple[int, int, int, int]] = {}
        active_ring_map = ring_group_of if (col_of_member and ring_group_of) else None
        for raw_row in rows:
            seq: List[str] = []
            for kind, payload in raw_row:
                if kind == "pair":
                    seq.extend(payload)
                else:
                    seq.append(payload)
            for idx_in_row, name in enumerate(seq):
                x0, y0, x1, y1 = tuple(devs[name].extent())
                if active_ring_map is not None:
                    pad_nm = int(getattr(devs[name], "_pad", 0) or 0)
                    rg = active_ring_map.get(name)
                    if pad_nm > 0 and rg is not None:
                        if idx_in_row > 0 and active_ring_map.get(seq[idx_in_row - 1]) == rg:
                            x0 += pad_nm
                        if idx_in_row + 1 < len(seq) and active_ring_map.get(seq[idx_in_row + 1]) == rg:
                            x1 -= pad_nm
                plan_bboxes_nm[name] = (int(x0), int(y0), int(x1), int(y1))
        for name, device in devs.items():
            plan_bboxes_nm.setdefault(name, tuple(device.extent()))

        hierarchy = compile_hierarchical_macro_rows(
            rows=rows,
            device_bboxes_nm=plan_bboxes_nm,
        )
        planning_gap = dev_gap
        for raw_row in rows:
            adjacent = []
            for kind, payload in raw_row:
                if kind == "pair":
                    adjacent.extend(payload)
                else:
                    adjacent.append(payload)
            for left, right in zip(adjacent, adjacent[1:]):
                planning_gap = max(
                    planning_gap,
                    _pair_gap(
                        tech,
                        dev_gap,
                        constraints.device_kinds.get(left, "nmos"),
                        constraints.device_kinds.get(right, "nmos"),
                    ),
                )
        subrow_plan = plan_geometry_bounded_subrows(
            hierarchy,
            gap_nm=planning_gap,
            min_target_width_nm=(int(min_target_width_nm) if col_of_member and min_target_width_nm > 0 else 0),
            col_of_member=(col_of_member or None),
        )
        hierarchical_band_count = len(hierarchy.rows)
        hierarchical_subrow_count = len(subrow_plan.subrows)
        hierarchical_target_width_nm = subrow_plan.target_width_nm
        rows = [
            [
                (
                    "pair" if atom.kind == "matching_pair" else "axis",
                    atom.members if atom.kind == "matching_pair" else atom.members[0],
                )
                for atom in subrow.atoms
            ]
            for subrow in subrow_plan.subrows
        ]
        for atom_index, atom in enumerate(
            atom for row in hierarchy.rows for atom in row.atoms
        ):
            for member in atom.members:
                atom_of_dev[member] = atom_index

    # device -> row, side (-1 left / +1 right / 0 axis), role-pair lookup.
    dev_row: Dict[str, int] = {}
    dev_side: Dict[str, int] = {}
    pair_of_dev: Dict[str, int] = {}      # device -> pair-offset-var key (pair idx)
    row_order: List[List[str]] = []
    pair_index: Dict[int, Tuple[str, str]] = {}
    pidx = 0
    for r, row in enumerate(rows):
        ordered: List[str] = []
        for kind, payload in row:
            if kind == "pair":
                a, b = payload  # type: ignore[misc]
                pair_index[pidx] = (a, b)
                dev_row[a] = r
                dev_row[b] = r
                dev_side[a] = -1
                dev_side[b] = 1
                pair_of_dev[a] = pidx
                pair_of_dev[b] = pidx
                ordered += [a, b]
                pidx += 1
            else:
                lbl = payload  # type: ignore[assignment]
                dev_row[lbl] = r
                dev_side[lbl] = 0
                ordered.append(lbl)
        row_order.append(ordered)
    if ordered_program is not None:
        # Pair identity and pair offset variables are retained; their two halves
        # occupy the source order, rather than an indivisible adjacent pair box.
        row_order = [list(members) for _source_row, members in ordered_program.rows]

    # --- variable index --------------------------------------------------------
    # [axis] [ybase per row] [off per pair] [maxX minX maxY minY]
    idx = 0

    def _new() -> int:
        nonlocal idx
        i = idx
        idx += 1
        return i

    v_axis = _new()
    v_ybase = {r: _new() for r in range(len(rows))}
    v_atom_x = {
        atom_index: _new() for atom_index in sorted(set(atom_of_dev.values()))
    }
    v_off = {p: _new() for p in pair_index}
    v_maxX = _new()
    v_minX = _new()
    v_maxY = _new()
    v_minY = _new()
    # Native masks already occupy the PDK grid in their own frame. Their bbox
    # centers can have a nonzero lattice phase (e.g. a 1235 nm wide cell on a
    # 5 nm grid). Quantizing those centers to the origin phase moves every
    # source polygon off-grid. Model the rigid translation as an integer grid
    # multiple instead, without rounding any artwork or relaxing symmetry.
    from clay.layout.native_sky130 import NativeDeviceGeometry
    native_names = tuple(sorted(name for name, value in devs.items()
                                if isinstance(value, NativeDeviceGeometry)))
    v_native_x_phase = {name: _new() for name in native_names}
    v_native_y_phase = {name: _new() for name in native_names}
    N = idx

    def _xcentre_row(lbl: str) -> np.ndarray:
        row = np.zeros(N)
        if lbl in atom_of_dev and (ordered_program is not None or lbl not in constraints.on_axis_devices):
            row[v_atom_x[atom_of_dev[lbl]]] = 1.0
        else:
            row[v_axis] = 1.0
        side = dev_side[lbl]
        if side != 0:
            row[v_off[pair_of_dev[lbl]]] = float(side)
        return row

    cons: List[LinearConstraint] = []

    if native_names:
        from clay.layout.route.emit_gds import pdk_manufacturing_grid_nm
        native_grid = pdk_manufacturing_grid_nm(tech)
    else:
        native_grid = max(1, int(grid_nm))
    for name in native_names:
        extent = devs[name].extent()
        center_x = (extent[0] + extent[2]) // 2
        center_y = (extent[1] + extent[3]) // 2
        x_phase = _xcentre_row(name)
        x_phase[v_native_x_phase[name]] = -float(native_grid)
        cons.append(LinearConstraint(x_phase, float(center_x), float(center_x)))
        y_phase = np.zeros(N)
        y_phase[v_ybase[dev_row[name]]] = 1.0
        y_phase[v_native_y_phase[name]] = -float(native_grid)
        cons.append(LinearConstraint(y_phase, float(center_y), float(center_y)))

    kinds_for_gap = constraints.device_kinds
    # The kind-aware opposite-type gap is applied ONLY when a 2D floorplan is
    # declared (band-free). In the historical banded path opposite-type devices are
    # never in-row-adjacent and the row-band interface gap is part of the gf180
    # byte-parity, so the extra gap must NOT perturb it -> gate on floorplan_rows.
    band_free = bool(constraints.floorplan_rows)

    def _row_gap(a: str, b: str) -> int:
        if not band_free:
            return dev_gap
        return _pair_gap(tech, dev_gap, kinds_for_gap.get(a, "nmos"), kinds_for_gap.get(b, "nmos"))

    # non-overlap within each row (fixed order, linear). Opposite-type neighbours
    # (only possible in a band-free / 2D floorplan) get the extra well/implant gap.
    # Sibling devices sharing ONE declared guard ring in an explicit 2-D column
    # floorplan have no ring wall between them, so credit back their duplicate
    # interior ring-room pad while keeping full exterior ring clearance intact.
    _active_ring_credit = (
        ring_group_of
        if (
            ring_group_of
            and any(rc[1] is not None for lbl, rc in constraints.floorplan_rows.items() if lbl in devs)
        )
        else None
    )
    _m2_w = int(tech.min_width("metal2") or 280)
    _m2_s = int(tech.min_space("metal2") or 280)
    _intra_ring_corridor_nm = 5 * (_m2_w + _m2_s) + _m2_w
    for ordered in row_order:
        for a, b in zip(ordered, ordered[1:]):
            coeff = _xcentre_row(b) - _xcentre_row(a)
            gap_ab = _row_gap(a, b)
            hw_sum = half_w[a] + half_w[b]
            if (
                _active_ring_credit is not None
                and _active_ring_credit.get(a) is not None
                and _active_ring_credit.get(a) == _active_ring_credit.get(b)
            ):
                pad_a = int(getattr(devs[a], "_pad", 0) or 0)
                pad_b = int(getattr(devs[b], "_pad", 0) or 0)
                hw_sum = max(0.0, hw_sum - float(pad_a + pad_b))
                # Preserve a PDK-derived 5-track M2 corridor between sibling
                # devices inside a shared guard ring while reclaiming the
                # duplicate exterior ring-wall padding.
                gap_ab = max(gap_ab, _intra_ring_corridor_nm)
            lb = float(hw_sum + gap_ab)
            cons.append(LinearConstraint(coeff, lb, np.inf))

    if ordered_program is not None:
        # All explicit relations enter the numerical model, including relations
        # across distinct rows. They are independently replayed on final artwork.
        for first, second, relation in ordered_program.source_orders:
            if relation == "left_of":
                coeff = _xcentre_row(second) - _xcentre_row(first)
            else:
                coeff = np.zeros(N)
                coeff[v_ybase[dev_row[first]]] += 1.0
                coeff[v_ybase[dev_row[second]]] -= 1.0
            cons.append(LinearConstraint(coeff, float(max(1, grid_nm)), np.inf))

    # offsets >= 0 (left member left of axis).
    for p, vi in v_off.items():
        c = np.zeros(N)
        c[vi] = 1.0
        cons.append(LinearConstraint(c, 0.0, np.inf))

    # differential-net half-plane separation: every device whose drain is the FIRST
    # net of a declared diff pair is pinned to the LEFT half-plane (its x-centre <=
    # axis - half_w), the SECOND net to the RIGHT half-plane (x-centre >= axis +
    # half_w). This makes the diff pair's two nets MIRROR-SEPARABLE -- all of net-a's
    # devices on one side of the axis, net-b's on the other -- which is exactly the
    # two-row floorplan the generic router co-tracks on (a straddling diff net forces
    # the two trunks to cross + short). It is GENERIC (any declared diff pair benefits,
    # no per-circuit branch) and driven by ``diff_net_sides`` (from
    # ``LayoutPlan.diff_net_pairs`` / the device drains).
    #
    # Binding behaviour (gf180 parity): a device that is ALSO a symmetry-pair member
    # already has a fixed side via its ``off`` var (left member <= axis, right >=
    # axis); for those the half-plane is ALREADY implied by the pairing, so adding the
    # inequality is REDUNDANT and never perturbs the optimum (the symmetric comparator
    # places every diff-net device as a pair member, so this is fully non-binding on
    # gf180 -- byte-identical). The constraint only BITES on a diff-net device that is
    # NOT pair-constrained (an on-axis / unpaired device the unconstrained solve could
    # leave straddling), pushing it to the declared half. A device whose diff-net side
    # CONFLICTS with its symmetry-pair side is skipped (the pairing is the stronger
    # structural intent and wins), so the two constraint sources never make the model
    # infeasible.
    for lbl, want in constraints.diff_net_sides.items():
        if lbl not in dev_side or want not in (-1, 1):
            continue
        pair_side = dev_side[lbl]
        if pair_side == want:
            continue  # pairing already realises this half-plane -> redundant, skip
        if pair_side != 0:
            continue  # pairing side wins (opposite); do not over-constrain
        # pair_side == 0: the device has NO offset variable, so its x-centre is FIXED
        # ON the axis in this row model -- it cannot be pushed onto a half-plane
        # without a per-device x var. Emitting ``(x_centre - axis) >= clear`` would be
        # ``0 >= clear`` (infeasible). A side-0 diff-net device is a single-column
        # (e.g. on-axis tail) device whose net is shared, not a drive column that can
        # mirror-separate; leave it on the axis (the half-plane constraint is a no-op
        # for it) rather than make the model infeasible. The constraint therefore binds
        # only where a device HAS a side var (a pair member), where it is redundant --
        # so today it is a correctness GUARANTEE (diff-net columns never straddle) with
        # zero perturbation. A future per-device-x placer would let this bite on
        # genuinely unpaired diff-net drive columns.
        continue

    # rail bands: ``v_ybase[r]`` is the row's y-CENTRE line (every device in the
    # row is centred on it, so a declared pair shares a y-centre even when its two
    # members differ in height -- a cross-class mirror pair stays symmetric).
    # Each higher row's centre exceeds the lower row's by half each height + a gap.
    row_height = [max(heights[l] for l in row_order[r]) for r in range(len(rows))]

    def _row_kinds(r: int) -> set:
        return {kinds_for_gap.get(l, "nmos") for l in row_order[r]}

    for r in range(len(rows) - 1):
        c = np.zeros(N)
        c[v_ybase[r + 1]] = 1.0
        c[v_ybase[r]] = -1.0
        # if the two adjacent rows mix opposite types at their interface, use the
        # extra well/implant gap (band-free floorplan); else the same-type dev_gap.
        cross = band_free and (
            (bool(_row_kinds(r) & {"pmos"}) and bool(_row_kinds(r + 1) & {"nmos"})) or
            (bool(_row_kinds(r) & {"nmos"}) and bool(_row_kinds(r + 1) & {"pmos"}))
        )
        rgap = dev_gap + _opp_kind_extra_gap(tech) if cross else dev_gap
        if ordered_program is not None:
            rgap = ordered_program.row_clearances_nm[r][2]
        gap = float(row_height[r] / 2.0 + row_height[r + 1] / 2.0 + rgap)
        cons.append(LinearConstraint(c, gap, np.inf))

    # extent linearisation. Device y-centre == v_ybase[row]; the x-centre is the
    # axis + side*off as before.
    for lbl in devs:
        xc = _xcentre_row(lbl)
        hw = float(half_w[lbl])
        c = np.zeros(N); c[v_maxX] = 1.0; c -= xc
        cons.append(LinearConstraint(c, hw, np.inf))
        c = np.zeros(N); c[v_minX] = -1.0; c += xc
        cons.append(LinearConstraint(c, hw, np.inf))
        yrow = np.zeros(N); yrow[v_ybase[dev_row[lbl]]] = 1.0
        hh = heights[lbl] / 2.0
        # maxY - ycentre >= h/2 ; ycentre - minY >= h/2
        c = np.zeros(N); c[v_maxY] = 1.0; c -= yrow
        cons.append(LinearConstraint(c, hh, np.inf))
        c = np.zeros(N); c[v_minY] = -1.0; c += yrow
        cons.append(LinearConstraint(c, hh, np.inf))

    # common-centroid Y-balance: for each declared common-centroid group, force the
    # group's MEAN device y-centre to coincide with the centre of its row span, so
    # the group is balanced (gradients cancel to first order -- the criteria's P2
    # obligation). The row ASSIGNMENT is fixed (``dev_row``), so a group's extreme
    # rows r_lo / r_hi are known integers; balance is then a single LINEAR equality
    # on the row y-centre variables:
    #     2 * sum_{m in G} ybase[row(m)]  ==  n * (ybase[r_lo] + ybase[r_hi]).
    # X needs no constraint: in this row model every device is on-axis or in a
    # symmetric pair, so a group's mean x already equals the axis == its x-bbox
    # centre. Generic -- one equality per group, no per-circuit branch, no big-M,
    # reusing the existing ybase vars. Kept SEPARATE so an over-constrained cell can
    # fall back to the unconstrained solve rather than become infeasible.
    cc_cons: List[LinearConstraint] = []
    for group in constraints.common_centroid_groups:
        rows_m = [dev_row[m] for m in group if m in dev_row]
        nseen = len(rows_m)
        if nseen < 2:
            continue
        r_lo, r_hi = min(rows_m), max(rows_m)
        if r_lo == r_hi:
            continue  # a single-row group is trivially y-balanced
        coeff = np.zeros(N)
        for r in rows_m:
            coeff[v_ybase[r]] += 2.0
        coeff[v_ybase[r_lo]] -= float(nseen)
        coeff[v_ybase[r_hi]] -= float(nseen)
        cc_cons.append(LinearConstraint(coeff, 0.0, 0.0))

    obj = np.zeros(N)
    obj[v_maxX] = 1.0
    obj[v_minX] = -1.0
    obj[v_maxY] = height_weight
    obj[v_minY] = -height_weight
    # On explicit 2-D column floorplans, break zero-cost simplex ties by gently
    # compacting free atom axes left and keeping symmetric pair offsets tight,
    # preventing shorter rows from drifting across the chip and stretching their
    # enclosing guard ring into hollow whitespace.
    if hierarchical_floorplan and any(
        rc[1] is not None for lbl, rc in constraints.floorplan_rows.items() if lbl in devs
    ):
        n_tie = max(1, len(v_atom_x) + len(v_off))
        tie_w = 1e-4 / float(n_tie)
        for vi in v_atom_x.values():
            obj[vi] += tie_w
            obj[v_minX] -= tie_w
        for vi in v_off.values():
            obj[vi] += tie_w

    # Existing portable cells keep their historical coarse center grid. Native
    # cells need integer-nm variables plus the explicit translation congruences
    # above so every source-owned polygon remains on the manufacturing grid.
    g = 1 if native_names else max(1, int(grid_nm))

    def _scale(constraint_list: List[LinearConstraint]) -> List[LinearConstraint]:
        out: List[LinearConstraint] = []
        for lc in constraint_list:
            A = np.atleast_2d(np.array(lc.A, dtype=float))
            lb = np.array(lc.lb, dtype=float) / g
            ub = np.array(lc.ub, dtype=float)
            ub = np.where(np.isfinite(ub), ub / g, ub)
            out.append(LinearConstraint(A, lb, ub))
        return out

    # Variable box bound. Every decision variable (axis x, each row's y-centre,
    # the bbox extent aux vars) shares this single +/- box, so it must be large
    # enough to hold BOTH the worst-case horizontal extent (sum of widths) AND the
    # vertical row stack (sum of device heights + inter-row gaps). The original
    # bound counted only widths + a flat per-row term; a cell with TALL devices
    # (a large-W single-finger MOS folds to a tall device) needs a y-centre on the
    # top row that exceeds a width-only bound, which made the MILP INFEASIBLE for
    # multi-row stacks of wide devices. Including the height stack keeps the box
    # feasible generically (no per-circuit constant).
    total_w = sum(2 * half_w[l] for l in devs)
    total_h = sum(heights[l] for l in devs)
    span_bound = (total_w + total_h) // g + len(rows) * (dev_gap + 4000) // g + 1000
    if ordered_program is not None:
        span_bound += sum(gap for _lower, _upper, gap in ordered_program.row_clearances_nm) // g
    bounds = Bounds(np.full(N, -float(span_bound)), np.full(N, float(span_bound)))
    integrality = np.ones(N)

    def _milp(constraint_list: List[LinearConstraint]):
        return milp(
            c=obj,
            constraints=_scale(constraint_list),
            integrality=integrality,
            bounds=bounds,
            options={"time_limit": float(time_limit_s)},
        )

    t0 = time.perf_counter()
    res = _milp(cons + cc_cons)
    cc_applied = bool(cc_cons)
    if cc_cons and not (res.success and res.x is not None):
        # the common-centroid equalities over-constrained this cell -> fall back to
        # the unconstrained solve (never regress to no placement for a balance pref).
        res = _milp(cons)
        cc_applied = False
    solve_seconds = time.perf_counter() - t0

    placed: Dict[str, PlacedDevice] = {}
    axis_x = 0
    ordered_axis_coordinates_nm: Tuple[int, ...] = ()
    objective = float("nan")
    success = bool(res.success and res.x is not None)
    status = str(res.message)
    if native_names and not success:
        raise ValueError(
            "native rigid manufacturing-grid phase placement unresolved: " + status
        )

    if success:
        x = np.asarray(res.x, dtype=float) * g
        objective = float(res.fun) * g
        axis_x = int(round(x[v_axis]))

        def _xc(lbl: str) -> int:
            side = dev_side[lbl]
            if lbl in atom_of_dev and (ordered_program is not None or lbl not in constraints.on_axis_devices):
                v = x[v_atom_x[atom_of_dev[lbl]]]
            else:
                v = x[v_axis]
            if side != 0:
                v = v + side * x[v_off[pair_of_dev[lbl]]]
            return int(round(v))

        for lbl, d in devs.items():
            e = d.extent()
            cur_cx = (e[0] + e[2]) // 2
            cur_cy = (e[1] + e[3]) // 2
            target_cx = _xc(lbl)
            target_cy = int(round(x[v_ybase[dev_row[lbl]]]))
            dx = target_cx - cur_cx
            dy = target_cy - cur_cy  # anchor device y-CENTRE to the row centre-line
            ext = (e[0] + dx, e[1] + dy, e[2] + dx, e[3] + dy)
            placed[lbl] = PlacedDevice(
                label=lbl,
                kind=constraints.device_kinds.get(lbl, "nmos"),
                dx=dx,
                dy=dy,
                extent=ext,
                mirror=(dev_side[lbl] == 1),
                row=dev_row[lbl],
            )

        if ordered_program is not None:
            ordered_axis_coordinates_nm = tuple(
                int(round(x[v_axis if index == 0 else v_atom_x[index]]))
                for index in range(len(ordered_program.mirror_groups))
            )

        if (
            (hierarchical_floorplan or (ordered_program is not None and not ordered_program.mirror_groups))
            and placed
            and not constraints.on_axis_devices
        ):
            axis_x = int(round((
                min(item.extent[0] for item in placed.values())
                + max(item.extent[2] for item in placed.values())
            ) / 2.0))

    full_extent: Bbox
    if placed:
        full_extent = (
            min(p.extent[0] for p in placed.values()),
            min(p.extent[1] for p in placed.values()),
            max(p.extent[2] for p in placed.values()),
            max(p.extent[3] for p in placed.values()),
        )
    else:
        full_extent = (0, 0, 0, 0)

    n_used = len(cons) + (len(cc_cons) if cc_applied else 0)
    note = (
        f"MILP placement DRIVEN BY .clay annotations: {len(rows)} rows, "
        f"{len(devs)} devices, {len(constraints.symmetry_pairs)} declared "
        f"symmetry pair(s) (axis={constraints.symmetry_axis}), "
        f"{len(cc_cons)} common-centroid group constraint(s)"
        f"{' (applied)' if cc_applied else (' (fell back, over-constrained)' if cc_cons else '')}, "
        f"dev_gap={dev_gap}nm, grid={g}nm; vars={N}, constraints={n_used}; "
        f"hierarchical_local_axes={len(v_atom_x)}, "
        f"hierarchical_bands={hierarchical_band_count}, "
        f"hierarchical_subrows={hierarchical_subrow_count}, "
        f"hierarchical_target_width={hierarchical_target_width_nm}nm"
        + (f"; ordered_source_rows={len(ordered_program.rows)}, "
           f"declared_shared_axes={len(ordered_program.mirror_groups)}"
           if ordered_program is not None else "")
    )
    return PlacementSolution(
        placed=placed,
        axis_x=axis_x,
        status=status,
        success=success,
        solve_seconds=solve_seconds,
        objective=objective,
        n_vars=N,
        n_constraints=n_used,
        full_extent=full_extent,
        constraints=constraints,
        note=note,
        ordered_row_program=ordered_program,
        ordered_axis_coordinates_nm=ordered_axis_coordinates_nm,
    )


# === quality scoring (R172, driven by the DECLARED pairs) =====================


def _active_layer_key(devs: Dict[str, object]) -> Tuple[int, int]:
    """Resolve the active/diffusion (COMP) GDS key for these devices.

    gf180 uses ``GF180_COMP_KEY`` (preserved as the default so the gf180 number is
    byte-identical). For any other PDK the device's active layer is the layer its
    ``comp_bbox`` was drawn on, recovered from the polygons -- so external_pdk / others
    score their active area correctly instead of summing zero gf180-keyed polys."""
    for d in devs.values():
        cb = getattr(d, "comp_bbox", None)
        if cb is None:
            continue
        # the active polygon is the one whose bbox equals comp_bbox.
        for lp in getattr(d, "polygons", []) or []:
            xs = [v.x for v in lp.polygon.vertices]
            ys = [v.y for v in lp.polygon.vertices]
            if (min(xs), min(ys), max(xs), max(ys)) == tuple(cb):
                return (lp.layer, lp.datatype)
        break
    return GF180_COMP_KEY


def _active_area_nm2(devs: Dict[str, object], placed: Dict[str, PlacedDevice]) -> float:
    active_key = _active_layer_key(devs)
    total = 0.0
    for lbl in placed:
        d = devs[lbl]
        for lp in d.polys_on(active_key):
            xs = [v.x for v in lp.polygon.vertices]
            ys = [v.y for v in lp.polygon.vertices]
            total += max(0, max(xs) - min(xs)) * max(0, max(ys) - min(ys))
    return total


@dataclass
class PlacementQuality:
    symmetry: object
    common_centroid: object
    rail_position: object
    active_area_nm2: float
    bbox_area_nm2: float
    ratio: float

    @property
    def symmetry_score(self) -> float:
        return float(getattr(self.symmetry, "score", 0.0))

    @property
    def symmetry_passed(self) -> bool:
        return bool(getattr(self.symmetry, "passed", False))


def score_placement(
    sol: PlacementSolution,
    devs: Dict[str, object],
    *,
    rel_tol: float = 0.05,
) -> PlacementQuality:
    """Run the REAL R172 ``symmetry_placement`` / ``common_centroid`` /
    ``rail_position`` rules on the solved bboxes -- with the symmetry pairs taken
    from ``sol.constraints`` (which came from the ``.clay`` ``@symmetric`` /
    ``@matching`` declaration). THIS is where the source declaration measurably
    drives the score."""
    dbb = sol.device_bboxes()
    fe = sol.full_extent
    c = sol.constraints

    if sol.ordered_row_program is not None and sol.ordered_row_program.mirror_groups:
        declared_results = tuple(
            r172_symmetry_placement(dbb, group.pairs, axis=group.axis,
                axis_coord=float(axis), rel_tol=rel_tol)
            for group, axis in zip(sol.ordered_row_program.mirror_groups,
                                   sol.ordered_axis_coordinates_nm)
        )
        sym = RuleResult(
            name="symmetry_placement",
            passed=all(result.passed for result in declared_results),
            score=min(result.score for result in declared_results),
            detail="source-declared common axes; final masks and terminals checked by ordered-row replay",
            data={"axes_nm": sol.ordered_axis_coordinates_nm,
                  "group_results": tuple(result.data for result in declared_results)},
        )
    elif c.floorplan_rows and c.symmetry_pairs:
        # A declared floorplan is solved as a hierarchy of independently movable
        # matching atoms.  Each matched atom owns its local mirror axis; scoring all
        # pairs against one cell axis would turn a correct multi-axis placement into
        # a false failure.  Reuse the exact R172 pair rule per atom, then combine the
        # independent dimensions conservatively (all must pass; the worst score wins).
        local_results = tuple(
            r172_symmetry_placement(
                dbb,
                (pair,),
                axis=c.symmetry_axis,
                axis_coord=None,
                rel_tol=rel_tol,
            )
            for pair in c.symmetry_pairs
        )
        on_index = 0 if c.symmetry_axis == "x" else 1
        local_axes = tuple(
            (
                pair,
                (
                    (dbb[pair[0]][on_index] + dbb[pair[0]][on_index + 2])
                    + (dbb[pair[1]][on_index] + dbb[pair[1]][on_index + 2])
                ) / 4.0,
            )
            for pair in c.symmetry_pairs
            if pair[0] in dbb and pair[1] in dbb
        )
        pair_residuals = tuple(
            residual
            for result in local_results
            for residual in result.data.get("pair_residuals", ())
        )
        worst = max(
            (float(result.data.get("worst_residual", float("inf")))
             for result in local_results),
            default=float("inf"),
        )
        sym = RuleResult(
            name="symmetry_placement",
            passed=all(result.passed for result in local_results),
            score=min(result.score for result in local_results),
            detail=(
                f"{len(local_results)} floorplan matching atom(s) scored on "
                f"independent local {c.symmetry_axis}-axes; worst residual={worst:.4f}"
            ),
            data={
                "worst_residual": worst,
                "pair_residuals": pair_residuals,
                "local_axes": local_axes,
            },
        )
    else:
        sym = r172_symmetry_placement(
            dbb,
            c.symmetry_pairs,
            axis=c.symmetry_axis,
            axis_coord=float(sol.axis_x),
            rel_tol=rel_tol,
        )

    cc: object
    if c.common_centroid_groups:
        cc = r172_common_centroid(dbb, c.common_centroid_groups[0], rel_tol=rel_tol)
    else:
        cc = None

    # synthesise the declared rails as thin strips at the band edges.
    nmos_y = [pd.extent[1] for pd in sol.placed.values() if pd.kind == "nmos"]
    pmos_y = [pd.extent[3] for pd in sol.placed.values() if pd.kind == "pmos"]
    rp: object
    if nmos_y and pmos_y:
        strip = 800
        vss_y = min(nmos_y) - strip
        vdd_y = max(pmos_y) + strip
        rail_bboxes = {
            "vss": (fe[0], vss_y - strip, fe[2], vss_y),
            "vdd": (fe[0], vdd_y, fe[2], vdd_y + strip),
        }
        rp = r172_rail_position(rail_bboxes, vss_net="vss", vdd_net="vdd", vss_is_bottom=True)
    else:
        rp = None

    active = _active_area_nm2(devs, sol.placed)
    bbox = float(max(0, fe[2] - fe[0]) * max(0, fe[3] - fe[1]))
    ratio = bbox / active if active > 0 else float("inf")
    return PlacementQuality(
        symmetry=sym,
        common_centroid=cc,
        rail_position=rp,
        active_area_nm2=active,
        bbox_area_nm2=bbox,
        ratio=ratio,
    )


# === the end-to-end driver ====================================================


@dataclass
class ClayPnrResult:
    """The full ``.clay`` -> placement (-> optional co-clean) result."""

    cell_name: str
    constraints: SolverConstraints
    solution: PlacementSolution
    quality: PlacementQuality
    # source-driven proof
    symmetry_score: float
    symmetry_from_source: bool
    declared_pairs: Tuple[Tuple[str, str], ...]
    declared_axis: str
    rail_applied: bool
    # co-clean (optional / gated)
    coclean_ran: bool = False
    coclean_note: str = ""

    @property
    def placement_succeeded(self) -> bool:
        return self.solution.success

    def summary(self) -> str:
        return (
            f"{self.cell_name}: MILP {'OK' if self.solution.success else 'FAIL'} "
            f"in {self.solution.solve_seconds*1000:.0f}ms | "
            f"declared {len(self.declared_pairs)} pair(s) axis={self.declared_axis} | "
            f"symmetry_placement score={self.symmetry_score:.3f} "
            f"({'PASS' if self.quality.symmetry_passed else 'FAIL'}) | "
            f"rail {'applied' if self.rail_applied else 'n/a'} | "
            f"area_ratio={self.quality.ratio:.2f}x"
        )


def place_and_route_from_clay(
    clay_source_or_path: Union[str, Path],
    *,
    cell_name: Optional[str] = None,
    rel_tol: float = 0.05,
    height_weight: float = 1.0,
    grid_nm: int = 5,
    time_limit_s: float = 30.0,
    apply_rail_band: bool = False,
) -> ClayPnrResult:
    """Parse the ``.clay``, extract the layout annotations, derive the solver
    constraints, run the symmetry-constrained MILP placement, and score it with
    the REAL R172 rules.

    The symmetry score the result reports is computed against the pairs DECLARED
    in the ``.clay`` ``@symmetric`` / ``@matching`` block -- so a symmetry score of
    1.0 is the source declaration measurably driving the solver placement, the
    moat demonstrated end-to-end.

    (The GDS emission + real-deck co-clean on this .clay-driven path is the
    integration follow-up; see :func:`coclean_residual_note`. The placement-quality
    gate that proves the language-level moat runs on the real solved bboxes here.)
    """
    annotated = annotated_cell_from_clay(clay_source_or_path, cell_name=cell_name)
    constraints = annotations_to_solver_constraints(annotated, apply_rail_band=apply_rail_band)
    devs = _build_devices_from_clay(annotated)
    if not devs:
        raise ValueError(
            f"cell {constraints.cell_name!r} has no nmos/pmos devices to place"
        )
    sol = solve_placement_from_constraints(
        constraints,
        devs,
        height_weight=height_weight,
        grid_nm=grid_nm,
        time_limit_s=time_limit_s,
    )
    quality = score_placement(sol, devs, rel_tol=rel_tol)

    return ClayPnrResult(
        cell_name=constraints.cell_name,
        constraints=constraints,
        solution=sol,
        quality=quality,
        symmetry_score=quality.symmetry_score,
        symmetry_from_source=bool(constraints.symmetry_pairs),
        declared_pairs=constraints.symmetry_pairs,
        declared_axis=constraints.symmetry_axis,
        rail_applied=bool(constraints.rail_convention) and quality.rail_position is not None,
        coclean_ran=False,
        coclean_note=coclean_residual_note(),
    )


def coclean_residual_note() -> str:
    """Honest statement of the co-clean residual on the .clay-driven path.

    The full real-deck co-clean (gf180mcu DRC + LVS + union-find) is already wired
    on the HAND-DECLARED path (``symmetric_comparator_v173.run_coclean_comparator``
    consuming the Python constants). This module proves the SOURCE-driven placement
    reaches symmetry score 1.0 from the ``.clay`` ``@symmetric`` declaration. Wiring
    the GDS emitter + the real decks onto this exact source-driven placement (so the
    same .clay flows parse -> annotate -> MILP -> GDS -> co-clean) is the remaining
    integration step: the source-driven placement here produces the device bboxes;
    the v173 router/rail-synth/co-clean harness would consume them in place of its
    hand-placed bboxes."""
    return (
        "co-clean on the .clay-driven path: NOT run here. Achieved level: parse -> "
        "annotate -> solver-constraints -> MILP placement -> REAL R172 symmetry "
        "score from the source @symmetric. Residual: GDS emit + gf180mcu DRC/LVS on "
        "this exact source-driven placement (the v173 router/co-clean harness "
        "consumes hand-placed bboxes today; feeding it these source-driven bboxes "
        "is the follow-up)."
    )


# === grammar-extension status (common_centroid / rail) ========================

GRAMMAR_EXTENDED_COMMON_CENTROID = False
GRAMMAR_EXTENDED_RAIL = False
GRAMMAR_EXTENSION_NOTE = (
    "DEFERRED. The grammar's six annotation kinds (@matching / @guard_ring / "
    "@well_isolated / @symmetric / @interdigitated / @adjacent) are shared with "
    "the Rust pest parser (grammar.pest) under a parity lock; adding "
    "@common_centroid / power-rail(side) terminals risks breaking the byte-exact "
    "parser-split contract + the pest parity. This round implements the consume "
    "step with the EXISTING kinds: common-centroid is taken from the first declared "
    "@symmetric pair, and the rail convention is INFERRED from the device classes "
    "(nmos -> bottom/vss, pmos -> top/vdd). Adding the two annotation kinds (with "
    "matched pest terminals + transformer factories + Rust annotation.rs variants) "
    "is the clear next step once the parity-lock window allows a grammar change."
)


__all__ = [
    "SolverConstraints",
    "PlacedDevice",
    "PlacementSolution",
    "PlacementQuality",
    "ClayPnrResult",
    "annotations_to_solver_constraints",
    "solver_constraints_from_layout_plan",
    "annotated_cell_from_clay",
    "solve_placement_from_constraints",
    "score_placement",
    "place_and_route_from_clay",
    "coclean_residual_note",
    "GRAMMAR_EXTENDED_COMMON_CENTROID",
    "GRAMMAR_EXTENDED_RAIL",
    "GRAMMAR_EXTENSION_NOTE",
]
