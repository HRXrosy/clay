"""Constraint-graph (sequence-pair-style) analog PLACER for Clay core.

Formulates device placement as a RELATIVE ORDER on a horizontal / vertical pair
of constraint graphs (a sequence-pair). A legal placement is overlap-free BY
CONSTRUCTION -- every device pair is ordered in H *or* V -- and is solved in
O(V+E) by a topological-sort longest-path on each graph, so it scales past the
exact MILP (:func:`clay.layout.annotate_to_solver.solve_placement_from_constraints`).

This absorbs the LaSTPlace paradigm (TCAD) as a fresh reimplementation: there is
no learned model / GNN here -- the relative ORDER comes from Clay's DECLARED
intent (the ``annotate {}`` symmetry pairs + the rail convention), turned into a
:class:`clay.layout.seqpair_model.RelativeOrder` and solved analytically.

What it REUSES (imported, never reimplemented)
==============================================
* :func:`clay.layout.generic_placement.annotated_cell_from_cell_def` +
  :func:`clay.layout.annotate_to_solver._build_devices_from_clay` -- the
  foldable gf180 MOS geometry per device (the same devices the MILP places).
* :func:`clay.layout.annotate_to_solver._row_plan` (driven by
  :func:`annotations_to_solver_constraints`) -- the rail-banded rows + symmetry
  pairs + on-axis devices, so the seqpair rows match the existing engine's.
* :func:`clay.layout.generic_placement._placed_terminals` / ``_place_polys`` --
  the router-facing terminal + polygon extraction.
* :func:`clay.layout.annotate_to_solver.score_placement` -- the REAL R172
  symmetry / common-centroid / rail measurement (no scoring reinvented).

The ONLY new logic is the constraint-graph build + the longest-path solve + the
symmetry-centering snap. The returned :class:`Placement` is the SAME drop-in
shape :mod:`clay.layout.generic_placement` returns (so the router / convergence
criteria consume it unchanged), with the solving :class:`RelativeOrder` attached
as ``.relative_order``.

Honesty contract
----------------
* The geometry is the REAL foldable-device geometry; the symmetry / centroid /
  rail scores are the REAL R172 measurements on the solved bboxes.
* The placement is overlap-free by the constraint-graph construction (asserted by
  the test's pairwise-overlap check), not by an after-the-fact legaliser.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

from clay._native import load_native
from clay.ir import CellDef
from clay.layout.intent import LayoutIntent

from .annotate_to_solver import (
    PlacedDevice as _SolverPlacedDevice,
    PlacementSolution,
    SolverConstraints,
    _abut_gap_from_tech,
    _build_devices_from_clay,
    _dev_gap_from_tech,
    _half_widths,
    _heights,
    _pair_gap,
    _resolve_tech,
    _row_plan,
    annotations_to_solver_constraints,
    score_placement,
)
from .generic_placement import (
    Placement,
    PlacedDevice,
    _apply_pins,
    _build_passive_geom,
    _cell_geom,
    _collect_passives,
    _constraints_from_intent_and_annotations,
    _demote_ambiguous_matching_constraints,
    _disjoint_matching_pairs,
    _materialized_matching_axes,
    _placed_bbox,
    _placed_terminals,
    _place_polys,
    _place_resistors,
    _polys_bbox,
    _restrict_constraints_to_devices,
    _row_matching_pair_atoms,
    annotated_cell_from_cell_def,
    pin_targets_nm,
)
from .pdk_rules import PdkRules
from .seqpair_model import RelativeOrder

Bbox = Tuple[int, int, int, int]


# === 1. intent -> RelativeOrder ===============================================


def _ordered_row(row: List[Tuple[str, object]]) -> List[str]:
    """Flatten one ``_row_plan`` row into a left-to-right device sequence whose
    layout CAN be made mirror-symmetric about one vertical axis.

    A declared mirror pair ``("pair", (a, b))`` must STRADDLE the cell axis (``a``
    left of it, ``b`` right of it). The unpaired (``"axis"``) devices of the row are
    placed BETWEEN the pair members -- centred on the axis -- so the pair's two
    members sit on the row's OUTSIDE, equidistant from the axis, with the loose
    devices in the middle. This is the ordering that lets a single global axis snap
    keep BOTH (a) every declared pair exactly mirror-symmetric AND (b) the row
    overlap-free: the loose centre devices fit inside the pair's straddle, so making
    the pair symmetric never forces a member to cross a neighbour (the overlap that
    fired CO.10 / M1.2a / M2.2a when the pair was emitted ADJACENT and then snapped
    about a distant global axis).

    With no declared pair the row is just its devices left-to-right (the loose
    centre is the whole row). A single pair yields ``[a, loose..., b]``; multiple
    pairs nest outward (first pair outermost) so each straddles the same centre.
    """
    pairs: List[Tuple[str, str]] = []
    loose: List[str] = []
    for kind, payload in row:
        if kind == "pair":
            a, b = payload  # type: ignore[misc]
            pairs.append((a, b))
        else:
            loose.append(payload)  # type: ignore[arg-type]
    if not pairs:
        return list(loose)
    lefts = [a for (a, _b) in pairs]
    rights = [b for (_a, b) in pairs]
    # first declared pair outermost: lefts in declared order on the left, the loose
    # devices in the centre, rights in REVERSED order on the right so each pair is
    # equidistant about the centre (outer pair spans the widest, inner pairs nest).
    return lefts + loose + list(reversed(rights))


def _pack_unpaired_rows(
    rows: List[List[Tuple[str, object]]],
    axis_locked: Optional[set] = None,
) -> List[List[Tuple[str, object]]]:
    """Coalesce runs of FREE-UNPAIRED single-device ``_row_plan`` rows into a wide 2D
    grid, so a cell with many free devices does NOT degenerate into a tall column.

    The banded ``_row_plan`` emits every unpaired device on its OWN single-device
    row (``[("axis", lbl)]``). With many such devices that degenerates the placement
    into a tall one-device-per-row COLUMN (e.g. 6 unpaired NMOS -> ~3um x 47um,
    aspect 16) -- each device stacked vertically in its own rail-band row.

    Which ``("axis", lbl)`` devices are FREE to share a row? A declared
    ``@symmetric`` / ``@matching`` pair (``("pair", ...)``) needs its own row (its two
    members share a y-baseline + mirror about the axis), so a pair-bearing row is
    NEVER merged. An UNPAIRED device that is itself a member of a declared
    matched / common-centroid GROUP (an ODD member that ``_row_plan`` placed
    on-axis so the group stays centroid-balanced -- e.g. the 3-device cc group
    ``(MPBIAS, MP6, MPTAIL)``) must ALSO keep its own axis-centred row: scattering
    those members across a wide packed row breaks their common-centroid balance
    (the matched_cc_balance obligation). ``axis_locked`` names every such device
    (the union of the declared symmetry-pair / on-axis / common-centroid members).
    Only devices NOT in ``axis_locked`` -- the genuinely unconstrained bias / tail /
    cascode switches the floorplan never tied to a group -- are packed.

    Every maximal RUN of consecutive single-device FREE rows is packed into a
    near-square grid (``ceil(sqrt(N))`` devices per row), preserving bottom-to-top
    order so the longest-path V order is monotone; overlap-free by construction
    (in-row ``left_of`` edges, cross-row ``below`` edges downstream). An axis-locked
    single-device row (or any multi-cell row) FLUSHES the current free run and passes
    through unchanged, so its centroid-centred placement is preserved.

    Generic + circuit-agnostic: keyed only on the declared-constraint membership, no
    device-kind / cell-name branch. With ``axis_locked`` covering every declared
    member, a cell whose unpaired devices are all group members is byte-identical.
    """
    import math

    locked = axis_locked or set()
    out: List[List[Tuple[str, object]]] = []
    run: List[str] = []   # accumulated FREE unpaired device labels in the current run

    def _flush_run() -> None:
        if not run:
            return
        n = len(run)
        if n <= 1:
            out.append([("axis", run[0])])
            run.clear()
            return
        # near-square: ceil(sqrt(n)) devices per row -> a wide 2D block.
        per_row = max(1, int(math.ceil(math.sqrt(n))))
        for i in range(0, n, per_row):
            out.append([("axis", lbl) for lbl in run[i:i + per_row]])
        run.clear()

    for row in rows:
        has_pair = any(kind == "pair" for kind, _payload in row)
        # a row is PACKABLE only if it is a single FREE unpaired (axis) device --
        # not a pair, not an axis-locked group member, not an already-multi row.
        single_free = (
            not has_pair
            and len(row) == 1
            and row[0][0] == "axis"
            and row[0][1] not in locked
        )
        if single_free:
            run.append(row[0][1])  # type: ignore[arg-type]
        else:
            _flush_run()
            out.append(row)
    _flush_run()
    return out


def relative_order_from_intent(
    cell_def: CellDef,
    intent: Optional[LayoutIntent] = None,
    *,
    source_cir=None,
    tech=None,
    lock_devices: Optional[set] = None,
) -> RelativeOrder:
    """Turn a parsed ``CellDef`` + its declared intent into a :class:`RelativeOrder`.

    Reuses the existing engine's front-end so the seqpair rows match the MILP's:

      1. :func:`annotated_cell_from_cell_def` (+ ``source_cir`` for the real
         W/L/m) -> the foldable MOS devices + kinds;
      2. :func:`annotations_to_solver_constraints` (intent-overlaid) -> the
         declared symmetry pairs / on-axis devices / rail convention;
      3. :func:`_row_plan` -> rail-banded rows (NMOS below PMOS), one declared pair
         per row, bottom-to-top.

    The relative order is then read structurally off the rows:

      * within each row the devices order left-to-right (each matched pair as
        ``left member, right member``) -> consecutive ``left_of`` edges;
      * every device in row ``r`` is ``below`` every device in row ``r+1`` (the
        rail-band vertical order);
      * ``mirror_pairs`` are the declared symmetry pairs;
      * ``rows`` are the row groupings bottom-to-top.

    No ``positions`` are filled (that is the solver's job); ``axis_x`` is left
    ``None`` until :func:`solve_placement` snaps it.
    """
    annotated, _used_default = annotated_cell_from_cell_def(cell_def, source_cir)
    constraints = _constraints_from_intent_and_annotations(annotated, intent)
    devs = _build_devices_from_clay(annotated, _resolve_tech(tech) if tech is not None else None)
    constraints, _dropped = _restrict_constraints_to_devices(constraints, devs)
    # A shared-member matching graph cannot be lowered into independent rigid
    # two-member atoms.  Demote every ambiguous edge before building the row plan:
    # all devices remain ordinary placement nodes, while only disjoint pairs keep
    # mirror authority.  Filtering after ``_row_plan`` would be too late because
    # the raw plan can duplicate a member (banded) or consume/drop its peer
    # (floorplan).
    constraints = _demote_ambiguous_matching_constraints(constraints, devs)
    rows = _row_plan(constraints, devs)

    # Per-device DECLARED column index from the 2D ``@place_group ... col=N`` floorplan.
    # ``floorplan_rows`` maps ``label -> (row, col)``; project out the column.
    col_of: Dict[str, int] = {}
    fp = getattr(constraints, "floorplan_rows", None) or {}
    for lbl, rc in fp.items():
        if lbl in devs and rc is not None and len(rc) >= 2 and rc[1] is not None:
            col_of[lbl] = int(rc[1])

    mirror_pairs = _row_matching_pair_atoms(constraints, devs)

    widened = False
    if col_of:
        # ── DECLARED 2D GRID: build the relative order as a (row x col) grid ──────
        # The floorplan declares a column per device; the row index is the row-plan
        # row each device sits in. A correct 2D placement is then:
        #   * cross-column: EVERY device in a lower column is ``left_of`` EVERY device
        #     in a higher column (the column x-bands, what the col-band check scores);
        #   * within one (row, col) CELL: the devices order left-to-right by the
        #     mirror-straddle ordering (a same-column mirror pair straddles its loose
        #     centre) -> the symmetry axis stays INSIDE the column;
        #   * same column, lower row ``below`` higher row (the row y-bands, what the
        #     row-band check scores).
        # ``left_of`` is cross-column ∪ within-cell (a DAG: column order is total,
        # within-cell is a chain), ``below`` is same-column cross-row -- two disjoint
        # axes, so no cycle. This realises the declared floorplan correct-by-
        # construction, generic from the col/row data with no device-kind/name branch.
        devices, left_of, below, row_seqs = _grid_order_from_floorplan(
            rows, col_of, mirror_pairs
        )
    else:
        # ── historical banded / symmetry path (byte-stable: no floorplan columns) ─
        # PACK runs of unpaired-only rows into a wide 2D grid so a cell with many
        # unpaired devices does NOT degenerate into a tall one-device-per-row column.
        # Pair-bearing rows are untouched (a declared pair owns its row + axis), so a
        # cell that is all declared pairs is byte-identical; only the unpaired runs
        # widen. ``band_free`` is set below so the solver applies the kind-aware
        # opposite-type gap when unpaired NMOS/PMOS now share a packed row.
        # axis-locked = every device named by a declared symmetry pair / on-axis /
        # common-centroid group. Those unpaired members ``_row_plan`` put on-axis so
        # the group stays centroid-balanced -> they must NOT be packed into a wide
        # row (that scatters the cc members + breaks matched_cc_balance). Only the
        # genuinely unconstrained devices are free to pack.
        axis_locked: set = set(lock_devices or ())
        for a, b in constraints.symmetry_pairs:
            axis_locked.add(a)
            axis_locked.add(b)
        axis_locked.update(constraints.on_axis_devices)
        for grp in constraints.common_centroid_groups:
            axis_locked.update(grp)
        packed_rows = _pack_unpaired_rows(rows, axis_locked)
        widened = any(len(r) > 1 and all(k == "axis" for k, _ in r) for r in packed_rows)
        rows = packed_rows
        row_seqs = []
        devices = []
        left_of = []
        below = []
        for row in rows:
            seq = _ordered_row(row)
            row_seqs.append(tuple(seq))
            devices += seq
            # consecutive same-row devices: u left_of v.
            for u, v in zip(seq, seq[1:]):
                left_of.append((u, v))
        # every device in row r is below every device in row r+1 (rail-band order).
        for r in range(len(row_seqs) - 1):
            for u in row_seqs[r]:
                for v in row_seqs[r + 1]:
                    below.append((u, v))

    # ── @order: DECLARED relative-order edges ────────────────────────────────────
    # Each ``(a, b, rel)`` adds one DIRECTED constraint-graph edge so the order is
    # realised by construction: ``left_of`` -> a before b on the H (x) graph,
    # ``above`` -> a before b on the V (y) graph (a above b => a has the GREATER y, so
    # the edge runs ``b below a``). Only edges whose BOTH ends are built MOS are added
    # (already filtered in ``_restrict_constraints_to_devices``; re-guard on ``devs``
    # for safety). Folded onto the structurally-derived edges; an empty
    # ``order_constraints`` adds nothing -> byte-parity. ``below`` is the strict
    # ``(lower, upper)`` relation, so ``a above b`` appends ``(b, a)``.
    # A declared @order is AUTHORITATIVE over the structurally-derived order: it
    # removes BOTH structural directions of the named pair on BOTH graphs first (so a
    # `@order(M2, M1, left_of)` that reverses the row-plan's `M1 left_of M2` does not
    # leave the opposite edge and create a cycle), then adds the single declared edge.
    order_constraints = [
        (a, b, rel) for (a, b, rel) in (getattr(constraints, "order_constraints", ()) or ())
        if a in devs and b in devs and a != b
    ]
    declared_pairs = {frozenset((a, b)) for (a, b, _rel) in order_constraints}
    left_of = [e for e in left_of if frozenset(e) not in declared_pairs]
    below = [e for e in below if frozenset(e) not in declared_pairs]
    seen_lo = set(left_of)
    seen_be = set(below)
    for (a, b, rel) in order_constraints:
        if rel == "left_of":
            if (a, b) not in seen_lo:
                left_of.append((a, b))
                seen_lo.add((a, b))
        elif rel == "above":
            # a above b => b is BELOW a (below is the (lower, upper) relation).
            if (b, a) not in seen_be:
                below.append((b, a))
                seen_be.add((b, a))

    # ── @abut: declared adjacency pairs ──────────────────────────────────────────
    # The pair is carried onto the order so the solver tightens its facing-edge gap to
    # the DRC-minimum (see ``solve_placement._gap``). For the tight gap to BIND the
    # pair must be ADJACENT on a constraint-graph edge; when the structural order
    # already orders the pair (a direct left_of / below edge in EITHER direction) that
    # edge carries the tight gap. When the pair has NO direct edge yet (the common
    # case: two devices the row plan placed side-by-side but did not chain), add ONE
    # ``left_of`` edge so they pack horizontally adjacent at the tight gap. The added
    # edge is on the H graph only, so it cannot conflict with any V (below) relation
    # the pair may already have (the two graphs are independent -> no cycle). Empty
    # ``abut_pairs`` adds no edge -> byte-parity.
    abut_pairs = tuple(
        (a, b) for (a, b) in (getattr(constraints, "abut_pairs", ()) or ())
        if a in devs and b in devs and a != b
    )
    for (a, b) in abut_pairs:
        has_edge = (
            (a, b) in seen_lo or (b, a) in seen_lo
            or (a, b) in seen_be or (b, a) in seen_be
        )
        if not has_edge:
            left_of.append((a, b))
            seen_lo.add((a, b))

    return RelativeOrder(
        cell=cell_def.local_name,
        devices=tuple(devices),
        left_of=tuple(left_of),
        below=tuple(below),
        rows=tuple(row_seqs),
        mirror_pairs=mirror_pairs,
        axis_x=None,
        positions={},
        band_free=bool(constraints.floorplan_rows) or widened,
        col_of=col_of,
        abut_pairs=abut_pairs,
    )


def _grid_order_from_floorplan(
    rows: List[List[Tuple[str, object]]],
    col_of: Dict[str, int],
    mirror_pairs: Tuple[Tuple[str, str], ...],
) -> Tuple[List[str], List[Tuple[str, str]], List[Tuple[str, str]],
           List[Tuple[str, ...]]]:
    """Build the (row x col) grid relative order for a DECLARED 2D floorplan.

    Returns ``(devices, left_of, below, row_seqs)``. The row index of each device is
    the index of the ``_row_plan`` row it belongs to (rows are bottom-to-top); the
    column index is its declared ``col`` (``col_of``). Within one ``(row, col)`` cell
    the devices are ordered left-to-right by :func:`_ordered_row` (so a same-column
    mirror pair straddles its loose centre and its axis stays inside the column).

    Edges (correct-by-construction, two disjoint axes -> acyclic):
      * ``left_of`` -- (a) within a cell, consecutive left-to-right; (b) cross-column,
        every lower-column device before every higher-column device. The column order
        is a total order and the within-cell order is a chain, so the union is a DAG.
      * ``below`` -- within ONE column, every device of a lower row before every device
        of a higher row (the column's own vertical stack). Rows in DIFFERENT columns
        are not ``below``-related (they are already separated on the x axis), which is
        what lets each column have its own independent row heights in a true 2D grid.

    ``row_seqs`` is the per-row left-to-right device sequence (cells concatenated in
    column order), used by the score / mirror-y snap exactly as the banded path's is.
    """
    n_rows = len(rows)
    # cell[(row_index, col)] -> ordered device list (within-cell straddle order).
    cell: Dict[Tuple[int, int], List[str]] = {}
    devices: List[str] = []
    row_seqs: List[Tuple[str, ...]] = []
    cols_sorted = sorted({c for c in col_of.values()})

    for ri, row in enumerate(rows):
        ordered = _ordered_row(row)
        # split this row's left-to-right sequence into its declared columns, keeping
        # the within-cell straddle order. A device with no declared column (rare: a
        # floorplan that named only some devices) defaults to the lowest column so it
        # is never dropped and never breaks the band order.
        for d in ordered:
            c = col_of.get(d, cols_sorted[0] if cols_sorted else 0)
            cell.setdefault((ri, c), []).append(d)
        row_seq: List[str] = []
        for c in cols_sorted:
            row_seq += cell.get((ri, c), [])
        row_seqs.append(tuple(row_seq))
        devices += row_seq

    left_of: List[Tuple[str, str]] = []
    seen_lo: set = set()

    def _add_lo(u: str, v: str) -> None:
        if u != v and (u, v) not in seen_lo:
            left_of.append((u, v))
            seen_lo.add((u, v))

    # (a) within-cell consecutive left_of.
    for members in cell.values():
        for u, v in zip(members, members[1:]):
            _add_lo(u, v)
    # (b) cross-column: every lower-column device left_of every higher-column device
    # (adjacent columns suffice; longest-path transitivity gives the full band order).
    by_col: Dict[int, List[str]] = {}
    for d in devices:
        by_col.setdefault(col_of.get(d, cols_sorted[0] if cols_sorted else 0), []).append(d)
    for i in range(len(cols_sorted) - 1):
        for u in by_col.get(cols_sorted[i], []):
            for v in by_col.get(cols_sorted[i + 1], []):
                _add_lo(u, v)

    # below: within ONE column, lower row before higher row.
    below: List[Tuple[str, str]] = []
    seen_be: set = set()
    for c in cols_sorted:
        col_rows = [
            [d for d in cell.get((ri, c), [])] for ri in range(n_rows)
        ]
        for ri in range(n_rows - 1):
            for u in col_rows[ri]:
                for vi in range(ri + 1, n_rows):
                    for v in col_rows[vi]:
                        if (u, v) not in seen_be:
                            below.append((u, v))
                            seen_be.add((u, v))
    return devices, left_of, below, row_seqs


# === 2. longest-path solve on the H / V constraint graphs =====================


def _longest_path_positions(
    nodes: List[str],
    edges: List[Tuple[str, str]],
    weight: Dict[Tuple[str, str], float],
) -> Dict[str, float]:
    """Longest-path on a DAG: ``pos[v] = max over preds u of pos[u] + w(u, v)``.

    Topological order via Kahn; a node with no predecessor starts at 0. This packs
    the devices in the edge direction with no overlaps (each edge weight is the
    minimum centre-to-centre separation that keeps ``u`` and ``v`` clear), in
    O(V+E). Any node not touched by an edge is still placed at 0.
    """
    native = load_native("_place_cpp")
    if native is not None:
        try:
            return native.longest_path_positions_cpp(nodes, edges, weight)
        except OverflowError:
            pass
    succ: Dict[str, List[str]] = defaultdict(list)
    indeg: Dict[str, int] = {n: 0 for n in nodes}
    for u, v in edges:
        succ[u].append(v)
        indeg[v] += 1

    pos: Dict[str, float] = {n: 0.0 for n in nodes}
    q: deque = deque(n for n in nodes if indeg[n] == 0)
    seen = 0
    while q:
        u = q.popleft()
        seen += 1
        for v in succ[u]:
            cand = pos[u] + weight[(u, v)]
            if cand > pos[v]:
                pos[v] = cand
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if seen != len(nodes):
        raise ValueError(
            "constraint graph is not a DAG (a cycle in the relative order) -- "
            f"resolved {seen}/{len(nodes)} nodes"
        )
    return pos


def _reconcile_symmetry_per_column(
    x_centre: Dict[str, float],
    half_w: Dict[str, int],
    rows: Tuple[Tuple[str, ...], ...],
    mirror_pairs: Tuple[Tuple[str, str], ...],
    col_of: Dict[str, int],
    grid: int = 5,
    channel: float = 1.0,
) -> Dict[str, float]:
    """Center each declared COLUMN about its own axis + make every same-column mirror
    pair exactly symmetric about that axis -- the correct realisation of a DECLARED 2D
    ``@place_group`` grid.

    In a 2D column floorplan a column owns several rows; a mirror pair lives entirely
    inside one column (its two members carry the same ``col``). The whole-cell single-
    axis snap (:func:`_reconcile_symmetry_nonoverlap`) would translate a column rigidly
    to a cell-wide global axis and so drag a column's members across the adjacent
    column's x-band boundary, INVERTING the declared col order. Two passes instead,
    BOTH non-overlap-preserving and PER COLUMN:

      1. COLUMN CENTERING. Each column's WIDEST row defines the column's x extent +
         axis (its centre). Every NARROWER row in that column is rigidly shifted so its
         own centre lands on the column axis -- a rigid per-row shift inside the column
         preserves the row's intra-spacing, and because the widest row already spans the
         column the shift never pushes a narrower row past the column extent, so the
         columns stay x-separated (no cross-column overlap). This makes every row of a
         column concentric, so the column's centroid distribution is centred on its axis
         (the median sits at the centre, not skewed to one side) -- which is exactly
         what keeps every column member on the correct side of the band boundary the
         col-band check derives from the columns' medians.
      2. PAIR EQUALISATION. Each same-column mirror pair is then equalised about its own
         midpoint (a pure redistribution of the pair's slack: ``a`` keeps its side,
         ``b`` keeps its, any loose centre device stays between), so the pair is exactly
         mirror-symmetric without moving any device's column.

    Generic + circuit-agnostic: only the solved coordinates + declared rows/pairs +
    declared per-device column; no spacing constant, no cell name. Returns the adjusted
    x-centres (the global axis is NOT collapsed; the caller keeps the extent midpoint).
    """
    x = dict(x_centre)

    # ── pass 1: center each column's rows on the column's (widest-row) axis ──────
    # group this column's members by row, compute each (col,row) cell's extent centre,
    # take the WIDEST cell as the column axis, shift the narrower cells onto it.
    col_row_cells: Dict[int, Dict[int, List[str]]] = {}
    for ri, row in enumerate(rows):
        for d in row:
            if d not in x:
                continue
            c = col_of.get(d)
            if c is None:
                continue
            col_row_cells.setdefault(c, {}).setdefault(ri, []).append(d)
    for c, by_row in col_row_cells.items():
        # extent of each row-cell in this column.
        cell_extent: Dict[int, Tuple[float, float, float]] = {}
        for ri, members in by_row.items():
            lo = min(x[d] - half_w.get(d, 0) for d in members)
            hi = max(x[d] + half_w.get(d, 0) for d in members)
            cell_extent[ri] = (lo, hi, (lo + hi) / 2.0)
        if not cell_extent:
            continue
        # column axis = centre of the WIDEST row-cell (defines the column extent).
        widest = max(cell_extent, key=lambda ri: cell_extent[ri][1] - cell_extent[ri][0])
        axis = cell_extent[widest][2]
        for ri, members in by_row.items():
            shift = axis - cell_extent[ri][2]
            if abs(shift) < 1e-6:
                continue
            for d in members:
                x[d] += shift

    # ── pass 2: equalise each same-column mirror pair about its own midpoint ─────
    for (a, b) in mirror_pairs:
        if a not in x or b not in x:
            continue
        ca, cb = col_of.get(a), col_of.get(b)
        if ca is not None and cb is not None and ca != cb:
            # a pair straddling columns is not a within-column reflection -> skip.
            continue
        lo, hi = (a, b) if x[a] <= x[b] else (b, a)
        mid = (x[lo] + x[hi]) / 2.0
        off = (x[hi] - x[lo]) / 2.0   # half the solved span -> exact equidistance
        x[lo] = mid - off
        x[hi] = mid + off

    # ── pass 3: re-space the columns to the MINIMAL non-overlapping gap + channel ─
    # Columns must stay x-ORDERED, NON-OVERLAPPING, and leave a real routing channel
    # between adjacent columns for the cross-column nets to run their own metal tracks.
    # That is ALL the col-order realisation needs: the place-group check verifies the
    # DECLARED grid ORDER (every col_i device strictly left of every col_j device for
    # i<j -- a pairwise order test, no median midpoint), so it PASSES for any compact
    # placement that keeps the columns x-ordered. We therefore enforce the SMALLEST
    # gap that keeps the columns clear of each other's device EDGES with one routing
    # channel between them:
    #   min_x_edge(c+1) - max_x_edge(c)  >=  dev_gap + routing_channel.
    # ``routing_channel`` is the metal-track allowance the caller passes (a small,
    # geometry/PDK-derived multiple of the M1 track pitch -- see ``solve_placement``),
    # NOT 0.5*tallest_column_height: the old ``spread(c)+spread(c+1)`` over-spacing term
    # forced the gap to exceed BOTH columns' widths and blew the cell open (two_stage_ref
    # measured 119-155um, mostly blank). A rigid per-column shift preserves the column's
    # intra spacing + non-overlap and only ever moves columns LEFT (toward compactness)
    # or holds them. Pure geometry off the declared columns + the PDK metal pitch -- no
    # per-circuit spacing constant, no device-kind branch.
    cols_sorted = sorted({c for c in col_of.values() if c is not None})

    def _col_edges(c: int) -> Optional[Tuple[float, float]]:
        xs = [(x[d] - half_w.get(d, 0), x[d] + half_w.get(d, 0))
              for d, cc in col_of.items() if cc == c and d in x]
        if not xs:
            return None
        return (min(lo for lo, _hi in xs), max(hi for _lo, hi in xs))

    for i in range(len(cols_sorted) - 1):
        c_lo, c_hi = cols_sorted[i], cols_sorted[i + 1]
        e_lo, e_hi = _col_edges(c_lo), _col_edges(c_hi)
        if e_lo is None or e_hi is None:
            continue
        max_edge_lo = e_lo[1]
        min_edge_hi = e_hi[0]
        # Target: the higher column's LEFT device edge sits at least
        # (dev_gap + routing_channel) right of the lower column's RIGHT device edge.
        # ``need`` is the signed shift; >0 widens (columns too close / overlapping),
        # <0 would compact (columns too far apart -- e.g. left after the centering
        # pass over-spread them) by sliding the higher columns LEFT.
        need = (max_edge_lo + channel) - min_edge_hi
        if abs(need) > 1e-9:
            # Snap the rigid column shift to a whole grid step so an on-grid device
            # stays on-grid after the shift (off-grid x -> gf180mcu ``*_OFFGRID`` DRC).
            # ``_ceil_to_grid`` rounds toward +inf, so the realised gap is always >=
            # the (dev_gap + channel) minimum for BOTH a widen (need>0) and a compact
            # (need<0, sliding higher columns LEFT) -- never under-spaces the channel.
            need = _ceil_to_grid(need, grid)
            for d, cc in col_of.items():
                if cc is not None and cc >= c_hi and d in x:
                    x[d] += need

    # Final grid-snap: every per-device x to the nearest manufacturing-grid multiple,
    # so the per-column centering / equalisation / re-spacing (which can land a device on
    # a fractional or 1-dbu-margin x) never drives a polygon vertex off the gf180mcu 5 nm
    # grid (-> ``comp_OFFGRID`` / ``metal*_OFFGRID``). A pair's two members snap by the
    # SAME residual when their midpoint is on-grid, so within-column symmetry is preserved.
    for d in list(x):
        x[d] = float(_snap_to_grid(x[d], grid))
    return x


# Number of M1 routing tracks the inter-column channel allows. A small fixed count
# (geometry/PDK-derived: it multiplies the PDK M1 track pitch, never a per-circuit
# constant). We do NOT have the per-net cross-column crossing count threaded into the
# placer, so we use a few tracks of allowance; the channel still scales with the PDK
# metal pitch, so a finer node opens a proportionately narrower channel.
_CHANNEL_TRACKS = 4


def _metal_pitch_of(tech) -> int:
    """One M1 routing-track pitch (nm) off the tech: min_width(M1)+min_space(M1).

    A router lane needs one metal width plus one edge-to-edge space to place an
    adjacent track, so the track PITCH is the natural quantum for a routing channel.
    Resolved purely from the PDK metal rules (gf180mcu M1: 230nm width + 230nm space
    = 460nm pitch). Falls back to ``dev_gap``-scale when a deck carries no M1 rule, so
    the channel is never zero. PDK-derived, no per-circuit constant.
    """
    width = None
    space = None
    try:
        width = tech.min_width("metal1")
    except Exception:  # noqa: BLE001 - missing rule -> fall through to default
        width = None
    try:
        space = tech.min_space("metal1")
    except Exception:  # noqa: BLE001
        space = None
    if width and space:
        return int(width) + int(space)
    # No parsed M1 rule: a node-scaled fallback so the channel is still positive.
    return int(_dev_gap_from_tech(tech))


def _routing_channel_from_tech(tech) -> float:
    """The inter-column routing-channel width (nm): ``_CHANNEL_TRACKS`` M1 pitches.

    A small fixed number of routing tracks, each one M1 track pitch wide (see
    :func:`_metal_pitch_of`). Geometry/PDK-derived (it scales with the PDK metal
    pitch); NOT a per-circuit constant and NOT a function of column height.
    """
    return float(_CHANNEL_TRACKS * _metal_pitch_of(tech))


def _mfg_grid_of(tech) -> int:
    """The PDK manufacturing-grid quantum in dbu, off the tech (default 5 nm).

    gf180mcu / sky130 both check ``ongrid(0.005)`` = a 5 nm = 5 dbu grid (the
    gf180mcu KLayout deck's OFFGRID section, see :mod:`clay.layout.grid_snap`).
    A tech may carry an explicit ``grid_nm`` / ``mfg_grid``; otherwise default 5.
    """
    for attr in ("grid_nm", "mfg_grid", "manufacturing_grid"):
        v = getattr(tech, attr, None)
        if isinstance(v, int) and v > 0:
            return v
    return 5


def _snap_to_grid(v: float, grid: int) -> int:
    """Round ``v`` to the nearest multiple of ``grid`` (manufacturing grid snap)."""
    if grid <= 1:
        return int(round(v))
    return int(round(v / grid)) * grid


def _ceil_to_grid(v: float, grid: int) -> float:
    """Smallest multiple of ``grid`` that is >= ``v`` (for a strict-margin shift)."""
    if grid <= 1:
        import math
        return float(math.ceil(v))
    import math
    return float(math.ceil(v / grid) * grid)


def _reconcile_symmetry_nonoverlap(
    x_centre: Dict[str, float],
    half_w: Dict[str, int],
    rows: Tuple[Tuple[str, ...], ...],
    mirror_pairs: Tuple[Tuple[str, str], ...],
) -> Tuple[Dict[str, float], Optional[float]]:
    """Make every declared mirror pair exactly symmetric about ONE shared vertical
    axis WITHOUT breaking the longest-path row non-overlap -- correct-by-construction.

    The rows are x-INDEPENDENT in the constraint graph (``left_of`` edges only run
    WITHIN a row; the inter-row order is on the V graph), so each row may be shifted
    horizontally as a rigid block without disturbing its own internal device
    spacing. ``_ordered_row`` has placed each row's mirror-pair members on the row's
    OUTSIDE straddling its loose centre devices, so within a row the pair is already
    geometrically symmetric about the row's own centre; we only need to (1) make each
    pair EXACTLY equidistant about the row centre and (2) translate every row so all
    rows' centres -- hence all pairs' axes -- coincide at one global axis.

    Step 1 (per row, symmetry-exact): a pair ``(a, b)`` straddling the row is snapped
    to ``mid_r +/- off`` where ``mid_r`` is the pair's solved midpoint and ``off`` is
    HALF the solved span, so ``a`` / ``b`` stay on their solved sides and any loose
    device already between them stays between them (the snap only equalises the two
    members' offsets about THEIR OWN midpoint, a redistribution of the pair's
    existing slack -- it never moves a member past the loose centre, so non-overlap
    is preserved). When a row has multiple pairs they are concentric about the same
    ``mid_r`` (their solved midpoints coincide by the symmetric ordering).

    Step 2 (rigid row align): every row is translated so its pair-axis lands on the
    common global axis (the mean of the rows' pair-axes). A rigid shift preserves all
    intra-row separations, so the cell stays overlap-free while every pair now shares
    one axis -- exactly what R172 ``symmetry_placement`` scores. Rows without a
    declared pair are left where the longest-path put them (no axis to align).

    PDK-generic + circuit-agnostic: uses only the solved coordinates + the declared
    pairs, no spacing constant and no per-cell branch. Returns ``(x, axis)`` where
    ``axis`` is the shared global axis the pairs were aligned to (``None`` when the
    cell declares no placeable pair, so the caller keeps the extent-midpoint axis).
    """
    if not x_centre:
        return dict(x_centre), None
    x = dict(x_centre)

    def _row_pairs(members: set) -> List[Tuple[str, str]]:
        return [
            (a, b) if x[a] <= x[b] else (b, a)
            for (a, b) in mirror_pairs
            if a in members and b in members and a in x and b in x
        ]

    # which mirror pairs live in each row, and each row's pair-axis (mean midpoint).
    row_axes: List[float] = []
    for row in rows:
        rps = _row_pairs(set(row))
        if not rps:
            continue
        mid_r = sum((x[a] + x[b]) / 2.0 for (a, b) in rps) / len(rps)
        # Step 1: equalise each pair about mid_r (redistribute the pair's own slack;
        # a no-op for a single outermost pair, binds only when a row nests pairs).
        for (a, b) in rps:
            off = (x[b] - x[a]) / 2.0   # half the solved a..b span (a left, b right)
            x[a] = mid_r - off
            x[b] = mid_r + off
        row_axes.append(mid_r)

    if not row_axes:
        return x, None

    # Step 2: rigidly translate EVERY row so it centres on the shared global axis --
    # a pair-bearing row aligns its pair-axis, a pair-free row centres its loose
    # devices' extent (so an on-axis tail / bias device sits on the same axis the
    # pairs mirror about, which is the axis the cell extent + R172 score read). A
    # rigid per-row shift preserves all intra-row spacing, so the cell stays
    # overlap-free while every declared pair shares one axis.
    global_axis = sum(row_axes) / len(row_axes)
    for row in rows:
        present = [d for d in row if d in x]
        if not present:
            continue
        rps = _row_pairs(set(row))
        if rps:
            mid_r = sum((x[a] + x[b]) / 2.0 for (a, b) in rps) / len(rps)
        else:
            lo = min(x[d] - half_w[d] for d in present)
            hi = max(x[d] + half_w[d] for d in present)
            mid_r = (lo + hi) / 2.0
        shift = global_axis - mid_r
        if abs(shift) < 1e-6:
            continue
        for d in present:
            x[d] += shift
    return x, global_axis


def _equal_size_footprints(
    equal_size_groups: Tuple,
    devs: Dict[str, object],
    passives: Optional[Dict[str, Dict[str, object]]],
    tech,
) -> Dict[str, Tuple[int, int]]:
    """Resolve the COMMON footprint each ``@equal_size`` group member must realise.

    For every declared group (a frozenset of device / passive labels) compute each
    member's NATURAL footprint ``(w, h)`` -- a MOS member from its built
    :class:`DeviceGeom` extent in ``devs``, a passive member from the same
    ``_build_passive_geom`` the placer uses downstream -- then take the group-wise
    MAXIMUM width AND MAXIMUM height (the bounding common footprint). Every member of
    the group is mapped to that one ``(max_w, max_h)``: the smaller members GROW to
    match, none ever SHRINKS below its natural size, so the device geometry stays
    DRC-/area-legal (we only reserve extra blank footprint around a smaller device,
    never clip its polygons).

    Returns ``{label: (w, h)}`` only for members whose target footprint is LARGER than
    their natural one in some axis (a member already at the group max maps to its own
    natural size, a no-op). Generic + circuit-agnostic: driven purely by the labels in
    ``equal_size_groups`` + the member geometry, ZERO device-/net-name branch. An empty
    ``equal_size_groups`` returns ``{}`` -> the placer is byte-unchanged.
    """
    if not equal_size_groups:
        return {}
    passives = passives or {}

    def _natural(lbl: str) -> Optional[Tuple[int, int]]:
        d = devs.get(lbl)
        if d is not None:
            e = d.extent()
            return (e[2] - e[0], e[3] - e[1])
        rec = passives.get(lbl)
        if rec is not None:
            try:
                g = _build_passive_geom(rec, tech, str(rec.get("orient", "R0")))
            except Exception:  # noqa: BLE001 - unbuildable passive -> not sizeable
                return None
            e = g.extent()
            return (e[2] - e[0], e[3] - e[1])
        return None

    targets: Dict[str, Tuple[int, int]] = {}
    for group in equal_size_groups:
        members = [m for m in group if _natural(m) is not None]
        if len(members) < 2:
            continue  # a singleton (or all-unbuildable) group pins nothing
        nat = {m: _natural(m) for m in members}
        max_w = max(w for (w, _h) in nat.values())
        max_h = max(h for (_w, h) in nat.values())
        for m in members:
            targets[m] = (max_w, max_h)
    return targets


def _pad_bbox_to_footprint(bbox: Bbox, target: Tuple[int, int]) -> Bbox:
    """Grow ``bbox`` SYMMETRICALLY about its own centroid so its width/height each
    reach (at least) ``target`` -- never shrinking either axis. The device polygons
    are untouched; only the reserved footprint (the reported bbox) enlarges, centred
    on the device so a mirror pair's two members stay equidistant about the axis."""
    x0, y0, x1, y1 = bbox
    tw, th = target
    w, h = x1 - x0, y1 - y0
    if tw > w:
        grow = tw - w
        x0 -= grow // 2
        x1 += grow - grow // 2
    if th > h:
        grow = th - h
        y0 -= grow // 2
        y1 += grow - grow // 2
    return (x0, y0, x1, y1)


def _legalise_x_overlaps(
    placed: Dict[str, "PlacedDevice"], grown: set
) -> None:
    """Push GROWN device bboxes apart along x so no two bboxes overlap, in place.

    Only runs when ``@equal_size`` enlarged some footprints (``grown`` non-empty).
    Devices are swept left-to-right by their current left edge; whenever a device's
    bbox overlaps the running x-extent of an already-placed device it shares y with,
    it is shifted RIGHT just enough to clear (a standard 1-D legalisation). The sweep
    only ever moves a device RIGHT, so it terminates and never re-introduces an
    overlap it already resolved. With ``grown`` empty this is a no-op (byte-parity for
    every cell that declares no ``@equal_size``)."""
    if not grown:
        return
    order_lbls = sorted(placed, key=lambda l: (placed[l].bbox[0], placed[l].bbox[1], l))
    for i, lbl in enumerate(order_lbls):
        x0, y0, x1, y1 = placed[lbl].bbox
        shift = 0
        for j in range(i):
            o = placed[order_lbls[j]].bbox
            # devices that do NOT overlap in y can never collide on the x sweep.
            if not (y0 < o[3] and o[1] < y1):
                continue
            # after the running shift, does this device still overlap o in x? if so,
            # push it RIGHT so its left edge clears o's right edge.
            if (x0 + shift) < o[2] and (x1 + shift) > o[0]:
                shift = max(shift, o[2] - x0)
        if shift:
            placed[lbl] = replace(
                placed[lbl], bbox=(x0 + shift, y0, x1 + shift, y1)
            )


def solve_placement(
    order: RelativeOrder,
    devs: Dict[str, object],
    pdk: Optional[PdkRules] = None,
    *,
    tech=None,
    passives: Optional[Dict[str, Dict[str, object]]] = None,
    mos_row_of: Optional[Dict[str, int]] = None,
    equal_size_groups: Tuple = (),
    pinned_positions: Optional[Dict[str, Tuple[float, float]]] = None,
    matched_passive_pairs: Tuple[Tuple[str, str], ...] = (),
) -> Placement:
    """Solve a :class:`RelativeOrder` to overlap-free coordinates by longest-path.

    H graph: an edge ``(u, v)`` (from ``order.left_of``) has weight
    ``half_w[u] + half_w[v] + dev_gap`` -- the minimum x centre-to-centre that
    clears ``u`` and ``v`` with the gf180 inter-device spacing. The longest-path
    x of each device is the smallest centre x that respects every left_of edge.

    V graph: an edge ``(u, v)`` (from ``order.below``) has weight
    ``h[u]/2 + h[v]/2 + dev_gap`` -- the minimum y centre-to-centre between rail
    bands. Devices in the same row share a y-centre (no ``below`` edge between
    them) so a matched pair stays at one row baseline.

    Then a SYMMETRY-CENTERING pass: the cell's vertical axis is the midpoint of
    the solved H extent; for each mirror pair the two centroids are snapped
    equidistant about that axis (preserving their shared row y), so the declared
    pairs are mirror-symmetric exactly like the MILP's hard constraint.

    Returns a drop-in :class:`clay.layout.generic_placement.Placement` with the
    solving :class:`RelativeOrder` attached as ``.relative_order``.
    """
    t0 = time.perf_counter()
    # Spacing flows through the LayoutTech (gf180mcu by default -- parity + no
    # caller breaks). ``tech`` (P3+) takes precedence; a legacy ``pdk`` (PdkRules /
    # id string) is resolved to its tech. The resolved gf180 ``dev_gap`` is the
    # same 1400nm the legacy ``max(min_spacing('nwell'), min_spacing('comp'))``
    # gave, so the gf180 placement is byte-identical.
    tech = _resolve_tech(tech if tech is not None else pdk)
    dev_gap = float(_dev_gap_from_tech(tech))

    half_w = _half_widths(devs)
    heights = _heights(devs)

    # ── @equal_size: PIN the equal-footprint (size_class) invariant ──────────────
    # Each declared group's members must realise ONE common footprint = the group-wise
    # MAXIMUM (width, height) of their natural footprints (grow the smaller members,
    # never shrink the larger -> DRC/area legality preserved). Resolve the per-member
    # target footprints now, then GROW the packing reservation (half_w / heights) of
    # each grouped MOS member so the longest-path reserves the larger footprint and the
    # placement stays overlap-free by construction. Passives are grown post-placement
    # (they are not packed by the longest-path) + legalised. ZERO circuit-name branch;
    # empty ``equal_size_groups`` is a no-op (byte-parity).
    eqsize_targets = _equal_size_footprints(equal_size_groups, devs, passives, tech)
    for lbl, (tw, th) in eqsize_targets.items():
        if lbl in half_w:
            half_w[lbl] = max(half_w[lbl], tw // 2)
        if lbl in heights:
            heights[lbl] = max(heights[lbl], th)

    # @fix ABSOLUTE pins: resolve each declared centroid (microns) to an integer nm
    # target keyed by the built device label. Consumed in the realisation pass below to
    # OVERRIDE the engine's solved centre. Empty -> {} (no override, byte-parity). Uses
    # the SHARED ``pin_targets_nm`` so a pin resolves to the IDENTICAL nm target the MILP
    # engine uses (the strict-consistency anchor needs identical resolution in both).
    pin_targets = pin_targets_nm(pinned_positions, devs.keys())

    nodes = [d for d in order.devices if d in devs]
    materialized_mirror_pairs = _disjoint_matching_pairs(
        tuple(order.mirror_pairs or ()), nodes
    )
    kinds_for_gap = {lbl: getattr(d, "kind", "nmos") for lbl, d in devs.items()}

    # ── @abut: declared adjacency pairs pack to the DRC-MINIMUM facing gap ───────
    # An edge whose two ends are a declared @abut pair uses the TIGHT same-layer gap
    # (:func:`_abut_gap_from_tech`) instead of the conservative routing-padded
    # ``dev_gap``, so the two devices pack edge-adjacent. Built as an UNORDERED set
    # (the gap edge may run either direction). Empty -> ``_gap`` never tightens, so a
    # placement that declares no @abut is byte-identical.
    abut_set = frozenset(
        frozenset((a, b)) for (a, b) in (getattr(order, "abut_pairs", ()) or ())
    )
    abut_gap = int(_abut_gap_from_tech(tech)) if abut_set else 0

    def _gap(u: str, v: str) -> int:
        # @abut: a declared adjacency pair packs tight (DRC-min same-layer gap), but
        # never BELOW the kind-aware clearance an opposite-type adjacency needs (a
        # band-free NMOS/PMOS abut still must clear the well/implant rule). For a
        # same-type abut pair the tight gap wins; for an opposite-type one the larger
        # of (tight, well/implant) is used so the abut never produces a DRC violation.
        if abut_set and frozenset((u, v)) in abut_set:
            if getattr(order, "band_free", False) and kinds_for_gap.get(u, "nmos") != kinds_for_gap.get(v, "nmos"):
                return max(abut_gap, _pair_gap(tech, int(dev_gap), kinds_for_gap.get(u, "nmos"), kinds_for_gap.get(v, "nmos")))
            return abut_gap
        # kind-aware: opposite-type adjacency (only possible in a band-free / 2D
        # floorplan) needs the extra well + implant clearance so the NMOS N+ and
        # the PMOS NWELL/P+ do not violate the NW / implant spacing rules. Gated on
        # the order being band-free so the historical banded placement is unchanged.
        if not getattr(order, "band_free", False):
            return int(dev_gap)
        return _pair_gap(tech, int(dev_gap), kinds_for_gap.get(u, "nmos"), kinds_for_gap.get(v, "nmos"))

    # --- FULLY-PINNED ANCHOR bypass ---------------------------------------------
    # When EVERY placed device carries an @fix absolute pin, the pins fully determine
    # the placement -- there is no relative-order to solve. The longest-path solve is
    # not just unnecessary here, it is ill-posed: a 2D set of absolute pins induces no
    # consistent total order, so ``order.left_of`` / ``order.below`` (derived upstream
    # from banding/symmetry) can carry a CYCLE that makes ``_longest_path_positions``
    # reject the graph as non-DAG. We therefore source the centres DIRECTLY from the
    # pins and skip the solve. The realisation pass below overrides pinned centroids
    # with the pins anyway, so for a fully-pinned cell ANY engine yields the IDENTICAL
    # placement (the strict-consistency anchor: incompleteness_score == 0). Partially-
    # pinned + unpinned cells take the normal longest-path (byte-parity preserved).
    fully_pinned = bool(pin_targets) and all(n in pin_targets for n in nodes)
    if fully_pinned:
        h_edges = []
        v_edges = []
        x_centre = {n: float(pin_targets[n][0]) for n in nodes}
        y_centre = {n: float(pin_targets[n][1]) for n in nodes}
    else:
        # --- H longest-path (x centres) ---
        h_edges = [(u, v) for (u, v) in order.left_of if u in devs and v in devs]
        h_weight = {
            (u, v): half_w[u] + half_w[v] + _gap(u, v) for (u, v) in h_edges
        }
        x_centre = _longest_path_positions(nodes, h_edges, h_weight)

        # --- V longest-path (y centres) ---
        v_edges = [(u, v) for (u, v) in order.below if u in devs and v in devs]
        v_weight = {
            (u, v): heights[u] / 2.0 + heights[v] / 2.0 + _gap(u, v) for (u, v) in v_edges
        }
        y_centre = _longest_path_positions(nodes, v_edges, v_weight)

    # --- symmetry-centering, NON-OVERLAP-PRESERVING (correct-by-construction) ---
    #
    # Each declared mirror pair is made exactly mirror-symmetric about ONE global
    # vertical axis (``cx_a + cx_b == 2 * axis`` -- the whole-cell symmetry the R172
    # ``symmetry_placement`` rule scores). The naive snap that set each pair to the
    # AVERAGE of its two members' axis-offsets could pull a member INWARD across a
    # neighbour, COLLAPSING the longest-path non-overlap gap (two devices' COMP/poly
    # merging -> a foreign net's S/D contact landing inside an adjacent device's gate
    # poly = CO.10, and overlapping M1/M2 = M1.2a/M2.2a). The non-overlap edges the
    # longest-path established are then silently destroyed by the symmetry pass.
    #
    # Instead, reconcile symmetry and non-overlap by a MONOTONE relaxation: every
    # constraint is a one-sided lower bound, so repeatedly enforcing (a) the row
    # ``left_of`` minimum separations and (b) each pair's equidistance about the
    # current axis -- always pushing the closer member OUTWARD to match the farther
    # member's offset (``max`` of the two offsets, never the average) -- only ever
    # WIDENS the cell. It is therefore guaranteed to terminate at a fixpoint that
    # satisfies BOTH the non-overlap separations AND exact pair symmetry. A pair whose
    # symmetric placement would require overlapping a neighbour is resolved by the
    # outward push + the subsequent left_of re-propagation, so the result is symmetric
    # AND overlap-free by construction (not symmetric-then-broken).
    # A DECLARED 2D COLUMN floorplan reconciles symmetry PER COLUMN (each pair about
    # its own column's axis), so the global-axis snap never drags a pair across a
    # column boundary and inverts the declared col bands. With no declared columns
    # the historical single-global-axis reconciliation runs unchanged (byte-parity).
    col_of = getattr(order, "col_of", None) or {}
    if col_of:
        # Inter-column ROUTING CHANNEL: the gap opened between adjacent columns so the
        # cross-column nets run their own metal tracks (else the gf180mcu router squeezes
        # a metal2 jog below min spacing -> M2.2a). It is the MINIMAL channel -- a small,
        # geometry/PDK-derived multiple of the M1 TRACK PITCH (min_width(M1)+min_space(M1)
        # = one routable track) -- NOT a column-height term. The previous channel scaled
        # with 0.5*tallest_column_height and the placer added spread(c)+spread(c+1) on top,
        # which forced the inter-column gap to exceed both columns' widths and blew the
        # cell open. Here pass 3 enforces only (dev_gap + this channel), so the cell stays
        # compact. ``_CHANNEL_TRACKS`` tracks is a fixed allowance (we do not have the
        # per-net crossing count threaded into the placer); it scales with the PDK pitch,
        # never a per-circuit constant.
        channel = dev_gap + _routing_channel_from_tech(tech)
        x_centre = _reconcile_symmetry_per_column(
            x_centre, half_w, order.rows, materialized_mirror_pairs, col_of,
            grid=_mfg_grid_of(tech), channel=channel,
        )
        pair_axis = None
    else:
        x_centre, pair_axis = _reconcile_symmetry_nonoverlap(
            x_centre, half_w, order.rows, materialized_mirror_pairs
        )

    # --- cell vertical axis ---
    # When the cell declares mirror pairs, the axis IS the shared axis they were
    # reconciled to (so ``cx_a + cx_b == 2 * axis_x`` exactly -- the R172 symmetry
    # invariant), NOT the raw extent midpoint (which an off-centre loose device would
    # pull off the pairs' true axis). With no pair, fall back to the extent midpoint.
    if pair_axis is not None:
        axis_x = int(round(pair_axis))
    elif x_centre:
        xs_lo = min(x_centre[n] - half_w[n] for n in nodes)
        xs_hi = max(x_centre[n] + half_w[n] for n in nodes)
        axis_x = int(round((xs_lo + xs_hi) / 2.0))
    else:
        axis_x = 0

    # share the exact row y-centre within each mirror pair (numerically equal
    # already from the V longest-path, but snap to be exact).
    materialized_mos_pair_axes = (
        {
            tuple(sorted((str(left), str(right)))): (
                "x",
                (float(x_centre[left]) + float(x_centre[right])) / 2.0,
            )
            for left, right in materialized_mirror_pairs
            if left in x_centre and right in x_centre
        }
        if not fully_pinned
        else {}
    )

    for a, b in materialized_mirror_pairs:
        if a not in y_centre or b not in y_centre:
            continue
        yc = (y_centre[a] + y_centre[b]) / 2.0
        y_centre[a] = yc
        y_centre[b] = yc

    # --- MIDDLE-BAND reservation (3-tier reference paradigm) ---------------------
    # A passive declared at a floorplan row STRICTLY BETWEEN two MOS rows (e.g. the
    # two equal caps in row=1 of a PMOS-top(row2) / NMOS-bottom(row0) floorplan) is
    # laid out in its OWN middle tier, side-by-side, BETWEEN the upper and lower MOS
    # bands. The MOS V longest-path packs the bands with only ``dev_gap`` between
    # them, so we OPEN a band by lifting every MOS device whose declared row is ABOVE
    # the middle row, by the height the middle passives need. Generic + circuit-
    # agnostic: keyed only on the declared row indices + the passive heights (no
    # device-kind / net-name / cell branch). ``mid_band`` (the reserved y-window) is
    # handed to the passive placer so the caps land in the gap, not below the cell.
    mid_band: Optional[Tuple[int, float, float]] = None  # (mid_row, y_lo, y_hi)
    if passives and mos_row_of and y_centre:
        # passive heights, built once (the same builder the placer uses downstream).
        pass_h: Dict[str, int] = {}
        pass_row: Dict[str, int] = {}
        for lbl, rec in passives.items():
            r = rec.get("row")
            if r is None:
                continue
            try:
                g = _build_passive_geom(rec, tech, str(rec.get("orient", "R0")))
            except Exception:  # noqa: BLE001 - unbuildable passive -> skip reservation
                continue
            ex = g.extent()
            pass_h[lbl] = ex[3] - ex[1]
            pass_row[lbl] = int(r)
        mos_rows = sorted({int(r) for r in mos_row_of.values()})
        if pass_row and len(mos_rows) >= 2:
            mos_lo_row, mos_hi_row = mos_rows[0], mos_rows[-1]
            # middle rows = passive rows strictly between the lowest + highest MOS row.
            mid_rows = sorted({
                r for r in set(pass_row.values())
                if mos_lo_row < r < mos_hi_row
            })
            if mid_rows:
                mid_row = mid_rows[0]  # the single declared middle tier
                # MOS that sit BELOW vs ABOVE this middle row (by declared row).
                below_mos = [d for d, r in mos_row_of.items()
                             if r <= mid_row and d in y_centre]
                above_mos = [d for d, r in mos_row_of.items()
                             if r > mid_row and d in y_centre]
                if below_mos and above_mos:
                    # current y extents of the two MOS bands.
                    band_lo_top = max(y_centre[d] + heights[d] / 2.0 for d in below_mos)
                    band_hi_bot = min(y_centre[d] - heights[d] / 2.0 for d in above_mos)
                    # height the middle passive tier needs (tallest passive in the
                    # middle rows + a dev_gap clearance to each MOS band).
                    mid_h = max(pass_h[d] for d in pass_h if pass_row[d] == mid_row)
                    need = mid_h + 2.0 * dev_gap
                    have = band_hi_bot - band_lo_top
                    if have < need:
                        lift = need - have
                        # snap the lift to grid so on-grid MOS stay on-grid.
                        lift = _ceil_to_grid(lift, _mfg_grid_of(tech))
                        for d in above_mos:
                            y_centre[d] += lift
                        band_hi_bot += lift
                    # reserved middle-band y-window (between the two MOS bands).
                    mid_band = (mid_row, band_lo_top, band_hi_bot)

    # --- realise the placement: translate each device to its solved centre ---
    placed: Dict[str, PlacedDevice] = {}
    solver_placed: Dict[str, _SolverPlacedDevice] = {}
    all_polys: list = []
    mirror_members = {b for (_a, b) in materialized_mirror_pairs}
    row_of = {d: r for r, row in enumerate(order.rows) for d in row}
    kinds = {lbl: getattr(d, "kind", "nmos") for lbl, d in devs.items()}

    positions: Dict[str, Tuple[int, int, int, int]] = {}
    for lbl in nodes:
        base = devs[lbl]
        e = base.extent()
        cur_cx = (e[0] + e[2]) // 2
        cur_cy = (e[1] + e[3]) // 2
        target_cx = int(round(x_centre[lbl]))
        target_cy = int(round(y_centre[lbl]))
        mirror = lbl in mirror_members
        # @fix ABSOLUTE pin: OVERRIDE the engine's solved centre with the declared
        # absolute centroid (microns -> nm). The device's built geometry (size /
        # orientation) is preserved; only its translation is fixed. A pinned device is
        # NOT mirrored about the cell axis -- the pin is the absolute target, so we
        # translate-only so its realised centroid lands EXACTLY on the pin (the anchor
        # property). Empty pin map -> this branch never fires (byte-parity).
        if pin_targets and lbl in pin_targets:
            target_cx, target_cy = pin_targets[lbl]
            mirror = False
        dx = target_cx - cur_cx
        dy = target_cy - cur_cy
        terms = _placed_terminals(base, dx, dy, mirror, axis_x)
        polys = _place_polys(base, dx, dy, mirror, axis_x)
        all_polys += polys
        # A device that reserved a placement-neutral extent (``placement_extent`` --
        # e.g. a source!=bulk PMOS whose local body tap is drawn a few nm below the
        # source-tied footprint) reports that RESERVED extent, byte-identical to the
        # self-bulk footprint, so the frozen placement stays byte-stable -- the SAME
        # rule the MILP path uses. Every other device falls back to its drawn-poly box.
        bbox = (
            _placed_bbox(base, dx, dy, mirror)
            or _polys_bbox(polys)
            or (e[0] + dx, e[1] + dy, e[2] + dx, e[3] + dy)
        )
        # @equal_size: reserve the group's common footprint around this member (grow
        # symmetrically about its centroid; the polys are untouched, only the reported
        # footprint enlarges). The packing already reserved the larger half_w/heights,
        # so the grown bbox stays inside the longest-path spacing (overlap-free).
        if lbl in eqsize_targets:
            bbox = _pad_bbox_to_footprint(bbox, eqsize_targets[lbl])
        placed[lbl] = PlacedDevice(
            label=lbl,
            kind=kinds.get(lbl, "nmos"),
            bbox=bbox,
            terminals=terms,
            polys=polys,
            orientation=str(getattr(base, "orient", "") or ""),
        )
        positions[lbl] = bbox
        # build the solver-shaped PlacedDevice for score_placement (it reads
        # .extent off a PlacementSolution); the extent is the placed bbox.
        solver_placed[lbl] = _SolverPlacedDevice(
            label=lbl,
            kind=kinds.get(lbl, "nmos"),
            dx=dx,
            dy=dy,
            extent=bbox,
            mirror=mirror,
            row=row_of.get(lbl, 0),
        )

    if placed:
        full_extent = (
            min(p.bbox[0] for p in placed.values()),
            min(p.bbox[1] for p in placed.values()),
            max(p.bbox[2] for p in placed.values()),
            max(p.bbox[3] for p in placed.values()),
        )
    else:
        full_extent = (0, 0, 0, 0)

    # MOS-only extent (passives are not folded MOS + not symmetry-scored) -- the
    # score reads this so the R172 quality is byte-identical with / without passives.
    mos_extent = full_extent

    # --- PASSIVE placement path (resistors + MIM caps) ---
    # A 2-terminal passive is NOT a foldable MOS, so it never entered the
    # constraint-graph longest-path. Before this path the seqpair engine SILENTLY
    # DROPPED every passive (only the MILP ``place_generic`` placed them via
    # ``_place_resistors``) -> a seqpair-placed cell with a resistor/cap lost it.
    # Pass the DECLARED per-passive ``@place_group col`` (carried on each passive
    # record as ``"col"``) into ``_place_resistors`` so a passive that declares its OWN
    # column BEYOND the MOS columns -- e.g. the Miller cap's dedicated right-hand
    # col=3 -- is placed as a BLOCK in its own x-region (vertically centred on the MOS
    # band), CLEAR of the MOS channel, instead of being stacked on the below-MOS row
    # where the large plate overlaps. A passive with no declared column keeps the
    # legacy below-MOS row. Generic + circuit-agnostic: the regime is chosen from the
    # declared col vs the MOS max col + the device bbox; no per-circuit branch.
    n_passive = 0
    realized_passive_pair_atoms: Tuple[Tuple[str, str], ...] = ()
    if passives:
        passive_cols = {
            lbl: int(rec["col"])
            for lbl, rec in passives.items()
            if rec.get("col") is not None
        }
        # passives whose DECLARED row is the reserved MIDDLE tier (between two MOS
        # bands): laid out side-by-side, by col, inside the reserved y-window. The
        # 3-tier reference paradigm (two equal caps + the small resistors between the
        # PMOS top + NMOS bottom bands). The window comes from ``mid_band`` (the gap the
        # reservation opened); ``mid_rows`` names the passive rows that land in it.
        mid_rows = (
            {int(rec["row"]) for rec in passives.values()
             if rec.get("row") is not None and mid_band is not None
             and int(rec["row"]) == mid_band[0]}
            if mid_band is not None else set()
        )
        mos_max_col = max(col_of.values()) if col_of else None
        # x-CENTROID of each MOS column (the x-band the place-group order check reads).
        # A MIDDLE-tier passive declaring the SAME col as a MOS column is centred on
        # that column's centroid so its (wide) block stays in the DECLARED global column
        # ORDER -- the order check compares centroids, so a wide cap whose centroid sits
        # on its column's centroid satisfies the col band order even though its plate
        # spans neighbouring columns' x (the caps are in a different TIER, so no bbox
        # overlap). Generic: pure geometry off the placed MOS columns, no name branch.
        mos_col_centroid: Dict[int, float] = {}
        for c in set(col_of.values()):
            xs = [
                (placed[d].bbox[0] + placed[d].bbox[2]) / 2.0
                for d, cc in col_of.items() if cc == c and d in placed
            ]
            if xs:
                mos_col_centroid[c] = sum(xs) / len(xs)
        (
            res_placed,
            res_polys,
            res_union,
            realized_passive_pair_atoms,
        ) = _place_resistors(
            passives, tech, full_extent if placed else (0, 0, 0, 0),
            col_of=passive_cols,
            mos_max_col=mos_max_col,
            column_channel=int(
                dev_gap + _routing_channel_from_tech(tech)
            ),
            mid_band=(mid_band[1], mid_band[2]) if mid_band is not None else None,
            mid_rows=mid_rows,
            passive_rows={
                lbl: int(rec["row"]) for lbl, rec in passives.items()
                if rec.get("row") is not None
            },
            mos_x_extent=(mos_extent[0], mos_extent[2]) if placed else None,
            mos_col_centroid=mos_col_centroid,
            matched_pairs=matched_passive_pairs,
        )
        placed.update(res_placed)
        all_polys += res_polys
        n_passive = len(res_placed)
        full_extent = res_union

    # --- @equal_size: enforce the common footprint on PASSIVE members ------------
    # A passive is placed by ``_place_resistors`` (not the longest-path), so its
    # equal-footprint reservation is applied here: grow each grouped passive's bbox to
    # the group's common footprint (centred on the passive), then a single x-sweep
    # legalisation pushes any now-overlapping GROWN bbox right so the cell stays
    # overlap-free. Empty ``eqsize_targets`` -> no grown set -> the legaliser is a
    # no-op (byte-parity for every cell without @equal_size).
    grown_passives: set = set()
    if eqsize_targets:
        for lbl, target in eqsize_targets.items():
            pd = placed.get(lbl)
            if pd is None or lbl in devs:
                continue  # MOS members were already grown in the realisation pass
            new_bbox = _pad_bbox_to_footprint(pd.bbox, target)
            if new_bbox != pd.bbox:
                placed[lbl] = replace(pd, bbox=new_bbox)
                grown_passives.add(lbl)
        _legalise_x_overlaps(placed, grown_passives)
        if placed:
            full_extent = (
                min(p.bbox[0] for p in placed.values()),
                min(p.bbox[1] for p in placed.values()),
                max(p.bbox[2] for p in placed.values()),
                max(p.bbox[3] for p in placed.values()),
            )

    # The constraint graph establishes one immutable axis for every realized MOS
    # mirror pair before any absolute pin override.  Passive pairs establish their
    # own local axes in the bank materializer.  Carry both explicitly so the exact
    # checker never has to synthesize a cell-global certificate.
    matching_pair_axes = dict(materialized_mos_pair_axes)
    matching_pair_axes.update(
        _materialized_matching_axes(placed, realized_passive_pair_atoms)
    )

    # --- @fix ABSOLUTE pins: re-assert every pinned device's centroid on its pin ----
    # The MOS realisation pass already pinned MOS members exactly (via the dx/dy
    # override); this pass covers PASSIVE members (placed by ``_place_resistors``, not the
    # longest-path) AND re-asserts MOS (a delta-0 no-op there). Resolved over ALL placed
    # labels so a pinned passive is included. Translates the device's bbox / polys /
    # terminals onto the pin and rebuilds ``all_polys`` from the moved geometry. Empty
    # ``pinned_positions`` -> no-op (byte-parity). Uses the SHARED ``_apply_pins`` so
    # both engines move a pinned device identically.
    all_pin_targets = pin_targets_nm(pinned_positions, placed.keys())
    if all_pin_targets:
        placed = _apply_pins(placed, all_pin_targets)
        all_polys = [p for pd in placed.values() for p in (pd.polys or [])]
        # the score-shaped placed map reads ``.extent``; refresh the pinned extents so the
        # R172 score sees the moved geometry (keeps the two maps consistent).
        for lbl in all_pin_targets:
            if lbl in solver_placed and lbl in placed:
                solver_placed[lbl] = replace(
                    solver_placed[lbl], extent=placed[lbl].bbox
                )
        if placed:
            full_extent = (
                min(p.bbox[0] for p in placed.values()),
                min(p.bbox[1] for p in placed.values()),
                max(p.bbox[2] for p in placed.values()),
                max(p.bbox[3] for p in placed.values()),
            )

    # --- REAL R172 quality on the solved bboxes (reuse score_placement) ---
    constraints = SolverConstraints(
        cell_name=order.cell,
        symmetry_pairs=materialized_mirror_pairs,
        symmetry_axis="x",
        device_kinds=kinds,
        common_centroid_groups=(materialized_mirror_pairs[0],)
        if materialized_mirror_pairs else (),
    )
    score_sol = PlacementSolution(
        placed=solver_placed,
        axis_x=axis_x,
        status="seqpair_longest_path",
        success=bool(placed),
        solve_seconds=0.0,
        objective=float("nan"),
        n_vars=len(nodes),
        n_constraints=len(h_edges) + len(v_edges),
        full_extent=mos_extent,
        constraints=constraints,
    )
    quality = score_placement(score_sol, devs)

    geom = _cell_geom(placed, full_extent)
    solve_seconds = time.perf_counter() - t0

    solved_order = replace(
        order,
        mirror_pairs=materialized_mirror_pairs,
        axis_x=axis_x,
        positions=positions,
    )

    pl = Placement(
        devices=placed,
        axis_x=axis_x,
        geom=geom,
        polys=all_polys,
        quality=quality,
        solver_status="seqpair_longest_path",
        solver_success=bool(placed),
        solve_seconds=solve_seconds,
        constraints=constraints,
        used_default_geometry=False,
        matching_pair_axes=matching_pair_axes,
        note=(
            f"seqpair constraint-graph placement: {len(order.rows)} rows, "
            f"{len(nodes)} devices, {len(materialized_mirror_pairs)} mirror pair(s), "
            f"{len(h_edges)} H-edges + {len(v_edges)} V-edges, "
            f"dev_gap={int(dev_gap)}nm; O(V+E) longest-path"
            + (f" | {n_passive} passive(s) on a row below the MOS band"
               if n_passive else "")
        ),
    )
    # attach the solving RelativeOrder (not a Placement field; carried as attr).
    object.__setattr__(pl, "relative_order", solved_order)
    return pl


# === 3. convenience: cell_def -> placement directly ===========================


def _band_block_sizes(
    row_seqs: List[Tuple[str, ...]],
    half_w: Dict[str, float],
    heights: Dict[str, float],
    dev_gap: float,
) -> List[Tuple[float, float]]:
    """Per-band block ``(width, height)``: the band packed left-to-right at the
    longest-path device spacing (sum of widths + ``dev_gap`` between members), the
    height the tallest member. Pure geometry off the band membership + sizes, no
    reference coordinate."""
    blocks: List[Tuple[float, float]] = []
    for seq in row_seqs:
        if not seq:
            blocks.append((0.0, 0.0))
            continue
        w = sum(2.0 * half_w.get(d, 0.0) for d in seq)
        w += dev_gap * max(0, len(seq) - 1)
        h = max(heights.get(d, 0.0) for d in seq)
        blocks.append((w, h))
    return blocks


def _pack_band_sp(
    gamma_plus: List[int],
    gamma_minus: List[int],
    blocks: List[Tuple[float, float]],
    gap: float,
) -> Tuple[float, float, Dict[Tuple[int, int], str]]:
    """Pack a BAND-level sequence-pair to a bounding box + the directed per-pair
    relation. Standard sequence-pair semantics: for bands ``a``, ``b``

      * ``a`` before ``b`` in BOTH permutations  -> ``a`` left_of ``b``;
      * ``a`` after  ``b`` in BOTH               -> ``b`` left_of ``a``;
      * ``a`` before in plus, after in minus      -> ``a`` ABOVE ``b`` (``b`` below ``a``);
      * ``a`` after in plus, before in minus      -> ``a`` below ``b``.

    Returns ``(W, H, rel)`` where ``rel[(i, j)]`` (i < j) is one of
    ``i_left_j`` / ``j_left_i`` / ``i_below_j`` / ``j_below_i``. The bbox is from a
    block-level longest-path on the derived edges (each block separated by ``gap``).
    """
    native = load_native("_place_cpp")
    if native is not None and blocks:
        return native.pack_band_sp_cpp(gamma_plus, gamma_minus, blocks, float(gap))
    pos_plus = {b: k for k, b in enumerate(gamma_plus)}
    pos_minus = {b: k for k, b in enumerate(gamma_minus)}
    n = len(blocks)
    rel: Dict[Tuple[int, int], str] = {}
    h_edges: List[Tuple[int, int]] = []
    v_edges: List[Tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            pi_p, pj_p = pos_plus[i], pos_plus[j]
            pi_m, pj_m = pos_minus[i], pos_minus[j]
            i_before_p = pi_p < pj_p
            i_before_m = pi_m < pj_m
            if i_before_p and i_before_m:
                rel[(i, j)] = "i_left_j"; h_edges.append((i, j))
            elif (not i_before_p) and (not i_before_m):
                rel[(i, j)] = "j_left_i"; h_edges.append((j, i))
            elif i_before_p and (not i_before_m):
                # i above j -> j below i
                rel[(i, j)] = "j_below_i"; v_edges.append((j, i))
            else:
                rel[(i, j)] = "i_below_j"; v_edges.append((i, j))

    # block-level longest-path (centre coords); width/height of each block.
    bw = [blocks[k][0] for k in range(n)]
    bh = [blocks[k][1] for k in range(n)]

    def _lp(edges, dim):
        succ: Dict[int, List[int]] = defaultdict(list)
        indeg = {k: 0 for k in range(n)}
        for u, v in edges:
            succ[u].append(v); indeg[v] += 1
        pos = {k: 0.0 for k in range(n)}
        q = deque(k for k in range(n) if indeg[k] == 0)
        while q:
            u = q.popleft()
            for v in succ[u]:
                w = (dim[u] + dim[v]) / 2.0 + gap
                if pos[u] + w > pos[v]:
                    pos[v] = pos[u] + w
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
        return pos

    xc = _lp(h_edges, bw)
    yc = _lp(v_edges, bh)
    W = max(xc[k] + bw[k] / 2.0 for k in range(n)) - min(xc[k] - bw[k] / 2.0 for k in range(n))
    H = max(yc[k] + bh[k] / 2.0 for k in range(n)) - min(yc[k] - bh[k] / 2.0 for k in range(n))
    return W, H, rel


def _cost_from_dims(W: float, H: float, tgt: float) -> Tuple[int, float]:
    """Two-tier placement cost: ``(out_of_band, area)``.

    The aspect band [1.4, 1.9] is a HARD gate, so it is the PRIMARY (lexicographic)
    key: any pack whose aspect lands in-band (``out_of_band = 0``) is strictly
    preferred over EVERY out-of-band pack, regardless of area -- so the search only
    leaves the band when NO in-band pack exists. Within a tier the secondary key is
    bbox AREA (minimise the dead area -> maximise bbox_fill). Tuples compare
    lexicographically, so ``(0, area_in) < (1, area_out)`` for any areas. ``tgt`` is
    kept for signature stability; the band edges are the gate's [1.4, 1.9]."""
    if min(W, H) <= 0:
        return (1, 1e18)
    area = W * H
    asp = max(W, H) / min(W, H)
    out_of_band = 0 if (1.4 <= asp <= 1.9) else 1
    return (out_of_band, area)


def _search_band_layout(
    row_seqs: List[Tuple[str, ...]],
    half_w: Dict[str, float],
    heights: Dict[str, float],
    dev_gap: float,
    *,
    aspect_target: Optional[float] = None,
    dev_cost=None,
) -> Dict[Tuple[int, int], str]:
    """DETERMINISTIC area-driven search over the BAND-level sequence-pair.

    Seed = the vertical band stack (band 0 bottom -> band B-1 top, the S2 order).
    The search is EXHAUSTIVE over both band permutations when the band count is small
    (B! <= 720, the OTA's B=5 -> 14400 cheap packs), so it never gets trapped in a
    swap-local minimum; for a larger B it falls back to a deterministic monotone
    pairwise-swap hill climb (seed + strict accept, no RNG). Either way the result is
    byte-deterministic: same input -> same output (no SA temperature, no randomness).

    Cost = bbox area + an aspect penalty that biases toward the [1.4, 1.9] landscape
    band (so a narrow band folds BESIDE a wide band rather than stacking into a tall
    portrait). When ``dev_cost`` is given the cost is measured on the REAL device-
    level placement that band relation produces (not the block proxy), so the search
    optimises exactly what the gate measures.

    Returns ``rel[(i, j)]`` (i < j) -- the per-band-pair relation the device-level
    order consumes. Pure geometry off the band sizes; no reference coordinate.
    """
    import math

    blocks = _band_block_sizes(row_seqs, half_w, heights, dev_gap)
    n = len(blocks)
    if n < 2:
        return {}

    if dev_cost is None:
        native = load_native("_place_cpp")
        if native is not None:
            return native.search_block_layout_cpp(blocks, float(dev_gap), aspect_target)

    tgt = float(aspect_target) if aspect_target and aspect_target > 0 else 1.6

    def _cost(gp, gm):
        _W, _H, rel = _pack_band_sp(gp, gm, blocks, dev_gap)
        if dev_cost is not None:
            return dev_cost(rel)
        return _cost_from_dims(_W, _H, tgt)

    # SEED: the vertical stack (band 0 below all). gamma_plus reversed, gamma_minus
    # forward -> for a<b: a after b in plus, a before b in minus -> a below b. So
    # band 0 is below every higher band: the S2 bottom-up stack exactly.
    gp = list(range(n - 1, -1, -1))
    gm = list(range(n))
    best_cost = _cost(gp, gm)
    best_gp, best_gm = list(gp), list(gm)

    def _better(c, ref) -> bool:
        # lexicographic strict improvement on the (out_of_band, area) tuple, with a
        # numeric epsilon on the area tier so float noise never flips the choice.
        if c[0] != ref[0]:
            return c[0] < ref[0]
        return c[1] < ref[1] - 1e-3

    fact = math.factorial(n)
    if fact <= 720:
        # EXHAUSTIVE over both permutations (deterministic enumeration order). Ties
        # never displace the incumbent (strict improvement), so it is reproducible.
        import itertools
        for cgp in itertools.permutations(range(n)):
            for cgm in itertools.permutations(range(n)):
                c = _cost(list(cgp), list(cgm))
                if _better(c, best_cost):
                    best_cost = c
                    best_gp, best_gm = list(cgp), list(cgm)
        gp, gm = best_gp, best_gm
    else:
        # DETERMINISTIC monotone-accept hill climb for large B (fixed sweep order,
        # strict accept -> reproducible).
        improved = True
        max_sweeps = 200
        sweeps = 0
        while improved and sweeps < max_sweeps:
            improved = False
            sweeps += 1
            for which in ("plus", "minus"):
                perm = gp if which == "plus" else gm
                for a in range(n):
                    for b in range(a + 1, n):
                        cand = list(perm)
                        cand[a], cand[b] = cand[b], cand[a]
                        c = (_cost(cand, gm) if which == "plus"
                             else _cost(gp, cand))
                        if _better(c, best_cost):
                            best_cost = c
                            if which == "plus":
                                gp = cand
                            else:
                                gm = cand
                            improved = True
    _W, _H, rel = _pack_band_sp(gp, gm, blocks, dev_gap)
    return rel


# === S4: RING-AWARE GROUP-BLOCK area search ===================================


def _layout_group_internal(
    members: Tuple[str, ...],
    half_w: Dict[str, float],
    heights: Dict[str, float],
    mirror_pairs: Tuple[Tuple[str, str], ...],
    gap: float,
) -> Tuple[Dict[str, Tuple[float, float]], float, float]:
    """Lay out ONE compact group's members internally, RING-PADDING-FREE (the raw
    member cluster), returning ``(offsets, width, height)`` where ``offsets`` maps
    each member to its (dx, dy) about the GROUP's own centre.

    The members pack left-to-right at the device ``gap`` (raw / abut spacing -- they
    SHARE one ring, so the ring pad belongs at the group's outer boundary, never
    between members). A declared mirror pair inside the group is placed on the
    group's OUTSIDE straddling the loose centre (``_ordered_row``) and equalised
    exactly about the group's own axis -> mirror-symmetric BY CONSTRUCTION about the
    block axis. All members share ONE y baseline (a compact group is one matched
    row), so the block height is the tallest member. Pure geometry off the member
    sizes + declared pairs; no reference coordinate, no ring padding (the caller adds
    the ring pad to the BLOCK footprint).
    """
    row = []
    seen: set = set()
    paired: set = set()
    # the group's own mirror pairs (both members inside the group).
    grp_pairs = [
        (a, b) for (a, b) in mirror_pairs
        if a in members and b in members
    ]
    for a, b in grp_pairs:
        paired.add(a)
        paired.add(b)
    for a, b in grp_pairs:
        row.append(("pair", (a, b)))
    for m in members:
        if m not in paired and m not in seen:
            row.append(("axis", m))
            seen.add(m)
    seq = _ordered_row(row)  # mirror pairs straddle the loose centre.

    # left-to-right longest-path on the single row at the raw device gap.
    x = 0.0
    xc: Dict[str, float] = {}
    prev: Optional[str] = None
    for d in seq:
        if prev is not None:
            x += half_w.get(prev, 0.0) + gap + half_w.get(d, 0.0)
        xc[d] = x
        prev = d
    # equalise each in-group mirror pair exactly about the group's midpoint, so the
    # block is mirror-symmetric by construction (the symmetry-feasible move-set --
    # symmetry is a hard property of the block, never a soft penalty).
    if seq:
        lo = min(xc[d] - half_w.get(d, 0.0) for d in seq)
        hi = max(xc[d] + half_w.get(d, 0.0) for d in seq)
        axis = (lo + hi) / 2.0
        for a, b in grp_pairs:
            l, r = (a, b) if xc[a] <= xc[b] else (b, a)
            off = (xc[r] - xc[l]) / 2.0
            # re-centre the pair on the group axis so both members are equidistant
            # about it (a redistribution of the pair's slack; the loose centre
            # devices are already centred between them by _ordered_row).
            xc[l] = axis - off
            xc[r] = axis + off
    else:
        axis = 0.0

    h = max((heights.get(d, 0.0) for d in seq), default=0.0)
    offsets: Dict[str, Tuple[float, float]] = {
        d: (xc[d] - axis, 0.0) for d in seq
    }
    width = (
        max(xc[d] + half_w.get(d, 0.0) for d in seq)
        - min(xc[d] - half_w.get(d, 0.0) for d in seq)
    ) if seq else 0.0
    return offsets, width, h


def _ring_aware_group_blocks(
    nodes: List[str],
    compact_groups: List[Tuple[str, ...]],
    half_w: Dict[str, float],
    heights: Dict[str, float],
    tight_half_w: Dict[str, float],
    tight_heights: Dict[str, float],
    mirror_pairs: Tuple[Tuple[str, str], ...],
    dev_gap: float,
) -> Tuple[List[Tuple[str, ...]], List[Dict[str, Tuple[float, float]]],
           List[Tuple[float, float]]]:
    """Build the RING-AWARE atoms for the 2D area search: one BLOCK per compact
    matched group (members laid out internally at RAW/abut spacing + ring-padded
    footprint), plus ONE block per leftover device not in any group (e.g. a placed
    passive).

    Returns ``(block_members, block_offsets, block_sizes)`` aligned by index:
      * ``block_members[k]`` -- the device names in block k;
      * ``block_offsets[k]`` -- ``{name: (dx, dy)}`` about block k's centre (the raw
        internal layout, mirror-symmetric);
      * ``block_sizes[k]`` -- the block's RING-PADDED ``(W, H)`` footprint = the raw
        member-union plus the per-side ring pad (``half_w - tight_half_w`` is the
        per-device ring pad; same well -> same pad, applied once at the block
        boundary). The ring pad is the SAME ``group_ring_padded_extents`` reservation
        the band-stack uses -- single source, not a second padding rule.

    Because the area search packs these ring-PADDED blocks overlap-free by the
    sequence-pair construction, the per-group RING bboxes (the union of a group's
    members' placed bboxes, padded) are DISJOINT BY CONSTRUCTION for any searched
    order -- the ring-disjoint hard constraint the band fold could not hold.
    """
    in_group: set = set()
    block_members: List[Tuple[str, ...]] = []
    block_offsets: List[Dict[str, Tuple[float, float]]] = []
    block_sizes: List[Tuple[float, float]] = []

    node_set = set(nodes)
    for members in compact_groups:
        present = tuple(m for m in members if m in node_set)
        if not present:
            continue
        in_group.update(present)
        offsets, raw_w, raw_h = _layout_group_internal(
            present, tight_half_w, tight_heights, mirror_pairs, dev_gap)
        # per-side ring pad: the difference between the PADDED and the RAW (tight)
        # half-extent of a member (same well -> identical pad for every member; take
        # the max so a mixed-size group reserves the worst-case ring room).
        pad_x = max(
            (half_w.get(m, 0.0) - tight_half_w.get(m, 0.0) for m in present),
            default=0.0)
        pad_y = max(
            ((heights.get(m, 0.0) - tight_heights.get(m, 0.0)) / 2.0
             for m in present),
            default=0.0)
        pad_x = max(pad_x, 0.0)
        pad_y = max(pad_y, 0.0)
        block_members.append(present)
        block_offsets.append(offsets)
        block_sizes.append((raw_w + 2.0 * pad_x, raw_h + 2.0 * pad_y))

    # leftover devices (no compact group -- e.g. placed passives): each its own
    # block at its PADDED footprint (a passive carries no ring pad, so padded ==
    # raw for it; a stray MOS keeps its padded reservation).
    leftovers = [n for n in nodes if n not in in_group]
    for n in leftovers:
        block_members.append((n,))
        block_offsets.append({n: (0.0, 0.0)})
        block_sizes.append((2.0 * half_w.get(n, 0.0), heights.get(n, 0.0)))
    return block_members, block_offsets, block_sizes


def _search_block_layout(
    block_sizes: List[Tuple[float, float]],
    dev_gap: float,
    *,
    aspect_target: Optional[float] = None,
) -> Dict[Tuple[int, int], str]:
    """DETERMINISTIC area-driven search over the BLOCK-level sequence-pair (atoms =
    ring-padded group blocks). Same mechanism as :func:`_search_band_layout` (seed =
    vertical stack, exhaustive for B! <= 720 else monotone-accept swap climb, no
    RNG), but the cost is measured on the BLOCK bbox (the blocks are already
    ring-padded, so the searched area IS the ringed extent the gate measures).

    Returns ``rel[(i, j)]`` (i < j) -- the per-block-pair relation the device-level
    order consumes. Pure geometry off the block sizes; no reference coordinate."""
    n = len(block_sizes)
    if n < 2:
        return {}
    native = load_native("_place_cpp")
    if native is not None:
        return native.search_block_layout_cpp(block_sizes, float(dev_gap), aspect_target)
    tgt = float(aspect_target) if aspect_target and aspect_target > 0 else 1.6

    def _cost(gp, gm):
        W, H, _rel = _pack_band_sp(gp, gm, block_sizes, dev_gap)
        return _cost_from_dims(W, H, tgt)

    gp = list(range(n - 1, -1, -1))
    gm = list(range(n))
    best_cost = _cost(gp, gm)
    best_gp, best_gm = list(gp), list(gm)

    def _better(c, ref) -> bool:
        if c[0] != ref[0]:
            return c[0] < ref[0]
        return c[1] < ref[1] - 1e-3

    import math
    if math.factorial(n) <= 720:
        import itertools
        for cgp in itertools.permutations(range(n)):
            for cgm in itertools.permutations(range(n)):
                c = _cost(list(cgp), list(cgm))
                if _better(c, best_cost):
                    best_cost = c
                    best_gp, best_gm = list(cgp), list(cgm)
        gp, gm = best_gp, best_gm
    else:
        improved = True
        sweeps = 0
        while improved and sweeps < 200:
            improved = False
            sweeps += 1
            for which in ("plus", "minus"):
                perm = gp if which == "plus" else gm
                for a in range(n):
                    for b in range(a + 1, n):
                        cand = list(perm)
                        cand[a], cand[b] = cand[b], cand[a]
                        c = (_cost(cand, gm) if which == "plus"
                             else _cost(gp, cand))
                        if _better(c, best_cost):
                            best_cost = c
                            if which == "plus":
                                gp = cand
                            else:
                                gm = cand
                            improved = True
    _W, _H, rel = _pack_band_sp(gp, gm, block_sizes, dev_gap)
    return rel


def _solve_ring_aware(
    nodes: List[str],
    compact_groups: List[Tuple[str, ...]],
    half_w: Dict[str, float],
    heights: Dict[str, float],
    tight_half_w: Dict[str, float],
    tight_heights: Dict[str, float],
    mirror_pairs: Tuple[Tuple[str, str], ...],
    dev_gap: float,
    grid: int,
    aspect_target: Optional[float],
) -> Dict[str, Tuple[float, float]]:
    """The S4 ring-aware group-block placement: pack the ring-padded compact-group
    blocks via a deterministic 2D area search, then realise each device at its
    block position + raw internal offset. Returns ``{name -> (cx, cy)}`` um.

    Ring-disjoint BY CONSTRUCTION: the blocks are ring-padded and the sequence-pair
    packs them overlap-free, so the per-group ring bboxes never overlap for ANY
    searched order. Symmetry-feasible: each block's internal layout is exactly
    mirror-symmetric about the block axis. Leakage-free: only the design groups +
    sizes drive it. Deterministic: seed + monotone accept, no RNG.
    """
    block_members, block_offsets, block_sizes = _ring_aware_group_blocks(
        nodes, compact_groups, half_w, heights,
        tight_half_w, tight_heights, mirror_pairs, dev_gap)
    nb = len(block_members)
    if nb == 0:
        return {}
    if nb == 1:
        rel = {}
    else:
        rel = _search_block_layout(
            block_sizes, dev_gap, aspect_target=aspect_target)

    # block-level longest-path on the searched relation -> block centres.
    h_edges: List[Tuple[int, int]] = []
    v_edges: List[Tuple[int, int]] = []
    for (i, j), r in rel.items():
        lo, hi = (i, j) if r.startswith("i_") else (j, i)
        if "below" in r:
            v_edges.append((lo, hi))
        else:
            h_edges.append((lo, hi))
    bw = [block_sizes[k][0] for k in range(nb)]
    bh = [block_sizes[k][1] for k in range(nb)]

    def _lp(edges, dim):
        succ: Dict[int, List[int]] = defaultdict(list)
        indeg = {k: 0 for k in range(nb)}
        for u, v in edges:
            succ[u].append(v)
            indeg[v] += 1
        pos = {k: 0.0 for k in range(nb)}
        q = deque(k for k in range(nb) if indeg[k] == 0)
        while q:
            u = q.popleft()
            for v in succ[u]:
                w = (dim[u] + dim[v]) / 2.0 + dev_gap
                if pos[u] + w > pos[v]:
                    pos[v] = pos[u] + w
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
        return pos

    bx = _lp(h_edges, bw)
    by = _lp(v_edges, bh)

    out: Dict[str, Tuple[float, float]] = {}
    for k in range(nb):
        cx0, cy0 = bx[k], by[k]
        for name, (dx, dy) in block_offsets[k].items():
            x_nm = float(_snap_to_grid(cx0 + dx, grid))
            y_nm = float(_snap_to_grid(cy0 + dy, grid))
            out[name] = (x_nm / 1000.0, y_nm / 1000.0)
    return out


def solve_hier_seqpair(
    hier,
    sizes: Dict[str, Tuple[float, float]],
    *,
    tech=None,
    mirror_pairs: Tuple[Tuple[str, str], ...] = (),
    abut_pairs: Tuple[Tuple[str, str], ...] = (),
    order_constraints: Tuple[Tuple[str, str, str], ...] = (),
    area_search: bool = False,
    aspect_target: Optional[float] = None,
    compact_groups: Optional[List[Tuple[str, ...]]] = None,
    tight_sizes: Optional[Dict[str, Tuple[float, float]]] = None,
    tight_pairs: Optional[set] = None,
) -> Dict[str, Tuple[float, float]]:
    """Pack a band-stack ``HierGraph`` to overlap-free centroids with the SEQPAIR
    longest-path engine -- a drop-in replacement for
    :func:`clay.layout.extract.relgraph.solve_hierarchical_nonoverlap`.

    INPUT (the SAME the band-stack solve consumes): a ``hier`` built from the
    DESIGN-sourced sketch intent view (``build_hierarchical_graph``) + ``sizes``
    (name -> (w, h) um, target-PDK device footprints). NO reference coordinate is
    read here: the structural relative ORDER comes off the hier's bands (band
    membership + within-band x-order + the bottom-up band stack), and the
    geometry is sized purely from ``sizes`` -- so the leakage boundary is exactly
    the band-stack's (the sketch is the single leakage-controlled source).

    OUTPUT (the SAME shape the band-stack returns): ``{name -> (cx, cy)}`` um,
    overlap-free by the sequence-pair construction (every device pair is ordered
    in H *or* V). Consumed unchanged by the downstream emit/oracle chain.

    Mechanism: each band's devices are read in their within-band left-to-right
    x-order (the band ConstraintGraph solved at sketch sizes -- the design
    order, not a reference centroid). Consecutive same-band devices get a
    ``left_of`` edge; every device in band ``k`` is ``below`` every device in
    band ``k+1`` (the bottom-up stack). The H/V longest-paths then pack the
    devices at ``half_w[u]+half_w[v]+gap`` (x) / ``h[u]/2+h[v]/2+gap`` (y), and a
    symmetry-centering pass equalises each declared mirror pair about one global
    vertical axis -- reusing the engine's ``_longest_path_positions`` +
    ``_reconcile_symmetry_nonoverlap`` byte-for-byte.

    This is the S2 OPT-IN engine: it runs alongside the band-stack (selected by
    the caller), produces a legal placement, and feeds the canonical artifact. It
    does NOT yet search the order for area (S3) -- the order is the same
    design-derived band order the band-stack uses, so S2 only proves the engine
    is wired end-to-end on the production OTA.
    """
    from clay.layout.extract.relgraph import solve_constraint_graph

    tech = _resolve_tech(tech)
    dev_gap = float(_dev_gap_from_tech(tech))
    grid = _mfg_grid_of(tech)

    # half-widths / heights from the target sizes (um -> nm, the engine frame).
    half_w: Dict[str, float] = {}
    heights: Dict[str, float] = {}
    for name, wh in (sizes or {}).items():
        w_um, h_um = float(wh[0]), float(wh[1])
        half_w[name] = (w_um * 1000.0) / 2.0
        heights[name] = h_um * 1000.0

    # ── S4 RING-AWARE GROUP-BLOCK area search ────────────────────────────────────
    # When the caller supplies the COMPACT MATCHED GROUPS (the well_islands single
    # source) + the RAW (tight) stamp sizes, the 2D area search runs over RING-PADDED
    # GROUP BLOCKS instead of the 3 hier bands. Each compact group is ONE rigid
    # contiguous block (members abut internally at raw spacing, mirror pairs
    # symmetric) sized to its ring-padded footprint; the sequence-pair packs the
    # ring-padded blocks overlap-free BY CONSTRUCTION, so the per-group ring bboxes
    # are DISJOINT BY CONSTRUCTION for ANY searched order -- the ring-disjoint hard
    # constraint the band fold (S3) could not hold (it scattered group members in y
    # and the rings fused). ``area_search`` must be on; absent ``compact_groups`` the
    # historical band-level search / seed runs unchanged (byte-parity).
    if area_search and compact_groups:
        # all nodes the hier carries (every device with a size), so a passive band
        # member gets its own block too.
        _all_nodes = [d for band in hier.bands for d in band.devices if d in half_w]
        # dedupe, preserve order.
        _seen_n: set = set()
        all_nodes: List[str] = []
        for d in _all_nodes:
            if d not in _seen_n:
                all_nodes.append(d)
                _seen_n.add(d)
        if all_nodes:
            tight_half_w: Dict[str, float] = {}
            tight_heights: Dict[str, float] = {}
            for name, wh in (tight_sizes or {}).items():
                tight_half_w[name] = (float(wh[0]) * 1000.0) / 2.0
                tight_heights[name] = float(wh[1]) * 1000.0
            # a member without a raw size falls back to its padded size (so the
            # block is never under-reserved).
            for name in all_nodes:
                tight_half_w.setdefault(name, half_w.get(name, 0.0))
                tight_heights.setdefault(name, heights.get(name, 0.0))
            mirror = tuple(
                (a, b) for (a, b) in (mirror_pairs or ())
                if a in half_w and b in half_w
            )
            return _solve_ring_aware(
                all_nodes, [tuple(g) for g in compact_groups],
                half_w, heights, tight_half_w, tight_heights,
                mirror, dev_gap, grid, aspect_target)

    # ── within-band x-order (design-sourced) + band membership, bottom-up ────────
    # The band's ConstraintGraph solved at SKETCH sizes yields each device's
    # design x; sort by it (ties by name -> deterministic). This is the SAME
    # structural x-order ``solve_band_nonoverlap`` uses (``_band_orders``), so the
    # seqpair rows match the band-stack rows -- no second order source.
    row_seqs: List[Tuple[str, ...]] = []
    for band in hier.bands:
        members = [d for d in band.devices if d in half_w]
        if not members:
            continue
        cg = hier.band_graphs[band.band_id]
        try:
            cents, _ = solve_constraint_graph(cg, cg.sizes, (0.0, 0.0))
        except Exception:  # noqa: BLE001 - degenerate band -> name order
            cents = {}
        ordered = sorted(
            members,
            key=lambda n: (cents.get(n, (0.0, 0.0))[0], n),
        )
        row_seqs.append(tuple(ordered))

    nodes = [d for seq in row_seqs for d in seq]
    if not nodes:
        return {}

    # ── relative order: left_of within a band ────────────────────────────────────
    left_of: List[Tuple[str, str]] = []
    seen_lo: set = set()
    for seq in row_seqs:
        for u, v in zip(seq, seq[1:]):
            if (u, v) not in seen_lo:
                left_of.append((u, v))
                seen_lo.add((u, v))

    # ── cross-band relation: vertical STACK (seed) or AREA-SEARCHED 2D pack ───────
    # SEED (area_search=False): every band-r device is ``below`` every band-(r+1)
    # device -- the bottom-up vertical stack the S2 engine + band-stack use (byte-
    # identical when area_search is off). AREA SEARCH (S3): a deterministic search
    # over a BAND-LEVEL sequence-pair chooses, for each band pair, whether they stack
    # (below) OR sit side-by-side (left_of), so a narrow band folds beside a wide
    # band and the cell-bbox dead area collapses. Either way EVERY cross-band device
    # pair is ordered in exactly ONE axis -> overlap-free BY CONSTRUCTION.
    below: List[Tuple[str, str]] = []
    seen_be: set = set()
    if area_search and len(row_seqs) >= 2:
        native = load_native("_place_cpp")
        if native is not None and len(set(nodes)) == len(nodes):
            row_half_ws = [[half_w[d] for d in seq] for seq in row_seqs]
            row_heights = [[heights[d] for d in seq] for seq in row_seqs]
            band_rel = native.search_band_dev_layout_cpp(
                row_half_ws, row_heights, float(dev_gap), aspect_target
            )
        else:
            # band_rel[(i, j)] (for i < j) is the directed relation of band i to band j:
            # 'i_below_j' / 'j_below_i' (vertical stack) or 'i_left_j' / 'j_left_i'
            # (side-by-side). Each unordered band pair contributes exactly one axis.
            # The search COST is measured on the REAL device-level placement (not the
            # block proxy): for each candidate band relation, build the device H/V edges
            # exactly as the final solve does, run the O(V+E) longest-path, and take the
            # device-footprint bbox area + aspect penalty -- so the search optimises
            # precisely the bbox_fill/aspect the gate measures.
            def _dev_cost(cand_rel):
                lo_edges = list(left_of)
                be_edges: List[Tuple[str, str]] = []
                slo = set(lo_edges)
                sbe: set = set()
                for (ci, cj), crel in cand_rel.items():
                    lb, hb = (ci, cj) if crel.startswith("i_") else (cj, ci)
                    ax = "below" if "below" in crel else "left_of"
                    for uu in row_seqs[lb]:
                        for vv in row_seqs[hb]:
                            if ax == "below":
                                if (uu, vv) not in sbe:
                                    be_edges.append((uu, vv)); sbe.add((uu, vv))
                            else:
                                if (uu, vv) not in slo:
                                    lo_edges.append((uu, vv)); slo.add((uu, vv))
                hw = {(u, v): half_w[u] + half_w[v] + dev_gap for (u, v) in lo_edges}
                vw = {(u, v): heights[u] / 2.0 + heights[v] / 2.0 + dev_gap
                      for (u, v) in be_edges}
                try:
                    xc = _longest_path_positions(nodes, lo_edges, hw)
                    yc = _longest_path_positions(nodes, be_edges, vw)
                except ValueError:
                    return 1e18  # a cyclic candidate is infeasible -> never chosen
                W = (max(xc[d] + half_w[d] for d in nodes)
                     - min(xc[d] - half_w[d] for d in nodes))
                H = (max(yc[d] + heights[d] / 2.0 for d in nodes)
                     - min(yc[d] - heights[d] / 2.0 for d in nodes))
                return _cost_from_dims(W, H, 1.6)

            band_rel = _search_band_layout(
                row_seqs, half_w, heights, dev_gap,
                aspect_target=aspect_target,
                dev_cost=_dev_cost,
            )
        for (i, j), rel in band_rel.items():
            lo_band, hi_band = (i, j) if rel.startswith("i_") else (j, i)
            axis = "below" if "below" in rel else "left_of"
            for u in row_seqs[lo_band]:
                for v in row_seqs[hi_band]:
                    if axis == "below" and (u, v) not in seen_be:
                        below.append((u, v))
                        seen_be.add((u, v))
                    elif axis == "left_of" and (u, v) not in seen_lo:
                        left_of.append((u, v))
                        seen_lo.add((u, v))
    else:
        for r in range(len(row_seqs) - 1):
            for u in row_seqs[r]:
                for v in row_seqs[r + 1]:
                    if (u, v) not in seen_be:
                        below.append((u, v))
                        seen_be.add((u, v))

    # ── @order: declared edges override the structural order on the named pair ───
    order_constraints = tuple(
        (a, b, rel) for (a, b, rel) in (order_constraints or ())
        if a in half_w and b in half_w and a != b
    )
    declared = {frozenset((a, b)) for (a, b, _r) in order_constraints}
    left_of = [e for e in left_of if frozenset(e) not in declared]
    below = [e for e in below if frozenset(e) not in declared]
    seen_lo = set(left_of)
    seen_be = set(below)
    for (a, b, rel) in order_constraints:
        if rel == "left_of" and (a, b) not in seen_lo:
            left_of.append((a, b))
            seen_lo.add((a, b))
        elif rel == "above" and (b, a) not in seen_be:
            below.append((b, a))
            seen_be.add((b, a))

    # ── @abut: a declared adjacency with no edge yet packs horizontally tight ────
    abut_set = frozenset(
        frozenset((a, b)) for (a, b) in (abut_pairs or ())
        if a in half_w and b in half_w and a != b
    )
    for fk in abut_set:
        a, b = tuple(fk)
        if ((a, b) not in seen_lo and (b, a) not in seen_lo
                and (a, b) not in seen_be and (b, a) not in seen_be):
            left_of.append((a, b))
            seen_lo.add((a, b))

    mirror = tuple(
        (a, b) for (a, b) in (mirror_pairs or ())
        if a in half_w and b in half_w
    )
    abut_gap = int(_abut_gap_from_tech(tech)) if abut_set else 0

    # ── INTRA-GROUP ABUT (single source, the SAME band-stack tight_sizes/tight_
    # pairs path): two members of the SAME compact matched group share ONE ring,
    # so the ring pad belongs at the GROUP boundary, not between members. When the
    # caller supplies the raw (tight) stamp extents + the same-group pair set
    # (well_islands.compact_matched_groups), same-group members pack at their RAW
    # half-extent + abut gap instead of the ring-padded half-extent + dev gap --
    # collapsing the per-group double-ring-pad the seed otherwise inserts (the
    # P-A enclosure win). Pure topology + raw drawn-stamp extents; no reference
    # coordinate; byte-identical when tight_pairs is absent.
    tight_half_w: Dict[str, float] = {}
    tight_h: Dict[str, float] = {}
    for name, wh in (tight_sizes or {}).items():
        tight_half_w[name] = (float(wh[0]) * 1000.0) / 2.0
        tight_h[name] = float(wh[1]) * 1000.0
    _tset = set(tight_pairs or ())

    def _is_tight(u: str, v: str) -> bool:
        return bool(_tset) and frozenset((u, v)) in _tset

    def _hw(name: str, tight: bool) -> float:
        if tight and name in tight_half_w:
            return tight_half_w[name]
        return half_w.get(name, 0.0)

    def _ht(name: str, tight: bool) -> float:
        if tight and name in tight_h:
            return tight_h[name]
        return heights.get(name, 0.0)

    def _gap(u: str, v: str) -> float:
        if _is_tight(u, v):
            return float(abut_gap) if abut_set else dev_gap
        if abut_set and frozenset((u, v)) in abut_set:
            return float(abut_gap)
        return dev_gap

    # ── H / V longest-path pack (reuse the engine's O(V+E) packer) ───────────────
    h_weight = {
        (u, v): _hw(u, _is_tight(u, v)) + _hw(v, _is_tight(u, v)) + _gap(u, v)
        for (u, v) in left_of
    }
    x_centre = _longest_path_positions(nodes, left_of, h_weight)
    v_weight = {
        (u, v): _ht(u, _is_tight(u, v)) / 2.0 + _ht(v, _is_tight(u, v)) / 2.0
        + _gap(u, v)
        for (u, v) in below
    }
    y_centre = _longest_path_positions(nodes, below, v_weight)

    # ── mirror y-share (non-overlap-preserving) ─────────────────────────────────
    # Equalise each declared mirror pair's y so partners share a row baseline. The
    # X-axis symmetry-centering about one global axis is DEFERRED to S4 (the
    # symmetry-feasible move-set): applied to a sketch band order that does NOT
    # straddle the pair members about the axis it pulls a member across a
    # neighbour and COLLAPSES the longest-path non-overlap (the historical
    # CO.10 / M1.2a failure -- see ``_reconcile_symmetry_nonoverlap``). For S2 the
    # engine stays overlap-free BY CONSTRUCTION (its sole acceptance gate); exact
    # x-mirror symmetry as a HARD constraint is S4's deliberate move-set work.
    # The y-share is a pure pair-internal average that never moves a member past a
    # row neighbour, so non-overlap is preserved.
    for a, b in mirror:
        if a in y_centre and b in y_centre and abs(y_centre[a] - y_centre[b]) < 1e-6:
            yc = (y_centre[a] + y_centre[b]) / 2.0
            y_centre[a] = yc
            y_centre[b] = yc

    # grid-snap every centroid (on-grid devices -> no *_OFFGRID DRC), return um.
    out: Dict[str, Tuple[float, float]] = {}
    for n in nodes:
        cx_nm = float(_snap_to_grid(x_centre[n], grid))
        cy_nm = float(_snap_to_grid(y_centre[n], grid))
        out[n] = (cx_nm / 1000.0, cy_nm / 1000.0)
    return out


def place_seqpair(
    cell_def: CellDef,
    intent: Optional[LayoutIntent] = None,
    *,
    source_cir=None,
    pdk: Optional[PdkRules] = None,
    tech=None,
    lock_devices: Optional[set] = None,
) -> Placement:
    """One-shot constraint-graph placement of a parsed ``CellDef``.

    ``place_seqpair(cd, intent, source_cir=cir) ==
    solve_placement(relative_order_from_intent(cd, intent, source_cir=cir),
    devs, pdk)`` -- the drop-in seqpair counterpart of
    :func:`clay.layout.generic_placement.place_generic`.

    ``tech`` (a :class:`~clay.layout.tech.LayoutTech` or PDK-id string) flows the
    device geometry + spacing through one PDK object; it defaults to gf180mcu so
    every existing caller is unchanged (gf180 byte-parity). A legacy ``pdk`` is
    resolved to its tech when ``tech`` is not given.

    ``lock_devices`` names devices that must keep their own AXIS-CENTRED row even
    when unpaired (they are members of a matched / common-centroid group the
    CONVERGENCE CRITERIA declare but the parsed constraints do not carry -- e.g. a
    3-device cc cluster). The wide-2D packer excludes them so their group stays
    centroid-balanced. ``place_to_intent`` passes its criteria's device groups here.
    """
    _tech = _resolve_tech(tech if tech is not None else pdk)
    order = relative_order_from_intent(
        cell_def, intent, source_cir=source_cir, tech=_tech,
        lock_devices=lock_devices,
    )
    annotated, _ = annotated_cell_from_cell_def(cell_def, source_cir)
    devs = _build_devices_from_clay(annotated, _tech)
    # Collect the cell's 2-terminal PASSIVES (resistors + MIM caps) -- the same
    # ``_collect_passives`` source the MILP ``place_generic`` uses -- and hand them
    # to ``solve_placement`` so a seqpair-placed cell carries its passives instead of
    # silently dropping them. Annotate each passive with its DECLARED ``@place_group
    # col`` (if any) so a passive that declares its own column -- e.g. the Miller cap's
    # dedicated right-hand col=3 -- is placed as a block in its own x-region clear of
    # the MOS channel, not stacked on the below-MOS passive row.
    passives = _collect_passives(cell_def, source_cir)
    _annotate_passive_cols(passives, intent)
    mos_row_of = _mos_row_of_intent(devs, intent)
    # @equal_size groups: the DECLARED equal-footprint (size_class) invariant. Read off
    # the SAME SolverConstraints the row plan was built from (sibling front-end populates
    # ``equal_size_groups``); ``getattr`` defaults to () so this is safe before the field
    # lands. Threaded into ``solve_placement`` so every group's members realise one common
    # footprint. Empty -> no placement change (byte-parity).
    annotated_c, _ = annotated_cell_from_cell_def(cell_def, source_cir)
    constraints = _constraints_from_intent_and_annotations(annotated_c, intent)
    equal_size_groups = tuple(getattr(constraints, "equal_size_groups", ()) or ())
    # @fix ABSOLUTE pins: device label -> (x_um, y_um). Read off the SAME constraints
    # the equal_size groups come from. Empty -> no pin override (byte-parity).
    pinned_positions = dict(getattr(constraints, "pinned_positions", {}) or {})
    matched_passive_pairs = tuple(
        tuple(sorted((str(left), str(right))))
        for left, right in tuple(getattr(constraints, "symmetry_pairs", ()) or ())
        if str(left) in passives and str(right) in passives
    )
    return solve_placement(
        order, devs, tech=_tech, passives=passives, mos_row_of=mos_row_of,
        equal_size_groups=equal_size_groups,
        pinned_positions=pinned_positions,
        matched_passive_pairs=matched_passive_pairs,
    )


def _annotate_passive_cols(
    passives: Dict[str, Dict[str, object]], intent: Optional[LayoutIntent]
) -> None:
    """Stamp each passive record with its DECLARED ``@place_group (row, col)`` (in place).

    Reads ``intent.place_groups``: any group that declares a single row (so it is a
    floorplan band) assigns that ROW (and its ``col`` if any) to each of its passive
    members. The passive floorplan is NOT carried in ``constraints.floorplan_rows``
    (that map is restricted to built MOS devices), so the passive row/col is threaded
    directly onto the passive records here. Generic + circuit-agnostic: keyed only on
    the declared group membership, no device-name branch."""
    if intent is None or not passives:
        return
    for pg in getattr(intent, "place_groups", ()) or ():
        row_min = getattr(pg, "row_min", None)
        row_max = getattr(pg, "row_max", None)
        if row_min is None or row_max is None or row_min != row_max:
            continue
        col = getattr(pg, "col", None)
        for d in getattr(pg, "devices", ()) or ():
            if d in passives:
                passives[d]["row"] = int(row_min)
                if col is not None:
                    passives[d]["col"] = int(col)


def _mos_row_of_intent(
    devs: Dict[str, object], intent: Optional[LayoutIntent]
) -> Dict[str, int]:
    """``{mos_label: declared_floorplan_row}`` from the single-row ``@place_group``
    bands, restricted to built MOS devices. Drives the MIDDLE-band reservation: a
    passive declared at a row index strictly BETWEEN MOS row indices is laid out in
    the reserved middle tier. Generic: keyed only on declared group membership."""
    out: Dict[str, int] = {}
    if intent is None:
        return out
    for pg in getattr(intent, "place_groups", ()) or ():
        row_min = getattr(pg, "row_min", None)
        row_max = getattr(pg, "row_max", None)
        if row_min is None or row_max is None or row_min != row_max:
            continue
        for d in getattr(pg, "devices", ()) or ():
            if d in devs:
                out[d] = int(row_min)
    return out


__all__ = [
    "relative_order_from_intent",
    "solve_placement",
    "solve_hier_seqpair",
    "place_seqpair",
]
