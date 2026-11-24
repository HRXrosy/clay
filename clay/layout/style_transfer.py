"""Layout STYLE TRANSFER for Clay (round seqpair-style, Agent B).

Transfer a REFERENCE layout's relative-order STYLE onto a NEW cell via graph
alignment -- a fresh reimplementation of the LaSTPlace (TCAD) layout-style-transfer
paradigm. The paper's trained GNN is NOT copied: the node matcher is a CLASSICAL
Tversky-index node-similarity (LaSTPlace Eq. 11) over device features, and the
reference is a COMMITTED Clay layout (any placed cell), so no training data is
needed. This is Clay's "learn from verified layouts" flywheel: a known-good
placement's relative order becomes a transferable prior for migration to a new
process / a structurally similar topology.

Two public functions, built on the FROZEN interface in
:mod:`clay.layout.seqpair_model` (:class:`RelativeOrder`,
:class:`StyleTransferResult`, :func:`constraint_graph_distance`):

* :func:`order_from_placement` -- derive a :class:`RelativeOrder` from ANY placed
  cell (a :class:`clay.layout.generic_placement.Placement`, or any object exposing
  ``device_centroids()`` + per-device bbox). This lets an existing placement serve
  as a style reference WITHOUT depending on the constraint-graph placer (Agent A).
* :func:`transfer_style` -- graph-align a NEW cell's devices to a reference order's
  devices (Tversky node-similarity over kind + neighbour-overlap + role, with
  greedy progressive matching) and COPY the reference's relative relations
  (left_of / below / mirror) onto the matched target devices -> a transferred
  :class:`RelativeOrder`. Unmatched target devices carry no transferred relation
  (the caller's own intent order fills them).

The transferred order composes with the placer through the frozen type:
``solve_placement(transfer_style(target, reference).order)``.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .seqpair_model import RelativeOrder, StyleTransferResult

Bbox = Tuple[float, float, float, float]


# === 1. order_from_placement ==================================================


def _placement_bboxes(placement: object) -> Dict[str, Bbox]:
    """Extract ``{device_label: (x0, y0, x1, y1)}`` from any placed cell.

    Accepts a :class:`clay.layout.generic_placement.Placement` (its ``devices``
    map carries a ``.bbox`` per :class:`PlacedDevice`), or any object exposing a
    ``devices`` mapping whose values have a ``bbox`` attribute, or a plain mapping
    ``{label: (x0,y0,x1,y1)}``. Returns an empty dict when nothing is extractable.
    """
    # plain mapping label -> bbox
    devs = getattr(placement, "devices", None)
    if devs is None and isinstance(placement, Mapping):
        out: Dict[str, Bbox] = {}
        for lbl, b in placement.items():
            if isinstance(b, (tuple, list)) and len(b) == 4:
                out[str(lbl)] = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
        return out
    if devs is None:
        return {}
    out = {}
    for lbl, pd in devs.items():
        b = getattr(pd, "bbox", None)
        if b is None and isinstance(pd, (tuple, list)) and len(pd) == 4:
            b = pd
        if isinstance(b, (tuple, list)) and len(b) == 4:
            out[str(lbl)] = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
    return out


def _device_kinds_of(placement: object, bboxes: Mapping[str, Bbox]) -> Dict[str, str]:
    """``{label: kind}`` for the placed devices (``kind`` from the PlacedDevice, or
    inferred from the ``MN``/``MP`` name prefix when the placement carries none)."""
    kinds: Dict[str, str] = {}
    devs = getattr(placement, "devices", None) or {}
    for lbl in bboxes:
        kind = None
        pd = devs.get(lbl) if isinstance(devs, Mapping) else None
        if pd is not None:
            kind = getattr(pd, "kind", None)
        kinds[lbl] = str(kind) if kind else _kind_from_name(str(lbl))
    return kinds


def _kind_from_name(name: str) -> str:
    up = name.strip().upper().lstrip("X")
    if up.startswith("M"):
        rest = up[1:].lstrip("_")
        if rest.startswith("P"):
            return "pmos"
        if rest.startswith("N"):
            return "nmos"
    return "dev"


def _row_band(centroids: Mapping[str, Tuple[float, float]], tol: float) -> List[List[str]]:
    """Group devices into rows by centroid-y (greedy clustering within ``tol``)."""
    by_y = sorted(centroids.items(), key=lambda kv: kv[1][1])
    rows: List[List[str]] = []
    cur: List[str] = []
    cur_y: Optional[float] = None
    for lbl, (_, y) in by_y:
        if cur_y is None or abs(y - cur_y) <= tol:
            cur.append(lbl)
            cur_y = y if cur_y is None else (cur_y + y) / 2.0
        else:
            rows.append(cur)
            cur = [lbl]
            cur_y = y
    if cur:
        rows.append(cur)
    return rows


def order_from_placement(
    placement: object, *, row_tol: Optional[float] = None
) -> RelativeOrder:
    """Derive a :class:`RelativeOrder` from a placed cell.

    For each ordered device pair (u, v):

    * a ``left_of`` relation is emitted when u's centroid-x is strictly less than
      v's by more than HALF the sum of their widths (i.e. they do not overlap in x
      and u sits to the left) -- the strict horizontal constraint;
    * a ``below`` relation is emitted when u's centroid-y is strictly less than v's
      (u sits below v) -- the strict vertical constraint;
    * ``mirror_pairs`` are inferred as same-row device pairs of the SAME kind whose
      centroid-x are symmetric about the cell's mean axis (``|x_u - axis| ~
      |x_v - axis|`` within the device half-width) -- the matched / common-centroid
      members the placement centred about the vertical axis.

    ``rows`` band the devices by centroid-y (so a downstream placer can re-band).
    This makes ANY existing placement usable as a style reference, independent of
    how it was solved.
    """
    bboxes = _placement_bboxes(placement)
    labels = sorted(bboxes)
    if not bboxes:
        return RelativeOrder(cell=str(getattr(placement, "cell", "")), devices=())

    centroids = {
        lbl: ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for lbl, b in bboxes.items()
    }
    widths = {lbl: (b[2] - b[0]) for lbl, b in bboxes.items()}
    heights = {lbl: (b[3] - b[1]) for lbl, b in bboxes.items()}
    kinds = _device_kinds_of(placement, bboxes)

    # half-width / half-height tolerance for "strictly left / below".
    left_of: List[Tuple[str, str]] = []
    below: List[Tuple[str, str]] = []
    for i, u in enumerate(labels):
        ux, uy = centroids[u]
        for v in labels[i + 1 :]:
            vx, vy = centroids[v]
            half_w = (widths[u] + widths[v]) / 2.0
            if ux + half_w < vx:
                left_of.append((u, v))
            elif vx + half_w < ux:
                left_of.append((v, u))
            half_h = (heights[u] + heights[v]) / 2.0
            if uy + half_h < vy:
                below.append((u, v))
            elif vy + half_h < uy:
                below.append((v, u))

    # rows by centroid-y.
    avg_h = sum(heights.values()) / max(1, len(heights))
    tol = row_tol if row_tol is not None else avg_h * 0.75
    rows = _row_band(centroids, tol)

    # mirror pairs: same row, same kind, symmetric about the mean x axis.
    axis_x = sum(x for x, _ in centroids.values()) / len(centroids)
    mirror_pairs: List[Tuple[str, str]] = []
    used: set = set()
    for row in rows:
        members = sorted(row, key=lambda l: centroids[l][0])
        for ai in range(len(members)):
            a = members[ai]
            if a in used:
                continue
            ax = centroids[a][0]
            for bi in range(len(members) - 1, ai, -1):
                b = members[bi]
                if b in used or kinds[a] != kinds[b]:
                    continue
                bx = centroids[b][0]
                offa = ax - axis_x
                offb = bx - axis_x
                # straddle the axis on opposite sides, equidistant within a half-width
                if offa * offb >= 0:
                    continue
                half_w = (widths[a] + widths[b]) / 2.0
                if abs(abs(offa) - abs(offb)) <= max(half_w, 1.0):
                    left, right = (a, b) if ax <= bx else (b, a)
                    mirror_pairs.append((left, right))
                    used.add(a)
                    used.add(b)
                    break

    return RelativeOrder(
        cell=str(getattr(placement, "cell", "")),
        devices=tuple(labels),
        left_of=tuple(left_of),
        below=tuple(below),
        rows=tuple(tuple(r) for r in rows),
        mirror_pairs=tuple(mirror_pairs),
        axis_x=int(round(axis_x)),
    )


# === 2. transfer_style (Tversky graph alignment) =============================


def _neighbour_sets(
    devices: Sequence[str],
    neighbors: Optional[Mapping[str, Iterable[str]]],
) -> Dict[str, set]:
    out: Dict[str, set] = {d: set() for d in devices}
    if not neighbors:
        return out
    for d in devices:
        out[d] = {str(n) for n in (neighbors.get(d, ()) or ())}
    return out


def _tversky(
    a_feat: set,
    b_feat: set,
    *,
    alpha: float = 0.5,
    beta: float = 0.5,
) -> float:
    """Tversky index (LaSTPlace Eq. 11) between two feature sets, in [0, 1].

    ``T = |A∩B| / (|A∩B| + α|A−B| + β|B−A|)``. With α=β=0.5 this is the Dice / F1
    overlap; the asymmetric form lets the caller weight unmatched features. Empty
    sets score 1.0 (two featureless nodes are trivially compatible)."""
    if not a_feat and not b_feat:
        return 1.0
    inter = len(a_feat & b_feat)
    a_only = len(a_feat - b_feat)
    b_only = len(b_feat - a_feat)
    denom = inter + alpha * a_only + beta * b_only
    if denom <= 0:
        return 0.0
    return inter / denom


def _node_similarity(
    t: str,
    r: str,
    *,
    t_kind: str,
    r_kind: str,
    t_role: Optional[str],
    r_role: Optional[str],
    t_nb: set,
    r_nb: set,
    aligned: Mapping[str, str],
) -> float:
    """LaSTPlace-style node similarity between target node ``t`` and reference node
    ``r``: kind gate (different kind -> 0) times a blend of role match and the
    Tversky overlap of their ALREADY-ALIGNED neighbour sets (progressive
    alignment: only neighbours both endpoints have already mapped count as shared
    structural context, so matching grows from the most confident anchors out)."""
    if t_kind != r_kind:
        return 0.0
    # neighbour structural overlap, mapped through the current alignment: the set
    # of REFERENCE nodes the target's already-aligned neighbours map to, compared
    # against the reference node's own already-aligned neighbours (LaSTPlace Eq. 11
    # progressive alignment -- only mutually-anchored context counts).
    aligned_refs = set(aligned.values())
    t_mapped = {aligned[n] for n in t_nb if n in aligned}
    r_anchored = r_nb & aligned_refs
    shared_struct = _tversky(t_mapped, r_anchored)
    # raw neighbour-count compatibility (a soft prior before any anchor exists).
    deg = _tversky({f"d{len(t_nb)}"}, {f"d{len(r_nb)}"}) if (t_nb or r_nb) else 1.0
    role_match = 1.0
    if t_role is not None and r_role is not None:
        role_match = 1.0 if str(t_role) == str(r_role) else 0.4
    # tie-break bonus: when the SAME device label exists on both sides (a same-cell
    # / isomorph re-derivation), prefer the identity anchor. Small enough not to
    # override a strong structural disagreement, large enough to break ties between
    # two structurally-symmetric candidates (e.g. a matched diff-pair M1/M2).
    identity = 1.0 if t == r else 0.0
    # raw shared-neighbour-name overlap (same-cell case): rewards keeping a device
    # next to the same-named neighbours it had in the reference.
    name_overlap = _tversky(t_nb, r_nb) if (t_nb or r_nb) else 1.0
    # kind already matched (0.5 base); blend structural overlap, degree, role, and
    # the identity / name tie-breakers.
    return (
        0.5
        + 0.20 * shared_struct
        + 0.05 * deg
        + 0.05 * role_match
        - 0.05 * (1 - role_match)
        + 0.10 * identity
        + 0.10 * name_overlap
    )


def _kinds_for(
    devices: Sequence[str], kinds: Optional[Mapping[str, str]]
) -> Dict[str, str]:
    if kinds:
        return {d: str(kinds.get(d, _kind_from_name(d))) for d in devices}
    return {d: _kind_from_name(d) for d in devices}


def transfer_style(
    target_devices: Sequence[str],
    reference: RelativeOrder,
    *,
    target_neighbors: Optional[Mapping[str, Iterable[str]]] = None,
    target_kinds: Optional[Mapping[str, str]] = None,
    reference_neighbors: Optional[Mapping[str, Iterable[str]]] = None,
    reference_kinds: Optional[Mapping[str, str]] = None,
    role_hint: Optional[Mapping[str, str]] = None,
    threshold: float = 0.5,
) -> StyleTransferResult:
    """Transfer ``reference``'s relative-order STYLE onto ``target_devices``.

    1. **Graph alignment** (Tversky node-similarity, LaSTPlace Eq. 11): each target
       device is matched to a reference device by KIND (hard gate) + neighbour-set
       overlap (shared already-aligned neighbours) + role (if ``role_hint`` maps
       both). Matching is GREEDY + PROGRESSIVE: the highest-similarity unmatched
       pair is locked in each step (so confident anchors seed structural context
       for the next), stopping when the best remaining similarity falls below
       ``threshold``.
    2. **Relation copy**: for each matched ``target <- reference`` pair, the
       reference's ``left_of`` / ``below`` / ``mirror`` relations between two
       reference devices are copied onto the corresponding target devices. Target
       devices with no match get no transferred relation (the caller's intent
       order fills them).

    Returns a :class:`StyleTransferResult` carrying the transferred
    :class:`RelativeOrder`, the ``(target, reference)`` matches, the coverage
    (fraction of target devices matched), and a note.
    """
    targets = [str(d) for d in target_devices]
    refs = list(reference.devices)
    t_kinds = _kinds_for(targets, target_kinds)
    r_kinds = _kinds_for(refs, reference_kinds)

    # reference neighbour sets: from explicit map, else from its own relations.
    if reference_neighbors:
        r_nb = _neighbour_sets(refs, reference_neighbors)
    else:
        r_nb = {d: set() for d in refs}
        for u, v in list(reference.left_of) + list(reference.below):
            if u in r_nb:
                r_nb[u].add(v)
            if v in r_nb:
                r_nb[v].add(u)
        for a, b in reference.mirror_pairs:
            if a in r_nb:
                r_nb[a].add(b)
            if b in r_nb:
                r_nb[b].add(a)
    t_nb = _neighbour_sets(targets, target_neighbors)

    role = {str(k): str(v) for k, v in (role_hint or {}).items()}

    aligned: Dict[str, str] = {}      # target -> reference
    used_refs: set = set()
    remaining_t = set(targets)

    # greedy progressive matching: recompute the best pair each step so the
    # neighbour-overlap term sees the alignment grown so far.
    while remaining_t:
        best: Optional[Tuple[float, str, str]] = None
        for t in remaining_t:
            for r in refs:
                if r in used_refs:
                    continue
                s = _node_similarity(
                    t,
                    r,
                    t_kind=t_kinds[t],
                    r_kind=r_kinds[r],
                    t_role=role.get(t),
                    r_role=role.get(r),
                    t_nb=t_nb[t],
                    r_nb=r_nb[r],
                    aligned=aligned,
                )
                if best is None or s > best[0]:
                    best = (s, t, r)
        if best is None or best[0] < threshold:
            break
        _, bt, br = best
        aligned[bt] = br
        used_refs.add(br)
        remaining_t.discard(bt)

    matched = tuple(sorted(aligned.items()))
    coverage = len(aligned) / len(targets) if targets else 0.0

    # === relation copy ========================================================
    rev = {r: t for t, r in aligned.items()}  # reference -> target

    def _both(u: str, v: str) -> Optional[Tuple[str, str]]:
        if u in rev and v in rev:
            return rev[u], rev[v]
        return None

    new_left: List[Tuple[str, str]] = []
    for u, v in reference.left_of:
        tp = _both(u, v)
        if tp:
            new_left.append(tp)
    new_below: List[Tuple[str, str]] = []
    for u, v in reference.below:
        tp = _both(u, v)
        if tp:
            new_below.append(tp)
    new_mirror: List[Tuple[str, str]] = []
    for u, v in reference.mirror_pairs:
        tp = _both(u, v)
        if tp:
            new_mirror.append(tp)
    new_rows: List[Tuple[str, ...]] = []
    for row in reference.rows:
        mapped = tuple(rev[d] for d in row if d in rev)
        if mapped:
            new_rows.append(mapped)

    transferred = RelativeOrder(
        cell=str(getattr(reference, "cell", "")) + "_styled",
        devices=tuple(targets),
        left_of=tuple(new_left),
        below=tuple(new_below),
        rows=tuple(new_rows),
        mirror_pairs=tuple(new_mirror),
        axis_x=reference.axis_x,
    )

    note = (
        f"Tversky graph-align: matched {len(aligned)}/{len(targets)} target devices "
        f"(coverage {coverage:.2f}); copied "
        f"{len(new_left)} left_of / {len(new_below)} below / "
        f"{len(new_mirror)} mirror relations from reference {reference.cell!r}."
    )
    if len(aligned) < len(targets):
        unmatched = sorted(set(targets) - set(aligned))
        note += f" Unmatched (intent order fills): {unmatched}."

    return StyleTransferResult(
        order=transferred,
        matched=matched,
        coverage=coverage,
        note=note,
    )


__all__ = ["order_from_placement", "transfer_style"]
