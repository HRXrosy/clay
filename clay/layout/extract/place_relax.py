"""Electrostatic/analytic RELAXATION placement backend (P1, 2026-06-12).

WHAT THIS IS
============
An opt-in continuous-relaxation REFINEMENT stage for the one placement
machine (``derive_design_sketch`` -> ``build_hierarchical_graph`` ->
``solve_hierarchical_nonoverlap``), in the architectural lineage of
ePlace/DREAMPlace (smooth wirelength + spreading forces, first-order
optimizer) and DC3 (hard structure enforced by a CHANGE OF VARIABLES, not a
penalty) -- a fresh implementation, pure numpy + stdlib, no external solver
and no torch dependency.

The energy, term by term (all DESIGN data -- see LEAKAGE below):

* SMOOTH WIRELENGTH: per-net log-sum-exp HPWL over the member centroids,
  hypergraph-downweighted (1/(k-1) per k-pin net, the same clique weighting
  ``topo_sketch._unit_connectivity`` uses) so a many-device rail does not
  dominate.
* OVERLAP: pairwise rectified overlap area with the device min-gap margin
  (17-23 devices need no FFT Poisson field; the term is an isolated block
  of ``_energy`` so a density-field implementation can replace it without
  touching the optimizer), annealed by the ePlace-style weight ramp (see
  ``RelaxKnobs.overlap_ramp0``).
* SYMMETRY RIGIDIFICATION (the DC3 completion idea): declared matched /
  mirror / diff partners that share a band are ONE rigid body -- the free
  variables are the body centroids, the member offsets are FIXED at their
  seed values. The symmetric offset is therefore preserved EXACTLY by
  construction; it is never a soft penalty that "usually" holds.
* SOFT STYLE BANDS (the declared softening face this backend exists for):
  each band of the seed's row stack is a y potential WELL (Huber) pulling
  its members back; a member whose connectivity outweighs the well may
  LEAVE its band, and every such deviation is REPORTED in
  ``RelaxReport.band_deviations`` -- honest self-declaration, the
  user/upper layer decides whether to keep it.
* CORRIDOR FORBIDDEN BANDS: a rail reservation row's y strip repels device
  area (the corridor must stay device-free; the projection re-presses the
  seams flush either way).
* OUTLINE TARGET: a one-sided hinge on the smooth outline width/height
  toward the area-ideal outline at the seed's aspect -- the spreading
  pressure that pushes devices out of over-wide rows into another band's
  dead zone (the analytic analog of ePlace's fixed-die density pressure).

OPTIMIZER: Adam plus DECAYING Gaussian gradient noise (a Langevin tail
against the many local minima of a small instance) under a DETERMINISTIC
seed sequence (``knobs.seed + restart_index`` -> PCG64; no clock, no global
RNG), multi-restart with deterministic jitter; the noiseless-energy best of
each restart is kept.

THE RELAXER NEVER SIGNS LEGALITY (the survey's iron rule). Every candidate
(and always the seed itself) is PROJECTED through the existing compaction
press (:func:`clay.layout.extract.compaction.compact_placement` -- the
v0.4.38 legalization machinery, not a second path); the candidate with the
best PROJECTED device outline area (HPWL tie-break, then restart index)
wins. If nothing beats the projected seed, the engine FALLS BACK to it and
says so (``report.selected == 'seed-fallback'``) -- engine='relax' is never
worse than engine='bands' on the selection objective. Downstream the
existing oracle chain (route contract / LVS / DRC) adjudicates; nothing
here issues a verdict.

MIRROR-RIGID CANDIDACY (P1b, 2026-06-13, opt-in)
================================================
``RelaxKnobs.mirror_pairs_rigid`` (or ``derive_design_sketch(
mirror_pairs_rigid=True)`` riding the view) rigidifies declared symmetry
pairs as mirror bodies about a common vertical axis -- the (axis_x, d, y)
change of variables: diff pairs always (cross-row pairs CONVERGE to an
enumerated host row under the soft band face, both rows tried through the
selection objective, the moved member's deviation reported), 2-member
mirror groups same-row only (cross-row honestly noted). The converged
composite insertion is recovered by ONE :func:`compaction.embed_search`
pass over the selected candidate (the same machinery, the chain's S6
channel/keepout guards riding along). Each pair's solved axis lands in
``RelaxReport.mirror_pairs`` -- the placement-side axis supply the routing
Mirror constraint consumes (anchor ``_mirror_route_constraints``). Default
False is byte-identical end to end.

LEAKAGE
=======
Inputs: netlist intent (nets / declared pairs), caller target sizes,
declared style facts (row membership/order as NAMES via the seed sketch),
and a seed placement that MUST be the design's own band solve output
(design-self-produced). No reference coordinate is read; the perturbation
guard rides tests/test_place_relax.py.

WIRING (the engine parameter face): ``derive_design_sketch(engine='relax')``
records the request on the sketch; ``build_hierarchical_graph`` carries the
intent view on ``HierGraph.relax_intent``; ``solve_hierarchical_nonoverlap``
calls :func:`refine_banded_placement` on its own (band+press) output and
parks the :class:`RelaxReport` on ``HierGraph.relax_report``. The default
``engine='bands'`` is byte-identical to the historical chain.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import (
    Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple,
)

import numpy as np

from clay.layout.extract.compaction import (
    CompactionSpec,
    compact_placement,
    embed_search,
)

_XY = Tuple[float, float]
_Bbox = Tuple[float, float, float, float]

# single-sourced rail pseudo-row prefix (relgraph owns it; importing the
# module at call time avoids any import-cycle risk with topo_sketch).
_RAIL_PREFIX = "__rail__"


# --------------------------------------------------------------------------- #
# knobs + report
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RelaxKnobs:
    """The declared parameter face of the relaxation backend. Every default
    is a fixed constant (DETERMINISM: no clock and no entropy source may
    ever feed a default)."""

    iters: int = 400
    restarts: int = 3
    step: float = 0.30                # Adam learning rate (um)
    gamma_um: float = 2.0             # log-sum-exp smoothing (um)
    w_wirelength: float = 1.0
    w_overlap: float = 8.0
    # ePlace-style density-weight RAMP: the overlap weight starts at
    # ``w_overlap * overlap_ramp0`` and anneals quadratically to the full
    # ``w_overlap`` -- a full-stiffness pairwise contact force from iteration
    # 0 freezes the legal seed solid (the x-resistance of a packed row pair
    # scales with its y-overlap length), so migration happens in the soft
    # early phase and separation in the stiff late phase.
    overlap_ramp0: float = 0.05
    w_band: float = 0.6               # style-band well weight (per um, Huber)
    band_huber_um: float = 3.0        # well quadratic->linear knee
    w_corridor: float = 8.0
    w_outline: float = 0.5
    # outline target (um): None = the area-ideal outline at the seed aspect
    # (sqrt(total device area * aspect), sqrt(area / aspect)), floored at the
    # widest/tallest single device.
    outline_target: Optional[_XY] = None
    noise0_um: float = 0.5            # initial Langevin noise sigma
    noise_decay_frac: float = 0.7     # noise reaches 0 at this iter fraction
    jitter_um: float = 3.0            # restart >0 initial jitter amplitude
    seed: int = 20260612              # deterministic base seed (constant)
    band_dev_tol_um: float = 2.0      # in-band drift worth a note
    # MIRROR-RIGID CANDIDACY (P1b, opt-in -- default False is byte-identical):
    # declared symmetry pairs become mirror RIGID BODIES about a common
    # vertical axis -- the (axis_x, d, y) change of variables: members sit at
    # axis_x -+ d on ONE y, so the mirror posture holds BY CONSTRUCTION
    # through relaxation AND the press (the partner equality edges carry it).
    # The candidacy rule is uniform by pair CLASS (never per circuit):
    # * a declared DIFF pair (``view.diff_pairs``) is always rigidified; a
    #   pair split across style rows CONVERGES to one row under the soft
    #   band face (the declared softening this backend exists for), host row
    #   chosen by a deterministic connectivity vote, the moved member's band
    #   deviation REPORTED -- never silent;
    # * a 2-member MIRROR group (``topo_sketch.mirror_groups`` -- the same
    #   candidate derivation the routing axis consumer uses) is re-posed
    #   only when it already shares a style row; a cross-row mirror group
    #   keeps its membership and the refusal is noted honestly.
    # Each realized pair's solved axis lands in ``RelaxReport.mirror_pairs``
    # (the placement's axis OUTPUT -- the routing Mirror constraint consumes
    # it; design data, never a reference coordinate).
    mirror_pairs_rigid: bool = False


@dataclass(frozen=True)
class RelaxReport:
    """The honest self-declaration face of one relaxation run.

    ``legality`` states the iron rule verbatim: the relaxer asserts NO
    legality; the projection provides the press-graph guarantee and the
    downstream oracle chain adjudicates. ``band_deviations`` lists every
    device the relaxation pulled off its style band (the softened-band
    contract: deviation is allowed, silence is not)."""

    engine: str
    selected: str                       # 'restart-K' | 'seed-fallback'
    #                                     (suffixed '@hostsN' under the P1b
    #                                     host enumeration -- multiple
    #                                     problems, one per host combination)
    legality: str
    restart_energies: Tuple[float, ...]
    selected_energy: Optional[float]
    energy_terms: Mapping[str, float]
    seed_placement: Mapping[str, _XY]
    relaxed_placement: Mapping[str, _XY]
    home_band: Mapping[str, int]
    band_assignment: Mapping[str, int]
    band_deviations: Tuple[str, ...]
    seed_outline_um2: float
    projected_outline_um2: float
    seed_hpwl_um: float
    projected_hpwl_um: float
    notes: Tuple[str, ...] = ()
    # P1b mirror-rigid pairs (empty unless ``knobs.mirror_pairs_rigid``):
    # one record per RIGIDIFIED pair, measured on the FINAL projected
    # placement -- {'a', 'b', 'kind' ('diff'|'mirror'), 'axis_x_um', 'd_um',
    # 'y_um', 'band', 'source_bands', 'realized'}. ``realized`` is the
    # honest posture verdict (same y within 1e-6 um); only a realized
    # pair's axis may be consumed downstream as a solved mirror axis.
    mirror_pairs: Tuple[Mapping[str, Any], ...] = ()


_LEGALITY_TEXT = (
    "not asserted: the relaxer issues no legality verdict; the compaction "
    "projection provides the press-graph separation and the downstream "
    "oracle chain (route contract / LVS / DRC) adjudicates"
)


# --------------------------------------------------------------------------- #
# problem assembly (design data only)
# --------------------------------------------------------------------------- #
def _declared_pairs(view: Any) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for attr in ("matched_pairs", "mirror_pairs", "diff_pairs"):
        for ab in getattr(view, attr, ()) or ():
            if len(ab) == 2:
                a, b = str(ab[0]), str(ab[1])
                out.append((a, b) if a < b else (b, a))
    return sorted(set(out))


def _mirror_rigid_candidates(view: Any) -> List[Tuple[str, str, str]]:
    """The declared mirror-candidate pairs, ``(a, b, kind)`` with names
    sorted and ``kind`` in {'diff', 'mirror'} -- the SAME candidate
    derivation the routing axis consumer uses (``intent.diff_pairs`` plus
    2-member ``topo_sketch.mirror_groups``). Diff pairs come first: where a
    member appears in both, the diff declaration wins (the stronger
    symmetry intent)."""
    from clay.layout.extract.topo_sketch import mirror_groups

    out: List[Tuple[str, str, str]] = []
    seen: set = set()
    for ab in getattr(view, "diff_pairs", ()) or ():
        if len(ab) == 2:
            a, b = sorted((str(ab[0]), str(ab[1])))
            if a != b and (a, b) not in seen:
                seen.add((a, b))
                out.append((a, b, "diff"))
    for g in mirror_groups(view):
        if len(g.members) == 2:
            a, b = g.members
            if (a, b) not in seen:
                seen.add((a, b))
                out.append((a, b, "mirror"))
    return out


def _net_pins(view: Any, members: Sequence[str]) -> List[Tuple[float, List[int]]]:
    """Per-net (weight, member-index list), hypergraph clique-downweighted
    (1/(k-1)); nets with < 2 placed members carry no force."""
    idx = {n: i for i, n in enumerate(members)}
    nets: Dict[str, set] = {}
    for rec in list(getattr(view, "devices", []) or []) + list(
            getattr(view, "passives", []) or []):
        i = idx.get(str(rec.name))
        if i is None:
            continue
        for net in (rec.nets or {}).values():
            nets.setdefault(str(net), set()).add(i)
    out: List[Tuple[float, List[int]]] = []
    for net in sorted(nets):
        ids = sorted(nets[net])
        if len(ids) < 2:
            continue
        out.append((1.0 / (len(ids) - 1), ids))
    return out


@dataclass
class _Problem:
    members: List[str]                  # movable device names (no rails)
    sizes_wh: np.ndarray                # (n, 2)
    body_of: np.ndarray                 # (n,) body index per member
    n_bodies: int
    body_names: List[str]               # canonical body key (min member name)
    offsets: np.ndarray                 # (n, 2) member - body anchor (seed)
    theta0: np.ndarray                  # (n_bodies, 2) seed body anchors
    nets: List[Tuple[float, List[int]]]
    home_band: Dict[str, int]           # member -> seed band id
    well_y: Dict[int, float]            # device band id -> band line y (seed)
    anchor_y: np.ndarray                # (n,) per-member seed y
    body_home: np.ndarray               # (n_bodies,) home band id
    corridors: List[Tuple[float, float]]  # rail y strips (lo, hi)
    min_gap: float
    target_wh: _XY                      # outline hinge target
    pair_ia: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    pair_ib: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    # P1b mirror-rigid pairs (empty unless knobs.mirror_pairs_rigid):
    # (left, right) in posture order, parallel kind/host/source-row tuples,
    # build-time notes, and the member set (for the deviation-note wording).
    mirror_pairs: Tuple[Tuple[str, str], ...] = ()
    mirror_kind: Tuple[str, ...] = ()
    mirror_host: Tuple[int, ...] = ()
    mirror_src: Tuple[Tuple[int, int], ...] = ()
    mirror_notes: Tuple[str, ...] = ()
    mirror_members: frozenset = frozenset()
    # per-BODY y lock (mirror bodies): the host row is a STRUCTURAL choice
    # (enumerated + objective-selected), so the relaxation may not re-litigate
    # the pair's y -- measured on t4 (2026-06-13): with y free, the wide
    # rigid pair body drifted between rows and plowed the field apart
    # (fill 74.8% -> 63.3%). x (the axis) stays fully free.
    y_locked: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))


def _build_problem(
    view: Any,
    seed: Mapping[str, _XY],
    sizes: Mapping[str, _XY],
    row_of: Mapping[str, int],
    rail_rows: frozenset,
    min_gap: float,
    knobs: RelaxKnobs,
    mirror_hosts: Optional[Mapping[Tuple[str, str], int]] = None,
) -> _Problem:
    members = sorted(n for n in seed if not str(n).startswith(_RAIL_PREFIX))
    rails = sorted(n for n in seed if str(n).startswith(_RAIL_PREFIX))
    n = len(members)
    sz = np.array([[float(sizes[m][0]), float(sizes[m][1])] for m in members],
                  dtype=float)
    pos = np.array([[float(seed[m][0]), float(seed[m][1])] for m in members],
                   dtype=float)

    # ---- P1b MIRROR-RIGID POSTURE (opt-in; the (axis_x, d, y) change of
    # variables -- see RelaxKnobs.mirror_pairs_rigid for the candidacy
    # rule). ``pos_eff`` carries the RE-POSED seed the rigid offsets are
    # frozen from; the TRUE seed (``pos``) keeps feeding the band wells and
    # the report, so every convergence shows up as a reported deviation.
    midx = {m: i for i, m in enumerate(members)}
    mirror_pairs: List[Tuple[str, str]] = []
    mirror_kind: List[str] = []
    mirror_host: List[int] = []
    mirror_src: List[Tuple[int, int]] = []
    mirror_notes: List[str] = []
    mirror_members: set = set()
    pos_eff = pos
    if bool(getattr(knobs, "mirror_pairs_rigid", False)):
        pos_eff = pos.copy()
        row_y: Dict[int, List[float]] = {}
        for i, m in enumerate(members):
            row_y.setdefault(int(row_of[m]), []).append(float(pos[i, 1]))
        row_mean = {k: sum(v) / len(v) for k, v in row_y.items()}
        recs = {str(r.name): r for r in
                list(getattr(view, "devices", []) or [])
                + list(getattr(view, "passives", []) or [])}
        # net -> [(member, row)] incidence (for the host-row vote).
        net_rows: Dict[str, List[Tuple[str, int]]] = {}
        for nm in members:
            rec = recs.get(nm)
            for net in (getattr(rec, "nets", None) or {}).values():
                net_rows.setdefault(str(net), []).append(
                    (nm, int(row_of[nm])))
        for a, b, kind in _mirror_rigid_candidates(view):
            if a not in midx or b not in midx:
                continue  # unplaced/composited member: no posture to pose
            if a in mirror_members or b in mirror_members:
                mirror_notes.append(
                    f"mirror-rigid skip {a}/{b}: a member already belongs "
                    "to an earlier mirror-rigid pair (first declared pair "
                    "wins, deterministic order)")
                continue
            ia, ib = midx[a], midx[b]
            ra, rb = int(row_of[a]), int(row_of[b])
            if ra != rb and kind != "diff":
                mirror_notes.append(
                    f"mirror-rigid note {a}/{b}: 2-member mirror group "
                    f"split across style rows {ra}/{rb} -- row membership "
                    "kept (only a declared DIFF pair converges across rows "
                    "under the soft band face)")
                continue
            wa = float(sizes[a][0])
            wb = float(sizes[b][0])
            sep_min = (wa + wb) / 2.0 + float(min_gap)
            if ra == rb:
                host = ra
                d = max(abs(float(pos[ib, 0] - pos[ia, 0])), sep_min) / 2.0
                cy = (float(pos[ia, 1]) + float(pos[ib, 1])) / 2.0
                mirror_notes.append(
                    f"mirror-rigid {a}/{b}: same-row {kind} pair re-posed "
                    f"to exact mirror posture (d={d:.2f} um)")
            else:
                forced = (mirror_hosts or {}).get((a, b))
                if forced is not None and int(forced) in (ra, rb):
                    # host fixed by the caller's deterministic ENUMERATION
                    # (both rows are tried; the selection objective decides
                    # -- see _refine's host-enumeration stage).
                    host = int(forced)
                    why = "host candidate under deterministic enumeration"
                else:
                    votes = {ra: 0, rb: 0}
                    pair_nets: List[str] = sorted({
                        str(net) for nm in (a, b)
                        for net in (getattr(recs.get(nm), "nets", None)
                                    or {}).values()
                    })
                    for net in pair_nets:
                        for nm, r in net_rows.get(net, ()):
                            if nm not in (a, b) and r in votes:
                                votes[r] += 1
                    host = max(sorted(votes), key=lambda r: (votes[r], -r))
                    why = ("connectivity vote "
                           f"{ {k: votes[k] for k in sorted(votes)} }")
                d = sep_min / 2.0
                cy = float(row_mean[host])
                mirror_notes.append(
                    f"mirror-rigid {a}/{b}: declared diff pair split "
                    f"across style rows {ra}/{rb} -- converged to row "
                    f"{host} under the soft band face ({why}); the moved "
                    "member's band deviation is reported below")
            cx = (float(pos[ia, 0]) + float(pos[ib, 0])) / 2.0
            if (float(pos[ia, 0]), a) <= (float(pos[ib, 0]), b):
                li, ri, left, right = ia, ib, a, b
            else:
                li, ri, left, right = ib, ia, b, a
            pos_eff[li] = (cx - d, cy)
            pos_eff[ri] = (cx + d, cy)
            mirror_pairs.append((left, right))
            mirror_kind.append(str(kind))
            mirror_host.append(int(host))
            mirror_src.append((ra, rb))
            mirror_members.update((a, b))

    # rigid bodies: declared partner pairs SHARING a band (union-find);
    # P1b mirror-rigid pairs merge UNCONDITIONALLY (their posture already
    # placed both members on one y -- the cross-row convergence case).
    parent = {m: m for m in members}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra_, rb_ = find(a), find(b)
        if ra_ != rb_:
            if ra_ < rb_:
                parent[rb_] = ra_
            else:
                parent[ra_] = rb_

    for left, right in mirror_pairs:
        union(left, right)
    for a, b in _declared_pairs(view):
        if a in parent and b in parent and row_of.get(a) == row_of.get(b):
            union(a, b)

    roots = sorted({find(m) for m in members})
    body_idx = {r: i for i, r in enumerate(roots)}
    body_of = np.array([body_idx[find(m)] for m in members], dtype=int)
    n_bodies = len(roots)

    # body anchors / frozen offsets from the EFFECTIVE seed (mirror posture
    # included); byte-identical to the historical path when pos_eff is pos.
    theta0 = np.zeros((n_bodies, 2), dtype=float)
    counts = np.zeros(n_bodies, dtype=float)
    np.add.at(theta0, body_of, pos_eff)
    np.add.at(counts, body_of, 1.0)
    theta0 /= counts[:, None]
    offsets = pos_eff - theta0[body_of]

    # home bands: the TRUE style rows (the report's honesty face -- a
    # mirror-converged member keeps its declared home so the move is a
    # reported deviation, never a rewritten history).
    home_band = {m: int(row_of[m]) for m in members}
    body_home = np.array([home_band[members[int(np.argmax(body_of == b))]]
                          for b in range(n_bodies)], dtype=int)
    # well centers: mean seed y per DEVICE band.
    well: Dict[int, List[float]] = {}
    for i, m in enumerate(members):
        well.setdefault(home_band[m], []).append(pos[i, 1])
    well_y = {b: float(sum(v) / len(v)) for b, v in sorted(well.items())}

    # P1b: a mirror body's well/displacement anchor is the HOST row -- the
    # band well pulls the WHOLE pair toward the host line (anchor_y = the
    # posture y) and the displacement-matching band assignment becomes
    # absolute nearest-well matching for exactly these bodies.
    anchor_y = pos[:, 1].copy()
    y_locked = np.zeros(n_bodies, dtype=bool)
    for (left, right), host in zip(mirror_pairs, mirror_host):
        body_home[body_of[midx[left]]] = int(host)
        anchor_y[midx[left]] = float(pos_eff[midx[left], 1])
        anchor_y[midx[right]] = float(pos_eff[midx[right], 1])
        y_locked[body_of[midx[left]]] = True

    corridors = []
    for r in rails:
        cy = float(seed[r][1])
        h = float(sizes[r][1])
        corridors.append((cy - h / 2.0, cy + h / 2.0))

    # outline hinge target: area-ideal at the seed aspect (design data only),
    # floored at the widest / tallest single device.
    if knobs.outline_target is not None:
        target_wh = (float(knobs.outline_target[0]),
                     float(knobs.outline_target[1]))
    else:
        x0 = float(np.min(pos[:, 0] - sz[:, 0] / 2.0))
        x1 = float(np.max(pos[:, 0] + sz[:, 0] / 2.0))
        y0 = float(np.min(pos[:, 1] - sz[:, 1] / 2.0))
        y1 = float(np.max(pos[:, 1] + sz[:, 1] / 2.0))
        seed_w, seed_h = max(x1 - x0, 1e-9), max(y1 - y0, 1e-9)
        aspect = seed_w / seed_h
        a_dev = float(np.sum(sz[:, 0] * sz[:, 1]))
        tw = max(math.sqrt(a_dev * aspect), float(np.max(sz[:, 0])))
        th = max(math.sqrt(a_dev / aspect), float(np.max(sz[:, 1])))
        target_wh = (tw, th)

    pia, pib = np.triu_indices(n, k=1)
    keep = body_of[pia] != body_of[pib]  # same-body pairs are rigid: no force
    return _Problem(
        members=members, sizes_wh=sz, body_of=body_of, n_bodies=n_bodies,
        body_names=roots, offsets=offsets, theta0=theta0,
        nets=_net_pins(view, members), home_band=home_band, well_y=well_y,
        anchor_y=anchor_y,
        body_home=body_home, corridors=corridors, min_gap=float(min_gap),
        target_wh=target_wh, pair_ia=pia[keep], pair_ib=pib[keep],
        mirror_pairs=tuple(mirror_pairs), mirror_kind=tuple(mirror_kind),
        mirror_host=tuple(mirror_host), mirror_src=tuple(mirror_src),
        mirror_notes=tuple(mirror_notes),
        mirror_members=frozenset(mirror_members),
        y_locked=y_locked,
    )


# --------------------------------------------------------------------------- #
# energy + gradient (member-space, accumulated onto bodies)
# --------------------------------------------------------------------------- #
def _smax(v: np.ndarray, gamma: float) -> Tuple[float, np.ndarray]:
    """gamma * logsumexp(v / gamma) and its gradient (softmax)."""
    m = float(np.max(v))
    e = np.exp((v - m) / gamma)
    s = float(np.sum(e))
    return m + gamma * math.log(s), e / s


def _energy(
    p: _Problem, theta: np.ndarray, knobs: RelaxKnobs,
    ov_scale: float = 1.0,
) -> Tuple[float, np.ndarray, Dict[str, float]]:
    """Total energy, gradient w.r.t. theta, and the per-term breakdown.
    ``ov_scale`` is the annealed overlap-weight multiplier (1.0 = final)."""
    pos = theta[p.body_of] + p.offsets          # (n, 2)
    g_mem = np.zeros_like(pos)
    terms: Dict[str, float] = {}
    gamma = float(knobs.gamma_um)

    # 1) smooth wirelength (LSE-HPWL per net, downweighted).
    e_wl = 0.0
    for w, ids in p.nets:
        for ax in (0, 1):
            v = pos[ids, ax]
            hi, ghi = _smax(v, gamma)
            lo, glo = _smax(-v, gamma)
            e_wl += w * (hi + lo)
            g_mem[ids, ax] += knobs.w_wirelength * w * (ghi - glo)
    terms["wirelength"] = e_wl

    # 2) pairwise rectified overlap (the swappable density seam).
    w_ov = knobs.w_overlap * float(ov_scale)
    e_ov = 0.0
    if p.pair_ia.size:
        ia, ib = p.pair_ia, p.pair_ib
        d = pos[ia] - pos[ib]                   # (m, 2)
        s = (p.sizes_wh[ia] + p.sizes_wh[ib]) / 2.0 + p.min_gap
        sgn = np.where(d >= 0.0, 1.0, -1.0)     # deterministic at 0
        o = np.maximum(0.0, s - np.abs(d))      # (m, 2)
        act = (o[:, 0] > 0.0) & (o[:, 1] > 0.0)
        if np.any(act):
            e_pair = o[act, 0] * o[act, 1]
            e_ov = float(np.sum(e_pair))
            # d(ox*oy)/dxa = -sign(dx)*oy ; symmetric for b.
            gx = -sgn[act, 0] * o[act, 1]
            gy = -sgn[act, 1] * o[act, 0]
            gpair = np.stack([gx, gy], axis=1) * w_ov
            np.add.at(g_mem, ia[act], gpair)
            np.add.at(g_mem, ib[act], -gpair)
    terms["overlap"] = e_ov

    # 3) style-band wells: Huber on the member's y DISPLACEMENT from its own
    # seed y (the band-anchored zero -- seed rows are bottom-aligned with
    # mixed heights, so a band-mean well would penalize the seed itself and
    # could misassign a tall member to the row above; the per-member anchor
    # keeps the well band-shaped while costing the seed exactly zero).
    e_band = 0.0
    delta = float(knobs.band_huber_um)
    e = pos[:, 1] - p.anchor_y
    ae = np.abs(e)
    quad = ae <= delta
    e_band = float(np.sum(np.where(quad, 0.5 * e * e,
                                   delta * (ae - 0.5 * delta))))
    g_mem[:, 1] += knobs.w_band * np.clip(e, -delta, delta)
    terms["band_well"] = e_band

    # 4) corridor forbidden bands (squared y-overlap with each rail strip).
    e_corr = 0.0
    for lo, hi in p.corridors:
        top = pos[:, 1] + p.sizes_wh[:, 1] / 2.0
        bot = pos[:, 1] - p.sizes_wh[:, 1] / 2.0
        o = np.minimum(top, hi) - np.maximum(bot, lo)
        act = o > 0.0
        if not np.any(act):
            continue
        e_corr += float(np.sum(o[act] ** 2))
        do = np.where(top < hi, 1.0, 0.0) - np.where(bot > lo, 1.0, 0.0)
        # plateau escape (device straddling/inside the strip): push away
        # from the strip center, +y on the exact center (deterministic).
        center = (lo + hi) / 2.0
        esc = np.where(pos[:, 1] >= center, 1.0, -1.0)
        do = np.where((do == 0.0), esc, do)
        g_mem[act, 1] += knobs.w_corridor * 2.0 * o[act] * do[act]
    terms["corridor"] = e_corr

    # 5) outline hinge toward the area-ideal target (smooth max extents).
    e_out = 0.0
    for ax, cap in ((0, p.target_wh[0]), (1, p.target_wh[1])):
        half = p.sizes_wh[:, ax] / 2.0
        hi, ghi = _smax(pos[:, ax] + half, gamma)
        lo, glo = _smax(-(pos[:, ax] - half), gamma)
        span = hi + lo
        excess = span - cap
        if excess > 0.0:
            e_out += excess * excess
            g_mem[:, ax] += knobs.w_outline * 2.0 * excess * (ghi - glo)
    terms["outline"] = e_out

    total = (knobs.w_wirelength * terms["wirelength"]
             + w_ov * terms["overlap"]
             + knobs.w_band * terms["band_well"]
             + knobs.w_corridor * terms["corridor"]
             + knobs.w_outline * terms["outline"])

    g_theta = np.zeros_like(theta)
    np.add.at(g_theta, p.body_of, g_mem)
    return total, g_theta, terms


def _optimize(
    p: _Problem, knobs: RelaxKnobs, restart: int,
) -> Tuple[np.ndarray, float]:
    """Adam + decaying Gaussian gradient noise from a deterministic seed;
    returns (theta, final noiseless energy)."""
    rng = np.random.Generator(np.random.PCG64(int(knobs.seed) + restart))
    theta = p.theta0.copy()
    locked = p.y_locked if p.y_locked.size == p.theta0.shape[0] \
        else np.zeros(p.theta0.shape[0], dtype=bool)
    if restart > 0:
        theta = theta + knobs.jitter_um * rng.uniform(-1.0, 1.0, theta.shape)
        theta[locked, 1] = p.theta0[locked, 1]  # y-locked bodies: no y jitter
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    b1, b2, eps = 0.9, 0.999, 1e-8
    n_iter = max(1, int(knobs.iters))
    knee = max(1.0, knobs.noise_decay_frac * n_iter)
    r0 = float(knobs.overlap_ramp0)
    for t in range(n_iter):
        frac = t / max(1, n_iter - 1)
        ov_scale = r0 + (1.0 - r0) * frac * frac  # quadratic anneal to 1.0
        _e, g, _terms = _energy(p, theta, knobs, ov_scale)
        sigma = knobs.noise0_um * max(0.0, 1.0 - t / knee)
        if sigma > 0.0:
            g = g + sigma * rng.standard_normal(theta.shape)
        g[locked, 1] = 0.0  # P1b: a mirror body's y is structural, not free
        m = b1 * m + (1.0 - b1) * g
        v = b2 * v + (1.0 - b2) * g * g
        mh = m / (1.0 - b1 ** (t + 1))
        vh = v / (1.0 - b2 ** (t + 1))
        theta = theta - knobs.step * mh / (np.sqrt(vh) + eps)
    e_final, _g, _terms = _energy(p, theta, knobs)
    return theta, float(e_final)


# --------------------------------------------------------------------------- #
# projection through the existing compaction press (no second legalizer)
# --------------------------------------------------------------------------- #
def _assign_bands(p: _Problem, theta: np.ndarray) -> Dict[str, int]:
    """Per-BODY band assignment by DISPLACEMENT matching: a body belongs to
    the band whose line offset (relative to its home band's line) best
    matches the body's mean y displacement from its seed -- robust to the
    bottom-aligned mixed-height rows where a tall member's centroid sits
    far from the band mean. Rigid groups stay together; ties keep the home
    band, then the smaller band id."""
    pos = theta[p.body_of] + p.offsets
    dy_mem = pos[:, 1] - p.anchor_y
    body_dy = np.zeros(p.n_bodies)
    counts = np.zeros(p.n_bodies)
    np.add.at(body_dy, p.body_of, dy_mem)
    np.add.at(counts, p.body_of, 1.0)
    body_dy /= counts
    bands = sorted(p.well_y)
    out: Dict[str, int] = {}
    for b in range(p.n_bodies):
        home = int(p.body_home[b])
        best = min(
            bands,
            key=lambda k: (abs(body_dy[b] - (p.well_y[k] - p.well_y[home])),
                           0 if k == home else 1, k),
        )
        for i in np.nonzero(p.body_of == b)[0]:
            out[p.members[int(i)]] = int(best)
    # P1b: a mirror-rigid body's band IS its host row BY CONTRACT -- the
    # declared posture/convergence fixed the pair's row membership, and the
    # press orders cross-row pairs by exactly that membership. Re-banding
    # the body off a mid-flight y (measured on t4: the converged pair
    # drifted between rows, was re-banded into a third row and the stack
    # lost 11pp of fill) would silently undo the declaration.
    for (left, _right), host in zip(p.mirror_pairs, p.mirror_host):
        b = int(p.body_of[p.members.index(left)])
        for i in np.nonzero(p.body_of == b)[0]:
            out[p.members[int(i)]] = int(host)
    return out


def _body_partners(p: _Problem, view: Any) -> Tuple[Tuple[str, str], ...]:
    """Declared pairs living inside ONE rigid body (their relative offset is
    the press's equality edge) -- shared by the projection press and the P1b
    mirror embed pass (one derivation, no second bookkeeping)."""
    body_root = {m: p.body_names[int(p.body_of[i])]
                 for i, m in enumerate(p.members)}
    return tuple(sorted(
        (a, b) for a, b in _declared_pairs(view)
        if a in body_root and b in body_root
        and body_root[a] == body_root[b]
    ))


def _project(
    p: _Problem,
    view: Any,
    theta: np.ndarray,
    seed: Mapping[str, _XY],
    sizes: Mapping[str, _XY],
    rail_row_of: Mapping[str, int],
    rail_rows: frozenset,
    assignment: Mapping[str, int],
    min_gap: float,
    band_channel: float,
    channel_um_per_gap: Optional[Mapping[int, float]],
    press_mode: str = "press",
) -> Dict[str, _XY]:
    """Legalize a relaxed solution with the EXISTING press: row partition =
    the band assignment (devices) + the rail pseudo rows; partner equality =
    the declared pairs inside one rigid body. ``press_mode='2d'`` (the
    compact='2d' chain) directs cross-row y edges by ROW ORDER, so a
    candidate that proposed an order inversion is pressed back, never
    legalized upside down. May raise ValueError (a press refusal on
    conflicting intent) -- the caller skips the candidate."""
    pos = theta[p.body_of] + p.offsets
    placed: Dict[str, _XY] = {
        m: (float(pos[i, 0]), float(pos[i, 1]))
        for i, m in enumerate(p.members)
    }
    for r, k in rail_row_of.items():
        placed[r] = (float(seed[r][0]), float(seed[r][1]))

    row_of = dict(rail_row_of)
    row_of.update({m: int(assignment[m]) for m in p.members})

    spec = CompactionSpec(row_of=row_of, rail_rows=rail_rows,
                          partners=_body_partners(p, view),
                          mode=str(press_mode))
    return compact_placement(
        placed, sizes, spec, min_gap=min_gap, band_channel=band_channel,
        channel_um_per_gap=channel_um_per_gap,
    )


def _dev_outline_um2(pos: Mapping[str, _XY], sizes: Mapping[str, _XY]) -> float:
    xs0, ys0, xs1, ys1 = [], [], [], []
    for n, (cx, cy) in pos.items():
        if str(n).startswith(_RAIL_PREFIX):
            continue
        w, h = float(sizes[n][0]), float(sizes[n][1])
        xs0.append(cx - w / 2.0)
        xs1.append(cx + w / 2.0)
        ys0.append(cy - h / 2.0)
        ys1.append(cy + h / 2.0)
    if not xs0:
        return 0.0
    return (max(xs1) - min(xs0)) * (max(ys1) - min(ys0))


def _hpwl_um(p: _Problem, pos: Mapping[str, _XY]) -> float:
    tot = 0.0
    for w, ids in p.nets:
        xs = [pos[p.members[i]][0] for i in ids]
        ys = [pos[p.members[i]][1] for i in ids]
        tot += w * ((max(xs) - min(xs)) + (max(ys) - min(ys)))
    return tot


# --------------------------------------------------------------------------- #
# @edge_anchor HARD pass (L): slide an anchored cap flush to a die wall and
# evacuate the few devices it would then cover into the freed space, all WITHIN
# the existing outline (the die must not grow). This is a CONSTRAINT, not a soft
# energy term -- the anchored cap's flush position is fixed and the area/HPWL
# objective only ranges over the freed devices. Design data only: the cap
# footprint comes from ``sizes`` (the relax cap size IS the canonical footprint),
# the wall set comes from the seed outline, the IO exclusion is a categorical
# side token -- NO reference coordinate is read.
# --------------------------------------------------------------------------- #
def _rect(nm: str, pos: Mapping[str, _XY],
          sizes: Mapping[str, _XY]) -> Optional[_Bbox]:
    xy = pos.get(nm)
    s = sizes.get(nm)
    if xy is None or s is None:
        return None
    w, h = float(s[0]) / 2.0, float(s[1]) / 2.0
    return (float(xy[0]) - w, float(xy[1]) - h,
            float(xy[0]) + w, float(xy[1]) + h)


def _rect_overlap(a: _Bbox, b: _Bbox) -> float:
    from math import ulp

    xhi, xlo = min(a[2], b[2]), max(a[0], b[0])
    yhi, ylo = min(a[3], b[3]), max(a[1], b[1])
    ox, oy = xhi - xlo, yhi - ylo
    # The same minimum-gap edge is evaluated through different additions and
    # subtractions. At ~20 um this produced a 7e-15 um phantom overlap and
    # rejected the sole free slot. Bound only floating arithmetic error, not a
    # physical geometry tolerance: even a real 1 nm overlap remains positive.
    if ox <= 4.0 * max(ulp(xhi), ulp(xlo)):
        ox = 0.0
    if oy <= 4.0 * max(ulp(yhi), ulp(ylo)):
        oy = 0.0
    return (ox * oy) if (ox > 0.0 and oy > 0.0) else 0.0


def apply_edge_anchor(
    placement: Mapping[str, _XY],
    sizes: Mapping[str, _XY],
    anchored: Iterable[str],
    *,
    min_gap: float,
    io_edges: Iterable[str] = (),
    rail_prefix: str = _RAIL_PREFIX,
) -> Tuple[Dict[str, _XY], List[str]]:
    """Slide each anchored device (a large MIM cap) FLUSH to ONE die wall and
    evacuate the few devices it would then cover, keeping the outline FIXED (the
    die must NOT grow). The anchored cap's flush position is a CONSTRAINT, not a
    soft pull; only the freed devices are repositioned. Returns the unchanged
    placement + an honest note when no wall frees a side within the existing die.

    The wall is SOLVED (never declared/read off a reference): among the four walls
    with room (die_span - cap_span > a min-gap margin), candidates are ranked by
    FEWEST blocking devices (the side that disturbs the fewest devices), then
    PREFER a wall NOT claimed by an IO edge (an IO wall is a weaker clear side,
    but NOT excluded -- the signed reference itself anchors flush to its VOUT/right
    wall), then most room, then side order R,L,T,B. Each candidate is TRIED in
    rank order; the first that succeeds within the existing outline is kept.

    Per candidate wall: (1) CONSOLIDATE every non-cap, non-blocker device AWAY
    from the wall by a deterministic 1D pack (``_consolidate_away``) so the
    whitespace fragments scattered through the (~62%-full) cluster merge into ONE
    contiguous band beside the wall -- the room the blockers need (the area always
    existed, only fragmented); (2) translate the cap flush; (3) place each blocker
    (largest first) in the nearest legal free slot (``_free_slot``). The trial is
    rejected -- and the next wall tried -- if a blocker has no slot, any pair
    overlaps, or the outline grew. Every moved device's deviation is recorded.

    Returns ``(new_placement, deviation_notes)``. Pure geometry, design data only
    (placement / sizes / die extent / IO side tokens), NO reference coordinate;
    deterministic."""
    anchored = [a for a in anchored if a in placement and a in sizes]
    if not anchored:
        return dict(placement), []
    io = {str(e).lower() for e in (io_edges or ())}
    _SIDE_IO = {"R": "right", "L": "left", "T": "top", "B": "bottom"}

    movable = [n for n in placement
               if not str(n).startswith(rail_prefix)
               and n in sizes and n not in anchored]
    die0 = _bbox_of(placement, sizes, rail_prefix)
    if die0 is None:
        return dict(placement), []
    out = {n: (float(v[0]), float(v[1])) for n, v in placement.items()}
    notes: List[str] = []
    margin = max(2.0 * float(min_gap), 1e-6)

    for cap in anchored:
        die = _bbox_of(out, sizes, rail_prefix)
        if die is None:
            break
        cb = _rect(cap, out, sizes)
        if cb is None:
            continue
        cw, ch = cb[2] - cb[0], cb[3] - cb[1]
        room_x = (die[2] - die[0]) - cw
        room_y = (die[3] - die[1]) - ch
        # the cap's flush footprint on each candidate wall (within the die).
        wall_box = {
            "R": (die[2] - cw, cb[1], die[2], cb[3]),
            "L": (die[0], cb[1], die[0] + cw, cb[3]),
            "T": (cb[0], die[3] - ch, cb[2], die[3]),
            "B": (cb[0], die[1], cb[2], die[1] + ch),
        }
        wall_room = {"R": room_x, "L": room_x, "T": room_y, "B": room_y}
        # candidate walls: those with room beyond a min-gap margin. The wall is
        # SOLVED -- ranked by FEWEST blockers (the side that disturbs the fewest
        # devices), then PREFER a wall NOT claimed by an IO edge (an IO wall is a
        # weaker clear side -- the IO net wants that boundary -- but NOT excluded,
        # the signed reference itself anchors flush to its VOUT (right) wall), then
        # most room, then deterministic side order R,L,T,B.
        cands = []
        for side in ("R", "L", "T", "B"):
            if wall_room[side] <= margin:
                continue
            fb = wall_box[side]
            blk = [n for n in movable if _rect_overlap(fb, _rect(n, out, sizes))
                   > 0.5]
            io_claimed = 1 if _SIDE_IO[side] in io else 0
            cands.append((len(blk), io_claimed, -wall_room[side],
                          ("R", "L", "T", "B").index(side), side, fb, blk))
        if not cands:
            notes.append(
                f"{cap}: no die wall frees a clear side within the existing "
                "outline (the cap spans the die on every axis); placement "
                "unchanged")
            continue
        cands.sort()
        # try each candidate wall in rank order; keep the first whose flush+
        # evacuate succeeds within the existing outline (honest fall-back to the
        # unchanged placement if none does).
        committed = False
        for _nblk, _io, _nroom, _ord, side, fb, blockers in cands:
            trial = dict(out)
            # 1) CONSOLIDATE: press every NON-cap, NON-blocker device AWAY from
            #    the chosen wall (a deterministic 1D compaction), so the
            #    whitespace fragments scattered through the cluster merge into ONE
            #    contiguous band beside the wall -- the room the blockers need
            #    (the field is ~62% full; the area exists but is fragmented, so a
            #    per-device free-slot scan fails without this pack). The cap and
            #    the blockers are NOT moved here (placed next); rails stay fixed.
            keep = [n for n in movable if n not in blockers]
            consolidated = _consolidate_away(trial, sizes, keep, side, die,
                                             min_gap)
            trial.update(consolidated)
            # 2) translate the cap flush to the chosen wall.
            trial[cap] = ((fb[0] + fb[2]) / 2.0, (fb[1] + fb[3]) / 2.0)
            # 3) evacuate each blocker into the FREE space anywhere in the die
            #    (the contiguous band the consolidation opened beside the wall +
            #    the cap's vacated footprint), closest to its current seat (least
            #    disruption), never into the flush cap.
            ok = True
            moved: List[Tuple[str, _XY]] = []
            # evacuate the LARGEST blockers first (the hardest to place gets the
            # widest choice of free slots; deterministic tie-break on name).
            order = sorted(blockers,
                           key=lambda n: (-float(sizes[n][0]) * float(sizes[n][1]),
                                          n))
            for nm in order:
                spot = _free_slot(nm, trial, sizes, die, fb, min_gap,
                                  rail_prefix, anchored, toward=out.get(nm))
                if spot is None:
                    ok = False
                    break
                trial[nm] = spot
                moved.append((nm, spot))
            if not ok:
                continue
            new_die = _bbox_of(trial, sizes, rail_prefix)
            grew = (new_die is None
                    or (new_die[2] - new_die[0]) > (die0[2] - die0[0]) + 1e-6
                    or (new_die[3] - new_die[1]) > (die0[3] - die0[1]) + 1e-6)
            if grew or _has_overlap(trial, sizes, min_gap, rail_prefix):
                continue
            for nm, spot in moved:
                notes.append(
                    f"{nm}: evacuated to ({spot[0]:.2f},{spot[1]:.2f}) -- it sat "
                    f"in the @edge_anchor cap {cap}'s flush footprint on the "
                    f"{side} wall (declared cap-flush outranks the style row); "
                    "review")
            notes.append(
                f"{cap}: anchored flush to the {side} die wall "
                f"(clear side: {side}); {len(moved)} device(s) evacuated, the "
                "outline did not grow")
            out = trial
            committed = True
            break
        if not committed:
            notes.append(
                f"{cap}: no feasible wall freed a clear side without growing the "
                "die or overlapping (honest fall-back); placement unchanged")
            continue
    return out, notes


def _bbox_of(pos: Mapping[str, _XY], sizes: Mapping[str, _XY],
             rail_prefix: str) -> Optional[_Bbox]:
    xs0, ys0, xs1, ys1 = [], [], [], []
    for n, xy in pos.items():
        if str(n).startswith(rail_prefix) or n not in sizes:
            continue
        r = _rect(n, pos, sizes)
        if r is None:
            continue
        xs0.append(r[0]); ys0.append(r[1]); xs1.append(r[2]); ys1.append(r[3])
    if not xs0:
        return None
    return (min(xs0), min(ys0), max(xs1), max(ys1))


def _consolidate_away(pos: Mapping[str, _XY], sizes: Mapping[str, _XY],
                      names: Iterable[str], wall: str, die: _Bbox,
                      min_gap: float) -> Dict[str, _XY]:
    """Push every device in ``names`` AWAY from ``wall`` as far as it goes
    without overlapping an already-pushed device (a deterministic 1D greedy
    compaction along the wall's axis), so the whitespace scattered through the
    cluster merges into ONE contiguous band beside the wall. y is preserved
    (the compaction is along the wall-perpendicular axis only); the die bound on
    the far side is the hard stop. Returns ONLY the moved devices' new centroids
    (the caller merges them). Pure geometry, deterministic."""
    names = list(names)
    gap = float(min_gap)
    horiz = wall in ("R", "L")
    ax = 0 if horiz else 1
    # push toward the LOW wall when the cap anchors to the HIGH wall, vice versa.
    toward_low = wall in ("R", "T")
    lo = die[0] if horiz else die[1]
    hi = die[2] if horiz else die[3]
    # process in the push direction so each device packs against the ones ahead.
    seq = sorted(names, key=lambda n: (pos[n][ax], n), reverse=not toward_low)
    placed: List[Tuple[str, _Bbox]] = []
    out: Dict[str, _XY] = {}
    for nm in seq:
        s = sizes.get(nm)
        p = pos.get(nm)
        if s is None or p is None:
            continue
        hw, hh = float(s[0]) / 2.0, float(s[1]) / 2.0
        cx, cy = float(p[0]), float(p[1])
        half = hw if horiz else hh
        perp0 = cy - hh if horiz else cx - hw
        perp1 = cy + hh if horiz else cx + hw
        # the target edge against the far wall, then back off past every placed
        # device whose perpendicular span overlaps this one's.
        if toward_low:
            edge = lo + half  # push to the LOW wall (centroid)
            for _n, b in placed:
                bp0 = b[1] if horiz else b[0]
                bp1 = b[3] if horiz else b[2]
                if min(perp1, bp1) - max(perp0, bp0) > 1e-9:
                    bhi = b[2] if horiz else b[3]
                    edge = max(edge, bhi + gap + half)
            edge = min(edge, hi - half)
        else:
            edge = hi - half
            for _n, b in placed:
                bp0 = b[1] if horiz else b[0]
                bp1 = b[3] if horiz else b[2]
                if min(perp1, bp1) - max(perp0, bp0) > 1e-9:
                    blo = b[0] if horiz else b[1]
                    edge = min(edge, blo - gap - half)
            edge = max(edge, lo + half)
        if horiz:
            ncx, ncy = edge, cy
        else:
            ncx, ncy = cx, edge
        out[nm] = (ncx, ncy)
        placed.append((nm, (ncx - hw, ncy - hh, ncx + hw, ncy + hh)))
    return out


def _free_slot(nm: str, pos: Mapping[str, _XY], sizes: Mapping[str, _XY],
               region: _Bbox, forbid: _Bbox, min_gap: float,
               rail_prefix: str, anchored: Iterable[str],
               toward: Optional[_XY] = None) -> Optional[_XY]:
    """Nearest free centroid for ``nm`` inside ``region`` (the cap's far side)
    that does not overlap any other placed device (with the min-gap margin) and
    does not re-enter ``forbid`` (the flush cap footprint). Deterministic FINE
    grid scan; the legal slot closest to ``toward`` (the device's current
    position -- least disruption / HPWL) is returned, ties broken by (x, y)."""
    s = sizes.get(nm)
    if s is None:
        return None
    hw, hh = float(s[0]) / 2.0, float(s[1]) / 2.0
    gap = float(min_gap)
    rx0, ry0, rx1, ry1 = region
    if (rx1 - rx0) < 2.0 * hw - 1e-9 or (ry1 - ry0) < 2.0 * hh - 1e-9:
        return None
    others = [(n, _rect(n, pos, sizes)) for n in pos
              if n != nm and not str(n).startswith(rail_prefix)
              and n in sizes and _rect(n, pos, sizes) is not None]

    def legal(cx: float, cy: float) -> bool:
        box = (cx - hw, cy - hh, cx + hw, cy + hh)
        if _rect_overlap(box, forbid) > 0.0:
            return False
        grown = (box[0] - gap, box[1] - gap, box[2] + gap, box[3] + gap)
        for _n, ob in others:
            if _rect_overlap(grown, ob) > 0.0:
                return False
        return True

    # FINE deterministic candidate grid (a fragmented whitespace field hides
    # slots smaller than a device step -- the grid must be finer than the device
    # to find them). Step = a fraction of the min-gap, floored so the scan stays
    # bounded; the legal centroid nearest ``toward`` wins.
    step = max(float(min_gap) / 2.0, 0.25)
    n_x = max(int((rx1 - rx0 - 2.0 * hw) / step) + 1, 1)
    n_y = max(int((ry1 - ry0 - 2.0 * hh) / step) + 1, 1)
    cxs = [round(rx0 + hw + k * step, 6) for k in range(n_x)]
    cys = [round(ry0 + hh + k * step, 6) for k in range(n_y)]
    tx = float(toward[0]) if toward is not None else (rx0 + rx1) / 2.0
    ty = float(toward[1]) if toward is not None else (ry0 + ry1) / 2.0
    grid = sorted(((cx, cy) for cx in cxs for cy in cys),
                  key=lambda c: ((c[0] - tx) ** 2 + (c[1] - ty) ** 2,
                                 c[0], c[1]))
    for cx, cy in grid:
        if legal(cx, cy):
            return (cx, cy)
    return None


def _has_overlap(pos: Mapping[str, _XY], sizes: Mapping[str, _XY],
                 min_gap: float, rail_prefix: str) -> bool:
    names = [n for n in pos if not str(n).startswith(rail_prefix)
             and n in sizes]
    rects = {n: _rect(n, pos, sizes) for n in names}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = rects[names[i]], rects[names[j]]
            if a is None or b is None:
                continue
            if _rect_overlap(a, b) > 1e-6:
                return True
    return False


# --------------------------------------------------------------------------- #
# the refinement core (shared by the chain hook and the standalone face)
# --------------------------------------------------------------------------- #
def _refine(
    view: Any,
    seed: Mapping[str, _XY],
    sizes: Mapping[str, _XY],
    row_of: Mapping[str, int],
    rail_rows: frozenset,
    *,
    min_gap: float,
    band_channel: float,
    channel_um_per_gap: Optional[Mapping[int, float]],
    knobs: Optional[RelaxKnobs],
    press_mode: str = "press",
    chain_spec: Optional[Any] = None,
) -> Tuple[Dict[str, _XY], RelaxReport]:
    knobs = knobs or RelaxKnobs()
    seed = {str(n): (float(x), float(y)) for n, (x, y) in seed.items()}
    rail_row_of = {n: int(row_of[n]) for n in seed
                   if str(n).startswith(_RAIL_PREFIX)}
    notes: List[str] = []

    members = [n for n in seed if not str(n).startswith(_RAIL_PREFIX)]
    if len(members) < 2:
        report = RelaxReport(
            engine="relax", selected="seed-fallback",
            legality=_LEGALITY_TEXT, restart_energies=(),
            selected_energy=None, energy_terms={},
            seed_placement=dict(seed), relaxed_placement=dict(seed),
            home_band={m: int(row_of[m]) for m in members},
            band_assignment={m: int(row_of[m]) for m in members},
            band_deviations=(), seed_outline_um2=_dev_outline_um2(seed, sizes),
            projected_outline_um2=_dev_outline_um2(seed, sizes),
            seed_hpwl_um=0.0, projected_hpwl_um=0.0,
            notes=("fewer than 2 movable devices; nothing to relax",),
        )
        return dict(seed), report

    # P1b HOST-ROW ENUMERATION: a cross-row DIFF pair's convergence target is
    # an underdetermined structural choice -- both of its style rows are
    # legitimate hosts. Measured on t4 (2026-06-13): the connectivity vote
    # alone (6:5, near-tie) packed MM4/MM18 into the already-widest row and
    # the press y-stacked it (band height 24.6 -> 64.4 um, fill 74.8% ->
    # 62.7%). So the host is ENUMERATED deterministically (both rows per
    # cross-row diff pair, full product up to 3 such pairs = 8 problems;
    # surplus pairs fall back to the vote with a note) and the SAME
    # selection objective (projected outline area, HPWL) adjudicates --
    # never the vote silently. host_choices == [None] (the default and every
    # no-cross-pair run) is the historical single-problem path, byte-identical.
    host_choices: List[Optional[Dict[Tuple[str, str], int]]] = [None]
    if bool(getattr(knobs, "mirror_pairs_rigid", False)):
        member_set = {n for n in seed
                      if not str(n).startswith(_RAIL_PREFIX)}
        cross: List[Tuple[Tuple[str, str], Tuple[int, int]]] = []
        taken: set = set()
        for a, b, kind in _mirror_rigid_candidates(view):
            if a not in member_set or b not in member_set:
                continue
            if a in taken or b in taken:
                continue
            taken.update((a, b))
            ra, rb = int(row_of[a]), int(row_of[b])
            if kind == "diff" and ra != rb:
                cross.append(((a, b), (min(ra, rb), max(ra, rb))))
        if cross:
            import itertools

            enum_pairs = cross[:3]
            host_choices = [
                {pair: h for (pair, _rows), h in zip(enum_pairs, combo)}
                for combo in itertools.product(
                    *[rows for _pair, rows in enum_pairs])
            ]
            if len(cross) > 3:
                notes.append(
                    "mirror-rigid host enumeration capped at 3 cross-row "
                    "diff pairs (8 problems); surplus pairs "
                    f"{[pr for pr, _ in cross[3:]]} fall back to the "
                    "connectivity vote")

    # candidates: per host choice, the projected SEED (always), then each
    # restart's relaxed solution projected. Selection: projected device
    # outline area, then projected HPWL, then candidate order -- all
    # deterministic.
    candidates: List[Tuple[str, Optional[float], Dict[str, int],
                           Optional[Dict[str, _XY]], Dict[str, _XY],
                           Dict[str, float], _Problem]] = []
    restart_energies: List[float] = []
    p0: Optional[_Problem] = None
    # @edge_anchor matched members (design data; empty -> default OFF path).
    try:
        from clay.layout.extract.edge_anchor import (
            edge_anchor_members as _anchor_members_fn,
        )
        am = set(_anchor_members_fn(view, seed))
    except Exception:
        am = set()
    for ci, hosts in enumerate(host_choices):
        p = _build_problem(view, seed, sizes, row_of, rail_rows, min_gap,
                           knobs, mirror_hosts=hosts)
        if p0 is None:
            p0 = p
        sfx = "" if len(host_choices) == 1 else f"@hosts{ci}"
        home_assign = {m: p.home_band[m] for m in p.members}
        # the seed candidate's band partition: with mirror-rigid pairs the
        # posture may already have moved a member across rows, so the seed
        # assignment must follow the BODIES (identical to home_assign when
        # no mirror pair exists -- the default path is untouched).
        seed_assign = _assign_bands(p, p.theta0) if p.mirror_pairs \
            else home_assign
        try:
            proj_seed = _project(
                p, view, p.theta0, seed, sizes, rail_row_of, rail_rows,
                seed_assign, min_gap, band_channel, channel_um_per_gap,
                press_mode,
            )
            candidates.append((f"seed-fallback{sfx}", None, seed_assign,
                               None, proj_seed, {}, p))
        except ValueError as exc:
            notes.append(f"seed projection{sfx} refused by the press: "
                         f"{exc}; the raw band solve output stands in as "
                         "the fallback")
            candidates.append((f"seed-fallback{sfx}", None, seed_assign,
                               None, dict(seed), {}, p))

        for r in range(max(1, int(knobs.restarts))):
            theta, e = _optimize(p, knobs, r)
            restart_energies.append(e)
            assignment = _assign_bands(p, theta)
            pos = theta[p.body_of] + p.offsets
            relaxed = {m: (float(pos[i, 0]), float(pos[i, 1]))
                       for i, m in enumerate(p.members)}
            relaxed.update({n: seed[n] for n in rail_row_of})
            _e, _g, terms_r = _energy(p, theta, knobs)
            try:
                proj = _project(
                    p, view, theta, seed, sizes, rail_row_of, rail_rows,
                    assignment, min_gap, band_channel, channel_um_per_gap,
                    press_mode,
                )
            except ValueError as exc:
                notes.append(f"restart {r}{sfx}: projection refused "
                             f"({exc}); candidate dropped")
                continue
            candidates.append((f"restart-{r}{sfx}", e, assignment,
                               relaxed, proj, terms_r, p))

    def _rank(c):
        return (round(_dev_outline_um2(c[4], sizes), 9),
                round(_hpwl_um(c[6], c[4]), 9))

    ranked = sorted(range(len(candidates)),
                    key=lambda i: (_rank(candidates[i]), i))
    sel = candidates[ranked[0]]
    tag, sel_energy, assignment, relaxed, projected, terms, p = sel
    # the SELECTED problem's build notes are the honest record (each host
    # candidate produced its own convergence note; only the winner's stands).
    notes.extend(p.mirror_notes)
    if len(host_choices) > 1:
        notes.append(
            "mirror-rigid host enumeration: "
            f"{len(host_choices)} host combination(s) tried through the "
            "selection objective; winner "
            f"{ {f'{a}/{b}': h for (a, b), h in zip(p.mirror_pairs, p.mirror_host)} }"
        )
    if tag.startswith("seed-fallback"):
        if p.mirror_pairs:
            notes.append("no relaxed restart beat the projected seed on the "
                         "selection objective (outline area, HPWL); the "
                         "band solve output is returned in MIRROR-RIGID "
                         "posture (the declared pairs were re-posed before "
                         "the projection)")
        else:
            notes.append("no relaxed restart beat the projected seed on the "
                         "selection objective (outline area, HPWL); the band "
                         "solve output is returned unchanged in substance")

    # P1b MIRROR EMBED PASS: a converged pair is a composite INSERTION into
    # its host row, exactly the move class the gradient cannot find and
    # :func:`compaction.embed_search` exists for (measured on t4,
    # 2026-06-13: pressed convergence alone 62.7% fill vs the plain-relax
    # 74.8%; one embed pass over the selected candidate recovers 74.6% with
    # the posture intact). The pass rides the SELECTED candidate only
    # (deterministic, bounded), under the SAME press (partner equality
    # edges keep the mirror posture rigid) and the CHAIN's S6 channel /
    # keepout guards (``chain_spec`` -- never an unguarded repack).
    if p.mirror_pairs:
        emb_row_of: Dict[str, int] = dict(rail_row_of)
        emb_row_of.update({m: int(assignment[m]) for m in p.members})
        emb_spec = CompactionSpec(
            row_of=emb_row_of, rail_rows=rail_rows,
            partners=_body_partners(p, view), mode=str(press_mode),
            channel_xnets=tuple(
                getattr(chain_spec, "channel_xnets", ()) or ()),
            keepout_names=tuple(
                getattr(chain_spec, "keepout_names", ()) or ()),
        )
        try:
            embedded, emb_notes = embed_search(
                projected, sizes, emb_spec, min_gap=min_gap,
                band_channel=band_channel,
                channel_um_per_gap=channel_um_per_gap,
            )
            projected = embedded
            notes.extend(emb_notes)
        except ValueError as exc:
            notes.append(f"mirror-rigid embed pass refused ({exc}); the "
                         "pressed projection stands")

    # @edge_anchor HARD pass (L): when the intent matches a device, slide every
    # anchored cap flush to a SOLVED die wall and evacuate the devices it would
    # cover into the freed space -- WITHIN the existing outline (the die must not
    # grow). A declared constraint that outranks the style-row order on the cap's
    # clear side. Default OFF: ``am`` empty -> ``apply_edge_anchor`` returns the
    # placement unchanged (byte-identical). The pass cannot grow the die (it
    # rejects any flush+evacuate that would, and falls back honestly).
    if am:
        io_edges = list(getattr(view, "edge_anchor_io_edges", None) or [])
        anchored_projected, anchor_notes = apply_edge_anchor(
            projected, sizes, am, min_gap=min_gap, io_edges=io_edges)
        projected = anchored_projected
        notes.extend(anchor_notes)

    # deviation notes: every device whose assigned band left its home band,
    # plus in-band members that drifted beyond the tolerance from their well.
    deviations: List[str] = []
    final_relaxed = relaxed if relaxed is not None else dict(seed)
    anchor = {m: float(p.anchor_y[i]) for i, m in enumerate(p.members)}
    for m in p.members:
        home = p.home_band[m]
        got = assignment[m]
        dy = final_relaxed[m][1] - anchor[m]
        if got != home:
            if m in p.mirror_members:
                deviations.append(
                    f"{m}: left style band {home} (band line "
                    f"y={p.well_y[home]:.2f}) for band {got} -- declared "
                    "mirror-rigid pair convergence (mirror_pairs_rigid): "
                    "the mirror posture outranked the band well; review, "
                    "or drop the pair declaration to keep the row"
                )
            else:
                deviations.append(
                    f"{m}: left style band {home} (band line "
                    f"y={p.well_y[home]:.2f}) for band {got} (dy={dy:+.2f} um) "
                    "-- connectivity/spreading outweighed the band well; "
                    "review, or pin with a declared order/row intent"
                )
        elif abs(dy) > float(knobs.band_dev_tol_um) + 1e-9 and \
                not tag.startswith("seed-fallback"):
            deviations.append(
                f"{m}: stayed in band {home} but sits {dy:+.2f} um off "
                "its band seat (within-band 2D freedom)"
            )

    # P1b mirror records: the placement's AXIS OUTPUT, measured on the FINAL
    # projected placement (the equality edges carry the posture through the
    # press, so axis = the exact pair midpoint). ``realized`` is the honest
    # posture verdict -- a press-refusal fallback that lost the posture is
    # recorded False and its axis must not be consumed downstream.
    mirror_records: List[Dict[str, Any]] = []
    for (left, right), kind, host, src in zip(
            p.mirror_pairs, p.mirror_kind, p.mirror_host, p.mirror_src):
        xa, ya = projected[left]
        xb, yb = projected[right]
        realized = abs(float(ya) - float(yb)) <= 1e-6 and \
            abs(float(xb) - float(xa)) > 0.0
        mirror_records.append({
            "a": left, "b": right, "kind": str(kind),
            "axis_x_um": (float(xa) + float(xb)) / 2.0,
            "d_um": abs(float(xb) - float(xa)) / 2.0,
            "y_um": (float(ya) + float(yb)) / 2.0,
            "band": int(assignment.get(left, host)),
            "source_bands": (int(src[0]), int(src[1])),
            "realized": bool(realized),
        })
    mirror_records.sort(key=lambda rec: (rec["a"], rec["b"]))

    report = RelaxReport(
        engine="relax",
        selected=tag,
        legality=_LEGALITY_TEXT,
        restart_energies=tuple(restart_energies),
        selected_energy=sel_energy,
        energy_terms=terms,
        seed_placement=dict(seed),
        relaxed_placement=final_relaxed
        if not tag.startswith("seed-fallback") else dict(seed),
        home_band=dict(p.home_band),
        band_assignment=dict(assignment),
        band_deviations=tuple(deviations),
        seed_outline_um2=_dev_outline_um2(seed, sizes),
        projected_outline_um2=_dev_outline_um2(projected, sizes),
        seed_hpwl_um=_hpwl_um(p, seed),
        projected_hpwl_um=_hpwl_um(p, projected),
        notes=tuple(notes),
        mirror_pairs=tuple(mirror_records),
    )
    return projected, report


# --------------------------------------------------------------------------- #
# the chain hook (solve_hierarchical_nonoverlap calls this on its own output)
# --------------------------------------------------------------------------- #
def refine_banded_placement(
    hier: Any,
    seed_pos: Mapping[str, _XY],
    sizes: Mapping[str, _XY],
    *,
    min_gap: float,
    band_channel: float,
    channel_um_per_gap: Optional[Mapping[int, float]] = None,
    knobs: Optional[RelaxKnobs] = None,
) -> Tuple[Dict[str, _XY], RelaxReport]:
    """The engine='relax' refinement stage over the band solve's OWN output
    (the design-self-produced seed). ``hier.relax_intent`` carries the
    intent view (nets / declared pairs); the band partition and the rail
    pseudo rows come from ``hier.bands``. Never a second placement path:
    the seed IS the chain's output and the projection IS the existing
    compaction press."""
    view = getattr(hier, "relax_intent", None)
    if view is None:
        raise ValueError(
            "refine_banded_placement: hier carries no relax_intent view "
            "(build the hier from a derive_design_sketch(engine='relax') "
            "sketch view)"
        )
    # P1b parameter ride-along: a derive_design_sketch(mirror_pairs_rigid=
    # True) request reaches this stage on the intent VIEW (the same channel
    # the engine request rides) -- fold it into the knobs face here so the
    # one chain needs no extra plumbing.
    if bool(getattr(view, "relax_mirror_rigid", False)):
        knobs = replace(knobs or RelaxKnobs(), mirror_pairs_rigid=True)
    from clay.layout.extract.relgraph import RESERVATION_ROW_PREFIX

    row_of: Dict[str, int] = {}
    rail_rows: set = set()
    for b in getattr(hier, "bands", []) or []:
        for n in b.devices:
            row_of[str(n)] = int(b.band_id)
            if str(n).startswith(RESERVATION_ROW_PREFIX):
                rail_rows.add(int(b.band_id))
    missing = sorted(n for n in seed_pos if str(n) not in row_of)
    if missing:
        raise ValueError(
            f"refine_banded_placement: seed names missing from the band "
            f"partition: {missing}"
        )
    return _refine(
        view, seed_pos, sizes, row_of, frozenset(rail_rows),
        min_gap=min_gap, band_channel=band_channel,
        channel_um_per_gap=channel_um_per_gap, knobs=knobs,
        press_mode=str(getattr(getattr(hier, "compaction", None), "mode",
                               "press") or "press"),
        chain_spec=getattr(hier, "compaction", None),
    )


# --------------------------------------------------------------------------- #
# the standalone declared face
# --------------------------------------------------------------------------- #
def relax_placement(
    intent: Any,
    sizes: Dict[str, _XY],
    *,
    style_order: Optional[Any] = None,
    rails: Optional[List[Dict[str, Any]]] = None,
    seed_placement: Optional[Mapping[str, _XY]] = None,
    knobs: Optional[RelaxKnobs] = None,
    balance_rows: bool = False,
    aspect_target: Optional[float] = None,
    channel_track_um: Optional[float] = None,
    min_gap_um: float = 1.0,
    band_channel_um: float = 1.0,
) -> Tuple[Dict[str, _XY], RelaxReport]:
    """Relax-and-project a placement for ``intent`` at the target ``sizes``.

    Runs the ONE placement chain (``derive_design_sketch(style_order=...,
    rails=..., compact=True)`` -> ``build_hierarchical_graph`` ->
    ``solve_hierarchical_nonoverlap``) to produce the SEED, then the
    relaxation + compaction projection of the module docstring.
    ``seed_placement`` (optional) replaces the seed COORDINATES and must be
    the design's own band-solve output for the same inputs (the leakage
    contract: a seed is design-self-produced, never a reference layout);
    the band partition still comes from the chain's sketch.

    Returns ``(placement, report)``; the placement includes any rail pseudo
    rows (strip ``__rail__*`` names exactly as the band-solve consumers do).
    The relaxer asserts NO legality (see :class:`RelaxReport.legality`)."""
    from clay.layout.extract import relgraph as _rg
    from clay.layout.extract import topo_sketch as _ts

    sketch = _ts.derive_design_sketch(
        intent, sizes, style_order=style_order, rails=rails,
        balance_rows=balance_rows, aspect_target=aspect_target,
        channel_track_um=channel_track_um, compact=True,
    )
    solve_sizes = dict(sizes)
    for pseudo, _net, _lab in sketch.rail_reservations:
        bb = sketch.bboxes[pseudo]
        solve_sizes[pseudo] = (float(bb[2]) - float(bb[0]),
                               float(bb[3]) - float(bb[1]))
    view = _ts.sketch_intent_view(intent, sketch)
    hier = _rg.build_hierarchical_graph(view)
    seed, _resid = _rg.solve_hierarchical_nonoverlap(
        hier, solve_sizes, (0.0, 0.0), band_channel_um=band_channel_um,
        min_gap=min_gap_um,
        channel_um_per_gap=_ts.sketch_channel_um_per_gap(
            sketch, channel_track_um) if channel_track_um else None,
    )
    if seed_placement is not None:
        missing = sorted(set(seed) - set(seed_placement))
        if missing:
            raise ValueError(
                f"relax_placement: seed_placement does not cover the chain's "
                f"placement names: {missing}"
            )
        seed = {n: (float(seed_placement[n][0]), float(seed_placement[n][1]))
                for n in seed}

    row_of: Dict[str, int] = {}
    rail_rows: set = set()
    for b in hier.bands:
        for n in b.devices:
            row_of[str(n)] = int(b.band_id)
            if str(n).startswith(_RAIL_PREFIX):
                rail_rows.add(int(b.band_id))
    return _refine(
        view, seed, solve_sizes, row_of, frozenset(rail_rows),
        min_gap=min_gap_um, band_channel=band_channel_um,
        channel_um_per_gap=_ts.sketch_channel_um_per_gap(
            sketch, channel_track_um) if channel_track_um else None,
        knobs=knobs,
        press_mode=str(getattr(sketch.compaction_spec, "mode", "press")
                       or "press"),
        chain_spec=sketch.compaction_spec,
    )


__all__ = [
    "RelaxKnobs",
    "RelaxReport",
    "refine_banded_placement",
    "relax_placement",
]
