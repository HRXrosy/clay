"""Placement-symmetry obligations: matched DEVICE CENTROIDS mirror about the axis.

This extends the Clay layout-intent contract (``clay.layout.intent``) from the
net-level ROUTING parasitics it already governs down to the PLACEMENT itself.
The designer's matched-device methodology (intent point 6) wants every matched
device pair placed MIRROR-SYMMETRICALLY: the two devices of a differential pair
sit at the same height (y) and at equal-and-opposite distances from the cell's
vertical axis. A skewed placement is a parasitic-balance defect that no amount of
symmetric routing can fully repair, so it is worth checking as a first-class
obligation -- the same declare -> check shape the parasitic budgets already use.

Where the parasitic-intent checker measures coupling caps from a PEX extraction,
this checker measures GEOMETRY straight off the placed cell, so it needs no Magic
/ PEX and is fast.

Mapping net-level diff pairs to DEVICES
=======================================
``LayoutIntent.diff_pairs`` are NET pairs (e.g. ``(intern, interp)``,
``(crossn, crossp)``). A net is carried by one terminal of several devices, so a
diff pair is physically realised by one or MORE matched DEVICE pairs. The matched
device pairs are DERIVED FROM THE INTENT, not from a per-circuit table: the
declared diff/mirror net pairs induce a net-swap permutation, and a matched device
pair is two devices whose terminal->net signatures map onto each other under that
swap (see :mod:`clay.layout.device_pairing`). The device->net map comes from the
authoritative ``COMP_DEVICE_PLAN`` (``magical_comparator_v171``). A derived pair
``(left, right)`` IMPLEMENTS a diff pair ``(net_a, net_b)`` when there is a
terminal kind ``tk`` (drain / gate / source) on which ``left`` carries ``net_a``
and ``right`` carries ``net_b`` (the mirror-consistent driver), in either net
order. Every implementing device pair gets its own placement obligation, so a
diff pair realised by several device pairs (e.g. ``crossn/crossp`` -> the NMOS
latch, the PMOS latch, the precharge pair, the output-inverter gates) is checked
at every one of them rather than at a single representative.

When the intent-derived pairing yields no device pair for a diff pair, the
checker falls back to the symmetric NAMING convention: the device labels in
``device_centroids`` are matched against the diff-pair nets through the same
``COMP_DEVICE_PLAN`` net membership, pairing a device on ``net_a`` with a device
on ``net_b`` that shares a terminal kind.

Mirror-load (current-mirror) DEVICE placement
==============================================
``LayoutIntent.mirror_pairs`` are single-ended-output CURRENT-MIRROR net pairs
(e.g. ``(outp, outn)`` for the 5T-OTA: ``outp`` is the diode/reference leg,
``outn`` the mirrored output). Their two NODES are asymmetric BY DESIGN, so they
carry no ``matched_cc_balance`` node-cap budget -- but the matched DEVICES that
implement the mirror (the two PMOS load transistors) are still a PLACEMENT
concern: a well-matched mirror wants its two devices placed mirror-symmetrically
so they see the same surround / gradient. :func:`mirror_placement_obligations`
checks exactly that, reusing the SAME centroid + axis + relative-misalignment
machinery as the differential check; the only difference is which DEVICES it maps
the net pair to (the current-mirror LOAD pair, see below) and that the emitted
obligation is tagged ``match_kind="mirror"`` so it is distinguishable from a
differential placement obligation.

Relative thresholds only
========================
Per the project-wide consistency rule, every reported misalignment is RELATIVE
(a fraction of the cell extent), never an absolute nm delta -- so the verdicts
are process- and scale-neutral (scaling every coordinate leaves them unchanged):

  * ``dx_imbalance = |dist_left - dist_right| / cell_width`` where ``dist_*`` is
    each device's |x - axis| distance from the vertical axis;
  * ``dy_offset = |y_left - y_right| / cell_height``.

An obligation PASSES when both are within ``rel_tol``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

# Default RELATIVE placement-symmetry tolerance (a fraction of the cell extent,
# never an absolute nm). A mirror-symmetric placement holds this with room to
# spare; a deliberately shifted device blows past it.
_DEFAULT_PLACEMENT_TOL = 0.02

Bbox = Tuple[int, int, int, int]


# ─────────────────────────────────────────────────────────────────────────────
# device centroids
# ─────────────────────────────────────────────────────────────────────────────

def _term_bboxes(routed_cell) -> Dict[str, Dict[str, Bbox]]:
    """Pull the per-device terminal bboxes (``label -> {s,d,g -> bbox}``).

    The routed comparator stores them on ``cell.placement.term``; a synthetic
    test double may expose ``.term`` directly. Either is accepted so the checker
    works on the real cell AND on a hand-built stand-in.
    """
    placement = getattr(routed_cell, "placement", None)
    term = getattr(placement, "term", None)
    if term is None:
        term = getattr(routed_cell, "term", None)
    if term is None:
        raise TypeError(
            "routed_cell exposes neither .placement.term nor .term; cannot read "
            "device placements"
        )
    return term


def device_centroids(routed_cell) -> Dict[str, Tuple[int, int]]:
    """Centroid ``(x, y)`` of every placed device, keyed by its device label.

    A device's placement bbox is the union of its source / drain / gate terminal
    rails (the geometry the router records per device); the centroid is the
    centre of that union bbox. Returns one entry per device label.
    """
    term = _term_bboxes(routed_cell)
    out: Dict[str, Tuple[int, int]] = {}
    for label, kinds in term.items():
        xs: List[int] = []
        ys: List[int] = []
        for b in kinds.values():
            xs.extend((b[0], b[2]))
            ys.extend((b[1], b[3]))
        if not xs:
            continue
        out[label] = ((min(xs) + max(xs)) // 2, (min(ys) + max(ys)) // 2)
    return out


def _cell_extent(centroids: Dict[str, Tuple[int, int]]) -> Tuple[int, int, int, int]:
    """(min_x, min_y, max_x, max_y) over every device centroid."""
    xs = [c[0] for c in centroids.values()]
    ys = [c[1] for c in centroids.values()]
    return (min(xs), min(ys), max(xs), max(ys))


# ─────────────────────────────────────────────────────────────────────────────
# device <-> net membership + matched-device pairing
# ─────────────────────────────────────────────────────────────────────────────

def _device_net_map() -> Dict[str, Dict[str, str]]:
    """``label -> {'d': drain_net, 'g': gate_net, 's': source_net}`` from the plan.

    The authoritative device->net membership is ``COMP_DEVICE_PLAN`` (imported
    read-only). Used to decide which device pair carries a diff pair's two nets.
    """
    from clay.layout.magical_comparator import COMP_DEVICE_PLAN

    out: Dict[str, Dict[str, str]] = {}
    for (label, _kind, d_net, g_net, s_net, *_rest) in COMP_DEVICE_PLAN:
        out[label] = {"d": d_net, "g": g_net, "s": s_net}
    return out


def _device_net_map_from_cell(routed_cell) -> Dict[str, Dict[str, str]]:
    """``label -> {tk -> net}`` read straight off the ROUTED CELL's own tags.

    The comparator's device->net membership lives in ``COMP_DEVICE_PLAN``
    (:func:`_device_net_map`), but other routed cells (the 5T-OTA) are not in that
    plan and expose no ``_PAIR_LEFT`` annotation. Their authoritative membership is
    the router's own per-terminal net tags: ``routed_cell.tagged`` is a list of
    ``(net, layer_key, bbox)`` and each device terminal rail's bbox in
    ``placement.term[label][tk]`` is tagged with the schematic net it carries. We
    join the two by bbox identity, giving the SAME ``label -> {tk -> net}`` shape
    ``_device_net_map`` returns, but sourced from the cell rather than a plan.

    Returns an empty map if the cell carries no ``.tagged`` list (a generic
    synthetic double); callers fall back to the plan-based map in that case.
    """
    tagged = getattr(routed_cell, "tagged", None)
    if not tagged:
        return {}
    term = _term_bboxes(routed_cell)
    # bbox -> list of (label, terminal-kind) it belongs to
    bbox_owner: Dict[Tuple[int, int, int, int], List[Tuple[str, str]]] = {}
    for label, kinds in term.items():
        for tk, b in kinds.items():
            bbox_owner.setdefault(tuple(b), []).append((label, tk))
    out: Dict[str, Dict[str, str]] = {}
    for entry in tagged:
        # tags are (net, layer_key, bbox); tolerate extra trailing fields.
        net = entry[0]
        b = entry[-1]
        for (label, tk) in bbox_owner.get(tuple(b), ()):
            out.setdefault(label, {})[tk] = str(net)
    return out


def _intent_device_pairs(
    dev_nets: Dict[str, Dict[str, str]],
    intent,
) -> List[Tuple[str, str]]:
    """Matched ``(left, right)`` device pairs DERIVED from the declared symmetry.

    Delegates to :func:`clay.layout.device_pairing.matched_device_pairs`: the
    declared diff/mirror net pairs induce a net-swap permutation, and a matched
    device pair is two devices whose terminal->net signatures map onto each other
    under that swap (an on-axis device is sigma-invariant and excluded). This
    derives the pairing from the INTENT + the device->net membership, with no
    per-circuit table or device label -- so it works on the comparator, the OTA,
    or any differential / mirror circuit alike.
    """
    from clay.layout.device_pairing import matched_device_pairs

    return matched_device_pairs(dev_nets, intent)


def _pairs_for_diff(
    net_a: str,
    net_b: str,
    device_pairs: Sequence[Tuple[str, str]],
    dev_nets: Dict[str, Dict[str, str]],
) -> List[Tuple[str, str]]:
    """Every matched device pair that IMPLEMENTS the diff pair ``(net_a, net_b)``.

    A device pair ``(L, R)`` implements ``(net_a, net_b)`` when some terminal
    kind ``tk`` has ``L[tk] == net_a`` and ``R[tk] == net_b``, in either net
    order; the returned pair is normalised so the first device carries ``net_a``.
    Deduplicated, preserving first-seen order.
    """
    out: List[Tuple[str, str]] = []
    seen = set()
    for (left, right) in device_pairs:
        ln = dev_nets.get(left, {})
        rn = dev_nets.get(right, {})
        for tk in ("d", "g", "s"):
            if ln.get(tk) == net_a and rn.get(tk) == net_b:
                hit = (left, right)
            elif ln.get(tk) == net_b and rn.get(tk) == net_a:
                hit = (right, left)
            else:
                continue
            if hit not in seen:
                seen.add(hit)
                out.append(hit)
            break
    return out


def _pairs_by_naming(
    net_a: str,
    net_b: str,
    centroids: Dict[str, Tuple[int, int]],
    dev_nets: Dict[str, Dict[str, str]],
) -> List[Tuple[str, str]]:
    """Fallback matched-device pairing by net membership + axis side.

    Used when no router matched-device annotation is available. For each terminal
    kind, finds a device on ``net_a`` and a device on ``net_b`` (both present in
    ``centroids``); the device on the LEFT of the axis is the ``net_a`` slot, its
    mirror partner the ``net_b`` slot. Pairs up the closest same-|x| candidates so
    a multi-device net binds its devices to their mirror images, not at random.
    """
    axis = sum(c[0] for c in centroids.values()) / max(1, len(centroids))
    out: List[Tuple[str, str]] = []
    seen = set()
    for tk in ("d", "g", "s"):
        a_devs = [l for l, n in dev_nets.items() if n.get(tk) == net_a and l in centroids]
        b_devs = [l for l, n in dev_nets.items() if n.get(tk) == net_b and l in centroids]
        for la in a_devs:
            # nearest b-device whose |x-axis| distance most closely mirrors la's
            da = abs(centroids[la][0] - axis)
            best = min(
                b_devs,
                key=lambda lb: abs(abs(centroids[lb][0] - axis) - da),
                default=None,
            )
            if best is None:
                continue
            hit = (la, best)
            if hit not in seen:
                seen.add(hit)
                out.append(hit)
    return out


def _mirror_load_pairs_for(
    net_a: str,
    net_b: str,
    dev_nets: Dict[str, Dict[str, str]],
) -> List[Tuple[str, str]]:
    """The current-mirror LOAD device pair carrying the mirror net pair ``(net_a, net_b)``.

    A mirror net pair (e.g. ``(outp, outn)``) is touched by SEVERAL device pairs:
    in a 5T-OTA both the input pair AND the load pair have drains on
    ``outp``/``outn``. The matched DEVICES the mirror methodology cares about are
    the current-mirror LOAD transistors, not the differential input pair. The
    distinguishing signature of a current mirror is its DIODE TIE: the two load
    transistors' GATES are commoned to the same reference node (one of the two
    mirror nets, the diode/reference leg). So we select the pair ``(L, R)`` whose
    DRAINS carry ``net_a``/``net_b`` AND whose GATES are equal to each other (the
    commoned mirror reference). This cleanly excludes the input pair, whose gates
    are the two distinct differential inputs.

    ``(L, R)`` is normalised so ``L`` carries ``net_a`` on its drain. Returns every
    such load pair (typically exactly one), deduplicated in first-seen order. If no
    pair has commoned gates (no diode-tied mirror in the membership), falls back to
    every pair whose drains carry the net pair -- so the obligation is still emitted
    rather than silently dropped.
    """
    drain_pairs: List[Tuple[str, str]] = []
    labels = sorted(dev_nets)
    for i, la in enumerate(labels):
        for lb in labels[i + 1:]:
            na, nb = dev_nets[la], dev_nets[lb]
            if na.get("d") == net_a and nb.get("d") == net_b:
                drain_pairs.append((la, lb))
            elif na.get("d") == net_b and nb.get("d") == net_a:
                drain_pairs.append((lb, la))

    def _gates_commoned(pair: Tuple[str, str]) -> bool:
        ga = dev_nets[pair[0]].get("g")
        gb = dev_nets[pair[1]].get("g")
        return ga is not None and ga == gb

    commoned = [p for p in drain_pairs if _gates_commoned(p)]
    chosen = commoned if commoned else drain_pairs
    out: List[Tuple[str, str]] = []
    seen = set()
    for p in chosen:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# obligations
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PlacementSymmetryObligation:
    """How mirror-symmetric one matched device pair's centroids are.

    ``dx_imbalance`` and ``dy_offset`` are RELATIVE fractions of the cell extent
    (never absolute nm); the obligation PASSES iff both stay within ``rel_tol``.
    """

    pair: Tuple[str, str]            # the matched NET pair this device pair carries
    devices: Tuple[str, str]         # the matched DEVICE labels measured
    dx_imbalance: float
    dy_offset: float
    rel_tol: float
    match_kind: str = "differential"  # "differential" diff pair | "mirror" current-mirror load

    @property
    def status(self) -> str:
        ok = self.dx_imbalance <= self.rel_tol and self.dy_offset <= self.rel_tol
        return "pass" if ok else "fail"

    @property
    def worst(self) -> float:
        """The larger of the two relative misalignments (ranks severity)."""
        return max(self.dx_imbalance, self.dy_offset)

    def as_dict(self) -> Dict[str, object]:
        return {
            "pair": self.pair,
            "devices": self.devices,
            "dx_imbalance": self.dx_imbalance,
            "dy_offset": self.dy_offset,
            "rel_tol": self.rel_tol,
            "status": self.status,
            "match_kind": self.match_kind,
        }


def _measure_pair(
    left: str,
    right: str,
    centroids: Dict[str, Tuple[int, int]],
    axis: float,
    cell_width: int,
    cell_height: int,
) -> Tuple[float, float]:
    """Relative ``(dx_imbalance, dy_offset)`` of one device pair about ``axis``.

    The two devices should sit at EQUAL distances from the axis (so the |x-axis|
    distances match) and at the SAME height. Both numbers are RELATIVE fractions
    of the cell extent, never absolute nm. Shared by the differential and the
    mirror-load checks so the geometry math lives in ONE place.
    """
    (lx, ly) = centroids[left]
    (rx, ry) = centroids[right]
    dx_imbalance = abs(abs(lx - axis) - abs(rx - axis)) / cell_width
    dy_offset = abs(ly - ry) / cell_height
    return dx_imbalance, dy_offset


def placement_symmetry_obligations(
    routed_cell,
    intent,
    *,
    axis: Optional[float] = None,
    rel_tol: float = _DEFAULT_PLACEMENT_TOL,
) -> List[Dict[str, object]]:
    """One placement-symmetry obligation per matched DEVICE pair of each diff pair.

    For every ``intent.diff_pairs`` net pair, finds the matched device pair(s)
    that implement it (the intent-derived net-swap pairing first, the symmetric
    naming convention as a fallback) and measures how mirror-symmetric the two
    devices' centroids are about the vertical ``axis``:

      * the two devices should sit at EQUAL distances from the axis ->
        ``dx_imbalance = |dist_left - dist_right| / cell_width``;
      * at the SAME height -> ``dy_offset = |y_left - y_right| / cell_height``.

    ``axis`` defaults to the mean x of every device centroid (a symmetric cell's
    axis), but a caller may pass an explicit axis (e.g. the matched pairs'
    midpoint, or a known 0). ``cell_width`` / ``cell_height`` are the centroid
    extents (>= 1 to avoid a zero divide on a degenerate cell). Both reported
    numbers are RELATIVE fractions; an obligation passes iff both are within
    ``rel_tol``.

    Returns a list of plain dicts (one per device pair), each carrying the
    diff-pair nets, the device labels measured, the two relative misalignments,
    ``rel_tol``, a ``pass`` / ``fail`` status and ``match_kind="differential"``.
    """
    centroids = device_centroids(routed_cell)
    if not centroids:
        return []
    x0, y0, x1, y1 = _cell_extent(centroids)
    cell_width = max(1, x1 - x0)
    cell_height = max(1, y1 - y0)
    if axis is None:
        axis = sum(c[0] for c in centroids.values()) / len(centroids)

    dev_nets = _device_net_map()
    device_pairs = _intent_device_pairs(dev_nets, intent)

    obligations: List[Dict[str, object]] = []
    emitted = set()
    for (net_a, net_b) in intent.diff_pairs:
        impl = _pairs_for_diff(net_a, net_b, device_pairs, dev_nets) if device_pairs else []
        if not impl:
            impl = _pairs_by_naming(net_a, net_b, centroids, dev_nets)
        for (left, right) in impl:
            if left not in centroids or right not in centroids:
                continue
            key = (net_a, net_b, left, right)
            if key in emitted:
                continue
            emitted.add(key)
            dx_imbalance, dy_offset = _measure_pair(
                left, right, centroids, axis, cell_width, cell_height
            )
            ob = PlacementSymmetryObligation(
                pair=(net_a, net_b),
                devices=(left, right),
                dx_imbalance=dx_imbalance,
                dy_offset=dy_offset,
                rel_tol=rel_tol,
                match_kind="differential",
            )
            obligations.append(ob.as_dict())
    return obligations


def mirror_placement_obligations(
    routed_cell,
    intent,
    *,
    axis: Optional[float] = None,
    rel_tol: float = _DEFAULT_PLACEMENT_TOL,
) -> List[Dict[str, object]]:
    """One placement-symmetry obligation per CURRENT-MIRROR LOAD device pair.

    For every ``intent.mirror_pairs`` net pair (a single-ended-output current
    mirror, e.g. ``(outp, outn)`` for the 5T-OTA), finds the matched current-mirror
    LOAD device pair carrying it and measures how mirror-symmetric the two devices'
    centroids are about the vertical ``axis`` -- the SAME relative ``dx_imbalance``
    / ``dy_offset`` criteria as :func:`placement_symmetry_obligations` (reusing the
    shared :func:`_measure_pair` machinery), so a mirror obligation is scale- and
    process-neutral exactly like a differential one. The mirror's two NODES are
    asymmetric by design (hence no cap budget), but its two DEVICES should still be
    placed symmetrically so the mirror matches.

    Mapping the mirror NET pair to its DEVICES
    ------------------------------------------
    A mirror net pair is touched by several device pairs (in a 5T-OTA both the
    input pair and the load pair have drains on ``outp``/``outn``). We need the
    current-mirror LOAD, not the input pair. The device->net membership is read
    straight off the routed cell's own per-terminal net tags
    (:func:`_device_net_map_from_cell`, by bbox identity) -- the cell-level analogue
    of the comparator's ``COMP_DEVICE_PLAN`` (the 5T-OTA is not in that plan and
    exposes no ``_PAIR_LEFT`` annotation). Among the device pairs whose DRAINS carry
    the mirror net pair, :func:`_mirror_load_pairs_for` selects the one whose GATES
    are commoned (the diode-tie signature of a current mirror), which cleanly picks
    the PMOS load pair over the differential input pair. If the cell exposes no
    tags, falls back to the plan-based ``COMP_DEVICE_PLAN`` map.

    Returns a list of plain dicts, each tagged ``match_kind="mirror"`` so a caller
    can tell mirror-device placement obligations apart from differential ones.
    """
    mirror_pairs = getattr(intent, "mirror_pairs", ())
    if not mirror_pairs:
        return []
    centroids = device_centroids(routed_cell)
    if not centroids:
        return []
    x0, y0, x1, y1 = _cell_extent(centroids)
    cell_width = max(1, x1 - x0)
    cell_height = max(1, y1 - y0)
    if axis is None:
        axis = sum(c[0] for c in centroids.values()) / len(centroids)

    dev_nets = _device_net_map_from_cell(routed_cell)
    if not dev_nets:
        dev_nets = _device_net_map()

    obligations: List[Dict[str, object]] = []
    emitted = set()
    for (net_a, net_b) in mirror_pairs:
        for (left, right) in _mirror_load_pairs_for(net_a, net_b, dev_nets):
            if left not in centroids or right not in centroids:
                continue
            key = (net_a, net_b, left, right)
            if key in emitted:
                continue
            emitted.add(key)
            dx_imbalance, dy_offset = _measure_pair(
                left, right, centroids, axis, cell_width, cell_height
            )
            ob = PlacementSymmetryObligation(
                pair=(net_a, net_b),
                devices=(left, right),
                dx_imbalance=dx_imbalance,
                dy_offset=dy_offset,
                rel_tol=rel_tol,
                match_kind="mirror",
            )
            obligations.append(ob.as_dict())
    return obligations


def placement_symmetry_verdict(obligations: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Aggregate a list of placement-symmetry obligations into a verdict.

    Returns ``{passed, worst, worst_pair, failed, n_obligations}``:

      * ``passed`` -- True iff EVERY obligation passes (vacuously True for an
        empty list: a cell that declares no diff pairs has no placement symmetry
        to violate);
      * ``worst`` -- the largest relative misalignment across all obligations
        (max of each obligation's ``dx_imbalance`` / ``dy_offset``);
      * ``worst_pair`` -- the ``(devices, pair)`` of the obligation that owns
        ``worst`` (the pair to fix first), or None;
      * ``failed`` -- the obligation dicts whose status is ``fail``.
    """
    failed = [o for o in obligations if o.get("status") == "fail"]
    worst = 0.0
    worst_pair = None
    for o in obligations:
        w = max(float(o["dx_imbalance"]), float(o["dy_offset"]))
        if w >= worst:
            worst = w
            worst_pair = {"devices": o["devices"], "pair": o["pair"]}
    return {
        "passed": len(failed) == 0,
        "worst": worst,
        "worst_pair": worst_pair,
        "failed": failed,
        "n_obligations": len(obligations),
    }


__all__ = [
    "PlacementSymmetryObligation",
    "device_centroids",
    "placement_symmetry_obligations",
    "mirror_placement_obligations",
    "placement_symmetry_verdict",
]
