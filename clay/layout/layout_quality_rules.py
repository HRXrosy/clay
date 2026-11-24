"""R172 (BL-172-QUAL) -- an OBJECTIVE layout-quality checker that turns "the
layout looks wrong" into measurable pass/fail scores, and in particular catches a
class of defect that DRC + LVS + the inter-net-short union-find all MISS.

The blind spot this module exists to catch
===========================================
Clay's current multi-block layouts pass the real gf180mcu DRC deck, pass the real
gf180mcu LVS deck, and pass the geometric inter-net-short union-find -- yet they
are not what they claim to be. The concrete failure (documented in
``whole_pll5_lvs_v170`` and ``lf_multiblock_lvs_v170``): the shared ``vdd`` / ``vss``
power nets are NOT physically connected. Each block carries its own metal island
with a ``vdd`` (or ``vss``) metal1 TEXT label, the shared supply rails are left
UNROUTED, and the LVS deck unifies those islands ONLY because they share the same
TEXT label name (``labels(34,10)`` + ``connect_global(sub, "vss")``). On silicon
those islands are OPENS -- there is no metal path between them. DRC sees each
island as legal; LVS "matches" because the label names collapse the nets; the
existing union-find checks for SHORTS (two DIFFERENT nets touching), not OPENS
(one net split across disconnected metal). So none of the three catches it.

This module's :func:`power_physical_connectivity` is the OPPOSITE of the inter-net
short check. The short check takes shapes already TAGGED with their net and asks
"do two distinct tags share a component?". This check is net-AGNOSTIC: it takes
ALL metal/via geometry, computes the physical connected components (same-layer
touch + via bridges), then asks "do all the shapes carrying a given power LABEL
land in ONE component?". If a power net's labelled shapes span N > 1 disconnected
metal components, the net is physically OPEN and the rule FAILS, reporting N.

The five objective rules
========================
1. :func:`power_physical_connectivity` -- the headline rule (above). FAILS the
   label-merge multi-block layouts; this is the demonstration the checker catches
   what DRC/LVS/union-find missed.
2. :func:`rail_position` -- vss is the bottommost horizontal power strip, vdd the
   topmost (configurable convention); reports y-positions + pass/fail.
3. :func:`symmetry_placement` -- declared symmetry pairs are mirror-placed about an
   axis within a RELATIVE tolerance; reports a per-pair + worst residual.
4. :func:`common_centroid` -- a matched group's device centroids coincide (relative
   to the group span); reports a centroid-spread residual.
5. :func:`area_ratio` -- cell bbox area / summed device active area; FAILS above a
   configurable RELATIVE threshold (catches the 28x sprawl). Per the project rule
   the threshold is a RATIO, never an absolute um^2.

:func:`score_layout` runs whichever rules the caller supplies inputs for and
returns the aggregate.

Honesty contract
----------------
Every Clay layout module + the union-find bridge is imported READ-ONLY; nothing
here mutates them. The connectivity rule reads a real emitted GDS with
``klayout.db`` and computes the physical components itself (it does NOT trust any
label-based net assignment). A rule reports its real measured value; thresholds
are explicit and relative; a FAIL is reported as a FAIL, never massaged to a pass.
The whole point of this checker is to FAIL the current layouts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# --- gf180mcu GDS layer keys (verified against the committed layout modules) ---
# M1=(34,0), M2=(36,0), M3=(42,0), M4=(46,0); the metal1 TEXT-label layer is
# (34,10); the via layers that bridge two metals are Via1=(35,0) Via2=(38,0)
# Via3=(40,0). These mirror coclean_5tota_v163 / maze_router*_v16x / substrate_tap_tie
# (imported nowhere -- duplicated as plain constants so this module stays standalone).
LayerKey = Tuple[int, int]
Bbox = Tuple[int, int, int, int]

GF180_METAL_LAYERS: Tuple[LayerKey, ...] = ((34, 0), (36, 0), (42, 0), (46, 0))
GF180_VIA_LAYERS: Tuple[LayerKey, ...] = ((35, 0), (38, 0), (40, 0))
GF180_METAL1_LABEL_KEY: LayerKey = (34, 10)

DEFAULT_POWER_NETS: Tuple[str, ...] = ("vdd", "vss")


# --- pure geometry helpers ----------------------------------------------------


def _normalise_bbox(b: Bbox) -> Bbox:
    x0, y0, x1, y1 = b
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def _touch_or_overlap(a: Bbox, b: Bbox) -> bool:
    """Inclusive AABB touch/overlap (edges/corners touching counts) -- the same
    rule the committed net-connectivity union-find uses."""
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def _point_in_bbox(x: int, y: int, b: Bbox) -> bool:
    return b[0] <= x <= b[2] and b[1] <= y <= b[3]


def _bbox_area(b: Bbox) -> int:
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def _layer_connects(a: LayerKey, b: LayerKey, via_layers: frozenset) -> bool:
    """Two touching shapes are physically connected iff they are on the SAME
    metal layer, or at least one is a via (a via bridges two metals)."""
    if a == b:
        return True
    return a in via_layers or b in via_layers


# --- the physical union-find over metal/via geometry (net-AGNOSTIC) ------------


@dataclass
class _PhysicalComponents:
    """The result of a physical union-find over metal+via shapes.

    ``shapes`` is the input ``[(layer_key, bbox), ...]`` list (input order);
    ``shape_root`` is the component root index per shape; ``component_count`` is
    the number of distinct components."""

    shapes: List[Tuple[LayerKey, Bbox]]
    shape_root: List[int]

    @property
    def component_count(self) -> int:
        return len({r for r in self.shape_root})

    def component_of_point(self, x: int, y: int, layer_filter=None) -> Optional[int]:
        """Return the component root of the shape under point ``(x, y)`` (the first
        matching shape; ``layer_filter`` optionally restricts to a layer set).
        ``None`` if no shape covers the point."""
        for idx, (k, b) in enumerate(self.shapes):
            if layer_filter is not None and k not in layer_filter:
                continue
            if _point_in_bbox(x, y, b):
                return self.shape_root[idx]
        return None


def physical_components(
    shapes: Sequence[Tuple[LayerKey, Bbox]],
    *,
    via_layers: Sequence[LayerKey] = GF180_VIA_LAYERS,
) -> _PhysicalComponents:
    """Compute the PHYSICAL connected components of metal/via geometry.

    Unlike the committed inter-net-short union-find (which is GIVEN each shape's
    net and finds two distinct nets that touch), this is net-AGNOSTIC: two shapes
    merge iff their bboxes touch/overlap AND they share a metal layer or one is a
    via. This is the true silicon connectivity -- a net is physically one node only
    if all its geometry lands in one component here.

    ``shapes`` is ``[(layer_key, bbox), ...]``. Returns a :class:`_PhysicalComponents`.
    """
    via_set = frozenset(tuple(v) for v in via_layers)
    items: List[Tuple[LayerKey, Bbox]] = [
        (tuple(k), _normalise_bbox(b)) for (k, b) in shapes
    ]
    n = len(items)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        ki, bi = items[i]
        for j in range(i + 1, n):
            kj, bj = items[j]
            if not _touch_or_overlap(bi, bj):
                continue
            if _layer_connects(ki, kj, via_set):
                union(i, j)

    shape_root = [find(idx) for idx in range(n)]
    return _PhysicalComponents(shapes=items, shape_root=shape_root)


# --- GDS readers (klayout.db) -------------------------------------------------


def read_metal_shapes_from_gds(
    gds_path,
    *,
    metal_layers: Sequence[LayerKey] = GF180_METAL_LAYERS,
    via_layers: Sequence[LayerKey] = GF180_VIA_LAYERS,
    topcell: Optional[str] = None,
) -> List[Tuple[LayerKey, Bbox]]:
    """Read every metal + via shape from a GDS as ``[(layer_key, bbox_dbu), ...]``.

    Uses ``klayout.db``; bboxes are in database units (nm at dbu=0.001). Raises
    ``ImportError`` if klayout.db is unavailable (the caller gates on
    :func:`klayout_db_available`)."""
    import klayout.db as db

    ly = db.Layout()
    ly.read(str(gds_path))
    top = ly.cell(topcell) if topcell is not None else ly.top_cell()
    if top is None:
        raise ValueError(f"topcell {topcell!r} not found in {gds_path}")

    wanted = frozenset(tuple(k) for k in metal_layers) | frozenset(
        tuple(k) for k in via_layers
    )
    li_by_key: Dict[LayerKey, int] = {}
    for li in ly.layer_indexes():
        info = ly.get_info(li)
        li_by_key[(info.layer, info.datatype)] = li

    shapes: List[Tuple[LayerKey, Bbox]] = []
    for k in wanted:
        li = li_by_key.get(k)
        if li is None:
            continue
        for s in top.each_shape(li):
            if s.is_text():
                continue
            b = s.bbox()
            shapes.append((k, (b.left, b.bottom, b.right, b.top)))
    return shapes


def read_power_labels_from_gds(
    gds_path,
    *,
    power_net_names: Sequence[str] = DEFAULT_POWER_NETS,
    label_layer: LayerKey = GF180_METAL1_LABEL_KEY,
    topcell: Optional[str] = None,
) -> List[Tuple[str, int, int]]:
    """Read the power-net TEXT labels as ``[(net_name, x_dbu, y_dbu), ...]``.

    Only TEXT objects whose string is one of ``power_net_names`` on ``label_layer``
    are returned. Uses ``klayout.db``."""
    import klayout.db as db

    ly = db.Layout()
    ly.read(str(gds_path))
    top = ly.cell(topcell) if topcell is not None else ly.top_cell()
    if top is None:
        raise ValueError(f"topcell {topcell!r} not found in {gds_path}")
    wanted = {str(n) for n in power_net_names}
    li_by_key: Dict[LayerKey, int] = {}
    for li in ly.layer_indexes():
        info = ly.get_info(li)
        li_by_key[(info.layer, info.datatype)] = li
    li = li_by_key.get(tuple(label_layer))
    out: List[Tuple[str, int, int]] = []
    if li is None:
        return out
    for s in top.each_shape(li):
        if not s.is_text():
            continue
        t = s.text
        name = t.string
        if name in wanted:
            disp = t.trans.disp
            out.append((name, int(disp.x), int(disp.y)))
    return out


def klayout_db_available() -> bool:
    """True iff ``klayout.db`` can be imported (the connectivity rule needs it)."""
    try:
        import klayout.db  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


# --- rule results -------------------------------------------------------------


@dataclass
class RuleResult:
    """One objective rule's verdict: a quantitative score + pass/fail + detail."""

    name: str
    passed: bool
    score: float
    detail: str = ""
    data: Dict[str, object] = field(default_factory=dict)


@dataclass
class PowerConnectivityResult:
    """The headline power-physical-connectivity verdict.

    ``per_net`` maps each power net -> the number of DISTINCT physical metal
    components its labelled shapes land in. A net is physically connected iff that
    count is exactly 1. ``passed`` iff EVERY checked power net is single-component.
    ``islands`` is the sum of (components - 1) over the disconnected nets (the
    number of extra, disconnected islands).
    """

    name: str
    passed: bool
    score: float
    per_net: Dict[str, int]
    per_net_labels: Dict[str, int]
    total_metal_components: int
    detail: str = ""

    @property
    def islands(self) -> int:
        return sum(max(0, c - 1) for c in self.per_net.values())

    def disconnected_nets(self) -> List[str]:
        return sorted(n for n, c in self.per_net.items() if c > 1)


# --- RULE 1: power physical connectivity (the headline) -----------------------


def power_physical_connectivity_from_shapes(
    metal_via_shapes: Sequence[Tuple[LayerKey, Bbox]],
    power_labels: Sequence[Tuple[str, int, int]],
    *,
    via_layers: Sequence[LayerKey] = GF180_VIA_LAYERS,
    metal_layers: Sequence[LayerKey] = GF180_METAL_LAYERS,
) -> PowerConnectivityResult:
    """Core of RULE 1, operating on already-extracted shapes + label points (so it
    can run on synthetic geometry AND on a real GDS).

    For each power label point, find which physical metal component the metal under
    that point belongs to; group the components per net name. A power net whose
    labelled shapes touch more than one distinct component is physically OPEN
    (separate metal islands merged only by sharing a label name) -- the rule FAILS
    and reports the island count.
    """
    comp = physical_components(metal_via_shapes, via_layers=via_layers)
    metal_filter = frozenset(tuple(k) for k in metal_layers)

    per_net_components: Dict[str, set] = {}
    per_net_labels: Dict[str, int] = {}
    unplaced: Dict[str, int] = {}
    for name, x, y in power_labels:
        per_net_labels[name] = per_net_labels.get(name, 0) + 1
        root = comp.component_of_point(x, y, layer_filter=metal_filter)
        if root is None:
            unplaced[name] = unplaced.get(name, 0) + 1
            continue
        per_net_components.setdefault(name, set()).add(root)

    per_net = {n: len(roots) for n, roots in per_net_components.items()}
    # a label sitting over no metal is itself a defect (a floating label).
    for n, cnt in unplaced.items():
        per_net.setdefault(n, 0)

    passed = bool(per_net) and all(c == 1 for c in per_net.values()) and not unplaced
    # score: 1.0 when every net is single-component; otherwise the fraction of
    # power nets that are single-component (so partial connectivity scores partial).
    if per_net:
        single = sum(1 for c in per_net.values() if c == 1)
        score = single / len(per_net)
    else:
        score = 0.0

    bits = []
    for n in sorted(per_net):
        c = per_net[n]
        tag = "OK" if c == 1 else f"DISCONNECTED({c} islands)"
        extra = f", {unplaced[n]} label(s) over no metal" if n in unplaced else ""
        bits.append(f"{n}: {c} component(s) over {per_net_labels.get(n,0)} label(s) [{tag}{extra}]")
    detail = "; ".join(bits) if bits else "no power labels found"

    return PowerConnectivityResult(
        name="power_physical_connectivity",
        passed=passed,
        score=score,
        per_net=per_net,
        per_net_labels=per_net_labels,
        total_metal_components=comp.component_count,
        detail=detail,
    )


def power_physical_connectivity(
    gds_path,
    *,
    power_net_names: Sequence[str] = DEFAULT_POWER_NETS,
    metal_layers: Sequence[LayerKey] = GF180_METAL_LAYERS,
    via_layers: Sequence[LayerKey] = GF180_VIA_LAYERS,
    label_layer: LayerKey = GF180_METAL1_LABEL_KEY,
    topcell: Optional[str] = None,
) -> PowerConnectivityResult:
    """RULE 1 (headline): does each power net's labelled metal land in ONE physical
    component?

    Reads ``gds_path`` with ``klayout.db``, gathers all metal+via geometry +
    every power TEXT label, runs a net-AGNOSTIC physical union-find, and for each
    power net counts the DISTINCT components its labels touch. FAILS (and reports
    N disconnected islands) when a power net spans more than one component -- the
    label-merge OPEN that DRC + LVS + the inter-net-short union-find all miss.

    Raises ``ImportError`` if ``klayout.db`` is unavailable.
    """
    shapes = read_metal_shapes_from_gds(
        gds_path, metal_layers=metal_layers, via_layers=via_layers, topcell=topcell
    )
    labels = read_power_labels_from_gds(
        gds_path,
        power_net_names=power_net_names,
        label_layer=label_layer,
        topcell=topcell,
    )
    return power_physical_connectivity_from_shapes(
        shapes, labels, via_layers=via_layers, metal_layers=metal_layers
    )


# --- RULE 2: rail position ----------------------------------------------------


def rail_position(
    rail_bboxes: Dict[str, Bbox],
    *,
    vss_net: str = "vss",
    vdd_net: str = "vdd",
    vss_is_bottom: bool = True,
) -> RuleResult:
    """RULE 2 (DIGITAL std-cell convention only; INFORMATIONAL for analog).

    Reports whether vss is the bottommost horizontal power strip and vdd the
    topmost. This is a standard-cell row convention -- it is NOT an analog-layout
    correctness rule. Analog power/ground rails are placed by IR-drop, current
    path, and device-abutment needs; there is NO requirement that vdd and vss be
    top/bottom or vertically symmetric. Treat ``passed``/``score`` here as an
    optional convention report, never as a defect for an analog cell. (Earlier
    guidance that required vdd/vss top-bottom symmetry was retracted -- no such
    rule exists in analog layout.)

    ``rail_bboxes`` maps net name -> the power strip's bbox (dbu). Reports both
    rail y-centres and passes iff their order matches the convention. The score is
    the normalised vertical separation (clamped to [0, 1]); 0 when the order is
    wrong.
    """
    if vss_net not in rail_bboxes or vdd_net not in rail_bboxes:
        return RuleResult(
            name="rail_position",
            passed=False,
            score=0.0,
            detail=f"missing rail bbox(es): need {vss_net!r} and {vdd_net!r}",
            data={"present": sorted(rail_bboxes)},
        )
    vb = _normalise_bbox(rail_bboxes[vss_net])
    db_ = _normalise_bbox(rail_bboxes[vdd_net])
    vss_yc = (vb[1] + vb[3]) / 2.0
    vdd_yc = (db_[1] + db_[3]) / 2.0
    if vss_is_bottom:
        passed = vss_yc < vdd_yc
    else:
        passed = vss_yc > vdd_yc
    sep = abs(vdd_yc - vss_yc)
    span = max(vb[3], db_[3]) - min(vb[1], db_[1])
    norm = (sep / span) if span > 0 else 0.0
    score = min(1.0, norm) if passed else 0.0
    conv = f"vss {'below' if vss_is_bottom else 'above'} vdd"
    detail = (
        f"vss y-centre={vss_yc:.0f}, vdd y-centre={vdd_yc:.0f}, separation={sep:.0f} dbu; "
        f"convention ({conv}) {'satisfied' if passed else 'VIOLATED'}"
    )
    return RuleResult(
        name="rail_position",
        passed=bool(passed),
        score=float(score),
        detail=detail,
        data={"vss_yc": vss_yc, "vdd_yc": vdd_yc, "separation": sep},
    )


# --- RULE 3: symmetry placement -----------------------------------------------


def _bbox_center(b: Bbox) -> Tuple[float, float]:
    b = _normalise_bbox(b)
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def symmetry_placement(
    device_bboxes: Dict[str, Bbox],
    symmetry_groups: Sequence[Tuple[str, str]],
    *,
    axis: str = "x",
    axis_coord: Optional[float] = None,
    rel_tol: float = 0.05,
) -> RuleResult:
    """RULE 3: each declared symmetry pair ``(a, b)`` is mirror-placed about a
    vertical (``axis="x"``) or horizontal (``axis="y"``) mirror line.

    For ``axis="x"`` the mirror line is a vertical line at ``axis_coord`` (default:
    the midpoint of all paired devices' x-centres); a pair is symmetric iff
    ``cx_a + cx_b == 2 * axis_coord`` (and the off-axis coordinate matches). The
    residual is the absolute placement error normalised by the layout span (a
    RELATIVE tolerance, per the project rule -- never an absolute dbu). Passes iff
    the WORST pair residual is within ``rel_tol``.
    """
    if not symmetry_groups:
        return RuleResult(
            name="symmetry_placement", passed=False, score=0.0,
            detail="no symmetry groups declared",
        )
    centers = {k: _bbox_center(b) for k, b in device_bboxes.items()}
    on_idx, off_idx = (0, 1) if axis == "x" else (1, 0)

    # the mirror line: caller-given, else the mean of all paired on-axis centres.
    paired_names = [n for pair in symmetry_groups for n in pair]
    missing = [n for n in paired_names if n not in centers]
    if missing:
        return RuleResult(
            name="symmetry_placement", passed=False, score=0.0,
            detail=f"missing device bbox(es): {sorted(set(missing))}",
        )
    on_vals = [centers[n][on_idx] for n in paired_names]
    if axis_coord is None:
        axis_coord = sum(on_vals) / len(on_vals)

    # the normalising span: the on-axis extent of all paired devices (so residual
    # is relative). Fall back to 1 to avoid divide-by-zero on a degenerate layout.
    span = max(on_vals) - min(on_vals)
    span = span if span > 0 else 1.0

    pair_residuals: List[Tuple[str, str, float]] = []
    for a, b in symmetry_groups:
        ca, cb = centers[a], centers[b]
        # on-axis mirror error: (ca_on + cb_on) should equal 2*axis_coord.
        on_err = abs((ca[on_idx] + cb[on_idx]) - 2.0 * axis_coord)
        # off-axis: the two devices should share the same off-axis coordinate.
        off_err = abs(ca[off_idx] - cb[off_idx])
        resid = (on_err + off_err) / span
        pair_residuals.append((a, b, resid))

    worst = max(r for _, _, r in pair_residuals)
    passed = worst <= rel_tol
    score = max(0.0, 1.0 - worst / rel_tol) if rel_tol > 0 else (1.0 if worst == 0 else 0.0)
    score = min(1.0, score)
    bits = [f"({a},{b}) resid={r:.4f}" for a, b, r in pair_residuals]
    detail = (
        f"axis={axis} at {axis_coord:.0f}, span={span:.0f}, worst residual={worst:.4f} "
        f"vs rel_tol={rel_tol}: {'PASS' if passed else 'FAIL'} | " + ", ".join(bits)
    )
    return RuleResult(
        name="symmetry_placement",
        passed=bool(passed),
        score=float(score),
        detail=detail,
        data={"worst_residual": worst, "pair_residuals": pair_residuals},
    )


# --- RULE 4: common centroid --------------------------------------------------


def common_centroid(
    device_bboxes: Dict[str, Bbox],
    group: Sequence[str],
    *,
    rel_tol: float = 0.05,
) -> RuleResult:
    """RULE 4: a matched group is laid out common-centroid (the group is BALANCED
    about its own geometric centre).

    A common-centroid placement (e.g. an ABBA cross-quad, an interdigitated pair)
    distributes the matched devices symmetrically so the centroid of the device
    centres coincides with the centre of the group's bounding box -- there is no net
    pull toward one side that a process gradient could exploit. This rule measures
    exactly that balance: the offset between the device-centroid mean and the
    group's bounding-box centre, normalised by the group span (a RELATIVE residual,
    per the project rule). A perfectly balanced quad scores ~0; devices clustered to
    one side leave a large offset and FAIL.

    (This is the right metric -- a per-device "distance to the shared centroid"
    would wrongly penalise the symmetric AROUND-the-centroid placement that a true
    common-centroid layout REQUIRES.)
    """
    if len(group) < 2:
        return RuleResult(
            name="common_centroid", passed=False, score=0.0,
            detail="common-centroid needs >= 2 devices",
        )
    missing = [n for n in group if n not in device_bboxes]
    if missing:
        return RuleResult(
            name="common_centroid", passed=False, score=0.0,
            detail=f"missing device bbox(es): {sorted(set(missing))}",
        )
    centers = [_bbox_center(device_bboxes[n]) for n in group]
    # the mean of the device centres (the placement's centroid).
    gx = sum(c[0] for c in centers) / len(centers)
    gy = sum(c[1] for c in centers) / len(centers)
    # the geometric centre of the group's bounding box (where a balanced centroid
    # should sit).
    nb = [_normalise_bbox(device_bboxes[n]) for n in group]
    bx0 = min(b[0] for b in nb)
    by0 = min(b[1] for b in nb)
    bx1 = max(b[2] for b in nb)
    by1 = max(b[3] for b in nb)
    bbox_cx = (bx0 + bx1) / 2.0
    bbox_cy = (by0 + by1) / 2.0
    span_x = bx1 - bx0
    span_y = by1 - by0
    span = (span_x ** 2 + span_y ** 2) ** 0.5
    span = span if span > 0 else 1.0
    # the imbalance: how far the centroid is pulled from the group centre.
    offset = ((gx - bbox_cx) ** 2 + (gy - bbox_cy) ** 2) ** 0.5
    norm = offset / span
    passed = norm <= rel_tol
    score = max(0.0, 1.0 - norm / rel_tol) if rel_tol > 0 else (1.0 if norm == 0 else 0.0)
    score = min(1.0, score)
    detail = (
        f"centroid=({gx:.0f},{gy:.0f}) vs group centre=({bbox_cx:.0f},{bbox_cy:.0f}), "
        f"imbalance offset={offset:.0f} dbu, normalised={norm:.4f} vs rel_tol={rel_tol}: "
        f"{'PASS' if passed else 'FAIL'}"
    )
    return RuleResult(
        name="common_centroid",
        passed=bool(passed),
        score=float(score),
        detail=detail,
        data={
            "centroid": (gx, gy),
            "group_centre": (bbox_cx, bbox_cy),
            "normalised_offset": norm,
        },
    )


# --- RULE 5: area ratio -------------------------------------------------------


def area_ratio(
    gds_bbox_area: float,
    total_device_active_area: float,
    *,
    max_ratio: float = 6.0,
) -> RuleResult:
    """RULE 5: cell bbox area / summed device active area. FAILS above
    ``max_ratio`` -- a RELATIVE threshold (a ratio, never an absolute um^2, per the
    project rule), so it is PDK-scale-invariant and catches the sprawl (e.g. the
    28x comparator).

    A compact, well-packed analog cell with routing overhead is typically a few x
    the summed active area; a single-row sprawl with over-the-top routing balloons
    far past that. ``max_ratio`` defaults to 6.0 (configurable).
    """
    if total_device_active_area <= 0:
        return RuleResult(
            name="area_ratio", passed=False, score=0.0,
            detail="total device active area must be > 0",
        )
    ratio = gds_bbox_area / total_device_active_area
    passed = ratio <= max_ratio
    # score: 1.0 at ratio<=1 (impossible-tight ideal), linearly down to 0 at the
    # threshold and below.
    score = max(0.0, min(1.0, (max_ratio - ratio) / max(1e-9, max_ratio - 1.0)))
    detail = (
        f"cell-area / active-area = {ratio:.2f}x vs max_ratio={max_ratio:.2f}x: "
        f"{'PASS' if passed else 'FAIL (sprawl)'}"
    )
    return RuleResult(
        name="area_ratio",
        passed=bool(passed),
        score=float(score),
        detail=detail,
        data={"ratio": ratio, "max_ratio": max_ratio},
    )


# --- aggregate scorer ---------------------------------------------------------


@dataclass
class LayoutScore:
    """The aggregate of every rule that ran, plus a composite pass/fail + score."""

    rules: List[object] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return bool(self.rules) and all(getattr(r, "passed", False) for r in self.rules)

    @property
    def composite_score(self) -> float:
        if not self.rules:
            return 0.0
        return sum(float(getattr(r, "score", 0.0)) for r in self.rules) / len(self.rules)

    def by_name(self) -> Dict[str, object]:
        return {getattr(r, "name", f"rule{i}"): r for i, r in enumerate(self.rules)}

    def summary(self) -> str:
        bits = [
            f"{getattr(r,'name','?')}={'PASS' if getattr(r,'passed',False) else 'FAIL'}"
            f"({getattr(r,'score',0.0):.2f})"
            for r in self.rules
        ]
        return (
            f"layout {'ALL-PASS' if self.all_passed else 'HAS-FAILURES'} "
            f"composite={self.composite_score:.2f}: " + ", ".join(bits)
        )


def score_layout(
    *,
    gds_path=None,
    power_net_names: Sequence[str] = DEFAULT_POWER_NETS,
    metal_layers: Sequence[LayerKey] = GF180_METAL_LAYERS,
    via_layers: Sequence[LayerKey] = GF180_VIA_LAYERS,
    topcell: Optional[str] = None,
    rail_bboxes: Optional[Dict[str, Bbox]] = None,
    device_bboxes: Optional[Dict[str, Bbox]] = None,
    symmetry_groups: Optional[Sequence[Tuple[str, str]]] = None,
    symmetry_axis: str = "x",
    common_centroid_group: Optional[Sequence[str]] = None,
    gds_bbox_area: Optional[float] = None,
    total_device_active_area: Optional[float] = None,
    area_max_ratio: float = 6.0,
    rel_tol: float = 0.05,
) -> LayoutScore:
    """Run whichever rules the caller has inputs for and return the aggregate.

    Every rule is OPTIONAL -- supply ``gds_path`` for the connectivity rule,
    ``rail_bboxes`` for rail position, ``device_bboxes`` + ``symmetry_groups`` for
    symmetry, ``device_bboxes`` + ``common_centroid_group`` for common-centroid,
    and the two area numbers for the area ratio. Rules with no inputs are skipped.
    """
    rules: List[object] = []

    if gds_path is not None:
        rules.append(
            power_physical_connectivity(
                gds_path,
                power_net_names=power_net_names,
                metal_layers=metal_layers,
                via_layers=via_layers,
                topcell=topcell,
            )
        )
    if rail_bboxes is not None:
        rules.append(rail_position(rail_bboxes))
    if device_bboxes is not None and symmetry_groups:
        rules.append(
            symmetry_placement(
                device_bboxes, symmetry_groups, axis=symmetry_axis, rel_tol=rel_tol
            )
        )
    if device_bboxes is not None and common_centroid_group:
        rules.append(
            common_centroid(device_bboxes, common_centroid_group, rel_tol=rel_tol)
        )
    if gds_bbox_area is not None and total_device_active_area is not None:
        rules.append(
            area_ratio(
                gds_bbox_area, total_device_active_area, max_ratio=area_max_ratio
            )
        )
    return LayoutScore(rules=rules)


__all__ = [
    "Bbox",
    "LayerKey",
    "GF180_METAL_LAYERS",
    "GF180_VIA_LAYERS",
    "GF180_METAL1_LABEL_KEY",
    "DEFAULT_POWER_NETS",
    "LayoutScore",
    "PowerConnectivityResult",
    "RuleResult",
    "area_ratio",
    "common_centroid",
    "klayout_db_available",
    "physical_components",
    "power_physical_connectivity",
    "power_physical_connectivity_from_shapes",
    "rail_position",
    "read_metal_shapes_from_gds",
    "read_power_labels_from_gds",
    "score_layout",
    "symmetry_placement",
]
