"""Reference-layout STYLE extraction as pure RELATIVE ORDER (plan S1).

THE STYLE != LEAKAGE BOUNDARY (iron rule, every function here): the style
channel carries ONLY name-level relative relations (row partition, who is
left-of / below whom). No absolute coordinate, axis coordinate or spacing
value may enter the returned objects -- ``RelativeOrder.positions`` stays
empty and ``axis_x`` stays None. Reading ``intent.device_bboxes`` INSIDE the
extractor is the migration-legal reference read (same legitimacy class as
``relgraph``'s migration use); what crosses the function boundary is relation
names only. Everything produced here is style-transfer provenance: consumers
must label their results ``provenance='style-transfer'`` and must never claim
a leakage-free placement.

Two functions:

* :func:`extract_reference_order` -- reference bboxes -> :class:`RelativeOrder`
  (rows recovered by ``relgraph.cluster_bands``, within-row left_of chain by
  centroid-x, cross-row below relations). Output = names + relations only.
* :func:`map_order_to_design` -- align the reference's device names onto a
  target design's device names (reusing the committed Tversky graph-alignment
  in :mod:`clay.layout.style_transfer`) and carry the reference relations over.
  Coverage is reported honestly; unaligned devices simply do not appear in the
  transferred order (the design-only path fills them). KNOWN LIMIT: fully
  symmetric renamed twins cannot be told apart by structure alone -- the
  alignment is deterministic but the which-twin choice is NOT a claim.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from clay.layout.seqpair_model import RelativeOrder, StyleTransferResult


def rows_to_relative_order(cell: str, rows: Sequence[Sequence[str]]) -> RelativeOrder:
    """Canonical rows -> :class:`RelativeOrder` conversion (names only).

    ``rows`` are bottom-up, each left-to-right. Emits the within-row
    consecutive ``left_of`` chain and the complete cross-row ``below`` relation
    (every device of a lower row is below every device of any higher row), so
    two orders built by this SAME convention are directly comparable by
    :func:`clay.layout.seqpair_model.constraint_graph_distance`. Carries no
    coordinate values by construction.
    """
    clean_rows: List[Tuple[str, ...]] = [
        tuple(str(n) for n in row) for row in rows if row
    ]
    left_of: List[Tuple[str, str]] = []
    for row in clean_rows:
        for i in range(len(row) - 1):
            left_of.append((row[i], row[i + 1]))
    below: List[Tuple[str, str]] = []
    for i in range(len(clean_rows)):
        for j in range(i + 1, len(clean_rows)):
            for u in clean_rows[i]:
                for v in clean_rows[j]:
                    below.append((u, v))
    devices = tuple(sorted(n for row in clean_rows for n in row))
    return RelativeOrder(
        cell=str(cell),
        devices=devices,
        left_of=tuple(left_of),
        below=tuple(below),
        rows=tuple(clean_rows),
    )


def _reference_rows(intent: Any) -> List[Tuple[str, ...]]:
    """Reference bboxes -> bottom-up device ROWS (names only, members
    left-to-right by reference centroid-x).

    ADAPTIVE band clustering (the R2 fidelity fix, GDS re-audit 2026-06-12):
    walking the devices bottom-up by centroid-y, a new band starts when the
    centroid gap to the previous device exceeds HALF THE TALLEST DEVICE of
    the band built so far -- a device whose centre lies beyond the current
    row's half-height envelope cannot belong to that physical row. The
    threshold is purely RELATIVE (device heights of the layout itself, no
    absolute um constant), so it scales across PDKs and, unlike the previous
    ``relgraph.cluster_bands`` median-gap cut, it preserves a short band
    sitting near a tall neighbour band (the reference's MM13+MM26 band was
    being merged into the top band -- the audited R2 infidelity).

    All centroid/height values are consumed INSIDE and discarded; only the
    name partition crosses out (the style channel's leakage boundary).
    """
    bboxes = {
        str(k): tuple(float(v) for v in bb)
        for k, bb in (getattr(intent, "device_bboxes", None) or {}).items()
    }
    if not bboxes:
        return []

    recs = sorted(
        ((bb[1] + bb[3]) / 2.0, bb[3] - bb[1], (bb[0] + bb[2]) / 2.0, n)
        for n, bb in bboxes.items()
    )
    rows_acc: List[List[Tuple[float, str]]] = [[]]
    row_max_h = 0.0
    prev_cy: float = 0.0
    for cy, h, cx, name in recs:
        if rows_acc[-1] and (cy - prev_cy) > 0.5 * row_max_h:
            rows_acc.append([])
            row_max_h = 0.0
        rows_acc[-1].append((cx, name))
        row_max_h = max(row_max_h, h)
        prev_cy = cy
    return [
        tuple(n for _cx, n in sorted(row, key=lambda t: (t[0], t[1])))
        for row in rows_acc
    ]


def extract_reference_order(intent: Any) -> RelativeOrder:
    """Reference bboxes -> RELATIVE order only; no coordinate value crosses out.

    Rows are recovered from the reference geometry by :func:`_reference_rows`
    (adaptive centroid-y band clustering, bottom-up -- band-faithful per the
    R2 audit); within each row the devices are ordered by reference
    centroid-x (ties by name); rows then convert to relations via
    :func:`rows_to_relative_order`. The centroids are consumed INSIDE this
    function and discarded -- the returned :class:`RelativeOrder` holds names
    and name-pairs only (``positions`` empty, ``axis_x`` None), which is the
    style channel's leakage boundary.
    """
    rows = _reference_rows(intent)
    if not rows:
        raise ValueError(
            "extract_reference_order: intent carries no device_bboxes -- the "
            "style extractor needs the reference correspondence "
            "(correspond_to_gds) to recover the reference rows."
        )
    return rows_to_relative_order(str(getattr(intent, "cell", "")), rows)


def _kinds_of(intent: Any) -> Dict[str, str]:
    kinds: Dict[str, str] = {}
    for d in getattr(intent, "devices", []) or []:
        kinds[str(d.name)] = str(getattr(d, "kind", "dev"))
    for p in getattr(intent, "passives", []) or []:
        kinds[str(p.name)] = str(getattr(p, "kind", "passive"))
    return kinds


def _net_neighbors(intent: Any) -> Dict[str, set]:
    """Design-sourced structural context: two devices are neighbours when they
    share a net (pure netlist topology -- no geometry)."""
    by_net: Dict[str, set] = {}
    for rec in list(getattr(intent, "devices", []) or []) + list(
        getattr(intent, "passives", []) or []
    ):
        name = str(rec.name)
        for net in (getattr(rec, "nets", {}) or {}).values():
            by_net.setdefault(str(net), set()).add(name)
    out: Dict[str, set] = {}
    for members in by_net.values():
        for m in members:
            out.setdefault(m, set()).update(members - {m})
    return out


def map_order_to_design(
    ref_order: RelativeOrder, ref_intent: Any, tgt_intent: Any
) -> StyleTransferResult:
    """Align reference device names onto target device names; carry relations.

    Reuses the committed Tversky graph alignment
    (:func:`clay.layout.style_transfer.transfer_style`) with DESIGN-sourced
    features on both sides: device kind (hard gate) + shared-net neighbour sets
    (netlist topology). Self-mapping (``ref_intent is tgt_intent`` or identical
    names) resolves to the identity with coverage 1.0. Cross-design twins that
    are structurally indistinguishable get a deterministic -- but NOT claimed --
    assignment; the honest account is the ``coverage`` + ``note`` fields, and
    unaligned target devices are absent from the transferred order (the
    design-only seriation fills them downstream).

    Style boundary: inputs and outputs are names and name-relations only.
    """
    from clay.layout.style_transfer import transfer_style

    targets = [str(d.name) for d in getattr(tgt_intent, "devices", []) or []]
    targets += [str(p.name) for p in getattr(tgt_intent, "passives", []) or []]

    return transfer_style(
        targets,
        ref_order,
        target_neighbors=_net_neighbors(tgt_intent),
        target_kinds=_kinds_of(tgt_intent),
        reference_neighbors=_net_neighbors(ref_intent),
        reference_kinds=_kinds_of(ref_intent),
    )


def _norm_net(net: Any) -> str:
    return str(net).strip().rstrip("!").lower()


def _power_nets(intent: Any) -> List[str]:
    """All nets the DESIGN classifies as power (width_model.classify_net_role --
    netlist topology + designer naming, never an ad-hoc local table), in their
    original spelling, deterministically ordered."""
    from clay.layout.route.width_model import classify_net_role

    seen: Dict[str, str] = {}
    for rec in list(getattr(intent, "devices", []) or []) + list(
        getattr(intent, "passives", []) or []
    ):
        for net in (getattr(rec, "nets", {}) or {}).values():
            seen.setdefault(_norm_net(net), str(net))
    return sorted(
        (orig for orig in seen.values()
         if classify_net_role(orig, intent) == "power"),
        key=str,
    )


def _declared_rail_entries(intent: Any) -> List[Dict[str, Any]]:
    """Normalise designer declarations from ``intent.rails`` (declaration wins
    over extraction). Recognised forms: ``{'net':, 'side':}`` /
    ``{'net':, 'between_rows':}`` dicts and ``(net, side)`` tuples. The legacy
    ``(net, [devices])`` rail-group form carries no position claim and is left
    for the extractor to position (it only marks the net as a rail)."""
    out: List[Dict[str, Any]] = []
    for entry in getattr(intent, "rails", []) or []:
        net = side = None
        between = None
        legacy_group = False
        if isinstance(entry, dict):
            net = entry.get("net")
            side = entry.get("side")
            between = entry.get("between_rows")
        elif isinstance(entry, (tuple, list)) and len(entry) == 2:
            net = entry[0]
            if isinstance(entry[1], str) and entry[1] in ("top", "bottom"):
                side = entry[1]
            else:
                legacy_group = True  # (net, [devices]) -- no position claim
        if net is None:
            continue
        if legacy_group or (side is None and between is None):
            out.append({
                "net": str(net), "side": None, "between_rows": None,
                "source": "declared",
                "reason": "declared rail carries no relative position; "
                          "left to reference extraction",
            })
            continue
        out.append({
            "net": str(net),
            "side": side if side in ("top", "bottom") else None,
            "between_rows": int(between) if between is not None else None,
            "source": "declared",
            "reason": "designer declaration (intent.rails)",
        })
    return out


def _default_reference_gds() -> str:
    """Absolute path of the read-only external_pdk OTA reference GDS -- the same
    artifact ``anchor_ota._reference_gds_path`` resolves (the production
    style anchor). A missing file degrades the evidence channel honestly
    (design-heuristic fallback), never an exception."""
    import os

    repo = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    return os.path.join(
        repo, "refCircuit", "twoStageAmp", "0208_amp", "project_tsmc18rf",
        "prj_gzc", "CMD", "GDS2", "layout.gds",
    )


def _reference_labels(gds_path: str) -> Optional[List[Tuple[str, float, float]]]:
    """Net-name TEXT labels from the reference GDS label layer ->
    ``[(label_string, x_um, y_um)]``, or None when the evidence channel is
    unavailable (missing/unreadable file or zero labels).

    The label layer / coordinates are reference-side facts consumed INSIDE
    the style extractors and discarded; no caller-visible output may carry
    them (the style != leakage boundary).
    """
    import os

    if not gds_path or not os.path.exists(gds_path):
        return None
    try:
        import klayout.db as kdb

        from clay.layout.extract.route_migrate import OPEN_PDK_LABEL_LAYER

        ly = kdb.Layout()
        ly.read(str(gds_path))
        top = ly.top_cell()
        dbu = ly.dbu
        li = ly.find_layer(int(OPEN_PDK_LABEL_LAYER[0]),
                           int(OPEN_PDK_LABEL_LAYER[1]))
        if li is None:
            return None
        labels: List[Tuple[str, float, float]] = []
        it = top.begin_shapes_rec(li)
        while not it.at_end():
            sh = it.shape()
            if sh.is_text():
                t = sh.text.transformed(it.trans())
                labels.append((str(t.string), t.x * dbu, t.y * dbu))
            it.next()
        return labels or None
    except Exception:
        return None


def _reference_rail_evidence(
    gds_path: str,
) -> Optional[Dict[str, Dict[str, Any]]]:
    """Per labelled net, the labelled metal ISLAND evidence from the
    reference GDS -> ``{norm_net: {'level': int, 'island_bbox': (x0,y0,x1,y1),
    'area': float}}`` (largest merged island containing the label point,
    across the reference metal stack), or None when the evidence channel is
    unavailable.

    INTERNAL evidence only: the returned values include absolute geometry and
    MUST be consumed and discarded inside :func:`extract_rail_intent` /
    :func:`extract_io_edge_intent` -- only relative/categorical facts cross
    those functions' boundaries.
    """
    labels = _reference_labels(gds_path)
    if labels is None:
        return None
    try:
        import klayout.db as kdb

        from clay.layout.extract.route_migrate import OPEN_PDK_METALS

        ly = kdb.Layout()
        ly.read(str(gds_path))
        top = ly.top_cell()
        dbu = ly.dbu

        # merged islands per metal LEVEL (the 'M<k>' names carry the level).
        islands: List[Tuple[int, Tuple[float, float, float, float], float, Any]] = []
        regions: List[Tuple[int, Any]] = []
        for mname, key in sorted(OPEN_PDK_METALS.items()):
            level = int(str(mname).lstrip("M"))
            li = ly.find_layer(int(key[0]), int(key[1]))
            if li is None:
                continue
            reg = kdb.Region(top.begin_shapes_rec(li))
            reg.merge()
            regions.append((level, reg))

        out: Dict[str, Dict[str, Any]] = {}
        for name, lx, lyy in labels:
            pt = kdb.Point(int(round(lx / dbu)), int(round(lyy / dbu)))
            best: Optional[Tuple[float, int, Tuple[float, float, float, float]]] = None
            for level, reg in regions:
                for p in reg.each():
                    if p.bbox().contains(pt) and p.inside(pt):
                        bb = p.bbox()
                        area = p.area() * dbu * dbu
                        bbox = (bb.left * dbu, bb.bottom * dbu,
                                bb.right * dbu, bb.top * dbu)
                        cand = (area, level, bbox)
                        if best is None or cand[0] > best[0]:
                            best = cand
            if best is not None:
                out.setdefault(_norm_net(name), {
                    "level": int(best[1]),
                    "island_bbox": best[2],
                    "area": float(best[0]),
                })
        return out
    except Exception:
        return None


# A labelled island counts as a through-going TRUNK when its x-span covers at
# least this fraction of the device-placement x-span -- a RELATIVE criterion
# (no absolute um), separating an edge power trunk from a stub/pin wire.
_TRUNK_MIN_XSPAN_FRACTION: float = 0.5


def extract_rail_intent(
    intent: Any,
    tech_id: str = "gf180mcu",
    *,
    ref_gds_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Power-rail BAND intent as RELATIVE/CATEGORICAL facts only -- never a
    coordinate.

    STYLE != LEAKAGE boundary (iron rule): every emitted entry carries net
    names, row-ORDER facts ('top'/'bottom' side or a between-rows gap index),
    a metal LEVEL class and prose reasons -- no absolute/axis coordinate or
    spacing value crosses the function boundary. Reference geometry (rows AND
    the labelled trunk islands) is consumed INSIDE (the migration-legal
    reference read, same legitimacy class as :func:`extract_reference_order`)
    and discarded. Output is style-transfer provenance: consumers must never
    present a rail-positioned placement as leakage-free.

    Rail POSITION IS INTENT, NOT A BUILT-IN RULE (the project explicitly
    retracted any "vss bottom / vdd top" convention): a position comes from
      1. a designer declaration in ``intent.rails`` (priority -- see
         :func:`_declared_rail_entries`), else
      2. REFERENCE EVIDENCE (the R1 fidelity fix, GDS re-audit 2026-06-12):
         the net's name label in the reference GDS and the labelled metal
         TRUNK island (largest merged island containing the label point,
         x-span >= half the device-placement x-span). The fact is the
         island's ORDER against the device rows:
           * island centre above every device-row centroid -> ``side='top'``;
           * below every device-row centroid -> ``side='bottom'``;
           * inside the stack -> ``between_rows=k`` (the row-order gap the
             island sits in).
         ``trunk_layer_class`` is the trunk's metal LEVEL (e.g. 2 = M2, a
         categorical fact); ``reference_trunk`` records whether the reference
         actually has a labelled trunk for the net. A power net with NO label
         / no trunk in the reference gets NO corridor fact (``side=None,
         between_rows=None, source='unresolved'``) -- the audited net10
         over-promotion; its width class remains the width model's business.
      3. fallback ONLY when the evidence channel is unavailable (no readable
         reference GDS / no labels): the majority-adjacency heuristic over
         the rail-attached devices' rows, self-declared
         ``source='design-heuristic'`` -- it must never impersonate
         'reference-style'.

    ``tech_id`` is interface surface for future PDK rail-convention DATA; the
    extractor consumes no target-PDK value (the reference metal stack is the
    reference's own, decoded from the GDS via cells). ``ref_gds_path``
    defaults to the canonical external_pdk OTA reference the production style
    anchor reads.

    Returns one dict per rail net: ``{'net', 'side', 'between_rows',
    'trunk_layer_class', 'reference_trunk', 'source', 'reason'}`` where
    ``source`` is ``'declared'`` / ``'reference-style'`` /
    ``'design-heuristic'`` / ``'unresolved'``. ``between_rows`` is None
    whenever edge evidence resolved a side.
    """
    declared = {_norm_net(e["net"]): e for e in _declared_rail_entries(intent)}

    rows = _reference_rows(intent)  # bottom-up; [] when no correspondence
    row_of: Dict[str, int] = {}
    for rid, row in enumerate(rows):
        for name in row:
            row_of[str(name)] = rid
    n_rows = len(rows)

    # reference-frame scalars for the EVIDENCE judgment (consumed + discarded)
    bboxes = {
        str(k): tuple(float(v) for v in bb)
        for k, bb in (getattr(intent, "device_bboxes", None) or {}).items()
    }
    dev_cys = [(bb[1] + bb[3]) / 2.0 for bb in bboxes.values()]
    dev_x_span = (
        max(bb[2] for bb in bboxes.values()) - min(bb[0] for bb in bboxes.values())
    ) if bboxes else 0.0
    row_mean_cy: List[float] = []
    for row in rows:
        cys = [(bboxes[n][1] + bboxes[n][3]) / 2.0 for n in row if n in bboxes]
        row_mean_cy.append(sum(cys) / len(cys) if cys else 0.0)

    evidence = _reference_rail_evidence(
        _default_reference_gds() if ref_gds_path is None else str(ref_gds_path)
    )

    rail_nets = list(_power_nets(intent))
    for key, e in declared.items():
        if all(_norm_net(n) != key for n in rail_nets):
            rail_nets.append(str(e["net"]))  # a declared rail IS a rail

    out: List[Dict[str, Any]] = []
    for net in rail_nets:
        key = _norm_net(net)
        dec = declared.get(key)
        if dec is not None and (dec["side"] is not None
                                or dec["between_rows"] is not None):
            entry = dict(dec)
            entry.setdefault("trunk_layer_class", None)
            entry.setdefault("reference_trunk", None)
            out.append(entry)
            continue

        entry = {
            "net": str(net), "side": None, "between_rows": None,
            "trunk_layer_class": None,
            "reference_trunk": None if evidence is None else False,
            "source": "unresolved", "reason": "",
        }
        if n_rows == 0:
            entry["reason"] = ("no reference row structure (no corresponded "
                               "bboxes); rail position not inferable")
            out.append(entry)
            continue

        if evidence is not None:
            # 2. REFERENCE EVIDENCE: label + trunk island, judged ordinally.
            ev = evidence.get(key)
            if ev is None:
                entry["reason"] = (
                    "reference GDS carries no net-name label for this power "
                    "net -- no reference corridor fact (width class remains "
                    "the width model's)")
            elif dev_x_span <= 0.0 or (
                ev["island_bbox"][2] - ev["island_bbox"][0]
            ) < _TRUNK_MIN_XSPAN_FRACTION * dev_x_span:
                entry["reason"] = (
                    "reference label found but the labelled island is not a "
                    "through-going trunk (x-span below half the device span) "
                    "-- no reference corridor fact")
            else:
                island_yc = (ev["island_bbox"][1] + ev["island_bbox"][3]) / 2.0
                level = int(ev["level"])
                entry["trunk_layer_class"] = level
                entry["reference_trunk"] = True
                entry["source"] = "reference-style"
                if island_yc > max(dev_cys):
                    entry["side"] = "top"
                    entry["reason"] = (
                        f"reference GDS labels this net on an edge M{level} "
                        "trunk island ABOVE every device row (top edge band)")
                elif island_yc < min(dev_cys):
                    entry["side"] = "bottom"
                    entry["reason"] = (
                        f"reference GDS labels this net on an edge M{level} "
                        "trunk island BELOW every device row (bottom edge "
                        "band)")
                else:
                    below = sum(1 for cy in row_mean_cy if cy < island_yc)
                    gap = max(0, min(n_rows - 2, below - 1))
                    entry["between_rows"] = int(gap)
                    entry["reason"] = (
                        f"reference GDS labels this net on an M{level} trunk "
                        "island inside the row stack; trunk band placed in "
                        "the corresponding row-order gap")
            out.append(entry)
            continue

        # 3. FALLBACK (evidence channel unavailable): majority-adjacency
        # heuristic over the rail-attached devices' rows -- self-declared
        # 'design-heuristic', never impersonating reference evidence.
        attached = [
            str(d.name) for d in getattr(intent, "devices", []) or []
            if any(_norm_net((getattr(d, "nets", {}) or {}).get(r, "")) == key
                   for r in ("source", "body"))
        ]
        att_rows = sorted({row_of[a] for a in attached if a in row_of})
        if not att_rows:
            entry["reason"] = ("no rail-attached device (source/body) found "
                               "in the reference rows; rail position not "
                               "inferable")
        elif n_rows == 1:
            entry["reason"] = ("reference has a single row; a rail side is "
                               "not a meaningful relative fact")
        elif att_rows == [0]:
            entry.update(side="bottom", source="design-heuristic",
                         reason="HEURISTIC (no reference GDS evidence): all "
                                "rail-attached devices sit in the bottom "
                                "reference row (edge band)")
        elif att_rows == [n_rows - 1]:
            entry.update(side="top", source="design-heuristic",
                         reason="HEURISTIC (no reference GDS evidence): all "
                                "rail-attached devices sit in the top "
                                "reference row (edge band)")
        else:
            # gap adjacent to the majority-attached row, toward the other
            # attached rows. Row-ORDER arithmetic -- ranks, not coordinates.
            counts: Dict[int, int] = {}
            for a in attached:
                if a in row_of:
                    counts[row_of[a]] = counts.get(row_of[a], 0) + 1
            major = max(sorted(counts), key=lambda r: counts[r])
            others = [row_of[a] for a in attached
                      if a in row_of and row_of[a] != major]
            below_n = sum(1 for r in others if r < major)
            above_n = len(others) - below_n
            if major == 0:
                gap = 0
            elif major == n_rows - 1:
                gap = major - 1
            elif above_n > below_n:
                gap = major
            else:
                gap = major - 1
            entry.update(
                between_rows=int(gap), source="design-heuristic",
                reason=(f"HEURISTIC (no reference GDS evidence): rail "
                        f"attaches devices across reference rows {att_rows}; "
                        f"trunk band placed in the row gap adjacent to "
                        f"majority row {major}, toward the other attached "
                        f"rows"),
            )
        out.append(entry)
    return out


def extract_io_edge_intent(
    intent: Any,
    *,
    ref_gds_path: Optional[str] = None,
    adjacency_um: float = 5.0,
) -> List[Dict[str, Any]]:
    """IO EDGE facts for reference-labelled SIGNAL nets (R4, GDS re-audit
    2026-06-12) -- purely CATEGORICAL output, zero numeric leaves.

    For every net-name label in the reference GDS that names a SIGNAL net of
    the design (power nets are the rail channel's business), emit::

        {'net', 'edge': 'left'|'right'|'top'|'bottom',
         'adjacent_group': [net, ...], 'source': 'reference-style'}

    * ``edge`` -- the extent edge NEAREST the label point, judged against the
      device-placement extent (the union of ``intent.device_bboxes``): a
      categorical side, never a coordinate.
    * ``adjacent_group`` -- the same-edge labelled signal nets whose label
      points sit within ``adjacency_um`` of each other (transitive): a pure
      GROUPING fact (the reference brings the differential pair in together),
      no position value attached.

    All label coordinates and the reference extent are consumed INSIDE and
    discarded (the style != leakage boundary). ``adjacency_um`` is a grouping
    tolerance knob, not a measured reference value, and never appears in the
    output. This is the E (extraction) face only; lowering an edge fact to an
    edge-stub routing target is the consumer's contract.

    Raises ValueError when the intent carries no reference correspondence
    (no extent to judge edges against); returns ``[]`` when the evidence
    channel is unavailable or no signal net is labelled (honest empty, never
    an invented edge).
    """
    from clay.layout.route.width_model import classify_net_role

    bboxes = {
        str(k): tuple(float(v) for v in bb)
        for k, bb in (getattr(intent, "device_bboxes", None) or {}).items()
    }
    if not bboxes:
        raise ValueError(
            "extract_io_edge_intent: intent carries no device_bboxes -- the "
            "IO edge judgment needs the reference correspondence "
            "(correspond_to_gds) for the placement extent."
        )
    x0 = min(bb[0] for bb in bboxes.values())
    y0 = min(bb[1] for bb in bboxes.values())
    x1 = max(bb[2] for bb in bboxes.values())
    y1 = max(bb[3] for bb in bboxes.values())

    labels = _reference_labels(
        _default_reference_gds() if ref_gds_path is None else str(ref_gds_path)
    )
    if labels is None:
        return []

    # normalised net name -> the DESIGN's original spelling.
    spell: Dict[str, str] = {}
    for rec in list(getattr(intent, "devices", []) or []) + list(
        getattr(intent, "passives", []) or []
    ):
        for net in (getattr(rec, "nets", {}) or {}).values():
            spell.setdefault(_norm_net(net), str(net))

    # labelled SIGNAL nets only; first label per net (deterministic file order)
    pts: Dict[str, Tuple[float, float]] = {}
    for name, lx, lyy in labels:
        key = _norm_net(name)
        if key not in spell or key in pts:
            continue
        if classify_net_role(spell[key], intent) == "power":
            continue  # power nets are the rail channel's business
        pts[key] = (float(lx), float(lyy))

    def _edge(p: Tuple[float, float]) -> str:
        px, py = p
        cands = [
            (abs(px - x0), "left"), (abs(x1 - px), "right"),
            (abs(py - y0), "bottom"), (abs(y1 - py), "top"),
        ]
        return min(cands)[1]

    edges = {key: _edge(p) for key, p in pts.items()}

    # same-edge adjacency groups (union-find, transitive within adjacency_um)
    keys = sorted(pts)
    parent = {k: k for k in keys}

    def _find(k: str) -> str:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    tol = float(adjacency_um)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if edges[a] != edges[b]:
                continue
            ax, ay = pts[a]
            bx, by = pts[b]
            if ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 <= tol:
                ra, rb = _find(a), _find(b)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)

    groups: Dict[str, List[str]] = {}
    for k in keys:
        groups.setdefault(_find(k), []).append(k)

    out: List[Dict[str, Any]] = []
    for key in keys:
        grp = sorted(spell[m] for m in groups[_find(key)])
        out.append({
            "net": spell[key],
            "edge": edges[key],
            "adjacent_group": grp,
            "source": "reference-style",
        })
    out.sort(key=lambda e: str(e["net"]))
    return out


__all__ = [
    "extract_io_edge_intent",
    "extract_rail_intent",
    "extract_reference_order",
    "map_order_to_design",
    "rows_to_relative_order",
]
