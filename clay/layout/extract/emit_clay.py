"""The EMITTER: ``ExtractedIntent`` -> a PARSEABLE Clay source string.

This is the second half of the CONSISTENT INTENT EXTRACTOR round-trip. The
extractor reads a reference design and produces an :class:`ExtractedIntent`
(devices / passives / matched-pairs / bands / orders / ... -- the dual of what
the placer consumes). :func:`emit_clay` turns that back into a Clay source that
the GENERIC place/route can realize, so the round-trip fidelity vs the reference
IS the extraction-consistency + vocabulary-completeness metric.

THE EMIT CONTRACT (the load-bearing discipline)
===============================================
We emit ONLY relations for which Clay has an IMPLEMENTED primitive in ACTIVE
syntax (so the emitted source PARSES and PLACES):

  * ``topology { device .. res .. cap .. }`` -- the device/passive set (M -> nf);
  * ``annotate { @matching(..); @cell: @equal_size(..); @interdigitated(..) }``;
  * ``layout_intent { @place_group(grp, devices=(..), row=<band>) ;
    @order(a, b, rel) ; @abut(a, b) ; @budget(..) }``;
  * ``policy { @place_policy(engine=seqpair) }`` + ``flow {}`` + ``quality {}``.

Every extracted relation that has NO implemented primitive yet
(``aligns`` / ``symmetry_axis`` / ``net_layers`` / ``spacings``) is emitted as a
``// EXTRACTED (not-yet-implemented): ...`` COMMENT
-- recorded + visible as the roadmap, but NOT as active syntax that would fail to
parse. This is what makes the round-trip honest: the gap between "extracted" and
"emitted-active" localizes the missing-primitive shortfall.

The block grammar matches ``cases/opamp/two_stage_ref_gf180/source/design.clay``
verbatim (that case is the live, parsing+placing reference for the grammar).
"""
from __future__ import annotations

from typing import Any, List, Optional, Sequence, Set, Tuple

# --------------------------------------------------------------------------- #
# Duck-typed view of the SHARED CONTRACT (clay/layout/extract/types.py).
#
# A sibling owns ``types.py`` (``ExtractedIntent`` + ``ExtractedDevice`` +
# ``ExtractedPassive``). We code against the field NAMES of that contract via
# ``getattr`` so this module works whether or not ``types.py`` is importable yet
# (when absent, any object exposing the same attributes -- e.g. the hand-built
# intent in the test -- is accepted). The fields we read:
#
#   intent.cell                      : str
#   intent.devices                   : [ExtractedDevice(name, kind, W, L, M, nets)]
#   intent.passives                  : [ExtractedPassive(name, kind, value, nets)]
#   intent.matched_pairs             : [(a, b)]
#   intent.diff_pairs                : [(a, b)]            (optional; informational)
#   intent.mirror_pairs              : [(a, b)]            (optional; informational)
#   intent.equal_size_groups         : [set(names)]
#   intent.bands                     : {name -> int}       (row band per device)
#   intent.orders                    : [(a, rel, b)]       (ACTIVE: @order(a,b,rel))
#   intent.abuts                     : [(a, b)]            (ACTIVE: @abut(a,b))
#   intent.aligns                    : [(axis, [names])]   (NOT-YET: comment)
#   intent.symmetry_axis             : optional            (NOT-YET: comment)
#   intent.spacings                  : [(a, b, min)]       (NOT-YET: comment)
#   intent.net_layers                : {net -> layer}      (NOT-YET: comment)
#   intent.device_bboxes             : {name -> bbox}      (informational)
#   intent.interdigitated_pairs      : [(a, b)]            (optional; @interdigitated)
#
# NEW (named + emitted, but as structured // @EXTRACTED-LAYOUT comments -- the
# grammar does NOT know these forms yet, so emitting them as active syntax would
# fail to parse; the structured comment channel makes them greppable + machine-
# readable now, ahead of the grammar/placer follow-up):
#   intent.edge_aligns               : [(a, b, edge)]      edge in {left,right,top,bottom}
#                                                          (dcsolve CORE edge-align)
#   intent.folds                     : {dev -> (n_rows, fingers_per_row)}
#                                                          (fold_intent.FoldChoice)
#   intent.rails                     : [(net, [devices])]  (power_intent rail group)
#   intent.free_devices              : [dev]               (don't-care classifier)
# --------------------------------------------------------------------------- #


def _get(obj: Any, name: str, default: Any) -> Any:
    """Field access against the duck-typed contract (missing field -> default)."""
    val = getattr(obj, name, None)
    return default if val is None else val


def _fmt_u(value: Any) -> str:
    """Format a micron length as a Clay ``<n>u`` literal (drop a trailing .0)."""
    f = float(value)
    if f == int(f):
        return f"{int(f)}u"
    # trim trailing zeros but keep at least one decimal
    s = ("%g" % f)
    return f"{s}u"


def _fmt_value(value: Any) -> str:
    """Format a passive value. If already a string carrying a unit suffix
    (``1.5233k`` / ``1.1p``) pass it through; else render the float."""
    if isinstance(value, str):
        return value
    return "%g" % float(value)


def _device_kind(kind: str) -> str:
    """Map an extracted device kind to a Clay primitive kind (nmos / pmos)."""
    k = (kind or "").strip().lower()
    if k in ("nmos", "nfet", "n"):
        return "nmos"
    if k in ("pmos", "pfet", "p"):
        return "pmos"
    # default: pass through (the parser will reject an unknown kind, surfacing it)
    return k or "nmos"


import re as _re


def _san(net: Any) -> str:
    """Sanitize a raw netlist net name into a valid Clay identifier.

    Reference SPICE/Spectre netlists carry global-net markers (``vdd!``,
    ``gnd!``) and node names that Clay's grammar rejects. We strip the global
    ``!`` suffix, map ground aliases to ``vss``, lowercase the conventional
    IO/supply nets, replace any other illegal char with ``_``, and never let an
    identifier start with a digit. Deterministic so the round-trip is stable.
    """
    s = str(net).strip()
    s = s.rstrip("!")  # Spectre global-net marker
    low = s.lower()
    if low in ("gnd", "0", "gnd!", "vss"):
        return "vss"
    if low in ("vdd", "vcc", "vdd!"):
        return "vdd"
    if low in ("vip", "vin", "vout", "vinp", "vinn"):
        return low
    s = _re.sub(r"[^A-Za-z0-9_]", "_", s)
    if s and s[0].isdigit():
        s = "n_" + s
    return s or "net_unnamed"


def _pair_set(pairs: Sequence[Tuple[str, str]]) -> Set[Tuple[str, str]]:
    """Canonicalize an undirected pair list to a set of sorted tuples."""
    out: Set[Tuple[str, str]] = set()
    for p in pairs or []:
        if not p or len(p) < 2:
            continue
        a, b = str(p[0]), str(p[1])
        if a == b:
            continue
        out.add(tuple(sorted((a, b))))  # type: ignore[arg-type]
    return out


# --------------------------------------------------------------------------- #
# THE EMITTER.
# --------------------------------------------------------------------------- #
def emit_clay(
    intent: Any,
    *,
    pinned: bool = False,
    pinned_subset: Optional[Set[str]] = None,
) -> str:
    """Render an :class:`ExtractedIntent` to a PARSEABLE Clay source string.

    Only IMPLEMENTED primitives become active syntax; every other extracted
    relation is recorded as a ``// EXTRACTED (not-yet-implemented): ...`` comment.
    The returned text is validated to PARSE via :func:`clay.parser.parse_clay_file`
    (see :func:`validate_parses`).

    Three modes (mutually exclusive on the pin axis):

    * ``pinned=True`` -- the ANCHOR (bottom-clause) form: EVERY device/passive
      with a known ``device_bboxes`` entry gets an absolute ``@fix(name, x=, y=)``
      at its centroid, and the relational primitives (@place_group/@order/...) are
      SUPPRESSED. Fully-pinned -> zero placer freedom -> reproduces the reference
      identically (incompleteness_score == 0) -- the strict-consistency ground
      term the convergence method anneals down from.

    * ``pinned_subset=<set of names>`` -- the RELAXED (annealing) form: only the
      devices IN the subset get an ``@fix`` pin; ALL devices additionally keep the
      ABSTRACT relational primitives (@matching / @equal_size / @place_group bands,
      with @order/@abut as comments). The non-pinned devices are governed purely by
      the abstract relations, so the placer must re-derive their positions. This is
      the per-step emit the relaxation/equivalence-oracle loop consumes. An empty
      subset == "no pins, all relations" (== the default ``pinned=False`` mode); a
      subset covering every device with a bbox == the all-pinned anchor PLUS the
      relations (a superset of ``pinned=True``).

    * default (``pinned=False``, ``pinned_subset=None``) -- NO pins; the abstract
      relational primitives for ALL devices (the most-relaxed, fully-abstract form).
    """
    if pinned and pinned_subset is not None:
        raise ValueError("emit_clay: pass at most one of pinned= / pinned_subset=")
    cell = str(_get(intent, "cell", "ExtractedCell"))
    devices = list(_get(intent, "devices", []))
    passives = list(_get(intent, "passives", []))

    # ---- collect every net mentioned, derive the port list ---------------- #
    # Ports = the nets the design exposes. We have no explicit port list in the
    # contract, so we treat the conventional supply/IO nets as ports when present
    # (vip/vin/vout/vdd/vss), falling back to ALL nets if none of those appear.
    all_nets: List[str] = []
    seen_nets: Set[str] = set()
    for d in devices:
        for n0 in _get(d, "nets", {}).values() if isinstance(_get(d, "nets", {}), dict) else []:
            n = _san(n0)
            if n not in seen_nets:
                seen_nets.add(n)
                all_nets.append(n)
    for p in passives:
        pn = _get(p, "nets", {})
        vals = pn.values() if isinstance(pn, dict) else pn
        for n0 in vals or []:
            n = _san(n0)
            if n not in seen_nets:
                seen_nets.add(n)
                all_nets.append(n)

    _CONVENTIONAL_PORTS = ["vip", "vin", "vout", "vdd", "vss"]
    ports = [p for p in _CONVENTIONAL_PORTS if p in seen_nets]
    if not ports:
        ports = list(all_nets)

    lines: List[str] = []
    lines.append(
        "// AUTO-EMITTED from an ExtractedIntent (consistent intent extractor)."
    )
    lines.append(
        "// Round-trip target: generic place/route this source + score fidelity"
    )
    lines.append("// vs the reference GDS. Only IMPLEMENTED primitives are active;")
    lines.append(
        "// not-yet-implemented extracted relations are recorded as comments below."
    )
    lines.append(f"package clay.layout.extract.emitted_{cell.lower()};")
    pdk = str(_get(intent, "pdk", "gf180mcu"))
    lines.append(f'pdk "{pdk}";')
    lines.append("")

    # ===================================================================== #
    # cell { topology { ... } spec { ... } }
    # ===================================================================== #
    lines.append(f"cell {cell}({', '.join(ports)}) {{")
    lines.append("  topology {")

    for d in devices:
        name = str(_get(d, "name", "M_?"))
        kind = _device_kind(str(_get(d, "kind", "nmos")))
        nets = _get(d, "nets", {})
        # nets is a dict terminal->net (drain/gate/source/body)
        drain = _san(nets.get("drain", nets.get("d", "")))
        gate = _san(nets.get("gate", nets.get("g", "")))
        source = _san(nets.get("source", nets.get("s", "")))
        body = _san(nets.get("body", nets.get("b", nets.get("source", nets.get("s", "")))))
        w = _get(d, "W", 1.0)
        l = _get(d, "L", 1.0)
        m = int(_get(d, "M", 1) or 1)
        lines.append(f"    device {name} : {kind} {{")
        lines.append(
            f"      drain = {drain}; gate = {gate}; source = {source}; body = {body};"
        )
        lines.append(f"      W = {_fmt_u(w)}; L = {_fmt_u(l)};")
        lines.append(f"      nf = {m};")  # M -> nf (device multiplicity)
        lines.append("    }")

    for p in passives:
        name = str(_get(p, "name", "X_?"))
        kind = (str(_get(p, "kind", "res")) or "res").strip().lower()
        pn = _get(p, "nets", {})
        if isinstance(pn, dict):
            t_p = _san(pn.get("p", pn.get("plus", "")))
            t_n = _san(pn.get("n", pn.get("minus", "")))
        else:
            seq = list(pn or [])
            t_p = _san(seq[0]) if len(seq) > 0 else ""
            t_n = _san(seq[1]) if len(seq) > 1 else ""
        val = _fmt_value(_get(p, "value", 1.0))
        if kind in ("cap", "capacitor", "c"):
            lines.append(f"    cap {name} {{ p = {t_p}; n = {t_n}; value = {val}; }}")
        else:  # resistor (default)
            lines.append(f"    res {name} {{ p = {t_p}; n = {t_n}; value = {val}; }}")

    lines.append("  }")
    lines.append("  spec {")
    lines.append("    gain_dc : real >= 40;")
    lines.append("  }")
    lines.append("}")
    lines.append("")

    # ===================================================================== #
    # annotate { @matching ; @interdigitated ; @cell: @equal_size }
    # ===================================================================== #
    matched = _pair_set(_get(intent, "matched_pairs", []))
    interdig = _pair_set(_get(intent, "interdigitated_pairs", []))
    eq_groups_raw = _get(intent, "equal_size_groups", [])
    eq_groups = [
        tuple(sorted(str(m) for m in g)) for g in eq_groups_raw if len(set(g)) >= 2
    ]

    # Which devices get an absolute @fix pin:
    #   pinned=True            -> every device with a bbox (the anchor);
    #   pinned_subset=<names>  -> only the named subset (the relaxed/anneal form);
    #   default                -> none.
    if pinned:
        dbb_pin = _get(intent, "device_bboxes", {}) or {}
        pin_names = sorted(dbb_pin)
    elif pinned_subset is not None:
        dbb_pin = _get(intent, "device_bboxes", {}) or {}
        pin_names = sorted(n for n in pinned_subset if n in dbb_pin)
    else:
        dbb_pin = {}
        pin_names = []
    pin_lines = []
    for nm in pin_names:
        bb = dbb_pin[nm]
        if not bb or len(bb) < 4:
            continue
        cx = round((float(bb[0]) + float(bb[2])) / 2.0, 4)
        cy = round((float(bb[1]) + float(bb[3])) / 2.0, 4)
        pin_lines.append(f"  @cell: @fix({nm}, x={cx}, y={cy});")

    if matched or interdig or eq_groups or pin_lines:
        lines.append(f"annotate {cell} {{")
        # ANCHOR (bottom-clause): pin every device at its extracted centroid.
        for pl in pin_lines:
            lines.append(pl)
        # @matching: emit BOTH directions (the case file declares both ways).
        for a, b in sorted(matched):
            lines.append(f"  {a}: @matching({b});")
            lines.append(f"  {b}: @matching({a});")
        # @interdigitated: only when a matched pair is flagged interdigitated.
        for a, b in sorted(interdig):
            lines.append(f"  {a}: @interdigitated({b}, abba);")
            lines.append(f"  {b}: @interdigitated({a}, abba);")
        # @cell: @equal_size(...) per equal_size group (>= 2 members).
        for g in eq_groups:
            lines.append(f"  @cell: @equal_size({', '.join(g)});")
        lines.append("}")
        lines.append("")

    # ANCHOR (bottom-clause) mode: the @fix pins live in the annotate{} block above;
    # the intent is PURELY the pins (zero placer freedom), so NO layout_intent /
    # relational block is emitted. Return the minimal pinned cell.
    if pinned:
        return "\n".join(lines) + "\n"

    # ===================================================================== #
    # layout_intent { @budget ; @place_group(grp, devices=(..), row=<band>) }
    # ===================================================================== #
    bands = _get(intent, "bands", {})  # name -> int row band
    lines.append(f"layout_intent {cell} {{")

    # @budget on the matched-input gates' parasitic load (the signal pair), if a
    # diff/matched pair is on the vip/vin nets. Generic: pick the matched pair whose
    # gates are vip/vin; only emitted when both gate nets exist.
    budget_emitted = False
    for d in devices:
        pass  # (budget derivation below uses gate nets directly)
    gate_of = {}
    for d in devices:
        nets = _get(d, "nets", {})
        gate_of[str(_get(d, "name", ""))] = nets.get("gate", nets.get("g", ""))
    for a, b in sorted(matched):
        ga, gb = gate_of.get(a), gate_of.get(b)
        if ga and gb and {ga, gb} == {"vip", "vin"}:
            lines.append(f"  @budget(matched_cc_balance, {ga}, {gb}, 0.10);")
            budget_emitted = True
            break
    if budget_emitted:
        lines.append("")

    # @place_group per (band, role-cluster). We group device/passive NAMES by their
    # declared band; within a band we keep them as ONE place_group per band (the
    # case keeps function sub-groups, but the contract gives us only the band map,
    # so we emit one group per band -- a generic, deterministic clustering).
    if bands:
        by_band: dict = {}
        for nm, row in bands.items():
            by_band.setdefault(int(row), []).append(str(nm))
        # passives in the MIDDLE-style band get an explicit col so the passive-
        # column placer can place them as blocks; we assign cols left-to-right.
        passive_names = {str(_get(p, "name", "")) for p in passives}
        for row in sorted(by_band):
            members = by_band[row]
            members_sorted = sorted(members)
            mos_members = [m for m in members_sorted if m not in passive_names]
            pas_members = [m for m in members_sorted if m in passive_names]
            grp = f"band{row}"
            if mos_members:
                devlist = ", ".join(mos_members)
                lines.append(
                    f"  @place_group({grp}_dev, devices=({devlist}), row={row});"
                )
            # each passive gets its own col index (block placement in its band)
            for col, pm in enumerate(pas_members):
                lines.append(
                    f"  @place_group({grp}_p{col}, devices=({pm}), "
                    f"row={row}, col={col});"
                )

    # ----------------------------------------------------------------------- #
    # @order / @abut -- NOW ACTIVE (implemented primitives).
    #
    # Grammar surface (clay/grammar.lark, layout_intent_entry):
    #     @order(A, B, rel);    rel in {left_of, above}  -- A is rel B
    #     @abut(A, B);
    # These lower to LayoutIntent.order_constraints (a, b, rel) / abut_pairs (a, b)
    # and are consumed by the seqpair placer + verified by check_order / check_abut.
    #
    # NOTE the ARG ORDER difference: the EXTRACTED contract stores orders as
    # ``(a, rel, b)`` triples, but the active grammar puts the relation LAST
    # (``@order(a, b, rel)``), so we reorder on emit.
    #
    # CYCLE/VALIDITY GUARD: an extracted order set can carry redundant or
    # contradictory edges (a self-edge a==a; both ``(a,b,rel)`` and the back-edge
    # ``(b,a,rel)`` for the same relation -- a 2-cycle that would make the placer's
    # longest-path constraint graph cyclic). We KEEP only the FIRST directed edge
    # seen for each unordered pair+relation and DROP self-edges, so the emitted
    # source stays valid + places by construction. (rel is grammar-checked to
    # {left_of, above}; anything else is dropped defensively.)
    seen_order: Set[Tuple[str, str, str]] = set()
    seen_order_pairrel: Set[Tuple[frozenset, str]] = set()
    for entry in _get(intent, "orders", []) or []:
        try:
            a, rel, b = entry
        except Exception:
            continue
        a, b, rel = str(a), str(b), str(rel)
        if a == b or rel not in ("left_of", "above"):
            continue  # self-edge or unknown relation -> drop
        key = (a, b, rel)
        if key in seen_order:
            continue
        pairrel = (frozenset((a, b)), rel)
        if pairrel in seen_order_pairrel:
            continue  # back-edge for the same pair+relation -> drop (2-cycle)
        seen_order.add(key)
        seen_order_pairrel.add(pairrel)
        lines.append(f"  @order({a}, {b}, {rel});")

    seen_abut: Set[Tuple[str, str]] = set()
    for entry in _get(intent, "abuts", []) or []:
        if not entry or len(entry) < 2:
            continue
        a, b = str(entry[0]), str(entry[1])
        if a == b:
            continue
        key = tuple(sorted((a, b)))  # undirected -> canonical
        if key in seen_abut:
            continue
        seen_abut.add(key)  # type: ignore[arg-type]
        lines.append(f"  @abut({a}, {b});")

    lines.append("}")
    lines.append("")

    # ===================================================================== #
    # EXTRACTED (not-yet-implemented) relations -> COMMENTS (the roadmap).
    # These have NO active Clay primitive, so emitting them as syntax would fail
    # to parse. They are recorded here, visible, as the vocabulary gap.
    # ===================================================================== #
    # NOTE: @order / @abut are NO LONGER here -- they are emitted ACTIVE inside
    # the layout_intent {} block above (their primitives are implemented).
    roadmap: List[str] = []
    for entry in _get(intent, "aligns", []) or []:
        try:
            axis, names = entry
            roadmap.append(f"@align({axis}, ({', '.join(str(n) for n in names)}))")
        except Exception:
            roadmap.append(f"@align({entry})")
    sym = _get(intent, "symmetry_axis", None)
    if sym is not None:
        roadmap.append(f"@symmetry_axis({sym})")
    for entry in _get(intent, "spacings", []) or []:
        try:
            a, b, mn = entry
            roadmap.append(f"@spacing({a}, {b}, min={mn})")
        except Exception:
            roadmap.append(f"@spacing({entry})")
    nl = _get(intent, "net_layers", {})
    if isinstance(nl, dict):
        for net, layer in nl.items():
            roadmap.append(f"@net_layer({net}, {layer})")

    if roadmap:
        lines.append(
            "// EXTRACTED (not-yet-implemented) relations -- recorded as the"
        )
        lines.append(
            "// vocabulary/primitive roadmap; NO active Clay primitive yet, so they"
        )
        lines.append("// are comments (emitting as syntax would fail to parse):")
        for r in roadmap:
            lines.append(f"//   @EXTRACTED: {r}")
        lines.append("")

    # ===================================================================== #
    # NEW LAYOUT PRIMITIVES -> structured // @EXTRACTED-LAYOUT comments.
    #
    # Four analog-layout primitives that already live in the Python extract
    # pipeline (dcsolve edge-align / fold_intent / power_intent / the don't-care
    # classifier) but have NO Clay grammar surface yet. We NAME + EMIT each in a
    # CONSISTENT, MACHINE-READABLE form inside the existing comment channel using
    # the fixed prefix ``// @EXTRACTED-LAYOUT:`` so tooling can grep them, and the
    # emitted text still PARSES (these are comments, not active syntax). The
    # eventual active grammar + placer consumption is specified in
    # docs/superpowers/specs/2026-06-07-clay-layout-primitive-additions.html.
    #
    #   1. EDGE-ALIGN  @align(a, b, edge=left|right|top|bottom)   (dcsolve core)
    #   2. FOLD        @fold(device, rows=N, fingers=M)           (FoldChoice)
    #   3. POWER-RAIL  @rail(net, devices=(...))                  (power_intent)
    #   4. DONT-CARE   @free(device)                              (perturbation)
    # ===================================================================== #
    _LAYOUT_PREFIX = "// @EXTRACTED-LAYOUT:"
    layout_primitives: List[str] = []

    # 1. EDGE-ALIGN: (a, b, edge) with edge in {left,right,top,bottom}.
    _EDGES = ("left", "right", "top", "bottom")
    seen_edge: Set[Tuple[str, str, str]] = set()
    for entry in _get(intent, "edge_aligns", []) or []:
        try:
            a, b, edge = entry
        except Exception:
            continue
        a, b, edge = str(a), str(b), str(edge).strip().lower()
        if a == b or edge not in _EDGES:
            continue
        key = (a, b, edge)
        if key in seen_edge:
            continue
        seen_edge.add(key)
        layout_primitives.append(f"@align({a}, {b}, edge={edge})")

    # 2. FOLD: device -> (n_rows, fingers_per_row).
    folds = _get(intent, "folds", {})
    if isinstance(folds, dict):
        for dev in sorted(folds, key=str):
            spec = folds[dev]
            try:
                rows, fingers = int(spec[0]), int(spec[1])
            except Exception:
                continue
            layout_primitives.append(f"@fold({dev}, rows={rows}, fingers={fingers})")

    # 3. POWER-RAIL: (net, [devices]) -- a validated supply rail group.
    seen_rail: Set[Tuple[str, Tuple[str, ...]]] = set()
    for entry in _get(intent, "rails", []) or []:
        try:
            net, devs = entry
        except Exception:
            continue
        net = _san(net)
        dev_list = tuple(str(d) for d in (devs or []))
        if len(dev_list) < 2:
            continue
        key = (net, dev_list)
        if key in seen_rail:
            continue
        seen_rail.add(key)
        layout_primitives.append(
            f"@rail({net}, devices=({', '.join(dev_list)}))"
        )

    # 4. DONT-CARE: a device whose absolute position is a declared design freedom.
    seen_free: Set[str] = set()
    for dev in _get(intent, "free_devices", []) or []:
        dev = str(dev)
        if not dev or dev in seen_free:
            continue
        seen_free.add(dev)
        layout_primitives.append(f"@free({dev})")

    if layout_primitives:
        lines.append(
            "// EXTRACTED (not-yet-implemented) LAYOUT primitives -- named + emitted"
        )
        lines.append(
            "// in a structured, machine-readable comment channel (the grammar does"
        )
        lines.append(
            "// not know these forms yet, so they stay comments to keep the source"
        )
        lines.append(
            "// parsing). Grep the fixed prefix to recover them; the active grammar"
        )
        lines.append(
            "// + placer consumption is in the 2026-06-07 primitive-additions spec:"
        )
        for lp in layout_primitives:
            lines.append(f"{_LAYOUT_PREFIX} {lp}")
        lines.append("")

    # ===================================================================== #
    # policy {} + flow {} + quality {}  (match the case grammar exactly)
    # ===================================================================== #
    policy_name = f"{cell}Seqpair"
    lines.append(f"policy {policy_name} {{")
    lines.append("  @place_policy(engine=seqpair);")
    lines.append("}")
    lines.append("")

    lines.append(f"flow {cell} {{")
    lines.append(f'  pdk "{pdk}";')
    lines.append("  corners (tt);")
    lines.append("  steps (compile, layout, drc, lvs, pex, post, intent, sign);")
    lines.append(f"  policy = {policy_name};")
    lines.append("}")
    lines.append("")

    lines.append(f"quality {cell} {{")
    lines.append('  @kpi(dc_gain_db, ">=", 40, unit="dB");')
    lines.append("  @require(intent_signed);")
    lines.append("}")
    lines.append("")

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Parse validation: write to a temp file + parse, return (ok, error|None).
# --------------------------------------------------------------------------- #
def validate_parses(clay_src: str, *, work_dir: Optional[str] = None) -> Tuple[bool, Optional[str]]:
    """Write ``clay_src`` to a temp ``.clay`` file and confirm it PARSES via
    :func:`clay.parser.parse_clay_file`. Returns ``(True, None)`` on success or
    ``(False, "<error>")`` on a parse failure. (Validation only -- no placement.)
    """
    import tempfile
    from pathlib import Path

    from clay.parser import parse_clay_file

    base = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="clay_emit_"))
    base.mkdir(parents=True, exist_ok=True)
    f = base / "design.clay"
    f.write_text(clay_src, encoding="utf-8")
    try:
        cf = parse_clay_file(f)
        # sanity: the emitted top cell must be present
        names = [getattr(c, "local_name", None) for c in getattr(cf, "cell_defs", [])]
        if not names:
            return False, "parsed but no cell_defs produced"
        return True, None
    except Exception as exc:  # parse / resolution error
        return False, f"{type(exc).__name__}: {exc}"
