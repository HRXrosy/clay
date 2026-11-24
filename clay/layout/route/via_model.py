"""Rule-driven, PDK-generic VIA METAL-ENCLOSURE model -- the dual of
:mod:`clay.layout.route.width_model`.

THE DUAL
========
``width_model`` derives a routed WIRE WIDTH from the PDK's ``min_width`` rule
(plus a design role/drive policy), generically, with NO per-PDK constant -- the
router gets a wire's physical width from the PDK contract, not a hardcoded number.

This module is the SAME pattern for VIAS: given the PDK's via rules (cut SIZE +
per-side metal ENCLOSURE -- the SAME ``via{k}_size`` / ``via{k}_enclosure_metal{k}``
values the synthesized DRC deck consumes), it returns, for one via cut at a centre,
the required METAL LANDING RECTANGLE on each adjacent routing metal: a square that
covers the cut and overhangs it by >= the rule enclosure on every side. Every value
comes from ``tech.via_rule(name)`` / ``tech.via_size(name)``; there is ZERO
``if pdk == ...`` branch. A PDK that declares a looser (or zero) enclosure simply
gets a smaller (or cut-sized) landing -- the model follows whatever each PDK's own
rules declare.

WHY (the missing contract)
==========================
The router emitted via CUTS but enforced no via metal-enclosure as a positive
obligation: it CLAMPED the rule to whatever incidental width the crossing wires
already had and checked coverage of that clamped (often ~0) enclosure, so a via on
a min-width wire drew a landing that did NOT overhang the cut by the rule. On
gf180 that is 230 real via-enclosure DRC violations (V1.3*/V1.4*/V2.3*/V2.4*).
external_pdk passed only because its enclosure happened to be met by the wider pads its
geometry already drew. The fix is to DERIVE the landing from the rule and draw it
at every via, exactly as ``width_model`` derives the wire width.

PROVENANCE (where the enclosure values come from)
=================================================
The per-side enclosure is ``tech.via_rule(name).enclosure_below`` (the lower
metal's overlap of the cut) and ``.enclosure_above`` (the upper metal's overlap);
the cut size is ``tech.via_size(name)``. For gf180mcu these are the foundry
deck's V1.3c/V1.4b (via1) and V2.3c/V2.4b (via2) 0.06um end-of-line overlaps and
the CO.4/CO.6 contact overlaps, carried in ``clay/pdk/gf180mcu.py`` ``_VIA_RULES``
(``provenance='parsed-deck'``). For external_pdk/external_pdk they come from those adapters'
``via_rule``. This module reads them through the ``tech`` contract and never holds
a literal of its own.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union

_Rect = Tuple[int, int, int, int]  # (x0, y0, x1, y1) integer nm
_LayerKey = Tuple[int, int]

# Own-metal coverage at a via cell, in one of two live forms (one predicate,
# two consumers -- the emit obligation today, the search pricing in D2):
#   * a sequence of rects: the net's ACTUAL kept metal on that layer (the emit
#     caller passes its live ``kept_rects`` list) -- coverage is exact
#     rect-subtraction;
#   * a number: the net's PLANNED wire width (nm) on that layer (the search
#     caller's conservative approximation before any metal exists) -- a wire
#     narrower than the cut cannot hold it at all (``cut-exceeds-neck``), and
#     covers the rule landing only when >= cut + 2*enclosure;
#   * None: no own metal claimed -> the full landing-pad obligation applies.
_OwnCover = Union[None, int, float, Sequence[_Rect]]


@dataclass(frozen=True)
class ViaLanding:
    """The rule-derived landing for ONE via cut between two adjacent metals.

    All coordinates are integer nm in the same absolute frame as the cut centre.

    Attributes
    ----------
    cut:
        The via cut square (cut size x cut size, centred on the requested point).
    below_metal_rect / above_metal_rect:
        The required metal landing rectangle on the LOWER metal / UPPER metal --
        the cut grown by the per-side enclosure (so it overhangs the cut by the
        rule on every side). The router must draw at least this much metal on the
        respective layer for the via to clear the foundry enclosure rules.
    enclosure_below_nm / enclosure_above_nm:
        The per-side enclosure (nm) the landings carry, verbatim from the PDK
        via rule (``0`` when the PDK declares none).
    cut_size_nm:
        The via cut size (nm) from the PDK.
    """

    cut: _Rect
    below_metal_rect: _Rect
    above_metal_rect: _Rect
    enclosure_below_nm: int
    enclosure_above_nm: int
    cut_size_nm: int


def _via_index(via_name: str) -> Optional[int]:
    """The numeric index of a via name (``via1`` -> 1, ``v2`` -> 2), else None."""
    for pfx in ("via", "vi", "v"):
        if via_name.startswith(pfx) and via_name[len(pfx):].isdigit():
            return int(via_name[len(pfx):])
    return None


def _parsed_enclosure_nm(tech: Any, via_name: str) -> Tuple[Optional[int], Optional[int]]:
    """Per-side enclosure ``(below, above)`` in nm READ FROM THE PARSED DECK RULES
    (the SAME ``drc_rules()`` ``via{k}_enclosure_metal{j}`` keys the synthesized DRC
    deck emits its enclosure rules from). ``None`` on a side the deck does not
    declare -- the deck emits NO enclosure rule there, so it is not an enforced
    obligation. ``(None, None)`` when the PDK exposes no parsed deck rules at all
    (e.g. gf180mcu, whose rules come through the static builder + real foundry deck,
    not synthesized parsed rules) -- the caller then falls back to ``via_rule``.

    GENERIC: the keys are ``via{k}_enclosure_metal{k}`` (below) /
    ``via{k}_enclosure_metal{k+1}`` (above), built from the via index; no per-PDK
    branch. This is the single rule source the milestone names ("the SAME drc_rules
    the deck consumes")."""
    k = _via_index(via_name)
    if k is None:
        return None, None
    try:
        parsed = tech.drc_rules() or {}
    except Exception:
        parsed = {}
    if not parsed:
        return None, None

    def _nm(key: str) -> Optional[int]:
        e = parsed.get(key)
        if not isinstance(e, dict):
            return None
        v = e.get("value_nm")
        if v is not None:
            return int(v)
        vu = e.get("value_um")
        return int(round(float(vu) * 1000.0)) if vu is not None else None

    below = _nm(f"via{k}_enclosure_metal{k}")
    above = _nm(f"via{k}_enclosure_metal{k + 1}")
    return below, above


def via_enclosure_nm(tech: Any, via_name: str) -> Tuple[int, int]:
    """The PDK per-side via enclosure ``(below, above)`` in nm for ``via_name``,
    derived from the SAME rule source the PDK's DRC deck consumes.

    Resolution (generic, no per-PDK branch):
      1. the PARSED deck rules (``via{k}_enclosure_metal{j}`` -- what the synthesized
         deck emits its enclosure check from): a side PRESENT here is the enforced
         value; a side ABSENT means the deck emits NO enclosure rule there, so the
         enforced obligation is 0 (the router must not fabricate one the oracle does
         not check -- that over-fattens and creates spacing violations);
      2. when the PDK has NO parsed deck rules at all (gf180mcu uses the static rule
         builder + the real foundry deck, not synthesized parsed rules), fall back
         to ``tech.via_rule(via_name).enclosure_below / enclosure_above`` -- the real
         foundry V1/V2 enclosure values that PDK's oracle deck enforces.

    This keeps the router's enforced enclosure == the PDK oracle's enforced
    enclosure on BOTH paths (external_pdk synthesized deck / gf180 foundry deck), so the
    landing model never over- or under-constrains relative to the signoff oracle."""
    has_parsed_deck = False
    try:
        has_parsed_deck = bool(tech.drc_rules())
    except Exception:
        has_parsed_deck = False
    if has_parsed_deck:
        # the PDK's oracle is the SYNTHESIZED deck, which emits an enclosure rule
        # for a side ONLY when its parsed ``via{k}_enclosure_metal{j}`` key exists.
        # A side absent here is NOT checked by the oracle -> enforced 0 (do not
        # fabricate a node-scaled enclosure the oracle never tests). The deck is
        # the single source; ``via_rule`` consolidated/node-scaled values are not
        # consulted on this path.
        pb, pa = _parsed_enclosure_nm(tech, via_name)
        return int(pb or 0), int(pa or 0)
    # no parsed deck (gf180mcu): the oracle is the real foundry deck; its enforced
    # enclosure lives in ``via_rule`` (V1/V2 values from the static rule builder).
    rb = ra = 0
    try:
        vr = tech.via_rule(via_name)
    except Exception:
        vr = None
    if vr is not None:
        eb = getattr(vr, "enclosure_below", None)
        ea = getattr(vr, "enclosure_above", None)
        if eb is not None:
            rb = int(getattr(eb, "value", eb) or 0)
        if ea is not None:
            ra = int(getattr(ea, "value", ea) or 0)
    return rb, ra


def via_cut_size_nm(tech: Any, via_name: str) -> Optional[int]:
    """The PDK via cut size (nm) for ``via_name`` (``None`` if the PDK is silent).

    D2 NOTE (this lookup is STRICTER than the engine resolver on a cut-silent
    PDK): it returns ``None`` when the PDK declares no cut for ``via_name``,
    which drives :func:`via_landing_feasible` to the ``no-cut-size`` verdict
    (ok=False -- a HARD reject). The engine's own ``engine_maze._via_cut_size_nm``
    instead FALLS BACK to the smaller adjacent metal-min so the cut square is
    never zero. The two therefore DISAGREE on a cut-silent PDK. Both the emit
    obligation and the D2 search gate must INJECT the engine-resolved
    ``cut_size_nm`` into :func:`via_landing_feasible` (never the ``via_name``
    path) so predicate and executor share the SAME number -- the byte-equivalence
    contract. This ``via_name`` path stays the strict last-resort lookup, not the
    production number source."""
    try:
        v = tech.via_size(via_name)
    except Exception:
        v = None
    return int(v) if v else None


def landing_rect_for_cut(
    cx: int,
    cy: int,
    cut_size_nm: int,
    enclosure_nm: int,
) -> _Rect:
    """The metal landing rectangle (integer nm) for a cut of ``cut_size_nm`` centred
    at ``(cx, cy)`` that overhangs the cut by ``enclosure_nm`` on every side.

    Pure geometry: ``half = ceil(cut/2) + enclosure`` so the landing covers the cut
    AND extends ``enclosure_nm`` past each cut edge. ``enclosure_nm == 0`` returns a
    landing flush with the cut (a PDK that declares no enclosure)."""
    vh = (cut_size_nm + 1) // 2  # round up so even cuts are fully covered
    half = vh + max(0, int(enclosure_nm))
    return (cx - half, cy - half, cx + half, cy + half)


def via_landing(
    tech: Any,
    via_name: str,
    cx: int,
    cy: int,
    *,
    cut_size_nm: Optional[int] = None,
) -> Optional[ViaLanding]:
    """The full rule-derived :class:`ViaLanding` for ``via_name`` centred at
    ``(cx, cy)`` (integer nm), or ``None`` when the PDK declares no cut size for it.

    ``cut_size_nm`` lets a caller pass the size the engine already resolved (so the
    cut square matches the engine's existing geometry byte-for-byte); otherwise the
    PDK value is read here. The enclosures always come from the PDK via rule -- this
    is the single rule-driven place a landing is sized."""
    cut_nm = cut_size_nm if cut_size_nm is not None else via_cut_size_nm(tech, via_name)
    if not cut_nm:
        return None
    rb, ra = via_enclosure_nm(tech, via_name)
    return ViaLanding(
        cut=landing_rect_for_cut(cx, cy, cut_nm, 0),
        below_metal_rect=landing_rect_for_cut(cx, cy, cut_nm, rb),
        above_metal_rect=landing_rect_for_cut(cx, cy, cut_nm, ra),
        enclosure_below_nm=rb,
        enclosure_above_nm=ra,
        cut_size_nm=int(cut_nm),
    )


def subtract_rects(rect: _Rect, holes: Sequence[_Rect]) -> List[_Rect]:
    """Subtract a set of axis-aligned ``holes`` from a single axis-aligned
    ``rect``; return the residual as a list of axis-aligned rectangles (possibly
    empty). Pure integer geometry -- the emitted wire pads/bridges/via squares are
    all axis-aligned rectangles, and the net-aware keep-out obstacles (cap plates,
    device pads) are too, so subtracting a foreign keep-out from emitted metal
    needs no general polygon engine.

    SINGLE TRUTH: this used to live in ``engine_maze`` (which still imports it
    under its old private name); it moved here so the shared via-landing
    feasibility predicate below judges coverage with the SAME geometry the emit
    path clips with -- one implementation, one worldview.

    The recursion peels ONE overlapping hole at a time: the part of ``rect`` outside
    the hole splits into up to four rectangular bands (left / right / below / above
    the hole's intersection with ``rect``), and the remaining holes are subtracted
    from each band in turn. Holes that do not overlap ``rect`` are ignored.
    Deterministic; terminates because each band is strictly smaller in area than
    ``rect`` once a real overlap is removed."""
    rx0, ry0, rx1, ry1 = rect
    if rx0 >= rx1 or ry0 >= ry1:
        return []  # degenerate (zero-area) -> nothing to keep
    for (hx0, hy0, hx1, hy1) in holes:
        # intersection of this hole with rect
        ix0 = max(rx0, hx0)
        iy0 = max(ry0, hy0)
        ix1 = min(rx1, hx1)
        iy1 = min(ry1, hy1)
        if ix0 >= ix1 or iy0 >= iy1:
            continue  # no real overlap with this hole -> try the next one
        out: List[_Rect] = []
        rest = holes
        if rx0 < ix0:   # left band
            out += subtract_rects((rx0, ry0, ix0, ry1), rest)
        if ix1 < rx1:   # right band
            out += subtract_rects((ix1, ry0, rx1, ry1), rest)
        if ry0 < iy0:   # bottom band (within the hole's x-span)
            out += subtract_rects((ix0, ry0, ix1, iy0), rest)
        if iy1 < ry1:   # top band (within the hole's x-span)
            out += subtract_rects((ix0, iy1, ix1, ry1), rest)
        return out
    return [rect]  # no hole overlapped -> rect survives whole


@dataclass(frozen=True)
class ViaFeasibility:
    """The verdict of :func:`via_landing_feasible` for ONE via cut.

    ``ok``/``reason`` are the honest judgment surface (no silent pass):

    ==========================  =====  ==============================================
    reason                      ok     meaning
    ==========================  =====  ==============================================
    ``""``                      True   feasible; any owed landings are clear
    ``"no-envelope-rule"``      True   PDK declares no enclosure on either side ->
                                       the LEGACY verbatim-cut path (no landing
                                       obligation exists; the emit path draws the
                                       bare cut, drop-if-buried stays its account)
    ``"spacing-residual"``      True   an owed landing reaches foreign ROUTED metal:
                                       a spacing breach the emit path TOLERATES
                                       (clip within the enclosure floor when the
                                       3-way clip holds, else the FULL rule pad is
                                       drawn and the residual reported honestly by
                                       the min-spacing checker). NEVER a
                                       suppression reason -- a suppressed via here
                                       is an unreported open (the LDO MM10 class).
    ``"device-pad-intrusion"``  False  an owed landing reaches a DEVICE/cap-plate
                                       keep-out: a short-class conflict, never
                                       tolerable -> the emit path relocates or
                                       suppresses WHOLE; the search must not
                                       commit this cell (F1).
    ``"cut-exceeds-neck"``      False  a width-form own cover is narrower than the
                                       cut itself -- the wire cannot hold the cut
                                       (the ``_plan_via_relocation`` w < cut test).
    ``"no-cut-size"``           False  the PDK declares no cut size for this hop.
    ==========================  =====  ==============================================

    ``landing_below`` / ``landing_above``: the FULL rule landing rect the emit
    obligation would draw on the lower / upper metal (``None`` when the side owes
    no pad: no enclosure declared, side already covered by own metal, or the
    legacy verbatim-cut path). On ``ok=False`` the slots carry whatever pads were
    judged before the reject (diagnostic; the emit path draws nothing then). The
    executor may CLIP an emitted pad against routed-metal keep-outs, but never
    below the enclosure floor these rects encode."""

    ok: bool
    reason: str
    landing_below: Optional[_Rect]
    landing_above: Optional[_Rect]


def _grown_intersects(pad: _Rect, rects: Sequence[_Rect], grow_nm: int) -> bool:
    """True when ``pad`` has a positive-area intersection with any of ``rects``
    grown by ``grow_nm`` on every side (``grow_nm=0`` tests the rects verbatim --
    exactly ``subtract_rects(pad, rects) != [pad]`` for a non-degenerate pad)."""
    px0, py0, px1, py1 = pad
    g = max(0, int(grow_nm))
    for (x0, y0, x1, y1) in rects:
        if x0 - g < px1 and px0 < x1 + g and y0 - g < py1 and py0 < y1 + g:
            return True
    return False


def via_landing_feasible(
    tech: Any,
    gi_low: int,
    cx_nm: int,
    cy_nm: int,
    *,
    own_cover_below: _OwnCover,
    own_cover_above: _OwnCover,
    foreign_device_rects_by_gi: Optional[Mapping[int, Sequence[_Rect]]],
    foreign_routed_rects_by_gi: Optional[Mapping[int, Sequence[_Rect]]],
    min_space_nm_by_gi: Optional[Mapping[int, int]] = None,
    min_width_nm_by_gi: Optional[Mapping[int, int]] = None,
    via_name: Optional[str] = None,
    cut_size_nm: Optional[int] = None,
    enclosure_below_nm: Optional[int] = None,
    enclosure_above_nm: Optional[int] = None,
) -> ViaFeasibility:
    """THE shared via-landing feasibility judgment for a cut between graph
    layers ``gi_low`` and ``gi_low+1`` centred at ``(cx_nm, cy_nm)`` -- one
    worldview for BOTH phases: the emit obligation consumes it today (Task D1,
    byte-stable), the search prices via moves with it next (Task D2, closing
    F1: search must stop committing via cells the emit obligation then
    suppresses).

    Rule provenance (zero per-PDK constants): the caller either INJECTS the
    numbers it already resolved (``cut_size_nm`` + ``enclosure_below_nm`` +
    ``enclosure_above_nm`` -- the emit path does this so predicate and executor
    share one set of numbers) or passes ``via_name`` and the numbers are read
    here through the SAME contract surface the DRC deck consumes
    (:func:`via_cut_size_nm` / :func:`via_enclosure_nm`). Refuses to guess:
    missing both is a ``ValueError``.

    Foreign-conductor convention: ``min_space_nm_by_gi=None`` means the foreign
    rects arrive PRE-inflated by min-space (the emit caller's
    ``_foreign_emit_rects`` / ``_foreign_metal_rects`` sets -- any positive-area
    intersection is already a breach). A mapping means the rects are RAW and
    the per-layer min-space is applied here (the search caller's convenience).
    Both classes inflate identically; what differs is the VERDICT: device-class
    contact is a short (reject), routed-metal contact is a spacing residual
    (tolerate + report).

    D2 NOTE (a missing key degrades to ZERO inflation): when a mapping is passed,
    a ``gi_side`` ABSENT from ``min_space_nm_by_gi`` reads ``0`` -- the pad is
    then tested against the RAW (un-inflated) foreign rect, so a genuine
    min-space breach on that side is UNDER-detected (a via the emit obligation
    would clip/relocate can read feasible in search). The D2 search caller must
    key the mapping for EVERY layer on which it hands RAW rects; a partial map is
    a silent under-block, not an error.

    ``min_width_nm_by_gi`` is carried for the D2 search consumer (clip-slack
    pricing shares this signature); the D1 emit judgment does not read it --
    the emit EXECUTOR applies the min-width floor in ``_clip_pad_to_min_space``.

    Judgment only: relocation ring-scan, clipping and suppression stay the
    emit executor's flow; this function decides, it does not draw."""
    # ---- rule resolution (injected numbers win; PDK contract otherwise) -----
    cut = cut_size_nm
    if cut is None:
        if via_name is None:
            raise ValueError(
                "via_landing_feasible: pass cut_size_nm (resolved) or via_name "
                "(PDK lookup) -- refusing to guess a cut size"
            )
        cut = via_cut_size_nm(tech, via_name)
    if not cut or int(cut) <= 0:
        return ViaFeasibility(False, "no-cut-size", None, None)
    vsz = int(cut)
    if enclosure_below_nm is None or enclosure_above_nm is None:
        if via_name is None:
            raise ValueError(
                "via_landing_feasible: pass enclosure_below_nm + "
                "enclosure_above_nm (resolved) or via_name (PDK lookup) -- "
                "refusing to guess enclosures"
            )
        pdk_rb, pdk_ra = via_enclosure_nm(tech, via_name)
        rb = int(pdk_rb if enclosure_below_nm is None else enclosure_below_nm)
        ra = int(pdk_ra if enclosure_above_nm is None else enclosure_above_nm)
    else:
        rb, ra = int(enclosure_below_nm), int(enclosure_above_nm)

    # ---- legacy verbatim-cut path (PINNED semantics) -------------------------
    # No enclosure declared on either side -> NO envelope obligation exists; the
    # emit path draws the bare cut (drop-if-buried stays the executor's account).
    if rb <= 0 and ra <= 0:
        return ViaFeasibility(True, "no-envelope-rule", None, None)

    sides = (
        (rb, gi_low, own_cover_below),
        (ra, gi_low + 1, own_cover_above),
    )

    # ---- neck test (width-form own cover only) -------------------------------
    # A planned wire narrower than the cut cannot hold it on that layer at all,
    # regardless of enclosure -- the same test _plan_via_relocation applies to a
    # relocation candidate (w < cut -> not a legal spot).
    for (_e, _gi_side, own) in sides:
        if isinstance(own, (int, float)) and not isinstance(own, bool):
            if int(own) < vsz:
                return ViaFeasibility(False, "cut-exceeds-neck", None, None)

    def _covers(own: _OwnCover, pad: _Rect, e: int) -> bool:
        if own is None:
            return False
        if isinstance(own, (int, float)) and not isinstance(own, bool):
            # width-form (search approximation): the wire covers the rule
            # landing only when its width holds cut + 2*enclosure across-axis
            # (along-axis the wire is assumed long -- conservative).
            return int(own) >= vsz + 2 * e
        rects = list(own)
        if not rects:
            return False
        return not subtract_rects(pad, rects)

    reason = ""
    landings: List[Optional[_Rect]] = [None, None]
    for slot, (e, gi_side, own) in enumerate(sides):
        if e <= 0:
            continue  # no enclosure owed on this side
        pad = landing_rect_for_cut(int(cx_nm), int(cy_nm), vsz, e)
        if _covers(own, pad, e):
            continue  # own metal already encloses the cut -> no pad obligation
        ms = 0
        if min_space_nm_by_gi:
            ms = int(min_space_nm_by_gi.get(gi_side, 0) or 0)
        # DEVICE foreign keep-out (cap plates / device pads): contact is a
        # SHORT-class conflict, never tolerable -> reject (the emit path
        # relocates/suppresses; the search must not commit this cell).
        dev = ()
        if foreign_device_rects_by_gi:
            dev = foreign_device_rects_by_gi.get(gi_side, ())
        if dev and _grown_intersects(pad, dev, ms):
            landings[slot] = pad
            return ViaFeasibility(
                False, "device-pad-intrusion", landings[0], landings[1]
            )
        # ROUTED-metal foreign keep-out: contact is a SPACING residual the emit
        # path TOLERATES (clip-or-full-pad, reported honestly) -- feasible.
        routed = ()
        if foreign_routed_rects_by_gi:
            routed = foreign_routed_rects_by_gi.get(gi_side, ())
        if routed and _grown_intersects(pad, routed, ms):
            reason = "spacing-residual"
        landings[slot] = pad
    return ViaFeasibility(True, reason, landings[0], landings[1])


__all__ = [
    "ViaLanding",
    "ViaFeasibility",
    "via_enclosure_nm",
    "via_cut_size_nm",
    "landing_rect_for_cut",
    "via_landing",
    "via_landing_feasible",
    "subtract_rects",
]
