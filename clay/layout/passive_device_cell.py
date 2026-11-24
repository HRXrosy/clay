"""Passive device-geometry builder (Clay intent-driven P&R) -- RESISTOR.

``build_resistor(tech, value_ohms, ...)`` builds ONE DRC-legal, LVS-extractable
gf180mcu poly resistor cell for any PDK whose facts a
:class:`~clay.layout.tech.LayoutTech` aggregates. It is the passive sibling of
:func:`clay.layout.device_cell.build_device` (the MOS builder): same
resolved-PDK-facts style, same :class:`DeviceGeom`-shaped output (polygons +
a terminal map the placer / router consume), but a NON-MOS geometry -- a poly
strip, not a folded finger array.

Why a poly resistor (gf180 ``ppolyf_u``)
========================================
The gf180mcu LVS deck (``rules/klayout/lvs/gf180mcu.lvs``) extracts an
unsalicided P+ poly resistor as::

    ppolyf_u_layer = pplus.and(poly2).and(sab).and(res_mk)
                          .not_interacting(resistor).not(dnwell)        # the BODY
    poly2_con      = poly2.not(res_mk).not(plfuse)                       # head/tail
    extract_devices(resistor_with_bulk("ppolyf_u", 350, BResistor),
                    {"R" => ppolyf_u_layer, "C" => poly2_con, "W" => sub})

So a legal, extractable resistor is a single poly2 strip on which:

  * the CENTRAL section (the resistive body) is covered by the P+ implant
    (``pplus``), the salicide block (``sab``) and the resistor marker
    (``res_mk``) -- this is what the deck ANDs into ``ppolyf_u_layer``;
  * the two ENDS (head / tail) stick out past ``res_mk`` (and past ``sab`` by
    SB.10) so they are bare ``poly2_con``, land a contact each, and rise to M1
    -- these are the ``p`` / ``n`` terminals;
  * NO high-Rs ``resistor`` (62/0) layer is drawn, so ``not_interacting(resistor)``
    holds and the body extracts as the base ``ppolyf_u`` (sheet 350 Ohm/sq), the
    same model + sheet-rho the flow's LVS reference card emits
    (``clay/flow/step.py::_lower_device_card`` -> ``ppolyf_u``, rho 350).

Every dimension is sourced from the resolved PDK tech (rule container) -- the
DRC-binding resistor rules + the contact/implant rules below -- so the cell is
DRC-legal BY CONSTRUCTION, never patched post-hoc.

DRC rule contract (gf180mcu, every value traced)
=================================================
  * LRES.1  body min width            0.8um   -> resolved ``min_width('res_poly')``
                                                  with an 800nm floor.
  * LRES.5  Nplus/Pplus implant overlap of the body  0.3um  -> implant grown
            300nm past the body on every side.
  * LRES.6  SAB overlap of the body in the width direction  0.28um  -> sab grown
            280nm past the body sides.
  * LRES.7 / SB.4  SAB-to-contact space  0.22 / 0.15um  -> the head/tail contact
            sits >= 220nm clear of the sab/res_mk edge (contact on bare poly).
  * SB.1   sab min width 0.42um ; SB.13 sab min area 2um^2  -> the sab box is
            grown to satisfy both.
  * SB.10  poly2 extension beyond related sab  0.22um  -> the poly head extends
            >= 220nm past the sab edge before the contact.
  * CO.1   contact size  ; CO.3 poly overlap of contact 0.07um ; CO.6 M1
            enclosure of contact 0.06um -> head/tail contact + M1 pad.

The value->geometry mapping uses the SAME closed form the LVS reference uses
(``L = (R / rho) * W``) so the extracted body geometry matches the reference
card's ``W`` / ``L`` and the BResistor comparison passes. ``W`` is the reference
width (1um, also the LRES.1 floor); ``rho`` is the PDK poly sheet-rho.

Scope: RESISTOR only. A MIM / MOM capacitor builder would slot in beside
``build_resistor`` (its own marker/plate layers + a 2-terminal map), driven by
the same tech facts; that is the NEXT increment, explicitly out of scope here.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

from .backend import LayerPolygon, Point, Polygon
from .capacitor_unit_occurrences import CapacitorUnitOccurrenceGeometry, with_capacitor_occurrences
from .device_cell import (  # reuse the MOS output contract + the orient transform
    DeviceGeom,
    Terminal,
    _bbox,
    _device_grid_nm,
    _ORIENTS,
    _poly_name,
    _rotate_box,
    _rotate_point,
    _rotate_poly,
)

LayerKey = Tuple[int, int]
Bbox = Tuple[int, int, int, int]


# --- gf180mcu poly-resistor DRC dimensions (nm), each traced to a rule ----------
# These are the gf180 calibrated values used when the rule container is silent on
# the resistor-specific LRES/SB rules (they are not single-token width/space keys
# the generic container exposes). For a non-gf180 PDK they node-scale from the
# contact size, keeping the resistor legal-by-construction.
_GF180_RES_BODY_MIN_WIDTH_NM = 800   # LRES.1
_GF180_RES_IMPLANT_OVERLAP_NM = 300  # LRES.5
_GF180_RES_SAB_OVERLAP_NM = 280      # LRES.6 (width direction)
_GF180_RES_SAB_TO_CONTACT_NM = 220   # LRES.7 (SAB-to-contact on the resistor poly)
_GF180_RES_POLY_PAST_SAB_NM = 220    # SB.10 (poly2 extension beyond related sab)
_GF180_SAB_MIN_WIDTH_NM = 420        # SB.1
_GF180_SAB_MIN_AREA_NM2 = 2_000_000  # SB.13 (2 um^2)
_GF180_POLY_CONTACT_ENC_NM = 70      # CO.3 (poly overlap of contact)
_GF180_M1_CONTACT_ENC_NM = 60        # CO.6 (metal1 overlap of contact)
_GF180_UNSAL_POLY_OVERLAP_NM = 180   # PP.9 / NP.9 (implant overlap of unsalicided poly2)
_GF180_GRID_NM = 5                    # gf180mcu manufacturing grid (0.005um == 5 DBU)
_REF_RES_WIDTH_UM = 1.0              # matches step._REF_RES_WIDTH_UM (LVS reference)

# --- serpentine FOLDING of a long poly resistor into a compact BANK -------------
# A high-value resistor's natural single strip is hundreds of um long (L = R/rho*W);
# drawn flat it sprawls (the LDO's 109k resistor is a 312um strip). A real layout
# FOLDS that strip into a serpentine BANK: the SAME continuous poly path, run as
# ``nseg`` parallel legs stacked in the width direction and joined alternately at
# each end, so the body's total centreline length (= the extracted resistance) is
# preserved while the drawn extent collapses to a near-square bank. The fold count
# is DATA-DRIVEN off the natural body length (fold whenever a leg would exceed the
# target), never off a circuit name -- a short resistor stays a single strip
# (``nseg == 1`` is byte-identical to the unfolded builder).
_RES_FOLD_TARGET_LEG_NM = 30_000     # target max drawn length of one serpentine leg
# A 16-leg ceiling is not a physical rule.  It leaves megohm-class 180 nm
# resistors as >180 um strips even though their declared compactness obligation
# calls for the same <=30 um unit-leg construction used by smaller values.  Keep
# a generous resource guard (rather than no guard at all), but let ordinary
# precision analog banks reach the length-derived leg count.  The authoritative
# Bandgap reference uses 75--100 legs; 128 covers that class while still bounding
# polygon count deterministically.
_RES_FOLD_MAX_SEGMENTS = 128


def resistor_fold_segments(
    body_l_nm: int, *, target_leg_nm: int = _RES_FOLD_TARGET_LEG_NM,
    max_segments: int = _RES_FOLD_MAX_SEGMENTS,
) -> int:
    """The number of serpentine legs to fold a resistor BODY of length ``body_l_nm``
    into so each leg's drawn length stays <= ``target_leg_nm`` (a compact bank).

    Pure arithmetic on the body length -- the SAME for every PDK / circuit (no
    per-circuit branch). 1 for a body already shorter than the target (the unfolded
    single strip), else ceil(body_l / target_leg) clamped to ``max_segments``."""
    if body_l_nm <= target_leg_nm:
        return 1
    n = (int(body_l_nm) + target_leg_nm - 1) // target_leg_nm
    return max(1, min(int(max_segments), n))


def _is_gf180(tech) -> bool:
    return getattr(tech, "pdk_id", "").lower() == "gf180mcu"


def _grid_nm(tech) -> int:
    """The PDK manufacturing grid (nm) -- gf180 is 5nm; other PDKs node-scale to a
    safe 5nm floor (every emitted edge is snapped to this so the cell is on-grid by
    construction, clearing the deck's OFFGRID checks)."""
    if _is_gf180(tech):
        return _GF180_GRID_NM
    return 5


def _res_implant_kind(tech) -> str:
    """The implant the poly-resistor BODY is covered by.

    Tied to the PDK's LVS poly-resistor reference model: a ``ppolyf_*`` model is a
    P+ poly resistor (pplus), an ``npolyf_*`` model is N+ poly (nplus). gf180's
    default reference (``clay/flow/step.py``) is ``ppolyf_u`` -> pplus. Resolved
    from the adapter's ``lvs_reference_models()['poly_res']`` when present so the
    drawn implant matches the model the reference card + extractor agree on."""
    model = ""
    try:
        models = tech.lvs_reference_models() or {}
        model = str(models.get("poly_res", "") or "")
    except Exception:  # noqa: BLE001
        model = ""
    if not model:
        model = getattr(getattr(tech, "adapter", None), "poly_res_model", "") or ""
    if not model:
        from clay.layout.passive_cell import _resolve_poly_res_facts

        try:
            facts, _provenance = _resolve_poly_res_facts(tech.pdk_id)
            model = str(facts.sheet_model or "")
        except KeyError:
            pass
    low = str(model).lower()
    if low.startswith(("npolyf", "nplus", "npolyf_u", "rnpo", "poly_res", "rnlpoly")):
        return "nplus"
    # default (gf180 ppolyf_u, and the general unsalicided-P+-poly convention)
    return "pplus"


class UnsupportedPdkFactError(ValueError):
    """A geometry builder asked for a process fact this PDK never published.
    5-state discipline: this is UNSUPPORTED -- never substitute another PDK's
    constant (the F3 incident: external_pdk silently sized with gf180's 350)."""


def _poly_sheet_rho(tech) -> float:
    """The poly resistor sheet resistance (Ohm/square) -- the SAME value the LVS
    reference card uses (``step._lvs_ref_profile``). Resolution order: the
    adapter's own ``poly_sheet_rho_ohm_sq`` (a real published fact) first, then
    the tabulated passive-facts table (``passive_cell._resolve_poly_res_facts``,
    which also lets a future adapter override win); if BOTH are silent this
    raises :class:`UnsupportedPdkFactError` rather than silently sizing with
    another PDK's constant (gf180's 350 was the F3 incident)."""
    rho = getattr(getattr(tech, "adapter", None), "poly_sheet_rho_ohm_sq", None)
    if rho:
        return float(rho)
    try:
        from clay.layout.passive_cell import _resolve_poly_res_facts

        facts, _prov = _resolve_poly_res_facts(tech.pdk_id)
        if facts.sheet_ohm_sq:
            return float(facts.sheet_ohm_sq)
    except KeyError:
        pass
    raise UnsupportedPdkFactError(
        f"{tech.pdk_id}: no poly sheet-rho published (adapter and passive "
        f"facts both silent) -- resistor synthesis is unsupported on this PDK")


def _res_dims(tech, csize: int) -> Dict[str, int]:
    """Resolve the integer resistor DRC dimensions (nm) for ``tech``.

    gf180 uses its calibrated LRES/SB constants (the resistor rules are not
    single-token container values); other PDKs node-scale from the contact size so
    the cell stays legal-by-construction."""
    if _is_gf180(tech):
        return {
            "body_min_w": _GF180_RES_BODY_MIN_WIDTH_NM,
            "implant_overlap": _GF180_RES_IMPLANT_OVERLAP_NM,
            "sab_overlap": _GF180_RES_SAB_OVERLAP_NM,
            "sab_to_contact": _GF180_RES_SAB_TO_CONTACT_NM,
            "poly_past_sab": _GF180_RES_POLY_PAST_SAB_NM,
            "sab_min_w": _GF180_SAB_MIN_WIDTH_NM,
            "sab_min_area": _GF180_SAB_MIN_AREA_NM2,
            "poly_contact_enc": _GF180_POLY_CONTACT_ENC_NM,
            "m1_contact_enc": _GF180_M1_CONTACT_ENC_NM,
        }
    # node-scaled fallbacks (keep clearances >= a contact pitch). Where the PDK
    # PUBLISHES the rule, the parsed value wins over the node-scaled guess: the
    # poly-over-contact enclosure guess (c//3 = 73nm at 0.18um) under-encloses
    # open_pdk's CO.EN.3 (parsed contact_enclosure_poly = 0.10um), and the
    # synthesized KLayout DRC deck flagged every emitted resistor head contact
    # (measured 2026-06-12: 4x CO.EN.3 on the canonical test4 route). A larger
    # enclosure only GROWS clearance -- legal-by-construction is preserved.
    c = csize or 200
    try:
        pce = int(tech.enclosure("contact", "poly") or 0)
    except Exception:
        pce = 0
    try:
        mce = int(tech.enclosure("contact", "metal1") or 0)
    except Exception:
        mce = 0
    return {
        "body_min_w": max(4 * c, 800),
        "implant_overlap": max(c, 300),
        "sab_overlap": max(c, 280),
        "sab_to_contact": max(c, 220),
        "poly_past_sab": max(c, 220),
        "sab_min_w": max(2 * c, 420),
        "sab_min_area": 0,
        "poly_contact_enc": max(pce, c // 3, 70),
        "m1_contact_enc": max(mce, c // 3, 60),
    }


def resistor_body_length_nm(value_ohms: float, tech, *, w_um: float = _REF_RES_WIDTH_UM) -> int:
    """The drawn poly-resistor BODY length (nm) for ``value_ohms``.

    Uses the SAME closed form the LVS reference card uses
    (``L = (R / rho) * W``, ``clay/flow/step.py::_lower_device_card``), so the
    extracted body's W/L matches the reference and the BResistor geometry compare
    passes. ``rho`` is the PDK poly sheet-rho; ``w_um`` is the body width (the
    reference width, also the LRES.1 floor)."""
    rho = _poly_sheet_rho(tech)
    length_um = (float(value_ohms) / rho) * float(w_um)
    return max(1, int(round(length_um * 1000.0)))


def materialized_resistor_body_length_nm(
    value_ohms: float,
    tech,
    *,
    w_um: float = _REF_RES_WIDTH_UM,
) -> int:
    """Exact body length the materializer emits after manufacturing-grid snap.

    The ideal sheet-resistance length is rounded to integer nanometres first,
    then grown to the PDK grid exactly as :func:`build_resistor` does.  Reference
    lowering and geometry generation share this function so LVS compares the
    declared obligation against materialized geometry, not an unsnapped ideal.
    """
    grid = _grid_nm(tech)
    raw_nm = resistor_body_length_nm(value_ohms, tech, w_um=w_um)
    return ((int(raw_nm) + grid - 1) // grid) * grid


def _emit_resistor_serpentine(
    _rect, _up, _dn, d, grid, w_nm, body_l, nseg, so, io, csize,
    head_run, _contact_and_pad, poly2_space, *, poly, sab, res_mk, implant,
):
    """Draw the resistor BODY as a serpentine of ``nseg`` parallel legs folded into
    a compact BANK, preserving the total centreline path length ``body_l`` (= the
    extracted resistance). Returns ``(p_pad, n_pad, comp_bbox)``; appends every drawn
    polygon to the shared ``polys`` list via ``_rect``.

    Geometry (legs run along x, stacked in y; current snakes leg0 -> turn -> leg1 ...)::

        head (p) ===[ leg 0 ]===========================+
                                                        | turn
                 +==========================[ leg 1 ]===+
                 | turn
                 +===[ leg 2 ]==========================... ===[ tail (n) ]

    Each leg is ``w_nm`` wide; the y-pitch between leg centres is ``w_nm + leg_gap``
    so adjacent legs clear the poly2 min-space. ``nseg-1`` vertical poly connectors
    (each ``w_nm`` wide) join the legs alternately at the right / left ends. The whole
    snake is ONE continuous poly path covered by sab + res_mk + implant so the deck
    ANDs it into a single ``ppolyf_u`` body whose length is the snake centreline. The
    two FREE ends stick out past sab/res_mk as bare poly2 (poly2_con) -> p / n
    contact heads, exactly like the unfolded strip's heads."""
    # leg-to-leg gap: adjacent serpentine legs are the SAME poly net (one continuous
    # snake), so the binding clearance is the poly2 min-space (``poly2_space``) -- the
    # gap a router-free same-net poly turn needs. The sab + res_mk cover is drawn as
    # ONE merged box over the whole legs span (below), so there is no sab-to-sab
    # sliver across the gap to worry about. Pure rule DATA, no literal.
    leg_gap = _up(max(poly2_space, grid))
    pitch = w_nm + leg_gap  # leg centre-to-centre in y

    # KLayout's BResistor sees the marked polygon, not an abstract centreline.
    # Each vertical connector overlaps a ``w_nm * w_nm`` square at BOTH joined
    # legs, so its newly contributed area/width is ``pitch - w_nm`` rather than
    # the full centreline pitch.  The merged marker also exposes ``so`` of the
    # two free poly heads.  Therefore the deck-visible effective length is:
    #
    #   nseg*leg_len + (nseg-1)*(pitch-w_nm) + 2*so
    #
    # Solve that exact materialization obligation and round the leg upward to
    # the manufacturing grid.  This is process-rule geometry only: no resistor
    # value, circuit, or PDK identity enters the correction.
    turn_length = (nseg - 1) * max(0, pitch - w_nm)
    end_length = 2 * so
    remaining = max(nseg * grid, body_l - turn_length - end_length)
    leg_len = max(grid, _up((remaining + nseg - 1) // nseg))
    # the marked (resistive) span of one leg: the sab/res_mk box is shortened by 2*so
    # on each leg end (same LRES.6 reasoning as the strip) so the union of marked legs
    # + connectors spans exactly the centreline (no fixed 2*so offset per leg).
    drawn_leg = max(grid, leg_len)

    legs_x0 = 0
    legs_x1 = legs_x0 + drawn_leg
    # leg i occupies y in [i*pitch, i*pitch + w_nm]; centre cy_i.
    def leg_y0(i: int) -> int:
        return i * pitch

    poly_boxes = []   # every poly sub-rect (legs + connectors)
    # --- legs ---------------------------------------------------------------
    for i in range(nseg):
        y0 = leg_y0(i)
        y1 = y0 + w_nm
        poly_boxes.append((legs_x0, y0, legs_x1, y1))
    # --- vertical connectors join leg i to leg i+1 at alternating x ends -----
    for i in range(nseg - 1):
        y_lo = leg_y0(i)
        y_hi = leg_y0(i + 1) + w_nm
        if i % 2 == 0:
            cx0, cx1 = legs_x1 - w_nm, legs_x1      # join at the RIGHT end
        else:
            cx0, cx1 = legs_x0, legs_x0 + w_nm      # join at the LEFT end
        poly_boxes.append((cx0, y_lo, cx1, y_hi))

    # The two FREE ends: leg 0 and leg (nseg-1). For an EVEN nseg they are on OPPOSITE
    # x-sides (leg0 free at left, last free at right); for ODD nseg both free ends are
    # on the LEFT (leg0 and last both have their open end at x=legs_x0). Each free end
    # gets a head extension of ``head_run`` past the sab so the contact lands on bare
    # poly2 (poly2_con), exactly like the strip heads.
    def _head(i: int, side: str):
        y0 = leg_y0(i)
        y1 = y0 + w_nm
        cy = _dn((y0 + y1) // 2)
        if side == "left":
            hx1 = legs_x0
            hx0 = _dn(legs_x0 - so - head_run)
            poly_boxes.append((hx0, y0, hx1, y1))
            cx = hx0 + d["poly_contact_enc"] + csize // 2
        else:
            hx0 = legs_x1
            hx1 = _up(legs_x1 + so + head_run)
            poly_boxes.append((hx0, y0, hx1, y1))
            cx = hx1 - d["poly_contact_enc"] - csize // 2
        return _contact_and_pad(cx, cy)

    if nseg % 2 == 0:
        p_pad = _head(0, "left")
        n_pad = _head(nseg - 1, "right")
    else:
        p_pad = _head(0, "left")
        n_pad = _head(nseg - 1, "left")

    # --- emit the poly snake -----------------------------------------------
    poly_x0 = min(b[0] for b in poly_boxes)
    poly_x1 = max(b[2] for b in poly_boxes)
    poly_y0 = min(b[1] for b in poly_boxes)
    poly_y1 = max(b[3] for b in poly_boxes)
    for b in poly_boxes:
        _rect(poly, b)

    # --- ONE merged sab + res_mk over the legs span (NOT the head extensions) ---
    # The marked (sab∩res_mk) region is a single box covering every leg + connector,
    # grown by ``so`` in the width direction (LRES.6) but stopping at the legs x-span
    # so the two head extensions stay BARE poly2 (poly2_con) for their contacts. One
    # merged cover avoids any sab-to-sab sliver between legs; the inter-leg field gaps
    # it also covers carry no poly, so marking them is benign. The extracted resistive
    # length is the snake centreline through this marked region == ``body_l``.
    sab_x0 = _dn(legs_x0 - so)
    sab_x1 = _up(legs_x1 + so)
    sab_y0 = _dn(poly_y0 - so)
    sab_y1 = _up(poly_y1 + so)
    _rect(sab, (sab_x0, sab_y0, sab_x1, sab_y1))
    _rect(res_mk, (sab_x0, sab_y0, sab_x1, sab_y1))

    pe = _up(max(d["implant_overlap"], _GF180_UNSAL_POLY_OVERLAP_NM))
    _rect(implant, (poly_x0 - pe, poly_y0 - pe, poly_x1 + pe, poly_y1 + pe))

    comp_bbox = (poly_x0, poly_y0 - io, poly_x1, poly_y1 + io)
    return p_pad, n_pad, comp_bbox


def build_resistor(
    tech,
    value_ohms: float,
    *,
    w_um: float = _REF_RES_WIDTH_UM,
    orient: str = "R0",
    nets: Optional[Dict[str, str]] = None,
) -> DeviceGeom:
    """Build a DRC-legal-by-construction, LVS-extractable poly resistor cell.

    Geometry (x = current-flow / body-length direction, y = body-width direction)::

        contact head (p)         RESISTIVE BODY              contact tail (n)
         M1 pad                  (pplus + sab + res_mk)              M1 pad
        +------+    +-----------------------------------------+    +------+
        | []   |====|  poly2 (covered by implant/sab/res_mk)  |====|   [] |
        +------+    +-----------------------------------------+    +------+
          poly head                  poly2 body                    poly tail
         (bare poly2_con)        L = (R/rho)*W                   (bare poly2_con)

    The single poly2 strip runs head->body->tail. The body is W (>= LRES.1) wide
    and L = ``resistor_body_length_nm`` long, covered by the implant + sab + res_mk
    so the deck ANDs it into ``ppolyf_u``. The heads stick out past sab/res_mk
    (SB.10) so they are bare ``poly2_con``; each lands a contact + an M1 pad =
    the ``p`` / ``n`` terminals the router connects.

    Parameters
    ----------
    tech : LayoutTech
    value_ohms : float
        The schematic resistance; mapped to body length via the PDK poly sheet-rho.
    w_um : float
        Body width in um (default the LVS-reference width = LRES.1 floor).
    nets : dict, optional
        ``{'p','n' -> net}`` binding stamped on the terminal map.

    Returns
    -------
    DeviceGeom
        Polygons on tech-resolved keys + a terminal map ``{'p','n'}`` (each an M1
        contact pad). ``kind == 'res'``; ``finger_count == 1``;
        ``gate_length_nm`` carries the body length, ``channel_width_nm`` the body
        width (so the caller can read the geometry uniformly).
    """
    if orient in ("MR0", "MR90", "MR180", "MR270"):
        return _transform_devicegeom(
            build_resistor(tech, value_ohms, w_um=w_um, orient="R0", nets=nets),
            orient,
            tech,
        )
    poly = tech.build_layer("poly")
    contact = tech.contact_key()
    metal1 = tech.metal_key(1)
    implant = tech.build_layer(_res_implant_kind(tech))
    sab = tech.marker_key("sab")
    res_mk = tech.marker_key("res_mk")
    if poly is None or contact is None or metal1 is None:
        raise ValueError(f"{getattr(tech, 'pdk_id', '?')}: poly/contact/metal1 required for a resistor")
    if sab is None or res_mk is None:
        raise ValueError(
            f"{getattr(tech, 'pdk_id', '?')}: poly resistor needs sab + res_mk marker "
            f"layers (tech.marker_key); got sab={sab}, res_mk={res_mk}"
        )

    contact_via = "contact"
    try:
        csize = tech.via_size(contact_via) or 0
    except Exception:  # noqa: BLE001
        csize = 0
    if not csize:
        csize = 220
    d = _res_dims(tech, csize)
    grid = _grid_nm(tech)

    def _up(v: int) -> int:
        """Round UP to the manufacturing grid (a dimension only ever grows, so the
        DRC clearance it realizes is never eroded by the snap)."""
        return ((int(v) + grid - 1) // grid) * grid

    def _dn(v: int) -> int:
        """Round DOWN to the grid (for a left/bottom edge that moves OUTWARD when it
        decreases -> the box only grows)."""
        return (int(v) // grid) * grid

    # snap the contact + its clearances to the grid so every derived edge is on-grid.
    csize = _up(csize)
    w_nm = _up(max(int(round(float(w_um) * 1000.0)), d["body_min_w"]))
    body_l = materialized_resistor_body_length_nm(
        value_ohms, tech, w_um=w_nm / 1000.0
    )

    polys: list[LayerPolygon] = []

    def _rect(key: LayerKey, box: Bbox) -> None:
        # snap every emitted rect OUTWARD to the grid (min edges down, max edges up)
        # so the cell is on-grid by construction and a snap never shrinks a feature.
        x0, y0, x1, y1 = box
        polys.append(
            LayerPolygon(
                layer=key[0], datatype=key[1],
                polygon=Polygon.rectangle(_dn(x0), _dn(y0), _up(x1), _up(y1)),
            )
        )

    so = _up(d["sab_overlap"])             # LRES.6 sab overlap of body (width dir)
    io = _up(d["implant_overlap"])         # LRES.5 implant overlap of body

    # --- head/tail contact + M1 pad geometry (shared strip + serpentine) -----
    # Each head extends past the sab edge (SB.10 poly-past-sab) + the SAB-to-contact
    # space (LRES.7) + room for the contact + its poly enclosure (CO.3), so the
    # head contact lands on bare poly2 (poly2_con) clear of sab/res_mk.
    head_run = _up(
        d["poly_past_sab"] + d["sab_to_contact"]
        + d["poly_contact_enc"] + csize + d["poly_contact_enc"]
    )
    m1_enc = d["m1_contact_enc"]
    m1_min_area = int(tech.min_area("metal1") or 0)
    m1_area_side = math.isqrt(max(0, m1_min_area))
    if m1_area_side * m1_area_side < m1_min_area:
        m1_area_side += 1
    # The terminal pad is a complete materialized M1 island.  A router landing
    # may sit wholly inside it and add no area, so the passive cell itself owes
    # both contact enclosure and the PDK's declared M1 minimum area.
    m1_pad = _up(max(csize + 2 * m1_enc, 380, m1_area_side))

    def _contact_and_pad(cx: int, cy: int) -> Bbox:
        cx = _dn(cx)
        half_c = csize // 2
        _rect(contact, (cx - half_c, cy - half_c, cx + half_c, cy + half_c))
        half_p = m1_pad // 2
        pad = (_dn(cx - half_p), _dn(cy - half_p), _up(cx + half_p), _up(cy + half_p))
        _rect(metal1, pad)
        return pad

    nets = nets or {}

    # --- FOLD DECISION: how many serpentine legs to fold the body into --------
    # Data-driven off the natural body length only (no circuit branch). nseg == 1
    # is the unfolded single strip (byte-identical to the historic builder); nseg>1
    # folds the SAME continuous poly path into a compact bank.
    nseg = resistor_fold_segments(body_l)

    if nseg == 1:
        # --- the BODY (origin at 0, body width on y in [0, w_nm]) ---------------
        # The EXTRACTED resistor length is the sab∩res_mk∩pplus∩poly2 region, whose
        # x-extent is the sab/res_mk box -- which overlaps the drawn poly body by ``so``
        # on EACH end (LRES.6). So the extracted resistive length is (inner body) + 2*so,
        # not the inner body alone. The LVS reference computes L = R/sheet_rho (=
        # ``resistor_body_length_nm``), which is the resistive length the EXTRACTOR sees.
        # To make the two agree, the inner drawn body is shortened by 2*so so the marked
        # (sab/res_mk) region spans EXACTLY ``body_l`` = R/sheet_rho. Without this the
        # extracted length is body_l + 2*so -- a FIXED 560nm (gf180) offset that is ~2 %
        # on a 10k strip (right at the passive tolerance) but ~13 % on a ~1.5k strip
        # (fails LVS). Generic: pure geometry off the PDK sab overlap, no per-value branch.
        # Floor at one grid step so a (sub-minimum) tiny value never inverts the body.
        body_x0 = 0
        body_x1 = max(grid, body_l - 2 * so)
        body_y0 = 0
        body_y1 = w_nm
        cy = _dn(w_nm // 2)

        # --- salicide block over the body (LRES.6, SB.1 min width, SB.13 min area) -
        sab_x0 = body_x0 - so
        sab_x1 = body_x1 + so
        sab_y0 = body_y0 - so
        sab_y1 = body_y1 + so
        if sab_x1 - sab_x0 < d["sab_min_w"]:
            mid = (sab_x0 + sab_x1) // 2
            sab_x0, sab_x1 = mid - _up(d["sab_min_w"] // 2), mid + _up(d["sab_min_w"] // 2)
        if sab_y1 - sab_y0 < d["sab_min_w"]:
            sab_y0, sab_y1 = cy - _up(d["sab_min_w"] // 2), cy + _up(d["sab_min_w"] // 2)
        if d["sab_min_area"] and (sab_x1 - sab_x0) * (sab_y1 - sab_y0) < d["sab_min_area"]:
            need_w = d["sab_min_area"] // max(1, (sab_y1 - sab_y0))
            if need_w > (sab_x1 - sab_x0):
                grow = _up((need_w - (sab_x1 - sab_x0) + 1) // 2)
                sab_x0 -= grow
                sab_x1 += grow
        sab_x0, sab_x1, sab_y0, sab_y1 = _dn(sab_x0), _up(sab_x1), _dn(sab_y0), _up(sab_y1)
        _rect(sab, (sab_x0, sab_y0, sab_x1, sab_y1))

        # --- the poly strip: head -> body -> tail ---------------------------
        poly_x0 = _dn(sab_x0 - head_run)
        poly_x1 = _up(sab_x1 + head_run)
        _rect(poly, (poly_x0, body_y0, poly_x1, body_y1))

        # --- implant: encloses the POLY box by implant_poly_enc 0.3um (PRES.5/LRES.5)
        # AND >= the unsalicided-poly overlap 0.18um (PP.9/NP.9). The gf180 PDK PCell
        # (cells/klayout/pymacros/cells/draw_res.py::draw_ppolyf_u_res) grows the implant
        # from the POLY box: ``pplus = poly +/- implant_poly_enc``. ``pres_poly`` =
        # poly2 ∩ pplus = the poly box (poly fully inside implant), so the implant
        # encloses it by exactly the overlap on every side. (Earlier growing the implant
        # from the wider SAB box left poly2∩pplus flush with the implant edge -> 0
        # enclosure -> PRES.5 failed; sourcing it from the poly box is the PCell recipe.)
        pe = _up(max(d["implant_overlap"], _GF180_UNSAL_POLY_OVERLAP_NM))
        _rect(implant, (poly_x0 - pe, body_y0 - pe, poly_x1 + pe, body_y1 + pe))

        # --- res_mk over the body (LRES.9a: covers the resistor; its length coincides
        # with the sab/resistor length, its width covers the poly). Sized to the sab x
        # extent and the poly width.
        _rect(res_mk, (sab_x0, body_y0, sab_x1, body_y1))

        # --- head / tail contacts on bare poly2 (poly2_con) + M1 pads --------
        head_cx = poly_x0 + d["poly_contact_enc"] + csize // 2
        tail_cx = poly_x1 - d["poly_contact_enc"] - csize // 2
        p_pad = _contact_and_pad(head_cx, cy)
        n_pad = _contact_and_pad(tail_cx, cy)

        comp_bbox = (poly_x0, min(body_y0 - io, sab_y0), poly_x1, max(body_y1 + io, sab_y1))
    else:
        # Same-net poly min-space (the leg-to-leg clearance the snake's parallel
        # legs need); resolve the target PDK's actual rule-container layer name
        # (GF180 ``poly2`` versus external_pdk/external_pdk ``poly``), with a safe grid floor when
        # the adapter is silent.
        try:
            poly2_space = int(tech.min_space(_poly_name(tech)) or 0)
        except Exception:  # noqa: BLE001
            poly2_space = 0
        if not poly2_space:
            poly2_space = max(grid, csize)
        p_pad, n_pad, comp_bbox = _emit_resistor_serpentine(
            _rect, _up, _dn, d, grid, w_nm, body_l, nseg, so, io, csize,
            head_run, _contact_and_pad, poly2_space,
            poly=poly, sab=sab, res_mk=res_mk, implant=implant,
        )

    # --- ORIENTATION: rotate the whole resistor about the origin by ``orient``,
    # then renormalise so its lower-left drawn corner returns to the R0 drawn
    # min corner (the placer translates by ``extent()`` so the local frame is all
    # that moves). For R90/R270 the head->body->tail axis runs VERTICALLY (the
    # p/n pads at top/bottom instead of left/right). Rotation is LVS-invariant:
    # each pad keeps its net, only its geometry moves. The grid re-snap confirms
    # the rotated rectilinear edges stay on the PDK grid (they already are).
    if orient not in _ORIENTS:
        raise ValueError(f"orient must be one of {_ORIENTS!r}, got {orient!r}")
    if orient != "R0":
        r0_min = (
            min(_bbox(lp)[0] for lp in polys),
            min(_bbox(lp)[1] for lp in polys),
        )
        polys = [_rotate_poly(lp, orient) for lp in polys]
        p_pad = _rotate_box(p_pad, orient)
        n_pad = _rotate_box(n_pad, orient)
        comp_bbox = _rotate_box(comp_bbox, orient)
        rot_min = (
            min(_bbox(lp)[0] for lp in polys),
            min(_bbox(lp)[1] for lp in polys),
        )
        ndx, ndy = r0_min[0] - rot_min[0], r0_min[1] - rot_min[1]
        gg = _device_grid_nm(tech)

        def _norm_snap(v: int, axis_shift: int) -> int:
            v += axis_shift
            return (v // gg) * gg if gg > 1 else v

        polys = [
            LayerPolygon(
                layer=lp.layer, datatype=lp.datatype,
                polygon=Polygon(
                    vertices=[
                        Point(_norm_snap(v.x, ndx), _norm_snap(v.y, ndy))
                        for v in lp.polygon.vertices
                    ]
                ),
            )
            for lp in polys
        ]

        def _shift_box(b: Bbox) -> Bbox:
            return (b[0] + ndx, b[1] + ndy, b[2] + ndx, b[3] + ndy)

        p_pad = _shift_box(p_pad)
        n_pad = _shift_box(n_pad)
        comp_bbox = _shift_box(comp_bbox)

    terminals: Dict[str, Terminal] = {
        "p": Terminal(role="p", bbox=p_pad, net=nets.get("p", "")),
        "n": Terminal(role="n", bbox=n_pad, net=nets.get("n", "")),
    }

    return DeviceGeom(
        polygons=polys,
        terminals=terminals,
        kind="res",
        finger_count=1,
        gate_length_nm=body_l,
        channel_width_nm=w_nm,
        comp_bbox=comp_bbox,
        orient=orient,
        source_rail_bbox=p_pad,   # back-compat: 'p' pad
        drain_rail_bbox=n_pad,    # back-compat: 'n' pad
    )


# --- gf180mcu MIM-capacitor (Option A) DRC dimensions (nm), each traced to a rule
# in pdks/gf180mcu/.../rules/klayout/drc/rule_decks/mim_capacitor_option_a_.drc.
# The MIM stack the LVS deck (gf180mcu.lvs, MIM_OPTION "A") extracts is:
#   bottom plate = Metal2 (36/0) under the FuseTop+1.06um oversize region
#   top plate    = FuseTop (75/0), inside cap_mk (117/5) AND mim_l_mk (117/10)
#   the device   = capacitor("mim_<d>fF", <d>e-15, MIMCap), {P1=>mim_virtual (bottom
#                  Metal2), P2=>fuse_cap (FuseTop top)}; connect(metal2,mim_virtual)
#                  and connect(fuse_cap,via2) -- so BOTH plates rise through Via2
#                  (38/0) to Metal3 (42/0) routable pads (no Via1 may touch the MIM
#                  bottom Metal2, MIM.10). The extracted C = density * A(fusetop).
_GF180_MIM_BOTPLATE_TO_METAL_NM = 1200  # MIM.1 bottom-plate Metal2 to other Metal2
_GF180_MIM_TOPPLATE_OVERLAP_NM = 600   # MIM.3 bottom-plate Metal2 overlap of FuseTop
_GF180_MIM_BOTPLATE_VIA2_ENC_NM = 400  # MIM.2 bottom-plate Metal2 overlap of Via2
_GF180_MIM_TOPPLATE_VIA2_ENC_NM = 400  # MIM.4 FuseTop overlap of Via2
_GF180_MIM_TOP_TO_BOTVIA2_NM = 400     # MIM.5 FuseTop -> bottom-plate Via2 spacing
_GF180_MIM_VIA2_SEA_SPACE_NM = 500     # MIM.9 Via2 sea-of-via spacing on top plate
_GF180_MIM_FUSETOP_MIN_AREA_NM2 = 25_000_000   # MIM.8a min FuseTop area (25 um^2)
_GF180_MIM_FUSETOP_MAX_AREA_NM2 = 10_000_000_000  # MIM.8b max single FuseTop (1e4 um^2)
_GF180_MIM_CAPMK_ENC_NM = 0            # MIM.7 needs only fusetop INSIDE cap_mk; grow
                                       # cap_mk/mim_l_mk a margin so the marker fully
                                       # encloses + interacts the top plate.
_GF180_MIM_MARK_MARGIN_NM = 200        # cap_mk / mim_l_mk grow past FuseTop (>=0)

#: gf180mcu MIM area-specific capacitance, F/um^2, KEYED by the deck's MIM_CAP
#: option ("1"/"1.5"/"2"). These are the EXACT second argument of the LVS
#: ``capacitor("mim_<d>fF", <density>, MIMCap)`` extraction calls (gf180mcu.lvs,
#: MIMCAP EXTRACTION) -- i.e. the F per um^2 KLayout multiplies by the FuseTop area
#: to report the extracted C. They equal the model-card nominal densities in
#: sm141064.ngspice (``mim_1p0fF`` = 1.0 fF/um^2 usable <=20V, ``mim_1p5fF`` = 1.5,
#: ``mim_2p0fF`` = 2.0). The default option is "1" (1.0 fF/um^2): the only MIM
#: flavour rated to the full 20 V, so it is safe for any 3.3 V / 6 V gf180 design.
_GF180_MIM_DENSITY_F_PER_UM2: Dict[str, float] = {
    "1": 1.0e-15,
    "1.5": 1.5e-15,
    "2": 2.0e-15,
}
_GF180_MIM_DEFAULT_CAP_OPTION = "1"

#: gf180mcu MIM device CLASS NAME keyed by the deck's MIM_CAP option -- the EXACT
#: first argument of the LVS ``capacitor("mim_<d>fF", <density>, MIMCap)``
#: extraction call (gf180mcu.lvs, MIMCAP EXTRACTION). The KLayout comparer pairs
#: devices by class name and the reference reader keys the cap class off the
#: card's MODEL token, so the flow's LVS reference C card must emit this exact
#: name (``clay/flow/step.py::_lower_device_card``) or a generic reference cap
#: never pairs with the named extracted device.
_GF180_MIM_MODEL_BY_CAP_OPTION: Dict[str, str] = {
    "1": "mim_1p0fF",
    "1.5": "mim_1p5fF",
    "2": "mim_single_2p0fF",
}


def mim_model_name(cap_option: str = _GF180_MIM_DEFAULT_CAP_OPTION) -> str:
    """The gf180 MIM device class name for ``cap_option`` (``mim_1p0fF`` etc.).

    The same name the LVS deck registers in ``capacitor("mim_<d>fF", ...)``; the
    reference C card emits it as its model token so the comparer pairs the
    reference cap with the extracted ``mim_<d>fF`` device class."""
    return _GF180_MIM_MODEL_BY_CAP_OPTION.get(
        str(cap_option), _GF180_MIM_MODEL_BY_CAP_OPTION[_GF180_MIM_DEFAULT_CAP_OPTION]
    )


def mim_density_f_per_um2(
    tech,
    cap_option: str = _GF180_MIM_DEFAULT_CAP_OPTION,
    *,
    mim_option: str = "A",
) -> float:
    """The MIM area-specific capacitance (F/um^2) for ``cap_option``.

    gf180: sourced from the PDK LVS deck's ``capacitor("mim_<d>fF", <density>,
    MIMCap)`` extraction second argument (gf180mcu.lvs); the extracted device C
    the LVS comparator emits is ``density * FuseTop_area_um2``. Used both to
    size the FuseTop plate (``area = C / density``) and to confirm the
    reference card's C matches the extracted one. Byte-identical to the prior
    cap_option-keyed table lookup (this branch never consumed ``tech``, and
    still does not need to -- gf180 has exactly one MIM stack).

    Non-gf180: the adapter surface publishes no MIM-density field today (a
    future adapter that does would win here, mirroring ``_poly_sheet_rho``'s
    adapter-first resolution order); resolved instead from the tabulated
    passive facts (``passive_cell._resolve_mim_facts``, whose
    ``density_ff_um2`` is fF/um^2 and is converted to F/um^2 by ``* 1e-15``).
    Raises :class:`UnsupportedPdkFactError` for a PDK with no tabulated MIM
    facts -- this function used to IGNORE ``tech`` entirely and always return
    the gf180 table (the A2 dead-param incident: external_pdk got 1.0e-15 vs its real
    0.971e-15 deck density, 2.99% apart); it now never substitutes another
    PDK's density."""
    if _is_gf180(tech):
        return _GF180_MIM_DENSITY_F_PER_UM2.get(
            str(cap_option), _GF180_MIM_DENSITY_F_PER_UM2[_GF180_MIM_DEFAULT_CAP_OPTION]
        )
    try:
        from clay.layout.passive_cell import _resolve_mim_facts

        facts, _prov = _resolve_mim_facts(tech.pdk_id)
        if facts.density_ff_um2:
            return float(facts.density_ff_um2) * 1e-15  # fF/um^2 -> F/um^2
    except KeyError:
        pass
    raise UnsupportedPdkFactError(
        f"{tech.pdk_id}: no MIM cap density published (passive facts silent) "
        f"-- MIM capacitor synthesis is unsupported on this PDK")


def mim_plate_side_nm(
    value_farads: float,
    tech,
    cap_option: str = _GF180_MIM_DEFAULT_CAP_OPTION,
) -> int:
    """The drawn square FuseTop top-plate side (nm) for ``value_farads``.

    ``area = C / density`` (density = :func:`mim_density_f_per_um2`); the plate is
    a square of that area, clamped UP to the MIM.8a 25 um^2 minimum and DOWN to
    the MIM.8b 10000 um^2 maximum single-cap area. The extracted C the LVS deck
    reports is ``density * area``.  Return the MATERIALIZED side, rounded up to
    the PDK manufacturing grid, rather than the ideal pre-grid side: both the
    polygon builder and the LVS reference consume this helper, so they must use
    the exact same A/P geometry."""
    density = mim_density_f_per_um2(tech, cap_option)
    area_um2 = float(value_farads) / density           # um^2
    area_nm2 = area_um2 * 1_000_000.0                   # nm^2
    area_nm2 = max(area_nm2, float(_GF180_MIM_FUSETOP_MIN_AREA_NM2))
    area_nm2 = min(area_nm2, float(_GF180_MIM_FUSETOP_MAX_AREA_NM2))
    side = max(1, int(round(area_nm2 ** 0.5)))
    grid = max(1, int(_grid_nm(tech)))
    return ((side + grid - 1) // grid) * grid


def equal_size_cap_plate_side_nm(
    values_farads,
    tech,
    cap_option: str = _GF180_MIM_DEFAULT_CAP_OPTION,
) -> int:
    """The COMMON FuseTop plate side (nm) for a set of caps in one ``@equal_size``
    group: the group-wise MAXIMUM natural plate side over ``values_farads``.

    Each cap's natural plate side is :func:`mim_plate_side_nm` (``area = C/density``);
    the group is pinned to the LARGEST so every smaller-value member grows its plate
    to match (capacitance preserved-or-increased, DRC-legal). Pass the result as
    :func:`build_capacitor`'s ``plate_side_override_nm`` for each member so the
    EMITTED plates come out equal. Returns 0 for an empty set."""
    sides = [
        mim_plate_side_nm(float(v), tech, cap_option) for v in (values_farads or [])
    ]
    return max(sides) if sides else 0


def _resolve_gf180_mim_stack(
    tech,
    *,
    mim_option: str = "A",
    metal_level: str = "5LM",
) -> Tuple[LayerKey, LayerKey, LayerKey, str, str, str]:
    """Return ``(bot_metal, top_metal, top_via, bot_name, top_name, via_name)``."""
    from clay.pdk.gf180mcu import normalize_gf180_mim_option

    opt = normalize_gf180_mim_option(mim_option)
    if opt == "A":
        bot_name, top_name, via_name, via_idx = "metal2", "metal3", "via2", 2
        bot_idx, top_idx = 2, 3
    else:
        lvl = str(metal_level or "5LM").strip().upper()
        if lvl == "5LM":
            bot_name, top_name, via_name, via_idx = "metal4", "metal5", "via4", 4
            bot_idx, top_idx = 4, 5
        elif lvl == "4LM":
            bot_name, top_name, via_name, via_idx = "metal3", "metal4", "via3", 3
            bot_idx, top_idx = 3, 4
        else:
            raise ValueError(
                f"gf180mcu Option-B MIM supports metal_level '4LM' or '5LM', got {metal_level!r}"
            )
    if hasattr(tech, "metal_key"):
        bot_metal = tech.metal_key(bot_idx)
        top_metal = tech.metal_key(top_idx)
        top_via = tech.via_key(via_idx)
    else:
        bot_metal = tech.layer(bot_name)
        top_metal = tech.layer(top_name)
        top_via = tech.layer(via_name)
    if bot_metal is None or top_metal is None or top_via is None:
        raise ValueError(
            f"{getattr(tech, 'pdk_id', '?')}: MIM Option-{opt} cap needs "
            f"{bot_name}/{top_name}/{via_name}"
        )
    return (
        tuple(bot_metal),
        tuple(top_metal),
        tuple(top_via),
        bot_name,
        top_name,
        via_name,
    )


def _gf180_mim_connect_spec(
    tech,
    *,
    mim_option: str = "A",
    metal_level: str = "5LM",
):
    """Return Option-A or Option-B :class:`MimConnectSpec` for GF180."""
    from clay.layout.device_terminal import gf180_mim_spec
    from clay.pdk.gf180mcu import normalize_gf180_mim_option

    opt = normalize_gf180_mim_option(mim_option)
    return gf180_mim_spec(tech, mim_option=opt, metal_level=metal_level)


def _build_capacitor(
    tech,
    value_farads: float,
    *,
    cap_option: str = _GF180_MIM_DEFAULT_CAP_OPTION,
    mim_option: Optional[str] = None,
    metal_level: str = "5LM",
    pex_backend: Optional[str] = None,
    orient: str = "R0",
    nets: Optional[Dict[str, str]] = None,
    plate_side_override_nm: Optional[int] = None,
) -> DeviceGeom:
    """Build a DRC-legal-by-construction, LVS-extractable gf180 MIM capacitor.

    PRIVATE PER-PDK PRIMITIVE -- not the public surface. This is the bare gf180-only
    unit cell (FuseTop/Metal2 stack); it RAISES on any non-gf180 PDK. Production code
    MUST go through the per-PDK dispatcher :func:`build_capacitor_for_tech` (gf180 ->
    here, open_pdk -> ``build_mim_capacitor_tsmc``, external_pdk -> the external_pdk MIM primitive),
    never this bare builder. The module-public name ``build_capacitor`` is a frozen
    backward-compat alias of this function (kept for gf180-specific unit tests and the
    per-PDK-dispatch obligation checker) and is intentionally absent from ``__all__``;
    the import-boundary guard ``tests/test_encapsulation_boundary.py`` fences NEW
    production importers of the bare name.

    The MIM sibling of :func:`build_resistor`: the SAME resolved-PDK-facts style
    and the SAME :class:`DeviceGeom` output (polygons + a ``{'p','n'}`` terminal
    map), realising the gf180mcu MIM-Option-A stack the LVS deck extracts.

    Geometry (x to the right, y up)::

        metal3 pad 'p'                              metal3 pad 'n'
       (over bottom-plate via2)                  (over top-plate via2 sea)
        +------+                                   +-----------------+
        | via2 |        FuseTop top plate          |  via2  via2 ... |   <- Metal3
        +------+        (mim_l_mk + cap_mk)         |  via2  via2 ... |
           ||      +------------------------------+ +-----------------+
        Metal2 bottom plate (overlaps FuseTop by   |   FuseTop 75/0   |
        MIM.3 0.6um; the via2 here is OUTSIDE the   +-----------------+
        FuseTop+1.06um zone so it is routable)

    * **Top plate** = ``FuseTop`` (75/0): a square of side
      :func:`mim_plate_side_nm` so its area * density = ``value_farads`` (density
      from the PDK LVS deck, :func:`mim_density_f_per_um2`). It is fully inside
      ``cap_mk`` (117/5) and ``mim_l_mk`` (117/10) so the deck's
      ``fuse_cap = fusetop.interacting(cap_mk).interacting(mim_l_mk)`` selects it.
      A sea of ``via2`` (38/0) lands on it (MIM.4 0.4um enclosure, MIM.9 0.5um
      via spacing) rising to a ``Metal3`` pad = the ``n`` terminal.
    * **Bottom plate** = ``Metal2`` (36/0): overlaps the FuseTop by MIM.3 0.6um on
      every side, then extends right to hold a routable ``via2`` (placed OUTSIDE
      the FuseTop+1.06um virtual-plate zone, >= MIM.5 0.4um from the top plate,
      enclosed by Metal2 by MIM.2 0.4um) rising to a second ``Metal3`` pad = the
      ``p`` terminal. (MIM.10: no ``via1`` touches the MIM bottom Metal2.)

    The extracted device is ``mim_<d>fF`` with ``C = density * area`` and the two
    plate nets; the flow's LVS reference card emits the SAME ``C`` so it matches
    (``clay/flow/step.py::_lower_device_card`` for ``C``). Every dimension is
    sourced from the resolved PDK tech / the MIM DRC rules above; the cell is
    on the 5nm grid by construction.

    Parameters
    ----------
    tech : LayoutTech  (gf180mcu)
    value_farads : float
        The schematic capacitance; mapped to FuseTop area via the MIM density.
    cap_option : str
        The MIM flavour ("1" / "1.5" / "2"); default "1" (1.0 fF/um^2, <=20V).
    nets : dict, optional
        ``{'p','n' -> net}`` binding stamped on the terminal map.
    plate_side_override_nm : int, optional
        Force the drawn square FuseTop top-plate side to this value (nm) instead
        of the value-derived :func:`mim_plate_side_nm`. This is how a declared
        ``@equal_size`` group makes every member emit the SAME plate GEOMETRY: the
        group's plate side is set to the group-wise MAXIMUM (the largest-value cap),
        so a smaller-value member GROWS its plate to match (never shrinks). The
        whole MIM stack (cap_mk, mim_l_mk, via sea, Metal2/Metal3 plates, the
        terminal exits) derives from ``side``, so the override keeps the cell
        DRC-legal-by-construction. The override is floored at the value-derived side
        (so it only ever grows the plate) and clamped to the MIM.8b max single-cap
        area. ``None`` (the default) -> the value-derived side (byte-parity for a
        cell that declares no ``@equal_size``). The card's electrical C annotation
        (the LVS reference) stays the nominal value; only the drawn plate grows.

    Returns
    -------
    DeviceGeom
        ``kind == 'cap'``; ``finger_count == 1``; the FuseTop side is carried in
        ``gate_length_nm`` / ``channel_width_nm`` (a square plate) so a caller can
        read the geometry uniformly. The ``p`` / ``n`` terminals are Metal3 pads.
    """
    if not _is_gf180(tech):
        raise ValueError(
            f"{getattr(tech, 'pdk_id', '?')}: MIM capacitor builder is gf180mcu-"
            f"specific (the MIM stack -- FuseTop top plate, Metal2/Metal4 bottom plate, "
            f"Via2/Via4 risers -- differs per process); no MIM stack resolved for this PDK"
        )
    from clay.pdk.gf180mcu import resolve_gf180_mim_config

    norm_option, cap_option, norm_metal_level = resolve_gf180_mim_config(
        tech,
        mim_option=mim_option,
        cap_option=cap_option,
        metal_level=metal_level,
        pex_backend=pex_backend,
    )
    (
        bot_metal,
        top_metal,
        top_via,
        bot_metal_name,
        top_metal_name,
        via_rule_name,
    ) = _resolve_gf180_mim_stack(
        tech, mim_option=norm_option, metal_level=norm_metal_level
    )
    fusetop = tech.marker_key("fusetop")
    cap_mk = tech.marker_key("cap_mk")
    mim_l_mk = tech.marker_key("mim_l_mk")
    if fusetop is None or cap_mk is None or mim_l_mk is None:
        raise ValueError(
            f"{getattr(tech, 'pdk_id', '?')}: MIM cap needs fusetop + cap_mk + "
            f"mim_l_mk marker layers (tech.marker_key); got fusetop={fusetop}, "
            f"cap_mk={cap_mk}, mim_l_mk={mim_l_mk}"
        )

    def _rv(x) -> int:
        return int(getattr(x, "value", x)) if x is not None else 0

    try:
        vr = tech.via_rule(via_rule_name)
        via_size = _rv(vr.size) if vr is not None else 0
        via_enc = _rv(vr.enclosure_below) if vr is not None else 0
    except Exception:  # noqa: BLE001
        via_size, via_enc = 0, 0
    if not via_size:
        via_size = int(tech.via_size(via_rule_name) or 260)
    via_size = int(via_size)
    via_enc = int(via_enc) if via_enc else 60
    try:
        bot_min_w = int(tech.min_width(bot_metal_name) or 0)
        top_min_w = int(tech.min_width(top_metal_name) or 0)
        bot_min_area = int(tech.min_area(bot_metal_name) or 0)
    except Exception:  # noqa: BLE001
        bot_min_w, top_min_w, bot_min_area = 0, 0, 0

    grid = _grid_nm(tech)

    def _up(v: float) -> int:
        return ((int(round(v)) + grid - 1) // grid) * grid

    def _dn(v: float) -> int:
        return (int(round(v)) // grid) * grid

    polys: list[LayerPolygon] = []

    def _rect(key: LayerKey, box: Bbox) -> None:
        x0, y0, x1, y1 = box
        polys.append(
            LayerPolygon(
                layer=key[0], datatype=key[1],
                polygon=Polygon.rectangle(_dn(x0), _dn(y0), _up(x1), _up(y1)),
            )
        )

    # --- the FuseTop top plate (a square sized so area*density == C) ----------
    # The value-derived side is the natural plate (area*density == C). When an
    # @equal_size group supplies ``plate_side_override_nm`` (the group-wise MAX
    # plate), the plate GROWS to that common side so every group member emits an
    # IDENTICAL plate -- floored at the natural side (never shrink, so C is
    # preserved-or-increased) and clamped to the MIM.8b max single-cap area.
    natural_side = _up(mim_plate_side_nm(value_farads, tech, cap_option))
    side = natural_side
    if plate_side_override_nm is not None:
        max_side = int(round(float(_GF180_MIM_FUSETOP_MAX_AREA_NM2) ** 0.5))
        side = min(_up(max(int(plate_side_override_nm), natural_side)), _dn(max_side))
        side = max(side, natural_side)
    bot_ov = _up(_GF180_MIM_TOPPLATE_OVERLAP_NM)          # MIM.3
    top_via_enc = _up(max(_GF180_MIM_TOPPLATE_VIA2_ENC_NM, via_enc))  # MIM.4
    bot_via_enc = _up(max(_GF180_MIM_BOTPLATE_VIA2_ENC_NM, via_enc))  # MIM.2
    top_to_botvia = _up(_GF180_MIM_TOP_TO_BOTVIA2_NM)     # MIM.5
    sea_space = _up(_GF180_MIM_VIA2_SEA_SPACE_NM)         # MIM.9

    # FuseTop sits with its lower-left at (bot_ov, bot_ov) so the bottom plate can
    # overlap it on the left/bottom too. (Origin stays at 0,0 = bottom-plate LL.)
    ft_x0 = bot_ov
    ft_y0 = bot_ov
    ft_x1 = ft_x0 + side
    ft_y1 = ft_y0 + side
    _rect(fusetop, (ft_x0, ft_y0, ft_x1, ft_y1))

    # cap_mk + mim_l_mk fully ENCLOSE the FuseTop (MIM.7 needs FuseTop inside
    # cap_mk; the deck also needs both markers to INTERACT the top plate).
    mk = _up(_GF180_MIM_MARK_MARGIN_NM)
    _rect(cap_mk, (ft_x0 - mk, ft_y0 - mk, ft_x1 + mk, ft_y1 + mk))
    _rect(mim_l_mk, (ft_x0 - mk, ft_y0 - mk, ft_x1 + mk, ft_y1 + mk))

    # --- sea of via2 on the top plate (FuseTop), rising to a Metal3 pad -------
    # Each cut is enclosed by the FuseTop by >= MIM.4; the cuts are spaced >= MIM.9.
    via_pitch = via_size + sea_space
    sea_x0 = ft_x0 + top_via_enc
    sea_y0 = ft_y0 + top_via_enc
    sea_x1 = ft_x1 - top_via_enc
    sea_y1 = ft_y1 - top_via_enc
    top_cuts: list[Bbox] = []
    if sea_x1 - sea_x0 >= via_size and sea_y1 - sea_y0 >= via_size:
        nx = max(1, (sea_x1 - sea_x0 - via_size) // via_pitch + 1)
        ny = max(1, (sea_y1 - sea_y0 - via_size) // via_pitch + 1)
        for ix in range(nx):
            for iy in range(ny):
                cx0 = _dn(sea_x0 + ix * via_pitch)
                cy0 = _dn(sea_y0 + iy * via_pitch)
                cut = (cx0, cy0, cx0 + via_size, cy0 + via_size)
                _rect(top_via, cut)
                top_cuts.append(cut)
    # The top-plate cover sits directly over the FuseTop (the gf180 MIM PCell
    # draws topmet == fusetop box); it encloses the top-via sea by the via above-enc.
    top_cover = (
        _dn(ft_x0 + top_via_enc - via_enc),
        _dn(ft_y0 + top_via_enc - via_enc),
        _up(ft_x1 - top_via_enc + via_enc),
        _up(ft_y1 - top_via_enc + via_enc),
    )
    _rect(top_metal, top_cover)

    # === TERMINALS ==========================================================
    pad = _up(max(via_size + 2 * via_enc, bot_min_w, top_min_w, 400))

    p_cy = _dn((ft_y0 + ft_y1) // 2)
    virtual_left = _dn(ft_x0 - _GF180_MIM_VIRTUAL_OVERSIZE_NM)
    p_via_x1 = _dn(virtual_left - grid)
    p_via_x0 = _dn(p_via_x1 - via_size)
    p_via = (
        p_via_x0,
        _dn(p_cy - via_size // 2),
        p_via_x1,
        _dn(p_cy - via_size // 2) + via_size,
    )
    bot_x0 = min(ft_x0 - bot_ov, p_via[0] - bot_via_enc)
    bot_y0 = ft_y0 - bot_ov
    bot_x1 = ft_x1 + bot_ov
    bot_y1 = ft_y1 + bot_ov
    _rect(bot_metal, (bot_x0, bot_y0, bot_x1, bot_y1))
    _rect(top_via, p_via)

    p_y0 = _dn(p_cy - pad // 2)
    p_y1 = _up(p_y0 + pad)
    # Put the declared terminal on a cover-metal-only tip beyond the bottom-plate
    # edge so terminal extraction publishes top_metal and prevents forbidden
    # lower-via descent onto the MIM bottom plate (MIM.10 / MIMTM.10).
    p_tip_x1 = _dn(bot_x0)
    p_tip_x0 = _dn(p_tip_x1 - pad)
    p_top_x1 = _up(p_via[2] + via_enc)
    _rect(top_metal, (p_tip_x0, p_y0, p_top_x1, p_y1))
    p_pad = (p_tip_x0, p_y0, p_tip_x1, p_y1)

    detached_ext = _up(1060 + _GF180_MIM_BOTPLATE_TO_METAL_NM + 200)  # 1.06 + 1.2 + margin
    n_via_x0 = _up(max(ft_x1 + detached_ext, bot_x1 + _GF180_MIM_BOTPLATE_TO_METAL_NM + via_enc))
    n_cy = _dn((ft_y0 + ft_y1) // 2)
    n_via = (n_via_x0, _dn(n_cy - via_size // 2), n_via_x0 + via_size, _dn(n_cy - via_size // 2) + via_size)
    _rect(top_metal, (top_cover[2] - via_enc, _dn(n_cy - pad // 2), n_via[2] + via_enc, _up(n_cy - pad // 2 + pad)))
    _rect(top_via, n_via)
    n_pad_h = pad
    n_pad_w = _up(n_via[2] + via_enc) - _dn(n_via[0] - via_enc)
    if bot_min_area > 0 and n_pad_w * n_pad_h < bot_min_area:
        n_pad_h = _up((bot_min_area + n_pad_w - 1) // max(1, n_pad_w))
    n_pad = (_dn(n_via[0] - via_enc), _dn(n_cy - n_pad_h // 2),
             _up(n_via[2] + via_enc), _up(n_cy - n_pad_h // 2 + n_pad_h))
    _rect(bot_metal, n_pad)

    comp_bbox = (
        min(bot_x0, p_pad[0], ft_x0 - mk),
        min(bot_y0, ft_y0 - mk),
        max(bot_x1, n_pad[2], ft_x1 + mk),
        max(bot_y1, ft_y1 + mk),
    )

    # Record the actual unrotated unit before the artwork's affine transform.
    unit_extent = (min(_bbox(lp)[0] for lp in polys), min(_bbox(lp)[1] for lp in polys),
                   max(_bbox(lp)[2] for lp in polys), max(_bbox(lp)[3] for lp in polys))
    occurrences = CapacitorUnitOccurrenceGeometry.from_unit_extent(unit_extent)
    ndx, ndy = 0, 0
    # --- ORIENTATION (LVS-invariant affine rotation, same as build_resistor) --
    if orient not in _ORIENTS:
        raise ValueError(f"orient must be one of {_ORIENTS!r}, got {orient!r}")
    if orient != "R0":
        r0_min = (
            min(_bbox(lp)[0] for lp in polys),
            min(_bbox(lp)[1] for lp in polys),
        )
        polys = [_rotate_poly(lp, orient) for lp in polys]
        p_pad = _rotate_box(p_pad, orient)
        n_pad = _rotate_box(n_pad, orient)
        comp_bbox = _rotate_box(comp_bbox, orient)
        rot_min = (
            min(_bbox(lp)[0] for lp in polys),
            min(_bbox(lp)[1] for lp in polys),
        )
        ndx, ndy = r0_min[0] - rot_min[0], r0_min[1] - rot_min[1]
        gg = _device_grid_nm(tech)

        def _norm_snap(v: int, axis_shift: int) -> int:
            v += axis_shift
            return (v // gg) * gg if gg > 1 else v

        polys = [
            LayerPolygon(
                layer=lp.layer, datatype=lp.datatype,
                polygon=Polygon(
                    vertices=[
                        Point(_norm_snap(v.x, ndx), _norm_snap(v.y, ndy))
                        for v in lp.polygon.vertices
                    ]
                ),
            )
            for lp in polys
        ]

        def _shift_box(b: Bbox) -> Bbox:
            return (b[0] + ndx, b[1] + ndy, b[2] + ndx, b[3] + ndy)

        p_pad = _shift_box(p_pad)
        n_pad = _shift_box(n_pad)
        comp_bbox = _shift_box(comp_bbox)

    terminals: Dict[str, Terminal] = {
        "p": Terminal(role="p", bbox=p_pad, net=(nets or {}).get("p", "")),
        "n": Terminal(role="n", bbox=n_pad, net=(nets or {}).get("n", "")),
    }

    geom = DeviceGeom(
        polygons=polys,
        terminals=terminals,
        kind="cap",
        finger_count=1,
        gate_length_nm=side,
        channel_width_nm=side,
        comp_bbox=comp_bbox,
        orient=orient,
        source_rail_bbox=p_pad,
        drain_rail_bbox=n_pad,
    )
    out_geom = with_capacitor_occurrences(
        geom,
        occurrences.transformed(orient, (ndx, ndy), geom.extent()),
    )
    out_geom.__dict__["mim_option"] = norm_option
    out_geom.__dict__["mim_metal_level"] = norm_metal_level if norm_option == "B" else ""
    return out_geom


def _bank_tile_gap_nm(tech, unit: DeviceGeom) -> int:
    """The x-gap (nm) between adjacent UNIT cells in a generic parallel-MIM bank.

    Derived purely from PDK rule DATA -- the spacing a router would need between two
    independent cap cells -- so the bank is DRC-legal-by-construction regardless of
    PDK. The gap is the MAX of:

      * the MIM Metal2 keep-out (:func:`mim_metal2_keepout_nm`): one unit's bottom-
        plate / virtual-plate zone must stay clear of the next unit's metal (gf180
        MIM.1, net-agnostic). Zero for a PDK whose stack needs no such keep-out;
      * the routing-metal min-space over every metal a terminal lands on (so the two
        units never violate same-layer min-space where their pads / straps meet).

    No literal: every term is a resolved PDK rule, so a NEW PDK supplies only its
    rules (DATA) -- the tiling reads them, never hand-codes a per-PDK gap."""
    gap = int(mim_metal2_keepout_nm(tech) or 0)
    for lvl in range(1, 8):
        try:
            mk = tech.metal_key(lvl)
        except Exception:  # noqa: BLE001
            continue
        if mk is None:
            continue
        try:
            sp = int(tech.min_space(f"metal{lvl}") or 0)
        except Exception:  # noqa: BLE001
            sp = 0
        if sp > gap:
            gap = sp
    return max(gap, _grid_nm(tech))


def _build_parallel_mim_bank(
    tech,
    value_farads: float,
    *,
    nets=None,
    m: int,
    unit_builder,
    unit_kw: dict,
    bank_rows: int | None = None,
    unit_transforms: tuple[str, ...] | None = None,
    bank_align: str = "",
    bank_interconnect_plan=None,
) -> DeviceGeom:
    """Build an ``m``-unit parallel MIM BANK GENERICALLY from a PDK's UNIT cap cell.

    The generalization of a parallel cap bank: a parallel cap is ``m`` identical
    UNIT caps whose ``p`` plates share one net and ``n`` plates share the other. This
    function builds that bank for ANY PDK by:

      1. building ``m`` UNIT cells with the PDK's OWN single-unit cap builder
         (``unit_builder`` -- :func:`build_capacitor` on gf180, the same data-
         dispatched primitive for any PDK), so the unit stack (plates / vias /
         markers) is entirely the PDK's MIM primitive -- no per-PDK code here;
      2. tiling them in the declared rectangular lattice (or the legacy single
         COLUMN when ``bank_rows`` is absent), with both pitches derived from the
         unit extent and :func:`_bank_tile_gap_nm` PDK rule DATA;
      3. STRAPPING same-role terminals into one net: each unit's ``p`` / ``n``
         routable PAD (the unit's own exit pad, on its own metal at a cell EDGE)
         is joined to the next unit's by a VERTICAL metal rect spanning the pad
         x-band, on THAT PAD's layer. Because the pad sits at a cell edge with
         only its own net's metal in that x-band, the strap crosses no foreign net.
         The strap merges the units' same-role nodes into ONE node, so the bank
         extracts as ONE cap per unit on exactly TWO plate nets (a parallel bank).

    No layer numbers, no MIM-stack literals, no ``if <pdk>``: the unit geometry is
    the PDK primitive, the strap layer is the PAD's own drawn layer, and the gap is
    pure PDK rule DATA. A PDK with no tabulated MIM connect spec cannot have its
    bank VERIFIED as two nodes generically -> the honest gap is NAMED (raise),
    never a per-PDK patch."""
    from clay.layout.device_terminal import extract_device_nodes, extract_terminal_access
    from clay.layout.route.pins import _mim_spec_for

    row_bus_plan = None
    if bank_interconnect_plan is not None:
        from clay.layout.bank_interconnect import (
            replay_bank_interconnect_plan, select_existing_connected_landings,
            verify_bank_interconnect,
        )
        bank_interconnect_plan = replay_bank_interconnect_plan(bank_interconnect_plan)
        from clay.layout.mim_row_bus import MimRowBusPlan, materialize_mim_row_bus
        if type(bank_interconnect_plan) is MimRowBusPlan:
            row_bus_plan = bank_interconnect_plan
        if type(m) is not int or m <= 1:
            raise ValueError("bank interconnect requires exact multiplicity m > 1")

    if _is_gf180(tech):
        from clay.pdk.gf180mcu import resolve_gf180_mim_config

        mim_opt, _, m_lvl = resolve_gf180_mim_config(
            tech,
            mim_option=unit_kw.get("mim_option"),
            cap_option=unit_kw.get("cap_option"),
            metal_level=unit_kw.get("metal_level"),
            pex_backend=unit_kw.get("pex_backend"),
        )
        spec = _gf180_mim_connect_spec(tech, mim_option=mim_opt, metal_level=m_lvl)
    else:
        spec = _mim_spec_for(tech)
    if spec is None:
        raise NotImplementedError(
            f"{getattr(tech, 'pdk_id', '?')}: parallel MIM bank (m={m}) needs a MIM "
            f"connect spec to verify the strapped unit plates stay two nodes, but this "
            f"PDK has no tabulated MIM spec (clay.layout.route.pins._mim_spec_for) -- "
            f"honest gap (the PDK must supply its MIM connect-model DATA), not a per-PDK "
            f"builder"
        )

    if bank_rows is not None:
        if type(bank_rows) is not int:
            raise TypeError("parallel MIM bank_rows must be an exact integer")
        if bank_rows < 1 or bank_rows > m:
            raise ValueError("parallel MIM bank_rows must be within [1, m]")
        if m % bank_rows:
            raise ValueError("parallel MIM bank_rows must exactly divide m")
    if unit_transforms is not None:
        if type(unit_transforms) is not tuple:
            raise TypeError("parallel MIM unit_transforms must be an exact tuple")
        if len(unit_transforms) != m:
            raise ValueError("parallel MIM unit_transforms must have exactly m members")
        allowed_transforms = {
            "R0", "R90", "R180", "R270",
            "MR0", "MR90", "MR180", "MR270",
        }
        if any(type(item) is not str or item not in allowed_transforms
               for item in unit_transforms):
            raise ValueError("parallel MIM unit_transforms contains an invalid transform")
    if bank_align not in ("", "bottom", "center", "top"):
        raise ValueError("parallel MIM bank_align must be bottom, center, or top")

    grid = _grid_nm(tech)

    def _up(v: int) -> int:
        return ((int(v) + grid - 1) // grid) * grid

    def _dn(v: int) -> int:
        return (int(v) // grid) * grid

    # --- build the m UNIT cells (the PDK's own primitive), each at R0 ----------
    # The bank is rotated as a whole at the end, so units are built unrotated.
    bank_orient = unit_kw.pop("orient", "R0")
    if bank_interconnect_plan is not None and bank_orient != "R0":
        raise ValueError("existing-landing bank interconnect currently requires bank orientation R0")
    units = [
        unit_builder(tech, value_farads, nets=nets, orient="R0", **unit_kw)
        for _ in range(m)
    ]
    for index, unit in enumerate(units):
        if getattr(unit, "capacitor_unit_occurrences", None) is None:
            unit = with_capacitor_occurrences(unit, CapacitorUnitOccurrenceGeometry.from_unit_extent(unit.extent()))
            units[index] = unit
        facts = unit.capacitor_unit_occurrences
        if (type(facts) is not CapacitorUnitOccurrenceGeometry or unit.kind != "cap"
                or unit.orient != "R0" or facts.bank_orientation != "R0"
                or facts.unit_count != 1 or facts.bank_extent_nm != unit.extent()):
            raise ValueError("capacitor bank requires one exact native R0 unit occurrence per copy")
    center_bases = {unit.capacitor_unit_occurrences.center_basis for unit in units}
    if len(center_bases) != 1:
        raise ValueError("capacitor bank requires the same explicit unit center basis for every copy")
    transforms = unit_transforms or tuple("R0" for _ in range(m))
    units = [
        unit if transform == "R0" else _transform_devicegeom(unit, transform, tech)
        for unit, transform in zip(units, transforms)
    ]
    u0 = units[0]
    ext0 = u0.extent()
    gap = _bank_tile_gap_nm(tech, u0)
    rows = m if bank_rows is None else bank_rows
    columns = (m + rows - 1) // rows
    extents = [unit.extent() for unit in units]
    widths = [box[2] - box[0] for box in extents]
    heights = [box[3] - box[1] for box in extents]
    alignment = bank_align or "bottom"

    offsets: list[tuple[int, int]] = []
    placed_extents: list[Bbox] = []
    structural_members = unit_transforms is not None or bool(bank_align)
    if not structural_members:
        # Frozen legacy path: preserve the original identical-unit pitch and
        # origin arithmetic byte-for-byte.
        pitch_x = _up(widths[0] + gap)
        pitch_y = _up(heights[0] + gap)
        offsets = [
            ((index // rows) * pitch_x, (index % rows) * pitch_y)
            for index in range(m)
        ]
    else:
        column_widths = [
            max(widths[index] for index in range(column * rows, (column + 1) * rows))
            for column in range(columns)
        ]
        row_heights = [
            max(heights[index] for index in range(row, m, rows))
            for row in range(rows)
        ]
        column_lefts = [ext0[0]]
        for width in column_widths[:-1]:
            column_lefts.append(column_lefts[-1] + _up(width + gap))
        row_bottoms = [ext0[1]]
        for height in row_heights[:-1]:
            row_bottoms.append(row_bottoms[-1] + _up(height + gap))
        for index, box in enumerate(extents):
            column = index // rows
            row = index % rows
            extra_y = row_heights[row] - heights[index]
            if alignment == "bottom":
                align_y = 0
            elif alignment == "center":
                align_y = _dn(extra_y // 2)
            else:
                align_y = extra_y
            # A plate boundary may be off-grid even though routing metal is
            # on-grid. Aligning that boundary must not move all via landings
            # off the PDK grid. Round the rigid translation, not each shape.
            dx = _up(column_lefts[column] - box[0])
            dy = _up(row_bottoms[row] + align_y - box[1])
            offsets.append((dx, dy))
    placed_extents = [
        (box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy)
        for box, (dx, dy) in zip(extents, offsets)
    ]

    metal_level: Dict[tuple, int] = {}
    for lvl in range(1, 8):
        try:
            mk = tech.metal_key(lvl)
        except Exception:  # noqa: BLE001
            continue
        if mk is not None:
            metal_level.setdefault(tuple(mk), lvl)

    def _pad_layer(unit: DeviceGeom, role: str, nodes, node_id):
        """Lowest drawn metal wholly covering the pad on its original node.

        Stacked MIM plates overlap in XY. A foreign plate's polygon bbox must
        not choose a strap layer. Derived conductors retain their real-layer
        mapping; generated lower-metal exit pads keep their original layer.
        """
        import klayout.db as kdb

        t = unit.terminals.get(role)
        if t is None or node_id is None:
            return None
        query = kdb.Region(kdb.Box(*t.bbox))
        if query.is_empty():
            return None
        for key in metal_level:
            drawn = kdb.Region()
            for lp in unit.polygons:
                if (lp.layer, lp.datatype) == key:
                    drawn.insert(kdb.Polygon([kdb.Point(v.x, v.y) for v in lp.polygon.vertices]))
            masks = []
            for node in nodes:
                mask = kdb.Region()
                for layer, region in node.items():
                    if spec.synthetic_to_real.get(layer, layer) == key:
                        mask += region
                masks.append(mask)
            if not (query - drawn).is_empty() or not (query - masks[node_id]).is_empty():
                continue
            if all(index == node_id or (query & mask).is_empty() for index, mask in enumerate(masks)):
                return key
        return None

    polys: list[LayerPolygon] = []
    # Per-role list of (pad_layer, pad_bbox_nm, column, row) after translation.
    pads: Dict[str, list] = {}
    # Keep original M1 access independent of explicit internal landing selection.
    external_pads: Dict[str, list] = {}
    planned_role_pads = []

    for i, u in enumerate(units):
        column = i // rows
        row = i % rows
        dx, dy = offsets[i]
        for lp in u.polygons:
            polys.append(
                LayerPolygon(
                    layer=lp.layer, datatype=lp.datatype,
                    polygon=Polygon(
                        vertices=[
                            Point(v.x + dx, v.y + dy)
                            for v in lp.polygon.vertices
                        ]
                    ),
                )
            )
        acc, rep = extract_terminal_access(u, tech=tech, mim_spec=spec)
        unit_nodes = extract_device_nodes(u, tech=tech, mim_spec=spec)
        if rep.collapsed:
            raise ValueError(
                f"{getattr(tech, 'pdk_id', '?')}: bank unit {i} has collapsed "
                f"plates {rep.collapsed} -- the unit cap is not a 2-node device, "
                f"cannot strap into a parallel bank"
            )
        landings = (select_existing_connected_landings(
            u, tech=tech, mim_spec=spec, plan=bank_interconnect_plan
        ) if bank_interconnect_plan is not None and row_bus_plan is None else None)
        for role in ("p", "n"):
            t = u.terminals.get(role)
            lay = _pad_layer(u, role, unit_nodes, acc[role].node_id if role in acc else None)
            if t is None or lay is None:
                raise ValueError(f"parallel MIM unit {i} role {role} has no complete original-node metal pad")
            bb = (
                t.bbox[0] + dx,
                t.bbox[1] + dy,
                t.bbox[2] + dx,
                t.bbox[3] + dy,
            )
            external_pads.setdefault(role, []).append((lay, bb, column, row))
            if row_bus_plan is not None:
                planned_role_pads.append({"unit": i, "role": role, "net": t.net,
                    "layer": lay, "bbox_nm": bb, "row": row, "column": column})
            if landings is not None:
                landing = landings[role]
                def translated(box):
                    return (box[0]+dx, box[1]+dy, box[2]+dx, box[3]+dy)
                planned_role_pads.append({
                    "unit": i, "role": role, "net": t.net,
                    "original_m1_bbox_nm": bb,
                    "bbox_nm": translated(landing["bbox_nm"]),
                    "existing_via_cuts_nm": tuple(translated(cut) for cut in landing["existing_via_cuts_nm"]),
                })
                lay, bb = landing["layer"], translated(landing["bbox_nm"])
            pads.setdefault(role, []).append((lay, bb, column, row))

    unit_boxes = [_bbox(lp) for lp in polys]
    unit_y0 = min(box[1] for box in unit_boxes)
    unit_y1 = max(box[3] for box in unit_boxes)
    if row_bus_plan is not None:
        added, row_bus_proof = materialize_mim_row_bus(polys, role_pads=planned_role_pads,
            unit_count=m, tech=tech, mim_spec=spec, plan=row_bus_plan)
        polys.extend(added)

    # Strap same-role pads into one net.  The legacy one-column path emits the
    # original single vertical rectangle byte-for-byte.  A rectangular lattice
    # first joins each column vertically in the pad's own x-band, then joins those
    # column trunks on an exterior horizontal bus: p below the array and n above.
    # Both bus offset and unit pitch consume PDK spacing authority; bus width comes
    # from the already-materialized terminal pad, never a circuit/PDK literal.
    for role, lst in (() if row_bus_plan is not None else pads.items()):
        if len(lst) < 2:
            continue
        layers = {tuple(item[0]) for item in lst}
        if len(layers) != 1:
            raise ValueError(
                f"parallel MIM {role} pads do not share one terminal layer"
            )
        lay = lst[0][0]
        if bank_rows is None or rows == m:
            bx0 = min(b[0] for _, b, _column, _row in lst)
            bx1 = max(b[2] for _, b, _column, _row in lst)
            by0 = min(b[1] for _, b, _column, _row in lst)
            by1 = max(b[3] for _, b, _column, _row in lst)
            polys.append(
                LayerPolygon(
                    layer=lay[0], datatype=lay[1],
                    polygon=Polygon.rectangle(
                        _dn(bx0), _dn(by0), _up(bx1), _up(by1)
                    ),
                )
            )
            continue

        by_column: Dict[int, list] = {}
        for _item_layer, box, column, _row in lst:
            by_column.setdefault(column, []).append(box)
        trunks = []
        for column in sorted(by_column):
            boxes = by_column[column]
            trunk = (
                min(box[0] for box in boxes),
                min(box[1] for box in boxes),
                max(box[2] for box in boxes),
                max(box[3] for box in boxes),
            )
            trunks.append(trunk)

        bus_height = _up(max(box[3] - box[1] for _lay, box, _c, _r in lst))
        if role == "p":
            bus_y1 = _dn(unit_y0 - gap)
            bus_y0 = _dn(bus_y1 - bus_height)
        else:
            bus_y0 = _up(unit_y1 + gap)
            bus_y1 = _up(bus_y0 + bus_height)
        bus_x0 = min(trunk[0] for trunk in trunks)
        bus_x1 = max(trunk[2] for trunk in trunks)
        polys.append(
            LayerPolygon(
                layer=lay[0], datatype=lay[1],
                polygon=Polygon.rectangle(
                    _dn(bus_x0), _dn(bus_y0), _up(bus_x1), _up(bus_y1)
                ),
            )
        )
        for trunk in trunks:
            polys.append(
                LayerPolygon(
                    layer=lay[0], datatype=lay[1],
                    polygon=Polygon.rectangle(
                        _dn(trunk[0]),
                        _dn(min(trunk[1], bus_y0)),
                        _up(trunk[2]),
                        _up(max(trunk[3], bus_y1)),
                    ),
                )
            )

    # --- the bank's two terminals: each role's PAD on the first (bottom) unit ---
    # (now part of the commoned strap net). The router lands on the bank net there.
    terminals: Dict[str, Terminal] = {}
    for role in ("p", "n"):
        t0 = u0.terminals.get(role)
        if t0 is None:
            continue
        first_pad = t0.bbox if not structural_members else (
            external_pads.get(role, [])[0][1] if external_pads.get(role) else (
            t0.bbox[0] + offsets[0][0],
            t0.bbox[1] + offsets[0][1],
            t0.bbox[2] + offsets[0][0],
            t0.bbox[3] + offsets[0][1],
            )
        )
        terminals[role] = Terminal(role=role, bbox=first_pad, net=t0.net)

    boxes = [_bbox(lp) for lp in polys]
    comp_bbox = (
        min(b[0] for b in boxes), min(b[1] for b in boxes),
        max(b[2] for b in boxes), max(b[3] for b in boxes),
    )

    p_pad = terminals["p"].bbox if "p" in terminals else u0.source_rail_bbox
    n_pad = terminals["n"].bbox if "n" in terminals else u0.drain_rail_bbox

    bank = DeviceGeom(
        polygons=polys,
        terminals=terminals,
        kind="cap",
        finger_count=m,
        gate_length_nm=u0.gate_length_nm,
        channel_width_nm=u0.channel_width_nm,
        comp_bbox=comp_bbox,
        orient="R0",
        source_rail_bbox=p_pad,
        drain_rail_bbox=n_pad,
    )
    bank.__dict__["parallel_mim_bank"] = {
        "unit_count": m,
        "rows": rows,
        "columns": columns,
        "r0_replica_offsets_nm": tuple(offsets),
        "bank_orientation": bank_orient,
    }
    bank = with_capacitor_occurrences(bank, CapacitorUnitOccurrenceGeometry(
        unit_count=m, bank_extent_nm=bank.extent(), bank_orientation="R0",
        center_basis=units[0].capacitor_unit_occurrences.center_basis,
        unit_centers_nm=tuple((x+dx, y+dy) for unit, (dx, dy) in zip(units, offsets)
                             for x, y in unit.capacitor_unit_occurrences.unit_centers_nm),
        unit_transforms=tuple(transform for unit in units
                              for transform in unit.capacitor_unit_occurrences.unit_transforms)))
    if unit_transforms is not None or bank_align:
        bank.__dict__["parallel_mim_bank"].update({
            "unit_transforms": tuple(transforms),
            "alignment": alignment,
            "unit_offsets_nm": tuple(offsets),
            "unit_extents_nm": tuple(placed_extents),
        })

    if bank_rows is not None and rows != m:
        accesses, report = extract_terminal_access(bank, tech=tech, mim_spec=spec)
        if (
            report.collapsed
            or report.node_count != 2
            or set(accesses) != {"p", "n"}
        ):
            raise ValueError(
                "rectangular parallel MIM bank does not replay exactly two "
                f"terminal nodes: nodes={report.node_count}, "
                f"collapsed={report.collapsed!r}, roles={sorted(accesses)!r}"
            )

    if row_bus_plan is not None:
        bank.__dict__["bank_interconnect"] = row_bus_proof
    elif bank_interconnect_plan is not None:
        bank.__dict__["bank_interconnect"] = verify_bank_interconnect(
            bank, tech=tech, mim_spec=spec, plan=bank_interconnect_plan,
            role_pads=planned_role_pads,
        )

    if bank_orient == "R0":
        return bank
    # rotate the whole bank as one rigid body (same idiom as build_capacitor).
    oriented = _orient_devicegeom(bank, bank_orient, tech)
    oriented.__dict__["parallel_mim_bank"] = dict(
        bank.__dict__["parallel_mim_bank"]
    )
    return oriented


def _transform_devicegeom(geom: DeviceGeom, transform: str, tech) -> DeviceGeom:
    """Apply one coordinate-free orthogonal R/MR transform to ``DeviceGeom``."""

    transforms = (
        "R0", "R90", "R180", "R270",
        "MR0", "MR90", "MR180", "MR270",
    )
    if transform not in transforms:
        raise ValueError(
            f"transform must be one of {transforms!r}, got {transform!r}"
        )
    if transform == "R0":
        return geom
    if not transform.startswith("MR"):
        return _orient_devicegeom(geom, transform, tech)
    if str(getattr(geom, "orient", "R0") or "R0") != "R0":
        raise ValueError("device transform requires one R0 input geometry")

    mirrored = transform.startswith("MR")
    angle = transform[2:] if mirrored else transform[1:]
    rotation = "R" + angle

    def _point(x: int, y: int) -> Tuple[int, int]:
        if mirrored:
            y = -y
        return _rotate_point(x, y, rotation)

    def _box(box: Optional[Bbox]) -> Optional[Bbox]:
        if box is None:
            return None
        x0, y0, x1, y1 = box
        points = tuple(
            _point(x, y)
            for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
        )
        return (
            min(point[0] for point in points),
            min(point[1] for point in points),
            max(point[0] for point in points),
            max(point[1] for point in points),
        )

    polys = list(geom.polygons)
    r0_min = (min(_bbox(lp)[0] for lp in polys), min(_bbox(lp)[1] for lp in polys))
    transformed_polys = []
    for polygon in polys:
        points = [_point(vertex.x, vertex.y) for vertex in polygon.polygon.vertices]
        if mirrored:
            points.reverse()
        transformed_polys.append(
            LayerPolygon(
                layer=polygon.layer,
                datatype=polygon.datatype,
                polygon=Polygon(
                    vertices=[Point(x, y) for x, y in points]
                ),
            )
        )
    polys = transformed_polys
    p_pad = _box(geom.source_rail_bbox)
    n_pad = _box(geom.drain_rail_bbox)
    comp_bbox = _box(geom.comp_bbox)
    rot_min = (min(_bbox(lp)[0] for lp in polys), min(_bbox(lp)[1] for lp in polys))
    ndx, ndy = r0_min[0] - rot_min[0], r0_min[1] - rot_min[1]
    gg = _device_grid_nm(tech)

    def _norm(v: int, shift: int) -> int:
        v += shift
        return (v // gg) * gg if gg > 1 else v

    polys = [
        LayerPolygon(
            layer=lp.layer, datatype=lp.datatype,
            polygon=Polygon(
                vertices=[Point(_norm(v.x, ndx), _norm(v.y, ndy))
                          for v in lp.polygon.vertices]
            ),
        )
        for lp in polys
    ]

    def _shift(b):
        # Apply the same translation and grid map as the metal vertices.
        # Otherwise rotated tips can protrude beyond their electrical node.
        return (_norm(b[0], ndx), _norm(b[1], ndy),
                _norm(b[2], ndx), _norm(b[3], ndy)) if b else b

    p_pad, n_pad, comp_bbox = _shift(p_pad), _shift(n_pad), _shift(comp_bbox)
    terminals = {
        role: Terminal(role=role, bbox=_shift(_box(t.bbox)), net=t.net)
        for role, t in (geom.terminals or {}).items()
    }
    from dataclasses import fields, replace
    transformed = replace(
        geom,
        polygons=polys,
        terminals=terminals,
        comp_bbox=comp_bbox,
        orient=transform,
        source_rail_bbox=p_pad,
        drain_rail_bbox=n_pad,
        gate_bus_bbox=_shift(_box(geom.gate_bus_bbox)),
        gate_contact_bbox=_shift(_box(geom.gate_contact_bbox)),
        nwell_bbox=_shift(_box(geom.nwell_bbox)),
        placement_extent=_shift(_box(geom.placement_extent)),
    )
    if geom.capacitor_unit_occurrences is not None:
        transformed = with_capacitor_occurrences(transformed, geom.capacitor_unit_occurrences.transformed(
            transform, (ndx, ndy), transformed.extent()))
    declared_fields = {field.name for field in fields(geom)}
    for key, value in geom.__dict__.items():
        if key in declared_fields:
            continue
        if key == "terminal_accesses" and type(value) is dict:
            transformed.__dict__[key] = {
                role: tuple(
                    Terminal(
                        role=terminal.role,
                        bbox=_shift(_box(terminal.bbox)),
                        net=terminal.net,
                    )
                    for terminal in accesses
                )
                for role, accesses in value.items()
            }
        else:
            transformed.__dict__[key] = value
    return transformed


def _orient_devicegeom(geom: DeviceGeom, orient: str, tech) -> DeviceGeom:
    """Rotate a whole :class:`DeviceGeom` rigidly by ``orient``.

    This is the frozen legacy R-only path.  Mirror support is deliberately kept
    in :func:`_transform_devicegeom` so existing generated banks remain byte-for-
    byte stable.
    """
    if orient not in _ORIENTS:
        raise ValueError(f"orient must be one of {_ORIENTS!r}, got {orient!r}")
    if orient == "R0":
        return geom
    polys = list(geom.polygons)
    r0_min = (min(_bbox(lp)[0] for lp in polys), min(_bbox(lp)[1] for lp in polys))
    polys = [_rotate_poly(lp, orient) for lp in polys]
    p_pad = _rotate_box(geom.source_rail_bbox, orient) if geom.source_rail_bbox else None
    n_pad = _rotate_box(geom.drain_rail_bbox, orient) if geom.drain_rail_bbox else None
    comp_bbox = _rotate_box(geom.comp_bbox, orient)
    rot_min = (min(_bbox(lp)[0] for lp in polys), min(_bbox(lp)[1] for lp in polys))
    ndx, ndy = r0_min[0] - rot_min[0], r0_min[1] - rot_min[1]
    gg = _device_grid_nm(tech)

    def _norm(v: int, shift: int) -> int:
        v += shift
        return (v // gg) * gg if gg > 1 else v

    polys = [
        LayerPolygon(
            layer=lp.layer, datatype=lp.datatype,
            polygon=Polygon(
                vertices=[Point(_norm(v.x, ndx), _norm(v.y, ndy))
                          for v in lp.polygon.vertices]
            ),
        )
        for lp in polys
    ]

    def _shift(b):
        # Terminal access and physical extents must follow serialized metal.
        return (_norm(b[0], ndx), _norm(b[1], ndy),
                _norm(b[2], ndx), _norm(b[3], ndy)) if b else b

    p_pad, n_pad, comp_bbox = _shift(p_pad), _shift(n_pad), _shift(comp_bbox)
    terminals = {
        role: Terminal(role=role, bbox=_shift(_rotate_box(t.bbox, orient)), net=t.net)
        for role, t in (geom.terminals or {}).items()
    }
    from dataclasses import fields, replace
    transformed = replace(
        geom, polygons=polys, terminals=terminals, comp_bbox=comp_bbox,
        orient=orient, source_rail_bbox=p_pad, drain_rail_bbox=n_pad,
        gate_bus_bbox=_shift(
            _rotate_box(geom.gate_bus_bbox, orient)
            if geom.gate_bus_bbox is not None else None
        ),
        gate_contact_bbox=_shift(
            _rotate_box(geom.gate_contact_bbox, orient)
            if geom.gate_contact_bbox is not None else None
        ),
        nwell_bbox=_shift(
            _rotate_box(geom.nwell_bbox, orient)
            if geom.nwell_bbox is not None else None
        ),
        placement_extent=_shift(
            _rotate_box(geom.placement_extent, orient)
            if geom.placement_extent is not None else None
        ),
    )
    if geom.capacitor_unit_occurrences is not None:
        transformed = with_capacitor_occurrences(transformed, geom.capacitor_unit_occurrences.transformed(
            orient, (ndx, ndy), transformed.extent()))
    declared_fields = {field.name for field in fields(geom)}
    for key, value in geom.__dict__.items():
        if key in declared_fields:
            continue
        if key == "terminal_accesses" and type(value) is dict:
            transformed.__dict__[key] = {
                role: tuple(
                    Terminal(
                        role=terminal.role,
                        bbox=_shift(_rotate_box(terminal.bbox, orient)),
                        net=terminal.net,
                    )
                    for terminal in accesses
                )
                for role, accesses in value.items()
            }
        else:
            transformed.__dict__[key] = value
    return transformed


def build_segmented_resistor(
    tech,
    value_ohms: float,
    *,
    count: int,
    connection: str,
    segment_width_um: float,
    segment_length_um: float,
    segment_resistance_ohms: float,
    value_quantum_ohms: float,
    model: str,
    segment_resistance_quantum_ohms: float = 0.0,
    orient: str = "R0",
    nets: Optional[Dict[str, str]] = None,
) -> DeviceGeom:
    """Build a declared, PDK-bound straight-segment resistor topology.

    The unannotated :func:`build_resistor` path is intentionally untouched.  A
    caller reaches this constructor only after consuming a complete typed
    declaration and proving its equivalent resistance.
    """

    from .segmented_resistor import build_smic_segmented_resistor_r0

    transforms = (
        "R0", "R90", "R180", "R270",
        "MR0", "MR90", "MR180", "MR270",
    )
    if orient not in transforms:
        raise ValueError(f"orient must be one of {transforms!r}, got {orient!r}")
    geom = build_smic_segmented_resistor_r0(
        tech,
        value_ohms,
        count=count,
        connection=connection,
        segment_width_um=segment_width_um,
        segment_length_um=segment_length_um,
        segment_resistance_ohms=segment_resistance_ohms,
        value_quantum_ohms=value_quantum_ohms,
        model=model,
        segment_resistance_quantum_ohms=segment_resistance_quantum_ohms,
        nets=nets,
    )
    # Preserve the frozen R-only byte path.  Mirror transforms use the same
    # orthogonal DeviceGeom transform already consumed by fixed MIM banks, which
    # moves terminal access with the artwork without rebinding its nets.
    return (
        _orient_devicegeom(geom, orient, tech)
        if orient in _ORIENTS
        else _transform_devicegeom(geom, orient, tech)
    )








def build_capacitor_for_tech(
    tech, value_farads: float, *, nets=None, m: int = 1, model: str = "", **kw
):
    """Build a MIM cap cell for ``tech``, dispatching to the per-PDK UNIT builder by
    ``pdk_id`` (the adapter pattern -- DATA dispatch, not a per-circuit branch).
    gf180mcu uses this module's :func:`build_capacitor` (FuseTop/Metal2 stack);
    open_pdk uses :func:`clay.layout.passive_device_cell_tsmc.build_mim_capacitor_tsmc`
    (CTM/Metal stack). A PDK with no MIM builder raises (the caller reports an honest
    gap).

    ``value_farads`` is PER-UNIT; ``m`` is the parallel-unit bank count and
    ``model`` the netlist ``$.MODEL`` annotation (both straight off
    ``ExtractedPassive`` -- the signed reference's ``CC1 830.76f m=2 $.MODEL=M5``
    form). On external_pdk the model annotation selects the CTM level unless an explicit
    ``level=`` kw (e.g. the stamp-detected level) overrides it.

    Parallel bank (``m > 1``)
    =========================
    external_pdk's UNIT builder natively tiles ``m`` CTM plates over one shared bottom plate
    (its primitive supports banking), so the external_pdk bank is delegated unchanged. For
    EVERY OTHER PDK the bank is built GENERICALLY by :func:`_build_parallel_mim_bank`:
    ``m`` copies of the PDK's OWN single-unit cap cell, tiled by a PDK-rule-derived
    gap and strapped same-role plate->plate on each terminal's connect-model landing
    layer. No per-PDK bank builder, no ``if gf180`` -- the unit cell is the PDK MIM
    primitive and the strap geometry is pure PDK DATA, so a NEW PDK with a MIM
    primitive + connect spec gets a parallel bank with zero new algorithm code."""
    pdk = str(getattr(tech, "pdk_id", "")).lower()
    if kw.get("bank_interconnect_plan") is not None and (type(m) is not int or m <= 1):
        raise ValueError("bank interconnect requires exact integer multiplicity m > 1")
    m = max(1, int(m))
    bank_rows = kw.pop("bank_rows", None)
    unit_transforms = kw.pop("unit_transforms", None)
    bank_align = kw.pop("bank_align", "")
    bank_interconnect_plan = kw.pop("bank_interconnect_plan", None)
    if bank_interconnect_plan is not None:
        from clay.layout.bank_interconnect import replay_bank_interconnect_plan
        bank_interconnect_plan = replay_bank_interconnect_plan(bank_interconnect_plan)
        if m <= 1:
            raise ValueError("bank interconnect requires m > 1")
    if bank_rows is not None:
        if type(bank_rows) is not int:
            raise TypeError("parallel MIM bank_rows must be an exact integer")
        if m <= 1:
            raise ValueError("parallel MIM bank_rows requires m > 1")
        if bank_rows < 1 or bank_rows > m:
            raise ValueError("parallel MIM bank_rows must be within [1, m]")
        if m % bank_rows:
            raise ValueError("parallel MIM bank_rows must exactly divide m")
    lv = kw.pop("level", None)
    if lv is not None:
        raise ValueError(
            f"{pdk}: 'level' is a external_pdk CTM-stack parameter; this PDK's cap builder "
            f"has no level concept (got level={lv!r})"
        )
    if not _is_gf180(tech):
        mim_opt = kw.pop("mim_option", None)
        kw.pop("metal_level", None)
        kw.pop("pex_backend", None)
        if mim_opt is not None and str(mim_opt).strip().upper() not in ("A", "OPT_A", "OPTION_A"):
            raise ValueError(
                f"{pdk}: mim_option={mim_opt!r} is only defined for gf180mcu"
            )
    # The PDK's OWN single-unit cap primitive. gf180 uses this module's
    # build_capacitor (Option-A M2/M3 or Option-B M4/M5); external_pdk 0.18 MIM10 PDKs
    # dispatch to the external_pdk MIM primitive wrapped as a DeviceGeom. DATA
    # dispatch on the PDK id -- the bank tiler stays generic (it composes whichever
    # unit primitive it is handed; no per-PDK bank code).
    unit_builder = _build_capacitor
    if m > 1:
        bank = _build_parallel_mim_bank(
            tech, value_farads, nets=nets, m=m,
            unit_builder=unit_builder, unit_kw=dict(kw), bank_rows=bank_rows,
            unit_transforms=unit_transforms, bank_align=bank_align,
            bank_interconnect_plan=bank_interconnect_plan,
        )
        if _is_gf180(tech):
            from clay.pdk.gf180mcu import resolve_gf180_mim_config

            mim_opt, _, m_lvl = resolve_gf180_mim_config(
                tech,
                mim_option=kw.get("mim_option"),
                cap_option=kw.get("cap_option"),
                metal_level=kw.get("metal_level"),
                pex_backend=kw.get("pex_backend"),
            )
            bank.__dict__["mim_option"] = mim_opt
            bank.__dict__["mim_metal_level"] = m_lvl if mim_opt == "B" else ""
        return bank
    return unit_builder(tech, value_farads, nets=nets, **kw)


def mim_keepout_nm(
    tech,
    *,
    mim_option: Optional[str] = None,
    pex_backend: Optional[str] = None,
) -> int:
    """Bottom-plate routing keep-out (nm) around GF180 FuseTop for Option-A/B."""
    if not _is_gf180(tech):
        return 0
    from clay.pdk.gf180mcu import resolve_gf180_mim_config

    resolve_gf180_mim_config(tech, mim_option=mim_option, pex_backend=pex_backend)
    return GF180_MIM_METAL2_KEEPOUT_NM


#: The Metal2 KEEP-OUT a router must hold around a gf180 MIM cap's FuseTop, in nm:
#: the FuseTop is oversized by 1.06um to the ``mim_virtual`` bottom plate, and
#: MIM.1 forbids any *other* Metal2 within 1.2um of that virtual plate. So any
#: routing Metal2 (other than the bottom-plate handle itself) must stay
#: ``1.06 + 1.2 = 2.26um`` clear of the FuseTop edge. The router grows each placed
#: FuseTop polygon by this margin to form a net-agnostic Metal2 obstacle (MIM.1 is
#: net-agnostic), so no trunk / jog / via pad is ever placed inside the keep-out.
_GF180_MIM_VIRTUAL_OVERSIZE_NM = 1060   # FuseTop.sized(1.06um) -> mim_virtual
GF180_MIM_METAL2_KEEPOUT_NM = (
    _GF180_MIM_VIRTUAL_OVERSIZE_NM + _GF180_MIM_BOTPLATE_TO_METAL_NM
)  # 1060 + 1200 = 2260 nm


def mim_metal2_keepout_nm(tech) -> int:
    """The Metal2 keep-out margin (nm) a router holds around a MIM cap's FuseTop on
    ``tech`` -- the ``mim_virtual`` oversize plus the MIM.1 bottom-plate-to-metal
    spacing. gf180mcu only (the MIM stack is gf180-specific); returns 0 for a PDK
    with no MIM stack so the router adds no keep-out there (byte-unchanged)."""
    return GF180_MIM_METAL2_KEEPOUT_NM if _is_gf180(tech) else 0


#: Frozen backward-compat alias of the PRIVATE per-PDK primitive ``_build_capacitor``.
#: Deliberately NOT in ``__all__`` (so ``import *`` cannot pull the bare gf180-only
#: builder) and fenced from NEW production importers by the import-boundary guard in
#: ``tests/test_encapsulation_boundary.py``. Production MUST build caps through the
#: per-PDK dispatcher :func:`build_capacitor_for_tech`; this name survives only for the
#: gf180-specific unit tests and the per-PDK-dispatch obligation checker.
build_capacitor = _build_capacitor


__all__ = [
    "build_resistor",
    "resistor_body_length_nm",
    "materialized_resistor_body_length_nm",
    "build_capacitor_for_tech",
    "mim_density_f_per_um2",
    "mim_model_name",
    "mim_plate_side_nm",
    "equal_size_cap_plate_side_nm",
    "mim_metal2_keepout_nm",
    "mim_keepout_nm",
    "GF180_MIM_METAL2_KEEPOUT_NM",
]
