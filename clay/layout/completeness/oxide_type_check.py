"""OXIDE-TYPE (thick-ox / OD2) obligation -- CONDITIONAL, 5-state, GDS-backed.

WHAT THIS NAMES (the false-green this closes)
=============================================
A open_pdk 3.3V device is a THICK-gate-oxide MOS: the foundry marks it by drawing
the OD2 layer (GDS ``4/0``) over its active/gate, and the sign-off LVS deck splits
the gate on OD2 (``HVGT = ALL_GATE AND OD2`` for 3.3V vs ``LVGT = ALL_GATE NOT OD2``
for 1.8V). If a design is BUILT with 1.8V core transistors while its schematic /
reference is a 3.3V THICK-oxide design, the wrong device is emitted -- and because
the generated deck was OD2-blind, the self-consistent LVS never saw the difference
(a FALSE GREEN). This obligation makes that fact MEASURABLE on the emitted GDS,
independently of the router/placer/deck, so a wrong-oxide layout reports RED.

CONDITIONAL (never an unconditional "every device needs OD2")
=============================================================
The obligation is keyed to the DECLARED oxide of the design's devices:

  * declared THICK  -- every transistor channel/OD MUST be enclosed by OD2 by at
    least the OD2.E.1 enclosure (>= 0.32um over OD). A thick design with NO OD2
    (our OD2-blind OTA) is RED.
  * declared THIN   -- NO transistor gate may be covered by OD2. A correct 1.8V
    design (the open_pdk comparator) carries NO OD2 and PASSES -- it is NOT
    falsely RED'd. (An unconditional "every device needs OD2" check would wrongly
    fail this correct 1.8V layout; that is exactly the anti-pattern this avoids.)

FIVE-STATE VERDICT (never a silent pass)
========================================
    pass         -- oracle read the geometry; the conditional obligation holds.
    fail         -- oracle read the geometry; the obligation is violated (RED,
                    blocking like DRC/LVS -- a wrong-oxide layout is not green).
    unmeasurable -- no transistor channel (poly & active) was found to judge, or
                    no oxide declaration was supplied. NEVER a silent pass.
    unsupported  -- klayout is unavailable (capability absent). NEVER a silent pass.
    advisory     -- reserved (this quantitative gate does not emit it).

INDEPENDENCE
============
The measurement is pure ``klayout.db`` boolean geometry on the emitted artifact
(gate = poly & active; OD2 enclosure via inward-sized OD2). It reads ONLY the GDS
-- never the deck it will be LVS'd with, never the placer/router state -- so it is
an independent oracle: if the OD2-aware deck and this checker ever disagree about a
layout, that is a bug, not something either side papers over.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

# Reuse the canonical 5-state set (single source; not edited here).
from clay.layout.spec.verdict import FIVE_STATES

LayerKey = Tuple[int, int]

# open_pdk sign-off layer numbers (foundry external_drc LVS deck LAYER block):
#   OD (active/diffusion) = 3/0, POLY (gate) = 13/0, OD2 (thick-gate-oxide) = 4/0.
# Defaults so a caller measuring a open_pdk layout need not restate them; a
# different PDK passes its own keys (or a ``tech`` that resolves them).
_TSMC180RF_DIFF: LayerKey = (3, 0)
_TSMC180RF_POLY: LayerKey = (13, 0)
_TSMC180RF_OD2: LayerKey = (4, 0)

# OD2.E.1: OD2 must extend past the OD (active) it marks by >= 0.32um. The signed
# external_pdk reference layout satisfies exactly this 0.32um OD-enclosure over every
# 3.3V channel (measured); a stricter value would false-RED that correct reference.
_OD2_ENCLOSE_OD_UM = 0.32


@dataclass(frozen=True)
class OxideTypeVerdict:
    """The independent oracle's 5-state verdict for the oxide-type obligation.

    ``state`` is one of :data:`FIVE_STATES`. ``violating_area_um2`` is the area of
    the obligation breach (for THICK: channel/OD area NOT enclosed by OD2; for
    THIN: gate area wrongly under OD2); 0.0 on a pass. The remaining fields are the
    measured context so a caller can report an honest, auditable number.
    """

    state: str
    declared_oxide: str
    n_gates: int
    gate_area_um2: float
    violating_area_um2: float
    od2_area_um2: float
    detail: str

    def __post_init__(self) -> None:
        if self.state not in FIVE_STATES:
            raise AssertionError(f"non-5-state oxide verdict: {self.state!r}")


def _resolve_keys(
    tech, diff_layer, poly_layer, od2_layer
) -> Tuple[LayerKey, LayerKey, Optional[LayerKey]]:
    """Resolve (diff, poly, od2) GDS keys.

    Explicit ``*_layer`` args win. When a ``tech`` (LayoutTech) is given and an arg
    is left None, resolve from it (``build_layer('diff'/'poly')`` + the OD2 marker
    token). OD2 may resolve to None for a PDK that does not declare it -- an all-thin
    design then measures cleanly (no OD2 anywhere), while a thick design without an
    OD2 layer is correctly RED (its channels cannot be OD2-enclosed)."""
    diff = diff_layer
    poly = poly_layer
    od2 = od2_layer
    if tech is not None:
        if diff is None:
            diff = tech.build_layer("diff")
        if poly is None:
            poly = tech.build_layer("poly")
        if od2 is None:
            # OD2 is a non-routing MARKER layer (resolved by rule-container token).
            od2 = tech.marker_key("od2")
    if diff is None:
        diff = _TSMC180RF_DIFF
    if poly is None:
        poly = _TSMC180RF_POLY
    if od2 is None and od2_layer is None and tech is None:
        od2 = _TSMC180RF_OD2
    return diff, poly, od2


def check_oxide_type(
    gds_path: str,
    *,
    declared_oxide: str,
    tech=None,
    diff_layer: Optional[LayerKey] = None,
    poly_layer: Optional[LayerKey] = None,
    od2_layer: Optional[LayerKey] = None,
    od2_enclose_od_um: float = _OD2_ENCLOSE_OD_UM,
) -> OxideTypeVerdict:
    """Measure the oxide-type obligation on the emitted ``gds_path``.

    Parameters
    ----------
    declared_oxide : {'thick','thin'}
        The oxide the design's transistors are DECLARED to be (from the schematic /
        reference model kind: PD/ND/pmos3v/nmos3v -> 'thick'; pch/nch/plain
        pmos/nmos -> 'thin'). This is what makes the check CONDITIONAL -- a thin
        design owes NO OD2, a thick one owes OD2 over every channel.
    tech : LayoutTech, optional
        When given, resolves the diff/poly/OD2 layer keys from the PDK (any arg left
        None is resolved from it). Otherwise the open_pdk defaults (3/0, 13/0, 4/0)
        are used unless overridden by the explicit ``*_layer`` args.
    od2_enclose_od_um : float
        The OD2.E.1 enclosure (default 0.32um) a thick device's OD owes.

    Returns
    -------
    OxideTypeVerdict
        5-state verdict + the measured areas. THICK fails when any channel OD is not
        OD2-enclosed; THIN fails when any gate is under OD2. ``unmeasurable`` when no
        channel is found or no oxide is declared; ``unsupported`` when klayout is
        absent -- never a silent pass.
    """
    ox = str(declared_oxide or "").strip().lower()
    if ox not in ("thick", "thin"):
        return OxideTypeVerdict(
            state="unmeasurable",
            declared_oxide=ox,
            n_gates=0,
            gate_area_um2=0.0,
            violating_area_um2=0.0,
            od2_area_um2=0.0,
            detail=(
                "no oxide declared (declared_oxide must be 'thick' or 'thin'); "
                "cannot judge the conditional obligation -- not a silent pass"
            ),
        )

    try:
        import klayout.db as db
    except Exception as exc:  # noqa: BLE001 -- capability absent, not a pass
        return OxideTypeVerdict(
            state="unsupported",
            declared_oxide=ox,
            n_gates=0,
            gate_area_um2=0.0,
            violating_area_um2=0.0,
            od2_area_um2=0.0,
            detail=f"klayout.db unavailable ({exc})",
        )

    diff_key, poly_key, od2_key = _resolve_keys(
        tech, diff_layer, poly_layer, od2_layer
    )

    layout = db.Layout()
    layout.read(gds_path)
    top = layout.top_cell()
    dbu = layout.dbu
    if top is None:
        return OxideTypeVerdict(
            state="unmeasurable",
            declared_oxide=ox,
            n_gates=0,
            gate_area_um2=0.0,
            violating_area_um2=0.0,
            od2_area_um2=0.0,
            detail="GDS has no top cell",
        )

    def region(key: Optional[LayerKey]) -> "db.Region":
        if key is None:
            return db.Region()
        idx = layout.find_layer(key[0], key[1])
        if idx is None:
            return db.Region()
        r = db.Region(top.begin_shapes_rec(idx))
        r.merge()
        return r

    diff = region(diff_key)
    poly = region(poly_key)
    od2 = region(od2_key)

    gate = poly & diff  # the transistor channels (poly over active)
    gate.merge()
    area2 = dbu * dbu

    if gate.is_empty():
        return OxideTypeVerdict(
            state="unmeasurable",
            declared_oxide=ox,
            n_gates=0,
            gate_area_um2=0.0,
            violating_area_um2=0.0,
            od2_area_um2=round(od2.area() * area2, 4),
            detail=(
                f"no transistor channel found (poly {poly_key} & active {diff_key} "
                "is empty); cannot observe the oxide obligation -- not a silent pass"
            ),
        )

    gate_area = round(gate.area() * area2, 4)
    od2_area = round(od2.area() * area2, 4)
    n_gates = gate.count()

    if ox == "thin":
        # THIN: no gate may be under OD2. A correct 1.8V design carries no OD2 -> the
        # intersection is empty -> PASS (NOT falsely RED'd).
        offending = gate & od2
        viol = round(offending.area() * area2, 4)
        satisfied = offending.is_empty()
        state = "pass" if satisfied else "fail"
        detail = (
            f"thin-ox design: {viol}um2 of gate is wrongly under OD2 {od2_key} "
            f"(must be 0)."
            if not satisfied
            else f"thin-ox design: 0 gate area under OD2 (correct; {n_gates} channels)."
        )
        return OxideTypeVerdict(
            state=state,
            declared_oxide=ox,
            n_gates=n_gates,
            gate_area_um2=gate_area,
            violating_area_um2=viol,
            od2_area_um2=od2_area,
            detail=detail,
        )

    # THICK: every channel OD must be OD2-enclosed by >= OD2.E.1 (0.32um). The
    # obligation breach is the channel-bearing OD area that OD2 fails to enclose by
    # that margin (inward-sized OD2). An OD2-blind design (0 OD2) fails wholesale.
    enc_nm = int(round(float(od2_enclose_od_um) / dbu))
    od_active = diff.interacting(gate)  # only the OD islands that carry a gate
    od_active.merge()
    od_active_area = round(od_active.area() * area2, 4)
    od2_inner = od2.sized(-enc_nm) if not od2.is_empty() else db.Region()
    under_enclosed = od_active - od2_inner
    viol = round(under_enclosed.area() * area2, 4)
    satisfied = under_enclosed.is_empty()
    state = "pass" if satisfied else "fail"
    if satisfied:
        detail = (
            f"thick-ox design: all {od_active_area}um2 of channel OD is OD2-enclosed "
            f"by >= {od2_enclose_od_um}um ({n_gates} channels; OD2 {od2_area}um2)."
        )
    else:
        detail = (
            f"thick-ox design: {viol}um2 of channel OD is NOT OD2-enclosed by "
            f"{od2_enclose_od_um}um (OD2 {od2_key} area {od2_area}um2; "
            f"channel OD {od_active_area}um2). A 3.3V device without OD2 extracts as "
            f"1.8V -- the false-green the OD2-aware deck closes."
        )
    return OxideTypeVerdict(
        state=state,
        declared_oxide=ox,
        n_gates=n_gates,
        gate_area_um2=gate_area,
        violating_area_um2=viol,
        od2_area_um2=od2_area,
        detail=detail,
    )


__all__ = ["OxideTypeVerdict", "check_oxide_type", "FIVE_STATES"]
